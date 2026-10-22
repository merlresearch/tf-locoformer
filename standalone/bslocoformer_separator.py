# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: Apache-2.0

import math
from itertools import accumulate
from typing import Union

import torch
import torch.nn as nn
from packaging.version import parse as V
from rotary_embedding_torch import RotaryEmbedding

from .tflocoformer_separator import TFLocoformerBlock

is_torch_2_0_plus = V(torch.__version__) >= V("2.0.0")

# same configuration as BS-Roformer: https://arxiv.org/abs/2309.02612
# (frequency range): num_bins
BAND_SPLIT = {(0, 1000): 2, (1000, 2000): 4, (2000, 4000): 12, (4000, 8000): 24, (8000, 16000): 48}


class BSLocoformerSeparator(nn.Module):
    """BS-Locoformer introduced in [1].

    Reference:
    [1] Kohei Saijo, Janek Ebbers, François G Germain, Gordon Wichern, Jonathan Le Roux,
    “Task-Aware Unified Source Separation,” Proc. ICASSP,2025.

    Args:
        num_spk: int
            number of output sources/speakers.
        n_layers: int
            number of Locoformer blocks.
        emb_dim: int
            Size of hidden dimension in the encoding Conv2D.
        norm_type: str
            Normalization layer. Must be either "layernorm" or "rmsgroupnorm".
        num_groups: int
            Number of groups in RMSGroupNorm layer.
        tf_order: str
            Order of frequency and temporal modeling. Must be either "ft" or "tf".
        n_heads: int
            Number of heads in multi-head self-attention.
        flash_attention: bool
            Whether to use flash attention. Only compatible with half precision.
        ffn_type: str or list
            Feed-forward network (FFN)-type chosen from "conv1d" or "swiglu_conv1d".
            Giving the list (e.g., ["conv1d", "conv1d"]) makes the model Macaron-style.
        ffn_hidden_dim: int or list
            Number of hidden dimensions in FFN.
            Giving the list (e.g., [256, 256]) makes the model Macaron-style.
        conv1d_kernel: int
            Kernel size in Conv1d.
        conv1d_shift: int
            Shift size of Conv1d kernel.
        dropout: float
            Dropout probability.
        eps: float
            Small constant for normalization layer.
        masking: bool
            Whether to estimate masks or not.
            When set to `False`, the model directly estimates the real and imaginary parts of sources,
            instead of estimating complex masks.
        stereo: bool
            Whether inputs are stereo or not. If set to `False`, inputs must be monaural signals.
    """

    def __init__(
        self,
        num_spk: int = 2,
        n_layers: int = 6,
        # general setup
        emb_dim: int = 128,
        norm_type: str = "rmsgroupnorm",
        num_groups: int = 4,  # used only in RMSGroupNorm
        tf_order: str = "ft",
        # self-attention related
        n_heads: int = 4,
        flash_attention: bool = False,  # available when using mixed precision
        attention_dim: int = 128,
        pos_enc: str = "rope",
        # ffn related
        ffn_type: Union[str, list] = "swiglu_conv1d",
        ffn_hidden_dim: Union[int, list] = 384,
        conv1d_kernel: int = 4,
        conv1d_shift: int = 1,
        dropout: float = 0.0,
        # band-split related
        sample_rate: int = 44100,
        stft_size: int = 2048,
        # others
        eps: float = 1.0e-5,
        masking: bool = True,
        stereo: bool = False,
    ):
        super().__init__()
        assert is_torch_2_0_plus, "Support only pytorch >= 2.0.0"
        self.num_spk = num_spk
        self.n_layers = n_layers
        assert attention_dim % n_heads == 0, (attention_dim, n_heads)

        if pos_enc == "nope":
            pe_freq = None
            pe_time = None
        elif pos_enc == "rope":
            pe_freq = RotaryEmbedding(attention_dim // n_heads)
            pe_time = RotaryEmbedding(attention_dim // n_heads)
        else:
            raise ValueError(f"Unsupported positional encoding: {pos_enc}")

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                TFLocoformerBlock(
                    pe_freq,
                    pe_time,
                    # general setup
                    emb_dim=emb_dim,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    tf_order=tf_order,
                    # self-attention related
                    n_heads=n_heads,
                    flash_attention=flash_attention,
                    attention_dim=attention_dim,
                    # ffn related
                    ffn_type=ffn_type,
                    ffn_hidden_dim=ffn_hidden_dim,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                    dropout=dropout,
                    eps=eps,
                )
            )

        self.band_split_module = BandSplitModule(num_spk, emb_dim, stft_size, sample_rate, stereo=stereo)

        self.masking = masking
        self.stereo = stereo

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            input (torch.Tensor): batched single- or multi-channel audio tensor with
                    M audio channels in TF-domain [B, M, T, F]
            ilens (torch.Tensor): input lengths [B]
            additional (Dict or None): other data, currently unused in this model.

        Returns:
            enhanced (List[Union(torch.Tensor)]):
                    [(B, T), ...] list of len n_srcs
                    of mono audio tensors with T samples.
            ilens (torch.Tensor): (B,)
            additional (Dict or None): other data, currently unused in this model,
                    we return it also in output.
        """
        if input.ndim == 3:
            # in case the input does not have channel dimension
            assert not self.stereo
            input = input.unsqueeze(1)  # [B, M, T, F]
        batch = input.movedim(1, -1)  # [B, T, F, M]
        batch = torch.cat((batch.real, batch.imag), dim=-1)  # [B, T, F, 2*M]

        # normal spectrogram -> band-split tensor
        batch = self.band_split_module.band_split(batch)  # [B, -1, T, F]

        # separation
        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        # band-split tensor -> normal spectrogram
        batch = self.band_split_module.bandwise_decoding(batch)  # [B, 2, N, (M), T, F]
        batch = batch.to(torch.float32)

        # mapping or masking
        batch = torch.complex(batch[:, 0], batch[:, 1])  # [B, N, (M), T, F]
        if self.masking:
            if self.stereo:
                input = input.unsqueeze(1)
            batch = input * batch
        return batch


class BandSplitModule(nn.Module):
    def __init__(self, num_src: int, emb_dim: int, stft_size: int, sample_rate: int, stereo: bool = False):
        super().__init__()

        self.num_src = num_src
        num_freq_bins = stft_size // 2 + 1

        # calculate number of bins in each band
        self.bands = []
        freq_each_bin = sample_rate // 2 / num_freq_bins
        for freq_range, num_bins in BAND_SPLIT.items():
            start, end = freq_range
            num_band = math.ceil((end - start) / (num_bins * freq_each_bin))
            self.bands.extend([num_bins] * num_band)

        # higher frequencies are divided into two bands
        rest = num_freq_bins - sum(self.bands)
        if sample_rate == 48000:
            self.bands.extend([rest // 4, rest // 4, rest // 4, rest // 4 + rest % 4])
        else:
            self.bands.extend([math.floor(rest / 2), math.ceil(rest / 2)])
        assert sum(self.bands) == num_freq_bins, (sum(self.bands), num_freq_bins, self.bands)
        print(f"Band-split module has {len(self.bands)} bands", flush=True)

        self.stereo = stereo
        coef = 4 if self.stereo else 2
        self.band_split_module = nn.ModuleList([])
        for band in self.bands:
            self.band_split_module.append(
                nn.Sequential(
                    nn.GroupNorm(1, band * coef),
                    nn.Conv1d(band * coef, emb_dim, kernel_size=1),
                )
            )

        self.bandwise_decoding_module = nn.ModuleList([])
        for band in self.bands:
            self.bandwise_decoding_module.append(
                nn.Sequential(
                    nn.GroupNorm(1, emb_dim),
                    nn.Conv1d(emb_dim, emb_dim * 4, kernel_size=1),
                    nn.Tanh(),
                    nn.Conv1d(emb_dim * 4, emb_dim * 4, kernel_size=1),
                    nn.Conv1d(
                        emb_dim * 4,
                        band * num_src * coef * 2,  # *2 for GLU
                        kernel_size=1,
                    ),
                    nn.GLU(dim=1),
                )
            )

        self.band_idx = [0] + self.bands
        self.band_idx = list(accumulate(self.band_idx))

    def band_split(self, input):
        """Band split process

        input: torch.Tensor (n_batch, n_frame, n_freq, 2)
        """
        n_batch, n_frame = input.shape[:2]
        input = input.movedim(1, -1)

        output = []
        for b in range(len(self.bands)):
            sub_band = input[:, self.band_idx[b] : self.band_idx[b + 1]]
            output.append(self.band_split_module[b](sub_band.reshape(n_batch, -1, n_frame)))
        output = torch.stack(output, dim=-1)
        return output  # (n_batch, emb_dim, n_frame, n_bands)

    def bandwise_decoding(self, input):
        """Band-wise decoding process

        input: torch.Tensor (n_batch, emb_dim, n_frame, n_bands)
        """
        n_batch, n_frame = input.shape[0], input.shape[2]
        output = []
        for b in range(len(self.bands)):
            sub_band = self.bandwise_decoding_module[b](input[..., b])
            if self.stereo:
                sub_band = sub_band.reshape(n_batch, 2, self.num_src, 2, -1, n_frame)
            else:
                sub_band = sub_band.reshape(n_batch, 2, self.num_src, -1, n_frame)
            output.append(sub_band)
        return torch.cat(output, dim=-2).transpose(-1, -2)  # (n_batch, 2, num_src, n_chan, n_frame, n_freq)
