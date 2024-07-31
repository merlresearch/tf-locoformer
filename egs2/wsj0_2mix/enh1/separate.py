# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: Apache-2.0


import argparse
from pathlib import Path

import soundfile as sf
from espnet2.bin.enh_inference import SeparateSpeech

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path", type=Path, required=True, help="Path to pre-trained model parameters (.pth file)."
    )
    parser.add_argument("--audio_path", type=Path, required=True, help="Path to the audio file to separate.")
    parser.add_argument(
        "--audio_output_dir", type=Path, default="./audio_outputs", help="Directory to save the separated audios."
    )
    args = parser.parse_args()

    config_path = args.model_path.parent / "config.yaml"

    separation_model = SeparateSpeech(
        train_config=config_path,
        model_file=args.model_path,
        normalize_output_wav=True,
        device="cuda:0",
    )

    mix, sample_rate = sf.read(args.audio_path, dtype="float32")

    # Normalize the input
    mix /= mix.std(axis=-1)

    # Shape of input mixture must be (1, n_samples)
    speeches = separation_model(mix[None], sample_rate)  # list of numpy arrays

    # Save the separated audios
    args.audio_output_dir.mkdir(exist_ok=True, parents=True)
    for i, speech in enumerate(speeches):
        filename = f"{args.audio_path.stem}_{i+1}.wav"
        sf.write(args.audio_output_dir / filename, speech[0], sample_rate)
