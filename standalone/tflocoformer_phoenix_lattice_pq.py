# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL) & 8b-is (https://8b.is)
#
# SPDX-License-Identifier: Apache-2.0

"""Zero-Float p/q RMSGroupNorm, SwiGLU, Z_{2^32} CORDIC RoPE & Laser Phoenix Lattice
STFT Phase Locking for TF-Locoformer Speech Separation and Enhancement.
"""

from enum import IntEnum
import math
from typing import List, Optional, Sequence, Tuple

# Canonical Golden Ratio rational convergent p/q = F_19 / F_18 = 4181 / 2584
PHI_P: int = 4181
PHI_Q: int = 2584

Q16_ONE: int = 1 << 16
Q32_MOD: int = 1 << 32
LN2_Q16: int = 45426  # round(ln(2) * 65536)


class PhoenixInterferenceState(IntEnum):
    """5-state Laser Phoenix Interference Lattice on Z_{2^32} for STFT phase bins."""

    BOUND = 0        # 0 deg   (|dtheta| < 22.5 deg)   - Coherent single speaker
    RELATED = 1      # 45 deg  (22.5 <= |dtheta| < 67.5) - Harmonic formant coupling
    INDEPENDENT = 2  # 90 deg  (67.5 <= |dtheta| < 112.5) - Orthogonal / quadrature source
    CONTRASTING = 3  # 135 deg (112.5 <= |dtheta| < 157.5) - Anti-harmonic crosstalk
    CONFLICTING = 4  # 180 deg (|dtheta| >= 157.5 deg) - Destructive phase cancellation


PHOENIX_LATTICE_ANCHORS_Q32: Tuple[int, int, int, int, int] = (
    0x00000000,  # 0.0 deg
    0x20000000,  # 45.0 deg
    0x40000000,  # 90.0 deg
    0x60000000,  # 135.0 deg
    0x80000000,  # 180.0 deg
)

# 11-iteration CORDIC atan(2^-i) table on Z_{2^32} (where 2^32 == 2*pi)
_CORDIC_ATAN_Q32: Tuple[int, ...] = (
    0x20000000,  # atan(2^0)  = 45.000000 deg
    0x12E4051E,  # atan(2^-1) = 26.565051 deg
    0x09FB385B,  # atan(2^-2) = 14.036243 deg
    0x051111D4,  # atan(2^-3) =  7.125016 deg
    0x028B0D43,  # atan(2^-4) =  3.576334 deg
    0x0145D7E1,  # atan(2^-5) =  1.789911 deg
    0x00A2F61E,  # atan(2^-6) =  0.895174 deg
    0x00517C55,  # atan(2^-7) =  0.447614 deg
    0x0028BE53,  # atan(2^-8) =  0.223811 deg
    0x00145F2F,  # atan(2^-9) =  0.111906 deg
    0x000A2F98,  # atan(2^-10)=  0.055953 deg
)
_CORDIC_GAIN_INV_Q16: int = 39797  # round((1 / K_11) * 65536) ~= 0.607252935 * 65536


def isqrt_u128(n: int) -> int:
    """Exact integer floor(sqrt(n)) for arbitrary-precision non-negative integers."""
    if n <= 0:
        return 0
    x = 1 << ((n.bit_length() + 1) >> 1)
    while True:
        y = (x + n // x) >> 1
        if y >= x:
            return x
        x = y


def classify_phoenix_interference_q32(signed_delta_q32: int) -> PhoenixInterferenceState:
    """Classifies a signed modular STFT phase difference on Z_{2^32} into the 5-state lattice."""
    abs_d = abs(int(signed_delta_q32)) & 0xFFFFFFFF
    if abs_d > 0x80000000:
        abs_d = 0x100000000 - abs_d
    if abs_d < 0x10000000:
        return PhoenixInterferenceState.BOUND
    if abs_d < 0x30000000:
        return PhoenixInterferenceState.RELATED
    if abs_d < 0x50000000:
        return PhoenixInterferenceState.INDEPENDENT
    if abs_d < 0x70000000:
        return PhoenixInterferenceState.CONTRASTING
    return PhoenixInterferenceState.CONFLICTING


def cordic_sincos_q16(phase_q32: int) -> Tuple[int, int]:
    """Computes (cos_q16, sin_q16) from a 32-bit cyclic phase on Z_{2^32} via 11-step integer CORDIC."""
    p = phase_q32 & 0xFFFFFFFF
    quadrant = p >> 30
    z = p & 0x3FFFFFFF
    if quadrant == 1:
        z = 0x40000000 - z
    elif quadrant == 3:
        z = 0x40000000 - z

    x = _CORDIC_GAIN_INV_Q16
    y = 0
    for i, atan_val in enumerate(_CORDIC_ATAN_Q32):
        dx = y >> i
        dy = x >> i
        if z >= 0:
            x -= dx
            y += dy
            z -= atan_val
        else:
            x += dx
            y -= dy
            z += atan_val

    if quadrant == 1:
        return -x, y
    if quadrant == 2:
        return -x, -y
    if quadrant == 3:
        return x, -y
    return x, y


def atan2_cordic_q32(y_q16: int, x_q16: int) -> Tuple[int, int]:
    """Vectoring-mode 11-step integer CORDIC returning (phase_q32, mag_q16)."""
    if x_q16 == 0 and y_q16 == 0:
        return 0, 0

    ax = abs(int(x_q16))
    ay = abs(int(y_q16))
    vx = ax << 4
    vy = ay << 4
    angle = 0

    for i, atan_val in enumerate(_CORDIC_ATAN_Q32):
        dx = vy >> i
        dy = vx >> i
        if vy > 0:
            vx += dx
            vy -= dy
            angle += atan_val
        else:
            vx -= dx
            vy += dy
            angle -= atan_val

    angle = max(0, min(0x40000000, angle))
    if x_q16 >= 0 and y_q16 >= 0:
        phase_q32 = angle
    elif x_q16 < 0 and y_q16 >= 0:
        phase_q32 = 0x80000000 - angle
    elif x_q16 < 0 and y_q16 < 0:
        phase_q32 = 0x80000000 + angle
    else:
        phase_q32 = (0x100000000 - angle) & 0xFFFFFFFF

    mag_q16 = ((vx * _CORDIC_GAIN_INV_Q16) + (1 << 19)) >> 20
    return phase_q32 & 0xFFFFFFFF, mag_q16


def exp_neg_pade22_q16(r_q16: int) -> int:
    """Range-reduced Padé-[2,2] rational p/q approximation to exp(-r) for r >= 0 in Q16."""
    if r_q16 <= 0:
        return Q16_ONE
    if r_q16 >= 12 * Q16_ONE:
        return 0
    k = r_q16 // LN2_Q16
    u_q16 = r_q16 - k * LN2_Q16
    # Padé-[2,2]: (12 - 6*u + u^2) / (12 + 6*u + u^2)
    u2_q16 = (u_q16 * u_q16) >> 16
    num = 12 * Q16_ONE - 6 * u_q16 + u2_q16
    den = 12 * Q16_ONE + 6 * u_q16 + u2_q16
    frac_q16 = ((num << 16) + (den >> 1)) // den
    return frac_q16 >> k


def silu_pq_q16(x_q16: int) -> int:
    """Zero-float p/q SiLU(x) = x / (1 + exp(-x)) in Q16."""
    if x_q16 >= 0:
        e_neg = exp_neg_pade22_q16(x_q16)
        sig_q16 = ((Q16_ONE << 16) + ((Q16_ONE + e_neg) >> 1)) // (Q16_ONE + e_neg)
    else:
        e_pos = exp_neg_pade22_q16(-x_q16)
        sig_q16 = ((e_pos << 16) + ((Q16_ONE + e_pos) >> 1)) // (Q16_ONE + e_pos)
    return (x_q16 * sig_q16) >> 16


def swiglu_pq(value_vec: Sequence[float], gate_vec: Sequence[float]) -> List[float]:
    """Zero-float p/q SwiGLU(x, g) = x * SiLU(g) in Q16."""
    out: List[float] = []
    for v, g in zip(value_vec, gate_vec):
        v_q16 = int(round(float(v) * Q16_ONE))
        g_q16 = int(round(float(g) * Q16_ONE))
        silu_g_q16 = silu_pq_q16(g_q16)
        prod_q16 = (v_q16 * silu_g_q16) >> 16
        out.append(float(prod_q16) / float(Q16_ONE))
    return out


def rms_group_norm_pq(
    x: Sequence[float],
    num_groups: int,
    gamma: Optional[Sequence[float]] = None,
    beta: Optional[Sequence[float]] = None,
    eps: float = 1e-5,
) -> List[float]:
    """Zero-float p/q RMSGroupNorm using 128-bit integer square root isqrt_u128."""
    dim = len(x)
    if dim % num_groups != 0:
        raise ValueError(f"dim {dim} must be divisible by num_groups {num_groups}")
    d_group = dim // num_groups
    eps_q16 = max(1, int(round(eps * Q16_ONE)))

    x_q16 = [int(round(float(v) * Q16_ONE)) for v in x]
    g_q16 = (
        [int(round(float(v) * Q16_ONE)) for v in gamma]
        if gamma is not None
        else [Q16_ONE] * dim
    )
    b_q16 = (
        [int(round(float(v) * Q16_ONE)) for v in beta]
        if beta is not None
        else [0] * dim
    )

    out: List[float] = []
    for g in range(num_groups):
        start = g * d_group
        group_slice = x_q16[start : start + d_group]
        # Mean square in Q32
        sum_sq_q32 = sum(v * v for v in group_slice)
        mean_sq_q32 = sum_sq_q32 // d_group
        rms_q16 = isqrt_u128(mean_sq_q32) + eps_q16
        for idx_in_g, v_q16 in enumerate(group_slice):
            ch = start + idx_in_g
            norm_q16 = (v_q16 << 16) // rms_q16
            scaled_q16 = ((norm_q16 * g_q16[ch]) >> 16) + b_q16[ch]
            out.append(float(scaled_q16) / float(Q16_ONE))
    return out


def apply_rope_phoenix_pq(
    vec_seq: Sequence[Sequence[float]], base_theta: float = 10000.0
) -> List[List[float]]:
    """Applies Rotary Position Embedding (RoPE) on the Z_{2^32} Laser Phoenix phase ring
    using 11-iteration shift-add integer CORDIC rotations.
    """
    if not vec_seq:
        return []
    dim = len(vec_seq[0])
    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even")

    half = dim // 2
    # Precompute angular velocities on Z_{2^32}: omega_k = round((2^32 / (2*pi)) * base^(-2k/dim))
    two_pi = 2.0 * math.pi
    omega_q32: List[int] = []
    for k in range(half):
        freq = 1.0 / (base_theta ** (2.0 * k / dim))
        omega_q32.append(int(round((freq / two_pi) * Q32_MOD)) & 0xFFFFFFFF)

    out_seq: List[List[float]] = []
    for t_idx, row in enumerate(vec_seq):
        out_row = [0.0] * dim
        for k in range(half):
            phase_q32 = (t_idx * omega_q32[k]) & 0xFFFFFFFF
            cos_q16, sin_q16 = cordic_sincos_q16(phase_q32)
            x0_q16 = int(round(float(row[2 * k]) * Q16_ONE))
            x1_q16 = int(round(float(row[2 * k + 1]) * Q16_ONE))
            # rotary_embedding_torch convention: [x0*cos - x1*sin, x1*cos + x0*sin]
            r0_q16 = (x0_q16 * cos_q16 - x1_q16 * sin_q16) >> 16
            r1_q16 = (x1_q16 * cos_q16 + x0_q16 * sin_q16) >> 16
            out_row[2 * k] = float(r0_q16) / float(Q16_ONE)
            out_row[2 * k + 1] = float(r1_q16) / float(Q16_ONE)
        out_seq.append(out_row)
    return out_seq


def _eval_catmull_rom_scalar_pq(
    p0: int, p1: int, p2: int, p3: int, k: int, span: int
) -> int:
    if span <= 0:
        return p1
    s2 = span * span
    s3 = s2 * span
    k2 = k * k
    k3 = k2 * k

    a0 = 2 * p1
    a1 = -p0 + p2
    a2 = 2 * p0 - 5 * p1 + 4 * p2 - p3
    a3 = -p0 + 3 * p1 - 3 * p2 + p3

    num = a0 * s3 + a1 * k * s2 + a2 * k2 * span + a3 * k3
    den = 2 * s3
    if num >= 0:
        return (num + den // 2) // den
    return -((-num + den // 2) // den)


def lock_stft_phase_phoenix_pq(
    stft_frames: Sequence[Tuple[float, float]],
    valid_mask: Optional[Sequence[bool]] = None,
) -> Tuple[List[Tuple[float, float]], List[int], List[PhoenixInterferenceState]]:
    """Locks a sequence of complex STFT bin values [(real, imag), ...] onto the Z_{2^32}
    Laser Phoenix Interference Lattice with Marine phase-jitter salience gating and
    circular Z_{2^32} Bessel-Overhauser Catmull-Rom frame dropout restoration.
    """
    n = len(stft_frames)
    if n == 0:
        return [], [], []

    if valid_mask is None:
        valid = [
            not (math.isnan(r) or math.isnan(im)) for r, im in stft_frames
        ]
    else:
        valid = [bool(v) for v in valid_mask]

    phases_q32 = [0] * n
    mags_q16 = [0] * n
    valid_indices: List[int] = []

    for i, (r, im) in enumerate(stft_frames):
        if valid[i]:
            rq16 = int(round(float(r) * Q16_ONE))
            imq16 = int(round(float(im) * Q16_ONE))
            ph, mg = atan2_cordic_q32(imq16, rq16)
            phases_q32[i] = ph
            mags_q16[i] = mg
            valid_indices.append(i)

    if not valid_indices:
        return [(0.0, 0.0)] * n, [0] * n, [PhoenixInterferenceState.BOUND] * n

    # Unwrap valid phases on Z_{2^32} via signed 32-bit modular differences
    unwrapped_q32: List[int] = [phases_q32[valid_indices[0]]]
    for k in range(1, len(valid_indices)):
        curr_p = phases_q32[valid_indices[k]]
        prev_p = phases_q32[valid_indices[k - 1]]
        diff = (curr_p - prev_p) & 0xFFFFFFFF
        if diff >= 0x80000000:
            diff -= 0x100000000
        unwrapped_q32.append(unwrapped_q32[-1] + diff)

    # Restore dropped STFT frames via Bessel-Overhauser Catmull-Rom on unwrapped phase & magnitude
    if len(valid_indices) < n:
        for i in range(n):
            if valid[i]:
                continue
            if i < valid_indices[0]:
                phases_q32[i] = phases_q32[valid_indices[0]]
                mags_q16[i] = mags_q16[valid_indices[0]]
                continue
            if i > valid_indices[-1]:
                phases_q32[i] = phases_q32[valid_indices[-1]]
                mags_q16[i] = mags_q16[valid_indices[-1]]
                continue

            pos = 0
            while pos + 1 < len(valid_indices) and valid_indices[pos + 1] < i:
                pos += 1
            k1 = pos
            k2 = min(len(valid_indices) - 1, pos + 1)
            k0 = max(0, k1 - 1)
            k3 = min(len(valid_indices) - 1, k2 + 1)

            idx0, idx1, idx2, idx3 = (
                valid_indices[k0],
                valid_indices[k1],
                valid_indices[k2],
                valid_indices[k3],
            )
            span = idx2 - idx1
            step_k = i - idx1
            h0 = idx1 - idx0
            h1 = span
            h2 = idx3 - idx2

            for arr_unwrap, target_arr, is_mod32 in (
                (unwrapped_q32, phases_q32, True),
                ([mags_q16[idx] for idx in valid_indices], mags_q16, False),
            ):
                y0, p1, p2, y3 = (
                    arr_unwrap[k0],
                    arr_unwrap[k1],
                    arr_unwrap[k2],
                    arr_unwrap[k3],
                )
                if h0 > 0 and h1 > 0:
                    m1 = ((h1 * h1 * (p1 - y0)) + (h0 * h0 * (p2 - p1))) // (
                        h0 * (h0 + h1)
                    )
                else:
                    m1 = p2 - p1
                if h2 > 0 and h1 > 0:
                    m2 = (
                        ((h2 * h2 * (p2 - p1)) + (h1 * h1 * (y3 - p2))) * h1
                    ) // (h2 * h1 * (h1 + h2))
                else:
                    m2 = p2 - p1
                p0 = p2 - 2 * m1
                p3 = p1 + 2 * m2
                val = _eval_catmull_rom_scalar_pq(p0, p1, p2, p3, step_k, span)
                target_arr[i] = (val & 0xFFFFFFFF) if is_mod32 else max(0, val)

    # Apply Z_{2^32} Laser Phoenix PLL + Marine phase-jitter salience gate
    locked_frames: List[Tuple[float, float]] = []
    coherence_q8_list: List[int] = []
    states: List[PhoenixInterferenceState] = []

    locked_p = phases_q32[0]
    vel_q32 = 0
    prev_raw_p = phases_q32[0]
    prev_raw_vel: Optional[int] = None
    jitter_ring: List[int] = [0] * 8
    j_idx = 0
    j_cnt = 0

    for i in range(n):
        raw_p = phases_q32[i]
        if i == 0:
            cos_q16, sin_q16 = cordic_sincos_q16(locked_p)
            r_out = ((mags_q16[i] * cos_q16) >> 16) / float(Q16_ONE)
            im_out = ((mags_q16[i] * sin_q16) >> 16) / float(Q16_ONE)
            locked_frames.append((r_out, im_out))
            coherence_q8_list.append(256)
            states.append(PhoenixInterferenceState.BOUND)
            continue

        raw_vel = (raw_p - prev_raw_p) & 0xFFFFFFFF
        if raw_vel >= 0x80000000:
            raw_vel -= 0x100000000
        prev_raw_p = raw_p

        if prev_raw_vel is None:
            prev_raw_vel = raw_vel
            vel_q32 = raw_vel
            locked_p = raw_p
            cos_q16, sin_q16 = cordic_sincos_q16(locked_p)
            r_out = ((mags_q16[i] * cos_q16) >> 16) / float(Q16_ONE)
            im_out = ((mags_q16[i] * sin_q16) >> 16) / float(Q16_ONE)
            locked_frames.append((r_out, im_out))
            coherence_q8_list.append(256)
            states.append(classify_phoenix_interference_q32(raw_vel))
            continue

        accel = abs(raw_vel - prev_raw_vel)
        prev_raw_vel = raw_vel
        jitter_ring[j_idx] = accel
        j_idx = (j_idx + 1) & 7
        if j_cnt < 8:
            j_cnt += 1

        mean_jitter = sum(jitter_ring[:j_cnt]) // j_cnt
        # Reference jitter: 22.5 deg on Z_{2^32} = 0x10000000
        penalty = (mean_jitter * 256) // 0x10000000
        coh_q8 = max(0, 256 - penalty)
        coherence_q8_list.append(coh_q8)

        pred_p = (locked_p + vel_q32) & 0xFFFFFFFF
        p_err = (raw_p - pred_p) & 0xFFFFFFFF
        if p_err >= 0x80000000:
            p_err -= 0x100000000

        state = classify_phoenix_interference_q32(p_err)
        states.append(state)

        # Marine Salience Gate: < 90/256 halves PLL innovation gain to suppress musical noise
        alpha_num = (PHI_Q // 2) if (coh_q8 < 90) else PHI_Q
        locked_p = (pred_p + ((p_err * alpha_num) // PHI_P)) & 0xFFFFFFFF
        beta_num = ((PHI_P - PHI_Q) // 4) if (coh_q8 < 90) else (PHI_P - PHI_Q)
        vel_q32 = vel_q32 + (((raw_vel - vel_q32) * beta_num) // PHI_P)

        cos_q16, sin_q16 = cordic_sincos_q16(locked_p)
        r_out = ((mags_q16[i] * cos_q16) >> 16) / float(Q16_ONE)
        im_out = ((mags_q16[i] * sin_q16) >> 16) / float(Q16_ONE)
        locked_frames.append((r_out, im_out))

    return locked_frames, coherence_q8_list, states
