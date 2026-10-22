# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL) & 8b-is (https://8b.is)
#
# SPDX-License-Identifier: Apache-2.0

import math
import random

from standalone.tflocoformer_phoenix_lattice_pq import (
    PHOENIX_LATTICE_ANCHORS_Q32,
    PhoenixInterferenceState,
    apply_rope_phoenix_pq,
    classify_phoenix_interference_q32,
    lock_stft_phase_phoenix_pq,
    rms_group_norm_pq,
    swiglu_pq,
)


def test_rms_group_norm_pq_accuracy():
    rng = random.Random(42)
    x = [rng.gauss(0.0, 1.5) for _ in range(32)]
    gamma = [rng.uniform(0.5, 1.5) for _ in range(32)]
    beta = [rng.uniform(-0.2, 0.2) for _ in range(32)]

    out_pq = rms_group_norm_pq(x, num_groups=4, gamma=gamma, beta=beta, eps=1e-5)

    # Float reference
    out_ref = []
    for g in range(4):
        s = x[g * 8 : (g + 1) * 8]
        rms = math.sqrt(sum(v * v for v in s) / 8.0) + 1e-5
        for idx, v in enumerate(s):
            ch = g * 8 + idx
            out_ref.append((v / rms) * gamma[ch] + beta[ch])

    for a, b in zip(out_pq, out_ref):
        assert abs(a - b) < 2e-4


def test_swiglu_pq_accuracy():
    val = [-2.5, -1.0, -0.25, 0.0, 0.5, 1.25, 3.0]
    gate = [-3.0, -1.2, -0.4, 0.0, 0.7, 1.8, 4.0]
    out_pq = swiglu_pq(val, gate)

    for v, g, pq in zip(val, gate, out_pq):
        silu_ref = g / (1.0 + math.exp(-g))
        ref = v * silu_ref
        assert abs(pq - ref) < 2e-4


def test_rope_phoenix_pq_norm_preservation():
    rng = random.Random(19)
    seq = [[rng.gauss(0.0, 1.0) for _ in range(16)] for _ in range(10)]
    rot = apply_rope_phoenix_pq(seq)
    for orig_row, rot_row in zip(seq, rot):
        n_orig = math.sqrt(sum(v * v for v in orig_row))
        n_rot = math.sqrt(sum(v * v for v in rot_row))
        assert abs(n_orig - n_rot) < 1e-3


def test_stft_phase_lock_and_dropout_recovery():
    states = [classify_phoenix_interference_q32(a) for a in PHOENIX_LATTICE_ANCHORS_Q32]
    assert states == [
        PhoenixInterferenceState.BOUND,
        PhoenixInterferenceState.RELATED,
        PhoenixInterferenceState.INDEPENDENT,
        PhoenixInterferenceState.CONTRASTING,
        PhoenixInterferenceState.CONFLICTING,
    ]

    # Create a complex STFT harmonic trajectory crossing the -pi / +pi phase boundary
    gt_frames = []
    for t in range(24):
        angle = 2.8 + 0.15 * t  # crosses pi at t=3
        gt_frames.append((math.cos(angle), math.sin(angle)))

    corrupted = list(gt_frames)
    # Drop 3 consecutive STFT frames right across the pi wrap-around boundary
    for t in (2, 3, 4):
        corrupted[t] = (float("nan"), float("nan"))

    locked, coh_q8, _ = lock_stft_phase_phoenix_pq(corrupted)
    assert len(locked) == 24
    for t in (2, 3, 4):
        r_rec, im_rec = locked[t]
        r_gt, im_gt = gt_frames[t]
        # Verify unit magnitude and phase alignment
        dot = r_rec * r_gt + im_rec * im_gt
        assert dot > 0.995


if __name__ == "__main__":
    test_rms_group_norm_pq_accuracy()
    print("[PASS] test_rms_group_norm_pq_accuracy")
    test_swiglu_pq_accuracy()
    print("[PASS] test_swiglu_pq_accuracy")
    test_rope_phoenix_pq_norm_preservation()
    print("[PASS] test_rope_phoenix_pq_norm_preservation")
    test_stft_phase_lock_and_dropout_recovery()
    print("[PASS] test_stft_phase_lock_and_dropout_recovery")
