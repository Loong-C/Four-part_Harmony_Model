from __future__ import annotations

import torch

from harmony.models.conditional_satb import ConditionalSATBTransformer


def _mini_model() -> ConditionalSATBTransformer:
    return ConditionalSATBTransformer(
        vocab_sizes={"key": 8, "roman": 16, "inversion": 8, "duration": 8, "voice": 32},
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        max_len=32,
    )


def test_condition_alignment():
    torch.manual_seed(0)
    model = _mini_model()
    bsz, seq = 2, 4
    key = torch.zeros((bsz, seq), dtype=torch.long)
    roman = torch.zeros((bsz, seq), dtype=torch.long)
    roman[1, :] = 1
    inv = torch.zeros((bsz, seq), dtype=torch.long)
    dur = torch.zeros((bsz, seq), dtype=torch.long)
    prev = torch.zeros((bsz, seq, 4), dtype=torch.long)
    out = model(key, roman, inv, dur, prev)
    assert not torch.allclose(out["B"][0], out["B"][1])


def test_stepwise_ordering():
    torch.manual_seed(1)
    model = _mini_model()
    bsz, seq = 1, 3
    key = torch.zeros((bsz, seq), dtype=torch.long)
    roman = torch.zeros((bsz, seq), dtype=torch.long)
    inv = torch.zeros((bsz, seq), dtype=torch.long)
    dur = torch.zeros((bsz, seq), dtype=torch.long)
    prev = torch.zeros((bsz, seq, 4), dtype=torch.long)

    teacher_a = torch.zeros((bsz, seq, 4), dtype=torch.long)
    teacher_b = torch.zeros((bsz, seq, 4), dtype=torch.long)
    teacher_b[:, :, 0] = 5

    out_a = model(key, roman, inv, dur, prev, teacher_tokens=teacher_a)
    out_b = model(key, roman, inv, dur, prev, teacher_tokens=teacher_b)
    assert not torch.allclose(out_a["T"], out_b["T"])

