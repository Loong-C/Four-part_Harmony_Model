from __future__ import annotations

import music21 as m21
import torch

from harmony.infer.generate_v2 import (
    GeneratorArtifacts,
    _score_candidate,
    format_human_readable_table,
    generate_harmony_v2,
    to_human_readable_rows,
)
from harmony.models.conditional_satb import ConditionalSATBTransformer


def _fake_artifacts() -> GeneratorArtifacts:
    vocabs = {
        "key_vocab": {"<PAD>": 0, "<UNK>": 1, "C:major": 2},
        "roman_vocab": {"<PAD>": 0, "NC": 1, "<UNK>": 2, "I": 3, "V": 4},
        "inversion_vocab": {"<PAD>": 0, "none": 1, "<UNK>": 2, "root": 3},
        "dur_vocab": {"<PAD>": 0, "<UNK>": 1, "1.0": 2},
        "voice_state_vocab": {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS_STEP>": 2,
            "<REST>": 3,
            "<HOLD>": 4,
            "ON_0_3": 5,
            "ON_4_4": 6,
            "ON_7_4": 7,
            "ON_0_5": 8,
        },
    }
    model = ConditionalSATBTransformer(
        vocab_sizes={"key": 3, "roman": 5, "inversion": 4, "duration": 3, "voice": 9},
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_len=32,
    )
    inv = {v: k for k, v in vocabs["voice_state_vocab"].items()}
    return GeneratorArtifacts(model=model, vocabs=vocabs, inv_voice_vocab=inv, device=torch.device("cpu"))


def test_controlled_progression_smoke():
    artifacts = _fake_artifacts()
    out = generate_harmony_v2(
        artifacts=artifacts,
        key_name="C:major",
        progression=["I", "V", "I"],
        durations=[1.0, 1.0, 1.0],
        num_candidates=2,
        temperature=1.0,
        top_k=0,
        seed=7,
    )
    assert len(out) == 3
    assert all(v in out[0] for v in ("S", "A", "T", "B"))


def test_soft_rerank_improves_theory():
    artifacts = _fake_artifacts()
    key = m21.key.Key("C", "major")
    prev_real = {"S": 72, "A": 67, "T": 64, "B": 48}
    good = {"B": 5, "T": 7, "A": 6, "S": 8}
    bad = {"B": 3, "T": 3, "A": 3, "S": 3}
    score_good, _, _ = _score_candidate(good, artifacts.inv_voice_vocab, {0, 4, 7}, key, prev_real)
    score_bad, _, _ = _score_candidate(bad, artifacts.inv_voice_vocab, {0, 4, 7}, key, prev_real)
    assert score_good > score_bad


def test_human_readable_format():
    artifacts = _fake_artifacts()
    out = generate_harmony_v2(
        artifacts=artifacts,
        key_name="C:major",
        progression=["I", "V"],
        durations=[1.0, 1.0],
        num_candidates=2,
        temperature=1.0,
        top_k=0,
        seed=11,
    )
    rows = to_human_readable_rows(out)
    table = format_human_readable_table(rows)
    assert len(rows) == 2
    assert "Soprano" in table
    assert "Chord" in table
