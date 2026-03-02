from __future__ import annotations

from harmony.eval.theory import evaluate_generated_batch


def _sample():
    return [
        {
            "rerank_detail": {"onset_count": 4, "chord_fit_count": 4},
            "S": {"kind": "onset", "degree_rel": 0, "midi_abs": 72},
            "A": {"kind": "onset", "degree_rel": 7, "midi_abs": 67},
            "T": {"kind": "onset", "degree_rel": 4, "midi_abs": 64},
            "B": {"kind": "onset", "degree_rel": 0, "midi_abs": 48},
        },
        {
            "rerank_detail": {"onset_count": 2, "chord_fit_count": 2},
            "S": {"kind": "hold", "degree_rel": None, "midi_abs": 72},
            "A": {"kind": "hold", "degree_rel": None, "midi_abs": 67},
            "T": {"kind": "onset", "degree_rel": 2, "midi_abs": 62},
            "B": {"kind": "onset", "degree_rel": 7, "midi_abs": 55},
        },
    ]


def test_metric_determinism():
    report_a = evaluate_generated_batch([_sample(), _sample()])
    report_b = evaluate_generated_batch([_sample(), _sample()])
    assert report_a == report_b


def test_report_schema():
    report = evaluate_generated_batch([_sample()])
    keys = {
        "range_violation_rate",
        "voice_crossing_rate",
        "parallel_rate",
        "tendency_resolution_rate",
        "chord_fit",
        "generated_samples",
    }
    assert keys.issubset(report.keys())

