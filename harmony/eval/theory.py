from __future__ import annotations

from typing import Any


VOICE_RANGES = {
    "S": (60, 81),
    "A": (53, 74),
    "T": (48, 69),
    "B": (41, 64),
}


def _interval_class(a: int, b: int) -> int:
    return abs(a - b) % 12


def _direction(prev: int, curr: int) -> int:
    if curr > prev:
        return 1
    if curr < prev:
        return -1
    return 0


def evaluate_generated_batch(samples: list[list[dict[str, Any]]]) -> dict[str, float]:
    total_active_notes = 0
    range_viol = 0
    total_steps = 0
    crossing = 0
    parallel_total = 0
    pair_steps = 0
    tendency_total = 0
    tendency_ok = 0
    chord_onset_total = 0
    chord_onset_fit = 0

    for sample in samples:
        prev_real = {"S": None, "A": None, "T": None, "B": None}
        prev_deg = {"S": None, "A": None, "T": None, "B": None}
        for step in sample:
            total_steps += 1
            rerank = step.get("rerank_detail", {})
            chord_onset_total += int(rerank.get("onset_count", 0))
            chord_onset_fit += int(rerank.get("chord_fit_count", 0))
            curr_real = {}
            curr_deg = {}
            for v in ("S", "A", "T", "B"):
                midi = step[v]["midi_abs"]
                kind = step[v]["kind"]
                deg = step[v]["degree_rel"]
                curr_real[v] = midi
                curr_deg[v] = deg
                if midi is not None:
                    total_active_notes += 1
                    low, high = VOICE_RANGES[v]
                    if not (low <= midi <= high):
                        range_viol += 1
            s, a, t, b = curr_real["S"], curr_real["A"], curr_real["T"], curr_real["B"]
            if s is not None and a is not None and s < a:
                crossing += 1
            if a is not None and t is not None and a < t:
                crossing += 1
            if t is not None and b is not None and t < b:
                crossing += 1

            for pair in (("S", "B"), ("S", "A"), ("A", "T"), ("T", "B")):
                p1, p2 = prev_real[pair[0]], prev_real[pair[1]]
                c1, c2 = curr_real[pair[0]], curr_real[pair[1]]
                if None in (p1, p2, c1, c2):
                    continue
                pair_steps += 1
                prev_int = _interval_class(p1, p2)
                curr_int = _interval_class(c1, c2)
                if prev_int in {0, 7} and curr_int in {0, 7}:
                    d1 = _direction(p1, c1)
                    d2 = _direction(p2, c2)
                    if d1 != 0 and d1 == d2:
                        parallel_total += 1

            for v in ("S", "A", "T", "B"):
                if prev_deg[v] == 11 and curr_deg[v] is not None:
                    tendency_total += 1
                    if curr_deg[v] in {0, 1}:
                        tendency_ok += 1
            prev_real = curr_real
            prev_deg = curr_deg

    return {
        "range_violation_rate": range_viol / max(1, total_active_notes),
        "voice_crossing_rate": crossing / max(1, total_steps),
        "parallel_rate": parallel_total / max(1, pair_steps),
        "tendency_resolution_rate": tendency_ok / max(1, tendency_total),
        "chord_fit": chord_onset_fit / max(1, chord_onset_total),
        "generated_samples": len(samples),
    }
