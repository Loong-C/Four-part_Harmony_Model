from __future__ import annotations

from harmony.data.dataset_v2 import HarmonyV2Dataset, note_state_to_token, token_to_note_state
from harmony.data.extract_v2 import collect_bach_paths, extract_piece
from harmony.data.vocab import build_vocabs


def _mini_pieces():
    piece_a = {
        "piece_id": "a",
        "source_path": "a",
        "global_key": "C:major",
        "events": [
            {
                "time": 0.0,
                "dur": 1.0,
                "roman": "I",
                "inversion": "root",
                "chord_tones_pc": [0, 4, 7],
                "voices": {
                    "S": {"kind": "onset", "midi_abs": 72, "degree_rel": 0, "octave_bucket": 5},
                    "A": {"kind": "onset", "midi_abs": 67, "degree_rel": 7, "octave_bucket": 4},
                    "T": {"kind": "onset", "midi_abs": 64, "degree_rel": 4, "octave_bucket": 4},
                    "B": {"kind": "onset", "midi_abs": 48, "degree_rel": 0, "octave_bucket": 3},
                },
            },
            {
                "time": 1.0,
                "dur": 1.0,
                "roman": "V",
                "inversion": "root",
                "chord_tones_pc": [7, 11, 2],
                "voices": {
                    "S": {"kind": "hold", "midi_abs": None, "degree_rel": None, "octave_bucket": None},
                    "A": {"kind": "hold", "midi_abs": None, "degree_rel": None, "octave_bucket": None},
                    "T": {"kind": "onset", "midi_abs": 62, "degree_rel": 2, "octave_bucket": 4},
                    "B": {"kind": "onset", "midi_abs": 55, "degree_rel": 7, "octave_bucket": 3},
                },
            },
            {
                "time": 2.0,
                "dur": 1.0,
                "roman": "I",
                "inversion": "root",
                "chord_tones_pc": [0, 4, 7],
                "voices": {
                    "S": {"kind": "onset", "midi_abs": 72, "degree_rel": 0, "octave_bucket": 5},
                    "A": {"kind": "onset", "midi_abs": 67, "degree_rel": 7, "octave_bucket": 4},
                    "T": {"kind": "onset", "midi_abs": 64, "degree_rel": 4, "octave_bucket": 4},
                    "B": {"kind": "onset", "midi_abs": 48, "degree_rel": 0, "octave_bucket": 3},
                },
            },
        ],
    }
    piece_b = {
        **piece_a,
        "piece_id": "b",
        "source_path": "b",
    }
    return [piece_a, piece_b]


def test_no_cross_piece_windows():
    pieces = _mini_pieces()
    vocabs = build_vocabs(pieces)
    ds = HarmonyV2Dataset(pieces, vocabs, context_length=2)
    # each piece len=3 with context=2 -> 2 windows each; total 4
    assert len(ds) == 4


def test_first_event_no_hold():
    path = collect_bach_paths()[0]
    piece, _ = extract_piece(path, piece_id="probe")
    first = piece.events[0]
    assert all(first.voices[v].kind != "hold" for v in ("S", "A", "T", "B"))


def test_vocab_roundtrip():
    token = note_state_to_token({"kind": "onset", "midi_abs": 60, "degree_rel": 0, "octave_bucket": 4})
    kind, deg, octv = token_to_note_state(token)
    assert kind == "onset"
    assert deg == 0
    assert octv == 4

