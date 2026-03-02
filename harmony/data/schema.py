from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NoteState:
    kind: str  # onset | hold | rest
    midi_abs: int | None
    degree_rel: int | None
    octave_bucket: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "midi_abs": self.midi_abs,
            "degree_rel": self.degree_rel,
            "octave_bucket": self.octave_bucket,
        }


@dataclass(frozen=True)
class Event:
    time: float
    dur: float
    roman: str
    inversion: str
    chord_tones_pc: list[int]
    voices: dict[str, NoteState]

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "dur": self.dur,
            "roman": self.roman,
            "inversion": self.inversion,
            "chord_tones_pc": self.chord_tones_pc,
            "voices": {k: v.to_dict() for k, v in self.voices.items()},
        }


@dataclass(frozen=True)
class Piece:
    piece_id: str
    source_path: str
    global_key: str
    events: list[Event]

    def to_dict(self) -> dict[str, Any]:
        return {
            "piece_id": self.piece_id,
            "source_path": self.source_path,
            "global_key": self.global_key,
            "events": [e.to_dict() for e in self.events],
        }

