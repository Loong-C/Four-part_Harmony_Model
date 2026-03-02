from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import music21 as m21

from .pitch_repr import key_to_name, midi_to_degree_rel, midi_to_octave_bucket
from .roman_normalize import normalize_inversion, normalize_roman_figure
from .schema import Event, NoteState, Piece

VOICE_ORDER = ("S", "A", "T", "B")


def _normalize_paths(paths: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in paths:
        p = str(item)
        if not p.lower().endswith((".mxl", ".xml")):
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def collect_bach_paths() -> list[str]:
    try:
        return _normalize_paths(m21.corpus.getComposer("bach"))
    except Exception:
        candidates = m21.corpus.getPaths(fileExtensions=("mxl", "xml"), name=("core", "local"))
        return _normalize_paths([p for p in candidates if "bach" in str(p).lower()])


def _infer_inversion(rn_obj: m21.roman.RomanNumeral) -> str:
    try:
        return normalize_inversion(rn_obj.inversionName())
    except Exception:
        return "none"


def _chord_state(
    start_time: float, chords: m21.stream.Stream, key_obj: m21.key.Key
) -> tuple[str, str, list[int], str]:
    for chord in chords:
        c_start = round(float(chord.offset), 6)
        c_end = round(float(chord.offset + chord.quarterLength), 6)
        if c_start <= start_time < c_end:
            try:
                rn_obj = m21.roman.romanNumeralFromChord(chord, key_obj)
                roman_raw = rn_obj.romanNumeralAlone or rn_obj.figure
                roman = normalize_roman_figure(roman_raw)
                inversion = _infer_inversion(rn_obj)
                pcs = sorted({int(p.pitchClass) for p in rn_obj.pitches})
                return roman, inversion, pcs, normalize_roman_figure(str(rn_obj.figure))
            except Exception:
                break
    return "NC", "none", [], "NC"


def _voice_state_at(
    flat_stream: m21.stream.Stream, start_time: float, key_obj: m21.key.Key
) -> NoteState:
    for el in flat_stream:
        el_start = round(float(el.offset), 6)
        el_end = round(float(el.offset + el.quarterLength), 6)
        if not (el_start <= start_time < el_end):
            continue

        if el.isRest:
            return NoteState(kind="rest", midi_abs=None, degree_rel=None, octave_bucket=None)

        if getattr(el, "isChord", False):
            note_obj = el.sortAscending().notes[-1]
            midi_abs = int(note_obj.pitch.midi)
            if start_time == el_start:
                return NoteState(
                    kind="onset",
                    midi_abs=midi_abs,
                    degree_rel=midi_to_degree_rel(midi_abs, key_obj),
                    octave_bucket=midi_to_octave_bucket(midi_abs),
                )
            return NoteState(kind="hold", midi_abs=None, degree_rel=None, octave_bucket=None)

        if getattr(el, "isNote", False):
            midi_abs = int(el.pitch.midi)
            if start_time == el_start:
                return NoteState(
                    kind="onset",
                    midi_abs=midi_abs,
                    degree_rel=midi_to_degree_rel(midi_abs, key_obj),
                    octave_bucket=midi_to_octave_bucket(midi_abs),
                )
            return NoteState(kind="hold", midi_abs=None, degree_rel=None, octave_bucket=None)

    return NoteState(kind="rest", midi_abs=None, degree_rel=None, octave_bucket=None)


def extract_piece(path: str, piece_id: str) -> tuple[Piece, dict[str, str]]:
    score = m21.converter.parse(path)
    key_obj = score.analyze("key")
    global_key = key_to_name(key_obj)

    if len(score.parts) < 4:
        raise ValueError(f"piece {piece_id} has less than 4 parts")

    voice_parts = {
        "S": score.parts[0].flatten().notesAndRests.stream(),
        "A": score.parts[1].flatten().notesAndRests.stream(),
        "T": score.parts[2].flatten().notesAndRests.stream(),
        "B": score.parts[3].flatten().notesAndRests.stream(),
    }
    chordified = score.chordify().flatten().getElementsByClass("Chord").stream()

    all_offsets: set[float] = set()
    for p in voice_parts.values():
        for el in p:
            all_offsets.add(round(float(el.offset), 6))
            all_offsets.add(round(float(el.offset + el.quarterLength), 6))
    sorted_offsets = sorted(all_offsets)
    if len(sorted_offsets) < 2:
        raise ValueError(f"piece {piece_id} has no valid offsets")

    events: list[Event] = []
    roman_map: dict[str, str] = {}
    for idx in range(len(sorted_offsets) - 1):
        start = sorted_offsets[idx]
        end = sorted_offsets[idx + 1]
        dur = round(end - start, 6)
        if dur <= 0:
            continue
        roman, inversion, chord_tones_pc, raw_figure = _chord_state(start, chordified, key_obj)
        roman_map[raw_figure] = roman
        voices = {name: _voice_state_at(voice_parts[name], start, key_obj) for name in VOICE_ORDER}
        events.append(
            Event(
                time=start,
                dur=dur,
                roman=roman,
                inversion=inversion,
                chord_tones_pc=chord_tones_pc,
                voices=voices,
            )
        )
    if not events:
        raise ValueError(f"piece {piece_id} has no events")

    # Enforce first event no-hold as a hard data invariant.
    first = events[0]
    repaired_voices = dict(first.voices)
    repaired = False
    for name in VOICE_ORDER:
        if repaired_voices[name].kind == "hold":
            repaired_voices[name] = NoteState(kind="rest", midi_abs=None, degree_rel=None, octave_bucket=None)
            repaired = True
    if repaired:
        events[0] = Event(
            time=first.time,
            dur=first.dur,
            roman=first.roman,
            inversion=first.inversion,
            chord_tones_pc=first.chord_tones_pc,
            voices=repaired_voices,
        )

    piece = Piece(
        piece_id=piece_id,
        source_path=os.fspath(Path(path)),
        global_key=global_key,
        events=events,
    )
    return piece, roman_map

