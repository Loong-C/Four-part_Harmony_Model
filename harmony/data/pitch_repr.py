from __future__ import annotations

import music21 as m21


def key_to_name(key_obj: m21.key.Key) -> str:
    mode = "major" if key_obj.mode == "major" else "minor"
    return f"{key_obj.tonic.name}:{mode}"


def parse_key_name(name: str) -> m21.key.Key:
    tonic, mode = name.split(":")
    return m21.key.Key(tonic, mode)


def midi_to_degree_rel(midi_abs: int, key_obj: m21.key.Key) -> int:
    tonic_pc = key_obj.tonic.pitchClass
    return (int(midi_abs) - tonic_pc) % 12


def pc_to_degree_rel(pc: int, key_obj: m21.key.Key) -> int:
    tonic_pc = key_obj.tonic.pitchClass
    return (int(pc) - tonic_pc) % 12


def midi_to_octave_bucket(midi_abs: int) -> int:
    p = m21.pitch.Pitch(int(midi_abs))
    return int(p.octave)


def degree_octave_to_midi(degree_rel: int, octave_bucket: int, key_obj: m21.key.Key) -> int:
    pitch_class = (key_obj.tonic.pitchClass + int(degree_rel)) % 12
    return (int(octave_bucket) + 1) * 12 + pitch_class

