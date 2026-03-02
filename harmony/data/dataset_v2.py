from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .pitch_repr import pc_to_degree_rel, parse_key_name
from .vocab import SPECIAL_TOKENS

VOICE_ORDER_MODEL = ("B", "T", "A", "S")


def load_pieces_jsonl(path: str | Path) -> list[dict[str, Any]]:
    pieces: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pieces.append(json.loads(line))
    return pieces


def save_pieces_jsonl(path: str | Path, pieces: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for piece in pieces:
            f.write(json.dumps(piece, ensure_ascii=False) + "\n")


def note_state_to_token(note_state: dict[str, Any]) -> str:
    kind = str(note_state["kind"])
    if kind == "rest":
        return SPECIAL_TOKENS["REST"]
    if kind == "hold":
        return SPECIAL_TOKENS["HOLD"]
    if kind == "onset":
        deg = note_state["degree_rel"]
        octv = note_state["octave_bucket"]
        if deg is None or octv is None:
            return SPECIAL_TOKENS["UNK"]
        return f"ON_{deg}_{octv}"
    return SPECIAL_TOKENS["UNK"]


def token_to_note_state(token: str) -> tuple[str, int | None, int | None]:
    if token == SPECIAL_TOKENS["REST"]:
        return "rest", None, None
    if token == SPECIAL_TOKENS["HOLD"]:
        return "hold", None, None
    if token.startswith("ON_"):
        _, deg, octv = token.split("_")
        return "onset", int(deg), int(octv)
    return "rest", None, None


def encode_piece(
    piece: dict[str, Any],
    vocabs: dict[str, dict[str, int]],
) -> dict[str, torch.Tensor]:
    key_vocab = vocabs["key_vocab"]
    roman_vocab = vocabs["roman_vocab"]
    inv_vocab = vocabs["inversion_vocab"]
    dur_vocab = vocabs["dur_vocab"]
    voice_vocab = vocabs["voice_state_vocab"]
    unk_key = key_vocab[SPECIAL_TOKENS["UNK"]]
    unk_roman = roman_vocab[SPECIAL_TOKENS["UNK"]]
    unk_inv = inv_vocab[SPECIAL_TOKENS["UNK"]]
    unk_dur = dur_vocab[SPECIAL_TOKENS["UNK"]]
    unk_voice = voice_vocab[SPECIAL_TOKENS["UNK"]]
    bos_voice = voice_vocab[SPECIAL_TOKENS["BOS_STEP"]]

    events = piece["events"]
    key_name = str(piece["global_key"])
    key_id = key_vocab.get(key_name, unk_key)
    key_obj = parse_key_name(key_name)

    n = len(events)
    key_ids = torch.full((n,), key_id, dtype=torch.long)
    roman_ids = torch.zeros((n,), dtype=torch.long)
    inv_ids = torch.zeros((n,), dtype=torch.long)
    dur_ids = torch.zeros((n,), dtype=torch.long)
    target_voices = torch.zeros((n, 4), dtype=torch.long)
    prev_voices = torch.zeros((n, 4), dtype=torch.long)
    chord_tones_degree = torch.zeros((n, 12), dtype=torch.float32)

    prev_voices[0, :] = bos_voice
    for i, ev in enumerate(events):
        roman_ids[i] = roman_vocab.get(str(ev["roman"]), unk_roman)
        inv_ids[i] = inv_vocab.get(str(ev["inversion"]), unk_inv)
        dur_ids[i] = dur_vocab.get(str(ev["dur"]), unk_dur)
        for v_idx, voice in enumerate(VOICE_ORDER_MODEL):
            token = note_state_to_token(ev["voices"][voice])
            target_voices[i, v_idx] = voice_vocab.get(token, unk_voice)

        chord_degrees = {pc_to_degree_rel(pc, key_obj) for pc in ev.get("chord_tones_pc", [])}
        for deg in chord_degrees:
            chord_tones_degree[i, int(deg)] = 1.0

        if i + 1 < n:
            prev_voices[i + 1] = target_voices[i]

    return {
        "key": key_ids,
        "roman": roman_ids,
        "inversion": inv_ids,
        "duration": dur_ids,
        "prev_voices": prev_voices,
        "target_voices": target_voices,
        "chord_tones_degree": chord_tones_degree,
    }


class HarmonyV2Dataset(Dataset):
    def __init__(
        self,
        pieces: list[dict[str, Any]],
        vocabs: dict[str, dict[str, int]],
        context_length: int,
    ) -> None:
        self.context_length = context_length
        self.encoded_pieces = [encode_piece(p, vocabs) for p in pieces]
        self.index: list[tuple[int, int]] = []
        for piece_idx, enc in enumerate(self.encoded_pieces):
            length = int(enc["key"].shape[0])
            if length < context_length:
                continue
            for start in range(0, length - context_length + 1):
                self.index.append((piece_idx, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        piece_idx, start = self.index[idx]
        enc = self.encoded_pieces[piece_idx]
        end = start + self.context_length
        return {
            "key": enc["key"][start:end],
            "roman": enc["roman"][start:end],
            "inversion": enc["inversion"][start:end],
            "duration": enc["duration"][start:end],
            "prev_voices": enc["prev_voices"][start:end],
            "target_voices": enc["target_voices"][start:end],
            "chord_tones_degree": enc["chord_tones_degree"][start:end],
        }

