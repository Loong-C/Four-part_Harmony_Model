from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SPECIAL_TOKENS = {
    "PAD": "<PAD>",
    "UNK": "<UNK>",
    "BOS_STEP": "<BOS_STEP>",
    "REST": "<REST>",
    "HOLD": "<HOLD>",
}


def _ordered_vocab(values: set[str], base_tokens: list[str]) -> dict[str, int]:
    out = {tok: idx for idx, tok in enumerate(base_tokens)}
    for val in sorted(values):
        if val not in out:
            out[val] = len(out)
    return out


def build_vocabs(pieces: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    keys: set[str] = set()
    romans: set[str] = set()
    inversions: set[str] = set()
    durs: set[str] = set()
    degrees: set[str] = set()
    octaves: set[str] = set()
    voice_states: set[str] = set()

    for piece in pieces:
        keys.add(str(piece["global_key"]))
        for ev in piece["events"]:
            romans.add(str(ev["roman"]))
            inversions.add(str(ev["inversion"]))
            durs.add(str(ev["dur"]))
            for v in ("S", "A", "T", "B"):
                ns = ev["voices"][v]
                kind = ns["kind"]
                if kind == "onset":
                    deg = str(ns["degree_rel"])
                    octv = str(ns["octave_bucket"])
                    degrees.add(deg)
                    octaves.add(octv)
                    voice_states.add(f"ON_{deg}_{octv}")

    key_vocab = _ordered_vocab(keys, [SPECIAL_TOKENS["PAD"], SPECIAL_TOKENS["UNK"]])
    roman_vocab = _ordered_vocab(romans, [SPECIAL_TOKENS["PAD"], "NC", SPECIAL_TOKENS["UNK"]])
    inv_vocab = _ordered_vocab(inversions, [SPECIAL_TOKENS["PAD"], "none", SPECIAL_TOKENS["UNK"]])
    dur_vocab = _ordered_vocab(durs, [SPECIAL_TOKENS["PAD"], SPECIAL_TOKENS["UNK"]])
    deg_vocab = _ordered_vocab(degrees, [SPECIAL_TOKENS["PAD"], SPECIAL_TOKENS["UNK"]])
    oct_vocab = _ordered_vocab(octaves, [SPECIAL_TOKENS["PAD"], SPECIAL_TOKENS["UNK"]])
    voice_vocab = _ordered_vocab(
        voice_states,
        [
            SPECIAL_TOKENS["PAD"],
            SPECIAL_TOKENS["UNK"],
            SPECIAL_TOKENS["BOS_STEP"],
            SPECIAL_TOKENS["REST"],
            SPECIAL_TOKENS["HOLD"],
        ],
    )
    return {
        "key_vocab": key_vocab,
        "roman_vocab": roman_vocab,
        "inversion_vocab": inv_vocab,
        "dur_vocab": dur_vocab,
        "degree_vocab": deg_vocab,
        "octave_vocab": oct_vocab,
        "voice_state_vocab": voice_vocab,
    }


def save_vocabs(vocabs: dict[str, dict[str, int]], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, vocab in vocabs.items():
        (out / f"{name}.json").write_text(
            json.dumps(vocab, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (out / "special_tokens.json").write_text(
        json.dumps(SPECIAL_TOKENS, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_vocab(path: str | Path) -> dict[str, int]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

