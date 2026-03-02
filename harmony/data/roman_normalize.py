from __future__ import annotations

import re
from dataclasses import dataclass


_REPLACEMENTS = {
    "ø": "o",
    "º": "o",
    "°": "o",
    "酶": "o",
    "♭": "b",
    "♯": "#",
}

_INVERSION_SET = {"root", "6", "64", "7", "65", "43", "42", "2", "other", "none"}


def normalize_roman_figure(raw: str) -> str:
    text = (raw or "NC").strip()
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = text.replace(" ", "")
    text = text.replace("[", "").replace("]", "")
    return text or "NC"


def normalize_inversion(raw: str | None) -> str:
    if raw is None:
        return "none"
    text = normalize_roman_figure(str(raw))
    if text in _INVERSION_SET:
        return text
    if text in {"53"}:
        return "root"
    if text in {"632", "63"}:
        return "6"
    if text in {"654", "6543"}:
        return "65"
    if text in {"743", "74"}:
        return "7"
    if text in {"7542", "7642", "742"}:
        return "42"
    return text if text in _INVERSION_SET else "other"


@dataclass(frozen=True)
class ParsedRomanToken:
    roman: str
    inversion: str


def parse_progression_token(token: str) -> ParsedRomanToken:
    clean = normalize_roman_figure(token)
    # Common shorthand: I64 / IV6 / V43 / V7
    m = re.match(r"^([#bA-Za-z+o]+?)(\d+)?$", clean)
    if not m:
        return ParsedRomanToken(roman=clean, inversion="none")
    roman = m.group(1) or clean
    inv = m.group(2)
    if inv is None:
        return ParsedRomanToken(roman=roman, inversion="none")
    if inv == "53":
        inv = "root"
    return ParsedRomanToken(roman=roman, inversion=normalize_inversion(inv))

