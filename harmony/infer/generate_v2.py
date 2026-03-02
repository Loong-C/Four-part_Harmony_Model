from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import music21 as m21
import torch

from harmony.common.seed import set_seed
from harmony.data.dataset_v2 import VOICE_ORDER_MODEL, token_to_note_state
from harmony.data.pitch_repr import degree_octave_to_midi, parse_key_name, pc_to_degree_rel
from harmony.data.roman_normalize import normalize_inversion, normalize_roman_figure, parse_progression_token
from harmony.data.vocab import load_vocab
from harmony.models.conditional_satb import ConditionalSATBTransformer

VOICE_RANGES = {
    "S": (60, 81),
    "A": (53, 74),
    "T": (48, 69),
    "B": (41, 64),
}
MODEL_TO_STD = {"B": "B", "T": "T", "A": "A", "S": "S"}


@dataclass
class GeneratorArtifacts:
    model: ConditionalSATBTransformer
    vocabs: dict[str, dict[str, int]]
    inv_voice_vocab: dict[int, str]
    device: torch.device


def _device_from_name(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_generator_artifacts(config: dict[str, Any], ckpt_path: str | None = None) -> GeneratorArtifacts:
    data_dir = Path(config.get("data_dir", "data/v2"))
    vocabs = {
        "key_vocab": load_vocab(data_dir / "key_vocab.json"),
        "roman_vocab": load_vocab(data_dir / "roman_vocab.json"),
        "inversion_vocab": load_vocab(data_dir / "inversion_vocab.json"),
        "dur_vocab": load_vocab(data_dir / "dur_vocab.json"),
        "voice_state_vocab": load_vocab(data_dir / "voice_state_vocab.json"),
    }
    inv_voice_vocab = {v: k for k, v in vocabs["voice_state_vocab"].items()}

    ckpt = torch.load(ckpt_path or config["checkpoint"], map_location="cpu")
    ckpt_cfg = ckpt.get("config", {})
    model_cfg = ckpt_cfg.get("model", config.get("model", {}))
    model = ConditionalSATBTransformer(
        vocab_sizes={
            "key": len(vocabs["key_vocab"]),
            "roman": len(vocabs["roman_vocab"]),
            "inversion": len(vocabs["inversion_vocab"]),
            "duration": len(vocabs["dur_vocab"]),
            "voice": len(vocabs["voice_state_vocab"]),
        },
        d_model=int(model_cfg.get("d_model", 384)),
        nhead=int(model_cfg.get("nhead", 8)),
        num_layers=int(model_cfg.get("num_layers", 6)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 1024)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_len=int(model_cfg.get("max_len", 1024)),
    )
    model.load_state_dict(ckpt["model"])
    device = _device_from_name(str(config.get("device", "auto")))
    model.to(device).eval()
    return GeneratorArtifacts(model=model, vocabs=vocabs, inv_voice_vocab=inv_voice_vocab, device=device)


def _sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    scaled = logits / max(temperature, 1e-4)
    if top_k > 0 and top_k < scaled.shape[-1]:
        values, indices = torch.topk(scaled, k=top_k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        pick = torch.multinomial(probs, 1).item()
        return int(indices[pick].item())
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def _interval_class(a: int, b: int) -> int:
    return abs(a - b) % 12


def _direction(prev: int, curr: int) -> int:
    if curr > prev:
        return 1
    if curr < prev:
        return -1
    return 0


def _build_step_condition(
    key_name: str,
    prog_token: str,
    dur: float,
    vocabs: dict[str, dict[str, int]],
) -> dict[str, Any]:
    key_obj = parse_key_name(key_name)
    try:
        rn = m21.roman.RomanNumeral(prog_token, key_obj)
        roman = normalize_roman_figure(rn.romanNumeralAlone or rn.figure)
        inversion = normalize_inversion(rn.inversionName())
        pcs = sorted({int(p.pitchClass) for p in rn.pitches})
    except Exception:
        parsed = parse_progression_token(prog_token)
        roman = parsed.roman
        inversion = parsed.inversion
        pcs = []

    chord_deg = {pc_to_degree_rel(pc, key_obj) for pc in pcs}
    chord_mask = torch.zeros(12, dtype=torch.float32)
    for deg in chord_deg:
        chord_mask[deg] = 1.0

    key_vocab = vocabs["key_vocab"]
    roman_vocab = vocabs["roman_vocab"]
    inv_vocab = vocabs["inversion_vocab"]
    dur_vocab = vocabs["dur_vocab"]
    return {
        "roman_raw": prog_token,
        "roman": roman,
        "inversion": inversion,
        "dur": dur,
        "key_id": key_vocab.get(key_name, key_vocab["<UNK>"]),
        "roman_id": roman_vocab.get(roman, roman_vocab["<UNK>"]),
        "inversion_id": inv_vocab.get(inversion, inv_vocab["<UNK>"]),
        "dur_id": dur_vocab.get(str(dur), dur_vocab["<UNK>"]),
        "chord_deg_set": chord_deg,
        "chord_deg_mask": chord_mask,
    }


def _decode_token_to_midi(
    token_id: int,
    inv_voice_vocab: dict[int, str],
    key_obj: m21.key.Key,
    prev_midi: int | None,
) -> tuple[str, int | None]:
    token = inv_voice_vocab[token_id]
    kind, deg, octv = token_to_note_state(token)
    if kind == "rest":
        return token, None
    if kind == "hold":
        return token, prev_midi
    if deg is None or octv is None:
        return token, None
    return token, degree_octave_to_midi(deg, octv, key_obj)


def _score_candidate(
    candidate: dict[str, int],
    inv_voice_vocab: dict[int, str],
    chord_deg_set: set[int],
    key_obj: m21.key.Key,
    prev_real: dict[str, int | None],
) -> tuple[float, dict[str, int | float], dict[str, int | None]]:
    current_real: dict[str, int | None] = {}
    onset_count = 0
    chord_fit_count = 0
    leap_count = 0
    range_viol = 0

    for voice in VOICE_ORDER_MODEL:
        tok, midi_val = _decode_token_to_midi(candidate[voice], inv_voice_vocab, key_obj, prev_real[voice])
        current_real[voice] = midi_val
        kind, deg, _ = token_to_note_state(tok)
        if kind == "onset":
            onset_count += 1
            if deg in chord_deg_set:
                chord_fit_count += 1
            if prev_real[voice] is not None and midi_val is not None and abs(midi_val - prev_real[voice]) > 7:
                leap_count += 1
        if midi_val is not None:
            low, high = VOICE_RANGES[MODEL_TO_STD[voice]]
            if not (low <= midi_val <= high):
                range_viol += 1

    crossing = 0
    s, a, t, b = current_real["S"], current_real["A"], current_real["T"], current_real["B"]
    if s is not None and a is not None and s < a:
        crossing += 1
    if a is not None and t is not None and a < t:
        crossing += 1
    if t is not None and b is not None and t < b:
        crossing += 1

    parallel = 0
    check_pairs = [("S", "B"), ("S", "A"), ("A", "T"), ("T", "B")]
    for v1, v2 in check_pairs:
        p1, p2 = prev_real[v1], prev_real[v2]
        c1, c2 = current_real[v1], current_real[v2]
        if None in (p1, p2, c1, c2):
            continue
        prev_int = _interval_class(p1, p2)
        curr_int = _interval_class(c1, c2)
        if prev_int in {0, 7} and curr_int in {0, 7}:
            d1 = _direction(p1, c1)
            d2 = _direction(p2, c2)
            if d1 != 0 and d1 == d2:
                parallel += 1

    chord_fit_ratio = chord_fit_count / max(1, onset_count)
    score = (
        2.0 * chord_fit_ratio
        - 1.8 * range_viol
        - 1.4 * crossing
        - 1.4 * parallel
        - 0.4 * leap_count
    )
    detail = {
        "onset_count": onset_count,
        "chord_fit_count": chord_fit_count,
        "range_violation": range_viol,
        "crossing": crossing,
        "parallel": parallel,
        "large_leap": leap_count,
    }
    return score, detail, current_real


def generate_harmony_v2(
    artifacts: GeneratorArtifacts,
    key_name: str,
    progression: list[str],
    durations: list[float],
    num_candidates: int = 8,
    temperature: float = 0.9,
    top_k: int = 24,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if len(progression) != len(durations):
        raise ValueError("progression and durations length mismatch")

    set_seed(seed)
    key_obj = parse_key_name(key_name)
    vocabs = artifacts.vocabs
    voice_vocab = vocabs["voice_state_vocab"]
    bos_id = voice_vocab["<BOS_STEP>"]

    conditions = [
        _build_step_condition(key_name, prog_token, dur, vocabs) for prog_token, dur in zip(progression, durations)
    ]

    key_seq: list[int] = []
    roman_seq: list[int] = []
    inv_seq: list[int] = []
    dur_seq: list[int] = []
    prev_voice_seq: list[list[int]] = []
    generated: list[dict[str, Any]] = []
    prev_tokens = {v: bos_id for v in VOICE_ORDER_MODEL}
    prev_real = {v: None for v in VOICE_ORDER_MODEL}

    for step_idx, cond in enumerate(conditions):
        key_seq.append(cond["key_id"])
        roman_seq.append(cond["roman_id"])
        inv_seq.append(cond["inversion_id"])
        dur_seq.append(cond["dur_id"])
        prev_voice_seq.append([prev_tokens["B"], prev_tokens["T"], prev_tokens["A"], prev_tokens["S"]])

        key_t = torch.tensor([key_seq], dtype=torch.long, device=artifacts.device)
        roman_t = torch.tensor([roman_seq], dtype=torch.long, device=artifacts.device)
        inv_t = torch.tensor([inv_seq], dtype=torch.long, device=artifacts.device)
        dur_t = torch.tensor([dur_seq], dtype=torch.long, device=artifacts.device)
        prev_voice_t = torch.tensor([prev_voice_seq], dtype=torch.long, device=artifacts.device)

        with torch.no_grad():
            hidden = artifacts.model.encode_context(key_t, roman_t, inv_t, dur_t, prev_voice_t)[:, -1, :]
            b_logits = artifacts.model.head_b(hidden).squeeze(0)

            best_score = float("-inf")
            best_choice: dict[str, int] | None = None
            best_detail: dict[str, int | float] | None = None
            best_real: dict[str, int | None] | None = None

            for _ in range(num_candidates):
                b_id = _sample_from_logits(b_logits, temperature=temperature, top_k=top_k)
                b_emb = artifacts.model.voice_emb(torch.tensor([b_id], device=artifacts.device))
                t_logits = artifacts.model.head_t(hidden + b_emb).squeeze(0)
                t_id = _sample_from_logits(t_logits, temperature=temperature, top_k=top_k)
                t_emb = artifacts.model.voice_emb(torch.tensor([t_id], device=artifacts.device))
                a_logits = artifacts.model.head_a(hidden + b_emb + t_emb).squeeze(0)
                a_id = _sample_from_logits(a_logits, temperature=temperature, top_k=top_k)
                a_emb = artifacts.model.voice_emb(torch.tensor([a_id], device=artifacts.device))
                s_logits = artifacts.model.head_s(hidden + b_emb + t_emb + a_emb).squeeze(0)
                s_id = _sample_from_logits(s_logits, temperature=temperature, top_k=top_k)

                candidate = {"B": b_id, "T": t_id, "A": a_id, "S": s_id}
                score, detail, current_real = _score_candidate(
                    candidate=candidate,
                    inv_voice_vocab=artifacts.inv_voice_vocab,
                    chord_deg_set=cond["chord_deg_set"],
                    key_obj=key_obj,
                    prev_real=prev_real,
                )
                if score > best_score:
                    best_score = score
                    best_choice = candidate
                    best_detail = detail
                    best_real = current_real

            assert best_choice is not None
            assert best_detail is not None
            assert best_real is not None
            prev_tokens = best_choice
            prev_real = best_real

            row: dict[str, Any] = {
                "step": step_idx + 1,
                "key": key_name,
                "roman": cond["roman"],
                "inversion": cond["inversion"],
                "dur": cond["dur"],
                "score": best_score,
                "rerank_detail": best_detail,
            }
            for voice in ("S", "A", "T", "B"):
                token = artifacts.inv_voice_vocab[best_choice[voice]]
                kind, deg, octv = token_to_note_state(token)
                midi_abs = best_real[voice]
                row[voice] = {
                    "token": token,
                    "kind": kind,
                    "degree_rel": deg,
                    "octave_bucket": octv,
                    "midi_abs": midi_abs,
                }
            generated.append(row)

    return generated


def save_generated(path: str | Path, payload: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _midi_to_note_name(midi_abs: int) -> str:
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = midi_abs // 12 - 1
    return f"{notes[midi_abs % 12]}{octave}"


def _voice_to_human_note(voice_payload: dict[str, Any]) -> str:
    kind = str(voice_payload.get("kind"))
    if kind == "rest":
        return "<REST>"
    if kind == "hold":
        return "<HOLD>"
    midi_abs = voice_payload.get("midi_abs")
    if midi_abs is None:
        return "<UNK>"
    return _midi_to_note_name(int(midi_abs))


def to_human_readable_rows(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in payload:
        inversion = str(step.get("inversion", "none"))
        chord = str(step.get("roman", "NC")) if inversion in {"none", "root"} else f"{step.get('roman', 'NC')}{inversion}"
        rows.append(
            {
                "step": int(step.get("step", 0)),
                "chord": chord,
                "dur": float(step.get("dur", 1.0)),
                "Soprano": _voice_to_human_note(step["S"]),
                "Alto": _voice_to_human_note(step["A"]),
                "Tenor": _voice_to_human_note(step["T"]),
                "Bass": _voice_to_human_note(step["B"]),
            }
        )
    return rows


def format_human_readable_table(rows: list[dict[str, Any]]) -> str:
    header = (
        f"{'Step':<6} | {'Chord':<8} | {'Dur':<5} | "
        f"{'Soprano':<8} | {'Alto':<8} | {'Tenor':<8} | {'Bass':<8}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['step']:<6} | {row['chord']:<8} | {row['dur']:<5} | "
            f"{row['Soprano']:<8} | {row['Alto']:<8} | {row['Tenor']:<8} | {row['Bass']:<8}"
        )
    return "\n".join(lines)


def save_human_readable(
    json_path: str | Path,
    txt_path: str | Path,
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    json_out = Path(json_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    txt_out = Path(txt_path)
    txt_out.parent.mkdir(parents=True, exist_ok=True)
    txt_out.write_text(format_human_readable_table(rows) + "\n", encoding="utf-8")
    return json_out, txt_out
