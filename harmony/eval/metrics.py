from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from harmony.data.dataset_v2 import HarmonyV2Dataset, load_pieces_jsonl
from harmony.data.vocab import load_vocab
from harmony.train.losses import build_voice_metadata, compute_total_loss


def evaluate_validation_metrics(
    model: torch.nn.Module,
    data_dir: str | Path,
    context_length: int,
    batch_size: int,
    device: torch.device,
    aux_weight: float = 0.2,
) -> dict[str, float]:
    data_path = Path(data_dir)
    vocabs = {
        "key_vocab": load_vocab(data_path / "key_vocab.json"),
        "roman_vocab": load_vocab(data_path / "roman_vocab.json"),
        "inversion_vocab": load_vocab(data_path / "inversion_vocab.json"),
        "dur_vocab": load_vocab(data_path / "dur_vocab.json"),
        "voice_state_vocab": load_vocab(data_path / "voice_state_vocab.json"),
    }
    val_pieces = load_pieces_jsonl(data_path / "val.jsonl")
    dataset = HarmonyV2Dataset(val_pieces, vocabs, context_length=context_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    metadata = build_voice_metadata(vocabs["voice_state_vocab"], device=device)
    deg_for_token = torch.argmax(metadata.degree_onehot, dim=1)

    model.eval()
    total_loss = 0.0
    total_batches = 0
    onset_correct = 0
    onset_total = 0
    chord_fit_hits = 0
    chord_fit_total = 0
    voice_to_idx = {"B": 0, "T": 1, "A": 2, "S": 3}

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            target = batch["target_voices"]
            outputs = model(
                key_ids=batch["key"],
                roman_ids=batch["roman"],
                inversion_ids=batch["inversion"],
                duration_ids=batch["duration"],
                prev_voices=batch["prev_voices"],
                teacher_tokens=target,
            )
            loss, _ = compute_total_loss(outputs, target, batch["chord_tones_degree"], metadata, aux_weight)
            total_loss += float(loss.item())
            total_batches += 1

            for voice, idx in voice_to_idx.items():
                pred = torch.argmax(outputs[voice], dim=-1)
                tgt = target[:, :, idx]
                tgt_onset = metadata.onset_mask[tgt]
                onset_correct += int(((pred == tgt) & tgt_onset).sum().item())
                onset_total += int(tgt_onset.sum().item())

                pred_onset = metadata.onset_mask[pred]
                pred_deg = deg_for_token[pred]
                fit = torch.gather(
                    batch["chord_tones_degree"], dim=-1, index=pred_deg.unsqueeze(-1)
                ).squeeze(-1) > 0
                chord_fit_hits += int((fit & pred_onset).sum().item())
                chord_fit_total += int(pred_onset.sum().item())

    return {
        "nll_proxy": total_loss / max(1, total_batches),
        "onset_accuracy": onset_correct / max(1, onset_total),
        "chord_fit": chord_fit_hits / max(1, chord_fit_total),
    }

