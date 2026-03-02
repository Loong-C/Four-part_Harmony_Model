from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class VoiceTokenMetadata:
    token_weights: torch.Tensor
    onset_mask: torch.Tensor
    degree_onehot: torch.Tensor


def build_voice_metadata(voice_vocab: dict[str, int], device: torch.device) -> VoiceTokenMetadata:
    vocab_size = len(voice_vocab)
    token_weights = torch.ones(vocab_size, dtype=torch.float32, device=device)
    onset_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    degree_onehot = torch.zeros(vocab_size, 12, dtype=torch.float32, device=device)
    for token, token_id in voice_vocab.items():
        if token.startswith("ON_"):
            _, deg, _ = token.split("_")
            deg_id = int(deg)
            onset_mask[token_id] = True
            degree_onehot[token_id, deg_id] = 1.0
            token_weights[token_id] = 1.5
        elif token == "<HOLD>":
            token_weights[token_id] = 0.7
        elif token == "<REST>":
            token_weights[token_id] = 1.0
    return VoiceTokenMetadata(
        token_weights=token_weights,
        onset_mask=onset_mask,
        degree_onehot=degree_onehot,
    )


def weighted_ce_loss(logits: torch.Tensor, target: torch.Tensor, token_weights: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_target = target.reshape(-1)
    base = F.cross_entropy(flat_logits, flat_target, reduction="none")
    weights = token_weights[flat_target]
    return (base * weights).sum() / torch.clamp(weights.sum(), min=1.0)


def chord_tone_aux_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    chord_tones_degree: torch.Tensor,
    metadata: VoiceTokenMetadata,
) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_target = target.reshape(-1)
    flat_chord = chord_tones_degree.reshape(-1, 12)
    probs = torch.softmax(flat_logits, dim=-1)
    allowed = (flat_chord @ metadata.degree_onehot.transpose(0, 1) > 0).float()
    chord_mass = torch.clamp((probs * allowed).sum(dim=-1), min=1e-8)
    onset_targets = metadata.onset_mask[flat_target]
    if not torch.any(onset_targets):
        return torch.zeros((), dtype=flat_logits.dtype, device=flat_logits.device)
    return -torch.log(chord_mass[onset_targets]).mean()


def compute_total_loss(
    outputs: dict[str, torch.Tensor],
    target_voices: torch.Tensor,
    chord_tones_degree: torch.Tensor,
    metadata: VoiceTokenMetadata,
    aux_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    voice_to_idx = {"B": 0, "T": 1, "A": 2, "S": 3}
    total_ce = torch.zeros((), dtype=torch.float32, device=target_voices.device)
    total_aux = torch.zeros((), dtype=torch.float32, device=target_voices.device)
    for voice, idx in voice_to_idx.items():
        logits = outputs[voice]
        target = target_voices[:, :, idx]
        total_ce = total_ce + weighted_ce_loss(logits, target, metadata.token_weights)
        total_aux = total_aux + chord_tone_aux_loss(logits, target, chord_tones_degree, metadata)
    total = total_ce + aux_weight * total_aux
    details = {
        "loss_ce": float(total_ce.detach().item()),
        "loss_aux": float(total_aux.detach().item()),
        "loss_total": float(total.detach().item()),
    }
    return total, details

