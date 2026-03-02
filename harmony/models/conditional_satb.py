from __future__ import annotations

import math

import torch
import torch.nn as nn


class ConditionalSATBTransformer(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict[str, int],
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 1024,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.key_emb = nn.Embedding(vocab_sizes["key"], d_model)
        self.roman_emb = nn.Embedding(vocab_sizes["roman"], d_model)
        self.inversion_emb = nn.Embedding(vocab_sizes["inversion"], d_model)
        self.duration_emb = nn.Embedding(vocab_sizes["duration"], d_model)
        self.voice_emb = nn.Embedding(vocab_sizes["voice"], d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head_b = nn.Linear(d_model, vocab_sizes["voice"])
        self.head_t = nn.Linear(d_model, vocab_sizes["voice"])
        self.head_a = nn.Linear(d_model, vocab_sizes["voice"])
        self.head_s = nn.Linear(d_model, vocab_sizes["voice"])

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def encode_context(
        self,
        key_ids: torch.Tensor,
        roman_ids: torch.Tensor,
        inversion_ids: torch.Tensor,
        duration_ids: torch.Tensor,
        prev_voices: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len = key_ids.shape
        pos = torch.arange(seq_len, device=key_ids.device).unsqueeze(0).expand(batch, seq_len)
        x = (
            self.key_emb(key_ids)
            + self.roman_emb(roman_ids)
            + self.inversion_emb(inversion_ids)
            + self.duration_emb(duration_ids)
            + self.voice_emb(prev_voices).sum(dim=2)
            + self.pos_emb(pos)
        )
        x = self.norm(self.dropout(x)) * math.sqrt(self.d_model)
        hidden = self.transformer(x, mask=self._causal_mask(seq_len, key_ids.device), is_causal=True)
        return hidden

    def forward(
        self,
        key_ids: torch.Tensor,
        roman_ids: torch.Tensor,
        inversion_ids: torch.Tensor,
        duration_ids: torch.Tensor,
        prev_voices: torch.Tensor,
        teacher_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode_context(key_ids, roman_ids, inversion_ids, duration_ids, prev_voices)
        b_logits = self.head_b(hidden)
        if teacher_tokens is None:
            b_tokens = b_logits.argmax(dim=-1)
        else:
            b_tokens = teacher_tokens[:, :, 0]

        t_logits = self.head_t(hidden + self.voice_emb(b_tokens))
        if teacher_tokens is None:
            t_tokens = t_logits.argmax(dim=-1)
        else:
            t_tokens = teacher_tokens[:, :, 1]

        a_logits = self.head_a(hidden + self.voice_emb(b_tokens) + self.voice_emb(t_tokens))
        if teacher_tokens is None:
            a_tokens = a_logits.argmax(dim=-1)
        else:
            a_tokens = teacher_tokens[:, :, 2]

        s_logits = self.head_s(
            hidden + self.voice_emb(b_tokens) + self.voice_emb(t_tokens) + self.voice_emb(a_tokens)
        )

        return {"B": b_logits, "T": t_logits, "A": a_logits, "S": s_logits}

