from __future__ import annotations

import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from harmony.common.config import save_json
from harmony.common.seed import set_seed
from harmony.data.dataset_v2 import HarmonyV2Dataset, load_pieces_jsonl
from harmony.data.vocab import load_vocab
from harmony.models.conditional_satb import ConditionalSATBTransformer
from harmony.train.losses import build_voice_metadata, compute_total_loss


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad or name not in self.shadow:
                    continue
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].data)
        return backup

    @staticmethod
    def restore(model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name].data)


def _device_from_config(config: dict[str, Any]) -> torch.device:
    forced = str(config.get("device", "auto"))
    if forced != "auto":
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_vocabs(data_dir: Path) -> dict[str, dict[str, int]]:
    return {
        "key_vocab": load_vocab(data_dir / "key_vocab.json"),
        "roman_vocab": load_vocab(data_dir / "roman_vocab.json"),
        "inversion_vocab": load_vocab(data_dir / "inversion_vocab.json"),
        "dur_vocab": load_vocab(data_dir / "dur_vocab.json"),
        "voice_state_vocab": load_vocab(data_dir / "voice_state_vocab.json"),
    }


def _load_split(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train = load_pieces_jsonl(data_dir / "train.jsonl")
    val = load_pieces_jsonl(data_dir / "val.jsonl")
    test = load_pieces_jsonl(data_dir / "test.jsonl")
    return train, val, test


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _evaluate(
    model: ConditionalSATBTransformer,
    loader: DataLoader,
    metadata,
    aux_weight: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    onset_correct = 0
    onset_total = 0
    chord_fit_hits = 0
    chord_fit_total = 0
    voice_to_idx = {"B": 0, "T": 1, "A": 2, "S": 3}
    deg_for_token = torch.argmax(metadata.degree_onehot, dim=1)

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
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
            total_loss += float(loss.item()) * int(target.shape[0])
            total_tokens += int(target.shape[0])

            for voice, idx in voice_to_idx.items():
                logits = outputs[voice]
                pred = torch.argmax(logits, dim=-1)
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

    avg_loss = total_loss / max(1, total_tokens)
    onset_acc = onset_correct / max(1, onset_total)
    chord_fit = chord_fit_hits / max(1, chord_fit_total)
    return {"val_nll_proxy": avg_loss, "onset_accuracy": onset_acc, "chord_fit": chord_fit}


def run_training(config: dict[str, Any]) -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = _device_from_config(config)

    data_dir = Path(config.get("data_dir", "data/v2"))
    out_dir = Path(config.get("output_dir", "checkpoints/v2"))
    out_dir.mkdir(parents=True, exist_ok=True)

    vocabs = _load_vocabs(data_dir)
    train_pieces, val_pieces, _ = _load_split(data_dir)

    context_length = int(config.get("context_length", 32))
    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 0))
    train_set = HarmonyV2Dataset(train_pieces, vocabs, context_length)
    val_set = HarmonyV2Dataset(val_pieces, vocabs, context_length)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model_cfg = config.get("model", {})
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
    ).to(device)

    lr = float(config.get("lr", 3e-4))
    weight_decay = float(config.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    metadata = build_voice_metadata(vocabs["voice_state_vocab"], device=device)

    epochs = int(config.get("epochs", 40))
    grad_clip = float(config.get("grad_clip", 1.0))
    grad_accum_steps = int(config.get("grad_accum_steps", 1))
    aux_weight = float(config.get("aux_weight", 0.2))
    ema = EMA(model, decay=float(config.get("ema_decay", 0.999)))
    patience = int(config.get("early_stop_patience", 6))
    use_amp = bool(config.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    (out_dir / "config_resolved.yaml").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        git_rev = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # nosec B603
        )
    except Exception:
        git_rev = "unknown"
    (out_dir / "git_rev.txt").write_text(git_rev + "\n", encoding="utf-8")
    metrics_path = out_dir / "metrics.jsonl"

    best_nll = float("inf")
    best_theory = float("-inf")
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(train_loader, start=1):
            batch = _to_device(batch, device)
            target = batch["target_voices"]
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with amp_ctx:
                outputs = model(
                    key_ids=batch["key"],
                    roman_ids=batch["roman"],
                    inversion_ids=batch["inversion"],
                    duration_ids=batch["duration"],
                    prev_voices=batch["prev_voices"],
                    teacher_tokens=target,
                )
                loss, _ = compute_total_loss(
                    outputs=outputs,
                    target_voices=target,
                    chord_tones_degree=batch["chord_tones_degree"],
                    metadata=metadata,
                    aux_weight=aux_weight,
                )
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            if step_idx % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

            running += float(loss.item() * grad_accum_steps)
            steps += 1

        backup = ema.apply_to(model)
        val_metrics = _evaluate(model, val_loader, metadata, aux_weight, device)
        EMA.restore(model, backup)

        train_loss = running / max(1, steps)
        theory_score = val_metrics["chord_fit"] + val_metrics["onset_accuracy"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "theory_score": theory_score,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        last_ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "vocabs": {k: str(data_dir / f"{k}.json") for k in vocabs.keys()},
        }
        torch.save(last_ckpt, out_dir / "last.ckpt")

        if val_metrics["val_nll_proxy"] < best_nll:
            best_nll = val_metrics["val_nll_proxy"]
            wait = 0
            torch.save(last_ckpt, out_dir / "best_nll.ckpt")
        else:
            wait += 1

        if theory_score > best_theory:
            best_theory = theory_score
            torch.save(last_ckpt, out_dir / "best_theory.ckpt")

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_nll={val_metrics['val_nll_proxy']:.4f} "
            f"onset_acc={val_metrics['onset_accuracy']:.4f} "
            f"chord_fit={val_metrics['chord_fit']:.4f}"
        )
        if wait >= patience:
            print(f"early stop at epoch {epoch}")
            break

    save_json(out_dir / "training_summary.json", {"best_nll": best_nll, "best_theory": best_theory})
