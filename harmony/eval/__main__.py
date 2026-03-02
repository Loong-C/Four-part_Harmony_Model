from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harmony.common.config import load_config
from harmony.eval.metrics import evaluate_validation_metrics
from harmony.eval.theory import evaluate_generated_batch
from harmony.infer.generate_v2 import generate_harmony_v2, load_generator_artifacts


def _normalize_key(case: dict[str, Any], default_mode: str) -> str:
    key = str(case["key"])
    if ":" in key:
        return key
    mode = str(case.get("mode", default_mode))
    return f"{key}:{mode}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Harmony V2 model.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--suite", type=str, required=True, help="Suite YAML/JSON path")
    parser.add_argument("--config", type=str, default="configs/gen_v2.yaml", help="Generator config path")
    parser.add_argument("--out", type=str, default="reports/eval_report_v2.json", help="Output report path")
    args = parser.parse_args()

    gen_config = load_config(args.config)
    gen_config["checkpoint"] = args.ckpt
    suite = load_config(args.suite)

    artifacts = load_generator_artifacts(gen_config, ckpt_path=args.ckpt)
    eval_cfg = load_config("configs/train_v2.yaml")
    val_metrics = evaluate_validation_metrics(
        model=artifacts.model,
        data_dir=gen_config.get("data_dir", "data/v2"),
        context_length=int(eval_cfg.get("context_length", 32)),
        batch_size=int(eval_cfg.get("eval_batch_size", 32)),
        device=artifacts.device,
        aux_weight=float(eval_cfg.get("aux_weight", 0.2)),
    )

    default_mode = str(gen_config.get("key_mode", "major"))
    samples: list[list[dict[str, Any]]] = []
    for case in suite.get("cases", []):
        repeats = int(case.get("repeats", 1))
        progression = list(case["progression"])
        durations = [float(x) for x in case.get("durations", [1.0] * len(progression))]
        key_name = _normalize_key(case, default_mode=default_mode)
        for rep in range(repeats):
            result = generate_harmony_v2(
                artifacts=artifacts,
                key_name=key_name,
                progression=progression,
                durations=durations,
                num_candidates=int(gen_config.get("num_candidates", 8)),
                temperature=float(gen_config.get("temperature", 0.9)),
                top_k=int(gen_config.get("top_k", 24)),
                seed=int(gen_config.get("seed", 42)) + rep,
            )
            samples.append(result)

    theory_metrics = evaluate_generated_batch(samples)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.ckpt,
        "suite": args.suite,
        "validation": val_metrics,
        "theory": theory_metrics,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")
    print(f"saved_abs: {out_path.resolve()}")


if __name__ == "__main__":
    main()
