from __future__ import annotations

import argparse
import json
from pathlib import Path

from harmony.common.config import load_config
from harmony.infer.generate_v2 import (
    format_human_readable_table,
    generate_harmony_v2,
    load_generator_artifacts,
    save_generated,
    save_human_readable,
    to_human_readable_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Harmony V2 SATB with controlled progression.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--key", type=str, required=True, help="Key tonic, e.g. D or a")
    parser.add_argument("--prog", type=str, required=True, help="Comma-separated progression, e.g. I,IV,V,I")
    parser.add_argument("--durs", type=str, default="", help="Comma-separated durations; defaults to all 1.0")
    parser.add_argument("--ckpt", type=str, default="", help="Override checkpoint path")
    args = parser.parse_args()

    config = load_config(args.config)
    key_mode = config.get("key_mode", "major")
    key_name = f"{args.key}:{key_mode}" if ":" not in args.key else args.key
    progression = [p.strip() for p in args.prog.split(",") if p.strip()]
    if args.durs.strip():
        durations = [float(x.strip()) for x in args.durs.split(",")]
    else:
        durations = [1.0] * len(progression)

    artifacts = load_generator_artifacts(config, ckpt_path=args.ckpt or None)
    result = generate_harmony_v2(
        artifacts=artifacts,
        key_name=key_name,
        progression=progression,
        durations=durations,
        num_candidates=int(config.get("num_candidates", 8)),
        temperature=float(config.get("temperature", 0.9)),
        top_k=int(config.get("top_k", 24)),
        seed=int(config.get("seed", 42)),
    )

    out_path = Path(config.get("output_path", "generated_score_v2.json"))
    readable_json_path = Path(config.get("readable_output_path", "generated_score_v2_readable.json"))
    readable_table_path = Path(config.get("readable_table_path", "generated_score_v2_readable.txt"))

    save_generated(out_path, result)
    readable_rows = to_human_readable_rows(result)
    saved_json, saved_table = save_human_readable(
        json_path=readable_json_path,
        txt_path=readable_table_path,
        rows=readable_rows,
    )

    print(format_human_readable_table(readable_rows))
    print()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")
    print(f"saved_readable_json: {saved_json}")
    print(f"saved_readable_table: {saved_table}")


if __name__ == "__main__":
    main()
