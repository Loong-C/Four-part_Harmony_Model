from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .dataset_v2 import save_pieces_jsonl
from .extract_v2 import collect_bach_paths, extract_piece
from .split import split_piece_ids, split_pieces
from .vocab import build_vocabs, save_vocabs


def run_build(out_dir: str, seed: int = 42) -> dict[str, int]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = collect_bach_paths()
    pieces: list[dict] = []
    failures: list[dict[str, str]] = []
    roman_map: dict[str, str] = {}

    for path in paths:
        piece_id = Path(path).stem
        try:
            piece, norm_map = extract_piece(path, piece_id=piece_id)
            pieces.append(piece.to_dict())
            roman_map.update(norm_map)
        except Exception as exc:
            failures.append({"piece_id": piece_id, "source_path": str(path), "error": str(exc)})

    save_pieces_jsonl(out / "pieces.jsonl", pieces)
    (out / "roman_normalization_map.json").write_text(
        json.dumps(roman_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    vocabs = build_vocabs(pieces)
    save_vocabs(vocabs, out)

    manifest = split_piece_ids([p["piece_id"] for p in pieces], seed=seed, train_ratio=0.8, val_ratio=0.1)
    (out / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    by_split = split_pieces(pieces, manifest)
    save_pieces_jsonl(out / "train.jsonl", by_split["train"])
    save_pieces_jsonl(out / "val.jsonl", by_split["val"])
    save_pieces_jsonl(out / "test.jsonl", by_split["test"])

    with (out / "failures.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["piece_id", "source_path", "error"])
        writer.writeheader()
        for row in failures:
            writer.writerow(row)

    stats = {
        "num_candidates": len(paths),
        "num_success": len(pieces),
        "num_failures": len(failures),
        "num_train": len(by_split["train"]),
        "num_val": len(by_split["val"]),
        "num_test": len(by_split["test"]),
    }
    (out / "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Harmony V2 dataset.")
    parser.add_argument("--out", type=str, required=True, help="Output directory, e.g. data/v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    stats = run_build(out_dir=args.out, seed=args.seed)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

