from __future__ import annotations

import random
from typing import Any


def split_piece_ids(
    piece_ids: list[str],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, list[str]]:
    ids = list(piece_ids)
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def split_pieces(pieces: list[dict[str, Any]], manifest: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    id_to_piece = {p["piece_id"]: p for p in pieces}
    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for split_name in ("train", "val", "test"):
        for pid in manifest[split_name]:
            piece = id_to_piece.get(pid)
            if piece is not None:
                out[split_name].append(piece)
    return out

