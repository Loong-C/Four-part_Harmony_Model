from __future__ import annotations

import argparse

from harmony.common.config import load_config
from harmony.train.runner import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Harmony V2 model.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()

