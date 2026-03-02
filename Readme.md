# Harmony V2

Harmony V2 is a controlled four-part harmony pipeline with:
- piece-level dataset building (no cross-piece training windows),
- key-aware relative pitch representation,
- conditional SATB Transformer with stepwise voice order `B -> T -> A -> S`,
- soft-constrained reranking at generation time,
- quantitative + theory-oriented evaluation.

## Deprecated V1 scripts

The original V1 scripts have been migrated into `legacy/` and are now **deprecated**.
Use the new `harmony/` package CLIs below for V2.
Legacy notes are in [legacy/README.md](legacy/README.md).

## V2 CLI

1. Build data:
```bash
python -m harmony.data.build_v2 --out data/v2
```

2. Train:
```bash
python -m harmony.train --config configs/train_v2.yaml
```

3. Generate:
```bash
python -m harmony.generate --config configs/gen_v2.yaml --key D --prog I,IV,V,I
```

4. Evaluate:
```bash
python -m harmony.eval --ckpt checkpoints/v2/best_nll.ckpt --suite configs/benchmark_progressions.yaml
```

## Key Outputs

- Dataset: `data/v2/pieces.jsonl`, `train.jsonl`, `val.jsonl`, `test.jsonl`
- Vocab: `data/v2/{roman_vocab,inversion_vocab,dur_vocab,degree_vocab,octave_vocab,voice_state_vocab}.json`
- Training artifacts: `checkpoints/v2/{last.ckpt,best_nll.ckpt,best_theory.ckpt,metrics.jsonl}`
- Generation output: `generated_score_v2.json` (configurable)
- Human-readable generation output: `generated_score_v2_readable.json` + `generated_score_v2_readable.txt`
- Evaluation report: `reports/eval_report_v2*.json`
