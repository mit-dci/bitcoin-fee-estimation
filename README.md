# Bitcoin Fee Estimation Pipeline

A two-stage structural model for Bitcoin transaction fee estimation based on VCG (Vickrey-Clarke-Groves) pricing theory. The model decomposes fees into a **delay technology** (how priority affects confirmation delay) and **preferences** (how impatience drives willingness to pay).

## Pipeline Overview

```
Stage 1 (Priority → Delay)   Fee-rate ranking within epochs → Random Forest
                              predicts delay from priority → isotonic regression
                              enforces monotonicity → finite differences compute
                              local slope W'_it

Stage 2 (Fee Equation)        Spline regression:
                              log(fee_rate) ~ s(impatience) + log(W') + controls
                              with epoch-clustered standard errors
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline from the pre-built dataset
python run_pipeline.py --dataset data/bitcoin_data.parquet
```

Output is written to `model_outputs/` by default.

## Usage

```bash
# Basic run from parquet data
python run_pipeline.py --dataset data/analysis_ready.parquet

# Include temporal variance analysis
python run_pipeline.py --dataset data/analysis_ready.parquet --temporal-variance

# Smoke test on a subset of blocks, skip plots
python run_pipeline.py --dataset data/analysis_ready.parquet --block-limit 100 --skip-plots

# Use monotone XGBoost instead of Random Forest for Stage 1
python run_pipeline.py --dataset data/analysis_ready.parquet --stage1-model xgb_monotone

# Full run with checkpointing (resumable after crash)
python run_pipeline.py --dataset data/analysis_ready.parquet --checkpoint-dir /tmp/ckpt/

# Custom output directory and verbose logging
python run_pipeline.py --dataset data/analysis_ready.parquet --output-dir results/ --log-level DEBUG
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | None | Path to pre-built Parquet dataset |
| `--stage1-model` | `rf` | Stage 1 learner: `rf` (Random Forest) or `xgb_monotone` |
| `--n-folds` | 5 | Cross-validation folds |
| `--epoch-mode` | `time` | Epoch strategy: `time`, `block`, or `fixed_size` |
| `--epoch-duration` | 30 | Minutes per epoch (for `--epoch-mode time`) |
| `--output-dir` | `model_outputs` | Directory for output files |
| `--checkpoint-dir` | None | Save/load checkpoints for resumability |
| `--skip-plots` | off | Skip matplotlib output |
| `--temporal-variance` | off | Run temporal variance analysis after the main pipeline |
| `--log-level` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## Files

| File | Description |
|------|-------------|
| `run_pipeline.py` | CLI entry point — runs the full two-stage model |
| `build_dataset.py` | Data loading and feature engineering |
| `temporal_variance_analysis.py` | Post-estimation temporal variance diagnostics |
| `bitcoin_fee_estimation_pipeline.ipynb` | Interactive notebook version of the pipeline |
| `requirements.txt` | Python dependencies |
