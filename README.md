# EvalPlus Pipeline

Code generation evaluation pipeline using vLLM + EvalPlus: single-run and batch automation, with pass@k results written to CSV.

## Quick Start

### 1. Environment

```bash
bash setup_env.sh
# optional: bash setup_env.sh -n myenv -p 3.12
conda activate evalplus-pipeline
```

### 2. Run

```bash
cd pipeline

# Single run: one model, one benchmark, one pass config
bash run_eval.sh config.py

# Batch: run all combinations from BATCH_MODELS / BATCH_BENCHMARKS / BATCH_KS
bash run_eval.sh --batch config.py
```

## Config (pipeline/config.py)

| Single run | Description |
|------------|-------------|
| `BENCHMARK` | humaneval / mbpp |
| `MODEL` | HuggingFace model ID |
| `PASS_MODE` | pass1 / passk |
| `LIMIT` | number of tasks; 0 = full benchmark |

| Batch | Description |
|-------|-------------|
| `BATCH_MODELS` | list of models |
| `BATCH_BENCHMARKS` | humaneval, mbpp |
| `BATCH_KS` | e.g. [1, 10] |
| `BATCH_CSV_PATH` | output CSV path |

## Directory Structure

```
pipeline/
├── config.py      # single + batch config
├── run_eval.sh    # entry: single / --batch
├── run_generate.py
├── run_batch.py
├── runs/          # per-run outputs
└── batch_results/ # batch summary CSV
```

## Dependencies

- Python 3.12+
- conda
- CUDA (vLLM requires GPU)
- `evalplus[vllm]` (installed by setup_env.sh)
