from dataclasses import dataclass, asdict
from typing import List, Optional

# benchmark
#   - "humaneval" -> HumanEval+
#   - "mbpp"     -> MBPP+
BENCHMARK: str = "mbpp"

# tasks limit
#   - 164 for full HumanEval+
#   - 974 for full MBPP+
#   - 0   for full benchmark (auto)
LIMIT: int = 974

# models
MODEL: str = "codellama/CodeLlama-7b-Instruct-hf"
# "codellama/CodeLlama-7b-Python-hf"
# "codellama/CodeLlama-7b-Instruct-hf"
# "codellama/CodeLlama-7b-hf"
# "meta-llama/Llama-2-7b-hf"
# "Qwen/Qwen2.5-Coder-7B-Instruct"

# pass@1 / pass@k
#   - pass1: deterministic
#   - passk: sampling with n>1
PASS_MODE: str = "pass1"

# sampling parameters
SEED: int = 0

# pass@1 defaults
PASS1_TEMPERATURE: float = 0.0
PASS1_N: int = 1
PASS1_TOP_P: float = 1.0
PASS1_TOP_K: int = -1

# pass@k defaults
PASSK_TEMPERATURE: float = 0.8
PASSK_N: int = 10
PASSK_TOP_P: float = 0.95
PASSK_TOP_K: int = -1

# max tokens
MAX_TOKENS: int = 512

# stop tokens
STOP: List[str] = ["```", "\n\n\n"]

# vLLM engine params
TP: int = 1  # tensor parallel size
DTYPE: str = "bfloat16"  # A100 friendly
MAX_MODEL_LEN: int = 8192  # prompt+completion max length budget
GPU_MEM_UTIL: float = 0.90  # vLLM gpu_memory_utilization
TRUST_REMOTE_CODE: bool = False  # keep default safe

# output / run naming
RUNS_ROOT: str = "runs"
RUN_NAME: str = ""


# batch run settings
BATCH_MODELS: List[str] = [
    "codellama/CodeLlama-7b-hf",
    "codellama/CodeLlama-7b-Python-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-7b-hf",
    "meta-llama/Llama-2-7b-hf",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
]
BATCH_BENCHMARKS: List[str] = ["humaneval", "mbpp"]
BATCH_KS: List[int] = [1, 10]
BATCH_CSV_PATH: str = "batch_results/summary.csv"


# helper: materialize a single dict config
def to_dict() -> dict:
    # flatten config into a JSON-serializable dict
    cfg = {
        "benchmark": BENCHMARK,
        "limit": LIMIT,
        "model": MODEL,
        "pass_mode": PASS_MODE,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "stop": STOP,
        "vllm": {
            "tp": TP,
            "dtype": DTYPE,
            "max_model_len": MAX_MODEL_LEN,
            "gpu_mem_util": GPU_MEM_UTIL,
            "trust_remote_code": TRUST_REMOTE_CODE,
        },
    }

    if PASS_MODE == "pass1":
        cfg["sampling"] = {
            "n": PASS1_N,
            "temperature": PASS1_TEMPERATURE,
            "top_p": PASS1_TOP_P,
            "top_k": PASS1_TOP_K,
        }
    elif PASS_MODE == "passk":
        cfg["sampling"] = {
            "n": PASSK_N,
            "temperature": PASSK_TEMPERATURE,
            "top_p": PASSK_TOP_P,
            "top_k": PASSK_TOP_K,
        }
    else:
        raise ValueError(f"Unknown PASS_MODE: {PASS_MODE}")

    cfg["output"] = {
        "runs_root": RUNS_ROOT,
        "run_name": RUN_NAME,
    }
    return cfg
