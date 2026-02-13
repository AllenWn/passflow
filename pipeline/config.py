from dataclasses import dataclass, asdict
from typing import List, Optional

# benchmark
#   - "humaneval" -> HumanEval+
#   - "mbpp"     -> MBPP+
BENCHMARK: str = "humaneval"

# tasks limit
#   - 164 for full HumanEval+
#   - 974 for full MBPP+
#   - 0   for full benchmark (auto)
LIMIT: int = 0

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

# prompting / chat settings
# prompting mode:
#   - "completion": always use benchmark prompt as completion prefix (base models)
#   - "chat": always wrap benchmark prompt into chat messages via tokenizer chat_template
#   - "vendor_default": (legacy) same as chat; system prompt selection controlled by flags below
#   - "auto": choose interface based on model_profiles.json (if present) else tokenizer.chat_template
PROMPT_MODE: str = "auto"

# system prompt control (applies to BOTH chat + completion inputs):
#   - if disabled, no system prompt will be added anywhere
#   - if enabled, prefer per-model override in model_profiles.json (models.<id>.system_prompt),
#     otherwise fall back to SYSTEM_PROMPT below.
ENABLE_SYSTEM_PROMPT: bool = True
SYSTEM_PROMPT: str = "You are a useful coding assistant. Output only valid Python code."
MODEL_PROFILES_PATH: str = "model_profiles.json"

# debug: dump the effective prompts for inspection
PROMPT_DEBUG: bool = False
PROMPT_DEBUG_N: int = 1


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

    cfg["prompting"] = {
        "mode": PROMPT_MODE,
        "enable_system_prompt": ENABLE_SYSTEM_PROMPT,
        "system_prompt": SYSTEM_PROMPT,
        "model_profiles_path": MODEL_PROFILES_PATH,
        "debug": PROMPT_DEBUG,
        "debug_n": PROMPT_DEBUG_N,
    }
    return cfg
