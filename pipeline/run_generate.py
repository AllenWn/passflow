import argparse
import importlib.util
import json
import os
import re
from typing import Any, Dict, List, Tuple

from vllm import LLM, SamplingParams


def load_config_module(config_path: str):
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("user_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load config module from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_name(s: str, max_len: int = 80) -> str:
    s = s.strip()
    s = s.replace("/", "_").replace(":", "_")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s[:max_len].strip("_")


def sorted_items(d: Dict[str, Any]) -> List[Tuple[str, Any]]:
    return [(k, d[k]) for k in sorted(d.keys())]


def load_evalplus_problems(benchmark: str) -> Dict[str, Any]:
    b = benchmark.lower()
    if b in ("humaneval", "humaneval+", "he", "he+"):
        from evalplus.data import get_human_eval_plus

        return get_human_eval_plus()
    if b in ("mbpp", "mbpp+", "mbppplus"):
        from evalplus.data import get_mbpp_plus

        return get_mbpp_plus()
    raise ValueError(f"Unsupported benchmark: {benchmark}. Use 'humaneval' or 'mbpp'.")


def default_run_name(cfg: Dict[str, Any]) -> str:
    model = safe_name(cfg["model"])
    bench = safe_name(cfg["benchmark"])
    mode = safe_name(cfg["pass_mode"])
    n = cfg["sampling"]["n"]
    temp = cfg["sampling"]["temperature"]
    return f"{bench}_{mode}_n{n}_t{temp}_{model}"


def main():
    ap = argparse.ArgumentParser(
        description="vLLM generation runner for EvalPlus benchmarks (writes samples.jsonl)."
    )
    ap.add_argument("--config", type=str, default="config.py", help="Path to config.py")
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Load config & problems, print summary, no generation.",
    )
    ap.add_argument(
        "--override_model", type=str, default=None, help="Override MODEL from config."
    )
    ap.add_argument(
        "--override_benchmark",
        type=str,
        default=None,
        help="Override BENCHMARK from config (humaneval/mbpp).",
    )
    ap.add_argument(
        "--override_pass_mode",
        type=str,
        default=None,
        help="Override PASS_MODE from config (pass1/passk).",
    )
    ap.add_argument(
        "--override_limit", type=int, default=None, help="Override LIMIT from config."
    )
    ap.add_argument(
        "--override_run_name",
        type=str,
        default=None,
        help="Override RUN_NAME from config.",
    )
    ap.add_argument(
        "--override_sampling_n",
        type=int,
        default=None,
        help="Override sampling.n (number of candidates per task).",
    )
    ap.add_argument(
        "--override_sampling_temperature",
        type=float,
        default=None,
        help="Override sampling.temperature.",
    )
    ap.add_argument(
        "--override_sampling_top_p",
        type=float,
        default=None,
        help="Override sampling.top_p.",
    )
    ap.add_argument(
        "--override_sampling_top_k",
        type=int,
        default=None,
        help="Override sampling.top_k.",
    )
    args = ap.parse_args()

    cfg_mod = load_config_module(args.config)

    if not hasattr(cfg_mod, "to_dict"):
        raise RuntimeError(
            "config.py must define a to_dict() function that returns a dict."
        )

    cfg: Dict[str, Any] = cfg_mod.to_dict()

    # Optional overrides
    if args.override_model is not None:
        cfg["model"] = str(args.override_model)
    if args.override_benchmark is not None:
        cfg["benchmark"] = str(args.override_benchmark)
    if args.override_pass_mode is not None:
        cfg["pass_mode"] = str(args.override_pass_mode)
    if args.override_limit is not None:
        cfg["limit"] = int(args.override_limit)
    if args.override_run_name is not None:
        cfg["output"]["run_name"] = args.override_run_name
    if args.override_sampling_n is not None:
        cfg.setdefault("sampling", {})["n"] = int(args.override_sampling_n)
    if args.override_sampling_temperature is not None:
        cfg.setdefault("sampling", {})["temperature"] = float(
            args.override_sampling_temperature
        )
    if args.override_sampling_top_p is not None:
        cfg.setdefault("sampling", {})["top_p"] = float(args.override_sampling_top_p)
    if args.override_sampling_top_k is not None:
        cfg.setdefault("sampling", {})["top_k"] = int(args.override_sampling_top_k)

    # Resolve output paths
    runs_root = cfg["output"]["runs_root"]
    run_name = cfg["output"]["run_name"] or default_run_name(cfg)
    outdir = os.path.join(runs_root, run_name)
    ensure_dir(outdir)

    samples_path = os.path.join(outdir, "samples.jsonl")
    config_dump_path = os.path.join(outdir, "run_config.json")

    # Load benchmark problems
    problems = load_evalplus_problems(cfg["benchmark"])
    total = len(problems)

    limit = int(cfg["limit"])
    # Minimal + convenient: allow limit<=0 to mean "run full benchmark".
    # This also makes batch runs with mixed benchmarks easier.
    if limit <= 0:
        limit = total

    items = sorted_items(problems)[:limit]
    task_ids = [tid for tid, _ in items]
    prompts = [prob["prompt"] for _, prob in items]

    print(
        f"[INFO] Config: benchmark={cfg['benchmark']} total={total} running={len(prompts)}"
    )
    print(f"[INFO] Model: {cfg['model']}")
    print(f"[INFO] Pass mode: {cfg['pass_mode']}")
    print(f"[INFO] Output dir: {outdir}")
    print(f"[INFO] Will write: {samples_path}")

    # Dump config for reproducibility (always)
    with open(config_dump_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote run_config.json: {config_dump_path}")

    if args.dry_run:
        print("[DRY RUN] Exiting before generation.")
        return

    # Init vLLM engine
    vcfg = cfg["vllm"]
    llm = LLM(
        model=cfg["model"],
        tensor_parallel_size=int(vcfg["tp"]),
        dtype=vcfg["dtype"],
        trust_remote_code=bool(vcfg["trust_remote_code"]),
        max_model_len=int(vcfg["max_model_len"]),
        gpu_memory_utilization=float(vcfg["gpu_mem_util"]),
    )

    # Sampling params
    scfg = cfg["sampling"]
    sampling_params = SamplingParams(
        n=int(scfg["n"]),
        temperature=float(scfg["temperature"]),
        top_p=float(scfg["top_p"]),
        top_k=int(scfg["top_k"]),
        max_tokens=int(cfg["max_tokens"]),
        stop=list(cfg["stop"]),
        seed=int(cfg["seed"]),
    )

    # Generate
    print("[INFO] Generating with vLLM...")
    outputs = llm.generate(prompts, sampling_params)

    # Build samples.jsonl
    # Use "solution" = prompt + completion for robustness.
    # For pass@k (n>1), write multiple lines per task_id (one per candidate).
    rows: List[Dict[str, Any]] = []
    for i, out in enumerate(outputs):
        tid = task_ids[i]
        prompt = prompts[i]
        for cand in out.outputs:
            rows.append(
                {
                    "task_id": tid,
                    "solution": prompt + cand.text,
                }
            )

    write_jsonl(samples_path, rows)

    n = int(scfg["n"])
    expected_lines = len(prompts) * n
    print(f"[OK] Wrote samples.jsonl: {samples_path}")
    print(
        f"[INFO] Lines: {len(rows)} (expected {expected_lines} = tasks({len(prompts)}) * n({n}))"
    )

    print("\nNext step (evaluation):")
    bench = cfg["benchmark"].lower()
    if bench in ("humaneval", "humaneval+"):
        bench = "humaneval"
    elif bench in ("mbpp", "mbpp+"):
        bench = "mbpp"
    print(f"  evalplus.evaluate {bench} --samples {samples_path}")


if __name__ == "__main__":
    main()
