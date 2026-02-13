import argparse
import importlib.util
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

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


def resolve_path(base_dir: str, maybe_path: Optional[str]) -> Optional[str]:
    if not maybe_path:
        return None
    if os.path.isabs(maybe_path):
        return maybe_path
    return os.path.abspath(os.path.join(base_dir, maybe_path))


def load_model_profiles(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data.get("models", {}) if isinstance(data.get("models", {}), dict) else {}


def infer_interface_with_tokenizer(model: str, trust_remote_code: bool) -> str:
    """
    Best-effort inference:
      - if tokenizer has a non-empty chat_template -> "chat"
      - else -> "completion"
    """
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        chat_template = getattr(tok, "chat_template", None)
        if isinstance(chat_template, str) and chat_template.strip():
            return "chat"
    except Exception:
        pass
    return "completion"


def build_chat_prompt(
    tokenizer: Any,
    system_prompt: str,
    user_content: str,
    add_generation_prompt: bool = True,
) -> str:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            "Tokenizer does not support apply_chat_template(); cannot build chat prompt."
        )
    # Different tokenizers may or may not accept add_generation_prompt.
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False)


def dump_prompt_debug(
    outdir: str,
    cfg: Dict[str, Any],
    task_ids: List[str],
    eval_prompts: List[str],
    model_prompts: List[str],
    chosen_interface: str,
    prompt_mode: str,
    enable_system_prompt: bool,
    system_prompt_source: str,
    sys_to_use: str,
    n: int,
) -> None:
    path = os.path.join(outdir, "prompt_debug.txt")
    n = max(0, min(int(n), len(task_ids)))
    with open(path, "w", encoding="utf-8") as f:
        f.write("== Prompt debug dump ==\n")
        f.write(f"model: {cfg.get('model')}\n")
        f.write(f"benchmark: {cfg.get('benchmark')}\n")
        f.write(f"prompting.mode: {prompt_mode}\n")
        f.write(f"chosen_interface: {chosen_interface}\n")
        f.write(f"enable_system_prompt: {enable_system_prompt}\n")
        f.write(f"system_prompt_source: {system_prompt_source}\n")
        f.write("---- system_prompt (effective) ----\n")
        f.write(sys_to_use + "\n")
        f.write("---- samples ----\n")
        for i in range(n):
            f.write(f"\n## i={i} task_id={task_ids[i]}\n")
            f.write("-- eval_prompt (benchmark) --\n")
            f.write(eval_prompts[i] + "\n")
            f.write("-- model_prompt (sent to model) --\n")
            f.write(model_prompts[i] + "\n")
    print(f"[DEBUG] Wrote prompt debug: {path}")


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

    config_path_abs = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path_abs)

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
    eval_prompts = [prob["prompt"] for _, prob in items]

    # Prompting / interface selection
    prompting_cfg = (
        cfg.get("prompting", {}) if isinstance(cfg.get("prompting", {}), dict) else {}
    )
    prompt_mode = str(prompting_cfg.get("mode", "completion")).lower()
    enable_system_prompt = bool(prompting_cfg.get("enable_system_prompt", True))
    global_system_prompt = str(prompting_cfg.get("system_prompt", ""))
    debug_prompting = bool(prompting_cfg.get("debug", False))
    debug_n = int(prompting_cfg.get("debug_n", 1) or 1)
    profiles_path = resolve_path(config_dir, prompting_cfg.get("model_profiles_path"))
    if profiles_path and not os.path.exists(profiles_path):
        print(f"[WARN] Model profiles path not found (ignored): {profiles_path}")
        profiles_path = None
    model_profiles = load_model_profiles(profiles_path)
    profile = (
        model_profiles.get(cfg["model"], {})
        if isinstance(model_profiles.get(cfg["model"], {}), dict)
        else {}
    )
    interface = str(profile.get("interface", "")).lower().strip()
    per_model_system_prompt = str(
        profile.get("system_prompt", "")
        or profile.get("vendor_system_prompt", "")
        or ""
    )

    vcfg = cfg["vllm"]
    trust_remote_code = bool(vcfg.get("trust_remote_code", False))

    if prompt_mode not in ("completion", "chat", "vendor_default", "auto"):
        raise ValueError(
            f"Unsupported prompting.mode: {prompt_mode}. Use completion/chat/vendor_default/auto."
        )

    if prompt_mode == "auto":
        if interface in ("chat", "completion"):
            chosen_interface = interface
        else:
            chosen_interface = infer_interface_with_tokenizer(
                cfg["model"], trust_remote_code
            )
    elif prompt_mode == "completion":
        chosen_interface = "completion"
    else:
        # chat or vendor_default
        chosen_interface = "chat"

    if enable_system_prompt:
        system_prompt_source = (
            "model_profiles.system_prompt"
            if per_model_system_prompt.strip()
            else "config.system_prompt"
        )
        sys_to_use = (
            per_model_system_prompt
            if per_model_system_prompt.strip()
            else global_system_prompt
        )
    else:
        system_prompt_source = "disabled"
        sys_to_use = ""

    # Build model prompts (may differ from eval prompts in chat mode).
    model_prompts: List[str]
    tokenizer = None
    if chosen_interface == "completion":
        if sys_to_use.strip():
            prefix = sys_to_use.rstrip() + "\n\n"
            model_prompts = [prefix + p for p in eval_prompts]
        else:
            model_prompts = list(eval_prompts)
    else:
        try:
            from transformers import AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(
                cfg["model"], trust_remote_code=trust_remote_code
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer for chat prompting: {cfg['model']}. "
                f"Install/verify transformers and model files. Original error: {e}"
            ) from e

        model_prompts = [
            build_chat_prompt(tokenizer, sys_to_use, p, add_generation_prompt=True)
            for p in eval_prompts
        ]

    if debug_prompting:
        dump_prompt_debug(
            outdir=outdir,
            cfg=cfg,
            task_ids=task_ids,
            eval_prompts=eval_prompts,
            model_prompts=model_prompts,
            chosen_interface=chosen_interface,
            prompt_mode=prompt_mode,
            enable_system_prompt=enable_system_prompt,
            system_prompt_source=system_prompt_source,
            sys_to_use=sys_to_use,
            n=debug_n,
        )

    print(
        f"[INFO] Config: benchmark={cfg['benchmark']} total={total} running={len(eval_prompts)}"
    )
    print(f"[INFO] Model: {cfg['model']}")
    print(f"[INFO] Pass mode: {cfg['pass_mode']}")
    if profiles_path:
        print(f"[INFO] Model profiles: {profiles_path}")
    print(f"[INFO] Prompting mode: {prompt_mode} -> interface={chosen_interface}")
    print(f"[INFO] System prompt enabled: {enable_system_prompt}")
    if enable_system_prompt and sys_to_use.strip():
        print(f"[INFO] System prompt source: {system_prompt_source}")
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
    outputs = llm.generate(model_prompts, sampling_params)

    # Build samples.jsonl
    # Use "solution" = benchmark_prompt + completion for robustness.
    #
    # IMPORTANT:
    #   - In chat mode, the *model input* prompt includes chat wrappers/tokens.
    #   - EvalPlus evaluation expects Python code starting from the benchmark prompt,
    #     so we always prefix with the original benchmark prompt here.
    # For pass@k (n>1), write multiple lines per task_id (one per candidate).
    rows: List[Dict[str, Any]] = []
    for i, out in enumerate(outputs):
        tid = task_ids[i]
        prompt = eval_prompts[i]
        for cand in out.outputs:
            rows.append(
                {
                    "task_id": tid,
                    "solution": prompt + cand.text,
                }
            )

    write_jsonl(samples_path, rows)

    n = int(scfg["n"])
    expected_lines = len(eval_prompts) * n
    print(f"[OK] Wrote samples.jsonl: {samples_path}")
    print(
        f"[INFO] Lines: {len(rows)} (expected {expected_lines} = tasks({len(eval_prompts)}) * n({n}))"
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
