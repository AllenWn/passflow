import argparse
import json
import os
from typing import Dict, Any, List, Tuple

from vllm import LLM, SamplingParams


def load_humaneval_plus() -> Dict[str, Any]:
    # EvalPlus data loader (HumanEval+)
    from evalplus.data import get_human_eval_plus

    return get_human_eval_plus()


def sorted_items(d: Dict[str, Any]) -> List[Tuple[str, Any]]:
    # stable order for reproducibility
    return [(k, d[k]) for k in sorted(d.keys())]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Generate HumanEval+ samples with vLLM (Pass@1) and write EvalPlus-compatible JSONL."
    )
    ap.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="HF model id or local path.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many tasks to run (default: 5 for sanity check). Use 164 for full HumanEval.",
    )
    ap.add_argument(
        "--outdir", type=str, default="runs/pass1_humaneval", help="Output directory."
    )
    ap.add_argument(
        "--max_tokens", type=int, default=512, help="Max new tokens to generate."
    )
    ap.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Max model length for vLLM engine.",
    )
    ap.add_argument(
        "--gpu_mem_util", type=float, default=0.90, help="vLLM gpu_memory_utilization."
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling (deterministic with temperature=0).",
    )
    ap.add_argument("--tp", type=int, default=1, help="Tensor parallel size.")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    samples_path = os.path.join(args.outdir, "samples.jsonl")
    config_path = os.path.join(args.outdir, "run_config.json")

    # 1) Load benchmark problems
    problems = load_humaneval_plus()
    items = sorted_items(problems)[: max(args.limit, 0)]
    if not items:
        raise RuntimeError(
            "No problems loaded. Check EvalPlus installation and dataset access."
        )

    task_ids = [tid for tid, _ in items]
    prompts = [prob["prompt"] for _, prob in items]

    print(
        f"[INFO] Loaded HumanEval+ problems: total={len(problems)}, running={len(prompts)}"
    )
    print(f"[INFO] First task_id: {task_ids[0]}")

    # 2) Initialize vLLM engine
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        trust_remote_code=False,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
    )

    # 3) Pass@1 sampling params (greedy/deterministic)
    # stop tokens help avoid endless markdown/code fences
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_tokens,
        stop=["```", "\n\n\n"],
        seed=args.seed,
    )

    # 4) Generate
    print("[INFO] Generating with vLLM (Pass@1: n=1, temperature=0.0)...")
    outputs = llm.generate(prompts, sampling_params)

    # 5) Build EvalPlus-compatible samples.jsonl
    # Use "solution" (prompt + completion) to be robust to markdown/noisy outputs.
    rows: List[Dict[str, Any]] = []
    for i, out in enumerate(outputs):
        # out.outputs is a list of candidate completions (length n=1 here)
        completion_text = out.outputs[0].text

        rows.append(
            {
                "task_id": task_ids[i],
                "solution": prompts[i] + completion_text,
            }
        )

    write_jsonl(samples_path, rows)

    # Save run config for reproducibility
    run_cfg = {
        "dataset": "humaneval",
        "dataset_version": "humaneval+ (via evalplus.data.get_human_eval_plus)",
        "model": args.model,
        "limit": args.limit,
        "sampling": {
            "mode": "pass@1",
            "n": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": args.max_tokens,
            "stop": ["```", "\\n\\n\\n"],
            "seed": args.seed,
        },
        "vllm_engine": {
            "tensor_parallel_size": args.tp,
            "dtype": "bfloat16",
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_mem_util,
            "trust_remote_code": False,
        },
        "output": {
            "samples_jsonl": samples_path,
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote samples: {samples_path} ({len(rows)} lines)")
    print(f"[OK] Wrote run config: {config_path}")
    print("\nNext step (evaluation):")
    print(f"  evalplus.evaluate humaneval --samples {samples_path}")


if __name__ == "__main__":
    main()
