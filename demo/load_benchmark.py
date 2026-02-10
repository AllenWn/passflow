import argparse
import json
from typing import Dict, Any, Iterable


def iter_items(d: Dict[str, Any]) -> Iterable[tuple[str, Any]]:
    for k in sorted(d.keys()):
        yield k, d[k]


def load_problems(dataset: str) -> Dict[str, Any]:
    dataset = dataset.lower()
    if dataset in ("humaneval", "humaneval+", "he", "he+"):
        from evalplus.data import get_human_eval_plus

        return get_human_eval_plus()
    elif dataset in ("mbpp", "mbpp+", "mbppplus"):
        from evalplus.data import get_mbpp_plus

        return get_mbpp_plus()
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use humaneval or mbpp.")


def main():
    ap = argparse.ArgumentParser(
        description="Load EvalPlus benchmark problems (data only)."
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="humaneval",
        help="humaneval or mbpp (EvalPlus '+' versions)",
    )
    ap.add_argument(
        "--limit", type=int, default=5, help="How many problems to preview/print."
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional: write a preview JSONL to this path.",
    )
    args = ap.parse_args()

    problems = load_problems(args.dataset)
    total = len(problems)
    print(f"[OK] Loaded dataset='{args.dataset}' with {total} problems.")

    # Preview first N
    preview = []
    for idx, (task_id, prob) in enumerate(iter_items(problems)):
        if idx >= args.limit:
            break

        prompt = prob.get("prompt", "")
        entry_point = prob.get("entry_point", None)
        canonical_solution = prob.get("canonical_solution", None)

        print("\n" + "=" * 80)
        print(f"#{idx} task_id: {task_id}")
        print(f"entry_point: {entry_point}")
        print(f"prompt_chars: {len(prompt)}")
        if canonical_solution is not None:
            print(f"has_canonical_solution: True (len={len(canonical_solution)})")
        else:
            print("has_canonical_solution: False")
        print("-" * 80)
        print(prompt[:400] + ("..." if len(prompt) > 400 else ""))

        preview.append(
            {
                "task_id": task_id,
                "entry_point": entry_point,
                "prompt": prompt,
            }
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in preview:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n[OK] Wrote preview JSONL: {args.out} ({len(preview)} lines)")


if __name__ == "__main__":
    main()


# python load_benchmark.py --dataset humaneval --limit 5 --out tasks_preview.jsonl
