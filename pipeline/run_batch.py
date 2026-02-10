#!/usr/bin/env python3
"""
Batch automation runner (minimal requirement):

- Read batch settings from config.py:
  - BATCH_MODELS
  - BATCH_BENCHMARKS
  - BATCH_KS
  - BATCH_CSV_PATH

- Reuse the exact single-run flow for each job:
  run_generate.py (with overrides) -> evalplus.evaluate -> write eval.log

- Append a summary row to CSV for each job.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import subprocess
import sys
from typing import Any, Dict, List


PASS_RE = re.compile(r"pass@(\d+):\s*([0-9]*\.?[0-9]+)")


def load_cfg_module(path: str):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("user_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import config: {path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parse_eval_stdout(text: str) -> Dict[str, float]:
    section = None  # base / plus
    out: Dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if "(base tests)" in line:
            section = "base"
            continue
        if "(base + extra tests)" in line:
            section = "plus"
            continue
        m = PASS_RE.search(line)
        if m and section:
            k = int(m.group(1))
            v = float(m.group(2))
            out[f"{section}_pass@{k}"] = v
    return out


def normalize_bench(b: str) -> str:
    b = (b or "").lower()
    if b in ("humaneval", "humaneval+"):
        return "humaneval"
    if b in ("mbpp", "mbpp+"):
        return "mbpp"
    return b


def run_cmd(cmd: List[str], cwd: str) -> tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return p.returncode, p.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch runner for evalPlus pipeline.")
    ap.add_argument(
        "--config",
        type=str,
        default="config.py",
        help="Path to config.py (same as single-run).",
    )
    args = ap.parse_args()

    config_path = os.path.abspath(args.config)
    repo_root = os.path.dirname(config_path)
    cfg_mod = load_cfg_module(config_path)

    # batch settings (top-level constants)
    models = list(getattr(cfg_mod, "BATCH_MODELS", []))
    benchmarks = list(getattr(cfg_mod, "BATCH_BENCHMARKS", []))
    ks = list(getattr(cfg_mod, "BATCH_KS", []))
    csv_path = str(getattr(cfg_mod, "BATCH_CSV_PATH", "batch_results/summary.csv"))

    # single-run defaults (reuse config parameters)
    limit = int(getattr(cfg_mod, "LIMIT"))
    seed = int(getattr(cfg_mod, "SEED"))
    pass1 = dict(
        n=int(getattr(cfg_mod, "PASS1_N")),
        temperature=float(getattr(cfg_mod, "PASS1_TEMPERATURE")),
        top_p=float(getattr(cfg_mod, "PASS1_TOP_P")),
        top_k=int(getattr(cfg_mod, "PASS1_TOP_K")),
    )
    passk_defaults = dict(
        temperature=float(getattr(cfg_mod, "PASSK_TEMPERATURE")),
        top_p=float(getattr(cfg_mod, "PASSK_TOP_P")),
        top_k=int(getattr(cfg_mod, "PASSK_TOP_K")),
    )

    if not models or not benchmarks or not ks:
        raise SystemExit(
            "Empty batch settings: BATCH_MODELS / BATCH_BENCHMARKS / BATCH_KS"
        )

    # CSV path: relative to repo root if not absolute
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(repo_root, csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    csv_exists = os.path.exists(csv_path)
    # Keep CSV minimal (as requested): one row per (model, benchmark, k).
    # "result" prefers EvalPlus' stricter (+extra) score if available; otherwise falls back to base score.
    fieldnames = ["model", "benchmark", "pass@k", "result"]

    total = len(models) * len(benchmarks) * len(ks)
    idx = 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            w.writeheader()

        for bench in benchmarks:
            for model in models:
                for k in ks:
                    idx += 1
                    k_int = int(k)
                    if k_int <= 1:
                        pass_mode = "pass1"
                        sampling = dict(pass1)
                    else:
                        pass_mode = "passk"
                        sampling = dict(passk_defaults)
                        sampling["n"] = k_int  # pass@k -> n=k

                    print(
                        f"\n[JOB {idx}/{total}] bench={bench} model={model} k={k_int} ({pass_mode})"
                    )

                    # 1) dry-run to get outdir
                    dry_cmd = [
                        sys.executable,
                        "run_generate.py",
                        "--config",
                        config_path,
                        "--dry_run",
                        "--override_benchmark",
                        bench,
                        "--override_model",
                        model,
                        "--override_pass_mode",
                        pass_mode,
                        "--override_limit",
                        str(limit),
                        "--override_sampling_n",
                        str(sampling["n"]),
                        "--override_sampling_temperature",
                        str(sampling["temperature"]),
                        "--override_sampling_top_p",
                        str(sampling["top_p"]),
                        "--override_sampling_top_k",
                        str(sampling["top_k"]),
                    ]
                    rc, out = run_cmd(dry_cmd, cwd=repo_root)
                    if rc != 0:
                        print(out)
                        raise SystemExit(rc)

                    m = re.search(r"^\[INFO\] Output dir:\s*(.+)$", out, flags=re.M)
                    if not m:
                        raise SystemExit(
                            "Failed to parse [INFO] Output dir from dry_run output"
                        )
                    outdir = m.group(1).strip()

                    samples_path = os.path.join(outdir, "samples.jsonl")
                    eval_log = os.path.join(outdir, "eval.log")

                    # 2) generation (same args without --dry_run)
                    gen_cmd = [x for x in dry_cmd if x != "--dry_run"]
                    rc, gen_out = run_cmd(gen_cmd, cwd=repo_root)
                    if rc != 0:
                        print(gen_out)
                        raise SystemExit(rc)

                    # 3) evaluation + write eval.log
                    eval_cmd = [
                        "evalplus.evaluate",
                        normalize_bench(bench),
                        "--samples",
                        samples_path,
                    ]
                    rc, eval_out = run_cmd(eval_cmd, cwd=repo_root)
                    os.makedirs(outdir, exist_ok=True)
                    with open(eval_log, "w", encoding="utf-8") as lf:
                        lf.write(eval_out)
                    print(eval_out)
                    if rc != 0:
                        raise SystemExit(rc)

                    metrics = parse_eval_stdout(eval_out)
                    # EvalPlus prints pass@{1,10,100} where k<=n. We record the k we ran.
                    result = metrics.get(f"plus_pass@{k_int}")
                    if result is None:
                        result = metrics.get(f"base_pass@{k_int}")

                    row: Dict[str, Any] = {
                        "model": model,
                        "benchmark": bench,
                        "pass@k": k_int,
                        "result": "" if result is None else result,
                    }
                    w.writerow(row)
                    f.flush()

    print(f"\n[DONE] Batch CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
