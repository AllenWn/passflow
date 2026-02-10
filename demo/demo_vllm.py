import os
from vllm import LLM, SamplingParams


def build_sampling_params(
    mode: str = "pass1",
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    n: int = 1,
    seed: int = 0,
):
    """
    mode:
      - "pass1": greedy / deterministic (good for pass@1)
      - "passk": sampling (good for pass@k)
    """
    if mode == "pass1":
        temperature = 0.0
        top_p = 1.0
        top_k = -1
        n = 1

    # stop tokens: prevent endless markdown/code fences or extra chatter
    stop = ["```", "\n\n\n"]

    return SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop=stop,
        seed=seed,
    )


def main():
    # Pick a small coder model for sanity check (swap to your lab model later)
    model = os.environ.get("MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")

    # vLLM engine config (commonly tuned knobs)
    llm = LLM(
        model=model,
        tensor_parallel_size=int(os.environ.get("TP", "1")),
        dtype="bfloat16",  # A100 friendly
        trust_remote_code=False,
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "8192")),
        gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.90")),
    )

    prompts = [
        "Write a Python function add(a, b) that returns a + b. "
        "Return exactly one Python code block and nothing else."
    ]

    # ---- Pass@1 style (greedy) ----
    params_pass1 = build_sampling_params(mode="pass1", max_tokens=128, seed=0)
    outs = llm.generate(prompts, params_pass1)
    print("\n=== PASS@1 (greedy) ===")
    print(outs[0].outputs[0].text.strip())

    # ---- Pass@k style (sampling) ----
    # Generate multiple candidates for the same prompt
    params_passk = build_sampling_params(
        mode="passk",
        n=5,  # number of samples per prompt
        temperature=0.8,
        top_p=0.95,
        max_tokens=128,
        seed=1,
    )
    outs_k = llm.generate(prompts, params_passk)

    print("\n=== PASS@K (sampling, n=5) ===")
    for j, cand in enumerate(outs_k[0].outputs):
        print(f"\n--- candidate {j} ---")
        print(cand.text.strip())


if __name__ == "__main__":
    main()
