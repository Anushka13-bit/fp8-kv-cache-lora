"""Evaluate one saved LoRA adapter under one KV-cache configuration."""
import argparse
import json
import math
import os

from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import config as cfg

PROMPT_TEMPLATE = (
    "You are a SQL expert. Given the database schema and a question, "
    "write the SQL query that answers it.\n\n"
    "### Schema:\n{context}\n\n"
    "### Question:\n{question}\n\n"
    "### SQL:\n"
)


def run_sql_generation(llm, lora_request, out_path):
    with open(f"{cfg.RESULTS_DIR}/held_out_test.jsonl") as f:
        test_rows = [json.loads(line) for line in f]

    prompts = [
        PROMPT_TEMPLATE.format(context=r["context"], question=r["question"])
        for r in test_rows
    ]

    params = SamplingParams(
        temperature=cfg.GEN_TEMPERATURE,
        max_tokens=cfg.MAX_NEW_TOKENS,
        seed=cfg.GEN_SEED,
        stop=["\n\n", "###"],
    )

    outputs = llm.generate(prompts, params, lora_request=lora_request)

    with open(out_path, "w") as f:
        for row, out in zip(test_rows, outputs):
            f.write(json.dumps({
                "question": row["question"],
                "context": row["context"],
                "gold_sql": row["answer"],
                "pred_sql": out.outputs[0].text.strip(),
            }) + "\n")

    print(f"Wrote {len(test_rows)} SQL predictions to {out_path}")


def run_perplexity(llm, lora_request, out_path):
    ds = load_dataset(cfg.PPL_DATASET, cfg.PPL_CONFIG, split=cfg.PPL_SPLIT)
    texts = [t for t in ds["text"] if len(t.strip()) > 200][:cfg.PPL_NUM_DOCS]

    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
    )
    outputs = llm.generate(texts, params, lora_request=lora_request)

    total_logprob = 0.0
    total_tokens = 0

    for out in outputs:
        prompt_ids = out.prompt_token_ids
        for idx, tok_logprobs in enumerate(out.prompt_logprobs or []):
            if tok_logprobs is None or idx >= len(prompt_ids):
                continue
            actual = tok_logprobs.get(prompt_ids[idx])
            if actual is not None:
                total_logprob += actual.logprob
                total_tokens += 1

    if total_tokens == 0:
        raise RuntimeError(
            "No prompt logprobs returned; do not trust the perplexity result."
        )

    perplexity = math.exp(-total_logprob / total_tokens)
    with open(out_path, "w") as f:
        json.dump({"perplexity": perplexity, "num_tokens": total_tokens}, f, indent=2)

    print(f"Perplexity: {perplexity:.3f} -> {out_path}")


def main(kv_cache_dtype, calculate_kv_scales):
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    llm = LLM(
        model=cfg.BASE_MODEL,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=cfg.LORA_R,
        kv_cache_dtype=kv_cache_dtype,
        calculate_kv_scales=calculate_kv_scales,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
    )

    lora_request = LoRARequest(cfg.ADAPTER_NAME, 1, cfg.ADAPTER_DIR)

    tag = kv_cache_dtype.replace("_", "-")
    if kv_cache_dtype.startswith("fp8") and calculate_kv_scales:
        tag += "-calibrated"

    run_sql_generation(
        llm, lora_request, f"{cfg.RESULTS_DIR}/sql_preds_{tag}.jsonl"
    )
    run_perplexity(
        llm, lora_request, f"{cfg.RESULTS_DIR}/perplexity_{tag}.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kv_cache_dtype",
        required=True,
        choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
    )
    parser.add_argument(
        "--calculate_kv_scales",
        action="store_true",
    )
    args = parser.parse_args()
    main(args.kv_cache_dtype, args.calculate_kv_scales)
