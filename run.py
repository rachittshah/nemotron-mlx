#!/usr/bin/env python3
"""Nemotron inference on Apple Silicon via MLX.

Supports Nemotron-3 Nano (30B-A3B), Nemotron-Super-49B, and
experimental 120B-A12B with mmap offloading.

Usage:
    python run.py                          # default: nano, interactive
    python run.py --model super-49b        # 49B 4-bit
    python run.py --model nano --benchmark # benchmark mode
    python run.py --prompt "Your question"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from config import MODELS, HARDWARE, InferenceConfig


def get_memory_usage_gb() -> float:
    """Current MLX Metal memory usage in GB."""
    try:
        return mx.get_active_memory() / (1024 ** 3)
    except AttributeError:
        try:
            return mx.metal.get_active_memory() / (1024 ** 3)
        except Exception:
            return 0.0


def get_peak_memory_gb() -> float:
    """Peak MLX Metal memory usage in GB."""
    try:
        return mx.get_peak_memory() / (1024 ** 3)
    except AttributeError:
        try:
            return mx.metal.get_peak_memory() / (1024 ** 3)
        except Exception:
            return 0.0


def reset_peak_memory():
    """Reset peak memory counter."""
    try:
        mx.reset_peak_memory()
    except AttributeError:
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass


def load_model(model_key: str):
    """Load model and tokenizer via mlx-lm."""
    from mlx_lm import load

    model_info = MODELS[model_key]
    hf_id = model_info["hf_id"]

    print(f"Loading {model_info['name']}...")
    print(f"  HuggingFace: {hf_id}")
    print(f"  Architecture: {model_info['arch']}")
    print(f"  Quantization: {model_info['quant']}")
    print(f"  Est. memory: {model_info['est_memory_gb']:.1f} GB")
    print(f"  Available: {HARDWARE['usable_memory_gb']:.1f} GB")
    print()

    t0 = time.perf_counter()
    model, tokenizer = load(hf_id, tokenizer_config={"trust_remote_code": True})
    load_time = time.perf_counter() - t0

    mem = get_memory_usage_gb()
    print(f"  Loaded in {load_time:.1f}s | Memory: {mem:.1f} GB")
    print()

    return model, tokenizer, load_time


def generate_response(model, tokenizer, prompt: str, cfg: InferenceConfig) -> dict:
    """Generate a response and return metrics."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = prompt

    reset_peak_memory()

    sampler = make_sampler(temp=cfg.temperature, top_p=cfg.top_p)

    t0 = time.perf_counter()

    response = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=cfg.max_tokens,
        sampler=sampler,
        verbose=False,
    )

    total_time = time.perf_counter() - t0

    # Count tokens
    output_tokens = len(tokenizer.encode(response))
    input_tokens = len(tokenizer.encode(formatted))

    tokens_per_sec = output_tokens / total_time if total_time > 0 else 0
    peak_mem = get_peak_memory_gb()

    return {
        "response": response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_time_s": round(total_time, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_memory_gb": round(peak_mem, 2),
    }


def run_benchmark(model, tokenizer, cfg: InferenceConfig) -> dict:
    """Run benchmark suite and return aggregated metrics."""
    prompts = [
        "Explain the key innovations in Apple's M4 chip architecture.",
        "Write a Python function to find the longest common subsequence of two strings.",
        "Compare transformer and mamba architectures for language modeling. Be specific about computational complexity.",
        "What are the main challenges in deploying large language models on edge devices?",
        "Describe the memory hierarchy of modern Apple Silicon chips and how it benefits ML inference.",
    ][: cfg.benchmark_prompts]

    print(f"Running benchmark with {len(prompts)} prompts...")
    print(f"  Max tokens per response: {cfg.max_tokens}")
    print()

    results = []
    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt[:60]}...")
        metrics = generate_response(model, tokenizer, prompt, cfg)
        results.append(metrics)
        print(
            f"         {metrics['output_tokens']} tokens | "
            f"{metrics['tokens_per_sec']} tok/s | "
            f"{metrics['total_time_s']}s | "
            f"peak {metrics['peak_memory_gb']} GB"
        )

    # Aggregate
    total_tokens = sum(r["output_tokens"] for r in results)
    total_time = sum(r["total_time_s"] for r in results)
    avg_tps = round(total_tokens / total_time, 1) if total_time > 0 else 0
    peak_mem = max(r["peak_memory_gb"] for r in results)

    summary = {
        "model": MODELS[cfg.model_key]["name"],
        "model_hf_id": MODELS[cfg.model_key]["hf_id"],
        "hardware": HARDWARE,
        "config": {
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "num_prompts": len(prompts),
        },
        "results": {
            "avg_tokens_per_sec": avg_tps,
            "total_output_tokens": total_tokens,
            "total_time_s": round(total_time, 2),
            "peak_memory_gb": peak_mem,
            "per_prompt": results,
        },
    }

    print()
    print("=" * 60)
    print(f"  Model:          {summary['model']}")
    print(f"  Avg tok/s:      {avg_tps}")
    print(f"  Total tokens:   {total_tokens}")
    print(f"  Total time:     {total_time:.1f}s")
    print(f"  Peak memory:    {peak_mem:.1f} GB")
    print(f"  Bandwidth util: {_bandwidth_util(avg_tps, cfg.model_key):.0f}%")
    print("=" * 60)

    return summary


def _bandwidth_util(tps: float, model_key: str) -> float:
    """Estimate memory bandwidth utilization percentage."""
    model_info = MODELS[model_key]
    quant = model_info["quant"]
    # Rough bytes per param at each quant level
    bytes_per_param = 0.5 if "4-bit" in quant else 1.0 if "8-bit" in quant else 2.0
    # Active params determine bytes moved per token
    active_str = model_info["active_params"]
    active_b = float(active_str.replace("B", ""))
    bytes_per_token = active_b * 1e9 * bytes_per_param
    actual_bandwidth = tps * bytes_per_token
    theoretical = HARDWARE["memory_bandwidth_gbs"] * 1e9
    return (actual_bandwidth / theoretical) * 100


def main():
    parser = argparse.ArgumentParser(description="Nemotron MLX Inference")
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()),
        default="nano",
        help="Model variant (default: nano)",
    )
    parser.add_argument("--prompt", "-p", type=str, help="Input prompt")
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Max output tokens"
    )
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run benchmark")
    parser.add_argument(
        "--benchmark-prompts", type=int, default=5, help="Number of benchmark prompts"
    )
    parser.add_argument("--output-json", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    cfg = InferenceConfig(
        model_key=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        benchmark=args.benchmark,
        benchmark_prompts=args.benchmark_prompts,
    )
    if args.prompt:
        cfg.prompt = args.prompt

    model_info = MODELS[cfg.model_key]
    if not model_info["fits_48gb"]:
        print(f"WARNING: {model_info['name']} estimated at {model_info['est_memory_gb']}GB")
        print(f"  Available memory: {HARDWARE['usable_memory_gb']}GB")
        print(f"  This model may require mmap/swap offloading.")
        print(f"  Proceeding anyway (mmap enabled)...")
        print()

    model, tokenizer, load_time = load_model(cfg.model_key)

    if cfg.benchmark:
        summary = run_benchmark(model, tokenizer, cfg)
        summary["load_time_s"] = round(load_time, 2)
        if args.output_json:
            Path(args.output_json).write_text(
                json.dumps(summary, indent=2, default=str)
            )
            print(f"\nResults saved to {args.output_json}")
    else:
        print(f"Prompt: {cfg.prompt}")
        print("-" * 60)
        metrics = generate_response(model, tokenizer, cfg.prompt, cfg)
        print(metrics["response"])
        print("-" * 60)
        print(
            f"{metrics['output_tokens']} tokens | "
            f"{metrics['tokens_per_sec']} tok/s | "
            f"{metrics['total_time_s']}s | "
            f"peak {metrics['peak_memory_gb']} GB"
        )


if __name__ == "__main__":
    main()
