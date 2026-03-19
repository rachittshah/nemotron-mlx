#!/usr/bin/env python3
"""Run Nemotron-3 Super 120B-A12B on M4 Pro 48GB via llama.cpp.

Implements the expert offloading strategy from FITTING_120B.md:
- MoE expert weights offloaded to CPU (-ot "exps=CPU")
- KV-cache quantized (q8_0 keys, q4_0 values)
- Flash attention enabled
- macOS GPU memory override

Usage:
    python run_120b.py --analyze          # memory analysis only
    python run_120b.py --server           # start OpenAI-compatible server
    python run_120b.py --prompt "..."     # single inference
    python run_120b.py --benchmark        # benchmark throughput
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

# --- Configuration ---

MODELS = {
    "iq2_xxs": {
        "repo": "bartowski/nvidia_Nemotron-3-Super-120B-A12B-GGUF",
        "pattern": "*IQ2_XXS*",
        "name": "Nemotron-3 Super 120B IQ2_XXS",
        "size_gb": 50.16,
        "quality": "Low (2-bit) — ~17% PPL degradation vs Q4",
    },
    "iq1_s": {
        "repo": "bartowski/nvidia_Nemotron-3-Super-120B-A12B-GGUF",
        "pattern": "*IQ1_S*",
        "name": "Nemotron-3 Super 120B IQ1_S",
        "size_gb": 46.38,
        "quality": "Extremely low (1-bit) — not recommended",
    },
    "iq2_xs": {
        "repo": "bartowski/nvidia_Nemotron-3-Super-120B-A12B-GGUF",
        "pattern": "*IQ2_XS*",
        "name": "Nemotron-3 Super 120B IQ2_XS",
        "size_gb": 52.04,
        "quality": "Low (2-bit) — slightly better than IQ2_XXS",
    },
    "q2_k": {
        "repo": "bartowski/nvidia_Nemotron-3-Super-120B-A12B-GGUF",
        "pattern": "*Q2_K-*",  # not Q2_K_L
        "name": "Nemotron-3 Super 120B Q2_K",
        "size_gb": 54.82,
        "quality": "Very low — surprisingly usable per bartowski",
    },
    "ud_iq2_m": {
        "repo": "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
        "pattern": "*UD-IQ2_M*",
        "name": "Nemotron-3 Super 120B UD-IQ2_M (Unsloth Dynamic)",
        "size_gb": 52.7,
        "quality": "Best at 2-bit — Unsloth keeps attention at higher precision",
    },
}

DEFAULT_QUANT = "iq2_xxs"

HARDWARE = {
    "chip": "Apple M4 Pro",
    "memory_gb": 48,
    "nvme_gbs": 7.4,
    "default_metal_gb": 39.3,  # recommendedMaxWorkingSetSize
    "override_metal_gb": 44.0,  # after iogpu.wired_limit_mb=45056
}

# MoE architecture constants
MOE_ACTIVE_PARAMS_B = 12.0
MOE_TOTAL_PARAMS_B = 120.6
MOE_EXPERTS = 128
MOE_ACTIVE_EXPERTS = 4  # per token


def check_llama_cpp():
    """Verify llama.cpp is installed."""
    for cmd in ["llama-server", "llama-cli"]:
        if not shutil.which(cmd):
            print(f"ERROR: {cmd} not found. Install: brew install llama.cpp")
            sys.exit(1)
    # Get version
    r = subprocess.run(["llama-cli", "--version"], capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "version:" in line:
            print(f"  llama.cpp {line.strip()}")
            break


def memory_analysis(quant_key: str = DEFAULT_QUANT):
    """Print detailed memory budget analysis."""
    model = MODELS[quant_key]
    vm = psutil.virtual_memory()

    model_gb = model["size_gb"]
    active_gb = MOE_ACTIVE_PARAMS_B * 0.5  # ~0.5 bytes/param at 2-bit effective
    shared_gb = model_gb * 0.07  # embeddings, norms, router ~7%
    working_set = active_gb + shared_gb
    cold_expert_gb = model_gb - working_set

    # Metal budget
    metal_gb = HARDWARE["override_metal_gb"]
    kv_cache_gb = 1.5  # 8K context, q8/q4
    compute_gb = 2.0
    metal_needed = working_set + kv_cache_gb + compute_gb

    print("=" * 65)
    print(f"  Memory Analysis: {model['name']}")
    print("=" * 65)
    print()
    print(f"  System RAM:              {HARDWARE['memory_gb']:.0f} GB")
    print(f"  Available now:           {vm.available / (1024**3):.1f} GB")
    print(f"  Metal GPU (default):     {HARDWARE['default_metal_gb']:.1f} GB")
    print(f"  Metal GPU (overridden):  {HARDWARE['override_metal_gb']:.1f} GB")
    print()
    print(f"  Model size ({quant_key}):  {model_gb:.1f} GB")
    print(f"  Quality:                 {model['quality']}")
    print()
    print("  MoE Decomposition:")
    print(f"    Active per token:      {active_gb:.1f} GB  ({MOE_ACTIVE_PARAMS_B:.0f}B of {MOE_TOTAL_PARAMS_B:.0f}B)")
    print(f"    Shared layers:         {shared_gb:.1f} GB  (embeddings, norms, router)")
    print(f"    Working set:           {working_set:.1f} GB  ← stays in RAM")
    print(f"    Cold experts:          {cold_expert_gb:.1f} GB  ← NVMe mmap ({HARDWARE['nvme_gbs']} GB/s)")
    print()
    print("  With Expert Offloading (-ot 'exps=CPU'):")
    print(f"    Metal GPU needs:       {metal_needed:.1f} GB  (working set + KV + compute)")
    print(f"    Metal GPU has:         {metal_gb:.1f} GB")
    metal_ok = metal_needed < metal_gb
    print(f"    Fits on Metal:         {'YES' if metal_ok else 'NO'} ({metal_gb - metal_needed:+.1f} GB)")
    print()

    overage = model_gb - HARDWARE["memory_gb"]
    if overage > 0:
        print(f"  Total model exceeds RAM: +{overage:.1f} GB")
        print(f"  → macOS will swap cold experts to NVMe")
        print(f"  → Expert routing is sparse ({MOE_ACTIVE_EXPERTS}/{MOE_EXPERTS} per token)")
        print(f"  → Hot experts stay resident, cold ones paged on demand")
        expert_load_ms = (cold_expert_gb / MOE_EXPERTS) / HARDWARE["nvme_gbs"] * 1000
        print(f"  → Single expert page-in: ~{expert_load_ms:.1f}ms from NVMe")
    else:
        print(f"  Model fits in RAM with {-overage:.1f} GB headroom")

    print()
    print(f"  Expected throughput:     ~5-15 tok/s (generation, 8K context)")
    print(f"  Expected TTFT:           ~2-8s (depending on prompt length)")
    print("=" * 65)
    return metal_ok


def find_model_file(quant_key: str) -> Path | None:
    """Find downloaded GGUF file."""
    model = MODELS[quant_key]
    local_dir = Path(f"models/{quant_key}")
    if local_dir.exists():
        for f in sorted(local_dir.rglob("*.gguf")):
            return f

    # Check HF cache
    cache_dir = Path.home() / ".cache/huggingface/hub"
    repo_slug = model["repo"].replace("/", "--")
    for d in cache_dir.glob(f"models--{repo_slug}*/snapshots/*/"):
        for f in sorted(d.rglob("*.gguf")):
            return f
    return None


def download_model(quant_key: str) -> Path:
    """Download GGUF model from HuggingFace."""
    model = MODELS[quant_key]
    local_dir = Path(f"models/{quant_key}")

    existing = find_model_file(quant_key)
    if existing:
        print(f"Model found: {existing}")
        return existing

    print(f"Downloading {model['name']} (~{model['size_gb']} GB)...")
    print(f"  Repo: {model['repo']}")
    print(f"  Pattern: {model['pattern']}")
    print("  This will take a while...")

    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["hf", "download", model["repo"],
         "--include", model["pattern"],
         "--local-dir", str(local_dir)],
        check=True,
    )

    result = find_model_file(quant_key)
    if not result:
        print("ERROR: Download completed but no .gguf file found")
        sys.exit(1)
    return result


def set_gpu_memory():
    """Override macOS GPU memory limit."""
    try:
        r = subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"],
            capture_output=True, text=True,
        )
        current = int(r.stdout.strip()) if r.returncode == 0 else 0
    except (ValueError, OSError):
        current = 0

    if current >= 44000:
        print(f"  GPU memory already set to {current} MB")
        return

    print("  Setting GPU memory to 44 GB (requires sudo)...")
    subprocess.run(
        ["sudo", "sysctl", "iogpu.wired_limit_mb=45056"],
        check=False,
    )


def build_llama_args(model_path: Path, mode: str = "server",
                     context: int = 8192, port: int = 8080,
                     prompt: str = "", max_tokens: int = 512) -> list[str]:
    """Build llama.cpp command-line arguments."""
    cmd = "llama-server" if mode == "server" else "llama-cli"

    args = [
        cmd,
        "-m", str(model_path),
        "-ngl", "99",                    # offload all layers to GPU
        "-ot", "exps=CPU",               # MoE experts to CPU
        "--cache-type-k", "q8_0",        # KV-cache: keys at Q8
        "--cache-type-v", "q4_0",        # KV-cache: values at Q4
        "-fa",                           # flash attention
        "-c", str(context),              # context length
        "-b", "4096",                    # batch size
        "-ub", "4096",                   # ubatch size
    ]

    if mode == "server":
        args += ["--host", "0.0.0.0", "--port", str(port)]
    elif mode == "cli":
        args += ["-n", str(max_tokens), "-p", prompt, "--no-display-prompt"]

    return args


def run_server(model_path: Path, context: int = 8192, port: int = 8080):
    """Start llama.cpp server."""
    args = build_llama_args(model_path, mode="server", context=context, port=port)

    print()
    print(f"Starting llama-server on port {port}...")
    print(f"  Model: {model_path.name}")
    print(f"  Context: {context}")
    print(f"  Expert offload: CPU")
    print(f"  KV-cache: k=q8_0, v=q4_0")
    print()
    print(f"  API: http://localhost:{port}/v1")
    print(f"  Chat: http://localhost:{port}")
    print()
    print("  Press Ctrl+C to stop")
    print()

    os.execvp(args[0], args)


def run_single(model_path: Path, prompt: str, max_tokens: int = 512,
               context: int = 8192) -> dict:
    """Run single inference and capture metrics."""
    args = build_llama_args(
        model_path, mode="cli", context=context,
        prompt=prompt, max_tokens=max_tokens,
    )

    print(f"Prompt: {prompt[:80]}...")
    print()

    t0 = time.perf_counter()
    result = subprocess.run(args, capture_output=True, text=True, timeout=300)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"ERROR (exit {result.returncode}):")
        print(result.stderr[-500:] if result.stderr else "no stderr")
        return {}

    response = result.stdout.strip()

    # Parse timing from stderr
    tps = 0.0
    ttft = 0.0
    for line in result.stderr.splitlines():
        if "eval time" in line and "token" in line:
            # llama_perf_context_print: eval time = X ms / Y tokens (Z ms per token, W tokens per second)
            parts = line.split(",")
            for p in parts:
                if "tokens per second" in p:
                    try:
                        tps = float(p.strip().split()[0])
                    except (ValueError, IndexError):
                        pass
        if "prompt eval time" in line:
            parts = line.split("=")
            if len(parts) > 1:
                try:
                    ttft = float(parts[1].strip().split()[0]) / 1000  # ms to s
                except (ValueError, IndexError):
                    pass

    output_tokens = len(response.split())  # rough word count as proxy

    print(response[:500])
    print()
    print("-" * 60)
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  tok/s:     {tps:.1f}")
    print(f"  TTFT:      {ttft:.1f}s")

    return {
        "response": response,
        "elapsed_s": round(elapsed, 2),
        "tokens_per_sec": round(tps, 1),
        "ttft_s": round(ttft, 1),
    }


def run_benchmark(model_path: Path, context: int = 8192) -> dict:
    """Run benchmark suite."""
    prompts = [
        "Explain mixture of experts architecture and why it enables efficient inference on memory-constrained devices.",
        "Write a Python function that implements binary search with detailed comments.",
        "Compare the M4 Pro and M4 Max chips for machine learning workloads. Be specific about memory bandwidth.",
    ]

    print(f"Benchmarking {model_path.name}...")
    print(f"  Context: {context}")
    print(f"  Prompts: {len(prompts)}")
    print()

    results = []
    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}]")
        r = run_single(model_path, prompt, max_tokens=256, context=context)
        if r:
            results.append(r)
        print()

    if not results:
        print("All prompts failed.")
        return {}

    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    avg_ttft = sum(r["ttft_s"] for r in results) / len(results)

    summary = {
        "model": model_path.name,
        "quant": model_path.stem,
        "context": context,
        "avg_tokens_per_sec": round(avg_tps, 1),
        "avg_ttft_s": round(avg_ttft, 1),
        "num_prompts": len(results),
        "per_prompt": results,
    }

    print("=" * 60)
    print(f"  Model:     {model_path.name}")
    print(f"  Avg tok/s: {avg_tps:.1f}")
    print(f"  Avg TTFT:  {avg_ttft:.1f}s")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Nemotron-3 Super 120B on 48GB Apple Silicon"
    )
    parser.add_argument(
        "--quant", "-q", choices=list(MODELS.keys()),
        default=DEFAULT_QUANT, help="Quantization variant",
    )
    parser.add_argument("--analyze", "-a", action="store_true",
                        help="Memory analysis only (no model download)")
    parser.add_argument("--server", "-s", action="store_true",
                        help="Start OpenAI-compatible server")
    parser.add_argument("--prompt", "-p", type=str,
                        help="Single inference prompt")
    parser.add_argument("--benchmark", "-b", action="store_true",
                        help="Run benchmark suite")
    parser.add_argument("--context", "-c", type=int, default=8192,
                        help="Context length (default: 8192)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Server port (default: 8080)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output-json", type=str,
                        help="Save results to JSON")
    parser.add_argument("--skip-gpu-override", action="store_true",
                        help="Don't try to set GPU memory limit")
    args = parser.parse_args()

    print("Nemotron-3 Super 120B-A12B on Apple Silicon")
    print()

    # Analysis mode
    if args.analyze:
        for qk in MODELS:
            memory_analysis(qk)
            print()
        return

    # Check deps
    check_llama_cpp()

    # Memory analysis for selected quant
    memory_analysis(args.quant)

    if not args.skip_gpu_override:
        set_gpu_memory()

    # Download model
    model_path = download_model(args.quant)

    if args.server:
        run_server(model_path, context=args.context, port=args.port)
    elif args.benchmark:
        summary = run_benchmark(model_path, context=args.context)
        if args.output_json and summary:
            Path(args.output_json).write_text(json.dumps(summary, indent=2))
            print(f"\nSaved to {args.output_json}")
    elif args.prompt:
        result = run_single(model_path, args.prompt,
                            max_tokens=args.max_tokens, context=args.context)
        if args.output_json and result:
            Path(args.output_json).write_text(json.dumps(result, indent=2))
    else:
        print()
        print("No action specified. Use --server, --prompt, or --benchmark")
        print("  --analyze: memory analysis for all quants")
        print("  --server: start OpenAI-compatible API server")
        print("  --prompt 'text': single inference")
        print("  --benchmark: throughput benchmark")


if __name__ == "__main__":
    main()
