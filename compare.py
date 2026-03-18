#!/usr/bin/env python3
"""Compare benchmark results across model variants."""

import json
import sys
from pathlib import Path


def load_results(results_dir: str = "results") -> list[dict]:
    """Load all benchmark JSON files."""
    results = []
    for p in sorted(Path(results_dir).glob("*.json")):
        with open(p) as f:
            data = json.load(f)
            data["_file"] = p.name
            results.append(data)
    return results


def print_comparison(results: list[dict]):
    """Print comparison table."""
    if not results:
        print("No benchmark results found in results/")
        return

    print("=" * 80)
    print("Nemotron MLX Benchmark Comparison — Apple M4 Pro 48GB")
    print("=" * 80)
    print()

    # Header
    print(f"{'Model':<35} {'Avg tok/s':>10} {'Peak mem':>10} {'Load time':>10}")
    print("-" * 70)

    for r in results:
        model = r.get("model", "unknown")
        avg_tps = r.get("results", {}).get("avg_tokens_per_sec", 0)
        peak_mem = r.get("results", {}).get("peak_memory_gb", 0)
        load_time = r.get("load_time_s", 0)

        print(f"{model:<35} {avg_tps:>9.1f} {peak_mem:>8.1f} GB {load_time:>8.1f}s")

    print()

    # Recommendation
    best = max(results, key=lambda r: r.get("results", {}).get("avg_tokens_per_sec", 0))
    most_capable = max(results, key=lambda r: float(
        r.get("model_hf_id", "0B").split("-")[-1].replace("bit", "").replace("B", "0")
        if "49B" not in r.get("model", "") else "490"
    ))

    print(f"Fastest:       {best['model']} ({best['results']['avg_tokens_per_sec']} tok/s)")
    print(f"Most capable:  {most_capable['model']}")
    print()


if __name__ == "__main__":
    results = load_results()
    print_comparison(results)
