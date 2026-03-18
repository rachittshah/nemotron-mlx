# Nemotron-3 on Apple Silicon via MLX

Efficient inference of NVIDIA's Nemotron-3 model family on Apple Silicon (M4 Pro, 48GB) using MLX + LLM-in-a-Flash techniques.

## Models

| Model | Total Params | Active Params | Quant | Memory | Fits 48GB? | Expected tok/s |
|-------|-------------|---------------|-------|--------|-----------|----------------|
| Nemotron-3 Nano 30B-A3B | 30B | 3B (MoE) | 8-bit | ~18 GB | Yes | ~83 |
| Nemotron-3 Nano 30B-A3B | 30B | 3B (MoE) | 4-bit | ~10 GB | Yes | ~100+ |
| Nemotron-Super 49B v1 | 49B | 49B | 4-bit | ~27 GB | Yes | ~15-25 |
| Nemotron-3 Super 120B-A12B | 120B | 12B (MoE) | Q4 | ~65 GB | Experimental | ~3-8 |

## Quick Start

```bash
pip install -r requirements.txt

# Run Nemotron-3 Nano (recommended — fastest, fits easily)
python run.py --model nano --prompt "Your question here"

# Run Nemotron-Super 49B (higher quality, still fits at 4-bit)
python run.py --model super-49b --prompt "Your question here"

# Benchmark mode
python run.py --model nano --benchmark --output-json results/nano_bench.json

# Experimental: 120B with LLM-in-a-Flash mmap offloading
python flash_loader.py --prompt "Your question" --analyze-only  # check feasibility first
python flash_loader.py --prompt "Your question"                 # run inference
```

## Architecture

### Nemotron-3 Family (March 2026)
- **Nano 30B-A3B**: Hybrid Mamba-Transformer MoE. 30B total params, only 3B active per token. Phenomenal efficiency on Apple Silicon.
- **Super 120B-A12B**: Hybrid Mamba-Transformer MoE. 120B total, 12B active. Needs >64GB RAM at Q4 — experimental on 48GB via mmap.
- **Llama-3.3-Nemotron-Super-49B**: Llama-3.3 backbone with NAS-optimized layers. Dense 49B, fits at 4-bit on 48GB.

### LLM-in-a-Flash Techniques Applied

From Apple's paper (Alizadeh et al., arXiv:2312.11514):

| Technique | Status | Notes |
|-----------|--------|-------|
| Flash loading (mmap) | Built into MLX | safetensors are memory-mapped by default |
| Sliding window attention | Model-dependent | Supported for architectures that define it |
| Row-column bundling | Not needed | Models fit in RAM at Q4 (except 120B) |
| Adaptive memory management | OS-level | macOS VM handles expert paging for MoE |
| Top-k FFN sparsity | MoE-native | Nemotron's MoE routing is inherently sparse |

### Memory Budget (M4 Pro 48GB)

```
Total unified memory:           48.0 GB
OS + system services:           -8.0 GB
MLX runtime + Python:           -1.5 GB
Available for model:            38.5 GB
```

## Research

This project was researched using [AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw).
Full research artifacts are in `research/`.
