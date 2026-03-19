# Fitting Nemotron-3 Super 120B-A12B on 48GB Apple Silicon

## Prerequisites

- **llama.cpp build 8400+** (Nemotron-3 Mamba-2/MoE merged Dec 2025 - Jan 2026)
  - PRs: #18058 (Nemotron 3 Nano), #18295 (sigmoid topk_moe for nemotron), #18599 (backend fix)
  - `brew install llama.cpp` gets you build 8400
- **Ollama 0.12+** also works (wraps llama.cpp) but doesn't expose expert-level offloading
- MLX does NOT work for models exceeding RAM (confirmed 0.025 tok/s with swap)

## The Problem

120B params. 48GB RAM. Even at the most aggressive quantization (IQ1_S), the GGUF
is 46.38 GB — leaving <2GB for OS, KV-cache, and compute buffers. Naive loading fails.

## GGUF Size Reality Check

**bartowski quants** (standard imatrix):

| Quant | File Size | Fits 48GB? | Quality |
|-------|----------|-----------|---------|
| IQ1_S | 46.38 GB | Barely | Extremely degraded |
| IQ1_M | 47.80 GB | No (with OS overhead) | Extremely degraded |
| IQ2_XXS | 50.16 GB | No (+2.2 GB over) | Very low |
| IQ2_XS | 52.04 GB | No | Low |
| Q2_K | 54.82 GB | No | Very low |
| IQ3_XXS | 61.63 GB | No | Lower |
| Q3_K_M | 64.65 GB | No | Low |
| IQ4_XS | 67.20 GB | No | Decent |
| Q4_K_M | 86.98 GB | No | Good (recommended) |

**Unsloth Dynamic 2.0** (keeps attention/shared layers at higher precision):

| Quant | File Size | Note |
|-------|----------|------|
| UD-IQ1_M | 52.7 GB | Higher quality than bartowski IQ1 |
| UD-IQ2_XXS | 52.7 GB | Better than standard IQ2 |
| UD-IQ2_M | 52.7 GB | Best at 2-bit |
| UD-Q2_K_XL | 54.7 GB | Dynamic mixed precision |

**Bottom line:** No quant fits natively. All approaches require memory management tricks.

## Why MoE Makes This Solvable

The 120B model has only **12B active params** per forward pass (LatentMoE routing).
This means:

```
Total model weights:     46-52 GB  (depending on quant)
Active weights/token:    ~5-6 GB   (12B at ~0.5 bytes/param)
Shared layers (attn):    ~4-5 GB   (always needed)
KV-cache (8K, Q4):       ~1-2 GB
Working set per token:   ~11-13 GB  ← fits in 48GB with room to spare
```

The remaining 35-40 GB of cold expert weights can live on NVMe SSD via mmap,
paged in on demand. This is exactly what Apple's "LLM in a Flash" paper describes.

## Three Approaches (ranked by viability)

### Approach 1: llama.cpp + Expert Offloading (BEST)

**How it works:** llama.cpp's `--cpu-moe` flag places MoE expert weights on
the CPU side of unified memory while keeping attention layers on Metal GPU. The
CPU-side weights are mmap'd, so macOS can page cold experts to NVMe swap.

**Why this works for MoE:** Expert routing is sparse — only 4-5 of 128 experts
fire per token. The "hot" experts stay resident in RAM; cold experts get paged
out. NVMe at 7.4 GB/s can reload an expert fast enough.

```bash
# Step 1: Maximize GPU memory allocation
sudo sysctl iogpu.wired_limit_mb=45056  # 44 GB for Metal, 4 GB for OS

# Step 2: Download the model (pick your quant)
# IQ2_XXS = best quality that's close to fitting (50.16 GB)
# IQ1_S = only one under 48GB (46.38 GB) but terrible quality

# Step 3: Run with expert offloading
llama-server \
  -m bartowski_Nemotron-3-Super-120B-A12B-IQ2_XXS.gguf \
  -ngl 99 \
  --cpu-moe \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  -fa on \
  -c 8192 \
  -b 4096 -ub 4096

# Alternative: offload N layers' experts to CPU (from highest layer down)
llama-server \
  -m model.gguf \
  -ngl 99 \
  --n-cpu-moe 30 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  -fa on -c 8192
```

**Expected performance:** ~5-15 tok/s generation (8K context)

**Memory breakdown:**
```
Attention + shared layers on Metal:  ~10-12 GB
KV-cache (8K, q8/q4):               ~1-2 GB
Compute buffers:                     ~2-3 GB
Metal total:                         ~15-17 GB

Cold expert weights (mmap'd):        ~35-40 GB
  → Hot experts in RAM:              ~5-8 GB
  → Cold experts on NVMe swap:       ~27-35 GB
```

### Approach 2: Aggressive Quant + Full Metal Load

**How it works:** Use bartowski IQ1_S (46.38 GB) and try to load entirely into
Metal GPU memory. Requires `iogpu.wired_limit_mb` override and aggressive KV quant.

```bash
sudo sysctl iogpu.wired_limit_mb=46080  # 45 GB for Metal

llama-server \
  -m bartowski_Nemotron-3-Super-120B-A12B-IQ1_S.gguf \
  -ngl 99 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -fa on \
  -c 4096
```

**Problems:**
- IQ1_S quality is extremely degraded (perplexity ~2x baseline)
- Leaves <2GB for KV-cache + compute — may OOM at >2K context
- 1-bit quant of MoE experts may break routing entirely

**Verdict:** Not recommended. Quality too low to be useful.

### Approach 3: MLX + mmap (DON'T DO THIS)

MLX does NOT gracefully handle models exceeding RAM. Confirmed by MLX maintainer:
"we can't mmap memory in a way that makes it available for the GPU as well."

Result: **0.025 tok/s** (240x slower than when model fits). Unusable.

## Key Techniques Applied

### KV-Cache Quantization
```
--cache-type-k q8_0 --cache-type-v q4_0
```
- Keys at Q8 (positional encoding sensitive to quantization)
- Values at Q4 (tolerates aggressive compression)
- 59% memory savings vs FP16 with <1% quality loss
- Saves ~3-4 GB at 8K context

### macOS GPU Memory Override
```bash
sudo sysctl iogpu.wired_limit_mb=45056
```
- Default: Metal gets ~75% of RAM = 36 GB
- Override: Metal gets 44 GB, leaving 4 GB for OS
- Resets on reboot. Persist via `/etc/sysctl.conf`

### Flash Attention
```
-fa on
```
- Required for memory-efficient attention at long context
- Reduces memory from O(n²) to O(n) for attention scores

### MoE Expert Offloading
```
--cpu-moe        # All experts to CPU
--n-cpu-moe 30        # Offload 30 layers' experts to CPU
```
- Expert weights computed by CPU instead of Metal GPU
- CPU is slower but avoids GPU memory pressure
- On Apple Silicon unified memory, "CPU" and "GPU" share the same physical RAM
- The advantage: mmap'd CPU tensors can be paged to swap without crashing Metal

## Why MoE Tolerates Extreme Quantization

Research confirms MoE models are **more quantization-resilient** than dense models:

1. **MoQE (arXiv:2310.02410):** Expert FFN layers have narrower weight
   distributions with fewer outliers than dense FFN. At 2-bit, dense models
   collapse while MoE maintains functionality.

2. **QuantMoE-Bench (arXiv:2406.08155):** Mixed-precision structure-aware
   allocation at 3-bit recovers 96% of 4-bit quality on Mixtral-8x7B.
   Key: shared experts need more bits than routed experts.

3. **NVIDIA NVFP4:** Nemotron-3 Super was trained with 4-bit quantization-aware
   training. The NVFP4 version is within ±1.3 points of BF16 on ALL benchmarks
   (MMLU-Pro, Arena-Hard, GPQA, etc.). Post-training GGUF Q4 cannot match this.

## Recommended Configuration

For daily use on M4 Pro 48GB:

```bash
# Best quality that's practically runnable:
# bartowski IQ2_XXS (50.16 GB) with expert offloading

sudo sysctl iogpu.wired_limit_mb=45056

llama-server \
  -m nvidia_Nemotron-3-Super-120B-A12B-IQ2_XXS.gguf \
  -ngl 99 \
  --cpu-moe \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  -fa on \
  -c 8192 \
  -b 4096 -ub 4096 \
  --host 0.0.0.0 --port 8080
```

Then connect via OpenAI-compatible API at `http://localhost:8080/v1`.

## Should You Actually Do This?

**Honest assessment:**

| Metric | 120B IQ2 on 48GB | Nano 30B Q4 on 48GB |
|--------|------------------|---------------------|
| tok/s | ~5-15 (estimated) | **69 avg / 88.6 peak** |
| Quality | Degraded (IQ2) | Full (4-bit, trained) |
| Memory headroom | 0 (tight) | **32 GB free** |
| Stability | Fragile (swap-dependent) | **Rock solid** |
| Context length | 4-8K max | **32K+ easily** |

The Nano 30B-A3B has only 3B active params but was specifically designed for
efficiency. At 4-bit on your hardware, it's 10x faster and more reliable than
trying to squeeze 120B into 48GB.

**When 120B makes sense:** When you need the full reasoning capability and are
willing to accept 5-15 tok/s with limited context. For agentic workloads where
quality per token matters more than throughput.

**When Nano wins:** Interactive use, long context, coding, anything where
speed matters. The Nano punches well above its weight class.

## References

- [bartowski Nemotron-3 Super GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-3-Super-120B-A12B-GGUF)
- [Unsloth Nemotron-3 Super GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF)
- [llama.cpp MoE offload guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)
- [MoQE: Mixture of Quantized Experts](https://arxiv.org/abs/2310.02410)
- [QuantMoE-Bench](https://arxiv.org/abs/2406.08155)
- [MxMoE (ICML 2025)](https://arxiv.org/abs/2505.05799)
- [KV-cache quantization benchmarks](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/)
- [Apple LLM in a Flash](https://arxiv.org/abs/2312.11514)
- [macOS GPU VRAM override](https://blog.peddals.com/en/fine-tune-vram-size-of-mac-for-llm/)
- [MLX mmap limitation](https://github.com/ml-explore/mlx/discussions/615)
