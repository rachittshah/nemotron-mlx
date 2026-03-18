# Research Decomposition: Nemotron-3 Super on Apple Silicon (M4 Pro, 48GB) via MLX

## Source
- NVIDIA Nemotron-3 Super technical reports & HuggingFace model cards
- Apple "LLM in a Flash: Efficient Large Language Model Inference with Limited Memory" (Dec 2023, arXiv:2312.11514)
- MLX framework docs & `mlx-lm` library (github.com/ml-explore/mlx-examples)
- llama.cpp Apple Silicon backend & GGUF spec
- HuggingFace Hub (nvidia/Nemotron-3-Super-*)

---

## Sub-questions

### SQ1: What is the exact architecture of Nemotron-3 Super, and is it MLX-convertible?
- Is it Llama-3.1-based (as rumored) or a custom arch? Which attention mechanism (GQA, MQA, MHA)?
- Available sizes: 8B, 49B — are both open-weight on HuggingFace under permissive license?
- Does `mlx-lm convert` handle the architecture natively, or does it require custom modeling code?
- Any architecture quirks (MoE layers, custom FFN, hybrid dense-sparse) that break standard conversion?

### SQ2: What is the maximum model size that fits in 48GB unified memory with 8K context?
- Memory budget: 48GB total − ~8GB OS/system − MLX runtime overhead (~1-2GB) = ~38-39GB usable
- Model weights at each quant level: 8B×Q4 ≈ 4.5GB, 8B×Q8 ≈ 8.5GB, 49B×Q4 ≈ 27GB, 49B×Q8 ≈ 52GB (won't fit)
- KV-cache at 8K context: for 49B with GQA, estimate ~2-6GB depending on head count and quant
- **Critical question**: Does 49B-Q4 + 8K KV-cache fit under ~38GB? If not, what's the max context length?
- Comparison: can flash-loading (SQ4) relax this constraint by streaming weights from SSD?

### SQ3: What are the achievable inference speeds across backends?
- `mlx-lm` generate: tokens/sec for 8B-Q4, 49B-Q4 on M4 Pro (16-core GPU, 273 GB/s bandwidth)
- llama.cpp with Metal backend: same configurations, GGUF format
- Time-to-first-token (prefill) at 2K/4K/8K prompt lengths
- Throughput ceiling = memory bandwidth / bytes-per-token. M4 Pro theoretical max vs observed
- Does `mlx-lm` support continuous batching, speculative decoding, or prompt caching?

### SQ4: Which LLM-in-a-Flash techniques are implementable in MLX today?
- **Windowed attention**: MLX already supports sliding window — confirm compatibility with Nemotron arch
- **Row-column bundling** for sparse weight loading: not in MLX natively — estimate implementation effort
- **Flash loading from SSD** via memory-mapped files: MLX uses `mmap` by default for safetensors — is this already equivalent? Or does Apple's technique add predictive pre-fetching?
- **Adaptive memory management** (eviction/reload of layers): requires custom layer scheduler — how much latency does NVMe add per layer swap on M4 Pro (~7GB/s read)?
- Net question: which techniques give >10% real improvement vs what MLX already does?

### SQ5: Can we produce a working end-to-end script with benchmarking?
- Download → convert → quantize → inference pipeline using `mlx-lm`
- Handle tokenizer compatibility (Nemotron may use custom chat template)
- Benchmarking harness: tok/s, TTFT, peak memory (via `mlx.core.metal.get_peak_memory()`)
- Graceful fallback: if 49B doesn't fit, auto-select 8B with higher quant

---

## Priority Ranking

| Priority | Sub-question | Rationale |
|----------|-------------|-----------|
| **P0** | SQ1 — Architecture compatibility | Blocking. If MLX can't convert the model, everything else is moot. Check HuggingFace config.json first. |
| **P0** | SQ2 — Memory budget | Blocking. Determines which variant is even feasible. Pure arithmetic — cheapest to answer. |
| **P1** | SQ5 — Working script | Core deliverable. Once SQ1+SQ2 confirm feasibility, this is the artifact that matters. |
| **P2** | SQ3 — Benchmark speeds | Validates whether the setup is *useful*, not just *possible*. Run after script works. |
| **P3** | SQ4 — LLM-in-a-Flash techniques | Optimization layer. Only matters if baseline perf from SQ3 is insufficient. Most techniques either already exist in MLX (mmap) or require significant custom work (layer eviction). |

**Execution order**: SQ1 → SQ2 → SQ5 → SQ3 → SQ4

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Nemotron-3 Super 49B uses custom arch not supported by `mlx-lm` | Medium | High — requires writing custom MLX model class | Check config.json `architectures` field immediately; if custom, fall back to 8B or use llama.cpp |
| 49B-Q4 exceeds 38GB usable memory with 8K KV-cache | Medium-High | High — forces 8B variant or shorter context | Pre-calculate exact budget before downloading 25GB+ model; consider Q3 or mixed quantization |
| Model weights gated/licensed restrictively | Low-Medium | High — blocks the entire pipeline | Verify HuggingFace access before investing in conversion tooling |
| LLM-in-a-Flash SSD streaming adds latency that negates memory savings | Medium | Medium — makes 49B viable but too slow for interactive use | Benchmark layer-swap latency on M4 Pro NVMe first (~7GB/s = ~3.8s for full 49B-Q4 model reload) |
| MLX `mmap` already implements flash-loading, making Apple paper techniques redundant | High | Low (positive risk) — less custom work needed | Profile MLX memory behavior under pressure to confirm mmap eviction works correctly |
| Nemotron-3 Super tokenizer incompatible with standard `mlx-lm` chat templates | Low | Low — fixable with custom template | Inspect tokenizer_config.json for chat_template field |

---

**Next step**: Run SQ1 — pull the HuggingFace model card and `config.json` for both Nemotron-3 Super variants to confirm architecture and weight availability. This is a 30-second check that gates everything else.