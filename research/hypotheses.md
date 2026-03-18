# Synthesized Research Hypotheses: Nemotron-3 Super on M4 Pro 48GB

## H1: Framework Showdown — MLX vs llama.cpp IQ4_XS, with FP16 as Quality Ceiling

**Rationale:** The pragmatist's H1+H2 and the contrarian's H1 converge: before optimizing anything, establish ground truth across frameworks AND precisions. The contrarian is right that llama.cpp's imatrix quantization may beat MLX's naive group quant on quality at the same bitwidth — this hasn't been tested. The pragmatist is right that FP16 feasibility on 48GB is the underexplored sweet spot.

**Protocol:**
1. Nemotron-3 Super 8B in three configs: MLX 4-bit, MLX FP16, llama.cpp IQ4_XS GGUF
2. Llama-3.1 8B Instruct as control in same three configs
3. Metrics: tok/s at 512/2K/8K prompt lengths, TTFT, peak RSS, perplexity (WikiText-2 sample), 100 MMLU questions, 50 HumanEval problems
4. Profile bandwidth utilization via `sudo powermetrics` during generation

**Measurable prediction:** FP16 scores 1-3 MMLU points above best 4-bit config. llama.cpp IQ4_XS matches or beats MLX 4-bit on quality. MLX FP16 generation ≥15 tok/s (usable for interactive work). GPU memory bandwidth >65% during 4-bit generation (confirming bandwidth-bound regime).

**Failure condition:** If MLX 4-bit beats llama.cpp on all three axes (speed, quality, memory) by >10%, the contrarian's framework skepticism is wrong. If FP16 quality delta <0.5 MMLU points over best 4-bit, quantization is lossless and FP16 is waste.

**Unresolved disagreement:** Innovator argues 4-bit speed dividend enables best-of-N to *exceed* FP16 quality. Contrarian says FP16 is the right default. **Resolution: test both.** Add a 4th config — MLX 4-bit with best-of-4 majority vote on the same MMLU/HumanEval sets. If best-of-4 Q4 ≥ FP16 greedy within wall-clock parity, the innovator's "negative quantization tax" is real and changes the entire decision framework.

**Effort:** ~3 hours, zero API cost, all local. **Run first.**

---

## H2: 49B Diagnostic — Conversion Autopsy + Architecture Feasibility Gate

**Rationale:** All three perspectives agree the 49B path needs a go/no-go signal before any investment. The pragmatist's H3 (diagnostic conversion attempt) is the cheapest way to get it. The contrarian's H2 (8B FP16 likely beats 49B Q4) sets the bar the 49B must clear.

**Protocol:**
1. Download 49B `config.json` + `model.safetensors.index.json` only (few KB)
2. Diff against standard Llama-3.1 config — enumerate all non-standard block types
3. Attempt `mlx-lm.convert` — capture full traceback
4. For each failure point, check: does MLX have an existing implementation? (Mixtral MoE support, etc.)
5. Estimate LOC for custom model class based on gap analysis

**Measurable prediction:** Conversion fails with ≥2 distinct unsupported layer types (linear attention, MoE routing). Config reveals block-type heterogeneity not present in any MLX-supported model. Custom implementation estimate: 200-500 LOC, 2-5 days.

**Failure condition (and best outcome):** Conversion succeeds end-to-end — the hybrid blocks are Llama-compatible at the weight level. Then immediately benchmark 49B Q4 vs 8B FP16 on MT-Bench to test the contrarian's "8B wins" claim.

**Kill criterion (contrarian's key contribution):** If custom MLX work estimate exceeds 3 days AND 8B FP16 from H1 scores within 85% of NVIDIA's reported 49B benchmarks, **abandon the 49B path.** The ROI is negative. This kill criterion must be set before results come in.

**Effort:** ~1-2 hours. **Run parallel with H1.**

---

## H3: Bandwidth Profiling — Identify the Real Bottleneck Before Optimizing

**Rationale:** The contrarian's sharpest insight: the synthesis misdiagnoses the constraint. LLM-in-a-Flash solves memory *capacity* — but on M4 Pro 48GB with an 8B model, you're capacity-rich and bandwidth-poor. The innovator's H3 (KV-cache bifurcation) and the contrarian's H3 both point to bandwidth as the binding constraint, but prescribe different solutions. **Profile first, optimize second.**

**Protocol:**
1. During H1 benchmarks, capture `powermetrics` GPU bandwidth and compute utilization traces
2. At 8K context: measure what fraction of token latency is weight-read vs attention-compute vs KV-cache access
3. Test at 32K context (if model supports): does KV-cache bandwidth become the new bottleneck?
4. If bandwidth >65% and compute <40%: the right optimizations are sparsity-aware weight loading, speculative decoding, sparse attention — NOT Flash-style memory management
5. If compute >60%: kernel optimization matters more than memory tricks

**Measurable prediction:** At 4-bit 8B generation, bandwidth utilization >65%, compute <40%. At 32K context, KV-cache access becomes >30% of per-token latency (validating the innovator's KV-cache concern, but reframing it as bandwidth not capacity).

**Failure condition:** If compute >60% — the system is compute-bound, standard kernel optimization is the path, and both Flash techniques and bandwidth tricks are irrelevant.

**What this unlocks:** The profiling data determines whether to pursue the innovator's KV-cache bifurcation (if KV bandwidth is the bottleneck at long context) or the contrarian's speculative decoding recommendation (if weight-read bandwidth is the bottleneck at short context). Without this data, both are guesses.

**Effort:** ~1 hour, piggybacked on H1. **Run parallel.**

---

## H4 (Conditional): OS-Level MoE Paging for 49B — Only If H2 Clears the Gate

**Rationale:** The innovator's boldest idea (H2: page-fault-driven MoE) and the pragmatist's H4 describe the same mechanism from different angles. The contrarian correctly flags that 49B Q4 may lose to 8B FP16 — but if it doesn't, this is the highest-upside finding: 49B on consumer hardware with zero custom memory management code.

**Protocol (only execute if H2 conversion succeeds AND 49B shows >15% quality uplift over 8B FP16):**
1. Load 49B 4-bit via MLX mmap
2. Constrain memory via `MLX_METAL_MEMORY_BUDGET` at 20GB, 25GB, 30GB
3. Monitor: RSS vs virtual, page faults (`vm_stat`), SSD read bandwidth (`iostat`), tok/s
4. If stable: run quality eval at constrained memory to check for corruption from page faults during computation

**Measurable prediction:** At 25GB budget, sustained generation ≥5 tok/s after 50-token warmup. Working set (active experts) ≈10-15GB. SSD reads ≈2-4 GB/s during generation.

**Failure condition:** OOM kill at any budget <35GB (MLX pins all weights in Metal buffers, bypassing mmap paging). Or <3 tok/s even at 30GB budget (expert activation isn't skewed enough, causing thrash).

**Effort:** ~2 hours, but gated on H2 success + quality threshold. **Do not start until H1+H2 results are in.**

---

## Execution Plan

```
Day 1 (parallel):
  H1: Framework + precision benchmark     [3 hrs, zero cost]
  H2: 49B conversion diagnostic            [1-2 hrs, zero cost]  
  H3: Bandwidth profiling (during H1)      [1 hr, piggybacked]

Day 1 gate:
  H1 results → decide: FP16 default or 4-bit+best-of-N?
  H2 results → decide: pursue 49B or kill?
  H3 results → decide: optimize for bandwidth or compute?

Day 2 (conditional):
  H4: 49B MoE paging — ONLY if H2 passes AND 49B quality > 8B FP16 + 15%
```

## Unresolved Disagreements (Preserved)

| Tension | Innovator | Contrarian | Resolution |
|---|---|---|---|
| Best-of-N recovers quantization loss? | Yes — 3.5x speed enables it | Unlikely on MMLU; maybe on reasoning | **Test in H1 with both MMLU and HumanEval** |
| 49B worth pursuing? | High upside | Negative ROI vs 8B FP16 | **H2 gate + kill criterion before investing** |
| LLM-in-a-Flash relevance? | KV-cache bifurcation idea is novel | Wrong paper for this hardware | **H3 profiling resolves which bottleneck is real** |
| MLX vs llama.cpp? | MLX assumed default | llama.cpp likely better | **H1 head-to-head settles it empirically** |

The contrarian's strongest contribution: **set kill criteria before running experiments, not after.** Every hypothesis above has an explicit failure condition and no hypothesis is pursued past its gate.