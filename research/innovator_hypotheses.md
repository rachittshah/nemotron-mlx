## Novel Hypotheses: Nemotron-3 Super on Apple Silicon

---

### Hypothesis 1: "Negative Quantization Tax" — 4-bit + Compute-Optimal Decoding Beats FP16 Greedy on Quality-Per-Second

**Bold claim:** On M4 Pro unified memory, Nemotron-3 Super 8B at 4-bit quantization with best-of-4 sampling will produce **higher quality outputs per wall-clock second** than FP16 greedy decoding — meaning quantization doesn't just save memory, it *improves effective intelligence* when you reinvest the bandwidth savings.

**Cross-domain inspiration:** Algorithmic trading — a faster but noisier signal with ensemble averaging beats a slower, cleaner signal. Same logic: 4-bit is noisier per-token but ~3-4x faster on bandwidth-bound Apple Silicon, so you can afford redundant inference passes and majority-vote.

**Rationale grounded in gaps:**
- Gap 3 notes zero benchmarks for Nemotron-3 Super on Apple Silicon. The synthesis assumes FP16 > 4-bit on quality — this is the default assumption everyone makes, but it ignores that Apple Silicon inference is **memory-bandwidth-bound**, not compute-bound. A 4-bit model moves 4x less data through the memory bus per token.
- The synthesis flags P2 ("determines if quantization is even needed for 8B") but frames it as a quality-loss question. It never considers that the speed dividend could be *reinvested* into quality-recovering decoding strategies.
- Literature on self-consistency (Wang et al., 2022) shows majority voting over multiple CoT samples improves accuracy by 5-15% on reasoning tasks. If 4-bit runs 3.5x faster, you get 3-4 samples in the time of one FP16 sample.

**Measurable prediction:**
- Run Nemotron-3 Super 8B FP16 greedy on 50 MMLU questions → score S_fp16, wall-clock time T_fp16
- Run 4-bit with best-of-4 (majority vote) on same questions → score S_q4_bo4, wall-clock time T_q4_bo4
- **Pass condition:** S_q4_bo4 ≥ S_fp16 AND T_q4_bo4 ≤ T_fp16
- **Failure condition:** S_q4_bo4 < S_fp16 - 2% OR T_q4_bo4 > T_fp16

**Risk level:** Medium — the speed gain is near-certain, but whether best-of-N recovers quantization loss on MMLU specifically is uncertain. MMLU may not have enough variance in sampled outputs. Stronger effect expected on math/reasoning benchmarks (GSM8K).

---

### Hypothesis 2: "Page-Fault-Driven MoE" — The 49B Fits at 8-bit by Letting macOS Virtual Memory Do Expert Routing

**Bold claim:** Nemotron-3 Super 49B at **8-bit** (49 GB, exceeding 48 GB physical RAM) can run on M4 Pro at **>5 tokens/sec** by memory-mapping the full model and relying on macOS unified memory's SSD swap + Apple's 7.4 GB/s NVMe to page in active MoE experts on demand — no custom sparse loading code needed. The OS virtual memory system *is* the adaptive memory manager that LLM-in-a-Flash describes, and MoE's sparse activation pattern makes it work.

**Cross-domain inspiration:** Database buffer pool management. PostgreSQL doesn't load entire tables into RAM — it relies on the OS page cache to keep hot pages resident and evict cold ones. MoE expert weights behave like database pages: only a subset is "hot" at any time, access patterns are skewed (Zipf-distributed expert selection), and the working set fits in RAM even when the full dataset doesn't.

**Rationale grounded in gaps:**
- Gap 2 identifies that LLM-in-a-Flash's row-column bundling and adaptive memory management aren't in MLX. But the synthesis **overlooks that macOS already provides this at the OS level** via mmap + unified memory. MLX uses mmap by default for safetensors — so the infrastructure is already there.
- Gap 4 notes that "neither MLX nor llama.cpp exploits [MoE sparsity] for memory savings." But the OS-level page eviction **automatically** exploits it: cold expert weights get paged out, hot ones stay resident. No framework support needed.
- The synthesis calculates 49B 8-bit at 52.2 GB = "doesn't fit." But this treats memory as a hard binary. On Apple Silicon with swap, "fitting" is a throughput question, not a yes/no. M4 Pro's NVMe can sustain ~7.4 GB/s reads, and per-token expert activation only touches ~15-20% of total expert parameters (top-2 out of 8+ experts per MoE layer).
- Key calculation: if only 20% of 49 GB expert weights are active per token = ~10 GB working set, which fits easily in the ~38.5 GB available budget. Page faults only occur on cold experts.

**Measurable prediction:**
- `mmap` the 49B 8-bit safetensors via MLX, run inference on 20 prompts
- **Pass condition:** Sustained generation ≥ 5 tokens/sec after warmup (first 50 tokens), no OOM kill
- **Failure condition:** < 3 tokens/sec sustained, OR macOS kills the process, OR generation halts due to memory pressure

**Risk level:** High — three things could break it: (1) macOS memory pressure killer may terminate the process before swap kicks in gracefully, (2) the 49B's hybrid architecture may fail MLX conversion entirely (Gap 1), making this untestable, (3) expert activation patterns may not be skewed enough, causing thrashing. Mitigation: test first with a known MoE model like Mixtral 8x22B that MLX already supports.

---

### Hypothesis 3: "KV-Cache Bifurcation" — Split KV-Cache Between Unified RAM (Recent) and Memory-Mapped SSD (Old), Achieving 128K Context on 8B

**Bold claim:** By memory-mapping the KV-cache and keeping only the most recent 4K tokens' cache in active RAM (sliding window), Nemotron-3 Super 8B can process **128K context** on M4 Pro 48GB at >15 tok/s generation — a 16x context extension over what the memory budget "allows" — with <2% quality degradation on long-context benchmarks vs. full-attention baselines.

**Cross-domain inspiration:** CPU L1/L2/L3 cache hierarchy. Modern CPUs don't keep all data in the fastest cache — they tier it by recency/frequency. Apply the same to KV-cache: "L1" = recent 4K tokens in unified memory (fast random access), "L2" = older tokens on NVMe SSD via mmap (high bandwidth sequential, acceptable random access at ~7.4 GB/s). Attention scores decay roughly exponentially with distance, so older KV entries are accessed less and tolerate higher latency.

**Rationale grounded in gaps:**
- Gap 5 identifies KV-cache compression as missing from MLX but only considers int8 quantization — a linear improvement. This hypothesis proposes a **hierarchical** approach that's multiplicative.
- The synthesis computes KV-cache at 8K context = ~1 GB for 8B. At 128K, that's ~16 GB — still fits in RAM! But the real constraint is that attention computation over 128K tokens is O(n²) and thrashes the memory bus. A sliding window for generation + sparse retrieval of old KV entries via mmap sidesteps this.
- LLM-in-a-Flash's core insight is SSD-as-extended-memory for weights. Nobody has applied this to **KV-cache**, which is the actual scaling bottleneck for long-context inference when models are small enough to fit in RAM.
- Recent work on StreamingLLM (Xiao et al., 2023) shows that keeping attention sinks (first few tokens) + recent window preserves generation quality. This hypothesis extends it: keep sinks + window in RAM, but also allow sparse retrieval of mid-context KV entries from SSD for retrieval-heavy tasks.

**Measurable prediction:**
- Implement a custom KV-cache manager in MLX that mmaps old entries and keeps recent 4K + first 4 tokens in active memory
- Test on RULER or Needle-in-a-Haystack at 32K and 128K context
- **Pass condition:** Needle-in-a-Haystack accuracy ≥ 90% at 32K context AND generation speed ≥ 15 tok/s AND total memory usage < 20 GB
- **Failure condition:** Accuracy < 80% at 32K OR generation speed < 8 tok/s OR SSD I/O becomes bottleneck (>50% of token latency)

**Risk level:** Medium-high — StreamingLLM validates the attention sink principle, but mmap'ing KV-cache introduces page fault latency in the attention computation hot path. The bet is that attention score sparsity means old KV entries are rarely accessed during generation (mostly needed for retrieval tasks, where latency is tolerable). 30-minute feasibility is tight — a minimal prototype with fixed window + mmap is doable, full sparse retrieval is not.

---

## Comparison Matrix

| | H1: Negative Quant Tax | H2: Page-Fault MoE | H3: KV-Cache Bifurcation |
|---|---|---|---|
| **Novelty** | Medium — reframes known tradeoff | High — zero-code architecture exploit | High — new memory hierarchy for KV |
| **30-min feasibility** | High — just run 2 configs + eval | Medium — depends on 49B MLX compat | Medium — needs ~50 LOC custom cache |
| **Upside if true** | Changes quantization decision framework | Enables 49B on consumer hardware | 16x context extension for free |
| **Risk** | Medium | High | Medium-high |
| **Falsifiability** | Clean — MMLU score comparison | Clean — tok/s threshold | Clean — accuracy + speed threshold |

**Recommended execution order:** H1 first (cheapest test, highest feasibility), then H3 (novel contribution, moderate effort), then H2 (highest risk, dependent on MLX 49B conversion).