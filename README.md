# erisinky (`inkling`)

A Rust + hand-fused Metal 4 inference engine for **Thinking Machines' Inkling 2 Mini** (`inkling2_loop`, 26.4B total / 3.4B active MoE) on Apple Silicon, reading GGUF weights directly via zero-copy `mmap`.

Beyond stock dense inference (**63 tok/s** decode, **2,200 tok/s** prefill on M5 Pro), `erisinky` implements **Zero-RoPE KV caching & sparse long-context infrastructure** (`Kv256kPool`, TokenDance/KVCOMM multi-agent KV sharing, and the companion [`inkling-ikv`](https://github.com/Eriskii/inkling-ikv) pre-compiled `.ikv` cartridge harness):

- **Full write-up, interactive diagrams, and timelapse benchmarks:** [Pre-Processed KV Cartridges & Sparse Paging (`eriskii.net/research/inkling-ikv`)](https://eriskii.net/research/inkling-ikv)

---

## The Inference Paradigm: Two Independent Levers

Because Inkling 2 Mini has **no RoPE** on its 8 global full-attention layers and uses a **1,024-token sliding window** on the other 40 layers, two orthogonal techniques stack cleanly on top of the base engine:

1. **Pre-processed `.ikv` files ([`inkling-ikv`](https://github.com/Eriskii/inkling-ikv)):** Compile workspace files or document archives into content-hashed, 4 KiB-aligned KV cartridges offline or while the GPU is idle (e.g., during builds or while you read). When an agent reads a whole file — or slices it via `sed`, `head`, `tail`, `cat`, or ranged `read_file` — the harness stitches the pre-computed global KV directly into live context at any offset with zero RoPE re-rotation, cutting file-ingestion latency by **14×–99×**. Works on top of *either* Dense FP16 or Sparse KV.
2. **Live cache compression + sparse paging (`src/kv256k.rs`):** Compresses live global KV with Walsh-Hadamard rotated quantization (`WHT-KV4` = `3.88×` smaller than FP16, `WHT-K3V2` = `6.10×`), tiers cold pages onto NVMe via `F_NOCACHE` (preventing macOS unified-memory bloat), and runs per-token `WHT-2bit-sketch` Top-K sparse paging (`O(N / P)` scan, `9,216` sharp tokens fixed + background centroids). Keeps 256K RAM around `~1–2 GiB` (`~172 MiB` cold-tiered) and stretches single-laptop context to **1M–10M tokens**. Works on top of *either* raw prefill or `.ikv`.

---

## Choosing the Right Mode & Knobs

Not every workload benefits from every lever. Use this guide to pick the right mode for your situation:

### Workload Decision Matrix

| Situation | Recommended Setup | Why |
|---|---|---|
| **Short interactive chat or one-shot unseen prompts** (`< 64K` ctx, files read once) | **Dense FP16** (`inkling chat` / `inkling run`) | Fastest raw prefill (`2,200 tok/s`) and fastest short-context decode (`53–63 tok/s`). Sparse paging adds upfront overhead you won't recoup under ~70K tokens. |
| **Coding agent on a real repository** (`50K–256K` ctx, repeated file reads & `sed` slices) | **Sparse `KV4`-pool + `KV4` `.ikv` + slicing (`M9`)** (`inkling-ikv tui --sparse -c 262144 --pool-bits 4 --kv-bits 4`) | Best overall balance: `4.5×–54×` faster file reads (`24.9 s` vs `112.2 s` TTFT), zero file-attribution blur (`6.0/6` rubric), flat `~29 tok/s` decode past 100K, `~1–2 GiB` KV RAM. |
| **Multi-file document search / browsing** (`--notools`, large archives `100K–256K`) | **Sparse `KV4`-pool + `KV4` `.ikv` (`M9`)** (`inkling-ikv tui --sparse -c 262016 --pool-bits 4 --kv-bits 4 --notools`) | Ingests `118K` tokens across 20 full documents in `9.35 s` (`25.5×` faster/byte than Sparse raw; `~99×` faster on a 42K-token file) while preserving exact entity/quote recall (`89/100` vs `78/100` for `K3V2`). |
| **Disk-constrained archive / fast topical gist** (millions of tokens on SSD, exact proper nouns less critical) | **Sparse `KV4`-pool + `K3V2` `.ikv` (`M3`)** (`--pool-bits 4 --kv-bits 3`) | Saves ~35% disk (`5.25` vs `8.25 KiB/tok`) and runs slightly faster, at the cost of softer proper-noun recall across large stitched glossaries. |
| **Mid-sized session (`< 70K`) with NO `.ikv` cache** (cold one-off codebase) | **Dense FP16 (`--dense --no-ikv`)** | Without `.ikv`, Dense beats Sparse raw prefill (`39.7 s` vs `112.2 s` TTFT; `39.4` vs `31.2 tok/s`). Crossover flips in Sparse's favor past `~70K–100K` (or immediately once `.ikv` is warm). |
| **Extreme context (`256K` → `1M–10M`) or 16–128 concurrent 256K streams** (48 GB Mac) | **`Kv256kPool` + `F_NOCACHE` disk tiering** (`bench-256k`, `needle-10m`) | Dense FP16 hits `8.16 GiB` per 256K stream (`320 GiB` at 10M); `Kv256kPool` drops 256K to `~172 MiB`/stream (`64` streams in `10.5 GiB` @ `337 tok/s`), `1M` to `3.37 GiB` (`19.8 tok/s`), and `10M` to `5.21 GiB` (`5–8 tok/s`). |
| **High-throughput parallel batch generation** (evals, rollouts, shared prefix) | **`inkling batch` / `bench-batch`** (`--shared`) | Cascade shared-prefix attention + GPU top-k/top-p sampling with `--max-mem` OOM guard. |
| **Multi-agent swarm sharing context behind different private prefixes** | **Zero-RoPE paste / TokenDance Master-Mirror** (`src/kv_share.rs`) | Direct KV paste with `0%` recompute already hits `95/96` argmax (`KL 0.0081`); adding `2–5%` sparse diffs hits `96/96` (`KL 0.0049–0.0072`) with `15–28×` less RAM and `1.19–1.28×` faster fused attention. |

### Key Knobs & Pitfalls to Avoid

- **Live sparse pool precision (`--pool-bits`):** Always use **`4` (`WHT-KV4`)** (now the default). Running `K3V2` (`--pool-bits 3`) in the live pool while mounting `.ikv` cartridges double-quantizes 2-bit Values (`16.3×` worse mid-file slice KL: `0.197 → 0.012`).
- **On-disk `.ikv` precision (`--kv-bits`):**
  - `--kv-bits 4` (`WHT-KV4`, `8.25 KiB/tok`, `3.88×` vs FP16): **Recommended default** for coding and entity-heavy search. Eliminates cross-file attribution blur for only `+14–34 ms` extra stitch time.
  - `--kv-bits 3` (`WHT-K3V2`, `5.25 KiB/tok`, `6.10×` vs FP16): Good for cold archival storage or huge corpora where disk space matters more than verbatim proper-noun recall.
  - `--kv-bits 16` (`FP16`): Exact bit-identical reference/debugging only.
- **Page size & Top-K:** Keep **64-token pages (`Top-K = 128`, `8,192` sharp far tokens)** through `256K` context (selected automatically). Only switch to `128`-token pages (`Top-K = 64`) above `256K` (`512K–10M`).
- **Reasoning effort (`--effort`) & sparse paging:** Enable light chain-of-thought (`--effort 0.1` or higher) whenever running sparse paging on multi-hop tasks. Because the `1.6 ms` `WHT-2bit-sketch` scan refreshes on *every* generated token, CoT hops dynamically fetch new pages (`17/25 → 25/25` on our multi-hop eval).
- **Sampling (`--temp`):** Keep Inkling's native **`--temp 1.0 --top-p 0.95 --top-k 64`** for agent loops. `--greedy` (`--temp 0`) performed worse in coding runs (`+110 s` slower, repeated slice loops, `4.5/6` rubric) — reserve `--greedy` for deterministic logit checks and throughput microbenchmarks.
- **Seam recomputation (`--seam`):** Leave at **`0`**. Offline cartridges already strip an 8-token throwaway BOS sink prefix (halving stitch KL from `0.029 → 0.015`); read-time seam recomputation (`seam=16`) actually degraded KL (`0.046` vs `0.032`).
- **Homogenous files under blind stitching:** Real source files and distinct essays stitch cleanly at `seam=0`. Near-identical synthetic logs with no manifest header blur file identity (`2/10` on synthetic RULER logs vs `14/15` on real Rust files) — prepend a short manifest or use raw prefill / `2–5%` check-layer refresh for homogeneous logs.
- **Memory safety (48 GB unified memory):** `Inkling-2-Mini-Q4_K_M.gguf` takes `~16.3 GiB` resident. Run **one** engine process at a time; `inkling-ikv` shares a single `Engine` cooperatively between foreground decode and background `.shadow-ikv/` packing.

---

## Build & Usage

Requires macOS on Apple Silicon with Metal 4 (M5 neural accelerators are used for prefill GEMMs and prefill attention).

```sh
# 1. Build the core engine (this repo)
cargo build --release

# 2. Optional: build the companion .ikv cartridge & agent harness (sibling checkout)
cargo build --release --manifest-path ../inkling-ikv/Cargo.toml
```

### Core Engine (`./target/release/inkling`)

```sh
# Interactive dense chat & single-prompt runs
./target/release/inkling chat                                # multi-turn dense FP16 chat
./target/release/inkling run -p "Explain NoPE" --effort low
./target/release/inkling run -f prompt.txt --greedy          # GPU-pipelined greedy decode
./target/release/inkling bench -f prompt.txt -n 256          # prefill + decode throughput

# High-throughput parallel batch generation
./target/release/inkling batch -b 32 -p "Write a haiku." --shared
./target/release/inkling bench-batch -b 1,8,32,64 --shared

# Sparse 256K / 1M–10M context & multi-agent KV-sharing benchmarks
./target/release/inkling bench-256k -b 64 -c 262144 --kv-bits 4 --topk-pages 64
./target/release/inkling needle-10m --subtest 1m             # 1M-token needle retrieval (~3.37 GiB)
./target/release/inkling needle-10m --subtest 10m            # 10M-token needle retrieval (~5.21 GiB)
./target/release/inkling kv-share-probe -n 32                # Zero-RoPE paste + TokenDance + KVCOMM probe
./target/release/inkling kv-probe -n 24                      # KV quantization & pruning ablation probe

# Logit parity against llama.cpp (tools/ref_logits.cpp)
./target/release/inkling check --ref <dump>
./target/release/inkling check-batch --ref <dump>

# Multi-stream & dual-persona LoRA servers
./target/release/inkling serve --adapter path/to/adapter.imsa --port 8765
```

### Pre-Compiled `.ikv` Cartridges & Agent Harness ([`Eriskii/inkling-ikv`](https://github.com/Eriskii/inkling-ikv))

```sh
IKV=../inkling-ikv/target/release/inkling-ikv

# Recommended coding agent (Sparse KV4-pool + KV4 .ikv + sed slicing + Seatbelt sandbox)
$IKV tui --dir /path/to/repo --sparse -c 262144 --pool-bits 4 --kv-bits 4 --effort 0.1

# Read-only document search over a large markdown/text archive (--notools = list_files + read_file only)
$IKV tui --dir /path/to/docs --sparse -c 262016 --pool-bits 4 --kv-bits 4 --notools

# Standalone cartridge workflow (pack once, stitch anywhere)
$IKV pack -i src/engine.rs -o engine.ikv --kv-bits 4
$IKV inspect engine.ikv
$IKV read --ikv engine.ikv,session.ikv -p "How does KV reservation work?"

# 25-task multi-hop eval & GPU roundtrip check
$IKV check -n 32 -c 8192
$IKV task-bench --effort 0.1 --with-kv4
```

Inside `inkling-ikv tui`, use `/shadow` to inspect `.shadow-ikv/` compile status, `/pack-now` to pre-compile queued files, `/mount <path>` to mount a cartridge manually, `/sparse` or `/dense` to toggle live KV backends, and `/tokens` for Dual Token Telemetry.

### Core CLI Options (`inkling`)

| Option | Description |
|---|---|
| `-m, --model PATH` | GGUF model path (default: LM Studio's `Inkling-2-Mini-Q4_K_M.gguf`) |
| `-p, --prompt TEXT` / `-f FILE` | Prompt text or file (`run` / `bench` / `batch`) |
| `--prompts FILE` | One prompt per line for independent `batch` / `bench-batch` |
| `-b, --batch N` | Batch size or comma-separated sweep (e.g. `-b 64`, `-b 1,8,32,64`) |
| `--shared` / `--independent` | Shared-prefix cascade attention vs. independent KV caches (`batch`) |
| `-n N` | Max generated tokens (default `2048`, `bench` `128`) |
| `-c, --ctx N` | Max context length (default `16384`; `bench-256k` uses `>= 262144`) |
| `--effort LEVEL` | Reasoning effort: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `0..0.99` |
| `--temp` / `--top-p` / `--top-k` / `--seed` | Sampling parameters (defaults: `1.0` / `0.95` / `64`; `--temp 0` or `--greedy` is GPU-pipelined) |
| `--kv-bits 3\|4` / `--topk-pages N` | `Kv256kPool` quantization & Top-K far-page count (`bench-256k`, `needle-10m`) |
| `--kv8` / `--kv-disk` | Legacy dense cache modes: 8-bit KV (50% RAM, approx) or bit-exact fp16 SSD streaming |
| `--max-mem GB` | Hard cap on batch working GPU memory in GiB |
| `--system TEXT` / `--hide-thinking` / `--raw` | System prompt / hide reasoning / skip chat template |

Environment overrides: `INK_CHUNK=N` (max prefill batch, default `4096`), `INK_NO_HGEMM=1` (disable register-epilogue tensor GEMM), `INK_NO_TENSOR=1` (simdgroup matrices instead of neural accelerators).

---

## Benchmarks

All numbers measured on **Apple M5 Pro (48 GB)** with `Inkling-2-Mini-Q4_K_M.gguf`.

### 1. Stock Dense Engine vs. `llama.cpp`

| Workload | LM Studio (`llama.cpp` 2.80.1) | `llama.cpp` master | **erisinky** | Speedup vs LM Studio |
|---|---:|---:|---:|---:|
| Decode, short context (greedy) | 33.1 tok/s | 32.7 tok/s | **63.0 tok/s** | **1.90×** |
| Decode @ 2.4K context | 31.7 tok/s | — | **58.2 tok/s** | **1.84×** |
| Prefill, 2,394-token prompt | 381 tok/s | 409 tok/s (`pp2048`) | **2,200 tok/s** | **5.8×** |

Sampling (`--temp > 0`) adds ~3% decode overhead. Parity vs `llama.cpp` (`tools/ref_logits.cpp`): `61/61` argmax on decode (worst KL `0.0026`), `25/25` after 2,379-token prefill (worst KL `0.0069`), token-for-token tokenizer parity on 74K tokens.

### 2. Real-World Multi-File Coding Agent (`matrix_v2`, 6-Requirement Rust Task)

| Run | Mode | Wall | TTFT Sum (Max) | Decode (`start → end`) | Rubric |
|---|---|---:|---:|---:|---:|
| `M2` | Dense 128K control (`--no-ikv`) | 402.3 s | 39.7 s (23.9 s) | **39.4 tok/s** (62.2 → 37.6) | 5.5 / 6 |
| `M4` | Sparse 128K `KV4`-pool control (`--no-ikv`) | 467.7 s | 112.2 s (53.4 s) | 31.2 tok/s (45.5 → 30.7) | **6.0 / 6** |
| `M3` | Sparse 128K `KV4`-pool + `K3V2`-`.ikv` + slice | **366.4 s** | 28.7 s (8.4 s) | 31.1 tok/s (45.6 → 30.9) | 5.0 / 6 |
| `M9` | Sparse 128K `KV4`-pool + **`KV4`-`.ikv`** + slice | 488.1 s | **24.9 s** (4.0 s) | 29.2 tok/s (45.5 → 29.3) | **6.0 / 6** |
| `M7` | Sparse **256K** (`P64·K128`) + `K3V2`-`.ikv` + slice | 489.5 s | 32.9 s (4.3 s) | 29.1 tok/s (38.8 → 28.8) | 5.5 / 6 |

On the exact same 3-file read (`46.4K` tokens), `.ikv` ingested everything in `1.76 s` vs `94.57 s` raw Sparse prefill (**`54×` faster**).

### 3. Multi-File Document Search (`--notools`, 47-File Ziz Archive, 243.4K Tokens)

| Run | Mode | Files / Data Read | `.ikv` Stitch | TTFT Sum (Max) | Total Read Wait | Wall | Grade |
|---|---|---:|---:|---:|---:|---:|---:|
| `M3` | Sparse 256K `KV4`-pool + `K3V2`-`.ikv` | 11 (72.3K tok) | **1.19 s** | **4.71 s** (0.87 s) | **5.90 s** | **143.7 s** | B- (78) |
| `M9` | Sparse 256K `KV4`-pool + **`KV4`-`.ikv`** | **20 (118.1K tok)** | **1.99 s** | **7.36 s** (1.08 s) | **9.35 s** | 214.2 s | **A- (89)** |
| `M4` | Sparse 256K `KV4`-pool control (`--no-ikv`) | 9 (264 KiB raw) | — | 128.64 s (90.6 s) | 128.64 s | 278.7 s | A (93) |
| `M2` | Dense 256K FP16 control (`--no-ikv`) | 19 (355 KiB raw) | — | 84.96 s (40.7 s) | 84.96 s | 338.3 s | A (94) |

On the single 42.3K-token glossary, `.ikv` stitched in **`410 ms`** vs `40.7 s` Dense raw (**`~99×` faster**) and `90.6 s` Sparse raw.

### 4. Extreme Context & Multi-Stream Scaling

- **25-task multi-hop codebase + RULER eval:** Sparse paging + CoT (`effort=0.1`) scored **`25/25 (100%)`** while pruning 88%–91% of far pages on every token.
- **Multi-stream 256K (`bench-256k`):** `~172 MiB` RAM/stream; **64 concurrent 256K streams** fit in **`10.5 GiB`** at **`337 tok/s`** aggregate; **128 streams** fit in **`11.9 GiB`** (vs `~1 TB` FP16).
- **1M-token needle (`needle-10m`, `3.37 GiB` RAM vs `32 GiB` FP16):** **`3/3 (100%)` exact verbatim recall** at 10% / 50% / 90% depth (`cent = OFF`) at **`19.7–19.8 tok/s`**.
- **10M-token needle (`needle-10m`, `5.21 GiB` RAM vs `320 GiB` FP16):** Runs live at **`5.0 tok/s`** (`cent = ON`) / **`7.8–8.1 tok/s`** (`cent = OFF`); `WHT-2bit-sketch` ranked the single needle page **`#1 / 81,918` (top `0.0012%`)** at all three depths (`9.44M`, `5.24M`, `1.05M` tokens back), recovering prefix digits (`CRANE-749`, `RIVER-382`, `DENVER-44`) under 80× tiled distractors.

---

## Model Architecture (`inkling2_loop`)

`inkling2_loop` executes **24 physical layers** as **48 layer executions** by looping layers 6–17 three times (48 distinct KV and convolution cache slots):

- **Block attention residuals:** Layer inputs are a learned softmax mixture over RMS-normalized earlier block outputs.
- **No RoPE:** Learned query-dependent 16-dim relative-position bias (`r·B[d]`, active only within `1,024` tokens, zero beyond), per-head input-projected attention sinks, QK norm clipping, and native `log(pos)` attention temperature scaling.
- **Causal short convolutions (width 4):** Recurrent state on K/V (input-dependent taps on full-attention layers) and residual branch outputs (`1.83 MiB` FP16 total).
- **5:1 SWA / full attention:** 40 sliding-window (`1,024`-token) layers + 8 global full-attention layers, with sigmoid output gating.
- **128-expert top-8 MoE:** Sigmoid routing with selection bias and SiTU activation (`β·tanh(g/β)·σ(g) · λ·tanh(u/λ)`).
- **Loop controllers:** Low-rank gating in passes 2 and 3 driven by messages mixed from the previous pass.

---

## Repository Layout

```text
src/main.rs             CLI entry point (chat, run, bench, batch, bench-256k, needle-10m, kv-share-probe, serve, check)
src/lib.rs              Library exports consumed by companion crates (e.g. ../inkling-ikv)
src/engine.rs           Weight binding, demand-sized GPU memory layout, fused decode & prefill schedules
src/session.rs          Single-stream context management, chunked prefill, pipelined decode, output rendering
src/kv256k.rs           Kv256kPool: WHT-KV4/K3V2 quantization, F_NOCACHE disk tiering, WHT-2bit-sketch Top-K paging, 1M/10M needle harness
src/kv_share.rs         Zero-RoPE verbatim KV paste, TokenDance Master-Mirror sparse diffs, KVCOMM anchor probes
src/kv_probe.rs         KV quantization, DC centering, and pruning diagnostic probes
src/batch.rs            Paged multi-sequence batch engine with shared-prefix cascade attention & GPU sampler
src/offload.rs          Experimental disk-backed MoE expert offloader
src/multistream.rs      7-channel Multi-Stream inference (.imsa adapters)
src/persona.rs          Dual-Persona LoRA serving
src/webui.rs            Interactive WebUI server
src/kernels.metal       Fused single-stream decode Metal kernels
src/prefill.metal       Batched prefill and M5 Metal 4 neural-accelerator matmul2d kernels
src/batch.metal         Batched paged-KV decode kernels
src/batch_opt.metal     Optimized B=1/batched Kv256kPool WHT quantize, sketch-scan, and sparse-attention kernels
src/kv_share.metal      Fused TokenDance Master-Mirror + sparse-diff attention kernels
src/bench_kernels.metal Matvec & GEMM micro-benchmark kernels
src/metal.rs            Minimal objc2-metal compute wrapper
src/gguf.rs             Zero-copy mmapped GGUF reader
src/tokenizer.rs        Byte-level BPE and GPT-4o pre-tokenizer
src/sampler.rs          CPU sampler (softcap, top-k, temperature, top-p)
tools/ref_logits.cpp    llama.cpp reference-logit dumper for parity verification
```

---

## License

Licensed under the **GNU General Public License v3.0 or later (`GPL-3.0-or-later`)** — Copyright (C) 2026 Eris. See [LICENSE](LICENSE) for the full text.

This license covers the engine source code only; Inkling 2 Mini model weights are not included in this repository and remain under their own terms.
