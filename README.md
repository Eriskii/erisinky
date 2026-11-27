# inkling

A Rust + hand-fused Metal 4 inference engine for **Thinking Machines' Inkling 2 Mini** (`inkling2_loop`, 26.4B total / 3.4B active MoE) on Apple Silicon, reading GGUF weights directly via zero-copy `mmap`.

## Benchmarks

Measured on **Apple M5 Pro (48 GB)** with `Inkling-2-Mini-Q4_K_M.gguf`, greedy decoding:

| Workload | LM Studio (`llama.cpp` 2.80.1) | `llama.cpp` master | **inkling** | Speedup vs LM Studio |
|---|---:|---:|---:|---:|
| Decode, short context | 33.1 tok/s | 32.7 tok/s | **63.0 tok/s** | **1.90x** |
| Decode @ 2.4K context | 31.7 tok/s | — | **58.2 tok/s** | **1.84x** |
| Prefill, 2,394-token prompt | 381 tok/s | 409 tok/s (`pp2048`) | **2,200 tok/s** | **5.8x** |

Sampling (`--temp > 0`) adds ~3% decode overhead for one CPU round-trip per token.

**Correctness.** Checked position-by-position against `llama.cpp` (`tools/ref_logits.cpp`): 61/61 argmax matches on decode (max logit diff ~0.005, worst KL 0.0026), 25/25 after a 2,379-token prefill (worst KL 0.0069), and 41/41 after a short prefill (worst KL 0.0001). The tokenizer matches `llama-tokenize` token-for-token on 74K tokens of code and markdown.

## Build & Usage

Requires macOS on Apple Silicon with Metal 4 (M5 neural accelerators are used for prefill GEMMs and prefill attention).

```sh
cargo build --release

./target/release/inkling chat                         # interactive multi-turn chat (KV cache reused)
./target/release/inkling run -p "Explain RoPE" --effort low
./target/release/inkling run -f prompt.txt --greedy   # fully GPU-pipelined greedy decode
./target/release/inkling bench -f prompt.txt -n 256   # prefill + decode throughput
./target/release/inkling check --ref <dump>           # verify logits against tools/ref_logits.cpp
```

### Options

| Option | Description |
|---|---|
| `-m, --model PATH` | GGUF model path (default: LM Studio's `Inkling-2-Mini-Q4_K_M.gguf`) |
| `-p, --prompt TEXT` / `-f FILE` | Prompt text or file (`run` / `bench`) |
| `-n N` | Max generated tokens (default `2048`, `bench` `128`) |
| `-c, --ctx N` | Max context length (default `16384`) |
| `--effort LEVEL` | Reasoning effort: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `0..0.99` (default `high`) |
| `--temp` / `--top-p` / `--top-k` / `--seed` | Sampling parameters (defaults: `1.0` / `0.95` / `64`; `--temp 0` or `--greedy` runs GPU-pipelined) |
| `--system TEXT` | System prompt |
| `--hide-thinking` / `--raw` | Hide reasoning output / skip chat template and print special tokens |
| `--kv8` | 8-bit quantized K/V cache (half cache RAM, same speed, approximate) |
| `--kv-disk` | Exact fp16 K/V cache with older half streamed from SSD during decode (half cache RAM, bit-identical) |

## Model Architecture

`inkling2_loop` runs **24 physical layers** as **48 layer executions** by looping layers 6–17 three times (48 distinct KV and convolution cache slots):

- **Block attention residuals:** layer inputs are a learned softmax mixture over RMS-normalized earlier block outputs.
- **No RoPE:** learned query-dependent 16-dim relative-position bias (`r·B[d]`), per-head input-projected **attention sinks**, QK norm clipping, and log-scaled temperature for long contexts.
- **Causal short convolutions (width 4):** recurrent state on K/V (input-dependent taps on full-attention layers) and on both residual branch outputs.
- **5:1 SWA / full attention:** 1,024-token sliding window on 5 of every 6 layers, with sigmoid attention output gating.
- **128-expert top-8 MoE:** sigmoid routing with selection bias and **SiTU** activation (`β·tanh(g/β)·σ(g) · λ·tanh(u/λ)`).
- **Loop controllers:** low-rank gating in loop passes 2 and 3 driven by messages mixed from the previous pass.

## Design & Implementation

- **GPU-resident decode:** One command buffer per token (~530 dispatches) with routing/top-k, attention split-combine, norms, and argmax on GPU. Greedy decoding pipelines two tokens in flight GPU-to-GPU with no CPU round-trips.
- **Fused Metal kernels:** Single-kernel depth-mix, segmented Q/K/V/gate/rel/sink/tap projection, fused K/V-conv + QK-clip + relative-bias + sink decode attention, post-conv/residual matvec epilogues, and fused SiTU + weighted-sum MoE kernels.
- **Bandwidth-tuned quantized matvecs:** Direct cache-read `Q4_K`/`Q6_K`/`Q8_0`/`Q5_0`/`bf16` matvecs (~285 GB/s isolated on M5 Pro), barrier-free shuffle-reduced decode attention with adaptive split-K, bf16 storage for dense router/controller weights, and one-time GPU pre-wiring of mmapped weight pages at load.
- **M5 neural-accelerator prefill:** Whole-prompt batching (up to 4,096 tokens) via MetalPerformancePrimitives `matmul2d` fp16 tensor ops with double-buffered 64×32 weight staging, register-resident accumulators/epilogues, cache-friendly y-major grid dispatch, GPU counting-sort + indirect dispatch for ragged MoE tiles, and tail-only evaluation on the final layer.
- **Demand-sized GPU memory:** The fp16 K/V cache grows lazily in 64-token steps (40 sliding-window rings capped at 1,024 tokens + 8 full-attention slots at 32 KiB/token: ~24 MiB in short chat, 240 MiB at 2.4K, 672 MiB at 16K). The prefill activation arena (~0.4 MiB/token) is allocated only for prefill and freed before decode (~0.38 GB non-weight footprint in short chat).

### Optional K/V Cache Modes

| Mode | Cache RAM (2.5K / 5.5K / 15K) | Decode speed (2.5K / 5.5K / 15K) | Exactness |
|---|---:|---:|---|
| Default (`fp16`) | 246 / 336 / 642 MiB | 58.7 / 57.4 / 53.2 tok/s | Exact reference path |
| `--kv8` | 50% of `fp16` | Unchanged (native int8 tensor ops) | Approximate (KL 0.0030 mean / 0.023 max @ 2.4K; 66/66 argmax) |
| `--kv-disk` | 118 / 164 / 314 MiB | 36.3 / 30.8 / 17.8 tok/s | Bit-identical to `fp16` (older half streamed from uncached temp file) |

## Layout

```text
src/main.rs             CLI (chat, run, bench, check) and profiling/diagnostic subcommands
src/session.rs          Context management, chunked prefill, pipelined decode, output rendering
src/engine.rs           Weight binding, GPU memory layout, decode and prefill schedules
src/kernels.metal       Fused decode Metal kernels
src/prefill.metal       Batched prefill and M5 tensor-op Metal kernels
src/bench_kernels.metal Matvec micro-benchmark kernels
src/metal.rs            Minimal objc2-metal compute wrapper
src/gguf.rs             Zero-copy mmapped GGUF reader
src/tokenizer.rs        Byte-level BPE and GPT-4o pre-tokenizer
src/unicode_tables.rs   Generated Unicode category tables for pre-tokenization
src/sampler.rs          CPU sampler (softcap, top-k, temperature, top-p)
tools/ref_logits.cpp    llama.cpp reference-logit dumper for validation
```

Environment overrides: `INK_CHUNK=N` (max prefill batch, default `4096`), `INK_NO_HGEMM=1` (fallback tensor path without register epilogues), `INK_NO_TENSOR=1` (simdgroup matrices instead of neural accelerators).

## License

Licensed under the **GNU General Public License v3.0 or later (`GPL-3.0-or-later`)** — Copyright (C) 2026 Eris. See [LICENSE](LICENSE) for the full text.

This license covers the engine source code only; Inkling 2 Mini model weights are not included in this repository and remain under their own terms.
