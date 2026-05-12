# DSv4-Flash on Spark — Strategic Checkpoint, 2026-05-12

This is the full analysis behind the headline numbers in the
[`README`](../README.md). The intended audience is anyone who has been
considering writing their own port, fork, or wrapper for DSv4-Flash on
DGX Spark — and wants the empirical answer to "is the donor good enough?"
before committing to that work.

Paired companion: [`METAL_VS_CUDA.md`](METAL_VS_CUDA.md) covers the implementation
side — how the CUDA backend actually realises the same `ds4_gpu.h` contract that
the Metal backend does, kernel by kernel. This document is the *what*; that one
is the *why*.

**One-line answer:** Yes. The donor's native CUDA path on Spark is
at ~95 % of the memory-bandwidth roofline at steady state. There is
no easy 2–4× decode speedup left on the table for this quant + this
hardware. The decision to write your own implementation should be
driven by features (multi-model hosting, custom serving, ecosystem
integration), not by performance headroom.

---

## 1. What changed

The donor `antirez/ds4` was Metal-only until **2026-05-11 07:27 UTC**,
when commit [`48beef8 CUDA support`](https://github.com/antirez/ds4/commit/48beef8)
landed a 9,666-LOC `ds4_cuda.cu` (106 kernels) behind a unified
`ds4_gpu.h` API. Same-day follow-ups added:

| Commit | Summary |
|---|---|
| `48beef8` | CUDA backend (9,666 LOC, 106 kernels) |
| `f5f414d` | CPU support refactor (`DS4_NO_GPU` build path) |
| `0ac5df3` | Multi-backend graph executor (Metal/CUDA/CPU) |
| `0c1c023` | `ds4-bench` incremental throughput benchmark |
| `66afd65` | `CUDA_ARCH` Makefile knob (first-class non-default targets) |
| `ae302c2` | Project renamed to **DwarfStar 4** (DS4 abbreviation kept) |

This made it possible to run DSv4-Flash on a Spark via the donor's own
runtime, without a port, without a wrapper, and without patching CUDA
dispatch. The `Makefile` defaults `CUDA_ARCH` to `native` and accepts any
`sm_NNN` override; `make CUDA_ARCH=sm_121` is GB10-correct out of the box.

At the implementation level the CUDA path links `-lcudart -lcublas`, uses
`cudaMallocManaged` for runtime tensors, mounts the 80 GiB GGUF via a
three-tier strategy (host-register → per-range pinning → 64 MiB chunked
`cudaMemcpy`), and pre-converts dense `Q8_0` weights into a device-side
`F16` cache so `cublasGemmEx` can run prefill matmuls on tensor cores. Routed
expert weights (IQ2_XXS / Q2_K) are *not* pre-converted; they stay quantised
and are dequantised inline in hand-written kernels. The full breakdown,
including the differences from the Metal backend, lives in
[`METAL_VS_CUDA.md`](METAL_VS_CUDA.md).

The donor's README still labels the project "alpha quality" overall;
the MTP path is specifically called out as experimental.

## 2. Hardware ceilings (measured on Spark)

| Probe | Bandwidth | Notes |
|---|---:|---|
| `nvbandwidth` H2D / D2H CE | 59 GB/s | Copy-engine path |
| `nvbandwidth` device_local_copy (CE) | 111 GB/s | Single-device CE |
| `bench/bw_bench.cu` copy (R+W) | **215 GB/s** | Kernel-driven memcpy |
| `bench/bw_bench.cu` read-only | **227 GB/s** | Pure read throughput |
| Published GB10 LPDDR5X peak | ~273 GB/s | 256-bit × 9400 MT/s theoretical |

**Effective kernel-accessible bandwidth: ~225 GB/s** = 82 % of theoretical
peak, which is normal for real workloads on LPDDR.

## 3. ds4 throughput (measured)

### 3a. `ds4-bench` (direct CLI)

| ctx | prefill t/s | decode t/s | KV |
|---:|---:|---:|---:|
| 2,048 | 287.8 | 13.50 | 52 MB |
| 6,144 | 332.6 | 13.47 | 109 MB |
| 10,240 | 300.9 | 13.14 | 165 MB |
| 14,336 | 303.3 | 13.00 | 221 MB |
| 16,384 | 290.9 | 12.92 | 250 MB |

### 3b. `llama-benchy` against `ds4-server` (OpenAI v1 HTTP)

| test | t/s | peak | ttfr (ms) |
|---|---:|---:|---:|
| pp2048 | 364.5 ± 2.6 | — | 5890 |
| tg32 @ d=0 | 29.2 ± 1.4 | 31.0 | — |
| tg128 @ d=0 | 28.0 ± 1.0 | 34.0 | — |
| tg512 @ d=0 | 22.8 ± 2.6 | 33.3 | — |
| pp2048 @ d=4k | 339.5 | — | 18712 |
| tg32 / tg128 / tg512 @ d=4k | 27.8 / 25.9 / 23.3 | — | — |
| pp2048 @ d=16k | 310.7 | — | 61401 |
| tg32 / tg128 / tg512 @ d=16k | 24.1 / 24.5 / 24.2 | — | — |

### 3c. Methodology gap explained

`ds4-bench` and `ds4-server`'s log report `total_gen / total_decode_wall`,
**including** the first-token cost (which is ~1.0–1.3 s post-prefill on
this model). `llama-benchy` reports `(N − 1) / (t_last − t_first)`,
**excluding** that overhead. For the 18k-context tg32 request:

- ds4-server log: `gen=32 ... avg=12.94 t/s 2.473s`
- First token: ~1.20 s
- Remaining 31 tokens: 1.27 s → 24.4 t/s
- llama-benchy: 24.14 ± 0.35 t/s (matches the steady-state slice)

Both are correct measurements. Pick by use case:

| Use case | Rate that dominates user perception |
|---|---|
| Interactive chat, agents, short replies | **~13 t/s** (first-token weight) |
| Long-form generation | **~25–29 t/s** (steady-state) |
| 16k-context prefill | **~1 minute** to first content token |

## 4. Bytes per token (from safetensors index)

Aggregated across all 17 shards of the donor-converted safetensors mirror
(88.4 GB total, layout 1:1 with the GGUF that the donor itself loads):

| Bucket | Total | Active per token |
|---|---:|---|
| Routed experts (IQ2_XXS gate/up + Q2_K down) | 78.28 GB | 6/256 active → **1.83 GB** |
| MLA attention + indexer + compressor | 7.05 GB | **all active** |
| Embed / head / final norm | 2.12 GB | **~1.0 GB** (head projection) |
| Shared expert (1 per MoE layer) | 0.74 GB | **0.74 GB** |
| MTP / HC / scalar gates / biases | 0.30 GB | **~0.22 GB** |
| KV cache reads @ 16k | — | **~0.25 GB** |

The model is dominated by routed experts (88.6 % of disk bytes), but
they're only 17 % of *active* bytes per token because of MoE routing.
The always-active path (~9 GB nominal) plus the active expert slice
(~1.8 GB) plus KV (~0.25 GB) sums to ~11 GB by static accounting.

**Effective bytes per token at steady state, back-derived from measurement:**

```
225 GB/s ÷ 28 t/s = ~8.0 GB / token
```

This is below the 11 GB static estimate — meaning some weights cache-hit
across layers, or fused kernels avoid re-reading, or both. Either way, the
donor's kernels are tighter than naïve bucketing predicts.

One thing worth noting up front: the decode roofline is read in the *stored*
quantisation, not the cached F16 copy. The donor pre-converts dense `Q8_0`
weights to an F16 cache at startup so cuBLAS can use tensor cores for
prefill (multi-token) GEMM — but the decode path (`n_tok=1`) skips that
cache and goes through hand-written Q8_0 matvecs where the stored bytes are
what gets read. See [`METAL_VS_CUDA.md`](METAL_VS_CUDA.md) §5 for the full
prefill-vs-decode dispatch logic. The 300–360 t/s prefill rate and ~28 t/s
decode rate are taking different routes through the matmul stack on the
same model in the same process.

## 5. Roofline

| Quantity | Value | Of |
|---|---:|---:|
| Kernel-effective BW | 225 GB/s | (82 % of theoretical) |
| Effective bytes / token | ~8.0 GB | — |
| **Strict roofline** | 225 / 8 = **28.1 t/s** | — |
| Measured steady-state | ~28 t/s | **~95–100 % of roofline** |
| First-token-inclusive rate | ~13 t/s | (different metric) |

**Steady-state decode is essentially saturated.** This means:

- **No easy decode wins on this hardware + this quant.**
- Going faster requires changing one of the three roofline variables:
  - **Bandwidth**: only achievable by switching hardware.
  - **Bytes per token**: tighter quant (FP4 dense, 1.5-bit experts) — quality tradeoff.
  - **Tokens per byte-read**: batched serving for multi-user (amortizes weight reads across users) or speculative decode (multiple tokens per weight-pass — but see §6, current MTP doesn't deliver).

The 13 t/s number that dominates *interactive* feel is **first-token
latency**, not steady-state. It's a separately-tractable optimization:
warm-cache reuse across turns, persistent KV across thinking/answer phases,
prefetch.

## 5.5. `ds4-server` does not batch concurrent requests

The OpenAI v1 server processes incoming requests strictly serially. Verified
via llama-benchy's `--concurrency 2` against a 35k-token prompt:

```
14:10:36 request 1 starts (35987 tokens)
14:13:15 request 1 finishes (159s)
14:13:15 request 2 starts (immediately on completion)
14:16:?? request 2 finishes
```

Two `concurrency=2` requests take ~2× one `concurrency=1` request. There is
no parallel batching in the C+CUDA path; the donor's architecture is
single-session. For multi-user serving on a single Spark, ds4 is not the
right runtime — vLLM / SGLang with paged-attention batching are. ds4 is
explicitly intentionally narrow (per donor README: "not a generic GGUF
runner, not a wrapper around another runtime, not a framework").

## 6. MTP (speculative decode) — methodology-dependent

The donor ships `--mtp <draft.gguf> --mtp-draft N`. The MTP support model
is a separate 3.6 GiB Q4K/Q8_0/F32 GGUF.

**Three draft depths, long-generation prompt (QuickSort essay, ~600 tokens):**

| Config | decode t/s |
|---|---:|
| no MTP | 13.5 (bench) / 14.88 (is_prime) |
| `--mtp-draft 1` | 13.81 |
| `--mtp-draft 2` | 13.62 |
| `--mtp-draft 4` | 13.63 |

All three within noise of each other and slightly below no-MTP. This rules
out "draft depth choice" as the issue — none of them help.

**High-predictability prompts at `--mtp-draft 2` (would maximize acceptance):**

| Prompt | no-MTP | MTP-2 | Δ |
|---|---:|---:|---:|
| Count 1 → 60 (deterministic) | 15.13 | 14.27 | **−5.7 %** |
| Alphabets (English / NATO / Greek) | 15.02 | 14.32 | −4.7 % |
| Declaration of Independence + 10 Presidents | 14.64 | 14.37 | −1.8 % |
| 27 EU capitals | 14.92 | 14.48 | −2.9 % |
| **mean** | **14.93** | **14.36** | **−3.8 %** |

**The "counting 1 → 60" case is the load-bearing test.** Every next token
is fully deterministic; the draft model's acceptance rate should be ~100 %.
MTP still makes it slower. That rules out low acceptance rate as the
explanation. The cost is in the draft-inference + verifier mechanics
themselves, not in rejected drafts.

Consistent with the donor's README labeling the MTP path "alpha quality."
Also consistent with the §5 roofline picture: steady-state decode is
already at ~95 % of the bandwidth ceiling, so even a hypothetical
zero-overhead MTP couldn't deliver large gains here.

If you need higher throughput on Spark, **MTP today is not the answer**.
The available wins are (a) batched multi-user serving, (b) tighter quant,
or (c) different hardware.

### 6a. MTP via llama-benchy (steady-state, first-token excluded)

llama-benchy methodology isolates steady-state decode by excluding the
first-token cost. Same hardware, 3 runs each, d=8192 tg=512:

| Config | tg512 t/s | peak t/s | prefill t/s |
|---|---:|---:|---:|
| no MTP | 22.14 ± 2.57 | **28.33** ± 1.25 | 328.09 ± 0.51 |
| MTP draft=2 | 23.61 ± 0.48 | **28.33** ± 0.47 | 328.49 ± 0.68 |
| Δ | +6.6 % (within noise) | identical | identical |

**Key observations:**

1. **Peak t/s is bit-identical** (28.33 t/s in both). The steady-state
   ceiling is the bandwidth roof, not the MTP path.
2. **Variance is much tighter under MTP** (±0.48 vs ±2.57). The draft+verify
   cadence smooths per-token jitter that hits the no-MTP path.
3. **This reconciles the CLI-vs-benchy gap.** Our CLI tests (first-token
   inclusive) showed MTP as a 3–6 % regression. The cost is MTP's
   per-request setup overhead, which lands on the first token. Llama-benchy
   excludes that; under llama-benchy methodology MTP is neutral.

So the headline shifts slightly: **MTP doesn't regress steady-state decode;
it just doesn't lift it either, because steady-state is already at the
bandwidth roof.** The only thing MTP buys you in practice on Spark today
is decode-rate stability, at the cost of a few hundred ms additional TTFT.

## 7. When ds4 is the right answer

| Goal | Recommendation |
|---|---|
| Run DSv4-Flash on Spark, single-user, interactive | **ds4-server**, no port needed |
| Same, but agent-integrated (tools, KV-checkpoint replay) | **ds4-server** — has rax-tree exact tool replay built in |
| Serve many users on one Spark | **vLLM** (not ds4 — donor explicitly intentionally narrow) |
| Host non-DSv4 models alongside | **vLLM or SGLang** — ds4 is DSv4-Flash-only by design |
| Maximum decode throughput | Tighter quant or different hardware; not a software win available on Spark |
| Custom serving / steering / experiments | ds4's `--dir-steering-*` flags expose direction-vector steering; or fork (see [`METAL_VS_CUDA.md`](METAL_VS_CUDA.md) for the implementation surface) |

## 8. When a separate port is the right answer

Writing your own implementation (Rust, Triton, vLLM plugin, etc.) is
justified if and only if you need at least one of:

1. **Ecosystem integration** the donor doesn't expose (specific Rust
   runtime, multi-model dispatch, custom serving protocols, observability).
2. **Custom features** beyond DSv4-Flash specifics (other models on the
   same stack, experimental architecture variants).
3. **Strategic independence** from a single upstream.
4. **Learning vehicle** — porting clarifies architecture in a way no
   consumer of a black-box server gets.

Before committing to any of those, read [`METAL_VS_CUDA.md`](METAL_VS_CUDA.md).
It defines the kernel surface, command lifecycle, and quantisation handling
you would have to reimplement (or wrap) to match the donor's behaviour.

The bad reasons to port:

- "I'll get better throughput." You won't, not on this hardware + this
  quant. The donor is at ~95 % of the bandwidth roofline.
- "MTP will close the gap." It currently doesn't, and §5 says it can't
  close much even with a perfect implementation.
- "Sunk cost." Investment is not justification.

## 9. Open questions (not yet measured)

These could change the picture if measured. Listed in order of how
likely they are to flip a decision:

- **Full-context (>=128k) decode rate.** We measured up to 16k. The
  model claims 1M context; the donor docs claim it's been tested at
  250k on 96 GB Macs. If decode falls off a cliff at 64–128k due to
  KV-bandwidth contention, custom KV management may matter.
- **Concurrency / batched serving.** ds4 is designed as a single-session
  server. If you need many concurrent users at one Spark, you need
  either a different runtime or a fork of ds4 with batched dispatch.
- **MTP improvement upstream.** The donor may improve MTP. Re-run the
  predictable-prompts MTP probe periodically.
- **Tighter quant.** A 1.5-bit or FP4-dense recipe could push bytes/token
  down and lift the roofline. Quality tradeoff to be measured.

## 10. Reproducing this analysis

```bash
# Install + smoke test
./install.sh --with-mtp

# Start server
./scripts/start-server.sh --port 8000 --ctx 32768

# llama-benchy sweep
./scripts/run-bench.sh --pp 2048 --tg 32 128 512 --depth 0 4096 16384

# Memory bandwidth ceiling
/usr/local/cuda/bin/nvcc -O3 -arch=sm_121 bench/bw_bench.cu -o /tmp/bw_bench
/tmp/bw_bench 8192

# MTP comparison (start a separate ds4-server with --with-mtp)
./scripts/start-server.sh --port 8001 --with-mtp --draft 2
./scripts/run-bench.sh --port 8001 --depth 0 --tg 128 --runs 3
```

All numbers in this writeup come from one DGX Spark
(`gn100-7710.local`, GB10 sm_121, CUDA 13.0.88) over a single 2-hour
session on 2026-05-12.
