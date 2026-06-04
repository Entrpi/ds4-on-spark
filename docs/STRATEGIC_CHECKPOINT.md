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

The donor's README labels the project "alpha quality" overall and the
MTP path specifically as experimental. We empirically traced the lack of
MTP speedup on CUDA to a single quant-format gap in `ds4_cuda.cu` — the
MoE kernel rejects the MTP draft GGUF's Q4_K-quantised experts — and
scoped the fix in [MTP_PARITY_GAP.md](MTP_PARITY_GAP.md). The C-side
speculative decode state machine, MTP weight binding, and batched
verifier are all sound; the gap is one missing CUDA dispatch case.
Metal is unaffected and has the parity dispatch already.

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
  - **Tokens per byte-read**: batched serving for multi-user (amortizes weight reads across users) or speculative decode (multiple tokens per weight-pass — but see §6, current MTP doesn't deliver *on CUDA today* for a fixable reason — [MTP_PARITY_GAP.md](MTP_PARITY_GAP.md)).

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

## 6. MTP (speculative decode) — works and bit-exact, but a single-stream loss

The donor ships `--mtp <draft.gguf> --mtp-draft N` with a separate 3.6 GiB
Q4K/Q8_0/F32 draft GGUF.

**Bottom line up front:** MTP now works on CUDA and is **bit-exact to non-MTP
greedy decode** (the earlier Q4_K draft-kernel reject is closed at `5625a99`),
but it is a **net throughput loss on single-stream decode**, so the strategic
answer for a single-user Spark is unchanged: **run no-MTP**. MTP is a
batched-serving feature, not a single-stream one. Detail in
[MTP_PARITY_GAP.md](MTP_PARITY_GAP.md).

**What changed since the earlier checkpoint.** This section previously reported
MTP as a silent no-op (the draft kernel never produced a token) and projected a
vLLM-comparable "+35–80 %" lift once a Q4_K MoE kernel landed. That kernel work
landed; the projection was wrong for single-stream. Corrected:

- **Drafts are produced** — `DS4_MTP_PROBE=1 … | grep -c "mtp probe draft
  failed"` prints **0** (was 58). The Q4_K MoE path was added to
  `routed_moe_launch`.
- **The verifier is bit-exact** — it runs the same mmq kernels as base decode
  and defaults to a sequential 2-row verify, so MTP output is byte-identical to
  no-MTP, eager and CUDA-graph-captured, on both arches. No-MTP determinism
  goldens unchanged.
- **But it's a net loss** — the exact verifier sweeps the weight-bound
  projections once *per draft row* (verify ≈ 2× a decode, architecture-
  independent), and acceptance saturates at ~1.4 (prose) / ~2.0 (numeric) suffix
  tokens. Measured A/B (captured, accept-gate off): prose **−21 %**, numeric
  **−15 %** vs no-MTP. Even at 100 % acceptance it loses (57 vs 51 ms/tok).

**Why the earlier projection was wrong — and why MTP still matters.**
Perf-positive MTP on DeepSeek-V4-Flash *is* real: vLLM runs it in production
(PR #43447) at acceptance ~1.8, K=3, with gains that grow with batch size
(+30.9 % at BS=1 → +88.4 % at BS=4). The miss is structural to ds4's serving
shape, not the model. Real spec-decode does one **batched** K-position verify (a
single weight sweep across the draft block, *lossless* but not bit-identical)
and amortizes it across a **request batch**. ds4's single-stream (BS=1),
per-row, bit-identical verify gets neither axis — it pays ~2 weight sweeps for
~1.6 accepted tokens.

So the strategic picture for "should you wait for MTP or work around it?":

- **Single-user interactive:** run no-MTP. ~19 t/s fastest-path decode at short
  context is your rate today; MTP does not improve it (and the accept-gate
  auto-disables it on prose, so with default flags it's a wash). Don't enable
  MTP for a single stream.
- **Higher throughput today:** batched multi-user serving via a different
  runtime (vLLM/SGLang), tighter quant, or different hardware.
- **Where the real MTP win is (contributor):** batched serving for ds4 —
  cross-request amortization + a lossless K-position verify over the multi-token
  prefill path. A single-stream weight-shared verifier was prototyped and is
  perf-neutral/negative (bit-exact, but the routed MoE batches to zero gain — the
  two draft rows route to disjoint experts). That's why the lever is the
  serving shape, not another kernel.

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
- "MTP will close the gap." It currently doesn't on CUDA, for a
  fixable kernel-level reason ([MTP_PARITY_GAP.md](MTP_PARITY_GAP.md));
  contributing the fix upstream is a much smaller investment than
  forking the engine. And even with MTP working the §5 roofline caps
  the steady-state win at ~75 % over baseline, not 10×.
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
- **MTP under batched serving.** The Q4_K parity gap is closed and MTP is
  bit-exact, but it's a single-stream loss (§6) — the real lift needs
  batched serving (cross-request amortization + a lossless K-position verify
  over the prefill path), not another single-stream kernel. A single-stream
  weight-shared verifier was prototyped and is perf-neutral/negative. The open
  item is the serving-shape work, scoped in [MTP_PARITY_GAP.md](MTP_PARITY_GAP.md) §5.
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
