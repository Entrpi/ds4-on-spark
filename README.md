# ds4-on-spark

[`antirez/ds4`](https://github.com/antirez/ds4) (DwarfStar 4) running on a single
NVIDIA DGX Spark (GB10 / SM121, 128 GiB unified memory), with measured
benchmarks and a roofline analysis grounded in the hardware ceiling.

**Status:** Working end-to-end. Single-prompt smoke test passes; ds4's
prefill + steady-state decode are within ~10–15 % of the bandwidth roofline
for this quant on this hardware. MTP speculative decode is shipped by the
donor but currently a 3–6 % regression on Spark across all prompt types tested.

- **Reference:** [`antirez/ds4`](https://github.com/antirez/ds4) — MIT-licensed C+CUDA inference engine. CUDA backend landed 2026-05-11; this writeup uses HEAD `920f987` as of 2026-05-12.
- **Model:** [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf) — 81 GiB asymmetric quant: IQ2_XXS for routed-expert gate/up, Q2_K for routed-expert down (these dominate model bytes), Q8_0 for everything else dense (shared expert, attention projections, output head, router), F16 for LoRA matrices and the compressor/indexer, F32 norms. (FP8 in ds4 is a *runtime* KV-cache quantization — E4M3FN round-trip — not a stored weight format.) Plus an optional 3.6 GiB MTP draft GGUF.
- **Hardware:** NVIDIA DGX Spark, GB10, SM121, 128 GiB LPDDR5X unified. The donor's `Makefile` defaults `CUDA_ARCH` to `native` and accepts any `sm_NNN` override, so `make CUDA_ARCH=sm_121` is GB10-correct with no patches needed.

## Quick start

On a DGX Spark with CUDA 13 installed:

```bash
curl -sSL https://raw.githubusercontent.com/entrpi/ds4-on-spark/main/install.sh | bash -s -- --with-mtp --start
```

That one command:

1. Verifies the host (aarch64, GB10/SM121, CUDA 13, ≥110 GiB free disk).
2. Clones `antirez/ds4` into `~/code/ds4` (or `$DS4_SRC_DIR`).
3. Builds `ds4`, `ds4-server`, `ds4-bench` with `CUDA_ARCH=sm_121` in ~8 s.
4. Downloads the Q2 GGUF (~81 GiB) and the MTP GGUF (~3.6 GiB) from
   `antirez/deepseek-v4-gguf` into `~/gguf` (or `$DS4_GGUF_DIR`).
5. Runs the "capital of France" smoke test and asserts "Paris" in the output.
6. Starts `ds4-server` on `:8000` with `-c 32768`.

To preview without running:

```bash
curl -sSL https://raw.githubusercontent.com/entrpi/ds4-on-spark/main/install.sh | bash -s -- --help
```

Common overrides: `--cuda-arch sm_120` (datacenter Blackwell), `--no-download`
(reuse existing GGUF), `--src-dir`, `--gguf-dir`, `--ctx`, `--port`, `--force`
(skip host check).

## Hardware requirements

| | |
|---|---|
| Validated on | NVIDIA DGX Spark (GB10, SM121, 128 GiB unified) |
| Likely to work | other Blackwell with `--cuda-arch sm_120`, untested |
| CUDA toolkit | 13.x (we tested 13.0.88) |
| Disk | ≥110 GiB free for the GGUFs |
| OS | aarch64 Linux (Grace) |
| RAM (system, unified) | 128 GiB is enough for the model + ~250 MB KV @ 16k context |

GB10 is detected via `nvidia-smi --query-gpu=compute_cap` returning `12.1`.
Anything else gets a warning + `--force` override path.

## What you get

| Binary | Purpose |
|---|---|
| `ds4` | Interactive / one-shot CLI |
| `ds4-server` | OpenAI v1-compatible HTTP server (`POST /v1/chat/completions`, SSE streaming) |
| `ds4-bench` | Direct prefill + decode throughput sweep (no HTTP) |

`ds4-server` is the recommended runtime. It exposes:

- `POST /v1/chat/completions` (OpenAI-compatible streaming, tool calls)
- `POST /v1/completions`
- `GET /v1/models`

It also speaks Anthropic-shape on `/v1/messages` (see donor README).

## Benchmarks

All numbers from a single DGX Spark, `compute_cap=12.1`, CUDA 13.0.88,
`ds4` HEAD at `920f987` (2026-05-12).

### Build + cold start

| Step | Time |
|---|---|
| `make -j20 CUDA_ARCH=sm_121` | **7.9 s** |
| Cold load: 80.76 GiB of tensors → GPU cache | **~20 s** |
| Time-to-first-token (cold process, 12-token prompt) | **~21 s** |

After cold start, all subsequent benchmarks here are on a warm process.

### Throughput sweep (`ds4-bench`, direct CLI, no HTTP)

ctx range 2k–16k with `--gen-tokens 64`:

| ctx | prefill t/s | decode t/s | KV size |
|---:|---:|---:|---:|
| 2,048 | 287.8 | 13.50 | 52 MB |
| 6,144 | 332.6 | 13.47 | 109 MB |
| 10,240 | 300.9 | 13.14 | 165 MB |
| 14,336 | 303.3 | 13.00 | 221 MB |
| 16,384 | 290.9 | 12.92 | 250 MB |

- Prefill steady at **~290–330 t/s** across 2k → 16k.
- Decode steady at **~13 t/s**, mild ~5 % falloff out to 16k.
- KV stays compact (250 MB at 16k) — compressed KV doing its job.

### `llama-benchy`-style numbers (HTTP, steady-state)

Same model, same hardware, via [`eugr/llama-benchy`](https://github.com/eugr/llama-benchy)
through `ds4-server`'s OpenAI endpoint. Methodology mirrors llama-bench:
`tg` measured as `(N − 1) / (t_last − t_first)` — **excludes first-token latency**.

| test | t/s | peak t/s | ttfr (ms) |
|---|---:|---:|---:|
| pp2048 (prefill) | **364.5 ± 2.6** | — | 5890 |
| tg32 @ d=0 | 29.2 ± 1.4 | 31.0 | — |
| tg128 @ d=0 | 28.0 ± 1.0 | 34.0 | — |
| tg512 @ d=0 | 22.8 ± 2.6 | 33.3 | — |
| pp2048 @ d=4k | 339.5 ± 0.3 | — | 18712 |
| tg32 @ d=4k | 27.8 ± 1.3 | 29.3 | — |
| tg128 @ d=4k | 25.9 ± 0.5 | 30.0 | — |
| tg512 @ d=4k | 23.3 ± 2.2 | 32.3 | — |
| pp2048 @ d=16k | 310.7 ± 0.5 | — | 61401 |
| tg32 @ d=16k | 24.1 ± 0.4 | 27.0 | — |
| tg128 @ d=16k | 24.5 ± 0.8 | 30.7 | — |
| tg512 @ d=16k | 24.2 ± 0.6 | 30.0 | — |

Reproduce:

```bash
scripts/run-bench.sh --pp 2048 --tg 32 128 512 --depth 0 4096 16384
```

### Two metrics, same workload

The decode rate differs by ~2× between ds4's own log (`avg=12.94 t/s`) and
llama-benchy's `tg` (`24.14 t/s`) on the same request. Both are correct; they
answer different questions:

- **ds4's `avg t/s`**: `total_gen_tokens / total_decode_wall_time` — **includes** the first-token post-prefill setup (~1.0–1.3 s on this model).
- **llama-benchy `tg`**: `(N − 1) / (t_last − t_first)` — **excludes** first-token latency.

Worked example for the 18k-context tg32 request:

| | seconds | tokens | rate |
|---|---:|---:|---:|
| First token alone | ~1.20 | 1 | 0.83 t/s |
| Steady-state tail | ~1.27 | 31 | 24.4 t/s |
| Total | 2.47 | 32 | **12.94 t/s** (ds4) |
| Steady-state only | 1.27 | 31 | **24.4 t/s** (llama-benchy) |

For **interactive / agent use**, the first-token-inclusive rate (~13 t/s)
matches user perception. For **long-form generation** the steady-state
rate (~25–29 t/s) dominates wall time.

## Roofline analysis

How far below the hardware ceiling is ds4 running?

### Memory bandwidth ceiling (measured on Spark)

| Probe | Bandwidth | Note |
|---|---:|---|
| `nvbandwidth` H2D / D2H CE | 59 GB/s | Copy-engine path, not relevant for kernels |
| `nvbandwidth` device_local_copy | 111 GB/s | CE on single device |
| `bench/bw_bench.cu` copy (R+W) | **215 GB/s** | Kernel-driven, what matters |
| `bench/bw_bench.cu` read-only | **227 GB/s** | Pure read throughput |
| Published GB10 LPDDR5X peak | ~273 GB/s | 256-bit × 9400 MT/s theoretical |

The kernel-effective ~225 GB/s is the relevant ceiling — ~82 % of theoretical
peak, normal for real workloads on LPDDR.

### Bytes per token at decode (from safetensors index)

Aggregated across all 17 shards (88.4 GB total):

| Bucket | Total bytes | Active per token |
|---|---:|---|
| Routed experts (IQ2_XXS + Q2_K) | 78.28 GB | 6/256 active → **1.83 GB** |
| MLA attention + indexer + compressor | 7.05 GB | **all active** |
| Embed / head / final norm | 2.12 GB | **~1.0 GB** (head projection) |
| Shared expert (1 per MoE layer) | 0.74 GB | **0.74 GB** |
| MTP + HC + other | 0.30 GB | **~0.22 GB** |
| KV cache reads (at 16k) | — | **~0.25 GB** |

**Effective bytes per token at steady state: ~8 GB**

### Roofline

| Quantity | Value | % roofline |
|---|---:|---:|
| Kernel-effective BW | 225 GB/s | — |
| Steady-state decode | **~28 t/s** | — |
| Implied effective bytes per token | 225 / 28 = **8.0 GB** | — |
| Strict roofline (BW / bytes-per-token) | 225 / 8 = 28.1 t/s | — |
| **Steady-state efficiency** | | **~95 % of bandwidth roofline** |
| First-token-inclusive rate | ~13 t/s | — |
| Drag from first-token latency | 1.0–1.3 s overhead per request | — |

**Steady-state decode is essentially saturated.** Decoding faster on the same
hardware + same quant requires either a tighter quant (FP4 / 1.5-bit experts),
batched serving (amortize weight reads across users), or genuinely faster
hardware. There is no easy 2–4× win available.

The 13 t/s "perceived" number is **first-token latency**, not steady-state.
That is a separately-tractable optimization target (warm-cache reuse from
prior turns, persistent KV across thinking/answer phases, prefetch).

## Under the hood: how the CUDA backend works

A side-by-side analysis of ds4's two GPU backends —
[**docs/METAL_VS_CUDA.md**](docs/METAL_VS_CUDA.md) — covers the kernel
surface, the command lifecycle, and the model-attach strategy on each
platform. TL;DR for someone running on Spark and asking *what is the
implementation actually doing?*:

- **`ds4_cuda.cu` is 9,666 LOC, 106 `__global__` kernels, links `-lcudart -lcublas`.**
  All compiled ahead of time by `nvcc` for the target `CUDA_ARCH` — the binary
  is not portable across SM generations.
- **Three-tier model attach.** `cudaHostRegister(... cudaHostRegisterMapped | ReadOnly)`
  on the mmap'd 80 GiB GGUF is tried first to get a zero-copy device pointer.
  If pinning fails (or `DS4_CUDA_COPY_MODEL` is set), the engine falls back to
  per-range pinning, then to chunked `cudaMalloc + cudaMemcpy` in 64 MiB chunks.
  This is what the ~20 s cold load is.
- **Q8 → F16 weight cache for prefill.** On startup, dense Q8_0 weights are
  dequantised once on-device into an F16 buffer; `cublasGemmEx` then uses
  tensor cores for multi-token prefill matmuls. That's why prefill is
  ~300–360 t/s while decode is ~28 t/s — they take different routes through
  the matmul stack. Decode (`n_tok=1`) skips cuBLAS and uses hand-written
  Q8_0 matvecs where the cuBLAS launch overhead wouldn't amortize.
- **Routed experts stay quantised.** IQ2_XXS / Q2_K kernels dequantise inline
  on every expert dot; the codebook lives in `__constant__` memory via
  `ds4_iq2_tables_cuda.inc`. Pre-converting all 256 experts to F16 would erase
  the q2 memory win.
- **Default stream, serial execution.** `begin_commands` is a no-op;
  `flush_commands`, `end_commands`, and `synchronize` all reduce to
  `cudaDeviceSynchronize()`. Two named streams (`g_model_prefetch_stream`,
  `g_model_upload_stream`) exist only for async model staging at startup.
  Combined with the engine's single-session worker thread, this is why
  `ds4-server` serialises concurrent clients (see next section).
- **No GDS / cuFile.** Direct file reads (via `ds4_gpu_set_model_fd`) use Linux
  `O_DIRECT` on a registered FD — kernel DMA, not GPU-side DMA.

If you're considering writing a port, fork, or alternative serving layer,
the analysis doc lays out the kernel surface, the `DS4_CUDA_*` env-var knobs,
and the places where the Metal and CUDA backends diverge structurally
(model mapping, command-buffer batching, library use).

## Concurrency on `ds4-server` — single-stream, serialized

The OpenAI v1 server **does not actually parallelize concurrent requests**.
When llama-benchy hits it with `--concurrency 2`, ds4-server processes the
requests strictly sequentially: 168 s per request, second one waits for the
first to finish before starting prefill.

| llama-benchy `concurrency` | observed wall-clock behaviour |
|---|---|
| 1 | one request at a time (expected) |
| 2 | also one at a time — server queues, doesn't batch |

That means `t/s (total)` at c>1 in llama-benchy's output is misleading for
this engine: it's `c × per-request t/s`, but the wall time is also `c ×`
single-request wall time. If you need many concurrent users on one Spark
you need a different runtime (vLLM/SGLang with paged-attention batching).
ds4 is single-session by design.

## MTP (speculative decode) — neutral at best, depends on what you measure

The donor ships `--mtp <draft.gguf> --mtp-draft N`. The MTP support GGUF
is a separate 3.6 GiB file (`DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`).
The donor README labels MTP as "alpha quality / experimental."

Measured at `draft=2` (donor-recommended value) against matching no-MTP
baselines, four high-predictability prompts:

| Prompt | no-MTP t/s | MTP-2 t/s | Δ |
|---|---:|---:|---:|
| Count 1 → 60 (fully deterministic) | 15.13 | 14.27 | **−5.7 %** |
| English / NATO / Greek alphabets | 15.02 | 14.32 | −4.7 % |
| Declaration of Independence + 10 Presidents | 14.64 | 14.37 | −1.8 % |
| 27 EU capitals alphabetical | 14.92 | 14.48 | −2.9 % |
| **mean** | **14.93** | **14.36** | **−3.8 %** |

Three separate `--mtp-draft` values (1, 2, 4) all land within noise of each
other and slightly below the no-MTP baseline:

| Config | decode t/s (QuickSort prompt) |
|---|---:|
| no MTP | 13.5 (bench) / 14.88 (`is_prime`) |
| `--mtp-draft 1` | 13.81 |
| `--mtp-draft 2` | 13.62 |
| `--mtp-draft 4` | 13.63 |

**Counting 1→60 is fully deterministic** — every next token is forced — so
acceptance rate should be ~100 %. MTP is still slower. This rules out
"low acceptance rate" as the explanation; the cost is in the
draft-inference + verifier mechanics themselves. Consistent with the donor's
"alpha" labeling.

It also fits the roofline picture: steady-state decode is already at ~95 % of
the bandwidth ceiling, so even a hypothetical zero-overhead MTP couldn't
deliver large gains on this hardware.

### MTP via llama-benchy (steady-state methodology)

llama-benchy excludes first-token latency, isolating the steady-state
decode rate. Same hardware, three runs each, d=8192 tg=512:

| Config | tg512 @ d=8192 t/s | peak t/s | prefill t/s |
|---|---:|---:|---:|
| no MTP | 22.14 ± 2.57 | **28.33** ± 1.25 | 328.09 ± 0.51 |
| MTP draft=2 | 23.61 ± 0.48 | **28.33** ± 0.47 | 328.49 ± 0.68 |
| Δ | +6.6 % (within noise) | identical | identical |

Three observations:

- **Peak t/s is bit-identical** between the two runs (28.33 t/s in both). The
  steady-state ceiling is set by the bandwidth roofline (§ Roofline analysis),
  not by MTP availability.
- **MTP narrows variance** (±0.48 vs ±2.57). The draft+verify rhythm
  smooths per-token jitter. The mean lands ~7 % above no-MTP, but that
  is well within the no-MTP error bar.
- **Reconciles the CLI-vs-benchy gap.** Our earlier CLI tests showed MTP
  as a 3–6 % *regression* (using first-token-inclusive metrics). That
  cost is the MTP path's per-request setup overhead — it lands on the
  first token. Once excluded, MTP is roughly neutral. Both measurements
  are correct; they answer different questions.

**Net:** MTP cannot meaningfully speed up decode on Spark because decode
is already at the bandwidth roofline. There's a small variance-smoothing
win, paid for with first-token setup cost. Choose `--with-mtp` only if
you care about decode-time stability and not about TTFT.

## Quality checks (qualitative)

ds4 produces clean output on first try across several probe types:

**Factual recall** — "What is the capital of France?" → "The capital of France is Paris." ✓

**Code + reasoning** — "`is_prime(n)` with 6k±1 optimization, list primes 100-130" →

```python
def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```

Prime numbers listed: **101, 103, 107, 109, 113, 127** — all six correct.

**Long-form structured** — "Explain QuickSort with worked example [38, 27, 43,
3, 9, 82, 10], full recursion, complexity analysis, optimizations" — clean
multi-section response with correct partitioning steps and complexity bounds.

## Repo layout

```
install.sh                  One-shot installer (curl | bash | --help)
scripts/
  smoke-test.sh               First-token sanity check
  start-server.sh             Start ds4-server (idempotent, with-MTP flag)
  run-bench.sh                Run llama-benchy via uvx
bench/
  bw_bench.cu                 Kernel-side memory-bandwidth probe
docs/
  STRATEGIC_CHECKPOINT.md     Detailed analysis of how this benchmark
                              affects the "should we keep porting?" decision
  METAL_VS_CUDA.md            Side-by-side comparison of ds4_metal.m and
                              ds4_cuda.cu — kernel surface, command lifecycle,
                              model attach, quantisation, and where each
                              backend's design diverges
```

## Reproducing the benchmarks

```bash
# 1. Build + download + smoke test
./install.sh --with-mtp

# 2. Start the server
./scripts/start-server.sh --port 8000 --ctx 32768

# 3. Run llama-benchy (installs uvx automatically if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh   # one-time
./scripts/run-bench.sh                            # default sweep
./scripts/run-bench.sh --depth 0 4096 16384 32768 --tg 32 128 512

# 4. Measure raw memory bandwidth ceiling
/usr/local/cuda/bin/nvcc -O3 -arch=sm_121 bench/bw_bench.cu -o /tmp/bw_bench
/tmp/bw_bench 8192

# 5. Compare MTP vs no-MTP yourself
./scripts/start-server.sh --port 8000 --with-mtp --draft 2
./scripts/run-bench.sh --depth 0 --tg 128 --runs 3
```

## How this fits with related work

| Piece | Role |
|---|---|
| [`antirez/ds4`](https://github.com/antirez/ds4) | The C+CUDA inference engine itself — narrow, DSv4-Flash-only by design |
| [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf) | The Q2/Q4 GGUFs ds4 is designed to consume |
| this repo | Spark-specific install + benchmark + analysis layer on top of the donor |
| [`Entrpi/ds4-spark-vllm`](https://github.com/Entrpi/ds4-spark-vllm) | Alternative path: same model via vLLM. Different perf profile, more flexible serving, larger surface area. |
| [`eugr/llama-benchy`](https://github.com/eugr/llama-benchy) | The benchmark methodology used here — generic, OpenAI v1, comparable to llama-bench / vllm bench |

## Acknowledgements

- [`antirez/ds4`](https://github.com/antirez/ds4) — the inference engine and the 2-bit recipe. MIT-licensed.
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) and GGML — the GGUF ecosystem, quant formats, and engineering knowledge ds4 stands on.
- [`deepseek-ai`](https://huggingface.co/deepseek-ai) — DeepSeek-V4-Flash upstream weights and architecture.
- [`eugr/llama-benchy`](https://github.com/eugr/llama-benchy) — benchmark methodology and tooling.

## License

MIT.
