# ds4-on-spark

This repo gets **[`Entrpi/ds4`](https://github.com/Entrpi/ds4)** — our
DGX-Spark-optimized, major-feature fork of
[`antirez/ds4`](https://github.com/antirez/ds4) (DwarfStar 4) — running on
your Spark with one command, serving **DeepSeek-V4-Flash** entirely on-device
(GB10 / SM121, 128 GB unified memory — ~119 GiB usable; RTX PRO 6000 /
5090-class `sm_120` also builds).

Compared to the upstream engine you get **double or more the prefill
throughput** (2× on GB10, ~4× on a PRO 6000), **1.15–1.7× the decode speed
across the full 2k–128k context range**, and a rich serving experience
upstream doesn't have: **continuous batching**, **prefix caching** (warm
start cuts time-to-first-token ~7× on shared prefixes, with fork-by-copy
fan-out for parallel branches), and **DSpark lossless speculative decode** —
together, a much faster multi-agent experience than single-stream serving.
The [fork delta table](#what-the-fork-adds-over-upstream) itemizes exactly
what changed and how each claim was measured; benchmarks, the roofline
model, and the engineering story follow below. Upstream remains the
architectural foundation and the model recipe — this repo pins and packages
the fork.

**Prefill and decode vs upstream, across the context frontier** (upstream
`main` measured on the same box, same GGUF; spec-decode off in this
comparison — it only widens the gap):

![Throughput across context frontiers: upstream antirez/ds4 main vs the fork](docs/v010_sweep_overlay.svg)

**Decode at the ship config** (DSpark + yield-quench + kv-gate) — solid line
is the adversarial-prose floor test, dashed is code continuation; the fork
wins 1.15–1.68× over upstream everywhere while never dropping below ~0.96×
its own plain decode:

![Ship decode across context frontiers: plain vs DSpark with quench and kv-gate, two corpora](docs/v011_decode_overlay.svg)

**Status:** Working end-to-end, pinned to the fork release
[**`v0.2.1`**](https://github.com/Entrpi/ds4/blob/v0.2.1/CHANGELOG.md) — the
robust-serving release: ship-path crash classes fixed, speculation on the
continuous path for tools/thinking, deep-context capacity to the 766K-class
on one box, and standing release gates (tool-calling, deep-context, churn
soak, 20/20 needle matrix, quality restamp) green on the tagged binary. On real
workloads, **DSpark lossless speculative decode reaches a suite mean of ~28 t/s
on GB10** — **1.38× plain continuous decode** (peak 1.71× on stepwise math at
89 % draft acceptance) — and since v0.1.1 it is **net-positive by default**: a
terminal yield-quench controller turns speculation off per request when
realized acceptance can't pay the verify cost, so the worst case is **~0.96×
plain on adversarial prose** where always-on speculation used to bottom out at
0.72×, and a kv-depth gate hands off to plain decode past 64k context.
Speculative gain is content-dependent (structured / code / math wins big,
open-ended prose sits at parity) — see the
[two-corpus frontier chart](docs/v011_decode_overlay.svg) and the
[break-even law](#the-break-even-law) below. Prefill runs ~2× the upstream
engine on GB10 (D2R tensor-core MoE GEMMs). The Metal backend is unaffected.

- **Reference:** [`antirez/ds4`](https://github.com/antirez/ds4) — MIT-licensed C+CUDA inference engine (CUDA backend landed 2026-05-11). **This repo pins the [`Entrpi/ds4`](https://github.com/Entrpi/ds4) fork at release [`v0.2.1`](https://github.com/Entrpi/ds4/blob/v0.2.1/CHANGELOG.md)** (2026-07-16): the batched-serving line — D2R tensor-core prefill, per-layer CUDA-graph decode capture, continuous batching, weight server, DSpark speculative decode with terminal yield quench (default on) + kv-depth gate, and v0.2's robust-serving layer (crash fixes, tools/thinking speculation on the continuous path, deep-context capacity, `--mtp` optional, FP8/FP4 compressed-KV opt-ins), plus v0.2.1 observability (per-request `timings`, Prometheus `/metrics`, human-readable `/v1/stats`). The fork `CHANGELOG.md` documents every fork-side change.
- **Model:** [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf) — 81 GiB asymmetric quant: IQ2_XXS for routed-expert gate/up, Q2_K for routed-expert down (these dominate model bytes), Q8_0 for everything else dense (shared expert, attention projections, output head, router), F16 for LoRA matrices and the compressor/indexer, F32 norms. (FP8 in ds4 is a *runtime* KV-cache quantization — E4M3FN round-trip — not a stored weight format.) Plus an optional 3.6 GiB MTP draft GGUF.
- **Hardware:** NVIDIA DGX Spark, GB10, SM121, 128 GB LPDDR5X unified (~119 GiB usable). The donor's `Makefile` has a `make cuda-spark` target that builds native `sm_121`, plus `make cuda CUDA_ARCH=sm_NNN` for an explicit override — both GB10-correct with no patches needed. (Building with an empty `-arch` measured ~25% slower prefill on GB10, so the explicit arch matters.)

## What the fork adds over upstream

[`Entrpi/ds4`](https://github.com/Entrpi/ds4) tracks upstream and specializes
it for **Blackwell CUDA serving performance**. Everything below is fork-side
work, measured engine-to-engine against upstream `main` on the same GB10, same
GGUF (upstream decode measured flat May → July 2026):

| Area | Fork | vs upstream |
|---|---|---|
| **Prefill** | D2R ("dequant-to-register") tensor-core MoE GEMMs — IQ2_XXS / Q2_K / Q8_0 expert weights dequantized directly into MMA fragments from weight-server SoA artifacts; token-tile HMMA attention; L2-reuse-aware expert-major CTA schedule | **~2× on GB10** (305 → 800 tok/s @12k over the fork's own arc; ~4× on RTX PRO 6000, `sm_120`) |
| **Decode** | Per-layer CUDA-graph capture of the batched decode step; split-K/vectorized F16 decode matmul; aligned-quant dispatch tiers | **1.15–1.68× across 2k–128k context** ([chart](docs/v011_decode_overlay.svg)) |
| **Speculation** | DSpark lossless block drafter (3-layer target-fused, Q2K) + terminal yield-quench (net-positive per request, default on) + kv-depth gate | upstream MTP is single-token, net-negative single-stream; fork suite mean **1.38×** its own plain decode |
| **Serving** | Continuous batching (mid-flight admit/evict, chunked prefill interleave), per-bank warm start (~7× TTFT on shared prefixes), fork-by-copy fanout, OpenAI + Anthropic-shape APIs | upstream serves one stream |
| **Ops** | Resident weight server (VMM-backed, IPC manifest) — engines import the 81 GiB model in seconds instead of multi-minute reloads; builds the aligned repack artifacts the fast kernels read in place | upstream reloads per process |
| **Telemetry** | Per-step speculative trace + offline policy replayer (`tools/dspark_trace_replay.py`), quench/gate/profile counters | — |

Every fork-side change is documented in the fork
[`CHANGELOG.md`](https://github.com/Entrpi/ds4/blob/v0.2.1/CHANGELOG.md);
the [roofline analysis](#roofline-why-speculation-and-batching-are-the-levers)
below explains why these are the changes that matter on this hardware.

## Quick start

On a DGX Spark with CUDA 13 installed:

```bash
curl -sSL https://raw.githubusercontent.com/entrpi/ds4-on-spark/main/install.sh | bash -s -- --start
```

That one command:

1. Verifies the host (aarch64, GB10/SM121, CUDA 13, ≥120 GiB free disk).
2. Clones the `Entrpi/ds4` fork at tag **`v0.2.1`** into `~/code/ds4` (or `$DS4_SRC_DIR`).
3. Builds `ds4`, `ds4-server`, `ds4-bench` with `CUDA_ARCH=sm_121` in ~8 s.
4. Downloads the Q2 GGUF (~81 GiB) + MTP GGUF (~3.6 GiB) from
   [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf)
   and the DSpark Q2K drafter (~6.5 GiB) from
   [`bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF`](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF)
   into `~/gguf` (or `$DS4_GGUF_DIR`).
5. Runs the "capital of France" smoke test and asserts "Paris" in the output.
6. Starts `ds4-server` on `:8000` with `-c 32768` serving the **full DSpark
   speculative stack** — lossless, suite mean **1.38× plain decode**, with the
   yield-quench controller and kv-depth gate riding the v0.2.1 defaults.

`--no-dspark` serves plain continuous decode instead (skips the drafter
download); `--with-mtp` alone gives MTP-2 speculation (a modest ~1.08×).

To preview without running:

```bash
curl -sSL https://raw.githubusercontent.com/entrpi/ds4-on-spark/main/install.sh | bash -s -- --help
```

Common overrides: `--cuda-arch sm_120` (RTX PRO 6000 / 5090-class Blackwell; datacenter B200/B300 is `sm_100`, untested), `--no-download`
(reuse existing GGUF), `--src-dir`, `--gguf-dir`, `--ctx`, `--port`, `--force`
(skip host check).

## Upgrading from an earlier install

Re-run the installer — every step is idempotent:

```bash
pkill -x ds4-server   # the installer starts servers but never stops old ones
curl -sSL https://raw.githubusercontent.com/entrpi/ds4-on-spark/main/install.sh | bash -s -- --start
```

What happens to an existing setup:

- **Your ds4 clone fast-forwards to the `v0.2.1` tag** (`git fetch` +
  `reset --hard`, remote repointed automatically if you installed back when
  this repo cloned `antirez/ds4`). Any local edits in that clone are
  **discarded** — it's an installer-managed tree.
- **GGUFs you already have are kept** (size-checked, not re-downloaded). Two
  new downloads happen on upgrade: the DSpark drafter (~6.5 GiB) and — if you
  installed before 2026-07-13 — the **imatrix base quant** (~81 GiB): the old
  installer default pointed at the plain `chat-v2.gguf` while the quality
  baselines were all measured on `chat-v2-imatrix.gguf`; the default now
  matches the validated build. To keep serving your existing non-imatrix
  file instead (and skip the 81 GiB):
  `GGUF_FILE=DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf ./install.sh --start`
- **The default serve mode changes** from plain decode to the DSpark
  speculative stack (same OpenAI-compatible API, lossless output, ~10 GiB
  more unified memory in use). `--no-dspark` restores the old behaviour.
- If a rebuild ever looks wrong after a big jump, `make -C ~/code/ds4 clean`
  and re-run.

## Hardware requirements

| | |
|---|---|
| Validated on | NVIDIA DGX Spark (GB10, SM121, 128 GB / ~119 GiB unified) |
| Likely to work | RTX PRO 6000 / 5090-class Blackwell with `--cuda-arch sm_120` (PRO 6000 prefill measured); `sm_100` datacenter untested |
| CUDA toolkit | 13.x (we tested 13.0.88) |
| Disk | ≥110 GiB free for the GGUFs |
| OS | aarch64 Linux (Grace) |
| RAM (system, unified) | 128 GB (~119 GiB usable) is enough for the model + ~250 MB KV @ 16k context |

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

## DSpark: the featured serving mode

Since fork `v0.1.1`, the featured runtime is **continuous batching + DSpark
lossless speculative decode** (a DFlash-style drafter verified by the target
model — the verify forward is the sole token source, so output is lossless by
construction; quality held at gsm8k 97.5 % / mbpp 92.5 % through the full
serving path at the release gate).

Measured on GB10 with llama-benchy's `mtp-bench` suite (9 workloads × 3 runs,
512 tokens, non-streaming wall tok/s):

| workload | plain t/s | DSpark t/s | gain | tok/step | accept |
|---|---|---|---|---|---|
| stepwise_math | 20.1 | 34.5 | **1.71×** | 4.00 | 89 % |
| qa_factual | 19.6 | 31.9 | 1.63× | 3.89 | 91 % |
| summarize | 19.6 | 31.0 | 1.59× | 3.88 | 87 % |
| code_cpp | 20.4 | 30.0 | 1.47× | 3.38 | 80 % |
| translation | 20.4 | 27.9 | 1.37× | 3.10 | 76 % |
| code_python | 20.5 | 26.4 | 1.29× | 2.94 | 72 % |
| explain_concept | 20.5 | 25.2 | 1.23× | 2.80 | 69 % |
| long_code_review | 18.8 | 22.1 | 1.17× | 2.62 | 66 % |
| creative_short | 20.6 | 19.9 | 0.96× | 2.18 | 58 % |
| **suite mean** | **20.1** | **27.7** | **1.38×** | — | — |

<sub>Stamped 2026-07-11 at the v0.1.0 release cut (Q4K drafter, `-c 4096`,
non-streaming `usage`-counted wall tok/s). The Q2K drafter — the v0.1.1
default — measured equal-or-better in a same-script A/B (suite mean 28.8 vs
28.3, acceptance within ±3 pp per workload).</sub>

![Decode frontier: plain vs DSpark ship across context depth, two corpora](docs/v011_decode_overlay.svg)

The chart is the honest version of the same story across context depth: the
solid line is **War & Peace continuation** — deliberately the hardest corpus
(~52 % acceptance, at break-even) — and the dashed line is **C-source
continuation** (mean 1.11×, raw peak 1.42× over plain at 47k). Content decides
the win; two safety nets bound the losses:

- a **terminal yield-quench controller** (default on) that turns speculation
  off for the rest of any request whose realized acceptance can't pay its
  verify cost — worst case ~0.96× plain, versus 0.72× for always-on
  speculation on the same corpus;
- a **kv-depth gate** (`DS4_DSPARK_MAX_KV=65536`) that hands deep-context
  requests to plain decode, which stays 1.15–1.2× the upstream engine out to
  128k.

### The break-even law

One DSpark verify step (1 committed row + 4 draft rows through the full
model) costs a depth-flat **~2.17 plain decode steps** on GB10. So

```
speedup = tokens-per-step ÷ 2.17        break-even ≈ 56 % draft acceptance
```

Everything above follows from this: structured content accepts 70–90 % of
drafts (1.3–1.7×), open-ended prose sits near 52 % (parity), and the quench
controller is just this law enforced per request — it accumulates the
cumulative regret `2.22 − tokens-per-step` each verify step and terminally
quenches the request when the deficit exceeds ~4 plain-step times. The
parameters were calibrated offline against per-step traces
(`DS4_DSPARK_TRACE=1` + `tools/dspark_trace_replay.py` in the fork) and the
in-engine controller is validated to reproduce the offline policy exactly.

### The drafter artifact

The installer downloads the prebuilt ~6.5 GiB Q2K drafter from
[`bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF`](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF)
by default. Q2K routed experts are the ship default (equal throughput and
acceptance to Q4K in A/B, 4.2 GiB smaller). To build a drafter yourself from
the official checkpoint's drafter shards (e.g. the Q4K-expert variant), use
the fork's extraction tool:

```bash
cd ~/code/ds4
python3 gguf-tools/dspark_extract.py --src <dir-with-drafter-shards> \
    --out ~/gguf/DSpark-drafter-Q4K-Q8.gguf --validate --experts q4_k
```

Kill switches if you ever need them: `DS4_DSPARK_QUENCH=0` (always-spec below
the kv gate), `--no-dspark` at install / `DS4_CONT_DSPARK` unset (no DSpark).

## Benchmarks

Hardware for every number in this README: a single DGX Spark — GB10,
`sm_121`, `compute_cap=12.1`, CUDA 13.0.88 — measured through the real
serving path ([`eugr/llama-benchy`](https://github.com/eugr/llama-benchy),
non-streaming wall tok/s) unless marked engine-side. The v0.1.1 headline
numbers, in one place:

| metric | value | evidence |
|---|---|---|
| Prefill @12k (engine-side) | ~800 tok/s — **~2× upstream** (~4× on RTX PRO 6000, `sm_120`) | [frontier chart](docs/v010_sweep_overlay.svg) |
| Plain continuous decode, short ctx | 18–20 tok/s | [frontier chart](docs/v010_sweep_overlay.svg) |
| DSpark decode, 9-workload suite | mean **27.7 tok/s (1.38×)**, best 34.5 (1.71×) | [suite table](#dspark-the-featured-serving-mode) |
| DSpark floor, adversarial prose | **≥0.96× plain** (yield-quench) | [two-corpus chart](docs/v011_decode_overlay.svg) |
| Deep context | decode 1.15–1.2× upstream out to 128k (past the 64k kv gate) | [two-corpus chart](docs/v011_decode_overlay.svg) |
| Concurrent serving | ~**30 tok/s aggregate**, saturating ≈4 requests | [concurrency chart](docs/v011_conc_throughput.svg) |
| Build | `make CUDA_ARCH=sm_121` ≈ 8 s | installer output |
| Cold start | ~20 s direct model attach; **seconds** as a weight-server client | — |

Reproduce on your own Spark:

```bash
# install + serve the DSpark ship default on :8000
curl -sSL https://raw.githubusercontent.com/entrpi/ds4-on-spark/main/install.sh | bash -s -- --start

# llama-benchy against the running server (needs uv: https://docs.astral.sh/uv/)
./scripts/run-bench.sh --pp 2048 --tg 128 512 --depth 0 4096 16384

# concurrency sweep (the concurrency chart's shape)
./scripts/run-bench.sh --pp 2048 --tg 256 --concurrency 1 2 4 8 16

# plain-decode comparison leg
pkill -x ds4-server; ./install.sh --start --no-dspark

# raw memory-bandwidth ceiling (feeds the roofline below)
/usr/local/cuda/bin/nvcc -O3 -arch=sm_121 bench/bw_bench.cu -o /tmp/bw_bench
/tmp/bw_bench 8192
```

## Roofline: why speculation and batching are the levers

Kernel-effective memory bandwidth measured on GB10 with
[`bench/bw_bench.cu`](bench/bw_bench.cu): **~215 GB/s** copy (read+write),
**~227 GB/s** read-only — ~82 % of the 273 GB/s theoretical LPDDR5X peak,
normal for real workloads. Decode activates **~11 GB per token** (bottom-up
from the safetensors index: 6/256 routed experts ≈ 1.8 GB, plus the dense
attention/indexer/compressor stack, output head, shared expert, and KV
reads — all of which are touched every token). That puts a hard bandwidth
roofline on single-stream plain decode of about **225 / 11 ≈ 20.5 tok/s**,
and the engine's plain decode measures 18–20 — i.e. **~90 %+ of the wall**.
(±20 % on the bytes-per-token estimate moves the efficiency figure, not the
conclusion.)

That is the design rationale for everything this fork features. Once plain
decode sits at the bandwidth wall, the only ways forward are:

- **commit more tokens per weight sweep** — DSpark verifies 1+4 draft rows in
  one sweep (suite mean 1.38×, structured content up to 1.71×); or
- **share the sweep across requests** — continuous batching (~30 tok/s
  aggregate at 4–16 concurrent requests); or
- **read fewer bytes** — a tighter quant (FP4 / 1.5-bit experts), not
  currently pursued.

## Under the hood

[**docs/METAL_VS_CUDA.md**](docs/METAL_VS_CUDA.md) is a side-by-side analysis
of the upstream engine's two GPU backends — kernel surface, command
lifecycle, model attach, quantization handling (May 2026 snapshot; still an
accurate map of upstream and of what the fork inherited). The fork's CUDA
backend has since diverged where it counts: D2R tensor-core MoE prefill
kernels, token-tile HMMA attention, per-layer CUDA-graph decode capture, a
multi-sequence batched forward, and the weight-server import path — each
documented in the fork
[`CHANGELOG.md`](https://github.com/Entrpi/ds4/blob/v0.2.1/CHANGELOG.md).

Two inherited facts worth knowing as an operator: the engine attaches the
mmap'd GGUF zero-copy via `cudaHostRegister` when the host allows it (the
~20 s cold load is the chunked-copy fallback; weight-server clients skip the
question entirely and import in seconds), and routed experts stay quantized
in memory with inline dequant — pre-converting all 256 experts to F16 would
erase the 2-bit memory win.

## Concurrency on `ds4-server`

Since the fork's batched-serving line (v0.1.0+), `ds4-server` runs
**continuous batching by default**: concurrent requests are admitted
mid-flight into per-sequence KV banks, prefill is chunked and interleaved
with live decode, and the multi-sequence forward scales ~6× from batch 1 to
128 in the engine. Per-bank warm start and fork-by-copy give large TTFT wins
on shared-prefix fan-out. `DS4_SERVER_CONTINUOUS=0` restores the old
serialized behaviour; for single-user latency-critical use that can still
be the right call. DSpark speculation engages on solo streams (the default
`DS4_DSPARK_MAX_NLIVE=1` regime) and hands concurrent traffic to plain
batched decode, which wins above N=1 today.

Measured at the ship default (v0.1.1, ctx 69632, pp=2048 tg=256 per request,
W&P prose — the adversarial floor corpus for the solo point):

![Served decode throughput vs concurrency at the ship default](docs/v011_conc_throughput.svg)

Aggregate decode rises 18 → 30 tok/s and saturates around 4 concurrent
requests; per-request throughput falls as requests share the machine
(18.1 / 14.2 / 7.6 / 4.8 / 3.2 tok/s at c = 1/2/4/8/16, each measured over
that request's own generation window). Note the c=1 point runs DSpark on the
floor corpus, i.e. ≈ parity — on structured content a solo stream reaches
27–35 tok/s (see the [decode chart](docs/v011_decode_overlay.svg)), which is
comparable to the whole box's batched aggregate. That is the practical
guidance: a single agent gets DSpark speeds; a fleet gets ~30 tok/s of
aggregate decode plus the prefix-cache TTFT wins.

## What about upstream MTP?

Upstream ships a single-token MTP head (`--mtp`, 3.6 GiB draft GGUF, labeled
alpha). Our June measurement campaign found single-stream MTP on this
hardware is a **net throughput loss** even at 100 % draft acceptance — the
bit-exact verifier pays ~2 weight sweeps for ~1.6 accepted tokens — and
concluded the path to a real win was batched serving, not a cleverer
single-stream verifier. That analysis is preserved in
[`docs/MTP_PARITY_GAP.md`](docs/MTP_PARITY_GAP.md), and its conclusion is
exactly what the fork then built: speculation rebuilt on the
continuous-batching path (MTP-2 mode, ~1.08× suite — still available as
`--with-mtp`), then the DSpark block drafter replacing the single-token head
(1.38× suite mean), and the v0.1.1 quench controller making it net-positive
per request. See [DSpark: the featured serving
mode](#dspark-the-featured-serving-mode).

## Quality

Served quality is gated on measurements, and speculation cannot change it:
DSpark's verify forward on the target model is the sole token source, so
output is the model's own by construction. At the v0.1.1 release gate,
through the full speculative serving path: **gsm8k 117/120 (97.5 %)** and
**mbpp 37/40 (92.5 %)**. The fork's standing eval baseline — GSM8K 97.6,
HumanEval 87.8–91.5, MBPP 90.0, IFEval 83.4 strict, MMLU 63.5,
needle-in-a-haystack **70/70 at 128k** — was last fully re-stamped at the
v0.1.0 cut (2026-07-10). Remarkable numbers for a 2-bit-expert quant; credit
the upstream recipe.

## Repo layout

```
install.sh                  One-shot installer (curl | bash | --help)
scripts/
  smoke-test.sh               First-token sanity check
  start-server.sh             Plain / MTP serving helper (the DSpark ship
                              default is served by install.sh --start)
  run-bench.sh                llama-benchy runner (uv)
  plot_decode_ctx.py          Chart generators
  plot_overlay.py
bench/
  bw_bench.cu                 Kernel-side memory-bandwidth probe
docs/
  v010_sweep_overlay.svg      Prefill + decode vs upstream across context
  v011_decode_overlay.svg     Ship decode, two corpora (floor + favorable)
  v011_conc_throughput.svg    Served throughput vs concurrency
  METAL_VS_CUDA.md            Upstream backend analysis (May 2026 snapshot)
  MTP_PARITY_GAP.md           Why single-stream MTP lost (resolved by DSpark)
  STRATEGIC_CHECKPOINT.md     Historical decision doc from the May bring-up
```

## How this fits with related work

| Piece | Role |
|---|---|
| [`antirez/ds4`](https://github.com/antirez/ds4) | The C+CUDA inference engine itself — narrow, DSv4-Flash-only by design |
| [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf) | The Q2/Q4 GGUFs ds4 is designed to consume |
| this repo | Spark-specific install + benchmark + analysis layer, pinning the fork at a validated release |
| [`Entrpi/ds4-spark-vllm`](https://github.com/Entrpi/ds4-spark-vllm) | Alternative path: same model via vLLM. Different perf profile, more flexible serving, larger surface area. |
| [`eugr/llama-benchy`](https://github.com/eugr/llama-benchy) | The benchmark methodology used here — generic, OpenAI v1, comparable to llama-bench / vllm bench |

## Acknowledgements

- [`antirez/ds4`](https://github.com/antirez/ds4) — the inference engine and the 2-bit recipe. MIT-licensed.
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) and GGML — the GGUF ecosystem, quant formats, and engineering knowledge ds4 stands on.
- [`deepseek-ai`](https://huggingface.co/deepseek-ai) — DeepSeek-V4-Flash upstream weights and architecture.
- [`eugr/llama-benchy`](https://github.com/eugr/llama-benchy) — benchmark methodology and tooling.

## License

MIT.
