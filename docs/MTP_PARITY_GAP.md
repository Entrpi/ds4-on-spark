# ds4 CUDA — MTP status: parity gap closed; the real gap is single-stream verify cost

**Status:** The original parity gap — the CUDA MoE launcher rejecting the MTP
block's Q4_K routed experts, so the draft kernel never produced a token — is
**closed**. On the `decode-perf-tuning` branch at `5625a99`, MTP **works** on
CUDA and is **bit-exact to non-MTP greedy decode**, in both eager and
CUDA-graph-captured mode, on both sm_120 (RTX PRO 6000) and sm_121 (DGX Spark
GB10). What remains is **performance, not correctness**: ds4's single-stream
exact verifier costs ~2× a decode, so MTP is a **net throughput loss on a
single request** — and the lever that fixes it is **batched serving**, not
another single-stream kernel.

**Supersedes** the original 2026-05-12 version of this doc, which was a handoff
to *write* a Q4_K CUDA MoE kernel and projected a "+35–80 %" lift once it
landed. The kernel work has effectively landed; the single-stream lift
projection was wrong. The original root-cause chain is preserved as
[§7 Appendix](#7-appendix--the-original-q4_k-parity-gap-historical) for history.

**Last verified:** 2026-06-03 on GB10 (sm_121, as a weight-server client),
cross-checked against PRO 6000 (sm_120) measurements from the same branch.

---

## 0. TL;DR

- **Closed.** `routed_moe_launch` no longer hard-rejects the MTP block's Q4_K
  routed experts — a Q4_K MoE path was added, and the reject now fires only for
  genuinely unsupported quant combinations. `DS4_MTP_PROBE=1 … | grep -c "mtp
  probe draft failed"` prints **0** (was 58). Every draft now produces a token.
- **Bit-exact.** At `5625a99` the verifier runs the same mmq kernels as base
  decode and defaults to a sequential 2-row verify, so MTP output is
  byte-identical to non-MTP greedy — eager *and* captured, on both arches. The
  no-MTP determinism goldens are unchanged (GB10 long-context `b165ddd4`,
  PRO 6000 opp-c golden).
- **Still a net loss on single-stream.** The exact 2-row verifier sweeps the
  weight-bound projections (Q/KV, attn-out, routed MoE) once *per draft row*
  (≈2 weight sweeps), so verify ≈ 2× a decode (GB10 ~105 ms vs ~52; PRO ~34 vs
  ~16 — the ratio is architecture-independent). The MTP head's acceptance
  saturates at ~1.4 suffix tokens (prose) / ~2.0 (numeric). Measured A/B:
  prose **−21 %**, numeric **−15 %** vs no-MTP (captured, accept-gate off). Even
  at 100 % acceptance it loses (57 vs 51 ms/tok).
- **Not the model's fault.** vLLM runs *perf-positive* MTP on this exact model
  (DeepSeek-V4-Flash): acceptance length 1.8, K=3, net-positive in production,
  gains growing with batch size. The miss is specific to ds4's **single-stream,
  per-row, bit-identical** verify, which gets neither amortization axis vLLM has.
- **The fix is batched serving.** A weight-shared single-stream verifier was
  prototyped and is perf-neutral-to-negative (§4). The real win needs
  cross-request amortization + a lossless K-position verify over the prefill
  path; that's tracked in the batched-serving work.

---

## 1. What "works" now means

### 1.1 Drafts are produced — the Q4_K reject is gone

The old blocker was real. `routed_moe_launch` (the CUDA routed-MoE dispatcher
in `ds4_cuda.cu`, formerly pinned at `:8849`) hard-coded one quant combination:

```c
// the old gate — rejected everything but the main-model expert quants
if (gate_type != 16u /*IQ2_XXS*/ || down_type != 10u /*Q2_K*/) return 0;
```

The MTP draft GGUF (`DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`) stores its routed
experts in **Q4_K (type 12)**. So every MTP draft failed inside the MoE
launcher, the wrapping `metal_graph_eval_mtp_draft_from_hc` returned false, and
the speculative state machine treated the cycle as "no draft available" and
committed one token. From the outside MTP looked like a no-op.

That gate is now guarded by a Q4_K path:

```c
const int q4k_path = (gate_type == 12u && down_type == 12u);
if (!q4k_path && (gate_type != 16u || down_type != 10u)) return 0;
```

so the MTP block's Q4_K experts run, and drafts are produced. The original
one-line repro — which counted one draft-kernel failure per generation step —
now counts zero:

```bash
DS4_MTP_PROBE=1 ./ds4 --cuda -m <base.gguf> --mtp <mtp.gguf> \
  --mtp-draft 2 --temp 0 --nothink \
  -p "List 20 prime numbers" 2>&1 | grep -c "mtp probe draft failed"
# 0   (was 58)
```

(Verified 2026-06-03 on GB10 as a weight-server client.)

### 1.2 The verifier is bit-exact to non-MTP

The fix landed at `5625a99` (*"fix(cuda): MTP verifier uses base mmq kernel by
default (bit-exact to no-MTP)"*), two parts in `ds4_cuda.cu` + `ds4.c`:

1. **The verifier keeps mmq active by default** — the same fused-dequant matmul
   the base decode path uses. (A stale "Option D / Bug-2" gate had been forcing
   the verifier onto a different warp8 kernel, which drifted from base. Inverting
   it makes the verify path numerically identical to base. Opt out via
   `DS4_CUDA_MTP_VERIFIER_FORCE_WARP8`.)
2. **The exact policy defaults to the sequential 2-row verifier** — bit-exact by
   construction (each row is a base-exact n=1 decode). The batched `decode2`
   verifier (which pairs warp8-family kernels) still drifts and is slower, so it
   is opt-in (`DS4_MTP_EXACT_DECODE2` / `_ADAPTIVE`).

Validated: MTP (draft=2) output is byte-identical to non-MTP greedy on prose and
numeric prompts, in **eager** (`DS4_CUDA_LAYER_GRAPHS=0`) and **captured**
(`=1`), on both arches. The no-MTP determinism goldens are unchanged: GB10
capture-smoke + long-context (6 FP8 profiles) bit-identical at `b165ddd4`;
PRO 6000 capture-smoke + opp-c golden unchanged. The eager==captured hard gate
holds with MTP on.

---

## 2. The real gap — single-stream verify cost

MTP verifies a K-token draft against the target before committing it. ds4's
exact verifier verifies the two draft rows **sequentially**, re-sweeping the
weight-bound projections (Q/KV, attn-out, routed MoE) once per row instead of
sharing one sweep across both. For a decode that is almost entirely
weight-bandwidth-bound, that means **verifying 2 tokens costs ~2× a single
decode**:

| Arch | base decode | 2-row exact verify | ratio |
|---|---:|---:|---:|
| GB10 (sm_121) | ~52 ms | ~105–120 ms | ~2.0× |
| PRO 6000 (sm_120) | ~16 ms | ~34 ms | ~2.1× |

The ratio is **architecture-independent** — it is the structure of the verify,
not an occupancy artifact. (An early intuition that PRO 6000's idle SMs would
let a batched verify run in < 2 decodes was tested and refuted; see §4.)

Meanwhile the MTP head's acceptance **saturates** well below 2 tokens:

| Workload | mean accepted suffix |
|---|---:|
| prose | ~1.4 |
| numeric lists | ~2.0 |
| code / JSON | ~1.6–1.7 |

So the cycle does ~2× the work for well under 2× the tokens. Measured A/B
(weight-server client, draft=2, temp 0, **CUDA-graph-captured** = the shipping
default, accept-gate forced off so the loss is visible):

| Workload | no-MTP t/s | MTP (seq verify) t/s | Δ |
|---|---:|---:|---:|
| prose (n=128) | 19.64 | 15.58 | **−21 %** |
| numeric (n=200) | 19.52 | 16.51 | **−15 %** |

Even at **100 % acceptance** MTP loses: 172 ms / 3 tokens = 57 ms/tok vs
51 ms/tok for plain decode. Increasing draft depth past 2 is strictly worse —
the head adds zero accepted tokens beyond saturation but lengthens the cycle, so
**draft=2 is optimal** and still negative.

In production the **accept-gate** auto-disables MTP when draft confidence is low
(most prose: accept < ~0.90), so with default flags MTP reads as a *wash* rather
than a loss — it just stops drafting. Either way there is no single-stream win,
so the featured benchmarks are no-MTP.

---

## 3. Why this is not the model — vLLM proves perf-positive MTP on DSv4-Flash

Perf-positive MTP on **DeepSeek-V4-Flash specifically** is real. vLLM PR #43447
(<https://github.com/vllm-project/vllm/pull/43447>) carries comment benchmarks
on this exact model with MTP live in production:

- mean acceptance length **1.80–1.86**, per-position accept **0.80–0.86**, draft
  acceptance 82–85 %, draft depth **K=3**;
- accepted throughput rising with batch size: **+30.9 % at BS=1 → +47.7 % at
  BS=2 → +88.4 % at BS=4** (the growth with batch size is the amortization tell).

The difference is structural, in two amortization axes ds4's single-stream
verify lacks:

1. **A batched K-position verify.** Real spec-decode runs **one weight sweep
   across the whole K-token draft block** (FlashAttention-style), and is
   *"lossless"* — it produces a valid greedy decode of the target — but is **not
   bit-identical** to the single-token path. ds4 imposed a *bit-identical-to-no-
   MTP* constraint, which forces the per-row scalar verify (≈2 sweeps). That
   constraint is the self-inflicted handicap.
2. **Cross-request amortization.** With several concurrent requests, one verify
   sweep serves them all; the win grows with batch size. **BS=1 single-stream is
   MTP's hardest case** — ds4 (CLI / single-agent) gets none of this.

So the conclusion isn't "MTP can't help this model" — it's "MTP can't help
*this serving shape*." The eager==captured determinism gate is orthogonal and
can still hold under a lossless batched verify.

---

## 4. What we tried — the weight-shared exact verifier (single-stream)

The obvious single-stream fix is a **weight-shared exact verifier**: keep the
two draft rows but batch the per-row-independent weight-bound projections into
one sweep, serializing only the cheap KV-bound attention scan + state mutation.
We built it (clean-room `metal_graph_verify_shared_exact`, opt-in via
`DS4_MTP_EXACT_SHARED`). Result: **bit-exact, but perf-neutral-to-negative on
single-stream.** Why each batched class doesn't pay off:

- **Routed MoE — zero gain.** The biggest verify cost (~29 %). The two draft
  rows route to **disjoint experts**, so a "shared" sweep reads the *union* of
  both rows' selected experts — no bandwidth saved. (The batched MoE kernel *is*
  bit-exact — `max_abs == 0` on all layers — it just buys nothing.)
- **attn-out — neutral.** `attn_output` is a small low-rank weight; sharing its
  read saves negligible bandwidth. (Bit-exact, as it's an arithmetic clone of
  the n=1 kernel.)
- **Sequential early-stop wins anyway.** The seq verifier stops as soon as a row
  is rejected (~1.6 forwards at saturation), whereas an always-both-rows
  interleave always pays both. Plus the interleave's per-(row, layer) decode-
  scalar republish (needed so each row's FFN router reads its own token) adds a
  drain-sync tax.

GB10 A/B (captured, NO_ACCEPT_GATE, draft=2): prose no-MTP **19.64** > seq
**15.58** > shared **13.02**; numeric no-MTP **19.52** > seq **16.51** > shared
**15.81**. Even at the no-publish ceiling the shared path only ~ties seq and
still loses to no-MTP.

**Conclusion:** keep the sequential verifier as the exact verifier; the
weight-shared verifier is a *correct bit-exact reference*, not a single-stream
perf win. (It is uncommitted on the branch.)

---

## 5. The path forward — batched serving

The two amortization axes vLLM uses (§3) are exactly what a batched-serving
runtime provides. The plan:

- **A lossless K-position verify over the prefill path.** ds4's prefill is
  already a multi-token, one-weight-sweep forward (and is faster per token than
  decode — ~30 vs ~19 t/s on GB10). Generalizing it to verify a K-token draft
  block at BS=1 with *lossless* (not bit-identical) semantics gets axis (1)
  without the per-row 2× cost — accepting occasional ULP-tie flips, which is
  what real spec-decode does.
- **Cross-request batching.** A scheduler that lets concurrent requests share
  the verify sweep gets axis (2); the win grows with batch size.

Both are part of the larger batched-processing effort (the prefill path is the
batched substrate). The hard constraint carried forward: the **eager==captured
determinism gate must still hold** on both sm_120 and sm_121 — a lossless
batched verify can break bit-identity-to-no-MTP, but must not break
eager-vs-captured agreement.

---

## 6. Reproduce / validate

1. **Drafts produced (probe):**
   ```bash
   DS4_MTP_PROBE=1 ./ds4 --cuda -m <base> --mtp <mtp> --mtp-draft 2 --temp 0 \
     --nothink -p "List 20 prime numbers" 2>&1 | grep -c "mtp probe draft failed"
   # 0
   ```
2. **Bit-exact gate (hard):** MTP (draft=2, accept-gate off) md5 == no-MTP md5,
   on prose AND numeric, in **eager** (`DS4_CUDA_LAYER_GRAPHS=0`) AND **captured**
   (`=1`), on both arches.
3. **Determinism goldens (regression):** GB10 cuda-capture-smoke +
   long-context-full (expect `b165ddd4`); PRO 6000 capture-smoke + opp-c-full
   `--check-expected` (golden unchanged).
4. **Perf A/B:** MTP (seq verify) vs no-MTP, `DS4_MTP_TIMING=1`, n=128, prose +
   numeric — confirms the −15 %/−21 % single-stream loss and verify ≈ 2× decode.

---

## 7. Appendix — the original Q4_K parity gap (historical)

*Preserved from the 2026-05-12 version of this doc. This describes the blocker
that has since been closed (§1.1); it is no longer a to-do.*

The CUDA backend shipped full target-decode kernels but its `routed_moe_launch`
hard-coded `gate_type == IQ2_XXS && down_type == Q2_K` (the main-model expert
quants) and returned failure for any other combination. The DSv4-Flash MTP draft
GGUF uses **Q4_K (type 12)** for its routed experts, so every MTP draft attempt
failed inside the MoE launcher; the C-side speculative state machine treated
that as "no draft available" and silently degraded to single-token decode.

The failure was localized empirically: `DS4_MTP_PROBE=1` showed 100 %
draft-kernel failure (58/58 steps); per-step injection narrowed it to
`metal_graph_encode_decode_layer` applied to the MTP block; the stage profiler
(`DS4_METAL_DECODE_STAGE_PROFILE=1`) showed all stages running up to `router`
then no `routed_moe` stage — the MoE launch returned false. The Metal backend
already had the parity dispatch (`g_moe_mul_mv_id_q4_k_pipeline`,
`metal/moe.metal:413` / `:831`), so MTP worked there.

The fix was scoped as a single-file (~700–900 LOC) Q4_K CUDA MoE kernel mirroring
the Metal reference: a `cuda_block_q4_K` device struct + 6-bit scale unpack, a
Q4_K × Q8_K dot primitive, gate+up paired matvec, an accumulating down matvec,
and a dispatcher branch in `routed_moe_launch`. That work has since landed (the
`q4k_path` branch in §1.1). The original projection — that closing this gap
would yield a vLLM-comparable "+35.7 % from MTP-2, up to +80 % stacked" — proved
wrong for single-stream: it assumed a batched (one-sweep) verifier, which ds4's
bit-identical single-stream path is not (§2). The amortized win is real but
requires batched serving (§5).
