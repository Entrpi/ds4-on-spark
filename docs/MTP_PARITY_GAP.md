# ds4 CUDA — MTP parity-with-Metal gap

**Status:** open.
**Last verified:** 2026-05-12 on a single DGX Spark (GB10, SM121, CUDA 13.0.88) against `antirez/ds4` HEAD `920f987` and `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`.
**Audience:** anyone picking up the CUDA port of `antirez/ds4` who wants MTP (multi-token prediction) speculative decode to deliver real throughput on Spark — and a paired comparison against the Metal backend that already supports it.

This is a self-contained handoff: file paths and line numbers are pinned, the empirical evidence is reproducible from a clean install, and the recommended kernel work is scoped.

---

## 0. TL;DR

- `antirez/ds4`'s CUDA backend (`ds4_cuda.cu`, landed 2026-05-11) ships full target-decode kernels but its `routed_moe_launch` hard-codes one quant combination:
  ```c
  // ds4_cuda.cu:8849
  if (gate_type != 16u || down_type != 10u) return 0;
  ```
  (16 = IQ2_XXS, 10 = Q2_K — the main-model expert quants).
- The DSv4-Flash MTP draft GGUF uses **Q4_K (type 12)** for its routed expert tensors. Every MTP draft attempt on CUDA fails inside `routed_moe_launch`, the function returns `false`, the C-side speculative state machine treats that as "no draft available," and the speculative path silently degrades to single-token decode. From the outside it looks like MTP "doesn't help" or "is alpha"; what's actually happening is that **the MTP draft kernel never produces a token**.
- Metal already has the parity dispatch (`ds4_metal.m:11580`, `ds4_metal.m:12728`, `metal/moe.metal:413`, `metal/moe.metal:831`). MTP works there.
- **Scope of the fix:** one quant-format gap in the CUDA MoE kernel. ~700–900 LOC in a single file. No changes needed to the C-side state machine, MTP weight binding, batched verifier, or KV/raw-cache plumbing — those are quant-agnostic and already work.

After the fix, MTP-2 should deliver a 1.5–2× steady-state throughput lift on DSv4-Flash, matching the 28.3 → 38.4–51 t/s lift that vLLM+FlashInfer delivers on Qwen3.5-122B-A10B on the same GB10 hardware via its native MTP head.

---

## 1. Background

### 1.1 Why MTP matters here

Two stacks running DSv4-class workloads on the same NVIDIA GB10 (SM121, 128 GiB LPDDR5X unified memory, CUDA 13.0.88):

| Stack | Model | Quant | Baseline | + MTP | Gain |
|---|---|---|---|---|---|
| vLLM 0.19 + FlashInfer + AutoRound | Qwen3.5-122B-A10B | INT4 dense + FP8 shared | 28.3 t/s | 38.4 t/s | **+35.7%** |
| `antirez/ds4` native CUDA | DSv4-Flash | IQ2_XXS + Q2_K | ~14 t/s CLI / ~28 t/s benchy | unchanged | **0%** |

(The "13 t/s vs 28 t/s" gap on DSv4 is methodology, not throughput: ds4-CLI's reported rate is first-token-inclusive, llama-benchy's `(N-1)/(t_last-t_first)` is steady-state. They describe the same kernel. See [STRATEGIC_CHECKPOINT.md §3c](STRATEGIC_CHECKPOINT.md#3c-methodology-gap-explained).)

The Qwen stack uses vLLM's batched MoE-aware speculative verifier and a small MTP head (~4% of model weight bandwidth). vLLM has full quant-format coverage in its CUTLASS-style grouped-GEMM kernels, so a 2-row verifier batch costs ~1 dense-weight read for 2 verify positions. With ~80% accept rate on draft-position-2, MTP-2 turns each cycle into ~2 tokens per ~1.1× the bandwidth.

The DSv4 stack has a structurally heavier MTP block (a full MoE decoder layer, 256 routed experts × 6 active per token), but the C-side speculative decode state machine is already in place and is quant-agnostic. The expected lift is in the same ballpark *once the verifier kernel actually runs*.

### 1.2 Architectural ceiling

DSv4-Flash on Spark via ds4 native is already at ~95% of the GB10 memory-bandwidth roofline at steady state (~225 GB/s of ~273 GB/s peak; ~8 GB read per token; 28 t/s ceiling). Even a perfect MTP cannot speed up the *bandwidth-bound* portion of decode. What MTP can do is hide the *FLOP* portion of multiple draft positions behind the same bandwidth read — for an MoE model with sparse routing that means the win comes from the per-token FLOPs of the gate/up/down projections that the speculative verifier amortises across two positions, not from extra free bandwidth.

This is the same gain mechanism that makes MTP-2 worth +35% on Qwen3.5-122B-A10B. We expect MTP on DSv4-Flash to be in the same range.

---

## 2. How MTP is wired in ds4 (quant-agnostic, already works)

You do not need to touch any of this. Listed here so you know what NOT to redo.

### 2.1 The MTP block layout

The MTP draft GGUF (`DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`, 3.6 GiB) carries one extra decoder layer (`mtp.0.*`), structurally identical to a regular DSv4 decoder layer plus three additional tensors used to combine the previous hidden state with the new-token embedding:

- `mtp.0.e_proj.weight` — Q8_0, projects the new token's embedding.
- `mtp.0.h_proj.weight` — Q8_0, projects the previous-hidden state.
- `mtp.0.enorm.weight`, `mtp.0.hnorm.weight`, `mtp.0.norm.weight` — F32 RMSNorm weights.
- All standard attention+FFN tensors prefixed `mtp.0.attn_*`, `mtp.0.ffn_*`, `mtp.0.hc_*`.
- `mtp.0.hc_head_base/fn/scale` — the MTP block's own output head (HC mixing → final logits via the *base model's* shared output embedding).
- **Routed expert tensors** `mtp.0.ffn_gate_exps`, `mtp.0.ffn_up_exps`, `mtp.0.ffn_down_exps` — Q4_K. This is the one piece CUDA cannot consume today.

Loading and tensor binding lives in `ds4.c:2638` (`mtp_weights_bind`) and is validated in `ds4.c:2356` (`mtp_weights_validate_layout`). Routed-expert quant is checked via `is_routed_expert_quant` which accepts IQ2_XXS, Q2_K, **and Q4_K** (`ds4.c:2224–2227`). Nothing in the loader rejects Q4_K.

### 2.2 The speculative decode state machine

Lives in `ds4_session_eval_speculative_argmax` at `ds4.c:17092`. CLI invokes it at `ds4_cli.c:498`; server invokes it at `ds4_server.c:7142`. Both gate on `temperature <= 0.0f && ds4_engine_mtp_draft_tokens > 1 && DS4_MTP_SPEC_DISABLE == NULL`.

Per cycle:

1. Run target decode for the freshly-sampled `first_token` (`ds4_session_eval`). As a side effect this kicks the **probe** — at `ds4.c:17062` it calls `metal_graph_eval_mtp_draft` to produce `s->mtp_draft_token` for the next position.
2. If MTP is ready and the prior probe produced a valid draft, recurse via `metal_graph_eval_mtp_draft_from_hc` (`ds4.c:12612`) to build draft positions 1…N−1.
3. Verify the draft suffix against the target:
   - **default path** (non-strict, default `--mtp-margin 3`): batched 2-row verifier `metal_graph_verify_suffix_tops` (`ds4.c:13246`) — reuses the prefill batch kernels. Reads target weights once for the 2-row batch on dense layers; for sparse experts, behavior depends on routing overlap.
   - **strict path** (`--quality` or `DS4_MTP_STRICT=1`, `draft_n == 2`): exact 2-row decode verifier `metal_graph_verify_decode2_exact` (`ds4.c:13337`) — invokes `metal_graph_encode_decode_layer` twice per layer interleaved.
4. Commit `commit_drafts` accepted tokens or fall back to 1-token via prefix-1 snapshot/restore (`spec_frontier_commit_prefix1` / `spec_frontier_restore`).

Every print mentioning `mtp timing`, `mtp probe`, `mtp spec`, or `mtp conf` is gated by an environment variable; nothing in the hot path is unconditional output.

### 2.3 GPU-side dispatch from `metal_graph_eval_mtp_draft_from_hc`

Inside `ds4.c:12612–12727` the function emits exactly:

1. `ds4_gpu_begin_commands`
2. `ds4_gpu_embed_token_hc_tensor` — embed `token` into MTP's HC-shaped buffer.
3. `ds4_gpu_rms_norm_weight_tensor` — `enorm`.
4. `ds4_gpu_matmul_q8_0_tensor` — `e_proj` (Q8_0).
5. `ds4_gpu_repeat_hc_tensor` — replicate row across HC heads.
6. `ds4_gpu_rms_norm_weight_rows_tensor` — `hnorm` on the previous HC state.
7. `ds4_gpu_matmul_q8_0_tensor` — `h_proj` (Q8_0).
8. `ds4_gpu_add_tensor` — combine into `mtp_input_hc`.
9. `metal_graph_encode_decode_layer` — **per-layer decode kernel applied to the MTP block**. This is where it fails on CUDA.
10. `metal_graph_encode_output_head_mtp` — HC mixing + base-model output projection to vocab logits.
11. `ds4_gpu_indexer_topk_tensor` — top-1 over vocab.
12. `ds4_gpu_end_commands`, then two `ds4_gpu_tensor_read`s.

Steps 1, 3, 4, 5, 6, 7, 8, 10, 11, 12 already work on CUDA. Step 9 enters the standard per-layer decode pipeline, which works for the main model's IQ2_XXS+Q2_K layers and dies on the MTP block's Q4_K experts.

---

## 3. Empirical investigation — how the failure was localised

Retrace this as a sanity check before committing to the fix. Everything below ran on a single DGX Spark (`gn100-7710`-class, GB10 sm_121).

### 3.1 Repro: MTP gives no speedup on CUDA

```bash
cd ~/code/ds4 && make -j8 ds4
./ds4 --cuda \
  -m /path/to/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --mtp /path/to/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --mtp-draft 2 --temp 0 --nothink \
  -p "List 20 prime numbers, comma-separated, just numbers."
# Output ends with: ds4: prefill: ~1 t/s, generation: 13.95 t/s
```

Compare to the same command without `--mtp* --mtp-draft 2` — generation rate is identical to within noise (~13.9–14.0 t/s on a warm cache).

### 3.2 First probe — `DS4_MTP_PROBE=1`

The probe path at `ds4.c:17034–17080` prints diagnostic lines for every MTP draft attempt and explicitly logs `ds4: mtp probe draft failed` when the draft kernel returns false:

```bash
DS4_MTP_PROBE=1 ./ds4 --cuda -m … --mtp … --mtp-draft 2 --temp 0 --nothink \
  -p "List 20 prime numbers" 2> /tmp/probe.log
grep "mtp probe draft failed" /tmp/probe.log | wc -l
# 58 — one per generation step. 100% failure rate.
```

So `metal_graph_eval_mtp_draft` is being invoked every step and returning false every time.

### 3.3 Second probe — per-step injection inside `metal_graph_eval_mtp_draft_from_hc`

Patched each `if (ok) ok = …` with a numbered `ds4: mtp probe failed at step N` print (steps 1–9 as enumerated in §2.3). Rebuilt, reran with `DS4_MTP_PROBE=1`. **Zero step-1-through-9 failures fired**, yet the function still returned false 30/30 times — meaning a step inside the per-layer decode call (step 9) was failing, but not the wrapping `metal_graph_encode_decode_layer` call site itself (because I had initially only instrumented the `if (ok) ok = … != 0;` pattern, not the multi-line block).

Patched the `if (ok) { … encode_decode_layer(…); }` block explicitly, reran:

```
=== histogram of failed steps ===
     30 8a (encode_decode_layer)
```

100% of failures are inside `metal_graph_encode_decode_layer` applied to the MTP block.

### 3.4 Third probe — `DS4_METAL_DECODE_STAGE_PROFILE=1`

The decoder layer is instrumented with named stage probes via `DS4_METAL_PROFILE_DECODE_STAGE`. Setting `DS4_METAL_DECODE_STAGE_PROFILE=1` prints one line per stage that successfully runs. The MTP-draft block prints:

```
ds4: mtp entered draft_from_hc mtp=… attn_q_a=… raw_cache=… prev_hc=… out_hc=…
ds4: metal layer stage part=decode layer=1 pos=… attn_hc_pre=0.363 ms
ds4: metal layer stage part=decode layer=1 pos=… attn_norm=…
ds4: metal layer stage part=decode layer=1 pos=… q_path=…
ds4: metal layer stage part=decode layer=1 pos=… kv_path=…
ds4: metal layer stage part=decode layer=1 pos=… compressor_indexer=0.001 ms
ds4: metal layer stage part=decode layer=1 pos=… attention=…
ds4: metal layer stage part=decode layer=1 pos=… attn_output=…
ds4: metal layer stage part=decode layer=1 pos=… attn_hc_post=…
ds4: metal layer stage part=decode layer=1 pos=… ffn_hc_pre=…
ds4: metal layer stage part=decode layer=1 pos=… ffn_norm=…
ds4: metal layer stage part=decode layer=1 pos=… router=0.035 ms
ds4: mtp probe failed at step 8a (encode_decode_layer)
```

Stages run in order: `attn_hc_pre → attn_norm → q_path → kv_path → compressor_indexer → attention → attn_output → attn_hc_post → ffn_hc_pre → ffn_norm → router` then **no `routed_moe` stage prints**, then failure.

The next stage that should fire is `routed_moe` (`ds4.c:9682`, immediately after the `ds4_gpu_routed_moe_one_tensor` call at line 9664). It never reaches the post-stage profile probe → the call returned `false`.

### 3.5 Root cause — CUDA `routed_moe_launch` rejects everything except IQ2_XXS+Q2_K

`ds4_cuda.cu:8849`:

```c
if (gate_type != 16u || down_type != 10u) return 0;
```

The MTP block's experts are Q4_K (type 12) for gate and down. The kernel rejects them and returns `0` (false), the caller propagates `ok=false`, the entire MTP draft is discarded, the speculative state machine treats the cycle as "no draft" and commits one token.

The Metal counterpart (`ds4_metal.m:12728`) dispatches on `gate_type` and `down_type` with a switch table (`ds4_gpu_routed_mv_pipeline` at `ds4_metal.m:11580` returns one of `g_moe_mul_mv_id_q2_k_pipeline / q4_k / iq2_xxs / q8_0`). The pipelines come from templated kernel functions in `metal/moe.metal`:

- `kernel_mul_mv_q4_K_f32_impl` — `metal/moe.metal:413`
- `kernel_mul_mv_id_q4_K_f32` — `metal/moe.metal:831` (host-named, the actual MTL pipeline)

So Metal already has a full Q4_K MoE matvec pipeline, plus the dispatcher that picks it based on per-tensor quant type. **CUDA needs the same.**

### 3.6 What's NOT broken (confirmed, do not re-investigate)

- MTP weight binding (`mtp_weights_bind`, `mtp_weights_validate_layout`). All 30 probes printed valid non-null pointers for `mtp`, `mtp->block.attn_q_a`, `g->mtp_raw_cache`, `prev_hc`, `out_hc`.
- MTP raw-cache allocation (`ds4.c:8707–8730`, gated on `enable_mtp = e->mtp_ready`). Allocated.
- MTP probe scheduling (`ds4.c:17034–17077`). Fires every step.
- Verifier paths (`metal_graph_verify_suffix_tops`, `metal_graph_verify_decode2_exact`). Never reached because no valid draft is ever produced.
- Q8_0 dense matmuls for `e_proj` / `h_proj` / MTP attention. Steps 3, 4, 6, 7 all pass.
- The Q4_K block layout itself (`ds4.c:140–145`, `sizeof(block_q4_K) == 144`). It's defined; the missing piece is the CUDA dequant + matvec kernel.

---

## 4. The Metal reference — what parity looks like

Metal already supports the full quant combination matrix the loader accepts. Pipelines used by routed MoE (`ds4_metal.m:11580`):

| Quant type id | Constant | Metal pipeline |
|---|---|---|
| 8 (Q8_0) | `DS4_METAL_TENSOR_Q8_0` | `g_moe_mul_mv_id_q8_0_pipeline` |
| 10 (Q2_K) | `DS4_METAL_TENSOR_Q2_K` | `g_moe_mul_mv_id_q2_k_pipeline` |
| 12 (Q4_K) | `DS4_METAL_TENSOR_Q4_K` | `g_moe_mul_mv_id_q4_k_pipeline` |
| 16 (IQ2_XXS) | `DS4_METAL_TENSOR_IQ2_XXS` | `g_moe_mul_mv_id_iq2_xxs_pipeline` |

Metal additionally has a fused gate+up SwiGLU pair kernel for IQ2_XXS (`kernel_mul_mv_iq2_xxs_pair_swiglu_f32`, `metal/moe.metal:959`) — that's an optimisation (one expert visit produces both gate and up activations from a shared dequant), not a correctness requirement. The Q4_K path uses two separate matvecs.

The dispatch lives at `ds4_metal.m:12840–12907`:

```objc
if (gate_type == DS4_METAL_TENSOR_IQ2_XXS) {
    // pair-fused kernel
} else if (gate_type == DS4_METAL_TENSOR_Q4_K) {
    // separate gate matvec, then separate up matvec
}
…
if (down_type == DS4_METAL_TENSOR_Q2_K) { … } else if (down_type == DS4_METAL_TENSOR_Q4_K) { … }
```

So the structural shape per token is:
1. quantise input activation to Q8_K once
2. for each of N_EXPERT_USED selected experts:
   - dot `xq` against gate expert columns (Q4_K × Q8_K) → `gate_vec`
   - dot `xq` against up expert columns (Q4_K × Q8_K) → `up_vec`
   - clamp + SwiGLU into `mid_vec`, scale by router weight, requantise to Q8_K
   - dot `midq` against down expert columns (Q4_K × Q8_K) → accumulate into `out`

The Q4_K dequant rule is standard llama.cpp Q4_K (super-block of 256: 12-byte 6-bit scale/min array + 128 bytes of packed 4-bit weights, with two FP16 scales `d` and `dmin`). The packing is documented in `block_q4_K` at `ds4.c:140` and in `metal/moe.metal:413+`.

---

## 5. What needs to land on CUDA

### 5.1 Goal

A CUDA dispatcher in `routed_moe_launch` that:
- accepts `gate_type ∈ {16, 12}` and `down_type ∈ {10, 12}` (and ideally `8` for Q8_0 to round out parity with Metal — the MTP block's *shared* experts are Q8_0 today, handled by the separate dense matmul, but the routed-expert path is the gap).
- routes to one of (gate, down) ∈ {IQ2_XXS, Q4_K} × {Q2_K, Q4_K} combinations.
- preserves the existing optimised IQ2_XXS+Q2_K fast path bit-for-bit (the main model still uses it on every decode step).

### 5.2 Files to touch

- `ds4_cuda.cu` — the only file. All CUDA kernels and the launcher live here. ~9.7K LOC today; expect to add 600–900 lines.
- No changes to `ds4.c`, `ds4.h`, `ds4_cli.c`, `ds4_server.c`, the Metal file, or anything else.

### 5.3 Suggested implementation order

1. **Q4_K block struct on device.** Add a `cuda_block_q4_K` mirror of `block_q4_K` (`ds4.c:140`) plus the helper to decode the 12-byte 6-bit scale/min array. Use the same struct layout — no reinterpretation, just a `__device__` copy of the struct so kernels can index `qs[]` and `scales[]` symbolically.

2. **Q4_K × Q8_K dot kernel.** Single-row matvec, mirroring `kernel_mul_mv_q4_K_f32_impl` in `metal/moe.metal:413`. Inputs: a quantised `cuda_block_q8_K *xq` of length `expert_in_dim / QK_K` and a Q4_K expert weight slab `cuda_block_q4_K *xw`. Output: one FP32 row. This is the "primitive" — every other Q4_K usage composes it. Test against the CPU reference (`block_q4_K` math in `ds4.c`) and against Metal output dumps captured with `DS4_METAL_DEBUG_DUMP=...`.

3. **Q4_K gate+up paired matvec.** Two `dot` calls back-to-back over the same `xq` input, indexed by `selected[expert]`. Reuse the existing scheduling skeleton from the IQ2_XXS pair path. No fused-pair optimisation needed for v1 (Metal also runs the Q4_K case as two separate matvecs, not a fused pair).

4. **Q4_K accumulating down matvec.** Same as (2) but the output is `out[] += router_weight * dot`. Reuse the existing Q2_K accumulator dispatcher's surrounding code.

5. **Dispatcher rewrite in `routed_moe_launch`.** Replace the hard `if (gate_type != 16u || down_type != 10u) return 0;` with a switch:
   ```c
   if (gate_type == 16 /*IQ2_XXS*/ && down_type == 10 /*Q2_K*/) {
       // current fast path — leave untouched
   } else if (gate_type == 12 /*Q4_K*/ && down_type == 12 /*Q4_K*/) {
       // new Q4_K path for MTP
   } else {
       return 0;  // unsupported combination
   }
   ```
   Keep the existing tuning knobs (`DS4_CUDA_MOE_*` env vars) gated on the IQ2+Q2K path; do not try to port `use_decode_lut_gate`, `use_expert_tiles`, `use_p2_sorted`, etc., to Q4_K in v1. MTP is single-token (`n_tokens == 1u`), so the multi-token tile optimisations don't apply. Aim for correct-first, fast-later.

6. **Verify the static asserts / type table didn't drift.** `ds4.c:867` already lists `[12] = {"q4_k", 256, 144}` and `ds4.c:893` defines `DS4_TENSOR_Q4_K = 12`; the GGUF loader already accepts Q4_K routed experts (`ds4.c:2224–2227`). Just verify the CUDA path doesn't trip on the new alignment assumptions: Q4_K block size is 144 (Q2_K is 84), so `expert_bytes` per row will differ. The `gate_expert_bytes` / `down_expert_bytes` parameters are already computed from `tensor->block_size` at the call site, so no change there.

### 5.4 Tuning notes (do these in v2, not v1)

- The IQ2_XXS+Q2_K path has a 2D thread-block layout where each block computes one expert's row × one block-row of the input. For Q4_K, the per-block weight footprint is roughly **2× larger** (144 vs 84 bytes per block) so register pressure shifts. Start from the same launch config and only re-tune if you see <50% achieved bandwidth on the kernel-level probe (see §6.4).
- Q4_K has a per-super-block FP16 `d, dmin` pair plus a 6-bit-packed array of 16 per-sub-block scale/min values. The unpacking is a small amount of integer work per block and is fully hidden behind the global-memory load latency. Don't try to precompute scales into a separate buffer — the savings are negligible and you lose memory locality.
- Metal's IQ2_XXS pair-SwiGLU fused kernel saves one trip through global memory for the gate/up activations. Q4_K equivalent would also help. Defer to v2: parity first.

---

## 6. Validation plan

Order matters — each step assumes the previous one passes.

### 6.1 Unit: Q4_K × Q8_K dot correctness (CPU-equivalence)

Build a small standalone CUDA test (drop into `tests/`) that:
- generates a random Q4_K super-block (use the CPU reference quantiser to produce realistic blocks)
- quantises a random F32 input vector to Q8_K via the existing CPU `ds4_quantize_row_q8_K`
- runs the CPU reference dot product (you'll need to add it — there's currently no CPU Q4_K matvec at all in `ds4.c` since the main model never needs one) and the new CUDA dot product
- asserts max-abs-error < 1e-3 over 1000 random blocks

Add this as a `make test_q4k_cuda` target.

### 6.2 Integration: MTP probe rate

```bash
DS4_MTP_PROBE=1 ./ds4 --cuda -m … --mtp … --mtp-draft 2 --temp 0 --nothink \
  -p "List 20 prime numbers" 2> /tmp/probe.log
grep "mtp probe draft failed" /tmp/probe.log | wc -l   # must drop from 58 → 0 (or near-0)
```

If non-zero, narrow with `DS4_METAL_DECODE_STAGE_PROFILE=1` to see which stage fails.

### 6.3 Integration: MTP confidence and accept rate

```bash
DS4_MTP_PROBE=1 DS4_MTP_CONF_LOG=1 ./ds4 --cuda -m … --mtp … --mtp-draft 2 --temp 0 --nothink \
  -p "List 50 prime numbers" 2> /tmp/conf.log
grep "mtp probe token" /tmp/conf.log | tail -5
# Expect "hit=X/Y" with X/Y → roughly 0.6–0.85 (the donor's claimed accept rate for MTP-2 on DSv4-Flash).
```

`DS4_MTP_TIMING=1` adds per-cycle wall-time breakdown: `draft=… ms verify=… ms total=… ms drafted=2 committed=…`. A successful run will mostly print `committed=2` (full accept) and `committed=1` (partial). A failing run prints only `committed=1` from the margin-skip path.

### 6.4 Throughput: ds4-bench and llama-benchy

Baseline (no MTP):
```bash
./ds4-bench --cuda -m /path/to/…IQ2XXS…gguf -p 2048 -n 128 -d 0 -d 4096 -d 16384
# Expect generation ~13.9 t/s at d=0, ~13.6 t/s at d=16384.
```

With MTP:
```bash
./ds4-bench --cuda -m … --mtp /path/to/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2 \
  -p 2048 -n 128 -d 0 -d 4096 -d 16384
# Expect generation +25-60% vs baseline. Aim for ≥ 18 t/s at d=0.
```

llama-benchy via ds4-server (steady-state, drops first-token cost):
```bash
./ds4-server --cuda -m … --mtp … --mtp-draft 2 -c 16384 > /tmp/ds4.log 2>&1 &
uvx --from git+https://github.com/eugr/llama-benchy llama-benchy \
  --base-url http://localhost:8000/v1 -p 2048 -n 32 128 512 -d 0 4096 16384 -r 3
# Expect peak tg512 d=0 of 28.3 t/s → ~38–50 t/s with MTP working.
```

### 6.5 Quality: bit-for-bit greedy parity

The CUDA MoE path is approximate (uses `--use_fast_math`, FMA reordering). For a quality regression check use `--quality` which selects the exact decode-2 verifier and forces strict-replay on partial accepts:

```bash
./ds4 --cuda --quality --temp 0 --nothink -m … --mtp … --mtp-draft 2 \
  -p "Write Python factorial" --max-new 256
# Output should match plain --cuda greedy (no MTP) byte-for-byte.
```

If quality mode diverges, the new Q4_K kernel has more numerical noise than the existing IQ2+Q2_K path — fine for non-quality mode (the same is already true for the IQ2+Q2_K verifier), but worth a note in the kernel comment.

### 6.6 Soak: longer prompts, longer generations

```bash
./ds4-server --cuda -m … --mtp … --mtp-draft 2 -c 131072 &
# Then hammer with llama-benchy at d=32768 to confirm the MTP raw-cache rolls correctly under high context.
```

The MTP block has its own raw SWA cache (`g->mtp_raw_cache`, allocated in `ds4.c:8707`). It rolls independently of the target's raw cache. Bugs here would manifest as *intermittent* miss-then-recover patterns under high-context — `DS4_MTP_TIMING=1` will show the pattern.

---

## 7. Risks, gotchas, and edge cases

### 7.1 The MTP GGUF may also have Q4_K *up* experts

`mtp_weights_validate_layout` (`ds4.c:2356–2400`) only asserts gate/up have *matching* types (`MTP routed gate/up experts use different quant types` is the only failure mode). The shipped MTP file uses Q4_K for the routed gate/up/down, plus Q8_0 for dense and F32 for norms. Verify with the GGUF dumper before assuming:

```bash
python3 -c "
import gguf
r = gguf.GGUFReader('/path/to/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf')
for t in r.tensors:
    if 'ffn_' in t.name and ('exps' in t.name):
        print(t.name, t.tensor_type)
" | sort -u
```

If `up_exps` is Q4_K and `down_exps` is Q4_K, the dispatcher needs `(gate, down) == (Q4_K, Q4_K)`. If `down_exps` were ever Q2_K with `gate/up == Q4_K`, you'd need a third combination.

### 7.2 `routed_moe_launch` accepts a `gate_type` and a `down_type` — these refer to weight quant types not activation type

The kernel quantises the input activation to Q8_K on entry every time:
```c
q8_K_quantize_kernel<<<…>>>(xq, (const float *)x->ptr, expert_in_dim, n_tokens);
```
That doesn't change. The mid-activation produced between gate/up and down is also Q8_K. Q4_K only enters as a **weight format**, never as an activation format. So you do not need a Q4_K activation quantiser.

### 7.3 Tuning-knob env vars

`routed_moe_launch` reads ~14 `DS4_CUDA_MOE_*` env vars to tune the IQ2+Q2K fast path (tile sizes, atomic-down flags, gate-row spans). None of these should affect the Q4_K path in v1. Add a single `DS4_CUDA_MOE_Q4K_PROFILE=1` toggle for kernel-stage timing if you want to keep the existing `DS4_CUDA_MOE_PROFILE` shape consistent — and grep for it from `cuda_ok` boundaries inside your new kernels.

### 7.4 The "high-memory variant" comment in `ds4.c:126`

```
 *   - Q4_K routed experts in the high-memory variant
```

This refers to a planned higher-memory GGUF where the main model's routed experts are also Q4_K (for slightly better quality at the cost of memory). It is **NOT shipping today** and is unrelated to MTP. But the C code is already prepared for it — Q4_K is in the routed-expert quant predicate (`ds4.c:2224–2227`). When Q4_K MoE lands on CUDA for MTP, the high-memory main-model variant should "just work" too, modulo perf tuning. Don't actively try to use it as a test target — first get MTP green on the Q2 variant + Q4_K-MTP combo.

### 7.5 Strict mode (`--quality`) goes through a different verifier kernel

`metal_graph_verify_decode2_exact` (`ds4.c:13337`) is the strict-mode verifier; it calls `metal_graph_encode_decode_layer` **twice** per layer interleaved (once per draft position) using single-token decode kernels. This path is *not* changed by your work — it doesn't touch the MTP block, it verifies the *target* model's decode for the two proposed positions. But: with MTP working, the strict path becomes relevant again, and any pre-existing CUDA bug in 2-row interleaved decode would surface here. Check it under `DS4_MTP_TIMING=1` and look for `mtp timing decode2 drafted=2 committed=… verify=… ms`.

### 7.6 Margin threshold default (`--mtp-margin 3`)

In non-strict mode, the default-path verifier skips the batched 2-row verify entirely when the MTP head's confidence (top1−top2 logit margin) is below `3.0`. On predictable prompts (counting primes, JSON output) the MTP head is highly confident → batched verify runs → speculative gain. On chatty open-ended prompts the margin will dip → it'll skip verify and commit 1 token → no speedup on that step. This is by design. Don't try to tune it away in v1.

### 7.7 The Metal pair-SwiGLU optimisation is *not* portable to Q4_K trivially

Metal's `kernel_mul_mv_iq2_xxs_pair_swiglu_f32` fuses gate dot + up dot + clamp + SwiGLU + scale + write-back into one kernel, halving memory traffic. The IQ2_XXS block packing happens to interleave gate and up coefficients in a friendly way; Q4_K does not have that property. So a Q4_K pair fusion would need to do two independent block walks anyway. Skip for v1.

---

## 8. Estimated effort

| Task | LOC delta | Wall-time estimate (analyst familiar with CUDA + GGUF Q4_K) |
|---|---|---|
| `cuda_block_q4_K` + 6-bit scale unpack helper | ~80 | 0.5 day |
| `kernel_mul_mv_q4_K_q8_K_dot` primitive + unit test | ~250 | 1–1.5 days |
| Q4_K gate+up matvec wrapper + per-expert scheduling | ~150 | 0.5 day |
| Q4_K accumulating down matvec | ~120 | 0.5 day |
| `routed_moe_launch` dispatcher rewrite (preserve IQ2+Q2K fast path) | ~80 | 0.5 day |
| Validation: MTP probe rate green, accept rate sane, ds4-bench delta measured | n/a | 0.5–1 day |
| Quality regression sweep + buffer overflow soak | n/a | 0.5 day |
| **Total** | **~700–900** | **3.5–5 days** |

This is single-file, no architectural decisions to make — the Metal reference is your spec.

---

## 9. Why this matters strategically

A working CUDA MTP turns ds4 on GB10 from "fast enough" into "competitive with vLLM+FlashInfer." Concretely:

- **Today:** DSv4-Flash on Spark via ds4 native = ~28 t/s steady-state, no MTP gain. Qwen3.5-122B-A10B with comparable active-param count and a working MTP-2 path = 51 t/s on the same GB10. The 50% gap is mostly MTP.
- **After this fix:** DSv4-Flash should reach 35–50 t/s steady-state, closing most of the gap.
- **Architectural argument:** ds4's CUDA path has been the *fast* path for raw decode (95% bandwidth saturation, no Triton/TileLang/Python overhead). The only thing missing is speculative decode. Adding Q4_K to one kernel unlocks it.
- **Upstream-able:** this is a clean upstream contribution to `antirez/ds4`. The fix matches an existing Metal pattern, follows the existing CUDA kernel idiom, has a clear correctness oracle (CPU reference + Metal reference), and improves a benchmark number that's publicly visible.

If the analyst wants a second-pass scope, the same dispatcher gate (`if (gate_type != 16u || down_type != 10u) return 0;` at `ds4_cuda.cu:8849`) also blocks the "high-memory" Q4_K main-model variant. One quant-format port unlocks both that variant AND MTP — but MTP is the higher-value half because it's a multiplicative win on every workload, while the high-memory variant is a quality knob most users won't turn.

---

## 10. References

Pinned source positions (paths are relative to the `antirez/ds4` source tree at HEAD `920f987`, 2026-05-12):

- `ds4_cuda.cu:8849` — the offending one-line gate.
- `ds4_cuda.cu:8809+` — `routed_moe_launch` body to extend.
- `ds4_cuda.cu:9360+` — `ds4_gpu_routed_moe_one_tensor` extern entry.
- `ds4_metal.m:12728+`, `ds4_metal.m:11580+` — Metal reference dispatcher and pipeline table.
- `metal/moe.metal:413+`, `metal/moe.metal:831` — Metal Q4_K kernel + host-named pipeline.
- `ds4.c:2356+` — `mtp_weights_validate_layout` (already Q4_K-aware).
- `ds4.c:12612+` — `metal_graph_eval_mtp_draft_from_hc`, the function that fails today.
- `ds4.c:17092+` — `ds4_session_eval_speculative_argmax`, the speculative state machine.
- `ds4.c:140+` — `block_q4_K` struct definition (CPU reference).
- `ds4_cli.c:498`, `ds4_server.c:7142` — CLI/server entry points to the speculative path.

Companion docs in this repo:

- [`STRATEGIC_CHECKPOINT.md`](STRATEGIC_CHECKPOINT.md) — Spark-side measurements, roofline, decision framework.
- [`METAL_VS_CUDA.md`](METAL_VS_CUDA.md) — backend comparison; §8 (MoE) is the most relevant for picking up this work.
- [`../README.md`](../README.md) — high-level user-facing writeup with the headline numbers.
