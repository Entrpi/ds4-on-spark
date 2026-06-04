# Metal vs CUDA in ds4 — A Side-by-Side Comparison

A focused comparison of the two GPU backends in DwarfStar 4: `ds4_metal.m` (+19 `.metal` kernel files, ~22.8k lines total) on macOS, and `ds4_cuda.cu` (+ `ds4_iq2_tables_cuda.inc`, ~9.7k lines) on Linux/NVIDIA. Both implement the same public C surface in `ds4_gpu.h`, so this is a study in *how* each platform realises the same contract.

> **Where this lives.** This analysis is paired with the empirical [Spark benchmark + roofline writeup](STRATEGIC_CHECKPOINT.md). The roofline writeup says *what* the CUDA path does on Spark hardware; this document says *why* the implementation looks the way it does, and what would (and wouldn't) change if you considered porting or forking. See the [README](../README.md#under-the-hood-how-the-cuda-backend-works) for the TL;DR.
>
> **Source references.** All file paths and line numbers in this document refer to upstream [`antirez/ds4`](https://github.com/antirez/ds4) at HEAD `920f987` (2026-05-12). They are intentionally not clickable links — the source is not vendored into this repo.
>
> **Naming note.** The codebase does not use Apple's MLX framework. The macOS path is **direct Metal** (Objective-C `ds4_metal.m` plus hand-written `.metal` compute shaders). Throughout this document "Metal" means that backend.

---

## 1. The contract both must satisfy

ds4_gpu.h defines what both backends owe the engine:

- One opaque `ds4_gpu_tensor` type with alloc / view / free / host-pointer accessor / write / read / device-copy.
- A four-call command lifecycle: `begin_commands → (dispatches) → flush_commands → end_commands`, plus `synchronize` for ad-hoc waits.
- Model attach functions: `set_model_map(ptr, size)`, `set_model_map_range(...)`, `set_model_fd(fd)`, `cache_model_range(...)`, `cache_q8_f16_range(...)`.
- A bank of kernel calls grouped by purpose: embedding/indexer, dense matmul (Q8_0 / F16 / F32 / pair-output), MoE (router + per-expert matmul), RoPE tail-only, RMS norm, flash attention, DSV4-specific HC split/expand/mix, KV FP8 round/store, compressor store, ratio-4 shift, directional steering, softmax/argsort/unary, get/set rows, etc. (For routed-expert quant coverage the two backends are **not** at parity today — Metal accepts Q8_0 / Q2_K / Q4_K / IQ2_XXS, CUDA accepts only IQ2_XXS+Q2_K. This is what breaks MTP on CUDA; see §8 and [`MTP_PARITY_GAP.md`](MTP_PARITY_GAP.md).)
- `print_memory_report(label)` and `set_quality(bool)`.

Different platforms; same surface area. Beneath that, the implementation strategies diverge sharply.

---

## 2. Tensor object model

| | Metal | CUDA |
|---|---|---|
| Storage type | `DS4MetalTensor` Objective-C class wrapping `id<MTLBuffer>` (ds4_metal.m:203-211) | Plain C struct `{ void *ptr; uint64_t bytes; int owner; }` (ds4_cuda.cu:27-31) |
| Allocation primitive | `[g_device newBufferWithLength: options: MTLResourceStorageModeShared]` | **`cudaMallocManaged()`** — unified memory, on-demand page migration |
| View | New `DS4MetalTensor` sharing the parent's `MTLBuffer`, with `offset += view_offset`; `owner=0` | New tensor struct with `ptr = base->ptr + offset`; `owner=0` |
| `tensor_contents()` returns | Host-mapped pointer: `[buffer contents] + offset`. Direct memcpy works because Shared storage is CPU-visible. | `tensor->ptr` directly — but **after `cudaDeviceSynchronize()`** (ds4_cuda.cu:1159-1162), so the CPU sees coherent results |
| Coherency model | Single shared physical page; GPU stalls if the CPU is writing during a kernel | UVM-managed: pages migrate on access; an explicit sync guarantees coherence |
| Owner flag | `owner=1` on alloc, `0` on view; freed only when `owner=1` | Same convention |
| Allocation tracking | Global counters `g_tensor_alloc_live_bytes`, `g_tensor_alloc_peak_bytes` (ds4_metal.m:3781-3790); optional per-alloc trace via `DS4_METAL_TRACE_ALLOCS` | None at the tensor layer; cumulative bytes tracked only for the model-range and Q8→F16 caches |
| Memory mode | All `MTLResourceStorageModeShared` (no distinction between Managed and Private) | All `cudaMallocManaged()` (no distinction between `cudaMalloc` device-only and pinned host) |

**Why it matters.** Both backends present "one pointer that both sides can read." On Metal this works because Apple Silicon physically shares RAM with the iGPU and the M-series page tables map the same page into both worlds. On CUDA it works because UVM transparently migrates 2 MB pages on demand. The user-facing API is identical; the cost models are not. On Metal a CPU write during a running kernel is a stall; on CUDA the page can be torn between the CPU and the (possibly dGPU) GPU, which is why every contents-access goes through a `cudaDeviceSynchronize()` first.

---

## 3. Command lifecycle and synchronization

This is the largest structural difference between the two backends.

### Metal: explicit batching, one encoder per command buffer

```objc
static id<MTLCommandQueue>  g_queue;      // single queue
static id<MTLCommandBuffer> g_batch_cb;   // one in flight
static id<MTLComputeCommandEncoder> g_batch_enc; // lazy
static NSMutableArray<id<MTLCommandBuffer>> *g_pending_cbs;
static NSMutableArray<id<MTLBuffer>> *g_transient_buffers;
```
- `ds4_gpu_begin_commands()` creates `g_batch_cb` (ds4_metal.m:3895-3900); it refuses to nest.
- The first kernel dispatch in a batch lazily creates `g_batch_enc`. Subsequent dispatches reuse it.
- `ds4_gpu_tensor_copy()` is a blit op: it **closes the compute encoder**, opens a blit encoder, then ends the blit encoder (ds4_metal.m:3883-3891). Next compute dispatch creates a fresh compute encoder. This explicit encoder lifecycle is how Metal avoids implicit compute-vs-blit reordering.
- `ds4_gpu_flush_commands()` commits the current CB, pushes it onto `g_pending_cbs`, and starts a new CB (ds4_metal.m:3902-3919). This lets the CPU keep submitting while the GPU drains.
- `ds4_gpu_end_commands()` commits and `waitUntilCompleted` blocks until done. Transient buffers are then reclaimed.
- `g_transient_buffers` is the arena for CPU-written scratch (mask tables, broadcast constants). They have to live until the CB completes, then they're freed.

### CUDA: no batching abstraction

```c
int ds4_gpu_begin_commands(void)  { return 1; }                              // no-op
int ds4_gpu_flush_commands(void)  { return cuda_ok(cudaDeviceSynchronize()); }
int ds4_gpu_end_commands(void)    { return cuda_ok(cudaDeviceSynchronize()); }
int ds4_gpu_synchronize(void)     { return cuda_ok(cudaDeviceSynchronize()); }
```
(ds4_cuda.cu:1190-1193)

Every kernel call goes to the default stream immediately. `begin_commands` is a no-op. `flush`, `end`, and `synchronize` are all `cudaDeviceSynchronize()` — a full device barrier.

There are two named streams (`g_model_prefetch_stream`, `g_model_upload_stream`, ds4_cuda.cu:65-66) but they're used only for asynchronous model staging; compute work runs on the default stream.

### Trade-off

|  | Metal | CUDA |
|---|---|---|
| Submission cost | Explicit batching; multiple kernels share a CB and an encoder | Each kernel launch is its own submit; CUDA's launch overhead is small enough that this is cheap |
| Blocking points | Only at `end_commands` (or on a flush that fills the pending list) | At every `flush`/`end`/`synchronize`, the host blocks until the GPU is idle |
| In-flight GPU work | Multiple committed CBs can be in flight simultaneously via the pending list | Default stream is strictly ordered; only the model-staging streams overlap |
| Encoder concept | Compute / blit encoders must be explicitly switched | No encoder concept; same kernel-vs-memcpy distinction is enforced by the runtime |
| Practical implication | The engine can fire dozens of kernels per "batch" with one CPU/GPU round trip | Each `cudaDeviceSynchronize` is a hard barrier — the CPU effectively walks the GPU one batch at a time |

Both designs match the engine's single-worker model. CUDA leans on cheap launch latency; Metal leans on encoder batching.

---

## 4. Model memory mapping — the most platform-specific subsystem

This is where the engine philosophies diverge most. The GGUF model is ~81 GB (q2) or ~153 GB (q4) — far too large for any single GPU buffer.

### Metal: mmap'd GGUF wrapped as overlapping MTLBuffer views

ds4_metal.m:408-531.

1. The CPU mmaps the GGUF file (in `ds4.c`); `set_model_map(ptr, size)` is then called.
2. Metal's `maxBufferLength` caps a single MTLBuffer at roughly 704 MB on Apple Silicon (constant `DS4_METAL_MODEL_MAX_TENSOR_BYTES`, ds4_metal.m:190).
3. The backend creates **N overlapping `MTLBuffer` views** over the mmap, each via `newBufferWithBytesNoCopy:length:options:deallocator:`. The overlap is `round_up(MAX_TENSOR_BYTES, page) + page` (ds4_metal.m:432-491).
4. Because every tensor is ≤ MAX_TENSOR_BYTES, the overlap guarantees that **any single tensor lies wholly within one view** — so each kernel dispatch needs exactly one MTLBuffer reference plus an inner offset. No cross-buffer scatter is ever needed.
5. On macOS 15+: `MTLResidencySet` is created, all model views are added, and `requestResidency` is called (ds4_metal.m:371-406). The driver prefaults GPU page tables, avoiding cold-start TLB stalls. Disabled via `DS4_METAL_NO_RESIDENCY`.
6. Optional warmup: `kernel_touch_u8_stride` reads one byte per stride (default 1 MB, override `DS4_METAL_MODEL_WARMUP_STRIDE_MB`) to materialise mmap pages and exercise the residency machinery (ds4_metal.m:676-740).

Result: **zero copy from disk to GPU**. The GGUF stays mmap'd; Metal sees the same physical pages via `BytesNoCopy`. Memory cost on the GPU side is essentially the address-space mapping plus the residency hint.

### CUDA: three-tier strategy with explicit fallback

ds4_cuda.cu:1195-1313, with chunked-range support at ds4_cuda.cu:173-278.

CUDA has no equivalent of `BytesNoCopy` on a file mmap. The backend tries progressively more expensive strategies until one works:

**Tier 1 — full-model copy** (gated by `DS4_CUDA_COPY_MODEL`, ds4_cuda.cu:1226-1248):
- `cudaMalloc(model_size)` then `cudaMemcpy(..., HostToDevice)` for the entire ~81 GB.
- Works only on GPUs with enough VRAM (DGX-class, A100 80GB, H100, etc.).
- After this, every weight access is a direct device pointer.

**Tier 2 — host-register + device pointer** (default first attempt, ds4_cuda.cu:1251-1268):
- `cudaHostRegister(model_map, model_size, cudaHostRegisterMapped | cudaHostRegisterReadOnly)` pins the mmap'd pages and maps them into device address space.
- `cudaHostGetDevicePointer()` returns the device-side virtual address.
- The GPU pages in over PCIe (or NVLink/C2C on integrated/GH systems) on access; the ReadOnly flag lets CUDA cache aggressively.
- If `cudaHostRegister` fails (file-backed mmaps sometimes can't be pinned, or the host pinned-memory limit is hit), this tier drops out.

**Tier 3 — per-range lazy chunked copy** (`cuda_model_range_ptr`, ds4_cuda.cu:173-278):
- For each weight range requested, look it up in `g_model_range_by_offset`. If unknown:
- First try a page-aligned `cudaHostRegister()` on just that range.
- If that fails: `cudaMalloc(range_bytes)` then `cudaMemcpy` from host in **64 MiB chunks** (default, override via `DS4_CUDA_MODEL_COPY_CHUNK_MB`).
- Cache the device pointer; subsequent uses are direct.

**Optional bonus — direct I/O via `set_model_fd`** (ds4_cuda.cu:1281-1313):
- On Linux, opens `/proc/self/fd/{fd}` with `O_DIRECT | O_RDONLY` to enable kernel-level DMA reads that bypass the page cache.
- Lets the range fetcher pull weights straight from disk without going through host RAM as a buffer.
- This is the CUDA-side analog of Apple's "the mmap *is* the buffer" — but it's a Linux-only escape hatch, not the default.

### Comparison

| | Metal | CUDA |
|---|---|---|
| Default zero-copy? | **Yes**, via mmap + BytesNoCopy + overlapping views | **No** — `cudaHostRegister` is the closest, but requires the host pages to actually be pinnable; otherwise CUDA must materialise data on device |
| Max single buffer | ~704 MB → solved by overlapping views | No relevant limit on device-allocated buffers |
| Prefault hint | `MTLResidencySet` + `requestResidency` (macOS 15+) | None equivalent; OS/driver handles page faults on access |
| Warmup | Optional stride-touch | Optional prefetch on `g_model_prefetch_stream` for `set_model_map_range` |
| Failure mode | Mmap-and-wrap always succeeds (cheap operation) | Three-tier fallback because some configurations can't pin/register |
| Knobs | `DS4_METAL_NO_RESIDENCY`, `DS4_METAL_NO_MODEL_WARMUP`, `DS4_METAL_MODEL_WARMUP_STRIDE_MB` | `DS4_CUDA_COPY_MODEL`, `DS4_CUDA_COPY_MODEL_CHUNKED`, `DS4_CUDA_MODEL_COPY_CHUNK_MB`, `DS4_CUDA_DIRECT_MODEL`, `DS4_CUDA_NO_FD_CACHE`, `DS4_CUDA_NO_DIRECT_IO` |

**Where this leads.** Metal's strategy works because Apple Silicon is unified memory: the mmap'd pages *are* the GPU's pages, just with different page tables. CUDA's hierarchy reflects that for a discrete GPU the model must somehow cross the bus, and the engine tries to avoid (or amortize) that crossing.

---

## 5. Quantisation handling

The model carries Q8_0 (most dense matmuls), F16 (LoRA projections), F32 (norms), and routed-expert quants IQ2_XXS / Q2_K / Q4_K. How each backend handles these is structurally different — and the two backends do **not** cover the same set of routed-expert quants today (see §8).

### Metal: always inline dequant in the kernel

For every matvec/matmul kernel, weights are loaded in their stored format and dequantised on the fly inside the kernel. There is **no pre-conversion cache**.

- Q8_0 matvec: metal/dense.metal:109-176 — `kernel_mul_mv_q8_0_f32_impl<NR0>` loads `block_q8_0` (`half d; int8_t qs[32]`), SIMDGROUP-reduces the dot product, multiplies by the scale `d` inline.
- IQ2_XXS expert: the 256-entry codebook is embedded in the Metal source as a `static constant ulong[256]` (metal/moe.metal:23-88). The expert matvec kernel reads it directly from the kernel's constant address space.
- Q2_K / Q4_K: templated `dequantize_q2_K` / `dequantize_q4_K` inline functions (metal/moe.metal:202-243) called inside the expert matmul.

### CUDA: pre-convert Q8_0 → F16/F32 once, hand off to cuBLAS

CUDA explicitly maintains **per-tensor pre-converted weight caches** so that the cuBLAS GEMM path can be used for prefill (multi-token) matmuls.

- `ds4_gpu_cache_q8_f16_range(model_map, size, offset, bytes, in_dim, out_dim, label)` (ds4_cuda.cu:1323-...) allocates an `out_dim × in_dim × sizeof(__half)` device buffer and launches `dequant_q8_0_to_f16_kernel` (ds4_cuda.cu:395) to fill it once at startup.
- `ds4_gpu_cache_q8_f32_range` does the same for F32 (gated by `DS4_CUDA_Q8_F32_PRELOAD`).
- The result is stored in `g_q8_f16_ranges` and indexed by offset in `g_q8_f16_by_offset` so subsequent kernel dispatches can find the cached F16 view.
- Dispatch logic (ds4_cuda.cu:4867 and friends): if `g_cublas_ready && n_tok > 1` and a cached F16/F32 view exists, the call routes to `cublasGemmEx` / `cublasSgemm`. Otherwise, fall through to the hand-written Q8_0 kernel (which is also fine for single-token decode where the cuBLAS overhead doesn't pay back).
- The cache is capped by `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB`; verbose logs via `DS4_CUDA_WEIGHT_CACHE_VERBOSE`; strict refusal via `DS4_CUDA_STRICT_WEIGHT_CACHE`.
- For IQ2_XXS, the codebook lives in `__constant__` memory via `ds4_iq2_tables_cuda.inc`:
  ```c
  __device__ __constant__ uint8_t  cuda_ksigns_iq2xs[128];
  __device__ __constant__ uint64_t cuda_iq2xxs_grid[256];
  ```
  The dequant decoder reads these on every expert dot, just like Metal — there's no pre-conversion for routed experts (which would defeat the 2-bit memory win).

### Why the asymmetry

The trade-off is precisely: a discrete NVIDIA GPU has **tensor cores** that accelerate F16 GEMM by an order of magnitude. Pre-dequantising Q8 to F16 once and paying that conversion in a startup kernel buys access to those tensor cores for every subsequent prefill. Apple Silicon has no comparable hardware path that's worth the conversion cost; the SIMDGROUP dequant-and-MAC loop in the Metal kernel is already close to memory-bandwidth-bound.

So the same Q8 weight tensor becomes:
- **Metal**: read from mmap, dequantised in registers, used once, never materialised in F16.
- **CUDA**: read from mmap (or registered host memory), dequantised once into a device F16 buffer that lives for the rest of the process, used by cuBLAS Hgemm thereafter.

For the **routed experts** — which dominate the model's memory footprint — both backends keep weights quantised and dequant inline. Pre-converting all 256 experts to F16 would defeat the purpose of having a 2-bit model.

The **set of routed-expert quants each backend can consume diverges** today:

|  | Metal | CUDA |
|---|---|---|
| IQ2_XXS routed | ✓ | ✓ |
| Q2_K routed | ✓ | ✓ |
| Q4_K routed | ✓ — base (`metal/moe.metal:413` + `:831`), fused gate+up+SwiGLU (`:1160`), fused n_expert=6 down+sum (`:1336`) | ✓ — MTP decode shape (n_tokens=1, n_expert=6) via `routed_moe_launch`'s Q4_K path, incl. fused n_expert=6 down+sum (`moe_down_q4K_sum6_qwarp32_kernel`); wider shapes via the mmq path |
| Q8_0 routed | ✓ | (unused; falls through) |

This **used to** be why MTP didn't work on CUDA: the MTP support GGUF stores its routed experts in Q4_K, and CUDA's MoE launcher rejected anything but IQ2_XXS+Q2_K, so the C-side speculative state machine saw every draft fail and treated it as "no draft available." That gap is now closed — a Q4_K path was added to `routed_moe_launch`, so MTP produces drafts and is bit-exact to non-MTP. MTP is still off by default because it's a single-stream throughput loss (the exact verifier sweeps the weights ~2× per cycle); see [`MTP_PARITY_GAP.md`](MTP_PARITY_GAP.md).

---

## 6. Kernel organisation

| | Metal | CUDA |
|---|---|---|
| Source layout | 19 `.metal` files in `metal/`, concatenated at runtime into one `MTLLibrary` (ds4_metal.m:1198-1265) | One 9.7k-line `ds4_cuda.cu` containing **all** kernels |
| Variant strategy | `MTLFunctionConstantValues` at indices 100/125/224/225/300+/600/601 (named: `NSG`, `NXPSG`, `nqptg`, `ncpsg`, has_mask, has_sinks, …). One source, many specialisations | Explicit named kernels per variant. ~106 `__global__` functions |
| C++ templating | None (Metal source is C-like) | Limited; mostly explicit instantiation, some template parameters for `NR0`/grid sizes |
| Library compilation | At process start, source → `MTLLibrary` once, then `MTLComputePipelineState` cached per `(function, constants)` key in `g_pipeline_cache` (ds4_metal.m:554-647) | At build time by `nvcc`, one cubin per `CUDA_ARCH`. Pipelines are implicit (PTX) |
| Source override knobs | One env var per file: `DS4_METAL_FLASH_ATTN_SOURCE`, `DS4_METAL_DENSE_SOURCE`, `DS4_METAL_MOE_SOURCE`, …, for swap-in diagnostics (ds4_metal.m:1198-1265) | None — kernel set is fixed at compile time |
| Threadgroup / blocksize | Function constants and dispatch-time `threadgroupMemoryLength` | `<<<grid, block>>>` literals or env-driven, e.g., `matmul_q8_0_preq_warp8_kernel<<<(out_dim+7)/8, 256>>>` (ds4_cuda.cu:4931) |

The Metal pattern is "one source, many specialisations, runtime-compiled, cached by signature." The CUDA pattern is "many concrete kernels, compiled ahead of time, dispatch chooses one of them." Both reach the same destination, but the Metal approach is more naturally tunable at runtime (swap one `.metal` file, restart), while the CUDA approach is more naturally optimal for any *particular* shape it has been hand-tuned for.

### Library usage (cuBLAS) — CUDA-only

CUDA links `-lcublas` (Makefile:29) and uses cuBLAS for the prefill matmul path:
- `cublasSgemm` / `cublasGemmEx` for F16/F32 matmuls when `n_tok > 1` (ds4_cuda.cu:4867-5247).
- `cublasSgemmStridedBatched` for raw-attention score computation (ds4_cuda.cu:5832-5866).
- Math mode defaults to `CUBLAS_TF32_TENSOR_OP_MATH`; `--quality` and `DS4_CUDA_NO_TF32` switch back to strict FP32.
- Single-token decode (`n_tok == 1`) and routed-expert matmuls always use the hand-written kernels — cuBLAS launch overhead doesn't amortize over one row.

Metal has no equivalent. Apple does not ship a tensor-core analog as a callable library, and the Metal Performance Shaders matmul kernels weren't a fit for the engine's quantised path. Everything is hand-rolled.

---

## 7. Flash attention

Both backends implement DS4's specialised attention (raw sliding window + ratio-4 indexer + ratio-128 compressed). The shape differs.

### Metal: four-phase pipeline

metal/flash_attn.metal (1426 lines) and helper sources:

1. **`kernel_flash_attn_ext_pad`** (metal/flash_attn.metal:139-207) — pads partial K/V blocks to 32-row multiples.
2. **`kernel_flash_attn_ext_blk`** (metal/flash_attn.metal:208-888) — tile-based block compute; function constants control `nqptg` (queries per thread group) and `ncpsg` (context per simdgroup).
3. **`kernel_flash_attn_ext`** (metal/flash_attn.metal:889-960) — full FA-2-style attention with mask, sink tokens, bias, and softcap support.
4. **`kernel_flash_attn_ext_vec`** + reduce companion (metal/flash_attn.metal:961-1385) — vectorised variant for larger blocks, with a separate reduce kernel.

Function constants (indices 300–322) select code paths at pipeline build time without source branching.

### CUDA: monolithic, one-block-per-(token,head)

ds4_cuda.cu:2268-3367.

- `attention_prefill_raw_kernel` (ds4_cuda.cu:2268) — one block per `(token, head)`, 256 threads, no tile-loop subdivision. Scores → softmax → V-weighted average all in one kernel.
- `attention_prefill_mixed_kernel` (ds4_cuda.cu:2324) — combines raw window with compressed rows.
- `attention_indexed_mixed_kernel` (ds4_cuda.cu:2739) — for ratio-4 layers, consumes the indexer's top-K mask.
- `attention_indexed_mixed_heads8_*` (ds4_cuda.cu:2899-3243) — specialised for 8-heads-per-block grouping.
- `attention_decode_mixed_*` (ds4_cuda.cu:2571-3367) — single-token decode variants.

The work isn't sliced across multiple kernels. The score matmul for raw attention is delegated to `cublasSgemmStridedBatched` (ds4_cuda.cu:5832-5866); softmax + weighted V is done inline in the attention kernel.

Trade-off: Metal's four-phase split lets the compiler specialise each phase tightly (the pad kernel is tiny, the block kernel is hot, the reduce kernel is final), and function constants make ten variants of each cheap. CUDA's monolithic style relies on the compiler and the launch config to do the same job; for FA, hand-tuning a single big kernel per shape works well enough on a stable hardware target.

---

## 8. MoE (router + 256 experts × top-6)

Both backends share the high-level scheme: a router selects 6 experts per token from 256, weights them, then 6 expert matvecs run per token (gate, up, SwiGLU, down) and the results are summed.

| | Metal | CUDA |
|---|---|---|
| Router | `kernel_dsv4_router_select_*` family (metal/dsv4_misc.metal) | `router_select_kernel` (ds4_cuda.cu:3903) + `_warp_topk_kernel` (ds4_cuda.cu:4017) + `_parallel_kernel` (ds4_cuda.cu:3955) |
| Expert dispatch geometry | One simdgroup per (token, slot), expert index read from router output | One block per (token, slot, output row) — `moe_gate_up_mid_kernel` (ds4_cuda.cu:7137), `_warp8_kernel` (ds4_cuda.cu:7196) |
| Quant types accepted by the MoE matvec | Q8_0 / Q2_K / Q4_K / IQ2_XXS (dispatched at `ds4_metal.m:11580`, kernels in `metal/moe.metal` lines 321/413/522 etc.) | IQ2_XXS+Q2_K, **plus Q4_K** for the MTP decode shape (added via `routed_moe_launch`'s `q4k_path`; the old `gate_type != 16u \|\| down_type != 10u` hard-reject now fires only for genuinely unsupported combos). Wider Q4_K shapes route through the mmq path. |
| IQ2_XXS dequant | Inline in the expert matvec; codebook in `constant` address space | Inline; codebook in `__constant__` memory via `dev_dot_iq2_xxs_q8_K_block_lut()` (ds4_cuda.cu:6722-6776) |
| Output reduction | Per-token accumulation in scratch | Atomic-add (`DS4_CUDA_MOE_ATOMIC_DOWN`) or non-atomic (`DS4_CUDA_MOE_NO_ATOMIC_DOWN`) — runtime-selectable |
| SwiGLU + weight fusion | `kernel_dsv4_moe_swiglu_weight` + `_f16` variants (metal/moe.metal:129-199); both IQ2_XXS and Q4_K additionally ship fused gate+up+SwiGLU kernels (`kernel_mul_mv_iq2_xxs_pair_swiglu_f32` at `metal/moe.metal:959`, `kernel_mul_mv_id_q4_K_pair_swiglu_f32` at `:1160`), and Q4_K also has a fused n_expert=6 down+sum (`kernel_mul_mv_id_q4_K_sum6_f32` at `:1336`) used by the MTP fast path | Inlined into the gate/up/mid kernel for the IQ2_XXS+Q2_K path; the Q4_K MTP path is implemented (paired gate/up matvec + fused n_expert=6 down+sum, `moe_down_q4K_sum6_qwarp32_kernel`) |
| Tuning knobs | Function constants on the kernels | `DS4_CUDA_MOE_GATE_ROW`, `_MOE_NO_GATE_ROW`, `_MOE_ATOMIC_DOWN`, `_MOE_NO_ATOMIC_DOWN`, `_MOE_PROFILE` (all on the IQ2_XXS+Q2_K fast path) |

Both implementations keep routed experts quantised throughout (no F16 pre-conversion). This is the choice that makes 2-bit ds4 useful: the experts are ~80% of the model bytes and converting them would erase the q2 win.

**This Q4_K gap used to break MTP on CUDA; it is now closed.** The MTP support GGUF stores its routed experts in Q4_K (the main model uses IQ2_XXS+Q2_K). The CUDA MoE launcher used to reject the Q4_K call and return failure, so the C-side state machine saw "no draft available" and fell back to one-token decode — which is why MTP "looked alpha" externally even though the C-side machinery was sound. A Q4_K path was since added (mirroring Metal's `gate_type=Q4_K, down_type=Q4_K` dispatch to `g_moe_mul_mv_id_q4_k_pipeline`), so MTP now produces drafts on CUDA and is bit-exact to non-MTP.

That Q4_K dispatch case has landed in `routed_moe_launch`. MTP is no longer correctness-blocked on CUDA — but it is a single-stream throughput loss (the exact verifier sweeps the weights ~2× per cycle), so the highest-leverage MTP work now is **batched serving**, not another kernel. See [`MTP_PARITY_GAP.md`](MTP_PARITY_GAP.md).

---

## 9. DSV4-specific kernels (HC, RoPE, KV FP8, indexer, compressor, steering)

These translate one-for-one between backends; semantics are identical because they implement the model's published recipe. Implementation differences are stylistic.

| Operation | Metal kernel(s) | CUDA kernel(s) |
|---|---|---|
| HC split with Sinkhorn balance | `kernel_dsv4_hc_split_sinkhorn` (metal/dsv4_hc.metal:83-247); `float4` fast path for HC=4 | Equivalent functionality in HC kernels in `ds4_cuda.cu` |
| HC weighted sum / expand | `kernel_dsv4_hc_weighted_sum`, `_expand_*`, `_split_weighted_sum_norm` | `embed_token_hc_kernel`, `repeat_hc_kernel`, `hc_expand_kernel` (ds4_cuda.cu:1359-3640) |
| Tail-only RoPE with YaRN | `kernel_dsv4_rope_tail_f32` (metal/dsv4_rope.metal:68-155); NeoX + interleaved + inverse modes | `rope_tail_kernel`, `head_rms_norm_rope_tail_kernel` (ds4_cuda.cu:2044-2164) |
| FP8 (E4M3FN) KV quantise/dequantise | `kernel_dsv4_fp8_kv_quantize_f32`, `_kv_fp8_store_f32` (metal/dsv4_kv.metal:79-178); E4M3FN via 256-entry LUT | `fp8_kv_quantize_kernel` (ds4_cuda.cu:2234) |
| Ratio-4 state shift | `kernel_dsv4_ratio4_shift_f32` (metal/dsv4_kv.metal:184-194) | `compressor_shift_ratio4_kernel` (ds4_cuda.cu:3885) |
| Compressor store (recurrent update) | `kernel_dsv4_compressor_store_one` (metal/dsv4_kv.metal:201-227) | `compressor_store_kernel`, `compressor_prefill_pool_kernel` (ds4_cuda.cu:3724) |
| Indexer score (top-K row selection) | `kernel_dsv4_indexer_score_one_direct`, `_scores_prefill`, `_scores_decode_batch` (metal/dsv4_misc.metal:151-...) | `indexer_scores_kernel`, `indexer_score_one_direct_kernel`, `indexer_topk_kernel` (ds4_cuda.cu:4183-4564) |
| Top-K mask | `kernel_dsv4_topk_mask` | Inline in the indexer top-K kernel |
| Indexed attention (heads-of-8) | `kernel_dsv4_indexed_attention_heads8_*` | `attention_indexed_mixed_heads8_*` (ds4_cuda.cu:2899-3243) |
| Directional steering (rank-1 subtract) | `kernel_dsv4_directional_steering_project_f32` (metal/dsv4_misc.metal:98-124) | Implicit in routing-weight kernels |

For everything in this table, the semantics and the test vectors are the same. Either backend should reproduce the official DeepSeek API's greedy continuation within the project's logprob tolerance (4.0 logprob delta, see `tests/ds4_test.c:438`).

---

## 10. Memory reporting

| | Metal | CUDA |
|---|---|---|
| Tracked categories | runtime tensors (live/peak), mmap wrapper buffers (count, total GiB, max single GiB), residency request count, **scratch broken down by subsystem** (flash mask, pad, tmp, blk, ring, kv, compressor, router, indexer, moe, f16, raw-store) | Just `cudaMemGetInfo()` — free / total of device VRAM |
| Optional verbose | `DS4_METAL_TRACE_ALLOCS` traces every alloc/free | `DS4_CUDA_WEIGHT_CACHE_VERBOSE` traces the F16/F32 weight cache |
| Function | `ds4_gpu_print_memory_report(label)` (ds4_metal.m:1072-1142) | `ds4_gpu_print_memory_report(label)` (ds4_cuda.cu:1341-1346) |

Metal's detailed breakdown is useful precisely because mmap wrappers, residency hints, and per-subsystem scratch tensors all live in the same address space and can interact in surprising ways. CUDA's reporting is thinner: VRAM is one bucket; the engine isn't trying to share host pages with the GPU in the same fine-grained way, so a flat free/total is more informative than it would be on Metal.

---

## 11. Build and portability

| | Metal | CUDA |
|---|---|---|
| Compiler | `cc` for Objective-C runtime + Metal compiler (runtime) for kernels | `nvcc` for the whole `ds4_cuda.cu` |
| Target | Universal binary (arm64 + x86_64) macOS app; kernels compiled per-device at process start | Per-architecture cubin baked into the binary at build time |
| Architecture flag | None needed (Metal compiler picks at runtime) | `CUDA_ARCH=native` default (Makefile:24); override with `make CUDA_ARCH=sm_120` etc. |
| Math flags | `-O3 -ffast-math -mcpu=native` (host); Metal compiler defaults for kernels | `-O3 --use_fast_math -arch=$(CUDA_ARCH)` + `-Xcompiler ...` for host bits (Makefile:28) |
| Required libraries | `-framework Foundation -framework Metal` | `-lcudart -lcublas` |
| Portability of built binary | Runs on any macOS machine with a supported GPU | Runs only on the GPU architecture compiled for; recompile required across GPU generations |

A `make` on a fresh Mac produces a working `ds4` and `ds4-server` for that user's hardware without needing to know which Apple GPU they have. A `make` on a Linux box compiles for the local GPU; trying to deploy that binary to a different SM family will fail or fall back to JIT (which `--use_fast_math` may interact with poorly).

---

## 12. Concurrency model

Both backends are designed for the engine's single-worker pattern: one CPU thread driving GPU work for one session at a time.

| | Metal | CUDA |
|---|---|---|
| Queues / streams | One `MTLCommandQueue` | One default compute stream + two named model-staging streams |
| In-flight work | Multiple pending command buffers can complete in parallel | Default stream is sequential; model-staging streams overlap only with model loading |
| CPU/GPU overlap | Significant — submit batches, then process the next request prep while GPU drains | Limited — `cudaDeviceSynchronize` is the unit of progress for the engine |
| Compute/blit concurrency | Disallowed in this code (compute encoder must be explicitly closed before a blit) | Implicit through stream ordering |
| Concurrent sessions | Single global state on both backends; the engine serializes already |

Neither backend tries to batch requests from different sessions onto the same kernel launch. That's deliberate: the README documents that the server runs one live session and serializes concurrent clients (README.md:188). Inter-request batching would be a fundamentally different engine.

---

## 13. Environment-variable surface

Both backends are tunable at runtime through a sprawl of env vars. Highlights:

### Metal-side knobs

Memory / mmap:
- `DS4_METAL_NO_RESIDENCY` — disable the macOS 15+ residency hint
- `DS4_METAL_NO_MODEL_WARMUP` — skip the stride-touch warmup pass
- `DS4_METAL_MODEL_WARMUP_STRIDE_MB` — tune warmup stride

Pipelines / kernel sources:
- `DS4_METAL_DISABLE_HOT_PIPELINE_STATICS` — force dynamic lookup
- `DS4_METAL_FLASH_ATTN_SOURCE`, `DS4_METAL_DENSE_SOURCE`, `DS4_METAL_MOE_SOURCE`, `DS4_METAL_DSV4_HC_SOURCE`, ... — swap in alternative kernel source files

Diagnostic / tracing:
- `DS4_METAL_TRACE_ALLOCS`, `DS4_METAL_ATTN_OUT_STAGE_PROFILE`

Specialisation toggles:
- `DS4_METAL_COMPRESSOR_PAIR_NR4`, `DS4_METAL_DISABLE_COMPRESSOR_STORE_ONE`
- `DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT`, `DS4_METAL_DISABLE_ATTN_OUT_IDS_CACHE`

### CUDA-side knobs

Model loading:
- `DS4_CUDA_COPY_MODEL`, `DS4_CUDA_COPY_MODEL_CHUNKED`, `DS4_CUDA_MODEL_COPY_CHUNK_MB`
- `DS4_CUDA_DIRECT_MODEL`, `DS4_CUDA_NO_FD_CACHE`, `DS4_CUDA_NO_DIRECT_IO`

Weight caching:
- `DS4_CUDA_WEIGHT_CACHE`, `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB`, `DS4_CUDA_WEIGHT_CACHE_VERBOSE`, `DS4_CUDA_STRICT_WEIGHT_CACHE`
- `DS4_CUDA_Q8_F32_PRELOAD`, `DS4_CUDA_ATTENTION_OUTPUT_PRELOAD`

Kernel selection:
- `DS4_CUDA_NO_Q`, `DS4_CUDA_SERIAL_F`, `DS4_CUDA_SERIAL_ROUTER`
- `DS4_CUDA_NO_WINDOW_ATTENTION`, `DS4_CUDA_INDEXED_TWOPASS`, `DS4_CUDA_NO_INDEXED_HEADS`

MoE:
- `DS4_CUDA_MOE_GATE_ROW`, `DS4_CUDA_MOE_NO_GATE_ROW`
- `DS4_CUDA_MOE_ATOMIC_DOWN`, `DS4_CUDA_MOE_NO_ATOMIC_DOWN`, `DS4_CUDA_MOE_PROFILE`

Precision:
- `DS4_CUDA_NO_TF32` — disable TF32 in cuBLAS

The shape of the knobs maps onto the shape of the backend: Metal exposes residency, library overrides, and kernel-source swaps; CUDA exposes model-staging tiers, weight-cache budgets, and kernel-variant selectors.

---

## 14. Where each backend wins

### Things Metal does that CUDA cannot

1. **True zero-copy model attach** via mmap + `newBufferWithBytesNoCopy` + overlapping views. CUDA's closest equivalent (`cudaHostRegister + cudaHostGetDevicePointer`) still requires the host pages to be pinnable and is gated by host pinned-memory limits; for a discrete GPU the model has to cross the PCIe bus regardless.
2. **Hot-swap kernel sources** via `DS4_METAL_*_SOURCE` env vars without recompiling — useful for kernel-level diagnostics on a running deploy.
3. **Single-binary GPU portability** — one Metal binary runs on M1, M3, M4 without recompilation.
4. **Detailed per-subsystem scratch reporting** that's actually useful for diagnosing memory pressure on a unified memory system.
5. **Broader routed-expert quant coverage in the MoE matvec.** Metal accepts IQ2_XXS / Q2_K / Q4_K / Q8_0 across all shapes with fused variants; CUDA now covers IQ2_XXS+Q2_K plus Q4_K for the MTP decode shape (wider Q4_K via the mmq path). The MTP-relevant gap that used to no-op MTP on CUDA is closed — see §8 and [`MTP_PARITY_GAP.md`](MTP_PARITY_GAP.md).

### Things CUDA does that Metal cannot

1. **Tensor-core acceleration via cuBLAS** on F16 GEMM. The Q8→F16 pre-conversion cache exists precisely so that prefill matmuls can use Hgemm/GemmEx and get the tensor-core throughput multiplier.
2. **Lazy per-range chunked staging** with `cudaMemcpy` fallback — works correctly even when the host can't pin the whole model. Necessary on systems that don't have enough pinned-page budget for ~80 GB.
3. **Linux `O_DIRECT` direct-read fast path** via `set_model_fd`, sidestepping the page cache when streaming weights from NVMe.
4. **Tuning knobs for prefill GEMM precision** (`DS4_CUDA_NO_TF32`) that don't apply on the Metal side because there's no equivalent fast path.

### Things both do the same way

- The high-level inference graph is identical: 43 layers, HC=4 streams, ratio-4 with indexer alternating with ratio-128, raw 128-token sliding window, MLA-style attention, 256-expert MoE with top-6 routing, tail-only RoPE with YaRN scaling, FP8 E4M3FN KV round-trip.
- Routed experts stay quantised in both — IQ2_XXS / Q2_K are dequantised inline at use time on both backends (and Metal additionally handles Q4_K inline, which CUDA does not today — see §8).
- Single-worker, single-session execution. No batching across requests on either side.
- Quality-vs-speed flag (`--quality`) flips both `DS4_CUDA_NO_TF32` and Metal's quality flag; same intent, different impact.
- Both reach the same logprob test vectors within the tolerance defined in `tests/ds4_test.c`.

---

## 15. Reading the source side-by-side

If you want to ground this comparison in code, the most efficient path is:

1. **Tensor model**: open `ds4_metal.m:203-211, 3767-3845` next to `ds4_cuda.cu:27-31, 1126-1187`. Note that the Metal version is OO and the CUDA version is a plain struct; both expose the same C API.
2. **Command lifecycle**: `ds4_metal.m:3895-3941` vs `ds4_cuda.cu:1190-1193`. The contrast — pages of code on one side, four no-ops/syncs on the other — is the clearest single example of how different the two platforms feel.
3. **Model attach**: `ds4_metal.m:408-531` vs `ds4_cuda.cu:1195-1313` and `:173-278`. Read the Metal side first to understand the goal (zero-copy weight access); then read the CUDA tiers to see how that goal is approximated when the GPU can't see host pages directly.
4. **Q8 matmul**: `metal/dense.metal:109-176` vs `ds4_cuda.cu:1596-1832, 4867-4916`. The Metal kernel is the dequant-and-MAC loop in isolation. The CUDA path is dispatch logic that decides whether to use cuBLAS on cached F16 or fall through to the same dequant-and-MAC pattern.
5. **MoE expert**: `metal/moe.metal:23-265` vs `ds4_cuda.cu:6722-6776, 7137-7430`. The IQ2_XXS codebook lookup and the swiglu fusion are the same algorithm written in two dialects.
6. **Public API contract**: re-read `ds4_gpu.h` after the above. Both backends end up implementing this header; everything else is a choice.

---

## 16. Summary in one paragraph

The Metal backend leans on the fact that an Apple Silicon mmap *is* a GPU buffer: it wraps the GGUF file with overlapping `MTLBuffer` views, requests residency on macOS 15+, dequantises Q8 inline at every matvec, batches kernel dispatches inside one explicit command buffer, and parametrises kernel variants through Metal function constants — one source per file, many specialisations, swappable at runtime via env vars. The CUDA backend assumes a discrete-or-bus-attached GPU where the model has to cross some boundary: it tries `cudaHostRegister` first, falls back to per-range pinning or 64 MiB-chunked `cudaMemcpy`, and aggressively pre-converts Q8 weights to F16 in a one-time device-side dequant pass so that `cublasGemmEx` can run prefill matmuls on tensor cores. Both backends share the same DSV4-specific kernels (HC mixer, tail RoPE, FP8 KV, ratio-4 indexer, compressor, directional steering, MoE with IQ2_XXS+Q2_K routed experts) and ultimately match the official DeepSeek API's logits within the project's test tolerance — but the routes they take to get there reflect their hardware: one path optimises for unified memory and runtime flexibility; the other optimises for bus crossings and tensor-core throughput. One area where they have **not** yet converged is the set of routed-expert quants the MoE matvec accepts: Metal handles Q4_K, CUDA does not, and that single gap is what currently breaks MTP speculative decode on CUDA ([`MTP_PARITY_GAP.md`](MTP_PARITY_GAP.md)).
