// bench/bw_bench.cu — kernel-side memory bandwidth probe for GB10.
//
// Builds a single-file CUDA copy/read benchmark to measure achievable
// memory bandwidth as seen by a kernel — i.e. the ceiling that matters
// for LLM decode roofline analysis, not the CE (copy-engine) bandwidth
// that nvbandwidth's H2D/D2H tests report.
//
// Build:
//   /usr/local/cuda/bin/nvcc -O3 -arch=sm_121 bench/bw_bench.cu -o /tmp/bw_bench
// Run:
//   /tmp/bw_bench [size_MiB=8192]
//
// Typical Spark GB10 numbers we observe:
//   copy_kernel  bytes=8589934592 ms=399  effective=215 GB/s  (R+W, 8 GiB)
//   read_kernel  bytes=8589934592 ms=189  effective=227 GB/s  (R only)
//
// vs published GB10 LPDDR5X peak ~273 GB/s — so the kernel-accessible
// ceiling is ~80% of theoretical, which is normal for real workloads.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

__global__ void copy_kernel(const float4 *__restrict__ src,
                            float4 *__restrict__ dst,
                            size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] = src[i];
}

__global__ void read_kernel(const float4 *__restrict__ src,
                            float *sink,
                            size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n; i += stride) {
        float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    // Force the read to materialize; sink bucket index varies per block so
    // the atomic doesn't serialize all threads.
    atomicAdd(sink + (blockIdx.x & 1023), acc * 1e-30f);
}

int main(int argc, char **argv) {
    size_t mib = 8192;
    if (argc > 1) mib = (size_t)atoll(argv[1]);
    size_t bytes = mib * 1024ULL * 1024ULL;
    size_t n = bytes / sizeof(float4);

    float4 *d_src=nullptr, *d_dst=nullptr;
    float *d_sink=nullptr;
    cudaMalloc(&d_src,  bytes);
    cudaMalloc(&d_dst,  bytes);
    cudaMalloc(&d_sink, 1024*sizeof(float));
    cudaMemset(d_src,  0x11, bytes);
    cudaMemset(d_dst,  0,    bytes);
    cudaMemset(d_sink, 0,    1024*sizeof(float));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    int block = 256, grid = 65535;
    const int iters = 5;

    // Copy: reads + writes = 2 * bytes per iteration.
    copy_kernel<<<grid,block>>>(d_src, d_dst, n);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++)
        copy_kernel<<<grid,block>>>(d_src, d_dst, n);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    double gb_copy = (double)bytes * 2.0 * iters / 1e9;
    printf("copy_kernel  bytes=%zu  iters=%d  ms=%.2f  effective=%.2f GB/s  (R+W)\n",
           bytes, iters, ms, gb_copy / (ms/1000.0));

    // Read-only.
    read_kernel<<<grid,block>>>(d_src, d_sink, n);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++)
        read_kernel<<<grid,block>>>(d_src, d_sink, n);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    double gb_read = (double)bytes * 1.0 * iters / 1e9;
    printf("read_kernel  bytes=%zu  iters=%d  ms=%.2f  effective=%.2f GB/s  (R only)\n",
           bytes, iters, ms, gb_read / (ms/1000.0));

    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_sink);
    return 0;
}
