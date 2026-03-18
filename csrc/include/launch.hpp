#pragma once

#if !defined(__CUDACC__)
#error "This header is CUDA-only. Include it only from a CUDA TU."
#endif

#include <c10/cuda/CUDAException.h>

#include <cute/tensor.hpp>

#include "config.hpp"
#include "params.hpp"
#include "stage1_kernel.hpp"
#include "stage2_kernel.hpp"
#include "stage3_kernel.hpp"

using namespace cute;

template <int HeadDim, int NHeads, int NHeadsKV, int PageSize>
void launch_tree_attn_stage1(const Stage1Params& params, cudaStream_t stream) {
    static constexpr int kBlockQ = 16;
    static constexpr int kBlockKV = HeadDim <= 64 ? 256 : (HeadDim <= 128 ? 128 : 64);
    static constexpr int kNWarps = 4;

    using Config = Stage1Config<HeadDim, NHeads, NHeadsKV, PageSize, kBlockQ, kBlockKV, kNWarps>;

    auto kernel = &tree_attn_stage1_kernel<Config>;

    static constexpr int kSmemSize = Config::kSmemSize;
    // https://stackoverflow.com/questions/63757245/using-maximum-shared-memory-in-cuda
    if constexpr (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    }

    dim3 grid(params.num_active_pages, NHeadsKV);
    kernel<<<grid, Config::kNThreads, kSmemSize, stream>>>(params);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int HeadDim, int NHeads, int NSplits>
void launch_tree_attn_stage2(const Stage2Params& params, cudaStream_t stream) {
    static constexpr int kBlockO = 16;
    static constexpr int kNWarps = 4;

    using Config = Stage2Config<HeadDim, NHeads, NSplits, kBlockO, kNWarps>;

    auto kernel = &tree_attn_stage2_kernel<Config>;

    dim3 grid(params.num_nodes, NHeads);
    kernel<<<grid, Config::kNThreads, 0, stream>>>(params);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int HeadDim, int NHeads>
void launch_tree_attn_stage3(const Stage3Params& params, cudaStream_t stream) {
    static constexpr int kBlockO = 16;
    static constexpr int kNWarps = 4;

    using Config = Stage3Config<HeadDim, NHeads, kBlockO, kNWarps>;

    auto kernel = &tree_attn_stage3_kernel<Config>;

    dim3 grid(params.num_edges, NHeads);
    kernel<<<grid, Config::kNThreads, 0, stream>>>(params);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
