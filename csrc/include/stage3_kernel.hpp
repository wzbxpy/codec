#pragma once

#include <cute/tensor.hpp>

#include "dispatch.hpp"
#include "params.hpp"

using namespace cute;

template <typename Config, bool OAligned>
__forceinline__ __device__ void
combine_kv_node(const Stage3Params& params, int parent_token_ofs, int child_token_ofs, int token_length) {
    static constexpr int kHeadDim = Config::kHeadDim;
    static constexpr int kNHeads = Config::kNHeads;
    static constexpr int kBlockO = Config::kBlockO;
    static constexpr int kNThreads = Config::kNThreads;
    static constexpr bool kOAligned = OAligned;

    const int tid = threadIdx.x;
    const int bid_head = blockIdx.y;  // The Q head index

    __shared__ half smem[2][kBlockO];

    // (kBlockO,)
    Tensor gLSE_parent = make_tensor(make_gmem_ptr(reinterpret_cast<half*>(params.lses_ptr) + parent_token_ofs * kNHeads),
                                     make_shape(Int<kBlockO>{}, Int<kNHeads>{}),
                                     GenRowMajor{})(_, bid_head);
    // (kBlockO,)
    Tensor gLSE_child = make_tensor(make_gmem_ptr(reinterpret_cast<half*>(params.lses_ptr) + child_token_ofs * kNHeads),
                                    make_shape(Int<kBlockO>{}, Int<kNHeads>{}),
                                    GenRowMajor{})(_, bid_head);

    // (kNSplits, kBlockO)
    Tensor sLSE = make_tensor(make_smem_ptr(reinterpret_cast<half*>(smem)), typename Config::SmemLayoutLSE{});

    static constexpr int kNLSEPerThread = cute::ceil_div(2 * kBlockO, kNThreads);
    static constexpr int kRowsPerLoadGmem = kNThreads / kBlockO;
#pragma unroll
    for (int i = 0; i < kNLSEPerThread; ++i) {
        const int row = i * kRowsPerLoadGmem + tid / kBlockO;
        const int col = tid % kBlockO;
        if (row < 2) {
            half lse = col < token_length
                           ? row == 0
                                 ? gLSE_parent(col)
                                 : gLSE_child(col)
                           : -CUDART_INF_FP16;
            sLSE(row, col) = lse;
        }
    }
    __syncthreads();

    static constexpr int kNRowsPerLoadSmem = cute::min(kNThreads / kBlockO, 2);
    static constexpr int kNThreadsPerO = 2 / kNLSEPerThread;
    static_assert(2 % kNLSEPerThread == 0 && kNThreadsPerO <= 32);
    Tensor rLSEaccum = make_tensor<half>(Shape<Int<kNLSEPerThread>>{});

#pragma unroll
    for (int i = 0; i < kNLSEPerThread; ++i) {
        const int row = i * kNRowsPerLoadSmem + tid % kNRowsPerLoadSmem;
        const int col = tid / kNRowsPerLoadSmem;
        rLSEaccum(i) = row < 2 && col < kBlockO ? sLSE(row, col) : -CUDART_INF_FP16;
    }

    // Compute lse_max
    half lse_max = rLSEaccum(0);
#pragma unroll
    for (int i = 1; i < kNLSEPerThread; ++i) {
        lse_max = __hmax(lse_max, rLSEaccum(i));
    }
#pragma unroll
    for (int offset = kNThreadsPerO / 2; offset >= 1; offset >>= 1) {
        lse_max = __hmax(lse_max, __shfl_xor_sync(uint32_t(-1), lse_max, offset));
    }

    // Compute safe lse_sum
    half lse_sum = hexp(rLSEaccum(0) - lse_max);
#pragma unroll
    for (int i = 1; i < kNLSEPerThread; ++i) {
        lse_sum += hexp(rLSEaccum(i) - lse_max);
    }
#pragma unroll
    for (int offset = kNThreadsPerO / 2; offset >= 1; offset >>= 1) {
        lse_sum += __shfl_xor_sync(uint32_t(-1), lse_sum, offset);
    }

    // Compute reduced lse
    half lse_logsum = lse_sum == CUDART_ZERO_FP16 || __hisnan(lse_sum) ? CUDART_INF_FP16 : hlog(lse_sum) + lse_max;

    // Write back LSE
    if (tid % kNThreadsPerO == 0 && tid / kNRowsPerLoadSmem < kBlockO) {
        gLSE_child(tid / kNRowsPerLoadSmem) = lse_logsum;
    }

    // Compute the scales, exp(lse - lse_logsum) in smem
#pragma unroll
    for (int i = 0; i < kNLSEPerThread; ++i) {
        const int row = i * kNRowsPerLoadSmem + tid % kNRowsPerLoadSmem;
        const int col = tid / kNRowsPerLoadSmem;
        if (row < 2 && col < kBlockO) {
            sLSE(row, col) = hexp(rLSEaccum(i) - lse_logsum);
        }
    }
    __syncthreads();

    // Apply scale to O and accumulate
    // (kBlockO, head_dim)
    Tensor gO_parent = make_tensor(make_gmem_ptr(reinterpret_cast<half*>(params.outs_ptr) + parent_token_ofs * kNHeads * kHeadDim),
                                   make_shape(Int<kBlockO>{}, Int<kNHeads>{}, Int<kHeadDim>{}),
                                   GenRowMajor{})(_, bid_head, _);
    // (kBlockO, head_dim)
    Tensor gO_child = make_tensor(make_gmem_ptr(reinterpret_cast<half*>(params.outs_ptr) + child_token_ofs * kNHeads * kHeadDim),
                                  make_shape(Int<kBlockO>{}, Int<kNHeads>{}, Int<kHeadDim>{}),
                                  GenRowMajor{})(_, bid_head, _);

    // Compute reduced O
    typename Config::GmemTiledCopyO gmem_tiled_copy_O;
    ThrCopy gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tid);
    Tensor tSgO_parent = gmem_thr_copy_O.partition_S(gO_parent);
    Tensor tSgO_child = gmem_thr_copy_O.partition_S(gO_child);
    Tensor tDrO = make_tensor<half>(shape(tSgO_child));
    Tensor tDrOaccum = make_tensor<half>(shape(tDrO));
    clear(tDrOaccum);

    Tensor cO = make_identity_tensor(Shape<Int<kBlockO>, Int<kHeadDim>>{});
    Tensor tScO = gmem_thr_copy_O.partition_S(cO);

    ::copy<kOAligned, true, false, false>(gmem_tiled_copy_O,
                                          tSgO_parent,
                                          tDrO,
                                          tScO,
                                          token_length, 0);
#pragma unroll
    for (int m = 0; m < size<1>(tDrO); ++m) {
        int row = get<0>(tScO(0, m, 0));
        half lse_scale = sLSE(0, row);
#pragma unroll
        for (int k = 0; k < size<2>(tDrO); ++k) {
#pragma unroll
            for (int i = 0; i < size<0>(tDrO); ++i) {
                tDrOaccum(i, m, k) += lse_scale * tDrO(i, m, k);
            }
        }
    }

    ::copy<kOAligned, true, false, false>(gmem_tiled_copy_O,
                                          tSgO_child,
                                          tDrO,
                                          tScO,
                                          token_length, 0);
#pragma unroll
    for (int m = 0; m < size<1>(tDrO); ++m) {
        int row = get<0>(tScO(0, m, 0));
        half lse_scale = sLSE(1, row);
#pragma unroll
        for (int k = 0; k < size<2>(tDrO); ++k) {
#pragma unroll
            for (int i = 0; i < size<0>(tDrO); ++i) {
                tDrOaccum(i, m, k) += lse_scale * tDrO(i, m, k);
            }
        }
    }

    // Write back O
    typename Config::GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    ThrCopy gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tid);
    Tensor tSrOaccum = gmem_thr_copy_Oaccum.retile_S(tDrOaccum);
    Tensor tDgOaccum = gmem_thr_copy_Oaccum.partition_D(gO_child);

    Tensor cOaccum = make_identity_tensor(Shape<Int<kBlockO>, Int<kHeadDim>>{});
    Tensor tScOaccum = gmem_thr_copy_O.partition_S(cO);

    ::copy<kOAligned, true, false, false>(gmem_tiled_copy_Oaccum,
                                          tSrOaccum,
                                          tDgOaccum,
                                          tScOaccum,
                                          token_length, 0);
}

template <typename Config>
__global__ void
__launch_bounds__(Config::kNThreads) tree_attn_stage3_kernel(__grid_constant__ const Stage3Params params) {
    const int bid_kv = blockIdx.x;  // The KV node index

    // KV page metadata (gridDim.x,)
    const auto [parent_token_ofs, child_token_ofs, token_length] = params.kv_edge_metadata_ptr[bid_kv];
    bool_switch(
        token_length == Config::kBlockO,
        [&](const auto expr) {
            static constexpr bool kOAligned = decltype(expr)::value;
            combine_kv_node<Config, kOAligned>(params, parent_token_ofs, child_token_ofs, token_length);
        });
}
