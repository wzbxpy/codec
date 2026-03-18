#pragma once

#include <cute/tensor.hpp>

#include "copy.hpp"
#include "params.hpp"

using namespace cute;

template <typename Config, bool OAligned>
__forceinline__ __device__ void
combine_kv_page(const Stage2Params& params, int o_token_ofs, int o_token_length, int num_splits) {
    static constexpr int kHeadDim = Config::kHeadDim;
    static constexpr int kNSplits = Config::kNSplits;
    static constexpr int kNHeads = Config::kNHeads;
    static constexpr int kBlockO = Config::kBlockO;
    static constexpr int kNThreads = Config::kNThreads;
    static constexpr int kOAligned = OAligned;
    static_assert((kNSplits & (kNSplits - 1)) == 0);

    const int tid = threadIdx.x;
    const int bid_head = blockIdx.y;  // The Q head index

    __shared__ half smem[kNSplits][kBlockO];

    // (num_splits, kBlockO, num_heads_qo)
    Tensor mLSE = make_tensor(make_gmem_ptr(reinterpret_cast<half*>(params.lses_ptr) + o_token_ofs * kNHeads),
                              make_shape(num_splits, Int<kBlockO>{}, Int<kNHeads>{}),
                              GenRowMajor{});
    // (kNSplits, kBlockO)
    Tensor gLSE = local_tile(mLSE(_, _, bid_head), Shape<Int<kNSplits>, Int<kBlockO>>{}, make_coord(0, 0));
    // (kBlockO,)
    Tensor gLSEaccum = gLSE(0, _);

    // (kBlockO, kNSplits)
    Tensor sLSE = make_tensor(make_smem_ptr(reinterpret_cast<half*>(smem)), typename Config::SmemLayoutLSE{});

    static constexpr int kNLSEPerThread = cute::ceil_div(kNSplits * kBlockO, kNThreads);
    static constexpr int kRowsPerLoadGmem = kNThreads / kBlockO;
#pragma unroll
    for (int i = 0; i < kNLSEPerThread; ++i) {
        const int row = i * kRowsPerLoadGmem + tid / kBlockO;
        const int col = tid % kBlockO;
        half lse = (row < num_splits && col < o_token_length) ? gLSE(row, col) : -CUDART_INF_FP16;
        if (row < kNSplits) {
            sLSE(row, col) = lse;
        }
    }
    __syncthreads();

    static constexpr int kNRowsPerLoadSmem = cute::min(kNThreads / kBlockO, kNSplits);
    static constexpr int kNThreadsPerO = kNSplits / kNLSEPerThread;
    static_assert(kNSplits % kNLSEPerThread == 0 && kNThreadsPerO <= 32);
    Tensor rLSEaccum = make_tensor<half>(Shape<Int<kNLSEPerThread>>{});

#pragma unroll
    for (int i = 0; i < kNLSEPerThread; ++i) {
        const int row = i * kNRowsPerLoadSmem + tid % kNRowsPerLoadSmem;
        const int col = tid / kNRowsPerLoadSmem;
        rLSEaccum(i) = row < kNSplits && col < kBlockO ? sLSE(row, col) : -CUDART_INF_FP16;
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
        gLSEaccum(tid / kNRowsPerLoadSmem) = lse_logsum;
    }

    // Compute the scales, exp(lse - lse_logsum) in smem
#pragma unroll
    for (int i = 0; i < kNLSEPerThread; ++i) {
        const int row = i * kNRowsPerLoadSmem + tid % kNRowsPerLoadSmem;
        const int col = tid / kNRowsPerLoadSmem;
        if (row < kNSplits && col < kBlockO) {
            sLSE(row, col) = hexp(rLSEaccum(i) - lse_logsum);
        }
    }
    __syncthreads();

    // Apply scale to O and accumulate
    // (num_splits, kBlockO, num_heads_qo, head_dim)
    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<half*>(params.outs_ptr) + o_token_ofs * kNHeads * kHeadDim),
                            make_shape(num_splits, Int<kBlockO>{}, Int<kNHeads>{}, Int<kHeadDim>{}),
                            GenRowMajor{});
    // (kNSplits, kBlockO, head_dim)
    Tensor gO = local_tile(mO(_, _, bid_head, _), Shape<Int<kNSplits>, Int<kBlockO>, Int<kHeadDim>>{}, make_coord(0, 0, 0));
    // (kBlockO, head_dim, kNSplits)
    Tensor gO_split_view = make_tensor(gO.data(), make_layout(layout<1>(gO), layout<2>(gO), layout<0>(gO)));

    // Compute reduced O
    typename Config::GmemTiledCopyO gmem_tiled_copy_O;
    ThrCopy gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tid);
    Tensor tSgO = gmem_thr_copy_O.partition_S(gO_split_view);
    Tensor tDrO = make_tensor<half>(shape(tSgO(_, _, _, 0)));
    Tensor tDrOaccum = make_tensor<half>(shape(tDrO));
    clear(tDrOaccum);

    Tensor cO = make_identity_tensor(Shape<Int<kBlockO>, Int<kHeadDim>>{});
    Tensor tScO = gmem_thr_copy_O.partition_S(cO);

    for (int split = 0; split < num_splits; ++split) {
        ::copy<kOAligned, true, false, false>(gmem_tiled_copy_O,
                                              tSgO(_, _, _, split),
                                              tDrO,
                                              tScO,
                                              o_token_length, 0);
#pragma unroll
        for (int m = 0; m < size<1>(tDrO); ++m) {
            int row = get<0>(tScO(0, m, 0));
            half lse_scale = sLSE(split, row);
#pragma unroll
            for (int k = 0; k < size<2>(tDrO); ++k) {
#pragma unroll
                for (int i = 0; i < size<0>(tDrO); ++i) {
                    tDrOaccum(i, m, k) += lse_scale * tDrO(i, m, k);
                }
            }
        }
    }

    // (kBlockO, head_dim)
    Tensor gOaccum = gO(0, _, _);

    // Write back O
    typename Config::GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    ThrCopy gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tid);
    Tensor tSrOaccum = gmem_thr_copy_Oaccum.retile_S(tDrOaccum);
    Tensor tDgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    Tensor cOaccum = make_identity_tensor(Shape<Int<kBlockO>, Int<kHeadDim>>{});
    Tensor tScOaccum = gmem_thr_copy_O.partition_S(cO);

    ::copy<kOAligned, true, false, false>(gmem_tiled_copy_Oaccum,
                                          tSrOaccum,
                                          tDgOaccum,
                                          tScOaccum,
                                          o_token_length, 0);
}

template <typename Config>
__global__ void
__launch_bounds__(Config::kNThreads) tree_attn_stage2_kernel(__grid_constant__ const Stage2Params params) {
    const int bid_kv = blockIdx.x;  // The KV node index

    // KV page metadata (gridDim.x,)
    const auto [o_token_ofs, o_token_length, num_splits] = params.kv_node_metadata_ptr[bid_kv];

    bool_switch(
        o_token_length % Config::kBlockO == 0,
        [&](auto expr) {
            static constexpr int kOAligned = decltype(expr)::value;
            combine_kv_page<Config, kOAligned>(params, o_token_ofs, o_token_length, num_splits);
        });
}
