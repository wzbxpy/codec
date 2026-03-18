#pragma once

#include <cute/tensor.hpp>

#include "copy.hpp"
#include "dispatch.hpp"
#include "gemm.hpp"
#include "mask.hpp"
#include "params.hpp"
#include "softmax.hpp"

using namespace cute;

template <typename Config, bool QAligned, bool KVAligned>
__forceinline__ __device__ void
compute_kv_page(const Stage1Params& params, int q_token_ofs, int q_token_length, int kv_page_idx, int kv_page_length) {
    static constexpr int kNHeads = Config::kNHeads;
    static constexpr int kNHeadsKV = Config::kNHeadsKV;
    static constexpr int kNHeadsPerKV = Config::kNHeadsPerKV;
    static constexpr int kPageSize = Config::kPageSize;
    static constexpr int kBlockQ = Config::kBlockQ;
    static constexpr int kBlockKV = Config::kBlockKV;
    static constexpr int kHeadDim = Config::kHeadDim;
    static constexpr int kQAligned = QAligned;
    static constexpr int kKVAligned = KVAligned;

    const int tid = threadIdx.x;
    const int bid_page = blockIdx.x;  // The KV page index
    const int bid_head = blockIdx.y;  // The KV head index

    extern __shared__ char smem[];

    // (batch_size, num_heads_kv, num_heads_per_kv, head_dim)
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<half*>(params.q_ptr) + q_token_ofs * kNHeads * kHeadDim),
                            make_shape(q_token_length, Int<kNHeadsKV>{}, Int<kNHeadsPerKV>{}, Int<kHeadDim>{}),
                            GenRowMajor{});
    // (num_pages, page_size, num_heads_kv, head_dim)
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<half*>(params.k_ptr)),
                            make_shape(params.num_pages, Int<kPageSize>{}, Int<kNHeadsKV>{}, Int<kHeadDim>{}),
                            GenRowMajor{});
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<half*>(params.v_ptr)),
                            make_shape(params.num_pages, Int<kPageSize>{}, Int<kNHeadsKV>{}, Int<kHeadDim>{}),
                            GenRowMajor{});

    // (kBlockQ, num_heads_per_kv, head_dim)
    Tensor gQ_sep_view = local_tile(mQ(_, bid_head, _, _), Shape<Int<kBlockQ>, Int<kNHeadsPerKV>, Int<kHeadDim>>{}, make_coord(0, 0, 0));
    Tensor gQ = group_modes<0, 2>(make_tensor(gQ_sep_view.data(), make_layout(layout<1>(gQ_sep_view), layout<0>(gQ_sep_view), layout<2>(gQ_sep_view))));
    // (kBlockKV, head_dim, n_kv_block)
    Tensor gK = local_tile(mK(kv_page_idx, _, bid_head, _), Shape<Int<kBlockKV>, Int<kHeadDim>>{}, make_coord(_, 0));
    Tensor gV = local_tile(mV(kv_page_idx, _, bid_head, _), Shape<Int<kBlockKV>, Int<kHeadDim>>{}, make_coord(_, 0));

    // (block_Q * num_heads_per_kv, head_dim)
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<half*>(smem)), typename Config::SmemLayoutQ{});
    // (block_KV, head_dim)
    Tensor sK = make_tensor(make_smem_ptr(sQ.data() + size(sQ)), typename Config::SmemLayoutKV{});
    Tensor sV = make_tensor(make_smem_ptr(sK.data() + size(sK)), typename Config::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Config::SmemLayoutVtransposed{});

    typename Config::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    ThrCopy gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tid);

    Tensor tSgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tDsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tSgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tDsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tSgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tDsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Config::TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_thread_slice(tid);
    Tensor tArQ = thr_mma.partition_fragment_A(sQ);
    Tensor tBrK = thr_mma.partition_fragment_B(sK);
    Tensor tBrVt = thr_mma.partition_fragment_B(sVt);

    Tensor tCrO = make_tensor<half>(partition_shape_C(tiled_mma, Shape<Shape<Int<kNHeadsPerKV>, Int<kBlockQ>>, Int<kHeadDim>>{}));

    TiledCopy smem_tiled_copy_Q = make_tiled_copy_A(typename Config::SmemCopyAtomQK{}, tiled_mma);
    ThrCopy smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    TiledCopy smem_tiled_copy_K = make_tiled_copy_B(typename Config::SmemCopyAtomQK{}, tiled_mma);
    ThrCopy smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tid);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    TiledCopy smem_tiled_copy_V = make_tiled_copy_B(typename Config::SmemCopyAtomVtransposed{}, tiled_mma);
    ThrCopy smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tid);
    Tensor tSsVt = smem_thr_copy_V.partition_S(sVt);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));

    Tensor tScQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tScKV = gmem_thr_copy_QKV.partition_S(cKV);

    int kv_block_idx = cute::ceil_div(kv_page_length, kBlockKV) - 1;

    ::copy<kQAligned, true, false, false>(gmem_tiled_copy_QKV,
                                          tSgQ,
                                          tDsQ,
                                          tScQ,
                                          q_token_length * kNHeadsPerKV, 0);
    ::copy<kKVAligned, true, false, false>(gmem_tiled_copy_QKV,
                                           tSgK(_, _, _, kv_block_idx),
                                           tDsK,
                                           tScKV,
                                           kv_page_length, 0);
    cp_async_fence();
    clear(tCrO);

    Softmax<2 * size<1>(tCrO)> softmax{params.softmax_scale, params.softmax_scale_log2};

    // First block
    {
        Tensor tCrS = make_tensor<half>(partition_shape_C(tiled_mma, Shape<Shape<Int<kNHeadsPerKV>, Int<kBlockQ>>, Int<kBlockKV>>{}));
        clear(tCrS);

        ::cp_async_wait<0>();
        __syncthreads();

        ::copy<kKVAligned, true, true, false>(gmem_tiled_copy_QKV,
                                              tSgV(_, _, _, kv_block_idx),
                                              tDsV,
                                              tScKV,
                                              kv_page_length, 0);
        cp_async_fence();

        gemm_ss(tCrS,
                tArQ, tBrK,
                tSsQ, tSsK,
                tiled_mma,
                smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K);

        if (!kKVAligned) {
            apply_mask(tCrS, kv_page_length, kv_block_idx * kBlockKV);
        }

        softmax.template rescale_o<true>(tCrS, tCrO);
        Tensor tArP = make_tensor(tCrS.data(), retile_A(tCrS.layout()));

        ::cp_async_wait<0>();
        __syncthreads();

        if (kv_block_idx > 0) {
            ::copy<true, true, false, false>(gmem_tiled_copy_QKV,
                                             tSgK(_, _, _, kv_block_idx - 1),
                                             tDsK,
                                             tScKV,
                                             0, 0);
            cp_async_fence();
        }

        gemm_rs(tCrO,
                tArP, tBrVt,
                tSsVt,
                tiled_mma,
                smem_tiled_copy_V, smem_thr_copy_V);

        --kv_block_idx;
    }

    for (; kv_block_idx >= 0; --kv_block_idx) {
        Tensor tCrS = make_tensor<half>(partition_shape_C(tiled_mma, Shape<Shape<Int<kNHeadsPerKV>, Int<kBlockQ>>, Int<kBlockKV>>{}));
        clear(tCrS);

        ::cp_async_wait<0>();
        __syncthreads();

        ::copy<true, true, false, false>(gmem_tiled_copy_QKV,
                                         tSgV(_, _, _, kv_block_idx),
                                         tDsV,
                                         tScKV,
                                         0, 0);
        cp_async_fence();

        gemm_ss(tCrS,
                tArQ, tBrK,
                tSsQ, tSsK,
                tiled_mma,
                smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K);

        softmax.template rescale_o<false>(tCrS, tCrO);
        Tensor tArP = make_tensor(tCrS.data(), retile_A(tCrS.layout()));

        ::cp_async_wait<0>();
        __syncthreads();

        if (kv_block_idx > 0) {
            ::copy<true, true, false, false>(gmem_tiled_copy_QKV,
                                             tSgK(_, _, _, kv_block_idx - 1),
                                             tDsK,
                                             tScKV,
                                             0, 0);
            cp_async_fence();
        }

        gemm_rs(tCrO,
                tArP, tBrVt,
                tSsVt,
                tiled_mma,
                smem_tiled_copy_V, smem_thr_copy_V);
    }

    Tensor lse = softmax.normalize(tCrO);

    // (kBlockQ * num_heads_per_kv, head_dim)
    Tensor sO = make_tensor(sQ.data(), typename Config::SmemLayoutO{});
    TiledCopy smem_tiled_copy_O = make_tiled_copy_C(typename Config::SmemCopyAtomO{}, tiled_mma);
    ThrCopy smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tid);
    Tensor tSrO = smem_thr_copy_O.retile_S(tCrO);
    Tensor tDsO = smem_thr_copy_O.partition_D(sO);
    copy(smem_tiled_copy_O, tSrO, tDsO);

    // (kBlockQ, num_heads_kv, num_heads_per_kv, head_dim)
    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<half*>(params.outs_ptr) + bid_page * kBlockQ * kNHeads * kHeadDim),
                            make_shape(q_token_length, Int<kNHeadsKV>{}, Int<kNHeadsPerKV>{}, Int<kHeadDim>{}),
                            GenRowMajor{});
    // (kBlockQ, num_heads_kv, num_heads_per_kv)
    Tensor mLSE = make_tensor(make_gmem_ptr(reinterpret_cast<half*>(params.lses_ptr) + bid_page * kBlockQ * kNHeads),
                              make_shape(q_token_length, Int<kNHeadsKV>{}, Int<kNHeadsPerKV>{}),
                              GenRowMajor{});

    // (kBlockQ * num_heads_per_kv, head_dim)
    Tensor gO_sep_view = local_tile(mO(_, bid_head, _, _), Shape<Int<kBlockQ>, Int<kNHeadsPerKV>, Int<kHeadDim>>{}, make_coord(0, 0, 0));
    Tensor gO = group_modes<0, 2>(make_tensor(gO_sep_view.data(), make_layout(layout<1>(gO_sep_view), layout<0>(gO_sep_view), layout<2>(gO_sep_view))));
    // (kBlockQ * num_heads_per_kv)
    Tensor gLSE = group_modes<0, 2>(mLSE(_, bid_head, _));

    typename Config::GmemTiledCopyO gmem_tiled_copy_O;
    ThrCopy gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tid);
    Tensor tSsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tDgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor rO = make_tensor<half>(shape(tDgO));
    copy(gmem_tiled_copy_O, tSsO, rO);

    Tensor cO = make_identity_tensor(Shape<Int<kNHeadsPerKV * kBlockQ>, Int<kHeadDim>>{});
    Tensor tLSEcO = thr_mma.partition_C(cO);
    static_assert(decltype(size<0>(tLSEcO))::value == 4);

    // Write back LSE
    // ((2, 2), MMA_M, MMA_N) -> (2, MMA_M)
    Tensor tLSEcO_row_view = logical_divide(tLSEcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(tLSEcO_row_view));
    if (get<1>(tLSEcO_row_view(0)) == 0) {
#pragma unroll
        for (int m = 0; m < size(lse); ++m) {
            const int row = get<0>(tLSEcO_row_view(m));
            if (row < q_token_length * kNHeadsPerKV) {
                gLSE(row) = lse(m);
            }
        }
    }

    // Write back O
    Tensor tDcO = gmem_thr_copy_O.partition_D(cO);
    ::copy<kQAligned, true, false, false>(gmem_tiled_copy_O,
                                          rO,
                                          tDgO,
                                          tDcO,
                                          q_token_length * kNHeadsPerKV, 0);
}

template <typename Config>
__global__ void
__launch_bounds__(Config::kNThreads) tree_attn_stage1_kernel(__grid_constant__ const Stage1Params params) {
    const int bid_kv = blockIdx.x;  // The KV page index

    // KV page metadata (gridDim.x,)
    const auto [q_token_ofs, q_token_length, kv_page_idx, kv_page_length] = params.kv_page_metadata_ptr[bid_kv];

    bool_switch(q_token_length % Config::kBlockQ == 0, [&](auto expr) {
        static constexpr int kQAligned = decltype(expr)::value;
        bool_switch(kv_page_length % Config::kBlockKV == 0, [&](auto expr) {
            static constexpr int kKVAligned = decltype(expr)::value;
            compute_kv_page<Config, kQAligned, kKVAligned>(params, q_token_ofs, q_token_length, kv_page_idx, kv_page_length);
        });
    });
}
