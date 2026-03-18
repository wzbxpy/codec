#pragma once

#include <cute/tensor.hpp>

using namespace cute;

template <
    typename CTensorR,
    typename ATensorR,
    typename BTensorR,
    typename BTensorS,
    typename TiledMMA,
    typename TiledCopy,
    typename ThrCopy>
__forceinline__ __device__ void
gemm_rs(
    CTensorR& rC,
    ATensorR& rA,
    BTensorR& rB,
    BTensorS const& sB,
    TiledMMA tiled_mma,
    TiledCopy smem_tiled_copy_B,
    ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(rA) == size<1>(rC));  // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(rB) == size<2>(rC));  // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(rA) == size<2>(rB));  // MMA_K

    Tensor tDrB_copy_view = smem_thr_copy_B.retile_D(rB);
    CUTE_STATIC_ASSERT_V(size<1>(sB) == size<1>(tDrB_copy_view));  // N

    // Pipeline data move & computation
    cute::copy(smem_tiled_copy_B, sB(_, _, _0{}), tDrB_copy_view(_, _, _0{}));

#pragma unroll
    for (int i = 0; i < size<2>(rA); ++i) {
        if (i < size<2>(rA) - 1) {
            cute::copy(smem_tiled_copy_B, sB(_, _, i + 1), tDrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, rA(_, _, i), rB(_, _, i), rC);
    }
}

template <
    typename CTensorR,
    typename ATensorR, typename BTensorR,
    typename ATensorS, typename BTensorS,
    typename TiledMMA,
    typename TiledCopyA, typename TiledCopyB,
    typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void
gemm_ss(CTensorR& rC,
        ATensorR& rA,
        BTensorR& rB,
        ATensorS const& sA,
        BTensorS const& sB,
        TiledMMA tiled_mma,
        TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
        ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(rA) == size<1>(rC));  // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(rB) == size<2>(rC));  // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(rA) == size<2>(rB));  // MMA_K

    Tensor tDrA_copy_view = smem_thr_copy_A.retile_D(rA);
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(tDrA_copy_view));  // M

    Tensor tDrB_copy_view = smem_thr_copy_B.retile_D(rB);
    CUTE_STATIC_ASSERT_V(size<1>(sB) == size<1>(tDrB_copy_view));  // N

    // Pipeline data move & computation
    cute::copy(smem_tiled_copy_A, sA(_, _, _0{}), tDrA_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_B, sB(_, _, _0{}), tDrB_copy_view(_, _, _0{}));

#pragma unroll
    for (int i = 0; i < size<2>(rA); ++i) {
        if (i < size<2>(rA) - 1) {
            cute::copy(smem_tiled_copy_A, sA(_, _, i + 1), tDrA_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_B, sB(_, _, i + 1), tDrB_copy_view(_, _, i + 1));
        }
        // No need to fence here because of the register scoreboard
        cute::gemm(tiled_mma, rA(_, _, i), rB(_, _, i), rC);
    }
}

template <typename CLayout>
__forceinline__ __device__ auto
retile_A(const CLayout& layout) {
    // assert that the shape is (MMA = 4, MMA_M, MMA_N)
    static_assert(decltype(size<0>(layout))::value == 4);
    static_assert(decltype(rank(layout))::value == 3);

    // (4, MMA_M, (2, MMA_N / 2))
    auto divided = logical_divide(layout, Shape<X, X, _2>{});
    // ((4, 2), MMA_M, MMA_N / 2)
    return make_layout(make_layout(get<0>(divided), get<2, 0>(divided)), get<1>(divided), get<2, 1>(divided));
}
