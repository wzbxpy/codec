#pragma once

#include <cute/tensor.hpp>

using namespace cute;

template <
    bool MNAligned,
    bool KAligned,
    bool ZeroMNOOB,
    bool InfKOOB,
    typename TiledCopy,
    typename SrcEngine, typename SrcLayout,
    typename DstEngine, typename DstLayout,
    typename CoordEngine, typename CoordLayout>
__forceinline__ __device__ void
copy(
    TiledCopy tiled_copy,
    Tensor<SrcEngine, SrcLayout> const& src,
    Tensor<DstEngine, DstLayout>& dst,
    Tensor<CoordEngine, CoordLayout> const& coord,
    const int max_MN,
    const int max_K) {
    CUTE_STATIC_ASSERT_V(rank(src) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(dst) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(src) == size<0>(dst));  // Atom
    CUTE_STATIC_ASSERT_V(size<1>(src) == size<1>(dst));  // MN
    CUTE_STATIC_ASSERT_V(size<2>(src) == size<2>(dst));  // K

#pragma unroll
    for (int mn = 0; mn < size<1>(src); ++mn) {
        if (MNAligned || get<0>(coord(0, mn, 0)) < max_MN) {
#pragma unroll
            for (int k = 0; k < size<2>(src); ++k) {
                if (KAligned || get<1>(coord(0, 0, k)) < max_K) {
                    copy(tiled_copy, src(_, mn, k), dst(_, mn, k));
                } else if constexpr (InfKOOB) {
                    fill(dst(_, mn, k), -CUDART_INF_FP16);
                }
            }
        } else if constexpr (ZeroMNOOB) {
            clear(dst(_, mn, _));
        }
    }
}

template <int N>
CUTE_HOST_DEVICE void
cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
