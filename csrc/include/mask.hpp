#pragma once

#include <cute/tensor.hpp>

#include "softmax.hpp"

using namespace cute;

template <
    typename Engine,
    typename Layout>
__forceinline__ __device__ void
apply_mask(
    Tensor<Engine, Layout>& rS,
    const int max_k,
    int col_idx_offset) {
    static_assert(Layout::rank == 3);
    static_assert(decltype(size<0>(rS))::value == 4);

    Tensor tensor = make_tensor(rS.data(), Softmax<0>::retile_mma(rS.layout()));
    const int lane_id = threadIdx.x % 32;
    col_idx_offset += (lane_id % 4) * 2;
#pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * 8;
#pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j;
#pragma unroll
            for (int mi = 0; mi < size<0>(tensor); ++mi) {
                if (col_idx >= max_k) {
                    tensor(mi, make_coord(j, nj)) = -CUDART_INF_FP16;
                }
            }
        }
    }
}
