#pragma once

#include <cute/tensor.hpp>

using namespace cute;

template <int NRows>
struct Softmax {
    decltype(make_tensor<half>(Shape<Int<NRows>>{})) row_max, row_sum;
    half softmax_scale, softmax_scale_log2;

    Softmax() = delete;

    __forceinline__ __device__ Softmax(const half scale, const half scale_log2) : softmax_scale(scale), softmax_scale_log2(scale_log2) {}

    template <typename Layout>
    static __forceinline__ __device__ auto
    retile_mma(Layout layout) {
        // assert that the shape is (MMA = 4, MMA_M, MMA_N)
        static_assert(decltype(size<0>(layout))::value == 4);
        static_assert(decltype(rank(layout))::value == 3);

        // ((2, 2), MMA_M, MMA_N)
        auto divided = logical_divide(layout, Shape<_2>{});
        // ((2, MMA_M), (2, MMA_N))
        return make_layout(make_layout(get<0, 1>(divided), get<1>(divided)), make_layout(get<0, 0>(divided), get<2>(divided)));
    }

    template <bool ZeroInit, typename STensor>
    __forceinline__ __device__ void
    reduce_row_max(STensor const& scores) {
        static_assert(decltype(rank(scores))::value == 2);
        static_assert(decltype(rank(row_max))::value == 1);
        CUTE_STATIC_ASSERT_V(size<0>(scores) == size<0>(row_max));

        // Thread level reduction
#pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            row_max(mi) = ZeroInit ? scores(mi, 0) : __hmax(row_max(mi), scores(mi, 0));
#pragma unroll
            for (int ni = 1; ni < size<1>(scores); ++ni) {
                row_max(mi) = __hmax(row_max(mi), scores(mi, ni));
            }
        }

        // Warp level reduction
#pragma unroll
        for (int i = 0; i < size<0>(row_max); ++i) {
            // Every consecutive 4 threads can get their own row_max(i)
#pragma unroll
            for (int offset = 2; offset >= 1; offset >>= 1) {
                row_max(i) = __hmax(row_max(i), __shfl_xor_sync(uint32_t(-1), row_max(i), offset));
            }
        }
    }

    template <typename STensor>
    __forceinline__ __device__ void
    safe_exp(STensor& scores) {
        static_assert(decltype(rank(scores))::value == 2);
        static_assert(decltype(rank(row_max))::value == 1);
        CUTE_STATIC_ASSERT_V(size<0>(scores) == size<0>(row_max));

#pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            const half scaled_max = __hisinf(row_max(mi)) == -1 ? CUDART_ZERO_FP16 : row_max(mi) * softmax_scale_log2;
#pragma unroll
            for (int ni = 0; ni < size<1>(scores); ++ni) {
                scores(mi, ni) = hexp2(scores(mi, ni) * softmax_scale_log2 - scaled_max);
            }
        }
    }

    template <bool ZeroInit, typename STensor>
    __forceinline__ __device__ void
    reduce_row_sum(STensor const& scores) {
        static_assert(decltype(rank(scores))::value == 2);
        static_assert(decltype(rank(row_sum))::value == 1);
        CUTE_STATIC_ASSERT_V(size<0>(scores) == size<0>(row_sum));

        // Thread level reduction
#pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            row_sum(mi) = ZeroInit ? scores(mi, 0) : row_sum(mi) + scores(mi, 0);
#pragma unroll
            for (int ni = 1; ni < size<1>(scores); ++ni) {
                row_sum(mi) = row_sum(mi) + scores(mi, ni);
            }
        }
    }

    template <bool Init, typename STensor, typename OTensor>
    __forceinline__ __device__ void
    rescale_o(STensor& rS, OTensor& rO) {
        Tensor scores = make_tensor(rS.data(), retile_mma(rS.layout()));
        static_assert(decltype(size<0>(scores))::value == NRows);
        if constexpr (Init) {
            reduce_row_max<true>(scores);
            safe_exp(scores);
            reduce_row_sum<true>(scores);
        } else {
            Tensor prev_row_max = make_fragment_like(row_max);
            cute::copy(row_max, prev_row_max);

            reduce_row_max<false>(scores);

            Tensor outs = make_tensor(rO.data(), retile_mma(rO.layout()));
            static_assert(decltype(size<0>(outs))::value == NRows);

#pragma unroll
            for (int mi = 0; mi < size<0>(outs); ++mi) {
                half cur_max = row_max(mi);
                half score_scale = hexp2((prev_row_max(mi) - cur_max) * softmax_scale_log2);
                row_sum(mi) *= score_scale;
#pragma unroll
                for (int ni = 0; ni < size<1>(outs); ++ni) {
                    outs(mi, ni) *= score_scale;
                }
            }

            safe_exp(scores);
            reduce_row_sum<false>(scores);
        }
    }

    template <typename OTensor>
    __forceinline__ __device__ auto
    normalize(OTensor& rO) {
        // Warp level reduction
#pragma unroll
        for (int i = 0; i < size<0>(row_max); ++i) {
            // Every consecutive 4 threads can get their own row_sum(i)
#pragma unroll
            for (int offset = 2; offset >= 1; offset >>= 1) {
                row_sum(i) = row_sum(i) + __shfl_xor_sync(uint32_t(-1), row_sum(i), offset);
            }
        }

        Tensor lse = make_fragment_like(row_sum);
        Tensor outs = make_tensor(rO.data(), retile_mma(rO.layout()));
        static_assert(decltype(size<0>(outs))::value == NRows);

#pragma unroll
        for (int mi = 0; mi < size<0>(outs); ++mi) {
            half sum = row_sum(mi);
            half inv_sum = sum == CUDART_ZERO_FP16 || __hisnan(sum) ? CUDART_ONE_FP16 : CUDART_ONE_FP16 / sum;
            lse(mi) = sum == CUDART_ZERO_FP16 || __hisnan(sum) ? -CUDART_INF_FP16 : row_max(mi) * softmax_scale + hlog(sum);
#pragma unroll
            for (int ni = 0; ni < size<1>(outs); ++ni) {
                outs(mi, ni) *= inv_sum;
            }
        }

        return lse;
    }
};
