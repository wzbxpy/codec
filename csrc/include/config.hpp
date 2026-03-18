#pragma once

#include <cute/tensor.hpp>

using namespace cute;

template <int HeadDim, int NHeads, int NHeadsKV, int PageSize, int BlockQ, int BlockKV, int NWarps>
struct Stage1Config {
    static constexpr int kHeadDim = HeadDim;
    static constexpr int kNHeads = NHeads;
    static constexpr int kNHeadsKV = NHeadsKV;
    static constexpr int kNHeadsPerKV = NHeads / NHeadsKV;
    static constexpr int kPageSize = PageSize;
    static constexpr int kBlockQ = BlockQ;
    static constexpr int kBlockKV = BlockKV;
    static constexpr int kNWarps = NWarps;
    static constexpr int kNThreads = NWarps * 32;

    using TiledMMA = TiledMMA<MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>, Layout<Shape<Int<NWarps>, _1, _1>>, Tile<Int<16 * NWarps>, _16, _16>>;

    static constexpr int kSwizzleB = 3;
    static constexpr int kSwizzleM = 3;
    static constexpr int kSwizzleS = 3;

    static constexpr int kSmemRowStride = 1 << (kSwizzleM + kSwizzleS);
    static constexpr int kSmemNRow = 1 << kSwizzleB;
    static_assert(HeadDim % kSmemRowStride == 0);

    using SmemLayoutAtomQKVO = decltype(composition(Swizzle<kSwizzleB, kSwizzleM, kSwizzleS>{}, Layout<Shape<Int<kSmemNRow>, Int<kSmemRowStride>>, Stride<Int<kSmemRowStride>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQKVO{}, Shape<Shape<Int<kNHeadsPerKV>, Int<BlockQ>>, Int<HeadDim>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtomQKVO{}, Shape<Int<BlockKV>, Int<HeadDim>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockKV>>{}, GenRowMajor{})));
    using SmemLayoutO = SmemLayoutQ;

    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(half);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(half);
    static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;

    using SmemCopyAtomQK = Copy_Atom<SM75_U32x4_LDSM_N, half>;
    using SmemCopyAtomVtransposed = Copy_Atom<SM75_U16x8_LDSM_T, half>;
    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, half>;

    static constexpr int kGmemCopyAtomSize = sizeof(uint128_t) / sizeof(half);
    static constexpr int kGmemThreadsPerRow = kSmemRowStride / kGmemCopyAtomSize;
    using GmemTiledCopyTLayoutQKVO = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>, Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyQKV = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half>{}, GmemTiledCopyTLayoutQKVO{}, Layout<Shape<_1, Int<kGmemCopyAtomSize>>>{}));
    using GmemTiledCopyO = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, half>{}, GmemTiledCopyTLayoutQKVO{}, Layout<Shape<_1, Int<kGmemCopyAtomSize>>>{}));
};

template <int HeadDim, int NHeads, int NSplits, int BlockO, int NWarps>
struct Stage2Config {
    static constexpr int kHeadDim = HeadDim;
    static constexpr int kNHeads = NHeads;
    static constexpr int kNSplits = NSplits;
    static constexpr int kBlockO = BlockO;
    static constexpr int kNWarps = NWarps;
    static constexpr int kNThreads = NWarps * 32;

    static constexpr int kSwizzleB = 1;
    static constexpr int kSwizzleM = 0;
    static constexpr int kSwizzleS = 3;

    static constexpr int kSmemRowStride = 1 << (kSwizzleM + kSwizzleS);
    static constexpr int kSmemNRow = 1 << kSwizzleB;

    using SmemLayoutAtomLSE = decltype(composition(Swizzle<kSwizzleB, kSwizzleM, kSwizzleS>{}, Layout<Shape<Int<kSmemNRow>, Int<kSmemRowStride>>, Stride<Int<kSmemRowStride>, _1>>{}));
    using SmemLayoutLSE = decltype(tile_to_shape(SmemLayoutAtomLSE{}, Shape<Int<NSplits>, Int<BlockO>>{}));

    // TODO: Compare with UniversalCopy
    static constexpr int kGmemThreadsPerRowO = kNThreads / kBlockO;
    using GmemTiledCopyTLayoutO = Layout<Shape<Int<kBlockO>, Int<kGmemThreadsPerRowO>>, Stride<Int<kGmemThreadsPerRowO>, _1>>;
    using GmemTiledCopyO = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, half>{},
                                                    GmemTiledCopyTLayoutO{},
                                                    Layout<Shape<_1, _4>>{}));  // Val layout, 4 vals per store
    using GmemTiledCopyOaccum = GmemTiledCopyO;
};

template <int HeadDim, int NHeads, int BlockO, int NWarps>
struct Stage3Config {
    static constexpr int kHeadDim = HeadDim;
    static constexpr int kNHeads = NHeads;
    static constexpr int kBlockO = BlockO;
    static constexpr int kNWarps = NWarps;
    static constexpr int kNThreads = NWarps * 32;

    static constexpr int kSwizzleB = 1;
    static constexpr int kSwizzleM = 0;
    static constexpr int kSwizzleS = 3;

    static constexpr int kSmemRowStride = 1 << (kSwizzleM + kSwizzleS);
    static constexpr int kSmemNRow = 1 << kSwizzleB;

    using SmemLayoutAtomLSE = decltype(composition(Swizzle<kSwizzleB, kSwizzleM, kSwizzleS>{}, Layout<Shape<Int<kSmemNRow>, Int<kSmemRowStride>>, Stride<Int<kSmemRowStride>, _1>>{}));
    using SmemLayoutLSE = decltype(tile_to_shape(SmemLayoutAtomLSE{}, Shape<Int<2>, Int<BlockO>>{}));

    // TODO: Compare with UniversalCopy
    static constexpr int kGmemThreadsPerRowO = kNThreads / kBlockO;
    using GmemTiledCopyTLayoutO = Layout<Shape<Int<kBlockO>, Int<kGmemThreadsPerRowO>>, Stride<Int<kGmemThreadsPerRowO>, _1>>;
    using GmemTiledCopyO = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, half>{},
                                                    GmemTiledCopyTLayoutO{},
                                                    Layout<Shape<_1, _4>>{}));  // Val layout, 4 vals per store
    using GmemTiledCopyOaccum = GmemTiledCopyO;
};
