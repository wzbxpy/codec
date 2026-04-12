/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/gemm/kernel/strided_batched_matmul_tla.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "golden.hpp"
#include "helper.hpp"


using namespace Catlass;
using namespace tla;

struct BatchedMatmulTlaOptions {
    const std::string HELPER =
        "problem_count m n k [device_id] [lda ldb ldc] [strideA strideB strideC] [layoutA layoutB]\n"
        "  layoutA/layoutB: row | col (case-insensitive)\n"
        "  Note: C is row in this example (ldc>=n, strideC based on row).";

    Catlass::GemmCoord problemShape{128, 128, 128};
    uint32_t problemCount{1};
    int32_t deviceId{0};

    // Stride customization (unit: elements).
    // - lda: stride of A on M axis (A is [M,K] RowMajor)
    // - ldb: stride of B on K axis (B is [K,N] RowMajor)
    // - ldc: stride of C on M axis (C is [M,N] RowMajor)
    int64_t lda{-1};
    int64_t ldb{-1};
    int64_t ldc{-1};
    // stride between batches
    int64_t strideA{-1};
    int64_t strideB{-1};
    int64_t strideC{-1};

    // Layout selection
    // - A: [M,K] (RowMajor or ColumnMajor)
    // - B: [K,N] (RowMajor or ColumnMajor)
    // - C: [M,N] (RowMajor only in this example)
    enum class MatrixLayout { RowMajor, ColumnMajor };
    MatrixLayout layoutA{MatrixLayout::RowMajor};
    MatrixLayout layoutB{MatrixLayout::RowMajor};

    BatchedMatmulTlaOptions() = default;

    static bool IsLayoutToken(const std::string &s) {
        if (s.empty()) {
            return false;
        }
        std::string t;
        t.resize(s.size());
        std::transform(s.begin(), s.end(), t.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return (t == "row" || t == "col");
    }

    static MatrixLayout ParseLayoutToken(const std::string &s) {
        std::string t;
        t.resize(s.size());
        std::transform(s.begin(), s.end(), t.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (t == "row") {
            return MatrixLayout::RowMajor;
        }
        return MatrixLayout::ColumnMajor;
    }

    int Parse(int argc, const char **argv) {
        // Supported forms:
        // 1) name problem_count m n k
        // 2) name problem_count m n k device_id
        // 3) name problem_count m n k device_id lda ldb ldc
        // 4) name problem_count m n k device_id lda ldb ldc strideA strideB strideC
        // Each form may optionally append: layoutA layoutB (two tokens), e.g. "row col".
        int argcEffective = argc;
        if (argc >= 7) {
            std::string maybeA(argv[argc - 2]);
            std::string maybeB(argv[argc - 1]);
            if (IsLayoutToken(maybeA) && IsLayoutToken(maybeB)) {
                layoutA = ParseLayoutToken(maybeA);
                layoutB = ParseLayoutToken(maybeB);
                argcEffective -= 2;
            }
        }

        if (!(argcEffective == 5 || argcEffective == 6 || argcEffective == 9 || argcEffective == 12)) {
            std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << " " << HELPER << std::endl;
            return -1;
        }


        problemCount = std::atoi(argv[1]);
        problemShape.m() = std::atoi(argv[2]);
        problemShape.n() = std::atoi(argv[3]);
        problemShape.k() = std::atoi(argv[4]);

        if (argcEffective >= 6) {
            deviceId = std::atoi(argv[5]);
        }

        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();
        uint32_t k = problemShape.k();

        // Default: contiguous per-matrix and contiguous between batches.
        // Interpret lda/ldb as the leading dimension in the chosen layout.
        lda = (layoutA == MatrixLayout::RowMajor) ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
        ldb = (layoutB == MatrixLayout::RowMajor) ? static_cast<int64_t>(n) : static_cast<int64_t>(k);
        ldc = static_cast<int64_t>(n); // C is RowMajor in this example

        strideA = (layoutA == MatrixLayout::RowMajor) ? static_cast<int64_t>(m) * lda : static_cast<int64_t>(k) * lda;
        strideB = (layoutB == MatrixLayout::RowMajor) ? static_cast<int64_t>(k) * ldb : static_cast<int64_t>(n) * ldb;
        strideC = static_cast<int64_t>(m) * ldc;

        if (argcEffective >= 9) {
            lda = std::atoll(argv[6]);
            ldb = std::atoll(argv[7]);
            ldc = std::atoll(argv[8]);

            strideA = (layoutA == MatrixLayout::RowMajor) ? static_cast<int64_t>(m) * lda : static_cast<int64_t>(k) * lda;
            strideB = (layoutB == MatrixLayout::RowMajor) ? static_cast<int64_t>(k) * ldb : static_cast<int64_t>(n) * ldb;
            strideC = static_cast<int64_t>(m) * ldc;
        }
        if (argcEffective == 12) {
            strideA = std::atoll(argv[9]);
            strideB = std::atoll(argv[10]);
            strideC = std::atoll(argv[11]);
        }

        // Basic validation for ND layouts.
        int64_t minLda = (layoutA == MatrixLayout::RowMajor) ? static_cast<int64_t>(k) : static_cast<int64_t>(m);
        int64_t minLdb = (layoutB == MatrixLayout::RowMajor) ? static_cast<int64_t>(n) : static_cast<int64_t>(k);
        if (lda < minLda || ldb < minLdb || ldc < static_cast<int64_t>(n)) {
            std::cerr << "Invalid leading dimensions: require lda>=" << minLda
                      << ", ldb>=" << minLdb
                      << ", ldc>=" << n << "." << std::endl;
            return -1;
        }

        int64_t minMatA = (layoutA == MatrixLayout::RowMajor)
            ? (static_cast<int64_t>(m - 1) * lda + static_cast<int64_t>(k))
            : (static_cast<int64_t>(k - 1) * lda + static_cast<int64_t>(m));
        int64_t minMatB = (layoutB == MatrixLayout::RowMajor)
            ? (static_cast<int64_t>(k - 1) * ldb + static_cast<int64_t>(n))
            : (static_cast<int64_t>(n - 1) * ldb + static_cast<int64_t>(k));
        int64_t minMatC = static_cast<int64_t>(m - 1) * ldc + static_cast<int64_t>(n);

        if (strideA < minMatA || strideB < minMatB || strideC < minMatC) {
            std::cerr << "Invalid batch strides: require strideA/strideB/strideC large enough for one matrix."
                      << std::endl;
            return -1;
        }

        return 0;
    }
};

using Options = BatchedMatmulTlaOptions;

template <typename LayoutTagA>
static auto MakeTlaLayoutA(uint32_t batchCount, uint32_t m, uint32_t k, int64_t strideA, int64_t lda) {
    if constexpr (std::is_same_v<LayoutTagA, layout::RowMajor>) {
        return tla::MakeLayout(
            tla::MakeShape(batchCount, m, k),
            tla::MakeStride(strideA, lda, tla::Int<1>{})
        );
    } else {
        return tla::MakeLayout(
            tla::MakeShape(batchCount, m, k),
            tla::MakeStride(strideA, tla::Int<1>{}, lda)
        );
    }
}

template <typename LayoutTagB>
static auto MakeTlaLayoutB(uint32_t batchCount, uint32_t k, uint32_t n, int64_t strideB, int64_t ldb) {
    if constexpr (std::is_same_v<LayoutTagB, layout::RowMajor>) {
        return tla::MakeLayout(
            tla::MakeShape(batchCount, k, n),
            tla::MakeStride(strideB, ldb, tla::Int<1>{})
        );
    } else {
        return tla::MakeLayout(
            tla::MakeShape(batchCount, k, n),
            tla::MakeStride(strideB, tla::Int<1>{}, ldb)
        );
    }
}

template <typename LayoutTagA, typename LayoutTagB>
static void RunWithLayouts(const Options &options) {
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t batchCount = options.problemCount;
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    using ElementA = half;
    using ElementB = half;
    using ElementC = half;

    using LayoutTagC = layout::RowMajor; // must be RowMajor
    LayoutTagA tagA{m, k, options.lda};
    LayoutTagB tagB{k, n, options.ldb};
    LayoutTagC tagC{m, n, options.ldc};

    // Capacity in elements (last element offset + 1)
    int64_t capA = (static_cast<int64_t>(batchCount) - 1) * options.strideA +
                   static_cast<int64_t>(tagA.GetOffset(MakeCoord(m - 1, k - 1))) + 1;
    int64_t capB = (static_cast<int64_t>(batchCount) - 1) * options.strideB +
                   static_cast<int64_t>(tagB.GetOffset(MakeCoord(k - 1, n - 1))) + 1;
    int64_t capC = (static_cast<int64_t>(batchCount) - 1) * options.strideC +
                   static_cast<int64_t>(tagC.GetOffset(MakeCoord(m - 1, n - 1))) + 1;

    size_t lenA = static_cast<size_t>(capA);
    size_t lenB = static_cast<size_t>(capB);
    size_t lenC = static_cast<size_t>(capC);

    size_t sizeA = lenA * sizeof(ElementA);
    size_t sizeB = lenB * sizeof(ElementB);
    size_t sizeC = lenC * sizeof(ElementC);

    // allocate memory of A and copy to device side
    std::vector<ElementA> hostA(lenA, 1.0f);
    golden::FillRandomData<ElementA>(hostA, -5.0f, 5.0f);
    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    // allocate memory of B and copy to device side
    std::vector<ElementB> hostB(lenB, 1.0f);
    golden::FillRandomData<ElementB>(hostB, -5.0f, 5.0f);
    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    // allocate memory of C
    std::vector<ElementC> hostC(lenC);
    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadPingpongTlaV2<ArchTag, true>;
    using L1TileShape = Shape<_128, _256, _256>;
    using L0TileShape = Shape<_128, _256, _64>;

    using TileCopy =
        Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    using BlockEpilogue = void;

    auto layoutA = MakeTlaLayoutA<LayoutTagA>(batchCount, m, k, options.strideA, options.lda);
    auto layoutB = MakeTlaLayoutB<LayoutTagB>(batchCount, k, n, options.strideB, options.ldb);
    auto layoutC = tla::MakeLayout(
        tla::MakeShape(batchCount, m, n),
        tla::MakeStride(options.strideC, options.ldc, tla::Int<1>{})
    );

    if (options.problemShape.m() > options.problemShape.n()) {
        // Swizzle offset is 3 and direction is 0.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::StridedBatchedMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;

        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        typename MatmulKernel::Arguments arguments{
            batchCount, options.problemShape,
            deviceA, layoutA,
            deviceB, layoutB,
            deviceC, layoutC
        };
        MatmulAdapter matmulOp;

        uint8_t *deviceWorkspace{nullptr};
        matmulOp.CanImplement(arguments);
        matmulOp.Initialize(arguments, deviceWorkspace);
        matmulOp(stream, aicCoreNum);
        ACL_CHECK(aclrtSynchronizeStream(stream));

    } else {
        // Swizzle offset is 3 and direction is 1.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::StridedBatchedMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;

        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        typename MatmulKernel::Arguments arguments{
            batchCount, options.problemShape,
            deviceA, layoutA,
            deviceB, layoutB,
            deviceC, layoutC
        };
        MatmulAdapter matmulOp;

        uint8_t *deviceWorkspace{nullptr};
        matmulOp.CanImplement(arguments);
        matmulOp.Initialize(arguments, deviceWorkspace);
        matmulOp(stream, aicCoreNum);
        ACL_CHECK(aclrtSynchronizeStream(stream));
    }

    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    // comparison of precision with matmul computed on cpu
    size_t packedLenC = static_cast<size_t>(batchCount) * m * n;
    std::vector<ElementC> packedC(packedLenC);
    std::vector<float> packedGolden(packedLenC);

    for (uint32_t b = 0; b < batchCount; ++b) {
        size_t basePacked = static_cast<size_t>(b) * m * n;
        size_t baseA = static_cast<size_t>(b) * static_cast<size_t>(options.strideA);
        size_t baseB = static_cast<size_t>(b) * static_cast<size_t>(options.strideB);
        size_t baseC = static_cast<size_t>(b) * static_cast<size_t>(options.strideC);
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < n; ++j) {
                size_t idxPacked = basePacked + static_cast<size_t>(i) * n + j;
                size_t offC = baseC + static_cast<size_t>(tagC.GetOffset(MakeCoord(i, j)));
                packedC[idxPacked] = hostC[offC];

                float acc = 0.0f;
                for (uint32_t kk = 0; kk < k; ++kk) {
                    size_t offA = baseA + static_cast<size_t>(tagA.GetOffset(MakeCoord(i, kk)));
                    size_t offB = baseB + static_cast<size_t>(tagB.GetOffset(MakeCoord(kk, j)));
                    acc += static_cast<float>(hostA[offA]) * static_cast<float>(hostB[offB]);
                }
                packedGolden[idxPacked] = acc;
            }
        }
    }

    std::vector<uint64_t> errorIndices = golden::CompareData(packedC, packedGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

static void Run(const Options &options) {
    using ML = Options::MatrixLayout;
    if (options.layoutA == ML::RowMajor && options.layoutB == ML::RowMajor) {
        RunWithLayouts<layout::RowMajor, layout::RowMajor>(options);
    } else if (options.layoutA == ML::RowMajor && options.layoutB == ML::ColumnMajor) {
        RunWithLayouts<layout::RowMajor, layout::ColumnMajor>(options);
    } else if (options.layoutA == ML::ColumnMajor && options.layoutB == ML::RowMajor) {
        RunWithLayouts<layout::ColumnMajor, layout::RowMajor>(options);
    } else {
        RunWithLayouts<layout::ColumnMajor, layout::ColumnMajor>(options);
    }
}

int main(int argc, const char **argv) {
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
