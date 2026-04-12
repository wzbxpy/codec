/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
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

#include <iostream>
#include <vector>
#include <cstdlib>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_scheduler_aswt.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_tla.hpp"

#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "golden.hpp"
#include "helper.hpp"
#include "tla/layout.hpp"

#if (defined (CATLASS_ARCH) && CATLASS_ARCH == 3510)
#include "catlass/epilogue/tile/tile_pertoken_dequant.hpp"
#endif

using namespace Catlass;
using namespace tla;

using Options = GroupedGemmOptions;

static void Run(Options const &options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t problemCount = options.problemCount;
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n * problemCount;
    size_t lenScale = static_cast<size_t>(n) * problemCount;
    size_t lenPerToken = static_cast<size_t>(m);
    size_t lenD = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(int8_t);
    size_t sizeB = lenB * sizeof(int8_t);
    size_t sizeScale = lenScale * sizeof(float);
    size_t sizePerToken = lenPerToken * sizeof(float);
    size_t sizeD = lenD * sizeof(fp16_t);

    std::vector<int8_t> hostA(lenA);
    std::vector<int8_t> hostB(lenB);
    std::vector<float> hostScale(lenScale);
    std::vector<float> hostPerToken(lenPerToken);

    golden::FillRandomData(hostA, -16, 16);
    golden::FillRandomData(hostB, -16, 16);
    golden::FillRandomData(hostScale, 0.0, 1.0);
    golden::FillRandomData(hostPerToken, 0.0, 1.0);

    auto groupList = golden::GenerateGroupList<int64_t>(m, problemCount);
    // 此处采取平均的方式，若需要随机，则注释该段
    for(int i = 0; i < problemCount; i++) {
        groupList[i] = (i + 1) * (m / problemCount);
    }

    size_t sizeGroupList = problemCount * sizeof(int64_t);
    uint8_t *deviceGroupList{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceGroupList), sizeGroupList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceGroupList, sizeGroupList, groupList.data(), sizeGroupList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceScale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceScale), sizeScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceScale, sizeScale, hostScale.data(), sizeScale, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *devicePerToken{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devicePerToken), sizePerToken, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(devicePerToken, sizePerToken, hostPerToken.data(), sizePerToken, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceD{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    size_t sizeWorkspace = 0;
    uint8_t *deviceWorkspace{nullptr};
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int32_t;
    using ElementScale = float;
    using ElementPerToken = float;
    using ElementD = half;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;
    using LayoutTagScale = layout::VectorLayout;
    using LayoutTagPerToken = layout::VectorLayout;
    using LayoutTagD = layout::RowMajor;

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    constexpr bool useHF32 = false;
    
    using DispatchPolicyMmadTla = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32>;
    using L1TileShape = Shape<Int<256>, Int<256>, Int<512>>;
    using L0TileShape = Shape<Int<256>, Int<256>, Int<128>>;
    using TileCopyMmadTla = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC,
        void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadTla = Gemm::Block::BlockMmadTla<
        DispatchPolicyMmadTla, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopyMmadTla>;
    
    constexpr bool ubDB = false;
    constexpr uint32_t UB_STAGES = ubDB ?  2 : 1;
    using EpilogueTileShape = MatrixShape<256, 256>;
    using DispatchPolicyDequant = Epilogue::EpilogueAscend950PerTokenDequantTla<UB_STAGES>;
    using TilePerTokenDequant = Epilogue::Tile::TilePerTokenDequant<
        ArchTag, ElementC, ElementScale, ElementPerToken, ElementD, EpilogueTileShape>;
    using TileCopyEpilogue = Epilogue::Tile::TileCopyDequantTla<
        ArchTag, ElementC, LayoutTagC, ElementScale, LayoutTagScale, ElementPerToken, LayoutTagPerToken, ElementD, LayoutTagD>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        DispatchPolicyDequant, EpilogueTileShape, ElementC, ElementScale, ElementPerToken, ElementD,
        TilePerTokenDequant, TileCopyEpilogue>;
    constexpr bool isGmm = true;
    using BlockScheduler = typename Gemm::Block::BlockSchedulerAswt<L1TileShape, L0TileShape, isGmm>;
    // kernel level
    using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceMPerTokenTla<
        BlockMmadTla, BlockEpilogue, BlockScheduler, int64_t>;
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    MatmulKernel::Arguments arguments{
        options.problemShape,
        problemCount,
        deviceGroupList, 
        deviceA, 
        deviceB,
        deviceScale,
        devicePerToken, 
        deviceD
    };
    // call a kernel
    MatmulAdapter matmul_op;
    //judge arguments can run
    matmul_op.CanImplement(arguments);
    // get workspace
    sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    if(sizeWorkspace > 0){
        ACL_CHECK(
        aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST);
        );
    }
    // initalize kernel argument
    matmul_op.Initialize(arguments, deviceWorkspace);
    matmul_op(stream, aicCoreNum);
    
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<fp16_t> hostD(lenD);
    ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

    LayoutTagA layoutTagA{m, k};
    LayoutTagB layoutTagB{k, n};
    LayoutTagScale layoutTagScale{n};
    LayoutTagPerToken layoutTagPerToken{m};
    LayoutTagD  layoutTagD{m, n};
    std::vector<float> hostGolden(lenD);
    golden::ComputeGroupedMatmulPerTokenDequant(
        options.problemShape, problemCount, groupList, hostA, layoutTagA, hostB, layoutTagB, hostScale, layoutTagScale, 
        hostPerToken, layoutTagPerToken, hostGolden, layoutTagD
    );

    std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, k, groupList[problemCount - 1] * n);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
        std::cerr << "Compare failed. errorIndices[0]: " << errorIndices[0] << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceGroupList));
    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceScale));
    ACL_CHECK(aclrtFree(devicePerToken));
    ACL_CHECK(aclrtFree(deviceD));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) == 0) {
        Run(options);
    }
    return 0;
}