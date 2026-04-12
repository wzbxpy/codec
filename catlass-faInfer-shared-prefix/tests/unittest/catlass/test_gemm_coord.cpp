/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>

#include <gtest/gtest.h>

#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
using namespace Catlass;
TEST(gemm_coord, normal_case)
{
    GemmCoord gemmCoord(256, 512, 1024);
    EXPECT_EQ(gemmCoord.m(), 256);
    EXPECT_EQ(gemmCoord.n(), 512);
    EXPECT_EQ(gemmCoord.k(), 1024);
    auto coordMN = gemmCoord.GetCoordMN();
    auto coordMK = gemmCoord.GetCoordMK();
    auto coordKN = gemmCoord.GetCoordKN();
    EXPECT_EQ(coordMN[0], 256);
    EXPECT_EQ(coordMN[1], 512);
    EXPECT_EQ(coordMK[0], 256);
    EXPECT_EQ(coordMK[1], 1024);
    EXPECT_EQ(coordKN[0], 1024);
    EXPECT_EQ(coordKN[1], 512);
}
TEST(gemm_coord, u32_max_case)
{
    // Edge case: all uint32_t max
    GemmCoord gemmCoord_u32_max(
        std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()
    );
    EXPECT_EQ(gemmCoord_u32_max.m(), std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(gemmCoord_u32_max.n(), std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(gemmCoord_u32_max.k(), std::numeric_limits<uint32_t>::max());
    auto gemmCoord_u32_max_MN = gemmCoord_u32_max.GetCoordMN();
    auto gemmCoord_u32_max_MK = gemmCoord_u32_max.GetCoordMK();
    auto gemmCoord_u32_max_KN = gemmCoord_u32_max.GetCoordKN();
    EXPECT_EQ(gemmCoord_u32_max_MN[0], std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(gemmCoord_u32_max_MN[1], std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(gemmCoord_u32_max_MK[0], std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(gemmCoord_u32_max_MK[1], std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(gemmCoord_u32_max_KN[0], std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(gemmCoord_u32_max_KN[1], std::numeric_limits<uint32_t>::max());
}