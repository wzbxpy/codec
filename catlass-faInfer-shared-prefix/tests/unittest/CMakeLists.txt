# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(GOOGLETEST_INCLUDE_DIR ${CATLASS_3RDPARTY_DIR}/googletest/googletest/include)

add_compile_options($<$<COMPILE_LANGUAGE:ASCEND>:--cce-aicore-arch=dav-c220>)
add_executable(catlass_unittest catlass_unittest.cpp)

# target_compile_options(catlass_unittest PRIVATE -fprofile-instr-generate -fcoverage-mapping)
file(GLOB_RECURSE CATLASS_UNITTEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(CATLASS_UNITTEST_SRC ${CATLASS_UNITTEST_SRCS})
set_source_files_properties(${CATLASS_UNITTEST_SRC} PROPERTIES LANGUAGE ASCEND)
endforeach()

target_sources(catlass_unittest PRIVATE ${CATLASS_UNITTEST_SRCS})

target_include_directories(catlass_unittest PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GOOGLETEST_INCLUDE_DIR}
    ${CATLASS_INCLUDE_DIR}
)

target_link_libraries(catlass_unittest PRIVATE gtest gtest_main m)
