# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
set(ASCEND_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${ASCEND_CMAKE_PATH})
project(Catlass LANGUAGES ASCEND CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

if(NOT DEFINED NPU_MODEL OR NPU_MODEL STREQUAL "")
    execute_process(
        COMMAND python -c "import acl;print(acl.get_soc_name())"
        OUTPUT_VARIABLE NPU_MODEL
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NPU_MODEL STREQUAL "")
        message(WARNING "Failed to get NPU_MODEL. Simulator or aclnn example may not work.\nYou can set NPU_MODEL manually by -DNPU_MODEL=Ascendxxxyy.")
    else()
        message("Detected NPU_MODEL: ${NPU_MODEL}")
    endif()
endif()

execute_process(
    COMMAND python -c "import acl;print(acl.get_cann_version(0)[0]['version'])"
    OUTPUT_VARIABLE CANN_VERSION
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message("Detected CANN_VERSION: ${CANN_VERSION}")

set(CATLASS_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(CATLASS_3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty)

if(NOT DEFINED CMAKE_BUILD_TYPE OR NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT DEFINED CATLASS_ARCH OR NOT CATLASS_ARCH)
    message(WARNING "CATLASS_ARCH is not defined, use default value \"2201\"")
    set(CATLASS_ARCH 2201)
endif()

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# Compile options for secure compile
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_SKIP_RPATH TRUE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fstack-protector-strong>)
    add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-Xhost-start -fstack-protector-strong -Xhost-end>")
    add_link_options(-Wl,-z,relro,-z,now)
    add_link_options(-Wl,-z,noexecstack)
    add_link_options(-s)
endif()

add_compile_definitions(CATLASS_BUILD_LEGACY)

add_subdirectory(examples)
add_subdirectory(tools)
add_subdirectory(3rdparty)

if(DEFINED BUILD_TESTS AND BUILD_TESTS)
    add_subdirectory(tests)
endif()
