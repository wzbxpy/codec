/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
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

// Helper methods to check for errors
#include "fai_kernel.h"
#include "fai_tiling.h"
#include "golden.hpp"
#include "helper.hpp"

using namespace std;

// This code section describes the parameters to execute the run function.
struct Options {
    static constexpr auto HELPER =
        "Usage: fai batch qSeqlen kvSeqlen numHeads kvHeads embeddingSize isVariedLen maskType cacheMode [--dtype DTYPE "
        "--datapath DATA_PATH --device DEVICE_ID]\n";
    static constexpr auto MIN_ARGS = 7;

    // Define default value.
    uint32_t batch{0};
    uint32_t qSeqlen{0};
    uint32_t kvSeqlen{0};
    uint32_t numHeads{0};
    uint32_t kvHeads{0};
    uint32_t embeddingSize{0};
    uint32_t isVariedLen{0};
    uint32_t maskType{0};
    uint32_t cacheMode{0};
    uint32_t deviceId{0};
    uint32_t blockSize{128};
    string dataType = "half";
    string dataPath = "../../examples/49_ascend950_flash_attention_infer/data";

    Options() = default;

    // Define function to parse the command-line arguments.
    int Parse(int argc, const char **argv) {
        // The number of arguments must >= 7.
        if (argc < MIN_ARGS) {
            printf(HELPER);
            return -1;
        }

        // Allocate arguments to parameters.
        uint32_t argIndex = 1;
        batch = atoi(argv[argIndex++]);
        qSeqlen = atoi(argv[argIndex++]);
        kvSeqlen = atoi(argv[argIndex++]);
        numHeads = atoi(argv[argIndex++]);
        kvHeads = atoi(argv[argIndex++]);
        embeddingSize = atoi(argv[argIndex++]);
        isVariedLen = atoi(argv[argIndex++]);
        maskType = atoi(argv[argIndex++]);
        cacheMode = atoi(argv[argIndex++]);
        while (argIndex < argc) {
            string flag = string(argv[argIndex++]);
            if (flag == "--datapath") {
                dataPath = string(argv[argIndex++]);
            } else if (flag == "--device") {
                deviceId = atoi(argv[argIndex++]);
            } else if (flag == "--dtype") {
                dataType = string(argv[argIndex++]);
            } else {
                printf(HELPER);
                return -1;
            }
        }
        return 0;
    }
};

void AllocMem(uint8_t **host, uint8_t **device, size_t size) {
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(host), size));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(device), size, ACL_MEM_MALLOC_HUGE_FIRST));
}

void FreeMem(uint8_t *host, uint8_t *device) {
    ACL_CHECK(aclrtFreeHost(host));
    ACL_CHECK(aclrtFree(device));
}

// Allocate several matrices in NPU device memory and call a
// CATLASS FAI kernel.
void Run(const Options &options) {
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // Parameters initialization.
    int32_t batch = options.batch;
    int32_t qSeqlen = options.qSeqlen;
    int32_t kvSeqlen = options.kvSeqlen;
    int32_t numHeads = options.numHeads;
    int32_t kvHeads = options.kvHeads;
    int32_t embeddingSize = options.embeddingSize;
    int32_t blockSize = options.blockSize;
    int32_t maskType = options.maskType;
    int32_t cacheMode = options.cacheMode;
    int32_t isVariedLen = options.isVariedLen;
    string dataType = options.dataType;
    string dataPath = options.dataPath;
    int32_t maxKvSeqlen = kvSeqlen;
    int32_t numBlocks = batch * ((maxKvSeqlen + blockSize - 1) / blockSize);

    if ((dataType != "half") && (dataType != "bf16")) {
        cerr << "[ERROR] dtype must be 'half' or 'bf16'." << endl;
        return;
    }
    if (isVariedLen > 0) {
        cerr << "[ERROR] isVariedLen must be '0'." << endl;
        return;
    }
    if (embeddingSize > 128) {
        cerr << "[ERROR] embeddingSize is only supported up to 128. The current template only implements BlockMmadPV output to UB. "
                "If larger embeddingSize need to be supported, the BlockMmadPV output to GM must be added."<< endl;
        return;
    }

    // read qNtokens num
    void *qNtokens = nullptr;
    ACL_CHECK(aclrtMallocHost(&qNtokens, 1 * sizeof(int32_t)));
    ReadFile(dataPath + "/q_ntokens.bin", qNtokens, 1 * sizeof(int32_t));
    int32_t numTokens = static_cast<int32_t *>(qNtokens)[0];

    uint64_t seqArraySize = batch * sizeof(int64_t);
    uint64_t qoSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)embeddingSize * sizeof(fp16_t);
    uint64_t kvSize = (uint64_t)numBlocks * (uint64_t)blockSize * (uint64_t)kvHeads * (uint64_t)embeddingSize
                      * sizeof(fp16_t);
    uint64_t maskSize = batch * qSeqlen * kvSeqlen * sizeof(uint8_t);
    uint64_t blockTableSize = static_cast<uint64_t>(
        batch * ((maxKvSeqlen + blockSize - 1) / blockSize) * sizeof(int32_t)
    );
    uint32_t tilingSize = sizeof(FATilingData);

    // Allocate matrices in host and device memory.
    uint8_t *qSeqHost;
    uint8_t *qSeqDevice;
    AllocMem(&qSeqHost, &qSeqDevice, seqArraySize);
    ReadFile(dataPath + "/q_seqlen.bin", qSeqHost, seqArraySize);
    ACL_CHECK(aclrtMemcpy(qSeqDevice, seqArraySize, qSeqHost, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory.
    uint8_t *kvSeqHost;
    uint8_t *kvSeqDevice;
    AllocMem(&kvSeqHost, &kvSeqDevice, seqArraySize);
    ReadFile(dataPath + "/kv_seqlen.bin", kvSeqHost, seqArraySize);
    ACL_CHECK(aclrtMemcpy(kvSeqDevice, seqArraySize, kvSeqHost, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix q.
    uint8_t *qHost;
    uint8_t *qDevice;
    AllocMem(&qHost, &qDevice, qoSize);
    ReadFile(dataPath + "/q.bin", qHost, qoSize);
    ACL_CHECK(aclrtMemcpy(qDevice, qoSize, qHost, qoSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix k.
    uint8_t *kHost;
    uint8_t *kDevice;
    AllocMem(&kHost, &kDevice, kvSize);
    ReadFile(dataPath + "/k.bin", kHost, kvSize);
    ACL_CHECK(aclrtMemcpy(kDevice, kvSize, kHost, kvSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix v.
    uint8_t *vHost;
    uint8_t *vDevice;
    AllocMem(&vHost, &vDevice, kvSize);
    ReadFile(dataPath + "/v.bin", vHost, kvSize);
    ACL_CHECK(aclrtMemcpy(vDevice, kvSize, vHost, kvSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix v.
    uint8_t *maskHost;
    uint8_t *maskDevice;
    if (maskType > 0) {
        AllocMem(&maskHost, &maskDevice, maskSize);
        ReadFile(dataPath + "/mask.bin", maskHost, maskSize);
        ACL_CHECK(aclrtMemcpy(maskDevice, maskSize, maskHost, maskSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // Allocate matrices in host and device memory and load Matrix block_table.
    uint8_t *blockTableHost;
    uint8_t *blockTableDevice;
    AllocMem(&blockTableHost, &blockTableDevice, blockTableSize);
    ReadFile(dataPath + "/block_table.bin", blockTableHost, blockTableSize);
    ACL_CHECK(aclrtMemcpy(blockTableDevice, blockTableSize, blockTableHost, blockTableSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in device memory for workspace.
    // One base workspace block contains 65536 elements.
    uint8_t *oDevice{nullptr};
    ACL_CHECK(aclrtMalloc((void **)(&oDevice), qoSize * 2, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *tilingDevice;
    ACL_CHECK(aclrtMalloc((void **)(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // get tiling
    void *tilingHost = nullptr;
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    uint32_t blockDim = aicCoreNum;

    FAInferTiling::FAInfo faInfo;
    faInfo.batchSize = batch;
    faInfo.numOfHeads = numHeads;
    faInfo.numOfKVHeads = kvHeads;
    faInfo.seqSize = qSeqlen;
    faInfo.seqInnerSize = kvSeqlen;
    faInfo.headSize = embeddingSize;
    faInfo.scaleValue = static_cast<float>(1.0 / std::sqrt(1.0 * faInfo.headSize));
    faInfo.blockSize = blockSize;
    faInfo.maxBlockNumPerBatch = (kvSeqlen + blockSize - 1) / blockSize;

    FATilingData faTilingData;
    FAInferTiling::GetFATilingParam(faInfo, blockDim, faTilingData);
    tilingHost = reinterpret_cast<void *>(&faTilingData);

    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));
    for (int i = 0; i < 1; i++) {
        if (dataType == "half") {
            if (maskType > 0 && cacheMode == 0) {
                FAInferTla<half, true, false><<<blockDim, nullptr, stream>>>(
                    qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, qSeqDevice, kvSeqDevice,
                    tilingDevice
                );
            } else if (maskType == 0 && cacheMode > 0) {
                FAInferTla<half, false, true><<<blockDim, nullptr, stream>>>(
                    qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, qSeqDevice, kvSeqDevice,
                    tilingDevice
                );
            } else if (maskType > 0 && cacheMode > 0) {
                FAInferTla<half, true, true><<<blockDim, nullptr, stream>>>(
                    qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, qSeqDevice, kvSeqDevice,
                    tilingDevice
                );
            } else {
                FAInferTla<half, false, false><<<blockDim, nullptr, stream>>>(
                    qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, qSeqDevice, kvSeqDevice,
                    tilingDevice
                );
            }
        } else {
            if (maskType > 0 && cacheMode == 0) {
                FAInferTla<bfloat16_t, true, false><<<blockDim, nullptr, stream>>>(
                    qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, qSeqDevice, kvSeqDevice,
                    tilingDevice
                );
            } else if (maskType == 0 && cacheMode > 0) {
                FAInferTla<bfloat16_t, false, true><<<blockDim, nullptr, stream>>>(
                    qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, qSeqDevice, kvSeqDevice,
                    tilingDevice
                );
            } else if (maskType > 0 && cacheMode > 0) {
                FAInferTla<bfloat16_t, true, true><<<blockDim, nullptr, stream>>>(
                    qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, qSeqDevice, kvSeqDevice,
                    tilingDevice
                );
            } else {
                FAInferTla<bfloat16_t, false, false><<<blockDim, nullptr, stream>>>(
                    qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, qSeqDevice, kvSeqDevice,
                    tilingDevice
                );
            }
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        // Copy the result from device to host
        vector<fp16_t> oHostHalf(qoSize / sizeof(fp16_t));
        vector<op::bfloat16> oHostBf16(qoSize / sizeof(op::bfloat16));
        if (dataType == "half") {
            ACL_CHECK(aclrtMemcpy(oHostHalf.data(), qoSize, oDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));
        } else if (dataType == "bf16") {
            ACL_CHECK(aclrtMemcpy(oHostBf16.data(), qoSize, oDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));
        }
        // Compute the golden result
        vector<float> goldenHost(qoSize / sizeof(fp16_t));
        const size_t goldenSize = qoSize * 2;
        ReadFile(dataPath + "/golden.bin", goldenHost.data(), goldenSize);

        // Compare the result
        vector<uint64_t> errorIndices = (dataType == "half") ? golden::CompareData(oHostHalf, goldenHost, kvSeqlen)
                                                             : golden::CompareData(oHostBf16, goldenHost, kvSeqlen);
        // vector<uint64_t> errorIndices = golden::CompareData(oHostHalf, goldenHost, kvSeqlen);
        if (errorIndices.empty()) {
            cout << "Compare success." << endl;
        } else {
            cout << "Compare failed. Error count: " << errorIndices.size() << endl;
        }
    }

    // Free host memory allocations.
    FreeMem(qSeqHost, qSeqDevice);
    FreeMem(kvSeqHost, kvSeqDevice);
    FreeMem(qHost, qDevice);
    FreeMem(kHost, kDevice);
    FreeMem(vHost, vDevice);
    if (maskType > 0) {
        FreeMem(maskHost, maskDevice);
    }
    FreeMem(blockTableHost, blockTableDevice);
    aclrtFree(oDevice);
    aclrtFree(tilingDevice);
    aclrtFreeHost(qNtokens);

    // Destroy specified Stream and reset device.
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

/// Entry point to mla example.

int main(int argc, const char **argv) {
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}