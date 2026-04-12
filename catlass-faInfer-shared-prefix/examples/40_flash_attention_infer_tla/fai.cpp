/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "fai_kernel.cpp"
#include "fai_tiling.cpp"
#include "golden.hpp"
#include "helper.hpp"
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

void WriteFile(const string& filename, const void* data, size_t size) {
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Failed to open file for writing: " << filename << endl;
        return;
    }
    file.write(static_cast<const char*>(data), size);
    file.close();
    cout << "Written" << size << "bytes to " << filename << endl;
}

// This code section describes the parameters to execute the run function.
struct Options {
    static constexpr auto HELPER =
        "Usage: fai batch qSeqlen kvSeqlen numHeads kvHeads embeddingSize isVariedLen maskType [--dtype DTYPE "
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
    uint32_t deviceId{0};
    uint32_t blockSize{128};
    string dataType = "half";
    string dataPath = "../../examples/40_flash_attention_infer_tla/data";

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

static void AllocMem(uint8_t **host, uint8_t **device, size_t size) {
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(host), size));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(device), size, ACL_MEM_MALLOC_HUGE_FIRST));
}

static void FreeMem(uint8_t *host, uint8_t *device) {
    ACL_CHECK(aclrtFreeHost(host));
    ACL_CHECK(aclrtFree(device));
}

// Allocate several matrices in NPU device memory and call a
// CATLASS FAI kernel.
static void Run(const Options &options) {
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
    string dataType = options.dataType;
    string dataPath = options.dataPath;

    int32_t sharedPreLen = kvSeqlen - qSeqlen; // 仅支持长度固定的Batch
    int32_t maskTypePre = 0;
    int32_t kvSeqlenPre = sharedPreLen;
    int32_t maxKvSeqlenPre = kvSeqlenPre;
    int32_t numBlocksPre = batch * ((maxKvSeqlenPre + blockSize - 1) / blockSize);

    int32_t kvSeqlenSuf = qSeqlen;
    int32_t maxKvSeqlenSuf = kvSeqlenSuf;
    int32_t numBlocksSuf = batch * ((maxKvSeqlenSuf + blockSize - 1) / blockSize);

    if ((dataType != "half") && (dataType != "bf16")) {
        cerr << "[ERROR] dtype must be 'half' or 'bf16'." << endl;
        return;
    }

    // read qNtokens num
    void *qNtokens = nullptr;
    ACL_CHECK(aclrtMallocHost(&qNtokens, 1 * sizeof(int32_t)));
    ReadFile(dataPath + "/q_ntokens.bin", qNtokens, 1 * sizeof(int32_t));
    int32_t numTokens = static_cast<int32_t *>(qNtokens)[0];

    uint64_t seqArraySize = batch * sizeof(int64_t);
    uint64_t qoSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)embeddingSize * sizeof(fp16_t);
    uint64_t kvSizePre = (uint64_t)numBlocksPre * (uint64_t)blockSize * (uint64_t)kvHeads * (uint64_t)embeddingSize
                      * sizeof(fp16_t);
    uint64_t kvSizeSuf = (uint64_t)numBlocksSuf * (uint64_t)blockSize * (uint64_t)kvHeads * (uint64_t)embeddingSize
                      * sizeof(fp16_t);
    uint64_t maskSize = 1024 * 1024 * sizeof(fp16_t);
    uint64_t blockTableSizePre = static_cast<uint64_t>(
        batch * ((maxKvSeqlenPre + blockSize - 1) / blockSize) * sizeof(int32_t)
    );
    uint64_t blockTableSizeSuf = static_cast<uint64_t>(
        batch * ((maxKvSeqlenSuf + blockSize - 1) / blockSize) * sizeof(int32_t)
    );
    uint32_t tilingSize = sizeof(FATilingData);

    // Allocate matrices in host and device memory.
    uint8_t *qSeqHost;
    uint8_t *qSeqDevice;
    AllocMem(&qSeqHost, &qSeqDevice, seqArraySize);
    ReadFile(dataPath + "/q_seqlen.bin", qSeqHost, seqArraySize);
    ACL_CHECK(aclrtMemcpy(qSeqDevice, seqArraySize, qSeqHost, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory.
    uint8_t *kvSeqHostPre;
    uint8_t *kvSeqDevicePre;
    AllocMem(&kvSeqHostPre, &kvSeqDevicePre, seqArraySize);
    ReadFile(dataPath + "/kv_seqlen_prefix.bin", kvSeqHostPre, seqArraySize);
    ACL_CHECK(aclrtMemcpy(kvSeqDevicePre, seqArraySize, kvSeqHostPre, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *kvSeqHostSuf;
    uint8_t *kvSeqDeviceSuf;
    AllocMem(&kvSeqHostSuf, &kvSeqDeviceSuf, seqArraySize);
    ReadFile(dataPath + "/kv_seqlen_suffix.bin", kvSeqHostSuf, seqArraySize);
    ACL_CHECK(aclrtMemcpy(kvSeqDeviceSuf, seqArraySize, kvSeqHostSuf, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix q.
    uint8_t *qHost;
    uint8_t *qDevice;
    AllocMem(&qHost, &qDevice, qoSize);
    ReadFile(dataPath + "/q.bin", qHost, qoSize);
    ACL_CHECK(aclrtMemcpy(qDevice, qoSize, qHost, qoSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix k.
    uint8_t *kHostPre;
    uint8_t *kDevicePre;
    AllocMem(&kHostPre, &kDevicePre, kvSizePre);
    ReadFile(dataPath + "/k_prefix.bin", kHostPre, kvSizePre);
    ACL_CHECK(aclrtMemcpy(kDevicePre, kvSizePre, kHostPre, kvSizePre, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *kHostSuf;
    uint8_t *kDeviceSuf;
    AllocMem(&kHostSuf, &kDeviceSuf, kvSizeSuf);
    ReadFile(dataPath + "/k_suffix.bin", kHostSuf, kvSizeSuf);
    ACL_CHECK(aclrtMemcpy(kDeviceSuf, kvSizeSuf, kHostSuf, kvSizeSuf, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix v.
    uint8_t *vHostPre;
    uint8_t *vDevicePre;
    AllocMem(&vHostPre, &vDevicePre, kvSizePre);
    ReadFile(dataPath + "/v_prefix.bin", vHostPre, kvSizePre);
    ACL_CHECK(aclrtMemcpy(vDevicePre, kvSizePre, vHostPre, kvSizePre, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *vHostSuf;
    uint8_t *vDeviceSuf;
    AllocMem(&vHostSuf, &vDeviceSuf, kvSizeSuf);
    ReadFile(dataPath + "/v_suffix.bin", vHostSuf, kvSizeSuf);
    ACL_CHECK(aclrtMemcpy(vDeviceSuf, kvSizeSuf, vHostSuf, kvSizeSuf, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix mask.
    uint8_t *maskHost;
    uint8_t *maskDevice;
    if (maskType == 1) {
        AllocMem(&maskHost, &maskDevice, maskSize);
        ReadFile(dataPath + "/mask.bin", maskHost, maskSize);
        ACL_CHECK(aclrtMemcpy(maskDevice, maskSize, maskHost, maskSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // Allocate matrices in host and device memory and load Matrix block_table.
    uint8_t *blockTableHostPre;
    uint8_t *blockTableDevicePre;
    AllocMem(&blockTableHostPre, &blockTableDevicePre, blockTableSizePre);
    ReadFile(dataPath + "/block_table_prefix.bin", blockTableHostPre, blockTableSizePre);
    ACL_CHECK(aclrtMemcpy(blockTableDevicePre, blockTableSizePre, blockTableHostPre, blockTableSizePre, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *blockTableHostSuf;
    uint8_t *blockTableDeviceSuf;
    AllocMem(&blockTableHostSuf, &blockTableDeviceSuf, blockTableSizeSuf);
    ReadFile(dataPath + "/block_table_suffix.bin", blockTableHostSuf, blockTableSizeSuf);
    ACL_CHECK(aclrtMemcpy(blockTableDeviceSuf, blockTableSizeSuf, blockTableHostSuf, blockTableSizeSuf, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *ExpRowSumDevicePre;
    uint8_t *ExpRowSumDeviceSuf;
    uint64_t RowSumSize = (uint64_t)numTokens * (uint64_t)numHeads * sizeof(float);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&ExpRowSumDevicePre), RowSumSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&ExpRowSumDeviceSuf), RowSumSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // Allocate matrices in device memory for workspace.
    // One base workspace block contains 65536 elements.
    uint64_t mm1OutSize = aicCoreNum * FAInferTiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * FAInferTiling::NUM3;
    uint64_t smOnlineOutSize = aicCoreNum * FAInferTiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(fp16_t)
                               * FAInferTiling::NUM3;
    uint64_t mm2OutSize = aicCoreNum * FAInferTiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * FAInferTiling::NUM3;
    uint64_t UpdateSize = aicCoreNum * FAInferTiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * FAInferTiling::NUM3;
    uint64_t workSpaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize;

    uint8_t *sDevice;
    ACL_CHECK(aclrtMalloc((void **)(&sDevice), mm1OutSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *pDevice;
    ACL_CHECK(aclrtMalloc((void **)(&pDevice), smOnlineOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *oTempDevice;
    ACL_CHECK(aclrtMalloc((void **)(&oTempDevice), mm2OutSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *oUpdateDevice;
    ACL_CHECK(aclrtMalloc((void **)(&oUpdateDevice), UpdateSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *oDevice{nullptr};
    ACL_CHECK(aclrtMalloc((void **)(&oDevice), qoSize * 2, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *tilingDevice;
    ACL_CHECK(aclrtMalloc((void **)(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // get tiling
    void *tilingHost = nullptr;
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    uint32_t blockDim = aicCoreNum;

    FAInferTiling::FAInfo faInfoPre;
    faInfoPre.numTokens = numTokens;
    faInfoPre.numHeads = numHeads;
    faInfoPre.embeddingSize = embeddingSize;
    faInfoPre.numBlocks = numBlocksPre;
    faInfoPre.blockSize = blockSize;
    faInfoPre.kvHeads = kvHeads;
    faInfoPre.batch = batch;
    faInfoPre.maskType = static_cast<FAInferTiling::MaskType>(maskTypePre);
    faInfoPre.qSeqlenList = reinterpret_cast<int64_t *>(qSeqHost);
    faInfoPre.kvSeqlenList = reinterpret_cast<int64_t *>(kvSeqHostPre);

    FAInferTiling::FAInfo faInfoSuf;
    faInfoSuf.numTokens = numTokens;
    faInfoSuf.numHeads = numHeads;
    faInfoSuf.embeddingSize = embeddingSize;
    faInfoSuf.numBlocks = numBlocksSuf;
    faInfoSuf.blockSize = blockSize;
    faInfoSuf.kvHeads = kvHeads;
    faInfoSuf.batch = batch;
    faInfoSuf.maskType = static_cast<FAInferTiling::MaskType>(maskType);
    faInfoSuf.qSeqlenList = reinterpret_cast<int64_t *>(qSeqHost);
    faInfoSuf.kvSeqlenList = reinterpret_cast<int64_t *>(kvSeqHostSuf);

    FATilingData faTilingData;
    FAInferTiling::GetFATilingParam(faInfoPre, blockDim, faTilingData);
    tilingHost = reinterpret_cast<void *>(&faTilingData);
    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    for (int i = 0; i < 1; i++) {
        if (dataType == "half") {
            FAInferTla<half><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevicePre, vDevicePre, maskDevice, blockTableDevicePre, oDevice, qSeqDevice, kvSeqDevicePre,
                sDevice, pDevice, oTempDevice, oUpdateDevice, tilingDevice, ExpRowSumDevicePre
            );
        } else {
            FAInferTla<bfloat16_t><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevicePre, vDevicePre, maskDevice, blockTableDevicePre, oDevice, qSeqDevice, kvSeqDevicePre,
                sDevice, pDevice, oTempDevice, oUpdateDevice, tilingDevice, ExpRowSumDevicePre
            );
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        // Copy the result from device to host
        vector<fp16_t> oHostHalf(qoSize / sizeof(fp16_t));
        vector<bfloat16> oHostBf16(qoSize / sizeof(bfloat16));
        if (dataType == "half") {
            ACL_CHECK(aclrtMemcpy(oHostHalf.data(), qoSize, oDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));
        } else if (dataType == "bf16") {
            ACL_CHECK(aclrtMemcpy(oHostBf16.data(), qoSize, oDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));
        }
        vector<float> rowSumHost(RowSumSize / sizeof(float));
        ACL_CHECK(aclrtMemcpy(rowSumHost.data(), RowSumSize, ExpRowSumDevicePre, RowSumSize, ACL_MEMCPY_DEVICE_TO_HOST));

        // Write oHost and rowSumHost to files for Python processing
        if (dataType == "half") {
            WriteFile(dataPath + "o_host_prefix_half.bin", oHostHalf.data(), qoSize);
        } else if (dataType == "bf16") {
            WriteFile(dataPath + "o_host_prefix_bf16.bin", oHostBf16.data(), qoSize);
        }
        WriteFile(dataPath + "row_sum_host_prefix.bin", rowSumHost.data(), RowSumSize);

        // Compute the golden result
        vector<float> goldenHost(qoSize / sizeof(fp16_t));
        const size_t goldenSize = qoSize * 2;
        ReadFile(dataPath + "/golden_prefix.bin", goldenHost.data(), goldenSize);

        vector<float> goldenHostRowSum(RowSumSize / sizeof(float));
        ReadFile(dataPath + "/golden_prefix_rowsum.bin", goldenHostRowSum.data(), RowSumSize);

        // Compare the result
        vector<uint64_t> errorIndices = (dataType == "half") ? golden::CompareData(oHostHalf, goldenHost, kvSeqlen)
                                                             : golden::CompareData(oHostBf16, goldenHost, kvSeqlen);
        if (errorIndices.empty()) {
            cout << "Compare success O Prefix." << endl;
        } else {
            cerr << "Compare failed O Prefix. Error count: " << errorIndices.size() << endl;
        }

        vector<uint64_t> errorIndices_rowsum = golden::CompareData(rowSumHost, goldenHostRowSum, kvSeqlen);
        if (errorIndices_rowsum.empty()) {
            cout << "Compare success RowSumPrefix." << endl;
        } else {
            cerr << "Compare failed RowSumPrefix. Error count: " << errorIndices_rowsum.size() << endl;
        }
    }

    FAInferTiling::GetFATilingParam(faInfoSuf, blockDim, faTilingData);
    tilingHost = reinterpret_cast<void *>(&faTilingData);
    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // Prepare FFTS address
    fftsAddr = 0;
    fftsLen = 0;
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    for (int i = 0; i < 1; i++) {
        if (dataType == "half") {
            FAInferTla<half><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDeviceSuf, vDeviceSuf, maskDevice, blockTableDeviceSuf, oDevice, qSeqDevice, kvSeqDeviceSuf,
                sDevice, pDevice, oTempDevice, oUpdateDevice, tilingDevice, ExpRowSumDeviceSuf
            );
        } else {
            FAInferTla<bfloat16_t><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDeviceSuf, vDeviceSuf, maskDevice, blockTableDevicePre, oDevice, qSeqDevice, kvSeqDeviceSuf,
                sDevice, pDevice, oTempDevice, oUpdateDevice, tilingDevice, ExpRowSumDeviceSuf
            );
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        // Copy the result from device to host
        vector<fp16_t> oHostHalf(qoSize / sizeof(fp16_t));
        vector<bfloat16> oHostBf16(qoSize / sizeof(bfloat16));
        if (dataType == "half") {
            ACL_CHECK(aclrtMemcpy(oHostHalf.data(), qoSize, oDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));
        } else if (dataType == "bf16") {
            ACL_CHECK(aclrtMemcpy(oHostBf16.data(), qoSize, oDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));
        }

        vector<float> rowSumHost(RowSumSize / sizeof(float));
        ACL_CHECK(aclrtMemcpy(rowSumHost.data(), RowSumSize, ExpRowSumDeviceSuf, RowSumSize, ACL_MEMCPY_DEVICE_TO_HOST));

        // Write oHost and rowSumHost to files for Python processing
        if (dataType == "half") {
            WriteFile(dataPath + "o_host_suffix_half.bin", oHostHalf.data(), qoSize);
        } else if (dataType == "bf16") {
            WriteFile(dataPath + "o_host_suffix_bf16.bin", oHostBf16.data(), qoSize);
        }
        WriteFile(dataPath + "row_sum_host_suffix.bin", rowSumHost.data(), RowSumSize);

        // Compute the golden result
        vector<float> goldenHost(qoSize / sizeof(fp16_t));
        const size_t goldenSize = qoSize * 2;
        ReadFile(dataPath + "/golden_suffix.bin", goldenHost.data(), goldenSize);

        vector<float> goldenHostRowSum(RowSumSize / sizeof(float));
        ReadFile(dataPath + "/golden_suffix_rowsum.bin", goldenHostRowSum.data(), RowSumSize);

        // Compare the result
        vector<uint64_t> errorIndices = (dataType == "half") ? golden::CompareData(oHostHalf, goldenHost, kvSeqlen)
                                                             : golden::CompareData(oHostBf16, goldenHost, kvSeqlen);
        if (errorIndices.empty()) {
            cout << "Compare success O Suffix." << endl;
        } else {
            cerr << "Compare failed O Suffix. Error count: " << errorIndices.size() << endl;
        }

        vector<uint64_t> errorIndices_rowsum = golden::CompareData(rowSumHost, goldenHostRowSum, kvSeqlen);
        if (errorIndices_rowsum.empty()) {
            cout << "Compare success RowSumSuffix." << endl;
        } else {
            cerr << "Compare failed RowSumSuffix. Error count: " << errorIndices_rowsum.size() << endl;
        }
    }

    // Free host memory allocations.
    FreeMem(qSeqHost, qSeqDevice);
    FreeMem(kvSeqHostPre, kvSeqDevicePre);
    FreeMem(kvSeqHostSuf, kvSeqDeviceSuf);
    FreeMem(qHost, qDevice);
    FreeMem(kHostPre, kDevicePre);
    FreeMem(kHostSuf, kDeviceSuf);
    FreeMem(vHostPre, vDevicePre);
    FreeMem(vHostSuf, vDeviceSuf);
    if (maskType == 1) {
        FreeMem(maskHost, maskDevice);
    }
    FreeMem(blockTableHostPre, blockTableDevicePre);
    FreeMem(blockTableHostSuf, blockTableDeviceSuf);
    aclrtFree(oDevice);
    aclrtFree(tilingDevice);
    aclrtFree(sDevice);
    aclrtFree(pDevice);
    aclrtFree(oTempDevice);
    aclrtFree(oUpdateDevice);
    aclrtFree(ExpRowSumDevicePre);
    aclrtFree(ExpRowSumDeviceSuf);
    aclrtFreeHost(tilingHost);
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