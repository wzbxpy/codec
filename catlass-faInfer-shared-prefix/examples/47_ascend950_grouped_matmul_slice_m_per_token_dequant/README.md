# 950_grouped_matmul_slice_m_per_token_dequant Example Readme

## 代码组织

```
├── 47_ascend950_grouped_matmul_slice_m_per_token_dequant
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── grouped_matmul_slice_m_per_token_dequant_tla.cpp # 主文件
```

## 功能介绍

该CV融合算子实现了在ascend950上的分组矩阵乘法（Grouped Matmul）与反量化（dequant）操作。主要解决了高效执行分组、切片（M轴）矩阵乘法，并融合per-token和per-channel反量化操作的需求。

## 方案概述

1. 新增[GroupedMatmulSliceMPerTokenTla模板类（Kernel）](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_tla.hpp)，通过[BlockMmad](../../include/catlass/gemm/block/block_mmad_pingpong_tla.hpp)、[Epilogue](../../include/catlass/epilogue/block/block_epilogue_per_token_dequant.hpp)和[Scheduler](../../include/catlass/gemm/block/block_scheduler_aswt.hpp)，支持对一组groupCount的矩阵进行反量化计算。
2. 为Ascend950新增了Epilogue模板[EpilogueAscend950PerTokenDequantTla](../../include/catlass/epilogue/block/block_epilogue_per_token_dequant.hpp)，实现了从GM加载量化系数、在同一UB执行反量化计算。
3. 为Ascend950新增了高性能反量化计算的Tile模板[TilePerTokenDequant](../../include/catlass/epilogue/tile/tile_pertoken_dequant.hpp)。

## 参数说明

|名称/Name|类型/Class|数据类型/Dtype|维度/Dims|格式/Format|描述/Description|
|---|---|---|---|---|---|
|matA|inTensor|int8|[m, k]|ND|左矩阵|
|matB|inTensor|int8|[groupCount, n, k]|ND|右矩阵，支持转置|
|groupList|inTensor|int32|[groupCount]|ND|m轴方向分组大小，累加和列表|
|scale|inTensor|bf16/fp16/fp32|[groupCount, n]|ND|perChannel量化系数|
|perToken|inTensor|bf16/fp16/fp32|[m]|ND|perToken量化系数|
|matD|outTensor|bf16/fp16/fp32|[m, n]|ND|输出矩阵|

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/1_Practice/01_quick_start.md#编译执行)，本用例为Ascend950算子，编译时需加-DCATLASS_ARCH=3510
- 执行算子

```
# 编译指定用例
bash scripts/build.sh 47_ascend950_grouped_matmul_slice_m_per_token_dequant -DCATLASS_ARCH=3510
cd output/bin
# 可执行文件名|group数量|矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./47_ascend950_grouped_matmul_slice_m_per_token_dequant 128 512 1024 2048 0
```

执行结果如下，说明精度比对成功。

```
Compare success.
```
