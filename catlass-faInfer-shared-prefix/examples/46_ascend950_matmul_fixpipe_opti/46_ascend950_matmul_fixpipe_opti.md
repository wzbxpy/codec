# CATLASS Matmul Fixpipe Opti 样例介绍

## 原型设计
|名称/Name|类型/Class|数据类型/Dtype|维度/Dims|格式/Format|描述/Description|
|---|---|---|---|---|---|
|matA|inTensor|half|[m, k]|ND|左矩阵|
|matB|inTensor|half|[n, k]|ND|右矩阵|
|matC|outTensor|float|[m, n]|ND|输出矩阵|

## 样例实现
CATLASS [`46_ascend950_matmul_fixpipe_opti`样例](./README.md)算子是基于CATLASS Gemm API实现的昇腾亲和Matmul算子，针对Fixpipe非对齐搬出场景优化设计，关键算子件包括以下几部分:
 - **Example组装**：[matmul_fixpipe_opti.cpp](./matmul_fixpipe_opti.cpp)
 - **Kernel实现**：
   - 主Kernel文件：[matmul_mix_fixpipe_opti.hpp](../../include/catlass/gemm/kernel/matmul_mix_fixpipe_opti.hpp)

 - **Block组件**，包含：
    - 通用的mmad组件[block_mmad_pingpong_tla.hpp](../../include/catlass/gemm/block/block_mmad_pingpong_tla.hpp)
    - 针对Fixpipe优化的后处理组件[block_epilogue_fixpipe.hpp](../../include/catlass/epilogue/block/block_epilogue_fixpipe.hpp)；
    - ASWT分核策略[BlockSchedulerAswt](../../include/catlass/gemm/block/block_scheduler_aswt.hpp)

## 方案设计
当前Fixpipe存在N轴非对齐写入性能不佳的问题，在K值较小、MN值较大的Matmul场景中，易引发Fixpipe Bound。针对上述问题，Fixpipe可基于Ascend 950硬件新特性（Ascend 950新增了L0C Buffer到UB的数据通路），通过使能dualDstCtrl，将Cube核L0C Buffer中的结果数据拆分为两路，并行写入两个Vector核（一个Cube核对应两个Vector核）的专属UB中，再使用DataCopyPad基础指令将UB中的数据搬运到Global Memory中。每个Vector核的UB支持独立开启Double Buffer以实现数据读写的流水线重叠，在向Global Memory传输数据的同时，持续接收来自L0C Buffer的数据，有效提升数据吞吐效率。

<img src="../../docs/figures/fixpipe-opti.png" width="50%">

## 性能收益
在使用相同的tileShape和scheduler策略情况下，本算子(matmul_fixpipe_opti)相较于使用Fixpipe直接搬出Global Memory的传统算子方法 (matmul_fixpipe)，性能对比及收益如下表。
|[M, N, K]|matmul_fixpipe|matmul_fixpipe_opti|加速比|备注|
|---------|--------------|-------------------|-------|----|
|[567, 488, 399]| 6.53us | 6.34us | 1.03  | MN非对齐|
|[1226, 1557, 399]| 15.06us | 12.05us | 1.25 | MN非对齐 |
|[2058, 2038, 256]| 20.69us | 12.64us | 1.64 | MN非对齐 |
|[2048, 2048, 256]| 11.97us | 12.09us | 0.99 | MN对齐 |
|[2058, 2048, 256]| 13.81us | 12.38us | 1.12 | M非对齐 |
|[2048, 2038, 256]| 19.09us | 12.35us | 1.55 | N非对齐 |


由表格结果可知，在M、N非对齐场景，特别是在N轴非对齐的情况下，matmul_fixpipe_opti的性能相较于基础版本的matmul_fixpipe实现了正向提升；且M、N值越大，K值越小，性能提升越显著。

### 说明
- L1TileShape: [256, 256, 128]
- L0TileShape: [256, 256, 64]
- scheduler策略：[ASWT](../../docs/2_Design/01_kernel_design/05_aswt.md)
- 测试环境说明：NPU型号为Ascend 950，CANN包版本为9.0.0。