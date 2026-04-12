# Gemm Identity Block Swizzle
>
> [代码位置](../../../../../../../include/catlass/gemm/block/block_swizzle.hpp)

## 功能说明

Swizzle策略决定了AI Core对基本任务块的分配关系和计算顺序，详见[Swizzle解释](../../../../../../../docs/2_Design/01_kernel_design/02_swizzle.md)。`GemmIdentityBlockSwizzle`策略对结果矩阵C在$M$、$N$方向切基本块，根据$SwizzleOffset$和$SwizzleDirection$决定基本块排布顺序，并在排布上按照物理核序号依次分配基本块。

如下图示例，AI Core共20个，基本块验箭头方向依次分配给AI Core。箭头方向则由$SwizzleOffset = 1$和$SwizzleDirection = 0$来决定：

<img src="https://raw.gitcode.com/user-images/assets/7801479/bc97a077-d2e2-4abd-8f31-55316b1e0906/image.png" width="60%">

## 常用Methods说明 

| 返回类型   | 函数名 | 入参 |功能|
| :------- | :------: |:------: |:------: |
| -  |  GemmIdentityBlockSwizzle  |GemmCoord const &problemShape_, MatrixCoord const &tileMN_  |构造函数 |
| -  |  GemmIdentityBlockSwizzle  |GemmCoord const &problemShape_, MatrixCoord const &tileMN_, MatrixCoord const &loopsMN_  |构造函数 |
| void |  Update  |GemmCoord const &problemShape_, MatrixCoord const &tileMN_  |更新`problemShape`、`tileMN`、`loopsMN` |
| void |  Update  |GemmCoord const &problemShape_, MatrixCoord const &tileMN_, MatrixCoord const &loopsMN_  |更新`problemShape`、`tileMN`、`loopsMN` |
| uint32_t |  GetCoreLoops  |-  |返回基本块个数（M、N维度切块数乘积） |
| uint32_t |  GetBatchIdx  |uint32_t taskIdx  |计算输入基本块idx对应的基本块所属batch编号 |
| GemmCoord |  GetBlockCoord  |uint32_t taskIdx  |计算输入基本块idx对应的基本块在各维度上的坐标`blockCoord` |
| GemmCoord |  GetActualBlockShape  |GemmCoord blockCoord  |根据输入的基本块坐标`blockCoord`，返回基本块的实际shape |

## 调用示例

### Block组装

参考[basic_matmul](../../../../../../../examples/00_basic_matmul/basic_matmul.cpp)

```
// Swizzle offset is 3 and direction is 0.
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

// kernel level
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
```

### Matmul的kernel使用

参考[basic_matmul](../../../../../../../include/catlass/gemm/kernel/basic_matmul.hpp)，在`kernel`代码的`void operator()<AscendC::AIC>`函数中。

实例化BlockScheduler

```
BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
```

获取基本块总数

```
uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
```

基本块遍历中，获取当前 `block`的block坐标和实际shape

```
for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        // Compute block location
        GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
        GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
        ...
}
```

### GroupMatmul的kernel使用

参考[grouped_matmul_slice_m_per_token_dequant_multistage_workspace](../../../../../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp)，在`kernel`代码的`void operator()<AscendC::AIC>`函数中。

实例化BlockScheduler

```
BlockScheduler blockScheduler;
```

group的遍历中，根据每个group的shape更新blockScheduler，并获取单个group的基本块总数

```
for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
    ...
    blockScheduler.Update(inGroupProblemShape, L1TileShape::ToCoordMN());
    uint32_t coreLoops = blockScheduler.GetCoreLoops();
    ...
}
```

单个group的基本块遍历中，获取当前 `block`的block坐标和实际shape

```
for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
        GemmCoord blockCoordMNK = blockScheduler.GetBlockCoord(loopIdx);
        GemmCoord actualBlockShapeMNK = blockScheduler.GetActualBlockShape(blockCoordMNK);
        ...
}
```
