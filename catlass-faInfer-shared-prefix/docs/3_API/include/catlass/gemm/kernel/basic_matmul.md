# Basic Matmul
>
> [代码位置](../../../../../../include/catlass/gemm/kernel/basic_matmul.hpp)

## 功能说明

基础矩阵乘，cube算子，无AIV计算，非TLA实现。

## 类模板概述

- 模板入参
    - `class BlockMmad_`：blockMmad类，矩阵乘组件
    - `class BlockEpilogue_`：blockEpilogue类，后处理组件，实际未使用
    - `class BlockScheduler_`：blockScheduler类，仅支持[Gemm::Block::GemmIdentityBlockSwizzle](../block/block_swizzle/block_swizzle.md)
- Params：

```
struct Params {
    GemmCoord problemShape;     //用例shape
    GM_ADDR ptrA;               //输入matA的GM起始地址
    LayoutA layoutA;            //输入matA的layout
    GM_ADDR ptrB;               //输入matB的GM起始地址
    LayoutB layoutB;            //输入matB的layout
    GM_ADDR ptrC;               //输出matC的GM起始地址
    LayoutC layoutC;            //输出matC的layout
...
}
```

- Arguments：

```
struct Arguments {
    GemmCoord problemShape;     //用例shape
    GM_ADDR ptrA;               //输入matA的GM起始地址
    GM_ADDR ptrB;               //输入matB的GM起始地址
    GM_ADDR ptrC;               //输出matC的GM起始地址
};
```

## 调用示例

kernel组装

```
using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
using BlockEpilogue = void;
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

// kernel level
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
```

## 约束说明

该kernel在void operator()\<AscendC::AIC\>核函数中，调用`blockMmad`的方式不涉及异步和Preload，故仅支持[block_mmad_pingpong](../block/block_mmad_pingpong.md)等简单blockMmad组件
