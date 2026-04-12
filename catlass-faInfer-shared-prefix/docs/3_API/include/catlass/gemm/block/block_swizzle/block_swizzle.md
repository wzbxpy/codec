# Gemm/Block Sizzle类模板概述
>
> [代码位置](../../../../../../../include/catlass/gemm/block/block_swizzle.hpp)

## 功能说明

Swizzle策略决定了AI Core对基本任务块的分配关系和计算顺序，详见[Swizzle解释](../../../../../../../docs/2_Design/01_kernel_design/02_swizzle.md)

## 常用Methods说明 

- 构造函数。基于实际shape和切分tile等参数，计算在相关维度的基本块切分数。
- 函数`void Update`。更新在相关维度的基本块切分数。
- 函数`uint32_t GetCoreLoops`。计算并返回基本块切分总数。
- 函数`GemmCoord GetBlockCoord`。根据输入的基本块序号，计算并返回基本块在各维度上的坐标`blockCoord`（坐标为block粒度，不是element粒度）。
- 函数`GemmCoord GetActualBlockShape`。根据输入的基本块坐标`blockCoord`，返回基本块的实际shape（element粒度）。

## 具体Swizzle策略

| 组件                        | 描述 |
| :----------------------------------------------------------- | :------: |
| [GemmIdentityBlockSwizzle](./GemmIdentityBlockSwizzle.md)    |   Gemm算子基础swizzle策略  |
