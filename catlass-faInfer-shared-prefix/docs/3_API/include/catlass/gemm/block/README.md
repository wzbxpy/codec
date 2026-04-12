# Gemm/Block 类模板概述

## API 清单

### blockMmad清单

| 组件                       | 描述 |
| :----------------------------------------------------------- | :------: |
| [block_mmad](./block_mmad.md#blockmmad)     |   基础模板，包含BlockMmad  |
| [block_mmad_pingpong](./block_mmad_pingpong.md)     |   BlockMmad偏特化实现，pingpong矩阵乘  |

### swizzle清单

| 组件                        | 描述 |
| :----------------------------------------------------------- | :------: |
| [block_swizzle](./block_swizzle/block_swizzle.md)    |   swizzle基本方法介绍  |
| [GemmIdentityBlockSwizzle](./block_swizzle/GemmIdentityBlockSwizzle.md)    |   Gemm算子基础swizzle策略  |

## API 拆解

### blockMmad
>
>封装了Block层的mmad计算（矩阵乘计算），对应于昇腾NPU的一个AI Core上的计算。 通过模板参数，BlockMmad_接收矩阵计算中的Shape（特征尺寸）、Layout（数据排布，如行优先、列优先排布）与DType（数据类型）方面的信息。

命名空间为`Catlass::Gemm::Block`，包含如下核心成员：

| 类型         | 名称 |功能|
| :----------------------------------------------------------- | :------: |------: |
| 构造函数    |   BlockMmad()  |通常包含初始化buffer、Event ID，插入流水间同步setFlag|
| 析构函数    |   ~BlockMmad()  |通常包含插入流水间同步waitFlag|
| 函数    |   void operator()  |执行一个block任务块的矩阵乘计算|
