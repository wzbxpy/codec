# Block Epilogue基础模板
>
> [代码位置](../../../../../../include/catlass/epilogue/block/block_epilogue.hpp)

[TOC]

## BlockEpilogue

### 功能说明

block层级尾处理，通过[DisPatchPolicy](../../../../../../include/catlass/gemm/dispatch_policy.hpp)等模板参数命中偏特化分支。

### 模板说明

```
template <
    class DispatchPolicy,   // 所用DisPatchPolicy
    class... Args           // 其他模板参数
>
struct BlockEpilogue
```
