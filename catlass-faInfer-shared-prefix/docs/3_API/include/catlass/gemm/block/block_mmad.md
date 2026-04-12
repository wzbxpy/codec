# Block Mmad基础模板
>
> [代码位置](../../../../../../include/catlass/gemm/block/block_mmad.hpp)

[TOC]

## BlockMmad

### 功能说明

block层级mmad计算，非TLA实现，实际计算由偏特化模板承载，通过[DisPatchPolicy](../../../../../../include/catlass/gemm/dispatch_policy.hpp)等模板参数命中偏特化分支。

### 模板说明

```
template <
    class DispatchPolicy,   // 所用DisPatchPolicy
    class L1TileShape,      // L1基本块
    class L0TileShape,      // L0基本块
    class AType,            // 封装了A矩阵的数据类型和排布信息
    class BType,            // 封装了B矩阵的数据类型和排布信息
    class CType,            // 封装了C矩阵的数据类型和排布信息
    class BiasType = void,  // 封装了Bias的数据类型和排布信息
    class TileCopy = Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>, // Tile层级搬运
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>         // Tile层级mmad计算
>
struct BlockMmad {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};
```

注意到模板参数中，BiasType默认使用`void`，TileCopy默认使用[Gemm::Tile::TileCopy](../../../../../../docs/3_API/include/catlass/gemm/tile/tile_copy.md)，TileMmad默认使用[Gemm::Tile::TileMmad](../../../../../../docs/3_API/include/catlass/gemm/tile/tile_mmad.md)

## BlockMmadTla

（待完善）

## BlockGemm

（待完善）

## BlockMmadAiv

（待完善）
