# Tile  Copy基础模板
>
> [代码位置](../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

## 功能说明

集成tile数据在硬件上搬运的所有通路偏特化分支、及数据类型等成员，主要包含：

- `ElementA`，左矩阵数据类型
- `ElementB`，右矩阵数据类型
- `ElementAccumulator`，L0C上累加的数据类型
- [`CopyGmToL1A`](./copy_gm_to_l1.md)，GM到L1的A矩阵tile块搬运偏特化实现
- `CopyGmToL1B`，GM到L1的B矩阵tile块搬运偏特化实现
- `CopyL1ToL0A`，L1到L0A的A矩阵tile块搬运偏特化实现
- `CopyL1ToL0B`，L1到L0B的B矩阵tile块搬运偏特化实现
- `CopyL0CToGm`，L0C到GM的结果矩阵tile块搬运偏特化实现
- `CopyGmToL1Bias`，GM到L1的Bias矩阵tile块搬运偏特化实现
- `CopyL1ToBT`，L1到BT的Bias矩阵tile块搬运偏特化实现

## 模板清单

### TileCopy

- 功能说明：
- 模板参数：
- 使用示例：

### TileCopyWithPrologueDeqPerTensor

- 功能说明：
- 模板参数：
- 使用示例：
