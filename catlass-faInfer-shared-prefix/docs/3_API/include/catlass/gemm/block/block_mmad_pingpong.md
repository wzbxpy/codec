# Block Mmad Pingpong
>
> [代码位置](../../../../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)

## 功能说明

[BlockMmad](./block_mmad.md#blockmmad)的偏特化实现，block层级mmad计算，不做bias计算，非异步计算，非TLA实现。

## 调度策略

```
// Now ENABLE_UNIT_FLAG_ must be false when intput element is int8
template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2Pingpong : public MmadAtlasA2  {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};
```

当ENABLE_UNIT_FLAG_为`True`时使能`L0C`同时搬出和写入，提高流水并行度。

## 调用示例

### Block组装

参考[basic_matmul](../../../../../../examples/00_basic_matmul/basic_matmul.cpp)

```
constexpr bool enableUnitFlag = true;
using MmadDispatchPolicy = Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
using L1TileShape = GemmShape<128, 256, 256>;
using L0TileShape = GemmShape<128, 256, 64>;
using AType = Gemm::GemmType<half, LayoutA>;
using BType = Gemm::GemmType<half, LayoutB>;
using CType = Gemm::GemmType<half, LayoutC>;
```

```
using BlockMmad = Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
```

### Block实例化

参考[basic_matmul](../../../../../../include/catlass/gemm/kernel/basic_matmul.hpp)，在`kernel`代码的`void operator()<AscendC::AIC>`函数中：

```
Arch::Resource<ArchTag> resource;
BlockMmad blockMmad(resource);
```

### Block执行

参考[basic_matmul](../../../../../../include/catlass/gemm/kernel/basic_matmul.hpp)，在`kernel`代码的`void operator()<AscendC::AIC>`函数中：

```
blockMmad(gmA[gmOffsetA],       // A矩阵的block块在GM上起始地址
        params.layoutA,         // A矩阵在GM上的layout
        gmB[gmOffsetB],         // B矩阵的block块在GM上起始地址
        params.layoutB,         // B矩阵在GM上的layout
        gmC[gmOffsetC],         // C矩阵的block块在GM上起始地址
        params.layoutC,         // C矩阵在GM上的layout
        actualBlockShape);      // block块的实际shape
```

## 约束说明

- 模板参数`BiasType_`没有实际使用，不支持bias计算
