# Tile Mmad基础模板
>
> [代码位置](../../../../../../include/catlass/gemm/tile/tile_mmad.hpp)

[TOC]

## TileMmad

### 功能说明

使用[AscendC::mmad基础API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0249.html)完成矩阵乘加（C += A * B）操作。矩阵ABC分别为L0A/L0B/L0C中的数据。ABC矩阵的数据排布格式分别为zZ、nZ、zN。非TLA实现。

### 原型

- 结构体模板

```
template <
    class ArchTag_,         // 架构标签
    class AType_,           // A矩阵操作数的Gemm类型
    class BType_,           // B矩阵操作数的Gemm类型
    class BiasType_         // Bias操作数的Gemm类型
>
struct TileMmad
```

- 不传入bias的调用

```
void operator() (
    AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,  // L0C上结果矩阵
    AscendC::LocalTensor<ElementA> const &l0ATensor,            // L0A上左矩阵
    AscendC::LocalTensor<ElementB> const &l0BTensor,            // L0B上右矩阵
    uint32_t m,             // tile块实际shape的M对齐
    uint32_t n,             // tile块实际shape的N对齐
    uint32_t k,             // tile块实际shape的K对齐
    bool initC = true,      // 是否初始化L0C，True为直接覆盖，False为原子累加
    uint8_t unitFlag = 0    // 是否开始uniFlag，True则mmad计算和L0C2GM搬运并行
    )
```

- 传入bias的调用

```
void operator() (
    AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,  // L0C上结果矩阵
    AscendC::LocalTensor<ElementA> const &l0ATensor,            // L0A上左矩阵
    AscendC::LocalTensor<ElementB> const &l0BTensor,            // L0B上右矩阵
    AscendC::LocalTensor<ElementAccumulator> const &l0BiasTensor,   // BT上bias
    uint32_t m,             // tile块实际shape的M对齐
    uint32_t n,             // tile块实际shape的N对齐
    uint32_t k,             // tile块实际shape的K对齐
    bool initC = true,      // 是否初始化L0C，True为直接覆盖，False为原子累加
    uint8_t unitFlag = 0    // 是否开始uniFlag，True则mmad计算和L0C2GM搬运并行
    )
```

### 调用示例

参考[block_mmad_pingpong](../../../../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)中使用方法。 

## TileMmadTla

### 功能说明

使用[AscendC::mmad基础API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0249.html)完成矩阵乘加（C += A * B）操作。矩阵ABC分别为L0A/L0B/L0C中的数据。ABC矩阵的数据排布格式分别为zZ、nZ、zN。TLA实现，不支持bias。

### 原型

- 结构体模板

```
template <
    class ArchTag_,         // 架构标签
    class ElementA,         // A矩阵操作数的element类型
    class LayoutTagL1A      // A矩阵操作数的layout标签
>
struct TileMmadTla
```

- 调用（不支持bias）

```
void operator() (
    TensorC const &l0CTensor,   // L0C上结果矩阵tensor
    TensorA const &l0ATensor,   // L0A上左矩阵tensor
    TensorB const &l0BTensor,   // L0B上右矩阵tensor
    uint32_t m,                 // tile块实际shape的M对齐
    uint32_t n,                 // tile块实际shape的N对齐
    uint32_t k,                 // tile块实际shape的K对齐
    bool initC = true,          // 是否初始化L0C，True为直接覆盖，False为原子累加
    uint8_t unitFlag = 0        // 是否开始uniFlag，True则mmad计算和L0C2GM搬运并行
    )
```

### 调用示例

参考[block_mmad_pingpong_tla](../../../../../../include/catlass/gemm/block/block_mmad_pingpong_tla.hpp)中使用方法。 
