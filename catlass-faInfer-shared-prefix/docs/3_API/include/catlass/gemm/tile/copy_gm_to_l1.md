# Copy Gm To L1基础模板
>
> [代码位置](../../../../../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)

[TOC]

## CopyGmToL1

### 功能说明

### 原型

- 结构体模板

```
template <
    class ArchTag,          // 架构标签
    class GmType,           // GM上操作数的Gemm类型
    class L1Type = void     // L1上操作数的Gemm类型
>
struct CopyGmToL1
```

- 偏特化实现

|template| ArchTag  | GmType | L1Type |
| :------ | :------: |------: |------: |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::RowMajor>  | Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1> |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::RowMajor>  |  Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1> |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::ColumnMajor>  | Gemm::GemmType<Element, layout::nN, AscendC::TPosition::A1> |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::ColumnMajor>  |  Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1> |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::ColumnMajor>  |  Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1> |
|<class ArchTag, class Element>|  ArchTag  | Gemm::GemmType<Element, layout::VectorLayout>   |  Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1> |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::NDC1HWC0, AscendC::TPosition::GM>  |  - |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::KDC1KHKWN1N0C0, AscendC::TPosition::GM>  |  - |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::ColumnMajor>  |  Gemm::GemmType<Element, layout::nN, AscendC::TPosition::B1> |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::RowMajor>  |  Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1> |
|\<class Element\>|  Arch::AtlasA2  |  Gemm::GemmType<Element, layout::RowMajor>  |  - |
|\<class Element\>|  Arch::AtlasA2  |  Gemm::GemmType<Element, layout::ColumnMajor>  |  - |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::zN>  |  - |
|<class ArchTag, class Element>|  ArchTag  |   Gemm::GemmType<Element, layout::nZ> |  - |
|\<class Element\>|  Arch::AtlasA2  |  Gemm::GemmType<Element, layout::PaddingRowMajor>  |  - |
|\<class Element\>|  Arch::AtlasA2  |   Gemm::GemmType<Element, layout::PaddingColumnMajor>  |  - |
|\<class Element\>|  Arch::AtlasA2  |  Gemm::GemmType<Element, layout::RowMajor>  |  Gemm::GemmType<Element, layout::RowMajor, AscendC::TPosition::A1> |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::GM>  |  Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1> |

- 调用

```
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,     // 目的操作数LocalTensor
    AscendC::GlobalTensor<Element> const &srcTensor,    // 源操作数LocalTensor
    LayoutDst const &layoutDst,         // 目的操作数layout
    LayoutSrc const &layoutSrc          // 源操作数layout
)
```

## CopyGmToL1IntervalDataCopy

### 功能说明

### 原型

- 结构体模板

```
template <
    class ArchTag,          // 架构标签
    class GmType,           // GM上操作数的Gemm类型
    class L1Type = void     // L1上操作数的Gemm类型
>
struct CopyGmToL1IntervalDataCopy
```

- 偏特化实现

|template| ArchTag  | GmType | L1Type |
| :------ | :------: |------: |------: |
|-|  Arch::AtlasA2  | Gemm::GemmType<half, layout::RowMajor>| -|
|-|  Arch::AtlasA2  | Gemm::GemmType<half, layout::PaddingRowMajor>| -|
|-|  Arch::AtlasA2  | Gemm::GemmType<half, layout::ColumnMajor>| -|
|-|  Arch::AtlasA2  | Gemm::GemmType<half, layout::PaddingColumnMajor>| -|

## CopyGmToL1GMMPTD

### 功能说明

### 原型

- 结构体模板

```
template <
    class ArchTag,          // 架构标签
    class GmType,           // GM上操作数的Gemm类型
    class L1Type = void     // L1上操作数的Gemm类型
>
struct CopyGmToL1GMMPTD
```

- 偏特化实现

|template| ArchTag  | GmType | L1Type |
| :------ | :------: |------: |------: |
|\<class Element\>|  Arch::AtlasA2  | Gemm::GemmType<Element, layout::RowMajor>| -|

## CopyGmToL1DynamicOptimized

### 功能说明

### 原型

- 结构体模板

```
template <
    class ArchTag,          // 架构标签
    class GmType,           // GM上操作数的Gemm类型
    class L1Type = void     // L1上操作数的Gemm类型
>
struct CopyGmToL1DynamicOptimized
```

- 偏特化实现

|template| ArchTag  | GmType | L1Type |
| :------ | :------: |------: |------: |
|\<class Element\>|  Arch::AtlasA2  | Gemm::GemmType<Element, layout::RowMajor>| -|
|\<class Element\>|  Arch::AtlasA2  | Gemm::GemmType<Element, layout::ColumnMajor>| -|
|\<class Element\>|  Arch::AtlasA2  |  Gemm::GemmType<Element, layout::zN>| -|
|\<class Element\>|  Arch::AtlasA2  |  Gemm::GemmType<Element, layout::nZ>| -|
|\<class Element\>|  Arch::AtlasA2  |  Gemm::GemmType<Element, layout::PaddingRowMajor>| -|
|\<class Element\>|  Arch::AtlasA2  |  Gemm::GemmType<Element, layout::PaddingColumnMajor>| -|

## TileCopyTla

### 功能说明

### 原型

- 结构体模板

```
template <
    class ElementSrc,   // 源操作数的数据类型
    class ElementDst,   // 目的操作数的数据类型
    class LayoutSrc,    // 操作数的layout
    class LayoutDst,    // 目的操作数的layout
    class CoordSrc,     // 源操作数在tensor中的坐标
    class CoordDst      // 目的操作数在tensor中的坐标
    >
struct TileCopyTla<
    Arch::AtlasA2,                                  // 架构标签
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>,
        LayoutSrc,
        CoordSrc,
        AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>,   // 源操作数的tensor结构
        LayoutDst, 
        CoordDst, 
        AscendC::TPosition::A1>,                    // 目的操作数的tensor结构
    std::enable_if_t<cond0 && cond1>              // 判断条件，cond0和cond1见下列偏特化实现
    >
```

- 偏特化实现

| cond0 | cond1 |
|------: |------: |
| tla::detail::isRowMajor\<LayoutSrc\>::value|tla::detail::iszN<ElementDst, LayoutDst>::value|
| tla::detail::isColumnMajor\<LayoutSrc\>::value|tla::detail::isnZ<ElementDst, LayoutDst>::value|
| tla::detail::iszN\<LayoutSrc\>::value|tla::detail::iszN<ElementDst, LayoutDst>::value|
| tla::detail::isnZ\<LayoutSrc\>::value|tla::detail::isnZ<ElementDst, LayoutDst>::value|

## TileCopyTlaExt

### 功能说明

### 原型

- 结构体模板

```
template <
    class ElementSrc,   // 源操作数的数据类型
    class ElementDst,   // 目的操作数的数据类型
    class LayoutSrc,    // 操作数的layout
    class LayoutDst,    // 目的操作数的layout
    class CoordSrc,     // 源操作数在tensor中的坐标
    class CoordDst      // 目的操作数在tensor中的坐标
    >
struct TileCopyTla<
    Arch::AtlasA2,                                  // 架构标签
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>,
        LayoutSrc,
        CoordSrc,
        AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>,   // 源操作数的tensor结构
        LayoutDst, 
        CoordDst, 
        AscendC::TPosition::A1>,                    // 目的操作数的tensor结构
    cond0,          // 见下面偏特化实现
    cond1           // 见下面偏特化实现
    >
```

- 偏特化实现

| cond0 | cond1 |
|------: |------: |
| layout::RowMajor|layout::zN|
| layout::PaddingRowMajor|layout::zN|
| layout::ColumnMajor|layout::nZ|
| layout::PaddingColumnMajor|layout::nZ|
| layout::zN|layout::zN|
| layout::nZ|layout::nZ|
