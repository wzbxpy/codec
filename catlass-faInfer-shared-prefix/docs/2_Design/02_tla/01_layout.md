# TLA Layouts

本文介绍 TLA（Tensor Layout Abstraction）中的 `Layout`。

如果把 Tensor 看成“逻辑上的多维数组”，那么 `Layout` 负责回答以下问题：

- 一个逻辑坐标 `(i, j, ...)` 对应到哪一个线性地址。
- 这块 Tensor 在逻辑上有多大。
- 当底层存在分块、对齐或填充时，哪些位置是逻辑有效数据。

因此，`Layout` 可以理解为“逻辑坐标到内存地址的映射规则”。算法通常依赖这套规则访问数据，而不直接依赖底层物理排布。这样，同一段计算逻辑就可以适配普通 ND 布局、行优先、列优先以及 `zN`、`nZ` 等分形布局。

## 先建立三个基本概念

### 逻辑坐标 coord

`coord` 表示元素在 Tensor 逻辑空间中的位置，约定如下：

- 坐标从 0 开始计数。
- 坐标单位是“元素”，不是字节，也不是 tile 编号。
- `coord` 的 rank 必须与 Tensor 或 Layout 的逻辑维度一致。
- 即使底层采用 `zN`、`nZ` 这类嵌套布局，`coord` 仍然描述逻辑上的行列位置，例如 `(row, col)`。

例如，对一个逻辑形状为 `(8, 16)` 的矩阵，`coord = (2, 4)` 表示第 2 行、第 4 列的元素。它不关心这块数据在内存中是按行连续、按列连续，还是按分形块组织。

### 逻辑形状与内存布局

在 TLA 中，这两个概念被刻意分离：

- 逻辑形状：从使用者视角看，Tensor 有多少行、多少列。
- 内存布局：这些逻辑元素在内存中如何排布，跨一个维度移动时需要跳过多少位置。

`Layout` 的核心价值，就是把“逻辑上多大”和“内存里怎样排”同时表达清楚。

### Tail tile

当矩阵尺寸不是 tile 大小的整数倍时，边界 tile 往往只包含部分有效元素。这类边界 tile 通常称为 tail tile。

TLA 使用 `originShape` 表达“逻辑上实际有效的范围”。因此，用户通常不需要手工推导每个边缘 tile 的真实尺寸。

## 基础类型

### Tuple

TLA 以 [`tla::tuple`](../../../include/tla/tuple.hpp) 为基础。它与 `std::tuple` 的用途相似，都是表达定长元素序列；不同之处在于，TLA 对模板元编程和高性能场景做了定制。

### IntTuple

[`IntTuple`](../../../include/tla/int_tuple.hpp) 是 TLA 中最常用的基础概念之一。它可以是：

- 一个整数，例如 `int{2}`、`size_t{16}`。
- 一个编译期整数，例如 `Int<3>{}` 或别名 `_3`。
- 一个由以上元素递归组成的 tuple，例如 `make_tuple(int{2}, Int<3>{})`。

因此，`IntTuple` 既可以表示一维尺寸，也可以表示带层次结构的嵌套尺寸。

常用操作如下：

- `rank(IntTuple)`：返回元素个数。
- `get<I>(IntTuple)`：返回第 `I` 个元素。
- `depth(IntTuple)`：返回嵌套层数；普通整数的 `depth` 为 0。

`IntTuple` 不仅用于 `Layout`，也用于 `Shape`、`Stride` 等类型，定义见 [`include/tla/layout.hpp`](../../../include/tla/layout.hpp)。

## Layout 由什么组成

`Layout` 本质上由三个 `IntTuple` 组成：`Shape`、`Stride` 和 `OriginShape`。

| 字段 | 作用 | 关注点 |
| --- | --- | --- |
| `Shape` | 用于内存布局计算的尺寸描述 | 决定布局结构，不一定等于逻辑实际尺寸 |
| `Stride` | 各维度上的步长 | 决定坐标如何映射到线性地址 |
| `OriginShape` | Tensor 的逻辑实际尺寸 | 决定哪些元素在逻辑上有效 |

可以先把它们理解成：

- `Shape` 说明“内存按什么结构排”。
- `Stride` 说明“每跨一步跳多远”。
- `OriginShape` 说明“逻辑上到底有多少有效数据”。

这里最容易混淆的是 `Shape` 和 `OriginShape`。两者并不重复：

- `Shape` 面向布局计算，允许包含对齐、分块和填充后的结构。
- `OriginShape` 面向逻辑语义，只描述真实有效的数据范围。

![Origin_Shape-layout_1.png](https://raw.gitcode.com/user-images/assets/7631999/936388d2-81b6-400d-afe2-788eaf140f81/Origin_Shape-layout_1.png 'Origin_Shape-layout_1.png')

`OriginShape` 用于把“内存怎样排”与“逻辑上哪些数据有效”区分开。

- `Shape`：服务于布局计算，可能包含对齐、分块或填充后的尺寸。
- `OriginShape`：服务于逻辑语义，描述真实有效的数据范围。

例如，一个逻辑大小为 `100 x 100` 的矩阵采用 `zN` 布局时，可能出现：

- `originShape = (100, 100)`
- `shape = ((16, 7), (16, 7))`

原因是：

- `16 * 7 = 112`，说明底层内存按 `112 x 112` 的块化结构组织。
- 但逻辑上只有 `100 x 100` 是有效元素。

这也是 TLA 能自动处理 tail tile 的基础。用户在 block 层和 kernel 层通常只需要按 tile 编程，边界有效范围由 `originShape` 传递和裁剪，无需每一层都手动判断尾块。

## Layout 的常用接口

`Layout` 提供了一组与 `IntTuple` 风格一致的访问接口：

- `rank(Layout)`：布局的逻辑维度。
- `get<I>(Layout)`：取出第 `I` 个分量。
- `depth(Layout)`：布局的嵌套层数。
- `shape(Layout)`：返回 `Shape`。
- `stride(Layout)`：返回 `Stride`。
- `originShape(Layout)`：返回 `OriginShape`。

另外还提供递归版本的辅助接口，例如：

- `get<I0, I1, ..., IN>(x)`：逐层向下取子单元。
- `rank<I...>(x)`：查看某个子单元的 rank。
- `depth<I...>(x)`：查看某个子单元的 depth。
- `shape<I...>(x)`：查看某个子单元的 shape。
- `originShape<I...>(x)`：查看某个子单元的 origin shape。

## Layout 构造

`Layout` 支持静态整数、动态整数及其混合构造，也支持普通矩阵布局和 Ascend 常用内部布局。

在昇腾 CUBE 核内部，常见内部格式包括 `zN`、`nZ`、`zZ`、`nN`、`L0C` 等；在 GEMV、Scale、Bias 等场景中，也会使用一维 `VectorLayout`。

```c++
using namespace tla;

// 1. 直接给 shape 和 stride，originShape 由系统推导
Layout w2xh4 = MakeLayout(MakeShape(Int<2>{}, 4),
                          MakeStride(Int<12>{}, Int<1>{}));

// 2. 嵌套布局，originShape 隐式推导为 (16*2, 16*3) = (32, 48)
Layout w32xh48 = MakeLayout(MakeShape(MakeShape(16, 2), MakeShape(16, 3)),
                            MakeStride(MakeStride(16, 256), MakeStride(1, 512)));

// 3. 显式指定 originShape
Layout w2xh4_explicit = MakeLayout(MakeShape(Int<2>{}, 4),
                                   MakeStride(Int<12>{}, Int<1>{}),
                                   MakeShape(2, 4));

Layout w32xh48_explicit = MakeLayout(MakeShape(MakeShape(16, 2), MakeShape(16, 3)),
                                     MakeStride(MakeStride(16, 256), MakeStride(1, 512)),
                                     MakeShape(32, 48));

// 4. rank=2 时，也可以用 LayoutTag + (rows, cols) 构造
auto rm = MakeLayout<float, Catlass::layout::RowMajor>(2, 4);

// 5. 一维 VectorLayout
auto vec = MakeLayout(128);
```

其中：

- `MakeLayout` 返回 `Layout`。
- `MakeShape` 返回 `Shape`。
- `MakeStride` 返回 `Stride`。

上面的布局可写成：

```text
w2xh4   : (_2, 4):(_12, _1)
w32xh48 : ((16, 2), (16, 3)):((16, 256), (1, 512))
```

读法如下：

- 前一部分是 `Shape`。
- 后一部分是 `Stride`。
- 如果省略 `OriginShape`，表示它可由 `Shape` 推导，或与逻辑尺寸一致。

## 从直观例子理解 Shape 与 Stride

### 2x3 行优先

```text
shape  = (2, 3)
stride = (3, 1)
```

含义是：

- 行维度前进一步，线性地址增加 3。
- 列维度前进一步，线性地址增加 1。

因此线性地址顺序为：

| 逻辑坐标 | 线性地址 |
| --- | --- |
| `(0, 0)` | `0` |
| `(0, 1)` | `1` |
| `(0, 2)` | `2` |
| `(1, 0)` | `3` |
| `(1, 1)` | `4` |
| `(1, 2)` | `5` |

### 2x3 列优先

```text
shape  = (2, 3)
stride = (1, 2)
```

含义是：

- 行维度前进一步，线性地址增加 1。
- 列维度前进一步，线性地址增加 2。

因此线性地址顺序为：

| 逻辑坐标 | 线性地址 |
| --- | --- |
| `(0, 0)` | `0` |
| `(1, 0)` | `1` |
| `(0, 1)` | `2` |
| `(1, 1)` | `3` |
| `(0, 2)` | `4` |
| `(1, 2)` | `5` |

### 以 `zN` 为例理解嵌套布局

示例布局：

```text
shape  = ((4, 2), (4, 3))
stride = ((4, 16), (1, 32))
```

可以理解为：

- 行方向先以 4 为一个内层块，再沿行方向重复 2 次。
- 列方向先以 4 为一个内层块，再沿列方向重复 3 次。
- 子块内部如何走、子块之间如何跳，分别由嵌套 `Stride` 给出。

关键点不在于记住每个数字，而在于理解：TLA 用嵌套 `Shape` 和 `Stride` 显式表达分块布局的结构层次，而不是把这类格式硬编码进算法。

## 坐标如何映射为索引

在 TLA 中，可以使用 `tla::crd2offset(coord, shape, stride)` 将逻辑坐标转换为线性索引。

约束如下：

- `coord`、`shape`、`stride` 的 rank 必须一致。
- `coord` 表示逻辑元素坐标，而不是字节偏移。

```cpp
auto shape  = Shape<Shape<_4, _2>, Shape<_4, _3>>{};
auto stride = Stride<Stride<_4, _16>, Stride<_1, _32>>{};

print(crd2offset(tla::MakeCoord(1, 5), shape, stride));  // 37
```

这段代码表示：在一个逻辑大小为 `(8, 12)`、底层按分形格式排布的矩阵中，逻辑坐标 `(1, 5)` 对应的线性索引为 `37`。

## 获取 TileLayout

TileLayout 可以通过 `GetTileLayout` 获取：

```cpp
template <class Layout, class TileShape, class Coord>
auto GetTileLayout(Layout const& layout,
                   TileShape const& tileShape,
                   Coord const& coord);

using namespace tla;
Layout a = Layout<Shape<Shape<_4, _2>, Shape<_4, _3>>,
                  Stride<Stride<_4, _16>, Stride<_1, _32>>,
                  Shape<_8, _12>>{};

Layout a0 = GetTileLayout(a, MakeShape(4, 4), MakeCoord(6, 10));
// 结果可理解为：stride 保持不变，逻辑有效范围裁剪为 (2, 2)
```

参数语义如下：

- `tileShape`：期望取出的 tile 大小，单位是元素。
- `coord`：tile 左上角在父 layout 逻辑空间中的元素坐标，单位也是元素。

也就是说，`coord = (6, 10)` 的含义是“从逻辑第 6 行、第 10 列开始取 tile”，而不是“第 6 个 tile、第 10 个 tile”。

### `GetTileLayout` 的核心语义

`GetTileLayout` 返回的是一个 tile 视图的 `Layout`，不会改变底层数据排布。它主要做三件事：

1. 保留原有 `stride()`，因为底层内存布局没有变化。
2. 用 `tileShape` 构造 tile 的 `shape()`；当父布局带有嵌套结构时，返回结果会在需要时保持同样的结构层次。
3. 根据父 layout 的 `originShape()` 和起始 `coord`，计算 tile 的 `originShape()`。

其中第 3 步最关键：

$$
origin\_shape[d] = \min(tileShape[d], \max(origin\_base[d] - coord[d], 0))
$$

它表示“从当前位置开始，在逻辑上还剩多少有效元素”。因此：

- 中间区域的 tile，`originShape == tileShape`。
- 触边的 tail tile，`originShape` 会自动缩小。

### “按父 layout 的结构转换成对应的 `shape()`”是什么意思

这句话的含义是：当父布局本身是嵌套布局时，tile 的 `shape()` 也需要保持同样的结构层次，这样后续访问规则才能继续复用。

例如，父布局的行和列都按 `16` 为内层块组织：

```text
parent shape = ((16, 7), (16, 7))
parent originShape = (100, 100)
```

如果希望取一个逻辑大小为 `(32, 48)` 的 tile，那么这个 tile 的逻辑尺寸可以直接写成 `(32, 48)`，但在父布局是 `zN` 的前提下，它对应的 `shape()` 会按父布局的结构表达成：

```text
tile logical size = (32, 48)
tile shape         = ((16, 2), (16, 3))
```

这里发生的是“结构转换”，不是“重新排布数据”：

- 逻辑上，tile 仍然是 `32 x 48`。
- 布局上，它被表达成“每维一个 16 的内层块，再乘以外层块个数”。
- `stride()` 仍继承自父布局，因此访问规则不变。

这样做的目的，是保证父 layout 和 tile layout 在结构层次上保持一致。

![Origin_Shape-layout_2.png](https://raw.gitcode.com/user-images/assets/7631999/649c84f3-981f-49eb-be77-6cbf6fd1e5b3/Origin_Shape-layout_2.png 'Origin_Shape-layout_2.png')

### 参数约束

- `tileShape` 与 `coord` 都必须是一层 tuple，即 `depth == 1`。
- `rank(coord) == rank(tileShape)`。

### 不同布局下的行为

- 如果父 layout 是普通 vector 或 matrix，返回 layout 的 `shape()` 通常就等于 `tileShape`。
- 如果父 layout 是嵌套或分形布局，例如 `zN`、`nZ`、`zZ`、`L0C`，当前实现仅支持 `rank == 2`，并会把 `(rows, cols)` 形式的 `tileShape` 转换成与父布局同结构的嵌套 `Shape`。
