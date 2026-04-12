# TLA Tensors

本文介绍 TLA 中的 `Tensor`。

如果说 `Layout` 负责描述“逻辑坐标如何映射到内存”，那么 `Tensor` 就是在 `Layout` 的基础上，再绑定具体数据、当前视图起点和存储层级后的可访问对象。

在本文中，`Tensor` 一律指逻辑视图：

- `MakeTensor` 创建的是视图，不发生数据拷贝。
- `operator()` 的切片结果是子视图，不发生数据拷贝。
- `GetTile` 与 `TileView` 返回的是 tile 视图，不发生数据拷贝。
- `MakeTensorLike` 只是把一块已有存储绑定成“与参考 Tensor 逻辑尺寸一致”的新视图，本身不执行数据搬运。

真正的数据移动应由显式的搬运或计算接口完成，而不是由这些视图构造接口隐式完成。

关于 `Layout` 的基础定义，请先参考 [Layout](./01_layout.md)。

## 先分清四个组成部分

`Tensor` 的模板参数是 `BuiltinTensor`、`Layout`、`Coord`、`Position`。第一次接触时，建议先把这四部分分开理解。

### BuiltinTensor

`BuiltinTensor` 是 AscendC 提供的底层张量对象，例如 `GlobalTensor` 或 `LocalTensor`。它表示“底层存储对象本身”。

### Layout

`Layout` 描述逻辑坐标如何映射到内存，以及逻辑有效范围如何表达。

### Coord

`Coord` 是当前 `Tensor` 视图在BuiltinTensor所表达的父逻辑空间中的起点坐标。

这里需要特别强调两点：

- `coord` 的单位是元素，不是字节。
- `coord` 表示“这个视图从BuiltinTensor所表达的父逻辑空间的哪里开始看”，不是 tile 编号。

例如，一个逻辑大小为 `(8, 16)` 的矩阵中，如果某个子 Tensor 的 `coord()` 是 `(2, 4)`，它表示“这个视图的左上角，对应父逻辑矩阵的第 2 行、第 4 列”。

### Position

`Position` 是 AscendC 中的位置标签，例如 `Arch::PositionGM{}`、`Arch::PositionL1{}`。它用于区分数据位于 GM、L1、L0 等哪一层存储。

## Tensor 构造

当前使用 `MakeTensor` 构造 `Tensor`。

```cpp
using namespace tla;
GlobalTensor<float> A = ...;

auto layout = tla::MakeLayout<float, Catlass::layout::RowMajor>(8, 16);

// 1. 默认从逻辑坐标 (0, 0) 开始
auto tensorA = MakeTensor(A, layout, Arch::PositionGM{});

// 2. 显式指定当前视图起点
auto tensorA_sub = MakeTensor(A, layout, tla::MakeCoord(1, 5), Arch::PositionGM{});
```

可以按下面的方式理解：

- `layout` 决定“如何解释这块内存”。
- `coord` 决定“当前视图从BuiltinTensor所表达的父逻辑空间的哪里开始”。

## Tensor 的常用接口

TLA `Tensor` 提供以下常用接口：

- `.data()`：返回底层内存对象。
- `.layout()`：返回布局。
- `.coord()`：返回当前视图起点。
- `.shape()`：返回 `layout.shape()`。
- `.stride()`：返回 `layout.stride()`。
- `.originShape()`：返回 `layout.originShape()`。
- `(coord0, coord1, ...)`：按坐标索引或切片。

## 统一理解三类“坐标”

TLA 文档中最容易混淆的是几类不同的“坐标”。下面给出统一约定。

### 元素坐标 element coord

元素坐标表示“按元素计数的逻辑位置”，例如 `(row, col)`。`GetTile`、`crd2offset`、普通索引访问等接口使用的都是这种坐标。

### tile 坐标 tile coord

tile 坐标表示“第几个 tile”，不是第几个元素。例如在 `tileShape = (64, 128)` 时：

- `tileCoord = (1, 2)` 表示第 1 个行 tile、第 2 个列 tile。
- 它对应的元素起点是 `(1 * 64, 2 * 128)`。

### 视图起点 view coord

`tensor.coord()` 表示当前 `Tensor` 视图在BuiltinTensor所表达的父逻辑空间中的起点。它由创建这个视图的操作决定，例如 `MakeTensor`、`GetTile`、`TileView` 或切片操作。

可以用一句话概括：

- `element coord` 是元素位置。
- `tile coord` 是 tile 编号。
- `tensor.coord()` 是当前视图的起点。

## 用一个完整示例理解 `coord()`

下面用同一个矩阵，串联 `MakeTensor`、`GetTile` 几种情形。

```cpp
using namespace tla;

GlobalTensor<float> A = ...;
auto layout = tla::MakeLayout<float, Catlass::layout::RowMajor>(8, 16);

auto tensorA = MakeTensor(A, layout, Arch::PositionGM{});
// tensorA.coord() == (0, 0)

auto tensorA_sub = MakeTensor(A, layout, MakeCoord(1, 5), Arch::PositionGM{});
// tensorA_sub.coord() == (1, 5)

auto tileA = GetTile(tensorA_sub, MakeCoord(2, 4), MakeShape(4, 8));
// tileA.coord() == (3, 9)
```

上面分别表示：

1. `tensorA` 直接观察整块逻辑矩阵，因此起点是 `(0, 0)`。
2. `tensorA_sub` 从BuiltinTensor所表达的父逻辑空间的 `(1, 5)` 开始观察，因此起点变为 `(1, 5)`。
3. `tileA` 在 `tensorA_sub` 的基础上再取一个起点为 `(2, 4)` 的 tile，因此新视图起点是 `(1, 5) + (2, 4) = (3, 9)`。

![Origin_Shape-tensor_1.png](https://raw.gitcode.com/user-images/assets/7631999/004cd08b-cda4-4c99-a5c1-63770d71f151/Origin_Shape-tensor_1.png 'Origin_Shape-tensor_1.png')

## 使用 `operator()` 进行索引与切片

TLA `Tensor` 支持使用 `operator()` 做索引，也支持使用 `tla::_` 表达整维切片，返回子 Tensor 视图。

### 基本规则

- 不带 `tla::_` 时，`tensor(i, j, ...)` 返回一个底层 `BuiltinTensor` 访问结果，本质上对应 `tensor.data()[offset]`。
- 带 `tla::_` 时，`tensor(..., tla::_, ...)` 返回子 Tensor 视图；被索引的维度会被固定，保留 `tla::_` 所在维度。
- 这里使用的坐标参数必须是一层 tuple，即每个维度都是标量或 `tla::_`，不支持嵌套 tuple。

等价语义可写为：

```cpp
tensor.data()[tensor.layout()(tensor.coord() + coord_arg)]
```

### 输出 Tensor 的维度

设输入张量 rank 为 $R$，`coord` 中出现 `tla::_` 的维度索引集合为 $\{d_0, d_1, ..., d_{k-1}\}$，则：

- 输出 Tensor 的 rank 为 $k$。
- 输出 Tensor 的 `layout.shape()`、`layout.stride()`、`layout.originShape()` 是输入布局在这些维度上的投影。
- 输出 Tensor 的 `coord()` 会重新从全 0 开始，因为它已经成为新的局部视图。

例如，对 3D 张量 `A(B, M, K)`：

```cpp
auto A2 = A3(b, tla::_, tla::_);  // 3D -> 2D，得到 (M, K) 视图
auto A1 = A2(r, tla::_) // 2D -> 1D，得到 (K)视图
```

![Origin_Shape-tensor_2.png](https://raw.gitcode.com/user-images/assets/7631999/dd1a947a-2371-4e6e-9fef-51a5ceb1556a/Origin_Shape-tensor_2.png 'Origin_Shape-tensor_2.png')

## 获取 TileTensor

### GetTile

`GetTile` 用于从父 Tensor 上切出一个 tile 视图，不拷贝数据。

```cpp
template <class Tensor, class Coord, class Shape>
auto GetTile(Tensor const& tensor,
             Coord const& coord,
             Shape const& shape);
```

参数语义如下：

- `coord`：元素坐标，表示 tile 左上角在父 Tensor 逻辑空间中的起点。
- `shape`：tile 的期望尺寸，单位是元素。

```cpp
using namespace tla;

auto layout = tla::MakeLayout<float, Catlass::layout::RowMajor>(8, 16);
auto tensor = MakeTensor(A, layout, Arch::PositionGM{});

// 从逻辑坐标 (2, 4) 开始，取一个 4 x 8 的 tile
auto tile = GetTile(tensor, tla::MakeCoord(2, 4), MakeShape(4, 8));
```

返回结果可理解为：

- `tile.coord()` = `tensor.coord()` + `(2, 4)`。
- `tile.layout().shape()` 表示期望 tile 尺寸或其与父布局结构一致的表达形式。
- `tile.layout().originShape()` 表示该 tile 真实有效的逻辑范围，触边时会自动裁剪。

### 使用约束

- 支持 `tensor.layout().depth == 1`。
- 若 `tensor.layout().depth > 1`，即分形或嵌套布局，当前 `GetTileLayout` 仅支持 `rank == 2`。
- `coord` 与 `shape` 都必须为一层 tuple，并满足 `rank(coord) == rank(shape) == Tensor::rank`。

### 边界行为

例如父 Tensor 的逻辑尺寸是 `(8, 16)`，执行：

```cpp
auto tail = GetTile(tensor, tla::MakeCoord(6, 10), MakeShape(4, 8));
```

那么：

- 期望尺寸仍然是 `(4, 8)`。
- 但逻辑上只剩下 2 行、6 列有效数据。
- 因此 `tail.layout().originShape()` 会变成 `(2, 6)`。

### TileView

`TileView` 与 `GetTile` 的行为等价，区别只在于输入坐标的单位不同：

- `GetTile` 接收元素坐标。
- `TileView` 接收 tile 坐标。

```cpp
template <class TensorT, class TileCoord, class TileShape>
auto TileView(TensorT const& tensor,
              TileCoord const& tileCoord,
              TileShape const& tileShape);
```

例如：

```cpp
auto tensorTileA = tla::TileView(
      tensorA,
      tla::MakeCoord(0u, kLoopIdx),
      tla::MakeShape(Int<L1_TILE_M>{}, Int<L1_TILE_K>{})
);
```

### 等价关系

`TileView` 与 `GetTile` 可以直接按下面的等式理解：

```cpp
TileView(t, tileCoord, tileShape) = GetTile(t, tileCoord ⊙ tileShape, tileShape)
```

这里的 `⊙` 表示逐维相乘，例如：

```cpp
(1, 2) ⊙ (64, 128) = (64, 256)
```

这条等式表示：

1. `TileView` 先把 tile 坐标转换为元素坐标。
2. 然后按 `GetTile` 的规则创建同一个 tile 视图。

因此，两者的差别只在于调用者提供的是哪一种坐标单位，而不是返回结果的逻辑语义。

![Origin_Shape-tensor_3.png](https://raw.gitcode.com/user-images/assets/7631999/b543ee2b-c938-4e19-87b8-40abd0c81b53/Origin_Shape-tensor_3.png 'Origin_Shape-tensor_3.png')

### 为什么 `TileView` 更适合分块循环

在实际 kernel 或 block 循环中，循环变量通常就是 tile 编号，而不是元素坐标。因此 `TileView` 往往更直接。

下面用同一个按 K 维分块的例子做对比。

#### 写法一：使用 `GetTile`

```cpp
constexpr uint32_t tileM = 64;
constexpr uint32_t tileK = 128;

for (uint32_t kTile = 0; kTile < kTiles; ++kTile) {
      auto coord = tla::MakeCoord(0u, kTile * tileK);
      auto shape = tla::MakeShape(tileM, tileK);
      auto tensorTileA = tla::GetTile(tensorA, coord, shape);
      // use tensorTileA
}
```

#### 写法二：使用 `TileView`

```cpp
constexpr uint32_t tileM = 64;
constexpr uint32_t tileK = 128;

for (uint32_t kTile = 0; kTile < kTiles; ++kTile) {
      auto tensorTileA = tla::TileView(
            tensorA,
            tla::MakeCoord(0u, kTile),
            tla::MakeShape(tileM, tileK)
      );
      // use tensorTileA
}
```

这两段代码的逻辑结果相同，但第二种写法直接使用 tile 坐标，更贴近分块循环本身的语义，也更不容易把“tile 坐标”和“元素坐标”混淆。

## 创建类似的 Tensor

### MakeTensorLike

`MakeTensorLike` 用于创建一个“逻辑尺寸与 `likeTensor` 一致”的新 Tensor。最常见的用途是：从一个已有 tile 视图出发，在另一层内存中构造对应 Tensor，并自动继承其 `originShape()`。

在未指定layoutbase时，行为为根据 LayoutTagDst 决定布局，从 LikeTensor::Element 推断 ElementDst，从 likeTensor 的 originShape 提取尺寸。调用MakeLayout<ElementDst, LayoutTagDst>(originShape())构造目标 layout（可能会因分型布局合法要求对shape进行以分型为粒度的向上取整）

指定layoutbase时，使用MakeLayout(layoutBase.shape(), layoutBase.stride(), likeTensor.originShape())构造目标layout。

这里仍然需要强调：`MakeTensorLike` 构造的是新视图，不执行数据搬运。它只是把用户传入的 `builtinTensor` 绑定成一个新的 TLA `Tensor`，并让这个新视图复用 `likeTensor` 的逻辑尺寸语义。

当前 `MakeTensorLike` 仅支持 `likeTensor.rank <= 2`。

接口分为三类典型场景。

```cpp
// 1) 从 LikeTensor::Element 推断 ElementDst
template <class LayoutTagDst, class BuiltinTensor, class LikeTensor, class PositionType>
auto MakeTensorLike(BuiltinTensor const& builtinTensor,
                    LikeTensor const& likeTensor,
                    PositionType);

// 2) 显式指定 ElementDst
template <class LayoutTagDst, class ElementDst, class BuiltinTensor, class LikeTensor, class PositionType>
auto MakeTensorLike(BuiltinTensor const& builtinTensor,
                    LikeTensor const& likeTensor,
                    PositionType);

// 3) 提供 layoutBase
template <class LayoutTagDst, class BuiltinTensor, class LikeTensor, class PositionType, class LayoutBase>
auto MakeTensorLike(BuiltinTensor const& builtinTensor,
                    LikeTensor const& likeTensor,
                    PositionType,
                    LayoutBase const& layoutBase);

template <class LayoutTagDst, class ElementDst, class BuiltinTensor, class LikeTensor, class PositionType, class LayoutBase>
auto MakeTensorLike(BuiltinTensor const& builtinTensor,
                    LikeTensor const& likeTensor,
                    PositionType,
                    LayoutBase const& layoutBase);
```

### 场景一：源和目标元素类型相同

这是最常见的场景。例如从 GM 中的一个 `half` tile 创建对应的 L1 Tensor，元素类型不变，只是存储层级改变。

```cpp
auto tensorTileA = tla::TileView(
      tensorA,
      tla::MakeCoord(blockM, kTile),
      tla::MakeShape(L1_TILE_M, L1_TILE_K)
);

auto tensorL1A = tla::MakeTensorLike<LayoutTagL1A>(
      l1ATensorList[l1ListId],
      tensorTileA,
      Arch::PositionL1{}
);

// 结果：
// 1. tensorL1A 使用 L1 目标布局
// 2. tensorL1A 的 originShape 与 tensorTileA 相同
// 3. 元素类型从 likeTensor 自动推断
```

### 场景二：目标元素类型不同

当目标 Tensor 的元素类型与源 Tensor 不一致时，需要显式指定 `ElementDst`。例如：

- L0C 中使用 accumulator 类型。
- 需要从 `half` 输入生成 `float` 累加视图。
- 目标内存对象的 `PrimType` 与 `LikeTensor::Element` 不同。

```cpp
auto tensorL0C = tla::MakeTensorLike<LayoutTagL0C, float>(
      l0cTensor,
      tensorTileC,
      Arch::PositionL0C{}
);

// 结果：
// 1. tensorL0C 的逻辑尺寸继承自 tensorTileC
// 2. 目标元素类型显式为 float
// 3. 适用于 accumulator 或类型提升场景
```

### 场景三：目标布局需要额外控制

有些场景下，仅指定 `LayoutTagDst` 还不够，因为目标布局的基础形状或步长需要用户显式给出。例如：

- 目标 Tensor 采用特定分形布局。
- 需要固定某个 L1 的物理排布。注意：L0的排布由originShape唯一确定，因此定制L0上的非预期排布为不合法行为。
- 需要预先给出特殊的 `shape/stride` 结构，但逻辑有效范围仍要继承自 `likeTensor`。

```cpp
auto layoutBaseL1A = tla::MakeLayout<half, LayoutTagL1A>(L1_TILE_M, L1_TILE_K);

auto tensorL1A = tla::MakeTensorLike<LayoutTagL1A>(
      l1ATensor,
      tensorTileA,
      Arch::PositionL1A{},
      layoutBaseL1A
);

// 结果：
// 1. tensorL1A 的 shape/stride 来自 layoutBaseL1A
// 2. tensorL1A 的 originShape 继承自GM上的 tensorTileA
// 3. 即使当前 tile 是尾块，逻辑有效范围也不会丢失
```

如果既要控制目标布局，又要显式指定目标元素类型，可以使用同时带 `layoutBase` 和 `ElementDst` 的重载。

## 实际使用模式

在 block 层和 kernel 层，常见写法通常是两步：

1. 用 `TileView` 从父 Tensor 得到 tile 视图，自动处理边界。
2. 用 `MakeTensorLike` 在目标内存层级构造对应 Tensor，自动继承 `originShape()`。

这套模式的价值在于：

- 主流程始终围绕 tile 编程。
- 尾块逻辑通过 `originShape` 自动传递。
- 数据搬运和计算阶段都能复用同一套逻辑尺寸语义，减少边界分支和歧义。
