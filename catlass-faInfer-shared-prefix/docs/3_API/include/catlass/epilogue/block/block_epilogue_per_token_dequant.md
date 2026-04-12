# Block Epilogue Per Token Dequant
>
> [代码位置](../../../../../../include/catlass/epilogue/block/block_epilogue_per_token_dequant.hpp)

## 功能说明

- [BlockEpilogue](./block_epilogue.md)偏特化实现，使用perTokenScale和perChannelScale对block数据做perToken和perChannel的反量化。

- 计算公式：$blockD_{ij} = blockC_{ij} * perChannelScale_j * perTokenScale_i$
- 当前支持的blockC、perChannelScale、perTokenScale、blockD数据类型

<table><thead>
  <tr>
    <th rowspan="2">blockC</th>
    <th rowspan="2">perChannelScale</th>
    <th rowspan="2">perTokenScale</th>
    <th rowspan="2">blockD</th>
  </tr></thead>
  <tbody>
  <tr>
    <td>int32</td>
    <td>half</td>
    <td>half</td>
    <td>half</td>
  </tr>
  <tr>
    <td>int32</td>
    <td>bfloat16_t</td>
    <td>bfloat16_t</td>
    <td>bfloat16_t</td>
  </tr>
  <tr>
    <td>int32</td>
    <td>float</td>
    <td>float</td>
    <td>half</td>
  </tr>
  <tr>
    <td>int32</td>
    <td>float</td>
    <td>float</td>
    <td>bfloat16_t</td>
  </tr>
</tbody>
</table>

## 调度策略

```
// For AtlasA2, per token dequant
template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequant {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};
```

## 调用示例

### Block组装

参考[样例12_quant_matmul](../../../../../../examples/12_quant_matmul/quant_matmul.cpp)

```
constexpr uint32_t ubStages = 2;
using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
using ScaleType = Gemm::GemmType<bfloat16_t, layout::VectorLayout>;
using PerTokenScaleType = Gemm::GemmType<bfloat16_t, layout::VectorLayout>;
using DType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;

using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

using EpilogueTileShape = MatrixShape<32, 256>;
using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
using TileBroadcastOneBlk =
    Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
using TileOneBlkColumnBroadcastMul =
    Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;
```

```
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    EpilogueDispatchPolicy,         // 选用的后处理调度策略
    CType,                          // 反量化前block的类型
    ScaleType,                      // perChannelScale的类型
    PerTokenScaleType,              // perTokenScale的类型
    DType,                          // 反量化后block的类型
    TileRowBroadcastMul,            // tile组件，将(1,n)的scale广播到(m,n)后与block相乘
    TileBroadcastOneBlk,            // tile组件，将(m,1)的perTokenScale广播到(m,32B)
    TileOneBlkColumnBroadcastMul,   // tile组件，将(m,32B)的perTokenScale广播到(m,n)后与block相乘
    TileCopy,                       // tileCopy组件
    TileScheduler                   // tile块切分调度
>;
```

### Block实例化

参考[quant_matmul_multistage_workspace](../../../../../../include/catlass/gemm/kernel/quant_matmul_multistage_workspace.hpp)，在`kernel`代码的`void operator()<AscendC::AIV>`函数中：

```
BlockEpilogue blockEpilogue(resource);
```

### Block更新params

参考[quant_matmul_multistage_workspace](../../../../../../include/catlass/gemm/kernel/quant_matmul_multistage_workspace.hpp)，在`kernel`代码的`void operator()<AscendC::AIV>`函数中：

```
EpilogueParams epilogueParams{
    params.ptrScale,            // perChannelScale的GM地址
    layoutScale,                // perChannelScale的layout
    params.ptrPerTokenScale,    // perTokenScale的GM地址
    layoutPerTokenScale,        // perTokenScale的layout
    params.ptrD,                // 输出矩阵的GM地址
    layoutD                     // 输出矩阵的layout
};

blockEpilogue.UpdateParams(epilogueParams);
```

### Block执行

参考[basic_matmul](../../../../../../include/catlass/gemm/kernel/basic_matmul.hpp)，在`kernel`代码的`void operator()<AscendC::AIC>`函数中：

```
blockEpilogue(
    blockShapeMNK,          // block的shape
    blockCoordMNK,          // block在输出矩阵中的坐标（block粒度）
    actualBlockShapeMNK,    // 待处理block的实际shape
    gmBlockC,               // 待处理block在GM上起始地址
    layoutBlockC            // 待处理block的layout
);
```

## 约束说明

- 当前仅支持blockC、blockD的layout均为`RowMajor`，perChannelScale、perTokenScale的layout均为`VectorLayout`。
