# CATLASS FlashAttention Infer

CATLASS FlashAttention Infer是基于CATLASS Gemm Api实现的亲和昇腾Ascend950硬件的FlashAttention推理算子，算子的结构可以分为以下几部分：
* Tiling计算；
* Kernel实现；
* Kernel中依赖适合FlashAttention推理运算的Block组件；
* 使用的Block组件依赖模板库提供的Tile组件。

## Tiling

Tiling计算的逻辑位于[fai_tiling.h](./fai_tiling.h)文件中，在调用算子前，需要准备好tiling计算所需的各项参数，赋值给FAInfo结构体，并调用`GetFATilingParam`函数。[fai.cpp](./fai.cpp)中提供了一个示例：

```c++
// 准备Tiling计算所需的中间结构体
FAInferTiling::FAInfo faInfo;
faInfo.batchSize = batch;
faInfo.numOfHeads = numHeads;
faInfo.numOfKVHeads = kvHeads;
faInfo.seqSize = qSeqlen;
faInfo.seqInnerSize = kvSeqlen;
faInfo.headSize = embeddingSize;
faInfo.scaleValue = static_cast<float>(1.0 / std::sqrt(1.0 * faInfo.headSize));
faInfo.blockSize = blockSize;

FATilingData faTilingData;
FAInferTiling::GetFATilingParam(faInfo, blockDim, faTilingData);
```

`GetFATilingParam`函数实现了对Batch、Head、qSeqLen三轴的贪心切分策略，使得每个AI Core上的计算量尽可能均衡。

## Kernel

本算子Kernel实现位于[fai_kernel.h](./fai_kernel.h)文件中，具有以下特性：
* Kernel主循环逻辑包括多核切分和单核内切分。多核循环包括Tiling切分后的`Batch`、`Head`、`qSeqLen`三层外循环，单核内循环包括`kvSeqlen`按照基本块进行切块，每次Attention运算的基块为一个基本块，CV流水提前Preload下一个基本块的QK Mmad与softmax，让不同基本块的CUBE与VECTOR阶段互相掩盖。
* 支持GQA功能。
* 支持Paged Attention模式，通过blockTable实现KV Cache的分页管理。
* 支持Attention Mask功能，支持左上和右下两种mask模式。
* 采用双缓冲流水线优化，AIC和AIV协同工作，提高计算效率。

在本算子中，使用了Block和Tile层级组件来组装Kernel，具体步骤为：
1. 组装attention计算中的两个BlockMmad（QK,PV）以及两个BlockEpilogue（softmax, rescaleO）。
2. 将Block组合在一起构建成`FAInferKernel`，并在Kernel类中完成对各个Block的循环调用。

这一过程也体现在Kernel入口的代码中：
```c++
// GEMM Block模块，实现Flash Attention Infer的Q * K^T
using DispatchPolicyQK = Gemm::MmadFAIQK<ArchTag, enablePaFlag>;
using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<
    ArchTag, ElementQ, LayoutTagQ, ElementK, LayoutTagK, ElementS, LayoutTagS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
using TileMmadQK = Gemm::Tile::TileMmadTla<ArchTag, ElementQ, typename TileCopyQK::LayoutTagL1A>;
using BlockMmadQK= Gemm::Block::BlockMmadTla<
    DispatchPolicyQK, L1TileShape, L0TileShape, ElementQ, ElementK, ElementS, void, TileCopyQK, TileMmadQK>;

// Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
using DispatchPolicySoftmax = Epilogue::EpilogueAscend950FASoftmax<enableMaskFlag>;
using TileCopySoftmax = Epilogue::Tile::TileCopySoftmax<
    ArchTag, ElementMask, ElementP, LayoutTagMask, LayoutTagP>;
using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<
    DispatchPolicySoftmax, L1TileShape, ElementP, ElementS, ElementMask, TileCopySoftmax>;

// GEMM Block模块，实现Flash Attention Infer的P * V
using DispatchPolicyPV = Gemm::MmadFAIPV<enablePaFlag>;
using TileCopyPV = Gemm::Tile::PackedTileCopyTlaToUB<
    ArchTag, ElementP, LayoutTagP, ElementV, LayoutTagV, ElementOTmp, LayoutTagV, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
using TileMmadPV = Gemm::Tile::TileMmadTla<ArchTag, ElementP, typename TileCopyPV::LayoutTagL1A>;
using BlockMmadPV = Gemm::Block::BlockMmadTla<
    DispatchPolicyPV, L1TileShape, L0TileShape, ElementP, ElementV, ElementOTmp, void, TileCopyPV, TileMmadPV>;

// Epilogue Block模块，实现Flash Attention Infer中当前O基块的更新
using DispatchPolicyRescaleO = Epilogue::EpilogueAscend950FARescaleO;
using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutTagO, LayoutTagOTmp>;
using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, L1TileShape, ElementO, ElementOTmp, TileCopyRescaleO>;

using FAInferKernel = FAInferKernel<
    BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, enablePaFlag>;
```

## Block Mmad

算子总共使用了两类Block Mmad组件，分别为：
* `BlockMmadQK`为BlockMmad模板类的偏特化，用于处理FlashAttention Infer中的Q与K的矩阵乘操作，头文件[block_mmad_fai_qk_tla.hpp](../../include/catlass/gemm/block/block_mmad_fai_qk_tla.hpp)。
* `BlockMmadPV`为BlockMmad模板类的偏特化，用于处理FlashAttention Infer中的P与V的矩阵乘操作，头文件[block_mmad_fai_pv_tla.hpp](../../include/catlass/gemm/block/block_mmad_fai_pv_tla.hpp)。

## Block Epilogue

算子总共使用了两类Block Epilogue组件，分别为：
* `EpilogueOnlineSoftmax`为BlockEpilogue模板类的偏特化，用于处理FlashAttention Infer中的online softmax操作，头文件[block_epilogue_fa_softmax_ascend950.hpp](../../include/catlass/epilogue/block/block_epilogue_fa_softmax_ascend950.hpp)。
* `EpilogueRescaleO`为BlockEpilogue模板类的偏特化，用于处理FlashAttention Infer中的rescaleO操作，头文件[block_epilogue_fa_rescale_o_ascend950.hpp](../../include/catlass/epilogue/block/block_epilogue_fa_rescale_o_ascend950.hpp)。

## Tile Mmad & Tile Copy

在Kernel使用的Block组件中，使用了位于tile_mmad.hpp中的TileMmadTla组件和位于tile_copy.hpp中的PackedTileCopyTlaToUB组件，并新增了针对FA Epilogue处理的TileCopySoftmax和TileCopyRescaleO组件，以及Ascend950新增的ub->l1通路CopyUb2L1Tla组件，例如：

```c++
using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<
    ArchTag, ElementQ, LayoutTagQ, ElementK, LayoutTagK, ElementS, LayoutTagS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
using TileMmadQK = Gemm::Tile::TileMmadTla<ArchTag, ElementQ, typename TileCopyQK::LayoutTagL1A>;

using TileCopySoftmax = Epilogue::Tile::TileCopySoftmax<
        ArchTag, ElementMask, ElementP, LayoutTagMask, LayoutTagP>;

using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutTagO, LayoutTagOTmp>;

using CopyUbToL1P = Tile::CopyUb2L1Tla<ArchTag, decltype(vf1OutUb), TensorDst>;
```

这些Tile组件负责数据在GM、L1、L0和UB之间的搬运，以及矩阵乘法和Sofemax的底层实现。PackedTileCopyTlaToUB支持TLA（Tensor Layout Abstraction）布局，能够高效地处理不同布局的数据搬运需求。Tile::CopyUb2L1Tla支持AIV Ub上的计算结果直接搬运到AIC L1上，相比之前Ub->GM->L1的搬运实现了效率提升。
