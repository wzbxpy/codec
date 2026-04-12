# Gemm/Tile 类模板概述

Gemm的tile层API作为[blockMmad](../../../../../../docs/3_API/include/catlass/gemm/block/block_mmad.md)的模板参数，一般不需要专门传入（blockMmad会设置默认值），仅在为了特定场景性能优化、或者实现特定功能时，需要在kernel模板组装时做声明。

## API 清单

| 组件名                         | 描述 |
| :----------------------------------------------------------- | :------: |
| [tile_copy](./tile_copy.md)     |   完成mmad所需要的所有tile层搬运模板的集合  |
| [tile_mmad](./tile_mmad.md)     |   tile层mmad计算  |
| [copy_gm_to_l1](./copy_gm_to_l1.md)     |   将tile块从GM搬运到L1  |
