# CATLASS API 列表

CATLASS提供分层的Gemm API接口，从低到高（Basic/Tile/Block/Kernel/Device）组装模板实现算子。开发者可以根据特定需求复用低层次组件、开发高层次组件，实现定制化算子开发。

| 组件分类                                       | 描述 |
| :----------------------------------------------------------- | :------: |
| [gemm/kernel](./include/catlass/gemm/kernel/README.md)     |    设备侧调用的入口，对应了所有Block在NPU上执行逻辑的集合  |
| [gemm/block](./include/catlass/gemm/block/README.md)     |   是矩阵乘（Block层级）累加（MMAD）主循环的主要接口    |
| [gemm/tile](./include/catlass/gemm/tile/README.md) |   使用基础API构建Gemm涉及的NPU微内核    |
| [epilogue/block](./include/catlass/epilogue/block/README.md)   |    Gemm的尾处理组件，也可用于Gemm之外的其他计算 |
| [epilogue/tile](./include/catlass/epilogue/tile/README.md)  |    使用基础API构建尾处理涉及的NPU微内核   |
|[TLA](./include/tla/README.md)   |   抽象数据存储细节，提供通用的访问多维数组算法  |
