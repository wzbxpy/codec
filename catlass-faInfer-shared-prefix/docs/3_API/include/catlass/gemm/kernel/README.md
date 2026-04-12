# Gemm/Kernel 类模板概述

## API 清单

| 组件名                         | 描述 |
| :----------------------------------------------------------- | :------: |
| [basic_matmul](./basic_matmul.md)     |   Common模板基础矩阵乘  |

## API 拆解

命名空间为`Catlass::Gemm::Kernel`，类模板包含如下核心成员：

| 类型         | 名称 |功能|
| :----------------------------------------------------------- | :------: |------: |
| struct    |   Params  |kernel通过`<<<>>>>`调用核函数时的入参|
| struct    |   Arguments  |device封装后kernel后使用的入参|
| 静态函数    |   bool CanImplement  |Arguments校验接口|
| 静态函数    |   size_t GetWorkspaceSize  |基于Arguments计算需要的workSpaceSize|
| 静态函数    |   Params ToUnderlyingArguments  |将Arguments转换为核函数入参Params|
| 函数    |   void operator()\<AscendC::AIC\>  |输入Params，完成AIC上mmad计算|
| 函数    |   void operator()\<AscendC::AIV\>  |输入Params，完成AIV上计算，如前处理、尾处理等|
