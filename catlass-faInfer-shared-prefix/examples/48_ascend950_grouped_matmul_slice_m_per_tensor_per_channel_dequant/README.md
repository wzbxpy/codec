# GroupedMatmulSliceMPerTensorPerChannelDequant Example Readme

## 代码组织

```
├── 48_ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── grouped_matmul_slice_m_per_tensor_per_channel_dequant.cpp # 主文件
```

## 功能介绍

该算子支持A矩阵在m轴切分，然后和B矩阵按照group分组进行矩阵乘，之后进行per_tensor或per_channel反量化操作。

A/B矩阵为int8类型，scale为float，输出结果为half。

### Fixpipe随路量化/反量化

对于特定输入输出数据类型，Fixpipe支持将计算结果从CO1搬出到Global Memory时，通过配置Fixpipe的量化/反量化模式和量化/反量化参数，对输出C矩阵元素执行数据量化或反量化操作。

- Matmul量化场景：Matmul计算时左矩阵A、右矩阵B为half数据类型，输出C矩阵为int8_t数据类型。该场景下，C矩阵的数据从CO1搬出到Global Memory时，会执行量化操作，将最终结果量化为int8_t类型，如下图所示。

    ![alt text](../../docs/figures/fixpipe_quant.png)

- Matmul反量化场景：Matmul计算时左矩阵A、右矩阵B为int8_t数据类型，输出C矩阵为half数据类型。该场景下，C矩阵的数据从CO1搬出到Global Memory时，会执行反量化操作，将最终结果反量化为对应的half类型，如下图所示。

    ![alt text](../../docs/figures/fixpipe_dequant.png)

Fixpipe提供了两种不同粒度的随路量化/反量化模式，即per_tensor和per_channel。

1. per_tensor：对整个Tensor进行量化/反量化，Tensor具有唯一的缩放因子。这种方法可以降低模型的存储和计算成本，但会降低模型的精度。
2. per_channel：对Tensor的每个通道单独进行量化/反量化，同一通道内共享同一缩放因子，通道间缩放因子则各不相同。这种方法可以更好地保留模型的精度，但会增加模型的存储和计算成本。

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/1_Practice/01_quick_start.md#编译执行)，本用例为Ascend 950算子，编译时需加-DCATLASS_ARCH=3510
- 执行算子

```
# 编译指定用例
bash scripts/build.sh 48_ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant -DCATLASS_ARCH=3510
cd output/bin
# 可执行文件名|group数量|矩阵m轴|n轴|k轴|量化模式|Device ID
# 量化模式可选0或1，0表示per_tensor，1表示per_channel
# Device ID可选，默认为0
./48_ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant 128 512 1024 2048 0 0
```

执行结果如下，说明精度比对成功。

```
Compare success.
```

## 使用说明

GroupedMatmulSliceMPerTensorPerChannelDequant默认使用的DispatchPolicy MmadDequant支持以下几个模板参数：

|模板参数|默认值|参数说明|
|---------|-----------------|-----------------|
|ArchTag| 无 | 指定架构型号 | 
|enableUnitFlag| false | 是否开启Unitflag，开启L0C多缓冲时必须设置为false |
|useHF32| false | 是否开启HF32，仅float类型支持 |
|l0CStages| 1 | 指定L0C的缓冲区数量，设置为2即可开启L0C双缓冲|
|enableL1Resident| false | 是否开启L1常驻 |
|l1AStages | 2 | L1上加载矩阵A的Buffer数量 |
|l1BStages | 2 | L1上加载矩阵B的Buffer数量 |
|l0AStages | 2 | L0上加载矩阵A的Buffer数量 |
|l0BStages | 2 | L0上加载矩阵B的Buffer数量 |

设矩阵Shape为`M N K`, L1上的分块大小为`m1 n1 k1`，M方向的分块数量`mTiles = CeilDiv(M, m1)`，N方向的分块数量`nTiles = CeilDiv(N, n1)`，总任务数为`taskBlocks = mTiles * nTiles`，在以下两种情况下可以选择开启enableL1Resident：

1.`mTies = 1`，且`nTiles > CoreNum`，且`K < 2 * k1`。此时还可以设置`l0CStages=2`(需要关闭enableUnitFlag)，如果空间不足无法设置`l0CStages=2`，则将`n1`设置为原来的一半。

2.`nTies = 1`，且`mTiles > CoreNum`, 且`K < 2 * k1`。此时还可以设置`l0CStages=2`(需要关闭enableUnitFlag)，如果空间不足无法设置`l0CStages=2`，则将`m1`设置为原来的一半。
