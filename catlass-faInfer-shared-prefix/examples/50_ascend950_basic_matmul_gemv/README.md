# BasicMatmulTlaGemv Example Readme

**注意：社区包暂不支持950能力，后续支持的版本敬请期待**

## 代码组织

```
├── 50_ascend950_basic_matmul_gemv
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── basic_matmul_tla.cpp # 主文件
```

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/1_Practice/01_quick_start.md#编译执行)，本用例为Ascend950算子，编译时需加-DCATLASS_ARCH=3510
- 执行算子

```
# 编译指定用例
bash scripts/build.sh 50_ascend950_basic_matmul_gemv -DCATLASS_ARCH=3510
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./50_ascend950_basic_matmul_gemv 1 128 127 0
```

执行结果如下，说明精度比对成功。

```
Compare success.
```

## 使用说明

BasicMatmul默认使用的DispatchPolicy MmadPingpong支持以下几个模板参数：

|模板参数|默认值|参数说明|
|---------|-----------------|-----------------|
|ArchTag| 无 | 指定架构型号 | 
|enableUnitFlag| false | 是否开启Unitflag，开启L0C多缓冲时必须设置为false |
|useHF32| false | 是否开启HF32，仅float类型支持 |
|l0CStages| 1 | 指定L0C的缓冲区数量，设置为2即可开启L0C双缓冲|
|enableL1Resident| false | 是否开启L1常驻 |
|l1AStages | 1 | L1上加载矩阵A的Buffer数量 |
|l1BStages | 1 | L1上加载矩阵B的Buffer数量 |
|l0AStages | 1 | L0上加载矩阵A的Buffer数量 |
|l0AStages | 1 | L0上加载矩阵B的Buffer数量 |

设矩阵Shape为`M N K`, L1上的分块大小为`m1 n1 k1`，M方向的分块数量`mTiles = CeilDiv(M, m1)`，N方向的分块数量`nTiles = CeilDiv(N, n1)`，总任务数为`taskBlocks = mTiles * nTiles`，在以下两种情况下可以选择开启enableL1Resident：

1.`mTies = 1`，且`nTiles > CoreNum`，且`K < 2 * k1`。此时还可以设置`l0CStages=2`(需要关闭enableUnitFlag)，如果空间不足无法设置`l0CStages=2`，则将`n1`设置为原来的一半。

2.`nTies = 1`，且`mTiles > CoreNum`, 且`K < 2 * k1`。此时还可以设置`l0CStages=2`(需要关闭enableUnitFlag)，如果空间不足无法设置`l0CStages=2`，则将`m1`设置为原来的一半。
