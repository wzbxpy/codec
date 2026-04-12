# QuantMatmulFullLoadA Example Readme

## 代码组织

```
├── 44_quant_matmul_full_loadA_tla
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   ├── 44_quant_matmul_full_loadA_tla.md # 设计文档
│   └── quant_matmul_full_loadA_tla.cpp # 主文件
```

## 功能介绍

该算子在12_quant_matmul基础上支持A矩阵全载，支持单核将A矩阵全部载入L1Cache并常驻，以减少A矩阵在部分矩阵计算场景中的重复搬运，提高性能。当前A矩阵全载模板暂不支持输入包含bias。

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/1_Practice/01_quick_start.md#编译执行)
- 执行算子

```
# 编译指定用例
bash scripts/build.sh 44_quant_matmul_full_loadA_tla
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./44_quant_matmul_full_loadA_tla 256 512 1024 0
```

执行结果如下，说明精度比对成功。5

```
Compare success.
```
