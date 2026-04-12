# StridedBatchedMatmulTla Example Readme

## 代码组织

```
├── 45_strided_batched_matmul_tla
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   └── strided_batched_matmul_tla.cpp # 主文件
```

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/1_Practice/01_quick_start.md#编译执行)
- 执行算子

```
# 编译指定用例
bash scripts/build.sh 45_strided_batched_matmul_tla
cd output/bin
# 基本用法：可执行文件名 batch轴|m轴|n轴|k轴|Device ID
# Device ID 可选，默认为0
./45_strided_batched_matmul_tla 5 256 512 1024 0

# layout 定制（仅支持 row/col，大小写不敏感；可选，默认 row row）
# - layoutA: A(M,K) 的 layout
# - layoutB: B(K,N) 的 layout
# layout 是一个“可选的尾部分组”，可以追加在任意一种参数组合的末尾；
./45_strided_batched_matmul_tla 5 256 512 1024 row col
./45_strided_batched_matmul_tla 5 256 512 1024 0 row col

# stride 定制（单位：elements）
# - lda/ldb/ldc：分别为 A(M,K)/B(K,N)/C(M,N) 的 leading dimension
#   - A: row 时 lda>=K；col 时 lda>=M
#   - B: row 时 ldb>=N；col 时 ldb>=K
#   - C: 本示例固定为 row，因此 ldc>=N
# - strideA/strideB/strideC：batch 维度上相邻两矩阵的步长
#
# 只指定 lda/ldb/ldc（strideBatch 默认连续）
./45_strided_batched_matmul_tla 5 256 512 1024 0 1100 600 600
#
# 同时指定 batch stride（支持 batch 间 padding）
./45_strided_batched_matmul_tla 5 256 512 1024 0 1100 600 600 300000 400000 500000

# layout + stride 混用（当传 layoutA/layoutB 时，必须放在命令行最后两个参数位置）
./45_strided_batched_matmul_tla 5 256 512 1024 0 1100 600 600 300000 400000 500000 col row
```

执行结果如下，说明精度比对成功。

```
Compare success.
```
