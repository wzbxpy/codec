# MatmulFixpipeOpti Example Readme
## 代码组织
```
├── 46_ascend950_matmul_fixpipe_opti
│   ├── CMakeLists.txt  # CMake编译文件
│   ├── README.md
│   ├── 46_ascend950_matmul_fixpipe_opti.md # 设计文档
│   └── matmul_fixpipe_opti.cpp  # 主文件
```
## 功能介绍
该算子每完成一个基本块的计算，结果数据即通过Fixpipe搬出到UB上。当启用双目标模式控制（dualDstCtrl）时，计算结果矩阵会被分成两部分，并行写入两个Vector核（一个Cube核对应两个Vector核）的专属UB中。每个Vector核的UB支持独立开启Double Buffer以加速流水效率。

## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)，本用例为Ascend 950算子，编译时需加-DCATLASS_ARCH=3510
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 46_ascend950_matmul_fixpipe_opti -DCATLASS_ARCH=3510
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./46_ascend950_matmul_fixpipe_opti 128 128 128 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```