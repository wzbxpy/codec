# FlashAttention Infer Example Readme

## 代码组织
```
├── 49_ascend950_flash_attention_infer
│   ├── CMakeLists.txt           # CMake编译文件
│   ├── gen_data.py              # 数据生成脚本
│   ├── fai_kernel_utils.h       # Kernel辅助文件
│   ├── tiling_data_def.h        # Tiling数据结构定义
│   ├── fai.cpp                  # 主程序入口
│   ├── fai_kernel.h             # Kernel实现
│   ├── fai_tiling.h             # Tiling计算实现
│   └── README.md
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)   

- 接下来，先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入, 执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据。   
- 然后执行算子，这里要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。

以下是一个完整的shell脚本示例
```
batch=1          # batch大小
qSeqlen=177      # query序列长度
kvSeqlen=512     # key/value序列长度
numHeads=1       # query head数量
kvHeads=1        # key/value head数量
headSize=128     # embeddingSize
isVariedLen=0    # 是否使用变长序列，当前仅支持0
maskType=1       # mask类型，0表示无mask，1表示使用mask
dtype="half"     # 数据类型，支持"half"或"bf16"
cacheMode=1      # 缓存模式，0表示非Paged Attention，1表示Paged Attention
device=0

function build() {
    rm -rf build
    rm -rf output
    bash scripts/build.sh 49_ascend950_flash_attention_infer -DCATLASS_ARCH=3510
}

function gen_data() {
    python3 examples/49_ascend950_flash_attention_infer/gen_data.py $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType $cacheMode "$dtype"
    echo "Data gen finished"
}

function run_kernel {
    echo 'Case: B=' $batch ' qS=' $qSeqlen ' kvS=' $kvSeqlen ' qN=' $numHeads ' kvN=' $kvHeads ' D=' $headSize ' mask=' $maskType
    cd output/bin/
    ./49_ascend950_flash_attention_infer $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType $cacheMode --device $device --dtype $dtype
}

build
gen_data
run_kernel
```

执行结果如下，说明精度比对成功。
```
Compare success.
```