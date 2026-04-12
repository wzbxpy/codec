# Adaptive Sliding Window Tiling策略说明

ASWT(Adaptive Sliding Window Tiling)策略决定了基本块的分核情况和计算顺序。与[Swizzle](./02_swizzle.md)类似，ASWT采用了S形滑窗机制来提高L2缓存命中率、减小数据读取开销。特别地，当基本块总数无法均分给每一个AI Core时，ASWT会对剩余基本块进一步切分，使其尽可能平均分配给每一个AI Core，达到负载均衡。

下方展示了ASWT策略。图中每一个方块表示C矩阵的一个基本块，方块中的序号代表AI Core的编号（该示例中假设AI Core数量为20）。按照S形滑窗顺序将基本块依次分配给AI Core进行处理，最后还剩9个基本块，无法均分给20个AI Core。为使每一个AI Core尽可能负载均衡，提升数据并行效率，对剩余的9个基本块进行切分，使得切分后的块数（18块）至少能够分配给AI Core总数的一半以上。

<img src="../../figures/aswt.png" width="50%">

## 适用场景
假设左矩阵shape为(m, k)，右矩阵shape为(k, n)，C矩阵上的基本块大小为(baseM, baseN)，则基本块的总个数tileNum：

$$\mathrm{tileNum = Ceil(\mathrm{m, baseM}) * Ceil(n, baseN)}$$

当基本块个数无法均分给所有AI Core，且剩余基本块个数不足AI Core总数的一半，即：

$$ \mathrm{tileNum \space \%  \space coreNum <= \frac{coreNum}{2} }$$

其中，coreNum表示使用的AI Core总核数。

此时，采用ASWT分核策略对基本块进行切分，使其尽可能均匀地分配给更多的AI Core，能够提升数据并行效率。

## 性能收益
在使用相同的tileShape和数据类型情况下，使用ASWT相较于使用[Swizzle](./02_swizzle.md)，basic_matmul的性能对比如下表。
|[M, N, K]|basic_matmul_swizzle|basic_matmul_aswt|加速比|
|---------|--------------|-------------------|-------|
|[1024, 1024, 1024]| 14.95us | 15.08us | 0.99 |
|[2048, 2048, 256]| 11.95us | 12.09us | 0.99 |
|[2208, 2048, 512]| 22.07us | 18.65us | 1.18 |  
|[2208, 2048, 1024]| 38.15us | 30.51us | 1.25 |
|[1024, 2368, 512]| 16.02us | 12.00us | 1.34 |
|[1024, 2368, 1024]| 26.18us | 19.82us | 1.32 |
|[1024, 2368, 2048]| 45.88us | 34.25us | 1.34 |

### 说明
- basic_matmul_swizzle表示使用Swizzle策略的[basic_matmul](../../../examples/43_ascend950_basic_matmul/README.md)。
- basic_matmul_aswt表示使用ASWT策略的basic_matmul。
- L1TileShape: [256, 256, 128]
- L0TileShape: [256, 256, 64]
- 输入A、B矩阵的数据类型为half，输出C矩阵的数据类型为float。
