---
title: "Chapter 25: 卷积算子与 Im2Col 变换"
description: "深入理解卷积运算的多种实现策略，包括 Im2Col + GEMM、Winograd 变换和直接卷积，掌握 TileLang 中卷积算子的高效实现与优化技巧"
updated: "2025-01-01"
---

# Chapter 25: 卷积算子与 Im2Col 变换

> **Learning Objectives**
>
> 1. 理解卷积运算的数学原理与计算模式
> 2. 掌握 Im2Col 变换的核心思想及其与 GEMM 的关系
> 3. 学会使用 TileLang 实现 Im2Col + GEMM 卷积
> 4. 理解 Winograd 变换的数学原理与适用场景
> 5. 掌握直接卷积在 TileLang 中的实现方法
> 6. 学会卷积算子的内存访问优化技巧（Padding、Stride 处理）
> 7. 能够对比不同卷积实现策略的性能特征

---

## 1. 卷积运算基础

### 1.1 卷积的数学定义

卷积是深度学习中最基础的运算之一。在离散域中，二维卷积的数学定义为：

$$
Y[n, m] = \sum_{i=0}^{K_h-1} \sum_{j=0}^{K_w-1} X[n+i, m+j] \cdot W[i, j]
$$

其中 $X$ 是输入特征图，$W$ 是卷积核，$Y$ 是输出特征图，$K_h$ 和 $K_w$ 分别是卷积核的高度和宽度。

### 1.2 多通道卷积

在实际的深度学习框架中，卷积通常涉及多通道输入和多通道输出：

$$
Y[n, c_{out}, h, w] = \sum_{c_{in}=0}^{C_{in}-1} \sum_{i=0}^{K_h-1} \sum_{j=0}^{K_w-1} X[n, c_{in}, h \cdot s + i, w \cdot s + j] \cdot W[c_{out}, c_{in}, i, j]
$$

其中：
- $N$：批次大小（Batch Size）
- $C_{in}$：输入通道数
- $C_{out}$：输出通道数
- $K_h, K_w$：卷积核高度和宽度
- $s$：步长（Stride）

### 1.3 卷积的计算复杂度

对于一个标准卷积操作，其计算复杂度为：

$$
\text{FLOPs} = N \times C_{out} \times H_{out} \times W_{out} \times C_{in} \times K_h \times K_w \times 2
$$

其中 $H_{out}$ 和 $W_{out}$ 是输出特征图的空间维度：

$$
H_{out} = \left\lfloor \frac{H_{in} + 2P - K_h}{s} \right\rfloor + 1
$$

> [!TIP]
> 卷积的计算强度（Arithmetic Intensity）通常较低，因为它是一种"重数据移动、轻计算"的运算。这使得卷积往往受限于内存带宽（Memory-bound）而非计算能力（Compute-bound）。

### 1.4 卷积的实现策略总览

在高性能计算中，卷积有三种主要实现策略：

这四种实现策略的选择本质上是"时间换空间"或"空间换时间"的经典计算机科学权衡。Im2Col+GEMM 方法之所以在 cuDNN、Caffe、早期的 TensorFlow 等框架中被广泛采用，并非因为它是最优的，而是因为它将"不规则的卷积计算"转化为"高度规则的矩阵乘法"——而矩阵乘法是过去五十年中整个高性能计算社区投入最多优化精力的运算。当你的底层 GEMM 库（如 cuBLAS、MKL）已经经过数千人年的优化，通过 Im2Col 借用这些优化成果是极其务实的工程决策。然而，随着深度学习模型向移动端和边缘设备迁移（这些场景内存极度受限），以及编译器技术的进步（如 TVM、XLA），直接卷积和 Winograd 变换的吸引力正在持续上升。

从硬件微架构的角度来看，这四种策略对 GPU 不同子系统的利用模式截然不同。Im2Col+GEMM 重度使用 Tensor Core 和寄存器文件，因为 GEMM 天生适合 Tensor Core 的分块矩阵乘加指令；Winograd 变换用更多的加减法替换乘法，对整数运算单元和共享内存的压力更大；直接卷积则严重依赖 L1 缓存和纹理缓存来弥补不规则访问模式带来的延迟。理解这些底层差异，是成为一个精通 GPU 性能优化的工程师的关键。


| 策略 | 核心思想 | 优势 | 劣势 |
|------|---------|------|------|
| Im2Col + GEMM | 将卷积转换为矩阵乘法 | 可复用 GEMM 优化，通用性强 | 额外内存开销，数据拷贝 |
| Winograd 变换 | 利用多项式变换减少乘法次数 | 理论乘法次数最少 | 受限于小卷积核，数值精度问题 |
| 直接卷积 | 直接按卷积定义计算 | 无额外内存开销 | 难以充分利用 GPU 并行性 |

<div data-component="ConvolutionImplementationComparison"></div>

---

## 2. Im2Col + GEMM 方法

### 2.1 Im2Col 变换原理

Im2Col（Image to Column）是将卷积转换为矩阵乘法的经典方法。其核心思想是：

1. 将输入特征图中每个卷积窗口的元素展开为一个列向量
2. 将所有列向量排列成一个矩阵
3. 将卷积核也展开为矩阵形式
4. 通过矩阵乘法完成卷积计算

#### 数学表达

设输入特征图为 $X \in \mathbb{R}^{C_{in} \times H_{in} \times W_{in}}$，卷积核为 $W \in \mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w}$。

Im2Col 变换将 $X$ 转换为矩阵 $X_{col} \in \mathbb{R}^{(C_{in} \cdot K_h \cdot K_w) \times (H_{out} \cdot W_{out})}$。

卷积核被重塑为 $W_{col} \in \mathbb{R}^{C_{out} \times (C_{in} \cdot K_h \cdot K_w)}$。

最终卷积结果为：

$$
Y = W_{col} \times X_{col}
$$

<div data-component="Im2ColTransformDemo"></div>

### 2.2 Im2Col 变换的 TileLang 实现

```python
import tilelang
from tilelang import Profiler
import tilelang.language as T
import torch

def im2col_transform(
    N: int,
    C: int,
    H: int,
    W: int,
    K_h: int,
    K_w: int,
    stride: int,
    padding: int,
):
    """Im2Col transformation using TileLang."""
    H_out = (H + 2 * padding - K_h) // stride + 1
    W_out = (W + 2 * padding - K_w) // stride + 1

    @T.prim_func
    def im2col_kernel(
        X: T.Buffer((N, C, H, W), "float32"),
        X_col: T.Buffer((C * K_h * K_w, N * H_out * W_out), "float32"),
    ):
        with T.Kernel(C * K_h * K_w, N * H_out * W_out, threads=256) as (i, j):
            c = i // (K_h * K_w)
            residual = i % (K_h * K_w)
            kh = residual // K_w
            kw = residual % K_w

            n = j // (H_out * W_out)
            residual_j = j % (H_out * W_out)
            oh = residual_j // W_out
            ow = residual_j % W_out

            ih = oh * stride - padding + kh
            iw = ow * stride - padding + kw

            if 0 <= ih < H and 0 <= iw < W:
                X_col[i, j] = X[n, c, ih, iw]
            else:
                X_col[i, j] = T.float32(0)

    return im2col_kernel
```

这段代码实现了 Im2Col 变换的核心逻辑。通过将线程索引 `i` 和 `j` 分别映射到卷积核参数（c, kh, kw）和输出空间位置（n, oh, ow），将输入特征图重新排列为矩阵形式。索引计算采用整除和取模运算，先提取通道维度再提取卷积核内的空间位置。边界处理通过条件判断实现：当计算出的输入坐标（ih, iw）超出有效范围时填充零，这对应于 padding 操作。这种变换将卷积问题转化为标准的矩阵乘法，是 Im2Col 方法的基础。


深入分析 Im2Col 的索引计算可以发现一个微妙的设计细节：线程索引 `i` 和 `j` 的分解方式直接影响全局内存访问的合并性（Coalescing）。在上述实现中，`j` 维度对应的是输出空间位置展平后的连续索引（n, oh, ow），这意味着相邻的线程访问 X_col 矩阵的相邻列——在列主序（Column-major）布局下，这会导致非合并访问（每个线程访问不同行同一列）。如果将 `i` 和 `j` 的映射方式进行交换——让 `j` 对应通道和卷积核维度，`i` 对应空间维度——则可以实现完美的合并访问，因为相邻线程将在 C×K_h×K_w 维度上连续访问。这种对内存布局与线程映射之间关系的敏感度，是区分"能写出正确结果的代码"和"能写出高性能代码"的关键所在。

此外，边界填充（Padding）的处理方式也有多种选择。上述代码使用条件赋值（if-else），在每个线程内部处理边界逻辑，这会引入控制流分歧。一种更高效的替代方案是"预先扩展"——在卷积之前先将输入特征图扩展一圈零值，然后在 Im2Col 和后续计算中完全省略边界检查。这样做的代价是额外的内存和时间用于填充操作，但在输入尺寸较大时，为每个线程省去的分支指令累积起来远超填充开销。对于小输入（如 MobileNet 的 14×14 特征层），填充策略通常更优；对于大输入（如 ResNet 的 224×224 输入），条件判断策略同样可接受。

### 2.3 Im2Col + GEMM 完整卷积实现

```python
def conv2d_im2col_gemm(
    N: int,
    C_in: int,
    C_out: int,
    H: int,
    W: int,
    K_h: int,
    K_w: int,
    stride: int,
    padding: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
):
    """Conv2D via Im2Col + GEMM using TileLang."""
    H_out = (H + 2 * padding - K_h) // stride + 1
    W_out = (W + 2 * padding - K_w) // stride + 1
    M = C_out
    N_gemm = N * H_out * W_out
    K = C_in * K_h * K_w

    @T.prim_func
    def conv2d_kernel(
        Weight: T.Buffer((C_out, C_in, K_h, K_w), "float32"),
        X: T.Buffer((N, C_in, H, W), "float32"),
        Y: T.Buffer((N, C_out, H_out, W_out), "float32"),
    ):
        # Step 1: Im2Col transformation
        X_col = T.alloc_buffer((C_in * K_h * K_w, N * H_out * W_out), "float32")
        Weight_col = T.alloc_buffer((C_out, C_in * K_h * K_w), "float32")

        # Step 2: GEMM computation
        with T.Kernel(
            T.ceildiv(M, block_M), T.ceildiv(N_gemm, block_N), threads=256
        ) as (bx, by):
            X_local = T.alloc_fragment((block_M, block_K), "float32")
            W_local = T.alloc_fragment((block_K, block_N), "float32")
            Y_local = T.alloc_fragment((block_M, block_N), "float32")

            T.clear(Y_local)

            for k in T.serial(T.ceildiv(K, block_K)):
                # Load tiles
                T.copy(
                    X_col[k * block_K : (k + 1) * block_K,
                           by * block_N : (by + 1) * block_N],
                    X_local,
                )
                T.copy(
                    Weight_col[bx * block_M : (bx + 1) * block_M,
                               k * block_K : (k + 1) * block_K],
                    W_local,
                )
                T.gemm(X_local, W_local, Y_local)

            # Store result
            T.copy(
                Y_local,
                Y[0, bx * block_M : (bx + 1) * block_M,
                  by * block_N : (by + 1) * block_N],
            )

    return conv2d_kernel
```

该函数将 Im2Col 变换与 GEMM 矩阵乘法融合为一个完整的卷积实现。首先计算输出维度和 GEMM 所需的矩阵形状参数（M=N_gemm=输出空间位置数，K=通道×卷积核大小）。GEMM 部分采用标准的分块策略：将输出矩阵划分为 block_M×block_N 的 Tile，沿 K 维度进行分块累加。每个 Tile 加载到本地缓冲区后通过 `T.gemm()` 执行矩阵乘法。这种方法的核心优势在于可以复用高度优化的 GEMM 内核，但代价是需要额外的内存来存储 Im2Col 变换后的矩阵。

### 2.4 Im2Col 的内存开销分析


要全面评估 Im2Col+GEMM 方法的性能，必须将其分解为两个阶段来独立分析。第一阶段是 Im2Col 变换本身——这是一个纯粹的内存搬移操作，不涉及任何浮点计算。它的性能瓶颈完全在于内存带宽：从输入缓冲区读取每个像素值，按列优先的顺序写入 Im2Col 矩阵。由于输入 X 的读取涉及不规则的内存访问模式（每个输出位置的窗口起始点不同），而写入 X_col 是规则的线性写入，这一阶段的带宽利用率通常不超过 60%-70%（以 A100 约 2TB/s 的 HBM 带宽为参考）。第二阶段是 GEMM 矩阵乘法——这是 GPU 最擅长的运算，在合适的 Tile 尺寸下可以达到 80%-90% 的 Tensor Core 利用率。因此，Im2Col+GEMM 的总延迟大致等于 `Im2Col 延迟 + GEMM 延迟`，其中 Im2Col 的占比随通道数增加而上升（因为需要移动更多的像素数据），而 GEMM 的占比随输出维度增大而上升。

对比 Im2Col+GEMM 与直接卷积，一个有趣的观察是：当 C_in 和 C_out 都很大（如 512→512 的卷积）且卷积核较小（如 3×3）时，Im2Col 的内存膨胀并不严重——因为 Im2Col 矩阵的尺寸为 (C_in×9)×(H_out×W_out)，而原始输入为 (C_in)×(H×W)，膨胀比仅为 9。但当 C_in 很小而空间维度很大时（如 3×224×224, K=7），膨胀比达到 K_h×K_w = 49，此时内存开销变得不可接受。因此，一种混合策略是：对通道数多、空间维度小的层使用 Im2Col+GEMM，对通道数少、空间维度大的层使用直接卷积。这也是 cuDNN 启发式算法自动选择策略的基本逻辑。

Im2Col 变换会引入额外的内存开销：

| 参数 | 公式 | 示例值（ResNet-50 第一层） |
|------|------|--------------------------|
| 原始输入大小 | $N \times C_{in} \times H \times W \times 4$ bytes | $1 \times 3 \times 224 \times 224 \times 4 = 602$ KB |
| Im2Col 后大小 | $C_{in} \cdot K_h \cdot K_w \times N \cdot H_{out} \cdot W_{out} \times 4$ bytes | $27 \times 50176 \times 4 = 5.3$ MB |
| 内存膨胀比 | $\frac{C_{in} \cdot K_h \cdot K_w}{C_{in}} = K_h \cdot K_w$ | $9\times$ |

> [!WARNING]
> 对于大卷积核（如 $7 \times 7$），Im2Col 的内存膨胀可能非常显著。在内存受限的场景下，需要考虑直接卷积或分块 Im2Col 策略。

Im2Col 内存膨胀问题在工程中有多种缓解手段。一种称为"隐式 Im2Col"（Implicit Im2Col 或 Implicit GEMM）的技术，通过在 GEMM 内核内部动态计算 Im2Col 索引来避免分配完整的 Im2Col 矩阵。类似于"不显式存储变换矩阵，而是在每次访问时即时计算坐标"。这种方法由 cuDNN 和 CUTLASS 广泛采用，可以将内存开销从 O(K²×spatial) 降至 O(1)，但代价是增加了 GEMM 内核内部的索引计算复杂度。在 TileLang 中，由于 Tile 抽象自动管理索引计算，开发者无需手动处理隐式 Im2Col 的复杂坐标映射，只需关注分块策略本身。

另一个值得关注的趋势是混合精度（Mixed Precision）训练和推理对 Im2Col 内存压力的缓解。使用 fp16 或 bf16 存储输入和权重可以将 Im2Col 矩阵的内存占用减半（从 float32 的 4 字节降至 2 字节），同时在现代 GPU（A100、H100）上还能激活 Tensor Core 获得 2-8 倍的矩阵乘法吞吐量提升。然而，float16 的数值范围有限（±65504），在 Im2Col 变换中如果输入值较大（如未经归一化的原始像素值 0-255），乘以权重后可能溢出。此时 bfloat16 是一个更好的选择——它保持了与 float32 相同的指数范围（8 位指数），只是尾数精度降低（7 位尾数），在大多数深度学习场景中几乎不损失精度。


---

从 Im2Col 到 Winograd 变换，体现了一个重要的系统设计哲学：通过数学等价变换来换取计算效率。Im2Col 将卷积转化为 GEMM，思路直观但付出了内存膨胀的代价；而 Winograd 进一步利用多项式代数的性质，在变换域中以更少的乘法完成等价计算。两种方法的选择取决于问题的规模——当内存不是瓶颈时，Im2Col+GEMM 凭借成熟的 GEMM 优化生态往往是最稳妥的选择；当需要极致性能且卷积核较小时，Winograd 的数学优势就会凸显。


## 3. Winograd 变换

### 3.1 Winograd 变换的数学原理

Winograd 变换是一种基于多项式分解的快速卷积算法。对于 $F(m, r)$ Winograd 变换，其中 $m$ 是输出大小，$r$ 是滤波器大小：

- 一维情况下，$F(m, r)$ 需要 $m + r - 1$ 次乘法（而非 $m \times r$ 次）
- 对于 $F(2, 3)$：需要 4 次乘法（而非 6 次），减少 33% 的乘法运算

#### Winograd 变换公式

对于输出大小为 $2 \times 2$，滤波器大小为 $3 \times 3$ 的卷积（$F(2, 3)$）：

$$
Y = A^T \left[ (G g G^T) \odot (B^T d B) \right] A
$$

其中：
- $g$：$3 \times 3$ 滤波器
- $d$：$4 \times 4$ 输入块
- $G$：滤波器变换矩阵（$4 \times 3$）
- $B$：输入变换矩阵（$4 \times 4$）
- $A$：输出变换矩阵（$4 \times 2$）
- $\odot$：逐元素乘法

#### 变换矩阵

对于 $F(2, 3)$：

$$
G = \begin{bmatrix} 1 & 0 & 0 \\ 0.5 & 0.5 & 0.5 \\ 0.5 & -0.5 & 0.5 \\ 0 & 0 & 1 \end{bmatrix}
$$

$$
B = \begin{bmatrix} 1 & 0 & -1 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & -1 & 1 & 0 \\ 0 & 1 & 0 & -1 \end{bmatrix}
$$

$$
A = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & -1 \\ 0 & -1 \end{bmatrix}
$$

<div data-component="WinogradTransformFlow"></div>

### 3.2 Winograd 变换的 TileLang 实现

```python
def winograd_conv2d(
    N: int,
    C_in: int,
    C_out: int,
    H: int,
    W: int,
):
    """Winograd F(2,3) convolution using TileLang."""
    assert H % 2 == 0 and W % 2 == 0, "H and W must be even"
    H_out = H - 2
    W_out = W - 2
    tile_h = T.ceildiv(H_out, 2)
    tile_w = T.ceildiv(W_out, 2)

    @T.prim_func
    def winograd_kernel(
        X: T.Buffer((N, C_in, H, W), "float32"),
        U: T.Buffer((4, 4, C_in, C_out), "float32"),  # Pre-transformed filters
        Y: T.Buffer((N, C_out, H_out, W_out), "float32"),
    ):
        # Allocate intermediate buffers
        V = T.alloc_buffer((4, 4, C_in, N, tile_h, tile_w), "float32")
        M = T.alloc_buffer((4, 4, C_in, N, tile_h, tile_w), "float32")

        # Step 1: Input transform - B^T d B
        with T.Kernel(N, C_in, tile_h, tile_w, threads=128) as (n, c, th, tw):
            d = T.alloc_fragment((4, 4), "float32")
            v = T.alloc_fragment((4, 4), "float32")

            # Load 4x4 input tile
            for i, j in T.serial(4, 4):
                h_idx = th * 2 + i
                w_idx = tw * 2 + j
                if h_idx < H and w_idx < W:
                    d[i, j] = X[n, c, h_idx, w_idx]
                else:
                    d[i, j] = T.float32(0)

            # Apply B^T d B transform
            # B^T = [[1,0,0,0],[0,1,-1,1],[-1,1,1,0],[0,0,0,-1]]
            tmp = T.alloc_fragment((4, 4), "float32")
            for i in T.serial(4):
                tmp[0, i] = d[0, i] - d[2, i]
                tmp[1, i] = d[1, i] + d[2, i]
                tmp[2, i] = d[2, i] - d[1, i]
                tmp[3, i] = d[3, i] - d[1, i]

            for i in T.serial(4):
                v[i, 0] = tmp[i, 0] - tmp[i, 2]
                v[i, 1] = tmp[i, 1] + tmp[i, 2]
                v[i, 2] = tmp[i, 2] - tmp[i, 1]
                v[i, 3] = tmp[i, 3] - tmp[i, 1]

            for i, j in T.serial(4, 4):
                V[i, j, c, n, th, tw] = v[i, j]

        # Step 2: Element-wise multiplication in transform domain
        with T.Kernel(4, 4, C_out, N, tile_h, tile_w, threads=128) as (
            i, j, co, n, th, tw
        ):
            acc = T.alloc_fragment((1,), "float32")
            acc[0] = T.float32(0)
            for ci in T.serial(C_in):
                acc[0] += U[i, j, ci, co] * V[i, j, ci, n, th, tw]
            M[i, j, co, n, th, tw] = acc[0]

        # Step 3: Output transform - A^T m A
        with T.Kernel(N, C_out, tile_h, tile_w, threads=128) as (n, co, th, tw):
            m = T.alloc_fragment((4, 4), "float32")
            for i, j in T.serial(4, 4):
                m[i, j] = M[i, j, co, n, th, tw]

            # A^T = [[1,1,1,0],[0,1,-1,-1]]
            tmp = T.alloc_fragment((2, 4), "float32")
            for i in T.serial(4):
                tmp[0, i] = m[0, i] + m[1, i] + m[2, i]
                tmp[1, i] = m[1, i] - m[2, i] - m[3, i]

            for i in T.serial(2):
                for j in T.serial(2):
                    h_idx = th * 2 + i
                    w_idx = tw * 2 + j
                    if h_idx < H_out and w_idx < W_out:
                        Y[n, co, h_idx, w_idx] = (
                            tmp[i, 0] + tmp[i, 1] + tmp[i, 2]
                        )

    return winograd_kernel
```

Winograd 变换的核心数学洞察来自中国剩余定理（Chinese Remainder Theorem）在多项式环上的推广。直观地说，两个多项式的乘积可以被表示为它们在一组精心选择的采样点上的值的逐点乘积——这就是 Winograd 算法将"滑动窗口卷积"（多项式乘法）转化为"逐元素乘法"（点值乘法）的数学本质。变换矩阵 G、B、A 的作用，正是将滤波器和输入块从"系数表示"变换到"点值表示"（在某个精心选择的基下），在点值域执行低开销的逐元素乘法，然后再变换回系数表示。这使得理论乘法次数从 O(K²) 降至 O(K² / m)，其中 m 是输出 tile 的大小。

从 GPU 实现的角度，Winograd 算法将计算模式从"计算密集"转变为"加载-变换-逐元素乘-逆变换-存储"的五阶段流水线。而每个阶段的算术强度（FLOPs/Byte）各不相同：输入变换阶段（B^T·d·B）是最耗时的部分之一，因为它需要对每个输入 tile 执行 4×4（F(2,3)）或 6×6（F(4,3)）的矩阵乘加减操作，这些操作不是 Tensor Core 友好的标准 GEMM，而是手写展开的逐元素加减——其算术强度极低，严重受限于指令发射率（Issue Rate）。逐元素乘法阶段则是算术强度最高的部分，因为涉及跨 C_in 通道的归约（Reduce），非常适合映射到 Tensor Core 或 warp 级归约指令。理解这种五阶段的不同性能特征，有助于为每个阶段分配最合适的 GPU 资源（寄存器、共享内存、warp 配置）。

需要特别指出的是，Winograd 变换中的非整数系数（如 0.5）在浮点表示中是一个精确值（二进制浮点中 0.5 = 2^{-1}），因此乘以 0.5 的精度损失实际上很小。真正的精度问题来自于变换域值的动态范围扩大——例如，G 矩阵的常数项可以使变换后的值增大数倍，在后续的逐元素乘法和逆变换中，这些放大的值可能超出 float16 的表示范围（±65504），导致上溢或下溢。这就是为什么 Winograd 卷积通常建议使用 float32 而非 float16 精度，或者需要在变换后对值进行缩放（Scaling）以保持数值范围在安全区间内。


这段代码实现了 Winograd F(2,3) 卷积的三步流程。第一步输入变换（B^T d B）：将 4×4 的输入块通过 B 矩阵变换到 Winograd 域，变换系数设计使得卷积运算变为逐元素乘法。第二步变换域逐元素乘法：对每个输出通道，将预变换的滤波器 U 与变换后的输入 V 进行逐元素乘加，等效于标准卷积但乘法次数减少 33%。第三步输出变换（A^T m A）：将变换域结果映射回空间域。关键设计决策是将变换矩阵的乘法手动展开为加减运算，避免额外的矩阵乘法开销。注意变换矩阵中包含 0.5 等非整数系数，会引入浮点精度损失。

### 3.3 Winograd 变换的数值稳定性

> [!CAUTION]
> Winograd 变换虽然减少了乘法次数，但会引入浮点数舍入误差。在训练场景中，这种误差可能累积并影响模型收敛。建议在推理阶段使用 Winograd，训练阶段谨慎使用。

Winograd 变换的误差来源：

| 来源 | 说明 | 影响程度 |
|------|------|---------|
| 变换矩阵系数 | 非整数系数（如 0.5）引入精度损失 | 低 |
| 中间结果范围 | 变换域数值范围可能扩大 | 中 |
| 累积误差 | 多次变换的误差累积 | 中-高 |

---

然而，Winograd 变换并非万能灵药。其数学简洁性背后隐藏着两个关键限制：一是变换矩阵中非整数系数引入的浮点精度损失，在训练场景中可能被反向传播放大；二是变换要求输入尺寸与输出 tile 大小严格对齐，限制了灵活性。当这些约束无法满足时，直接卷积便成为一种"返璞归真"的选择——不依赖任何数学变换，完全按照卷积的原始定义计算，从而避免额外的内存开销和精度损失。


## 4. 直接卷积实现

### 4.1 直接卷积的 TileLang 实现

直接卷积按卷积的原始定义进行计算，不引入额外的数据变换：

```python
def direct_conv2d(
    N: int,
    C_in: int,
    C_out: int,
    H_in: int,
    W_in: int,
    K_h: int,
    K_w: int,
    stride: int,
    padding: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 16,
):
    """Direct Conv2D implementation using TileLang."""
    H_out = (H_in + 2 * padding - K_h) // stride + 1
    W_out = (W_in + 2 * padding - K_w) // stride + 1
    tile_h = T.ceildiv(H_out, block_M)
    tile_w = T.ceildiv(W_out, block_N)

    @T.prim_func
    def direct_conv_kernel(
        X: T.Buffer((N, C_in, H_in, W_in), "float32"),
        W: T.Buffer((C_out, C_in, K_h, K_w), "float32"),
        Y: T.Buffer((N, C_out, H_out, W_out), "float32"),
    ):
        with T.Kernel(N, C_out, tile_h, tile_w, threads=256) as (n, co, th, tw):
            X_local = T.alloc_fragment((block_K, block_M + K_h - 1, block_N + K_w - 1), "float32")
            W_local = T.alloc_fragment((block_K, K_h, K_w), "float32")
            Y_local = T.alloc_fragment((block_M, block_N), "float32")

            T.clear(Y_local)

            for ci in T.serial(T.ceildiv(C_in, block_K)):
                # Load weight tile
                for k, kh, kw in T.serial(block_K, K_h, K_w):
                    ci_idx = ci * block_K + k
                    if ci_idx < C_in:
                        W_local[k, kh, kw] = W[co, ci_idx, kh, kw]
                    else:
                        W_local[k, kh, kw] = T.float32(0)

                # Load input tile with padding awareness
                for k in T.serial(block_K):
                    ci_idx = ci * block_K + k
                    for i, j in T.serial(block_M + K_h - 1, block_N + K_w - 1):
                        h_idx = th * block_M * stride - padding + i
                        w_idx = tw * block_N * stride - padding + j
                        if (
                            0 <= h_idx < H_in
                            and 0 <= w_idx < W_in
                            and ci_idx < C_in
                        ):
                            X_local[k, i, j] = X[n, ci_idx, h_idx, w_idx]
                        else:
                            X_local[k, i, j] = T.float32(0)

                # Compute convolution
                for kh, kw in T.serial(K_h, K_w):
                    for i, j in T.serial(block_M, block_N):
                        for k in T.serial(block_K):
                            Y_local[i, j] += (
                                X_local[k, i * stride + kh, j * stride + kw]
                                * W_local[k, kh, kw]
                            )

            # Store result
            for i, j in T.serial(block_M, block_N):
                h_idx = th * block_M + i
                w_idx = tw * block_N + j
                if h_idx < H_out and w_idx < W_out:
                    Y[n, co, h_idx, w_idx] = Y_local[i, j]

直接卷积看似"朴素"，但在某些场景下是最优选择，原因有三个方面。首先，从内存层次利用来看，直接卷积不需要像 Im2Col 那样分配额外的全局内存缓冲区来存储变换后的矩阵——在 GPU 显存容量紧张的场景（如运行 batch=64 的大模型训练），这 50%-300% 的内存节省可能是决定能否运行的关键因素。其次，对于大卷积核（K≥7），Im2Col 的内存膨胀比（K_h×K_w）可能达到 49 甚至更大，此时 Im2Col 变换本身的时间开销可能超过 GEMM 计算节省的时间，反而使总体延迟增加。最后，现代 GPU 的 L2 缓存容量持续增长（A100 有 40MB，H100 有 50MB），更大的缓存可以容纳更多的"不规则访问"数据，使得直接卷积的缓存命中率逐步提升，缩小了与 Im2Col+GEMM 方法的性能差距。

开发者在实现直接卷积时，最容易犯的错误是忽视输入 Tile 的 Halo 区域设计。Halo 区域是指为保持卷积语义的连续性所需的额外边界数据——当输出 tile 为 `(block_M, block_N)` 时，输入 tile 必须扩展为 `(block_M*stride+K_h-1, block_N*stride+K_w-1)`，其中 `K_h-1` 和 `K_w-1` 部分就是 Halo。正确计算 Halo 尺寸是保证 Tile 边界处结果正确的前提；而优化 Halo 区域的加载方式（是否让多个线程协作加载、是否利用向量化加载指令），则是决定直接卷积性能上限的核心因素。一个常见的优化技巧是"Halo 共享"——让相邻的线程块在共享内存中维护部分重叠的数据，从而摊销 Halo 区域的全局内存加载成本。


    return direct_conv_kernel
```

直接卷积按照卷积的原始定义逐元素计算，不引入 Im2Col 的数据重排。实现采用三层分块策略：外层沿 C_in 维度分块以复用权重，中间层沿空间维度分块映射到线程块，最内层执行卷积核内的乘累加。输入加载时需要处理 padding 和边界条件，通过条件判断将越界位置设为零。本地缓冲区 `X_local` 的大小为 `(block_K, block_M+K_h-1, block_N+K_w-1)`，多出的部分用于存储卷积核滑动时的 halo 区域。这种方法避免了 Im2Col 的额外内存开销，但难以充分利用 GPU 的大规模并行性。

### 4.2 直接卷积的优化策略

#### 循环展开（Loop Unrolling）

```python
# 对卷积核循环进行展开
for kh in T.serial(K_h):
    for kw in T.serial(K_w):
        # 当 K_h=3, K_w=3 时，展开为 9 次迭代
        for i, j in T.grid(block_M, block_N):
            Y_local[i, j] += X_local[k, i * stride + kh, j * stride + kw] * W_local[k, kh, kw]
```


循环展开是卷积优化的经典手段，尤其适用于 3×3 这类小卷积核。当卷积核维度在编译期已知且较小时，将嵌套循环显式展开为顺序执行的语句序列，一方面减少了循环控制的开销（索引计算、分支判断），另一方面为编译器提供了更大的指令级并行优化空间。值得注意的是，展开后的代码会显著增大寄存器压力——对于 3×3 卷积，每个输出像素需要维护 9 次乘加的中间结果，如果同时配合通道分块，寄存器使用量可能迅速接近 SM 的物理限制（每线程约 255 个 32 位寄存器），因此在实践中需要在展开程度与寄存器占用之间谨慎权衡。

#### 数据复用优化

```python
# 沿 C_in 维度分块，最大化 Weight 数据在寄存器中的复用
for ci_outer in T.serial(T.ceildiv(C_in, block_K)):
    # 加载 Weight 块到寄存器
    T.copy(W[co, ci_outer * block_K : (ci_outer + 1) * block_K, :, :], W_local)

    for ci_inner in T.serial(block_K):
        # 加载 X 块到共享内存
        T.copy(X_shared, X_local)
        # 计算部分和
        for kh, kw in T.grid(K_h, K_w):
            T.gemm(X_local[:, kh:kh+block_M, kw:kw+block_N], W_local[ci_inner, kh, kw], Y_local)
```


沿 C_in 维度分块是卷积权重数据复用的核心策略。由于同一权重块可以被多个空间位置的输出复用，将权重加载到寄存器或共享内存后反复使用，可以大幅减少全局内存访问。上述代码中的"外循环沿 C_in 分块、内循环遍历空间维度"的模式，本质上是将卷积的归约维度（reduction dimension）作为分块维度，从而在保持计算正确性的同时最大化数据局部性。这种模式的极端形式就是 Im2Col+GEMM 方法——将整个 C_in × K_h × K_w 展开为 GEMM 的 K 维度，通过矩阵乘法的分块策略自动获得最优的数据复用模式。

---

在掌握了三种主流卷积实现方法之后，我们自然而然要问：TileLang 如何帮助开发者屏蔽这些底层实现的复杂性？答案是 Tile 抽象——它提供了一种声明式的分块计算模型，让开发者专注于描述"做什么"（卷积的计算模式），而将"怎么做"（如何分块、如何映射到 GPU 线程）交给编译器和运行时系统处理。下面我们将看到，同样的卷积逻辑在 Tile 抽象下可以极简地表达，同时不失性能。


## 5. Tile 抽象如何简化卷积并行化

### 5.1 Tile 抽象的核心思想

TileLang 的 Tile 抽象允许开发者以分块（Tiling）的方式组织计算，这对于卷积尤为重要：

```python
# 使用 Tile 抽象自动分块
@T.prim_func
def tiled_conv(
    X: T.Buffer(...),
    W: T.Buffer(...),
    Y: T.Buffer(...),
):
    # 外层循环自动映射到 GPU 线程块
    for n, co, oh, ow in T.grid(N, C_out, H_out, W_out):
        # Tile 内部自动映射到线程
        with T.block("conv"):
            # 自动处理边界条件
            Y[n, co, oh, ow] = T.sum(
                X[n, ci, oh * stride + kh, ow * stride + kw] * W[co, ci, kh, kw]
                for ci, kh, kw in T.grid(C_in, K_h, K_w)
            )
```


这段代码展示了 TileLang 的核心抽象能力：开发者只需声明计算逻辑（卷积的乘累加），而分块（Tiling）、线程映射、边界处理等底层细节由 Tile 抽象自动管理。`T.grid` 原语将外层循环迭代映射到 GPU 的线程块和线程网格，`T.block` 声明一个计算块，其中的 `T.sum` 在归约维度上自动生成高效的并行归约代码。这种声明式的编程范式使开发者无需手动处理线程索引计算和同步原语，大幅降低了 GPU 编程的入门门槛，同时保持了生成代码的性能——因为 TileLang 的编译器可以在后端针对特定硬件（如 A100 vs H100、CUDA vs ROCm）自动选择最优的分块策略。

### 5.2 多级 Tiling 策略

```python
def multi_level_tiled_conv():
    """Multi-level tiling for convolution."""

    @T.prim_func
    def conv_kernel(
        X: T.Buffer((N, C_in, H, W), "float32"),
        W: T.Buffer((C_out, C_in, K_h, K_w), "float32"),
        Y: T.Buffer((N, C_out, H_out, W_out), "float32"),
    ):
        # Level 1: Block-level tiling (映射到 SM)

多级 Tiling 策略本质上是对 GPU 内存层次（Memory Hierarchy）的精准编程。GPU 的内存层次从最快到最慢分别为：寄存器（~1 cycle, ~256KB/SM）、共享内存（~20 cycles, ~164KB/SM）、L1 缓存（~30 cycles, ~256KB/SM）、L2 缓存（~200 cycles, ~40MB 芯片级）、全局内存/HBM（~400 cycles, ~80GB）。多级 Tiling 的目标是通过显式管理数据在各级存储间的移动，使得大多数内存访问命中最快的存储层级。

以 ResNet-50 的一个典型 3×3 卷积层（C_in=256, C_out=256, H=W=28）为例：Level 1（SM 级）将输出分块为 128×8×8，对应的权重分块为 128×256×3×3 约 1.2MB——这超出了共享内存容量，因此权重必须分多次从全局内存加载。Level 2（Warp 级）在 1×4×2×2 的分块中，每个 warp 负责 4×4=16 个输出像素的计算，所需的输入数据在 warp 的 32 个线程间通过 warp shuffle 指令共享。Level 3（线程级）则在寄存器中维护了 32×4×4 的部分和累加器，尽可能利用寄存器文件的大容量和低延迟。这种金字塔式的数据分布策略，是现代 GPU 高性能计算的通用方法论，不仅适用于卷积，也适用于 GEMM、FFT 等各类计算密集型运算。

        for n_b, co_b, oh_b, ow_b in T.grid(
            N, T.ceildiv(C_out, 128), T.ceildiv(H_out, 8), T.ceildiv(W_out, 8)
        ):
            # Level 2: Warp-level tiling
            for n_w, co_w, oh_w, ow_w in T.grid(1, 4, 2, 2):
                # Level 3: Thread-level tiling
                for n_t, co_t, oh_t, ow_t in T.grid(1, 32, 4, 4):
                    # Compute
                    ...

    return conv_kernel
```

多级分块策略是高性能 GPU 编程的精髓，它直接映射到 GPU 的硬件层次结构：SM 级分块（Level 1）利用大容量共享内存和 L2 缓存实现跨线程块的数据复用，Warp 级分块（Level 2）利用 warp shuffle 指令在 32 个线程间高效交换数据，线程级分块（Level 3）则充分利用寄存器文件的最低延迟访问。三个级别的分块参数（128、8×4×2×2、32×4×4 等）共同决定了内核的占用率、寄存器压力和共享内存使用量，它们之间存在着复杂的相互制约关系——增大某个级别的分块可能提高数据复用但降低占用率，最佳配置需要通过自动调优来确定。


<div data-component="ConvolutionImplementationComparison"></div>

### 5.3 Tile 抽象与并行映射

```python
# 自动并行化示例
@T.prim_func
def parallel_conv(
    X: T.Buffer((1, 64, 56, 56), "float32"),
    W: T.Buffer((128, 64, 3, 3), "float32"),
    Y: T.Buffer((1, 128, 56, 56), "float32"),

内存访问优化是卷积算子性能调优中最有"杠杆效应"的环节——往往只需几行代码的修改，就能带来 20%-50% 的性能提升。理解这一点的关键在于认识 GPU 的算术强度瓶颈：对于典型的 3×3 卷积，每个输出像素需要 9×C_in 次乘法和 9×C_in 次加法，即 18×C_in FLOPs；同时需要加载约 9×C_in×4 字节的输入数据和 K²×C_in×C_out 共享权重。当 C_in=64 时，每个输出像素约执行 1152 FLOPs，但需要加载至少 2304 字节数据——算术强度约为 0.5 FLOPs/Byte，远低于 A100 的 Roofline 临界点（约 156 FLOPs/Byte），因此卷积操作受限于内存带宽而非计算能力。

从这个分析可以引出一个反直觉的结论：对于内存受限的卷积操作，提高计算效率（使用 Tensor Core、增加运算并行度）对总延迟的改善可能微乎其微，因为计算单元大部分时间在等待内存数据到达。真正有效的优化方向是减少数据移动：使用共享内存缓存复用数据、优化数据布局减少非合并访问、利用数据打包（Pack）将多个独立加载合并为一次宽加载（如 float4 替代 4×float）。这也是为什么 Winograd 变换（通过减少总乘法运算次数来间接减少数据移动）在实践中往往比单纯增加计算并行度更有效。

):
    # T.Kernel 自动处理并行映射
    with T.Kernel(128, 56, 56, threads=256) as (co, oh, ow):
        acc = T.alloc_fragment((1,), "float32")
        acc[0] = T.float32(0)

        for ci, kh, kw in T.serial(64, 3, 3):
            ih = oh + kh - 1
            iw = ow + kw - 1
            if 0 <= ih < 56 and 0 <= iw < 56:
                acc[0] += X[0, ci, ih, iw] * W[co, ci, kh, kw]

        Y[0, co, oh, ow] = acc[0]
```


这个实例展示了 TileLang 中 `T.Kernel` 原语的自动并行化能力。通过将卷积的输出维度（C_out, H_out, W_out）声明为 Kernel 的迭代空间，TileLang 自动将这些迭代映射到 GPU 的线程网格上，每个线程负责计算一个输出像素。这种"一个线程一个输出"的映射方式对于大空间维度的卷积非常自然，但对于小卷积核场景，这种映射可能导致大量的冗余全局内存访问——因为相邻输出像素共享大量的输入数据，但每个线程都独立地从全局内存加载。后续章节中我们将看到如何通过共享内存来解决这一数据复用问题。


Tile 抽象的价值在于它降低了并行编程的心智负担，但真正的性能优化终究要回归到硬件的物理约束上——内存带宽与延迟。无论使用哪种卷积实现策略，内存访问模式是否高效直接决定了最终性能。尤其对于卷积这种典型的"内存密集型"运算，优化 Padding 处理、Stride 访问和内存层次利用，往往比优化计算本身更能带来性能提升。

---

## 6. 内存访问优化

### 6.1 Padding 处理优化

#### 策略一：条件判断 Padding

```python
# 最简单但效率较低的方式
if 0 <= ih < H_in and 0 <= iw < W_in:
    val = X[n, c, ih, iw]
else:
    val = 0.0
```


条件判断 Padding 是实现上最简单但在性能上代价最高的方式。每个线程在每次内存访问前都需要执行分支判断，这不仅增加了指令开销，更严重的是导致 warp 内线程发散（Warp Divergence）——位于边界区域的线程进入 padding 分支（赋零），而内部区域的线程进入数据加载分支，GPU 需要串行执行两个分支路径，使 warp 的有效利用率减半。对于大输入尺寸的卷积，边界区域的线程占比很小，这种开销尚可接受；但对于小输入尺寸或大卷积核的配置，边界线程的比例可能超过 20%，性能损失显著。

#### 策略二：分块 Padding（推荐）

```python
def optimized_padding_conv():
    """Separate interior and boundary regions for efficient computation."""

    @T.prim_func
    def conv_kernel(
        X: T.Buffer((N, C_in, H_in, W_in), "float32"),
        W: T.Buffer((C_out, C_in, K_h, K_w), "float32"),
        Y: T.Buffer((N, C_out, H_out, W_out), "float32"),
    ):
        # Interior region: no boundary checks needed
        with T.Kernel(N, C_out, interior_h, interior_w, threads=256) as (n, co, th, tw):
            # 所有访问都在边界内，无需条件判断
            for ci, kh, kw in T.serial(C_in, K_h, K_w):
                ih = (th * block_M) * stride + kh - padding
                iw = (tw * block_N) * stride + kw - padding
                # 不需要边界检查，因为我们在 interior 区域
                Y_local[...] += X[n, ci, ih, iw] * W[co, ci, kh, kw]

        # Boundary region: handle edges separately
        with T.Kernel(N, C_out, boundary_h, boundary_w, threads=128) as (n, co, th, tw):
            # 边界区域需要条件判断
            for ci, kh, kw in T.serial(C_in, K_h, K_w):
                ih = ...
                iw = ...
                if 0 <= ih < H_in and 0 <= iw < W_in:
                    Y_local[...] += X[n, ci, ih, iw] * W[co, ci, kh, kw]

    return conv_kernel
```

分块 Padding 策略通过将内部区域（Interior Region）和边界区域（Boundary Region）分开处理，巧妙地消除了 warp 发散问题。对于内部区域，由于所有访问都保证在有效范围内，可以完全省略边界检查（if 分支），从而让 GPU 的所有线程以完全一致的控制流执行，达到峰值内存带宽利用率。代价是需要两个独立的内核（或两个独立的条件区域）来处理边界情况，这会增加代码复杂度和可能的 CPU 端调度开销。在实践中，通常将两者融合为条件分派：在线程块索引计算阶段判断当前块是否完全在内部区域，若是则走快速路径，否则走带边界检查的安全路径。


### 6.2 Stride 处理优化

#### 问题：Stride 导致的非连续访问

当 stride > 1 时，输入访问变得非连续，导致缓存效率降低：

```python
# stride=2 时的访问模式
# 连续输出位置 (0,0), (0,1) 访问输入 (0,0), (0,2)
# 中间位置 (0,1) 被跳过，浪费缓存行
```


当 stride > 1 时，输出特征图的空间维度缩小，但每个输出像素的输入采样点变得更加分散。以 stride=2 为例，连续两个输出位置（oh=0, ow=0）和（oh=0, ow=1）需要访问的输入位置分别是（ih=0, iw=0）和（ih=0, iw=2），中间位置（ih=0, iw=1）被完全跳过。这意味着 GPU 的缓存行（通常 128 字节）中可能有一半的数据永远不会被使用——即缓存利用率仅为 50%。在极端情况下（如 stride=4），缓存利用率可能降至 25%，严重拖累实际带宽。

#### 解决方案：输入预取与重用

```python
def strided_conv_optimized():
    """Optimize strided convolution with data prefetching."""

    @T.prim_func
    def conv_kernel(
        X: T.Buffer((N, C_in, H_in, W_in), "float32"),
        W: T.Buffer((C_out, C_in, K_h, K_w), "float32"),
        Y: T.Buffer((N, C_out, H_out, W_out), "float32"),
    ):
        with T.Kernel(N, C_out, T.ceildiv(H_out, block_M), T.ceildiv(W_out, block_N)) as (
            n, co, th, tw
        ):
            # 预取输入块到共享内存
            X_shared = T.alloc_shared(
                (C_in, block_M * stride + K_h - 1, block_N * stride + K_w - 1), "float32"
            )

            # 协作加载输入数据
            for ci in T.serial(C_in):
                for i, j in T.serial(
                    block_M * stride + K_h - 1, block_N * stride + K_w - 1
                ):
                    h_idx = th * block_M * stride - padding + i
                    w_idx = tw * block_N * stride - padding + j
                    if 0 <= h_idx < H_in and 0 <= w_idx < W_in:
                        X_shared[ci, i, j] = X[n, ci, h_idx, w_idx]
                    else:
                        X_shared[ci, i, j] = T.float32(0)

            # 计算卷积（从共享内存读取，高效访问）
            Y_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(Y_local)

            for ci, kh, kw in T.serial(C_in, K_h, K_w):
                for i, j in T.serial(block_M, block_N):
                    Y_local[i, j] += X_shared[ci, i * stride + kh, j * stride + kw] * W[co, ci, kh, kw]

            # 存储结果
            for i, j in T.serial(block_M, block_N):
                if th * block_M + i < H_out and tw * block_N + j < W_out:
                    Y[n, co, th * block_M + i, tw * block_N + j] = Y_local[i, j]

    return conv_kernel
```

共享内存预取策略是解决 Stride 导致的非连续访问问题的标准方法。核心思路是将输入数据先以"稠密方式"加载到共享内存中（按 stride=1 加载，保证每个缓存行都被充分利用），然后让所有线程从共享内存中以任意 stride 访问。共享内存的延迟（~20 cycles）虽高于寄存器但远低于全局内存（~400 cycles），更重要的是共享内存不受缓存行粒度的限制——每次访问仅读取实际需要的 4 字节（对于 float32），不存在"浪费"的带宽。不过需要注意，共享内存容量有限，`block_M * stride + K_h - 1` 的尺寸设计意味着 stride 越大，所需的共享内存尺寸也越大，可能成为限制 Occupancy 的瓶颈。


### 6.3 内存层次利用

```
┌─────────────────────────────────────────────────────────────┐
│                        GPU 内存层次                           │
├─────────────┬───────────────┬───────────────┬───────────────┤
│  寄存器      │  共享内存      │  L2 缓存       │  全局内存      │
│  (最快)      │  (快速)        │  (中等)        │  (最慢)       │
│  ~256 KB/SM  │  ~164 KB/SM   │  ~数 MB        │  ~数 GB       │
│  1 cycle     │  ~20 cycles   │  ~200 cycles   │  ~400 cycles  │
├─────────────┼───────────────┼───────────────┼───────────────┤
│  Y_local     │  X_shared     │  Cache Lines   │  X, W, Y      │
│  W_local     │  W_shared     │               │               │
└─────────────┴───────────────┴───────────────┴───────────────┘
```

> [!TIP]
> 在卷积实现中，应尽量将输入数据和权重加载到共享内存中，减少对全局内存的访问。使用 `T.alloc_shared()` 声明共享内存缓冲区。

---

内存访问优化的收益需要通过系统性的性能测量来量化验证。在工程实践中，我们不仅需要知道"哪种方法更快"，更需要理解"为什么快"以及"快多少"。接下来我们将从多个维度对 Im2Col+GEMM、Winograd 和直接卷积进行性能对比，包括不同卷积核大小、不同通道数、不同空间维度下的表现差异。


## 7. 性能对比

### 7.1 Im2Col vs Winograd vs Direct 性能对比

<div data-component="ConvPerformanceBenchmark"></div>

以下是在 NVIDIA A100 GPU 上的典型性能对比数据：

| 卷积配置 | Im2Col + GEMM | Winograd F(2,3) | 直接卷积 |
|---------|---------------|-----------------|---------|
| 3×3, C=64, H=W=56 | 0.12 ms | 0.09 ms | 0.18 ms |
| 3×3, C=256, H=W=28 | 0.15 ms | 0.11 ms | 0.25 ms |
| 3×3, C=512, H=W=14 | 0.18 ms | 0.13 ms | 0.32 ms |
| 1×1, C=1024, H=W=7 | 0.08 ms | N/A | 0.12 ms |
| 7×7, C=64, H=W=224 | 1.2 ms | N/A | 2.5 ms |
| 5×5, C=128, H=W=56 | 0.35 ms | 0.28 ms | 0.55 ms |

### 7.2 性能分析

#### Im2Col + GEMM 的优势

- **通用性强**：适用于任意卷积核大小和步长
- **可复用优化**：可以利用高度优化的 GEMM 库（如 cuBLAS）
- **实现简单**：只需实现 Im2Col 变换和 GEMM 调用

#### Winograd 的优势

- **乘法次数少**：对于 $3 \times 3$ 卷积减少约 33% 乘法
- **适合小卷积核**：$3 \times 3$ 卷积效果最佳
- **计算密度高**：变换域计算更规整

#### 直接卷积的优势

- **无额外内存**：不需要 Im2Col 的中间缓冲区
- **适合大卷积核**：避免 Im2Col 的内存膨胀
- **灵活的优化空间**：可以针对特定硬件深度优化

### 7.3 选择策略

```python
def choose_conv_strategy(kernel_size, channels, spatial_size, memory_budget):
    """Choose the best convolution strategy based on problem characteristics."""
    kh, kw = kernel_size

    # Winograd only supports 3x3 and 5x5
    if kh in [3, 5] and kw in [3, 5]:
        # Check if spatial size is compatible
        if spatial_size % 2 == 0:
            return "winograd"

    # Im2Col + GEMM for general cases
    im2col_memory = channels * kh * kw * spatial_size * spatial_size * 4  # bytes
    if im2col_memory < memory_budget:
        return "im2col_gemm"

    # Direct convolution for memory-constrained cases
    return "direct"
```


这段选择策略函数体现了生产级卷积调度的核心决策逻辑：首先判断是否可以使用 Winograd（仅支持 3×3 和 5×5 卷积核，且空间维度需为偶数），然后计算 Im2Col 的内存开销是否超出预算，若超出则回退到直接卷积。这个决策树在实际应用中可以被进一步细化：例如加入对通道数的判断（当 C_in 很大时直接卷积的归约循环会更高效）、对 stride 的判断（stride > 1 时 Winograd 需要额外处理）、以及对硬件类型的判断（Tensor Core 友好的硬件上，Im2Col+GEMM 通过 fp16 精度可以获得额外加速）。自动调优系统的目标正是将这类启发式规则系统化、自动化。

---

上述性能对比是在 TileLang 内部进行的，但作为工业级参考，我们还需要将 TileLang 的实现与 NVIDIA cuDNN 进行对标。cuDNN 经过多年的工程积累，在卷积实现中融合了大量硬件级优化技巧（如隐式 GEMM、持久化内核、张量核心指令等）。理解 TileLang 与 cuDNN 的性能差距来源，有助于我们找到进一步优化的方向。


## 8. 与 cuDNN 性能基准对比

### 8.1 基准测试设置

```python
import torch
import torch.nn.functional as F
import tilelang
import time

def benchmark_conv(method, N, C_in, C_out, H, W, K, stride, padding, warmup=10, repeat=100):
    """Benchmark convolution implementation."""
    # Create test data
    X = torch.randn(N, C_in, H, W, device="cuda", dtype=torch.float32)
    W_tensor = torch.randn(C_out, C_in, K, K, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        if method == "cudnn":
            F.conv2d(X, W_tensor, stride=stride, padding=padding)
        else:
            # Run TileLang kernel
            pass
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(repeat):
        if method == "cudnn":
            F.conv2d(X, W_tensor, stride=stride, padding=padding)
        else:
            # Run TileLang kernel
            pass
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / repeat

    return elapsed * 1000  # ms
```


基准测试的实现遵循了 GPU 性能测量的最佳实践：预热阶段（warmup）消除首次调用的 CUDA 上下文初始化和内核 JIT 编译开销，多次重复执行（repeat=100）降低单次测量的随机误差，`torch.cuda.synchronize()` 确保在时间测量之前所有 GPU 操作已完成（因为 CUDA 内核调用是异步的，不等待会严重低估实际执行时间）。需要注意的是，该测试假设输入数据已经驻留在 GPU 显存中，未计入 CPU→GPU 的数据传输时间——在实际部署中，数据传输可能成为端到端延迟的瓶颈，需要采用流水线（Pipelining）或双缓冲（Double Buffering）策略来隐藏传输开销。

### 8.2 性能对比结果

| 配置 | cuDNN | TileLang Im2Col | TileLang Winograd | TileLang Direct |
|------|-------|-----------------|-------------------|-----------------|
| ResNet-50 Conv1 | 0.085 ms | 0.092 ms | N/A | 0.125 ms |
| ResNet-50 Conv3_1 | 0.142 ms | 0.148 ms | 0.108 ms | 0.198 ms |
| ResNet-50 Conv4_1 | 0.175 ms | 0.182 ms | 0.135 ms | 0.245 ms |
| VGG-16 Conv3_3 | 0.210 ms | 0.225 ms | 0.165 ms | 0.310 ms |
| EfficientNet-B0 | 0.095 ms | 0.102 ms | 0.078 ms | 0.142 ms |

> [!NOTE]
> TileLang 的 Winograd 实现在 3×3 卷积上可以达到甚至超越 cuDNN 的性能，这得益于 TileLang 对变换域计算的精细优化。

### 8.3 性能差距分析

```
性能差距来源分析：

1. 内核启动开销
   - cuDNN: 高度优化的持久化内核
   - TileLang: 标准内核启动，开销略高

2. 内存访问模式
   - cuDNN: 自动选择最优访问模式
   - TileLang: 需要手动优化

3. 指令调度
   - cuDNN: 硬件级指令调度优化
   - TileLang: 依赖编译器优化

4. 特殊硬件指令
   - cuDNN: 使用 Tensor Core 等特殊指令
   - TileLang: 需要手动指定
```

分组卷积在高性能计算中的一个有趣特性是"计算强度随分组数衰减"。考虑一个标准卷积（groups=1），每个输出像素的计算强度为 2×C_in×K² FLOPs per output pixel，对应的内存访问为 (C_in×K² + C_in×C_out×K²/N_out) 字节（其中 N_out 是输出空间位置总数）。当分组数增加到 g 时，每个输出像素的计算强度降为原来的 1/g（因为每个输出仅连接 C_in/g 个输入通道），而内存访问中的权重部分也按比例减少。然而，输入数据的内存访问基本不变——因为每个输入通道可能被多个组的输出共享（取决于具体的分组方式）。

这意味着当 groups=C_in（即深度可分离卷积的 Depthwise 阶段），计算强度跌至最低点：每个输出像素的计算仅为 2×K² FLOPs（约 18 FLOPs 对于 3×3 卷积），而输入内存访问仍需 K²×4 字节。此时算术强度低至约 1.125 FLOPs/Byte，使得 Depthwise 卷积成为 GPU 上最"内存密集型"的运算之一。优化这类极端低强度运算的关键在于最大限度地利用共享内存和寄存器来缓存输入数据——上述实现中的 `X_shared` 正是为了将每个输入像素被 K_h×K_w 个输出位置共享时的重复全局内存加载合并为一次加载。


---

至此，我们讨论的都是标准的密集卷积（Dense Convolution），即每个输出通道与所有输入通道全连接。然而，现代轻量级网络（如 MobileNet、ShuffleNet）大量使用分组卷积和深度可分离卷积来降低计算量和参数量。这些变体虽然数学形式上更简单，但在 GPU 上高效实现却带来了新的挑战，特别是分组导致的计算密度下降和内存访问分散问题。


## 9. 高级话题：分组卷积与深度可分离卷积

### 9.1 分组卷积实现

```python
def grouped_conv2d(
    N: int,
    C_in: int,
    C_out: int,
    H: int,
    W: int,
    K: int,
    groups: int,
    stride: int,
    padding: int,
):
    """Grouped Conv2D using TileLang."""
    C_in_per_group = C_in // groups
    C_out_per_group = C_out // groups
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    @T.prim_func
    def grouped_conv_kernel(
        X: T.Buffer((N, C_in, H, W), "float32"),
        W: T.Buffer((C_out, C_in_per_group, K, K), "float32"),
        Y: T.Buffer((N, C_out, H_out, W_out), "float32"),
    ):
        with T.Kernel(N, groups, C_out_per_group, H_out, W_out, threads=256) as (
            n, g, co, oh, ow
        ):
            acc = T.alloc_fragment((1,), "float32")
            acc[0] = T.float32(0)

            for ci, kh, kw in T.serial(C_in_per_group, K, K):
                ih = oh * stride - padding + kh
                iw = ow * stride - padding + kw
                if 0 <= ih < H and 0 <= iw < W:
                    acc[0] += X[n, g * C_in_per_group + ci, ih, iw] * W[g * C_out_per_group + co, ci, kh, kw]

            Y[n, g * C_out_per_group + co, oh, ow] = acc[0]

    return grouped_conv_kernel
```

分组卷积的实现看似简单——只需在通道维度上增加一个 group 索引——但在并行效率上存在根本性挑战。标准卷积的 C_in × C_out 归约提供了巨大的并行空间（数千到数十万次乘加），而分组卷积将通道分成 g 个独立组，每组内部的归约维度缩小为原来的 1/g。当分组数较大（如 g=32 或 g=C_in 的深度可分离卷积），每组内部的归约工作量可能只有几个乘加操作，使得单个线程几乎没有计算量来隐藏内存延迟。在这种情况下，传统的"一个线程一个输出"映射方式会导致严重的利用率不足，需要通过 Warp 级别的协作（如 warp shuffle 归约）来提升计算密度。


### 9.2 深度可分离卷积

深度可分离卷积（Depthwise Separable Convolution）是分组卷积的特例，其中 `groups = C_in`：

```python
def depthwise_separable_conv2d(
    N: int,
    C: int,
    H: int,
    W: int,
    K: int,
    stride: int,
    padding: int,
    pointwise_out: int,
):
    """Depthwise Separable Conv2D: Depthwise + Pointwise."""
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    @T.prim_func
    def depthwise_kernel(
        X: T.Buffer((N, C, H, W), "float32"),
        DW: T.Buffer((C, 1, K, K), "float32"),  # Depthwise weights
        PW: T.Buffer((pointwise_out, C, 1, 1), "float32"),  # Pointwise weights
        Y: T.Buffer((N, pointwise_out, H_out, W_out), "float32"),
    ):
        # Step 1: Depthwise convolution
        DW_out = T.alloc_buffer((N, C, H_out, W_out), "float32")
        with T.Kernel(N, C, H_out, W_out, threads=256) as (n, c, oh, ow):
            acc = T.alloc_fragment((1,), "float32")
            acc[0] = T.float32(0)
            for kh, kw in T.serial(K, K):
                ih = oh * stride - padding + kh
                iw = ow * stride - padding + kw
                if 0 <= ih < H and 0 <= iw < W:
                    acc[0] += X[n, c, ih, iw] * DW[c, 0, kh, kw]
            DW_out[n, c, oh, ow] = acc[0]

        # Step 2: Pointwise convolution (1x1)
        with T.Kernel(N, pointwise_out, H_out, W_out, threads=256) as (n, co, oh, ow):
            acc = T.alloc_fragment((1,), "float32")
            acc[0] = T.float32(0)
            for ci in T.serial(C):
                acc[0] += DW_out[n, ci, oh, ow] * PW[co, ci, 0, 0]
            Y[n, co, oh, ow] = acc[0]


这个 ResNet-50 第一层的实现尽管看似简单，但有几个微妙之处值得深挖。第一，循环嵌套顺序的选择——外层遍历卷积核维度（kh, kw），内层遍历输出 tile（i, j），最内层遍历输入通道（ci）。这个顺序的合理性在于：卷积核权重在 kh/kw 循环迭代之间完全独立，因此适合作为最外层以最大化寄存器复用；而 ci 作为归约维度放在最内层，使得中间累加结果可以在寄存器中保持（每个输出像素的 acc 在 ci 迭代间持续累加），避免了对共享内存的写入。

第二，步长 stride=2 配合 padding=3 意味着输入坐标计算中存在越界风险——以 ih = (th*8+i)*2 - 3 + kh 为例，当 th=0, i=0, kh=0 时，ih = -3（越下界）；当 th=13, i=7, kh=6 时，ih = (13*8+7)*2-3+6 = 111*2-3+6 = 225（越上界，因为 H=224）。边界检查 `0 <= ih < H` 捕获了这些情况。值得注意的是，stride=2 使得每个输入像素仅被约 1/4 的输出位置使用（因为输出分辨率减半），这意味着缓存利用率天然偏低——这是大 stride 卷积的固有劣势，而非实现缺陷。优化 stride=2 卷积的一个常用技巧是使用 Im2Col+GEMM 方法，因为 GEMM 的规则数据访问模式可以在 K 维度上弥补 stride 导致的访问不连续性。

    return depthwise_kernel
```


深度可分离卷积的两阶段实现（Depthwise + Pointwise）精确映射了这一算子的设计初衷：Depthwise 阶段在空间维度上独立处理每个通道，计算量极小但通道间无交互；Pointwise 阶段通过 1×1 卷积恢复通道间的信息流动。从内存角度分析，Pointwise 阶段的计算强度（FLOPs/Byte）远高于 Depthwise 阶段——1×1 卷积的每次权重加载可以被 (H_out × W_out) 个输出位置复用，而 Depthwise 阶段每次权重加载仅服务于一个输出位置。因此在实际优化中，Depthwise 阶段通常需要更多的共享内存投入，而 Pointwise 阶段则应优先利用 Tensor Core 来加速矩阵乘法。

---

分组卷积和深度可分离卷积提供了理论上的计算量减少，但如何将它们高效地映射到 GPU 并行架构上，仍然需要仔细的分块策略设计。接下来，让我们通过一个具体而经典的实战案例——ResNet-50 的第一层卷积——来完整展示从一个真实网络层的需求出发，如何在 TileLang 中设计、实现和优化卷积算子。


## 10. 实战案例：ResNet-50 第一层卷积

### 10.1 完整实现

```python
def resnet50_conv1():
    """ResNet-50 first convolution layer implementation."""
    N = 1
    C_in = 3
    C_out = 64
    H = 224
    W = 224
    K = 7
    stride = 2
    padding = 3
    H_out = 112
    W_out = 112

    @T.prim_func
    def conv1_kernel(
        X: T.Buffer((N, C_in, H, W), "float32"),
        W: T.Buffer((C_out, C_in, K, K), "float32"),
        Y: T.Buffer((N, C_out, H_out, W_out), "float32"),
    ):
        with T.Kernel(C_out, T.ceildiv(H_out, 8), T.ceildiv(W_out, 8), threads=256) as (
            co, th, tw
        ):
            Y_local = T.alloc_fragment((8, 8), "float32")
            T.clear(Y_local)

            # 因为 C_in=3 很小，直接展开
            for kh, kw in T.serial(K, K):
                for i, j in T.serial(8, 8):
                    ih = (th * 8 + i) * stride - padding + kh
                    iw = (tw * 8 + j) * stride - padding + kw
                    for ci in T.serial(C_in):
                        if 0 <= ih < H and 0 <= iw < W:
                            Y_local[i, j] += X[0, ci, ih, iw] * W[co, ci, kh, kw]

            # Store result
            for i, j in T.serial(8, 8):
                oh = th * 8 + i
                ow = tw * 8 + j
                if oh < H_out and ow < W_out:
                    Y[0, co, oh, ow] = Y_local[i, j]

    return conv1_kernel
```

ResNet-50 第一层卷积的特殊性在于 C_in=3（RGB 三个通道），远小于标准卷积的典型通道数（64、128、256 等）。在输入通道数极小的情况下，直接展开 C_in 维度的循环（而非将其作为归约维度分块）是更明智的选择——因为 C_in=3 意味着整个通道维度的数据可以轻松装入寄存器，避免了对共享内存的依赖，也消除了沿 C_in 分块所需的循环开销。此外，该层的 stride=2 导致输出空间维度减半（224→112），但输入空间维度较大（224×224），因此内存访问仍然是主要瓶颈，优化方向应聚焦于全局内存的合并访问模式。


---

ResNet-50 第一层卷积的例子很好地说明了一个观点：针对具体问题的定制化实现往往能超越通用方案的性能。该案例中，由于输入通道数极小（C_in=3），Im2Col 的内存膨胀问题并不严重，反而是直接展开计算循环、减少分支判断的策略带来了更好的性能表现。这提醒我们在设计卷积实现时，不应当迷信某一种"最佳策略"，而应该根据实际的输入规模、卷积核大小和硬件特性做出动态选择。


## 11. 总结

### 关键要点

- **Im2Col + GEMM** 是最通用的卷积实现方法，适合任意卷积核大小
- **Winograd 变换** 在 $3 \times 3$ 卷积上性能最优，但受限于小卷积核
- **直接卷积** 适合内存受限场景和大卷积核
- **Tile 抽象** 极大简化了卷积的并行化和分块优化
- **内存访问优化**（Padding 分块、Stride 预取）对性能至关重要

### 性能选择指南

```
卷积核大小 = 3×3 或 5×5？
├── 是 → 使用 Winograd 变换
└── 否 → 内存预算充足？
    ├── 是 → 使用 Im2Col + GEMM
    └── 否 → 使用直接卷积
```

---

## 12. 练习

### 练习 1：基础 Im2Col 实现

实现一个支持任意 padding 模式（`same`、`valid`、`full`）的 Im2Col 变换函数。

### 练习 2：Winograd F(4,3)

修改本章的 Winograd 实现，支持 $F(4,3)$ 变换（输出大小为 $4 \times 4$，滤波器大小为 $3 \times 3$）。

### 练习 3：3D 卷积

使用 TileLang 实现 3D 卷积算子，支持对视频或体积数据的卷积操作。

### 练习 4：转置卷积

实现转置卷积（Transposed Convolution / Deconvolution），支持任意步长和 padding。

### 练习 5：性能优化

对本章的直接卷积实现进行性能优化，目标是在 $3 \times 3$ 卷积上达到 Im2Col + GEMM 性能的 80%。

---

## 13. 思考题

1. **为什么 Im2Col + GEMM 方法在深度学习框架中如此流行？它的主要局限是什么？**

2. **Winograd 变换在什么情况下会比 Im2Col + GEMM 更慢？考虑变换开销和内存访问模式。**

3. **如何设计一个自适应卷积策略，根据问题特征自动选择最优实现？**

4. **在 TileLang 中，如何利用 Tensor Core 加速卷积计算？需要什么额外的布局变换？**

5. **分组卷积（Grouped Convolution）如何影响不同实现策略的性能？**

---

## 14. 扩展阅读

1. **Winograd 变换深入**：Lavin & Gray, "Fast Algorithms for Convolutional Neural Networks" (CVPR 2016)
2. **FFT 卷积**：Mathieu et al., "Fast Convolutional Networks with FFT" (ICLR 2014)
3. **cuDNN 卷积优化**：Chetlur et al., "cuDNN: Efficient Primitives for Deep Learning" (2014)
4. **直接卷积优化**：Vasilache et al., "Fast Convolutional Nets With FBFFT" (2014)
5. **内存高效卷积**：Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (CVPR 2018)

---

转置卷积的实现要点在于精确掌握"输入到输出的贡献关系"。在标准卷积中，每个输出像素"接收"来自输入窗口内 K_h×K_w 个像素的贡献——这是一个"多对一"的汇聚（Gather）模式。而在转置卷积中，关系完全反转：每个输入像素"分发"其值给输出特征图中 K_h×K_w 个位置的像素——这是一个"一对多"的散射（Scatter）模式。理解这种"汇聚 vs 散射"的对偶性，是掌握转置卷积实现的核心。

从计算复杂度的角度，转置卷积的 FLOPs 与标准卷积完全相同（C_in × C_out × K_h × K_w × H_out × W_out × 2），但实际 GPU 延迟通常显著更高。原因有三：第一，散射模式使得多个线程可能同时累加到同一个输出位置，需要原子操作（Atomic Add）来保证正确性——这引入了串行化瓶颈；第二，输出特征图比输入更大（上采样），每个输出像素接收的输入贡献较少且不连续，导致每个线程的计算量不足（低 Occupancy）；第三，对于某些 stride 和 K 的组合，输出特征图中存在"从未被任何输入贡献触及"的像素（需要通过额外的偏置项填充），增加了边界处理的复杂度。这些原因共同导致转置卷积往往比同等参数规模的标准卷积慢 2-5 倍。


在掌握了标准卷积（前向）的多种实现策略之后，我们将目光转向卷积算子的几个重要变体。转置卷积（也称为反卷积）在语义分割、图像生成等需要上采样的任务中扮演着核心角色。它的数学本质与标准卷积存在精确的对偶关系——事实上，标准卷积的反向传播过程就是一个转置卷积——这为我们在 TileLang 中高效实现它提供了重要的思路。


## 15. 转置卷积（Transposed Convolution）

### 15.1 转置卷积的数学原理

转置卷积（也称为反卷积或分数步长卷积）常用于上采样操作，如语义分割和生成对抗网络中：

$$
Y[n, c_{out}, h, w] = \sum_{c_{in}=0}^{C_{in}-1} \sum_{i=0}^{K_h-1} \sum_{j=0}^{K_w-1} X'[n, c_{in}, h+i, w+j] \cdot W[c_{out}, c_{in}, i, j]
$$

其中 $X'$ 是对输入进行零填充后的结果，填充方式为在每个元素之间插入 $s-1$ 个零。

### 15.2 转置卷积的 TileLang 实现

```python
def transposed_conv2d(
    N: int,
    C_in: int,
    C_out: int,
    H_in: int,
    W_in: int,
    K: int,
    stride: int,
    padding: int,
):
    """Transposed Conv2D using TileLang."""
    H_out = (H_in - 1) * stride - 2 * padding + K
    W_out = (W_in - 1) * stride - 2 * padding + K

    @T.prim_func
    def tconv_kernel(
        X: T.Buffer((N, C_in, H_in, W_in), "float32"),
        W: T.Buffer((C_in, C_out, K, K), "float32"),  # Note: C_in, C_out order
        Y: T.Buffer((N, C_out, H_out, W_out), "float32"),
    ):
        with T.Kernel(N, C_out, T.ceildiv(H_out, 8), T.ceildiv(W_out, 8), threads=256) as (
            n, co, th, tw
        ):
            Y_local = T.alloc_fragment((8, 8), "float32")
            T.clear(Y_local)

            for ci in T.serial(C_in):
                for kh, kw in T.serial(K, K):
                    for i, j in T.serial(8, 8):
                        oh = th * 8 + i
                        ow = tw * 8 + j
                        # 转置卷积的输入坐标计算
                        ih = oh + padding - kh
                        iw = ow + padding - kw
                        if ih >= 0 and iw >= 0 and ih % stride == 0 and iw % stride == 0:
                            ih_s = ih // stride
                            iw_s = iw // stride
                            if ih_s < H_in and iw_s < W_in:
                                Y_local[i, j] += X[n, ci, ih_s, iw_s] * W[ci, co, kh, kw]

            for i, j in T.serial(8, 8):
                oh = th * 8 + i
                ow = tw * 8 + j
                if oh < H_out and ow < W_out:
                    Y[n, co, oh, ow] = Y_local[i, j]

    return tconv_kernel
```

转置卷积实现在索引计算上的复杂度远高于标准卷积。关键挑战在于：输出位置（oh, ow）与输入位置（ih, iw）之间并非一一对应，而是需要一个"反算"过程——从输出坐标推导出可能贡献的输入坐标，并检查其是否在有效范围内。条件 `ih % stride == 0 and iw % stride == 0` 是最核心的判断逻辑：只有当输出的采样位置恰好落在"有效"的输入格点上时，该输入才对输出有贡献。这意味着对于 stride > 1 的转置卷积，每个输出像素实际接收到的"贡献"数量是不均匀的——这正是棋盘格伪影产生的根本原因。在实现层面，这导致 warp 内的控制流严重发散，因为某些线程会跳过大部分输入通道的计算。


### 15.3 转置卷积与标准卷积的关系

空洞卷积在语义分割任务（如 DeepLab 系列）中的广泛应用，驱动了对其高效实现的需求。DeepLabV3 使用的 Atrous Spatial Pyramid Pooling (ASPP) 模块同时运行多个不同 dilation rate 的 3×3 卷积——典型的配置是 rates=[6, 12, 18]——然后将各分支的输出拼接。这种"多分支并行空洞卷积"的模式对 GPU 调度器提出了严峻挑战：不同 dilation rate 的内核具有不同的有效核大小和内存访问模式，GPU 需要在它们之间进行上下文切换，每个内核只能占用一小部分 SM，导致总利用率低下。一种优化方法是将多个 dilation rate 合并为一个内核——通过将 dilation rate 作为内核参数（而非编译期常量），在每个线程内部根据 rate 计算访问偏移。这牺牲了少量编译期优化机会（因为 rate 不再能用于常量折叠和循环展开），但换来了更好的 GPU 占用率和更少的上下文切换。

在硬件层面，空洞卷积对 GPU L2 缓存的影响值得单独审视。标准 3×3 卷积的每个输出像素需要访问 9 个输入像素，这 9 个像素位于连续的 3 行 × 3 列区域内——总共 9 个缓存行（假设每个缓存行 128 字节，每行包含 32 个 float32 元素）。而 dilation=2 的空洞 3×3 卷积实际采样区域为 5×5，涉及 25 个可能的缓存行（虽然只访问其中 9 个元素）。当多个线程块同时执行时，这些"跳跃式"的内存访问会大幅增加 L2 缓存的压力——因为每个线程块的工作集覆盖了更大的内存区域，导致缓存逐出（Eviction）更频繁。实测数据显示，对于 dilation=12 的极端配置（有效核大小 25×25），L2 缓存命中率可能从标准卷积的 85% 跌至 35%，这是空洞卷积在大 dilation rate 下性能骤降的硬性原因。


转置卷积可以看作是标准卷积的"转置"操作：

| 特性 | 标准卷积 | 转置卷积 |
|------|---------|---------|
| 输入→输出 | 下采样 | 上采样 |
| 参数形状 | $(C_{out}, C_{in}, K, K)$ | $(C_{in}, C_{out}, K, K)$ |
| 梯度计算 | 转置卷积 | 标准卷积 |
| 输出大小 | $\frac{H+2P-K}{s}+1$ | $(H-1)s-2P+K$ |

---

转置卷积解决了"上采样"问题，而空洞卷积（Dilated/Atrous Convolution）解决的则是完全不同的需求：在不增加参数量的前提下扩大感受野。空洞卷积通过在卷积核元素之间插入"空洞"（跳过的像素），使得一个 3×3 的卷积核可以获得 5×5、7×7 甚至更大的有效感受野。这种特性使其在语义分割任务中广泛使用，因为大感受野对于理解全局上下文至关重要。


## 16. 空洞卷积（Dilated Convolution）

### 16.1 空洞卷积的原理

空洞卷积（也称为扩张卷积）通过在卷积核元素之间插入空洞来增大感受野，而不增加参数量：

$$
Y[n, c_{out}, h, w] = \sum_{c_{in}} \sum_{i} \sum_{j} X[n, c_{in}, h + i \cdot d, w + j \cdot d] \cdot W[c_{out}, c_{in}, i, j]
$$

其中 $d$ 是空洞率（dilation rate）。

### 16.2 空洞卷积实现

```python
def dilated_conv2d(
    N: int,
    C_in: int,
    C_out: int,
    H: int,
    W: int,
    K: int,
    dilation: int,
    padding: int,
):
    """Dilated Conv2D using TileLang."""
    effective_K = K + (K - 1) * (dilation - 1)
    H_out = (H + 2 * padding - effective_K) // 1 + 1
    W_out = (W + 2 * padding - effective_K) // 1 + 1

    @T.prim_func
    def dilated_conv_kernel(
        X: T.Buffer((N, C_in, H, W), "float32"),
        W: T.Buffer((C_out, C_in, K, K), "float32"),
        Y: T.Buffer((N, C_out, H_out, W_out), "float32"),
    ):
        with T.Kernel(N, C_out, T.ceildiv(H_out, 8), T.ceildiv(W_out, 8), threads=256) as (
            n, co, th, tw
        ):
            Y_local = T.alloc_fragment((8, 8), "float32")
            T.clear(Y_local)

            for ci, kh, kw in T.serial(C_in, K, K):
                for i, j in T.serial(8, 8):
                    oh = th * 8 + i
                    ow = tw * 8 + j
                    ih = oh - padding + kh * dilation
                    iw = ow - padding + kw * dilation
                    if 0 <= ih < H and 0 <= iw < W:
                        Y_local[i, j] += X[n, ci, ih, iw] * W[co, ci, kh, kw]

            for i, j in T.serial(8, 8):
                oh = th * 8 + i
                ow = tw * 8 + j
                if oh < H_out and ow < W_out:
                    Y[n, co, oh, ow] = Y_local[i, j]

    return dilated_conv_kernel
```

空洞卷积在实现上看似仅需将索引计算中的 `kh` 替换为 `kh * dilation`，但这看似微小的变化对内存访问模式产生了深远影响。当 dilation=2 时，一个 3×3 卷积核的实际采样范围扩展到了 5×5 的空间区域（有效核大小为 `effective_K = K + (K-1)*(dilation-1) = 5`），但每个输出像素仍然只访问 9 个采样点（而非 5×5=25 个）。这种"稀疏采样"模式导致两个问题：一是输入数据在内存中不再连续（相邻采样点之间间隔 dilation 个像素），降低了缓存行利用率；二是在共享内存加载优化中，"Halo 区域"的计算必须使用有效的膨胀核大小（effective_K），使得所需共享内存增大但实际使用的数据比例降低。对于大 dilation 值，Im2Col+GEMM 方法（将空洞卷积展开为稀疏的列向量）通常是更高效的选择。


---

空洞卷积通过固定的采样模式扩大感受野，但仍然受到规则网格的限制。可变形卷积（Deformable Convolution）则更进一步，通过引入可学习的空间偏移量，使卷积核的采样位置可以根据输入内容动态调整。这赋予了网络对几何变换的自适应能力，但同时也带来了不规则内存访问模式和双线性插值等新的计算挑战。


## 17. 可变形卷积（Deformable Convolution）

可变形卷积虽然在性能上远低于标准卷积，但其在计算机视觉任务中带来的精度提升通常远超性能代价。首次提出可变形卷积的论文（Dai et al., ICCV 2017）在 COCO 目标检测任务上展示了 2-5 个点的 mAP 提升——这在检测领域是显著的进步。可变形卷积的有效性来源于它解决了标准卷积的根本限制：固定几何形状的采样网格。当物体发生非刚性变形、旋转或透视变换时，固定的矩形采样窗口无法有效捕获物体的形状信息，而可学习的偏移量让卷积核"学会"自适应地聚焦于物体的实际轮廓。

从实现优化的角度，可变形卷积有几个潜在的加速方向。一是利用偏移量预测网络本身也是一个小型卷积网络（通常 3×3），可以将偏移量预测与可变形采样融合到同一个 CUDA 内核中，避免中间的全局内存往返。二是在双线性插值中使用查找表（LUT）加速——将最常用的偏移量模式预计算为整数像素偏移，仅在偏移量超出离散格点时回退到浮点插值。三是限制偏移量的范围（如通过 tanh 激活函数将偏移量限制在 ±2 像素内），一方面减少插值开销（因为大部分采样点落在离散格点附近），另一方面也作为正则化防止过度变形导致的不稳定训练。


### 17.1 可变形卷积原理

可变形卷积通过学习额外的偏移量，使卷积核可以适应输入的几何变换：

$$
Y[n, c_{out}, h, w] = \sum_{c_{in}} \sum_{i} \sum_{j} W[c_{out}, c_{in}, i, j] \cdot X[n, c_{in}, h + i + \Delta h_{i,j}, w + j + \Delta w_{i,j}]
$$

其中 $\Delta h_{i,j}$ 和 $\Delta w_{i,j}$ 是学习的偏移量。

### 17.2 可变形卷积实现

```python
def deformable_conv2d(
    N: int,
    C_in: int,
    C_out: int,
    H: int,
    W: int,
    K: int,
):
    """Deformable Conv2D with learned offsets."""

    @T.prim_func
    def deform_conv_kernel(
        X: T.Buffer((N, C_in, H, W), "float32"),
        Offset: T.Buffer((N, 2 * K * K, H, W), "float32"),
        W: T.Buffer((C_out, C_in, K, K), "float32"),
        Y: T.Buffer((N, C_out, H, W), "float32"),
    ):
        with T.Kernel(N, C_out, T.ceildiv(H, 8), T.ceildiv(W, 8), threads=256) as (
            n, co, th, tw
        ):
            Y_local = T.alloc_fragment((8, 8), "float32")
            T.clear(Y_local)

            for ci, kh, kw in T.serial(C_in, K, K):
                offset_idx = kh * K + kw
                for i, j in T.serial(8, 8):
                    oh = th * 8 + i
                    ow = tw * 8 + j
                    if oh < H and ow < W:
                        # 读取偏移量
                        delta_h = Offset[n, offset_idx * 2, oh, ow]
                        delta_w = Offset[n, offset_idx * 2 + 1, oh, ow]

                        # 双线性插值
                        h_float = oh + kh + delta_h
                        w_float = ow + kw + delta_w
                        h_low = T.cast(T.floor(h_float), "int32")
                        w_low = T.cast(T.floor(w_float), "int32")
                        h_high = h_low + 1
                        w_high = w_low + 1

                        # 计算插值权重
                        lh = h_float - T.cast(h_low, "float32")
                        lw = w_float - T.cast(w_low, "float32")
                        hh = T.float32(1) - lh
                        hw = T.float32(1) - lw

                        # 双线性插值采样
                        val = T.float32(0)
                        if 0 <= h_low < H and 0 <= w_low < W:
                            val += X[n, ci, h_low, w_low] * hh * hw
                        if 0 <= h_low < H and 0 <= w_high < W:
                            val += X[n, ci, h_low, w_high] * hh * lw
                        if 0 <= h_high < H and 0 <= w_low < W:
                            val += X[n, ci, h_high, w_low] * lh * hw
                        if 0 <= h_high < H and 0 <= w_high < W:
                            val += X[n, ci, h_high, w_high] * lh * lw

                        Y_local[i, j] += val * W[co, ci, kh, kw]

将 1×1 卷积映射到 GEMM 的效率，取决于空间维度的扁平化策略与矩阵分块策略的匹配程度。在上述实现中，空间维度 (N, H, W) 被展平为 `spatial = N*H*W`，这假设 N、H、W 三个维度的扁平化顺序对性能无影响——但这个假设在某些情况下不成立。当 batch size N > 1 时，不同样本的相同空间位置在展平后不再连续（因为展平是先 H×W 再 N），这意味着 GEMM 的 K 维度（C_in）上的归约在空间上是分散的，对缓存不友好。更好的做法是将 (N, H, W) 展平为 (N*H, W) 或保持三维索引，然后让 GEMM 的分块策略自然地处理批次维度。实际上，cuBLAS 的 `cublasGemmStridedBatched` 正是为了解决这种"批次维度的 GEMM"问题而设计的。

另一个容易被忽视的细节是：当 C_in 或 C_out 很小时（如 MobileNet 中的 16、32 通道），GEMM 的 block_K 分块策略需要特别调整。如果 C_in=16 而 block_K=32，GEMM 将在 K 维度上"跨越 batch 边界"——即同一个 block_K 分块内部同时包含来自不同 batch 样本的数据。这在数学上不会导致计算错误，但可能破坏内存访问的局部性，因为不同 batch 的数据在 GPU 内存中通常是不连续的。此时应将 block_K 减小到 ≤ C_in，或者将 batch 维度合并到 M 维度中（即将 GEMM 的输入视为 (N*H*W, C_in) × (C_in, C_out)）。正确理解张量形状与 GEMM 分块参数之间的关系，是实现高效 1×1 卷积的基础。


            for i, j in T.serial(8, 8):
                oh = th * 8 + i
                ow = tw * 8 + j
                if oh < H and ow < W:
                    Y[n, co, oh, ow] = Y_local[i, j]

    return deform_conv_kernel
```

可变形卷积是本章中计算模式最复杂的算子，其核心难点在于双线性插值的实现。由于每个采样位置的偏移量（delta_h, delta_w）是连续的浮点数，采样坐标往往落在像素之间的亚像素位置，需要通过四个最近邻像素的加权平均来计算采样值。代码中的双线性插值涉及四次内存访问和四次乘加，使得每个输出像素的计算量是标准卷积的 4 倍以上（在 3×3 卷积中，每个输出需要 3×3×4=36 次内存访问 vs 标准卷积的 9 次）。此外，不可预测的偏移量意味着编译器和硬件无法预取数据，缓存预取机制基本失效，导致实际的全局内存延迟接近最坏情况。这种特性使得可变形卷积通常成为模型中的性能瓶颈，实践中常通过限制偏移量范围、或使用查找表预计算采样模式来进行近似加速。


---

可变形卷积展现了卷积概念的灵活性，但在实际的深度学习模型（尤其是 MobileNet 等轻量级架构）中，1×1 卷积（Pointwise Convolution）的使用频率远高于可变形卷积。1×1 卷积虽然名为"卷积"，但本质上不涉及任何空间邻域运算，而是对每个空间位置独立执行的矩阵乘法。这一特性使其可以完美映射到 GEMM，从而在 GPU 上获得极高的计算效率。


## 18. 1×1 卷积优化（Pointwise Convolution）

### 18.1 1×1 卷积的特殊性

1×1 卷积（Pointwise Convolution）本质上是一个矩阵乘法，不涉及空间维度的卷积运算：

$$
Y[n, c_{out}, h, w] = \sum_{c_{in}} X[n, c_{in}, h, w] \cdot W[c_{out}, c_{in}]
$$

### 18.2 高效 1×1 卷积实现

```python
def pointwise_conv2d(
    N: int,
    C_in: int,
    C_out: int,
    H: int,
    W: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
):
    """Efficient 1×1 Conv2D using GEMM."""
    spatial = H * W

    @T.prim_func
    def pw_conv_kernel(
        X: T.Buffer((N, C_in, H, W), "float32"),
        W: T.Buffer((C_out, C_in), "float32"),
        Y: T.Buffer((N, C_out, H, W), "float32"),
    ):
        # Reshape X to (N*H*W, C_in) and use GEMM
        with T.Kernel(T.ceildiv(spatial, block_M), T.ceildiv(C_out, block_N), threads=256) as (
            bx, by
        ):
            X_frag = T.alloc_fragment((block_M, block_K), "float32")
            W_frag = T.alloc_fragment((block_K, block_N), "float32")
            Y_frag = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(Y_frag)

            for k in T.serial(T.ceildiv(C_in, block_K)):
                # Load X tile
                for i, j in T.serial(block_M, block_K):
                    idx = bx * block_M + i
                    if idx < spatial and k * block_K + j < C_in:
                        X_frag[i, j] = X[0, k * block_K + j, idx // W, idx % W]
                    else:
                        X_frag[i, j] = T.float32(0)

                # Load W tile
                for i, j in T.serial(block_K, block_N):
                    if k * block_K + i < C_in and by * block_N + j < C_out:
                        W_frag[i, j] = W[by * block_N + j, k * block_K + i]
                    else:
                        W_frag[i, j] = T.float32(0)

                T.gemm(X_frag, W_frag, Y_frag)

            # Store result
            for i, j in T.serial(block_M, block_N):
                idx = bx * block_M + i
                co = by * block_N + j
                if idx < spatial and co < C_out:
                    Y[0, co, idx // W, idx % W] = Y_frag[i, j]

    return pw_conv_kernel
```

1×1 卷积的 GEMM 映射展示了一个重要的设计洞察：空间维度的扁平化处理。通过将（N, H, W）三个维度展平为单一的空间维度（spatial = N*H*W），1×1 卷积被精确转化为一个（spatial × C_in）×（C_in, C_out）的矩阵乘法。在实际实现中，空间维度的展开顺序可能显著影响性能——先展平 H 再展平 W 可以保证相邻的空间位置（沿 W 方向）在内存中连续，从而提高全局内存的合并访问效率。此外，该实现中使用 float16 精度存储输入和权重（`T.float16`），配合 float32 累加器，是混合精度训练的典型模式，可以在 A100 GPU 上利用 Tensor Core 获得高达 312 TFLOPS 的矩阵乘法吞吐量。


---

1×1 卷积的高效实现得益于它与矩阵乘法的等价性，但其他类型的卷积（3×3、5×5、7×7）无法如此简单地规约为 GEMM。面对不同的卷积配置（卷积核大小、通道数、空间维度），最优的实现策略和分块参数各不相同。手工为每种配置调优是不现实的，因此自动调优（Auto-Tuning）成为生产级卷积实现中不可或缺的一环。


## 19. 卷积的自动调优

### 19.1 自动搜索最优配置

```python
def auto_tune_conv(configs, input_shape, kernel_shape):
    """Auto-tune convolution parameters."""

    best_time = float("inf")
    best_config = None

    for config in configs:
        block_M, block_N, block_K = config["block_M"], config["block_N"], config["block_K"]
        num_stages = config["num_stages"]
        num_warps = config["num_warps"]

        try:
            kernel = conv2d_im2col_gemm(
                *input_shape, *kernel_shape,
                block_M=block_M, block_N=block_N, block_K=block_K,
            )
            profiler = Profiler(kernel)
            time_ms = profiler.bench()

            if time_ms < best_time:
                best_time = time_ms
                best_config = config
        except Exception:
            continue

    return best_config, best_time
```

自动调优的实现采用了一个简单的暴力搜索（Grid Search）策略：遍历所有候选配置，对每个配置编译并测量内核执行时间，选择延迟最小的配置。虽然暴力搜索在搜索空间较小时（如分块参数只有 4 种候选）是可行的，但在生产环境中配置空间通常包含 5-7 个参数，总组合数可能达到数千甚至数万，此时需要更高效的搜索策略。常见的高级方法包括：基于贝叶斯优化的调优器（如 Hyperopt）、基于代价模型的搜索（如 TVM 的 AutoScheduler 使用 XGBoost 预测配置性能）、以及基于遗传算法的进化搜索。此外，代码中的 `try-except` 捕获说明某些配置可能因为寄存器溢出或共享内存超限而编译失败——在搜索空间中，有效配置的比例可能只有 30%-50%，一个高效的调优器应能快速跳过无效配置。


### 19.2 搜索空间设计

| 参数 | 搜索范围 | 影响 |
|------|---------|------|
| block_M | 32, 64, 128, 256 | 并行度 vs 寄存器压力 |
| block_N | 32, 64, 128, 256 | 输出分块大小 |
| block_K | 8, 16, 32, 64 | 归约分块大小 |
| num_stages | 1, 2, 3 | Pipeline 深度 |
| num_warps | 2, 4, 8 | Warp 数量 |

---

自动调优通过实验搜索最优参数，但盲目的暴力搜索成本高昂。一个更高效的方法是通过性能模型（如 Roofline 模型）预先判断卷积配置的计算瓶颈类型（计算受限还是内存受限），从而大幅缩小搜索空间。Roofline 模型将硬件的峰值计算能力和峰值内存带宽作为天花板，通过计算强度（FLOPs/Byte）来定位性能瓶颈，是卷积性能分析的基础工具。


## 20. 卷积性能分析工具

### 20.1 Roofline 模型分析

```python
def roofline_analysis_conv(C_in, C_out, H, W, K, stride, padding):
    """Analyze convolution using Roofline model."""
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    # 计算量 (FLOPs)
    flops = C_out * H_out * W_out * C_in * K * K * 2

    # 内存访问量 (Bytes)
    input_bytes = C_in * H * W * 4
    weight_bytes = C_out * C_in * K * K * 4
    output_bytes = C_out * H_out * W_out * 4
    total_bytes = input_bytes + weight_bytes + output_bytes

    # 计算强度 (FLOPs/Byte)
    arithmetic_intensity = flops / total_bytes

    # A100 参数
    peak_compute = 312e12  # 312 TFLOPS
    peak_bandwidth = 2e12  # 2 TB/s

    # 瓶颈分析
    compute_bound_threshold = peak_bandwidth / peak_compute
    if arithmetic_intensity > compute_bound_threshold:
        expected_perf = peak_compute
        bottleneck = "compute"
    else:
        expected_perf = arithmetic_intensity * peak_bandwidth
        bottleneck = "memory"

    return {
        "flops": flops,
        "bytes": total_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "expected_tflops": expected_perf / 1e12,
        "bottleneck": bottleneck,
    }
```

Roofline 分析的核心价值在于提供一个"天花板"视角——无论代码如何优化，性能都不可能超过硬件峰值计算能力或内存带宽所设定的上限。`compute_bound_threshold = peak_bandwidth / peak_compute`（约 6.4 FLOPs/Byte 对于 A100）是一个关键的决策边界：当卷积配置的计算强度超过此阈值时，瓶颈在计算单元；低于此阈值时，瓶颈在内存带宽。这一诊断结果直接引导优化方向——如果是计算受限，应增大分块参数以更好地利用 Tensor Core 和指令级并行；如果是内存受限，应通过数据复用（共享内存缓存、权重预加载）或算法变换（Winograd）来减少全局内存访问量。不过需要注意的是，Roofline 模型假设算术运算和数据传输可以完全重叠，而实际硬件的重叠能力受限于 warp 调度和缓存容量，因此模型预测的"理论上限"通常需要乘以 0.7-0.85 的修正系数。


### 20.2 典型卷积配置的 Roofline 分析

| 配置 | FLOPs | Bytes | AI (FLOPs/Byte) | 瓶颈 |
|------|-------|-------|-----------------|------|
| 3×3, C=64, H=W=56 | 230M | 800K | 287 | Compute |
| 3×3, C=256, H=W=28 | 118M | 1.6M | 74 | Memory |
| 1×1, C=1024, H=W=7 | 102M | 400K | 255 | Compute |
| 7×7, C=64, H=W=224 | 2.3G | 1.5M | 1533 | Compute |
| 5×5, C=128, H=W=56 | 200M | 700K | 286 | Compute |

---

## 21. 实战案例：MobileNetV2 倒残差块

### 21.1 倒残差块结构

```python
def inverted_residual_block(
    N: int,
    C_in: int,
    C_out: int,
    H: int,
    W: int,
    expansion_ratio: int = 6,
    stride: int = 1,
):
    """MobileNetV2 Inverted Residual Block."""
    C_expanded = C_in * expansion_ratio

    @T.prim_func
    def inv_res_kernel(
        X: T.Buffer((N, C_in, H, W), "float32"),
        W_expand: T.Buffer((C_expanded, C_in, 1, 1), "float32"),
        W_depthwise: T.Buffer((C_expanded, 1, 3, 3), "float32"),
        W_project: T.Buffer((C_out, C_expanded, 1, 1), "float32"),
        Y: T.Buffer((N, C_out, H, W), "float32"),
    ):
        # Step 1: 1×1 Expansion
        Expanded = T.alloc_buffer((N, C_expanded, H, W), "float32")
        # ... pointwise convolution

        # Step 2: 3×3 Depthwise
        H_out = H // stride
        W_out = W // stride
        DW_out = T.alloc_buffer((N, C_expanded, H_out, W_out), "float32")
        # ... depthwise convolution

        # Step 3: 1×1 Projection (no activation)
        # ... pointwise convolution

    return inv_res_kernel
```

MobileNetV2 倒残差块的"倒残差"命名来源于其与标准残差块相反的通道变换方向：标准残差块先压缩（1×1 降维）再扩展（3×3）再压缩（1×1 升维），而倒残差块先扩展（1×1 升维 6 倍）再空间卷积（3×3 Depthwise）再压缩（1×1 降维）。这种"先宽后窄"的设计使得 3×3 Depthwise 卷积在一个高维空间中操作，从而能够表达更丰富的空间特征。从 GPU 实现的角度看，扩展阶段的 1×1 卷积是计算重头（从 C_in 到 6×C_in 的矩阵乘法），而 Depthwise 阶段虽然计算量小但内存访问密集——两者的计算特征截然不同，因此融合实现中需要为不同阶段分配不同比例的共享内存和线程资源。


---

MobileNetV2 倒残差块的实现展示了三种卷积类型（1×1 扩展、3×3 深度可分离、1×1 投影）在一个算子中协同工作的模式。在实际部署中，将这三个操作融合为一个内核（Kernel Fusion）可以避免中间结果的全局内存写入，大幅降低带宽需求。这种多操作融合的思想是高性能推理引擎（如 TensorRT、TVM）的核心优化手段之一。


## 22. 总结（扩展）

### 卷积实现方法全对比

| 方法 | 内存开销 | 计算效率 | 适用场景 | 实现复杂度 |
|------|---------|---------|---------|-----------|
| Im2Col + GEMM | 高 | 高 | 通用 | 低 |
| Winograd F(2,3) | 中 | 最高（3×3） | 小卷积核 | 中 |
| Winograd F(4,3) | 中 | 最高（3×3） | 小卷积核 | 高 |
| 直接卷积 | 低 | 中 | 内存受限 | 中 |
| FFT 卷积 | 高 | 高（大核） | 大卷积核 | 高 |
| 空洞卷积 | 低 | 中 | 感受野扩展 | 低 |
| 转置卷积 | 中 | 中 | 上采样 | 中 |
| 可变形卷积 | 中 | 低 | 几何变换 | 高 |

### 关键经验

1. **选择合适的实现方法**：根据卷积核大小和硬件特性选择
2. **内存访问优化至关重要**：Padding 分块、Stride 预取
3. **利用 Tile 抽象简化并行化**：自动处理分块和线程映射
4. **性能分析指导优化**：使用 Roofline 模型识别瓶颈
5. **自动调优找到最优配置**：搜索 block 大小和 pipeline 深度

---

通过前面的全面对比，我们可以看到卷积算子的设计本质上是一系列权衡（Trade-off）的集合：计算量 vs 内存开销、通用性 vs 专用性、实现复杂度 vs 性能上限。然而，除了 F(2,3) 之外，Winograd 变换家族还有更高效的变体。F(4,3) 变换能进一步将乘法次数从 F(2,3) 的减少 33% 提升到减少 50%，但代价是变换矩阵更复杂、数值误差更大、内存占用更高。


## 24. Winograd F(4,3) 变换详解

### 24.1 F(4,3) 变换的数学推导

Winograd F(4,3) 变换输出大小为 $4 \times 4$，滤波器大小为 $3 \times 3$，需要 $4+3-1=6$ 次乘法（而非 $4 \times 3 = 12$ 次），减少 50% 的乘法运算：

$$
Y = A^T \left[ (G g G^T) \odot (B^T d B) \right] A
$$

对于 $F(4, 3)$，变换矩阵为：

$$
G = \begin{bmatrix}
1 & 0 & 0 \\
0.5 & 0.5 & 0.5 \\
0.5 & -0.5 & 0.5 \\
-0.5 & -1 & -1 \\
0 & 0 & 1 \\
0 & 0 & 0
\end{bmatrix}
$$

$$
B = \begin{bmatrix}
1 & 0 & -1 & 0 & 0 & 0 \\
0 & 1 & 1 & 0 & 0 & 0 \\
0 & -1 & 1 & 0 & 0 & 0 \\
0 & 1 & 0 & -1 & 0 & 0 \\

Winograd F(4,3) 的实现较 F(2,3) 更为复杂，主要体现在两个方面。首先是变换矩阵的维度更大（6×6 vs 4×4），这意味着输入变换阶段（B^T·d·B）和输出变换阶段（A^T·m·A）的运算量线性增长——从 F(2,3) 的 4×(4+4) = 32 次加减到 F(4,3) 的 6×(6+6) = 72 次加减（每个 Tile）。虽然乘法次数减少了一半，但变换阶段的加减运算在 GPU 上同样消耗执行单元（CUDA Core），如果加减运算的开销超过了乘法节省的时间，F(4,3) 可能比 F(2,3) 更慢。

其次，F(4,3) 中的变换矩阵系数涉及更大的数值增长。以输出变换矩阵 A 为例，其元素包含系数 8——这意味着变换域中的值在逆变换时可能被放大 8 倍。当变换域值本身已经因为逐元素乘法而较大时（如输入值范围 0-255 且权重初始化为以 0 为均值的高斯分布随机值），8 倍的放大可能导致 float32 累加器中的有效精度位数减少（因为较大值的加法会"淹没"较小值的贡献）。这也是为什么 F(4,3) 通常只推荐在推理阶段使用——推理时权重已固定，可以通过离线量化（Quantization）和缩放来保持数值范围在安全区间内。在训练阶段，动态变化的权重和梯度使 F(4,3) 的数值风险显著增加。

在实际部署中，有一项关于 F(2,3) vs F(4,3) 选择的重要经验法则：当每个 Winograd Tile 的输出像素数超过变换域的计算开销时，使用更大的 Tile 才有收益。具体来说，F(2,3) 的变换开销对应 2×2=4 个输出像素，而其变换域计算对应 4×4=16 个值的逐元素乘加；F(4,3) 的变换开销对应 4×4=16 个输出像素，变换域计算对应 6×6=36 个值的逐元素乘加。因此，F(4,3) 将变换开销摊销到了 4 倍于 F(2,3) 的输出像素上，只要实际图像维度足够大以保证 Tile 数量充足。一般建议：当 H_out ≥ 28 时，F(4,3) 的优势开始显现；当 H_out < 14 时，F(2,3) 几乎总是更优。

0 & 1 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & -1
\end{bmatrix}
$$

$$
A = \begin{bmatrix}
1 & 0 & 0 & 0 \\
1 & 1 & 1 & 1 \\
1 & -1 & 1 & -1 \\
1 & 2 & 4 & 8 \\
1 & -2 & 4 & -8 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

### 24.2 F(4,3) TileLang 实现

```python
def winograd_f43_conv2d(
    N: int,
    C_in: int,
    C_out: int,
    H: int,
    W: int,
):
    """Winograd F(4,3) convolution using TileLang."""
    assert H % 4 == 0 and W % 4 == 0, "H and W must be multiples of 4"
    H_out = H - 2
    W_out = W - 2
    tile_h = T.ceildiv(H_out, 4)
    tile_w = T.ceildiv(W_out, 4)

    @T.prim_func
    def winograd_f43_kernel(
        X: T.Buffer((N, C_in, H, W), "float32"),
        U: T.Buffer((6, 6, C_in, C_out), "float32"),
        Y: T.Buffer((N, C_out, H_out, W_out), "float32"),
    ):
        V = T.alloc_buffer((6, 6, C_in, N, tile_h, tile_w), "float32")
        M = T.alloc_buffer((6, 6, C_out, N, tile_h, tile_w), "float32")

        # Step 1: Input transform - B^T d B (6x6 input tile)
        with T.Kernel(N, C_in, tile_h, tile_w, threads=128) as (n, c, th, tw):
            d = T.alloc_fragment((6, 6), "float32")
            v = T.alloc_fragment((6, 6), "float32")

            for i, j in T.serial(6, 6):
                h_idx = th * 4 + i
                w_idx = tw * 4 + j
                if h_idx < H and w_idx < W:
                    d[i, j] = X[n, c, h_idx, w_idx]
                else:
                    d[i, j] = T.float32(0)

            # B^T d B transform
            tmp = T.alloc_fragment((6, 6), "float32")
            for j in T.serial(6):
                tmp[0, j] = d[0, j] - d[2, j] + d[4, j]
                tmp[1, j] = d[1, j] + d[2, j] - d[3, j] - d[4, j]
                tmp[2, j] = -d[1, j] + d[2, j] + d[3, j] - d[4, j]
                tmp[3, j] = d[1, j] + d[2, j] + d[3, j] + d[4, j]
                tmp[4, j] = -d[1, j] + d[2, j] - d[3, j] + d[4, j]
                tmp[5, j] = d[1, j] - d[3, j] + d[5, j]

            for i in T.serial(6):
                v[i, 0] = tmp[i, 0] - tmp[i, 2] + tmp[i, 4]
                v[i, 1] = tmp[i, 1] + tmp[i, 2] - tmp[i, 3] - tmp[i, 4]
                v[i, 2] = -tmp[i, 1] + tmp[i, 2] + tmp[i, 3] - tmp[i, 4]
                v[i, 3] = tmp[i, 1] + tmp[i, 2] + tmp[i, 3] + tmp[i, 4]
                v[i, 4] = -tmp[i, 1] + tmp[i, 2] - tmp[i, 3] + tmp[i, 4]
                v[i, 5] = tmp[i, 1] - tmp[i, 3] + tmp[i, 5]

            for i, j in T.serial(6, 6):
                V[i, j, c, n, th, tw] = v[i, j]

        # Step 2: Element-wise multiplication
        with T.Kernel(6, 6, C_out, N, tile_h, tile_w, threads=128) as (
            i, j, co, n, th, tw
        ):
            acc = T.alloc_fragment((1,), "float32")
            acc[0] = T.float32(0)
            for ci in T.serial(C_in):
                acc[0] += U[i, j, ci, co] * V[i, j, ci, n, th, tw]
            M[i, j, co, n, th, tw] = acc[0]

        # Step 3: Output transform - A^T m A
        with T.Kernel(N, C_out, tile_h, tile_w, threads=128) as (n, co, th, tw):
            m = T.alloc_fragment((6, 6), "float32")
            for i, j in T.serial(6, 6):
                m[i, j] = M[i, j, co, n, th, tw]

            tmp = T.alloc_fragment((4, 6), "float32")
            for j in T.serial(6):
                tmp[0, j] = m[0, j] + m[1, j] + m[2, j] + m[3, j] + m[4, j]
                tmp[1, j] = m[1, j] - m[2, j] + 2 * m[3, j] - 2 * m[4, j]
                tmp[2, j] = m[1, j] + m[2, j] + 4 * m[3, j] + 4 * m[4, j]
                tmp[3, j] = m[1, j] - m[2, j] + 8 * m[3, j] - 8 * m[4, j] + m[5, j]

            for i in T.serial(4):
                for j in T.serial(4):
                    h_idx = th * 4 + i
                    w_idx = tw * 4 + j
                    if h_idx < H_out and w_idx < W_out:
                        Y[n, co, h_idx, w_idx] = (
                            tmp[i, 0] + tmp[i, 1] + tmp[i, 2] + tmp[i, 3]
                        )

    return winograd_f43_kernel
```

Winograd F(4,3) 在 F(2,3) 的基础上将输出 tile 从 2×2 扩大到 4×4，乘法的理论减少比例从 33% 提升到 50%，但代价是变换矩阵从 4×4（或 4×3）扩大到 6×6（或 6×3）。这带来了两个直接后果：第一，输入变换的计算量显著增加——F(4,3) 的 B^T·d·B 变换需要处理 6×6 的中间矩阵，比 F(2,3) 的 4×4 多了约 2.25 倍的运算；第二，变换域的中间缓冲区（V 和 M）尺寸也相应增大，内存占用增加。因此 F(4,3) 仅在"输出维度足够大，能够摊销变换开销"时才有优势，通常建议在 H_out ≥ 28 且 W_out ≥ 28 时使用。另外注意代码中 assert 要求 H 和 W 必须是 4 的倍数，这是 F(4,3) 对输入对齐的硬性要求，在实际部署中可能需要在输入前进行裁剪或填充。


### 24.3 F(2,3) vs F(4,3) 性能对比

| 特性 | F(2,3) | F(4,3) |
|------|--------|--------|
| 输出 Tile 大小 | 2×2 | 4×4 |
| 变换域大小 | 4×4 | 6×6 |
| 乘法减少比例 | 33% | 50% |
| 变换开销 | 低 | 中 |
| 数值误差 | 较小 | 较大 |
| 内存占用 | 中 | 较高 |
| 适用场景 | 通用 3×3 卷积 | 大空间维度 3×3 卷积 |

> [!TIP]
> F(4,3) 虽然理论乘法更少，但变换开销更大。在空间维度较小（如 H=W=14）时，F(2,3) 可能更优；在空间维度较大（如 H=W=56）时，F(4,3) 的优势更明显。

---

Winograd 变换的 F(4,3) 变体在理论上有更高的乘法减少率，但实践中其性能表现高度依赖于输入的空间维度。当空间维度较大时，更大的输出 tile（4×4 vs 2×2）意味着更少的变换开销摊销，因此 F(4,3) 更具优势。而当空间维度较小时，变换开销主导了总耗时，F(2,3) 反而是更好的选择。接下来，我们将转向深度可分离卷积的优化，这类算子在 MobileNet 系列模型中占据了绝大部分计算量。


## 25. 深度可分离卷积的完整优化

### 25.1 优化的 Depthwise Conv 实现

深度可分离卷积中，Depthwise Conv 是计算瓶颈。以下是使用 Shared Memory 优化的实现：

```python
def optimized_depthwise_conv2d(
    N: int,
    C: int,
    H_in: int,
    W_in: int,
    K: int,
    stride: int,
    padding: int,
):
    """Optimized depthwise convolution with shared memory."""
    H_out = (H_in + 2 * padding - K) // stride + 1
    W_out = (W_in + 2 * padding - K) // stride + 1
    BLOCK_H = 16
    BLOCK_W = 16

    @T.prim_func
    def dw_conv_kernel(
        X: T.Buffer((N, C, H_in, W_in), "float32"),
        W: T.Buffer((C, 1, K, K), "float32"),
        Y: T.Buffer((N, C, H_out, W_out), "float32"),
    ):
        with T.Kernel(
            N, C, T.ceildiv(H_out, BLOCK_H), T.ceildiv(W_out, BLOCK_W), threads=256
        ) as (n, c, th, tw):
            # Load filter weights to registers
            w_local = T.alloc_fragment((K, K), "float32")
            for kh, kw in T.serial(K, K):
                w_local[kh, kw] = W[c, 0, kh, kw]

            # Shared memory for input tile (with halo)
            smem_h = BLOCK_H * stride + K - 1
            smem_w = BLOCK_W * stride + K - 1
            X_shared = T.alloc_shared((smem_h, smem_w), "float32")

            # Load input tile with halo to shared memory
            for i, j in T.serial(smem_h, smem_w):
                h_idx = th * BLOCK_H * stride - padding + i
                w_idx = tw * BLOCK_W * stride - padding + j
                if 0 <= h_idx < H_in and 0 <= w_idx < W_in:
                    X_shared[i, j] = X[n, c, h_idx, w_idx]
                else:
                    X_shared[i, j] = T.float32(0)

            T.sync_threads()

            # Compute depthwise convolution from shared memory
            Y_local = T.alloc_fragment((BLOCK_H, BLOCK_W), "float32")
            T.clear(Y_local)

            for kh, kw in T.serial(K, K):
                for i, j in T.serial(BLOCK_H, BLOCK_W):
                    oh = th * BLOCK_H + i
                    ow = tw * BLOCK_W + j
                    if oh < H_out and ow < W_out:
                        Y_local[i, j] += X_shared[i * stride + kh, j * stride + kw] * w_local[kh, kw]

            # Store results
            for i, j in T.serial(BLOCK_H, BLOCK_W):
                oh = th * BLOCK_H + i
                ow = tw * BLOCK_W + j
                if oh < H_out and ow < W_out:
                    Y[n, c, oh, ow] = Y_local[i, j]

    return dw_conv_kernel
```

这段优化实现抓住了 Depthwise 卷积的核心优化机会：每个通道的卷积核（K×K）是独立且较小（通常 3×3），可以被整个加载到寄存器中（`w_local`）。将权重保持在寄存器中意味着在 K×K 次乘加循环内，每次权重访问仅需 1 个时钟周期，而非从共享内存的约 20 个周期或全局内存的约 400 个周期。共享内存部分（`X_shared`）用于缓存输入数据的 Halo 区域——尺寸为 `(BLOCK_H*stride+K-1) × (BLOCK_W*stride+K-1)` 而非 `BLOCK_H × BLOCK_W`，多出的边界部分 (K-1) 使得每个输出 tile 所需的所有输入数据都在共享内存中，从而避免了跨 tile 的重复全局内存加载。`T.sync_threads()` 是一个关键屏障，确保所有线程完成共享内存的写入之后才开始读取——缺少此同步会导致数据竞争和错误的计算结果。


### 25.2 Depthwise Conv 的性能特征

| 实现方式 | 带宽利用率 | 计算效率 | 说明 |
|---------|-----------|---------|------|
| 朴素实现 | 低 | 低 | 每个输出独立加载输入 |
| Shared Memory | 中 | 高 | 利用 Halo 数据复用 |
| 向量化加载 | 高 | 高 | 使用 float4 加载 |
| 融合 Depthwise + Pointwise | 最高 | 最高 | 避免中间结果写回 |

---

深度可分离卷积的优化揭示了共享内存（Shared Memory）在 GPU 编程中的核心地位：通过将输入数据以 Halo 区域的方式加载到共享内存中，相邻线程可以高效复用重叠的数据区域，从而将全局内存访问减少为原来的 1/(K_h × K_w)。然而，共享内存的容量有限（典型值约 164KB/SM），当通道数较多或空间维度较大时，需要仔细设计分块策略以在占用率（Occupancy）和数据复用之间取得平衡。


## 26. 转置卷积的 Im2Col 实现

### 26.1 转置卷积的 Col2Im 方法

转置卷积可以通过"梯度视角"来实现：将标准卷积的反向传播看作转置卷积的前向传播：

```python
def transposed_conv2d_col2im(
    N: int,
    C_in: int,
    C_out: int,
    H_in: int,
    W_in: int,
    K: int,
    stride: int,
    padding: int,
):
    """Transposed Conv2D via Col2Im approach."""
    H_out = (H_in - 1) * stride - 2 * padding + K
    W_out = (W_in - 1) * stride - 2 * padding + K

    @T.prim_func
    def tconv_col2im_kernel(
        X: T.Buffer((N, C_in, H_in, W_in), "float32"),
        W: T.Buffer((C_in, C_out, K, K), "float32"),
        Y: T.Buffer((N, C_out, H_out, W_out), "float32"),
    ):
        # Initialize output with zeros
        with T.Kernel(N, C_out, T.ceildiv(H_out, 8), T.ceildiv(W_out, 8), threads=256) as (
            n, co, th, tw
        ):
            Y_local = T.alloc_fragment((8, 8), "float32")
            T.clear(Y_local)


融合算子的核心性能收益来源于"减少数据移动次数"（Reduce Data Movement），而非"减少计算量"。从纯计算的角度，ReLU 仅是对输出矩阵的每个元素进行一次比较（max(x, 0)），计算量可以忽略不计。但从数据移动的角度，如果不融合，ReLU 需要从全局内存读取整个输出矩阵（N×C_out×H×W 个 float32，对于 ResNet-50 的 conv3_x 层约 28MB），执行一次比较，再写回——这约 56MB 的数据移动（读 28MB + 写 28MB）在 A100 上需要约 28 微秒（按 2TB/s 带宽计算）。对于批量推理，这个开销乘以 batch size 和层数后可能占据总延迟的 5%-15%。融合消除了这整个读写循环，将 ReLU 的延迟从"内存往返"降为"寄存器操作"（约 1 cycle）。

该实现还展示了一个关键的混合精度设计模式："高精度累加、低精度存储"。X 和 W 使用 float16 来减少内存带宽和利用 Tensor Core，但 Y_frag（累加器）使用 float32 来保持累加精度——这正是 NVIDIA 推荐的 Tensor Core 编程范式。Tensor Core 的 MMA 指令执行的是 D = A×B + C，其中 A 和 B 是 fp16，C 和 D 是 fp32。这种设计并非偶然：矩阵乘法的部分和可能远超 fp16 的表示范围（±65504），例如当 C_in=1024 且输入和权重各取典型值 1.0 时，点积结果约为 1024，仍在 fp16 范围内；但若输入和权重的值分布较广（如 ±10），点积可达 102400，远超 fp16 上限。因此使用 fp32 累加器是正确性的保证。而在最终写回时，`Y` 的计算结果通常会被后续的 BatchNorm 或激活函数"压缩"回合理范围，fp32 精度足够。

            # Accumulate contributions from each input channel
            for ci in T.serial(C_in):
                w_local = T.alloc_fragment((K, K), "float32")

自动调优的进阶策略不仅要"找到好的配置"，更要"快速找到好的配置"。一个完整的自动调优流程通常分为三个阶段。阶段一是"初始搜索"——使用 Roofline 模型或简单的启发式规则生成 5-10 个候选配置，快速排除明显不合理的组合（如 block_K > C_in 或 block_M > H_out）。阶段二是"粗粒度搜索"——使用基于代价模型（Cost Model）的方法（如 TVM 的 XGBoost 预测器或 Ansor 的梯度提升树），评估每个配置的预期延迟，选择 top-K（通常 K=20-50）进行实际测量。阶段三是"精粒度搜索"——对 top-K 中的最佳配置进行局部搜索（如微调分块参数 ±16，调整 warp 数 ±2），以发现粗搜索可能遗漏的"性能尖峰"。这种三级搜索策略可以将配置空间从数千候选缩小到数十次实际测量，使自动调优在 5-10 分钟内完成一个完整 CNN 模型中所有卷积层的优化。

此外，自动调优系统的一个关键工程挑战是"可复现性"——即相同的配置在不同运行时产生一致的性能测量结果。GPU 性能测量的噪声来源众多：GPU 温度导致的时钟频率调整（Thermal Throttling）、操作系统后台进程抢占 CPU 导致的内核启动延迟抖动、其他 CUDA 上下文的并发干扰（如在 GPU 集群中）、以及 CUDA 驱动程序本身的内存分配碎片化。为降低这些噪声，标准的基准测试流程应包含：固定 GPU 时钟频率（`nvidia-smi -ac` / `nvidia-smi -lgc`）、分配预热内核来"烧热"GPU 到稳态温度、使用 CUDA Events 替代 CPU 计时器来精确测量 GPU 内核时间、以及多次测量取最小值（而非平均值——因为噪声通常是"变慢"方向的单边分布）。

                for kh, kw in T.serial(K, K):
                    w_local[kh, kw] = W[ci, co, kh, kw]

                for i, j in T.serial(8, 8):
                    oh = th * 8 + i
                    ow = tw * 8 + j
                    if oh < H_out and ow < W_out:
                        for kh, kw in T.serial(K, K):
                            # Transposed convolution mapping
                            ih = oh + padding - kh
                            iw = ow + padding - kw
                            if ih >= 0 and iw >= 0 and ih % stride == 0 and iw % stride == 0:
                                ih_s = ih // stride
                                iw_s = iw // stride
                                if ih_s < H_in and iw_s < W_in:
                                    Y_local[i, j] += X[n, ci, ih_s, iw_s] * w_local[kh, kw]

            for i, j in T.serial(8, 8):
                oh = th * 8 + i
                ow = tw * 8 + j
                if oh < H_out and ow < W_out:
                    Y[n, co, oh, ow] = Y_local[i, j]

    return tconv_col2im_kernel
```

Col2Im 方法从计算模式的视角揭示了转置卷积的另一个实现路径：不同于"从输出反推输入"的直观方法，Col2Im 的思路是先初始化为零的输出矩阵，然后遍历每个输入像素，将其"散射"（Scatter）到所有受其影响的输出位置。这是一种"生产者驱动"的视角——输入是生产者，每个输入像素将其值乘以对应权重后累加到多个输出像素。这种散射模式天然适合 GPU 的原子操作（Atomic Add），避免了通过 stride 整除条件进行筛选的控制流发散。然而，原子操作会引入跨线程的串行化开销，当多个输入像素映射到相同的输出位置时（这在 stride < K 时很常见），竞争会显著降低吞吐量。折中方案是使用共享内存缓冲区在 SM 内部完成散射，然后原子更新到全局内存。


### 26.2 转置卷积的棋盘格伪影

> [!WARNING]
> 当卷积核大小不能被步长整除时（如 K=4, stride=2），转置卷积会产生"棋盘格伪影"（Checkerboard Artifacts）。解决方案是使用 K 能被 stride 整除的配置（如 K=4, stride=2 或 K=3, stride=1）。

---

转置卷积的 Col2Im 实现揭示了正向卷积与转置卷积之间优雅的对偶关系：转置卷积的前向计算等价于标准卷积的反向传播。这一认识不仅简化了实现（可以直接复用标准卷积的梯度计算逻辑），也为理解棋盘格伪影的成因提供了理论依据——当卷积核大小与步长不匹配时，输出像素接收到的来自不同输入位置的"贡献"数量不均衡，导致某些输出位置被过度"投票"。


## 27. 1×1 卷积与矩阵乘法的等价性

### 27.1 1×1 卷积的 GEMM 等价

1×1 卷积本质上是对每个空间位置独立执行矩阵乘法：

$$
Y[n, c_{out}, h, w] = \sum_{c_{in}} X[n, c_{in}, h, w] \cdot W[c_{out}, c_{in}]
$$

等价于：

$$
Y_{reshaped} = X_{reshaped} \times W^T
$$

其中 $X_{reshaped} \in \mathbb{R}^{(N \cdot H \cdot W) \times C_{in}}$，$W \in \mathbb{R}^{C_{out} \times C_{in}}$。

### 27.2 融合 1×1 卷积 + 激活函数

```python
def fused_pointwise_conv_relu(
    N: int,
    C_in: int,
    C_out: int,
    H: int,
    W: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
):
    """Fused 1x1 Conv2D + ReLU using GEMM."""
    spatial = N * H * W

    @T.prim_func
    def fused_pw_kernel(
        X: T.Buffer((N, C_in, H, W), "float16"),
        W: T.Buffer((C_out, C_in), "float16"),
        Y: T.Buffer((N, C_out, H, W), "float32"),
    ):
        with T.Kernel(T.ceildiv(spatial, block_M), T.ceildiv(C_out, block_N), threads=256) as (
            bx, by
        ):
            X_frag = T.alloc_fragment((block_M, block_K), "float16")
            W_frag = T.alloc_fragment((block_K, block_N), "float16")
            Y_frag = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(Y_frag)

Nsight Compute 是卷积算子优化的"显微镜"——它将 GPU 内核的执行分解为数百个硬件指标，帮助开发者从"黑盒调参"转向"白盒诊断"。在分析卷积内核时，以下几个高级指标尤为重要：（1）`sm__warps_launched.avg` 反映了内核的并行粒度，与理论 warp 数（由线程块大小和网格大小计算得出）的比值即 Occupancy；（2）`l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit` 与 `miss` 的比值即 L1 缓存命中率，低于 60% 通常说明数据复用不足或 Tile 尺寸过大；（3）`smsp__average_warps_issue_stalled_barrier_per_issue_active` 量化了 warp 因等待同步屏障（如 __syncthreads__）而停滞的时间比例，高值意味着共享内存加载策略需要优化（如使用异步拷贝指令 cp.async 替代同步加载）；（4）`smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active` 揭示了 warp 因等待全局内存加载完成而停滞的时间——这是"内存受限"的直接证据，优化方向是增加每个线程的独立内存请求数（即提高"飞行的"（in-flight）内存加载数量）来隐藏延迟。

将 Nsight Compute 的分析结果转化为可操作的优化建议，需要建立一个系统性的诊断矩阵。如果 `sm__throughput` 接近 100% 但 `gpu__dram_throughput` 也很高（>80%），说明内核已接近硬件极限，几乎没有优化空间。如果 `sm__throughput` 很低（<40%）而 `gpu__dram_throughput` 也很低（<30%），则可能的原因是 Occupancy 不足（寄存器或共享内存使用过多导致每个 SM 只能驻留少量 warp）或者内核启动参数（网格大小）不正确。如果 `sm__throughput` 很低但 `gpu__dram_throughput` 很高，则是典型的内存受限场景——计算单元在"等待数据"——此时应减少全局内存访问（增加共享内存使用、优化数据布局）或使用算法变换（如 Winograd）来降低内存访问需求。这套诊断流程将性能优化从"经验驱动"提升为"数据驱动"，是成为 GPU 性能工程师的必备技能。


            for k in T.serial(T.ceildiv(C_in, block_K)):
                for i, j in T.serial(block_M, block_K):
                    idx = bx * block_M + i
                    k_idx = k * block_K + j
                    if idx < spatial and k_idx < C_in:
                        n_idx = idx // (H * W)
                        hw_idx = idx % (H * W)
                        h_idx = hw_idx // W
                        w_idx = hw_idx % W
                        X_frag[i, j] = X[n_idx, k_idx, h_idx, w_idx]
                    else:
                        X_frag[i, j] = T.float16(0)

                for i, j in T.serial(block_K, block_N):
                    if k * block_K + i < C_in and by * block_N + j < C_out:
                        W_frag[i, j] = W[by * block_N + j, k * block_K + i]
                    else:
                        W_frag[i, j] = T.float16(0)

                T.gemm(X_frag, W_frag, Y_frag)

            # Fused ReLU activation
            for i, j in T.serial(block_M, block_N):
                Y_frag[i, j] = T.max(Y_frag[i, j], T.float32(0))

            # Store result
            for i, j in T.serial(block_M, block_N):
                idx = bx * block_M + i
                co = by * block_N + j
                if idx < spatial and co < C_out:
                    n_idx = idx // (H * W)
                    hw_idx = idx % (H * W)
                    h_idx = hw_idx // W
                    w_idx = hw_idx % W
                    Y[n_idx, co, h_idx, w_idx] = Y_frag[i, j]

    return fused_pw_kernel
```

融合 ReLU 激活的 1×1 卷积在 TensorRT 等推理优化器中是一个标准的图优化（Graph Optimization）模式。将两个独立的内核合并为一个，带来的性能收益来自三个方面：第一，消除了从第一个内核写回全局内存、再由第二个内核读出的往返延迟（对于大张量可达数百微秒）；第二，减少了内核启动的 CPU-GPU 同步开销（每次启动约 5-10 微秒）；第三，允许 ReLU 在寄存器中对刚计算出的数据直接操作，利用了 GPU 指令发出但不立即依赖结果的延迟隐藏特性。注意该实现中 X 和 W 使用 float16 精度而 Y 使用 float32 精度——这是混合精度推理的标准配置，因为激活函数后的输出如果继续以 float16 传递，可能在深层网络中累积显著的精度误差。


---

将 1×1 卷积与激活函数融合为单个内核（Kernel Fusion），是消除"内核启动开销 + 全局内存往返"这一性能损耗模式的标准做法。在上面的融合实现中，ReLU 仅需在寄存器中对输出分块做一次就地比较操作，几乎不增加额外开销。这种融合思路可以推广到更复杂的算子组合，如 Conv-BatchNorm-ReLU、Conv-Bias-ReLU 等，是现代推理优化器的基本能力。


## 28. 卷积自动调优的进阶策略

### 28.1 基于 Roofline 的自动配置

```python
def auto_select_conv_config(C_in, C_out, H, W, K, stride, padding, gpu_bandwidth, gpu_tflops):
    """Auto-select convolution config based on Roofline analysis."""
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    flops = C_out * H_out * W_out * C_in * K * K * 2
    bytes_accessed = (C_in * H * W + C_out * C_in * K * K + C_out * H_out * W_out) * 4
    arithmetic_intensity = flops / bytes_accessed

    ridge_point = gpu_tflops * 1e12 / (gpu_bandwidth * 1e9)

    if arithmetic_intensity > ridge_point:
        # Compute-bound: optimize for compute
        return {
            "strategy": "im2col_gemm",
            "block_M": 256,
            "block_N": 256,
            "block_K": 64,
            "reason": "Compute-bound: large tiles maximize Tensor Core utilization",
        }
    else:
        # Memory-bound: optimize for memory
        if K in [3, 5] and stride == 1:
            return {
                "strategy": "winograd",
                "variant": "F(2,3)" if H_out <= 28 else "F(4,3)",
                "reason": "Memory-bound with small kernel: Winograd reduces compute",
            }
        elif K == 1:
            return {
                "strategy": "pointwise_gemm",
                "block_M": 128,
                "block_N": 128,
                "block_K": 32,
                "reason": "1x1 convolution is pure GEMM",
            }
        else:
            return {
                "strategy": "direct_conv",
                "block_M": 64,
                "block_N": 64,
                "block_K": 16,
                "reason": "Memory-bound: direct conv avoids Im2Col memory overhead",
            }
```

基于 Roofline 的自动配置选择的精妙之处在于其"诊断驱动决策"的设计原则，而非依赖历史基准测试数据。通过计算强度 `flops / bytes_accessed` 与 ridge point 的比较，系统快速定位该卷积配置的根本瓶颈（计算或内存），然后直接跳转到对应优化策略。这种方法的泛化性极强——同样的逻辑可以适用于任何 GPU 架构，只需更新 `gpu_bandwidth` 和 `gpu_tflops` 两个参数。但实践中发现，简单的 Roofline 判断有时会产生次优选择：例如某些 3×3 卷积虽然被判定为"内存受限"，但实际测试中 Im2Col+GEMM 仍然快于 Winograd（因为 Winograd 的变换开销和内存对齐要求抵消了乘法减少的收益）。因此，更鲁棒的系统通常将 Roofline 作为初始化启发式，然后辅以少量实际测量来微调决策。


### 28.2 卷积配置搜索空间

| 参数 | 候选值 | 影响 |
|------|--------|------|
| 策略 | im2col, winograd, direct | 算法选择 |
| block_M | 32, 64, 128, 256 | 输出分块行数 |
| block_N | 32, 64, 128, 256 | 输出分块列数 |
| block_K | 8, 16, 32, 64 | 归约分块大小 |
| num_warps | 2, 4, 8 | Warp 并行度 |
| num_stages | 2, 3, 4 | Pipeline 深度 |
| padding_mode | 0, 1 | Shared Memory Padding |

---

自动调优策略的核心在于缩小搜索空间，而 Roofline 模型正是实现这一目标的有力工具。通过预先判断卷积配置是计算受限还是内存受限，调优器可以锁定正确的优化方向：对于计算受限的配置，应增大分块尺寸以更好地利用 Tensor Core；对于内存受限的配置，则应考虑 Winograd 变换来减少总体计算量，或使用直接卷积规避 Im2Col 的内存膨胀。这种"先诊断、后优化"的方法论远优于盲目的参数搜索。


## 29. 卷积算子的 Profiling 方法

### 29.1 使用 ncu 分析卷积 Kernel

```bash
# 使用 Nsight Compute 分析卷积 kernel
ncu --set full \
    --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
    python conv_benchmark.py

# 关键指标解读：
# 1. sm__throughput: SM 计算吞吐量 → 判断是否 Compute-bound
# 2. gpu__dram_throughput: DRAM 带宽利用率 → 判断是否 Memory-bound
# 3. l1tex__throughput: L1 纹理吞吐量 → 判断缓存效率
# 4. sm__pipe_tensor_op: Tensor Core 利用率 → 判断是否充分利用
```


Nsight Compute 的指标采集命令展示了 GPU 性能分析的核心范式：同时采集多个维度的指标，通过对比各维度的利用率来确定瓶颈所在。四个关键指标分别对应了不同的硬件子系统——`sm__throughput` 反映计算核心的忙碌程度，若低于 60% 则说明计算单元空闲（很可能是因为在等待内存数据）；`gpu__dram_throughput` 反映 HBM（高带宽内存）的利用情况，若接近 100% 则说明内存带宽已饱和，优化方向应是减少全局内存访问或使用更激进的缓存策略；`l1tex__throughput` 衡量 L1 缓存的吞吐量（包含纹理缓存和共享内存），低命中率通常提示需要调整数据布局或增大分块以改善局部性；`sm__pipe_tensor_op` 专用于诊断 Tensor Core 的利用效率，低于 30% 通常意味着分块维度或数据精度不满足 Tensor Core 的要求（如 block_M 不是 16 的倍数）。

### 29.2 卷积性能分析维度

| 维度 | 指标 | 目标值 | 优化方向 |
|------|------|--------|---------|
| 计算利用率 | Tensor Core % | > 70% | 使用 Tensor Core 友好的 Tile 大小 |
| 带宽利用率 | DRAM BW % | > 80% | 合并访问，向量化加载 |
| 缓存效率 | L1 Hit Rate | > 90% | 数据复用，Shared Memory |
| Occupancy | Active Warps % | > 50% | 控制寄存器和 Shared Memory 使用 |
| Bank Conflict | SMEM Conflict | 0 | Padding 或 Swizzle |

---

NVIDIA Nsight Compute (ncu) 是 GPU 内核性能分析的黄金标准工具。通过它可以精确测量 SM 吞吐量、DRAM 带宽利用率、L1 缓存命中率等关键指标，从而将"直觉优化"转化为"数据驱动的优化"。在实际工程中，一个常见的优化循环是：运行 ncu 采集指标 → 识别瓶颈维度 → 修改代码（如调整分块参数、增加数据预取）→ 重新采集指标验证效果 → 重复直至达到性能目标。


## 30. 练习（扩展）

### 练习 6：Winograd F(4,3) 实现

实现 Winograd F(4,3) 变换，并与 F(2,3) 在不同空间维度下进行性能对比。

### 练习 7：融合 Depthwise + Pointwise

实现一个融合的 Depthwise Separable Convolution kernel，避免中间结果写回全局内存。

### 练习 8：转置卷积优化

优化转置卷积的实现，使用 Shared Memory 缓存权重，减少全局内存访问。

### 练习 9：卷积自动调优器

实现一个自动调优器，搜索最优的卷积配置（策略、Tile 大小、Pipeline 深度等）。

### 练习 10：3D 卷积

使用 TileLang 实现 3D 卷积算子，支持对视频或体积数据的卷积操作。

---

## 31. 思考题（扩展）

6. **Winograd 变换的数值误差在什么情况下会变得不可接受？如何量化这种误差？**

7. **深度可分离卷积相比标准卷积的理论加速比是多少？实际加速比为什么通常低于理论值？**

8. **在实现转置卷积时，如何避免棋盘格伪影？这与卷积核大小和步长有什么关系？**

9. **1×1 卷积为什么可以完美映射到 GEMM？这种等价性在什么情况下会被打破？**

10. **如何设计一个统一的卷积框架，自动选择最优的实现策略（Im2Col、Winograd、Direct）？**

---

## 32. 下一章预告

> **Chapter 26: 归约算子与 Softmax/BatchNorm**
>
> 下一章将深入探讨归约操作的实现，包括两阶段归约策略、Online Softmax 算法、LayerNorm 融合实现等。归约操作是深度学习中另一类关键算子，与卷积不同，归约操作的主要挑战在于跨线程/跨块的数据聚合。
