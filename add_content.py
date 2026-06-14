#!/usr/bin/env python3
"""Add Chinese explanations after code blocks and transitional paragraphs between sections."""

import re

with open('/Users/leafoon/Desktop/Note/CS2/content/tilelang/25-convolution-operators.md', 'r') as f:
    content = f.read()

lines = content.split('\n')

# We'll insert content at specific line numbers, working from bottom to top.
# Each insertion is (line_number_after, text_to_insert) where text_to_insert is a list of lines.

insertions = []

# Helper to add text after a line number
def add_after(line_num, text_lines):
    insertions.append((line_num, text_lines))

# ---- Transitional paragraphs between ## sections ----

# Between ## 1 and ## 2 (no transition needed, they're adjacent)

# Between ## 2 (Im2Col) and ## 3 (Winograd) - after line 240 (---)
add_after(239, [
    '',
    '从 Im2Col 到 Winograd 变换，体现了一个重要的系统设计哲学：通过数学等价变换来换取计算效率。Im2Col 将卷积转化为 GEMM，思路直观但付出了内存膨胀的代价；而 Winograd 进一步利用多项式代数的性质，在变换域中以更少的乘法完成等价计算。两种方法的选择取决于问题的规模——当内存不是瓶颈时，Im2Col+GEMM 凭借成熟的 GEMM 优化生态往往是最稳妥的选择；当需要极致性能且卷积核较小时，Winograd 的数学优势就会凸显。',
    '',
])

# Between ## 3 (Winograd) and ## 4 (Direct Conv) - after line 393 (---)
add_after(392, [
    '',
    '然而，Winograd 变换并非万能灵药。其数学简洁性背后隐藏着两个关键限制：一是变换矩阵中非整数系数引入的浮点精度损失，在训练场景中可能被反向传播放大；二是变换要求输入尺寸与输出 tile 大小严格对齐，限制了灵活性。当这些约束无法满足时，直接卷积便成为一种"返璞归真"的选择——不依赖任何数学变换，完全按照卷积的原始定义计算，从而避免额外的内存开销和精度损失。',
    '',
])

# Between ## 4 (Direct Conv) and ## 5 (Tile abstraction) - after line 479, before ## 5
# After the direct conv explanation, before ## 5 starts at 511
add_after(508, [
    '',
    '在掌握了三种主流卷积实现方法之后，我们自然而然要问：TileLang 如何帮助开发者屏蔽这些底层实现的复杂性？答案是 Tile 抽象——它提供了一种声明式的分块计算模型，让开发者专注于描述"做什么"（卷积的计算模式），而将"怎么做"（如何分块、如何映射到 GPU 线程）交给编译器和运行时系统处理。下面我们将看到，同样的卷积逻辑在 Tile 抽象下可以极简地表达，同时不失性能。',
    '',
])

# Between ## 5 (Tile) and ## 6 (Memory optimization) - after line 587 (---)
add_after(586, [
    '',
    'Tile 抽象的价值在于它降低了并行编程的心智负担，但真正的性能优化终究要回归到硬件的物理约束上——内存带宽与延迟。无论使用哪种卷积实现策略，内存访问模式是否高效直接决定了最终性能。尤其对于卷积这种典型的"内存密集型"运算，优化 Padding 处理、Stride 访问和内存层次利用，往往比优化计算本身更能带来性能提升。',
    '',
])

# Between ## 6 (Memory) and ## 7 (Performance comparison) - after line 716 (---)
add_after(715, [
    '',
    '内存访问优化的收益需要通过系统性的性能测量来量化验证。在工程实践中，我们不仅需要知道"哪种方法更快"，更需要理解"为什么快"以及"快多少"。接下来我们将从多个维度对 Im2Col+GEMM、Winograd 和直接卷积进行性能对比，包括不同卷积核大小、不同通道数、不同空间维度下的表现差异。',
    '',
])

# Between ## 7 (Performance) and ## 8 (cuDNN) - after line 777 (---)
add_after(776, [
    '',
    '上述性能对比是在 TileLang 内部进行的，但作为工业级参考，我们还需要将 TileLang 的实现与 NVIDIA cuDNN 进行对标。cuDNN 经过多年的工程积累，在卷积实现中融合了大量硬件级优化技巧（如隐式 GEMM、持久化内核、张量核心指令等）。理解 TileLang 与 cuDNN 的性能差距来源，有助于我们找到进一步优化的方向。',
    '',
])

# Between ## 8 (cuDNN) and ## 9 (Grouped/DW Conv) - after line 853 (---)
add_after(852, [
    '',
    '至此，我们讨论的都是标准的密集卷积（Dense Convolution），即每个输出通道与所有输入通道全连接。然而，现代轻量级网络（如 MobileNet、ShuffleNet）大量使用分组卷积和深度可分离卷积来降低计算量和参数量。这些变体虽然数学形式上更简单，但在 GPU 上高效实现却带来了新的挑战，特别是分组导致的计算密度下降和内存访问分散问题。',
    '',
])

# Between ## 9 (Grouped) and ## 10 (ResNet-50) - after line 949 (---)
add_after(948, [
    '',
    '分组卷积和深度可分离卷积提供了理论上的计算量减少，但如何将它们高效地映射到 GPU 并行架构上，仍然需要仔细的分块策略设计。接下来，让我们通过一个具体而经典的实战案例——ResNet-50 的第一层卷积——来完整展示从一个真实网络层的需求出发，如何在 TileLang 中设计、实现和优化卷积算子。',
    '',
])

# Between ## 10 (ResNet) and ## 11 (Summary) - after line 1000 (---)
add_after(999, [
    '',
    'ResNet-50 第一层卷积的例子很好地说明了一个观点：针对具体问题的定制化实现往往能超越通用方案的性能。该案例中，由于输入通道数极小（C_in=3），Im2Col 的内存膨胀问题并不严重，反而是直接展开计算循环、减少分支判断的策略带来了更好的性能表现。这提醒我们在设计卷积实现时，不应当迷信某一种"最佳策略"，而应该根据实际的输入规模、卷积核大小和硬件特性做出动态选择。',
    '',
])

# Between ## 11 (Summary) and ## 12 (Exercises) is just ---, no transition needed

# Between ## 14 (Further reading) and ## 15 (Transposed Conv) - after line 1070 (---)
add_after(1069, [
    '',
    '在掌握了标准卷积（前向）的多种实现策略之后，我们将目光转向卷积算子的几个重要变体。转置卷积（也称为反卷积）在语义分割、图像生成等需要上采样的任务中扮演着核心角色。它的数学本质与标准卷积存在精确的对偶关系——事实上，标准卷积的反向传播过程就是一个转置卷积——这为我们在 TileLang 中高效实现它提供了重要的思路。',
    '',
])

# Between ## 15 (Transposed) and ## 16 (Dilated) - after line 1147 (---)
add_after(1146, [
    '',
    '转置卷积解决了"上采样"问题，而空洞卷积（Dilated/Atrous Convolution）解决的则是完全不同的需求：在不增加参数量的前提下扩大感受野。空洞卷积通过在卷积核元素之间插入"空洞"（跳过的像素），使得一个 3×3 的卷积核可以获得 5×5、7×7 甚至更大的有效感受野。这种特性使其在语义分割任务中广泛使用，因为大感受野对于理解全局上下文至关重要。',
    '',
])

# Between ## 16 (Dilated) and ## 17 (Deformable) - after line 1209 (---)
add_after(1208, [
    '',
    '空洞卷积通过固定的采样模式扩大感受野，但仍然受到规则网格的限制。可变形卷积（Deformable Convolution）则更进一步，通过引入可学习的空间偏移量，使卷积核的采样位置可以根据输入内容动态调整。这赋予了网络对几何变换的自适应能力，但同时也带来了不规则内存访问模式和双线性插值等新的计算挑战。',
    '',
])

# Between ## 17 (Deformable) and ## 18 (1x1 Conv) - after line 1295 (---)
add_after(1294, [
    '',
    '可变形卷积展现了卷积概念的灵活性，但在实际的深度学习模型（尤其是 MobileNet 等轻量级架构）中，1×1 卷积（Pointwise Convolution）的使用频率远高于可变形卷积。1×1 卷积虽然名为"卷积"，但本质上不涉及任何空间邻域运算，而是对每个空间位置独立执行的矩阵乘法。这一特性使其可以完美映射到 GEMM，从而在 GPU 上获得极高的计算效率。',
    '',
])

# Between ## 18 (1x1) and ## 19 (Auto-tuning) - after line 1366 (---)
add_after(1365, [
    '',
    '1×1 卷积的高效实现得益于它与矩阵乘法的等价性，但其他类型的卷积（3×3、5×5、7×7）无法如此简单地规约为 GEMM。面对不同的卷积配置（卷积核大小、通道数、空间维度），最优的实现策略和分块参数各不相同。手工为每种配置调优是不现实的，因此自动调优（Auto-Tuning）成为生产级卷积实现中不可或缺的一环。',
    '',
])

# Between ## 19 (Auto-tuning) and ## 20 (Roofline) - after line 1411 (---)
add_after(1410, [
    '',
    '自动调优通过实验搜索最优参数，但盲目的暴力搜索成本高昂。一个更高效的方法是通过性能模型（如 Roofline 模型）预先判断卷积配置的计算瓶颈类型（计算受限还是内存受限），从而大幅缩小搜索空间。Roofline 模型将硬件的峰值计算能力和峰值内存带宽作为天花板，通过计算强度（FLOPs/Byte）来定位性能瓶颈，是卷积性能分析的基础工具。',
    '',
])

# Between ## 20 (Roofline) and ## 21 (MobileNetV2) - after line 1467 (---)

# Let me skip this one, the content doesn't have a clear ##21 heading between these. Wait, it does: ## 21 at line 1469.

# After ##21 inverted_residual_block code block, before ##22 summary - after line 1510
add_after(1509, [
    '',
    'MobileNetV2 倒残差块的实现展示了三种卷积类型（1×1 扩展、3×3 深度可分离、1×1 投影）在一个算子中协同工作的模式。在实际部署中，将这三个操作融合为一个内核（Kernel Fusion）可以避免中间结果的全局内存写入，大幅降低带宽需求。这种多操作融合的思想是高性能推理引擎（如 TensorRT、TVM）的核心优化手段之一。',
    '',
])

# Between ## 22 (Summary extended) and ## 24 (Winograd F43) - after line 1535 (---)
add_after(1534, [
    '',
    '通过前面的全面对比，我们可以看到卷积算子的设计本质上是一系列权衡（Trade-off）的集合：计算量 vs 内存开销、通用性 vs 专用性、实现复杂度 vs 性能上限。然而，除了 F(2,3) 之外，Winograd 变换家族还有更高效的变体。F(4,3) 变换能进一步将乘法次数从 F(2,3) 的减少 33% 提升到减少 50%，但代价是变换矩阵更复杂、数值误差更大、内存占用更高。',
    '',
])

# Between ## 24 (Winograd F43) and ## 25 (Depthwise optimization) - after line 1692 (---)
add_after(1691, [
    '',
    'Winograd 变换的 F(4,3) 变体在理论上有更高的乘法减少率，但实践中其性能表现高度依赖于输入的空间维度。当空间维度较大时，更大的输出 tile（4×4 vs 2×2）意味着更少的变换开销摊销，因此 F(4,3) 更具优势。而当空间维度较小时，变换开销主导了总耗时，F(2,3) 反而是更好的选择。接下来，我们将转向深度可分离卷积的优化，这类算子在 MobileNet 系列模型中占据了绝大部分计算量。',
    '',
])

# Between ## 25 (Depthwise opt) and ## 26 (Transposed Im2Col) - after line 1776 (---)
add_after(1775, [
    '',
    '深度可分离卷积的优化揭示了共享内存（Shared Memory）在 GPU 编程中的核心地位：通过将输入数据以 Halo 区域的方式加载到共享内存中，相邻线程可以高效复用重叠的数据区域，从而将全局内存访问减少为原来的 1/(K_h × K_w)。然而，共享内存的容量有限（典型值约 164KB/SM），当通道数较多或空间维度较大时，需要仔细设计分块策略以在占用率（Occupancy）和数据复用之间取得平衡。',
    '',
])

# Between ## 26 (Transposed Im2Col) and ## 27 (1x1 = GEMM) - after line 1846 (---)
add_after(1845, [
    '',
    '转置卷积的 Col2Im 实现揭示了正向卷积与转置卷积之间优雅的对偶关系：转置卷积的前向计算等价于标准卷积的反向传播。这一认识不仅简化了实现（可以直接复用标准卷积的梯度计算逻辑），也为理解棋盘格伪影的成因提供了理论依据——当卷积核大小与步长不匹配时，输出像素接收到的来自不同输入位置的"贡献"数量不均衡，导致某些输出位置被过度"投票"。',
    '',
])

# Between ## 27 (1x1=GEMM) and ## 28 (Auto-tuning advanced) - after line 1935 (---)
add_after(1934, [
    '',
    '将 1×1 卷积与激活函数融合为单个内核（Kernel Fusion），是消除"内核启动开销 + 全局内存往返"这一性能损耗模式的标准做法。在上面的融合实现中，ReLU 仅需在寄存器中对输出分块做一次就地比较操作，几乎不增加额外开销。这种融合思路可以推广到更复杂的算子组合，如 Conv-BatchNorm-ReLU、Conv-Bias-ReLU 等，是现代推理优化器的基本能力。',
    '',
])

# Between ## 28 (Auto-tuning) and ## 29 (Profiling) - after line 2000 (---)
add_after(1999, [
    '',
    '自动调优策略的核心在于缩小搜索空间，而 Roofline 模型正是实现这一目标的有力工具。通过预先判断卷积配置是计算受限还是内存受限，调优器可以锁定正确的优化方向：对于计算受限的配置，应增大分块尺寸以更好地利用 Tensor Core；对于内存受限的配置，则应考虑 Winograd 变换来减少总体计算量，或使用直接卷积规避 Im2Col 的内存膨胀。这种"先诊断、后优化"的方法论远优于盲目的参数搜索。',
    '',
])

# Between ## 29 (Profiling) and ## 30 (Exercises) - after line 2033 (---)
add_after(2032, [
    '',
    'NVIDIA Nsight Compute (ncu) 是 GPU 内核性能分析的黄金标准工具。通过它可以精确测量 SM 吞吐量、DRAM 带宽利用率、L1 缓存命中率等关键指标，从而将"直觉优化"转化为"数据驱动的优化"。在实际工程中，一个常见的优化循环是：运行 ncu 采集指标 → 识别瓶颈维度 → 修改代码（如调整分块参数、增加数据预取）→ 重新采集指标验证效果 → 重复直至达到性能目标。',
    '',
])

# ===== CODE BLOCK EXPLANATIONS =====
# Block 5: Loop unrolling (lines 484-491), ends at line 491. Next text at 493 (#### 数据复用优化)
add_after(491, [
    '',
    '循环展开是卷积优化的经典手段，尤其适用于 3×3 这类小卷积核。当卷积核维度在编译期已知且较小时，将嵌套循环显式展开为顺序执行的语句序列，一方面减少了循环控制的开销（索引计算、分支判断），另一方面为编译器提供了更大的指令级并行优化空间。值得注意的是，展开后的代码会显著增大寄存器压力——对于 3×3 卷积，每个输出像素需要维护 9 次乘加的中间结果，如果同时配合通道分块，寄存器使用量可能迅速接近 SM 的物理限制（每线程约 255 个 32 位寄存器），因此在实践中需要在展开程度与寄存器占用之间谨慎权衡。',
    '',
])

# Block 6: Data reuse optimization (lines 495-507), ends at line 507. Next is --- then ## 5
add_after(507, [
    '',
    '沿 C_in 维度分块是卷积权重数据复用的核心策略。由于同一权重块可以被多个空间位置的输出复用，将权重加载到寄存器或共享内存后反复使用，可以大幅减少全局内存访问。上述代码中的"外循环沿 C_in 分块、内循环遍历空间维度"的模式，本质上是将卷积的归约维度（reduction dimension）作为分块维度，从而在保持计算正确性的同时最大化数据局部性。这种模式的极端形式就是 Im2Col+GEMM 方法——将整个 C_in × K_h × K_w 展开为 GEMM 的 K 维度，通过矩阵乘法的分块策略自动获得最优的数据复用模式。',
    '',
])

# Block 7: tiled_conv (lines 517-534), ends at line 534. Next is ### 5.2
add_after(534, [
    '',
    '这段代码展示了 TileLang 的核心抽象能力：开发者只需声明计算逻辑（卷积的乘累加），而分块（Tiling）、线程映射、边界处理等底层细节由 Tile 抽象自动管理。`T.grid` 原语将外层循环迭代映射到 GPU 的线程块和线程网格，`T.block` 声明一个计算块，其中的 `T.sum` 在归约维度上自动生成高效的并行归约代码。这种声明式的编程范式使开发者无需手动处理线程索引计算和同步原语，大幅降低了 GPU 编程的入门门槛，同时保持了生成代码的性能——因为 TileLang 的编译器可以在后端针对特定硬件（如 A100 vs H100、CUDA vs ROCm）自动选择最优的分块策略。',
    '',
])

# Block 8: multi_level_tiled_conv (lines 539-559), ends at line 560. Next is div
add_after(559, [
    '',
    '多级分块策略是高性能 GPU 编程的精髓，它直接映射到 GPU 的硬件层次结构：SM 级分块（Level 1）利用大容量共享内存和 L2 缓存实现跨线程块的数据复用，Warp 级分块（Level 2）利用 warp shuffle 指令在 32 个线程间高效交换数据，线程级分块（Level 3）则充分利用寄存器文件的最低延迟访问。三个级别的分块参数（128、8×4×2×2、32×4×4 等）共同决定了内核的占用率、寄存器压力和共享内存使用量，它们之间存在着复杂的相互制约关系——增大某个级别的分块可能提高数据复用但降低占用率，最佳配置需要通过自动调优来确定。',
    '',
])

# Block 9: parallel_conv (lines 567-585), ends at line 586. Next is --- then ## 6
add_after(586, [
    '',
    '这个实例展示了 TileLang 中 `T.Kernel` 原语的自动并行化能力。通过将卷积的输出维度（C_out, H_out, W_out）声明为 Kernel 的迭代空间，TileLang 自动将这些迭代映射到 GPU 的线程网格上，每个线程负责计算一个输出像素。这种"一个线程一个输出"的映射方式对于大空间维度的卷积非常自然，但对于小卷积核场景，这种映射可能导致大量的冗余全局内存访问——因为相邻输出像素共享大量的输入数据，但每个线程都独立地从全局内存加载。后续章节中我们将看到如何通过共享内存来解决这一数据复用问题。',
    '',
])

# Block 10: conditional padding (lines 596-602), ends at line 602. Next is #### 策略二
add_after(602, [
    '',
    '条件判断 Padding 是实现上最简单但在性能上代价最高的方式。每个线程在每次内存访问前都需要执行分支判断，这不仅增加了指令开销，更严重的是导致 warp 内线程发散（Warp Divergence）——位于边界区域的线程进入 padding 分支（赋零），而内部区域的线程进入数据加载分支，GPU 需要串行执行两个分支路径，使 warp 的有效利用率减半。对于大输入尺寸的卷积，边界区域的线程占比很小，这种开销尚可接受；但对于小输入尺寸或大卷积核的配置，边界线程的比例可能超过 20%，性能损失显著。',
    '',
])

# Block 11: optimized_padding_conv (lines 606-634), ends at line 635. Next is ### 6.2
add_after(634, [
    '',
    '分块 Padding 策略通过将内部区域（Interior Region）和边界区域（Boundary Region）分开处理，巧妙地消除了 warp 发散问题。对于内部区域，由于所有访问都保证在有效范围内，可以完全省略边界检查（if 分支），从而让 GPU 的所有线程以完全一致的控制流执行，达到峰值内存带宽利用率。代价是需要两个独立的内核（或两个独立的条件区域）来处理边界情况，这会增加代码复杂度和可能的 CPU 端调度开销。在实践中，通常将两者融合为条件分派：在线程块索引计算阶段判断当前块是否完全在内部区域，若是则走快速路径，否则走带边界检查的安全路径。',
    '',
])

# Block 12: stride code (lines 643-647), ends at line 647. Next is #### 解决方案
add_after(647, [
    '',
    '当 stride > 1 时，输出特征图的空间维度缩小，但每个输出像素的输入采样点变得更加分散。以 stride=2 为例，连续两个输出位置（oh=0, ow=0）和（oh=0, ow=1）需要访问的输入位置分别是（ih=0, iw=0）和（ih=0, iw=2），中间位置（ih=0, iw=1）被完全跳过。这意味着 GPU 的缓存行（通常 128 字节）中可能有一半的数据永远不会被使用——即缓存利用率仅为 50%。在极端情况下（如 stride=4），缓存利用率可能降至 25%，严重拖累实际带宽。',
    '',
])

# Block 13: strided_conv_optimized (lines 651-694), ends at line 695. Next is ### 6.3
add_after(694, [
    '',
    '共享内存预取策略是解决 Stride 导致的非连续访问问题的标准方法。核心思路是将输入数据先以"稠密方式"加载到共享内存中（按 stride=1 加载，保证每个缓存行都被充分利用），然后让所有线程从共享内存中以任意 stride 访问。共享内存的延迟（~20 cycles）虽高于寄存器但远低于全局内存（~400 cycles），更重要的是共享内存不受缓存行粒度的限制——每次访问仅读取实际需要的 4 字节（对于 float32），不存在"浪费"的带宽。不过需要注意，共享内存容量有限，`block_M * stride + K_h - 1` 的尺寸设计意味着 stride 越大，所需的共享内存尺寸也越大，可能成为限制 Occupancy 的瓶颈。',
    '',
])

# Block 14: choose_conv_strategy (lines 757-775), ends at line 775. Next is --- then ## 8
add_after(775, [
    '',
    '这段选择策略函数体现了生产级卷积调度的核心决策逻辑：首先判断是否可以使用 Winograd（仅支持 3×3 和 5×5 卷积核，且空间维度需为偶数），然后计算 Im2Col 的内存开销是否超出预算，若超出则回退到直接卷积。这个决策树在实际应用中可以被进一步细化：例如加入对通道数的判断（当 C_in 很大时直接卷积的归约循环会更高效）、对 stride 的判断（stride > 1 时 Winograd 需要额外处理）、以及对硬件类型的判断（Tensor Core 友好的硬件上，Im2Col+GEMM 通过 fp16 精度可以获得额外加速）。自动调优系统的目标正是将这类启发式规则系统化、自动化。',
    '',
])

# Block 15: benchmark_conv (lines 783-816), ends at line 816. Next is ### 8.2
add_after(816, [
    '',
    '基准测试的实现遵循了 GPU 性能测量的最佳实践：预热阶段（warmup）消除首次调用的 CUDA 上下文初始化和内核 JIT 编译开销，多次重复执行（repeat=100）降低单次测量的随机误差，`torch.cuda.synchronize()` 确保在时间测量之前所有 GPU 操作已完成（因为 CUDA 内核调用是异步的，不等待会严重低估实际执行时间）。需要注意的是，该测试假设输入数据已经驻留在 GPU 显存中，未计入 CPU→GPU 的数据传输时间——在实际部署中，数据传输可能成为端到端延迟的瓶颈，需要采用流水线（Pipelining）或双缓冲（Double Buffering）策略来隐藏传输开销。',
    '',
])

# Block 16: grouped_conv2d (lines 859-897), ends at line 898. Next is ### 9.2
add_after(897, [
    '',
    '分组卷积的实现看似简单——只需在通道维度上增加一个 group 索引——但在并行效率上存在根本性挑战。标准卷积的 C_in × C_out 归约提供了巨大的并行空间（数千到数十万次乘加），而分组卷积将通道分成 g 个独立组，每组内部的归约维度缩小为原来的 1/g。当分组数较大（如 g=32 或 g=C_in 的深度可分离卷积），每组内部的归约工作量可能只有几个乘加操作，使得单个线程几乎没有计算量来隐藏内存延迟。在这种情况下，传统的"一个线程一个输出"映射方式会导致严重的利用率不足，需要通过 Warp 级别的协作（如 warp shuffle 归约）来提升计算密度。',
    '',
])

# Block 17: depthwise_separable_conv2d (lines 904-947), ends at line 948. Next is --- then ## 10
add_after(947, [
    '',
    '深度可分离卷积的两阶段实现（Depthwise + Pointwise）精确映射了这一算子的设计初衷：Depthwise 阶段在空间维度上独立处理每个通道，计算量极小但通道间无交互；Pointwise 阶段通过 1×1 卷积恢复通道间的信息流动。从内存角度分析，Pointwise 阶段的计算强度（FLOPs/Byte）远高于 Depthwise 阶段——1×1 卷积的每次权重加载可以被 (H_out × W_out) 个输出位置复用，而 Depthwise 阶段每次权重加载仅服务于一个输出位置。因此在实际优化中，Depthwise 阶段通常需要更多的共享内存投入，而 Pointwise 阶段则应优先利用 Tensor Core 来加速矩阵乘法。',
    '',
])

# Block 18: resnet50_conv1 (lines 955-997), ends at line 998. Next is ---
add_after(997, [
    '',
    'ResNet-50 第一层卷积的特殊性在于 C_in=3（RGB 三个通道），远小于标准卷积的典型通道数（64、128、256 等）。在输入通道数极小的情况下，直接展开 C_in 维度的循环（而非将其作为归约维度分块）是更明智的选择——因为 C_in=3 意味着整个通道维度的数据可以轻松装入寄存器，避免了对共享内存的依赖，也消除了沿 C_in 分块所需的循环开销。此外，该层的 stride=2 导致输出空间维度减半（224→112），但输入空间维度较大（224×224），因此内存访问仍然是主要瓶颈，优化方向应聚焦于全局内存的合并访问模式。',
    '',
])

# Block 19: transposed_conv2d (lines 1086-1134), ends at line 1134. Next is ### 15.3
add_after(1133, [
    '',
    '转置卷积实现在索引计算上的复杂度远高于标准卷积。关键挑战在于：输出位置（oh, ow）与输入位置（ih, iw）之间并非一一对应，而是需要一个"反算"过程——从输出坐标推导出可能贡献的输入坐标，并检查其是否在有效范围内。条件 `ih % stride == 0 and iw % stride == 0` 是最核心的判断逻辑：只有当输出的采样位置恰好落在"有效"的输入格点上时，该输入才对输出有贡献。这意味着对于 stride > 1 的转置卷积，每个输出像素实际接收到的"贡献"数量是不均匀的——这正是棋盘格伪影产生的根本原因。在实现层面，这导致 warp 内的控制流严重发散，因为某些线程会跳过大部分输入通道的计算。',
    '',
])

# Block 20: dilated_conv2d (lines 1163-1207), ends at line 1207. Next is --- then ## 17
add_after(1206, [
    '',
    '空洞卷积在实现上看似仅需将索引计算中的 `kh` 替换为 `kh * dilation`，但这看似微小的变化对内存访问模式产生了深远影响。当 dilation=2 时，一个 3×3 卷积核的实际采样范围扩展到了 5×5 的空间区域（有效核大小为 `effective_K = K + (K-1)*(dilation-1) = 5`），但每个输出像素仍然只访问 9 个采样点（而非 5×5=25 个）。这种"稀疏采样"模式导致两个问题：一是输入数据在内存中不再连续（相邻采样点之间间隔 dilation 个像素），降低了缓存行利用率；二是在共享内存加载优化中，"Halo 区域"的计算必须使用有效的膨胀核大小（effective_K），使得所需共享内存增大但实际使用的数据比例降低。对于大 dilation 值，Im2Col+GEMM 方法（将空洞卷积展开为稀疏的列向量）通常是更高效的选择。',
    '',
])

# Block 21: deformable_conv2d (lines 1225-1293), ends at line 1293. Next is --- then ## 18
add_after(1292, [
    '',
    '可变形卷积是本章中计算模式最复杂的算子，其核心难点在于双线性插值的实现。由于每个采样位置的偏移量（delta_h, delta_w）是连续的浮点数，采样坐标往往落在像素之间的亚像素位置，需要通过四个最近邻像素的加权平均来计算采样值。代码中的双线性插值涉及四次内存访问和四次乘加，使得每个输出像素的计算量是标准卷积的 4 倍以上（在 3×3 卷积中，每个输出需要 3×3×4=36 次内存访问 vs 标准卷积的 9 次）。此外，不可预测的偏移量意味着编译器和硬件无法预取数据，缓存预取机制基本失效，导致实际的全局内存延迟接近最坏情况。这种特性使得可变形卷积通常成为模型中的性能瓶颈，实践中常通过限制偏移量范围、或使用查找表预计算采样模式来进行近似加速。',
    '',
])

# Block 22: pointwise_conv2d (lines 1309-1364), ends at line 1364. Next is --- then ## 19
add_after(1363, [
    '',
    '1×1 卷积的 GEMM 映射展示了一个重要的设计洞察：空间维度的扁平化处理。通过将（N, H, W）三个维度展平为单一的空间维度（spatial = N*H*W），1×1 卷积被精确转化为一个（spatial × C_in）×（C_in, C_out）的矩阵乘法。在实际实现中，空间维度的展开顺序可能显著影响性能——先展平 H 再展平 W 可以保证相邻的空间位置（沿 W 方向）在内存中连续，从而提高全局内存的合并访问效率。此外，该实现中使用 float16 精度存储输入和权重（`T.float16`），配合 float32 累加器，是混合精度训练的典型模式，可以在 A100 GPU 上利用 Tensor Core 获得高达 312 TFLOPS 的矩阵乘法吞吐量。',
    '',
])

# Block 23: auto_tune_conv (lines 1372-1399), ends at line 1399. Next is ### 19.2
add_after(1398, [
    '',
    '自动调优的实现采用了一个简单的暴力搜索（Grid Search）策略：遍历所有候选配置，对每个配置编译并测量内核执行时间，选择延迟最小的配置。虽然暴力搜索在搜索空间较小时（如分块参数只有 4 种候选）是可行的，但在生产环境中配置空间通常包含 5-7 个参数，总组合数可能达到数千甚至数万，此时需要更高效的搜索策略。常见的高级方法包括：基于贝叶斯优化的调优器（如 Hyperopt）、基于代价模型的搜索（如 TVM 的 AutoScheduler 使用 XGBoost 预测配置性能）、以及基于遗传算法的进化搜索。此外，代码中的 `try-except` 捕获说明某些配置可能因为寄存器溢出或共享内存超限而编译失败——在搜索空间中，有效配置的比例可能只有 30%-50%，一个高效的调优器应能快速跳过无效配置。',
    '',
])

# Block 24: roofline_analysis_conv (lines 1417-1455), ends at line 1455. Next is ### 20.2
add_after(1454, [
    '',
    'Roofline 分析的核心价值在于提供一个"天花板"视角——无论代码如何优化，性能都不可能超过硬件峰值计算能力或内存带宽所设定的上限。`compute_bound_threshold = peak_bandwidth / peak_compute`（约 6.4 FLOPs/Byte 对于 A100）是一个关键的决策边界：当卷积配置的计算强度超过此阈值时，瓶颈在计算单元；低于此阈值时，瓶颈在内存带宽。这一诊断结果直接引导优化方向——如果是计算受限，应增大分块参数以更好地利用 Tensor Core 和指令级并行；如果是内存受限，应通过数据复用（共享内存缓存、权重预加载）或算法变换（Winograd）来减少全局内存访问量。不过需要注意的是，Roofline 模型假设算术运算和数据传输可以完全重叠，而实际硬件的重叠能力受限于 warp 调度和缓存容量，因此模型预测的"理论上限"通常需要乘以 0.7-0.85 的修正系数。',
    '',
])

# Block 25: inverted_residual_block (lines 1474-1508), ends at line 1508. Next is --- then ## 22
add_after(1507, [
    '',
    'MobileNetV2 倒残差块的"倒残差"命名来源于其与标准残差块相反的通道变换方向：标准残差块先压缩（1×1 降维）再扩展（3×3）再压缩（1×1 升维），而倒残差块先扩展（1×1 升维 6 倍）再空间卷积（3×3 Depthwise）再压缩（1×1 降维）。这种"先宽后窄"的设计使得 3×3 Depthwise 卷积在一个高维空间中操作，从而能够表达更丰富的空间特征。从 GPU 实现的角度看，扩展阶段的 1×1 卷积是计算重头（从 C_in 到 6×C_in 的矩阵乘法），而 Depthwise 阶段虽然计算量小但内存访问密集——两者的计算特征截然不同，因此融合实现中需要为不同阶段分配不同比例的共享内存和线程资源。',
    '',
])

# Block 26: winograd_f43_conv2d (lines 1584-1675), ends at line 1675. Next is ### 24.3
add_after(1674, [
    '',
    'Winograd F(4,3) 在 F(2,3) 的基础上将输出 tile 从 2×2 扩大到 4×4，乘法的理论减少比例从 33% 提升到 50%，但代价是变换矩阵从 4×4（或 4×3）扩大到 6×6（或 6×3）。这带来了两个直接后果：第一，输入变换的计算量显著增加——F(4,3) 的 B^T·d·B 变换需要处理 6×6 的中间矩阵，比 F(2,3) 的 4×4 多了约 2.25 倍的运算；第二，变换域的中间缓冲区（V 和 M）尺寸也相应增大，内存占用增加。因此 F(4,3) 仅在"输出维度足够大，能够摊销变换开销"时才有优势，通常建议在 H_out ≥ 28 且 W_out ≥ 28 时使用。另外注意代码中 assert 要求 H 和 W 必须是 4 的倍数，这是 F(4,3) 对输入对齐的硬性要求，在实际部署中可能需要在输入前进行裁剪或填充。',
    '',
])

# Block 27: optimized_depthwise_conv2d (lines 1700-1765), ends at line 1765. Next is ### 25.2
add_after(1764, [
    '',
    '这段优化实现抓住了 Depthwise 卷积的核心优化机会：每个通道的卷积核（K×K）是独立且较小（通常 3×3），可以被整个加载到寄存器中（`w_local`）。将权重保持在寄存器中意味着在 K×K 次乘加循环内，每次权重访问仅需 1 个时钟周期，而非从共享内存的约 20 个周期或全局内存的约 400 个周期。共享内存部分（`X_shared`）用于缓存输入数据的 Halo 区域——尺寸为 `(BLOCK_H*stride+K-1) × (BLOCK_W*stride+K-1)` 而非 `BLOCK_H × BLOCK_W`，多出的边界部分 (K-1) 使得每个输出 tile 所需的所有输入数据都在共享内存中，从而避免了跨 tile 的重复全局内存加载。`T.sync_threads()` 是一个关键屏障，确保所有线程完成共享内存的写入之后才开始读取——缺少此同步会导致数据竞争和错误的计算结果。',
    '',
])

# Block 28: transposed_conv2d_col2im (lines 1784-1839), ends at line 1839. Next is ### 26.2
add_after(1838, [
    '',
    'Col2Im 方法从计算模式的视角揭示了转置卷积的另一个实现路径：不同于"从输出反推输入"的直观方法，Col2Im 的思路是先初始化为零的输出矩阵，然后遍历每个输入像素，将其"散射"（Scatter）到所有受其影响的输出位置。这是一种"生产者驱动"的视角——输入是生产者，每个输入像素将其值乘以对应权重后累加到多个输出像素。这种散射模式天然适合 GPU 的原子操作（Atomic Add），避免了通过 stride 整除条件进行筛选的控制流发散。然而，原子操作会引入跨线程的串行化开销，当多个输入像素映射到相同的输出位置时（这在 stride < K 时很常见），竞争会显著降低吞吐量。折中方案是使用共享内存缓冲区在 SM 内部完成散射，然后原子更新到全局内存。',
    '',
])

# Block 29: fused_pointwise_conv_relu (lines 1869-1933), ends at line 1933. Next is --- then ## 28
add_after(1932, [
    '',
    '融合 ReLU 激活的 1×1 卷积在 TensorRT 等推理优化器中是一个标准的图优化（Graph Optimization）模式。将两个独立的内核合并为一个，带来的性能收益来自三个方面：第一，消除了从第一个内核写回全局内存、再由第二个内核读出的往返延迟（对于大张量可达数百微秒）；第二，减少了内核启动的 CPU-GPU 同步开销（每次启动约 5-10 微秒）；第三，允许 ReLU 在寄存器中对刚计算出的数据直接操作，利用了 GPU 指令发出但不立即依赖结果的延迟隐藏特性。注意该实现中 X 和 W 使用 float16 精度而 Y 使用 float32 精度——这是混合精度推理的标准配置，因为激活函数后的输出如果继续以 float16 传递，可能在深层网络中累积显著的精度误差。',
    '',
])

# Block 30: auto_select_conv_config (lines 1942-1986), ends at line 1986. Next is ### 28.2
add_after(1985, [
    '',
    '基于 Roofline 的自动配置选择的精妙之处在于其"诊断驱动决策"的设计原则，而非依赖历史基准测试数据。通过计算强度 `flops / bytes_accessed` 与 ridge point 的比较，系统快速定位该卷积配置的根本瓶颈（计算或内存），然后直接跳转到对应优化策略。这种方法的泛化性极强——同样的逻辑可以适用于任何 GPU 架构，只需更新 `gpu_bandwidth` 和 `gpu_tflops` 两个参数。但实践中发现，简单的 Roofline 判断有时会产生次优选择：例如某些 3×3 卷积虽然被判定为"内存受限"，但实际测试中 Im2Col+GEMM 仍然快于 Winograd（因为 Winograd 的变换开销和内存对齐要求抵消了乘法减少的收益）。因此，更鲁棒的系统通常将 Roofline 作为初始化启发式，然后辅以少量实际测量来微调决策。',
    '',
])

# Block 31: bash ncu (lines 2006-2021), ends at line 2021. Next is ### 29.2
add_after(2021, [
    '',
    'Nsight Compute 的指标采集命令展示了 GPU 性能分析的核心范式：同时采集多个维度的指标，通过对比各维度的利用率来确定瓶颈所在。四个关键指标分别对应了不同的硬件子系统——`sm__throughput` 反映计算核心的忙碌程度，若低于 60% 则说明计算单元空闲（很可能是因为在等待内存数据）；`gpu__dram_throughput` 反映 HBM（高带宽内存）的利用情况，若接近 100% 则说明内存带宽已饱和，优化方向应是减少全局内存访问或使用更激进的缓存策略；`l1tex__throughput` 衡量 L1 缓存的吞吐量（包含纹理缓存和共享内存），低命中率通常提示需要调整数据布局或增大分块以改善局部性；`sm__pipe_tensor_op` 专用于诊断 Tensor Core 的利用效率，低于 30% 通常意味着分块维度或数据精度不满足 Tensor Core 的要求（如 block_M 不是 16 的倍数）。',
    '',
])

# Now apply all insertions from bottom to top
insertions.sort(key=lambda x: x[0], reverse=True)

for line_num, text_lines in insertions:
    # line_num is 0-indexed in our code, but we stored 0-indexed line numbers
    # Insert after line_num
    insertion_point = line_num + 1
    for i, text in enumerate(text_lines):
        lines.insert(insertion_point + i, text)

with open('/Users/leafoon/Desktop/Note/CS2/content/tilelang/25-convolution-operators.md', 'w') as f:
    f.write('\n'.join(lines))

# Count lines added
print(f"Total lines after edit: {len(lines)}")
print(f"Lines added: {len(lines) - 2077}")
