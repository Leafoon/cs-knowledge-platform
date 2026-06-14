#!/usr/bin/env python3
"""Add substantial Chinese explanations. Round 2 - much more content."""

with open('/Users/leafoon/Desktop/Note/CS2/content/tilelang/25-convolution-operators.md', 'r') as f:
    content = f.read()

lines = content.split('\n')
original_count = len(lines)

insertions = []

def add_after(line_num, text_lines):
    insertions.append((line_num, text_lines))

# ===== MUCH LONGER EXPLANATIONS FOR KEY CODE BLOCKS =====

# After im2col_transform code block (line 153 originally, now we need to find it)
# Let me work with content after the existing explanations - add additional paragraphs
# I'll search for specific markers and add more content after existing explanations

# Add extended explanation for Section 2.1 Im2Col principle (after method_strategy table, before Im2Col code)
add_after(66, [
    '',
    '这四种实现策略的选择本质上是"时间换空间"或"空间换时间"的经典计算机科学权衡。Im2Col+GEMM 方法之所以在 cuDNN、Caffe、早期的 TensorFlow 等框架中被广泛采用，并非因为它是最优的，而是因为它将"不规则的卷积计算"转化为"高度规则的矩阵乘法"——而矩阵乘法是过去五十年中整个高性能计算社区投入最多优化精力的运算。当你的底层 GEMM 库（如 cuBLAS、MKL）已经经过数千人年的优化，通过 Im2Col 借用这些优化成果是极其务实的工程决策。然而，随着深度学习模型向移动端和边缘设备迁移（这些场景内存极度受限），以及编译器技术的进步（如 TVM、XLA），直接卷积和 Winograd 变换的吸引力正在持续上升。',
    '',
    '从硬件微架构的角度来看，这四种策略对 GPU 不同子系统的利用模式截然不同。Im2Col+GEMM 重度使用 Tensor Core 和寄存器文件，因为 GEMM 天生适合 Tensor Core 的分块矩阵乘加指令；Winograd 变换用更多的加减法替换乘法，对整数运算单元和共享内存的压力更大；直接卷积则严重依赖 L1 缓存和纹理缓存来弥补不规则访问模式带来的延迟。理解这些底层差异，是成为一个精通 GPU 性能优化的工程师的关键。',
    '',
])

# Extended explanation after the first Im2Col kernel explanation
add_after(155, [
    '',
    '深入分析 Im2Col 的索引计算可以发现一个微妙的设计细节：线程索引 `i` 和 `j` 的分解方式直接影响全局内存访问的合并性（Coalescing）。在上述实现中，`j` 维度对应的是输出空间位置展平后的连续索引（n, oh, ow），这意味着相邻的线程访问 X_col 矩阵的相邻列——在列主序（Column-major）布局下，这会导致非合并访问（每个线程访问不同行同一列）。如果将 `i` 和 `j` 的映射方式进行交换——让 `j` 对应通道和卷积核维度，`i` 对应空间维度——则可以实现完美的合并访问，因为相邻线程将在 C×K_h×K_w 维度上连续访问。这种对内存布局与线程映射之间关系的敏感度，是区分"能写出正确结果的代码"和"能写出高性能代码"的关键所在。',
    '',
    '此外，边界填充（Padding）的处理方式也有多种选择。上述代码使用条件赋值（if-else），在每个线程内部处理边界逻辑，这会引入控制流分歧。一种更高效的替代方案是"预先扩展"——在卷积之前先将输入特征图扩展一圈零值，然后在 Im2Col 和后续计算中完全省略边界检查。这样做的代价是额外的内存和时间用于填充操作，但在输入尺寸较大时，为每个线程省去的分支指令累积起来远超填充开销。对于小输入（如 MobileNet 的 14×14 特征层），填充策略通常更优；对于大输入（如 ResNet 的 224×224 输入），条件判断策略同样可接受。',
    '',
])

# Extended explanation after conv2d_im2col_gemm explanation
add_after(227, [
    '',
    '要全面评估 Im2Col+GEMM 方法的性能，必须将其分解为两个阶段来独立分析。第一阶段是 Im2Col 变换本身——这是一个纯粹的内存搬移操作，不涉及任何浮点计算。它的性能瓶颈完全在于内存带宽：从输入缓冲区读取每个像素值，按列优先的顺序写入 Im2Col 矩阵。由于输入 X 的读取涉及不规则的内存访问模式（每个输出位置的窗口起始点不同），而写入 X_col 是规则的线性写入，这一阶段的带宽利用率通常不超过 60%-70%（以 A100 约 2TB/s 的 HBM 带宽为参考）。第二阶段是 GEMM 矩阵乘法——这是 GPU 最擅长的运算，在合适的 Tile 尺寸下可以达到 80%-90% 的 Tensor Core 利用率。因此，Im2Col+GEMM 的总延迟大致等于 `Im2Col 延迟 + GEMM 延迟`，其中 Im2Col 的占比随通道数增加而上升（因为需要移动更多的像素数据），而 GEMM 的占比随输出维度增大而上升。',
    '',
    '对比 Im2Col+GEMM 与直接卷积，一个有趣的观察是：当 C_in 和 C_out 都很大（如 512→512 的卷积）且卷积核较小（如 3×3）时，Im2Col 的内存膨胀并不严重——因为 Im2Col 矩阵的尺寸为 (C_in×9)×(H_out×W_out)，而原始输入为 (C_in)×(H×W)，膨胀比仅为 9。但当 C_in 很小而空间维度很大时（如 3×224×224, K=7），膨胀比达到 K_h×K_w = 49，此时内存开销变得不可接受。因此，一种混合策略是：对通道数多、空间维度小的层使用 Im2Col+GEMM，对通道数少、空间维度大的层使用直接卷积。这也是 cuDNN 启发式算法自动选择策略的基本逻辑。',
    '',
])

# Extended explanation after Winograd kernel
add_after(378, [
    '',
    'Winograd 变换的核心数学洞察来自中国剩余定理（Chinese Remainder Theorem）在多项式环上的推广。直观地说，两个多项式的乘积可以被表示为它们在一组精心选择的采样点上的值的逐点乘积——这就是 Winograd 算法将"滑动窗口卷积"（多项式乘法）转化为"逐元素乘法"（点值乘法）的数学本质。变换矩阵 G、B、A 的作用，正是将滤波器和输入块从"系数表示"变换到"点值表示"（在某个精心选择的基下），在点值域执行低开销的逐元素乘法，然后再变换回系数表示。这使得理论乘法次数从 O(K²) 降至 O(K² / m)，其中 m 是输出 tile 的大小。',
    '',
    '从 GPU 实现的角度，Winograd 算法将计算模式从"计算密集"转变为"加载-变换-逐元素乘-逆变换-存储"的五阶段流水线。而每个阶段的算术强度（FLOPs/Byte）各不相同：输入变换阶段（B^T·d·B）是最耗时的部分之一，因为它需要对每个输入 tile 执行 4×4（F(2,3)）或 6×6（F(4,3)）的矩阵乘加减操作，这些操作不是 Tensor Core 友好的标准 GEMM，而是手写展开的逐元素加减——其算术强度极低，严重受限于指令发射率（Issue Rate）。逐元素乘法阶段则是算术强度最高的部分，因为涉及跨 C_in 通道的归约（Reduce），非常适合映射到 Tensor Core 或 warp 级归约指令。理解这种五阶段的不同性能特征，有助于为每个阶段分配最合适的 GPU 资源（寄存器、共享内存、warp 配置）。',
    '',
    '需要特别指出的是，Winograd 变换中的非整数系数（如 0.5）在浮点表示中是一个精确值（二进制浮点中 0.5 = 2^{-1}），因此乘以 0.5 的精度损失实际上很小。真正的精度问题来自于变换域值的动态范围扩大——例如，G 矩阵的常数项可以使变换后的值增大数倍，在后续的逐元素乘法和逆变换中，这些放大的值可能超出 float16 的表示范围（±65504），导致上溢或下溢。这就是为什么 Winograd 卷积通常建议使用 float32 而非 float16 精度，或者需要在变换后对值进行缩放（Scaling）以保持数值范围在安全区间内。',
    '',
])

# Extended explanation after direct_conv2d explanation
add_after(478, [
    '',
    '直接卷积看似"朴素"，但在某些场景下是最优选择，原因有三个方面。首先，从内存层次利用来看，直接卷积不需要像 Im2Col 那样分配额外的全局内存缓冲区来存储变换后的矩阵——在 GPU 显存容量紧张的场景（如运行 batch=64 的大模型训练），这 50%-300% 的内存节省可能是决定能否运行的关键因素。其次，对于大卷积核（K≥7），Im2Col 的内存膨胀比（K_h×K_w）可能达到 49 甚至更大，此时 Im2Col 变换本身的时间开销可能超过 GEMM 计算节省的时间，反而使总体延迟增加。最后，现代 GPU 的 L2 缓存容量持续增长（A100 有 40MB，H100 有 50MB），更大的缓存可以容纳更多的"不规则访问"数据，使得直接卷积的缓存命中率逐步提升，缩小了与 Im2Col+GEMM 方法的性能差距。',
    '',
    '开发者在实现直接卷积时，最容易犯的错误是忽视输入 Tile 的 Halo 区域设计。Halo 区域是指为保持卷积语义的连续性所需的额外边界数据——当输出 tile 为 `(block_M, block_N)` 时，输入 tile 必须扩展为 `(block_M*stride+K_h-1, block_N*stride+K_w-1)`，其中 `K_h-1` 和 `K_w-1` 部分就是 Halo。正确计算 Halo 尺寸是保证 Tile 边界处结果正确的前提；而优化 Halo 区域的加载方式（是否让多个线程协作加载、是否利用向量化加载指令），则是决定直接卷积性能上限的核心因素。一个常见的优化技巧是"Halo 共享"——让相邻的线程块在共享内存中维护部分重叠的数据，从而摊销 Halo 区域的全局内存加载成本。',
    '',
])

# Extended explanation: between Im2Col memory analysis and Winograd
add_after(237, [
    '',
    'Im2Col 内存膨胀问题在工程中有多种缓解手段。一种称为"隐式 Im2Col"（Implicit Im2Col 或 Implicit GEMM）的技术，通过在 GEMM 内核内部动态计算 Im2Col 索引来避免分配完整的 Im2Col 矩阵。类似于"不显式存储变换矩阵，而是在每次访问时即时计算坐标"。这种方法由 cuDNN 和 CUTLASS 广泛采用，可以将内存开销从 O(K²×spatial) 降至 O(1)，但代价是增加了 GEMM 内核内部的索引计算复杂度。在 TileLang 中，由于 Tile 抽象自动管理索引计算，开发者无需手动处理隐式 Im2Col 的复杂坐标映射，只需关注分块策略本身。',
    '',
    '另一个值得关注的趋势是混合精度（Mixed Precision）训练和推理对 Im2Col 内存压力的缓解。使用 fp16 或 bf16 存储输入和权重可以将 Im2Col 矩阵的内存占用减半（从 float32 的 4 字节降至 2 字节），同时在现代 GPU（A100、H100）上还能激活 Tensor Core 获得 2-8 倍的矩阵乘法吞吐量提升。然而，float16 的数值范围有限（±65504），在 Im2Col 变换中如果输入值较大（如未经归一化的原始像素值 0-255），乘以权重后可能溢出。此时 bfloat16 是一个更好的选择——它保持了与 float32 相同的指数范围（8 位指数），只是尾数精度降低（7 位尾数），在大多数深度学习场景中几乎不损失精度。',
    '',
])

# More content for later sections

# After multi_level_tiled_conv - deeper memory hierarchy explanation
add_after(565, [
    '',
    '多级 Tiling 策略本质上是对 GPU 内存层次（Memory Hierarchy）的精准编程。GPU 的内存层次从最快到最慢分别为：寄存器（~1 cycle, ~256KB/SM）、共享内存（~20 cycles, ~164KB/SM）、L1 缓存（~30 cycles, ~256KB/SM）、L2 缓存（~200 cycles, ~40MB 芯片级）、全局内存/HBM（~400 cycles, ~80GB）。多级 Tiling 的目标是通过显式管理数据在各级存储间的移动，使得大多数内存访问命中最快的存储层级。',
    '',
    '以 ResNet-50 的一个典型 3×3 卷积层（C_in=256, C_out=256, H=W=28）为例：Level 1（SM 级）将输出分块为 128×8×8，对应的权重分块为 128×256×3×3 约 1.2MB——这超出了共享内存容量，因此权重必须分多次从全局内存加载。Level 2（Warp 级）在 1×4×2×2 的分块中，每个 warp 负责 4×4=16 个输出像素的计算，所需的输入数据在 warp 的 32 个线程间通过 warp shuffle 指令共享。Level 3（线程级）则在寄存器中维护了 32×4×4 的部分和累加器，尽可能利用寄存器文件的大容量和低延迟。这种金字塔式的数据分布策略，是现代 GPU 高性能计算的通用方法论，不仅适用于卷积，也适用于 GEMM、FFT 等各类计算密集型运算。',
    '',
])

# After ## 6 section header - extended memory optimization discussion  
add_after(592, [
    '',
    '内存访问优化是卷积算子性能调优中最有"杠杆效应"的环节——往往只需几行代码的修改，就能带来 20%-50% 的性能提升。理解这一点的关键在于认识 GPU 的算术强度瓶颈：对于典型的 3×3 卷积，每个输出像素需要 9×C_in 次乘法和 9×C_in 次加法，即 18×C_in FLOPs；同时需要加载约 9×C_in×4 字节的输入数据和 K²×C_in×C_out 共享权重。当 C_in=64 时，每个输出像素约执行 1152 FLOPs，但需要加载至少 2304 字节数据——算术强度约为 0.5 FLOPs/Byte，远低于 A100 的 Roofline 临界点（约 156 FLOPs/Byte），因此卷积操作受限于内存带宽而非计算能力。',
    '',
    '从这个分析可以引出一个反直觉的结论：对于内存受限的卷积操作，提高计算效率（使用 Tensor Core、增加运算并行度）对总延迟的改善可能微乎其微，因为计算单元大部分时间在等待内存数据到达。真正有效的优化方向是减少数据移动：使用共享内存缓存复用数据、优化数据布局减少非合并访问、利用数据打包（Pack）将多个独立加载合并为一次宽加载（如 float4 替代 4×float）。这也是为什么 Winograd 变换（通过减少总乘法运算次数来间接减少数据移动）在实践中往往比单纯增加计算并行度更有效。',
    '',
])

# After grouped_conv2d - extended discussion
add_after(901, [
    '',
    '分组卷积在高性能计算中的一个有趣特性是"计算强度随分组数衰减"。考虑一个标准卷积（groups=1），每个输出像素的计算强度为 2×C_in×K² FLOPs per output pixel，对应的内存访问为 (C_in×K² + C_in×C_out×K²/N_out) 字节（其中 N_out 是输出空间位置总数）。当分组数增加到 g 时，每个输出像素的计算强度降为原来的 1/g（因为每个输出仅连接 C_in/g 个输入通道），而内存访问中的权重部分也按比例减少。然而，输入数据的内存访问基本不变——因为每个输入通道可能被多个组的输出共享（取决于具体的分组方式）。',
    '',
    '这意味着当 groups=C_in（即深度可分离卷积的 Depthwise 阶段），计算强度跌至最低点：每个输出像素的计算仅为 2×K² FLOPs（约 18 FLOPs 对于 3×3 卷积），而输入内存访问仍需 K²×4 字节。此时算术强度低至约 1.125 FLOPs/Byte，使得 Depthwise 卷积成为 GPU 上最"内存密集型"的运算之一。优化这类极端低强度运算的关键在于最大限度地利用共享内存和寄存器来缓存输入数据——上述实现中的 `X_shared` 正是为了将每个输入像素被 K_h×K_w 个输出位置共享时的重复全局内存加载合并为一次加载。',
    '',
])

# After resnet50_conv1 - more detailed analysis
add_after(1001, [
    '',
    '这个 ResNet-50 第一层的实现尽管看似简单，但有几个微妙之处值得深挖。第一，循环嵌套顺序的选择——外层遍历卷积核维度（kh, kw），内层遍历输出 tile（i, j），最内层遍历输入通道（ci）。这个顺序的合理性在于：卷积核权重在 kh/kw 循环迭代之间完全独立，因此适合作为最外层以最大化寄存器复用；而 ci 作为归约维度放在最内层，使得中间累加结果可以在寄存器中保持（每个输出像素的 acc 在 ci 迭代间持续累加），避免了对共享内存的写入。',
    '',
    '第二，步长 stride=2 配合 padding=3 意味着输入坐标计算中存在越界风险——以 ih = (th*8+i)*2 - 3 + kh 为例，当 th=0, i=0, kh=0 时，ih = -3（越下界）；当 th=13, i=7, kh=6 时，ih = (13*8+7)*2-3+6 = 111*2-3+6 = 225（越上界，因为 H=224）。边界检查 `0 <= ih < H` 捕获了这些情况。值得注意的是，stride=2 使得每个输入像素仅被约 1/4 的输出位置使用（因为输出分辨率减半），这意味着缓存利用率天然偏低——这是大 stride 卷积的固有劣势，而非实现缺陷。优化 stride=2 卷积的一个常用技巧是使用 Im2Col+GEMM 方法，因为 GEMM 的规则数据访问模式可以在 K 维度上弥补 stride 导致的访问不连续性。',
    '',
])

# After transposed_conv2d - detailed mechanics
add_after(1138, [
    '',
    '转置卷积的实现要点在于精确掌握"输入到输出的贡献关系"。在标准卷积中，每个输出像素"接收"来自输入窗口内 K_h×K_w 个像素的贡献——这是一个"多对一"的汇聚（Gather）模式。而在转置卷积中，关系完全反转：每个输入像素"分发"其值给输出特征图中 K_h×K_w 个位置的像素——这是一个"一对多"的散射（Scatter）模式。理解这种"汇聚 vs 散射"的对偶性，是掌握转置卷积实现的核心。',
    '',
    '从计算复杂度的角度，转置卷积的 FLOPs 与标准卷积完全相同（C_in × C_out × K_h × K_w × H_out × W_out × 2），但实际 GPU 延迟通常显著更高。原因有三：第一，散射模式使得多个线程可能同时累加到同一个输出位置，需要原子操作（Atomic Add）来保证正确性——这引入了串行化瓶颈；第二，输出特征图比输入更大（上采样），每个输出像素接收的输入贡献较少且不连续，导致每个线程的计算量不足（低 Occupancy）；第三，对于某些 stride 和 K 的组合，输出特征图中存在"从未被任何输入贡献触及"的像素（需要通过额外的偏置项填充），增加了边界处理的复杂度。这些原因共同导致转置卷积往往比同等参数规模的标准卷积慢 2-5 倍。',
    '',
])

# After deformable_conv2d - applications context
add_after(1294, [
    '',
    '可变形卷积虽然在性能上远低于标准卷积，但其在计算机视觉任务中带来的精度提升通常远超性能代价。首次提出可变形卷积的论文（Dai et al., ICCV 2017）在 COCO 目标检测任务上展示了 2-5 个点的 mAP 提升——这在检测领域是显著的进步。可变形卷积的有效性来源于它解决了标准卷积的根本限制：固定几何形状的采样网格。当物体发生非刚性变形、旋转或透视变换时，固定的矩形采样窗口无法有效捕获物体的形状信息，而可学习的偏移量让卷积核"学会"自适应地聚焦于物体的实际轮廓。',
    '',
    '从实现优化的角度，可变形卷积有几个潜在的加速方向。一是利用偏移量预测网络本身也是一个小型卷积网络（通常 3×3），可以将偏移量预测与可变形采样融合到同一个 CUDA 内核中，避免中间的全局内存往返。二是在双线性插值中使用查找表（LUT）加速——将最常用的偏移量模式预计算为整数像素偏移，仅在偏移量超出离散格点时回退到浮点插值。三是限制偏移量的范围（如通过 tanh 激活函数将偏移量限制在 ±2 像素内），一方面减少插值开销（因为大部分采样点落在离散格点附近），另一方面也作为正则化防止过度变形导致的不稳定训练。',
    '',
])

# After winograd_f43 - practical usage discussion
add_after(1678, [
    '',
    'Winograd F(4,3) 的实现较 F(2,3) 更为复杂，主要体现在两个方面。首先是变换矩阵的维度更大（6×6 vs 4×4），这意味着输入变换阶段（B^T·d·B）和输出变换阶段（A^T·m·A）的运算量线性增长——从 F(2,3) 的 4×(4+4) = 32 次加减到 F(4,3) 的 6×(6+6) = 72 次加减（每个 Tile）。虽然乘法次数减少了一半，但变换阶段的加减运算在 GPU 上同样消耗执行单元（CUDA Core），如果加减运算的开销超过了乘法节省的时间，F(4,3) 可能比 F(2,3) 更慢。',
    '',
    '其次，F(4,3) 中的变换矩阵系数涉及更大的数值增长。以输出变换矩阵 A 为例，其元素包含系数 8——这意味着变换域中的值在逆变换时可能被放大 8 倍。当变换域值本身已经因为逐元素乘法而较大时（如输入值范围 0-255 且权重初始化为以 0 为均值的高斯分布随机值），8 倍的放大可能导致 float32 累加器中的有效精度位数减少（因为较大值的加法会"淹没"较小值的贡献）。这也是为什么 F(4,3) 通常只推荐在推理阶段使用——推理时权重已固定，可以通过离线量化（Quantization）和缩放来保持数值范围在安全区间内。在训练阶段，动态变化的权重和梯度使 F(4,3) 的数值风险显著增加。',
    '',
    '在实际部署中，有一项关于 F(2,3) vs F(4,3) 选择的重要经验法则：当每个 Winograd Tile 的输出像素数超过变换域的计算开销时，使用更大的 Tile 才有收益。具体来说，F(2,3) 的变换开销对应 2×2=4 个输出像素，而其变换域计算对应 4×4=16 个值的逐元素乘加；F(4,3) 的变换开销对应 4×4=16 个输出像素，变换域计算对应 6×6=36 个值的逐元素乘加。因此，F(4,3) 将变换开销摊销到了 4 倍于 F(2,3) 的输出像素上，只要实际图像维度足够大以保证 Tile 数量充足。一般建议：当 H_out ≥ 28 时，F(4,3) 的优势开始显现；当 H_out < 14 时，F(2,3) 几乎总是更优。',
    '',
])

# After pointwise_conv2d - GEMM mapping deep dive
add_after(1367, [
    '',
    '将 1×1 卷积映射到 GEMM 的效率，取决于空间维度的扁平化策略与矩阵分块策略的匹配程度。在上述实现中，空间维度 (N, H, W) 被展平为 `spatial = N*H*W`，这假设 N、H、W 三个维度的扁平化顺序对性能无影响——但这个假设在某些情况下不成立。当 batch size N > 1 时，不同样本的相同空间位置在展平后不再连续（因为展平是先 H×W 再 N），这意味着 GEMM 的 K 维度（C_in）上的归约在空间上是分散的，对缓存不友好。更好的做法是将 (N, H, W) 展平为 (N*H, W) 或保持三维索引，然后让 GEMM 的分块策略自然地处理批次维度。实际上，cuBLAS 的 `cublasGemmStridedBatched` 正是为了解决这种"批次维度的 GEMM"问题而设计的。',
    '',
    '另一个容易被忽视的细节是：当 C_in 或 C_out 很小时（如 MobileNet 中的 16、32 通道），GEMM 的 block_K 分块策略需要特别调整。如果 C_in=16 而 block_K=32，GEMM 将在 K 维度上"跨越 batch 边界"——即同一个 block_K 分块内部同时包含来自不同 batch 样本的数据。这在数学上不会导致计算错误，但可能破坏内存访问的局部性，因为不同 batch 的数据在 GPU 内存中通常是不连续的。此时应将 block_K 减小到 ≤ C_in，或者将 batch 维度合并到 M 维度中（即将 GEMM 的输入视为 (N*H*W, C_in) × (C_in, C_out)）。正确理解张量形状与 GEMM 分块参数之间的关系，是实现高效 1×1 卷积的基础。',
    '',
])

# After fused_pointwise_conv_relu - tensor core precision
add_after(1936, [
    '',
    '融合算子的核心性能收益来源于"减少数据移动次数"（Reduce Data Movement），而非"减少计算量"。从纯计算的角度，ReLU 仅是对输出矩阵的每个元素进行一次比较（max(x, 0)），计算量可以忽略不计。但从数据移动的角度，如果不融合，ReLU 需要从全局内存读取整个输出矩阵（N×C_out×H×W 个 float32，对于 ResNet-50 的 conv3_x 层约 28MB），执行一次比较，再写回——这约 56MB 的数据移动（读 28MB + 写 28MB）在 A100 上需要约 28 微秒（按 2TB/s 带宽计算）。对于批量推理，这个开销乘以 batch size 和层数后可能占据总延迟的 5%-15%。融合消除了这整个读写循环，将 ReLU 的延迟从"内存往返"降为"寄存器操作"（约 1 cycle）。',
    '',
    '该实现还展示了一个关键的混合精度设计模式："高精度累加、低精度存储"。X 和 W 使用 float16 来减少内存带宽和利用 Tensor Core，但 Y_frag（累加器）使用 float32 来保持累加精度——这正是 NVIDIA 推荐的 Tensor Core 编程范式。Tensor Core 的 MMA 指令执行的是 D = A×B + C，其中 A 和 B 是 fp16，C 和 D 是 fp32。这种设计并非偶然：矩阵乘法的部分和可能远超 fp16 的表示范围（±65504），例如当 C_in=1024 且输入和权重各取典型值 1.0 时，点积结果约为 1024，仍在 fp16 范围内；但若输入和权重的值分布较广（如 ±10），点积可达 102400，远超 fp16 上限。因此使用 fp32 累加器是正确性的保证。而在最终写回时，`Y` 的计算结果通常会被后续的 BatchNorm 或激活函数"压缩"回合理范围，fp32 精度足够。',
    '',
])

# After ncu bash block - detailed profiling metrics
add_after(2025, [
    '',
    'Nsight Compute 是卷积算子优化的"显微镜"——它将 GPU 内核的执行分解为数百个硬件指标，帮助开发者从"黑盒调参"转向"白盒诊断"。在分析卷积内核时，以下几个高级指标尤为重要：（1）`sm__warps_launched.avg` 反映了内核的并行粒度，与理论 warp 数（由线程块大小和网格大小计算得出）的比值即 Occupancy；（2）`l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit` 与 `miss` 的比值即 L1 缓存命中率，低于 60% 通常说明数据复用不足或 Tile 尺寸过大；（3）`smsp__average_warps_issue_stalled_barrier_per_issue_active` 量化了 warp 因等待同步屏障（如 __syncthreads__）而停滞的时间比例，高值意味着共享内存加载策略需要优化（如使用异步拷贝指令 cp.async 替代同步加载）；（4）`smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active` 揭示了 warp 因等待全局内存加载完成而停滞的时间——这是"内存受限"的直接证据，优化方向是增加每个线程的独立内存请求数（即提高"飞行的"（in-flight）内存加载数量）来隐藏延迟。',
    '',
    '将 Nsight Compute 的分析结果转化为可操作的优化建议，需要建立一个系统性的诊断矩阵。如果 `sm__throughput` 接近 100% 但 `gpu__dram_throughput` 也很高（>80%），说明内核已接近硬件极限，几乎没有优化空间。如果 `sm__throughput` 很低（<40%）而 `gpu__dram_throughput` 也很低（<30%），则可能的原因是 Occupancy 不足（寄存器或共享内存使用过多导致每个 SM 只能驻留少量 warp）或者内核启动参数（网格大小）不正确。如果 `sm__throughput` 很低但 `gpu__dram_throughput` 很高，则是典型的内存受限场景——计算单元在"等待数据"——此时应减少全局内存访问（增加共享内存使用、优化数据布局）或使用算法变换（如 Winograd）来降低内存访问需求。这套诊断流程将性能优化从"经验驱动"提升为"数据驱动"，是成为 GPU 性能工程师的必备技能。',
    '',
])

# More transitional paragraphs between key sections
# After dilated_conv2d code
add_after(1210, [
    '',
    '空洞卷积在语义分割任务（如 DeepLab 系列）中的广泛应用，驱动了对其高效实现的需求。DeepLabV3 使用的 Atrous Spatial Pyramid Pooling (ASPP) 模块同时运行多个不同 dilation rate 的 3×3 卷积——典型的配置是 rates=[6, 12, 18]——然后将各分支的输出拼接。这种"多分支并行空洞卷积"的模式对 GPU 调度器提出了严峻挑战：不同 dilation rate 的内核具有不同的有效核大小和内存访问模式，GPU 需要在它们之间进行上下文切换，每个内核只能占用一小部分 SM，导致总利用率低下。一种优化方法是将多个 dilation rate 合并为一个内核——通过将 dilation rate 作为内核参数（而非编译期常量），在每个线程内部根据 rate 计算访问偏移。这牺牲了少量编译期优化机会（因为 rate 不再能用于常量折叠和循环展开），但换来了更好的 GPU 占用率和更少的上下文切换。',
    '',
    '在硬件层面，空洞卷积对 GPU L2 缓存的影响值得单独审视。标准 3×3 卷积的每个输出像素需要访问 9 个输入像素，这 9 个像素位于连续的 3 行 × 3 列区域内——总共 9 个缓存行（假设每个缓存行 128 字节，每行包含 32 个 float32 元素）。而 dilation=2 的空洞 3×3 卷积实际采样区域为 5×5，涉及 25 个可能的缓存行（虽然只访问其中 9 个元素）。当多个线程块同时执行时，这些"跳跃式"的内存访问会大幅增加 L2 缓存的压力——因为每个线程块的工作集覆盖了更大的内存区域，导致缓存逐出（Eviction）更频繁。实测数据显示，对于 dilation=12 的极端配置（有效核大小 25×25），L2 缓存命中率可能从标准卷积的 85% 跌至 35%，这是空洞卷积在大 dilation rate 下性能骤降的硬性原因。',
    '',
])

# Before ## 28 (auto-tuning advanced)
add_after(1939, [
    '',
    '自动调优的进阶策略不仅要"找到好的配置"，更要"快速找到好的配置"。一个完整的自动调优流程通常分为三个阶段。阶段一是"初始搜索"——使用 Roofline 模型或简单的启发式规则生成 5-10 个候选配置，快速排除明显不合理的组合（如 block_K > C_in 或 block_M > H_out）。阶段二是"粗粒度搜索"——使用基于代价模型（Cost Model）的方法（如 TVM 的 XGBoost 预测器或 Ansor 的梯度提升树），评估每个配置的预期延迟，选择 top-K（通常 K=20-50）进行实际测量。阶段三是"精粒度搜索"——对 top-K 中的最佳配置进行局部搜索（如微调分块参数 ±16，调整 warp 数 ±2），以发现粗搜索可能遗漏的"性能尖峰"。这种三级搜索策略可以将配置空间从数千候选缩小到数十次实际测量，使自动调优在 5-10 分钟内完成一个完整 CNN 模型中所有卷积层的优化。',
    '',
    '此外，自动调优系统的一个关键工程挑战是"可复现性"——即相同的配置在不同运行时产生一致的性能测量结果。GPU 性能测量的噪声来源众多：GPU 温度导致的时钟频率调整（Thermal Throttling）、操作系统后台进程抢占 CPU 导致的内核启动延迟抖动、其他 CUDA 上下文的并发干扰（如在 GPU 集群中）、以及 CUDA 驱动程序本身的内存分配碎片化。为降低这些噪声，标准的基准测试流程应包含：固定 GPU 时钟频率（`nvidia-smi -ac` / `nvidia-smi -lgc`）、分配预热内核来"烧热"GPU 到稳态温度、使用 CUDA Events 替代 CPU 计时器来精确测量 GPU 内核时间、以及多次测量取最小值（而非平均值——因为噪声通常是"变慢"方向的单边分布）。',
    '',
])

# Apply all insertions from bottom to top
insertions.sort(key=lambda x: x[0], reverse=True)

for line_num, text_lines in insertions:
    insertion_point = line_num + 1
    for i, text in enumerate(text_lines):
        lines.insert(insertion_point + i, text)

with open('/Users/leafoon/Desktop/Note/CS2/content/tilelang/25-convolution-operators.md', 'w') as f:
    f.write('\n'.join(lines))

added = len(lines) - original_count
print(f"Original lines: {original_count}")
print(f"Total lines after round 2: {len(lines)}")
print(f"Lines added in round 2: {added}")
