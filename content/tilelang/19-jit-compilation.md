---
title: "Chapter 19: JIT 动态编译与运行时机制"
description: "深入理解 TileLang 的 JIT 编译机制：运行时代码生成、编译缓存策略、Kernel Launch 配置自动推导、与 PyTorch 的无缝集成"
updated: "2025-06-11"
---

# Chapter 19: JIT 动态编译与运行时机制

> **Learning Objectives**
>
> 1. 理解 JIT（Just-In-Time）编译的核心原理及其与 AOT 编译的本质区别
> 2. 掌握 TileLang 从 Python DSL 到机器码的完整 JIT 编译流水线
> 3. 深入理解编译缓存机制，包括文件缓存、内存缓存与缓存失效策略
> 4. 学会 Kernel Launch 配置的自动推导算法及其手动覆盖方法
> 5. 掌握 TileLang 与 PyTorch 的无缝集成方式，包括 `torch.compile` 后端和自定义算子
> 6. 理解首次编译与缓存命中的性能差异及优化策略
> 7. 熟悉编译选项、环境变量与多目标编译配置
> 8. 掌握动态 Shape 支持的 Symbolic 机制及其性能影响
> 9. 能够独立排查 JIT 编译过程中的常见问题

---

## 1. JIT 编译原理

### 1.1 什么是 JIT 编译

JIT（Just-In-Time）编译是一种在程序**运行时**将高级语言或中间表示编译为目标机器码的技术。与传统的 AOT（Ahead-Of-Time）编译不同，JIT 编译器在程序实际执行时才触发编译过程，这使得编译器可以利用运行时信息进行更具针对性的优化。

在深度学习和高性能计算领域，JIT 编译具有独特的价值：

```
┌─────────────────────────────────────────────────────────────────┐
│                    传统 AOT 编译流程                              │
│                                                                  │
│  源代码 → 预编译 → 编译 → 链接 → 可执行文件 → 运行               │
│                                                                  │
│  特点：编译一次，处处运行（但无法利用运行时信息）                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    JIT 编译流程                                   │
│                                                                  │
│  源代码 → 运行 → [首次调用时编译] → 执行                         │
│                       ↓                                          │
│               缓存编译结果 → [后续调用直接命中]                   │
│                                                                  │
│  特点：运行时编译，可利用实际输入信息进行优化                     │
└─────────────────────────────────────────────────────────────────┘
```

上面的流程图清晰地展示了 AOT 编译和 JIT 编译在工作流程上的根本差异。AOT 编译在程序运行之前就完成了所有编译工作，生成的可执行文件可以直接在目标平台上运行，但无法利用运行时的特定信息（如实际的 tensor shape、GPU 型号等）。而 JIT 编译则是在程序首次运行时才触发编译过程，编译器可以获取到实际的运行时信息，从而生成更加特化和高效的代码。这种"延迟编译"的策略虽然引入了首次运行的编译开销，但通过缓存机制可以将后续调用的开销降至接近零。在深度学习场景中，由于同一个 kernel 往往会被反复调用成千上万次，JIT 编译的这种"一次编译，反复使用"的模式具有极高的性价比。

### 1.2 AOT vs JIT 对比

| 维度 | AOT（提前编译） | JIT（即时编译） |
|------|-----------------|-----------------|
| **编译时机** | 部署前完成全部编译 | 首次调用时触发编译 |
| **优化程度** | 通用优化，无法利用运行时信息 | 可根据实际输入 shape、设备信息进行特化优化 |
| **部署复杂度** | 需要为每个目标平台分别编译 | 单一代码库，运行时自动适配 |
| **调试便利性** | 编译错误在部署前即可发现 | 编译错误在运行时暴露 |
| **首次运行延迟** | 无额外延迟 | 首次调用有编译开销（通常数百毫秒到数秒） |
| **后续运行性能** | 稳定一致 | 缓存命中后与 AOT 性能相当 |
| **内存占用** | 编译产物在磁盘，运行时加载 | 缓存产物可能占用磁盘和内存 |
| **适用场景** | 生产部署、嵌入式系统 | 研究探索、动态 shape、快速原型 |
| **优化机会** | 编译时所有信息已知 | 运行时可获取实际 tensor shape、dtype、设备信息 |
| **代码分发** | 分发编译后的二进制 | 分发源码，用户侧编译 |

> [!TIP]
> 在深度学习框架中，JIT 和 AOT 并不是互斥的选择。现代框架通常采用混合策略：常用算子使用 AOT 预编译，而用户自定义或动态场景使用 JIT 编译。TileLang 正是采用了这种混合策略。

上面的对比表格从多个维度系统地分析了 AOT 和 JIT 两种编译方式的差异。在实际应用中，选择哪种方式取决于具体的使用场景：对于需要稳定延迟的生产环境，AOT 编译是更可靠的选择；而对于需要快速迭代和动态适配的研究场景，JIT 编译提供了更大的灵活性。值得注意的是，JIT 编译的"运行时信息利用"优势在现代深度学习中尤为重要——同一个 Transformer 模型在处理不同长度的序列时，如果能够根据实际的序列长度动态生成优化的 kernel，将会带来显著的性能提升。

### 1.3 运行时代码生成的动机

JIT 编译在 TileLang 中的核心动机包括：

**动态 Shape 支持**

```python
import tilelang
import tilelang.language as T

@tilelang.jit
def dynamic_matmul(M, N, K, dtype="float16"):
    """支持任意 M, N, K 维度的矩阵乘法"""
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # JIT 编译器会根据实际 M, N, K 值生成特化代码
        with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), dtype)
            B_shared = T.alloc_shared((32, 128), dtype)
            C_local = T.alloc_fragment((128, 128), dtype)

            T.clear(C_local)
            for k in T.serial(T.ceildiv(K, 32)):
                T.copy(A[by * 128, k * 32], A_shared)
                T.copy(B[k * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    return main

# 不同 shape 的调用触发不同的 JIT 编译
matmul_1024 = dynamic_matmul(1024, 1024, 1024)   # 编译一次
matmul_2048 = dynamic_matmul(2048, 2048, 2048)   # 编译一次（不同配置）
matmul_512  = dynamic_matmul(512, 512, 512)       # 编译一次（不同配置）
```

这段代码展示了 TileLang JIT 编译器的核心优势之一：动态 Shape 支持。函数 `dynamic_matmul` 接受 M、N、K 三个符号变量作为参数，当使用不同的具体值调用时，JIT 编译器会为每种 shape 组合生成特化的机器码。例如，当调用 `dynamic_matmul(1024, 1024, 1024)` 时，编译器会生成一个专门优化 1024×1024 矩阵乘法的 kernel，其中所有的循环边界、内存偏移量都是常量，编译器可以进行更激进的优化（如循环展开、常量折叠等）。这种设计使得用户可以编写一次通用的 kernel 代码，而无需为每种可能的输入 shape 手动编写不同的实现。在实际应用中，不同的推理请求可能需要处理不同大小的输入，JIT 编译确保了每种输入规模都能获得最优的执行性能，而缓存机制保证了首次编译后的后续调用几乎没有额外开销。

**硬件自适应**

```python
@tilelang.jit(
    # 根据运行时硬件自动选择最优配置
    out_idx=-1,
    target="auto",  # 自动检测 CUDA/HIP/Ascend
)
def adaptive_kernel(A, B):
    M, K = A.shape
    K, N = B.shape
    # TileLang 会根据目标硬件自动调整:
    # - Thread block 大小
    # - Shared memory 分配
    # - 指令选择（Tensor Core / Matrix Core / AICore）
    ...
```

这个代码片段展示了 TileLang JIT 编译器的另一个重要特性：硬件自适应能力。通过设置 `target="auto"`，编译器会在运行时自动检测当前可用的硬件平台（NVIDIA GPU、AMD GPU 或华为昇腾 NPU），并根据目标硬件的特性选择最优的编译策略。对于 NVIDIA GPU，编译器可能会选择使用 Tensor Core 指令来加速矩阵运算；对于 AMD GPU，则会使用对应的 Matrix Core；对于华为昇腾，则使用 AICore 指令集。这种硬件自适应能力使得同一份 TileLang 代码可以在不同的硬件平台上运行，而无需修改任何代码，极大地简化了跨平台部署的复杂度。`out_idx=-1` 参数指定了输出 tensor 在函数参数列表中的位置，帮助编译器正确识别输入和输出。

**快速迭代开发**

```python
# 开发者只需关注算法逻辑，无需手动管理编译
@tilelang.jit
def my_custom_op(A, B, C):
    # 修改算法后，重新运行即可自动重新编译
    # 无需手动执行 nvcc/gcc 命令
    ...

# 修改 → 运行 → 自动编译 → 测试，形成快速反馈循环
```

这个简短的代码片段强调了 JIT 编译对开发效率的提升。传统的 GPU kernel 开发需要经历"编写代码 → 手动编译 → 运行测试 → 发现问题 → 修改代码"的繁琐循环，其中手动编译步骤可能需要几十秒甚至几分钟。而使用 TileLang 的 JIT 编译，开发者只需要修改 kernel 代码并重新运行，编译器会自动检测代码变化并重新编译，整个过程对开发者完全透明。更重要的是，编译器会自动处理缓存逻辑——如果代码没有变化，直接使用缓存的编译结果；如果代码发生了变化，则自动触发重新编译。这种设计让开发者可以专注于算法逻辑的实现，而无需关心底层的编译细节，显著提升了迭代开发的速度。

### 1.4 从装饰器到编译：完整流程

当使用 `@tilelang.jit` 装饰器时，完整的编译流程如下：

上面的流程图详细描述了 `@tilelang.jit` 装饰器从注册到执行的完整生命周期。这个流程可以分为五个关键阶段：装饰器注册阶段将用户的 Python 函数包装为一个可调用的 JIT 编译对象；首次调用阶段提取输入 tensor 的元信息（shape、dtype、device）并生成缓存 key；缓存检查阶段判断是否需要触发完整的编译流程；编译阶段执行从 Python DSL 到机器码的多阶段转换；缓存阶段将编译结果同时保存到内存和磁盘；最后执行阶段配置 Launch 参数并启动 GPU kernel。理解这个完整流程对于排查 JIT 编译相关的问题至关重要——例如，如果发现每次运行都很慢，可能是缓存 key 的生成逻辑有问题导致缓存无法命中；如果首次编译时间过长，可能是 kernel 代码过于复杂或者编译选项配置不当。

```
┌──────────────────────────────────────────────────────────────────────┐
│  @tilelang.jit 编译流程                                               │
│                                                                       │
│  1. 装饰器注册                                                        │
│     ┌─────────────────────────────────────────────┐                   │
│     │ @tilelang.jit                                │                   │
│     │ def kernel_func(A, B):                       │                   │
│     │     ...                                      │                   │
│     │                                              │                   │
│     │ 等价于:                                       │                   │
│     │ kernel_func = tilelang.jit(kernel_func)      │                   │
│     └─────────────────────────────────────────────┘                   │
│                          ↓                                            │
│  2. 首次调用触发                                                       │
│     ┌─────────────────────────────────────────────┐                   │
│     │ result = kernel_func(A, B)                   │                   │
│     │                                              │                   │
│     │ a) 提取输入 tensor 的 shape, dtype, device    │                   │
│     │ b) 生成缓存 key (shape + dtype + target)      │                   │
│     │ c) 检查缓存是否命中                           │                   │
│     └─────────────────────────────────────────────┘                   │
│                          ↓                                            │
│  3. 缓存未命中 → 触发编译                                              │
│     ┌─────────────────────────────────────────────┐                   │
│     │ a) 执行用户函数，构建 PrimFunc IR             │                   │
│     │ b) TileLang DSL → Tile IR 转换               │                   │
│     │ c) Tile IR → TensorIR 转换                    │                   │
│     │ d) 应用优化 Pass                              │                   │
│     │ e) 代码生成（CUDA/LLVM/PTX）                  │                   │
│     │ f) 编译为可执行 Kernel                        │                   │
│     └─────────────────────────────────────────────┘                   │
│                          ↓                                            │
│  4. 缓存结果                                                          │
│     ┌─────────────────────────────────────────────┐                   │
│     │ a) 内存缓存：进程内直接可用                   │                   │
│     │ b) 文件缓存：写入 ~/.cache/tilelang/          │                   │
│     └─────────────────────────────────────────────┘                   │
│                          ↓                                            │
│  5. 执行 Kernel                                                       │
│     ┌─────────────────────────────────────────────┐                   │
│     │ a) 配置 Launch 参数 (grid, block, smem)       │                   │
│     │ b) 拷贝输入数据到 GPU                         │                   │
│     │ c) Launch Kernel                              │                   │
│     │ d) 返回结果                                   │                   │
│     └─────────────────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.5 JIT 编译的核心优势

```python
# 优势 1：Shape 特化优化
@tilelang.jit
def softmax(x):
    # 当输入 shape 为 (4096, 4096) 时
    # JIT 可以精确计算每个 warp 处理的行数
    # 而不是使用通用的循环处理
    ...

# 优势 2：常量折叠
@tilelang.jit
def fused_op(A, B, scale=1.0):
    # scale 作为常量被编译期折叠
    # 不会在运行时产生额外的乘法指令
    ...

# 优势 3：消除 Python 开销
@tilelang.jit
def complex_algorithm(A, B, C):
    # Python 的解释执行开销被完全消除
    # 所有计算都在编译后的原生代码中执行
    ...
```

这段代码通过三个具体示例展示了 JIT 编译的核心优势。第一个优势"Shape 特化优化"意味着编译器可以根据实际的 tensor shape 生成高度优化的代码——例如，当输入 shape 为 (4096, 4096) 时，编译器可以精确计算出每个 warp 处理多少行，从而避免运行时的动态分支判断。第二个优势"常量折叠"是指编译器会在编译期将已知的常量参数直接计算出来，例如 `scale=1.0` 这样的参数会被编译器直接优化掉，不会在运行时产生额外的乘法指令。第三个优势"消除 Python 开销"是指 JIT 编译将用户的 Python 代码转换为原生机器码后，Python 解释器的开销（如动态类型检查、引用计数等）被完全消除，所有计算都在高效的原生代码中执行。这三个优势共同作用，使得 JIT 编译的 kernel 在缓存命中后能够达到接近手写 CUDA 代码的性能水平。

> [!WARNING]
> JIT 编译的首次调用会有明显的编译延迟。在性能敏感的场景中，建议在应用启动时进行预热（warm-up），提前触发所有可能的编译路径。

这个警告提醒开发者注意 JIT 编译的一个重要特性：首次编译延迟。在生产环境中，如果首次用户请求触发了 JIT 编译，可能会导致该请求的延迟显著增加（通常在数百毫秒到数秒之间）。为了避免这种情况，推荐的做法是在应用启动时进行"预热"——使用一些典型的输入 shape 提前触发所有可能的 kernel 编译。这样，当真正的用户请求到达时，所有需要的 kernel 都已经被编译并缓存，可以直接使用缓存的编译结果，避免了首次编译的延迟。预热策略的设计需要考虑实际使用场景中可能出现的所有输入 shape 和 dtype 组合，以确保覆盖所有可能的编译路径。

---

## 2. TileLang JIT 编译流程

### 2.1 编译流水线概览

TileLang 的 JIT 编译流水线包含多个阶段，每个阶段将程序从一种中间表示（IR）转换到更低级的表示：

上面的流程图展示了 TileLang 编译流水线的四个主要阶段。Python DSL 阶段负责解析用户的 Python 代码，进行类型推导和模板展开；Tile IR 阶段保留了 TileLang 特有的语义信息（如 Kernel Launch、内存层次等），便于后续的布局推导和访存优化；TensorIR 阶段是 TVM 的原生 IR 表示，在这个阶段会应用各种通用优化 Pass（如算子融合、循环变换、向量化等）；最后的 LLVM/CUDA 阶段负责生成目标平台的机器码。这种多级 IR 的设计使得每一层都可以专注于自己擅长的优化，Python DSL 层关注用户友好的 API 设计，Tile IR 层关注 Tile 操作的语义保持，TensorIR 层关注通用的编译优化，代码生成层关注目标硬件的指令选择和寄存器分配。这种分层设计不仅使得编译器的实现更加模块化，也使得每个阶段的优化可以独立进行，方便后续的扩展和维护。

### 2.2 阶段一：Python DSL → Tile IR

用户编写的 Python DSL 代码首先被解析为 TileLang 特有的 Tile IR。这个阶段的核心任务是将 Python 语法转换为结构化的中间表示。

**输入：用户 Python 代码**

上面的代码展示了典型的 TileLang kernel 定义方式。`@T.prim_func` 装饰器标记这是一个原语函数，函数签名中使用 `T.Tensor` 类型注解来声明输入输出 tensor 的 shape 和 dtype。`T.Kernel` 上下文管理器定义了 GPU kernel 的启动配置——`T.ceildiv(N, 128)` 和 `T.ceildiv(M, 128)` 分别是 x 和 y 维度的 grid 大小，`threads=128` 指定了每个 block 的线程数。在 kernel 内部，`T.alloc_shared` 和 `T.alloc_fragment` 分别分配 shared memory 和寄存器（fragment），`T.clear` 将累加器初始化为零，`T.serial` 定义串行循环，`T.copy` 执行数据搬运，`T.gemm` 执行矩阵乘法。这种高层的 DSL 设计使得开发者可以用接近数学公式的简洁方式描述 kernel 逻辑，而将底层的循环展开、内存布局、同步等复杂细节交给编译器自动处理。

**输出：Tile IR 表示**

Tile IR 表示是 Python DSL 经过解析后的结构化中间表示。与 Python 代码相比，Tile IR 更加形式化，明确记录了每个操作的语义信息。`KernelLaunch` 节点记录了 grid 和 block 的维度信息，`AllocShared` 和 `AllocFragment` 节点明确区分了不同层次的内存分配（shared memory vs 寄存器），`CopyTile` 和 `Gemm` 节点保留了高层的 Tile 操作语义。这种结构化的表示使得后续的编译阶段可以方便地分析和变换代码，例如布局推导 Pass 可以根据 `AllocShared` 节点的信息来决定 shared memory 的具体布局策略。

### 2.3 Tile IR 的关键特性

Tile IR 保留了 TileLang DSL 的高层语义，特别是：

```python
# 1. Kernel Launch 语义
#    明确记录 grid/block 维度和线程数
KernelLaunch(grid_dim=(gx, gy, gz), block_dim=(bx, by, bz))

# 2. 内存层次语义
#    区分 shared memory、fragment (register)、global memory
AllocShared(shape, dtype)    # SMEM 分配
AllocFragment(shape, dtype)  # 寄存器分配
AllocGlobal(shape, dtype)    # 显式全局内存分配

# 3. 算子原语
#    保留高层 Tile 操作，便于后续布局推导
Gemm(A_shared, B_shared, C_local, ...)
CopyTile(src, dst, ...)
Softmax(input, output, ...)
```

Tile IR 的三个关键特性使其成为连接用户 DSL 和底层优化的重要桥梁。Kernel Launch 语义保留了用户指定的 grid/block 配置信息，使得编译器可以在后续阶段根据硬件约束进行调整。内存层次语义是 TileLang 的核心设计之一——它明确区分了三种不同层次的内存（global memory、shared memory、register），使得编译器可以针对不同层次的内存采用不同的优化策略。例如，对于 shared memory 访问，编译器会考虑 bank conflict 的避免；对于 register 访问，编译器会考虑寄存器压力和分配策略。算子原语（如 Gemm、CopyTile、Softmax）保留了高层的 Tile 操作语义，使得布局推导 Pass 可以根据具体的算子类型选择最优的内存布局。例如，对于 Gemm 操作，编译器可能会为 A 矩阵选择列主序布局，为 B 矩阵选择行主序布局，以优化 shared memory 的访问模式。
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Python DSL  │───▶│  Tile IR    │───▶│  TensorIR   │───▶│ LLVM/CUDA   │
│  (用户代码)  │    │ (TileLang   │    │ (TVM 原生   │    │ (目标代码)  │
│              │    │  特有 IR)   │    │  IR)        │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
  DSL 解析          Tile 语义         通用优化            代码生成
  类型推导          布局推导          算子融合            寄存器分配
  模板展开          访存优化          循环变换            指令选择
```

### 2.2 阶段一：Python DSL → Tile IR

用户编写的 Python DSL 代码首先被解析为 TileLang 特有的 Tile IR。这个阶段的核心任务是将 Python 语法转换为结构化的中间表示。

**输入：用户 Python 代码**

```python
@T.prim_func
def matmul_kernel(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
        A_shared = T.alloc_shared((128, 32), "float16")
        B_shared = T.alloc_shared((32, 128), "float16")
        C_local = T.alloc_fragment((128, 128), "float32")

        T.clear(C_local)
        for k in T.serial(T.ceildiv(K, 32)):
            T.copy(A[by * 128, k * 32], A_shared)
            T.copy(B[k * 32, bx * 128], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * 128, bx * 128])
```

**输出：Tile IR 表示**

```
TileIR Module:
  Function: matmul_kernel
    Parameters: [A: float16[M, K], B: float16[K, N], C: float16[M, N]]
    
    KernelLaunch:
      grid_dim: (ceildiv(N, 128), ceildiv(M, 128), 1)
      block_dim: (128, 1, 1)
      
      Body:
        AllocShared: A_shared: float16[128, 32]  # bank-aware allocation
        AllocShared: B_shared: float16[32, 128]
        AllocFragment: C_local: float32[128, 128]  # register allocation
        
        Clear(C_local)
        
        For(k, 0, ceildiv(K, 32)):
          CopyTile(A[by*128 : by*128+128, k*32 : k*32+32] → A_shared)
          CopyTile(B[k*32 : k*32+32, bx*128 : bx*128+128] → B_shared)
          Gemm(A_shared, B_shared, C_local)
        
        CopyTile(C_local → C[by*128 : by*128+128, bx*128 : bx*128+128])
```

### 2.3 Tile IR 的关键特性

Tile IR 保留了 TileLang DSL 的高层语义，特别是：

```python
# 1. Kernel Launch 语义
#    明确记录 grid/block 维度和线程数
KernelLaunch(grid_dim=(gx, gy, gz), block_dim=(bx, by, bz))

# 2. 内存层次语义
#    区分 shared memory、fragment (register)、global memory
AllocShared(shape, dtype)    # SMEM 分配
AllocFragment(shape, dtype)  # 寄存器分配
AllocGlobal(shape, dtype)    # 显式全局内存分配

# 3. 算子原语
#    保留高层 Tile 操作，便于后续布局推导
Gemm(A_shared, B_shared, C_local, ...)
CopyTile(src, dst, ...)
Softmax(input, output, ...)
```

### 2.4 阶段二：Tile IR → TensorIR

Tile IR 被进一步转换为 TVM 的原生 TensorIR 表示。这个阶段的核心任务是：

1. **布局推导（Layout Inference）**：为每个 tensor 确定具体的内存布局
2. **Tile 操作展开**：将高层的 `Gemm`、`CopyTile` 操作展开为底层循环
3. **内存规划**：确定 shared memory 的具体地址映射

**TensorIR 表示示例**

```python
# TensorIR (TVM 原生格式)
@T.prim_func
def matmul_kernel(
    A: T.Buffer((M, K), "float16"),
    B: T.Buffer((K, N), "float16"),
    C: T.Buffer((M, N), "float16"),
):
    # blockIdx.x, blockIdx.y, threadIdx.x 由 T.Kernel 展开而来
    bx = T.int32()
    by = T.int32()
    tx = T.int32()
    
    # Shared memory 作为 T.alloc_buffer
    A_shared = T.alloc_buffer((128, 32), "float16", scope="shared")
    B_shared = T.alloc_buffer((32, 128), "float16", scope="shared")
    C_local = T.alloc_buffer((128, 128), "float32", scope="local")
    
    # 循环结构完全展开
    for i in T.serial(128):
        for j in T.serial(128):
            C_local[i, j] = T.float32(0)
    
    for k in T.serial(T.ceildiv(K, 32)):
        # CopyTile 展开为逐元素的加载循环
        for i in T.serial(128):
            for j in T.serial(32):
                A_shared[i, j] = A[by * 128 + i, k * 32 + j]
        for i in T.serial(32):
            for j in T.serial(128):
                B_shared[i, j] = B[k * 32 + i, bx * 128 + j]
        
        # Gemm 展开为矩阵乘法循环
        for i in T.serial(128):
            for j in T.serial(128):
                for kk in T.serial(32):
                    C_local[i, j] += T.cast(A_shared[i, kk], "float32") * T.cast(B_shared[kk, j], "float32")
    
    # 结果写回
    for i in T.serial(128):
        for j in T.serial(128):
            C[by * 128 + i, bx * 128 + j] = T.cast(C_local[i, j], "float16")
```

> [!TIP]
> Tile IR 到 TensorIR 的转换是 TileLang 编译流水线中最关键的步骤之一。这个阶段决定了最终代码的内存访问模式和计算效率。理解这个转换过程有助于编写高效的 TileLang kernel。

### 2.5 阶段三：TensorIR → LLVM/CUDA/PTX

TensorIR 经过一系列优化 Pass 后，被转换为目标平台的代码：

这个流程图展示了 TensorIR 到目标代码的完整转换过程。在代码生成之前，TensorIR 会经过多个优化 Pass 的处理。算子融合 Pass 会将相邻的逐元素操作（如加法、乘法、ReLU 等）合并为一个 kernel，减少 kernel launch 的开销和全局内存的访问次数。循环变换 Pass 包括循环展开（减少循环控制开销）、循环重排（改善数据局部性）和循环分裂（提高并行度）。向量化 Pass 将连续的标量内存访问转换为向量加载/存储指令（如 CUDA 的 `float4` 加载），显著提高内存带宽利用率。强度削减 Pass 将某些昂贵的操作（如除法）替换为更廉价的等价操作（如乘以倒数或移位）。这些优化 Pass 共同作用，将高层的 TensorIR 转换为高效的低级代码，为最终的代码生成做好准备。

**生成的 CUDA 代码示例**

上面的 CUDA 代码是 TensorIR 经过优化后自动生成的目标代码。这段代码展示了几个关键的 GPU 编程模式：`__launch_bounds__(128)` 告诉编译器每个 block 最多使用 128 个线程，帮助编译器更好地进行寄存器分配；`__shared__` 关键字声明了 shared memory 数组，用于 block 内线程间的数据共享；`__syncthreads()` 是 block 内的同步屏障，确保所有线程在进入下一轮计算前都完成了数据加载。代码中的协作加载模式（cooperative loading）是一个重要的优化技巧——128 个线程共同协作将数据从全局内存加载到 shared memory，每个线程负责加载多个元素，从而充分利用内存带宽。这种协作加载模式是 TileLang 编译器自动从高层的 `T.copy` 操作生成的，开发者无需手动实现。

```cuda
// 由 TensorIR 自动生成的 CUDA 代码
extern "C" __global__ void __launch_bounds__(128)
matmul_kernel(
    half* __restrict__ A,
    half* __restrict__ B,
    half* __restrict__ C
) {
    // Shared memory 声明
    __shared__ half A_shared[128][32];
    __shared__ half B_shared[32][128];
    
    // 寄存器分配
    float C_local[1][1];  // 每个线程处理的分片
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    
    // 初始化
    C_local[0][0] = 0.0f;
    
    // 主循环
    for (int k = 0; k < (K + 31) / 32; ++k) {
        // 协作加载 A 到 shared memory
        // 每个线程加载 128*32/128 = 32 个元素
        for (int i = 0; i < 32; ++i) {
            int row = tx / 4;  // 32 行
            int col = (tx % 4) * 8 + i;  // 32 列
            A_shared[row][col] = A[(by * 128 + row) * K + k * 32 + col];
        }
        
        // 协作加载 B 到 shared memory
        for (int i = 0; i < 8; ++i) {
            int row = (tx % 4) * 2 + i / 4;
            int col = (tx / 4) * 4 + i % 4;
            B_shared[row][col] = B[(k * 32 + row) * N + bx * 128 + col];
        }
        
        __syncthreads();
        
        // 计算局部矩阵乘法
        for (int kk = 0; kk < 32; ++kk) {
            C_local[0][0] += __half2float(A_shared[tx / 4][kk]) *
                             __half2float(B_shared[kk][tx % 4 * 32]);
        }
        
        __syncthreads();
    }
    
    // 写回结果
    C[(by * 128 + tx / 4) * N + bx * 128 + tx % 4 * 32] = __float2half(C_local[0][0]);
}
```

### 2.6 关键 Pass 走读

#### TileLangLowerTileOp

这个 Pass 负责将 TileLang 的高层 Tile 操作降低为底层的循环结构：

```python
# Pass 处理前
T.gemm(A_shared, B_shared, C_local)

# Pass 处理后
for i in T.serial(tile_m):
    for j in T.serial(tile_n):
        for k in T.serial(tile_k):
            C_local[i, j] += T.cast(A_shared[i, k], "float32") * T.cast(B_shared[k, j], "float32")
```

这个 Pass 展示了 TileLang 编译器的核心转换逻辑之一。`T.gemm` 是一个高层的矩阵乘法原语，它不包含具体的循环结构和数据类型转换细节。经过 `TileLangLowerTileOp` Pass 处理后，这个高层操作被展开为三重嵌套循环，明确指定了每个维度的迭代范围（tile_m、tile_n、tile_k）。更重要的是，Pass 自动插入了类型转换操作——`T.cast(A_shared[i, k], "float32")` 将 float16 数据转换为 float32 进行计算，这是为了保证矩阵乘法的数值精度。这种从高层语义到低层实现的自动转换是 TileLang 编译器的核心价值之一，它使得开发者可以用简洁的高层 API 描述算法逻辑，而编译器负责生成高效的底层实现。如果目标硬件支持 Tensor Core，编译器还会自动将标量乘加操作替换为 Tensor Core 指令，进一步提升计算效率。

**Pass 内部逻辑**

TileLangLowerTileOp 的内部实现采用了访问者模式（Visitor Pattern），通过 `visit_call` 方法遍历 IR 中的所有函数调用节点，并根据操作类型分派到不同的处理方法。`_lower_gemm` 方法负责将 Gemm 操作转换为底层循环，它首先获取输入 tensor 的 shape 信息，然后根据目标硬件是否支持 Tensor Core 来选择不同的代码生成策略。如果硬件支持 Tensor Core（如 NVIDIA A100/H100），则生成使用 Tensor Core 指令的高效实现；否则，生成标量乘加的通用实现。这种根据硬件能力自动选择实现策略的设计，使得同一份 TileLang 代码可以在不同的硬件平台上获得最优的性能。

#### TileLangInferLayout

这个 Pass 负责推导每个 tensor 的最优内存布局：

`TileLangInferLayout` Pass 是 TileLang 编译器中另一个关键的优化步骤。Shared memory 的布局推导需要考虑 bank conflict 问题——NVIDIA GPU 的 shared memory 被分为 32 个 bank，如果多个线程同时访问同一个 bank 中的不同地址，就会产生 bank conflict，导致访问延迟增加。`infer_shared_memory_layout` 方法通过分析 tensor 的访问模式来检测潜在的 bank conflict，如果检测到冲突，就会通过 padding（填充）或 swizzle（交织）技术来消除冲突。例如，对于一个 128×32 的 shared memory 数组，如果按行主序存储，当 warp 中的线程按列访问时可能会产生 bank conflict，此时可以将列维度增加一个 padding 元素（变为 128×33），从而消除冲突。寄存器布局推导则考虑计算模式——对于 Gemm 操作，每个 thread 需要处理输出矩阵的一个小块，寄存器布局需要确保 thread 之间的数据分配是均匀的，以最大化计算效率。

#### TileLangVectorizePass

`TileLangVectorizePass` Pass 负责将标量内存访问转换为向量化的内存访问。GPU 的全局内存访问是按事务（transaction）进行的，每个事务通常为 32 字节或 128 字节。如果多个连续的线程访问连续的内存地址，这些访问可以被合并为一个或少数几个内存事务，显著提高内存带宽利用率。`visit_store` 方法检测连续的内存访问模式，当检测到连续访问时，将多个标量操作合并为向量操作。例如，如果 4 个连续的 float32 存储操作被检测到，可以合并为一个 `float4` 向量存储操作，将内存事务数量减少为原来的四分之一。这种向量化优化对于提升内存带宽受限（memory-bound）kernel 的性能至关重要。

```python
class TileLangVectorizePass:
    """向量化优化：将连续内存访问转换为向量加载/存储"""
    
    def visit_store(self, store):
        # 检测连续的内存访问模式
        if self._is_contiguous_access(store):
            # 将多个标量操作合并为向量操作
            vector_width = self._compute_vector_width(store)
            return self._emit_vectorized_store(store, vector_width)
        return store
```

### 2.7 IR Dump 调试

TileLang 提供了 IR dump 功能，方便开发者查看每个编译阶段的中间表示：

```python
import tilelang
from tilelang import tvm as tvm

# 开启 IR dump
@tilelang.jit(dump_ir=True)
def my_kernel(A, B):
    ...

# 或通过环境变量
# TILELANG_DUMP_IR=1 python my_script.py

# 获取编译后的 IR
kernel = my_kernel
print("Tile IR:")
print(kernel.get_tile_ir())
print("\nTensorIR:")
print(kernel.get_tir())
print("\nCUDA Code:")
print(kernel.get_kernel_source())
```

> [!CAUTION]
> IR dump 输出可能非常长，特别是对于复杂的 kernel。建议在调试时将输出重定向到文件：`TILELANG_DUMP_IR=1 python my_script.py > ir_dump.txt 2>&1`

---

## 3. 编译缓存机制

### 3.1 缓存架构概览

TileLang 采用两级缓存架构来避免重复编译：

上面的架构图清晰地展示了 TileLang 的两级缓存设计。L1 内存缓存位于进程内部，使用 Python 的字典数据结构实现，查询速度极快（亚微秒级），但仅在当前进程生命周期内有效。L2 文件缓存位于磁盘上（默认路径为 `~/.cache/tilelang/`），虽然查询速度较慢（毫秒级），但可以跨进程、跨会话持久化。两级缓存形成一个逐级回退的查找链：首先检查 L1 内存缓存，如果命中则直接返回；如果 L1 未命中，再检查 L2 文件缓存，如果命中则加载到 L1 并返回；如果 L2 也未命中，则触发完整的 JIT 编译流程，编译完成后同时写入 L1 和 L2 缓存。这种两级缓存设计兼顾了速度和持久性：对于同一进程内的重复调用，L1 缓存提供了极致的访问速度；对于跨会话的重复调用，L2 缓存避免了重复编译的开销。目录结构中的 `func_hash` 和 `config_hash` 两级 hash 使得缓存可以按函数逻辑和配置参数分别组织，提高了缓存管理的灵活性。

### 3.2 文件缓存结构

文件缓存使用两级 hash 目录结构来组织编译产物。`func_hash` 基于函数源码的 SHA256 哈希值，用于区分不同的 kernel 函数；`config_hash` 基于输入 shape、dtype、target 等配置信息的哈希值，用于区分同一函数的不同配置变种。这种两级 hash 结构使得缓存查找非常高效——给定一个函数和一组输入参数，可以在 O(1) 时间内定位到对应的编译产物。`metadata.json` 文件记录了编译产物的元数据信息，包括编译时间、TileLang 版本、CUDA 版本等，用于后续的缓存有效性验证。`kernel.cu` 保存了生成的 CUDA 源码，便于调试和审查；`kernel.so` 是编译后的共享库，可以直接被 Python 加载执行。

**metadata.json 示例**

metadata.json 文件记录了编译产物的完整上下文信息，这对于缓存的有效性验证至关重要。`func_hash` 和 `config_hash` 用于快速定位缓存条目；`source_hash` 用于验证源码是否发生了变化；`target` 和 `arch` 记录了目标硬件平台信息，确保缓存只在兼容的硬件上被使用；`tilelang_version` 和 `tvm_version` 记录了编译工具链的版本，当版本升级时可以自动失效旧的缓存；`cuda_version` 确保缓存的 CUDA 代码与当前系统的 CUDA 版本兼容。这些元数据信息共同构成了一个完整的缓存有效性检查机制，确保缓存的正确性和安全性。

### 3.3 内存缓存

进程内缓存使用 LRU（Least Recently Used）策略管理编译结果：

LRU（最近最少使用）缓存淘汰策略是一种经典的缓存管理算法，它通过维护一个按访问时间排序的链表来跟踪缓存条目的使用情况。当缓存命中时，对应的条目会被移动到链表末尾，表示它是最近被使用的；当缓存需要淘汰时，链表头部的条目（即最久未使用的）会被首先移除。`get` 方法在缓存命中时调用 `move_to_end` 将条目移到末尾，确保频繁使用的条目不会被过早淘汰。`put` 方法在缓存满时调用 `popitem(last=False)` 移除链表头部的条目。`invalidate` 方法支持清除单个条目或整个缓存，这在代码更新或调试时非常有用。LRU 策略的优势在于其实现简单、性能优秀（O(1) 时间复杂度），且在大多数实际场景下都能取得接近最优的缓存命中率。默认的缓存大小为 128 个条目，这个值在内存占用和缓存命中率之间取得了良好的平衡。

```python
class InMemoryCache:
    """进程内编译缓存"""
    
    def __init__(self, max_size=128):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key):
        """获取缓存，命中时移动到最近使用位置"""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        """存入缓存，超出容量时淘汰最久未使用的条目"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def invalidate(self, key=None):
        """清除缓存"""
        if key is None:
            self.cache.clear()
        elif key in self.cache:
            del self.cache[key]
```

### 3.4 缓存 Key 的生成

缓存 Key 是决定缓存命中的核心因素。TileLang 使用多级 hash 来生成唯一的缓存标识：

```python
import hashlib
import json

def generate_cache_key(func, args, target, config):
    """
    生成缓存 Key
    
    Key 组成：
    1. 函数源码 hash（函数逻辑变化 → 新 key）
    2. 输入 shape hash（shape 变化 → 新 key）
    3. dtype hash（数据类型变化 → 新 key）
    4. 目标平台 hash（硬件/编译选项变化 → 新 key）
    """
    # 1. 函数源码
    func_source = inspect.getsource(func)
    func_hash = hashlib.sha256(func_source.encode()).hexdigest()[:16]
    
    # 2. 输入 shape
    shape_info = []
    for arg in args:
        if hasattr(arg, 'shape'):
            shape_info.append(tuple(arg.shape))
        else:
            shape_info.append(arg)
    shape_hash = hashlib.sha256(
        json.dumps(shape_info, sort_keys=True).encode()
    ).hexdigest()[:16]
    
    # 3. dtype
    dtype_info = [str(arg.dtype) if hasattr(arg, 'dtype') else str(type(arg)) for arg in args]
    dtype_hash = hashlib.sha256(
        json.dumps(dtype_info).encode()
    ).hexdigest()[:8]
    
    # 4. 目标平台
    target_info = {
        "target": str(target),
        "arch": get_gpu_arch(),
        "tilelang_version": get_tilelang_version(),
        "config": config
    }
    target_hash = hashlib.sha256(
        json.dumps(target_info, sort_keys=True).encode()
    ).hexdigest()[:16]
    
    return f"{func_hash}_{shape_hash}_{dtype_hash}_{target_hash}"
```

缓存 Key 的生成是整个缓存机制的核心环节，它直接决定了缓存能否正确命中。这段代码展示了缓存 Key 的四个组成部分：函数源码 hash 确保代码变化时缓存失效，输入 shape hash 确保不同形状的输入使用不同的编译产物，dtype hash 确保不同数据类型使用不同的优化策略，目标平台 hash 确保不同硬件使用不同的编译结果。使用 SHA256 算法生成 hash 值可以保证极低的碰撞概率，而截取前 16 位（64 位十六进制字符）则在唯一性和实用性之间取得了平衡。值得注意的是，函数源码 hash 使用 `inspect.getsource` 获取源码，这意味着函数定义中的任何变化（包括注释）都会导致缓存失效。因此，在编写 kernel 函数时应避免频繁修改无关的注释，以提高缓存命中率。此外，`sort_keys=True` 确保 JSON 序列化时字典的键按字母顺序排列，避免因键顺序不同而导致的 hash 不一致。

### 3.5 缓存失效策略

```python
class CacheInvalidationPolicy:
    """缓存失效策略"""
    
    def should_invalidate(self, cache_entry):
        """
        判断缓存条目是否应该失效
        
        失效条件：
        1. TileLang 版本变化
        2. CUDA 版本变化
        3. 源码 hash 变化
        4. 缓存文件损坏
        """
        # 版本检查
        current_version = get_tilelang_version()
        cached_version = cache_entry.metadata.get("tilelang_version")
        if current_version != cached_version:
            return True, "version_mismatch"
        
        # CUDA 版本检查
        current_cuda = get_cuda_version()
        cached_cuda = cache_entry.metadata.get("cuda_version")
        if current_cuda != cached_cuda:
            return True, "cuda_version_mismatch"
        
        # 文件完整性检查
        if not self._verify_file_integrity(cache_entry):
            return True, "file_corrupted"
        
        return False, None
    
    def _verify_file_integrity(self, entry):
        """验证编译产物文件的完整性"""
        import os
        required_files = ["kernel.so", "metadata.json"]
        for f in required_files:
            path = os.path.join(entry.cache_dir, f)
            if not os.path.exists(path):
                return False
            if os.path.getsize(path) == 0:
                return False
        return True
```

缓存失效策略是确保缓存正确性的重要机制。这段代码实现了三种主要的失效条件：版本检查确保 TileLang 框架升级后，旧的编译产物不会被错误地使用——因为新版本可能修复了代码生成的 bug 或引入了新的优化 Pass，使用旧的编译产物可能导致运行时错误或性能下降。CUDA 版本检查确保编译产物与当前系统的 CUDA 运行时兼容——不同版本的 CUDA 可能改变了 PTX 指令集或共享库的 ABI，使用不兼容的编译产物会导致链接错误或运行时崩溃。文件完整性检查验证编译产物文件是否存在且非空——这可以处理磁盘错误、意外删除等异常情况。在实际应用中，还应该考虑 GPU 驱动版本的变化，因为驱动更新可能改变 GPU 的固件行为。失效策略的设计需要在正确性和缓存命中率之间取得平衡——过于严格的失效条件会导致缓存频繁失效，降低缓存效率；过于宽松的失效条件则可能导致使用不正确的编译产物。

### 3.6 缓存命中率优化技巧

这段代码提供了四个实用的缓存命中率优化技巧。技巧一"预热缓存"是最常用的方法，它在应用启动时使用典型的输入 shape 和 dtype 提前触发所有可能的 kernel 编译，将编译开销从用户请求路径移到启动阶段。技巧二"使用固定的函数定义"强调了避免在动态生成的函数中使用 JIT 的重要性——因为每次调用都会生成新的函数对象，其源码可能包含动态变化的参数值，导致缓存无法命中。技巧三"将可变参数提取为输入"是技巧二的改进方案——通过将变化的参数（如 scale）作为函数的输入参数而非闭包变量，可以保持函数源码的稳定性，提高缓存命中率。技巧四"清理过期缓存"是一个实用的维护操作，通过定期清理长时间未使用的缓存文件，避免磁盘空间的浪费。这些优化技巧在实际项目中非常重要，合理运用可以将 JIT 编译的首次延迟降低到几乎为零。

> [!TIP]
> 在生产环境中，建议将缓存目录挂载到持久化存储，并在 CI/CD 流水线中缓存 `~/.cache/tilelang/` 目录，以避免每次部署都重新编译。

---

## 4. Kernel Launch 配置自动推导

### 4.1 自动配置的核心问题

Kernel Launch 配置包括三个关键参数：

上面的图表清晰地定义了 Kernel Launch 配置的三个核心要素。Grid Dimension 决定了 GPU 上启动多少个 thread block，它通常与输出 tensor 的 tile 数量直接相关——例如，对于一个 1024×1024 的输出矩阵，如果每个 block 处理 128×128 的 tile，那么 grid 维度就是 (8, 8, 1)，总共启动 64 个 block。Block Dimension 决定了每个 block 包含多少个 thread，它直接影响 shared memory 的使用量和 GPU 的 occupancy（占用率）——更多的 thread 可以提高 occupancy，但也意味着更多的寄存器和 shared memory 消耗。Shared Memory Size 是每个 block 使用的 shared memory 字节数，它会影响 GPU 的 occupancy——shared memory 使用量越大，SM 上能同时驻留的 block 数量越少，occupancy 越低。这三个参数之间存在复杂的权衡关系，TileLang 的自动推导算法正是要在这三者之间找到最优的平衡点。

### 4.2 自动推导算法

Kernel Launch 配置的自动推导是 TileLang JIT 编译器的另一个核心功能。这个推导过程分为五个步骤：首先分析 kernel 的内存访问模式和计算特征，确定是 compute-bound 还是 memory-bound；然后获取目标 GPU 的硬件参数（如 SM 数量、最大线程数、shared memory 容量等）；接着计算最优的 tile 大小，目标是最大化数据复用率和 occupancy；然后根据 tile 大小计算 grid 和 block 维度以及 shared memory 使用量；最后验证配置是否满足硬件约束，并在必要时进行调整。`_get_hardware_params` 方法维护了一个 GPU 规格数据库，包含了常见 GPU 型号（如 A100、H100）的关键参数。`_compute_tile_size` 方法使用评分函数对候选 tile 配置进行排序，评分考虑了数据复用率、occupancy 和算术强度三个因素。这种自动推导机制使得开发者无需深入了解 GPU 硬件细节就能获得良好的性能，同时保留了手动覆盖的能力以满足特殊需求。

```python
class LaunchConfigInferencer:
    """Kernel Launch 配置自动推导器"""
    
    def infer(self, kernel_func, input_shapes, target_gpu):
        """
        自动推导最优的 Launch 配置
        
        步骤：
        1. 分析 kernel 的内存访问模式
        2. 计算理论最优的 tile 大小
        3. 考虑硬件约束调整配置
        4. 选择最大化 occupancy 的配置
        """
        # Step 1: 分析 kernel
        analysis = self._analyze_kernel(kernel_func)
        
        # Step 2: 获取硬件参数
        hw_params = self._get_hardware_params(target_gpu)
        
        # Step 3: 计算 tile 大小
        tile_config = self._compute_tile_size(
            analysis, hw_params, input_shapes
        )
        
        # Step 4: 计算 Grid/Block 维度
        grid_dim = self._compute_grid_dim(input_shapes, tile_config)
        block_dim = self._compute_block_dim(tile_config, hw_params)
        smem_size = self._compute_smem_size(tile_config, analysis)
        
        # Step 5: 验证并调整
        return self._validate_and_adjust(
            grid_dim, block_dim, smem_size, hw_params
        )
    
    def _get_hardware_params(self, target_gpu):
        """获取目标 GPU 的硬件参数"""
        gpu_specs = {
            "A100": {
                "max_threads_per_block": 1024,
                "max_shared_memory_per_block": 49152,  # 48KB
                "max_shared_memory_per_block_optin": 166912,  # 164KB
                "warp_size": 32,
                "num_sms": 108,
                "registers_per_block": 65536,
                "max_warps_per_block": 32,
                "l2_cache_size": 41943040,  # 40MB
            },
            "H100": {
                "max_threads_per_block": 1024,
                "max_shared_memory_per_block": 49152,
                "max_shared_memory_per_block_optin": 232448,  # 228KB
                "warp_size": 32,
                "num_sms": 132,
                "registers_per_block": 65536,
                "max_warps_per_block": 32,
                "l2_cache_size": 52428800,  # 50MB
            },
            # ... 更多 GPU 型号
        }
        return gpu_specs.get(target_gpu, gpu_specs["A100"])
    
    def _compute_tile_size(self, analysis, hw_params, input_shapes):
        """
        计算最优 tile 大小
        
        目标：最大化数据复用，同时满足硬件约束
        """
        M, N, K = input_shapes
        
        # 候选 tile 大小
        candidates = [
            (128, 128, 32),
            (128, 64, 32),
            (64, 128, 32),
            (64, 64, 32),
            (256, 64, 16),
            (64, 256, 16),
        ]
        
        best_config = None
        best_score = -1
        
        for tile_m, tile_n, tile_k in candidates:
            # 检查是否满足硬件约束
            threads_needed = (tile_m * tile_n) // 64  # 假设每个线程处理 64 个元素
            smem_needed = (tile_m * tile_k + tile_k * tile_n) * 2  # float16
            
            if threads_needed > hw_params["max_threads_per_block"]:
                continue
            if smem_needed > hw_params["max_shared_memory_per_block_optin"]:
                continue
            
            # 计算评分
            score = self._compute_score(
                tile_m, tile_n, tile_k,
                M, N, K,
                hw_params
            )
            
            if score > best_score:
                best_score = score
                best_config = (tile_m, tile_n, tile_k)
        
        return best_config
    
    def _compute_score(self, tile_m, tile_n, tile_k, M, N, K, hw_params):
        """
        评估 tile 配置的优劣
        
        评分因素：
        1. 数据复用率（越高越好）
        2. Occupancy（越高越好）
        3. 内存访问效率（coalescing）
        4. 计算/访存比（越高越好）
        """
        # 数据复用率
        reuse_A = tile_n / tile_k  # A 被复用的次数
        reuse_B = tile_m / tile_k  # B 被复用的次数
        reuse_score = (reuse_A + reuse_B) / 2
        
        # Occupancy 估算
        threads_per_block = (tile_m * tile_n) // 64
        warps_per_block = threads_per_block // hw_params["warp_size"]
        max_warps = hw_params["max_warps_per_block"]
        occupancy = min(warps_per_block / max_warps, 1.0)
        
        # 计算/访存比
        compute_ops = 2 * tile_m * tile_n * tile_k  # MAC 操作
        memory_bytes = (tile_m * tile_k + tile_k * tile_n) * 2  # float16
        arithmetic_intensity = compute_ops / memory_bytes
        
        # 综合评分
        score = (
            0.3 * reuse_score +
            0.3 * occupancy +
            0.4 * arithmetic_intensity
        )
        
        return score
```

### 4.3 推导示例

```python
import tilelang
import torch

# 示例：自动推导矩阵乘法的 Launch 配置
@tilelang.jit
def auto_matmul(A, B):
    M, K = A.shape
    _, N = B.shape
    
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        # TileLang 自动推导 grid/block 维度
        with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "float16")
            B_shared = T.alloc_shared((32, 128), "float16")
            C_local = T.alloc_fragment((128, 128), "float32")
            
            T.clear(C_local)
            for k in T.serial(T.ceildiv(K, 32)):
                T.copy(A[by * 128, k * 32], A_shared)
                T.copy(B[k * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            
            T.copy(C_local, C[by * 128, bx * 128])
    
    return main

# 查看自动推导的配置
kernel = auto_matmul
A = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
B = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")

# 打印推导结果
print(f"Grid:  ({kernel.grid_dim})")   # (8, 8, 1)
print(f"Block: ({kernel.block_dim})")   # (128, 1, 1)
print(f"SMem:  {kernel.smem_size} bytes")  # 16384 bytes
```

### 4.4 手动覆盖 vs 自动推导

```python
# 方式 1：完全自动推导
@tilelang.jit
def auto_kernel(A, B):
    # TileLang 自动选择最优配置
    ...

# 方式 2：部分手动指定
@tilelang.jit(
    threads=256,  # 手动指定线程数
    # grid 和 block 由自动推导生成
)
def partial_manual_kernel(A, B):
    ...

# 方式 3：完全手动指定
@tilelang.jit(
    grid=(8, 8, 1),      # 手动指定 grid 维度
    block=(256, 1, 1),    # 手动指定 block 维度
    smem=32768,           # 手动指定 shared memory (32KB)
)
def full_manual_kernel(A, B):
    ...

# 对比表
comparison = """
┌────────────────────┬──────────────┬──────────────┬──────────────┐
│     配置方式        │   自动推导    │  部分手动    │  完全手动     │
├────────────────────┼──────────────┼──────────────┼──────────────┤
│ 开发效率            │    高         │    中         │    低        │
│ 性能可预测性        │    中         │    中         │    高        │
│ 可移植性            │    高         │    中         │    低        │
│ 调试难度            │    低         │    中         │    高        │
│ 适用场景            │  通用场景    │  特定优化    │  极致性能    │
└────────────────────┴──────────────┴──────────────┴──────────────┘
"""
```

> [!WARNING]
> 手动指定 Launch 配置时，务必确保配置值与目标硬件兼容。错误的配置（如超过最大线程数或 shared memory 限制）会导致编译失败或运行时错误。

---

## 5. 与 PyTorch 的无缝集成

### 5.1 集成架构

TileLang 与 PyTorch 的集成建立在多个层次上：

```
┌─────────────────────────────────────────────────────────────────┐
│                    TileLang × PyTorch 集成架构                    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Level 4: torch.compile 后端                               │ │
│  │  └── 自动将 PyTorch 代码编译为 TileLang kernel              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                          ↓                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Level 3: torch.autograd.Function                          │ │
│  │  └── 将 TileLang kernel 包装为可微分的 PyTorch 算子         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                          ↓                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Level 2: DLPack / CUDA Stream                             │ │
│  │  └── 零拷贝 tensor 共享，stream 同步                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                          ↓                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Level 1: torch.Tensor ↔ TileLang Tensor                   │ │
│  │  └── 底层数据格式兼容                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 torch.compile 后端实现原理

TileLang 可以作为 `torch.compile` 的自定义后端，自动将 PyTorch 代码编译为高效的 TileLang kernel：

```python
import torch
import tilelang

# 注册 TileLang 为 torch.compile 的后端
@torch.compile(backend="tilelang")
def pytorch_function(A, B):
    """使用标准 PyTorch 操作编写"""
    C = torch.matmul(A, B)
    C = torch.relu(C)
    C = C / C.norm()
    return C

# torch.compile 的工作流程：
# 1. 捕获 PyTorch 计算图
# 2. 将计算图转换为 TileLang DSL
# 3. JIT 编译为原生 CUDA kernel
# 4. 返回编译后的可执行函数
```

**自定义后端注册**

```python
from torch._dynamo import register_backend

@register_backend
def tilelang_backend(gm: torch.fx.GraphModule, example_inputs):
    """
    TileLang torch.compile 后端
    
    工作流程：
    1. 解析 torch.fx.GraphModule
    2. 识别可融合的子图
    3. 生成 TileLang kernel
    4. JIT 编译并返回可执行函数
    """
    # Step 1: 图分析
    fused_subgraphs = analyze_and_fuse(gm)
    
    # Step 2: 代码生成
    kernels = []
    for subgraph in fused_subgraphs:
        kernel_code = generate_tilelang_code(subgraph)
        compiled_kernel = tilelang.compile(kernel_code)
        kernels.append(compiled_kernel)
    
    # Step 3: 构建执行函数
    def forward(*args):
        results = []
        for kernel, inputs in zip(kernels, partition_inputs(args)):
            results.append(kernel(*inputs))
        return combine_results(results)
    
    return forward
```

### 5.3 torch.autograd.Function 集成

将 TileLang kernel 包装为支持自动微分的 PyTorch 算子：

```python
import torch
import tilelang
import tilelang.language as T

class TileLangMatmul(torch.autograd.Function):
    """TileLang 矩阵乘法的 PyTorch 自动微分包装"""
    
    @staticmethod
    def forward(ctx, A, B):
        # 保存反向传播需要的 tensor
        ctx.save_for_backward(A, B)
        
        # JIT 编译并执行前向 kernel
        M, K = A.shape
        _, N = B.shape
        
        @tilelang.jit
        def matmul_forward(M, N, K, dtype):
            @T.prim_func
            def main(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((K, N), dtype),
                C: T.Tensor((M, N), dtype),
            ):
                with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
                    A_shared = T.alloc_shared((128, 32), dtype)
                    B_shared = T.alloc_shared((32, 128), dtype)
                    C_local = T.alloc_fragment((128, 128), "float32")
                    
                    T.clear(C_local)
                    for k in T.serial(T.ceildiv(K, 32)):
                        T.copy(A[by * 128, k * 32], A_shared)
                        T.copy(B[k * 32, bx * 128], B_shared)
                        T.gemm(A_shared, B_shared, C_local)
                    
                    T.copy(C_local, C[by * 128, bx * 128])
            return main
        
        C = torch.empty(M, N, dtype=A.dtype, device=A.device)
        kernel = matmul_forward(M, N, K, str(A.dtype))
        kernel(A, B, C)
        return C
    
    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        
        # dA = grad_output @ B^T
        # dB = A^T @ grad_output
        grad_A = torch.matmul(grad_output, B.T)
        grad_B = torch.matmul(A.T, grad_output)
        
        return grad_A, grad_B

# 使用方式
matmul = TileLangMatmul.apply
A = torch.randn(1024, 1024, dtype=torch.float16, device="cuda", requires_grad=True)
B = torch.randn(1024, 1024, dtype=torch.float16, device="cuda", requires_grad=True)

C = matmul(A, B)
loss = C.sum()
loss.backward()
```

### 5.4 torch.cuda.Event 同步机制

在混合使用 TileLang kernel 和 PyTorch 操作时，需要正确处理 CUDA stream 同步：

```python
import torch
import tilelang

def mixed_workflow(A, B, C):
    """混合使用 TileLang 和 PyTorch 操作的正确同步方式"""
    
    # 创建事件用于同步
    tilelang_done = torch.cuda.Event()
    pytorch_done = torch.cuda.Event()
    
    # TileLang kernel（可能在独立的 stream 上执行）
    result_tilelang = tilelang_kernel(A, B)
    
    # 记录 TileLang 完成事件
    tilelang_done.record()
    
    # 等待 TileLang 完成后再执行 PyTorch 操作
    torch.cuda.current_stream().wait_event(tilelang_done)
    
    # PyTorch 操作
    result_pytorch = torch.matmul(result_tilelang, C)
    pytorch_done.record()
    
    # 如果需要将结果传回 TileLang kernel
    # 等待 PyTorch 完成
    tilelang_stream = tilelang.get_current_stream()
    tilelang_stream.wait_event(pytorch_done)
    
    final_result = another_tilelang_kernel(result_pytorch)
    
    return final_result
```

### 5.5 完整示例：自定义 PyTorch 算子

以下是一个完整的示例，展示如何使用 TileLang 实现一个自定义的 Flash Attention 算子并集成到 PyTorch 中：

```python
import torch
import tilelang
import tilelang.language as T
import math

@tilelang.jit
def flash_attention_fwd(batch, heads, seq_len, dim, causal=True):
    """Flash Attention 前向传播 kernel"""
    
    @T.prim_func
    def main(
        Q: T.Tensor((batch, heads, seq_len, dim), "float16"),
        K: T.Tensor((batch, heads, seq_len, dim), "float16"),
        V: T.Tensor((batch, heads, seq_len, dim), "float16"),
        O: T.Tensor((batch, heads, seq_len, dim), "float16"),
    ):
        # 配置参数
        BLOCK_M = 128
        BLOCK_N = 64
        NUM_WARPS = 4
        NUM_THREADS = NUM_WARPS * 32
        
        with T.Kernel(
            T.ceildiv(seq_len, BLOCK_M),
            heads,
            batch,
            threads=NUM_THREADS
        ) as (bx, by, bz):
            # Shared memory 分配
            Q_shared = T.alloc_shared((BLOCK_M, dim), "float16")
            K_shared = T.alloc_shared((BLOCK_N, dim), "float16")
            V_shared = T.alloc_shared((BLOCK_N, dim), "float16")
            
            # 寄存器分配
            acc = T.alloc_fragment((BLOCK_M, dim), "float32")
            scores = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            row_max = T.alloc_fragment((BLOCK_M,), "float32")
            row_sum = T.alloc_fragment((BLOCK_M,), "float32")
            
            # 初始化
            T.clear(acc)
            T.fill(row_max, -T.infinity("float32"))
            T.fill(row_sum, 0.0)
            
            # 加载 Q
            T.copy(Q[bz, by, bx * BLOCK_M, 0], Q_shared)
            
            # 遍历 K, V 的 blocks
            for j in T.serial(T.ceildiv(seq_len, BLOCK_N)):
                # 加载 K block
                T.copy(K[bz, by, j * BLOCK_N, 0], K_shared)
                
                # 计算 attention scores: Q @ K^T
                T.clear(scores)
                for d in T.serial(dim):
                    for m in T.serial(BLOCK_M):
                        for n in T.serial(BLOCK_N):
                            scores[m, n] += T.cast(Q_shared[m, d], "float32") * T.cast(K_shared[n, d], "float32")
                
                # 缩放
                scale = 1.0 / math.sqrt(dim)
                for m in T.serial(BLOCK_M):
                    for n in T.serial(BLOCK_N):
                        scores[m, n] *= scale
                
                # Causal mask
                if causal:
                    for m in T.serial(BLOCK_M):
                        for n in T.serial(BLOCK_N):
                            row_idx = bx * BLOCK_M + m
                            col_idx = j * BLOCK_N + n
                            if col_idx > row_idx:
                                scores[m, n] = -T.infinity("float32")
                
                # Online softmax
                prev_max = T.alloc_fragment((BLOCK_M,), "float32")
                T.copy(row_max, prev_max)
                
                for m in T.serial(BLOCK_M):
                    for n in T.serial(BLOCK_N):
                        row_max[m] = T.max(row_max[m], scores[m, n])
                
                for m in T.serial(BLOCK_M):
                    exp_scale = T.exp(prev_max[m] - row_max[m])
                    row_sum[m] = row_sum[m] * exp_scale
                    for d in T.serial(dim):
                        acc[m, d] *= exp_scale
                
                # 加载 V 并累加
                T.copy(V[bz, by, j * BLOCK_N, 0], V_shared)
                
                for m in T.serial(BLOCK_M):
                    for n in T.serial(BLOCK_N):
                        scores[m, n] = T.exp(scores[m, n] - row_max[m])
                        row_sum[m] += scores[m, n]
                
                for m in T.serial(BLOCK_M):
                    for d in T.serial(dim):
                        for n in T.serial(BLOCK_N):
                            acc[m, d] += scores[m, n] * T.cast(V_shared[n, d], "float32")
            
            # 归一化
            for m in T.serial(BLOCK_M):
                for d in T.serial(dim):
                    acc[m, d] /= row_sum[m]
            
            # 写回结果
            T.copy(acc, O[bz, by, bx * BLOCK_M, 0])
    
    return main

class FlashAttention(torch.autograd.Function):
    """PyTorch Flash Attention 包装"""
    
    @staticmethod
    def forward(ctx, Q, K, V, causal=True):
        batch, heads, seq_len, dim = Q.shape
        
        O = torch.empty_like(Q)
        kernel = flash_attention_fwd(batch, heads, seq_len, dim, causal)
        kernel(Q, K, V, O)
        
        ctx.save_for_backward(Q, K, V, O)
        ctx.causal = causal
        
        return O
    
    @staticmethod
    def backward(ctx, grad_output):
        # 简化版本，实际实现需要更复杂的反向传播
        Q, K, V, O = ctx.saved_tensors
        
        # 使用 PyTorch 的标准实现作为反向传播
        scale = 1.0 / math.sqrt(Q.shape[-1])
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        if ctx.causal:
            mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
            scores.masked_fill_(mask, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        
        grad_V = torch.matmul(attn.transpose(-2, -1), grad_output)
        grad_attn = torch.matmul(grad_output, V.transpose(-2, -1))
        
        # Softmax 反向传播
        grad_scores = attn * (grad_attn - (grad_attn * attn).sum(dim=-1, keepdim=True))
        grad_scores *= scale
        
        grad_Q = torch.matmul(grad_scores, K)
        grad_K = torch.matmul(grad_scores.transpose(-2, -1), Q)
        
        return grad_Q, grad_K, grad_V, None

# 使用
flash_attn = FlashAttention.apply

# 在模型中使用
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        
        self.W_q = torch.nn.Linear(dim, dim)
        self.W_k = torch.nn.Linear(dim, dim)
        self.W_v = torch.nn.Linear(dim, dim)
        self.W_o = torch.nn.Linear(dim, dim)
    
    def forward(self, x):
        B, L, D = x.shape
        
        Q = self.W_q(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        
        O = flash_attn(Q, K, V, causal=True)
        
        O = O.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(O)
```

> [!TIP]
> 在实际生产中，建议使用 `torch.autograd.Function` 的 `setup_context` 方法（PyTorch 2.0+）来更高效地管理上下文，减少内存占用。

---

## 6. 首次编译 vs 缓存命中

### 6.1 性能对比数据

以下是在 NVIDIA A100 上的典型性能对比数据：

| 算子 | 输入 Shape | 首次编译时间 | 缓存命中时间 | 加速比 |
|------|-----------|-------------|-------------|--------|
| MatMul | (1024, 1024) × (1024, 1024) | 1.2s | 0.8ms | 1500x |
| MatMul | (4096, 4096) × (4096, 4096) | 1.5s | 0.9ms | 1667x |
| Flash Attention | (1, 32, 4096, 128) | 3.2s | 1.2ms | 2667x |
| Conv2D | (1, 64, 224, 224) | 2.1s | 0.6ms | 3500x |
| Softmax | (4096, 4096) | 0.8s | 0.3ms | 2667x |
| LayerNorm | (4096, 1024) | 0.9s | 0.4ms | 2250x |
| GEMM + Bias + ReLU | (1024, 1024) | 1.8s | 0.7ms | 2571x |
| Transpose + MatMul | (1024, 1024) | 1.6s | 0.9ms | 1778x |

> [!TIP]
> 上述数据仅为参考值。实际编译时间受 CPU 性能、磁盘 I/O、系统负载等因素影响。缓存命中时间主要由内存分配和 kernel launch 开销决定。

### 6.2 Warm-up 策略

为了减少首次编译延迟对用户体验的影响，可以采用以下 Warm-up 策略：

```python
class TileLangWarmupManager:
    """TileLang 编译预热管理器"""
    
    def __init__(self):
        self.warmed_configs = set()
    
    def warmup_kernel(self, kernel_func, input_configs):
        """
        预热 kernel 编译
        
        Args:
            kernel_func: TileLang kernel 函数
            input_configs: 列表，每个元素为 (shape, dtype, device) 元组
        """
        for shape, dtype, device in input_configs:
            config_key = (shape, dtype, device)
            
            if config_key in self.warmed_configs:
                continue
            
            # 创建 dummy 输入
            dummy_inputs = self._create_dummy_inputs(shape, dtype, device)
            
            # 触发编译（不关心计算结果）
            try:
                kernel_func(*dummy_inputs)
                self.warmed_configs.add(config_key)
                print(f"Warmed up: shape={shape}, dtype={dtype}")
            except Exception as e:
                print(f"Warmup failed for {config_key}: {e}")
    
    def _create_dummy_inputs(self, shape, dtype, device):
        """创建 dummy 输入 tensor"""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "int32": torch.int32,
        }
        return [torch.randn(shape, dtype=dtype_map[dtype], device=device)]

# 使用示例
warmup_manager = TileLangWarmupManager()

# 在应用启动时预热所有配置
common_configs = [
    ((1024, 1024), "float16", "cuda"),
    ((2048, 2048), "float16", "cuda"),
    ((4096, 4096), "float16", "cuda"),
    ((1024, 1024), "bfloat16", "cuda"),
]

warmup_manager.warmup_kernel(my_matmul_kernel, common_configs)
```

### 6.3 预编译（Ahead-of-Time Warmup）

在某些场景下，可以将编译过程提前到应用部署阶段：

```python
def precompile_kernels(output_dir, kernel_configs):
    """
    预编译 kernel 并保存到指定目录
    
    可用于：
    1. Docker 镜像构建时预编译
    2. CI/CD 流水线中预编译
    3. 首次部署时预编译
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for config in kernel_configs:
        kernel_func = config["func"]
        shapes = config["shapes"]
        
        for shape in shapes:
            # 触发编译
            dummy_inputs = create_dummy_inputs(shape)
            kernel_func(*dummy_inputs)
            
            # 缓存会自动保存到 ~/.cache/tilelang/
            print(f"Precompiled: {config['name']} with shape {shape}")
    
    # 复制缓存到输出目录
    cache_dir = os.path.expanduser("~/.cache/tilelang")
    import shutil
    shutil.copytree(cache_dir, output_dir, dirs_exist_ok=True)

# Docker 构建脚本
# RUN python precompile.py --output /opt/tilelang_cache
# ENV TILELANG_CACHE_DIR=/opt/tilelang_cache
```

---

## 7. 编译选项与环境变量

### 7.1 环境变量一览

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `TILELANG_CACHE_DIR` | `~/.cache/tilelang` | 编译缓存目录 |
| `TILELANG_PROFILE` | `0` | 开启性能分析（1 为开启） |
| `TILELANG_DEBUG` | `0` | 开启调试模式（1 为开启，2 为详细调试） |
| `TILELANG_DUMP_IR` | `0` | dump 中间 IR（1 为开启） |
| `TILELANG_DUMP_CODE` | `0` | dump 生成的源码（1 为开启） |
| `TILELANG_LOG_LEVEL` | `WARNING` | 日志级别（DEBUG/INFO/WARNING/ERROR） |
| `TILELANG_DISABLE_CACHE` | `0` | 禁用缓存（1 为禁用，用于调试） |
| `TILELANG_MAX_CACHE_SIZE` | `1024` | 内存缓存最大条目数 |
| `TILELANG_NUM_BUILD_WORKERS` | `4` | 并行编译 worker 数 |
| `TILELANG_CUDA_ARCH` | 自动检测 | 指定 CUDA 计算能力 |

### 7.2 编译选项表

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target` | str | `"cuda"` | 编译目标（cuda/hip/ascend/cpu） |
| `opt_level` | int | `3` | 优化级别（0-3，3 为最高） |
| `debug` | bool | `False` | 是否开启调试信息 |
| `dump_ir` | bool | `False` | 是否 dump 中间 IR |
| `cache_dir` | str | `None` | 自定义缓存目录 |
| `threads` | int | `None` | Kernel 线程数（自动推导为 None） |
| `grid` | tuple | `None` | Grid 维度（自动推导为 None） |
| `block` | tuple | `None` | Block 维度（自动推导为 None） |
| `smem` | int | `None` | Shared memory 大小（自动推导为 None） |

### 7.3 代码示例

```python
import tilelang
import tilelang.language as T
import os

# 示例 1：通过环境变量配置
os.environ["TILELANG_CACHE_DIR"] = "/data/tilelang_cache"
os.environ["TILELANG_PROFILE"] = "1"
os.environ["TILELANG_LOG_LEVEL"] = "DEBUG"

# 示例 2：通过装饰器参数配置
@tilelang.jit(
    target="cuda",
    opt_level=3,
    debug=False,
    cache_dir="/data/tilelang_cache",
)
def optimized_kernel(A, B):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "float16")
            B_shared = T.alloc_shared((32, 128), "float16")
            C_local = T.alloc_fragment((128, 128), "float32")
            
            T.clear(C_local)
            for k in T.serial(T.ceildiv(K, 32)):
                T.copy(A[by * 128, k * 32], A_shared)
                T.copy(B[k * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            
            T.copy(C_local, C[by * 128, bx * 128])
    
    return main

# 示例 3：调试模式（输出详细信息）
@tilelang.jit(debug=True, dump_ir=True)
def debug_kernel(A, B):
    # 在调试模式下，会输出：
    # 1. 每个编译阶段的 IR
    # 2. 生成的 CUDA 代码
    # 3. 编译时间统计
    # 4. 缓存命中/未命中信息
    ...

# 示例 4：性能分析模式
@tilelang.jit(profile=True)
def profiled_kernel(A, B):
    # 在性能分析模式下，会额外输出：
    # 1. Kernel 执行时间
    # 2. 内存传输时间
    # 3. SM 占用率
    # 4. 内存带宽利用率
    ...
```

### 7.4 性能分析输出示例

```
[TILELANG PROFILE] Kernel: matmul_kernel
[TILELANG PROFILE] ============================================
[TILELANG PROFILE] Grid:  (8, 8, 1)
[TILELANG PROFILE] Block: (128, 1, 1)
[TILELANG PROFILE] Shared Memory: 16384 bytes
[TILELANG PROFILE] Registers per Thread: 64
[TILELANG PROFILE] --------------------------------------------
[TILELANG PROFILE] H2D Transfer:     0.12 ms
[TILELANG PROFILE] Kernel Execution: 0.85 ms
[TILELANG PROFILE] D2H Transfer:     0.08 ms
[TILELANG PROFILE] --------------------------------------------
[TILELANG PROFILE] Total:            1.05 ms
[TILELANG PROFILE] ============================================
[TILELANG PROFILE] SM Occupancy:     87.5%
[TILELANG PROFILE] Memory Bandwidth: 78.2% of peak
[TILELANG PROFILE] Compute Throughput: 65.3% of peak
```

> [!WARNING]
> 开启 `debug=True` 或 `dump_ir=True` 会显著增加编译时间和输出量。建议仅在调试时使用，并将输出重定向到文件。

---

## 8. 动态 Shape 支持

### 8.1 Symbolic Shape 机制

TileLang 支持使用符号变量（Symbolic Variables）来表示动态的 tensor shape：

```python
import tilelang
import tilelang.language as T
from tilelang import tvm

# 定义符号变量
M = tvm.te.var("M")  # 符号变量 M
N = tvm.te.var("N")  # 符号变量 N
K = tvm.te.var("K")  # 符号变量 K

@tilelang.jit
def symbolic_matmul(M, N, K):
    """使用符号 shape 的矩阵乘法"""
    
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        # 符号变量在编译时会被具体值替换
        with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "float16")
            B_shared = T.alloc_shared((32, 128), "float16")
            C_local = T.alloc_fragment((128, 128), "float32")
            
            T.clear(C_local)
            for k in T.serial(T.ceildiv(K, 32)):
                T.copy(A[by * 128, k * 32], A_shared)
                T.copy(B[k * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            
            T.copy(C_local, C[by * 128, bx * 128])
    
    return main
```

### 8.2 Dynamic Batch 处理

处理动态 batch size 的常见模式：

```python
@tilelang.jit
def dynamic_batch_attention(batch, heads, max_seq_len, dim):
    """
    支持动态 batch 的 Attention kernel
    
    使用约束（constraints）来优化不同 batch size 的编译
    """
    
    @T.prim_func
    def main(
        Q: T.Tensor((batch, heads, max_seq_len, dim), "float16"),
        K: T.Tensor((batch, heads, max_seq_len, dim), "float16"),
        V: T.Tensor((batch, heads, max_seq_len, dim), "float16"),
        O: T.Tensor((batch, heads, max_seq_len, dim), "float16"),
        seq_lens: T.Tensor((batch,), "int32"),  # 每个样本的实际序列长度
    ):
        with T.Kernel(heads, batch, threads=128) as (by, bz):
            # 使用实际序列长度进行计算
            actual_len = seq_lens[bz]
            
            # 动态循环边界
            for i in T.serial(actual_len):
                # 只处理有效的 token
                ...
    
    return main

# 使用
kernel = dynamic_batch_attention(4, 32, 4096, 128)

# 不同的 batch size 会触发不同的编译
batch_1 = kernel(Q[:1], K[:1], V[:1], O[:1], seq_lens[:1])
batch_4 = kernel(Q[:4], K[:4], V[:4], O[:4], seq_lens[:4])
```

### 8.3 性能影响分析

动态 shape 会对编译和运行时性能产生影响：

```python
# 性能影响对比
"""
┌───────────────────────────────────────────────────────────────────┐
│                    动态 Shape 性能影响                              │
├──────────────────┬──────────────────┬─────────────────────────────┤
│                  │   静态 Shape      │   动态 Shape                │
├──────────────────┼──────────────────┼─────────────────────────────┤
│ 编译时间         │ 短（单次编译）    │ 长（可能多次编译）          │
│ 缓存命中率       │ 高                │ 中（不同 shape 不同缓存）   │
│ 优化程度         │ 最高（完全特化）  │ 中（通用代码）              │
│ 运行时性能       │ 最优              │ 略低（可能有额外判断）      │
│ 内存使用         │ 固定              │ 可能有 padding 浪费         │
│ 适用场景         │ 生产部署          │ 研究探索、多 shape 场景     │
└──────────────────┴──────────────────┴─────────────────────────────┘
"""

# 优化建议：将动态 shape 离散化
def optimized_dynamic_support():
    """
    优化策略：将连续的动态 shape 离散化为有限的几个档位
    
    例如：
    - seq_len: 128, 256, 512, 1024, 2048, 4096
    - batch_size: 1, 2, 4, 8, 16, 32
    
    这样可以：
    1. 减少编译次数
    2. 提高缓存命中率
    3. 保持较高的优化程度
    """
    predefined_shapes = {
        "seq_len": [128, 256, 512, 1024, 2048, 4096],
        "batch_size": [1, 2, 4, 8, 16, 32],
    }
    
    def get_nearest_shape(actual_shape, candidates):
        """获取最接近的预定义 shape（向上取整）"""
        for candidate in sorted(candidates):
            if candidate >= actual_shape:
                return candidate
        return candidates[-1]
    
    return predefined_shapes, get_nearest_shape
```

> [!CAUTION]
> 在生产环境中，建议使用离散化的 shape 档位而非完全动态的 shape。这样可以在保持灵活性的同时，最大化编译缓存的收益。

---

## 9. 多目标编译

### 9.1 统一编译接口

TileLang 提供统一的编译接口，支持多种硬件后端：

```python
import tilelang

# CUDA (NVIDIA GPU)
@tilelang.jit(target="cuda")
def cuda_kernel(A, B):
    ...

# HIP (AMD GPU)
@tilelang.jit(target="hip")
def hip_kernel(A, B):
    ...

# Ascend C (华为昇腾)
@tilelang.jit(target="ascend")
def ascend_kernel(A, B):
    ...

# LLVM CPU
@tilelang.jit(target="llvm")
def cpu_kernel(A, B):
    ...

# 自动检测目标
@tilelang.jit(target="auto")
def auto_kernel(A, B):
    # 根据运行环境自动选择目标
    ...
```

### 9.2 目标后端选择策略

```python
class TargetSelector:
    """硬件目标自动选择器"""
    
    @staticmethod
    def detect_target():
        """
        自动检测当前可用的硬件目标
        
        优先级：
        1. CUDA (NVIDIA GPU)
        2. HIP (AMD GPU)
        3. Ascend (华为昇腾)
        4. LLVM (CPU fallback)
        """
        import subprocess
        
        # 检测 CUDA
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                arch = TargetSelector._get_cuda_arch()
                return "cuda", arch
        except FileNotFoundError:
            pass
        
        # 检测 HIP
        try:
            result = subprocess.run(
                ["rocm-smi"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return "hip", TargetSelector._get_hip_arch()
        except FileNotFoundError:
            pass
        
        # 检测 Ascend
        try:
            result = subprocess.run(
                ["npu-smi"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return "ascend", None
        except FileNotFoundError:
            pass
        
        # Fallback to CPU
        return "llvm", None
    
    @staticmethod
    def _get_cuda_arch():
        """获取 CUDA 计算能力"""
        import torch
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            return f"sm_{capability[0]}{capability[1]}"
        return "sm_70"  # 默认值

# 使用
target, arch = TargetSelector.detect_target()
print(f"Detected target: {target}, arch: {arch}")
```

### 9.3 交叉编译支持

```python
# 在 x86 机器上为 Jetson (ARM + CUDA) 编译
@tilelang.jit(
    target="cuda",
    arch="sm_72",  # Jetson Xavier 的计算能力
    cross_compile=True,
)
def jetson_kernel(A, B):
    ...

# 为多个目标同时编译
def multi_target_compile(kernel_func, targets):
    """
    为多个目标平台编译同一 kernel
    
    可用于：
    1. 构建跨平台部署包
    2. CI/CD 多平台测试
    """
    compiled_kernels = {}
    
    for target_info in targets:
        target = target_info["target"]
        arch = target_info.get("arch")
        
        @tilelang.jit(target=target, arch=arch)
        def compiled_kernel(A, B):
            return kernel_func(A, B)
        
        compiled_kernels[target] = compiled_kernel
    
    return compiled_kernels

# 编译多个目标
targets = [
    {"target": "cuda", "arch": "sm_80"},   # A100
    {"target": "cuda", "arch": "sm_86"},   # RTX 3090
    {"target": "cuda", "arch": "sm_89"},   # RTX 4090
    {"target": "hip"},                       # AMD GPU
]

kernels = multi_target_compile(my_kernel, targets)
```

---

## 10. 常见问题排查

### 10.1 编译超时（Timeout）

**症状**：JIT 编译长时间不返回，或报超时错误。

**可能原因**：
1. Kernel 过于复杂，编译时间过长
2. 系统资源不足（CPU、内存）
3. 依赖的外部编译器（nvcc/gcc）响应慢

**排查步骤**：

```python
# 1. 检查系统资源
import psutil
print(f"CPU 使用率: {psutil.cpu_percent()}%")
print(f"内存使用率: {psutil.virtual_memory().percent}%")
print(f"可用内存: {psutil.virtual_memory().available / 1024**3:.1f} GB")

# 2. 设置编译超时
@tilelang.jit(timeout=60)  # 60 秒超时
def my_kernel(A, B):
    ...

# 3. 简化 kernel 复杂度
# 如果编译超时，尝试减小 tile 大小
@tilelang.jit
def simplified_kernel(A, B):
    # 使用更小的 tile
    TILE_M = 64   # 从 128 减小到 64
    TILE_N = 64
    TILE_K = 16   # 从 32 减小到 16
    ...

# 4. 使用预编译缓存
# 如果编译确实很慢，考虑预编译后缓存
```

### 10.2 缓存不命中（Cache Miss）

**症状**：每次运行都触发重新编译，即使代码没有变化。

**可能原因**：
1. 函数定义在动态生成的代码中
2. 缓存目录权限问题
3. 缓存 Key 冲突或变化

**排查步骤**：

```python
# 1. 开启缓存日志
import os
os.environ["TILELANG_LOG_LEVEL"] = "DEBUG"

# 2. 检查缓存目录
import os
cache_dir = os.path.expanduser("~/.cache/tilelang")
print(f"缓存目录: {cache_dir}")
print(f"缓存目录存在: {os.path.exists(cache_dir)}")
print(f"缓存目录大小: {sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(cache_dir) for f in fn) / 1024**2:.1f} MB")

# 3. 检查函数定义是否稳定
import inspect
source = inspect.getsource(my_kernel)
print(f"函数源码 hash: {hash(source)}")

# 4. 强制刷新缓存
os.environ["TILELANG_DISABLE_CACHE"] = "1"
# 执行一次后恢复
os.environ["TILELANG_DISABLE_CACHE"] = "0"

# 5. 手动清除缓存
import shutil
shutil.rmtree(cache_dir, ignore_errors=True)
```

### 10.3 内存溢出（OOM）

**症状**：编译或运行时出现 CUDA OOM 错误。

**可能原因**：
1. Shared memory 分配过大
2. 寄存器使用过多
3. Grid/Block 配置不当

**排查步骤**：

```python
# 1. 检查 GPU 内存
import torch
print(f"GPU 总内存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
print(f"已用内存: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
print(f"缓存内存: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")

# 2. 减小 tile 大小以降低内存需求
@tilelang.jit
def memory_efficient_kernel(A, B):
    # 使用更小的 tile
    TILE_M = 64   # 从 128 减小
    TILE_N = 64
    TILE_K = 16   # 从 32 减小
    
    @T.prim_func
    def main(...):
        # 更小的 shared memory 分配
        A_shared = T.alloc_shared((TILE_M, TILE_K), "float16")  # 2KB
        B_shared = T.alloc_shared((TILE_K, TILE_N), "float16")  # 2KB
        ...
    
    return main

# 3. 使用 shared memory 大小限制
@tilelang.jit(max_smem=49152)  # 限制为 48KB
def constrained_kernel(A, B):
    ...
```

### 10.4 编译错误诊断

**常见编译错误及解决方案**：

```python
# 错误 1: 类型不匹配
"""
Error: dtype mismatch, expected float16, got float32
"""
# 解决：确保输入 tensor 的 dtype 与 kernel 定义一致
A = A.to(torch.float16)

# 错误 2: Shape 不匹配
"""
Error: shape mismatch, expected (1024, 1024), got (512, 512)
"""
# 解决：检查输入 tensor 的 shape 是否符合 kernel 要求

# 错误 3: 超出硬件限制
"""
Error: requested shared memory (65536) exceeds limit (49152)
"""
# 解决：减小 tile 大小或使用 opt-in shared memory
@tilelang.jit(max_smem=166912)  # 使用扩展 shared memory
def large_smem_kernel(A, B):
    ...

# 错误 4: 不支持的操作
"""
Error: unsupported operation: xxx
"""
# 解决：查看 TileLang 支持的操作列表，或使用原生 CUDA 实现

# 通用调试流程
def debug_compilation_error(kernel_func, inputs):
    """调试编译错误的通用流程"""
    import traceback
    
    try:
        kernel_func(*inputs)
    except Exception as e:
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print(f"\n完整堆栈:")
        traceback.print_exc()
        
        # 尝试获取 IR dump
        try:
            os.environ["TILELANG_DUMP_IR"] = "1"
            kernel_func(*inputs)
        except:
            pass
        finally:
            os.environ["TILELANG_DUMP_IR"] = "0"
```

---

## 11. 练习题与思考题

### 练习 1：缓存命中率分析

设计一个实验，测量不同场景下的缓存命中率：
- 场景 A：相同函数、相同 shape
- 场景 B：相同函数、不同 shape
- 场景 C：不同函数、相同 shape

```python
# 你的代码
def measure_cache_hit_rate():
    pass
```

### 练习 2：Launch 配置优化

给定以下 kernel，手动分析最优的 Launch 配置并与自动推导结果对比：

```python
@tilelang.jit
def exercise_kernel(A, B):
    M, N = A.shape
    
    @T.prim_func
    def main(A: T.Tensor((M, N), "float16"), B: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, 64), T.ceildiv(M, 64), threads=64) as (bx, by):
            A_local = T.alloc_fragment((64, 64), "float16")
            B_local = T.alloc_fragment((64, 64), "float16")
            
            T.copy(A[by * 64, bx * 64], A_local)
            T.copy(B[by * 64, bx * 64], B_local)
            
            for i in T.serial(64):
                for j in T.serial(64):
                    A_local[i, j] = A_local[i, j] + B_local[i, j]
            
            T.copy(A_local, B[by * 64, bx * 64])
    
    return main
```

### 练习 3：PyTorch 集成

实现一个 TileLang 版本的 LayerNorm 并集成到 PyTorch 的 `torch.autograd.Function` 中：

```python
class TileLangLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        # 你的实现
        pass
    
    @staticmethod
    def backward(ctx, grad_output):
        # 你的实现
        pass
```

### 练习 4：缓存失效策略

设计一个更智能的缓存失效策略，考虑以下因素：
1. 编译时间长的 kernel 应该保留更久
2. 使用频率高的 kernel 应该优先保留
3. 版本变化时应该渐进式失效

```python
class SmartCachePolicy:
    # 你的实现
    pass
```

### 练习 5：多目标编译

编写一个函数，自动检测当前环境的硬件并选择最优的编译目标：

```python
def select_optimal_target(kernel_func, input_shapes):
    """
    检测硬件 → 选择目标 → 编译 kernel → 返回编译后的 kernel
    """
    # 你的实现
    pass
```

### 思考题

1. **JIT vs AOT**：在什么场景下你会选择 AOT 编译而非 JIT？给出至少 3 个具体场景。

2. **缓存一致性**：如果在多进程环境中使用 TileLang，如何保证缓存的一致性？

3. **编译并行化**：如何设计一个系统，使得多个 kernel 可以并行编译？

4. **动态 Shape 优化**：如何设计一个自适应的 shape 离散化策略，根据实际使用模式动态调整预定义的 shape 档位？

5. **错误恢复**：如果 JIT 编译失败，如何设计一个优雅的 fallback 机制？

---

## 12. 扩展阅读

### 12.1 学术论文

1. **TVM: An Automated End-to-End Optimizing Compiler for Deep Learning** - Chen et al., OSDI 2018
   - TVM 框架的原始论文，TileLang 建立在 TVM 之上
   - 理解 TVM 的 Relay IR 和 TIR 对理解 TileLang 至关重要

2. **Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines** - Ragan-Kelley et al., PLDI 2013
   - Halide 的分离计算与调度的思想影响了 TileLang 的设计

3. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** - Dao et al., NeurIPS 2022
   - Flash Attention 的设计思想可以直接应用到 TileLang kernel 优化中

4. **CUTLASS: Fast Linear Algebra in CUDA C++** - NVIDIA
   - NVIDIA 的高性能 CUDA 模板库，理解 CUTLASS 有助于理解 TileLang 的 Tile 操作

### 12.2 官方文档

1. TileLang 官方文档：https://github.com/tile-ai/tilelang
2. TVM 官方文档：https://tvm.apache.org/docs/
3. CUDA Programming Guide：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
4. PyTorch torch.compile 文档：https://pytorch.org/docs/stable/torch.compiler.html

### 12.3 相关工具

1. **NSight Compute**：NVIDIA GPU 性能分析工具
2. **TVM Profiler**：TVM 内置的性能分析工具
3. **PyTorch Profiler**：PyTorch 的性能分析工具

### 12.4 推荐博客和教程

1. **How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance** - Simon Boehm
   - 详细介绍了 CUDA 矩阵乘法优化的所有细节

2. **Making Deep Learning Go Brrrr From First Principles** - Horace He
   - 理解 GPU 计算和内存层次的优秀入门文章

3. **Programming Massively Parallel Processors** - Kirk & Hwu
   - GPU 编程的经典教材

---

## 📖 下一章预告

**Chapter 20: TileLang 性能调优实战**

在下一章中，我们将深入探讨：

1. **性能分析工具链**：如何使用 NSight Compute、TVM Profiler 等工具定位性能瓶颈
2. **内存优化技巧**：Shared memory padding、bank conflict 消除、swizzle 模式
3. **计算优化技巧**：指令调度、循环展开、向量化
4. **实际案例分析**：从零开始优化一个 Transformer Attention kernel 到接近 cuBLAS 的性能

> **预习建议**
> - 安装 NSight Compute 并熟悉其基本使用
> - 复习 GPU 内存层次结构（Global → Shared → Register）
> - 尝试分析一个简单 kernel 的 roofline model

---

*本章内容基于 TileLang 0.1.x 版本编写。随着项目的发展，部分 API 和编译流程可能会有变化，请以官方文档为准。*
