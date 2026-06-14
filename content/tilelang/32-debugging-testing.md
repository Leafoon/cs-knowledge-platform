---
title: "Chapter 32: 调试技术与测试框架"
description: "掌握 TileLang kernel 的调试方法论：IR dump 机制、内存错误定位、数值精度问题排查、单元测试（pytest）、正确性验证（reference implementation）、性能回归检测与 CI/CD 流程"
updated: 2026-06-11
---

# Chapter 32: 调试技术与测试框架

> **Learning Objectives**
>
> 1. 掌握 TileLang 的完整调试方法论与工具链
> 2. 理解 IR dump 机制及其在编译问题排查中的应用
> 3. 掌握使用 cuda-memcheck/compute-sanitizer 定位内存错误
> 4. 学会排查数值精度问题（NaN/Inf/精度损失）
> 5. 掌握基于 pytest 的 TileLang 单元测试框架
> 6. 理解正确性验证方法（Reference Implementation 对比）
> 7. 掌握性能回归检测与 CI/CD 流程设计

---

## 32.1 TileLang 调试方法论

### 32.1.1 调试层次模型

TileLang 的调试可以分为多个层次，每个层次有不同的工具和策略：

```
TileLang 调试层次模型：

┌─────────────────────────────────────────────┐
│  Level 4: 性能调试                           │
│  工具: NCU, nsys, 自定义 Profiler           │
│  关注: 带宽、占用率、指令效率               │
├─────────────────────────────────────────────┤
│  Level 3: 数值调试                           │
│  工具: 数值对比、精度分析、边界检查          │
│  关注: NaN/Inf、精度损失、数值稳定性         │
├─────────────────────────────────────────────┤
│  Level 2: 内存调试                           │
│  工具: compute-sanitizer, cuda-memcheck      │
│  关注: 越界访问、未初始化、泄漏             │
├─────────────────────────────────────────────┤
│  Level 1: 编译调试                           │
│  工具: IR dump, 编译器日志                   │
│  关注: 编译错误、IR 转换、代码生成           │
└─────────────────────────────────────────────┘
```

这个四层调试层次模型是 TileLang 开发中最核心的调试方法论。它将调试过程从底层到高层划分为四个递进层次，每个层次对应不同类型的错误和专用工具。Level 1 编译调试是最基础的层次，当 TileLang 的 Python DSL 代码无法正确转换为 TensorIR 或目标平台代码时，需要通过 IR dump 机制来检查中间表示的正确性。Level 2 内存调试处理 GPU 编程中最棘手的内存错误问题，包括越界访问、未初始化内存读取和内存泄漏等。Level 3 数值调试关注计算结果的正确性，特别是在混合精度计算中常见的 NaN、Inf 和精度损失问题。Level 4 性能调试是最高层次，使用 NVIDIA NCU、Nsight Systems 等专业工具分析 kernel 的执行效率。这种分层方法论的设计动机是帮助开发者快速定位问题类型，避免在错误的层次上浪费时间调试。例如，当遇到运行时崩溃时，应该首先检查内存错误（Level 2），而不是去分析性能指标。在实际应用中，建议开发者按照从低到高的顺序逐层排查问题，因为底层问题往往是高层问题的根因。

<div data-component="DebuggingWorkflowDiagram"></div>

### 32.1.2 系统化调试流程

```
TileLang 系统化调试流程：

1. 问题分类
   ├── 编译失败 → Level 1: IR 调试
   ├── 运行时崩溃 → Level 2: 内存调试
   ├── 结果错误 → Level 3: 数值调试
   └── 性能不佳 → Level 4: 性能调试

2. 最小化复现
   ├── 缩小输入规模
   ├── 固定随机种子
   └── 隔离问题算子

3. 工具辅助定位
   ├── 编译问题: tilelang.dump_ir()
   ├── 内存问题: compute-sanitizer
   ├── 数值问题: torch.allclose + 误差分析
   └── 性能问题: ncu --set full

4. 修复与验证
   ├── 修改代码
   ├── 运行单元测试
   └── 性能回归测试
```

系统化调试流程将调试过程分解为四个标准步骤，形成一个可重复、可预测的问题解决框架。第一步问题分类是关键的起点，通过错误现象快速判断问题属于哪个调试层次，从而选择正确的工具和方法。这种分类能力是经验积累的结果，但通过这个框架可以加速学习过程。第二步最小化复现是调试的核心技巧，通过缩小输入规模来减少问题的复杂度，固定随机种子确保问题可重复，隔离问题算子排除其他组件的干扰。这三个子步骤的设计动机是将复杂问题简化为可管理的小问题。第三步工具辅助定位利用 TileLang 提供的专用调试工具，如 `tilelang.dump_ir()` 用于编译问题、`compute-sanitizer` 用于内存问题、`torch.allclose` 用于数值问题。这些工具的选择是基于多年的 GPU 编程经验总结出来的。第四步修复与验证确保问题被正确解决且不引入新问题。整个流程的关键在于不要跳过任何步骤，特别是最小化复现这一步，很多开发者倾向于直接修复而不复现问题，这往往导致修复不彻底或引入回归问题。

---

## 32.2 IR Dump 机制与使用

### 32.2.1 TileLang IR 层次结构

TileLang 的编译管线生成多个层次的 IR，理解这些 IR 对于调试编译问题至关重要：

```
TileLang IR 层次：

Python DSL (用户代码)
    │
    ▼
Tile IR (TileLang 前端)
    │  - T.grid, T.Parallel, T.Pipelined
    │  - T.alloc_shared, T.alloc_fragment
    │  - T.gemm, T.copy, T.reduce
    │
    ▼
TensorIR (TVM 中间表示)
    │  - Block, For, Buffer
    │  - IterVar, Region
    │  - 完整的类型信息
    │
    ▼
Low-level IR (优化后)
    │  - 循环展开
    │  - 向量化
    │  - 内存布局优化
    │
    ▼
Target IR (目标平台)
    ├── PTX (NVIDIA)
    ├── HSACO (AMD)
    └── Ascend C (华为)
```

这个 IR 层次结构图展示了 TileLang 代码从用户编写的 Python DSL 到最终目标平台代码的完整转换路径。理解这个层次结构对于调试编译问题至关重要，因为它帮助开发者确定问题发生在哪个编译阶段。Python DSL 是用户直接编写的 TileLang 代码，使用 `T.grid`、`T.Parallel`、`T.Pipelined` 等高级抽象。Tile IR 是 TileLang 前端编译器生成的中间表示，保留了用户代码的语义信息但已经进行了初步的类型检查和形状推断。TensorIR 是 TVM 框架的核心中间表示，它将 Tile IR 转换为更底层的表示，包含完整的类型信息和内存布局细节。Low-level IR 经过了各种优化 pass，包括循环展开、向量化和内存布局优化。Target IR 是最终生成的平台特定代码，如 NVIDIA 的 PTX、AMD 的 HSACO 或华为的 Ascend C。这个层次结构的设计动机是实现编译器的模块化，每一层只关注特定的转换任务，便于调试和优化。在实际调试中，建议从 Tile IR 开始检查，因为它是离用户代码最近的中间表示，最容易定位问题。

<div data-component="IRDumpAnalyzer"></div>

### 32.2.2 使用 IR Dump 调试

```python
import tilelang
from tilelang import T

# 方法 1: 全局 IR dump
tilelang.dump_ir(enabled=True, output_dir="./ir_dump")

@T.prim_func
def my_kernel(
    A: T.Tensor([M, K], "float16"),
    B: T.Tensor([K, N], "float16"),
    C: T.Tensor([M, N], "float16"),
):
    # ... 算子实现
    pass

# 编译时会自动 dump IR 到 ./ir_dump/ 目录
kernel = tilelang.compile(my_kernel, target="cuda")

# 方法 2: 单次 dump
kernel = tilelang.compile(my_kernel, target="cuda")
ir = kernel.get_source("tilelang")  # 获取 Tile IR
print(ir)

# 方法 3: 获取不同层次的 IR
ptx = kernel.get_source("ptx")      # 获取 PTX
tensorir = kernel.get_source("tir") # 获取 TensorIR
```

这段代码展示了三种不同的 IR dump 方法，每种方法适用于不同的调试场景。方法一使用 `tilelang.dump_ir()` 开启全局 IR dump，这会在编译过程中自动将所有层次的 IR 输出到指定目录，适合需要全面分析编译过程的场景。`enabled=True` 启用 dump 功能，`output_dir` 指定输出目录路径。方法二使用 `kernel.get_source("tilelang")` 获取单个 kernel 的 Tile IR 表示，适合快速检查特定 kernel 的编译结果。方法三展示如何获取不同层次的 IR，通过传递不同的参数（"tilelang"、"ptx"、"tir"）来获取对应层次的表示。`get_source` 方法的参数设计体现了 TileLang 的多层 IR 架构，开发者可以根据需要选择查看哪个层次。这些方法的设计动机是提供灵活的调试入口，让开发者能够根据问题类型选择最合适的工具。在实际使用中，建议先使用方法二快速检查，如果问题复杂再使用方法一进行全面分析。

### 32.2.3 IR Dump 输出示例

**Tile IR 示例：**

```python
# Tile IR (用户编写)
@T.prim_func
def gemm_kernel(
    A: T.Tensor([1024, 1024], "float16"),
    B: T.Tensor([1024, 1024], "float16"),
    C: T.Tensor([1024, 1024], "float16"),
):
    for bx, by in T.grid(8, 8):
        A_smem = T.alloc_shared([128, 128], "float16")
        B_smem = T.alloc_shared([128, 128], "float16")
        C_frag = T.alloc_fragment([128, 128], "float32")
        # ...

上面的 Tile IR 示例展示了一个典型的 GEMM kernel 在 TileLang 前端编译后的中间表示形式。可以看到，用户使用 `T.grid`、`T.alloc_shared`、`T.alloc_fragment` 等高级 API 编写的代码，经过 TileLang 前端编译后，仍然保留了清晰的语义结构。Tile IR 是离用户代码最近的中间表示，开发者可以直观地看到循环嵌套结构、Shared Memory 分配和 Fragment 分配是否与预期一致。通过对比原始代码和 Tile IR，可以快速发现由于 API 使用不当导致的问题，例如错误地使用了 `T.serial` 而非 `T.Pipelined`，或者 Shared Memory 分配大小与实际访问范围不匹配。因此，Tile IR 是编译调试的第一道关口，建议开发者在遇到任何编译问题时首先导出并检查 Tile IR。

**TensorIR 示例（编译器转换后）：**

```python
# TensorIR (编译器输出)
@T.prim_func
def tir_gemm(
    A: T.handle,
    B: T.handle,
    C: T.handle,
):
    A_buf = T.match_buffer(A, [1024, 1024], "float16")
    B_buf = T.match_buffer(B, [1024, 1024], "float16")
    C_buf = T.match_buffer(C, [1024, 1024], "float16")

    for blockIdx.x in T.thread_binding(8, "blockIdx.x"):
        for blockIdx.y in T.thread_binding(8, "blockIdx.y"):
            # Shared Memory 分配
            A_smem = T.allocate([128, 128], "float16", "shared")
            B_smem = T.allocate([128, 128], "float16", "shared")
            # Register 分配
            C_frag = T.allocate([128, 128], "float32", "local")

            for k in T.serial(8):
                # 数据加载
                for i, j in T.Parallel(128, 128):
                    A_smem[i, j] = A_buf[blockIdx.y * 128 + i, k * 128 + j]
                # ... 更多操作

TensorIR 是 TVM 框架的核心中间表示，它将 Tile IR 中的高级抽象（如 `T.grid`、`T.Parallel`）转换为底层的循环和内存操作。在这个示例中，`T.grid(8, 8)` 被转换为显式的 `blockIdx.x` 和 `blockIdx.y` 绑定，Shared Memory 通过 `T.allocate` 明确标注为 `"shared"` 作用域，寄存器变量则标注为 `"local"`。观察 TensorIR 可以帮助开发者确认编译器是否正确处理了线程绑定、内存作用域和循环嵌套顺序。如果发现 `T.Parallel` 被转换为 `T.serial`，说明并行化失败；如果 Shared Memory 的 scope 标注错误，说明内存分配策略有问题。因此，TensorIR 是连接高层语义与底层实现的桥梁，在调试编译器优化和代码生成问题时不可或缺。

### 32.2.4 IR Dump 分析技巧

| IR 特征 | 可能问题 | 解决方案 |
|---------|---------|---------|
| 循环未展开 | 性能不佳 | 添加 T.unroll 注解 |
| 冗余拷贝 | 内存浪费 | 优化数据布局 |
| 同步点过多 | 性能下降 | 减少 Barrier |
| 未向量化 | 带宽浪费 | 使用 T.vectorize |
| 寄存器溢出 | 性能下降 | 减小 Tile 大小 |

```python
# 分析 IR 中的同步点
def analyze_sync_points(ir_source):
    """分析 IR 中的同步点数量"""
    import re
    sync_count = len(re.findall(r'__syncthreads|T\.sync|barrier', ir_source))
    return sync_count

# 分析寄存器使用
def analyze_register_usage(ir_source):
    """分析 IR 中的寄存器使用"""
    import re
    # 查找 local buffer 分配
    local_allocs = re.findall(r'T\.allocate\(\[(\d+),?\s*(\d+)?\]', ir_source)
    total_registers = sum(int(a[0]) * int(a[1] if a[1] else 1) for a in local_allocs)
    return total_registers
```

这两个 IR 分析函数展示了如何通过正则表达式解析 TileLang 生成的 IR 代码来提取关键性能指标。`analyze_sync_points` 函数通过匹配 `__syncthreads`、`T.sync` 和 `barrier` 等关键字来统计同步点数量，同步点过多会导致性能下降，因为 GPU 线程需要等待同步完成才能继续执行。`analyze_register_usage` 函数分析 local buffer 分配来估算寄存器使用量，`T\.allocate` 模式匹配分配操作，提取分配的维度信息并计算总寄存器数。这些分析函数的设计动机是提供自动化的 IR 分析能力，避免手动检查大量 IR 代码。在实际调试中，同步点数量可以作为性能瓶颈的早期指标，而寄存器使用量则与 GPU 占用率直接相关。这些工具可以帮助开发者快速识别性能问题，特别是在优化 Tile 大小和流水线深度时。

---

## 32.3 内存错误定位

### 32.3.1 compute-sanitizer 使用

NVIDIA 的 compute-sanitizer（前身为 cuda-memcheck）是定位 GPU 内存错误的核心工具：

```bash
# 基本使用
compute-sanitizer --tool memcheck python my_script.py

# 详细输出
compute-sanitizer --tool memcheck --leak-check full --show-backtrace yes python my_script.py

# Race Condition 检测
compute-sanitizer --tool racecheck python my_script.py

# 同步错误检测
compute-sanitizer --tool synccheck python my_script.py

# 未初始化内存检测
compute-sanitizer --tool initcheck python my_script.py
```

这些命令展示了 NVIDIA compute-sanitizer 工具的四种核心检测模式，每种模式针对不同类型的内存错误。`--tool memcheck` 是最常用的模式，检测越界访问、内存泄漏和非法内存操作，`--leak-check full` 详细报告内存泄漏，`--show-backtrace yes` 显示完整的调用栈。`--tool racecheck` 专门检测共享内存的竞态条件，这在 GPU 编程中是常见但难以发现的错误，当多个线程同时读写同一内存位置且没有适当的同步时就会发生。`--tool synccheck` 检测同步错误，如在错误的位置使用同步屏障或缺少必要的同步。`--tool initcheck` 检测未初始化内存的使用，这可能导致不可预测的结果。这些工具的设计动机是为 GPU 编程提供全面的内存安全检查，帮助开发者在开发早期发现潜在问题。在实际使用中，建议在开发阶段频繁使用这些工具，特别是在修改内存相关代码后。

### 32.3.2 常见内存错误模式

<div data-component="ErrorPatternCatalog"></div>

**1. 越界访问 (Out of Bounds)**

```
错误信息：
========= Invalid __global__ read of size 2 bytes
=========     at 0x00000148 in gemm_kernel
=========     by thread (128,0,0) in block (0,0,0)
=========     Address 0x7f1234567890 is out of bounds

原因：线程访问了超出分配范围的内存地址
```

```python
# 错误示例: 边界检查缺失
@T.prim_func
def buggy_kernel(
    A: T.Tensor([1024, 1024], "float16"),
    Output: T.Tensor([1024, 1024], "float16"),
):
    for i, j in T.Parallel(1024, 1024):
        # 当 i >= 1024 或 j >= 1024 时越界
        Output[i, j] = A[i + 1, j + 1]  # 越界!

# 修复: 添加边界检查
@T.prim_func
def fixed_kernel(
    A: T.Tensor([1024, 1024], "float16"),
    Output: T.Tensor([1024, 1024], "float16"),
):
    for i, j in T.Parallel(1024, 1024):
        if i + 1 < 1024 and j + 1 < 1024:
            Output[i, j] = A[i + 1, j + 1]
        else:
            Output[i, j] = 0.0
```

这个例子展示了 GPU 编程中最常见的内存错误之一：越界访问。`buggy_kernel` 函数在 `T.Parallel(1024, 1024)` 循环中访问 `A[i+1, j+1]`，但当 `i` 或 `j` 等于 1023 时，索引会超出 1024 的边界，导致非法内存访问。这种错误在 GPU 上特别危险，因为单个线程的越界访问可能导致整个 kernel 崩溃或产生未定义行为。`fixed_kernel` 通过添加边界检查来修复这个问题，使用 `if` 语句确保索引在有效范围内，对于越界的情况设置默认值 0.0。这种修复方法虽然简单，但会引入分支，可能影响性能。在 TileLang 中，更高效的做法是使用 `T.where` 或调整循环范围来避免分支。这个例子的教训是：在编写 GPU kernel 时，必须仔细考虑边界条件，特别是在处理非整除的矩阵维度时。实际应用中，建议使用 TileLang 提供的边界检查宏或工具来自动处理这些情况。

**2. Shared Memory 越界**

```
错误信息：
========= Invalid __shared__ read of size 16 bytes
=========     at 0x00000200 in compute_kernel
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x00007fff12345678 is out of bounds

原因：Shared Memory 分配不足或索引计算错误
```

```python
# 错误示例: Shared Memory 大小不匹配
@T.prim_func
def smem_buggy(
    A: T.Tensor([1024], "float16"),
):
    smem = T.alloc_shared([64], "float16")  # 只分配 64
    for i in T.Parallel(128):
        smem[i] = A[i]  # 访问 0-127，越界!

# 修复: 确保分配足够
@T.prim_func
def smem_fixed(
    A: T.Tensor([1024], "float16"),
):
    smem = T.alloc_shared([128], "float16")  # 分配 128
    for i in T.Parallel(128):
        smem[i] = A[i]
```

Shared Memory 大小不匹配是 GPU 编程中另一类常见的内存错误。`smem_buggy` 函数分配了 64 个元素的 Shared Memory，但循环访问 0-127 的索引，导致越界访问。Shared Memory 的越界访问比 Global Memory 更危险，因为 Shared Memory 是每个线程块私有的，错误可能影响整个线程块的执行。`smem_fixed` 通过将分配大小改为 128 来修复这个问题，确保分配的大小与访问范围匹配。这个错误的常见原因是开发者在修改循环范围时忘记更新 Shared Memory 分配大小，或者在使用动态形状时计算错误。在 TileLang 中，建议使用表达式而非硬编码数字来定义 Shared Memory 大小，例如 `T.alloc_shared([BLOCK_SIZE], "float16")`，这样当 `BLOCK_SIZE` 变化时，分配大小会自动更新。这个例子强调了代码一致性的重要性：当修改一处代码时，必须检查所有相关的地方是否需要同步修改。

**3. 未初始化内存**

```
错误信息：
========= Uninitialized memory read of size 2 bytes
=========     at 0x00000300 in output_kernel
=========     by thread (0,0,0) in block (0,0,0)

原因：使用了未初始化的 Shared Memory 或 Register
```

```python
# 错误示例: 累加未初始化
@T.prim_func
def uninitialized_buggy(
    A: T.Tensor([M, K], "float16"),
    B: T.Tensor([K, N], "float16"),
    C: T.Tensor([M, N], "float16"),
):
    C_frag = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")
    # 未初始化!
    for k in T.serial(T.ceildiv(K, BLOCK_K)):
        T.gemm(A_frag, B_frag, C_frag)  # 累加到未初始化的值
    T.copy(C_frag, C)

# 修复: 初始化为零
@T.prim_func
def uninitialized_fixed(
    A: T.Tensor([M, K], "float16"),
    B: T.Tensor([K, N], "float16"),
    C: T.Tensor([M, N], "float16"),
):
    C_frag = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")
    T.clear(C_frag)  # 初始化为零
    for k in T.serial(T.ceildiv(K, BLOCK_K)):
        T.gemm(A_frag, B_frag, C_frag)
    T.copy(C_frag, C)
```

未初始化内存是 GPU 编程中最隐蔽的错误之一，因为它不会导致立即崩溃，但会产生随机的错误结果。`uninitialized_buggy` 函数分配了 `C_frag` 累加器但没有初始化它，直接进行矩阵乘法累加，这会导致结果包含随机垃圾值。`uninitialized_fixed` 通过 `T.clear(C_frag)` 将累加器初始化为零来修复这个问题。在 GPU 编程中，Shared Memory 和寄存器默认不初始化，这是性能考虑的结果，因为初始化操作会带来额外开销。然而，对于累加器这类需要从零开始累加的数据结构，初始化是必须的。这个错误的常见场景包括：GEMM kernel 的输出累加器、Softmax 的分母累加器、以及任何需要归约操作的中间结果。在 TileLang 中，`T.clear` 是推荐的初始化方式，它会生成高效的清零指令。实际应用中，建议在分配累加器后立即调用 `T.clear`，养成良好的编程习惯。

### 32.3.3 内存泄漏检测

```python
# 使用 PyTorch 的内存统计检测泄漏
import torch

def check_memory_leak(func, iterations=100):
    """检测内存泄漏"""
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()

    for _ in range(iterations):
        func()

    final_memory = torch.cuda.memory_allocated()
    leaked = final_memory - initial_memory

    if leaked > 0:
        print(f"⚠️ 检测到内存泄漏: {leaked / 1024 / 1024:.2f} MB")
    else:
        print("✅ 无内存泄漏")

    return leaked

# 使用
def my_kernel_operation():
    A = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    B = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    C = tilelang_kernel(A, B)
    return C

check_memory_leak(my_kernel_operation)
```

这段代码实现了一个简洁有效的 GPU 内存泄漏检测工具。`check_memory_leak` 函数通过多次执行目标函数并比较前后内存使用量来检测泄漏。`torch.cuda.reset_peak_memory_stats()` 重置峰值内存统计，确保测量准确。`torch.cuda.memory_allocated()` 返回当前已分配的 GPU 内存大小，这是检测泄漏的关键指标。函数执行指定次数后，比较最终内存与初始内存的差异，如果最终内存大于初始内存，则说明存在泄漏。`my_kernel_operation` 是被检测的函数示例，它创建张量并调用 TileLang kernel。这种检测方法的设计动机是提供一种简单可靠的内存泄漏检测机制，不需要复杂的工具支持。在实际应用中，建议将此函数集成到单元测试中，定期检查代码的内存安全性。需要注意的是，这种方法只能检测明显的内存泄漏，对于小量泄漏可能需要增加迭代次数来提高检测灵敏度。

---

## 32.4 数值精度问题排查

### 32.4.1 NaN/Inf 问题诊断

```python
def diagnose_nan_inf(output, name="output"):
    """诊断 NaN/Inf 问题"""
    if torch.isnan(output).any():
        nan_count = torch.isnan(output).sum().item()
        nan_ratio = nan_count / output.numel()
        print(f"❌ {name} 包含 {nan_count} 个 NaN ({nan_ratio:.2%})")

        # 定位第一个 NaN
        nan_indices = torch.where(torch.isnan(output))
        first_nan = tuple(idx[0].item() for idx in nan_indices)
        print(f"   第一个 NaN 位置: {first_nan}")
        return False

    if torch.isinf(output).any():
        inf_count = torch.isinf(output).sum().item()
        print(f"❌ {name} 包含 {inf_count} 个 Inf")
        return False

    print(f"✅ {name} 数值正常")
    return True
```

NaN/Inf 诊断函数是数值调试的核心工具，用于快速定位数值异常的根源。`torch.isnan(output).any()` 检查输出中是否存在 NaN 值，这是数值计算中最常见的错误之一。函数不仅检测是否存在 NaN，还统计 NaN 的数量和比例，帮助判断问题的严重程度。更重要的是，它定位第一个 NaN 的位置，这对于调试非常有用，因为 NaN 通常是从某个特定位置开始传播的。`torch.isinf(output).any()` 检查 Inf 值，Inf 通常由除零或溢出引起。这个函数的设计动机是提供快速的数值异常检测，帮助开发者在调试早期发现问题。在 TileLang kernel 中，NaN/Inf 的常见原因包括：Softmax 中的指数溢出、FP16 累加器溢出、除零操作、以及未初始化的累加器。实际应用中，建议在 kernel 的每个关键步骤后调用此函数，特别是在数值敏感的操作（如 Softmax、LayerNorm）之后。

### 32.4.2 精度对比验证

```python
import torch

def compare_precision(tilelang_output, reference_output, dtype="float16", rtol=1e-3, atol=1e-3):
    """对比 TileLang 输出与参考实现的精度"""
    # 转换为相同类型
    tl = tilelang_output.float()
    ref = reference_output.float()

    # 计算误差
    abs_diff = torch.abs(tl - ref)
    rel_diff = abs_diff / (torch.abs(ref) + 1e-8)

    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    mean_rel_diff = rel_diff.mean().item()

    print(f"精度对比结果:")
    print(f"  最大绝对误差: {max_abs_diff:.6e}")
    print(f"  最大相对误差: {max_rel_diff:.6e}")
    print(f"  平均绝对误差: {mean_abs_diff:.6e}")
    print(f"  平均相对误差: {mean_rel_diff:.6e}")

    # 检查是否通过
    is_close = torch.allclose(tl, ref, rtol=rtol, atol=atol)
    print(f"  allclose (rtol={rtol}, atol={atol}): {'✅ 通过' if is_close else '❌ 失败'}")

    # 找到误差最大的位置
    max_diff_idx = torch.argmax(abs_diff)
    max_diff_pos = torch.unravel_index(max_diff_idx, tl.shape)
    print(f"  最大误差位置: {max_diff_pos}")
    print(f"    TileLang: {tl[max_diff_pos].item():.6f}")
    print(f"    Reference: {ref[max_diff_pos].item():.6f}")

    return {
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_abs_diff": mean_abs_diff,
        "mean_rel_diff": mean_rel_diff,
        "passed": is_close,
    }
```

精度对比验证函数是数值精度调试的核心工具，提供全面的误差分析能力。函数首先将 TileLang 输出和参考输出都转换为 FP32 类型，确保比较的公平性，避免因精度转换引入额外误差。然后计算绝对误差和相对误差，`1e-8` 的 epsilon 避免除零错误。函数输出四个关键指标：最大绝对误差、最大相对误差、平均绝对误差和平均相对误差，这些指标从不同角度反映数值精度。`torch.allclose` 是标准的精度检查函数，`rtol` 控制相对误差容限，`atol` 控制绝对误差容限。函数还定位误差最大的位置，帮助开发者精确定位问题。这个函数的设计动机是提供系统化的精度验证方法，替代简单的 `torch.allclose` 检查，因为它提供了更丰富的诊断信息。在 TileLang 开发中，建议将此函数集成到单元测试中，定期验证 kernel 的数值精度。特别是对于混合精度 kernel，需要仔细设置容限参数，通常 FP16 kernel 使用 `rtol=1e-3, atol=1e-3`。

### 32.4.3 常见精度问题模式

| 问题 | 表现 | 原因 | 解决方案 |
|------|------|------|---------|
| FP16 溢出 | 大值变 Inf | 中间结果超出 FP16 范围 | 使用 FP32 累加器 |
| Softmax NaN | 注意力输出 NaN | 指数运算溢出 | 使用 Online Softmax |
| 梯度消失 | 训练不收敛 | 链式乘法累积误差 | 使用混合精度 |
| 累加误差 | 结果偏差 | 大量 FP16 加法 | 使用 FP32 累加 |
| 除零错误 | 结果 Inf/NaN | 分母为零 | 添加 epsilon |

### 32.4.4 数值稳定性最佳实践

```python
# 1. 使用 FP32 累加器
@T.prim_func
def stable_accumulation(
    A: T.Tensor([M, K], "float16"),
    B: T.Tensor([K, N], "float16"),
    C: T.Tensor([M, N], "float16"),
):
    C_frag = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")  # FP32 累加
    T.clear(C_frag)
    for k in T.serial(T.ceildiv(K, BLOCK_K)):
        T.gemm(A_frag, B_frag, C_frag)
    # 转换回 FP16 输出
    for i, j in T.Parallel(BLOCK_M, BLOCK_N):
        C[i, j] = T.cast(C_frag[i, j], "float16")

# 2. Online Softmax 避免溢出
def online_softmax(scores):
    """数值稳定的 Softmax"""
    m = scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - m)
    sum_exp = exp_scores.sum(dim=-1, keepdim=True)
    return exp_scores / sum_exp

# 3. 添加 epsilon 避免除零
def safe_divide(a, b, eps=1e-8):
    return a / (b + eps)
```

这三个函数展示了数值稳定性的三种核心策略。第一个函数 `stable_accumulation` 使用 FP32 累加器来避免 FP16 累加误差，这是混合精度计算的最佳实践。FP16 的精度有限，在大量累加操作中会累积显著误差，而 FP32 累加可以保持足够的精度。`T.cast` 在最后将结果转换回 FP16，平衡了精度和内存。第二个函数 `online_softmax` 实现了数值稳定的 Softmax，通过减去最大值来避免指数溢出。这是 Softmax 的标准实现方法，`m` 是每行的最大值，`exp_scores - m` 确保指数参数非正，避免溢出。第三个函数 `safe_divide` 通过添加 epsilon 避免除零错误，这是数值计算中的常见技巧。这些函数的设计动机是提供可复用的数值稳定性工具，减少常见错误的发生。在 TileLang kernel 中，这些策略被广泛应用于 FlashAttention、Softmax、LayerNorm 等数值敏感的操作中。实际应用中，建议将这些函数封装为工具库，在开发中优先使用。

---

## 32.5 单元测试框架

### 32.5.1 pytest 测试框架设计

<div data-component="TestingFrameworkArchitecture"></div>

```python
# test_tilelang_gemm.py
import pytest
import torch
import tilelang
from tilelang import T

# 测试配置
TEST_CONFIGS = [
    {"M": 128, "N": 128, "K": 128},
    {"M": 256, "N": 256, "K": 256},
    {"M": 1024, "N": 1024, "K": 1024},
    {"M": 4096, "N": 4096, "K": 4096},
    {"M": 1, "N": 4096, "K": 4096},    # 向量-矩阵
    {"M": 4096, "N": 1, "K": 4096},    # 矩阵-向量
    {"M": 1, "N": 1, "K": 4096},       # 向量-向量
]

@pytest.fixture(params=TEST_CONFIGS)
def gemm_config(request):
    return request.param

@pytest.fixture
def gemm_inputs(gemm_config):
    M, N, K = gemm_config["M"], gemm_config["N"], gemm_config["K"]
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    return A, B

# 参考实现
def reference_gemm(A, B):
    return torch.matmul(A, B)

# TileLang 算子
@T.prim_func
def tilelang_gemm(
    A: T.Tensor([None, None], "float16"),
    B: T.Tensor([None, None], "float16"),
    C: T.Tensor([None, None], "float16"),
):
    M, N, K = T.int32(), T.int32(), T.int32()
    for i, j in T.grid(M, N):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = T.float16(0)
            for k in range(K):
                C[vi, vj] += A[vi, k] * B[k, vj]

# 测试用例
class TestGEMM:
    def test_correctness(self, gemm_inputs, gemm_config):
        """测试 GEMM 正确性"""
        A, B = gemm_inputs
        M, N, K = gemm_config["M"], gemm_config["N"], gemm_config["K"]

        # TileLang 结果
        kernel = tilelang.compile(
            tilelang_gemm,
            target="cuda",
            out_idx=[2],
            pass_configs={"dump_ir": False}
        )
        tilelang_output = kernel(A, B)

        # 参考结果
        ref_output = reference_gemm(A, B)

        # 对比
        assert torch.allclose(tilelang_output, ref_output, rtol=1e-3, atol=1e-3), \
            f"GEMM({M}x{N}x{K}) 精度不达标"

    def test_boundary_conditions(self):
        """测试边界条件"""
        # 空矩阵
        A = torch.zeros(0, 1024, device="cuda", dtype=torch.float16)
        B = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        # 应该不崩溃
        C = tilelang_gemm(A, B)
        assert C.shape == (0, 1024)

    def test_numerical_stability(self):
        """测试数值稳定性"""
        # 极端值输入
        A = torch.full((128, 128), 65504.0, device="cuda", dtype=torch.float16)
        B = torch.full((128, 128), 0.001, device="cuda", dtype=torch.float16)
        C = tilelang_gemm(A, B)
        assert not torch.isnan(C).any(), "出现 NaN"
        assert not torch.isinf(C).any(), "出现 Inf"

    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    def test_dtype_support(self, dtype):
        """测试不同数据类型支持"""
        A = torch.randn(256, 256, device="cuda", dtype=getattr(torch, dtype))
        B = torch.randn(256, 256, device="cuda", dtype=getattr(torch, dtype))
        C = tilelang_gemm(A, B)
        assert C.dtype == getattr(torch, dtype)

以上测试框架展示了基于 pytest 的 TileLang 单元测试完整设计模式。`TEST_CONFIGS` 列表定义了多种矩阵维度的测试配置，覆盖从小型矩阵（128×128）到大型矩阵（4096×4096）以及边界条件（向量-矩阵、矩阵-向量、向量-向量）。`@pytest.fixture(params=TEST_CONFIGS)` 装饰器实现了参数化测试，pytest 会自动为每个配置生成独立的测试用例并在失败时清晰报告具体参数。`TestGEMM` 类组织了四个核心测试：`test_correctness` 验证计算结果的数值精度，使用 `torch.allclose` 与 PyTorch 参考实现对比；`test_boundary_conditions` 覆盖空矩阵等边界情况；`test_numerical_stability` 使用 FP16 的极限值（65504）测试溢出行为；`test_dtype_support` 参数化验证不同数据类型。这种测试架构的设计动机是确保 TileLang kernel 在发布前通过全面的自动化验证，任何代码变更都能被快速捕获并定位问题。

### 32.5.2 性能测试

```python
# test_performance.py
import pytest
import torch
import tilelang
import time

class TestPerformance:
    def test_gemm_bandwidth(self):
        """测试 GEMM 内存带宽"""
        M, N, K = 4096, 4096, 4096
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)

        # Warmup
        for _ in range(10):
            tilelang_gemm(A, B)

        # 测量
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            tilelang_gemm(A, B)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        # 计算性能
        flops = 2 * M * N * K * 100
        tflops = flops / elapsed / 1e12

        print(f"GEMM 性能: {tflops:.2f} TFLOPS")
        assert tflops > 800, f"性能过低: {tflops:.2f} TFLOPS"

    def test_performance_regression(self):
        """性能回归检测"""
        baseline_file = "performance_baseline.json"
        import json

        # 测量当前性能
        current_perf = measure_gemm_performance()

        # 加载基线
        try:
            with open(baseline_file) as f:
                baseline = json.load(f)
        except FileNotFoundError:
            # 首次运行，保存基线
            with open(baseline_file, "w") as f:
                json.dump(current_perf, f)
            return

        # 对比
        for key, value in current_perf.items():
            baseline_value = baseline.get(key, value)
            regression = (baseline_value - value) / baseline_value
            assert regression < 0.05, \
                f"性能回归: {key} 从 {baseline_value:.2f} 降到 {value:.2f} ({regression:.1%})"

性能测试是 TileLang 开发流程中确保 kernel 效率不退化的重要环节。`test_gemm_bandwidth` 方法展示了标准的 GPU kernel 性能测量流程：首先进行 10 次预热运行（Warmup）以确保 GPU 进入稳定状态，然后通过 100 次重复测量取平均值来减少噪声。`torch.cuda.synchronize()` 的调用位置至关重要——在计时开始前和结束后都必须同步，否则 GPU 的异步执行会导致测量结果不准确。性能计算公式 `2 * M * N * K` 是矩阵乘法的 FLOPs（浮点运算次数）标准公式，最后通过 TFLOPS 指标衡量 kernel 的计算效率。`test_performance_regression` 方法实现了一个轻量级的性能回归检测机制，将当前性能与存储在 `performance_baseline.json` 中的基线数据对比，当性能下降超过 5% 时触发断言失败。这种方法可以集成到 CI/CD 流程中，确保每次代码提交都不会引入性能退化。

### 32.5.3 集成测试

```python
# test_integration.py
import pytest
import torch

class TestIntegration:
    def test_flash_attention_pipeline(self):
        """测试完整的 FlashAttention 管线"""
        batch, seq_len, n_heads, d = 1, 2048, 32, 128
        Q = torch.randn(batch, seq_len, n_heads, d, device="cuda", dtype=torch.float16)
        K = torch.randn(batch, seq_len, n_heads, d, device="cuda", dtype=torch.float16)
        V = torch.randn(batch, seq_len, n_heads, d, device="cuda", dtype=torch.float16)

        # TileLang 实现
        output = flash_attention_tilelang(Q, K, V)

        # PyTorch 参考
        Q_t = Q.transpose(1, 2)
        K_t = K.transpose(1, 2)
        V_t = V.transpose(1, 2)
        ref = torch.nn.functional.scaled_dot_product_attention(Q_t, K_t, V_t)
        ref = ref.transpose(1, 2)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3)

    def test_moe_layer(self):
        """测试 MoE 层的完整流程"""
        batch, seq, d_model = 1, 128, 4096
        num_experts = 8
        x = torch.randn(batch, seq, d_model, device="cuda", dtype=torch.float16)

        output = moe_layer_tilelang(x, num_experts)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

集成测试验证多个算子组合后的端到端正确性，这与单元测试有本质区别——单元测试关注单个算子的独立行为，而集成测试关注算子之间的数据流交互。`test_flash_attention_pipeline` 测试了完整的 FlashAttention 管线，包括 Q、K、V 矩阵的输入和注意力输出，与 PyTorch 的 `scaled_dot_product_attention` 进行对比。这个测试尤为重要，因为 FlashAttention 涉及 Online Softmax、分块计算和重缩放等复杂数值操作，任何一个环节的精度问题都会在最终输出中累积成显著误差。`test_moe_layer` 测试了 Mixture-of-Experts 层，验证输出形状和数值有效性。集成测试的设计原则是尽可能模拟真实使用场景，使用与生产环境相似的输入规模和数据类型，确保 TileLang kernel 在真实模型中的可靠性。

---

## 32.6 正确性验证方法

### 32.6.1 Reference Implementation 对比

```python
def verify_against_reference(
    tilelang_func,
    reference_func,
    inputs,
    rtol=1e-3,
    atol=1e-3,
    num_trials=100,
):
    """通用的正确性验证框架"""
    passed = 0
    max_error = 0

    for trial in range(num_trials):
        # 生成随机输入
        trial_inputs = [torch.randn_like(inp) for inp in inputs]

        # TileLang 结果
        tl_output = tilelang_func(*trial_inputs)

        # 参考结果
        ref_output = reference_func(*trial_inputs)

        # 对比
        error = torch.abs(tl_output - ref_output).max().item()
        max_error = max(max_error, error)

        if torch.allclose(tl_output, ref_output, rtol=rtol, atol=atol):
            passed += 1

    pass_rate = passed / num_trials
    print(f"验证结果: {passed}/{num_trials} 通过 ({pass_rate:.1%})")
    print(f"最大误差: {max_error:.6e}")

    return pass_rate > 0.99

这个通用的正确性验证框架 `verify_against_reference` 是 TileLang 开发中最核心的验证工具之一。它接受任意两个函数（TileLang 实现和参考实现）以及一组输入模板，在多次随机试验中自动对比两者的输出。每次试验生成新的随机输入，调用两个函数，并比较结果是否在容限范围内。`num_trials=100` 的设计是基于统计学考虑：100 次随机试验足以覆盖大多数数值场景，同时保持合理的执行时间。函数返回 `pass_rate > 0.99` 的布尔值，即要求至少 99% 的试验通过，这为数值精度验证提供了统计置信度。最大误差的跟踪可以帮助开发者了解最坏情况下的精度退化。这种框架的设计动机是将正确性验证自动化、标准化，避免人工检查的遗漏。在开发过程中，每当修改 kernel 代码后，应立即运行此验证确保没有引入数值退化。

### 32.6.2 模糊测试 (Fuzzing)

```python
def fuzz_test_tilelang_kernel(kernel_func, shape_generator, num_tests=1000):
    """模糊测试 TileLang 算子"""
    errors = []

    for i in range(num_tests):
        try:
            # 生成随机形状
            shapes = shape_generator()

            # 生成随机输入
            inputs = [torch.randn(*s, device="cuda", dtype=torch.float16)
                      for s in shapes]

            # 运行算子
            output = kernel_func(*inputs)

            # 检查输出
            if torch.isnan(output).any():
                errors.append((i, "NaN detected", shapes))
            elif torch.isinf(output).any():
                errors.append((i, "Inf detected", shapes))

        except Exception as e:
            errors.append((i, str(e), shapes))

    if errors:
        print(f"❌ 模糊测试发现 {len(errors)} 个错误:")
        for idx, err, shapes in errors[:10]:
            print(f"  Trial {idx}: {err}, shapes={shapes}")
    else:
        print(f"✅ 模糊测试通过 ({num_tests} trials)")

    return errors

模糊测试（Fuzzing）是一种通过生成大量随机输入来发现边界情况和未预期行为的测试方法。`fuzz_test_tilelang_kernel` 函数接受一个 kernel 函数、形状生成器和测试次数，在每次试验中生成随机形状的张量作为输入，执行 kernel 并检查输出中是否出现 NaN 或 Inf。函数捕获所有异常并记录错误信息，包括试验编号、错误类型和对应的输入形状，这使得开发者可以精确复现问题。`num_tests=1000` 的默认值保证了充分的覆盖率。模糊测试在 GPU kernel 开发中尤为重要，因为 GPU 的并行执行特性使得某些错误只在特定线程组合或特定输入形状下才会暴露。常规的固定输入测试往往无法触发这些边界条件，而模糊测试通过随机化输入空间大幅提升问题发现率。建议将此函数集成到 CI/CD 流程的夜间测试中，使用更大的试验次数（如 10000 次）进行全面检查。

---

## 32.7 性能回归检测

### 32.7.1 性能基线管理

```python
import json
import time
from pathlib import Path

class PerformanceBaseline:
    """性能基线管理器"""

    def __init__(self, baseline_path="perf_baseline.json"):
        self.baseline_path = Path(baseline_path)
        self.baseline = self._load_baseline()

    def _load_baseline(self):
        if self.baseline_path.exists():
            with open(self.baseline_path) as f:
                return json.load(f)
        return {}

    def _save_baseline(self):
        with open(self.baseline_path, "w") as f:
            json.dump(self.baseline, f, indent=2)

    def measure(self, name, func, warmup=10, rep=100):
        """测量性能"""
        # Warmup
        for _ in range(warmup):
            func()

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(rep):
            func()
        torch.cuda.synchronize()
        elapsed = time.time() - start

        perf = rep / elapsed  # ops/sec
        return perf

    def check_regression(self, name, current_perf, threshold=0.05):
        """检查性能回归"""
        if name not in self.baseline:
            self.baseline[name] = current_perf
            self._save_baseline()
            return True

        baseline_perf = self.baseline[name]
        regression = (baseline_perf - current_perf) / baseline_perf

        if regression > threshold:
            print(f"❌ 性能回归: {name}")
            print(f"   基线: {baseline_perf:.2f}")
            print(f"   当前: {current_perf:.2f}")
            print(f"   回归: {regression:.1%}")
            return False

        print(f"✅ 性能正常: {name} ({current_perf:.2f})")
        return True

    def update_baseline(self, name, perf):
        """更新基线"""
        self.baseline[name] = perf
        self._save_baseline()
```

### 32.7.2 性能监控仪表板

```python
class PerformanceDashboard:
    """性能监控仪表板"""

    def __init__(self):
        self.history = []

    def record(self, kernel_name, metrics):
        """记录性能指标"""
        entry = {
            "timestamp": time.time(),
            "kernel": kernel_name,
            "metrics": metrics,
        }
        self.history.append(entry)

    def check_alerts(self, thresholds):
        """检查性能告警"""
        alerts = []
        for entry in self.history[-100:]:  # 最近 100 次
            for metric, value in entry["metrics"].items():
                if metric in thresholds and value < thresholds[metric]:
                    alerts.append({
                        "kernel": entry["kernel"],
                        "metric": metric,
                        "value": value,
                        "threshold": thresholds[metric],
                    })
        return alerts
```

---

## 32.8 CI/CD 流程

### 32.8.1 CI Pipeline 设计

```yaml
# .github/workflows/tilelang-ci.yml
name: TileLang CI

on: [push, pull_request]

jobs:
  test:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3

      - name: Setup Environment
        run: |
          pip install tilelang pytest torch
          pip install compute-sanitizer

      - name: Lint Check
        run: |
          ruff check .
          mypy . --ignore-missing-imports

      - name: Unit Tests
        run: |
          pytest tests/ -v --tb=short

      - name: Memory Check
        run: |
          compute-sanitizer --tool memcheck pytest tests/test_kernels.py

      - name: Performance Baseline
        run: |
          pytest tests/test_performance.py -v --benchmark

      - name: IR Dump Verification
        run: |
          python tests/verify_ir_dump.py

      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: |
            test-results/
            performance-baseline.json
```

### 32.8.2 预提交检查

```python
# pre-commit checks
def pre_commit_checks():
    """预提交检查"""
    checks = [
        ("Lint", run_lint),
        ("Type Check", run_type_check),
        ("Unit Tests", run_unit_tests),
        ("Performance", run_perf_check),
    ]

    results = {}
    for name, func in checks:
        print(f"Running {name}...")
        try:
            result = func()
            results[name] = "PASS" if result else "FAIL"
        except Exception as e:
            results[name] = f"ERROR: {e}"

    # 打印结果
    print("\n" + "=" * 50)
    print("Pre-commit Check Results:")
    for name, result in results.items():
        status = "✅" if result == "PASS" else "❌"
        print(f"  {status} {name}: {result}")

    return all(r == "PASS" for r in results.values())
```

---

## 32.9 常见错误模式与解决方案

### 32.9.1 错误模式目录

| 错误类型 | 典型表现 | 根因 | 解决方案 |
|----------|---------|------|---------|
| 编译失败 | IR 转换错误 | 不支持的语法 | 检查 TileLang API |
| 越界访问 | CUDA Error | 索引计算错误 | 添加边界检查 |
| Bank Conflict | 性能低 50% | 内存布局不当 | 使用 Swizzled Layout |
| NaN 输出 | 数值异常 | 溢出/除零 | FP32 累加 + epsilon |
| 性能回归 | 速度变慢 | 代码改动 | 性能基线对比 |
| OOM | 显存不足 | Tile 过大 | 减小 Tile 大小 |

### 32.9.2 调试工具速查表

```
TileLang 调试工具速查：

编译调试:
  tilelang.dump_ir()           → IR 输出
  kernel.get_source("tir")     → TensorIR
  kernel.get_source("ptx")     → PTX 汇编

内存调试:
  compute-sanitizer --tool memcheck    → 内存错误
  compute-sanitizer --tool racecheck   → 竞态条件
  torch.cuda.memory_stats()            → 显存统计

数值调试:
  torch.isnan()                → NaN 检测
  torch.isinf()                → Inf 检测
  torch.allclose()             → 精度对比
  torch.max(torch.abs(diff))   → 最大误差

性能调试:
  ncu --set full               → NCU 分析
  nsys profile                 → 系统追踪
  torch.cuda.Event()           → GPU 计时
```

---

## 32.10 高级调试技术

### 32.10.1 GPU Core Dump 分析

当 Kernel 崩溃时，GPU Core Dump 可以提供详细的执行状态：

```bash
# 启用 GPU Core Dump
export CUDA_CORE_DUMP=1
export CUDA_CORE_DUMP_FILE="./core_dump.%h.%p"

# 运行程序
python my_script.py

# 分析 Core Dump
compute-sanitizer --tool memcheck --core-file ./core_dump.xxx python analyze_core.py
```

```python
def analyze_gpu_core_dump(core_file):
    """分析 GPU Core Dump"""
    import cuda.bindings.driver as cuda

    # 加载 Core Dump
    core = cuda.load_core_dump(core_file)

    # 获取崩溃位置
    crash_info = core.get_crash_info()
    print(f"崩溃位置: {crash_info.file}:{crash_info.line}")
    print(f"崩溃原因: {crash_info.reason}")
    print(f"线程: ({crash_info.threadIdx.x}, {crash_info.threadIdx.y}, {crash_info.threadIdx.z})")
    print(f"Block: ({crash_info.blockIdx.x}, {crash_info.blockIdx.y}, {crash_info.blockIdx.z})")

    # 获取寄存器状态
    registers = core.get_registers()
    print(f"寄存器状态:")
    for i, val in enumerate(registers[:16]):
        print(f"  R{i}: {val:#018x}")

    # 获取 Shared Memory 状态
    smem = core.get_shared_memory()
    print(f"Shared Memory ({len(smem)} bytes):")
    for i in range(0, min(256, len(smem)), 16):
        hex_str = " ".join(f"{b:02x}" for b in smem[i:i+16])
        print(f"  {i:04x}: {hex_str}")
```

### 32.10.2 Warp 状态分析

```python
def analyze_warp_state(ncu_output):
    """分析 Warp 执行状态"""
    warp_states = {
        "active": "活跃 Warp 数",
        "stall_mio_throttle": "内存 IO 节流",
        "stall_wait": "等待依赖",
        "stall_not_selected": "未被调度",
        "stall_barrier": "同步屏障",
        "stall_membar": "内存屏障",
        "stall_drain": "排空等待",
    }

    print("Warp 状态分析:")
    for state, desc in warp_states.items():
        value = ncu_output.get_metric(f"warp_state_{state}", 0)
        bar = "█" * int(value / 10)
        print(f"  {desc:20s}: {value:6.1f}% {bar}")

    # 识别瓶颈
    max_state = max(warp_states.items(),
                    key=lambda x: ncu_output.get_metric(f"warp_state_{x[0]}", 0))
    print(f"\n主要瓶颈: {max_state[1]}")
```

### 32.10.3 Shared Memory Bank Conflict 检测

```python
def detect_bank_conflicts(kernel_func, input_shapes):
    """检测 Shared Memory Bank Conflict"""
    # 使用 NCU 检测
    import subprocess

    cmd = [
        "ncu", "--set", "full",
        "--metrics", "shared_load_bank_conflict,shared_store_bank_conflict",
        "python", "-c", f"""
import torch
import tilelang
A = torch.randn({input_shapes[0]}, device='cuda', dtype=torch.float16)
B = torch.randn({input_shapes[1]}, device='cuda', dtype=torch.float16)
kernel = tilelang.compile('''{kernel_func}''')
for _ in range(100):
    kernel(A, B)
"""
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # 解析结果
    load_conflicts = parse_metric(result.stdout, "shared_load_bank_conflict")
    store_conflicts = parse_metric(result.stdout, "shared_store_bank_conflict")

    print(f"Bank Conflict 分析:")
    print(f"  Load Conflicts:  {load_conflicts}")
    print(f"  Store Conflicts: {store_conflicts}")

    if load_conflicts > 0.1 or store_conflicts > 0.1:
        print("  ⚠️ 存在显著 Bank Conflict，建议使用 Swizzled Layout")
    else:
        print("  ✅ 无显著 Bank Conflict")

    return load_conflicts, store_conflicts
```

### 32.10.4 性能 Counter 分析

```python
class PerformanceCounterAnalyzer:
    """性能 Counter 分析器"""

    def __init__(self):
        self.counters = {}

    def collect(self, kernel_func, input_shapes):
        """收集性能 Counter"""
        # 使用 NCU 收集
        counters = {
            # 计算利用率
            "sm_active": "SM 活跃率",
            "achieved_occupancy": "实际占用率",
            "sm_efficiency": "SM 效率",
            "tensor_core_utilization": "Tensor Core 利用率",

            # 内存利用率
            "dram_throughput": "DRAM 吞吐",
            "l2_throughput": "L2 吞吐",
            "shared_memory_throughput": "Shared Memory 吞吐",

            # 指令效率
            "inst_executed": "执行指令数",
            "inst_per_warp": "每 Warp 指令数",
            "ipc": "IPC",

            # 瓶颈
            "warp_issue_stalled_long_scoreboard": "长等待",
            "warp_issue_stalled_short_scoreboard": "短等待",
            "warp_issue_stalled_wait": "等待",
        }

        for counter, desc in counters.items():
            value = self._collect_counter(kernel_func, counter)
            self.counters[counter] = value

        return self.counters

    def analyze_bottleneck(self):
        """分析性能瓶颈"""
        bottlenecks = []

        # 检查计算利用率
        if self.counters.get("tensor_core_utilization", 0) < 50:
            bottlenecks.append("Tensor Core 利用率低")

        # 检查内存瓶颈
        if self.counters.get("dram_throughput", 0) > 80:
            bottlenecks.append("DRAM 带宽瓶颈")

        # 检查占用率
        if self.counters.get("achieved_occupancy", 0) < 50:
            bottlenecks.append("占用率过低")

        return bottlenecks

    def suggest_optimizations(self):
        """建议优化方向"""
        bottlenecks = self.analyze_bottleneck()
        suggestions = []

        if "Tensor Core 利用率低" in bottlenecks:
            suggestions.append("1. 确保使用 Tensor Core 指令 (T.gemm)")
            suggestions.append("2. 检查矩阵维度是否对齐到 Tensor Core 要求")

        if "DRAM 带宽瓶颈" in bottlenecks:
            suggestions.append("1. 增加数据复用 (增大 Tile 大小)")
            suggestions.append("2. 使用 Software Pipelining")
            suggestions.append("3. 考虑数据压缩/量化")

        if "占用率过低" in bottlenecks:
            suggestions.append("1. 减小 Tile 大小")
            suggestions.append("2. 减少寄存器使用")
            suggestions.append("3. 减少 Shared Memory 使用")

        return suggestions
```

### 32.10.5 自动化调试工具

```python
class TileLangDebugger:
    """TileLang 自动化调试工具"""

    def __init__(self, kernel_func, input_shapes):
        self.kernel = kernel_func
        self.shapes = input_shapes

    def full_debug(self):
        """完整调试流程"""
        print("=" * 60)
        print("TileLang 自动化调试")
        print("=" * 60)

        # Step 1: 编译检查
        print("\n[1/5] 编译检查...")
        try:
            compiled = self.check_compilation()
            print("  ✅ 编译成功")
        except Exception as e:
            print(f"  ❌ 编译失败: {e}")
            return

        # Step 2: 正确性检查
        print("\n[2/5] 正确性检查...")
        correctness = self.check_correctness()
        if correctness["passed"]:
            print(f"  ✅ 正确性检查通过 (误差: {correctness['max_error']:.6e})")
        else:
            print(f"  ❌ 正确性检查失败 (误差: {correctness['max_error']:.6e})")

        # Step 3: 内存检查
        print("\n[3/5] 内存检查...")
        memory_ok = self.check_memory()
        if memory_ok:
            print("  ✅ 无内存错误")
        else:
            print("  ❌ 存在内存错误")

        # Step 4: 性能检查
        print("\n[4/5] 性能检查...")
        perf = self.check_performance()
        print(f"  性能: {perf['tflops']:.2f} TFLOPS")
        print(f"  带宽利用率: {perf['bandwidth_util']:.1%}")

        # Step 5: 优化建议
        print("\n[5/5] 优化建议...")
        suggestions = self.get_suggestions()
        for i, s in enumerate(suggestions, 1):
            print(f"  {i}. {s}")

        print("\n" + "=" * 60)

    def check_compilation(self):
        """检查编译"""
        return tilelang.compile(self.kernel)

    def check_correctness(self):
        """检查正确性"""
        inputs = [torch.randn(*s, device="cuda", dtype=torch.float16)
                  for s in self.shapes]
        output = self.kernel(*inputs)
        ref = self.reference_impl(*inputs)
        max_error = torch.abs(output - ref).max().item()
        return {
            "passed": max_error < 1e-3,
            "max_error": max_error,
        }

    def check_memory(self):
        """检查内存"""
        import subprocess
        result = subprocess.run(
            ["compute-sanitizer", "--tool", "memcheck", "python", "-c",
             f"import torch; kernel(...); torch.cuda.synchronize()"],
            capture_output=True
        )
        return result.returncode == 0

    def check_performance(self):
        """检查性能"""
        inputs = [torch.randn(*s, device="cuda", dtype=torch.float16)
                  for s in self.shapes]

        # Warmup
        for _ in range(10):
            self.kernel(*inputs)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            self.kernel(*inputs)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        flops = self.estimate_flops()
        tflops = flops * 100 / elapsed / 1e12

        return {"tflops": tflops, "bandwidth_util": 0.85}
```

### 32.10.6 常见错误的快速修复

```
错误快速修复指南：

1. "CUDA error: invalid configuration"
   原因: Grid/Block 配置无效
   修复: 检查 blockDim.x * blockDim.y * blockDim.z <= 1024

2. "CUDA error: out of memory"
   原因: 显存不足
   修复: 减小 batch_size 或 Tile 大小

3. "NaN detected in output"
   原因: 数值溢出
   修复: 使用 FP32 累加器，检查 Softmax

4. "Performance regression detected"
   原因: 代码改动导致性能下降
   修复: 对比 IR dump，检查新增的同步点

5. "Bank conflict detected"
   原因: Shared Memory 访问模式不当
   修复: 使用 Swizzled Layout 或调整访问模式

6. "Register spill detected"
   原因: 寄存器使用超限
   修复: 减小 Tile 大小或使用 T.alloc_L1
```

---

## Summary

| 主题 | 核心要点 |
|------|---------|
| 调试层次 | 编译→内存→数值→性能 四层模型 |
| IR Dump | 多层次 IR 查看：Tile IR → TensorIR → PTX |
| 内存调试 | compute-sanitizer 四种工具 |
| 数值精度 | FP32 累加 + Online Softmax + epsilon |
| 测试框架 | pytest + 参考实现对比 + 模糊测试 |
| 性能回归 | 基线管理 + CI/CD 自动检测 |
| 常见问题 | 6 种典型错误模式与解决方案 |

---

## Exercises

### Exercise 1: IR Dump 分析
编写一个 TileLang 算子，使用 `tilelang.dump_ir()` 输出 IR，分析：
- 循环结构是否正确
- 内存分配是否合理
- 是否有冗余操作

### Exercise 2: 内存错误修复
以下代码包含内存错误，请使用 compute-sanitizer 定位并修复：

```python
@T.prim_func
def buggy_kernel(A: T.Tensor([1024], "float16"), Output: T.Tensor([1024], "float16")):
    smem = T.alloc_shared([64], "float16")
    for i in T.Parallel(128):
        smem[i] = A[i]
    for i in T.Parallel(128):
        Output[i] = smem[i]
```

### Exercise 3: 性能回归检测
实现一个简单的性能回归检测系统：
- 记录算子的基线性能
- 在 CI 中自动检测性能回归
- 当回归超过 5% 时报警

---

## Thinking Questions

1. **为什么 TileLang 的 IR dump 对调试编译问题特别重要？** 提示：考虑编译器的多层转换过程。

2. **在数值精度调试中，为什么使用 FP32 累加器而不是 FP64？** 提示：考虑性能与精度的平衡。

3. **如何设计一个既全面又高效的 CI/CD 测试流程？** 提示：考虑测试时间、覆盖率和硬件成本。

4. **模糊测试在 TileLang 调试中为什么重要？** 提示：考虑用户可能的各种输入组合。

---

## Extension Reading

1. **CUDA-MEMCHECK User Guide** - NVIDIA 内存检查工具文档
2. **pytest Documentation** - Python 测试框架
3. **TVM Debugging Guide** - TVM 编译器调试指南
4. **GPU Debugging Best Practices** - NVIDIA GPU 调试最佳实践
5. **Continuous Integration for CUDA Projects** - CUDA 项目 CI/CD 指南

---

## 32.11 IR Dump 分析实例

### 32.11.1 GEMM IR 分析

```python
# 完整的 GEMM IR Dump 分析示例

# 原始 TileLang 代码
@T.prim_func
def gemm_kernel(
    A: T.Tensor([1024, 1024], "float16"),
    B: T.Tensor([1024, 1024], "float16"),
    C: T.Tensor([1024, 1024], "float16"),
):
    with T.Kernel(8, 8, threads=256) as (bx, by):
        A_smem = T.alloc_shared([128, 32], "float16")
        B_smem = T.alloc_shared([32, 128], "float16")
        C_frag = T.alloc_fragment([128, 128], "float32")

        T.clear(C_frag)
        for k in T.serial(32):
            T.copy(A[bx*128:(bx+1)*128, k*32:(k+1)*32], A_smem)
            T.copy(B[k*32:(k+1)*32, by*128:(by+1)*128], B_smem)
            T.syncthreads()
            T.gemm(A_smem, B_smem, C_frag)
            T.syncthreads()

        T.copy(C_frag, C[bx*128:(bx+1)*128, by*128:(by+1)*128])
```

**Tile IR 分析：**

```
# Tile IR (编译器前端输出)
# 关键观察点:
# 1. T.Kernel(8, 8) → 8x8 = 64 个 Block
# 2. T.alloc_shared → Shared Memory 分配
# 3. T.alloc_fragment → 寄存器分配
# 4. T.gemm → Tensor Core 指令

# 常见问题:
# - 如果看到 T.serial 而非 T.Pipelined → 可能缺少 Pipeline 优化
# - 如果看到多次 T.syncthreads → 可能有冗余同步
# - 如果看到 T.alloc_shared 大小不合理 → 可能有内存浪费
```

**TensorIR 分析：**

```
# TensorIR (TVM 中间表示)
# 关键观察点:
# 1. blockIdx.x, blockIdx.y → Block 映射
# 2. threadIdx.x → 线程映射
# 3. T.allocate(..., "shared") → Shared Memory
# 4. T.allocate(..., "local") → 寄存器

# 常见问题:
# - 如果循环未展开 → 添加 T.unroll 注解
# - 如果向量化失败 → 检查数据对齐
# - 如果寄存器溢出 → 减小 Tile 大小
```

**PTX 分析：**

```
# PTX (底层汇编)
# 关键观察点:
# 1. ld.shared → Shared Memory 读取
# 2. st.shared → Shared Memory 写入
# 3. ldmatrix → Tensor Core 加载
# 4. mma.sync → Tensor Core 计算
# 5. bar.sync → 同步屏障

# 常见问题:
# - 如果看到大量 ld.global → 数据未缓存到 Shared Memory
# - 如果看到 bank conflict → 需要 Swizzle
# - 如果看到 register spill → 减小 Tile 大小
```

### 32.11.2 FlashAttention IR 分析

```python
# FlashAttention IR 分析要点

# 1. Online Softmax 的实现
#    - 查找 max 的累积更新
#    - 查找 exp(x - m) 的计算
#    - 查找 l 的累积更新

# 2. Causal Mask 的实现
#    - 查找条件判断: if q_idx >= k_idx
#    - 查找 mask 应用: S = where(mask, S, -inf)

# 3. Pipeline 的实现
#    - 查找 T.Pipelined 注解
#    - 查找多级 buffer 分配
#    - 查找 prefetch 逻辑

# 4. 内存访问模式
#    - Q: 每个 Block 加载一次
#    - K, V: 在循环中多次加载
#    - O: 每个 Block 写出一次
```

---

## 32.12 内存错误模式扩展

### 32.12.1 Warp 级内存错误

```
错误信息:
========= Invalid __shared__ read of size 4 bytes
=========     at 0x00000100 in kernel
=========     by thread (31,0,0) in block (0,0,0)
=========     Address 0x00007fff00000100 is out of bounds

原因: Warp 内某个线程访问了越界的 Shared Memory
```

```python
# 错误示例: Warp 级别边界问题
@T.prim_func
def warp_boundary_bug(
    A: T.Tensor([100], "float32"),
    B: T.Tensor([100], "float32"),
):
    smem = T.alloc_shared([32], "float32")
    # 线程 0-31 都会执行，但只有 0-99 有效
    tid = T.thread_id()
    if tid < 100:
        smem[tid % 32] = A[tid]  # 当 tid >= 32 时，覆盖之前的值
    T.syncthreads()
    # 读取时可能读到被覆盖的值
    B[tid] = smem[tid % 32]
```

### 32.12.2 Race Condition 检测

```bash
# 使用 compute-sanitizer 检测 Race Condition
compute-sanitizer --tool racecheck python my_script.py

# 输出示例:
# ========= Race reported between Write access at 0x100 in kernel
# ========= and Read access at 0x100 in kernel
# =========     by thread (0,0,0) in block (0,0,0)
```

```python
# 错误示例: Race Condition
@T.prim_func
def race_condition_bug(A: T.Tensor([256], "float32")):
    smem = T.alloc_shared([256], "float32")
    tid = T.thread_id()

    # 线程 0 写入
    if tid == 0:
        for i in T.serial(256):
            smem[i] = A[i]

    # 其他线程可能在写入完成前就读取
    # 缺少 T.syncthreads()!
    val = smem[tid]

# 修复: 添加同步
@T.prim_func
def race_condition_fixed(A: T.Tensor([256], "float32")):
    smem = T.alloc_shared([256], "float32")
    tid = T.thread_id()

    if tid == 0:
        for i in T.serial(256):
            smem[i] = A[i]
    T.syncthreads()  # 确保写入完成

    val = smem[tid]
```

### 32.12.3 内存对齐错误

```python
# 错误示例: 内存对齐问题
@T.prim_func
def alignment_bug(A: T.Tensor([100], "float32")):
    # 尝试向量化访问，但起始地址未对齐
    smem = T.alloc_shared([100], "float32")
    # 某些硬件要求 128-bit 对齐的向量化访问
    for i in T.serial(0, 100, 4):
        # 如果 i 不是 4 的倍数，可能出错
        val = T.vector_load(smem, i, 4)  # 加载 4 个 float32
```

---

## 32.13 数值精度调试扩展

### 32.13.1 混合精度调试

```python
def debug_mixed_precision(kernel_func, inputs, reference_func):
    """调试混合精度问题"""

    # 测试不同精度组合
    precisions = [
        ("float16", "float16", "float32"),   # 输入FP16，累加FP32
        ("bfloat16", "bfloat16", "float32"), # 输入BF16，累加FP32
        ("float32", "float32", "float32"),   # 全FP32
    ]

    results = {}
    for in_dtype, w_dtype, acc_dtype in precisions:
        # 转换输入精度
        inputs_cast = [x.to(getattr(torch, in_dtype)) for x in inputs]

        # 运行 kernel
        output = kernel_func(*inputs_cast)

        # 对比参考
        ref = reference_func(*inputs_cast)

        # 计算误差
        abs_diff = torch.abs(output.float() - ref.float())
        rel_diff = abs_diff / (torch.abs(ref.float()) + 1e-8)

        results[(in_dtype, w_dtype, acc_dtype)] = {
            "max_abs": abs_diff.max().item(),
            "max_rel": rel_diff.max().item(),
            "mean_abs": abs_diff.mean().item(),
        }

    # 打印对比
    print("精度对比:")
    for key, val in results.items():
        print(f"  {key}: max_abs={val['max_abs']:.2e}, max_rel={val['max_rel']:.2e}")

    return results
```

### 32.13.2 数值稳定性检查工具

```python
class NumericalStabilityChecker:
    """数值稳定性检查器"""

    def __init__(self, kernel_func, reference_func):
        self.kernel = kernel_func
        self.reference = reference_func

    def check_overflow(self, inputs, name="kernel"):
        """检查溢出"""
        output = self.kernel(*inputs)

        if torch.isnan(output).any():
            print(f"❌ {name}: 检测到 NaN")
            nan_mask = torch.isnan(output)
            print(f"   NaN 位置: {torch.where(nan_mask)}")
            return False

        if torch.isinf(output).any():
            print(f"❌ {name}: 检测到 Inf")
            inf_mask = torch.isinf(output)
            print(f"   Inf 位置: {torch.where(inf_mask)}")
            return False

        print(f"✅ {name}: 无溢出")
        return True

    def check_gradient_stability(self, inputs, name="kernel"):
        """检查梯度稳定性"""
        # 创建需要梯度的输入
        inputs_grad = [x.clone().requires_grad_(True) for x in inputs]

        # 前向传播
        output = self.kernel(*inputs_grad)

        # 反向传播
        loss = output.sum()
        loss.backward()

        # 检查梯度
        for i, inp in enumerate(inputs_grad):
            if inp.grad is not None:
                if torch.isnan(inp.grad).any():
                    print(f"❌ {name}: 输入 {i} 的梯度包含 NaN")
                    return False
                if torch.isinf(inp.grad).any():
                    print(f"❌ {name}: 输入 {i} 的梯度包含 Inf")
                    return False
                if inp.grad.abs().max() > 1e6:
                    print(f"⚠️ {name}: 输入 {i} 的梯度过大 ({inp.grad.abs().max():.2e})")

        print(f"✅ {name}: 梯度稳定")
        return True

    def run_stress_test(self, shape_generator, num_tests=100):
        """压力测试"""
        passed = 0
        failed = 0

        for i in range(num_tests):
            inputs = shape_generator()
            if self.check_overflow(inputs, f"Test {i}"):
                passed += 1
            else:
                failed += 1

        print(f"\n压力测试结果: {passed}/{num_tests} 通过")
        return passed, failed
```

---

## 32.14 pytest Fixture 高级用法

### 32.14.1 参数化 Fixture

```python
import pytest
import torch

# 多维度参数化
@pytest.fixture(params=[
    (128, 128, 128, "float16"),
    (256, 256, 256, "float16"),
    (512, 512, 512, "float16"),
    (1024, 1024, 1024, "float16"),
    (128, 128, 128, "bfloat16"),
    (256, 256, 256, "bfloat16"),
], ids=[
    "small-fp16", "medium-fp16", "large-fp16", "xlarge-fp16",
    "small-bf16", "medium-bf16",
])
def gemm_config(request):
    M, N, K, dtype = request.param
    return {
        "M": M, "N": N, "K": K,
        "dtype": dtype,
        "A": torch.randn(M, K, device="cuda", dtype=getattr(torch, dtype)),
        "B": torch.randn(K, N, device="cuda", dtype=getattr(torch, dtype)),
    }

# Fixture 依赖
@pytest.fixture
def compiled_kernel(gemm_config):
    """编译 kernel（耗时操作，共享给多个测试）"""
    M, N, K = gemm_config["M"], gemm_config["N"], gemm_config["K"]
    kernel = tilelang.compile(
        create_gemm_kernel(M, N, K),
        target="cuda",
        out_idx=[2],
    )
    return kernel

@pytest.fixture
def reference_output(gemm_config):
    """参考实现输出"""
    A, B = gemm_config["A"], gemm_config["B"]
    return torch.matmul(A, B)
```

### 32.14.2 自动化测试生成

```python
# conftest.py - 自动生成测试用例
import pytest

def generate_test_cases():
    """自动生成测试用例"""
    cases = []

    # 不同矩阵大小
    sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    for M in sizes:
        for N in sizes:
            for K in sizes:
                cases.append({
                    "id": f"M{M}_N{N}_K{K}",
                    "M": M, "N": N, "K": K,
                })

    # 边界条件
    boundary_cases = [
        {"id": "M1", "M": 1, "N": 1024, "K": 1024},
        {"id": "N1", "M": 1024, "N": 1, "K": 1024},
        {"id": "K1", "M": 1024, "N": 1024, "K": 1},
        {"id": "odd_M", "M": 100, "N": 128, "K": 128},
        {"id": "odd_N", "M": 128, "N": 100, "K": 128},
        {"id": "odd_K", "M": 128, "N": 128, "K": 100},
    ]
    cases.extend(boundary_cases)

    return cases

@pytest.fixture(params=generate_test_cases(), ids=lambda c: c["id"])
def test_case(request):
    return request.param
```

### 32.14.3 性能测试 Fixture

```python
# 性能测试专用 fixture
@pytest.fixture
def performance_timer():
    """GPU 计时器"""
    class GPUTimer:
        def __init__(self):
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

        def start(self):
            torch.cuda.synchronize()
            self.start_event.record()

        def stop(self):
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event)

    return GPUTimer()

@pytest.fixture
def warmup_and_benchmark(performance_timer):
    """预热和基准测试"""
    def _benchmark(func, warmup=10, repeat=100):
        # 预热
        for _ in range(warmup):
            func()

        # 测试
        times = []
        for _ in range(repeat):
            performance_timer.start()
            func()
            elapsed = performance_timer.stop()
            times.append(elapsed)

        return {
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "std": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        }

    return _benchmark
```

---

## 32.15 CI/CD 配置扩展

### 32.15.1 完整 CI Pipeline

```yaml
# .github/workflows/tilelang-ci-complete.yml
name: TileLang CI Complete

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  CUDA_VERSION: "12.2"
  PYTHON_VERSION: "3.10"
  TILELANG_VERSION: "latest"

jobs:
  # Job 1: 代码质量检查
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install linting tools
        run: |
          pip install ruff mypy black isort

      - name: Ruff check
        run: ruff check .

      - name: MyPy type check
        run: mypy . --ignore-missing-imports

      - name: Black format check
        run: black --check .

      - name: Import sort check
        run: isort --check-only .

  # Job 2: 单元测试
  unit-tests:
    runs-on: [self-hosted, gpu]
    needs: code-quality
    strategy:
      matrix:
        gpu: [a100, h100]
    steps:
      - uses: actions/checkout@v3

      - name: Setup Environment
        run: |
          pip install tilelang pytest torch
          pip install pytest-xdist pytest-timeout

      - name: Run unit tests
        run: |
          pytest tests/unit/ -v \
            --timeout=300 \
            --tb=short \
            -x \
            --junitxml=test-results/unit-${{ matrix.gpu }}.xml

      - name: Upload test results
        uses: actions/upload-artifact@v3
        with:
          name: unit-test-results-${{ matrix.gpu }}
          path: test-results/

  # Job 3: 集成测试
  integration-tests:
    runs-on: [self-hosted, gpu]
    needs: unit-tests
    steps:
      - uses: actions/checkout@v3

      - name: Setup Environment
        run: |
          pip install tilelang pytest torch

      - name: Run integration tests
        run: |
          pytest tests/integration/ -v \
            --timeout=600 \
            --tb=long

  # Job 4: 内存检查
  memory-check:
    runs-on: [self-hosted, gpu]
    needs: unit-tests
    steps:
      - uses: actions/checkout@v3

      - name: Setup Environment
        run: |
          pip install tilelang pytest torch

      - name: Memory check with compute-sanitizer
        run: |
          compute-sanitizer --tool memcheck \
            --leak-check full \
            pytest tests/test_kernels.py -v

  # Job 5: 性能基准测试
  performance-benchmark:
    runs-on: [self-hosted, gpu, h100]
    needs: integration-tests
    steps:
      - uses: actions/checkout@v3

      - name: Setup Environment
        run: |
          pip install tilelang pytest torch

      - name: Run performance benchmarks
        run: |
          pytest tests/performance/ -v \
            --benchmark \
            --benchmark-json=benchmark-results.json

      - name: Check performance regression
        run: |
          python scripts/check_regression.py \
            --baseline performance-baseline.json \
            --current benchmark-results.json \
            --threshold 0.05

      - name: Update baseline (on main branch)
        if: github.ref == 'refs/heads/main'
        run: |
          cp benchmark-results.json performance-baseline.json
          git add performance-baseline.json
          git commit -m "Update performance baseline" || true

      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: |
            benchmark-results.json
            performance-baseline.json

  # Job 6: IR Dump 验证
  ir-dump-verification:
    runs-on: [self-hosted, gpu]
    needs: unit-tests
    steps:
      - uses: actions/checkout@v3

      - name: Setup Environment
        run: |
          pip install tilelang pytest torch

      - name: Verify IR dumps
        run: |
          python tests/verify_ir_dump.py

      - name: Upload IR dumps
        uses: actions/upload-artifact@v3
        with:
          name: ir-dumps
          path: ir-dump/
```

### 32.15.2 性能回归检测脚本

```python
# scripts/check_regression.py
import json
import sys
import argparse

def load_json(path):
    with open(path) as f:
        return json.load(f)

def check_regression(baseline, current, threshold=0.05):
    """检查性能回归"""
    regressions = []

    for test_name, current_perf in current.items():
        if test_name not in baseline:
            print(f"⚠️ 新测试 {test_name}，跳过回归检查")
            continue

        baseline_perf = baseline[test_name]
        regression = (baseline_perf - current_perf) / baseline_perf

        if regression > threshold:
            regressions.append({
                "test": test_name,
                "baseline": baseline_perf,
                "current": current_perf,
                "regression": regression,
            })

    return regressions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    current = load_json(args.current)

    regressions = check_regression(baseline, current, args.threshold)

    if regressions:
        print("❌ 性能回归检测:")
        for reg in regressions:
            print(f"  {reg['test']}: {reg['baseline']:.2f} → {reg['current']:.2f} "
                  f"({reg['regression']:.1%} 回归)")
        sys.exit(1)
    else:
        print("✅ 无性能回归")
        sys.exit(0)

if __name__ == "__main__":
    main()
```

---

## 32.16 常见错误目录扩展

### 32.16.1 编译错误

| 错误信息 | 原因 | 解决方案 |
|----------|------|---------|
| `SyntaxError: invalid syntax` | Python 语法错误 | 检查 `@T.prim_func` 装饰器 |
| `TypeError: Buffer shape mismatch` | Buffer 形状不匹配 | 检查 Tensor 形状定义 |
| `ValueError: unsupported dtype` | 不支持的数据类型 | 使用 FP16/BF16/FP32 |
| `RuntimeError: out of shared memory` | Shared Memory 超限 | 减小 Tile 大小 |
| `RuntimeError: register spill` | 寄存器溢出 | 减小 Tile 大小或使用 L1 |

### 32.16.2 运行时错误

| 错误信息 | 原因 | 解决方案 |
|----------|------|---------|
| `CUDA error: illegal memory access` | 越界访问 | 添加边界检查 |
| `CUDA error: misaligned address` | 内存对齐问题 | 确保数据对齐 |
| `CUDA error: too many resources` | 资源超限 | 减小 Block 大小 |
| `CUDA error: launch timeout` | Kernel 执行超时 | 减小计算量或分批 |
| `RuntimeError: NaN detected` | 数值溢出 | 使用 FP32 累加器 |

### 32.16.3 性能问题

| 现象 | 可能原因 | 诊断方法 | 解决方案 |
|------|---------|---------|---------|
| 性能低于预期 50%+ | Bank Conflict | NCU 分析 | 使用 Swizzled Layout |
| 性能低于预期 30%+ | 内存未合并 | NCU 分析 | 调整数据布局 |
| 性能低于预期 20%+ | 寄存器溢出 | IR Dump 分析 | 减小 Tile 大小 |
| 性能不稳定 | 分支发散 | NCU 分析 | 减少 warp 内分支 |
| 编译时间过长 | 搜索空间过大 | 计时 | 减少 auto-tune 范围 |

---

## Summary (更新)

| 主题 | 核心要点 |
|------|---------|
| 调试层次 | 编译→内存→数值→性能 四层模型 |
| IR Dump | 多层次 IR 查看：Tile IR → TensorIR → PTX |
| IR 分析 | 循环展开、向量化、同步点、寄存器使用 |
| 内存调试 | compute-sanitizer 四种工具 + Race Condition 检测 |
| 数值精度 | FP32 累加 + Online Softmax + epsilon + 混合精度调试 |
| 测试框架 | pytest + 参数化 Fixture + 自动化测试生成 |
| 性能回归 | 基线管理 + CI/CD 自动检测 + 回归检测脚本 |
| 常见问题 | 编译/运行时/性能三类错误目录 |
| CI/CD | 完整 Pipeline: 代码质量→单元测试→集成测试→内存检查→性能基准→IR 验证 |

---

## Extension Reading (更新)

1. **CUDA-MEMCHECK User Guide** - NVIDIA 内存检查工具文档
2. **pytest Documentation** - Python 测试框架
3. **TVM Debugging Guide** - TVM 编译器调试指南
4. **GPU Debugging Best Practices** - NVIDIA GPU 调试最佳实践
5. **Continuous Integration for CUDA Projects** - CUDA 项目 CI/CD 指南
6. **NVIDIA Nsight Compute** - 性能分析工具
7. **Compute Sanitizer** - 内存错误检测工具
8. **GitHub Actions for CUDA** - CI/CD 配置参考

---

## Next Chapter Preview

> **Chapter 33: 稀疏算子与结构化稀疏**
>
> 下一章将深入探讨稀疏计算在 TileLang 中的实现，包括 2:4 结构化稀疏、Block Sparse、Sparse FlashAttention 等，展示 TileLang 的 Tile 抽象如何大幅简化稀疏编程的复杂度。
