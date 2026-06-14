---
title: "Chapter 13: 编译管线全景——从 Python 到机器码"
description: "完整解析 TileLang 编译管线的每个阶段：Python DSL → Tile IR → TensorIR → TIR → LLVM → PTX/HSACO/Ascend C"
updated: 2026-06-11
---

# Chapter 13: 编译管线全景——从 Python 到机器码

> **Learning Objectives**
>
> 1. 理解从 Python DSL 到机器码的完整编译流程
> 2. 掌握每个编译阶段的关键 Pass 及其作用
> 3. 学习常用的编译优化策略（常量折叠、死代码消除、循环优化）
> 4. 了解针对不同硬件后端的 CodeGen 差异
> 5. 掌握编译选项和调试 Flag 的使用方法
> 6. 能够阅读和理解编译器 IR dump 输出

---

## 13.1 编译管线总览

### 13.1.1 六阶段编译模型

<div data-component="FullCompilationPipelineDiagram"></div>

TileLang 的编译管线可以分为六个主要阶段：

```
完整编译管线：

阶段 1: Python DSL
┌─────────────────────────────┐
│  @T.prim_func               │
│  def kernel(A, B, C):       │
│      with T.Kernel(...)     │
│          T.gemm(...)        │
└─────────────┬───────────────┘
              │ 解析
              ▼
阶段 2: TileLang IR
┌─────────────────────────────┐
│  Tile/Region/Block 抽象     │
│  高层计算表示                │
└─────────────┬───────────────┘
              │ Lowering
              ▼
阶段 3: TensorIR
┌─────────────────────────────┐
│  Buffer/Block/Loop 表示     │
│  标准 TVM IR                │
└─────────────┬───────────────┘
              │ TIR Passes
              ▼
阶段 4: TIR (Lowered)
┌─────────────────────────────┐
│  低级 TIR 表示              │
│  已优化的循环/内存访问       │
└─────────────┬───────────────┘
              │ CodeGen
              ▼
阶段 5: Target Dialect
┌─────────────────────────────┐
│  PTX (NVIDIA)               │
│  HSACO (AMD)                │
│  Ascend C (华为)            │
│  LLVM IR (CPU)              │
└─────────────┬───────────────┘
              │ 链接
              ▼
阶段 6: 可执行文件
┌─────────────────────────────┐
│  .cubin / .hsaco / .o       │
│  可直接加载执行              │
└─────────────────────────────┘
```

上述六阶段编译管线图展示了从 Python DSL 到可执行文件的完整转换路径。阶段 1 的 Python DSL 是用户的唯一入口，通过 `@T.prim_func` 装饰器和 `T.Kernel`、`T.gemm` 等高层 API 描述计算语义。阶段 2 将这些 Python 语法结构解析为 TileLang IR，这是一个面向 tile-based 计算的高层中间表示，核心抽象包括 Tile（计算分块）、Region（数据区域）和 Block（执行块）。阶段 3 的 Lowering 是最关键的转换步骤——它将高层的 tile 语义展开为低层的循环结构和缓冲区访问，完成从“做什么”到“怎么做”的第一次细化。阶段 4 的 TIR 优化 Passes 则在这个未经优化的低级表示上应用一系列编译器优化，包括常量折叠、死代码消除和循环变换。阶段 5 的 CodeGen 将优化后的 TIR 翻译为目标平台的代码：NVIDIA GPU 生成 PTX 汇编，AMD GPU 生成 HSACO，华为昇腾生成 Ascend C，CPU 则生成 LLVM IR。最后阶段 6 的链接器将多个编译单元拼接为可直接加载执行的文件。这种分层设计使得中间各阶段可以复用优化逻辑，而硬件差异被隔离在 CodeGen 层。

> **常见问题**：初学者容易混淆阶段 3（Lowering）和阶段 4（优化）的边界。Lowering 是语义等价的结构转换（从 tile 抽象展开为循环），而优化是保持语义但改变性能特性（如循环展开、向量化）。在实际代码中，这两类 Pass 可能交错执行，但逻辑上理解它们的区别有助于调试编译问题。

### 13.1.2 各阶段耗时分布

```
典型 kernel 编译耗时分布 (FlashMLA, H100):

┌──────────────────────────────────────────────┐
│  阶段              │  耗时 (ms)  │  占比      │
├────────────────────┼────────────┼───────────┤
│  Python 解析       │  5.2       │  8.3%     │
│  TileLang IR 构建  │  3.1       │  4.9%     │
│  TileLang → TensorIR│  8.4      │  13.4%    │
│  TIR 优化 Passes   │  15.6      │  24.8%    │
│  CodeGen (PTX)     │  22.3      │  35.5%    │
│  链接与优化        │  8.2       │  13.1%    │
│  总计              │  62.8      │  100%     │
└────────────────────┴────────────┴───────────┘
```

从上表可以看出，CodeGen 阶段（35.5%）和 TIR 优化阶段（24.8%）占据了编译时间的大头，两者合计超过 60%。这是合理的分配：CodeGen 阶段需要执行寄存器分配、指令选择等 NP-hard 问题，而 TIR 优化涉及大量的数据流分析和循环变换。相比之下，Python 解析和 TileLang IR 构建耗时很少（合计约 13%），这说明高层 DSL 的引入并没有带来显著的编译开销。开发者在进行编译调优时，应重点关注 TIR 优化 Passes 和 CodeGen 阶段——例如，合理设置 tile 大小可以减少搜索空间、加速优化收敛；使用编译缓存可以避免重复的 CodeGen 开销；对于开发期间的快速迭代，可以临时降低优化级别（如 O1）来减少编译等待时间。在 H100 等高端 GPU 上，CodeGen 耗时之所以突出，还因为 TMA（Tensor Memory Accelerator）等新硬件的指令选择需要更复杂的模式匹配。

---

## 13.2 阶段 1: Python DSL 解析

### 13.2.1 解析器架构

```python
# TileLang Python DSL 解析器
class TileLangParser:
    """
    将 @T.prim_func 装饰的 Python 函数解析为 TileLang IR
    """

    def __init__(self):
        self.ast_transformer = ASTTransformer()
        self.type_inferencer = TypeInferencer()

    def parse(self, func):
        """
        解析 Python 函数为 TileLang IR
        """
        # 1. 获取函数源码
        source = inspect.getsource(func)

        # 2. 解析为 AST
        ast_tree = ast.parse(source)

        # 3. AST 变换
        # - 将 T.Buffer 声明转换为 IR Buffer
        # - 将 T.Kernel 转换为 IR Kernel
        # - 将 T.gemm, T.copy 等转换为 IR 操作
        transformed = self.ast_transformer.visit(ast_tree)

        # 4. 类型推断
        typed = self.type_inferencer.infer(transformed)

        # 5. 构建 TileLang IR
        ir = self._build_ir(typed)

        return ir
```

TileLangParser 是整个编译管线的入口，负责将用户编写的 Python 函数转换为 TileLang IR。其工作流程分为五个步骤：第一步 `inspect.getsource(func)` 获取函数的原始源代码字符串，这依赖于 Python 的反射机制——因此被 `@T.prim_func` 装饰的函数不能是 lambda 或闭包（它们没有可靠的源码表示）。第二步 `ast.parse(source)` 将源码字符串解析为 Python 的标准 AST 树，这是 Python 官方提供的解析能力，精确保持了语法结构。第三步 `ast_transformer.visit(ast_tree)` 遍历 AST 树进行模式匹配和变换：它将 `T.Buffer` 调用识别为缓冲区声明、将 `T.Kernel` 上下文管理器识别为 kernel 作用域、将 `T.gemm` 和 `T.copy` 等函数调用识别为计算操作。第四步 `type_inferencer.infer(transformed)` 执行类型推断，推导每个中间表达式的形状和数据类型（例如 GEMM 操作的输出形状是 `[A_rows, B_cols]`）。第五步 `_build_ir(typed)` 将完成了类型标注的 AST 节点组装为 TileLang IR 的数据结构。设计上采用 AST 遍历而非字符串宏展开，是因为类型推断和形状推导需要语义级别的理解，纯文本替换无法处理如“根据输入形状动态推导 tile 分割”这样的场景。常见注意事项：如果函数使用了未导入的 TileLang API（如拼写错误的 `T.gem` 而非 `T.gemm`），解析器会在 AST 遍历阶段抛出 `AttributeError`；建议在编写 kernel 时保持函数体的简洁性，避免在 DSL 函数内部混入普通的 Python 计算逻辑。

### 13.2.2 关键解析规则

```python
# 解析规则示例

# 规则 1: T.Buffer 声明
"""
Python:
  A: T.Buffer[(M, N), "float16"]

IR:
  Buffer(name="A", shape=[M, N], dtype="float16", scope="global")
"""

# 规则 2: T.Kernel 声明
"""
Python:
  with T.Kernel(grid_m, grid_n, threads=256) as (bx, by):

IR:
  Kernel(grid=[grid_m, grid_n], threads=256,
         block_vars=[bx, by])
"""

# 规则 3: T.alloc_shared
"""
Python:
  A_smem = T.alloc_shared([128, 64], "float16")

IR:
  AllocBuffer(name="A_smem", shape=[128, 64],
              dtype="float16", scope="shared")
"""

# 规则 4: T.gemm
"""
Python:
  T.gemm(A, B, C, transpose_B=False)

IR:
  GEMM(A=A, B=B, C=C, transpose_A=False, transpose_B=False)
"""
```

上述四条解析规则覆盖了 TileLang DSL 中最核心的四种语法结构。规则 1 处理缓冲区声明：Python 中的 `T.Buffer[(M, N), "float16"]` 被解析为包含名称、形状、数据类型和作用域的 IR Buffer 对象——其中 `scope` 默认为 `"global"`，表示全局内存。规则 2 处理 Kernel 声明：`with T.Kernel(grid_m, grid_n, threads=256) as (bx, by)` 中的网格维度和线程数被提取为 IR Kernel 的配置参数，而 `bx`、`by` 则成为 kernel 体内部的块索引变量——编译器需要用这些变量来生成 GPU 的 blockIdx 绑定。规则 3 处理共享内存分配：`T.alloc_shared([128, 64], "float16")` 被解析为作用域为 `"shared"` 的缓冲区，这直接影响后续的 CodeGen 阶段（PTX 生成器看到 `scope="shared"` 时会生成 `.shared` 声明）。规则 4 处理矩阵乘法操作：`T.gemm(A, B, C, transpose_B=False)` 被解析为包含操作数引用和转置标志的 IR 操作——这些信息在 Lowering 阶段被展开为三重嵌套循环。解析规则的设计遵循“最小惊讶原则”：Python 中的书写形式与 IR 中的含义一一对应，不引入隐式转换。编写自定义解析规则时需注意：AST 节点类型可能因 Python 版本不同而略有差异（如 Python 3.8 和 3.12 对 `with` 语句的 AST 表示不同），因此 TileLang 通常会锁定支持的 Python 版本范围。

### 13.2.3 类型推断

```python
class TypeInferencer:
    """
    类型推断器
    推断中间表达式的类型和形状
    """

    def infer(self, ir):
        """
        对 IR 进行类型推断
        """
        # 对于每个操作，推断输出类型
        for op in ir.ops:
            if isinstance(op, GEMM):
                # GEMM 输出形状: [M, N]
                A_shape = self._get_shape(op.A)
                B_shape = self._get_shape(op.B)
                op.output_shape = [A_shape[0], B_shape[1]]
                op.output_dtype = self._promote_dtype(
                    op.A.dtype, op.B.dtype
                )
            elif isinstance(op, Copy):
                op.output_shape = op.input_shape
                op.output_dtype = op.input_dtype
            # ... 其他操作

        return ir
```

类型推断器在编译管线中扮演“形状传播者”的角色——它确保每个操作的输出形状和数据类型在编译时完全确定，这是后续 Lowering 和代码生成的前提条件。以 GEMM 操作为例：输入 A 的形状为 `[M, K]`，B 的形状为 `[K, N]`，推断器通过 `op.output_shape = [A_shape[0], B_shape[1]]` 计算出输出形状为 `[M, N]`；同时通过 `_promote_dtype()` 执行数据类型提升（如 `float16 × float16 → float16`，`float16 × float32 → float32`）。对于 Copy 操作，推断规则是直通的——输出的形状和类型直接继承自输入。这种形状推导不仅用于编译时检查（确保所有操作的维度兼容），还用于确定后续内存分配的大小、循环边界以及 tile 分割策略。设计上采用对每个 IR 操作逐个分发（`isinstance` 检查）的方式，而非统一的符号推导引擎；这是因为 TileLang 的操作为数不多（约 10 种），显式分发比通用的符号代数系统更简单、更易调试。**常见问题**：如果用户在未知维度上使用符号变量（如 `T.Buffer[(M, None), "float16"]`），推断器会因无法解析形状而报错——所有维度必须在解析阶段为已知的符号或常量。

---

## 13.3 阶段 2: TileLang IR

### 13.3.1 IR 数据结构

```python
@dataclass
class TileLangIR:
    """
    TileLang IR 的顶层表示
    """
    functions: Dict[str, TileLangFunction]
    metadata: Dict[str, Any]

@dataclass
class TileLangFunction:
    """
    TileLang 函数
    """
    name: str
    params: List[Buffer]
    body: KernelBlock
    grid_dims: List[Expr]
    num_threads: int

@dataclass
class KernelBlock:
    """
    Kernel 块
    """
    block_vars: List[Var]  # bx, by, ...
    shared_buffers: List[SharedBuffer]
    fragments: List[LocalBuffer]
    ops: List[Operation]
    sync_points: List[SyncBarrier]
```

TileLang IR 的四层数据结构构成了编译器中“高层语义”的核心载体。`TileLangIR` 是顶层容器，持有函数字典和元数据字典——这种设计支持一个 `.py` 文件定义多个 kernel 函数。`TileLangFunction` 描述单个 kernel 函数：`params` 列出输入输出缓冲区（与 Python 函数签名中的 `T.Buffer` 参数一一对应），`body` 是 `KernelBlock` 类型的计算体，`grid_dims` 指定 GPU 网格的维度数，`num_threads` 指定每个线程块的线程数。`KernelBlock` 是 kernel 体的具体表示：`block_vars` 保存块索引变量（如 bx、by），`shared_buffers` 和 `fragments` 分别管理共享内存和寄存器级别的本地缓冲区，`ops` 是顺序执行的操作列表，`sync_points` 标记需要在何处插入同步屏障。这个数据结构的设计刻意保持了顺序语义——`ops` 是列表而非图，意味着操作按照 Python 中 `with T.Kernel(...)` 体内的书写顺序依次执行。与 TVM 的 TensorIR 相比，TileLang IR 多了 `shared_buffers`、`sync_points` 等 GPU 特有的概念，因为它从一开始就面向 GPU 编程，而非通用的张量计算。

### 13.3.2 IR 操作类型

```python
# TileLang IR 支持的操作类型

class CopyOp(Operation):
    """数据搬运操作"""
    src: BufferRegion
    dst: BufferRegion
    transform: Optional[LayoutTransform]  # 可选的布局变换

class GEMMOp(Operation):
    """矩阵乘法操作"""
    A: Buffer
    B: Buffer
    C: Buffer
    transpose_A: bool
    transpose_B: bool
    scale: Optional[float]

class ElementwiseOp(Operation):
    """逐元素操作"""
    inputs: List[Buffer]
    output: Buffer
    compute_fn: Callable  # 计算函数

class ReduceOp(Operation):
    """归约操作"""
    input: Buffer
    output: Buffer
    axis: List[int]
    reduce_type: str  # sum, max, min, etc.

class SyncOp(Operation):
    """同步操作"""
    scope: str  # "threads", "warp", "block"
```

TileLang IR 定义了五种核心操作类型，覆盖了 GPU 计算中的主要模式。`CopyOp`（数据搬运）是最常见的操作——它不仅描述从源到目标的拷贝，还通过 `transform` 字段支持布局变换（如矩阵转置、数据 swizzle），这使得同一个数据搬运可以在传输过程中完成格式转换，避免额外 pass。`GEMMOp`（矩阵乘法）是计算密集型操作的核心——`transpose_A` 和 `transpose_B` 标志允许在计算语义层面表达转置需求，而无需在数据搬运层面显式处理；`scale` 字段用于融合缩放操作（如 `alpha * A @ B`），减少内存往返。`ElementwiseOp`（逐元素操作）通过 `compute_fn` 回调函数支持任意逐元素计算——这是一个 Python callable，在 Lowering 阶段被展开为具体的算术表达式。`ReduceOp`（归约操作）支持沿指定轴进行求和、最大值、最小值等归约，是 softmax、层归一化等算子的基础。`SyncOp`（同步操作）在不同的层级（线程、warp、块）插入屏障，确保数据一致性。这种操作分类遵循“正交覆盖”原则——每种操作有明确的功能边界，组合使用可以表达任意 GPU 计算模式，同时避免过于细粒度的 IR 指令导致 Lowering 复杂化。

---

## 13.4 阶段 3: TileLang → TensorIR Lowering

<div data-component="PassPipelineExplorer"></div>

### 13.4.1 Pass: TileLangLowerTileOp

```python
class TileLangLowerTileOp(tvm.tir.PrimFuncPass):
    """
    将 TileLang 的高层 tile 操作降低到 TensorIR

    主要转换：
    1. T.Kernel → For + Block
    2. T.copy → Block (element-wise)
    3. T.gemm → Block (nested loop)
    4. T.alloc_shared → alloc_buffer
    """

    def transform_function(self, func, mod, ctx):
        # 1. 查找 Kernel 块
        kernel = self._find_kernel_block(func)

        # 2. 创建 Grid 循环
        grid_loops = self._create_grid_loops(kernel)

        # 3. 创建 Thread 绑定
        thread_binding = self._create_thread_binding(kernel)

        # 4. 降低计算体
        lowered_body = self._lower_body(kernel.body)

        # 5. 组装为 TensorIR
        return self._assemble_tensorir(
            func, grid_loops, thread_binding, lowered_body
        )

    def _lower_body(self, body):
        """降低计算体中的每个操作"""
        lowered_ops = []
        for op in body.ops:
            if isinstance(op, CopyOp):
                lowered_ops.append(self._lower_copy(op))
            elif isinstance(op, GEMMOp):
                lowered_ops.append(self._lower_gemm(op))
            elif isinstance(op, SyncOp):
                lowered_ops.append(self._lower_sync(op))
        return lowered_ops

    def _lower_copy(self, copy_op):
        """
        降低 T.copy 操作

        T.copy(src, dst)
        →
        for i in T.serial(n):
            for j in T.serial(m):
                with T.block("copy"):
                    dst[i, j] = src[i, j]
        """
        shape = copy_op.src.shape
        loops = []
        for dim, size in enumerate(shape):
            loop = tvm.tir.For(
                loop_var=tvm.tir.Var(f"i{dim}", "int32"),
                min_val=0,
                extent=size,
                kind=tvm.tir.ForKind.SERIAL,
                body=None,
            )
            loops.append(loop)

        # 创建 Block
        block = tvm.tir.Block(
            iter_vars=[...],
            reads=[copy_op.src],
            writes=[copy_op.dst],
            body=tvm.tir.Store(
                copy_op.dst.buffer,
                copy_op.dst.indices,
                tvm.tir.Load(copy_op.src.dtype, copy_op.src.buffer, copy_op.src.indices),
            ),
        )

        return self._nest_loops_with_block(loops, block)

    def _lower_gemm(self, gemm_op):
        """
        降低 T.gemm 操作

        T.gemm(A, B, C)
        →
        for i in T.serial(M):
            for j in T.serial(N):
                with T.init():
                    C[i, j] = 0.0
                for k in T.serial(K):
                    C[i, j] += A[i, k] * B[k, j]
        """
        M, K = gemm_op.A.shape
        K2, N = gemm_op.B.shape

        # 创建嵌套循环
        loops = self._create_nested_loops([M, N, K])

        # 创建初始化 Block
        init_block = self._create_init_block(gemm_op.C, 0.0)

        # 创建计算 Block
        compute_block = self._create_compute_block(
            gemm_op.A, gemm_op.B, gemm_op.C,
            reduction_axes=["k"],
        )

        return self._assemble_gemm_ir(loops, init_block, compute_block)
```

`TileLangLowerTileOp` 是编译管线中最复杂也最关键的 Pass——它完成了从“高层 tile 语义”到“低层循环结构”的转换。`transform_function` 方法按五个步骤执行 Lowering：第一步定位 kernel 块（`_find_kernel_block`），找到 IR 中的 `KernelBlock` 节点；第二步创建 Grid 循环（`_create_grid_loops`），为每个网格维度生成并行循环——这些循环在 CodeGen 阶段会被映射为 GPU 的 `blockIdx`；第三步创建 Thread 绑定（`_create_thread_binding`），将 kernel 的线程映射为 TIR 的线程绑定属性；第四步降低计算体（`_lower_body`），对每个操作分别调用对应的 Lowering 方法；第五步组装（`_assemble_tensorir`）将前四步的结果拼接为标准 TensorIR。`_lower_copy` 方法的实现最为直观——它读取源缓冲区的形状，为每个维度创建一个串行 `For` 循环，循环体是一个 `Block` 节点，包含从源到目标的逐元素 `Store(Load(...))`。`_lower_gemm` 更为复杂：它生成 `i`（行）、`j`（列）、`k`（归约）三重嵌套循环，并插入初始化 block（将 C[i, j] 置零）和累加 block（C[i, j] += A[i, k] * B[k, j]）。这种递归嵌套的设计——先生成循环框架，再填充 block 体——使得每个操作类型的 Lowering 逻辑高度模块化。在前后编译阶段的关系上，Lowering 前的 TileLang IR 还保留了“做什么”的语义（如矩阵乘法），Lowering 后的 TensorIR 则完全展开了“怎么做”（循环+内存访问），为后续的 TIR 优化 Passes 提供了可分析的细粒度表示。

### 13.4.2 Pass: TileLangInferLayout

```python
class TileLangInferLayout(tvm.tir.PrimFuncPass):
    """
    推断和优化内存布局

    主要功能：
    1. 分析共享内存访问模式
    2. 检测 bank conflict
    3. 应用 swizzle/padding 优化
    """

    def transform_function(self, func, mod, ctx):
        # 1. 收集所有共享内存 buffer
        shared_buffers = self._collect_shared_buffers(func)

        # 2. 分析每个 buffer 的访问模式
        for buf in shared_buffers:
            access_pattern = self._analyze_access_pattern(func, buf)

            # 3. 检测 bank conflict
            conflicts = self._detect_bank_conflicts(access_pattern)

            # 4. 选择优化策略
            if conflicts:
                optimization = self._choose_optimization(
                    buf, access_pattern, conflicts
                )
                func = self._apply_optimization(func, buf, optimization)

        return func

    def _detect_bank_conflicts(self, access_pattern):
        """
        检测 bank conflict

        规则：当多个线程同时访问同一 bank 的不同地址时，
        会产生 bank conflict，导致访问串行化
        """
        conflicts = []
        for access in access_pattern.accesses:
            # 计算每个线程访问的 bank
            banks = [addr % 32 for addr in access.addresses]

            # 检测是否有多个线程访问同一 bank
            from collections import Counter
            bank_counts = Counter(banks)
            for bank, count in bank_counts.items():
                if count > 1:
                    conflicts.append({
                        "bank": bank,
                        "threads": count,
                        "access": access,
                    })

        return conflicts

    def _choose_optimization(self, buf, pattern, conflicts):
        """
        选择优化策略

        策略 1: Swizzle - 重排数据在 bank 中的位置
        策略 2: Padding - 添加填充来错开访问
        策略 3: 改变访问模式 - 调整循环顺序
        """
        if pattern.access_type == "row_major":
            # 行优先访问，使用 swizzle
            return SwizzleOptimization(offset=3)
        elif pattern.access_type == "column_major":
            # 列优先访问，使用 padding
            return PaddingOptimization(padding=1)
        else:
            return None
```

`TileLangInferLayout` 是 GPU 性能优化的关键 Pass——共享内存的 bank conflict 是 GPU kernel 性能的头号杀手之一。该 Pass 的工作流程分为四步：第一步收集所有共享内存缓冲区；第二步分析每个缓冲区的访问模式（`_analyze_access_pattern`），记录哪些线程在哪些时刻访问哪些地址；第三步检测 bank conflict（`_detect_bank_conflicts`）——核心算法是计算每个访问的 bank 编号（`addr % 32`），然后用 `Counter` 统计每个 bank 被多少线程同时访问；如果任一 bank 的并发访问数大于 1，则记录冲突详情。第四步选择优化策略（`_choose_optimization`）——对于行优先访问使用 swizzle（异或地址位来重排 bank 映射，通常用 offset=3），对于列优先访问使用 padding（在每行末尾添加 1 个元素使下一行偏移到新 bank）。这些策略的实际效果显著：以 128×64 的 float16 共享内存块为例，无优化时可能出现 4-way bank conflict，导致有效带宽降至 25%；应用 swizzle 后可消除冲突，带宽恢复至接近峰值。须注意，swizzle 和 padding 都会消耗额外的共享内存，因此在共享内存紧张时需要有取舍——编译器通常会设置优先级：先尝试 swizzle（开销小），若不满足再尝试 padding。

### 13.4.3 Pass: TileLegalizeLegacy

```python
class TileLegalizeLegacy(tvm.tir.PrimFuncPass):
    """
    合法化 Pass

    检查和转换：
    1. 共享内存大小是否超出限制
    2. 线程数量是否合法
    3. 数据类型是否支持
    4. 边界条件是否处理
    """

    def transform_function(self, func, mod, ctx):
        # 1. 检查共享内存大小
        shared_mem_usage = self._check_shared_memory(func)
        if shared_mem_usage > self.max_shared_memory:
            raise CompilationError(
                f"Shared memory usage {shared_mem_usage} exceeds limit "
                f"{self.max_shared_memory}"
            )

        # 2. 检查线程配置
        thread_config = self._check_thread_config(func)
        if thread_config.threads > 1024:
            raise CompilationError(
                f"Thread count {thread_config.threads} exceeds limit 1024"
            )

        # 3. 处理边界条件
        func = self._handle_boundary_conditions(func)

        # 4. 类型转换
        func = self._convert_types(func)

        return func

    def _handle_boundary_conditions(self, func):
        """
        处理边界条件

        当 tile 大小不能整除矩阵维度时，需要处理边界
        """
        for block in self._find_blocks(func):
            if self._needs_boundary_check(block):
                # 添加边界检查
                guard = tvm.tir.IfThenElse(
                    condition=self._create_boundary_condition(block),
                    then_body=block.body,
                    else_body=self._create_padding_value(block),
                )
                func = self._replace_block_body(func, block, guard)

        return func
```

`TileLegalizeLegacy` 是编译管线中的“安检员”——它在 Lowering 完成后、TIR 优化之前运行，确保所有 IR 节点符合目标硬件的约束。首先检查共享内存使用量是否超出限制：不同 GPU 有不同限制（如 A100 最大 164KB 静态共享内存），超出则直接抛出 `CompilationError` 而非静默失败——这是有意为之，因为超出共享内存的 kernel 无法在 GPU 上启动，尽早报错比运行时崩溃更易调试。其次检查线程数是否合法：CUDA 限制每块最多 1024 个线程，AMD 为 1024 个 work-item。第三步处理边界条件（`_handle_boundary_conditions`）：当 tile 大小不能整除矩阵维度时（如 128×128 的 tile 处理 200×200 的矩阵），需要为越界访问插入 `IfThenElse` 守卫——在 then 分支执行正常计算，在 else 分支写入填充值（如 0），防止越界内存访问。第四步执行类型转换：将不兼容的数据类型统一为硬件支持的类型。设计理念遵循“fail fast”原则——在 Lowering 后立即验证，避免将不合法的 IR 送入后续的优化 Passes，造成难以追踪的错误。开发者在调整 tile 大小时特别需要关注此 Pass 的输出——tile 越大性能可能越好，但超出硬件限制时会被此 Pass 拒绝。

---

## 13.5 阶段 4: TIR 优化 Passes

<div data-component="OptimizationPassComparison"></div>

阶段 4 是编译管线中唯一“不改变语义、只改变性能”的阶段。TIR 优化 Passes 接收 Lowering 后的 TensorIR 作为输入，应用一系列经典的编译器优化技术，输出优化后的 TIR。这个阶段的输入是未经优化的循环和内存访问（包含大量可简化的常量表达式、未使用的计算、可展开的小循环），输出则是经过优化的紧凑表示。关键优化策略分为四类：常量折叠消除编译时可计算的表达式，减少运行时指令；死代码消除删除对输出无贡献的计算，缩小代码体积；循环优化（展开、向量化、分块、交换）改变执行模式以更好地利用硬件流水线和内存层次；内存访问优化（合并、预取、复用）提高带宽利用率。这四种优化之间有依赖关系——常量折叠可能暴露新的死代码，循环展开可能创造向量化机会——因此通常按迭代方式执行：先折叠常量，再消除死代码，然后优化循环，最后优化内存，重复若干轮直到收敛。

### 13.5.1 常量折叠 (Constant Folding)

```python
class ConstantFolding(tvm.tir.PrimFuncPass):
    """
    常量折叠 Pass

    在编译时计算常量表达式，减少运行时计算
    """

    def transform_function(self, func, mod, ctx):
        return self._fold_constants(func)

    def _fold_constants(self, expr):
        """
        递归折叠常量表达式

        示例：
        输入: C[i, j] = 2 * 3 + A[i, j]
        输出: C[i, j] = 6 + A[i, j]
        """
        if isinstance(expr, tvm.tir.Add):
            lhs = self._fold_constants(expr.a)
            rhs = self._fold_constants(expr.b)

            # 如果两边都是常量，直接计算
            if isinstance(lhs, tvm.tir.IntImm) and isinstance(rhs, tvm.tir.IntImm):
                return tvm.tir.IntImm(expr.dtype, lhs.value + rhs.value)

            return tvm.tir.Add(lhs, rhs)

        elif isinstance(expr, tvm.tir.Mul):
            lhs = self._fold_constants(expr.a)
            rhs = self._fold_constants(expr.b)

            if isinstance(lhs, tvm.tir.IntImm) and isinstance(rhs, tvm.tir.IntImm):
                return tvm.tir.IntImm(expr.dtype, lhs.value * rhs.value)

            return tvm.tir.Mul(lhs, rhs)

        # ... 其他操作

        return expr
```

常量折叠是最基础也最安全的编译器优化——它在编译时计算所有操作数都是常量的表达式，将结果替换为立即数。以示例 `C[i, j] = 2 * 3 + A[i, j]` 为例：`_fold_constants` 递归遍历表达式树，遇到 `Mul(2, 3)` 时检测到两个操作数都是 `IntImm`（整数立即数），于是计算 `2 * 3 = 6` 并替换为 `IntImm(6)`；同理 `Add(6, A[i, j])` 中只有左侧是常量，右侧是变量引用 `A[i, j]`，因此放弃折叠保留原样。这种递归后序遍历保证了最深层的子表达式先被折叠。设计上采用模式匹配（`isinstance` 检查）而非在 AST 节点上定义虚函数，是因为常量折叠的逻辑高度集中且简单——一个类中的几十行代码即可涵盖所有算术和逻辑操作。在前后编译阶段的关系中，常量折叠在 Lowering 后立即执行尤为有效：Lowering 过程生成了许多形如 `stride * 0 + offset` 的索引表达式（因为循环展开模板统一生成索引公式），折叠后可以大幅简化后续 Pass 的分析负担。需要注意：浮点常量折叠需谨慎——由于 IEEE 754 浮点运算不完全满足结合律，`(a + b) + c` 和 `a + (b + c)` 可能结果不同；编译器通常在 `-ffast-math` 模式下才执行结合律变换。

### 13.5.2 死代码消除 (Dead Code Elimination)

```python
class DeadCodeElimination(tvm.tir.PrimFuncPass):
    """
    死代码消除 Pass

    移除不影响输出的计算
    """

    def transform_function(self, func, mod, ctx):
        # 1. 标记所有有用的计算（从输出反向追踪）
        useful = self._mark_useful(func)

        # 2. 移除未标记的计算
        func = self._remove_dead_code(func, useful)

        return func

    def _mark_useful(self, func):
        """
        从输出开始，反向标记所有有用的计算
        """
        useful = set()

        # 输出是必须有用的
        for output in func.outputs:
            useful.add(output.name)

        # 反向追踪依赖
        changed = True
        while changed:
            changed = False
            for stmt in func.body:
                if stmt.output.name in useful:
                    for input in stmt.inputs:
                        if input.name not in useful:
                            useful.add(input.name)
                            changed = True

        return useful

    def _remove_dead_code(self, func, useful):
        """
        移除不在 useful 集合中的计算
        """
        new_body = []
        for stmt in func.body:
            if stmt.output.name in useful:
                new_body.append(stmt)
            else:
                # 可以安全移除
                pass

        func.body = new_body
        return func
```

死代码消除（DCE）通过“反向标记-正向删除”的两阶段算法移除不影响输出的计算。第一阶段 `_mark_useful` 从函数输出出发，以迭代不动点方式反向追踪依赖链：初始时只有输出变量被标记为“有用”；然后反复遍历所有语句——如果某语句的输出已标记为有用，则将其所有输入也标记为有用；重复直到不再有新变量被标记（`changed = False`）。第二阶段 `_remove_dead_code` 正向遍历语句列表，删除输出不在有用集合中的语句。以典型场景为例：用户定义了一个中间变量 `temp = A * 2` 但后续从未使用 `temp`——标记阶段从输出反向追踪时，不会经过 `temp`，因此 `temp` 永远不会被标记为有用，删除阶段将其安全移除。这种算法的正确性依赖于“无副作用”假设：GPU kernel 中的计算没有隐式的全局状态修改，所有效果都通过显式的输出缓冲区体现。与常量折叠的协同效应在于：常量折叠可能使某些分支的条件变为永远为假，导致分支体成为死代码；因此 DCE 通常在常量折叠之后运行。在实际开发中，DCE 对调试尤为重要——编译时可以通过 IR dump 观察哪些操作被消除了，帮助开发者确认代码逻辑是否按预期被保留。

### 13.5.3 循环优化

```python
class LoopOptimization(tvm.tir.PrimFuncPass):
    """
    循环优化 Pass

    包括：
    1. 循环展开 (Loop Unrolling)
    2. 循环向量化 (Loop Vectorization)
    3. 循环分块 (Loop Tiling)
    4. 循环交换 (Loop Interchange)
    """

    def transform_function(self, func, mod, ctx):
        # 1. 循环展开
        func = self._unroll_loops(func)

        # 2. 循环向量化
        func = self._vectorize_loops(func)

        # 3. 循环分块 (如果尚未分块)
        func = self._tile_loops(func)

        return func

    def _unroll_loops(self, func):
        """
        展开小循环

        规则：循环次数 ≤ 4 的循环自动展开
        """
        for loop in self._find_loops(func):
            if loop.extent <= 4:
                # 展开循环
                unrolled_body = []
                for i in range(loop.extent):
                    substituted = self._substitute(
                        loop.body, loop.loop_var, i
                    )
                    unrolled_body.append(substituted)

                func = self._replace_loop(func, loop, unrolled_body)

        return func

    def _vectorize_loops(self, func):
        """
        向量化连续访问的循环

        规则：连续访问的循环可以向量化
        """
        for loop in self._find_loops(func):
            if self._is_vectorizable(loop):
                # 将循环标记为向量化
                loop = tvm.tir.For(
                    loop_var=loop.loop_var,
                    min_val=loop.min_val,
                    extent=loop.extent,
                    kind=tvm.tir.ForKind.VECTORIZED,
                    body=loop.body,
                )
                func = self._replace_loop(func, loop, loop)

        return func
```

`LoopOptimization` 是 TIR 优化 Passes 中性能收益最大的一个——它集成了四种循环优化技术，按固定顺序执行。循环展开（`_unroll_loops`）是第一步：对于循环次数 ≤ 4 的小循环，将循环体复制展开为顺序语句，消除循环控制开销（比较-跳转指令），并暴露指令级并行（ILP）机会。展开时通过 `_substitute` 将循环变量替换为具体值（0, 1, 2, 3），确保每次迭代的代码是独立的标量操作。循环向量化（`_vectorize_loops`）是第二步：检测连续内存访问模式（如 `for i: C[i] = A[i] + B[i]`），将循环标记为 `VECTORIZED`——这个标记会在 CodeGen 阶段触发向量指令的生成（如 PTX 的 `ld.global.v4.f16` 一次加载 4 个 half）。循环分块（`_tile_loops`）是第三步：将大循环拆分为多层嵌套（外层遍历 tile，内层在 tile 内迭代），以改善缓存局部性——虽然 TileLang 的 Lowering 阶段已执行初步分块，但此处可以基于成本模型进行二次调整。设计上有意将展开放在向量化之前：展开小循环使得循环体变大、提供更多向量化候选；向量化后的连续访问又有利于后续的内存合并优化。实际使用中需注意：过度展开会增加寄存器压力导致 spilling（寄存器溢出到内存），因此在寄存器资源紧张的 kernel 中应谨慎调大展开阈值。

### 13.5.4 内存访问优化

```python
class MemoryAccessOptimization(tvm.tir.PrimFuncPass):
    """
    内存访问优化 Pass

    优化：
    1. 访问合并 (Coalescing)
    2. 预取 (Prefetch)
    3. 缓冲区复用 (Buffer Reuse)
    """

    def transform_function(self, func, mod, ctx):
        # 1. 访问合并优化
        func = self._optimize_coalescing(func)

        # 2. 添加预取指令
        func = self._add_prefetch(func)

        # 3. 缓冲区复用
        func = self._reuse_buffers(func)

        return func

    def _optimize_coalescing(self, func):
        """
        优化内存访问合并

        规则：确保相邻线程访问相邻的内存地址
        """
        for block in self._find_blocks(func):
            for access in self._find_memory_accesses(block):
                if not self._is_coalesced(access):
                    # 尝试变换访问模式以实现合并
                    new_access = self._transform_for_coalescing(access)
                    func = self._replace_access(func, access, new_access)

        return func

    def _add_prefetch(self, func):
        """
        添加数据预取指令

        在计算当前块时，预取下一个块的数据
        """
        for loop in self._find_loops(func):
            if self._should_prefetch(loop):
                # 在循环体开始添加预取
                prefetch = self._create_prefetch(loop, lookahead=1)
                loop.body = tvm.tir.SeqStmt([prefetch, loop.body])
                func = self._replace_loop(func, loop, loop)

        return func
```

内存访问优化是 GPU 性能的关键——GPU 的全局内存带宽虽然高（H100 约 3.35 TB/s），但延迟也大（数百周期），必须通过合并访问和预取来隐藏延迟。访问合并优化（`_optimize_coalescing`）确保同一 warp 内的 32 个线程访问连续的 128 字节对齐的内存地址——GPU 内存控制器可以将这些请求合并为一次宽总线传输。如果检测到未合并的访问模式（如线程按列而非按行访问），则尝试变换访问模式。数据预取（`_add_prefetch`）在前一次迭代的计算期间提前发起下一块数据的加载请求——由于 GPU 的 Load 指令是非阻塞的（只要不立即使用加载结果），计算和传输可以流水线化重叠。预取通过在循环体开头插入 `prefetch` 语句实现，并使用 `lookahead=1` 参数控制预取窗口。缓冲区复用（`_reuse_buffers`）分析缓冲区的活跃区间，将生命周期不重叠的缓冲区复用同一块内存空间——这类似于传统编译器中的寄存器分配，但在共享内存层面操作，可以有效减少共享内存使用量。这三项优化的叠加效果显著：理想情况下，访问合并可提升带宽利用率 4-8 倍，预取可隐藏 50-70% 的访存延迟，缓冲复用可节省 30-50% 的共享内存。需要注意的是，过于激进的缓冲复用可能导致不同 warp 之间的 false sharing，引入难以诊断的性能抖动。

---

## 13.6 阶段 5: CodeGen

<div data-component="CompilationStageVisualizer"></div>

阶段 5 是编译管线中硬件差异最大的环节——它将优化后的、硬件无关的 TIR 翻译为目标平台的代码。CodeGen 的输入是所有 Pass 执行完毕的 TIR（循环已优化、内存已布局、类型已统一），输出则因目标而异：NVIDIA GPU 输出 PTX 汇编（再由 ptxas 编译为 SASS），AMD GPU 输出 HIP C++ 源码（再由 hipcc 编译为 HSACO），华为昇腾输出 Ascend C 源码（再由 Ascend 编译器编译为 NPU 可执行码），CPU 输出 LLVM IR（再由 LLVM 后端编译为目标机器码）。尽管目标各异，所有 CodeGen 都遵循相似的五步结构：生成函数头（声明入口和参数）、分配寄存器/声明共享内存、逐语句翻译计算逻辑、插入必要的同步和跳转指令、生成函数尾。关键挑战在于指令选择——如何将高层 TIR 操作（如一个带有 `VECTORIZED` 标记的 Load）映射为最优的机器指令序列——这通常通过模式匹配或代价模型来实现。

### 13.6.1 NVIDIA PTX CodeGen

```python
class PTXCodeGen(tvm.target.codegen.CodeGenCUDA):
    """
    NVIDIA PTX 代码生成器

    将 TIR 转换为 PTX 汇编代码
    """

    def generate(self, func):
        """
        生成 PTX 代码
        """
        # 1. 生成函数头
        ptx = self._generate_function_header(func)

        # 2. 生成寄存器分配
        ptx += self._allocate_registers(func)

        # 3. 生成共享内存声明
        ptx += self._generate_shared_memory(func)

        # 4. 生成计算代码
        for stmt in func.body:
            ptx += self._generate_statement(stmt)

        # 5. 生成函数尾
        ptx += self._generate_function_footer()

        return ptx

    def _generate_statement(self, stmt):
        """生成单条语句的 PTX 代码"""
        if isinstance(stmt, tvm.tir.Store):
            # 存储操作
            if stmt.buffer.scope == "shared":
                return f"st.shared.{self._get_ptx_type(stmt.dtype)} [{stmt.indices}], {stmt.value}"
            else:
                return f"st.global.{self._get_ptx_type(stmt.dtype)} [{stmt.indices}], {stmt.value}"

        elif isinstance(stmt, tvm.tir.Load):
            # 加载操作
            if stmt.buffer.scope == "shared":
                return f"ld.shared.{self._get_ptx_type(stmt.dtype)} {stmt.var}, [{stmt.indices}]"
            else:
                return f"ld.global.{self._get_ptx_type(stmt.dtype)} {stmt.var}, [{stmt.indices}]"

        elif isinstance(stmt, tvm.tir.Add):
            return f"add.{self._get_ptx_type(stmt.dtype)} {stmt.var}, {stmt.a}, {stmt.b}"

        # ... 其他操作

    def _generate_shared_memory(self, func):
        """生成共享内存声明"""
        ptx = ""
        for buf in func.shared_buffers:
            size = self._compute_size(buf)
            ptx += f".shared .align 4 .b8 {buf.name}[{size}];\n"
        return ptx
```

`PTXCodeGen` 将 TIR 转换为 NVIDIA PTX（Parallel Thread Execution）汇编——一种介于高级 IR 和 GPU 机器码（SASS）之间的中间汇编语言。生成函数分为五个步骤：第一步 `_generate_function_header` 输出 `.visible .entry kernel_name(...)` 声明，`.visible` 表示该 entry 对 host 可见，`.entry` 表示 GPU kernel 入口而非普通函数。第二步分配寄存器——PTX 使用 `.reg` 指令声明虚拟寄存器（如 `.reg .f16 %h<256>` 声明 256 个 f16 寄存器），GPU 硬件在 SASS 编译阶段将其映射到物理寄存器。第三步 `_generate_shared_memory` 输出 `.shared .align 4 .b8 buf_name[size]` 声明共享内存——`.align 4` 确保 16 字节对齐以满足合并访问要求，`.b8` 表示按字节编址。第四步逐语句生成是关键：`tvm.tir.Store` 根据 buffer 作用域生成 `st.shared` 或 `st.global` 指令，`tvm.tir.Load` 同理生成 `ld.shared` 或 `ld.global`，`tvm.tir.Add` 生成 `add` 指令。这种分发式翻译要求 CodeGen 开发者对 PTX 指令集有深入了解——每个 TIR IR 节点至少对应一种 PTX 指令模式。在编译管线中，PTX 是最终在 GPU 上执行前的最后一道“可见”关卡——开发者可以通过 `dump_ptx` 选项查看生成的 PTX，验证关键循环是否按预期展开、寄存器使用是否合理。

PTX 输出示例：

```ptx
// FlashMLA kernel 的 PTX 代码片段
.visible .entry flash_mla(
    .param .u64 Q,
    .param .u64 c_KV,
    .param .u64 W_UK,
    .param .u64 Output
)
{
    .reg .pred  %p<10>;
    .reg .f16   %h<256>;
    .reg .f32   %f<256>;
    .reg .b32   %r<64>;
    .reg .b64   %rd<32>;

    // 共享内存声明
    .shared .align 4 .b8 Q_smem[16384];    // 128 * 64 * 2
    .shared .align 4 .b8 cKV_smem[65536];  // 64 * 512 * 2
    .shared .align 4 .b8 K_smem[16384];    // 64 * 128 * 2

    // 加载 Q 到共享内存
    ld.param.u64    %rd1, [Q];
    cvta.to.global.u64  %rd2, %rd1;
    // ... 加载逻辑 ...

    // 主循环
    LOOP:
        // 加载 cKV tile
        // ... 加载逻辑 ...

        // 上投影 K = cKV @ W_UK
        // ... GEMM 逻辑 ...

        // 注意力计算
        // ... 注意力逻辑 ...

        // 循环控制
        // ... 跳转逻辑 ...

    // 输出
    // ... 输出逻辑 ...
}
```

上述 PTX 代码是 FlashMLA kernel 在 H100 GPU 上的编译产物片段，展示了生成代码的实际结构。第一个关键部分是共享内存声明：`Q_smem[16384]` 对应 Q 矩阵的 128×64×2 字节（float16），`cKV_smem[65536]` 对应 cKV 矩阵的 64×512×2 字节，`K_smem[16384]` 对应上投影后的 K 矩阵。观察这些声明可以发现编译器的内存规划策略：Q、cKV、K 三块共享内存总计约 96KB，在 H100 的 228KB 共享内存限制以内（使用 `maxDynamicSharedMemorySize` 可进一步扩展）。第二个关键部分是加载逻辑：`ld.param.u64 %rd1, [Q]` 从 kernel 参数读取 Q 的全局内存地址到 64 位寄存器，`cvta.to.global.u64 %rd2, %rd1` 将地址转换为全局地址空间（在统一内存架构下这是必要的转换）。第三个关键部分是主循环 LOOP 标签——PTX 使用标签和分支指令实现循环控制，而非 C 语言的 for 循环。从这段 PTX 片段可以反向推断编译器做了哪些优化：如果循环体被完全展开则不会出现 LOOP 标签，如果使用了 TMA 异步拷贝则会出现 `cp.async.bulk` 指令。阅读 PTX 输出是调试 GPU kernel 性能的最高效手段之一——通过数寄存器声明行（`.reg .f16 %h<256>` 表示使用了 up to 256 个 f16 寄存器）可以预估寄存器压力。

### 13.6.2 AMD HSACO CodeGen

```python
class HSACOCodeGen(tvm.target.codegen.CodeGenHIP):
    """
    AMD HSACO 代码生成器

    将 TIR 转换为 HIP/ROCm 代码
    """

    def generate(self, func):
        """
        生成 HIP 代码，然后编译为 HSACO
        """
        # 1. 生成 HIP 源码
        hip_code = self._generate_hip_source(func)

        # 2. 调用 hipcc 编译
        hsaco = self._compile_hip_to_hsaco(hip_code)

        return hsaco

    def _generate_hip_source(self, func):
        """
        生成 HIP 源码

        HIP 与 CUDA 类似，但有以下区别：
        1. 使用 __global__ 而非 __global__
        2. 使用 hipLaunchKernelGGL 而非 <<<>>>
        3. 使用 rocm 对应的 intrinsic
        """
        code = ""
        code += "#include <hip/hip_runtime.h>\n\n"

        # 生成 kernel 函数
        code += "__global__ void flash_mla(...) {\n"

        # 共享内存
        for buf in func.shared_buffers:
            code += f"    __shared__ {self._get_hip_type(buf.dtype)} {buf.name}[{self._compute_size(buf)}];\n"

        # 计算逻辑
        for stmt in func.body:
            code += self._generate_hip_statement(stmt, indent=4)

        code += "}\n"

        return code
```

`HSACOCodeGen` 采用与 PTX CodeGen 不同的两步式策略——先生成 HIP C++ 源码，再调用 `hipcc` 编译为 HSACO（HSA Code Object）。这种设计选择源于 AMD 的工具链生态：ROCm 平台没有类似 PTX 的公开稳定中间汇编层，而是通过 hipcc（基于 Clang）直接从 HIP C++ 编译到 GPU 机器码。HIP 语法与 CUDA 高度相似但有关键差异：kernel 使用 `__global__` 关键字、共享内存使用 `__shared__` 声明、launch 使用 `hipLaunchKernelGGL` 宏而非 `<<<>>>` 语法。这种映射关系使得从 CUDA 后端迁移到 AMD 后端的代码复用率很高——CodeGen 的主体逻辑可以共用，只需在代码模板层面做适配（如头文件从 `<cuda_runtime.h>` 变为 `<hip/hip_runtime.h>`，共享内存关键字从 `__shared__` 保持一致但底层实现不同）。生成 HIP 源码后，hipcc 执行标准的 Clang 编译流程：HIP → LLVM IR → AMDGPU 后端 → HSACO。分两步编译的一个优势是：HIP 源码是人类可读的 C++，开发者可以直接阅读和调试，而 PTX 作为汇编语言对多数开发者不够友好。另一方面，两步编译也增加了编译时间——HIP → HSACO 的编译开销通常高于 PTX → SASS。

### 13.6.3 华为 Ascend C CodeGen

```python
class AscendCCodeGen:
    """
    华为 Ascend C 代码生成器

    将 TIR 转换为 Ascend C 代码（用于昇腾 NPU）
    """

    def generate(self, func):
        """
        生成 Ascend C 代码

        Ascend C 特点：
        1. 使用 AI Core 的向量/矩阵计算单元
        2. 三级内存架构 (L1/L2/UB)
        3. 数据搬运通过 DA (Data Agent) 控制
        """
        code = ""

        # 1. 头文件
        code += '#include "kernel_operator.h"\n\n'

        # 2. Kernel 函数
        code += 'extern "C" __global__ __aicore__ void flash_mla(...) {\n'

        # 3. 数据搬运定义
        code += self._generate_data_mover(func)

        # 4. 计算逻辑
        code += self._generate_compute_logic(func)

        code += '}\n'

        return code

    def _generate_data_mover(self, func):
        """
        生成数据搬运代码

        Ascend C 使用 GM → L1 → UB 的搬运模式
        """
        code = ""
        for buf in func.shared_buffers:
            if buf.scope == "L1":
                code += f"    TPipe pipe;\n"
                code += f"    TBuf<TPosition::CO1> {buf.name}_buf;\n"
                code += f"    LocalTensor<{self._get_ascend_type(buf.dtype)}> {buf.name} = {buf.name}_buf.Get<{self._get_ascend_type(buf.dtype)}>();\n"
            elif buf.scope == "UB":
                code += f"    TBuf<TPosition::VECCALC> {buf.name}_buf;\n"
                code += f"    LocalTensor<{self._get_ascend_type(buf.dtype)}> {buf.name} = {buf.name}_buf.Get<{self._get_ascend_type(buf.dtype)}>();\n"
        return code
```

华为昇腾 NPU 的编程模型与 GPU 有根本性不同，因此 `AscendCCodeGen` 的结构与其他 CodeGen 差异最大。昇腾 NPU 采用“AI Core + 三级内存”架构：每个 AI Core 内部有 Cube Unit（矩阵计算单元，类似 NVIDIA Tensor Core）和 Vector Unit（向量计算单元），内存层次分为 GM（Global Memory，HBM）、L1（L1 Buffer，在 AI Core 之间共享）和 UB（Unified Buffer，AI Core 独占的近存）。`_generate_data_mover` 方法生成数据搬运代码——Ascend C 使用 `TPipe` 和 `TBuf` 抽象来描述数据在三级内存之间的流动，`DataCopy` 或 `DataCopyPad` 指令负责实际搬运。`LocalTensor` 是对 UB 中张量的引用，类型参数 `TPosition::CO1` 表示位于 Cube Unit 的输入位置。与 CUDA 的 `__shared__` 声明式内存模型不同，Ascend C 的内存管理是显式指令式的——开发者（或编译器）需要精确指定每块数据在哪个时刻位于哪一级内存，这虽然增加了编程复杂度，但提供了对数据流最精细的控制，在 attention 等算子中可以实现接近硬件理论峰值的效率。在编译管线中，Ascend C CodeGen 的输出是 C++ 源码（而非汇编），需要经华为的 Ascend 编译器（基于自研 TVM 后端）进一步编译为 NPU 可执行码。

### 13.6.4 CPU LLVM CodeGen

```python
class LLVMCodeGen(tvm.target.codegen.CodeGenLLVM):
    """
    CPU LLVM 代码生成器

    用于 CPU 上的 TileLang kernel
    """

    def generate(self, func):
        """
        生成 LLVM IR
        """
        # 1. 创建 LLVM 模块
        module = llvm.Module("tilelang_module")

        # 2. 生成函数
        llvm_func = self._generate_function(func, module)

        # 3. 优化
        module = self._optimize_llvm(module)

        return module

    def _optimize_llvm(self, module):
        """
        应用 LLVM 优化 Pass
        """
        pass_manager = llvm.PassManager()

        # 标准优化
        pass_manager.add(llvm.create_instruction_combining_pass())
        pass_manager.add(llvm.create_reassociate_pass())
        pass_manager.add(llvm.create_gvn_pass())
        pass_manager.add(llvm.create_simplify_cfg_pass())

        # 循环优化
        pass_manager.add(llvm.create_loop_rotate_pass())
        pass_manager.add(llvm.create_loop_unroll_pass())
        pass_manager.add(llvm.create_loop_vectorize_pass())

        # 向量化优化
        pass_manager.add(llvm.create_slp_vectorize_pass())

        pass_manager.run(module)

        return module
```

`LLVMCodeGen` 将 TIR 翻译为 LLVM IR——这是 CPU 后端和多平台支持的基础。生成流程是先创建 LLVM 模块（`llvm.Module("tilelang_module")`），然后在模块内生成函数，最后通过 LLVM 的 Pass Manager 应用标准优化管道。`_optimize_llvm` 方法展示了 LLVM 优化 Pass 的典型组合：指令合并（`InstructionCombining`）消除冗余的算术/逻辑操作；重关联（`Reassociate`）重新排列表达式树以暴露公共子表达式；全局值编号（`GVN`）消除冗余的 Load 和计算；控制流简化（`SimplifyCFG`）合并基本块、消除不可达代码。循环优化层面的 Pass 包括循环旋转（`LoopRotate`——将 while 循环转换为 do-while 以便后续优化）、循环展开（`LoopUnroll`）和循环向量化（`LoopVectorize`——利用 CPU SIMD 指令如 AVX-512）。SLP 向量化（`SLPVectorize`）则将多个独立的标量操作打包为向量操作。这种设计体现了编译管线的一个重要设计选择：复用了 LLVM 生态的成熟优化基础设施，避免在 TIR 层面重新实现已有的优化。但代价是 LLVM 优化管道主要针对 CPU 设计（如分支预测友好的控制流优化），对于 GPU kernel 的优化效果可能不如手写的 GPU 专用优化。在 TileLang 中，LLVM 后端主要用于 CPU 上的 kernel 调试和测试——在 GPU 上开发和验证逻辑正确性之前，可以先用 CPU 后端运行并对比结果。

---

## 13.7 阶段 6: 链接与可执行文件

阶段 6 是编译管线的收尾环节——它将 CodeGen 阶段产生的一个或多个目标代码文件链接为可被 GPU 驱动加载的单一可执行文件。输入可能是多个独立的模块（例如主 kernel 和 epilogue kernel 分开编译），输出是平台特定的可执行格式（CUDA 为 cubin/fatbin，AMD 为 HSACO，升腾为 NPU 可执行文件）。链接过程不仅合并符号表，还可能执行跨模块优化——例如将多个 kernel 的共享内存布局对齐以减少加载时的内存碎片。对于不支持链接的目标平台，此阶段简化为包装和序列化。

### 13.7.1 模块链接

```python
class ModuleLinker:
    """
    模块链接器

    将多个编译单元链接为一个可执行文件
    """

    def link(self, modules, target):
        """
        链接多个模块
        """
        if target.kind.name == "cuda":
            return self._link_cuda(modules)
        elif target.kind.name == "rocm":
            return self._link_rocm(modules)
        elif target.kind.name == "llvm":
            return self._link_llvm(modules)

    def _link_cuda(self, modules):
        """
        链接 CUDA 模块

        使用 nvlink 或 fatbin 链接多个 .cubin 文件
        """
        # 1. 收集所有 PTX/cubin
        ptx_modules = []
        cubin_modules = []
        for mod in modules:
            if mod.format == "ptx":
                ptx_modules.append(mod)
            elif mod.format == "cubin":
                cubin_modules.append(mod)

        # 2. 编译 PTX 到 cubin
        for ptx_mod in ptx_modules:
            cubin = self._compile_ptx_to_cubin(ptx_mod)
            cubin_modules.append(cubin)

        # 3. 链接所有 cubin
        linked = self._nvlink(cubin_modules)

        return linked
```

这段代码是 13.7.1 模块链接 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 13.7.2 运行时加载

```python
class RuntimeLoader:
    """
    运行时加载器

    将编译好的模块加载到 GPU 并准备执行
    """

    def load(self, executable, device):
        """
        加载可执行文件
        """
        if device.type == "cuda":
            return self._load_cuda(executable, device)
        elif device.type == "rocm":
            return self._load_rocm(executable, device)

    def _load_cuda(self, executable, device):
        """
        加载 CUDA 可执行文件

        步骤：
        1. 加载 cubin 到 GPU
        2. 获取 kernel 函数指针
        3. 设置 kernel 参数
        4. 创建 CUDA stream
        """
        # 1. 加载 cubin
        module = cuda.module_from_buffer(executable.cubin)

        # 2. 获取 kernel
        kernel = module.get_function("flash_mla")

        # 3. 创建执行器
        executor = CUDAExecutor(kernel, device)

        return executor
```

这段代码是 13.7.2 运行时加载 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 13.8 编译选项与调试

### 13.8.1 编译选项

```python
# TileLang 编译选项

compile_options = {
    # 基本选项
    "target": "cuda -arch=sm_90",          # 目标设备
    "opt_level": 3,                         # 优化级别 (0-3)

    # TileLang 特定选项
    "tile_size": [128, 128, 32],           # 默认 tile 大小
    "num_threads": 256,                     # 默认线程数
    "enable_shared_memory": True,           # 启用共享内存
    "enable_prefetch": True,                # 启用数据预取

    # 调试选项
    "dump_ir": True,                        # dump IR
    "dump_ptx": True,                       # dump PTX
    "profile": True,                        # 启用 profiling

    # 优化选项
    "enable_constant_folding": True,        # 启用常量折叠
    "enable_dead_code_elimination": True,   # 启用死代码消除
    "enable_loop_unrolling": True,          # 启用循环展开
    "enable_vectorization": True,           # 启用向量化
}

# 使用编译选项
compiled = tilelang.compile(
    my_kernel,
    target=compile_options["target"],
    options=compile_options,
)
```

这段代码是 13.8.1 编译选项 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 13.8.2 调试 Flag

```python
# 调试 Flag 使用

# 1. IR Dump
export TILELANG_DUMP_IR=1              # dump 所有中间 IR
export TILELANG_DUMP_IR_DIR=./ir_dump  # dump 目录

# 2. PTX Dump
export TILELANG_DUMP_PTX=1             # dump PTX 代码
export TILELANG_DUMP_PTX_DIR=./ptx_dump

# 3. 性能分析
export TILELANG_PROFILE=1              # 启用 profiling
export TILELANG_PROFILE_DIR=./profile  # profile 输出目录

# 4. 详细日志
export TILELANG_LOG_LEVEL=DEBUG        # 日志级别
export TILELANG_LOG_FILE=./tilelang.log # 日志文件

# 5. 编译验证
export TILELANG_VERIFY=1               # 启用编译后验证
```

这段代码是 13.8.2 调试 Flag 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 13.8.3 IR Dump 示例

```python
# 使用 Python API 进行 IR dump

import tilelang
from tilelang import ir

# 编译并 dump 所有阶段的 IR
compiled = tilelang.compile(
    my_kernel,
    target="cuda",
    dump_ir=True,
    ir_dump_dir="./ir_dump",
)

# 手动 dump 特定阶段
print("=== TileLang IR ===")
print(ir.dump_tilelang(my_kernel))

print("\n=== TensorIR ===")
tensorir = ir.lower_tilelang(my_kernel)
print(tensorir)

print("\n=== TIR (Optimized) ===")
tir = ir.optimize(tensorir)
print(tir)

print("\n=== PTX ===")
ptx = ir.codegen(tir, target="cuda")
print(ptx)
```

这段代码是 13.8.3 IR Dump 示例 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 13.9 实战：编译一个完整的 Kernel

### 13.9.1 示例：FlashMLA 编译全流程

```python
import tilelang
from tilelang import T, ir
import tvm

# 1. 定义 TileLang kernel
@T.prim_func
def flash_mla(
    Q: T.Buffer[(batch, heads, d_model), "float16"],
    c_KV: T.Buffer[(batch, seq_len, d_compress), "float16"],
    W_UK: T.Buffer[(heads, d_compress, d_head), "float16"],
    Output: T.Buffer[(batch, heads, d_model), "float16"],
):
    with T.Kernel(heads, batch, threads=256) as (hid, bid):
        Q_smem = T.alloc_shared([64, d_head], "float16")
        cKV_smem = T.alloc_shared([64, d_compress], "float16")
        UK_smem = T.alloc_shared([d_compress, d_head], "float16")
        K_tile = T.alloc_shared([64, d_head], "float16")
        S_local = T.alloc_fragment([64, 64], "float32")
        O_local = T.alloc_fragment([64, d_head], "float32")

        T.copy(Q[bid, hid, :], Q_smem)
        T.copy(W_UK[hid, :, :], UK_smem)

        for j in T.serial(seq_len // 64):
            T.copy(c_KV[bid, j*64:(j+1)*64, :], cKV_smem)
            T.gemm(cKV_smem, UK_smem, K_tile)
            T.gemm(Q_smem, K_tile, S_local, transpose_B=True)
            T.online_softmax(S_local, O_local, ...)

        T.copy(O_local, Output[bid, hid, :])

# 2. 查看各阶段 IR
print("=" * 60)
print("阶段 1: TileLang IR")
print("=" * 60)
tilelang_ir = ir.dump_tilelang(flash_mla)
print(tilelang_ir)

print("\n" + "=" * 60)
print("阶段 2: TensorIR (Lowered)")
print("=" * 60)
tensorir = ir.lower_tilelang(flash_mla)
print(tensorir)

print("\n" + "=" * 60)
print("阶段 3: TIR (Optimized)")
print("=" * 60)
tir = ir.optimize(tensorir)
print(tir)

print("\n" + "=" * 60)
print("阶段 4: PTX 代码")
print("=" * 60)
ptx = ir.codegen(tir, target="cuda")
print(ptx[:2000])  # 只打印前 2000 行

# 3. 编译并执行
compiled = tilelang.compile(flash_mla, target="cuda")

# 4. 性能测试
import torch
Q = torch.randn(batch, heads, d_model, dtype=torch.float16, device="cuda")
c_KV = torch.randn(batch, seq_len, d_compress, dtype=torch.float16, device="cuda")
W_UK = torch.randn(heads, d_compress, d_head, dtype=torch.float16, device="cuda")

output = compiled(Q, c_KV, W_UK)
print(f"\nOutput shape: {output.shape}")
```

这段代码是 13.9.1 示例：FlashMLA 编译全流程 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 13.9.2 编译时间分析

```python
import time

def profile_compilation(kernel_func):
    """
    分析各阶段的编译时间
    """
    times = {}

    # 阶段 1: 解析
    start = time.time()
    tilelang_ir = ir.parse(kernel_func)
    times["parse"] = time.time() - start

    # 阶段 2: Lowering
    start = time.time()
    tensorir = ir.lower_tilelang(tilelang_ir)
    times["lowering"] = time.time() - start

    # 阶段 3: 优化
    start = time.time()
    tir = ir.optimize(tensorir)
    times["optimization"] = time.time() - start

    # 阶段 4: CodeGen
    start = time.time()
    ptx = ir.codegen(tir, target="cuda")
    times["codegen"] = time.time() - start

    # 阶段 5: 链接
    start = time.time()
    executable = ir.link(ptx)
    times["linking"] = time.time() - start

    # 打印结果
    total = sum(times.values())
    print("编译时间分析:")
    print(f"{'阶段':<20} {'时间 (ms)':<15} {'占比':<10}")
    print("-" * 45)
    for stage, t in times.items():
        print(f"{stage:<20} {t*1000:<15.2f} {t/total*100:<10.1f}%")
    print("-" * 45)
    print(f"{'总计':<20} {total*1000:<15.2f} 100.0%")

    return times
```

这段代码是 13.9.2 编译时间分析 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 13.10 常见编译问题与解决

### 13.10.1 编译错误

```
常见编译错误及解决方法：

错误 1: Shared memory limit exceeded
  原因: 共享内存使用超出 GPU 限制
  解决: 减小 tile 大小，或使用动态共享内存

错误 2: Thread block size too large
  原因: 线程块大小超出限制 (1024)
  解决: 减少线程数或重新设计 kernel

错误 3: Bank conflict detected
  原因: 共享内存访问模式导致 bank conflict
  解决: 使用 swizzle 或 padding 优化

错误 4: Register spilling
  原因: 寄存器使用超出限制
  解决: 减小 tile 大小或使用 __launch_bounds__

错误 5: Divergent branching
  原因: warp 内线程执行不同分支
  解决: 重构代码减少分支分歧
```

这个代码块或示意图用于说明 13.10.1 编译错误 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 13.10.2 性能调优

```python
# 编译时性能调优

# 1. 选择合适的 tile 大小
tile_configs = [
    {"tile_m": 64, "tile_n": 64, "tile_k": 32},
    {"tile_m": 128, "tile_n": 64, "tile_k": 32},
    {"tile_m": 64, "tile_n": 128, "tile_k": 32},
    {"tile_m": 128, "tile_n": 128, "tile_k": 32},
]

# 2. 自动调优
from tilelang import autotune

@autotune(
    configs=tile_configs,
    keys=["tile_m", "tile_n", "tile_k"],
    warmup=10,
    rep=100,
)
def optimized_kernel(tile_m=64, tile_n=64, tile_k=32):
    @T.prim_func
    def kernel(A, B, C):
        with T.Kernel(...) as (bx, by):
            # ... 使用 tile_m, tile_n, tile_k ...
            pass
    return kernel
```

这段代码是 13.10.2 性能调优 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 13.11 总结

### 编译管线速查表

<div data-component="FullCompilationPipelineDiagram"></div>

```
编译管线速查：

阶段 1: Python DSL
  输入: @T.prim_func 装饰的 Python 函数
  输出: Python AST
  关键: 类型注解、T.Buffer、T.Kernel

阶段 2: TileLang IR
  输入: Python AST
  输出: TileLang IR (Tile/Region/Block)
  关键: 高层抽象、计算语义

阶段 3: TensorIR
  输入: TileLang IR
  输出: TensorIR (Buffer/Block/Loop)
  Pass: TileLangLowerTileOp, TileLangInferLayout
  关键: 循环结构、内存管理

阶段 4: TIR (优化后)
  输入: TensorIR
  输出: 优化的 TIR
  Pass: 常量折叠、死代码消除、循环优化
  关键: 性能优化

阶段 5: 目标代码
  输入: 优化的 TIR
  输出: PTX/HSACO/Ascend C/LLVM IR
  关键: 硬件特定优化

阶段 6: 可执行文件
  输入: 目标代码
  输出: .cubin/.hsaco/.o
  关键: 链接、加载
```

这个代码块或示意图用于说明 编译管线速查表 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

---

## 练习

### 基础练习

1. **IR 观察**：编写一个简单的向量加法 kernel，使用 `ir.dump_tilelang()` 查看每个阶段的 IR。

2. **编译选项**：实验不同的编译选项（如 tile 大小、线程数），观察对编译结果的影响。

3. **错误诊断**：故意编写一个超出共享内存限制的 kernel，观察编译错误信息。

### 进阶练习

4. **自定义 Pass**：实现一个简单的 TIR Pass，将所有乘以 2 的操作替换为左移 1 位。

5. **CodeGen 扩展**：为一个新的硬件后端（如 Intel GPU）实现简单的代码生成器。

6. **编译优化**：对比不同优化级别（O0、O1、O2、O3）的编译结果，分析性能差异。

---

## 思考题

1. **编译器设计**：为什么 TileLang 选择多阶段编译而不是单阶段？各阶段的抽象层次有什么意义？

2. **优化策略**：常量折叠、死代码消除、循环优化这三种优化，哪种对 GPU kernel 性能影响最大？为什么？

3. **硬件适配**：不同硬件（NVIDIA GPU、AMD GPU、华为 NPU）的 CodeGen 有什么根本差异？能否统一？

---

## 13.12 编译器优化的理论基础

### 13.12.1 数据流分析

```python
# 数据流分析基础

"""
数据流分析是编译器优化的基础:

1. 到达定义分析 (Reaching Definitions)
   - 分析每个程序点有哪些定义可能到达
   - 用于常量传播、死代码消除

2. 活跃变量分析 (Live Variable Analysis)
   - 分析每个程序点哪些变量是活跃的
   - 用于寄存器分配

3. 可用表达式分析 (Available Expressions)
   - 分析每个程序点哪些表达式已经计算过
   - 用于公共子表达式消除
"""

class DataFlowAnalyzer:
    """
    数据流分析器
    """

    def reaching_definitions(self, cfg):
        """
        到达定义分析

        对于每个程序点，计算可能到达的定义集合
        """
        # 初始化
        for block in cfg.blocks:
            block.gen = set()   # 生成的定义
            block.kill = set()  # 杀死的定义

        # 迭代计算
        changed = True
        while changed:
            changed = False
            for block in cfg.blocks:
                # 计算输入: 所有前驱的输出的并集
                block.input = set()
                for pred in block.predecessors:
                    block.input |= pred.output

                # 计算输出: (输入 - 杀死) ∪ 生成
                new_output = (block.input - block.kill) | block.gen
                if new_output != block.output:
                    block.output = new_output
                    changed = True

        return cfg

    def live_variable_analysis(self, cfg):
        """
        活跃变量分析

        对于每个程序点，计算哪些变量是活跃的
        """
        # 初始化
        for block in cfg.blocks:
            block.defs = set()    # 定义的变量
            block.uses = set()    # 使用的变量

        # 迭代计算 (反向)
        changed = True
        while changed:
            changed = False
            for block in reversed(cfg.blocks):
                # 计算输出: 所有后继的输入的并集
                block.live_out = set()
                for succ in block.successors:
                    block.live_out |= succ.live_in

                # 计算输入: 使用 ∪ (输出 - 定义)
                new_live_in = block.uses | (block.live_out - block.defs)
                if new_live_in != block.live_in:
                    block.live_in = new_live_in
                    changed = True

        return cfg
```

这段代码是 13.12.1 数据流分析 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 13.12.2 支配树分析

```python
class DominatorTree:
    """
    支配树分析

    支配关系: 如果从入口到节点 n 的所有路径都经过节点 d，
    则 d 支配 n (d dom n)
    """

    def compute(self, cfg):
        """
        计算支配树
        """
        # 初始化
        all_nodes = set(cfg.blocks)
        for node in all_nodes:
            node.dom = all_nodes.copy()

        cfg.entry.dom = {cfg.entry}

        # 迭代计算
        changed = True
        while changed:
            changed = False
            for node in all_nodes - {cfg.entry}:
                new_dom = {node}
                for pred in node.predecessors:
                    new_dom &= pred.dom
                new_dom |= {node}

                if new_dom != node.dom:
                    node.dom = new_dom
                    changed = True

        # 构建支配树
        self._build_dominator_tree(cfg)

        return cfg

    def _build_dominator_tree(self, cfg):
        """
        构建支配树
        """
        for node in cfg.blocks:
            # 直接支配者 (idom)
            # 是 node.dom 中除了 node 自身外最接近 node 的节点
            node.idom = self._find_idom(node)
```

这段代码是 13.12.2 支配树分析 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 13.12.3 循环分析

```python
class LoopAnalyzer:
    """
    循环分析器
    """

    def find_natural_loops(self, cfg):
        """
        查找自然循环

        自然循环的条件:
        1. 有唯一的入口节点
        2. 从循环内任何节点都能到达入口
        """
        loops = []

        # 查找回边 (back edge)
        for block in cfg.blocks:
            for succ in block.successors:
                if self._dominates(succ, block):
                    # block -> succ 是回边
                    loop = self._find_natural_loop(succ, block)
                    loops.append(loop)

        return loops

    def _find_natural_loop(self, header, tail):
        """
        查找自然循环的节点集合
        """
        loop_nodes = {header, tail}

        # 从 tail 开始反向遍历
        worklist = [tail]
        while worklist:
            node = worklist.pop()
            for pred in node.predecessors:
                if pred not in loop_nodes:
                    loop_nodes.add(pred)
                    worklist.append(pred)

        return {
            "header": header,
            "nodes": loop_nodes,
        }
```

这段代码是 13.12.3 循环分析 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 13.13 高级优化技术

### 13.13.1 循环不变量外提

```python
class LoopInvariantCodeMotion(tvm.tir.PrimFuncPass):
    """
    循环不变量外提 (LICM)

    将循环内不变的计算移到循环外
    """

    def transform_function(self, func, mod, ctx):
        # 1. 查找所有循环
        loops = self._find_loops(func)

        for loop in loops:
            # 2. 分析循环不变量
            invariants = self._find_invariants(loop)

            # 3. 将不变量移到循环外
            if invariants:
                func = self._hoist_invariants(func, loop, invariants)

        return func

    def _find_invariants(self, loop):
        """
        查找循环不变量

        循环不变量的条件:
        1. 定义在循环外
        2. 或者定义的操作数都是循环不变量
        """
        invariants = set()
        changed = True

        while changed:
            changed = False
            for stmt in loop.body:
                if self._is_invariant(stmt, invariants):
                    if stmt not in invariants:
                        invariants.add(stmt)
                        changed = True

        return invariants

    def _is_invariant(self, stmt, known_invariants):
        """
        判断语句是否是循环不变量
        """
        if isinstance(stmt, tvm.tir.Store):
            # 检查所有操作数
            for operand in self._get_operands(stmt):
                if not self._is_loop_invariant(operand, known_invariants):
                    return False
            return True
        return False
```

这段代码是 13.13.1 循环不变量外提 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 13.13.2 强度削减

```python
class StrengthReduction(tvm.tir.PrimFuncPass):
    """
    强度削减

    将昂贵的操作替换为等价的便宜操作
    例: 乘法 → 移位 + 加法
    """

    def transform_function(self, func, mod, ctx):
        # 查找可以强度削减的操作
        for stmt in self._find_statements(func):
            if isinstance(stmt, tvm.tir.Mul):
                # 检查是否是 2 的幂次乘法
                if self._is_power_of_two(stmt.b):
                    # 替换为移位
                    shift_amount = int(math.log2(stmt.b))
                    func = self._replace_statement(
                        func, stmt,
                        tvm.tir.LeftShift(stmt.a, shift_amount)
                    )

            elif isinstance(stmt, tvm.tir.Div):
                # 检查是否是 2 的幂次除法
                if self._is_power_of_two(stmt.b):
                    # 替换为右移
                    shift_amount = int(math.log2(stmt.b))
                    func = self._replace_statement(
                        func, stmt,
                        tvm.tir.RightShift(stmt.a, shift_amount)
                    )

        return func

    def _is_power_of_two(self, expr):
        """检查是否是 2 的幂次"""
        if isinstance(expr, tvm.tir.IntImm):
            return expr.value > 0 and (expr.value & (expr.value - 1)) == 0
        return False
```

这段代码是 13.13.2 强度削减 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 13.13.3 向量化优化

```python
class VectorizationOptimization(tvm.tir.PrimFuncPass):
    """
    向量化优化

    将标量操作转换为向量操作
    """

    def transform_function(self, func, mod, ctx):
        # 1. 查找可以向量化的循环
        vectorizable_loops = self._find_vectorizable_loops(func)

        # 2. 对每个循环进行向量化
        for loop in vectorizable_loops:
            func = self._vectorize_loop(func, loop)

        return func

    def _find_vectorizable_loops(self, func):
        """
        查找可以向量化的循环

        向量化的条件:
        1. 循环体没有依赖
        2. 访问模式是连续的
        3. 循环次数是向量宽度的倍数
        """
        vectorizable = []

        for loop in self._find_loops(func):
            if self._check_vectorizable(loop):
                vectorizable.append(loop)

        return vectorizable

    def _vectorize_loop(self, func, loop):
        """
        向量化循环

        将:
          for i in range(N):
            C[i] = A[i] + B[i]

        转换为:
          for i in range(N / VECTOR_WIDTH):
            C[i*VW:(i+1)*VW] = A[i*VW:(i+1)*VW] + B[i*VW:(i+1)*VW]
        """
        vector_width = self._get_vector_width()
        new_body = self._create_vector_body(loop.body, vector_width)
        new_loop = tvm.tir.For(
            loop.loop_var,
            0,
            loop.extent // vector_width,
            tvm.tir.ForKind.SERIAL,
            new_body,
        )
        return self._replace_loop(func, loop, new_loop)
```

这段代码是 13.13.3 向量化优化 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 13.14 Target-Specific 优化

### 13.14.1 NVIDIA GPU 特定优化

```python
class NVIDIAOptimizer:
    """
    NVIDIA GPU 特定优化
    """

    def optimize(self, tir, arch):
        """
        针对特定 NVIDIA 架构的优化
        """
        if arch >= "sm_80":  # Ampere (A100)
            tir = self._optimize_for_ampere(tir)
        if arch >= "sm_90":  # Hopper (H100)
            tir = self._optimize_for_hopper(tir)

        return tir

    def _optimize_for_ampere(self, tir):
        """
        Ampere 架构优化

        特性:
        - 异步内存拷贝
        - 共享内存异步屏障
        - TF32 支持
        """
        # 添加异步拷贝指令
        tir = self._insert_async_copy(tir)

        # 添加异步屏障
        tir = self._insert_async_barrier(tir)

        return tir

    def _optimize_for_hopper(self, tir):
        """
        Hopper 架构优化

        特性:
        - TMA (Tensor Memory Accelerator)
        - 异步事务屏障
        - FP8 支持
        - 分布式共享内存
        """
        # 使用 TMA 进行数据搬运
        tir = self._use_tma(tir)

        # 使用异步事务屏障
        tir = self._use_async_transaction_barrier(tir)

        # FP8 支持
        tir = self._use_fp8(tir)

        return tir
```

这段代码是 13.14.1 NVIDIA GPU 特定优化 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 13.14.2 AMD GPU 特定优化

```python
class AMDGPUOptimizer:
    """
    AMD GPU 特定优化
    """

    def optimize(self, tir, arch):
        """
        针对特定 AMD 架构的优化
        """
        if arch == "gfx90a":  # MI200
            tir = self._optimize_for_mi200(tir)
        elif arch == "gfx942":  # MI300
            tir = self._optimize_for_mi300(tir)

        return tir

    def _optimize_for_mi200(self, tir):
        """
        MI200 架构优化

        特性:
        - Matrix Core 支持
        - 64-wide wavefront
        - Infinity Fabric 互联
        """
        # 使用 Matrix Core
        tir = self._use_matrix_core(tir)

        # 优化 wavefront 利用率
        tir = self._optimize_wavefront(tir)

        return tir

    def _optimize_for_mi300(self, tir):
        """
        MI300 架构优化

        特性:
        - 统一内存架构
        - 更多计算单元
        - FP8 支持
        """
        # 利用统一内存
        tir = self._use_unified_memory(tir)

        # FP8 支持
        tir = self._use_fp8(tir)

        return tir
```

这段代码是 13.14.2 AMD GPU 特定优化 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 13.14.3 华为昇腾特定优化

```python
class AscendOptimizer:
    """
    华为昇腾 NPU 特定优化
    """

    def optimize(self, tir, chip):
        """
        针对特定昇腾芯片的优化
        """
        if chip == "910b":
            tir = self._optimize_for_910b(tir)

        return tir

    def _optimize_for_910b(self, tir):
        """
        910B 架构优化

        特性:
        - Cube Unit (矩阵计算)
        - Vector Unit (向量计算)
        - 三级内存层次
        - Data Agent 控制
        """
        # 优化数据搬运
        tir = self._optimize_data_movement(tir)

        # 使用 Cube Unit
        tir = self._use_cube_unit(tir)

        # 内存层次优化
        tir = self._optimize_memory_hierarchy(tir)

        return tir
```

这段代码是 13.14.3 华为昇腾特定优化 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 13.15 编译缓存与增量编译

### 13.15.1 编译缓存

```python
class CompilationCache:
    """
    编译缓存系统
    避免重复编译相同的 kernel
    """

    def __init__(self, cache_dir="./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = {}

    def get(self, kernel_func, target):
        """
        从缓存获取编译结果
        """
        # 计算缓存键
        cache_key = self._compute_cache_key(kernel_func, target)

        # 检查内存缓存
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 检查磁盘缓存
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                compiled = pickle.load(f)
                self.cache[cache_key] = compiled
                return compiled

        return None

    def put(self, kernel_func, target, compiled):
        """
        将编译结果放入缓存
        """
        cache_key = self._compute_cache_key(kernel_func, target)

        # 内存缓存
        self.cache[cache_key] = compiled

        # 磁盘缓存
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(compiled, f)

    def _compute_cache_key(self, kernel_func, target):
        """
        计算缓存键

        基于:
        1. 函数源码的 hash
        2. 目标设备信息
        3. 编译选项
        """
        import hashlib

        # 获取函数源码
        source = inspect.getsource(kernel_func)

        # 计算 hash
        content = f"{source}:{target}"
        return hashlib.md5(content.encode()).hexdigest()
```

这段代码是 13.15.1 编译缓存 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 13.15.2 增量编译

```python
class IncrementalCompiler:
    """
    增量编译器

    只重新编译修改过的部分
    """

    def __init__(self):
        self.previous_ir = None
        self.previous_compiled = None

    def compile(self, kernel_func, target):
        """
        增量编译
        """
        # 解析当前 IR
        current_ir = ir.parse(kernel_func)

        if self.previous_ir is None:
            # 首次编译
            compiled = self._full_compile(current_ir, target)
        else:
            # 计算差异
            diff = self._compute_diff(self.previous_ir, current_ir)

            if diff.is_empty():
                # 没有变化，使用之前的编译结果
                compiled = self.previous_compiled
            elif diff.is_minor():
                # 小变化，增量编译
                compiled = self._incremental_compile(
                    self.previous_compiled, diff, target
                )
            else:
                # 大变化，完全重新编译
                compiled = self._full_compile(current_ir, target)

        # 更新状态
        self.previous_ir = current_ir
        self.previous_compiled = compiled

        return compiled

    def _compute_diff(self, old_ir, new_ir):
        """
        计算两个 IR 的差异
        """
        diff = IRDiff()

        # 比较函数签名
        if old_ir.signature != new_ir.signature:
            diff.add_significant_change("signature")

        # 比较循环结构
        old_loops = self._extract_loops(old_ir)
        new_loops = self._extract_loops(new_ir)

        if len(old_loops) != len(new_loops):
            diff.add_significant_change("loop_count")
        else:
            for old, new in zip(old_loops, new_loops):
                if old.extent != new.extent:
                    diff.add_minor_change("loop_extent")

        return diff
```

这段代码是 13.15.2 增量编译 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 13.16 错误处理与诊断

### 13.16.1 编译错误诊断

```python
class CompilationErrorDiagnostics:
    """
    编译错误诊断系统
    """

    def diagnose(self, error, kernel_func, target):
        """
        诊断编译错误
        """
        diagnostics = []

        # 1. 分析错误类型
        error_type = self._classify_error(error)

        # 2. 提供诊断信息
        if error_type == "shared_memory_exceeded":
            diagnostics.append(self._diagnose_shared_memory(error, kernel_func))
        elif error_type == "thread_block_too_large":
            diagnostics.append(self._diagnose_thread_block(error, kernel_func))
        elif error_type == "register_spilling":
            diagnostics.append(self._diagnose_register_spilling(error, kernel_func))
        elif error_type == "bank_conflict":
            diagnostics.append(self._diagnose_bank_conflict(error, kernel_func))

        # 3. 提供修复建议
        suggestions = self._generate_suggestions(diagnostics)

        return {
            "error_type": error_type,
            "diagnostics": diagnostics,
            "suggestions": suggestions,
        }

    def _diagnose_shared_memory(self, error, kernel_func):
        """诊断共享内存问题"""
        # 分析共享内存使用
        shared_buffers = self._find_shared_buffers(kernel_func)

        total_size = 0
        for buf in shared_buffers:
            size = self._compute_buffer_size(buf)
            total_size += size

        limit = self._get_shared_memory_limit()

        return {
            "issue": "Shared memory exceeded",
            "used": total_size,
            "limit": limit,
            "overshoot": total_size - limit,
            "buffers": shared_buffers,
        }

    def _generate_suggestions(self, diagnostics):
        """生成修复建议"""
        suggestions = []

        for diag in diagnostics:
            if diag["issue"] == "Shared memory exceeded":
                suggestions.extend([
                    f"减小 tile 大小 (当前使用 {diag['used']} bytes)",
                    f"使用 FP8 量化减少内存",
                    f"检查是否有不必要的共享内存分配",
                ])
            elif diag["issue"] == "Thread block too large":
                suggestions.extend([
                    "减少线程数到 1024 以下",
                    "考虑使用多个较小的线程块",
                ])

        return suggestions
```

这段代码是 13.16.1 编译错误诊断 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 13.16.2 运行时错误诊断

```python
class RuntimeErrorDiagnostics:
    """
    运行时错误诊断系统
    """

    def diagnose(self, error, kernel_func, inputs):
        """
        诊断运行时错误
        """
        diagnostics = []

        # 1. 分析错误类型
        error_type = self._classify_error(error)

        # 2. 提供诊断信息
        if error_type == "illegal_memory_access":
            diagnostics.append(
                self._diagnose_memory_access(error, kernel_func, inputs)
            )
        elif error_type == "nan_detected":
            diagnostics.append(
                self._diagnose_nan(error, kernel_func, inputs)
            )
        elif error_type == "incorrect_result":
            diagnostics.append(
                self._diagnose_incorrect_result(error, kernel_func, inputs)
            )

        return {
            "error_type": error_type,
            "diagnostics": diagnostics,
        }

    def _diagnose_memory_access(self, error, kernel_func, inputs):
        """诊断内存访问错误"""
        # 分析输入形状
        input_shapes = {name: tensor.shape for name, tensor in inputs.items()}

        # 检查边界条件
        boundary_issues = self._check_boundary_conditions(
            kernel_func, input_shapes
        )

        return {
            "issue": "Illegal memory access",
            "input_shapes": input_shapes,
            "boundary_issues": boundary_issues,
        }
```

这段代码是 13.16.2 运行时错误诊断 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 13.17 编译器验证

### 13.17.1 正确性验证

```python
class CompilationVerifier:
    """
    编译器正确性验证
    """

    def verify(self, original_func, compiled_func, test_inputs):
        """
        验证编译后的函数与原始函数行为一致
        """
        results = {}

        for inputs in test_inputs:
            # 运行原始函数
            expected = self._run_reference(original_func, inputs)

            # 运行编译后的函数
            actual = compiled_func(*inputs)

            # 比较结果
            match = self._compare_results(expected, actual)

            results[str(inputs)] = {
                "match": match,
                "max_error": self._compute_max_error(expected, actual),
            }

        return results

    def _compare_results(self, expected, actual, rtol=1e-3, atol=1e-3):
        """比较两个结果是否一致"""
        import numpy as np
        return np.allclose(expected, actual, rtol=rtol, atol=atol)
```

这段代码是 13.17.1 正确性验证 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 13.17.2 性能回归测试

```python
class PerformanceRegressionTest:
    """
    性能回归测试

    确保编译器优化不会导致性能下降
    """

    def __init__(self, baseline_results):
        self.baseline = baseline_results

    def test(self, kernel_func, target, tolerance=0.1):
        """
        测试性能是否回归

        tolerance: 允许的性能下降百分比
        """
        # 编译并测量性能
        compiled = tilelang.compile(kernel_func, target=target)
        current_perf = self._measure_performance(compiled)

        # 与基线比较
        baseline_perf = self.baseline.get(self._get_kernel_name(kernel_func))

        if baseline_perf is None:
            # 没有基线，记录当前性能
            return {"status": "no_baseline", "performance": current_perf}

        # 计算性能变化
        perf_change = (current_perf - baseline_perf) / baseline_perf

        if perf_change > tolerance:
            # 性能回归
            return {
                "status": "regression",
                "baseline": baseline_perf,
                "current": current_perf,
                "change": perf_change,
            }
        else:
            # 性能正常
            return {
                "status": "ok",
                "baseline": baseline_perf,
                "current": current_perf,
                "change": perf_change,
            }
```

这段代码是 13.17.2 性能回归测试 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 扩展阅读

1. **LLVM 编译器设计**：理解现代编译器的架构和 Pass 管理
2. **GPU 编程指南**：深入理解 GPU 的内存层次和执行模型
3. **TVM CodeGen 文档**：TVM 的代码生成框架和扩展方法
4. **PTX 手册**：NVIDIA PTX 指令集参考

---

## 下一章预告

> **Chapter 14: Thread Binding 与硬件线程映射**
>
> 在理解了完整的编译管线之后，下一章我们将深入探讨 Thread Binding——如何将 Tile 维度映射到硬件线程。这是影响 GPU kernel 性能的关键因素。我们将涵盖：
>
> - Thread Binding 概念与原理
> - CUDA/Warp/Thread 映射
> - AMD Wavefront 映射
> - 昇腾 AI Core 映射
> - 最优 Thread Binding 策略
