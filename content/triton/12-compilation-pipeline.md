---
title: "Chapter 12: 编译管线全景——从 Python 到机器码"
description: "深入理解 Triton 的完整编译管线，从 Python DSL 源码到 PTX/HSACO 机器码的六阶段变换过程，掌握每个阶段的 IR 形式、关键 Pass 和源码位置。"
date: "2026-06-11"
---

# Chapter 12: 编译管线全景——从 Python 到机器码

> **学习目标**：
> - 理解 Triton 编译管线的完整六阶段架构：Python DSL → AST → Triton IR → TritonGPU Dialect → LLVM Dialect → PTX/HSACO
> - 掌握 `@triton.jit` 装饰器如何通过 `inspect.getsource` 和 `ast.parse` 实现 Python 源码捕获与 AST 重写
> - 理解代码生成器如何将 Python AST 节点映射为 Triton IR 指令
> - 掌握 TritonGPU Dialect 的 layout encoding 注入与硬件特化过程
> - 了解完整的 MLIR Pass 管线及其各 Pass 的作用
> - 能够使用调试工具（TRITON_PRINT_IR、IR dump）追踪和检查编译中间结果

---

## 12.1 编译管线总览

Triton 的核心创新之一是将 Python DSL 代码自动编译为高效的 GPU 机器码。整个编译过程经历六个主要阶段，每一阶段都产生不同层次的中间表示（IR），逐步从高级抽象降低到硬件特定的机器指令。

### 12.1.1 六阶段管线架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Triton 编译管线（6 阶段）                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  阶段 1        阶段 2          阶段 3            阶段 4              │
│  Python DSL    Python AST      Triton IR         TritonGPU Dialect  │
│  ─────────►   ─────────►     ─────────►        ─────────►          │
│  @triton.jit   ast.parse      code_gen          make_triton_ir      │
│  源码捕获       AST 重写       IR 生成           Layout 注入         │
│                                                                     │
│  阶段 5                阶段 6                                       │
│  LLVM Dialect          PTX / HSACO                                  │
│  ─────────►           ─────────►                                   │
│  TritonGPUToLLVM       NVVM/ROCM                                   │
│  Dialect Lowering       机器码生成                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 12.1.2 各阶段产物与关键源码

| 阶段 | 输入 | 输出 | 关键源码文件 |
|:---|:---|:---|:---|
| 1. 源码捕获 | Python 函数 | Python AST | `python/triton/runtime/jit.py` |
| 2. AST → Triton IR | Python AST | Triton IR (MLIR) | `python/triton/compiler/code_generator.py` |
| 3. Triton IR → TritonGPU | Triton IR | TritonGPU Dialect | `lib/Dialect/TritonGPU/Transforms/TritonGPUConversion.cpp` |
| 4. GPU 优化 Pass | TritonGPU Dialect | 优化后 TritonGPU | `lib/Dialect/TritonGPU/Transforms/*.cpp` |
| 5. Lowering to LLVM | TritonGPU Dialect | LLVM Dialect | `lib/Conversion/TritonGPUToLLVM/*.cpp` |
| 6. 代码生成 | LLVM Dialect | PTX / HSACO | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/` |

### 12.1.3 演示用例：vector_add kernel

本章将追踪以下简单 kernel 在各编译阶段的 IR 形式：

```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)
```

调用方式：

```python
import torch

N = 1024 * 1024
a = torch.randn(N, device='cuda', dtype=torch.float32)
b = torch.randn(N, device='cuda', dtype=torch.float32)
c = torch.empty_like(a)

BLOCK_SIZE = 1024
grid = (triton.cdiv(N, BLOCK_SIZE),)
vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
```

---

## 12.2 阶段一：Python → AST

编译的第一步是捕获 Python 函数的源代码并解析为抽象语法树（AST）。这是 Triton JIT 编译的入口。

### 12.2.1 `@triton.jit` 装饰器的工作原理

`@triton.jit` 装饰器定义在 `python/triton/runtime/jit.py` 中。当一个函数被 `@triton.jit` 装饰时，装饰器不会立即编译函数，而是创建一个 `JITFunction` 对象：

```python
# python/triton/runtime/jit.py（简化）
class JITFunction:
    def __init__(self, fn):
        self.fn = fn
        self._src = None

    @property
    def src(self):
        if self._src is None:
            self._src = inspect.getsource(self.fn)
        return self._src
```

当 kernel 通过 `kernel_name[grid](args...)` 语法被调用时，触发 `__getitem__` → `__call__` 链，最终进入编译流程。

### 12.2.2 源码捕获：`inspect.getsource`

Triton 使用 Python 标准库的 `inspect.getsource()` 获取函数的源代码文本。这意味着：

1. **函数必须定义在 `.py` 文件中**——交互式环境（如 Jupyter notebook 的某些情况下）可能无法获取源码
2. **源码以文本形式捕获**——函数尚未被"执行"，而是被"分析"
3. **行号信息被保留**——用于后续的错误报告和调试

```python
import inspect

# inspect.getsource 返回函数的完整源码文本
source = inspect.getsource(vector_add_kernel)
print(source)
# 输出：
# @triton.jit
# def vector_add_kernel(
#     a_ptr, b_ptr, c_ptr,
#     n_elements,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     pid = tl.program_id(0)
#     ...
```

### 12.2.3 AST 解析与重写

获取源码后，Triton 使用 Python 内置的 `ast.parse()` 将其解析为 AST，然后进行一系列重写变换：

```python
# python/triton/compiler/code_generator.py（简化）
import ast

tree = ast.parse(source)
```

**关键 AST 重写操作**：

| 重写操作 | 目的 | 示例 |
|:---|:---|:---|
| constexpr 处理 | 将类型标注为 `tl.constexpr` 的参数标记为编译期常量 | `BLOCK_SIZE: tl.constexpr` → 编译期替换 |
| type annotation 处理 | 解析函数参数的类型注解 | `a_ptr` → `tl.pointer_type(tl.float32)` |
| 内联展开 | 将 `tl.` 前缀的调用映射为 IR 操作 | `tl.load(...)` → `tt.load` |
| 控制流标准化 | 将 Python 的 `if/for/while` 转换为显式控制流 | `for i in range(...)` → `scf.for` |

**AST 节点类型与对应处理**：

```python
# code_generator.py 中的 visit 模式（简化）
class CodeGenerator(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
        # 处理函数定义，提取参数和类型注解
        ...

    def visit_Assign(self, node):
        # 处理赋值语句：pid = tl.program_id(0)
        ...

    def visit_Call(self, node):
        # 处理函数调用：tl.load(), tl.store(), tl.arange()
        ...

    def visit_BinOp(self, node):
        # 处理二元运算：a + b, offsets < n_elements
        ...

    def visit_If(self, node):
        # 处理条件语句
        ...

    def visit_For(self, node):
        # 处理循环：for i in range(...)
        ...
```

### 12.2.4 vector_add 的 AST 表示

对于我们的 `vector_add_kernel`，`ast.parse` 生成的 AST 结构如下（简化表示）：

```
Module
 └─ FunctionDef(name="vector_add_kernel")
     ├─ args:
     │   ├─ arg(a_ptr)
     │   ├─ arg(b_ptr)
     │   ├─ arg(c_ptr)
     │   ├─ arg(n_elements)
     │   └─ arg(BLOCK_SIZE, annotation=tl.constexpr)
     └─ body:
         ├─ Assign(pid, Call(tl.program_id, 0))
         ├─ Assign(block_start, BinOp(pid, Mult, BLOCK_SIZE))
         ├─ Assign(offsets, BinOp(block_start, Add, Call(tl.arange, 0, BLOCK_SIZE)))
         ├─ Assign(mask, Compare(offsets, Lt, n_elements))
         ├─ Assign(a, Call(tl.load, a_ptr+offsets, mask=mask))
         ├─ Assign(b, Call(tl.load, b_ptr+offsets, mask=mask))
         ├─ Assign(c, BinOp(a, Add, b))
         └─ Expr(Call(tl.store, c_ptr+offsets, c, mask=mask))
```

### 12.2.5 使用 Python ast 模块查看实际 AST

我们可以直接用 Python 的 `ast` 模块打印实际的 AST 节点：

```python
import ast
import inspect

source = inspect.getsource(vector_add_kernel)
tree = ast.parse(source)
print(ast.dump(tree, indent=2))
```

输出的关键部分（简化）：

```
Module(
  body=[
    FunctionDef(
      name='vector_add_kernel',
      args=arguments(
        posonlyargs=[],
        args=[
          arg(arg='a_ptr'),
          arg(arg='b_ptr'),
          arg(arg='c_ptr'),
          arg(arg='n_elements'),
          arg(arg='BLOCK_SIZE',
              annotation=Attribute(
                value=Name(id='tl'),
                attr='constexpr'))
        ]),
      body=[
        Assign(
          targets=[Name(id='pid')],
          value=Call(
            func=Attribute(value=Name(id='tl'), attr='program_id'),
            args=[Constant(value=0)])),
        Assign(
          targets=[Name(id='block_start')],
          value=BinOp(
            left=Name(id='pid'),
            op=Mult(),
            right=Name(id='BLOCK_SIZE'))),
        Assign(
          targets=[Name(id='offsets')],
          value=BinOp(
            left=Name(id='block_start'),
            op=Add(),
            right=Call(
              func=Attribute(value=Name(id='tl'), attr='arange'),
              args=[Constant(value=0), Name(id='BLOCK_SIZE')]))),
        Assign(
          targets=[Name(id='mask')],
          value=Compare(
            left=Name(id='offsets'),
            ops=[Lt()],
            comparators=[Name(id='n_elements')])),
        Assign(
          targets=[Name(id='a')],
          value=Call(
            func=Attribute(value=Name(id='tl'), attr='load'),
            args=[BinOp(left=Name(id='a_ptr'), op=Add(), right=Name(id='offsets'))],
            keywords=[keyword(arg='mask', value=Name(id='mask'))])),
        ...
      ])
  ])
```

### 12.2.6 constexpr 参数的特殊处理

`tl.constexpr` 类型标注在 AST 阶段被特殊处理。代码生成器识别到 `BLOCK_SIZE` 的注解为 `tl.constexpr` 后，会：

1. 将该参数标记为"编译期常量"
2. 在编译时直接用实际值（如 1024）替换所有出现的位置
3. 该参数不会出现在生成的 Triton IR 中

```python
# code_generator.py 中的 constexpr 处理（简化）
def visit_FunctionDef(self, node):
    for arg in node.args.args:
        if is_constexpr_annotation(arg.annotation):
            # 标记为编译期常量
            self.constexpr_args.add(arg.arg)
        else:
            # 生成 IR 函数参数
            ir_arg = self._create_ir_arg(arg)
            self.fn_args.append(ir_arg)
```

**为什么 constexpr 很重要**：

| 方面 | 使用 constexpr | 不使用 constexpr |
|:---|:---|:---|
| 张量形状 | `tl.zeros([BLOCK], ...)` 可在编译期确定 | 无法创建固定形状张量 |
| 循环展开 | `range(0, K, BLOCK_K)` 可完全展开 | 循环次数未知，无法展开 |
| 寄存器分配 | 编译器知道精确的寄存器需求 | 必须假设最坏情况 |
| 性能 | 最优 | 可能差 2-10 倍 |

<div data-component="ASTViewer"></div>

[组件：ASTViewer - 交互式 Python AST 可视化，展示 vector_add_kernel 的完整 AST 树结构]

---

## 12.3 阶段二：AST → Triton IR

代码生成器（`code_generator.py`）遍历 Python AST，将每个节点转换为对应的 Triton IR 指令。这是从 Python 语义到 MLIR 方言的关键桥梁。

### 12.3.1 代码生成器架构

代码生成器的核心是 `CodeGenerator` 类，它继承自 `ast.NodeVisitor`，通过 visitor 模式遍历 AST 的每个节点：

```python
# python/triton/compiler/code_generator.py（架构简化）
class CodeGenerator(ast.NodeVisitor):
    def __init__(self, context, prototype, options):
        self.builder = ir.builder(context)  # MLIR builder
        self.module = self.builder.create_module()
        self.fn = None
        self.scf_stack = []  # 控制流栈

    def generate(self, tree):
        """主入口：遍历 AST 生成 Triton IR"""
        self.visit(tree)
        return self.module
```

### 12.3.2 关键 AST → IR 映射规则

下表列出了 Python 代码元素到 Triton IR 指令的映射关系：

| Python 代码 | Triton IR 指令 | 说明 |
|:---|:---|:---|
| `tl.program_id(0)` | `tt.get_program_id x` | 获取 program ID |
| `tl.arange(0, N)` | `tt.make_range <N>` | 创建索引张量 |
| `ptr + offsets` | `tt.addptr %ptr, %offsets` | 指针偏移计算 |
| `tl.load(ptr, mask=m)` | `tt.load %ptr, %mask` | 条件加载 |
| `tl.store(ptr, val, mask=m)` | `tt.store %ptr, %val, %mask` | 条件存储 |
| `a + b`（tensor） | `arith.addf %a, %b` | 逐元素加法 |
| `offsets < n` | `arith.cmpi slt, %offsets, %n` | 逐元素比较 |
| `tl.constexpr` 参数 | 编译期常量替换 | 不出现在 IR 中 |

### 12.3.3 vector_add 生成的 Triton IR

经过代码生成器处理后，`vector_add_kernel` 被转换为以下 Triton IR：

```mlir
// 阶段 2 输出：Triton IR（ttir）
module attributes {"triton_gpu.num-warps" = 4 : i32,
                    "triton_gpu.num-ctas" = 1 : i32,
                    "triton_gpu.compute-capability" = 80 : i32} {
  tt.func @vector_add_kernel(
    %arg0: !tt.ptr<f32>,
    %arg1: !tt.ptr<f32>,
    %arg2: !tt.ptr<f32>,
    %arg3: i32
  ) attributes {noinline = false} {
    // pid = tl.program_id(0)
    %0 = tt.get_program_id x : i32

    // block_start = pid * BLOCK_SIZE (BLOCK_SIZE=1024，constexpr 已替换)
    %c1024_i32 = arith.constant 1024 : i32
    %1 = arith.muli %0, %c1024_i32 : i32

    // offsets = block_start + tl.arange(0, 1024)
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : (i32) -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>

    // mask = offsets < n_elements
    %5 = tt.splat %arg3 : (i32) -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>

    // a = tl.load(a_ptr + offsets, mask=mask)
    %7 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %9 = tt.load %8, %6 : tensor<1024xf32>

    // b = tl.load(b_ptr + offsets, mask=mask)
    %10 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %12 = tt.load %11, %6 : tensor<1024xf32>

    // c = a + b
    %13 = arith.addf %9, %12 : tensor<1024xf32>

    // tl.store(c_ptr + offsets, c, mask=mask)
    %14 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %15, %13, %6 : tensor<1024xf32>

    tt.return
  }
}
```

### 12.3.4 Triton IR 的关键特性

**1. 类型系统**

Triton IR 使用 MLIR 的类型系统，主要类型包括：

| 类型 | 含义 | 示例 |
|:---|:---|:---|
| `!tt.ptr<T>` | 指向类型 T 的指针 | `!tt.ptr<f32>` |
| `tensor<NxT>` | N 元素的一维张量 | `tensor<1024xf32>` |
| `i32` / `f32` | 标量整数/浮点数 | `i32`, `f32` |
| `tensor<Nx!tt.ptr<T>>` | 指针张量 | `tensor<1024x!tt.ptr<f32>>` |

**2. `tt.splat` 操作**

`splat` 将标量广播为同形状的张量，是 Triton IR 中的高频操作。在 vector_add 中，它将标量指针和标量 `n_elements` 广播为 1024 元素的张量：

```mlir
// 标量指针 → 指针张量
%7 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>

// 标量 i32 → i32 张量
%5 = tt.splat %arg3 : (i32) -> tensor<1024xi32>
```

**3. 逐元素操作**

在 Triton IR 中，算术运算（`arith.addf`、`arith.addi`、`arith.cmpi`）直接作用于张量，表示逐元素运算：

```mlir
// 逐元素浮点加法
%13 = arith.addf %9, %12 : tensor<1024xf32>

// 逐元素整数比较
%6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
```

### 12.3.5 属性标注

Triton IR 的 `module` 级别包含重要的编译配置属性：

```mlir
module attributes {
  "triton_gpu.num-warps" = 4 : i32,        // 每个 CTA 的 warp 数
  "triton_gpu.num-ctas" = 1 : i32,         // 每个 CTA 的数量（cluster）
  "triton_gpu.compute-capability" = 80 : i32  // GPU 计算能力
}
```

这些属性在后续的 TritonGPU lowering 阶段被用于硬件特化。

<div data-component="IRStageViewer"></div>

[组件：IRStageViewer - 交互式 IR 查看器，展示 Triton IR 与源码的逐行对应关系]

---

## 12.4 阶段三：Triton IR → TritonGPU Dialect

这一阶段将硬件无关的 Triton IR 转换为硬件感知的 TritonGPU Dialect，注入 GPU 特定的 layout encoding 和并行化信息。

### 12.4.0 MLIR 方言系统基础

Triton 的编译管线建立在 MLIR（Multi-Level Intermediate Representation）框架之上。理解 MLIR 的核心概念对掌握 Triton 编译至关重要。

**MLIR 的核心概念**：

| 概念 | 含义 | Triton 中的示例 |
|:---|:---|:---|
| **Dialect** | 方言，一组相关的操作和类型 | `tt`（Triton）、`triton_gpu`、`arith` |
| **Operation** | 方言中的原子操作 | `tt.load`、`arith.addf` |
| **Type** | 值的类型 | `tensor<1024xf32>`、`!tt.ptr<f32>` |
| **Attribute** | 操作或类型的元数据 | `#blocked`、`{start = 0, end = 1024}` |
| **Pass** | IR 变换规则 | `TritonGPUCombineOps` |
| **Pattern** | 模式匹配与重写 | 将 `tt.dot` 替换为 Tensor Core 指令 |

**Triton 使用的主要方言**：

```
┌─────────────────────────────────────────────────────────┐
│                  Triton 的 MLIR 方言层次                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  tt (Triton Dialect)                                    │
│  ├── tt.load, tt.store, tt.dot, tt.make_range          │
│  └── 高级张量操作，硬件无关                               │
│                                                         │
│  triton_gpu (TritonGPU Dialect)                         │
│  ├── triton_gpu.convert_layout                          │
│  ├── triton_gpu.local_alloc, triton_gpu.local_load      │
│  └── 硬件感知的张量布局操作                               │
│                                                         │
│  arith (Arithmetic Dialect)                             │
│  ├── arith.addf, arith.mulf, arith.cmpi                │
│  └── 标量和张量算术运算                                  │
│                                                         │
│  scf (Structured Control Flow)                          │
│  ├── scf.for, scf.if, scf.yield                         │
│  └── 结构化控制流                                        │
│                                                         │
│  llvm (LLVM Dialect)                                    │
│  ├── llvm.load, llvm.store, llvm.inline_asm             │
│  └── 低级 LLVM IR 表示                                  │
│                                                         │
│  nvvm / rocdl (GPU 方言)                                 │
│  ├── nvvm.barrier0, nvvm.read.ptx.sreg                  │
│  └── GPU 硬件特定操作                                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**MLIR 的 Pass 管理机制**：

```cpp
// Triton 的 Pass Pipeline 定义（简化）
// lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp

void pipeline::addTritonGPUOptimizationPasses(
    mlir::OpPassManager &pm, const TritonGPUOptions &options) {
    // 添加优化 Pass，按顺序执行
    pm.addPass(mlir::triton::gpu::createTritonGPUAccelerateMatmul());
    pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeDotOperands());
    pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
    pm.addPass(mlir::triton::gpu::createTritonGPUCombineOps());
    pm.addPass(mlir::triton::gpu::createTritonGPUPrefetch());
    // ... 更多 Pass
}
```

### 12.4.1 转换入口：`make_triton_ir()` 与 `ttir_to_ttgir()`

编译管线的驱动代码位于 `python/triton/compiler/compiler.py`：

```python
# python/triton/compiler/compiler.py（简化）
def compile(src, target=None, options=None):
    # 阶段 1-2：Python → Triton IR
    metadata, module = make_triton_ir(src, target, options)

    # 阶段 3：Triton IR → TritonGPU Dialect
    module = ttir_to_ttgir(module, options)

    # 阶段 4：GPU 优化 Pass
    module = optimize_ttgir(module, options)

    # 阶段 5-6：LLVM → PTX/HSACO
    ptx, metadata = ttgir_to_llir(module, options)
    return ptx, metadata
```

### 12.4.2 Layout Encoding 概念

TritonGPU Dialect 的核心是 **layout encoding**，它描述了张量数据在 GPU 硬件上的物理分布方式。主要的 encoding 类型：

| Encoding 类型 | 含义 | 使用场景 |
|:---|:---|:---|
| `#blocked` | 按连续块分配到同一个 warp | 通用计算，load/store |
| `#sliced` | 从 blocked encoding 派生的降维切片 | reduce 操作的中间结果 |
| `#shared` | SMEM 布局 | 共享内存中的数据 |
| `#dot_op` | 矩阵乘法操作数布局 | `tt.dot` 的输入 |
| `#mma` / `#wmma` | Tensor Core 硬件原生布局 | NVIDIA / AMD 矩阵核心 |

### 12.4.3 Blocked Encoding 详解

对于我们的 `vector_add_kernel`，张量被标记为 `#blocked` encoding：

```mlir
// blocked encoding 定义
#blocked = #triton_gpu.blocked<{sizePerThread = [4],
                                threadsPerWarp = [32],
                                warpsPerCTA = [4],
                                order = [0]}>
```

各字段含义：

| 字段 | 值 | 含义 |
|:---|:---|:---|
| `sizePerThread` | `[4]` | 每个线程处理 4 个元素 |
| `threadsPerWarp` | `[32]` | 每个 warp 有 32 个线程 |
| `warpsPerCTA` | `[4]` | 每个 CTA 有 4 个 warp |
| `order` | `[0]` | 维度遍历顺序（最内层优先） |

总元素数 = 4 × 32 × 4 = 512 或 1024（取决于 tile size 配置）。

### 12.4.4 vector_add 的 TritonGPU IR

经过 layout 注入后，IR 形式如下：

```mlir
// 阶段 3 输出：TritonGPU Dialect
#blocked = #triton_gpu.blocked<{sizePerThread = [4],
                                threadsPerWarp = [32],
                                warpsPerCTA = [4],
                                order = [0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1,
                              order = [0], hasLeadingOffset = false}>

module attributes {"triton_gpu.num-warps" = 4 : i32,
                    "triton_gpu.num-ctas" = 1 : i32,
                    "triton_gpu.compute-capability" = 80 : i32} {
  tt.func @vector_add_kernel(
    %arg0: !tt.ptr<f32>,
    %arg1: !tt.ptr<f32>,
    %arg2: !tt.ptr<f32>,
    %arg3: i32
  ) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32
    %c1024_i32 = arith.constant 1024 : i32
    %1 = arith.muli %0, %c1024_i32 : i32

    // tt.make_range 现在携带 layout encoding
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32}
        : tensor<1024xi32, #blocked>

    %3 = tt.splat %1 : (i32) -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>

    %5 = tt.splat %arg3 : (i32) -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked>

    %7 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>, #blocked>,
                             tensor<1024xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<1024xf32, #blocked>

    %10 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>, #blocked>,
                               tensor<1024xi32, #blocked>
    %12 = tt.load %11, %6 : tensor<1024xf32, #blocked>

    %13 = arith.addf %9, %12 : tensor<1024xf32, #blocked>

    %14 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>, #blocked>,
                               tensor<1024xi32, #blocked>
    tt.store %15, %13, %6 : tensor<1024xf32, #blocked>

    tt.return
  }
}
```

**关键变化**：每个张量类型现在都附加了 `#blocked` layout encoding，告诉编译器如何将张量元素映射到线程。

### 12.4.5 更复杂 kernel 的 Layout 差异

对于包含矩阵乘法的 kernel，layout 转换更加复杂：

```mlir
// 矩阵乘法 kernel 中的不同 layout
#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4],
                                threadsPerWarp = [2, 16],
                                warpsPerCTA = [4, 1],
                                order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0,
                                warpsPerCTA = [2, 2],
                                instrShape = [16, 8]}>

// load 结果使用 #blocked layout
%A_blocked = tt.load ... : tensor<128x32xf32, #blocked>

// 转换为 #mma layout 供 Tensor Core 使用
%A_mma = triton_gpu.convert_layout %A_blocked
    : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #mma>

// Tensor Core 矩阵乘法（使用 #mma layout）
%C = tt.dot %A_mma, %B_mma, %C_init
    : tensor<128x32xf32, #mma> * tensor<32x128xf32, #mma> -> tensor<128x128xf32, #mma>
```

<div data-component="LayoutVisualizer"></div>

[组件：LayoutVisualizer - 交互式 layout encoding 可视化，展示不同 encoding 下数据在线程间的分布模式]

---

## 12.5 阶段四：TritonGPU Dialect 优化 Pass 管线

获得带 layout encoding 的 TritonGPU Dialect 后，编译器运行一系列优化 Pass 来提升性能。这是 Triton 性能优化的核心阶段。

### 12.5.1 完整 Pass List

以下是 TritonGPU 优化管线中的主要 Pass（按执行顺序排列）：

| 序号 | Pass 名称 | 作用 | 源码位置 |
|:---|:---|:---|:---|
| 1 | `TritonGPUCombineOps` | 合并相邻的布局转换操作 | `lib/Dialect/TritonGPU/Transforms/Combine.cpp` |
| 2 | `TritonGPURemoveLayoutConversions` | 消除冗余的 layout 转换 | `lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp` |
| 3 | `TritonGPUOptimizeDotOperands` | 优化矩阵乘法操作数的布局 | `lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp` |
| 4 | `TritonGPUAccelerateMatmul` | 将通用 dot 提升为 Tensor Core 操作 | `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp` |
| 5 | `TritonGPUPrefetch` | 预取下一次迭代的数据 | `lib/Dialect/TritonGPU/Transforms/Prefetch.cpp` |
| 6 | `TritonGPUOptimizeAccumulatorInit` | 优化累加器初始化 | `lib/Dialect/TritonGPU/Transforms/OptimizeAccumulatorInit.cpp` |
| 7 | `TritonGPURewriteTensorPointer` | 重写张量指针为块指针 | `lib/Dialect/TritonGPU/Transforms/RewriteTensorPointer.cpp` |
| 8 | `TritonGPUUtility` | 工具性变换（canonicalize 等） | `lib/Dialect/TritonGPU/Transforms/Utility.cpp` |

### 12.5.2 Pass 1：CombineOps

`TritonGPUCombineOps` 合并相邻且可以融合的布局转换操作，减少不必要的数据搬运：

```mlir
// 优化前：两次独立的 convert_layout
%a_blocked = triton_gpu.convert_layout %a : tensor<128xf32, #blocked> -> tensor<128xf32, #shared>
%b_blocked = triton_gpu.convert_layout %b : tensor<128xf32, #blocked> -> tensor<128xf32, #shared>

// 优化后：如果两次转换的源和目标 layout 相同，可以共享转换逻辑
```

### 12.5.3 Pass 2：RemoveLayoutConversions

这是最重要的优化 Pass 之一。它通过数据流分析，发现可以避免的 layout 转换：

```mlir
// 优化前：不必要的 layout 转换
%x = triton_gpu.convert_layout %a : ... #blocked -> #shared
%y = some_op %x  // 结果仍然是 #shared
%z = triton_gpu.convert_layout %y : ... #shared -> #blocked

// 如果 %z 的使用者可以直接接受 #shared layout，则两次 convert_layout 都可以消除
```

核心算法：
1. 对每个 `convert_layout` 操作，分析其使用者是否可以接受源 layout
2. 如果所有使用者都可以接受，消除该 `convert_layout`
3. 迭代直到没有更多可消除的转换

### 12.5.4 Pass 3：OptimizeDotOperands

Tensor Core 操作对输入布局有严格要求。此 Pass 优化 `tt.dot` 操作数的布局，减少不必要的数据重排：

```mlir
// 优化前：在 dot 之前做显式 convert_layout
%A_cvt = triton_gpu.convert_layout %A : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #mma>
%C = tt.dot %A_cvt, %B, %C_init

// 优化后：将 convert_layout 融入 dot 操作的 loading 阶段
// 使用 shared memory 作为中介，避免直接的 register-to-register 转换
%A_smem = triton_gpu.local_alloc %A : tensor<128x32xf32, #blocked> -> !triton_gpu.memdesc<128x32xf32, #shared>
%A_dot = triton_gpu.local_load %A_smem : !triton_gpu.memdesc<128x32xf32, #shared> -> tensor<128x32xf32, #mma>
%C = tt.dot %A_dot, %B, %C_init
```

### 12.5.5 Pass 4：AccelerateMatmul

此 Pass 将通用的 `tt.dot` 操作提升为 Tensor Core 硬件原生操作：

```mlir
// 优化前：通用矩阵乘法
%C = tt.dot %A, %B, %C_init : tensor<128x32xf32> * tensor<32x128xf32> -> tensor<128x128xf32>

// 优化后（NVIDIA）：使用 MMA v2 指令
// 自动将 f32 操作数转换为 f16（如果选项允许）
%A_cvt = arith.truncf %A : tensor<128x32xf32> -> tensor<128x32xf16>
%B_cvt = arith.truncf %B : tensor<32x128xf32> -> tensor<32x128xf16>
%C = tt.dot %A_cvt, %B_cvt, %C_init
    : tensor<128x32xf16, #mma> * tensor<32x128xf16, #mma> -> tensor<128x128xf32, #mma>
```

**MMA 版本选择**：

| GPU 架构 | 默认 MMA 版本 | 支持的数据类型 |
|:---|:---|:---|
| SM70 (V100) | mma v1 | f16 |
| SM80 (A100) | mma v2 | f16, bf16, tf32, int8 |
| SM89 (L40S) | mma v2 | + fp8 (e4m3, e5m2) |
| SM90 (H100) | mma v3 (WGMMA) | + fp8, 64-bit 累加 |

### 12.5.6 Pass 5：Prefetch

`TritonGPUPrefetch` 为矩阵乘法循环添加数据预取，将 load 与计算重叠：

```mlir
// 优化前：顺序执行
scf.for %i = %lb to %ub step %step {
    %A = tt.load %a_ptr    // 等待加载
    %B = tt.load %b_ptr    // 等待加载
    %C = tt.dot %A, %B     // 计算
}

// 优化后：预取下一次迭代的数据
%A0 = tt.load %a_ptr_init   // 预加载第一次迭代的数据
%B0 = tt.load %b_ptr_init
scf.for %i = %lb to %ub step %step {
    %A_next = tt.load %a_ptr_next  // 预取下一次迭代
    %B_next = tt.load %b_ptr_next
    %C = tt.dot %A0, %B0          // 使用之前预取的数据计算
    %A0 = %A_next                  // 更新为下一次迭代的数据
    %B0 = %B_next
}
```

### 12.5.7 Pass 6-8：其他优化

**OptimizeAccumulatorInit**：优化矩阵乘法累加器的初始化模式，避免不必要的 zero-fill。

```mlir
// 优化前：显式初始化累加器
%init = arith.constant dense<0.0> : tensor<128x128xf32, #mma>
%C = scf.for ... iter_args(%acc = %init) {
    ...
}

// 优化后：利用 tt.dot 的累加语义，延迟初始化
// 在某些情况下，可以将 zero-fill 移出循环
```

**RewriteTensorPointer**：将块指针（block pointer）操作重写为传统的指针+偏移计算，为后续 LLVM lowering 做准备。

```mlir
// 优化前：使用高级块指针 API（tl.make_block_pointer）
%block_ptr = tt.make_block_pointer [%base, %offset],
    shape=[128, 32], strides=[%stride_m, %stride_k],
    offsets=[%off_m, %off_k], block_shape=[BLOCK_M, BLOCK_K],
    order=[1, 0]
%data = tt.load %block_ptr

// 优化后：展开为传统指针+偏移计算
%ptrs = tt.splat %base : (!tt.ptr<f32>) -> tensor<128x32x!tt.ptr<f32>>
%offsets = ... // 计算每个元素的偏移
%masked_ptrs = tt.addptr %ptrs, %offsets
%data = tt.load %masked_ptrs
```

**Utility**：提供通用的辅助变换，包括 canonicalization、dead code elimination 等。

### 12.5.8 共享内存 Bank Conflict 避免

在矩阵乘法 kernel 中，共享内存的访问模式对性能至关重要。Triton 编译器通过 swizzling 技术避免 bank conflict：

```mlir
// 共享内存 encoding 中的 swizzling
#shared = #triton_gpu.shared<{
    vec = 8,              // 向量化访问宽度
    perPhase = 2,         // 每个 phase 的行数
    maxPhase = 4,         // 最大 phase 数
    order = [1, 0],       // 维度顺序
    hasLeadingOffset = true  // 是否有 leading offset（SM90+）
}>
```

**Bank conflict 的原理**：

```
共享内存有 32 个 bank，每个 bank 4 字节
连续的 4 字节地址映射到连续的 bank

Thread 0 读取 SMEM[0]   → Bank 0
Thread 1 读取 SMEM[4]   → Bank 1
...
Thread 31 读取 SMEM[124] → Bank 31

如果 Thread 0 和 Thread 32 都读取 Bank 0 → 冲突！

Swizzling 通过重新映射地址来避免冲突：
SMEM[addr] → SMEM[addr XOR (addr / stride) * offset]
```

### 12.5.9 软件流水线（Software Pipelining）

Triton 的 `num_stages` 参数控制软件流水线的深度。软件流水线将 load 和 compute 操作重叠执行：

```python
# num_stages=2 的软件流水线示意

# 时钟周期     0    1    2    3    4    5
# ─────────────────────────────────────────
# Load 迭代0  [====]
# Load 迭代1        [====]
# Compute 0         [====]
# Compute 1              [====]
# Load 迭代2              [====]
# Compute 2                    [====]

# num_stages=3 允许更多重叠：
# 时钟周期     0    1    2    3    4    5    6
# ───────────────────────────────────────────────
# Load 迭代0  [====]
# Load 迭代1        [====]
# Load 迭代2              [====]
# Compute 0              [====]
# Compute 1                    [====]
# Compute 2                          [====]
```

**num_stages 的选择策略**：

| GPU 架构 | 推荐 num_stages | 原因 |
|:---|:---|:---|
| SM80 (A100) | 3-4 | 大量 SMEM 可用于双缓冲/三缓冲 |
| SM89 (L40S) | 2-3 | SMEM 较小 |
| SM90 (H100) | 4-5 | 支持 TMA，SMEM 更大 |
| AMD MI250 | 2 | 不同的内存层次结构 |

### 12.5.8 完整 Pass Pipeline 执行流

```
输入：TritonGPU Dialect（带 layout encoding）
 │
 ├─► TritonGPUCombineOps
 │   └─ 合并可融合的 layout 转换
 │
 ├─► TritonGPUAccelerateMatmul
 │   └─ 提升 dot → Tensor Core 指令
 │
 ├─► TritonGPUOptimizeDotOperands
 │   └─ 优化 dot 操作数的布局路径
 │
 ├─► TritonGPURemoveLayoutConversions
 │   └─ 消除冗余的 convert_layout
 │
 ├─► TritonGPUPrefetch
 │   └─ 插入数据预取
 │
 ├─► TritonGPUOptimizeAccumulatorInit
 │   └─ 优化累加器初始化
 │
 ├─► TritonGPURewriteTensorPointer
 │   └─ 重写块指针为传统指针
 │
 └─► Canonicalize + DCE
     └─ 通用清理
         │
         ▼
输出：优化后的 TritonGPU Dialect
```

<div data-component="PassPipelineVisualizer"></div>

[组件：PassPipelineVisualizer - 交互式 Pass 管线可视化，展示每个 Pass 前后的 IR 变化]

---

## 12.6 阶段五：LLVM Dialect → PTX/HSACO

最后两个阶段将优化后的 TritonGPU Dialect 转换为目标硬件的可执行代码。

### 12.6.1 TritonGPU → LLVM Dialect Lowering

`TritonGPUToLLVM` 转换是整个管线中最复杂的步骤之一，它将高级的 TritonGPU 操作降低为 LLVM 方言的低级指令：

```python
# 编译管线中的调用（简化）
# lib/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.cpp

# 主要转换内容：
# 1. tt.load/tt.store → PTX/ROCM 内存访问指令
# 2. tt.dot → HMMA/MMA 指令序列
# 3. triton_gpu.convert_layout → 共享内存读写序列
# 4. 标量算术 → LLVM 指令
```

**load/store 的 lowering**：

```mlir
// 优化后的 TritonGPU IR
%val = tt.load %ptr, %mask : tensor<1024xf32, #blocked>

// Lowering 后的 LLVM Dialect（简化）
// 1. 计算每个线程的实际地址
%lane_id = ...  // 线程 ID
%base_addr = ...  // 基地址
%offset = ...  // 每个线程的偏移（由 blocked encoding 决定）
%addr = llvm.add %base_addr, %offset

// 2. 生成 PTX 级别的条件加载
// cp.async 或 ld.global（取决于架构和优化选项）
%val = llvm.inline_asm "ld.global.b32 $0, [$1];" "=r,l" %addr
```

**dot → Tensor Core 指令的 lowering**：

```mlir
// TritonGPU IR
%C = tt.dot %A, %B, %C_init : tensor<128x32xf16, #mma> * tensor<32x128xf16, #mma> -> tensor<128x128xf32, #mma>

// Lowering 后的 LLVM Dialect（简化）
// 对应 PTX 指令：mma.m16n8k16
%result = llvm.inline_asm
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
     {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};"
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    %a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1, %c2, %c3
```

### 12.6.2 共享内存操作的 Lowering

TritonGPU Dialect 中的共享内存操作被降低为显式的 SMEM 地址计算和访问指令：

```mlir
// TritonGPU 层面
%smem = triton_gpu.local_alloc %data : tensor<128x32xf32, #blocked> -> !triton_gpu.memdesc<128x32xf32, #shared>

// LLVM 层面（简化）
// 1. 计算 SMEM 地址
%smem_base = nvvm.read.ptx.sreg.smid  // 获取 SMEM 基地址
%offset = ...  // 根据 shared encoding 计算偏移
%smem_addr = llvm.add %smem_base, %offset

// 2. 写入共享内存
// st.shared 或 st.shared::cta（取决于是否使用 cluster）
llvm.inline_asm "st.shared.b32 [$0], $1;" "l,r" %smem_addr, %val

// 3. 同步
// barrier.sync 或 cp.async.commit_group
nvvm.barrier0
```

### 12.6.3 NVIDIA 目标：PTX 代码生成

对于 NVIDIA GPU，LLVM Dialect 最终通过 NVVM 后端生成 PTX 代码：

```
// vector_add_kernel 的 PTX 输出（简化，SM80）
.version 7.8
.target sm_80
.address_size 64

.visible .entry vector_add_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 n_elements
)
{
    .reg .pred  %p<2>;
    .reg .f32   %f<4>;
    .reg .b32   %r<8>;
    .reg .b64   %rd<8>;

    // pid = blockIdx.x
    mov.u32 %r1, %ctaid.x;

    // block_start = pid * 1024
    shl.b32 %r2, %r1, 10;

    // offsets = block_start + threadIdx.x (简化为单线程版本)
    mov.u32 %r3, %tid.x;
    add.s32 %r4, %r2, %r3;

    // mask = offsets < n_elements
    ld.param.u32 %r5, [n_elements];
    setp.lt.s32 %p1, %r4, %r5;

    // 条件加载 a
    @%p1 ld.param.u64 %rd1, [a_ptr];
    @%p1 cvta.to.global.u64 %rd2, %rd1;
    @%p1 mad.wide.u32 %rd3, %r4, 4, %rd2;
    @%p1 ld.global.f32 %f1, [%rd3];

    // 条件加载 b
    @%p1 ld.param.u64 %rd4, [b_ptr];
    @%p1 cvta.to.global.u64 %rd5, %rd4;
    @%p1 mad.wide.u32 %rd6, %r4, 4, %rd5;
    @%p1 ld.global.f32 %f2, [%rd6];

    // c = a + b
    add.f32 %f3, %f1, %f2;

    // 条件存储 c
    @%p1 ld.param.u64 %rd7, [c_ptr];
    @%p1 cvta.to.global.u64 %rd8, %rd7;
    @%p1 mad.wide.u32 %rd9, %r4, 4, %rd8;
    @%p1 st.global.f32 [%rd9], %f3;

    ret;
}
```

**实际的 PTX 中的向量化加载**：

真实的 Triton 编译结果会使用向量化加载指令来提高带宽利用率：

```ptx
// 使用 128-bit 向量化加载（一次加载 4 个 float32）
ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd1];

// 对应的向量化存储
st.global.v4.f32 [%rd1], {%f1, %f2, %f3, %f4};
```

### 12.6.4 PTX → CUBIN

PTX 代码通过 NVIDIA 的 `ptxas` 工具编译为 CUBIN（GPU 二进制代码）：

```python
# python/triton/backends/nvidia/driver.py（简化）
class CudaDriver:
    def compile_ptx(self, ptx, capability):
        # 调用 ptxas 编译 PTX → CUBIN
        ptxas_cmd = [
            "ptxas",
            f"-arch=sm_{capability}",
            "-v",  # verbose
            "-o", "output.cubin",
            "input.ptx"
        ]
        subprocess.run(ptxas_cmd)
```

### 12.6.5 AMD 目标：HSACO 代码生成

对于 AMD GPU，编译路径有所不同：

```
TritonGPU Dialect
    │
    ├─► LLVM Dialect
    │       │
    │       ▼
    ├─► LLVM IR（通过 LLVM 方言到 LLVM IR 的转换）
    │       │
    │       ▼
    ├─► AMDGPU 目标码（.o 文件）
    │       │
    │       ▼
    └─► HSACO（通过 hipModuleLoad 加载）
```

关键差异：

| 方面 | NVIDIA | AMD |
|:---|:---|:---|
| 中间表示 | PTX | LLVM IR |
| 目标三元组 | `nvptx64-nvidia-cuda` | `amdgcn-amd-amdhsa` |
| 汇编器 | `ptxas` | `clang` (via `lld`) |
| 二进制格式 | CUBIN | HSACO |
| Tensor Core 指令 | HMMA (mma.sync) | MFMA (v_mfma_) |
| 加载 API | `cuModuleLoad` | `hipModuleLoad` |

### 12.6.6 vector_add 的 PTX 核心代码段

以下是 vector_add_kernel 编译得到的 PTX 中最关键的计算部分（SM80，1024 元素 block）：

```ptx
// 每个线程处理 4 个元素（sizePerThread = [4]）
// 向量化加载：ld.global.v4.f32
ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd4 + 0];   // 加载 a[0:4]
ld.global.v4.f32 {%f5, %f6, %f7, %f8}, [%rd5 + 0];   // 加载 b[0:4]

// 逐元素加法
add.f32 %f9,  %f1, %f5;    // c[0] = a[0] + b[0]
add.f32 %f10, %f2, %f6;    // c[1] = a[1] + b[1]
add.f32 %f11, %f3, %f7;    // c[2] = a[2] + b[2]
add.f32 %f12, %f4, %f8;    // c[3] = a[3] + b[3]

// 向量化存储：st.global.v4.f32
st.global.v4.f32 [%rd6 + 0], {%f9, %f10, %f11, %f12};
```

<div data-component="PTXViewer"></div>

[组件：PTXViewer - 交互式 PTX 代码查看器，高亮显示向量化加载/存储和 Tensor Core 指令]

---

## 12.7 JIT 编译驱动

### 12.7.1 编译入口：`compiler.py`

Triton 的 JIT 编译由 `python/triton/compiler/compiler.py` 驱动。当 kernel 首次被调用时，触发完整的编译流程：

```python
# python/triton/compiler/compiler.py（核心流程简化）
def compile(src, target=None, options=None, key=None):
    """
    主编译函数。从 Python 源码到可执行代码的完整流程。
    """
    # 1. 检查缓存
    if key in cache:
        return cache[key]

    # 2. 阶段 1-2：Python → Triton IR
    metadata = {
        "num_warps": options.num_warps,
        "num_stages": options.num_stages,
        "debug": options.debug,
    }
    module = make_triton_ir(src, target, options)
    # module 现在是 Triton IR（MLIR ModuleOp）

    # 3. 阶段 3：Triton IR → TritonGPU Dialect
    module = ttir_to_ttgir(module, options)
    # module 现在是 TritonGPU Dialect

    # 4. 阶段 4：GPU 优化
    module = optimize_ttgir(module, options)
    # module 现在是优化后的 TritonGPU Dialect

    # 5. 阶段 5-6：LLVM → PTX/HSACO
    if target.backend == "nvidia":
        ptx, kernel_name = ttgir_to_ptx(module, options)
        cubin = ptx_to_cubin(ptx, target.capability)
    elif target.backend == "amd":
        hsaco, kernel_name = ttgir_to_hsaco(module, options)

    # 6. 缓存结果
    cache[key] = CompiledKernel(cubin, metadata)
    return cache[key]
```

### 12.7.2 JITFunction 的调用链

当用户通过 `kernel[grid](args...)` 调用 kernel 时，完整的调用链如下：

```
用户代码: vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
    │
    ├─► JITFunction.__getitem__(grid)
    │   └─ 返回 Launcher 对象（绑定 grid）
    │
    ├─► Launcher.__call__(a, b, c, N, BLOCK_SIZE)
    │   │
    │   ├─► 参数类型推断
    │   │   └─ 从 torch.Tensor → tl.pointer_type(tl.float32)
    │   │   └─ 从 int → tl.int32
    │   │   └─ BLOCK_SIZE: tl.constexpr → 1024
    │   │
    │   ├─► 生成编译缓存 key
    │   │   └─ key = hash(源码 + 参数类型 + constexpr值 + target)
    │   │
    │   ├─► 检查编译缓存
    │   │   └─ 命中 → 直接使用已编译的 cubin/hsaco
    │   │   └─ 未命中 → 调用 compile() 触发完整编译
    │   │
    │   └─► 启动 kernel
    │       └─ cuLaunchKernel(cubin, grid, block, args)
    │
    └─► kernel 在 GPU 上执行
```

### 12.7.3 参数类型推断

Triton 的类型系统是隐式的——用户不需要显式标注参数类型，编译器会从运行时参数自动推断：

```python
# python/triton/runtime/jit.py（简化）
def _type_of(obj):
    """从运行时对象推断 Triton 类型"""
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return tl.pointer_type(tl.float32)  # 或 float16, bfloat16
        else:
            return tl.pointer_type(tl.int32)
    elif isinstance(obj, int):
        if -(2**31) <= obj < 2**31:
            return tl.int32
        else:
            return tl.int64
    elif isinstance(obj, float):
        return tl.float32
    elif isinstance(obj, tl.constexpr):
        return tl.constexpr  # 编译期常量
```

**类型推断示例**：

```python
# 调用：vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)

# a: torch.Tensor (float32, cuda)  →  !tt.ptr<f32>
# b: torch.Tensor (float32, cuda)  →  !tt.ptr<f32>
# c: torch.Tensor (float32, cuda)  →  !tt.ptr<f32>
# N: int (1048576)                 →  i32
# BLOCK_SIZE: constexpr(1024)      →  编译期常量，不出现在 IR 中
```

### 12.7.4 缓存机制

Triton 的 JIT 编译结果会被缓存，避免重复编译：

```python
# 缓存 key 的组成
cache_key = {
    "hash": hashlib.sha256(source_code.encode()).hexdigest(),
    "arg_types": tuple(type_of(arg) for arg in args),
    "constants": {name: value for name, value in constexprs.items()},
    "num_warps": options.num_warps,
    "num_stages": options.num_stages,
    "target": target.fingerprint(),
}
```

缓存位置：
- 默认路径：`~/.triton/cache/`
- 环境变量：`TRITON_CACHE_DIR`

缓存目录结构：

```
~/.triton/cache/
├── <hash1>/
│   ├── __triton_launcher.py     # Python launcher 代码
│   ├── <kernel_name>.cubin      # 编译后的二进制
│   └── <kernel_name>.json       # 编译元数据
├── <hash2>/
│   ├── ...
```

### 12.7.5 Kernel Launch

编译完成后，kernel 通过 CUDA/ROCm driver API 启动：

```python
# python/triton/backends/nvidia/driver.py（简化）
class CudaDriver:
    def launch_kernel(self, kernel, grid, block, args, stream):
        """
        grid: (gridDimX, gridDimY, gridDimZ)
        block: (blockDimX, blockDimY, blockDimZ) = (num_warps * 32, 1, 1)
        """
        cu_launch_kernel(
            kernel.cu_function,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem_size,
            stream,
            args,
            None
        )
```

### 12.7.6 特殊参数处理

`tl.constexpr` 参数的特殊之处在于它们参与编译但不出现在运行时：

```python
@triton.jit
def kernel(X, BLOCK: tl.constexpr):
    ...

# BLOCK=1024 和 BLOCK=2048 会产生不同的编译结果
# 因为 constexpr 值在编译期被替换进 IR
kernel[grid](x, BLOCK=1024)   # 编译版本 1
kernel[grid](x, BLOCK=2048)   # 编译版本 2（不同的 cache key）
```

<div data-component="CompilationTimeline"></div>

[组件：CompilationTimeline - 交互式编译时间线，展示 JIT 编译各阶段的耗时和缓存命中情况]

---

## 12.8 编译选项与调试

### 12.8.1 编译选项一览

| 选项 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `num_warps` | int | 4 | 每个 CTA 的 warp 数（影响并行度和寄存器压力） |
| `num_stages` | int | None | 软件流水线阶段数（影响延迟隐藏） |
| `num_ctas` | int | 1 | 每个 CTA 的 cluster 数（H100+） |
| `maxnreg` | int | None | 每个线程的最大寄存器数 |
| `debug` | bool | False | 启用调试模式（禁用优化，保存中间 IR） |
| `sanitize_overflow` | bool | False | 启用内存访问溢出检测 |

```python
# 使用编译选项
@triton.jit
def kernel(...):
    ...

# 指定 num_warps 和 num_stages
kernel[grid](args, num_warps=8, num_stages=3)

# 通过 triton.Config 指定多个配置用于 auto-tuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def vector_add_kernel(...):
    ...
```

### 12.8.2 调试模式：`debug=True`

启用调试模式会禁用所有优化 Pass，并保存每个阶段的中间 IR：

```python
# 启用调试模式
os.environ["TRITON_DEBUG"] = "1"

# 或通过编译选项
kernel[grid](args, debug=True)
```

调试模式的行为：
1. **禁用优化**：TritonGPU 优化 Pass 被跳过
2. **保存中间 IR**：每个阶段的 IR 被写入磁盘
3. **启用断言**：在生成的代码中插入运行时断言

### 12.8.3 IR 打印：`TRITON_PRINT_IR`

环境变量 `TRITON_PRINT_IR` 控制编译过程中 IR 的打印行为：

```bash
# 打印所有阶段的 IR
export TRITON_PRINT_IR=1

# 打印特定阶段的 IR
export TRITON_PRINT_IR=ttir           # 只打印 Triton IR
export TRITON_PRINT_IR=ttgir          # 只打印 TritonGPU IR
export TRITON_PRINT_IR=llir           # 只打印 LLVM IR
export TRITON_PRINT_IR=ptx            # 只打印 PTX

# 打印多个阶段
export TRITON_PRINT_IR=ttir,ttgir,ptx
```

**实际输出示例**：

```bash
$ export TRITON_PRINT_IR=ttir
$ python vector_add.py

// -----// IR Dump: ttir //----- //
module attributes {"triton_gpu.num-warps" = 4 : i32, ...} {
  tt.func @vector_add_kernel(%arg0: !tt.ptr<f32>, ...) {
    %0 = tt.get_program_id x : i32
    ...
  }
}
```

### 12.8.4 IR Dump 目录

中间 IR 文件的保存位置：

```bash
# 默认 dump 目录
~/.triton/dump/

# 可通过环境变量指定
export TRITON_DUMP_DIR=/path/to/dump

# dump 目录结构
~/.triton/dump/
├── <kernel_name>/
│   ├── ttir.mlir              # 阶段 2 输出：Triton IR
│   ├── ttgir.mlir             # 阶段 3 输出：TritonGPU Dialect
│   ├── optimized_ttgir.mlir   # 阶段 4 输出：优化后的 TritonGPU
│   ├── llir.mlir              # 阶段 5 输出：LLVM Dialect
│   ├── llir.ll                # LLVM IR 文本格式
│   └── kernel.ptx             # 阶段 6 输出：PTX 代码
```

### 12.8.5 MLIR 优化管线的调试

要查看特定优化 Pass 的效果，可以使用 MLIR 的 pass-timing 功能：

```bash
# 打印每个 Pass 的执行时间
export MLIR_ENABLE_TIMING=1

# 打印每个 Pass 前后的 IR（需要 debug build）
export MLIR_PRINT_AFTER_PASS=1
export MLIR_PRINT_BEFORE_PASS=1
```

### 12.8.6 NVIDIA 的 IR 工具

NVIDIA 的 nsight compute 可以用于 profiling 编译后的 kernel：

```bash
# 使用 ncu 分析 kernel 性能
ncu --set full python vector_add.py

# 输出包括：
# - Kernel 执行时间
# - 内存带宽利用率
# - 计算吞吐量
# - 寄存器使用量
# - 共享内存使用量
```

### 12.8.7 常见编译错误与调试

**错误 1：不支持的 Python 特性**

```python
@triton.jit
def bad_kernel(x):
    # 错误：不支持列表操作
    items = [1, 2, 3]
    return items[0]  # Triton 不支持动态索引
```

错误信息：
```
TritonTypeError: list is not supported in Triton kernels
```

**错误 2：类型不匹配**

```python
@triton.jit
def bad_kernel(x):
    # 错误：float 和 int 不能直接比较
    val = tl.load(x)
    if val > 0:  # 可能导致类型错误
        ...
```

**错误 3：constexpr 使用不当**

```python
@triton.jit
def bad_kernel(x, N):
    # 错误：N 不是 constexpr，不能用于 shape
    a = tl.zeros([N], dtype=tl.float32)  # N 必须是 constexpr
```

修复：
```python
@triton.jit
def good_kernel(x, N: tl.constexpr):  # 标注为 constexpr
    a = tl.zeros([N], dtype=tl.float32)  # OK
```

### 12.8.8 完整调试工作流

一个典型的调试流程：

```python
import os
import triton

# 1. 启用调试模式和 IR dump
os.environ["TRITON_DEBUG"] = "1"
os.environ["TRITON_PRINT_IR"] = "ttir,ttgir,ptx"

# 2. 运行 kernel
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)

N = 1024
a = torch.randn(N, device='cuda')
b = torch.randn(N, device='cuda')
c = torch.empty_like(a)

# 3. 检查输出 IR
vector_add_kernel[(1,)](a, b, c, N, BLOCK=1024)

# 4. 验证结果
print(torch.allclose(c, a + b))  # 应该输出 True
```

### 12.8.9 IR 形式检查工具

可以使用 `triton-opt` 命令行工具直接检查和转换 IR：

```bash
# 将 Triton IR 转换为 TritonGPU Dialect
triton-opt input.ttir --tritongpu-combine --tritongpu-remove-layout-conversions

# 打印所有可用的 Pass
triton-opt --help

# 验证 IR 的合法性
triton-opt --verify-diagnostics input.ttir
```

### 12.8.10 编译性能分析

Triton 支持编译时间分析，帮助定位编译瓶颈：

```bash
# 启用 MLIR Pass 时间统计
export MLIR_ENABLE_TIMING=1

# 输出示例：
# Pass Execution Timing Report
# ─────────────────────────────────────────
#   Total Execution Time: 0.042 seconds
#
#   ---Wall Time---  ---Name---
#   0.005 ( 11.9%)  TritonGPUAccelerateMatmul
#   0.003 (  7.1%)  TritonGPUOptimizeDotOperands
#   0.008 ( 19.0%)  TritonGPURemoveLayoutConversions
#   0.002 (  4.8%)  TritonGPUCombineOps
#   0.012 ( 28.6%)  TritonGPUToLLVM
#   0.012 ( 28.6%)  Other
```

### 12.8.11 性能回归检测

在开发 Triton kernel 时，建立性能基准和回归检测非常重要：

```python
import triton
import torch

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(10, 28)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={},
    )
)
def benchmark(N, provider):
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    c = torch.empty_like(a)

    if provider == 'triton':
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vector_add_kernel[(triton.cdiv(N, 1024),)](a, b, c, N, 1024),
            quantiles=quantiles
        )
    else:
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.add(a, b, out=c),
            quantiles=quantiles
        )

    # 计算带宽：3 次内存访问（读 a, 读 b, 写 c）× 4 字节
    gbps = lambda ms: 3 * N * 4 / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True)
```

### 12.8.12 使用 `triton-opt` 手动执行 Pass

`triton-opt` 是 MLIR 生态中的命令行工具，可以手动执行单个或多个 Pass：

```bash
# 1. 将 Triton IR 转换为 TritonGPU Dialect
triton-opt input.ttir \
    --convert-triton-to-tritongpu=num-warps=4 \
    -o output.ttgir

# 2. 执行单个优化 Pass
triton-opt input.ttgir \
    --tritongpu-accelerate-matmul \
    -o accelerated.ttgir

# 3. 执行完整的优化管线
triton-opt input.ttgir \
    --tritongpu-accelerate-matmul \
    --tritongpu-optimize-dot-operands \
    --tritongpu-remove-layout-conversions \
    --tritongpu-combine \
    --tritongpu-prefetch \
    --canonicalize \
    -o optimized.ttgir

# 4. 转换为 LLVM Dialect
triton-opt optimized.ttgir \
    --convert-triton-gpu-to-llvm \
    -o output.mlir

# 5. 验证 IR 正确性（检查类型一致性、操作合法性等）
triton-opt --verify-diagnostics input.ttgir
```

**常用的 `triton-opt` 诊断选项**：

| 选项 | 作用 |
|:---|:---|
| `--verify-diagnostics` | 验证 IR 并打印诊断信息 |
| `--mlir-print-ir-after-all` | 打印每个 Pass 后的 IR |
| `--mlir-print-ir-before-all` | 打印每个 Pass 前的 IR |
| `--mlir-disable-threading` | 禁用多线程（便于调试） |
| `--allow-unregistered-dialect` | 允许未注册的方言操作 |

### 12.8.13 完整的调试工作流示例

以下是一个完整的调试流程，展示了如何定位和修复性能问题：

```python
# 步骤 1：设置调试环境
import os
os.environ["TRITON_PRINT_IR"] = "ttir,ttgir,ptx"
os.environ["TRITON_DUMP_DIR"] = "/tmp/triton_debug"
os.environ["MLIR_ENABLE_TIMING"] = "1"

# 步骤 2：运行 kernel 并收集 IR
@triton.jit
def my_kernel(...):
    ...

my_kernel[grid](args)

# 步骤 3：检查 Triton IR（阶段 2 输出）
# 检查：是否有不必要的 tt.load/tt.store？
# 检查：张量形状是否正确？
# 检查：循环是否被正确识别？

# 步骤 4：检查 TritonGPU Dialect（阶段 3 输出）
# 检查：layout encoding 是否合理？
# 检查：是否有过多的 convert_layout？
# 检查：共享内存使用量是否在预算内？

# 步骤 5：检查 PTX（阶段 6 输出）
# 检查：是否有向量化加载（ld.global.v4）？
# 检查：是否使用了 Tensor Core 指令？
# 检查：寄存器使用量是否合理？

# 步骤 6：使用 ncu 进行性能分析
# ncu --set full python my_kernel.py
# 关注指标：
# - Memory Throughput（内存吞吐量）
# - SM Throughput（SM 利用率）
# - Register Usage（寄存器使用）
# - Occupancy（占用率）
```

<div data-component="DebugConsole"></div>

[组件：DebugConsole - 交互式调试控制台，模拟 TRITON_PRINT_IR 和 TRITON_DEBUG 的输出]

---

## 12.9 阶段间 IR 对比总结

### 12.9.1 各阶段 IR 特性对比

| 特性 | Triton IR | TritonGPU Dialect | LLVM Dialect | PTX |
|:---|:---|:---|:---|:---|
| **抽象级别** | 高 | 中 | 低 | 最低 |
| **硬件感知** | 无 | 有（layout） | 有（指令） | 完全 |
| **张量表示** | `tensor<NxT>` | `tensor<NxT, #encoding>` | `!llvm.array` | 寄存器 |
| **内存模型** | `tt.load/store` | 同左 + SMEM 操作 | `llvm.load/store` | `ld/st.global` |
| **并行模型** | `tl.program_id` | 同左 | `nvvm.read.ptx.sreg` | `%ctaid.x` |
| **优化程度** | 无 | 最高 | 中 | 固定 |

### 12.9.2 vector_add 各阶段 IR 大小对比

| 阶段 | IR 行数 | 张量类型数 | 关键变化 |
|:---|:---|:---|:---|
| Python 源码 | 12 行 | N/A | 用户编写 |
| Python AST | ~60 节点 | N/A | 结构化表示 |
| Triton IR | ~25 行 MLIR | 7 种张量类型 | 标量→张量广播 |
| TritonGPU Dialect | ~30 行 MLIR | 7 种 + layout | 布局编码注入 |
| 优化后 TritonGPU | ~28 行 MLIR | 同上 | 小幅优化 |
| LLVM Dialect | ~200 行 MLIR | 无张量 | 低级表示 |
| PTX | ~100 行 | N/A | 寄存器级别 |

### 12.9.3 IR 到硬件的映射

```
Triton IR                    GPU 硬件
─────────                    ────────
tl.program_id(0)        ──►  blockIdx.x
tl.arange(0, N)         ──►  threadIdx.x (在每个线程中计算实际值)
tt.load(ptr, mask)      ──►  ld.global (有条件的)
tt.store(ptr, val, mask) ──►  st.global (有条件的)
arith.addf(a, b)        ──►  add.f32 (逐元素，SIMT)
tt.dot(A, B, C)         ──►  mma.sync / HMMA (Tensor Core)
```

---

## 12.10 端到端编译实例

### 12.10.1 带矩阵乘法的 kernel

让我们追踪一个更复杂的 kernel——矩阵乘法——的完整编译过程：

```python
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    tl.store(C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn), accumulator)
```

**阶段 2：Triton IR（部分）**

```mlir
// matmul_kernel 的 Triton IR（循环部分简化）
scf.for %k = %c0_i32 to %K step %c32_i32 iter_args(%acc = %zero) {
    // 加载 A 块
    %a_mask = arith.cmpi slt, %offs_k_broadcast_plus_k, %K_splat
    %a = tt.load %a_ptrs, %a_mask, %cst_zero : tensor<128x32xf32>

    // 加载 B 块
    %b_mask = arith.cmpi slt, %offs_k_broadcast_plus_k, %K_splat
    %b = tt.load %b_ptrs, %b_mask, %cst_zero : tensor<32x128xf32>

    // 矩阵乘法累加
    %acc_next = tt.dot %a, %b, %acc : tensor<128x32xf32> * tensor<32x128xf32> -> tensor<128x128xf32>

    // 更新指针
    %a_ptrs_next = tt.addptr %a_ptrs, %a_offset_step
    %b_ptrs_next = tt.addptr %b_ptrs, %b_offset_step

    scf.yield %acc_next : tensor<128x128xf32>
}
```

**阶段 3：TritonGPU Dialect（关键变化）**

```mlir
// 注入 MMA layout 用于 Tensor Core
#blocked_a = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8],
                                   warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_b = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4],
                                   warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0,
                                warpsPerCTA = [2, 2], instrShape = [16, 8]}>

// 循环中的操作现在带有 layout encoding
scf.for %k = ... iter_args(%acc = %zero) {
    %a_blocked = tt.load ... : tensor<128x32xf32, #blocked_a>
    %b_blocked = tt.load ... : tensor<32x128xf32, #blocked_b>

    // 布局转换：blocked → MMA
    %a_mma = triton_gpu.convert_layout %a_blocked
        : tensor<128x32xf32, #blocked_a> -> tensor<128x32xf32, #mma>
    %b_mma = triton_gpu.convert_layout %b_blocked
        : tensor<32x128xf32, #blocked_b> -> tensor<32x128xf32, #mma>

    // Tensor Core 矩阵乘法
    %acc_next = tt.dot %a_mma, %b_mma, %acc
        : tensor<128x32xf32, #mma> * tensor<32x128xf32, #mma> -> tensor<128x128xf32, #mma>

    scf.yield %acc_next
}
```

**阶段 4 优化后**：

```mlir
// 优化后的关键变化：
// 1. convert_layout 被优化为通过 shared memory 的间接转换
// 2. load 预取被插入
// 3. shared memory 的 swizzling 被优化以避免 bank conflict

// 使用 shared memory 作为中介
scf.for %k = ... iter_args(%acc = %acc_init) {
    // 1. 加载到 blocked layout
    %a_blocked = tt.load ... : tensor<128x32xf32, #blocked>
    %b_blocked = tt.load ... : tensor<32x128xf32, #blocked>

    // 2. 写入 shared memory
    %a_smem = triton_gpu.local_alloc %a_blocked
        : tensor<128x32xf32, #blocked> -> !triton_gpu.memdesc<128x32xf32, #shared>
    %b_smem = triton_gpu.local_alloc %b_blocked
        : tensor<32x128xf32, #blocked> -> !triton_gpu.memdesc<32x128xf32, #shared>

    // 3. 从 shared memory 加载到 MMA layout
    %a_mma = triton_gpu.local_load %a_smem
        : !triton_gpu.memdesc<128x32xf32, #shared> -> tensor<128x32xf32, #mma>
    %b_mma = triton_gpu.local_load %b_smem
        : !triton_gpu.memdesc<32x128xf32, #shared> -> tensor<32x128xf32, #mma>

    // 4. Tensor Core 计算
    %acc_next = tt.dot %a_mma, %b_mma, %acc
        : tensor<128x32xf32, #mma> * tensor<32x128xf32, #mma> -> tensor<128x128xf32, #mma>

    scf.yield %acc_next
}
```

### 12.10.2 matmul kernel 的 PTX 关键片段

```ptx
// Tensor Core 矩阵乘法指令（HMMA m16n8k16）
// 每条指令计算 16×8 的输出块

// 循环展开后的计算序列
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%f1, %f2, %f3, %f4},     // 输出 C（累加器）
    {%r1, %r2, %r3, %r4},     // 输入 A
    {%r5, %r6},                // 输入 B
    {%f1, %f2, %f3, %f4};     // 输入 C（累加）

mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%f5, %f6, %f7, %f8},
    {%r1, %r2, %r3, %r4},
    {%r7, %r8},
    {%f5, %f6, %f7, %f8};

// ... 更多 HMMA 指令以覆盖完整的 128×128 输出块
```

### 12.10.3 编译过程中的关键决策点

在整个编译管线中，有几个关键的决策点会显著影响最终代码质量：

**决策 1：Tile Size 选择**

```python
# BLOCK_M, BLOCK_N, BLOCK_K 的选择影响：
# 1. 每个线程的计算量 → 影响 ILP（指令级并行）
# 2. 共享内存使用量 → 影响 occupancy
# 3. 寄存器压力 → 影响 register spilling

# 典型的 tile size 配置（A100，f16 矩阵乘法）：
# BLOCK_M=128, BLOCK_N=256, BLOCK_K=32
# → 每个 CTA 使用 128×32×2 + 32×256×2 = 32KB SMEM
# → A100 有 164KB SMEM，可以放 5 个 CTA
```

**决策 2：Warp 数量**

```python
# num_warps 影响：
# 1. 每个 CTA 的线程数 = num_warps × 32
# 2. 每个 warp 处理的 tile 部分
# 3. warp 间的同步需求

# 典型配置：
# 小矩阵：num_warps = 4  (128 threads)
# 中矩阵：num_warps = 8  (256 threads)
# 大矩阵：num_warps = 16 (512 threads)
```

**决策 3：MMA 版本选择**

```python
# AccelerateMatmul Pass 的选择逻辑：
# 1. 检查 GPU compute capability
# 2. 检查数据类型（f16/bf16/fp8/tf32）
# 3. 选择最优的 MMA 指令版本

# SM80 (A100) 的 MMA 版本矩阵：
# ┌──────────┬─────────┬──────────────┐
# │ 数据类型  │ MMA 版本 │ instrShape   │
# ├──────────┼─────────┼──────────────┤
# │ f16      │ v2      │ [16, 8, 16]  │
# │ bf16     │ v2      │ [16, 8, 16]  │
# │ tf32     │ v2      │ [16, 8, 8]   │
# │ int8     │ v2      │ [16, 8, 32]  │
# └──────────┴─────────┴──────────────┘

# SM90 (H100) 额外支持：
# ┌──────────┬─────────┬──────────────┐
# │ fp8      │ v3      │ [16, 8, 32]  │
# │ (e4m3)   │ (WGMMA) │              │
# └──────────┴─────────┴──────────────┘
```

### 12.10.4 Block Pointer API 的编译

Triton 2.1+ 引入了 Block Pointer API（`tl.make_block_pointer`），它提供了更高级的内存访问抽象：

```python
# 使用 Block Pointer API
@triton.jit
def matmul_block_ptr(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # 创建块指针（比手动计算偏移更清晰）
    A_block_ptr = tl.make_block_pointer(
        base=A,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        offsets=[pid_m * BLOCK_M, 0],
        block_shape=[BLOCK_M, BLOCK_K],
        order=[1, 0],
    )
    B_block_ptr = tl.make_block_pointer(
        base=B,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        offsets=[0, pid_n * BLOCK_N],
        block_shape=[BLOCK_K, BLOCK_N],
        order=[0, 1],
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_block_ptr)
        b = tl.load(B_block_ptr)
        acc = tl.dot(a, b, acc)
        A_block_ptr = tl.advance(A_block_ptr, [0, BLOCK_K])
        B_block_ptr = tl.advance(B_block_ptr, [BLOCK_K, 0])

    C_block_ptr = tl.make_block_pointer(
        base=C,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        offsets=[pid_m * BLOCK_M, pid_n * BLOCK_N],
        block_shape=[BLOCK_M, BLOCK_N],
        order=[1, 0],
    )
    tl.store(C_block_ptr, acc)
```

**Block Pointer 的编译优势**：

| 方面 | 传统偏移计算 | Block Pointer |
|:---|:---|:---|
| **边界检查** | 需要显式 mask | 自动处理边界 |
| **指针更新** | 手动 addptr | `tl.advance`（更清晰） |
| **编译优化** | 编译器需推断访问模式 | 编译器直接获取访问模式 |
| **代码可读性** | 低（大量偏移计算） | 高（声明式） |
| **支持硬件 TMA** | 不支持 | 支持（SM90+） |

Block Pointer 在编译阶段 4（`RewriteTensorPointer` Pass）中被展开为传统的指针计算，但在 SM90+ 上，如果使用了 TMA（Tensor Memory Accelerator），则保留为 TMA 指令。

---

## 本章小结

本章深入剖析了 Triton 从 Python DSL 到 GPU 机器码的完整编译管线。以下是关键要点：

**六阶段管线架构**：

| 阶段 | 变换 | 关键技术 |
|:---|:---|:---|
| 1 | Python → AST | `inspect.getsource` + `ast.parse` |
| 2 | AST → Triton IR | 代码生成器的 visitor 模式 |
| 3 | Triton IR → TritonGPU | Layout encoding 注入 |
| 4 | TritonGPU 优化 | 8+ 个 MLIR Pass 的管线 |
| 5 | LLVM Dialect lowering | TritonGPU → LLVM → PTX/HSACO |
| 6 | 机器码生成 | ptxas/clang 编译 |

**核心设计思想**：

1. **渐进式降低**：从高级 Python 语法逐步降低到硬件指令，每一阶段只负责一个抽象层次的转换
2. **MLIR 方言复用**：利用 MLIR 的方言系统实现类型安全的 IR 变换
3. **硬件特化延迟**：直到阶段 3 才引入硬件特定信息，保持前端的可移植性
4. **Pass 组合**：优化 Pass 可以自由组合和排序，适应不同的硬件和算法特性

**调试工具链**：

- `TRITON_PRINT_IR`：查看各阶段 IR
- `TRITON_DEBUG`：禁用优化，保存中间结果
- `TRITON_DUMP_DIR`：指定 IR dump 目录
- `triton-opt`：命令行 IR 检查工具

---

## 思考题

1. **为什么 Triton 选择在 Python 层面使用 `inspect.getsource` 而不是直接执行 Python 函数并追踪？这种设计有什么优缺点？**

   提示：考虑以下方面：
   - 与 PyTorch eager mode 的对比（torch.jit.script vs torch.jit.trace）
   - 对 Python 动态特性的支持程度
   - 编译时的类型推断能力
   - 调试体验（错误信息指向源码行号）

2. **在阶段 3 中，layout encoding 的选择如何影响最终代码性能？如果为 vector_add 使用 `#mma` encoding 而不是 `#blocked`，会发生什么？**

   提示：
   - `#blocked` 和 `#mma` 的线程-数据映射方式有何不同？
   - 对于逐元素运算（如 `arith.addf`），哪种 encoding 更高效？
   - `#mma` encoding 是否会引入不必要的数据重排？

3. **TritonGPU 的优化 Pass 中，`RemoveLayoutConversions` 和 `CombineOps` 都处理 layout 转换。它们的区别是什么？能否交换执行顺序？**

   提示：
   - `CombineOps` 关注的是"合并"——将多个相同的转换合并为一个
   - `RemoveLayoutConversions` 关注的是"消除"——移除不需要的转换
   - 考虑：如果先执行 Remove，再执行 Combine，会发生什么？

4. **对比 NVIDIA 和 AMD 的编译路径，为什么 AMD 选择直接使用 LLVM IR 而不是引入 PTX 这样的中间汇编层？这对可维护性有什么影响？**

   提示：
   - PTX 是 NVIDIA 的虚拟 ISA，而 AMD 使用 LLVM IR 作为中间表示
   - 考虑：PTX 的版本兼容性保证 vs LLVM IR 的不稳定性
   - 考虑：调试 PTX vs 调试 LLVM IR 的难度
   - 考虑：AMD 的 GCN/RDNA 架构与 NVIDIA 的 CUDA 核心架构差异

5. **`tl.constexpr` 参数会导致同一个 kernel 产生多个编译版本。如何在 kernel 设计中平衡灵活性和编译时间？请以 BLOCK_SIZE 为例讨论。**

   提示：
   - 分析：如果支持 BLOCK_SIZE = 64, 128, 256, 512, 1024，会产生几个编译版本？
   - 编译缓存如何缓解这个问题？
   - `@triton.autotuner` 的作用是什么？
   - 考虑：constexpr 的性能收益 vs 编译时间成本

6. **在矩阵乘法 kernel 的编译过程中，`AccelerateMatmul` Pass 将 `tt.dot` 提升为 Tensor Core 操作。如果用户的矩阵维度不能整除 Tensor Core 的 `instrShape`（如 m16n8k16），编译器如何处理？**

   提示：
   - 考虑 M=100, N=50, K=30 的情况
   - 编译器是否会插入 padding 操作？
   - 这种情况下的性能损失有多大？
   - 用户如何通过调整 tile size 来避免这个问题？

7. **JIT 缓存使用源码 hash 作为 key。如果用户修改了 kernel 中的注释但不修改逻辑，会导致缓存失效吗？Triton 是否应该做更智能的缓存策略？**

   提示：
   - 考虑：Python AST 中是否包含注释？
   - `inspect.getsource` 返回的文本是否包含注释？
   - 如果做"语义级缓存"（基于 AST 而非源码文本），会有什么挑战？

8. **在调试 Triton kernel 时，你发现某个 Pass 导致了错误的计算结果。描述你会如何使用 IR dump 和 `triton-opt` 工具来定位问题。**

   提示：
   - 使用 `TRITON_PRINT_IR` 查看各阶段 IR
   - 使用 `triton-opt --mlir-print-ir-after-all` 逐 Pass 检查
   - 使用 `git bisect` 思路：二分法定位是哪个 Pass 引入的 bug
   - 对比 debug 模式（无优化）和正常模式的输出

9. **（扩展思考）Triton 的编译管线如何支持多种硬件后端（NVIDIA、AMD、CPU）？如果要添加一个新的硬件后端（如 Intel GPU），需要修改哪些组件？**

   提示：
   - 考虑 `lib/Backend/` 目录结构
   - 每个后端需要实现哪些接口？
   - layout encoding 如何扩展到新硬件？
   - LLVM Dialect → 目标代码的转换如何实现？

10. **（扩展思考）在 SM90 (H100) 上，Triton 支持 TMA (Tensor Memory Accelerator) 和 WGMMA (Warp Group MMA)。这些新硬件特性如何融入现有的编译管线？哪些 Pass 需要被修改或新增？**

    提示：
    - TMA 允许从全局内存直接加载到共享内存，无需经过寄存器
    - WGMMA 是 warp group 级别的矩阵乘法，需要跨 warp 协调
    - 考虑：Prefetch Pass 如何利用 TMA？
    - 考虑：AccelerateMatmul Pass 如何选择 MMA v2 vs WGMMA？
