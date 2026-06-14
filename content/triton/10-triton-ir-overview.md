---
title: "Chapter 10: Triton IR 概述与核心指令"
description: "深入理解 Triton IR 的设计目标与 SSA 形式、语法结构与类型系统、核心指令分类（arith/math/memory/control/tensor）、tt.dot/tt.load/tt.store/tt.reduce 指令详解、Python 到 Triton IR 的转换过程，以及 IR 验证与调试工具"
date: "2026-06-11"
---

# Chapter 10: Triton IR 概述与核心指令

> **学习目标**：
> - 理解 Triton IR 的设计目标——为什么需要自定义 IR？Triton IR 与 LLVM IR 的定位差异
> - 掌握 Triton IR 的语法结构：模块（module）、函数（func）、基本块（basic block）、操作（operation）
> - 理解 Triton IR 的类型系统：tensor type、pointer type、scalar type 的定义与约束
> - 掌握核心指令分类：arith（算术）、math（数学函数）、memory（load/store）、control（cf.br/cf.cond_br）、tensor（tt.expand_dims/tt.reshape）
> - 深入理解 tt.dot 指令的 IR 表示、类型约束与 Tensor Core 映射
> - 深入理解 tt.load/tt.store 指令的指针类型、掩码、边界检查与缓存修饰符
> - 深入理解 tt.reduce 指令的归约区域与支持的归约类型
> - 理解从 Python 到 Triton IR 的 AST 转换过程，能够阅读 IR dump 输出
> - 掌握 mlir-opt 等 IR 验证与调试工具的使用

---

## 10.1 Triton IR 的设计目标

### 10.1.1 为什么需要自定义 IR？

在理解 Triton IR 之前，我们需要回答一个根本问题：**为什么 Triton 不直接生成 LLVM IR 或 PTX？**

传统的深度学习编译器（如 TVM、XLA）通常采用以下流程：

```
Python 前端 → 高层 IR → 优化 Pass → LLVM IR → PTX/机器码
```

而 Triton 选择了一条不同的路线——引入一个**自定义的、Tile 级别的 IR**：

```
Python DSL → Triton IR (tt dialect) → TritonGPU IR → LLVM Dialect → PTX/机器码
```

Triton IR 存在的核心理由有三点：

**理由一：Tile 级抽象的语义表达**

Triton 的核心编程模型是 **Tile 级操作**——程序员操作的是整块张量（tile），而不是标量或单个线程。这种抽象无法直接用 LLVM IR 表达。LLVM IR 是**标量级**的 SSA IR，每个值都是标量，不理解"张量"或"Tile"的概念。

```
Triton Python:
    a = tl.load(ptr_a + offsets)   # 一次加载一整行 Tile
    b = tl.load(ptr_b + offsets)
    c = a + b                       # Tile 级加法

LLVM IR (无法直接表达):
    ; 需要展开为数百条标量 load + 标量 add
    ; 失去了 Tile 的语义信息
    %val0 = load float, ptr %p0
    %val1 = load float, ptr %p1
    %sum0 = fadd float %val0, %val1
    ; ... 重复 BLOCK_SIZE 次
```

**理由二：高层优化的空间**

Triton IR 保留了足够的语义信息，使得编译器可以在**降低到标量之前**做高层优化：

| 优化类型 | 在 Triton IR 中 | 在 LLVM IR 中 |
|:---|:---|:---|
| Tile 分块策略 | 直接操作 Tile，优化空间大 | 已展开为标量，信息丢失 |
| 内存访问模式 | 可以分析整块的访问模式 | 只能看到单个 load/store |
| Tensor Core 映射 | tt.dot 直接映射到 mma.sync | 需要复杂的模式匹配 |
| 循环优化 | Tile 级循环，优化直观 | 标量循环，优化复杂 |

**理由三：硬件无关性**

Triton IR 是**硬件无关**的。同一份 Triton IR 可以被不同的后端（NVIDIA、AMD、Intel、Ascend）编译为目标平台的代码。硬件特定的优化在 TritonGPU Dialect 中处理。

### 10.1.2 Triton IR 与 LLVM IR 的定位差异

```
┌─────────────────────────────────────────────────────────────┐
│                    Triton 编译管线                            │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐ │
│  │ Python   │    │ Triton   │    │ TritonGPU│    │ LLVM  │ │
│  │ DSL      │───→│ IR       │───→│ IR       │───→│ IR    │ │
│  │          │    │ (tt)     │    │ (triton. │    │       │ │
│  │ Tile 级  │    │ Tile 级  │    │  gpu)    │    │ 标量级│ │
│  └──────────┘    └──────────┘    └──────────┘    └───────┘ │
│       │              │               │               │      │
│    AST 解析      类型检查         硬件映射         指令选择  │
│                  语义验证         内存分配          寄存器分配│
│                  高层优化         Layout 优化       指令调度  │
└─────────────────────────────────────────────────────────────┘
```

| 特性 | Triton IR | LLVM IR |
|:---|:---|:---|
| **抽象级别** | Tile 级（一整个张量块） | 标量级（单个值） |
| **SSA 形式** | 是 | 是 |
| **类型系统** | tensor、pointer、scalar | 标量、向量、指针、结构体 |
| **内存模型** | 隐式（编译器管理共享内存） | 显式（程序员管理一切） |
| **并行模型** | 自动映射到 warp/thread | 需要手动展开 |
| **主要方言** | tt、ttir | LLVM Dialect |
| **目标受众** | AI 编译器开发者 | 系统级开发者 |

### 10.1.3 SSA 形式

Triton IR 采用 **SSA（Static Single Assignment）** 形式。SSA 的核心规则是：**每个变量只被定义一次，使用前必须先定义。**

```
// 非 SSA 形式（伪代码）：
%x = load ptr_a
%x = add %x, 1      // x 被重新赋值，违反 SSA
%x = mul %x, 2

// SSA 形式：
%x0 = load ptr_a
%x1 = add %x0, 1     // 新的名字 %x1
%x2 = mul %x1, 2     // 新的名字 %x2
```

SSA 的优势：
1. **简化数据流分析**：每个定义唯一，可以直接构建 def-use 链
2. **简化优化**：死代码消除、常量传播等优化更易实现
3. **自然支持并行**：没有同一变量的多次赋值，天然适合并行执行

当控制流导致一个变量有多个可能的定义时，SSA 使用 **phi 节点**合并：

```
^bb0:
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  %val1 = arith.constant 1.0 : f32
  cf.br ^bb3(%val1 : f32)

^bb2:
  %val2 = arith.constant 2.0 : f32
  cf.br ^bb3(%val2 : f32)

^bb3(%result : f32):    // phi 节点：根据控制流选择 %val1 或 %val2
  // %result 在 bb1 来时是 1.0，在 bb2 来时是 2.0
```

<div data-component="SSAFormDiagram"></div>

[组件：SSAFormDiagram - 可视化 SSA 形式的数据流图，展示 def-use 链和 phi 节点]

---

## 10.2 Triton IR 的语法结构

### 10.2.1 整体层次结构

Triton IR 的语法基于 **MLIR（Multi-Level Intermediate Representation）** 框架。一个完整的 Triton IR 程序由以下层次组成：

```
module {                                    // 模块：最外层容器
  tt.func @kernel_name(                     // 函数：对应一个 kernel
    %arg0: !tt.ptr<f32>,                    // 参数：指针类型
    %arg1: !tt.ptr<f32>,
    %arg2: i32                              // 参数：标量类型
  ) {
    %c0 = arith.constant 0 : i32            // 操作：算术常量
    %c1 = arith.constant 1 : i32
    %block_id = tt.program_id x : i32       // 操作：获取 block ID

    ^bb0:                                   // 基本块入口
      // ... 更多操作 ...
      tt.return                             // 函数返回
  }
}
```

让我们逐层分析：

**Module（模块）**

```mlir
module {
  // 模块是 IR 的最外层容器
  // 一个模块可以包含多个函数、全局常量等
  // 在 Triton 中，一次编译通常只有一个模块
}
```

**Function（函数）**

```mlir
tt.func @vector_add(
    %arg0: !tt.ptr<f32>,       // 第一个参数：float32 指针（输入 A）
    %arg1: !tt.ptr<f32>,       // 第二个参数：float32 指针（输入 B）
    %arg2: !tt.ptr<f32>,       // 第三个参数：float32 指针（输出 C）
    %arg3: i32                 // 第四个参数：标量（元素数量 N）
) {
  // 函数体由基本块组成
  tt.return
}
```

- `tt.func`：Triton 的函数定义指令
- `@vector_add`：函数名（符号引用）
- `%arg0`、`%arg1` 等：函数参数，使用 `%` 前缀标识 SSA 值
- 类型注解：`!tt.ptr<f32>` 表示指向 f32 的指针，`i32` 表示 32 位整数

**Basic Block（基本块）**

基本块是一段**顺序执行**的指令序列，只有一个入口和一个出口：

```mlir
^bb0:                              // 基本块标签，以 ^ 开头
  %offsets = tt.make_range {end = 256} : tensor<256xi32>
  %ptr = tt.addptr %base_ptr, %offsets
  %val = tt.load %ptr
  // ... 更多操作 ...
  tt.return                        // 块的终止操作
```

基本块的规则：
1. 入口处没有分支跳入（除了函数的第一个基本块）
2. 出口处必须有一个**终止操作**（terminator），如 `tt.return`、`cf.br`、`cf.cond_br`
3. 中间不能有跳转指令

**Operation（操作）**

操作是 IR 的基本执行单元，格式为：

```
%result = dialect.operation_name %operand1, %operand2 {attributes} : type
```

示例分解：

```mlir
%result = arith.addf %a, %b {fastmath = #arith.fastmath<none>} : f32
│         │      │      │   │                              │
│         │      │      │   │                              └─ 返回类型
│         │      │      │   └─ 属性（可选的元数据）
│         │      │      └─ 操作数（SSA 值）
│         │      └─ 操作名
│         └─ 方言名（arith = 算术方言）
└─ 结果（SSA 值名）
```

### 10.2.2 类型系统

Triton IR 的类型系统分为三大类：**标量类型**、**张量类型**和**指针类型**。

**标量类型（Scalar Type）**

标量类型表示单个值，与 LLVM IR 的标量类型一致：

| 类型 | 说明 | 示例值 |
|:---|:---|:---|
| `i1` | 1 位布尔 | true / false |
| `i8` | 8 位整数 | 42 |
| `i16` | 16 位整数 | 1000 |
| `i32` | 32 位整数 | 65536 |
| `i64` | 64 位整数 | 2^40 |
| `f16` | 16 位浮点（IEEE 754） | 3.14 |
| `bf16` | Brain Float 16 | 3.14 |
| `f32` | 32 位浮点 | 3.14159 |
| `f64` | 64 位浮点 | 3.14159265 |
| `index` | 平台相关的索引类型 | 1024 |

```mlir
// 标量类型示例
%c = arith.constant 42 : i32            // 32 位整数常量
%f = arith.constant 3.14 : f32          // 32 位浮点常量
%b = arith.constant true : i1           // 布尔常量
```

**张量类型（Tensor Type）**

张量类型是 Triton IR 的核心——表示一个 **Tile**（张量块）：

```
tensor<SHAPE x ELEMENT_TYPE>
```

```mlir
// 一维张量
tensor<128xf32>           // 128 个 float32 元素的 1D 张量
tensor<256xi32>           // 256 个 int32 元素的 1D 张量

// 二维张量
tensor<128x64xf16>        // 128×64 的 float16 矩阵
tensor<64x64xf32>         // 64×64 的 float32 矩阵

// 标量张量（0 维）
tensor<f32>               // 单个 float32 值（作为张量）
```

张量的关键特性：
1. **形状在编译期固定**：维度大小必须是常量，不能是运行时变量
2. **元素类型必须一致**：所有元素共享同一个标量类型
3. **映射到寄存器**：编译后，张量的元素存储在 GPU 寄存器中

```mlir
// 张量类型在 IR 中的使用
%zeros = tt.splat %c0.0 : (f32) -> tensor<128xf32>
//                           │                    │
//                           │                    └─ 返回类型：128 元素的 f32 张量
//                           └─ 输入类型：标量 f32
// %zeros 是一个 128 元素的张量，所有元素都是 0.0
```

**指针类型（Pointer Type）**

指针类型表示一个指向内存的指针：

```
!tt.ptr<POINTED_TYPE>
```

```mlir
// 指针类型示例
!tt.ptr<f32>              // 指向 float32 的指针
!tt.ptr<i32>              // 指向 int32 的指针
!tt.ptr<f16>              // 指向 float16 的指针
!tt.ptr<tensor<128xf32>>  // 指向张量的指针（Block Pointer 模式）
!tt.ptr<i8>               // 指向 int8 的指针（用于字节级访问）
```

在函数签名中，指针参数对应 Python kernel 的指针参数：

```mlir
// Python:
// @triton.jit
// def kernel(A_ptr, B_ptr, C_ptr, N):
//     ...

// 对应的 Triton IR:
tt.func @kernel(
    %A_ptr: !tt.ptr<f32>,      // Python 的 A_ptr
    %B_ptr: !tt.ptr<f32>,      // Python 的 B_ptr
    %C_ptr: !tt.ptr<f32>,      // Python 的 C_ptr
    %N: i32                     // Python 的 N
) {
  // ...
}
```

<div data-component="TypeSystemDiagram"></div>

[组件：TypeSystemDiagram - 可视化 Triton IR 类型系统的层次结构]

### 10.2.3 完整的 IR 示例

让我们看一个完整的 vector_add kernel 的 Triton IR：

```mlir
// 对应 Python kernel:
// @triton.jit
// def vector_add(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
//     pid = tl.program_id(0)
//     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
//     mask = offsets < N
//     a = tl.load(A_ptr + offsets, mask=mask)
//     b = tl.load(B_ptr + offsets, mask=mask)
//     tl.store(C_ptr + offsets, a + b, mask=mask)

module {
  tt.func @vector_add(
      %arg0: !tt.ptr<f32>,                  // A_ptr
      %arg1: !tt.ptr<f32>,                  // B_ptr
      %arg2: !tt.ptr<f32>,                  // C_ptr
      %arg3: i32                             // N
  ) attributes {noinline = false} {
    // ===== 计算偏移量 =====
    // pid = tl.program_id(0)
    %0 = tt.program_id x : i32

    // offsets = pid * BLOCK_SIZE
    %c256_i32 = arith.constant 256 : i32     // BLOCK_SIZE = 256
    %1 = arith.muli %0, %c256_i32 : i32

    // arange(0, 256) → [0, 1, 2, ..., 255]
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>

    // splat(pid * BLOCK_SIZE) → [base, base, ..., base]
    %3 = tt.splat %1 : (i32) -> tensor<256xi32>

    // offsets = base + arange → [base, base+1, ..., base+255]
    %4 = arith.addi %3, %2 : tensor<256xi32>

    // ===== 构造掩码 =====
    // splat(N) → [N, N, ..., N]
    %5 = tt.splat %arg3 : (i32) -> tensor<256xi32>

    // mask = offsets < N → [True, True, ..., False, ...]
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>

    // ===== 加载数据 =====
    // A_ptr + offsets
    %7 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>

    // a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    %cst = arith.constant 0.000000e+00 : f32
    %9 = tt.load %8, %6, %cst : tensor<256xf32>
    //     │    │   │   │
    //     │    │   │   └─ other 值（mask 为 False 时的填充值）
    //     │    │   └─ 掩码张量
    //     │    └─ 指针张量
    //     └─ 加载指令

    // B_ptr + offsets
    %10 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>

    // b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    %12 = tt.load %11, %6, %cst : tensor<256xf32>

    // ===== 计算 =====
    // a + b
    %13 = arith.addf %9, %12 : tensor<256xf32>

    // ===== 存储结果 =====
    // C_ptr + offsets
    %14 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>

    // tl.store(C_ptr + offsets, a + b, mask=mask)
    tt.store %15, %13, %6 : tensor<256xf32>
    //    │    │   │   │
    //    │    │   │   └─ 掩码张量
    //    │    │   └─ 要存储的值张量
    //    │    └─ 指针张量
    //    └─ 存储指令

    tt.return
  }
}
```

---

## 10.3 核心指令分类

Triton IR 的指令按方言（dialect）分类。以下是与 Triton 编程最相关的指令分类：

```
Triton IR 指令分类
├── arith dialect（算术指令）
│   ├── arith.addf / arith.subf / arith.mulf / arith.divf   // 浮点算术
│   ├── arith.addi / arith.subi / arith.muli / arith.divsi  // 整数算术
│   ├── arith.cmpf / arith.cmpi                              // 比较
│   ├── arith.select                                         // 条件选择
│   └── arith.constant                                       // 常量
├── math dialect（数学函数）
│   ├── math.exp / math.log / math.sqrt                      // 基本数学
│   ├── math.sin / math.cos / math.tan                       // 三角函数
│   └── math.absf / math.ceil / math.floor                   // 其他
├── tt dialect（Triton 核心）
│   ├── tt.load / tt.store                                   // 内存访问
│   ├── tt.dot                                               // 矩阵乘法
│   ├── tt.reduce                                            // 归约
│   ├── tt.make_range / tt.splat / tt.expand_dims            // 张量构造
│   ├── tt.addptr                                            // 指针算术
│   ├── tt.program_id                                        // 程序 ID
│   └── tt.func / tt.return                                  // 函数控制
├── cf dialect（控制流）
│   ├── cf.br                                                // 无条件分支
│   ├── cf.cond_br                                           // 条件分支
│   └── cf.switch                                            // 多路分支
└── scf dialect（结构化控制流）
    ├── scf.for                                              // 循环
    ├── scf.if                                               // 条件
    └── scf.while                                            // while 循环
```

### 10.3.1 arith 方言——算术指令

arith 方言提供基础的算术和逻辑运算。它操作标量和张量两种类型。

**标量算术运算**

```mlir
// 浮点加法
%a = arith.constant 1.0 : f32
%b = arith.constant 2.0 : f32
%c = arith.addf %a, %b : f32          // %c = 3.0

// 整数乘法
%x = arith.constant 10 : i32
%y = arith.constant 20 : i32
%z = arith.muli %x, %y : i32          // %z = 200

// 浮点乘法
%p = arith.mulf %a, %b : f32          // %p = 2.0

// 整数除法（有符号）
%d = arith.divsi %z, %x : i32         // %d = 20

// 浮点除法
%q = arith.divf %b, %a : f32          // %q = 2.0
```

**张量算术运算（逐元素）**

当操作数是张量类型时，arith 指令自动变为**逐元素**运算：

```mlir
// 张量逐元素加法
%ta = tt.splat %a : (f32) -> tensor<4xf32>    // [1.0, 1.0, 1.0, 1.0]
%tb = tt.splat %b : (f32) -> tensor<4xf32>    // [2.0, 2.0, 2.0, 2.0]
%tc = arith.addf %ta, %tb : tensor<4xf32>     // [3.0, 3.0, 3.0, 3.0]

// 张量逐元素乘法
%tp = arith.mulf %ta, %tb : tensor<4xf32>     // [2.0, 2.0, 2.0, 2.0]
```

**比较指令**

```mlir
// 整数比较
%x = arith.constant 10 : i32
%y = arith.constant 20 : i32
%eq = arith.cmpi eq, %x, %y : i32     // false (相等比较)
%lt = arith.cmpi slt, %x, %y : i32    // true  (有符号小于)

// 浮点比较
%a = arith.constant 1.0 : f32
%b = arith.constant 2.0 : f32
%lt_f = arith.cmpf olt, %a, %b : f32  // true (有序小于)
```

比较谓词（predicate）：
- 整数：`eq`(相等)、`ne`(不等)、`slt`(有符号<)、`sle`(有符号≤)、`sgt`(有符号>)、`sge`(有符号≥)、`ult`(无符号<) 等
- 浮点：`oeq`(有序=)、`one`(不等)、`olt`(有序<)、`ole`(有序≤)、`ogt`(有序>)、`uge`(无序≥) 等

**条件选择指令**

```mlir
// arith.select：根据条件选择值
%cond = arith.constant true : i1
%a = arith.constant 1.0 : f32
%b = arith.constant 2.0 : f32
%result = arith.select %cond, %a, %b : f32   // %result = 1.0

// 在张量上使用（逐元素选择）
%mask = arith.cmpi slt, %offsets, %N : tensor<256xi32>
%values = arith.select %mask, %data, %zeros : tensor<256xf32>
// mask 为 True 的位置选择 data，否则选择 zeros
```

### 10.3.2 math 方言——数学函数

math 方言提供超越函数（transcendental functions），这些函数通常映射到 GPU 的特殊指令或软件实现。

```mlir
// 指数函数（常用于 softmax）
%x = arith.constant 1.0 : f32
%exp_x = math.exp %x : f32             // e^1.0 ≈ 2.718

// 对数函数
%log_x = math.exp2 %x : f32            // 2^1.0 = 2.0

// 平方根
%sqrt_x = math.sqrt %x : f32           // √1.0 = 1.0

// 绝对值
%neg = arith.constant -3.0 : f32
%abs = math.absf %neg : f32            // |-3.0| = 3.0

// 三角函数
%sin_x = math.sin %x : f32             // sin(1.0) ≈ 0.841
%cos_x = math.cos %x : f32             // cos(1.0) ≈ 0.540
```

math 方言同样支持张量类型（逐元素）：

```mlir
// 对整个 Tile 做 exp（softmax 的核心操作）
%logits = ... : tensor<128xf32>
%exp_logits = math.exp %logits : tensor<128xf32>
// 每个元素独立计算 exp，编译器自动并行化
```

**常见数学函数汇总**

| 指令 | 功能 | Python 等价 |
|:---|:---|:---|
| `math.exp` | e^x | `torch.exp()` |
| `math.exp2` | 2^x | `2 ** x` |
| `math.log` | ln(x) | `torch.log()` |
| `math.log2` | log₂(x) | `math.log2()` |
| `math.sqrt` | √x | `torch.sqrt()` |
| `math.absf` | \|x\| | `torch.abs()` |
| `math.sin` | sin(x) | `torch.sin()` |
| `math.cos` | cos(x) | `torch.cos()` |
| `math.ceil` | 向上取整 | `math.ceil()` |
| `math.floor` | 向下取整 | `math.floor()` |

### 10.3.3 memory 方言与 tt 方言——内存指令

Triton 的内存操作主要通过 `tt.load` 和 `tt.store` 实现（将在 10.5 节详细讨论）。这里先看基础的指针操作：

**tt.addptr — 指针算术**

```mlir
// 基础指针 + 偏移量
%base_ptr = ... : !tt.ptr<f32>
%offset = arith.constant 100 : i32
%new_ptr = tt.addptr %base_ptr, %offset : !tt.ptr<f32>, i32
// %new_ptr 指向 base_ptr + 100 * sizeof(f32) 的位置
```

在张量上使用时，生成**指针张量**：

```mlir
// 指针张量 + 偏移张量
%base_ptrs = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
%offsets = tt.make_range {end = 256} : tensor<256xi32>
%ptrs = tt.addptr %base_ptrs, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
// %ptrs[i] = %arg0 + i * sizeof(f32)，每个元素指向不同的内存位置
```

### 10.3.4 control flow 方言——控制流指令

Triton 的控制流在 IR 层面分为两套：**结构化控制流**（scf 方言）和**非结构化控制流**（cf 方言）。

**cf.br — 无条件分支**

```mlir
^bb0:
  %val = arith.constant 42 : i32
  cf.br ^bb1    // 无条件跳转到 ^bb1

^bb1:
  // 使用 %val ...
  tt.return
```

**cf.cond_br — 条件分支**

```mlir
^bb0:
  %cond = arith.cmpi slt, %x, %y : i32
  cf.cond_br %cond, ^bb1, ^bb2    // if %cond then goto bb1 else goto bb2

^bb1:
  // x < y 的情况
  cf.br ^bb3

^bb2:
  // x >= y 的情况
  cf.br ^bb3

^bb3:
  // 汇合点
  tt.return
```

**scf.for — 结构化循环**

```mlir
// 对应 Python: for i in range(0, N, step):
%c0 = arith.constant 0 : index
%cN = arith.constant 1024 : index
%c1 = arith.constant 1 : index

%init = arith.constant 0.0 : f32

%result = scf.for %iv = %c0 to %cN step %c1 iter_args(%acc = %init) -> f32 {
    // %iv: 循环变量（当前迭代的值）
    // %acc: 累加器（通过 iter_args 传递）
    %val = ... : f32
    %new_acc = arith.addf %acc, %val : f32
    scf.yield %new_acc : f32    // 将 %new_acc 传给下一次迭代的 %acc
}
```

scf.for 的关键概念：
- `%iv`：归纳变量（loop induction variable）
- `iter_args`：循环携带的值（loop-carried values），类似于 SSA 的 phi 节点
- `scf.yield`：每次迭代结束时 yield 的值，传递给下一次迭代

**scf.if — 条件执行**

```mlir
%cond = arith.constant true : i1

%result = scf.if %cond -> f32 {
    // then 分支
    %val = arith.constant 1.0 : f32
    scf.yield %val : f32
} else {
    // else 分支
    %val = arith.constant 2.0 : f32
    scf.yield %val : f32
}
```

### 10.3.5 tensor 方言与 tt 方言——张量构造指令

**tt.make_range — 创建索引范围**

```mlir
// 对应 Python: tl.arange(0, BLOCK_SIZE)
%range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
// %range = [0, 1, 2, 3, ..., 255]
```

**tt.splat — 标量广播为张量**

```mlir
// 对应 Python: 标量被广播到整个 Tile
%scalar = arith.constant 5.0 : f32
%splatted = tt.splat %scalar : (f32) -> tensor<128xf32>
// %splatted = [5.0, 5.0, 5.0, ..., 5.0]  （128 个 5.0）
```

**tt.expand_dims — 扩展维度**

```mlir
// 对应 Python: tl.expand_dims(x, axis)
%row = tt.make_range {end = 128} : tensor<128xi32>
%row_2d = tt.expand_dims %row {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// [0, 1, ..., 127] → [[0], [1], ..., [127]]
// 从 1D 变成 2D，axis=1 表示在第 1 维扩展

// 常用于构造二维索引
%col = tt.make_range {end = 64} : tensor<64xi32>
%col_2d = tt.expand_dims %col {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// [0, 1, ..., 63] → [[0, 1, ..., 63]]

// 广播加法：行索引 + 列索引 → 二维索引矩阵
%indices = arith.addi %row_2d, %col_2d : tensor<128x64xi32>
// indices[i][j] = i + j
```

**tt.reshape — 重塑张量**

```mlir
// 对应 Python: tl.reshape(x, shape)
%flat = ... : tensor<256xf32>
%matrix = tt.reshape %flat : tensor<256xf32> -> tensor<16x16xf32>
// 将 1D 的 256 元素重塑为 16×16 的 2D 矩阵
// 注意：总元素数必须相同
```

**tt.trans — 转置张量**

```mlir
// 对应 Python: tl.trans(x) 或 x.T
%matrix = ... : tensor<128x64xf32>
%transposed = tt.trans %matrix : tensor<128x64xf32> -> tensor<64x128xf32>
// 行列互换
```

<div data-component="TensorOpsVisualizer"></div>

[组件：TensorOpsVisualizer - 可视化 tt.make_range、tt.expand_dims、tt.broadcast 的索引张量构建过程]

---

## 10.4 tt.dot 指令详解

### 10.4.1 矩阵乘法的 IR 表示

`tt.dot` 是 Triton 中最重要的指令之一，它执行**分块矩阵乘法**，并直接映射到 GPU 的 Tensor Core 硬件。

```mlir
// Python: c = tl.dot(a, b)
%c = tt.dot %a, %b, %c_init {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} :
    tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
//     │           │              │           │              │
//     │           │              │           │              └─ 输出类型
//     │           │              │           └─ 输入 B 类型
//     │           │              └─ 输入 A 类型
//     │           └─ 第二个操作数（矩阵 B）
//     └─ 第一个操作数（矩阵 A）
```

更详细的 IR 示例：

```mlir
// 假设 A: 128×64, B: 64×128, C_init: 128×128 (累加器)
%a = ... : tensor<128x64xf16>
%b = ... : tensor<64x128xf16>
%c_init = arith.constant dense<0.0> : tensor<128x128xf32>

// 矩阵乘法: C = A @ B + C_init
%c = tt.dot %a, %b, %c_init {allowTF32 = true} :
    tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
```

### 10.4.2 输入输出类型约束

`tt.dot` 对输入输出类型有严格约束：

| 参数 | 类型要求 | 说明 |
|:---|:---|:---|
| A (左矩阵) | `tensor<M×K×dtype_a>` | M 行 K 列 |
| B (右矩阵) | `tensor<K×N×dtype_b>` | K 行 N 列 |
| C_init (累加器) | `tensor<M×N×dtype_c>` | M 行 N 列，初始值 |
| 输出 | `tensor<M×N×dtype_c>` | 与 C_init 类型相同 |

**K 维度必须匹配**：A 的列数必须等于 B 的行数。

**支持的 dtype 组合**：

| dtype_a | dtype_b | dtype_c | Tensor Core 指令 |
|:---|:---|:---|:---|
| f16 | f16 | f16 | `mma.sync.aligned.m16n8k16.row.col.f16` |
| f16 | f16 | f32 | `mma.sync.aligned.m16n8k16.row.col.f32.f16` |
| bf16 | bf16 | f32 | `mma.sync.aligned.m16n8k16.row.col.f32.bf16` |
| tf32 | tf32 | f32 | `mma.sync.aligned.m16n8k8.row.col.f32.tf32` |
| int8 | int8 | int32 | `mma.sync.aligned.m16n8k32.row.col.s32.s8` |

**精度提升模式**：输入是 f16，输出是 f32，意味着累加在 f32 精度下进行，减少累积误差。

### 10.4.3 与 Tensor Core 的映射

NVIDIA GPU 的 Tensor Core 执行**固定大小的矩阵乘法**（如 16×8×16）。Triton 的 `tt.dot` 会被编译器自动分解为多个 Tensor Core 指令：

```
Triton IR:
    tt.dot: tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>

编译后（概念性）:
    for m in range(128/16):      // 8 次
      for n in range(128/8):     // 16 次
        for k in range(64/16):   // 4 次
          mma.sync m16n8k16     // 一条 Tensor Core 指令

总共: 8 × 16 × 4 = 512 条 mma 指令
```

```
┌─────────────────────────────────────────────┐
│     tt.dot: 128×64 × 64×128 → 128×128      │
│                                             │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┐                │
│  │  │  │  │  │  │  │  │  │ ← 每个小块     │
│  ├──┼──┼──┼──┼──┼──┼──┼──┤    是一条       │
│  │  │  │  │  │  │  │  │  │    mma.sync     │
│  ├──┼──┼──┼──┼──┼──┼──┼──┤    指令         │
│  │  │  │  │  │  │  │  │  │    (16×8)       │
│  ├──┼──┼──┼──┼──┼──┼──┼──┤                │
│  │  │  │  │  │  │  │  │  │                │
│  └──┴──┴──┴──┴──┴──┴──┴──┘                │
│  8行 × 16列 = 128 个 m16n8 子块             │
│  每个子块沿 K 维度迭代 4 次（64/16）          │
└─────────────────────────────────────────────┘
```

### 10.4.4 tt.dot 的完整 IR 示例

以下是一个 matmul kernel 中 tt.dot 的完整上下文：

```mlir
// 矩阵乘法 kernel 的核心循环体
// C[M, N] = A[M, K] @ B[K, N]

// 初始化累加器
%c_init = arith.constant dense<0.0> : tensor<128x128xf32>

// K 维度循环
%result = scf.for %k = %c0 to %cK step %c64 iter_args(%acc = %c_init) -> tensor<128x128xf32> {
    // 加载 A 的一个块: 128×64
    %a_ptrs = ... : tensor<128x64x!tt.ptr<f16>>
    %a_block = tt.load %a_ptrs : tensor<128x64xf16>

    // 加载 B 的一个块: 64×128
    %b_ptrs = ... : tensor<64x128x!tt.ptr<f16>>
    %b_block = tt.load %b_ptrs : tensor<64x128xf16>

    // 矩阵乘法累加: acc += a_block @ b_block
    %new_acc = tt.dot %a_block, %b_block, %acc {allowTF32 = true} :
        tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>

    scf.yield %new_acc : tensor<128x128xf32>
}

// result 包含最终的 C 矩阵块
```

---

<div data-component="DotOperationVisualizer"></div>

[组件：DotOperationVisualizer - 可视化 tt.dot 指令的 Tensor Core 分块与数据流]

---

## 10.5 tt.load / tt.store 指令详解

### 10.5.1 tt.load 指令的 IR 表示

`tt.load` 是 Triton 中最核心的内存指令，它从内存加载一个 Tile 到寄存器。

**基本语法**：

```mlir
%result = tt.load %ptr, %mask, %other {cache_modifier, eviction_policy, ...} : type
```

**完整参数说明**：

| 参数 | 类型 | 说明 |
|:---|:---|:---|
| `%ptr` | `tensor<Nx!tt.ptr<T>>` 或 `!tt.ptr<T>` | 指针（张量或标量） |
| `%mask` | `tensor<Nxi1>` (可选) | 掩码，True 表示加载，False 表示跳过 |
| `%other` | `tensor<NxT>` 或标量 T (可选) | 掩码为 False 时的填充值 |
| 输出 | `tensor<NxT>` | 加载结果 |

**不带掩码的加载**：

```mlir
// Python: a = tl.load(ptr + offsets)
%ptrs = ... : tensor<256x!tt.ptr<f32>>
%a = tt.load %ptrs : tensor<256xf32>
// 从 256 个地址加载 256 个 f32 值
// 所有地址都必须合法（无越界保护）
```

**带掩码的加载**：

```mlir
// Python: a = tl.load(ptr + offsets, mask=mask, other=0.0)
%ptrs = ... : tensor<256x!tt.ptr<f32>>
%mask = ... : tensor<256xi1>
%cst0 = arith.constant 0.0 : f32

%a = tt.load %ptrs, %mask, %cst0 : tensor<256xf32>
//    │     │      │      │
//    │     │      │      └─ mask=False 时填充值为 0.0
//    │     │      └─ 掩码张量
//    │     └─ 指针张量
//    └─ 加载指令
```

掩码加载的工作原理：

```
指针:  [p0, p1, p2, p3, p4, p5, p6, p7]
掩码:  [T,  T,  T,  T,  T,  T,  F,  F ]
                                ↓   ↓
结果:  [v0, v1, v2, v3, v4, v5, 0.0, 0.0]
                               ↑   ↑
                        other 填充值
```

### 10.5.2 边界检查的 IR 表示

在实际使用中，加载操作需要处理**数组越界**问题。Triton 通过掩码实现：

```mlir
// Python:
// offsets = pid * BLOCK + tl.arange(0, BLOCK)
// mask = offsets < N
// a = tl.load(A_ptr + offsets, mask=mask, other=0.0)

// 对应的 Triton IR:
%pid = tt.program_id x : i32
%c256 = arith.constant 256 : i32
%base = arith.muli %pid, %c256 : i32
%range = tt.make_range {end = 256} : tensor<256xi32>
%base_vec = tt.splat %base : (i32) -> tensor<256xi32>
%offsets = arith.addi %base_vec, %range : tensor<256xi32>

// 构造掩码
%N_vec = tt.splat %N : (i32) -> tensor<256xi32>
%mask = arith.cmpi slt, %offsets, %N_vec : tensor<256xi32>
// mask[i] = (offsets[i] < N)

// 带掩码加载
%ptrs = tt.splat %A_ptr : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
%ptrs_with_offset = tt.addptr %ptrs, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
%cst0 = arith.constant 0.0 : f32
%a = tt.load %ptrs_with_offset, %mask, %cst0 : tensor<256xf32>
```

### 10.5.3 cache_modifier 和 eviction_policy

`tt.load` 支持缓存修饰符和驱逐策略，用于优化内存层次的行为：

**cache_modifier（缓存修饰符）**

```mlir
// 使用 .ca（cache at all levels）—— 默认行为
%a = tt.load %ptrs {cacheModifier = 1 : i32} : tensor<256xf32>
// 数据缓存在 L1 和 L2 中

// 使用 .cg（cache at global level）—— 仅缓存在 L2
%b = tt.load %ptrs {cacheModifier = 2 : i32} : tensor<256xf32>
// 跳过 L1，适合一次性访问的数据

// 使用 .cs（streaming）—— 低优先级缓存
%c = tt.load %ptrs {cacheModifier = 3 : i32} : tensor<256xf32>
// 适合流式访问模式，不污染缓存
```

| cache_modifier | 值 | 含义 | 使用场景 |
|:---|:---|:---|:---|
| (无) | 0 | 默认 | 一般情况 |
| `.ca` | 1 | Cache at all levels | 数据会被复用 |
| `.cg` | 2 | Cache at global level | 一次性访问 |
| `.cs` | 3 | Streaming | 流式处理 |
| `.cv` | 4 | Don't cache | 强制从内存读取 |

**eviction_policy（驱逐策略）**

```mlir
// evict_first — 优先驱逐（低优先级数据）
%a = tt.load %ptrs {evictionPolicy = 1 : i32} : tensor<256xf32>

// evict_last — 最后驱逐（高优先级数据）
%b = tt.load %ptrs {evictionPolicy = 2 : i32} : tensor<256xf32>
```

| eviction_policy | 值 | 含义 | 使用场景 |
|:---|:---|:---|:---|
| (无) | 0 | 默认 | 一般情况 |
| `evict_first` | 1 | 优先被驱逐 | 低复用率数据 |
| `evict_normal` | 2 | 正常驱逐 | 默认行为 |
| `evict_last` | 3 | 最后被驱逐 | 高复用率数据 |

### 10.5.4 tt.store 指令

`tt.store` 的语法与 `tt.load` 类似，但方向相反——将寄存器中的数据写入内存：

```mlir
// 基本存储
tt.store %ptrs, %values : tensor<256xf32>
// 将 %values 写入 %ptrs 指向的地址

// 带掩码存储
tt.store %ptrs, %values, %mask : tensor<256xf32>
// mask=False 的位置不写入

// 带缓存修饰符
tt.store %ptrs, %values {cacheModifier = ...} : tensor<256xf32>
```

**完整示例**：

```mlir
// Python:
// tl.store(C_ptr + offsets, result, mask=mask)

// Triton IR:
%C_ptrs = tt.splat %C_ptr : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
%C_ptrs_off = tt.addptr %C_ptrs, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
%result = arith.addf %a, %b : tensor<256xf32>
tt.store %C_ptrs_off, %result, %mask : tensor<256xf32>
//     │          │         │      │
//     │          │         │      └─ 掩码（越界位置不写入）
//     │          │         └─ 要写入的数据
//     │          └─ 目标地址（指针张量）
//     └─ 存储指令
```

### 10.5.5 Block Pointer 模式

Triton 还支持 **Block Pointer** 模式，用于更高效的块级内存访问：

```mlir
// Python:
// block_ptr = tl.make_block_ptr(base, shape, strides, offsets, block_shape)
// data = tl.load(block_ptr)

// Triton IR (Block Pointer 模式):
%base = ... : !tt.ptr<f32>
%shape = ... : tensor<2xi32>       // [M, N]
%strides = ... : tensor<2xi32>     // [N, 1]
%offsets = ... : tensor<2xi32>     // [block_m, block_n]
%block_shape = ... : tensor<2xi32> // [BLOCK_M, BLOCK_N]

// 创建 block pointer
%block_ptr = tt.make_block_ptr %base, %shape, %strides, %offsets, %block_shape :
    !tt.ptr<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>

// 块级加载（自动生成边界检查）
%data = tt.load %block_ptr : tensor<128x64xf32>
```

---

<div data-component="LoadStoreVisualizer"></div>

[组件：LoadStoreVisualizer - 可视化 tt.load/tt.store 的内存访问模式、缓存策略与边界检查行为]

---

## 10.6 tt.reduce 指令详解

### 10.6.1 归约操作的 IR 表示

`tt.reduce` 执行沿指定维度的归约操作（如求和、求最大值）。它是 `tl.sum`、`tl.max`、`tl.min` 等高级 API 的底层实现。

**基本语法**：

```mlir
%result = tt.reduce (%input) [%init] {
    // 归约区域（reduction region）
    ^bb0(%a: TYPE, %b: TYPE):
        %combined = arith.addf %a, %b : TYPE
        scf.yield %combined : TYPE
} : tensor<MxNxf32> -> tensor<Mxf32>
//    │                        │
//    │                        └─ 归约后的类型（维度 N 被归约掉）
//    └─ 输入张量类型
```

### 10.6.2 归约区域详解

归约区域是一个**匿名函数**（lambda），定义了如何将两个值合并为一个值。

**沿 axis=1 归约（归约列维度）**：

```mlir
// Python: row_sums = tl.sum(input, axis=1)
%input = ... : tensor<128x64xf32>

%sums = tt.reduce (%input) [%cst0] {
    ^bb0(%a: f32, %b: f32):
        %sum = arith.addf %a, %b : f32
        scf.yield %sum : f32
} : tensor<128x64xf32> -> tensor<128xf32>
// 输入: 128×64 矩阵
// 输出: 128 元素向量（每行的和）
```

归约过程的可视化：

```
输入矩阵 (128×64):
┌──────────────────────────────┐
│ 1.0  2.0  3.0  ...  4.0     │  → 归约 → sum = 130.5
│ 5.0  6.0  7.0  ...  8.0     │  → 归约 → sum = 256.3
│ ...                          │
│ 9.0  1.0  2.0  ...  3.0     │  → 归约 → sum = 198.7
└──────────────────────────────┘
        ↓
输出向量 (128):
[130.5, 256.3, ..., 198.7]
```

**沿 axis=0 归约（归约行维度）**：

```mlir
// Python: col_sums = tl.sum(input, axis=0)
%input = ... : tensor<128x64xf32>

%sums = tt.reduce (%input) [%cst0] {
    ^bb0(%a: f32, %b: f32):
        %sum = arith.addf %a, %b : f32
        scf.yield %sum : f32
} : tensor<128x64xf32> -> tensor<64xf32>
// 输入: 128×64 矩阵
// 输出: 64 元素向量（每列的和）
```

### 10.6.3 支持的归约类型

通过修改归约区域的操作，可以实现不同的归约语义：

**求和（Sum）**

```mlir
%sum = tt.reduce (%input) [%cst0] {
    ^bb0(%a: f32, %b: f32):
        %r = arith.addf %a, %b : f32
        scf.yield %r : f32
} : tensor<256xf32> -> tensor<f32>
```

**求最大值（Max）**

```mlir
%max = tt.reduce (%input) [%neg_inf] {
    ^bb0(%a: f32, %b: f32):
        %r = arith.maximumf %a, %b : f32
        scf.yield %r : f32
} : tensor<256xf32> -> tensor<f32>
```

**求最小值（Min）**

```mlir
%min = tt.reduce (%input) [%pos_inf] {
    ^bb0(%a: f32, %b: f32):
        %r = arith.minimumf %a, %b : f32
        scf.yield %r : f32
} : tensor<256xf32> -> tensor<f32>
```

**逻辑或（Any）**

```mlir
%any = tt.reduce (%input) [%false] {
    ^bb0(%a: i1, %b: i1):
        %r = arith.ori %a, %b : i1
        scf.yield %r : i1
} : tensor<256xi1> -> tensor<i1>
```

**逻辑与（All）**

```mlir
%all = tt.reduce (%input) [%true] {
    ^bb0(%a: i1, %b: i1):
        %r = arith.andi %a, %b : i1
        scf.yield %r : i1
} : tensor<256xi1> -> tensor<i1>
```

**Argmax（带索引的最大值）**

归约区域可以 yield 复合值，实现 argmax：

```mlir
// Python 伪代码: max_val, max_idx = argmax(input)

// 构造 (值, 索引) 对
%values = ... : tensor<256xf32>
%indices = tt.make_range {end = 256} : tensor<256xi32>

// 将值和索引打包为张量对（在 Triton IR 中通常用 tuple 或分别处理）
// 这里展示概念性的实现：
%result = tt.reduce (%values, %indices) [%neg_inf, %c0_i32] {
    ^bb0(%val_a: f32, %idx_a: i32, %val_b: f32, %idx_b: i32):
        %is_greater = arith.cmpf ogt, %val_a, %val_b : f32
        %max_val = arith.select %is_greater, %val_a, %val_b : f32
        %max_idx = arith.select %is_greater, %idx_a, %idx_b : i32
        scf.yield %max_val, %max_idx : f32, i32
} : tensor<256xf32>, tensor<256xi32> -> f32, i32
```

<div data-component="ReduceOperationDiagram"></div>

[组件：ReduceOperationDiagram - 可视化 tt.reduce 的树形归约过程]

### 10.6.4 tt.reduce 的底层实现

`tt.reduce` 在编译时会被展开为**树形归约**（tree reduction）或**warp shuffle 归约**：

```
树形归约示例（8 个元素，2 个 warp，每个 warp 4 个线程）：

Step 1: 线程内归约（寄存器级）
Thread 0: [a0, a1, a2, a3] → a0+a1, a2+a3
Thread 1: [b0, b1, b2, b3] → b0+b1, b2+b3

Step 2: Warp 内归约（shuffle）
Thread 0: (a0+a1)+(a2+a3) → partial_sum_0
Thread 1: (b0+b1)+(b2+b3) → partial_sum_1

Step 3: Warp 间归约（shared memory 或 shuffle）
partial_sum_0 + partial_sum_1 → final_sum
```

---

## 10.7 从 Python 到 Triton IR 的转换

### 10.7.1 AST Visitor 机制

当用户调用 `@triton.jit` 装饰的函数时，Triton 的 JIT 编译器会执行以下步骤：

```
1. 捕获 Python 函数的源代码
2. 解析为 Python AST（抽象语法树）
3. 通过 TritonIRVisitor 遍历 AST
4. 将每个 Python 语句转换为对应的 Triton IR 操作
5. 输出 Triton IR（MLIR 格式）
```

```python
# 用户代码
@triton.jit
def add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    tl.store(Z + offsets, x + y, mask=mask)
```

Python AST 到 Triton IR 的映射：

| Python AST 节点 | Triton IR 操作 | 说明 |
|:---|:---|:---|
| `tl.program_id(0)` | `tt.program_id x : i32` | 获取 block ID |
| `tl.arange(0, N)` | `tt.make_range {start=0, end=N}` | 创建索引范围 |
| `a * b` (int) | `arith.muli %a, %b : i32` | 整数乘法 |
| `a + b` (int) | `arith.addi %a, %b : i32` | 整数加法 |
| `a + b` (float) | `arith.addf %a, %b : f32` | 浮点加法 |
| `a < b` | `arith.cmpi slt, %a, %b` | 小于比较 |
| `tl.load(ptr, mask=...)` | `tt.load %ptr, %mask, %other` | 带掩码加载 |
| `tl.store(ptr, val, mask=...)` | `tt.store %ptr, %val, %mask` | 带掩码存储 |
| `tl.dot(a, b)` | `tt.dot %a, %b, %acc` | 矩阵乘法 |
| `tl.sum(x, axis=0)` | `tt.reduce (%x) [...] {...}` | 归约求和 |
| `tl.exp(x)` | `math.exp %x` | 指数函数 |
| `for i in range(N):` | `scf.for %i = ...` | 循环 |
| `if cond:` | `scf.if %cond` 或 `arith.select` | 条件 |

### 10.7.2 使用 TRITON_PRINT_IR=1 查看 IR

Triton 提供了环境变量来打印编译过程中的 IR：

```bash
# 打印 Triton IR（tt dialect）
TRITON_PRINT_IR=1 python your_kernel.py

# 打印所有阶段的 IR（包括 TritonGPU、LLVM 等）
TRITON_PRINT_AUTOTUNING=1 python your_kernel.py

# 只打印特定 pass 的 IR
TRITON_PRINT_IR=1 python your_kernel.py 2>&1 | head -100
```

### 10.7.3 实际 IR Dump 示例

让我们通过一个具体的例子，观察 Python 代码如何被转换为 Triton IR。

**Python 源代码**：

```python
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)
```

**TRITON_PRINT_IR=1 输出的 Triton IR**（带详细中文注释）：

```mlir
// =====================================================
// 阶段 1: 初始 Triton IR (tt dialect)
// 这是 AST-to-IR 转换的直接输出，尚未优化
// =====================================================

#loc = loc("/tmp/softmax.py":0:0)           // 源文件位置信息
module {
  tt.func @softmax_kernel(
      %arg0: !tt.ptr<f32>,                  // output_ptr
      %arg1: !tt.ptr<f32>,                  // input_ptr
      %arg2: i32,                           // input_row_stride
      %arg3: i32,                           // output_row_stride
      %arg4: i32                            // n_cols
  ) attributes {noinline = false} {
    // ---- row_idx = tl.program_id(0) ----
    %0 = tt.program_id x : i32              // 获取当前 block 的 x 维度 ID

    // ---- row_start_ptr = input_ptr + row_idx * input_row_stride ----
    %1 = arith.muli %0, %arg2 : i32         // row_idx * input_row_stride
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32  // input_ptr + offset

    // ---- col_offsets = tl.arange(0, BLOCK_SIZE) ----
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // col_offsets = [0, 1, 2, ..., 1023]

    // ---- input_ptrs = row_start_ptr + col_offsets ----
    %4 = tt.splat %2 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    // 将标量指针广播为指针张量
    %5 = tt.addptr %4, %3 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // input_ptrs[i] = row_start_ptr + i

    // ---- mask = col_offsets < n_cols ----
    %6 = tt.splat %arg4 : (i32) -> tensor<1024xi32>
    %7 = arith.cmpi slt, %3, %6 : tensor<1024xi32>
    // mask[i] = (i < n_cols)

    // ---- row = tl.load(input_ptrs, mask=mask, other=-inf) ----
    %cst = arith.constant 0xFF800000 : f32  // -inf 的 IEEE 754 表示
    %8 = tt.load %5, %7, %cst : tensor<1024xf32>
    // 加载一行数据，越界位置填充 -inf

    // ---- max_val = tl.max(row, axis=0) ----
    %cst_0 = arith.constant 0xFF800000 : f32  // 初始值 -inf
    %9 = tt.reduce (%8) [%cst_0] {
    ^bb0(%arg5: f32, %arg6: f32):
      %20 = arith.maximumf %arg5, %arg6 : f32
      scf.yield %20 : f32
    } : tensor<1024xf32> -> f32
    // 对整行求最大值，结果是标量

    // ---- row_minus_max = row - max_val ----
    %10 = tt.splat %9 : (f32) -> tensor<1024xf32>
    %11 = arith.subf %8, %10 : tensor<1024xf32>
    // 每个元素减去最大值（数值稳定性优化）

    // ---- numerator = tl.exp(row_minus_max) ----
    %12 = math.exp %11 : tensor<1024xf32>
    // 计算 exp(x - max)

    // ---- denominator = tl.sum(numerator, axis=0) ----
    %cst_1 = arith.constant 0.000000e+00 : f32  // 初始值 0.0
    %13 = tt.reduce (%12) [%cst_1] {
    ^bb0(%arg5: f32, %arg6: f32):
      %20 = arith.addf %arg5, %arg6 : f32
      scf.yield %20 : f32
    } : tensor<1024xf32> -> f32
    // 对 exp 结果求和，得到分母

    // ---- softmax_output = numerator / denominator ----
    %14 = tt.splat %13 : (f32) -> tensor<1024xf32>
    %15 = arith.divf %12, %14 : tensor<1024xf32>
    // 逐元素除法：exp(x-max) / sum(exp(x-max))

    // ---- tl.store(output_ptrs, softmax_output, mask=mask) ----
    %16 = arith.muli %0, %arg3 : i32         // row_idx * output_row_stride
    %17 = tt.addptr %arg0, %16 : !tt.ptr<f32>, i32  // output_row_ptr
    %18 = tt.splat %17 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    %19 = tt.addptr %18, %3 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %19, %15, %7 : tensor<1024xf32>
    // 存储 softmax 结果，越界位置不写入

    tt.return
  }
}
```

### 10.7.4 IR 中的关键模式识别

通过阅读大量 Triton IR dump，我们可以总结出常见的 IR 模式：

**模式 1：指针计算模式**

```mlir
// Python: ptr + offsets
// IR 模式：
%splat_ptr = tt.splat %scalar_ptr : (!tt.ptr<T>) -> tensor<Nx!tt.ptr<T>>
%ptrs = tt.addptr %splat_ptr, %offsets : tensor<Nx!tt.ptr<T>>, tensor<Nxi32>
```

**模式 2：掩码构造模式**

```mlir
// Python: mask = offsets < N
// IR 模式：
%splat_N = tt.splat %N : (i32) -> tensor<Nxi32>
%mask = arith.cmpi slt, %offsets, %splat_N : tensor<Nxi32>
```

**模式 3：归约模式**

```mlir
// Python: result = tl.sum/ tl.max(x, axis=0)
// IR 模式：
%result = tt.reduce (%input) [%init_value] {
^bb0(%a: T, %b: T):
    %r = arith.addf/arith.maximumf %a, %b : T
    scf.yield %r : T
} : tensor<NxT> -> T
```

**模式 4：数值稳定 softmax 模式**

```mlir
// Python: row - tl.max(row) → exp → tl.sum → divide
// IR 模式：
%max = tt.reduce (%row) [%neg_inf] { ... maximumf ... } : tensor -> f32
%shifted = arith.subf %row, (tt.splat %max) : tensor
%exp_val = math.exp %shifted : tensor
%sum = tt.reduce (%exp_val) [%zero] { ... addf ... } : tensor -> f32
%result = arith.divf %exp_val, (tt.splat %sum) : tensor
```

<div data-component="IRConversionStep"></div>

[组件：IRConversionStep - 交互式展示 Python 代码到 Triton IR 的逐步转换过程，高亮对应关系]

---

## 10.8 Triton IR 的验证与打印

### 10.8.1 mlir-opt 工具使用

`mlir-opt` 是 MLIR 框架提供的通用 IR 分析和转换工具。Triton 扩展了 mlir-opt，增加了 Triton 特有的 pass。

**基本用法**：

```bash
# 验证 IR（检查语法和类型正确性）
mlir-opt input.mlir --mlir-print-op-generic -o /dev/null

# 打印 IR（美化格式）
mlir-opt input.mlir --mlir-print-op-generic

# 运行特定 pass
mlir-opt input.mlir --triton-canonicalize

# 组合多个 pass
mlir-opt input.mlir \
    --triton-canonicalize \
    --triton-combine \
    --print-ir-after-all
```

**从 Python 程序导出 IR**：

```python
import triton
import triton.language as tl
import os

@triton.jit
def my_kernel(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    tl.store(Y + offs, x * 2)

# 设置环境变量导出 IR
os.environ["TRITON_PRINT_IR"] = "1"

# 触发编译
x = torch.randn(256, device='cuda')
y = torch.empty_like(x)
my_kernel[(1,)](x, y, BLOCK=256)
```

### 10.8.2 IR 验证规则

Triton IR 有严格的验证规则，违反时会报错。主要验证规则包括：

**规则 1：SSA 一致性**

```mlir
// 错误：使用未定义的值
tt.func @bad() {
    %result = arith.addf %undefined, %also_undefined : f32
    //         ^ 错误：%undefined 未定义
    tt.return
}

// 正确：先定义后使用
tt.func @good() {
    %a = arith.constant 1.0 : f32
    %b = arith.constant 2.0 : f32
    %c = arith.addf %a, %b : f32    // %a 和 %b 已定义
    tt.return
}
```

**规则 2：类型匹配**

```mlir
// 错误：类型不匹配
%a = arith.constant 1.0 : f32
%b = arith.constant 2 : i32
%c = arith.addf %a, %b : f32
//         ^ 错误：%b 是 i32，不能与 f32 做 addf

// 正确：类型一致
%a = arith.constant 1.0 : f32
%b = arith.constant 2.0 : f32
%c = arith.addf %a, %b : f32    // 两个 f32
```

**规则 3：张量形状匹配**

```mlir
// 错误：张量形状不匹配
%a = ... : tensor<128xf32>
%b = ... : tensor<256xf32>
%c = arith.addf %a, %b : tensor<128xf32>
//         ^ 错误：128 ≠ 256

// 正确：形状一致
%a = ... : tensor<128xf32>
%b = ... : tensor<128xf32>
%c = arith.addf %a, %b : tensor<128xf32>
```

**规则 4：tt.dot 的维度约束**

```mlir
// 错误：K 维度不匹配
%a = ... : tensor<128x64xf16>
%b = ... : tensor<32x128xf16>     // K=32 ≠ 64
%c = tt.dot %a, %b, ... :
    tensor<128x64xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
//  ^ 错误：A 的列数(64) ≠ B 的行数(32)

// 正确：K 维度匹配
%a = ... : tensor<128x64xf16>
%b = ... : tensor<64x128xf16>     // K=64 匹配
%c = tt.dot %a, %b, ... :
    tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
```

**规则 5：基本块终止**

```mlir
// 错误：基本块没有终止操作
^bb0:
    %a = arith.constant 1 : i32
    // 缺少 tt.return 或 cf.br

// 正确：基本块有终止操作
^bb0:
    %a = arith.constant 1 : i32
    tt.return
```

### 10.8.3 常见 IR 错误及修复

**错误 1：constexpr 类型不匹配**

```
错误信息: 'tt.make_range' op attribute 'end' must be a 32-bit integer

原因: BLOCK_SIZE 不是 constexpr
修复: 将 BLOCK_SIZE 声明为 tl.constexpr
```

```python
# 错误
@triton.jit
def bad_kernel(X, BLOCK_SIZE):
    offsets = tl.arange(0, BLOCK_SIZE)  # BLOCK_SIZE 不是 constexpr

# 正确
@triton.jit
def good_kernel(X, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)  # BLOCK_SIZE 是 constexpr
```

**错误 2：指针类型错误**

```
错误信息: 'tt.load' op operand #0 must be ptr type or tensor of ptr type

原因: 传递了非指针类型给 tt.load
修复: 确保 load 的第一个参数是指针或指针张量
```

**错误 3：归约区域 yield 类型不匹配**

```
错误信息: 'scf.yield' op types mismatch between yield and loop-carried values

原因: yield 的类型与 iter_args 的类型不匹配
修复: 确保 yield 的类型与输入张量的元素类型一致
```

### 10.8.4 调试技巧

**技巧 1：逐步打印 IR**

```bash
# 打印每个 pass 后的 IR
TRITON_PRINT_IR=1 python kernel.py 2>&1 | tee ir_dump.txt

# 只看第一个 pass 的输出
TRITON_PRINT_IR=1 python kernel.py 2>&1 | head -50
```

**技巧 2：使用 IR 验证**

```python
import triton
import triton.language as tl

@triton.jit
def kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x + 1, mask=mask)

# 编译时会自动验证 IR
# 如果 IR 有误，会在此处报错
kernel[(1,)](x, y, N, BLOCK=256)
```

**技巧 3：IR 比较**

```bash
# 生成两个版本的 IR 并比较差异
TRITON_PRINT_IR=1 python kernel_v1.py 2> ir_v1.txt
TRITON_PRINT_IR=1 python kernel_v2.py 2> ir_v2.txt
diff ir_v1.txt ir_v2.txt
```

<div data-component="IRDebuggingTools"></div>

[组件：IRDebuggingTools - 交互式 IR 调试工具，支持 IR dump 查看、验证和比较]

---

## 10.9 综合实例：完整 Kernel 的 IR 分析

### 10.9.1 向量加法的完整 IR 走读

让我们完整分析一个向量加法 kernel 从 Python 到 Triton IR 的转换：

**Python 源代码**：

```python
@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

**对应的完整 Triton IR**（带逐行注释）：

```mlir
module {
  tt.func @add_kernel(
      %arg0: !tt.ptr<f32>,                  // x_ptr
      %arg1: !tt_ptr<f32>,                  // y_ptr
      %arg2: !tt.ptr<f32>,                  // output_ptr
      %arg3: i32                             // n_elements
  ) attributes {noinline = false} {

    // ===== pid = tl.program_id(0) =====
    %0 = tt.program_id x : i32
    // %0 = 当前 block 在 x 维度的 ID

    // ===== block_start = pid * BLOCK_SIZE =====
    %c1024_i32 = arith.constant 1024 : i32
    // BLOCK_SIZE = 1024（编译期常量）
    %1 = arith.muli %0, %c1024_i32 : i32
    // %1 = pid * 1024

    // ===== offsets = block_start + tl.arange(0, BLOCK_SIZE) =====
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // %2 = [0, 1, 2, ..., 1023]
    %3 = tt.splat %1 : (i32) -> tensor<1024xi32>
    // %3 = [block_start, block_start, ..., block_start]  （广播）
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    // %4 = [block_start, block_start+1, ..., block_start+1023]

    // ===== mask = offsets < n_elements =====
    %5 = tt.splat %arg3 : (i32) -> tensor<1024xi32>
    // %5 = [n_elements, n_elements, ..., n_elements]  （广播）
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    // %6 = [True, True, ..., False, ...]  （越界位置为 False）

    // ===== x = tl.load(x_ptr + offsets, mask=mask) =====
    %7 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    // %7 = [x_ptr, x_ptr, ..., x_ptr]  （广播指针）
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // %8 = [x_ptr+0, x_ptr+1, ..., x_ptr+1023]  （指针加偏移）
    %cst = arith.constant 0.000000e+00 : f32
    // other = 0.0（掩码为 False 时的填充值）
    %9 = tt.load %8, %6, %cst : tensor<1024xf32>
    // %9 = [x[0], x[1], ..., x[N-1], 0.0, 0.0, ...]  （加载 x）

    // ===== y = tl.load(y_ptr + offsets, mask=mask) =====
    %10 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %12 = tt.load %11, %6, %cst : tensor<1024xf32>
    // %12 = [y[0], y[1], ..., y[N-1], 0.0, 0.0, ...]  （加载 y）

    // ===== output = x + y =====
    %13 = arith.addf %9, %12 : tensor<1024xf32>
    // %13 = [x[0]+y[0], x[1]+y[1], ...]  （逐元素加法）

    // ===== tl.store(output_ptr + offsets, output, mask=mask) =====
    %14 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %15, %13, %6 : tensor<1024xf32>
    // 将结果写入 output，越界位置不写入

    tt.return
  }
}
```

### 10.9.2 矩阵乘法的核心 IR 片段

以下是矩阵乘法 kernel 中最关键的部分——循环体内的 `tt.dot`：

```mlir
// 矩阵乘法核心循环
// C[128, 128] += A[128, 64] @ B[64, 128]

%c_init = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// 累加器初始化为全零

%result = scf.for %k = %c0 to %cK step %c64
    iter_args(%acc = %c_init) -> tensor<128x128xf32> {
    // ---- 加载 A 的块 ----
    %a_ptrs = ... : tensor<128x64x!tt.ptr<f16>>
    %a_block = tt.load %a_ptrs : tensor<128x64xf16>
    // 加载 A 的 128×64 子块

    // ---- 加载 B 的块 ----
    %b_ptrs = ... : tensor<64x128x!tt.ptr<f16>>
    %b_block = tt.load %b_ptrs : tensor<64x128xf16>
    // 加载 B 的 64×128 子块

    // ---- 矩阵乘法 ----
    %new_acc = tt.dot %a_block, %b_block, %acc
        {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} :
        tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
    // acc += A_block @ B_block
    // f16 输入，f32 累加（减少精度损失）

    scf.yield %new_acc : tensor<128x128xf32>
}

// %result 包含最终的 C 矩阵块
```

---

## 10.10 常见 IR 模式与惯用写法

### 10.10.1 地址计算的标准模式

在 Triton kernel 中，地址计算是一个重复出现的模式。以下是标准的地址计算 IR 模式：

```mlir
// 标准模式：一维连续地址计算
// Python: addr = base_ptr + pid * BLOCK + tl.arange(0, BLOCK)

// 步骤 1: 获取 program ID
%pid = tt.get_program_id x : i32

// 步骤 2: 计算 block 起始地址（标量级）
%block_start = arith.muli %pid, %BLOCK_i32 : i32

// 步骤 3: 创建 block 内偏移量（张量级）
%local_offsets = tt.make_range {end = %BLOCK : i32, start = 0 : i32} : tensor<?xi32>

// 步骤 4: 广播标量到张量
%block_start_vec = tt.splat %block_start : (i32) -> tensor<?xi32>

// 步骤 5: 计算全局偏移量
%global_offsets = arith.addi %block_start_vec, %local_offsets : tensor<?xi32>

// 步骤 6: 广播基址指针
%base_vec = tt.splat %base_ptr : (!tt.ptr<f32>) -> tensor<?x!tt.ptr<f32>>

// 步骤 7: 计算目标地址
%addrs = tt.addptr %base_vec, %global_offsets : tensor<?x!tt.ptr<f32>>, tensor<?xi32>
```

```mlir
// 标准模式：二维连续地址计算
// Python: addr = base_ptr + (pid_m * BM + arange(BM)) * stride_m + (pid_n * BN + arange(BN))

// M 维度索引
%pid_m = tt.get_program_id x : i32
%m_start = arith.muli %pid_m, %BM_i32 : i32
%m_local = tt.make_range {end = %BM : i32, start = 0 : i32} : tensor<?xi32>
%m_global_tmp = tt.splat %m_start : (i32) -> tensor<?xi32>
%m_indices = arith.addi %m_global_tmp, %m_local : tensor<?xi32>

// K 维度索引
%k_local = tt.make_range {end = %BK : i32, start = 0 : i32} : tensor<?xi32>

// 构造二维偏移量：m_indices 扩展为列向量，k_local 扩展为行向量
%m_expanded = tt.expand_dims %m_indices {axis = 1 : i32}
    : tensor<?xi32> -> tensor<?x1xi32>
%k_expanded = tt.expand_dims %k_local {axis = 0 : i32}
    : tensor<?xi32> -> tensor<1x?xi32>

// 广播得到完整的二维偏移量
// 实际实现中，编译器会利用广播语义避免显式扩展
```

### 10.10.2 掩码加载的标准模式

```mlir
// 标准模式：带边界检查的加载
// Python: data = tl.load(ptr + offsets, mask=offsets < N, other=0.0)

// 构造掩码
%N_vec = tt.splat %N : (i32) -> tensor<128xi32>
%mask = arith.cmpi slt, %global_offsets, %N_vec : tensor<128xi32>

// 构造填充值
%other = arith.constant dense<0.0> : tensor<128xf32>

// 带掩码加载
%data = tt.load %addrs, %mask, %other {
    cache = 1 : i32,    // ca: 缓存到所有级别
    evict = 1 : i32      // evict_normal
} : tensor<128xf32>

// 带掩码存储
tt.store %addrs, %data, %mask : tensor<128xf32>
```

### 10.10.3 累加循环的标准模式

```mlir
// 标准模式：矩阵乘法中的 K 维度累加
// Python:
//   acc = tl.zeros((BM, BN), dtype=tl.float32)
//   for k in range(0, K, BK):
//       a = tl.load(a_ptr + ...)
//       b = tl.load(b_ptr + ...)
//       acc = tl.dot(a, b, acc)

// 初始化累加器
%c0_i32 = arith.constant 0 : i32
%c1_i32 = arith.constant 1 : i32
%init_acc = tt.zeros {elem_type = f32, shape = [128, 128]} : tensor<128x128xf32>

// K 维度循环
%final_acc = scf.for %k = %c0_i32 to %K_i32 step %BK_i32
    iter_args(%acc = %init_acc) -> tensor<128x128xf32> {

  // 加载 A 的一个 tile: [BM × BK]
  %a_tile = ... : tensor<128x64xf16>

  // 加载 B 的一个 tile: [BK × BN]
  %b_tile = ... : tensor<64x128xf16>

  // 矩阵乘法并累加
  %updated_acc = tt.dot %a_tile, %b_tile, %acc
      : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>

  // 推进 Block Pointer
  %a_ptr_next = tt.advance %a_ptr, [%c0_i32, %BK_i32]
      : !tt.ptr<tensor<128x64xf16>>
  %b_ptr_next = tt.advance %b_ptr, [%BK_i32, %c0_i32]
      : !tt.ptr<tensor<64x128xf16>>

  scf.yield %updated_acc : tensor<128x128xf32>
}
```

### 10.10.4 归约 + Softmax 模式

```mlir
// 标准模式：行级 Softmax
// Python:
//   row = tl.load(ptr, mask=mask, other=-inf)
//   row_max = tl.max(row, axis=0)
//   numerator = tl.exp(row - row_max)
//   denominator = tl.sum(numerator, axis=0)
//   softmax = numerator / denominator

// 加载一行数据
%row = tl.load %row_ptrs, %mask, %other : tensor<128xf32>

// 沿 axis=0 归约求最大值
%row_max = tt.reduce %row {axis = 0 : i32, redOp = 1 : i32}
    : tensor<128xf32> -> f32

// 广播标量最大值到张量
%row_max_vec = tt.splat %row_max : (f32) -> tensor<128xf32>

// 计算 exp(row - max)
%shifted = arith.subf %row, %row_max_vec : tensor<128xf32>
%numerator = math.exp %shifted : tensor<128xf32>

// 沿 axis=0 归约求和
%denominator = tt.reduce %numerator {axis = 0 : i32, redOp = 0 : i32}
    : tensor<128xf32> -> f32

// 广播并计算 softmax
%denom_vec = tt.splat %denominator : (f32) -> tensor<128xf32>
%softmax = arith.divf %numerator, %denom_vec : tensor<128xf32>
```

### 10.10.5 scf.if 条件执行模式

```mlir
// 标准模式：条件边界检查
// Python:
//   if offsets < N:
//       tl.store(ptr, val)

%N_vec = tt.splat %N : (i32) -> tensor<128xi32>
%in_bounds = arith.cmpi slt, %offsets, %N_vec : tensor<128xi32>

// 方式 1: 使用掩码（推荐，避免分支发散）
tt.store %ptr, %val, %in_bounds : tensor<128xf32>

// 方式 2: 使用 scf.if（仅在条件对所有线程一致时使用）
// 判断是否所有元素都在边界内
%all_in_bounds = arith.constant true  // 实际需要归约判断
scf.if %all_in_bounds {
  tt.store %ptr, %val : tensor<128xf32>
} else {
  tt.store %ptr, %val, %in_bounds : tensor<128xf32>
}
```

---

## 10.11 IR 打印与调试技巧

### 10.11.1 IR Dump 配置

```bash
# 环境变量配置
export TRITON_PRINT_AUTOTUNING=1    # 打印自动调优过程
export TRITON_DUMP_IR=1             # dump IR 到文件
export TRITON_PRINT_IR=1            # 打印 IR 到标准输出

# IR dump 目录
# 默认保存到 /tmp/triton/dump/<kernel_name>/
# 包含多个阶段的 IR 文件：
#   - ttir.mlir          # Triton IR
#   - ttgir.mlir         # TritonGPU IR
#   - llir.mlir          # LLVM IR (MLIR 形式)
#   - llir.ll            # LLVM IR (标准形式)
#   - ptx.ptx            # PTX 汇编
#   - cubin.cubin        # 编译后的二进制
```

### 10.11.2 Python 中获取 IR

```python
import triton
import triton.compiler as compiler
import triton.language as tl

@triton.jit
def my_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)
    tl.store(Y_ptr + offsets, x + y, mask=mask)

# 获取编译后的 IR
src = compiler.ASTSource(
    fn=my_kernel,
    signature={"X_ptr": "*fp32", "Y_ptr": "*fp32", "N": "i32"},
    constants={"BLOCK": 1024}
)
compiled = compiler.compile(src)

# 打印各个阶段的 IR
print("=== Triton IR ===")
print(compiled.asm["ttir"])

print("=== TritonGPU IR ===")
print(compiled.asm["ttgir"])

print("=== LLVM IR ===")
print(compiled.asm["llir"])

print("=== PTX ===")
print(compiled.asm["ptx"])
```

### 10.11.3 IR 阅读技巧

**技巧一：从函数签名开始**

```mlir
// 首先看函数签名，了解参数类型
tt.func @my_kernel(
    %arg0: !tt.ptr<f32>,      // 第一个指针参数
    %arg1: !tt.ptr<f32>,      // 第二个指针参数
    %arg2: i32,               // 标量参数
    %arg3: i32                // constexpr 参数
) {
```

**技巧二：追踪数据流**

```mlir
// 从结果往回追踪
%9 = tt.load %8, %6 : tensor<1024xf32>
// %8 来自哪里？→ tt.addptr %7, %4
// %7 来自哪里？→ tt.splat %arg0
// %6 来自哪里？→ arith.cmpi slt, %4, %5
// 逐步回溯，理解数据流
```

**技巧三：识别常见模式**

```mlir
// 模式 1: 地址计算
%pid = tt.get_program_id x : i32
%start = arith.muli %pid, %BLOCK : i32
%offsets = tt.make_range ...
%splat = tt.splat %start
%global = arith.addi %splat, %offsets

// 模式 2: 掩码加载
%mask = arith.cmpi slt, %offsets, %N_vec
%data = tt.load %ptrs, %mask, %other

// 模式 3: 累加循环
%acc = scf.for %k = ... iter_args(%cur = %init) -> ... {
  %partial = tt.dot %a, %b, %cur
  scf.yield %partial
}
```

---

## 本章小结

本章系统介绍了 Triton IR 的设计目标、语法结构、类型系统和核心指令。以下是关键要点：

1. **Triton IR 的设计目标**：Triton IR 是一个 Tile 级别的 SSA IR，位于 Python 前端和 LLVM IR 之间。它保留了张量块（Tile）的语义信息，使得编译器可以在降低到标量之前进行高层优化（如 Tensor Core 映射、内存访问模式分析）。与 LLVM IR 的本质区别在于抽象级别——Triton IR 操作的是整个 Tile，而 LLVM IR 操作的是标量。

2. **语法结构**：Triton IR 基于 MLIR 框架，由 module → func → basic block → operation 四层组成。每个操作遵循 `%result = dialect.operation %operands {attributes} : type` 的统一格式。基本块以终止操作（tt.return、cf.br、cf.cond_br）结束。

3. **类型系统**：三大类型——标量类型（i1/i32/f16/f32 等）、张量类型（`tensor<MxNxT>`，编译期固定形状）、指针类型（`!tt.ptr<T>`，指向标量或张量）。张量类型是 Triton IR 的核心，映射到 GPU 寄存器。

4. **指令分类**：arith（算术，支持标量和张量逐元素运算）、math（超越函数）、tt（Triton 核心：load/store/dot/reduce/make_range/splat 等）、cf/scf（控制流：分支、循环、条件）。

5. **tt.dot**：执行 Tile 级矩阵乘法，支持 f16×f16→f32、bf16×bf16→f32、tf32×tf32→f32 等组合。编译后自动分解为多条 Tensor Core mma.sync 指令。

6. **tt.load/tt.store**：支持掩码（mask）实现边界检查、other 参数指定填充值、cache_modifier 和 eviction_policy 控制缓存行为。指针张量模式下，每个元素对应一个独立的内存地址。

7. **tt.reduce**：通过 reduction region（匿名函数）定义归约语义。支持 sum、max、min、any、all、argmax 等。编译后展开为树形归约 + warp shuffle。

8. **Python 到 IR 的转换**：Triton 的 JIT 编译器通过 AST Visitor 将 Python 代码转换为 Triton IR。每个 Python 语句（算术、load/store、reduce 等）都有对应的 IR 模式。使用 `TRITON_PRINT_IR=1` 可以查看生成的 IR。

9. **IR 验证**：mlir-opt 工具用于验证和转换 IR。主要验证规则包括 SSA 一致性、类型匹配、张量形状匹配、tt.dot 维度约束等。

---

## 思考题

### 概念理解题

1. **Triton IR vs LLVM IR**：为什么 Triton 选择引入自定义 IR 而不是直接生成 LLVM IR？从抽象级别、优化空间和硬件无关性三个角度分析。

2. **SSA 的必要性**：Triton IR 采用 SSA 形式有什么好处？如果没有 SSA（允许多次赋值），编译器优化会遇到什么困难？

3. **Tile 级 vs 标量级**：在 Triton IR 中，`arith.addf %a, %b : tensor<128xf32>` 对应 128 个并行加法操作。为什么不展开为 128 条标量指令？展开会丢失什么信息？

4. **tt.dot 的类型约束**：为什么 tt.dot 要求输入 A 的列数等于输入 B 的行数？如果放宽这个约束（允许自动 padding），编译器需要做什么额外工作？

5. **掩码加载的意义**：tt.load 的 mask 和 other 参数为什么是可选的？在什么场景下可以省略 mask？省略后编译器的行为有什么不同？

### 实践题

6. **IR 翻译**：将以下 Python 代码手动翻译为 Triton IR：

```python
@triton.jit
def scale_kernel(X, Y, alpha, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x * alpha, mask=mask)
```

7. **IR Dump 分析**：使用 `TRITON_PRINT_IR=1` 编译以下 kernel，分析 IR 中 tt.reduce 的具体形式：

```python
@triton.jit
def row_max_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = tl.load(X + pid * N + tl.arange(0, BLOCK),
                  mask=tl.arange(0, BLOCK) < N,
                  other=-float('inf'))
    max_val = tl.max(row, axis=0)
    tl.store(Y + pid, max_val)
```

8. **验证规则测试**：编写一个故意违反 Triton IR 类型匹配规则的 Python kernel（例如在 tt.dot 中使用不匹配的维度），观察编译器报错信息。

### 设计思考题

9. **自定义归约操作**：如果要在 tt.reduce 中实现"加权求和"（weighted sum），归约区域应该怎么写？与标准 sum 有什么区别？

10. **IR 优化设计**：假设你要为 Triton 编译器设计一个新的 IR 优化 pass——"常量折叠"（constant folding）。在 Triton IR 中，哪些模式可以被折叠？给出至少 3 个例子。

11. **跨硬件 IR**：Triton IR 是硬件无关的。如果要支持一个新的 AI 加速器（它没有 Tensor Core，但有向量单元），tt.dot 应该如何被降低？与 NVIDIA 后端有什么不同？

12. **Block Pointer 的优势**：Block Pointer 模式与传统指针张量模式在 IR 层面有什么区别？从 IR 指令数量和编译器优化空间的角度分析 Block Pointer 的优势。
