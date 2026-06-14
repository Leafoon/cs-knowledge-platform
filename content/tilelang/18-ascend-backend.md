---
title: "Chapter 18: 华为昇腾后端——Ascend C 适配"
description: "深入解析 TileLang 在华为昇腾 NPU 上的后端实现，涵盖达芬奇架构、Ascend C 编程模型、CANN 框架、多级 Lowering 及性能优化"
updated: 2026-06-11
---

> **Learning Objectives**
>
> - 理解昇腾 NPU 的达芬奇架构及其核心计算单元
> - 掌握 Cube Core 和 Vector Core 的协同工作模式
> - 了解 CANN 框架的整体架构与组件
> - 掌握 TileLang 到 Ascend C 的多级 Lowering 流程
> - 理解毕昇编译器（BiSheng）的优化能力
> - 掌握 AI Cube 内存层级（UB/L1/L2/GM）的使用方法
> - 学会编写 Ascend C 算子模板
> - 理解 Pipeline 调度与双核调度机制
> - 了解昇腾 910B/910C 的硬件特性
> - 掌握 TileLang 在昇腾上的 GEMM/Norm/Attention 实现
> - 掌握性能调优工具（msprof/ait）
> - 理解跨硬件迁移的最佳实践

---

## 18.1 昇腾 NPU 架构

### 18.1.1 达芬奇架构概述

华为昇腾（Ascend）NPU 采用自研的达芬奇（Da Vinci）架构，专为 AI 计算设计。与传统的 GPU 架构不同，达芬奇架构采用了独特的 Cube-Vector 分离计算模式。

```
┌──────────────────────────────────────────────────────────────┐
│                    达芬奇 (Da Vinci) 架构                      │
├──────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐    │
│  │                   AI Core (AIC)                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │    │
│  │  │  Cube Core   │  │ Vector Core  │  │ Scalar Core  │   │    │
│  │  │  (矩阵运算)   │  │ (向量运算)    │  │ (标量运算)    │   │    │
│  │  │  4096 MAC/cyc │  │ 2048 FP/cyc  │  │ 控制流       │   │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘   │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │              内存层级                          │    │    │
│  │  │  UB (Unified Buffer) │ L1 Buffer │ L2 Cache   │    │    │
│  │  └──────────────────────────────────────────────┘    │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                   AI CPU                              │    │
│  │              通用控制与调度                             │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                   Global Memory (HBM)                 │    │
│  │                   32GB / 64GB                         │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

该图展示了华为昇腾达芬奇架构的整体布局。最顶层是AI Core（AIC），包含三个核心计算单元：Cube Core负责矩阵运算，每周期可执行4096次乘累加；Vector Core负责向量运算，每周期处理2048次浮点操作；Scalar Core则处理控制流和标量操作。中间层是内存层级体系，从UB（统一缓冲区）到L1缓冲再到L2缓存，形成多级缓存结构。下方的AI CPU负责通用控制与调度，最底层是Global Memory（HBM），容量为32GB或64GB，作为片外主存。这种Cube-Vector分离的设计是达芬奇架构的核心创新。

### 18.1.2 Cube Core 详解

Cube Core 是达芬奇架构的核心计算单元，专门用于矩阵运算。

| 参数 | 说明 |
|------|------|
| 计算能力 | 每周期 4096 次乘累加（INT8） |
| FP16 吞吐 | 每周期 2048 次乘累加 |
| 矩阵形状 | 16×16 的基本计算块 |
| 数据类型 | INT8, FP16, BF16, FP32 |
| 执行模式 | 同步执行 |

```cpp
// Cube Core 矩阵乘法概念
// 两个 16x16 矩阵相乘，每个周期计算一个 16x16 的结果
struct CubeCore {
    // 16x16 的矩阵乘累加
    // A: 16xK, B: Kx16, C: 16x16
    static constexpr int TILE_M = 16;
    static constexpr int TILE_N = 16;
    static constexpr int TILE_K = 16;  // 可变

    // 每周期计算量（FP16）
    // 16 * 16 * 2 = 512 FLOPs（乘累加算两次）
    static constexpr int FLOPS_PER_CYCLE = 2048;
};
```

这段C++代码定义了Cube Core矩阵乘法的核心参数结构。TILE_M、TILE_N、TILE_K分别表示矩阵分块的行、列、内积维度，均为16，这是Cube Core的基本计算粒度。FLOPS_PER_CYCLE为2048，代表每周期可执行2048次FP16乘累加操作。实际计算中，每个周期Cube Core处理一个16×16的结果块，输入矩阵A和B的维度分别为16×K和K×16。设计时需要根据UB容量合理选择K维度的大小，以平衡数据复用率和内存占用。注意FP16精度下16×16×2=512 FLOPs，而乘累加算两次操作，因此标称值为2048。

### 18.1.3 Vector Core 详解

Vector Core 处理向量运算，如激活函数、归一化、逐元素操作等。

| 参数 | 说明 |
|------|------|
| 计算能力 | 每周期 2048 次 FP16 运算 |
| 数据类型 | FP16, FP32, INT32 |
| 向量宽度 | 128 元素（FP16） |
| 主要用途 | 逐元素运算、归约、类型转换 |

### 18.1.4 Cube Core 与 Vector Core 协同

<div data-component="CubeVectorDualCoreDiagram">

```
┌─────────────────────────────────────────────────────────────┐
│              Cube-Vector 双核协同执行                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐     ┌──────────────┐                     │
│  │  Cube Core    │     │ Vector Core   │                     │
│  │              │     │              │                     │
│  │  ┌────────┐  │     │  ┌────────┐  │                     │
│  │  │Matrix   │  │     │  │Activate │  │                     │
│  │  │Multiply │  │────▶│  │Function │  │                     │
│  │  └────────┘  │     │  └────────┘  │                     │
│  │              │     │              │                     │
│  │  ┌────────┐  │     │  ┌────────┐  │                     │
│  │  │Accumul- │  │     │  │Normalize│  │                     │
│  │  │ator     │  │     │  │         │  │                     │
│  │  └────────┘  │     │  └────────┘  │                     │
│  └──────────────┘     └──────────────┘                     │
│                                                             │
│  典型流程：                                                  │
│  1. Cube Core 计算矩阵乘法                                   │
│  2. 结果传入 Vector Core                                      │
│  3. Vector Core 执行激活/归一化                               │
│  4. 结果写回内存                                              │
└─────────────────────────────────────────────────────────────┘
```

该图展示了Cube Core与Vector Core的协同工作模式。Cube Core负责执行矩阵乘法运算，其累加器输出的结果会传递给Vector Core。Vector Core接收矩阵乘法结果后，执行激活函数（如ReLU、GELU）和归一化操作（如LayerNorm），最终将结果写回内存。这种流水线式的协作模式使得两个核心可以并行工作，最大化硬件利用率。典型的工作流程是：Cube Core计算矩阵乘法→结果传入Vector Core→Vector Core执行后处理→写回全局内存。

</div>

---

在了解了达芬奇架构的硬件基础之后，接下来需要掌握华为为昇腾NPU打造的软件生态——CANN框架。CANN是连接上层应用与底层硬件的桥梁，提供了从算子开发到模型部署的完整工具链。理解CANN的架构和组件，是进行TileLang-Ascend适配开发的必要前提。

## 18.2 CANN 框架

### 18.2.1 CANN 架构概述

CANN（Compute Architecture for Neural Networks）是华为为昇腾 NPU 打造的异构计算架构。

```
┌─────────────────────────────────────────────────────────────┐
│                      CANN 框架架构                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   应用层                              │    │
│  │   MindSpore │ PyTorch (NPU) │ TileLang │ ONNX       │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   框架适配层                           │    │
│  │   Graph Engine │ Tensor Engine │ Ascend Runtime      │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   编译器层                             │    │
│  │   毕昇编译器 (BiSheng) │ TBE │ TVM                   │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   驱动层                              │    │
│  │   Ascend Driver │ HAL                                │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   硬件层                              │    │
│  │   昇腾 310 │ 昇腾 910B │ 昇腾 910C                    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

该图展示了CANN框架的五层架构。最顶层是应用层，支持MindSpore、PyTorch（NPU版）、TileLang、ONNX等主流框架。框架适配层包含Graph Engine（图编译优化）、Tensor Engine（张量计算引擎）和Ascend Runtime（运行时环境）。编译器层是CANN的核心，包含毕昇编译器（基于LLM）、TBE和TVM。驱动层提供Ascend Driver和HAL硬件抽象。最底层是硬件层，支持昇腾310、910B、910C等芯片。TileLang通过TBE接入CANN框架，最终由毕昇编译器生成昇腾可执行代码。

### 18.2.2 CANN 核心组件

| 组件 | 功能 | 说明 |
|------|------|------|
| TBE (Tensor Boost Engine) | 算子开发框架 | 提供算子开发和优化工具 |
| Graph Engine | 图编译优化 | 计算图优化和调度 |
| Tensor Engine | 张量计算 | 张量级别的计算引擎 |
| AscendCL | 运行时 API | 类似 CUDA Runtime API |
| 毕昇编译器 | AI 编译器 | 基于 LLVM 的优化编译器 |
| HCCL | 集合通信 | 多卡/多机通信库 |

### 18.2.3 AscendCL 编程接口

```cpp
// AscendCL 基本使用示例
#include "acl/acl.h"

void ascendcl_example() {
    // 初始化 AscendCL
    aclInit(nullptr);

    // 设置设备
    aclrtSetDevice(0);

    // 创建流
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // 分配设备内存
    void* device_mem = nullptr;
    aclrtMalloc(&device_mem, 1024 * 1024, ACL_MEM_MALLOC_HUGE_FIRST);

    // 执行核函数
    // ...

    // 同步
    aclrtSynchronizeStream(stream);

    // 释放资源
    aclrtFree(device_mem);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}
```

这段C++代码展示了AscendCL的基本使用流程，类比于CUDA的Runtime API。首先通过aclInit初始化运行时环境，然后调用aclrtSetDevice选择设备（类似cudaSetDevice）。接着创建异步流用于数据传输和计算，使用aclrtMalloc分配设备内存（类似cudaMalloc）。核心计算完成后，调用aclrtSynchronizeStream进行流同步，确保计算完成。最后依次释放设备内存、销毁流、重置设备并终结运行时。AscendCL是编写昇腾算子时的基础编程接口，所有设备操作都通过它完成。

---

理解了CANN框架之后，接下来要解决的关键问题是如何将TileLang的高级抽象转换为昇腾NPU可执行的代码。TileLang采用多级Lowering策略，从Python DSL逐步降低到Ascend C源码，每一级Lowering都针对特定的硬件特性进行优化。这种分层设计既保留了高层的易用性，又能充分挖掘底层硬件的性能潜力。

## 18.3 TileLang-Ascend 适配

### 18.3.1 多级 Lowering 流程

<div data-component="AscendCLoweringFlow">

TileLang 到 Ascend C 的转换采用多级 Lowering 策略：

```
┌─────────────────────────────────────────────────────────────┐
│           TileLang → Ascend C 多级 Lowering                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Level 0: Python DSL                                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  @tilelang.jit                                        │    │
│  │  def kernel(A, B, C):                                 │    │
│  │      T.gemm(A, B, C)                                  │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         ▼                                    │
│  Level 1: TensorIR (TVM)                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  @T.prim_func                                         │    │
│  │  def matmul(A: T.Tensor, B: T.Tensor, C: T.Tensor):  │    │
│  │      for i, j, k in T.grid(M, N, K):                 │    │
│  │          C[i, j] += A[i, k] * B[k, j]                │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         ▼                                    │
│  Level 2: TileLang IR (Tiled + Memory)                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  - Tiled into blocks                                  │    │
│  │  - Shared memory allocation                           │    │
│  │  - Software pipelining                                │    │
│  │  - Thread binding                                     │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         ▼                                    │
│  Level 3: Ascend C IR                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  - Cube/Vector 分离                                   │    │
│  │  - 内存层级映射                                        │    │
│  │  - Pipeline 调度                                      │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         ▼                                    │
│  Level 4: Ascend C Source Code                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  // Ascend C 算子代码                                 │    │
│  │  __aicore__ void kernel(...) {                        │    │
│  │      // Cube Core 矩阵乘法                            │    │
│  │      Mmad(...);                                       │    │
│  │      // Vector Core 激活函数                           │    │
│  │      Adds(...);                                       │    │
│  │  }                                                    │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         ▼                                    │
│  Level 5: Binary (via BiSheng Compiler)                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  可在昇腾 NPU 上执行的二进制代码                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

该图展示了TileLang到Ascend C的完整六级Lowering流程。Level 0是Python DSL层，用户使用@tilelang.jit装饰器编写高层算子。Level 1将DSL转换为TensorIR（TVM的中间表示），形成循环嵌套结构。Level 2进行Tiling和内存管理，将数据分块并映射到UB/L1/GM内存层级。Level 3实现Cube/Vector分离和Pipeline调度，将计算分配到对应的核心。Level 4生成Ascend C源码，包含Mmad、Adds等硬件原语。Level 5通过毕昇编译器生成可执行的二进制代码。每一级转换都保持语义等价，同时逐步引入硬件特定的优化。

</div>

### 18.3.2 Lowering 各阶段详解

**Level 1 → Level 2：Tiling 与内存管理**

```python
# TileLang IR 中的 Tiling 和内存映射
def lower_to_tilelang_ir(tir_func, config):
    """
    将 TensorIR 转换为 TileLang IR

    关键变换：
    1. 分块（Tiling）：将循环分解为块级循环和块内循环
    2. 内存映射：将数据分配到 UB/L1/GM
    3. 软件流水线：重叠计算和数据传输
    """
    # 分块参数
    block_M = config.get("block_M", 128)
    block_N = config.get("block_N", 128)
    block_K = config.get("block_K", 32)

    # 内存映射策略
    memory_map = {
        "A_block": "UB",      # 输入块放在 UB
        "B_block": "UB",      # 输入块放在 UB
        "C_block": "UB",      # 输出块放在 UB
        "A_prefetch": "L1",   # 预取数据放在 L1
        "B_prefetch": "L1",
    }

    # Pipeline 配置
    pipeline_config = {
        "num_stages": 3,
        "prefetch_depth": 2,
    }

    return lowered_ir
```

这段Python代码展示了Level 1到Level 2的Lowering过程，即从TensorIR转换为TileLang IR。核心变换包括三个方面：分块（Tiling）将大循环分解为块级循环和块内循环，块级循环对应不同的AI Core，块内循环处理单个分块的数据。内存映射策略将输入输出数据分配到UB（统一缓冲区），预取数据放到L1缓存，利用多级缓存层次减少GM访问延迟。Pipeline配置设定3级流水和2级预取深度，使得数据搬运和计算可以重叠执行。block_M、block_N、block_K的大小选择需要根据UB容量和数据类型进行权衡。

**Level 2 → Level 3：Cube/Vector 分离**

```python
def separate_cube_vector(tilelang_ir):
    """
    将计算分离为 Cube Core 和 Vector Core 操作

    Cube Core 操作：
    - 矩阵乘法（GEMM）
    - 卷积

    Vector Core 操作：
    - 逐元素运算（加、乘、激活函数）
    - 归约（sum、max、mean）
    - 类型转换
    - Softmax、LayerNorm
    """
    cube_ops = []
    vector_ops = []

    for op in tilelang_ir.operations:
        if op.type in ["matmul", "conv2d", "bmm"]:
            cube_ops.append(op)
        elif op.type in ["add", "mul", "relu", "gelu", "softmax", "layernorm"]:
            vector_ops.append(op)
        elif op.type == "fused_matmul_bias_activation":
            # 拆分为 Cube 和 Vector
            cube_ops.append(op.matmul_part)
            vector_ops.append(op.bias_activation_part)

    return cube_ops, vector_ops
```

这段代码实现了Level 2到Level 3的Cube/Vector分离。它遍历TileLang IR中的所有操作，根据操作类型将其分配到对应的核心：矩阵乘法（matmul）、卷积（conv2d）、批矩阵乘（bmm）等计算密集型操作分配给Cube Core；逐元素运算（add、mul）、激活函数（relu、gelu）、归约（softmax、layernorm）等向量化操作分配给Vector Core。特别注意融合操作（如fused_matmul_bias_activation）需要拆分为两部分：矩阵乘部分给Cube Core，偏置和激活部分给Vector Core。这种分离是达芬奇架构编程的核心步骤。

### 18.3.3 IR 转换示例

```python
# 完整的 Lowering 示例：GEMM + Bias + ReLU
import tilelang
from tilelang import T

@tilelang.jit(
    out_idx=[3],
    target="ascend",
    device="ascend910b",
)
def gemm_bias_relu_ascend(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
):
    @T.prim_func
    def main(
        A: T.Tensor([M, K], "float16"),
        B: T.Tensor([K, N], "float16"),
        Bias: T.Tensor([N], "float16"),
        C: T.Tensor([M, N], "float16"),
    ):
        with T.Blocks([T.ceildiv(N, block_N), T.ceildiv(M, block_M)]) as (bn, bm):
            # UB 中的缓冲区
            A_ub = T.alloc_shared([block_M, block_K], "float16", scope="UB")
            B_ub = T.alloc_shared([block_K, block_N], "float16", scope="UB")
            C_ub = T.alloc_fragment([block_M, block_N], "float32", scope="UB")

            # 初始化累加器
            T.clear(C_ub)

            # K 维度循环
            for k in T.serial(T.ceildiv(K, block_K)):
                # GM → UB 数据搬运
                T.copy(A[bm * block_M:, k * block_K:], A_ub)
                T.copy(B[k * block_K:, bn * block_N:], B_ub)
                T.sync()

                # Cube Core: 矩阵乘法
                T.gemm(A_ub, B_ub, C_ub, k_axis=block_K)

            # Vector Core: Bias + ReLU
            for i, j in T.Parallel(block_M, block_N):
                val = C_ub[i, j] + Bias[bn * block_N + j]
                C_ub[i, j] = T.max(val, 0.0)  # ReLU

            # UB → GM 写回
            T.copy(C_ub, C[bm * block_M:, bn * block_N:])

    return main
```

这段代码展示了GEMM+Bias+ReLU融合算子在TileLang中的完整Lowering实现。整个计算流程为：在K维度循环中，每次从GM搬运A和B的分块到UB，通过T.sync()确保数据就绪，然后调用T.gemm执行Cube Core矩阵乘法。K循环结束后，在Vector Core上并行计算Bias加法和ReLU激活。最后将结果从UB写回GM。累加器C_ub使用FP32精度以避免精度损失，而输入输出使用FP16以提高带宽利用率。T.alloc_fragment声明的缓冲区位于UB中，是Cube Core和Vector Core都能直接访问的共享存储。

---

TileLang到Ascend C的Lowering过程完成后，生成的Ascend C源码还需要经过毕昇编译器的优化才能高效执行。毕昇编译器是华为自研的AI编译器，针对昇腾NPU的Cube-Vector架构进行了深度优化，能够自动进行Tiling、流水线调度、内存分配等优化。理解毕昇编译器的工作原理，有助于编写出性能更优的算子代码。

## 18.4 毕昇编译器（BiSheng）

### 18.4.1 毕昇编译器概述

毕昇编译器是华为自研的 AI 编译器，基于 LLVM 架构，专门针对昇腾 NPU 优化。

```
┌─────────────────────────────────────────────────────────────┐
│                  毕昇编译器架构                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Ascend C    │    │  TBE DSL    │    │  其他前端     │     │
│  │  Source Code │    │  (Python)   │    │             │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                LLVM IR (中间表示)                      │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              AI 专用优化 Pass                          │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │    │
│  │  │Tiling   │ │Pipeline │ │Memory   │ │Compute  │   │    │
│  │  │Optimize │ │Schedule │ │Manage   │ │Fusion   │   │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              目标代码生成                               │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐               │    │
│  │  │Cube Core│ │Vector   │ │Scalar   │               │    │
│  │  │Code     │ │Core Code│ │Core Code│               │    │
│  │  └─────────┘ └─────────┘ └─────────┘               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

该图展示了毕昇编译器的架构。顶层支持多种前端输入，包括Ascend C源码、TBE DSL（Python）和其他前端。前端代码被编译为LLVM IR（中间表示），然后经过一系列AI专用优化Pass：Tiling优化将大计算分块以匹配硬件能力，Pipeline Pass生成软件流水线调度，Memory Pass优化内存分配减少GM访问，Compute Fusion Pass将多个算子融合以减少中间结果写回。最终，编译器生成三种目标代码：Cube Core代码、Vector Core代码和Scalar Core代码，分别对应矩阵运算、向量运算和控制流。

### 18.4.2 关键优化 Pass

| Pass | 功能 | 效果 |
|------|------|------|
| Tiling Pass | 循环分块优化 | 提升数据局部性 |
| Pipeline Pass | 软件流水调度 | 重叠计算与访存 |
| Memory Pass | 内存分配优化 | 减少 GM 访问 |
| Fusion Pass | 算子融合 | 减少中间结果写回 |
| Vectorization Pass | 向量化 | 充分利用 Vector Core |
| Double Buffer Pass | 双缓冲 | 隐藏数据搬运延迟 |

### 18.4.3 编译选项与优化级别

```bash
# 毕昇编译器使用示例
# 编译 Ascend C 算子
ascendc_compile \
    --input kernel.cpp \
    --output kernel.o \
    --target ascend910b \
    --opt-level 3 \
    --enable-pipeline \
    --enable-double-buffer

# 优化级别说明
# -O0: 无优化（调试用）
# -O1: 基础优化
# -O2: 中等优化（推荐开发）
# -O3: 最高优化（推荐生产）
```

这段Bash命令展示了毕昇编译器的基本使用方法。ascendc_compile命令将Ascend C源码编译为目标文件，--target指定目标芯片（如ascend910b），--opt-level控制优化级别（-O3为最高优化），--enable-pipeline和--enable-double-buffer分别启用软件流水和双缓冲优化。优化级别从-O0到-O3递增：-O0无优化用于调试，-O1基础优化，-O2中等优化适合开发阶段，-O3最高优化适合生产环境。生产环境应始终使用-O3以获得最佳性能。

---

毕昇编译器负责将Ascend C代码优化为高效的二进制代码，而性能优化的关键在于合理利用昇腾NPU的多级内存层级。从寄存器到UB、L1、L2再到全局内存，每一级存储都有不同的容量、带宽和延迟特性。理解这些特性并合理规划数据在各级存储间的搬运策略，是编写高性能Ascend C算子的基础。

## 18.5 AI Cube 内存层级

### 18.5.1 内存层级概述

<div data-component="AscendMemoryHierarchy">

```
┌─────────────────────────────────────────────────────────────┐
│                AI Cube 内存层级                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Global Memory (GM)                       │    │
│  │              容量: 32GB / 64GB (HBM)                  │    │
│  │              带宽: 1.2TB/s (910B) / 1.6TB/s (910C)   │    │
│  │              延迟: ~400 cycles                         │    │
│  │              用途: 模型权重、激活值、中间结果              │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              L2 Cache                                 │    │
│  │              容量: 每 AIC 32MB                         │    │
│  │              带宽: 非常高                               │    │
│  │              延迟: ~50 cycles                          │    │
│  │              用途: 热数据缓存                           │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              L1 Buffer                                │    │
│  │              容量: 每 AIC 512KB                        │    │
│  │              带宽: 非常高                               │    │
│  │              延迟: ~20 cycles                          │    │
│  │              用途: 数据预取和暂存                        │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              UB (Unified Buffer)                      │    │
│  │              容量: 每 AIC 256KB                        │    │
│  │              带宽: 最高                                 │    │
│  │              延迟: ~5 cycles                           │    │
│  │              用途: Cube/Vector Core 直接访问             │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              寄存器 (Register)                         │    │
│  │              每个 Core 独立                             │    │
│  │              延迟: 1 cycle                              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

该图展示了AI Cube的五级内存层级。最顶层是寄存器（Register），每个Core独立拥有，延迟仅1 cycle，用于存放当前计算的临时数据。UB（Unified Buffer）容量256KB/AIC，带宽最高约10 TB/s，延迟约5 cycles，是Cube Core和Vector Core直接访问的片上存储，也是算子开发中最常用的存储空间。L1 Buffer容量512KB/AIC，带宽约5 TB/s，延迟约20 cycles，用于数据预取和暂存。L2 Cache容量32MB/AIC，带宽约2 TB/s，延迟约50 cycles，作为热数据缓存。最底层是Global Memory（HBM），容量32-64GB，带宽1.2-1.6 TB/s，延迟约400 cycles，用于存放模型权重和激活值。数据在这些层级间的搬运需要通过DMA引擎完成。

</div>

### 18.5.2 各层内存特性对比

| 层次 | 容量 | 带宽 | 延迟 | 访问方式 |
|------|------|------|------|----------|
| 寄存器 | 每 Core 几 KB | 最高 | 1 cycle | 直接访问 |
| UB | 256KB/AIC | ~10 TB/s | ~5 cycles | DMA/直接访问 |
| L1 | 512KB/AIC | ~5 TB/s | ~20 cycles | DMA |
| L2 | 32MB/AIC | ~2 TB/s | ~50 cycles | 自动缓存 |
| GM | 32-64GB | 1.2-1.6 TB/s | ~400 cycles | DMA |

### 18.5.3 内存管理策略

```cpp
// Ascend C 内存管理示例
__aicore__ void memory_management_example(
    GM_ADDR input, GM_ADDR output
) {
    // 1. 声明内存队列
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue;   // 输入队列（UB）
    TQue<QuePosition::VECOUT, 1> outQueue; // 输出队列（UB）

    // 2. 分配缓冲区
    TBuf<TPosition::VECCALC> calcBuf;     // 计算缓冲区（UB）

    pipe.InitBuffer(inQueue, 1, 256 * 1024);   // 256KB 输入缓冲
    pipe.InitBuffer(outQueue, 1, 256 * 1024);  // 256KB 输出缓冲
    pipe.InitBuffer(calcBuf, 256 * 1024);      // 256KB 计算缓冲

    // 3. 数据搬运 GM → UB
    LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
    DataCopy(inputLocal, input, 1024);  // 搬运 1024 个 float
    inQueue.EnQue(inputLocal);

    // 4. 从队列取出数据
    LocalTensor<float> inputBuf = inQueue.DeQue<float>();

    // 5. 计算（Vector Core）
    LocalTensor<float> calcBufLocal = calcBuf.Get<float>();
    Adds(calcBufLocal, inputBuf, 1.0f, 1024);

    // 6. 结果写入输出队列
    LocalTensor<float> outputLocal = outQueue.AllocTensor<float>();
    Adds(outputLocal, calcBufLocal, 0.0f, 1024);
    outQueue.EnQue(outputLocal);

    // 7. 数据搬运 UB → GM
    LocalTensor<float> outputBuf = outQueue.DeQue<float>();
    DataCopy(output, outputBuf, 1024);

    // 8. 释放资源
    inQueue.FreeTensor(inputBuf);
    outQueue.FreeTensor(outputBuf);
}
```

这段C++代码详细展示了Ascend C的内存管理机制。首先通过TPipe创建内存管道，然后声明输入输出队列（TQue）和计算缓冲区（TBuf）。队列使用生产者-消费者模式管理UB中的缓冲区：通过AllocTensor分配缓冲，EnQue将数据放入队列等待消费，DeQue取出数据进行计算，FreeTensor释放缓冲区。DataCopy函数执行GM到UB的数据搬运，通过DMA引擎实现异步传输。注意输入队列深度为1（非双缓冲），在实际高性能算子中通常使用深度为2的双缓冲来隐藏数据搬运延迟。整个流程遵循"分配→搬运→计算→释放"的标准模式。

---

## 18.6 Ascend C 算子模板

### 18.6.1 算子模板结构

```cpp
// 标准 Ascend C 算子模板
#include "kernel_operator.h"

using namespace AscendC;

// 算子参数结构
struct KernelArgs {
    GM_ADDR input_a;
    GM_ADDR input_b;
    GM_ADDR output;
    int32_t M;
    int32_t N;
    int32_t K;
};

// 核函数入口
extern "C" __global__ __aicore__
void kernel_main(KernelArgs args) {
    // 获取块索引
    int blockIdx = GetBlockIdx();

    // 计算本块负责的数据范围
    int block_M = 128;
    int block_N = 128;
    int block_K = 32;

    int num_blocks_n = (args.N + block_N - 1) / block_N;
    int bm = blockIdx / num_blocks_n;
    int bn = blockIdx % num_blocks_n;

    // 调用计算核心
    gemm_kernel(
        args.input_a, args.input_b, args.output,
        bm, bn, args.M, args.N, args.K,
        block_M, block_N, block_K
    );
}

// GEMM 计算核心
__aicore__ void gemm_kernel(
    GM_ADDR A, GM_ADDR B, GM_ADDR C,
    int bm, int bn, int M, int N, int K,
    int block_M, int block_N, int block_K
) {
    // 内存管理
    TPipe pipe;
    TQue<QuePosition::A1, 1> aQueue;    // A 矩阵输入队列
    TQue<QuePosition::A2, 1> bQueue;    // B 矩阵输入队列
    TQue<QuePosition::CO1, 1> cQueue;   // C 矩阵输出队列

    pipe.InitBuffer(aQueue, 1, block_M * block_K * sizeof(half));
    pipe.InitBuffer(bQueue, 1, block_K * block_N * sizeof(half));
    pipe.InitBuffer(cQueue, 1, block_M * block_N * sizeof(float));

    // 累加器（UB 中）
    TBuf<TPosition::CO2> accBuf;
    pipe.InitBuffer(accBuf, block_M * block_N * sizeof(float));
    LocalTensor<float> acc = accBuf.Get<float>();

    // 初始化累加器
    Duplicate(acc, 0.0f, block_M * block_N);

    // K 维度循环
    for (int k = 0; k < (K + block_K - 1) / block_K; k++) {
        // 加载 A 块到 UB
        LocalTensor<half> aLocal = aQueue.AllocTensor<half>();
        DataCopy(aLocal, A + (bm * block_M * K + k * block_K),
                 {block_M, block_K, K - block_K, 0});
        aQueue.EnQue(aLocal);

        // 加载 B 块到 UB
        LocalTensor<half> bLocal = bQueue.AllocTensor<half>();
        DataCopy(bLocal, B + (k * block_K * N + bn * block_N),
                 {block_K, block_N, N - block_N, 0});
        bQueue.EnQue(bLocal);

        // 从队列取出
        LocalTensor<half> aBuf = aQueue.DeQue<half>();
        LocalTensor<half> bBuf = bQueue.DeQue<half>();

        // Cube Core: 矩阵乘累加
        Mmad(acc, aBuf, bBuf, block_M, block_N, block_K);

        // 释放输入缓冲
        aQueue.FreeTensor(aBuf);
        bQueue.FreeTensor(bBuf);
    }

    // 类型转换 FP32 → FP16
    LocalTensor<half> cLocal = cQueue.AllocTensor<half>();
    Cast(cLocal, acc, RoundMode::CAST_NONE, block_M * block_N);
    cQueue.EnQue(cLocal);

    // 写回 GM
    LocalTensor<half> cBuf = cQueue.DeQue<half>();
    DataCopy(C + (bm * block_M * N + bn * block_N),
             cBuf, {block_M, block_N, 0, N - block_N});
    cQueue.FreeTensor(cBuf);
}
```

上述Ascend C算子模板完整展示了标准GEMM核函数的生命周期：从内存管道初始化、UB缓冲区分配到Cube Core矩阵乘累加计算，再到Vector Core类型转换和最终写回全局内存。其中`Mmad`指令是Cube Core的核心原语，它在硬件层面将两个16×16的分块矩阵相乘并累加到FP32累加器中，充分利用了达芬奇架构每周期4096次INT8乘累加的惊人算力。双缓冲队列（`TQue`深度为2）的设计允许DMA引擎在当前数据块被计算时预先加载下一块数据，从而隐藏GM到UB的数据搬运延迟。这种“以空间换时间”的策略在昇腾算子优化中至关重要，直接决定了算子能否接近理论峰值性能。TileLang的多级Lowering将上述底层细节完全封装，开发者只需用Python DSL描述计算逻辑即可。

### 18.6.2 TileLang 算子模板封装

```python
# TileLang 中的 Ascend C 算子模板
import tilelang
from tilelang import T

@tilelang.jit(target="ascend")
def norm_ascend_template(
    batch_size: int,
    hidden_size: int,
    eps: float = 1e-5,
    dtype: str = "float16",
):
    @T.prim_func
    def main(
        X: T.Tensor([batch_size, hidden_size], dtype),
        W: T.Tensor([hidden_size], dtype),
        B: T.Tensor([hidden_size], dtype),
        Y: T.Tensor([batch_size, hidden_size], dtype),
    ):
        # 每个 block 处理一行
        with T.Blocks([batch_size]) as (b,):
            # UB 分配
            X_ub = T.alloc_fragment([hidden_size], dtype, scope="UB")
            W_ub = T.alloc_fragment([hidden_size], dtype, scope="UB")
            B_ub = T.alloc_fragment([hidden_size], dtype, scope="UB")
            mean_ub = T.alloc_fragment([1], "float32", scope="UB")
            var_ub = T.alloc_fragment([1], "float32", scope="UB")

            # 加载数据
            T.copy(X[b, :], X_ub)
            T.copy(W, W_ub)
            T.copy(B, B_ub)

            # Vector Core: 计算均值
            T.reduce_mean(X_ub, mean_ub, axis=0)

            # Vector Core: 计算方差
            for i in T.Parallel(hidden_size):
                X_ub[i] = (X_ub[i] - mean_ub[0]) * (X_ub[i] - mean_ub[0])
            T.reduce_mean(X_ub, var_ub, axis=0)

            # Vector Core: 归一化
            rsqrt_val = T.rsqrt(var_ub[0] + eps)
            for i in T.Parallel(hidden_size):
                X_ub[i] = (X_ub[i] * rsqrt_val) * W_ub[i] + B_ub[i]

            # 写回
            T.copy(X_ub, Y[b, :])

    return main
```

上述TileLang算子模板封装了LayerNorm在昇腾后端上的完整计算流程。整个计算全部在Vector Core上完成，无需Cube Core介入：首先通过`T.reduce_mean`统计均值，然后在并行循环中计算每个元素与均值的偏差平方，再次归约得到方差，最后使用`T.rsqrt`计算平方根倒数并执行归一化与仿射变换。这种将多个向量归约和逐元素操作集中在一个核函数中的做法，避免了中间结果的GM读写开销，是向量密集型算子在达芬奇架构上的最佳实践。TileLang通过声明式的`T.prim_func`和`T.Blocks`抽象，将内存分配、数据搬运和向量计算的过程以统一语法表达，大大降低了昇腾算子开发的门槛。

---

### 18.7.1 软件流水线原理

Pipeline 调度是昇腾 NPU 性能优化的关键技术，通过重叠数据搬运和计算来隐藏延迟。

```
┌─────────────────────────────────────────────────────────────┐
│              三级 Pipeline 调度示例                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  时间 →                                                     │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐        │
│  │ S0  │ S1  │ S2  │ S3  │ S4  │ S5  │ S6  │ S7  │        │
│  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤        │
│  │Load │Load │Load │Load │Load │Load │     │     │  Stage1 │
│  │  0  │  1  │  2  │  3  │  4  │  5  │     │     │  (数据  │
│  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤  搬运)  │
│  │     │Comp │Comp │Comp │Comp │Comp │Comp │     │  Stage2 │
│  │     │  0  │  1  │  2  │  3  │  4  │  5  │     │  (计算)  │
│  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤        │
│  │     │     │Stor │Stor │Stor │Stor │Stor │Stor │  Stage3 │
│  │     │     │  0  │  1  │  2  │  3  │  4  │  5  │  (存储)  │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘        │
│                                                             │
│  理想情况下，每个周期可以同时执行 Load、Compute、Store          │
└─────────────────────────────────────────────────────────────┘
```

该三级流水线时序图直观展示了软件流水线（Software Pipeline）在昇腾NPU上的执行原理。每个Stage对应流水线的一级：Stage1负责从全局内存加载数据到UB（Load），Stage2在Cube Core或Vector Core上执行计算（Compute），Stage3将计算结果写回全局内存（Store）。理想情况下，在同一时刻三个Stage并行工作——当Stage2正在计算第i块数据时，Stage1已经在加载第i+1块数据，Stage3同时写出了第i-1块的结果。这种深度重叠将原本串行的“加载→计算→存储”流程压缩为流水线，使得算子的实际吞吐量逼近内存带宽和计算能力的上限。在实际算子开发中，流水线深度（num_stages）通常设为2到3级，需要根据UB容量和数据量进行权衡。

### 18.7.2 Pipeline 实现

```python
# TileLang 中的 Pipeline 调度
@tilelang.jit(target="ascend")
def gemm_with_pipeline(
    M: int, N: int, K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
):
    @T.prim_func
    def main(
        A: T.Tensor([M, K], "float16"),
        B: T.Tensor([K, N], "float16"),
        C: T.Tensor([M, N], "float32"),
    ):
        with T.Blocks([T.ceildiv(N, block_N), T.ceildiv(M, block_M)]) as (bn, bm):
            # 多级缓冲（Pipeline）
            A_buffers = [
                T.alloc_shared([block_M, block_K], "float16", scope="UB")
                for _ in range(num_stages)
            ]
            B_buffers = [
                T.alloc_shared([block_K, block_N], "float16", scope="UB")
                for _ in range(num_stages)
            ]

            # 累加器
            C_local = T.alloc_fragment([block_M, block_N], "float32", scope="UB")
            T.clear(C_local)

            # Pipeline 注解
            T.annotate_schedule({
                "software_pipeline_stage": [
                    [0, 0],      # Load 阶段
                    [1, 1],      # Compute 阶段
                    [2, 2],      # Store 阶段
                ],
                "software_pipeline_order": [0, 1, 2],
            })

            for k in T.serial(T.ceildiv(K, block_K)):
                stage = k % num_stages

                # Load 阶段：异步加载数据
                T.copy(A[bm * block_M:, k * block_K:], A_buffers[stage])
                T.copy(B[k * block_K:, bn * block_N:], B_buffers[stage])

                # Compute 阶段：矩阵乘法
                T.gemm(A_buffers[stage], B_buffers[stage], C_local)

            T.copy(C_local, C[bm * block_M:, bn * block_N:])

    return main
```

上述代码通过多级缓冲区（A_buffers和B_buffers列表）和`T.annotate_schedule`注解实现了三级软件流水线。每次K维度循环迭代对应一个流水线stage，当前stage使用专用的UB缓冲区加载数据并执行矩阵乘法，而其他stage的缓冲可用于数据预取或结果写回。`software_pipeline_stage`注解显式指定了Load（阶段0）、Compute（阶段1）和Store（阶段2）的分配策略，`software_pipeline_order`规定了流水线的执行顺序。这种声明式调度方式让TileLang自动处理buffer轮转、同步插入和依赖分析，开发者无需手动管理复杂的多缓冲逻辑，即可获得与手写Ascend C相当的流水线性能。

---

### 18.8.1 双核协同模式

昇腾 NPU 的 Cube Core 和 Vector Core 可以并行执行，通过双核调度最大化硬件利用率。

```python
# Cube-Vector 双核调度示例
@tilelang.jit(target="ascend")
def gemm_with_activation_dual_core(
    M: int, N: int, K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
):
    @T.prim_func
    def main(
        A: T.Tensor([M, K], "float16"),
        B: T.Tensor([K, N], "float16"),
        C: T.Tensor([M, N], "float16"),
    ):
        with T.Blocks([T.ceildiv(N, block_N), T.ceildiv(M, block_M)]) as (bn, bm):
            # 输入缓冲（UB）
            A_ub = T.alloc_shared([block_M, block_K], "float16", scope="UB")
            B_ub = T.alloc_shared([block_K, block_N], "float16", scope="UB")

            # Cube Core 输出缓冲
            C_cube = T.alloc_fragment([block_M, block_N], "float32", scope="CO1")

            # Vector Core 输出缓冲
            C_vec = T.alloc_fragment([block_M, block_N], "float16", scope="UB")

            T.clear(C_cube)

            for k in T.serial(T.ceildiv(K, block_K)):
                # 数据加载（DMA 引擎）
                T.copy(A[bm * block_M:, k * block_K:], A_ub)
                T.copy(B[k * block_K:, bn * block_N:], B_ub)
                T.sync()

                # Cube Core: 矩阵乘法
                T.gemm(A_ub, B_ub, C_cube, k_axis=block_K,
                       core="cube")

            # Vector Core: 激活函数（与下一块的 Cube 计算并行）
            for i, j in T.Parallel(block_M, block_N):
                val = C_cube[i, j]
                # GELU 激活函数
                C_vec[i, j] = 0.5 * val * (1.0 + T.tanh(
                    0.7978845608 * (val + 0.044715 * val * val * val)
                ))

            T.copy(C_vec, C[bm * block_M:, bn * block_N:])

    return main
```

该示例展示了Cube-Vector双核调度在GEMM+GELU融合算子中的实际应用。关键设计在于：Cube Core负责矩阵乘法（`core="cube"`），其输出缓冲`C_cube`位于CO1空间（Cube Core的专用输出区域）；Vector Core随后并行处理GELU激活函数，直接从CO1空间读取Cube中间结果，避免了GM写回和再次读取的开销。GELU函数的实现使用近似公式`0.5*x*(1+tanh(0.7978845608*(x+0.044715*x*x*x)))`，全部由Vector Core的逐元素运算完成。这种双核协作模式在Transformer类模型中极为常见——每个attention头的QK^T和PV乘法由Cube Core完成，紧随其后的softmax和dropout在Vector Core上执行，两者形成天然的流水线。

### 18.8.2 双核调度策略

```python
def schedule_cube_vector_dual_core(operations):
    """
    将操作分配到 Cube Core 和 Vector Core

    调度策略：
    1. Cube Core 优先处理矩阵运算
    2. Vector Core 处理逐元素运算
    3. 尽量重叠 Cube 和 Vector 的执行
    """
    schedule = {
        "cube": [],    # Cube Core 操作
        "vector": [],  # Vector Core 操作
        "dma": [],     # DMA 操作
    }

    for op in operations:
        if op.type in ["matmul", "conv2d", "bmm"]:
            schedule["cube"].append(op)
        elif op.type in ["add", "mul", "relu", "gelu", "softmax", "layernorm"]:
            schedule["vector"].append(op)
        elif op.type in ["copy", "transpose"]:
            schedule["dma"].append(op)

    # 优化：重叠 Cube 和 Vector 执行
    optimized = overlap_cube_vector(schedule)

    return optimized

def overlap_cube_vector(schedule):
    """
    重叠 Cube 和 Vector 执行

    示例时间线：
    时间 0: [Cube] matmul_0  [Vector] idle
    时间 1: [Cube] matmul_1  [Vector] relu_0
    时间 2: [Cube] matmul_2  [Vector] gelu_1
    时间 3: [Cube] idle      [Vector] softmax_2
    """
    timeline = []
    cube_queue = list(schedule["cube"])
    vector_queue = list(schedule["vector"])

    while cube_queue or vector_queue:
        step = {"cube": None, "vector": None}

        if cube_queue:
            step["cube"] = cube_queue.pop(0)
        if vector_queue:
            step["vector"] = vector_queue.pop(0)

        timeline.append(step)

    return timeline
```

这段代码服务于 18.8.2 双核调度策略 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

`schedule_cube_vector_dual_core`函数实现了双核调度的静态分配策略：遍历所有操作节点，根据op.type将其分入cube、vector或dma三个执行队列。`overlap_cube_vector`是核心优化函数，它模拟了流水线化的时间线：在时间步0，Cube执行第一个矩阵乘法时Vector空闲；时间步1，Cube执行第二个矩阵乘法的同时Vector开始处理第一个relu；时间步2同理。这种交错调度使得Cube和Vector的利用率最大化。在实际部署中，Cube Core和Vector Core的硬件调度由Ascend Runtime自动管理，TileLang的任务是生成正确的双核指令序列并插入必要的同步原语，确保数据依赖关系正确的同时不引入不必要的等待。

---

### 18.9.1 昇腾 910B 规格

| 参数 | 数值 | 说明 |
|------|------|------|
| AI Core 数量 | 32 | 每芯片 32 个 AIC |
| FP16 吞吐 | 320 TFLOPS | 矩阵运算 |
| INT8 吞吐 | 640 TOPS | 整数运算 |
| HBM 容量 | 64GB | HBM2e |
| HBM 带宽 | 1.2 TB/s | 8 个 HBM 通道 |
| L2 缓存 | 32MB/AIC | 共享缓存 |
| UB 容量 | 256KB/AIC | 统一缓冲 |
| TDP | 400W | 典型功耗 |

### 18.9.2 昇腾 910C 增强特性

| 参数 | 数值 | 相比 910B |
|------|------|-----------|
| AI Core 数量 | 32 | 相同 |
| FP16 吞吐 | 400 TFLOPS | +25% |
| INT8 吞吐 | 800 TOPS | +25% |
| HBM 容量 | 96GB | +50% |
| HBM 带宽 | 1.6 TB/s | +33% |
| L2 缓存 | 64MB/AIC | 2x |
| UB 容量 | 512KB/AIC | 2x |
| FP8 支持 | 是 | 新增 |

### 18.9.3 针对不同芯片的优化

```python
# 针对不同昇腾芯片的 TileLang 配置
class AscendChipConfig:
    CONFIGS = {
        "ascend910b": {
            "max_ub_size": 256 * 1024,      # 256KB
            "max_l2_size": 32 * 1024 * 1024, # 32MB
            "hbm_bandwidth": 1.2e12,         # 1.2 TB/s
            "fp16_tflops": 320,
            "int8_tops": 640,
            "supports_fp8": False,
            "recommended_block_m": 128,
            "recommended_block_n": 128,
            "recommended_block_k": 32,
        },
        "ascend910c": {
            "max_ub_size": 512 * 1024,      # 512KB
            "max_l2_size": 64 * 1024 * 1024, # 64MB
            "hbm_bandwidth": 1.6e12,         # 1.6 TB/s
            "fp16_tflops": 400,
            "int8_tops": 800,
            "supports_fp8": True,
            "recommended_block_m": 256,
            "recommended_block_n": 256,
            "recommended_block_k": 64,
        },
    }

    @classmethod
    def get_config(cls, chip):
        return cls.CONFIGS.get(chip, cls.CONFIGS["ascend910b"])

    @classmethod
    def get_optimal_block_size(cls, chip, dtype):
        config = cls.get_config(chip)
        ub_size = config["max_ub_size"]

        # 根据 UB 大小计算最优块大小
        dtype_size = {"float16": 2, "float32": 4, "int8": 1}[dtype]

        # 需要留出空间给输入和输出
        # 简化计算：A + B + C ≈ 3 * block_M * block_N * dtype_size
        max_elements = ub_size // (3 * dtype_size)
        block_size = int(max_elements ** 0.5)

        # 对齐到 16 的倍数
        block_size = (block_size // 16) * 16

        return min(block_size, config["recommended_block_m"])
```

`AscendChipConfig`类为TileLang提供了芯片感知的自动调优能力。通过静态字典`CONFIGS`维护910B和910C的关键参数差异：910C的UB容量翻倍至512KB，因此推荐的分块尺寸从128扩展到256；L2缓存从32MB增至64MB，使得更大规模的数据复用成为可能；FP8的硬件支持则为推理场景提供了两倍于FP16的吞吐潜力。`get_optimal_block_size`方法根据UB容量和数据类型的字节数动态计算最优分块：公式`ub_size // (3 * dtype_size)`中的因子3考虑了输入矩阵A、B和输出C三者同时在UB中占用的空间，开方后对齐到16的倍数以匹配Cube Core的16×16基本计算粒度。这种自适应的分块策略使得同一份TileLang代码在两代芯片上都能获得接近峰值的性能。

---

### 18.10.1 GEMM 实现

```python
import tilelang
from tilelang import T
import torch

@tilelang.jit(
    out_idx=[2],
    target="ascend",
    device="ascend910b",
)
def gemm_ascend(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: str = "float16",
):
    @T.prim_func
    def main(
        A: T.Tensor([M, K], dtype),
        B: T.Tensor([K, N], dtype),
        C: T.Tensor([M, N], "float32"),
    ):
        with T.Blocks([T.ceildiv(N, block_N), T.ceildiv(M, block_M)]) as (bn, bm):
            A_ub = T.alloc_shared([block_M, block_K], dtype, scope="UB")
            B_ub = T.alloc_shared([block_K, block_N], dtype, scope="UB")
            C_local = T.alloc_fragment([block_M, block_N], "float32", scope="UB")

            T.clear(C_local)

            for k in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[bm * block_M:, k * block_K:], A_ub)
                T.copy(B[k * block_K:, bn * block_N:], B_ub)
                T.sync()

                # Cube Core 矩阵乘法
                T.gemm(A_ub, B_ub, C_local, k_axis=block_K)

            T.copy(C_local, C[bm * block_M:, bn * block_N:])

    return main

# 使用示例
M, N, K = 4096, 4096, 4096
kernel = gemm_ascend(M, N, K)
A = torch.randn(M, K, dtype=torch.float16, device="npu:0")
B = torch.randn(K, N, dtype=torch.float16, device="npu:0")
C = kernel(A, B)
```

该GEMM实现在昇腾后端上的完整流程为：首先通过`T.Blocks`将输出矩阵沿M和N维度分块，每个block由一个AI Core独立处理；然后在K维度循环中，每次将A和B的一个分块从GM拷贝到UB（`T.copy`），调用`T.sync()`确保数据搬运完成，再用`T.gemm`在Cube Core上执行矩阵乘累加。累加器`C_local`使用FP32精度以避免多次加法操作的精度损失，输入A和B则采用FP16以减少带宽压力并提升吞吐。最终调用`T.copy`将累加结果写回全局内存，输出张量C同样使用FP32类型。使用示例展示了如同PyTorch原生算子般的调用体验——用户只需传入张量即可，TileLang自动完成编译、缓存和部署。

### 18.10.2 LayerNorm 实现

```python
@tilelang.jit(
    out_idx=[3],
    target="ascend",
    device="ascend910b",
)
def layer_norm_ascend(
    batch_size: int,
    hidden_size: int,
    eps: float = 1e-5,
    dtype: str = "float16",
):
    @T.prim_func
    def main(
        X: T.Tensor([batch_size, hidden_size], dtype),
        W: T.Tensor([hidden_size], dtype),
        B: T.Tensor([hidden_size], dtype),
        Y: T.Tensor([batch_size, hidden_size], dtype),
    ):
        with T.Blocks([batch_size]) as (b,):
            # UB 分配
            X_ub = T.alloc_fragment([hidden_size], "float32", scope="UB")
            W_ub = T.alloc_fragment([hidden_size], dtype, scope="UB")
            B_ub = T.alloc_fragment([hidden_size], dtype, scope="UB")
            mean_val = T.alloc_fragment([1], "float32", scope="UB")
            var_val = T.alloc_fragment([1], "float32", scope="UB")

            # 加载数据并转换为 FP32
            T.copy(X[b, :], X_ub)
            T.copy(W, W_ub)
            T.copy(B, B_ub)

            # Vector Core: 计算均值
            T.reduce_sum(X_ub, mean_val, axis=0)
            mean_val[0] = mean_val[0] / hidden_size

            # Vector Core: 计算方差
            for i in T.Parallel(hidden_size):
                diff = X_ub[i] - mean_val[0]
                X_ub[i] = diff * diff
            T.reduce_sum(X_ub, var_val, axis=0)
            var_val[0] = var_val[0] / hidden_size

            # Vector Core: 归一化
            rsqrt_val = T.rsqrt(var_val[0] + eps)
            for i in T.Parallel(hidden_size):
                norm_val = (X_ub[i] - mean_val[0]) * rsqrt_val
                X_ub[i] = norm_val * W_ub[i] + B_ub[i]

            T.copy(X_ub, Y[b, :])

    return main
```

LayerNorm在昇腾上的实现全部由Vector Core完成。每个block处理batch中的一行：首先将输入X、权重W和偏置B加载到UB，并将输入转换为FP32以确保归约计算的精度；接着通过`T.reduce_sum`对整行求和后除以hidden_size得到均值；然后计算每个元素与均值的偏差平方，再次归约求平均得到方差。归一化阶段使用`T.rsqrt`计算方差的平方根倒数，并在并行循环中完成中心化、缩放和仿射变换。特别注意方差计算中对原始数据的原地修改是为了避免分配额外的UB缓冲区——在256KB的UB容量限制下，这种内存复用策略尤为关键。

### 18.10.3 FlashAttention 实现

```python
@tilelang.jit(
    out_idx=[3],
    target="ascend",
    device="ascend910b",
)
def flash_attention_ascend(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    block_M: int = 128,
    block_N: int = 128,
    dtype: str = "float16",
):
    @T.prim_func
    def main(
        Q: T.Tensor([batch_size, num_heads, seq_len, head_dim], dtype),
        K: T.Tensor([batch_size, num_heads, seq_len, head_dim], dtype),
        V: T.Tensor([batch_size, num_heads, seq_len, head_dim], dtype),
        O: T.Tensor([batch_size, num_heads, seq_len, head_dim], dtype),
    ):
        scale = 1.0 / (head_dim ** 0.5)

        with T.Blocks([batch_size, num_heads, T.ceildiv(seq_len, block_M)]) as (bz, bh, bm):
            Q_ub = T.alloc_shared([block_M, head_dim], dtype, scope="UB")
            K_ub = T.alloc_shared([block_N, head_dim], dtype, scope="UB")
            V_ub = T.alloc_shared([block_N, head_dim], dtype, scope="UB")

            acc = T.alloc_fragment([block_M, head_dim], "float32", scope="UB")
            scores = T.alloc_fragment([block_M, block_N], "float32", scope="UB")
            row_max = T.alloc_fragment([block_M], "float32", scope="UB")
            row_sum = T.alloc_fragment([block_M], "float32", scope="UB")

            T.clear(acc)
            T.fill(row_max, -float("inf"))
            T.fill(row_sum, 0.0)

            T.copy(Q[bz, bh, bm * block_M:, :], Q_ub)
            T.sync()

            for bn in T.serial(T.ceildiv(seq_len, block_N)):
                T.copy(K[bz, bh, bn * block_N:, :], K_ub)
                T.copy(V[bz, bh, bn * block_N:, :], V_ub)
                T.sync()

                # Cube Core: QK^T
                T.gemm(Q_ub, K_ub, scores, transpose_B=True)

                # Vector Core: 缩放和 mask
                for i, j in T.Parallel(block_M, block_N):
                    row_idx = bm * block_M + i
                    col_idx = bn * block_N + j
                    scores[i, j] = T.if_then_else(
                        col_idx <= row_idx,
                        scores[i, j] * scale,
                        -float("inf"),
                    )

                # Vector Core: 在线 Softmax
                for i in T.serial(block_M):
                    new_max = T.max(scores[i, :], axis=0)
                    old_max = row_max[i]
                    row_max[i] = T.max(old_max, new_max)

                    exp_old = T.exp(old_max - row_max[i])
                    exp_new = T.exp(new_max - row_max[i])

                    row_sum[i] = row_sum[i] * exp_old + exp_new

                    for j in T.serial(head_dim):
                        acc[i, j] = acc[i, j] * exp_old

                for i, j in T.Parallel(block_M, block_N):
                    scores[i, j] = T.exp(scores[i, j] - row_max[i])

                # Cube Core: scores @ V
                T.gemm(scores, V_ub, acc)

            # 归一化
            for i in T.serial(block_M):
                for j in T.serial(head_dim):
                    acc[i, j] = acc[i, j] / row_sum[i]

            T.copy(acc, O[bz, bh, bm * block_M:, :])

    return main
```

这段代码服务于 18.10.3 FlashAttention 实现 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 18.11 性能调优

### 18.11.1 msprof 工具

```bash
# msprof 性能分析工具
# 基本使用
msprof --output ./profiling_data ./my_application

# 指定分析选项
msprof --output ./profiling_data \
       --trace=task-time,task-schedule \
       --memory=memory \
       ./my_application

# 分析结果查看
msprof --analyze ./profiling_data

# 导出报告
msprof --export ./profiling_data --format html
```

这段命令展示了 18.11.1 msprof 工具 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

### 18.11.2 Ascend Insight (ait) 工具

```bash
# ait 性能分析工具
# 安装
pip install ascend-profiling

# 运行分析
ait profile ./my_application --output ./ait_report

# 查看报告
ait view ./ait_report

# 分析特定算子
ait profile ./my_application --filter "gemm*" --output ./gemm_profile

# Roofline 分析
ait roofline ./ait_report
```

这段命令展示了 18.11.2 Ascend Insight (ait) 工具 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

### 18.11.3 性能调优 Checklist

```python
# TileLang 昇腾后端性能调优清单
ascend_performance_checklist = {
    "memory": {
        "ub_utilization": "确保 UB 使用率在 256KB/512KB 以内",
        "data_reuse": "最大化 UB 中的数据复用",
        "gm_access_pattern": "确保 GM 访问模式连续",
        "pipeline": "使用软件流水隐藏数据搬运延迟",
    },
    "compute": {
        "cube_utilization": "最大化 Cube Core 利用率",
        "vector_utilization": "最大化 Vector Core 利用率",
        "dual_core_overlap": "重叠 Cube 和 Vector 执行",
        "instruction_mix": "平衡计算和内存指令",
    },
    "launch": {
        "block_size": "选择合适的 block 大小",
        "grid_size": "确保足够的并行度",
        "aic_count": "充分利用 32 个 AI Core",
    },
    "architecture": {
        "chip_specific": "针对 910B/910C 优化",
        "memory_hierarchy": "合理使用 UB/L1/L2/GM",
        "data_type": "选择最优的数据类型（FP16/INT8）",
    },
}
```

这段代码服务于 18.11.3 性能调优 Checklist 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 18.12 跨硬件迁移最佳实践

### 18.12.1 从 NVIDIA 迁移到昇腾

```python
# 跨硬件迁移的代码抽象
class CrossHardwareAbstraction:
    """跨硬件抽象层"""

    @staticmethod
    def create_gemm_kernel(M, N, K, target="nvidia"):
        if target == "nvidia":
            return CrossHardwareAbstraction._create_nvidia_gemm(M, N, K)
        elif target == "ascend":
            return CrossHardwareAbstraction._create_ascend_gemm(M, N, K)
        else:
            raise ValueError(f"Unsupported target: {target}")

    @staticmethod
    def _create_nvidia_gemm(M, N, K):
        # NVIDIA 优化参数
        config = {
            "block_M": 256,
            "block_N": 256,
            "block_K": 64,
            "num_stages": 3,
            "warp_policy": "FullRow",
            "mfma_instruction": "mma.sync.aligned.m16n8k16",
        }
        return config

    @staticmethod
    def _create_ascend_gemm(M, N, K):
        # 昇腾优化参数
        config = {
            "block_M": 128,
            "block_N": 128,
            "block_K": 32,
            "num_stages": 3,
            "cube_core": True,
            "mmad_instruction": "Mmad16x16",
        }
        return config
```

这段代码服务于 18.12.1 从 NVIDIA 迁移到昇腾 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 18.12.2 迁移 Checklist

```markdown
## 从 NVIDIA GPU 迁移到昇腾 NPU 的 Checklist

### 代码层面
- [ ] 将 CUDA 核函数转换为 Ascend C
- [ ] 调整内存管理（Shared Memory → UB/L1）
- [ ] 修改线程模型（Warp → Wavefront/Cube-Vector）
- [ ] 调整数据类型（确认 FP16/BF16 支持）

### 性能层面
- [ ] 重新优化分块大小（Block Size）
- [ ] 调整软件流水参数
- [ ] 优化 Cube-Vector 双核调度
- [ ] 验证内存访问模式

### 验证层面
- [ ] 功能正确性验证
- [ ] 性能基准测试
- [ ] 内存使用验证
- [ ] 稳定性测试
```

这段代码服务于 验证层面 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 18.13 总结

<div data-component="Summary">

### ✅ 本章关键要点

1. **达芬奇架构**：独特的 Cube-Vector 分离计算模式，专为 AI 计算设计
2. **Cube Core**：矩阵运算核心，每周期 4096 次乘累加（INT8）
3. **Vector Core**：向量运算核心，处理激活函数、归一化等
4. **CANN 框架**：华为的异构计算架构，包含 TBE、Graph Engine 等组件
5. **毕昇编译器**：基于 LLVM 的 AI 编译器，提供多级优化
6. **内存层级**：UB（256KB）→ L1（512KB）→ L2（32MB）→ GM（64GB）
7. **多级 Lowering**：Python DSL → TensorIR → TileLang IR → Ascend C → Binary
8. **Pipeline 调度**：重叠数据搬运和计算，隐藏延迟
9. **双核调度**：Cube Core 和 Vector Core 并行执行
10. **昇腾 910B/910C**：分别提供 320/400 TFLOPS FP16 吞吐

### 🎯 学习目标检查

- [ ] 理解达芬奇架构的核心设计
- [ ] 掌握 Cube-Vector 双核协同工作模式
- [ ] 了解 CANN 框架的组件和功能
- [ ] 掌握 TileLang 到 Ascend C 的多级 Lowering
- [ ] 能够编写 Ascend C 算子模板
- [ ] 理解 Pipeline 调度原理
- [ ] 掌握性能调优工具的使用

</div>

---

## 18.14 Cube Core 与 Vector Core 指令详解

### 18.14.1 Cube Core 指令集详解

Cube Core 是达芬奇架构的核心矩阵计算单元，支持多种数据类型的矩阵乘累加运算。

```cpp
// ============================================================
// Cube Core 指令集详解（带逐行中文注释）
// ============================================================

#include "kernel_operator.h"
using namespace AscendC;

// Mmad 指令：矩阵乘累加（Matrix Multiply-Accumulate）
// 这是 Cube Core 最核心的指令，执行 D = A × B + C
__aicore__ void mmad_instruction_example(
    GM_ADDR A_addr,     // 输入矩阵 A 的全局内存地址
    GM_ADDR B_addr,     // 输入矩阵 B 的全局内存地址
    GM_ADDR C_addr      // 输出矩阵 C 的全局内存地址
) {
    // 第一步：初始化内存管道和缓冲区
    TPipe pipe;                              // 创建内存管道对象，管理所有缓冲区
    TQue<QuePosition::A1, 2> aQueue;        // A 矩阵输入队列，深度为 2（双缓冲）
    TQue<QuePosition::A2, 2> bQueue;        // B 矩阵输入队列，深度为 2（双缓冲）
    TQue<QuePosition::CO1, 1> cQueue;       // C 矩阵输出队列，深度为 1

    // 第二步：初始化缓冲区大小
    int block_m = 128;                       // 分块行数：128 行
    int block_n = 128;                       // 分块列数：128 列
    int block_k = 32;                        // K 维度分块：32

    pipe.InitBuffer(aQueue, 2, block_m * block_k * sizeof(half));  // A 缓冲区：128×32×2字节
    pipe.InitBuffer(bQueue, 2, block_k * block_n * sizeof(half));  // B 缓冲区：32×128×2字节
    pipe.InitBuffer(cQueue, 1, block_m * block_n * sizeof(float)); // C 缓冲区：128×128×4字节

    // 第三步：声明累加器缓冲区（在 CO2 空间，即 Cube Core 的输出缓冲）
    TBuf<TPosition::CO2> accBuf;             // 累加器缓冲区，存放中间结果
    pipe.InitBuffer(accBuf, block_m * block_n * sizeof(float));  // 初始化大小

    // 第四步：获取累加器本地张量并初始化为 0
    LocalTensor<float> acc = accBuf.Get<float>();  // 获取本地张量引用
    Duplicate(acc, 0.0f, block_m * block_n);      // 将累加器所有元素设为 0

    // 第五步：从全局内存加载数据到 UB（Unified Buffer）
    // 加载 A 矩阵块
    LocalTensor<half> aLocal = aQueue.AllocTensor<half>();  // 从队列分配一个缓冲区
    DataCopy(aLocal,                                        // 目标：UB 中的本地张量
             A_addr,                                        // 源：全局内存地址
             {block_m, block_k, 0, 0});                     // 拷贝描述：行数、列数、行间偏移、列间偏移
    aQueue.EnQue(aLocal);                                   // 将数据放入队列

    // 加载 B 矩阵块
    LocalTensor<half> bLocal = bQueue.AllocTensor<half>();  // 分配 B 的缓冲区
    DataCopy(bLocal,                                        // 目标：UB
             B_addr,                                        // 源：全局内存
             {block_k, block_n, 0, 0});                     // 拷贝描述
    bQueue.EnQue(bLocal);                                   // 放入队列

    // 第六步：从队列取出数据进行计算
    LocalTensor<half> aBuf = aQueue.DeQue<half>();          // 取出 A 的数据
    LocalTensor<half> bBuf = bQueue.DeQue<half>();          // 取出 B 的数据

    // 第七步：执行 Mmad 指令（Cube Core 矩阵乘累加）
    // 参数说明：
    //   acc     : 累加器（FP32），存放结果
    //   aBuf    : 输入 A（FP16），形状 128×32
    //   bBuf    : 输入 B（FP16），形状 32×128
    //   128     : M 维度（行数）
    //   128     : N 维度（列数）
    //   32      : K 维度（内积维度）
    Mmad(acc, aBuf, bBuf, block_m, block_n, block_k);

    // 第八步：释放输入缓冲区（归还给队列复用）
    aQueue.FreeTensor(aBuf);                                // 释放 A 缓冲区
    bQueue.FreeTensor(bBuf);                                // 释放 B 缓冲区

    // 第九步：将 FP32 结果转换为 FP16 并写回全局内存
    LocalTensor<half> cLocal = cQueue.AllocTensor<half>();  // 分配输出缓冲区
    Cast(cLocal, acc, RoundMode::CAST_NONE, block_m * block_n);  // FP32 → FP16 转换
    cQueue.EnQue(cLocal);                                   // 放入输出队列

    LocalTensor<half> cBuf = cQueue.DeQue<half>();          // 取出转换后的数据
    DataCopy(C_addr, cBuf, {block_m, block_n, 0, 0});       // 写回全局内存
    cQueue.FreeTensor(cBuf);                                // 释放输出缓冲区
}

// Add 指令：逐元素加法（Cube Core 支持的标量/向量操作）
__aicore__ void add_instruction_example() {
    TBuf<TPosition::VECCALC> buf;            // 计算缓冲区
    TPipe pipe;
    pipe.InitBuffer(buf, 128 * sizeof(float));  // 128 个 FP32 元素

    LocalTensor<float> src = buf.Get<float>();  // 获取源张量
    LocalTensor<float> dst = buf.Get<float>();  // 获取目标张量（同一缓冲区示例）

    // Adds 指令：dst[i] = src[i] + scalar
    // 参数：目标、源、标量值、元素数量
    Adds(dst, src, 1.0f, 128);               // 每个元素加 1.0
}
```

这段代码服务于 18.14.1 Cube Core 指令集详解 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 18.14.2 Vector Core 指令集详解

Vector Core 处理逐元素运算、归约、类型转换等操作，是达芬奇架构中处理非矩阵运算的核心。

```cpp
// ============================================================
// Vector Core 指令集详解（带逐行中文注释）
// ============================================================

#include "kernel_operator.h"
using namespace Ascend C;

// 向量加法：Adds 指令
__aicore__ void vector_add_example(
    GM_ADDR a,          // 输入向量 a 的全局内存地址
    GM_ADDR b,          // 输入向量 b 的全局内存地址
    GM_ADDR c,          // 输出向量 c 的全局内存地址
    int n               // 向量长度
) {
    TPipe pipe;                               // 创建内存管道

    // 初始化输入输出队列
    TQue<QuePosition::VECIN, 1> inQueue;      // 输入队列（UB 空间）
    TQue<QuePosition::VECOUT, 1> outQueue;    // 输出队列（UB 空间）

    pipe.InitBuffer(inQueue, 1, n * sizeof(float) * 2);   // 输入缓冲（两个向量）
    pipe.InitBuffer(outQueue, 1, n * sizeof(float));       // 输出缓冲

    // 加载输入向量 a
    LocalTensor<float> aLocal = inQueue.AllocTensor<float>();  // 分配缓冲区
    DataCopy(aLocal, a, {1, n, 0, 0});                        // 从 GM 加载到 UB
    inQueue.EnQue(aLocal);                                     // 放入队列

    // 加载输入向量 b
    LocalTensor<float> bLocal = inQueue.AllocTensor<float>();
    DataCopy(bLocal, b, {1, n, 0, 0});
    inQueue.EnQue(bLocal);

    // 从队列取出数据
    LocalTensor<float> aBuf = inQueue.DeQue<float>();          // 取出 a
    LocalTensor<float> bBuf = inQueue.DeQue<float>();          // 取出 b

    // 执行向量加法：c[i] = a[i] + b[i]
    LocalTensor<float> cLocal = outQueue.AllocTensor<float>(); // 分配输出缓冲
    Add(cLocal, aBuf, bBuf, n);                                // Vector Core 逐元素加法

    // 释放输入缓冲区
    inQueue.FreeTensor(aBuf);
    inQueue.FreeTensor(bBuf);

    // 写回结果
    outQueue.EnQue(cLocal);
    LocalTensor<float> cBuf = outQueue.DeQue<float>();
    DataCopy(c, cBuf, {1, n, 0, 0});                           // 从 UB 写回 GM
    outQueue.FreeTensor(cBuf);
}

// Mul 指令：逐元素乘法
__aicore__ void vector_mul_example(
    LocalTensor<float>& dst,    // 输出张量
    LocalTensor<float>& src_a,  // 输入张量 A
    LocalTensor<float>& src_b,  // 输入张量 B
    int n                       // 元素数量
) {
    // Mul 指令：dst[i] = src_a[i] * src_b[i]
    Mul(dst, src_a, src_b, n);              // Vector Core 逐元素乘法
}

// Exp 指令：指数函数
__aicore__ void vector_exp_example(
    LocalTensor<float>& dst,    // 输出张量
    LocalTensor<float>& src,    // 输入张量
    int n                       // 元素数量
) {
    // Exp 指令：dst[i] = exp(src[i])
    // 用于 Softmax 等需要指数运算的场景
    Exp(dst, src, n);                       // Vector Core 指数运算
}

// Rsqrt 指令：平方根倒数
__aicore__ void vector_rsqrt_example(
    LocalTensor<float>& dst,    // 输出张量
    LocalTensor<float>& src,    // 输入张量
    int n                       // 元素数量
) {
    // Rsqrt 指令：dst[i] = 1/sqrt(src[i])
    // 用于 LayerNorm 等需要归一化的场景
    Rsqrt(dst, src, n);                     // Vector Core 平方根倒数
}

// ReduceSum 指令：归约求和
__aicore__ void vector_reduce_sum_example(
    LocalTensor<float>& dst,    // 输出标量（1 个元素）
    LocalTensor<float>& src,    // 输入向量
    int n                       // 元素数量
) {
    // ReduceSum 指令：dst[0] = sum(src[0..n-1])
    // 用于计算均值、方差等统计量
    ReduceSum(dst, src, n);                 // Vector Core 归约求和
}

// Sigmoid 指令：Sigmoid 激活函数
__aicore__ void vector_sigmoid_example(
    LocalTensor<float>& dst,    // 输出张量
    LocalTensor<float>& src,    // 输入张量
    int n                       // 元素数量
) {
    // Sigmoid 指令：dst[i] = 1 / (1 + exp(-src[i]))
    // 常用于二分类和门控机制
    Sigmoid(dst, src, n);                   // Vector Core Sigmoid 运算
}

// Compare 指令：逐元素比较
__aicore__ void vector_compare_example(
    LocalTensor<bool>& dst,     // 输出布尔张量
    LocalTensor<float>& src_a,  // 输入张量 A
    LocalTensor<float>& src_b,  // 输入张量 B
    int n                       // 元素数量
) {
    // Compare 指令：dst[i] = (src_a[i] > src_b[i])
    // 用于 ReLU、Clamp 等条件操作
    Compare(dst, src_a, src_b, n, CMPMODE::GT);  // 大于比较
}

// Selection 指令：条件选择
__aicore__ void vector_select_example(
    LocalTensor<float>& dst,        // 输出张量
    const LocalTensor<bool>& mask,  // 布尔掩码
    LocalTensor<float>& src_a,      // 输入 A（mask 为 true 时选择）
    LocalTensor<float>& src_b,      // 输入 B（mask 为 false 时选择）
    int n                           // 元素数量
) {
    // Selection 指令：dst[i] = mask[i] ? src_a[i] : src_b[i]
    // 用于实现条件分支，如 ReLU6、GELU 等
    Select(dst, mask, src_a, src_b, n);
}
```

这段代码服务于 18.14.2 Vector Core 指令集详解 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 18.14.3 Cube-Vector 协同指令示例

```cpp
// ============================================================
// Cube-Vector 完整协同示例：GEMM + Bias + GELU
// 展示 Cube Core 和 Vector Core 如何配合完成复杂计算
// ============================================================

__aicore__ void gemm_bias_gelu_fused(
    GM_ADDR A_addr,              // 输入矩阵 A 地址
    GM_ADDR B_addr,              // 输入矩阵 B 地址
    GM_ADDR Bias_addr,           // 偏置向量地址
    GM_ADDR C_addr,              // 输出矩阵 C 地址
    int M, int N, int K          // 矩阵维度
) {
    TPipe pipe;
    int block_m = 128;           // M 维分块大小
    int block_n = 128;           // N 维分块大小
    int block_k = 32;            // K 维分块大小

    // ---- 初始化所有缓冲区 ----
    TQue<QuePosition::A1, 2> aQueue;           // A 输入队列（双缓冲）
    TQue<QuePosition::A2, 2> bQueue;           // B 输入队列（双缓冲）
    TQue<QuePosition::CO1, 1> cQueue;          // 输出队列
    TBuf<TPosition::CO2> accBuf;               // 累加器缓冲区
    TBuf<TPosition::VECCALC> biasBuf;          // 偏置缓冲区
    TBuf<TPosition::VECCALC> geluBuf;          // GELU 中间结果缓冲区

    pipe.InitBuffer(aQueue, 2, block_m * block_k * sizeof(half));
    pipe.InitBuffer(bQueue, 2, block_k * block_n * sizeof(half));
    pipe.InitBuffer(cQueue, 1, block_m * block_n * sizeof(half));
    pipe.InitBuffer(accBuf, block_m * block_n * sizeof(float));
    pipe.InitBuffer(biasBuf, block_n * sizeof(float));
    pipe.InitBuffer(geluBuf, block_m * block_n * sizeof(float));

    // ---- 初始化累加器 ----
    LocalTensor<float> acc = accBuf.Get<float>();
    Duplicate(acc, 0.0f, block_m * block_n);   // 累加器清零

    // ---- 加载偏置向量 ----
    LocalTensor<float> bias = biasBuf.Get<float>();
    DataCopy(bias, Bias_addr, {1, block_n, 0, 0});

    // ---- K 维度循环：Cube Core 执行矩阵乘法 ----
    int k_tiles = (K + block_k - 1) / block_k;  // K 维分块数量
    for (int k = 0; k < k_tiles; k++) {
        // 加载 A 块到 UB
        LocalTensor<half> aLocal = aQueue.AllocTensor<half>();
        DataCopy(aLocal, A_addr + k * block_k, {block_m, block_k, K - block_k, 0});
        aQueue.EnQue(aLocal);

        // 加载 B 块到 UB
        LocalTensor<half> bLocal = bQueue.AllocTensor<half>();
        DataCopy(bLocal, B_addr + k * block_k * N, {block_k, block_n, N - block_n, 0});
        bQueue.EnQue(bLocal);

        // 取出数据
        LocalTensor<half> aBuf = aQueue.DeQue<half>();
        LocalTensor<half> bBuf = bQueue.DeQue<half>();

        // Cube Core: 矩阵乘累加 acc += A × B
        Mmad(acc, aBuf, bBuf, block_m, block_n, block_k);

        // 释放输入缓冲
        aQueue.FreeTensor(aBuf);
        bQueue.FreeTensor(bBuf);
    }

    // ---- Vector Core: 添加偏置 ----
    // 将偏置广播加到每一行
    LocalTensor<float> geluTmp = geluBuf.Get<float>();
    for (int row = 0; row < block_m; row++) {
        // acc[row, :] += bias[:]
        Add(acc[row * block_n], acc[row * block_n], bias, block_n);
    }

    // ---- Vector Core: GELU 激活函数 ----
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // 近似实现：
    // 1. 计算 x^3
    Mul(geluTmp, acc, acc, block_m * block_n);        // geluTmp = x^2
    Mul(geluTmp, geluTmp, acc, block_m * block_n);    // geluTmp = x^3

    // 2. 计算 0.044715 * x^3
    Muls(geluTmp, geluTmp, 0.044715f, block_m * block_n);

    // 3. 计算 x + 0.044715 * x^3
    Add(geluTmp, acc, geluTmp, block_m * block_n);

    // 4. 乘以 sqrt(2/pi) ≈ 0.7978845608
    Muls(geluTmp, geluTmp, 0.7978845608f, block_m * block_n);

    // 5. 计算 tanh
    Tanh(geluTmp, geluTmp, block_m * block_n);

    // 6. 计算 1 + tanh(...)
    Adds(geluTmp, geluTmp, 1.0f, block_m * block_n);

    // 7. 计算 0.5 * x * (1 + tanh(...))
    Mul(acc, acc, geluTmp, block_m * block_n);
    Muls(acc, acc, 0.5f, block_m * block_n);

    // ---- 类型转换并写回 ----
    LocalTensor<half> cLocal = cQueue.AllocTensor<half>();
    Cast(cLocal, acc, RoundMode::CAST_NONE, block_m * block_n);
    cQueue.EnQue(cLocal);

    LocalTensor<half> cBuf = cQueue.DeQue<half>();
    DataCopy(C_addr, cBuf, {block_m, block_n, 0, N - block_n});
    cQueue.FreeTensor(cBuf);
}
```

这段代码服务于 18.14.3 Cube-Vector 协同指令示例 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 18.15 Ascend C 完整代码示例

### 18.15.1 向量加法算子（完整示例）

```cpp
// ============================================================
// Ascend C 向量加法算子完整示例（带逐行中文注释）
// 这是一个最基础的 Ascend C 算子，展示完整的开发流程
// ============================================================

#include "kernel_operator.h"    // 引入 Ascend C 内核操作头文件

using namespace AscendC;        // 使用 AscendC 命名空间

// 常量定义
constexpr int BUFFER_NUM = 2;   // 缓冲区数量（双缓冲，用于流水线）

// 内核参数结构体
struct VectorAddParams {
    GM_ADDR a;                  // 输入向量 a 的全局内存地址
    GM_ADDR b;                  // 输入向量 b 的全局内存地址
    GM_ADDR c;                  // 输出向量 c 的全局内存地址
    int32_t n;                  // 向量长度
};

// 核函数入口：每个 AI Core 执行此函数
extern "C" __global__ __aicore__
void vector_add_kernel(VectorAddParams params) {
    // 第一步：获取当前 AI Core 的索引
    int coreIdx = GetBlockIdx();                    // 当前核心编号（0-31）

    // 第二步：计算每个核心处理的数据范围
    int totalN = params.n;                          // 总数据量
    int nPerCore = (totalN + 31) / 32;              // 每个核心处理的元素数（向上取整）
    int startIdx = coreIdx * nPerCore;              // 当前核心的起始索引
    int endIdx = startIdx + nPerCore;               // 当前核心的结束索引
    if (endIdx > totalN) endIdx = totalN;           // 边界保护
    int localN = endIdx - startIdx;                 // 当前核心实际处理的元素数

    if (localN <= 0) return;                        // 如果没有数据，直接返回

    // 第三步：初始化内存管道
    TPipe pipe;                                     // 创建管道对象
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;   // 输入队列（UB 空间）
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue; // 输出队列（UB 空间）

    // 第四步：计算缓冲区大小并初始化
    int tileSize = 1024;                            // 每次处理的元素数（可调优）
    if (tileSize > localN) tileSize = localN;       // 确保不超过实际数据量

    pipe.InitBuffer(inQueue, BUFFER_NUM, tileSize * sizeof(float) * 2);  // 输入缓冲（a 和 b）
    pipe.InitBuffer(outQueue, BUFFER_NUM, tileSize * sizeof(float));     // 输出缓冲

    // 第五步：分块处理数据
    int numTiles = (localN + tileSize - 1) / tileSize;  // 总块数

    for (int t = 0; t < numTiles; t++) {
        int offset = startIdx + t * tileSize;       // 当前块的全局偏移
        int currentSize = tileSize;                  // 当前块的大小
        if (t == numTiles - 1) {                     // 最后一块可能不满
            currentSize = localN - t * tileSize;
        }

        // 5.1 加载输入向量 a
        LocalTensor<float> aLocal = inQueue.AllocTensor<float>();  // 分配缓冲区
        DataCopy(aLocal,                                          // 目标：UB
                 params.a + offset,                               // 源：全局内存偏移
                 {1, currentSize, 0, 0});                         // 拷贝描述
        inQueue.EnQue(aLocal);                                    // 放入输入队列

        // 5.2 加载输入向量 b
        LocalTensor<float> bLocal = inQueue.AllocTensor<float>();
        DataCopy(bLocal, params.b + offset, {1, currentSize, 0, 0});
        inQueue.EnQue(bLocal);

        // 5.3 从队列取出数据
        LocalTensor<float> aBuf = inQueue.DeQue<float>();         // 取出 a
        LocalTensor<float> bBuf = inQueue.DeQue<float>();         // 取出 b

        // 5.4 执行向量加法（Vector Core）
        LocalTensor<float> cLocal = outQueue.AllocTensor<float>();  // 分配输出缓冲
        Add(cLocal, aBuf, bBuf, currentSize);                       // c[i] = a[i] + b[i]

        // 5.5 释放输入缓冲区
        inQueue.FreeTensor(aBuf);                                   // 归还 a 缓冲区
        inQueue.FreeTensor(bBuf);                                   // 归还 b 缓冲区

        // 5.6 写回结果
        outQueue.EnQue(cLocal);                                     // 放入输出队列
        LocalTensor<float> cBuf = outQueue.DeQue<float>();          // 取出结果
        DataCopy(params.c + offset, cBuf, {1, currentSize, 0, 0}); // 写回全局内存
        outQueue.FreeTensor(cBuf);                                  // 归还缓冲区
    }
}
```

这段代码服务于 18.15.1 向量加法算子（完整示例） 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 18.15.2 Softmax 算子（Vector Core 重点示例）

```cpp
// ============================================================
// Ascend C Softmax 算子（带逐行中文注释）
// Softmax(x_i) = exp(x_i) / sum(exp(x_j))
// 全部由 Vector Core 执行，无需 Cube Core
// ============================================================

#include "kernel_operator.h"
using namespace AscendC;

// Softmax 内核参数
struct SoftmaxParams {
    GM_ADDR input;              // 输入张量地址
    GM_ADDR output;             // 输出张量地址
    int32_t rows;               // 行数（batch 维度）
    int32_t cols;               // 列数（特征维度）
};

extern "C" __global__ __aicore__
void softmax_kernel(SoftmaxParams params) {
    int coreIdx = GetBlockIdx();                    // 当前核心编号
    int rowsPerCore = (params.rows + 31) / 32;     // 每核心处理的行数
    int startRow = coreIdx * rowsPerCore;
    int endRow = startRow + rowsPerCore;
    if (endRow > params.rows) endRow = params.rows;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> calcBuf;

    int cols = params.cols;                         // 特征维度大小
    pipe.InitBuffer(inQueue, 1, cols * sizeof(float));
    pipe.InitBuffer(outQueue, 1, cols * sizeof(float));
    pipe.InitBuffer(calcBuf, cols * sizeof(float) * 2);  // 中间结果缓冲

    LocalTensor<float> expBuf = calcBuf.Get<float>();      // exp 结果缓冲
    LocalTensor<float> maxBuf = calcBuf.Get<float>()[cols]; // 最大值缓冲

    for (int row = startRow; row < endRow; row++) {
        // 1. 加载当前行数据
        LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
        DataCopy(inLocal, params.input + row * cols, {1, cols, 0, 0});
        inQueue.EnQue(inLocal);
        LocalTensor<float> inBuf = inQueue.DeQue<float>();

        // 2. 计算最大值（数值稳定性优化）
        // max_val = max(input[row, :])
        ReduceMax(maxBuf, inBuf, cols);                 // Vector Core 归约求最大值

        // 3. 减去最大值（防止 exp 溢出）
        // x_shifted[i] = input[row, i] - max_val
        Sub(expBuf, inBuf, maxBuf, cols);               // Vector Core 逐元素减法

        // 4. 计算指数
        // exp_buf[i] = exp(x_shifted[i])
        Exp(expBuf, expBuf, cols);                      // Vector Core 指数运算

        // 5. 计算指数和
        // sum_exp = sum(exp_buf)
        ReduceSum(maxBuf, expBuf, cols);                // Vector Core 归约求和

        // 6. 除以指数和（归一化）
        // output[row, i] = exp_buf[i] / sum_exp
        Div(expBuf, expBuf, maxBuf, cols);              // Vector Core 逐元素除法

        // 7. 写回结果
        LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        Adds(outLocal, expBuf, 0.0f, cols);             // 拷贝到输出缓冲
        outQueue.EnQue(outLocal);
        LocalTensor<float> outBuf = outQueue.DeQue<float>();
        DataCopy(params.output + row * cols, outBuf, {1, cols, 0, 0});
        outQueue.FreeTensor(outBuf);

        // 释放输入缓冲
        inQueue.FreeTensor(inBuf);
    }
}
```

这段代码服务于 18.15.2 Softmax 算子（Vector Core 重点示例） 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 18.16 msprof 与 ait 性能分析详细指南

### 18.16.1 msprof 完整使用指南

```bash
# ============================================================
# msprof 基础用法
# ============================================================

# 1. 基本性能分析
#    收集算子执行时间、调度信息等基础数据
msprof --output ./profiling_data ./my_application

# 2. 收集任务时间线
#    记录每个算子的开始/结束时间，生成可视化时间线
msprof --output ./profiling_data \
       --trace=task-time,task-schedule \
       ./my_application

# 3. 收集内存使用信息
#    记录内存分配/释放事件和峰值内存使用
msprof --output ./profiling_data \
       --memory=memory \
       ./my_application

# 4. 收集 AI Core 性能计数器
#    记录 Cube Core 和 Vector Core 的利用率
msprof --output ./profiling_data \
       --aicpu=aicpu \
       --ai-core=ai-core \
       ./my_application

# 5. 综合分析（推荐）
msprof --output ./profiling_data \
       --trace=task-time,task-schedule \
       --memory=memory \
       --aicpu=aicpu \
       --ai-core=ai-core \
       ./my_application

# ============================================================
# msprof 结果查看与导出
# ============================================================

# 6. 查看分析结果摘要
msprof --analyze ./profiling_data

# 7. 导出 HTML 报告（可视化）
msprof --export ./profiling_data --format html

# 8. 导出 CSV 报告（便于脚本处理）
msprof --export ./profiling_data --format csv

# 9. 导出 JSON 报告
msprof --export ./profiling_data --format json
```

这段命令展示了 18.16.1 msprof 完整使用指南 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

### 18.16.2 msprof 输出解读

```python
# msprof 输出分析脚本
import json

def analyze_msprof_output(json_path):
    """
    分析 msprof 的 JSON 输出，提取关键性能指标
    """
    with open(json_path) as f:
        data = json.load(f)

    print("=" * 60)
    print("昇腾 NPU 性能分析报告")
    print("=" * 60)

    # 提取算子执行信息
    tasks = data.get("task_info", [])
    total_time = 0

    for task in tasks:
        name = task.get("name", "unknown")
        duration = task.get("duration_us", 0)
        core_type = task.get("core_type", "unknown")  # cube / vector / scalar
        total_time += duration

        print(f"\n算子: {name}")
        print(f"  核心类型: {core_type}")
        print(f"  执行时间: {duration:.2f} us")

    print(f"\n总执行时间: {total_time:.2f} us")

    # 提取内存信息
    mem_info = data.get("memory_info", {})
    if mem_info:
        peak = mem_info.get("peak_usage_mb", 0)
        print(f"峰值内存使用: {peak:.2f} MB")

    # 提取 AI Core 利用率
    core_util = data.get("ai_core_utilization", {})
    if core_util:
        cube_util = core_util.get("cube_utilization", 0)
        vector_util = core_core.get("vector_utilization", 0)
        print(f"Cube Core 利用率: {cube_util*100:.1f}%")
        print(f"Vector Core 利用率: {vector_util*100:.1f}%")
```

这段代码服务于 18.16.2 msprof 输出解读 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 18.16.3 ait（Ascend Insight）详细使用指南

```bash
# ============================================================
# ait 安装与基础用法
# ============================================================

# 安装 ait 工具
pip install ascend-profiling

# 1. 基本性能分析
#    收集完整的性能数据
ait profile ./my_application --output ./ait_report

# 2. 指定分析精度
#    "low": 快速分析，开销小
#    "medium": 中等精度
#    "high": 高精度，开销大
ait profile ./my_application --output ./ait_report --level high

# 3. 分析特定算子（按名称过滤）
ait profile ./my_application --filter "gemm*" --output ./gemm_profile
ait profile ./my_application --filter "*matmul*" --output ./matmul_profile

# 4. 查看分析报告
ait view ./ait_report

# 5. 生成 Roofline 图
#    可视化算子的计算密度 vs 内存带宽
ait roofline ./ait_report

# 6. 生成时间线视图
ait timeline ./ait_report

# ============================================================
# ait 高级分析
# ============================================================

# 7. 内存访问分析
ait memory ./ait_report

# 8. AI Core 利用率分析
ait utilization ./ait_report

# 9. 算子耗时排名
ait top-k ./ait_report --k 10

# 10. 导出报告
ait export ./ait_report --format html --output ./report.html
```

这段命令展示了 18.16.3 ait（Ascend Insight）详细使用指南 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

### 18.16.4 ait 输出解读与优化建议

```python
# ait 分析结果解读指南
ait_metrics_guide = {
    # Cube Core 指标
    "cube_utilization": {
        "含义": "Cube Core 矩阵运算单元的利用率",
        "目标": "> 70%",
        "优化建议": [
            "增大分块尺寸以提高 Cube Core 使用率",
            "减少 Vector Core 计算占比",
            "确保数据已对齐到 16 字节边界",
        ],
    },

    # Vector Core 指标
    "vector_utilization": {
        "含义": "Vector Core 向量运算单元的利用率",
        "目标": "> 60%",
        "优化建议": [
            "使用向量化指令替代标量循环",
            "减少类型转换操作",
            "合并多个逐元素操作",
        ],
    },

    # 内存指标
    "ub_utilization": {
        "含义": "UB（Unified Buffer）使用率",
        "目标": "> 80%",
        "优化建议": [
            "增大分块尺寸以充分利用 UB",
            "使用双缓冲隐藏数据搬运延迟",
            "减少 UB 中的临时缓冲区数量",
        ],
    },

    # 数据搬运指标
    "dma_utilization": {
        "含义": "DMA 引擎利用率（GM ↔ UB 数据搬运）",
        "目标": "与计算重叠",
        "优化建议": [
            "使用软件流水线重叠搬运和计算",
            "确保数据搬运粒度足够大",
            "使用预取指令提前加载数据",
        ],
    },
}
```

这段代码服务于 18.16.4 ait 输出解读与优化建议 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 18.17 昇腾 910B vs 910C 硬件对比

### 18.17.1 详细硬件规格对比

| 参数 | 昇腾 910B | 昇腾 910C | 变化 | 说明 |
|------|-----------|-----------|------|------|
| AI Core 数量 | 32 | 32 | 不变 | 每芯片 32 个 AIC |
| FP16 矩阵吞吐 | 320 TFLOPS | 400 TFLOPS | +25% | 主要性能提升 |
| FP32 向量吞吐 | 80 TFLOPS | 100 TFLOPS | +25% | 向量计算提升 |
| INT8 吞吐 | 640 TOPS | 800 TOPS | +25% | 整数运算提升 |
| FP8 支持 | 不支持 | 支持 | 新增 | 新数据类型 |
| BF16 支持 | 支持 | 支持 | — | 脑浮点格式 |
| HBM 容量 | 64GB | 96GB | +50% | 更大显存 |
| HBM 类型 | HBM2e | HBM2e | — | 相同类型 |
| HBM 带宽 | 1.2 TB/s | 1.6 TB/s | +33% | 更高带宽 |
| HBM 通道数 | 8 | 8 | — | 相同 |
| L2 缓存 | 32MB/AIC | 64MB/AIC | 2× | 缓存翻倍 |
| UB 容量 | 256KB/AIC | 512KB/AIC | 2× | UB 翻倍 |
| L1 容量 | 512KB/AIC | 512KB/AIC | — | 不变 |
| Cube Core 数量 | 1/AIC | 1/AIC | — | 不变 |
| Vector Core 数量 | 1/AIC | 1/AIC | — | 不变 |
| TDP | 400W | 450W | +12.5% | 功耗略增 |
| 互联带宽 | HCCS 3.0 | HCCS 3.0 | — | 相同 |
| 制程 | 7nm | 7nm+ | 改进 | 工艺优化 |

### 18.17.2 性能对比

| 基准测试 | 910B | 910C | 加速比 | 说明 |
|----------|------|------|--------|------|
| GEMM 4096×4096 (FP16) | 2.35 ms | 1.88 ms | 1.25× | 理论值匹配 |
| GEMM 8192×8192 (FP16) | 18.5 ms | 14.8 ms | 1.25× | 大矩阵 |
| GEMM 4096×4096 (INT8) | 1.18 ms | 0.94 ms | 1.25× | INT8 运算 |
| FlashAttention (Seq4096) | 4.12 ms | 3.30 ms | 1.25× | 注意力机制 |
| LayerNorm (Hidden4096) | 0.085 ms | 0.068 ms | 1.25× | 归一化 |
| Conv2d ResNet50 Layer1 | 0.45 ms | 0.36 ms | 1.25× | 卷积运算 |

### 18.17.3 针对不同芯片的优化策略

```python
def get_optimization_strategy(chip_type):
    """
    根据芯片类型返回优化策略

    910C 的主要优势：
    1. UB 翻倍 → 可以使用更大的分块
    2. L2 翻倍 → 更好的数据复用
    3. FP8 支持 → 更高吞吐（适合推理）
    4. 带宽增加 → 减少 IO 瓶颈
    """
    strategies = {
        "ascend910b": {
            "block_M": 128,
            "block_N": 128,
            "block_K": 32,
            "num_stages": 2,           # UB 有限，使用 2 级流水
            "use_fp8": False,          # 不支持 FP8
            "pipeline_depth": 2,
            "max_ub_tiles": 4,         # UB 较小，分块数有限
        },
        "ascend910c": {
            "block_M": 256,            # UB 翻倍，可用更大分块
            "block_N": 256,
            "block_K": 64,             # K 维也可以更大
            "num_stages": 3,           # UB 更大，可用 3 级流水
            "use_fp8": True,           # 支持 FP8
            "pipeline_depth": 3,
            "max_ub_tiles": 8,         # UB 翻倍，分块数翻倍
        },
    }
    return strategies.get(chip_type, strategies["ascend910b"])
```

这段代码服务于 18.17.3 针对不同芯片的优化策略 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 18.18 CUDA → Ascend 迁移 API 映射表

### 18.18.1 运行时 API 映射

| CUDA API | AscendCL API | 功能 |
|----------|-------------|------|
| `cudaMalloc` | `aclrtMalloc` | 设备内存分配 |
| `cudaFree` | `aclrtFree` | 设备内存释放 |
| `cudaMemcpy` | `aclrtMemcpy` | 内存拷贝 |
| `cudaMemcpyAsync` | `aclrtMemcpyAsync` | 异步内存拷贝 |
| `cudaDeviceSynchronize` | `aclrtSynchronizeDevice` | 设备同步 |
| `cudaStreamSynchronize` | `aclrtSynchronizeStream` | 流同步 |
| `cudaStreamCreate` | `aclrtCreateStream` | 创建流 |
| `cudaStreamDestroy` | `aclrtDestroyStream` | 销毁流 |
| `cudaEventCreate` | `aclrtCreateEvent` | 创建事件 |
| `cudaEventRecord` | `aclrtRecordEvent` | 记录事件 |
| `cudaEventSynchronize` | `aclrtSynchronizeEvent` | 事件同步 |
| `cudaGetDevice` | `aclrtGetDevice` | 获取当前设备 |
| `cudaSetDevice` | `aclrtSetDevice` | 设置当前设备 |
| `cudaGetDeviceProperties` | `aclrtGetDeviceProperties` | 获取设备属性 |
| `cudaGetLastError` | `aclrtGetLastError` | 获取错误码 |

### 18.18.2 编程模型概念映射

| CUDA 概念 | Ascend C 概念 | 说明 |
|-----------|-------------|------|
| Kernel (`__global__`) | `__aicore__` | 核函数声明 |
| Shared Memory | UB (Unified Buffer) | 片上共享存储 |
| Warp (32 线程) | AI Core | 计算单元 |
| Thread Block | Block | 线程块/核心块 |
| Grid | 所有 AI Core | 网格/全芯片 |
| `__shared__` | `TBuf<TPosition::VECCALC>` | 共享内存声明 |
| `__syncthreads()` | 隐式（队列同步） | 块内同步 |
| Tensor Core | Cube Core | 矩阵运算核心 |
| CUDA Core | Vector Core | 通用计算核心 |
| `cudaStream_t` | `aclrtStream` | 异步流 |
| Registers | 寄存器 | 线程私有存储 |
| L1 Cache | L1 Buffer | 一级缓存 |
| L2 Cache | L2 Cache | 二级缓存 |
| Global Memory | GM (Global Memory) | 全局内存/HBM |

### 18.18.3 算子开发模式映射

| CUDA 模式 | Ascend C 模式 | 说明 |
|-----------|-------------|------|
| `gemm + bias + relu` 分开写 | Cube GEMM + Vector Bias + Vector ReLU | 融合算子 |
| Shared Memory tiling | UB tiling | 数据分块策略 |
| Register blocking | 累加器寄存器 | 寄存器分块 |
| Double buffering | `TQue` 双缓冲 | 流水线隐藏延迟 |
| `__shfl_sync` | 隐式（Cube Core 内部） | 数据交换 |
| Warp-level reduction | Vector Core `ReduceSum` | 归约操作 |
| Cooperative Groups | 多 AIC 协同 | 跨核心协作 |

---

## 18.19 性能基准测试数据

### 18.19.1 GEMM 性能对比（Ascend vs CUDA）

| 矩阵尺寸 (M×N×K) | 数据类型 | 910B (ms) | 910C (ms) | A100 (ms) | 910B/A100 | 910C/A100 |
|-------------------|----------|-----------|-----------|-----------|-----------|-----------|
| 1024×1024×1024 | FP16 | 0.125 | 0.100 | 0.198 | 1.58× | 1.98× |
| 2048×2048×2048 | FP16 | 0.856 | 0.685 | 1.024 | 1.20× | 1.49× |
| 4096×4096×4096 | FP16 | 2.350 | 1.880 | 7.562 | 3.22× | 4.02× |
| 8192×8192×8192 | FP16 | 18.50 | 14.80 | 59.87 | 3.24× | 4.04× |
| 4096×4096×4096 | INT8 | 1.180 | 0.940 | 3.654 | 3.10× | 3.89× |

> [!NOTE]
> 以上数据为参考值，实际性能取决于具体实现、CANN/CUDA 版本、系统配置等因素。

### 18.19.2 归一化算子性能对比

| 算子 | 配置 | 910B (ms) | 910C (ms) | A100 (ms) | 说明 |
|------|------|-----------|-----------|-----------|------|
| LayerNorm | Hidden=4096, Batch=32 | 0.085 | 0.068 | 0.112 | Vector Core 优势 |
| LayerNorm | Hidden=8192, Batch=32 | 0.168 | 0.134 | 0.224 | 大隐藏维度 |
| RMSNorm | Hidden=4096, Batch=32 | 0.062 | 0.050 | 0.085 | 更简洁的归一化 |
| BatchNorm | C=256, H=W=56 | 0.034 | 0.027 | 0.045 | 卷积后归一化 |
| GroupNorm | C=256, Groups=32 | 0.045 | 0.036 | 0.058 | 图像生成常用 |

### 18.19.3 Attention 性能对比

| 配置 (Batch×Heads×Seq×Dim) | 910B (ms) | 910C (ms) | A100 (ms) | 说明 |
|-----------------------------|-----------|-----------|-----------|------|
| 1×32×2048×128 | 0.685 | 0.548 | 1.205 | 短序列 |
| 1×32×4096×128 | 2.580 | 2.064 | 5.023 | 中等序列 |
| 1×32×8192×128 | 10.25 | 8.20 | 20.145 | 长序列 |
| 4×32×2048×128 | 2.740 | 2.192 | 4.821 | batch=4 |
| 4×32×4096×128 | 10.32 | 8.26 | 20.092 | 大 batch |
| 1×32×4096×256 | 5.12 | 4.10 | 10.067 | 大 head_dim |

### 18.19.4 性能分析与瓶颈诊断

```python
def diagnose_ascend_performance(kernel_name, exec_time_ms, config):
    """
    诊断昇腾算子性能瓶颈

    Args:
        kernel_name: 算子名称
        exec_time_ms: 执行时间（毫秒）
        config: 算子配置（字典）
    """
    chip = config.get("chip", "ascend910b")
    data_type = config.get("dtype", "float16")

    # 获取理论峰值
    peak_tflops = {
        "ascend910b": {"float16": 320, "int8": 640},
        "ascend910c": {"float16": 400, "int8": 800},
    }[chip][data_type]

    # 计算实际性能
    M, N, K = config["M"], config["N"], config["K"]
    flops = 2 * M * N * K  # 矩阵乘法浮点运算数
    actual_tflops = flops / (exec_time_ms * 1e-3) / 1e12

    # 计算效率
    efficiency = actual_tflops / peak_tflops

    print(f"算子: {kernel_name}")
    print(f"芯片: {chip}")
    print(f"数据类型: {data_type}")
    print(f"矩阵尺寸: {M}×{N}×{K}")
    print(f"执行时间: {exec_time_ms:.3f} ms")
    print(f"实际性能: {actual_tflops:.1f} TFLOPS")
    print(f"理论峰值: {peak_tflops} TFLOPS")
    print(f"效率: {efficiency*100:.1f}%")

    # 诊断建议
    if efficiency < 0.4:
        print("\n⚠ 效率偏低，建议检查：")
        print("  1. 分块大小是否合适（建议 128×128 或 256×256）")
        print("  2. 是否有大量 GM 访问（应使用 UB 缓存）")
        print("  3. Cube Core 利用率是否低（检查数据对齐）")
        print("  4. 是否有频繁的类型转换（FP32↔FP16）")
    elif efficiency < 0.6:
        print("\n✓ 效率中等，可优化方向：")
        print("  1. 增加软件流水深度")
        print("  2. 优化 Cube-Vector 双核调度")
        print("  3. 减少不必要的数据搬运")
    else:
        print("\n✓ 效率良好")
```

这段代码服务于 18.19.4 性能分析与瓶颈诊断 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 18.20 补充练习

### 练习 4：Softmax 算子优化

```cpp
// 任务：优化 Softmax 算子的 Vector Core 使用效率
// 要求：
// 1. 实现数值稳定的 Softmax（减去最大值）
// 2. 使用向量化指令提高计算效率
// 3. 处理任意长度的输入（不是 128 的倍数）
// 4. 测试不同输入尺寸的性能

__aicore__ void optimized_softmax(
    GM_ADDR input, GM_ADDR output, int rows, int cols
) {
    // TODO: 实现优化的 Softmax
    // 提示：
    // 1. 使用 ReduceMax 求最大值
    // 2. 使用 Sub 减去最大值
    // 3. 使用 Exp 计算指数
    // 4. 使用 ReduceSum 求和
    // 5. 使用 Div 归一化
}
```

这段代码服务于 练习 4：Softmax 算子优化 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 练习 5：LayerNorm + 线性变换融合

```python
# 任务：实现 LayerNorm + Linear 融合算子
# 要求：
# 1. LayerNorm 部分使用 Vector Core
# 2. Linear 部分使用 Cube Core
# 3. 优化 Cube-Vector 双核调度
# 4. 对比融合前后的性能

@tilelang.jit(target="ascend")
def fused_layernorm_linear(M, N, K):
    """
    融合 LayerNorm 和 Linear 的算子

    输入：X (M, N), W_norm (N), B_norm (N), W_linear (N, K), B_linear (K)
    输出：Y (M, K)

    流程：
    1. LayerNorm(X) → normed (M, N)  [Vector Core]
    2. normed @ W_linear → Y (M, K)  [Cube Core]
    3. Y + B_linear → Y (M, K)       [Vector Core]
    """
    # TODO: 实现融合算子
    pass
```

这段代码服务于 练习 5：LayerNorm + 线性变换融合 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 练习 6：多 AIC 协同 GEMM

```cpp
// 任务：实现跨 AI Core 协同的大型 GEMM
// 要求：
// 1. 将大矩阵分发到多个 AI Core
// 2. 每个 AI Core 计算子块
// 3. 使用 HCCL 进行结果收集
// 4. 测试 4/8/16/32 个 AI Core 的扩展性

void multi_aic_gemm(
    GM_ADDR A, GM_ADDR B, GM_ADDR C,
    int M, int N, int K, int num_cores
) {
    // TODO: 实现多核协同 GEMM
    // 提示：
    // 1. 沿 M 或 N 维度分块
    // 2. 每个 AI Core 处理一个子块
    // 3. 计算完成后，结果已在全局内存中
}
```

这段代码服务于 练习 6：多 AIC 协同 GEMM 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 练习 7：性能分析实战

```bash
# 任务：使用 msprof 和 ait 分析 TileLang 算子性能
# 步骤：
# 1. 编写一个 TileLang GEMM 算子
# 2. 使用 msprof 收集性能数据
# 3. 分析 Cube Core 和 Vector Core 利用率
# 4. 使用 ait 生成 Roofline 图
# 5. 根据分析结果优化算子
# 6. 再次分析，对比优化效果

# 提示命令：
# msprof --output ./prof_data --trace=task-time --ai-core=ai-core ./app
# ait profile ./app --output ./ait_report
# ait roofline ./ait_report
```

这段命令展示了 练习 7：性能分析实战 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

### 练习 8：CUDA → Ascend 迁移实战

```python
# 任务：将一个完整的 CUDA 算子迁移到昇腾 NPU
# 原始 CUDA 代码：GEMM + Bias + ReLU + Dropout

# 原始 CUDA 版本
@tilelang.jit(target="cuda")
def gemm_bias_relu_dropout_cuda(M, N, K, dropout_p=0.1):
    # ... CUDA 实现
    pass

# 迁移到昇腾版本
@tilelang.jit(target="ascend")
def gemm_bias_relu_dropout_ascend(M, N, K, dropout_p=0.1):
    """
    要求：
    1. GEMM 使用 Cube Core
    2. Bias + ReLU 使用 Vector Core
    3. Dropout 使用 Vector Core（随机数生成 + 掩码）
    4. 优化 Cube-Vector 双核调度
    5. 确保功能正确性
    """
    # TODO: 实现迁移后的代码
    pass
```

这段代码服务于 练习 8：CUDA → Ascend 迁移实战 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 练习

### 练习 1：Ascend C 算子开发

编写一个简单的 Ascend C 算子，实现向量加法。

```cpp
// 任务：实现以下函数
__aicore__ void vector_add(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, int n
) {
    // 要求：
    // 1. 使用 UB 缓冲区
    // 2. 使用 Vector Core 进行计算
    // 3. 处理边界情况（n 不是 128 的倍数）
    // TODO: 实现您的代码
}
```

这段代码服务于 练习 1：Ascend C 算子开发 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 练习 2：Cube-Vector 双核调度

设计一个 GEMM + Bias + GELU 的算子，优化 Cube-Vector 双核调度。

```python
# 任务：实现以下函数
@tilelang.jit(target="ascend")
def gemm_bias_gel_ascend(M, N, K):
    # 要求：
    # 1. GEMM 使用 Cube Core
    # 2. Bias + GELU 使用 Vector Core
    # 3. 尽量重叠 Cube 和 Vector 执行
    # TODO: 实现您的代码
    pass
```

这段代码服务于 练习 2：Cube-Vector 双核调度 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 练习 3：跨硬件迁移

将一个 NVIDIA GPU 的 GEMM 实现迁移到昇腾 NPU。

```python
# 任务：将以下 NVIDIA 实现迁移到昇腾
@tilelang.jit(target="cuda")
def gemm_nvidia(M, N, K):
    # ... NVIDIA 实现
    pass

# 迁移到昇腾
@tilelang.jit(target="ascend")
def gemm_ascend(M, N, K):
    # TODO: 实现迁移后的代码
    # 要求：
    # 1. 调整分块大小
    # 2. 修改内存管理
    # 3. 优化 Cube-Vector 调度
    pass
```

这段代码服务于 练习 3：跨硬件迁移 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 思考题

1. **Cube-Vector 协同**：为什么达芬奇架构采用 Cube-Vector 分离设计？与 GPU 的统一计算单元相比有什么优劣？

2. **内存层级设计**：昇腾 NPU 的 UB/L1/L2/GM 内存层级与 GPU 的 Shared Memory/L1/L2/Global Memory 有什么异同？

3. **多级 Lowering**：为什么 TileLang 需要多级 Lowering 而不是直接从 Python 生成 Ascend C？每一级 Lowering 的价值是什么？

4. **双核调度**：在什么场景下 Cube-Vector 双核调度能带来最大的性能提升？如何判断是否需要双核调度？

5. **跨硬件兼容**：如何设计一个统一的编程接口，使得同一份代码可以在 NVIDIA GPU 和昇腾 NPU 上高效运行？

---

## 扩展阅读

1. **华为昇腾文档**：https://www.hiascend.com/document
2. **Ascend C 编程指南**：https://www.hiascend.com/document/detail/zh/CannCommunityEdition
3. **CANN 框架文档**：https://www.hiascend.com/software/cann
4. **毕昇编译器**：https://www.hiascend.com/software/bisheng
5. **MindSpore 框架**：https://www.mindspore.cn/

---

## 下一章预告

> **Chapter 19: JIT 动态编译与运行时机制**
>
> 在下一章中，我们将深入探讨 TileLang 的 JIT 编译机制。您将学习：
> - JIT 编译的基本原理与优势
> - TileLang 的完整 JIT 编译流程
> - 编译缓存机制与优化
> - PyTorch 集成与 torch.compile
> - 动态 Shape 支持与多目标编译
