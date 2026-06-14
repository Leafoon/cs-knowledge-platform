---
title: "Chapter 14: 内存模型与共享内存管理"
description: "深入理解 GPU 内存层次结构、Triton 的内存抽象机制、内存合并访问、共享内存管理、Bank Conflict 规避、异步拷贝流水线与内存优化策略，掌握高性能 kernel 编写的内存优化核心技术"
date: "2026-06-11"
---

# Chapter 14: 内存模型与共享内存管理

> **学习目标**：
> - 理解 GPU 内存层次结构（HBM → L2 → Shared Memory → Register）的延迟、带宽与容量特征
> - 掌握 Triton 的内存抽象模型：tt.load 如何自动管理 shared memory 分配与释放
> - 理解内存合并访问（Memory Coalescing）原理，分析 access pattern 对带宽利用率的影响
> - 掌握 Triton GPU 方言中 local_alloc / local_load / local_store 操作的 IR 表示与编译器行为
> - 理解共享内存 Bank Conflict 的成因、性能影响及 Triton 编译器的规避策略
> - 掌握异步拷贝（cp.async）在 Triton 中的映射及 num_stages 流水线优化

---

## 14.1 GPU 内存层次结构回顾

### 14.1.1 内存层次总览

GPU 的内存系统是一个多层次的存储层次，每一层在容量、带宽和延迟之间做出不同的权衡。理解这个层次结构是编写高性能 kernel 的基础。

```
GPU 内存层次结构（以 NVIDIA A100 SXM 为例）

┌─────────────────────────────────────────────────────────────────────────┐
│                              GPU 芯片                                    │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        SM 0                                       │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │  Register File                                             │  │  │
│  │  │  容量: 65,536 × 32-bit = 256 KB                            │  │  │
│  │  │  带宽: ~80 TB/s (per SM)                                   │  │  │
│  │  │  延迟: 1 cycle                                             │  │  │
│  │  │  → 每个线程最多 255 个 32-bit 寄存器                        │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │  Shared Memory / L1 Cache (统一)                           │  │  │
│  │  │  容量: 最大 164 KB (可配置 SMEM/L1 比例)                   │  │  │
│  │  │  带宽: ~19 TB/s (per SM, 32 bank × 4B × 32 路交织)        │  │  │
│  │  │  延迟: ~20-30 cycles                                       │  │  │
│  │  │  → 所有线程块内线程共享；可配置 partition                   │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        SM 1                                       │  │
│  │                        ...                                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                        (共 108 个 SM)                                   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  L2 Cache                                                        │  │
│  │  容量: 40 MB (所有 SM 共享)                                      │  │
│  │  带宽: ~5 TB/s                                                   │  │
│  │  延迟: ~200 cycles                                               │  │
│  │  → 全局内存的缓存层；可被显式管理（eviction policy）              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ 内存总线 (HBM2e)
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  HBM (High Bandwidth Memory) — 全局内存                                │
│  容量: 80 GB                                                           │
│  带宽: 2.0 TB/s (A100 SXM) / 1.5 TB/s (A100 PCIe)                    │
│  延迟: ~400-600 cycles                                                 │
│  → 所有 SM 可访问；kernel 间数据传递的主要介质                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 14.1.2 各层次关键参数对比

| 存储层次 | 容量 | 带宽 | 延迟 | 可见性 | 管理方式 |
|---|---|---|---|---|---|
| **Register File** | 256 KB/SM | ~80 TB/s | 1 cycle | 线程私有 | 编译器自动分配 |
| **Shared Memory** | 164 KB/SM | ~19 TB/s | 20-30 cycles | 线程块共享 | 程序员/Triton 编译器 |
| **L1 Cache** | 128 KB/SM | ~19 TB/s | 30-40 cycles | 线程块（透明） | 硬件自动管理 |
| **L2 Cache** | 40 MB | ~5 TB/s | 200 cycles | 全局（透明） | 硬件 + eviction hint |
| **HBM (Global)** | 80 GB | 2.0 TB/s | 400-600 cycles | 全局 | 程序员显式读写 |

```
带宽对比（对数刻度，近似值）:

  Register    ████████████████████████████████████████████████  ~80 TB/s
  Shared Mem  ██████████████████████████                        ~19 TB/s
  L2 Cache    ██████████                                        ~5 TB/s
  HBM         ████                                              ~2 TB/s

延迟对比（线性刻度）:

  Register    █                                                 1 cycle
  Shared Mem  █████                                             20 cycles
  L1 Cache    ████████                                          30 cycles
  L2 Cache    ████████████████████████████████████               200 cycles
  HBM         ██████████████████████████████████████████████████ 400 cycles
```

### 14.1.3 为什么内存层次如此重要

深度学习 kernel 的性能通常受限于内存带宽而非计算能力。以 A100 为例：

```
Roofline 模型分析:

  性能 (TFLOPS/s)
    ^
312 |           ____________________________________  FP16 Tensor Core
    |          /
    |         /
    |        /
    |       /
    |      /
    |     /
    |    /
    |   /
    |  /
    | /
    |/________________________________________________> 计算强度 (FLOPs/Byte)
    0        156        312        468        624

  Ridge Point = Peak FLOPS / Memory Bandwidth
              = 312 TFLOPS / 2 TB/s
              = 156 FLOPs/Byte

  结论:
    计算强度 < 156 → Memory-bound（受限于内存带宽）
    计算强度 > 156 → Compute-bound（受限于计算能力）

  典型 kernel 的计算强度:
    Element-wise (加法):    ~0.5 FLOPs/Byte  → 严重 memory-bound
    LayerNorm:              ~2-5 FLOPs/Byte  → memory-bound
    GEMM (K=1024):          ~341 FLOPs/Byte  → compute-bound
    FlashAttention:         取决于块大小      → 接近 compute-bound
```

**核心洞察**：对于 memory-bound 的 kernel，优化内存访问模式是提升性能的唯一途径。对于 compute-bound 的 kernel，减少内存访问可以释放计算单元的等待时间，同样能提升整体吞吐。

---

## 14.2 Triton 的内存抽象

### 14.2.1 从程序员视角看 Triton 内存模型

Triton 的设计哲学是**让程序员专注于算法逻辑，编译器负责底层内存管理**。与 CUDA 中需要手动管理 shared memory 不同，Triton 中的内存操作被高度抽象：

```
Triton 内存抽象层次:

  程序员视角                    编译器行为                    硬件执行
  ─────────────                ─────────────                ──────────
  tl.load(ptr)          →     生成 load 指令         →     HBM → Register
                               如果需要，插入 SMEM

  tl.store(ptr, val)    →     生成 store 指令        →     Register → HBM

  tl.dot(a, b)          →     自动插入 SMEM 操作     →     HBM → SMEM → Register
                               (local_alloc/load)           → Tensor Core

  tl.where(cond, a, b)  →     纯寄存器操作           →     Register only
```

### 14.2.2 tt.load 的完整工作流程

`tl.load` 是 Triton 中最基础的内存操作。编译器会根据上下文自动决定是否需要经过 shared memory：

```python
import triton
import triton.language as tl

@triton.jit
def simple_load_kernel(
    in_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # 简单的 load → store
    # 编译器生成: global load → register → global store
    # 不涉及 shared memory
    data = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, data, mask=mask)
```

编译器对上述代码的 IR 变换：

```mlir
// Triton IR（前端生成）
%0 = tt.load %ptr, %mask : tensor<128xf32>
tt.store %out_ptr, %0, %mask : tensor<128xf32>

// TritonGPU IR（GPU 方言变换后）
// tt.load 被映射为 ttg.global_load，包含 layout 信息
%0 = ttg.global_load %ptr, %mask : tensor<128xf32, #blocked>
ttg.global_store %out_ptr, %0, %mask : tensor<128xf32, #blocked>
```

### 14.2.3 tl.dot 中隐式的 Shared Memory 操作

当执行矩阵乘法时，编译器会**自动**在 global load 和 dot 之间插入 shared memory 操作：

```python
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
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

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K

        # 加载 A 和 B 的分块
        # 编译器可能将这些数据先放入 shared memory
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + (k_offset + offs_k[None, :]) * stride_ak,
                     mask=(offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K), other=0.0)
        b = tl.load(B_ptr + (k_offset + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                     mask=((k_offset + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)

        # 矩阵乘加
        # 编译器在此处插入 local_alloc + local_load
        accumulator = tl.dot(a, b, accumulator)

    # 存储结果
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, accumulator,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

编译器对 `tl.dot` 的 IR 变换过程：

```mlir
// ============ 阶段 1: Triton IR (前端) ============
%a = tt.load %a_ptr, %a_mask : tensor<128x32xf16>
%b = tt.load %b_ptr, %b_mask : tensor<32x128xf16>
%acc = tt.dot %a, %b, %init : tensor<128x128xf32>

// ============ 阶段 2: Insert Shared Memory Ops ============
// 编译器自动插入 shared memory 分配和传输
%smem_a = triton_gpu.local_alloc %a : (tensor<128x32xf16>) -> !ttg.memdesc<128x32xf16, #shared>
%smem_b = triton_gpu.local_alloc %b : (tensor<32x128xf16>) -> !ttg.memdesc<32x128xf16, #shared>

// 从 shared memory 加载到寄存器（用于 tensor core 操作）
%reg_a = triton_gpu.local_load %smem_a : !ttg.memdesc<128x32xf16, #shared> -> tensor<128x32xf16, #mma>
%reg_b = triton_gpu.local_load %smem_b : !ttg.memdesc<32x128xf16, #shared> -> tensor<32x128xf16, #mma>

// 使用 tensor core 布局的寄存器执行 dot
%acc = tt.dot %reg_a, %reg_b, %init : tensor<128x128xf32, #mma>

// ============ 阶段 3: LLVM IR (后端) ============
// shared memory 分配映射到动态 shared memory
// local_load 映射到 ldmatrix 指令
// tt.dot 映射到 mma.sync 指令
```

### 14.2.4 Shared Memory 的自动生命周期管理

Triton 编译器通过分析 tensor 的使用范围（liveness），自动管理 shared memory 的分配和释放：

```
Shared Memory 生命周期（编译器视角）:

  代码位置            Shared Memory 操作              状态
  ──────────        ────────────────────           ──────
  循环开始前          %smem = local_alloc(...)       分配 buffer
  循环体中
    ├─ load 阶段      global_load → %smem            写入数据
    ├─ compute 阶段   %reg = local_load(%smem)       读取数据
    └─ (可选) store   local_store(...)               写入中间结果
  循环结束后          local_dealloc(%smem)           释放 buffer

  关键点:
  1. 分配大小由编译器根据 tile 大小和数据类型自动计算
  2. 当 buffer 不再需要时立即释放，减少峰值 SMEM 使用
  3. 嵌套循环中的 buffer 可能被提升（hoisted）到外层循环
```

```python
# 编译器计算 shared memory 大小的简化逻辑
# 对于 matmul kernel: BLOCK_M=128, BLOCK_N=256, BLOCK_K=32, dtype=fp16
smem_A_size = BLOCK_M * BLOCK_K * sizeof(fp16)  # 128 * 32 * 2 = 8 KB
smem_B_size = BLOCK_K * BLOCK_N * sizeof(fp16)  # 32 * 256 * 2 = 16 KB
total_smem  = smem_A_size + smem_B_size          # 24 KB

# 这个数值会影响 occupancy（每个 SM 能运行多少个线程块）
# A100 SMEM 容量: 164 KB → 164 / 24 ≈ 6 个线程块/SM（受其他因素限制可能更少）
```

### 14.2.5 内存操作的 IR 层次

Triton 编译器中内存操作的 IR 变换是一个多层次的过程：

```
IR 变换管线中的内存操作:

  ┌─────────────────────────────────────────────────────────────────┐
  │  Triton IR (前端 Python → MLIR)                                │
  │    tt.load / tt.store / tt.dot                                  │
  │    → 高级语义，不含 layout 信息                                  │
  └──────────────────────┬──────────────────────────────────────────┘
                         │  triton-opt: --tritongpu-materialize-op-info
                         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  TritonGPU IR (GPU 方言)                                       │
  │    ttg.global_load / ttg.global_store                           │
  │    ttg.local_alloc / ttg.local_load / ttg.local_store           │
  │    → 包含 distributed layout 和 shared layout 信息              │
  └──────────────────────┬──────────────────────────────────────────┘
                         │  triton-opt: --convert-triton-to-tritongpu
                         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  优化 Pass                                                      │
  │    --tritongpu-coalesce: 合并访问优化                           │
  │    --tritongpu-optimize-dot-operands: dot 操作数布局优化         │
  │    --tritongpu-pipeline: 流水线（异步拷贝插入）                  │
  │    --tritongpu-reduce-bank-conflicts: bank conflict 消除         │
  └──────────────────────┬──────────────────────────────────────────┘
                         │  triton-opt: --convert-tritongpu-to-llvm
                         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  LLVM IR → PTX                                                  │
  │    ld.global → ldmatrix / cp.async                              │
  │    st.shared / ld.shared → st.shared / ldmatrix                 │
  │    mma.sync / wgmma                                             │
  └─────────────────────────────────────────────────────────────────┘
```

<div data-component="MemoryHierarchyDiagram"></div>

[组件：MemoryHierarchyDiagram - 交互式 GPU 内存层次图，可点击查看各层的带宽、延迟和容量参数]

---

## 14.3 内存合并访问（Memory Coalescing）

### 14.3.1 什么是合并访问

GPU 的全局内存（HBM）以 128 字节或 32 字节的粒度（segment）被访问。当一个 warp（32 个线程）同时执行 load 指令时，硬件会将这些线程的内存请求合并为尽可能少的 segment 访问。

```
合并访问的基本原理:

  一个 warp 有 32 个线程，每个线程请求 4 字节（float32）

  ✅ 完全合并（最佳情况）:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Thread:   T0    T1    T2    T3    ... T28   T29   T30   T31      │
  │  Address:  0x00  0x04  0x08  0x0C  ... 0x70  0x74  0x78  0x7C     │
  │                                                                     │
  │  32 × 4B = 128B → 合并为 1 次 128-byte segment 访问               │
  │  带宽利用率: 100%                                                   │
  └─────────────────────────────────────────────────────────────────────┘

  ❌ 完全不合并（最差情况）:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Thread:   T0    T1    T2    T3    ... T28   T29   T30   T31      │
  │  Address:  0x00  0x80  0x100 0x180 ... 0xE00 0xE80 0xF00 0xF80    │
  │                                                                     │
  │  每个地址在不同 128-byte segment → 32 次独立访存                    │
  │  带宽利用率: 1/32 = 3.125%                                         │
  └─────────────────────────────────────────────────────────────────────┘

  ⚠️ 部分合并:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Thread:   T0-T7   T8-T15  T16-T23  T24-T31                       │
  │  Segment:  seg_0    seg_0    seg_1    seg_1                         │
  │                                                                     │
  │  32 个线程的地址分布在 2 个 segment → 2 次访存                      │
  │  带宽利用率: 64B / 256B = 50%                                      │
  └─────────────────────────────────────────────────────────────────────┘
```

### 14.3.2 为什么合并访问对性能至关重要

```
带宽利用率 vs 实际吞吐:

  合并模式              访存请求次数    带宽利用率    等效带宽 (A100)
  ──────────────       ────────────   ──────────   ────────────────
  完全合并 (128B)       1              100%         2.0 TB/s
  2-way 合并            2              50%          1.0 TB/s
  4-way 合并            4              25%          0.5 TB/s
  8-way 合并            8              12.5%        0.25 TB/s
  32-way (完全不合并)    32             3.1%         0.062 TB/s

  实际影响示例:
  加载一个 128×128 的 fp16 分块 (32 KB):
    完全合并: 32 KB / 2 TB/s = 16 μs
    32-way:   32 KB / 0.062 TB/s = 516 μs  → 慢 32 倍！
```

### 14.3.3 Triton 如何保证合并访问

Triton 编译器通过 `blocked` layout（也称为 `sliced_layout` 或 `contiguous_layout`）来自动确保合并访问。其核心机制是将 tensor 的维度映射到线程 ID，使得相邻线程访问相邻的内存地址。

```python
# Triton 中最常见的合并访问模式
@triton.jit
def coalesced_access_example(
    ptr, output,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # offsets: [0, 1, 2, ..., BLOCK_SIZE-1] + pid * BLOCK_SIZE
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # tl.arange(0, BLOCK_SIZE) 映射到线程 ID
    # Thread i 访问 ptr[pid * BLOCK_SIZE + i]
    # 相邻线程访问相邻地址 → 完全合并！
    data = tl.load(ptr + offsets, mask=mask)
    tl.store(output + offsets, data, mask=mask)
```

对应的 blocked layout 分析：

```mlir
// 编译器生成的 layout:
// #blocked<{sizePerThread=[1], threadsPerWarp=[32], warpsPerCTA=[4], order=[0]}>
//
// 这意味着:
//   - 每个线程处理 1 个元素（sizePerThread=[1]）
//   - 一个 warp 有 32 个线程，沿维度 0 排列
//   - 线程 i 访问 offset = warp_id * 32 + i
//   → 相邻线程访问相邻地址 → 合并访问

// 如果改成 sizePerThread=[4]:
// #blocked<{sizePerThread=[4], threadsPerWarp=[32], warpsPerCTA=[4], order=[0]}>
//
// 每个线程处理 4 个连续元素:
//   Thread 0: offset 0, 1, 2, 3
//   Thread 1: offset 4, 5, 6, 7
//   ...
//   Thread 31: offset 124, 125, 126, 127
// → 一个 warp 覆盖 128 个连续元素 → 1 次 128-byte 访问（fp32）
```

### 14.3.4 非合并访问模式及性能影响

```python
# ❌ 反模式 1: 列优先访问行优先矩阵
@triton.jit
def bad_access_pattern_col_major(
    A_ptr, output,
    stride_m, stride_n,  # A 是行优先: stride_m >> stride_n
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    # 加载一列: A[:, pid*BLOCK_N : (pid+1)*BLOCK_N]
    # Thread i 访问 A[i, n] = A_ptr + i * stride_m + n * stride_n
    # 相邻线程 i 和 i+1 的地址差 = stride_m (可能很大！)
    # → 高度不合并！
    col_data = tl.load(A_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)
```

```
列优先访问的问题:

  行优先矩阵 A (M=1024, N=1024, fp16):
    stride_m = 1024 * 2 = 2048 bytes
    stride_n = 2 bytes

  加载 A[:, 0:32] (一列分块):
    Thread 0 访问: A_ptr + 0 * 2048 + 0 = A_ptr + 0x0000
    Thread 1 访问: A_ptr + 1 * 2048 + 0 = A_ptr + 0x0800
    Thread 2 访问: A_ptr + 2 * 2048 + 0 = A_ptr + 0x1000
    ...
    Thread 31 访问: A_ptr + 31 * 2048 + 0 = A_ptr + 0xF800

    地址间隔: 2048 bytes = 16 个 128-byte segment
    32 个线程的请求分散在 32 个不同 segment → 32 次访存！
    带宽利用率: 3.125%

  对比加载 A[0, :] (一行分块):
    Thread 0 访问: A_ptr + 0 * 2048 + 0 * 2 = A_ptr + 0x0000
    Thread 1 访问: A_ptr + 0 * 2048 + 1 * 2 = A_ptr + 0x0002
    Thread 2 访问: A_ptr + 0 * 2048 + 2 * 2 = A_ptr + 0x0004
    ...

    地址间隔: 2 bytes → 连续！
    32 × 2B = 64B → 1 次 128-byte 访问
    带宽利用率: 50%（因为只用了 64B/128B）但只有 1 次访存
```

### 14.3.5 Triton 编译器的 Coalesce Pass

Triton 编译器有一个专门的 `coalesce` pass，负责优化内存访问模式。其核心逻辑在 `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp` 中：

```cpp
// lib/Dialect/TritonGPU/Transforms/Coalesce.cpp
// 核心逻辑：分析每个 load/store 操作的访问模式，
// 选择最优的 layout 使得线程沿连续维度访问

// 伪代码描述:
struct CoalescePass {
    void runOnOperation() {
        for (auto& op : getOperation().getOps<tt::LoadOp>()) {
            // 1. 分析指针的 access pattern
            //    - 计算每个线程访问的地址间距
            //    - 找到最连续的维度

            // 2. 选择最优的 sizePerThread
            //    - 保证相邻线程沿连续维度访问
            //    - 每个线程处理的元素数适中

            // 3. 选择 threadsPerWarp 和 warpsPerCTA
            //    - 一个 warp 沿连续维度覆盖 128 bytes
            //    - 确保完全合并

            // 4. 替换 layout
            changeLayout(op, newLayout);
        }
    }
};
```

### 14.3.6 二维分块的合并访问

在实际 kernel（如 matmul）中，数据通常是二维分块的。Triton 编译器需要为二维 tensor 选择合适的 layout：

```python
# 二维分块的加载
@triton.jit
def two_dim_load_example(
    A_ptr, stride_m, stride_n,
    pid_m, pid_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [0, 1, ..., 127]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [0, 1, ..., 255]

    # A_block: (128, 256) 的二维 tensor
    A_block = tl.load(A_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)
```

```
二维 layout 分析:

  Tensor 形状: (128, 256)
  Layout: #blocked<{sizePerThread=[1, 4], threadsPerWarp=[1, 32], warpsPerCTA=[128, 2], order=[1, 0]}>

  线程到元素的映射:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  维度 0 (行): 每个线程 1 个元素，由 warp 内线程沿 dim 0 分配      │
  │  维度 1 (列): 每个线程 4 个连续元素，由 warp 内线程沿 dim 1 分配   │
  │                                                                     │
  │  Warp 0, Thread 0: (0, [0,1,2,3])     → 地址: 0*stride_m + 0..3   │
  │  Warp 0, Thread 1: (0, [4,5,6,7])     → 地址: 0*stride_m + 4..7   │
  │  ...                                                                │
  │  Warp 0, Thread 31: (0, [124..127])   → 地址: 0*stride_m + 124..127│
  │  Warp 1, Thread 0: (1, [0,1,2,3])     → 地址: 1*stride_m + 0..3   │
  │  ...                                                                │
  │                                                                     │
  │  order=[1, 0]: 维度 1（列）是连续维度                             │
  │  → warp 内线程沿列方向排列 → 相邻线程访问相邻列 → 合并访问        │
  └─────────────────────────────────────────────────────────────────────┘
```

<div data-component="CoalescingVisualizer"></div>

[组件：CoalescingVisualizer - 可视化不同 layout 下的内存访问模式，对比合并与不合并的带宽利用率]

---

## 14.4 共享内存管理

### 14.4.1 TritonGPU 方言中的 Shared Memory 操作

Triton 编译器在 TritonGPU 方言（`ttg`）中定义了三个核心的 shared memory 操作：

```mlir
// 1. local_alloc: 在 shared memory 中分配一块缓冲区
//    对应 PTX: 动态 shared memory 分配（在 kernel 启动时确定大小）
%buffer = triton_gpu.local_alloc %source
    : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared>

// 2. local_load: 从 shared memory 加载到寄存器
//    对应 PTX: ldmatrix / ld.shared 指令
%reg = triton_gpu.local_load %buffer
    : !ttg.memdesc<128x32xf16, #shared> -> tensor<128x32xf16, #mma>

// 3. local_store: 将寄存器数据存入 shared memory
//    对应 PTX: st.shared 指令
triton_gpu.local_store %reg, %buffer
    : tensor<128x32xf16, #blocked> -> !ttg.memdesc<128x32xf16, #shared>
```

### 14.4.2 local_alloc 的工作原理

`local_alloc` 不是一个简单的内存分配指令——它表示将一个寄存器中的 tensor **物化**到 shared memory 中，并为它创建一个共享内存描述符（`memdesc`）。

```
local_alloc 的语义:

  输入: tensor<128x32xf16, #blocked>  (寄存器中的 tensor，blocked layout)
  输出: !ttg.memdesc<128x32xf16, #shared>  (shared memory 中的描述符)

  编译器行为:
  1. 在 shared memory 中分配 128 × 32 × 2 = 8192 bytes
  2. 将数据从寄存器写入 shared memory
  3. 返回一个 memdesc（包含 shared memory 地址和 layout 信息）

  关键: local_alloc 同时改变了 tensor 的 layout
    输入: #blocked (适合 global memory 合并访问)
    输出: #shared (适合 dot 操作数的加载)
```

### 14.4.3 local_load 与 ldmatrix 指令

`local_load` 从 shared memory 加载数据到寄存器，并转换 layout 以匹配后续操作（如 `tl.dot`）的需求。在 NVIDIA GPU 上，这通常映射到 `ldmatrix` 指令：

```mlir
// local_load 在 IR 中的表示
%reg_a = triton_gpu.local_load %smem_a
    : !ttg.memdesc<128x32xf16, #shared> -> tensor<128x32xf16, #mma_op_a>

// 编译到 PTX 时，映射为:
// ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0, r1, r2, r3}, [addr];
//
// ldmatrix 是专门为 Tensor Core 设计的加载指令:
// - 从 shared memory 加载一个 8×8 的矩阵片段到寄存器
// - 自动处理 transposition
// - 每个线程负责矩阵的一个元素，通过 shared memory 地址协同
```

```
ldmatrix 的工作方式:

  8×8 矩阵（fp16，每个元素 2 bytes）:
  ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
  │ T0    │ T1    │ T2    │ T3    │ T4    │ T5    │ T6    │ T7    │  Row 0
  ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
  │ T8    │ T9    │ T10   │ T11   │ T12   │ T13   │ T14   │ T15   │  Row 1
  ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
  │ T16   │ T17   │ T18   │ T19   │ T20   │ T21   │ T22   │ T23   │  Row 2
  ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
  │ T24   │ T25   │ T26   │ T27   │ T28   │ T29   │ T30   │ T31   │  Row 3
  ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
  │ ...   │ ...   │ ...   │ ...   │ ...   │ ...   │ ...   │ ...   │  Row 4-7
  └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘

  每个线程负责 1 个元素（或 2 个，取决于 .x2/.x4 变体）
  线程 i 从 shared memory 读取自己的元素到寄存器
  → 加载结果可以直接用于 mma.sync 指令的输入
```

### 14.4.4 Shared Memory 的 Layout 变换

Shared memory 的一个重要用途是**layout 变换**（transposition / swizzling）。当一个 tensor 需要从一种 layout 转换为另一种 layout 时，shared memory 充当中间缓冲：

```
Layout 变换示例: #blocked → #mma

  Global Memory (blocked layout):
  ┌─────────────────────────────────────────┐
  │  [0, 1, 2, 3, 4, 5, 6, 7, ...]        │  ← 连续存储，适合合并访问
  └─────────────────────────────────────────┘
           │ ttg.global_load
           ▼
  Register (blocked layout):
  ┌─────────────────────────────────────────┐
  │  T0: elem 0, T1: elem 1, T2: elem 2 ... │  ← 一个 warp 覆盖连续区域
  └─────────────────────────────────────────┘
           │ ttg.local_alloc + ttg.local_store
           ▼
  Shared Memory (shared layout):
  ┌─────────────────────────────────────────┐
  │  按 swizzled pattern 存储               │  ← 避免 bank conflict
  │  元素被重新排列以满足 mma layout 要求    │
  └─────────────────────────────────────────┘
           │ ttg.local_load (ldmatrix)
           ▼
  Register (mma layout):
  ┌─────────────────────────────────────────┐
  │  每个线程持有矩阵的一个片段              │  ← 可直接用于 mma.sync
  │  T0: frag[0], T1: frag[1], ...          │
  └─────────────────────────────────────────┘
```

### 14.4.5 实际 IR 走读：Matmul 中的 Shared Memory 操作

以下是一个完整 matmul kernel 中编译器插入的 shared memory 操作链：

```mlir
// ===== 完整的 Matmul K-loop 中的 Shared Memory 操作 =====

// 进入循环前: 分配 shared memory buffer
// 编译器计算总 SMEM 需求: 2 × BLOCK_M × BLOCK_K × sizeof(fp16) = 2 × 128 × 32 × 2 = 16 KB

scf.for %k = 0 to %num_k_steps step 1 {
    // 步骤 1: 从全局内存加载到寄存器 (blocked layout)
    %a_blocked = tt.load %a_ptr, %a_mask, %a_other
        : tensor<128x32xf16, #blocked>
    %b_blocked = tt.load %b_ptr, %b_mask, %b_other
        : tensor<32x128xf16, #blocked>

    // 步骤 2: 将寄存器数据存入 shared memory (layout 变换)
    %a_smem = triton_gpu.local_alloc %a_blocked
        : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared, #smem_alloc_a>
    %b_smem = triton_gpu.local_alloc %b_blocked
        : (tensor<32x128xf16, #blocked>) -> !ttg.memdesc<32x128xf16, #shared, #smem_alloc_b>

    // 步骤 3: 从 shared memory 加载到寄存器 (mma layout)
    %a_mma = triton_gpu.local_load %a_smem
        : !ttg.memdesc<128x32xf16, #shared, #smem_alloc_a> -> tensor<128x32xf16, #mma_op_a>
    %b_mma = triton_gpu.local_load %b_smem
        : !ttg.memdesc<32x128xf16, #shared, #smem_alloc_b> -> tensor<32x128xf16, #mma_op_b>

    // 步骤 4: 使用 mma layout 的寄存器执行 dot
    %acc = tt.dot %a_mma, %b_mma, %acc
        : tensor<128x32xf16, #mma_op_a> * tensor<32x128xf16, #mma_op_b> -> tensor<128x128xf32, #mma_acc>
}

// 循环结束后: shared memory buffer 自动释放
// (在实际代码中，buffer 可能在循环开始前分配，循环结束后释放)
```

### 14.4.6 Shared Memory 大小与 Occupancy 的权衡

```
Shared Memory 大小对 Occupancy 的影响（A100）:

  SM 总 SMEM 容量: 164 KB (可配置 SMEM/L1 比例)
  SM 最大线程块数: 32

  场景 1: SMEM = 8 KB/block
    → 164 / 8 = 20 blocks/SM (受 SMEM 限制)
    → 可能受寄存器限制更早达到上限
    → High occupancy → 高延迟隐藏能力

  场景 2: SMEM = 48 KB/block
    → 164 / 48 = 3 blocks/SM
    → Low occupancy → 可能无法完全隐藏延迟
    → 但每个 block 有更多数据复用

  场景 3: SMEM = 100 KB/block
    → 164 / 100 = 1 block/SM
    → Very low occupancy → 严重限制 latency hiding
    → 除非使用 async copy + 足够的 pipeline stages

  权衡公式:
    Effective Throughput ≈ min(Compute Throughput, Memory Throughput) × Occupancy_factor

    其中 Occupancy_factor 不是线性关系:
    - 低 occupancy 时，增加 occupancy 效果显著
    - 高 occupancy 时（>50%），继续增加效果递减
```

<div data-component="OccupancyCalculator"></div>

[组件：OccupancyCalculator - 交互式计算器，输入 SMEM 大小和寄存器数量，计算 A100 上的 occupancy]

---

## 14.5 Bank Conflict

### 14.5.1 Shared Memory 的 Bank 结构

Shared memory 被组织为多个 bank，每个 bank 以交织（interleaved）方式存储数据。这种设计允许多个 bank 同时被访问，但当多个线程访问同一个 bank 的不同地址时，就会产生 bank conflict。

```
Shared Memory Bank 结构（A100 / SM 8.0）:

  共 32 个 bank，每个 bank 宽度 4 bytes (32 bits)

  地址映射规则:
    元素地址 addr → bank = (addr / 4) % 32

  存储布局示意（以 fp32 为例）:
  ┌────────┬────────┬────────┬────────┬─────┬────────┐
  │ Bank 0 │ Bank 1 │ Bank 2 │ Bank 3 │ ... │ Bank 31│
  ├────────┼────────┼────────┼────────┼─────┼────────┤
  │ addr 0 │ addr 4 │ addr 8 │addr 12 │ ... │addr 124│
  │ 128    │ 132    │ 136    │ 140    │ ... │ 252    │
  │ 256    │ 260    │ 264    │ 268    │ ... │ 380    │
  │  ...   │  ...   │  ...   │  ...   │ ... │  ...   │
  └────────┴────────┴────────┴────────┴─────┴────────┘

  fp16 数据 (2 bytes per element):
    两个 fp16 元素共享一个 4-byte bank slot
    addr 0 和 addr 2 → 都映射到 Bank 0
    addr 1 和 addr 3 → 都映射到 Bank 0 (但偏移不同)
```

### 14.5.2 Bank Conflict 的成因

当一个 warp 中的多个线程**同时**访问**同一个 bank** 的**不同地址**时，这些访问必须被序列化，这就是 bank conflict。

```
Bank Conflict 类型:

  ✅ 无 Bank Conflict（理想情况）:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Thread:  T0    T1    T2    T3    ... T28   T29   T30   T31       │
  │  Bank:    B0    B1    B2    B3    ... B28   B29   B30   B31       │
  │  Address: 0x00  0x04  0x08  0x0C  ... 0x70  0x74  0x78  0x7C      │
  │                                                                     │
  │  每个线程访问不同 bank → 32 路并行 → 1 cycle 完成                  │
  └─────────────────────────────────────────────────────────────────────┘

  ❌ 2-way Bank Conflict:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Thread:  T0    T1    T2    T3    T4    T5    T6    T7    ...      │
  │  Bank:    B0    B1    B2    B3    B0    B1    B2    B3    ...      │
  │  Address: 0x00  0x04  0x08  0x0C  0x80  0x84  0x88  0x8C  ...      │
  │                                                                     │
  │  T0 和 T4 都访问 Bank 0 (不同行) → 2-way conflict → 2 cycles      │
  │  T1 和 T5 都访问 Bank 1 (不同行) → 2-way conflict                  │
  │  ...                                                                │
  │  总计: 2 cycles（而非 1 cycle）                                    │
  └─────────────────────────────────────────────────────────────────────┘

  ❌ 32-way Bank Conflict（最差情况）:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  所有 32 个线程都访问同一个 bank 的不同地址                        │
  │  → 32 cycles（而非 1 cycle）→ 性能下降 32 倍！                    │
  └─────────────────────────────────────────────────────────────────────┘

  ✅ Broadcast（多个线程访问同一地址，不算 conflict）:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Thread:  T0    T1    T2    ... T15   T16   T17   ... T31         │
  │  Bank:    B0    B0    B0    ... B0    B0    B0    ... B0           │
  │  Address: 0x00  0x00  0x00  ... 0x00  0x00  0x00  ... 0x00         │
  │                                                                     │
  │  所有线程访问同一个地址 → Broadcast → 1 cycle (不冲突！)          │
  │  硬件检测到同一地址的访问会自动 broadcast                          │
  └─────────────────────────────────────────────────────────────────────┘
```

### 14.5.3 二维 Tensor 的 Bank Conflict 分析

在实际 kernel 中，shared memory 中存储的通常是二维 tensor。bank conflict 的严重程度取决于 tensor 的列宽度（leading dimension）。

```
二维 Tensor 的 Bank Conflict 分析:

  情况 1: 列宽度 = 32 个 fp32 元素（128 bytes）
  ┌──────────────────────────────────────────────────────────────────┐
  │  共享内存中的 32×32 fp32 矩阵:                                  │
  │                                                                  │
  │  行 0: [B0 B1 B2 B3 ... B31]   ← 32 个元素，每列一个 bank     │
  │  行 1: [B0 B1 B2 B3 ... B31]   ← 同样的 bank 分布              │
  │  行 2: [B0 B1 B2 B3 ... B31]                                   │
  │  ...                                                            │
  │                                                                  │
  │  加载一列 (如 T0-T31 都访问第 0 列):                            │
  │  所有线程都访问 Bank 0 → 32-way bank conflict！                 │
  └──────────────────────────────────────────────────────────────────┘

  情况 2: 列宽度 = 33 个 fp32 元素（+1 padding）
  ┌──────────────────────────────────────────────────────────────────┐
  │  行 0: [B0 B1 B2 B3 ... B31 B0]   ← 第 33 个元素回到 Bank 0    │
  │  行 1: [B1 B2 B3 B4 ... B0 B1]   ← 每行起始 bank 偏移 1        │
  │  行 2: [B2 B3 B4 B5 ... B1 B2]                                 │
  │  ...                                                            │
  │                                                                  │
  │  加载一列 (T0-T31 都访问第 0 列):                               │
  │  T0→Bank0, T1→Bank1, T2→Bank2, ... T31→Bank31                  │
  │  → 无 bank conflict！                                           │
  └──────────────────────────────────────────────────────────────────┘
```

### 14.5.4 Triton 编译器的 Bank Conflict 规避策略

Triton 编译器在 `ReduceBankConflict` pass 中使用 **swizzling** 技术来消除 bank conflict。Swizzling 通过重新映射 shared memory 地址，使得逻辑上相邻的元素在物理上分布在不同的 bank 中。

```cpp
// lib/Dialect/TritonGPU/Transforms/ReduceBankConflicts.cpp
// 核心策略: 通过 swizzle pattern 打乱 shared memory 地址

// Swizzle 的基本思想:
//   原始地址: addr
//   Swizzled 地址: addr XOR (row_id << shift)
//
// 效果: 相邻行的同一列元素被映射到不同 bank
```

```
Swizzle 模式示意（以 4×8 矩阵，8 个 bank 为例）:

  原始布局（无 swizzle）:
  ┌─────────────────────────────────────────────────────┐
  │     Col0  Col1  Col2  Col3  Col4  Col5  Col6  Col7  │
  │  Row0 [B0   B1   B2   B3   B4   B5   B6   B7]     │
  │  Row1 [B0   B1   B2   B3   B4   B5   B6   B7]     │  ← 每行相同 bank
  │  Row2 [B0   B1   B2   B3   B4   B5   B6   B7]     │
  │  Row3 [B0   B1   B2   B3   B4   B5   B6   B7]     │
  │                                                      │
  │  加载列 0: Bank 冲突！[B0, B0, B0, B0] → 4-way     │
  └─────────────────────────────────────────────────────┘

  Swizzled 布局（XOR-based）:
  ┌─────────────────────────────────────────────────────┐
  │     Col0  Col1  Col2  Col3  Col4  Col5  Col6  Col7  │
  │  Row0 [B0   B1   B2   B3   B4   B5   B6   B7]     │
  │  Row1 [B1   B0   B3   B2   B5   B4   B7   B6]     │  ← XOR(row, col)
  │  Row2 [B2   B3   B0   B1   B6   B7   B4   B5]     │
  │  Row3 [B3   B2   B1   B0   B7   B6   B5   B4]     │
  │                                                      │
  │  加载列 0: [B0, B1, B2, B3] → 无冲突！             │
  └─────────────────────────────────────────────────────┘

  Swizzle 计算:
    swizzled_col = col XOR row  (对于 8 bank 的情况)
    physical_addr = row * stride + swizzled_col
```

### 14.5.5 Triton 中 Bank Conflict 的实际观察

Triton 编译器通常能自动避免大部分 bank conflict，但在某些情况下仍需注意：

```python
# 触发 bank conflict 的场景：手动布局不当时
@triton.jit
def potential_bank_conflict(
    smem_ptr,  # 指向 shared memory 的指针
    output,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tid = tl.arange(0, BLOCK)

    # 如果 STRIDE 是 32 的倍数（fp32）或 16 的倍数（fp16），
    # 且所有线程访问同一列的不同行 → bank conflict
    # 但在 Triton 中，编译器通常会自动插入 swizzle 来避免

    data = tl.load(smem_ptr + tid * STRIDE)  # 可能有 bank conflict
    tl.store(output + tid, data)

# 通常情况下，Triton 编译器会:
# 1. 在 local_alloc 时选择 swizzled shared layout
# 2. 在生成 ldmatrix/st.shared 时自动计算 swizzled 地址
# 3. 对于 dot 操作数，保证 bank-conflict-free 访问
```

### 14.5.6 Bank Conflict 的性能影响量化

```
Bank Conflict 对 Shared Memory 带宽的影响:

  无冲突:    1 cycle per access → 32 elements / cycle (per warp)
  2-way:     2 cycles per access → 16 elements / cycle → 带宽减半
  4-way:     4 cycles per access → 8 elements / cycle → 带宽 1/4
  8-way:     8 cycles per access → 4 elements / cycle → 带宽 1/8
  32-way:    32 cycles per access → 1 element / cycle → 带宽 1/32

  实际 kernel 影响:
  ┌──────────────────────────────────────────────────────────────────┐
  │  场景                    SMEM 带宽    实际性能影响               │
  ├──────────────────────────────────────────────────────────────────┤
  │  GEMM K-loop 内          高度敏感     每个 cycle 的 bank conflict │
  │                                      直接累加到 K-loop 延迟中   │
  │                                                                  │
  │  Softmax reduce           中等敏感    reduce 阶段的 bank conflict │
  │                                      影响 softmax 的整体延迟    │
  │                                                                  │
  │  Element-wise             低敏感      不使用 shared memory       │
  │                                      bank conflict 不适用       │
  └──────────────────────────────────────────────────────────────────┘

  经验法则: bank conflict 导致的性能损失通常在 5%-30% 之间
           对于 compute-bound kernel，影响较小
           对于 memory-bound kernel，影响较大
```

<div data-component="BankConflictVisualizer"></div>

[组件：BankConflictVisualizer - 交互式 shared memory bank 访问模拟器，输入访问模式显示 bank conflict 情况]

---

## 14.6 异步拷贝（Async Copy）

### 14.6.1 cp.async 指令简介

从 Ampere 架构（SM 8.0）开始，NVIDIA GPU 引入了 `cp.async` 指令族，允许从全局内存**直接**加载数据到 shared memory，**不经过寄存器**。这显著减少了寄存器压力并允许计算与访存重叠。

```
传统加载路径 vs 异步拷贝路径:

  传统路径:
    HBM → L1/L2 → Register → Shared Memory
    步骤: 1) global load → register
          2) register → shared memory store
    问题: 寄存器被中间数据占用；两步操作不能完全流水线化

  异步拷贝路径 (cp.async):
    HBM → L1/L2 → Shared Memory  (跳过寄存器！)
    步骤: 1) cp.async: 直接从 global memory 到 shared memory
          2) 副本完成时通过 barrier 同步
    优势: 寄存器不被占用；可以实现计算与访存完全重叠

  时序对比:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  传统路径:                                                          │
  │  Cycle:  [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]                 │
  │  Load:   [===global load===]                                       │
  │  Store:                   [==shared store==]                        │
  │  Compute:                                         [===dot===]       │
  │                                                                     │
  │  异步拷贝路径:                                                      │
  │  Cycle:  [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]                 │
  │  cp.async:[========global → shared========]                        │
  │  Compute: [========previous dot========]  [===dot===]              │
  │                     ↑ 计算与访存重叠！                              │
  └─────────────────────────────────────────────────────────────────────┘
```

### 14.6.2 Triton 中的异步拷贝映射

Triton 编译器在 `pipeline` pass 中自动将同步的全局内存加载替换为异步拷贝操作。

```python
# Triton 源码（程序员写）
@triton.jit
def matmul_pipelined(
    A_ptr, B_ptr, C_ptr,
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

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + (k_offset + offs_k[None, :]) * stride_ak,
                     mask=(offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K), other=0.0)
        b = tl.load(B_ptr + (k_offset + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                     mask=((k_offset + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
        accumulator = tl.dot(a, b, accumulator)

    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, accumulator,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

```mlir
// ===== 编译器 pipeline pass 后的 IR =====
// 关键变换: 同步的 tt.load 被分解为:
//   1. cp.async (全局内存 → shared memory)
//   2. cp.async.commit_group
//   3. cp.async.wait_group

// 初始化阶段: 预取前 num_stages-1 个迭代的数据
scf.for %init_k = 0 to %num_stages_minus_1 {
    %a = tt.load %a_ptr_init, %mask : tensor<128x32xf16>
    // 分配 shared memory buffer 并异步拷贝
    %a_smem = triton_gpu.local_alloc %a
    triton_gpu.local_store %a, %a_smem
    // 提交异步拷贝组
    // cp.async.commit_group
}

// 主循环: 计算当前迭代 + 预取未来迭代
scf.for %k = 0 to %num_k_steps {
    // 等待最早的异步拷贝组完成
    // cp.async.wait_group(N-2)

    // 从 shared memory 加载（数据已就绪）
    %a_reg = triton_gpu.local_load %a_smem_current
    %b_reg = triton_gpu.local_load %b_smem_current

    // 执行计算（与下一次加载重叠）
    %acc = tt.dot %a_reg, %b_reg, %acc

    // 启动下一次异步拷贝
    %a_next = tt.load %a_ptr_next, %mask : tensor<128x32xf16>
    triton_gpu.local_store %a_next, %a_smem_next
    // cp.async.commit_group
}
```

### 14.6.3 num_stages 与 Pipeline 的关系

`num_stages` 参数控制软件流水线的深度，即同时"在飞行中"（in-flight）的异步拷贝组数量。

```
num_stages = 2 的流水线时序:

  时间 →  T0    T1    T2    T3    T4    T5    T6    T7
  ─────────────────────────────────────────────────────
  加载:
    iter0 [===cp.async iter0===]
    iter1                          [===cp.async iter1===]
  计算:
    iter0                     [===dot iter0===]
    iter1                                        [===dot iter1===]

  说明:
  - T0-T2: 异步加载 iter0 的数据到 shared memory
  - T2: iter0 数据就绪，开始计算
  - T3-T5: 计算 iter0 的同时，异步加载 iter1 的数据
  - T5: iter1 数据就绪，开始计算
  → 计算与访存完全重叠！

num_stages = 3 的流水线时序:

  时间 →  T0    T1    T2    T3    T4    T5    T6    T7    T8    T9
  ─────────────────────────────────────────────────────────────────
  加载:
    iter0 [===cp.async iter0===]
    iter1           [===cp.async iter1===]
    iter2                     [===cp.async iter2===]
  计算:
    iter0                          [===dot iter0===]
    iter1                                        [===dot iter1===]
    iter2                                                       [===dot...]

  优势: 更深的流水线可以更好地隐藏内存延迟
  代价: 更多的 shared memory 使用（N 个 buffer instead of 1）
```

### 14.6.4 Pipeline Pass 的 IR 级详解

Triton 编译器的 pipeline pass（`lib/Dialect/TritonGPU/Transforms/Pipeline.cpp`）是实现异步拷贝的核心：

```cpp
// lib/Dialect/TritonGPU/Transforms/Pipeline.cpp
// 核心逻辑:

struct PipelinePass {
    void runOnOperation() {
        // 1. 识别循环中的全局内存加载操作
        for (auto& forOp : getOperation().getOps<scf::ForOp>()) {
            SmallVector<tt::LoadOp> loads = findGlobalLoads(forOp);

            // 2. 分析数据依赖: 确定哪些 load 可以被提前（prefetch）
            //    - load 的结果必须在循环体的后半部分才被使用
            //    - load 和使用之间必须有足够的"距离"

            // 3. 为每个 load 分配 shared memory buffer
            for (auto load : loads) {
                auto bufferType = getSharedMemType(load.getType());
                allocateSharedMemoryBuffer(load, bufferType);
            }

            // 4. 重写循环: 将 load 移到循环体的前面
            //    并插入 cp.async + barrier 操作
            rewriteLoopWithPipeline(forOp, loads);

            // 5. 处理循环的 prologue（预填充）和 epilogue（排空）
            addPrologueEpilogue(forOp, loads);
        }
    }
};
```

### 14.6.5 async_agent_group_barrier 和 Token 机制

Triton 使用 token 机制来管理异步操作的依赖关系：

```mlir
// 使用 token 链来保证正确的执行顺序

// 初始化: 创建初始 token
%token_0 = triton_gpu.async_agent_group_barrier : !ttg.token

// 提交异步拷贝: 消耗旧 token，产生新 token
%a = tt.load %ptr : tensor<128x32xf16>  // 底层映射为 cp.async
%token_1 = triton_gpu.async_agent_group_barrier [%token_0] : !ttg.token

// 等待: 消耗 token，保证之前的异步操作完成
triton_gpu.async_agent_group_barrier [%token_1] : !ttg.token

// 此时可以安全使用之前加载的数据
%reg = triton_gpu.local_load %smem
```

### 14.6.6 num_stages 的选择指南

```
num_stages 选择的经验法则:

  num_stages = 1:
    - 不使用异步流水线（同步模式）
    - SMEM 使用最小
    - 适合: 计算量小、内存延迟不重要的 kernel

  num_stages = 2:
    - 双缓冲（double buffering）
    - SMEM 使用 = 2 × tile_size
    - 适合: 大多数 GEMM kernel
    - 最常见的选择

  num_stages = 3:
    - 三缓冲（triple buffering）
    - SMEM 使用 = 3 × tile_size
    - 适合: 非常大的 GEMM 或内存延迟特别高的场景
    - 需要足够的 SMEM 容量

  num_stages = 4+:
    - 更深的流水线
    - SMEM 使用 = N × tile_size
    - 适合: H100 等新架构（更大的 SMEM 容量）
    - 收益递减，通常 3-4 是最优

  A100 上的典型配置:
    GEMM (fp16, BLOCK_M=128, BLOCK_N=256, BLOCK_K=32):
      tile_smem = (128×32 + 32×256) × 2B = 24 KB
      num_stages=2: 48 KB → occupancy ≈ 3 blocks/SM
      num_stages=3: 72 KB → occupancy ≈ 2 blocks/SM
      推荐: num_stages=2 或 3（通过 autotuning 确定）

    GEMM (fp16, BLOCK_M=256, BLOCK_N=128, BLOCK_K=64):
      tile_smem = (258×64 + 64×128) × 2B = 48 KB (约)
      num_stages=2: 96 KB → occupancy ≈ 1 block/SM
      推荐: num_stages=2（受 SMEM 限制）
```

<div data-component="PipelineVisualizer"></div>

[组件：PipelineVisualizer - 交互式流水线时序图，可调整 num_stages 观察计算与访存的重叠情况]

---

## 14.7 Eviction Policy

### 14.7.1 L2 Cache 与 Eviction Policy 概述

L2 Cache 是所有 SM 共享的缓存层，容量为 40 MB（A100）。当 kernel 访问全局内存时，数据会被缓存到 L2。Triton 提供了 `eviction_policy` 参数来控制数据在 L2 中的保留策略。

```
L2 Cache 的作用:

  ┌─────────────────────────────────────────────────────────────────────┐
  │                          GPU 芯片                                   │
  │  ┌─────────────┐  ┌─────────────┐        ┌─────────────┐          │
  │  │    SM 0     │  │    SM 1     │  ...   │   SM 107    │          │
  │  │  L1 + SMEM  │  │  L1 + SMEM  │        │  L1 + SMEM  │          │
  │  └──────┬──────┘  └──────┬──────┘        └──────┬──────┘          │
  │         │                │                      │                  │
  │         └────────────────┼──────────────────────┘                  │
  │                          │                                          │
  │                   ┌──────┴──────┐                                   │
  │                   │   L2 Cache  │  40 MB, ~5 TB/s                 │
  │                   │  (共享)     │                                   │
  │                   └──────┬──────┘                                   │
  └──────────────────────────┼──────────────────────────────────────────┘
                             │
                      ┌──────┴──────┐
                      │    HBM      │  80 GB, 2 TB/s
                      └─────────────┘

  L2 Cache 的重要性:
  - 如果数据在 L2 中命中，延迟从 ~400 cycles 降到 ~200 cycles
  - 对于频繁访问的权重矩阵（如 GEMM 中的 A、B），L2 命中率至关重要
  - 对于大模型，L2 可能不足以缓存所有数据，需要策略管理
```

### 14.7.2 eviction_policy 参数

Triton 的 `tl.load` 支持 `eviction_policy` 参数，控制加载数据在 L2 中的缓存策略：

```python
# 使用 eviction_policy 参数

# 策略 1: evict_first (低优先级保留)
# 加载的数据在 L2 中标记为"优先驱逐"
# 适合: 临时数据，不会被重复访问
data = tl.load(ptr, eviction_policy="evict_first")

# 策略 2: evict_last (高优先级保留)
# 加载的数据在 L2 中标记为"最后驱逐"
# 适合: 权重矩阵等会被反复访问的数据
weights = tl.load(ptr, eviction_policy="evict_last")
```

### 14.7.3 Eviction Policy 的实际应用

在 GEMM kernel 中，合理使用 eviction policy 可以显著提升 L2 命中率：

```python
@triton.jit
def matmul_with_eviction_policy(
    A_ptr, B_ptr, C_ptr,
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

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K

        # A 矩阵: 每个 program 沿 K 维度遍历所有行
        # → 可能被多个 program 重复访问 → 使用 evict_last
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + (k_offset + offs_k[None, :]) * stride_ak,
            mask=(offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K),
            other=0.0,
            eviction_policy="evict_last",  # 保留更久
        )

        # B 矩阵: 每个 program 沿 K 维度遍历所有列
        # → 可能被多个 program 重复访问 → 使用 evict_last
        b = tl.load(
            B_ptr + (k_offset + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
            mask=((k_offset + offs_k[:, None]) < K) & (offs_n[None, :] < N),
            other=0.0,
            eviction_policy="evict_last",  # 保留更久
        )

        accumulator = tl.dot(a, b, accumulator)

    # C 矩阵: 每个 program 只写一次 → 不需要缓存
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )
```

### 14.7.4 L2 Cache 友好的访问模式

```
L2 Cache 友好 vs 不友好的访问模式:

  ✅ 友好: 行优先遍历（同一 tile 的多个 program 访问相邻行）
  ┌──────────────────────────────────────────────────────────────────┐
  │  矩阵 A (M×K):                                                  │
  │                                                                  │
  │  Program 0: A[0:128, :]      ← 行 0-127                        │
  │  Program 1: A[128:256, :]    ← 行 128-255 (与 P0 部分重叠)     │
  │  Program 2: A[256:384, :]    ← 行 256-383                       │
  │                                                                  │
  │  沿 K 维度循环时，同一 program 反复访问同一行                   │
  │  → 第一次访问后数据在 L2 中 → 后续访问命中                     │
  └──────────────────────────────────────────────────────────────────┘

  ❌ 不友好: 列优先遍历（每个 program 访问不同的列）
  ┌──────────────────────────────────────────────────────────────────┐
  │  如果沿 N 维度（列方向）分配 program:                           │
  │                                                                  │
  │  Program 0: A[:, 0:128]      ← 每次循环访问不同的 K 列          │
  │  Program 1: A[:, 128:256]    ← 访问不同的 K 列                  │
  │                                                                  │
  │  沿 K 维度循环时，每个 program 跳跃式访问                       │
  │  → L2 cache line 被反复淘汰和重新加载                           │
  │  → L2 命中率低                                                  │
  └──────────────────────────────────────────────────────────────────┘

  优化策略:
  1. Program 调度顺序: 优先沿 M 维度（行方向）分配 program
  2. 数据布局: 确保访问模式与 cache line 对齐
  3. Tile 大小: 使 tile 宽度是 cache line 大小的整数倍
```

### 14.7.5 多级 Cache 协同

```
完整的内存访问路径与 Cache 层次:

  tl.load(ptr)
    │
    ▼
  L1 Cache (per SM, 128 KB)
    │ 命中 → 直接返回 (~30 cycles)
    │ 未命中 ↓
    ▼
  L2 Cache (shared, 40 MB)
    │ 命中 → 返回到 L1 → 返回到 kernel (~200 cycles)
    │ 未命中 ↓
    ▼
  HBM (80 GB, 2 TB/s)
    │ 读取 → 返回到 L2 → 返回到 L1 → 返回到 kernel (~400 cycles)
    ▼
  数据到达寄存器

  Cache Line 大小:
  - L1 Cache Line: 128 bytes
  - L2 Cache Line: 32 bytes (注意: L2 line 更小！)
  → 一次 tl.load 可能触发多个 L2 cache line 请求
```

<div data-component="EvictionPolicyDemo"></div>

[组件：EvictionPolicyDemo - 对比演示 evict_first 和 evict_last 对 L2 命中率和性能的影响]

---

## 14.8 内存使用优化指南

### 14.8.1 优化原则总览

```
内存优化的核心原则:

  1. 减少全局内存访问次数
     - 使用分块（tiling）: 每个数据元素只从 HBM 加载一次
     - 利用 L2 cache: 合理的 program 调度顺序增加 L2 命中

  2. 最大化数据复用
     - Shared memory 中的数据应被多次计算使用
     - 计算强度 = 计算量 / 访存量，越高越接近 compute-bound

  3. 避免 shared memory 浪费
     - 控制 tile 大小，避免 SMEM 过度使用降低 occupancy
     - 不需要的数据不要放入 shared memory

  4. 避免寄存器溢出（register spilling）
     - 减少同时活跃的变量数量
     - 使用 tl.where 等操作减少临时 tensor
```

### 14.8.2 减少全局内存访问次数

```python
# ❌ 差的模式: 数据被多次从 HBM 加载
@triton.jit
def bad_multi_load(A_ptr, B_ptr, C_ptr, M, N, K, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    # 第一次加载 A
    a1 = tl.load(A_ptr + offs)
    # ... 一些计算 ...

    # 又一次加载 A（相同数据！）
    a2 = tl.load(A_ptr + offs)  # ← 浪费！数据可能已从 L2 淘汰

    # 结果
    tl.store(C_ptr + offs, a1 + a2)


# ✅ 好的模式: 数据只加载一次，多次使用
@triton.jit
def good_single_load(A_ptr, B_ptr, C_ptr, M, N, K, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    # 只加载一次
    a = tl.load(A_ptr + offs)
    # ... 多次使用 a ...
    result = a * 2 + a * 3  # 复用寄存器中的值

    tl.store(C_ptr + offs, result)
```

```
数据复用分析:

  理想情况: 每个数据元素只从 HBM 加载一次

  实际 GEMM 分析:
    A 矩阵 (M×K): 每行被 N/BLOCK_N 个 program 使用
    B 矩阵 (K×N): 每列被 M/BLOCK_M 个 program 使用

    如果 M=4096, BLOCK_M=128:
      A 的每行被 4096/128 = 32 个 program 沿 N 方向使用
      → A 的数据被加载了 32 次（每个 N-program 一次）

    优化: 利用 L2 cache
      如果 32 个 program 按顺序调度（先 N0, N1, N2, ...），
      A 的数据在 L2 中，后续 program 可以直接命中
      → 等效于只从 HBM 加载 1 次

    更好的优化: 使用 group-based program 调度（Triton 自动支持）
      将访问相同 A 行的 program 分组调度
      → L2 命中率最大化
```

### 14.8.3 最大化数据复用

```python
# 数据复用的关键: 让每个数据元素参与尽可能多的计算

# GEMM 的复用分析:
#   A[m, k] 被所有计算 C[m, :] 的 program 使用 → 复用次数 = N / BLOCK_N
#   B[k, n] 被所有计算 C[:, n] 的 program 使用 → 复用次数 = M / BLOCK_M
#
# 如果 M = N = 4096, BLOCK_M = BLOCK_N = 128:
#   A 的复用次数 = 4096/128 = 32
#   B 的复用次数 = 4096/128 = 32
#   → 理论计算强度 = 2 * K * 32 / (2 * 32) = K FLOPs/Byte
#   → K = 1024 时，计算强度 = 1024 >> 156 (A100 ridge point)

# 对于非 GEMM 的 kernel，如 LayerNorm:
@triton.jit
def layernorm_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    stride_xm, stride_ym,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offset = pid * stride_xm
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # 加载一行数据 (只加载一次！)
    x = tl.load(X_ptr + row_offset + cols, mask=mask, other=0.0)

    # 计算均值和方差 (多次使用 x)
    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N

    # 归一化 (再次使用 x)
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd

    # 加载权重和偏置
    w = tl.load(W_ptr + cols, mask=mask)
    b = tl.load(B_ptr + cols, mask=mask)

    # 仿射变换
    y = x_norm * w + b

    # 存储结果
    tl.store(Y_ptr + row_offset + cols, y, mask=mask)

    # 复用分析:
    # x 从 HBM 加载 1 次，但在寄存器中被使用 5 次
    # 计算强度: ~10 N / (4N + 8) ≈ 2.5 FLOPs/Byte (memory-bound!)
```

### 14.8.4 避免 Shared Memory 浪费

```
Shared Memory 使用分析:

  浪费模式 1: 过大的 tile
  ┌──────────────────────────────────────────────────────────────────┐
  │  BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, fp16                     │
  │  SMEM = (256×64 + 64×256) × 2B = 64 KB                          │
  │  num_stages=2: 128 KB → 只能放 1 个 block/SM (A100: 164 KB)     │
  │  occupancy = 1 block/SM → 可能无法完全隐藏延迟                   │
  │                                                                  │
  │  更好的选择:                                                     │
  │  BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, fp16                     │
  │  SMEM = (128×32 + 32×128) × 2B = 16 KB                          │
  │  num_stages=2: 32 KB → 可放 5 个 block/SM                       │
  │  occupancy = 5 blocks/SM → 更好的延迟隐藏                        │
  └──────────────────────────────────────────────────────────────────┘

  浪费模式 2: 不必要的 shared memory 使用
  ┌──────────────────────────────────────────────────────────────────┐
  │  对于简单的 element-wise kernel:                                │
  │                                                                  │
  │  ❌ 不需要 shared memory:                                       │
  │  @triton.jit                                                     │
  │  def bad_kernel(ptr, BLOCK: tl.constexpr):                       │
  │      data = tl.load(ptr + offs)                                  │
  │      # 数据在寄存器中，不需要 SMEM                              │
  │      # 但某些操作可能触发编译器自动插入 SMEM 操作               │
  │      result = data * 2                                           │
  │      tl.store(ptr + offs, result)                                │
  │                                                                  │
  │  ✅ 简单操作直接在寄存器中完成:                                 │
  │  → tl.where, tl.exp, tl.log 等都是纯寄存器操作                 │
  │  → 不应触发 shared memory 分配                                   │
  └──────────────────────────────────────────────────────────────────┘
```

### 14.8.5 寄存器溢出（Register Spilling）的识别与避免

当 kernel 使用的寄存器数量超过硬件限制（每线程 255 个 32-bit 寄存器，A100）时，编译器会将部分变量"溢出"到 local memory（实际上是 HBM 的 per-thread 区域），这会导致严重的性能下降。

```
寄存器溢出的影响:

  正常访问（寄存器）:
    latency = 1 cycle
    throughput = 无限（每个周期完成）

  溢出访问（local memory，实际是 HBM）:
    latency = 400-600 cycles
    throughput = 受 HBM 带宽限制

  性能影响:
    如果一个内层循环的变量发生溢出:
    - 每次循环迭代额外增加 400+ cycles
    - 对于 K=1024 的循环，额外开销 = 1024 × 400 = 409,600 cycles
    - 而正常计算可能只需要 1024 × 10 = 10,240 cycles
    → 性能下降 40 倍！
```

```python
# 寄存器溢出的常见原因和解决方案

# 原因 1: 循环中保存过多中间结果
@triton.jit
def register_heavy(
    A_ptr, B_ptr, C_ptr, D_ptr, E_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    # 加载 5 个 tensor → 5 × BLOCK 个寄存器
    a = tl.load(A_ptr + offs)
    b = tl.load(B_ptr + offs)
    c = tl.load(C_ptr + offs)
    d = tl.load(D_ptr + offs)
    e = tl.load(E_ptr + offs)

    # 如果 BLOCK = 128, 每个 tensor 占 128 个寄存器
    # 5 × 128 = 640 个寄存器 → 超过 255 限制！
    # 编译器必须溢出部分变量到 local memory

    result = a + b + c + d + e
    tl.store(E_ptr + offs, result)


# 解决方案: 分阶段计算，减少同时活跃的变量
@triton.jit
def register_efficient(
    A_ptr, B_ptr, C_ptr, D_ptr, E_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    # 分阶段加载和计算
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    acc += tl.load(A_ptr + offs)  # 加载 A，累加，释放 A 的寄存器
    acc += tl.load(B_ptr + offs)  # 加载 B，累加，释放 B 的寄存器
    acc += tl.load(C_ptr + offs)
    acc += tl.load(D_ptr + offs)
    acc += tl.load(E_ptr + offs)

    # 任何时候最多只有 1 个额外的寄存器变量 + acc
    # 总寄存器使用 ≈ BLOCK + BLOCK ≈ 2 × 128 = 256（刚好够用）

    tl.store(E_ptr + offs, acc)
```

### 14.8.6 内存优化 Checklist

```
内存优化 Checklist（按优先级排序）:

  □ 1. 合并访问 (Coalesced Access)
    - 确保 tl.arange 映射到线程 ID 使得相邻线程访问相邻地址
    - 检查 blocked layout 的 order 字段，连续维度应为 order[0]

  □ 2. Shared Memory 合理使用
    - 确保 SMEM 使用不会过度降低 occupancy
    - 使用 autotuning 搜索最优 BLOCK_M/N/K 组合

  □ 3. 异步流水线
    - 使用 num_stages ≥ 2 实现计算与访存重叠
    - 通过 autotuning 确定最优 stage 数

  □ 4. L2 Cache 友好
    - 使用 eviction_policy="evict_last" 加载会被复用的数据
    - 优化 program 调度顺序以增加 L2 命中率

  □ 5. 避免 Bank Conflict
    - 信任 Triton 编译器的自动 swizzle
    - 使用 triton-prof 查看 SMEM 带宽利用率

  □ 6. 控制寄存器使用
    - 减少同时活跃的变量数量
    - 使用分阶段计算模式
    - 检查编译器输出的寄存器使用信息

  □ 7. 数据类型选择
    - 使用 fp16/bf16 减少一半的内存带宽需求
    - 使用 fp8 进一步减少带宽（需要 H100+）

  □ 8. 分块策略优化
    - BLOCK_M/N/K 的选择应使 tile 能充分利用计算单元
    - 参考 NVIDIA 的最佳实践:
      A100: BLOCK_M=128, BLOCK_N=256, BLOCK_K=32 (fp16)
      H100: BLOCK_M=128, BLOCK_N=256, BLOCK_K=64 (fp16)
```

### 14.8.7 内存性能分析工具

```python
# 使用 Triton 的内置性能分析功能

# 方法 1: 查看编译后的 PTX（检查是否使用了 cp.async）
import triton
import triton.language as tl

@triton.jit
def my_kernel(...):
    ...

# 编译并打印 PTX
src = triton.compiler.ASTSource(fn=my_kernel, signature={...})
compiled = triton.compile(src)
print(compiled.asm["ptx"])  # 查看 PTX 中的 ld.global / cp.async 指令

# 方法 2: 使用 ncu (Nsight Compute) 分析内存瓶颈
# ncu --metrics \
#   l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,\
#   lts__t_bytes.sum.per_second,\
#   l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
#   smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
#   python my_script.py

# 关键指标:
#   global_load_throughput: 接近 2 TB/s → 合并访问良好
#   smem_throughput: 接近理论带宽 → 无 bank conflict
#   l2_hit_rate: > 50% → L2 缓存有效
#   register_spills: = 0 → 无寄存器溢出
```

<div data-component="MemoryOptimizationChecklist"></div>

[组件：MemoryOptimizationChecklist - 交互式内存优化检查工具，逐项检查 kernel 的内存访问模式并给出优化建议]

---

## 14.9 实战案例：Matmul Kernel 的内存访问分析

### 14.9.1 完整的内存访问流程

让我们以一个完整的 GEMM kernel 为例，分析其内存访问的每一个阶段：

```python
@triton.jit
def matmul_full_analysis(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # === 阶段 0: Program 调度 ===
    # 使用 group-based 调度优化 L2 cache 命中率
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # === 阶段 1: 计算地址偏移 ===
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # === 阶段 2: 初始化累加器（寄存器）===
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # 寄存器使用: BLOCK_M × BLOCK_N × 4 bytes (fp32)

    # === 阶段 3: K 维度循环 ===
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K

        # 步骤 3a: 从 HBM 加载 A 分块
        # 编译器可能生成:
        #   1) cp.async A → SMEM_A  (如果 pipeline)
        #   2) 或 tt.load A → REG  (如果不 pipeline)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k_offset + offs_k[None, :]) * stride_ak
        a_mask = (offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        # 内存访问: HBM → L1/L2 → Register (或 SMEM)
        # 合并检查: 相邻线程沿 offs_k (列方向) 访问相邻地址
        #          → 如果 stride_ak = sizeof(fp16) = 2B → 完全合并

        # 步骤 3b: 从 HBM 加载 B 分块
        b_ptrs = B_ptr + (k_offset + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        b_mask = ((k_offset + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        # 内存访问: 同上
        # 合并检查: 相邻线程沿 offs_n (列方向) 访问相邻地址
        #          → 如果 stride_bn = sizeof(fp16) = 2B → 完全合并

        # 步骤 3c: Shared memory 操作 (编译器自动插入)
        # a_smem = local_alloc(a)  → 寄存器 → shared memory
        # b_smem = local_alloc(b)  → 寄存器 → shared memory
        # a_reg  = local_load(a_smem)  → shared memory → 寄存器 (ldmatrix)
        # b_reg  = local_load(b_smem)  → shared memory → 寄存器 (ldmatrix)

        # 步骤 3d: Tensor Core 计算
        accumulator = tl.dot(a, b, accumulator)
        # → mma.sync.aligned.m16n8k16 指令
        # 输入: 寄存器中的 a_reg 和 b_reg
        # 输出: 累加到 accumulator 寄存器

    # === 阶段 4: 存储结果 ===
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
    # 内存访问: Register → HBM
    # 合并检查: 相邻线程沿 offs_n (列方向) 访问相邻地址 → 完全合并
```

### 14.9.2 每次循环迭代的内存访问计数

```
单次 K-loop 迭代的内存访问分析:
  参数: BLOCK_M=128, BLOCK_N=256, BLOCK_K=32, fp16

  ┌──────────────────────────────────────────────────────────────────┐
  │  操作                方向      数据量        访存类型           │
  ├──────────────────────────────────────────────────────────────────┤
  │  load A (128×32)    HBM→Reg   128×32×2B=8KB  global load       │
  │  load B (32×256)    HBM→Reg   32×256×2B=16KB global load       │
  │  local_alloc A      Reg→SMEM  8KB            st.shared         │
  │  local_alloc B      Reg→SMEM  16KB           st.shared         │
  │  local_load A       SMEM→Reg  8KB            ldmatrix          │
  │  local_load B       SMEM→Reg  16KB           ldmatrix          │
  │  dot (128×256)      Reg→TC    无额外访存     mma.sync          │
  ├──────────────────────────────────────────────────────────────────┤
  │  HBM 总读取:        8KB + 16KB = 24KB                          │
  │  SMEM 总写入:       8KB + 16KB = 24KB                          │
  │  SMEM 总读取:       8KB + 16KB = 24KB                          │
  │  FLOPs:             2 × 128 × 256 × 32 = 2,097,152             │
  │  计算强度 (HBM):    2M / 24K ≈ 87 FLOPs/Byte → memory-bound   │
  │  计算强度 (SMEM):   2M / 24K ≈ 87 FLOPs/Byte                  │
  └──────────────────────────────────────────────────────────────────┘

  完整 GEMM (K=1024, 32 次循环迭代):
    HBM 总读取: 32 × 24KB = 768 KB (A 和 B 各加载一次)
    HBM 写入:   128 × 256 × 2B = 64 KB (C 矩阵)
    FLOPs:      2 × 128 × 256 × 1024 = 67,108,864
    计算强度:   67M / (768K + 64K) ≈ 81 FLOPs/Byte
    → 仍然 memory-bound (81 < 156 for A100 FP16)
    → 需要更大的 BLOCK_M/N 或多次复用来提高计算强度
```

### 14.9.3 优化后的内存访问模式

```
优化策略的效果:

  原始配置: BLOCK_M=128, BLOCK_N=256, BLOCK_K=32
    每个 program 的 HBM 读取: 24KB/iter × 32 iter = 768 KB
    计算强度: 81 FLOPs/Byte

  优化 1: 增大 BLOCK_K 到 64
    每个 program 的 HBM 读取: (128×64 + 64×256) × 2B = 48KB/iter × 16 iter = 768 KB
    HBM 读取不变！但:
    - SMEM 使用翻倍: (8K + 32K) × 2 = 80KB (num_stages=2)
    - 可能降低 occupancy
    → 需要平衡

  优化 2: 使用 GROUP_SIZE_M 增加 L2 命中率
    GROUP_SIZE_M=8 → 同组内的 program 访问相邻的 M 行
    → A 矩阵的同一行被组内 8 个 program 复用
    → L2 命中率从 ~12.5% 提升到 ~100%（对于 A 矩阵）
    → 等效 HBM 读取降低到 ~1/8

  优化 3: num_stages=2 (双缓冲)
    - 计算与访存重叠
    - 带宽利用率提升（隐藏了部分延迟）
    → 实际吞吐接近理论峰值
```

---

## 14.10 高级话题：Warp Specialization 与 TMA

### 14.10.1 Warp Specialization 概述

在 Hopper 架构（SM 9.0, H100）中，NVIDIA 引入了 **warp specialization** 模式，允许不同的 warp 承担不同的角色：

```
Warp Specialization 模式:

  传统模式（所有 warp 执行相同代码）:
  ┌─────────────────────────────────────────────┐
  │  Warp 0: [load] [compute] [store]           │
  │  Warp 1: [load] [compute] [store]           │
  │  Warp 2: [load] [compute] [store]           │
  │  Warp 3: [load] [compute] [store]           │
  │                                              │
  │  问题: load 和 compute 不能完全重叠         │
  │        每个 warp 都需要寄存器保存 load 数据  │
  └─────────────────────────────────────────────┘

  Warp Specialization 模式:
  ┌─────────────────────────────────────────────┐
  │  Warp 0 (Producer): [load] [load] [load]    │  ← 专职加载
  │  Warp 1 (Producer): [load] [load] [load]    │
  │  Warp 2 (Consumer): [compute] [compute]     │  ← 专职计算
  │  Warp 3 (Consumer): [compute] [compute]     │
  │                                              │
  │  优势: Producer warp 专职访存               │
  │        Consumer warp 专职计算               │
  │        通过 barrier 同步                     │
  │        → 完全重叠，更高 occupancy            │
  └─────────────────────────────────────────────┘
```

### 14.10.2 TMA (Tensor Memory Accelerator)

Hopper 架构引入了 TMA 硬件单元，专门用于高效的数据搬运：

```
TMA vs 传统 cp.async:

  传统 cp.async:
    - 每个线程负责计算自己的地址
    - 32 个线程协作完成一个 128-byte 传输
    - 地址计算消耗 ALU 资源

  TMA (Tensor Memory Accelerator):
    - 硬件专用引擎，程序员只提供 base address + 描述符
    - 自动处理边界检查、swizzle、多维索引
    - 支持 1D/2D/3D/4D/5D tensor 传输
    - 释放 SM 的计算资源用于真正的计算

  TMA 在 Triton 中的表示:
  // TMA 描述符（包含 tensor 的形状、步长等元信息）
  %desc = triton_nvidia_gpu.create_tma_descriptor %ptr, [%shape], [%strides]

  // 使用 TMA 进行异步传输
  triton_nvidia_gpu.async_tma_copy_global_to_local %desc, %smem, [%coords]
  triton_nvidia_gpu.async_tma_commit_group
  triton_nvidia_gpu.async_tma_wait_group
```

---

## 本章小结

| 概念 | 核心要点 |
|---|---|
| **GPU 内存层次** | HBM(2TB/s) → L2(5TB/s) → SMEM(19TB/s) → Register(80TB/s)；延迟从 400 cycles 降到 1 cycle |
| **Triton 内存抽象** | `tl.load` 自动管理；`tl.dot` 自动插入 SMEM 操作；程序员无需手动管理 shared memory |
| **Memory Coalescing** | 相邻线程访问相邻地址 → 1 次 128B 访问；不合并可导致 32x 性能下降 |
| **Shared Memory 操作** | `local_alloc`(Reg→SMEM) → `local_load`(SMEM→Reg)；layout 变换通过 SMEM 中转 |
| **Bank Conflict** | 32 个 bank 交织；同 bank 不同地址访问被序列化；Triton 使用 swizzle 自动规避 |
| **Async Copy** | `cp.async` 跳过寄存器直接 HBM→SMEM；`num_stages` 控制流水线深度 |
| **Eviction Policy** | `evict_last` 保留权重在 L2；`evict_last` 适合重复访问数据 |
| **优化原则** | 合并访问 > SMEM 合理使用 > 异步流水线 > L2 友好 > 避免 register spilling |

```
内存优化优先级金字塔:

                    ┌──────────────┐
                    │  避免 reg    │
                    │  spilling    │
                  ┌─┴──────────────┴─┐
                  │  L2 cache 友好   │
                  │  + eviction hint │
                ┌─┴──────────────────┴─┐
                │  Async pipeline      │
                │  (num_stages ≥ 2)    │
              ┌─┴──────────────────────┴─┐
              │  Shared Memory 合理使用   │
              │  (SMEM 大小 vs occupancy)│
            ┌─┴──────────────────────────┴─┐
            │  Memory Coalescing            │
            │  (必须首先满足！)             │
            └──────────────────────────────┘
```

---

## 思考题

1. **内存层次理解**：在 A100 上运行一个 element-wise 加法 kernel（`C = A + B`，4096×4096 fp32），该 kernel 是 compute-bound 还是 memory-bound？理论最大吞吐是多少？

2. **合并访问分析**：给定一个行优先存储的 1024×1024 fp32 矩阵，以下两种加载模式哪种更高效？为什么？
   - 模式 A: 每个 warp 加载一行中的 32 个连续元素
   - 模式 B: 每个 warp 加载一列中的 32 个元素（stride=1024）

3. **Bank Conflict 诊断**：在 shared memory 中存储一个 32×32 的 fp32 矩阵，如果 warp 中的 32 个线程同时加载同一列的不同行元素，会发生什么？如何通过 padding 解决？

4. **流水线深度选择**：为什么 num_stages=2 通常是 GEMM kernel 的最优选择？什么情况下 num_stages=3 会更好？

5. **Eviction Policy 应用**：在一个 Transformer 的 self-attention kernel 中，Q、K、V 三个矩阵的 eviction_policy 应该如何设置？为什么？

6. **寄存器溢出识别**：编写一个简单的 Triton kernel，故意触发寄存器溢出，然后通过优化消除溢出。使用 `triton.compile` 查看 PTX 输出验证。

7. **Shared Memory 大小估算**：对于 BLOCK_M=256, BLOCK_N=128, BLOCK_K=64, fp16, num_stages=3 的 GEMM kernel，估算需要多少 shared memory？在 A100 上能达到的 occupancy 是多少？

8. **综合优化**：设计一个 Triton kernel，计算矩阵的逐行 softmax。分析其内存访问模式，并应用本章学到的所有优化技术（合并访问、SMEM 使用、流水线等）来最大化性能。
