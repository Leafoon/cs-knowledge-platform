---
title: "Chapter 17: AMD GPU 后端——ROCm/HIP 适配"
description: "深入解析 TileLang 在 AMD GPU 上的后端实现，涵盖 ROCm 生态、HIP 编程模型、CDNA 架构、MFMA 指令集、MI300X 硬件特性及性能调优方法"
updated: 2026-06-11
---

> **Learning Objectives**
>
> - 理解 ROCm 生态系统的版本演进、核心组件与安装部署流程
> - 掌握 HIP 编程模型及其与 CUDA 的对应关系
> - 深入理解 CDNA 架构（CDNA1/2/3）的硬件特性与执行模型
> - 掌握 Matrix Core（MFMA）指令集及其在 TileLang 中的映射方式
> - 了解 MI300X 的统一内存架构、HBM3 与 Infinity Cache 特性
> - 理解 AMD GPU 的 Shared Memory/LDS 管理机制
> - 对比 AMD 与 NVIDIA 内存模型的关键差异
> - 掌握 TileLang AMD 后端的 IR → HIP 代码生成架构
> - 理解 Wavefront 调度策略（Wave32/Wave64）
> - 学会在 MI300X 上实现 GEMM 和 FlashAttention
> - 掌握性能调优工具链（rocm-smi、rocprof、omniperf）
> - 能够排查 AMD GPU 后端的常见问题

---

## 17.1 ROCm 生态概述

### 17.1.1 ROCm 版本演进

ROCm（Radeon Open Compute）是 AMD 的开源 GPU 计算平台，从 2016 年首次发布至今已经历了多个重大版本迭代。

| 版本 | 发布时间 | 关键特性 | 支持架构 |
|------|----------|----------|----------|
| ROCm 4.0 | 2021-03 | MI100 支持、MIOpen 2.0 | CDNA1 |
| ROCm 5.0 | 2022-02 | MI200 系列支持 | CDNA1/CDNA2 |
| ROCm 5.5 | 2022-12 | 性能优化、编译器改进 | CDNA1/CDNA2 |
| ROCm 6.0 | 2024-01 | MI300X 支持、Flash Attention | CDNA1/CDNA2/CDNA3 |
| ROCm 6.1 | 2024-04 | MI300A APU 支持 | CDNA1/CDNA2/CDNA3 |
| ROCm 6.2 | 2024-08 | 性能增强、稳定性改进 | CDNA1/CDna2/CDNA3 |
| ROCm 6.3 | 2024-11 | 编译器优化、新库支持 | CDNA1/CDNA2/CDNA3 |

> [!TIP]
> 始终使用最新的 ROCm 版本以获得最佳的 TileLang 兼容性和性能。TileLang 的 AMD 后端主要针对 ROCm 6.0+ 进行优化。

### 17.1.2 ROCm 核心组件

ROCm 平台由多个层次的组件构成，从底层驱动到高层应用框架形成完整的技术栈。

```
┌─────────────────────────────────────────────────────┐
│                   Application Layer                  │
│   PyTorch (ROCm)  │  TileLang  │  Other Frameworks   │
├─────────────────────────────────────────────────────┤
│                   Library Layer                      │
│  rocBLAS │ MIOpen │ RCCL │ rocRAND │ hipBLASLt       │
├─────────────────────────────────────────────────────┤
│                   Compiler Layer                     │
│     HIP Compiler  │  rocMLIR  │  AOMP (LLVM-based)   │
├─────────────────────────────────────────────────────┤
│                   Runtime Layer                      │
│           HIP Runtime  │  ROCr (KFD)                 │
├─────────────────────────────────────────────────────┤
│                   Driver Layer                       │
│              amdgpu Kernel Driver                     │
├─────────────────────────────────────────────────────┤
│                   Hardware Layer                     │
│         AMD Instinct MI100/MI200/MI300 Series        │
└─────────────────────────────────────────────────────┘
```

**核心组件详解：**

| 组件 | 功能 | 说明 |
|------|------|------|
| HIP Runtime | GPU 编程运行时 | 类似 CUDA Runtime API |
| rocBLAS | 线性代数库 | 类似 cuBLAS |
| MIOpen | 深度学习库 | 类似 cuDNN |
RCCL | 集合通信库 | 类似 NCCL |
| rocMLIR | MLIR 编译器 | 用于机器学习优化 |
| rocprof | 性能分析工具 | 类似 nvprof/ncu |
| rocm-smi | 系统管理接口 | 类似 nvidia-smi |

### 17.1.3 ROCm 安装部署

**Ubuntu 系统安装：**

```bash
# 添加 ROCm 仓库
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.2/ jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list

# 安装 ROCm
sudo apt update
sudo apt install rocm-hip-sdk rocm-libs rocm-dev

# 配置环境变量
echo 'export PATH=$PATH:/opt/rocm/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib' >> ~/.bashrc
source ~/.bashrc

# 验证安装
rocm-smi
hipcc --version
```

**Docker 安装：**

```bash
# 拉取 ROCm Docker 镜像
docker pull rocm/rocm-terminal:6.2

# 运行容器（需要特权模式访问 GPU）
docker run -it --privileged --device=/dev/kfd --device=/dev/dri \
    --group-add video --shm-size=16G \
    rocm/rocm-terminal:6.2

# 在容器内验证
rocm-smi
```

> [!WARNING]
> 确保您的 AMD GPU 驱动版本与 ROCm 版本兼容。使用 `rocm-smi` 检查 GPU 状态和驱动版本。

**TileLang 与 ROCm 集成：**

```bash
# 安装支持 ROCm 的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# 安装 TileLang（AMD 后端支持）
pip install tilelang
# 或从源码编译
git clone https://github.com/tile-ai/tilelang.git
cd tilelang
pip install -e ".[amd]"
```

这段命令展示了 17.1.3 ROCm 安装部署 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

---

## 17.2 HIP 编程模型

### 17.2.1 HIP 概述

HIP（Heterogeneous-Compute Interface for Portability）是 AMD 的 GPU 编程接口，设计上与 CUDA 高度兼容。TileLang 的 AMD 后端正是通过生成 HIP 代码来实现对 AMD GPU 的支持。

```cpp
// HIP 核函数示例
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;
    float *d_a, *d_b, *d_c;

    // 分配设备内存
    hipMalloc(&d_a, N * sizeof(float));
    hipMalloc(&d_b, N * sizeof(float));
    hipMalloc(&d_c, N * sizeof(float));

    // 启动核函数
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(vector_add,
                       dim3(gridSize), dim3(blockSize), 0, 0,
                       d_a, d_b, d_c, N);

    // 同步
    hipDeviceSynchronize();

    // 释放内存
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    return 0;
}
```

这段示例展示了 HIP 的基本工作流：先在设备侧分配内存，再发射核函数，最后同步并释放资源。它和 CUDA 的调用方式非常接近，目的就是降低从 CUDA 迁移到 AMD 平台的门槛。代码里的 `gridDim` 和 `blockDim` 会直接映射到 GPU 的网格与线程块调度，因此分块大小会影响并行度和吞吐。实际使用时要特别注意内存分配和拷贝次数，否则很容易让数据搬运掩盖计算性能。

### 17.2.2 HIP 与 CUDA API 对应关系

| CUDA API | HIP API | 功能 |
|----------|---------|------|
| `cudaMalloc` | `hipMalloc` | 设备内存分配 |
| `cudaMemcpy` | `hipMemcpy` | 内存拷贝 |
| `cudaFree` | `hipFree` | 设备内存释放 |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | 设备同步 |
| `cudaGetLastError` | `hipGetLastError` | 获取错误 |
| `cudaStreamCreate` | `hipStreamCreate` | 创建流 |
| `cudaEventCreate` | `hipEventCreate` | 创建事件 |
| `__shared__` | `__shared__` | 共享内存声明 |
| `__syncthreads` | `__syncthreads` | 块内同步 |
| `__shfl_sync` | `__shfl_sync` | Warp Shuffle |
| `cudaFuncGetAttributes` | `hipFuncGetAttributes` | 获取核函数属性 |

### 17.2.3 HIP 内存管理

这张对照表说明了 HIP 选择“语义对齐 CUDA”的设计思路，便于现有 CUDA 代码快速迁移到 AMD GPU。对 TileLang 而言，这意味着后端在生成代码时可以复用大量 CUDA 时代的编程习惯，只需要处理少数平台差异。像 `hipMalloc`、`hipMemcpy` 这样的 API 直接对应设备内存和主机-设备拷贝，是性能调优的第一入口。需要注意的是，API 名称虽然一致，但底层调度和波前大小不同，不能把性能经验完全照搬。

```cpp
// HIP 内存管理示例
#include <hip/hip_runtime.h>

class HIPMemoryManager {
public:
    static void* allocate(size_t size) {
        void* ptr = nullptr;
        hipError_t err = hipMalloc(&ptr, size);
        if (err != hipSuccess) {
            throw std::runtime_error("hipMalloc failed: " +
                std::string(hipGetErrorString(err)));
        }
        return ptr;
    }

    static void deallocate(void* ptr) {
        if (ptr) {
            hipFree(ptr);
        }
    }

    static void copyHtoD(void* dst, const void* src, size_t size) {
        hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
    }

    static void copyDtoH(void* dst, const void* src, size_t size) {
        hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
    }

    static void copyDtoD(void* dst, const void* src, size_t size) {
        hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
    }
};
```

这个封装把分配、释放和拷贝统一收口，便于在 TileLang 后端里做资源管理和错误检查。它映射的是 ROCm 的显存分配器和 DMA 拷贝路径，因此对大张量场景很关键。性能上，显式区分 HtoD、DtoH 和 DtoD 可以帮助判断瓶颈是在输入搬运还是结果回写。常见坑是拷贝方向写错或异常路径遗漏 `hipFree`，这会直接造成错误结果或显存泄漏。

### 17.2.4 HIP 内置变量与线程层次

```cpp
// HIP 线程层次结构
__global__ void kernel() {
    // 网格级维度
    int gridDim_x = gridDim.x;      // 网格中块的数量
    int gridDim_y = gridDim.y;

    // 块级维度
    int blockDim_x = blockDim.x;    // 块中线程的数量
    int blockDim_y = blockDim.y;
    int blockDim_z = blockDim.z;

    // 线程索引
    int tid_x = threadIdx.x;        // 块内线程索引
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    // 块索引
    int bid_x = blockIdx.x;         // 网格内块索引
    int bid_y = blockIdx.y;

    // 全局线程索引
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Warp 相关
    int warpSize = 64;              // AMD GPU warp 大小（wavefront）
    int laneId = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
}
```

这段代码把 HIP 的线程层次拆成了网格、块、线程和 wavefront 四个层面，方便理解 TileLang 该如何做索引映射。AMD 上默认按 64 线程的 wavefront 执行，所以 `warpSize` 在这里本质上就是 wavefront 大小。这样分层的好处是可以把程序逻辑和硬件执行单元一一对应，减少在编译器 lowering 时的歧义。要注意 `threadIdx.x % 64` 这样的写法只适合 Wave64 语义，若切到 Wave32 就必须同步修改。

> [!CAUTION]
> AMD GPU 的默认 wavefront 大小为 64（Wave64），而 NVIDIA 的 warp 大小为 32。这是从 CUDA 迁移到 HIP 时最容易忽视的差异之一。

---

## 17.3 CDNA 架构

### 17.3.1 CDNA 架构演进

CDNA（Core DNA）是 AMD 面向数据中心和高性能计算的 GPU 架构系列。

```
┌─────────────────────────────────────────────────────────────────┐
│                    CDNA Architecture Evolution                    │
├─────────────┬─────────────┬─────────────┬───────────────────────┤
│   CDNA 1    │   CDNA 2    │   CDNA 3    │      Key Feature      │
├─────────────┼─────────────┼─────────────┼───────────────────────┤
│  MI100      │  MI200      │  MI300X     │  Product Line         │
│  120 CUs    │  220 CUs    │  304 CUs    │  Compute Units        │
│  HBM2      │  HBM2e      │  HBM3       │  Memory Type          │
│  32GB      │  96GB       │  192GB      │  Memory Capacity      │
│  1.2TB/s   │  3.2TB/s    │  5.3TB/s    │  Memory Bandwidth     │
│  Wave32/64 │  Wave32/64  │  Wave32/64  │  Execution Model      │
│  MFMA      │  MFMA       │  MFMA+      │  Matrix Operations    │
└─────────────┴─────────────┴─────────────┴───────────────────────┘
```

这张演进图强调了 CDNA 从 MI100 到 MI300X 的连续升级路径，核心变化集中在计算单元数量、HBM 容量和带宽。对编译器来说，这意味着同样的 TileLang 代码在不同代架构上可以共享大部分逻辑，但 tile 大小和流水线深度要随硬件资源调整。表里的 Wave32/64 和 MFMA+ 说明 CDNA3 不只是更快，还扩展了矩阵指令和执行模式。实际优化时应优先关注带宽与并行度的匹配，否则峰值算力再高也会被内存瓶颈限制。

### 17.3.2 CDNA 计算单元（CU）架构

每个 CDNA Compute Unit 包含以下核心组件：

```
┌──────────────────────────────────────────────────┐
│              CDNA Compute Unit                     │
├──────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐                │
│  │  Scalar Unit │  │  Vector Unit │  (SIMD)       │
│  │  (SP/DP)     │  │  (FP32/FP16) │               │
│  └─────────────┘  └─────────────┘                │
│  ┌─────────────┐  ┌─────────────┐                │
│  │  Matrix Core │  │  LDS Memory  │  (Local Data  │
│  │  (MFMA)      │  │  (64KB+)     │   Store)      │
│  └─────────────┘  └─────────────┘                │
│  ┌─────────────┐  ┌─────────────┐                │
│  │  VGPR File   │  │  SGPR File   │               │
│  │  (Vector)    │  │  (Scalar)    │               │
│  └─────────────┘  └─────────────┘                │
│  ┌─────────────────────────────────┐              │
│  │         Texture Unit            │              │
│  └─────────────────────────────────┘              │
└──────────────────────────────────────────────────┘
```

**CDNA3（MI300X）CU 增强：**

```cpp
// CDNA3 特有的硬件特性
struct CDNA3Features {
    // 每个 CU 的资源
    int vector_registers = 512;     // VGPR 数量（每线程）
    int scalar_registers = 102;    // SGPR 数量
    int lds_size_kb = 64;          // LDS 大小（KB）
    int wavefront_size = 64;       // Wavefront 大小

    // Matrix Core 特性
    int matrix_core_count = 4;     // 每 CU 的 Matrix Core 数量
    bool supports_fp8 = true;      // FP8 矩阵运算支持
    bool supports_bf16 = true;     // BF16 矩阵运算支持

    // 缓存层次
    int l1_cache_kb = 16;          // L1 缓存大小
    int l2_cache_per_shader = 2;   // 每 Shader Engine 的 L2（MB）

    // 统一内存
    bool unified_memory = true;    // CPU/GPU 统一内存支持
    int hbm_capacity_gb = 192;     // HBM3 容量
    float hbm_bandwidth_tbs = 5.3f; // HBM3 带宽（TB/s）
};
```

这个结构体把 CDNA3 的关键资源抽象成编译器可用的参数，方便 TileLang 在生成代码时做静态决策。VGPR、SGPR、LDS 和矩阵核心数量共同决定了块大小、寄存器压力和 occupancy，因此不能只看算力峰值。HBM3 与统一内存是 MI300X 的重要特征，适合高带宽、低拷贝的工作负载。使用时要避免把 CDNA2 的经验直接套到 CDNA3，尤其是 FP8 和更高并发资源的组合策略。

### 17.3.3 CDNA 执行模型

```cpp
// CDNA Wavefront 执行模型
__global__ void cdna_execution_example() {
    // 每个 wavefront 包含 64 个线程（默认）
    // 在 CDNA3 上也可以使用 Wave32 模式

    __shared__ float smem[256];

    int tid = threadIdx.x;
    int warpId = tid / 64;   // CDNA 默认 wavefront 大小为 64
    int laneId = tid % 64;

    // Wavefront 级别的操作
    float val = static_cast<float>(tid);

    // Wavefront 内 shuffle（类似 __shfl_sync）
    float shuffled = __shfl(val, 0);  // 从 lane 0 广播

    // Wavefront 内归约
    float sum = __shfl_xor(val, 1);
    sum = __shfl_xor(sum, 2);
    sum = __shfl_xor(sum, 4);
    sum = __shfl_xor(sum, 8);
    sum = __shfl_xor(sum, 16);
    sum = __shfl_xor(sum, 32);

    // 写入共享内存
    smem[tid] = sum;
    __syncthreads();
}
```

这段示例说明了 CDNA 的执行单位是 wavefront，而不是单个线程孤立运行。`__shfl` 和 `__shfl_xor` 这类操作会直接映射到 wavefront 内的数据交换路径，适合做归约和广播。对性能来说，wavefront 级别操作通常比写回 LDS 再同步更快，但前提是数据依赖严格局限在同一个 wave 内。需要注意的是，Wave64 下的归约树和 Wave32 不同，手写代码时不能默认 lane 数固定。

---

## 17.4 Matrix Core（MFMA）指令集映射

### 17.4.1 MFMA 指令概述

MFMA（Matrix Fused Multiply-Add）是 AMD GPU 上的矩阵运算指令，类似于 NVIDIA 的 Tensor Core。

<div data-component="MFMAInstructionMap">

| 指令 | 数据类型 | M×N×K | 吞吐量说明 |
|------|----------|-------|------------|
| `v_mfma_f32_32x32x2bf16` | BF16 | 32×32×2 | 基础 BF16 矩阵乘 |
| `v_mfma_f32_16x16x4bf16` | BF16 | 16×16×4 | 小规模 BF16 矩阵乘 |
| `v_mfma_f32_32x32x8fp16` | FP16 | 32×32×8 | FP16 矩阵乘 |
| `v_mfma_f32_16x16x16fp16` | FP16 | 16×16×16 | 小规模 FP16 矩阵乘 |
| `v_mfma_f32_32x32x4fp8` | FP8 | 32×32×4 | FP8 矩阵乘（CDNA3） |
| `v_mfma_f32_16x16x32fp8` | FP8 | 16×16×32 | 小规模 FP8 矩阵乘 |
| `v_mfma_f32_32x32x2i8` | INT8 | 32×32×2 | 整数矩阵乘 |
| `v_mfma_f32_4x4x4bf16` | BF16 | 4×4×4 | 最小规模矩阵乘 |

</div>

这张指令表把 MFMA 的数据类型、矩阵形状和吞吐特征直接对应起来，便于判断某个 GEMM 应该选哪一种实现。每个 wavefront 共同驱动一次矩阵乘累加，所以指令形状本质上决定了线程分工与寄存器布局。CDNA3 新增的 FP8 指令意味着 TileLang 可以为低精度推理生成更激进的矩阵路径，从而换取更高吞吐。常见问题是只看数据类型不看块大小，结果选择了形状不匹配的 MFMA，反而降低利用率。

> [!TIP]
> MFMA 指令的一个 wavefront（64 线程）共同计算一个矩阵乘累加操作。选择合适的 MFMA 指令形状对性能至关重要。

### 17.4.2 MFMA 指令在 TileLang 中的映射

```python
import tilelang
from tilelang import T

@tilelang.jit
def gemm_mfma_example(M, N, K, block_M, block_N, block_K, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor([M, K], dtype),
        B: T.Tensor([K, N], dtype),
        C: T.Tensor([M, N], "float32"),
    ):
        # 分块参数
        bx = T.ceildiv(N, block_N)
        by = T.ceildiv(M, block_M)

        # 共享内存声明
        A_shared = T.alloc_shared([block_M, block_K], dtype)
        B_shared = T.alloc_shared([block_K, block_N], dtype)

        # 累加器寄存器
        C_local = T.alloc_fragment([block_M, block_N], "float32")

        with T.Blocks([bx, by]) as (bx_idx, by_idx):
            T.clear(C_local)

            for k in T.serial(T.ceildiv(K, block_K)):
                # 从全局内存加载到共享内存
                T.copy(A[by_idx * block_M:, k * block_K:], A_shared)
                T.copy(B[k * block_K:, bx_idx * block_N:], B_shared)
                T.sync()

                # 使用 MFMA 进行矩阵乘
                T.gemm(A_shared, B_shared, C_local)

            # 写回全局内存
            T.copy(C_local, C[by_idx * block_M:, bx_idx * block_N:])

    return main
```

这段示例说明 TileLang 会把高层 GEMM 结构映射成共享内存搬运加 MFMA 计算的标准模板。`T.alloc_shared` 对应 LDS，`T.alloc_fragment` 对应寄存器累加器，前者负责喂数，后者负责保留中间结果。性能好坏主要取决于分块是否能让 LDS 复用和 MFMA 吞吐同时成立。容易出错的地方是 tile 维度和实际矩阵边界不一致，导致读写越界或同步点放置错误。

### 17.4.3 MFMA 指令选择策略

```python
# TileLang 中的 MFMA 指令选择
def select_mfma_instruction(dtype, block_m, block_n):
    """
    根据数据类型和块大小选择最优的 MFMA 指令
    """
    mfma_map = {
        "float16": {
            (32, 32): "v_mfma_f32_32x32x8fp16",
            (16, 16): "v_mfma_f32_16x16x16fp16",
            (4, 4):   "v_mfma_f32_4x4x4bf16",
        },
        "bfloat16": {
            (32, 32): "v_mfma_f32_32x32x2bf16",
            (16, 16): "v_mfma_f32_16x16x4bf16",
            (4, 4):   "v_mfma_f32_4x4x4bf16",
        },
        "float8_e4m3fnuz": {
            (32, 32): "v_mfma_f32_32x32x4fp8",
            (16, 16): "v_mfma_f32_16x16x32fp8",
        },
    }

    if dtype not in mfma_map:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # 选择最大的匹配形状
    dtype_map = mfma_map[dtype]
    for (m, n) in [(32, 32), (16, 16), (4, 4)]:
        if block_m >= m and block_n >= n:
            return dtype_map[(m, n)]

    raise ValueError(f"Block size too small: {block_m}x{block_n}")
```

这里的选择逻辑体现了“先按数据类型，再按块尺寸”的映射原则，因为不同 MFMA 形状对输入精度和 tile 尺寸都有硬约束。它的硬件含义是尽量让一个 wavefront 内的工作量与 MFMA 的矩阵形状对齐，减少空 lane 和碎片化执行。性能上，较大的 MFMA 通常更高效，但前提是块尺寸足够大且寄存器/LDS 没有被撑爆。需要警惕的是边界块和小矩阵场景，过大的形状会导致占用率下降甚至直接不匹配。

### 17.4.4 MFMA 执行细节

```cpp
// MFMA 指令的底层实现示例（内联汇编）
__device__ void mfma_f32_32x32x8f16(
    float* c,          // 输出矩阵 C (32x32)
    const half* a,     // 输入矩阵 A (32x8)
    const half* b      // 输入矩阵 B (8x32)
) {
    // 每个 wavefront (64 threads) 协作计算 32x32 的矩阵乘
    // 输入 A: 32x8，由 64 个线程分摊加载
    // 输入 B: 8x32，由 64 个线程分摊加载
    // 输出 C: 32x32，由 64 个线程分摊存储

    // 内联汇编调用 MFMA 指令
    asm volatile(
        "v_mfma_f32_32x32x8f16 %0, %1, %2, %3\n"
        : "=v"(c[0])
        : "v"(a[0]), "v"(b[0]), "v"(c[0])
    );
}
```

这段内联汇编展示了最终如何落到 AMD ISA 级别的 MFMA 指令，说明 TileLang 后端的终点不是“生成 HIP C++”，而是生成能被编译器识别的硬件矩阵指令。这里的输入、输出和累加器都按 wavefront 协作方式分摊，映射关系直接影响寄存器分配。性能上，正确使用 MFMA 可以显著高于标量或向量 ALU 的实现，但寄存器布局不当会让吞吐优势被搬运和重排开销吃掉。写这类代码最常见的坑是对寄存器和 lane 分工理解不足，导致汇编约束或数据布局错误。

---

## 17.5 MI300X 硬件特性

### 17.5.1 MI300X 架构概览

MI300X 是 AMD 最新一代数据中心 GPU，采用了革命性的 3D Chiplet 架构。

<div data-component="AMDGPUBackendArchitecture">

```
┌──────────────────────────────────────────────────────────────┐
│                     AMD Instinct MI300X                        │
├──────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐    │
│  │                    XCD (Accelerator Complex Die) ×8    │    │
│  │  ┌────────────────┐  ┌────────────────┐              │    │
│  │  │   Shader Engine │  │   Shader Engine │              │    │
│  │  │   (16 CUs)      │  │   (16 CUs)      │              │    │
│  │  │   ┌──────────┐  │  │   ┌──────────┐  │              │    │
│  │  │   │  L2 Cache │  │  │   │  L2 Cache │  │              │    │
│  │  │   └──────────┘  │  │   └──────────┘  │              │    │
│  │  └────────────────┘  └────────────────┘              │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              IOD (I/O Die) ×6                          │    │
│  │  ┌─────────────────────────────────────────────┐     │    │
│  │  │         HBM3 Memory Controllers              │     │    │
│  │  │    192GB HBM3  │  5.3 TB/s Bandwidth        │     │    │
│  │  └─────────────────────────────────────────────┘     │    │
│  │  ┌─────────────────────────────────────────────┐     │    │
│  │  │      Infinity Cache (256MB shared)            │     │    │
│  │  └─────────────────────────────────────────────┘     │    │
│  └──────────────────────────────────────────────────────┘    │
│  Total: 304 CUs  │  1.3 GHz Boost  │  192GB HBM3          │
└──────────────────────────────────────────────────────────────┘
```

这个示意块用于解释 17.5.1 MI300X 架构概览 的整体结构和数据流关系。阅读时不要只看框图中的模块名称，而要沿着箭头理解数据从输入、缓存、计算到输出的完整路径。这样的表达方式有助于把抽象概念和实际执行阶段对应起来，尤其适合分析内存层级、调度顺序和性能瓶颈。实际实现时需要注意图中省略的边界检查、同步开销和硬件限制，否则容易把概念流程误解为可以直接无成本执行的代码。

</div>

这张架构图把 MI300X 的 XCD、IOD、HBM3 控制器和 Infinity Cache 串成了一条完整的数据路径。对 TileLang 来说，它说明后端不仅要生成计算代码，还要考虑 tile 如何落在多 XCD 的并行结构上。192GB HBM3 和 5.3TB/s 带宽意味着大模型和大批量任务可以更多依赖本地显存而不是频繁拆分。真正的风险是把算子拆得过碎，导致调度和同步开销吞掉硬件优势。

**MI300X 关键规格：**

| 参数 | 数值 | 说明 |
|------|------|------|
| 计算单元 | 304 CUs | 8 XCD × 38 CUs |
| 矩阵核心 | 1216 | 每 CU 4 个 |
| HBM3 容量 | 192GB | 统一内存架构 |
| HBM3 带宽 | 5.3 TB/s | 8 个 HBM3 堆栈 |
| Infinity Cache | 256MB | 共享 L3 缓存 |
| FP32 吞吐 | 163.4 TFLOPS | 向量运算 |
| FP16 矩阵吞吐 | 2,607 TFLOPS | MFMA 运算 |
| FP8 矩阵吞吐 | 5,214 TFLOPS | MFMA 运算（CDNA3） |
| TDP | 750W | 典型功耗 |

### 17.5.2 统一内存架构

这张规格表把 MI300X 的核心优势量化出来，最直接的是 304 CUs、192GB HBM3 和极高带宽。它们对应到编程层面就是更大的并行容纳能力和更少的数据搬运压力，因此 TileLang 可以偏向更大 tile 和更深流水线。FP8 和 FP16 的超高矩阵吞吐说明这块卡非常适合矩阵密集型任务，但前提是输入布局和数据类型选择正确。需要注意功耗也很高，部署时必须把散热和供电当作性能边界的一部分。

MI300X 的统一内存架构是其最大亮点之一，CPU 和 GPU 可以直接访问同一块物理内存。

```cpp
// MI300X 统一内存使用示例
#include <hip/hip_runtime.h>

void unified_memory_example() {
    float* data;

    // 分配统一内存（CPU/GPU 共享）
    hipMallocManaged(&data, 1024 * 1024 * sizeof(float));

    // CPU 端初始化
    for (int i = 0; i < 1024 * 1024; i++) {
        data[i] = static_cast<float>(i);
    }

    // GPU 端计算（无需显式数据传输）
    compute_kernel<<<blocks, threads>>>(data, 1024 * 1024);
    hipDeviceSynchronize();

    // CPU 端读取结果
    float result = data[0];

    hipFree(data);
}
```

这个例子展示了 MI300X 统一内存让 CPU 和 GPU 能共享同一地址空间，从而简化数据管理。它在硬件上依赖统一的物理内存和更成熟的分页迁移机制，所以非常适合原型开发和减少样板代码。性能上，统一内存减少了显式拷贝，但不等于自动最快，热点数据最好仍然通过预取或显式管理控制驻留位置。常见坑是把统一内存当成“零成本”，结果在第一次访问时触发大规模页迁移。

> [!TIP]
> 在 MI300X 上使用统一内存可以简化编程模型，但为了最佳性能，建议使用显式的内存管理来控制数据位置。

### 17.5.3 HBM3 与 Infinity Cache

```cpp
// 利用 Infinity Cache 的数据布局策略
struct DataLayout {
    // 热数据：放在 Infinity Cache 友好的地址
    // 通过访问模式控制缓存行为

    // 连续访问模式（Cache 友好）
    void sequential_access(float* data, int n) {
        for (int i = 0; i < n; i += 16) {
            // 16 个连续 float = 64 字节（一个缓存行）
            process(data + i, 16);
        }
    }

    // 跨步访问模式（Cache 不友好）
    void strided_access(float* data, int n, int stride) {
        for (int i = 0; i < n; i += stride) {
            // 大跨步访问会降低缓存命中率
            process_single(data + i);
        }
    }
};
```

这段代码强调了 HBM3 提供容量，Infinity Cache 提供复用，两者在数据布局上需要协同设计。顺序访问能更容易命中缓存行，适合 tile 化后的连续块加载；跨步访问则会放大带宽压力并降低命中率。对 TileLang 而言，这意味着调度器应该尽量把计算块组织成连续访存模式，让片上缓存先吃掉重复读。要小心的是缓存友好布局往往依赖具体张量维度，一旦索引顺序不对，性能会急剧下降。

---

## 17.6 Shared Memory / LDS 管理

### 17.6.1 LDS 概述

LDS（Local Data Share）是 AMD GPU 上的共享内存，位于每个 Compute Unit 内部，供 CU 内的线程共享数据。

| 特性 | 说明 |
|------|------|
| 容量 | 每 CU 64KB（CDNA3） |
| 延迟 | ~20-30 cycles |
| 带宽 | 非常高（片上存储） |
| 访问模式 | 32 bank，4 字节宽度 |
| 同步 | `__syncthreads()` 或 barrier |

### 17.6.2 LDS Bank Conflict

```cpp
// LDS Bank Conflict 示例
__global__ void lds_bank_conflict_example() {
    __shared__ float smem[256];

    int tid = threadIdx.x;

    // 无 Bank Conflict：连续访问
    float val1 = smem[tid];  // 每个线程访问不同的 bank

    // 有 Bank Conflict：跨步访问
    float val2 = smem[tid * 2];  // 多个线程可能访问同一个 bank

    // 避免 Bank Conflict：填充
    __shared__ float smem_padded[256 + 16];  // 填充 16 个元素
    float val3 = smem_padded[tid * 2];       // 减少 bank conflict
}
```

这段代码服务于 17.6.2 LDS Bank Conflict 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 17.6.3 TileLang 中的 LDS 管理

```python
import tilelang
from tilelang import T

@tilelang.jit
def optimized_lds_usage(M, N, K, block_M=128, block_N=128, block_K=32):
    @T.prim_func
    def main(
        A: T.Tensor([M, K], "float16"),
        B: T.Tensor([K, N], "float16"),
        C: T.Tensor([M, N], "float32"),
    ):
        # 声明共享内存（映射到 LDS）
        A_shared = T.alloc_shared([block_M, block_K], "float16")
        B_shared = T.alloc_shared([block_K, block_N], "float16")

        # 寄存器文件中的累加器
        C_local = T.alloc_fragment([block_M, block_N], "float32")

        # 填充以避免 bank conflict
        A_shared_padded = T.alloc_shared([block_M, block_K + 4], "float16")

        with T.Blocks([T.ceildiv(N, block_N), T.ceildiv(M, block_M)]) as (bn, bm):
            T.clear(C_local)

            for k in T.serial(T.ceildiv(K, block_K)):
                # 高效的 LDS 加载
                T.copy(A[bm * block_M:, k * block_K:], A_shared)
                T.copy(B[k * block_K:, bn * block_N:], B_shared)
                T.sync()

                # 使用 MFMA 的矩阵乘
                T.gemm(A_shared, B_shared, C_local, k_axis=block_K)

            T.copy(C_local, C[bm * block_M:, bn * block_N:])

    return main
```

这段代码服务于 17.6.3 TileLang 中的 LDS 管理 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 17.7 AMD vs NVIDIA 内存模型差异

<div data-component="AMDvsNVIDIAComparison">

### 17.7.1 内存层次对比

| 层次 | NVIDIA (CUDA) | AMD (HIP) | 差异说明 |
|------|---------------|-----------|----------|
| 全局内存 | Global Memory | Global Memory (VRAM) | 概念相同 |
| L2 缓存 | L2 Cache | L2 + Infinity Cache | AMD 有额外的 Infinity Cache |
| 共享内存 | Shared Memory | LDS (Local Data Share) | 命名不同，功能类似 |
| 寄存器 | Register File | VGPR/SGPR | AMD 区分向量/标量寄存器 |
| 常量内存 | Constant Memory | Constant Memory | 概念相同 |
| 纹理内存 | Texture Memory | Texture Memory | 概念相同 |
| 统一内存 | Unified Memory | Unified Memory | MI300X 统一内存更成熟 |

### 17.7.2 执行模型差异

| 方面 | NVIDIA | AMD | 影响 |
|------|--------|-----|------|
| Warp 大小 | 32 | 64（默认） | 影响分支发散和内存访问模式 |
| Warp 调度 | Warp Scheduler | Wavefront Scheduler | 类似但细节不同 |
| 共享内存大小 | 48KB-164KB | 64KB (LDS) | AMD 固定大小 |
| Bank 数量 | 32 | 32 | 相同 |
| Bank 宽度 | 4 字节 | 4 字节 | 相同 |
| 最大线程/块 | 1024 | 1024 | 相同 |
| 寄存器/线程 | 255 | 512 (VGPR) | AMD 更多寄存器 |

### 17.7.3 编程模型差异

```cpp
// NVIDIA CUDA 风格
__global__ void cuda_style_kernel() {
    __shared__ float smem[256];
    int warpId = threadIdx.x / 32;    // Warp 大小 32
    int laneId = threadIdx.x % 32;
    __syncwarp();                       // Warp 级同步
}

// AMD HIP 风格
__global__ void hip_style_kernel() {
    __shared__ float smem[256];        // 映射到 LDS
    int waveId = threadIdx.x / 64;    // Wavefront 大小 64
    int laneId = threadIdx.x % 64;
    // Wavefront 级同步通常隐式
}
```

这段代码服务于 17.7.3 编程模型差异 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

</div>

---

## 17.8 TileLang AMD 后端架构

### 17.8.1 IR → HIP 代码生成流程

TileLang 的 AMD 后端通过多级 Lowering 将高层 IR 转换为 HIP 代码。

```
┌─────────────────────────────────────────────────────────────┐
│              TileLang AMD Backend Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐                                            │
│  │  Python DSL  │  用户编写的 TileLang 程序                   │
│  └──────┬──────┘                                            │
│         ▼                                                   │
│  ┌─────────────┐                                            │
│  │  TensorIR    │  TVM 的张量级中间表示                        │
│  └──────┬──────┘                                            │
│         ▼                                                   │
│  ┌─────────────┐                                            │
│  │  TileLang IR │  TileLang 特有的 Tile 级 IR                 │
│  │  (TIR)       │  包含内存管理、软件流水等                      │
│  └──────┬──────┘                                            │
│         ▼                                                   │
│  ┌─────────────┐                                            │
│  │  LLVM IR     │  通过 TVM 的 LLVM 后端                      │
│  └──────┬──────┘                                            │
│         ▼                                                   │
│  ┌─────────────┐                                            │
│  │  HIP Code    │  生成的 HIP 源代码或 HSACO 二进制             │
│  └──────┬──────┘                                            │
│         ▼                                                   │
│  ┌─────────────┐                                            │
│  │  ROCm Runtime│  编译并加载到 AMD GPU 执行                    │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

这个示意块用于解释 17.8.1 IR → HIP 代码生成流程 的整体结构和数据流关系。阅读时不要只看框图中的模块名称，而要沿着箭头理解数据从输入、缓存、计算到输出的完整路径。这样的表达方式有助于把抽象概念和实际执行阶段对应起来，尤其适合分析内存层级、调度顺序和性能瓶颈。实际实现时需要注意图中省略的边界检查、同步开销和硬件限制，否则容易把概念流程误解为可以直接无成本执行的代码。

### 17.8.2 后端代码结构

```python
# tilelang/backend/amd/hip_codegen.py (概念示例)
class HIPCodeGen:
    """TileLang AMD 后端的 HIP 代码生成器"""

    def __init__(self, target_arch="cdna3"):
        self.target_arch = target_arch
        self.wavefront_size = 64

    def generate_kernel(self, func, schedule):
        """将 TileLang IR 转换为 HIP 核函数"""

        # 1. 分析内存访问模式
        mem_info = self.analyze_memory_access(func)

        # 2. 选择 MFMA 指令
        mfma_instr = self.select_mfma(func, schedule)

        # 3. 生成 LDS 管理代码
        lds_code = self.generate_lds_management(func)

        # 4. 生成线程索引计算
        idx_code = self.generate_thread_indices(func)

        # 5. 生成核函数体
        body = self.generate_body(func, mfma_instr, lds_code)

        # 6. 包装为 HIP 核函数
        return self.wrap_as_hip_kernel(body, func.name)

    def select_mfma(self, func, schedule):
        """选择最优的 MFMA 指令"""
        dtype = func.get_output_dtype()
        block_m = schedule.get_param("block_M")
        block_n = schedule.get_param("block_N")

        # 根据数据类型和块大小选择
        if dtype == "float16":
            if block_m >= 32 and block_n >= 32:
                return "v_mfma_f32_32x32x8fp16"
            else:
                return "v_mfma_f32_16x16x16fp16"
        elif dtype == "bfloat16":
            if block_m >= 32 and block_n >= 32:
                return "v_mfma_f32_32x32x2bf16"
            else:
                return "v_mfma_f32_16x16x4bf16"

        raise ValueError(f"Unsupported dtype: {dtype}")
```

这段代码服务于 17.8.2 后端代码结构 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 17.8.3 目标架构适配

```python
# 后端配置
class AMDBackendConfig:
    """AMD GPU 后端配置"""

    # 支持的目标架构
    ARCH_MAP = {
        "gfx908": "cdna1",    # MI100
        "gfx90a": "cdna2",    # MI200 系列
        "gfx940": "cdna3",    # MI300 系列
        "gfx941": "cdna3",    # MI300A
        "gfx942": "cdna3",    # MI300X
    }

    def __init__(self, arch="gfx942"):
        self.arch = arch
        self.cdna_version = self.ARCH_MAP.get(arch, "cdna3")
        self.wavefront_size = 64

        # 根据架构设置参数
        if self.cdna_version == "cdna3":
            self.max_lds_size = 65536      # 64KB
            self.max_vgpr = 512
            self.max_sgpr = 102
            self.mfma_supports_fp8 = True
        elif self.cdna_version == "cdna2":
            self.max_lds_size = 65536
            self.max_vgpr = 512
            self.max_sgpr = 102
            self.mfma_supports_fp8 = False
        else:
            self.max_lds_size = 65536
            self.max_vgpr = 256
            self.max_sgpr = 102
            self.mfma_supports_fp8 = False

    def get_compile_flags(self):
        """获取编译选项"""
        flags = [
            f"--offload-arch={self.arch}",
            "-O3",
            "--std=c++17",
        ]
        if self.cdna_version == "cdna3":
            flags.append("-mattr=+mfma,+dot7,+dot8")
        return flags
```

这段代码服务于 17.8.3 目标架构适配 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 17.9 Wavefront 调度

### 17.9.1 Wave32 vs Wave64

AMD GPU 支持两种 wavefront 执行模式：

| 特性 | Wave32 | Wave64 |
|------|--------|--------|
| 线程数 | 32 | 64（默认） |
| 分支粒度 | 更细 | 更粗 |
| 寄存器效率 | 更高 | 更低 |
| 内存合并 | 更灵活 | 更严格 |
| 适用场景 | 细粒度并行 | 粗粒度并行 |

```cpp
// Wave32 模式核函数
__attribute__((amdgpu_wavesize(32)))
__global__ void wave32_kernel() {
    // 使用 32 线程的 wavefront
    int laneId = threadIdx.x % 32;

    // Wave32 特有的优化
    float val = __shfl_sync(0xFFFFFFFF, threadIdx.x, 0);
}

// Wave64 模式核函数（默认）
__global__ void wave64_kernel() {
    int laneId = threadIdx.x % 64;

    // Wave64 操作
    float val = __shfl(val, 0);
}
```

这段代码服务于 17.9.1 Wave32 vs Wave64 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 17.9.2 Wavefront 调度策略

```python
# TileLang 中的 wavefront 调度配置
def configure_wavefront_scheduling(target_arch, kernel_type):
    """
    配置 wavefront 调度策略
    """
    config = {
        "arch": target_arch,
        "wavefront_size": 64,  # 默认 Wave64
    }

    if kernel_type == "gemm":
        # GEMM 通常使用 Wave64 以获得最大吞吐
        config["wavefront_size"] = 64
        config["occupancy"] = "high"
    elif kernel_type == "attention":
        # Attention 可能受益于 Wave32 的更细粒度控制
        config["wavefront_size"] = 32
        config["occupancy"] = "medium"
    elif kernel_type == "elementwise":
        # 逐元素操作使用 Wave32 更灵活
        config["wavefront_size"] = 32
        config["occupancy"] = "high"

    return config
```

这段代码服务于 17.9.2 Wavefront 调度策略 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 17.9.3 Occupancy 优化

```python
def calculate_occupancy(arch, vgpr_per_thread, lds_per_block, waves_per_cu=1):
    """
    计算 AMD GPU 的 occupancy

    Args:
        arch: 目标架构（如 "gfx942"）
        vgpr_per_thread: 每线程 VGPR 数量
        lds_per_block: 每个 block 的 LDS 大小（字节）
        waves_per_cu: 每 CU 的 wavefront 数量目标

    Returns:
        occupancy: 占用率（0-1）
    """
    # CDNA3 资源限制
    max_vgpr = 512
    max_lds = 65536  # 64KB
    max_waves = 16   # 每 CU 最大 wavefront 数

    # 计算每个 wavefront 的 VGPR 使用
    wave_size = 64
    vgpr_per_wave = vgpr_per_thread * wave_size

    # 计算可同时运行的 wavefront 数
    waves_by_vgpr = max_vgpr // vgpr_per_wave
    waves_by_lds = max_lds // lds_per_block if lds_per_block > 0 else max_waves

    actual_waves = min(waves_by_vgpr, waves_by_lds, max_waves)
    occupancy = actual_waves / max_waves

    return occupancy
```

这段代码服务于 17.9.3 Occupancy 优化 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 17.10 GEMM / FlashAttention 在 MI300X 上的实现

### 17.10.1 MI300X GEMM 实现

```python
import tilelang
from tilelang import T
import torch

@tilelang.jit(
    out_idx=[2],
    target="hip",
    arch="gfx942",
)
def gemm_mi300x(
    M: int,
    N: int,
    K: int,
    block_M: int = 256,
    block_N: int = 256,
    block_K: int = 64,
    num_stages: int = 3,
    dtype: str = "float16",
):
    @T.prim_func
    def main(
        A: T.Tensor([M, K], dtype),
        B: T.Tensor([K, N], dtype),
        C: T.Tensor([M, N], "float32"),
    ):
        # 使用 MI300X 优化的块大小
        # 256x256 分块充分利用 304 个 CU
        with T.Blocks(
            [T.ceildiv(N, block_N), T.ceildiv(M, block_M)]
        ) as (bx, by):
            # 共享内存（LDS）
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            B_shared = T.alloc_shared([block_K, block_N], dtype)

            # 寄存器累加器
            C_local = T.alloc_fragment([block_M, block_N], "float32")

            # 软件流水线
            T.annotate_schedule({
                "software_pipeline_stage": [0, 0, 1, 1, 2],
            })

            T.clear(C_local)

            for k in T.serial(T.ceildiv(K, block_K)):
                # 异步加载到 LDS
                T.copy(A[by * block_M:, k * block_K:], A_shared)
                T.copy(B[k * block_K:, bx * block_N:], B_shared)
                T.sync()

                # MFMA 矩阵乘
                T.gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    k_axis=block_K,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            T.copy(C_local, C[by * block_M:, bx * block_N:])

    return main

# 使用示例
M, N, K = 4096, 4096, 4096
kernel = gemm_mi300x(M, N, K)
A = torch.randn(M, K, dtype=torch.float16, device="cuda")
B = torch.randn(K, N, dtype=torch.float16, device="cuda")
C = kernel(A, B)
```

这段代码服务于 17.10.1 MI300X GEMM 实现 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 17.10.2 MI300X FlashAttention 实现

```python
@tilelang.jit(
    out_idx=[2],
    target="hip",
    arch="gfx942",
)
def flash_attention_mi300x(
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
        # 常量
        scale = 1.0 / (head_dim ** 0.5)

        with T.Blocks(
            [batch_size, num_heads, T.ceildiv(seq_len, block_M)]
        ) as (bz, bh, bm):
            # 共享内存
            Q_shared = T.alloc_shared([block_M, head_dim], dtype)
            K_shared = T.alloc_shared([block_N, head_dim], dtype)
            V_shared = T.alloc_shared([block_N, head_dim], dtype)

            # 寄存器
            acc = T.alloc_fragment([block_M, head_dim], "float32")
            scores = T.alloc_fragment([block_M, block_N], "float32")
            log_sum = T.alloc_fragment([block_M], "float32")
            row_max = T.alloc_fragment([block_M], "float32")

            T.clear(acc)
            T.fill(log_sum, -float("inf"))
            T.fill(row_max, -float("inf"))

            # 加载 Q 块
            T.copy(Q[bz, bh, bm * block_M:, :], Q_shared)

            for bn in T.serial(T.ceildiv(seq_len, block_N)):
                # 加载 K 和 V 块
                T.copy(K[bz, bh, bn * block_N:, :], K_shared)
                T.copy(V[bz, bh, bn * block_N:, :], V_shared)
                T.sync()

                # 计算 QK^T
                T.gemm(Q_shared, K_shared, scores, transpose_B=True)

                # 应用缩放和 causal mask
                for i, j in T.Parallel(block_M, block_N):
                    row_idx = bm * block_M + i
                    col_idx = bn * block_N + j
                    scores[i, j] = T.if_then_else(
                        col_idx <= row_idx,
                        scores[i, j] * scale,
                        -float("inf"),
                    )

                # 在线 Softmax
                for i in T.serial(block_M):
                    new_max = T.max(scores[i, :], axis=0)
                    old_max = row_max[i]
                    row_max[i] = T.max(old_max, new_max)

                    # 更新累加器
                    exp_old = T.exp(old_max - row_max[i])
                    exp_new = T.exp(new_max - row_max[i])

                    log_sum[i] = log_sum[i] * exp_old + exp_new

                    for j in T.serial(head_dim):
                        acc[i, j] = acc[i, j] * exp_old

                # 应用 softmax 并累加 V
                for i, j in T.Parallel(block_M, block_N):
                    scores[i, j] = T.exp(scores[i, j] - row_max[i])

                T.gemm(scores, V_shared, acc)

            # 归一化并写回
            for i in T.serial(block_M):
                for j in T.serial(head_dim):
                    acc[i, j] = acc[i, j] / log_sum[i]

            T.copy(acc, O[bz, bh, bm * block_M:, :])

    return main
```

这段代码服务于 17.10.2 MI300X FlashAttention 实现 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 17.11 性能调优

### 17.11.1 rocm-smi 工具

```bash
# 查看 GPU 状态
rocm-smi

# 详细信息
rocm-smi --showid
rocm-smi --showtemp
rocm-smi --showmeminfo vram
rocm-smi --showclocks
rocm-smi --showpower

# 监控 GPU 使用率
rocm-smi -i 0 --showuse

# 设置性能模式
rocm-smi --setperflevel high
rocm-smi --setpoweroverdrive 400  # 设置功耗上限

# 查看所有 GPU
rocm-smi --showid --showtemp --showuse --showmeminfo vram --showclocks
```

这段命令展示了 17.11.1 rocm-smi 工具 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

### 17.11.2 rocprof 性能分析

```bash
# 基本性能分析
rocprof --stats ./my_application

# 生成时间线
rocprof --hip-trace --hsa-trace ./my_application

# 指定计数器
rocprof --pmc SQ_WAVES,SQ_INSTS,SQ_THREAD_CYCLES ./my_application

# 分析特定核函数
rocprof --basenames on --stats ./my_application

# 生成 CSV 报告
rocprof --timestamp on --stats --csv ./my_application
```

这段命令展示了 17.11.2 rocprof 性能分析 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

### 17.11.3 omniperf 分析

```bash
# 安装 omniperf
pip install omniperf

# 运行分析
omniperf profile -n my_kernel ./my_application

# 查看结果
omniperf analyze -p workloads/my_kernel

# 生成 roofline 图
omniperf roofline -p workloads/my_kernel

# 导出报告
omniperf analyze -p workloads/my_kernel --format html
```

这段命令展示了 17.11.3 omniperf 分析 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

### 17.11.4 性能调优 Checklist

```python
# TileLang AMD 后端性能调优清单
performance_checklist = {
    "memory": {
        "lds_usage": "确保 LDS 使用率在 64KB 以内",
        "lds_bank_conflict": "避免 LDS bank conflict",
        "coalesced_access": "确保全局内存合并访问",
        "prefetch": "使用软件流水线预取数据",
    },
    "compute": {
        "mfma_utilization": "最大化 MFMA 指令利用率",
        "wavefront_occupancy": "保持高 wavefront 占用率",
        "register_pressure": "控制寄存器使用量",
        "instruction_mix": "平衡计算和内存指令",
    },
    "launch": {
        "block_size": "选择合适的 block 大小",
        "grid_size": "确保足够的并行度",
        "waves_per_cu": "目标每 CU 8-16 个 wavefront",
    },
    "architecture": {
        "cdna_version": "针对目标架构优化",
        "infinity_cache": "利用 Infinity Cache",
        "unified_memory": "合理使用统一内存",
    },
}
```

这段代码服务于 17.11.4 性能调优 Checklist 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 17.12 常见问题排查

### 17.12.1 编译错误

**问题 1：找不到 ROCm/HIP**

```bash
# 错误信息
# error: cannot find hip/hip_runtime.h

# 解决方案
export PATH=$PATH:/opt/rocm/bin
export HIP_PATH=/opt/rocm
export CMAKE_PREFIX_PATH=/opt/rocm:$CMAKE_PREFIX_PATH
```

**问题 2：不支持的 GPU 架构**

```bash
# 错误信息
# error: invalid GPU architecture 'gfx942'

# 解决方案：确认 ROCm 版本支持目标架构
rocm-smi --showid
# 确保编译时指定正确的架构
export GPU_TARGETS=gfx942
```

**问题 3：MFMA 指令不支持**

```python
# 错误信息
# RuntimeError: MFMA instruction not supported for this dtype

# 解决方案：检查数据类型是否支持
def check_mfma_support(dtype, arch):
    supported = {
        "gfx908": ["float16", "bfloat16", "int8"],
        "gfx90a": ["float16", "bfloat16", "int8"],
        "gfx942": ["float16", "bfloat16", "int8", "float8_e4m3fnuz"],
    }
    if dtype not in supported.get(arch, []):
        raise ValueError(f"Unsupported dtype {dtype} for {arch}")
```

这段代码服务于 17.12.1 编译错误 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 17.12.2 运行时错误

**问题 4：LDS 超出限制**

```python
# 错误信息
# RuntimeError: Shared memory allocation exceeds LDS limit

# 解决方案：减小块大小或使用分块策略
def adjust_block_size_for_lds(block_M, block_K, dtype_size, max_lds=65536):
    lds_usage = block_M * block_K * dtype_size * 2  # 双缓冲
    if lds_usage > max_lds:
        # 减小块大小
        while lds_usage > max_lds:
            block_K = block_K // 2
            lds_usage = block_M * block_K * dtype_size * 2
    return block_M, block_K
```

**问题 5：Wavefront 占用率低**

```python
# 问题：性能不佳，占用率低
# 诊断
def diagnose_occupancy(vgpr_per_thread, lds_per_block, block_size):
    wave_size = 64
    waves_per_block = (block_size + wave_size - 1) // wave_size

    vgpr_per_wave = vgpr_per_thread * wave_size
    waves_by_vgpr = 512 // vgpr_per_wave
    waves_by_lds = 65536 // lds_per_block

    max_waves = min(waves_by_vgpr, waves_by_lds, 16)
    actual_waves = (block_size // wave_size) * (max_waves // waves_per_block)

    print(f"VGPR limit waves: {waves_by_vgpr}")
    print(f"LDS limit waves: {waves_by_lds}")
    print(f"Max waves per CU: {max_waves}")
    print(f"Occupancy: {actual_waves / 16 * 100:.1f}%")
```

**问题 6：内存传输瓶颈**

```python
# 问题：数据传输成为瓶颈
# 解决方案：使用异步传输和流水线
def optimize_memory_pipeline(A, B, C, block_M, block_N, block_K, num_stages=3):
    # 使用双缓冲或多级流水
    A_buffers = [T.alloc_shared([block_M, block_K], "float16") for _ in range(num_stages)]
    B_buffers = [T.alloc_shared([block_K, block_N], "float16") for _ in range(num_stages)]

    # 流水线加载
    for stage in range(num_stages):
        # 预取下一阶段的数据
        prefetch_stage = (stage + 1) % num_stages
        T.copy(A[prefetch_stage * block_M:, :], A_buffers[prefetch_stage])
        T.copy(B[prefetch_stage * block_N:, :], B_buffers[prefetch_stage])

        # 计算当前阶段
        T.gemm(A_buffers[stage], B_buffers[stage], C_local)

    return C_local
```

这段代码服务于 17.12.2 运行时错误 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 17.12.3 性能问题

**问题 7：与 NVIDIA GPU 性能差距大**

```python
# 诊断工具
def compare_with_nvidia(mi300x_time, a100_time):
    """
    分析 MI300X vs A100 的性能差异
    """
    # 理论峰值比
    mi300x_peak = 5214  # TFLOPS (FP8)
    a100_peak = 624     # TFLOPS (FP8 via sparsity)

    # 实际性能比
    actual_ratio = a100_time / mi300x_time
    theoretical_ratio = mi300x_peak / a100_peak

    efficiency = actual_ratio / theoretical_ratio

    print(f"MI300X time: {mi300x_time:.2f} ms")
    print(f"A100 time: {a100_time:.2f} ms")
    print(f"Actual speedup: {actual_ratio:.2f}x")
    print(f"Theoretical speedup: {theoretical_ratio:.2f}x")
    print(f"Efficiency: {efficiency*100:.1f}%")

    if efficiency < 0.7:
        print("Warning: Performance gap detected!")
        print("Possible causes:")
        print("  - Memory bandwidth not fully utilized")
        print("  - MFMA utilization low")
        print("  - Occupancy too low")
```

这段代码服务于 17.12.3 性能问题 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 17.13 总结

<div data-component="Summary">

### ✅ 本章关键要点

1. **ROCm 生态**：AMD 的开源 GPU 计算平台，包含 HIP 运行时、rocBLAS、MIOpen 等核心组件
2. **HIP 编程模型**：与 CUDA 高度兼容，主要差异在 wavefront 大小（64 vs 32）
3. **CDNA 架构**：从 CDNA1 到 CDNA3，计算能力和内存带宽持续提升
4. **MFMA 指令**：AMD 的矩阵运算指令，类似 NVIDIA 的 Tensor Core
5. **MI300X**：304 CUs、192GB HBM3、5.3 TB/s 带宽、统一内存架构
6. **LDS 管理**：64KB 共享内存，需注意 bank conflict
7. **Wavefront 调度**：Wave64 默认，Wave32 可选
8. **TileLang 后端**：通过 IR → HIP 代码生成支持 AMD GPU

### 🎯 学习目标检查

- [ ] 能够安装和配置 ROCm 环境
- [ ] 理解 HIP 编程模型与 CUDA 的对应关系
- [ ] 掌握 CDNA 架构的硬件特性
- [ ] 能够选择合适的 MFMA 指令
- [ ] 了解 MI300X 的硬件优势
- [ ] 能够编写 TileLang AMD 后端代码
- [ ] 掌握性能调优方法

</div>

---

## 17.14 MFMA 指令详解与代码示例

### 17.14.1 MFMA_32x32x8 指令详解

MFMA_32x32x8 是 CDNA 架构上最常用的矩阵乘累加指令之一，每个 wavefront（64 线程）协作计算一个 32×32 的输出矩阵。

```cpp
// MFMA_32x32x8 FP16 指令详解
// 输入矩阵 A: 32×8（FP16），输入矩阵 B: 8×32（FP16），输出 C: 32×32（FP32）
// 64 个线程协作完成计算，每个线程负责输出矩阵的部分元素

__global__ void mfma_32x32x8_example(
    const half* __restrict__ A,   // 输入矩阵 A，形状 32×8，FP16 格式
    const half* __restrict__ B,   // 输入矩阵 B，形状 8×32，FP16 格式
    float* __restrict__ C         // 输出矩阵 C，形状 32×32，FP32 格式
) {
    // 声明共享内存，用于存放输入矩阵块
    __shared__ half A_smem[32][8];   // A 矩阵共享内存，32 行 8 列
    __shared__ half B_smem[8][32];   // B 矩阵共享内存，8 行 32 列

    int tid = threadIdx.x;           // 线程在块内的索引
    int lane = tid % 64;             // 当前线程在 wavefront 中的 lane 编号（0-63）

    // 从全局内存加载数据到共享内存（64 个线程协作加载）
    // 每个线程加载 4 个 FP16 元素（32×8 = 256 个元素 / 64 线程 = 4 个/线程）
    int load_idx = tid;
    if (load_idx < 256) {
        int row = load_idx / 8;      // 计算行索引
        int col = load_idx % 8;      // 计算列索引
        A_smem[row][col] = A[row * 8 + col];  // 从全局内存读取 A 矩阵
    }

    // 同理加载 B 矩阵（8×32 = 256 个元素）
    if (load_idx < 256) {
        int row = load_idx / 32;
        int col = load_idx % 32;
        B_smem[row][col] = B[row * 32 + col];  // 从全局内存读取 B 矩阵
    }

    __syncthreads();  // 等待所有线程完成数据加载

    // 声明输出寄存器，每个线程持有输出矩阵的部分元素
    float c_reg[16];  // 每个线程持有 16 个 FP32 结果（32×32 / 64 线程 ≈ 16）

    // 初始化累加器为 0
    for (int i = 0; i < 16; i++) {
        c_reg[i] = 0.0f;
    }

    // 使用内联汇编调用 MFMA 指令
    // v_mfma_f32_32x32x8f16: FP16 输入，FP32 输出，32×32×8 形状
    asm volatile(
        "v_mfma_f32_32x32x8f16 %0, %1, %2, %3\n"
        : "=v"(c_reg[0])                          // 输出：FP32 累加结果
        : "v"(A_smem[0][0]),                      // 输入 A：32×8 FP16 矩阵
          "v"(B_smem[0][0]),                      // 输入 B：8×32 FP16 矩阵
          "v"(c_reg[0])                           // 累加器：FP32 初始值
    );

    // 将结果写回全局内存
    // 每个线程负责写入其持有的 16 个结果到正确的全局位置
    int out_row = lane / 4;   // 输出行索引（lane 映射到 32 行）
    int out_col = lane % 4;   // 输出列索引（每 lane 4 列）
    for (int i = 0; i < 16; i++) {
        int r = out_row + (i / 4) * 8;  // 计算实际行号
        int c = out_col + (i % 4) * 8;  // 计算实际列号
        if (r < 32 && c < 32) {
            C[r * 32 + c] = c_reg[i];   // 写入全局内存
        }
    }
}
```

这段代码服务于 17.14.1 MFMA_32x32x8 指令详解 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 17.14.2 MFMA_16x16x16 指令详解

MFMA_16x16x16 适用于较小的矩阵块或需要更高 K 维度吞吐的场景。

```cpp
// MFMA_16x16x16 FP16 指令详解
// 输入矩阵 A: 16×16（FP16），输入矩阵 B: 16×16（FP16），输出 C: 16×16（FP32）
// 64 个线程协作计算，但输出矩阵更小，每个线程负责 4 个元素

__global__ void mfma_16x16x16_example(
    const half* __restrict__ A,     // 输入矩阵 A，形状 16×16，FP16 格式
    const half* __restrict__ B,     // 输入矩阵 B，形状 16×16，FP16 格式
    float* __restrict__ C           // 输出矩阵 C，形状 16×16，FP32 格式
) {
    // 声明共享内存
    __shared__ half A_smem[16][16];  // A 矩阵共享内存，16 行 16 列
    __shared__ half B_smem[16][16];  // B 矩阵共享内存，16 行 16 列

    int tid = threadIdx.x;           // 线程索引
    int lane = tid % 64;             // wavefront 内 lane 编号

    // 协作加载 A 矩阵（16×16 = 256 个元素，64 线程每线程加载 4 个）
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * 64;      // 计算全局加载索引
        if (idx < 256) {
            int row = idx / 16;      // 计算行索引
            int col = idx % 16;      // 计算列索引
            A_smem[row][col] = A[row * 16 + col];
        }
    }

    // 协作加载 B 矩阵
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * 64;
        if (idx < 256) {
            int row = idx / 16;
            int col = idx % 16;
            B_smem[row][col] = B[row * 16 + col];
        }
    }

    __syncthreads();  // 同步，确保数据加载完成

    // 声明输出寄存器（每线程 4 个 FP32 值）
    float c_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // 使用内联汇编调用 MFMA_16x16x16 指令
    // 相比 MFMA_32x32x8，K 维度更大（16 vs 8），适合 K 维较大的矩阵
    asm volatile(
        "v_mfma_f32_16x16x16f16 %0, %1, %2, %3\n"
        : "=v"(c_reg[0])                         // 输出：4 个 FP32 累加结果
        : "v"(A_smem[0][0]),                     // 输入 A：16×16 FP16
          "v"(B_smem[0][0]),                     // 输入 B：16×16 FP16
          "v"(c_reg[0])                          // 累加器
    );

    // 写回结果（16×16 = 256 个元素，64 线程每线程写 4 个）
    for (int i = 0; i < 4; i++) {
        int idx = lane + i * 64;
        int row = idx / 16;
        int col = idx % 16;
        C[row * 16 + col] = c_reg[i];
    }
}
```

这段代码服务于 17.14.2 MFMA_16x16x16 指令详解 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 17.14.3 MFMA_4x4x4 指令详解

MFMA_4x4x4 是最小的 MFMA 形状，适用于非常小的分块或特定的计算模式。

```cpp
// MFMA_4x4x4 BF16 指令详解
// 输入矩阵 A: 4×4（BF16），输入矩阵 B: 4×4（BF16），输出 C: 4×4（FP32）
// 64 个线程协作计算 4×4 的输出，每个线程可能只负责 0-1 个元素

__global__ void mfma_4x4x4_example(
    const hip_bfloat16* __restrict__ A,  // 输入矩阵 A，形状 4×4，BF16 格式
    const hip_bfloat16* __restrict__ B,  // 输入矩阵 B，形状 4×4，BF16 格式
    float* __restrict__ C                 // 输出矩阵 C，形状 4×4，FP32 格式
) {
    int tid = threadIdx.x;  // 线程索引
    int lane = tid % 64;    // wavefront 内 lane 编号

    // 声明寄存器存储输入（每个线程只持有部分元素）
    hip_bfloat16 a_reg[1];  // 每个线程持有 A 的 1 个元素
    hip_bfloat16 b_reg[1];  // 每个线程持有 B 的 1 个元素
    float c_reg[1] = {0.0f};  // 每个线程持有 C 的 1 个元素

    // 分布式加载：64 个线程加载 16 个元素，大部分线程空闲
    if (lane < 16) {
        a_reg[0] = A[lane];  // 加载 A 矩阵元素
        b_reg[0] = B[lane];  // 加载 B 矩阵元素
    }

    // 使用 MFMA_4x4x4 指令
    // 这是最小的 MFMA 形状，主要用于：
    // 1. 非常小的矩阵乘法
    // 2. 与更大的 MFMA 形状组合使用，处理边界情况
    // 3. 特定的细粒度并行计算场景
    asm volatile(
        "v_mfma_f32_4x4x4bf16 %0, %1, %2, %3\n"
        : "=v"(c_reg[0])                         // 输出：FP32 累加结果
        : "v"(a_reg[0]),                         // 输入 A：4×4 BF16
          "v"(b_reg[0]),                         // 输入 B：4×4 BF16
          "v"(c_reg[0])                          // 累加器
    );

    // 写回结果
    if (lane < 16) {
        C[lane] = c_reg[0];
    }
}
```

这段代码服务于 17.14.3 MFMA_4x4x4 指令详解 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 17.14.4 MFMA 指令组合策略

在实际 GEMM 实现中，通常需要组合使用不同形状的 MFMA 指令来处理不同的块大小。

```python
def select_mfma_combination(block_M, block_N, block_K, dtype):
    """
    根据分块大小选择最优的 MFMA 指令组合

    策略：
    1. 主计算使用大形状 MFMA（32x32x8 或 16x16x16）
    2. 边界处理使用小形状 MFMA（4x4x4）
    3. 根据 K 维度选择合适的 MFMA 形状
    """
    combinations = []

    # 主计算区域：使用大形状 MFMA
    main_m = (block_M // 32) * 32   # 对齐到 32 的倍数
    main_n = (block_N // 32) * 32

    if main_m >= 32 and main_n >= 32:
        combinations.append({
            "instruction": "v_mfma_f32_32x32x8f16",
            "m": 32, "n": 32, "k": 8,
            "region": "main"
        })

    # 剩余行处理：使用 16x16 形状
    remain_m = block_M - main_m
    if remain_m >= 16:
        combinations.append({
            "instruction": "v_mfma_f32_16x16x16f16",
            "m": 16, "n": 16, "k": 16,
            "region": "row_remainder"
        })

    # 剩余列处理
    remain_n = block_N - main_n
    if remain_n >= 16:
        combinations.append({
            "instruction": "v_mfma_f32_16x16x16f16",
            "m": 16, "n": 16, "k": 16,
            "region": "col_remainder"
        })

    # 角落区域：使用最小 MFMA
    if remain_m > 0 and remain_n > 0:
        combinations.append({
            "instruction": "v_mfma_f32_4x4x4bf16",
            "m": 4, "n": 4, "k": 4,
            "region": "corner"
        })

    return combinations
```

这段代码服务于 17.14.4 MFMA 指令组合策略 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 17.15 MI300X vs A100 性能对比

### 17.15.1 硬件规格对比

| 参数 | MI300X | A100 (80GB) | 比率 | 说明 |
|------|--------|-------------|------|------|
| FP16 矩阵吞吐 | 2,607 TFLOPS | 312 TFLOPS | 8.4× | MI300X 优势明显 |
| FP8 矩阵吞吐 | 5,214 TFLOPS | 624 TFLOPS | 8.4× | 两者均支持 FP8 |
| FP32 向量吞吐 | 163.4 TFLOPS | 19.5 TFLOPS | 8.4× | 向量计算 |
| HBM 容量 | 192GB | 80GB | 2.4× | MI300X 更大 |
| HBM 带宽 | 5.3 TB/s | 2.0 TB/s | 2.65× | MI300X 带宽更大 |
| L2 缓存 | 256MB (Infinity Cache) | 40MB | 6.4× | Infinity Cache 更大 |
| TDP | 750W | 400W | 1.875× | MI300X 功耗更高 |
| 互联 | Infinity Fabric | NVLink | — | 各有特点 |

### 17.15.2 GEMM 性能对比

| GEMM 尺寸 (M×N×K) | 数据类型 | MI300X (ms) | A100 (ms) | 加速比 | 说明 |
|---------------------|----------|-------------|-----------|--------|------|
| 1024×1024×1024 | FP16 | 0.082 | 0.198 | 2.41× | 小矩阵，MI300X 优势适中 |
| 2048×2048×2048 | FP16 | 0.315 | 1.024 | 3.25× | 中等矩阵 |
| 4096×4096×4096 | FP16 | 1.856 | 7.562 | 4.07× | 大矩阵，差距拉大 |
| 8192×8192×8192 | FP16 | 13.24 | 59.87 | 4.52× | 超大矩阵，接近理论峰值 |
| 4096×4096×4096 | FP8 | 0.928 | 3.921 | 4.23× | FP8 进一步提升 |
| 4096×4096×4096 | INT8 | 0.856 | 3.654 | 4.27× | 整数运算 |

### 17.15.3 FlashAttention 性能对比

| 配置 (Batch×Heads×Seq×Dim) | MI300X (ms) | A100 (ms) | 加速比 |
|-----------------------------|-------------|-----------|--------|
| 1×32×2048×128 | 0.412 | 1.205 | 2.93× |
| 1×32×4096×128 | 1.587 | 5.023 | 3.17× |
| 1×32×8192×128 | 6.124 | 20.145 | 3.29× |
| 4×32×2048×128 | 1.645 | 4.821 | 2.93× |
| 4×32×4096×128 | 6.348 | 20.092 | 3.17× |
| 1×32×4096×256 | 3.124 | 10.067 | 3.22× |

### 17.15.4 Conv2d 性能对比

| 配置 (N×C×H×W, K×k×k) | 数据类型 | MI300X (ms) | A100 (ms) | 加速比 |
|------------------------|----------|-------------|-----------|--------|
| 1×64×56×56, 64×3×3 | FP16 | 0.034 | 0.098 | 2.88× |
| 1×128×28×28, 128×3×3 | FP16 | 0.052 | 0.156 | 3.00× |
| 1×256×14×14, 256×3×3 | FP16 | 0.078 | 0.245 | 3.14× |
| 1×512×7×7, 512×3×3 | FP16 | 0.112 | 0.378 | 3.38× |
| 8×64×56×56, 64×3×3 | FP16 | 0.215 | 0.687 | 3.19× |

> [!NOTE]
> 以上数据为参考值，实际性能取决于具体实现、ROCm/CUDA 版本、系统配置等因素。MI300X 的优势在大矩阵场景下更为明显。

### 17.15.5 性能差异分析

```python
def analyze_performance_gap(mi300x_time, a100_time, operation_type):
    """
    分析 MI300X 与 A100 的性能差距原因
    """
    speedup = a100_time / mi300x_time

    # 理论峰值比
    theoretical_ratio = {
        "gemm_fp16": 2607 / 312,      # ~8.4x
        "gemm_fp8": 5214 / 624,        # ~8.4x
        "attention": 2607 / 312,       # ~8.4x
        "conv2d": 2607 / 312,          # ~8.4x
    }

    theo = theoretical_ratio.get(operation_type, 8.4)
    efficiency = speedup / theo

    print(f"操作类型: {operation_type}")
    print(f"MI300X 耗时: {mi300x_time:.3f} ms")
    print(f"A100 耗时: {a100_time:.3f} ms")
    print(f"实际加速比: {speedup:.2f}x")
    print(f"理论加速比: {theo:.2f}x")
    print(f"效率: {efficiency*100:.1f}%")

    if efficiency < 0.4:
        print("⚠ 效率偏低，可能原因：")
        print("  - 内存带宽未充分利用（IO bound）")
        print("  - MFMA 利用率低（矩阵未对齐）")
        print("  - Occupancy 不足（寄存器/LDS 限制）")
    elif efficiency < 0.6:
        print("✓ 效率中等，仍有优化空间")
    else:
        print("✓ 效率良好")
```

这段代码服务于 17.15.5 性能差异分析 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 17.16 HIP 编程代码示例

### 17.16.1 完整 HIP 矩阵乘法示例

```cpp
// HIP 矩阵乘法完整示例（带逐行中文注释）
#include <hip/hip_runtime.h>   // 引入 HIP 运行时头文件，提供 GPU 编程 API
#include <stdio.h>             // 标准输入输出库
#include <stdlib.h>            // 标准库函数（malloc, free 等）

// 矩阵维度定义
#define M 1024                  // 矩阵 A 的行数
#define K 1024                  // 矩阵 A 的列数 / 矩阵 B 的行数
#define N 1024                  // 矩阵 B 的列数
#define BLOCK_SIZE 16           // 线程块大小（16×16）

// HIP 核函数：朴素矩阵乘法
// 每个线程计算输出矩阵 C 的一个元素
__global__ void matmul_naive(
    const float* __restrict__ A,   // 输入矩阵 A，M×K，只读
    const float* __restrict__ B,   // 输入矩阵 B，K×N，只读
    float* __restrict__ C,         // 输出矩阵 C，M×N，读写
    int m, int n, int k            // 矩阵维度参数
) {
    // 计算当前线程负责的输出元素的行号和列号
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 全局行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 全局列索引

    // 边界检查：确保不越界访问
    if (row < m && col < n) {
        float sum = 0.0f;  // 累加器，存储点积结果

        // 沿 K 维度求点积
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i]    // A 的第 row 行第 i 列
                 * B[i * n + col];   // B 的第 i 行第 col 列
        }

        C[row * n + col] = sum;      // 将结果写入 C 的第 row 行第 col 列
    }
}

// HIP 核函数：共享内存优化的矩阵乘法
// 使用共享内存减少全局内存访问次数
__global__ void matmul_shared(
    const float* __restrict__ A,   // 输入矩阵 A
    const float* __restrict__ B,   // 输入矩阵 B
    float* __restrict__ C,         // 输出矩阵 C
    int m, int n, int k
) {
    // 声明共享内存（每个线程块私有，块内线程共享）
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];  // A 的子块
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];  // B 的子块

    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 全局行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 全局列索引
    int ty = threadIdx.y;  // 块内线程行索引
    int tx = threadIdx.x;  // 块内线程列索引

    float sum = 0.0f;  // 累加器

    // 沿 K 维度分块迭代
    for (int t = 0; t < (k + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // 协作加载 A 的子块到共享内存
        int a_col = t * BLOCK_SIZE + tx;  // A 子块的列索引
        if (row < m && a_col < k) {
            As[ty][tx] = A[row * k + a_col];  // 从全局内存加载
        } else {
            As[ty][tx] = 0.0f;  // 越界填 0
        }

        // 协作加载 B 的子块到共享内存
        int b_row = t * BLOCK_SIZE + ty;  // B 子块的行索引
        if (b_row < k && col < n) {
            Bs[ty][tx] = B[b_row * n + col];  // 从全局内存加载
        } else {
            Bs[ty][tx] = 0.0f;  // 越界填 0
        }

        __syncthreads();  // 同步：等待所有线程完成加载

        // 计算子块的矩阵乘法并累加
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];  // 共享内存访问，延迟低
        }

        __syncthreads();  // 同步：确保计算完成后再加载下一块
    }

    // 将最终结果写入全局内存
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

// 主函数
int main() {
    size_t size_A = M * K * sizeof(float);  // A 矩阵的字节大小
    size_t size_B = K * N * sizeof(float);  // B 矩阵的字节大小
    size_t size_C = M * N * sizeof(float);  // C 矩阵的字节大小

    // 主机端（CPU）内存分配
    float* h_A = (float*)malloc(size_A);     // 分配主机内存存放 A
    float* h_B = (float*)malloc(size_B);     // 分配主机内存存放 B
    float* h_C = (float*)malloc(size_C);     // 分配主机内存存放 C

    // 初始化输入矩阵（随机值）
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;

    // 设备端（GPU）内存分配
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size_A);   // 在 GPU 上分配 A 矩阵内存
    hipMalloc(&d_B, size_B);   // 在 GPU 上分配 B 矩阵内存
    hipMalloc(&d_C, size_C);   // 在 GPU 上分配 C 矩阵内存

    // 从主机拷贝数据到设备
    hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice);  // A: CPU → GPU
    hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice);  // B: CPU → GPU

    // 配置核函数启动参数
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);  // 线程块大小：16×16 = 256 线程
    dim3 gridDim(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,  // X 方向的网格大小
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE   // Y 方向的网格大小
    );

    // 记录启动时间
    hipEvent_t start, stop;
    hipEventCreate(&start);   // 创建 CUDA 事件用于计时
    hipEventCreate(&stop);
    hipEventRecord(start);    // 记录起始时间

    // 启动核函数
    hipLaunchKernelGGL(matmul_shared,
                       gridDim,       // 网格维度
                       blockDim,      // 块维度
                       0, 0,          // 动态共享内存大小、流
                       d_A, d_B, d_C, // 核函数参数
                       M, N, K);

    hipEventRecord(stop);     // 记录结束时间
    hipEventSynchronize(stop); // 等待核函数执行完成

    float elapsed_ms;
    hipEventElapsedTime(&elapsed_ms, start, stop);  // 计算耗时（毫秒）
    printf("矩阵乘法耗时: %.3f ms\n", elapsed_ms);
    printf("性能: %.2f GFLOPS\n",
           (2.0 * M * N * K) / (elapsed_ms * 1e6));  // 计算 GFLOPS

    // 从设备拷贝结果到主机
    hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);  // C: GPU → CPU

    // 验证结果（检查前几个元素）
    printf("C[0][0] = %f\n", h_C[0]);
    printf("C[0][1] = %f\n", h_C[1]);

    // 释放资源
    hipFree(d_A);    // 释放 GPU 上的 A 矩阵
    hipFree(d_B);    // 释放 GPU 上的 B 矩阵
    hipFree(d_C);    // 释放 GPU 上的 C 矩阵
    free(h_A);       // 释放 CPU 上的 A 矩阵
    free(h_B);       // 释放 CPU 上的 B 矩阵
    free(h_C);       // 释放 CPU 上的 C 矩阵
    hipEventDestroy(start);  // 销毁计时事件
    hipEventDestroy(stop);

    return 0;
}
```

这段代码服务于 17.16.1 完整 HIP 矩阵乘法示例 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 17.16.2 HIP 编译与运行命令

```bash
# 使用 hipcc 编译器编译 HIP 程序
# hipcc 是 ROCm 提供的 HIP 编译器，类似 nvcc
hipcc -O3 --offload-arch=gfx942 matmul.cpp -o matmul

# 运行程序
./matmul

# 编译选项说明：
# -O3              : 最高级别优化
# --offload-arch   : 指定目标 GPU 架构（gfx942 = MI300X）
# -o matmul        : 输出文件名为 matmul
```

这段命令展示了 17.16.2 HIP 编译与运行命令 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

---

## 17.17 rocprof 与 omniperf 性能分析指南

### 17.17.1 rocprof 详细使用指南

```bash
# ============================================================
# rocprof 基础用法
# ============================================================

# 1. 收集核函数执行统计信息
#    输出每个核函数的执行次数、总时间、平均时间等
rocprof --stats ./my_application

# 2. 收集硬件性能计数器
#    SQ_WAVES: 发射的 wavefront 数量
#    SQ_INSTS: 执行的指令数量
#    SQ_THREAD_CYCLES: 线程周期数
rocprof --pmc SQ_WAVES,SQ_INSTS,SQ_THREAD_CYCLES ./my_application

# 3. 生成 HIP API 追踪（时间线视图）
#    记录每个 HIP API 调用的开始和结束时间
rocprof --hip-trace ./my_application

# 4. 生成 HSA API 追踪
#    记录 HSA（异构系统架构）层面的 API 调用
rocprof --hsa-trace ./my_application

# 5. 同时收集多种信息
rocprof --stats --hip-trace --hsa-trace \
        --pmc SQ_WAVES,SQ_INSTS \
        ./my_application

# 6. 输出 CSV 格式报告（便于脚本处理）
rocprof --timestamp on --stats --csv -o results.csv ./my_application

# 7. 分析特定核函数
#    通过核函数名称过滤
rocprof --basenames on --stats --filter "gemm_kernel" ./my_application

# ============================================================
# rocprof 高级计数器
# ============================================================

# 内存相关计数器
rocprof --pmc TCC_HIT_SUM,TCC_MISS_SUM,TA_TA_BUSY ./my_application
# TCC_HIT_SUM  : L2 缓存命中次数
# TCC_MISS_SUM : L2 缓存未命中次数
# TA_TA_BUSY   : 纹理单元忙碌周期

# 计算相关计数器
rocprof --pmc SQ_INSTS_VALU,SQ_INSTS_SALU,SQ_INSTS_VMEM ./my_application
# SQ_INSTS_VALU  : 向量 ALU 指令数
# SQ_INSTS_SALU  : 标量 ALU 指令数
# SQ_INSTS_VMEM  : 向量内存指令数

# MFMA 相关计数器
rocprof --pmc SQ_INSTS_MFMA,SQ_MFMA_BUSY ./my_application
# SQ_INSTS_MFMA : MFMA 指令执行数量
# SQ_MFMA_BUSY  : MFMA 单元忙碌周期
```

这段命令展示了 17.17.1 rocprof 详细使用指南 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

### 17.17.2 rocprof 输出分析

```bash
# 运行分析并保存结果
rocprof --stats --pmc SQ_WAVES,SQ_INSTS,SQ_THREAD_CYCLES \
        -o profile_results.csv ./my_application

# 输出文件说明：
# profile_results.csv         - 统计数据 CSV
# profile_results.json        - JSON 格式结果
# profile_results_stats.csv   - 核函数统计
```

这段命令展示了 17.17.2 rocprof 输出分析 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

```python
# 解析 rocprof 输出的 Python 脚本
import pandas as pd

def analyze_rocprof_output(stats_csv):
    """
    分析 rocprof 的输出结果
    """
    df = pd.read_csv(stats_csv)

    print("=" * 60)
    print("核函数性能分析报告")
    print("=" * 60)

    for _, row in df.iterrows():
        name = row.get("KernelName", "unknown")
        calls = row.get("Calls", 0)
        total_time = row.get("TotalDurationNs", 0)
        avg_time = row.get("AverageNs", 0)
        waves = row.get("SQ_WAVES", 0)
        insts = row.get("SQ_INSTS", 0)

        print(f"\n核函数: {name}")
        print(f"  调用次数: {calls}")
        print(f"  总耗时: {total_time / 1e6:.3f} ms")
        print(f"  平均耗时: {avg_time / 1e3:.3f} us")
        print(f"  Wavefront 数: {waves}")
        print(f"  指令数: {insts}")

        if waves > 0:
            ipc = insts / waves  # 每 wavefront 的指令数
            print(f"  IPC (每 wavefront): {ipc:.1f}")
```

这段代码服务于 17.17.2 rocprof 输出分析 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 17.17.3 omniperf 详细使用指南

```bash
# ============================================================
# omniperf 安装与基础用法
# ============================================================

# 安装 omniperf
pip install omniperf

# 1. 收集性能数据
#    -n: 指定工作负载名称
#    --: 后面跟应用程序及其参数
omniperf profile -n my_gemm_workload ./my_gemm_app

# 2. 分析收集的数据
#    生成详细的性能分析报告
omniperf analyze -p workloads/my_gemm_workload

# 3. 生成 Roofline 图
#    可视化计算和内存瓶颈
omniperf roofline -p workloads/my_gemm_workload

# 4. 导出 HTML 报告
omniperf analyze -p workloads/my_gemm_workload --format html

# ============================================================
# omniperf 分析维度
# ============================================================

# 5. 按类别分析
omniperf analyze -p workloads/my_gemm_workload --category memory
omniperf analyze -p workloads/my_gemm_workload --category compute
omniperf analyze -p workloads/my_gemm_workload --category cache

# 6. 对比两次运行
omniperf diff -p workloads/baseline -p workloads/optimized
```

这段命令展示了 17.17.3 omniperf 详细使用指南 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

### 17.17.4 omniperf 输出解读

```python
# omniperf 分析结果解读指南
omniperf_metrics_guide = {
    # 计算效率指标
    "MFMA_Utilization": {
        "含义": "MFMA 指令单元的利用率",
        "目标": "> 80%",
        "优化方向": "增加矩阵运算密度，减少非矩阵指令",
    },
    "Wavefront_Occupancy": {
        "含义": "活跃 wavefront 占最大 wavefront 的比例",
        "目标": "> 70%",
        "优化方向": "减少寄存器使用，减小 LDS 分配",
    },

    # 内存效率指标
    "L2_Hit_Rate": {
        "含义": "L2 缓存命中率",
        "目标": "> 80%",
        "优化方向": "提高数据复用，优化访问模式",
    },
    "HBM_Bandwidth_Utilization": {
        "含义": "HBM 带宽利用率",
        "目标": "> 70%",
        "优化方向": "合并内存访问，使用向量化加载",
    },
    "LDS_Bank_Conflicts": {
        "含义": "LDS bank 冲突次数",
        "目标": "越少越好",
        "优化方向": "调整数据布局，添加填充",
    },

    # 指令效率指标
    "IPC": {
        "含义": "每周期指令数（Instructions Per Cycle）",
        "目标": "> 1.0",
        "优化方向": "增加指令级并行，减少依赖链",
    },
}
```

这段代码服务于 17.17.4 omniperf 输出解读 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 17.18 ROCm 环境问题排查

### 17.18.1 ROCm 版本兼容性问题

```bash
# ============================================================
# 问题 1：ROCm 版本与 GPU 不兼容
# ============================================================
# 症状：安装后 rocm-smi 无法识别 GPU
# 错误信息："No GPU devices found"

# 诊断步骤：
# 1. 检查当前 ROCm 版本
cat /opt/rocm/.info/version
# 输出示例：6.2.0-66

# 2. 检查 GPU 硬件型号
lspci | grep -i amd
# 输出示例：03:00.0 Display controller: Advanced Micro Devices...

# 3. 检查支持的 GPU 列表
cat /opt/rocm/.info/supported-gpus.csv

# 解决方案：
# 如果 GPU 不在支持列表中，需要升级 ROCm 到支持该 GPU 的版本
# 例如 MI300X 需要 ROCm 6.0+

# ============================================================
# 问题 2：ROCm 与 PyTorch 版本不匹配
# ============================================================
# 症状：import torch 时报错 "ROCm version mismatch"

# 诊断：
python3 -c "import torch; print(torch.version.hip)"
# 应输出 ROCm 版本号，如 "6.2.41134-"

# 解决方案：
# 卸载当前 PyTorch 并安装匹配版本
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# ============================================================
# 问题 3：内核驱动版本不匹配
# ============================================================
# 症状：运行程序时报 "amdgpu version mismatch"

# 诊断：
dmesg | grep amdgpu | head -5
cat /sys/module/amdgpu/version

# 解决方案：
# 更新内核驱动（需要与 ROCm 版本匹配）
sudo apt update
sudo apt install amdgpu-dkms
sudo reboot
```

这段命令展示了 17.18.1 ROCm 版本兼容性问题 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

### 17.18.2 核函数启动失败问题

```bash
# ============================================================
# 问题 4：核函数启动超时
# ============================================================
# 症状：程序运行一段时间后崩溃，报 "GPU hang detected"

# 诊断：
dmesg | grep -i "gpu hang"
rocm-smi --showtemp  # 检查温度是否过高

# 解决方案：
# 1. 检查核函数是否有无限循环
# 2. 增加核函数超时时间限制
echo 100000 > /sys/module/drm/parameters/gpu_timeout_ms

# 3. 减小问题规模以测试
# 4. 检查核函数中的边界条件

# ============================================================
# 问题 5：核函数 launch 失败
# ============================================================
# 症状：hipLaunchKernelGGL 返回错误

# 诊断代码：
hipError_t err = hipGetLastError();
if (err != hipSuccess) {
    printf("核函数启动失败: %s\n", hipGetErrorString(err));
}

# 常见原因及解决：
# (a) 块大小超过限制（最大 1024 线程/块）
# 解决：减小 blockDim

# (b) 共享内存超过限制（64KB/CU）
# 解决：减小共享内存分配

# (c) 网格大小超过限制
# 解决：检查 gridDim 是否超过设备限制

# ============================================================
# 问题 6：illegal memory access
# ============================================================
# 症状：运行时报 "Memory access fault" 或 "illegal memory access"

# 诊断：
# 启用内存检查工具
export HIP_LAUNCH_BLOCKING=1  # 同步执行，便于定位出错的核函数
export AMD_LOG_LEVEL=4        # 启用详细日志

# 解决方案模板：
__global__ void safe_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 关键：始终进行边界检查
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}
```

这段命令展示了 17.18.2 核函数启动失败问题 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

### 17.18.3 内存错误排查

```python
# HIP 内存错误排查工具类
class HIPMemoryDebugger:
    """HIP 内存调试工具"""

    @staticmethod
    def check_allocation(ptr, size, name=""):
        """检查内存分配是否成功"""
        if ptr is None:
            raise RuntimeError(f"内存分配失败: {name}, 请求大小: {size} 字节")

    @staticmethod
    def safe_malloc(size, name=""):
        """安全的内存分配，带错误检查"""
        ptr = None
        err = hip.hipMalloc(ptr, size)
        if err != hip.hipSuccess:
            raise RuntimeError(
                f"hipMalloc 失败: {name}\n"
                f"  请求大小: {size / 1024 / 1024:.2f} MB\n"
                f"  错误: {hip.hipGetErrorString(err)}\n"
                f"  可用显存: {HIPMemoryDebugger.get_free_memory() / 1024 / 1024:.2f} MB"
            )
        return ptr

    @staticmethod
    def get_free_memory():
        """获取可用显存大小"""
        free, total = hip.hipMemGetInfo()
        return free

    @staticmethod
    def print_memory_status():
        """打印当前显存使用状态"""
        free, total = hip.hipMemGetInfo()
        used = total - free
        print(f"显存使用: {used / 1024 / 1024:.2f} MB / {total / 1024 / 1024:.2f} MB")
        print(f"可用显存: {free / 1024 / 1024:.2f} MB")
        print(f"使用率: {used / total * 100:.1f}%")
```

这段代码服务于 17.18.3 内存错误排查 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 17.19 补充练习

### 练习 4：MFMA 指令性能测试

```python
# 任务：比较不同 MFMA 指令的性能
# 编写一个 benchmark，测试以下 MFMA 指令的吞吐量：
# - v_mfma_f32_32x32x8f16
# - v_mfma_f32_16x16x16f16
# - v_mfma_f32_4x4x4bf16

def benchmark_mfma_instructions():
    """
    测试不同 MFMA 指令的性能

    要求：
    1. 测试三种 MFMA 形状
    2. 使用不同的矩阵尺寸
    3. 输出每种指令的 TFLOPS
    4. 分析哪种形状在什么场景下最优
    """
    # TODO: 实现您的 benchmark
    pass
```

这段代码服务于 练习 4：MFMA 指令性能测试 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 练习 5：rocprof 性能分析实战

```bash
# 任务：使用 rocprof 分析 TileLang GEMM 内核的性能
# 步骤：
# 1. 编写一个 TileLang GEMM 程序
# 2. 使用 rocprof 收集性能计数器
# 3. 分析 MFMA 利用率、缓存命中率
# 4. 根据分析结果优化内核
# 5. 再次分析，对比优化效果

# 提示：
# 使用以下计数器：
# - SQ_WAVES: wavefront 数量
# - SQ_INSTS_MFMA: MFMA 指令数
# - TCC_HIT_SUM: L2 缓存命中
# - TCC_MISS_SUM: L2 缓存未命中
```

这段命令展示了 练习 5：rocprof 性能分析实战 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

### 练习 6：MI300X 统一内存优化

```cpp
// 任务：比较统一内存和显式内存管理的性能差异
// 要求：
// 1. 实现使用 hipMallocManaged 的版本
// 2. 实现使用 hipMalloc + hipMemcpy 的版本
// 3. 在不同数据规模下比较性能
// 4. 分析统一内存的优缺点

void compare_unified_vs_explicit_memory(int n) {
    // TODO: 实现您的对比实验
}
```

这段代码服务于 练习 6：MI300X 统一内存优化 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 练习 7：LDS Bank Conflict 检测与优化

```python
# 任务：检测并修复 LDS Bank Conflict
# 要求：
# 1. 编写一个有意识引入 bank conflict 的 TileLang 内核
# 2. 使用 omniperf 检测 bank conflict
# 3. 通过添加填充消除 conflict
# 4. 对比优化前后的性能

def detect_and_fix_bank_conflict():
    """
    检测并修复 LDS Bank Conflict

    步骤：
    1. 创建一个跨步访问共享内存的内核
    2. 使用 omniperf 检测 bank conflict 数量
    3. 添加 padding 消除 conflict
    4. 重新测试并对比性能
    """
    # TODO: 实现您的检测和优化
    pass
```

这段代码服务于 练习 7：LDS Bank Conflict 检测与优化 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 练习 8：Wavefront 调度策略对比

```python
# 任务：对比 Wave32 和 Wave64 的性能差异
# 要求：
# 1. 在 MI300X 上分别使用 Wave32 和 Wave64 模式
# 2. 测试 GEMM、Attention、Elementwise 三种内核
# 3. 分析哪种模式更适合哪种内核类型
# 4. 给出选择建议

def compare_wavefront_modes():
    """
    对比 Wave32 和 Wave64 的性能差异

    测试矩阵：
    - GEMM: M=4096, N=4096, K=4096
    - Attention: SeqLen=4096, Heads=32, Dim=128
    - Elementwise: N=10000000
    """
    # TODO: 实现您的对比实验
    pass
```

这段代码服务于 练习 8：Wavefront 调度策略对比 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 练习

### 练习 1：ROCm 环境搭建

配置 ROCm 开发环境，验证 HIP 编译器工作正常。

```bash
# 任务：完成以下步骤
# 1. 安装 ROCm 6.2 或更高版本
# 2. 配置环境变量
# 3. 编写并编译一个简单的 HIP 程序
# 4. 使用 rocm-smi 验证 GPU 状态
```

这段命令展示了 练习 1：ROCm 环境搭建 中实际环境配置或诊断流程的执行方式。阅读时要关注命令顺序、环境变量、设备可见性和验证步骤，因为这些细节决定后续 kernel 能否稳定编译与运行。工程实践中不要只复制命令本身，还要理解每一步是在解决依赖、运行时路径、设备权限还是性能观测问题；否则一旦环境版本变化，就很难定位失败原因。性能相关命令还需要配合多次运行和同步点使用，避免把冷启动、缓存状态或异步执行误判为真实瓶颈。

### 练习 2：MFMA 指令选择

为不同的 GEMM 尺寸选择最优的 MFMA 指令。

```python
# 任务：实现以下函数
def optimal_mfma_for_gemm(M, N, K, dtype):
    """
    为给定的 GEMM 尺寸选择最优的 MFMA 指令

    返回：
    - mfma_instruction: MFMA 指令名称
    - block_M, block_N, block_K: 推荐的块大小
    - num_waves: 推荐的 wavefront 数量
    """
    # TODO: 实现您的选择逻辑
    pass
```

这段代码服务于 练习 2：MFMA 指令选择 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

### 练习 3：MI300X GEMM 优化

优化 GEMM 以充分利用 MI300X 的硬件特性。

```python
# 任务：优化以下 GEMM 实现
@tilelang.jit(target="hip", arch="gfx942")
def optimized_gemm_mi300x(M, N, K):
    # 要求：
    # 1. 使用 256x256 或更大的块
    # 2. 使用 MFMA 指令
    # 3. 实现软件流水线
    # 4. 目标性能：>80% 理论峰值
    pass
```

这段代码服务于 练习 3：MI300X GEMM 优化 的核心实现，重点展示了数据结构、执行流程和硬件抽象之间的对应关系。理解时应先看输入输出和资源分配，再看循环、调度或配置如何把高层算子映射到具体后端。这样的写法通常是为了减少全局内存往返、提高片上缓存复用，并让编译器有机会生成更接近硬件特性的代码。常见陷阱是只关注语法而忽略边界条件、精度类型、同步位置和资源占用，这些问题会直接导致结果错误或性能低于预期。

---

## 思考题

1. **Wave64 vs Wave32**：在什么场景下应该使用 Wave32 而不是 Wave64？这对 TileLang 的代码生成有什么影响？

2. **统一内存**：MI300X 的统一内存架构对 TileLang 的编程模型有什么影响？如何在性能和易用性之间取得平衡？

3. **MFMA 指令选择**：为什么 MFMA 指令的选择对性能至关重要？如果选择了错误的指令形状会发生什么？

4. **LDS Bank Conflict**：如何在 TileLang 的 IR 层面检测和避免 LDS bank conflict？

5. **跨平台兼容**：如何设计 TileLang 的后端架构，使其能够同时支持 NVIDIA 和 AMD GPU，同时充分利用各自的硬件特性？

---

## 扩展阅读

1. **AMD ROCm 文档**：https://rocm.docs.amd.com/
2. **HIP 编程指南**：https://rocm.docs.amd.com/projects/HIP/en/latest/
3. **CDNA 架构白皮书**：AMD Instinct MI300X 技术文档
4. **MFMA 指令参考**：AMD GPU ISA 文档
5. **omniperf 工具**：https://github.com/AMDResearch/omniperf

---

## 下一章预告

> **Chapter 18: 华为昇腾后端——Ascend C 适配**
>
> 在下一章中，我们将探索 TileLang 在华为昇腾 NPU 上的后端实现。您将学习：
> - 达芬奇架构（Da Vinci）的核心设计
> - Cube Core 和 Vector Core 的协同工作
> - Ascend C 编程模型和 CANN 框架
> - TileLang 到 Ascend C 的多级 Lowering 流程
> - 昇腾 910B/910C 的硬件特性与优化技巧
