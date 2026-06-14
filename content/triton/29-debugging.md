# Chapter 29: 调试技术与常见问题排查

> **学习目标**：
> - 掌握 Triton kernel 的调试方法论
> - 了解 Triton IR dump 机制与 IR 检查方法
> - 掌握内存错误定位与数值精度问题排查
> - 了解性能回归检测与调试技巧

---

## 29.1 调试方法论概述

Triton kernel 的调试与普通 Python 代码调试有本质区别。Kernel 运行在 GPU 上，无法直接使用 `pdb` 等传统调试器。调试 Triton 需要建立一套系统化的方法论，结合多种工具和技巧。

### 29.1.1 调试分层模型

Triton 调试可以分为四个层次：

| 层次 | 工具/方法 | 适用场景 |
|------|-----------|----------|
| Python 层 | print 调试、断言 | 输入验证、接口测试 |
| Triton IR 层 | IR dump、`triton-opt` | 编译问题、类型错误 |
| 运行时层 | `tl.debug`、CUDA sanitizer | 内存错误、数值异常 |
| 性能层 | Nsight Compute、binary search | 性能回归、瓶颈定位 |

### 29.1.2 调试基本原则

**最小复现原则**：在调试 Triton kernel 时，首先尝试构造最小可复现用例。将问题隔离到最小的代码单元，避免复杂交互带来的干扰。

**从输出倒推原则**：当 kernel 输出异常时，从输出结果反向追踪，定位问题发生的具体位置。

**分步验证原则**：将复杂的 kernel 拆分为多个简单的子步骤，逐步验证每一步的正确性。

**对比验证原则**：使用 PyTorch 原生实现作为参考基准，对比 Triton kernel 的输出。

---

## 29.2 Print 调试与 `tl.debug`

### 29.2.1 基础 print 调试

Triton 支持在 kernel 中直接使用 `print` 输出调试信息：

```python
import triton
import triton.language as tl

@triton.jit
def debug_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Print 调试信息
    print(f"pid={pid}, offset_start={offsets[0]}, offset_end={offsets[-1]}")
    
    x = tl.load(x_ptr + offsets)
    print(f"loaded x values: {x}")
    
    y = x * 2.0
    print(f"computed y values: {y}")
    
    tl.store(y_ptr + offsets, y)
```

**注意事项**：
- `print` 只能打印程序 ID（`tl.program_id`）和标量值
- 张量的打印需要特殊的处理方式
- Print 输出在 GPU 上，需要同步后才能看到

### 29.2.2 `tl.debug.dump` 详解

`tl.debug` 模块提供了更强大的调试功能：

```python
import triton
import triton.language as tl

@triton.jit
def kernel_with_dump(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 将张量 dump 到文件，供后续检查
    tl.debug.dump(x, f"/tmp/triton_debug/x_tensor_pid{pid}.npy")
    
    y = x * 2.0
    tl.debug.dump(y, f"/tmp/triton_debug/y_tensor_pid{pid}.npy")
    
    tl.store(y_ptr + offsets, y, mask=mask)
```

### 29.2.3 `tl.debug.inspect` 函数

`inspect` 函数可以检查张量的元信息：

```python
@triton.jit
def kernel_with_inspect(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    x = tl.load(x_ptr + offsets)
    
    # 检查张量属性
    print(f"x dtype: {x.dtype}")
    print(f"x shape: {x.shape}")
    print(f"x num_elements: {x.numel()}")
    
    y = x + 1.0
    print(f"y dtype: {y.dtype}")
    print(f"y shape: {y.shape}")
    
    tl.store(y_ptr + offsets, y)
```

### 29.2.4 条件调试

在复杂 kernel 中，通常需要条件触发调试信息：

```python
@triton.jit
def conditional_debug_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 条件调试：只在第一个 program 中打印
    if tl.program_id(axis=0) == 0:
        print(f"First program: x[0]={x[0]}, x[-1]={x[-1]}")
    
    # 条件调试：检查是否有 NaN
    has_nan = tl.any(tl.math.isnan(x))
    if has_nan:
        print(f"Warning: NaN detected in program {pid}")
    
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)
```

---

## 29.3 IR Dump 机制

### 29.3.1 `TRITON_PRINT_IR` 环境变量

Triton 提供了环境变量来控制 IR dump：

```bash
# 打印所有中间 IR
export TRITON_PRINT_IR=1

# 只打印特定阶段的 IR
export TRITON_PRINT_IR=ttir          # 只打印 Triton IR
export TRITON_PRINT_IR=ttgir         # 只打印 TritonGPU IR
export TRITON_PRINT_IR=llir          # 只打印 LLVM IR

# 指定 dump 路径
export TRITON_PRINT_IR=1
export TRITON_PRINT_DIR=/tmp/triton_ir_dump/
```

### 29.3.2 使用 `triton-opt` 工具

`triton-opt` 是一个独立的 IR 优化工具，可以用于检查和转换 IR：

```bash
# 将 Python 转换为 Triton IR
python -c "
import triton
import triton.language as tl

@triton.jit
def simple_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = x * 2.0
    tl.store(y_ptr + offsets, y)
" | triton-opt --triton-convert-to-ttgir

# 直接对 IR 文件进行优化
triton-opt --triton-convert-to-ttgir < input.ttir > output.ttgir
```

### 29.3.3 中间 IR 检查示例

```python
import triton
import triton.language as tl
import os

# 设置 IR dump
os.environ["TRITON_PRINT_IR"] = "1"
os.environ["TRITON_PRINT_DIR"] = "/tmp/triton_ir_debug/"

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_am[:, None] < M) & (offs_cn[None, :] < N))
```

### 29.3.4 IR 检查要点

检查 Triton IR 时，重点关注以下方面：

| 检查项 | 关注内容 | 常见问题 |
|--------|----------|----------|
| 类型一致性 | 所有操作数类型是否匹配 | 类型不匹配导致编译失败 |
| 内存访问 | load/store 的边界检查 | 越界访问、未对齐 |
| 控制流 | if/else 分支是否被正确处理 | 条件编译问题 |
| 循环结构 | for 循环是否正确展开 | 无限循环、性能问题 |
| 计算顺序 | 浮点运算顺序 | 数值精度问题 |

---

## 29.4 内存错误调试

### 29.4.1 Out-of-Bounds (OOB) 错误

OOB 错误是 Triton kernel 中最常见的问题之一：

```python
import triton
import triton.language as tl

@triton.jit
def oob_example_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 错误：没有边界检查
    x = tl.load(x_ptr + offsets)  # 当 offsets >= N 时会越界
    
    # 正确：使用 mask 进行边界检查
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)
```

### 29.4.2 Pointer Aliasing 问题

指针别名（aliasing）是另一个常见问题：

```python
@triton.jit
def aliasing_example_kernel(a_ptr, b_ptr, c_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # 潜在问题：a_ptr 和 b_ptr 可能指向同一内存
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # 如果 a_ptr == b_ptr，读取的值可能不一致
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)
```

### 29.4.3 使用 `boundary_check` 参数

Triton 提供了 `boundary_check` 参数来自动处理边界问题：

```python
@triton.jit
def safe_load_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 使用 boundary_check 自动处理边界
    x = tl.load(x_ptr + offsets, boundary_check=True, padding_option=0)
    
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=offsets < N)
```

### 29.4.4 Compute Sanitizer 检测

使用 NVIDIA Compute Sanitizer 检测内存错误：

```bash
# 运行 memory checker
compute-sanitizer --tool memcheck python my_kernel.py

# 运行 racecheck（检测竞态条件）
compute-sanitizer --tool racecheck python my_kernel.py

# 运行 memcheck 并输出详细信息
compute-sanitizer --tool memcheck --log-file /tmp/memcheck.log python my_kernel.py
```

---

## 29.5 数值精度问题

### 29.5.1 NaN/Inf 检测

在 Triton kernel 中检测 NaN 和 Inf：

```python
import triton
import triton.language as tl

@triton.jit
def nan_detection_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 检测 NaN
    is_nan = tl.math.isnan(x)
    if tl.any(is_nan):
        print(f"NaN detected in program {pid}")
    
    # 检测 Inf
    is_inf = tl.math.isinf(x)
    if tl.any(is_inf):
        print(f"Inf detected in program {pid}")
    
    # 检测负数（如果预期是正数）
    is_negative = x < 0
    if tl.any(is_negative & mask):
        print(f"Negative values detected in program {pid}")
    
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)
```

### 29.5.2 浮点累加误差

浮点数累加会产生精度误差，特别是在大量元素累加时：

```python
@triton.jit
def accumulation_error_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 问题：直接累加可能产生精度误差
    # 对于大数组，误差会累积
    total = tl.sum(x, axis=0)
    
    # 解决方案：使用更高精度进行累加
    total_fp64 = tl.sum(x.to(tl.float64), axis=0)
    
    # 或者使用 Kahan 求和算法（需要手动实现）
    # 这里演示基本思路
    sum_val = tl.zeros([], dtype=tl.float32)
    c = tl.zeros([], dtype=tl.float32)  # 补偿值
    
    for i in tl.static_range(BLOCK_SIZE):
        if i + pid * BLOCK_SIZE < N:
            y = x[i] - c
            t = sum_val + y
            c = (t - sum_val) - y
            sum_val = t
    
    tl.store(y_ptr + pid, sum_val)
```

### 29.5.3 低精度计算的影响

使用 `tf32`、`fp16` 等低精度格式会影响计算精度：

```python
@triton.jit
def precision_comparison_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # 使用 float32 累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # tl.dot 默认使用 tf32 进行计算（如果支持）
        # 如果需要更高精度，可以使用 tl.dot 的 allow_tf32 参数
        accumulator += tl.dot(a, b, allow_tf32=False)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # 最终转换为 fp16 存储
    c = accumulator.to(tl.float16)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_am[:, None] < M) & (offs_cn[None, :] < N))
```

### 29.5.4 精度验证方法

使用 PyTorch 参考实现进行精度验证：

```python
import torch
import triton
import triton.language as tl

def verify_precision(triton_output, torch_output, atol=1e-5, rtol=1e-5, name=""):
    """验证 Triton 输出与 PyTorch 输出的精度差异"""
    diff = torch.abs(triton_output - torch_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"{name} - Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    
    if not torch.allclose(triton_output, torch_output, atol=atol, rtol=rtol):
        print(f"  WARNING: Precision check failed for {name}")
        return False
    return True

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_am[:, None] < M) & (offs_cn[None, :] < N))

def test_matmul_precision(M, N, K, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32):
    # 生成测试数据
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    
    # PyTorch 参考结果
    torch_output = torch.matmul(a, b)
    
    # Triton 计算
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # 验证精度
    verify_precision(c, torch_output, atol=1e-2, rtol=1e-2, name=f"matmul_{M}x{N}x{K}")
```

---

## 29.6 性能调试

### 29.6.1 性能回归定位

当性能下降时，使用二分搜索定位回归 commit：

```bash
#!/bin/bash
# performance_bisect.sh - 性能回归定位脚本

# 定义性能测试函数
run_benchmark() {
    local commit=$1
    git checkout $commit
    
    # 运行性能测试
    python benchmark.py --output /tmp/bench_${commit}.json 2>&1
    
    # 返回性能指标（例如：执行时间）
    python -c "import json; data=json.load(open('/tmp/bench_${commit}.json')); print(data['avg_time'])"
}

# 性能阈值（超过此值认为性能下降）
PERF_THRESHOLD=1.1  # 允许 10% 的波动

# 开始二分搜索
git bisect start

# 标记已知的 good 和 bad commit
git bisect good <known_good_commit>
git bisect bad <known_bad_commit>

# 定义测试脚本
git bisect run bash -c "
    CURRENT_COMMIT=\$(git rev-parse HEAD)
    CURRENT_PERF=\$(run_benchmark \$CURRENT_COMMIT)
    LAST_PERF=\$(run_benchmark <known_good_commit>)
    
    # 比较性能
    python -c \"
import sys
current = float('\$CURRENT_PERF')
last = float('\$LAST_PERF')
ratio = current / last
if ratio > $PERF_THRESHOLD:
    sys.exit(1)  # bad
else:
    sys.exit(0)  # good
\"
"
```

### 29.6.2 调优配置影响分析

分析不同调优配置对性能的影响：

```python
import torch
import triton
import triton.language as tl
import time

@triton.jit
def tuned_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)

def benchmark_config(N, block_sizes, num_warmup=10, num_runs=100):
    """测试不同 BLOCK_SIZE 配置的性能"""
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    
    results = {}
    for BLOCK_SIZE in block_sizes:
        # Warmup
        for _ in range(num_warmup):
            grid = (triton.cdiv(N, BLOCK_SIZE),)
            tuned_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            grid = (triton.cdiv(N, BLOCK_SIZE),)
            tuned_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
        torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / num_runs * 1000  # ms
        results[BLOCK_SIZE] = avg_time
        print(f"BLOCK_SIZE={BLOCK_SIZE}: {avg_time:.3f} ms")
    
    return results

# 运行测试
N = 1024 * 1024
block_sizes = [256, 512, 1024, 2048, 4096]
results = benchmark_config(N, block_sizes)
```

### 29.6.3 Nsight Compute 集成

使用 Nsight Compute 进行详细的性能分析：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def profiled_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)

# 使用 torch.profiler 进行分析
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    N = 1024 * 1024
    BLOCK_SIZE = 1024
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    
    for step in range(5):
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        profiled_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## 29.7 编译错误调试

### 29.7.1 常见编译错误类型

| 错误类型 | 错误信息示例 | 解决方法 |
|----------|--------------|----------|
| 类型不匹配 | `expected 'i32', got 'i64'` | 显式类型转换 |
| 维度不匹配 | `cannot mix scalar and tensor` | 检查张量形状 |
| constexpr 错误 | `non-constexpr argument` | 确保参数为编译期常量 |
| 未定义操作 | `unknown operation: 'tt.load'` | 检查导入和命名空间 |
| GPU 不支持 | `sm_XX not supported` | 检查 GPU 计算能力 |

### 29.7.2 类型不匹配修复

```python
import triton
import triton.language as tl

@triton.jit
def type_mismatch_example(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 错误：int 和 float 类型不匹配
    # index = tl.arange(0, BLOCK_SIZE)  # int32
    # y = x + index  # 类型不匹配
    
    # 正确：显式类型转换
    index = tl.arange(0, BLOCK_SIZE).to(tl.float32)
    y = x + index
    
    tl.store(y_ptr + offsets, y, mask=mask)
```

### 29.7.3 编码不兼容问题

```python
@triton.jit
def encoding_incompatibility_example(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # 问题：不同编码的张量不能直接混合运算
    a = tl.load(a_ptr + tl.arange(0, M))
    b = tl.load(b_ptr + tl.arange(0, N))
    
    # 错误：维度不匹配
    # c = tl.dot(a, b)  # a 是 1D，b 是 1D
    
    # 正确：调整维度
    a_2d = tl.reshape(a, (M, 1))
    b_2d = tl.reshape(b, (1, N))
    c = tl.dot(a_2d, b_2d)
    
    tl.store(c_ptr + tl.arange(0, M * N), tl.reshape(c, (M * N,)))
```

### 29.7.4 使用 `tl.static_range` 优化循环

```python
@triton.jit
def static_range_example(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 使用 tl.static_range 替代 range 可以获得更好的性能
    # tl.static_range 在编译期展开循环
    result = tl.zeros([], dtype=tl.float32)
    for i in tl.static_range(BLOCK_SIZE):
        if i + pid * BLOCK_SIZE < N:
            result += x[i]
    
    tl.store(y_ptr + pid, result)
```

---

## 29.8 调试工作流与最佳实践

### 29.8.1 调试工作流 Checklist

| 步骤 | 操作 | 检查项 |
|------|------|--------|
| 1. 问题确认 | 运行 kernel，确认错误现象 | 输出是否正确？是否有异常？ |
| 2. 最小复现 | 构造最小测试用例 | 能否用最小代码复现问题？ |
| 3. 输入验证 | 检查输入数据 | 数据类型、形状、范围是否正确？ |
| 4. 边界检查 | 添加 mask 和边界检查 | 所有 load/store 都有边界检查？ |
| 5. 类型检查 | 检查类型一致性 | 所有操作的类型是否匹配？ |
| 6. 精度验证 | 与 PyTorch 参考对比 | 输出差异是否在允许范围内？ |
| 7. 性能分析 | 使用 profiler 和 Nsight | 是否有性能瓶颈？ |
| 8. 文档记录 | 记录问题和解决方案 | 便于后续参考 |

### 29.8.2 从简单到复杂的测试策略

```python
import torch
import triton
import triton.language as tl

def test_kernel_simple():
    """简单测试：验证基本功能"""
    N = 256
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    
    @triton.jit
    def simple_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = x * 2.0
        tl.store(y_ptr + offsets, y, mask=mask)
    
    grid = (triton.cdiv(N, 256),)
    simple_kernel[grid](x, y, N, BLOCK_SIZE=256)
    
    assert torch.allclose(y, x * 2.0), "Simple test failed"
    print("Simple test passed")

def test_kernel_medium():
    """中等测试：验证边界条件"""
    # 测试非对齐大小
    N = 1000  # 不是 256 的倍数
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    
    @triton.jit
    def medium_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = x * 2.0
        tl.store(y_ptr + offsets, y, mask=mask)
    
    grid = (triton.cdiv(N, 256),)
    medium_kernel[grid](x, y, N, BLOCK_SIZE=256)
    
    assert torch.allclose(y, x * 2.0), "Medium test failed"
    print("Medium test passed")

def test_kernel_complex():
    """复杂测试：验证性能和精度"""
    M, N, K = 1024, 1024, 512
    
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    
    @triton.jit
    def complex_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        
        c = accumulator.to(tl.float16)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, c, mask=(offs_am[:, None] < M) & (offs_cn[None, :] < N))
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    complex_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
    )
    
    # 验证精度
    torch_ref = torch.matmul(a.float(), b.float()).half()
    assert torch.allclose(c, torch_ref, atol=1e-2, rtol=1e-2), "Complex test failed"
    print("Complex test passed")

# 运行所有测试
if __name__ == "__main__":
    test_kernel_simple()
    test_kernel_medium()
    test_kernel_complex()
```

### 29.8.3 单元测试编写

```python
import pytest
import torch
import triton
import triton.language as tl

class TestTritonKernel:
    """Triton kernel 单元测试框架"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """测试前的设置"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        self.device = 'cuda'
    
    def test_basic_kernel(self):
        """测试基本 kernel 功能"""
        N = 1024
        x = torch.randn(N, device=self.device, dtype=torch.float32)
        y = torch.empty_like(x)
        
        @triton.jit
        def basic_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = x * 2.0
            tl.store(y_ptr + offsets, y, mask=mask)
        
        grid = (triton.cdiv(N, 256),)
        basic_kernel[grid](x, y, N, BLOCK_SIZE=256)
        
        assert torch.allclose(y, x * 2.0, atol=1e-6)
    
    @pytest.mark.parametrize("N", [256, 512, 1000, 1024, 2048])
    def test_various_sizes(self, N):
        """测试不同大小的输入"""
        x = torch.randn(N, device=self.device, dtype=torch.float32)
        y = torch.empty_like(x)
        
        @triton.jit
        def size_test_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = x * 2.0
            tl.store(y_ptr + offsets, y, mask=mask)
        
        grid = (triton.cdiv(N, 256),)
        size_test_kernel[grid](x, y, N, BLOCK_SIZE=256)
        
        assert torch.allclose(y, x * 2.0, atol=1e-6)
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 空输入
        N = 0
        x = torch.randn(N, device=self.device, dtype=torch.float32)
        y = torch.empty_like(x)
        
        @triton.jit
        def edge_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = x * 2.0
            tl.store(y_ptr + offsets, y, mask=mask)
        
        if N == 0:
            # 确保不崩溃
            pass
        else:
            grid = (triton.cdiv(N, 256),)
            edge_kernel[grid](x, y, N, BLOCK_SIZE=256)
            assert torch.allclose(y, x * 2.0, atol=1e-6)
    
    def test_numerical_stability(self):
        """测试数值稳定性"""
        # 测试大数值
        x = torch.tensor([1e38, -1e38, 1e-38, -1e-38], device=self.device, dtype=torch.float32)
        y = torch.empty_like(x)
        
        @triton.jit
        def stability_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = x * 2.0
            tl.store(y_ptr + offsets, y, mask=mask)
        
        grid = (1,)
        stability_kernel[grid](x, y, 4, BLOCK_SIZE=4)
        
        # 检查没有产生 NaN 或 Inf
        assert not torch.any(torch.isnan(y))
        assert not torch.any(torch.isinf(y))
```

---

## 29.9 高级调试技巧

### 29.9.1 调试信息收集脚本

```python
#!/usr/bin/env python3
"""
Triton 调试信息收集脚本
收集 GPU 信息、Triton 版本、编译选项等
"""

import torch
import triton
import triton.runtime
import os
import json

def collect_debug_info():
    """收集调试信息"""
    info = {
        "triton_version": triton.__version__,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = torch.cuda.get_device_capability(0)
        info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    
    # 收集环境变量
    env_vars = [
        "TRITON_PRINT_IR",
        "TRITON_PRINT_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_LAUNCH_BLOCKING",
        "CUDA_MEM_CHECK",
    ]
    for var in env_vars:
        info[var] = os.environ.get(var, "Not set")
    
    return info

def save_debug_info(output_path="/tmp/triton_debug_info.json"):
    """保存调试信息"""
    info = collect_debug_info()
    with open(output_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Debug info saved to {output_path}")
    return info

if __name__ == "__main__":
    info = save_debug_info()
    print(json.dumps(info, indent=2))
```

### 29.9.2 调试输出格式化

```python
import triton
import triton.language as tl
import numpy as np

@triton.jit
def debug_output_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 格式化输出：只在第一个 program 中打印
    if tl.program_id(axis=0) == 0:
        print("=" * 50)
        print(f"Kernel Debug Output")
        print(f"N = {N}, BLOCK_SIZE = {BLOCK_SIZE}")
        print(f"Number of programs = {tl.cdiv(N, BLOCK_SIZE)}")
        print(f"Program 0: x[0]={x[0]}, x[-1]={x[-1]}")
        print(f"Mean = {tl.mean(x, axis=0):.6f}")
        print("=" * 50)
    
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)
```

---

## 29.10 常见问题与解决方案

### 29.10.1 问题分类

| 问题类型 | 典型症状 | 诊断方法 | 解决方案 |
|----------|----------|----------|----------|
| 内存错误 | 程序崩溃、段错误 | Compute Sanitizer | 添加边界检查 |
| 数值错误 | 输出 NaN/Inf、精度差 | 精度验证脚本 | 使用更高精度累加 |
| 编译错误 | 编译失败、类型错误 | IR dump | 检查类型一致性 |
| 性能问题 | 速度慢、不达标 | Nsight Compute | 优化内存访问 |
| 正确性问题 | 输出不匹配 | 对比测试 | 检查算法逻辑 |

### 29.10.2 快速诊断清单

- [ ] 确认输入数据类型和形状是否正确
- [ ] 检查所有 load 操作是否有边界检查
- [ ] 验证输出与 PyTorch 参考实现的差异
- [ ] 检查是否有 NaN 或 Inf 输出
- [ ] 确认 grid 和 block 配置是否合理
- [ ] 检查内存访问模式是否对齐
- [ ] 验证所有 constexpr 参数是否为编译期常量
- [ ] 确认 GPU 计算能力是否支持所需特性

---

## 29.11 调试工具链详解

### 29.11.1 Triton 自带调试工具

Triton 提供了一系列内置调试工具，帮助开发者定位问题：

```python
import triton
import triton.language as tl

# 检查 Triton 版本和配置
def check_triton_config():
    """检查 Triton 配置信息"""
    print(f"Triton version: {triton.__version__}")
    print(f"Default backend: {triton.runtime.BACKEND}")
    
    # 检查编译缓存目录
    cache_dir = triton.runtime.get_cache_dir()
    print(f"Cache directory: {cache_dir}")
    
    # 检查 GPU 信息
    if triton.runtime.is_cuda():
        device = triton.runtime.get_current_device()
        print(f"Device: {device}")
        print(f"Compute capability: {triton.runtime.get_device_capability(device)}")

# 使用装饰器进行性能分析
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
    ],
    key=["N"],
)
@triton.jit
def autotuned_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)
```

### 29.11.2 内存访问模式分析

分析和优化内存访问模式是 Triton 调试的重要环节：

```python
import triton
import triton.language as tl

@triton.jit
def memory_access_analysis_kernel(
    x_ptr, y_ptr, N: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
    STRIDE: tl.constexpr
):
    """分析内存访问模式的 kernel"""
    pid = tl.program_id(axis=0)
    
    # 顺序访问模式（高效）
    sequential_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = sequential_offsets < N
    x_sequential = tl.load(x_ptr + sequential_offsets, mask=mask, other=0.0)
    
    # 跨步访问模式（低效）
    strided_offsets = pid * BLOCK_SIZE * STRIDE + tl.arange(0, BLOCK_SIZE) * STRIDE
    mask_strided = strided_offsets < N
    x_strided = tl.load(x_ptr + strided_offsets, mask=mask_strided, other=0.0)
    
    # 混合访问模式
    mixed_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mixed_offsets = tl.where(mixed_offsets % 2 == 0, mixed_offsets, mixed_offsets * STRIDE)
    mask_mixed = mixed_offsets < N
    x_mixed = tl.load(x_ptr + mixed_offsets, mask=mask_mixed, other=0.0)
    
    # 计算并存储结果
    y = (x_sequential + x_strided + x_mixed) / 3.0
    tl.store(y_ptr + sequential_offsets, y, mask=mask)
```

### 29.11.3 调试信息输出格式化

```python
import triton
import triton.language as tl
from datetime import datetime

@triton.jit
def debug_formatted_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 格式化调试输出
    if tl.program_id(axis=0) == 0:
        print("=" * 60)
        print("Triton Kernel Debug Information")
        print("=" * 60)
        print(f"N = {N}, BLOCK_SIZE = {BLOCK_SIZE}")
        print(f"Grid size = {tl.cdiv(N, BLOCK_SIZE)}")
        print(f"Total elements = {N}")
        print(f"Programs launched = {tl.num_programs(axis=0)}")
        print("-" * 60)
        print("Input statistics:")
        print(f"  x[0] = {x[0]:.6f}")
        print(f"  x[-1] = {x[-1]:.6f}")
        print(f"  x.mean() = {tl.mean(x, axis=0):.6f}")
        print(f"  x.std() = {tl.sqrt(tl.var(x, axis=0)):.6f}")
        print(f"  x.min() = {tl.min(x, axis=0):.6f}")
        print(f"  x.max() = {tl.max(x, axis=0):.6f}")
        print("=" * 60)
    
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)
```

---

## 29.12 高级调试场景

### 29.12.1 多 Kernel 协作调试

当多个 kernel 协作完成计算时，调试变得更加复杂：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def reduction_kernel_1(x_ptr, partial_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """第一阶段：部分归约"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 计算部分和
    partial_sum = tl.sum(x, axis=0)
    
    # 存储部分结果
    if tl.program_id(axis=0) == 0:
        print(f"Program {pid}: partial_sum = {partial_sum:.6f}")
    
    tl.store(partial_ptr + pid, partial_sum)

@triton.jit
def reduction_kernel_2(partial_ptr, result_ptr, NUM_BLOCKS: tl.constexpr):
    """第二阶段：最终归约"""
    # 加载所有部分结果
    partial_sums = tl.load(partial_ptr + tl.arange(0, NUM_BLOCKS))
    
    # 计算最终结果
    final_sum = tl.sum(partial_sums, axis=0)
    
    print(f"Final reduction result: {final_sum:.6f}")
    tl.store(result_ptr, final_sum)

def multi_kernel_reduction(x):
    """多 kernel 协作的归约示例"""
    N = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    # 分配部分结果存储
    partial = torch.empty(num_blocks, device='cuda', dtype=torch.float32)
    result = torch.empty(1, device='cuda', dtype=torch.float32)
    
    # 第一阶段
    reduction_kernel_1[(num_blocks,)](
        x, partial, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 第二阶段
    reduction_kernel_2[(1,)](
        partial, result, NUM_BLOCKS=num_blocks
    )
    
    return result
```

### 29.12.2 动态形状调试

处理动态形状时需要特别注意：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def dynamic_shape_kernel(
    x_ptr, y_ptr, 
    N,  # 运行时传入，不是 constexpr
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)

def test_dynamic_shapes():
    """测试不同形状的输入"""
    test_cases = [
        (128, "Small"),
        (1024, "Medium"),
        (10000, "Large"),
        (100000, "Very large"),
        (1, "Single element"),
        (0, "Empty"),
    ]
    
    for N, desc in test_cases:
        print(f"\nTesting {desc} (N={N})...")
        
        if N == 0:
            print("  Skipping empty input")
            continue
        
        x = torch.randn(N, device='cuda', dtype=torch.float32)
        y = torch.empty_like(x)
        
        grid = (triton.cdiv(N, 256),)
        dynamic_shape_kernel[grid](x, y, N, BLOCK_SIZE=256)
        
        # 验证结果
        expected = x * 2.0
        if torch.allclose(y, expected, atol=1e-6):
            print(f"  PASSED")
        else:
            print(f"  FAILED: max diff = {(y - expected).abs().max().item():.6e}")
```

### 29.12.3 条件编译调试

使用 constexpr 进行条件编译：

```python
import triton
import triton.language as tl

@triton.jit
def conditional_compilation_kernel(
    x_ptr, y_ptr, N: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
    DEBUG_MODE: tl.constexpr = False,
    USE_FAST_MATH: tl.constexpr = True
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if DEBUG_MODE:
        print(f"Debug: pid={pid}, x[0]={x[0]}, x[-1]={x[-1]}")
    
    if USE_FAST_MATH:
        # 使用快速数学运算
        y = tl.math.sqrt(x * x + 1.0)
    else:
        # 使用精确数学运算
        y = tl.sqrt(x * x + 1.0)
    
    if DEBUG_MODE:
        print(f"Debug: y[0]={y[0]}, y[-1]={y[-1]}")
    
    tl.store(y_ptr + offsets, y, mask=mask)

# 使用不同的编译配置
def test_conditional_compilation():
    x = torch.randn(1024, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    
    # 调试模式
    conditional_compilation_kernel[(1,)](
        x, y, 1024, BLOCK_SIZE=1024, 
        DEBUG_MODE=True, USE_FAST_MATH=True
    )
    
    # 生产模式
    conditional_compilation_kernel[(1,)](
        x, y, 1024, BLOCK_SIZE=1024,
        DEBUG_MODE=False, USE_FAST_MATH=False
    )
```

---

## 29.13 错误处理与恢复

### 29.13.1 错误检测与报告

```python
import torch
import triton
import triton.language as tl

class TritonDebugger:
    """Triton 调试器类"""
    
    def __init__(self, enable_debug=True):
        self.enable_debug = enable_debug
        self.errors = []
        self.warnings = []
    
    def log_error(self, message):
        """记录错误"""
        self.errors.append(message)
        if self.enable_debug:
            print(f"[ERROR] {message}")
    
    def log_warning(self, message):
        """记录警告"""
        self.warnings.append(message)
        if self.enable_debug:
            print(f"[WARNING] {message}")
    
    def check_tensor(self, tensor, name, expected_dtype=None, expected_device=None):
        """检查张量属性"""
        if tensor is None:
            self.log_error(f"{name} is None")
            return False
        
        if expected_dtype and tensor.dtype != expected_dtype:
            self.log_error(f"{name} dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")
            return False
        
        if expected_device and tensor.device.type != expected_device:
            self.log_error(f"{name} device mismatch: expected {expected_device}, got {tensor.device.type}")
            return False
        
        if torch.any(torch.isnan(tensor)):
            self.log_warning(f"{name} contains NaN")
        
        if torch.any(torch.isinf(tensor)):
            self.log_warning(f"{name} contains Inf")
        
        return True
    
    def compare_outputs(self, triton_output, torch_output, name, atol=1e-5, rtol=1e-5):
        """比较 Triton 和 PyTorch 输出"""
        if not self.check_tensor(triton_output, f"{name}_triton"):
            return False
        
        if not self.check_tensor(torch_output, f"{name}_torch"):
            return False
        
        if triton_output.shape != torch_output.shape:
            self.log_error(f"{name} shape mismatch: {triton_output.shape} vs {torch_output.shape}")
            return False
        
        diff = torch.abs(triton_output - torch_output)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        self.log_warning(f"{name} max diff: {max_diff:.6e}, mean diff: {mean_diff:.6e}")
        
        if not torch.allclose(triton_output, torch_output, atol=atol, rtol=rtol):
            self.log_error(f"{name} precision check failed")
            return False
        
        return True
    
    def get_summary(self):
        """获取调试摘要"""
        return {
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "error_messages": self.errors,
            "warning_messages": self.warnings
        }
```

### 29.13.2 自动化测试框架

```python
import torch
import triton
import triton.language as tl
from typing import Callable, Tuple, List
import pytest

class TritonTestFramework:
    """Triton 测试框架"""
    
    def __init__(self):
        self.test_results = []
    
    def run_test(
        self,
        kernel_fn: Callable,
        reference_fn: Callable,
        test_inputs: Tuple,
        test_name: str,
        rtol: float = 1e-5,
        atol: float = 1e-5
    ):
        """运行单个测试"""
        try:
            # 运行 Triton kernel
            triton_output = kernel_fn(*test_inputs)
            
            # 运行参考实现
            torch_output = reference_fn(*test_inputs)
            
            # 比较结果
            if torch.allclose(triton_output, torch_output, rtol=rtol, atol=atol):
                self.test_results.append({
                    "name": test_name,
                    "status": "PASSED",
                    "max_diff": (triton_output - torch_output).abs().max().item()
                })
                return True
            else:
                self.test_results.append({
                    "name": test_name,
                    "status": "FAILED",
                    "max_diff": (triton_output - torch_output).abs().max().item()
                })
                return False
        except Exception as e:
            self.test_results.append({
                "name": test_name,
                "status": "ERROR",
                "error": str(e)
            })
            return False
    
    def run_test_suite(
        self,
        kernel_fn: Callable,
        reference_fn: Callable,
        test_cases: List[Tuple[str, Tuple]]
    ):
        """运行测试套件"""
        print("Running test suite...")
        print("=" * 60)
        
        for test_name, test_inputs in test_cases:
            print(f"Testing: {test_name}...", end=" ")
            self.run_test(kernel_fn, reference_fn, test_inputs, test_name)
        
        print("=" * 60)
        self.print_summary()
    
    def print_summary(self):
        """打印测试摘要"""
        passed = sum(1 for r in self.test_results if r["status"] == "PASSED")
        failed = sum(1 for r in self.test_results if r["status"] == "FAILED")
        errors = sum(1 for r in self.test_results if r["status"] == "ERROR")
        
        print(f"\nTest Summary:")
        print(f"  Total: {len(self.test_results)}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Errors: {errors}")
        
        if failed > 0 or errors > 0:
            print("\nFailed/Error tests:")
            for r in self.test_results:
                if r["status"] != "PASSED":
                    print(f"  - {r['name']}: {r['status']}")
                    if "error" in r:
                        print(f"    Error: {r['error']}")
                    if "max_diff" in r:
                        print(f"    Max diff: {r['max_diff']:.6e}")
```

---

## 29.14 调试最佳实践总结

### 29.14.1 调试检查表

| 类别 | 检查项 | 优先级 |
|------|--------|--------|
| 输入验证 | 数据类型正确 | 高 |
| 输入验证 | 数据形状正确 | 高 |
| 输入验证 | 数据范围合理 | 中 |
| 边界检查 | 所有 load 有 mask | 高 |
| 边界检查 | 所有 store 有 mask | 高 |
| 边界检查 | mask 覆盖所有访问 | 高 |
| 类型一致 | 操作数类型匹配 | 高 |
| 类型一致 | 累加器精度足够 | 中 |
| 类型一致 | 输出类型正确 | 中 |
| 性能优化 | 内存访问对齐 | 中 |
| 性能优化 | 减少 bank conflict | 中 |
| 性能优化 | 合理使用共享内存 | 低 |
| 正确性 | 与参考实现对比 | 高 |
| 正确性 | 边界条件测试 | 中 |
| 正确性 | 大规模数据测试 | 中 |

### 29.14.2 调试流程图

```
开始调试
    │
    ▼
确认错误现象
    │
    ├─ 编译错误 → 检查 IR，修复类型/语法
    │
    ├─ 运行时错误 → 使用 Compute Sanitizer
    │
    ├─ 数值错误 → 检查精度，对比参考实现
    │
    └─ 性能问题 → 使用 Profiler 分析
    │
    ▼
构造最小复现用例
    │
    ▼
分步调试
    │
    ├─ 打印中间值
    │
    ├─ 检查 IR
    │
    └─ 验证边界条件
    │
    ▼
定位问题根因
    │
    ▼
修复并验证
    │
    ▼
记录解决方案
```

### 29.14.3 调试技巧速查表

| 技巧 | 使用场景 | 命令/代码 |
|------|----------|-----------|
| Print 调试 | 快速定位问题 | `print(f"x={x}")` |
| IR Dump | 编译问题 | `TRITON_PRINT_IR=1` |
| Compute Sanitizer | 内存错误 | `compute-sanitizer --tool memcheck` |
| Nsight Compute | 性能分析 | `ncu --set full python script.py` |
| Binary Search | 定位回归 | `git bisect` |
| 对比测试 | 正确性验证 | `torch.allclose()` |
| 条件调试 | 复杂场景 | `if tl.program_id(axis=0) == 0:` |
| 边界检查 | 内存安全 | `mask = offsets < N` |

---

## 29.15 实战案例

### 29.15.1 案例：矩阵乘法精度问题

**问题描述**：Triton 实现的矩阵乘法在某些情况下精度不足。

**调试过程**：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_debug_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    DEBUG: tl.constexpr = False
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # 使用 float32 累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        if DEBUG and pid == 0:
            print(f"k={k}, a mean={tl.mean(a):.6f}, b mean={tl.mean(b):.6f}")
        
        accumulator += tl.dot(a, b)
        
        if DEBUG and pid == 0:
            print(f"accumulator mean={tl.mean(accumulator):.6f}")
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_am[:, None] < M) & (offs_cn[None, :] < N))

def debug_matmul_precision(M, N, K):
    """调试矩阵乘法精度"""
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    
    # PyTorch 参考
    torch_ref = torch.matmul(a.float(), b.float()).half()
    
    # Triton 计算
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    matmul_debug_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        DEBUG=True
    )
    
    # 分析精度差异
    diff = (c.float() - torch_ref.float()).abs()
    print(f"\nPrecision analysis:")
    print(f"  Max diff: {diff.max().item():.6e}")
    print(f"  Mean diff: {diff.mean().item():.6e}")
    print(f"  Std diff: {diff.std().item():.6e}")
    
    # 检查是否有 NaN/Inf
    print(f"  Triton output has NaN: {torch.any(torch.isnan(c))}")
    print(f"  Triton output has Inf: {torch.any(torch.isinf(c))}")
    
    return diff

# 运行调试
debug_matmul_precision(1024, 1024, 512)
```

### 29.15.2 案例：性能回归调试

**问题描述**：kernel 性能在某个 commit 后下降了 20%。

```python
import torch
import triton
import triton.language as tl
import time

@triton.jit
def performance_test_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 模拟计算
    for _ in range(10):
        x = x * 2.0 + 1.0
    
    tl.store(y_ptr + offsets, x, mask=mask)

def benchmark_kernel(N, BLOCK_SIZE, num_warmup=10, num_runs=100):
    """基准测试 kernel 性能"""
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Warmup
    for _ in range(num_warmup):
        performance_test_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        performance_test_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_runs * 1000  # ms
    return avg_time

def performance_regression_test():
    """性能回归测试"""
    print("Performance regression test")
    print("=" * 60)
    
    # 测试不同配置
    configs = [
        (1024 * 1024, 256),
        (1024 * 1024, 512),
        (1024 * 1024, 1024),
        (2 * 1024 * 1024, 256),
        (2 * 1024 * 1024, 512),
    ]
    
    results = {}
    for N, BLOCK_SIZE in configs:
        time_ms = benchmark_kernel(N, BLOCK_SIZE)
        results[(N, BLOCK_SIZE)] = time_ms
        print(f"N={N}, BLOCK_SIZE={BLOCK_SIZE}: {time_ms:.3f} ms")
    
    print("=" * 60)
    
    # 分析结果
    print("\nPerformance analysis:")
    for (N, BLOCK_SIZE), time_ms in results.items():
        throughput = N / time_ms / 1e6  # M elements/ms
        print(f"  N={N}, BLOCK_SIZE={BLOCK_SIZE}: {throughput:.2f} M elements/ms")

# 运行测试
performance_regression_test()
```

### 29.15.3 案例：内存错误调试

**问题描述**：kernel 在特定输入大小上崩溃。

```python
import torch
import triton
import triton.language as tl

@triton.jit
def memory_error_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 问题：没有边界检查
    # 当 N 不是 BLOCK_SIZE 的倍数时，会越界访问
    x = tl.load(x_ptr + offsets)  # 错误！
    
    y = x * 2.0
    tl.store(y_ptr + offsets, y)  # 错误！没有边界检查

@triton.jit
def memory_error_fixed_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N  # 添加边界检查
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)  # 正确
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)  # 正确

def test_memory_error():
    """测试内存错误"""
    print("Testing memory error...")
    
    # 使用非对齐大小触发错误
    N = 1000  # 不是 256 的倍数
    BLOCK_SIZE = 256
    
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    
    # 测试有错误的版本（可能会崩溃）
    try:
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        memory_error_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
        print("  No crash (unexpected)")
    except Exception as e:
        print(f"  Crash detected: {e}")
    
    # 测试修复后的版本
    try:
        y_fixed = torch.empty_like(x)
        memory_error_fixed_kernel[grid](x, y_fixed, N, BLOCK_SIZE=BLOCK_SIZE)
        
        # 验证结果
        expected = x * 2.0
        if torch.allclose(y_fixed, expected, atol=1e-6):
            print("  Fixed version: PASSED")
        else:
            print("  Fixed version: FAILED")
    except Exception as e:
        print(f"  Fixed version error: {e}")

# 运行测试
test_memory_error()
```

---

## 本章小结

本章全面介绍了 Triton kernel 的调试技术和常见问题排查方法：

1. **调试方法论**：建立了分层调试模型，从 Python 层到性能层，提供了系统化的调试思路。

2. **Print 调试与 tl.debug**：详细介绍了基础 print 调试、`tl.debug.dump` 和 `inspect` 函数的使用方法，以及条件调试技巧。

3. **IR Dump 机制**：深入讲解了 `TRITON_PRINT_IR` 环境变量、`triton-opt` 工具和中间 IR 检查方法。

4. **内存错误调试**：介绍了 OOB 错误、pointer aliasing 问题、`boundary_check` 使用和 Compute Sanitizer 检测。

5. **数值精度问题**：讲解了 NaN/Inf 检测、浮点累加误差、低精度计算影响和精度验证方法。

6. **性能调试**：介绍了性能回归定位、调优配置影响分析和 Nsight Compute 集成。

7. **编译错误调试**：总结了常见编译错误类型、类型不匹配修复和编码不兼容问题。

8. **最佳实践**：提供了调试工作流 checklist、从简单到复杂的测试策略和单元测试编写方法。

9. **高级调试技巧**：介绍了调试工具链、内存访问模式分析、调试信息格式化等高级技巧。

10. **实战案例**：通过矩阵乘法精度问题、性能回归调试、内存错误调试等案例，展示了调试技术的实际应用。

掌握这些调试技术，可以帮助开发者快速定位和解决 Triton kernel 开发中遇到的各种问题，提高开发效率和代码质量。

---

## 思考题

1. **问题描述**：假设你有一个 Triton kernel 在小规模数据上运行正常，但在大规模数据上出现 NaN 输出。请设计一个系统化的调试流程来定位问题。

2. **IR 检查**：当使用 `TRITON_PRINT_IR=1` 导出 IR 后，你会检查哪些关键点来判断 IR 是否正确？请列出至少 5 个检查项。

3. **精度验证**：编写一个测试函数，验证 Triton 实现的矩阵乘法与 PyTorch 参考实现的精度差异，要求支持不同大小的矩阵。

4. **性能回归**：当发现某个 commit 导致性能下降时，除了二分搜索，还有什么其他方法可以快速定位回归原因？

5. **调试工具选择**：针对以下三种问题，分别选择最合适的调试工具：
   - Kernel 输出与 PyTorch 参考不匹配
   - Kernel 在特定 GPU 上崩溃
   - Kernel 运行速度比预期慢 5 倍

6. **边界条件测试**：设计一组测试用例，覆盖以下边界条件：
   - N = 0（空输入）
   - N 不是 BLOCK_SIZE 的倍数
   - N 远大于 BLOCK_SIZE * grid_size
   - 输入数据包含 NaN 和 Inf

7. **调试脚本编写**：编写一个调试脚本，自动收集以下信息：Triton 版本、PyTorch 版本、CUDA 版本、GPU 信息、环境变量设置，并将结果保存到 JSON 文件。

8. **多 Kernel 调试**：当多个 Triton kernel 协作完成计算时，如何调试中间结果的正确性？请设计一个调试方案。

9. **动态形状处理**：在处理动态形状输入时，可能会遇到哪些调试挑战？如何解决？

10. **调试效率优化**：如何提高 Triton kernel 的调试效率？请从工具使用、测试策略、代码组织等方面给出建议。

---

## 扩展阅读

- [Triton 官方文档 - Debugging](https://triton-lang.org/latest/debugging.html)
- [NVIDIA Compute Sanitizer 文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/Text/index.html)
- [Nsight Compute 用户指南](https://docs.nvidia.com/nsight-compute/)
- [Triton IR 规范](https://github.com/triton-lang/triton/blob/main/docs/triton_introduction.md)
- [CUDA 调试最佳实践](https://developer.nvidia.com/blog/effective-debugging-cuda-code/)
- [GPU 编程调试技巧](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
