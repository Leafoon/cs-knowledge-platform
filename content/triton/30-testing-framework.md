# Chapter 30: 测试框架与质量保障

> **学习目标**：
> - 掌握 Triton 的测试体系（单元测试、回归测试、性能测试）
> - 了解 TritonBench 性能基准测试框架
> - 掌握正确性验证（reference implementation）的方法
> - 了解 CI/CD 流程与自动化测试

---

## 30.1 Triton 测试架构概览

### 30.1.1 测试体系全景

Triton 作为高性能编译器框架，其测试体系是保障代码质量和性能稳定性的核心基础设施。与传统软件不同，GPU 编程框架的测试面临独特挑战：硬件依赖性、数值精度问题、性能回归检测等。

```
Triton 测试体系
├── 单元测试 (Unit Tests)
│   ├── Python 前端测试
│   ├── IR 转换测试
│   ├── 编译管线测试
│   └── Kernel 功能测试
├── 正确性验证 (Correctness Verification)
│   ├── Reference Implementation 对比
│   ├── 数值精度测试 (Tolerance)
│   └── 边界条件测试
├── 性能基准测试 (Performance Benchmarking)
│   ├── TritonBench
│   ├── triton.testing.Benchmark
│   └── 回归检测
├── 集成测试 (Integration Tests)
│   ├── 多后端测试
│   ├── 跨硬件兼容性
│   └── API 兼容性
└── CI/CD 流程
    ├── GitHub Actions
    ├── 多硬件矩阵
    └── 自动化流水线
```

### 30.1.2 仓库目录结构

Triton 的测试代码主要分布在 `python/test/` 目录下，采用 pytest 框架组织。

```
triton/
├── python/
│   ├── test/
│   │   ├── __init__.py
│   │   ├── conftest.py                 # pytest 配置与 fixtures
│   │   ├── pytest.ini                  # pytest 配置文件
│   │   │
│   │   ├── unit/                       # 单元测试
│   │   │   ├── test_basic.py           # 基础功能测试
│   │   │   ├── test_math.py            # 数学运算测试
│   │   │   ├── test_random.py          # 随机数生成测试
│   │   │   └── ...
│   │   │
│   │   ├── kernel/                     # Kernel 功能测试
│   │   │   ├── test_gemm.py            # 矩阵乘法测试
│   │   │   ├── test_softmax.py         # Softmax 测试
│   │   │   ├── test_attention.py       # Attention 测试
│   │   │   ├── test_reduce.py          # 归约操作测试
│   │   │   └── ...
│   │   │
│   │   ├── compiler/                   # 编译器测试
│   │   │   ├── test_codegen.py         # 代码生成测试
│   │   │   ├── test_optimization.py    # 优化 Pass 测试
│   │   │   └── ...
│   │   │
│   │   ├── language/                   # 语言特性测试
│   │   │   ├── test_types.py           # 类型系统测试
│   │   │   ├── test_pointer.py         # 指针操作测试
│   │   │   └── ...
│   │   │
│   │   └── backends/                   # 后端测试
│   │       ├── test_cuda.py            # NVIDIA 后端测试
│   │       ├── test_hip.py             # AMD 后端测试
│   │       └── ...
│   │
│   ├── triton/
│   │   └── testing.py                  # 测试工具模块
│   │
│   └── benchmarks/                     # 性能基准测试
│       ├── matmul/                     # 矩阵乘法基准
│       ├── attention/                  # Attention 基准
│       └── ...
│
└── ci/                                 # CI 配置
    ├── .github/
    │   └── workflows/
    │       ├── tests.yml               # 主测试流水线
    │       ├── benchmarks.yml          # 性能基准流水线
    │       └── nightly.yml             # 每日构建流水线
    └── Dockerfile
```

### 30.1.3 pytest 配置

Triton 使用 pytest 作为测试框架，通过 `conftest.py` 和 `pytest.ini` 进行配置。

```python
# conftest.py - Triton 测试的全局配置
import pytest
import torch
import triton


def pytest_configure(config):
    """注册自定义标记"""
    config.addinivalue_line(
        "markers", "noautotest: skip tests that require autotune"
    )
    config.addinivalue_line(
        "markers", "small: tests that run quickly"
    )
    config.addinivalue_line(
        "markers", "large: tests that require significant memory"
    )


def pytest_collection_modifyitems(config, items):
    """根据硬件能力自动标记 skip"""
    if not torch.cuda.is_available():
        skip_no_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            item.add_marker(skip_no_cuda)
    else:
        device = torch.cuda.get_device_name()
        if "A100" not in device:
            skip_no_a100 = pytest.mark.skip(reason="Requires A100")
            for item in items:
                if "a100_only" in item.keywords:
                    item.add_marker(skip_no_a100)


@pytest.fixture
def device():
    """提供默认 CUDA 设备"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    pytest.skip("CUDA not available")


@pytest.fixture
def random_seed():
    """设置可重现的随机种子"""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    yield
    torch.cuda.empty_cache()


@pytest.fixture(params=["float16", "float32", "bfloat16"])
def dtype(request):
    """参数化 dtype fixture"""
    return getattr(torch, request.param)
```

```ini
# pytest.ini
[pytest]
testpaths = python/test
python_files = test_*.py
python_functions = test_*
markers =
    noautotest: skip tests that require autotune
    small: tests that run quickly
    large: tests that require significant memory
    a100_only: tests that require A100 GPU
addopts = -v --tb=short
timeout = 300
```

### 30.1.4 测试组织原则

Triton 的测试组织遵循以下原则：

```
测试金字塔
        ╱╲
       ╱  ╲
      ╱ E2E╲          少量端到端测试
     ╱──────╲
    ╱ 集成测试 ╲       中等数量集成测试
   ╱────────────╲
  ╱   单元测试    ╲     大量单元测试
 ╱────────────────╲
```

| 测试层级 | 数量 | 执行时间 | 覆盖范围 |
|---------|------|---------|---------|
| 单元测试 | 多 | 毫秒级 | 单个函数/操作 |
| 集成测试 | 中 | 秒级 | 多个组件交互 |
| Kernel 测试 | 中 | 秒级 | 完整 kernel |
| 端到端测试 | 少 | 分钟级 | 完整流程 |

---

## 30.2 正确性验证方法论

### 30.2.1 Reference Implementation 对比

正确性验证是 Triton 测试的核心。基本思路是：**将 Triton kernel 的输出与参考实现（通常是 PyTorch）的输出进行对比**。

```python
import torch
import triton
import triton.language as tl


# Triton kernel 实现
@triton.jit
def _add_kernel(
    X, Y, Z,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = tl.load(Y + offs, mask=mask)
    z = x + y
    tl.store(Z + offs, z, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    z = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _add_kernel[grid](x, y, z, N=n, BLOCK_SIZE=1024)
    return z


# 正确性验证
def test_add_basic():
    """基础正确性测试"""
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    y = torch.randn(1024, device="cuda", dtype=torch.float32)
    
    # Triton 结果
    z_triton = add(x, y)
    
    # PyTorch 参考实现
    z_ref = x + y
    
    # 对比
    torch.testing.assert_close(z_triton, z_ref, rtol=1e-5, atol=1e-5)
```

### 30.2.2 数值精度与容差设置

GPU 浮点运算存在非确定性，需要合理设置容差参数：

```python
import torch
import triton.testing


def test_precision_levels():
    """不同精度等级的容差设置"""
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    y = torch.randn(1024, device="cuda", dtype=torch.float32)
    
    z_triton = add(x, y)
    z_ref = x + y
    
    # 1. 严格模式 - 逐位匹配（适用于整数或确定性操作）
    torch.testing.assert_close(
        z_triton.to(torch.int32),
        z_ref.to(torch.int32),
        rtol=0,
        atol=0
    )
    
    # 2. 标准模式 - 相对/绝对容差（适用于 FP32）
    torch.testing.assert_close(
        z_triton, z_ref,
        rtol=1e-5,  # 相对容差
        atol=1e-5   # 绝对容差
    )
    
    # 3. 宽松模式 - 适用于 FP16/BF16
    torch.testing.assert_close(
        z_triton.half(), z_ref.half(),
        rtol=1e-2,
        atol=1e-2
    )
    
    # 4. 使用 triton.testing.assert_close（更灵活）
    triton.testing.assert_close(
        z_triton, z_ref,
        rtol=1e-5,
        atol=1e-5,
        msg="Add kernel mismatch"
    )


def test_tolerance_for_accumulation():
    """累加操作需要更宽松的容差"""
    # 矩阵乘法涉及大量累加，精度损失更大
    M, N, K = 1024, 1024, 1024
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    
    # Triton GEMM
    z_triton = triton_matmul(a, b)
    
    # PyTorch 参考
    z_ref = torch.mm(a, b)
    
    # FP16 矩阵乘法需要更宽松的容差
    # 随着 K 增大，累加误差增大
    rtol = 1e-2 if K <= 1024 else 5e-2
    torch.testing.assert_close(
        z_triton, z_ref,
        rtol=rtol,
        atol=rtol
    )


def test_absolute_vs_relative_tolerance():
    """理解绝对容差与相对容差的区别"""
    # 绝对容差 (atol): |a - b| <= atol
    # 相对容差 (rtol): |a - b| <= rtol * max(|a|, |b|)
    
    a = torch.tensor([1000.0], device="cuda")
    b = torch.tensor([1000.5], device="cuda")
    
    # rtol=1e-3 可以通过 (差异 0.5 < 1000 * 1e-3 = 1.0)
    torch.testing.assert_close(a, b, rtol=1e-3, atol=0)
    
    # atol=1.0 可以通过 (差异 0.5 < 1.0)
    torch.testing.assert_close(a, b, rtol=0, atol=1.0)
    
    # 小数值的情况
    a_small = torch.tensor([0.001], device="cuda")
    b_small = torch.tensor([0.002], device="cuda")
    
    # rtol=1e-3 失败 (差异 0.001 > 0.002 * 1e-3)
    # 需要 atol=0.001 来通过
    torch.testing.assert_close(a_small, b_small, rtol=0, atol=0.001)
```

### 30.2.3 完整的验证框架

```python
import torch
import triton
import triton.testing
from dataclasses import dataclass
from typing import Callable, Optional
import sys


@dataclass
class CorrectnessConfig:
    """正确性验证配置"""
    rtol: float = 1e-5
    atol: float = 1e-5
    max_abs_diff: Optional[float] = None
    max_rel_diff: Optional[float] = None
    check_inf: bool = True
    check_nan: bool = True
    msg: str = ""


class CorrectnessVerifier:
    """Triton kernel 正确性验证器"""
    
    def __init__(self, reference_fn: Callable):
        self.reference_fn = reference_fn
        self.results = []
    
    def verify(
        self,
        triton_fn: Callable,
        config: CorrectnessConfig,
        *args,
        **kwargs
    ):
        """执行一次验证"""
        # Triton 结果
        triton_result = triton_fn(*args, **kwargs)
        
        # 参考结果
        ref_result = self.reference_fn(*args, **kwargs)
        
        # 检查类型和形状
        assert triton_result.shape == ref_result.shape, \
            f"Shape mismatch: {triton_result.shape} vs {ref_result.shape}"
        assert triton_result.dtype == ref_result.dtype, \
            f"Dtype mismatch: {triton_result.dtype} vs {ref_result.dtype}"
        
        # 检查 NaN/Inf
        if config.check_nan:
            assert not torch.isnan(triton_result).any(), \
                "Triton result contains NaN"
        if config.check_inf:
            assert not torch.isinf(triton_result).any(), \
                "Triton result contains Inf"
        
        # 数值对比
        torch.testing.assert_close(
            triton_result, ref_result,
            rtol=config.rtol,
            atol=config.atol,
            msg=config.msg or "Correctness check failed"
        )
        
        # 额外统计
        abs_diff = (triton_result - ref_result).abs()
        rel_diff = abs_diff / (ref_result.abs() + 1e-8)
        
        stats = {
            "max_abs_diff": abs_diff.max().item(),
            "mean_abs_diff": abs_diff.mean().item(),
            "max_rel_diff": rel_diff.max().item(),
            "mean_rel_diff": rel_diff.mean().item(),
        }
        
        if config.max_abs_diff:
            assert stats["max_abs_diff"] <= config.max_abs_diff, \
                f"Max absolute diff {stats['max_abs_diff']} exceeds {config.max_abs_diff}"
        
        if config.max_rel_diff:
            assert stats["max_rel_diff"] <= config.max_rel_diff, \
                f"Max relative diff {stats['max_rel_diff']} exceeds {config.max_rel_diff}"
        
        self.results.append(stats)
        return stats
    
    def summary(self):
        """输出验证摘要"""
        if not self.results:
            print("No verification results")
            return
        
        max_abs = max(r["max_abs_diff"] for r in self.results)
        max_rel = max(r["max_rel_diff"] for r in self.results)
        mean_abs = sum(r["mean_abs_diff"] for r in self.results) / len(self.results)
        
        print(f"Verification Summary ({len(self.results)} tests):")
        print(f"  Max absolute diff: {max_abs:.6e}")
        print(f"  Max relative diff: {max_rel:.6e}")
        print(f"  Mean absolute diff: {mean_abs:.6e}")
        print(f"  All passed: ✓")


# 使用示例
def test_softmax_kernel():
    """验证 Softmax kernel"""
    verifier = CorrectnessVerifier(
        reference_fn=lambda x: torch.softmax(x, dim=-1)
    )
    
    config = CorrectnessConfig(
        rtol=1e-4,
        atol=1e-4,
        msg="Softmax verification failed"
    )
    
    for size in [256, 1024, 4096]:
        x = torch.randn(16, size, device="cuda", dtype=torch.float16)
        verifier.verify(triton_softmax, config, x)
    
    verifier.summary()
```

### 30.2.4 多维度验证策略

```python
import torch
import random


def test_kernel_comprehensive():
    """综合验证策略"""
    torch.manual_seed(42)
    
    # 1. 多种 shape 测试
    shapes = [
        (1, 1),           # 最小
        (1, 1024),        # 向量
        (1024, 1),        # 转置向量
        (64, 64),         # 小矩阵
        (1024, 1024),     # 中等矩阵
        (4096, 4096),     # 大矩阵
        (1, 4096),        # 行向量
        (4096, 1),        # 列向量
        (128, 384),       # 非对齐
        (100, 100),       # 非 2 的幂
    ]
    
    # 2. 多种数据类型
    dtypes = [torch.float16, torch.float32, torch.bfloat16]
    
    # 3. 多种数值范围
    value_ranges = [
        (lambda s: torch.randn(s), "normal"),
        (lambda s: torch.rand(s), "uniform"),
        (lambda s: torch.zeros(s), "zeros"),
        (lambda s: torch.ones(s), "ones"),
        (lambda s: torch.full(s, 1e6), "large"),
        (lambda s: torch.full(s, 1e-6), "small"),
    ]
    
    # 4. 多种设备（如果可用）
    devices = ["cuda:0"]
    if torch.cuda.device_count() > 1:
        devices.append("cuda:1")
    
    for shape in shapes:
        for dtype in dtypes:
            for gen_fn, gen_name in value_ranges:
                for device in devices:
                    try:
                        x = gen_fn(shape).to(device=device, dtype=dtype)
                        y = gen_fn(shape).to(device=device, dtype=dtype)
                        
                        # 执行 Triton kernel
                        z_triton = my_kernel(x, y)
                        
                        # PyTorch 参考
                        z_ref = x + y  # 或对应的参考操作
                        
                        # 根据 dtype 设置容差
                        rtol, atol = {
                            torch.float16: (1e-2, 1e-2),
                            torch.bfloat16: (1e-2, 1e-2),
                            torch.float32: (1e-5, 1e-5),
                        }[dtype]
                        
                        torch.testing.assert_close(
                            z_triton, z_ref,
                            rtol=rtol, atol=atol,
                            msg=f"Failed: shape={shape}, dtype={dtype}, "
                                f"gen={gen_name}, device={device}"
                        )
                    except Exception as e:
                        print(f"Test failed: {e}")
                        raise


def test_boundary_conditions():
    """边界条件测试"""
    # 空输入
    # 注：大多数 kernel 不支持空输入，应确保不崩溃
    
    # 单元素
    x = torch.tensor([42.0], device="cuda")
    y = torch.tensor([1.0], device="cuda")
    z = my_kernel(x, y)
    assert z.item() == 43.0
    
    # 非常大的值
    x = torch.full((1024,), 1e30, device="cuda")
    y = torch.full((1024,), 1e30, device="cuda")
    z = my_kernel(x, y)
    # 检查是否溢出
    assert not torch.isinf(z).any() or not torch.isinf(x + y).any()
    
    # 非常小的值
    x = torch.full((1024,), 1e-30, device="cuda")
    y = torch.full((1024,), 1e-30, device="cuda")
    z = my_kernel(x, y)
    ref = x + y
    torch.testing.assert_close(z, ref, rtol=1e-2, atol=1e-30)
    
    # NaN 和 Inf 处理
    x = torch.tensor([float("nan"), float("inf"), float("-inf")], device="cuda")
    y = torch.zeros(3, device="cuda")
    z = my_kernel(x, y)
    ref = x + y
    # NaN 传播
    assert torch.isnan(z[0])
    # Inf 传播
    assert z[1] == float("inf")
    assert z[2] == float("-inf")
```

---

## 30.3 单元测试编写指南

### 30.3.1 Kernel 测试标准模板

```python
import torch
import pytest
import triton
import triton.language as tl


# ===== 被测 Kernel =====
@triton.jit
def _vector_add_kernel(
    X_ptr, Y_ptr, Z_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """向量加法 kernel"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask)
    y = tl.load(Y_ptr + offs, mask=mask)
    z = x + y
    tl.store(Z_ptr + offs, z, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """向量加法 Python 包装"""
    assert x.shape == y.shape
    n = x.numel()
    z = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _vector_add_kernel[grid](
        x, y, z,
        N=n,
        BLOCK_SIZE=1024,
    )
    return z


# ===== 测试类 =====
class TestVectorAdd:
    """向量加法 kernel 测试套件"""
    
    def test_basic(self, device):
        """基础功能测试"""
        x = torch.randn(1024, device=device)
        y = torch.randn(1024, device=device)
        z = vector_add(x, y)
        torch.testing.assert_close(z, x + y)
    
    def test_different_sizes(self, device):
        """不同大小测试"""
        for n in [1, 16, 256, 1024, 4096, 10000]:
            x = torch.randn(n, device=device)
            y = torch.randn(n, device=device)
            z = vector_add(x, y)
            torch.testing.assert_close(z, x + y)
    
    def test_different_dtypes(self, device):
        """不同数据类型测试"""
        for dtype in [torch.float16, torch.float32, torch.bfloat16]:
            x = torch.randn(1024, device=device, dtype=dtype)
            y = torch.randn(1024, device=device, dtype=dtype)
            z = vector_add(x, y)
            rtol, atol = {
                torch.float16: (1e-2, 1e-2),
                torch.bfloat16: (1e-2, 1e-2),
                torch.float32: (1e-5, 1e-5),
            }[dtype]
            torch.testing.assert_close(z, x + y, rtol=rtol, atol=atol)
    
    def test_2d_tensor(self, device):
        """2D 张量测试"""
        x = torch.randn(128, 256, device=device)
        y = torch.randn(128, 256, device=device)
        z = vector_add(x, y)
        torch.testing.assert_close(z, x + y)
    
    def test_negative_values(self, device):
        """负数测试"""
        x = torch.randn(1024, device=device) * 100
        y = torch.randn(1024, device=device) * -100
        z = vector_add(x, y)
        torch.testing.assert_close(z, x + y)
    
    def test_large_tensor(self, device):
        """大张量测试"""
        x = torch.randn(1024 * 1024, device=device)
        y = torch.randn(1024 * 1024, device=device)
        z = vector_add(x, y)
        torch.testing.assert_close(z, x + y)
    
    def test_same_tensor(self, device):
        """同一张量测试"""
        x = torch.randn(1024, device=device)
        z = vector_add(x, x)
        torch.testing.assert_close(z, x + x)


# ===== 参数化测试 =====
@pytest.mark.parametrize("n", [1, 64, 256, 1024, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_vector_add_parametrized(device, n, dtype):
    """参数化测试"""
    x = torch.randn(n, device=device, dtype=dtype)
    y = torch.randn(n, device=device, dtype=dtype)
    z = vector_add(x, y)
    
    rtol = 1e-2 if dtype == torch.float16 else 1e-5
    atol = 1e-2 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(z, x + y, rtol=rtol, atol=atol)


# ===== 标记测试 =====
@pytest.mark.small
def test_vector_add_small(device):
    """小规模测试（快速）"""
    x = torch.randn(256, device=device)
    y = torch.randn(256, device=device)
    z = vector_add(x, y)
    torch.testing.assert_close(z, x + y)


@pytest.mark.large
def test_vector_add_large(device):
    """大规模测试（慢速）"""
    x = torch.randn(1024 * 1024 * 16, device=device)
    y = torch.randn(1024 * 1024 * 16, device=device)
    z = vector_add(x, y)
    torch.testing.assert_close(z, x + y)
```

### 30.3.2 参数化测试详解

```python
import pytest
import torch
import triton
import triton.language as tl


# 使用 pytest.mark.parametrize 进行参数化
@pytest.mark.parametrize("M,N,K", [
    (32, 32, 32),       # 小矩阵
    (128, 128, 128),    # 中等
    (1024, 1024, 1024), # 大矩阵
    (256, 512, 1024),   # 非方阵
    (1, 128, 128),      # 行向量
    (128, 1, 128),      # 列向量
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_gemm(device, M, N, K, dtype):
    """参数化 GEMM 测试"""
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    
    z_triton = triton_matmul(a, b)
    z_ref = torch.mm(a, b.to(dtype))
    
    rtol = 1e-2 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(z_triton, z_ref, rtol=rtol, atol=rtol)


# 使用 pytest.fixture 进行参数化
@pytest.fixture(params=["float16", "float32", "bfloat16"])
def(request):
    return getattr(torch, request.param)


# 使用 product 组合参数
@pytest.mark.parametrize("block_m", [64, 128])
@pytest.mark.parametrize("block_n", [64, 128])
@pytest.mark.parametrize("block_k", [32, 64])
def test_gemm_block_sizes(device, block_m, block_n, block_k):
    """测试不同分块大小"""
    M, N, K = 256, 256, 256
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)
    
    z_triton = triton_matmul(a, b, block_m=block_m, block_n=block_n, block_k=block_k)
    z_ref = torch.mm(a, b)
    
    torch.testing.assert_close(z_triton, z_ref, rtol=1e-2, atol=1e-2)


# 条件跳过
@pytest.mark.skipif(
    torch.cuda.get_device_capability() < (8, 0),
    reason="Requires Ampere or newer GPU"
)
def test_fp8_gemm(device):
    """FP8 GEMM 测试（仅 Ampere+）"""
    # FP8 测试代码


@pytest.mark.skipif(
    not hasattr(torch, 'float8_e4m3fn'),
    reason="Requires PyTorch 2.1+"
)
def test_torch_float8(device):
    """PyTorch FP8 支持测试"""


# xfail 标记预期失败
@pytest.mark.xfail(reason="Known issue with large matrices")
def test_gemm_large():
    """已知问题的测试"""
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    # 可能失败的代码


# 自定义标记
@pytest.mark.slow
@pytest.mark.gpu
def test_benchmark():
    """性能基准测试"""
    pass


# 使用 fixture 管理测试状态
@pytest.fixture
def random_matrices():
    """生成随机矩阵对"""
    torch.manual_seed(42)
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    return a, b


def test_gemm_with_fixture(random_matrices):
    """使用 fixture 的测试"""
    a, b = random_matrices
    z = triton_matmul(a, b)
    z_ref = torch.mm(a, b)
    torch.testing.assert_close(z, z_ref, rtol=1e-2, atol=1e-2)
```

### 30.3.3 异常测试

```python
import pytest
import torch
import triton


def test_invalid_input_shapes():
    """无效输入形状测试"""
    a = torch.randn(128, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.float16)
    
    # 维度不匹配应抛出异常
    with pytest.raises(RuntimeError, match="dimension"):
        triton_matmul(a, b)


def test_cpu_tensor_error():
    """CPU 张量错误处理"""
    a = torch.randn(128, 128)  # CPU tensor
    b = torch.randn(128, 128)
    
    with pytest.raises(AssertionError, match="device"):
        triton_matmul(a, b)


def test_mixed_dtypes():
    """混合数据类型处理"""
    a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(128, 128, device="cuda", dtype=torch.float32)
    
    # 应抛出类型不匹配异常或自动转换
    with pytest.raises(TypeError):
        triton_matmul(a, b)


def test_empty_tensor():
    """空张量处理"""
    a = torch.empty(0, device="cuda", dtype=torch.float32)
    b = torch.empty(0, device="cuda", dtype=torch.float32)
    
    # 空张量应返回空张量或抛出异常
    try:
        z = vector_add(a, b)
        assert z.numel() == 0
    except Exception:
        pass  # 或者预期抛出异常


def test_kernel_timeout():
    """Kernel 执行超时测试"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Kernel execution timed out")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # 5 秒超时
    
    try:
        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")
        z = vector_add(x, y)
    finally:
        signal.alarm(0)  # 取消定时器
```

### 30.3.4 测试辅助工具

```python
import torch
import triton
import triton.testing
import functools


def skip_if_no_cuda(fn):
    """跳过没有 CUDA 的环境"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return fn(*args, **kwargs)
    return wrapper


def skip_if_capability_below(min_capability):
    """跳过 GPU 能力不足的情况"""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                if capability < min_capability:
                    pytest.skip(
                        f"Requires capability {min_capability}, "
                        f"got {capability}"
                    )
            return fn(*args, **kwargs)
        return wrapper
    return decorator


@skip_if_capability_below((8, 0))
def test_ampere_feature():
    """仅在 Ampere+ 上运行的测试"""
    pass


class TestUtils:
    """测试辅助工具类"""
    
    @staticmethod
    def generate_test_tensor(
        shape,
        dtype=torch.float32,
        device="cuda",
        distribution="normal",
        seed=42,
    ):
        """生成测试张量"""
        torch.manual_seed(seed)
        
        if distribution == "normal":
            return torch.randn(shape, dtype=dtype, device=device)
        elif distribution == "uniform":
            return torch.rand(shape, dtype=dtype, device=device)
        elif distribution == "zeros":
            return torch.zeros(shape, dtype=dtype, device=device)
        elif distribution == "ones":
            return torch.ones(shape, dtype=dtype, device=device)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    @staticmethod
    def assert_kernel_output_close(
        triton_fn,
        ref_fn,
        inputs,
        rtol=1e-5,
        atol=1e-5,
        msg="",
    ):
        """通用 kernel 输出对比"""
        triton_result = triton_fn(*inputs)
        ref_result = ref_fn(*inputs)
        
        torch.testing.assert_close(
            triton_result, ref_result,
            rtol=rtol,
            atol=atol,
            msg=msg,
        )
    
    @staticmethod
    def get_memory_usage():
        """获取当前 GPU 内存使用"""
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**2,
            "cached": torch.cuda.memory_reserved() / 1024**2,
        }


# 使用示例
def test_with_utils(device):
    """使用工具类的测试"""
    utils = TestUtils()
    
    x = utils.generate_test_tensor((1024,), seed=1)
    y = utils.generate_test_tensor((1024,), seed=2)
    
    mem_before = utils.get_memory_usage()
    
    z = vector_add(x, y)
    
    mem_after = utils.get_memory_usage()
    
    utils.assert_kernel_output_close(
        vector_add,
        lambda x, y: x + y,
        (x, y),
        rtol=1e-5,
        atol=1e-5,
    )
```

---

## 30.4 性能基准测试

### 30.4.1 triton.testing.Benchmark 类

Triton 提供了内置的性能基准测试工具，用于系统化地测量和比较 kernel 性能。

```python
import torch
import triton
import triton.testing
import triton.language as tl


# Benchmark 配置
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],           # 参数名
        x_vals=[128 * i for i in range(1, 8)],  # 参数值
        x_log=False,                         # 是否对数坐标
        line_arg="provider",                 # 区分不同实现的参数
        line_vals=["triton", "torch"],       # 不同实现的标签
        line_names=["Triton", "PyTorch"],    # 显示名称
        styles=[("blue", "-"), ("green", "--")],  # 线条样式
        ylabel="GB/s",                       # Y 轴标签
        plot_name="matmul-performance",      # 图表名称
        args={},                             # 额外参数
    )
)
def benchmark_matmul(M, N, K, provider):
    """矩阵乘法性能基准"""
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    
    if provider == "triton":
        return triton_matmul(a, b)
    elif provider == "torch":
        return torch.mm(a, b)


# 运行基准测试
benchmark_matmul.run(
    show_plots=True,   # 显示图表
    print_data=True,   # 打印数据
    save_path="./benchmarks"  # 保存路径
)
```

### 30.4.2 Benchmark 配置详解

```python
import triton.testing
import torch


# 详细配置示例
benchmark_config = triton.testing.Benchmark(
    # ===== 参数配置 =====
    x_names=["M", "N", "K"],           # X 轴参数名
    x_vals=[256 * i for i in range(1, 9)],  # X 轴参数值: 256, 512, ..., 2048
    x_log=False,                         # X 轴是否对数
    x_range=None,                        # X 轴范围
    
    # ===== 线条配置 =====
    line_arg="provider",                 # 区分实现的参数名
    line_vals=["triton", "torch", "cuda"],  # 实现标识
    line_names=["Triton", "PyTorch", "CUDA"],  # 显示名称
    styles=[
        ("blue", "-"),    # Triton: 蓝色实线
        ("green", "--"),  # PyTorch: 绿色虚线
        ("red", "-."),    # CUDA: 红色点划线
    ],
    
    # ===== 标签配置 =====
    ylabel="Throughput (GB/s)",          # Y 轴标签
    xlabel="Matrix Size (M=N=K)",        # X 轴标签
    plot_name="matmul-benchmark",        # 图表文件名
    title="Matrix Multiplication Performance",  # 图表标题
    
    # ===== 额外参数 =====
    args={},                             # 传递给 benchmark 函数的额外参数
    
    # ===== 格式配置 =====
    ylabel_spec=":.2f",                  # Y 轴数字格式
    args_spec="",                        # 参数显示格式
)


# 多维度 Benchmark
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["BLOCK_SIZE"],
        x_vals=[32, 64, 128, 256, 512],
        x_log=True,  # 对数坐标
        line_arg="num_warps",
        line_vals=[2, 4, 8],
        line_names=["2 warps", "4 warps", "8 warps"],
        styles=[("blue", "-"), ("green", "--"), ("red", "-.")],
        ylabel="GB/s",
        plot_name="kernel-block-size-sweep",
    )
)
def benchmark_sweep(BLOCK_SIZE, num_warps):
    """参数扫描基准测试"""
    n = 1024 * 1024
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    
    # 带有 autotune 参数的 kernel
    z = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    my_kernel[grid](x, z, N=n, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return z
```

### 30.4.3 自动化性能报告

```python
import torch
import triton
import triton.testing
import json
import os
from datetime import datetime


class PerformanceReporter:
    """自动化性能报告生成器"""
    
    def __init__(self, output_dir="./benchmarks"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
    
    def add_result(self, name, metrics):
        """添加性能结果"""
        self.results.append({
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "metrics": metrics,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A",
        })
    
    def benchmark_kernel(
        self,
        kernel_fn,
        input_fn,
        n_warmup=10,
        n_repeat=100,
        name="kernel",
    ):
        """通用 kernel 基准测试"""
        # Warmup
        for _ in range(n_warmup):
            inputs = input_fn()
            kernel_fn(*inputs)
        
        torch.cuda.synchronize()
        
        # 计时
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
        
        latencies = []
        for i in range(n_repeat):
            inputs = input_fn()
            start_events[i].record()
            kernel_fn(*inputs)
            end_events[i].record()
        
        torch.cuda.synchronize()
        
        for i in range(n_repeat):
            latencies.append(start_events[i].elapsed_time(end_events[i]))
        
        import statistics
        metrics = {
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "std_ms": statistics.stdev(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "n_repeat": n_repeat,
        }
        
        self.add_result(name, metrics)
        return metrics
    
    def save_report(self, filename="report.json"):
        """保存报告"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Report saved to {filepath}")
    
    def print_summary(self):
        """打印摘要"""
        print("\n" + "=" * 60)
        print("Performance Summary")
        print("=" * 60)
        for result in self.results:
            m = result["metrics"]
            print(f"{result['name']:30s} | "
                  f"{m['mean_ms']:8.3f} ms ± {m['std_ms']:.3f} ms")
        print("=" * 60)


# 使用示例
def test_performance():
    """性能测试示例"""
    reporter = PerformanceReporter()
    
    # 测试不同大小
    for n in [1024, 4096, 16384]:
        reporter.benchmark_kernel(
            kernel_fn=lambda x, y: vector_add(x, y),
            input_fn=lambda: (
                torch.randn(n, device="cuda"),
                torch.randn(n, device="cuda"),
            ),
            n_warmup=5,
            n_repeat=50,
            name=f"vector_add_n={n}",
        )
    
    reporter.print_summary()
    reporter.save_report()
```

---

## 30.5 TritonBench 性能基准平台

### 30.5.1 TritonBench 概述

TritonBench 是 Triton 社区维护的性能基准测试平台，用于系统化地评估各种 kernel 实现的性能。

```
TritonBench 架构
├── 核心组件
│   ├── Benchmark Runner     基准测试执行器
│   ├── Result Collector     结果收集器
│   ├── Report Generator     报告生成器
│   └── History Manager      历史数据管理
├── 测试类型
│   ├── Compute Benchmarks   计算基准
│   ├── Memory Benchmarks    内存基准
│   ├── Bandwidth Benchmarks 带宽基准
│   └── Latency Benchmarks   延迟基准
└── 硬件支持
    ├── NVIDIA (CUDA)
    ├── AMD (ROCm/HIP)
    └── Intel (SYCL)
```

### 30.5.2 TritonBench 使用方法

```python
import triton
import torch
import triton.language as tl


# 定义 Triton kernel
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, c)


def matmul(a, b):
    """Python 包装"""
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_SIZE_M = 8
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return c


# TritonBench 测试定义
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[128 * i for i in range(1, 33)],
        x_log=False,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="TFLOPs/s",
        plot_name="matmul-bench-f16",
        args={"N": 4096, "K": 4096},
    )
)
def matmul_benchmark(M, N, K, provider):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    
    if provider == "triton":
        return matmul(a, b)
    else:
        return torch.mm(a, b)


# 运行基准测试
if __name__ == "__main__":
    matmul_benchmark.run(
        show_plots=True,
        print_data=True,
        save_path="./tritonbench_results"
    )
```

### 30.5.3 测试矩阵定义

```python
import itertools
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class BenchmarkMatrix:
    """测试矩阵定义"""
    name: str
    shapes: List[Tuple[int, int, int]]
    dtypes: List[str]
    backends: List[str]
    block_sizes: List[int]
    num_warps: List[int]
    
    def generate_configs(self) -> List[Dict]:
        """生成所有配置组合"""
        configs = []
        for shape, dtype, backend, block_size, num_warps in itertools.product(
            self.shapes, self.dtypes, self.backends,
            self.block_sizes, self.num_warps
        ):
            configs.append({
                "name": f"{self.name}_{shape}_{dtype}_{backend}_b{block_size}_w{num_warps}",
                "M": shape[0],
                "N": shape[1],
                "K": shape[2],
                "dtype": dtype,
                "backend": backend,
                "block_size": block_size,
                "num_warps": num_warps,
            })
        return configs


# 定义测试矩阵
matmul_matrix = BenchmarkMatrix(
    name="matmul",
    shapes=[
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (256, 4096, 4096),   # 窄矩阵
        (4096, 256, 4096),   # 宽矩阵
        (4096, 4096, 256),   # 浅矩阵
    ],
    dtypes=["float16", "bfloat16", "float32"],
    backends=["cuda"],
    block_sizes=[64, 128, 256],
    num_warps=[4, 8],
)


# 生成配置
configs = matmul_matrix.generate_configs()
print(f"Total configurations: {len(configs)}")
for config in configs[:5]:
    print(f"  {config['name']}")
```

### 30.5.4 结果对比与分析

```python
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    config: Dict
    throughput: float  # GFLOPs/s
    latency_ms: float
    bandwidth_gb_s: float
    timestamp: str


class BenchmarkAnalyzer:
    """基准测试结果分析器"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def load_results(self, filepath: str):
        """加载结果文件"""
        with open(filepath) as f:
            data = json.load(f)
            for entry in data:
                self.results.append(BenchmarkResult(**entry))
    
    def compare(
        self,
        baseline_name: str,
        target_name: str,
        metric: str = "throughput",
    ) -> Dict:
        """对比两个实现的性能"""
        baseline = [r for r in self.results if r.name == baseline_name]
        target = [r for r in self.results if r.name == target_name]
        
        if not baseline or not target:
            raise ValueError("No matching results found")
        
        # 计算平均性能比
        ratios = []
        for b, t in zip(baseline, target):
            if metric == "throughput":
                ratio = t.throughput / b.throughput
            elif metric == "latency":
                ratio = b.latency_ms / t.latency_ms
            else:
                raise ValueError(f"Unknown metric: {metric}")
            ratios.append(ratio)
        
        return {
            "mean_ratio": np.mean(ratios),
            "std_ratio": np.std(ratios),
            "min_ratio": np.min(ratios),
            "max_ratio": np.max(ratios),
            "speedup": f"{np.mean(ratios):.2f}x",
        }
    
    def plot_comparison(
        self,
        names: List[str],
        metric: str = "throughput",
        save_path: Optional[str] = None,
    ):
        """绘制性能对比图"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(names))
        width = 0.35
        
        for i, name in enumerate(names):
            results = [r for r in self.results if r.name == name]
            values = [getattr(r, metric) for r in results]
            ax.bar(x + i * width, values, width, label=name)
        
        ax.set_xlabel("Test Case")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title("Performance Comparison")
        ax.set_xticks(x + width * (len(names) - 1) / 2)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
    
    def generate_report(self) -> str:
        """生成文本报告"""
        lines = ["=" * 70, "Performance Report", "=" * 70, ""]
        
        # 按名称分组
        by_name = {}
        for r in self.results:
            if r.name not in by_name:
                by_name[r.name] = []
            by_name[r.name].append(r)
        
        for name, results in by_name.items():
            throughputs = [r.throughput for r in results]
            latencies = [r.latency_ms for r in results]
            
            lines.append(f"Provider: {name}")
            lines.append(f"  Throughput: {np.mean(throughputs):.2f} ± {np.std(throughputs):.2f} GFLOPs/s")
            lines.append(f"  Latency:    {np.mean(latencies):.3f} ± {np.std(latencies):.3f} ms")
            lines.append("")
        
        lines.append("=" * 70)
        return "\n".join(lines)


# 使用示例
def analyze_benchmarks():
    """分析基准测试结果"""
    analyzer = BenchmarkAnalyzer()
    
    # 加载结果
    analyzer.load_results("./benchmarks/results.json")
    
    # 对比
    comparison = analyzer.compare("torch", "triton", metric="throughput")
    print(f"Speedup: {comparison['speedup']}")
    
    # 生成报告
    report = analyzer.generate_report()
    print(report)
    
    # 绘图
    analyzer.plot_comparison(
        ["torch", "triton"],
        metric="throughput",
        save_path="./benchmarks/comparison.png"
    )
```

---

## 30.6 回归测试

### 30.6.1 性能回归检测

性能回归是指代码修改后，性能意外下降。Triton 使用历史数据对比来检测回归。

```python
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class PerformanceBaseline:
    """性能基线"""
    name: str
    throughput: float
    timestamp: str
    git_commit: str
    hardware: str


class RegressionDetector:
    """性能回归检测器"""
    
    def __init__(self, baseline_file: str = "performance_baselines.json"):
        self.baseline_file = baseline_file
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self._load_baselines()
    
    def _load_baselines(self):
        """加载历史基线"""
        if os.path.exists(self.baseline_file):
            with open(self.baseline_file) as f:
                data = json.load(f)
                for name, entry in data.items():
                    self.baselines[name] = PerformanceBaseline(**entry)
    
    def _save_baselines(self):
        """保存基线"""
        data = {}
        for name, baseline in self.baselines.items():
            data[name] = {
                "name": baseline.name,
                "throughput": baseline.throughput,
                "timestamp": baseline.timestamp,
                "git_commit": baseline.git_commit,
                "hardware": baseline.hardware,
            }
        with open(self.baseline_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def update_baseline(self, name: str, throughput: float, git_commit: str):
        """更新基线"""
        self.baselines[name] = PerformanceBaseline(
            name=name,
            throughput=throughput,
            timestamp=datetime.now().isoformat(),
            git_commit=git_commit,
            hardware=self._get_hardware_info(),
        )
        self._save_baselines()
    
    def check_regression(
        self,
        name: str,
        current_throughput: float,
        threshold: float = 0.9,  # 允许 10% 的波动
    ) -> Dict:
        """检查是否有性能回归"""
        if name not in self.baselines:
            return {
                "regression": False,
                "message": f"No baseline found for {name}, creating new baseline",
                "create_baseline": True,
            }
        
        baseline = self.baselines[name]
        ratio = current_throughput / baseline.throughput
        
        if ratio < threshold:
            return {
                "regression": True,
                "message": (
                    f"Performance regression detected for {name}: "
                    f"{baseline.throughput:.2f} -> {current_throughput:.2f} "
                    f"({ratio:.2%} of baseline)"
                ),
                "baseline": baseline.throughput,
                "current": current_throughput,
                "ratio": ratio,
                "threshold": threshold,
            }
        
        return {
            "regression": False,
            "message": (
                f"Performance OK for {name}: "
                f"{current_throughput:.2f} GFLOPs/s "
                f"({ratio:.2%} of baseline)"
            ),
            "baseline": baseline.throughput,
            "current": current_throughput,
            "ratio": ratio,
        }
    
    @staticmethod
    def _get_hardware_info() -> str:
        """获取硬件信息"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name()
        return "N/A"
    
    def generate_report(self) -> str:
        """生成回归检测报告"""
        lines = [
            "=" * 60,
            "Performance Regression Report",
            "=" * 60,
            f"Hardware: {self._get_hardware_info()}",
            f"Baselines: {len(self.baselines)}",
            "",
        ]
        
        for name, baseline in self.baselines.items():
            lines.append(
                f"  {name:40s} | {baseline.throughput:8.2f} GFLOPs/s | "
                f"{baseline.git_commit[:8]}"
            )
        
        lines.append("=" * 60)
        return "\n".join(lines)


# 使用示例
def test_performance_regression():
    """性能回归检测测试"""
    detector = RegressionDetector()
    
    # 运行当前版本的性能测试
    current_throughput = run_benchmark()
    
    # 检查回归
    result = detector.check_regression(
        "matmul_fp16",
        current_throughput,
        threshold=0.95,  # 允许 5% 的波动
    )
    
    if result["regression"]:
        print(f"WARNING: {result['message']}")
        # 在 CI 中可以设置为失败
        raise AssertionError(result["message"])
    else:
        print(result["message"])
    
    # 更新基线（如果是新硬件或有意的性能变化）
    if result.get("create_baseline"):
        detector.update_baseline(
            "matmul_fp16",
            current_throughput,
            git_commit=get_current_commit()
        )
```

### 30.6.2 CI 中的性能基准

```yaml
# .github/workflows/benchmarks.yml
name: Performance Benchmarks

on:
  push:
    branches: [main, release/*]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # 每天凌晨 2 点运行

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    strategy:
      matrix:
        gpu: [A100, A10, RTX4090]
        python-version: ['3.10']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e python/
        pip install pytest pytest-benchmark
    
    - name: Run benchmarks
      run: |
        python -m pytest python/benchmarks/ \
          --benchmark-json=benchmark_results.json \
          --benchmark-compare
    
    - name: Check performance regression
      run: |
        python -c "
        import json
        import sys
        
        with open('benchmark_results.json') as f:
            results = json.load(f)
        
        # 加载基线
        with open('performance_baselines.json') as f:
            baselines = json.load(f)
        
        failed = False
        for bench in results['benchmarks']:
            name = bench['name']
            current = bench['stats']['mean']
            
            if name in baselines:
                baseline = baselines[name]['throughput']
                ratio = current / baseline
                
                if ratio < 0.95:  # 5% 回归阈值
                    print(f'REGRESSION: {name}: {ratio:.2%} of baseline')
                    failed = True
        
        if failed:
            sys.exit(1)
        "
    
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results-${{ matrix.gpu }}
        path: benchmark_results.json
    
    - name: Update baselines (main branch only)
      if: github.ref == 'refs/heads/main' && success()
      run: |
        python -c "
        import json
        
        with open('benchmark_results.json') as f:
            results = json.load(f)
        
        baselines = {}
        for bench in results['benchmarks']:
            baselines[bench['name']] = {
                'throughput': bench['stats']['mean'],
                'timestamp': '${{ github.sha }}',
            }
        
        with open('performance_baselines.json', 'w') as f:
            json.dump(baselines, f, indent=2)
        "
    
    - name: Commit baselines
      if: github.ref == 'refs/heads/main' && success()
      run: |
        git config user.name "github-actions[bot]"
        git config user.email "github-actions[bot]@users.noreply.github.com"
        git add performance_baselines.json
        git diff --staged --quiet || git commit -m "Update performance baselines [skip ci]"
        git push
```

### 30.6.3 阈值设定策略

```python
import numpy as np
from typing import Dict, Tuple


class ThresholdManager:
    """性能回归阈值管理器"""
    
    def __init__(self):
        # 默认阈值配置
        self.thresholds = {
            # 计算密集型 kernel - 更严格的阈值
            "gemm": {"relative": 0.05, "absolute": 0.1},
            "conv": {"relative": 0.05, "absolute": 0.1},
            
            # 内存密集型 kernel - 中等阈值
            "softmax": {"relative": 0.10, "absolute": 0.2},
            "layernorm": {"relative": 0.10, "absolute": 0.2},
            
            # 延迟敏感型 kernel - 更宽松的阈值
            "attention": {"relative": 0.15, "absolute": 0.3},
            
            # 默认阈值
            "default": {"relative": 0.10, "absolute": 0.2},
        }
    
    def get_threshold(self, kernel_name: str) -> Dict[str, float]:
        """获取 kernel 的阈值"""
        for key, threshold in self.thresholds.items():
            if key in kernel_name.lower():
                return threshold
        return self.thresholds["default"]
    
    def should_fail(
        self,
        kernel_name: str,
        baseline: float,
        current: float,
    ) -> Tuple[bool, str]:
        """判断是否应该标记为回归"""
        threshold = self.get_threshold(kernel_name)
        
        # 相对差异
        relative_diff = abs(current - baseline) / baseline
        
        # 绝对差异（GFLOPs/s）
        absolute_diff = abs(current - baseline)
        
        # 检查是否超过阈值
        if relative_diff > threshold["relative"]:
            return True, (
                f"Relative regression: {relative_diff:.2%} > {threshold['relative']:.2%} "
                f"(baseline={baseline:.2f}, current={current:.2f})"
            )
        
        if absolute_diff > threshold["absolute"]:
            return True, (
                f"Absolute regression: {absolute_diff:.2f} GFLOPs/s > "
                f"{threshold['absolute']:.2f} GFLOPs/s"
            )
        
        return False, (
            f"Performance OK: {current:.2f} GFLOPs/s "
            f"({relative_diff:.2%} change from baseline {baseline:.2f})"
        )
    
    def configure_threshold(
        self,
        kernel_pattern: str,
        relative: float,
        absolute: float,
    ):
        """配置自定义阈值"""
        self.thresholds[kernel_pattern] = {
            "relative": relative,
            "absolute": absolute,
        }
    
    def analyze_history(
        self,
        history: Dict[str, List[float]],
        window: int = 10,
    ) -> Dict[str, Dict]:
        """分析历史数据，自动设定阈值"""
        suggestions = {}
        
        for name, values in history.items():
            if len(values) < window:
                continue
            
            recent = values[-window:]
            mean = np.mean(recent)
            std = np.std(recent)
            
            # 基于标准差自动设定阈值
            suggested_relative = max(0.05, 3 * std / mean)  # 至少 5%，或 3 倍标准差
            
            suggestions[name] = {
                "mean": mean,
                "std": std,
                "suggested_relative": suggested_relative,
                "suggested_absolute": 3 * std,
                "data_points": len(recent),
            }
        
        return suggestions


# 使用示例
def test_threshold_management():
    """阈值管理测试"""
    manager = ThresholdManager()
    
    # GEMM 测试
    should_fail, message = manager.should_fail(
        "matmul_fp16",
        baseline=100.0,  # 100 GFLOPs/s
        current=95.0,     # 95 GFLOPs/s
    )
    print(f"GEMM: {should_fail}, {message}")
    # 输出: GEMM: False, Performance OK: 95.00 GFLOPs/s (5.00% change...)
    
    # Softmax 测试
    should_fail, message = manager.should_fail(
        "softmax_kernel",
        baseline=50.0,
        current=44.0,  # 12% 下降
    )
    print(f"Softmax: {should_fail}, {message}")
    # 输出: Softmax: True, Relative regression: 12.00% > 10.00%...
```

---

## 30.7 CI/CD 集成

### 30.7.1 GitHub Actions 配置

```yaml
# .github/workflows/tests.yml
name: Triton Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:

env:
  TRITON_CACHE_DIR: ${{ github.workspace }}/.triton_cache

jobs:
  # ===== 单元测试 =====
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e "python[tests]"
    
    - name: Run unit tests
      run: |
        python -m pytest python/test/unit/ \
          -v \
          --tb=short \
          --junitxml=test-results/unit-tests.xml \
          -x  # 遇到第一个失败就停止
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: unit-test-results-${{ matrix.python-version }}
        path: test-results/

  # ===== GPU 测试 =====
  gpu-tests:
    runs-on: [self-hosted, gpu, "${{ matrix.gpu }}"]
    needs: unit-tests
    strategy:
      fail-fast: false
      matrix:
        gpu: [A100, A10, RTX4090]
        include:
          - gpu: A100
            cuda-version: '12.1'
          - gpu: A10
            cuda-version: '12.1'
          - gpu: RTX4090
            cuda-version: '12.1'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up environment
      run: |
        nvidia-smi
        python --version
    
    - name: Install Triton
      run: |
        pip install -e python/
        pip install -e "python[tests]"
    
    - name: Run kernel tests
      run: |
        python -m pytest python/test/kernel/ \
          -v \
          --tb=short \
          --timeout=300 \
          --junitxml=test-results/kernel-tests-${{ matrix.gpu }}.xml
    
    - name: Run compiler tests
      run: |
        python -m pytest python/test/compiler/ \
          -v \
          --tb=short \
          --junitxml=test-results/compiler-tests-${{ matrix.gpu }}.xml
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: gpu-test-results-${{ matrix.gpu }}
        path: test-results/

  # ===== 正确性验证 =====
  correctness:
    runs-on: [self-hosted, gpu, A100]
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Triton
      run: |
        pip install -e python/
        pip install -e "python[tests]"
    
    - name: Run correctness tests
      run: |
        python -m pytest python/test/correctness/ \
          -v \
          --tb=long \
          --timeout=600 \
          -k "not slow"  # 跳过慢速测试
    
    - name: Run comprehensive correctness tests
      run: |
        python -m pytest python/test/correctness/ \
          -v \
          --tb=long \
          --timeout=3600 \
          -k "slow"  # 运行完整测试

  # ===== 代码质量 =====
  code-quality:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install linting tools
      run: |
        pip install ruff mypy black isort
    
    - name: Run ruff linter
      run: |
        ruff check python/triton/
    
    - name: Run black formatter check
      run: |
        black --check python/triton/
    
    - name: Run isort import check
      run: |
        isort --check-only python/triton/
    
    - name: Run mypy type check
      run: |
        mypy python/triton/ --ignore-missing-imports
```

### 30.7.2 多硬件测试矩阵

```yaml
# .github/workflows/multi-hardware.yml
name: Multi-Hardware Tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  hardware-matrix:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        include:
          # NVIDIA GPUs
          - os: self-hosted
            gpu: NVIDIA-A100-80GB
            cuda: '12.1'
            python: '3.10'
            name: A100-80GB
            
          - os: self-hosted
            gpu: NVIDIA-A10-24GB
            cuda: '12.1'
            python: '3.10'
            name: A10-24GB
            
          - os: self-hosted
            gpu: NVIDIA-H100
            cuda: '12.2'
            python: '3.10'
            name: H100
            
          - os: self-hosted
            gpu: NVIDIA-RTX4090
            cuda: '12.1'
            python: '3.10'
            name: RTX4090
          
          # AMD GPUs (ROCm)
          - os: self-hosted
            gpu: AMD-MI250
            rocm: '5.7'
            python: '3.10'
            name: MI250
            
          - os: self-hosted
            gpu: AMD-MI300X
            rocm: '6.0'
            python: '3.10'
            name: MI300X
    
    name: Test on ${{ matrix.name }}
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup CUDA (NVIDIA)
      if: startsWith(matrix.gpu, 'NVIDIA')
      run: |
        # CUDA 环境设置
        export CUDA_HOME=/usr/local/cuda-${{ matrix.cuda }}
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    - name: Setup ROCm (AMD)
      if: startsWith(matrix.gpu, 'AMD')
      run: |
        # ROCm 环境设置
        export ROCM_PATH=/opt/rocm-${{ matrix.rocm }}
        export PATH=$ROCM_PATH/bin:$PATH
        export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
    
    - name: Set up Python ${{ matrix.python }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python }}
    
    - name: Install Triton
      run: |
        pip install -e python/
        pip install -e "python[tests]"
    
    - name: Verify hardware
      run: |
        if [[ "${{ matrix.gpu }}" == NVIDIA* ]]; then
          nvidia-smi
        elif [[ "${{ matrix.gpu }}" == AMD* ]]; then
          rocm-smi
        fi
    
    - name: Run tests
      run: |
        python -m pytest python/test/ \
          -v \
          --tb=short \
          --timeout=600 \
          -x
    
    - name: Run device-specific tests
      run: |
        if [[ "${{ matrix.gpu }}" == NVIDIA* ]]; then
          python -m pytest python/test/backends/test_cuda.py -v
        elif [[ "${{ matrix.gpu }}" == AMD* ]]; then
          python -m pytest python/test/backends/test_hip.py -v
        fi
```

### 30.7.3 自动化测试流程

```yaml
# .github/workflows/nightly.yml
name: Nightly Build

on:
  schedule:
    - cron: '0 4 * * *'  # UTC 凌晨 4 点
  workflow_dispatch:

jobs:
  nightly-tests:
    runs-on: [self-hosted, gpu, A100]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up environment
      run: |
        pip install -e python/
        pip install -e "python[tests]"
    
    - name: Run full test suite
      run: |
        python -m pytest python/test/ \
          -v \
          --tb=long \
          --timeout=3600 \
          --cov=python/triton/ \
          --cov-report=xml:coverage.xml
    
    - name: Run extended correctness tests
      run: |
        python -m pytest python/test/correctness/ \
          -v \
          --timeout=7200 \
          -k "comprehensive or extended"
    
    - name: Run performance benchmarks
      run: |
        python -m pytest python/benchmarks/ \
          --benchmark-json=nightly_benchmarks.json
    
    - name: Check for regressions
      run: |
        python -c "
        from triton.testing import RegressionDetector
        detector = RegressionDetector()
        # 加载基线并检查
        results = json.load(open('nightly_benchmarks.json'))
        # ... 检查逻辑
        "
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: coverage.xml
    
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: nightly-benchmarks-${{ github.run_id }}
        path: nightly_benchmarks.json
    
    - name: Notify on failure
      if: failure()
      run: |
        # 发送失败通知
        echo "Nightly build failed!"
        # 可以集成 Slack/Email 通知
```

### 30.7.4 测试脚本示例

```python
#!/usr/bin/env python3
"""
Triton 测试运行脚本
用于本地开发和 CI 环境
"""

import subprocess
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class TestCategory(Enum):
    """测试类别"""
    UNIT = "unit"
    KERNEL = "kernel"
    COMPILER = "compiler"
    CORRECTNESS = "correctness"
    BENCHMARK = "benchmark"
    ALL = "all"


@dataclass
class TestConfig:
    """测试配置"""
    categories: List[TestCategory]
    verbose: bool = True
    timeout: int = 300
    parallel: bool = False
    coverage: bool = False
    gpu_required: bool = True
    markers: Optional[List[str]] = None


class TritonTestRunner:
    """Triton 测试运行器"""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.test_dir = self.root_dir / "python" / "test"
        self.results_dir = self.root_dir / "test-results"
        self.results_dir.mkdir(exist_ok=True)
    
    def check_prerequisites(self) -> bool:
        """检查前置条件"""
        # 检查 CUDA
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print("WARNING: nvidia-smi failed, GPU tests may fail")
                return False
            print(f"CUDA available: {result.stdout.split(chr(10))[2]}")
        except FileNotFoundError:
            print("WARNING: nvidia-smi not found")
            return False
        
        # 检查 Triton
        try:
            import triton
            print(f"Triton version: {triton.__version__}")
        except ImportError:
            print("ERROR: Triton not installed")
            return False
        
        return True
    
    def run_tests(self, config: TestConfig) -> int:
        """运行测试"""
        pytest_args = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-v",
            f"--timeout={config.timeout}",
        ]
        
        # 输出格式
        pytest_args.extend([
            f"--junitxml={self.results_dir}/test-results.xml",
        ])
        
        # 覆盖率
        if config.coverage:
            pytest_args.extend([
                f"--cov={self.root_dir / 'python' / 'triton'}",
                f"--cov-report=xml:{self.results_dir}/coverage.xml",
                "--cov-report=html:htmlcov",
            ])
        
        # 标记过滤
        if config.markers:
            for marker in config.markers:
                pytest_args.extend(["-m", marker])
        
        # 测试类别
        if config.categories != [TestCategory.ALL]:
            test_paths = []
            for cat in config.categories:
                test_path = self.test_dir / cat.value
                if test_path.exists():
                    test_paths.append(str(test_path))
            if test_paths:
                pytest_args[2] = test_paths[0]  # 替换测试路径
                for tp in test_paths[1:]:
                    pytest_args.append(tp)
        
        print(f"Running: {' '.join(pytest_args)}")
        
        result = subprocess.run(pytest_args)
        return result.returncode
    
    def run_benchmarks(self) -> int:
        """运行性能基准测试"""
        benchmark_dir = self.root_dir / "python" / "benchmarks"
        if not benchmark_dir.exists():
            print("No benchmarks directory found")
            return 0
        
        pytest_args = [
            "python", "-m", "pytest",
            str(benchmark_dir),
            "-v",
            f"--benchmark-json={self.results_dir}/benchmarks.json",
            "--benchmark-only",
        ]
        
        result = subprocess.run(pytest_args)
        return result.returncode
    
    def generate_report(self) -> str:
        """生成测试报告"""
        lines = [
            "=" * 60,
            "Triton Test Report",
            "=" * 60,
            f"Test directory: {self.test_dir}",
            f"Results directory: {self.results_dir}",
            "",
        ]
        
        # 检查结果文件
        results_file = self.results_dir / "test-results.xml"
        if results_file.exists():
            import xml.etree.ElementTree as ET
            tree = ET.parse(results_file)
            root = tree.getroot()
            
            tests = int(root.get("tests", 0))
            failures = int(root.get("failures", 0))
            errors = int(root.get("errors", 0))
            skipped = int(root.get("skipped", 0))
            
            lines.extend([
                f"Tests: {tests}",
                f"Failures: {failures}",
                f"Errors: {errors}",
                f"Skipped: {skipped}",
                f"Pass rate: {(tests - failures - errors) / tests * 100:.1f}%"
                if tests > 0 else "N/A",
            ])
        
        lines.append("=" * 60)
        return "\n".join(lines)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Triton Test Runner")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=[c.value for c in TestCategory],
        default=["unit", "kernel"],
        help="Test categories to run"
    )
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Run benchmarks"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable coverage reporting"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Test timeout in seconds"
    )
    parser.add_argument(
        "--no-gpu-check",
        action="store_true",
        help="Skip GPU availability check"
    )
    
    args = parser.parse_args()
    
    runner = TritonTestRunner()
    
    # 检查前置条件
    if not args.no_gpu_check:
        if not runner.check_prerequisites():
            print("Prerequisites check failed")
            sys.exit(1)
    
    # 构建配置
    categories = [TestCategory(c) for c in args.categories]
    config = TestConfig(
        categories=categories,
        timeout=args.timeout,
        coverage=args.coverage,
    )
    
    # 运行测试
    exit_code = runner.run_tests(config)
    
    # 运行基准测试
    if args.benchmarks:
        bench_code = runner.run_benchmarks()
        if bench_code != 0:
            exit_code = bench_code
    
    # 生成报告
    report = runner.generate_report()
    print(report)
    
    # 保存报告
    report_file = runner.results_dir / "test-report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
```

---

## 30.8 测试最佳实践

### 30.8.1 测试覆盖率管理

```python
# 测试覆盖率配置
# .coveragerc
"""
[run]
source = python/triton
omit =
    python/triton/tests/*
    python/triton/benchmarks/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
    pass

show_missing = true
fail_under = 80

[html]
directory = htmlcov
"""

# 在 CI 中使用覆盖率
# 运行测试并收集覆盖率
# python -m pytest --cov=python/triton --cov-report=xml --cov-report=html

# 生成覆盖率报告
# coverage html
# coverage report --fail-under=80
```

```python
# 核心模块覆盖率要求
COVERAGE_REQUIREMENTS = {
    "python/triton/compiler": 85,      # 编译器核心
    "python/triton/language": 90,      # 语言核心
    "python/triton/runtime": 85,       # 运行时
    "python/triton/backends": 75,      # 后端（硬件相关）
    "overall": 80,                      # 整体覆盖率
}


def check_coverage(coverage_report: dict) -> bool:
    """检查覆盖率是否达标"""
    all_pass = True
    
    for module, required in COVERAGE_REQUIREMENTS.items():
        if module == "overall":
            actual = coverage_report.get("total_percent", 0)
        else:
            actual = coverage_report.get(module, {}).get("percent", 0)
        
        if actual < required:
            print(f"FAIL: {module} coverage {actual}% < {required}%")
            all_pass = False
        else:
            print(f"PASS: {module} coverage {actual}% >= {required}%")
    
    return all_pass
```

### 30.8.2 核心算子测试策略

```python
import pytest
import torch
import triton


# ===== 核心算子测试清单 =====
CORE_OPS = {
    # 基础算子
    "elementwise": ["add", "sub", "mul", "div", "abs", "neg", "exp", "log"],
    
    # 归约算子
    "reduction": ["sum", "max", "min", "mean", "argmax", "argmin"],
    
    # 矩阵算子
    "matmul": ["gemm", "transpose", "broadcast"],
    
    # 索引算子
    "indexing": ["gather", "scatter", "masking"],
    
    # 排序算子
    "sorting": ["sort", "argsort", "topk"],
    
    # 神经网络算子
    "nn": ["softmax", "layer_norm", "rms_norm", "gelu", "relu", "silu"],
}


class TestCoreOperators:
    """核心算子测试套件"""
    
    @pytest.mark.parametrize("op", CORE_OPS["elementwise"])
    def test_elementwise(self, device, op):
        """元素级算子测试"""
        x = torch.randn(1024, device=device)
        y = torch.randn(1024, device=device)
        
        # Triton 实现
        z_triton = eval(f"triton_{op}")(x, y) if op in ["add", "sub", "mul", "div"] \
            else eval(f"triton_{op}")(x)
        
        # PyTorch 参考
        z_ref = eval(f"torch.{op}")(x, y) if op in ["add", "sub", "mul", "div"] \
            else eval(f"torch.{op}")(x)
        
        # 验证
        rtol = 1e-5 if op != "div" else 1e-4
        torch.testing.assert_close(z_triton, z_ref, rtol=rtol, atol=rtol)
    
    @pytest.mark.parametrize("op", CORE_OPS["reduction"])
    def test_reduction(self, device, op):
        """归约算子测试"""
        x = torch.randn(64, 1024, device=device)
        
        # Triton 实现
        z_triton = eval(f"triton_{op}")(x, dim=-1)
        
        # PyTorch 参考
        z_ref = eval(f"torch.{op}")(x, dim=-1)
        
        # 验证
        torch.testing.assert_close(z_triton, z_ref, rtol=1e-5, atol=1e-5)
    
    @pytest.mark.parametrize("M,N,K", [
        (32, 32, 32),
        (128, 128, 128),
        (1024, 1024, 1024),
    ])
    def test_gemm(self, device, M, N, K):
        """矩阵乘法测试"""
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)
        
        z_triton = triton_matmul(a, b)
        z_ref = torch.mm(a, b)
        
        torch.testing.assert_close(z_triton, z_ref, rtol=1e-2, atol=1e-2)
    
    @pytest.mark.parametrize("op", CORE_OPS["nn"])
    def test_nn_operators(self, device, op):
        """神经网络算子测试"""
        x = torch.randn(16, 1024, device=device)
        
        if op == "softmax":
            z_triton = triton_softmax(x)
            z_ref = torch.softmax(x, dim=-1)
        elif op == "layer_norm":
            weight = torch.randn(1024, device=device)
            bias = torch.randn(1024, device=device)
            z_triton = triton_layer_norm(x, weight, bias)
            z_ref = torch.nn.functional.layer_norm(x, (1024,), weight, bias)
        # ... 其他算子
        
        torch.testing.assert_close(z_triton, z_ref, rtol=1e-4, atol=1e-4)
```

### 30.8.3 边界条件与异常测试

```python
import pytest
import torch
import triton


class TestEdgeCases:
    """边界条件测试"""
    
    def test_empty_input(self, device):
        """空输入测试"""
        x = torch.empty(0, device=device)
        y = torch.empty(0, device=device)
        
        try:
            z = vector_add(x, y)
            assert z.numel() == 0
        except (RuntimeError, ValueError):
            pass  # 预期可能抛出异常
    
    def test_single_element(self, device):
        """单元素测试"""
        x = torch.tensor([42.0], device=device)
        y = torch.tensor([1.0], device=device)
        z = vector_add(x, y)
        assert z.item() == 43.0
    
    def test_very_large_tensor(self, device):
        """大张量测试"""
        n = 1024 * 1024 * 100  # 100M 元素
        x = torch.randn(n, device=device)
        y = torch.randn(n, device=device)
        z = vector_add(x, y)
        torch.testing.assert_close(z, x + y)
    
    def test_extreme_values(self, device):
        """极端值测试"""
        # 最大值
        x = torch.full((1024,), torch.finfo(torch.float32).max, device=device)
        y = torch.full((1024,), torch.finfo(torch.float32).max, device=device)
        z = vector_add(x, y)
        # 检查是否正确处理溢出
        assert not torch.isnan(z).any() or not torch.isnan(x + y).any()
        
        # 最小正规数
        x = torch.full((1024,), torch.finfo(torch.float32).tiny, device=device)
        y = torch.full((1024,), torch.finfo(torch.float32).tiny, device=device)
        z = vector_add(x, y)
        torch.testing.assert_close(z, x + y, rtol=1e-3, atol=1e-38)
    
    def test_nan_handling(self, device):
        """NaN 处理测试"""
        x = torch.tensor([float("nan"), 1.0, 2.0], device=device)
        y = torch.tensor([1.0, float("nan"), 2.0], device=device)
        z = vector_add(x, y)
        
        # NaN 传播
        assert torch.isnan(z[0])
        assert torch.isnan(z[1])
        assert z[2] == 4.0
    
    def test_inf_handling(self, device):
        """Inf 处理测试"""
        x = torch.tensor([float("inf"), float("-inf"), 1e30], device=device)
        y = torch.tensor([1.0, 1.0, 1e30], device=device)
        z = vector_add(x, y)
        
        assert z[0] == float("inf")
        assert z[1] == float("-inf")
        # 大数相加可能溢出
        assert not torch.isnan(z[2]).any() or not torch.isnan(x + y).any()
    
    def test_non_contiguous_tensor(self, device):
        """非连续张量测试"""
        x = torch.randn(1024, 1024, device=device)[:, ::2]  # 非连续
        y = torch.randn(1024, 512, device=device)
        
        assert not x.is_contiguous()
        
        # 测试非连续张量的处理
        z = vector_add_2d(x, y)
        torch.testing.assert_close(z, x + y)
    
    def test_different_strides(self, device):
        """不同步长测试"""
        x = torch.randn(1024, 1024, device=device)
        y = torch.randn(1024, 1024, device=device).t()  # 转置
        
        # 测试不同步长的处理
        z = triton_matmul(x, y)
        torch.testing.assert_close(z, torch.mm(x, y), rtol=1e-2, atol=1e-2)
    
    def test_shared_memory_limit(self, device):
        """共享内存限制测试"""
        # 测试接近共享内存限制的情况
        shared_mem_size = torch.cuda.get_device_properties(device).shared_memory_per_block
        
        # 尝试分配接近限制的共享内存
        try:
            x = torch.randn(1024, device=device)
            z = torch.empty_like(x)
            kernel_with_large_shared_mem[1,](x, z, BLOCK_SIZE=1024)
        except RuntimeError as e:
            if "shared memory" in str(e).lower():
                pass  # 预期的共享内存溢出
            else:
                raise
```

### 30.8.4 测试组织最佳实践

```python
# tests/test_softmax.py - 完整的 Softmax 测试示例

import torch
import pytest
import triton
import triton.language as tl


# ===== Fixture =====
@pytest.fixture
def input_tensor():
    """生成 Softmax 输入张量"""
    torch.manual_seed(42)
    return torch.randn(16, 1024, device="cuda", dtype=torch.float16)


@pytest.fixture
def large_input():
    """生成大输入张量"""
    torch.manual_seed(42)
    return torch.randn(256, 4096, device="cuda", dtype=torch.float16)


# ===== 基础测试 =====
class TestSoftmaxBasic:
    """Softmax 基础功能测试"""
    
    def test_correctness(self, input_tensor):
        """正确性测试"""
        z_triton = triton_softmax(input_tensor)
        z_ref = torch.softmax(input_tensor, dim=-1)
        
        torch.testing.assert_close(
            z_triton, z_ref,
            rtol=1e-3,
            atol=1e-3,
            msg="Softmax correctness check failed"
        )
    
    def test_output_sum_to_one(self, input_tensor):
        """输出和为 1 测试"""
        z = triton_softmax(input_tensor)
        sum_along_dim = z.sum(dim=-1)
        
        torch.testing.assert_close(
            sum_along_dim,
            torch.ones_like(sum_along_dim),
            rtol=1e-3,
            atol=1e-3,
        )
    
    def test_numerical_stability(self):
        """数值稳定性测试"""
        # 大数值输入
        x = torch.randn(16, 1024, device="cuda", dtype=torch.float16)
        x = x * 100  # 放大
        
        z = triton_softmax(x)
        
        # 检查输出
        assert not torch.isnan(z).any(), "Softmax produced NaN"
        assert not torch.isinf(z).any(), "Softmax produced Inf"
        
        # 检查和为 1
        torch.testing.assert_close(
            z.sum(dim=-1),
            torch.ones(16, device="cuda", dtype=torch.float16),
            rtol=1e-2,
            atol=1e-2,
        )


# ===== 参数化测试 =====
class TestSoftmaxParametrized:
    """Softmax 参数化测试"""
    
    @pytest.mark.parametrize("shape", [
        (1, 64),
        (16, 256),
        (32, 1024),
        (64, 4096),
        (128, 8192),
    ])
    def test_various_shapes(self, shape):
        """不同形状测试"""
        x = torch.randn(*shape, device="cuda", dtype=torch.float16)
        
        z_triton = triton_softmax(x)
        z_ref = torch.softmax(x, dim=-1)
        
        torch.testing.assert_close(
            z_triton, z_ref,
            rtol=1e-3,
            atol=1e-3,
        )
    
    @pytest.mark.parametrize("dtype", [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ])
    def test_various_dtypes(self, dtype):
        """不同数据类型测试"""
        x = torch.randn(16, 1024, device="cuda", dtype=dtype)
        
        z_triton = triton_softmax(x)
        z_ref = torch.softmax(x, dim=-1)
        
        rtol, atol = {
            torch.float16: (1e-2, 1e-2),
            torch.float32: (1e-5, 1e-5),
            torch.bfloat16: (1e-2, 1e-2),
        }[dtype]
        
        torch.testing.assert_close(
            z_triton, z_ref,
            rtol=rtol,
            atol=atol,
        )
    
    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_different_dims(self, dim):
        """不同维度测试"""
        x = torch.randn(16, 1024, device="cuda", dtype=torch.float16)
        
        z_triton = triton_softmax(x, dim=dim)
        z_ref = torch.softmax(x, dim=dim)
        
        torch.testing.assert_close(
            z_triton, z_ref,
            rtol=1e-3,
            atol=1e-3,
        )


# ===== 性能测试 =====
class TestSoftmaxPerformance:
    """Softmax 性能测试"""
    
    @pytest.mark.slow
    def test_throughput(self, large_input):
        """吞吐量测试"""
        import time
        
        # Warmup
        for _ in range(10):
            triton_softmax(large_input)
        
        torch.cuda.synchronize()
        
        # 计时
        start = time.time()
        n_iter = 100
        for _ in range(n_iter):
            triton_softmax(large_input)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        throughput = large_input.numel() * n_iter / elapsed / 1e9
        print(f"\nSoftmax throughput: {throughput:.2f} GElements/s")
        
        # 至少达到 10 GElements/s
        assert throughput > 10, f"Throughput {throughput} GElements/s too low"
    
    @pytest.mark.benchmark
    def test_benchmark(self):
        """基准测试"""
        import triton.testing
        
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["N"],
                x_vals=[64 * i for i in range(1, 33)],
                line_arg="provider",
                line_vals=["triton", "torch"],
                line_names=["Triton", "PyTorch"],
                ylabel="GB/s",
                plot_name="softmax-benchmark",
            )
        )
        def bench(N, provider):
            x = torch.randn(16, N, device="cuda", dtype=torch.float16)
            if provider == "triton":
                return triton_softmax(x)
            else:
                return torch.softmax(x, dim=-1)
        
        bench.run(print_data=True)


# ===== 边界条件测试 =====
class TestSoftmaxEdgeCases:
    """Softmax 边界条件测试"""
    
    def test_single_element(self):
        """单元素测试"""
        x = torch.tensor([[1.0]], device="cuda", dtype=torch.float16)
        z = triton_softmax(x)
        
        torch.testing.assert_close(
            z,
            torch.tensor([[1.0]], device="cuda", dtype=torch.float16),
            rtol=1e-3,
            atol=1e-3,
        )
    
    def test_all_same_values(self):
        """所有值相同测试"""
        x = torch.ones(16, 1024, device="cuda", dtype=torch.float16)
        z = triton_softmax(x)
        
        expected = 1.0 / 1024.0
        torch.testing.assert_close(
            z,
            torch.full_like(z, expected),
            rtol=1e-3,
            atol=1e-3,
        )
    
    def test_extreme_values(self):
        """极端值测试"""
        x = torch.randn(16, 1024, device="cuda", dtype=torch.float16)
        x[:, 0] = 1000.0  # 大值
        x[:, 1] = -1000.0  # 小值
        
        z = triton_softmax(x)
        
        assert not torch.isnan(z).any()
        assert not torch.isinf(z).any()
        torch.testing.assert_close(
            z.sum(dim=-1),
            torch.ones(16, device="cuda", dtype=torch.float16),
            rtol=1e-2,
            atol=1e-2,
        )


# ===== 回归测试 =====
class TestSoftmaxRegression:
    """Softmax 回归测试"""
    
    def test_no_regression_vs_pytorch(self):
        """对比 PyTorch 无回归"""
        x = torch.randn(16, 1024, device="cuda", dtype=torch.float16)
        
        z_triton = triton_softmax(x)
        z_ref = torch.softmax(x, dim=-1)
        
        # 严格对比
        torch.testing.assert_close(
            z_triton, z_ref,
            rtol=1e-3,
            atol=1e-3,
        )
    
    @pytest.mark.skipif(
        torch.cuda.get_device_capability() < (8, 0),
        reason="Requires Ampere+"
    )
    def test_ampere_specific(self):
        """Ampere+ 特定测试"""
        # 使用 Tensor Core 优化的 Softmax
        x = torch.randn(32, 2048, device="cuda", dtype=torch.float16)
        z = triton_softmax_tensor_core(x)
        z_ref = torch.softmax(x, dim=-1)
        
        torch.testing.assert_close(z, z_ref, rtol=1e-2, atol=1e-2)
```

### 30.8.5 测试文档与注释

```python
"""
Softmax Kernel 测试套件

本模块包含 Triton Softmax kernel 的完整测试覆盖。

测试分类:
    1. 基础功能测试 - 验证核心正确性
    2. 参数化测试 - 覆盖不同输入配置
    3. 边界条件测试 - 验证极端情况处理
    4. 性能测试 - 验证性能基线
    5. 回归测试 - 检测性能回归

运行方式:
    # 运行所有测试
    python -m pytest tests/test_softmax.py -v
    
    # 只运行基础测试
    python -m pytest tests/test_softmax.py::TestSoftmaxBasic -v
    
    # 运行性能测试
    python -m pytest tests/test_softmax.py -m benchmark
    
    # 运行快速测试（跳过慢速测试）
    python -m pytest tests/test_softmax.py -m "not slow"

依赖:
    - pytest >= 7.0
    - torch >= 2.0
    - triton >= 2.0
"""

import torch
import pytest
import triton
import triton.language as tl


# ... 测试代码 ...
```

---

## 本章小结

### 核心知识点

1. **Triton 测试架构**
   - 测试代码组织在 `python/test/` 目录下
   - 使用 pytest 框架，通过 `conftest.py` 管理 fixtures
   - 测试分为单元测试、集成测试、性能测试等层次

2. **正确性验证**
   - 使用 PyTorch 作为 reference implementation
   - 合理设置容差（rtol/atol）考虑浮点精度
   - 多维度验证：多 shape、多 dtype、多数值范围

3. **单元测试编写**
   - 使用 `@pytest.mark.parametrize` 进行参数化
   - 使用标记（`@pytest.mark.slow`）控制测试执行
   - 异常测试确保错误处理正确

4. **性能基准测试**
   - `triton.testing.Benchmark` 提供标准化的性能测试
   - 自动生成报告和图表
   - 支持多维度性能对比

5. **TritonBench**
   - 系统化的性能基准平台
   - 支持测试矩阵定义
   - 历史数据对比和趋势分析

6. **回归测试**
   - 基于基线数据检测性能回归
   - 阈值管理策略
   - CI 中自动运行和告警

7. **CI/CD 集成**
   - GitHub Actions 配置
   - 多硬件测试矩阵
   - 自动化测试流程

### 最佳实践

| 实践 | 说明 |
|------|------|
| 参考实现对比 | 所有 kernel 都应与 PyTorch 实现对比 |
| 合理容差 | 根据数据类型和操作设置合适的容差 |
| 参数化测试 | 覆盖多种输入配置 |
| 边界条件测试 | 测试空输入、单元素、极端值等 |
| 性能基线 | 建立性能基线，检测回归 |
| CI 集成 | 所有测试在 CI 中自动运行 |
| 文档 | 测试代码应包含清晰的文档 |

---

## 思考题

### 基础题

1. **为什么 Triton 测试需要设置数值容差（rtol/atol），而不是直接比较浮点数相等？**

2. **解释 `@pytest.mark.parametrize` 和 `@pytest.fixture` 在测试中的作用和区别。**

3. **在性能基准测试中，为什么需要进行 warmup？不进行 warmup 会导致什么问题？**

### 进阶题

4. **设计一个完整的 Triton kernel 测试策略，包括正确性、性能、边界条件三个维度。你会如何组织测试代码？**

5. **如何在 CI/CD 流程中实现性能回归检测？讨论阈值设定的策略和挑战。**

6. **Triton 的多硬件测试矩阵（NVIDIA/AMD/Intel）有哪些技术挑战？如何设计测试策略来保证跨硬件兼容性？**

### 实践题

7. **编写一个完整的测试套件，测试一个自定义的 Triton kernel（如 LayerNorm），包括：**
   - 正确性测试（至少 5 种输入配置）
   - 性能测试（与 PyTorch 对比）
   - 边界条件测试（空输入、极端值等）

8. **配置一个 GitHub Actions 工作流，实现：**
   - 单元测试自动运行
   - 性能基准测试和回归检测
   - 多 GPU 硬件测试
   - 测试结果报告生成

9. **分析 TritonBench 的设计理念，讨论：**
   - 如何设计可重复的性能测试？
   - 如何处理硬件差异对性能的影响？
   - 如何构建有意义的性能对比？

### 思考题

10. **随着 AI 硬件的快速发展（NVIDIA H100/B100、AMD MI300X、Intel Gaudi 等），Triton 测试框架需要如何演进？讨论测试框架的可扩展性设计。**

11. **在实际生产环境中，Triton kernel 的测试与普通软件测试有哪些本质区别？如何平衡测试覆盖率与测试执行时间？**

12. **讨论 AI Agent 辅助生成的 Triton kernel 的测试挑战，如何验证自动生成的代码的正确性和性能？**

---

**下一章预告**：第 31 章将介绍 AI Agent 辅助算子生成的概念与技术路线，探索如何利用大语言模型辅助 Triton kernel 的开发与优化。
