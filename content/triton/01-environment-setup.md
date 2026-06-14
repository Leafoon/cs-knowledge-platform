---
title: "Chapter 1: 开发环境搭建与源码结构"
description: "掌握 Triton 的完整安装流程（pip 源码编译、conda、Docker），理解官方仓库的目录结构与构建系统，配置 CUDA/ROCm 开发环境"
date: "2026-06-11"
---

# Chapter 1: 开发环境搭建与源码结构

> **学习目标**：
> - 掌握 Triton 的四种安装方式（pip、源码编译、Docker、conda-forge）及其适用场景
> - 理解 Triton 的完整依赖链：Python 3.8+、CUDA 11.7+/ROCm 5.6+、LLVM、PyTorch
> - 能够独立走读 triton-lang/triton 仓库的顶层目录结构，理解每个目录的职责
> - 深入理解 python/triton/ 下的核心模块：core.py、runtime/、compiler/、tools/
> - 掌握 lib/ 下的编译器核心：Dialect 定义、Conversion Pass、Target 代码生成
> - 理解 CMake 构建系统与 Python 包构建的完整流程
> - 能够配置 VS Code 调试环境，使用 IR dump 等开发调试工具
> - 能够编写并验证第一个 Triton kernel

---

## 1.1 安装方式总览

Triton 提供了多种安装方式，适用于不同的使用场景。选择合适的安装方式是高效开发的第一步。

### 1.1.1 安装方式对比

| 安装方式 | 适用场景 | 优点 | 缺点 |
|:---|:---|:---|:---|
| **pip install triton** | 快速体验、生产使用 | 最简单、预编译二进制 | 版本可能滞后 |
| **源码编译** | 开发者、需要最新特性 | 可自定义、可调试 | 构建时间长、依赖复杂 |
| **Docker 镜像** | CI/CD、隔离环境 | 环境一致、开箱即用 | 镜像较大、灵活性有限 |
| **conda-forge** | conda 用户、多环境管理 | 依赖管理方便 | 版本更新可能较慢 |

<div data-component="InstallationMethodSelector"></div>

[组件：InstallationMethodSelector - 交互式安装方式选择器，根据用户场景推荐最佳安装方式]

### 1.1.2 方式一：从 PyPI 安装（推荐入门）

最简单的安装方式是通过 pip 从 PyPI 安装预编译的 wheel 包：

```bash
# 基础安装（仅支持 NVIDIA GPU）
pip install triton

# 指定版本安装
pip install triton==2.3.1

# 升级到最新版本
pip install --upgrade triton
```

安装完成后，验证安装：

```bash
python -c "import triton; print(triton.__version__)"
# 输出示例: 2.3.1
```

**PyPI 安装的注意事项**：

1. **CUDA 版本兼容性**：PyPI 上的预编译 wheel 通常绑定特定 CUDA 版本。如果你的 CUDA 版本不匹配，可能会遇到运行时错误。
2. **Python 版本要求**：需要 Python 3.8 或更高版本。
3. **操作系统**：目前支持 Linux（x86_64/aarch64）和 macOS（实验性）。

```bash
# 检查当前环境的 Python 和 CUDA 版本
python --version
# Python 3.10.12

nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on Tue_Aug_15_22:02:13_PDT_2023
# Cuda compilation tools, release 12.2, V12.2.140
```

### 1.1.3 方式二：从源码编译（推荐开发者）

从源码编译是开发者最常用的方式，可以获取最新特性、自定义构建选项、并支持调试。

**步骤 1：克隆仓库**

```bash
# 克隆官方仓库（triton-lang 是目前的主仓库）
git clone https://github.com/triton-lang/triton.git
cd triton

# 查看当前分支和最新提交
git log --oneline -5
# 示例输出:
# a1b2c3d [BACKEND] Fix dot operand layout inference
# e4f5g6h [FRONTEND] Add support for tl.dot with FP8
# i7j8k9l [BUILD] Update LLVM to 20240101
# ...
```

**步骤 2：安装 Python 依赖**

```bash
# 创建虚拟环境（推荐）
python -m venv triton-dev
source triton-dev/bin/activate

# 安装构建依赖
pip install --upgrade pip
pip install setuptools wheel ninja cmake>=3.24
```

**步骤 3：执行安装**

```bash
# 方式 A：pip editable 安装（推荐开发使用，修改代码后自动生效）
pip install -e python

# 方式 B：直接运行 setup.py
cd python
python setup.py develop
cd ..

# 方式 C：仅编译 C++ 库（不安装 Python 包）
mkdir build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
cd ..
```

**步骤 4：验证源码安装**

```bash
python -c "
import triton
print(f'Triton version: {triton.__version__}')
print(f'Triton location: {triton.__file__}')

# 测试基本功能
import triton.language as tl
print(f'Triton language module: {tl}')
"
```

预期输出：

```
Triton version: 2.3.1+gitXXXXXXX
Triton location: /path/to/triton/python/triton/__init__.py
Triton language module: <module 'triton.language' from '...'>
```

### 1.1.4 方式三：Docker 镜像

Docker 镜像提供了完全隔离的环境，适合 CI/CD 和快速复现：

```bash
# 拉取官方镜像（如果提供）
docker pull ghcr.io/triton-lang/triton:main

# 或者使用 NVIDIA 的 NGC 镜像（包含 PyTorch + Triton）
docker pull nvcr.io/nvidia/pytorch:24.01-py3

# 启动容器（挂载当前目录，启用 GPU 支持）
docker run --gpus all -it \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/pytorch:24.01-py3 \
    bash

# 在容器内验证
python -c "import triton; print(triton.__version__)"
```

**自定义 Dockerfile**：

```dockerfile
# Dockerfile.triton
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# 安装 PyTorch（可选但推荐）
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu122

# 从源码编译 Triton
RUN git clone https://github.com/triton-lang/triton.git /opt/triton \
    && cd /opt/triton \
    && pip install -e python

WORKDIR /workspace
CMD ["/bin/bash"]
```

构建并运行自定义镜像：

```bash
docker build -f Dockerfile.triton -t triton-dev .
docker run --gpus all -it -v $(pwd):/workspace triton-dev
```

### 1.1.5 方式四：conda-forge 安装

conda-forge 提供了预编译的 conda 包，适合使用 conda 管理环境的用户：

```bash
# 创建专用环境
conda create -n triton python=3.10
conda activate triton

# 从 conda-forge 安装
conda install -c conda-forge triton

# 或者使用 mamba（更快的依赖解析）
mamba install -c conda-forge triton
```

**conda 安装的优势**：

1. **依赖管理**：conda 会自动处理 CUDA toolkit、LLVM 等复杂依赖
2. **多环境隔离**：可以轻松创建多个独立的 Triton 环境
3. **跨平台**：conda-forge 支持 Linux 和 macOS

<div data-component="InstallationComparisonTable"></div>

[组件：InstallationComparisonTable - 四种安装方式的详细对比表格，包含依赖处理、构建时间、适用场景等维度]

---

## 1.2 依赖环境详解

Triton 的编译和运行依赖多个关键组件。理解这些依赖关系是排查安装问题的基础。

### 1.2.1 Python 版本要求

Triton 要求 Python 3.8 或更高版本。不同版本的兼容性如下：

| Python 版本 | 支持状态 | 备注 |
|:---|:---|:---|
| Python 3.8 | 支持 | 最低要求版本 |
| Python 3.9 | 支持 | 推荐 |
| Python 3.10 | 支持 | 推荐 |
| Python 3.11 | 支持 | 推荐 |
| Python 3.12 | 支持 | 较新版本 |
| Python 3.7 及以下 | 不支持 | 缺少必要的语法特性 |

```bash
# 检查 Python 版本
python --version
# Python 3.10.12

# 检查 Python 解释器路径
which python
# /usr/bin/python

# 使用 pyenv 管理多版本（推荐）
pyenv install 3.10.12
pyenv virtualenv 3.10.12 triton-dev
pyenv activate triton-dev
```

### 1.2.2 CUDA 环境要求

对于 NVIDIA GPU 用户，需要安装 CUDA Toolkit：

```bash
# 检查 CUDA 版本
nvcc --version
# cuda compilation tools, release 12.2, V12.2.140

# 检查 NVIDIA 驱动版本
nvidia-smi
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2     |
# |-------------------------------+----------------------+----------------------+
```

**CUDA 版本兼容性矩阵**：

| Triton 版本 | 最低 CUDA 版本 | 推荐 CUDA 版本 | 备注 |
|:---|:---|:---|:---|
| triton >= 2.3 | CUDA 11.7 | CUDA 12.1+ | 支持 Hopper 架构 |
| triton >= 2.2 | CUDA 11.7 | CUDA 12.0+ | 支持 FP8 |
| triton >= 2.1 | CUDA 11.4 | CUDA 11.8+ | 基础支持 |

**环境变量配置**：

```bash
# ~/.bashrc 或 ~/.zshrc
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 验证配置
echo $CUDA_HOME
# /usr/local/cuda
```

### 1.2.3 ROCm 环境要求（AMD GPU）

对于 AMD GPU 用户，需要安装 ROCm：

```bash
# 检查 ROCm 版本
rocm-smi --showproductname
# GPU[0] : Card Series: AMD Instinct MI250X

# 检查 HIP 版本
hipconfig --version
# HIP version: 5.7.31921-
```

**ROCm 版本兼容性**：

| Triton 版本 | 最低 ROCm 版本 | 推荐 ROCm 版本 | 支持的 GPU |
|:---|:---|:---|:---|
| triton >= 2.3 | ROCm 5.6 | ROCm 5.7+ | MI200/MI300 系列 |
| triton >= 2.2 | ROCm 5.5 | ROCm 5.6+ | MI200 系列 |

```bash
# ROCm 环境变量
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

### 1.2.4 LLVM 依赖

Triton 依赖 LLVM/MLIR 进行中间表示的编译和优化。Triton 通常**自带预编译的 LLVM 版本**，无需用户手动安装。

```bash
# 查看 Triton 使用的 LLVM 版本
python -c "
import triton._C.libtriton.triton as triton_c
print(dir(triton_c))
"

# 如果从源码编译，LLVM 会在构建过程中自动下载
# 查看 CMakeLists.txt 中的 LLVM 配置
cat CMakeLists.txt | grep -i llvm
```

Triton 对 LLVM 的依赖体现在以下方面：

1. **MLIR 框架**：Triton 的 IR 基于 MLIR（Multi-Level Intermediate Representation）构建
2. **方言转换**：从 Triton 方言到 LLVM 方言的转换依赖 MLIR 基础设施
3. **代码生成**：最终的 PTX/HSACO 代码生成依赖 LLVM 的 NVPTX/AMDGPU 后端

```python
# 查看 Triton 内部 LLVM 的版本信息
import os
import subprocess

# Triton 预编译 LLVM 的位置
triton_llvm_path = os.path.join(
    os.path.dirname(__file__),
    "triton/_C/libtriton/llvm"
)
print(f"LLVM path: {triton_llvm_path}")
```

### 1.2.5 PyTorch（可选但推荐）

PyTorch 不是 Triton 的硬性依赖，但在实际使用中几乎总是需要的：

```bash
# 安装 PyTorch（以 CUDA 12.1 为例）
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 验证 PyTorch + CUDA
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

预期输出：

```
PyTorch version: 2.3.0+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 4090
```

PyTorch 在 Triton 生态中的角色：

| 角色 | 说明 |
|:---|:---|
| **数据提供者** | PyTorch tensor 是 Triton kernel 最常见的输入 |
| **编译器后端** | torch.compile 使用 Triton 作为 GPU 后端 |
| **性能基准** | PyTorch 的算子实现是 Triton kernel 的对照基准 |
| **生态集成** | FlashAttention、vLLM 等项目同时依赖两者 |

<div data-component="DependencyGraph"></div>

[组件：DependencyGraph - 交互式依赖关系图，展示 Python、CUDA/ROCm、LLVM、PyTorch 之间的依赖关系]

### 1.2.6 完整依赖检查脚本

以下脚本可以一键检查所有依赖是否满足：

```python
#!/usr/bin/env python3
"""Triton 环境依赖检查脚本"""

import sys
import os
import subprocess
import shutil

def check_python():
    """检查 Python 版本"""
    version = sys.version_info
    status = "✅" if version >= (3, 8) else "❌"
    print(f"{status} Python {version.major}.{version.minor}.{version.micro}")
    if version < (3, 8):
        print("   需要 Python 3.8 或更高版本")
    return version >= (3, 8)

def check_cuda():
    """检查 CUDA 环境"""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        print("⚠️  CUDA toolkit 未找到（AMD GPU 用户可忽略）")
        return True  # 不是硬性要求
    
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True
        )
        for line in result.stdout.split("\n"):
            if "release" in line:
                print(f"✅ CUDA: {line.strip()}")
                break
    except Exception as e:
        print(f"❌ CUDA 检查失败: {e}")
        return False
    return True

def check_nvidia_driver():
    """检查 NVIDIA 驱动"""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        print("⚠️  nvidia-smi 未找到")
        return True
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True
        )
        for line in result.stdout.strip().split("\n"):
            parts = line.split(", ")
            if len(parts) == 3:
                name, driver, memory = parts
                print(f"✅ GPU: {name}")
                print(f"   驱动: {driver}, 显存: {memory}")
    except Exception as e:
        print(f"❌ GPU 检查失败: {e}")
        return False
    return True

def check_pytorch():
    """检查 PyTorch"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        status = "✅" if cuda_available else "⚠️"
        print(f"{status} PyTorch {torch.__version__}")
        print(f"   CUDA 可用: {cuda_available}")
        if cuda_available:
            print(f"   CUDA 版本: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("⚠️  PyTorch 未安装（可选但推荐）")
    return True

def check_triton():
    """检查 Triton"""
    try:
        import triton
        print(f"✅ Triton {triton.__version__}")
        print(f"   安装路径: {triton.__file__}")
    except ImportError:
        print("❌ Triton 未安装")
        return False
    return True

def main():
    print("=" * 50)
    print("Triton 环境依赖检查")
    print("=" * 50)
    print()
    
    checks = [
        ("Python", check_python),
        ("NVIDIA Driver", check_nvidia_driver),
        ("CUDA Toolkit", check_cuda),
        ("PyTorch", check_pytorch),
        ("Triton", check_triton),
    ]
    
    results = []
    for name, check_fn in checks:
        print(f"[{name}]")
        try:
            results.append(check_fn())
        except Exception as e:
            print(f"❌ 检查失败: {e}")
            results.append(False)
        print()
    
    print("=" * 50)
    if all(results):
        print("✅ 所有检查通过！可以开始使用 Triton。")
    else:
        print("⚠️  部分检查未通过，请根据提示修复。")
    print("=" * 50)

if __name__ == "__main__":
    main()
```

运行检查脚本：

```bash
python check_triton_env.py
```

预期输出：

```
==================================================
Triton 环境依赖检查
==================================================

[Python]
✅ Python 3.10.12

[NVIDIA Driver]
✅ GPU: NVIDIA GeForce RTX 4090
   驱动: 535.129.03, 显存: 24564 MiB

[CUDA Toolkit]
✅ CUDA: cuda compilation tools, release 12.2, V12.2.140

[PyTorch]
✅ PyTorch 2.3.0+cu121
   CUDA 可用: True
   CUDA 版本: 12.1
   GPU: NVIDIA GeForce RTX 4090

[Triton]
✅ Triton 2.3.1+gitXXXXXXX
   安装路径: /path/to/triton/python/triton/__init__.py

==================================================
✅ 所有检查通过！可以开始使用 Triton。
==================================================
```

---

## 1.3 官方仓库目录结构

理解 Triton 官方仓库的目录结构是深入学习源码的基础。Triton 的仓库从 OpenAI 迁移到了 triton-lang 组织下，但目录结构保持一致。

### 1.3.1 仓库顶层结构

```
triton/                          # 仓库根目录
├── CMakeLists.txt               # 顶层 CMake 构建配置
├── LICENSE                      # MIT 许可证
├── README.md                    # 项目说明文档
├── VERSION                      # 版本号文件
├── python/                      # Python 前端与包定义
│   ├── setup.py                 # Python 包构建脚本
│   ├── pyproject.toml           # Python 项目配置
│   └── triton/                  # Triton Python 包（核心）
├── lib/                         # C++ 编译器核心库
│   ├── Dialect/                 # MLIR 方言定义
│   ├── Conversion/              # 方言转换 Pass
│   └── Target/                  # 后端代码生成
├── include/                     # C++ 头文件
│   └── triton/                  # Triton 编译器头文件
├── third_party/                 # 第三方依赖与硬件后端
│   ├── nvidia/                  # NVIDIA GPU 后端
│   ├── amd/                     # AMD GPU 后端
│   └── intel/                   # Intel GPU 后端
├── test/                        # 测试用例
│   ├── TritonGPU/               # TritonGPU 相关测试
│   └── Conversion/              # 转换 Pass 测试
├── docs/                        # 文档
├── .github/                     # GitHub Actions CI 配置
└── bin/                         # 辅助脚本
```

<div data-component="RepoDirectoryTree"></div>

[组件：RepoDirectoryTree - 交互式目录树，可展开/折叠查看各子目录的详细结构]

### 1.3.2 顶层文件说明

| 文件 | 说明 |
|:---|:---|
| `CMakeLists.txt` | 顶层 CMake 构建配置，定义 LLVM 依赖、编译选项、目标库 |
| `VERSION` | 版本号文件，格式为 `MAJOR.MINOR.PATCH` |
| `LICENSE` | MIT 许可证 |
| `README.md` | 项目说明、快速开始、构建指南 |
| `CONTRIBUTING.md` | 贡献指南（如果存在） |

```bash
# 查看版本号
cat VERSION
# 2.3.1

# 查看 CMakeLists.txt 的关键配置
head -50 CMakeLists.txt
```

### 1.3.3 仓库组织的设计哲学

Triton 的仓库组织遵循以下设计原则：

1. **Python 前端与 C++ 后端分离**：`python/` 包含用户接口，`lib/` 包含编译器核心
2. **方言驱动的编译器架构**：`lib/Dialect/` 定义 IR，`lib/Conversion/` 定义转换
3. **硬件后端可插拔**：`third_party/` 通过插件机制支持不同硬件
4. **测试与实现同构**：`test/` 目录结构镜像 `lib/` 的组织方式

```
设计哲学可视化：

    用户视角                 编译器视角
    ┌──────────┐            ┌──────────────┐
    │ python/  │            │    lib/      │
    │ ┌──────┐ │            │ ┌──────────┐ │
    │ │ DSL  │ │  ──JIT──>  │ │ Dialect/ │ │
    │ │ API  │ │            │ │ IR 定义  │ │
    │ └──────┘ │            │ └────┬─────┘ │
    │ ┌──────┐ │            │      │       │
    │ │runtime│ │           │ ┌────▼─────┐ │
    │ │ JIT  │ │            │ │Conversion│ │
    │ └──────┘ │            │ │ Pass     │ │
    └──────────┘            │ └────┬─────┘ │
                            │      │       │
                            │ ┌────▼─────┐ │
                            │ │ Target/  │ │
                            │ │ 代码生成 │ │
                            │ └────┬─────┘ │
                            └──────┼───────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
               ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
               │ NVIDIA  │   │  AMD   │   │ Intel  │
               │ PTX     │   │ HSACO  │   │ SPIR-V │
               └─────────┘   └────────┘   └────────┘
```

---

## 1.4 python/ 子目录详解

`python/` 目录包含 Triton 的 Python 前端，是用户与 Triton 交互的主要入口。

### 1.4.1 python/ 目录结构

```
python/
├── setup.py                     # 包构建脚本
├── pyproject.toml               # 项目元数据
├── MANIFEST.in                  # 打包清单
└── triton/                      # 核心 Python 包
    ├── __init__.py              # 包初始化，导出公共 API
    ├── core.py                  # 语言核心（已迁移到 language/）
    ├── language/                # Triton 语言核心
    │   ├── __init__.py          # 导出 tl.load, tl.store, tl.dot 等
    │   ├── core.py              # 核心语义定义
    │   ├── math.py              # 数学函数
    │   └── standard.py          # 标准库函数
    ├── runtime/                 # JIT 运行时
    │   ├── __init__.py
    │   ├── driver.py            # GPU 驱动抽象
    │   ├── jit.py               # JIT 编译核心
    │   ├── cache.py             # 编译缓存管理
    │   └── autotuner.py         # 自动调优框架
    ├── compiler/                # 编译驱动
    │   ├── __init__.py
    │   ├── code_generator.py    # Python AST → Triton IR
    │   ├── make_launcher.py     # 启动器生成
    │   └── errors.py            # 编译错误处理
    ├── tools/                   # 调试与分析工具
    │   ├── __init__.py
    │   ├── disassembler.py      # 反汇编工具
    │   └── debugger.py          # 调试辅助
    ├── _C/                      # C++ 扩展绑定
    │   └── libtriton/           # 编译后的 C++ 库
    ├── backends/                # 后端抽象层
    │   ├── __init__.py
    │   ├── nvidia/              # NVIDIA 后端
    │   ├── amd/                 # AMD 后端
    │   └── driver.py            # 后端驱动接口
    └── ops/                     # 高级算子（已废弃或迁移）
```

### 1.4.2 core.py 与 language/ 模块

`language/` 模块（早期版本中是 `core.py`）定义了 Triton 的核心编程接口：

```python
# python/triton/language/__init__.py 的核心导出
from .core import (
    # 张量创建
    zeros, zeros_like, full, arange,
    
    # 内存操作
    load, store,
    
    # 算术运算
    dot, where,
    
    # 归约操作
    sum, max, min,
    
    # 类型转换
    cast,
    
    # 程序控制
    program_id, num_programs,
    static_range, static_assert,
    
    # 常量
    constexpr,
)
```

**核心 API 分类**：

| 类别 | API | 说明 |
|:---|:---|:---|
| **程序控制** | `tl.program_id(axis)` | 获取当前 program 在指定轴上的 ID |
| **程序控制** | `tl.num_programs(axis)` | 获取指定轴上的 program 总数 |
| **张量创建** | `tl.zeros(shape, dtype)` | 创建全零张量 |
| **张量创建** | `tl.full(shape, value, dtype)` | 创建指定值的张量 |
| **张量创建** | `tl.arange(start, end)` | 创建一维连续整数张量 |
| **内存操作** | `tl.load(ptr, mask, other)` | 从全局内存加载数据 |
| **内存操作** | `tl.store(ptr, value, mask)` | 向全局内存存储数据 |
| **计算** | `tl.dot(a, b)` | 矩阵乘法（映射到 Tensor Core） |
| **归约** | `tl.sum(input, axis)` | 沿轴求和 |
| **类型** | `tl.constexpr(value)` | 编译期常量 |

```python
# 查看 language 模块的完整内容
import triton.language as tl

# 列出所有公共 API
public_apis = [name for name in dir(tl) if not name.startswith('_')]
print("Triton Language 公共 API:")
for api in sorted(public_apis):
    print(f"  tl.{api}")
```

### 1.4.3 runtime/ 模块

`runtime/` 模块负责 JIT 编译和 kernel 执行：

```python
# python/triton/runtime/jit.py 的核心结构
class JITFunction:
    """Triton JIT 编译函数的核心类"""
    
    def __init__(self, fn, version=None, do_not_specialize=None):
        self.fn = fn                    # 原始 Python 函数
        self.version = version          # 版本号（用于缓存失效）
        self.do_not_specialize = do_not_specialize  # 不特化的参数
        
    def run(self, *args, **kwargs):
        """执行 JIT 编译并运行 kernel"""
        # 1. 参数特化（根据常量参数生成特化版本）
        # 2. 查找缓存
        # 3. 缓存未命中时触发编译
        # 4. 调用编译后的 kernel
        pass
    
    def __call__(self, *args, **kwargs):
        """使 JITFunction 可以像普通函数一样调用"""
        return self.run(*args, **kwargs)
```

**JIT 编译流程**：

```
@triton.jit 装饰的函数调用
        │
        ▼
┌─────────────────┐
│ 1. 参数特化     │  识别 constexpr 参数（如 BLOCK_SIZE）
│    (Specialize) │  根据参数类型生成特化 key
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. 缓存查找     │  检查 .triton/cache/ 目录
│    (Cache Hit?) │  使用 hash(key) 查找
└────────┬────────┘
         │ 未命中
         ▼
┌─────────────────┐
│ 3. JIT 编译     │  AST 捕获 → Triton IR → PTX/HSACO
│    (Compile)    │  编译结果存入缓存
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. 执行         │  加载编译后的二进制
│    (Execute)    │  设置 grid 参数，启动 kernel
└─────────────────┘
```

### 1.4.4 compiler/ 模块

`compiler/` 模块负责将 Python 代码转换为 Triton IR：

```python
# python/triton/compiler/code_generator.py 的核心结构
class CodeGenerator(ast.NodeVisitor):
    """Python AST 到 Triton IR 的代码生成器"""
    
    def __init__(self, context, prototype, options):
        self.context = context          # MLIR 上下文
        self.prototype = prototype      # 函数签名
        self.options = options          # 编译选项
        
    def visit_FunctionDef(self, node):
        """处理函数定义"""
        pass
    
    def visit_For(self, node):
        """处理 for 循环（映射到 tl.static_range）"""
        pass
    
    def visit_Call(self, node):
        """处理函数调用（映射到 tl.load, tl.store 等）"""
        pass
    
    def visit_BinOp(self, node):
        """处理二元运算（+, -, *, /）"""
        pass
```

**编译驱动流程**：

```python
# python/triton/compiler/__init__.py
def compile(fn, signature, device_type, constants, configs):
    """
    编译 Triton kernel 的主入口
    
    Args:
        fn: JIT 函数
        signature: 参数类型签名
        device_type: 设备类型 ("cuda" 或 "hip")
        constants: 常量参数映射
        configs: 调优配置列表
    
    Returns:
        CompiledKernel: 编译后的 kernel 对象
    """
    # 1. Python AST → Triton IR
    # 2. Triton IR → MLIR Triton Dialect
    # 3. MLIR Triton Dialect → TritonGPU Dialect
    # 4. TritonGPU Dialect → LLVM Dialect
    # 5. LLVM Dialect → PTX/HSACO
    pass
```

### 1.4.5 tools/ 模块

`tools/` 模块提供调试和分析工具：

```python
# python/triton/tools/disassembler.py
def disassemble(kernel_path):
    """
    反汇编编译后的 kernel
    
    Args:
        kernel_path: 编译产物路径（.cubin 或 .hsaco）
    """
    pass

# 使用示例
from triton.tools import disassembler
disassembler.disassemble("/tmp/triton_cache/kernel.cubin")
```

常用工具：

| 工具 | 说明 | 使用场景 |
|:---|:---|:---|
| `disassembler` | 反汇编 PTX/CUBIN | 查看生成的机器码 |
| `debugger` | 调试辅助 | 定位 kernel 错误 |
| `profiler` | 性能分析 | 分析 kernel 性能瓶颈 |

### 1.4.6 backends/ 模块

`backends/` 模块定义了硬件后端的抽象接口：

```python
# python/triton/backends/driver.py
class DriverBase:
    """GPU 驱动基类"""
    
    def __init__(self):
        pass
    
    def get_device_properties(self, device):
        """获取设备属性"""
        raise NotImplementedError
    
    def allocate(self, size):
        """分配设备内存"""
        raise NotImplementedError
    
    def launch_kernel(self, kernel, grid, *args):
        """启动 kernel"""
        raise NotImplementedError
```

```
backends/ 模块结构：

    backends/
    ├── __init__.py
    ├── driver.py              # 驱动基类
    ├── nvidia/                # NVIDIA 后端
    │   ├── __init__.py
    │   ├── driver.py          # NVIDIA 驱动实现
    │   ├── compiler.py        # NVIDIA 编译器
    │   └── utils.py           # NVIDIA 工具函数
    ├── amd/                   # AMD 后端
    │   ├── __init__.py
    │   ├── driver.py          # AMD 驱动实现
    │   ├── compiler.py        # AMD 编译器
    │   └── utils.py           # AMD 工具函数
    └── __init__.py
```

---

## 1.5 lib/ 子目录详解

`lib/` 目录包含 Triton 编译器的 C++ 核心实现，基于 MLIR 框架构建。

### 1.5.1 lib/ 目录结构

```
lib/
├── CMakeLists.txt               # lib 目录的 CMake 配置
├── Dialect/                     # MLIR 方言定义
│   ├── Triton/                  # Triton 基础方言 (tt)
│   │   ├── IR/                  # 方言 IR 定义
│   │   │   ├── TritonOps.td     # Operation 定义（TableGen）
│   │   │   ├── TritonDialect.td # 方言定义
│   │   │   └── TritonTypes.td   # Type 定义
│   │   └── Transform/           # 方言级变换 Pass
│   ├── TritonGPU/               # TritonGPU 方言 (ttg)
│   │   ├── IR/                  # GPU 特定 IR 定义
│   │   │   ├── TritonGPUOps.td  # GPU Operation 定义
│   │   │   └── TritonGPUTypes.td# GPU Type 定义（Layout 等）
│   │   └── Transform/           # GPU 优化 Pass
│   └── TritonNvidiaGPU/         # NVIDIA 特定方言
├── Conversion/                  # 方言转换 Pass
│   ├── TritonToTritonGPU/       # Triton → TritonGPU
│   ├── TritonGPUToLLVM/         # TritonGPU → LLVM
│   └── TritonToTritonGPU/       # 其他转换
├── Target/                      # 后端代码生成
│   ├── NVPTX/                   # NVIDIA PTX 生成
│   └── AMDGPU/                  # AMD GPU 代码生成
├── Analysis/                    # 分析 Pass
│   ├── Allocation.cpp           # 内存分配分析
│   └── AxisInfo.cpp             # 轴信息分析
└── Dialect/                     # 方言基础设施
    └── Triton/Transforms/       # 变换 Pass 实现
```

### 1.5.2 Dialect/ 方言定义

`Dialect/` 目录定义了 Triton 的 MLIR 方言，这是 Triton 编译器的核心 IR：

**Triton 方言（tt）**：

```tablegen
// include/triton/Dialect/Triton/IR/TritonOps.td（简化示意）
def TT_LoadOp : TT_Op<"load", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
    let summary = "Load data from memory";
    let arguments = (ins
        TT_PtrType:$ptr,          // 指针
        TT_MaskType:$mask,        // 掩码（可选）
        TT_OtherType:$other,      // 掩码外的默认值（可选）
        DefaultValuedAttr<BoolAttr, "true">:$cache,
        DefaultValuedAttr<BoolAttr, "true">:$evict
    );
    let results = (outs TT_TensorType:$result);
}
```

**TritonGPU 方言（ttg）**：

```tablegen
// include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td（简化示意）
def TritonGPU_BlockedEncodingAttr : TritonGPU_Attr<"BlockedEncoding"> {
    let mnemonic = "blocked";
    let parameters = (ins
        "ArrayRef<int>":$sizePerThread,    // 每线程处理的元素数
        "ArrayRef<int>":$threadsPerWarp,   // 每 warp 的线程数
        "ArrayRef<int>":$warpsPerCTA,      // 每 CTA 的 warp 数
        "ArrayRef<int>":$order,            // 维度顺序
        "TritonGPU_CTAEncoding":$CTALayout // CTA 布局
    );
}
```

**方言层次关系**：

```
MLIR 方言栈：

    ┌─────────────────────────────────────┐
    │         Triton 方言 (tt)            │  用户级 IR
    │   tt.load, tt.store, tt.dot, ...   │  硬件无关
    └──────────────────┬──────────────────┘
                       │ TritonToTritonGPU
                       ▼
    ┌─────────────────────────────────────┐
    │      TritonGPU 方言 (ttg)           │  硬件感知 IR
    │   ttg.convert_layout, ttg.dot, ... │  含 Layout 信息
    └──────────────────┬──────────────────┘
                       │ TritonGPUToLLVM
                       ▼
    ┌─────────────────────────────────────┐
    │        LLVM 方言 (llvm)             │  底层 IR
    │   llvm.load, llvm.store, ...       │  接近机器码
    └──────────────────┬──────────────────┘
                       │ LLVM CodeGen
                       ▼
    ┌─────────────────────────────────────┐
    │    PTX / HSACO / SPIR-V            │  目标代码
    └─────────────────────────────────────┘
```

### 1.5.3 Conversion/ 方言转换

`Conversion/` 目录实现了方言之间的转换 Pass：

```
lib/Conversion/
├── TritonToTritonGPU/
│   ├── TritonToTritonGPUPass.cpp     # Triton → TritonGPU 转换
│   └── CMakeLists.txt
├── TritonGPUToLLVM/
│   ├── TritonGPUToLLVMPass.cpp       # TritonGPU → LLVM 转换
│   ├── DotOpToLLVM.cpp               # 矩阵乘法转换
│   ├── LoadStoreOpToLLVM.cpp         # 内存操作转换
│   ├── ReduceOpToLLVM.cpp            # 归约操作转换
│   └── CMakeLists.txt
└── TritonGPUToTriton/
    └── ...                           # 反向转换（调试用）
```

**转换 Pass 的职责**：

| 转换 Pass | 输入方言 | 输出方言 | 职责 |
|:---|:---|:---|:---|
| `TritonToTritonGPU` | tt | ttg | 注入 Layout 信息，插入 layout 转换 |
| `TritonGPUToLLVM` | ttg | llvm | 展开 tile 操作为标量/向量操作 |
| `TritonGPUToTriton` | ttg | tt | 反向转换（用于调试） |

```cpp
// lib/Conversion/TritonGPUToLLVM/DotOpToLLVM.cpp（简化示意）
LogicalResult convertDotOp(DotOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) {
    // 获取操作数的 layout
    auto aEncoding = op.getA().getType().getEncoding();
    auto bEncoding = op.getB().getType().getEncoding();
    
    // 根据 layout 类型选择实现策略
    if (auto mmaEncoding = dyn_cast<NvidiaMmaEncodingAttr>(aEncoding)) {
        // 使用 Tensor Core 指令 (mma.sync)
        return convertDotOpToMMA(op, adaptor, rewriter, mmaEncoding);
    } else {
        // 使用标量 FMA 指令
        return convertDotOpToScalar(op, adaptor, rewriter);
    }
}
```

### 1.5.4 Target/ 代码生成

`Target/` 目录负责最终的目标代码生成：

```
lib/Target/
├── NVPTX/
│   ├── NVPTXTranslation.cpp          # LLVM IR → PTX
│   └── CMakeLists.txt
└── AMDGPU/
    ├── AMDGPUTranslation.cpp         # LLVM IR → HSACO
    └── CMakeLists.txt
```

```cpp
// lib/Target/NVPTX/NVPTXTranslation.cpp（简化示意）
std::string translateLLVMIRToPTX(llvm::Module &module,
                                  int cc, int ptxVersion) {
    // 1. 配置 NVPTX 后端
    auto target = llvm::TargetRegistry::lookupTarget("nvptx64-nvidia-cuda");
    
    // 2. 创建 TargetMachine
    auto machine = target->createTargetMachine(
        "nvptx64-nvidia-cuda", 
        "sm_" + std::to_string(cc),
        "+ptx" + std::to_string(ptxVersion),
        options, Reloc::PIC_
    );
    
    // 3. 生成 PTX 汇编
    llvm::SmallVector<char, 0> buffer;
    llvm::raw_svector_ostream stream(buffer);
    machine->addPassesToEmitFile(module, stream, nullptr, CGFT_AssemblyFile);
    
    return std::string(buffer.begin(), buffer.end());
}
```

### 1.5.5 include/ 头文件

`include/` 目录包含编译器的头文件定义：

```
include/
└── triton/
    ├── Dialect/
    │   ├── Triton/
    │   │   └── IR/
    │   │       ├── TritonOps.h          # Operation 声明
    │   │       ├── TritonDialect.h       # 方言声明
    │   │       └── TritonTypes.h         # Type 声明
    │   └── TritonGPU/
    │       └── IR/
    │           ├── TritonGPUOps.h
    │           └── TritonGPUTypes.h
    ├── Conversion/
    │   ├── TritonToTritonGPU/
    │   │   └── TritonToTritonGPU.h
    │   └── TritonGPUToLLVM/
    │       └── TritonGPUToLLVM.h
    └── Target/
        ├── NVPTX/
        │   └── NVPTXTranslation.h
        └── AMDGPU/
            └── AMDGPUTranslation.h
```

---

## 1.6 third_party/ 硬件后端

`third_party/` 目录包含硬件后端的插件实现，这是 Triton 支持多硬件平台的关键机制。

### 1.6.1 后端插件架构

```
third_party/
├── nvidia/                        # NVIDIA GPU 后端
│   ├── CMakeLists.txt
│   ├── backend/
│   │   ├── __init__.py            # Python 后端注册
│   │   ├── compiler.py            # NVIDIA 编译器驱动
│   │   ├── driver.py              # NVIDIA 驱动接口
│   │   └── utils.py               # 工具函数
│   ├── lib/
│   │   └── TritonNvidiaGPU/       # NVIDIA 特定方言
│   └── include/
│       └── triton/
│           └── Dialect/
│               └── TritonNvidiaGPU/
├── amd/                           # AMD GPU 后端
│   ├── CMakeLists.txt
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── compiler.py            # AMD 编译器驱动
│   │   ├── driver.py              # AMD 驱动接口
│   │   └── utils.py
│   ├── lib/
│   │   └── TritonAMDGPU/          # AMD 特定方言
│   └── include/
│       └── triton/
│           └── Dialect/
│               └── TritonAMDGPU/
└── intel/                         # Intel GPU 后端（实验性）
    └── ...
```

### 1.6.2 后端注册机制

```python
# third_party/nvidia/backend/__init__.py
from triton.backends.driver import DriverBase

class CUDADriver(DriverBase):
    """NVIDIA CUDA 驱动实现"""
    
    def __init__(self):
        super().__init__()
        self._device_count = None
    
    def get_device_properties(self, device):
        """获取 CUDA 设备属性"""
        import torch
        return torch.cuda.get_device_properties(device)
    
    def allocate(self, size):
        """分配 CUDA 设备内存"""
        import torch
        return torch.empty(size, dtype=torch.uint8, device='cuda')
    
    def launch_kernel(self, kernel, grid, *args):
        """启动 CUDA kernel"""
        # 调用 CUDA driver API
        pass

# 注册后端
def init_backend():
    """初始化 NVIDIA 后端"""
    return CUDADriver()
```

### 1.6.3 NVIDIA 后端详解

NVIDIA 后端是 Triton 最成熟的后端，支持从 Pascal（SM 6.0）到 Hopper（SM 9.0）的架构：

```python
# third_party/nvidia/backend/compiler.py
class CUDABackend:
    """NVIDIA 后端编译器"""
    
    def __init__(self):
        self.capability = self._get_capability()
    
    def _get_capability(self):
        """获取 GPU 计算能力"""
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()[0] * 10 + \
                   torch.cuda.get_device_capability()[1]
        return 80  # 默认 SM 80
    
    def compile(self, src, metadata):
        """编译 Triton IR 到 PTX/CUBIN"""
        # 1. Triton IR → MLIR
        # 2. MLIR Pass 管线（包括 layout 优化）
        # 3. LLVM IR → PTX
        # 4. PTX → CUBIN（使用 ptxas）
        pass
```

**支持的 NVIDIA 架构**：

| 架构 | SM 版本 | GPU 示例 | Triton 支持特性 |
|:---|:---|:---|:---|
| Pascal | SM 6.0 | GTX 1080, P100 | 基础支持 |
| Volta | SM 7.0 | V100 | Tensor Core v1 |
| Turing | SM 7.5 | RTX 2080, T4 | Tensor Core v2 |
| Ampere | SM 8.0/8.6 | A100, RTX 3090 | TF32, BF16 |
| Hopper | SM 9.0 | H100 | FP8, TMA |

### 1.6.4 AMD 后端详解

AMD 后端支持 ROCm 平台，包括 CDNA（MI 系列）和 RDNA（RX 系列）架构：

```python
# third_party/amd/backend/compiler.py
class HIPBackend:
    """AMD HIP 后端编译器"""
    
    def __init__(self):
        self.arch = self._get_arch()
    
    def _get_arch(self):
        """获取 GPU 架构"""
        import subprocess
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True
        )
        # 解析架构信息
        if "MI300" in result.stdout:
            return "gfx942"
        elif "MI250" in result.stdout:
            return "gfx90a"
        elif "MI210" in result.stdout:
            return "gfx90a"
        return "gfx90a"  # 默认
    
    def compile(self, src, metadata):
        """编译 Triton IR 到 HSACO"""
        # 1. Triton IR → MLIR
        # 2. MLIR Pass 管线
        # 3. LLVM IR → AMDGPU IR
        # 4. AMDGPU IR → HSACO（使用 hipcc）
        pass
```

**支持的 AMD 架构**：

| 架构 | GPU 系列 | 示例 | Triton 支持特性 |
|:---|:---|:---|:---|
| CDNA 1 | MI100 | MI100 | 基础支持 |
| CDNA 2 | MI200 | MI250X | MFMA 指令 |
| CDNA 3 | MI300 | MI300X | FP8, 更大共享内存 |
| RDNA 3 | RX 7000 | RX 7900 XTX | 实验性支持 |

---

## 1.7 构建系统详解

Triton 的构建系统涉及 CMake（C++ 部分）和 Python setuptools（Python 包部分）两个层面。

### 1.7.1 CMake 构建配置

顶层 `CMakeLists.txt` 定义了整个 C++ 编译器的构建逻辑：

```cmake
# CMakeLists.txt（简化示意，展示关键结构）

cmake_minimum_required(VERSION 3.18)
project(triton LANGUAGES C CXX)

# ==================== 选项配置 ====================
option(TRITON_BUILD_TUTORIALS "Build C++ Triton tutorials" OFF)
option(TRITON_BUILD_UNIT_TESTS "Build C++ Triton unit tests" OFF)
option(TRITON_BUILD_PYTHON_MODULE "Build Python Triton bindings" ON)

# ==================== LLVM 依赖 ====================
# Triton 使用预编译的 LLVM，从 PyPI 下载
set(TRITON_LLVM_BUILD_DIR "${CMAKE_BINARY_DIR}/llvm")

# 下载预编译 LLVM
include(FetchContent)
FetchContent_Declare(
    llvm-project
    URL https://github.com/llvm/llvm-project/releases/download/...
    URL_HASH SHA256=...
)
FetchContent_MakeAvailable(llvm-project)

# ==================== 核心库构建 ====================
# Triton 编译器核心库
add_subdirectory(lib)

# 方言定义
add_subdirectory(lib/Dialect/Triton)
add_subdirectory(lib/Dialect/TritonGPU)

# 转换 Pass
add_subdirectory(lib/Conversion/TritonToTritonGPU)
add_subdirectory(lib/Conversion/TritonGPUToLLVM)

# 目标代码生成
add_subdirectory(lib/Target/NVPTX)
add_subdirectory(lib/Target/AMDGPU)

# ==================== Python 绑定 ====================
if(TRITON_BUILD_PYTHON_MODULE)
    add_subdirectory(python)
endif()
```

### 1.7.2 LLVM 依赖管理

Triton 使用预编译的 LLVM 二进制，避免用户从源码编译 LLVM（通常需要数小时）：

```cmake
# LLVM 版本配置（在 CMakeLists.txt 或独立配置文件中）
set(TRITON_LLVM_VERSION "llvm-18")
set(TRITON_LLVM_HASH "abc123...")

# 平台特定的 LLVM URL
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(LLVM_URL "https://triton-lang.s3.amazonaws.com/llvm/linux-x86_64/${TRITON_LLVM_VERSION}.tar.gz")
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(LLVM_URL "https://triton-lang.s3.amazonaws.com/llvm/linux-aarch64/${TRITON_LLVM_VERSION}.tar.gz")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(LLVM_URL "https://triton-lang.s3.amazonaws.com/llvm/macos-arm64/${TRITON_LLVM_VERSION}.tar.gz")
endif()
```

**LLVM 在 Triton 中的作用**：

```
LLVM 在 Triton 编译管线中的位置：

    Triton IR
        │
        ▼
    ┌─────────────────────────────┐
    │    MLIR 基础设施             │  LLVM 提供
    │    - 方言框架               │
    │    - Pass 管理器            │
    │    - Pattern Rewriting      │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │    LLVM IR                  │  LLVM 提供
    │    - 优化 Pass              │
    │    - 指令选择               │
    └──────────────┬──────────────┘
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
    ┌──────────┐      ┌──────────┐
    │ NVPTX    │      │ AMDGPU   │  LLVM 提供
    │ 后端     │      │ 后端     │
    └────┬─────┘      └────┬─────┘
         │                 │
         ▼                 ▼
       PTX              HSACO
```

### 1.7.3 Python 包构建

Python 包通过 `setup.py` 构建，它会触发 CMake 编译 C++ 库：

```python
# python/setup.py（简化示意）

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import os

class CMakeBuild(build_ext):
    """自定义构建命令：调用 CMake 编译 C++ 库"""
    
    def build_extension(self, ext):
        # 获取源码目录
        source_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(source_dir, "build")
        
        # 配置 CMake
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DTRITON_BUILD_PYTHON_MODULE=ON",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        
        subprocess.check_call(
            ["cmake", source_dir] + cmake_args,
            cwd=build_dir
        )
        
        # 编译
        subprocess.check_call(
            ["cmake", "--build", build_dir, "--", "-j"],
            cwd=build_dir
        )

setup(
    name="triton",
    version="2.3.1",
    packages=find_packages(),
    ext_modules=[...],  # C++ 扩展
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        "cmake>=3.18",
        "ninja",
        "filelock",
        "torch>=2.0",
    ],
)
```

**构建流程图**：

```
pip install -e python
        │
        ▼
┌─────────────────┐
│ setup.py        │
│ 解析配置        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CMake 配置      │
│ cmake ..        │
│ 下载 LLVM      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CMake 编译      │
│ ninja -j        │
│ 生成 libtriton  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Python 安装     │
│ pip install -e  │
│ 创建 .egg-link │
└─────────────────┘
```

### 1.7.4 开发模式安装

对于开发者，推荐使用 editable 安装模式：

```bash
# editable 安装（修改 Python 代码后无需重新安装）
pip install -e python

# 或者使用 setup.py develop
cd python && python setup.py develop && cd ..

# 验证 editable 安装
python -c "
import triton
print(f'Triton path: {triton.__file__}')
# 应该指向源码目录，而不是 site-packages
"
```

editable 安装的工作原理：

```
源码目录: /home/user/triton/python/triton/
    │
    │  pip install -e python
    ▼
site-packages/triton.egg-link
    │
    │  指向源码目录
    ▼
/home/user/triton/python/triton/

修改源码后：
/home/user/triton/python/triton/core.py
    │
    │  自动生效（无需重新安装）
    ▼
import triton  # 使用修改后的代码
```

---

## 1.8 开发环境配置

### 1.8.1 VS Code 配置

VS Code 是开发 Triton 的推荐 IDE，支持 Python 和 C++ 的混合调试。

**推荐的 VS Code 扩展**：

```jsonc
// .vscode/extensions.json
{
    "recommendations": [
        "ms-python.python",           // Python 支持
        "ms-vscode.cpptools",         // C++ 支持
        "ms-vscode.cmake-tools",      // CMake 支持
        "xaver.clang-format",         // 代码格式化
        "ms-python.black-formatter",  // Python 格式化
        "charliermarsh.ruff",         // Python lint
    ]
}
```

**VS Code 工作区配置**：

```jsonc
// .vscode/settings.json
{
    // Python 配置
    "python.defaultInterpreterPath": "${workspaceFolder}/triton-dev/bin/python",
    "python.analysis.extraPaths": [
        "${workspaceFolder}/python"
    ],
    
    // C++ 配置
    "C_Cpp.default.includePath": [
        "${workspaceFolder}/include",
        "${workspaceFolder}/build/include",
        "${workspaceFolder}/build/llvm/include"
    ],
    "C_Cpp.default.cppStandard": "c++17",
    
    // CMake 配置
    "cmake.buildDirectory": "${workspaceFolder}/build",
    "cmake.configureArgs": [
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DTRITON_BUILD_PYTHON_MODULE=ON"
    ],
    
    // 文件排除
    "files.exclude": {
        "**/build/**": true,
        "**/__pycache__/**": true,
        "**/.triton/**": true
    }
}
```

**Python 调试配置**：

```jsonc
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Triton Script",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {
                "TRITON_PRINT_AUTOTUNING": "1",
                "TRITON_PRINT_IR": "1",
                "CUDA_LAUNCH_BLOCKING": "1"
            }
        },
        {
            "name": "Debug Triton Tests",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": ["${file}", "-v", "-s"],
            "console": "integratedTerminal"
        }
    ]
}
```

### 1.8.2 关键环境变量

Triton 提供了多个环境变量用于开发调试：

| 环境变量 | 值 | 说明 |
|:---|:---|:---|
| `TRITON_PRINT_AUTOTUNING` | `1` | 打印自动调优过程 |
| `TRITON_PRINT_IR` | `1` | 打印中间 IR |
| `TRITON_DUMP_DIR` | 路径 | IR dump 目录 |
| `TRITON_CACHE_DIR` | 路径 | 编译缓存目录 |
| `CUDA_LAUNCH_BLOCKING` | `1` | 同步 CUDA 执行（便于调试） |
| `TRITON_DEBUG` | `1` | 启用调试模式 |

```bash
# 设置环境变量（临时）
export TRITON_PRINT_AUTOTUNING=1
export TRITON_PRINT_IR=1
export TRITON_DUMP_DIR=/tmp/triton_ir_dump

# 运行脚本时自动 dump IR
python my_kernel.py

# 查看 dump 的 IR
ls /tmp/triton_ir_dump/
# 输出示例:
# kernel_00ttir.mlir          # Triton IR
# kernel_01ttgir.mlir         # TritonGPU IR
# kernel_02llvm.mlir          # LLVM Dialect IR
# kernel_03llir.ll            # LLVM IR
# kernel_04.ptx               # PTX 汇编
# kernel_05.cubin             # 编译后的二进制
```

### 1.8.3 IR Dump 配置详解

IR dump 是理解 Triton 编译过程的最重要工具：

```python
# 方法一：通过环境变量（全局生效）
import os
os.environ["TRITON_PRINT_IR"] = "1"
os.environ["TRITON_DUMP_DIR"] = "/tmp/triton_ir"

# 方法二：通过编译选项（单个 kernel）
@triton.jit
def my_kernel(...):
    ...

# 编译时指定 dump
compiled = triton.compile(
    my_kernel,
    signature="*fp32,*fp32,i32",
    print_ir=True,           # 打印 IR
    dump_ir="/tmp/triton_ir" # dump 到目录
)
```

**IR dump 文件解读**：

```
dump 目录结构：

/tmp/triton_ir/
├── kernel_vec_add/
│   ├── 00_ttir.mlir              # 原始 Triton IR
│   │   # tt.func @kernel_vec_add(%arg0: !tt.ptr<f32>, ...)
│   │   #   %0 = tt.load %arg0 : tensor<256xf32>
│   │   #   %1 = tt.load %arg1 : tensor<256xf32>
│   │   #   %2 = arith.addf %0, %1 : tensor<256xf32>
│   │   #   tt.store %arg2, %2 : tensor<256xf32>
│   │
│   ├── 01_ttgir.mlir             # TritonGPU IR（带 layout）
│   │   # tt.func @kernel_vec_add(...) attributes {noinline = false} {
│   │   #   %0 = ttg.convert_layout %arg0 : tensor<256xf32, #blocked>
│   │   #   ...
│   │
│   ├── 02_llvm.mlir              # LLVM Dialect IR
│   │   # llvm.func @kernel_vec_add(...) {
│   │   #   %0 = llvm.load %arg0 : !llvm.ptr -> vector<4xf32>
│   │   #   ...
│   │
│   ├── 03_llvm_opt.ll            # 优化后的 LLVM IR
│   │   ; Function Attrs: nounwind
│   │   define void @kernel_vec_add(...) {
│   │   entry:
│   │     %0 = load <4 x float>, ptr %arg0, align 16
│   │     ...
│   │
│   ├── 04_ptx.s                  # PTX 汇编
│   │   .visible .entry kernel_vec_add(
│   │     .param .u64 kernel_vec_add_param_0,
│   │     ...
│   │   ) {
│   │     ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd1];
│   │     ...
│   │
│   └── 05_cubin                  # CUBIN 二进制
│       (二进制文件)
```

### 1.8.4 GDB 调试 C++ 代码

对于需要调试 C++ 编译器代码的场景：

```bash
# 以 Debug 模式编译
mkdir build && cd build
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-g -O0" \
    -DTRITON_BUILD_PYTHON_MODULE=ON
ninja -j$(nproc)

# 使用 GDB 调试 Python 脚本中的 C++ 代码
gdb -ex run python my_kernel.py

# 在 GDB 中设置断点
(gdb) b lib/Conversion/TritonGPUToLLVM/DotOpToLLVM.cpp:42
(gdb) run
```

### 1.8.5 性能分析工具

```bash
# NVIDIA Nsight Systems（系统级分析）
nsys profile -o report python my_kernel.py

# NVIDIA Nsight Compute（kernel 级分析）
ncu -o report python my_kernel.py

# 使用 Triton 内置的性能测试
python -m triton.testing.benchmark_matmul
```

---

## 1.9 验证安装：第一个 Triton Kernel

### 1.9.1 向量加法 Kernel

让我们编写并运行第一个 Triton kernel——向量加法：

```python
# 第一个 Triton kernel: 向量加法
# 文件: 01_vec_add.py

import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr,        # 第一个输入向量的指针
    y_ptr,        # 第二个输入向量的指针
    output_ptr,   # 输出向量的指针
    n_elements,   # 向量长度
    BLOCK_SIZE: tl.constexpr,  # 每个 program 处理的元素数
):
    # 获取当前 program 的 ID
    pid = tl.program_id(axis=0)
    
    # 计算当前 program 负责的元素范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 创建掩码，防止越界访问
    mask = offsets < n_elements
    
    # 从全局内存加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 执行加法运算
    output = x + y
    
    # 将结果写回全局内存
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x, y):
    """向量加法的 Python 封装"""
    # 分配输出张量
    output = torch.empty_like(x)
    
    # 检查输入
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    n_elements = x.shape[0]
    
    # 计算 grid 大小（向上取整）
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # 启动 kernel
    vector_add_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# 测试
if __name__ == "__main__":
    # 创建测试数据
    size = 1 << 20  # 1M 元素
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    
    # 运行 Triton kernel
    output_triton = vector_add(x, y)
    
    # 运行 PyTorch 参考实现
    output_torch = x + y
    
    # 验证正确性
    triton.testing.assert_close(output_triton, output_torch)
    print("✅ 向量加法验证通过！")
```

运行脚本：

```bash
python 01_vec_add.py
# ✅ 向量加法验证通过！
```

### 1.9.2 代码解析

让我们逐行解析这个 kernel 的关键部分：

```python
@triton.jit
def vector_add_kernel(
    x_ptr,                # 参数 1: 指针类型（自动推断）
    y_ptr,                # 参数 2: 指针类型
    output_ptr,           # 参数 3: 指针类型
    n_elements,           # 参数 4: 整数类型（自动特化）
    BLOCK_SIZE: tl.constexpr,  # 参数 5: 编译期常量
):
```

**参数类型推断**：

| 参数 | 类型 | 说明 |
|:---|:---|:---|
| `x_ptr` | `!tt.ptr<f32>` | 从调用处的 tensor 推断 |
| `y_ptr` | `!tt.ptr<f32>` | 同上 |
| `output_ptr` | `!tt.ptr<f32>` | 同上 |
| `n_elements` | `i32` | 从 Python int 推断 |
| `BLOCK_SIZE` | `constexpr` | 编译期常量，触发特化 |

```python
    # 获取当前 program 的 ID
    pid = tl.program_id(axis=0)
```

`tl.program_id(axis=0)` 返回当前 program 在第 0 维上的 ID。这类似于 CUDA 的 `blockIdx.x`。

```python
    # 计算当前 program 负责的元素范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

`tl.arange(0, BLOCK_SIZE)` 创建一个一维张量 `[0, 1, 2, ..., BLOCK_SIZE-1]`。加上 `block_start` 得到当前 program 负责的全局索引。

```python
    # 创建掩码，防止越界访问
    mask = offsets < n_elements
```

当 `n_elements` 不是 `BLOCK_SIZE` 的整数倍时，最后一个 program 的部分索引会越界。掩码确保这些位置不会被访问。

```python
    # 从全局内存加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
```

`tl.load` 从全局内存加载数据。`mask` 参数指定哪些位置是有效的。被掩码的位置不会触发内存访问。

```python
    # 执行加法运算
    output = x + y
```

这是普通的张量加法，编译器会自动向量化。

```python
    # 将结果写回全局内存
    tl.store(output_ptr + offsets, output, mask=mask)
```

`tl.store` 将结果写回全局内存。同样使用掩码防止越界写入。

### 1.9.3 矩阵乘法 Kernel

更复杂的例子——矩阵乘法：

```python
# 矩阵乘法 kernel
# 文件: 01_matmul.py

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,  # 矩阵指针
    M, N, K,              # 矩阵维度
    stride_am, stride_ak, # A 矩阵的 stride
    stride_bk, stride_bn, # B 矩阵的 stride
    stride_cm, stride_cn, # C 矩阵的 stride
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 获取二维 program ID
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # 计算当前 block 的起始位置
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 沿 K 维度循环
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载 A 的一个 block
        a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # 加载 B 的一个 block
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # 累加矩阵乘法结果
        accumulator += tl.dot(a, b)
        
        # 推进 K 维度指针
        offs_k += BLOCK_SIZE_K
    
    # 类型转换并存储结果
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(A, B):
    """矩阵乘法的 Python 封装"""
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


# 测试
if __name__ == "__main__":
    M, N, K = 1024, 1024, 1024
    A = torch.randn((M, K), device='cuda', dtype=torch.float16)
    B = torch.randn((K, N), device='cuda', dtype=torch.float16)
    
    # Triton 实现
    C_triton = matmul(A, B)
    
    # PyTorch 参考
    C_torch = torch.matmul(A, B)
    
    # 验证正确性
    triton.testing.assert_close(C_triton, C_torch, atol=1e-2, rtol=1e-2)
    print("✅ 矩阵乘法验证通过！")
    
    # 性能基准测试
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['M', 'N', 'K'],
            x_vals=[128 * i for i in range(2, 33)],
            line_arg='provider',
            line_vals=['cublas', 'triton'],
            line_names=['cuBLAS', 'Triton'],
            styles=[('green', '-'), ('blue', '--')],
            ylabel='TFLOPS',
            plot_name='matmul-performance',
            args={},
        )
    )
    def benchmark(M, N, K, provider):
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        if provider == 'cublas':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.matmul(a, b)
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: matmul(a, b)
            )
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)
    
    benchmark.run(save_path='.', print_data=True)
```

运行矩阵乘法示例：

```bash
python 01_matmul.py
# ✅ 矩阵乘法验证通过！
# 
# matmul-performance:
#       M     cuBLAS    Triton
#    256.0    142.31    128.45
#    384.0    156.78    148.92
#    512.0    168.23    161.34
#    ...
```

### 1.9.4 Softmax Kernel

融合 Softmax kernel 展示了 Triton 的归约操作能力：

```python
# 融合 Softmax kernel
# 文件: 01_softmax.py

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexplore,
):
    # 获取当前 program ID（每行一个 program）
    row_idx = tl.program_id(0)
    
    # 计算当前行的起始指针
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # 生成列偏移
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # 创建掩码
    mask = col_offsets < n_cols
    
    # 加载当前行数据
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # 计算 softmax
    # 步骤 1: 减去最大值（数值稳定性）
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    
    # 步骤 2: 计算分母
    denominator = tl.sum(numerator, axis=0)
    
    # 步骤 3: 归一化
    softmax_output = numerator / denominator
    
    # 写回结果
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x):
    """Softmax 的 Python 封装"""
    n_rows, n_cols = x.shape
    
    # 选择合适的 BLOCK_SIZE
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # 分配输出
    output = torch.empty_like(x)
    
    # 启动 kernel（每行一个 program）
    grid = (n_rows,)
    softmax_kernel[grid](
        output, x,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# 测试
if __name__ == "__main__":
    x = torch.randn(1024, 512, device='cuda')
    
    # Triton 实现
    output_triton = softmax(x)
    
    # PyTorch 参考
    output_torch = torch.softmax(x, dim=1)
    
    # 验证
    triton.testing.assert_close(output_triton, output_torch)
    print("✅ Softmax 验证通过！")
```

### 1.9.5 性能基准测试

使用 `triton.testing` 模块进行性能对比：

```python
# 性能基准测试
# 文件: 01_benchmark.py

import torch
import triton
import triton.language as tl

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],                          # x 轴变量
        x_vals=[2**i for i in range(10, 25)],   # x 轴取值范围
        x_log=True,                              # x 轴对数坐标
        line_arg='provider',                     # 不同实现的参数
        line_vals=['triton', 'torch'],           # 实现列表
        line_names=['Triton', 'PyTorch'],        # 显示名称
        styles=[('blue', '-'), ('green', '--')], # 线条样式
        ylabel='GB/s',                           # y 轴标签
        plot_name='vector-add-performance',      # 图表标题
        args={},                                 # 其他固定参数
    )
)
def benchmark(N, provider):
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = torch.randn(N, device='cuda', dtype=torch.float32)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x + y, quantiles=quantiles
        )
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vector_add(x, y), quantiles=quantiles
        )
    
    # 计算带宽（读 + 写）
    gbps = lambda ms: 3 * N * 4 * 1e-9 / (ms * 1e-3)  # 3 次访问 × 4 字节
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    benchmark.run(save_path='.', print_data=True)
```

运行基准测试：

```bash
python 01_benchmark.py
```

输出示例：

```
vector-add-performance:
          N      Triton    PyTorch
       1024.0    156.23    142.56
       2048.0    234.56    221.34
       4096.0    345.67    332.45
       ...
    16777216.0    892.34    876.23
```

<div data-component="FirstKernelInteractiveDemo"></div>

[组件：FirstKernelInteractiveDemo - 交互式 kernel 演示，可调整参数（BLOCK_SIZE、向量大小）并实时查看性能]

### 1.9.6 常见问题排查

如果遇到安装或运行问题，请参考以下排查指南：

| 错误信息 | 可能原因 | 解决方案 |
|:---|:---|:---|
| `ModuleNotFoundError: No module named 'triton'` | Triton 未安装 | `pip install triton` |
| `CUDA error: no kernel image available` | GPU 架构不兼容 | 升级 Triton 或 CUDA |
| `RuntimeError: Triton requires CUDA 11.7+` | CUDA 版本过低 | 升级 CUDA Toolkit |
| `ImportError: libtriton.so not found` | C++ 库未编译 | 重新从源码编译 |
| `LLVM ERROR: ...` | LLVM 版本不匹配 | 清理 build 目录重新编译 |
| `AssertionError: ...assert_close...` | 数值精度问题 | 检查 atol/rtol 参数 |

```bash
# 清理 Triton 缓存
rm -rf ~/.triton/cache
rm -rf /tmp/triton_*

# 重新安装
pip uninstall triton -y
pip install triton

# 如果从源码安装出错，清理后重新编译
cd triton
rm -rf build/
pip install -e python
```

---

## 本章小结

本章详细介绍了 Triton 开发环境的搭建过程与源码结构。核心要点如下：

1. **四种安装方式**各有适用场景：PyPI 最简单适合快速体验，源码编译适合开发者，Docker 适合 CI/CD，conda-forge 适合多环境管理。

2. **依赖环境**包括 Python 3.8+、CUDA 11.7+ 或 ROCm 5.6+、LLVM（预编译）、PyTorch（可选但推荐）。理解依赖关系是排查问题的基础。

3. **仓库目录结构**遵循 Python 前端与 C++ 后端分离的设计：`python/` 包含用户接口，`lib/` 包含编译器核心，`third_party/` 包含硬件后端。

4. **python/triton/ 模块**是用户的主要接口：`language/` 定义编程 API，`runtime/` 管理 JIT 编译，`compiler/` 驱动编译过程，`backends/` 抽象硬件差异。

5. **lib/ 模块**是编译器的核心：`Dialect/` 定义 MLIR 方言，`Conversion/` 实现方言转换，`Target/` 生成目标代码。

6. **CMake 构建系统**管理 C++ 编译，通过 `setup.py` 与 Python 包集成。LLVM 依赖通过预编译二进制管理。

7. **开发调试工具**包括 VS Code 配置、环境变量（`TRITON_PRINT_IR`、`TRITON_DUMP_DIR`）、IR dump 机制。

8. **验证安装**的最佳方式是运行向量加法和矩阵乘法 kernel，使用 `triton.testing.assert_close` 验证正确性。

---

## 思考题

### 概念理解题

1. **安装方式选择**：如果你要在生产环境中部署使用 Triton 的服务，你会选择哪种安装方式？为什么？考虑因素包括：版本稳定性、依赖管理、容器化支持。

2. **LLVM 依赖**：为什么 Triton 选择使用预编译的 LLVM 而不是让用户提供自己的 LLVM？这种设计有什么优缺点？

3. **后端插件架构**：Triton 的 `third_party/` 目录采用了插件机制来支持不同硬件。这种架构有什么优势？如果要添加对新硬件（如 Intel GPU）的支持，需要修改哪些文件？

4. **JIT 缓存**：Triton 的 JIT 编译会缓存编译结果。缓存的 key 是什么？什么情况下缓存会失效？为什么要这样设计？

### 实践题

5. **环境搭建**：在你的机器上完成 Triton 的安装（推荐从源码编译），并成功运行本章的向量加法和矩阵乘法示例。记录你遇到的问题和解决方案。

6. **IR Dump 分析**：使用 `TRITON_PRINT_IR=1` 环境变量运行向量加法 kernel，分析 dump 出的各个阶段的 IR（Triton IR、TritonGPU IR、LLVM IR、PTX）。每个阶段的 IR 有什么特点？

7. **修改 kernel**：修改向量加法 kernel，使其支持：
   - 不同数据类型（FP16、BF16、INT32）
   - 二维向量（矩阵）加法
   - 带有缩放因子的加法：`output = alpha * x + beta * y`

8. **性能分析**：使用 `triton.testing.perf_report` 对向量加法进行性能基准测试，分析：
   - 不同向量大小下的带宽利用率
   - 不同 `BLOCK_SIZE` 对性能的影响
   - Triton 实现与 PyTorch 原生实现的性能对比

### 设计思考题

9. **构建系统设计**：Triton 使用 CMake 构建 C++ 部分，使用 setuptools 构建 Python 包。为什么不用单一的构建系统？这种混合构建方式有什么挑战？

10. **源码组织**：Triton 的 `lib/Dialect/` 目录下有 `Triton/`、`TritonGPU/`、`TritonNvidiaGPU/` 等子目录。为什么要将方言拆分成这么多层？每层的职责边界是什么？

11. **跨平台支持**：Triton 目前支持 NVIDIA 和 AMD GPU。如果要支持 Apple Silicon（MPS 后端），你需要在 `third_party/` 下创建什么文件？需要实现哪些接口？

### 进阶题

12. **自定义编译选项**：研究 `python/triton/compiler/` 目录，找到控制以下行为的选项：
    - 是否启用自动调优
    - 是否打印 IR
    - 选择优化级别（-O0, -O1, -O2, -O3）
    - 如何添加自定义的编译 Pass

13. **调试 C++ 代码**：以 Debug 模式编译 Triton，使用 GDB 或 LLDB 调试 `lib/Conversion/TritonGPUToLLVM/` 中的代码。设置断点在 `DotOpToLLVM.cpp` 的 `convertDotOp` 函数上，观察矩阵乘法的编译过程。

14. **构建自定义 Docker 镜像**：基于 `nvidia/cuda:12.2.0-devel-ubuntu22.04` 创建一个包含 Triton、PyTorch 和常用开发工具的 Docker 镜像。要求：
    - 支持源码编译模式（editable install）
    - 包含调试工具（GDB、nsys、ncu）
    - 配置好 VS Code Remote Container 支持
