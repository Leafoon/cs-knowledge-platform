---
title: "Chapter 1: 开发环境搭建与源码结构"
description: "掌握 TileLang 的完整安装流程（pip 源码编译、Docker），理解官方仓库（tile-lang/tile-lang）的目录结构与构建系统（CMake/TVM），配置 CUDA/HIP/Ascend C 开发环境。"
updated: "2025-06-11"
---

# Chapter 1: 开发环境搭建与源码结构

<div data-component="VersionCompatibilityMatrix"></div>

> [!NOTE]
> **学习目标**
>
> - 掌握 TileLang 的完整安装流程（pip 源码编译、Docker）
> - 理解官方仓库 tile-lang/tile-lang 的目录结构
> - 了解 CMake + TVM 的构建系统
> - 配置 CUDA/HIP/Ascend C 开发环境
> - 运行验证脚本确认环境正确

---

## 1. 安装前准备

### 1.1 系统要求

在安装 TileLang 之前，需要确认系统满足以下基本要求：

| 组件 | 最低要求 | 推荐配置 |
|:---|:---|:---|
| **操作系统** | Ubuntu 20.04 / CentOS 7 | Ubuntu 22.04 |
| **Python** | 3.8+ | 3.10+ |
| **GCC/G++** | 9.0+ | 11.0+ |
| **CMake** | 3.18+ | 3.24+ |
| **LLVM** | 15+ | 17+ |
| **CUDA Toolkit** | 11.8 (NVIDIA) | 12.1+ |
| **ROCm** | 5.6 (AMD) | 6.0+ |
| **CANN** | 8.0 (昇腾) | 8.0+ |
| **内存** | 16 GB | 32 GB+ |
| **磁盘空间** | 20 GB | 50 GB+ |

```bash
# 检查系统版本
lsb_release -a

# 检查 Python 版本
python3 --version

# 检查 GCC 版本
gcc --version

# 检查 CMake 版本
cmake --version

# 检查 CUDA 版本 (NVIDIA)
nvcc --version
nvidia-smi

# 检查 ROCm 版本 (AMD)
rocm-smi
hipcc --version
```

在安装 TileLang 之前，务必先确认当前系统的基本工具链版本。`lsb_release -a` 用于查看操作系统发行版信息，`python3 --version` 和 `gcc --version` 分别检查 Python 和编译器版本，`cmake --version` 确认构建工具就绪。对于 NVIDIA GPU 用户，`nvcc --version` 和 `nvidia-smi` 可以同时验证 CUDA Toolkit 版本和 GPU 驱动状态；对于 AMD GPU 用户，则使用 `rocm-smi` 和 `hipcc --version`。建议在开始安装前逐一执行这些命令，确保所有依赖版本满足最低要求，避免后续编译时出现版本不兼容的错误。

### 1.2 依赖项安装

#### Ubuntu/Debian 系统

```bash
# 更新包管理器
sudo apt update && sudo apt upgrade -y

# 安装基础依赖
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    python3-venv \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm-17-dev \
    libclang-17-dev

# 安装 Python 包管理工具
pip install --upgrade pip setuptools wheel
```

Ubuntu/Debian 系统的依赖安装使用 `apt` 包管理器。`build-essential` 包含了 GCC/G++ 等核心编译工具链，`cmake` 和 `git` 分别用于构建和版本控制。Python 相关的 `python3-dev`、`python3-pip`、`python3-venv` 提供了完整的 Python 开发环境。值得注意的是，TileLang 依赖 LLVM 进行编译优化，因此需要安装 `llvm-17-dev` 和 `libclang-17-dev`。`libffi-dev` 和 `libssl-dev` 是 Python C 扩展的必要依赖。最后通过 `pip install --upgrade pip setuptools wheel` 确保 Python 包管理工具链为最新版本，避免后续安装 wheel 包时出现问题。

#### CentOS/RHEL 系统

```bash
# 安装基础依赖
sudo yum groupinstall -y "Development Tools"
sudo yum install -y \
    cmake3 \
    git \
    python3-devel \
    openssl-devel \
    libffi-devel \
    zlib-devel \
    bzip2-devel \
    readline-devel \
    sqlite-devel \
    wget \
    curl

# CentOS 7 需要安装更新的 GCC
sudo yum install -y devtoolset-11
source /opt/rh/devtoolset-11/enable
```

CentOS/RHEL 系统与 Ubuntu/Debian 差异较大，使用 `yum` 包管理器。`groupinstall "Development Tools"` 一次性安装 GCC、make 等完整编译工具链。`cmake3` 是 CentOS 7 上 CMake 的包名（与 Ubuntu 的 `cmake` 不同）。CentOS 7 自带的 GCC 4.8.5 版本过旧，不满足 TileLang 对 C++17 的要求，因此必须通过 `devtoolset-11` 安装 GCC 11，再用 `source` 命令临时启用新版编译器。常见错误是忘记执行 `source /opt/rh/devtoolset-11/enable`，导致仍使用旧 GCC 编译失败，报出 C++17 语法不支持的错误。

### 1.3 CUDA 环境配置 (NVIDIA GPU)

```bash
# 方法 1: 使用 NVIDIA 官方仓库安装 CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4

# 方法 2: 使用 runfile 安装
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run

# 配置环境变量
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
nvcc --version
nvidia-smi
```

CUDA Toolkit 安装有 apt 仓库和 runfile 两种方式。apt 方式推荐大多数用户使用，能自动处理依赖关系；runfile 适合需要自定义安装路径或多 CUDA 版本共存的场景。安装后必须配置 `CUDA_HOME`、`PATH` 和 `LD_LIBRARY_PATH` 三个环境变量，并通过 `source ~/.bashrc` 使其在当前终端生效。常见错误是修改 `.bashrc` 后未执行 source 命令，导致 `nvcc` 路径无效，TileLang 编译时找不到 CUDA 工具链。验证时 `nvcc --version` 和 `nvidia-smi` 应同时返回正确信息。

```bash
# 检查 GPU 计算能力
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}, Compute Capability: {cap[0]}.{cap[1]}')
else:
    print('CUDA not available')
"
```

此脚本通过 PyTorch 检查 GPU 计算能力（Compute Capability），这是 TileLang 能否正常运行的关键指标。`torch.cuda.is_available()` 先确认 CUDA 运行时可用，`get_device_capability()` 返回架构主次版本号（如 8.0 表示 Ampere、9.0 表示 Hopper）。TileLang 依赖 Tensor Core 指令，要求计算能力不低于 8.0。低于此版本的 GPU（如 V100 的 7.0、T4 的 7.5）无法充分利用 tile 级优化，部分功能不可用或性能大幅下降。部署前务必确认 GPU 型号满足最低架构要求。

### 1.4 ROCm 环境配置 (AMD GPU)

```bash
# 安装 ROCm 仓库
sudo apt install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install -y python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME

# 添加 ROCm 仓库
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.0 jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install -y rocm-hip-sdk rocm-hip-runtime-devel

# 配置环境变量
echo 'export ROCM_PATH=/opt/rocm' >> ~/.bashrc
echo 'export PATH=$ROCM_PATH/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
rocm-smi
hipcc --version
```

ROCm 是 AMD GPU 的并行计算平台，与 NVIDIA CUDA 对等。配置前需确保内核头文件匹配（`linux-headers-$(uname -r)`），并将当前用户加入 `render,video` 组以获得 GPU 直接访问权限（需重新登录生效）。仓库通过 GPG 密钥签名确保安全性，安装包 `rocm-hip-sdk` 和 `rocm-hip-runtime-devel` 提供 HIP（Heterogeneous-compute Interface for Portability）开发环境。`ROCM_PATH`、`PATH` 和 `LD_LIBRARY_PATH` 三个环境变量是 Rocm 运行的基础，TileLang 通过 HIP 编译器 `hipcc` 将 IR 翻译为 AMD GPU 可执行代码。验证时 `rocm-smi` 应输出 GPU 状态，`hipcc --version` 返回编译器版本。

### 1.5 CANN 环境配置 (华为昇腾)

```bash
# 安装 CANN (Compute Architecture for Neural Networks)
# 请从华为昇腾官网下载对应版本的 CANN Toolkit
# https://www.hiascend.com/developer/cann-community

# 假设安装包已下载
chmod +x Ascend-cann-toolkit_8.0.RC1.alpha002_linux-x86_64.run
./Ascend-cann-toolkit_8.0.RC1.alpha002_linux-x86_64.run --install

# 配置环境变量
source ~/.bashrc
echo 'export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest' >> ~/.bashrc
echo 'export PATH=$ASCEND_HOME_PATH/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
npu-smi info
```

这段命令对应 1.5 CANN 环境配置 (华为昇腾) 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

CANN（Compute Architecture for Neural Networks）是华为昇腾芯片的异构计算架构，相当于 NVIDIA 的 CUDA Toolkit。安装包需从华为昇腾官网手动下载，目前不提供 apt/pip 等包管理方式。`chmod +x` 赋予安装脚本执行权限后运行 `.run` 安装器。环境变量 `ASCEND_HOME_PATH` 指向 toolkit 根目录，`PATH` 和 `LD_LIBRARY_PATH` 确保编译器和运行时库正确链接。TileLang 通过 CANN 后端生成适配昇腾芯片的 kernel 代码，支持 Ascend 910B/910C 等型号。验证命令 `npu-smi info` 类似 NVIDIA 的 `nvidia-smi`，用于查看 NPU 状态和利用率。

---

## 2. TileLang 安装方法

### 2.1 方法一：pip 安装 (推荐)

```bash
# 最简单的安装方式
pip install tilelang

# 指定版本安装
pip install tilelang==0.1.0

# 安装开发版本 (从 GitHub)
pip install git+https://github.com/tile-ai/tilelang.git

# 安装并指定 CUDA 架构
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" pip install tilelang
```

pip 安装是最快捷的 TileLang 安装方式，适合大多数用户。直接 `pip install tilelang` 会从 PyPI 拉取预编译的 wheel 包。若需指定 GPU 架构优化编译，可使用 `TORCH_CUDA_ARCH_LIST` 环境变量，列出目标 GPU 的计算能力版本，多个架构用分号分隔。pip 方式也支持直接从 GitHub 安装开发分支（`pip install git+https://...`），方便体验最新特性。不足之处是预编译包可能未针对特定 GPU 架构优化，对性能极致追求的场景建议采用源码编译安装以获得最优性能。

### 2.2 方法二：源码编译安装

```bash
# 克隆仓库
git clone --recursive https://github.com/tile-ai/tilelang.git
cd tilelang

# 创建虚拟环境 (推荐)
python3 -m venv venv
source venv/bin/activate

# 安装 Python 依赖
pip install -r requirements.txt

# 设置 TVM 子模块
# TileLang 内置了 TVM 作为子模块
git submodule update --init --recursive

# 配置构建
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    -DUSE_LLVM=/usr/bin/llvm-config-17 \
    -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")

# 编译
make -j$(nproc)

# 安装 Python 包
cd ..
pip install -e .

# 验证安装
python3 -c "import tilelang; print(f'TileLang version: {tilelang.__version__}')"
```

源码编译安装虽然步骤更多，但能针对当前硬件环境生成最优代码，性能通常优于 pip 预编译包。`git clone --recursive` 会同时拉取 TVM 等关键子模块。`python3 -m venv venv` 创建隔离的虚拟环境，避免与系统 Python 包冲突。CMake 配置阶段通过 `-DUSE_CUDA=ON` 等选项控制后端支持，`-DUSE_LLVM` 指定 LLVM 路径（TileLang 利用 LLVM 进行编译优化）。`make -j$(nproc)` 使用所有 CPU 核心并行编译，`pip install -e .` 以开发模式安装。常见错误是子模块未拉取完整，导致 `3rdparty/tvm` 为空，编译时报缺少头文件。源码编译虽然比pip安装多花几分钟时间，但能够获得针对特定GPU架构的编译优化（如SM代码生成），对于追求极致性能的生产环境部署尤为推荐。建议在开发过程中使用`pip install -e .`的编辑模式安装，这样修改Python代码后无需重新安装即可生效。

### 2.3 方法三：Docker 安装

```bash
# 拉取官方 Docker 镜像
docker pull tilelang/tilelang:latest

# 或者使用特定版本
docker pull tilelang/tilelang:0.1.0-cuda12.4

# 运行容器
docker run -it --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    tilelang/tilelang:latest \
    bash

# 在容器中验证
python3 -c "import tilelang; print('TileLang installed successfully')"
```

Docker 安装提供了最一致的环境，消除了系统依赖差异导致的编译问题。官方仓库 `tilelang/tilelang` 提供预构建的镜像，`tilelang:latest` 指向最新稳定版，`tilelang:0.1.0-cuda12.4` 等标签精确绑定 CUDA 版本。`--gpus all` 将宿主机所有 GPU 透传给容器，是使用 GPU 的必需参数。`-v $(pwd):/workspace` 挂载当前目录到容器的工作区，方便在容器内编辑代码而在宿主机持久化文件。Docker 方式特别适合快速尝试验证，也是生产部署的推荐方案，但开发时文件 I/O 性能略低于原生安装。

```dockerfile
# 自定义 Dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    build-essential cmake git \
    llvm-17-dev libclang-17-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 克隆并安装 TileLang
RUN git clone --recursive https://github.com/tile-ai/tilelang.git /opt/tilelang
WORKDIR /opt/tilelang
RUN pip3 install -e .

# 设置工作目录
WORKDIR /workspace
CMD ["/bin/bash"]
```

Docker安装方式提供了完整的隔离环境，适合团队协作和持续集成流水线。`FROM nvidia/cuda:12.4.0-devel-ubuntu22.04`选用CUDA开发镜像作为基础，而非运行时（runtime）镜像，因为TileLang编译时需要CUDA头文件和编译器。`git clone --recursive`拉取TVM子模块，`pip install -e .`以开发模式安装。自定义Dockerfile的关键在于选择合适的CUDA基础镜像版本和LLVM版本，确保与主机驱动的兼容性。多阶段构建（multi-stage build）可以在第一阶段编译所有依赖，第二阶段生成精简的运行镜像，大幅减小最终产物体积。建议将常用Python包（PyTorch、NumPy等）也加入Dockerfile，避免每次启动容器后重复安装。Docker方式的另一个优势是可以在同一台机器上维护多个TileLang版本环境，互不干扰。

### 2.4 方法四：Conda 安装

```bash
# 创建 conda 环境
conda create -n tilelang python=3.10
conda activate tilelang

# 安装依赖
conda install -c conda-forge cmake llvmdev=17 clangdev=17
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# 安装 TileLang
pip install tilelang

# 验证
python3 -c "import tilelang; print('OK')"
```

Conda安装结合了包管理器和环境管理器两个优势，特别适合同时管理多个深度学习框架的用户。`conda create -n tilelang python=3.10`创建隔离环境，`-c conda-forge cmake llvmdev=17 clangdev=17`从conda-forge频道安装LLVM 17，确保版本一致性。PyTorch通过官方pytorch频道安装（含CUDA支持），TileLang自身通过pip安装。这种混合方式（conda管理环境和系统依赖、pip管理Python包）兼顾了conda在环境隔离方面的优势和pip在Python包版本控制方面的灵活性。需要注意的是，conda-forge的LLVM包可能更新滞后，若遇到版本问题可改用pip直接从官方渠道安装。conda环境的优势是可随时通过`conda env export > environment.yml`导出环境配置，方便团队共享和复现。

---

## 3. 版本兼容性矩阵

<div data-component="VersionCompatibilityMatrix"></div>

### 3.1 TileLang 版本兼容性

| TileLang | Python | CUDA | ROCm | PyTorch | TVM |
|:---|:---|:---|:---|:---|:---|
| 0.1.0 | 3.8-3.12 | 11.8-12.4 | 5.6-6.0 | 2.0-2.3 | 0.15+ |
| 0.1.1 | 3.8-3.12 | 11.8-12.4 | 5.7-6.1 | 2.1-2.4 | 0.16+ |
| 0.2.0 | 3.9-3.12 | 12.0-12.6 | 6.0-6.2 | 2.2-2.5 | 0.17+ |

### 3.2 GPU 架构支持

| GPU 架构 | 计算能力 | CUDA 支持 | TileLang 支持 |
|:---|:---|:---|:---|
| Ampere (A100) | 8.0 | ✅ | ✅ 完全支持 |
| Ada Lovelace (RTX 4090) | 8.9 | ✅ | ✅ 完全支持 |
| Hopper (H100) | 9.0 | ✅ | ✅ 完全支持 |
| Blackwell (B200) | 10.0 | ✅ | 🔧 实验支持 |
| CDNA 2 (MI250X) | - | ROCm | ✅ 完全支持 |
| CDNA 3 (MI300X) | - | ROCm | ✅ 完全支持 |
| Ascend 910B | - | CANN | ✅ 支持 |
| Ascend 910C | - | CANN | 🔧 开发中 |

### 3.3 常见兼容性问题

```python
# 问题 1: CUDA 版本不匹配
# 错误信息: RuntimeError: CUDA version mismatch
# 解决方案:
import torch
print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"System CUDA: ", end="")
import subprocess
result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
print(result.stdout.split("release")[-1].strip())

# 确保两者版本一致
```

这段代码用于诊断 CUDA 版本不匹配问题——TileLang 编译时的 CUDA 与 PyTorch 内嵌的 CUDA 版本必须一致。`torch.version.cuda` 返回 PyTorch 编译所用的 CUDA 版本，而 `nvcc --version` 的输出则反映系统安装的 CUDA Toolkit 版本。常见场景是：用 CUDA 12.4 编译了 PyTorch，但系统安装了 CUDA 11.8 作为默认版本，导致 TileLang 编译 kernel 时使用错误的 PTX ISA 目标。解决方案是确保 `CUDA_HOME` 指向与 PyTorch 匹配的 CUDA 版本，或重新安装对应版本的 PyTorch。

```python
# 问题 2: 计算能力不支持
# 错误信息: RuntimeError: CUDA architecture not supported
# 解决方案: 检查 GPU 计算能力
import torch
cap = torch.cuda.get_device_capability()
print(f"Compute Capability: {cap[0]}.{cap[1]}")
# TileLang 需要 >= 8.0 (Ampere)
```

GPU计算能力是TileLang能否运行的核心判据。Ampere架构（A100，计算能力8.0）是TileLang的最低要求，低于此版本的GPU（如T4的7.5、V100的7.0）无法支持Tensor Core和关键指令集，编译时直接报错。如果检测到计算能力不足，唯一的解决方案是更换GPU硬件。此脚本适合在CI流水线中作为前置检查步骤，避免在不兼容硬件上浪费编译时间。

```python
# 问题 3: LLVM 版本不兼容
# 错误信息: LLVM version mismatch
# 解决方案:
import subprocess
result = subprocess.run(["llvm-config-17", "--version"], capture_output=True, text=True)
print(f"LLVM Version: {result.stdout.strip()}")
# 需要 LLVM 15+
```

LLVM版本不兼容是TileLang编译失败的常见原因之一。TileLang依赖LLVM进行IR优化和代码生成，要求LLVM版本至少为15+。`llvm-config-17 --version`可快速确认系统中LLVM 17的安装情况。如果返回空值或版本号小于15，需要重新安装LLVM开发包。Ubuntu 22.04用户可通过`apt install llvm-17-dev libclang-17-dev`安装，CentOS用户需从LLVM官方仓库获取。安装后需确保CMake配置中`-DUSE_LLVM`指向正确的llvm-config路径，常见的错误是CMake自动检测到了旧版本LLVM导致编译失败。建议在构建脚本中明确指定LLVM版本，避免自动检测的不确定性。

---

## 4. 官方仓库结构走读

<div data-component="RepositoryStructureExplorer"></div>

### 4.1 仓库概览

```
tile-lang/tile-lang/
├── CMakeLists.txt              # 顶层 CMake 配置
├── LICENSE                     # Apache 2.0 许可证
├── README.md                   # 项目说明
├── setup.py                    # Python 包安装脚本
├── pyproject.toml              # Python 项目配置
├── requirements.txt            # Python 依赖
│
├── 3rdparty/                   # 第三方依赖
│   ├── tvm/                    # TVM 子模块 (核心依赖)
│   ├── cutlass/                # NVIDIA CUTLASS (可选)
│   └── googletest/             # Google Test (测试框架)
│
├── src/                        # C++ 源码
│   ├── tl/                     # TileLang 核心实现
│   │   ├── ir/                 # TileLang IR 定义
│   │   ├── transform/          # IR 变换 Pass
│   │   ├── codegen/            # 代码生成
│   │   │   ├── cuda/           # NVIDIA CUDA 后端
│   │   │   ├── rocm/           # AMD ROCm 后端
│   │   │   └── ascend/         # 华为昇腾后端
│   │   └── runtime/            # 运行时支持
│   └── tl_cpp/                 # C++ 工具库
│
├── tilelang/                   # Python 包
│   ├── __init__.py             # 包入口
│   ├── engine/                 # 编译引擎
│   │   ├── __init__.py
│   │   ├── lower.py            # IR Lowering
│   │   └── param.py            # 参数管理
│   ├── analysis/               # 分析工具
│   │   ├── __init__.py
│   │   └── layout_inference.py # Layout 推理
│   ├── transform/              # 变换 Pass
│   │   ├── __init__.py
│   │   ├── pipeline.py         # Software Pipelining
│   │   └── layout.py           # Layout 变换
│   ├── intrin/                 # 内建函数
│   │   ├── __init__.py
│   │   └── cuda_tensorcore.py  # CUDA Tensor Core
│   └── utils/                  # 工具函数
│       ├── __init__.py
│       └── tensor.py           # Tensor 工具
│
├── testing/                    # 测试代码
│   ├── test_gemm.py            # GEMM 测试
│   ├── test_attention.py       # Attention 测试
│   └── test_memory.py          # 内存管理测试
│
├── examples/                   # 示例代码
│   ├── basic/                  # 基础示例
│   │   ├── vector_add.py
│   │   ├── matmul.py
│   │   └── softmax.py
│   ├── advanced/               # 高级示例
│   │   ├── flash_attention.py
│   │   ├── flash_mla.py
│   │   └── dequant_gemm.py
│   └── benchmarks/             # 性能基准
│       ├── gemm_bench.py
│       └── attn_bench.py
│
├── docs/                       # 文档
│   ├── getting_started.md
│   ├── api_reference.md
│   └── tutorials/
│
└── ci/                         # CI/CD 配置
    ├── github-actions/
    └── docker/
```

这个代码块或示意图用于说明 4.1 仓库概览 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 4.2 核心目录详解

#### 4.2.1 `tilelang/` - Python 包

```python
# tilelang/__init__.py - 包入口
"""
TileLang: A tile-level programming language for GPU computing
"""

# 核心导入
from .engine import compile, lower
from . import T  # 编程接口模块

# 版本信息
__version__ = "0.1.0"

# 快捷导入
def prim_func(func):
    """装饰器: 将函数标记为 TileLang 原语函数"""
    return T.prim_func(func)
```

这段代码是 4.2.1 `tilelang/` - Python 包 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

```python
# tilelang/engine/lower.py - IR Lowering
"""
TileLang IR Lowering 模块
将 TileLang IR 转换为 TensorIR (TVM)
"""

def lower(func, target, ...):
    """
    将 TileLang 函数 Lowering 到目标代码

    Args:
        func: TileLang 函数
        target: 目标平台 ("cuda", "rocm", "ascend")
        ...: 其他参数

    Returns:
        编译后的模块
    """
    # 1. 解析 TileLang IR
    tl_ir = parse_tilelang_ir(func)

    # 2. 应用变换 Pass
    tl_ir = apply_transforms(tl_ir, ...)

    # 3. Lowering 到 TensorIR
    tensor_ir = lower_to_tensorir(tl_ir)

    # 4. 生成目标代码
    code = generate_target_code(tensor_ir, target)

    return code
```

这段代码是 4.2.1 `tilelang/` - Python 包 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

```python
# tilelang/analysis/layout_inference.py - Layout 推理
"""
Layout 推理模块
自动推导数据在内存中的排列方式
"""

class LayoutInference:
    """Layout 推理器"""

    def __init__(self, func):
        self.func = func

    def infer(self):
        """
        推理最优的 Layout

        Returns:
            Layout 描述
        """
        # 分析数据访问模式
        access_pattern = self.analyze_access_pattern()

        # 推理最优 Layout
        layout = self.compute_optimal_layout(access_pattern)

        # 验证 Layout 兼容性
        self.validate_layout(layout)

        return layout

    def analyze_access_pattern(self):
        """分析数据访问模式"""
        pass

    def compute_optimal_layout(self, pattern):
        """计算最优 Layout"""
        pass

    def validate_layout(self, layout):
        """验证 Layout 兼容性"""
        pass
```

这段代码是 4.2.1 `tilelang/` - Python 包 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

#### 4.2.2 `src/tl/` - C++ 核心

```cpp
// src/tl/ir/tile_ir.h - TileLang IR 定义
#pragma once

#include <tvm/ir.h>
#include <tvm/expr.h>

namespace tilelang {
namespace ir {

// Tile 级循环节点
class TileForNode : public tvm::StmtNode {
public:
    // 循环变量
    tvm::Var loop_var;
    // 循环范围
    tvm::Range range;
    // Tile 大小
    tvm::Expr tile_size;
    // 循环体
    tvm::Stmt body;

    // 访问器方法
    void VisitAttrs(tvm::AttrVisitor* v) {
        v->Visit("loop_var", &loop_var);
        v->Visit("range", &range);
        v->Visit("tile_size", &tile_size);
        v->Visit("body", &body);
    }

    static constexpr const char* _type_key = "tl.TileFor";
    TVM_DECLARE_FINAL_OBJECT_INFO(TileForNode, tvm::StmtNode);
};

// 内存分配节点
class AllocNode : public tvm::StmtNode {
public:
    // 内存类型 (Shared, L1, Fragment)
    enum MemoryType { Shared, L1, Fragment };
    MemoryType mem_type;
    // 缓冲区
    tvm::Buffer buffer;
    // 分配大小
    tvm::Expr size;

    static constexpr const char* _type_key = "tl.Alloc";
    TVM_DECLARE_FINAL_OBJECT_INFO(AllocNode, tvm::StmtNode);
};

}  // namespace ir
}  // namespace tilelang
```

这段代码是 4.2.2 `src/tl/` - C++ 核心 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

```cpp
// src/tl/codegen/codegen_cuda.h - CUDA 代码生成
#pragma once

#include <tvm/codegen.h>
#include <tl/ir/tile_ir.h>

namespace tilelang {
namespace codegen {

class CodeGenCUDA : public tvm::codegen::CodeGenLLVM {
public:
    // 生成 CUDA 内核代码
    std::string Generate(tvm::IRModule mod);

    // 生成 Shared Memory 声明
    void GenSharedMemoryDecl(const ir::AllocNode* op);

    // 生成 Tensor Core 指令
    void GenTensorCoreOp(const tvm::CallNode* op);

    // 生成 Warp 级同步
    void GenWarpSync();

protected:
    // 访问各类 IR 节点
    void VisitStmt_(const ir::TileForNode* op) override;
    void VisitStmt_(const ir::AllocNode* op) override;
    void VisitExpr_(const tvm::CallNode* op) override;
};

}  // namespace codegen
}  // namespace tilelang
```

这段代码是 4.2.2 `src/tl/` - C++ 核心 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

#### 4.2.3 `examples/` - 示例代码

```python
# examples/basic/vector_add.py - 向量加法示例
import tilelang
from tilelang import T
import torch

@T.prim_func
def vector_add(
    A: T.Buffer[(1024,), "float32"],
    B: T.Buffer[(1024,), "float32"],
    C: T.Buffer[(1024,), "float32"],
):
    for i in T.serial(1024):
        with T.block("add"):
            vi = T.axis.spatial(1024, i)
            C[vi] = A[vi] + B[vi]

if __name__ == "__main__":
    # 编译
    kernel = tilelang.compile(vector_add, target="cuda")

    # 运行
    A = torch.randn(1024, device="cuda")
    B = torch.randn(1024, device="cuda")
    C = torch.zeros(1024, device="cuda")

    kernel(A, B, C)

    # 验证
    torch.testing.assert_close(C, A + B)
    print("✓ 向量加法测试通过")
```

这段代码是 4.2.3 `examples/` - 示例代码 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

```python
# examples/basic/matmul.py - 矩阵乘法示例
import tilelang
from tilelang import T
import torch

@T.prim_func
def matmul(
    A: T.Buffer[(1024, 1024), "float16"],
    B: T.Buffer[(1024, 1024), "float16"],
    C: T.Buffer[(1024, 1024), "float32"],
):
    BLOCK = 128
    for bx, by in T.grid(1024 // BLOCK, 1024 // BLOCK):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])
            C_local = T.alloc_fragment((BLOCK, BLOCK), "float32")
            T.fill(C_local, 0.0)
            for k in T.serial(1024 // 32):
                A_shared = T.alloc_shared((BLOCK, 32), "float16")
                B_shared = T.alloc_shared((32, BLOCK), "float16")
                T.copy(A[vbx*BLOCK:(vbx+1)*BLOCK, k*32:(k+1)*32], A_shared)
                T.copy(B[k*32:(k+1)*32, vby*BLOCK:(vby+1)*BLOCK], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[vbx*BLOCK:(vbx+1)*BLOCK, vby*BLOCK:(vby+1)*BLOCK])

if __name__ == "__main__":
    kernel = tilelang.compile(matmul, target="cuda")

    A = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    B = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    C = torch.zeros(1024, 1024, dtype=torch.float32, device="cuda")

    kernel(A, B, C)

    ref = torch.matmul(A.float(), B.float())
    torch.testing.assert_close(C, ref, rtol=1e-2, atol=1e-2)
    print("✓ 矩阵乘法测试通过")
```

这段代码是 4.2.3 `examples/` - 示例代码 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 5. 构建系统详解

<div data-component="BuildSystemFlow"></div>

### 5.1 CMake 构建流程

TileLang 使用 CMake 作为构建系统，整个构建流程如下：

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   CMake      │    │   TVM        │    │   TileLang   │    │   Python     │
│   Configure  │───→│   Build      │───→│   C++ Build  │───→│   Package    │
│              │    │              │    │              │    │              │
│  检测环境    │    │  编译 TVM    │    │  编译核心    │    │  pip install │
│  设置选项    │    │  依赖        │    │  代码生成    │    │  -e .        │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

这个代码块或示意图用于说明 5.1 CMake 构建流程 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 5.2 CMakeLists.txt 解析

```cmake
# CMakeLists.txt - 顶层构建配置

cmake_minimum_required(VERSION 3.18)
project(TileLang CXX C)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(CUDA REQUIRED)
find_package(LLVM REQUIRED CONFIG)

# TVM 子模块
add_subdirectory(3rdparty/tvm)

# TileLang 核心库
add_library(tilelang_core SHARED
    src/tl/ir/tile_ir.cc
    src/tl/transform/pipeline.cc
    src/tl/transform/layout.cc
    src/tl/codegen/codegen_cuda.cc
    src/tl/codegen/codegen_rocm.cc
    src/tl/codegen/codegen_ascend.cc
)

# 链接依赖
target_link_libraries(tilelang_core
    PUBLIC tvm
    PUBLIC tvm_runtime
    PRIVATE LLVM
)

# CUDA 支持
if(USE_CUDA)
    target_compile_definitions(tilelang_core PRIVATE USE_CUDA=1)
    target_include_directories(tilelang_core PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

# ROCm 支持
if(USE_ROCM)
    target_compile_definitions(tilelang_core PRIVATE USE_ROCM=1)
endif()

# 安装
install(TARGETS tilelang_core DESTINATION lib)
```

这个代码块或示意图用于说明 5.2 CMakeLists.txt 解析 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 5.3 构建选项

| 选项 | 默认值 | 说明 |
|:---|:---|:---|
| `USE_CUDA` | ON | 启用 NVIDIA CUDA 支持 |
| `USE_ROCM` | OFF | 启用 AMD ROCm 支持 |
| `USE_ASCEND` | OFF | 启用华为昇腾支持 |
| `USE_LLVM` | 自动检测 | LLVM 路径 |
| `CMAKE_BUILD_TYPE` | Release | 构建类型 |
| `USE_TENSOR_CORE` | ON | 启用 Tensor Core 支持 |
| `TVM_HOME` | 3rdparty/tvm | TVM 路径 |

```bash
# 完整的构建命令示例
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    -DUSE_ROCM=OFF \
    -DUSE_ASCEND=OFF \
    -DUSE_LLVM=/usr/bin/llvm-config-17 \
    -DUSE_TENSOR_CORE=ON \
    -DTVM_HOME=$(pwd)/../3rdparty/tvm \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DPYTHON_EXECUTABLE=$(which python3)

# 编译 (使用所有 CPU 核心)
make -j$(nproc)

# 安装
sudo make install
```

这段命令对应 5.3 构建选项 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

---

## 6. 验证安装

### 6.1 基础验证脚本

```python
#!/usr/bin/env python3
"""
TileLang 安装验证脚本
运行: python verify_install.py
"""

import sys
import os

def check_import():
    """检查基本导入"""
    print("=" * 50)
    print("1. 检查基本导入")
    print("=" * 50)

    try:
        import tilelang
        print(f"  ✓ tilelang 导入成功")
        print(f"    版本: {tilelang.__version__}")
    except ImportError as e:
        print(f"  ✗ tilelang 导入失败: {e}")
        return False

    try:
        from tilelang import T
        print(f"  ✓ T 模块导入成功")
    except ImportError as e:
        print(f"  ✗ T 模块导入失败: {e}")
        return False

    return True

def check_cuda():
    """检查 CUDA 环境"""
    print("\n" + "=" * 50)
    print("2. 检查 CUDA 环境")
    print("=" * 50)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA 可用")
            print(f"    设备数: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
                cap = torch.cuda.get_device_capability(i)
                print(f"    计算能力: {cap[0]}.{cap[1]}")
        else:
            print(f"  ✗ CUDA 不可用")
            return False
    except ImportError:
        print(f"  ✗ PyTorch 未安装")
        return False

    return True

def check_compile():
    """检查编译功能"""
    print("\n" + "=" * 50)
    print("3. 检查编译功能")
    print("=" * 50)

    try:
        import tilelang
        from tilelang import T
        import torch

        # 定义一个简单的 kernel
        @T.prim_func
        def vector_add(
            A: T.Buffer[(64,), "float32"],
            B: T.Buffer[(64,), "float32"],
            C: T.Buffer[(64,), "float32"],
        ):
            for i in T.serial(64):
                with T.block("add"):
                    vi = T.axis.spatial(64, i)
                    C[vi] = A[vi] + B[vi]

        # 编译
        kernel = tilelang.compile(vector_add, target="cuda")
        print(f"  ✓ Kernel 编译成功")

        # 运行
        A = torch.randn(64, device="cuda")
        B = torch.randn(64, device="cuda")
        C = torch.zeros(64, device="cuda")

        kernel(A, B, C)

        # 验证
        torch.testing.assert_close(C, A + B)
        print(f"  ✓ Kernel 运行成功")
        print(f"  ✓ 结果验证通过")

    except Exception as e:
        print(f"  ✗ 编译/运行失败: {e}")
        return False

    return True

def check_performance():
    """检查性能"""
    print("\n" + "=" * 50)
    print("4. 检查性能基准")
    print("=" * 50)

    try:
        import tilelang
        from tilelang import T
        import torch
        import time

        # 简单的 GEMM kernel
        @T.prim_func
        def gemm_simple(
            A: T.Buffer[(256, 256), "float16"],
            B: T.Buffer[(256, 256), "float16"],
            C: T.Buffer[(256, 256), "float32"],
        ):
            for i, j in T.grid(256, 256):
                with T.block("gemm"):
                    vi, vj = T.axis.spatial("SS", [i, j])
                    C[vi, vj] = T.float32(0)
                    for k in T.serial(256):
                        C[vi, vj] += T.cast(A[vi, k], "float32") * T.cast(B[k, vj], "float32")

        kernel = tilelang.compile(gemm_simple, target="cuda")

        A = torch.randn(256, 256, dtype=torch.float16, device="cuda")
        B = torch.randn(256, 256, dtype=torch.float16, device="cuda")
        C = torch.zeros(256, 256, dtype=torch.float32, device="cuda")

        # Warmup
        for _ in range(10):
            kernel(A, B, C)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            kernel(A, B, C)
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_ms = (end - start) / 100 * 1000
        flops = 2 * 256 * 256 * 256
        tflops = flops / (avg_ms * 1e-3) / 1e12

        print(f"  ✓ GEMM 性能: {avg_ms:.3f} ms, {tflops:.2f} TFLOPS")

    except Exception as e:
        print(f"  ✗ 性能测试失败: {e}")
        return False

    return True

def main():
    """主函数"""
    print("\n" + "=" * 50)
    print("TileLang 安装验证")
    print("=" * 50 + "\n")

    results = []

    results.append(("基本导入", check_import()))
    results.append(("CUDA 环境", check_cuda()))
    results.append(("编译功能", check_compile()))
    results.append(("性能基准", check_performance()))

    # 总结
    print("\n" + "=" * 50)
    print("验证结果总结")
    print("=" * 50)

    all_pass = True
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
        if not result:
            all_pass = False

    print("\n" + "=" * 50)
    if all_pass:
        print("🎉 所有验证通过! TileLang 安装成功!")
    else:
        print("⚠️  部分验证失败，请检查环境配置")
    print("=" * 50)

    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
```

这段代码是 6.1 基础验证脚本 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 6.2 运行验证

```bash
# 保存上述脚本为 verify_install.py
python verify_install.py

# 预期输出:
# ==================================================
# TileLang 安装验证
# ==================================================
#
# ==================================================
# 1. 检查基本导入
# ==================================================
#   ✓ tilelang 导入成功
#     版本: 0.1.0
#   ✓ T 模块导入成功
#
# ==================================================
# 2. 检查 CUDA 篰境
# ==================================================
#   ✓ CUDA 可用
#     设备数: 1
#     GPU 0: NVIDIA A100-SXM4-80GB
#     计算能力: 8.0
#
# ==================================================
# 3. 检查编译功能
# ==================================================
#   ✓ Kernel 编译成功
#   ✓ Kernel 运行成功
#   ✓ 结果验证通过
#
# ==================================================
# 4. 检查性能基准
# ==================================================
#   ✓ GEMM 性能: 0.045 ms, 0.75 TFLOPS
#
# ==================================================
# 验证结果总结
# ==================================================
#   基本导入: ✓ 通过
#   CUDA 篰境: ✓ 通过
#   编译功能: ✓ 通过
#   性能基准: ✓ 通过
#
# ==================================================
# 🎉 所有验证通过! TileLang 安装成功!
# ==================================================
```

这段命令对应 6.2 运行验证 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

---

## 7. 常见问题与解决方案

### 7.1 安装问题

#### 问题 1: `ModuleNotFoundError: No module named 'tilelang'`

```bash
# 原因: TileLang 未正确安装
# 解决方案:
pip install tilelang

# 如果仍然失败，检查 Python 路径
python -c "import sys; print(sys.path)"

# 确保使用正确的 Python 环境
which python
which pip
```

这段命令对应 问题 1: `ModuleNotFoundError: No module named 'tilelang'` 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

#### 问题 2: `RuntimeError: CUDA not available`

```bash
# 原因: CUDA 环境未配置
# 解决方案:

# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA Toolkit
nvcc --version

# 检查 PyTorch CUDA 支持
python -c "import torch; print(torch.cuda.is_available())"

# 如果 PyTorch 不支持 CUDA，重新安装
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

这段命令对应 问题 2: `RuntimeError: CUDA not available` 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

#### 问题 3: `CompilationError: LLVM not found`

```bash
# 原因: LLVM 未安装或版本不匹配
# 解决方案:

# 安装 LLVM 17
sudo apt install -y llvm-17-dev libclang-17-dev

# 设置 LLVM 路径
export LLVM_CONFIG=/usr/bin/llvm-config-17

# 重新编译
cd tilelang && pip install -e .
```

这段命令对应 问题 3: `CompilationError: LLVM not found` 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

#### 问题 4: `RuntimeError: TVM runtime not found`

```bash
# 原因: TVM 子模块未初始化
# 解决方案:

cd tilelang
git submodule update --init --recursive

# 重新编译
mkdir -p build && cd build
cmake .. -DUSE_CUDA=ON
make -j$(nproc)
cd .. && pip install -e .
```

这段命令对应 问题 4: `RuntimeError: TVM runtime not found` 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

### 7.2 运行时问题

#### 问题 5: `Kernel launch failed`

```python
# 原因: Kernel 配置错误
# 解决方案: 检查 Thread Block 大小

# 错误示例
@T.prim_func
def bad_kernel(A, B, C):
    # Thread Block 超过硬件限制
    for i in T.serial(1024 * 1024):  # 太多线程
        ...

# 正确示例
@T.prim_func
def good_kernel(A, B, C):
    # 合理的 Thread Block 大小
    for i in T.serial(256):  # 256 线程
        ...
```

这段代码是 问题 5: `Kernel launch failed` 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

#### 问题 6: `Shared Memory 超出限制`

```python
# 原因: 分配的 Shared Memory 超过硬件限制
# 解决方案: 减少 Tile 大小

# 错误示例
@T.prim_func
def bad_kernel(A, B, C):
    # 每个 SM 最多 164 KB (A100)
    A_shared = T.alloc_shared((1024, 1024), "float16")  # 2 MB!

# 正确示例
@T.prim_func
def good_kernel(A, B, C):
    # 合理的 Shared Memory 大小
    A_shared = T.alloc_shared((128, 128), "float16")  # 32 KB
```

这段代码是 问题 6: `Shared Memory 超出限制` 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 8. 开发工具配置

### 8.1 VS Code 配置

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "/path/to/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "editor.formatOnSave": true,
    "files.associations": {
        "*.tl": "python"
    }
}
```

这段代码是 8.1 VS Code 配置 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

```json
// .vscode/launch.json - 调试配置
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "TileLang Debug",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {
                "TILELANG_DEBUG": "1",
                "CUDA_LAUNCH_BLOCKING": "1"
            }
        }
    ]
}
```

这段代码是 8.1 VS Code 配置 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 8.2 PyCharm 配置

```
1. 打开 PyCharm → Settings → Project → Python Interpreter
2. 选择虚拟环境中的 Python 解释器
3. 安装 tilelang 包
4. 配置 Run/Debug Configuration:
   - Environment Variables: TILELANG_DEBUG=1, CUDA_LAUNCH_BLOCKING=1
   - Working Directory: 项目根目录
```

这个代码块或示意图用于说明 8.2 PyCharm 配置 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 8.3 Jupyter Notebook 配置

```bash
# 安装 Jupyter
pip install jupyter

# 创建 TileLang 专用 kernel
python -m ipykernel install --user --name tilelang --display-name "TileLang"

# 启动 Jupyter
jupyter notebook
```

这段命令对应 8.3 Jupyter Notebook 配置 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

```python
# notebook 示例
import tilelang
from tilelang import T
import torch

# 检查环境
print(f"TileLang 版本: {tilelang.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

这段代码是 8.3 Jupyter Notebook 配置 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## ✅ 本章总结

### 核心要点

🎯 **安装方法**：
- pip 安装（推荐）：`pip install tilelang`
- 源码编译：克隆仓库 → CMake 构建 → pip install -e
- Docker：`docker pull tilelang/tilelang:latest`

🎯 **环境要求**：
- Python 3.8+, CUDA 11.8+, LLVM 15+, CMake 3.18+
- NVIDIA GPU (A100+) 或 AMD GPU (MI250X+) 或华为昇腾

🎯 **仓库结构**：
- `tilelang/`：Python 包（核心接口）
- `src/tl/`：C++ 核心（IR、代码生成）
- `examples/`：示例代码
- `testing/`：测试代码

🎯 **验证安装**：
- 运行 `verify_install.py` 脚本
- 检查导入、CUDA、编译、性能四个维度

### 关键命令

```bash
# 安装
pip install tilelang

# 验证
python -c "import tilelang; print(tilelang.__version__)"

# 源码安装
git clone --recursive https://github.com/tile-ai/tilelang.git
cd tilelang && pip install -e .

# Docker
docker pull tilelang/tilelang:latest
```

这段命令对应 关键命令 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

---

## 📝 练习题

### 练习 1：环境搭建

1. 在你的机器上安装 TileLang（选择任意一种方法）。
2. 运行验证脚本，确认所有检查项通过。
3. 记录你的环境配置（操作系统、Python 版本、CUDA 版本等）。

### 练习 2：源码探索

1. 克隆 TileLang 仓库：`git clone --recursive https://github.com/tile-ai/tilelang.git`
2. 阅读 `tilelang/__init__.py`，理解包的入口结构。
3. 找到 `T.gemm()` 函数的实现位置。
4. 阅读 `examples/basic/` 目录下的示例代码。

### 练习 3：构建系统

1. 尝试使用 CMake 手动构建 TileLang。
2. 修改 CMake 选项，观察构建结果的变化。
3. 理解 TVM 子模块在构建中的作用。

### 练习 4：问题排查

故意制造以下错误，观察错误信息并修复：

```python
# 错误 1: 导入不存在的模块
from tilelang import nonexistent

# 错误 2: 使用错误的数据类型
@T.prim_func
def bad_dtype(A: T.Buffer[(1024,), "int256"]):  # 不存在的类型
    pass

# 错误 3: 分配过大的 Shared Memory
@T.prim_func
def bad_alloc(A: T.Buffer[(1024, 1024), "float16"]):
    A_shared = T.alloc_shared((1024, 1024), "float16")  # 超出限制
```

这段代码是 练习 4：问题排查 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 🔗 扩展阅读

- [TileLang GitHub 仓库](https://github.com/tile-ai/tilelang)
- [TileLang 安装文档](https://tile-ai.github.io/tilelang/getting-started/installation)
- [CUDA Toolkit 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
- [CANN 安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/)
- [CMake 官方文档](https://cmake.org/cmake/help/latest/)
- [TVM 安装指南](https://tvm.apache.org/docs/install/from_source.html)

---

## 9. 多平台开发环境配置

### 9.1 多 GPU 环境配置

```bash
# 多 GPU 开发环境配置
# 当机器有多个 GPU 时，需要正确配置

# 查看所有 GPU
nvidia-smi -L

# 设置可见 GPU
export CUDA_VISIBLE_DEVICES=0,1  # 只使用 GPU 0 和 1

# 或者在 Python 中设置
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 验证
import torch
print(f"可用 GPU 数量: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

这段命令对应 9.1 多 GPU 环境配置 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

### 9.2 Docker 多阶段构建

```dockerfile
# 多阶段构建 Dockerfile
# 阶段 1: 构建环境
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    python3-dev python3-pip \
    llvm-17-dev libclang-17-dev

# 克隆 TileLang
RUN git clone --recursive https://github.com/tile-ai/tilelang.git /opt/tilelang

# 编译 TileLang
WORKDIR /opt/tilelang
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON && \
    make -j$(nproc)

# 安装 Python 包
RUN pip install -e .

# 阶段 2: 运行环境
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# 复制编译好的文件
COPY --from=builder /opt/tilelang /opt/tilelang
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10

# 设置环境变量
ENV PYTHONPATH=/opt/tilelang:$PYTHONPATH
ENV LD_LIBRARY_PATH=/opt/tilelang/build/lib:$LD_LIBRARY_PATH

WORKDIR /workspace
CMD ["/bin/bash"]
```

这个代码块或示意图用于说明 9.2 Docker 多阶段构建 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 9.3 Conda 环境配置

```yaml
# environment.yml - Conda 环境配置
name: tilelang
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - cmake>=3.24
  - llvmdev=17
  - clangdev=17
  - pytorch>=2.2
  - torchvision
  - torchaudio
  - pytorch-cuda=12.4
  - numpy
  - scipy
  - matplotlib
  - jupyter
  - pip:
    - tilelang
    - triton
    - cuda-python
```

这段代码是 9.3 Conda 环境配置 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

```bash
# 使用 Conda 环境
conda env create -f environment.yml
conda activate tilelang

# 验证
python -c "import tilelang; print('OK')"
```

这段命令对应 9.3 Conda 环境配置 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

### 9.4 VS Code Remote 开发

```json
// .vscode/settings.json - Remote 开发配置
{
    "python.defaultInterpreterPath": "/opt/conda/envs/tilelang/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.analysis.typeCheckingMode": "basic",
    "files.associations": {
        "*.tl": "python",
        "*.tilelang": "python"
    },
    "C_Cpp.default.includePath": [
        "/usr/local/cuda/include",
        "/opt/tilelang/3rdparty/tvm/include",
        "/opt/tilelang/src"
    ],
    "C_Cpp.default.defines": [
        "USE_CUDA=1",
        "USE_LLVM=1"
    ]
}
```

这段代码是 9.4 VS Code Remote 开发 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

```json
// .vscode/launch.json - 调试配置
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "TileLang Debug",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {
                "TILELANG_DEBUG": "1",
                "CUDA_LAUNCH_BLOCKING": "1",
                "TILELANG_DUMP_IR": "1"
            },
            "args": []
        },
        {
            "name": "TileLang Test",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["${file}", "-v"],
            "console": "integratedTerminal",
            "env": {
                "TILELANG_DEBUG": "1"
            }
        }
    ]
}
```

这段代码是 9.4 VS Code Remote 开发 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 9.5 CI/CD 配置

```yaml
# .github/workflows/ci.yml - GitHub Actions CI
name: TileLang CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]
        cuda-version: ["11.8", "12.1", "12.4"]

    steps:
      - uses: actions/checkout@v3
        with:
          submodules: recursive

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt

      - name: Build TileLang
        run: |
          mkdir build && cd build
          cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
          make -j$(nproc)
          cd .. && pip install -e .

      - name: Run tests
        run: |
          python -m pytest testing/ -v

      - name: Run examples
        run: |
          python examples/basic/vector_add.py
          python examples/basic/matmul.py
```

这段代码是 9.5 CI/CD 配置 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 10. 常用开发工具详解

### 10.1 ncu 性能分析

```bash
# ncu (Nsight Compute) 性能分析

# 基本分析
ncu python my_kernel.py

# 详细分析
ncu --set full python my_kernel.py

# 分析特定 kernel
ncu --kernel-name "gemm" python my_kernel.py

# 输出到文件
ncu --output report.ncu-rep python my_kernel.py

# 分析内存访问
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second python my_kernel.py

# 分析 Shared Memory
ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum python my_kernel.py

# 分析 Bank Conflict
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum python my_kernel.py
```

这段命令对应 10.1 ncu 性能分析 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

### 10.2 nsys 系统级分析

```bash
# nsys (Nsight Systems) 系统级分析

# 基本分析
nsys profile python my_kernel.py

# 输出到文件
nsys profile -o report python my_kernel.py

# 分析 CUDA API 调用
nsys profile --trace=cuda python my_kernel.py

# 分析内存操作
nsys profile --trace=cuda,nvtx python my_kernel.py

# 查看报告
nsys-ui report.nsys-rep
```

这段命令对应 10.2 nsys 系统级分析 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

### 10.3 cuda-memcheck 内存检查

```bash
# cuda-memcheck 内存检查

# 检查内存错误
cuda-memcheck python my_kernel.py

# 检查内存泄漏
cuda-memcheck --leak-check full python my_kernel.py

# 检查竞争条件
cuda-memcheck --racecheck python my_kernel.py
```

这段命令对应 10.3 cuda-memcheck 内存检查 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

### 10.4 TileLang 内置调试工具

```python
# TileLang 内置调试工具

import tilelang
from tilelang import T

# 1. IR Dump
# 设置环境变量 TILELANG_DUMP_IR=1
# 编译器会输出每个阶段的 IR
import os
os.environ["TILELANG_DUMP_IR"] = "1"

# 2. 性能分析
# 设置环境变量 TILELANG_PROFILE=1
os.environ["TILELANG_PROFILE"] = "1"

# 3. 内存检查
# 设置环境变量 TILELANG_CHECK_MEMORY=1
os.environ["TILELANG_CHECK_MEMORY"] = "1"

# 4. 调试模式
# 设置环境变量 TILELANG_DEBUG=1
os.environ["TILELANG_DEBUG"] = "1"
```

这段代码是 10.4 TileLang 内置调试工具 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 11. 环境故障排除手册

### 11.1 安装问题排查

```bash
# 问题: pip install tilelang 失败
# 排查步骤:

# 1. 检查 Python 版本
python3 --version
# 需要 3.8+

# 2. 检查 pip 版本
pip --version
# 建议最新版本
pip install --upgrade pip

# 3. 检查网络连接
ping pypi.org

# 4. 使用国内镜像
pip install tilelang -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 从源码安装
git clone --recursive https://github.com/tile-ai/tilelang.git
cd tilelang
pip install -e .
```

这段命令对应 11.1 安装问题排查 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

```bash
# 问题: 编译失败
# 排查步骤:

# 1. 检查 CMake 版本
cmake --version
# 需要 3.18+

# 2. 检查 GCC 版本
gcc --version
# 需要 9.0+

# 3. 检查 CUDA 版本
nvcc --version
# 需要 11.8+

# 4. 检查 LLVM 版本
llvm-config-17 --version
# 需要 15+

# 5. 检查 TVM 子模块
git submodule status
# 应该显示 3rdparty/tvm
```

这段命令对应 11.1 安装问题排查 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

### 11.2 运行时问题排查

```python
# 问题: Kernel 运行失败
# 排查步骤:

import tilelang
from tilelang import T
import torch

# 1. 检查 CUDA 是否可用
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# 2. 检查 GPU 状态
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory: {torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB")

# 3. 检查 TileLang 版本
import tilelang
print(f"TileLang version: {tilelang.__version__}")

# 4. 简单测试
@T.prim_func
def test_kernel(A: T.Buffer[(64,), "float32"], B: T.Buffer[(64,), "float32"]):
    for i in T.serial(64):
        with T.block("test"):
            vi = T.axis.spatial(64, i)
            B[vi] = A[vi] * T.float32(2)

try:
    kernel = tilelang.compile(test_kernel, target="cuda")
    A = torch.randn(64, device="cuda")
    B = torch.zeros(64, device="cuda")
    kernel(A, B)
    print("✓ Kernel test passed")
except Exception as e:
    print(f"✗ Kernel test failed: {e}")
```

这段代码是 11.2 运行时问题排查 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 11.3 性能问题排查

```python
# 问题: 性能不如预期
# 排查步骤:

# 1. 检查 GPU 利用率
# 使用 nvidia-smi 监控
# nvidia-smi -l 1

# 2. 检查内存带宽
# 使用 ncu 分析
# ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second python my_kernel.py

# 3. 检查计算利用率
# ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed python my_kernel.py

# 4. 检查 Bank Conflict
# ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum python my_kernel.py

# 5. 检查 Occupancy
# ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_elapsed python my_kernel.py

# 6. 常见性能问题:
# - Tile 大小不合适
# - Shared Memory 使用过多
# - Bank Conflict
# - 内存访问不合并
# - 计算/访存比低
```

这段代码是 11.3 性能问题排查 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 11.4 兼容性问题排查

```bash
# 问题: 版本不兼容
# 排查步骤:

# 1. 检查所有版本
python3 -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')

import tilelang
print(f'TileLang: {tilelang.__version__}')
"

# 2. 检查 CUDA 兼容性
nvcc --version
nvidia-smi

# 3. 检查 PyTorch CUDA 兼容性
python3 -c "
import torch
print(f'PyTorch CUDA: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')
"

# 4. 版本兼容性矩阵
# TileLang 0.1.x: Python 3.8-3.12, CUDA 11.8-12.4, PyTorch 2.0-2.3
# TileLang 0.2.x: Python 3.9-3.12, CUDA 12.0-12.6, PyTorch 2.2-2.5
```

这段命令对应 11.4 兼容性问题排查 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

---

## 12. 开发环境最佳实践

### 12.1 虚拟环境管理

```bash
# 最佳实践: 使用虚拟环境隔离开发环境

# 方案 1: venv
python3 -m venv ~/tilelang-env
source ~/tilelang-env/bin/activate
pip install tilelang

# 方案 2: conda
conda create -n tilelang python=3.10
conda activate tilelang
pip install tilelang

# 方案 3: Docker
docker run -it --gpus all -v $(pwd):/workspace tilelang/tilelang:latest bash

# 推荐: Docker 用于生产，venv/conda 用于开发
```

这段命令对应 12.1 虚拟环境管理 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

### 12.2 版本控制

```bash
# 最佳实践: 版本控制

# 1. 使用 git 管理代码
git init
git add .
git commit -m "Initial commit"

# 2. 使用 requirements.txt 管理依赖
pip freeze > requirements.txt

# 3. 使用 .gitignore 排除构建文件
echo "build/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.egg-info/" >> .gitignore
```

这段命令对应 12.2 版本控制 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

### 12.3 测试策略

```python
# 最佳实践: 测试策略

import pytest
import tilelang
from tilelang import T
import torch

# 1. 单元测试
def test_vector_add():
    """测试向量加法"""
    @T.prim_func
    def vector_add(A: T.Buffer[(64,), "float32"], B: T.Buffer[(64,), "float32"], C: T.Buffer[(64,), "float32"]):
        for i in T.serial(64):
            with T.block("add"):
                vi = T.axis.spatial(64, i)
                C[vi] = A[vi] + B[vi]

    kernel = tilelang.compile(vector_add, target="cuda")
    A = torch.randn(64, device="cuda")
    B = torch.randn(64, device="cuda")
    C = torch.zeros(64, device="cuda")

    kernel(A, B, C)
    torch.testing.assert_close(C, A + B)

# 2. 性能测试
def test_gemm_performance():
    """测试 GEMM 性能"""
    @T.prim_func
    def gemm(A: T.Buffer[(1024, 1024), "float16"], B: T.Buffer[(1024, 1024), "float16"], C: T.Buffer[(1024, 1024), "float32"]):
        # ... GEMM 实现 ...
        pass

    kernel = tilelang.compile(gemm, target="cuda")
    A = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    B = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    C = torch.zeros(1024, 1024, dtype=torch.float32, device="cuda")

    # 性能测试
    import time
    start = time.perf_counter()
    for _ in range(100):
        kernel(A, B, C)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) / 100 * 1000
    assert avg_ms < 10.0  # 应该在 10ms 以内

# 3. 正确性测试
def test_gemm_correctness():
    """测试 GEMM 正确性"""
    @T.prim_func
    def gemm(A: T.Buffer[(256, 256), "float16"], B: T.Buffer[(256, 256), "float16"], C: T.Buffer[(256, 256), "float32"]):
        # ... GEMM 实现 ...
        pass

    kernel = tilelang.compile(gemm, target="cuda")
    A = torch.randn(256, 256, dtype=torch.float16, device="cuda")
    B = torch.randn(256, 256, dtype=torch.float16, device="cuda")
    C = torch.zeros(256, 256, dtype=torch.float32, device="cuda")

    kernel(A, B, C)
    ref = torch.matmul(A.float(), B.float())
    torch.testing.assert_close(C, ref, rtol=1e-2, atol=1e-2)
```

这段代码是 12.3 测试策略 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 12.4 代码组织

```
# 最佳实践: 代码组织

project/
├── src/
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── gemm.py          # GEMM kernel
│   │   ├── attention.py      # Attention kernel
│   │   └── normalize.py      # Normalization kernel
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── benchmark.py      # 性能测试工具
│   │   └── validation.py     # 正确性验证
│   └── main.py
├── tests/
│   ├── test_gemm.py
│   ├── test_attention.py
│   └── test_normalize.py
├── benchmarks/
│   ├── gemm_benchmark.py
│   └── attention_benchmark.py
├── docs/
│   └── ...
├── requirements.txt
├── setup.py
└── README.md
```

这个代码块或示意图用于说明 12.4 代码组织 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

---

## 📖 下一章预告

**Chapter 2: 三级编程接口——Beginner/Developer/Expert**

在下一章中，我们将：
- 学习 TileLang 的三级编程接口设计
- 使用 Beginner 级接口编写第一个 kernel（零调度知识）
- 使用 Developer 级接口进行显式内存管理
- 使用 Expert 级接口控制 Thread Binding
- 对比三级接口的性能差异
