---
title: "Chapter 0. 环境准备与预备知识"
description: "PyTorch 简介、环境配置最佳实践与必要的数学基础回顾"
updated: "2026-01-22"
---

# Chapter 0. 环境准备与预备知识

> **Learning Objectives**
> *   了解 PyTorch 在现代 AI 栈中的地位 (vs TensorFlow/JAX)。
> *   学会使用 Conda 管理虚拟环境，避免 "Dependency Hell"。
> *   掌握验证 GPU 环境 (`CUDA`) 的标准流程。
> *   复习这一门课所需的数学基础 (Matrix, Chain Rule)。

---

## 0.1 为什么选择 PyTorch?

2024 年的深度学习领域，PyTorch 是绝对的霸主。
*   **研究界**: 90% 以上的顶级顶会论文 (NeurIPS/ICLR/CVPR) 使用 PyTorch。
*   **工业界**: 随着 PyTorch 2.0 (`compile`), TorchServe, CPP Extension 的完善，部署能力大幅提升。
*   **生态**: HuggingFace Transformers, Diffusers, Detectron2 等核心库均以 PyTorch 为主。

**Pythonic 哲学**:
PyTorch 采用 "Eager Execution" (动态图)，代码逻辑与原生 Python 几乎一致。你可以随时 `print(tensor)`，随时 `if x > 0:`，这使得调试体验极佳。

---

## 0.2 环境配置最佳实践

> [!WARNING]
> **永远不要** 在系统 Python (System Python) 中直接 pip install。这会导致系统库冲突，甚至为了修复它需要重装系统。

### 0.2.1 使用 Conda 管理环境

我们推荐使用 Miniconda (或 Anaconda)。

```bash
# 1. 创建名为 cs2-torch 的环境，Python 版本 3.10
conda create -n cs2-torch python=3.10

# 2. 激活环境
conda activate cs2-torch
```

### 0.2.2 安装 PyTorch

访问 [pytorch.org](https://pytorch.org/) 获取最新命令。
**关键点**：确保 CUDA 版本与你显卡驱动支持的版本匹配。运行 `nvidia-smi` 查看 Driver Version。

```bash
# Example: CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 0.3 验证安装：Hello GPU

创建一个 Python 脚本 `check_env.py`，运行以下代码。这是你每次配置新机器时必须做的第一件事。

```python
import torch
import platform

print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print("-" * 30)

if torch.cuda.is_available():
    print("✅ GPU is available!")
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    # 简单的 CUDA 运算测试
    x = torch.tensor([1.0]).cuda()
    print(f"Tensor on GPU: {x}")
else:
    print("⚠️ GPU NOT available. Using CPU.")
```

> [!TIP]
> **Mac M1/M2/M3 用户**:
> PyTorch 支持 MPS (Metal Performance Shaders) 加速。
> 检查 `torch.backends.mps.is_available()` 是否为 True。

---

## 0.4 数学基础回顾

不需要精通所有数学，但以下概念必须烂熟于心：

### 0.4.1 矩阵乘法 (Dot Product)

形状匹配规则：$(M \times K) \cdot (K \times N) \rightarrow (M \times N)$。
如果不匹配，程序会直接 Crash。

### 0.4.2 链式法则 (Chain Rule)

神经网络反向传播 (Backpropagation) 的灵魂。
如果 $y = f(u)$ 且 $u = g(x)$，那么：
$$ \frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} $$

PyTorch 的 `Autograd` 引擎就是在这个公式上构建的自动化系统。我们在 Chapter 2 会深入讲解。

---

## 0.5 常见问题 (Troubleshooting)

<details>
<summary><strong>Q: pip install 下载太慢怎么办？</strong></summary>

国内用户请使用清华源或阿里源：
<code>pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple</code>
</details>

<details>
<summary><strong>Q: 报错 RuntimeError: CUDA out of memory</strong></summary>

显存炸了。尝试减小 Batch Size，或者检查是否有未释放的僵尸进程 (用 nvidia-smi 此时显存占用)。
</details>
