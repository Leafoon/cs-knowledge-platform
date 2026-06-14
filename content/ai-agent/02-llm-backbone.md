---
title: "第2章：大模型基座 — Agent 的大脑"
description: "深入理解 LLM 作为 Agent 推理引擎的核心能力，掌握 Transformer 架构、上下文窗口、Token 机制、温度与采样策略对 Agent 行为的影响，主流模型对比与选型，API 调用最佳实践。"
date: "2026-06-11"
---

# 第2章：大模型基座 — Agent 的大脑

大语言模型是 Agent 的推理核心。本章从 Transformer 架构原理出发，深入理解 LLM 的工作机制，掌握模型参数调优、主流模型选型以及 API 调用的最佳实践。

---

## 2.1 LLM 作为 Agent 推理引擎

### 2.1.1 为什么 LLM 适合作为 Agent 大脑

LLM 具备几个关键能力使其成为理想的 Agent 推理引擎：

| 能力 | 说明 | 对 Agent 的意义 | 涌现规模 |
|:---|:---|:---|:---|
| **通用推理** | 基于自然语言的逻辑推理 | 理解任务、分解计划、分析结果 | ~10B |
| **上下文学习** | 从少量示例中学习新任务 | Few-shot 工具使用指导 | ~10B |
| **指令遵循** | 按照提示词中的指令执行任务 | 遵循 Agent 行为规范 | ~100B |
| **工具理解** | 理解函数签名、参数含义 | 正确选择和调用工具 | ~100B |
| **代码能力** | 理解、生成和调试代码 | 代码 Agent、数据分析 Agent | ~100B |
| **多语言** | 支持多种人类语言 | 全球化 Agent 应用 | ~10B |
| **结构化输出** | 生成符合 JSON Schema 的数据 | Function Calling、数据提取 | ~10B |

**LLM vs 传统规则引擎**：

| 维度 | 传统规则引擎 | LLM Agent |
|:---|:---|:---|
| 规则定义 | 人工编写每条规则 | LLM 自主推理 |
| 新场景 | 需要人工添加新规则 | 自动适应 |
| 规则冲突 | 难以调试 | 自然语言推理避免 |
| 可解释性 | 规则可追溯 | 推理过程可解释 |

> **关键洞察**：小模型（<10B 参数）虽然也能做简单的工具调用，但在复杂场景下（多步推理、错误恢复、模糊意图理解）表现显著差于大模型（>100B）。**Agent 场景对模型规模有硬性要求**。

### 2.1.2 涌现能力与 Agent 能力的关系

当模型规模超过某个阈值时，会出现训练阶段未明确优化的**涌现能力（Emergent Abilities）**：

$$
\text{Performance}(s) = \begin{cases} \text{random} & s < s_{\text{threshold}} \\ f(s) & s \geq s_{\text{threshold}} \end{cases}
$$

| 涌现能力 | 出现规模 | Agent 应用 |
|:---|:---|:---|
| **Chain-of-Thought** | ~100B 参数 | 多步推理规划 |
| **In-Context Learning** | ~10B 参数 | Few-shot 工具学习 |
| **Instruction Following** | ~100B 参数 | 遵循 Agent 指令 |
| **Code Generation** | ~100B 参数 | 代码 Agent |
| **Tool Use** | ~100B 参数 | Function Calling |
| **Self-Consistency** | ~100B 参数 | 多路径推理 |

---

## 2.2 Transformer 架构速览

### 2.2.1 Self-Attention 机制

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """Self-Attention 的完整 PyTorch 实现"""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(output)
```

**多头注意力的直觉理解**：

| Head | 关注方面 | 示例 |
|:---|:---|:---|
| Head 1 | 语法关系 | "cat" 关注 "the" |
| Head 2 | 语义关系 | "cat" 关注 "pet" |
| Head 3 | 位置关系 | "cat" 关注前面的词 |
| Head 4 | 长程依赖 | 句首关注句尾 |

### 2.2.2 模型架构演进

| 模型 | 年份 | 架构特点 | 上下文窗口 | Agent 影响 |
|:---|:---|:---|:---|:---|
| GPT-3 | 2020 | Decoder-only, 175B | 4K | 基础推理能力 |
| GPT-4 | 2023 | MoE 架构 | 8K/32K | Function Calling |
| GPT-4o | 2024 | 多模态统一 | 128K | 多模态 Agent |
| Claude 3.5 | 2024 | 长上下文优化 | 200K | 超长文档处理 |
| Claude 4 | 2025 | 深度推理增强 | 200K | 复杂规划 Agent |
| Gemini 2.5 | 2025 | 原生多模态 | 1M | 超长上下文 Agent |
| Llama 4 | 2025 | MoE 开源 | 128K | 本地部署 Agent |

---

## 2.3 Token 机制与上下文窗口

### 2.3.1 Token 化算法

```python
import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

# 英文
text = "Hello, how are you?"
tokens = encoder.encode(text)
print(f"Token 数: {len(tokens)}")
print(f"逆向解码: {[encoder.decode([t]) for t in tokens]}")
# ['Hello', ',', ' how', ' are', ' you', '?']

# 中文（每个汉字约 1-1.5 token）
text_cn = "你好，世界！AI Agent 是未来。"
tokens_cn = encoder.encode(text_cn)
print(f"中文 Token 数: {len(tokens_cn)}, 字符数: {len(text_cn)}")
# Token/字符比约 1.07
```

**不同语言的 Token 效率**：

| 语言 | Token/字符比 | 说明 |
|:---|:---|:---|
| 英文 | ~0.25 | 每个单词约 1 token |
| 中文 | ~1.0-1.5 | 每个汉字约 1-1.5 token |
| 代码 | ~0.3-0.5 | 变量名、关键字 |
| 数字 | ~0.1-0.2 | 每个数字约 0.1 token |

> **Agent 开发启示**：
> 1. 中文输入的 Token 成本约为英文的 3-4 倍
> 2. System Prompt 使用英文可以节省 50%+ 的 Token 成本
> 3. 工具描述建议使用英文

### 2.3.2 Token 计数与成本计算

```python
def calculate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o") -> dict:
    """计算 API 调用成本（美元）"""
    pricing = {
        "gpt-4o":           {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4o-mini":      {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "claude-sonnet-4-20250514": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
        "deepseek-r1":      {"input": 0.55 / 1_000_000, "output": 2.19 / 1_000_000},
    }
    price = pricing.get(model, pricing["gpt-4o"])
    input_cost = input_tokens * price["input"]
    output_cost = output_tokens * price["output"]
    return {"input_cost": input_cost, "output_cost": output_cost, "total_cost": input_cost + output_cost}

# Agent 成本分析
scenarios = [
    {"name": "简单问答", "input": 1500, "output": 500},
    {"name": "工具调用 (3步)", "input": 4000, "output": 800},
    {"name": "复杂推理 (5步)", "input": 8000, "output": 2000},
]
for s in scenarios:
    cost = calculate_cost(s["input"], s["output"])
    print(f"{s['name']}: 单次 ${cost['total_cost']:.4f}, 月(10K次) ${cost['total_cost']*10000*30:.2f}")
```

### 2.3.3 上下文窗口对 Agent 能力的约束

$$
\text{可用上下文} = \text{窗口大小} - \text{系统提示} - \text{工具定义} - \text{输出预留}
$$

```python
class ContextWindowManager:
    def __init__(self, max_tokens: int = 128000, reserved: int = 4096):
        self.max_tokens = max_tokens
        self.reserved = reserved
        self.available = max_tokens - reserved

    def allocate(self, system_prompt: int, tools: int) -> dict:
        remaining = self.available - system_prompt - tools
        return {
            "chat_history": int(remaining * 0.6),
            "rag_context": int(remaining * 0.3),
            "scratchpad": int(remaining * 0.1),
            "total_available": remaining,
        }
```

---

## 2.4 模型参数与 Agent 行为

### 2.4.1 Temperature 的影响

$$
P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

| Temperature | 行为特征 | Agent 场景 |
|:---|:---|:---|
| $T = 0$ | 确定性输出 | 工具调用、代码生成 |
| $T = 0.3$ | 低随机性 | 事实性问答 |
| $T = 0.7$ | 中等随机性 | 通用对话 |
| $T = 1.0$ | 高随机性 | 创意写作 |

> **Agent 最佳实践**：工具调用场景始终使用 `temperature=0`，确保确定性输出。

### 2.4.2 其他关键参数

| 参数 | 推荐值 | 说明 |
|:---|:---|:---|
| `max_tokens` | 4096 | 足够输出但不过长 |
| `top_p` | 1.0 | 与 temperature=0 配合 |
| `frequency_penalty` | 0.0 | Agent 不需要多样化 |
| `presence_penalty` | 0.0 | Agent 不需要鼓励新话题 |
| `seed` | 调试时设置 | 确保可复现 |

---

## 2.5 主流模型对比与选型

### 2.5.1 模型能力对比表

| 模型 | 推理 | 工具调用 | 代码 | 中文 | 上下文 | 输入价格 | 输出价格 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| GPT-4o | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★ | 128K | $2.5/M | $10/M |
| Claude Sonnet 4 | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | 200K | $3/M | $15/M |
| Gemini 2.5 Pro | ★★★★★ | ★★★★ | ★★★★★ | ★★★★ | 1M | $1.25/M | $10/M |
| DeepSeek R1 | ★★★★★ | ★★★ | ★★★★★ | ★★★★★ | 128K | $0.55/M | $2.19/M |
| GPT-4o-mini | ★★★★ | ★★★★ | ★★★★ | ★★★ | 128K | $0.15/M | $0.6/M |

### 2.5.2 Agent 场景选型指南

| 场景 | 推荐模型 | 理由 |
|:---|:---|:---|
| 通用 Agent | GPT-4o / Claude Sonnet 4 | 综合能力最强 |
| 代码 Agent | Claude Sonnet 4 / DeepSeek R1 | 代码突出 |
| 长文档 Agent | Gemini 2.5 Pro / GPT-4.1 | 1M 上下文 |
| 低成本 Agent | GPT-4o-mini / DeepSeek V3 | 性价比最高 |
| 本地部署 | Llama 4 / Qwen 3 | 开源免费 |

---

## 2.6 API 调用最佳实践

### 2.6.1 带重试的 OpenAI API

```python
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

client = OpenAI()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def call_openai_with_retry(messages, tools=None, model="gpt-4o"):
    kwargs = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 4096, "timeout": 30}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message
```

### 2.6.2 LangChain 统一接口

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

gpt4 = ChatOpenAI(model="gpt-4o", temperature=0)
claude = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

messages = [SystemMessage(content="你是一个有帮助的助手。"), HumanMessage(content="你好")]
response_gpt = gpt4.invoke(messages)
response_claude = claude.invoke(messages)
```

---

## 2.7 本地模型部署

### 2.7.1 Ollama 快速部署

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b
ollama serve
```

```python
from langchain_ollama import ChatOllama
local_llm = ChatOllama(model="llama3.1:8b", temperature=0)
```

### 2.7.2 量化技术对比

| 技术 | 精度 | 模型大小(7B) | 速度损失 | 质量损失 |
|:---|:---|:---|:---|:---|
| FP16 | 16-bit | ~14GB | 基准 | 无 |
| GPTQ | 4-bit | ~4GB | ~10% | 很小 |
| AWQ | 4-bit | ~4GB | ~5% | 很小 |
| GGUF Q4 | 4-bit | ~4GB | ~15% | 小 |

---

## 2.8 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| LLM 角色 | Agent 的推理核心 |
| Transformer | Self-Attention + 多头注意力 + 因果 mask |
| Token 机制 | BPE 分词，中文约 1-1.5 token/字 |
| Temperature | Agent 工具调用必须 T=0 |
| 模型选型 | 综合考虑能力、成本、延迟、上下文 |
| 本地部署 | Ollama 最简单，vLLM 性能最好 |

> **下一章预告**
>
> 在第 3 章中，我们将深入 Agent 的指令系统——提示词工程。
