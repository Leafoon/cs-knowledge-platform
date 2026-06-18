---
title: "第1章：Agent 核心循环 — 感知-决策-执行"
description: "深入理解 Agent 的感知-决策-执行（PDA）循环，掌握状态空间、动作空间与环境建模，理解 Agent 与环境交互的完整数据流，从零构建完整 Agent Step 函数。"
date: "2026-06-15"
---

 # 第1章：Agent 核心循环 — 感知-决策-执行

 Agent 的核心是一个**循环**——感知环境、做出决策、执行行动、观察结果，然后再次感知。这个看似简单的循环，却是所有智能体系统的基础骨架。无论是自动驾驶系统、游戏 AI、还是 ChatGPT 插件调用，都遵循这一基本范式。本章将深入剖析这个循环的每一个环节，从数学形式化到工程实现，从理论框架到代码细节，构建你对 Agent 核心机制的完整认知。

 ---

 ## 1.1 Agent 的感知-决策-执行循环

 ### 1.1.1 PDA 循环的完整定义

 下面的交互式演示展示了感知-决策-执行循环的完整过程：

 <div data-component="PerceptionDecisionActionDemo"></div>

### 1.1.2 PDA 循环的数据流详解

理解 PDA 循环的关键在于理解数据如何在各个阶段之间流动。让我们用一个具体的例子来追踪完整的数据流。

**场景**：用户问"北京今天天气怎么样？请帮我搜索并总结。"

```
┌──────────────────────────────────────────────────────────────────┐
│                    Agent PDA 循环 — 完整数据流                     │
│                                                                   │
│  ┌─────────────┐                                                  │
│  │  用户输入    │ "北京今天天气怎么样？请帮我搜索并总结。"              │
│  └──────┬──────┘                                                  │
│         │                                                         │
│         ▼                                                         │
│  ┌──────────────────────────────────────────────────┐             │
│  │              Perception（感知层）                   │             │
│  │                                                    │             │
│  │  Step 1: 输入解析                                  │             │
│  │    • 识别用户意图：天气查询 + 搜索 + 总结            │             │
│  │    • 提取实体：地点 = "北京"，时间 = "今天"          │             │
│  │    • 检测语言：中文                                 │             │
│  │                                                    │             │
│  │  Step 2: 上下文组装                                │             │
│  │    • System Prompt: "你是一个天气助手..."            │             │
│  │    • 历史消息: [无]                                 │             │
│  │    • 当前输入: "北京今天天气怎么样？..."             │             │
│  │    • 可用工具描述: [search_weather, search_web]     │             │
│  │                                                    │             │
│  │  Step 3: Token 预算管理                            │             │
│  │    • 总 Token 计算: ~150 tokens                     │             │
│  │    • 预算剩余: 8000 - 150 = 7850                    │             │
│  │    • 无需裁剪                                       │             │
│  │                                                    │             │
│  │  Output: Context_t (消息列表)                       │             │
│  └──────────────────────┬───────────────────────────┘             │
│                         │                                         │
│                         ▼                                         │
│  ┌──────────────────────────────────────────────────┐             │
│  │              Decision（决策层）                     │             │
│  │                                                    │             │
│  │  Step 1: LLM 推理                                  │             │
│  │    • 模型: GPT-4o                                  │             │
│  │    • 输入: Context_t                               │             │
│  │    • 推理过程:                                     │             │
│  │      "用户想知道北京天气，我需要先搜索天气信息"       │             │
│  │                                                    │             │
│  │  Step 2: 工具选择                                  │             │
│  │    • 候选工具: search_weather, search_web           │             │
│  │    • 选择: search_weather（直接天气查询更高效）       │             │
│  │                                                    │             │
│  │  Step 3: 参数生成                                  │             │
│  │    • city: "北京"                                   │             │
│  │    • date: "today"                                  │             │
│  │                                                    │             │
│  │  Output: Action_t = {tool: search_weather,          │             │
│  │          args: {city: "北京", date: "today"}}       │             │
│  └──────────────────────┬───────────────────────────┘             │
│                         │                                         │
│                         ▼                                         │
│  ┌──────────────────────────────────────────────────┐             │
│  │              Action（执行层）                       │             │
│  │                                                    │             │
│  │  Step 1: 工具调用                                  │             │
│  │    • 调用 search_weather(city="北京", date="today") │             │
│  │    • 发起 HTTP 请求到天气 API                       │             │
│  │    • 等待响应...                                    │             │
│  │                                                    │             │
│  │  Step 2: 结果接收                                  │             │
│  │    • 响应时间: 230ms                                │             │
│  │    • HTTP 状态码: 200                               │             │
│  │    • 响应体: {temp: 28, condition: "晴", ...}       │             │
│  │                                                    │             │
│  │  Output: o_{t+1} = 执行结果                         │             │
│  └──────────────────────┬───────────────────────────┘             │
│                         │                                         │
│                         ▼                                         │
│  ┌──────────────────────────────────────────────────┐             │
│  │              Observation（观察层）                  │             │
│  │                                                    │             │
│  │  Step 1: 结果解析                                  │             │
│  │    • JSON 解析成功                                  │             │
│  │    • 数据验证通过                                   │             │
│  │    • 提取关键信息: 28°C, 晴, 湿度 45%              │             │
│  │                                                    │             │
│  │  Step 2: 格式化为 LLM 消息                         │             │
│  │    • "[search_weather 执行成功]                     │             │
│  │       温度: 28°C, 天气: 晴, 湿度: 45%"             │             │
│  │                                                    │             │
│  │  Output: ToolMessage(content=..., tool_call_id=...) │             │
│  └──────────────────────┬───────────────────────────┘             │
│                         │                                         │
│                         ▼                                         │
│              ┌───────────────────┐                                │
│              │   任务完成？       │                                │
│              │  用户要搜索+总结   │                                │
│              │  目前只完成了搜索  │──No──► 回到 Perception          │
│              └─────────┬─────────┘                                │
│                        │ Yes                                      │
│                        ▼                                          │
│              ┌───────────────────┐                                │
│              │    最终输出        │                                │
│              │  "北京今天28°C,   │                                │
│              │   晴天, 湿度45%"  │                                │
│              └───────────────────┘                                │
└──────────────────────────────────────────────────────────────────┘
```

### 1.1.3 与强化学习的 Agent-Environment 循环对比

PDA 循环与强化学习中的 Agent-Environment 循环有着深刻的联系，但也存在本质区别。理解这些区别对于正确设计 Agent 系统至关重要。

| 维度 | RL Agent-Environment 循环 | LLM Agent PDA 循环 |
|:---|:---|:---|
| **决策函数** | 策略网络 $\pi_\theta(a \mid s)$ | LLM 推理 $P(a \mid \text{context})$ |
| **状态表示** | 固定维度的向量/图像张量 | 自然语言文本（变长序列） |
| **动作空间** | 离散动作集或连续动作空间 | 自然语言 + 结构化工具调用 |
| **环境转移** | 概率转移 $P(s' \mid s, a)$ | 确定性工具返回（可有副作用） |
| **学习方式** | 在线学习（梯度更新参数） | 上下文学习（In-Context Learning） |
| **奖励信号** | 标量奖励函数 $r(s, a)$ | 任务完成度 / 用户反馈 |
| **探索策略** | $\epsilon$-greedy, UCB, Boltzmann | 多次采样 + Self-Consistency |
| **记忆机制** | Replay Buffer / MCTS | 消息历史 + 滑动窗口 |
| **终止条件** | Episode 结束（到达终态或超时） | 任务完成 / 最大迭代 / 错误累积 |
| **状态转移确定性** | 通常随机（$P(s'|s,a)$ 是概率分布） | 通常确定性（同一工具调用返回相同结果） |
| **动作效果可逆性** | 环境不可逆（episode 一旦发生） | 工具调用可能有副作用（不可逆） |

<div data-component="AgentCapabilityMatrixV5"></div>

**关键区别深入分析**

1. **状态表示的差异**

在传统 RL 中，状态 $s \in \mathbb{R}^d$ 是一个固定维度的向量。例如，Atari 游戏的状态是 84×84×4 的图像张量，机器人控制的状态是 17 维的关节角度向量。而在 LLM Agent 中，状态是**变长的自然语言文本**。这意味着：

- 状态空间是**无限的**——任何合法的文本序列都可以是状态
- 状态转移是**确定性的**——给定相同的上下文和工具定义，LLM 的输出分布是固定的（虽然采样有随机性）
- 状态可以包含**任意复杂的语义信息**——但受限于上下文窗口长度

2. **学习机制的本质差异**

RL Agent 通过**梯度下降**更新策略参数 $\theta$：

$$
\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t)
$$

而 LLM Agent 通过**上下文学习（In-Context Learning）**来适应新任务。LLM 的参数 $\theta$ 在推理时是**冻结的**，Agent 的"学习"完全发生在上下文窗口中：

$$
a_t = \text{LLM}_\theta(\text{SystemPrompt} \oplus \text{History}_{1:t-1} \oplus \text{Observation}_t)
$$

其中 $\oplus$ 表示序列拼接。这意味着 LLM Agent 的"记忆"受限于上下文窗口大小 $L$，而 RL Agent 的"记忆"可以存储在无限大的 Replay Buffer 中。

3. **奖励信号的差异**

RL 中的奖励是**标量** $r \in \mathbb{R}$，精确但信息量有限。LLM Agent 的"奖励"更为丰富——它包括任务是否完成的**语义判断**、用户是否满意的**主观评价**、以及工具返回结果的**结构化信息**。这种丰富的反馈信号使得 LLM Agent 可以进行更复杂的决策，但也使得形式化分析更加困难。

### 1.1.4 状态空间、动作空间与观察空间的定义

理解 PDA 循环需要精确区分三个关键空间：

**状态空间 $\mathcal{S}$**

状态空间 $\mathcal{S}$ 是 Agent 所有可能内部状态的集合。在 LLM Agent 中，一个状态 $s \in \mathcal{S}$ 包含：

$$
s = (\mathcal{M}, \mathcal{P}, \mathcal{R}, \mathcal{E}, n)
$$

其中：
- $\mathcal{M}$：消息历史（Message History），包含所有对话轮次
- $\mathcal{P}$：执行计划（Plan），当前的任务分解方案
- $\mathcal{R}$：中间结果（Results），已完成步骤的输出
- $\mathcal{E}$：错误记录（Errors），执行过程中的异常信息
- $n$：当前迭代次数

**动作空间 $\mathcal{A}$**

动作空间 $\mathcal{A}$ 是 Agent 可以采取的所有行动的集合。在 LLM Agent 中，动作空间是**动态的**——它取决于当前可用的工具集。一个动作 $a \in \mathcal{A}$ 可以是：

$$
a = \begin{cases}
(\text{tool\_call}, \text{tool\_name}, \text{args}) & \text{工具调用} \\
(\text{final\_answer}, \text{content}) & \text{直接回答} \\
(\text{think}, \text{reasoning}) & \text{内部推理（Chain-of-Thought）}
\end{cases}
$$

**观察空间 $\mathcal{O}$**

观察空间 $\mathcal{O}$ 是环境可能返回的所有观察的集合。观察 $o \in \mathcal{O}$ 包括：

$$
o = \begin{cases}
\text{user\_input} & \text{用户的新输入} \\
\text{tool\_result} & \text{工具执行结果} \\
\text{system\_event} & \text{系统事件（超时、错误等）} \\
\text{environment\_change} & \text{环境变化（文件修改等）}
\end{cases}
$$

**三者之间的关系**

PDA 循环中三个空间的交互可以形式化为：

$$
\begin{aligned}
\text{感知}: &\quad \mathcal{S} \times \mathcal{O} \rightarrow \mathcal{C} \quad \text{（状态与观察映射到上下文）} \\
\text{决策}: &\quad \mathcal{C} \rightarrow \mathcal{A} \quad \text{（上下文映射到动作）} \\
\text{执行}: &\quad \mathcal{A} \times \mathcal{E} \rightarrow \mathcal{O} \quad \text{（动作与环境交互产生观察）} \\
\text{更新}: &\quad \mathcal{S} \times \mathcal{A} \times \mathcal{O} \rightarrow \mathcal{S} \quad \text{（新状态由旧状态、动作和观察决定）}
\end{aligned}
$$

其中 $\mathcal{E}$ 表示环境（External Environment），$\mathcal{C}$ 表示上下文（Context）。

### 1.1.5 PDA 循环的数学形式化

我们可以将 PDA 循环形式化为一个**非平稳马尔可夫决策过程（Non-stationary MDP）**。

**定义 1.1（LLM Agent MDP）**

一个 LLM Agent 的行为可以形式化为元组 $\langle \mathcal{S}, \mathcal{A}, \mathcal{O}, T, \text{Perception}, \text{Decision}, \text{Action} \rangle$，其中：

- $\mathcal{S}$：状态空间（消息历史、计划、结果的笛卡尔积）
- $\mathcal{A}$：动作空间（工具调用集合，动态变化）
- $\mathcal{O}$：观察空间（工具返回值、用户输入）
- $T: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{S})$：状态转移函数（通常确定性）
- $\text{Perception}: \mathcal{S} \times \mathcal{O} \rightarrow \mathcal{C}$：感知函数
- $\text{Decision}: \mathcal{C} \rightarrow \Delta(\mathcal{A})$：决策函数（LLM 采样）
- $\text{Action}: \mathcal{A} \rightarrow \mathcal{O}$：执行函数

**定理 1.1（PDA 循环的不动点）**

如果 Agent 的决策函数 $\text{Decision}$ 是确定性的（temperature=0），且环境转移 $T$ 也是确定性的，则 PDA 循环在有限步内必然收敛到一个不动点（终止状态），前提是：

$$
\exists t^* < \infty \quad \text{使得} \quad \text{Decision}(\text{Perception}(s_{t^*}, o_{t^*})) = (\text{final\_answer}, \cdot)
$$

即 LLM 最终会决定给出最终回答。

在实践中，我们通过设置最大迭代次数 $n_{\max}$ 来保证循环一定终止：

$$
t \leq n_{\max} \implies \text{循环终止}
$$

---

## 1.2 感知（Perception）

感知是 PDA 循环的起点，负责将外部世界的原始输入转化为 LLM 能够理解和处理的结构化上下文。感知层的质量直接决定了 Agent 的决策质量——如果输入给 LLM 的信息不准确、不完整或过于冗余，即使是最强大的模型也无法做出正确的决策。

### 1.2.1 用户输入解析的详细机制

用户输入是 Agent 感知的起点。解析用户输入需要处理多种复杂情况：

**输入类型分类**

| 输入类型 | 描述 | 解析策略 | 示例 |
|:---|:---|:---|:---|
| **简单查询** | 单一问题，无需工具 | 直接传递给 LLM | "什么是机器学习？" |
| **复合任务** | 包含多个子任务 | 分解为子目标 | "搜索天气并总结" |
| **隐含意图** | 需要推理用户真正需求 | 意图识别 + 消歧 | "帮我看看这个" |
| **多模态输入** | 包含图像/音频/视频 | 多模态预处理 | "这张图片里有什么？" |
| **上下文依赖** | 依赖对话历史 | 指代消解 + 上下文补全 | "再试试另一个" |
| **纠正性输入** | 用户纠正之前的回答 | 识别并应用修正 | "不是这个，换一个" |

**输入解析的完整流程**

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import re


class InputType(Enum):
    """输入类型枚举"""
    SIMPLE_QUERY = "simple_query"          # 简单查询
    COMPOUND_TASK = "compound_task"        # 复合任务
    IMPLICIT_INTENT = "implicit_intent"    # 隐含意图
    MULTIMODAL = "multimodal"              # 多模态
    CONTEXT_DEPENDENT = "context_dependent" # 上下文依赖
    CORRECTION = "correction"              # 纠正性输入


@dataclass
class ParsedInput:
    """解析后的输入结构"""
    raw_text: str                           # 原始文本
    input_type: InputType                   # 输入类型
    intent: str                             # 识别出的意图
    entities: dict[str, str] = field(default_factory=dict)  # 提取的实体
    sub_tasks: list[str] = field(default_factory=list)      # 子任务列表
    references: list[str] = field(default_factory=list)     # 对历史的引用
    language: str = "zh"                    # 语言
    priority: int = 0                       # 优先级
    metadata: dict = field(default_factory=dict)            # 额外元数据


def parse_user_input(
    raw_input: str,
    conversation_history: list[dict],
    system_capabilities: list[str]
) -> ParsedInput:
    """解析用户输入的完整流程"""
    
    # Step 1: 基础文本清洗
    cleaned_text = raw_input.strip()
    if not cleaned_text:
        return ParsedInput(
            raw_text=raw_input,
            input_type=InputType.SIMPLE_QUERY,
            intent="empty_input",
            entities={"error": "输入为空"}
        )
    
    # Step 2: 检测输入类型
    input_type = _classify_input_type(cleaned_text, conversation_history)
    
    # Step 3: 语言检测
    language = _detect_language(cleaned_text)
    
    # Step 4: 意图识别
    intent = _recognize_intent(cleaned_text, system_capabilities)
    
    # Step 5: 实体提取
    entities = _extract_entities(cleaned_text)
    
    # Step 6: 子任务分解（复合任务时）
    sub_tasks = []
    if input_type == InputType.COMPOUND_TASK:
        sub_tasks = _decompose_task(cleaned_text)
    
    # Step 7: 指代消解（上下文依赖时）
    references = []
    if input_type == InputType.CONTEXT_DEPENDENT:
        references = _resolve_references(cleaned_text, conversation_history)
    
    return ParsedInput(
        raw_text=cleaned_text,
        input_type=input_type,
        intent=intent,
        entities=entities,
        sub_tasks=sub_tasks,
        references=references,
        language=language
    )


def _classify_input_type(
    text: str,
    history: list[dict]
) -> InputType:
    """基于规则的输入类型分类"""
    
    # 检测纠正性输入
    correction_patterns = [
        r"不是", r"错了", r"重新", r"换一个", r"不对",
        r"纠正", r"修改", r"不要.*要"
    ]
    for pattern in correction_patterns:
        if re.search(pattern, text):
            return InputType.CORRECTION
    
    # 检测复合任务（包含连接词）
    compound_markers = ["并且", "然后", "接着", "同时", "还有", "以及"]
    if any(marker in text for marker in compound_markers):
        return InputType.COMPOUND_TASK
    
    # 检测上下文依赖（包含指代词）
    context_markers = ["这个", "那个", "它", "上面", "之前", "再"]
    if any(marker in text for marker in context_markers) and history:
        return InputType.CONTEXT_DEPENDENT
    
    # 检测隐含意图（过于简短或模糊）
    if len(text) < 5 and not text.endswith("?") and not text.endswith("？"):
        return InputType.IMPLICIT_INTENT
    
    return InputType.SIMPLE_QUERY


def _detect_language(text: str) -> str:
    """简单语言检测"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0:
        return "unknown"
    if chinese_chars / total_chars > 0.3:
        return "zh"
    return "en"


def _recognize_intent(text: str, capabilities: list[str]) -> str:
    """意图识别"""
    intent_keywords = {
        "search": ["搜索", "查找", "查询", "search", "find", "look up"],
        "calculate": ["计算", "算", "求", "calculate", "compute"],
        "create": ["创建", "生成", "写", "create", "generate", "write"],
        "analyze": ["分析", "解析", "研究", "analyze", "examine"],
        "explain": ["解释", "说明", "介绍", "explain", "describe"],
    }
    
    for intent, keywords in intent_keywords.items():
        if any(kw in text.lower() for kw in keywords):
            return intent
    
    return "general"


def _extract_entities(text: str) -> dict[str, str]:
    """实体提取"""
    entities = {}
    
    # 时间实体
    time_patterns = {
        "today": r"今天|今日|today",
        "tomorrow": r"明天|明日|tomorrow",
        "yesterday": r"昨天|昨日|yesterday",
    }
    for time_type, pattern in time_patterns.items():
        if re.search(pattern, text):
            entities["time"] = time_type
            break
    
    # 地点实体（简单正则）
    location_match = re.search(r'[\u4e00-\u9fff]{2,4}(?:市|区|县|省)', text)
    if location_match:
        entities["location"] = location_match.group()
    
    return entities


def _decompose_task(text: str) -> list[str]:
    """复合任务分解"""
    # 按连接词分割
    separators = ["并且", "然后", "接着", "同时", "以及"]
    tasks = [text]
    for sep in separators:
        new_tasks = []
        for task in tasks:
            parts = task.split(sep)
            new_tasks.extend([p.strip() for p in parts if p.strip()])
        tasks = new_tasks
    
    return tasks


def _resolve_references(
    text: str,
    history: list[dict]
) -> list[str]:
    """指代消解"""
    references = []
    reference_words = ["这个", "那个", "它", "上面", "之前"]
    
    for word in reference_words:
        if word in text:
            # 在历史中查找最近的相关内容
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    references.append(msg.get("content", "")[:100])
                    break
    
    return references
```

### 1.2.2 多模态感知处理

现代 LLM Agent 需要处理多种模态的输入。多模态感知的核心挑战是如何将不同模态的信息统一转化为 LLM 可以处理的文本表示。

**多模态输入类型**

| 模态 | 输入格式 | 处理方式 | Token 开销 |
|:---|:---|:---|:---|
| **文本** | UTF-8 字符串 | 直接使用 | ~1 token/字符 |
| **图像** | JPEG/PNG/WebP | Base64 编码或 URL 引用 | 85-170 tokens/图 |
| **音频** | WAV/MP3 | ASR 转录为文本 | 转录文本的 token 数 |
| **视频** | MP4/WebM | 关键帧提取 + 描述 | 每帧 85-170 tokens |
| **文件** | PDF/DOC/XLSX | 内容提取 + 格式化 | 视内容长度 |

**多模态感知的完整实现**

```python
import base64
import mimetypes
from pathlib import Path
from typing import Union
from dataclasses import dataclass


@dataclass
class MultimodalContent:
    """多模态内容的统一表示"""
    text_parts: list[str]           # 文本片段
    image_parts: list[dict]         # 图像（base64 或 URL）
    audio_parts: list[str]          # 音频转录文本
    total_tokens_estimate: int      # 估算的总 Token 数
    
    def to_langchain_message(self):
        """转换为 LangChain 消息格式"""
        content = []
        
        # 添加文本部分
        if self.text_parts:
            combined_text = "\n".join(self.text_parts)
            content.append({"type": "text", "text": combined_text})
        
        # 添加图像部分
        for img in self.image_parts:
            content.append({
                "type": "image_url",
                "image_url": img
            })
        
        return content


def process_multimodal_input(
    text: str = None,
    image_paths: list[str] = None,
    audio_path: str = None,
    file_paths: list[str] = None,
    max_image_tokens: int = 1000
) -> MultimodalContent:
    """处理多模态输入的统一接口"""
    
    text_parts = []
    image_parts = []
    audio_parts = []
    total_tokens = 0
    
    # 处理文本
    if text:
        text_parts.append(text)
        total_tokens += estimate_text_tokens(text)
    
    # 处理图像
    if image_paths:
        for img_path in image_paths:
            if not Path(img_path).exists():
                text_parts.append(f"[图像文件不存在: {img_path}]")
                continue
            
            # 检查 Token 预算
            if total_tokens + max_image_tokens > 8000:
                text_parts.append(f"[图像因 Token 预算不足被跳过: {img_path}]")
                continue
            
            # 读取并编码图像
            img_data = encode_image(img_path)
            mime_type = mimetypes.guess_type(img_path)[0] or "image/jpeg"
            
            image_parts.append({
                "url": f"data:{mime_type};base64,{img_data}"
            })
            total_tokens += max_image_tokens  # 估算图像 Token
    
    # 处理音频（通过 ASR 转录）
    if audio_path:
        if Path(audio_path).exists():
            transcription = transcribe_audio(audio_path)
            audio_parts.append(transcription)
            text_parts.append(f"[音频转录]\n{transcription}")
            total_tokens += estimate_text_tokens(transcription)
        else:
            text_parts.append(f"[音频文件不存在: {audio_path}]")
    
    # 处理文件
    if file_paths:
        for fpath in file_paths:
            if Path(fpath).exists():
                file_content = extract_file_content(fpath)
                text_parts.append(f"[文件: {Path(fpath).name}]\n{file_content}")
                total_tokens += estimate_text_tokens(file_content)
    
    return MultimodalContent(
        text_parts=text_parts,
        image_parts=image_parts,
        audio_parts=audio_parts,
        total_tokens_estimate=total_tokens
    )


def encode_image(image_path: str) -> str:
    """将图像编码为 Base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def transcribe_audio(audio_path: str) -> str:
    """音频转录（简化版，实际应调用 ASR API）"""
    # 实际实现中，这里会调用 Whisper API 或其他 ASR 服务
    return f"[音频转录占位符: {audio_path}]"


def extract_file_content(file_path: str) -> str:
    """提取文件内容"""
    suffix = Path(file_path).suffix.lower()
    
    if suffix == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif suffix == ".pdf":
        return f"[PDF 文件: {file_path}，需使用 PDF 解析库]"
    else:
        return f"[不支持的文件格式: {suffix}]"


def estimate_text_tokens(text: str) -> int:
    """估算文本的 Token 数"""
    # 粗略估算：中文约 1.5 token/字，英文约 0.75 token/词
    chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english_words = len(text.split()) - chinese_chars
    return int(chinese_chars * 1.5 + english_words * 0.75)
```

### 1.2.3 上下文窗口管理策略

上下文窗口（Context Window）是 LLM Agent 最宝贵的资源。所有信息——系统提示、对话历史、工具描述、当前输入——都需要塞进有限的 Token 预算中。如何在有限空间内最大化信息价值，是感知层最核心的工程挑战。

**上下文窗口的组成**

一个典型的 Agent 上下文窗口由以下部分组成：

```
┌─────────────────────────────────────────┐
│           上下文窗口 (L tokens)           │
│                                          │
│  ┌────────────────────────────────┐      │
│  │  System Prompt (S tokens)      │      │
│  │  • 角色定义                     │      │
│  │  • 行为规范                     │      │
│  │  • 输出格式要求                  │      │
│  └────────────────────────────────┘      │
│                                          │
│  ┌────────────────────────────────┐      │
│  │  Tool Descriptions (T tokens)  │      │
│  │  • 工具名称和描述               │      │
│  │  • 参数 schema                  │      │
│  │  • 使用示例                     │      │
│  └────────────────────────────────┘      │
│                                          │
│  ┌────────────────────────────────┐      │
│  │  Conversation History (H)      │      │
│  │  • 早期消息（可能被摘要）        │      │
│  │  • 近期消息（完整保留）          │      │
│  └────────────────────────────────┘      │
│                                          │
│  ┌────────────────────────────────┐      │
│  │  Current Input (I tokens)      │      │
│  │  • 用户当前输入                 │      │
│  │  • 工具返回结果                 │      │
│  │  • 中间推理（CoT）              │      │
│  └────────────────────────────────┘      │
│                                          │
│  ┌────────────────────────────────┐      │
│  │  Output Reserve (R tokens)     │      │
│  │  • 预留给 LLM 输出              │      │
│  └────────────────────────────────┘      │
│                                          │
│  约束: S + T + H + I + R ≤ L            │
└─────────────────────────────────────────┘
```

**Token 预算分配策略**

不同的分配策略适用于不同的场景：

| 策略 | System Prompt | 工具描述 | 历史消息 | 当前输入 | 输出预留 |
|:---|:---|:---|:---|:---|:---|
| **平衡型** | 10% | 15% | 40% | 20% | 15% |
| **工具密集型** | 5% | 25% | 30% | 25% | 15% |
| **长对话型** | 8% | 10% | 55% | 12% | 15% |
| **代码执行型** | 15% | 20% | 25% | 25% | 15% |

**上下文窗口管理的详细实现**

```python
import tiktoken
from dataclasses import dataclass, field
from typing import Optional
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage,
    AIMessage, ToolMessage
)


@dataclass
class TokenBudget:
    """Token 预算配置"""
    total: int = 128000                 # 总 Token 预算
    system_prompt: int = 2000           # 系统提示词配额
    tool_descriptions: int = 3000       # 工具描述配额
    conversation_history: int = 60000   # 对话历史配额
    current_input: int = 10000          # 当前输入配额
    output_reserve: int = 4096          # 输出预留
    safety_margin: int = 1000           # 安全余量


@dataclass
class ManagedContext:
    """管理后的上下文"""
    messages: list[BaseMessage]         # 最终消息列表
    token_usage: dict[str, int]         # 各部分 Token 使用量
    compression_applied: bool           # 是否应用了压缩
    summary: Optional[str] = None       # 摘要（如果被压缩了）


class ContextWindowManager:
    """上下文窗口管理器"""
    
    def __init__(self, model: str = "gpt-4o", budget: TokenBudget = None):
        # 初始化 Token 计数器
        self.encoding = tiktoken.encoding_for_model(model)
        self.budget = budget or TokenBudget()
    
    def count_tokens(self, text: str) -> int:
        """计算文本的 Token 数"""
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def count_message_tokens(self, message: BaseMessage) -> int:
        """计算单条消息的 Token 数"""
        content = message.content or ""
        if isinstance(content, list):
            # 多模态消息
            total = 0
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += self.count_tokens(part["text"])
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    total += 100  # 图像估算
            return total + 4  # 消息头开销
        return self.count_tokens(content) + 4  # +4 为消息头开销
    
    def manage_context(
        self,
        system_prompt: str,
        tool_descriptions: list[str],
        history: list[BaseMessage],
        current_input: str,
        preserve_recent: int = 8
    ) -> ManagedContext:
        """核心方法：管理上下文窗口"""
        
        messages = []
        usage = {}
        
        # 1. System Prompt（不可压缩）
        messages.append(SystemMessage(content=system_prompt))
        usage["system_prompt"] = self.count_tokens(system_prompt)
        
        # 2. Tool Descriptions（不可压缩）
        tool_text = "\n\n".join(tool_descriptions)
        messages.append(SystemMessage(
            content=f"[可用工具]\n{tool_text}"
        ))
        usage["tool_descriptions"] = self.count_tokens(tool_text)
        
        # 3. 计算剩余预算
        remaining = (
            self.budget.total
            - usage["system_prompt"]
            - usage["tool_descriptions"]
            - self.budget.output_reserve
            - self.budget.safety_margin
        )
        
        history_budget = min(remaining * 0.7, self.budget.conversation_history)
        input_budget = min(remaining * 0.3, self.budget.current_input)
        
        # 4. 处理历史消息
        compressed_history, history_tokens = self._manage_history(
            history, int(history_budget), preserve_recent
        )
        messages.extend(compressed_history)
        usage["conversation_history"] = history_tokens
        
        # 5. 处理当前输入
        input_tokens = self.count_tokens(current_input)
        if input_tokens > input_budget:
            current_input = current_input[:input_budget * 2]  # 粗略截断
            input_tokens = self.count_tokens(current_input)
        messages.append(HumanMessage(content=current_input))
        usage["current_input"] = input_tokens
        
        # 6. 计算总用量
        usage["total"] = sum(usage.values())
        usage["remaining"] = self.budget.total - usage["total"]
        
        compression_applied = usage["conversation_history"] < sum(
            self.count_message_tokens(m) for m in history
        )
        
        return ManagedContext(
            messages=messages,
            token_usage=usage,
            compression_applied=compression_applied
        )
    
    def _manage_history(
        self,
        history: list[BaseMessage],
        budget: int,
        preserve_recent: int
    ) -> tuple[list[BaseMessage], int]:
        """管理历史消息：摘要 + 滑动窗口"""
        
        if not history:
            return [], 0
        
        # 计算所有消息的 Token 数
        token_counts = [self.count_message_tokens(m) for m in history]
        total_tokens = sum(token_counts)
        
        # 如果总 Token 在预算内，直接返回
        if total_tokens <= budget:
            return history, total_tokens
        
        # 策略：保留最近 N 条消息，其余摘要
        recent_count = min(preserve_recent, len(history))
        recent_messages = history[-recent_count:]
        old_messages = history[:-recent_count]
        
        recent_tokens = sum(token_counts[-recent_count:])
        old_tokens = total_tokens - recent_tokens
        
        if old_tokens <= 0:
            return recent_messages, recent_tokens
        
        # 对旧消息生成摘要
        summary = self._summarize_messages(old_messages)
        summary_tokens = self.count_tokens(summary)
        
        result = [SystemMessage(content=f"[对话历史摘要]\n{summary}")]
        result.extend(recent_messages)
        
        return result, summary_tokens + recent_tokens
    
    def _summarize_messages(self, messages: list[BaseMessage]) -> str:
        """消息摘要生成"""
        summary_parts = []
        
        for msg in messages:
            role = msg.type
            content = msg.content or ""
            
            if isinstance(content, list):
                # 多模态消息，只取文本部分
                text_parts = [
                    p["text"] for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = " ".join(text_parts)
            
            # 截断过长的内容
            if len(content) > 200:
                content = content[:200] + "..."
            
            summary_parts.append(f"[{role}] {content}")
        
        return "\n".join(summary_parts)
```

### 1.2.4 输入验证与安全检查

Agent 必须对用户输入进行安全检查，防止注入攻击和恶意输入。

**安全威胁分类**

| 威胁类型 | 描述 | 防御策略 |
|:---|:---|:---|
| **Prompt Injection** | 用户输入中嵌入系统指令 | 输入过滤 + 角色隔离 |
| **Jailbreak** | 试图绕过安全限制 | 意图检测 + 行为监控 |
| **Data Exfiltration** | 尝试获取敏感信息 | 输出过滤 + 权限控制 |
| **Resource Exhaustion** | 超长输入耗尽 Token | 输入长度限制 |
| **Tool Abuse** | 滥用工具进行危险操作 | 工具权限分级 |

```python
import re
from dataclasses import dataclass


@dataclass
class SecurityCheckResult:
    """安全检查结果"""
    is_safe: bool
    risk_level: str       # low, medium, high, critical
    threats: list[str]
    sanitized_input: str


class InputSanitizer:
    """输入安全检查器"""
    
    # 危险模式列表
    DANGEROUS_PATTERNS = [
        (r"忽略.*(?:之前的|上面的|所有).*指令", "prompt_injection"),
        (r"你是一个.*(?:没有|不受).*限制", "jailbreak"),
        (r"(?:系统|system)\s*(?:提示|prompt|message)", "prompt_injection"),
        (r"(?:sudo|rm\s+-rf|DROP\s+TABLE)", "command_injection"),
        (r"(?:api[_-]?key|secret|password|token)\s*[=:]", "data_exfiltration"),
    ]
    
    def check(self, user_input: str) -> SecurityCheckResult:
        """执行安全检查"""
        threats = []
        risk_level = "low"
        
        for pattern, threat_type in self.DANGEROUS_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                threats.append(threat_type)
                risk_level = "high"
        
        # 检查输入长度
        if len(user_input) > 10000:
            threats.append("excessive_length")
            risk_level = "medium"
        
        # 清洗输入
        sanitized = self._sanitize(user_input)
        
        return SecurityCheckResult(
            is_safe=len(threats) == 0,
            risk_level=risk_level,
            threats=threats,
            sanitized_input=sanitized
        )
    
    def _sanitize(self, text: str) -> str:
        """清洗用户输入"""
        # 移除控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        # 限制长度
        if len(text) > 5000:
            text = text[:5000]
        return text.strip()
```

### 1.2.5 Token 预算分配策略

Token 预算分配是上下文窗口管理的核心决策。不同的任务类型需要不同的分配策略。

**动态预算分配算法**

```python
def dynamic_budget_allocation(
    total_budget: int,
    task_type: str,
    history_length: int,
    tool_count: int,
    input_complexity: float  # 0.0 ~ 1.0
) -> dict[str, int]:
    """根据任务特征动态分配 Token 预算"""
    
    # 基础分配比例
    base_ratios = {
        "system": 0.08,
        "tools": 0.12,
        "history": 0.45,
        "input": 0.20,
        "output": 0.15,
    }
    
    # 根据任务类型调整
    if task_type == "tool_heavy":
        # 工具密集型任务：增加工具描述预算
        base_ratios["tools"] = 0.22
        base_ratios["history"] = 0.35
    elif task_type == "long_conversation":
        # 长对话：增加历史消息预算
        base_ratios["history"] = 0.55
        base_ratios["input"] = 0.12
    elif task_type == "code_generation":
        # 代码生成：增加输入和输出预算
        base_ratios["input"] = 0.25
        base_ratios["output"] = 0.20
    elif task_type == "analysis":
        # 分析任务：增加输入和历史预算
        base_ratios["history"] = 0.50
        base_ratios["input"] = 0.22
    
    # 根据历史长度调整
    if history_length > 20:
        # 历史很长时，需要更多摘要空间
        base_ratios["history"] *= 1.2
        base_ratios["tools"] *= 0.8
    
    # 根据工具数量调整
    if tool_count > 10:
        base_ratios["tools"] *= 1.3
        base_ratios["history"] *= 0.7
    
    # 根据输入复杂度调整
    if input_complexity > 0.7:
        base_ratios["input"] *= 1.2
        base_ratios["history"] *= 0.8
    
    # 归一化
    total_ratio = sum(base_ratios.values())
    for key in base_ratios:
        base_ratios[key] /= total_ratio
    
    # 计算实际 Token 数
    allocation = {}
    for key, ratio in base_ratios.items():
        allocation[key] = int(total_budget * ratio)
    
    return allocation
```

---

## 1.3 决策（Decision / Reasoning）

决策是 PDA 循环的核心——LLM 作为"大脑"，基于感知层提供的上下文，做出关于下一步行动的决策。决策的质量直接决定了 Agent 的表现。

### 1.3.1 LLM 推理引擎的核心角色

LLM 在 Agent 中扮演三重角色：

1. **理解器（Comprehender）**：理解用户意图和当前状态
2. **推理器（Reasoner）**：进行逻辑推理和规划
3. **生成器（Generator）**：生成工具调用参数或最终回答

$$
\text{Decision} = \text{LLM}_\theta(\text{Context}) \rightarrow \begin{cases} \text{ToolCall}(name, args) & \text{需要工具} \\ \text{FinalAnswer}(content) & \text{直接回答} \\ \text{Think}(reasoning) & \text{内部推理} \end{cases}
$$

**LLM 推理的内部过程**

```
输入 Context:
  [System] 你是一个数据分析助手...
  [Tools] available_tools: [search, calculate, visualize]
  [User] 请分析这个数据集的趋势并生成图表

LLM 内部推理过程:
  Step 1: 理解意图 → 数据分析 + 可视化
  Step 2: 分解任务 → ① 加载数据 ② 分析趋势 ③ 生成图表
  Step 3: 选择工具 → 先用 search 获取数据，再用 calculate 分析
  Step 4: 生成参数 → search(dataset="sales_2026.csv")
  Step 5: 输出决策 → ToolCall(search, {dataset: "sales_2026.csv"})
```

### 1.3.2 推理策略分类

Agent 的推理策略可以分为三个层次，对应不同的认知复杂度：

**直觉响应（System 1 Thinking）**

这是最简单的推理模式——LLM 直接基于模式匹配给出回答，无需复杂推理。适用于简单查询和常见问题。

$$
a_t = \text{LLM}_\theta(\text{context}) \quad \text{(直接采样)}
$$

特点：
- 延迟低（一次 LLM 调用）
- Token 消耗少
- 适合事实性问题和简单任务

```python
# 直觉响应示例
# 输入: "什么是 Python？"
# LLM 直接回答，无需工具调用
# 输出: "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年发布。"
```

**链式推理（Chain-of-Thought, System 2 Thinking）**

这是更复杂的推理模式——LLM 通过逐步推理来解决问题。对应 Daniel Kahneman 的"系统 2"思维。

$$
\text{CoT}: \quad \text{Thought}_1 \rightarrow \text{Thought}_2 \rightarrow \cdots \rightarrow \text{Thought}_k \rightarrow \text{Action}
$$

特点：
- 延迟较高（需要生成推理链）
- Token 消耗多
- 适合复杂推理、数学问题、多步规划

```python
# Chain-of-Thought 示例
# 输入: "小明有 100 元，买了 3 本书每本 25 元，还剩多少钱？"
# 
# Thought 1: 这是一个简单的减法问题
# Thought 2: 3 本书的总价 = 3 × 25 = 75 元
# Thought 3: 剩余金额 = 100 - 75 = 25 元
# 
# 输出: 25 元
```

**树状搜索（Tree of Thoughts）**

这是最复杂的推理模式——LLM 探索多个可能的推理路径，评估每条路径的前景，选择最优路径。

$$
\text{ToT}: \quad \text{State} \rightarrow \begin{cases} \text{Path}_1 \rightarrow \text{Score}_1 \\ \text{Path}_2 \rightarrow \text{Score}_2 \\ \vdots \\ \text{Path}_k \rightarrow \text{Score}_k \end{cases} \rightarrow \text{BestPath}
$$

特点：
- 延迟最高（多次 LLM 调用）
- Token 消耗最多
- 适合策略规划、创意生成、复杂决策

```python
# Tree of Thoughts 示例
# 输入: "设计一个高并发消息队列系统"
#
# Path 1: Kafka 方案
#   优点: 高吞吐、持久化
#   缺点: 运维复杂、延迟较高
#   Score: 7/10
#
# Path 2: RabbitMQ 方案
#   优点: 易用、低延迟
#   缺点: 性能瓶颈、不适合超大规模
#   Score: 6/10
#
# Path 3: Pulsar 方案
#   优点: 云原生、多租户
#   缺点: 生态较小、学习曲线陡
#   Score: 5/10
#
# 选择 Path 1 并详细设计
```

**三种推理策略的对比**

| 维度 | 直觉响应 | 链式推理 | 树状搜索 |
|:---|:---|:---|:---|
| **LLM 调用次数** | 1 次 | 1 次（长输出） | 多次 |
| **Token 消耗** | 低 | 中 | 高 |
| **延迟** | 低（< 1s） | 中（1-3s） | 高（3-10s+） |
| **适用场景** | 简单查询 | 复杂推理 | 策略规划 |
| **可解释性** | 低 | 高（有推理链） | 高（有搜索树） |
| **可靠性** | 高 | 中 | 最高 |
| **成本** | 低 | 中 | 高 |

<div data-component="ReasoningStrategySelectorV14"></div>

### 1.3.3 决策的不确定性处理

LLM 的输出本质上是概率性的——即使 temperature=0，模型也可能对同一输入给出不同的决策。处理这种不确定性是 Agent 设计的重要课题。

**不确定性来源**

1. **采样随机性**：LLM 输出是通过采样得到的，temperature > 0 时存在随机性
2. **上下文歧义**：输入信息不足或存在多种解读
3. **工具选择不确定**：多个工具都能完成任务时的选择困难
4. **参数生成不确定**：工具参数的取值存在多种可能

**不确定性处理策略**

```python
from dataclasses import dataclass
from typing import Optional
import random


@dataclass
class DecisionCandidate:
    """决策候选"""
    action_type: str              # tool_call / final_answer / think
    tool_name: Optional[str]
    arguments: dict
    confidence: float             # 0.0 ~ 1.0
    reasoning: str                # 推理过程


class UncertaintyHandler:
    """不确定性处理器"""
    
    def __init__(
        self,
        confidence_threshold: float = 0.8,
        resample_threshold: float = 0.5,
        max_resamples: int = 3
    ):
        self.confidence_threshold = confidence_threshold
        self.resample_threshold = resample_threshold
        self.max_resamples = max_resamples
    
    def process_decision(
        self,
        candidates: list[DecisionCandidate]
    ) -> dict:
        """处理决策不确定性"""
        
        if not candidates:
            return {
                "type": "error",
                "message": "没有可用的决策候选"
            }
        
        # 按置信度排序
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        best = candidates[0]
        
        # 策略 1: 高置信度 → 直接执行
        if best.confidence >= self.confidence_threshold:
            return {
                "type": "execute",
                "decision": best,
                "strategy": "high_confidence"
            }
        
        # 策略 2: 中等置信度 → 请求用户确认
        if best.confidence >= self.resample_threshold:
            return {
                "type": "confirm_required",
                "decision": best,
                "alternatives": candidates[1:3],
                "message": f"我有 {best.confidence:.0%} 的把握，是否继续？"
            }
        
        # 策略 3: 低置信度 → 重新采样
        return {
            "type": "resample",
            "current_best": best,
            "strategy": "self_consistency",
            "sample_count": min(3, self.max_resamples)
        }
    
    def self_consistency_vote(
        self,
        decisions: list[DecisionCandidate]
    ) -> DecisionCandidate:
        """Self-Consistency 投票机制"""
        
        if not decisions:
            raise ValueError("决策列表为空")
        
        # 按工具名分组
        groups = {}
        for d in decisions:
            key = (d.action_type, d.tool_name)
            if key not in groups:
                groups[key] = []
            groups[key].append(d)
        
        # 选择投票最多的组
        best_group = max(groups.values(), key=len)
        
        # 在组内选择置信度最高的
        best = max(best_group, key=lambda c: c.confidence)
        
        # 调整置信度：投票数越多，置信度越高
        vote_ratio = len(best_group) / len(decisions)
        best.confidence = min(1.0, best.confidence * (0.5 + 0.5 * vote_ratio))
        
        return best
```

### 1.3.4 Self-Consistency 投票机制

Self-Consistency 是一种强大的推理增强技术，通过多次采样并投票来提高决策质量。

**核心思想**

$$
a^* = \arg\max_{a \in \mathcal{A}} \sum_{i=1}^{N} \mathbb{1}[\text{LLM}_\theta(\text{context} + \epsilon_i) = a]
$$

其中 $\epsilon_i$ 表示第 $i$ 次采样时的随机性（temperature 引入）。

```python
class SelfConsistencyDecider:
    """Self-Consistency 决策器"""
    
    def __init__(self, llm_client, n_samples: int = 5, temperature: float = 0.7):
        self.llm_client = llm_client
        self.n_samples = n_samples
        self.temperature = temperature
    
    def decide(self, context: list[dict], tools: list[dict]) -> dict:
        """多次采样并投票"""
        
        decisions = []
        
        for i in range(self.n_samples):
            # 每次采样使用不同的 temperature
            sample_temp = self.temperature * (0.8 + 0.4 * random.random())
            
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=context,
                tools=tools,
                tool_choice="auto",
                temperature=sample_temp,
            )
            
            message = response.choices[0].message
            
            if message.tool_calls:
                decisions.append({
                    "type": "tool_call",
                    "tool": message.tool_calls[0].function.name,
                    "args": message.tool_calls[0].function.arguments,
                })
            else:
                decisions.append({
                    "type": "final_answer",
                    "content": message.content,
                })
        
        # 投票
        return self._vote(decisions)
    
    def _vote(self, decisions: list[dict]) -> dict:
        """多数投票"""
        from collections import Counter
        
        # 将决策序列化为可哈希的格式
        serialized = [
            json.dumps(d, sort_keys=True, ensure_ascii=False)
            for d in decisions
        ]
        
        counter = Counter(serialized)
        most_common = counter.most_common(1)[0]
        
        # 返回最频繁的决策
        return json.loads(most_common[0])
```

### 1.3.5 推理质量的评估方法

如何评估 Agent 的决策质量？我们需要从多个维度进行评估：

| 评估维度 | 指标 | 计算方法 |
|:---|:---|:---|
| **正确性** | 任务完成率 | 成功完成的轮次 / 总轮次 |
| **效率** | 平均迭代次数 | 总迭代次数 / 总任务数 |
| **工具选择** | 工具匹配率 | 正确选择工具的次数 / 总工具调用次数 |
| **参数准确性** | 参数正确率 | 参数完全正确的次数 / 总调用次数 |
| **推理质量** | CoT 评分 | 人工或 LLM-as-Judge 评分 |
 | **成本效益** | Token 效率 | 有效输出 Token / 总消耗 Token |

工具选择是决策层的关键任务——错误的工具选择会导致整个任务失败。下面的交互式演示展示了工具选择的完整过程：

<div data-component="ToolSelectionDemo"></div>

---

## 1.4 执行（Action / Execution）

执行层是 Agent 与外部世界交互的桥梁。决策层产生的"意图"在这里被转化为实际的操作——调用 API、执行代码、读写文件、发送消息。执行层的设计质量直接影响 Agent 的可靠性和安全性。

### 1.4.1 工具调用执行完整流程

工具调用是 Agent 执行层最核心的能力。一个完整的工具调用流程包括：参数验证 → 安全检查 → 执行 → 结果处理。

```python
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum


class ToolStatus(Enum):
    """工具执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ToolResult:
    """工具执行结果"""
    status: ToolStatus
    tool_name: str
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    token_usage: int = 0
    metadata: dict = field(default_factory=dict)


class ToolExecutor:
    """工具执行器"""
    
    def __init__(
        self,
        tools: dict[str, dict],
        default_timeout: float = 30.0,
        max_retries: int = 2,
        retry_backoff: float = 1.0
    ):
        """
        初始化工具执行器
        
        Args:
            tools: 工具注册表，格式 {name: {func, schema, validator}}
            default_timeout: 默认超时时间（秒）
            max_retries: 最大重试次数
            retry_backoff: 重试退避因子
        """
        self.tools = tools
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.execution_history: list[ToolResult] = []
    
    async def execute(
        self,
        tool_name: str,
        arguments: dict,
        timeout: Optional[float] = None
    ) -> ToolResult:
        """执行单个工具调用"""
        
        start_time = time.time()
        timeout = timeout or self.default_timeout
        
        # Step 1: 工具存在性检查
        if tool_name not in self.tools:
            return ToolResult(
                status=ToolStatus.FAILED,
                tool_name=tool_name,
                error=f"工具 '{tool_name}' 不存在。可用工具: {list(self.tools.keys())}",
                execution_time=time.time() - start_time
            )
        
        tool = self.tools[tool_name]
        
        # Step 2: 参数验证
        if "validator" in tool:
            validation_error = tool["validator"](arguments)
            if validation_error:
                return ToolResult(
                    status=ToolStatus.FAILED,
                    tool_name=tool_name,
                    error=f"参数验证失败: {validation_error}",
                    execution_time=time.time() - start_time
                )
        
        # Step 3: 执行（带重试）
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self._call_tool(tool["func"], arguments),
                    timeout=timeout
                )
                
                execution_time = time.time() - start_time
                
                tool_result = ToolResult(
                    status=ToolStatus.SUCCESS,
                    tool_name=tool_name,
                    result=result,
                    execution_time=execution_time,
                    metadata={"attempt": attempt + 1}
                )
                
                self.execution_history.append(tool_result)
                return tool_result
                
            except asyncio.TimeoutError:
                last_error = f"执行超时（{timeout}秒）"
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                    
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                if attempt < self.max_retries and self._is_retryable(str(e)):
                    await asyncio.sleep(self.retry_backoff * (2 ** attempt))
                    continue
        
        execution_time = time.time() - start_time
        tool_result = ToolResult(
            status=ToolStatus.FAILED,
            tool_name=tool_name,
            error=last_error,
            execution_time=execution_time,
            metadata={"attempts": self.max_retries + 1}
        )
        
        self.execution_history.append(tool_result)
        return tool_result
    
    async def _call_tool(self, func: Callable, arguments: dict) -> Any:
        """调用工具函数"""
        if asyncio.iscoroutinefunction(func):
            return await func(**arguments)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(**arguments))
    
    def _is_retryable(self, error: str) -> bool:
        """判断错误是否可重试"""
        retryable_patterns = [
            "TimeoutError",
            "ConnectionError",
            "RateLimitError",
            "TemporaryFailure",
            "503",  # Service Unavailable
            "429",  # Too Many Requests
        ]
        return any(pattern in error for pattern in retryable_patterns)
    
    async def execute_parallel(
        self,
        tool_calls: list[dict],
        max_concurrency: int = 5
    ) -> list[ToolResult]:
        """并行执行多个工具调用"""
        
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def limited_execute(tc):
            async with semaphore:
                return await self.execute(tc["name"], tc["args"])
        
        tasks = [limited_execute(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks)
        
        return list(results)
```

### 1.4.2 代码执行与沙箱机制

代码执行是 Agent 最强大但也最危险的能力。沙箱机制确保代码在受控环境中运行，防止对系统造成损害。

```python
import subprocess
import tempfile
import os
import signal
from pathlib import Path
from typing import Optional


class CodeSandbox:
    """代码执行沙箱"""
    
    # 危险操作黑名单
    DANGEROUS_PATTERNS = [
        "import os", "import subprocess", "import shutil",
        "os.system", "os.remove", "os.rmdir", "os.unlink",
        "__import__", "eval(", "exec(", "compile(",
        "open('/etc", "open('/var", "open('/usr",
        "subprocess.call", "subprocess.run", "subprocess.Popen",
        "shutil.rmtree", "shutil.move",
    ]
    
    def __init__(
        self,
        timeout: int = 30,
        max_output_length: int = 10000,
        max_memory_mb: int = 256,
        allowed_modules: Optional[list[str]] = None
    ):
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.max_memory_mb = max_memory_mb
        self.allowed_modules = allowed_modules or [
            "math", "random", "json", "re", "datetime",
            "collections", "itertools", "functools",
            "numpy", "pandas", "matplotlib",
        ]
    
    def execute_python(self, code: str) -> dict:
        """在沙箱中执行 Python 代码"""
        
        # Step 1: 安全检查
        security_result = self._security_check(code)
        if not security_result["safe"]:
            return {
                "status": "error",
                "error": f"安全限制: {security_result['reason']}"
            }
        
        # Step 2: 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8'
        ) as f:
            # 添加输出限制包装
            wrapped_code = self._wrap_code(code)
            f.write(wrapped_code)
            temp_path = f.name
        
        try:
            # Step 3: 执行代码
            result = subprocess.run(
                ["python3", temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=self._get_sandbox_env()
            )
            
            # Step 4: 处理输出
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"
            
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n... (输出已截断)"
            
            return {
                "status": "success" if result.returncode == 0 else "error",
                "output": output,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": f"执行超时（{self.timeout}秒）"
            }
        finally:
            os.unlink(temp_path)
    
    def _security_check(self, code: str) -> dict:
        """安全检查"""
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in code:
                return {
                    "safe": False,
                    "reason": f"包含危险操作: {pattern}"
                }
        return {"safe": True, "reason": None}
    
    def _wrap_code(self, code: str) -> str:
        """包装代码以限制输出"""
        return f"""
import sys
import io

# 限制输出长度
class OutputLimiter:
    def __init__(self, max_length={self.max_output_length}):
        self.max_length = max_length
        self.length = 0
        self.truncated = False
    
    def write(self, text):
        if self.length + len(text) > self.max_length:
            remaining = self.max_length - self.length
            sys.__stdout__.write(text[:remaining])
            sys.__stdout__.write("\\n... (输出已截断)")
            self.truncated = True
            raise SystemExit(0)
        sys.__stdout__.write(text)
        self.length += len(text)

sys.stdout = OutputLimiter()

# 执行用户代码
{code}
"""
    
    def _get_sandbox_env(self) -> dict:
        """获取沙箱环境变量"""
        env = os.environ.copy()
        # 只保留必要的环境变量
        safe_keys = ["PATH", "HOME", "LANG", "LC_ALL"]
        return {k: v for k, v in env.items() if k in safe_keys}
```

### 1.4.3 多工具并行执行

当 Agent 决策层一次性输出多个工具调用时，执行层需要高效地并行执行这些调用。

```python
class ParallelToolExecutor:
    """并行工具执行器"""
    
    def __init__(self, tool_executor: ToolExecutor, max_workers: int = 5):
        self.tool_executor = tool_executor
        self.max_workers = max_workers
    
    async def execute_batch(
        self,
        tool_calls: list[dict]
    ) -> list[ToolResult]:
        """批量执行工具调用"""
        
        # 分析依赖关系
        independent, dependent = self._analyze_dependencies(tool_calls)
        
        results = []
        
        # Step 1: 并行执行独立的工具调用
        if independent:
            independent_results = await self.tool_executor.execute_parallel(
                independent, max_concurrency=self.max_workers
            )
            results.extend(independent_results)
        
        # Step 2: 串行执行有依赖的工具调用
        for dep_call in dependent:
            result = await self.tool_executor.execute(
                dep_call["name"], dep_call["args"]
            )
            results.append(result)
        
        return results
    
    def _analyze_dependencies(
        self,
        tool_calls: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """分析工具调用之间的依赖关系"""
        
        independent = []
        dependent = []
        
        # 简化的依赖分析：检查输出是否被其他调用引用
        for tc in tool_calls:
            args_str = json.dumps(tc.get("args", {}))
            # 如果参数引用了其他工具的输出
            if "$ref:" in args_str or "${" in args_str:
                dependent.append(tc)
            else:
                independent.append(tc)
        
        return independent, dependent
```

### 1.4.4 错误处理与重试机制

错误处理是执行层的关键组件。Agent 必须能够优雅地处理各种错误情况，并做出合理的重试决策。

```python
class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        self.error_counts: dict[str, int] = {}
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
    
    def handle_tool_error(
        self,
        tool_name: str,
        error: str,
        context: dict
    ) -> dict:
        """处理工具执行错误"""
        
        # 记录错误
        self.error_counts[tool_name] = self.error_counts.get(tool_name, 0) + 1
        self.consecutive_errors += 1
        
        # 判断错误类型和处理策略
        strategy = self._determine_strategy(error, context)
        
        return {
            "error_type": self._classify_error(error),
            "strategy": strategy,
            "should_retry": strategy == "retry",
            "should_fallback": strategy == "fallback",
            "should_abort": strategy == "abort",
            "error_message": self._format_error_message(error, tool_name),
            "suggestion": self._generate_suggestion(error, tool_name)
        }
    
    def reset_consecutive_errors(self):
        """重置连续错误计数"""
        self.consecutive_errors = 0
    
    def _determine_strategy(self, error: str, context: dict) -> str:
        """确定错误处理策略"""
        
        # 连续错误过多 → 中止
        if self.consecutive_errors >= self.max_consecutive_errors:
            return "abort"
        
        # 可重试错误 → 重试
        if self._is_retryable(error):
            return "retry"
        
        # 工具不可用 → 降级
        if self._is_unavailable(error):
            return "fallback"
        
        # 其他错误 → 返回错误信息让 LLM 决策
        return "report"
    
    def _classify_error(self, error: str) -> str:
        """错误分类"""
        error_lower = error.lower()
        
        if "timeout" in error_lower:
            return "timeout"
        elif "permission" in error_lower or "access denied" in error_lower:
            return "permission"
        elif "not found" in error_lower or "does not exist" in error_lower:
            return "not_found"
        elif "rate limit" in error_lower or "429" in error:
            return "rate_limit"
        elif "connection" in error_lower:
            return "connection"
        else:
            return "unknown"
    
    def _is_retryable(self, error: str) -> bool:
        """判断是否可重试"""
        retryable = ["timeout", "connection", "rate_limit", "503", "429"]
        return any(r in error.lower() for r in retryable)
    
    def _is_unavailable(self, error: str) -> bool:
        """判断工具是否不可用"""
        return "does not exist" in error.lower() or "not found" in error.lower()
    
    def _format_error_message(self, error: str, tool_name: str) -> str:
        """格式化错误消息"""
        return f"工具 '{tool_name}' 执行失败: {error}"
    
    def _generate_suggestion(self, error: str, tool_name: str) -> str:
        """生成处理建议"""
        error_type = self._classify_error(error)
        
        suggestions = {
            "timeout": "可以尝试增加超时时间，或简化请求参数",
            "permission": "请检查工具的访问权限配置",
            "not_found": "请确认工具名称是否正确，或检查工具是否已注册",
            "rate_limit": "请稍后重试，或减少请求频率",
            "connection": "请检查网络连接，或稍后重试",
            "unknown": "请检查工具配置，或联系管理员",
        }
        
        return suggestions.get(error_type, "请检查错误详情")
```

### 1.4.5 执行结果的格式化

工具执行结果需要被格式化为 LLM 可理解的消息格式。

```python
def format_tool_result(
    tool_name: str,
    result: ToolResult,
    max_content_length: int = 4000
) -> str:
    """将工具执行结果格式化为 LLM 消息"""
    
    if result.status == ToolStatus.SUCCESS:
        content = str(result.result) if result.result else "执行成功（无输出）"
        
        if len(content) > max_content_length:
            content = content[:max_content_length] + f"\n... (已截断，共 {len(content)} 字符)"
        
        return (
            f"[工具 {tool_name} 执行成功]\n"
            f"执行时间: {result.execution_time:.2f}秒\n"
            f"结果:\n{content}"
        )
    
    elif result.status == ToolStatus.TIMEOUT:
        return (
            f"[工具 {tool_name} 执行超时]\n"
            f"超时时间: {result.execution_time:.2f}秒\n"
            f"建议: 增加超时时间或简化请求"
        )
    
    else:
        return (
            f"[工具 {tool_name} 执行失败]\n"
            f"错误: {result.error}\n"
            f"执行时间: {result.execution_time:.2f}秒"
        )
```

---

## 1.5 观察与反馈（Observation）

观察层是 PDA 循环的收尾环节——它接收执行层的输出，将其转化为 LLM 可以理解的结构化信息，更新 Agent 状态，并决定循环是否继续。观察层的设计决定了 Agent 能否从错误中恢复、能否积累经验、能否高效利用上下文窗口。

### 1.5.1 工具输出的结构化解析

工具输出可以是任意格式的数据——JSON、纯文本、表格、错误信息。观察层需要将这些异构数据统一解析为结构化的观察。

```python
import json
from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum


class ObservationType(Enum):
    """观察类型"""
    TEXT = "text"
    JSON = "json"
    TABLE = "table"
    ERROR = "error"
    IMAGE = "image"
    CODE_OUTPUT = "code_output"


@dataclass
class StructuredObservation:
    """结构化观察"""
    observation_type: ObservationType
    content: str                       # 格式化后的内容
    raw_data: Any                      # 原始数据
    summary: str                       # 简短摘要
    is_error: bool                     # 是否是错误观察
    confidence: float = 1.0            # 观察的置信度
    metadata: dict = None              # 元数据


class ObservationParser:
    """观察解析器"""
    
    def __init__(self, max_content_length: int = 4000):
        self.max_content_length = max_content_length
    
    def parse(
        self,
        tool_name: str,
        raw_output: Any,
        execution_metadata: Optional[dict] = None
    ) -> StructuredObservation:
        """解析工具输出为结构化观察"""
        
        # 尝试 JSON 解析
        if isinstance(raw_output, str):
            try:
                json_data = json.loads(raw_output)
                return self._parse_json(tool_name, json_data)
            except json.JSONDecodeError:
                pass
        
        # 尝试字典解析
        if isinstance(raw_output, dict):
            return self._parse_dict(tool_name, raw_output)
        
        # 尝试列表解析
        if isinstance(raw_output, list):
            return self._parse_list(tool_name, raw_output)
        
        # 默认：文本解析
        return self._parse_text(tool_name, str(raw_output))
    
    def _parse_json(self, tool_name: str, data: dict) -> StructuredObservation:
        """解析 JSON 输出"""
        
        # 检查是否是错误响应
        if "error" in data or "status" in data and data["status"] == "error":
            return StructuredObservation(
                observation_type=ObservationType.ERROR,
                content=self._format_error(tool_name, data),
                raw_data=data,
                summary=data.get("error", "未知错误"),
                is_error=True
            )
        
        # 格式化 JSON 输出
        formatted = json.dumps(data, ensure_ascii=False, indent=2)
        if len(formatted) > self.max_content_length:
            formatted = formatted[:self.max_content_length] + "\n... (已截断)"
        
        return StructuredObservation(
            observation_type=ObservationType.JSON,
            content=f"[{tool_name} 返回结果]\n{formatted}",
            raw_data=data,
            summary=f"成功获取 {len(data)} 个字段",
            is_error=False
        )
    
    def _parse_dict(self, tool_name: str, data: dict) -> StructuredObservation:
        """解析字典输出"""
        return self._parse_json(tool_name, data)
    
    def _parse_list(self, tool_name: str, data: list) -> StructuredObservation:
        """解析列表输出"""
        formatted = json.dumps(data, ensure_ascii=False, indent=2)
        if len(formatted) > self.max_content_length:
            formatted = formatted[:self.max_content_length] + "\n... (已截断)"
        
        return StructuredObservation(
            observation_type=ObservationType.TABLE,
            content=f"[{tool_name} 返回 {len(data)} 条记录]\n{formatted}",
            raw_data=data,
            summary=f"返回 {len(data)} 条记录",
            is_error=False
        )
    
    def _parse_text(self, tool_name: str, text: str) -> StructuredObservation:
        """解析文本输出"""
        if len(text) > self.max_content_length:
            text = text[:self.max_content_length] + "\n... (已截断)"
        
        return StructuredObservation(
            observation_type=ObservationType.TEXT,
            content=f"[{tool_name} 输出]\n{text}",
            raw_data=text,
            summary=text[:100] + "..." if len(text) > 100 else text,
            is_error=False
        )
    
    def _format_error(self, tool_name: str, data: dict) -> str:
        """格式化错误信息"""
        error_msg = data.get("error", "未知错误")
        details = data.get("details", "")
        
        result = f"[{tool_name} 执行失败]\n错误: {error_msg}"
        if details:
            result += f"\n详情: {details}"
        
        return result
```

### 1.5.2 观察压缩策略

当观察数据过大时，需要进行压缩以适应上下文窗口限制。

```python
class ObservationCompressor:
    """观察压缩器"""
    
    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
    
    def compress(
        self,
        observation: StructuredObservation,
        strategy: str = "adaptive"
    ) -> StructuredObservation:
        """压缩观察结果"""
        
        content = observation.content
        
        # 估算当前 Token 数
        estimated_tokens = len(content) // 2  # 粗略估算
        
        if estimated_tokens <= self.max_tokens:
            return observation
        
        if strategy == "truncate":
            compressed = self._truncate(content)
        elif strategy == "head_tail":
            compressed = self._head_tail(content)
        elif strategy == "summarize":
            compressed = self._summarize(content)
        elif strategy == "adaptive":
            compressed = self._adaptive_compress(content)
        else:
            compressed = self._truncate(content)
        
        return StructuredObservation(
            observation_type=observation.observation_type,
            content=compressed,
            raw_data=observation.raw_data,
            summary=observation.summary,
            is_error=observation.is_error,
            metadata={"compressed": True, "original_tokens": estimated_tokens}
        )
    
    def _truncate(self, content: str) -> str:
        """简单截断"""
        max_chars = self.max_tokens * 2
        return content[:max_chars] + f"\n... (已截断，原始长度 ~{len(content)//2} tokens)"
    
    def _head_tail(self, content: str) -> str:
        """保留头部和尾部"""
        max_chars = self.max_tokens * 2
        head_len = max_chars // 3
        tail_len = max_chars // 3
        
        return (
            content[:head_len]
            + f"\n\n... (省略中间 {len(content) - head_len - tail_len} 字符) ...\n\n"
            + content[-tail_len:]
        )
    
    def _summarize(self, content: str) -> str:
        """摘要压缩"""
        # 提取关键行（非空行的前 N 行）
        lines = content.split("\n")
        key_lines = [l for l in lines if l.strip()][:20]
        
        return (
            "[观察摘要]\n"
            + "\n".join(key_lines)
            + f"\n... (共 {len(lines)} 行，已摘要)"
        )
    
    def _adaptive_compress(self, content: str) -> str:
        """自适应压缩"""
        # 根据内容类型选择压缩策略
        if content.startswith("[") and content.endswith("]"):
            # JSON 数组 → 保留前几项
            return self._compress_json_array(content)
        elif "{" in content and "}" in content:
            # JSON 对象 → 保留关键字段
            return self._compress_json_object(content)
        else:
            # 纯文本 → head_tail
            return self._head_tail(content)
    
    def _compress_json_array(self, content: str) -> str:
        """压缩 JSON 数组"""
        try:
            data = json.loads(content)
            if len(data) > 5:
                return (
                    json.dumps(data[:5], ensure_ascii=False, indent=2)
                    + f"\n... (共 {len(data)} 项，仅显示前 5 项)"
                )
        except json.JSONDecodeError:
            pass
        return self._head_tail(content)
    
    def _compress_json_object(self, content: str) -> str:
        """压缩 JSON 对象"""
        try:
            data = json.loads(content)
            # 保留前 10 个字段
            keys = list(data.keys())[:10]
            compressed = {k: data[k] for k in keys}
            result = json.dumps(compressed, ensure_ascii=False, indent=2)
            if len(data) > 10:
                result += f"\n... (共 {len(data)} 个字段，仅显示前 10 个)"
            return result
        except json.JSONDecodeError:
            return self._head_tail(content)
```

### 1.5.3 错误反馈与重试决策

当工具执行失败时，观察层需要决定如何处理错误——是重试、降级、还是向 LLM 报告并请求新的决策。

```python
class FeedbackDecision:
    """反馈决策"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
    
    def decide(
        self,
        observation: StructuredObservation,
        current_state: dict
    ) -> dict:
        """根据观察结果决定下一步行动"""
        
        if not observation.is_error:
            return {"action": "continue", "reason": "执行成功"}
        
        # 分析错误
        error_info = self.error_handler.handle_tool_error(
            tool_name=current_state.get("current_tool", "unknown"),
            error=observation.content,
            context=current_state
        )
        
        if error_info["should_abort"]:
            return {
                "action": "abort",
                "reason": f"连续错误过多: {error_info['error_message']}"
            }
        
        if error_info["should_retry"]:
            return {
                "action": "retry",
                "reason": f"可重试错误: {error_info['error_message']}",
                "backoff": current_state.get("retry_count", 0) * 2
            }
        
        if error_info["should_fallback"]:
            return {
                "action": "fallback",
                "reason": f"工具不可用: {error_info['error_message']}",
                "suggestion": error_info["suggestion"]
            }
        
        # 报告给 LLM 决策
        return {
            "action": "report_to_llm",
            "error_info": error_info,
            "reason": "需要 LLM 重新决策"
        }
```

### 1.5.4 反馈循环的终止条件

PDA 循环的终止条件决定了 Agent 何时停止执行并返回结果。

```python
class TerminationChecker:
    """终止条件检查器"""
    
    def __init__(self, max_iterations: int = 15, max_consecutive_errors: int = 3):
        self.max_iterations = max_iterations
        self.max_consecutive_errors = max_consecutive_errors
    
    def check(
        self,
        state: dict,
        last_decision: dict,
        last_observation: StructuredObservation
    ) -> tuple[bool, str]:
        """检查是否应该终止循环"""
        
        # 条件 1: LLM 决定给出最终回答
        if last_decision.get("type") == "final_answer":
            return True, "task_completed"
        
        # 条件 2: 达到最大迭代次数
        if state.get("iteration_count", 0) >= self.max_iterations:
            return True, "max_iterations_reached"
        
        # 条件 3: 连续错误过多
        consecutive_errors = state.get("consecutive_errors", 0)
        if last_observation.is_error:
            consecutive_errors += 1
        else:
            consecutive_errors = 0
        
        if consecutive_errors >= self.max_consecutive_errors:
            return True, "too_many_consecutive_errors"
        
        # 条件 4: Token 预算耗尽
        if state.get("token_usage", {}).get("total", 0) > state.get("token_budget", 128000) * 0.95:
            return True, "token_budget_exhausted"
        
        # 条件 5: 用户主动终止
        if state.get("user_requested_stop", False):
            return True, "user_requested_stop"
        
        return False, "continue"
```

### 1.5.5 输出质量评估

Agent 的输出质量需要从多个维度进行评估：

```python
@dataclass
class OutputQualityMetrics:
    """输出质量指标"""
    completeness: float     # 完整性：是否回答了所有问题
    accuracy: float         # 准确性：信息是否正确
    relevance: float        # 相关性：是否切题
    efficiency: float       # 效率：Token 使用是否高效
    safety: float           # 安全性：是否包含有害内容
    overall: float          # 综合评分


def evaluate_output_quality(
    user_query: str,
    agent_output: str,
    tool_results: list[ToolResult],
    conversation_history: list[dict]
) -> OutputQualityMetrics:
    """评估 Agent 输出质量"""
    
    # 完整性评估：检查是否回答了所有子问题
    completeness = _assess_completeness(user_query, agent_output)
    
    # 准确性评估：基于工具结果验证
    accuracy = _assess_accuracy(agent_output, tool_results)
    
    # 相关性评估：输出与查询的语义相关度
    relevance = _assess_relevance(user_query, agent_output)
    
    # 效率评估：有效信息密度
    efficiency = _assess_efficiency(agent_output, tool_results)
    
    # 安全性评估：是否包含有害内容
    safety = _assess_safety(agent_output)
    
    # 综合评分
    overall = (
        completeness * 0.25
        + accuracy * 0.30
        + relevance * 0.20
        + efficiency * 0.15
        + safety * 0.10
    )
    
    return OutputQualityMetrics(
        completeness=completeness,
        accuracy=accuracy,
        relevance=relevance,
        efficiency=efficiency,
        safety=safety,
        overall=overall
    )


def _assess_completeness(query: str, output: str) -> float:
    """评估完整性"""
    # 简化的完整性评估
    query_parts = query.split("？")
    answered = sum(1 for part in query_parts if part.strip() and part.strip() in output)
    return answered / max(len(query_parts), 1)


def _assess_accuracy(output: str, tool_results: list[ToolResult]) -> float:
    """评估准确性"""
    if not tool_results:
        return 0.8  # 没有工具结果时给予默认分
    
    # 检查输出是否引用了工具结果
    success_results = [r for r in tool_results if r.status == ToolStatus.SUCCESS]
    referenced = sum(1 for r in success_results if str(r.result)[:50] in output)
    
    return referenced / max(len(success_results), 1)


def _assess_relevance(query: str, output: str) -> float:
    """评估相关性"""
    # 简化的关键词匹配
    query_words = set(query.split())
    output_words = set(output.split())
    
    if not query_words:
        return 0.5
    
    overlap = len(query_words & output_words)
    return min(overlap / len(query_words), 1.0)


def _assess_efficiency(output: str, tool_results: list[ToolResult]) -> float:
    """评估效率"""
    # 有效信息密度：关键信息长度 / 总输出长度
    if len(output) == 0:
        return 0.0
    
    # 简化：去除冗余标记后的有效内容比例
    effective = output.replace("[", "").replace("]", "").strip()
    return min(len(effective) / len(output), 1.0)


def _assess_safety(output: str) -> float:
    """评估安全性"""
    # 检查是否包含有害内容
    unsafe_patterns = [
        "暴力", "歧视", "色情", "违法",
        "hack", "exploit", "bypass"
    ]
    
    for pattern in unsafe_patterns:
        if pattern in output.lower():
            return 0.0
    
    return 1.0
```

---

## 1.6 Agent 状态管理

状态管理是 Agent 系统的"记忆中枢"——它负责维护 Agent 的完整状态，包括对话历史、任务进度、工具调用记录等。良好的状态管理是构建可靠 Agent 的基础。

### 1.6.1 AgentState 的数据结构设计

```python
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime
from enum import Enum
import json
import hashlib


class AgentPhase(Enum):
    """Agent 当前阶段"""
    IDLE = "idle"
    PERCEIVING = "perceiving"
    DECIDING = "deciding"
    EXECUTING = "executing"
    OBSERVING = "observing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ToolCallRecord:
    """工具调用记录"""
    tool_name: str
    arguments: dict
    result: Any
    status: str           # success / failed / timeout
    execution_time: float
    timestamp: datetime
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_type: str        # think / observe / act
    content: str
    timestamp: datetime
    token_count: int = 0


@dataclass
class AgentState:
    """Agent 完整状态"""
    
    # === 核心状态 ===
    phase: AgentPhase = AgentPhase.IDLE
    task: str = ""                          # 当前任务描述
    
    # === 消息历史 ===
    messages: list[dict] = field(default_factory=list)
    
    # === 执行计划 ===
    plan: list[str] = field(default_factory=list)
    current_step: int = 0
    
    # === 工具调用记录 ===
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    
    # === 中间结果 ===
    intermediate_results: dict[str, Any] = field(default_factory=dict)
    
    # === 推理过程 ===
    reasoning_chain: list[ReasoningStep] = field(default_factory=list)
    
    # === 迭代控制 ===
    iteration_count: int = 0
    max_iterations: int = 15
    
    # === 错误管理 ===
    errors: list[str] = field(default_factory=list)
    consecutive_errors: int = 0
    
    # === Token 管理 ===
    token_usage: dict[str, int] = field(default_factory=lambda: {
        "input": 0, "output": 0, "total": 0
    })
    
    # === 元数据 ===
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    
    # === 会话标识 ===
    session_id: str = ""
    
    def add_message(self, role: str, content: str, **kwargs):
        """添加消息到历史"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def add_tool_call(self, record: ToolCallRecord):
        """添加工具调用记录"""
        self.tool_calls.append(record)
        if record.status == "success":
            self.consecutive_errors = 0
        else:
            self.consecutive_errors += 1
        self.updated_at = datetime.now()
    
    def add_reasoning_step(self, step_type: str, content: str, token_count: int = 0):
        """添加推理步骤"""
        step = ReasoningStep(
            step_type=step_type,
            content=content,
            timestamp=datetime.now(),
            token_count=token_count
        )
        self.reasoning_chain.append(step)
        self.updated_at = datetime.now()
    
    def increment_iteration(self):
        """递增迭代计数"""
        self.iteration_count += 1
        self.updated_at = datetime.now()
    
    def get_summary(self) -> str:
        """获取状态摘要"""
        return (
            f"任务: {self.task[:50]}...\n"
            f"阶段: {self.phase.value}\n"
            f"迭代: {self.iteration_count}/{self.max_iterations}\n"
            f"消息数: {len(self.messages)}\n"
            f"工具调用: {len(self.tool_calls)}\n"
            f"错误数: {len(self.errors)}\n"
            f"Token 使用: {self.token_usage['total']}"
        )
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "phase": self.phase.value,
            "task": self.task,
            "messages": self.messages,
            "plan": self.plan,
            "current_step": self.current_step,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "status": tc.status,
                    "execution_time": tc.execution_time,
                    "timestamp": tc.timestamp.isoformat(),
                }
                for tc in self.tool_calls
            ],
            "intermediate_results": self.intermediate_results,
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
            "errors": self.errors,
            "token_usage": self.token_usage,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AgentState":
        """从字典反序列化"""
        state = cls()
        state.phase = AgentPhase(data.get("phase", "idle"))
        state.task = data.get("task", "")
        state.messages = data.get("messages", [])
        state.plan = data.get("plan", [])
        state.current_step = data.get("current_step", 0)
        state.iteration_count = data.get("iteration_count", 0)
        state.max_iterations = data.get("max_iterations", 15)
        state.errors = data.get("errors", [])
        state.token_usage = data.get("token_usage", {"input": 0, "output": 0, "total": 0})
        state.metadata = data.get("metadata", {})
        return state
    
    def compute_hash(self) -> str:
        """计算状态哈希用于一致性检查"""
        state_str = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]
```

### 1.6.2 消息历史管理（详细实现）

```python
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage,
    AIMessage, ToolMessage
)
import tiktoken


class MessageHistoryManager:
    """消息历史管理器"""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        max_messages: int = 50,
        max_tokens: int = 16000,
        preserve_recent: int = 8
    ):
        self.encoding = tiktoken.encoding_for_model(model)
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.preserve_recent = preserve_recent
        self.messages: list[BaseMessage] = []
        self.summary: str = ""
    
    def add(self, message: BaseMessage):
        """添加消息"""
        self.messages.append(message)
        self._trim()
    
    def get_messages(self) -> list[BaseMessage]:
        """获取消息列表"""
        result = []
        
        # 如果有摘要，先添加摘要
        if self.summary:
            result.append(SystemMessage(
                content=f"[对话历史摘要]\n{self.summary}"
            ))
        
        result.extend(self.messages)
        return result
    
    def _trim(self):
        """裁剪消息历史"""
        # 检查消息数量
        if len(self.messages) <= self.max_messages:
            # 检查 Token 数
            total_tokens = self._count_total_tokens()
            if total_tokens <= self.max_tokens:
                return
        
        # 需要裁剪
        system_msgs = [m for m in self.messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in self.messages if not isinstance(m, SystemMessage)]
        
        # 保留最近的消息
        preserve_count = min(self.preserve_recent, len(other_msgs))
        recent = other_msgs[-preserve_count:]
        old = other_msgs[:-preserve_count]
        
        # 对旧消息生成摘要
        if old:
            self.summary = self._create_summary(old)
        
        # 重建消息列表
        self.messages = system_msgs + recent
    
    def _count_total_tokens(self) -> int:
        """计算总 Token 数"""
        total = 0
        for msg in self.messages:
            content = msg.content or ""
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total += len(self.encoding.encode(part["text"]))
            else:
                total += len(self.encoding.encode(content))
        return total
    
    def _create_summary(self, messages: list[BaseMessage]) -> str:
        """创建消息摘要"""
        parts = []
        
        for msg in messages:
            role = msg.type
            content = msg.content or ""
            
            if isinstance(content, list):
                text_parts = [
                    p["text"] for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = " ".join(text_parts)
            
            # 截断
            if len(content) > 150:
                content = content[:150] + "..."
            
            parts.append(f"[{role}] {content}")
        
        return "\n".join(parts)
    
    def clear(self):
        """清空消息历史"""
        self.messages = []
        self.summary = ""
    
    def export_for_llm(self) -> list[dict]:
        """导出为 LLM 格式"""
        result = []
        
        if self.summary:
            result.append({
                "role": "system",
                "content": f"[对话历史摘要]\n{self.summary}"
            })
        
        for msg in self.messages:
            content = msg.content or ""
            if isinstance(content, list):
                # 多模态消息，只取文本
                text_parts = [
                    p["text"] for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            
            result.append({
                "role": msg.type if msg.type != "tool" else "assistant",
                "content": content
            })
        
        return result
```

### 1.6.3 中间推理结果存储

```python
class IntermediateResultStore:
    """中间结果存储"""
    
    def __init__(self):
        self.results: dict[str, Any] = {}
        self.access_count: dict[str, int] = {}
        self.creation_time: dict[str, datetime] = {}
    
    def store(self, key: str, value: Any, metadata: dict = None):
        """存储中间结果"""
        self.results[key] = {
            "value": value,
            "metadata": metadata or {},
        }
        self.access_count[key] = 0
        self.creation_time[key] = datetime.now()
    
    def retrieve(self, key: str) -> Optional[Any]:
        """检索中间结果"""
        if key in self.results:
            self.access_count[key] += 1
            return self.results[key]["value"]
        return None
    
    def get_all_keys(self) -> list[str]:
        """获取所有存储的键"""
        return list(self.results.keys())
    
    def get_stats(self) -> dict:
        """获取存储统计"""
        return {
            "total_results": len(self.results),
            "most_accessed": max(
                self.access_count.items(),
                key=lambda x: x[1],
                default=("none", 0)
            ),
            "total_accesses": sum(self.access_count.values()),
        }
    
    def clear(self):
        """清空存储"""
        self.results.clear()
        self.access_count.clear()
        self.creation_time.clear()
```

### 1.6.4 状态序列化与恢复

```python
class StatePersistenceManager:
    """状态持久化管理器"""
    
    def __init__(self, storage_path: str = "./agent_states"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, session_id: str, state: AgentState) -> str:
        """保存状态"""
        file_path = self.storage_path / f"{session_id}.json"
        
        state_dict = state.to_dict()
        state_dict["_hash"] = state.compute_hash()
        state_dict["_saved_at"] = datetime.now().isoformat()
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2)
        
        return str(file_path)
    
    def load(self, session_id: str) -> Optional[AgentState]:
        """加载状态"""
        file_path = self.storage_path / f"{session_id}.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, "r", encoding="utf-8") as f:
            state_dict = json.load(f)
        
        # 验证哈希
        saved_hash = state_dict.pop("_hash", None)
        state = AgentState.from_dict(state_dict)
        
        if saved_hash and saved_hash != state.compute_hash():
            raise ValueError("状态文件完整性验证失败")
        
        return state
    
    def list_sessions(self) -> list[str]:
        """列出所有会话"""
        return [
            f.stem for f in self.storage_path.glob("*.json")
        ]
    
    def delete(self, session_id: str) -> bool:
        """删除会话"""
        file_path = self.storage_path / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
```

### 1.6.5 状态一致性保证

```python
class StateConsistencyChecker:
    """状态一致性检查器"""
    
    @staticmethod
    def validate(state: AgentState) -> list[str]:
        """验证状态一致性"""
        issues = []
        
        # 检查迭代次数
        if state.iteration_count > state.max_iterations:
            issues.append(f"迭代次数 {state.iteration_count} 超过最大限制 {state.max_iterations}")
        
        # 检查消息与工具调用的一致性
        tool_call_count = len(state.tool_calls)
        tool_messages = sum(
            1 for m in state.messages
            if isinstance(m, dict) and m.get("role") == "tool"
        )
        
        if tool_call_count != tool_messages:
            issues.append(
                f"工具调用次数 ({tool_call_count}) 与工具消息数 ({tool_messages}) 不匹配"
            )
        
        # 检查连续错误
        if state.consecutive_errors > 5:
            issues.append(f"连续错误数 {state.consecutive_errors} 过高")
        
        # 检查 Token 使用
        if state.token_usage.get("total", 0) > 200000:
            issues.append(f"Token 总使用量 {state.token_usage['total']} 过高")
        
        return issues
    
    @staticmethod
    def repair(state: AgentState) -> AgentState:
        """修复状态问题"""
        # 重置连续错误
        if state.consecutive_errors > 5:
            state.consecutive_errors = 3
        
        # 限制迭代次数
        if state.iteration_count > state.max_iterations:
            state.iteration_count = state.max_iterations
        
        # 清理过期的中间结果
        if len(state.intermediate_results) > 50:
            # 保留最近 20 个
            keys = list(state.intermediate_results.keys())
            for key in keys[:-20]:
                del state.intermediate_results[key]
        
        return state
```

---

## 1.7 完整 Agent Step 实现

现在我们有了所有组件，可以构建一个完整的 Agent Step 函数。这是本章的核心实现——将感知、决策、执行、观察四个阶段整合为一个统一的流程。

### 1.7.1 从零构建完整 Agent Step 函数

```python
"""
完整 Agent 实现：从零构建 PDA 循环
这是一个教学级别的完整实现，展示了 Agent 的核心机制。
"""
import openai
import json
import time
import asyncio
from typing import Any, Optional, Callable
from dataclasses import dataclass, field


class AgentStepResult:
    """Agent 单步执行结果"""
    
    def __init__(
        self,
        phase: str,
        success: bool,
        output: Any = None,
        error: Optional[str] = None,
        token_usage: Optional[dict] = None,
        execution_time: float = 0.0,
        metadata: Optional[dict] = None
    ):
        self.phase = phase
        self.success = success
        self.output = output
        self.error = error
        self.token_usage = token_usage or {}
        self.execution_time = execution_time
        self.metadata = metadata or {}
    
    def __repr__(self):
        status = "✓" if self.success else "✗"
        return f"AgentStepResult({status}, {self.phase}, {self.execution_time:.2f}s)"


class FullAgent:
    """完整 Agent 实现"""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        tools: Optional[dict] = None,
        system_prompt: str = "你是一个有用的 AI 助手，可以使用工具来完成任务。",
        max_iterations: int = 15,
        max_tokens_per_step: int = 4096,
        temperature: float = 0,
        verbose: bool = True
    ):
        # 初始化 LLM 客户端
        self.client = openai.OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens_per_step = max_tokens_per_step
        
        # 工具注册
        self.tools = tools or {}
        
        # 系统提示词
        self.system_prompt = system_prompt
        
        # 迭代控制
        self.max_iterations = max_iterations
        
        # 日志控制
        self.verbose = verbose
        
        # 组件初始化
        self.context_manager = ContextWindowManager(model=model)
        self.tool_executor = ToolExecutor(tools=self.tools)
        self.observation_parser = ObservationParser()
        self.error_handler = ErrorHandler()
        self.termination_checker = TerminationChecker(max_iterations=max_iterations)
        self.state = AgentState(max_iterations=max_iterations)
    
    def run(self, user_message: str) -> str:
        """运行 Agent 直到任务完成"""
        
        self.state.task = user_message
        self.state.add_message("user", user_message)
        
        self._log(f"\n{'='*60}")
        self._log(f"Agent 开始执行任务: {user_message[:50]}...")
        self._log(f"{'='*60}")
        
        for iteration in range(self.max_iterations):
            self._log(f"\n--- Step {iteration + 1} ---")
            
            # 执行一个完整的 PDA 循环
            step_result = self._execute_step()
            
            self.state.increment_iteration()
            
            if self.verbose:
                self._log(f"Step 结果: {step_result}")
            
            # 检查终止条件
            should_stop, reason = self.termination_checker.check(
                state=self.state.to_dict(),
                last_decision=step_result.metadata.get("decision", {}),
                last_observation=step_result.metadata.get("observation",
                    StructuredObservation(
                        observation_type=ObservationType.TEXT,
                        content="",
                        raw_data=None,
                        summary="",
                        is_error=not step_result.success
                    ))
            )
            
            if should_stop:
                self._log(f"\n循环终止: {reason}")
                if step_result.output:
                    return step_result.output
                return f"任务未能完成: {reason}"
        
        return "达到最大迭代次数，任务未完成。"
    
    def _execute_step(self) -> AgentStepResult:
        """执行一个完整的 PDA 步骤"""
        
        step_start = time.time()
        
        try:
            # === Phase 1: 感知 ===
            self.state.phase = AgentPhase.PERCEIVING
            context = self._perceive()
            self._log(f"[Perception] 上下文构建完成，{len(context)} 条消息")
            
            # === Phase 2: 决策 ===
            self.state.phase = AgentPhase.DECIDING
            decision = self._decide(context)
            self._log(f"[Decision] 决策类型: {decision.get('type')}")
            
            # === Phase 3: 执行 ===
            self.state.phase = AgentPhase.EXECUTING
            if decision["type"] == "final_answer":
                self.state.phase = AgentPhase.COMPLETED
                return AgentStepResult(
                    phase="completed",
                    success=True,
                    output=decision["content"],
                    execution_time=time.time() - step_start,
                    metadata={"decision": decision}
                )
            
            execution_result = self._act(decision)
            self._log(f"[Action] 执行结果: {execution_result.status.value}")
            
            # === Phase 4: 观察 ===
            self.state.phase = AgentPhase.OBSERVING
            observation = self._observe(execution_result)
            self._log(f"[Observation] 观察类型: {observation.observation_type.value}")
            
            # 更新状态
            self.state.add_message("assistant", decision.get("reasoning", ""))
            self.state.add_message("tool", observation.content)
            
            step_time = time.time() - step_start
            
            return AgentStepResult(
                phase="observe",
                success=True,
                output=None,
                execution_time=step_time,
                token_usage=self.state.token_usage,
                metadata={
                    "decision": decision,
                    "observation": observation,
                    "execution_result": execution_result
                }
            )
            
        except Exception as e:
            self.state.phase = AgentPhase.FAILED
            self.state.errors.append(str(e))
            
            return AgentStepResult(
                phase="error",
                success=False,
                error=str(e),
                execution_time=time.time() - step_start
            )
    
    def _perceive(self) -> list[dict]:
        """感知阶段：构建上下文"""
        
        # 构建工具描述
        tool_descriptions = []
        for name, tool in self.tools.items():
            desc = tool.get("description", "")
            params = tool.get("schema", {}).get("function", {}).get("parameters", {})
            tool_descriptions.append(f"- {name}: {desc}\n  参数: {json.dumps(params, ensure_ascii=False)}")
        
        # 使用上下文管理器
        managed = self.context_manager.manage_context(
            system_prompt=self.system_prompt,
            tool_descriptions=tool_descriptions,
            history=self.state.messages,
            current_input=self.state.messages[-1]["content"] if self.state.messages else ""
        )
        
        # 转换为 LLM 格式
        messages = []
        for msg in managed.messages:
            if isinstance(msg, SystemMessage):
                messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                messages.append({"role": "tool", "content": msg.content})
        
        # 更新 Token 使用
        self.state.token_usage = managed.token_usage
        
        return messages
    
    def _decide(self, context: list[dict]) -> dict:
        """决策阶段：调用 LLM"""
        
        # 准备工具 schema
        tool_schemas = []
        for name, tool in self.tools.items():
            tool_schemas.append(tool.get("schema", {}))
        
        # 调用 LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=context,
            tools=tool_schemas if tool_schemas else None,
            tool_choice="auto" if tool_schemas else None,
            temperature=self.temperature,
            max_tokens=self.max_tokens_per_step,
        )
        
        message = response.choices[0].message
        
        # 更新 Token 使用
        if response.usage:
            self.state.token_usage["input"] += response.usage.prompt_tokens
            self.state.token_usage["output"] += response.usage.completion_tokens
            self.state.token_usage["total"] += response.usage.total_tokens
        
        # 解析决策
        if message.tool_calls:
            # 工具调用决策
            tool_call = message.tool_calls[0]
            return {
                "type": "tool_call",
                "tool_name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments),
                "tool_call_id": tool_call.id,
                "reasoning": message.content or "",
            }
        else:
            # 最终回答
            return {
                "type": "final_answer",
                "content": message.content or "",
                "reasoning": message.content or "",
            }
    
    def _act(self, decision: dict) -> ToolResult:
        """执行阶段：调用工具"""
        
        tool_name = decision["tool_name"]
        arguments = decision["arguments"]
        
        self._log(f"  调用工具: {tool_name}({json.dumps(arguments, ensure_ascii=False)[:100]})")
        
        # 同步执行工具
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult(
                status=ToolStatus.FAILED,
                tool_name=tool_name,
                error=f"工具 '{tool_name}' 不存在"
            )
        
        try:
            func = tool["func"]
            start = time.time()
            
            # 执行函数
            if asyncio.iscoroutinefunction(func):
                result = asyncio.get_event_loop().run_until_complete(func(**arguments))
            else:
                result = func(**arguments)
            
            execution_time = time.time() - start
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                tool_name=tool_name,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILED,
                tool_name=tool_name,
                error=f"{type(e).__name__}: {str(e)}"
            )
    
    def _observe(self, execution_result: ToolResult) -> StructuredObservation:
        """观察阶段：解析结果"""
        
        # 解析观察
        observation = self.observation_parser.parse(
            tool_name=execution_result.tool_name,
            raw_output=execution_result.result if execution_result.status == ToolStatus.SUCCESS
                       else execution_result.error
        )
        
        # 记录工具调用
        record = ToolCallRecord(
            tool_name=execution_result.tool_name,
            arguments={},
            result=execution_result.result,
            status=execution_result.status.value,
            execution_time=execution_result.execution_time,
            timestamp=datetime.now(),
            error=execution_result.error
        )
        self.state.add_tool_call(record)
        
        return observation
    
    def _log(self, message: str):
        """日志输出"""
        if self.verbose:
            print(message)
```

### 1.7.2 使用示例

```python
# 创建 Agent
agent = FullAgent(
    model="gpt-4o",
    system_prompt="你是一个有用的数据分析助手。你可以搜索信息、进行计算、分析数据。",
    max_iterations=10,
    verbose=True
)

# 注册工具
def search_web(query: str) -> str:
    """模拟网络搜索"""
    # 实际实现中，这里会调用搜索引擎 API
    return f"搜索结果: 关于 '{query}' 的相关信息..."

def calculate(expression: str) -> str:
    """安全的数学计算"""
    import ast
    try:
        # 使用 ast.literal_eval 进行安全评估
        tree = ast.parse(expression, mode='eval')
        # 简化：实际应该实现完整的安全计算
        result = eval(compile(tree, '<string>', 'eval'))
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

def read_file(file_path: str) -> str:
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取错误: {e}"

# 注册工具
agent.tools = {
    "search_web": {
        "func": search_web,
        "description": "搜索网络获取信息",
        "schema": {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "搜索网络获取最新信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    },
    "calculate": {
        "func": calculate,
        "description": "执行数学计算",
        "schema": {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "安全地执行数学表达式计算",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学表达式，如 '2 + 3 * 4'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    },
    "read_file": {
        "func": read_file,
        "description": "读取本地文件内容",
        "schema": {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取指定路径的文件内容",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "文件路径"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        }
    }
}

# 运行示例
result = agent.run("请搜索 Python 3.12 的新特性，并计算 2 的 10 次方")
print(f"\n最终结果:\n{result}")
```

### 1.7.3 与 LangChain AgentExecutor 的对比分析

| 维度 | 我们的实现 | LangChain AgentExecutor |
|:---|:---|:---|
| **架构复杂度** | 简单直接 | 抽象层多，灵活但复杂 |
| **状态管理** | 手动管理 AgentState | 内置 RunnableState |
| **工具调用** | 直接函数调用 | Tool 抽象 + Toolkit |
| **错误处理** | 自定义 ErrorHandler | 内置 retry 机制 |
| **上下文管理** | 手动 Token 预算 | ChatMessageHistory |
| **可观测性** | print 日志 | LangSmith 集成 |
| **并行执行** | asyncio.gather | 内置 parallel 工具调用 |
| **学习曲线** | 低（原理清晰） | 中（需要理解抽象） |
| **生产就绪** | 需要额外工作 | 内置生产特性 |
| **可定制性** | 完全控制 | 通过回调/扩展点 |

<div data-component="AgentArchitectureComparisonV2"></div>

**选择建议**

- **学习/研究**：使用我们的实现，理解底层原理
- **快速原型**：使用 LangChain AgentExecutor，快速搭建
- **生产系统**：基于我们的实现理念，构建定制化方案

---

## 1.8 PDA 循环的优化策略

优化 PDA 循环的目标是：更快（延迟更低）、更省（成本更低）、更稳（可靠性更高）。

### 1.8.1 并行化策略

并行化可以显著减少 Agent 的执行时间。主要的并行化机会包括：

| 并行化机会 | 描述 | 加速比 |
|:---|:---|:---|
| **工具并行** | 多个独立工具同时执行 | 2-5x |
| **推理批处理** | 多个推理任务并行 | 1.5-3x |
| **流水线并行** | 感知-决策-执行流水线化 | 1.2-2x |
| **推测执行** | 预测可能的下一步并提前执行 | 1.5-4x |

```python
class ParallelPDAOptimizer:
    """PDA 循环并行化优化器"""
    
    def __init__(self, max_parallel_tools: int = 5):
        self.max_parallel_tools = max_parallel_tools
    
    async def optimize_step(
        self,
        agent: FullAgent,
        context: list[dict]
    ) -> AgentStepResult:
        """优化的 Agent Step"""
        
        # 1. 并行调用 LLM 获取多个候选决策
        candidates = await self._parallel_decide(context, n_candidates=3)
        
        # 2. 评估候选决策
        best_decision = self._select_best(candidates)
        
        # 3. 如果有多个独立工具调用，并行执行
        if best_decision.get("type") == "parallel_tool_calls":
            results = await self._parallel_execute(best_decision["tool_calls"])
            return self._merge_results(results)
        
        # 4. 否则串行执行
        return await agent._execute_step()
    
    async def _parallel_decide(
        self,
        context: list[dict],
        n_candidates: int = 3
    ) -> list[dict]:
        """并行获取多个决策候选"""
        
        async def get_candidate(index: int) -> dict:
            # 使用不同的 temperature 获取多样性
            temp = 0.3 + index * 0.2
            # 调用 LLM...
            return {"type": "tool_call", "tool_name": "search", "confidence": 0.8}
        
        tasks = [get_candidate(i) for i in range(n_candidates)]
        return await asyncio.gather(*tasks)
    
    def _select_best(self, candidates: list[dict]) -> dict:
        """选择最佳决策"""
        return max(candidates, key=lambda c: c.get("confidence", 0))
    
    async def _parallel_execute(self, tool_calls: list[dict]) -> list[ToolResult]:
        """并行执行多个工具"""
        executor = ToolExecutor(tools={})
        return await executor.execute_parallel(tool_calls)
    
    def _merge_results(self, results: list[ToolResult]) -> AgentStepResult:
        """合并并行执行结果"""
        # 合并所有结果
        merged_content = "\n\n".join([
            f"[{r.tool_name}] {r.result}" for r in results if r.status == ToolStatus.SUCCESS
        ])
        
        return AgentStepResult(
            phase="parallel_execute",
            success=all(r.status == ToolStatus.SUCCESS for r in results),
            output=merged_content,
            metadata={"results": results}
        )
```

### 1.8.2 缓存策略

缓存可以避免重复的 LLM 调用和工具执行，显著降低成本和延迟。

```python
import hashlib
from typing import Optional
from datetime import datetime, timedelta


class AgentCache:
    """Agent 缓存系统"""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self.ttl = timedelta(seconds=ttl_seconds)
        self.max_size = max_size
        self.cache: dict[str, dict] = {}
    
    def _make_key(self, messages: list[dict], tools: list[str]) -> str:
        """生成缓存键"""
        content = json.dumps({
            "messages": messages,
            "tools": tools
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, messages: list[dict], tools: list[str]) -> Optional[dict]:
        """获取缓存"""
        key = self._make_key(messages, tools)
        
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry["timestamp"] < self.ttl:
                return entry["value"]
            else:
                del self.cache[key]
        
        return None
    
    def set(self, messages: list[dict], tools: list[str], value: dict):
        """设置缓存"""
        # 检查缓存大小
        if len(self.cache) >= self.max_size:
            # 删除最旧的条目
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k]["timestamp"]
            )
            del self.cache[oldest_key]
        
        key = self._make_key(messages, tools)
        self.cache[key] = {
            "value": value,
            "timestamp": datetime.now()
        }
    
    def invalidate(self, pattern: Optional[str] = None):
        """使缓存失效"""
        if pattern:
            keys_to_delete = [k for k in self.cache if pattern in k]
            for key in keys_to_delete:
                del self.cache[key]
        else:
            self.cache.clear()
```

### 1.8.3 提前终止策略

提前终止可以避免不必要的计算，减少资源消耗。

```python
class EarlyTerminationPolicy:
    """提前终止策略"""
    
    def __init__(self):
        self.policies = [
            self._check_confidence,
            self._check_progress,
            self._check_repetition,
            self._check_cost,
        ]
    
    def should_terminate_early(
        self,
        state: AgentState,
        decision: dict
    ) -> tuple[bool, str]:
        """检查是否应该提前终止"""
        
        for policy in self.policies:
            should_stop, reason = policy(state, decision)
            if should_stop:
                return True, reason
        
        return False, "continue"
    
    def _check_confidence(
        self,
        state: AgentState,
        decision: dict
    ) -> tuple[bool, str]:
        """检查决策置信度"""
        confidence = decision.get("confidence", 1.0)
        if confidence > 0.95:
            return True, "high_confidence_termination"
        return False, ""
    
    def _check_progress(
        self,
        state: AgentState,
        decision: dict
    ) -> tuple[bool, str]:
        """检查进展"""
        # 如果连续 3 次迭代没有新信息，提前终止
        if len(state.tool_calls) >= 3:
            recent_results = [tc.result for tc in state.tool_calls[-3:]]
            if len(set(str(r) for r in recent_results)) == 1:
                return True, "no_progress_termination"
        return False, ""
    
    def _check_repetition(
        self,
        state: AgentState,
        decision: dict
    ) -> tuple[bool, str]:
        """检查重复"""
        # 检查是否重复调用同一个工具
        if len(state.tool_calls) >= 2:
            recent_tools = [tc.tool_name for tc in state.tool_calls[-2:]]
            if len(set(recent_tools)) == 1:
                return True, "repetition_termination"
        return False, ""
    
    def _check_cost(
        self,
        state: AgentState,
        decision: dict
    ) -> tuple[bool, str]:
        """检查成本"""
        if state.token_usage.get("total", 0) > 100000:
            return True, "cost_termination"
        return False, ""
```

### 1.8.4 错误恢复策略

错误恢复策略确保 Agent 在遇到错误时能够优雅地恢复，而不是直接崩溃。

```python
class ErrorRecoveryStrategy:
    """错误恢复策略"""
    
    def __init__(self):
        self.recovery_strategies = {
            "tool_not_found": self._recover_tool_not_found,
            "timeout": self._recover_timeout,
            "permission": self._recover_permission,
            "rate_limit": self._recover_rate_limit,
            "invalid_output": self._recover_invalid_output,
        }
    
    def recover(
        self,
        state: AgentState,
        error_type: str,
        error_info: dict
    ) -> dict:
        """执行错误恢复"""
        
        strategy = self.recovery_strategies.get(error_type)
        if strategy:
            return strategy(state, error_info)
        
        return {
            "action": "report",
            "message": f"未知错误类型: {error_type}",
            "should_continue": False
        }
    
    def _recover_tool_not_found(
        self,
        state: AgentState,
        error_info: dict
    ) -> dict:
        """恢复工具不存在错误"""
        return {
            "action": "suggest_alternative",
            "message": f"工具 '{error_info.get('tool_name')}' 不存在，建议使用其他工具",
            "should_continue": True,
            "alternative_tools": self._find_alternatives(error_info.get("tool_name"))
        }
    
    def _recover_timeout(
        self,
        state: AgentState,
        error_info: dict
    ) -> dict:
        """恢复超时错误"""
        return {
            "action": "retry_with_simplified_params",
            "message": "工具执行超时，尝试简化参数后重试",
            "should_continue": True,
            "simplified_params": self._simplify_params(error_info.get("params", {}))
        }
    
    def _recover_permission(
        self,
        state: AgentState,
        error_info: dict
    ) -> dict:
        """恢复权限错误"""
        return {
            "action": "fallback_to_alternative",
            "message": f"权限不足: {error_info.get('error')}",
            "should_continue": True
        }
    
    def _recover_rate_limit(
        self,
        state: AgentState,
        error_info: dict
    ) -> dict:
        """恢复限流错误"""
        return {
            "action": "wait_and_retry",
            "message": "请求被限流，等待后重试",
            "should_continue": True,
            "wait_seconds": 30
        }
    
    def _recover_invalid_output(
        self,
        state: AgentState,
        error_info: dict
    ) -> dict:
        """恢复无效输出错误"""
        return {
            "action": "retry_with_different_prompt",
            "message": "工具输出格式无效，尝试不同的提示词",
            "should_continue": True
        }
    
    def _find_alternatives(self, tool_name: str) -> list[str]:
        """查找替代工具"""
        # 简化的实现
        alternatives = {
            "search_web": ["search_knowledge", "browse_url"],
            "calculate": ["eval_expression", "python_exec"],
        }
        return alternatives.get(tool_name, [])
    
    def _simplify_params(self, params: dict) -> dict:
        """简化工具参数"""
        simplified = {}
        for key, value in params.items():
            if isinstance(value, str) and len(value) > 100:
                simplified[key] = value[:100] + "..."
            else:
                simplified[key] = value
        return simplified
```

---

在实际工程中，PDA 循环有多种不同的实现方式，各有优劣。下面的交互式对比可以帮助你根据项目需求选择最合适的实现方案：

<div data-component="AgentArchitectureComparisonV3"></div>

---

## 1.9 本章小结

本章深入剖析了 Agent 的核心循环机制——感知-决策-执行（PDA）循环。这是理解所有 Agent 系统的基础。

| 知识点 | 核心要点 |
|:---|:---|
| **PDA 循环** | Agent 的核心是感知-决策-执行循环，可形式化为非平稳 MDP |
| **与 RL 对比** | LLM Agent 使用上下文学习而非在线学习，状态空间无限 |
| **感知层** | 负责输入解析、多模态处理、上下文组装、Token 预算管理 |
| **决策层** | LLM 作为推理引擎，支持直觉响应、链式推理、树状搜索三种策略 |
| **执行层** | 工具调用、代码执行、并行执行、错误处理与重试机制 |
| **观察层** | 结果解析、压缩策略、错误反馈、终止条件判断 |
| **状态管理** | AgentState 数据结构、消息历史、中间结果、序列化恢复 |
| **完整实现** | 从零构建 FullAgent，展示 PDA 循环全貌 |
| **优化策略** | 并行化、缓存、提前终止、错误恢复 |

> **关键洞察**
>
> 1. PDA 循环的本质是**信息流的循环**——从环境获取信息，经过 LLM 处理，产生行动，影响环境，再次获取信息。
> 2. 上下文窗口是 Agent 最宝贵的资源——所有信息都必须在有限的 Token 预算内高效组织。
> 3. LLM Agent 的"学习"发生在上下文中，而非参数更新中——这既是优势（无需训练），也是限制（受窗口大小约束）。
> 4. 错误处理是 Agent 可靠性的关键——优雅的错误恢复比完美的正确执行更重要。

> **下一章预告**
>
> 在第 2 章中，我们将深入 Agent 的"大脑"——大语言模型，理解 Transformer 架构、Token 机制、模型参数对 Agent 行为的影响，以及如何选择和配置最适合你场景的 LLM。

---

## 附录：本章核心概念速查表

| 概念 | 定义 | 公式/表示 |
|:---|:---|:---|
| PDA 循环 | Perception-Decision-Action 循环 | $\text{Step}_t = \text{Action}(\text{Decision}(\text{Perception}(s_t, o_t)))$ |
| 状态空间 | Agent 所有可能内部状态的集合 | $\mathcal{S} = \mathcal{M} \times \mathcal{P} \times \mathcal{R} \times \mathcal{E} \times \mathbb{N}$ |
| 动作空间 | Agent 可以采取的所有行动的集合 | $\mathcal{A} = \{(\text{tool}, name, args)\} \cup \{(\text{answer}, content)\}$ |
| 观察空间 | 环境可能返回的所有观察的集合 | $\mathcal{O} = \{\text{user\_input}, \text{tool\_result}, \text{system\_event}\}$ |
| Token 预算 | 上下文窗口的分配策略 | $S + T + H + I + R \leq L$ |
| Self-Consistency | 多次采样并投票的决策机制 | $a^* = \arg\max_{a} \sum_{i=1}^{N} \mathbb{1}[\text{LLM}_\theta(\text{ctx} + \epsilon_i) = a]$ |
| AgentState | Agent 完整状态的数据结构 | $s = (\mathcal{M}, \mathcal{P}, \mathcal{R}, \mathcal{E}, n)$ |
| Context Manager | 上下文窗口管理器 | 管理 Token 预算，执行压缩和摘要 |
| Tool Executor | 工具执行器 | 支持并行执行、超时控制、错误重试 |
| Termination Checker | 终止条件检查器 | 多条件综合判断循环是否终止 |
