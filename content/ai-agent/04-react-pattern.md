---
title: "第4章：ReAct 范式 — 推理与行动的统一"
description: "深入理解 ReAct 范式的理论基础与实现细节，掌握 Thought-Action-Observation 循环，对比纯推理与纯行动策略的优劣，手动实现 ReAct Agent，掌握 LangGraph 集成。"
date: "2026-06-11"
---

 # 第4章：ReAct 范式 — 推理与行动的统一

 ReAct（Reasoning + Acting）是现代 LLM Agent 的基础范式。本章深入剖析 ReAct 的理论基础、实现细节与工程实践。理解 ReAct 是掌握 Agent 开发的关键一步——它不仅是一种技术方案，更是一种思考 Agent 的方式。

 下面的交互式演示展示了 ReAct 循环的工作流程：

 <div data-component="ReactPatternDemo"></div>

 ## 4.1 ReAct 理论基础

在 ReAct 诞生之前，大语言模型（LLM）在任务解决方面主要沿着两条独立的路径发展：

**路径一：纯推理（Reasoning-only）**

以 Chain-of-Thought（CoT）为代表，通过诱导 LLM 生成中间推理步骤来提升复杂问题的求解能力。Wei et al. (2022) 在 "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" 中证明，简单的 few-shot prompt 就能让 LLM 展现出强大的推理能力。

CoT 的核心思想是：将复杂问题分解为一系列中间步骤，每一步都基于前一步的结论进行推理。这种"分而治之"的策略使得 LLM 能够处理需要多步推理的复杂问题。

```
# CoT 推理示例
Question: Roger has 5 tennis balls. He buys 2 more cans of 3 tennis balls each. How many does he have now?

Thought: Roger started with 5 balls. 2 cans of 3 is 6. 5 + 6 = 11.
Answer: 11
```

CoT 的优势在于推理过程可解释，但致命缺陷是**无法获取外部信息**。LLM 只能基于训练数据和上下文窗口中的信息进行推理，无法查询实时数据或访问外部知识库。

**路径二：纯行动（Acting-only）**

以 Toolformer（Schick et al., 2023）和各种工具增强方法为代表，让 LLM 直接调用外部工具获取信息。这种方法的核心优势是能够获取真实、最新的信息。

```
# Act-only 示例
Question: Who was the president of the US when the Eiffel Tower was built?

Action 1: search("Eiffel Tower construction date")
Observation 1: The Eiffel Tower was built in 1889.

Action 2: search("US president in 1889")
Observation 2: Benjamin Harrison was the 23rd president, serving 1889-1893.

Answer: Benjamin Harrison
```

Act-only 的优势是信息准确，但缺陷是**缺乏推理深度**——LLM 可能选择了错误的工具，或者无法正确解读工具返回的结果。更重要的是，整个过程缺乏可解释性：我们不知道 LLM 为什么选择这些工具，也不知道它如何将工具返回的信息联系起来。

### 4.1.2 ReAct 的核心洞察

ReAct（Yao et al., 2022）的核心洞察是：**推理和行动不应该分离，而应该交替进行、互相增强**。

这种交替循环的动机来自认知科学的研究：人类在解决复杂问题时，往往是"边想边做"的——先思考当前情况，决定下一步行动，执行行动后观察结果，然后根据结果继续思考。这种"思考-行动-观察"的循环是人类问题解决的基本模式。

ReAct 将这种认知模式形式化为 LLM Agent 的工作范式：

1. **Thought（推理）**：LLM 分析当前状态，理解已经获取的信息，决定下一步该做什么
2. **Action（行动）**：LLM 选择一个工具并生成调用参数
3. **Observation（观察）**：工具执行的结果反馈给 LLM，为下一轮 Thought 提供新信息

这种交替进行的方式使得推理有了信息支撑（来自工具返回的真实数据），行动有了方向指导（来自 LLM 的推理分析）。

### 4.1.3 ReAct 的形式化描述

ReAct 的核心循环可以形式化为：

$$
\text{ReAct}(q) = \text{Loop}(\text{Thought}(s_t), \text{Action}(s_t), \text{Observation}(s_{t+1}))
$$

其中 $q$ 是用户问题，$s_t$ 是第 $t$ 步的状态。

更具体地，每一步的状态转移可以表示为：

$$
s_{t+1} = f(s_t, \text{Thought}_t, \text{Action}_t, \text{Observation}_t)
$$

其中 $f$ 是状态转移函数，它将当前状态、推理、行动和观察组合成新的状态。

状态 $s_t$ 可以进一步分解为：

$$
s_t = (q, h_1, h_2, \ldots, h_t)
$$

其中 $h_i = (\text{Thought}_i, \text{Action}_i, \text{Observation}_i)$ 是第 $i$ 步的历史记录。

ReAct 的终止条件是 LLM 生成最终答案（Final Answer），而不是继续选择工具：

 $$
 \text{Termination}: \text{Action}_t = \text{FinalAnswer}
 $$

ReAct 框架是推理与行动交替进行的经典 Agent 架构。下面的交互式演示展示了 ReAct 的完整工作流程：

<div data-component="ReActDemoV5"></div>

 ### 4.1.4 为什么推理与行动的结合是必要的

让我们通过一个具体的例子来理解为什么推理与行动的结合是必要的：

**问题**：2024年诺贝尔物理学奖得主在哪个大学工作？

**CoT-only 的表现**：
```
Thought: 2024年诺贝尔物理学奖...我不确定具体是谁获得了这个奖项。
Answer: 我不确定。
```

LLM 的训练数据截止到某个时间点，可能不包含 2024 年的信息。即使包含了，也可能不准确。

**Act-only 的表现**：
```
Action: search("2024 Nobel Prize Physics winner")
Observation: John Hopfield and Geoffrey Hinton won the 2024 Nobel Prize in Physics for their work on neural networks.

Action: search("John Hopfield university")
Observation: John Hopfield is a professor at Princeton University.

Action: search("Geoffrey Hinton university")
Observation: Geoffrey Hinton was a professor at the University of Toronto.
```

虽然获取了正确信息，但整个过程缺乏推理——我们不知道 LLM 为什么先搜索获奖者，再分别搜索他们的大学。

**ReAct 的表现**：
```
Thought: 要回答这个问题，我需要先知道 2024 年诺贝尔物理学奖得主是谁，然后找到他们的工作单位。
Action: search("2024 Nobel Prize Physics winner")
Observation: John Hopfield and Geoffrey Hinton won the 2024 Nobel Prize in Physics for their work on neural networks.

Thought: 有两个获奖者：John Hopfield 和 Geoffrey Hinton。我需要分别查找他们的工作单位。
Action: search("John Hopfield university affiliation")
Observation: John Hopfield is a professor at Princeton University.

Thought: John Hopfield 在普林斯顿大学。现在查找 Geoffrey Hinton。
Action: search("Geoffrey Hinton university affiliation")
Observation: Geoffrey Hinton was a professor at the University of Toronto until 2023, when he left to focus on AI safety.

Thought: 两位获奖者分别在普林斯顿大学和多伦多大学工作。
Answer: John Hopfield 在普林斯顿大学，Geoffrey Hinton 在多伦多大学。
```

ReAct 的优势显而易见：
1. **推理提供了方向指导**：LLM 先分析需要什么信息，再决定搜索什么
2. **行动提供了真实数据**：工具返回的信息确保了准确性
3. **推理解释了行动的理由**：每一步搜索都有明确的目的
4. **可解释性**：整个过程可以被人类理解和审查

---

## 4.2 论文核心思想

### 4.2.1 论文概述

ReAct 论文（Yao et al., 2022）的核心贡献是：

1. **提出 ReAct 范式**：将推理和行动统一在一个交替循环中
2. **实验证明有效性**：在多个基准测试上验证 ReAct 的优势
3. **分析可解释性**：展示 ReAct 如何提供可解释的推理路径

### 4.2.2 ReAct 的算法框架

ReAct 的算法框架可以描述为：

```python
def react_algorithm(question, tools, llm, max_steps=10):
    """
    ReAct 算法框架

    参数:
        question: 用户问题
        tools: 可用工具列表
        llm: 大语言模型
        max_steps: 最大步数

    返回:
        最终答案
    """
    # 初始化历史记录
    history = []

    for step in range(max_steps):
        # 构建 prompt，包含历史记录
        prompt = build_react_prompt(question, history, tools)

        # 调用 LLM 获取响应
        response = llm.generate(prompt)

        # 解析响应
        parsed = parse_react_output(response)

        # 检查是否生成最终答案
        if parsed["type"] == "final_answer":
            return parsed["answer"]

        # 执行工具调用
        observation = execute_tool(parsed["action"], parsed["action_input"])

        # 记录历史
        history.append({
            "thought": parsed["thought"],
            "action": parsed["action"],
            "action_input": parsed["action_input"],
            "observation": observation
        })

    # 达到最大步数，返回当前最佳答案
    return "达到最大步数限制，无法完成任务"
```

### 4.2.3 关键设计决策

**1. Thought 的位置**

ReAct 将 Thought 放在 Action 之前，而不是之后。这种设计有两个优势：
- Thought 可以帮助 LLM 选择正确的工具和参数
- Thought 可以帮助 LLM 理解工具返回的结果

**2. Observation 的格式**

Observation 是工具执行的原始结果，ReAct 不对 Observation 进行处理。这种设计的优势是简单直接，但劣势是 Observation 可能很长，消耗大量 token。

**3. 终止条件**

ReAct 使用 "Final Answer" 作为终止条件。当 LLM 判断已经收集到足够信息时，它会生成 "Final Answer" 而不是继续调用工具。

### 4.2.4 实验结果分析

ReAct 在多个基准测试上取得了优异的结果：

| 任务 | CoT | Act-only | ReAct |
|:---|:---:|:---:|:---:|
| HotpotQA (多跳问答) | 35.1% | 27.3% | **45.2%** |
| FEVER (事实验证) | 64.6% | 60.9% | **67.3%** |
| AlfWorld (交互式决策) | - | 73.5% | **85.3%** |
| WebShop (网页购物) | - | 56.2% | **68.4%** |

这些结果表明，ReAct 在需要多步推理和外部信息的任务上显著优于纯推理或纯行动的方法。

---

## 4.3 三种范式对比

### 4.3.1 详细对比表格

| 维度 | CoT-only | Act-only | ReAct |
|:---|:---|:---|:---|
| **核心思想** | 纯推理，分步思考 | 纯行动，调用工具 | 推理与行动交替 |
| **信息来源** | 训练数据 + 上下文 | 工具返回的真实数据 | 两者结合 |
| **推理深度** | 高（可解释） | 低（缺乏推理） | 高（可解释） |
| **幻觉率** | 高 | 低 | 低 |
| **Token 效率** | 高 | 中 | 低 |
| **适用场景** | 简单问答 | 信息检索 | 复杂任务 |
| **可解释性** | 高 | 低 | 高 |
| **错误恢复** | 低 | 中 | 高 |
| **工具依赖** | 无 | 强 | 中 |
| **实现复杂度** | 低 | 中 | 高 |

### 4.3.2 CoT-only 的深度分析

**优势**：
- 推理过程完全可解释
- 不依赖外部工具
- Token 使用效率高
- 适合简单推理任务

**劣势**：
- 无法获取外部信息
- 容易产生幻觉
- 无法验证推理结果
- 不适合需要实时信息的任务

**典型应用场景**：
- 数学推理
- 逻辑推理
- 简单常识问答
- 文本分析

### 4.3.3 Act-only 的深度分析

**优势**：
- 能够获取真实信息
- 幻觉率低
- 适合信息检索任务
- 实现相对简单

**劣势**：
- 缺乏推理深度
- 可能选择错误的工具
- 无法正确解读工具结果
- 可解释性差

**典型应用场景**：
- 事实查询
- 数据检索
- API 调用
- 简单的信息获取任务

### 4.3.4 ReAct 的深度分析

**优势**：
- 推理与行动互相增强
- 既可解释又准确
- 适合复杂任务
- 错误恢复能力强

**劣势**：
- Token 使用效率低
- 实现复杂度高
- 需要仔细的 prompt 设计
- 可能陷入循环

**典型应用场景**：
- 多跳问答
- 复杂研究任务
- 交互式决策
- 需要多步推理的信息检索

### 4.3.5 选择建议

选择哪种范式取决于具体任务需求：

```
if task_requires_reasoning and task_requires_external_info:
    use ReAct()
elif task_requires_reasoning and not task_requires_external_info:
    use CoT()
elif not task_requires_reasoning and task_requires_external_info:
    use Act_only()
  else:
      use simple_prompt()
  ```

不同推理策略对比如下，你可以使用下面的交互式工具根据任务需求选择最合适的策略：

<div data-component="ReasoningStrategySelectorV6"></div>

 ---

 ## 4.4 Thought/Action/Observation 详解

### 4.4.1 Thought 的作用与实现

Thought 是 ReAct 中最独特也最重要的部分。它的作用类似于人类的"内心独白"——帮助 LLM 组织思路、规划策略、评估进展。

**Thought 的核心功能**：

1. **状态理解**：分析当前已经获取的信息，理解任务进展
2. **策略制定**：决定下一步应该做什么，选择合适的工具
3. **进展评估**：判断是否已经收集到足够信息
4. **错误处理**：如果之前的行动失败，分析原因并调整策略

**Thought 的实现细节**：

```python
class ThoughtGenerator:
    """Thought 生成器"""

    def __init__(self, llm, max_thought_length=200):
        """
        初始化 Thought 生成器

        参数:
            llm: 大语言模型
            max_thought_length: 最大 Thought 长度
        """
        self.llm = llm
        self.max_thought_length = max_thought_length

    def generate_thought(self, question, history, tools):
        """
        生成 Thought

        参数:
            question: 用户问题
            history: 历史记录
            tools: 可用工具

        返回:
            Thought 文本
        """
        # 构建 prompt
        prompt = self._build_thought_prompt(question, history, tools)

        # 调用 LLM
        thought = self.llm.generate(prompt)

        # 截断过长的 Thought
        if len(thought) > self.max_thought_length:
            thought = thought[:self.max_thought_length] + "..."

        return thought

    def _build_thought_prompt(self, question, history, tools):
        """构建 Thought prompt"""
        prompt = f"任务: {question}\n\n"

        if history:
            prompt += "历史记录:\n"
            for i, record in enumerate(history, 1):
                prompt += f"步骤 {i}:\n"
                prompt += f"  思考: {record['thought']}\n"
                prompt += f"  行动: {record['action']}\n"
                prompt += f"  结果: {record['observation']}\n\n"

        prompt += "可用工具:\n"
        for tool in tools:
            prompt += f"- {tool['name']}: {tool['description']}\n"

        prompt += "\n请分析当前情况，决定下一步行动。"
        return prompt
```

**Thought 的质量标准**：

| 标准 | 描述 | 示例 |
|:---|:---|:---|
| **相关性** | Thought 应该与当前任务相关 | "我需要查找 X 的信息" |
| **具体性** | Thought 应该具体明确 | "我需要搜索 X 的出生日期" |
| **逻辑性** | Thought 应该符合逻辑 | "因为 X，所以我需要 Y" |
| **简洁性** | Thought 应该简洁明了 | 避免冗长的描述 |

### 4.4.2 Action 的选择与执行

Action 是 Agent 与外部世界交互的唯一方式。选择正确的 Action 是 ReAct 成功的关键。

**Action 选择的三个层次**：

1. **工具选择**：选择哪个工具
2. **参数生成**：为工具生成正确的参数
3. **参数验证**：验证参数是否有效

**工具选择策略**：

```python
class ActionSelector:
    """Action 选择器"""

    def __init__(self, tools, llm):
        """
        初始化 Action 选择器

        参数:
            tools: 可用工具列表
            llm: 大语言模型
        """
        self.tools = {tool['name']: tool for tool in tools}
        self.llm = llm

    def select_action(self, thought, history):
        """
        选择 Action

        参数:
            thought: 当前 Thought
            history: 历史记录

        返回:
            (action_name, action_input)
        """
        # 构建选择 prompt
        prompt = self._build_selection_prompt(thought, history)

        # 调用 LLM
        response = self.llm.generate(prompt)

        # 解析响应
        action_name, action_input = self._parse_action(response)

        # 验证工具是否存在
        if action_name not in self.tools:
            raise ValueError(f"未知工具: {action_name}")

        # 验证参数
        self._validate_input(action_name, action_input)

        return action_name, action_input

    def _build_selection_prompt(self, thought, history):
        """构建选择 prompt"""
        prompt = "基于以下思考，选择一个工具并生成参数：\n\n"
        prompt += f"思考: {thought}\n\n"
        prompt += "可用工具:\n"
        for name, tool in self.tools.items():
            prompt += f"- {name}: {tool['description']}\n"
            prompt += f"  参数: {tool['parameters']}\n\n"
        prompt += "请以以下格式回答：\n"
        prompt += "Action: 工具名称\n"
        prompt += "Action Input: 参数\n"
        return prompt

    def _parse_action(self, response):
        """解析 Action"""
        lines = response.strip().split('\n')
        action_name = None
        action_input = None

        for line in lines:
            if line.startswith('Action:'):
                action_name = line.split(':', 1)[1].strip()
            elif line.startswith('Action Input:'):
                action_input = line.split(':', 1)[1].strip()

        if not action_name or not action_input:
            raise ValueError("无法解析 Action")

        return action_name, action_input

    def _validate_input(self, action_name, action_input):
        """验证参数"""
        tool = self.tools[action_name]
        # 这里可以添加更复杂的参数验证逻辑
        # 例如：检查必填参数、类型验证等
        pass
```

**Action 执行的错误处理**：

```python
class ActionExecutor:
    """Action 执行器"""

    def __init__(self, tools, timeout=30):
        """
        初始化 Action 执行器

        参数:
            tools: 工具实现字典
            timeout: 超时时间（秒）
        """
        self.tools = tools
        self.timeout = timeout

    def execute(self, action_name, action_input):
        """
        执行 Action

        参数:
            action_name: 工具名称
            action_input: 工具参数

        返回:
            执行结果
        """
        # 检查工具是否存在
        if action_name not in self.tools:
            return f"错误: 工具 '{action_name}' 不存在"

        # 获取工具实现
        tool = self.tools[action_name]

        # 执行工具
        try:
            result = tool.execute(action_input, timeout=self.timeout)
            return result
        except TimeoutError:
            return f"错误: 工具 '{action_name}' 执行超时"
        except Exception as e:
            return f"错误: 工具执行失败 - {str(e)}"
```

### 4.4.3 Observation 的处理

Observation 是工具执行的结果，为下一轮 Thought 提供新信息。Observation 的处理质量直接影响 ReAct 的效果。

**Observation 处理的挑战**：

1. **长度问题**：Observation 可能很长，消耗大量 token
2. **噪声问题**：Observation 可能包含无关信息
3. **格式问题**：Observation 的格式可能不一致

**Observation 处理策略**：

```python
class ObservationProcessor:
    """Observation 处理器"""

    def __init__(self, max_length=500, llm=None):
        """
        初始化 Observation 处理器

        参数:
            max_length: 最大长度
            llm: 用于摘要的 LLM（可选）
        """
        self.max_length = max_length
        self.llm = llm

    def process(self, observation, context=""):
        """
        处理 Observation

        参数:
            observation: 原始 Observation
            context: 上下文信息

        返回:
            处理后的 Observation
        """
        # 如果 Observation 已经足够短，直接返回
        if len(observation) <= self.max_length:
            return observation

        # 策略1: 截断
        truncated = self._truncate(observation)

        # 策略2: 摘要（如果 LLM 可用）
        if self.llm:
            summarized = self._summarize(observation, context)
            # 选择较短的结果
            if len(summarized) < len(truncated):
                return summarized

        return truncated

    def _truncate(self, observation):
        """截断 Observation"""
        # 尝试在句子边界截断
        truncated = observation[:self.max_length]
        last_period = truncated.rfind('。')
        last_space = truncated.rfind(' ')

        if last_period > self.max_length * 0.8:
            return truncated[:last_period + 1]
        elif last_space > self.max_length * 0.8:
            return truncated[:last_space] + "..."
        else:
            return truncated + "..."

    def _summarize(self, observation, context):
        """使用 LLM 摘要 Observation"""
        prompt = f"请将以下信息压缩为简洁的摘要，保留关键信息：\n\n"
        prompt += f"任务: {context}\n" if context else ""
        prompt += f"信息: {observation}\n\n"
        prompt += "摘要:"

        return self.llm.generate(prompt)
```

### 4.4.4 三者的协作模式

Thought、Action、Observation 三者形成一个紧密的协作循环：

```
Question: 谁是 iPhone 的创始人？他是什么时候出生的？

Thought 1: 要回答这个问题，我需要先知道 iPhone 的创始人是谁，然后查找他的出生日期。
Action 1: search("iPhone founder")
Observation 1: iPhone 是由苹果公司推出的，苹果公司的创始人是 Steve Jobs。

Thought 2: iPhone 的创始人是 Steve Jobs。现在我需要查找他的出生日期。
Action 2: search("Steve Jobs birth date")
Observation 2: Steve Jobs 于 1955 年 2 月 24 日出生。

Thought 3: 我已经收集到所有需要的信息。iPhone 的创始人是 Steve Jobs，他出生于 1955 年 2 月 24 日。
Answer: iPhone 的创始人是 Steve Jobs，他出生于 1955 年 2 月 24 日。
```

在这个例子中：
- **Thought 1** 分析任务，制定搜索策略
- **Action 1** 执行搜索，获取 iPhone 创始人信息
- **Observation 1** 提供搜索结果
- **Thought 2** 分析结果，确定下一步搜索
- **Action 2** 执行搜索，获取出生日期
- **Observation 2** 提供出生日期
- **Thought 3** 综合所有信息，得出最终答案

---

## 4.5 手动实现完整 ReAct Agent

### 4.5.1 整体架构设计

我们将实现一个完整的 ReAct Agent，包含以下组件：

1. **ReActAgent**：主控制类，协调整个流程
2. **ToolManager**：工具管理器，负责工具注册和调用
3. **PromptBuilder**：Prompt 构建器，负责构建 LLM 的输入
4. **OutputParser**：输出解析器，负责解析 LLM 的输出
5. **MemoryManager**：记忆管理器，负责维护历史记录

```python
import json
import re
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

class ActionType(Enum):
    """行动类型枚举"""
    THOUGHT = "thought"
    ACTION = "action"
    FINAL_ANSWER = "final_answer"

@dataclass
class StepRecord:
    """步骤记录"""
    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class AgentConfig:
    """Agent 配置"""
    max_steps: int = 10
    max_retries: int = 3
    timeout: int = 30
    verbose: bool = True
```

### 4.5.2 工具管理器实现

```python
class Tool:
    """工具类"""

    def __init__(self, name: str, description: str, func: Callable,
                 parameters: Dict[str, str] = None):
        """
        初始化工具

        参数:
            name: 工具名称
            description: 工具描述
            func: 工具函数
            parameters: 参数描述
        """
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {}

    def execute(self, input_text: str, timeout: int = 30) -> str:
        """
        执行工具

        参数:
            input_text: 输入文本
            timeout: 超时时间

        返回:
            执行结果
        """
        try:
            result = self.func(input_text)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {str(e)}"

class ToolManager:
    """工具管理器"""

    def __init__(self):
        """初始化工具管理器"""
        self.tools: Dict[str, Tool] = {}

    def register(self, name: str, description: str, func: Callable,
                 parameters: Dict[str, str] = None):
        """
        注册工具

        参数:
            name: 工具名称
            description: 工具描述
            func: 工具函数
            parameters: 参数描述
        """
        self.tools[name] = Tool(name, description, func, parameters)

    def get_tool(self, name: str) -> Optional[Tool]:
        """
        获取工具

        参数:
            name: 工具名称

        返回:
            工具对象，如果不存在则返回 None
        """
        return self.tools.get(name)

    def get_tools_description(self) -> str:
        """
        获取所有工具的描述

        返回:
            工具描述文本
        """
        descriptions = []
        for name, tool in self.tools.items():
            desc = f"- {name}: {tool.description}"
            if tool.parameters:
                desc += f"\n  参数: {tool.parameters}"
            descriptions.append(desc)
        return "\n".join(descriptions)

    def get_tool_names(self) -> List[str]:
        """
        获取所有工具名称

        返回:
            工具名称列表
        """
        return list(self.tools.keys())
```

### 4.5.3 Prompt 构建器实现

```python
class PromptBuilder:
    """Prompt 构建器"""

    def __init__(self, tool_manager: ToolManager):
        """
        初始化 Prompt 构建器

        参数:
            tool_manager: 工具管理器
        """
        self.tool_manager = tool_manager

    def build_system_prompt(self) -> str:
        """
        构建系统 prompt

        返回:
            系统 prompt
        """
        return """你是一个能够使用工具解决复杂问题的 AI 助手。

你需要通过思考（Thought）、行动（Action）和观察（Observation）的循环来解决问题。

请严格遵循以下格式：
Thought: [你的思考过程]
Action: [工具名称]
Action Input: [工具输入]

当你得到最终答案时：
Thought: [你的最终思考]
Final Answer: [最终答案]

重要规则：
1. 每次只能调用一个工具
2. 工具名称必须在可用工具列表中
3. 如果工具执行失败，分析原因并尝试其他方法
4. 在收集到足够信息后，给出最终答案
"""

    def build_agent_prompt(self, question: str, history: List[StepRecord],
                          step_number: int) -> str:
        """
        构建 Agent prompt

        参数:
            question: 用户问题
            history: 历史记录
            step_number: 当前步数

        返回:
            Agent prompt
        """
        prompt = f"问题: {question}\n\n"

        # 添加工具描述
        prompt += "可用工具:\n"
        prompt += self.tool_manager.get_tools_description()
        prompt += "\n\n"

        # 添加历史记录
        if history:
            prompt += "历史记录:\n"
            for record in history:
                prompt += f"步骤 {record.step_number}:\n"
                prompt += f"  Thought: {record.thought}\n"
                if record.action:
                    prompt += f"  Action: {record.action}\n"
                if record.action_input:
                    prompt += f"  Action Input: {record.action_input}\n"
                if record.observation:
                    prompt += f"  Observation: {record.observation}\n"
                prompt += "\n"

        # 添加当前步数提示
        prompt += f"当前是第 {step_number} 步。\n\n"
        prompt += "请按照格式回答：\n"
        prompt += "Thought: [你的思考]\n"

        return prompt

    def build_final_answer_prompt(self, question: str,
                                  history: List[StepRecord]) -> str:
        """
        构建最终答案 prompt

        参数:
            question: 用户问题
            history: 历史记录

        返回:
            最终答案 prompt
        """
        prompt = f"问题: {question}\n\n"

        prompt += "已收集的信息:\n"
        for record in history:
            if record.observation:
                prompt += f"- {record.observation}\n"

        prompt += "\n请根据以上信息，给出最终答案。\n"
        prompt += "格式：Final Answer: [你的答案]"

        return prompt
```

### 4.5.4 输出解析器实现

```python
class OutputParser:
    """输出解析器"""

    # 正则表达式模式
    THOUGHT_PATTERN = re.compile(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', re.DOTALL)
    ACTION_PATTERN = re.compile(r'Action:\s*(.+?)(?=Action Input:|$)', re.DOTALL)
    ACTION_INPUT_PATTERN = re.compile(r'Action Input:\s*(.+?)(?=Observation:|Thought:|Final Answer:|$)', re.DOTALL)
    FINAL_ANSWER_PATTERN = re.compile(r'Final Answer:\s*(.+?)$', re.DOTALL)

    @classmethod
    def parse(cls, output: str) -> Dict[str, Any]:
        """
        解析 LLM 输出

        参数:
            output: LLM 输出文本

        返回:
            解析结果字典
        """
        result = {
            "type": ActionType.THOUGHT,
            "thought": "",
            "action": None,
            "action_input": None,
            "final_answer": None
        }

        # 提取 Thought
        thought_match = cls.THOUGHT_PATTERN.search(output)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # 检查是否是最终答案
        final_answer_match = cls.FINAL_ANSWER_PATTERN.search(output)
        if final_answer_match:
            result["type"] = ActionType.FINAL_ANSWER
            result["final_answer"] = final_answer_match.group(1).strip()
            return result

        # 提取 Action
        action_match = cls.ACTION_PATTERN.search(output)
        if action_match:
            result["type"] = ActionType.ACTION
            result["action"] = action_match.group(1).strip()

        # 提取 Action Input
        action_input_match = cls.ACTION_INPUT_PATTERN.search(output)
        if action_input_match:
            result["action_input"] = action_input_match.group(1).strip()

        return result

    @classmethod
    def validate(cls, parsed: Dict[str, Any], tool_manager: ToolManager) -> bool:
        """
        验证解析结果

        参数:
            parsed: 解析结果
            tool_manager: 工具管理器

        返回:
            是否有效
        """
        if parsed["type"] == ActionType.THOUGHT:
            return True

        if parsed["type"] == ActionType.FINAL_ANSWER:
            return bool(parsed["final_answer"])

        if parsed["type"] == ActionType.ACTION:
            # 验证工具是否存在
            if parsed["action"] not in tool_manager.get_tool_names():
                return False
            # 验证输入是否存在
            if not parsed["action_input"]:
                return False
            return True

        return False
```

### 4.5.5 记忆管理器实现

```python
class MemoryManager:
    """记忆管理器"""

    def __init__(self, max_history: int = 20):
        """
        初始化记忆管理器

        参数:
            max_history: 最大历史记录数
        """
        self.max_history = max_history
        self.history: List[StepRecord] = []
        self.summary: Optional[str] = None

    def add_step(self, record: StepRecord):
        """
        添加步骤记录

        参数:
            record: 步骤记录
        """
        self.history.append(record)

        # 如果历史记录过长，进行压缩
        if len(self.history) > self.max_history:
            self._compress_history()

    def get_history(self) -> List[StepRecord]:
        """
        获取历史记录

        返回:
            历史记录列表
        """
        return self.history.copy()

    def get_context(self) -> str:
        """
        获取上下文信息

        返回:
            上下文文本
        """
        context = ""
        if self.summary:
            context += f"历史摘要: {self.summary}\n\n"

        for record in self.history:
            context += f"步骤 {record.step_number}:\n"
            context += f"  Thought: {record.thought}\n"
            if record.observation:
                context += f"  Observation: {record.observation}\n"
            context += "\n"

        return context

    def _compress_history(self):
        """压缩历史记录"""
        # 保留最近的一半记录
        keep_count = len(self.history) // 2
        old_records = self.history[:keep_count]
        self.history = self.history[keep_count:]

        # 生成摘要
        summary_parts = []
        for record in old_records:
            if record.observation:
                summary_parts.append(record.observation)

        if summary_parts:
            self.summary = "之前获取的信息: " + "; ".join(summary_parts)

    def clear(self):
        """清空历史记录"""
        self.history = []
        self.summary = None
```

### 4.5.6 主 Agent 实现

```python
class ReActAgent:
    """ReAct Agent 主类"""

    def __init__(self, llm_func: Callable, config: AgentConfig = None):
        """
        初始化 ReAct Agent

        参数:
            llm_func: LLM 调用函数
            config: 配置
        """
        self.llm_func = llm_func
        self.config = config or AgentConfig()
        self.tool_manager = ToolManager()
        self.prompt_builder = PromptBuilder(self.tool_manager)
        self.memory = MemoryManager()

    def register_tool(self, name: str, description: str, func: Callable,
                     parameters: Dict[str, str] = None):
        """注册工具"""
        self.tool_manager.register(name, description, func, parameters)

    def run(self, question: str) -> str:
        """
        运行 Agent

        参数:
            question: 用户问题

        返回:
            最终答案
        """
        self.memory.clear()
        step_number = 1

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"问题: {question}")
            print(f"{'='*60}")

        for step in range(self.config.max_steps):
            # 构建 prompt
            prompt = self.prompt_builder.build_agent_prompt(
                question, self.memory.get_history(), step_number
            )

            # 调用 LLM
            llm_output = self._call_llm(prompt)

            # 解析输出
            parsed = OutputParser.parse(llm_output)

            # 验证输出
            if not OutputParser.validate(parsed, self.tool_manager):
                # 如果验证失败，重试
                if self.config.verbose:
                    print(f"[警告] 输出验证失败，重试中...")
                continue

            # 处理 Thought
            if self.config.verbose:
                print(f"\n[步骤 {step_number}]")
                print(f"Thought: {parsed['thought']}")

            # 如果是最终答案
            if parsed["type"] == ActionType.FINAL_ANSWER:
                if self.config.verbose:
                    print(f"Final Answer: {parsed['final_answer']}")
                    print(f"{'='*60}")
                return parsed["final_answer"]

            # 执行 Action
            if parsed["type"] == ActionType.ACTION:
                tool = self.tool_manager.get_tool(parsed["action"])
                observation = tool.execute(parsed["action_input"], self.config.timeout)

                if self.config.verbose:
                    print(f"Action: {parsed['action']}")
                    print(f"Action Input: {parsed['action_input']}")
                    print(f"Observation: {observation}")

                # 记录步骤
                record = StepRecord(
                    step_number=step_number,
                    thought=parsed["thought"],
                    action=parsed["action"],
                    action_input=parsed["action_input"],
                    observation=observation
                )
                self.memory.add_step(record)

                step_number += 1

        # 达到最大步数
        return self._generate_final_answer(question)

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        try:
            return self.llm_func(prompt)
        except Exception as e:
            return f"Thought: 我需要重新思考这个问题。\nAction: search\nAction Input: {prompt}"

    def _generate_final_answer(self, question: str) -> str:
        """生成最终答案"""
        prompt = self.prompt_builder.build_final_answer_prompt(
            question, self.memory.get_history()
        )
        return self._call_llm(prompt)
```

### 4.5.7 使用示例

```python
# 示例工具实现
def search_tool(query: str) -> str:
    """搜索工具"""
    # 这里可以接入真实的搜索 API
    # 为了演示，返回模拟结果
    mock_results = {
        "python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年发布。",
        "react": "React 是一个用于构建用户界面的 JavaScript 库，由 Facebook 开发。",
        "reAct": "ReAct 是一种将推理和行动结合的 LLM Agent 范式。"
    }

    for key, value in mock_results.items():
        if key.lower() in query.lower():
            return value

    return f"未找到关于 '{query}' 的信息"

def calculator_tool(expression: str) -> str:
    """计算器工具"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 创建 Agent
def llm_function(prompt: str) -> str:
    """模拟 LLM 函数"""
    # 在实际应用中，这里应该调用真实的 LLM API
    # 为了演示，返回模拟响应
    if "Python" in prompt and "Action" not in prompt:
        return "Thought: 用户想了解 Python 的创始人。我需要搜索相关信息。\nAction: search\nAction Input: Python founder"
    elif "Guido" in prompt:
        return "Thought: 我已经知道 Python 的创始人是 Guido van Rossum。现在需要查找他的出生日期。\nAction: search\nAction Input: Guido van Rossum birth date"
    elif "1956" in prompt:
        return "Thought: 我已经收集到所有信息。Python 的创始人是 Guido van Rossum，出生于 1956 年。\nFinal Answer: Python 的创始人是 Guido van Rossum，他出生于 1956 年 1 月 31 日。"
    else:
        return "Thought: 我需要搜索更多信息。\nAction: search\nAction Input: Python programming language"

# 使用示例
agent = ReActAgent(llm_function)
agent.register_tool("search", "搜索工具，用于查找信息", search_tool)
agent.register_tool("calculator", "计算器工具，用于数学计算", calculator_tool)

# 运行
answer = agent.run("Python 的创始人是谁？他是什么时候出生的？")
 print(f"\n最终答案: {answer}")
 ```

工具选择是 ReAct 循环中的关键决策环节，错误的选择会导致任务失败。下面的交互式演示展示了工具选择的完整过程：

<div data-component="ToolSelectionDemoV5"></div>

 ---

 ## 4.6 LangGraph 集成

### 4.6.1 LangGraph 简介

LangGraph 是 LangChain 生态系统中的一个库，专门用于构建有状态的、多步骤的 Agent 应用。它基于图（Graph）的概念，允许我们定义 Agent 的工作流程。

LangGraph 的核心优势：
1. **状态管理**：内置状态管理机制
2. **流程控制**：支持条件分支和循环
3. **可检查点**：支持断点续传和调试
4. **流式支持**：支持流式输出
5. **人机交互**：支持人类介入

### 4.6.2 基础 ReAct Agent 实现

```python
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

# 定义状态
class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_action: str

# 定义工具
@tool
def search(query: str) -> str:
    """搜索工具，用于查找信息"""
    # 这里可以接入真实的搜索 API
    mock_results = {
        "python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年发布。",
        "react": "React 是一个用于构建用户界面的 JavaScript 库，由 Facebook 开发。",
        "langchain": "LangChain 是一个用于构建 LLM 应用的框架。"
    }

    for key, value in mock_results.items():
        if key.lower() in query.lower():
            return value

    return f"未找到关于 '{query}' 的信息"

@tool
def calculator(expression: str) -> str:
    """计算器工具，用于数学计算"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 创建工具节点
tools = [search, calculator]
tool_node = ToolNode(tools)
```

### 4.6.3 定义 Agent 逻辑

```python
# 定义 Agent 节点
from langchain_openai import ChatOpenAI

def create_react_agent(model_name: str = "gpt-4"):
    """
    创建 ReAct Agent

    参数:
        model_name: 模型名称

    返回:
        编译后的 Agent 图
    """
    # 创建模型
    model = ChatOpenAI(model=model_name)
    model = model.bind_tools(tools)

    # 定义 Agent 节点
    def agent(state: AgentState):
        """
        Agent 节点：调用 LLM 决定下一步行动
        """
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}

    # 定义条件路由
    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        """
        决定是否继续执行

        如果 LLM 决定调用工具，则继续；否则结束。
        """
        messages = state["messages"]
        last_message = messages[-1]

        # 如果 LLM 决定调用工具
        if last_message.tool_calls:
            return "tools"
        # 否则结束
        return "__end__"

    # 创建图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)

    # 添加边
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    workflow.add_edge("tools", "agent")

    # 编译图
    return workflow.compile()
```

### 4.6.4 增强版 ReAct Agent（带记忆和检查点）

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage

class EnhancedReActAgent:
    """增强版 ReAct Agent"""

    def __init__(self, model_name: str = "gpt-4"):
        """
        初始化增强版 Agent

        参数:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.model = ChatOpenAI(model=model_name)
        self.model = self.model.bind_tools(tools)

        # 创建检查点保存器
        self.memory = SqliteSaver.from_conn_string(":memory:")

        # 创建 Agent
        self.agent = self._create_agent()

    def _create_agent(self):
        """创建 Agent 图"""
        # 定义状态
        class State(TypedDict):
            messages: Annotated[Sequence[BaseMessage], operator.add]

        # 定义节点
        def agent_node(state: State):
            """Agent 节点"""
            messages = state["messages"]
            response = self.model.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: State) -> Literal["tools", "__end__"]:
            """条件路由"""
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                return "tools"
            return "__end__"

        # 创建图
        workflow = StateGraph(State)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "__end__": END}
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.memory)

    def run(self, question: str, thread_id: str = "default") -> str:
        """
        运行 Agent

        参数:
            question: 用户问题
            thread_id: 线程 ID（用于保持对话状态）

        返回:
            最终答案
        """
        # 创建配置
        config = {"configurable": {"thread_id": thread_id}}

        # 创建初始消息
        initial_state = {"messages": [HumanMessage(content=question)]}

        # 运行 Agent
        result = self.agent.invoke(initial_state, config)

        # 提取最终答案
        messages = result["messages"]
        final_message = messages[-1]

        return final_message.content

    def stream(self, question: str, thread_id: str = "default"):
        """
        流式运行 Agent

        参数:
            question: 用户问题
            thread_id: 线程 ID

        返回:
            生成器，产生每一步的输出
        """
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {"messages": [HumanMessage(content=question)]}

        for event in self.agent.stream(initial_state, config):
            yield event

    def get_conversation_history(self, thread_id: str) -> list:
        """
        获取对话历史

        参数:
            thread_id: 线程 ID

        返回:
            对话历史列表
        """
        config = {"configurable": {"thread_id": thread_id}}
        history = []

        for message in self.agent.get_state(config).values["messages"]:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                if message.content:
                    history.append({"role": "assistant", "content": message.content})
                if message.tool_calls:
                    history.append({"role": "assistant", "tool_calls": message.tool_calls})
            elif isinstance(message, ToolMessage):
                history.append({"role": "tool", "content": message.content})

        return history
```

### 4.6.5 使用示例

```python
# 创建 Agent
agent = EnhancedReActAgent(model_name="gpt-4")

# 运行 Agent
answer = agent.run("Python 的创始人是谁？")
print(f"答案: {answer}")

# 流式输出
print("\n流式输出:")
for event in agent.stream("什么是机器学习？"):
    print(event)

# 获取对话历史
history = agent.get_conversation_history("default")
print(f"\n对话历史: {history}")

# 带检查点的对话（可以断点续传）
answer1 = agent.run("你好，我想了解 Python", thread_id="session1")
answer2 = agent.run("它的创始人是谁？", thread_id="session1")  # 会记住上文
```

### 4.6.6 自定义 ReAct 逻辑

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator

class CustomReActState(TypedDict):
    """自定义 ReAct 状态"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    thoughts: list
    actions: list
    observations: list
    step_count: int

def create_custom_react_agent():
    """创建自定义 ReAct Agent"""

    model = ChatOpenAI(model="gpt-4")
    model = model.bind_tools(tools)

    def think_node(state: CustomReActState):
        """思考节点：分析当前状态，决定下一步"""
        messages = state["messages"]
        step_count = state["step_count"] + 1

        # 构建思考 prompt
        think_prompt = f"当前是第 {step_count} 步。请分析当前情况并决定下一步行动。"

        response = model.invoke([HumanMessage(content=think_prompt)])

        return {
            "messages": [response],
            "thoughts": [response.content],
            "step_count": step_count
        }

    def act_node(state: CustomReActState):
        """行动节点：执行工具调用"""
        messages = state["messages"]
        last_message = messages[-1]

        if last_message.tool_calls:
            # 执行工具调用
            tool_results = []
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                # 找到对应的工具
                for t in tools:
                    if t.name == tool_name:
                        result = t.invoke(tool_args)
                        tool_results.append(ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"]
                        ))
                        break

            return {
                "messages": tool_results,
                "actions": [last_message.tool_calls],
                "observations": [str(r.content) for r in tool_results]
            }

        return {"actions": [], "observations": []}

    def should_continue(state: CustomReActState):
        """条件路由"""
        messages = state["messages"]
        last_message = messages[-1]

        # 如果有工具调用，继续执行
        if last_message.tool_calls:
            return "act"

        # 如果步数过多，结束
        if state["step_count"] >= 5:
            return END

        # 否则继续思考
        return "think"

    # 创建图
    workflow = StateGraph(CustomReActState)
    workflow.add_node("think", think_node)
    workflow.add_node("act", act_node)
    workflow.set_entry_point("think")
    workflow.add_conditional_edges("think", should_continue, {"act": "act", END: END})
    workflow.add_edge("act", "think")

    return workflow.compile()

# 使用示例
custom_agent = create_custom_react_agent()
result = custom_agent.invoke({
    "messages": [HumanMessage(content="Python 的创始人是谁？")],
    "thoughts": [],
    "actions": [],
    "observations": [],
    "step_count": 0
})
```

---

## 4.7 性能分析

### 4.7.1 Token 使用效率

ReAct 的主要性能瓶颈是 Token 使用效率。每一步都需要发送完整的历史记录给 LLM，导致 Token 使用量随步数线性增长。

**Token 使用量分析**：

假设：
- 系统 prompt: $T_{sys}$ tokens
- 每步 Thought: $T_{thought}$ tokens
- 每步 Action: $T_{action}$ tokens
- 每步 Observation: $T_{obs}$ tokens

则 $n$ 步后的总 Token 使用量为：

$$
T_{total} = T_{sys} + n \times (T_{thought} + T_{action} + T_{obs}) + \sum_{i=1}^{n} \sum_{j=1}^{i} (T_{thought_j} + T_{action_j} + T_{obs_j})
$$

简化后：

$$
T_{total} \approx T_{sys} + n \times T_{step} + \frac{n(n+1)}{2} \times T_{avg}
$$

其中 $T_{step}$ 是每步固定开销，$T_{avg}$ 是平均每步的历史记录长度。

**Token 优化策略**：

| 策略 | 描述 | 优势 | 劣势 |
|:---|:---|:---|:---|
| **历史压缩** | 定期压缩历史记录 | 减少 Token | 可能丢失信息 |
| **滑动窗口** | 只保留最近 $k$ 步 | 简单有效 | 无法回顾早期信息 |
| **摘要提取** | 使用 LLM 生成摘要 | 保留关键信息 | 增加延迟 |
| **重要性筛选** | 只保留重要步骤 | 精准控制 | 实现复杂 |

### 4.7.2 延迟分析

ReAct 的延迟主要来自：

1. **LLM 调用延迟**：每步都需要调用 LLM
2. **工具执行延迟**：工具可能需要较长时间
3. **网络延迟**：API 调用的网络开销

**延迟优化策略**：

```python
class OptimizedReActAgent:
    """优化版 ReAct Agent"""

    def __init__(self, llm_func, tools, config):
        self.llm_func = llm_func
        self.tools = tools
        self.config = config

        # 预编译正则表达式
        self.thought_pattern = re.compile(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', re.DOTALL)
        self.action_pattern = re.compile(r'Action:\s*(.+?)(?=Action Input:|$)', re.DOTALL)

        # 缓存常用工具描述
        self.tools_description = self._build_tools_description()

    def run_optimized(self, question: str) -> str:
        """优化版运行"""
        history = []
        step_number = 1

        while step_number <= self.config.max_steps:
            # 构建 prompt（使用缓存的工具描述）
            prompt = self._build_prompt_fast(question, history, step_number)

            # 调用 LLM
            llm_output = self.llm_func(prompt)

            # 快速解析
            parsed = self._fast_parse(llm_output)

            if parsed["type"] == "final_answer":
                return parsed["final_answer"]

            if parsed["type"] == "action":
                # 并行执行工具（如果支持）
                observation = self._execute_tool_fast(
                    parsed["action"],
                    parsed["action_input"]
                )

                history.append({
                    "thought": parsed["thought"],
                    "action": parsed["action"],
                    "observation": observation
                })

                step_number += 1

        return "达到最大步数限制"

    def _build_prompt_fast(self, question, history, step_number):
        """快速构建 prompt"""
        # 使用字符串拼接而非格式化
        parts = [f"问题: {question}\n\n"]
        parts.append("可用工具:\n")
        parts.append(self.tools_description)  # 使用缓存
        parts.append(f"\n\n当前是第 {step_number} 步。\n\n")

        if history:
            parts.append("历史记录:\n")
            for record in history[-3:]:  # 只保留最近3步
                parts.append(f"Thought: {record['thought']}\n")
                parts.append(f"Observation: {record['observation']}\n\n")

        parts.append("请按照格式回答：\nThought: [你的思考]\n")
        return "".join(parts)

    def _fast_parse(self, output):
        """快速解析"""
        result = {"type": "thought", "thought": "", "action": None, "action_input": None}

        thought_match = self.thought_pattern.search(output)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        if "Final Answer:" in output:
            result["type"] = "final_answer"
            result["final_answer"] = output.split("Final Answer:")[-1].strip()
            return result

        action_match = self.action_pattern.search(output)
        if action_match:
            result["type"] = "action"
            result["action"] = action_match.group(1).strip()
            result["action_input"] = output.split("Action Input:")[-1].strip()

        return result

    def _execute_tool_fast(self, action_name, action_input):
        """快速执行工具"""
        tool = self.tools.get(action_name)
        if tool:
            return tool.execute(action_input)
        return f"工具 {action_name} 不存在"
```

### 4.7.3 准确率分析

ReAct 的准确率受多个因素影响：

1. **Thought 质量**：高质量的 Thought 能引导正确的行动
2. **工具选择**：选择正确的工具是成功的关键
3. **参数生成**：为工具生成正确的参数
4. **Observation 理解**：正确理解工具返回的结果

**准确率提升策略**：

```python
class AccuracyOptimizer:
    """准确率优化器"""

    def __init__(self, llm_func, tools):
        self.llm_func = llm_func
        self.tools = tools

    def self_reflection(self, question, history, current_thought):
        """
        自我反思：评估当前进展

        参数:
            question: 用户问题
            history: 历史记录
            current_thought: 当前 Thought

        返回:
            反思结果
        """
        prompt = f"""请评估当前的解题进展：

问题: {question}

历史记录:
{self._format_history(history)}

当前思考: {current_thought}

请回答：
1. 当前进展如何？
2. 是否偏离了主题？
3. 是否需要调整策略？
4. 下一步应该做什么？

反思:"""

        return self.llm_func(prompt)

    def multi_path_explore(self, question, current_state, num_paths=3):
        """
        多路径探索：尝试多种可能的行动

        参数:
            question: 用户问题
            current_state: 当前状态
            num_paths: 探索路径数

        返回:
            最佳路径
        """
        paths = []

        for i in range(num_paths):
            prompt = f"""基于当前状态，提出一种可能的下一步行动：

问题: {question}
当前状态: {current_state}

请提出第 {i+1} 种可能的行动方案：
Thought: [你的思考]
Action: [工具名称]
Action Input: [参数]"""

            path = self.llm_func(prompt)
            paths.append(path)

        # 选择最佳路径
        best_path = self._select_best_path(question, paths)
        return best_path

    def _select_best_path(self, question, paths):
        """选择最佳路径"""
        prompt = f"""请评估以下几种行动方案，选择最佳的一个：

问题: {question}

方案:
{chr(10).join([f"方案 {i+1}: {path}" for i, path in enumerate(paths)])}

请选择最佳方案（只输出方案编号）："""

        result = self.llm_func(prompt)

        # 解析选择
        try:
            choice = int(result.strip()) - 1
            if 0 <= choice < len(paths):
                return paths[choice]
        except ValueError:
            pass

        return paths[0]  # 默认选择第一个

    def _format_history(self, history):
        """格式化历史记录"""
        if not history:
            return "无"

        parts = []
        for i, record in enumerate(history, 1):
            parts.append(f"步骤 {i}:")
            parts.append(f"  Thought: {record.get('thought', 'N/A')}")
            parts.append(f"  Observation: {record.get('observation', 'N/A')}")
        return "\n".join(parts)
```

---

## 4.8 局限性与改进

### 4.8.1 ReAct 的主要局限性

**1. Token 效率低**

ReAct 需要发送完整的历史记录，导致 Token 使用量随步数线性增长。对于需要多步推理的复杂任务，这可能导致：
- API 成本显著增加
- 上下文窗口溢出
- 响应延迟增加

**2. 错误传播**

如果早期步骤出现错误，后续步骤会基于错误信息继续推理，导致错误累积：

```
Thought 1: X 是 Y。（错误）
Action 1: search("Y 的信息")
Observation 1: Y 的相关信息...

Thought 2: 基于 X 是 Y，我需要查找...（基于错误假设）
```

**3. 无全局规划**

ReAct 是"边想边做"的模式，缺乏全局规划能力。对于复杂任务，可能导致：
- 重复搜索相同信息
- 忽略重要信息
- 无法识别任务依赖关系

**4. 循环风险**

LLM 可能陷入重复循环，反复执行相同的行动：

```
Thought: 我需要查找 X 的信息
Action: search("X")
Observation: 找不到 X 的信息
Thought: 我需要查找 X 的信息
Action: search("X")
...
```

**5. 工具依赖**

ReAct 高度依赖工具的可用性和准确性。如果工具不可用或返回错误信息，整个流程会失败。

### 4.8.2 ReAct+：压缩历史记录

ReAct+ 通过压缩历史记录来提高 Token 效率：

```python
class ReActPlus:
    """ReAct+：压缩历史记录"""

    def __init__(self, llm_func, tools, max_history=5):
        self.llm_func = llm_func
        self.tools = tools
        self.max_history = max_history

    def run(self, question):
        """运行 ReAct+"""
        history = []
        compressed_history = []
        step_number = 1

        while step_number <= 10:
            # 构建 prompt（使用压缩历史）
            prompt = self._build_prompt(question, compressed_history, step_number)

            # 调用 LLM
            llm_output = self.llm_func(prompt)

            # 解析输出
            parsed = self._parse(llm_output)

            if parsed["type"] == "final_answer":
                return parsed["final_answer"]

            if parsed["type"] == "action":
                # 执行工具
                observation = self._execute_tool(parsed["action"], parsed["action_input"])

                # 记录历史
                history.append({
                    "thought": parsed["thought"],
                    "action": parsed["action"],
                    "observation": observation
                })

                # 压缩历史
                if len(history) > self.max_history:
                    compressed_history = self._compress_history(history)
                else:
                    compressed_history = history.copy()

                step_number += 1

        return "达到最大步数限制"

    def _compress_history(self, history):
        """压缩历史记录"""
        # 策略1: 保留最近几步
        recent = history[-3:]

        # 策略2: 对早期历史生成摘要
        early = history[:-3]
        if early:
            summary = self._summarize(early)
            return [{"thought": "历史摘要", "observation": summary}] + recent

        return recent

    def _summarize(self, history):
        """生成历史摘要"""
        prompt = "请将以下历史记录压缩为简洁的摘要：\n\n"
        for record in history:
            prompt += f"Thought: {record['thought']}\n"
            prompt += f"Observation: {record['observation']}\n\n"

        return self.llm_func(prompt)

    def _build_prompt(self, question, history, step_number):
        """构建 prompt"""
        prompt = f"问题: {question}\n\n"

        if history:
            prompt += "历史记录:\n"
            for i, record in enumerate(history, 1):
                prompt += f"{i}. {record['observation']}\n"
            prompt += "\n"

        prompt += f"当前是第 {step_number} 步。\n"
        prompt += "请按照格式回答：\nThought: [你的思考]\n"

        return prompt

    def _parse(self, output):
        """解析输出"""
        # 简化版解析
        if "Final Answer:" in output:
            return {"type": "final_answer", "final_answer": output.split("Final Answer:")[-1].strip()}

        if "Action:" in output:
            action = output.split("Action:")[-1].split("\n")[0].strip()
            action_input = output.split("Action Input:")[-1].split("\n")[0].strip()
            thought = output.split("Thought:")[-1].split("Action:")[0].strip()
            return {"type": "action", "thought": thought, "action": action, "action_input": action_input}

        return {"type": "thought", "thought": output}

    def _execute_tool(self, action_name, action_input):
        """执行工具"""
        tool = self.tools.get(action_name)
        if tool:
            return tool.execute(action_input)
        return f"工具 {action_name} 不存在"
```

### 4.8.3 Reflexion：反思与改进

Reflexion（Shinn et al., 2023）在 ReAct 的基础上增加了反思机制，让 Agent 能够从失败中学习：

```python
class ReflexionAgent:
    """Reflexion Agent：带反思的 Agent"""

    def __init__(self, llm_func, tools, max_reflections=3):
        self.llm_func = llm_func
        self.tools = tools
        self.max_reflections = max_reflections
        self.reflections = []

    def run(self, question):
        """运行 Reflexion Agent"""
        best_result = None
        best_score = 0

        for reflection_round in range(self.max_reflections):
            # 构建包含反思的 prompt
            prompt = self._build_reflexion_prompt(question, reflection_round)

            # 运行 ReAct
            result = self._run_react(prompt)

            # 评估结果
            score = self._evaluate(question, result)

            # 如果结果更好，更新最佳结果
            if score > best_score:
                best_score = score
                best_result = result

            # 如果分数足够高，提前结束
            if score >= 0.9:
                break

            # 生成反思
            reflection = self._reflect(question, result, score)
            self.reflections.append(reflection)

        return best_result

    def _build_reflexion_prompt(self, question, round_number):
        """构建包含反思的 prompt"""
        prompt = f"问题: {question}\n\n"

        if self.reflections:
            prompt += "之前的反思:\n"
            for i, reflection in enumerate(self.reflections, 1):
                prompt += f"第 {i} 次尝试的反思: {reflection}\n"
            prompt += "\n"

        prompt += f"这是第 {round_number + 1} 次尝试。\n"
        prompt += "请基于之前的反思，尝试更好地解决问题。\n\n"
        prompt += "请按照格式回答：\nThought: [你的思考]\n"

        return prompt

    def _run_react(self, prompt):
        """运行 ReAct"""
        # 简化版 ReAct 实现
        history = []
        step_number = 1

        while step_number <= 5:
            full_prompt = prompt + "\n\n历史记录:\n"
            for record in history:
                full_prompt += f"Thought: {record['thought']}\n"
                full_prompt += f"Observation: {record['observation']}\n\n"

            llm_output = self.llm_func(full_prompt)

            if "Final Answer:" in llm_output:
                return llm_output.split("Final Answer:")[-1].strip()

            if "Action:" in llm_output:
                action = llm_output.split("Action:")[-1].split("\n")[0].strip()
                action_input = llm_output.split("Action Input:")[-1].split("\n")[0].strip()
                thought = llm_output.split("Thought:")[-1].split("Action:")[0].strip()

                observation = self._execute_tool(action, action_input)
                history.append({"thought": thought, "observation": observation})

            step_number += 1

        return "无法完成任务"

    def _evaluate(self, question, result):
        """评估结果"""
        prompt = f"""请评估以下答案的质量：

问题: {question}
答案: {result}

请给出 0-1 之间的分数（1 表示完全正确）："""

        score_str = self.llm_func(prompt)

        try:
            return float(score_str.strip())
        except ValueError:
            return 0.5

    def _reflect(self, question, result, score):
        """生成反思"""
        prompt = f"""请反思这次尝试：

问题: {question}
结果: {result}
分数: {score}

请分析：
1. 什么做得好？
2. 什么做得不好？
3. 下次应该如何改进？

反思:"""

        return self.llm_func(prompt)

    def _execute_tool(self, action_name, action_input):
        """执行工具"""
        tool = self.tools.get(action_name)
        if tool:
            return tool.execute(action_input)
        return f"工具 {action_name} 不存在"
```

### 4.8.4 Plan-and-Execute：规划与执行分离

Plan-and-Execute 将任务分解为规划和执行两个阶段，先制定完整计划，再逐步执行：

```python
class PlanAndExecuteAgent:
    """Plan-and-Execute Agent：规划与执行分离"""

    def __init__(self, llm_func, tools):
        self.llm_func = llm_func
        self.tools = tools

    def run(self, question):
        """运行 Plan-and-Execute Agent"""
        # 阶段1: 规划
        plan = self._create_plan(question)

        # 阶段2: 执行
        results = []
        for step in plan:
            result = self._execute_step(step, results)
            results.append(result)

            # 如果步骤失败，重新规划
            if "错误" in result:
                plan = self._replan(question, results)
                results = []

        # 阶段3: 综合
        final_answer = self._synthesize(question, results)

        return final_answer

    def _create_plan(self, question):
        """创建计划"""
        prompt = f"""请为以下问题制定一个详细的执行计划：

问题: {question}

可用工具:
{self._get_tools_description()}

请列出需要执行的步骤（每行一个步骤）："""

        plan_text = self.llm_func(prompt)
        steps = [step.strip() for step in plan_text.split('\n') if step.strip()]

        return steps

    def _execute_step(self, step, previous_results):
        """执行步骤"""
        prompt = f"""请执行以下步骤：

步骤: {step}

之前的结果:
{self._format_results(previous_results)}

请按照格式回答：
Thought: [你的思考]
Action: [工具名称]
Action Input: [参数]"""

        llm_output = self.llm_func(prompt)

        # 解析并执行
        if "Action:" in llm_output:
            action = llm_output.split("Action:")[-1].split("\n")[0].strip()
            action_input = llm_output.split("Action Input:")[-1].split("\n")[0].strip()

            observation = self._execute_tool(action, action_input)
            return {"step": step, "result": observation}

        return {"step": step, "result": llm_output}

    def _replan(self, question, failed_results):
        """重新规划"""
        prompt = f"""之前的计划执行失败，请重新制定计划：

问题: {question}

失败的步骤:
{self._format_results(failed_results)}

请制定新的计划："""

        plan_text = self.llm_func(prompt)
        steps = [step.strip() for step in plan_text.split('\n') if step.strip()]

        return steps

    def _synthesize(self, question, results):
        """综合结果"""
        prompt = f"""请综合以下结果，给出最终答案：

问题: {question}

执行结果:
{self._format_results(results)}

最终答案:"""

        return self.llm_func(prompt)

    def _get_tools_description(self):
        """获取工具描述"""
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)

    def _format_results(self, results):
        """格式化结果"""
        if not results:
            return "无"

        parts = []
        for i, result in enumerate(results, 1):
            parts.append(f"{i}. {result.get('step', 'N/A')}: {result.get('result', 'N/A')}")
        return "\n".join(parts)

    def _execute_tool(self, action_name, action_input):
        """执行工具"""
        tool = self.tools.get(action_name)
        if tool:
            return tool.execute(action_input)
        return f"工具 {action_name} 不存在"
```

### 4.8.5 改进方向对比

| 方法 | 核心思想 | 优势 | 劣势 |
|:---|:---|:---|:---|
| **ReAct** | 推理与行动交替 | 简单直观 | Token 效率低 |
| **ReAct+** | 压缩历史记录 | 提高 Token 效率 | 可能丢失信息 |
| **Reflexion** | 反思与改进 | 能从失败中学习 | 增加延迟 |
| **Plan-and-Execute** | 规划与执行分离 | 有全局视角 | 规划可能不准确 |
| **LATS** | 树搜索 | 探索多种可能 | 计算成本高 |
| **ADaPT** | 自适应规划 | 灵活调整策略 | 实现复杂 |

---

## 4.9 工程最佳实践

### 4.9.1 Prompt 设计最佳实践

**1. 系统 prompt 设计**

```python
SYSTEM_PROMPT = """你是一个能够使用工具解决复杂问题的 AI 助师。

## 核心能力
- 分析复杂问题并制定解决策略
- 使用工具获取和验证信息
- 综合多个信息源得出准确结论

## 工作流程
1. 仔细分析问题，理解用户需求
2. 制定解决策略，确定需要哪些信息
3. 使用工具获取信息
4. 分析工具返回的结果
5. 如果信息不足，继续获取
6. 综合所有信息，给出最终答案

## 输出格式
每次行动前，你必须先进行思考：

Thought: [分析当前情况，决定下一步行动]
Action: [工具名称]
Action Input: [工具参数]

当收集到足够信息后：
Thought: [综合分析所有信息]
Final Answer: [最终答案]

## 重要规则
1. 每次只能调用一个工具
2. 工具名称必须准确无误
3. 工具参数必须符合要求
4. 如果工具执行失败，分析原因并尝试其他方法
5. 不要编造信息，如果不确定就承认
"""
```

**2. Few-shot 示例**

```python
FEW_SHOT_EXAMPLES = """
示例 1:
问题: 谁是 iPhone 的创始人？他是什么时候出生的？

Thought: 要回答这个问题，我需要先知道 iPhone 的创始人是谁，然后查找他的出生日期。
Action: search
Action Input: iPhone founder
Observation: iPhone 是由苹果公司推出的，苹果公司的创始人是 Steve Jobs。

Thought: iPhone 的创始人是 Steve Jobs。现在我需要查找他的出生日期。
Action: search
Action Input: Steve Jobs birth date
Observation: Steve Jobs 于 1955 年 2 月 24 日出生。

Thought: 我已经收集到所有需要的信息。iPhone 的创始人是 Steve Jobs，他出生于 1955 年 2 月 24 日。
Final Answer: iPhone 的创始人是 Steve Jobs，他出生于 1955 年 2 月 24 日。

示例 2:
问题: Python 和 Java 哪个更适合初学者？

Thought: 这是一个比较问题，我需要从多个角度分析 Python 和 Java 的特点。
Action: search
Action Input: Python vs Java for beginners
Observation: Python 语法简洁，学习曲线平缓；Java 语法严格，但更接近底层。

Thought: 我已经获取了基本信息，现在需要更深入的比较。
Action: search
Action Input: Python learning curve vs Java
Observation: Python 被认为是更容易学习的语言，因为它有简洁的语法和大量的学习资源。

Thought: 综合以上信息，Python 更适合初学者。
Final Answer: 对于初学者来说，Python 是更好的选择。它的语法简洁、学习曲线平缓，有丰富的学习资源。
"""
```

### 4.9.2 错误处理最佳实践

```python
class RobustReActAgent:
    """健壮的 ReAct Agent"""

    def __init__(self, llm_func, tools, config):
        self.llm_func = llm_func
        self.tools = tools
        self.config = config
        self.error_counts = {}

    def run(self, question):
        """运行 Agent"""
        history = []
        step_number = 1

        while step_number <= self.config.max_steps:
            try:
                # 构建 prompt
                prompt = self._build_prompt(question, history, step_number)

                # 调用 LLM（带重试）
                llm_output = self._call_llm_with_retry(prompt)

                # 解析输出
                parsed = self._parse(llm_output)

                # 处理结果
                if parsed["type"] == "final_answer":
                    return parsed["final_answer"]

                if parsed["type"] == "action":
                    # 执行工具（带错误处理）
                    observation = self._execute_tool_safely(
                        parsed["action"],
                        parsed["action_input"]
                    )

                    # 记录历史
                    history.append({
                        "thought": parsed["thought"],
                        "action": parsed["action"],
                        "observation": observation
                    })

                    step_number += 1

            except Exception as e:
                # 记录错误
                self._log_error(e, step_number)

                # 尝试恢复
                recovery = self._attempt_recovery(e, history)
                if recovery:
                    history.append(recovery)
                else:
                    # 无法恢复，返回当前最佳答案
                    return self._generate_fallback_answer(question, history)

        return "达到最大步数限制"

    def _call_llm_with_retry(self, prompt, max_retries=3):
        """带重试的 LLM 调用"""
        for attempt in range(max_retries):
            try:
                return self.llm_func(prompt)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 指数退避

    def _execute_tool_safely(self, action_name, action_input):
        """安全执行工具"""
        try:
            tool = self.tools.get(action_name)
            if not tool:
                return f"工具 '{action_name}' 不存在"

            result = tool.execute(action_input, timeout=self.config.timeout)
            return result

        except TimeoutError:
            return f"工具 '{action_name}' 执行超时"
        except Exception as e:
            return f"工具执行错误: {str(e)}"

    def _attempt_recovery(self, error, history):
        """尝试从错误中恢复"""
        prompt = f"""工具执行失败，请分析错误并提出解决方案：

错误: {str(error)}

历史记录:
{self._format_history(history)}

请分析：
1. 错误的原因是什么？
2. 如何避免这个错误？
3. 下一步应该怎么做？

Thought: """

        try:
            response = self.llm_func(prompt)
            return {"thought": response, "observation": f"错误恢复: {str(error)}"}
        except:
            return None

    def _generate_fallback_answer(self, question, history):
        """生成备用答案"""
        prompt = f"""基于已有的信息，尝试给出答案：

问题: {question}

已收集的信息:
{self._format_history(history)}

如果信息不足，请说明还需要什么信息。"""

        return self.llm_func(prompt)

    def _log_error(self, error, step_number):
        """记录错误"""
        error_type = type(error).__name__
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1

        print(f"[错误] 步骤 {step_number}: {error_type} - {str(error)}")

    def _build_prompt(self, question, history, step_number):
        """构建 prompt"""
        # 实现省略
        pass

    def _parse(self, output):
        """解析输出"""
        # 实现省略
        pass

    def _format_history(self, history):
        """格式化历史记录"""
        # 实现省略
        pass
```

### 4.9.3 监控与日志

```python
import logging
from datetime import datetime
from typing import Dict, Any

class ReActMonitor:
    """ReAct 监控器"""

    def __init__(self, log_file: str = "react_agent.log"):
        """
        初始化监控器

        参数:
            log_file: 日志文件路径
        """
        self.logger = logging.getLogger("ReActAgent")
        self.logger.setLevel(logging.INFO)

        # 文件处理器
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # 统计信息
        self.stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_steps": 0,
            "total_tokens": 0,
            "average_steps": 0,
            "error_counts": {}
        }

    def log_step(self, step_number: int, thought: str, action: str,
                 observation: str, duration: float):
        """记录步骤"""
        self.logger.info(
            f"Step {step_number}: "
            f"Action={action}, "
            f"Duration={duration:.2f}s"
        )

    def log_run_start(self, question: str):
        """记录运行开始"""
        self.stats["total_runs"] += 1
        self.logger.info(f"Run started: {question[:100]}...")

    def log_run_end(self, success: bool, answer: str, total_steps: int,
                    total_tokens: int):
        """记录运行结束"""
        if success:
            self.stats["successful_runs"] += 1
        else:
            self.stats["failed_runs"] += 1

        self.stats["total_steps"] += total_steps
        self.stats["total_tokens"] += total_tokens
        self.stats["average_steps"] = (
            self.stats["total_steps"] / self.stats["total_runs"]
        )

        self.logger.info(
            f"Run ended: success={success}, "
            f"steps={total_steps}, tokens={total_tokens}"
        )

    def log_error(self, error: Exception, context: Dict[str, Any]):
        """记录错误"""
        error_type = type(error).__name__
        if error_type not in self.stats["error_counts"]:
            self.stats["error_counts"][error_type] = 0
        self.stats["error_counts"][error_type] += 1

        self.logger.error(
            f"Error: {error_type} - {str(error)}\n"
            f"Context: {context}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()

    def print_stats(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("ReAct Agent 统计信息")
        print("="*60)
        print(f"总运行次数: {self.stats['total_runs']}")
        print(f"成功次数: {self.stats['successful_runs']}")
        print(f"失败次数: {self.stats['failed_runs']}")
        print(f"成功率: {self.stats['successful_runs']/max(self.stats['total_runs'], 1)*100:.1f}%")
        print(f"总步数: {self.stats['total_steps']}")
        print(f"平均步数: {self.stats['average_steps']:.1f}")
        print(f"总 Token 数: {self.stats['total_tokens']}")
        print("\n错误统计:")
        for error_type, count in self.stats["error_counts"].items():
            print(f"  {error_type}: {count}")
        print("="*60)
```

### 4.9.4 测试策略

```python
import unittest
from unittest.mock import Mock, patch

class TestReActAgent(unittest.TestCase):
    """ReAct Agent 测试"""

    def setUp(self):
        """测试设置"""
        self.mock_llm = Mock()
        self.tools = {
            "search": Mock(execute=Mock(return_value="搜索结果")),
            "calculator": Mock(execute=Mock(return_value="42"))
        }

    def test_simple_question(self):
        """测试简单问题"""
        # 设置 LLM 响应
        self.mock_llm.return_value = (
            "Thought: 我需要搜索相关信息\n"
            "Action: search\n"
            "Action Input: test query\n"
        )

        agent = ReActAgent(self.mock_llm)
        agent.register_tool("search", "搜索工具", lambda x: "搜索结果")

        # 运行 Agent
        with patch.object(agent, '_call_llm') as mock_call:
            mock_call.side_effect = [
                "Thought: 我需要搜索\nAction: search\nAction Input: test",
                "Thought: 我得到了答案\nFinal Answer: 测试答案"
            ]
            result = agent.run("测试问题")

        self.assertEqual(result, "测试答案")

    def test_tool_not_found(self):
        """测试工具不存在"""
        agent = ReActAgent(self.mock_llm)

        with patch.object(agent, '_call_llm') as mock_call:
            mock_call.return_value = (
                "Thought: 我需要使用工具\n"
                "Action: nonexistent_tool\n"
                "Action Input: test"
            )

            # 应该处理工具不存在的情况
            result = agent.run("测试问题")
            self.assertIsNotNone(result)

    def test_max_steps(self):
        """测试最大步数限制"""
        agent = ReActAgent(self.mock_llm, config=AgentConfig(max_steps=2))

        with patch.object(agent, '_call_llm') as mock_call:
            mock_call.return_value = (
                "Thought: 继续\nAction: search\nAction Input: test"
            )
            result = agent.run("测试问题")

        # 应该在达到最大步数后停止
        self.assertIsNotNone(result)

    def test_memory_compression(self):
        """测试记忆压缩"""
        memory = MemoryManager(max_history=5)

        # 添加超过限制的记录
        for i in range(10):
            record = StepRecord(
                step_number=i,
                thought=f"思考 {i}",
                observation=f"观察 {i}"
            )
            memory.add_step(record)

        # 应该已经压缩
        self.assertLessEqual(len(memory.history), 5)
        self.assertIsNotNone(memory.summary)

 if __name__ == "__main__":
     unittest.main()
 ```

Agent 规划是复杂任务解决的关键。下面的交互式可视化展示了 Agent 规划树的搜索过程：

<div data-component="AgentPlanningDemoV4"></div>

 ---

 ## 4.10 本章小结

### 4.10.1 核心知识点回顾

| 知识点 | 核心要点 |
|:---|:---|
| **ReAct 本质** | 推理（Thought）与行动（Action）交替进行，互相增强 |
| **理论基础** | 结合 CoT 的推理能力和工具的信息获取能力 |
| **三种范式** | CoT-only（纯推理）、Act-only（纯行动）、ReAct（推理+行动） |
| **Thought 作用** | 理解状态、制定策略、评估进展、处理错误 |
| **Action 选择** | 基于工具描述、Few-shot 示例、语义相似度 |
| **Observation 处理** | 截断、摘要、结构化、过滤 |
| **手动实现** | 展示底层机制，适合理解和定制 |
| **LangGraph** | 生产级实现，支持流式、检查点、错误恢复 |
| **局限性** | Token 效率低、错误传播、无全局规划 |
| **改进方向** | ReAct+（压缩）、Reflexion（反思）、Plan-and-Execute（规划） |

### 4.10.2 实践建议

**选择合适的范式**：

1. **简单问答**：使用 CoT-only
2. **信息检索**：使用 Act-only
3. **复杂任务**：使用 ReAct
4. **需要反思**：使用 Reflexion
5. **需要全局规划**：使用 Plan-and-Execute

**工程实践要点**：

1. **Prompt 设计**：清晰的系统 prompt + Few-shot 示例
2. **错误处理**：重试机制 + 优雅降级
3. **监控日志**：记录关键指标 + 错误追踪
4. **性能优化**：历史压缩 + Token 优化
5. **测试验证**：单元测试 + 集成测试

### 4.10.3 未来展望

ReAct 范式仍在快速发展中，未来可能的发展方向包括：

1. **更高效的推理**：减少 Token 使用，提高推理速度
2. **更强的规划能力**：结合规划算法，实现更复杂的任务
3. **更好的错误恢复**：自动检测和修复错误
4. **多模态支持**：支持图像、音频等多种模态
5. **自主学习**：从经验中学习，不断提升能力

---

## 4.11 思考题

### 理解题

1. **ReAct 与 CoT 的本质区别是什么？为什么说 ReAct 结合了两者的优势？**

2. **在 ReAct 中，Thought 的作用是什么？如果没有 Thought，ReAct 会变成什么？**

3. **ReAct 的三种范式对比中，为什么 ReAct 在多跳问答任务上表现最好？**

### 实践题

4. **请实现一个简单的 ReAct Agent，要求：**
   - 支持至少 3 种工具（搜索、计算器、翻译）
   - 包含错误处理机制
   - 记录执行日志

5. **使用 LangGraph 实现一个带检查点的 ReAct Agent，要求：**
   - 支持对话状态保持
   - 支持流式输出
   - 支持人类介入

6. **设计一个实验，对比 ReAct 和 ReAct+ 在以下方面的性能：**
   - Token 使用量
   - 响应时间
   - 准确率

### 设计题

7. **如果你要设计一个新的 ReAct 改进方案，你会如何改进？请说明：**
   - 你想解决什么问题
   - 你的改进方案是什么
   - 你会如何验证改进效果

8. **在实际生产环境中，如何平衡 ReAct 的准确性和效率？请给出具体建议。**

9. **思考 ReAct 在以下场景中的应用：**
   - 客服机器人
   - 代码助手
   - 研究助理

   请分析每个场景的特点，并设计相应的 ReAct 工作流程。

### 扩展题

10. **阅读以下论文，分析它们与 ReAct 的关系：**
    - Tree of Thoughts (Yao et al., 2023)
    - LATS (Zhou et al., 2023)
    - ADaPT (Wang et al., 2023)

    请说明这些方法是如何改进 ReAct 的，以及它们各自的优缺点。

---

> **下一章预告**
>
> 在第 5 章中，我们将深入 Function Calling 的底层机制，理解 LLM 如何生成工具调用请求，以及 OpenAI 和 Anthropic API 的实现差异。我们将探讨如何设计高质量的工具接口，以及如何处理复杂的多工具调用场景。
