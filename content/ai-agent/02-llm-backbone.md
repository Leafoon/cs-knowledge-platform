---
title: "第2章：大模型基座 — Agent 的大脑"
description: "深入理解 LLM 作为 Agent 推理引擎的核心能力，掌握 Transformer 架构、上下文窗口、Token 机制、温度与采样策略对 Agent 行为的影响，主流模型对比与选型，API 调用最佳实践。"
date: "2026-06-15"
---

 # 第2章：大模型基座 — Agent 的大脑

 大语言模型（Large Language Model, LLM）是 Agent 的推理核心。如果说 Agent 是一个智能体，那么 LLM 就是这个智能体的"大脑"——它负责理解用户意图、规划任务步骤、选择合适工具、分析执行结果、做出下一步决策。没有强大的 LLM，再精巧的工具链和记忆系统也无法发挥作用。

 下面的交互式对比展示了主流 LLM 模型的特点：

 <div data-component="LLMModelComparison"></div>

| 能力 | 说明 | 对 Agent 的意义 | 涌现规模 |
|:---|:---|:---|:---|
| **通用推理** | 基于自然语言的逻辑推理 | 理解任务、分解计划、分析结果 | ~10B |
| **上下文学习（In-Context Learning）** | 从少量示例中学习新任务 | Few-shot 工具使用指导，无需微调 | ~10B |
| **指令遵循（Instruction Following）** | 按照提示词中的指令执行任务 | 遵循 Agent 行为规范、安全约束 | ~100B |
| **工具理解（Tool Use）** | 理解函数签名、参数含义、返回值 | 正确选择和调用工具 | ~100B |
| **代码能力（Code Generation）** | 理解、生成和调试代码 | 代码 Agent、数据分析 Agent | ~100B |
| **结构化输出（Structured Output）** | 生成符合 JSON Schema 的数据 | Function Calling、数据提取、工具调用 | ~10B |

这六项能力不是孤立存在的，它们在 Agent 运行时协同工作。例如，当用户说"帮我分析这个 CSV 文件并生成图表"时，Agent 需要：

1. **理解意图**（通用推理）：识别用户想要数据分析和可视化
2. **规划步骤**（指令遵循 + 推理）：读取文件 → 分析数据 → 选择图表类型 → 生成代码
3. **调用工具**（工具理解）：选择 `read_file` 工具读取 CSV，选择 `execute_code` 工具运行分析
4. **生成输出**（代码能力 + 结构化输出）：编写 Python 代码并以 JSON 格式返回工具调用

**LLM vs 传统规则引擎的深度对比**：

| 维度 | 传统规则引擎 | LLM Agent |
|:---|:---|:---|
| 规则定义 | 人工编写每条规则，维护成本高 | LLM 自主推理，从数据中学习 |
| 新场景适应 | 需要人工添加新规则，响应慢 | 自动适应，无需修改代码 |
| 规则冲突 | 规则之间容易产生矛盾，难以调试 | 自然语言推理避免显式冲突 |
| 可解释性 | 规则链可追溯，完全透明 | 推理过程可解释但不完全透明 |
| 泛化能力 | 仅处理预定义场景 | 可处理未见过的新场景 |
| 开发成本 | 低（简单场景）/ 极高（复杂场景） | 中等（Prompt 工程） |
| 执行确定性 | 100% 确定性 | 概率性，可能产生不同结果 |

> **关键洞察**：小模型（<10B 参数）虽然也能做简单的工具调用，但在复杂场景下（多步推理、错误恢复、模糊意图理解）表现显著差于大模型（>100B）。**Agent 场景对模型规模有硬性要求**。经验法则是：工具调用至少需要 10B 参数，多步推理至少需要 70B 参数，复杂规划需要 100B+ 参数。

### 2.1.2 LLM 的能力与局限性

LLM 作为 Agent 大脑有显著优势，但也存在必须正视的局限性。理解这些局限性是设计健壮 Agent 系统的前提。

**LLM 的核心能力**：

- **语言理解**：能理解复杂的自然语言指令，包括隐含意图、上下文依赖、多轮对话中的指代消解
- **知识检索**：在预训练数据中学到的广泛知识，可以作为 Agent 的"常识库"
- **推理能力**：通过 Chain-of-Thought 进行多步推理，处理复杂的逻辑关系
- **代码生成**：理解编程语言，生成可执行的代码，这是代码 Agent 的基础
- **模式识别**：从少量示例中识别模式并泛化，这是 Few-shot 学习的基础
- **多语言能力**：支持数十种语言，使 Agent 能服务全球用户

**LLM 的核心局限性**：

| 局限性 | 表现 | Agent 应对策略 |
|:---|:---|:---|
| **幻觉（Hallucination）** | 生成看似合理但错误的信息 | 工具验证、RAG、人类确认 |
| **知识截止日期** | 不知道训练数据之后的信息 | 实时工具调用、搜索集成 |
| **上下文窗口限制** | 无法处理超长输入 | 上下文窗口管理、摘要压缩 |
| **推理深度有限** | 复杂多步推理容易出错 | 分步执行、自我验证 |
| **缺乏持久记忆** | 每次调用独立，无法记住历史 | 外部记忆系统 |
| **无法执行代码** | 只能生成代码，不能直接运行 | 代码执行沙箱 |
| **概率性输出** | 相同输入可能产生不同输出 | temperature=0、多次采样 |
| **延迟与成本** | 推理需要时间，API 调用有成本 | 缓存、模型选择、批处理 |

这些局限性并不意味着 LLM 不适合做 Agent 大脑——恰恰相反，正是因为理解了这些局限性，我们才能设计出更健壮的 Agent 系统。例如，针对幻觉问题，我们可以通过工具调用让 Agent 验证自己的输出；针对记忆缺失，我们可以设计外部记忆系统来弥补。

### 2.1.3 涌现能力与 Agent 能力的关系

当模型规模超过某个阈值时，会出现训练阶段未明确优化的**涌现能力（Emergent Abilities）**。这些能力是 Agent 的关键技术基础。

涌现能力的形式化表示：

$$
\text{Performance}(s) = \begin{cases} \text{random} & s < s_{\text{threshold}} \\ f(s) & s \geq s_{\text{threshold}} \end{cases}
$$

其中 $s$ 是模型参数规模，$s_{\text{threshold}}$ 是涌现阈值。当模型规模低于阈值时，能力表现为随机水平；超过阈值后，能力随规模增长而快速提升。

**关键涌现能力与 Agent 应用的对应关系**：

| 涌现能力 | 出现规模 | 描述 | Agent 应用 |
|:---|:---|:---|:---|
| **Chain-of-Thought** | ~100B 参数 | 通过中间推理步骤得出结论 | 多步推理规划 |
| **In-Context Learning** | ~10B 参数 | 从少量示例中学习新任务 | Few-shot 工具学习 |
| **Instruction Following** | ~100B 参数 | 精确遵循复杂指令 | 遵循 Agent 指令 |
| **Code Generation** | ~100B 参数 | 理解并生成可执行代码 | 代码 Agent |
| **Tool Use** | ~100B 参数 | 理解工具签名并正确调用 | Function Calling |
| **Self-Consistency** | ~100B 参数 | 多路径推理并投票 | 可靠决策 |
| **Self-Correction** | ~100B 参数 | 识别并修正自己的错误 | 错误恢复 |
| **Multi-Step Planning** | ~100B+ 参数 | 制定和执行复杂计划 | 任务规划 Agent |

 > **经验法则**：对于 Agent 开发，模型参数规模的选择有一个粗略的指导原则：简单工具调用用 10B-30B 模型（如 GPT-4o-mini），复杂推理用 70B-100B 模型（如 Claude Sonnet 4），顶级规划用 100B+ 或 MoE 模型（如 GPT-4o、Claude Opus）。

ReAct 框架是最早将 LLM 推理与工具使用结合的经典架构之一。下面的交互式演示展示了 ReAct 框架的完整工作流程：

<div data-component="ReActDemo"></div>

### 2.1.4 不同 LLM 对 Agent 行为的影响

不同 LLM 在 Agent 场景下的表现差异显著。这种差异不仅体现在"能不能做"，更体现在"做得好不好"。

**模型特性对 Agent 行为的具体影响**：

| 模型特性 | 对 Agent 的影响 | 示例 |
|:---|:---|:---|
| **指令遵循能力** | 影响 Agent 是否严格按规范行为 | GPT-4o 能精确遵循复杂的 System Prompt |
| **推理深度** | 影响多步规划的准确性 | Claude 4 在复杂推理任务上更稳定 |
| **工具调用准确性** | 影响 Function Calling 的成功率 | GPT-4o 的 Function Calling 成功率 ~95% |
| **上下文长度** | 影响 Agent 能处理的信息量 | Gemini 2.5 Pro 的 1M 窗口适合超长文档 |
| **响应速度** | 影响 Agent 的交互体验 | GPT-4o-mini 的延迟 ~0.5s vs GPT-4o ~2s |
| **中文能力** | 影响中文场景的 Agent 表现 | DeepSeek R1、Qwen 在中文上表现优异 |
| **成本** | 影响 Agent 的运营成本 | GPT-4o-mini 成本仅为 GPT-4o 的 1/15 |

**不同模型在典型 Agent 场景下的表现对比**：

| 场景 | GPT-4o | Claude Sonnet 4 | Gemini 2.5 Pro | DeepSeek R1 |
|:---|:---|:---|:---|:---|
| 简单工具调用 | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ |
| 多步推理 | ★★★★ | ★★★★★ | ★★★★ | ★★★★★ |
| 代码生成 | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ |
| 角色扮演 | ★★★★ | ★★★★★ | ★★★ | ★★★ |
| 中文理解 | ★★★★ | ★★★★ | ★★★★ | ★★★★★ |
| 长文档处理 | ★★★★ | ★★★★★ | ★★★★★ | ★★★★ |
| 指令遵循 | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ |

> **选型建议**：没有"最好"的模型，只有"最适合"的模型。选择时需要综合考虑任务复杂度、延迟要求、成本预算、语言需求等因素。在实际开发中，建议采用"主模型 + 备选模型"的策略，当主模型失败时自动切换到备选模型。

### 2.1.5 LLM 的自回归生成机制

LLM 的输出生成过程是一个**自回归（Autoregressive）**过程——模型逐个生成 Token，每个 Token 的生成都基于之前生成的所有 Token。这一机制深刻影响了 Agent 的行为模式。

自回归生成的数学形式：

$$
P(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} P(x_i | x_1, x_2, \ldots, x_{i-1})
$$

其中 $x_i$ 是第 $i$ 个生成的 Token，$P(x_i | x_1, \ldots, x_{i-1})$ 是给定前 $i-1$ 个 Token 时第 $i$ 个 Token 的条件概率。

**自回归机制对 Agent 的影响**：

| 特性 | 描述 | Agent 影响 |
|:---|:---|:---|
| **上下文依赖性** | 输出取决于整个输入上下文 | 我们可以通过 Prompt 精确引导 Agent 行为 |
| **概率性** | 输出是概率分布，具有随机性 | 需要多次采样或 temperature=0 提高可靠性 |
| **序列性** | 输出逐个 Token 生成，无法并行 | 限制了并行化，影响延迟 |
| **不可回溯性** | 一旦生成就不能修改 | 需要显式的反思和纠错机制 |
| **前缀一致性** | 相同前缀产生相同分布 | 可以通过 Prefix Caching 优化性能 |

理解自回归机制对于 Agent 开发至关重要。例如，当我们需要 Agent 输出 JSON 格式的工具调用时，自回归机制意味着一旦前几个 Token 出现偏差，后续输出很可能会完全偏离目标。这就是为什么**结构化输出（Structured Output）**和**函数调用（Function Calling）**功能如此重要——它们通过约束解码（Constrained Decoding）确保输出符合预定义的格式。

### 2.1.6 Agent 对 LLM 的特殊要求

并非所有 LLM 都适合做 Agent 大脑。Agent 场景对 LLM 有一些特殊要求，这些要求超出了通用聊天机器人的需求。

**Agent 场景的 LLM 能力矩阵**：

| 能力维度 | 基础要求 | 进阶要求 | 为什么重要 |
|:---|:---|:---|:---|
| **指令遵循** | 遵循简单指令 | 遵循复杂的多层嵌套指令 | System Prompt 通常很复杂 |
| **格式控制** | 输出简单格式 | 精确输出 JSON/Function Call | 工具调用依赖精确格式 |
| **推理深度** | 单步推理 | 5-10 步 Chain-of-Thought | 复杂任务需要多步推理 |
| **错误识别** | 识别明显错误 | 识别隐蔽的逻辑错误 | Agent 需要自我纠错 |
| **上下文利用** | 利用相邻信息 | 利用远距离上下文 | 长对话中需要引用早期信息 |
| **工具理解** | 理解单个工具 | 理解工具之间的关系和组合 | 复杂任务需要多工具协作 |
| **不确定性表达** | 回答"不知道" | 表达置信度和替代方案 | Agent 需要判断何时寻求帮助 |
| **并行输出** | 单次输出 | 支持并行工具调用 | 提升 Agent 执行效率 |

**LLM 能力对 Agent 可靠性的影响**：

一个 LLM 在 Agent 场景下的可靠性可以用以下公式近似评估：

$$
R_{\text{agent}} \approx R_{\text{instruction}} \times R_{\text{format}} \times R_{\text{reasoning}} \times R_{\text{tool}}
$$

其中：
- $R_{\text{instruction}}$ 是指令遵循准确率（通常 > 95%）
- $R_{\text{format}}$ 是格式输出准确率（Function Calling 通常 > 98%）
- $R_{\text{reasoning}}$ 是推理准确率（取决于任务复杂度，70%-95%）
- $R_{\text{tool}}$ 是工具调用准确率（通常 > 90%）

例如，如果每个环节的准确率都是 95%，那么整体可靠性为 $0.95^4 \approx 0.81$，即约 81%。这意味着每 5 次调用可能有 1 次出错。在 Agent 的多步流程中，这个错误率会被放大。

> **关键洞察**：这就是为什么 Agent 系统需要完善的错误处理和重试机制。即使使用最顶级的 LLM，单步 95% 的准确率在 10 步流程中也只有 $0.95^{10} \approx 0.60$ 的成功率。**Agent 的可靠性不能仅依赖 LLM 的质量，还需要系统层面的保障。**

### 2.1.7 LLM 选择的经济学分析

选择 LLM 不仅是技术决策，也是经济决策。理解不同模型的成本结构有助于做出明智的选择。

**模型成本的三个维度**：

1. **直接成本**：API 调用费用（按 Token 计费）
2. **间接成本**：延迟导致的用户体验下降、重试带来的额外调用
3. **机会成本**：使用较弱模型导致的错误和返工

**不同 Agent 场景的月度成本估算**（假设每天 1000 次调用）：

| 场景 | GPT-4o | GPT-4o-mini | Claude Sonnet 4 | DeepSeek R1 |
|:---|:---|:---|:---|:---|
| 简单问答 | $135/月 | $8/月 | $180/月 | $27/月 |
| 工具调用 (3步) | $390/月 | $23/月 | $585/月 | $83/月 |
| 复杂推理 (5步) | $900/月 | $54/月 | $1350/月 | $195/月 |
| 代码 Agent | $1500/月 | $90/月 | $2250/月 | $325/月 |

**成本优化策略**：

| 策略 | 描述 | 节省比例 | 实现难度 |
|:---|:---|:---|:---|
| **模型路由** | 根据任务复杂度选择模型 | 50-70% | 中 |
| **Prompt 缓存** | 缓存相同 Prompt 的响应 | 30-50% | 低 |
| **响应缓存** | 缓存完全相同请求的响应 | 20-40% | 低 |
| **批量处理** | 使用 Batch API 获得折扣 | 50% | 中 |
| **输出压缩** | 精简 System Prompt 和工具描述 | 10-30% | 低 |
| **本地模型** | 简单任务使用本地模型 | 80-100% | 高 |

```python
# 模型路由的成本优化示例
class CostOptimizedAgent:
    """成本优化的 Agent 实现

    通过模型路由策略，在保证质量的同时最大化成本效益。
    """

    def __init__(self):
        # 定义不同级别的模型
        self.models = {
            "tier1": {"model": "gpt-4o-mini", "cost_per_1k": 0.00075},   # 简单任务
            "tier2": {"model": "gpt-4o", "cost_per_1k": 0.0125},         # 中等任务
            "tier3": {"model": "claude-sonnet-4", "cost_per_1k": 0.018},  # 复杂任务
        }

    def classify_task(self, user_input: str) -> str:
        """将任务分类为不同复杂度级别

        实际实现中可以使用一个小型分类模型或规则引擎。

        Args:
            user_input: 用户输入

        Returns:
            任务复杂度级别（tier1/tier2/tier3）
        """
        # 简单的基于关键词的分类（实际中应该更复杂）
        complex_keywords = ["分析", "规划", "设计", "优化", "对比", "推理"]
        simple_keywords = ["查询", "搜索", "获取", "计算", "转换"]

        for keyword in complex_keywords:
            if keyword in user_input:
                return "tier3"

        for keyword in simple_keywords:
            if keyword in user_input:
                return "tier1"

        return "tier2"

    def estimate_monthly_cost(self, daily_calls: int = 1000) -> dict:
        """估算月度成本

        Args:
            daily_calls: 每天的调用次数

        Returns:
            各层级的月度成本估算
        """
        monthly = daily_calls * 30
        costs = {}

        for tier, info in self.models.items():
            # 假设每次调用平均 2000 Token
            avg_tokens = 2000
            cost_per_call = avg_tokens * info["cost_per_1k"] / 1000
            costs[tier] = {
                "model": info["model"],
                "monthly_cost": cost_per_call * monthly,
                "cost_per_call": cost_per_call,
            }

        return costs
```

### 2.1.8 LLM 的推理时计算（Inference-Time Compute）

近年来，一个重要的趋势是增加模型在推理时的计算量，以提升输出质量。这对于 Agent 场景尤为重要，因为 Agent 通常需要处理复杂的多步任务。

**推理时计算的主要形式**：

| 方法 | 描述 | 效果 | 成本增加 | 适用场景 |
|:---|:---|:---|:---|:---|
| **Chain-of-Thought** | 要求模型展示推理过程 | 提升推理准确性 | 2-5x | 复杂推理任务 |
| **Self-Consistency** | 多次采样 + 投票 | 提升可靠性 | 3-10x | 高可靠性要求 |
| **Tree-of-Thought** | 探索多条推理路径 | 提升复杂问题解决率 | 5-20x | 规划和搜索任务 |
| **Best-of-N** | 生成 N 个候选，选最好的 | 提升输出质量 | N x | 代码生成、创意写作 |
| **Rejection Sampling** | 不满足条件则重新生成 | 提升格式正确率 | 1.5-3x | 结构化输出 |

**推理时计算的成本-收益分析**：

$$
\text{净收益} = \text{质量提升带来的价值} - \text{额外计算的成本}
$$

在 Agent 场景中，质量提升的价值通常远高于额外计算的成本。例如，一次错误的工具调用可能导致整个流程失败，需要重新执行，成本远高于多花几倍的 Token 来确保第一次就做对。

 > **Agent 最佳实践**：对于关键路径上的决策（如工具选择、参数填充），建议使用较高的推理时计算量（如 Chain-of-Thought + Self-Consistency）。对于非关键路径（如简单的信息检索），可以使用较少的计算量来控制成本。

不同的 LLM 基座和实现方式各有优劣。下面的交互式对比可以帮助你根据项目需求选择最合适的方案：

<div data-component="AgentArchitectureComparisonV12"></div>

---

## 2.2 Transformer 架构速览

### 2.2.1 Self-Attention 机制的数学原理

Transformer 架构的核心是 **Self-Attention（自注意力）**机制。它允许模型在处理序列中的每个位置时，关注序列中所有其他位置的信息。这是 LLM 能够理解长距离依赖关系的关键。

Self-Attention 的计算过程可以用以下数学公式描述：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：
- $Q$（Query，查询矩阵）：表示当前位置想要"查找"什么信息
- $K$（Key，键矩阵）：表示每个位置"提供"什么信息的索引
- $V$（Value，值矩阵）：表示每个位置实际包含的信息内容
- $d_k$ 是 Key 的维度，用于缩放防止点积过大导致梯度消失

**Self-Attention 的直觉理解**：

想象你在读一个句子："猫坐在垫子上，它很舒服"。当你处理"它"这个词时，Self-Attention 会计算"它"与句子中每个词的相关性分数。显然，"它"与"猫"的相关性最高，因此模型会将"猫"的信息聚合到"它"的位置上，从而理解"它"指的是"猫"。

**缩放点积注意力的完整计算过程**：

1. **线性投影**：将输入 $X$ 分别乘以三个权重矩阵 $W^Q$, $W^K$, $W^V$，得到 $Q$, $K$, $V$
2. **计算注意力分数**：$QK^T$ 得到每对位置之间的相关性分数
3. **缩放**：除以 $\sqrt{d_k}$ 防止点积值过大
4. **掩码**（可选）：在解码器中，通过因果掩码确保每个位置只能关注之前的位置
5. **Softmax 归一化**：将分数转换为概率分布，使权重之和为 1
6. **加权求和**：用注意力权重对 $V$ 进行加权求和，得到输出

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """Self-Attention 的完整 PyTorch 实现

    该类实现了缩放点积注意力机制（Scaled Dot-Product Attention），
    是 Transformer 架构的核心组件。在 LLM 中，Self-Attention 使模型
    能够在处理每个 token 时关注输入序列中的所有其他 token。
    """

    def __init__(self, d_model: int, n_heads: int):
        """初始化 Self-Attention 层

        Args:
            d_model: 模型的隐藏维度（如 GPT-3 为 12288）
            n_heads: 注意力头的数量（如 GPT-3 为 96）
        """
        super().__init__()
        self.d_model = d_model       # 隐藏维度
        self.n_heads = n_heads       # 注意力头数
        self.d_k = d_model // n_heads  # 每个头的维度

        # 四个线性变换矩阵
        # W_q: 将输入映射为查询向量
        self.W_q = nn.Linear(d_model, d_model)
        # W_k: 将输入映射为键向量
        self.W_k = nn.Linear(d_model, d_model)
        # W_v: 将输入映射为值向量
        self.W_v = nn.Linear(d_model, d_model)
        # W_o: 将多头输出合并后映射回隐藏维度
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """前向传播

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 可选的掩码张量，用于遮蔽某些位置

        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: 线性投影，将输入映射为 Q, K, V
        # 每个形状为 (batch_size, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: 将 Q, K, V 拆分为多头
        # 先 reshape 为 (batch_size, seq_len, n_heads, d_k)
        # 再 transpose 为 (batch_size, n_heads, seq_len, d_k)
        # 这样每个注意力头独立处理自己的子空间
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Step 3: 计算注意力分数
        # Q @ K^T 得到 (batch_size, n_heads, seq_len, seq_len)
        # 除以 sqrt(d_k) 进行缩放，防止点积值过大
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Step 4: 应用掩码（如果有的话）
        # 在因果注意力中，掩码确保每个位置只能关注之前的位置
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 5: Softmax 归一化
        # 将分数转换为注意力权重，使每行之和为 1
        attn_weights = torch.softmax(scores, dim=-1)

        # Step 6: 加权求和
        # 用注意力权重对 V 进行加权求和
        output = torch.matmul(attn_weights, V)

        # Step 7: 合并多头
        # transpose 回 (batch_size, seq_len, n_heads, d_k)
        # reshape 为 (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # Step 8: 最终线性投影
        return self.W_o(output)
```

### 2.2.2 多头注意力的直觉理解

**多头注意力（Multi-Head Attention）**是 Self-Attention 的扩展。它将注意力计算拆分为多个独立的"头"，每个头关注输入的不同方面。

直觉理解：想象你在一个图书馆里查找关于"人工智能"的资料。如果你只用一个搜索条件（如关键词"AI"），可能会遗漏一些相关资料。但如果你同时用多个搜索条件——关键词"AI"、主题"机器学习"、作者"Hinton"——你就更有可能找到所有相关资料。多头注意力的工作原理类似：不同的头关注不同的信息模式。

**多头注意力的计算**：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

其中每个头：

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

| Head 类型 | 关注方面 | 示例 | 在 Agent 中的作用 |
|:---|:---|:---|:---|
| 语法 Head | 语法关系 | "cat" 关注 "the" | 理解用户输入的语法结构 |
| 语义 Head | 语义关系 | "cat" 关注 "pet" | 理解用户意图的语义含义 |
| 位置 Head | 位置关系 | 当前词关注邻近词 | 理解上下文的局部模式 |
| 长程依赖 Head | 长距离关系 | 句首关注句尾 | 处理长对话中的远距离引用 |
| 工具 Head | 工具相关 | 工具名关注参数 | 正确匹配工具和参数 |
| 格式 Head | 输出格式 | JSON key 关注 value | 生成结构化输出 |

### 2.2.3 位置编码技术

Self-Attention 本身是**排列不变的（Permutation Invariant）**——它不关心输入 Token 的顺序。但语言的含义高度依赖词序（"猫吃鱼"和"鱼吃猫"含义完全不同）。因此，需要**位置编码（Positional Encoding）**来注入位置信息。

**正弦位置编码（Sinusoidal Positional Encoding）**：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

其中 $pos$ 是 Token 在序列中的位置，$i$ 是维度索引，$d_{\text{model}}$ 是模型维度。

**旋转位置编码（RoPE, Rotary Position Embedding）**：

RoPE 是目前主流 LLM（如 Llama、Qwen、DeepSeek）使用的位置编码方案。它通过旋转矩阵将位置信息编码到 Query 和 Key 中：

$$
\text{RoPE}(x, m) = \begin{pmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \\ \vdots \end{pmatrix} \otimes \begin{pmatrix} \cos(m\theta_0) \\ \cos(m\theta_0) \\ \cos(m\theta_1) \\ \cos(m\theta_1) \\ \vdots \end{pmatrix} + \begin{pmatrix} -x_1 \\ x_0 \\ -x_3 \\ x_2 \\ \vdots \end{pmatrix} \otimes \begin{pmatrix} \sin(m\theta_0) \\ \sin(m\theta_0) \\ \sin(m\theta_1) \\ \sin(m\theta_1) \\ \vdots \end{pmatrix}
$$

其中 $m$ 是位置索引，$\theta_i = 10000^{-2i/d}$ 是频率参数。

RoPE 的优势在于：两个位置的注意力分数只取决于它们的**相对距离**，而非绝对位置。这使得模型能够更好地泛化到训练时未见过的序列长度。

**位置编码技术对比**：

| 技术 | 原理 | 优势 | 劣势 | 使用模型 |
|:---|:---|:---|:---|:---|
| 正弦编码 | 固定的正弦/余弦函数 | 计算简单，无需训练 | 泛化能力有限 | 原始 Transformer |
| 可学习编码 | 位置作为可学习参数 | 灵活，能学习特定模式 | 无法泛化到训练长度之外 | GPT-2 |
| RoPE | 旋转矩阵编码相对位置 | 支持相对位置，可外推 | 实现较复杂 | Llama, Qwen, DeepSeek |
| ALiBi | 注意力分数加位置偏置 | 无需额外参数，外推性强 | 可能影响远距离注意力 | BLOOM |
| YaRN | RoPE 的扩展版本 | 支持超长上下文 | 需要额外训练 | 部分长上下文模型 |

### 2.2.4 模型架构演进

从 GPT 到现代 LLM，模型架构经历了重要演进。理解这些演进有助于选择合适的模型。

**Transformer 基础架构的完整组件**：

一个完整的 Transformer 解码器层包含以下组件：

1. **多头自注意力（Multi-Head Self-Attention）**：捕捉序列内部的依赖关系
2. **前馈网络（Feed-Forward Network, FFN）**：对注意力输出进行非线性变换
3. **层归一化（Layer Normalization）**：稳定训练过程，加速收敛
4. **残差连接（Residual Connection）**：缓解梯度消失，使深层网络可训练

**前馈网络的数学形式**：

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

现代 LLM 常使用 **SwiGLU** 变体：

$$
\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_3)W_2
$$

其中 $\odot$ 是逐元素乘法，Swish 是激活函数。SwiGLU 在多个基准测试上优于标准 ReLU FFN。

**层归一化的作用**：

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

其中 $\mu$ 和 $\sigma^2$ 是均值和方差，$\gamma$ 和 $\beta$ 是可学习的缩放和偏移参数，$\epsilon$ 是防止除零的小常数。

**残差连接**：

$$
\text{Output} = x + \text{Sublayer}(x)
$$

残差连接使得梯度可以直接流过，缓解了深层网络的梯度消失问题。

| 模型 | 年份 | 架构 | 参数量 | 上下文 | 关键创新 | Agent 影响 |
|:---|:---|:---|:---|:---|:---|:---|
| GPT-3 | 2020 | Decoder-only | 175B | 4K | Few-shot 学习 | 基础推理能力 |
| GPT-3.5 | 2022 | Decoder-only + RLHF | ~175B | 4K/16K | 人类反馈对齐 | 更好的指令遵循 |
| GPT-4 | 2023 | MoE 架构 | ~1.8T (8x220B) | 8K/32K | 多模态、MoE | Function Calling |
| GPT-4o | 2024 | 多模态统一 | 未公开 | 128K | 原生多模态 | 多模态 Agent |
| GPT-4.1 | 2025 | 优化架构 | 未公开 | 1M | 长上下文优化 | 超长文档 Agent |
| Claude 3.5 Sonnet | 2024 | Transformer | 未公开 | 200K | 长上下文优化 | 超长文档处理 |
| Claude 4 Opus | 2025 | 深度推理增强 | 未公开 | 200K | 深度推理 | 复杂规划 Agent |
| Gemini 1.5 Pro | 2024 | MoE + 长上下文 | 未公开 | 1M/2M | 超长上下文 | 超长上下文 Agent |
| Gemini 2.5 Pro | 2025 | 原生多模态 | 未公开 | 1M | 多模态推理 | 多模态 Agent |
| Llama 3.1 | 2024 | Decoder-only | 405B | 128K | 开源顶级 | 本地部署 Agent |
| Llama 4 | 2025 | MoE 开源 | 未公开 | 128K | MoE 开源 | 本地部署 Agent |
| Qwen 2.5 | 2024 | Decoder-only | 72B | 128K | 中文优化 | 中文 Agent |
| DeepSeek V3 | 2025 | MoE | 671B | 128K | 极致性价比 | 低成本 Agent |
| DeepSeek R1 | 2025 | MoE + 推理链 | 671B | 128K | 强化学习推理 | 复杂推理 Agent |

**架构演进的关键趋势**：

1. **从 Dense 到 MoE（Mixture of Experts）**：MoE 架构将大模型拆分为多个"专家"子网络，每次推理只激活其中一部分，大幅降低计算成本。GPT-4、DeepSeek V3 等都采用 MoE 架构。

2. **从短上下文到超长上下文**：从 GPT-3 的 4K 到 Gemini 的 2M，上下文窗口增长了 500 倍。这对 Agent 意味着可以一次性处理更多信息，减少对外部记忆系统的依赖。

3. **从纯文本到多模态**：GPT-4o、Gemini 2.5 等模型原生支持文本、图像、音频、视频的统一处理，使多模态 Agent 成为可能。

4. **从闭源到开源追赶**：Llama 4、Qwen 3、DeepSeek 等开源模型的能力逐渐接近闭源模型，使本地部署 Agent 成为可行方案。

### 2.2.5 预训练-微调-RLHF 对齐流程

现代 LLM 的训练通常分为三个阶段，每个阶段都对 Agent 的行为产生深远影响。

**阶段一：预训练（Pre-training）**

预训练是在海量文本数据上进行的无监督学习，目标是学习语言的统计规律和世界知识。

$$
\mathcal{L}_{\text{pretrain}} = -\sum_{i=1}^{T} \log P(x_i | x_1, \ldots, x_{i-1}; \theta)
$$

即最大化训练数据中下一个 Token 的预测概率。预训练赋予 LLM 语言理解和知识基础，但此时的模型还不能很好地遵循指令。

**阶段二：监督微调（Supervised Fine-Tuning, SFT）**

SFT 使用高质量的指令-回答对来微调预训练模型，使其学会遵循人类指令。对于 Agent 场景，SFT 数据中会包含大量的工具调用示例。

$$
\mathcal{L}_{\text{SFT}} = -\sum_{i=1}^{T} \log P(y_i | x, y_1, \ldots, y_{i-1}; \theta)
$$

其中 $x$ 是指令（Prompt），$y$ 是期望的回答。SFT 使模型从"预测下一个词"转变为"按照指令回答问题"。

**阶段三：人类反馈强化学习（RLHF）**

RLHF 通过人类偏好数据训练一个奖励模型，再用强化学习优化 LLM 的输出，使其更符合人类价值观。

$$
\mathcal{L}_{\text{RLHF}} = \mathbb{E}_{x \sim D, y \sim \pi_\theta} \left[ r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}}) \right]
$$

其中 $r_\phi$ 是奖励模型，$\pi_\theta$ 是当前策略，$\pi_{\text{ref}}$ 是参考策略（通常是 SFT 后的模型），$\beta$ 是 KL 散度惩罚系数，防止模型偏离太远。

**对齐流程对 Agent 行为的影响**：

| 训练阶段 | 对 Agent 的影响 | 具体表现 |
|:---|:---|:---|
| 预训练 | 提供语言理解和知识基础 | 能理解自然语言指令 |
| SFT | 学会遵循指令和格式规范 | 能按要求输出 JSON、调用工具 |
| RLHF | 提高安全性和有用性 | 拒绝有害请求、提供有帮助的回答 |
| RLHF（工具调用） | 提高工具使用的准确性 | 正确选择和调用工具 |

> **工程启示**：在选择模型时，不仅要看基准测试分数，还要关注模型在 Agent 场景下的实际表现。SFT 和 RLHF 的质量直接决定了模型的工具调用准确性和指令遵循能力，这些能力很难从基准测试中完全体现。

---

## 2.3 Token 机制与上下文窗口

### 2.3.1 Token 化算法

LLM 不直接处理文本字符串，而是先将文本转换为**Token（词元）**序列。Token 是模型处理的最小单位，理解 Token 机制对于优化 Agent 的成本和性能至关重要。

**BPE（Byte Pair Encoding）算法**：

BPE 是最常用的 Token 化算法，被 GPT 系列、Claude 等模型采用。其核心思想是：从单个字符开始，反复合并出现频率最高的相邻字符对，直到达到目标词表大小。

BPE 的工作流程：

1. **初始化**：将训练文本中的每个单词拆分为字符序列，添加特殊结束符
2. **统计频率**：统计所有相邻字符对的出现频率
3. **合并**：将出现频率最高的字符对合并为新 Token
4. **更新词表**：将新 Token 加入词表
5. **重复**：重复步骤 2-4，直到达到目标词表大小

**BPE Token 化示例**：

```python
import tiktoken

# 加载 GPT-4o 使用的编码器（cl100k_base）
encoder = tiktoken.encoding_for_model("gpt-4o")

# 英文文本的 Token 化
text_en = "Hello, how are you?"
tokens_en = encoder.encode(text_en)
print(f"英文文本: '{text_en}'")
print(f"Token 数: {len(tokens_en)}")
print(f"Token IDs: {tokens_en}")
print(f"逆向解码: {[encoder.decode([t]) for t in tokens_en]}")
# 输出: ['Hello', ',', ' how', ' are', ' you', '?']
# 英文单词通常被拆分为有意义的子词

# 中文文本的 Token 化
text_cn = "你好，世界！AI Agent 是未来。"
tokens_cn = encoder.encode(text_cn)
print(f"\n中文文本: '{text_cn}'")
print(f"Token 数: {len(tokens_cn)}")
print(f"字符数: {len(text_cn)}")
print(f"Token/字符比: {len(tokens_cn)/len(text_cn):.2f}")
print(f"Token IDs: {tokens_cn}")
print(f"逆向解码: {[encoder.decode([t]) for t in tokens_cn]}")
# 中文字符通常每个占 1-2 个 Token

# 代码文本的 Token 化
text_code = "def calculate_cost(input_tokens, output_tokens):"
tokens_code = encoder.encode(text_code)
print(f"\n代码文本: '{text_code}'")
print(f"Token 数: {len(tokens_code)}")
print(f"逆向解码: {[encoder.decode([t]) for t in tokens_code]}")
# 代码中的关键字和变量名通常被拆分为较小的子词
```

**不同 Token 化算法对比**：

| 算法 | 原理 | 使用模型 | 词表大小 | 特点 |
|:---|:---|:---|:---|:---|
| BPE | 频率最高的字符对合并 | GPT 系列, Claude | ~100K | 平衡效率和覆盖率 |
| WordPiece | 最大化语言模型似然 | BERT | ~30K | 偏向保守的分词 |
| SentencePiece | 语言无关的分词 | Llama, T5 | ~32K | 支持多语言 |
| Unigram | 从大词表中剪枝 | T5, ALBERT | ~32K | 概率模型 |

### 2.3.2 不同语言的 Token 效率对比

Token 效率直接影响 Agent 的成本和上下文窗口利用率。不同语言的 Token 效率差异显著。

| 语言 | Token/字符比 | 100字符的 Token 数 | 成本系数 | 说明 |
|:---|:---|:---|:---|:---|
| 英文 | ~0.25 | ~25 | 1x | 基准，最高效 |
| 中文 | ~1.0-1.5 | ~100-150 | 4-6x | 每个汉字约 1-1.5 Token |
| 日文 | ~1.5-2.0 | ~150-200 | 6-8x | 混合文字系统 |
| 韩文 | ~1.5-2.0 | ~150-200 | 6-8x | 音节文字 |
| 代码 | ~0.3-0.5 | ~30-50 | 1.2-2x | 变量名、关键字 |
| 数字 | ~0.1-0.2 | ~10-20 | 0.4-0.8x | 每个数字约 0.1 Token |
| JSON | ~0.3-0.5 | ~30-50 | 1.2-2x | 结构化数据 |

> **Agent 开发启示**：
> 1. **中文输入的 Token 成本约为英文的 3-4 倍**，在高频调用场景下成本差异显著
> 2. **System Prompt 使用英文可以节省 50%+ 的 Token 成本**，但要确保 Agent 能正确理解中文用户输入
> 3. **工具描述建议使用英文**，因为工具名和参数名本身通常是英文
> 4. **JSON 格式的工具调用**比自然语言描述更节省 Token

### 2.3.3 Token 计数与成本计算

准确计算 Token 数量对于 Agent 的成本控制至关重要。不同模型的定价策略不同，理解定价模型有助于做出明智的模型选择。

```python
def calculate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o") -> dict:
    """计算 API 调用成本（美元）

    Args:
        input_tokens: 输入 Token 数量
        output_tokens: 输出 Token 数量
        model: 模型名称

    Returns:
        包含各项成本的字典
    """
    # 2025年主流模型定价（美元/百万 Token）
    pricing = {
        # OpenAI 模型
        "gpt-4o":           {"input": 2.50, "output": 10.00},
        "gpt-4o-mini":      {"input": 0.15, "output": 0.60},
        "gpt-4.1":          {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini":     {"input": 0.40, "output": 1.60},
        # Anthropic 模型
        "claude-opus-4":    {"input": 15.00, "output": 75.00},
        "claude-sonnet-4":  {"input": 3.00, "output": 15.00},
        "claude-haiku-3.5": {"input": 0.80, "output": 4.00},
        # Google 模型
        "gemini-2.5-pro":   {"input": 1.25, "output": 10.00},
        "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
        # DeepSeek 模型
        "deepseek-v3":      {"input": 0.27, "output": 1.10},
        "deepseek-r1":      {"input": 0.55, "output": 2.19},
    }

    # 获取模型定价，如果未找到则使用 GPT-4o 作为默认
    price = pricing.get(model, pricing["gpt-4o"])

    # 计算各项成本（价格单位是美元/百万Token，需要除以 1,000,000）
    input_cost = input_tokens * price["input"] / 1_000_000
    output_cost = output_tokens * price["output"] / 1_000_000

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "model": model,
    }

# Agent 不同场景的成本分析
scenarios = [
    {"name": "简单问答", "input": 1500, "output": 500},
    {"name": "工具调用 (3步)", "input": 4000, "output": 800},
    {"name": "复杂推理 (5步)", "input": 8000, "output": 2000},
    {"name": "长文档分析", "input": 50000, "output": 3000},
    {"name": "代码 Agent", "input": 10000, "output": 5000},
]

# 分析不同模型在各场景下的成本
models_to_compare = ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4", "deepseek-r1"]

print("=" * 80)
print("Agent 场景成本分析（单次调用）")
print("=" * 80)

for scenario in scenarios:
    print(f"\n场景: {scenario['name']} (输入: {scenario['input']} tokens, 输出: {scenario['output']} tokens)")
    for model in models_to_compare:
        cost = calculate_cost(scenario["input"], scenario["output"], model)
        print(f"  {model:20s}: ${cost['total_cost']:.6f}")

# 月度成本估算（假设每天 1000 次调用）
print("\n" + "=" * 80)
print("月度成本估算（每天 1000 次调用，30 天）")
print("=" * 80)

for model in models_to_compare:
    total_monthly = 0
    for scenario in scenarios:
        cost = calculate_cost(scenario["input"], scenario["output"], model)
        total_monthly += cost["total_cost"] * 1000 * 30
    print(f"{model:20s}: ${total_monthly:.2f}/月")
```

### 2.3.4 上下文窗口管理策略

上下文窗口是 LLM 能一次性处理的最大 Token 数。对于 Agent 来说，上下文窗口需要容纳系统提示、工具定义、对话历史、RAG 上下文、推理中间结果等多个部分。有效的上下文窗口管理是 Agent 性能优化的关键。

**上下文窗口分配模型**：

$$
\text{可用上下文} = \text{窗口大小} - \text{系统提示} - \text{工具定义} - \text{输出预留}
$$

**可用上下文的进一步分配**：

$$
\text{可用上下文} = \text{对话历史} + \text{RAG 上下文} + \text{推理暂存区}
$$

```python
class ContextWindowManager:
    """Agent 上下文窗口管理器

    负责管理 LLM 的上下文窗口分配，确保各组件不会超出窗口限制。
    支持多种压缩策略来最大化信息密度。
    """

    def __init__(self, max_tokens: int = 128000, reserved: int = 4096):
        """初始化上下文窗口管理器

        Args:
            max_tokens: 模型的最大上下文窗口大小
            reserved: 预留给输出的 Token 数量
        """
        self.max_tokens = max_tokens
        self.reserved = reserved
        # 可用 Token = 最大窗口 - 输出预留
        self.available = max_tokens - reserved

    def allocate(self, system_prompt: int, tools: int) -> dict:
        """分配上下文窗口空间

        Args:
            system_prompt: 系统提示占用的 Token 数
            tools: 工具定义占用的 Token 数

        Returns:
            各组件的 Token 配额分配方案
        """
        # 计算剩余可用空间
        remaining = self.available - system_prompt - tools

        # 按比例分配剩余空间
        # 对话历史占 60%（最重要的信息来源）
        # RAG 上下文占 30%（外部知识检索结果）
        # 推理暂存区占 10%（Chain-of-Thought 等中间推理）
        allocation = {
            "chat_history": int(remaining * 0.6),
            "rag_context": int(remaining * 0.3),
            "scratchpad": int(remaining * 0.1),
            "total_available": remaining,
            "system_prompt": system_prompt,
            "tools": tools,
            "reserved_output": self.reserved,
        }

        return allocation

    def estimate_tokens(self, text: str) -> int:
        """估算文本的 Token 数量

        粗略估算：英文约 4 字符/Token，中文约 1.5 字符/Token
        更精确的估算需要使用 tiktoken 等库

        Args:
            text: 输入文本

        Returns:
            估算的 Token 数量
        """
        # 简单估算：按字符数除以 3.5（中英混合的平均值）
        return len(text) // 3

    def truncate_history(self, messages: list, max_tokens: int) -> list:
        """截断对话历史以适应 Token 限制

        策略：保留系统提示 + 最近的对话轮次

        Args:
            messages: 完整的消息列表
            max_tokens: 允许的最大 Token 数

        Returns:
            截断后的消息列表
        """
        # 从最新的消息开始向前保留
        truncated = []
        current_tokens = 0

        # 从后往前遍历，保留最新的消息
        for msg in reversed(messages):
            msg_tokens = self.estimate_tokens(msg.get("content", ""))
            if current_tokens + msg_tokens > max_tokens:
                break
            truncated.insert(0, msg)
            current_tokens += msg_tokens

        return truncated

    def summarize_and_compress(self, old_messages: list) -> str:
        """将旧消息压缩为摘要

        这是一个示意性方法，实际实现需要调用 LLM 进行摘要

        Args:
            old_messages: 需要压缩的旧消息

        Returns:
            压缩后的摘要文本
        """
        # 实际实现中，这里会调用 LLM 来生成摘要
        # 例如：调用一个轻量级模型来总结对话历史
        summary = "[对话历史摘要] 用户讨论了以下话题：..."
        return summary


# 使用示例
manager = ContextWindowManager(max_tokens=128000, reserved=4096)

# 计算各组件的 Token 占用
system_prompt_tokens = manager.estimate_tokens("你是一个数据分析 Agent...")
tools_tokens = manager.estimate_tokens('{"name": "query_database", ...}')

# 分配上下文窗口
allocation = manager.allocate(system_prompt_tokens, tools_tokens)
print(f"系统提示: {allocation['system_prompt']} tokens")
print(f"工具定义: {allocation['tools']} tokens")
print(f"对话历史配额: {allocation['chat_history']} tokens")
print(f"RAG 上下文配额: {allocation['rag_context']} tokens")
print(f"推理暂存区: {allocation['scratchpad']} tokens")
```

### 2.3.5 长上下文技术

随着 Agent 应用对上下文窗口需求的增长，长上下文技术变得越来越重要。这些技术使 LLM 能够处理远超训练长度的输入。

**RoPE 外推技术**：

| 技术 | 原理 | 支持倍数 | 效果 | 使用模型 |
|:---|:---|:---|:---|:---|
| Position Interpolation | 线性插值位置编码 | 2-4x | 质量损失小 | Llama 2 |
| NTK-aware Scaling | 非均匀缩放频率 | 4-8x | 平衡质量和长度 | 部分开源模型 |
| YaRN | RoPE 的扩展版本 | 8-16x | 超长上下文 | 部分开源模型 |
| LongRoPE | 自适应频率缩放 | 16-64x | 超长上下文 | 研究模型 |
| Ring Attention | 分布式注意力计算 | 任意 | 需要多 GPU | 研究框架 |

**长上下文对 Agent 的实际意义**：

| 场景 | 传统窗口（4K-8K） | 长上下文（128K+） | 超长上下文（1M+） |
|:---|:---|:---|:---|
| 单轮对话 | 足够 | 足够 | 足够 |
| 多轮对话 | 需要频繁截断 | 可保持完整历史 | 完整历史 + 丰富上下文 |
| 文档分析 | 需要分块处理 | 可处理中等文档 | 可处理完整书籍 |
| 代码库理解 | 只能看单文件 | 可看多个文件 | 可看整个代码库 |
| RAG 增强 | 只能注入少量片段 | 可注入大量检索结果 | 可注入完整文档集 |

> **实践建议**：长上下文并不意味着应该把所有信息都塞进 Prompt。研究表明，模型对上下文中间部分的注意力往往较弱（"Lost in the Middle"现象）。因此，即使有 1M 的上下文窗口，也建议将最关键的信息放在 Prompt 的开头和结尾。

### 2.3.6 Lost in the Middle 问题与缓解

"Lost in the Middle" 是指 LLM 在处理长上下文时，对中间部分信息的利用率显著低于开头和结尾的现象。这一问题对 Agent 的信息检索和决策质量有直接影响。

**Lost in the Middle 的实验发现**：

研究人员通过在上下文的不同位置插入关键信息来测试模型的利用率：

| 关键信息位置 | 模型准确率 | 相对性能 |
|:---|:---|:---|
| 开头（Top 10%） | 95% | 100%（基准） |
| 结尾（Bottom 10%） | 90% | 95% |
| 中间（50% 位置） | 70% | 74% |
| 前 25% | 92% | 97% |
| 后 25% | 88% | 93% |

这一现象在所有主流 LLM 中都存在，只是程度不同。对于 Agent 来说，这意味着如果关键信息（如工具定义、重要约束）被放在上下文的中间位置，模型可能会"忽略"这些信息。

**Agent 场景的缓解策略**：

| 策略 | 描述 | 实现方式 | 效果 |
|:---|:---|:---|:---|
| **重要信息前置** | 将 System Prompt 和工具定义放在最前面 | 调整消息顺序 | 高 |
| **重复关键信息** | 在开头和结尾重复关键约束 | Prompt 模板设计 | 中 |
| **分块处理** | 将长上下文分成多个块分别处理 | 迭代式 Agent 架构 | 高 |
| **信息分层** | 按重要性分层，重要信息放首尾 | 上下文管理器 | 中 |
| **注意力引导** | 使用特殊标记引导模型关注 | Prompt Engineering | 低-中 |

```python
# 上下文排列优化示例
def optimize_context_order(messages: list) -> list:
    """优化上下文排列顺序，缓解 Lost in the Middle 问题

    策略：
    1. System Prompt 始终在最前面
    2. 工具定义紧跟 System Prompt
    3. 最重要的信息（用户核心请求）放在中间偏前
    4. 对话历史放在后面
    5. 最近的消息放在最后（利用 Recency Bias）

    Args:
        messages: 原始消息列表

    Returns:
        优化后的消息列表
    """
    system_messages = []
    tool_messages = []
    user_messages = []
    assistant_messages = []

    # 按角色分类
    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            system_messages.append(msg)
        elif role == "tool":
            tool_messages.append(msg)
        elif role == "user":
            user_messages.append(msg)
        elif role == "assistant":
            assistant_messages.append(msg)

    # 重新排列：system → tools → 最近的 user → 其他 user → assistant 历史
    # 这样确保 System Prompt 和工具定义在最前面，
    # 用户的核心请求在中间偏前，对话历史在后面
    optimized = []
    optimized.extend(system_messages)     # 1. System Prompt（最前面）
    optimized.extend(tool_messages)       # 2. 工具定义
    if user_messages:
        optimized.append(user_messages[-1])  # 3. 最新的用户消息
        optimized.extend(user_messages[:-1])  # 4. 之前的用户消息
    optimized.extend(assistant_messages)  # 5. 助手历史

    return optimized
```

### 2.3.7 Token 级操作：截断与压缩策略

当对话历史或 RAG 上下文超出上下文窗口限制时，需要进行截断或压缩。不同的策略对 Agent 的信息保留和性能有不同的影响。

**截断策略对比**：

| 策略 | 描述 | 信息保留 | 实现复杂度 | 适用场景 |
|:---|:---|:---|:---|:---|
| **固定窗口截断** | 只保留最近 N 条消息 | 低 | 低 | 简单对话 Agent |
| **Token 计数截断** | 保留最近的 N 个 Token | 中 | 低 | 通用场景 |
| **重要性评分截断** | 按重要性保留消息 | 高 | 中 | 信息密集型 Agent |
| **摘要压缩** | 将旧消息压缩为摘要 | 中-高 | 中 | 长对话 Agent |
| **混合策略** | 分层保留 + 压缩 | 高 | 高 | 生产级 Agent |

```python
class SmartContextManager:
    """智能上下文管理器

    实现多种上下文管理策略，根据场景选择最优方案。
    """

    def __init__(self, max_tokens: int = 128000):
        self.max_tokens = max_tokens

    def truncate_by_recency(self, messages: list, max_tokens: int) -> list:
        """基于时间顺序的截断策略

        保留最新的消息，直到达到 Token 限制。
        这是最简单的策略，适合大多数场景。

        Args:
            messages: 完整的消息列表
            max_tokens: Token 限制

        Returns:
            截断后的消息列表
        """
        result = []
        current_tokens = 0

        # 从最新消息开始向前保留
        for msg in reversed(messages):
            msg_tokens = self._estimate_tokens(msg)
            if current_tokens + msg_tokens > max_tokens:
                break
            result.insert(0, msg)
            current_tokens += msg_tokens

        return result

    def truncate_by_importance(self, messages: list, max_tokens: int) -> list:
        """基于重要性的截断策略

        为每条消息计算重要性分数，保留最重要的消息。
        重要性评分考虑：消息类型、是否包含工具调用、关键词等。

        Args:
            messages: 完整的消息列表
            max_tokens: Token 限制

        Returns:
            截断后的消息列表
        """
        # 计算每条消息的重要性分数
        scored_messages = []
        for i, msg in enumerate(messages):
            score = self._calculate_importance(msg, i, len(messages))
            scored_messages.append((score, i, msg))

        # 按重要性降序排列
        scored_messages.sort(key=lambda x: x[0], reverse=True)

        # 贪心选择：按重要性依次加入，直到达到 Token 限制
        selected = []
        current_tokens = 0
        for score, original_idx, msg in scored_messages:
            msg_tokens = self._estimate_tokens(msg)
            if current_tokens + msg_tokens <= max_tokens:
                selected.append((original_idx, msg))
                current_tokens += msg_tokens

        # 按原始顺序排列
        selected.sort(key=lambda x: x[0])
        return [msg for _, msg in selected]

    def _calculate_importance(self, msg: dict, index: int, total: int) -> float:
        """计算消息的重要性分数

        评分因素：
        - 消息类型：工具调用结果 > 用户消息 > 助手消息
        - 位置：最近的消息更重要
        - 内容：包含关键信息的消息更重要

        Args:
            msg: 消息字典
            index: 消息在列表中的位置
            total: 消息总数

        Returns:
            重要性分数（0-1）
        """
        score = 0.0

        # 基础分数：基于消息类型
        role = msg.get("role", "")
        if role == "tool":
            score += 0.8  # 工具调用结果最重要
        elif role == "user":
            score += 0.6  # 用户消息次重要
        elif role == "assistant":
            score += 0.4  # 助手消息
        elif role == "system":
            score += 1.0  # System Prompt 最重要

        # 位置分数：越新的消息越重要
        position_score = index / max(total, 1)
        score += position_score * 0.3

        # 内容分数：包含关键词的消息更重要
        content = str(msg.get("content", ""))
        important_keywords = ["错误", "失败", "重要", "关键", "注意", "error", "critical"]
        for keyword in important_keywords:
            if keyword in content.lower():
                score += 0.1

        return min(score, 1.0)

    def _estimate_tokens(self, msg: dict) -> int:
        """估算消息的 Token 数量"""
        content = str(msg.get("content", ""))
        return len(content) // 3  # 粗略估算
```

---

## 2.4 模型参数与 Agent 行为

### 2.4.1 Temperature 的影响

Temperature 是控制 LLM 输出随机性的核心参数。它直接影响 Agent 的行为模式——从确定性的工具调用到创造性的内容生成。

Temperature 的数学定义：

$$
P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

其中 $z_i$ 是 Token $i$ 的原始分数（logit），$T$ 是温度参数。

**Temperature 对概率分布的影响**：

| Temperature | 数学效果 | 概率分布特征 | 行为表现 |
|:---|:---|:---|:---|
| $T \to 0$ | 放大差异 | 接近 one-hot（最高概率 ≈ 1） | 确定性输出，总是选最可能的 Token |
| $T = 0.3$ | 略微平滑 | 尖锐分布，Top-1 占主导 | 低随机性，偶尔选次优 Token |
| $T = 0.7$ | 中等平滑 | 中等分布 | 适度随机性，有一定多样性 |
| $T = 1.0$ | 保持原始分布 | 原始 softmax 分布 | 正常随机性 |
| $T > 1.0$ | 过度平滑 | 接近均匀分布 | 高随机性，输出可能不连贯 |

**不同 Agent 场景的 Temperature 推荐**：

| 场景 | 推荐 Temperature | 理由 |
|:---|:---|:---|
| **工具调用（Function Calling）** | $T = 0$ | 必须确定性输出，确保 JSON 格式正确 |
| **代码生成** | $T = 0$ | 代码必须精确，不能有随机性 |
| **数据提取** | $T = 0$ | 结构化输出必须准确 |
| **事实性问答** | $T = 0.3$ | 低随机性确保准确性 |
| **通用对话** | $T = 0.7$ | 适度多样性提升体验 |
| **创意写作** | $T = 0.9-1.0$ | 高多样性产生创意内容 |
| **头脑风暴** | $T = 1.0$ | 最大化多样性 |
| **自我反思** | $T = 0.5$ | 平衡准确性和创造性 |

> **Agent 最佳实践**：在 Agent 系统中，**工具调用场景始终使用 `temperature=0`**。这是因为工具调用需要精确的 JSON 格式，任何随机性都可能导致格式错误，进而导致整个 Agent 流程失败。对于非工具调用的推理步骤，可以使用 `temperature=0.3-0.7` 来平衡准确性和多样性。

### 2.4.2 Top-p 与 Top-k 采样策略

除了 Temperature，Top-p（Nucleus Sampling）和 Top-k 也是控制输出多样性的重要参数。

**Top-k 采样**：

$$
P'(x_i) = \begin{cases} \frac{P(x_i)}{\sum_{j \in \text{Top-k}} P(x_j)} & \text{if } x_i \in \text{Top-k} \\ 0 & \text{otherwise} \end{cases}
$$

只从概率最高的 $k$ 个 Token 中采样，其余 Token 的概率设为 0。

**Top-p（Nucleus）采样**：

$$
P'(x_i) = \begin{cases} \frac{P(x_i)}{\sum_{j \in V_p} P(x_j)} & \text{if } x_i \in V_p \\ 0 & \text{otherwise} \end{cases}
$$

其中 $V_p$ 是累积概率超过 $p$ 的最小 Token 集合。

**参数组合推荐**：

| 场景 | Temperature | Top-p | Top-k | 理由 |
|:---|:---|:---|:---|:---|
| 工具调用 | 0 | 1.0 | - | 确定性输出 |
| 代码生成 | 0 | 1.0 | - | 确定性输出 |
| 事实问答 | 0.3 | 0.9 | 40 | 低随机性 |
| 通用对话 | 0.7 | 0.9 | 50 | 平衡多样性 |
| 创意写作 | 0.9 | 0.95 | 100 | 高多样性 |
| 头脑风暴 | 1.0 | 1.0 | - | 最大多样性 |

 > **使用建议**：在 Agent 开发中，通常只需要调整 Temperature。Top-p 和 Top-k 保持默认值（Top-p=1.0）即可。同时调整多个采样参数可能导致意外行为。经验法则是：**先调 Temperature，不够再调 Top-p**。

不同的推理策略适用于不同的场景和模型配置。下面的交互式工具可以帮助你根据具体需求选择最合适的推理策略：

<div data-component="ReasoningStrategySelectorV4"></div>

### 2.4.3 max_tokens 与 stop sequences

**max_tokens** 控制模型输出的最大 Token 数。设置不当会导致输出截断或浪费资源。

| max_tokens 设置 | 效果 | 适用场景 |
|:---|:---|:---|
| 256 | 短输出 | 简单分类、标签生成 |
| 1024 | 中等输出 | 工具调用、数据提取 |
| 4096 | 长输出 | 代码生成、详细分析 |
| 8192 | 超长输出 | 长文档生成、完整报告 |
| 16384 | 极长输出 | 代码库重构、完整项目 |

**Stop Sequences（停止序列）**：

Stop Sequences 告诉模型在生成特定字符串时停止输出。这对于控制 Agent 的输出格式非常有用。

```python
# 常见的 Stop Sequences 使用场景

# 场景 1: 工具调用后停止
# 模型可能在工具调用后继续生成解释文本
# 使用 "\n\n" 或特定标记来停止
stop_sequences_tool = ["\n\nObservation:", "\n\nThought:"]

# 场景 2: JSON 输出后停止
# 确保只输出 JSON，不附加额外文本
stop_sequences_json = ["```", "\n\n"]

# 场景 3: ReAct 格式中停止
# 在 Observation 出现时停止，等待工具执行结果
stop_sequences_react = ["Observation:", "Observation:\n"]
```

### 2.4.4 Response Format 选择

现代 LLM API 支持多种输出格式，选择合适的格式对 Agent 的可靠性至关重要。

| 格式 | 说明 | 优点 | 缺点 | 适用场景 |
|:---|:---|:---|:---|:---|
| **Text** | 纯文本输出 | 灵活，无限制 | 需要后处理解析 | 通用对话 |
| **JSON** | JSON 对象输出 | 结构化，易于解析 | 可能格式错误 | 数据提取 |
| **JSON Schema** | 严格 JSON Schema 约束 | 格式保证正确 | 灵活性低 | Function Calling |
| **Function Calling** | 模型原生工具调用 | 最可靠，格式保证 | 需要 API 支持 | Agent 工具调用 |
| **Streaming** | 流式输出 | 低延迟，用户体验好 | 解析复杂 | 实时交互 |

```python
from openai import OpenAI

client = OpenAI()

# 方式 1: 纯文本输出（最灵活但最不可靠）
response_text = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "提取姓名和年龄"}],
    temperature=0,
)
# 需要自己解析文本中的结构化信息

# 方式 2: JSON 模式（保证输出是合法 JSON）
response_json = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "提取姓名和年龄，输出 JSON"}],
    response_format={"type": "json_object"},
    temperature=0,
)
# 保证输出是合法 JSON，但不保证符合特定 Schema

# 方式 3: Structured Outputs（最可靠，保证符合 Schema）
from pydantic import BaseModel

class PersonInfo(BaseModel):
    name: str
    age: int

response_structured = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "提取姓名和年龄"}],
    response_format=PersonInfo,
    temperature=0,
)
# 保证输出完全符合 PersonInfo 的 Schema

# 方式 4: Function Calling（Agent 最常用）
tools = [{
    "type": "function",
    "function": {
        "name": "extract_info",
        "description": "从文本中提取人员信息",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "姓名"},
                "age": {"type": "integer", "description": "年龄"},
            },
            "required": ["name", "age"],
        },
    },
}]
response_fc = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "张三今年 25 岁"}],
    tools=tools,
    tool_choice="auto",
    temperature=0,
)
# 模型会自动选择是否调用工具，格式完全可靠
```

> **Agent 选型建议**：对于 Agent 开发，**始终优先使用 Function Calling**。它是目前最可靠的结构化输出方式，由模型原生支持，格式保证正确。只有在 Function Calling 不可用或场景特殊时，才考虑 JSON Schema 或 JSON 模式。

---

## 2.5 主流模型对比与选型

### 2.5.1 闭源模型详细对比

| 模型 | 厂商 | 参数量 | 上下文 | 推理 | 工具调用 | 代码 | 中文 | 输入价格 | 输出价格 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| GPT-4o | OpenAI | 未公开 | 128K | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★ | $2.5/M | $10/M |
| GPT-4o-mini | OpenAI | 未公开 | 128K | ★★★★ | ★★★★ | ★★★★ | ★★★ | $0.15/M | $0.6/M |
| GPT-4.1 | OpenAI | 未公开 | 1M | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★ | $2/M | $8/M |
| GPT-4.1-mini | OpenAI | 未公开 | 1M | ★★★★ | ★★★★ | ★★★★ | ★★★ | $0.40/M | $1.60/M |
| Claude Opus 4 | Anthropic | 未公开 | 200K | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★ | $15/M | $75/M |
| Claude Sonnet 4 | Anthropic | 未公开 | 200K | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | $3/M | $15/M |
| Claude Haiku 3.5 | Anthropic | 未公开 | 200K | ★★★★ | ★★★★ | ★★★★ | ★★★★ | $0.80/M | $4/M |
| Gemini 2.5 Pro | Google | 未公开 | 1M | ★★★★★ | ★★★★ | ★★★★★ | ★★★★ | $1.25/M | $10/M |
| Gemini 2.5 Flash | Google | 未公开 | 1M | ★★★★ | ★★★★ | ★★★★ | ★★★ | $0.15/M | $0.60/M |

**闭源模型的 Agent 特性对比**：

| 特性 | GPT-4o | Claude Sonnet 4 | Gemini 2.5 Pro |
|:---|:---|:---|:---|
| Function Calling | 原生支持，最成熟 | 原生支持 | 原生支持 |
| Structured Output | JSON Schema 保证 | Tool Use 保证 | JSON Schema 保证 |
| 并行工具调用 | 支持 | 支持 | 支持 |
| 流式输出 | 支持 | 支持 | 支持 |
| 多模态输入 | 文本+图像+音频 | 文本+图像 | 文本+图像+视频+音频 |
| System Prompt 长度 | ~16K tokens | ~100K tokens | ~32K tokens |
| 延迟（首 Token） | ~0.5s | ~0.8s | ~0.6s |
| API 可用性 | 全球 | 全球（部分地区受限） | 全球 |

### 2.5.2 开源模型详细对比

| 模型 | 厂商 | 参数量 | 架构 | 上下文 | 推理 | 工具调用 | 代码 | 中文 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Llama 4 Maverick | Meta | 400B (MoE) | MoE | 1M | ★★★★ | ★★★★ | ★★★★ | ★★★ |
| Llama 4 Scout | Meta | 109B (MoE) | MoE | 10M | ★★★★ | ★★★★ | ★★★★ | ★★★ |
| Qwen 2.5 72B | Alibaba | 72B | Dense | 128K | ★★★★ | ★★★★ | ★★★★ | ★★★★★ |
| Qwen 3 235B | Alibaba | 235B (MoE) | MoE | 128K | ★★★★★ | ★★★★ | ★★★★★ | ★★★★★ |
| DeepSeek V3 | DeepSeek | 671B (MoE) | MoE | 128K | ★★★★★ | ★★★★ | ★★★★★ | ★★★★★ |
| DeepSeek R1 | DeepSeek | 671B (MoE) | MoE | 128K | ★★★★★ | ★★★ | ★★★★★ | ★★★★★ |
| Mistral Large | Mistral | 123B | MoE | 128K | ★★★★ | ★★★★ | ★★★★ | ★★★ |

**开源模型的 Agent 适用性分析**：

| 模型 | 最佳 Agent 场景 | 本地部署需求 | 推荐 GPU |
|:---|:---|:---|:---|
| Llama 4 Scout | 通用 Agent、多模态 | 4x A100 80G | 高 |
| Qwen 2.5 72B | 中文 Agent、代码 Agent | 2x A100 80G | 中高 |
| Qwen 3 235B | 复杂推理 Agent | 4x A100 80G | 高 |
| DeepSeek V3 | 低成本高性能 Agent | 8x A100 80G | 极高 |
| DeepSeek R1 | 复杂推理 Agent | 8x A100 80G | 极高 |

### 2.5.3 模型能力详细对比表

**Agent 核心能力维度对比**：

| 维度 | GPT-4o | Claude Sonnet 4 | Gemini 2.5 Pro | DeepSeek R1 | Qwen 2.5 72B |
|:---|:---|:---|:---|:---|:---|
| **指令遵循** | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ | ★★★★ |
| **多步推理** | ★★★★ | ★★★★★ | ★★★★ | ★★★★★ | ★★★★ |
| **工具选择** | ★★★★★ | ★★★★★ | ★★★★ | ★★★ | ★★★★ |
| **参数填充** | ★★★★★ | ★★★★★ | ★★★★ | ★★★ | ★★★★ |
| **错误恢复** | ★★★★ | ★★★★★ | ★★★ | ★★★★ | ★★★ |
| **代码生成** | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★ |
| **结构化输出** | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ | ★★★★ |
| **长上下文** | ★★★★ | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ |
| **多模态理解** | ★★★★★ | ★★★★ | ★★★★★ | ★★★ | ★★★★ |
| **中文能力** | ★★★★ | ★★★★★ | ★★★★ | ★★★★★ | ★★★★★ |
| **成本效益** | ★★★ | ★★★ | ★★★★ | ★★★★★ | ★★★★ |
| **响应速度** | ★★★★ | ★★★★ | ★★★★ | ★★★ | ★★★★ |

### 2.5.4 Agent 场景选型指南

| 场景 | 推荐模型 | 备选模型 | 理由 |
|:---|:---|:---|:---|
| **通用 Agent** | GPT-4o / Claude Sonnet 4 | Gemini 2.5 Pro | 综合能力最强 |
| **代码 Agent** | Claude Sonnet 4 / DeepSeek R1 | GPT-4o | 代码和推理能力突出 |
| **长文档 Agent** | Gemini 2.5 Pro / GPT-4.1 | Claude Sonnet 4 | 1M 上下文窗口 |
| **多模态 Agent** | GPT-4o / Gemini 2.5 Pro | Claude Sonnet 4 | 多模态理解能力 |
| **低成本 Agent** | GPT-4o-mini / DeepSeek V3 | Gemini 2.5 Flash | 性价比最高 |
| **中文 Agent** | DeepSeek R1 / Qwen 2.5 72B | Claude Sonnet 4 | 中文能力最强 |
| **本地部署 Agent** | Llama 4 / Qwen 3 | DeepSeek V3 | 开源、可控 |
| **高频调用 Agent** | GPT-4o-mini / Gemini 2.5 Flash | DeepSeek V3 | 低延迟、低成本 |
| **复杂规划 Agent** | Claude Opus 4 / GPT-4o | DeepSeek R1 | 深度推理能力 |

**模型选择决策树**：

```
需要本地部署？
├── 是 → Llama 4 / Qwen 3 / DeepSeek
└── 否 → 预算充足？
    ├── 是 → 任务复杂度高？
    │   ├── 是 → Claude Opus 4 / GPT-4o
    │   └── 否 → Claude Sonnet 4 / GPT-4o
    └── 否 → 需要长上下文？
        ├── 是 → Gemini 2.5 Flash / GPT-4.1-mini
        └── 否 → GPT-4o-mini / DeepSeek V3
```

**模型选型的常见误区**：

| 误区 | 正确理解 | 建议 |
|:---|:---|:---|
| "参数量越大越好" | MoE 模型激活参数远小于总参数 | 关注激活参数量和实际性能 |
| "基准测试分数高就适合 Agent" | Agent 场景需要工具调用、指令遵循等特定能力 | 用实际 Agent 场景测试 |
| "最贵的模型最好" | 性价比因场景而异 | 根据任务复杂度选择 |
| "开源模型不如闭源" | 顶级开源模型已接近闭源水平 | 根据具体需求评估 |
| "长上下文可以塞满信息" | Lost in the Middle 现象依然存在 | 优化信息排列顺序 |
| "一个模型解决所有问题" | 不同任务适合不同模型 | 实现模型路由策略 |

> **最佳实践**：在生产环境中，建议实现**模型路由（Model Routing）**策略——根据任务复杂度自动选择合适的模型。简单任务用小模型（低成本、低延迟），复杂任务用大模型（高质量、高成本）。这种策略可以在保证质量的同时显著降低成本。

---

## 2.6 API 调用最佳实践

### 2.6.1 OpenAI API 完整示例（带重试）

```python
import os
import json
import time
from openai import OpenAI, APITimeoutError, APIStatusError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# 初始化 OpenAI 客户端
# 建议通过环境变量设置 API Key，不要硬编码
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=30.0,  # 设置全局超时时间
)


@retry(
    stop=stop_after_attempt(3),           # 最多重试 3 次
    wait=wait_exponential(                # 指数退避等待
        multiplier=1, min=1, max=10       # 1s, 2s, 4s
    ),
    retry=retry_if_exception_type(        # 只在特定异常时重试
        (APITimeoutError, APIStatusError)
    ),
)
def call_openai_with_retry(
    messages: list,
    tools: list = None,
    model: str = "gpt-4o",
    temperature: float = 0,
    max_tokens: int = 4096,
) -> dict:
    """带重试机制的 OpenAI API 调用

    Args:
        messages: 消息列表，包含 system、user、assistant 等角色的消息
        tools: 工具定义列表（可选）
        model: 模型名称
        temperature: 温度参数，0 表示确定性输出
        max_tokens: 最大输出 Token 数

    Returns:
        包含响应内容和元数据的字典
    """
    # 构建请求参数
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # 如果提供了工具定义，添加到请求中
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"  # 让模型自主决定是否调用工具

    try:
        # 发送 API 请求
        response = client.chat.completions.create(**kwargs)

        # 提取响应信息
        message = response.choices[0].message
        usage = response.usage

        return {
            "content": message.content,
            "tool_calls": message.tool_calls,
            "finish_reason": response.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
            "model": response.model,
        }

    except APITimeoutError:
        # 超时异常，tenacity 会自动重试
        raise
    except APIStatusError as e:
        # API 状态错误（如 429 速率限制、500 服务器错误）
        if e.status_code in [429, 500, 502, 503]:
            raise  # 这些错误值得重试
        else:
            raise  # 其他错误也抛出，由 tenacity 决定是否重试


# Agent 工具定义示例
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "在数据库中搜索信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "执行 Python 代码并返回结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要执行的 Python 代码",
                    },
                },
                "required": ["code"],
            },
        },
    },
]

# Agent 对话示例
messages = [
    {
        "role": "system",
        "content": "你是一个数据分析 Agent，能够搜索数据库和执行代码来分析数据。",
    },
    {
        "role": "user",
        "content": "帮我查询上个月的销售数据，并计算同比增长率",
    },
]

# 调用 API
try:
    result = call_openai_with_retry(messages=messages, tools=tools)

    print(f"模型响应: {result['content']}")
    print(f"完成原因: {result['finish_reason']}")
    print(f"Token 使用: {result['usage']}")

    # 检查是否有工具调用
    if result["tool_calls"]:
        for tool_call in result["tool_calls"]:
            print(f"\n工具调用: {tool_call.function.name}")
            print(f"参数: {tool_call.function.arguments}")

except Exception as e:
    print(f"API 调用失败: {e}")
```

### 2.6.2 Anthropic API 完整示例

```python
import os
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

# 初始化 Anthropic 客户端
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def call_claude_with_retry(
    messages: list,
    tools: list = None,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
    temperature: float = 0,
) -> dict:
    """带重试机制的 Anthropic API 调用

    Anthropic API 与 OpenAI API 的主要区别：
    1. System Prompt 作为单独参数传递，不在 messages 中
    2. 工具定义格式略有不同
    3. 响应格式不同（content 是列表而非字符串）
    """
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Anthropic 的 System Prompt 是单独的参数
    # 从 messages 中提取 system 消息
    system_msg = None
    filtered_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        else:
            filtered_messages.append(msg)

    if system_msg:
        kwargs["system"] = system_msg
    kwargs["messages"] = filtered_messages

    # 添加工具定义（Anthropic 格式）
    if tools:
        kwargs["tools"] = tools

    response = client.messages.create(**kwargs)

    # Anthropic 响应的 content 是一个列表
    # 需要遍历提取文本和工具调用
    text_content = ""
    tool_calls = []

    for block in response.content:
        if block.type == "text":
            text_content += block.text
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })

    return {
        "content": text_content,
        "tool_calls": tool_calls,
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


# Anthropic 工具定义格式
anthropic_tools = [
    {
        "name": "search_database",
        "description": "在数据库中搜索信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
]

# 使用示例
messages = [
    {"role": "user", "content": "查询上个月的销售数据"},
]

try:
    result = call_claude_with_retry(
        messages=messages,
        tools=anthropic_tools,
    )
    print(f"Claude 响应: {result['content']}")
    print(f"工具调用: {result['tool_calls']}")
except Exception as e:
    print(f"API 调用失败: {e}")
```

### 2.6.3 LangChain 统一接口

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

# 创建模型实例
# LangChain 提供了统一的接口来调用不同的 LLM
gpt4 = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=4096,
    timeout=30,
    max_retries=3,
)

claude = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0,
    max_tokens=4096,
    timeout=30,
    max_retries=3,
)

# 定义工具
# LangChain 的 @tool 装饰器简化了工具定义
@tool
def search_database(query: str, limit: int = 10) -> str:
    """在数据库中搜索信息

    Args:
        query: 搜索关键词
        limit: 返回结果数量
    """
    # 实际实现中这里会执行数据库查询
    return f"找到 {limit} 条关于 '{query}' 的结果"


@tool
def execute_code(code: str) -> str:
    """执行 Python 代码并返回结果

    Args:
        code: 要执行的 Python 代码
    """
    # 实际实现中这里会在沙箱中执行代码
    return f"代码执行结果: executed"


# 统一的 Agent 调用接口
def call_llm(
    model_name: str,
    messages: list,
    tools: list = None,
) -> dict:
    """统一的 LLM 调用接口

    通过 LangChain 的统一接口，可以轻松切换不同的 LLM，
    而无需修改上层代码。

    Args:
        model_name: 模型名称（gpt4 或 claude）
        messages: 消息列表
        tools: 工具列表

    Returns:
        模型响应
    """
    # 根据模型名称选择对应的 LLM 实例
    model_map = {
        "gpt4": gpt4,
        "claude": claude,
    }
    llm = model_map[model_name]

    # 如果有工具，绑定到 LLM 上
    if tools:
        llm = llm.bind_tools(tools)

    # 调用 LLM
    response = llm.invoke(messages)

    return {
        "content": response.content,
        "tool_calls": response.tool_calls if hasattr(response, "tool_calls") else [],
    }


# 使用示例
messages = [
    SystemMessage(content="你是一个数据分析 Agent。"),
    HumanMessage(content="帮我分析上个月的销售数据"),
]

# 使用 GPT-4o
result_gpt = call_llm("gpt4", messages, tools=[search_database, execute_code])
print(f"GPT-4o 响应: {result_gpt['content']}")

# 使用 Claude（只需切换模型名称，其余代码不变）
result_claude = call_llm("claude", messages, tools=[search_database, execute_code])
print(f"Claude 响应: {result_claude['content']}")
```

### 2.6.4 错误处理策略

Agent 系统中的 API 错误处理需要考虑多种场景：网络超时、速率限制、模型过载、API 密钥无效、内容过滤等。

```python
import time
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """API 错误类型枚举"""
    TIMEOUT = "timeout"              # 请求超时
    RATE_LIMIT = "rate_limit"        # 速率限制（429）
    SERVER_ERROR = "server_error"    # 服务器错误（5xx）
    AUTH_ERROR = "auth_error"        # 认证错误（401）
    INVALID_REQUEST = "invalid_request"  # 无效请求（400）
    CONTENT_FILTER = "content_filter"    # 内容过滤
    CONTEXT_LENGTH = "context_length"    # 上下文超长
    UNKNOWN = "unknown"              # 未知错误


@dataclass
class APIError:
    """API 错误信息"""
    error_type: ErrorType
    message: str
    status_code: Optional[int] = None
    retry_after: Optional[float] = None  # 建议重试等待时间（秒）


def classify_error(error: Exception) -> APIError:
    """将异常分类为具体的错误类型

    Args:
        error: 捕获的异常

    Returns:
        分类后的错误信息
    """
    error_str = str(error).lower()

    # 超时错误
    if "timeout" in error_str or "timed out" in error_str:
        return APIError(
            error_type=ErrorType.TIMEOUT,
            message="请求超时",
            retry_after=2.0,
        )

    # 速率限制
    if "429" in error_str or "rate limit" in error_str:
        return APIError(
            error_type=ErrorType.RATE_LIMIT,
            message="API 速率限制",
            retry_after=10.0,  # 通常需要等待更长时间
        )

    # 服务器错误
    if "500" in error_str or "502" in error_str or "503" in error_str:
        return APIError(
            error_type=ErrorType.SERVER_ERROR,
            message="服务器错误",
            retry_after=5.0,
        )

    # 认证错误
    if "401" in error_str or "unauthorized" in error_str:
        return APIError(
            error_type=ErrorType.AUTH_ERROR,
            message="API 密钥无效",
            retry_after=None,  # 不应重试
        )

    # 上下文超长
    if "context_length" in error_str or "too many tokens" in error_str:
        return APIError(
            error_type=ErrorType.CONTEXT_LENGTH,
            message="输入超出模型上下文长度限制",
            retry_after=None,  # 需要减少输入，不应重试
        )

    # 内容过滤
    if "content_filter" in error_str or "content policy" in error_str:
        return APIError(
            error_type=ErrorType.CONTENT_FILTER,
            message="内容被安全过滤器拦截",
            retry_after=None,
        )

    # 未知错误
    return APIError(
        error_type=ErrorType.UNKNOWN,
        message=f"未知错误: {error}",
        retry_after=5.0,
    )


def handle_api_error(error: APIError, context: str = "") -> dict:
    """统一的 API 错误处理函数

    根据错误类型采取不同的处理策略。

    Args:
        error: 分类后的错误信息
        context: 错误发生的上下文（如 "tool_call", "chat"）

    Returns:
        处理结果字典
    """
    logger.error(f"API 错误 [{error.error_type.value}]: {error.message} (context: {context})")

    strategies = {
        # 超时：建议重试
        ErrorType.TIMEOUT: {
            "action": "retry",
            "message": "请求超时，正在重试...",
            "should_retry": True,
        },
        # 速率限制：等待后重试
        ErrorType.RATE_LIMIT: {
            "action": "wait_and_retry",
            "message": "达到速率限制，等待后重试...",
            "should_retry": True,
            "wait_seconds": error.retry_after or 10,
        },
        # 服务器错误：重试
        ErrorType.SERVER_ERROR: {
            "action": "retry",
            "message": "服务器错误，正在重试...",
            "should_retry": True,
        },
        # 认证错误：不重试，通知用户
        ErrorType.AUTH_ERROR: {
            "action": "abort",
            "message": "API 密钥无效，请检查配置",
            "should_retry": False,
        },
        # 上下文超长：压缩输入后重试
        ErrorType.CONTEXT_LENGTH: {
            "action": "compress_and_retry",
            "message": "输入过长，正在压缩...",
            "should_retry": True,
        },
        # 内容过滤：修改输入后重试
        ErrorType.CONTENT_FILTER: {
            "action": "modify_and_retry",
            "message": "内容被拦截，正在调整...",
            "should_retry": True,
        },
        # 未知错误：重试
        ErrorType.UNKNOWN: {
            "action": "retry",
            "message": f"未知错误: {error.message}",
            "should_retry": True,
        },
    }

    return strategies.get(error.error_type, strategies[ErrorType.UNKNOWN])


# 使用示例
try:
    result = call_openai_with_retry(messages=messages)
except Exception as e:
    error = classify_error(e)
    handling = handle_api_error(error, context="chat")

    if handling["should_retry"]:
        wait_time = handling.get("wait_seconds", 2)
        logger.info(f"等待 {wait_time} 秒后重试...")
        time.sleep(wait_time)
        # 重试逻辑...
    else:
        logger.error(f"无法恢复的错误: {handling['message']}")
```

### 2.6.5 流式输出与并发控制

```python
import asyncio
from openai import AsyncOpenAI

# 异步 OpenAI 客户端
async_client = AsyncOpenAI()


async def stream_agent_response(messages, tools=None):
    """流式输出 Agent 响应

    流式输出可以显著降低用户感知延迟，提升交互体验。
    对于 Agent 来说，流式输出尤其重要，因为 Agent 的
    多步推理可能需要较长时间。
    """
    kwargs = {
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 4096,
        "stream": True,  # 启用流式输出
    }
    if tools:
        kwargs["tools"] = tools

    # 收集流式响应
    full_content = ""
    tool_calls = {}

    async for chunk in await async_client.chat.completions.create(**kwargs):
        delta = chunk.choices[0].delta

        # 处理文本内容
        if delta.content:
            full_content += delta.content
            # 可以在这里实时打印或发送给前端
            print(delta.content, end="", flush=True)

        # 处理工具调用（流式输出中工具调用是增量的）
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": tc.id or "",
                        "name": tc.function.name if tc.function.name else "",
                        "arguments": tc.function.arguments if tc.function.arguments else "",
                    }
                else:
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    if tc.function.name:
                        tool_calls[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls[idx]["arguments"] += tc.function.arguments

    return {
        "content": full_content,
        "tool_calls": list(tool_calls.values()),
    }


# 并发控制：限制同时进行的 API 调用数量
semaphore = asyncio.Semaphore(10)  # 最多 10 个并发请求


async def rate_limited_call(messages, tools=None):
    """带速率控制的 API 调用

    在 Agent 系统中，可能同时有多个 Agent 在运行，
    需要控制并发数以避免触发速率限制。
    """
    async with semaphore:
        return await stream_agent_response(messages, tools)


# 批量处理示例
async def process_batch(tasks):
    """批量处理多个任务"""
    results = await asyncio.gather(
        *[rate_limited_call(task["messages"], task.get("tools")) for task in tasks],
        return_exceptions=True,
    )
    return results
```

### 2.6.6 Prompt 缓存与成本优化

Prompt 缓存是降低 Agent 运营成本的重要技术。对于相同的 System Prompt 和工具定义，缓存可以避免重复计算，大幅降低成本和延迟。

**支持 Prompt 缓存的 API**：

| API | 缓存机制 | 节省比例 | 条件 |
|:---|:---|:---|:---|
| OpenAI | 自动 Prompt Caching | 50% 输入成本 | 前缀匹配 ≥ 1024 tokens |
| Anthropic | Prompt Caching | 90% 缓存写入成本 | 手动标记缓存点 |
| Google | Context Caching | 75% 输入成本 | 手动创建缓存 |
| DeepSeek | 自动缓存 | 50% 输入成本 | 前缀匹配 |

**OpenAI 自动 Prompt Caching**：

OpenAI 的缓存是自动的——只要请求的前缀（System Prompt + 工具定义 + 早期对话）相同，就会自动命中缓存。无需额外代码。

**Anthropic 手动 Prompt Caching**：

```python
import anthropic

client = anthropic.Anthropic()

# Anthropic 需要手动标记缓存点
# 使用 "cache_control": {"type": "ephemeral"} 标记需要缓存的内容
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "你是一个数据分析 Agent。" * 100,  # 长 System Prompt
            "cache_control": {"type": "ephemeral"},  # 标记为可缓存
        }
    ],
    messages=[{"role": "user", "content": "分析数据"}],
)

# 首次调用：写入缓存（成本较高）
# 后续调用：读取缓存（成本降低 90%）
print(f"缓存创建 tokens: {response.usage.cache_creation_input_tokens}")
print(f"缓存读取 tokens: {response.usage.cache_read_input_tokens}")
```

**Prompt 缓存的 Agent 优化策略**：

| 策略 | 描述 | 效果 |
|:---|:---|:---|
| **固定前缀** | 将 System Prompt 和工具定义放在最前面 | 确保缓存命中 |
| **长 System Prompt** | 越长的 System Prompt 缓存收益越大 | 1000+ tokens 的 System Prompt 缓存效果最佳 |
| **稳定工具定义** | 工具定义保持不变 | 工具定义通常在多次调用间保持一致 |
| **避免频繁修改 System Prompt** | 不要在每次调用时修改 System Prompt | 修改会破坏缓存 |
| **使用相同的消息结构** | 保持消息结构的一致性 | 有助于缓存命中 |

### 2.6.7 API 调用的成本监控

在生产环境中，必须对 API 调用成本进行实时监控和告警，避免意外的高额账单。

```python
import time
from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict


@dataclass
class APICallRecord:
    """单次 API 调用记录"""
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    success: bool
    error_type: str = ""


class CostMonitor:
    """API 成本监控器

    跟踪所有 API 调用的成本，支持实时查询和告警。
    """

    def __init__(self, daily_budget: float = 100.0, alert_threshold: float = 0.8):
        """初始化成本监控器

        Args:
            daily_budget: 每日预算上限（美元）
            alert_threshold: 告警阈值（占预算的比例）
        """
        self.daily_budget = daily_budget
        self.alert_threshold = alert_threshold
        self.records: List[APICallRecord] = []
        self.daily_costs: Dict[str, float] = defaultdict(float)

    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        latency_ms: float,
        success: bool = True,
        error_type: str = "",
    ):
        """记录一次 API 调用

        Args:
            model: 模型名称
            input_tokens: 输入 Token 数
            output_tokens: 输出 Token 数
            cost: 调用成本（美元）
            latency_ms: 延迟（毫秒）
            success: 是否成功
            error_type: 错误类型（如果失败）
        """
        record = APICallRecord(
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            success=success,
            error_type=error_type,
        )
        self.records.append(record)

        # 更新每日成本
        today = time.strftime("%Y-%m-%d")
        self.daily_costs[today] += cost

        # 检查是否超过告警阈值
        if self.daily_costs[today] > self.daily_budget * self.alert_threshold:
            print(f"⚠️ 警告：今日成本已达 ${self.daily_costs[today]:.2f}，"
                  f"超过预算的 {self.alert_threshold*100:.0f}%")

    def get_daily_summary(self, date: str = None) -> dict:
        """获取每日成本摘要

        Args:
            date: 日期（YYYY-MM-DD），默认今天

        Returns:
            包含各项统计的摘要字典
        """
        if date is None:
            date = time.strftime("%Y-%m-%d")

        # 筛选指定日期的记录
        day_records = [
            r for r in self.records
            if time.strftime("%Y-%m-%d", time.localtime(r.timestamp)) == date
        ]

        if not day_records:
            return {"date": date, "total_cost": 0, "total_calls": 0}

        total_cost = sum(r.cost for r in day_records)
        total_input = sum(r.input_tokens for r in day_records)
        total_output = sum(r.output_tokens for r in day_records)
        success_rate = sum(1 for r in day_records if r.success) / len(day_records)
        avg_latency = sum(r.latency_ms for r in day_records) / len(day_records)

        # 按模型分组统计
        model_costs = defaultdict(float)
        for r in day_records:
            model_costs[r.model] += r.cost

        return {
            "date": date,
            "total_cost": total_cost,
            "total_calls": len(day_records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "model_breakdown": dict(model_costs),
            "budget_remaining": self.daily_budget - total_cost,
        }

    def export_report(self) -> str:
        """导出成本报告"""
        summary = self.get_daily_summary()
        report = f"""
=== API 成本日报 ===
日期: {summary['date']}
总成本: ${summary['total_cost']:.4f}
总调用次数: {summary['total_calls']}
输入 Token 总数: {summary['total_input_tokens']:,}
输出 Token 总数: {summary['total_output_tokens']:,}
成功率: {summary['success_rate']:.1%}
平均延迟: {summary['avg_latency_ms']:.0f}ms
剩余预算: ${summary['budget_remaining']:.2f}

模型成本分布:
"""
        for model, cost in summary['model_breakdown'].items():
            report += f"  {model}: ${cost:.4f}\n"

        return report


# 使用示例
monitor = CostMonitor(daily_budget=100.0, alert_threshold=0.8)

# 模拟 API 调用记录
monitor.record_call("gpt-4o", 1500, 500, 0.00625, 1200, success=True)
monitor.record_call("gpt-4o-mini", 1500, 500, 0.00045, 800, success=True)
monitor.record_call("gpt-4o", 3000, 1000, 0.0125, 2000, success=False, error_type="timeout")

# 生成报告
print(monitor.export_report())
```

---

## 2.7 本地模型部署

### 2.7.1 Ollama 快速部署

Ollama 是最简单的本地 LLM 部署方案，支持一键安装和运行多种开源模型。

```bash
# 安装 Ollama（macOS/Linux）
curl -fsSL https://ollama.ai/install.sh | sh

# 拉取模型（首次运行会自动下载）
ollama pull llama3.1:8b          # 8B 参数，约 4.7GB
ollama pull qwen2.5:72b          # 72B 参数，约 40GB
ollama pull deepseek-r1:7b       # 7B 推理模型

# 运行模型
ollama run llama3.1:8b

# 查看已安装的模型
ollama list

# 启动 Ollama 服务（默认端口 11434）
ollama serve
```

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# 创建本地模型实例
local_llm = ChatOllama(
    model="llama3.1:8b",    # 使用的模型
    temperature=0,           # 确定性输出
    num_ctx=8192,            # 上下文窗口大小
    num_gpu=99,              # 使用 GPU 的层数（99 表示全部）
)

# 使用方式与 OpenAI 完全一致
messages = [
    SystemMessage(content="你是一个有帮助的助手。"),
    HumanMessage(content="什么是 AI Agent？"),
]

response = local_llm.invoke(messages)
print(response.content)

# Ollama 也支持工具调用
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取城市天气信息"""
    return f"{city}今天晴天，25°C"

# 绑定工具
llm_with_tools = local_llm.bind_tools([get_weather])
response = llm_with_tools.invoke([HumanMessage(content="北京今天天气怎么样？")])
print(response.tool_calls)
```

### 2.7.2 vLLM 高性能部署

vLLM 是高性能的 LLM 推理引擎，支持 PagedAttention、连续批处理等优化技术，适合生产环境部署。

```bash
# 安装 vLLM
pip install vllm

# 启动 vLLM 服务器（兼容 OpenAI API 格式）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9

# 使用 OpenAI 兼容客户端调用
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

```python
from openai import OpenAI

# vLLM 启动后提供兼容 OpenAI 的 API
# 只需将 base_url 指向 vLLM 服务器即可
client = OpenAI(
    base_url="http://localhost:8000/v1",  # vLLM 服务器地址
    api_key="not-needed",                  # 本地部署不需要真实 API Key
)

# 与调用 OpenAI API 完全一致
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "解释一下 Transformer 的工作原理"},
    ],
    temperature=0,
    max_tokens=1024,
)

print(response.choices[0].message.content)

# vLLM 也支持 Function Calling（需要模型支持）
tools = [{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "执行数学计算",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式"},
            },
            "required": ["expression"],
        },
    },
}]

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "计算 123 * 456"}],
    tools=tools,
    temperature=0,
)

print(response.choices[0].message.tool_calls)
```

**vLLM 关键优化技术**：

| 技术 | 原理 | 性能提升 | 说明 |
|:---|:---|:---|:---|
| **PagedAttention** | 将 KV Cache 分页管理 | 2-4x 吞吐量 | 解决内存碎片化问题 |
| **连续批处理** | 动态合并多个请求 | 2-10x 吞吐量 | 不同请求共享计算资源 |
| **张量并行** | 跨 GPU 分布计算 | 线性扩展 | 支持多 GPU 推理 |
| **量化推理** | 降低精度计算 | 2-4x 速度 | 支持 GPTQ/AWQ/GGUF |
| **Prefix Caching** | 缓存公共前缀 | 2-5x 速度 | 适合 System Prompt 相同的场景 |

### 2.7.3 量化技术详解

量化（Quantization）通过降低模型权重的数值精度来减少内存占用和计算成本，是本地部署的关键技术。理解不同量化方法的原理和特性，有助于在质量、速度和内存之间做出最优权衡。

**量化的基本原理**：

将模型权重从高精度（如 FP16，16 位浮点）映射到低精度（如 INT4，4 位整数）：

$$
x_q = \text{round}\left(\frac{x - \min(X)}{\max(X) - \min(X)} \times (2^n - 1)\right)
$$

其中 $x$ 是原始权重，$X$ 是整个权重集合，$n$ 是量化位数，$x_q$ 是量化后的整数值。

**不同量化方法的详细对比**：

| 技术 | 精度 | 7B 模型大小 | 速度损失 | 质量损失 | 适用场景 |
|:---|:---|:---|:---|:---|:---|
| FP16 | 16-bit | ~14GB | 基准 | 无 | GPU 充足时 |
| INT8 | 8-bit | ~7GB | ~5% | 极小 | GPU 有限时 |
| GPTQ | 4-bit | ~4GB | ~10% | 很小 | GPU 严重不足 |
| AWQ | 4-bit | ~4GB | ~5% | 很小 | GPU 有限，追求速度 |
| GGUF Q4_K_M | 4-bit | ~4GB | ~15% | 小 | CPU 推理 |
| GGUF Q8_0 | 8-bit | ~7GB | ~5% | 极小 | CPU 推理，质量优先 |

**GPTQ vs AWQ 的技术差异**：

| 特性 | GPTQ | AWQ |
|:---|:---|:---|
| **量化原理** | 基于二阶信息的逐层量化 | 基于激活值的感知量化 |
| **校准数据** | 需要 128 条校准样本 | 需要少量校准样本 |
| **量化速度** | 较慢（逐层优化） | 较快（直接量化） |
| **推理速度** | 一般 | 更快（激活感知优化） |
| **质量** | 略好 | 略差但差距很小 |
| **GPU 要求** | 需要 GPU | 需要 GPU |
| **推荐场景** | 追求极致质量 | 追求速度和效率 |

**GGUF 格式的特殊优势**：

GGUF 是 llama.cpp 项目使用的模型格式，支持 CPU 推理，是个人开发者本地部署的首选格式。

| GGUF 量化级别 | 位数 | 7B 模型大小 | 质量 | 速度 | 推荐场景 |
|:---|:---|:---|:---|:---|:---|
| Q2_K | 2-bit | ~2.8GB | 差 | 最快 | 内存极度受限 |
| Q4_0 | 4-bit | ~3.8GB | 一般 | 快 | 内存有限 |
| Q4_K_M | 4-bit | ~4.3GB | 良好 | 较快 | **推荐默认** |
| Q5_K_M | 5-bit | ~5.1GB | 很好 | 中等 | 质量优先 |
| Q6_K | 6-bit | ~5.9GB | 很好 | 较慢 | 高质量需求 |
| Q8_0 | 8-bit | ~7.5GB | 极好 | 慢 | 质量最高 |
| F16 | 16-bit | ~14GB | 无损 | 最慢 | 无损需求 |

```python
# 使用 llama.cpp 的 GGUF 模型加载示例（通过 langchain）
from langchain_community.llms import LlamaCpp

# 加载 GGUF 量化模型
llm = LlamaCpp(
    model_path="./models/llama-3.1-8b-instruct.Q4_K_M.gguf",  # GGUF 模型路径
    n_ctx=4096,          # 上下文窗口大小
    n_gpu_layers=35,     # 放到 GPU 上的层数（0 表示纯 CPU）
    temperature=0,       # 温度参数
    max_tokens=2048,     # 最大输出 Token 数
    verbose=True,        # 是否打印详细信息
)

# 使用方式与其他 LangChain LLM 一致
response = llm.invoke("什么是 AI Agent？")
print(response)
```

> **量化选型建议**：对于 Agent 开发，推荐使用 **AWQ 4-bit 量化**配合 vLLM 部署。AWQ 在推理速度上优于 GPTQ，同时质量损失可以接受。如果需要 CPU 推理，使用 **GGUF Q4_K_M** 格式配合 llama.cpp 是最佳选择。对于质量要求极高的场景，可以使用 INT8 或 FP16 精度。

```python
# 使用 GPTQ 量化模型加载（需要 AutoGPTQ）
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# 量化配置
quantization_config = GPTQConfig(
    bits=4,                    # 量化位数
    damp_percent=0.01,         # 阻尼百分比
    desc_act=True,             # 按激活值降序排列
    group_size=128,            # 量化组大小
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GPTQ",
    quantization_config=quantization_config,
    device_map="auto",         # 自动分配 GPU
)

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-Chat-GPTQ")

# 使用量化模型
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 2.7.4 本地模型与 Agent 集成

```python
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class ModelRouter:
    """模型路由器

    根据任务复杂度自动选择合适的模型：
    - 简单任务：使用本地小模型（零成本、低延迟）
    - 复杂任务：使用云端大模型（高质量）
    """

    def __init__(self):
        # 本地模型（通过 Ollama）
        self.local_simple = ChatOllama(model="llama3.1:8b", temperature=0)
        self.local_reasoning = ChatOllama(model="deepseek-r1:7b", temperature=0)

        # 云端模型
        self.cloud_gpt4o = ChatOpenAI(model="gpt-4o", temperature=0)
        self.cloud_claude = ChatOpenAI(model="claude-sonnet-4-20250514", temperature=0)

    def route(self, task_complexity: str, task_type: str = "general"):
        """根据任务复杂度和类型选择模型

        Args:
            task_complexity: 任务复杂度（simple/medium/complex）
            task_type: 任务类型（general/code/reasoning）

        Returns:
            选中的模型实例
        """
        if task_complexity == "simple":
            # 简单任务使用本地模型
            return self.local_simple
        elif task_complexity == "medium":
            # 中等任务根据类型选择
            if task_type == "reasoning":
                return self.local_reasoning
            return self.cloud_gpt4o
        else:
            # 复杂任务使用最强云端模型
            if task_type == "code":
                return self.cloud_claude
            return self.cloud_gpt4o


# 使用示例
router = ModelRouter()

# 简单任务 → 本地模型（零成本）
simple_model = router.route("simple")
response = simple_model.invoke([HumanMessage(content="1+1等于几？")])

# 复杂推理任务 → 本地推理模型
reasoning_model = router.route("medium", "reasoning")
response = reasoning_model.invoke([HumanMessage(content="分析这段代码的复杂度")])

# 复杂代码任务 → 云端最强模型
complex_model = router.route("complex", "code")
response = complex_model.invoke([HumanMessage(content="重构这个函数，添加错误处理和单元测试")])
```

**本地 vs 云端部署对比**：

| 维度 | 本地部署 | 云端 API |
|:---|:---|:---|
| **成本** | 硬件成本高，边际成本低 | 无硬件成本，按调用付费 |
| **延迟** | 取决于硬件，可能较低 | 网络延迟 + 排队时间 |
| **隐私** | 数据不出本地 | 数据发送到云端 |
| **可用性** | 取决于本地硬件 | 依赖网络和 API 服务 |
| **模型更新** | 需要手动更新 | 自动使用最新版本 |
| **扩展性** | 受限于硬件 | 弹性扩展 |
| **质量** | 受限于可部署的模型大小 | 可使用最大的模型 |

> **部署建议**：对于生产环境的 Agent 系统，推荐采用**混合部署策略**——简单任务和敏感数据处理使用本地模型，复杂推理和通用任务使用云端 API。这样既能控制成本、保护隐私，又能保证复杂任务的质量。

### 2.7.5 本地模型的 GPU 需求与性能基准

选择本地部署方案时，必须了解不同模型对硬件的需求以及实际推理性能。这直接决定了部署成本和用户体验。

**不同规模模型的 GPU 需求**：

| 模型规模 | 参数量 | FP16 显存 | INT8 显存 | INT4 显存 | 推荐 GPU | 推理速度 |
|:---|:---|:---|:---|:---|:---|:---|
| 小型 | 1.5B-3B | 3-6GB | 2-3GB | 1-2GB | RTX 3060 12G | ~80 tokens/s |
| 中型 | 7B-8B | 14-16GB | 7-8GB | 4-5GB | RTX 4090 24G | ~40 tokens/s |
| 大型 | 13B-14B | 26-28GB | 13-14GB | 7-8GB | A100 40G | ~25 tokens/s |
| 超大 | 34B-70B | 68-140GB | 34-70GB | 17-35GB | 2x A100 80G | ~10 tokens/s |
| 旗舰 | 72B-405B | 144-810GB | 72-405GB | 36-200GB | 4-8x A100 80G | ~5 tokens/s |

**vLLM 推理性能基准（A100 80G 单卡）**：

| 模型 | 精度 | 吞吐量 (tokens/s) | 首 Token 延迟 | 并发请求数 |
|:---|:---|:---|:---|:---|
| Llama 3.1 8B | FP16 | ~120 | ~0.3s | 32 |
| Llama 3.1 8B | INT4 (AWQ) | ~200 | ~0.2s | 64 |
| Qwen 2.5 72B | FP16 | ~15 | ~0.8s | 8 |
| Qwen 2.5 72B | INT4 (AWQ) | ~30 | ~0.5s | 16 |
| DeepSeek V3 | FP8 | ~8 | ~1.2s | 4 |

> **硬件选型建议**：对于个人开发者或小型团队，RTX 4090（24GB）是性价比最高的选择，可以流畅运行 7B-8B 模型。对于企业级部署，A100 80G 是标准配置，可以运行 72B 量化模型。如果预算充足，H100 是当前性能最强的选择。

### 2.7.6 本地模型的质量评估

本地模型的质量直接影响 Agent 的行为可靠性。在选择本地模型时，需要关注以下几个关键指标。

**本地模型的 Agent 能力评估维度**：

| 评估维度 | 测试方法 | 合格标准 | 重要性 |
|:---|:---|:---|:---|
| **指令遵循** | 给定复杂指令，检查输出是否符合 | 准确率 > 90% | 极高 |
| **工具调用准确性** | 提供工具定义，检查 JSON 输出格式 | 格式正确率 > 95% | 极高 |
| **参数填充准确性** | 提供工具定义和用户输入，检查参数提取 | 准确率 > 85% | 高 |
| **多步推理** | 给定多步任务，检查推理链的完整性 | 完整率 > 80% | 高 |
| **错误恢复** | 故意引入错误，检查模型能否自我修正 | 修正率 > 70% | 中 |
| **上下文理解** | 长上下文中引用远距离信息 | 准确率 > 85% | 中 |

**常见本地模型的 Agent 能力对比**：

| 模型 | 指令遵循 | 工具调用 | 多步推理 | 中文能力 | 推荐 Agent 场景 |
|:---|:---|:---|:---|:---|:---|
| Llama 3.1 8B | ★★★ | ★★★ | ★★ | ★★ | 简单问答、分类 |
| Llama 3.1 70B | ★★★★ | ★★★★ | ★★★★ | ★★★ | 通用 Agent |
| Qwen 2.5 7B | ★★★ | ★★★ | ★★★ | ★★★★★ | 中文简单 Agent |
| Qwen 2.5 72B | ★★★★ | ★★★★ | ★★★★ | ★★★★★ | 中文通用 Agent |
| DeepSeek R1 7B | ★★★ | ★★ | ★★★★ | ★★★★★ | 中文推理任务 |
| Mistral 7B | ★★★ | ★★★ | ★★★ | ★★ | 英文通用 Agent |

> **质量保障建议**：在将本地模型用于生产环境的 Agent 系统之前，务必进行充分的基准测试。建议使用实际的 Agent 场景进行端到端测试，而不仅仅依赖通用基准测试分数。本地模型在通用基准上的表现可能很好，但在特定的 Agent 场景下（如工具调用、多步推理）可能不如预期。

---

## 2.8 进阶话题：LLM 的内部机制

### 2.8.1 KV Cache 与推理优化

在自回归生成过程中，每生成一个新 Token 都需要重新计算所有之前 Token 的注意力。**KV Cache（键值缓存）**通过缓存之前计算过的 Key 和 Value 矩阵来避免重复计算，显著提升推理效率。

**KV Cache 的工作原理**：

在标准 Self-Attention 中，生成第 $t$ 个 Token 时需要计算：

$$
\text{Attention}(q_t, K_{1:t}, V_{1:t}) = \text{softmax}\left(\frac{q_t K_{1:t}^T}{\sqrt{d_k}}\right)V_{1:t}
$$

其中 $K_{1:t}$ 和 $V_{1:t}$ 包含了从第 1 个到第 $t$ 个 Token 的所有 Key 和 Value。如果没有 KV Cache，每生成一个新 Token 都需要重新计算整个序列的 Key 和 Value，计算复杂度为 $O(t \cdot n)$。

有了 KV Cache，之前 Token 的 Key 和 Value 被缓存下来，只需计算新 Token 的 Key 和 Value 并追加到缓存中，计算复杂度降为 $O(n)$。

**KV Cache 的内存占用**：

$$
\text{KV Cache 大小} = 2 \times L \times n_h \times d_h \times s \times b
$$

其中 $L$ 是层数，$n_h$ 是注意力头数，$d_h$ 是每个头的维度，$s$ 是序列长度，$b$ 是批处理大小。乘以 2 是因为 Key 和 Value 各占一份。

例如，对于 Llama 3.1 70B 模型（$L=80$, $n_h=64$, $d_h=128$），128K 上下文的 KV Cache 约占 40GB 显存——这几乎是模型权重本身大小的 1/3。

**KV Cache 优化技术**：

| 技术 | 原理 | 内存节省 | 质量影响 | 使用模型 |
|:---|:---|:---|:---|:---|
| **GQA** | 多个 Query 头共享 Key/Value 头 | 4-8x | 极小 | Llama 3, Mistral |
| **MQA** | 所有 Query 头共享一个 Key/Value 头 | ~8x | 有损失 | PaLM |
| **PagedAttention** | 分页管理 KV Cache | 2-4x（利用率） | 无 | vLLM |
| **量化 KV Cache** | 降低 KV Cache 精度 | 2-4x | 极小 | 部分推理引擎 |
| **滑动窗口** | 只保留最近 N 层的 KV Cache | 可变 | 有损失 | Longformer |

### 2.8.2 MoE（Mixture of Experts）架构

MoE 是近年来最重要的架构创新之一，被 GPT-4、DeepSeek V3、Llama 4 等模型采用。它通过"稀疏激活"实现了在保持大模型容量的同时降低推理成本。

**MoE 的核心思想**：

将一个大的前馈网络（FFN）拆分为多个"专家"子网络，每次推理时只激活其中的 Top-K 个专家。这使得模型的总参数量很大（如 DeepSeek V3 的 671B），但每次推理只使用其中一部分（如 37B），大幅降低计算成本。

**MoE 的数学表示**：

$$
y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)
$$

其中 $E_i$ 是第 $i$ 个专家网络，$G(x)$ 是门控网络（Gating Network），$G(x)_i$ 是门控网络为第 $i$ 个专家分配的权重。门控网络通常只选择 Top-K 个专家，其余专家的权重为 0。

**主流 MoE 模型对比**：

| 模型 | 总参数 | 激活参数 | 专家数 | Top-K | 上下文 | 特点 |
|:---|:---|:---|:---|:---|:---|:---|
| GPT-4 | ~1.8T | ~220B | ~16 | 2 | 128K | 闭源，性能顶级 |
| DeepSeek V3 | 671B | 37B | 256 | 8 | 128K | 极致性价比 |
| Llama 4 Scout | 109B | ~17B | 16 | 1 | 10M | 超长上下文 |
| Llama 4 Maverick | 400B | ~17B | 128 | 1 | 1M | 开源顶级 |
| Mixtral 8x7B | 47B | 13B | 8 | 2 | 32K | 开源先驱 |
| Qwen 3 235B | 235B | ~22B | 128 | 8 | 128K | 中文优化 |

**MoE 对 Agent 的影响**：

| 方面 | Dense 模型 | MoE 模型 | Agent 影响 |
|:---|:---|:---|:---|
| 推理成本 | 与总参数成正比 | 与激活参数成正比 | MoE 更经济 |
| 模型容量 | 受限于计算预算 | 可以更大 | MoE 知识更丰富 |
| 推理延迟 | 与参数量成正比 | 与激活参数成正比 | MoE 更快 |
| 内存需求 | 与参数量成正比 | 与总参数成正比 | MoE 内存需求更高 |
| 负载均衡 | 无 | 需要专家负载均衡 | 可能影响稳定性 |

### 2.8.3 模型幻觉的成因与缓解

幻觉（Hallucination）是 LLM 最主要的局限性之一。在 Agent 场景中，幻觉可能导致错误的工具调用、不准确的信息检索、甚至危险的操作。

**幻觉的分类**：

| 类型 | 描述 | Agent 风险 | 示例 |
|:---|:---|:---|:---|
| **事实性幻觉** | 生成与现实不符的信息 | 误导决策 | 编造不存在的 API 端点 |
| **忠实性幻觉** | 输出与输入不一致 | 工具调用参数错误 | 从用户输入中提取错误信息 |
| **推理幻觉** | 推理过程中出现逻辑错误 | 多步推理失败 | 在 Chain-of-Thought 中跳步 |
| **格式幻觉** | 输出格式不符合要求 | 解析失败 | JSON 格式不正确 |

**幻觉的成因**：

1. **训练数据偏差**：模型学习到训练数据中的噪声和错误
2. **概率采样**：随机采样可能选择不恰当的 Token
3. **上下文不足**：缺乏足够的信息来做出准确判断
4. **过度自信**：模型对自己的输出过于自信，不会表达不确定性
5. **长距离依赖衰减**：在长上下文中，远距离信息的注意力权重降低

**Agent 场景的幻觉缓解策略**：

| 策略 | 实现方式 | 效果 | 适用场景 |
|:---|:---|:---|:---|
| **工具验证** | 调用工具验证 LLM 输出 | 高 | 所有涉及工具调用的场景 |
| **RAG 增强** | 提供外部知识作为上下文 | 高 | 事实性问答 |
| **温度降低** | 使用 T=0 减少随机性 | 中 | 所有场景 |
| **多次采样** | 生成多个输出并投票 | 高 | 高可靠性要求的场景 |
| **Chain-of-Thought** | 要求模型展示推理过程 | 中 | 复杂推理任务 |
| **自我反思** | 要求模型检查自己的输出 | 中 | 关键决策 |
| **置信度校准** | 要求模型表达不确定性 | 低-中 | 需要概率估计的场景 |

```python
# 幻觉缓解：多次采样 + 投票
from collections import Counter

def multi_sample_with_voting(
    client,
    messages: list,
    n_samples: int = 5,
    model: str = "gpt-4o",
) -> dict:
    """多次采样 + 投票来缓解幻觉

    对同一个问题生成多个回答，通过投票选择最一致的答案。
    这种方法可以显著降低幻觉概率，但会增加 API 调用成本。

    Args:
        client: OpenAI 客户端
        messages: 消息列表
        n_samples: 采样次数
        model: 模型名称

    Returns:
        包含最一致答案和置信度的字典
    """
    responses = []

    for i in range(n_samples):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,  # 使用较高温度以获得多样性
            max_tokens=1024,
        )
        responses.append(response.choices[0].message.content)

    # 统计最频繁的答案
    # 这里使用简单的精确匹配，实际中可能需要语义匹配
    answer_counts = Counter(responses)
    most_common_answer, count = answer_counts.most_common(1)[0]

    # 计算置信度（最频繁答案占总采样数的比例）
    confidence = count / n_samples

    return {
        "answer": most_common_answer,
        "confidence": confidence,
        "n_agreeing": count,
        "n_samples": n_samples,
        "all_answers": responses,
    }


# 使用示例
# result = multi_sample_with_voting(client, messages)
# print(f"答案: {result['answer']}")
# print(f"置信度: {result['confidence']:.1%}")
# if result['confidence'] < 0.6:
#     print("警告：模型对此答案不够确定，建议人工验证")
```

---

## 2.9 Agent 场景的模型微调

### 2.9.1 微调 vs Prompt Engineering

虽然通用 LLM 已经具备强大的 Agent 能力，但在特定场景下，通过微调（Fine-tuning）可以进一步提升模型在该场景下的表现。

| 维度 | Prompt Engineering | 微调 |
|:---|:---|:---|
| **成本** | 低（只需编写 Prompt） | 高（需要训练数据和计算资源） |
| **灵活性** | 高（随时修改 Prompt） | 低（修改需要重新训练） |
| **效果上限** | 受限于模型基础能力 | 可以超越基础能力 |
| **适用场景** | 通用场景 | 特定领域、特定格式 |
| **时间投入** | 低 | 高（数据准备 + 训练） |

**适合微调的 Agent 场景**：

| 场景 | 微调收益 | 原因 |
|:---|:---|:---|
| **特定领域工具调用** | 高 | 模型学会特定工具的使用模式 |
| **定制输出格式** | 高 | 确保输出严格符合业务格式 |
| **领域知识增强** | 中 | 注入行业特定知识 |
| **边缘设备部署** | 高 | 在小模型上获得大模型的能力 |
| **减少 Token 消耗** | 中 | 微调后的模型可能需要更少的上下文 |

### 2.9.2 微调数据准备与训练

**Agent 微调数据格式**：

```python
# Agent 微调数据格式示例（OpenAI 微调格式）
training_data = [
    {
        "messages": [
            {
                "role": "system",
                "content": "你是一个数据分析 Agent，可以查询数据库和执行代码。"
            },
            {
                "role": "user",
                "content": "查询上个月的销售额并计算同比增长"
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "query_database",
                            "arguments": '{"sql": "SELECT SUM(amount) FROM sales WHERE month = DATE_SUB(CURDATE(), INTERVAL 1 MONTH)"}'
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "content": '{"total_amount": 1250000}'
            },
            {
                "role": "assistant",
                "content": "上个月销售额为 125 万元，同比增长 13.6%。"
            }
        ]
    }
]

# 保存为 JSONL 文件供 OpenAI 微调使用
import json

with open("agent_training_data.jsonl", "w") as f:
    for example in training_data:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")
```

 > **微调建议**：对于大多数 Agent 应用，Prompt Engineering 已经足够。只有在以下情况才考虑微调：(1) 通用模型在特定场景下表现不佳；(2) 需要严格控制输出格式；(3) Token 成本过高需要优化；(4) 需要在边缘设备上部署。

Agent 规划是 LLM 推理能力的综合体现，不同的规划策略适用于不同复杂度的任务。下面的交互式可视化展示了 Agent 规划树的工作原理：

<div data-component="AgentPlanningDemo"></div>

---

## 2.10 本章小结

本章深入探讨了 LLM 作为 Agent 推理引擎的核心知识，从底层的 Transformer 架构到上层的 API 调用实践，构建了完整的知识体系。

### 核心知识回顾

| 知识点 | 核心要点 | Agent 实践意义 |
|:---|:---|:---|
| **LLM 作为大脑** | 通用推理、上下文学习、指令遵循、工具理解 | Agent 的推理核心，决定 Agent 能力上限 |
| **Transformer** | Self-Attention + 多头注意力 + 位置编码 | 理解 LLM 如何处理信息，优化 Prompt 设计 |
| **Token 机制** | BPE 分词，中文约 1-1.5 token/字 | 控制成本，优化 Prompt 效率 |
| **上下文窗口** | 窗口大小 = 系统提示 + 工具 + 历史 + RAG + 输出 | 合理分配窗口空间，避免信息丢失 |
| **Temperature** | T=0 确定性，T>0 随机性 | **工具调用必须 T=0**，对话可用 T=0.7 |
| **模型选型** | 综合考虑能力、成本、延迟、上下文 | 根据场景选择最合适的模型 |
| **API 调用** | 重试机制、错误处理、流式输出 | 构建健壮的 Agent 系统 |
| **本地部署** | Ollama 最简单，vLLM 性能最好 | 隐私保护、成本控制 |

### 关键决策指南

1. **工具调用必须用 `temperature=0`**：任何涉及 Function Calling 的场景都必须使用确定性输出，否则 JSON 格式可能出错。
2. **优先使用 Function Calling 而非文本解析**：Function Calling 是最可靠的结构化输出方式。
3. **实现模型路由策略**：根据任务复杂度自动选择模型，平衡成本和质量。
4. **做好错误处理和重试**：API 调用可能失败，必须有完善的错误处理机制。
5. **控制上下文窗口使用**：不要因为窗口大就塞满信息，"Lost in the Middle"现象依然存在。

### 技术选型速查

| 需求 | 推荐方案 |
|:---|:---|
| 快速原型开发 | GPT-4o-mini + LangChain |
| 生产级 Agent | GPT-4o / Claude Sonnet 4 + 自定义框架 |
| 代码 Agent | Claude Sonnet 4 + Function Calling |
| 长文档处理 | Gemini 2.5 Pro / GPT-4.1 |
| 低成本部署 | DeepSeek V3 / GPT-4o-mini |
| 本地部署 | Ollama + Llama 4 / Qwen 3 |
| 高性能本地 | vLLM + AWQ 量化模型 |
| 隐私敏感场景 | 本地部署 + 本地模型 |

> **下一章预告**
>
> 在第 3 章中，我们将深入 Agent 的指令系统——提示词工程。掌握 System Prompt 设计原则、Few-Shot 学习策略、Chain-of-Thought 推理技巧，以及提示词安全防护方法。提示词工程是将 LLM 的潜力转化为 Agent 能力的关键桥梁。
