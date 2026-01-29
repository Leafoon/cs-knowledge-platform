---
applyTo: '**'
---
请为我设计并生成一套全面、系统、由浅入深的 LangChain 生态学习内容，直接适配填充到教学/演示/知识管理界面中使用。内容必须以官方网站（https://www.langchain.com/）**、**官方文档（https://docs.langchain.com/）**、**Python / JS 参考文档（https://api.python.langchain.com/ 或 https://api.js.langchain.com/）**、**GitHub 仓库（https://github.com/langchain-ai/langchain、https://github.com/langchain-ai/langgraph、https://github.com/langchain-ai/langsmith-sdk 等）、官方博客、示例代码（templates/、cookbook/ 等）为最主要、最权威的依据。严禁凭空捏造不存在的组件、链、图节点、回调、评估器、部署方式或已废弃的旧 API。所有代码、概念、最佳实践均应与当前主分支（main）或最新稳定版本保持一致，并适当注明参考版本或文档链接。

总体要求：
1. 内容深度与广度：从零基础（快速构建简单链、聊天机器人）开始，逐步推进到中高级（复杂链组合、自定义组件）、状态管理与控制流（LangGraph）、可观测性与评估（LangSmith）、生产部署（LangServe）、高级 agent 设计（工具调用、规划、重试、人机交互、长期记忆、多 agent 协作）、性能优化与可靠性工程。宁可内容有适度冗余与多层次重复讲解，也绝不允许核心概念、模式或生产级实践缺失。同一机制可在入门、中级、高级章节多次出现，但每次聚焦点、复杂度、工程维度需递增。
2. 讲解形式与要求：
    - 每个重要概念需包含：官方定义、设计动机与权衡、与其他框架（如 LlamaIndex、Haystack、AutoGen）的对比（若适用）、机制原理说明（必要时使用 LaTeX 表示提示模板、状态图、评分公式等）、典型企业级应用场景。
    - 必须提供大量可直接运行的代码示例（Python 优先，必要时补充 JS/TS；包含完整 import、环境依赖提示、模型提供商配置），并附上预期输出（链执行结果、JSON 结构、日志、追踪 ID、评估分数、生成文本等）。
    - 对于复杂或底层机制（例如：Runnable 协议与流式处理、LangGraph 的 Pregel 执行循环、条件边与动态路由、checkpointing 与持久化状态、LangSmith 的 tracing 埋点与 span、hub pull 提示管理、tool calling 与 structured output、agent supervisor 与 hierarchical 编排、human-in-the-loop 中断与审批、LangServe 的批处理与流式响应等），强烈建议采用交互式动画、动态流程图、步进可视化进行讲解（例如：LangGraph 状态机演进动画、消息流与内存更新过程、tracing 时间线与 span 嵌套图、tool 调用决策树、retry / fallback 路径模拟、多 agent 协作通信图等）。若当前界面支持嵌入动画、可交互图表、Mermaid 图增强或 Streamlit-like 组件，请优先采用此类形式。

3. 结构要求：
    - 首先生成一份非常详细的分级目录（建议采用 Chapter 0、Chapter 1 … 形式），层级至少到三级（大章 → 小节 → 具体组件/模式/示例类型）。
    - 目录需清晰体现主题与难度递进：零基础快速上手 → 核心抽象与 LCEL → 提示工程与输出解析 → 记忆与检索 → LangGraph 状态图与控制流 → Agent 设计与工具集成 → LangSmith 可观测性、调试与评估 → LangServe 部署与生产化 → 高级模式（多 agent、规划、重试、长期任务） → 优化与可靠性工程 → 生态集成与迁移。
    - 目录生成后，一个章节一个章节地详细展开内容（不要一次性输出全部），每个章节内部建议再细分小节，并保持相对统一的讲解结构：概念说明 → 代码示例（含依赖与配置） → 运行输出 / 追踪示例 → 常见问题、陷阱与调试技巧 → 扩展阅读（官方文档/模板/博客链接）。

4. 重点覆盖但不限于以下高级主题（应有独立或深度章节）：
    - LCEL（LangChain Expression Language）：Runnable、pipe、并行、fallback、配置化、流式、batch、astream 等
    - 提示管理：PromptTemplate、ChatPromptTemplate、FewShot、LangChain Hub、动态提示
    - 输出解析：OutputParser、Pydantic、JSON、structured tool calling、with_structured_output
    - 记忆系统：ConversationBuffer、Summary、Entity、VectorStore-backed、持久化
    - 检索增强生成（RAG）：Document loaders、Text splitters、Embeddings、VectorStores、Retrievers、Contextual compression、multi-query / parent-document 等
    - LangGraph：StateGraph、节点、边（条件/无条件）、checkpoint、persistence、human-in-the-loop、编译为 Pregel、streaming、时间旅行调试
    - Agent 架构：ReAct、OpenAI function/tool calling、自定义 agent、multi-agent、supervisor、hierarchical、planning agents、reflection / self-critique
    - LangSmith：Tracing、数据集、在线/离线评估、A/B 测试、监控、警报、反馈收集、Prompt playground
    - LangServe：链/图部署为 REST API、批处理、流式、OpenAPI、Playground、集成 FastAPI
    - 可靠性工程：Retries、fallbacks、rate limiting、error handling、tool error recovery、long-running task management
    - 性能优化：Caching、async、batch、streaming tokens、model routing、token 计数与成本控制
    - 生态集成：与 LlamaIndex、Haystack、AutoGen、CrewAI 的对比与迁移、与 LangGraph 的深度结合、与主流 LLM 提供商的适配

5. 其他细节偏好：
    - 优先使用现代 LCEL 写法，避免旧式 Chain 组合（除非讲解历史演进）；优先 OpenAI / Anthropic / Grok 等主流模型，但展示如何切换提供商。
    - 数据集与示例优先使用官方模板（https://github.com/langchain-ai/langchain/tree/master/templates）、cookbook、LangChain Hub 公开提示。
    - 在合适位置插入小练习、对比实验建议、思考题（如：LCEL 与传统 Chain 的性能差异？LangGraph checkpoint 如何实现时间旅行？LangSmith 评估指标如何自定义？）。
    - 鼓励展示定量对比（延迟、token 消耗、准确率、召回率、成本、稳定性等）。
    - 语言风格：正式、严谨、结构清晰、逻辑严密，如同撰写给 AI 工程师、研究员或企业 AI 团队的高质量内部培训教材。


请先输出详细的章节大纲（包含 Chapter 编号、章节标题、二级/三级子标题），待我确认或提出修改意见后再逐章详细展开具体内容。
（可根据 LangChain / LangGraph / LangSmith 当前主分支最新特性、官方模板、博客更新进行适度微调，但核心渐进结构与深度要求保持不变。）