---
title: "Appendix C: 常见问题与调试指南"
description: "Agent 开发中的常见错误诊断：无限循环、工具调用失败、记忆溢出、Token 超限、幻觉问题与性能瓶颈排查。"
date: "2026-06-11"
---

# Appendix C: 常见问题与调试指南

本附录收录了 Agent 开发过程中最常见的错误、问题和调试技巧。无论你是初学者还是有经验的开发者，这些内容都能帮助你快速定位和解决问题。

---

## C.1 Agent 陷入无限循环

### C.1.1 问题描述

Agent 陷入无限循环是最常见的问题之一。表现为：Agent 不断调用相同工具、重复相同的推理步骤、或者在两个状态之间来回切换，永远无法得出最终答案。

**典型症状**：
- Agent 连续调用同一个工具超过 10 次
- LLM 输出的内容与之前的输出几乎完全相同
- Token 消耗持续增长但没有进展
- 执行时间远超预期

### C.1.2 根本原因

**原因一：LLM 没有识别到任务已完成**。LLM 可能不知道什么条件意味着"任务完成"，因此一直在尝试。

**原因二：工具返回的结果不够明确**。如果工具返回的结果模糊不清，LLM 可能认为需要继续调用工具。

**原因三：缺乏最大迭代限制**。如果没有设置最大迭代次数，Agent 可能永远不会停止。

### C.1.3 解决方案

**方案一：设置最大迭代次数**。

```python
agent = create_react_agent(model=llm, tools=tools)
result = agent.invoke(input, config={"recursion_limit": 15})
```

**方案二：在系统提示中明确终止条件**。

```python
prompt = """当你得到足够信息时，直接给出最终答案。
不要继续调用工具。如果工具返回的结果足够回答问题，立即给出最终答案。"""
```

**方案三：检测重复调用**。

```python
def detect_loop(tool_calls: list, threshold: int = 3) -> bool:
    """检测工具调用循环"""
    if len(tool_calls) < threshold:
        return False
    last_three = [(tc["name"], str(tc["args"])) for tc in tool_calls[-threshold:]]
    return len(set(last_three)) == 1  # 三次调用完全相同
```

**方案四：监控执行状态**。

```python
def monitor_agent_execution(agent, input_data, max_time=60):
    """监控 Agent 执行，防止无限循环"""
    import time
    start_time = time.time()
    call_count = 0
    
    for chunk in agent.stream(input_data, stream_mode="updates"):
        elapsed = time.time() - start_time
        if elapsed > max_time:
            print(f"警告：执行时间超过 {max_time} 秒")
            break
        call_count += 1
        if call_count > 20:
            print("警告：调用次数超过 20 次")
            break
```

---

## C.2 工具调用失败

### C.2.1 常见错误类型

| 错误类型 | 表现 | 原因 |
|:---|:---|:---|
| **参数格式错误** | JSON 解析失败 | LLM 生成的参数不符合 schema |
| **参数值错误** | 工具执行异常 | 参数值超出合理范围 |
| **工具不存在** | "Tool not found" | 工具名称拼写错误 |
| **超时** | 执行超时 | 工具执行时间过长 |
| **权限不足** | 访问被拒绝 | API Key 或权限配置问题 |
| **网络错误** | 连接失败 | 网络不稳定或 API 不可用 |

### C.2.2 解决方案

```python
def robust_tool_executor(tool_calls, tools_map):
    """健壮的工具执行器"""
    results = []
    for tc in tool_calls:
        func_name = tc.function.name
        
        # 1. 参数解析
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError as e:
            results.append({"tool_call_id": tc.id, "content": f"参数解析错误：{str(e)}"})
            continue
        
        # 2. 工具存在性检查
        if func_name not in tools_map:
            results.append({"tool_call_id": tc.id, "content": f"工具 '{func_name}' 不存在。可用工具：{list(tools_map.keys())}"})
            continue
        
        # 3. 执行工具（带重试）
        for attempt in range(3):
            try:
                result = tools_map[func_name](**args)
                results.append({"tool_call_id": tc.id, "content": str(result)})
                break
            except TimeoutError:
                if attempt == 2:
                    results.append({"tool_call_id": tc.id, "content": f"工具 '{func_name}' 执行超时"})
            except Exception as e:
                if attempt == 2:
                    results.append({"tool_call_id": tc.id, "content": f"执行错误：{str(e)}"})
    
    return results
```

---

## C.3 Token 超限

### C.3.1 问题描述

当对话历史、工具返回结果和系统提示的总 Token 数超过 LLM 的上下文窗口限制时，会导致 `context_length_exceeded` 错误。

### C.3.2 解决方案

**方案一：压缩消息历史**。

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    max_token_limit=4000,
    return_messages=True,
)
```

**方案二：截断过长输出**。

```python
def truncate_output(output: str, max_len: int = 2000) -> str:
    """截断过长的工具输出"""
    if len(output) > max_len:
        return output[:max_len] + "\n...(已截断，共 " + str(len(output)) + " 字符)"
    return output
```

**方案三：使用更大上下文窗口的模型**。

```python
# 128K 上下文窗口
llm = ChatOpenAI(model="gpt-4o")  # 128K tokens

# 200K 上下文窗口
llm = ChatAnthropic(model="claude-sonnet-4-20250514")  # 200K tokens
```

**方案四：动态上下文管理**。

```python
def manage_context(messages, max_tokens=8000):
    """动态管理上下文大小"""
    encoder = tiktoken.encoding_for_model("gpt-4o")
    total = sum(len(encoder.encode(m.content or "")) for m in messages)
    
    if total <= max_tokens:
        return messages
    
    # 保留系统消息和最近消息
    system_msgs = [m for m in messages if m.type == "system"]
    other_msgs = [m for m in messages if m.type != "system"]
    
    recent = other_msgs[-6:]
    old = other_msgs[:-6]
    
    if old:
        summary = summarize_messages(old)
        return system_msgs + [SystemMessage(content=f"[摘要]\n{summary}")] + recent
    
    return messages
```

---

## C.4 Agent 幻觉

### C.4.1 问题描述

Agent 幻觉是指 LLM 生成了看似合理但实际错误的信息，特别是在工具返回结果不足或模糊时。

### C.4.2 解决方案

**方案一：在提示中强调基于事实**。

```python
prompt = """只使用工具返回的信息回答问题。
如果工具返回的信息不足以回答问题，诚实地说"我没有足够的信息来回答这个问题"。
不要编造或猜测任何信息。"""
```

**方案二：使用 RAG 增强事实性**。通过检索增强生成，为 LLM 提供可靠的信息来源。

**方案三：添加事实检查步骤**。

```python
def fact_check_node(state):
    """事实检查节点"""
    answer = state["messages"][-1].content
    sources = state.get("sources", [])
    
    # 验证回答是否基于检索到的来源
    check_prompt = f"""验证以下回答是否基于提供的来源。

回答：{answer}
来源：{sources}

如果回答包含来源中没有的信息，标记为"可能不准确"。"""
    
    check_result = llm.invoke([HumanMessage(content=check_prompt)])
    return {"fact_check": check_result.content}
```

---

## C.5 性能瓶颈

### C.5.1 常见瓶颈

| 瓶颈 | 症状 | 诊断方法 |
|:---|:---|:---|
| **LLM 延迟高** | 响应时间 > 5秒 | 检查模型和 Token 数 |
| **工具执行慢** | 工具调用耗时长 | 检查工具实现 |
| **检索质量差** | 检索结果不相关 | 检查分块和 Embedding |
| **内存溢出** | 进程被杀死 | 检查消息历史长度 |

### C.5.2 优化方案

| 瓶颈 | 优化方案 |
|:---|:---|
| LLM 延迟高 | 使用更快的模型（如 GPT-4o-mini）、压缩 Prompt、流式输出 |
| 工具执行慢 | 异步执行、缓存结果、优化工具实现 |
| 检索质量差 | 优化分块策略、使用 Rerank、调整 Embedding 模型 |
| 内存溢出 | 限制历史轮数、摘要压缩、使用更大的实例 |

---

## C.6 调试技巧

### C.6.1 启用详细日志

```python
import langchain
langchain.debug = True  # 启用 LangChain 调试模式
```

### C.6.2 使用 LangSmith 追踪

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_..."
os.environ["LANGCHAIN_PROJECT"] = "my-agent"
```

### C.6.3 查看 Agent 中间状态

```python
for chunk in agent.stream(input, stream_mode="updates"):
    for node, update in chunk.items():
        print(f"[{node}]")
        if "messages" in update:
            for msg in update["messages"]:
                print(f"  {msg.type}: {msg.content[:200]}")
```

### C.6.4 使用 LangSmith 评估

```python
# 在 LangSmith UI 中可以：
# 1. 查看完整的执行链路
# 2. 分析每一步的输入输出
# 3. 追踪 Token 消耗
# 4. 识别性能瓶颈
# 5. 对比不同 Prompt 的效果
```

---

## C.7 其他常见问题

### C.7.1 工具描述不清晰

**症状**：LLM 选择错误的工具或生成错误的参数。

**解决方案**：优化工具描述，添加使用示例和约束条件。

### C.7.2 多轮对话上下文丢失

**症状**：Agent 在多轮对话中忘记之前的信息。

**解决方案**：使用 Memory 系统（如 ConversationSummaryMemory）维护上下文。

### C.7.3 Agent 输出格式不正确

**症状**：Agent 的输出不符合预期格式。

**解决方案**：在系统提示中明确输出格式要求，使用 Structured Output。

### C.7.4 并发请求冲突

**症状**：多个并发请求互相干扰。

**解决方案**：为每个会话使用独立的 thread_id，确保状态隔离。

---

## C.8 调试工具推荐

| 工具 | 用途 | 链接 |
|:---|:---|:---|
| **LangSmith** | 追踪、调试、评估 | https://smith.langchain.com/ |
| **Langfuse** | 开源可观测性 | https://langfuse.com/ |
| **Playwright** | Web Agent 调试 | https://playwright.dev/ |
| **Rich** | 终端输出美化 | https://rich.readthedocs.io/ |
| **tiktoken** | Token 计数 | https://github.com/openai/tiktoken |

---

## C.9 Agent 架构常见问题

### C.9.1 状态管理问题

**问题：状态在节点之间丢失**。

在 LangGraph 中，状态通过 TypedDict 定义。如果节点返回的状态字段与定义不匹配，可能会导致状态丢失。

```python
# 正确：返回的状态字段与定义匹配
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    count: int

def my_node(state: AgentState):
    return {"messages": [response], "count": state.get("count", 0) + 1}
```

### C.9.2 条件边问题

**问题：条件边的路由函数返回值不匹配**。

```python
# 错误：路由函数返回值不在映射中
graph.add_conditional_edges("node", router, {"a": "node_a", "b": "node_b"})
# 如果 router 返回 "c"，会报错

# 正确：确保所有可能的返回值都在映射中
graph.add_conditional_edges("node", router, {"a": "node_a", "b": "node_b", "default": END})
```

### C.9.3 子图数据传递问题

**问题：子图之间无法正确传递数据**。

子图之间通过状态传递数据。确保子图的输入输出状态字段一致。

---

## C.10 记忆系统常见问题

### C.10.1 记忆溢出

**问题：记忆系统占用过多内存**。

**解决方案**：
- 使用 Window Memory 限制记忆大小
- 使用 Summary Memory 压缩旧记忆
- 定期清理过期记忆

### C.10.2 跨会话记忆丢失

**问题：Agent 在新会话中忘记之前的信息**。

**解决方案**：
- 使用持久化存储（如 Redis、PostgreSQL）
- 实现会话 ID 管理
- 定期同步记忆到持久化存储

### C.10.3 记忆检索不准确

**问题：从记忆中检索到的信息与当前查询不相关**。

**解决方案**：
- 优化 Embedding 模型
- 使用混合检索（BM25 + 向量检索）
- 调整检索参数（top_k、相似度阈值）

---

## C.11 RAG 系统常见问题

### C.11.1 检索质量差

**问题：检索到的文档与查询不相关**。

**解决方案**：
- 优化分块策略（调整 chunk_size 和 chunk_overlap）
- 使用更好的 Embedding 模型
- 添加重排序（Rerank）步骤
- 使用 Multi-Query 扩展查询

### C.11.2 生成质量差

**问题：LLM 基于检索结果生成的回答不准确**。

**解决方案**：
- 优化 Prompt 模板
- 增加上下文数量（top_k）
- 使用更好的 LLM 模型
- 添加事实检查步骤

### C.11.3 检索延迟高

**问题：RAG 系统的响应时间过长**。

**解决方案**：
- 使用更快的 Embedding 模型
- 优化向量数据库索引
- 缓存常见查询结果
- 使用异步检索

---

## C.12 多 Agent 系统常见问题

### C.12.1 Agent 间通信失败

**问题：Agent 之间无法正确传递消息**。

**解决方案**：
- 使用标准消息格式
- 实现消息验证机制
- 添加重试和超时处理

### C.12.2 任务分配不均

**问题：某些 Agent 负载过重，其他 Agent 空闲**。

**解决方案**：
- 实现负载均衡策略
- 动态调整任务分配
- 监控 Agent 负载指标

### C.12.3 死锁

**问题：多个 Agent 互相等待对方完成**。

**解决方案**：
- 设置超时机制
- 实现死锁检测
- 使用优先级队列

---

## C.13 安全相关问题

### C.13.1 提示注入攻击

**问题：恶意用户通过输入操纵 Agent 行为**。

**解决方案**：
- 输入验证和过滤
- 指令层级隔离
- 输出过滤
- 审计日志

### C.12.2 数据泄露

**问题：Agent 输出敏感信息**。

**解决方案**：
- 输出过滤敏感信息
- 使用 Access Control 限制数据访问
- 记录所有数据访问日志

---

## C.14 性能优化清单

| 优化项 | 预期效果 | 实现难度 |
|:---|:---|:---|
| 模型路由 | 降低 40-60% 成本 | 中 |
| 语义缓存 | 降低 30-50% 成本 | 中 |
| Prompt 压缩 | 降低 20-30% Token | 低 |
| 并行工具调用 | 降低 50% 延迟 | 中 |
| 流式输出 | 降低感知延迟 | 低 |
| 异步执行 | 降低阻塞时间 | 中 |

---

## C.15 调试工作流

### C.15.1 标准调试流程

1. **复现问题**：确保问题可以稳定复现
2. **收集信息**：记录输入、输出、错误信息、执行日志
3. **定位问题**：使用 LangSmith 追踪执行链路
4. **分析原因**：检查 Prompt、工具描述、状态管理
5. **实施修复**：修改代码或配置
6. **验证修复**：重新运行测试
7. **记录经验**：更新文档和知识库

### C.15.2 调试工具使用

```python
# 1. 启用调试模式
import langchain
langchain.debug = True

# 2. 使用 LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# 3. 打印中间状态
for chunk in agent.stream(input, stream_mode="updates"):
    for node, update in chunk.items():
        print(f"[{node}] {update}")

# 4. 使用 Python debugger
import pdb; pdb.set_trace()
```

---

## C.16 Prompt 工程常见问题

### C.16.1 Prompt 过长导致上下文溢出

**问题**：System Prompt 太长，占用了大部分上下文窗口。

**解决方案**：
- 精简 System Prompt，去除冗余信息
- 将详细说明移到外部文档，通过 RAG 检索
- 使用更小的 Prompt，依赖 Few-shot 示例

### C.16.2 Prompt 中的指令冲突

**问题**：System Prompt 中的不同指令相互矛盾。

**解决方案**：
- 检查并消除矛盾的指令
- 使用层级结构组织指令
- 安全约束放在最前面

### C.16.3 Few-shot 示例选择不当

**问题**：Few-shot 示例与实际任务不匹配。

**解决方案**：
- 使用动态示例选择（SemanticSimilarityExampleSelector）
- 确保示例覆盖典型场景
- 定期更新示例库

---

## C.17 框架集成常见问题

### C.17.1 LangChain 版本兼容性

**问题**：不同版本的 LangChain API 不兼容。

**解决方案**：
- 固定版本号：`pip install langchain==0.1.0`
- 查看迁移指南
- 使用最新稳定版本

### C.17.2 LangGraph 状态定义错误

**问题**：TypedDict 定义的状态与节点返回值不匹配。

**解决方案**：
- 确保节点返回的状态字段与 TypedDict 定义一致
- 使用 `Annotated[list, operator.add]` 控制列表合并

### C.17.3 AutoGen 对话无限循环

**问题**：AutoGen Agent 之间无限对话。

**解决方案**：
- 设置 `max_consecutive_auto_reply`
- 设置 `max_round` 限制群聊轮数
- 在系统提示中明确终止条件

---

## C.18 部署相关问题

### C.18.1 Docker 容器启动失败

**问题**：Agent 服务在 Docker 容器中无法启动。

**解决方案**：
- 检查 Dockerfile 中的依赖安装
- 检查环境变量是否正确传递
- 检查端口映射
- 查看容器日志：`docker logs <container_id>`

### C.18.2 Kubernetes Pod 重启

**问题**：Agent Pod 频繁重启。

**解决方案**：
- 检查资源限制（CPU、内存）
- 检查 Liveness Probe 配置
- 检查 OOM Kill 日志
- 增加资源配额

### C.18.3 API 限流

**问题**：LLM API 返回 429 Too Many Requests。

**解决方案**：
- 实现指数退避重试
- 使用速率限制器
- 分散请求到多个 API Key
- 使用缓存减少 API 调用

---

## C.19 数据相关问题

### C.19.1 向量数据库索引损坏

**问题**：向量数据库索引损坏，无法检索。

**解决方案**：
- 定期备份向量数据库
- 实现索引重建机制
- 使用持久化存储

### C.19.2 Embedding 不一致

**问题**：不同版本的 Embedding 模型生成的向量不兼容。

**解决方案**：
- 固定 Embedding 模型版本
- 重新索引所有文档
- 使用兼容的模型版本

### C.19.3 文档分块质量差

**问题**：分块后的文档片段不完整或不相关。

**解决方案**：
- 调整 chunk_size 和 chunk_overlap
- 使用语义分块而非固定长度分块
- 保留文档结构信息

---

## C.20 安全相关问题进阶

### C.20.1 间接提示注入防御

```python
class IndirectInjectionDefense:
    def sanitize(self, content: str) -> str:
        """清洗检索到的内容"""
        import re
        patterns = [
            r'ignore\s+(previous|all)\s+instructions',
            r'忽略.*指令',
            r'system\s*prompt',
        ]
        for p in patterns:
            if re.search(p, content, re.IGNORECASE):
                return "[内容安全检查未通过，已过滤]"
        return content
```

### C.20.2 工具权限控制

```python
class ToolPermissionManager:
    def __init__(self):
        self.permissions = {
            "read": "allow",
            "write": "require_approval",
            "delete": "deny",
            "execute": "require_approval",
        }
    
    def check_permission(self, tool_name, action):
        perm = self.permissions.get(action, "deny")
        if perm == "deny":
            return False, f"操作 '{action}' 被禁止"
        if perm == "require_approval":
            return None, f"操作 '{action}' 需要人工审批"
        return True, "允许"
```

---

## C.21 监控与告警

### C.21.1 关键监控指标

| 指标 | 说明 | 告警阈值 |
|:---|:---|:---|
| 请求延迟 P95 | 95% 的请求延迟 | > 30 秒 |
| 错误率 | 失败请求比例 | > 5% |
| Token 消耗 | 每次请求的 Token 数 | > 10000 |
| 工具调用成功率 | 工具调用成功比例 | < 90% |
| 并发请求数 | 同时处理的请求数 | > 100 |

### C.21.2 告警配置

```yaml
groups:
  - name: agent_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(agent_requests_total{status="error"}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Agent 错误率过高"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, agent_latency_seconds) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Agent P95 延迟超过 30 秒"
      
      - alert: HighTokenUsage
        expr: rate(agent_tokens_total[5m]) > 100000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Agent Token 消耗过高"
```

---

## C.22 测试策略

### C.22.1 单元测试

```python
def test_tool_execution():
    """测试工具执行"""
    tool = MyTool()
    result = tool.run({"query": "test"})
    assert result is not None
    assert "error" not in result.lower()

def test_state_management():
    """测试状态管理"""
    state = {"messages": [], "count": 0}
    new_state = my_node(state)
    assert new_state["count"] == 1
```

### C.22.2 集成测试

```python
def test_agent_integration():
    """测试 Agent 完整流程"""
    agent = create_react_agent(model=llm, tools=tools)
    result = agent.invoke({"messages": [("user", "测试问题")]})
    assert result["messages"][-1].content is not None
    assert len(result["messages"]) > 1
```

### C.22.3 端到端测试

```python
def test_end_to_end():
    """端到端测试"""
    # 模拟真实用户场景
    test_cases = [
        {"input": "简单查询", "expected_contains": "回答"},
        {"input": "工具调用", "expected_contains": "结果"},
        {"input": "多步推理", "expected_contains": "分析"},
    ]
    
    for case in test_cases:
        result = agent.invoke({"messages": [("user", case["input"])]})
        assert case["expected_contains"] in result["messages"][-1].content
```

---

## C.23 性能优化清单

| 优化项 | 预期效果 | 实现难度 | 优先级 |
|:---|:---|:---|:---|
| 模型路由 | 降低 40-60% 成本 | 中 | 高 |
| 语义缓存 | 降低 30-50% 成本 | 中 | 高 |
| Prompt 压缩 | 降低 20-30% Token | 低 | 中 |
| 并行工具调用 | 降低 50% 延迟 | 中 | 高 |
| 流式输出 | 降低感知延迟 | 低 | 中 |
| 异步执行 | 降低阻塞时间 | 中 | 中 |
| 本地模型 | 降低 API 成本 | 高 | 低 |

---

## C.24 调试工作流

### C.24.1 标准调试流程

1. **复现问题**：确保问题可以稳定复现
2. **收集信息**：记录输入、输出、错误信息、执行日志
3. **定位问题**：使用 LangSmith 追踪执行链路
4. **分析原因**：检查 Prompt、工具描述、状态管理
5. **实施修复**：修改代码或配置
6. **验证修复**：重新运行测试
7. **记录经验**：更新文档和知识库

### C.24.2 调试工具使用

```python
# 1. 启用调试模式
import langchain
langchain.debug = True

# 2. 使用 LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# 3. 打印中间状态
for chunk in agent.stream(input, stream_mode="updates"):
    for node, update in chunk.items():
        print(f"[{node}] {update}")

# 4. 使用 Rich 美化输出
from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="Agent 执行过程")
table.add_column("步骤", style="cyan")
table.add_column("操作", style="green")
table.add_column("结果", style="yellow")

for i, step in enumerate(execution_steps):
    table.add_row(str(i+1), step["action"], step["result"][:50])

console.print(table)
```

---

## C.25 框架特定问题

### C.25.1 LangChain 相关问题

**问题：LangChain 版本升级后 API 不兼容**。

解决方案：查看官方迁移指南，使用兼容性包装器，逐步迁移代码。

**问题：LCEL 链执行失败**。

解决方案：检查每个 Runnable 的输入输出类型是否匹配，使用 `with_config` 添加调试信息。

### C.25.2 LangGraph 相关问题

**问题：条件边的路由函数返回值不在映射中**。

```python
# 确保所有可能的返回值都在映射中
graph.add_conditional_edges("node", router, {
    "a": "node_a",
    "b": "node_b",
    "default": END  # 兜底路径
})
```

**问题：子图状态无法正确传递**。

确保子图的输入输出状态字段与主图一致。

### C.25.3 AutoGen 相关问题

**问题：Agent 之间无限对话**。

解决方案：
- 设置 `max_consecutive_auto_reply`
- 设置 `max_round` 限制群聊轮数
- 在系统提示中明确终止条件

### C.25.4 CrewAI 相关问题

**问题：任务依赖未正确设置**。

解决方案：使用 `context` 参数明确任务依赖关系。

---

## C.26 部署相关问题

### C.26.1 Docker 容器启动失败

**检查清单**：
1. Dockerfile 中的依赖是否正确安装
2. 环境变量是否正确传递
3. 端口映射是否正确
4. 容器日志：`docker logs <container_id>`

### C.26.2 Kubernetes Pod 重启

**检查清单**：
1. 资源限制（CPU、内存）是否足够
2. Liveness Probe 配置是否正确
3. 是否 OOM Kill
4. 增加资源配额

### C.26.3 API 限流

**解决方案**：
- 实现指数退避重试
- 使用速率限制器
- 分散请求到多个 API Key
- 使用缓存减少 API 调用

---

## C.27 数据相关问题

### C.27.1 向量数据库索引损坏

**解决方案**：
- 定期备份向量数据库
- 实现索引重建机制
- 使用持久化存储

### C.27.2 Embedding 不一致

**解决方案**：
- 固定 Embedding 模型版本
- 重新索引所有文档
- 使用兼容的模型版本

### C.27.3 文档分块质量差

**解决方案**：
- 调整 chunk_size 和 chunk_overlap
- 使用语义分块而非固定长度分块
- 保留文档结构信息

---

## C.28 监控与告警

### C.28.1 关键监控指标

| 指标 | 说明 | 告警阈值 |
|:---|:---|:---|
| 请求延迟 P95 | 95% 的请求延迟 | > 30 秒 |
| 错误率 | 失败请求比例 | > 5% |
| Token 消耗 | 每次请求的 Token 数 | > 10000 |
| 工具调用成功率 | 工具调用成功比例 | < 90% |

### C.28.2 告警配置

```yaml
groups:
  - name: agent_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(agent_requests_total{status="error"}[5m]) > 0.1
        labels:
          severity: warning
      - alert: HighLatency
        expr: histogram_quantile(0.95, agent_latency_seconds) > 30
        labels:
          severity: warning
```

---

## C.29 测试策略

### C.29.1 单元测试

```python
def test_tool_execution():
    tool = MyTool()
    result = tool.run({"query": "test"})
    assert result is not None
```

### C.29.2 集成测试

```python
def test_agent_integration():
    agent = create_react_agent(model=llm, tools=tools)
    result = agent.invoke({"messages": [("user", "测试问题")]})
    assert result["messages"][-1].content is not None
```

### C.29.3 端到端测试

```python
def test_end_to_end():
    test_cases = [
        {"input": "简单查询", "expected": "回答"},
        {"input": "工具调用", "expected": "结果"},
    ]
    for case in test_cases:
        result = agent.invoke({"messages": [("user", case["input"])]})
        assert case["expected"] in result["messages"][-1].content
```

---

## C.30 记忆系统深度调试

### C.30.1 记忆衰减调试

```python
import math
from datetime import datetime

class MemoryDebugger:
    """记忆系统调试器"""
    
    def analyze_memory_strength(self, memories):
        """分析记忆强度分布"""
        for mem in memories:
            hours = (datetime.now() - mem.created_at).total_seconds() / 3600
            decay = math.exp(-0.1 * hours)
            access_boost = 1 + 0.2 * mem.access_count
            strength = mem.importance * decay * access_boost
            
            print(f"内容: {mem.content[:50]}...")
            print(f"  创建时间: {mem.created_at}")
            print(f"  访问次数: {mem.access_count}")
            print(f"  原始重要性: {mem.importance}")
            print(f"  时间衰减: {decay:.4f}")
            print(f"  访问强化: {access_boost:.4f}")
            print(f"  最终强度: {strength:.4f}")
            print()
    
    def diagnose_decay_issue(self, memories):
        """诊断衰减问题"""
        issues = []
        
        # 检查是否有记忆衰减过快
        for mem in memories:
            hours = (datetime.now() - mem.created_at).total_seconds() / 3600
            if hours < 1 and mem.importance > 0.8:
                strength = mem.importance * math.exp(-0.1 * hours)
                if strength < 0.5:
                    issues.append(f"高重要性记忆衰减过快: {mem.content[:30]}...")
        
        # 检查是否有记忆从未被访问
        for mem in memories:
            if mem.access_count == 0 and mem.importance > 0.5:
                issues.append(f"高重要性记忆从未被访问: {mem.content[:30]}...")
        
        return issues
```

### C.30.2 记忆检索调试

```python
class MemoryRetrievalDebugger:
    """记忆检索调试器"""
    
    def diagnose_retrieval(self, query, memories, retriever, top_k=5):
        """诊断记忆检索问题"""
        print(f"查询: {query}")
        print(f"记忆总数: {len(memories)}")
        
        # 执行检索
        results = retriever.invoke(query)
        
        print(f"检索结果数: {len(results)}")
        for i, doc in enumerate(results[:top_k]):
            print(f"  {i+1}. {doc.page_content[:100]}...")
            print(f"     相关性分数: {doc.metadata.get('score', 'N/A')}")
        
        # 检查是否有相关记忆未被检索到
        relevant_count = sum(1 for mem in memories if self._is_relevant(query, mem.content))
        retrieved_count = len(results)
        
        print(f"\n相关记忆数: {relevant_count}")
        print(f"检索到的相关记忆数: {retrieval_count}")
        print(f"召回率: {retrieval_count / relevant_count if relevant_count > 0 else 0:.2%}")
```

---

## C.31 RAG 系统深度调试

### C.31.1 检索质量调试

```python
class RAGDebugger:
    """RAG 系统调试器"""
    
    def diagnose_retrieval(self, query, docs, top_k=5):
        """诊断检索质量"""
        print(f"查询: {query}")
        print(f"检索到的文档数: {len(docs)}")
        
        for i, doc in enumerate(docs[:top_k]):
            relevance = self._estimate_relevance(query, doc.page_content)
            print(f"  {i+1}. 相关性: {relevance:.2f}")
            print(f"     内容: {doc.page_content[:100]}...")
            print(f"     来源: {doc.metadata.get('source', 'N/A')}")
            print()
    
    def _estimate_relevance(self, query, content):
        """估算查询与内容的相关性"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        overlap = len(query_words & content_words)
        return overlap / len(query_words) if query_words else 0
    
    def diagnose_generation(self, query, context, answer):
        """诊断生成质量"""
        print(f"查询: {query}")
        print(f"上下文长度: {len(context)} 字符")
        print(f"回答长度: {len(answer)} 字符")
        
        # 检查回答是否基于上下文
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(context_words & answer_words)
        coverage = overlap / len(answer_words) if answer_words else 0
        
        print(f"上下文覆盖率: {coverage:.2%}")
        if coverage < 0.3:
            print("警告: 回答可能未基于上下文")
```

---

## C.32 多 Agent 系统深度调试

### C.32.1 Agent 间通信调试

```python
class MultiAgentDebugger:
    """多 Agent 系统调试器"""
    
    def diagnose_communication(self, messages):
        """诊断 Agent 间通信"""
        print(f"消息总数: {len(messages)}")
        
        for i, msg in enumerate(messages):
            sender = msg.get("sender", "unknown")
            receiver = msg.get("receiver", "unknown")
            msg_type = msg.get("type", "unknown")
            content = msg.get("content", "")[:100]
            
            print(f"  {i+1}. [{sender}] -> [{receiver}] ({msg_type})")
            print(f"     {content}...")
        
        # 检查是否有未回复的消息
        unanswered = self._find_unanswered(messages)
        if unanswered:
            print(f"\n警告: {len(unanswered)} 条消息未被回复")
    
    def _find_unanswered(self, messages):
        """查找未回复的消息"""
        unanswered = []
        sent = set()
        replied = set()
        
        for msg in messages:
            if msg.get("type") == "request":
                sent.add(msg.get("id"))
            elif msg.get("type") == "response":
                replied.add(msg.get("in_reply_to"))
        
        for msg in messages:
            if msg.get("type") == "request" and msg.get("id") not in replied:
                unanswered.append(msg)
        
        return unanswered
```

---

## C.33 安全深度调试

### C.33.1 提示注入检测

```python
class InjectionDetector:
    """提示注入检测器"""
    
    def __init__(self):
        self.patterns = [
            (r"忽略.*指令", "直接注入"),
            (r"ignore.*instruction", "直接注入"),
            (r"你现在是", "角色扮演"),
            (r"system prompt", "系统提示泄露"),
            (r"输出.*密码", "敏感信息泄露"),
            (r"发送.*数据.*到", "数据外泄"),
        ]
    
    def detect(self, user_input):
        """检测提示注入"""
        threats = []
        for pattern, threat_type in self.patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                threats.append({
                    "type": threat_type,
                    "pattern": pattern,
                    "input": user_input[:100]
                })
        return threats
```

### C.33.2 工具权限审计

```python
class ToolPermissionAuditor:
    """工具权限审计器"""
    
    def audit(self, tool_calls, permissions):
        """审计工具调用权限"""
        violations = []
        
        for tc in tool_calls:
            tool_name = tc.get("name")
            action = tc.get("action", "execute")
            
            perm = permissions.get(tool_name, {}).get(action, "deny")
            
            if perm == "deny":
                violations.append({
                    "tool": tool_name,
                    "action": action,
                    "reason": "操作被禁止"
                })
            elif perm == "require_approval":
                violations.append({
                    "tool": tool_name,
                    "action": action,
                    "reason": "需要人工审批"
                })
        
        return violations
```

---

## C.34 性能优化深度指南

### C.34.1 延迟优化

```python
class LatencyOptimizer:
    """延迟优化器"""
    
    def optimize(self, agent_config):
        """优化 Agent 延迟"""
        optimizations = []
        
        # 1. 模型选择
        if agent_config.get("model") == "gpt-4o":
            optimizations.append({
                "type": "model_downgrade",
                "description": "使用 GPT-4o-mini 处理简单任务",
                "expected_saving": "50-70% 延迟"
            })
        
        # 2. 流式输出
        if not agent_config.get("streaming"):
            optimizations.append({
                "type": "enable_streaming",
                "description": "启用流式输出降低感知延迟",
                "expected_saving": "30-50% 感知延迟"
            })
        
        # 3. 缓存
        if not agent_config.get("cache"):
            optimizations.append({
                "type": "enable_cache",
                "description": "启用语义缓存",
                "expected_saving": "30-50% 成本"
            })
        
        return optimizations
```

### C.34.2 成本优化

```python
class CostOptimizer:
    """成本优化器"""
    
    def analyze_cost(self, api_calls):
        """分析 API 调用成本"""
        total_input_tokens = sum(call["input_tokens"] for call in api_calls)
        total_output_tokens = sum(call["output_tokens"] for call in api_calls)
        
        # GPT-4o 价格
        input_cost = total_input_tokens * 2.50 / 1_000_000
        output_cost = total_output_tokens * 10.00 / 1_000_000
        total_cost = input_cost + output_cost
        
        return {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    
    def suggest_optimizations(self, cost_analysis):
        """建议优化方案"""
        suggestions = []
        
        if cost_analysis["total_input_tokens"] > 1000000:
            suggestions.append("考虑使用 Prompt 压缩减少输入 Token")
        
        if cost_analysis["total_output_tokens"] > 500000:
            suggestions.append("考虑使用更小的模型处理简单任务")
        
        return suggestions
```

---

## C.35 部署深度调试

### C.35.1 Docker 调试

```python
class DockerDebugger:
    """Docker 调试器"""
    
    def diagnose_container(self, container_id):
        """诊断容器问题"""
        import subprocess
        
        # 检查容器状态
        result = subprocess.run(
            ["docker", "inspect", container_id],
            capture_output=True, text=True
        )
        info = json.loads(result.stdout)
        
        status = info[0]["State"]["Status"]
        oom_killed = info[0]["State"].get("OOMKilled", False)
        restart_count = info[0]["RestartPolicy"]["MaximumRetryCount"]
        
        diagnostics = {
            "status": status,
            "oom_killed": oom_killed,
            "restart_count": restart_count,
        }
        
        if oom_killed:
            diagnostics["recommendation"] = "增加内存限制"
        
        return diagnostics
```

### C.35.2 Kubernetes 调试

```python
class K8sDebugger:
    """Kubernetes 调试器"""
    
    def diagnose_pod(self, pod_name, namespace="default"):
        """诊断 Pod 问题"""
        import subprocess
        
        # 检查 Pod 状态
        result = subprocess.run(
            ["kubectl", "get", "pod", pod_name, "-n", namespace, "-o", "json"],
            capture_output=True, text=True
        )
        info = json.loads(result.stdout)
        
        status = info["status"]["phase"]
        restart_count = info["status"].get("containerStatuses", [{}])[0].get("restartCount", 0)
        
        diagnostics = {
            "status": status,
            "restart_count": restart_count,
        }
        
        if restart_count > 0:
            diagnostics["recommendation"] = "检查 Pod 日志，可能存在 OOM 或启动失败"
        
        return diagnostics
```

---

## C.36 调试工具推荐

| 工具 | 用途 | 链接 |
|:---|:---|:---|
| **LangSmith** | 追踪、调试、评估 | https://smith.langchain.com/ |
| **Langfuse** | 开源可观测性 | https://langfuse.com/ |
| **Playwright** | Web Agent 调试 | https://playwright.dev/ |
| **Rich** | 终端输出美化 | https://rich.readthedocs.io/ |
| **tiktoken** | Token 计数 | https://github.com/openai/tiktoken |
| **Prometheus** | 指标监控 | https://prometheus.io/ |
| **Grafana** | 指标可视化 | https://grafana.com/ |

---

## C.37 调试工作流

### C.37.1 标准调试流程

1. **复现问题**：确保问题可以稳定复现
2. **收集信息**：记录输入、输出、错误信息、执行日志
3. **定位问题**：使用 LangSmith 追踪执行链路
4. **分析原因**：检查 Prompt、工具描述、状态管理
5. **实施修复**：修改代码或配置
6. **验证修复**：重新运行测试
7. **记录经验**：更新文档和知识库

### C.37.2 调试工具使用

```python
# 1. 启用调试模式
import langchain
langchain.debug = True

# 2. 使用 LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# 3. 打印中间状态
for chunk in agent.stream(input, stream_mode="updates"):
    for node, update in chunk.items():
        print(f"[{node}] {update}")

# 4. 使用 Rich 美化输出
from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="Agent 执行过程")
table.add_column("步骤", style="cyan")
table.add_column("操作", style="green")
table.add_column("结果", style="yellow")
```

---

## C.38 常见错误代码对照表

| 错误代码 | 含义 | 解决方案 |
|:---|:---|:---|
| `context_length_exceeded` | Token 超限 | 压缩历史、截断输出 |
| `invalid_api_key` | API Key 无效 | 检查 API Key 配置 |
| `rate_limit_exceeded` | 请求频率过高 | 降低请求频率、使用缓存 |
| `model_not_found` | 模型不存在 | 检查模型名称 |
| `tool_not_found` | 工具不存在 | 检查工具注册 |
| `json_parse_error` | JSON 解析失败 | 检查参数格式 |
| `timeout` | 执行超时 | 增加超时时间 |
| `connection_error` | 连接失败 | 检查网络连接 |

---

## C.39 高级调试技巧

### C.39.1 使用 Python Debugger

```python
# 在 Agent 执行的关键点设置断点
import pdb

def debug_agent_step(state):
    """调试 Agent 步骤"""
    pdb.set_trace()  # 设置断点
    # 在断点处可以检查 state 的内容
    print(f"当前状态: {state}")
    # 使用 n (next) 单步执行
    # 使用 c (continue) 继续执行
    # 使用 p (print) 打印变量
    # 使用 q (quit) 退出调试
```

### C.39.2 使用 logging 模块

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 在 Agent 中使用日志
def agent_step(state):
    logger.debug(f"输入状态: {state}")
    # ... 执行逻辑 ...
    logger.debug(f"输出状态: {new_state}")
```

### C.39.3 使用 LangSmith 评估

```python
# 在 LangSmith UI 中可以：
# 1. 查看完整的执行链路
# 2. 分析每一步的输入输出
# 3. 追踪 Token 消耗
# 4. 识别性能瓶颈
# 5. 对比不同 Prompt 的效果
# 6. 进行 A/B 测试
```

---

## C.40 常见场景调试指南

### C.40.1 场景一：Agent 无法理解用户意图

**症状**：Agent 的回答与用户问题无关。

**调试步骤**：
1. 检查 System Prompt 是否清晰定义了 Agent 的角色
2. 检查是否提供了足够的上下文
3. 尝试使用 Few-shot 示例引导
4. 检查 LLM 的 temperature 设置

### C.40.2 场景二：工具调用参数错误

**症状**：LLM 生成的工具调用参数不符合 JSON Schema。

**调试步骤**：
1. 检查工具的 parameters 定义是否准确
2. 在工具描述中添加参数示例
3. 使用 StructuredTool 代替 @tool
4. 添加参数校验逻辑

### C.40.3 场景三：RAG 检索质量差

**症状**：检索到的文档与查询不相关。

**调试步骤**：
1. 检查分块策略（chunk_size 和 chunk_overlap）
2. 尝试不同的 Embedding 模型
3. 添加重排序（Rerank）步骤
4. 使用 Multi-Query 扩展查询

### C.40.4 场景四：多 Agent 协作失败

**症状**：Agent 之间无法正确协作。

**调试步骤**：
1. 检查 Agent 的系统提示是否明确
2. 检查消息格式是否一致
3. 检查是否有死锁
4. 添加超时和重试机制

### C.40.5 场景五：Agent 输出格式不正确

**症状**：Agent 的输出不符合预期格式。

**调试步骤**：
1. 在系统提示中明确输出格式
2. 使用 Structured Output
3. 添加输出解析逻辑
4. 在工具描述中说明返回格式

---

## C.41 调试检查清单

### C.41.1 启动前检查

- [ ] 环境变量已配置
- [ ] API Key 有效
- [ ] 依赖包已安装
- [ ] 配置文件正确

### C.41.2 运行时检查

- [ ] 最大迭代次数已设置
- [ ] 超时时间已设置
- [ ] 错误处理已实现
- [ ] 日志已启用

### C.41.3 性能检查

- [ ] 延迟在可接受范围内
- [ ] Token 消耗在预算内
- [ ] 错误率低于阈值
- [ ] 资源使用正常

### C.41.4 安全检查

- [ ] 输入验证已实现
- [ ] 工具权限已配置
- [ ] 输出过滤已启用
- [ ] 审计日志已记录

---

## C.42 性能优化清单

| 优化项 | 预期效果 | 实现难度 | 优先级 |
|:---|:---|:---|:---|
| 模型路由 | 降低 40-60% 成本 | 中 | 高 |
| 语义缓存 | 降低 30-50% 成本 | 中 | 高 |
| Prompt 压缩 | 降低 20-30% Token | 低 | 中 |
| 并行工具调用 | 降低 50% 延迟 | 中 | 高 |
| 流式输出 | 降低感知延迟 | 低 | 中 |
| 异步执行 | 降低阻塞时间 | 中 | 中 |
| 本地模型 | 降低 API 成本 | 高 | 低 |

---

## C.43 调试工作流

### C.43.1 标准调试流程

1. **复现问题**：确保问题可以稳定复现
2. **收集信息**：记录输入、输出、错误信息、执行日志
3. **定位问题**：使用 LangSmith 追踪执行链路
4. **分析原因**：检查 Prompt、工具描述、状态管理
5. **实施修复**：修改代码或配置
6. **验证修复**：重新运行测试
7. **记录经验**：更新文档和知识库

### C.43.2 调试工具使用

```python
# 1. 启用调试模式
import langchain
langchain.debug = True

# 2. 使用 LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# 3. 打印中间状态
for chunk in agent.stream(input, stream_mode="updates"):
    for node, update in chunk.items():
        print(f"[{node}] {update}")

# 4. 使用 Rich 美化输出
from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="Agent 执行过程")
table.add_column("步骤", style="cyan")
table.add_column("操作", style="green")
table.add_column("结果", style="yellow")
```

---

## C.44 常见错误代码对照表

| 错误代码 | 含义 | 解决方案 |
|:---|:---|:---|
| `context_length_exceeded` | Token 超限 | 压缩历史、截断输出 |
| `invalid_api_key` | API Key 无效 | 检查 API Key 配置 |
| `rate_limit_exceeded` | 请求频率过高 | 降低请求频率、使用缓存 |
| `model_not_found` | 模型不存在 | 检查模型名称 |
| `tool_not_found` | 工具不存在 | 检查工具注册 |
| `json_parse_error` | JSON 解析失败 | 检查参数格式 |
| `timeout` | 执行超时 | 增加超时时间 |
| `connection_error` | 连接失败 | 检查网络连接 |

---

## C.45 框架特定问题深度解析

### C.45.1 LangChain 深度问题

**问题：LCEL 链执行时类型不匹配**。

```python
# 错误：Runnable 类型不匹配
chain = prompt | llm | StrOutputParser()
# 如果 prompt 输出的不是 dict，会导致错误

# 正确：确保每个 Runnable 的输入输出类型一致
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

**问题：LangGraph 状态更新冲突**。

当多个节点同时更新同一个状态字段时，可能会产生冲突。使用 `Annotated[list, operator.add]` 来控制合并策略。

### C.45.2 AutoGen 深度问题

**问题：GroupChat 中某些 Agent 没有发言机会**。

解决方案：使用 `round_robin` 策略确保每个 Agent 都有机会发言。

### C.45.3 CrewAI 深度问题

**问题：任务依赖未正确设置导致执行顺序错误**。

解决方案：使用 `context` 参数明确任务依赖关系。

---

## C.46 部署深度调试

### C.46.1 Docker 调试

```python
class DockerDebugger:
    def diagnose_container(self, container_id):
        import subprocess
        result = subprocess.run(
            ["docker", "inspect", container_id],
            capture_output=True, text=True
        )
        info = json.loads(result.stdout)
        status = info[0]["State"]["Status"]
        oom_killed = info[0]["State"].get("OOMKilled", False)
        return {"status": status, "oom_killed": oom_killed}
```

### C.46.2 Kubernetes 调试

```python
class K8sDebugger:
    def diagnose_pod(self, pod_name, namespace="default"):
        import subprocess
        result = subprocess.run(
            ["kubectl", "get", "pod", pod_name, "-n", namespace, "-o", "json"],
            capture_output=True, text=True
        )
        info = json.loads(result.stdout)
        status = info["status"]["phase"]
        restart_count = info["status"].get("containerStatuses", [{}])[0].get("restartCount", 0)
        return {"status": status, "restart_count": restart_count}
```

---

## C.47 调试工具推荐

| 工具 | 用途 | 链接 |
|:---|:---|:---|
| **LangSmith** | 追踪、调试、评估 | https://smith.langchain.com/ |
| **Langfuse** | 开源可观测性 | https://langfuse.com/ |
| **Playwright** | Web Agent 调试 | https://playwright.dev/ |
| **Rich** | 终端输出美化 | https://rich.readthedocs.io/ |
| **tiktoken** | Token 计数 | https://github.com/openai/tiktoken |
| **Prometheus** | 指标监控 | https://prometheus.io/ |
| **Grafana** | 指标可视化 | https://grafana.com/ |

---

## C.48 调试检查清单

### C.48.1 启动前检查

- [ ] 环境变量已配置
- [ ] API Key 有效
- [ ] 依赖包已安装
- [ ] 配置文件正确

### C.48.2 运行时检查

- [ ] 最大迭代次数已设置
- [ ] 超时时间已设置
- [ ] 错误处理已实现
- [ ] 日志已启用

### C.48.3 性能检查

- [ ] 延迟在可接受范围内
- [ ] Token 消耗在预算内
- [ ] 错误率低于阈值
- [ ] 资源使用正常

### C.48.4 安全检查

- [ ] 输入验证已实现
- [ ] 工具权限已配置
- [ ] 输出过滤已启用
- [ ] 审计日志已记录

---

## C.49 常见场景深度调试指南

### C.49.1 场景一：Agent 无法理解用户意图

**症状**：Agent 的回答与用户问题无关，或者回答过于笼统。

**调试步骤**：

1. **检查 System Prompt**：确保 Prompt 清晰定义了 Agent 的角色、能力和约束。模糊的 Prompt 会导致 Agent 行为不可预测。

2. **检查上下文**：确保 Agent 能够看到足够的上下文信息。如果上下文窗口被截断，重要信息可能丢失。

3. **尝试 Few-shot 示例**：在 Prompt 中提供 2-3 个示例，帮助 LLM 理解预期的行为模式。

4. **检查 temperature**：过高的 temperature 会导致 LLM 输出不稳定。对于工具调用场景，建议使用 temperature=0。

5. **使用不同的 LLM**：某些 LLM 可能对特定类型的任务表现更好。尝试使用 GPT-4o 或 Claude Sonnet 4。

### C.49.2 场景二：工具调用参数错误

**症状**：LLM 生成的工具调用参数不符合 JSON Schema，导致工具执行失败。

**调试步骤**：

1. **检查工具的 parameters 定义**：确保 Schema 准确描述了每个参数的类型、含义和约束。

2. **在工具描述中添加示例**：示例可以帮助 LLM 理解如何构造参数。

3. **使用 StructuredTool**：使用 Pydantic 模型定义参数，提供更严格的类型检查。

4. **添加参数校验**：在工具执行前校验参数，提供清晰的错误信息。

### C.49.3 场景三：RAG 检索质量差

**症状**：检索到的文档与查询不相关，导致生成的回答不准确。

**调试步骤**：

1. **检查分块策略**：调整 chunk_size 和 chunk_overlap。小块更精确，大块保留更多上下文。

2. **尝试不同的 Embedding 模型**：不同的 Embedding 模型对不同类型的文本效果不同。

3. **添加重排序**：使用 Cohere Rerank 或 Cross-Encoder 对检索结果进行精排。

4. **使用 Multi-Query**：将用户查询扩展为多个不同角度的查询，提升召回率。

5. **检查文档质量**：确保文档内容准确、完整、格式规范。

### C.49.4 场景四：多 Agent 协作失败

**症状**：Agent 之间无法正确协作，任务无法完成。

**调试步骤**：

1. **检查 Agent 的系统提示**：确保每个 Agent 的角色和职责明确定义。

2. **检查消息格式**：确保 Agent 之间的消息格式一致。

3. **检查是否有死锁**：多个 Agent 可能互相等待对方完成。

4. **添加超时和重试机制**：防止单个 Agent 卡住导致整个系统停滞。

5. **使用 LangSmith 追踪**：查看每个 Agent 的执行过程，定位协作问题。

### C.49.5 场景五：Agent 输出格式不正确

**症状**：Agent 的输出不符合预期格式。

**调试步骤**：

1. **在系统提示中明确输出格式**：使用具体的格式示例。

2. **使用 Structured Output**：使用 Pydantic 模型或 JSON Schema 约束输出。

3. **添加输出解析逻辑**：在 Agent 外部添加格式检查和修正。

4. **在工具描述中说明返回格式**：确保工具返回的数据格式一致。

---

## C.50 调试工具使用指南

### C.50.1 LangSmith 使用指南

```python
# 1. 配置 LangSmith
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_..."
os.environ["LANGCHAIN_PROJECT"] = "my-agent"

# 2. 运行 Agent（自动追踪）
agent = create_react_agent(model=llm, tools=tools)
result = agent.invoke({"messages": [("user", "你好")]})

# 3. 在 LangSmith UI 中查看
# - 执行链路：查看每一步的输入输出
# - Token 消耗：分析成本分布
# - 延迟分析：识别性能瓶颈
# - 错误追踪：定位错误原因
```

### C.50.2 Langfuse 使用指南

```python
from langfuse import Langfuse

langfuse = Langfuse()

# 创建追踪
trace = langfuse.trace(name="agent-execution")

# 记录 Span
span = trace.span(name="llm-call")
span.input = {"messages": messages}
span.output = {"response": response}
span.end()

# 记录事件
trace.event(name="tool-call", input={"tool": "search"}, output={"result": "..."})
```

### C.50.3 Rich 终端输出

```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# 创建表格
table = Table(title="Agent 执行过程")
table.add_column("步骤", style="cyan")
table.add_column("操作", style="green")
table.add_column("结果", style="yellow")

for i, step in enumerate(execution_steps):
    table.add_row(str(i+1), step["action"], step["result"][:50])

console.print(table)

# 创建面板
console.print(Panel("Agent 执行完成", title="结果", border_style="green"))
```

---

## C.51 调试最佳实践

### C.51.1 日志最佳实践

```python
# 使用结构化日志
import logging
import json

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def log_agent_step(self, step, input_data, output_data, duration):
        self.logger.info(json.dumps({
            "event": "agent_step",
            "step": step,
            "input": str(input_data)[:200],
            "output": str(output_data)[:200],
            "duration_ms": duration * 1000,
            "timestamp": datetime.now().isoformat()
        }))
    
    def log_tool_call(self, tool_name, args, result, duration):
        self.logger.info(json.dumps({
            "event": "tool_call",
            "tool": tool_name,
            "args": args,
            "result": str(result)[:200],
            "duration_ms": duration * 1000,
            "timestamp": datetime.now().isoformat()
        }))
    
    def log_error(self, error_type, error_message, context):
        self.logger.error(json.dumps({
            "event": "error",
            "type": error_type,
            "message": error_message,
            "context": str(context)[:200],
            "timestamp": datetime.now().isoformat()
        }))
```

### C.51.2 追踪最佳实践

```python
# 使用 LangSmith 追踪
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_..."
os.environ["LANGCHAIN_PROJECT"] = "my-agent"

# 运行 Agent（自动追踪）
agent = create_react_agent(model=llm, tools=tools)
result = agent.invoke({"messages": [("user", "你好")]})

# 在 LangSmith UI 中查看：
# - 执行链路：每一步的输入输出
# - Token 消耗：成本分析
# - 延迟分析：性能瓶颈
# - 错误追踪：错误原因
```

---

## C.52 常见错误代码对照表

| 错误代码 | 含义 | 解决方案 |
|:---|:---|:---|
| `context_length_exceeded` | Token 超限 | 压缩历史、截断输出 |
| `invalid_api_key` | API Key 无效 | 检查 API Key 配置 |
| `rate_limit_exceeded` | 请求频率过高 | 降低请求频率、使用缓存 |
| `model_not_found` | 模型不存在 | 检查模型名称 |
| `tool_not_found` | 工具不存在 | 检查工具注册 |
| `json_parse_error` | JSON 解析失败 | 检查参数格式 |
| `timeout` | 执行超时 | 增加超时时间 |
| `connection_error` | 连接失败 | 检查网络连接 |
| `permission_denied` | 权限不足 | 检查 API Key 权限 |
| `quota_exceeded` | 配额用尽 | 升级计划或等待重置 |

---

## C.53 调试检查清单

### C.53.1 启动前检查

- [ ] 环境变量已配置（OPENAI_API_KEY、LANGCHAIN_API_KEY 等）
- [ ] API Key 有效且未过期
- [ ] 依赖包已安装（pip install langchain langgraph）
- [ ] 配置文件正确（.env 文件格式正确）
- [ ] 网络连接正常

### C.53.2 运行时检查

- [ ] 最大迭代次数已设置（防止无限循环）
- [ ] 超时时间已设置（防止长时间阻塞）
- [ ] 错误处理已实现（try-except）
- [ ] 日志已启用（logging 或 print）
- [ ] 状态管理正确（TypedDict 字段匹配）

### C.53.3 性能检查

- [ ] 延迟在可接受范围内（P95 < 30秒）
- [ ] Token 消耗在预算内（每次 < 10000）
- [ ] 错误率低于阈值（< 5%）
- [ ] 资源使用正常（CPU < 80%, 内存 < 80%）

### C.53.4 安全检查

- [ ] 输入验证已实现（检测提示注入）
- [ ] 工具权限已配置（危险工具受限）
- [ ] 输出过滤已启用（防止敏感信息泄露）
- [ ] 审计日志已记录（记录所有操作）

---

## C.54 性能优化清单

| 优化项 | 预期效果 | 实现难度 | 优先级 |
|:---|:---|:---|:---|
| 模型路由 | 降低 40-60% 成本 | 中 | 高 |
| 语义缓存 | 降低 30-50% 成本 | 中 | 高 |
| Prompt 压缩 | 降低 20-30% Token | 低 | 中 |
| 并行工具调用 | 降低 50% 延迟 | 中 | 高 |
| 流式输出 | 降低感知延迟 | 低 | 中 |
| 异步执行 | 降低阻塞时间 | 中 | 中 |
| 本地模型 | 降低 API 成本 | 高 | 低 |

---

## C.55 调试工作流

### C.55.1 标准调试流程

1. **复现问题**：确保问题可以稳定复现
2. **收集信息**：记录输入、输出、错误信息、执行日志
3. **定位问题**：使用 LangSmith 追踪执行链路
4. **分析原因**：检查 Prompt、工具描述、状态管理
5. **实施修复**：修改代码或配置
6. **验证修复**：重新运行测试
7. **记录经验**：更新文档和知识库

### C.55.2 调试工具使用

```python
# 1. 启用调试模式
import langchain
langchain.debug = True

# 2. 使用 LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# 3. 打印中间状态
for chunk in agent.stream(input, stream_mode="updates"):
    for node, update in chunk.items():
        print(f"[{node}] {update}")

# 4. 使用 Rich 美化输出
from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="Agent 执行过程")
table.add_column("步骤", style="cyan")
table.add_column("操作", style="green")
table.add_column("结果", style="yellow")
```

---

## C.56 常见错误代码对照表

| 错误代码 | 含义 | 解决方案 |
|:---|:---|:---|
| `context_length_exceeded` | Token 超限 | 压缩历史、截断输出 |
| `invalid_api_key` | API Key 无效 | 检查 API Key 配置 |
| `rate_limit_exceeded` | 请求频率过高 | 降低请求频率、使用缓存 |
| `model_not_found` | 模型不存在 | 检查模型名称 |
| `tool_not_found` | 工具不存在 | 检查工具注册 |
| `json_parse_error` | JSON 解析失败 | 检查参数格式 |
| `timeout` | 执行超时 | 增加超时时间 |
| `connection_error` | 连接失败 | 检查网络连接 |
| `permission_denied` | 权限不足 | 检查 API Key 权限 |
| `quota_exceeded` | 配额用尽 | 升级计划或等待重置 |

---

## C.57 调试检查清单

### C.57.1 启动前检查

- [ ] 环境变量已配置
- [ ] API Key 有效
- [ ] 依赖包已安装
- [ ] 配置文件正确
- [ ] 网络连接正常

### C.57.2 运行时检查

- [ ] 最大迭代次数已设置
- [ ] 超时时间已设置
- [ ] 错误处理已实现
- [ ] 日志已启用
- [ ] 状态管理正确

### C.57.3 性能检查

- [ ] 延迟在可接受范围内
- [ ] Token 消耗在预算内
- [ ] 错误率低于阈值
- [ ] 资源使用正常

### C.57.4 安全检查

- [ ] 输入验证已实现
- [ ] 工具权限已配置
- [ ] 输出过滤已启用
- [ ] 审计日志已记录

---

## C.58 最佳实践总结

1. **始终设置最大迭代次数**：防止无限循环
2. **使用详细日志**：记录每一步的推理和行动
3. **集成可观测性工具**：LangSmith 或 Langfuse
4. **优化工具描述**：清晰的描述提升调用准确率
5. **处理 Token 超限**：压缩历史、截断输出
6. **检测幻觉**：基于事实回答，不编造信息
7. **性能监控**：监控延迟、Token 消耗、错误率
8. **测试覆盖**：单元测试、集成测试、端到端测试
9. **安全防护**：输入验证、工具权限、输出过滤
10. **文档维护**：保持文档与代码同步更新
