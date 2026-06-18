---
title: "第14章：LangChain Agent 深度实战"
description: "深入 LangChain/LangGraph 的 Agent 实现：StateGraph、create_react_agent、自定义 Agent、工具集成、错误恢复与时间旅行调试。"
date: "2026-06-11"
---

 # 第14章：LangChain Agent 深度实战

ReAct 框架是 Agent 推理与行动交替的经典范式。下面的交互式演示展示了完整的 ReAct 工作流程：

<div data-component="ReActDemoV8"></div>

工具选择是 Agent 决策中的关键环节。下面的交互式演示展示了完整的工具选择决策过程：

<div data-component="ToolSelectionDemoV10"></div>

不同的 Agent 实现方式各有优劣。下面的交互式对比可以帮助你选择合适的方案：

<div data-component="AgentArchitectureComparisonV10"></div>

 ---

 ## 14.1 LangGraph 基础

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI

def agent_node(state: MessagesState):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return {"messages": [llm.invoke(state["messages"])]}

def tool_node(state: MessagesState):
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        result = execute_tool(tc["name"], tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}

def should_continue(state):
    last = state["messages"][-1]
    return "tools" if hasattr(last, "tool_calls") and last.tool_calls else END

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
app = graph.compile()
```

---

## 14.2 create_react_agent

```python
from langgraph.prebuilt import create_react_agent

@tool
def search(query: str) -> str:
    """搜索互联网"""
    return f"搜索 '{query}' 的结果..."

@tool
def calculator(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=[search, calculator],
)
result = agent.invoke({"messages": [("user", "2+3等于多少？")]})
```

---

## 14.3 流式执行

```python
for chunk in agent.stream({"messages": [("user", "搜索 AI Agent")]}, stream_mode="updates"):
    for node, update in chunk.items():
        print(f"\n[{node}]")
        if "messages" in update:
            for msg in update["messages"]: msg.pretty_print()
```

---

## 14.4 Human-in-the-Loop

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory, interrupt_before=["tools"])

config = {"configurable": {"thread_id": "user_1"}}
result = app.invoke({"messages": [("user", "删除日志文件")]}, config=config)
user_approved = True
if user_approved: result = app.invoke(None, config=config)
```

---

## 14.5 时间旅行调试

```python
history = list(app.get_state_history(config))
past_state = history[2]
app.update_state(config, past_state.values)
result = app.invoke(None, config=config)
```

---

## 14.6 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| StateGraph | 用图编排 Agent 行为 |
| create_react_agent | 最简单的 ReAct Agent |
| 流式输出 | 实时查看推理过程 |
| HITL | 人工审批机制 |
| 时间旅行 | 检查点回溯调试 |

> **下一章预告**
>
> 在第 15 章中，我们将深入 LangGraph 有状态 Agent。

---

## 14.7 LangGraph StateGraph 深度解析

### 14.7.1 StateGraph 核心概念

StateGraph 是 LangGraph 的核心组件，用于构建有状态的 Agent 工作流。

```python
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class AgentState(TypedDict):
    """自定义Agent状态"""
    messages: List[Dict[str, Any]]
    current_step: str
    context: Dict[str, Any]
    iteration: int
    max_iterations: int

def create_advanced_agent_graph():
    """创建高级Agent图"""
    # 定义状态图
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("initialize", initialize_node)
    graph.add_node("process", process_node)
    graph.add_node("decide", decide_node)
    graph.add_node("execute", execute_node)
    graph.add_node("finalize", finalize_node)

    # 添加边
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "process")
    graph.add_conditional_edges("process", should_continue, {
        "continue": "decide",
        "end": "finalize"
    })
    graph.add_conditional_edges("decide", route_decision, {
        "execute": "execute",
        "skip": "process"
    })
    graph.add_edge("execute", "process")
    graph.add_edge("finalize", END)

    return graph.compile()

def initialize_node(state: AgentState):
    """初始化节点"""
    return {
        "messages": state["messages"],
        "current_step": "initialized",
        "context": {"start_time": time.time()},
        "iteration": 0,
        "max_iterations": state.get("max_iterations", 10)
    }

def process_node(state: AgentState):
    """处理节点"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 构建提示
    messages = state["messages"]
    system_message = SystemMessage(content="你是一个智能助手，请根据对话历史处理用户请求。")

    response = llm.invoke([system_message] + messages)

    return {
        "messages": messages + [AIMessage(content=response.content)],
        "current_step": "processed",
        "iteration": state["iteration"] + 1
    }

def should_continue(state: AgentState) -> str:
    """判断是否继续"""
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    if state["current_step"] == "processed":
        return "continue"
    return "end"

def decide_node(state: AgentState):
    """决策节点"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 分析当前状态
    prompt = f"""
    分析当前对话状态，决定下一步行动：
    对话历史：{state['messages'][-3:]}
    当前迭代：{state['iteration']}

    请决定：execute（执行操作）或 skip（继续处理）
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().lower()

    return {"current_step": decision}

def route_decision(state: AgentState) -> str:
    """路由决策"""
    if state["current_step"] == "execute":
        return "execute"
    return "skip"

def execute_node(state: AgentState):
    """执行节点"""
    # 这里可以执行实际操作
    return {
        "current_step": "executed",
        "context": {**state["context"], "executed": True}
    }

def finalize_node(state: AgentState):
    """完成节点"""
    return {
        "current_step": "completed",
        "context": {**state["context"], "end_time": time.time()}
    }
```

### 14.7.2 高级状态管理

```python
class StateManager:
    """状态管理器"""
    def __init__(self):
        self.state_history = []
        self.state_checkpoints = {}

    def save_state(self, state: AgentState, checkpoint_name: str):
        """保存状态检查点"""
        self.state_checkpoints[checkpoint_name] = {
            "state": state.copy(),
            "timestamp": time.time()
        }

    def load_state(self, checkpoint_name: str) -> AgentState:
        """加载状态检查点"""
        if checkpoint_name in self.state_checkpoints:
            return self.state_checkpoints[checkpoint_name]["state"]
        return None

    def get_state_history(self) -> List[Dict]:
        """获取状态历史"""
        return self.state_history

    def analyze_state_transitions(self) -> Dict:
        """分析状态转换"""
        transitions = {}
        for i in range(1, len(self.state_history)):
            prev_state = self.state_history[i-1]["state"]
            curr_state = self.state_history[i]["state"]

            transition = f"{prev_state['current_step']} -> {curr_state['current_step']}"
            transitions[transition] = transitions.get(transition, 0) + 1

        return transitions

# 使用示例
state_manager = StateManager()

# 保存状态
state_manager.save_state(initial_state, "checkpoint_1")

# 加载状态
loaded_state = state_manager.load_state("checkpoint_1")
```

### 14.7.3 条件分支与循环

```python
class ConditionalAgent:
    """条件分支Agent"""
    def __init__(self, llm):
        self.llm = llm
        self.conditions = {}

    def add_condition(self, condition_name: str, condition_func: Callable):
        """添加条件"""
        self.conditions[condition_name] = condition_func

    def create_conditional_graph(self):
        """创建条件图"""
        graph = StateGraph(AgentState)

        # 添加条件节点
        graph.add_node("check_condition", self._check_condition_node)
        graph.add_node("branch_a", self._branch_a_node)
        graph.add_node("branch_b", self._branch_b_node)
        graph.add_node("merge", self._merge_node)

        # 添加条件边
        graph.add_conditional_edges(
            "check_condition",
            self._route_based_on_condition,
            {
                "branch_a": "branch_a",
                "branch_b": "branch_b"
            }
        )

        graph.add_edge("branch_a", "merge")
        graph.add_edge("branch_b", "merge")
        graph.add_edge(START, "check_condition")
        graph.add_edge("merge", END)

        return graph.compile()

    def _check_condition_node(self, state: AgentState):
        """检查条件节点"""
        # 检查所有条件
        results = {}
        for name, condition_func in self.conditions.items():
            results[name] = condition_func(state)

        return {"context": {**state["context"], "condition_results": results}}

    def _route_based_on_condition(self, state: AgentState):
        """基于条件路由"""
        condition_results = state["context"].get("condition_results", {})

        if condition_results.get("condition_a", False):
            return "branch_a"
        return "branch_b"
```

---

## 14.8 create_react_agent 深度实战

### 14.8.1 高级 ReAct Agent

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 定义高级工具
@tool
def advanced_search(query: str, search_type: str = "web") -> str:
    """高级搜索工具"""
    if search_type == "web":
        return f"网页搜索 '{query}' 的结果..."
    elif search_type == "academic":
        return f"学术搜索 '{query}' 的结果..."
    else:
        return f"搜索 '{query}' 的结果..."

@tool
def data_analysis(data: str, analysis_type: str = "summary") -> str:
    """数据分析工具"""
    if analysis_type == "summary":
        return f"数据摘要：{data[:100]}..."
    elif analysis_type == "statistics":
        return f"统计分析：{data[:100]}..."
    else:
        return f"分析结果：{data[:100]}..."

@tool
def code_execution(code: str, language: str = "python") -> str:
    """代码执行工具"""
    # 注意：实际使用时需要安全沙箱
    return f"执行 {language} 代码：{code[:50]}..."

# 创建高级Agent
def create_advanced_react_agent():
    """创建高级ReAct Agent"""
    tools = [advanced_search, data_analysis, code_execution]

    agent = create_react_agent(
        model=ChatOpenAI(model="gpt-4o", temperature=0),
        tools=tools,
        state_modifier="你是一个高级AI助手，可以使用多种工具完成任务。"
    )

    return agent

# 使用示例
agent = create_advanced_react_agent()
result = agent.invoke({
    "messages": [("user", "搜索关于AI Agent的最新研究，并分析主要趋势")]
})
```

### 14.8.2 自定义 ReAct 循环

```python
class CustomReActAgent:
    """自定义ReAct Agent"""
    def __init__(self, llm, tools, max_iterations=10):
        self.llm = llm.bind_tools(tools)
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations

    def run(self, task: str) -> Dict[str, Any]:
        """执行任务"""
        messages = [
            SystemMessage(content="你是一个智能助手，使用ReAct模式思考和行动。"),
            HumanMessage(content=task)
        ]

        for iteration in range(self.max_iterations):
            # 思考
            response = self.llm.invoke(messages)
            messages.append(response)

            # 检查是否有工具调用
            if not response.tool_calls:
                return {
                    "result": response.content,
                    "iterations": iteration + 1,
                    "messages": messages
                }

            # 行动
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name in self.tools:
                    # 执行工具
                    result = self.tools[tool_name].invoke(tool_args)

                    # 观察结果
                    observation = f"工具 {tool_name} 执行结果：{result}"
                    messages.append(ToolMessage(
                        content=observation,
                        tool_call_id=tool_call["id"]
                    ))

        return {
            "result": "达到最大迭代次数",
            "iterations": self.max_iterations,
            "messages": messages
        }

# 使用示例
llm = ChatOpenAI(model="gpt-4o")
tools = [advanced_search, data_analysis]
agent = CustomReActAgent(llm, tools)
result = agent.run("分析AI Agent的发展趋势")
```

### 14.8.3 多工具协调

```python
class MultiToolCoordinator:
    """多工具协调器"""
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.tool_chains = {}

    def coordinate_execution(self, task: str) -> Dict[str, Any]:
        """协调执行"""
        # 分析任务，确定需要的工具
        tool_plan = self._plan_tool_usage(task)

        # 按计划执行工具
        results = {}
        for step in tool_plan:
            tool_name = step["tool"]
            tool_args = step["args"]

            if tool_name in self.tools:
                result = self.tools[tool_name].invoke(tool_args)
                results[step["step_id"]] = result

        # 整合结果
        final_result = self._integrate_results(results, task)

        return {
            "task": task,
            "tool_plan": tool_plan,
            "intermediate_results": results,
            "final_result": final_result
        }

    def _plan_tool_usage(self, task: str) -> List[Dict]:
        """规划工具使用"""
        prompt = f"""
        为以下任务规划工具使用步骤：
        任务：{task}
        可用工具：{list(self.tools.keys())}

        请输出工具使用计划（JSON格式）：
        [
            {{"step_id": 1, "tool": "工具名", "args": {{"参数": "值"}}}},
            ...
        ]
        """

        response = self.llm.invoke([HumanMessage(content=prompt)])

        # 解析JSON计划
        try:
            import json
            plan = json.loads(response.content)
            return plan
        except:
            return []

    def _integrate_results(self, results: Dict, task: str) -> str:
        """整合结果"""
        prompt = f"""
        整合以下工具执行结果，回答原始任务：
        任务：{task}
        工具结果：{results}

        请给出最终答案："""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
```

---

## 14.9 自定义 Agent 提示词

### 14.9.1 提示词工程

```python
class PromptEngineer:
    """提示词工程师"""
    def __init__(self, llm):
        self.llm = llm
        self.prompt_templates = {}

    def create_system_prompt(self, role: str, capabilities: List[str], constraints: List[str]) -> str:
        """创建系统提示词"""
        capabilities_str = "\n".join(f"- {cap}" for cap in capabilities)
        constraints_str = "\n".join(f"- {constraint}" for constraint in constraints)

        prompt = f"""
你是一个{role}，具有以下能力：
{capabilities_str}

你需要遵守以下约束：
{constraints_str}

请根据用户需求提供帮助。
"""
        return prompt

    def create_few_shot_prompt(self, examples: List[Dict], task: str) -> str:
        """创建少样本提示词"""
        examples_str = ""
        for i, example in enumerate(examples, 1):
            examples_str += f"""
示例{i}：
输入：{example['input']}
输出：{example['output']}
"""

        prompt = f"""
参考以下示例：

{examples_str}

现在请处理以下任务：
输入：{task}
输出："""
        return prompt

    def create_chain_of_thought_prompt(self, task: str) -> str:
        """创建思维链提示词"""
        prompt = f"""
请按照以下步骤思考并解决问题：

1. 理解问题：{task}
2. 分析问题：识别关键信息和约束条件
3. 制定计划：确定解决步骤
4. 执行计划：逐步实施
5. 验证结果：检查答案是否合理

请详细展示你的思考过程：
"""
        return prompt

    def optimize_prompt(self, original_prompt: str, feedback: str) -> str:
        """优化提示词"""
        prompt = f"""
原始提示词：{original_prompt}
反馈：{feedback}

请优化提示词以提高效果："""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
```

### 14.9.2 动态提示词生成

```python
class DynamicPromptGenerator:
    """动态提示词生成器"""
    def __init__(self, llm):
        self.llm = llm
        self.context_history = []

    def generate_contextual_prompt(self, task: str, context: Dict) -> str:
        """生成上下文提示词"""
        # 分析上下文
        context_analysis = self._analyze_context(context)

        # 生成提示词
        prompt = f"""
基于以下上下文信息生成提示词：

任务：{task}
上下文分析：{context_analysis}
历史交互：{self.context_history[-3:] if self.context_history else '无'}

生成的提示词："""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _analyze_context(self, context: Dict) -> str:
        """分析上下文"""
        analysis = []
        for key, value in context.items():
            if isinstance(value, str):
                analysis.append(f"{key}: {value[:100]}...")
            else:
                analysis.append(f"{key}: {type(value)}")
        return "\n".join(analysis)

    def adapt_prompt_based_on_feedback(self, prompt: str, feedback: str) -> str:
        """基于反馈调整提示词"""
        adaptation_prompt = f"""
当前提示词：{prompt}
用户反馈：{feedback}

请调整提示词以更好地满足用户需求："""

        response = self.llm.invoke([HumanMessage(content=adaptation_prompt)])
        return response.content
```

---

## 14.10 流式执行深入

### 14.10.1 高级流式处理

```python
class AdvancedStreamingAgent:
    """高级流式Agent"""
    def __init__(self, agent):
        self.agent = agent
        self.stream_handlers = {}

    def stream_with_handlers(self, task: str, handlers: Dict[str, Callable]):
        """带处理器的流式执行"""
        self.stream_handlers = handlers

        for chunk in self.agent.stream(
            {"messages": [("user", task)]},
            stream_mode="updates"
        ):
            for node, update in chunk.items():
                # 调用处理器
                if node in self.stream_handlers:
                    self.stream_handlers[node](update)
                else:
                    # 默认处理
                    self._default_handler(node, update)

    def _default_handler(self, node: str, update: Dict):
        """默认处理器"""
        print(f"\n[{node}]")
        if "messages" in update:
            for msg in update["messages"]:
                if hasattr(msg, "pretty_print"):
                    msg.pretty_print()
                else:
                    print(msg)

    def stream_with_filter(self, task: str, node_filter: List[str]):
        """带过滤的流式执行"""
        for chunk in self.agent.stream(
            {"messages": [("user", task)]},
            stream_mode="updates"
        ):
            for node, update in chunk.items():
                if node in node_filter:
                    yield {"node": node, "update": update}

    def stream_with_aggregation(self, task: str):
        """带聚合的流式执行"""
        aggregated_results = {}

        for chunk in self.agent.stream(
            {"messages": [("user", task)]},
            stream_mode="updates"
        ):
            for node, update in chunk.items():
                if node not in aggregated_results:
                    aggregated_results[node] = []

                aggregated_results[node].append(update)

        return aggregated_results
```

### 14.10.2 流式输出可视化

```python
class StreamingVisualizer:
    """流式输出可视化"""
    def __init__(self):
        self.visual_elements = []

    def visualize_stream(self, stream_generator):
        """可视化流式输出"""
        import matplotlib.pyplot as plt
        from IPython.display import display, clear_output

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        plt.ion()

        for update in stream_generator:
            # 清除之前的图形
            clear_output(wait=True)

            # 更新可视化
            self._update_visualization(axes, update)

            plt.pause(0.1)

        plt.ioff()
        plt.show()

    def _update_visualization(self, axes, update):
        """更新可视化"""
        node = update["node"]
        data = update["update"]

        # 这里可以根据节点类型绘制不同的图表
        if "messages" in data:
            ax = axes[0, 0]
            ax.clear()
            ax.set_title("消息历史")
            # 绘制消息历史
```

---

## 14.11 Human-in-the-Loop 深入

### 14.11.1 高级 HITL 实现

```python
class AdvancedHITL:
    """高级Human-in-the-Loop"""
    def __init__(self, agent, memory):
        self.agent = agent
        self.memory = memory
        self.approval_callbacks = {}

    def execute_with_approval(self, task: str, approval_steps: List[str]):
        """需要审批的执行"""
        config = {"configurable": {"thread_id": f"approval_{int(time.time())}"}}

        # 执行直到需要审批
        result = self.agent.invoke(
            {"messages": [("user", task)]},
            config=config
        )

        # 检查是否需要审批
        if self._needs_approval(result, approval_steps):
            # 等待用户审批
            approval = self._wait_for_approval(result)

            if approval["approved"]:
                # 继续执行
                result = self.agent.invoke(None, config=config)
            else:
                # 取消执行
                result = {"status": "cancelled", "reason": approval.get("reason")}

        return result

    def _needs_approval(self, result: Dict, approval_steps: List[str]) -> bool:
        """检查是否需要审批"""
        # 分析结果，判断是否需要审批
        for step in approval_steps:
            if step in str(result):
                return True
        return False

    def _wait_for_approval(self, result: Dict) -> Dict:
        """等待审批"""
        # 这里可以实现实际的审批流程
        # 例如：发送邮件、Slack通知等

        print("需要用户审批：")
        print(f"操作：{result}")
        approval = input("是否批准？(yes/no): ")

        return {
            "approved": approval.lower() == "yes",
            "approver": "user",
            "timestamp": time.time()
        }

    def register_approval_callback(self, step: str, callback: Callable):
        """注册审批回调"""
        self.approval_callbacks[step] = callback
```

### 14.11.2 审批工作流

```python
class ApprovalWorkflow:
    """审批工作流"""
    def __init__(self):
        self.approval_chains = {}
        self.approval_history = []

    def create_approval_chain(self, chain_name: str, approvers: List[str]):
        """创建审批链"""
        self.approval_chains[chain_name] = {
            "approvers": approvers,
            "current_index": 0,
            "status": "pending"
        }

    def request_approval(self, chain_name: str, request_data: Dict) -> Dict:
        """请求审批"""
        if chain_name not in self.approval_chains:
            return {"error": "审批链不存在"}

        chain = self.approval_chains[chain_name]
        current_approver = chain["approvers"][chain["current_index"]]

        # 发送审批请求
        approval_request = {
            "chain_name": chain_name,
            "request_data": request_data,
            "approver": current_approver,
            "timestamp": time.time()
        }

        self.approval_history.append(approval_request)

        return {
            "status": "pending",
            "approver": current_approver,
            "request_id": len(self.approval_history) - 1
        }

    def submit_approval(self, request_id: int, approved: bool, comments: str = ""):
        """提交审批"""
        if request_id >= len(self.approval_history):
            return {"error": "无效的请求ID"}

        request = self.approval_history[request_id]
        chain_name = request["chain_name"]
        chain = self.approval_chains[chain_name]

        if approved:
            # 移动到下一个审批者
            chain["current_index"] += 1

            if chain["current_index"] >= len(chain["approvers"]):
                # 所有审批者都已批准
                chain["status"] = "approved"
                return {"status": "approved", "message": "所有审批者已批准"}
            else:
                return {"status": "pending", "message": "已批准，等待下一个审批者"}
        else:
            # 拒绝
            chain["status"] = "rejected"
            return {"status": "rejected", "message": f"被 {request['approver']} 拒绝"}
```

---

## 14.12 时间旅行调试深入

### 14.12.1 高级时间旅行调试

```python
class AdvancedTimeTravelDebugger:
    """高级时间旅行调试器"""
    def __init__(self, agent, memory):
        self.agent = agent
        self.memory = memory
        self.debug_sessions = {}

    def start_debug_session(self, session_id: str, initial_state: Dict):
        """开始调试会话"""
        self.debug_sessions[session_id] = {
            "initial_state": initial_state,
            "checkpoints": [],
            "current_index": 0,
            "breakpoints": []
        }

    def add_checkpoint(self, session_id: str, state: Dict, description: str = ""):
        """添加检查点"""
        if session_id not in self.debug_sessions:
            return {"error": "调试会话不存在"}

        checkpoint = {
            "index": len(self.debug_sessions[session_id]["checkpoints"]),
            "state": state,
            "description": description,
            "timestamp": time.time()
        }

        self.debug_sessions[session_id]["checkpoints"].append(checkpoint)

        return {"checkpoint_id": checkpoint["index"]}

    def time_travel(self, session_id: str, checkpoint_id: int):
        """时间旅行到指定检查点"""
        if session_id not in self.debug_sessions:
            return {"error": "调试会话不存在"}

        session = self.debug_sessions[session_id]
        if checkpoint_id >= len(session["checkpoints"]):
            return {"error": "无效的检查点ID"}

        # 获取检查点状态
        checkpoint = session["checkpoints"][checkpoint_id]
        session["current_index"] = checkpoint_id

        # 恢复状态
        return {
            "status": "success",
            "checkpoint": checkpoint,
            "state": checkpoint["state"]
        }

    def set_breakpoint(self, session_id: str, condition: Callable):
        """设置断点"""
        if session_id not in self.debug_sessions:
            return {"error": "调试会话不存在"}

        breakpoint_id = len(self.debug_sessions[session_id]["breakpoints"])
        self.debug_sessions[session_id]["breakpoints"].append({
            "id": breakpoint_id,
            "condition": condition,
            "hit_count": 0
        })

        return {"breakpoint_id": breakpoint_id}

    def debug_execution(self, session_id: str, task: str):
        """调试执行"""
        if session_id not in self.debug_sessions:
            return {"error": "调试会话不存在"}

        session = self.debug_sessions[session_id]
        initial_state = session["initial_state"]

        # 执行任务，但检查断点
        config = {"configurable": {"thread_id": session_id}}

        for chunk in self.agent.stream(
            {"messages": [("user", task)]},
            config=config,
            stream_mode="updates"
        ):
            # 检查断点
            for breakpoint in session["breakpoints"]:
                if breakpoint["condition"](chunk):
                    breakpoint["hit_count"] += 1
                    # 添加检查点
                    self.add_checkpoint(session_id, chunk, f"断点命中: {breakpoint['id']}")

            yield chunk
```

### 14.12.2 调试会话管理

```python
class DebugSessionManager:
    """调试会话管理器"""
    def __init__(self):
        self.sessions = {}
        self.session_history = []

    def create_session(self, session_name: str, agent_config: Dict):
        """创建调试会话"""
        session_id = f"session_{int(time.time())}"

        self.sessions[session_id] = {
            "name": session_name,
            "config": agent_config,
            "created_at": time.time(),
            "status": "active",
            "checkpoints": [],
            "logs": []
        }

        self.session_history.append(session_id)

        return {"session_id": session_id}

    def log_event(self, session_id: str, event_type: str, data: Dict):
        """记录事件"""
        if session_id not in self.sessions:
            return {"error": "会话不存在"}

        log_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "data": data
        }

        self.sessions[session_id]["logs"].append(log_entry)

    def get_session_summary(self, session_id: str) -> Dict:
        """获取会话摘要"""
        if session_id not in self.sessions:
            return {"error": "会话不存在"}

        session = self.sessions[session_id]

        return {
            "name": session["name"],
            "status": session["status"],
            "duration": time.time() - session["created_at"],
            "checkpoint_count": len(session["checkpoints"]),
            "log_count": len(session["logs"]),
            "recent_logs": session["logs"][-5:] if session["logs"] else []
        }

    def export_session(self, session_id: str, format: str = "json"):
        """导出会话"""
        if session_id not in self.sessions:
            return {"error": "会话不存在"}

        session = self.sessions[session_id]

        if format == "json":
            import json
            return json.dumps(session, indent=2, default=str)
        elif format == "markdown":
            return self._export_as_markdown(session)
        else:
            return {"error": f"不支持的格式: {format}"}
```

---

## 14.13 错误处理与恢复

### 14.13.1 高级错误处理

```python
class AdvancedErrorHandling:
    """高级错误处理"""
    def __init__(self, agent):
        self.agent = agent
        self.error_handlers = {}
        self.retry_policies = {}
        self.fallback_strategies = {}

    def register_error_handler(self, error_type: str, handler: Callable):
        """注册错误处理器"""
        self.error_handlers[error_type] = handler

    def register_retry_policy(self, operation: str, max_retries: int, delay: float):
        """注册重试策略"""
        self.retry_policies[operation] = {
            "max_retries": max_retries,
            "delay": delay
        }

    def execute_with_error_handling(self, task: str, config: Dict = None):
        """带错误处理的执行"""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                result = self.agent.invoke(
                    {"messages": [("user", task)]},
                    config=config
                )
                return {"status": "success", "result": result}

            except Exception as e:
                error_type = type(e).__name__

                # 检查是否有自定义处理器
                if error_type in self.error_handlers:
                    return self.error_handlers[error_type](e, task, attempt)

                # 检查是否需要重试
                if attempt < max_retries - 1:
                    print(f"尝试 {attempt + 1} 失败，{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避

        return {"status": "failed", "error": "达到最大重试次数"}

    def create_fallback_agent(self, primary_agent, fallback_agent):
        """创建备用Agent"""
        def fallback_strategy(task: str):
            try:
                return primary_agent.invoke({"messages": [("user", task)]})
            except Exception:
                print("主Agent失败，使用备用Agent...")
                return fallback_agent.invoke({"messages": [("user", task)]})

        return fallback_strategy
```

### 14.13.2 状态恢复机制

```python
class StateRecovery:
    """状态恢复机制"""
    def __init__(self, memory):
        self.memory = memory
        self.recovery_points = []

    def create_recovery_point(self, state: Dict, description: str = ""):
        """创建恢复点"""
        recovery_point = {
            "id": len(self.recovery_points),
            "state": state.copy(),
            "description": description,
            "timestamp": time.time()
        }
        self.recovery_points.append(recovery_point)

        return {"recovery_point_id": recovery_point["id"]}

    def recover_to_point(self, recovery_point_id: int):
        """恢复到指定点"""
        if recovery_point_id >= len(self.recovery_points):
            return {"error": "无效的恢复点ID"}

        recovery_point = self.recovery_points[recovery_point_id]

        return {
            "status": "success",
            "recovered_state": recovery_point["state"],
            "description": recovery_point["description"]
        }

    def auto_recovery(self, error: Exception, current_state: Dict):
        """自动恢复"""
        # 寻找最近的恢复点
        if self.recovery_points:
            latest_recovery = self.recovery_points[-1]
            return self.recover_to_point(latest_recovery["id"])

        # 如果没有恢复点，尝试从检查点恢复
        return self._recover_from_checkpoint(current_state)

    def _recover_from_checkpoint(self, state: Dict):
        """从检查点恢复"""
        # 这里可以实现从持久化存储恢复
        return {
            "status": "partial",
            "message": "从最近的检查点恢复",
            "state": state
        }
```

---

## 14.14 性能优化

### 14.14.1 缓存策略

```python
class CachingStrategy:
    """缓存策略"""
    def __init__(self):
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}

    def cached_agent_call(self, agent, task: str, cache_key: str = None):
        """缓存的Agent调用"""
        if cache_key is None:
            cache_key = self._generate_cache_key(task)

        # 检查缓存
        if cache_key in self.cache:
            self.cache_stats["hits"] += 1
            return self.cache[cache_key]

        # 执行Agent调用
        self.cache_stats["misses"] += 1
        result = agent.invoke({"messages": [("user", task)]})

        # 存入缓存
        self.cache[cache_key] = result

        return result

    def _generate_cache_key(self, task: str) -> str:
        """生成缓存键"""
        import hashlib
        return hashlib.md5(task.encode()).hexdigest()

    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total if total > 0 else 0

        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }

    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0}
```

### 14.14.2 并发优化

```python
class ConcurrencyOptimizer:
    """并发优化器"""
    def __init__(self, agent):
        self.agent = agent
        self.task_queue = asyncio.Queue()
        self.results = {}

    async def process_concurrent_tasks(self, tasks: List[str]):
        """处理并发任务"""
        import asyncio

        # 创建任务
        async def process_task(task_id: int, task: str):
            result = await asyncio.to_thread(
                self.agent.invoke,
                {"messages": [("user", task)]}
            )
            self.results[task_id] = result

        # 并发执行
        await asyncio.gather(*[
            process_task(i, task) for i, task in enumerate(tasks)
        ])

        return self.results

    def parallel_stream(self, tasks: List[str]):
        """并行流式处理"""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._stream_task, task): i
                for i, task in enumerate(tasks)
            }

            for future in futures:
                task_id = futures[future]
                try:
                    result = future.result()
                    yield {"task_id": task_id, "result": result}
                except Exception as e:
                    yield {"task_id": task_id, "error": str(e)}

    def _stream_task(self, task: str):
        """流式处理单个任务"""
        results = []
        for chunk in self.agent.stream(
            {"messages": [("user", task)]},
            stream_mode="updates"
        ):
            results.append(chunk)
        return results
```

---

## 14.15 监控与日志

### 14.15.1 监控系统

```python
class AgentMonitoring:
    """Agent监控系统"""
    def __init__(self):
        self.metrics = {
            "request_count": 0,
            "success_count": 0,
            "error_count": 0,
            "avg_response_time": 0,
            "tool_usage": {}
        }
        self.alerts = []

    def record_request(self, success: bool, response_time: float, tools_used: List[str]):
        """记录请求"""
        self.metrics["request_count"] += 1

        if success:
            self.metrics["success_count"] += 1
        else:
            self.metrics["error_count"] += 1

        # 更新平均响应时间
        total_time = self.metrics["avg_response_time"] * (self.metrics["request_count"] - 1)
        self.metrics["avg_response_time"] = (total_time + response_time) / self.metrics["request_count"]

        # 记录工具使用
        for tool in tools_used:
            self.metrics["tool_usage"][tool] = self.metrics["tool_usage"].get(tool, 0) + 1

        # 检查告警
        self._check_alerts()

    def _check_alerts(self):
        """检查告警"""
        # 检查错误率
        if self.metrics["request_count"] > 10:
            error_rate = self.metrics["error_count"] / self.metrics["request_count"]
            if error_rate > 0.1:
                self.alerts.append({
                    "type": "high_error_rate",
                    "message": f"错误率过高: {error_rate:.2%}",
                    "timestamp": time.time()
                })

        # 检查响应时间
        if self.metrics["avg_response_time"] > 10:
            self.alerts.append({
                "type": "slow_response",
                "message": f"平均响应时间过长: {self.metrics['avg_response_time']:.2f}秒",
                "timestamp": time.time()
            })

    def get_dashboard(self) -> Dict:
        """获取监控仪表板"""
        return {
            "metrics": self.metrics,
            "alerts": self.alerts,
            "health_score": self._compute_health_score()
        }

    def _compute_health_score(self) -> float:
        """计算健康分数"""
        if self.metrics["request_count"] == 0:
            return 1.0

        success_rate = self.metrics["success_count"] / self.metrics["request_count"]
        time_score = max(0, 1 - self.metrics["avg_response_time"] / 30)

        return (success_rate + time_score) / 2
```

---

## 14.16 本章总结与展望

### 14.16.1 核心知识点回顾

本章深入介绍了LangChain/LangGraph Agent实战：

1. **StateGraph**：有状态的Agent工作流。
2. **create_react_agent**：预构建的ReAct Agent。
3. **自定义提示词**：提示词工程和动态生成。
4. **流式执行**：实时查看推理过程。
5. **Human-in-the-Loop**：人工审批机制。
6. **时间旅行调试**：检查点回溯调试。

### 14.16.2 实践建议

1. **从简单开始**：先使用基本功能，再逐步引入高级特性。
2. **注重监控**：建立完善的监控和日志系统。
3. **错误处理**：实现健壮的错误处理和恢复机制。
4. **性能优化**：根据需求进行缓存和并发优化。

### 14.16.3 下一步学习方向

1. **高级状态管理**：学习更复杂的状态管理模式。
2. **多Agent协作**：构建多Agent协作系统。
3. **生产部署**：学习容器化和Kubernetes部署。
4. **性能调优**：深入学习性能优化技术。

掌握本章内容，你将能够使用LangChain/LangGraph构建高效的Agent系统，为实际项目开发奠定坚实基础。

> **下一章预告**
>
> 在第 15 章中，我们将深入 LangGraph 有状态 Agent，学习更高级的状态管理和工作流编排技术。

---

## 14.17 高级 LangGraph 模式

### 14.17.1 子图模式

```python
from langgraph.graph import StateGraph, MessagesState, START, END

def create_subgraph():
    """创建子图"""
    # 子图状态
    class SubgraphState(TypedDict):
        subtask: str
        result: str
        status: str

    # 子图节点
    def process_subtask(state: SubgraphState):
        """处理子任务"""
        llm = ChatOpenAI(model="gpt-4o")
        response = llm.invoke([HumanMessage(content=f"处理子任务：{state['subtask']}")])
        return {"result": response.content, "status": "completed"}

    # 创建子图
    subgraph = StateGraph(SubgraphState)
    subgraph.add_node("process", process_subtask)
    subgraph.add_edge(START, "process")
    subgraph.add_edge("process", END)

    return subgraph.compile()

def create_main_graph_with_subgraph():
    """创建包含子图的主图"""
    # 主图状态
    class MainState(TypedDict):
        task: str
        subtasks: List[str]
        results: List[str]
        final_result: str

    # 主图节点
    def decompose_task(state: MainState):
        """分解任务"""
        llm = ChatOpenAI(model="gpt-4o")
        response = llm.invoke([HumanMessage(content=f"将任务分解为子任务：{state['task']}")])
        subtasks = [s.strip() for s in response.content.split("\n") if s.strip()]
        return {"subtasks": subtasks}

    def process_with_subgraph(state: MainState):
        """使用子图处理"""
        subgraph = create_subgraph()
        results = []

        for subtask in state["subtasks"]:
            # 调用子图
            sub_result = subgraph.invoke({"subtask": subtask, "result": "", "status": "pending"})
            results.append(sub_result["result"])

        return {"results": results}

    def aggregate_results(state: MainState):
        """聚合结果"""
        aggregated = "\n\n".join(state["results"])
        return {"final_result": aggregated}

    # 创建主图
    main_graph = StateGraph(MainState)
    main_graph.add_node("decompose", decompose_task)
    main_graph.add_node("process", process_with_subgraph)
    main_graph.add_node("aggregate", aggregate_results)

    main_graph.add_edge(START, "decompose")
    main_graph.add_edge("decompose", "process")
    main_graph.add_edge("process", "aggregate")
    main_graph.add_edge("aggregate", END)

    return main_graph.compile()
```

### 14.17.2 并行执行模式

```python
class ParallelExecution:
    """并行执行模式"""
    def __init__(self):
        self.execution_pool = None

    def create_parallel_graph(self):
        """创建并行执行图"""
        class ParallelState(TypedDict):
            tasks: List[str]
            results: List[str]
            final_result: str

        def parallel_process(state: ParallelState):
            """并行处理"""
            import concurrent.futures

            def process_task(task: str) -> str:
                llm = ChatOpenAI(model="gpt-4o")
                response = llm.invoke([HumanMessage(content=task)])
                return response.content

            # 并行执行
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_task, task) for task in state["tasks"]]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            return {"results": results}

        def merge_results(state: ParallelState):
            """合并结果"""
            merged = "\n\n".join([f"结果{i+1}: {result}" for i, result in enumerate(state["results"])])
            return {"final_result": merged}

        # 创建图
        graph = StateGraph(ParallelState)
        graph.add_node("parallel_process", parallel_process)
        graph.add_node("merge", merge_results)

        graph.add_edge(START, "parallel_process")
        graph.add_edge("parallel_process", "merge")
        graph.add_edge("merge", END)

        return graph.compile()

    def execute_parallel(self, tasks: List[str]):
        """执行并行任务"""
        graph = self.create_parallel_graph()
        initial_state = {
            "tasks": tasks,
            "results": [],
            "final_result": ""
        }

        result = graph.invoke(initial_state)
        return result["final_result"]
```

### 14.17.3 动态图构建

```python
class DynamicGraphBuilder:
    """动态图构建器"""
    def __init__(self):
        self.node_registry = {}
        self.edge_registry = {}

    def register_node(self, name: str, func: Callable):
        """注册节点"""
        self.node_registry[name] = func

    def register_edge(self, from_node: str, to_node: str, condition: Callable = None):
        """注册边"""
        if from_node not in self.edge_registry:
            self.edge_registry[from_node] = []
        self.edge_registry[from_node].append({"to": to_node, "condition": condition})

    def build_graph(self, state_class):
        """动态构建图"""
        graph = StateGraph(state_class)

        # 添加所有注册的节点
        for name, func in self.node_registry.items():
            graph.add_node(name, func)

        # 添加边
        for from_node, edges in self.edge_registry.items():
            for edge in edges:
                if edge["condition"]:
                    graph.add_conditional_edges(
                        from_node,
                        edge["condition"],
                        {edge["to"]: edge["to"]}
                    )
                else:
                    graph.add_edge(from_node, edge["to"])

        return graph.compile()

    def add_dynamic_branch(self, graph, node_name: str, branch_func: Callable, branches: Dict):
        """添加动态分支"""
        # 这里可以实现动态添加分支的逻辑
        pass
```

---

## 14.18 实战案例：智能客服系统

### 14.18.1 系统架构

```python
class CustomerServiceSystem:
    """智能客服系统"""
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
        self.intent_classifier = IntentClassifier(self.llm)
        self.response_generator = ResponseGenerator(self.llm)
        self.knowledge_base = KnowledgeBase()

    def create_service_graph(self):
        """创建服务图"""
        class ServiceState(TypedDict):
            user_message: str
            intent: str
            context: Dict[str, Any]
            response: str
            session_id: str

        def classify_intent(state: ServiceState):
            """分类意图"""
            intent = self.intent_classifier.classify(state["user_message"])
            return {"intent": intent}

        def route_by_intent(state: ServiceState):
            """根据意图路由"""
            intent = state["intent"]
            if intent == "question":
                return "answer_question"
            elif intent == "complaint":
                return "handle_complaint"
            elif intent == "request":
                return "process_request"
            else:
                return "general_response"

        def answer_question(state: ServiceState):
            """回答问题"""
            # 查询知识库
            relevant_info = self.knowledge_base.search(state["user_message"])

            # 生成回答
            response = self.response_generator.generate_answer(
                question=state["user_message"],
                context=relevant_info
            )

            return {"response": response}

        def handle_complaint(state: ServiceState):
            """处理投诉"""
            response = self.response_generator.generate_complaint_response(
                complaint=state["user_message"]
            )
            return {"response": response}

        def process_request(state: ServiceState):
            """处理请求"""
            response = self.response_generator.generate_request_response(
                request=state["user_message"]
            )
            return {"response": response}

        def general_response(state: ServiceState):
            """通用回复"""
            response = self.response_generator.generate_general_response(
                message=state["user_message"]
            )
            return {"response": response}

        # 创建图
        graph = StateGraph(ServiceState)

        # 添加节点
        graph.add_node("classify", classify_intent)
        graph.add_node("answer_question", answer_question)
        graph.add_node("handle_complaint", handle_complaint)
        graph.add_node("process_request", process_request)
        graph.add_node("general_response", general_response)

        # 添加边
        graph.add_edge(START, "classify")
        graph.add_conditional_edges("classify", route_by_intent, {
            "answer_question": "answer_question",
            "handle_complaint": "handle_complaint",
            "process_request": "process_request",
            "general_response": "general_response"
        })

        graph.add_edge("answer_question", END)
        graph.add_edge("handle_complaint", END)
        graph.add_edge("process_request", END)
        graph.add_edge("general_response", END)

        return graph.compile()

class IntentClassifier:
    """意图分类器"""
    def __init__(self, llm):
        self.llm = llm

    def classify(self, message: str) -> str:
        """分类意图"""
        prompt = f"""
        分类以下用户消息的意图：
        消息：{message}

        可能的意图：question, complaint, request, general

        请只输出意图名称："""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip().lower()

class ResponseGenerator:
    """回复生成器"""
    def __init__(self, llm):
        self.llm = llm

    def generate_answer(self, question: str, context: str) -> str:
        """生成回答"""
        prompt = f"""
        基于以下信息回答问题：
        信息：{context}
        问题：{question}

        请提供准确、友好的回答："""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
```

### 14.18.2 会话管理

```python
class ConversationManager:
    """会话管理器"""
    def __init__(self):
        self.sessions = {}
        self.conversation_history = {}

    def create_session(self, user_id: str) -> str:
        """创建会话"""
        session_id = f"session_{user_id}_{int(time.time())}"
        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": time.time(),
            "status": "active"
        }
        self.conversation_history[session_id] = []
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        """添加消息"""
        if session_id not in self.conversation_history:
            return {"error": "会话不存在"}

        self.conversation_history[session_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

    def get_context(self, session_id: str, max_messages: int = 10) -> List[Dict]:
        """获取上下文"""
        if session_id not in self.conversation_history:
            return []

        history = self.conversation_history[session_id]
        return history[-max_messages:] if len(history) > max_messages else history

    def end_session(self, session_id: str):
        """结束会话"""
        if session_id in self.sessions:
            self.sessions[session_id]["status"] = "ended"
            self.sessions[session_id]["ended_at"] = time.time()

    def get_session_stats(self, session_id: str) -> Dict:
        """获取会话统计"""
        if session_id not in self.sessions:
            return {"error": "会话不存在"}

        session = self.sessions[session_id]
        history = self.conversation_history.get(session_id, [])

        return {
            "session_id": session_id,
            "duration": time.time() - session["created_at"],
            "message_count": len(history),
            "status": session["status"]
        }
```

---

## 14.19 生产环境部署

### 14.19.1 Docker化部署

```python
# Dockerfile示例
dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# docker-compose.yml示例
docker_compose_content = """
version: '3.8'

services:
  agent-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=agent_db
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
"""

# requirements.txt示例
requirements_content = """
langchain>=0.1.0
langgraph>=0.0.10
langchain-openai>=0.0.2
fastapi>=0.104.0
uvicorn>=0.24.0
redis>=5.0.0
sqlalchemy>=2.0.0
pydantic>=2.0.0
"""
```

### 14.19.2 Kubernetes部署

```python
class KubernetesDeployment:
    """Kubernetes部署"""
    def __init__(self):
        self.deployment_config = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "agent-service"},
            "spec": {
                "replicas": 3,
                "selector": {"matchLabels": {"app": "agent"}},
                "template": {
                    "metadata": {"labels": {"app": "agent"}},
                    "spec": {
                        "containers": [{
                            "name": "agent",
                            "image": "agent-service:latest",
                            "ports": [{"containerPort": 8000}],
                            "env": [
                                {"name": "OPENAI_API_KEY", "valueFrom": {"secretKeyRef": {"name": "api-secrets", "key": "openai-key"}}}
                            ],
                            "resources": {
                                "requests": {"memory": "256Mi", "cpu": "250m"},
                                "limits": {"memory": "512Mi", "cpu": "500m"}
                            }
                        }]
                    }
                }
            }
        }

        self.service_config = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "agent-service"},
            "spec": {
                "selector": {"app": "agent"},
                "ports": [{"port": 80, "targetPort": 8000}],
                "type": "LoadBalancer"
            }
        }

    def generate_manifests(self):
        """生成Kubernetes清单"""
        import yaml
        return yaml.dump(self.deployment_config) + "---\n" + yaml.dump(self.service_config)
```

---

## 14.20 本章最终总结

### 14.20.1 知识体系总结

本章全面介绍了LangChain/LangGraph Agent实战：

1. **基础架构**：StateGraph、create_react_agent。
2. **高级模式**：子图、并行执行、动态构建。
3. **提示工程**：自定义提示词、动态生成。
4. **流式处理**：实时输出、可视化。
5. **人工交互**：HITL、审批工作流。
6. **调试技术**：时间旅行、状态恢复。
7. **生产部署**：Docker、Kubernetes。

### 14.20.2 实践技能清单

完成本章学习后，你应该能够：

- [ ] 使用LangGraph构建有状态Agent
- [ ] 实现自定义ReAct Agent
- [ ] 设计提示词工程策略
- [ ] 实现流式执行和监控
- [ ] 构建HITL审批系统
- [ ] 进行时间旅行调试
- [ ] 部署Agent到生产环境

### 14.20.3 项目实战建议

1. **从小项目开始**：先实现简单功能，再逐步复杂化。
2. **注重测试**：建立完善的测试体系。
3. **监控优先**：部署前先建立监控系统。
4. **文档完善**：编写清晰的使用文档。

### 14.20.4 下一步学习方向

1. **高级状态管理**：学习复杂状态模式。
2. **多Agent系统**：构建协作Agent系统。
3. **性能优化**：深入性能调优技术。
4. **安全加固**：学习Agent安全实践。

掌握本章内容，你将能够使用现代框架构建生产级的Agent系统，为AI应用开发奠定坚实基础。

> **下一章预告**
>
> 在第 15 章中，我们将深入 LangGraph 有状态 Agent，探索更高级的状态管理和工作流编排技术，构建更复杂的Agent系统。

---

## 14.21 扩展资源

### 14.21.1 官方文档

- LangChain 官方文档：https://docs.langchain.com/
- LangGraph 官方文档：https://langchain-ai.github.io/langgraph/
- OpenAI API 文档：https://platform.openai.com/docs

### 14.21.2 社区资源

- LangChain GitHub：https://github.com/langchain-ai/langchain
- LangGraph GitHub：https://github.com/langchain-ai/langgraph
- Discord 社区：https://discord.gg/langchain

### 14.21.3 学习路径

1. **入门**：完成 LangChain 官方教程
2. **进阶**：学习 LangGraph 状态图
3. **实战**：构建完整项目
4. **优化**：性能调优和部署

### 14.21.4 常见问题

**Q：如何选择模型？**
A：根据任务复杂度和成本选择，简单任务用GPT-3.5，复杂任务用GPT-4。

**Q：如何处理API限制？**
A：实现重试机制、缓存和请求队列。

**Q：如何保证安全性？**
A：使用环境变量管理密钥，实现输入验证和输出过滤。

掌握本章内容，你将能够使用现代框架构建生产级的Agent系统，为AI应用开发奠定坚实基础。
