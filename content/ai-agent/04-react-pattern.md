---
title: "第4章：ReAct 范式 — 推理与行动的统一"
description: "深入理解 ReAct 范式的理论基础与实现细节，掌握 Thought-Action-Observation 循环，对比纯推理与纯行动策略的优劣，手动实现 ReAct Agent，掌握 LangGraph 集成。"
date: "2026-06-11"
---

# 第4章：ReAct 范式 — 推理与行动的统一

ReAct（Reasoning + Acting）是现代 LLM Agent 的基础范式。

---

## 4.1 ReAct 的理论基础

### 4.1.1 论文核心思想

ReAct 由 Yao et al. (2022) 提出，核心：**推理与行动交替进行，互相增强**。

| 范式 | 优点 | 缺点 | 幻觉率 |
|:---|:---|:---|:---|
| **CoT-only** | 推理过程可解释 | 无法获取外部信息 | 高 |
| **Act-only** | 能获取真实信息 | 缺乏推理 | 低 |
| **ReAct** | 推理+行动互相增强 | Token 消耗较高 | 低 |

```
CoT-only:
  Q: "Who was the president when the Eiffel Tower was built?"
  T1: The Eiffel Tower was built in 1889.
  T2: The president in 1889 was Benjamin Harrison.
  A: Benjamin Harrison
  (可能幻觉)

ReAct:
  Q: "Who was the president when the Eiffel Tower was built?"
  T1: I need to find when the Eiffel Tower was built first.
  A1: search("Eiffel Tower construction date")
  O1: The Eiffel Tower was built in 1889.
  T2: Now I need to find who was the US president in 1889.
  A2: search("US president in 1889")
  O2: Benjamin Harrison was the 23rd president, serving 1889-1893.
  A: Benjamin Harrison
  (有推理过程，可解释，基于真实信息)
```

### 4.1.2 性能对比

| 方法 | HotpotQA Acc. | FEVER Acc. | Token 消耗 |
|:---|:---|:---|:---|
| CoT-only | 32.1% | 60.9% | 低 |
| Act-only | 25.3% | 56.8% | 中 |
| ReAct | 35.1% | 64.6% | 高 |

---

## 4.2 ReAct 循环详解

### 4.2.1 Thought 的作用

1. **理解当前状态**：分析 Observation 的含义
2. **制定策略**：决定下一步该做什么
3. **评估进展**：判断是否已经接近答案
4. **处理错误**：分析工具调用失败的原因

### 4.2.2 完整的 ReAct Prompt

```python
REACT_SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
```

---

## 4.3 手动实现 ReAct Agent

```python
class ReActAgent:
    def __init__(self, model="gpt-4o", max_steps=10, verbose=True):
        self.client = OpenAI()
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose
        self.tools = {}

    def register_tool(self, name, func, description):
        self.tools[name] = {"func": func, "description": description}

    def _build_tools_description(self) -> str:
        return "\n".join(f"- {name}: {info['description']}" for name, info in self.tools.items())

    def _build_prompt(self, question, history):
        history_text = "\n".join([f"{h['type'].title()}: {h['content']}" for h in history])
        return f"""Answer the following questions as best you can. You have access to the following tools:

{self._build_tools_description()}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{', '.join(self.tools.keys())}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
{history_text}"""

    def _parse_output(self, text):
        import re
        final_match = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", text, re.DOTALL)
        if final_match:
            return {"type": "final_answer", "content": final_match.group(1).strip()}
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
        input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text)
        if action_match and input_match:
            action_name = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            try: action_input = json.loads(action_input)
            except: action_input = {"input": action_input}
            return {"type": "action", "name": action_name, "input": action_input}
        return {"type": "unknown", "content": text}

    def _execute_tool(self, name, args):
        tool = self.tools.get(name)
        if not tool: return f"错误：工具 '{name}' 未注册"
        try: return str(tool["func"](**args))
        except Exception as e: return f"错误：{str(e)}"

    def run(self, question):
        history = []
        for step in range(self.max_steps):
            if self.verbose: print(f"\n{'='*50}\nStep {step + 1}/{self.max_steps}")
            prompt = self._build_prompt(question, history)
            response = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=1000)
            llm_output = response.choices[0].message.content
            if self.verbose: print(f"LLM Output:\n{llm_output}")
            parsed = self._parse_output(llm_output)
            if parsed["type"] == "final_answer":
                if self.verbose: print(f"\n✅ Final Answer: {parsed['content']}")
                return parsed["content"]
            elif parsed["type"] == "action":
                thought_match = re.search(r"Thought:\s*(.+?)(?:\nAction:)", llm_output, re.DOTALL)
                if thought_match: history.append({"type": "thought", "content": thought_match.group(1).strip()})
                history.append({"type": "action", "content": f"{parsed['name']}({parsed['input']})"})
                observation = self._execute_tool(parsed["name"], parsed["input"])
                if self.verbose: print(f"📋 Observation: {observation[:200]}...")
                history.append({"type": "observation", "content": observation})
            else:
                history.append({"type": "observation", "content": "请按格式输出。"})
        return "达到最大步数限制，任务未完成。"
```

---

## 4.4 LangChain/LangGraph ReAct Agent

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索互联网获取信息"""
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
print(result["messages"][-1].content)
```

---

## 4.5 ReAct 的局限性与改进

| 局限 | 说明 | 改进方案 |
|:---|:---|:---|
| Token 效率低 | 每步输出 Thought 文本 | ReAct+、Plan-and-Solve |
| 错误传播 | 错误的 Thought 误导后续步骤 | Reflexion |
| 无全局规划 | 只看当前步 | Plan-and-Execute |
| 单步决策 | 每次只执行一个工具 | 工具批处理 |

---

## 4.6 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| ReAct 本质 | 推理与行动交替进行，互相增强 |
| Thought 作用 | 理解状态、制定策略、评估进展、处理错误 |
| 手动实现 | 展示底层机制，适合理解和定制 |
| LangGraph | 生产级实现，支持流式、检查点、错误恢复 |
| 局限性 | Token 效率低、错误传播、无全局规划 |

> **下一章预告**
>
> 在第 5 章中，我们将深入 Function Calling 的底层机制。
