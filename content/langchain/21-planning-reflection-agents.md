# Chapter 21: Planning 与 Self-Critique Agent

## 本章概览

前面的章节中,我们学习了基础 Agent（ReAct）和多 Agent 系统。然而，复杂任务往往需要更强的规划能力和自我改进机制。本章将深入探讨两种高级 Agent 模式：**Planning Agent**（先规划后执行）和 **Reflection Agent**（自我批评与迭代改进）。这些模式能够显著提升 Agent 在复杂任务上的成功率和输出质量。

**本章重点**：
- Plan-and-Execute 框架原理与实现
- 任务分解策略与动态重规划
- Self-Critique 机制设计
- 迭代改进循环（Reflection Loop）
- Tool Error Recovery 容错机制
- Memory-Augmented Agent 长期记忆
- 企业级可靠性工程实践

---

## 21.1 Planning Agent：先规划后执行

### 21.1.1 为什么需要 Planning？

传统 ReAct Agent 的问题：

```python
# ReAct Agent：边思考边行动
user: "帮我组织一次团建活动"
agent: Thought: 我需要先了解预算
      Action: ask_user("预算是多少？")
      Observation: 5000元
      Thought: 我需要查找活动场地
      Action: search("北京团建场地")
      Observation: [场地列表]
      Thought: 我需要...
      # 问题：没有整体规划，容易遗漏步骤、重复工作
```

Planning Agent 的优势：

```python
# Planning Agent：先规划，再执行
user: "帮我组织一次团建活动"

# Step 1: 制定计划
plan = [
    "收集需求（预算、人数、时间、偏好）",
    "搜索并筛选场地",
    "设计活动流程",
    "预算分配",
    "预定场地和服务",
    "发送通知"
]

# Step 2: 按计划执行
for step in plan:
    execute(step)
    
# 优势：结构化、可预测、易于监控
```

### 21.1.2 Plan-and-Execute 框架原理

<div data-component="PlanExecuteFlowDiagram"></div>

核心流程：

```
用户输入
   ↓
规划器 (Planner)
   ↓
生成计划 [Step1, Step2, Step3, ...]
   ↓
执行器 (Executor) ← 循环执行每个步骤
   ↓
观察结果
   ↓
需要重新规划？ 
   ├─ 是 → 回到规划器
   └─ 否 → 继续下一步
   ↓
所有步骤完成 → 输出最终结果
```

### 21.1.3 实现基础 Plan-and-Execute Agent

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from typing import List, TypedDict, Annotated
import operator

# 1. 定义计划结构
class Step(BaseModel):
    """单个执行步骤"""
    id: int
    description: str
    tool: str = Field(description="需要使用的工具名称")
    dependencies: List[int] = Field(default=[], description="依赖的步骤ID")

class Plan(BaseModel):
    """完整的执行计划"""
    goal: str = Field(description="总目标")
    steps: List[Step] = Field(description="执行步骤列表")
    estimated_time: int = Field(description="预计耗时（分钟）")

# 2. 创建 Planner
def create_planner():
    """创建规划器"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的任务规划专家。

你的职责：
1. 理解用户的目标
2. 将复杂任务分解为具体的、可执行的步骤
3. 确定步骤之间的依赖关系
4. 为每个步骤分配合适的工具

可用工具：
- search: 搜索信息
- calculator: 数学计算
- python: 执行Python代码
- database: 查询数据库
- email: 发送邮件

输出格式：JSON，包含 goal, steps, estimated_time"""),
        ("user", "{user_input}")
    ])
    
    planner = prompt | llm.with_structured_output(Plan)
    return planner

# 3. 创建 Executor
def create_executor(tools):
    """创建执行器"""
    from langchain.agents import create_react_agent, AgentExecutor
    
    llm = ChatOpenAI(model="gpt-4")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是任务执行专家。
        严格按照给定的步骤描述执行任务。
        使用提供的工具完成任务。"""),
        ("user", "执行步骤：{step_description}"),
        ("assistant", "我会使用合适的工具完成这个步骤。"),
    ])
    
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return executor

# 4. 定义状态
class PlanExecuteState(TypedDict):
    input: str  # 用户输入
    plan: Plan  # 生成的计划
    current_step: int  # 当前执行到第几步
    step_results: Annotated[List[dict], operator.add]  # 每步的执行结果
    final_output: str  # 最终输出
    need_replan: bool  # 是否需要重新规划

# 5. 实现节点
def planner_node(state: PlanExecuteState):
    """规划节点"""
    planner = create_planner()
    
    # 如果是重新规划，考虑已有结果
    if state.get("need_replan", False):
        context = f"原目标：{state['input']}\n已完成步骤：\n"
        for i, result in enumerate(state.get("step_results", [])):
            context += f"- 步骤{i+1}: {result['description']} → {result['status']}\n"
        
        plan_input = f"{context}\n请重新规划剩余任务。"
    else:
        plan_input = state["input"]
    
    plan = planner.invoke({"user_input": plan_input})
    
    return {
        "plan": plan,
        "current_step": 0,
        "need_replan": False
    }

def executor_node(state: PlanExecuteState):
    """执行节点"""
    plan = state["plan"]
    current_step = state["current_step"]
    
    if current_step >= len(plan.steps):
        # 所有步骤已完成
        return {"final_output": summarize_results(state["step_results"])}
    
    step = plan.steps[current_step]
    
    # 检查依赖是否满足
    for dep_id in step.dependencies:
        dep_result = state["step_results"][dep_id - 1]
        if dep_result["status"] != "success":
            # 依赖步骤失败，需要重新规划
            return {"need_replan": True}
    
    # 执行当前步骤
    executor = create_executor(get_tools_for_step(step.tool))
    
    try:
        result = executor.invoke({
            "step_description": step.description,
            "context": get_dependency_outputs(state["step_results"], step.dependencies)
        })
        
        step_result = {
            "id": step.id,
            "description": step.description,
            "status": "success",
            "output": result["output"]
        }
    except Exception as e:
        step_result = {
            "id": step.id,
            "description": step.description,
            "status": "failed",
            "error": str(e)
        }
        
        # 执行失败，触发重新规划
        return {
            "step_results": [step_result],
            "need_replan": True
        }
    
    return {
        "step_results": [step_result],
        "current_step": current_step + 1
    }

def summarize_results(results: List[dict]) -> str:
    """汇总所有步骤的结果"""
    summary = "任务执行完成！\n\n执行过程：\n"
    for r in results:
        summary += f"✓ {r['description']}\n  结果: {r['output'][:100]}...\n\n"
    return summary

# 6. 构建 Plan-Execute Graph
def build_plan_execute_graph():
    workflow = StateGraph(PlanExecuteState)
    
    # 添加节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    
    # 设置入口
    workflow.set_entry_point("planner")
    
    # 规划器 → 执行器
    workflow.add_edge("planner", "executor")
    
    # 执行器的条件路由
    def should_continue(state: PlanExecuteState):
        if state.get("need_replan", False):
            return "planner"  # 重新规划
        elif state.get("final_output"):
            return "end"  # 完成
        else:
            return "executor"  # 继续执行
    
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "planner": "planner",
            "executor": "executor",
            "end": END
        }
    )
    
    return workflow.compile()

# 使用示例
graph = build_plan_execute_graph()

result = graph.invoke({
    "input": """帮我准备一个关于"LangChain Agent设计模式"的技术分享：
    1. 需要包含最新的技术动态
    2. 准备一些代码示例
    3. 制作PPT大纲"""
})

print("=== 执行计划 ===")
print(f"目标: {result['plan'].goal}")
for step in result['plan'].steps:
    print(f"{step.id}. {step.description} (工具: {step.tool})")

print("\n=== 最终输出 ===")
print(result['final_output'])
```

**输出示例**：

```
=== 执行计划 ===
目标: 准备LangChain Agent设计模式技术分享
1. 搜索LangChain最新动态和Agent设计模式 (工具: search)
2. 整理核心设计模式并编写代码示例 (工具: python)
3. 根据内容生成PPT大纲 (工具: python)

=== 执行过程 ===
执行步骤1: 搜索LangChain最新动态...
✓ 找到5篇相关文章和文档

执行步骤2: 编写代码示例...
✓ 生成ReAct、Plan-Execute、Reflection三个示例

执行步骤3: 生成PPT大纲...
✓ 创建包含10页的演示大纲

=== 最终输出 ===
技术分享准备完成！内容包括：
- 最新技术动态（LangGraph 0.2、Agent优化）
- 3个完整代码示例
- 10页PPT大纲（含引言、模式对比、案例分析、最佳实践）
```

### 21.1.4 高级规划策略

#### 分层规划 (Hierarchical Planning)

对于超大型任务，可以使用分层规划：

```python
class HierarchicalPlan(BaseModel):
    """分层计划"""
    goal: str
    high_level_steps: List[str]  # 高层步骤
    detailed_plans: dict  # 每个高层步骤的详细计划

def hierarchical_planner(goal: str):
    """两层规划器"""
    
    # 第一层：高层规划
    high_level_prompt = f"""将目标分解为3-5个主要阶段：
    目标：{goal}
    
    输出格式：
    1. 阶段1
    2. 阶段2
    3. 阶段3"""
    
    high_level_plan = llm.invoke(high_level_prompt)
    
    # 第二层：每个阶段的详细规划
    detailed_plans = {}
    for phase in high_level_plan.phases:
        detailed_plan = create_planner().invoke({
            "user_input": f"详细规划：{phase}"
        })
        detailed_plans[phase] = detailed_plan
    
    return HierarchicalPlan(
        goal=goal,
        high_level_steps=high_level_plan.phases,
        detailed_plans=detailed_plans
    )

# 使用
plan = hierarchical_planner("构建一个完整的电商网站")

"""
输出：
高层计划：
1. 需求分析和系统设计
2. 后端开发
3. 前端开发
4. 测试和部署

详细计划（阶段1）：
- 收集业务需求
- 设计数据库模型
- 设计API接口
- 制定技术选型
...
"""
```

#### 动态重规划 (Re-planning)

当执行过程中遇到问题时，自动重新规划：

```python
def adaptive_executor(state: PlanExecuteState):
    """自适应执行器"""
    step = state["plan"].steps[state["current_step"]]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = execute_step(step)
            
            # 验证结果质量
            if validate_output(result):
                return {"step_results": [result], "current_step": state["current_step"] + 1}
            else:
                # 质量不达标，调整步骤描述后重试
                step.description = refine_step_description(step, result)
                
        except Exception as e:
            if attempt == max_retries - 1:
                # 多次失败，触发重规划
                return {
                    "need_replan": True,
                    "replan_reason": f"步骤 {step.id} 执行失败: {e}"
                }
    
    return {"need_replan": True}

def replanner_node(state: PlanExecuteState):
    """重规划节点"""
    original_plan = state["plan"]
    completed_steps = state["step_results"]
    
    replan_prompt = f"""原计划执行遇到问题，需要重新规划。

原目标: {original_plan.goal}

已完成步骤:
{format_completed_steps(completed_steps)}

失败原因: {state.get('replan_reason', '未知')}

请生成新的执行计划，考虑已完成的工作，避免重复劳动。"""
    
    new_plan = create_planner().invoke({"user_input": replan_prompt})
    
    return {
        "plan": new_plan,
        "current_step": 0,
        "need_replan": False
    }
```

---

## 21.2 Reflection Agent：自我批评与改进

### 21.2.1 Self-Critique 机制原理

Reflection Agent 能够评估自己的输出质量，并进行迭代改进：

```
初始输出 → 自我评估 → 发现问题 → 改进 → 新输出 → 再评估 → ...
```

<div data-component="ReflectionLoopVisualizer"></div>

**核心理念**：
- 第一次输出往往不是最优的
- 通过自我批评发现问题
- 迭代改进直到满足质量标准

### 21.2.2 实现基础 Reflection Agent

```python
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Literal

# 1. 定义评估结构
class Critique(BaseModel):
    """批评意见"""
    aspect: str = Field(description="评估的方面（如准确性、完整性、可读性）")
    score: int = Field(description="评分 1-10")
    issues: List[str] = Field(description="发现的具体问题")
    suggestions: List[str] = Field(description="改进建议")

class SelfAssessment(BaseModel):
    """自我评估"""
    overall_score: int = Field(description="总体评分 1-10")
    critiques: List[Critique]
    is_acceptable: bool = Field(description="是否达到可接受标准")

# 2. 创建 Generator（生成器）
def create_generator(task_type: str):
    """创建内容生成器"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    prompts = {
        "essay": """你是专业作家。任务：撰写一篇文章。
        
要求：
- 逻辑清晰，论证充分
- 语言流畅，表达准确
- 结构完整（引言、正文、结论）""",
        
        "code": """你是资深工程师。任务：编写高质量代码。
        
要求：
- 代码正确、高效
- 遵循最佳实践
- 注释充分""",
        
        "analysis": """你是数据分析师。任务：进行深入分析。
        
要求：
- 数据准确
- 分析全面
- 结论有说服力"""
    }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts.get(task_type, prompts["essay"])),
        ("user", "{task_description}"),
        ("assistant", "{previous_attempt}"),  # 如果是改进版本
        ("user", "{improvement_instructions}")  # 改进指导
    ])
    
    return prompt | llm

# 3. 创建 Critic（批评家）
def create_critic(task_type: str):
    """创建自我批评器"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    critic_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是严格的质量评审专家，负责评估{task_type}的质量。

评估维度：
1. 准确性 (Accuracy)
2. 完整性 (Completeness)
3. 清晰度 (Clarity)
4. 专业性 (Professionalism)
5. 创新性 (Creativity)

对每个维度打分（1-10），指出具体问题和改进建议。
如果总体评分低于8分，标记为不可接受。"""),
        ("user", "请评估以下内容：\n\n{content}")
    ])
    
    return critic_prompt | llm.with_structured_output(SelfAssessment)

# 4. 实现 Reflection Loop
class ReflectionState(TypedDict):
    task: str  # 任务描述
    current_output: str  # 当前输出
    critiques: List[SelfAssessment]  # 历次评估
    iteration: int  # 迭代次数
    final_output: str  # 最终输出

def generator_node(state: ReflectionState):
    """生成节点"""
    generator = create_generator("essay")
    
    # 首次生成
    if state["iteration"] == 0:
        result = generator.invoke({
            "task_description": state["task"],
            "previous_attempt": "",
            "improvement_instructions": ""
        })
    else:
        # 改进版本
        last_critique = state["critiques"][-1]
        improvement_instructions = format_improvement_instructions(last_critique)
        
        result = generator.invoke({
            "task_description": state["task"],
            "previous_attempt": state["current_output"],
            "improvement_instructions": improvement_instructions
        })
    
    return {
        "current_output": result.content,
        "iteration": state["iteration"] + 1
    }

def critic_node(state: ReflectionState):
    """批评节点"""
    critic = create_critic("essay")
    
    assessment = critic.invoke({"content": state["current_output"]})
    
    return {
        "critiques": state.get("critiques", []) + [assessment]
    }

def should_continue(state: ReflectionState):
    """决定是否继续迭代"""
    MAX_ITERATIONS = 5
    
    if state["iteration"] >= MAX_ITERATIONS:
        return "finish"  # 达到最大迭代次数
    
    last_critique = state["critiques"][-1]
    if last_critique.is_acceptable:
        return "finish"  # 质量达标
    
    return "continue"  # 继续改进

def build_reflection_graph():
    """构建反思循环图"""
    workflow = StateGraph(ReflectionState)
    
    workflow.add_node("generate", generator_node)
    workflow.add_node("critique", critic_node)
    
    workflow.set_entry_point("generate")
    
    # 生成 → 批评
    workflow.add_edge("generate", "critique")
    
    # 批评 → 决定下一步
    workflow.add_conditional_edges(
        "critique",
        should_continue,
        {
            "continue": "generate",  # 继续改进
            "finish": END
        }
    )
    
    return workflow.compile()

# 使用示例
reflection_graph = build_reflection_graph()

result = reflection_graph.invoke({
    "task": "写一篇关于'AI Agent 在企业中的应用'的技术博客",
    "iteration": 0,
    "critiques": []
})

print(f"迭代次数: {result['iteration']}")
print("\n=== 历次评估 ===")
for i, critique in enumerate(result['critiques'], 1):
    print(f"\n第{i}次评估:")
    print(f"总分: {critique.overall_score}/10")
    for c in critique.critiques:
        print(f"- {c.aspect}: {c.score}/10")
        if c.issues:
            print(f"  问题: {', '.join(c.issues)}")

print("\n=== 最终输出 ===")
print(result['current_output'])
```

**执行过程示例**：

```
迭代1:
生成初稿 → 评估: 6/10 (缺少具体案例、结构不够清晰)

迭代2:
改进版本 → 评估: 7.5/10 (案例较好，但技术深度不足)

迭代3:
再次改进 → 评估: 8.5/10 (达到可接受标准) → 完成
```

### 21.2.3 多角度批评机制

使用多个批评家从不同角度评估：

```python
def create_multi_perspective_critics():
    """创建多角度批评家"""
    
    critics = {
        "technical": ChatPromptTemplate.from_messages([
            ("system", "你是技术专家，评估技术准确性和深度。"),
            ("user", "{content}")
        ]),
        
        "readability": ChatPromptTemplate.from_messages([
            ("system", "你是编辑，评估可读性和表达质量。"),
            ("user", "{content}")
        ]),
        
        "structure": ChatPromptTemplate.from_messages([
            ("system", "你是逻辑专家，评估结构和论证逻辑。"),
            ("user", "{content}")
        ])
    }
    
    return {name: prompt | llm.with_structured_output(Critique) 
            for name, prompt in critics.items()}

def comprehensive_critique(content: str):
    """综合评估"""
    critics = create_multi_perspective_critics()
    
    critiques = {}
    for name, critic in critics.items():
        critiques[name] = critic.invoke({"content": content})
    
    # 计算综合评分
    avg_score = sum(c.score for c in critiques.values()) / len(critiques)
    
    return {
        "critiques": critiques,
        "average_score": avg_score,
        "is_acceptable": avg_score >= 8.0
    }
```

### 21.2.4 对比改进策略

除了自我批评，还可以通过生成多个版本进行对比：

```python
def generate_multiple_versions(task: str, num_versions: int = 3):
    """生成多个版本并选择最佳"""
    generator = create_generator("essay")
    critic = create_critic("essay")
    
    versions = []
    
    for i in range(num_versions):
        # 每个版本使用不同的温度参数
        version = generator.invoke({
            "task_description": task,
            "temperature": 0.5 + i * 0.2  # 0.5, 0.7, 0.9
        })
        
        assessment = critic.invoke({"content": version.content})
        
        versions.append({
            "content": version.content,
            "score": assessment.overall_score,
            "assessment": assessment
        })
    
    # 选择得分最高的版本
    best_version = max(versions, key=lambda v: v["score"])
    
    return best_version
```

---

## 21.3 Memory-Augmented Agent：长期记忆

### 21.3.1 为什么需要长期记忆？

Agent 在处理多个任务时，可以从历史经验中学习：

```python
# 无记忆Agent：每次都是新手
task1 = agent.invoke("分析销售数据")  # 摸索如何分析
task2 = agent.invoke("分析用户数据")  # 又重新摸索
task3 = agent.invoke("分析财务数据")  # 再次摸索

# 有记忆Agent：积累经验
task1 = agent.invoke("分析销售数据")  # 学到分析方法
# 存储：成功的分析流程、常见问题、最佳实践

task2 = agent.invoke("分析用户数据")  # 复用之前的方法
# 存储：新的insights、改进的流程

task3 = agent.invoke("分析财务数据")  # 综合运用所有经验
```

### 21.3.2 实现长期记忆系统

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from datetime import datetime

class AgentMemory:
    """Agent长期记忆系统"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name="agent_memory",
            embedding_function=self.embeddings
        )
        self.episodic_memory = []  # 情景记忆（具体任务）
        self.semantic_memory = {}  # 语义记忆（概念、方法）
    
    def store_episode(self, task: str, solution: str, outcome: str, success: bool):
        """存储任务执行记录"""
        episode = {
            "task": task,
            "solution": solution,
            "outcome": outcome,
            "success": success,
            "timestamp": datetime.now(),
            "metadata": {
                "task_type": classify_task(task),
                "tools_used": extract_tools(solution)
            }
        }
        
        self.episodic_memory.append(episode)
        
        # 同时存入向量数据库，支持语义检索
        self.vectorstore.add_texts(
            texts=[f"任务: {task}\n解决方案: {solution}\n结果: {outcome}"],
            metadatas=[episode["metadata"]],
            ids=[f"episode_{len(self.episodic_memory)}"]
        )
    
    def retrieve_similar_experiences(self, current_task: str, k: int = 3):
        """检索相似的历史经验"""
        results = self.vectorstore.similarity_search(current_task, k=k)
        
        similar_episodes = []
        for doc in results:
            # 解析存储的episode
            similar_episodes.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return similar_episodes
    
    def extract_lessons(self):
        """从历史中提取经验教训"""
        # 分析成功和失败的模式
        successful_tasks = [e for e in self.episodic_memory if e["success"]]
        failed_tasks = [e for e in self.episodic_memory if not e["success"]]
        
        lessons = {
            "success_patterns": analyze_patterns(successful_tasks),
            "common_pitfalls": analyze_patterns(failed_tasks),
            "best_practices": extract_best_practices(successful_tasks)
        }
        
        self.semantic_memory["lessons"] = lessons
        return lessons

# 集成到 Agent
class MemoryAugmentedAgent:
    """带记忆的Agent"""
    
    def __init__(self):
        self.agent = create_react_agent(...)
        self.memory = AgentMemory()
    
    def invoke(self, task: str):
        # 1. 检索相关历史经验
        similar_experiences = self.memory.retrieve_similar_experiences(task)
        
        # 2. 将经验注入到prompt
        context = self.format_experiences(similar_experiences)
        enhanced_prompt = f"""{task}

参考以下历史经验：
{context}

请根据这些经验执行任务。"""
        
        # 3. 执行任务
        try:
            result = self.agent.invoke(enhanced_prompt)
            success = True
        except Exception as e:
            result = str(e)
            success = False
        
        # 4. 存储本次经验
        self.memory.store_episode(
            task=task,
            solution=result.get("output", ""),
            outcome=result,
            success=success
        )
        
        return result
    
    def format_experiences(self, experiences: List[dict]) -> str:
        """格式化历史经验"""
        if not experiences:
            return "暂无相关历史经验。"
        
        formatted = "相关历史经验：\n"
        for i, exp in enumerate(experiences, 1):
            formatted += f"\n{i}. {exp['content']}\n"
        
        return formatted

# 使用示例
agent = MemoryAugmentedAgent()

# 第一次任务
agent.invoke("分析Q1销售数据，找出增长驱动因素")

# 第二次任务（自动利用第一次的经验）
agent.invoke("分析Q2销售数据，对比Q1找出变化")

# 提取经验教训
lessons = agent.memory.extract_lessons()
print("学到的最佳实践：", lessons["best_practices"])
```

### 21.3.3 经验迁移学习

将成功经验应用到新领域：

```python
def transfer_learning(source_domain: str, target_domain: str):
    """领域迁移学习"""
    
    # 1. 提取源领域的成功模式
    source_experiences = memory.retrieve_by_domain(source_domain, success_only=True)
    patterns = extract_abstract_patterns(source_experiences)
    
    # 2. 将模式抽象化
    abstract_strategies = [
        "先探索数据特征，再选择分析方法",
        "使用可视化辅助理解",
        "验证假设时使用多种方法交叉验证"
    ]
    
    # 3. 应用到目标领域
    adapted_strategies = adapt_strategies(abstract_strategies, target_domain)
    
    return adapted_strategies
```

---

## 21.4 Tool Error Recovery：容错机制

### 21.4.1 工具调用失败的常见原因

| 错误类型 | 原因 | 示例 |
|---------|------|------|
| **参数错误** | Agent传递了错误的参数 | `search(query=123)` # query应该是字符串 |
| **超时** | 工具执行时间过长 | 网络请求超时、数据库查询慢 |
| **权限不足** | 没有访问权限 | 无法读取受限文件 |
| **资源不可用** | 依赖的服务宕机 | API服务500错误 |
| **结果解析失败** | 返回格式不符合预期 | JSON解析错误 |

### 21.4.2 实现容错机制

```python
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Any, Callable

class RobustTool:
    """带容错机制的工具包装器"""
    
    def __init__(self, tool: Callable, fallback_tools: List[Callable] = None):
        self.tool = tool
        self.fallback_tools = fallback_tools or []
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def invoke_with_retry(self, *args, **kwargs):
        """带重试的调用"""
        try:
            return self.tool(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Tool {self.tool.__name__} failed: {e}")
            raise
    
    def invoke_with_fallback(self, *args, **kwargs):
        """带降级的调用"""
        try:
            return self.invoke_with_retry(*args, **kwargs)
        except Exception as e:
            logging.error(f"Primary tool failed: {e}")
            
            # 尝试备用工具
            for fallback_tool in self.fallback_tools:
                try:
                    logging.info(f"Trying fallback: {fallback_tool.__name__}")
                    return fallback_tool(*args, **kwargs)
                except Exception as fallback_error:
                    logging.warning(f"Fallback {fallback_tool.__name__} also failed: {fallback_error}")
                    continue
            
            # 所有工具都失败
            raise Exception(f"All tools failed for {self.tool.__name__}")

# 使用示例
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun

# 主工具：DuckDuckGo搜索
primary_search = DuckDuckGoSearchRun()

# 备用工具：Wikipedia
fallback_search = WikipediaQueryRun()

# 创建健壮的搜索工具
robust_search = RobustTool(
    tool=primary_search,
    fallback_tools=[fallback_search]
)

# Agent使用
result = robust_search.invoke_with_fallback("LangChain tutorials")
```

### 21.4.3 错误反馈与自愈

将错误信息反馈给 Agent，让其调整策略：

```python
class SelfHealingAgent:
    """自愈Agent"""
    
    def __init__(self, agent, max_heal_attempts: int = 3):
        self.agent = agent
        self.max_heal_attempts = max_heal_attempts
    
    def invoke(self, task: str):
        messages = [HumanMessage(content=task)]
        
        for attempt in range(self.max_heal_attempts):
            try:
                result = self.agent.invoke({"messages": messages})
                return result  # 成功
                
            except ToolException as e:
                # 工具错误，提供错误信息让Agent调整
                error_feedback = f"""工具调用失败：{e}

可能的原因：
1. 参数格式不正确
2. 工具暂时不可用
3. 权限不足

请：
1. 检查参数是否正确
2. 考虑使用其他工具
3. 调整执行策略

请重新尝试。"""
                
                messages.append(AIMessage(content=str(result)))
                messages.append(HumanMessage(content=error_feedback))
                
                logging.info(f"Self-healing attempt {attempt + 1}/{self.max_heal_attempts}")
                
        raise Exception(f"Agent failed after {self.max_heal_attempts} self-healing attempts")

# 使用
agent = SelfHealingAgent(create_react_agent(...))
result = agent.invoke("搜索LangChain最新功能并总结")
```

<div data-component="ErrorRecoveryFlowDiagram"></div>

### 21.4.4 部分结果处理

即使某些步骤失败，也能利用已成功的部分：

```python
class PartialResultHandler:
    """部分结果处理器"""
    
    def execute_plan_with_partial_results(self, plan: Plan):
        """执行计划，容忍部分失败"""
        results = {
            "successful_steps": [],
            "failed_steps": [],
            "partial_output": None
        }
        
        for step in plan.steps:
            try:
                result = execute_step(step)
                results["successful_steps"].append({
                    "step": step,
                    "result": result
                })
            except Exception as e:
                results["failed_steps"].append({
                    "step": step,
                    "error": str(e)
                })
                
                # 评估是否可以继续
                if step.critical:
                    # 关键步骤失败，终止
                    break
                else:
                    # 非关键步骤，继续执行
                    logging.warning(f"Non-critical step {step.id} failed, continuing...")
        
        # 基于成功的部分生成输出
        if results["successful_steps"]:
            results["partial_output"] = generate_partial_output(
                results["successful_steps"]
            )
        
        return results
```

---

## 21.5 企业级 Agent 可靠性工程

### 21.5.1 超时控制

```python
from langchain_core.runnables import RunnableConfig
import signal

class TimeoutAgent:
    """带超时控制的Agent"""
    
    def invoke_with_timeout(self, input_data: dict, timeout_seconds: int = 60):
        """设置超时时间"""
        
        config = RunnableConfig(
            timeout=timeout_seconds,
            max_concurrency=5
        )
        
        try:
            result = self.agent.invoke(input_data, config=config)
            return result
        except TimeoutError:
            logging.error(f"Agent timed out after {timeout_seconds} seconds")
            return {
                "status": "timeout",
                "partial_result": self.get_partial_result()
            }

# 使用UNIX信号实现更严格的超时
def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def strict_timeout_invoke(agent, input_data, timeout: int):
    """严格的超时控制"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        result = agent.invoke(input_data)
        signal.alarm(0)  # 取消超时
        return result
    except TimeoutError:
        signal.alarm(0)
        return {"error": "timeout"}
```

### 21.5.2 成本控制

```python
class CostAwareAgent:
    """成本感知Agent"""
    
    def __init__(self, budget_tokens: int = 100000):
        self.budget_tokens = budget_tokens
        self.used_tokens = 0
    
    def invoke(self, input_data: dict):
        # 预估token消耗
        estimated_tokens = estimate_tokens(input_data)
        
        if self.used_tokens + estimated_tokens > self.budget_tokens:
            raise BudgetExceededError(
                f"Token budget exceeded: {self.used_tokens}/{self.budget_tokens}"
            )
        
        # 使用回调追踪实际消耗
        from langchain.callbacks import get_openai_callback
        
        with get_openai_callback() as cb:
            result = self.agent.invoke(input_data)
            
            self.used_tokens += cb.total_tokens
            
            logging.info(f"Used {cb.total_tokens} tokens, "
                        f"Total: {self.used_tokens}/{self.budget_tokens}")
        
        return result
    
    def get_cost_report(self):
        """成本报告"""
        return {
            "total_budget": self.budget_tokens,
            "used": self.used_tokens,
            "remaining": self.budget_tokens - self.used_tokens,
            "utilization": f"{self.used_tokens/self.budget_tokens*100:.1f}%"
        }
```

### 21.5.3 幻觉检测与缓解

```python
class HallucinationDetector:
    """幻觉检测器"""
    
    def detect(self, output: str, context: str) -> dict:
        """检测潜在的幻觉"""
        
        checks = {
            "factual_consistency": self.check_factual_consistency(output, context),
            "citation_validity": self.check_citations(output),
            "numerical_accuracy": self.check_numbers(output, context)
        }
        
        hallucination_score = sum(
            1 for check in checks.values() if not check["passed"]
        ) / len(checks)
        
        return {
            "is_likely_hallucination": hallucination_score > 0.3,
            "confidence": 1 - hallucination_score,
            "checks": checks
        }
    
    def check_factual_consistency(self, output: str, context: str):
        """检查事实一致性"""
        # 使用LLM验证
        verifier = ChatOpenAI(model="gpt-4")
        
        prompt = f"""验证以下输出是否与给定上下文一致：

上下文：
{context}

输出：
{output}

返回：
- consistent: true/false
- inconsistencies: [列出不一致的地方]"""
        
        result = verifier.invoke(prompt)
        # 解析结果
        return {"passed": "consistent: true" in result.content.lower()}

# 使用
detector = HallucinationDetector()

output = agent.invoke("介绍LangChain")
detection = detector.detect(output, context=search_results)

if detection["is_likely_hallucination"]:
    logging.warning("Potential hallucination detected!")
    # 触发重新生成或人工审核
```

### 21.5.4 输出验证

```python
class OutputValidator:
    """输出验证器"""
    
    def validate(self, output: Any, schema: BaseModel) -> dict:
        """验证输出格式和内容"""
        
        validations = {
            "format": self.validate_format(output, schema),
            "completeness": self.validate_completeness(output, schema),
            "quality": self.validate_quality(output)
        }
        
        all_passed = all(v["passed"] for v in validations.values())
        
        return {
            "valid": all_passed,
            "validations": validations,
            "issues": [v["error"] for v in validations.values() if not v["passed"]]
        }
    
    def validate_format(self, output: Any, schema: BaseModel):
        """验证格式"""
        try:
            schema.parse_obj(output)
            return {"passed": True}
        except Exception as e:
            return {"passed": False, "error": f"Format error: {e}"}
    
    def validate_completeness(self, output: Any, schema: BaseModel):
        """验证完整性"""
        required_fields = get_required_fields(schema)
        missing = [f for f in required_fields if f not in output]
        
        return {
            "passed": len(missing) == 0,
            "error": f"Missing fields: {missing}" if missing else None
        }
    
    def validate_quality(self, output: str):
        """验证质量"""
        quality_checks = {
            "min_length": len(output) >= 100,
            "has_structure": bool(re.search(r'\n\s*\n', output)),  # 有段落
            "no_placeholders": "TODO" not in output and "..." not in output
        }
        
        passed = all(quality_checks.values())
        
        return {
            "passed": passed,
            "error": f"Quality issues: {[k for k, v in quality_checks.items() if not v]}" if not passed else None
        }
```

---

## 21.6 完整案例：智能研究助手

将 Planning、Reflection、Memory 结合的完整系统：

```python
class AdvancedResearchAssistant:
    """高级研究助手：Planning + Reflection + Memory"""
    
    def __init__(self):
        self.planner = create_planner()
        self.executor = create_executor(tools)
        self.critic = create_critic("research")
        self.memory = AgentMemory()
    
    def research(self, topic: str, quality_threshold: float = 8.0):
        """执行研究任务"""
        
        # Phase 1: Planning
        print("📋 制定研究计划...")
        
        # 检索相似历史经验
        similar_research = self.memory.retrieve_similar_experiences(topic)
        
        plan = self.planner.invoke({
            "user_input": topic,
            "historical_insights": format_insights(similar_research)
        })
        
        print(f"计划包含 {len(plan.steps)} 个步骤")
        
        # Phase 2: Execute with Reflection
        research_output = None
        iteration = 0
        max_iterations = 3
        
        while iteration < max_iterations:
            print(f"\n🔄 执行轮次 {iteration + 1}")
            
            # 执行计划
            execution_results = []
            for step in plan.steps:
                print(f"  执行: {step.description}")
                result = self.executor.invoke(step)
                execution_results.append(result)
            
            # 整合结果
            research_output = synthesize_results(execution_results)
            
            # Self-Critique
            print("  🔍 质量评估...")
            critique = self.critic.invoke({"content": research_output})
            
            print(f"  评分: {critique.overall_score}/10")
            
            if critique.overall_score >= quality_threshold:
                print("  ✓ 达到质量标准")
                break
            else:
                print(f"  ✗ 需要改进: {', '.join(critique.critiques[0].issues)}")
                
                # 根据批评改进计划
                plan = self.refine_plan(plan, critique, execution_results)
                iteration += 1
        
        # Phase 3: Store Experience
        self.memory.store_episode(
            task=topic,
            solution=str(plan),
            outcome=research_output,
            success=critique.overall_score >= quality_threshold
        )
        
        # 生成报告
        report = self.generate_report(
            topic=topic,
            plan=plan,
            output=research_output,
            critique=critique,
            iterations=iteration + 1
        )
        
        return report
    
    def refine_plan(self, original_plan: Plan, critique: SelfAssessment, 
                    results: List) -> Plan:
        """根据批评意见改进计划"""
        
        refinement_prompt = f"""原计划存在以下问题：
{format_critique(critique)}

已执行步骤及结果：
{format_results(results)}

请生成改进的计划，针对性地解决这些问题。"""
        
        new_plan = self.planner.invoke({"user_input": refinement_prompt})
        return new_plan
    
    def generate_report(self, **kwargs) -> dict:
        """生成研究报告"""
        return {
            "topic": kwargs["topic"],
            "final_output": kwargs["output"],
            "quality_score": kwargs["critique"].overall_score,
            "iterations": kwargs["iterations"],
            "plan_summary": summarize_plan(kwargs["plan"]),
            "timestamp": datetime.now()
        }

# 使用
assistant = AdvancedResearchAssistant()

report = assistant.research(
    topic="LangGraph在企业AI系统中的应用模式与最佳实践",
    quality_threshold=8.5
)

print("\n" + "="*60)
print("研究报告")
print("="*60)
print(f"主题: {report['topic']}")
print(f"质量评分: {report['quality_score']}/10")
print(f"迭代次数: {report['iterations']}")
print(f"\n{report['final_output']}")
```

**输出示例**：

```
📋 制定研究计划...
找到3条相似历史研究
计划包含 5 个步骤

🔄 执行轮次 1
  执行: 搜索LangGraph最新文档和案例
  执行: 分析企业应用模式
  执行: 整理最佳实践
  执行: 编写技术分析
  执行: 生成代码示例
  🔍 质量评估...
  评分: 7.5/10
  ✗ 需要改进: 缺少实际案例数据, 最佳实践不够具体

🔄 执行轮次 2
  执行: 补充实际案例研究
  执行: 深化最佳实践分析
  执行: 优化技术分析深度
  🔍 质量评估...
  评分: 8.7/10
  ✓ 达到质量标准

============================================================
研究报告
============================================================
主题: LangGraph在企业AI系统中的应用模式与最佳实践
质量评分: 8.7/10
迭代次数: 2

[详细的研究内容...]
```

---

## 本章总结

本章深入探讨了高级 Agent 模式：

**核心概念**：
- **Planning Agent**：先规划后执行，结构化任务处理
- **Reflection Agent**：自我批评与迭代改进
- **Memory-Augmented Agent**：从历史经验中学习
- **Error Recovery**：工具失败的容错与自愈

**关键技术**：
- Plan-and-Execute 框架实现
- 分层规划与动态重规划
- 多角度批评机制
- 长期记忆的存储与检索
- 工具调用的重试与降级
- 成本控制与质量验证

**最佳实践**：
- 设置合理的迭代次数上限（3-5次）
- 使用结构化输出提升规划质量
- 建立多维度的评估标准
- 平衡质量要求与执行成本
- 完善的监控和日志

**生产部署建议**：
- 超时控制：防止无限执行
- 成本控制：Token预算管理
- 质量保证：输出验证与幻觉检测
- 可观测性：完整的执行追踪

这些高级模式显著提升了 Agent 在复杂任务上的表现，是构建企业级 AI 应用的关键技术。

---

## 练习题

1. **Planning优化**：设计一个能够并行执行独立步骤的 Plan-Execute Agent

2. **Reflection改进**：实现一个支持"多专家投票"的批评机制

3. **Memory应用**：构建一个能够从失败中学习的 Agent（分析失败模式）

4. **容错挑战**：实现一个支持"部分重试"的 Agent（只重试失败的步骤）

5. **综合案例**：构建一个"智能代码审查Agent"，结合Planning（审查计划）+ Reflection（多轮改进）+ Memory（编码规范库）

---

## 扩展阅读

- [Plan-and-Solve Prompting (Paper)](https://arxiv.org/abs/2305.04091)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [LangGraph Plan-and-Execute Tutorial](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/)
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
- [Tree of Thoughts: Deliberate Problem Solving with LLM](https://arxiv.org/abs/2305.10601)
