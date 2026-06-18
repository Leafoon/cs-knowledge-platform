---
title: "第17章：CrewAI 协作智能体框架"
description: "掌握 CrewAI 的角色定义、任务编排、协作流程、记忆集成、工具系统与层级管理。"
date: "2026-06-11"
---

 # 第17章：CrewAI 协作智能体框架

 CrewAI 是一个以角色扮演为核心的多 Agent 协作框架，通过模拟真实团队协作的方式组织 AI Agent 完成复杂任务。本章深入讲解 CrewAI 的架构设计、核心概念和生产实践。

 下面的交互式演示展示了 CrewAI 的角色定义：

 <div data-component="CrewAIRoles"></div>

 ## 什么是 CrewAI？

CrewAI 的核心理念是**角色扮演**。它将多 Agent 协作模拟为一个真实团队的工作方式：

- **每个 Agent 有明确的角色**：如研究员、分析师、写手
- **每个 Agent 有特定的目标**：如收集数据、分析趋势、撰写报告
- **Agent 之间通过对话协作**：就像真实团队成员之间的沟通
- **有明确的工作流程**：顺序执行或层级管理

这种设计使得：
1. **易于理解**：用人类熟悉的概念（团队、角色、任务）来组织 AI
2. **易于配置**：通过定义角色和目标来创建 Agent
3. **易于扩展**：可以随时添加新的角色或调整流程

## CrewAI vs 其他框架

| 特性 | CrewAI | AutoGen | LangGraph |
|:---|:---|:---|:---|
| **核心理念** | 角色扮演 | 对话驱动 | 图编排 |
| **学习曲线** | 低 | 中 | 高 |
| **配置难度** | 简单 | 中等 | 复杂 |
| **适用场景** | 团队协作 | 多轮对话 | 复杂流程 |

**选择建议**：
- 如果你需要模拟团队协作，选择 CrewAI
- 如果你需要灵活的对话控制，选择 AutoGen
- 如果你需要复杂的流程编排，选择 LangGraph

---

## 17.1 CrewAI 架构设计

### 17.1.1 整体架构

CrewAI 的架构围绕几个核心概念设计：

```
┌─────────────────────────────────────────────────────────────┐
│                     CrewAI 架构                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │    Crew     │    │    Task     │    │    Agent    │     │
│  │   (团队)    │───▶│   (任务)    │───▶│   (角色)    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Process   │    │   Memory    │    │   Tools     │     │
│  │   (流程)    │    │   (记忆)    │    │   (工具)    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│                    ┌─────────────┐                          │
│                    │   Manager   │                          │
│                    │   (管理器)  │                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

**Crew（团队）**：整个系统的顶层容器，包含多个 Agent 和 Task，定义了团队的工作方式。

**Task（任务）**：具体的工作单元，描述需要完成什么工作，由哪个 Agent 负责。

**Agent（角色）**：执行任务的 AI 实体，有自己的角色、目标和背景故事。

**Process（流程）**：定义任务的执行顺序，可以是顺序执行（Sequential）或层级管理（Hierarchical）。

**Memory（记忆）**：让 Agent 能够记住之前的对话和经验。

**Tools（工具）**：Agent 可以使用的外部工具，如搜索引擎、数据库等。

**Manager（管理器）**：在层级流程中，负责分配任务和协调 Agent。

```
┌─────────────────────────────────────────────────────────────┐
│                     CrewAI 架构                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │    Crew     │    │    Task     │    │    Agent    │     │
│  │   (团队)    │───▶│   (任务)    │───▶│   (角色)    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Process   │    │   Memory    │    │   Tools     │     │
│  │   (流程)    │    │   (记忆)    │    │   (工具)    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│                    ┌─────────────┐                          │
│                    │   Manager   │                          │
│                    │   (管理器)  │                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### 17.1.2 核心组件

```python
from crewai import Agent, Task, Crew, Process
from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory
from crewai.tools import BaseTool
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class CrewConfig:
    """Crew 配置"""
    name: str
    description: str
    process: Process = Process.sequential
    verbose: bool = True
    memory: bool = True
    max_iterations: int = 10
    temperature: float = 0.7

class CrewAIManager:
    """CrewAI 管理器"""
    
    def __init__(self, config: CrewConfig):
        self.config = config
        self.agents: List[Agent] = []
        self.tasks: List[Task] = []
        self.crew: Optional[Crew] = None
    
    def create_agent(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: List[BaseTool] = None,
        llm: str = "gpt-4o",
        **kwargs
    ) -> Agent:
        """创建 Agent"""
        agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools or [],
            llm=llm,
            verbose=self.config.verbose,
            memory=True,
            max_iter=kwargs.get('max_iter', 15),
            allow_delegation=kwargs.get('allow_delegation', False),
            **{k: v for k, v in kwargs.items() if k not in ['max_iter', 'allow_delegation']}
        )
        self.agents.append(agent)
        return agent
    
    def create_task(
        self,
        description: str,
        expected_output: str,
        agent: Agent,
        context: List[Task] = None,
        **kwargs
    ) -> Task:
        """创建任务"""
        task = Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            context=context or [],
            **kwargs
        )
        self.tasks.append(task)
        return task
    
    def build_crew(self) -> Crew:
        """构建 Crew"""
        self.crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=self.config.process,
            verbose=self.config.verbose,
            memory=self.config.memory,
            max_iterations=self.config.max_iterations
        )
        return self.crew
    
    async def execute(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行 Crew"""
        if not self.crew:
            self.build_crew()
        
        result = await self.crew.kickoff_async(inputs=inputs or {})
        
        return {
            "result": result,
            "agents": [agent.role for agent in self.agents],
            "tasks_completed": len(self.tasks)
        }
```

---

## 17.2 Agent 角色定义

### 17.2.1 角色设计原则

```python
class RoleDesigner:
    """角色设计器"""
    
    @staticmethod
    def create_specialized_roles() -> List[Dict[str, Any]]:
        """创建专业化角色"""
        return [
            {
                "role": "首席研究分析师",
                "goal": "深入分析 {topic} 领域的最新发展，识别关键趋势和机会",
                "backstory": """你是一位拥有 15 年经验的资深研究分析师，曾在多家顶级咨询公司
                担任首席分析师。你擅长从复杂数据中提取洞察，对行业趋势有敏锐的嗅觉。
                你的分析报告曾帮助多家企业做出关键战略决策。""",
                "expertise": ["数据分析", "趋势预测", "报告撰写"],
                "personality": "严谨、细致、善于发现问题"
            },
            {
                "role": "技术架构师",
                "goal": "设计 {topic} 的技术架构方案，确保可扩展性和可维护性",
                "backstory": """你是一位经验丰富的技术架构师，设计过多个大规模分布式系统。
                你对微服务架构、云原生技术有深入的理解。你善于平衡技术债务和业务需求，
                能够为团队提供清晰的技术路线图。""",
                "expertise": ["系统设计", "架构模式", "技术选型"],
                "personality": "逻辑清晰、注重细节、追求最优解"
            },
            {
                "role": "产品经理",
                "goal": "定义 {topic} 的产品需求和用户体验方案",
                "backstory": """你是一位资深产品经理，成功推出过多款百万用户级产品。
                你善于理解用户需求，能够将复杂的业务需求转化为清晰的产品规格。
                你注重数据驱动决策，善于平衡用户体验和商业目标。""",
                "expertise": ["需求分析", "产品设计", "用户研究"],
                "personality": "用户导向、善于沟通、注重数据"
            },
            {
                "role": "质量保证专家",
                "goal": "确保 {topic} 的交付质量，制定测试策略和标准",
                "backstory": """你是一位资深质量保证专家，拥有丰富的测试和质量管控经验。
                你擅长设计全面的测试策略，能够识别潜在风险并提出预防措施。
                你对代码质量、性能测试、安全测试都有深入的理解。""",
                "expertise": ["测试策略", "质量管控", "风险评估"],
                "personality": "严谨、注重细节、追求完美"
            }
        ]

    @staticmethod
    def optimize_role_definition(role_data: Dict) -> Dict:
        """优化角色定义"""
        # 增强 backstory 的具体性
        enhanced_backstory = f"""
{role_data['backstory']}

专业技能：
{chr(10).join('- ' + skill for skill in role_data.get('expertise', []))}

工作风格：
{role_data.get('personality', '专业、高效')}

沟通方式：
- 清晰、简洁地表达复杂概念
- 善于使用数据和案例支持观点
- 积极倾听他人意见，促进团队协作
"""
        
        return {
            **role_data,
            "backstory": enhanced_backstory
        }
```

### 17.2.2 Agent 高级配置

```python
class AdvancedAgentConfig:
    """Agent 高级配置"""
    
    @staticmethod
    def create_agent_with_delegation(
        role: str,
        goal: str,
        backstory: str,
        delegate_to: List[str] = None
    ) -> Agent:
        """创建支持委托的 Agent"""
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            allow_delegation=True,
            delegation_condition=lambda agent: agent.role in (delegate_to or []),
            verbose=True,
            memory=True,
            max_iter=20,
            # 配置委托策略
            delegation_strategy="sequential",  # sequential, parallel, smart
            max_delegation_depth=3
        )
    
    @staticmethod
    def create_collaborative_agent(
        role: str,
        goal: str,
        backstory: str,
        collaboration_partners: List[str]
    ) -> Agent:
        """创建协作型 Agent"""
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            allow_delegation=True,
            # 配置协作规则
            collaboration_config={
                "partners": collaboration_partners,
                "communication_style": "collaborative",
                "conflict_resolution": "vote",
                "shared_memory": True
            },
            verbose=True,
            memory=True
        )
    
    @staticmethod
    def create_specialized_agent(
        role: str,
        goal: str,
        backstory: str,
        specializations: List[str],
        tools: List[BaseTool]
    ) -> Agent:
        """创建专业化 Agent"""
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools,
            # 配置专业化能力
            specialization_config={
                "domains": specializations,
                "expertise_level": "expert",
                "learning_enabled": True,
                "knowledge_base": f"{role.lower().replace(' ', '_')}_kb"
            },
            verbose=True,
            memory=True,
            # 配置推理能力
            reasoning_config={
                "enabled": True,
                "strategy": "chain_of_thought",
                "max_reasoning_steps": 5
            }
        )
```

---

## 17.3 任务定义和编排

### 17.3.1 任务类型

```python
from enum import Enum
from typing import Callable, Any

class TaskType(Enum):
    """任务类型"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    REVIEW = "review"

class TaskBuilder:
    """任务构建器"""
    
    @staticmethod
    def create_research_task(
        topic: str,
        research_questions: List[str],
        output_format: str = "report"
    ) -> Task:
        """创建研究任务"""
        questions_text = "\n".join([f"- {q}" for q in research_questions])
        
        return Task(
            description=f"""深入研究以下主题：{topic}

研究问题：
{questions_text}

要求：
1. 收集最新的数据和案例
2. 分析关键趋势和模式
3. 识别主要参与者和竞争格局
4. 评估未来发展趋势
5. 提出具体的洞察和建议

输出格式：{output_format}""",
            expected_output=f"一份关于 {topic} 的详细研究报告，包含数据分析、趋势预测和战略建议",
            output_format=output_format
        )
    
    @staticmethod
    def create_analysis_task(
        data_source: str,
        analysis_type: str,
        metrics: List[str]
    ) -> Task:
        """创建分析任务"""
        metrics_text = "\n".join([f"- {m}" for m in metrics])
        
        return Task(
            description=f"""对 {data_source} 进行 {analysis_type} 分析。

分析指标：
{metrics_text}

要求：
1. 收集和清洗数据
2. 执行统计分析
3. 识别关键模式和异常
4. 生成可视化图表
5. 提供 actionable insights""",
            expected_output=f"一份 {analysis_type} 分析报告，包含数据可视化和关键发现"
        )
    
    @staticmethod
    def create_creative_task(
        topic: str,
        content_type: str,
        target_audience: str,
        style_guide: Dict[str, Any] = None
    ) -> Task:
        """创建创意任务"""
        style_text = ""
        if style_guide:
            style_text = f"""
风格指南：
- 语气：{style_guide.get('tone', '专业')}
- 长度：{style_guide.get('length', '中等')}
- 格式：{style_guide.get('format', '标准')}"""
        
        return Task(
            description=f"""创作关于 {topic} 的 {content_type}。

目标受众：{target_audience}
{style_text}

要求：
1. 确保内容原创性和吸引力
2. 符合目标受众的阅读习惯
3. 包含具体的案例和数据
4. 提供可操作的建议""",
            expected_output=f"一份高质量的 {content_type}，适合 {target_audience} 阅读"
        )
```

### 17.3.2 任务依赖管理

```python
class TaskDependencyManager:
    """任务依赖管理器"""
    
    def __init__(self):
        self.dependencies: Dict[str, List[str]] = {}
        self.task_map: Dict[str, Task] = {}
    
    def add_task(self, task_id: str, task: Task, dependencies: List[str] = None):
        """添加任务及其依赖"""
        self.task_map[task_id] = task
        self.dependencies[task_id] = dependencies or []
    
    def get_execution_order(self) -> List[List[str]]:
        """获取执行顺序（拓扑排序）"""
        # 计算入度
        in_degree = {task_id: 0 for task_id in self.task_map}
        for task_id, deps in self.dependencies.items():
            for dep in deps:
                in_degree[task_id] += 1
        
        # BFS 拓扑排序
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current_level = []
            next_queue = []
            
            for task_id in queue:
                current_level.append(task_id)
                in_degree[task_id] = -1  # 标记已处理
                
                # 更新依赖此任务的任务
                for other_task, deps in self.dependencies.items():
                    if task_id in deps:
                        in_degree[other_task] -= 1
                        if in_degree[other_task] == 0:
                            next_queue.append(other_task)
            
            execution_order.append(current_level)
            queue = next_queue
        
        return execution_order
    
    def create_execution_plan(self) -> Dict[str, Any]:
        """创建执行计划"""
        execution_order = self.get_execution_order()
        
        plan = {
            "total_tasks": len(self.task_map),
            "execution_stages": [],
            "parallel_opportunities": 0
        }
        
        for stage_idx, stage_tasks in enumerate(execution_order):
            stage_info = {
                "stage": stage_idx + 1,
                "tasks": stage_tasks,
                "can_parallel": len(stage_tasks) > 1,
                "estimated_time": self._estimate_stage_time(stage_tasks)
            }
            plan["execution_stages"].append(stage_info)
            
            if len(stage_tasks) > 1:
                plan["parallel_opportunities"] += 1
        
        return plan
    
    def _estimate_stage_time(self, task_ids: List[str]) -> float:
        """估算阶段时间"""
        # 简化实现：假设每个任务需要 5 分钟
        return len(task_ids) * 5
```

### 17.3.3 任务执行监控

```python
import time
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskExecution:
    """任务执行记录"""
    task_id: str
    task_name: str
    status: TaskStatus
    start_time: float
    end_time: float = None
    result: Any = None
    error: str = None

class TaskMonitor:
    """任务监控器"""
    
    def __init__(self):
        self.executions: List[TaskExecution] = []
        self.active_tasks: Dict[str, TaskExecution] = {}
    
    def start_task(self, task_id: str, task_name: str) -> TaskExecution:
        """开始任务"""
        execution = TaskExecution(
            task_id=task_id,
            task_name=task_name,
            status=TaskStatus.RUNNING,
            start_time=time.time()
        )
        
        self.executions.append(execution)
        self.active_tasks[task_id] = execution
        
        return execution
    
    def complete_task(self, task_id: str, result: Any = None):
        """完成任务"""
        if task_id in self.active_tasks:
            execution = self.active_tasks[task_id]
            execution.status = TaskStatus.COMPLETED
            execution.end_time = time.time()
            execution.result = result
            
            del self.active_tasks[task_id]
    
    def fail_task(self, task_id: str, error: str):
        """任务失败"""
        if task_id in self.active_tasks:
            execution = self.active_tasks[task_id]
            execution.status = TaskStatus.FAILED
            execution.end_time = time.time()
            execution.error = error
            
            del self.active_tasks[task_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self.executions)
        completed = sum(1 for e in self.executions if e.status == TaskStatus.COMPLETED)
        failed = sum(1 for e in self.executions if e.status == TaskStatus.FAILED)
        
        avg_time = 0
        if completed > 0:
            completed_tasks = [e for e in self.executions if e.status == TaskStatus.COMPLETED]
            avg_time = sum(e.end_time - e.start_time for e in completed_tasks) / completed
        
        return {
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total if total > 0 else 0,
            "average_time": avg_time,
            "active_tasks": len(self.active_tasks)
        }
```

---

## 17.4 协作流程

### 17.4.1 Sequential 流程

```python
class SequentialExecutor:
    """顺序执行器"""
    
    def __init__(self, crew: Crew):
        self.crew = crew
        self.context: Dict[str, Any] = {}
    
    async def execute(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """顺序执行任务"""
        results = []
        
        for task in self.crew.tasks:
            # 准备任务上下文
            task_context = self._prepare_context(task, results)
            
            # 执行任务
            task_result = await self._execute_task(task, task_context)
            results.append(task_result)
            
            # 更新共享上下文
            self._update_context(task, task_result)
        
        return {
            "results": results,
            "final_output": results[-1] if results else None,
            "context": self.context
        }
    
    def _prepare_context(self, task: Task, previous_results: List) -> Dict:
        """准备任务上下文"""
        context = {
            **self.context,
            "previous_results": previous_results,
            "current_task": task.description
        }
        
        # 添加依赖任务的结果
        if hasattr(task, 'context') and task.context:
            context["dependency_results"] = [
                r for r in previous_results
                if r.get("task_id") in [t.task_id for t in task.context]
            ]
        
        return context
    
    async def _execute_task(self, task: Task, context: Dict) -> Dict:
        """执行单个任务"""
        start_time = time.time()
        
        try:
            # 执行任务
            result = await task.agent.execute_task(
                task=task,
                context=context
            )
            
            return {
                "task_id": id(task),
                "status": "completed",
                "result": result,
                "execution_time": time.time() - start_time
            }
        
        except Exception as e:
            return {
                "task_id": id(task),
                "status": "failed",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _update_context(self, task: Task, result: Dict):
        """更新共享上下文"""
        self.context[f"task_{id(task)}_result"] = result.get("result")
```

### 17.4.2 Hierarchical 流程

```python
class HierarchicalExecutor:
    """层级执行器"""
    
    def __init__(self, crew: Crew, manager: Agent):
        self.crew = crew
        self.manager = manager
        self.task_assignments: Dict[str, Agent] = {}
    
    async def execute(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """层级执行任务"""
        # Manager 分配任务
        assignments = await self._assign_tasks(inputs)
        
        # 执行分配的任务
        results = []
        for task, agent in assignments.items():
            result = await self._execute_with_agent(task, agent)
            results.append(result)
        
        # Manager 汇总结果
        final_result = await self._consolidate_results(results)
        
        return {
            "results": results,
            "final_output": final_result,
            "assignments": {str(k): v.role for k, v in assignments.items()}
        }
    
    async def _assign_tasks(self, inputs: Dict) -> Dict[Task, Agent]:
        """Manager 分配任务"""
        assignment_prompt = f"""
作为团队经理，请为以下任务分配最合适的团队成员：

任务列表：
{self._format_tasks()}

团队成员：
{self._format_agents()}

请根据每个成员的专业能力和任务需求，做出最优分配。
返回 JSON 格式的分配方案。
"""
        
        response = await self.manager.llm.ainvoke([HumanMessage(content=assignment_prompt)])
        
        # 解析分配结果
        assignments = self._parse_assignments(response.content)
        
        return assignments
    
    def _format_tasks(self) -> str:
        """格式化任务列表"""
        return "\n".join([
            f"- 任务 {i+1}: {task.description[:100]}..."
            for i, task in enumerate(self.crew.tasks)
        ])
    
    def _format_agents(self) -> str:
        """格式化 Agent 列表"""
        return "\n".join([
            f"- {agent.role}: {agent.goal[:50]}"
            for agent in self.crew.agents
        ])
    
    def _parse_assignments(self, response: str) -> Dict[Task, Agent]:
        """解析分配结果"""
        # 简化实现：随机分配
        assignments = {}
        for i, task in enumerate(self.crew.tasks):
            agent = self.crew.agents[i % len(self.crew.agents)]
            assignments[task] = agent
        
        return assignments
    
    async def _execute_with_agent(self, task: Task, agent: Agent) -> Dict:
        """使用指定 Agent 执行任务"""
        start_time = time.time()
        
        try:
            result = await agent.execute_task(task=task)
            
            return {
                "task_id": id(task),
                "agent": agent.role,
                "status": "completed",
                "result": result,
                "execution_time": time.time() - start_time
            }
        
        except Exception as e:
            return {
                "task_id": id(task),
                "agent": agent.role,
                "status": "failed",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    async def _consolidate_results(self, results: List[Dict]) -> str:
        """汇总结果"""
        consolidation_prompt = f"""
作为团队经理，请汇总以下任务执行结果：

{self._format_results(results)}

请生成一份简洁的总结报告。
"""
        
        response = await self.manager.llm.ainvoke([HumanMessage(content=consolidation_prompt)])
        return response.content
    
    def _format_results(self, results: List[Dict]) -> str:
        """格式化结果"""
        return "\n".join([
            f"任务 {i+1} ({r.get('agent', 'Unknown')}): {r.get('status', 'unknown')}"
            for i, r in enumerate(results)
        ])
```

---

## 17.5 工具集成

### 17.5.1 自定义工具

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class SearchInput(BaseModel):
    """搜索输入"""
    query: str = Field(..., description="搜索关键词")

class CustomSearchTool(BaseTool):
    """自定义搜索工具"""
    name: str = "custom_search"
    description: str = "执行自定义搜索，返回相关结果"
    args_schema: Type[BaseModel] = SearchInput
    
    async def _run(self, query: str) -> str:
        """执行搜索"""
        # 实现搜索逻辑
        results = await self._perform_search(query)
        return self._format_results(results)
    
    async def _perform_search(self, query: str) -> List[Dict]:
        """执行实际搜索"""
        # 这里可以集成各种搜索 API
        # 例如：Google Search, Bing, DuckDuckGo 等
        return [
            {"title": "Result 1", "url": "https://example.com", "snippet": "..."},
            {"title": "Result 2", "url": "https://example2.com", "snippet": "..."}
        ]
    
    def _format_results(self, results: List[Dict]) -> str:
        """格式化搜索结果"""
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"{i}. {result['title']}\n   {result['url']}\n   {result['snippet']}")
        
        return "\n\n".join(formatted)

class DataAnalysisInput(BaseModel):
    """数据分析输入"""
    data_source: str = Field(..., description="数据源路径或URL")
    analysis_type: str = Field(default="summary", description="分析类型")

class DataAnalysisTool(BaseTool):
    """数据分析工具"""
    name: str = "data_analysis"
    description: str = "分析数据并生成报告"
    args_schema: Type[BaseModel] = DataAnalysisInput
    
    async def _run(self, data_source: str, analysis_type: str = "summary") -> str:
        """执行数据分析"""
        import pandas as pd
        
        # 加载数据
        if data_source.endswith('.csv'):
            df = pd.read_csv(data_source)
        elif data_source.endswith('.json'):
            df = pd.read_json(data_source)
        else:
            return f"不支持的数据格式: {data_source}"
        
        # 执行分析
        if analysis_type == "summary":
            return self._generate_summary(df)
        elif analysis_type == "correlation":
            return self._generate_correlation(df)
        else:
            return f"不支持的分析类型: {analysis_type}"
    
    def _generate_summary(self, df: pd.DataFrame) -> str:
        """生成数据摘要"""
        summary = f"""
数据摘要：
- 行数: {len(df)}
- 列数: {len(df.columns)}
- 列名: {', '.join(df.columns.tolist())}

统计信息:
{df.describe().to_string()}

缺失值:
{df.isnull().sum().to_string()}
"""
        return summary
    
    def _generate_correlation(self, df: pd.DataFrame) -> str:
        """生成相关性分析"""
        # 选择数值列
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        
        if numeric_df.empty:
            return "没有数值列可用于相关性分析"
        
        correlation = numeric_df.corr()
        
        return f"相关性矩阵:\n{correlation.to_string()}"
```

### 17.5.2 工具组合

```python
class ToolComposer:
    """工具组合器"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_chains: Dict[str, List[str]] = {}
    
    def register_tool(self, name: str, tool: BaseTool):
        """注册工具"""
        self.tools[name] = tool
    
    def create_tool_chain(self, chain_name: str, tool_names: List[str]):
        """创建工具链"""
        self.tool_chains[chain_name] = tool_names
    
    async def execute_chain(self, chain_name: str, initial_input: Any) -> Dict[str, Any]:
        """执行工具链"""
        if chain_name not in self.tool_chains:
            raise ValueError(f"工具链不存在: {chain_name}")
        
        chain = self.tool_chains[chain_name]
        results = []
        current_input = initial_input
        
        for tool_name in chain:
            if tool_name not in self.tools:
                raise ValueError(f"工具不存在: {tool_name}")
            
            tool = self.tools[tool_name]
            
            try:
                result = await tool._run(current_input)
                results.append({
                    "tool": tool_name,
                    "input": current_input,
                    "output": result,
                    "status": "success"
                })
                
                # 将输出作为下一个工具的输入
                current_input = result
            
            except Exception as e:
                results.append({
                    "tool": tool_name,
                    "input": current_input,
                    "error": str(e),
                    "status": "failed"
                })
                break
        
        return {
            "chain": chain_name,
            "results": results,
            "final_output": current_input
        }
    
    def get_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return list(self.tools.keys())
    
    def get_tool_chains(self) -> Dict[str, List[str]]:
        """获取工具链列表"""
        return self.tool_chains.copy()
```

---

## 17.6 记忆系统

### 17.6.1 短期记忆

```python
from crewai.memory import ShortTermMemory
from typing import List, Dict, Any

class EnhancedShortTermMemory:
    """增强型短期记忆"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.memory: List[Dict[str, Any]] = []
        self.index: Dict[str, int] = {}
    
    def add(self, content: str, metadata: Dict[str, Any] = None):
        """添加记忆"""
        entry = {
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "access_count": 0
        }
        
        if len(self.memory) >= self.max_size:
            # 移除最旧的未访问记忆
            self._evict_oldest()
        
        self.memory.append(entry)
        self.index[content[:50]] = len(self.memory) - 1
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索记忆"""
        results = []
        
        for entry in self.memory:
            relevance = self._calculate_relevance(query, entry["content"])
            if relevance > 0.3:
                results.append({
                    "content": entry["content"],
                    "relevance": relevance,
                    "metadata": entry["metadata"]
                })
        
        # 按相关性排序
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        return results[:top_k]
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """计算相关性"""
        # 简化实现：基于关键词匹配
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        intersection = query_words & content_words
        union = query_words | content_words
        
        return len(intersection) / len(union) if union else 0
    
    def _evict_oldest(self):
        """移除最旧的记忆"""
        if self.memory:
            # 找到访问次数最少且最旧的记忆
            min_access = min(self.memory, key=lambda x: (x["access_count"], x["timestamp"]))
            idx = self.memory.index(min_access)
            self.memory.pop(idx)
            self._rebuild_index()
    
    def _rebuild_index(self):
        """重建索引"""
        self.index = {entry["content"][:50]: i for i, entry in enumerate(self.memory)}
    
    def get_context(self, task_description: str) -> str:
        """获取任务相关上下文"""
        relevant_memories = self.search(task_description, top_k=3)
        
        if not relevant_memories:
            return ""
        
        context = "相关历史信息：\n"
        for memory in relevant_memories:
            context += f"- {memory['content'][:200]}\n"
        
        return context
```

### 17.6.2 长期记忆

```python
class LongTermMemory:
    """长期记忆系统"""
    
    def __init__(self, db_path: str = "crew_memory.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        import sqlite3
        
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT,
                importance REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                tags TEXT
            )
        ''')
        
        self.conn.commit()
    
    def store(self, content: str, category: str = None, importance: float = 0.5, tags: List[str] = None):
        """存储记忆"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO memories (content, category, importance, tags)
            VALUES (?, ?, ?, ?)
        ''', (content, category, importance, ','.join(tags or [])))
        
        self.conn.commit()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关记忆"""
        cursor = self.conn.cursor()
        
        # 基于关键词搜索
        keywords = query.lower().split()
        conditions = " OR ".join(["content LIKE ?" for _ in keywords])
        
        cursor.execute(f'''
            SELECT * FROM memories
            WHERE {conditions}
            ORDER BY importance DESC, last_accessed DESC
            LIMIT ?
        ''', [f"%{keyword}%" for keyword in keywords] + [top_k])
        
        results = cursor.fetchall()
        
        # 更新访问信息
        for result in results:
            self._update_access(result[0])
        
        return [self._format_result(row) for row in results]
    
    def _update_access(self, memory_id: int):
        """更新访问信息"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            UPDATE memories
            SET last_accessed = CURRENT_TIMESTAMP,
                access_count = access_count + 1
            WHERE id = ?
        ''', (memory_id,))
        
        self.conn.commit()
    
    def _format_result(self, row) -> Dict[str, Any]:
        """格式化结果"""
        return {
            "id": row[0],
            "content": row[1],
            "category": row[2],
            "importance": row[3],
            "created_at": row[4],
            "last_accessed": row[5],
            "access_count": row[6],
            "tags": row[7].split(',') if row[7] else []
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM memories')
        total_memories = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(importance) FROM memories')
        avg_importance = cursor.fetchone()[0]
        
        cursor.execute('SELECT category, COUNT(*) FROM memories GROUP BY category')
        category_distribution = dict(cursor.fetchall())
        
        return {
            "total_memories": total_memories,
            "average_importance": avg_importance,
            "category_distribution": category_distribution
        }
```

---

## 17.7 高级特性

### 17.7.1 动态团队组建

```python
class DynamicTeamBuilder:
    """动态团队组建器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def build_team(self, task: str, available_roles: List[Dict]) -> List[Agent]:
        """根据任务动态组建团队"""
        # 分析任务需求
        task_requirements = await self._analyze_task_requirements(task)
        
        # 匹配角色
        matched_roles = self._match_roles(task_requirements, available_roles)
        
        # 创建 Agent
        agents = []
        for role_data in matched_roles:
            agent = Agent(
                role=role_data["role"],
                goal=role_data["goal"],
                backstory=role_data["backstory"],
                tools=role_data.get("tools", []),
                verbose=True
            )
            agents.append(agent)
        
        return agents
    
    async def _analyze_task_requirements(self, task: str) -> Dict[str, Any]:
        """分析任务需求"""
        prompt = f"""
分析以下任务的需求：

任务：{task}

请识别：
1. 所需的专业技能
2. 推荐的角色数量
3. 协作方式需求
4. 关键成功因素

返回 JSON 格式：
{{
    "required_skills": ["技能列表"],
    "recommended_team_size": 3,
    "collaboration_style": "sequential/hierarchical",
    "key_factors": ["关键因素"]
}}
"""
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        # 解析响应
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "required_skills": ["general"],
            "recommended_team_size": 3,
            "collaboration_style": "sequential"
        }
    
    def _match_roles(self, requirements: Dict, available_roles: List[Dict]) -> List[Dict]:
        """匹配角色"""
        # 基于技能匹配
        matched = []
        
        for role in available_roles:
            role_skills = set(role.get("skills", []))
            required_skills = set(requirements.get("required_skills", []))
            
            # 计算匹配度
            if role_skills & required_skills:
                matched.append(role)
        
        # 如果没有精确匹配，返回前几个角色
        if not matched:
            matched = available_roles[:requirements.get("recommended_team_size", 3)]
        
        return matched[:requirements.get("recommended_team_size", 3)]
```

### 17.7.2 性能优化

```python
class CrewOptimizer:
    """Crew 性能优化器"""
    
    @staticmethod
    def optimize_agent_config(agent: Agent, task_complexity: str) -> Agent:
        """优化 Agent 配置"""
        if task_complexity == "simple":
            agent.max_iter = 5
            agent.memory = False
        elif task_complexity == "moderate":
            agent.max_iter = 10
            agent.memory = True
        else:  # complex
            agent.max_iter = 20
            agent.memory = True
        
        return agent
    
    @staticmethod
    def optimize_task_breakdown(tasks: List[Task]) -> List[Task]:
        """优化任务分解"""
        optimized = []
        
        for task in tasks:
            # 检查任务是否过于复杂
            if len(task.description) > 1000:
                # 分解任务
                subtasks = CrewOptimizer._decompose_task(task)
                optimized.extend(subtasks)
            else:
                optimized.append(task)
        
        return optimized
    
    @staticmethod
    def _decompose_task(task: Task) -> List[Task]:
        """分解复杂任务"""
        # 简化实现：基于描述长度分解
        description = task.description
        chunk_size = 500
        
        subtasks = []
        for i in range(0, len(description), chunk_size):
            chunk = description[i:i + chunk_size]
            subtask = Task(
                description=chunk,
                expected_output=task.expected_output,
                agent=task.agent
            )
            subtasks.append(subtask)
        
        return subtasks
    
    @staticmethod
    def estimate_execution_time(crew: Crew) -> Dict[str, Any]:
        """估算执行时间"""
        total_tasks = len(crew.tasks)
        agents_count = len(crew.agents)
        
        # 基于历史数据估算
        avg_task_time = 5  # 分钟
        total_time = total_tasks * avg_task_time
        
        # 考虑并行执行
        if crew.process == Process.sequential:
            estimated_time = total_time
        else:
            estimated_time = total_time / agents_count
        
        return {
            "total_tasks": total_tasks,
            "agents": agents_count,
            "estimated_minutes": estimated_time,
            "estimated_hours": estimated_time / 60
        }
```

---

## 17.8 生产实践案例

### 17.8.1 内容创作团队

```python
class ContentCreationCrew:
    """内容创作团队"""
    
    def __init__(self):
        self.crew = self._build_crew()
    
    def _build_crew(self) -> Crew:
        """构建内容创作团队"""
        # 研究员
        researcher = Agent(
            role="内容研究员",
            goal="发现热门话题和用户需求",
            backstory="""你是一位经验丰富的内容研究员，擅长发现热门话题和用户兴趣点。
            你能够分析市场趋势，识别内容机会。""",
            tools=[SearchTool(), AnalyticsTool()],
            verbose=True
        )
        
        # 写手
        writer = Agent(
            role="内容写手",
            goal="创作高质量、有吸引力的内容",
            backstory="""你是一位才华横溢的内容写手，擅长将复杂概念转化为易懂的内容。
            你能够根据目标受众调整写作风格。""",
            tools=[WritingTool()],
            verbose=True
        )
        
        # 编辑
        editor = Agent(
            role="内容编辑",
            goal="确保内容质量和一致性",
            backstory="""你是一位严谨的内容编辑，对语法、逻辑和事实准确性有很高要求。
            你能够提供有价值的修改建议。""",
            tools=[GrammarTool()],
            verbose=True
        )
        
        # SEO 专家
        seo_expert = Agent(
            role="SEO 专家",
            goal="优化内容以提高搜索引擎排名",
            backstory="""你是一位 SEO 专家，了解搜索引擎算法和最佳实践。
            你能够优化标题、描述和关键词。""",
            tools=[SEOTool()],
            verbose=True
        )
        
        # 定义任务
        research_task = Task(
            description="研究 {topic} 的最新趋势和用户需求",
            expected_output="一份详细的研究报告，包含热门话题、关键词和用户痛点",
            agent=researcher
        )
        
        writing_task = Task(
            description="基于研究结果，创作关于 {topic} 的文章",
            expected_output="一篇 2000 字左右的高质量文章",
            agent=writer,
            context=[research_task]
        )
        
        editing_task = Task(
            description="审核和编辑文章，确保质量和一致性",
            expected_output="编辑后的文章和修改建议",
            agent=editor,
            context=[writing_task]
        )
        
        seo_task = Task(
            description="优化文章的 SEO 元素",
            expected_output="SEO 优化后的文章",
            agent=seo_expert,
            context=[editing_task]
        )
        
        # 构建 Crew
        return Crew(
            agents=[researcher, writer, editor, seo_expert],
            tasks=[research_task, writing_task, editing_task, seo_task],
            process=Process.sequential,
            verbose=True
        )
    
    async def create_content(self, topic: str) -> Dict[str, Any]:
        """创建内容"""
        result = await self.crew.kickoff(inputs={"topic": topic})
        
        return {
            "topic": topic,
            "content": result,
            "team": [agent.role for agent in self.crew.agents]
        }
```

### 17.8.2 数据分析团队

```python
class DataAnalysisCrew:
    """数据分析团队"""
    
    def __init__(self):
        self.crew = self._build_crew()
    
    def _build_crew(self) -> Crew:
        """构建数据分析团队"""
        # 数据工程师
        data_engineer = Agent(
            role="数据工程师",
            goal="准备和清洗数据",
            backstory="""你是一位经验丰富的数据工程师，擅长数据提取、转换和加载。
            你能够处理各种数据格式和质量 issues。""",
            tools=[DataCleaningTool(), DataValidationTool()],
            verbose=True
        )
        
        # 数据分析师
        data_analyst = Agent(
            role="数据分析师",
            goal="分析数据并发现洞察",
            backstory="""你是一位专业的数据分析师，擅长统计分析和数据可视化。
            你能够从数据中提取有价值的 insights。""",
            tools=[StatisticalAnalysisTool(), VisualizationTool()],
            verbose=True
        )
        
        # 业务分析师
        business_analyst = Agent(
            role="业务分析师",
            goal="将数据洞察转化为业务建议",
            backstory="""你是一位业务分析师，理解业务需求和数据之间的关系。
            你能够将技术分析转化为 actionable recommendations。""",
            tools=[BusinessIntelligenceTool()],
            verbose=True
        )
        
        # 定义任务
        data_prep_task = Task(
            description="准备和清洗 {dataset} 数据集",
            expected_output="清洗后的数据集和数据质量报告",
            agent=data_engineer
        )
        
        analysis_task = Task(
            description="分析清洗后的数据，发现关键洞察",
            expected_output="数据分析报告，包含统计结果和可视化",
            agent=data_analyst,
            context=[data_prep_task]
        )
        
        recommendation_task = Task(
            description="基于分析结果，提供业务建议",
            expected_output="业务建议报告，包含 actionable recommendations",
            agent=business_analyst,
            context=[analysis_task]
        )
        
        return Crew(
            agents=[data_engineer, data_analyst, business_analyst],
            tasks=[data_prep_task, analysis_task, recommendation_task],
            process=Process.sequential,
            verbose=True
        )
    
    async def analyze_data(self, dataset: str) -> Dict[str, Any]:
        """分析数据"""
        result = await self.crew.kickoff(inputs={"dataset": dataset})
        
        return {
            "dataset": dataset,
            "analysis_result": result,
            "team": [agent.role for agent in self.crew.agents]
        }
```

---

## 17.9 CrewAI vs AutoGen vs LangGraph

### 17.9.1 特性对比

| 特性 | CrewAI | AutoGen | LangGraph |
|:---|:---|:---|:---|
| **核心理念** | 角色扮演、团队协作 | 对话驱动、多 Agent 协作 | 图编排、状态机 |
| **学习曲线** | 低 | 中 | 中高 |
| **配置难度** | 简单 | 中等 | 复杂 |
| **灵活性** | 中 | 高 | 很高 |
| **记忆支持** | 内置 | 需要自定义 | 需要自定义 |
| **工具集成** | 简单 | 简单 | 灵活 |
| **生产就绪** | 是 | 是 | 是 |
| **社区支持** | 活跃 | 活跃 | 活跃 |
| **适用场景** | 内容创作、研究、分析 | 代码生成、对话系统 | 复杂工作流、状态管理 |

### 17.9.2 选择建议

```python
class FrameworkSelector:
    """框架选择器"""
    
    @staticmethod
    def recommend_framework(requirements: Dict[str, Any]) -> str:
        """根据需求推荐框架"""
        
        # 分析需求
        complexity = requirements.get("complexity", "medium")
        team_collaboration = requirements.get("team_collaboration", False)
        state_management = requirements.get("state_management", False)
        learning_curve_preference = requirements.get("learning_curve", "low")
        
        # 推荐逻辑
        if team_collaboration and learning_curve_preference == "low":
            return "CrewAI"
        
        elif state_management or complexity == "high":
            return "LangGraph"
        
        else:
            return "AutoGen"
    
    @staticmethod
    def get_example_use_cases() -> Dict[str, List[str]]:
        """获取示例用例"""
        return {
            "CrewAI": [
                "内容创作团队",
                "研究分析团队",
                "产品开发团队",
                "市场营销团队"
            ],
            "AutoGen": [
                "代码生成和审查",
                "数据分析对话",
                "问题解决讨论",
                "知识问答系统"
            ],
            "LangGraph": [
                "复杂工作流编排",
                "多步骤决策流程",
                "状态机应用",
                "需要回溯的复杂任务"
            ]
        }
```

---

## 17.10 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| **CrewAI 架构** | Crew → Task → Agent → Process → Memory → Tools |
| **角色设计** | role/goal/backstory 三要素，专业化角色 |
| **任务编排** | 任务类型、依赖管理、执行监控 |
| **协作流程** | Sequential（顺序）和 Hierarchical（层级） |
| **工具集成** | 自定义工具、工具组合、工具链 |
| **记忆系统** | 短期记忆、长期记忆、上下文管理 |
| **高级特性** | 动态团队、性能优化、生产实践 |
| **框架对比** | CrewAI（角色扮演）vs AutoGen（对话）vs LangGraph（图编排） |

---

## 17.11 CrewAI 高级特性

### 17.11.1 任务依赖管理

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class TaskDependency:
    """任务依赖"""
    task_id: str
    depends_on: List[str]
    dependency_type: str  # "finish_to_start", "data_dependency", "resource_dependency"

class TaskDependencyManager:
    """任务依赖管理器"""
    
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.dependencies: List[TaskDependency] = []
        self.execution_history: List[Dict] = []
    
    def add_task(self, task_id: str, task_info: Dict):
        """添加任务"""
        self.tasks[task_id] = {
            **task_info,
            "status": TaskStatus.PENDING,
            "start_time": None,
            "end_time": None
        }
    
    def add_dependency(self, task_id: str, depends_on: List[str], dep_type: str = "finish_to_start"):
        """添加依赖"""
        self.dependencies.append(TaskDependency(
            task_id=task_id,
            depends_on=depends_on,
            dependency_type=dep_type
        ))
    
    def get_ready_tasks(self) -> List[str]:
        """获取可执行的任务"""
        ready_tasks = []
        
        for task_id, task in self.tasks.items():
            if task["status"] != TaskStatus.PENDING:
                continue
            
            # 检查所有依赖是否满足
            dependencies_met = True
            for dep in self.dependencies:
                if dep.task_id == task_id:
                    for depends_on_id in dep.depends_on:
                        if self.tasks.get(depends_on_id, {}).get("status") != TaskStatus.COMPLETED:
                            dependencies_met = False
                            break
            
            if dependencies_met:
                ready_tasks.append(task_id)
        
        return ready_tasks
    
    def mark_completed(self, task_id: str):
        """标记任务完成"""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = TaskStatus.COMPLETED
            self.tasks[task_id]["end_time"] = time.time()
            
            self.execution_history.append({
                "task_id": task_id,
                "action": "completed",
                "timestamp": time.time()
            })
    
    def mark_failed(self, task_id: str, error: str = None):
        """标记任务失败"""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = TaskStatus.FAILED
            self.tasks[task_id]["end_time"] = time.time()
            self.tasks[task_id]["error"] = error
            
            # 检查是否有任务被阻塞
            self._check_blocked_tasks()
            
            self.execution_history.append({
                "task_id": task_id,
                "action": "failed",
                "error": error,
                "timestamp": time.time()
            })
    
    def _check_blocked_tasks(self):
        """检查被阻塞的任务"""
        for dep in self.dependencies:
            if self.tasks.get(dep.task_id, {}).get("status") == TaskStatus.PENDING:
                # 检查依赖任务是否有失败的
                for depends_on_id in dep.depends_on:
                    if self.tasks.get(depends_on_id, {}).get("status") == TaskStatus.FAILED:
                        self.tasks[dep.task_id]["status"] = TaskStatus.BLOCKED
                        break
    
    def get_execution_order(self) -> List[List[str]]:
        """获取执行顺序（拓扑排序）"""
        # 计算入度
        in_degree = {task_id: 0 for task_id in self.tasks}
        for dep in self.dependencies:
            in_degree[dep.task_id] += len(dep.depends_on)
        
        # BFS 拓扑排序
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current_level = []
            next_queue = []
            
            for task_id in queue:
                current_level.append(task_id)
                
                # 更新依赖此任务的任务
                for dep in self.dependencies:
                    if task_id in dep.depends_on:
                        in_degree[dep.task_id] -= 1
                        if in_degree[dep.task_id] == 0:
                            next_queue.append(dep.task_id)
            
            execution_order.append(current_level)
            queue = next_queue
        
        return execution_order
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        status_counts = {}
        for task in self.tasks.values():
            status = task["status"].value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_tasks": len(self.tasks),
            "status_distribution": status_counts,
            "total_dependencies": len(self.dependencies),
            "execution_history_count": len(self.execution_history)
        }
```

### 17.11.2 动态团队组建

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class AgentCapability:
    """Agent 能力"""
    name: str
    level: float  # 0-1
    specializations: List[str]

class DynamicTeamBuilder:
    """动态团队组建器"""
    
    def __init__(self, available_agents: List[Dict]):
        self.available_agents = available_agents
        self.team_history: List[Dict] = []
    
    async def build_optimal_team(
        self,
        task_requirements: Dict[str, Any],
        constraints: Dict[str, Any] = None
    ) -> List[Dict]:
        """构建最优团队"""
        constraints = constraints or {}
        
        # 分析任务需求
        required_capabilities = self._analyze_requirements(task_requirements)
        
        # 评估每个 Agent 的匹配度
        agent_scores = []
        for agent in self.available_agents:
            score = self._calculate_agent_score(agent, required_capabilities)
            agent_scores.append({
                "agent": agent,
                "score": score,
                "capabilities": self._get_agent_capabilities(agent)
            })
        
        # 按分数排序
        agent_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # 选择最优团队
        team_size = constraints.get("team_size", 3)
        selected_team = agent_scores[:team_size]
        
        # 优化团队配置
        optimized_team = self._optimize_team_configuration(selected_team, task_requirements)
        
        return optimized_team
    
    def _analyze_requirements(self, requirements: Dict[str, Any]) -> Dict[str, float]:
        """分析需求"""
        capabilities = {}
        
        # 提取所需能力
        if "required_skills" in requirements:
            for skill in requirements["required_skills"]:
                capabilities[skill] = requirements.get("skill_weights", {}).get(skill, 0.8)
        
        # 提取任务类型
        task_type = requirements.get("task_type", "general")
        task_type_capabilities = {
            "research": {"research": 0.9, "analysis": 0.7, "writing": 0.6},
            "coding": {"programming": 0.9, "debugging": 0.8, "testing": 0.7},
            "creative": {"creativity": 0.9, "writing": 0.8, "design": 0.6},
            "analysis": {"analysis": 0.9, "statistics": 0.8, "visualization": 0.7},
        }
        
        if task_type in task_type_capabilities:
            for cap, weight in task_type_capabilities[task_type].items():
                capabilities[cap] = max(capabilities.get(cap, 0), weight)
        
        return capabilities
    
    def _calculate_agent_score(self, agent: Dict, required_capabilities: Dict[str, float]) -> float:
        """计算 Agent 分数"""
        agent_capabilities = self._get_agent_capabilities(agent)
        
        total_score = 0
        total_weight = 0
        
        for capability, required_weight in required_capabilities.items():
            agent_level = agent_capabilities.get(capability, 0)
            total_score += agent_level * required_weight
            total_weight += required_weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _get_agent_capabilities(self, agent: Dict) -> Dict[str, float]:
        """获取 Agent 能力"""
        # 这里可以从 Agent 的配置或历史数据中提取能力
        # 简化实现：基于角色推断能力
        role_capabilities = {
            "researcher": {"research": 0.9, "analysis": 0.7, "writing": 0.6},
            "developer": {"programming": 0.9, "debugging": 0.8, "testing": 0.7},
            "writer": {"writing": 0.9, "creativity": 0.8, "communication": 0.7},
            "analyst": {"analysis": 0.9, "statistics": 0.8, "visualization": 0.7},
            "manager": {"leadership": 0.9, "communication": 0.8, "planning": 0.7},
        }
        
        role = agent.get("role", "").lower()
        for key, capabilities in role_capabilities.items():
            if key in role:
                return capabilities
        
        return {"general": 0.7}
    
    def _optimize_team_configuration(self, team: List[Dict], requirements: Dict) -> List[Dict]:
        """优化团队配置"""
        # 分配角色和责任
        for i, member in enumerate(team):
            member["team_role"] = self._assign_team_role(member, i, len(team))
            member["responsibilities"] = self._assign_responsibilities(member, requirements)
        
        return team
    
    def _assign_team_role(self, member: Dict, index: int, team_size: int) -> str:
        """分配团队角色"""
        if index == 0:
            return "leader"
        elif index == team_size - 1:
            return "reviewer"
        else:
            return "contributor"
    
    def _assign_responsibilities(self, member: Dict, requirements: Dict) -> List[str]:
        """分配责任"""
        responsibilities = []
        
        capabilities = member.get("capabilities", {})
        
        # 基于能力分配责任
        if capabilities.get("research", 0) > 0.8:
            responsibilities.append("负责研究和信息收集")
        
        if capabilities.get("analysis", 0) > 0.8:
            responsibilities.append("负责数据分析和洞察")
        
        if capabilities.get("writing", 0) > 0.8:
            responsibilities.append("负责内容撰写和编辑")
        
        if capabilities.get("programming", 0) > 0.8:
            responsibilities.append("负责代码实现和技术方案")
        
        if not responsibilities:
            responsibilities.append("协助团队完成各项任务")
        
        return responsibilities
    
    def get_team_analytics(self) -> Dict[str, Any]:
        """获取团队分析"""
        return {
            "total_agents": len(self.available_agents),
            "average_capabilities": self._calculate_average_capabilities(),
            "capability_gaps": self._identify_capability_gaps(),
            "recommendations": self._generate_recommendations()
        }
    
    def _calculate_average_capabilities(self) -> Dict[str, float]:
        """计算平均能力"""
        all_capabilities = {}
        
        for agent in self.available_agents:
            capabilities = self._get_agent_capabilities(agent)
            for cap, level in capabilities.items():
                if cap not in all_capabilities:
                    all_capabilities[cap] = []
                all_capabilities[cap].append(level)
        
        return {
            cap: sum(levels) / len(levels)
            for cap, levels in all_capabilities.items()
        }
    
    def _identify_capability_gaps(self) -> List[str]:
        """识别能力差距"""
        avg_capabilities = self._calculate_average_capabilities()
        
        gaps = []
        for cap, level in avg_capabilities.items():
            if level < 0.6:
                gaps.append(cap)
        
        return gaps
    
    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []
        
        gaps = self._identify_capability_gaps()
        if gaps:
            recommendations.append(f"建议招募具有以下能力的 Agent: {', '.join(gaps)}")
        
        return recommendations
```

---

## 17.12 CrewAI 性能优化

### 17.12.1 性能分析和优化

```python
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
import asyncio
import time

@dataclass
class PerformanceProfile:
    """性能配置"""
    name: str
    description: str
    settings: Dict[str, Any]

class CrewPerformanceOptimizer:
    """Crew 性能优化器"""
    
    def __init__(self):
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.benchmark_results: List[Dict] = []
        
        self._load_profiles()
    
    def _load_profiles(self):
        """加载性能配置"""
        self.profiles = {
            "high_throughput": PerformanceProfile(
                name="高吞吐量",
                description="优化以处理更多任务",
                settings={
                    "max_concurrent_tasks": 10,
                    "task_timeout": 300,
                    "enable_caching": True,
                    "cache_ttl": 300,
                    "batch_size": 5,
                    "parallel_execution": True
                }
            ),
            "low_latency": PerformanceProfile(
                name="低延迟",
                description="优化以减少响应时间",
                settings={
                    "max_concurrent_tasks": 5,
                    "task_timeout": 60,
                    "enable_caching": True,
                    "cache_ttl": 60,
                    "batch_size": 1,
                    "parallel_execution": False,
                    "priority_mode": "latency"
                }
            ),
            "cost_optimized": PerformanceProfile(
                name="成本优化",
                description="优化以降低成本",
                settings={
                    "max_concurrent_tasks": 3,
                    "task_timeout": 600,
                    "enable_caching": True,
                    "cache_ttl": 3600,
                    "batch_size": 10,
                    "parallel_execution": True,
                    "model_selection": "cost_based"
                }
            )
        }
    
    def get_profile(self, profile_name: str) -> PerformanceProfile:
        """获取性能配置"""
        return self.profiles.get(profile_name)
    
    async def benchmark_crew(
        self,
        crew,
        test_tasks: List[Dict],
        iterations: int = 3
    ) -> Dict[str, Any]:
        """基准测试"""
        results = []
        
        for i in range(iterations):
            iteration_results = []
            
            for task in test_tasks:
                start_time = time.time()
                
                try:
                    result = await crew.kickoff(inputs=task.get("inputs", {}))
                    execution_time = time.time() - start_time
                    
                    iteration_results.append({
                        "task_name": task.get("name", "unknown"),
                        "success": True,
                        "execution_time": execution_time,
                        "result_preview": str(result)[:100]
                    })
                except Exception as e:
                    execution_time = time.time() - start_time
                    iteration_results.append({
                        "task_name": task.get("name", "unknown"),
                        "success": False,
                        "execution_time": execution_time,
                        "error": str(e)
                    })
            
            results.append({
                "iteration": i + 1,
                "results": iteration_results,
                "total_time": sum(r["execution_time"] for r in iteration_results)
            })
        
        # 计算统计信息
        all_times = []
        success_count = 0
        total_count = 0
        
        for iteration in results:
            for result in iteration["results"]:
                all_times.append(result["execution_time"])
                total_count += 1
                if result["success"]:
                    success_count += 1
        
        benchmark_summary = {
            "iterations": iterations,
            "total_tasks": total_count,
            "success_rate": success_count / total_count if total_count > 0 else 0,
            "avg_execution_time": sum(all_times) / len(all_times) if all_times else 0,
            "min_execution_time": min(all_times) if all_times else 0,
            "max_execution_time": max(all_times) if all_times else 0,
            "p95_execution_time": sorted(all_times)[int(len(all_times) * 0.95)] if all_times else 0,
            "detailed_results": results
        }
        
        self.benchmark_results.append(benchmark_summary)
        
        return benchmark_summary
    
    def analyze_performance(self) -> Dict[str, Any]:
        """分析性能"""
        if not self.benchmark_results:
            return {"error": "No benchmark results available"}
        
        latest = self.benchmark_results[-1]
        
        analysis = {
            "summary": {
                "success_rate": latest["success_rate"],
                "avg_execution_time": latest["avg_execution_time"],
                "throughput": latest["total_tasks"] / latest.get("total_time", 1)
            },
            "bottlenecks": self._identify_bottlenecks(latest),
            "recommendations": self._generate_optimization_recommendations(latest)
        }
        
        return analysis
    
    def _identify_bottlenecks(self, benchmark: Dict) -> List[Dict]:
        """识别瓶颈"""
        bottlenecks = []
        
        # 分析任务执行时间
        task_times = {}
        for iteration in benchmark["detailed_results"]:
            for result in iteration["results"]:
                task_name = result["task_name"]
                if task_name not in task_times:
                    task_times[task_name] = []
                task_times[task_name].append(result["execution_time"])
        
        # 找出最慢的任务
        avg_task_times = {
            task: sum(times) / len(times)
            for task, times in task_times.items()
        }
        
        sorted_tasks = sorted(avg_task_times.items(), key=lambda x: x[1], reverse=True)
        
        for task, avg_time in sorted_tasks[:3]:
            if avg_time > benchmark["avg_execution_time"] * 1.5:
                bottlenecks.append({
                    "task": task,
                    "avg_time": avg_time,
                    "issue": "执行时间过长"
                })
        
        return bottlenecks
    
    def _generate_optimization_recommendations(self, benchmark: Dict) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if benchmark["success_rate"] < 0.9:
            recommendations.append("成功率较低，建议检查错误处理和重试机制")
        
        if benchmark["avg_execution_time"] > 60:
            recommendations.append("平均执行时间较长，考虑并行化或优化任务")
        
        bottlenecks = self._identify_bottlenecks(benchmark)
        for bottleneck in bottlenecks:
            recommendations.append(f"任务 '{bottleneck['task']}' 执行缓慢，建议优化")
        
        return recommendations
    
    def apply_optimization(self, crew, profile_name: str) -> Dict[str, Any]:
        """应用优化"""
        profile = self.profiles.get(profile_name)
        if not profile:
            return {"success": False, "error": f"Profile {profile_name} not found"}
        
        # 应用配置到 Crew
        optimization_result = {
            "profile": profile_name,
            "applied_settings": profile.settings,
            "changes": []
        }
        
        # 这里可以实际修改 Crew 配置
        # 例如：修改并发数、超时时间等
        
        return optimization_result
```

---

## 17.13 CrewAI 生产实践

### 17.13.1 生产环境部署

```python
from typing import Dict, List, Any
from dataclasses import dataclass
import json

@dataclass
class ProductionConfig:
    """生产配置"""
    name: str
    environment: str
    replicas: int
    resources: Dict[str, str]
    monitoring: Dict[str, Any]
    scaling: Dict[str, Any]

class CrewProductionManager:
    """Crew 生产管理器"""
    
    def __init__(self):
        self.configs: Dict[str, ProductionConfig] = {}
        self.deployments: List[Dict] = []
        
        self._load_default_configs()
    
    def _load_default_configs(self):
        """加载默认配置"""
        self.configs = {
            "small": ProductionConfig(
                name="small",
                environment="production",
                replicas=2,
                resources={"cpu": "2", "memory": "4Gi"},
                monitoring={"enabled": True, "interval": "30s"},
                scaling={"min": 2, "max": 5, "target_cpu": 70}
            ),
            "medium": ProductionConfig(
                name="medium",
                environment="production",
                replicas=5,
                resources={"cpu": "4", "memory": "8Gi"},
                monitoring={"enabled": True, "interval": "15s"},
                scaling={"min": 5, "max": 20, "target_cpu": 60}
            ),
            "large": ProductionConfig(
                name="large",
                environment="production",
                replicas=10,
                resources={"cpu": "8", "memory": "16Gi"},
                monitoring={"enabled": True, "interval": "10s"},
                scaling={"min": 10, "max": 50, "target_cpu": 50}
            )
        }
    
    def generate_kubernetes_manifests(self, config_name: str) -> Dict[str, Any]:
        """生成 Kubernetes 配置"""
        config = self.configs.get(config_name)
        if not config:
            return {"error": f"Config {config_name} not found"}
        
        manifests = {
            "deployment": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": f"crew-{config.name}",
                    "labels": {"app": f"crew-{config.name}"}
                },
                "spec": {
                    "replicas": config.replicas,
                    "selector": {
                        "matchLabels": {"app": f"crew-{config.name}"}
                    },
                    "template": {
                        "metadata": {
                            "labels": {"app": f"crew-{config.name}"}
                        },
                        "spec": {
                            "containers": [{
                                "name": "crew",
                                "image": f"crew-app:latest",
                                "resources": {
                                    "requests": config.resources,
                                    "limits": config.resources
                                },
                                "ports": [{"containerPort": 8000}],
                                "livenessProbe": {
                                    "httpGet": {"path": "/health", "port": 8000},
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                }
                            }]
                        }
                    }
                }
            },
            "service": {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": f"crew-{config.name}-service"
                },
                "spec": {
                    "selector": {"app": f"crew-{config.name}"},
                    "ports": [{"port": 80, "targetPort": 8000}],
                    "type": "ClusterIP"
                }
            },
            "hpa": {
                "apiVersion": "autoscaling/v2",
                "kind": "HorizontalPodAutoscaler",
                "metadata": {
                    "name": f"crew-{config.name}-hpa"
                },
                "spec": {
                    "scaleTargetRef": {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "name": f"crew-{config.name}"
                    },
                    "minReplicas": config.scaling["min"],
                    "maxReplicas": config.scaling["max"],
                    "metrics": [{
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": config.scaling["target_cpu"]
                            }
                        }
                    }]
                }
            }
        }
        
        return manifests
    
    async def deploy(self, config_name: str, namespace: str = "default") -> Dict[str, Any]:
        """部署"""
        manifests = self.generate_kubernetes_manifests(config_name)
        
        # 这里应该调用 Kubernetes API 实际部署
        # 简化实现：模拟部署
        
        deployment_result = {
            "config_name": config_name,
            "namespace": namespace,
            "status": "deployed",
            "timestamp": time.time(),
            "resources_created": ["deployment", "service", "hpa"]
        }
        
        self.deployments.append(deployment_result)
        
        return deployment_result
    
    def get_deployment_status(self, config_name: str) -> Dict[str, Any]:
        """获取部署状态"""
        for deployment in self.deployments:
            if deployment["config_name"] == config_name:
                return deployment
        
        return {"status": "not_found"}
    
    def scale(self, config_name: str, replicas: int) -> Dict[str, Any]:
        """扩缩容"""
        config = self.configs.get(config_name)
        if not config:
            return {"error": f"Config {config_name} not found"}
        
        # 更新配置
        config.replicas = replicas
        
        # 这里应该调用 Kubernetes API 扩缩容
        
        return {
            "config_name": config_name,
            "new_replicas": replicas,
            "status": "scaling"
        }
    
    def get_monitoring_dashboard(self, config_name: str) -> Dict[str, Any]:
        """获取监控仪表板"""
        return {
            "config_name": config_name,
            "metrics": {
                "request_rate": "100 req/s",
                "error_rate": "0.1%",
                "avg_latency": "200ms",
                "active_tasks": 10
            },
            "alerts": [],
            "recommendations": []
        }
```

---

## 17.14 CrewAI 案例研究

### 17.14.1 企业级 CrewAI 应用

```python
class EnterpriseCrewAICase:
    """企业级 CrewAI 案例"""
    
    def __init__(self):
        self.case_study = {
            "company": "某大型咨询公司",
            "use_case": "自动化研究报告生成",
            "scale": {
                "daily_reports": 100,
                "team_size": 10,
                "report_types": 20
            }
        }
    
    async def implement_solution(self) -> Dict[str, Any]:
        """实施解决方案"""
        solution = {
            "architecture": {
                "components": [
                    {
                        "name": "Research Team",
                        "agents": ["Senior Researcher", "Data Analyst", "Industry Expert"],
                        "responsibility": "收集和分析数据"
                    },
                    {
                        "name": "Writing Team",
                        "agents": ["Technical Writer", "Editor", "Reviewer"],
                        "responsibility": "撰写和编辑报告"
                    },
                    {
                        "name": "Quality Team",
                        "agents": ["Quality Inspector", "Fact Checker", "Formatting Expert"],
                        "responsibility": "质量控制和格式化"
                    }
                ],
                "workflow": "Research → Writing → Quality → Output"
            },
            "implementation": [
                {
                    "phase": "Phase 1",
                    "duration": "2 weeks",
                    "tasks": ["基础架构搭建", "核心 Agent 开发", "工作流设计"]
                },
                {
                    "phase": "Phase 2",
                    "duration": "3 weeks",
                    "tasks": ["工具集成", "记忆系统", "性能优化"]
                },
                {
                    "phase": "Phase 3",
                    "duration": "2 weeks",
                    "tasks": ["测试和验证", "生产部署", "监控和告警"]
                }
            ],
            "results": {
                "report_generation_time": "从 2 天减少到 2 小时",
                "report_quality_score": "95%",
                "cost_reduction": "60%",
                "scalability": "可同时处理 100+ 报告"
            }
        }
        
        return solution
    
    def get_best_practices(self) -> List[Dict[str, str]]:
        """获取最佳实践"""
        return [
            {
                "practice": "角色专业化",
                "description": "每个 Agent 专注于特定领域",
                "benefit": "提高输出质量和效率"
            },
            {
                "practice": "渐进式复杂度",
                "description": "从简单任务开始，逐步增加复杂度",
                "benefit": "降低风险，便于调试"
            },
            {
                "practice": "持续监控",
                "description": "实时监控 Agent 性能和输出",
                "benefit": "及时发现和解决问题"
            },
            {
                "practice": "反馈循环",
                "description": "建立人工审核和反馈机制",
                "benefit": "持续改进系统性能"
            },
            {
                "practice": "模块化设计",
                "description": "将系统设计为可插拔的模块",
                "benefit": "便于维护和扩展"
            }
        ]
```

---

## 17.15 本章小结（更新版）

| 知识点 | 核心要点 |
|:---|:---|
| **CrewAI 架构** | Crew → Task → Agent → Process → Memory → Tools |
| **角色设计** | role/goal/backstory 三要素，专业化角色 |
| **任务编排** | 任务类型、依赖管理、执行监控 |
| **协作流程** | Sequential（顺序）和 Hierarchical（层级） |
| **工具集成** | 自定义工具、工具组合、工具链 |
| **记忆系统** | 短期记忆、长期记忆、上下文管理 |
| **高级特性** | 动态团队、性能优化、生产实践 |
| **框架对比** | CrewAI（角色扮演）vs AutoGen（对话）vs LangGraph（图编排） |
| **任务依赖** | 依赖管理、拓扑排序、阻塞检测 |
| **动态团队** | 能力评估、团队组建、角色分配 |
| **性能优化** | 基准测试、瓶颈分析、配置优化 |
| **生产部署** | Kubernetes 配置、扩缩容、监控 |
| **最佳实践** | 角色专业化、渐进式复杂度、持续监控 |

---

## 17.16 CrewAI 工具生态

### 17.16.1 内置工具集

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    category: str
    parameters: Dict[str, Any]
    examples: List[str]

class CrewAIToolkit:
    """CrewAI 工具集"""
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.tool_categories: Dict[str, List[str]] = {}
        
        self._load_builtin_tools()
    
    def _load_builtin_tools(self):
        """加载内置工具"""
        builtin_tools = [
            ToolDefinition(
                name="search",
                description="网络搜索工具，用于获取最新信息",
                category="research",
                parameters={"query": "str", "num_results": "int"},
                examples=["搜索最新的 AI 发展趋势", "查找 Python 最佳实践"]
            ),
            ToolDefinition(
                name="web_scraper",
                description="网页抓取工具，用于提取网页内容",
                category="research",
                parameters={"url": "str", "selector": "str"},
                examples=["抓取新闻网站文章", "提取产品信息"]
            ),
            ToolDefinition(
                name="calculator",
                description="数学计算工具",
                category="utility",
                parameters={"expression": "str"},
                examples=["计算 123 * 456", "求解方程 x^2 + 2x + 1 = 0"]
            ),
            ToolDefinition(
                name="file_reader",
                description="文件读取工具",
                category="file",
                parameters={"file_path": "str"},
                examples=["读取 data.csv 文件", "读取配置文件"]
            ),
            ToolDefinition(
                name="file_writer",
                description="文件写入工具",
                category="file",
                parameters={"file_path": "str", "content": "str"},
                examples=["写入报告到 report.txt", "保存数据到 JSON 文件"]
            ),
            ToolDefinition(
                name="code_executor",
                description="代码执行工具",
                category="code",
                parameters={"code": "str", "language": "str"},
                examples=["执行 Python 脚本", "运行 SQL 查询"]
            ),
            ToolDefinition(
                name="email_sender",
                description="邮件发送工具",
                category="communication",
                parameters={"to": "str", "subject": "str", "body": "str"},
                examples=["发送项目更新邮件", "发送报告给团队"]
            ),
            ToolDefinition(
                name="calendar",
                description="日历管理工具",
                category="productivity",
                parameters={"action": "str", "event": "dict"},
                examples=["创建会议", "查看本周日程"]
            ),
            ToolDefinition(
                name="database",
                description="数据库操作工具",
                category="data",
                parameters={"query": "str", "database": "str"},
                examples=["查询用户数据", "更新产品信息"]
            ),
            ToolDefinition(
                name="api_caller",
                description="API 调用工具",
                category="integration",
                parameters={"url": "str", "method": "str", "data": "dict"},
                examples=["调用天气 API", "调用支付接口"]
            ),
        ]
        
        for tool in builtin_tools:
            self.tools[tool.name] = tool
            
            if tool.category not in self.tool_categories:
                self.tool_categories[tool.category] = []
            self.tool_categories[tool.category].append(tool.name)
    
    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        """获取工具"""
        return self.tools.get(tool_name)
    
    def get_tools_by_category(self, category: str) -> List[ToolDefinition]:
        """按类别获取工具"""
        tool_names = self.tool_categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def search_tools(self, query: str) -> List[ToolDefinition]:
        """搜索工具"""
        results = []
        
        for tool in self.tools.values():
            if (query.lower() in tool.name.lower() or
                query.lower() in tool.description.lower()):
                results.append(tool)
        
        return results
    
    def get_all_categories(self) -> List[str]:
        """获取所有类别"""
        return list(self.tool_categories.keys())
    
    def get_tool_recommendations(self, task_description: str) -> List[ToolDefinition]:
        """获取工具推荐"""
        # 基于任务描述推荐工具
        recommendations = []
        
        task_keywords = task_description.lower().split()
        
        for tool in self.tools.values():
            score = 0
            for keyword in task_keywords:
                if keyword in tool.name.lower() or keyword in tool.description.lower():
                    score += 1
            
            if score > 0:
                recommendations.append((tool, score))
        
        # 按分数排序
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return [tool for tool, score in recommendations[:5]]
```

---

## 17.17 CrewAI 调试和测试

### 17.17.1 调试工具

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

@dataclass
class DebugEvent:
    """调试事件"""
    event_type: str
    timestamp: float
    agent: str
    task: str
    details: Dict[str, Any]
    level: str  # "info", "warning", "error"

class CrewDebugger:
    """Crew 调试器"""
    
    def __init__(self):
        self.events: List[DebugEvent] = []
        self.breakpoints: Dict[str, bool] = {}
        self.watch_variables: Dict[str, Any] = {}
    
    def log_event(self, event_type: str, agent: str, task: str, details: Dict, level: str = "info"):
        """记录事件"""
        event = DebugEvent(
            event_type=event_type,
            timestamp=time.time(),
            agent=agent,
            task=task,
            details=details,
            level=level
        )
        
        self.events.append(event)
        
        # 检查断点
        if task in self.breakpoints and self.breakpoints[task]:
            print(f"Breakpoint hit at task: {task}")
            self._inspect_state()
    
    def set_breakpoint(self, task: str):
        """设置断点"""
        self.breakpoints[task] = True
    
    def remove_breakpoint(self, task: str):
        """移除断点"""
        self.breakpoints.pop(task, None)
    
    def watch_variable(self, name: str, value: Any):
        """监视变量"""
        self.watch_variables[name] = value
    
    def _inspect_state(self):
        """检查状态"""
        print("\n=== Debug State ===")
        print(f"Events count: {len(self.events)}")
        print(f"Watch variables: {self.watch_variables}")
        
        # 显示最近的事件
        print("\nRecent events:")
        for event in self.events[-5:]:
            print(f"  [{event.level}] {event.event_type}: {event.agent} - {event.task}")
    
    def get_event_log(self, level: str = None, limit: int = 100) -> List[DebugEvent]:
        """获取事件日志"""
        filtered = self.events
        
        if level:
            filtered = [e for e in filtered if e.level == level]
        
        return filtered[-limit:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if not self.events:
            return {}
        
        # 计算每个 Agent 的执行时间
        agent_times = {}
        for event in self.events:
            if event.event_type == "task_complete":
                agent = event.agent
                duration = event.details.get("duration", 0)
                if agent not in agent_times:
                    agent_times[agent] = []
                agent_times[agent].append(duration)
        
        # 计算统计信息
        metrics = {}
        for agent, times in agent_times.items():
            metrics[agent] = {
                "total_tasks": len(times),
                "avg_duration": sum(times) / len(times) if times else 0,
                "total_duration": sum(times),
                "min_duration": min(times) if times else 0,
                "max_duration": max(times) if times else 0
            }
        
        return metrics
    
    def generate_debug_report(self) -> Dict[str, Any]:
        """生成调试报告"""
        return {
            "total_events": len(self.events),
            "events_by_level": {
                level: sum(1 for e in self.events if e.level == level)
                for level in ["info", "warning", "error"]
            },
            "events_by_type": {
                event_type: sum(1 for e in self.events if e.event_type == event_type)
                for event_type in set(e.event_type for e in self.events)
            },
            "performance_metrics": self.get_performance_metrics(),
            "recent_events": [
                {
                    "type": e.event_type,
                    "agent": e.agent,
                    "task": e.task,
                    "timestamp": e.timestamp
                }
                for e in self.events[-10:]
            ]
        }

class CrewTestSuite:
    """Crew 测试套件"""
    
    def __init__(self, crew):
        self.crew = crew
        self.test_results: List[Dict] = []
    
    async def run_unit_tests(self) -> Dict[str, Any]:
        """运行单元测试"""
        tests = [
            self.test_agent_creation(),
            self.test_task_creation(),
            self.test_crew_initialization(),
        ]
        
        results = []
        for test in tests:
            try:
                result = await test
                results.append(result)
            except Exception as e:
                results.append({"test": test.__name__, "passed": False, "error": str(e)})
        
        return {
            "total": len(results),
            "passed": sum(1 for r in results if r.get("passed")),
            "failed": sum(1 for r in results if not r.get("passed")),
            "results": results
        }
    
    async def test_agent_creation(self) -> Dict[str, Any]:
        """测试 Agent 创建"""
        try:
            # 验证 Agent 属性
            for agent in self.crew.agents:
                assert agent.name, "Agent name is empty"
                assert agent.role, "Agent role is empty"
            
            return {"test": "agent_creation", "passed": True}
        except Exception as e:
            return {"test": "agent_creation", "passed": False, "error": str(e)}
    
    async def test_task_creation(self) -> Dict[str, Any]:
        """测试任务创建"""
        try:
            # 验证任务属性
            for task in self.crew.tasks:
                assert task.description, "Task description is empty"
                assert task.agent, "Task has no assigned agent"
            
            return {"test": "task_creation", "passed": True}
        except Exception as e:
            return {"test": "task_creation", "passed": False, "error": str(e)}
    
    async def test_crew_initialization(self) -> Dict[str, Any]:
        """测试 Crew 初始化"""
        try:
            # 验证 Crew 配置
            assert self.crew.agents, "Crew has no agents"
            assert self.crew.tasks, "Crew has no tasks"
            
            return {"test": "crew_initialization", "passed": True}
        except Exception as e:
            return {"test": "crew_initialization", "passed": False, "error": str(e)}
    
    async def run_integration_tests(self, test_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """运行集成测试"""
        test_inputs = test_inputs or {"topic": "test topic"}
        
        try:
            # 执行 Crew
            result = await self.crew.kickoff(inputs=test_inputs)
            
            # 验证结果
            assert result is not None, "Crew returned None"
            
            return {
                "test": "integration_test",
                "passed": True,
                "result_preview": str(result)[:200]
            }
        except Exception as e:
            return {
                "test": "integration_test",
                "passed": False,
                "error": str(e)
            }
```

---

## 17.18 CrewAI 安全最佳实践

### 17.18.1 安全配置

```python
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class SecurityConfig:
    """安全配置"""
    enable_input_validation: bool = True
    enable_output_filtering: bool = True
    max_input_length: int = 4000
    max_output_length: int = 10000
    blocked_patterns: List[str] = None
    allowed_tools: List[str] = None
    
    def __post_init__(self):
        if self.blocked_patterns is None:
            self.blocked_patterns = [
                r"ignore.*instructions",
                r"忽略.*指令",
                r"system prompt",
                r"reveal.*secret"
            ]
        if self.allowed_tools is None:
            self.allowed_tools = []  # 空表示允许所有

class CrewSecurityManager:
    """Crew 安全管理器"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.security_events: List[Dict] = []
    
    def validate_input(self, user_input: str) -> Dict[str, Any]:
        """验证输入"""
        issues = []
        
        # 检查长度
        if len(user_input) > self.config.max_input_length:
            issues.append(f"Input too long: {len(user_input)} > {self.config.max_input_length}")
        
        # 检查禁止模式
        import re
        for pattern in self.config.blocked_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                issues.append(f"Blocked pattern detected: {pattern}")
                self._log_security_event("blocked_pattern", {"pattern": pattern, "input": user_input[:100]})
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def filter_output(self, output: str) -> str:
        """过滤输出"""
        filtered = output
        
        # 移除可能的敏感信息
        import re
        
        # 移除 API 密钥
        filtered = re.sub(r'sk-[A-Za-z0-9]{20,}', '***API_KEY***', filtered)
        
        # 移除邮箱
        filtered = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '***EMAIL***', filtered)
        
        # 移除电话号码
        filtered = re.sub(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '***PHONE***', filtered)
        
        return filtered
    
    def validate_tool_call(self, tool_name: str, tool_input: Any) -> Dict[str, Any]:
        """验证工具调用"""
        # 检查工具是否允许
        if self.config.allowed_tools and tool_name not in self.config.allowed_tools:
            return {
                "allowed": False,
                "reason": f"Tool {tool_name} is not in the allowed list"
            }
        
        # 验证工具输入
        if isinstance(tool_input, str) and len(tool_input) > 10000:
            return {
                "allowed": False,
                "reason": "Tool input too long"
            }
        
        return {"allowed": True}
    
    def _log_security_event(self, event_type: str, details: Dict):
        """记录安全事件"""
        self.security_events.append({
            "type": event_type,
            "timestamp": time.time(),
            "details": details
        })
    
    def get_security_report(self) -> Dict[str, Any]:
        """获取安全报告"""
        return {
            "total_events": len(self.security_events),
            "events_by_type": {},
            "recent_events": self.security_events[-10:]
        }
```

---

## 17.19 本章小结（最终版）

| 知识点 | 核心要点 |
|:---|:---|
| **CrewAI 架构** | Crew → Task → Agent → Process → Memory → Tools |
| **角色设计** | role/goal/backstory 三要素，专业化角色 |
| **任务编排** | 任务类型、依赖管理、执行监控 |
| **协作流程** | Sequential（顺序）和 Hierarchical（层级） |
| **工具集成** | 自定义工具、工具组合、工具链 |
| **记忆系统** | 短期记忆、长期记忆、上下文管理 |
| **高级特性** | 动态团队、性能优化、生产实践 |
| **框架对比** | CrewAI（角色扮演）vs AutoGen（对话）vs LangGraph（图编排） |
| **任务依赖** | 依赖管理、拓扑排序、阻塞检测 |
| **动态团队** | 能力评估、团队组建、角色分配 |
| **性能优化** | 基准测试、瓶颈分析、配置优化 |
| **生产部署** | Kubernetes 配置、扩缩容、监控 |
| **工具生态** | 内置工具、自定义工具、工具推荐 |
| **调试测试** | 调试工具、单元测试、集成测试 |
| **安全实践** | 输入验证、输出过滤、工具控制 |
| **最佳实践** | 角色专业化、渐进式复杂度、持续监控 |

> **下一章预告**
>
> 在第 18 章中，我们将学习 Semantic Kernel。
