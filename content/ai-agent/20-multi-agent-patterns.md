---
title: "第20章：多智能体协作模式"
description: "深入掌握主从模式、对等协作、辩论模式、投票模式、流水线模式与黑板模式"
updated: "2025-06-15"
---

 # 第20章：多智能体协作模式

 > **学习目标**：
 > - 理解多智能体协作的核心设计理念
 > - 掌握主从模式的设计与实现
 > - 学会对等协作模式的架构
 > - 熟练掌握辩论模式与投票模式
 > - 实现流水线模式与黑板模式
 > - 制定模式选型指南

 下面的交互式可视化展示了不同的多 Agent 协作模式：

 <div data-component="MultiAgentPatterns"></div>

 ---

 ## 20.1 多智能体协作概述

### 20.1.1 为什么需要多智能体协作

单一 Agent 在处理复杂任务时往往面临以下挑战：

1. **能力有限**：单个 Agent 难以同时具备所有专业能力
2. **上下文窗口限制**：复杂任务需要处理大量信息，超出单个 Agent 的上下文窗口
3. **错误累积**：单点故障风险高，错误容易累积
4. **效率瓶颈**：串行处理无法充分利用并行能力

多智能体协作通过将任务分解并分配给多个专业 Agent，可以有效解决这些问题。

> **核心思想**：多个 Agent 像人类团队一样协作，各自发挥专长，共同完成复杂任务。

### 20.1.2 协作模式分类

```
多智能体协作模式分类：

┌─────────────────────────────────────────────────────────┐
│                  多智能体协作模式                          │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │            层级模式                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │   │
│  │  │   主从模式  │  │  层级模式   │  │ 委托模式│  │   │
│  │  └─────────────┘  └─────────────┘  └─────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │            对等模式                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │   │
│  │  │  对等协作   │  │  辩论模式   │  │ 投票模式│  │   │
│  │  └─────────────┘  └─────────────┘  └─────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │            流程模式                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │   │
│  │  │  流水线模式 │  │  黑板模式   │  │ 混合模式│  │   │
│  │  └─────────────┘  └─────────────┘  └─────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 20.1.3 核心概念速览

| 模式 | 核心思想 | 适用场景 |
|------|----------|----------|
| **主从模式** | 主 Agent 分配任务，从 Agent 执行 | 任务分配、调度 |
| **对等协作** | Agent 地位平等，共同决策 | 团队协作、头脑风暴 |
| **辩论模式** | 多个 Agent 辩论，达成共识 | 争议解决、方案评估 |
| **投票模式** | 多个 Agent 投票，多数决定 | 决策制定、评审 |
| **流水线模式** | Agent 按顺序处理任务 | 流程化处理、数据管道 |
| **黑板模式** | Agent 通过共享黑板协作 | 知识共享、协作编辑 |

---

## 20.2 主从模式（Master-Slave）

### 20.2.1 主从模式概述

主从模式是最经典的多智能体协作模式。主 Agent（Master）负责任务分配和协调，从 Agent（Slave）负责执行具体任务。

> **核心思想**：主 Agent 像项目经理，负责分配任务；从 Agent 像员工，负责执行任务。

### 20.2.2 基础主从模式实现

```python
from typing import List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Task:
    """任务定义"""
    id: str
    description: str
    assigned_to: str = None
    status: str = "pending"
    result: Any = None

class Agent(ABC):
    """Agent 基类"""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
    
    @abstractmethod
    def execute(self, task: Task) -> Any:
        """执行任务"""
        pass

class MasterAgent(Agent):
    """主 Agent"""
    
    def __init__(self, name: str):
        super().__init__(name, "master")
        self.slaves: List[Agent] = []
        self.tasks: List[Task] = []
    
    def add_slave(self, slave: Agent):
        """添加从 Agent"""
        self.slaves.append(slave)
    
    def assign_task(self, task: Task) -> Task:
        """分配任务"""
        # 选择合适的从 Agent
        slave = self._select_slave(task)
        task.assigned_to = slave.name
        task.status = "assigned"
        self.tasks.append(task)
        
        print(f"[{self.name}] 将任务 '{task.description}' 分配给 {slave.name}")
        return task
    
    def _select_slave(self, task: Task) -> Agent:
        """选择从 Agent（简单轮询）"""
        if not self.slaves:
            raise ValueError("没有可用的从 Agent")
        
        # 简单轮询选择
        task_count = {}
        for slave in self.slaves:
            task_count[slave.name] = sum(
                1 for t in self.tasks if t.assigned_to == slave.name
            )
        
        # 选择任务最少的从 Agent
        selected = min(self.slaves, key=lambda s: task_count.get(s.name, 0))
        return selected
    
    def execute(self, task: Task) -> Any:
        """执行主 Agent 的协调任务"""
        print(f"[{self.name}] 开始协调任务")
        return "协调完成"

class SlaveAgent(Agent):
    """从 Agent"""
    
    def __init__(self, name: str, specialization: str):
        super().__init__(name, "slave")
        self.specialization = specialization
    
    def execute(self, task: Task) -> Any:
        """执行任务"""
        print(f"[{self.name}] 执行任务: {task.description}")
        
        # 模拟执行
        result = f"{self.name} 完成了任务: {task.description}"
        task.status = "completed"
        task.result = result
        
        return result

# 使用示例
def master_slave_example():
    """主从模式示例"""
    # 创建主 Agent
    master = MasterAgent("项目经理")
    
    # 创建从 Agent
    coder = SlaveAgent("程序员", "编程")
    tester = SlaveAgent("测试员", "测试")
    writer = SlaveAgent("写手", "写作")
    
    # 添加从 Agent
    master.add_slave(coder)
    master.add_slave(tester)
    master.add_slave(writer)
    
    # 创建任务
    tasks = [
        Task(id="1", description="编写登录功能"),
        Task(id="2", description="测试登录功能"),
        Task(id="3", description="编写用户文档"),
    ]
    
    # 分配任务
    for task in tasks:
        master.assign_task(task)
    
    # 执行任务
    for task in tasks:
        slave = next(s for s in master.slaves if s.name == task.assigned_to)
        result = slave.execute(task)
        print(f"结果: {result}")

master_slave_example()
```

### 20.2.3 高级主从模式

```python
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio

class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

@dataclass
class AdvancedTask:
    """高级任务定义"""
    id: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    assigned_to: str = None
    status: str = "pending"
    result: Any = None
    metadata: Dict = field(default_factory=dict)

class AdvancedMasterAgent:
    """高级主 Agent"""
    
    def __init__(self, name: str):
        self.name = name
        self.slaves: Dict[str, 'AdvancedSlaveAgent'] = {}
        self.tasks: Dict[str, AdvancedTask] = {}
        self.task_queue: List[str] = []
        self.completed_tasks: List[str] = []
    
    def register_slave(self, slave: 'AdvancedSlaveAgent'):
        """注册从 Agent"""
        self.slaves[slave.name] = slave
        print(f"[{self.name}] 注册从 Agent: {slave.name} ({slave.specialization})")
    
    def submit_task(self, task: AdvancedTask) -> str:
        """提交任务"""
        self.tasks[task.id] = task
        self.task_queue.append(task.id)
        print(f"[{self.name}] 收到任务: {task.description} (优先级: {task.priority.name})")
        
        # 触发任务分配
        self._assign_pending_tasks()
        
        return task.id
    
    def _assign_pending_tasks(self):
        """分配待处理任务"""
        # 按优先级排序
        pending_tasks = [
            self.tasks[task_id]
            for task_id in self.task_queue
            if self.tasks[task_id].status == "pending"
        ]
        pending_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        
        for task in pending_tasks:
            # 检查依赖是否完成
            if all(dep in self.completed_tasks for dep in task.dependencies):
                # 分配任务
                slave = self._select_slave_for_task(task)
                if slave:
                    task.assigned_to = slave.name
                    task.status = "assigned"
                    self.task_queue.remove(task.id)
                    
                    # 异步执行
                    asyncio.create_task(self._execute_task(task))
    
    def _select_slave_for_task(self, task: AdvancedTask) -> 'AdvancedSlaveAgent':
        """为任务选择从 Agent"""
        # 根据任务类型选择专业 Agent
        available_slaves = [
            slave for slave in self.slaves.values()
            if slave.is_available()
        ]
        
        if not available_slaves:
            return None
        
        # 选择最合适的 Agent
        for slave in available_slaves:
            if task.description.lower() in slave.specialization.lower():
                return slave
        
        # 默认选择第一个可用的
        return available_slaves[0]
    
    async def _execute_task(self, task: AdvancedTask):
        """执行任务"""
        slave = self.slaves[task.assigned_to]
        
        print(f"[{self.name}] {slave.name} 开始执行: {task.description}")
        
        # 执行任务
        result = await slave.execute_task(task)
        
        # 更新状态
        task.status = "completed"
        task.result = result
        self.completed_tasks.append(task.id)
        
        print(f"[{self.name}] 任务完成: {task.description}")
        
        # 检查是否有新任务可以分配
        self._assign_pending_tasks()
    
    def get_status(self) -> Dict:
        """获取状态"""
        return {
            "total_tasks": len(self.tasks),
            "pending": sum(1 for t in self.tasks.values() if t.status == "pending"),
            "assigned": sum(1 for t in self.tasks.values() if t.status == "assigned"),
            "completed": sum(1 for t in self.tasks.values() if t.status == "completed"),
            "slaves": len(self.slaves),
        }

class AdvancedSlaveAgent:
    """高级从 Agent"""
    
    def __init__(self, name: str, specialization: str):
        self.name = name
        self.specialization = specialization
        self.current_task: AdvancedTask = None
        self.completed_count = 0
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return self.current_task is None
    
    async def execute_task(self, task: AdvancedTask) -> Any:
        """执行任务"""
        self.current_task = task
        
        # 模拟异步执行
        await asyncio.sleep(1)
        
        # 生成结果
        result = {
            "agent": self.name,
            "task": task.description,
            "output": f"完成: {task.description}",
        }
        
        self.current_task = None
        self.completed_count += 1
        
        return result

# 使用示例
async def advanced_master_slave_example():
    """高级主从模式示例"""
    # 创建主 Agent
    master = AdvancedMasterAgent("项目总监")
    
    # 创建从 Agent
    master.register_slave(AdvancedSlaveAgent("前端开发", "前端开发"))
    master.register_slave(AdvancedSlaveAgent("后端开发", "后端开发"))
    master.register_slave(AdvancedSlaveAgent("测试工程师", "测试"))
    
    # 提交任务
    task1 = AdvancedTask(
        id="1",
        description="设计用户界面",
        priority=TaskPriority.HIGH,
    )
    
    task2 = AdvancedTask(
        id="2",
        description="实现后端API",
        priority=TaskPriority.HIGH,
        dependencies=["1"],
    )
    
    task3 = AdvancedTask(
        id="3",
        description="编写测试用例",
        priority=TaskPriority.MEDIUM,
        dependencies=["2"],
    )
    
    master.submit_task(task1)
    master.submit_task(task2)
    master.submit_task(task3)
    
    # 等待所有任务完成
    await asyncio.sleep(5)
    
    # 获取状态
    print(f"\n状态: {master.get_status()}")

asyncio.run(advanced_master_slave_example())
```

---

## 20.3 对等协作模式（Peer-to-Peer）

### 20.3.1 对等协作模式概述

对等协作模式中，所有 Agent 地位平等，没有主从之分。它们通过直接通信和协作来完成任务。

> **核心思想**：Agent 之间像同事一样平等协作，通过沟通和协商达成目标。

### 20.3.2 对等协作模式实现

```python
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio

@dataclass
class Message:
    """消息定义"""
    sender: str
    receiver: str
    content: Any
    message_type: str = "info"
    timestamp: float = 0

class PeerAgent(ABC):
    """对等 Agent 基类"""
    
    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities
        self.inbox: List[Message] = []
        self.peers: Dict[str, 'PeerAgent'] = {}
    
    def register_peer(self, peer: 'PeerAgent'):
        """注册对等 Agent"""
        self.peers[peer.name] = peer
    
    def send_message(self, receiver_name: str, content: Any, message_type: str = "info"):
        """发送消息"""
        if receiver_name in self.peers:
            message = Message(
                sender=self.name,
                receiver=receiver_name,
                content=content,
                message_type=message_type,
            )
            self.peers[receiver_name].receive_message(message)
    
    def receive_message(self, message: Message):
        """接收消息"""
        self.inbox.append(message)
        self._process_message(message)
    
    @abstractmethod
    def _process_message(self, message: Message):
        """处理消息"""
        pass
    
    @abstractmethod
    def contribute(self, task: str) -> Any:
        """贡献能力"""
        pass

class ResearchPeer(PeerAgent):
    """研究对等 Agent"""
    
    def __init__(self, name: str):
        super().__init__(name, ["research", "analysis"])
        self.research_data: Dict[str, Any] = {}
    
    def _process_message(self, message: Message):
        """处理消息"""
        if message.message_type == "research_request":
            # 执行研究
            result = self.contribute(message.content)
            self.send_message(message.sender, result, "research_result")
    
    def contribute(self, task: str) -> Any:
        """贡献研究能力"""
        print(f"[{self.name}] 进行研究: {task}")
        
        # 模拟研究
        result = {
            "topic": task,
            "findings": [f"发现1: {task} 的关键点", f"发现2: {task} 的趋势"],
            "confidence": 0.85,
        }
        
        self.research_data[task] = result
        return result

class AnalysisPeer(PeerAgent):
    """分析对等 Agent"""
    
    def __init__(self, name: str):
        super().__init__(name, ["analysis", "synthesis"])
        self.analysis_results: Dict[str, Any] = {}
    
    def _process_message(self, message: Message):
        """处理消息"""
        if message.message_type == "analysis_request":
            result = self.contribute(message.content)
            self.send_message(message.sender, result, "analysis_result")
    
    def contribute(self, task: str) -> Any:
        """贡献分析能力"""
        print(f"[{self.name}] 进行分析: {task}")
        
        # 模拟分析
        result = {
            "topic": task,
            "insights": [f"洞察1: {task} 的影响", f"洞察2: {task} 的建议"],
            "score": 0.9,
        }
        
        self.analysis_results[task] = result
        return result

class WritingPeer(PeerAgent):
    """写作对等 Agent"""
    
    def __init__(self, name: str):
        super().__init__(name, ["writing", "editing"])
        self.documents: Dict[str, str] = {}
    
    def _process_message(self, message: Message):
        """处理消息"""
        if message.message_type == "writing_request":
            result = self.contribute(message.content)
            self.send_message(message.sender, result, "writing_result")
    
    def contribute(self, task: str) -> Any:
        """贡献写作能力"""
        print(f"[{self.name}] 进行写作: {task}")
        
        # 模拟写作
        document = f"关于 {task} 的文档:\n\n这是一个重要的主题..."
        
        self.documents[task] = document
        return {"document": document, "word_count": len(document)}

# 对等协作管理器
class PeerCollaborationManager:
    """对等协作管理器"""
    
    def __init__(self):
        self.peers: Dict[str, PeerAgent] = {}
        self.collaboration_history: List[Dict] = []
    
    def register_peer(self, peer: PeerAgent):
        """注册对等 Agent"""
        self.peers[peer.name] = peer
        
        # 注册所有其他 peer
        for existing_peer in self.peers.values():
            if existing_peer.name != peer.name:
                existing_peer.register_peer(peer)
                peer.register_peer(existing_peer)
    
    async def collaborate(self, task: str) -> Dict[str, Any]:
        """协作执行任务"""
        print(f"\n开始协作任务: {task}")
        
        results = {}
        
        # 并行执行所有 peer 的贡献
        tasks = []
        for peer in self.peers.values():
            tasks.append(self._peer_contribute(peer, task))
        
        contributions = await asyncio.gather(*tasks)
        
        # 收集结果
        for peer_name, result in contributions:
            results[peer_name] = result
        
        # 记录协作历史
        self.collaboration_history.append({
            "task": task,
            "results": results,
        })
        
        return results
    
    async def _peer_contribute(self, peer: PeerAgent, task: str):
        """单个 peer 贡献"""
        result = peer.contribute(task)
        return peer.name, result
    
    def get_collaboration_summary(self) -> Dict:
        """获取协作摘要"""
        return {
            "total_collaborations": len(self.collaboration_history),
            "peers": list(self.peers.keys()),
        }

# 使用示例
async def peer_collaboration_example():
    """对等协作示例"""
    # 创建管理器
    manager = PeerCollaborationManager()
    
    # 创建对等 Agent
    researcher = ResearchPeer("研究员")
    analyst = AnalysisPeer("分析师")
    writer = WritingPeer("写手")
    
    # 注册
    manager.register_peer(researcher)
    manager.register_peer(analyst)
    manager.register_peer(writer)
    
    # 协作执行任务
    results = await manager.collaborate("AI Agent 发展趋势")
    
    print(f"\n协作结果:")
    for peer_name, result in results.items():
        print(f"  {peer_name}: {result}")
    
    print(f"\n协作摘要: {manager.get_collaboration_summary()}")

asyncio.run(peer_collaboration_example())
```

---

## 20.4 辩论模式（Debate）

### 20.4.1 辩论模式概述

辩论模式中，多个 Agent 就同一问题进行辩论，通过论证和反驳来达成共识或最佳答案。

> **核心思想**：多个 Agent 像辩论赛一样，各自持有不同观点，通过辩论来探索问题的多个角度。

### 20.4.2 辩论模式实现

```python
from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio

class Stance(Enum):
    """立场"""
    FOR = "支持"
    AGAINST = "反对"
    NEUTRAL = "中立"

@dataclass
class Argument:
    """论点"""
    agent_name: str
    stance: Stance
    content: str
    strength: float  # 论点强度 0-1
    evidence: List[str] = field(default_factory=list)

@dataclass
class DebateRound:
    """辩论轮次"""
    round_number: int
    topic: str
    arguments: List[Argument]
    consensus: str = None

class DebateAgent:
    """辩论 Agent"""
    
    def __init__(self, name: str, stance: Stance, expertise: str):
        self.name = name
        self.stance = stance
        self.expertise = expertise
        self.arguments_history: List[Argument] = []
    
    def generate_argument(self, topic: str, previous_arguments: List[Argument]) -> Argument:
        """生成论点"""
        # 分析之前的论点
        opposing_args = [
            arg for arg in previous_arguments
            if arg.stance != self.stance
        ]
        
        # 生成反驳或支持
        if opposing_args:
            content = self._generate_rebuttal(topic, opposing_args)
        else:
            content = self._generate_initial_argument(topic)
        
        argument = Argument(
            agent_name=self.name,
            stance=self.stance,
            content=content,
            strength=self._calculate_strength(content),
            evidence=self._gather_evidence(topic),
        )
        
        self.arguments_history.append(argument)
        return argument
    
    def _generate_initial_argument(self, topic: str) -> str:
        """生成初始论点"""
        if self.stance == Stance.FOR:
            return f"作为{self.expertise}专家，我认为 {topic} 是正确的，因为..."
        elif self.stance == Stance.AGAINST:
            return f"作为{self.expertise}专家，我认为 {topic} 是错误的，因为..."
        else:
            return f"作为{self.expertise}专家，我认为 {topic} 需要更多分析..."
    
    def _generate_rebuttal(self, topic: str, opposing_args: List[Argument]) -> str:
        """生成反驳"""
        rebuttal_points = []
        for arg in opposing_args:
            rebuttal_points.append(f"针对 {arg.agent_name} 的观点，我认为...")
        
        return f"基于{self.expertise}的专业知识，{' '.join(rebuttal_points)}"
    
    def _calculate_strength(self, content: str) -> float:
        """计算论点强度"""
        # 简单启发式计算
        base_strength = 0.5
        if len(content) > 100:
            base_strength += 0.1
        if "因为" in content or "所以" in content:
            base_strength += 0.1
        return min(base_strength, 1.0)
    
    def _gather_evidence(self, topic: str) -> List[str]:
        """收集证据"""
        return [f"{self.expertise}领域的相关研究", "行业最佳实践"]

class DebateManager:
    """辩论管理器"""
    
    def __init__(self, topic: str, max_rounds: int = 3):
        self.topic = topic
        self.max_rounds = max_rounds
        self.agents: List[DebateAgent] = []
        self.rounds: List[DebateRound] = []
        self.consensus: str = None
    
    def add_agent(self, agent: DebateAgent):
        """添加辩论 Agent"""
        self.agents.append(agent)
    
    async def run_debate(self) -> str:
        """运行辩论"""
        print(f"\n{'='*50}")
        print(f"辩论主题: {self.topic}")
        print(f"{'='*50}")
        
        all_arguments = []
        
        for round_num in range(1, self.max_rounds + 1):
            print(f"\n--- 第 {round_num} 轮 ---")
            
            round_arguments = []
            
            # 每个 Agent 发言
            for agent in self.agents:
                argument = agent.generate_argument(self.topic, all_arguments)
                round_arguments.append(argument)
                all_arguments.append(argument)
                
                print(f"\n{agent.name} ({agent.stance.value}):")
                print(f"  论点: {argument.content}")
                print(f"  强度: {argument.strength:.2f}")
            
            # 记录轮次
            debate_round = DebateRound(
                round_number=round_num,
                topic=self.topic,
                arguments=round_arguments,
            )
            self.rounds.append(debate_round)
        
        # 达成共识
        self.consensus = self._reach_consensus(all_arguments)
        
        print(f"\n{'='*50}")
        print(f"辩论结论: {self.consensus}")
        print(f"{'='*50}")
        
        return self.consensus
    
    def _reach_consensus(self, arguments: List[Argument]) -> str:
        """达成共识"""
        # 统计各方观点
        for_count = sum(1 for arg in arguments if arg.stance == Stance.FOR)
        against_count = sum(1 for arg in arguments if arg.stance == Stance.AGAINST)
        
        # 计算平均强度
        for_avg = (
            sum(arg.strength for arg in arguments if arg.stance == Stance.FOR) / for_count
            if for_count > 0 else 0
        )
        against_avg = (
            sum(arg.strength for arg in arguments if arg.stance == Stance.AGAINST) / against_count
            if against_count > 0 else 0
        )
        
        if for_avg > against_avg:
            return f"支持方观点更强 (支持: {for_avg:.2f}, 反对: {against_avg:.2f})"
        elif against_avg > for_avg:
            return f"反对方观点更强 (支持: {for_avg:.2f}, 反对: {against_avg:.2f})"
        else:
            return f"双方势均力敌 (支持: {for_avg:.2f}, 反对: {against_avg:.2f})"

# 使用示例
async def debate_example():
    """辩论示例"""
    # 创建辩论管理器
    manager = DebateManager(
        topic="AI 是否会取代人类工作",
        max_rounds=3,
    )
    
    # 添加辩论 Agent
    manager.add_agent(DebateAgent("支持方专家A", Stance.FOR, "人工智能"))
    manager.add_agent(DebateAgent("反对方专家B", Stance.AGAINST, "劳动经济学"))
    manager.add_agent(DebateAgent("中立方专家C", Stance.NEUTRAL, "社会学"))
    
    # 运行辩论
    consensus = await manager.run_debate()
    
    print(f"\n最终结论: {consensus}")

asyncio.run(debate_example())
```

---

## 20.5 投票模式（Voting）

### 20.5.1 投票模式概述

投票模式中，多个 Agent 对同一问题进行独立评估，然后通过投票来达成最终决策。

> **核心思想**：多个 Agent 像评审团一样，独立评分后通过投票或加权平均来做出决策。

### 20.5.2 投票模式实现

```python
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio

class VoteChoice(Enum):
    """投票选项"""
    APPROVE = "批准"
    REJECT = "拒绝"
    ABSTAIN = "弃权"

@dataclass
class Vote:
    """投票"""
    agent_name: str
    choice: VoteChoice
    confidence: float  # 置信度 0-1
    reason: str = ""

@dataclass
class VotingProposal:
    """投票提案"""
    id: str
    title: str
    description: str
    votes: List[Vote] = field(default_factory=list)
    result: str = None
    passed: bool = None

class VotingAgent:
    """投票 Agent"""
    
    def __init__(self, name: str, expertise: str, weight: float = 1.0):
        self.name = name
        self.expertise = expertise
        self.weight = weight  # 投票权重
        self.voting_history: List[Vote] = []
    
    def evaluate_proposal(self, proposal: VotingProposal) -> Vote:
        """评估提案"""
        # 根据专业背景进行评估
        confidence = self._calculate_confidence(proposal)
        choice = self._make_choice(proposal, confidence)
        reason = self._generate_reason(proposal, choice)
        
        vote = Vote(
            agent_name=self.name,
            choice=choice,
            confidence=confidence,
            reason=reason,
        )
        
        self.voting_history.append(vote)
        return vote
    
    def _calculate_confidence(self, proposal: VotingProposal) -> float:
        """计算置信度"""
        # 简单启发式计算
        base_confidence = 0.7
        
        # 根据提案描述长度调整
        if len(proposal.description) > 100:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _make_choice(self, proposal: VotingProposal, confidence: float) -> VoteChoice:
        """做出选择"""
        # 简单逻辑：高置信度批准，低置信度弃权
        if confidence > 0.8:
            return VoteChoice.APPROVE
        elif confidence < 0.5:
            return VoteChoice.ABSTAIN
        else:
            # 随机选择（实际应基于更复杂的逻辑）
            import random
            return random.choice([VoteChoice.APPROVE, VoteChoice.REJECT])
    
    def _generate_reason(self, proposal: VotingProposal, choice: VoteChoice) -> str:
        """生成理由"""
        if choice == VoteChoice.APPROVE:
            return f"作为{self.expertise}专家，我认为此提案符合最佳实践"
        elif choice == VoteChoice.REJECT:
            return f"作为{self.expertise}专家，我认为此提案存在风险"
        else:
            return f"作为{self.expertise}专家，我需要更多信息来做出决定"

class VotingManager:
    """投票管理器"""
    
    def __init__(self, quorum: int = None):
        self.agents: List[VotingAgent] = []
        self.proposals: List[VotingProposal] = []
        self.quorum = quorum  # 法定人数
    
    def add_agent(self, agent: VotingAgent):
        """添加投票 Agent"""
        self.agents.append(agent)
    
    def submit_proposal(self, proposal: VotingProposal):
        """提交提案"""
        self.proposals.append(proposal)
        print(f"\n提案提交: {proposal.title}")
        print(f"描述: {proposal.description}")
    
    async def conduct_voting(self, proposal: VotingProposal) -> VotingProposal:
        """进行投票"""
        print(f"\n{'='*50}")
        print(f"开始投票: {proposal.title}")
        print(f"{'='*50}")
        
        # 收集投票
        votes = []
        for agent in self.agents:
            vote = agent.evaluate_proposal(proposal)
            votes.append(vote)
            proposal.votes.append(vote)
            
            print(f"\n{agent.name} ({agent.expertise}):")
            print(f"  选择: {vote.choice.value}")
            print(f"  置信度: {vote.confidence:.2f}")
            print(f"  权重: {agent.weight}")
            print(f"  理由: {vote.reason}")
        
        # 计算结果
        result = self._calculate_result(proposal)
        proposal.result = result
        proposal.passed = self._check_passed(proposal)
        
        print(f"\n{'='*50}")
        print(f"投票结果: {result}")
        print(f"是否通过: {'是' if proposal.passed else '否'}")
        print(f"{'='*50}")
        
        return proposal
    
    def _calculate_result(self, proposal: VotingProposal) -> str:
        """计算投票结果"""
        # 加权投票
        approve_weight = 0
        reject_weight = 0
        abstain_weight = 0
        
        for vote, agent in zip(proposal.votes, self.agents):
            weight = agent.weight * vote.confidence
            
            if vote.choice == VoteChoice.APPROVE:
                approve_weight += weight
            elif vote.choice == VoteChoice.REJECT:
                reject_weight += weight
            else:
                abstain_weight += weight
        
        total_weight = approve_weight + reject_weight + abstain_weight
        
        if total_weight == 0:
            return "无效投票"
        
        approve_ratio = approve_weight / total_weight
        reject_ratio = reject_weight / total_weight
        
        return f"批准: {approve_ratio:.1%}, 拒绝: {reject_ratio:.1%}"
    
    def _check_passed(self, proposal: VotingProposal) -> bool:
        """检查是否通过"""
        # 检查法定人数
        if self.quorum and len(proposal.votes) < self.quorum:
            return False
        
        # 计算加权票数
        approve_count = sum(
            agent.weight * vote.confidence
            for vote, agent in zip(proposal.votes, self.agents)
            if vote.choice == VoteChoice.APPROVE
        )
        
        total_weight = sum(agent.weight for agent in self.agents)
        
        # 简单多数通过
        return approve_count > total_weight / 2

# 使用示例
async def voting_example():
    """投票示例"""
    # 创建投票管理器
    manager = VotingManager(quorum=3)
    
    # 添加投票 Agent
    manager.add_agent(VotingAgent("技术专家", "技术架构", weight=1.5))
    manager.add_agent(VotingAgent("业务专家", "业务需求", weight=1.2))
    manager.add_agent(VotingAgent("安全专家", "安全合规", weight=1.3))
    manager.add_agent(VotingAgent("用户体验", "用户体验", weight=1.0))
    
    # 提交提案
    proposal = VotingProposal(
        id="P001",
        title="采用新的微服务架构",
        description="提议将现有单体应用重构为微服务架构，以提高可扩展性和维护性。",
    )
    manager.submit_proposal(proposal)
    
    # 进行投票
    result = await manager.conduct_voting(proposal)
    
    print(f"\n最终结果: {result.result}")
    print(f"是否通过: {result.passed}")

asyncio.run(voting_example())
```

---

## 20.6 流水线模式（Pipeline）

### 20.6.1 流水线模式概述

流水线模式中，任务按照预定义的顺序在多个 Agent 之间传递，每个 Agent 负责处理任务的一个阶段。

> **核心思想**：任务像工厂流水线一样，经过多个工位（Agent）依次处理。

### 20.6.2 流水线模式实现

```python
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio

@dataclass
class PipelineTask:
    """流水线任务"""
    id: str
    data: Any
    current_stage: int = 0
    stages_completed: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"

class PipelineStage(ABC):
    """流水线阶段基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def process(self, task: PipelineTask) -> PipelineTask:
        """处理任务"""
        pass

class DataCollectionStage(PipelineStage):
    """数据收集阶段"""
    
    def __init__(self):
        super().__init__("数据收集", "收集原始数据")
    
    async def process(self, task: PipelineTask) -> PipelineTask:
        """处理数据收集"""
        print(f"[{self.name}] 收集数据: {task.data}")
        
        # 模拟数据收集
        collected_data = {
            "raw_data": task.data,
            "source": "api",
            "timestamp": "2024-01-01",
        }
        
        task.results[self.name] = collected_data
        task.stages_completed.append(self.name)
        task.current_stage += 1
        
        return task

class DataProcessingStage(PipelineStage):
    """数据处理阶段"""
    
    def __init__(self):
        super().__init__("数据处理", "清洗和转换数据")
    
    async def process(self, task: PipelineTask) -> PipelineTask:
        """处理数据"""
        print(f"[{self.name}] 处理数据")
        
        # 获取上一阶段的结果
        prev_data = task.results.get("数据收集", {})
        
        # 模拟数据处理
        processed_data = {
            "cleaned_data": prev_data.get("raw_data", []),
            "transformations": ["去除空值", "标准化", "特征提取"],
            "quality_score": 0.95,
        }
        
        task.results[self.name] = processed_data
        task.stages_completed.append(self.name)
        task.current_stage += 1
        
        return task

class AnalysisStage(PipelineStage):
    """分析阶段"""
    
    def __init__(self):
        super().__init__("数据分析", "分析处理后的数据")
    
    async def process(self, task: PipelineTask) -> PipelineTask:
        """分析数据"""
        print(f"[{self.name}] 分析数据")
        
        # 获取上一阶段的结果
        prev_data = task.results.get("数据处理", {})
        
        # 模拟分析
        analysis_result = {
            "insights": ["趋势分析", "异常检测", "模式识别"],
            "confidence": 0.88,
            "recommendations": ["建议1", "建议2"],
        }
        
        task.results[self.name] = analysis_result
        task.stages_completed.append(self.name)
        task.current_stage += 1
        
        return task

class ReportGenerationStage(PipelineStage):
    """报告生成阶段"""
    
    def __init__(self):
        super().__init__("报告生成", "生成最终报告")
    
    async def process(self, task: PipelineTask) -> PipelineTask:
        """生成报告"""
        print(f"[{self.name}] 生成报告")
        
        # 获取所有阶段的结果
        all_results = task.results
        
        # 模拟报告生成
        report = {
            "title": "数据分析报告",
            "summary": "基于数据的分析结果",
            "sections": [
                {"title": "数据概览", "content": "数据收集和处理完成"},
                {"title": "分析结果", "content": "发现重要趋势"},
                {"title": "建议", "content": "建议采取的行动"},
            ],
            "metadata": {
                "stages_completed": task.stages_completed,
                "total_stages": len(task.stages_completed),
            }
        }
        
        task.results[self.name] = report
        task.stages_completed.append(self.name)
        task.current_stage += 1
        task.status = "completed"
        
        return task

class Pipeline:
    """流水线"""
    
    def __init__(self, name: str):
        self.name = name
        self.stages: List[PipelineStage] = []
        self.tasks: List[PipelineTask] = []
    
    def add_stage(self, stage: PipelineStage):
        """添加阶段"""
        self.stages.append(stage)
        print(f"添加阶段: {stage.name} - {stage.description}")
    
    async def execute(self, task: PipelineTask) -> PipelineTask:
        """执行流水线"""
        print(f"\n{'='*50}")
        print(f"流水线 [{self.name}] 开始执行")
        print(f"任务: {task.id}")
        print(f"{'='*50}")
        
        self.tasks.append(task)
        
        # 依次执行每个阶段
        for stage in self.stages:
            print(f"\n--- 执行阶段: {stage.name} ---")
            task = await stage.process(task)
            print(f"阶段完成: {stage.name}")
        
        print(f"\n{'='*50}")
        print(f"流水线 [{self.name}] 执行完成")
        print(f"已完成阶段: {task.stages_completed}")
        print(f"{'='*50}")
        
        return task
    
    def get_status(self) -> Dict:
        """获取状态"""
        return {
            "pipeline_name": self.name,
            "total_stages": len(self.stages),
            "stages": [stage.name for stage in self.stages],
            "total_tasks": len(self.tasks),
            "completed_tasks": sum(1 for t in self.tasks if t.status == "completed"),
        }

# 使用示例
async def pipeline_example():
    """流水线示例"""
    # 创建流水线
    pipeline = Pipeline("数据分析流水线")
    
    # 添加阶段
    pipeline.add_stage(DataCollectionStage())
    pipeline.add_stage(DataProcessingStage())
    pipeline.add_stage(AnalysisStage())
    pipeline.add_stage(ReportGenerationStage())
    
    # 创建任务
    task = PipelineTask(
        id="T001",
        data=["数据点1", "数据点2", "数据点3"],
    )
    
    # 执行流水线
    result = await pipeline.execute(task)
    
    print(f"\n最终结果:")
    print(f"报告: {result.results.get('报告生成', {}).get('title', 'N/A')}")
    print(f"流水线状态: {pipeline.get_status()}")

asyncio.run(pipeline_example())
```

---

## 20.7 黑板模式（Blackboard）

### 20.7.1 黑板模式概述

黑板模式中，多个 Agent 通过共享的"黑板"（知识库）进行协作。Agent 可以读取和写入黑板，通过黑板来共享信息和协调工作。

> **核心思想**：Agent 之间通过共享的知识空间（黑板）进行间接通信和协作。

### 20.7.2 黑板模式实现

```python
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import threading

@dataclass
class BlackboardEntry:
    """黑板条目"""
    key: str
    value: Any
    author: str
    timestamp: float = 0
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)

class Blackboard:
    """黑板"""
    
    def __init__(self):
        self.entries: Dict[str, BlackboardEntry] = {}
        self.lock = threading.Lock()
        self.observers: List['BlackboardAgent'] = []
    
    def write(self, key: str, value: Any, author: str, confidence: float = 1.0):
        """写入黑板"""
        with self.lock:
            entry = BlackboardEntry(
                key=key,
                value=value,
                author=author,
                confidence=confidence,
            )
            self.entries[key] = entry
            
            # 通知观察者
            self._notify_observers(key, entry)
            
            print(f"[黑板] {author} 写入: {key} = {value}")
    
    def read(self, key: str) -> Optional[BlackboardEntry]:
        """读取黑板"""
        with self.lock:
            return self.entries.get(key)
    
    def read_all(self) -> Dict[str, BlackboardEntry]:
        """读取所有条目"""
        with self.lock:
            return self.entries.copy()
    
    def search(self, query: str) -> List[BlackboardEntry]:
        """搜索黑板"""
        with self.lock:
            results = []
            for key, entry in self.entries.items():
                if query.lower() in key.lower() or query.lower() in str(entry.value).lower():
                    results.append(entry)
            return results
    
    def register_observer(self, agent: 'BlackboardAgent'):
        """注册观察者"""
        self.observers.append(agent)
    
    def _notify_observers(self, key: str, entry: BlackboardEntry):
        """通知观察者"""
        for observer in self.observers:
            observer.on_blackboard_update(key, entry)

class BlackboardAgent(ABC):
    """黑板 Agent 基类"""
    
    def __init__(self, name: str, blackboard: Blackboard):
        self.name = name
        self.blackboard = blackboard
        self.blackboard.register_observer(self)
    
    @abstractmethod
    def on_blackboard_update(self, key: str, entry: BlackboardEntry):
        """黑板更新回调"""
        pass
    
    @abstractmethod
    async def contribute(self):
        """贡献知识"""
        pass

class DataGathererAgent(BlackboardAgent):
    """数据收集 Agent"""
    
    def __init__(self, name: str, blackboard: Blackboard):
        super().__init__(name, blackboard)
        self.data_sources = ["API", "数据库", "文件"]
    
    def on_blackboard_update(self, key: str, entry: BlackboardEntry):
        """黑板更新回调"""
        if key.startswith("request_"):
            print(f"[{self.name}] 收到新请求: {entry.value}")
    
    async def contribute(self):
        """贡献数据"""
        # 收集数据
        data = {
            "source": "API",
            "records": 1000,
            "quality": 0.95,
        }
        
        # 写入黑板
        self.blackboard.write(
            key="collected_data",
            value=data,
            author=self.name,
            confidence=0.9,
        )

class AnalystAgent(BlackboardAgent):
    """分析师 Agent"""
    
    def __init__(self, name: str, blackboard: Blackboard):
        super().__init__(name, blackboard)
    
    def on_blackboard_update(self, key: str, entry: BlackboardEntry):
        """黑板更新回调"""
        if key == "collected_data":
            print(f"[{self.name}] 检测到新数据，准备分析")
    
    async def contribute(self):
        """贡献分析"""
        # 读取数据
        data_entry = self.blackboard.read("collected_data")
        
        if data_entry:
            # 进行分析
            analysis = {
                "trends": ["趋势1", "趋势2"],
                "anomalies": ["异常1"],
                "insights": ["洞察1", "洞察2"],
            }
            
            # 写入黑板
            self.blackboard.write(
                key="analysis_result",
                value=analysis,
                author=self.name,
                confidence=0.85,
            )

class ReportWriterAgent(BlackboardAgent):
    """报告撰写 Agent"""
    
    def __init__(self, name: str, blackboard: Blackboard):
        super().__init__(name, blackboard)
    
    def on_blackboard_update(self, key: str, entry: BlackboardEntry):
        """黑板更新回调"""
        if key == "analysis_result":
            print(f"[{self.name}] 检测到分析结果，准备撰写报告")
    
    async def contribute(self):
        """贡献报告"""
        # 读取分析结果
        analysis_entry = self.blackboard.read("analysis_result")
        
        if analysis_entry:
            # 生成报告
            report = {
                "title": "数据分析报告",
                "executive_summary": "基于数据的分析发现",
                "findings": analysis_entry.value.get("insights", []),
                "recommendations": ["建议1", "建议2"],
            }
            
            # 写入黑板
            self.blackboard.write(
                key="final_report",
                value=report,
                author=self.name,
                confidence=0.95,
            )

class BlackboardOrchestrator:
    """黑板协调器"""
    
    def __init__(self):
        self.blackboard = Blackboard()
        self.agents: List[BlackboardAgent] = []
    
    def add_agent(self, agent: BlackboardAgent):
        """添加 Agent"""
        self.agents.append(agent)
    
    async def run_collaboration(self, initial_request: str):
        """运行协作"""
        print(f"\n{'='*50}")
        print(f"黑板协作开始")
        print(f"初始请求: {initial_request}")
        print(f"{'='*50}")
        
        # 写入初始请求
        self.blackboard.write(
            key="request_001",
            value=initial_request,
            author="orchestrator",
        )
        
        # 并行执行所有 Agent
        tasks = [agent.contribute() for agent in self.agents]
        await asyncio.gather(*tasks)
        
        # 获取最终结果
        final_report = self.blackboard.read("final_report")
        
        print(f"\n{'='*50}")
        print(f"黑板协作完成")
        if final_report:
            print(f"最终报告: {final_report.value.get('title', 'N/A')}")
        print(f"{'='*50}")
        
        return final_report
    
    def get_blackboard_state(self) -> Dict:
        """获取黑板状态"""
        entries = self.blackboard.read_all()
        return {
            "total_entries": len(entries),
            "keys": list(entries.keys()),
            "agents": [agent.name for agent in self.agents],
        }

# 使用示例
async def blackboard_example():
    """黑板模式示例"""
    # 创建协调器
    orchestrator = BlackboardOrchestrator()
    
    # 添加 Agent
    orchestrator.add_agent(DataGathererAgent("数据收集员", orchestrator.blackboard))
    orchestrator.add_agent(AnalystAgent("分析师", orchestrator.blackboard))
    orchestrator.add_agent(ReportWriterAgent("报告撰写者", orchestrator.blackboard))
    
    # 运行协作
    result = await orchestrator.run_collaboration("分析用户行为数据")
    
    print(f"\n黑板状态: {orchestrator.get_blackboard_state()}")

asyncio.run(blackboard_example())
```

---

## 20.8 模式选型指南

### 20.8.1 选型决策矩阵

| 需求维度 | 主从模式 | 对等协作 | 辩论模式 | 投票模式 | 流水线 | 黑板模式 |
|----------|----------|----------|----------|----------|--------|----------|
| **任务复杂度** | 中等 | 高 | 高 | 中等 | 中等 | 高 |
| **协调开销** | 低 | 高 | 高 | 中等 | 低 | 中等 |
| **并行能力** | 中等 | 高 | 低 | 高 | 低 | 高 |
| **容错性** | 低 | 中等 | 中等 | 高 | 低 | 高 |
| **扩展性** | 中等 | 高 | 低 | 中等 | 中等 | 高 |
| **实现难度** | 低 | 中等 | 高 | 中等 | 低 | 高 |

### 20.8.2 场景推荐

```python
# 场景1：简单任务分配 -> 主从模式
# 场景2：团队头脑风暴 -> 对等协作
# 场景3：方案评估辩论 -> 辩论模式
# 场景4：多人评审决策 -> 投票模式
# 场景5：数据处理管道 -> 流水线模式
# 场景6：知识协作编辑 -> 黑板模式

def recommend_pattern(requirements: Dict) -> str:
    """根据需求推荐模式"""
    if requirements.get("task_type") == "simple":
        return "主从模式"
    elif requirements.get("need_debate"):
        return "辩论模式"
    elif requirements.get("need_voting"):
        return "投票模式"
    elif requirements.get("need_pipeline"):
        return "流水线模式"
    elif requirements.get("need_shared_knowledge"):
        return "黑板模式"
    else:
        return "对等协作"
```

### 20.8.3 混合模式

```python
# 实际应用中，常常需要混合多种模式

class HybridMultiAgentSystem:
    """混合多智能体系统"""
    
    def __init__(self):
        # 主从模式用于任务分配
        self.master = MasterAgent("主控")
        
        # 流水线模式用于数据处理
        self.pipeline = Pipeline("数据处理")
        
        # 投票模式用于决策
        self.voting = VotingManager()
        
        # 黑板模式用于知识共享
        self.blackboard = Blackboard()
    
    async def execute(self, task: str):
        """执行混合任务"""
        # 1. 主从模式分配任务
        subtasks = self.master.decompose_task(task)
        
        # 2. 流水线模式处理数据
        for subtask in subtasks:
            await self.pipeline.execute(subtask)
        
        # 3. 投票模式做决策
        proposal = VotingProposal(
            id="decision_1",
            title="最终决策",
            description="基于处理结果的决策",
        )
        await self.voting.conduct_voting(proposal)
        
        # 4. 黑板模式共享知识
        self.blackboard.write(
            key="final_result",
            value={"task": task, "status": "completed"},
            author="system",
        )
```

---

## 20.9 高级特性与最佳实践

### 20.9.1 模式组合策略

```python
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class PatternConfig:
    """模式配置"""
    name: str
    type: str
    agents: List[str]
    parameters: Dict[str, Any]

class PatternCombiner:
    """模式组合器"""
    
    def __init__(self):
        self.patterns: List[PatternConfig] = []
        self.execution_order: List[str] = []
    
    def add_pattern(self, config: PatternConfig):
        """添加模式"""
        self.patterns.append(config)
        self.execution_order.append(config.name)
    
    async def execute_hybrid(self, task: str) -> Dict[str, Any]:
        """执行混合模式"""
        results = {}
        
        for pattern_name in self.execution_order:
            pattern = next(p for p in self.patterns if p.name == pattern_name)
            
            print(f"\n执行模式: {pattern.name} ({pattern.type})")
            
            # 根据模式类型执行
            if pattern.type == "master_slave":
                result = await self._execute_master_slave(task, pattern)
            elif pattern.type == "pipeline":
                result = await self._execute_pipeline(task, pattern)
            elif pattern.type == "voting":
                result = await self._execute_voting(task, pattern)
            else:
                result = {"status": "executed"}
            
            results[pattern_name] = result
        
        return results
    
    async def _execute_master_slave(self, task: str, config: PatternConfig) -> Dict:
        """执行主从模式"""
        return {"task": task, "agents": config.agents, "status": "completed"}
    
    async def _execute_pipeline(self, task: str, config: PatternConfig) -> Dict:
        """执行流水线模式"""
        return {"task": task, "stages": len(config.agents), "status": "completed"}
    
    async def _execute_voting(self, task: str, config: PatternConfig) -> Dict:
        """执行投票模式"""
        return {"task": task, "voters": len(config.agents), "status": "completed"}

# 使用示例
async def hybrid_example():
    """混合模式示例"""
    combiner = PatternCombiner()
    
    # 添加模式
    combiner.add_pattern(PatternConfig(
        name="任务分配",
        type="master_slave",
        agents=["manager", "worker1", "worker2"],
        parameters={"strategy": "round_robin"},
    ))
    
    combiner.add_pattern(PatternConfig(
        name="数据处理",
        type="pipeline",
        agents=["collector", "processor", "analyzer"],
        parameters={"parallel": False},
    ))
    
    combiner.add_pattern(PatternConfig(
        name="结果评审",
        type="voting",
        agents=["reviewer1", "reviewer2", "reviewer3"],
        parameters={"quorum": 2},
    ))
    
    # 执行
    results = await combiner.execute_hybrid("分析用户数据并生成报告")
    
    print(f"\n混合模式执行结果:")
    for pattern_name, result in results.items():
        print(f"  {pattern_name}: {result}")

asyncio.run(hybrid_example())
```

### 20.9.2 性能优化策略

```python
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.metrics = {
            "total_executions": 0,
            "total_time": 0,
            "parallel_executions": 0,
        }
    
    async def parallel_execute(self, tasks: List[Dict]) -> List[Dict]:
        """并行执行任务"""
        start_time = time.time()
        
        # 创建异步任务
        async_tasks = []
        for task in tasks:
            async_tasks.append(self._execute_single_task(task))
        
        # 并行执行
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # 更新指标
        duration = time.time() - start_time
        self.metrics["total_executions"] += len(tasks)
        self.metrics["total_time"] += duration
        self.metrics["parallel_executions"] += 1
        
        return results
    
    async def _execute_single_task(self, task: Dict) -> Dict:
        """执行单个任务"""
        # 模拟任务执行
        await asyncio.sleep(0.1)
        return {
            "task_id": task.get("id"),
            "status": "completed",
            "result": f"完成: {task.get('description', 'N/A')}",
        }
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        avg_time = (
            self.metrics["total_time"] / self.metrics["parallel_executions"]
            if self.metrics["parallel_executions"] > 0
            else 0
        )
        
        return {
            **self.metrics,
            "average_time": avg_time,
            "tasks_per_second": (
                self.metrics["total_executions"] / self.metrics["total_time"]
                if self.metrics["total_time"] > 0
                else 0
            ),
        }

# 使用示例
async def performance_example():
    """性能优化示例"""
    optimizer = PerformanceOptimizer(max_workers=5)
    
    # 创建任务
    tasks = [
        {"id": f"task_{i}", "description": f"任务 {i}"}
        for i in range(20)
    ]
    
    # 并行执行
    results = await optimizer.parallel_execute(tasks)
    
    print(f"执行结果: {len(results)} 个任务完成")
    print(f"性能统计: {optimizer.get_performance_stats()}")

asyncio.run(performance_example())
```

### 20.9.3 错误处理与容错

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
from enum import Enum

class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AgentError:
    """Agent 错误"""
    agent_name: str
    error_type: str
    message: str
    severity: ErrorSeverity
    recoverable: bool = True

class FaultTolerantSystem:
    """容错系统"""
    
    def __init__(self):
        self.errors: List[AgentError] = []
        self.recovery_strategies: Dict[str, callable] = {}
        self.max_retries: int = 3
    
    def register_recovery_strategy(self, error_type: str, strategy: callable):
        """注册恢复策略"""
        self.recovery_strategies[error_type] = strategy
    
    async def execute_with_fault_tolerance(
        self,
        agent_name: str,
        task: callable,
        *args,
        **kwargs
    ) -> Any:
        """带容错的执行"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                result = await task(*args, **kwargs)
                return result
                
            except Exception as e:
                error = AgentError(
                    agent_name=agent_name,
                    error_type=type(e).__name__,
                    message=str(e),
                    severity=self._classify_error(e),
                    recoverable=self._is_recoverable(e),
                )
                
                self.errors.append(error)
                last_error = error
                
                print(f"[容错] {agent_name} 执行失败 (尝试 {attempt + 1}): {e}")
                
                # 尝试恢复
                if error.recoverable and error.error_type in self.recovery_strategies:
                    recovery_result = await self.recovery_strategies[error.error_type](error)
                    if recovery_result:
                        continue
                
                # 不可恢复或达到最大重试次数
                if not error.recoverable or attempt == self.max_retries - 1:
                    break
                
                # 等待后重试
                await asyncio.sleep(1 * (attempt + 1))
        
        raise Exception(f"执行失败: {last_error.message}")
    
    def _classify_error(self, error: Exception) -> ErrorSeverity:
        """分类错误严重程度"""
        error_type = type(error).__name__
        
        if error_type in ["ConnectionError", "TimeoutError"]:
            return ErrorSeverity.MEDIUM
        elif error_type in ["ValueError", "TypeError"]:
            return ErrorSeverity.LOW
        elif error_type in ["MemoryError", "SystemError"]:
            return ErrorSeverity.CRITICAL
        else:
            return ErrorSeverity.HIGH
    
    def _is_recoverable(self, error: Exception) -> bool:
        """判断是否可恢复"""
        recoverable_errors = [
            "ConnectionError",
            "TimeoutError",
            "TemporaryError",
        ]
        return type(error).__name__ in recoverable_errors
    
    def get_error_stats(self) -> Dict:
        """获取错误统计"""
        stats = {}
        for error in self.errors:
            error_type = error.error_type
            if error_type not in stats:
                stats[error_type] = {"count": 0, "severity": error.severity.value}
            stats[error_type]["count"] += 1
        
        return stats

# 使用示例
async def fault_tolerance_example():
    """容错示例"""
    system = FaultTolerantSystem()
    
    # 注册恢复策略
    async def connection_error_recovery(error: AgentError) -> bool:
        print(f"  尝试恢复连接错误...")
        await asyncio.sleep(1)
        return True
    
    system.register_recovery_strategy("ConnectionError", connection_error_recovery)
    
    # 模拟任务
    async def risky_task():
        import random
        if random.random() < 0.5:
            raise ConnectionError("模拟连接错误")
        return "任务成功"
    
    # 执行
    try:
        result = await system.execute_with_fault_tolerance(
            "测试Agent",
            risky_task,
        )
        print(f"结果: {result}")
    except Exception as e:
        print(f"最终失败: {e}")
    
    print(f"错误统计: {system.get_error_stats()}")

asyncio.run(fault_tolerance_example())
```

### 20.9.4 监控与可观测性

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

@dataclass
class Span:
    """追踪跨度"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    attributes: Dict[str, Any] = field(default_factory=dict)

class MultiAgentTracer:
    """多 Agent 追踪器"""
    
    def __init__(self):
        self.traces: Dict[str, List[Span]] = {}
        self.current_trace_id: Optional[str] = None
    
    def start_trace(self, trace_id: str) -> str:
        """开始追踪"""
        self.current_trace_id = trace_id
        self.traces[trace_id] = []
        return trace_id
    
    def start_span(
        self,
        name: str,
        parent_span_id: Optional[str] = None,
        attributes: Dict[str, Any] = None
    ) -> Span:
        """开始跨度"""
        span = Span(
            trace_id=self.current_trace_id,
            span_id=f"span_{len(self.traces.get(self.current_trace_id, []))}",
            parent_span_id=parent_span_id,
            name=name,
            start_time=datetime.now(),
            attributes=attributes or {},
        )
        
        self.traces[self.current_trace_id].append(span)
        return span
    
    def end_span(self, span: Span, status: str = "completed"):
        """结束跨度"""
        span.end_time = datetime.now()
        span.status = status
    
    def get_trace_duration(self, trace_id: str) -> float:
        """获取追踪持续时间"""
        spans = self.traces.get(trace_id, [])
        if not spans:
            return 0
        
        start_time = min(s.start_time for s in spans)
        end_time = max(s.end_time for s in spans if s.end_time)
        
        if end_time:
            return (end_time - start_time).total_seconds()
        return 0
    
    def get_trace_summary(self, trace_id: str) -> Dict:
        """获取追踪摘要"""
        spans = self.traces.get(trace_id, [])
        
        return {
            "trace_id": trace_id,
            "total_spans": len(spans),
            "duration": self.get_trace_duration(trace_id),
            "spans": [
                {
                    "name": s.name,
                    "status": s.status,
                    "duration": (s.end_time - s.start_time).total_seconds() if s.end_time else None,
                }
                for s in spans
            ],
        }

# 使用示例
async def monitoring_example():
    """监控示例"""
    tracer = MultiAgentTracer()
    
    # 开始追踪
    trace_id = tracer.start_trace("task_001")
    
    # 模拟 Agent 执行
    span1 = tracer.start_span("数据收集")
    await asyncio.sleep(0.1)
    tracer.end_span(span1)
    
    span2 = tracer.start_span("数据分析", parent_span_id=span1.span_id)
    await asyncio.sleep(0.2)
    tracer.end_span(span2)
    
    span3 = tracer.start_span("报告生成", parent_span_id=span2.span_id)
    await asyncio.sleep(0.1)
    tracer.end_span(span3)
    
    # 获取摘要
    summary = tracer.get_trace_summary(trace_id)
    print(f"追踪摘要: {summary}")

asyncio.run(monitoring_example())
```

### 20.9.5 最佳实践总结

> **最佳实践清单**：
> 
> 1. **模式选择**
>    - 根据任务特点选择合适的模式
>    - 考虑团队规模和协作需求
>    - 必要时混合使用多种模式
> 
> 2. **Agent 设计**
>    - 为每个 Agent 定义清晰的职责
>    - 实现标准化的接口
>    - 支持动态注册和注销
> 
> 3. **通信机制**
>    - 选择合适的通信方式（直接/间接）
>    - 实现消息队列解耦
>    - 支持异步通信
> 
> 4. **容错设计**
>    - 实现重试机制
>    - 支持故障转移
>    - 记录错误日志
> 
> 5. **性能优化**
>    - 并行执行独立任务
>    - 实现结果缓存
>    - 监控资源使用
> 
> 6. **可观测性**
>    - 实现分布式追踪
>    - 监控关键指标
>    - 建立告警机制

### 20.9.6 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| **死锁** | 循环依赖 | 检查 Agent 依赖关系 |
| **活锁** | 无限循环 | 设置最大迭代次数 |
| **性能瓶颈** | 串行处理 | 改用并行模式 |
| **通信失败** | 网络问题 | 实现重试和超时 |
| **状态不一致** | 并发写入 | 使用锁或事务 |
| **错误累积** | 缺少容错 | 实现错误恢复 |

---

## 本章小结

本章深入探讨了多智能体协作的核心模式：

1. **主从模式**：主 Agent 分配任务，从 Agent 执行，适合简单任务分配
2. **对等协作**：Agent 地位平等，共同决策，适合团队协作
3. **辩论模式**：多个 Agent 辩论，达成共识，适合争议解决
4. **投票模式**：多个 Agent 投票，多数决定，适合决策制定
5. **流水线模式**：Agent 按顺序处理任务，适合流程化处理
6. **黑板模式**：Agent 通过共享黑板协作，适合知识共享

> **核心要点**：没有一种模式适合所有场景。实际应用中，应根据任务特点、团队规模和协作需求选择合适的模式，甚至混合使用多种模式。

---

## 思考题

1. 在什么场景下应该选择主从模式而非对等协作模式？
2. 辩论模式和投票模式各有什么优缺点？
3. 如何设计一个支持动态模式切换的多智能体系统？
4. 黑板模式如何处理并发写入冲突？
5. 如何评估多智能体协作系统的性能？

---

## 参考资源

- [多智能体系统导论](https://www.masfai.com/)
- [协作 AI 研究](https://arxiv.org/abs/2308.08155)
- [OpenAI 多 Agent 研究](https://openai.com/research/)
- [Microsoft AutoGen](https://microsoft.github.io/autogen/)
- [CrewAI 文档](https://docs.crewai.com/)
