---
title: "第30章：工作流编排"
description: "深入解析 Agent 工作流编排的核心技术：DAG 编排、条件分支、并行处理、HITL 人工审批与 Saga 错误补偿"
updated: "2025-06-15"
---


下面的交互式演示展示了监控仪表板的核心指标：

<div data-component="MonitoringDashboardDemo"></div>

# 第30章：工作流编排

> **学习目标**：
> - 掌握 DAG 编排与 StateGraph 的设计模式
> - 理解条件分支与动态路由的实现机制
> - 熟练使用 asyncio 实现并行任务处理
> - 掌握 Human-in-the-Loop (HITL) 人工审批流程
> - 理解 Saga 模式在分布式事务中的错误补偿策略
> - 能够设计并实现复杂的企业级 Agent 工作流

## 30.1 工作流编排概述

### 30.1.1 为什么需要工作流编排

单个 Agent 的能力有限，复杂任务需要多个 Agent 或多个步骤协同完成。工作流编排解决以下核心问题：

| 问题 | 手动编排的痛点 | 工作流编排方案 |
|------|---------------|---------------|
| 任务依赖 | 手动管理执行顺序 | DAG 自动解析依赖 |
| 并行执行 | 手动创建线程池 | 声明式并行配置 |
| 错误处理 | 每个步骤单独 try/catch | 统一的补偿/重试策略 |
| 状态管理 | 全局变量传递 | 类型安全的 State 机制 |
| 人工介入 | 中断流程等待输入 | HITL 审批节点 |

### 30.1.2 编排模式分类

```
工作流编排模式
├── 顺序编排（Sequential）
│   ├── 线性流水线
│   └── 链式调用
├── 并行编排（Parallel）
│   ├── Fork-Join 模式
│   ├── Map-Reduce 模式
│   └── Fan-out/Fan-in 模式
├── 条件编排（Conditional）
│   ├── If-Else 分支
│   ├── Switch-Case 路由
│   └── 动态路由
├── 循环编排（Loop）
│   ├── While 循环
│   ├── ForEach 迭代
│   └── 递归调用
├── 事件驱动编排（Event-Driven）
│   ├── 发布-订阅模式
│   └── 事件溯源
└── 混合编排（Hybrid）
    ├── 子图嵌套
    └── 跨工作流调用
```

## 30.2 DAG 编排与 StateGraph

### 30.2.1 DAG 基础数据结构

```python
# DAG 有向无环图实现
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections import defaultdict, deque
from enum import Enum
import uuid
import asyncio


class NodeStatus(Enum):
    """节点执行状态"""
    PENDING = "pending"         # 等待执行
    RUNNING = "running"         # 执行中
    COMPLETED = "completed"     # 执行完成
    FAILED = "failed"          # 执行失败
    SKIPPED = "skipped"        # 已跳过
    CANCELLED = "cancelled"    # 已取消


@dataclass
class DAGNode:
    """DAG 节点定义"""
    id: str                              # 节点唯一标识
    name: str                            # 节点名称
    handler: Callable | None = None      # 处理函数
    config: dict[str, Any] = field(default_factory=dict)  # 节点配置
    dependencies: list[str] = field(default_factory=list)  # 依赖节点 ID
    retry_count: int = 0                 # 重试次数
    timeout: float | None = None         # 超时时间（秒）
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None                   # 执行结果
    error: str | None = None             # 错误信息
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "result": str(self.result)[:100] if self.result else None,
            "error": self.error
        }


@dataclass
class DAGEdge:
    """DAG 边定义 - 连接两个节点"""
    source: str                          # 源节点 ID
    target: str                          # 目标节点 ID
    condition: Callable | None = None    # 条件函数（可选）
    
    def evaluate_condition(self, context: dict) -> bool:
        """评估边的条件"""
        if self.condition is None:
            return True
        return self.condition(context)


class DAG:
    """有向无环图 - 工作流编排的核心数据结构"""
    
    def __init__(self, name: str = "workflow"):
        self.name = name
        self.nodes: dict[str, DAGNode] = {}
        self.edges: list[DAGEdge] = []
        self._adjacency: dict[str, list[str]] = defaultdict(list)
        self._in_degree: dict[str, int] = defaultdict(int)
    
    def add_node(
        self,
        id: str,
        name: str,
        handler: Callable | None = None,
        dependencies: list[str] = None,
        **kwargs
    ) -> DAGNode:
        """添加节点"""
        if id in self.nodes:
            raise ValueError(f"节点 '{id}' 已存在")
        
        node = DAGNode(
            id=id,
            name=name,
            handler=handler,
            dependencies=dependencies or [],
            **{k: v for k, v in kwargs.items() if hasattr(DAGNode, k)}
        )
        self.nodes[id] = node
        
        # 更新入度
        for dep in node.dependencies:
            self._in_degree[id] += 1
        
        return node
    
    def add_edge(
        self,
        source: str,
        target: str,
        condition: Callable | None = None
    ) -> None:
        """添加边"""
        if source not in self.nodes:
            raise ValueError(f"源节点 '{source}' 不存在")
        if target not in self.nodes:
            raise ValueError(f"目标节点 '{target}' 不存在")
        
        edge = DAGEdge(source=source, target=target, condition=condition)
        self.edges.append(edge)
        self._adjacency[source].append(target)
        self._in_degree[target] += 1
    
    def validate(self) -> tuple[bool, str]:
        """验证 DAG 的有效性
        
        检查：
        1. 是否存在环
        2. 所有依赖节点是否存在
        3. 是否有孤立节点
        """
        # 拓扑排序检测环
        in_degree = dict(self._in_degree)
        queue = deque([n for n, d in in_degree.items() if d == 0])
        visited = 0
        
        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in self._adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if visited != len(self.nodes):
            return False, "DAG 中存在环"
        
        # 检查依赖节点是否存在
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    return False, f"节点 '{node.id}' 依赖的节点 '{dep}' 不存在"
        
        return True, "DAG 验证通过"
    
    def get_execution_order(self) -> list[list[str]]:
        """获取拓扑排序的执行层级（支持并行）"""
        in_degree = dict(self._in_degree)
        levels = []
        
        while True:
            # 找出所有入度为 0 的节点（可并行执行）
            current_level = [
                n for n, d in in_degree.items() if d == 0
            ]
            
            if not current_level:
                break
            
            levels.append(current_level)
            
            # 更新入度
            for node in current_level:
                for neighbor in self._adjacency.get(node, []):
                    in_degree[neighbor] -= 1
                del in_degree[node]
        
        return levels
    
    def get_node(self, id: str) -> DAGNode:
        """获取节点"""
        return self.nodes.get(id)
    
    def get_predecessors(self, id: str) -> list[str]:
        """获取节点的所有前驱节点"""
        return [
            e.source for e in self.edges if e.target == id
        ]
    
    def get_successors(self, id: str) -> list[str]:
        """获取节点的所有后继节点"""
        return self._adjacency.get(id, [])
    
    def visualize(self) -> str:
        """生成 DAG 的文本可视化"""
        lines = [f"DAG: {self.name}", "=" * 40]
        
        levels = self.get_execution_order()
        for i, level in enumerate(levels):
            lines.append(f"\n层级 {i}:")
            for node_id in level:
                node = self.nodes[node_id]
                deps = ", ".join(node.dependencies) if node.dependencies else "无"
                lines.append(f"  [{node_id}] {node.name} (依赖: {deps})")
        
        return "\n".join(lines)
```

### 30.2.2 StateGraph 状态图引擎

```python
# StateGraph 状态图引擎实现
import copy
from typing import Any


@dataclass
class GraphState:
    """图执行状态 - 类型安全的状态容器"""
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """安全获取状态值"""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """设置状态值"""
        self.data[key] = value
    
    def update(self, values: dict[str, Any]) -> None:
        """批量更新状态"""
        self.data.update(values)
    
    def snapshot(self) -> dict[str, Any]:
        """创建状态快照"""
        return copy.deepcopy(self.data)


@dataclass
class GraphNode:
    """状态图节点"""
    id: str
    name: str
    processor: Callable[[GraphState], GraphState]  # 状态处理器
    # 条件边：返回下一个节点 ID
    conditional_edges: dict[str, Callable[[GraphState], bool]] = field(
        default_factory=dict
    )
    # 默认下一节点
    default_next: str | None = None


class StateGraph:
    """状态图引擎 - 类似 LangGraph 的工作流引擎
    
    核心特性：
    - 基于状态的节点路由
    - 条件分支支持
    - 子图嵌套
    - 检查点与恢复
    """
    
    def __init__(self, name: str = "state_graph"):
        self.name = name
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, str | dict[str, str]] = {}
        self.entry_point: str | None = None
        self._state: GraphState = GraphState()
        self._checkpoints: list[dict] = []
    
    def add_node(
        self,
        id: str,
        processor: Callable[[GraphState], GraphState],
        name: str | None = None
    ) -> "StateGraph":
        """添加节点"""
        node = GraphNode(
            id=id,
            name=name or id,
            processor=processor
        )
        self.nodes[id] = node
        return self
    
    def add_edge(
        self,
        source: str,
        target: str
    ) -> "StateGraph":
        """添加边 - source 执行后跳转到 target"""
        if source not in self.nodes:
            raise ValueError(f"源节点 '{source}' 不存在")
        if target not in self.nodes:
            raise ValueError(f"目标节点 '{target}' 不存在")
        
        self.edges[source] = target
        self.nodes[source].default_next = target
        return self
    
    def add_conditional_edges(
        self,
        source: str,
        condition: Callable[[GraphState], str],
        targets: dict[str, str]
    ) -> "StateGraph":
        """添加条件边 - 根据状态选择目标节点
        
        Args:
            source: 源节点 ID
            condition: 条件函数，返回目标节点 key
            targets: key -> 目标节点 ID 的映射
        """
        if source not in self.nodes:
            raise ValueError(f"源节点 '{source}' 不存在")
        
        self.edges[source] = {"condition": condition, "targets": targets}
        return self
    
    def set_entry_point(self, id: str) -> "StateGraph":
        """设置入口节点"""
        if id not in self.nodes:
            raise ValueError(f"节点 '{id}' 不存在")
        self.entry_point = id
        return self
    
    def compile(self) -> "CompiledGraph":
        """编译状态图为可执行图"""
        if not self.entry_point:
            raise ValueError("未设置入口节点")
        
        # 验证所有目标节点存在
        for source, edge in self.edges.items():
            if isinstance(edge, dict):
                for key, target in edge["targets"].items():
                    if target not in self.nodes:
                        raise ValueError(
                            f"条件边目标节点 '{target}' 不存在"
                        )
            elif edge not in self.nodes:
                raise ValueError(f"边目标节点 '{edge}' 不存在")
        
        return CompiledGraph(self)
    
    def get_checkpoint(self) -> dict:
        """创建当前状态的检查点"""
        checkpoint = {
            "state": self._state.snapshot(),
            "node_statuses": {
                nid: "pending" for nid in self.nodes
            }
        }
        self._checkpoints.append(checkpoint)
        return checkpoint
    
    def restore_checkpoint(self, checkpoint: dict) -> None:
        """从检查点恢复"""
        self._state = GraphState(data=checkpoint["state"])
    
    def visualize(self) -> str:
        """可视化状态图"""
        lines = [f"StateGraph: {self.name}", "=" * 40]
        lines.append(f"入口节点: {self.entry_point}")
        lines.append("")
        
        for node_id, node in self.nodes.items():
            lines.append(f"节点: {node_id} ({node.name})")
            
            edge = self.edges.get(node_id)
            if edge is None:
                lines.append("  → [END]")
            elif isinstance(edge, str):
                lines.append(f"  → {edge}")
            elif isinstance(edge, dict):
                lines.append("  条件路由:")
                for key, target in edge["targets"].items():
                    lines.append(f"    [{key}] → {target}")
        
        return "\n".join(lines)


class CompiledGraph:
    """编译后的状态图 - 可执行的工作流"""
    
    def __init__(self, graph: StateGraph):
        self.graph = graph
        self._execution_log: list[dict] = []
    
    async def run(
        self,
        initial_state: dict[str, Any] = None,
        max_iterations: int = 100
    ) -> GraphState:
        """执行工作流
        
        Args:
            initial_state: 初始状态
            max_iterations: 最大迭代次数（防止无限循环）
        
        Returns:
            最终状态
        """
        state = GraphState(data=initial_state or {})
        current_node_id = self.graph.entry_point
        iteration = 0
        
        while current_node_id and iteration < max_iterations:
            iteration += 1
            
            # 获取当前节点
            node = self.graph.nodes.get(current_node_id)
            if not node:
                raise RuntimeError(f"节点 '{current_node_id}' 未找到")
            
            # 执行节点处理器
            self._log("node_start", current_node_id, state)
            
            try:
                state = node.processor(state)
            except Exception as e:
                self._log("node_error", current_node_id, state, error=str(e))
                state.set("_error", str(e))
                state.set("_failed_node", current_node_id)
                break
            
            self._log("node_complete", current_node_id, state)
            
            # 决定下一个节点
            edge = self.graph.edges.get(current_node_id)
            
            if edge is None:
                # 无后继节点，结束
                current_node_id = None
            elif isinstance(edge, str):
                # 简单边，直接跳转
                current_node_id = edge
            elif isinstance(edge, dict):
                # 条件边，评估条件
                condition = edge["condition"]
                key = condition(state)
                current_node_id = edge["targets"].get(key)
        
        if iteration >= max_iterations:
            state.set("_error", "达到最大迭代次数")
            state.set("_halted", True)
        
        return state
    
    def _log(
        self,
        event: str,
        node_id: str,
        state: GraphState,
        error: str = None
    ) -> None:
        """记录执行日志"""
        entry = {
            "event": event,
            "node_id": node_id,
            "state_keys": list(state.data.keys()),
        }
        if error:
            entry["error"] = error
        self._execution_log.append(entry)
    
    def get_execution_log(self) -> list[dict]:
        """获取执行日志"""
        return self._execution_log
```

### 30.2.3 StateGraph 实战示例

```python
# 构建一个完整的 StateGraph 工作流
async def demo_stategraph():
    """演示 StateGraph 工作流"""
    
    def preprocess(state: GraphState) -> GraphState:
        """预处理节点 - 清洗输入数据"""
        raw_data = state.get("raw_data", "")
        cleaned = raw_data.strip().lower()
        state.set("cleaned_data", cleaned)
        state.set("data_length", len(cleaned))
        return state
    
    def analyze(state: GraphState) -> GraphState:
        """分析节点 - 分析数据特征"""
        data = state.get("cleaned_data", "")
        
        # 简单的分析逻辑
        word_count = len(data.split())
        char_count = len(data)
        
        state.set("word_count", word_count)
        state.set("char_count", char_count)
        state.set("complexity", "high" if word_count > 100 else "low")
        
        return state
    
    def route_by_complexity(state: GraphState) -> str:
        """路由函数 - 根据复杂度选择处理路径"""
        complexity = state.get("complexity", "low")
        if complexity == "high":
            return "deep_analysis"
        return "simple_report"
    
    def deep_analysis(state: GraphState) -> GraphState:
        """深度分析节点"""
        data = state.get("cleaned_data", "")
        # 模拟深度分析
        state.set("analysis_result", f"深度分析完成: {len(data)} 字符")
        state.set("next_action", "generate_report")
        return state
    
    def simple_report(state: GraphState) -> GraphState:
        """简单报告节点"""
        state.set("analysis_result", "简单分析完成")
        state.set("next_action", "generate_report")
        return state
    
    def generate_report(state: GraphState) -> GraphState:
        """生成报告节点"""
        analysis = state.get("analysis_result", "")
        state.set("final_report", f"报告: {analysis}")
        return state
    
    # 构建状态图
    graph = StateGraph("data_analysis_workflow")
    
    graph.add_node("preprocess", preprocess, "数据预处理")
    graph.add_node("analyze", analyze, "数据分析")
    graph.add_node("deep_analysis", deep_analysis, "深度分析")
    graph.add_node("simple_report", simple_report, "简单报告")
    graph.add_node("generate_report", generate_report, "生成报告")
    
    graph.set_entry_point("preprocess")
    
    graph.add_edge("preprocess", "analyze")
    graph.add_conditional_edges(
        "analyze",
        route_by_complexity,
        {
            "high": "deep_analysis",
            "low": "simple_report"
        }
    )
    graph.add_edge("deep_analysis", "generate_report")
    graph.add_edge("simple_report", "generate_report")
    
    # 编译并执行
    compiled = graph.compile()
    
    result = await compiled.run({
        "raw_data": "  Hello World  This is a test message  "
    })
    
    print(f"最终报告: {result.get('final_report')}")
    print(f"执行日志: {len(compiled.get_execution_log())} 步")
    
    # 打印可视化
    print(graph.visualize())
```

## 30.3 条件分支与动态路由

### 30.3.1 多级条件路由

```python
# 多级条件路由引擎
from typing import Any
from dataclasses import dataclass, field


class RoutingStrategy(Enum):
    """路由策略"""
    PRIORITY = "priority"     # 优先级路由
    ROUND_ROBIN = "round_robin"  # 轮询路由
    LEAST_LOADED = "least_loaded"  # 最少负载路由
    RANDOM = "random"         # 随机路由
    CONTENT_BASED = "content_based"  # 内容路由


@dataclass
class RouteRule:
    """路由规则"""
    name: str                          # 规则名称
    condition: Callable[[GraphState], bool]  # 条件函数
    target_node: str                   # 目标节点
    priority: int = 0                  # 优先级（越大越优先）
    metadata: dict = field(default_factory=dict)
    
    def matches(self, state: GraphState) -> bool:
        """检查条件是否匹配"""
        return self.condition(state)


class DynamicRouter:
    """动态路由器 - 支持运行时动态添加/移除路由规则"""
    
    def __init__(self):
        self._rules: dict[str, list[RouteRule]] = defaultdict(list)
        self._load_counters: dict[str, int] = defaultdict(int)
        self._round_robin_index: dict[str, int] = defaultdict(int)
    
    def add_rule(
        self,
        source_node: str,
        rule: RouteRule
    ) -> None:
        """添加路由规则"""
        self._rules[source_node].append(rule)
        # 按优先级排序
        self._rules[source_node].sort(
            key=lambda r: r.priority, reverse=True
        )
    
    def remove_rule(self, source_node: str, rule_name: str) -> bool:
        """移除路由规则"""
        rules = self._rules.get(source_node, [])
        for i, rule in enumerate(rules):
            if rule.name == rule_name:
                rules.pop(i)
                return True
        return False
    
    def route(
        self,
        source_node: str,
        state: GraphState,
        strategy: RoutingStrategy = RoutingStrategy.PRIORITY
    ) -> str | None:
        """根据策略选择目标节点"""
        rules = self._rules.get(source_node, [])
        
        if not rules:
            return None
        
        matching_rules = [r for r in rules if r.matches(state)]
        
        if not matching_rules:
            return None
        
        if strategy == RoutingStrategy.PRIORITY:
            # 返回优先级最高的匹配规则
            return matching_rules[0].target_node
        
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            # 轮询匹配的规则
            idx = self._round_robin_index[source_node]
            target = matching_rules[idx % len(matching_rules)].target_node
            self._round_robin_index[source_node] = idx + 1
            return target
        
        elif strategy == RoutingStrategy.LEAST_LOADED:
            # 选择负载最小的目标
            min_load = float('inf')
            best_target = None
            for rule in matching_rules:
                load = self._load_counters[rule.target_node]
                if load < min_load:
                    min_load = load
                    best_target = rule.target_node
            return best_target
        
        elif strategy == RoutingStrategy.RANDOM:
            import random
            return random.choice(matching_rules).target_node
        
        elif strategy == RoutingStrategy.CONTENT_BASED:
            # 基于内容的路由
            for rule in matching_rules:
                if rule.matches(state):
                    return rule.target_node
        
        return matching_rules[0].target_node
    
    def increment_load(self, node_id: str) -> None:
        """增加节点负载计数"""
        self._load_counters[node_id] += 1
    
    def decrement_load(self, node_id: str) -> None:
        """减少节点负载计数"""
        self._load_counters[node_id] = max(
            0, self._load_counters[node_id] - 1
        )
    
    def get_load_distribution(self) -> dict[str, int]:
        """获取负载分布"""
        return dict(self._load_counters)
```

### 30.3.2 动态路由配置

```python
# 动态路由配置系统
class RoutingConfig:
    """路由配置管理"""
    
    def __init__(self):
        self._configs: dict[str, dict] = {}
    
    def register(
        self,
        workflow_name: str,
        routing_rules: list[dict]
    ) -> None:
        """注册工作流路由配置"""
        self._configs[workflow_name] = {
            "rules": routing_rules,
            "created_at": datetime.now().isoformat()
        }
    
    def get_rules(self, workflow_name: str) -> list[dict]:
        """获取工作流路由规则"""
        config = self._configs.get(workflow_name, {})
        return config.get("rules", [])
    
    def update_rule(
        self,
        workflow_name: str,
        rule_name: str,
        updates: dict
    ) -> bool:
        """更新路由规则"""
        config = self._configs.get(workflow_name)
        if not config:
            return False
        
        for rule in config["rules"]:
            if rule["name"] == rule_name:
                rule.update(updates)
                return True
        
        return False


# 使用示例
def demo_routing():
    """演示动态路由"""
    router = DynamicRouter()
    state = GraphState(data={"intent": "complaint", "priority": "high"})
    
    # 添加路由规则
    router.add_rule("classifier", RouteRule(
        name="complaint_high",
        condition=lambda s: s.get("intent") == "complaint" and s.get("priority") == "high",
        target_node="urgent_handler",
        priority=10
    ))
    
    router.add_rule("classifier", RouteRule(
        name="complaint_normal",
        condition=lambda s: s.get("intent") == "complaint",
        target_node="normal_handler",
        priority=5
    ))
    
    router.add_rule("classifier", RouteRule(
        name="default",
        condition=lambda s: True,
        target_node="default_handler",
        priority=0
    ))
    
    # 路由
    target = router.route("classifier", state)
    print(f"路由结果: {target}")  # urgent_handler
    
    # 验证输出
    assert target == "urgent_handler", f"期望 urgent_handler，实际 {target}"
```

## 30.4 并行处理与 asyncio

### 30.4.1 并行执行引擎

```python
# 并行任务执行引擎
import asyncio
from typing import Any
from dataclasses import dataclass, field
from enum import Enum
import time


class ParallelStrategy(Enum):
    """并行策略"""
    ALL = "all"              # 等待所有完成
    ANY = "any"              # 任意一个完成即可
    RACE = "race"            # 竞争模式（最快优先）
    THROTTLE = "throttle"    # 限流并行


@dataclass
class ParallelTask:
    """并行任务"""
    id: str
    name: str
    coroutine: Callable                    # 异步协程
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    timeout: float | None = None           # 任务超时
    priority: int = 0                      # 优先级
    result: Any = None                     # 执行结果
    error: str | None = None               # 错误信息
    duration: float | None = None          # 执行耗时


class ParallelExecutor:
    """并行任务执行器"""
    
    def __init__(
        self,
        max_concurrent: int = 10,
        strategy: ParallelStrategy = ParallelStrategy.ALL
    ):
        self.max_concurrent = max_concurrent
        self.strategy = strategy
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._results: dict[str, ParallelTask] = {}
    
    async def execute(
        self, tasks: list[ParallelTask]
    ) -> dict[str, ParallelTask]:
        """执行并行任务"""
        self._results = {}
        
        if self.strategy == ParallelStrategy.ALL:
            return await self._execute_all(tasks)
        elif self.strategy == ParallelStrategy.ANY:
            return await self._execute_any(tasks)
        elif self.strategy == ParallelStrategy.RACE:
            return await self._execute_race(tasks)
        elif self.strategy == ParallelStrategy.THROTTLE:
            return await self._execute_throttle(tasks)
        
        return self._results
    
    async def _execute_all(
        self, tasks: list[ParallelTask]
    ) -> dict[str, ParallelTask]:
        """等待所有任务完成"""
        async def run_task(task: ParallelTask) -> ParallelTask:
            async with self._semaphore:
                start = time.time()
                try:
                    if task.timeout:
                        result = await asyncio.wait_for(
                            task.coroutine(*task.args, **task.kwargs),
                            timeout=task.timeout
                        )
                    else:
                        result = await task.coroutine(
                            *task.args, **task.kwargs
                        )
                    task.result = result
                    task.status = NodeStatus.COMPLETED
                except asyncio.TimeoutError:
                    task.error = f"任务超时 ({task.timeout}s)"
                    task.status = NodeStatus.FAILED
                except Exception as e:
                    task.error = str(e)
                    task.status = NodeStatus.FAILED
                finally:
                    task.duration = time.time() - start
                    self._results[task.id] = task
                return task
        
        # 并发执行所有任务
        await asyncio.gather(*[run_task(t) for t in tasks])
        return self._results
    
    async def _execute_any(
        self, tasks: list[ParallelTask]
    ) -> dict[str, ParallelTask]:
        """任意一个完成即可"""
        async def run_task(task: ParallelTask) -> ParallelTask:
            async with self._semaphore:
                start = time.time()
                try:
                    result = await task.coroutine(
                        *task.args, **task.kwargs
                    )
                    task.result = result
                    task.status = NodeStatus.COMPLETED
                except Exception as e:
                    task.error = str(e)
                    task.status = NodeStatus.FAILED
                finally:
                    task.duration = time.time() - start
                    self._results[task.id] = task
                return task
        
        # 创建所有任务
        coros = [run_task(t) for t in tasks]
        
        # 等待任意一个完成
        done, pending = await asyncio.wait(
            coros, return_when=asyncio.FIRST_COMPLETED
        )
        
        # 取消剩余任务
        for coro in pending:
            coro.cancel()
        
        return self._results
    
    async def _execute_race(
        self, tasks: list[ParallelTask]
    ) -> dict[str, ParallelTask]:
        """竞争模式 - 返回最快的结果"""
        results = await self._execute_all(tasks)
        
        # 找出最快的成功结果
        fastest = None
        for task in results.values():
            if task.status == NodeStatus.COMPLETED:
                if fastest is None or task.duration < fastest.duration:
                    fastest = task
        
        if fastest:
            self._results = {"winner": fastest}
        
        return self._results
    
    async def _execute_throttle(
        self, tasks: list[ParallelTask]
    ) -> dict[str, ParallelTask]:
        """限流并行 - 按批次执行"""
        batch_size = self.max_concurrent
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            await self._execute_all(batch)
        
        return self._results
    
    def get_summary(self) -> dict:
        """获取执行摘要"""
        total = len(self._results)
        success = sum(
            1 for t in self._results.values()
            if t.status == NodeStatus.COMPLETED
        )
        failed = sum(
            1 for t in self._results.values()
            if t.status == NodeStatus.FAILED
        )
        total_duration = sum(
            t.duration for t in self._results.values()
            if t.duration
        )
        
        return {
            "total": total,
            "success": success,
            "failed": failed,
            "total_duration": round(total_duration, 2),
            "avg_duration": round(total_duration / total, 2) if total else 0
        }
```

### 30.4.2 Fork-Join 模式

```python
# Fork-Join 并行模式实现
class ForkJoinExecutor:
    """Fork-Join 执行器
    
    将任务分叉(fork)到多个并行分支，等待所有分支完成(join)后继续
    """
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self._semaphore = asyncio.Semaphore(max_workers)
    
    async def fork_join(
        self,
        fork_point: str,
        branches: list[dict[str, Any]],
        join_strategy: str = "merge"
    ) -> dict:
        """Fork-Join 执行
        
        Args:
            fork_point: 分叉点名称
            branches: 分支配置列表
            join_strategy: 合并策略 (merge|reduce|collect)
        
        Returns:
            合并后的结果
        """
        print(f"[Fork-Join] 在 '{fork_point}' 处分叉 {len(branches)} 个分支")
        
        start_time = time.time()
        
        # 并行执行所有分支
        tasks = []
        for branch in branches:
            task = self._execute_branch(branch)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并结果
        merged = self._merge_results(results, join_strategy)
        
        duration = time.time() - start_time
        print(
            f"[Fork-Join] {len(branches)} 个分支完成，"
            f"耗时 {duration:.2f}s"
        )
        
        return merged
    
    async def _execute_branch(
        self, branch: dict[str, Any]
    ) -> Any:
        """执行单个分支"""
        async with self._semaphore:
            handler = branch["handler"]
            args = branch.get("args", ())
            kwargs = branch.get("kwargs", {})
            name = branch.get("name", "unnamed")
            
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(*args, **kwargs)
                else:
                    result = handler(*args, **kwargs)
                return {"name": name, "result": result, "error": None}
            except Exception as e:
                return {"name": name, "result": None, "error": str(e)}
    
    def _merge_results(
        self, results: list, strategy: str
    ) -> dict:
        """合并分支结果"""
        if strategy == "merge":
            # 简单合并
            merged = {}
            for r in results:
                if isinstance(r, dict) and not r.get("error"):
                    merged[r["name"]] = r["result"]
            return merged
        
        elif strategy == "reduce":
            # 归约合并
            values = [
                r["result"] for r in results
                if isinstance(r, dict) and not r.get("error")
            ]
            if values:
                return {"reduced": values[0].__class__.__add__(
                    *values
                ) if len(values) > 1 else values[0]}
            return {"reduced": None}
        
        elif strategy == "collect":
            # 收集所有结果
            return {
                "results": [
                    {
                        "name": r.get("name", "unknown"),
                        "result": r.get("result"),
                        "error": r.get("error")
                    }
                    for r in results if isinstance(r, dict)
                ]
            }
        
        return {"raw": results}


# 使用示例
async def demo_fork_join():
    """演示 Fork-Join 模式"""
    executor = ForkJoinExecutor(max_workers=3)
    
    async def fetch_user_data(user_id: int) -> dict:
        await asyncio.sleep(0.1)  # 模拟网络请求
        return {"user_id": user_id, "name": f"User_{user_id}"}
    
    async def fetch_user_orders(user_id: int) -> list:
        await asyncio.sleep(0.15)
        return [{"order_id": i, "user_id": user_id} for i in range(3)]
    
    async def fetch_user_recommendations(user_id: int) -> list:
        await asyncio.sleep(0.2)
        return [f"rec_{i}" for i in range(5)]
    
    user_id = 123
    
    result = await executor.fork_join(
        fork_point="user_dashboard",
        branches=[
            {
                "name": "user_data",
                "handler": fetch_user_data,
                "args": (user_id,)
            },
            {
                "name": "user_orders",
                "handler": fetch_user_orders,
                "args": (user_id,)
            },
            {
                "name": "recommendations",
                "handler": fetch_user_recommendations,
                "args": (user_id,)
            }
        ],
        join_strategy="merge"
    )
    
    print(f"Fork-Join 结果: {result}")
```

### 30.4.3 Map-Reduce 模式

```python
# Map-Reduce 并行模式实现
class MapReduceExecutor:
    """Map-Reduce 执行器
    
    将数据分片并行处理(map)，然后合并结果(reduce)
    """
    
    def __init__(self, chunk_size: int = 10, max_workers: int = 5):
        self.chunk_size = chunk_size
        self.max_workers = max_workers
    
    async def execute(
        self,
        data: list[Any],
        map_fn: Callable[[Any], Any],
        reduce_fn: Callable[[list[Any]], Any],
        combine_fn: Callable[[Any, Any], Any] | None = None
    ) -> Any:
        """执行 Map-Reduce
        
        Args:
            data: 输入数据列表
            map_fn: Map 函数（并行处理每个数据项）
            reduce_fn: Reduce 函数（合并所有结果）
            combine_fn: 可选的中间合并函数
        
        Returns:
            Reduce 后的最终结果
        """
        # Step 1: 分片
        chunks = [
            data[i:i + self.chunk_size]
            for i in range(0, len(data), self.chunk_size)
        ]
        
        print(f"[Map-Reduce] {len(data)} 项数据分片为 {len(chunks)} 块")
        
        # Step 2: Map（并行处理）
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def map_chunk(chunk: list) -> list:
            async with semaphore:
                results = []
                for item in chunk:
                    if asyncio.iscoroutinefunction(map_fn):
                        result = await map_fn(item)
                    else:
                        result = map_fn(item)
                    results.append(result)
                return results
        
        mapped_results = await asyncio.gather(
            *[map_chunk(c) for c in chunks]
        )
        
        # 展平结果
        flat_results = [
            item for chunk in mapped_results for item in chunk
        ]
        
        print(f"[Map-Reduce] Map 完成，{len(flat_results)} 个结果")
        
        # Step 3: Reduce
        if combine_fn:
            # 使用 combine_fn 逐步合并
            final = flat_results[0]
            for item in flat_results[1:]:
                final = combine_fn(final, item)
            return final
        else:
            return reduce_fn(flat_results)


# 使用示例
async def demo_map_reduce():
    """演示 Map-Reduce"""
    executor = MapReduceExecutor(chunk_size=5)
    
    # 模拟数据处理
    data = list(range(1, 101))  # 1 到 100
    
    async def square(x: int) -> int:
        """Map: 计算平方"""
        await asyncio.sleep(0.01)
        return x * x
    
    def sum_results(results: list[int]) -> int:
        """Reduce: 求和"""
        return sum(results)
    
    result = await executor.execute(
        data=data,
        map_fn=square,
        reduce_fn=sum_results
    )
    
    print(f"Map-Reduce 结果: {result}")  # 338350
```

## 30.5 Human-in-the-Loop (HITL) 人工审批

### 30.5.1 HITL 审批流程引擎

```python
# Human-in-the-Loop 审批引擎
import uuid
from datetime import datetime, timedelta
from typing import Any
from enum import Enum


class ApprovalStatus(Enum):
    """审批状态"""
    PENDING = "pending"           # 等待审批
    APPROVED = "approved"         # 已批准
    REJECTED = "rejected"         # 已拒绝
    ESCALATED = "escalated"       # 已升级
    EXPIRED = "expired"           # 已过期
    CANCELLED = "cancelled"       # 已取消


class ApprovalPriority(Enum):
    """审批优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ApprovalRequest:
    """审批请求"""
    id: str                          # 请求 ID
    workflow_id: str                 # 工作流 ID
    node_id: str                     # 触发审批的节点
    requester: str                   # 请求者（Agent 名称）
    approvers: list[str]             # 审批人列表
    title: str                       # 审批标题
    description: str                 # 审批描述
    context: dict[str, Any]          # 审批上下文数据
    status: ApprovalStatus = ApprovalStatus.PENDING
    priority: ApprovalPriority = ApprovalPriority.MEDIUM
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    expires_at: str | None = None    # 过期时间
    decision: str | None = None      # 审批决定
    comment: str | None = None       # 审批意见
    decided_by: str | None = None    # 审批人
    decided_at: str | None = None    # 决定时间
    
    def is_expired(self) -> bool:
        """检查是否已过期"""
        if not self.expires_at:
            return False
        return datetime.now() > datetime.fromisoformat(self.expires_at)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "node_id": self.node_id,
            "requester": self.requester,
            "approvers": self.approvers,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "decision": self.decision,
            "comment": self.comment,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at
        }


class HITLEngine:
    """Human-in-the-Loop 审批引擎"""
    
    def __init__(self):
        self._requests: dict[str, ApprovalRequest] = {}
        self._pending_queue: asyncio.Queue = asyncio.Queue()
        self._callbacks: dict[str, Callable] = {}
        self._timeout_checker_task: asyncio.Task | None = None
    
    async def start(self) -> None:
        """启动超时检查器"""
        self._timeout_checker_task = asyncio.create_task(
            self._check_timeouts()
        )
    
    async def stop(self) -> None:
        """停止引擎"""
        if self._timeout_checker_task:
            self._timeout_checker_task.cancel()
    
    def create_request(
        self,
        workflow_id: str,
        node_id: str,
        requester: str,
        approvers: list[str],
        title: str,
        description: str,
        context: dict[str, Any],
        priority: ApprovalPriority = ApprovalPriority.MEDIUM,
        timeout_hours: float = 24,
        callback: Callable | None = None
    ) -> ApprovalRequest:
        """创建审批请求"""
        request_id = str(uuid.uuid4())
        
        expires_at = (
            datetime.now() + timedelta(hours=timeout_hours)
        ).isoformat()
        
        request = ApprovalRequest(
            id=request_id,
            workflow_id=workflow_id,
            node_id=node_id,
            requester=requester,
            approvers=approvers,
            title=title,
            description=description,
            context=context,
            priority=priority,
            expires_at=expires_at
        )
        
        self._requests[request_id] = request
        
        if callback:
            self._callbacks[request_id] = callback
        
        return request
    
    def approve(
        self,
        request_id: str,
        approver: str,
        comment: str = ""
    ) -> bool:
        """批准请求"""
        request = self._requests.get(request_id)
        if not request:
            return False
        
        if approver not in request.approvers:
            return False
        
        if request.status != ApprovalStatus.PENDING:
            return False
        
        request.status = ApprovalStatus.APPROVED
        request.decision = "approved"
        request.comment = comment
        request.decided_by = approver
        request.decided_at = datetime.now().isoformat()
        
        # 触发回调
        if request_id in self._callbacks:
            self._callbacks[request_id](request)
        
        return True
    
    def reject(
        self,
        request_id: str,
        approver: str,
        reason: str = ""
    ) -> bool:
        """拒绝请求"""
        request = self._requests.get(request_id)
        if not request:
            return False
        
        if approver not in request.approvers:
            return False
        
        if request.status != ApprovalStatus.PENDING:
            return False
        
        request.status = ApprovalStatus.REJECTED
        request.decision = "rejected"
        request.comment = reason
        request.decided_by = approver
        request.decided_at = datetime.now().isoformat()
        
        if request_id in self._callbacks:
            self._callbacks[request_id](request)
        
        return True
    
    def escalate(self, request_id: str, reason: str = "") -> bool:
        """升级请求（通知更高级别审批人）"""
        request = self._requests.get(request_id)
        if not request:
            return False
        
        request.status = ApprovalStatus.ESCALATED
        request.comment = f"已升级: {reason}"
        
        return True
    
    def cancel(self, request_id: str) -> bool:
        """取消请求"""
        request = self._requests.get(request_id)
        if not request:
            return False
        
        if request.status == ApprovalStatus.PENDING:
            request.status = ApprovalStatus.CANCELLED
            return True
        
        return False
    
    async def wait_for_approval(
        self,
        request_id: str,
        timeout: float | None = None
    ) -> ApprovalRequest:
        """等待审批结果"""
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"审批请求 '{request_id}' 未找到")
        
        start_time = time.time()
        
        while request.status == ApprovalStatus.PENDING:
            if timeout and (time.time() - start_time) > timeout:
                request.status = ApprovalStatus.EXPIRED
                break
            
            # 检查是否过期
            if request.is_expired():
                request.status = ApprovalStatus.EXPIRED
                break
            
            await asyncio.sleep(0.5)
        
        return request
    
    def get_pending_requests(
        self,
        approver: str | None = None
    ) -> list[ApprovalRequest]:
        """获取待审批请求"""
        pending = [
            r for r in self._requests.values()
            if r.status == ApprovalStatus.PENDING
        ]
        
        if approver:
            pending = [
                r for r in pending
                if approver in r.approvers
            ]
        
        # 按优先级排序
        priority_order = {
            ApprovalPriority.CRITICAL: 0,
            ApprovalPriority.HIGH: 1,
            ApprovalPriority.MEDIUM: 2,
            ApprovalPriority.LOW: 3
        }
        pending.sort(key=lambda r: priority_order[r.priority])
        
        return pending
    
    async def _check_timeouts(self) -> None:
        """定期检查超时请求"""
        while True:
            await asyncio.sleep(60)  # 每分钟检查一次
            
            for request in list(self._requests.values()):
                if (
                    request.status == ApprovalStatus.PENDING
                    and request.is_expired()
                ):
                    request.status = ApprovalStatus.EXPIRED
                    if request.id in self._callbacks:
                        self._callbacks[request.id](request)
```

### 30.5.2 HITL 与 StateGraph 集成

```python
# HITL 节点集成到 StateGraph
class HITLNode:
    """HITL 审批节点 - 可嵌入 StateGraph"""
    
    def __init__(
        self,
        node_id: str,
        hitl_engine: HITLEngine,
        approvers: list[str],
        timeout_hours: float = 24,
        auto_approve: bool = False
    ):
        self.node_id = node_id
        self.hitl_engine = hitl_engine
        self.approvers = approvers
        self.timeout_hours = timeout_hours
        self.auto_approve = auto_approve
    
    def create_processor(
        self,
        title_template: str = "工作流审批",
        description_template: str = "需要人工审批"
    ) -> Callable[[GraphState], GraphState]:
        """创建 StateGraph 节点处理器"""
        
        def processor(state: GraphState) -> GraphState:
            workflow_id = state.get("_workflow_id", "unknown")
            
            # 生成审批描述
            title = title_template.format(**state.data)
            description = description_template.format(**state.data)
            
            # 创建审批请求
            request = self.hitl_engine.create_request(
                workflow_id=workflow_id,
                node_id=self.node_id,
                requester=state.get("_agent_name", "system"),
                approvers=self.approvers,
                title=title,
                description=description,
                context=state.data,
                timeout_hours=self.timeout_hours
            )
            
            state.set("_approval_request_id", request.id)
            state.set("_waiting_approval", True)
            
            # 自动审批模式（用于测试）
            if self.auto_approve:
                self.hitl_engine.approve(request.id, "auto_approver")
                state.set("_waiting_approval", False)
                state.set("approved", True)
            
            return state
        
        return processor


# 使用示例
async def demo_hitl():
    """演示 HITL 审批流程"""
    
    # 创建 HITL 引擎
    hitl = HITLEngine()
    await hitl.start()
    
    # 创建 StateGraph
    graph = StateGraph("hitl_demo")
    
    def prepare_request(state: GraphState) -> GraphState:
        state.set("amount", 50000)
        state.set("requester", "Alice")
        state.set("purpose", "采购服务器")
        return state
    
    def process_approved(state: GraphState) -> GraphState:
        state.set("result", "采购请求已批准，进入执行阶段")
        return state
    
    def process_rejected(state: GraphState) -> GraphState:
        state.set("result", "采购请求被拒绝")
        return state
    
    # HITL 节点
    hitl_node = HITLNode(
        node_id="approval",
        hitl_engine=hitl,
        approvers=["manager", "finance"],
        timeout_hours=48
    )
    
    graph.add_node("prepare", prepare_request, "准备审批请求")
    graph.add_node(
        "approval",
        hitl_node.create_processor(
            title_template="采购审批: {purpose}",
            description_template="{requester} 请求 {purpose}，金额 {amount}"
        ),
        "等待审批"
    )
    graph.add_node("approved", process_approved, "审批通过")
    graph.add_node("rejected", process_rejected, "审批拒绝")
    
    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "approval")
    graph.add_conditional_edges(
        "approval",
        lambda s: "approved" if s.get("approved") else "rejected",
        {"approved": "approved", "rejected": "rejected"}
    )
    
    # 编译执行
    compiled = graph.compile()
    result = await compiled.run({"_workflow_id": "wf-001"})
    
    # 模拟审批
    pending = hitl.get_pending_requests("manager")
    if pending:
        hitl.approve(pending[0].id, "manager", "同意采购")
    
    # 重新执行等待审批的节点
    request_id = result.get("_approval_request_id")
    if request_id:
        request = await hitl.wait_for_approval(request_id, timeout=5)
        print(f"审批结果: {request.status.value}")
```

## 30.6 Saga 模式与错误补偿

### 30.6.1 Saga 模式实现

```python
# Saga 模式错误补偿引擎
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import uuid
import asyncio


class SagaStepStatus(Enum):
    """Saga 步骤状态"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"


class SagaStatus(Enum):
    """Saga 整体状态"""
    RUNNING = "running"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"


@dataclass
class SagaStep:
    """Saga 步骤"""
    id: str
    name: str
    execute: Callable                       # 正向执行函数
    compensate: Callable                    # 补偿函数
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    status: SagaStepStatus = SagaStepStatus.PENDING
    result: Any = None
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 30.0                   # 超时时间


class SagaOrchestrator:
    """Saga 编排器
    
    实现分布式事务的 Saga 模式：
    - 每个步骤都有对应的补偿操作
    - 任何步骤失败时，逆序执行已完成步骤的补偿操作
    - 支持重试和超时
    """
    
    def __init__(self, name: str = "saga"):
        self.name = name
        self._steps: list[SagaStep] = []
        self._status = SagaStatus.RUNNING
        self._execution_log: list[dict] = []
    
    def add_step(
        self,
        name: str,
        execute: Callable,
        compensate: Callable,
        args: tuple = (),
        kwargs: dict = None,
        max_retries: int = 3,
        timeout: float = 30.0
    ) -> "SagaOrchestrator":
        """添加 Saga 步骤"""
        step_id = f"step_{len(self._steps)}"
        
        step = SagaStep(
            id=step_id,
            name=name,
            execute=execute,
            compensate=compensate,
            args=args,
            kwargs=kwargs or {},
            max_retries=max_retries,
            timeout=timeout
        )
        
        self._steps.append(step)
        return self
    
    async def execute(self) -> dict:
        """执行 Saga 流程"""
        self._status = SagaStatus.RUNNING
        executed_steps = []
        
        for step in self._steps:
            step.status = SagaStepStatus.EXECUTING
            self._log("step_start", step)
            
            success = False
            last_error = None
            
            # 重试逻辑
            for attempt in range(step.max_retries):
                try:
                    if asyncio.iscoroutinefunction(step.execute):
                        result = await asyncio.wait_for(
                            step.execute(*step.args, **step.kwargs),
                            timeout=step.timeout
                        )
                    else:
                        result = step.execute(*step.args, **step.kwargs)
                    
                    step.result = result
                    step.status = SagaStepStatus.COMPLETED
                    success = True
                    executed_steps.append(step)
                    
                    self._log("step_complete", step)
                    break
                    
                except asyncio.TimeoutError:
                    last_error = f"步骤超时 ({step.timeout}s)"
                    step.retry_count += 1
                    self._log("step_retry", step, last_error)
                    
                except Exception as e:
                    last_error = str(e)
                    step.retry_count += 1
                    self._log("step_retry", step, last_error)
            
            if not success:
                step.error = last_error
                step.status = SagaStepStatus.FAILED
                self._log("step_failed", step, last_error)
                
                # 开始补偿流程
                self._status = SagaStatus.COMPENSATING
                await self._compensate(executed_steps)
                
                return {
                    "status": "failed",
                    "failed_step": step.name,
                    "error": last_error,
                    "compensated_steps": len(executed_steps),
                    "log": self._execution_log
                }
        
        self._status = SagaStatus.COMPLETED
        self._log("saga_complete", None)
        
        return {
            "status": "completed",
            "steps_executed": len(executed_steps),
            "log": self._execution_log
        }
    
    async def _compensate(
        self, executed_steps: list[SagaStep]
    ) -> None:
        """逆序执行补偿操作"""
        self._log("compensation_start", None)
        
        # 逆序补偿
        for step in reversed(executed_steps):
            step.status = SagaStepStatus.COMPENSATING
            self._log("compensating", step)
            
            try:
                if asyncio.iscoroutinefunction(step.compensate):
                    await asyncio.wait_for(
                        step.compensate(*step.args, **step.kwargs),
                        timeout=step.timeout
                    )
                else:
                    step.compensate(*step.args, **step.kwargs)
                
                step.status = SagaStepStatus.COMPENSATED
                self._log("compensated", step)
                
            except Exception as e:
                step.error = f"补偿失败: {e}"
                self._log("compensation_failed", step, str(e))
        
        self._status = SagaStatus.COMPENSATED
    
    def _log(
        self, event: str, step: SagaStep | None, error: str = None
    ) -> None:
        """记录执行日志"""
        entry = {
            "event": event,
            "timestamp": datetime.now().isoformat()
        }
        if step:
            entry["step"] = step.name
            entry["step_id"] = step.id
        if error:
            entry["error"] = error
        self._execution_log.append(entry)
    
    def get_status(self) -> dict:
        """获取 Saga 状态"""
        return {
            "name": self.name,
            "status": self._status.value,
            "steps": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "retries": s.retry_count
                }
                for s in self._steps
            ]
        }
```

### 30.6.2 补偿策略模式

```python
# 补偿策略模式
class CompensationStrategy(Enum):
    """补偿策略"""
    UNDO = "undo"               # 撤销操作
    REDO = "redo"               # 重新执行反向操作
    COMPENSATE = "compensate"   # 执行补偿逻辑
    IGNORE = "ignore"           # 忽略（幂等操作）


@dataclass
class CompensationAction:
    """补偿动作定义"""
    strategy: CompensationStrategy
    handler: Callable
    description: str = ""


class CompensationRegistry:
    """补偿注册表 - 管理各种操作的补偿策略"""
    
    def __init__(self):
        self._actions: dict[str, CompensationAction] = {}
    
    def register(
        self,
        operation: str,
        strategy: CompensationStrategy,
        handler: Callable,
        description: str = ""
    ) -> None:
        """注册补偿动作"""
        self._actions[operation] = CompensationAction(
            strategy=strategy,
            handler=handler,
            description=description
        )
    
    def get_compensator(
        self, operation: str
    ) -> CompensationAction | None:
        """获取补偿动作"""
        return self._actions.get(operation)
    
    @staticmethod
    def create_undo_compensator(
        original_fn: Callable,
        undo_fn: Callable
    ) -> Callable:
        """创建撤销型补偿器"""
        async def compensator(*args, **kwargs):
            return await undo_fn(*args, **kwargs)
        return compensator
    
    @staticmethod
    def create_idempotent_compensator() -> Callable:
        """创建幂等补偿器（无操作）"""
        async def compensator(*args, **kwargs):
            return {"compensated": True, "action": "noop"}
        return compensator


# 常见业务场景的补偿实现
class OrderCompensation:
    """订单相关操作的补偿器"""
    
    @staticmethod
    async def compensate_payment(
        order_id: str, amount: float
    ) -> dict:
        """补偿：退款"""
        print(f"[补偿] 退还订单 {order_id} 金额 {amount}")
        return {"refunded": True, "amount": amount}
    
    @staticmethod
    async def compensate_inventory(
        product_id: str, quantity: int
    ) -> dict:
        """补偿：恢复库存"""
        print(
            f"[补偿] 恢复商品 {product_id} 库存 {quantity} 件"
        )
        return {"restored": True, "quantity": quantity}
    
    @staticmethod
    async def compensate_shipping(
        shipment_id: str
    ) -> dict:
        """补偿：取消发货"""
        print(f"[补偿] 取消发货 {shipment_id}")
        return {"cancelled": True}
    
    @staticmethod
    async def compensate_notification(
        notification_id: str
    ) -> dict:
        """补偿：发送取消通知"""
        print(f"[补偿] 发送取消通知 {notification_id}")
        return {"notified": True}


# 使用示例
async def demo_saga():
    """演示 Saga 模式"""
    
    # 模拟业务操作
    async def create_order(order_id: str) -> dict:
        print(f"创建订单 {order_id}")
        return {"order_id": order_id, "status": "created"}
    
    async def process_payment(order_id: str, amount: float) -> dict:
        print(f"处理支付: {order_id}, 金额: {amount}")
        if amount > 10000:
            raise ValueError("金额超限，模拟支付失败")
        return {"payment_id": f"pay_{order_id}", "amount": amount}
    
    async def reserve_inventory(product_id: str, qty: int) -> dict:
        print(f"预留库存: {product_id}, 数量: {qty}")
        return {"reserved": True, "quantity": qty}
    
    async def create_shipment(order_id: str) -> dict:
        print(f"创建发货: {order_id}")
        return {"shipment_id": f"ship_{order_id}"}
    
    async def send_confirmation(order_id: str) -> dict:
        print(f"发送确认: {order_id}")
        return {"sent": True}
    
    # 创建 Saga
    saga = SagaOrchestrator("order_processing")
    
    saga.add_step(
        name="创建订单",
        execute=create_order,
        compensate=OrderCompensation.create_idempotent_compensator(),
        args=("ORD-001",)
    )
    
    saga.add_step(
        name="处理支付",
        execute=process_payment,
        compensate=OrderCompensation.compensate_payment,
        args=("ORD-001", 5000)
    )
    
    saga.add_step(
        name="预留库存",
        execute=reserve_inventory,
        compensate=OrderCompensation.compensate_inventory,
        args=("PROD-001", 2)
    )
    
    saga.add_step(
        name="创建发货",
        execute=create_shipment,
        compensate=OrderCompensation.compensate_shipping,
        args=("ORD-001",)
    )
    
    saga.add_step(
        name="发送确认",
        execute=send_confirmation,
        compensate=OrderCompensation.create_idempotent_compensator(),
        args=("ORD-001",)
    )
    
    # 执行 Saga
    result = await saga.execute()
    
    print(f"\nSaga 结果: {result['status']}")
    print(f"执行日志: {len(result['log'])} 条记录")
    print(f"状态: {saga.get_status()}")
```

## 30.7 完整工作流引擎

### 30.7.1 WorkflowEngine 整合

```python
# 完整的工作流引擎 - 整合所有编排能力
class WorkflowEngine:
    """工作流引擎 - 整合 DAG、并行、HITL、Saga"""
    
    def __init__(self, name: str):
        self.name = name
        self.hitl_engine = HITLEngine()
        self.parallel_executor = ParallelExecutor()
        self._workflows: dict[str, StateGraph] = {}
        self._running: dict[str, asyncio.Task] = {}
    
    async def start(self) -> None:
        """启动引擎"""
        await self.hitl_engine.start()
    
    async def stop(self) -> None:
        """停止引擎"""
        await self.hitl_engine.stop()
        for task in self._running.values():
            task.cancel()
    
    def register_workflow(
        self,
        name: str,
        graph: StateGraph
    ) -> None:
        """注册工作流"""
        self._workflows[name] = graph
    
    async def execute_workflow(
        self,
        name: str,
        initial_state: dict[str, Any]
    ) -> GraphState:
        """执行工作流"""
        graph = self._workflows.get(name)
        if not graph:
            raise ValueError(f"工作流 '{name}' 未找到")
        
        compiled = graph.compile()
        return await compiled.run(initial_state)
    
    async def execute_parallel_workflows(
        self,
        workflows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """并行执行多个工作流"""
        tasks = []
        for wf in workflows:
            task = ParallelTask(
                id=wf.get("name", str(uuid.uuid4())),
                name=wf.get("name", "unnamed"),
                coroutine=self.execute_workflow,
                args=(wf["name"], wf.get("initial_state", {}))
            )
            tasks.append(task)
        
        results = await self.parallel_executor.execute(tasks)
        return {task_id: task.result for task_id, task in results.items()}
    
    def get_status(self) -> dict:
        """获取引擎状态"""
        return {
            "name": self.name,
            "registered_workflows": list(self._workflows.keys()),
            "running_count": len(self._running),
            "pending_approvals": len(
                self.hitl_engine.get_pending_requests()
            )
        }
```

### 30.7.2 实战：订单处理工作流

```python
# 实战：完整的订单处理工作流
async def demo_order_workflow():
    """演示订单处理工作流"""
    
    # 创建工作流引擎
    engine = WorkflowEngine("order_system")
    await engine.start()
    
    # 定义状态图
    graph = StateGraph("order_processing")
    
    def validate_order(state: GraphState) -> GraphState:
        """验证订单"""
        order = state.get("order", {})
        if not order.get("items"):
            state.set("_error", "订单无商品")
            return state
        state.set("validated", True)
        state.set("total", sum(
            item.get("price", 0) * item.get("quantity", 0)
            for item in order.get("items", [])
        ))
        return state
    
    def check_inventory(state: GraphState) -> GraphState:
        """检查库存"""
        order = state.get("order", {})
        items = order.get("items", [])
        
        # 模拟库存检查
        in_stock = all(item.get("quantity", 0) <= 10 for item in items)
        state.set("inventory_ok", in_stock)
        
        return state
    
    def process_payment(state: GraphState) -> GraphState:
        """处理支付"""
        total = state.get("total", 0)
        state.set("payment_id", f"pay_{uuid.uuid4().hex[:8]}")
        state.set("payment_status", "completed")
        return state
    
    def create_shipment(state: GraphState) -> GraphState:
        """创建发货"""
        state.set("shipment_id", f"ship_{uuid.uuid4().hex[:8]}")
        state.set("shipment_status", "pending")
        return state
    
    def send_notification(state: GraphState) -> GraphState:
        """发送通知"""
        state.set("notification_sent", True)
        return state
    
    # 路由函数
    def route_after_inventory(state: GraphState) -> str:
        if state.get("inventory_ok"):
            return "payment"
        return "out_of_stock"
    
    # 构建图
    graph.add_node("validate", validate_order, "验证订单")
    graph.add_node("check_inventory", check_inventory, "检查库存")
    graph.add_node("payment", process_payment, "处理支付")
    graph.add_node("shipment", create_shipment, "创建发货")
    graph.add_node("notify", send_notification, "发送通知")
    
    graph.set_entry_point("validate")
    graph.add_edge("validate", "check_inventory")
    graph.add_conditional_edges(
        "check_inventory",
        route_after_inventory,
        {"payment": "payment", "out_of_stock": "error"}
    )
    graph.add_edge("payment", "shipment")
    graph.add_edge("shipment", "notify")
    
    # 注册并执行
    engine.register_workflow("order", graph)
    
    result = await engine.execute_workflow("order", {
        "order": {
            "id": "ORD-001",
            "items": [
                {"name": "商品A", "price": 100, "quantity": 2},
                {"name": "商品B", "price": 50, "quantity": 1}
            ]
        }
    })
    
    print(f"订单处理完成: {result.data}")
    print(f"引擎状态: {engine.get_status()}")
    
    await engine.stop()
```

## 30.8 本章小结

本章系统性地介绍了 Agent 工作流编排的核心技术：

1. **DAG 编排**：通过有向无环图管理任务依赖关系，支持拓扑排序和并行层级执行。

2. **StateGraph**：基于状态的工作流引擎，支持条件分支、动态路由和子图嵌套。

3. **并行处理**：利用 asyncio 实现 Fork-Join、Map-Reduce 等并行模式，提升执行效率。

4. **HITL 审批**：在工作流中嵌入人工审批节点，支持优先级、超时和升级机制。

5. **Saga 模式**：通过正向执行+反向补偿的策略，实现分布式事务的最终一致性。

## 30.9 思考题

1. 在 StateGraph 中，如何避免无限循环？请设计一个循环检测机制。

2. 比较 Fork-Join 和 Map-Reduce 模式的适用场景，各有什么优缺点？

3. 在 HITL 审批流程中，如果审批人在超时后才做出决定，系统应如何处理？

4. 设计一个支持嵌套 Saga 的补偿机制，即每个步骤的补偿操作本身也是一个 Saga。

5. 如何在工作流引擎中实现断点续传（Checkpoint & Resume）功能？

6. 在高并发场景下，如何保证工作流引擎的线程安全性？

7. 设计一个支持动态修改工作流定义的热更新机制。

8. 讨论工作流编排中的幂等性设计，如何确保重复执行不会产生副作用？
