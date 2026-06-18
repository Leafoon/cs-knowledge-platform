---
title: "第12章：高级规划算法 — ToT、GoT 与 Plan-and-Solve"
description: "掌握前沿规划算法：Tree of Thoughts、Graph of Thoughts、LATS、Self-Refine、Reflexion 的原理与完整实现。"
date: "2026-06-11"
---

 # 第12章：高级规划算法 — ToT、GoT 与 Plan-and-Solve

 高级规划算法将 Agent 的推理能力提升到新的层次。

Tree of Thoughts 是一种前沿的规划算法。下面的交互式可视化展示了 ToT 的搜索过程：

<div data-component="ToTVisualizer"></div>

不同的推理策略适用于不同的场景。下面的交互式工具可以帮助你选择最合适的推理策略：

<div data-component="ReasoningStrategySelectorV10"></div>

 ---

 ## 12.1 Tree of Thoughts (ToT)

```python
class TreeOfThoughts:
    def __init__(self, llm, breadth=3, depth=3):
        self.llm = llm
        self.breadth = breadth
        self.depth = depth

    def generate_thoughts(self, problem, current_path, n=3):
        prompt = f"问题：{problem}\n当前推理：{current_path or '(开始)'}\n生成 {n} 个可能的下一步："
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return [t.strip() for t in response.content.split("\n") if t.strip()][:n]

    def evaluate_thought(self, problem, thought):
        prompt = f"评估以下推理的质量（0-10分）：\n问题：{problem}\n推理：{thought}\n分数："
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try: return float(response.content.strip()) / 10
        except: return 0.5

    def search(self, problem, strategy="bfs"):
        root = ThoughtNode(content="", value=0, depth=0)
        if strategy == "bfs": return self._bfs(problem, root)
        else:
            best = {"value": 0, "content": ""}
            self._dfs(root, problem, best)
            return best["content"]
```

---

## 12.2 Graph of Thoughts (GoT)

```python
class GraphOfThoughts:
    def aggregate(self, thoughts):
        prompt = f"将以下思考整合为一个更强的洞察：\n{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(thoughts))}"
        return self.llm.invoke([HumanMessage(content=prompt)]).content

    def refine(self, thought, feedback):
        prompt = f"原思考：{thought}\n反馈：{feedback}\n改进后："
        return self.llm.invoke([HumanMessage(content=prompt)]).content
```

---

## 12.3 Reflexion

```python
class ReflexionAgent:
    def __init__(self, llm, tools, max_retries=3):
        self.llm = llm
        self.tools = tools
        self.max_retries = max_retries
        self.memory = []

    def run(self, task):
        for attempt in range(self.max_retries):
            result = self._execute_with_reflection(task)
            if self._evaluate(task, result): return result
            reflection = self._reflect(task, result, attempt)
            self.memory.append({"attempt": attempt, "reflection": reflection})
        return "达到最大重试次数"

    def _reflect(self, task, failed_result, attempt):
        prompt = f"任务：{task}\n失败结果：{failed_result[:500]}\n请反思根本原因和改进策略："
        return self.llm.invoke([HumanMessage(content=prompt)]).content
```

---

## 12.4 Self-Refine

```python
class SelfRefine:
    def __init__(self, llm, max_iterations=3):
        self.llm = llm
        self.max_iterations = max_iterations

    def run(self, task):
        current = self.llm.invoke([HumanMessage(content=f"请完成：{task}")]).content
        for _ in range(self.max_iterations):
            feedback = self.llm.invoke([HumanMessage(content=f"评估质量（PASS/改进点）：\n任务：{task}\n输出：{current[:1000]}")]).content
            if "PASS" in feedback.upper(): return current
            current = self.llm.invoke([HumanMessage(content=f"改进：\n任务：{task}\n输出：{current[:1000]}\n反馈：{feedback}")]).content
        return current
```

---

## 12.5 LATS

```python
class LATSAgent:
    def search(self, task, n_simulations=10):
        root = {"state": task, "children": [], "visits": 0, "value": 0}
        for _ in range(n_simulations):
            node = self._select(root)
            children = self._expand(node)
            for child in children:
                child["value"] = self._simulate(child)
                child["visits"] = 1
            self._backpropagate(root)
        return self._get_best_path(root)
```

---

## 12.6 算法对比

| 算法 | 搜索策略 | Token 效率 | 适用场景 |
|:---|:---|:---|:---|
| CoT | 线性 | 高 | 简单推理 |
| ToT | 树状 BFS/DFS | 中 | 多方案探索 |
| GoT | 图状 | 低 | 聚合洞察 |
| Reflexion | 循环迭代 | 中 | 从错误学习 |
| Self-Refine | 生成-改进 | 中 | 高质量输出 |
| LATS | MCTS | 低 | 复杂决策 |

---

## 12.7 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| ToT | 搜索树找最优路径 |
| GoT | 图状推理，支持聚合 |
| Reflexion | 执行-反思-改进 |
| Self-Refine | 生成后持续精炼 |
| LATS | MCTS + LLM |

> **下一章预告**
>
> 在第 13 章中，我们将系统梳理 Agent 架构设计模式。

---

## 12.8 Tree of Thoughts (ToT) 详解

### 12.8.1 ToT 核心思想

Tree of Thoughts (ToT) 是一种基于树状搜索的推理算法，它允许模型在多个推理路径中进行探索和选择，而不是线性地生成单一路径。

### 12.8.2 ToT 完整实现

```python
from typing import List, Dict, Optional, Tuple
import heapq

class ThoughtNode:
    def __init__(self, content: str, value: float = 0.0, depth: int = 0, parent=None):
        self.content = content
        self.value = value
        self.depth = depth
        self.parent = parent
        self.children = []

    def __lt__(self, other):
        return self.value < other.value

class TreeOfThoughts:
    def __init__(self, llm, breadth: int = 3, depth: int = 3, evaluation_strategy: str = "value"):
        self.llm = llm
        self.breadth = breadth
        self.depth = depth
        self.evaluation_strategy = evaluation_strategy

    def generate_thoughts(self, problem: str, current_path: str, n: int = 3) -> List[str]:
        """生成可能的下一步思考"""
        prompt = f"""
        问题：{problem}
        当前推理路径：{current_path or '(开始)'}

        请生成 {n} 个可能的下一步思考，每个思考应该是一个独立的推理步骤。
        思考列表：
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        thoughts = [t.strip() for t in response.content.split("\n") if t.strip()]
        return thoughts[:n]

    def evaluate_thought(self, problem: str, thought: str) -> float:
        """评估单个思考的质量"""
        if self.evaluation_strategy == "value":
            prompt = f"""
            评估以下推理步骤的质量（0-10分）：
            问题：{problem}
            推理步骤：{thought}

            请只输出一个数字分数："""
            response = self.llm.invoke([HumanMessage(content=prompt)])
            try:
                score = float(response.content.strip()) / 10
                return min(max(score, 0.0), 1.0)  # 限制在0-1之间
            except:
                return 0.5
        else:
            # 使用投票策略
            return self._vote_evaluation(problem, thought)

    def _vote_evaluation(self, problem: str, thought: str) -> float:
        """使用投票评估"""
        prompt = f"""
        对以下推理步骤进行投票评估：
        问题：{problem}
        推理步骤：{thought}

        请从以下选项中选择一个：
        A. 非常好 (1.0)
        B. 好 (0.8)
        C. 一般 (0.6)
        D. 差 (0.4)
        E. 非常差 (0.2)

        选择："""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        choice = response.content.strip().upper()
        scores = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2}
        return scores.get(choice, 0.5)

    def search(self, problem: str, strategy: str = "bfs") -> str:
        """执行ToT搜索"""
        root = ThoughtNode(content="", value=0, depth=0)

        if strategy == "bfs":
            return self._bfs(problem, root)
        elif strategy == "dfs":
            best = {"value": 0, "content": ""}
            self._dfs(root, problem, best)
            return best["content"]
        elif strategy == "beam":
            return self._beam_search(problem, root)
        else:
            raise ValueError(f"不支持的搜索策略: {strategy}")

    def _bfs(self, problem: str, root: ThoughtNode) -> str:
        """广度优先搜索"""
        queue = [root]
        best_leaf = root

        while queue:
            current = queue.pop(0)

            if current.depth >= self.depth:
                if current.value > best_leaf.value:
                    best_leaf = current
                continue

            # 生成子节点
            thoughts = self.generate_thoughts(problem, self._get_path(current))
            for thought in thoughts:
                child = ThoughtNode(
                    content=thought,
                    value=self.evaluate_thought(problem, thought),
                    depth=current.depth + 1,
                    parent=current
                )
                current.children.append(child)
                queue.append(child)

                # 更新最佳叶子节点
                if child.value > best_leaf.value:
                    best_leaf = child

        return self._get_path(best_leaf)

    def _dfs(self, node: ThoughtNode, problem: str, best: Dict):
        """深度优先搜索"""
        if node.depth >= self.depth:
            if node.value > best["value"]:
                best["value"] = node.value
                best["content"] = self._get_path(node)
            return

        # 生成子节点
        thoughts = self.generate_thoughts(problem, self._get_path(node))
        for thought in thoughts:
            child = ThoughtNode(
                content=thought,
                value=self.evaluate_thought(problem, thought),
                depth=node.depth + 1,
                parent=node
            )
            node.children.append(child)
            self._dfs(child, problem, best)

    def _beam_search(self, problem: str, root: ThoughtNode) -> str:
        """束搜索"""
        current_level = [root]

        for depth in range(self.depth):
            next_level = []

            for node in current_level:
                thoughts = self.generate_thoughts(problem, self._get_path(node))
                for thought in thoughts:
                    child = ThoughtNode(
                        content=thought,
                        value=self.evaluate_thought(problem, thought),
                        depth=depth + 1,
                        parent=node
                    )
                    node.children.append(child)
                    next_level.append(child)

            # 保留top-breadth个节点
            current_level = sorted(next_level, key=lambda x: x.value, reverse=True)[:self.breadth]

        # 返回最佳路径
        best_node = max(current_level, key=lambda x: x.value) if current_level else root
        return self._get_path(best_node)

    def _get_path(self, node: ThoughtNode) -> str:
        """获取从根到节点的路径"""
        path = []
        current = node
        while current:
            if current.content:
                path.append(current.content)
            current = current.parent
        return " -> ".join(reversed(path))
```

### 12.8.3 ToT 的优化技巧

1. **剪枝策略**：提前终止低质量路径的探索。
2. **启发式评估**：使用更精确的评估函数。
3. **并行搜索**：同时探索多个路径。
4. **记忆化**：缓存已评估的节点。

---

## 12.9 Graph of Thoughts (GoT) 详解

### 12.9.1 GoT 核心思想

Graph of Thoughts 将推理过程建模为一个图结构，允许思考之间的聚合、精炼和链接，支持更复杂的推理模式。

### 12.9.2 GoT 完整实现

```python
import networkx as nx
from typing import Set, List, Dict, Any

class GraphOfThoughts:
    def __init__(self, llm):
        self.llm = llm
        self.graph = nx.DiGraph()
        self.thought_counter = 0

    def add_thought(self, content: str, thought_type: str = "initial") -> int:
        """添加思考节点"""
        thought_id = self.thought_counter
        self.thought_counter += 1

        self.graph.add_node(thought_id, content=content, type=thought_type)
        return thought_id

    def link_thoughts(self, from_id: int, to_id: int, relationship: str = "leads_to"):
        """链接两个思考"""
        self.graph.add_edge(from_id, to_id, relationship=relationship)

    def aggregate(self, thought_ids: List[int]) -> str:
        """聚合多个思考"""
        if not thought_ids:
            return ""

        # 收集所有思考内容
        thoughts = []
        for tid in thought_ids:
            if tid in self.graph.nodes:
                thoughts.append(self.graph.nodes[tid]["content"])

        # 聚合提示
        prompt = f"""
        将以下思考整合为一个更强的洞察：
        {chr(10).join(f'{i+1}. {t}' for i, t in enumerate(thoughts))}

        整合后的洞察："""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        aggregated_content = response.content

        # 添加聚合节点
        agg_id = self.add_thought(aggregated_content, "aggregated")

        # 链接到原始思考
        for tid in thought_ids:
            self.link_thoughts(tid, agg_id, "aggregated_to")

        return aggregated_content

    def refine(self, thought_id: int, feedback: str) -> str:
        """精炼单个思考"""
        if thought_id not in self.graph.nodes:
            return ""

        original = self.graph.nodes[thought_id]["content"]

        prompt = f"""
        原思考：{original}
        反馈：{feedback}

        请根据反馈改进这个思考："""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        refined_content = response.content

        # 添加精炼节点
        refined_id = self.add_thought(refined_content, "refined")
        self.link_thoughts(thought_id, refined_id, "refined_from")

        return refined_content

    def generate_thought(self, context: List[int], problem: str) -> str:
        """基于上下文生成新思考"""
        # 收集上下文思考
        context_content = []
        for tid in context:
            if tid in self.graph.nodes:
                context_content.append(self.graph.nodes[tid]["content"])

        prompt = f"""
        问题：{problem}
        上下文思考：
        {chr(10).join(f'- {c}' for c in context_content)}

        基于以上思考，生成一个新的洞察："""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        new_content = response.content

        # 添加新思考
        new_id = self.add_thought(new_content, "generated")

        # 链接到上下文
        for tid in context:
            self.link_thoughts(tid, new_id, "context_for")

        return new_content

    def solve(self, problem: str, max_iterations: int = 5) -> str:
        """使用GoT解决问题"""
        # 初始化
        initial_id = self.add_thought(f"问题：{problem}", "problem")
        current_thoughts = [initial_id]

        for iteration in range(max_iterations):
            # 生成新思考
            new_thought = self.generate_thought(current_thoughts, problem)
            new_id = self.add_thought(new_thought, "iteration")

            # 评估思考质量
            quality = self._evaluate_thought(new_id, problem)

            if quality > 0.8:
                # 高质量思考，尝试聚合
                if len(current_thoughts) > 1:
                    aggregated = self.aggregate(current_thoughts + [new_id])
                    # 替换为聚合结果
                    current_thoughts = [self.add_thought(aggregated, "aggregated")]
                else:
                    current_thoughts.append(new_id)
            else:
                # 低质量思考，精炼
                refined = self.refine(new_id, "需要更深入的分析")
                current_thoughts.append(self.add_thought(refined, "refined"))

        # 返回最佳思考
        best_thought = self._get_best_thought(current_thoughts, problem)
        return self.graph.nodes[best_thought]["content"]

    def _evaluate_thought(self, thought_id: int, problem: str) -> float:
        """评估思考质量"""
        content = self.graph.nodes[thought_id]["content"]
        prompt = f"""
        评估以下思考对解决问题的质量（0-1）：
        问题：{problem}
        思考：{content}

        质量分数："""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            return float(response.content.strip())
        except:
            return 0.5

    def _get_best_thought(self, thought_ids: List[int], problem: str) -> int:
        """获取最佳思考"""
        best_id = thought_ids[0]
        best_score = 0

        for tid in thought_ids:
            score = self._evaluate_thought(tid, problem)
            if score > best_score:
                best_score = score
                best_id = tid

        return best_id
```

### 12.9.3 GoT 的高级操作

1. **聚合操作**：将多个思考合并为一个更强的洞察。
2. **精炼操作**：根据反馈改进思考质量。
3. **分裂操作**：将复杂思考分解为更简单的部分。
4. **路由操作**：根据条件选择不同的推理路径。

---

## 12.10 Reflexion 深度解析

### 12.10.1 Reflexion 原理

Reflexion 是一种通过反思来改进推理的算法，它从失败中学习，逐步提升推理质量。

### 12.10.2 Reflexion 完整实现

```python
from typing import List, Dict, Any, Optional
import json

class ReflexionAgent:
    def __init__(self, llm, tools: List = None, max_retries: int = 3):
        self.llm = llm
        self.tools = tools or []
        self.max_retries = max_retries
        self.memory: List[Dict[str, Any]] = []
        self.evaluation_history: List[Dict] = []

    def run(self, task: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """执行任务并进行反思"""
        best_result = None
        best_score = -1

        for attempt in range(self.max_retries):
            # 执行任务
            result = self._execute_task(task)

            # 评估结果
            score = self._evaluate_result(task, result, ground_truth)

            # 记录评估
            self.evaluation_history.append({
                "attempt": attempt,
                "result": result,
                "score": score,
                "task": task
            })

            # 更新最佳结果
            if score > best_score:
                best_score = score
                best_result = result

            # 检查是否成功
            if score >= 0.9:  # 成功阈值
                return {
                    "success": True,
                    "result": best_result,
                    "attempts": attempt + 1,
                    "final_score": score
                }

            # 反思失败原因
            if attempt < self.max_retries - 1:
                reflection = self._reflect(task, result, score, attempt)
                self.memory.append({
                    "task": task,
                    "attempt": attempt,
                    "reflection": reflection,
                    "result": result
                })

        return {
            "success": False,
            "result": best_result,
            "attempts": self.max_retries,
            "final_score": best_score,
            "memory": self.memory
        }

    def _execute_task(self, task: str) -> str:
        """执行任务"""
        # 构建提示，包含历史反思
        prompt = self._build_prompt(task)

        # 调用LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _build_prompt(self, task: str) -> str:
        """构建包含反思历史的提示"""
        prompt_parts = [f"任务：{task}"]

        # 添加历史反思
        if self.memory:
            prompt_parts.append("\n历史反思：")
            for mem in self.memory[-3:]:  # 只使用最近3次反思
                prompt_parts.append(f"- 尝试 {mem['attempt']}: {mem['reflection']}")

        prompt_parts.append("\n请执行任务：")
        return "\n".join(prompt_parts)

    def _evaluate_result(self, task: str, result: str, ground_truth: Optional[str] = None) -> float:
        """评估结果质量"""
        if ground_truth:
            # 有标准答案时，直接比较
            return self._compare_with_ground_truth(result, ground_truth)
        else:
            # 无标准答案时，使用LLM评估
            return self._llm_evaluate(task, result)

    def _compare_with_ground_truth(self, result: str, ground_truth: str) -> float:
        """与标准答案比较"""
        # 简单字符串匹配
        if result.strip() == ground_truth.strip():
            return 1.0
        # 计算相似度
        similarity = self._compute_similarity(result, ground_truth)
        return similarity

    def _llm_evaluate(self, task: str, result: str) -> float:
        """使用LLM评估结果"""
        prompt = f"""
        评估以下任务执行结果的质量（0-1）：
        任务：{task}
        结果：{result}

        评估标准：
        1. 准确性
        2. 完整性
        3. 可读性

        质量分数："""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            return float(response.content.strip())
        except:
            return 0.5

    def _reflect(self, task: str, result: str, score: float, attempt: int) -> str:
        """反思失败原因"""
        prompt = f"""
        任务：{task}
        执行结果：{result[:500]}...
        质量分数：{score}
        尝试次数：{attempt + 1}

        请反思：
        1. 结果的主要问题是什么？
        2. 改进策略是什么？
        3. 下次应该如何避免类似错误？"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 简单实现：基于共同词汇
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
```

### 12.10.3 Reflexion 的应用场景

1. **代码生成**：从编译错误中学习。
2. **数学推理**：从错误答案中反思。
3. **对话系统**：从用户反馈中改进。
4. **决策系统**：从失败决策中学习。

---

## 12.11 Self-Refine 深度解析

### 12.11.1 Self-Refine 原理

Self-Refine 是一种迭代改进算法，通过生成-评估-改进的循环来提升输出质量。

### 12.11.2 Self-Refine 完整实现

```python
from typing import Callable, Optional, Dict, Any

class SelfRefine:
    def __init__(self, llm, max_iterations: int = 3, improvement_threshold: float = 0.1):
        self.llm = llm
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.iteration_history: List[Dict[str, Any]] = []

    def run(self, task: str, initial_output: Optional[str] = None) -> Dict[str, Any]:
        """执行Self-Refine"""
        # 初始生成
        if initial_output:
            current_output = initial_output
        else:
            current_output = self._initial_generation(task)

        current_score = self._evaluate_output(task, current_output)

        # 迭代改进
        for iteration in range(self.max_iterations):
            # 生成反馈
            feedback = self._generate_feedback(task, current_output, current_score)

            # 检查是否达到停止条件
            if "PASS" in feedback.upper():
                return {
                    "output": current_output,
                    "score": current_score,
                    "iterations": iteration,
                    "feedback_history": self.iteration_history
                }

            # 改进输出
            improved_output = self._improve_output(task, current_output, feedback)

            # 评估改进后的输出
            new_score = self._evaluate_output(task, improved_output)

            # 记录迭代
            self.iteration_history.append({
                "iteration": iteration,
                "input": current_output,
                "feedback": feedback,
                "output": improved_output,
                "old_score": current_score,
                "new_score": new_score
            })

            # 检查改进是否显著
            if new_score - current_score < self.improvement_threshold:
                # 改进不显著，停止
                break

            # 更新当前输出
            current_output = improved_output
            current_score = new_score

        return {
            "output": current_output,
            "score": current_score,
            "iterations": self.max_iterations,
            "feedback_history": self.iteration_history
        }

    def _initial_generation(self, task: str) -> str:
        """初始生成"""
        prompt = f"请完成以下任务：\n{task}"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _evaluate_output(self, task: str, output: str) -> float:
        """评估输出质量"""
        prompt = f"""
        评估以下输出的质量（0-1）：
        任务：{task}
        输出：{output[:1000]}

        评估标准：
        1. 相关性
        2. 准确性
        3. 完整性
        4. 可读性

        质量分数："""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            return float(response.content.strip())
        except:
            return 0.5

    def _generate_feedback(self, task: str, output: str, score: float) -> str:
        """生成改进反馈"""
        prompt = f"""
        任务：{task}
        当前输出：{output[:1000]}
        当前质量分数：{score}

        请提供具体的改进建议。如果输出质量足够好（分数>0.8），请回复"PASS"。
        否则，请提供详细的改进点："""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _improve_output(self, task: str, current_output: str, feedback: str) -> str:
        """改进输出"""
        prompt = f"""
        任务：{task}
        当前输出：{current_output[:1000]}
        改进建议：{feedback}

        请根据建议改进输出："""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
```

### 12.11.3 Self-Refine 的变体

1. **多轮Self-Refine**：多次迭代改进。
2. **并行Self-Refine**：同时尝试多种改进方向。
3. **条件Self-Refine**：根据条件决定是否继续改进。
4. **自适应Self-Refine**：动态调整改进策略。

---

## 12.12 LATS (Language Agent Tree Search) 深度解析

### 12.12.1 LATS 原理

LATS 将蒙特卡洛树搜索 (MCTS) 与语言模型结合，用于复杂决策和规划问题。

### 12.12.2 LATS 完整实现

```python
import math
from typing import List, Dict, Any, Optional

class LATSAgent:
    def __init__(self, llm, exploration_weight: float = 1.0, max_simulations: int = 10):
        self.llm = llm
        self.exploration_weight = exploration_weight
        self.max_simulations = max_simulations
        self.tree = {}

    def search(self, task: str, n_simulations: int = None) -> Dict[str, Any]:
        """执行LATS搜索"""
        if n_simulations is None:
            n_simulations = self.max_simulations

        # 初始化根节点
        root = self._create_node(task, parent=None)

        for _ in range(n_simulations):
            # 选择
            node = self._select(root)

            # 扩展
            children = self._expand(node)

            # 模拟
            for child in children:
                value = self._simulate(child)
                child["value"] = value
                child["visits"] = 1

            # 反向传播
            self._backpropagate(root)

        # 获取最佳路径
        return self._get_best_path(root)

    def _create_node(self, state: str, parent: Optional[Dict] = None) -> Dict[str, Any]:
        """创建节点"""
        node = {
            "state": state,
            "parent": parent,
            "children": [],
            "visits": 0,
            "value": 0.0,
            "depth": (parent["depth"] + 1) if parent else 0
        }
        return node

    def _select(self, node: Dict) -> Dict:
        """选择节点（UCT算法）"""
        if not node["children"]:
            return node

        # 使用UCT公式选择子节点
        best_child = None
        best_score = -float('inf')

        for child in node["children"]:
            if child["visits"] == 0:
                # 未访问过的节点优先
                score = float('inf')
            else:
                # UCT公式
                exploitation = child["value"] / child["visits"]
                exploration = self.exploration_weight * math.sqrt(
                    math.log(node["visits"]) / child["visits"]
                )
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return self._select(best_child) if best_child else node

    def _expand(self, node: Dict) -> List[Dict]:
        """扩展节点"""
        # 生成子节点
        prompt = f"""
        当前状态：{node['state']}
        请生成3个可能的下一步行动："""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        actions = [a.strip() for a in response.content.split("\n") if a.strip()]

        children = []
        for action in actions[:3]:  # 限制为3个子节点
            child_state = f"{node['state']} -> {action}"
            child = self._create_node(child_state, parent=node)
            node["children"].append(child)
            children.append(child)

        return children

    def _simulate(self, node: Dict) -> float:
        """模拟节点价值"""
        # 使用LLM评估当前状态的价值
        prompt = f"""
        评估以下状态的价值（0-1）：
        状态：{node['state']}

        价值分数："""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            return float(response.content.strip())
        except:
            return 0.5

    def _backpropagate(self, node: Dict):
        """反向传播更新节点价值"""
        if node["parent"]:
            # 更新父节点
            node["parent"]["visits"] += 1
            node["parent"]["value"] += node["value"]

            # 递归更新
            self._backpropagate(node["parent"])

    def _get_best_path(self, root: Dict) -> Dict[str, Any]:
        """获取最佳路径"""
        if not root["children"]:
            return {"path": root["state"], "value": root["value"]}

        # 选择访问次数最多的子节点
        best_child = max(root["children"], key=lambda x: x["visits"])

        # 递归获取路径
        sub_path = self._get_best_path(best_child)

        return {
            "path": f"{root['state']} -> {sub_path['path']}",
            "value": sub_path["value"],
            "visits": root["visits"]
        }
```

### 12.12.3 LATS 的优化策略

1. **并行模拟**：同时进行多个模拟。
2. **启发式剪枝**：剪枝低质量分支。
3. **记忆化搜索**：缓存已搜索过的状态。
4. **自适应探索**：动态调整探索权重。

---

## 12.13 高级规划算法对比分析

### 12.13.1 算法特性对比

| 算法 | 搜索策略 | 记忆机制 | 评估方式 | 适用场景 | 计算复杂度 |
|:---|:---|:---|:---|:---|:---|
| CoT | 线性 | 无 | 单次评估 | 简单推理 | O(n) |
| ToT | 树状BFS/DFS | 路径记忆 | 多路径评估 | 多方案探索 | O(b^d) |
| GoT | 图状 | 图结构记忆 | 聚合评估 | 复杂洞察 | O(V+E) |
| Reflexion | 循环迭代 | 经验记忆 | 重复评估 | 从错误学习 | O(k*n) |
| Self-Refine | 生成-改进 | 迭代记忆 | 迭代评估 | 高质量输出 | O(k*n) |
| LATS | MCTS | 树+价值记忆 | 模拟评估 | 复杂决策 | O(n^2) |

### 12.13.2 性能对比

```python
class AlgorithmBenchmark:
    def __init__(self):
        self.algorithms = {
            "CoT": CoTAlgorithm(),
            "ToT": TreeOfThoughtsAlgorithm(),
            "GoT": GraphOfThoughtsAlgorithm(),
            "Reflexion": ReflexionAlgorithm(),
            "Self-Refine": SelfRefineAlgorithm(),
            "LATS": LATSAlgorithm()
        }

    def benchmark(self, tasks: List[str], metrics: List[str]) -> Dict:
        """运行基准测试"""
        results = {}

        for algo_name, algorithm in self.algorithms.items():
            algo_results = []
            for task in tasks:
                # 执行算法
                start_time = time.time()
                result = algorithm.solve(task)
                end_time = time.time()

                # 计算指标
                task_result = {
                    "task": task,
                    "result": result,
                    "time": end_time - start_time
                }

                # 计算其他指标
                for metric in metrics:
                    if metric == "accuracy":
                        task_result[metric] = self._compute_accuracy(result, task)
                    elif metric == "quality":
                        task_result[metric] = self._compute_quality(result, task)

                algo_results.append(task_result)

            results[algo_name] = algo_results

        return self._analyze_results(results)

    def _analyze_results(self, results: Dict) -> Dict:
        """分析结果"""
        analysis = {}
        for algo_name, algo_results in results.items():
            analysis[algo_name] = {
                "avg_time": sum(r["time"] for r in algo_results) / len(algo_results),
                "avg_accuracy": sum(r.get("accuracy", 0) for r in algo_results) / len(algo_results),
                "avg_quality": sum(r.get("quality", 0) for r in algo_results) / len(algo_results)
            }
        return analysis
```

### 12.13.3 算法选择指南

| 场景 | 推荐算法 | 理由 |
|:---|:---|:---|
| 简单问答 | CoT | 快速、高效 |
| 多方案探索 | ToT | 支持并行探索 |
| 复杂洞察生成 | GoT | 支持聚合和精炼 |
| 从错误学习 | Reflexion | 自我反思机制 |
| 高质量输出 | Self-Refine | 迭代改进 |
| 复杂决策 | LATS | MCTS优化 |

---

## 12.14 高级规划算法的工程实践

### 12.14.1 算法选择框架

```python
class AlgorithmSelector:
    def __init__(self):
        self.algorithms = {
            "cot": CoTAlgorithm(),
            "tot": TreeOfThoughtsAlgorithm(),
            "got": GraphOfThoughtsAlgorithm(),
            "reflexion": ReflexionAlgorithm(),
            "self_refine": SelfRefineAlgorithm(),
            "lats": LATSAlgorithm()
        }

        self.selection_criteria = {
            "task_complexity": ["low", "medium", "high"],
            "quality_requirement": ["low", "medium", "high"],
            "time_constraint": ["loose", "medium", "tight"]
        }

    def select_algorithm(self, task_info: Dict) -> str:
        """根据任务信息选择算法"""
        complexity = task_info.get("complexity", "medium")
        quality = task_info.get("quality", "medium")
        time = task_info.get("time", "medium")

        # 决策逻辑
        if complexity == "low":
            return "cot"
        elif complexity == "medium" and quality == "high":
            return "self_refine"
        elif complexity == "high" and time == "loose":
            return "lats"
        elif quality == "high":
            return "got"
        else:
            return "tot"
```

### 12.14.2 混合算法策略

```python
class HybridAlgorithm:
    def __init__(self):
        self.algorithms = {
            "initial": CoTAlgorithm(),
            "refinement": SelfRefineAlgorithm(),
            "exploration": TreeOfThoughtsAlgorithm()
        }

    def solve(self, task: str, strategy: str = "sequential") -> str:
        """混合算法解决"""
        if strategy == "sequential":
            return self._sequential_solve(task)
        elif strategy == "parallel":
            return self._parallel_solve(task)
        else:
            return self._adaptive_solve(task)

    def _sequential_solve(self, task: str) -> str:
        """顺序执行算法"""
        # 第一步：初始生成
        initial = self.algorithms["initial"].solve(task)

        # 第二步：精炼
        refined = self.algorithms["refinement"].solve(task, initial_output=initial)

        # 第三步：探索最佳方案
        final = self.algorithms["exploration"].solve(refined)

        return final

    def _parallel_solve(self, task: str) -> str:
        """并行执行算法"""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                name: executor.submit(algo.solve, task)
                for name, algo in self.algorithms.items()
            }

            results = {}
            for name, future in futures.items():
                results[name] = future.result()

        # 选择最佳结果
        return self._select_best(results)

    def _select_best(self, results: Dict) -> str:
        """选择最佳结果"""
        # 简单实现：选择最长的结果
        return max(results.values(), key=len)
```

### 12.14.3 错误处理与恢复

```python
class RobustAlgorithm:
    def __init__(self, algorithm, fallback_algorithm=None):
        self.algorithm = algorithm
        self.fallback = fallback_algorithm or CoTAlgorithm()
        self.max_retries = 3

    def solve_with_recovery(self, task: str) -> str:
        """带错误恢复的解决"""
        for attempt in range(self.max_retries):
            try:
                result = self.algorithm.solve(task)
                # 验证结果
                if self._validate_result(result, task):
                    return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # 使用回退算法
                    return self.fallback.solve(task)

        return self.fallback.solve(task)

    def _validate_result(self, result: str, task: str) -> bool:
        """验证结果"""
        # 基本验证：结果不为空且长度合理
        return bool(result) and len(result) > 10
```

---

## 12.15 高级规划算法的评估

### 12.15.1 评估指标体系

```python
class AdvancedEvaluationMetrics:
    def __init__(self):
        self.metrics = {
            "accuracy": self._compute_accuracy,
            "quality": self._compute_quality,
            "efficiency": self._compute_efficiency,
            "robustness": self._compute_robustness,
            "explainability": self._compute_explainability
        }

    def evaluate(self, algorithm, tasks: List[str], ground_truths: List[str] = None) -> Dict:
        """评估算法"""
        results = {}

        for metric_name, metric_func in self.metrics.items():
            metric_values = []
            for i, task in enumerate(tasks):
                ground_truth = ground_truths[i] if ground_truths else None
                value = metric_func(algorithm, task, ground_truth)
                metric_values.append(value)

            results[metric_name] = {
                "mean": sum(metric_values) / len(metric_values),
                "min": min(metric_values),
                "max": max(metric_values),
                "std": self._compute_std(metric_values)
            }

        return results

    def _compute_accuracy(self, algorithm, task: str, ground_truth: str = None) -> float:
        """计算准确率"""
        result = algorithm.solve(task)
        if ground_truth:
            return self._compare_results(result, ground_truth)
        return self._self_evaluate(result, task)

    def _compute_quality(self, algorithm, task: str, ground_truth: str = None) -> float:
        """计算质量"""
        result = algorithm.solve(task)
        return self._assess_quality(result, task)

    def _compute_efficiency(self, algorithm, task: str, ground_truth: str = None) -> float:
        """计算效率"""
        import time
        start_time = time.time()
        algorithm.solve(task)
        end_time = time.time()

        # 归一化效率（假设10秒为基准）
        efficiency = 1.0 - min((end_time - start_time) / 10.0, 1.0)
        return efficiency

    def _compute_robustness(self, algorithm, task: str, ground_truth: str = None) -> float:
        """计算鲁棒性"""
        # 通过多次运行测试稳定性
        results = []
        for _ in range(5):
            result = algorithm.solve(task)
            results.append(result)

        # 计算结果一致性
        unique_results = set(results)
        robustness = 1.0 - (len(unique_results) - 1) / 4.0  # 归一化
        return max(robustness, 0.0)

    def _compute_explainability(self, algorithm, task: str, ground_truth: str = None) -> float:
        """计算可解释性"""
        result = algorithm.solve(task)
        # 简单评估：结果长度和结构
        explainability = min(len(result) / 1000.0, 1.0)  # 假设1000字符为满分
        return explainability
```

### 12.15.2 A/B 测试框架

```python
class AlgorithmABTest:
    def __init__(self):
        self.experiments = {}

    def create_experiment(self, name: str, algorithm_a, algorithm_b, traffic_split: float = 0.5):
        """创建A/B测试"""
        self.experiments[name] = {
            "algorithm_a": algorithm_a,
            "algorithm_b": algorithm_b,
            "traffic_split": traffic_split,
            "results_a": [],
            "results_b": [],
            "start_time": time.time()
        }

    def run_experiment(self, name: str, tasks: List[str]):
        """运行实验"""
        experiment = self.experiments[name]

        for task in tasks:
            # 随机分配
            if random.random() < experiment["traffic_split"]:
                # 算法A
                result = experiment["algorithm_a"].solve(task)
                experiment["results_a"].append({"task": task, "result": result})
            else:
                # 算法B
                result = experiment["algorithm_b"].solve(task)
                experiment["results_b"].append({"task": task, "result": result})

        return self._analyze_experiment(name)

    def _analyze_experiment(self, name: str) -> Dict:
        """分析实验结果"""
        experiment = self.experiments[name]

        # 计算指标
        metrics_a = self._compute_metrics(experiment["results_a"])
        metrics_b = self._compute_metrics(experiment["results_b"])

        # 统计显著性测试
        significance = self._compute_significance(metrics_a, metrics_b)

        return {
            "algorithm_a": metrics_a,
            "algorithm_b": metrics_b,
            "significance": significance,
            "winner": "A" if metrics_a["quality"] > metrics_b["quality"] else "B"
        }
```

---

## 12.16 高级规划算法的部署

### 12.16.1 生产环境架构

```python
class ProductionPlanningService:
    def __init__(self):
        self.algorithms = {
            "cot": CoTAlgorithm(),
            "tot": TreeOfThoughtsAlgorithm(),
            "got": GraphOfThoughtsAlgorithm(),
            "reflexion": ReflexionAlgorithm(),
            "self_refine": SelfRefineAlgorithm(),
            "lats": LATSAlgorithm()
        }

        self.cache = {}
        self.monitoring = MonitoringSystem()

    def plan(self, task: str, algorithm: str = "auto") -> Dict[str, Any]:
        """规划接口"""
        # 检查缓存
        cache_key = self._get_cache_key(task, algorithm)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 选择算法
        if algorithm == "auto":
            algorithm = self._select_algorithm(task)

        # 执行规划
        start_time = time.time()
        result = self.algorithms[algorithm].solve(task)
        end_time = time.time()

        # 构建响应
        response = {
            "result": result,
            "algorithm": algorithm,
            "execution_time": end_time - start_time,
            "timestamp": time.time()
        }

        # 缓存结果
        self.cache[cache_key] = response

        # 监控记录
        self.monitoring.record("plan_execution", response)

        return response

    def _select_algorithm(self, task: str) -> str:
        """自动选择算法"""
        # 基于任务特征选择
        task_length = len(task)
        if task_length < 100:
            return "cot"
        elif task_length < 500:
            return "tot"
        else:
            return "lats"
```

### 12.16.2 监控与告警

```python
class MonitoringSystem:
    def __init__(self):
        self.metrics = {}
        self.alerts = []

    def record(self, metric_name: str, value: Any):
        """记录指标"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

        self.metrics[metric_name].append({
            "value": value,
            "timestamp": time.time()
        })

        # 检查告警条件
        self._check_alerts(metric_name, value)

    def _check_alerts(self, metric_name: str, value: Any):
        """检查告警"""
        if metric_name == "plan_execution":
            execution_time = value.get("execution_time", 0)
            if execution_time > 30:  # 超过30秒
                self.alerts.append({
                    "type": "slow_execution",
                    "message": f"规划执行时间过长: {execution_time:.2f}秒",
                    "timestamp": time.time()
                })

    def get_dashboard(self) -> Dict:
        """获取监控仪表板"""
        return {
            "metrics": self.metrics,
            "alerts": self.alerts,
            "summary": self._generate_summary()
        }

    def _generate_summary(self) -> Dict:
        """生成摘要"""
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    "count": len(values),
                    "latest": values[-1]["value"],
                    "average": self._compute_average(values)
                }
        return summary
```

---

## 12.17 本章总结与展望

### 12.17.1 核心算法回顾

本章深入介绍了六种高级规划算法：

1. **Tree of Thoughts (ToT)**：树状搜索，支持多路径探索。
2. **Graph of Thoughts (GoT)**：图状推理，支持聚合和精炼。
3. **Reflexion**：通过反思从错误中学习。
4. **Self-Refine**：迭代改进输出质量。
5. **LATS**：结合MCTS的复杂决策算法。

### 12.17.2 实践建议

1. **算法选择**：根据任务特性选择合适的算法。
2. **混合使用**：结合多种算法的优势。
3. **持续评估**：建立完善的评估体系。
4. **生产部署**：考虑性能、稳定性和可维护性。

### 12.17.3 未来发展方向

1. **多模态规划**：结合视觉、语音等多种模态。
2. **自适应算法**：根据环境动态调整算法策略。
3. **群体智能**：多Agent协作规划。
4. **可解释规划**：提高规划过程的可解释性。

掌握本章内容，你将能够设计和实现各种复杂的规划算法，为构建强大的AI Agent系统奠定坚实基础。

> **下一章预告**
>
> 在第 13 章中，我们将系统梳理 Agent 架构设计模式，包括单Agent、多Agent、微服务等架构。

---

## 12.18 高级规划算法的前沿研究

### 12.18.1 自监督规划算法

自监督规划算法通过数据增强和对比学习来提升规划能力：

```python
class SelfSupervisedPlanning:
    def __init__(self, llm):
        self.llm = llm
        self.contrastive_pairs = []

    def generate_contrastive_examples(self, task: str) -> List[Dict]:
        """生成对比学习样本"""
        # 生成正例（正确规划）
        positive_plan = self._generate_positive_plan(task)

        # 生成负例（错误规划）
        negative_plans = self._generate_negative_plans(task, n=3)

        # 构建对比对
        contrastive_pairs = []
        for neg_plan in negative_plans:
            contrastive_pairs.append({
                "task": task,
                "positive": positive_plan,
                "negative": neg_plan
            })

        return contrastive_pairs

    def _generate_positive_plan(self, task: str) -> str:
        """生成正确规划"""
        prompt = f"为以下任务生成正确的规划：\n{task}"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _generate_negative_plans(self, task: str, n: int = 3) -> List[str]:
        """生成错误规划"""
        negative_plans = []
        for i in range(n):
            prompt = f"""
            为以下任务生成一个有缺陷的规划（缺陷类型：{['逻辑错误', '遗漏步骤', '顺序错误'][i % 3]}）：
            {task}"""
            response = self.llm.invoke([HumanMessage(content=prompt)])
            negative_plans.append(response.content)
        return negative_plans

    def train_contrastive(self, tasks: List[str]):
        """训练对比学习模型"""
        for task in tasks:
            pairs = self.generate_contrastive_examples(task)
            self.contrastive_pairs.extend(pairs)

        # 这里可以接入对比学习训练
        print(f"生成了 {len(self.contrastive_pairs)} 个对比学习样本")
```

### 12.18.2 元学习规划

元学习规划使规划器能够快速适应新任务：

```python
class MetaPlanning:
    def __init__(self, base_planner):
        self.base_planner = base_planner
        self.meta_knowledge = {}

    def meta_train(self, task_distribution: List[List[str]]):
        """元训练"""
        for task_set in task_distribution:
            # 在任务集上训练
            self._adapt_to_task_set(task_set)

        return self.meta_knowledge

    def _adapt_to_task_set(self, task_set: List[str]):
        """适应任务集"""
        # 提取任务特征
        features = self._extract_features(task_set)

        # 学习任务特征与规划策略的关系
        self.meta_knowledge[features] = self._learn_strategy(task_set)

    def fast_adapt(self, new_task: str, n_adaptations: int = 5) -> str:
        """快速适应新任务"""
        # 提取新任务特征
        new_features = self._extract_features([new_task])

        # 查找最相似的元知识
        similar_strategy = self._find_similar_strategy(new_features)

        # 应用策略
        adapted_plan = self._apply_strategy(new_task, similar_strategy)

        # 小样本适应
        for _ in range(n_adaptations):
            adapted_plan = self._refine_plan(new_task, adapted_plan)

        return adapted_plan
```

### 12.18.3 多目标规划

多目标规划同时考虑多个优化目标：

```python
class MultiObjectivePlanning:
    def __init__(self, llm):
        self.llm = llm
        self.objectives = []

    def add_objective(self, objective: str, weight: float = 1.0):
        """添加优化目标"""
        self.objectives.append({
            "description": objective,
            "weight": weight
        })

    def plan_with_multiple_objectives(self, task: str) -> Dict:
        """多目标规划"""
        # 为每个目标生成规划
        objective_plans = {}
        for obj in self.objectives:
            plan = self._plan_for_objective(task, obj)
            objective_plans[obj["description"]] = plan

        # 聚合规划
        aggregated_plan = self._aggregate_plans(objective_plans)

        # 评估多目标平衡
        evaluation = self._evaluate_multi_objective(aggregated_plan)

        return {
            "plan": aggregated_plan,
            "objective_plans": objective_plans,
            "evaluation": evaluation
        }

    def _plan_for_objective(self, task: str, objective: Dict) -> str:
        """为单个目标规划"""
        prompt = f"""
        任务：{task}
        优化目标：{objective['description']}

        请生成针对该目标的规划："""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
```

---

## 12.19 高级规划算法的应用案例

### 12.19.1 案例1：复杂代码生成

```python
def case_complex_code_generation():
    """复杂代码生成案例"""
    task = "实现一个分布式缓存系统，支持一致性哈希、故障转移和负载均衡"

    # 使用ToT探索多种设计方案
    tot = TreeOfThoughts(llm)
   设计方案 = tot.search(task)

    # 使用GoT聚合最佳设计
    got = GraphOfThoughts(llm)
    # 添加多个设计方案作为初始思考
    designs =设计方案.split("\n")
    thought_ids = [got.add_thought(design) for design in designs]

    # 聚合设计方案
    final_design = got.aggregate(thought_ids)

    # 使用Self-Refine改进代码质量
    self_refine = SelfRefine(llm)
    refined_code = self_refine.run(f"根据以下设计实现代码：{final_design}")

    return refined_code
```

### 12.19.2 案例2：科学发现

```python
def case_scientific_discovery():
    """科学发现案例"""
    task = "分析以下实验数据，提出新的科学假设：\n[实验数据...]"

    # 使用Reflexion从错误假设中学习
    reflexion = ReflexionAgent(llm)
    result = reflexion.run(task)

    # 使用LATS优化假设空间
    lats = LATSAgent(llm)
    optimized_hypothesis = lats.search(f"基于以下假设优化：{result['result']}")

    return {
        "initial_hypothesis": result["result"],
        "optimized_hypothesis": optimized_hypothesis,
        "learning_process": result["memory"]
    }
```

### 12.19.3 案例3：战略规划

```python
def case_strategic_planning():
    """战略规划案例"""
    task = "为一家科技公司制定未来5年的发展战略"

    # 使用混合算法
    hybrid = HybridAlgorithm()
    strategy = hybrid.solve(task, strategy="sequential")

    # 使用多目标规划优化
    multi_obj = MultiObjectivePlanning(llm)
    multi_obj.add_objective("利润最大化", weight=0.4)
    multi_obj.add_objective("市场份额", weight=0.3)
    multi_obj.add_objective("技术创新", weight=0.3)

    optimized_strategy = multi_obj.plan_with_multiple_objectives(strategy)

    return optimized_strategy
```

---

## 12.20 高级规划算法的性能优化

### 12.20.1 并行化优化

```python
class ParallelPlanning:
    def __init__(self, planner):
        self.planner = planner

    def parallel_solve(self, tasks: List[str], max_workers: int = 4) -> List[str]:
        """并行解决多个任务"""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.planner.solve, task): task for task in tasks}
            results = []

            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results.append({"task": task, "result": result})
                except Exception as e:
                    results.append({"task": task, "error": str(e)})

        return results

    def parallel_explore(self, task: str, n_variations: int = 4) -> List[str]:
        """并行探索多个变体"""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_variations) as executor:
            futures = []
            for i in range(n_variations):
                # 为每个变体生成不同的提示
                prompt = f"变体{i+1}：{task}"
                futures.append(executor.submit(self.planner.solve, prompt))

            results = [future.result() for future in futures]

        return results
```

### 12.20.2 缓存优化

```python
class CachingPlanner:
    def __init__(self, planner, cache_size: int = 1000):
        self.planner = planner
        self.cache = {}
        self.cache_size = cache_size
        self.access_order = []

    def solve(self, task: str) -> str:
        """带缓存的解决"""
        # 检查缓存
        cache_key = self._get_cache_key(task)
        if cache_key in self.cache:
            # 更新访问顺序
            self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            return self.cache[cache_key]

        # 执行规划
        result = self.planner.solve(task)

        # 存入缓存
        if len(self.cache) >= self.cache_size:
            # 移除最久未访问的项
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

        self.cache[cache_key] = result
        self.access_order.append(cache_key)

        return result

    def _get_cache_key(self, task: str) -> str:
        """生成缓存键"""
        import hashlib
        return hashlib.md5(task.encode()).hexdigest()
```

### 12.20.3 增量规划

```python
class IncrementalPlanner:
    def __init__(self, planner):
        self.planner = planner
        self.current_plan = None
        self.plan_history = []

    def incremental_solve(self, new_info: str) -> str:
        """增量解决"""
        if self.current_plan is None:
            # 首次规划
            self.current_plan = self.planner.solve(new_info)
        else:
            # 增量更新
            self.current_plan = self._update_plan(self.current_plan, new_info)

        self.plan_history.append(self.current_plan)
        return self.current_plan

    def _update_plan(self, current_plan: str, new_info: str) -> str:
        """更新计划"""
        prompt = f"""
        现有计划：{current_plan}
        新信息：{new_info}

        请更新计划以包含新信息："""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
```

---

## 12.21 高级规划算法的评估框架

### 12.21.1 综合评估体系

```python
class ComprehensiveEvaluation:
    def __init__(self):
        self.evaluation_dimensions = {
            "accuracy": self._evaluate_accuracy,
            "efficiency": self._evaluate_efficiency,
            "creativity": self._evaluate_creativity,
            "robustness": self._evaluate_robustness,
            "explainability": self._evaluate_explainability
        }

    def evaluate_algorithm(self, algorithm, test_cases: List[Dict]) -> Dict:
        """评估算法"""
        results = {}

        for dimension, evaluator in self.evaluation_dimensions.items():
            dimension_scores = []
            for case in test_cases:
                score = evaluator(algorithm, case)
                dimension_scores.append(score)

            results[dimension] = {
                "mean": sum(dimension_scores) / len(dimension_scores),
                "min": min(dimension_scores),
                "max": max(dimension_scores),
                "std": self._compute_std(dimension_scores)
            }

        # 计算综合得分
        results["overall"] = self._compute_overall_score(results)

        return results

    def _evaluate_accuracy(self, algorithm, case: Dict) -> float:
        """评估准确性"""
        result = algorithm.solve(case["task"])
        if "expected" in case:
            return self._compare_results(result, case["expected"])
        return self._self_evaluate(result, case["task"])

    def _evaluate_efficiency(self, algorithm, case: Dict) -> float:
        """评估效率"""
        import time
        start_time = time.time()
        algorithm.solve(case["task"])
        end_time = time.time()

        # 归一化效率
        execution_time = end_time - start_time
        efficiency = 1.0 - min(execution_time / 30.0, 1.0)  # 30秒为基准
        return max(efficiency, 0.0)
```

### 12.21.2 可解释性评估

```python
class ExplainabilityEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def evaluate_explainability(self, algorithm, task: str) -> Dict:
        """评估可解释性"""
        # 获取算法输出
        result = algorithm.solve(task)

        # 评估可解释性维度
        dimensions = {
            "clarity": self._evaluate_clarity(result),
            "structure": self._evaluate_structure(result),
            "reasoning": self._evaluate_reasoning(result),
            "transparency": self._evaluate_transparency(result)
        }

        # 计算综合可解释性分数
        overall_score = sum(dimensions.values()) / len(dimensions)

        return {
            "dimensions": dimensions,
            "overall_score": overall_score,
            "suggestions": self._generate_suggestions(dimensions)
        }

    def _evaluate_clarity(self, result: str) -> float:
        """评估清晰度"""
        # 使用LLM评估清晰度
        prompt = f"""
        评估以下输出的清晰度（0-1）：
        {result[:500]}

        清晰度分数："""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            return float(response.content.strip())
        except:
            return 0.5
```

---

## 12.22 高级规划算法的部署实践

### 12.22.1 微服务架构部署

```python
class MicroservicePlanningService:
    def __init__(self):
        self.services = {
            "cot": {"algorithm": CoTAlgorithm(), "port": 8001},
            "tot": {"algorithm": TreeOfThoughtsAlgorithm(), "port": 8002},
            "got": {"algorithm": GraphOfThoughtsAlgorithm(), "port": 8003},
            "reflexion": {"algorithm": ReflexionAlgorithm(), "port": 8004},
            "self_refine": {"algorithm": SelfRefineAlgorithm(), "port": 8005},
            "lats": {"algorithm": LATSAlgorithm(), "port": 8006}
        }

        self.load_balancer = LoadBalancer()

    def start_services(self):
        """启动所有服务"""
        from fastapi import FastAPI
        import uvicorn

        for name, service in self.services.items():
            app = FastAPI()

            @app.post(f"/{name}/solve")
            async def solve_endpoint(request: SolveRequest):
                result = service["algorithm"].solve(request.task)
                return {"result": result, "algorithm": name}

            # 在实际中，这里会启动多个进程
            print(f"启动服务 {name} 在端口 {service['port']}")

    def route_request(self, task: str, algorithm: str = "auto") -> Dict:
        """路由请求"""
        if algorithm == "auto":
            algorithm = self._select_algorithm(task)

        # 选择服务实例
        service = self.services[algorithm]

        # 执行请求
        start_time = time.time()
        result = service["algorithm"].solve(task)
        end_time = time.time()

        return {
            "result": result,
            "algorithm": algorithm,
            "service_port": service["port"],
            "execution_time": end_time - start_time
        }
```

### 12.22.2 容器化部署

```python
class ContainerizedDeployment:
    def __init__(self):
        self.containers = {}

    def create_docker_compose(self):
        """生成docker-compose.yml"""
        compose_content = """
version: '3.8'
services:
"""
        for name, config in self.services.items():
            compose_content += f"""
  {name}:
    build: ./algorithms/{name}
    ports:
      - "{config['port']}:{config['port']}"
    environment:
      - ALGORITHM={name}
    volumes:
      - ./data:/app/data
"""

        return compose_content

    def deploy_to_kubernetes(self):
        """部署到Kubernetes"""
        k8s_config = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "planning-service"},
            "spec": {
                "replicas": 3,
                "selector": {"matchLabels": {"app": "planning"}},
                "template": {
                    "metadata": {"labels": {"app": "planning"}},
                    "spec": {
                        "containers": [{
                            "name": "planning",
                            "image": "planning-service:latest",
                            "ports": [{"containerPort": 8000}]
                        }]
                    }
                }
            }
        }
        return k8s_config
```

---

## 12.23 高级规划算法的监控与运维

### 12.23.1 监控指标体系

```python
class PlanningMonitoring:
    def __init__(self):
        self.metrics = {
            "request_count": 0,
            "success_count": 0,
            "error_count": 0,
            "avg_response_time": 0,
            "algorithm_usage": {},
            "performance_trends": []
        }

    def record_request(self, algorithm: str, success: bool, response_time: float):
        """记录请求"""
        self.metrics["request_count"] += 1
        if success:
            self.metrics["success_count"] += 1
        else:
            self.metrics["error_count"] += 1

        # 更新平均响应时间
        total_time = self.metrics["avg_response_time"] * (self.metrics["request_count"] - 1)
        self.metrics["avg_response_time"] = (total_time + response_time) / self.metrics["request_count"]

        # 更新算法使用统计
        if algorithm not in self.metrics["algorithm_usage"]:
            self.metrics["algorithm_usage"][algorithm] = 0
        self.metrics["algorithm_usage"][algorithm] += 1

    def get_dashboard(self) -> Dict:
        """获取监控仪表板"""
        success_rate = self.metrics["success_count"] / self.metrics["request_count"] if self.metrics["request_count"] > 0 else 0

        return {
            "summary": {
                "total_requests": self.metrics["request_count"],
                "success_rate": success_rate,
                "avg_response_time": self.metrics["avg_response_time"]
            },
            "algorithm_usage": self.metrics["algorithm_usage"],
            "alerts": self._check_alerts()
        }

    def _check_alerts(self) -> List[Dict]:
        """检查告警"""
        alerts = []

        # 检查错误率
        if self.metrics["request_count"] > 10:
            error_rate = self.metrics["error_count"] / self.metrics["request_count"]
            if error_rate > 0.1:
                alerts.append({
                    "type": "high_error_rate",
                    "message": f"错误率过高: {error_rate:.2%}",
                    "severity": "high"
                })

        # 检查响应时间
        if self.metrics["avg_response_time"] > 10:
            alerts.append({
                "type": "slow_response",
                "message": f"平均响应时间过长: {self.metrics['avg_response_time']:.2f}秒",
                "severity": "medium"
            })

        return alerts
```

### 12.23.2 日志分析

```python
class LogAnalyzer:
    def __init__(self):
        self.logs = []

    def add_log(self, log_entry: Dict):
        """添加日志"""
        log_entry["timestamp"] = time.time()
        self.logs.append(log_entry)

    def analyze_patterns(self) -> Dict:
        """分析日志模式"""
        patterns = {
            "error_patterns": self._analyze_errors(),
            "performance_patterns": self._analyze_performance(),
            "usage_patterns": self._analyze_usage()
        }
        return patterns

    def _analyze_errors(self) -> Dict:
        """分析错误模式"""
        errors = [log for log in self.logs if log.get("level") == "error"]
        error_types = {}
        for error in errors:
            error_type = error.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        return error_types

    def generate_report(self) -> str:
        """生成分析报告"""
        patterns = self.analyze_patterns()

        report = f"""
规划算法日志分析报告
====================

总日志数: {len(self.logs)}
错误数: {len([log for log in self.logs if log.get('level') == 'error'])}

错误模式:
{self._format_dict(patterns['error_patterns'])}

性能模式:
{self._format_dict(patterns['performance_patterns'])}
"""
        return report
```

---

## 12.24 高级规划算法的安全考虑

### 12.24.1 安全规划框架

```python
class SecurePlanningFramework:
    def __init__(self, planner):
        self.planner = planner
        self.security_policies = []

    def add_security_policy(self, policy: Dict):
        """添加安全策略"""
        self.security_policies.append(policy)

    def secure_solve(self, task: str) -> Dict:
        """安全规划"""
        # 输入验证
        validated_task = self._validate_input(task)

        # 应用安全策略
        secured_task = self._apply_security_policies(validated_task)

        # 执行规划
        result = self.planner.solve(secured_task)

        # 输出过滤
        filtered_result = self._filter_output(result)

        return {
            "result": filtered_result,
            "security_checks": self._get_security_checks()
        }

    def _validate_input(self, task: str) -> str:
        """验证输入"""
        # 检查恶意内容
        if self._contains_malicious_content(task):
            raise SecurityError("检测到恶意内容")

        # 检查长度限制
        if len(task) > 10000:
            raise SecurityError("输入过长")

        return task

    def _apply_security_policies(self, task: str) -> str:
        """应用安全策略"""
        for policy in self.security_policies:
            if policy["type"] == "content_filter":
                task = self._apply_content_filter(task, policy["rules"])
            elif policy["type"] == "length_limit":
                task = self._apply_length_limit(task, policy["max_length"])
        return task
```

### 12.24.2 对抗性测试

```python
class AdversarialTesting:
    def __init__(self, planner):
        self.planner = planner

    def adversarial_test(self, task: str) -> Dict:
        """对抗性测试"""
        # 生成对抗性输入
        adversarial_inputs = self._generate_adversarial_inputs(task)

        results = []
        for adv_input in adversarial_inputs:
            try:
                result = self.planner.solve(adv_input)
                results.append({
                    "input": adv_input,
                    "result": result,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "input": adv_input,
                    "error": str(e),
                    "success": False
                })

        return {
            "total_tests": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "results": results
        }

    def _generate_adversarial_inputs(self, task: str) -> List[str]:
        """生成对抗性输入"""
        adversarial = []
        # 添加矛盾条件
        adversarial.append(f"{task}（但要求输出为负数）")
        # 添加不可能约束
        adversarial.append(f"{task}（且必须在0秒内完成）")
        # 添加歧义
        adversarial.append(task.replace("计算", "可能计算也可能不计算"))
        return adversarial
```

---

## 12.25 本章总结与学习路径

### 12.25.1 知识体系总结

本章深入介绍了六种高级规划算法：

1. **Tree of Thoughts (ToT)**：树状搜索，支持多路径探索和评估。
2. **Graph of Thoughts (GoT)**：图状推理，支持聚合、精炼和链接。
3. **Reflexion**：通过反思从错误中学习，提升推理质量。
4. **Self-Refine**：迭代改进输出质量，生成-评估-改进循环。
5. **LATS**：结合MCTS的复杂决策算法，支持模拟和反向传播。
6. **混合算法**：结合多种算法优势的复合策略。

### 12.25.2 学习路径建议

1. **基础阶段**：理解每种算法的核心思想和基本实现。
2. **进阶阶段**：学习算法优化、性能调优和错误处理。
3. **实践阶段**：将算法应用到实际问题中，积累经验。
4. **研究阶段**：探索算法改进和新算法设计。

### 12.25.3 下一步学习方向

1. **多Agent规划**：多个Agent协作完成复杂任务。
2. **实时规划**：支持实时动态调整的规划算法。
3. **跨领域迁移**：规划知识在不同领域间的迁移。
 4. **可解释规划**：提高规划过程的可解释性。

 掌握本章内容，你将能够设计和实现各种复杂的规划算法，为构建强大的AIAgent系统奠定坚实基础。

工具选择是 Agent 决策中的关键环节。下面的交互式演示展示了完整的工具选择决策过程：

<div data-component="ToolSelectionDemoV9"></div>

不同的推理策略适用于不同的场景。下面的交互式工具可以帮助你选择最合适的推理策略：

<div data-component="ReasoningStrategySelectorV3"></div>

 > **下一章预告**
 >
 > 在第 13 章中，我们将系统梳理 Agent 架构设计模式，包括单Agent、多Agent、微服务等架构，帮助你设计和实现高效的Agent系统。
