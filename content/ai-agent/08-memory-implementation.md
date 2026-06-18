---
title: 第8章：记忆工程实现 — 从理论到生产
description: 深入实现AI Agent记忆系统的工程细节，包括统一接口设计、混合记忆管理器、LangGraph Checkpointer、跨会话记忆、持久化序列化等核心模块的完整代码实现
tags:
  - AI-Agent
  - Memory
  - LangGraph
  - 持久化
  - 混合检索
---

 # 第8章：记忆工程实现 — 从理论到生产

 记忆工程是构建生产级 Agent 系统的关键。本章将理论转化为可运行的代码。

 下面的交互式演示展示了向量数据库的核心操作：

 <div data-component="VectorDatabaseOps"></div>

 ## 8.1 统一记忆接口设计

### 8.1.1 为什么要设计统一接口

在实际项目中，不同的LLM提供商、不同的存储后端、不同的检索策略会产生大量异构的记忆系统。如果没有统一的接口层，开发者将被迫在每个新项目中重复适配记忆逻辑。统一接口的价值在于：

1. **解耦存储与逻辑**：业务代码不关心记忆是存在Redis、PostgreSQL还是本地文件
2. **策略可插拔**：检索策略、遗忘曲线、冲突解决算法都可以通过接口替换
3. **测试友好**：Mock实现可以轻松替换真实存储后端

### 8.1.2 核心数据结构定义

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
import uuid


class MemoryType(Enum):
    """记忆类型枚举 - 对应认知科学中的记忆分类"""
    SHORT_TERM = "short_term"      # 短期记忆：工作记忆，保持当前对话上下文
    LONG_TERM = "long_term"        # 长期记忆：持久化的重要信息
    EPISODIC = "episodic"          # 情节记忆：具体事件的完整记录
    SEMANTIC = "semantic"          # 语义记忆：抽象知识和概念
    PROCEDURAL = "procedural"      # 程序记忆：操作流程和技能


@dataclass
class Memory:
    """统一的记忆条目数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""                          # 记忆的文本内容
    memory_type: MemoryType = MemoryType.SHORT_TERM
    embedding: Optional[List[float]] = None    # 向量表示，用于语义检索
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: Optional[datetime] = None     # 最后访问时间，用于衰减计算
    access_count: int = 0                      # 访问频率，用于重要性评估
    relevance_score: float = 1.0               # 相关性评分
    decay_factor: float = 1.0                  # 衰减因子，模拟遗忘曲线
    tags: List[str] = field(default_factory=list)  # 标签，用于分类检索
     source: str = ""                           # 记忆来源（用户输入/系统/工具调用）
     session_id: str = ""                       # 所属会话ID
     agent_id: str = ""                         # 所属智能体ID

     def update_access(self) -> None:
         """更新访问记录，触发衰减因子重计算"""
         self.accessed_at = datetime.now()
         self.access_count += 1
         self._recalculate_decay()

     def _recalculate_decay(self) -> None:
         """
         基于Ebbinghaus遗忘曲线重新计算衰减因子
         公式: R = e^(-t/S)

感知-决策-行动循环是 Agent 的核心。下面的交互式可视化展示了完整的 PDA 循环：

<div data-component="PDAFlowChartV6"></div>
        R: 记忆强度（0-1）
        t: 自上次访问后经过的时间（小时）
        S: 稳定性因子，与访问频率正相关
        """
        import math
        if self.accessed_at is None:
            return
        hours_since_access = (
            datetime.now() - self.accessed_at
        ).total_seconds() / 3600
        stability = 1.0 + self.access_count * 0.5
        self.decay_factor = math.exp(-hours_since_access / stability)


@dataclass
class MemoryQuery:
    """记忆查询请求结构"""
    query_text: str                           # 查询文本
    query_embedding: Optional[List[float]] = None  # 查询向量
    memory_types: List[MemoryType] = field(default_factory=list)  # 过滤类型
    top_k: int = 10                           # 返回数量
    min_relevance: float = 0.0                # 最小相关性阈值
    time_range: Optional[tuple] = None        # 时间范围过滤
    tags: List[str] = field(default_factory=list)  # 标签过滤
    session_id: Optional[str] = None          # 会话过滤
    include_decay: bool = True                # 是否考虑衰减因子


@dataclass
class MemorySearchResult:
    """记忆检索结果"""
    memories: List[Memory]                    # 检索到的记忆列表
    scores: List[float]                       # 对应的相似度分数
    query_time_ms: float                      # 查询耗时（毫秒）
    total_candidates: int                     # 候选总数
```

### 8.1.3 抽象基类设计

```python
from abc import ABC, abstractmethod


class MemoryStore(ABC):
    """
    记忆存储抽象基类
    所有存储后端（Redis、PostgreSQL、文件系统等）都必须实现此接口
    """

    @abstractmethod
    async def initialize(self) -> None:
        """初始化存储连接和必要资源"""
        pass

    @abstractmethod
    async def store(self, memory: Memory) -> str:
        """
        存储一条记忆
        Args:
            memory: 要存储的记忆对象
        Returns:
            存储后的记忆ID
        """
        pass

    @abstractmethod
    async def retrieve(self, memory_id: str) -> Optional[Memory]:
        """根据ID检索单条记忆"""
        pass

    @abstractmethod
    async def search(self, query: MemoryQuery) -> MemorySearchResult:
        """
        根据查询条件检索记忆
        这是核心检索接口，支持向量相似度检索和元数据过滤
        """
        pass

    @abstractmethod
    async def update(self, memory: Memory) -> bool:
        """更新已有记忆"""
        pass

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """删除指定记忆"""
        pass

    @abstractmethod
    async def list_by_session(
        self, session_id: str, limit: int = 100
    ) -> List[Memory]:
        """列出某个会话的所有记忆"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        pass

    async def close(self) -> None:
        """关闭连接，清理资源"""
        pass


class MemoryRetriever(ABC):
    """
    记忆检索策略抽象基类
    不同的检索策略可以实现不同的排序和过滤逻辑
    """

    @abstractmethod
    async def retrieve(
        self,
        query: MemoryQuery,
        candidates: List[Memory]
    ) -> List[tuple[Memory, float]]:
        """
        从候选集中检索最相关的记忆
        Returns:
            排序后的 (记忆, 分数) 元组列表
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """返回检索策略名称"""
        pass
```

### 8.1.4 接口设计原则

在设计统一接口时，我们遵循以下原则：

| 原则 | 说明 | 实现方式 |
|------|------|----------|
| 单一职责 | 每个类只做一件事 | 存储、检索、管理三层分离 |
| 依赖倒置 | 依赖抽象而非具体实现 | 抽象基类 + 具体实现类 |
| 开闭原则 | 对扩展开放，对修改关闭 | 策略模式实现可插拔检索 |
| 里氏替换 | 子类可替换父类 | 所有MemoryStore实现可互换 |
| 最小知识 | 模块间低耦合 | 通过Memory数据结构传递信息 |

> 记忆工程的第一步不是选择向量数据库，而是设计好数据在系统中的流转方式。一个良好的接口设计可以让后续的存储后端替换成本降低到个位数的文件修改。

---

## 8.2 混合记忆管理器完整实现

本节实现一个生产级的混合记忆管理器，它整合了短期记忆（滑动窗口）、长期记忆（向量检索）、工作记忆（当前上下文）三种模式，并支持记忆的自动衰减和重要性评估。

### 8.2.1 记忆的重要性评分算法

```python
import math
from typing import List, Tuple
import numpy as np


class ImportanceScorer:
    """
    记忆重要性评分器
    综合考虑多个因素计算记忆的最终重要性分数
    """

    # 各因素的权重配置
    WEIGHTS = {
        'recency': 0.25,       # 时效性权重
        'frequency': 0.20,     # 频率权重
        'relevance': 0.35,     # 相关性权重（最高，因为语义相关最重要）
        'emotional': 0.10,     # 情感强度权重
        'information': 0.10,   # 信息密度权重
    }

    # 情感关键词列表（实际项目中可使用情感分析模型）
    EMOTIONAL_KEYWORDS = [
        '紧急', '重要', '必须', '关键', '核心',
        '危险', '警告', '错误', '失败', '成功',
        'urgent', 'important', 'critical', 'key'
    ]

    def calculate(self, memory: Memory) -> float:
        """
        计算记忆的综合重要性分数

        公式: S = w_r * R(t) + w_f * F(f) + w_rel * Rel(s) + w_e * E(k) + w_i * I(d)

        其中:
        - R(t) = e^(-λt) 时效性衰减函数
        - F(f) = 1 - e^(-αf) 频率饱和函数
        - Rel(s) = s / max(s) 归一化相关性
        - E(k) = Σ(命中关键词数) / 总关键词数 情感强度
        - I(d) = d / max_length 信息密度
        """
        scores = {
            'recency': self._recency_score(memory),
            'frequency': self._frequency_score(memory),
            'relevance': memory.relevance_score,
            'emotional': self._emotional_score(memory.content),
            'information': self._information_density(memory.content),
        }

        final_score = sum(
            self.WEIGHTS[k] * v for k, v in scores.items()
        )
        return min(max(final_score, 0.0), 1.0)  # 钳位到[0, 1]

    def _recency_score(self, memory: Memory) -> float:
        """时效性评分：越近越重要"""
        if memory.accessed_at is None:
            return 1.0
        hours = (datetime.now() - memory.accessed_at).total_seconds() / 3600
        return math.exp(-0.1 * hours)  # λ=0.1，约7小时衰减到50%

    def _frequency_score(self, memory: Memory) -> float:
        """频率评分：访问越多越重要，但有饱和上限"""
        return 1 - math.exp(-0.5 * memory.access_count)

    def _emotional_score(self, text: str) -> float:
        """情感强度评分：包含情感关键词越多越重要"""
        hits = sum(1 for kw in self.EMOTIONAL_KEYWORDS if kw in text.lower())
        return min(hits / 3, 1.0)  # 最多3个关键词即满分

    def _information_density(self, text: str) -> float:
        """信息密度评分：文本中独特词汇占比"""
        if not text:
            return 0.0
        words = text.split()
        if len(words) == 0:
            return 0.0
        unique_ratio = len(set(words)) / len(words)
        length_factor = min(len(text) / 500, 1.0)  # 500字符满分
        return unique_ratio * length_factor
```

### 8.2.2 混合记忆管理器核心实现

```python
import asyncio
import json
import hashlib
from collections import deque
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class HybridMemoryManager:
    """
    混合记忆管理器 - 完整的生产级实现

    整合三种记忆类型:
    1. 工作记忆 (Working Memory): 当前对话的滑动窗口
    2. 短期记忆 (Short-term): 最近几次会话的摘要
    3. 长期记忆 (Long-term): 通过向量检索的持久化重要信息

    支持功能:
    - 自动记忆衰减（基于遗忘曲线）
    - 重要性评估与筛选
    - 智能记忆压缩（避免上下文窗口溢出）
    - 混合检索（语义 + 关键词 + 元数据过滤）
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        embedding_fn: Any,
        working_memory_size: int = 20,
        long_term_memory_limit: int = 50,
        importance_threshold: float = 0.3,
        enable_auto_consolidation: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化混合记忆管理器

        Args:
            memory_store: 长期记忆存储后端
            embedding_fn: 向量嵌入函数，接受文本列表返回向量列表
            working_memory_size: 工作记忆窗口大小（消息数）
            long_term_memory_limit: 每次检索返回的长期记忆上限
            importance_threshold: 重要性低于此阈值的记忆将被淘汰
            enable_auto_consolidation: 是否启用自动记忆整合
            config: 额外配置参数
        """
        self.memory_store = memory_store
        self.embedding_fn = embedding_fn
        self.working_memory_size = working_memory_size
        self.long_term_memory_limit = long_term_memory_limit
        self.importance_threshold = importance_threshold
        self.enable_auto_consolidation = enable_auto_consolidation
        self.config = config or {}

        # 工作记忆：使用双端队列实现滑动窗口
        self.working_memory: deque[Memory] = deque(
            maxlen=working_memory_size
        )

        # 短期记忆缓冲区：用于记忆整合前的临时存储
        self.short_term_buffer: List[Memory] = []

        # 记忆重要性评分器
        self.scorer = ImportanceScorer()

        # 去重哈希集合，避免重复记忆
        self._seen_hashes: set = set()

        # 统计计数器
        self.stats = {
            'total_stored': 0,
            'total_retrieved': 0,
            'consolidations': 0,
            'deletions': 0,
        }

    async def initialize(self) -> None:
        """初始化存储后端"""
        await self.memory_store.initialize()
        logger.info("混合记忆管理器初始化完成")

    def _compute_content_hash(self, content: str) -> str:
        """计算内容哈希用于去重"""
        normalized = content.strip().lower()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        source: str = "user",
        session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Memory]:
        """
        添加新记忆到系统

        流程:
        1. 计算内容哈希去重
        2. 创建Memory对象
        3. 生成向量嵌入
        4. 添加到工作记忆
        5. 重要记忆同步到长期记忆
        """
        # 去重检查
        content_hash = self._compute_content_hash(content)
        if content_hash in self._seen_hashes:
            logger.debug(f"记忆内容重复，跳过: {content[:50]}...")
            return None

        self._seen_hashes.add(content_hash)

        # 创建记忆对象
        memory = Memory(
            content=content,
            memory_type=memory_type,
            metadata=metadata or {},
            source=source,
            session_id=session_id,
            tags=tags or [],
            accessed_at=datetime.now(),
        )

        # 生成向量嵌入
        try:
            embeddings = await self.embedding_fn([content])
            memory.embedding = embeddings[0] if embeddings else None
        except Exception as e:
            logger.warning(f"向量嵌入生成失败: {e}")

        # 计算初始重要性
        importance = self.scorer.calculate(memory)
        memory.relevance_score = importance

        # 添加到工作记忆（滑动窗口自动淘汰最旧的）
        self.working_memory.append(memory)

        # 添加到短期缓冲区
        self.short_term_buffer.append(memory)

        # 如果启用自动整合，检查是否需要整合到长期记忆
        if self.enable_auto_consolidation:
            if importance >= self.importance_threshold:
                await self._consolidate_to_long_term(memory)

        self.stats['total_stored'] += 1
        logger.info(
            f"新记忆已存储 [类型={memory_type.value}, "
            f"重要性={importance:.3f}]: {content[:80]}..."
        )
        return memory

    async def _consolidate_to_long_term(self, memory: Memory) -> None:
        """将重要记忆整合到长期存储"""
        memory.memory_type = MemoryType.LONG_TERM
        try:
            stored_id = await self.memory_store.store(memory)
            memory.id = stored_id
            self.stats['consolidations'] += 1
            logger.debug(f"记忆已整合到长期存储: {stored_id}")
        except Exception as e:
            logger.error(f"长期存储写入失败: {e}")

    async def retrieve_relevant_memories(
        self,
        query: str,
        context: Optional[str] = None,
        include_working: bool = True,
        top_k: Optional[int] = None,
    ) -> List[Memory]:
        """
        综合检索最相关的记忆

        混合检索策略:
        1. 工作记忆：直接从滑动窗口获取最近的上下文
        2. 长期记忆：通过向量相似度 + 元数据过滤检索
        3. 去重合并：移除重复记忆
        4. 重新排序：按综合评分排序

        Args:
            query: 查询文本
            context: 额外上下文信息（用于更精确的检索）
            include_working: 是否包含工作记忆
            top_k: 返回的记忆数量上限
        """
        if top_k is None:
            top_k = self.long_term_memory_limit

        candidate_memories: List[Memory] = []

        # 1. 获取工作记忆
        if include_working:
            working_mems = list(self.working_memory)
            candidate_memories.extend(working_mems)

        # 2. 检索长期记忆
        try:
            query_embedding = None
            if self.embedding_fn:
                embeddings = await self.embedding_fn([query])
                query_embedding = embeddings[0] if embeddings else None

            memory_query = MemoryQuery(
                query_text=query,
                query_embedding=query_embedding,
                top_k=top_k * 2,  # 多检索一些用于后续筛选
            )
            long_term_result = await self.memory_store.search(memory_query)
            candidate_memories.extend(long_term_result.memories)
        except Exception as e:
            logger.error(f"长期记忆检索失败: {e}")

        # 3. 去重
        unique_memories = self._deduplicate_memories(candidate_memories)

        # 4. 重新评分和排序
        scored_memories = []
        for mem in unique_memories:
            # 更新访问记录
            mem.update_access()
            # 计算综合分数
            score = self.scorer.calculate(mem)
            scored_memories.append((mem, score))

        # 按分数降序排序
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # 截取top_k
        result = [mem for mem, _ in scored_memories[:top_k]]

        self.stats['total_retrieved'] += len(result)
        logger.info(f"检索到 {len(result)} 条相关记忆")
        return result

    def _deduplicate_memories(
        self, memories: List[Memory]
    ) -> List[Memory]:
        """基于内容哈希去重"""
        seen = set()
        unique = []
        for mem in memories:
            h = self._compute_content_hash(mem.content)
            if h not in seen:
                seen.add(h)
                unique.append(mem)
        return unique

    async def run_consolidation(self) -> Dict[str, Any]:
        """
        执行记忆整合周期

        整合逻辑:
        1. 遍历短期记忆缓冲区
        2. 根据重要性评分筛选
        3. 高重要性记忆转为长期记忆
        4. 低重要性记忆衰减或删除
        5. 压缩相似记忆
        """
        if not self.short_term_buffer:
            return {'consolidated': 0, 'deleted': 0, 'compressed': 0}

        consolidated = 0
        deleted = 0
        compressed = 0

        for memory in self.short_term_buffer:
            importance = self.scorer.calculate(memory)

            if importance >= self.importance_threshold:
                # 高重要性 → 整合到长期记忆
                if memory.memory_type != MemoryType.LONG_TERM:
                    await self._consolidate_to_long_term(memory)
                    consolidated += 1
            elif importance < 0.1:
                # 极低重要性 → 标记删除
                self._seen_hashes.discard(
                    self._compute_content_hash(memory.content)
                )
                deleted += 1
            else:
                # 中等重要性 → 保留但不整合
                pass

        # 清空缓冲区（已被处理的记忆）
        self.short_term_buffer.clear()

        result = {
            'consolidated': consolidated,
            'deleted': deleted,
            'compressed': compressed,
            'buffer_size_before': len(self.short_term_buffer),
        }

        logger.info(f"记忆整合完成: {result}")
        return result

    async def get_context_for_llm(
        self,
        current_query: str,
        max_tokens: int = 4000,
        token_estimate_fn: Any = None,
    ) -> str:
        """
        为LLM生成包含记忆上下文的输入

        格式:
        [工作记忆: 最近的对话]
        [长期记忆: 检索到的相关信息]
        [当前问题]

        Args:
            current_query: 当前用户查询
            max_tokens: token数上限
            token_estimate_fn: token估算函数
        """
        if token_estimate_fn is None:
            token_estimate_fn = lambda text: len(text) // 2  # 粗略估算

        # 获取相关记忆
        relevant = await self.retrieve_relevant_memories(current_query)

        # 构建上下文
        context_parts = []

        # 工作记忆部分
        if self.working_memory:
            working_text = "\n".join(
                f"- {m.content}" for m in list(self.working_memory)[-5:]
            )
            context_parts.append(f"## 工作记忆\n{working_text}")

        # 长期记忆部分
        if relevant:
            long_term_text = "\n".join(
                f"- [{m.memory_type.value}] {m.content}"
                for m in relevant[:10]
            )
            context_parts.append(f"## 相关记忆\n{long_term_text}")

        # 组合并截断到token限制
        full_context = "\n\n".join(context_parts)

        # 简单截断（生产环境应使用更精确的token计数）
        while token_estimate_fn(full_context) > max_tokens and len(full_context) > 100:
            full_context = full_context[:int(len(full_context) * 0.9)]

        return full_context

    async def get_stats(self) -> Dict[str, Any]:
        """获取记忆管理器统计信息"""
        store_stats = await self.memory_store.get_stats()
        return {
            **self.stats,
            'working_memory_size': len(self.working_memory),
            'short_term_buffer_size': len(self.short_term_buffer),
            'seen_hashes_count': len(self._seen_hashes),
            'store_stats': store_stats,
        }

    async def cleanup(self) -> None:
        """清理过期记忆"""
        await self.memory_store.close()
        self.working_memory.clear()
        self.short_term_buffer.clear()
        self._seen_hashes.clear()
        logger.info("记忆管理器已清理")
```

### 8.2.3 记忆衰减与遗忘机制

```python
class MemoryDecayManager:
    """
    记忆衰减管理器

    实现基于Ebbinghaus遗忘曲线的自动记忆衰减
    定期运行以维护记忆系统的健康度
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        decay_threshold: float = 0.1,    # 衰减到此值以下则删除
        check_interval_hours: int = 24,   # 检查间隔（小时）
    ):
        self.memory_store = memory_store
        self.decay_threshold = decay_threshold
        self.check_interval_hours = check_interval_hours

    async def apply_decay(self) -> Dict[str, int]:
        """
        对所有长期记忆应用衰减

        衰减公式: R(t) = e^(-t/S)
        其中 S = base_stability + access_count * stability_growth
        """
        stats = {'decayed': 0, 'deleted': 0, 'preserved': 0}

        # 获取所有需要检查的记忆
        query = MemoryQuery(
            query_text="",
            memory_types=[MemoryType.LONG_TERM],
            top_k=1000,  # 大量检索
        )
        result = await self.memory_store.search(query)

        for memory in result.memories:
            memory.update_access()
            decay = memory.decay_factor

            if decay < self.decay_threshold:
                # 衰减过严重，删除
                await self.memory_store.delete(memory.id)
                stats['deleted'] += 1
            elif decay < 0.5:
                # 中度衰减，降低重要性
                memory.relevance_score *= decay
                await self.memory_store.update(memory)
                stats['decayed'] += 1
            else:
                stats['preserved'] += 1

  logger.info(f"记忆衰减处理完成: {stats}")
         return stats
 ```

 > 混合记忆管理器的核心思想是：**不是所有记忆都值得长期保存**。通过自动衰减和重要性筛选，系统能够自适应地管理有限的存储资源，只保留真正有价值的信息。

Agent 环境循环是 Agent 与环境交互的核心机制。下面的交互式演示展示了 Agent 的完整交互过程：

<div data-component="AgentEnvironmentLoop"></div>

 ---

 ## 8.3 LangGraph Checkpointer详解

LangGraph的Checkpointer是Agent状态管理的基础设施。本节深入分析其设计原理和使用方式。

### 8.3.1 Checkpointer架构原理

LangGraph Checkpointer采用**快照（Snapshot）**机制来保存Agent的执行状态。每次图执行完毕后，Checkpointer会将整个状态序列化并存储，支持在任意时间点恢复状态。

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator


class AgentState(TypedDict):
    """Agent状态定义"""
    messages: Annotated[List[dict], operator.add]  # 消息列表（追加语义）
    current_step: str
    memory_context: str
    iteration_count: int


# 方式1：内存存储（适合开发和测试）
memory_checkpointer = MemorySaver()

# 方式2：SQLite存储（适合单机生产环境）
sqlite_checkpointer = SqliteSaver.from_conn_string(
    "./checkpoints.db"
)

# 方式3：PostgreSQL存储（适合分布式生产环境）
postgres_checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/agent_db"
)


def create_agent_graph(checkpointer=None):
    """创建带Checkpointer的Agent图"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("process_query", process_query_node)
    workflow.add_node("generate_response", generate_response_node)

    # 添加边
    workflow.set_entry_point("retrieve_memory")
    workflow.add_edge("retrieve_memory", "process_query")
    workflow.add_conditional_edges(
        "process_query",
        should_continue,
        {
            "continue": "generate_response",
            "end": END
        }
    )
    workflow.add_edge("generate_response", "retrieve_memory")

    # 编译时传入Checkpointer
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["process_query"]  # 在处理前支持人工中断
    )
```

### 8.3.2 自定义Postgres Checkpointer

```python
import json
import pickle
import zlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import asyncpg
import logging

logger = logging.getLogger(__name__)


class ProductionPostgresCheckpointer:
    """
    生产级PostgreSQL Checkpointer

    特性:
    - 支持状态压缩（zlib）
    - 自动清理旧检查点
    - 支持分支（fork）操作
    - 并发安全
    """

    def __init__(
        self,
        dsn: str,
        max_checkpoints_per_thread: int = 100,
        compression_enabled: bool = True,
    ):
        self.dsn = dsn
        self.max_checkpoints = max_checkpoints_per_thread
        self.compression_enabled = compression_enabled
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self) -> None:
        """初始化连接池和数据库表"""
        self.pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=10)

        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    checkpoint_data BYTEA NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (thread_id, checkpoint_id)
                );
                CREATE INDEX IF NOT EXISTS idx_checkpoints_thread
                    ON langgraph_checkpoints(thread_id, created_at);
            """)
        logger.info("PostgreSQL Checkpointer初始化完成")

    async def put(
        self,
        thread_id: str,
        checkpoint_id: str,
        checkpoint: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        parent_checkpoint_id: Optional[str] = None,
    ) -> None:
        """保存检查点"""
        # 序列化
        data = pickle.dumps(checkpoint)
        if self.compression_enabled:
            data = zlib.compress(data, level=6)

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO langgraph_checkpoints
                    (thread_id, checkpoint_id, parent_checkpoint_id,
                     checkpoint_data, metadata)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (thread_id, checkpoint_id)
                DO UPDATE SET
                    checkpoint_data = EXCLUDED.checkpoint_data,
                    metadata = EXCLUDED.metadata
            """, thread_id, checkpoint_id, parent_checkpoint_id,
                 data, json.dumps(metadata or {}))

            # 清理旧检查点
            await self._cleanup_old_checkpoints(conn, thread_id)

    async def get(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """获取检查点"""
        async with self.pool.acquire() as conn:
            if checkpoint_id:
                row = await conn.fetchrow("""
                    SELECT checkpoint_data, metadata
                    FROM langgraph_checkpoints
                    WHERE thread_id = $1 AND checkpoint_id = $2
                """, thread_id, checkpoint_id)
            else:
                row = await conn.fetchrow("""
                    SELECT checkpoint_data, metadata
                    FROM langgraph_checkpoints
                    WHERE thread_id = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                """, thread_id)

            if row is None:
                return None

            data = row['checkpoint_data']
            if self.compression_enabled:
                data = zlib.decompress(data)
            checkpoint = pickle.loads(data)
            metadata = json.loads(row['metadata'])

            return (checkpoint, metadata)

    async def list_checkpoints(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """列出检查点历史"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT checkpoint_id, parent_checkpoint_id, metadata, created_at
                FROM langgraph_checkpoints
                WHERE thread_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, thread_id, limit)

            return [
                {
                    'checkpoint_id': r['checkpoint_id'],
                    'parent_id': r['parent_checkpoint_id'],
                    'metadata': json.loads(r['metadata']),
                    'created_at': r['created_at'].isoformat(),
                }
                for r in rows
            ]

    async def _cleanup_old_checkpoints(
        self, conn, thread_id: str
    ) -> None:
        """清理超出限制的旧检查点"""
        await conn.execute("""
            DELETE FROM langgraph_checkpoints
            WHERE thread_id = $1
            AND checkpoint_id NOT IN (
                SELECT checkpoint_id
                FROM langgraph_checkpoints
                WHERE thread_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            )
        """, thread_id, self.max_checkpoints)

    async def close(self) -> None:
        """关闭连接池"""
        if self.pool:
            await self.pool.close()
```

> LangGraph Checkpointer的本质是一个**状态快照系统**。它不只保存数据，更保存了Agent的"思维轨迹"——每一步决策、每一次状态变迁都被忠实记录，使得Agent可以在任何时间点"醒来"并继续之前的工作。

---

## 8.4 混合检索策略

### 8.4.1 三种检索策略的实现

```python
from typing import List, Tuple
import numpy as np
import math


class SemanticRetriever(MemoryRetriever):
    """
    语义检索策略

    基于向量相似度的检索，适合理解查询的语义意图
    """

    def __init__(self, similarity_threshold: float = 0.3):
        self.similarity_threshold = similarity_threshold

    async def retrieve(
        self,
        query: MemoryQuery,
        candidates: List[Memory],
    ) -> List[Tuple[Memory, float]]:
        """语义检索：基于余弦相似度"""
        if not query.query_embedding or not candidates:
            return [(c, 0.0) for c in candidates]

        results = []
        for memory in candidates:
            if memory.embedding is None:
                continue
            similarity = self._cosine_similarity(
                query.query_embedding, memory.embedding
            )
            if similarity >= self.similarity_threshold:
                results.append((memory, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _cosine_similarity(
        self, vec_a: List[float], vec_b: List[float]
    ) -> float:
        """余弦相似度计算"""
        a = np.array(vec_a)
        b = np.array(vec_b)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def get_strategy_name(self) -> str:
        return "semantic"


class KeywordRetriever(MemoryRetriever):
    """
    关键词检索策略

    基于TF-IDF或BM25的检索，适合精确匹配
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        # BM25参数
        self.k1 = k1
        self.b = b

    async def retrieve(
        self,
        query: MemoryQuery,
        candidates: List[Memory],
    ) -> List[Tuple[Memory, float]]:
        """BM25关键词检索"""
        if not query.query_text or not candidates:
            return [(c, 0.0) for c in candidates]

        query_terms = query.query_text.lower().split()
        doc_count = len(candidates)

        # 计算平均文档长度
        doc_lengths = [len(m.content.split()) for m in candidates]
        avg_dl = sum(doc_lengths) / doc_count if doc_count > 0 else 1

        results = []
        for i, memory in enumerate(candidates):
            doc_terms = memory.content.lower().split()
            dl = doc_lengths[i]
            score = 0.0

            for term in query_terms:
                tf = doc_terms.count(term)
                if tf == 0:
                    continue

                # IDF部分：简化计算
                df = sum(
                    1 for m in candidates
                    if term in m.content.lower().split()
                )
                idf = math.log(
                    (doc_count - df + 0.5) / (df + 0.5) + 1
                )

                # TF部分（带长度归一化）
                tf_normalized = (
                    tf * (self.k1 + 1)
                ) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / avg_dl)
                )

                score += idf * tf_normalized

            if score > 0:
                results.append((memory, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_strategy_name(self) -> str:
        return "keyword"


class MetadataRetriever(MemoryRetriever):
    """
    元数据过滤检索策略

    基于标签、时间范围、类型等元数据的精确过滤
    """

    async def retrieve(
        self,
        query: MemoryQuery,
        candidates: List[Memory],
    ) -> List[Tuple[Memory, float]]:
        """元数据过滤检索"""
        results = []
        for memory in candidates:
            score = 1.0  # 基础分数

            # 类型过滤
            if (query.memory_types and
                    memory.memory_type not in query.memory_types):
                continue

            # 标签过滤
            if query.tags:
                tag_matches = len(set(query.tags) & set(memory.tags))
                if tag_matches == 0:
                    continue
                score *= (tag_matches / len(query.tags))

            # 时间范围过滤
            if query.time_range:
                start, end = query.time_range
                if not (start <= memory.created_at <= end):
                    continue

            # 会话过滤
            if (query.session_id and
                    memory.session_id != query.session_id):
                continue

            # 最小相关性过滤
            if memory.relevance_score < query.min_relevance:
                continue

            results.append((memory, score * memory.relevance_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_strategy_name(self) -> str:
        return "metadata"
```

### 8.4.2 混合检索融合器

```python
class HybridRetriever:
    """
    混合检索融合器

    将多种检索策略的结果通过RRF（Reciprocal Rank Fusion）算法融合

    RRF公式: RRF_score = Σ 1/(k + rank_i)
    其中 k 通常取60，rank_i 是第i个排序中的位置
    """

    def __init__(
        self,
        retrievers: List[Tuple[MemoryRetriever, float]],
        rrf_k: int = 60,
    ):
        """
        Args:
            retrievers: (检索器, 权重) 元组列表
            rrf_k: RRF公式中的常数k
        """
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    async def retrieve(
        self,
        query: MemoryQuery,
        candidates: List[Memory],
        top_k: int = 10,
    ) -> List[Tuple[Memory, float]]:
        """
        执行混合检索并融合结果
        """
        # 收集各检索器的排名
        all_ranks: Dict[str, Dict[int, int]] = {}

        for retriever, weight in self.retrievers:
            ranked_results = await retriever.retrieve(query, candidates)
            for rank, (memory, score) in enumerate(ranked_results):
                mem_id = memory.id
                if mem_id not in all_ranks:
                    all_ranks[mem_id] = {}
                all_ranks[mem_id][retriever.get_strategy_name()] = rank + 1

        # 计算RRF分数
        fused_scores: Dict[str, float] = {}
        for mem_id, ranks in all_ranks.items():
            score = 0.0
            for strategy_name, rank in ranks.items():
                # 找到对应的权重
                weight = 1.0
                for ret, w in self.retrievers:
                    if ret.get_strategy_name() == strategy_name:
                        weight = w
                        break
                score += weight / (self.rrf_k + rank)
            fused_scores[mem_id] = score

        # 按融合分数排序
        sorted_mem_ids = sorted(
            fused_scores.keys(),
            key=lambda x: fused_scores[x],
            reverse=True
        )

        # 构建结果
        memory_map = {m.id: m for m in candidates}
        results = []
        for mem_id in sorted_mem_ids[:top_k]:
            if mem_id in memory_map:
                results.append((memory_map[mem_id], fused_scores[mem_id]))

        return results
```

### 8.4.3 检索策略对比表

| 策略 | 适用场景 | 优势 | 劣势 | 典型召回率 |
|------|----------|------|------|-----------|
| 语义检索 | 理解意图、概念匹配 | 语义理解能力强 | 对专业术语不够敏感 | 70-85% |
| 关键词检索 | 精确匹配、术语查询 | 精确度高 | 无法理解同义词 | 60-75% |
| 元数据过滤 | 精确分类、时间过滤 | 召回精准 | 依赖元数据质量 | 50-70% |
| 混合检索 | 综合场景 | 平衡全面 | 计算成本较高 | 80-90% |

> 混合检索的精髓在于：**没有单一的检索策略是完美的，但多个策略的融合可以接近完美**。RRF算法的美妙之处在于它不关心各策略的分数尺度是否一致，只关心排名，因此可以直接融合不同维度的检索结果。

---

## 8.5 跨会话记忆实现

### 8.5.1 跨会话记忆的数据模型

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import asyncio
import json
import uuid


@dataclass
class UserProfile:
    """用户画像：跨会话的用户偏好和特征"""
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    communication_style: str = "neutral"  # formal/casual/neutral
    topics_of_interest: List[str] = field(default_factory=list)
    expertise_level: str = "intermediate"  # beginner/intermediate/expert
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class CrossSessionMemory:
    """跨会话记忆条目"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    session_ids: Set[str] = field(default_factory=set)  # 关联的会话列表
    memory_type: str = "fact"  # fact/preference/goal/context
    content: str = ""
    embedding: Optional[List[float]] = None
    confidence: float = 1.0  # 置信度：多次确认后提高
    source_sessions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_confirmed: Optional[datetime] = None
    confirmation_count: int = 0
    is_active: bool = True


@dataclass
class SessionSummary:
    """会话摘要：用于跨会话记忆整合"""
    session_id: str
    user_id: str
    summary_text: str
    key_facts: List[str]
    user_preferences_discovered: List[str]
    goals_mentioned: List[str]
    important_entities: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None
```

### 8.5.2 跨会话记忆管理器

```python
class CrossSessionMemoryManager:
    """
    跨会话记忆管理器

    核心功能:
    1. 会话摘要生成与存储
    2. 跨会话事实提取与去重
    3. 用户画像更新
    4. 历史上下文检索
    5. 记忆置信度管理
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        embedding_fn: Any,
        llm_fn: Any,
        user_profile_store: Any,
        confidence_threshold: float = 0.7,
        max_history_sessions: int = 20,
    ):
        self.memory_store = memory_store
        self.embedding_fn = embedding_fn
        self.llm_fn = llm_fn
        self.user_profile_store = user_profile_store
        self.confidence_threshold = confidence_threshold
        self.max_history_sessions = max_history_sessions

        # 跨会话记忆缓存
        self._memory_cache: Dict[str, List[CrossSessionMemory]] = {}
        self._profile_cache: Dict[str, UserProfile] = {}

    async def create_session_summary(
        self,
        session_id: str,
        user_id: str,
        messages: List[Dict[str, str]],
    ) -> SessionSummary:
        """
        为一个会话生成摘要

        使用LLM从对话历史中提取关键信息
        """
        # 构建摘要提示
        messages_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
            for m in messages[-30:]  # 最近30条消息
        )

        prompt = f"""请分析以下对话，提取关键信息并生成结构化摘要。

对话内容:
{messages_text}

请返回JSON格式:
{{
    "summary": "对话摘要（2-3句话）",
    "key_facts": ["提取的事实1", "事实2", ...],
    "preferences": ["发现的用户偏好1", "偏好2", ...],
    "goals": ["用户提到的目标1", "目标2", ...],
    "entities": ["重要实体（人名/项目名/技术名）", ...]
}}"""

        try:
            response = await self.llm_fn(prompt)
            # 解析JSON响应
            data = json.loads(response) if isinstance(response, str) else response
        except Exception as e:
            logger.warning(f"LLM摘要生成失败: {e}")
            data = {
                "summary": "会话摘要生成失败",
                "key_facts": [],
                "preferences": [],
                "goals": [],
                "entities": [],
            }

        # 生成摘要向量
        embedding = None
        try:
            embeddings = await self.embedding_fn([data["summary"]])
            embedding = embeddings[0] if embeddings else None
        except Exception:
            pass

        summary = SessionSummary(
            session_id=session_id,
            user_id=user_id,
            summary_text=data["summary"],
            key_facts=data.get("key_facts", []),
            user_preferences_discovered=data.get("preferences", []),
            goals_mentioned=data.get("goals", []),
            important_entities=data.get("entities", []),
            embedding=embedding,
        )

        # 存储摘要
        await self._store_session_summary(summary)

        return summary

    async def _store_session_summary(
        self, summary: SessionSummary
    ) -> None:
        """存储会话摘要到长期记忆"""
        memory = Memory(
            content=f"[会话摘要] {summary.summary_text}",
            memory_type=MemoryType.EPISODIC,
            embedding=summary.embedding,
            metadata={
                'type': 'session_summary',
                'key_facts': summary.key_facts,
                'preferences': summary.user_preferences_discovered,
                'goals': summary.goals_mentioned,
                'entities': summary.important_entities,
            },
            session_id=summary.session_id,
            source="system",
        )
        await self.memory_store.store(memory)

    async def extract_cross_session_facts(
        self,
        user_id: str,
        session_summary: SessionSummary,
    ) -> List[CrossSessionMemory]:
        """
        从会话摘要中提取跨会话记忆

        与已有记忆对比，更新或创建新记忆
        """
        new_facts = []

        for fact in session_summary.key_facts:
            # 检查是否已有类似记忆
            existing = await self._find_similar_memory(user_id, fact)

            if existing:
                # 已有类似记忆，增强置信度
                existing.confirmation_count += 1
                existing.last_confirmed = datetime.now()
                existing.confidence = min(
                    1.0,
                    existing.confidence + 0.1
                )
                existing.session_ids.add(session_summary.session_id)
                await self.memory_store.update(existing)
            else:
                # 新记忆
                new_memory = CrossSessionMemory(
                    user_id=user_id,
                    session_ids={session_summary.session_id},
                    memory_type="fact",
                    content=fact,
                    source_sessions=[session_summary.session_id],
                    last_confirmed=datetime.now(),
                    confirmation_count=1,
                )
                # 生成向量
                try:
                    embeddings = await self.embedding_fn([fact])
                    new_memory.embedding = embeddings[0] if embeddings else None
                except Exception:
                    pass

                # 存储
                memory = Memory(
                    content=fact,
                    memory_type=MemoryType.SEMANTIC,
                    embedding=new_memory.embedding,
                    metadata={
                        'type': 'cross_session_fact',
                        'user_id': user_id,
                        'confidence': new_memory.confidence,
                    },
                    source="cross_session_extraction",
                )
                await self.memory_store.store(memory)
                new_facts.append(new_memory)

        return new_facts

    async def _find_similar_memory(
        self, user_id: str, content: str
    ) -> Optional[CrossSessionMemory]:
        """查找相似的已有跨会话记忆"""
        try:
            embeddings = await self.embedding_fn([content])
            query = MemoryQuery(
                query_text=content,
                query_embedding=embeddings[0] if embeddings else None,
                top_k=1,
                min_relevance=0.85,  # 高阈值确保真正相似
            )
            result = await self.memory_store.search(query)

            if result.memories:
                mem = result.memories[0]
                return CrossSessionMemory(
                    id=mem.id,
                    user_id=user_id,
                    content=mem.content,
                    embedding=mem.embedding,
                    confidence=mem.metadata.get('confidence', 1.0),
                )
        except Exception:
            pass
        return None

    async def update_user_profile(
        self,
        user_id: str,
        session_summary: SessionSummary,
    ) -> UserProfile:
        """
        根据会话摘要更新用户画像

        增量学习用户偏好，避免重复覆盖
        """
        profile = await self._get_or_create_profile(user_id)

        # 更新兴趣话题
        for entity in session_summary.important_entities:
            if entity not in profile.topics_of_interest:
                profile.topics_of_interest.append(entity)

        # 更新偏好
        for pref in session_summary.user_preferences_discovered:
            pref_key = pref.split(":")[0].strip() if ":" in pref else pref
            pref_value = pref.split(":")[1].strip() if ":" in pref else "true"
            profile.preferences[pref_key] = pref_value

        # 更新目标（保留最近5个）
        if session_summary.goals_mentioned:
            profile.preferences['recent_goals'] = (
                session_summary.goals_mentioned[-5:]
            )

        profile.updated_at = datetime.now()
        await self.user_profile_store.save(profile)
        self._profile_cache[user_id] = profile

        return profile

    async def _get_or_create_profile(
        self, user_id: str
    ) -> UserProfile:
        """获取或创建用户画像"""
        if user_id in self._profile_cache:
            return self._profile_cache[user_id]

        profile = await self.user_profile_store.get(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id)
        self._profile_cache[user_id] = profile
        return profile

    async def get_cross_session_context(
        self,
        user_id: str,
        current_query: str,
        max_memories: int = 10,
    ) -> str:
        """
        获取跨会话上下文

        返回格式化的跨会话记忆，供LLM使用
        """
        try:
            embeddings = await self.embedding_fn([current_query])
            query = MemoryQuery(
                query_text=current_query,
                query_embedding=embeddings[0] if embeddings else None,
                top_k=max_memories,
                min_relevance=0.5,
            )
            result = await self.memory_store.search(query)
        except Exception:
            result = MemorySearchResult(
                memories=[], scores=[], query_time_ms=0,
                total_candidates=0
            )

        # 格式化上下文
        context_parts = []

        # 用户画像
        profile = await self._get_or_create_profile(user_id)
        if profile.topics_of_interest:
            context_parts.append(
                f"用户兴趣领域: {', '.join(profile.topics_of_interest[:5])}"
            )
        if profile.preferences:
            prefs = "; ".join(
                f"{k}: {v}" for k, v in list(profile.preferences.items())[:3]
            )
            context_parts.append(f"用户偏好: {prefs}")

        # 历史记忆
        for mem, score in zip(result.memories, result.scores):
            if score > 0.6:
                context_parts.append(f"[历史记忆] {mem.content}")

        return "\n".join(context_parts) if context_parts else "暂无历史上下文"

    async def end_session(
        self,
        session_id: str,
        user_id: str,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        会话结束时的处理流程

        1. 生成会话摘要
        2. 提取跨会话记忆
        3. 更新用户画像
        4. 清理临时数据
        """
        # 1. 生成摘要
        summary = await self.create_session_summary(
            session_id, user_id, messages
        )

        # 2. 提取跨会话记忆
        new_facts = await self.extract_cross_session_facts(
            user_id, summary
        )

        # 3. 更新用户画像
        profile = await self.update_user_profile(user_id, summary)

        result = {
            'session_id': session_id,
            'summary': summary.summary_text,
            'new_facts_count': len(new_facts),
            'profile_topics': len(profile.topics_of_interest),
            'key_facts': summary.key_facts,
        }

        logger.info(f"会话结束处理完成: {result}")
        return result
```

### 8.5.3 跨会话记忆数据流

```
[当前会话消息] ──→ [会话摘要生成] ──→ [跨会话事实提取]
                                          │
                    ┌─────────────────────┤
                    ↓                     ↓
            [与已有记忆对比]        [用户画像更新]
                    │                     │
              ┌─────┴─────┐               ↓
              ↓           ↓          [持久化存储]
         [更新已有]   [创建新记忆]
              │           │
              └─────┬─────┘
                    ↓
              [置信度更新]
                    │
                    └──→ [下一次会话可检索]
```

> 跨会话记忆的核心挑战在于**信息的一致性和时效性**。用户可能在不同会话中表达矛盾的观点，或者他们的情况发生了变化。通过置信度机制和定期验证，系统可以逐步建立对用户准确、及时的理解。

---

## 8.6 记忆持久化与序列化

### 8.6.1 多后端存储适配器

```python
import pickle
import json
import hashlib
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class InMemoryStore(MemoryStore):
    """
    内存存储实现
    适合开发测试和小规模场景
    """

    def __init__(self):
        self._store: Dict[str, Memory] = {}
        self._session_index: Dict[str, List[str]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True
        logger.info("内存存储初始化完成")

    async def store(self, memory: Memory) -> str:
        self._store[memory.id] = memory
        # 维护会话索引
        if memory.session_id:
            if memory.session_id not in self._session_index:
                self._session_index[memory.session_id] = []
            self._session_index[memory.session_id].append(memory.id)
        return memory.id

    async def retrieve(self, memory_id: str) -> Optional[Memory]:
        return self._store.get(memory_id)

    async def search(self, query: MemoryQuery) -> MemorySearchResult:
        """简单的内存搜索（生产环境应使用向量索引）"""
        import time
        start = time.time()

        candidates = list(self._store.values())

        # 类型过滤
        if query.memory_types:
            candidates = [
                m for m in candidates
                if m.memory_type in query.memory_types
            ]

        # 会话过滤
        if query.session_id:
            candidates = [
                m for m in candidates
                if m.session_id == query.session_id
            ]

        # 简单的文本匹配评分
        scored = []
        for mem in candidates:
            score = 0.0
            if query.query_text.lower() in mem.content.lower():
                score = 0.8
            elif any(
                term in mem.content.lower()
                for term in query.query_text.lower().split()
            ):
                score = 0.5

            # 结合衰减因子
            if query.include_decay:
                mem.update_access()
                score *= mem.decay_factor

            if score >= query.min_relevance:
                scored.append((mem, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:query.top_k]

        elapsed = (time.time() - start) * 1000
        return MemorySearchResult(
            memories=[m for m, _ in results],
            scores=[s for _, s in results],
            query_time_ms=elapsed,
            total_candidates=len(candidates),
        )

    async def update(self, memory: Memory) -> bool:
        if memory.id in self._store:
            self._store[memory.id] = memory
            return True
        return False

    async def delete(self, memory_id: str) -> bool:
        if memory_id in self._store:
            del self._store[memory_id]
            return True
        return False

    async def list_by_session(
        self, session_id: str, limit: int = 100
    ) -> List[Memory]:
        ids = self._session_index.get(session_id, [])[:limit]
        return [self._store[i] for i in ids if i in self._store]

    async def get_stats(self) -> Dict[str, Any]:
        return {
            'total_memories': len(self._store),
            'total_sessions': len(self._session_index),
            'memory_types': {},
        }


class RedisMemoryStore(MemoryStore):
    """
    Redis存储实现
    适合分布式部署，支持TTL
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "agent_memory:",
        default_ttl: int = 86400 * 7,  # 默认7天过期
    ):
        self.redis_url = redis_url
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.redis = None

    async def initialize(self) -> None:
        import aioredis
        self.redis = await aioredis.from_url(
            self.redis_url, decode_responses=False
        )
        logger.info(f"Redis存储初始化完成: {self.redis_url}")

    async def store(self, memory: Memory) -> str:
        key = f"{self.prefix}{memory.id}"
        data = self._serialize_memory(memory)
        await self.redis.setex(key, self.default_ttl, data)

        # 维护会话索引
        if memory.session_id:
            index_key = f"{self.prefix}session:{memory.session_id}"
            await self.redis.sadd(index_key, memory.id)
            await self.redis.expire(index_key, self.default_ttl)

        return memory.id

    async def retrieve(self, memory_id: str) -> Optional[Memory]:
        key = f"{self.prefix}{memory_id}"
        data = await self.redis.get(key)
        if data is None:
            return None
        return self._deserialize_memory(data)

    async def search(self, query: MemoryQuery) -> MemorySearchResult:
        """Redis搜索（使用SCAN + 应用层过滤）"""
        import time
        start = time.time()

        # 扫描所有匹配的键
        pattern = f"{self.prefix}[0-9a-f]*"
        cursor = 0
        candidates = []

        while True:
            cursor, keys = await self.redis.scan(
                cursor, match=pattern, count=100
            )
            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode()
                # 跳过索引键
                if "session:" in key or "index:" in key:
                    continue
                data = await self.redis.get(key)
                if data:
                    mem = self._deserialize_memory(data)
                    candidates.append(mem)
            if cursor == 0:
                break

        # 应用过滤
        if query.memory_types:
            candidates = [
                m for m in candidates
                if m.memory_type in query.memory_types
            ]

        # 简单评分
        scored = []
        for mem in candidates:
            score = 0.0
            if query.query_text.lower() in mem.content.lower():
                score = 0.8
            scored.append((mem, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:query.top_k]

        elapsed = (time.time() - start) * 1000
        return MemorySearchResult(
            memories=[m for m, _ in results],
            scores=[s for _, s in results],
            query_time_ms=elapsed,
            total_candidates=len(candidates),
        )

    async def update(self, memory: Memory) -> bool:
        key = f"{self.prefix}{memory.id}"
        if await self.redis.exists(key):
            data = self._serialize_memory(memory)
            await self.redis.setex(key, self.default_ttl, data)
            return True
        return False

    async def delete(self, memory_id: str) -> bool:
        key = f"{self.prefix}{memory_id}"
        result = await self.redis.delete(key)
        return result > 0

    async def list_by_session(
        self, session_id: str, limit: int = 100
    ) -> List[Memory]:
        index_key = f"{self.prefix}session:{session_id}"
        member_ids = await self.redis.smembers(index_key)
        memories = []
        for mid in list(member_ids)[:limit]:
            if isinstance(mid, bytes):
                mid = mid.decode()
            mem = await self.retrieve(mid)
            if mem:
                memories.append(mem)
        return memories

    async def get_stats(self) -> Dict[str, Any]:
        keys = await self.redis.keys(f"{self.prefix}*")
        return {
            'total_keys': len(keys),
            'redis_memory': await self.redis.info('memory'),
        }

    def _serialize_memory(self, memory: Memory) -> bytes:
        """序列化记忆对象"""
        data = {
            'id': memory.id,
            'content': memory.content,
            'memory_type': memory.memory_type.value,
            'embedding': memory.embedding,
            'metadata': memory.metadata,
            'created_at': memory.created_at.isoformat(),
            'accessed_at': (
                memory.accessed_at.isoformat()
                if memory.accessed_at else None
            ),
            'access_count': memory.access_count,
            'relevance_score': memory.relevance_score,
            'decay_factor': memory.decay_factor,
            'tags': memory.tags,
            'source': memory.source,
            'session_id': memory.session_id,
            'agent_id': memory.agent_id,
        }
        return json.dumps(data).encode('utf-8')

    def _deserialize_memory(self, data: bytes) -> Memory:
        """反序列化记忆对象"""
        data = json.loads(data.decode('utf-8'))
        return Memory(
            id=data['id'],
            content=data['content'],
            memory_type=MemoryType(data['memory_type']),
            embedding=data.get('embedding'),
            metadata=data.get('metadata', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            accessed_at=(
                datetime.fromisoformat(data['accessed_at'])
                if data.get('accessed_at') else None
            ),
            access_count=data.get('access_count', 0),
            relevance_score=data.get('relevance_score', 1.0),
            decay_factor=data.get('decay_factor', 1.0),
            tags=data.get('tags', []),
            source=data.get('source', ''),
            session_id=data.get('session_id', ''),
            agent_id=data.get('agent_id', ''),
        )

    async def close(self) -> None:
        if self.redis:
            await self.redis.close()
```

### 8.6.2 序列化策略对比

| 格式 | 速度 | 体积 | 可读性 | 向量支持 | 适用场景 |
|------|------|------|--------|----------|----------|
| JSON | 中 | 大 | 高 | 需转换 | API交互、调试 |
| Pickle | 快 | 中 | 无 | 原生 | Python内部传输 |
| MessagePack | 快 | 小 | 无 | 需转换 | 高性能网络传输 |
 | Protobuf | 很快 | 小 | 无 | 原生 | 分布式系统 |
 | BSON | 快 | 中 | 无 | 需转换 | MongoDB集成 |

 > 持久化不只是"把数据写到磁盘"。一个好的持久化方案需要考虑：**序列化效率、查询性能、数据一致性、容灾恢复**四个维度。在生产环境中，我们通常使用JSON作为主要序列化格式，因为它在可读性和性能之间取得了良好的平衡。

工具使用模式分析对于优化 Agent 性能至关重要。下面的交互式可视化展示了常见的工具调用模式：

<div data-component="ToolUsagePatterns"></div>

 ---

 ## 8.7 记忆一致性与冲突解决

### 8.7.1 冲突检测与分类

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """记忆冲突类型"""
    TEMPORAL = "temporal"          # 时间冲突：新旧信息不一致
    CONTRADICTORY = "contradictory"  # 逻辑矛盾：A说X，非A说非X
    OVERLAPPING = "overlapping"    # 信息重叠：重复但不完全相同
    PARTIAL = "partial"            # 部分更新：新信息只更新了旧信息的一部分


@dataclass
class MemoryConflict:
    """记忆冲突描述"""
    conflict_id: str
    conflict_type: ConflictType
    memory_a: Memory
    memory_b: Memory
    similarity_score: float
    conflict_description: str
    suggested_resolution: str


class ConflictResolver:
    """
    记忆冲突解决器

    负责检测、分类和解决记忆之间的冲突
    """

    def __init__(
        self,
        embedding_fn: Any = None,
        llm_fn: Any = None,
        auto_resolve: bool = True,
    ):
        self.embedding_fn = embedding_fn
        self.llm_fn = llm_fn
        self.auto_resolve = auto_resolve

    async def detect_conflicts(
        self,
        memories: List[Memory],
        threshold: float = 0.8,
    ) -> List[MemoryConflict]:
        """
        检测记忆集合中的冲突

        冲突检测策略:
        1. 高相似度检测：内容高度相似但不完全相同
        2. 时间排序分析：后置信息可能更新前置信息
        3. 语义矛盾检测：使用LLM判断是否存在矛盾
        """
        conflicts = []

        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                mem_a = memories[i]
                mem_b = memories[j]

                # 计算相似度
                similarity = await self._calculate_similarity(
                    mem_a.content, mem_b.content
                )

                if similarity >= threshold:
                    # 检测具体冲突类型
                    conflict = await self._classify_conflict(
                        mem_a, mem_b, similarity
                    )
                    if conflict:
                        conflicts.append(conflict)

        return conflicts

    async def _calculate_similarity(
        self, text_a: str, text_b: str
    ) -> float:
        """计算文本相似度"""
        if self.embedding_fn:
            try:
                embeddings = await self.embedding_fn([text_a, text_b])
                if len(embeddings) >= 2:
                    return self._cosine_similarity(
                        embeddings[0], embeddings[1]
                    )
            except Exception:
                pass

        # 回退到简单的Jaccard相似度
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0.0

    def _cosine_similarity(
        self, vec_a: List[float], vec_b: List[float]
    ) -> float:
        """余弦相似度计算"""
        import numpy as np
        a = np.array(vec_a)
        b = np.array(vec_b)
        dot = np.dot(a, b)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        return float(dot / (norm_a * norm_b)) if (norm_a * norm_b) > 0 else 0.0

    async def _classify_conflict(
        self,
        mem_a: Memory,
        mem_b: Memory,
        similarity: float,
    ) -> Optional[MemoryConflict]:
        """分类冲突类型"""
        # 时间判断
        time_a = mem_a.created_at
        time_b = mem_b.created_at

        # 如果内容几乎完全相同，是重叠
        if similarity > 0.95:
            return MemoryConflict(
                conflict_id=f"conflict_{mem_a.id}_{mem_b.id}",
                conflict_type=ConflictType.OVERLAPPING,
                memory_a=mem_a,
                memory_b=mem_b,
                similarity_score=similarity,
                conflict_description="两条记忆内容高度重叠",
                suggested_resolution="保留更新的一条，删除旧的",
            )

        # 如果内容部分相似且时间有先后，可能是部分更新
        if 0.8 <= similarity <= 0.95:
            newer = mem_b if time_b > time_a else mem_a
            older = mem_a if time_b > time_a else mem_b

            return MemoryConflict(
                conflict_id=f"conflict_{mem_a.id}_{mem_b.id}",
                conflict_type=ConflictType.PARTIAL,
                memory_a=older,
                memory_b=newer,
                similarity_score=similarity,
                conflict_description=f"记忆 '{newer.content[:30]}' 可能更新了 '{older.content[:30]}'",
                suggested_resolution="保留新记忆，标记旧记忆为过期",
            )

        # 使用LLM检测逻辑矛盾
        if self.llm_fn and similarity > 0.7:
            is_contradictory = await self._check_contradiction(
                mem_a.content, mem_b.content
            )
            if is_contradictory:
                newer = mem_b if time_b > time_a else mem_a
                older = mem_a if time_b > time_a else mem_b
                return MemoryConflict(
                    conflict_id=f"conflict_{mem_a.id}_{mem_b.id}",
                    conflict_type=ConflictType.CONTRADICTORY,
                    memory_a=older,
                    memory_b=newer,
                    similarity_score=similarity,
                    conflict_description=f"存在逻辑矛盾",
                    suggested_resolution="保留更新的信息，删除旧信息",
                )

        return None

    async def _check_contradiction(
        self, text_a: str, text_b: str
    ) -> bool:
        """使用LLM检测两段文本是否矛盾"""
        prompt = f"""判断以下两段文本是否存在逻辑矛盾。只回答"是"或"否"。

文本A: {text_a}
文本B: {text_b}

两段文本是否矛盾？"""

        try:
            response = await self.llm_fn(prompt)
            return "是" in str(response)
        except Exception:
            return False

    async def resolve_conflicts(
        self,
        conflicts: List[MemoryConflict],
        memory_store: MemoryStore,
    ) -> Dict[str, Any]:
        """
        解决检测到的冲突

        策略:
        1. 重叠 → 保留最新，删除旧的
        2. 部分更新 → 合并信息，标记旧的为过期
        3. 矛盾 → 保留新信息，删除旧信息
        4. 时间冲突 → 保留更新的时间戳
        """
        stats = {'resolved': 0, 'merged': 0, 'deleted': 0, 'skipped': 0}

        for conflict in conflicts:
            try:
                if conflict.conflict_type == ConflictType.OVERLAPPING:
                    # 保留更新的
                    newer = (
                        conflict.memory_b
                        if conflict.memory_b.created_at > conflict.memory_a.created_at
                        else conflict.memory_a
                    )
                    older = (
                        conflict.memory_a
                        if newer == conflict.memory_b
                        else conflict.memory_b
                    )
                    await memory_store.delete(older.id)
                    stats['deleted'] += 1

                elif conflict.conflict_type == ConflictType.PARTIAL:
                    # 合并信息
                    merged_content = await self._merge_memories(
                        conflict.memory_a, conflict.memory_b
                    )
                    newer = max(
                        conflict.memory_a, conflict.memory_b,
                        key=lambda m: m.created_at
                    )
                    newer.content = merged_content
                    newer.metadata['merged_from'] = [
                        conflict.memory_a.id,
                        conflict.memory_b.id,
                    ]
                    await memory_store.update(newer)
                    await memory_store.delete(
                        min(
                            conflict.memory_a, conflict.memory_b,
                            key=lambda m: m.created_at
                        ).id
                    )
                    stats['merged'] += 1

                elif conflict.conflict_type == ConflictType.CONTRADICTORY:
                    # 保留新信息
                    newer = max(
                        conflict.memory_a, conflict.memory_b,
                        key=lambda m: m.created_at
                    )
                    older = min(
                        conflict.memory_a, conflict.memory_b,
                        key=lambda m: m.created_at
                    )
                    newer.metadata['supersedes'] = older.id
                    await memory_store.update(newer)
                    await memory_store.delete(older.id)
                    stats['deleted'] += 1

                stats['resolved'] += 1

            except Exception as e:
                logger.error(f"冲突解决失败: {e}")
                stats['skipped'] += 1

        return stats

    async def _merge_memories(
        self, mem_a: Memory, mem_b: Memory
    ) -> str:
        """合并两条记忆的内容"""
        if self.llm_fn:
            prompt = f"""请合并以下两段信息，保留所有重要细节，去除重复内容。

信息A: {mem_a.content}
信息B: {mem_b.content}

合并后的信息:"""

            try:
                response = await self.llm_fn(prompt)
                return str(response)
            except Exception:
                pass

        # 简单合并
        return f"{mem_a.content} {mem_b.content}"
```

### 8.7.2 一致性保障机制

```python
class MemoryConsistencyManager:
    """
    记忆一致性管理器

    提供以下保障:
    1. 事务性写入：要么全部成功，要么全部回滚
    2. 版本控制：每次更新递增版本号
    3. 乐观锁：防止并发写入冲突
    4. 审计日志：记录所有修改操作
    """

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self._audit_log: List[Dict[str, Any]] = []

    async def transactional_store(
        self, memories: List[Memory]
    ) -> Dict[str, Any]:
        """
        事务性存储多条记忆

        任何一条失败都会回滚所有已存储的记忆
        """
        stored_ids = []
        success = True

        try:
            for memory in memories:
                memory_id = await self.memory_store.store(memory)
                stored_ids.append(memory_id)
                self._log_operation('store', memory_id, memory.content[:50])

        except Exception as e:
            success = False
            logger.error(f"事务存储失败，回滚 {len(stored_ids)} 条记录")
            for mid in stored_ids:
                await self.memory_store.delete(mid)
            raise

        return {
            'success': success,
            'stored_count': len(stored_ids),
            'stored_ids': stored_ids,
        }

    async def versioned_update(
        self,
        memory_id: str,
        new_content: str,
        expected_version: int,
    ) -> Tuple[bool, Optional[Memory]]:
        """
        乐观锁版本控制更新

        只有当当前版本号匹配预期值时才执行更新
        防止并发修改导致的数据丢失
        """
        current = await self.memory_store.retrieve(memory_id)
        if current is None:
            return False, None

        current_version = current.metadata.get('version', 0)

        if current_version != expected_version:
            logger.warning(
                f"版本冲突: 期望 {expected_version}, "
                f"实际 {current_version}"
            )
            return False, current

        # 执行更新
        current.content = new_content
        current.metadata['version'] = current_version + 1
        current.metadata['last_updated_at'] = datetime.now().isoformat()

        await self.memory_store.update(current)
        self._log_operation(
            'update', memory_id,
            f"v{current_version} -> v{current_version + 1}"
        )

        return True, current

    def _log_operation(
        self, operation: str, memory_id: str, detail: str
    ) -> None:
        """记录审计日志"""
        self._audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'memory_id': memory_id,
            'detail': detail,
        })

    def get_audit_log(
        self, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取审计日志"""
        return self._audit_log[-limit:]
```

 > 记忆一致性是Agent系统中最容易被忽视但又最关键的问题。一个没有一致性保障的记忆系统，就像一个没有事务的数据库——在并发环境下，数据损坏几乎是必然的。乐观锁和版本控制是解决这类问题的标准范式。

Agent 规划能力对于处理复杂任务至关重要。下面的交互式可视化展示了 Agent 规划树的工作原理：

<div data-component="AgentPlanningDemoV8"></div>

 ---

 ## 8.8 生产环境记忆系统架构

### 8.8.1 整体架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                       生产级记忆系统架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Agent层  │───→│ 记忆管理层   │───→│ 存储适配层           │  │
│  └──────────┘    └──────────────┘    └──────────────────────┘  │
│       │               │                      │                  │
│       ▼               ▼                      ▼                  │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ API网关  │    │ 冲突解决器   │    │ PostgreSQL (主存储)  │  │
│  │ (限流)   │    │ 一致性管理   │    │ Redis (缓存)        │  │
│  └──────────┘    │ 衰减管理器   │    │ 向量数据库           │  │
│                  └──────────────┘    │ (Milvus/Qdrant)     │  │
│                                      └──────────────────────┘  │
│                                              │                  │
│                                              ▼                  │
│                                      ┌──────────────────────┐  │
│                                      │ 监控与告警           │  │
│                                      │ Prometheus + Grafana │  │
│                                      └──────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.8.2 生产级配置与部署

```python
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class MemorySystemConfig:
    """记忆系统生产配置"""

    # 环境
    environment: Environment = Environment.DEVELOPMENT

    # 数据库配置
    postgres_dsn: str = "postgresql://localhost:5432/agent_memory"
    redis_url: str = "redis://localhost:6379/0"
    vector_db_url: str = "http://localhost:19530"

    # 记忆管理配置
    working_memory_size: int = 20
    long_term_memory_limit: int = 50
    importance_threshold: float = 0.3
    decay_threshold: float = 0.1
    consolidation_interval_hours: int = 6

    # 检索配置
    default_top_k: int = 10
    similarity_threshold: float = 0.3
    rrf_k: int = 60

    # 缓存配置
    cache_ttl_seconds: int = 300
    cache_max_size: int = 1000

    # 监控配置
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"

    # 安全配置
    enable_encryption: bool = False
    encryption_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "MemorySystemConfig":
        """从环境变量加载配置"""
        env = os.getenv("AGENT_ENV", "development")
        return cls(
            environment=Environment(env),
            postgres_dsn=os.getenv(
                "POSTGRES_DSN",
                "postgresql://localhost:5432/agent_memory"
            ),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            vector_db_url=os.getenv(
                "VECTOR_DB_URL", "http://localhost:19530"
            ),
            working_memory_size=int(
                os.getenv("WORKING_MEMORY_SIZE", "20")
            ),
            long_term_memory_limit=int(
                os.getenv("LONG_TERM_MEMORY_LIMIT", "50")
            ),
            enable_metrics=os.getenv("ENABLE_METRICS", "true") == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def validate(self) -> bool:
        """验证配置有效性"""
        assert self.working_memory_size > 0
        assert self.long_term_memory_limit > 0
        assert 0 < self.importance_threshold < 1
        assert 0 < self.decay_threshold < 1
        return True


class MemorySystemFactory:
    """
    记忆系统工厂

    根据配置自动构建完整的记忆管理系统
    """

    @staticmethod
    async def create(
        config: MemorySystemConfig,
        embedding_fn: Any,
        llm_fn: Any,
    ) -> HybridMemoryManager:
        """
        创建完整的记忆管理系统

        返回: 配置好的混合记忆管理器实例
        """
        config.validate()

        # 选择存储后端
        if config.environment == Environment.DEVELOPMENT:
            store = InMemoryStore()
        elif config.environment == Environment.STAGING:
            store = RedisMemoryStore(
                redis_url=config.redis_url
            )
        else:  # PRODUCTION
            store = PostgresMemoryStore(
                dsn=config.postgres_dsn,
                embedding_fn=embedding_fn,
            )

        await store.initialize()

        # 创建管理器
        manager = HybridMemoryManager(
            memory_store=store,
            embedding_fn=embedding_fn,
            working_memory_size=config.working_memory_size,
            long_term_memory_limit=config.long_term_memory_limit,
            importance_threshold=config.importance_threshold,
        )

        await manager.initialize()
        return manager
```

### 8.8.3 性能优化策略

```python
import time
from functools import wraps
from typing import Callable, Any
import asyncio


class MemoryCache:
    """
    记忆缓存层

    使用LRU策略缓存热点记忆查询结果
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_order: List[str] = []

    def _make_key(self, query: MemoryQuery) -> str:
        """生成缓存键"""
        import hashlib
        key_parts = [
            query.query_text,
            str(query.top_k),
            str(query.memory_types),
            str(query.tags),
        ]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: MemoryQuery) -> Optional[MemorySearchResult]:
        """获取缓存"""
        key = self._make_key(query)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                # 更新访问顺序
                self._access_order.remove(key)
                self._access_order.append(key)
                return result
            else:
                # 过期，删除
                del self._cache[key]
                self._access_order.remove(key)
        return None

    def set(self, query: MemoryQuery, result: MemorySearchResult) -> None:
        """设置缓存"""
        key = self._make_key(query)

        # 如果缓存已满，淘汰最久未访问的
        if len(self._cache) >= self.max_size:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]

        self._cache[key] = (result, time.time())
        self._access_order.append(key)

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._access_order.clear()

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        # 简化实现，实际应记录命中/未命中次数
        return 0.0


def cached_query(ttl: int = 300):
    """查询缓存装饰器"""
    cache = MemoryCache(ttl_seconds=ttl)

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(
            self, query: MemoryQuery, *args, **kwargs
        ) -> MemorySearchResult:
            # 尝试从缓存获取
            cached = cache.get(query)
            if cached is not None:
                return cached

            # 执行查询
            result = await func(self, query, *args, **kwargs)

            # 存入缓存
            cache.set(query, result)
            return result
        return wrapper
    return decorator
```

### 8.8.4 监控与告警

```python
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List
from collections import defaultdict
import threading


@dataclass
class MetricsCollector:
    """
    记忆系统指标收集器

    收集关键性能指标:
    - 存储操作延迟
    - 检索QPS和延迟分布
    - 缓存命中率
    - 内存使用量
    """

    _counters: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _histograms: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _gauges: Dict[str, float] = field(
        default_factory=dict
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock
    )

    def record_latency(self, operation: str, latency_ms: float) -> None:
        """记录操作延迟"""
        with self._lock:
            self._histograms[f"{operation}_latency"].append(latency_ms)
            # 保留最近1000个样本
            if len(self._histograms[f"{operation}_latency"]) > 1000:
                self._histograms[f"{operation}_latency"] = \
                    self._histograms[f"{operation}_latency"][-1000:]

    def increment_counter(self, name: str, value: int = 1) -> None:
        """增加计数器"""
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        """设置仪表盘值"""
        with self._lock:
            self._gauges[name] = value

    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        summary = {
            'counters': dict(self._counters),
            'gauges': dict(self._gauges),
            'histograms': {},
        }

        for name, values in self._histograms.items():
            if values:
                sorted_vals = sorted(values)
                summary['histograms'][name] = {
                    'count': len(values),
                    'min': sorted_vals[0],
                    'max': sorted_vals[-1],
                    'avg': sum(values) / len(values),
                    'p50': sorted_vals[len(sorted_vals) // 2],
                    'p95': sorted_vals[int(len(sorted_vals) * 0.95)],
                    'p99': sorted_vals[int(len(sorted_vals) * 0.99)],
                }

        return summary


class HealthChecker:
    """
    记忆系统健康检查

    定期检查各组件的健康状态
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        check_interval_seconds: int = 60,
    ):
        self.memory_store = memory_store
        self.check_interval = check_interval_seconds
        self._health_status: Dict[str, str] = {}

    async def check_all(self) -> Dict[str, str]:
        """执行所有健康检查"""
        checks = {
            'store': await self._check_store(),
            'memory_count': await self._check_memory_count(),
        }
        self._health_status = checks
        return checks

    async def _check_store(self) -> str:
        """检查存储后端连通性"""
        try:
            # 尝试执行一个简单操作
            stats = await self.memory_store.get_stats()
            return 'healthy'
        except Exception as e:
            logger.error(f"存储健康检查失败: {e}")
            return f'unhealthy: {str(e)[:50]}'

    async def _check_memory_count(self) -> str:
        """检查记忆数量是否合理"""
        try:
            stats = await self.memory_store.get_stats()
            count = stats.get('total_memories', 0)
            if count > 1_000_000:
                return f'warning: {count} memories (too many)'
            elif count == 0:
                return 'info: no memories stored'
            return f'healthy: {count} memories'
        except Exception:
            return 'unknown'

    def get_health_report(self) -> Dict[str, Any]:
        """生成健康报告"""
        return {
            'status': 'healthy' if all(
                v.startswith('healthy') or v.startswith('info')
                for v in self._health_status.values()
            ) else 'degraded',
            'checks': self._health_status,
            'timestamp': datetime.now().isoformat(),
        }
```

> 生产级记忆系统不是一个单独的组件，而是一个完整的生态系统。它需要**存储、缓存、检索、一致性、监控**五个层面的协同工作。任何一层的缺失都可能成为系统的单点故障。

---

## 8.9 本章小结

本章从工程实践的角度，完整实现了AI Agent记忆系统的核心模块：

1. **统一接口设计**：通过抽象基类定义了MemoryStore和MemoryRetriever接口，确保了存储后端的可替换性和检索策略的可插拔性。

2. **混合记忆管理器**：实现了整合工作记忆、短期记忆、长期记忆的完整管理器，支持自动记忆整合和智能筛选。

3. **LangGraph Checkpointer**：深入分析了LangGraph的状态快照机制，并实现了支持压缩和自动清理的生产级PostgreSQL Checkpointer。

4. **混合检索策略**：实现了语义检索、关键词检索、元数据过滤三种策略，并通过RRF算法进行融合，达到了单一策略无法实现的检索效果。

5. **跨会话记忆**：通过会话摘要生成、事实提取、用户画像更新等机制，实现了记忆的跨会话延续。

6. **持久化与序列化**：对比了多种序列化格式，实现了Redis和内存两种存储后端的适配器。

7. **一致性与冲突解决**：通过冲突检测、分类、解决三级机制，保障了记忆系统在并发环境下的数据一致性。

8. **生产环境架构**：给出了完整的生产级架构设计，包括配置管理、性能优化、监控告警等工程实践。

ReAct 是 Agent 推理与行动交替的经典框架。下面的交互式演示展示了完整的 ReAct 工作流程：

<div data-component="ReActDemoV6"></div>

### 关键技术要点回顾

| 技术点 | 核心思想 | 实现方案 |
|--------|----------|----------|
| 记忆衰减 | 模拟人类遗忘曲线 | Ebbinghaus遗忘曲线 + 指数衰减 |
| 重要性评分 | 多因素加权评估 | 时效性+频率+相关性+情感+信息密度 |
| 混合检索 | 多策略融合 | RRF算法融合多种排序结果 |
| 冲突解决 | 时序+语义双重判断 | 分类检测 + LLM辅助判断 |
| 一致性保障 | 乐观锁+版本控制 | 事务写入 + 审计日志 |

---

## 8.10 思考题

### 基础题

1. **接口设计**：为什么MemoryStore要使用抽象基类而不是直接定义具体类？如果需要支持新的存储后端（如MongoDB），需要修改哪些代码？

2. **遗忘曲线**：Ebbinghaus遗忘曲线公式 $$R = e^{-t/S}$$ 中，稳定性因子S受哪些因素影响？如何根据实际使用场景调整S的计算方式？

3. **RRF融合**：Reciprocal Rank Fusion公式 $$RRF_{score} = \sum \frac{1}{k + rank_i}$$ 中，参数k的取值如何影响融合结果？尝试分析k=1和k=100时的区别。

### 进阶题

4. **记忆膨胀**：当系统运行数月后，长期记忆可能积累到数百万条。设计一个记忆压缩策略，在保持检索质量的前提下减少存储量。

5. **隐私保护**：跨会话记忆中可能包含用户的敏感信息。设计一个机制，在记忆存储前自动检测并脱敏敏感数据。

6. **分布式一致性**：在多Agent系统中，多个Agent可能同时修改同一用户的记忆。设计一个分布式锁机制来保证一致性。

### 开放题

7. **记忆迁移**：如果用户要从一个Agent平台迁移到另一个平台，如何设计记忆数据的标准化导出和导入格式？

8. **多模态记忆**：当前的记忆系统主要处理文本。如何扩展支持图像、音频等多模态记忆？数据结构和检索策略需要如何调整？

9. **记忆评估**：设计一套评估指标，衡量Agent记忆系统的"智能程度"。从检索准确率、响应时间、存储效率、用户满意度四个维度思考。

---

> 本章代码虽然可以直接用于生产环境，但每个项目的具体需求不同。建议读者从本章代码出发，根据自己的业务场景调整配置参数和策略权重。记忆工程没有银弹，只有最适合你场景的方案。
