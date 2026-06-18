---
title: "第7章：记忆系统 — Agent 的记忆与学习"
description: "全面掌握 Agent 记忆架构：认知科学基础、短期记忆、工作记忆、长期记忆、情景记忆与语义记忆的分离与协同，Buffer Memory、Window Memory、Summary Memory、Vector Memory 详解，记忆衰减机制，LangChain Memory 模块实战，Mem0 生产级记忆框架。"
date: "2026-06-15"
---

 # 第7章：记忆系统 — Agent 的记忆与学习

 记忆系统让 Agent 能够记住历史对话、学习用户偏好、积累知识。本章深入探讨记忆系统的架构设计。

 下面的交互式图表展示了 Agent 记忆系统的三层架构：

 <div data-component="MemoryArchitecture"></div>

 ## 7.1 认知科学基础：人类记忆理论

### 7.1.1 记忆的本质

> "记忆不是储存在大脑某个特定位置的静态信息，而是由多个相互协作的神经系统动态重建的过程。"
> —— *Cognitive Psychology: A Student's Handbook*

记忆是人类认知能力的核心支柱。从认知科学的角度来看，记忆是指神经系统对信息的编码、存储和提取的能力。对于 AI Agent 而言，理解人类记忆的工作机制是设计高效记忆系统的理论基础。

人类记忆系统具有以下关键特征：

1. **编码（Encoding）**：将外部信息转化为神经系统可以处理的形式
2. **存储（Storage）**：在神经系统中保持信息的过程
3. **提取（Retrieval）**：从存储中访问和使用信息的能力

这三个过程在 Agent 记忆系统中有着直接的对应：

```python
# Agent 记忆系统的三阶段过程
class AgentMemorySystem:
    """Agent 记忆系统的抽象实现"""
    
    def __init__(self):
        self.short_term = []  # 短期存储
        self.long_term = []   # 长期存储
        self.index = {}       # 检索索引
    
    def encode(self, information: dict) -> str:
        """
        编码阶段：将原始信息转化为结构化表示
        类似于人类大脑对感觉信息的初步处理
        """
        # 信息清洗和标准化
        cleaned = self._clean(information)
        
        # 生成唯一标识符
        memory_id = self._generate_id(cleaned)
        
        # 创建记忆条目
        memory_entry = {
            "id": memory_id,
            "content": cleaned["content"],
            "timestamp": cleaned.get("timestamp"),
            "importance": self._calculate_importance(cleaned),
            "tags": self._extract_tags(cleaned)
        }
        
        # 存入短期记忆
        self.short_term.append(memory_entry)
        
        # 建立检索索引
        self.index[memory_id] = {
            "location": "short_term",
            "tags": memory_entry["tags"]
        }
        
        return memory_id
    
    def store(self, memory_id: str, long_term: bool = False):
        """
        存储阶段：将信息持久化
        如果是重要信息，转移到长期记忆
        """
        # 查找记忆条目
        entry = self._find_entry(memory_id)
        if not entry:
            return False
        
        if long_term:
            # 转移到长期记忆
            self.long_term.append(entry)
            self.short_term.remove(entry)
            self.index[memory_id]["location"] = "long_term"
        
        return True
    
    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        提取阶段：根据查询检索相关记忆
        结合关键词匹配和语义相似度
        """
        results = []
        
        # 关键词匹配
        keyword_matches = self._keyword_search(query)
        
        # 语义相似度匹配（模拟向量检索）
        semantic_matches = self._semantic_search(query)
        
        # 融合结果
        combined = self._merge_results(keyword_matches, semantic_matches)
        
        # 按相关性排序
        ranked = sorted(combined, key=lambda x: x["score"], reverse=True)
        
        return ranked[:top_k]
    
    def _clean(self, information: dict) -> dict:
        """数据清洗"""
        # 移除多余空白
        content = information.get("content", "").strip()
        # 标准化格式
        return {"content": content, "timestamp": information.get("timestamp")}
    
    def _generate_id(self, entry: dict) -> str:
        """生成唯一标识符"""
        import hashlib
        content = entry["content"] + str(entry.get("timestamp", ""))
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _calculate_importance(self, entry: dict) -> float:
        """计算信息重要性分数"""
        # 基于内容长度、关键词等计算
        content = entry["content"]
        score = 0.5  # 基础分数
        
        # 包含问号可能表示需要回答
        if "?" in content or "？" in content:
            score += 0.2
        
        # 包含关键动词
        key_verbs = ["记住", "保存", "重要", "注意"]
        for verb in key_verbs:
            if verb in content:
                score += 0.1
        
        return min(score, 1.0)
    
    def _extract_tags(self, entry: dict) -> list:
        """提取标签"""
        tags = []
        content = entry["content"]
        
        # 简单的关键词提取
        topic_keywords = ["代码", "函数", "变量", "算法", "数据库"]
        for kw in topic_keywords:
            if kw in content:
                tags.append(kw)
        
        return tags
    
    def _find_entry(self, memory_id: str) -> dict:
        """查找记忆条目"""
        for entry in self.short_term:
            if entry["id"] == memory_id:
                return entry
        for entry in self.long_term:
            if entry["id"] == memory_id:
                return entry
        return None
    
    def _keyword_search(self, query: str) -> list:
        """关键词搜索"""
        results = []
        query_terms = set(query.split())
        
        all_entries = self.short_term + self.long_term
        for entry in all_entries:
            content_terms = set(entry["content"].split())
            overlap = query_terms & content_terms
            if overlap:
                score = len(overlap) / len(query_terms)
                results.append({"entry": entry, "score": score})
        
        return results
    
    def _semantic_search(self, query: str) -> list:
        """语义搜索（模拟）"""
        # 实际应用中会使用向量相似度
        results = []
        all_entries = self.short_term + self.long_term
        
        for entry in all_entries:
            # 简化的相似度计算
            score = self._simple_similarity(query, entry["content"])
            if score > 0.3:
                results.append({"entry": entry, "score": score})
        
        return results
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """简化的文本相似度计算"""
        words1 = set(text1)
        words2 = set(text2)
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0
    
    def _merge_results(self, results1: list, results2: list) -> list:
        """合并搜索结果"""
        merged = {}
        for r in results1:
            entry_id = r["entry"]["id"]
            merged[entry_id] = {"entry": r["entry"], "score": r["score"]}
        
        for r in results2:
            entry_id = r["entry"]["id"]
            if entry_id in merged:
                merged[entry_id]["score"] += r["score"]
            else:
                merged[entry_id] = {"entry": r["entry"], "score": r["score"]}
        
        return list(merged.values())
```

### 7.1.2 Atkinson-Shiffrin 多存储模型

Atkinson 和 Shiffrin 在 1968 年提出的多存储模型（Multi-Store Model）是记忆研究领域最具影响力的理论之一。该模型将记忆分为三个主要存储系统：

**1. 感觉记忆（Sensory Memory）**

感觉记忆是信息处理的第一站，负责暂存来自感官的原始信息。其特点是：
- 存储时间极短（通常 < 1 秒）
- 存储容量较大
- 信息以感觉模态的形式保存

在 Agent 系统中，感觉记忆对应的是：
```python
# 感觉记忆：API 原始响应的临时缓冲
class SensoryBuffer:
    """API 响应的临时存储，类似感觉记忆"""
    
    def __init__(self, ttl_seconds: float = 1.0):
        self.buffer = {}
        self.ttl = ttl_seconds
        self.timestamps = {}
    
    def store(self, key: str, data: any):
        """存储原始响应"""
        import time
        self.buffer[key] = data
        self.timestamps[key] = time.time()
    
    def retrieve(self, key: str) -> any:
        """检索（可能已过期）"""
        import time
        if key not in self.buffer:
            return None
        
        # 检查是否过期
        if time.time() - self.timestamps[key] > self.ttl:
            # 过期数据，自动清除
            del self.buffer[key]
            del self.timestamps[key]
            return None
        
        return self.buffer[key]
```

**2. 短期记忆（Short-Term Memory）**

短期记忆负责临时存储当前正在处理的信息。其特点是：
- 存储时间约 15-30 秒
- 容量有限（Miller's Law：7±2 个组块）
- 需要主动维持（如复述）

 **3. 长期记忆（Long-Term Memory）**
 长期记忆是信息的永久存储仓库。其特点是：
 - 存储时间理论上无限
 - 容量理论上无限
 - 信息经过编码后可以长期保存

感知-决策-行动循环是 Agent 与环境交互的核心。下面的交互式可视化展示了完整的 PDA 循环：

<div data-component="PDAFlowChartV5"></div>

 ### 7.1.3 Baddeley 工作记忆模型

Alan Baddeley 在 1974 年提出的工作记忆模型是对 Atkinson-Shiffrin 模型的重要扩展。该模型将短期记忆进一步细分为多个组件：

```python
class WorkingMemoryModel:
    """
    Baddeley 工作记忆模型的实现
    包含中央执行系统、语音环路、视空间画板、情景缓冲区
    """
    
    def __init__(self):
        # 中央执行系统：注意力控制和资源分配
        self.central_executive = CentralExecutive()
        
        # 语音环路：处理听觉和语言信息
        self.phonological_loop = PhonologicalLoop()
        
        # 视空间画板：处理视觉和空间信息
        self.visuospatial_sketchpad = VisuospatialSketchpad()
        
        # 情景缓冲区：整合多模态信息
        self.episodic_buffer = EpisodicBuffer()
    
    def process_input(self, input_data: dict) -> dict:
        """
        处理输入信息
        根据信息类型分配到不同的子系统
        """
        results = {}
        
        # 语言/音频信息
        if input_data.get("type") == "audio":
            # 语音环路处理
            processed = self.phonological_loop.process(input_data["content"])
            results["phonological"] = processed
        
        # 视觉/空间信息
        elif input_data.get("type") == "visual":
            # 视空间画板处理
            processed = self.visuospatial_sketchpad.process(input_data["content"])
            results["visuospatial"] = processed
        
        # 综合信息
        else:
            # 情景缓冲区整合
            integrated = self.episodic_buffer.integrate(input_data)
            results["episodic"] = integrated
        
        # 中央执行系统协调
        self.central_executive.allocate_attention(results)
        
        return results


class CentralExecutive:
    """中央执行系统"""
    
    def __init__(self):
        self.attention_focus = None
        self.task_queue = []
    
    def allocate_attention(self, information: dict):
        """分配注意力资源"""
        # 选择最重要的信息
        if information:
            priority = self._calculate_priority(information)
            self.attention_focus = max(priority, key=priority.get)
    
    def _calculate_priority(self, information: dict) -> dict:
        """计算优先级"""
        priority = {}
        for key, value in information.items():
            # 基于新颖性和重要性计算优先级
            priority[key] = len(str(value))
        return priority


class PhonologicalLoop:
    """语音环路"""
    
    def __init__(self):
        self.phonological_store = []  # 语音存储
        self.articulatory_process = None  # 发音复述过程
    
    def process(self, content: str) -> list:
        """处理语音信息"""
        # 将文本分解为语音单元
        phonemes = self._text_to_phonemes(content)
        
        # 存入语音存储
        self.phonological_store.extend(phonemes)
        
        # 通过发音复述维持
        self._articulatory_rehearsal()
        
        return self.phonological_store[-10:]  # 返回最近10个
    
    def _text_to_phonemes(self, text: str) -> list:
        """文本转音素（简化）"""
        return list(text)  # 简化为字符
    
    def _articulatory_rehearsal(self):
        """发音复述"""
        # 模拟内部语音循环
        if self.phonological_store:
            self.articulatory_process = self.phonological_store.copy()


class VisuospatialSketchpad:
    """视空间画板"""
    
    def __init__(self):
        self.visual_cache = []  # 视觉缓存
        self.spatial_map = {}   # 空间映射
    
    def process(self, content: str) -> dict:
        """处理视觉空间信息"""
        # 提取视觉特征
        visual_features = self._extract_visual(content)
        
        # 更新空间映射
        self.spatial_map[content] = {
            "features": visual_features,
            "position": len(self.visual_cache)
        }
        
        self.visual_cache.append(content)
        
        return self.spatial_map[content]
    
    def _extract_visual(self, content: str) -> dict:
        """提取视觉特征（简化）"""
        return {
            "length": len(content),
            "complexity": content.count(" ") + 1
        }


class EpisodicBuffer:
    """情景缓冲区"""
    
    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self.episodes = []
    
    def integrate(self, information: dict) -> list:
        """整合多模态信息为情景"""
        episode = {
            "content": information,
            "timestamp": self._get_timestamp(),
            "context": self._get_context()
        }
        
        self.episodes.append(episode)
        
        # 保持容量限制
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)
        
        return self.episodes
    
    def _get_timestamp(self):
        import time
        return time.time()
    
    def _get_context(self):
        return {"episodes_count": len(self.episodes)}
```

### 7.1.4 Tulving 的记忆分类

Endel Tulving 在 1972 年提出了情景记忆和语义记忆的区分，这是长期记忆研究的里程碑：

| 记忆类型 | 定义 | 特点 | Agent 实现 |
|:---|:---|:---|:---|
| **情景记忆** | 对个人经历的具体事件的记忆 | 包含时间、地点、情感等上下文 | 历史对话存储 |
| **语义记忆** | 对一般知识和概念的记忆 | 与个人经历无关的客观知识 | 知识库、文档索引 |

这种区分对 Agent 记忆系统设计有重要指导意义：

```python
class TulvingMemorySystem:
    """
    基于 Tulving 分类的记忆系统
    分离情景记忆和语义记忆
    """
    
    def __init__(self):
        # 情景记忆：存储具体经历
        self.episodic_memory = EpisodicMemory()
        
        # 语义记忆：存储一般知识
        self.semantic_memory = SemanticMemory()
    
    def record_experience(self, experience: dict):
        """
        记录一次经历（情景记忆）
        """
        episode = {
            "what": experience.get("content"),
            "when": experience.get("timestamp"),
            "where": experience.get("location"),
            "emotional_state": experience.get("emotion", "neutral"),
            "participants": experience.get("participants", [])
        }
        
        self.episodic_memory.store(episode)
    
    def learn_knowledge(self, fact: dict):
        """
        学习新知识（语义记忆）
        """
        knowledge = {
            "concept": fact.get("concept"),
            "definition": fact.get("definition"),
            "relations": fact.get("relations", []),
            "confidence": fact.get("confidence", 1.0)
        }
        
        self.semantic_memory.store(knowledge)
    
    def recall(self, query: str, memory_type: str = "auto") -> dict:
        """
        根据查询检索记忆
        自动判断应该检索哪种记忆类型
        """
        if memory_type == "auto":
            # 判断查询类型
            if self._is_experience_query(query):
                return self.episodic_memory.search(query)
            else:
                return self.semantic_memory.search(query)
        elif memory_type == "episodic":
            return self.episodic_memory.search(query)
        else:
            return self.semantic_memory.search(query)
    
    def _is_experience_query(self, query: str) -> bool:
        """判断是否是经历查询"""
        experience_indicators = ["之前", "上次", "昨天", "第一次", "记得"]
        return any(indicator in query for indicator in experience_indicators)


class EpisodicMemory:
    """情景记忆实现"""
    
    def __init__(self):
        self.episodes = []
    
    def store(self, episode: dict):
        """存储情景"""
        episode["id"] = len(self.episodes)
        self.episodes.append(episode)
    
    def search(self, query: str) -> list:
        """搜索情景记忆"""
        results = []
        for episode in self.episodes:
            score = self._relevance_score(query, episode)
            if score > 0.3:
                results.append({"episode": episode, "score": score})
        
        return sorted(results, key=lambda x: x["score"], reverse=True)
    
    def _relevance_score(self, query: str, episode: dict) -> float:
        """计算相关性分数"""
        # 基于关键词匹配
        query_words = set(query)
        content_words = set(str(episode.get("what", "")))
        
        intersection = query_words & content_words
        union = query_words | content_words
        
        return len(intersection) / len(union) if union else 0


class SemanticMemory:
    """语义记忆实现"""
    
    def __init__(self):
        self.knowledge_base = []
        self.concept_index = {}
    
    def store(self, knowledge: dict):
        """存储知识"""
        knowledge["id"] = len(self.knowledge_base)
        self.knowledge_base.append(knowledge)
        
        # 建立概念索引
        concept = knowledge.get("concept", "")
        if concept not in self.concept_index:
            self.concept_index[concept] = []
        self.concept_index[concept].append(knowledge["id"])
    
    def search(self, query: str) -> list:
        """搜索语义记忆"""
        results = []
        
        # 先尝试精确匹配概念
        if query in self.concept_index:
            for kid in self.concept_index[query]:
                results.append({
                    "knowledge": self.knowledge_base[kid],
                    "score": 1.0
                })
        
        # 再进行模糊搜索
        for knowledge in self.knowledge_base:
            score = self._relevance_score(query, knowledge)
            if score > 0.3 and not any(r["knowledge"]["id"] == knowledge["id"] for r in results):
                results.append({"knowledge": knowledge, "score": score})
        
        return sorted(results, key=lambda x: x["score"], reverse=True)
    
    def _relevance_score(self, query: str, knowledge: dict) -> float:
        """计算相关性分数"""
        concept = knowledge.get("concept", "")
        definition = knowledge.get("definition", "")
        
        all_text = concept + " " + definition
        query_chars = set(query)
        text_chars = set(all_text)
        
        intersection = query_chars & text_chars
        union = query_chars | text_chars
        
        return len(intersection) / len(union) if union else 0
```

### 7.1.5 Ebbinghaus 遗忘曲线

Hermann Ebbinghaus 在 1885 年通过实验发现了著名的遗忘曲线，揭示了记忆随时间衰减的规律：

> "记忆的保持量随时间呈指数衰减，但在每次复习后，衰减速率会降低。"

遗忘曲线的数学表达式为：

$$R = e^{-t/S}$$

其中：
- $R$ 是记忆保持率（Retention）
- $t$ 是自上次学习后的时间
- $S$ 是记忆强度（Strength）

这个公式启发了 Agent 记忆衰减机制的设计：

```python
import math
from datetime import datetime, timedelta
from typing import Optional


class ForgettingCurve:
    """
    Ebbinghaus 遗忘曲线的实现
    用于 Agent 记忆衰减机制
    """
    
    def __init__(self, initial_strength: float = 1.0):
        """
        初始化遗忘曲线
        
        Args:
            initial_strength: 初始记忆强度 (0-1)
        """
        self.initial_strength = initial_strength
        self.stability = 1.0  # 记忆稳定性
    
    def get_retention(self, time_elapsed: float) -> float:
        """
        计算记忆保持率
        
        Args:
            time_elapsed: 经过的时间（小时）
        
        Returns:
            记忆保持率 (0-1)
        """
        # R = e^(-t/S)
        retention = math.exp(-time_elapsed / self.stability)
        return max(0, min(1, retention))
    
    def strengthen(self, amount: float):
        """
        加强记忆（类似复习效果）
        
        Args:
            amount: 加强的强度
        """
        # 复习会增加记忆稳定性
        self.stability *= (1 + amount)
        self.initial_strength = min(1.0, self.initial_strength + amount * 0.1)
    
    def get_optimal_review_time(self, target_retention: float = 0.8) -> float:
        """
        计算最佳复习时间
        
        Args:
            target_retention: 目标保持率
        
        Returns:
            建议的复习间隔（小时）
        """
        # 反解遗忘曲线公式
        # R = e^(-t/S) => t = -S * ln(R)
        if target_retention <= 0 or target_retention >= 1:
            return 0
        
        optimal_time = -self.stability * math.log(target_retention)
        return max(0, optimal_time)


class AdaptiveForgettingCurve:
    """
    自适应遗忘曲线
    根据信息的重要性动态调整衰减速度
    """
    
    def __init__(self):
        self.curves = {}  # 不同重要性级别的曲线
    
    def get_curve(self, importance: str) -> ForgettingCurve:
        """
        获取指定重要性级别的遗忘曲线
        
        Args:
            importance: 重要性级别 (low, medium, high, critical)
        
        Returns:
            对应的遗忘曲线
        """
        if importance not in self.curves:
            # 不同重要性对应不同的记忆稳定性
            stability_map = {
                "low": 1.0,      # 低重要性：快速遗忘
                "medium": 5.0,   # 中等重要性：中等遗忘速度
                "high": 20.0,    # 高重要性：慢速遗忘
                "critical": 100.0  # 关键信息：极慢遗忘
            }
            
            self.curves[importance] = ForgettingCurve(
                initial_strength=1.0
            )
            self.curves[importance].stability = stability_map.get(importance, 5.0)
        
        return self.curves[importance]
    
    def calculate_memory_strength(
        self,
        created_at: datetime,
        importance: str,
        last_accessed: Optional[datetime] = None,
        access_count: int = 0
    ) -> float:
        """
        计算记忆的当前强度
        
        Args:
            created_at: 创建时间
            importance: 重要性级别
            last_accessed: 最后访问时间
            access_count: 访问次数
        
        Returns:
            当前记忆强度 (0-1)
        """
        curve = self.get_curve(importance)
        
        # 计算自创建以来的时间
        now = datetime.now()
        hours_elapsed = (now - created_at).total_seconds() / 3600
        
        # 基础保持率
        base_retention = curve.get_retention(hours_elapsed)
        
        # 访问次数加成（每次访问相当于一次复习）
        access_boost = 1.0 + 0.1 * min(access_count, 10)
        
        # 最后访问时间加成（最近访问过会增强记忆）
        recency_boost = 1.0
        if last_accessed:
            hours_since_access = (now - last_accessed).total_seconds() / 3600
            recency_boost = 1.0 + 0.2 * math.exp(-hours_since_access / 24)
        
        final_strength = base_retention * access_boost * recency_boost
        
        return max(0, min(1, final_strength))
```

---

## 7.2 人类记忆模型类比表

### 7.2.1 详细对比表格

下表详细对比了人类记忆系统与 Agent 记忆系统的各个组件：

| 人类记忆类型 | 持续时间 | 容量 | 编码方式 | Agent 对应 | 实现方式 | 典型应用场景 |
|:---|:---|:---|:---|:---|:---|:---|
| **感觉记忆** | <1秒 | 大 | 感觉模态 | API 原始返回 | 内存缓冲区 | 数据预处理 |
| **瞬时记忆** | 1-2秒 | 有限 | 视觉/听觉 | 请求/响应缓存 | LRU Cache | 重复查询优化 |
| **短期记忆** | 15-30秒 | 7±2组块 | 听觉/语义 | 当前对话上下文 | 消息列表 | 多轮对话 |
| **工作记忆** | 中等 | 受限 | 多模态 | Agent Scratchpad | 状态变量 | 任务执行 |
| **情景记忆** | 永久 | 大 | 情感/情境 | 历史对话存储 | 向量数据库 | 个性化推荐 |
| **语义记忆** | 永久 | 极大 | 概念/规则 | 知识库/文档 | 图数据库/RAG | 知识问答 |
| **程序性记忆** | 永久 | 大 | 自动化 | 工具使用经验 | Few-shot 示例 | 任务自动化 |

### 7.2.2 记忆特性对比

| 特性 | 人类记忆 | Agent 记忆 | 差异分析 |
|:---|:---|:---|:---|
| **编码速度** | 毫秒级 | 纳秒级 | Agent 更快 |
| **存储容量** | 理论无限 | 受硬件限制 | Agent 受限 |
| **检索方式** | 关联/模式匹配 | 精确/语义搜索 | 各有优势 |
| **遗忘机制** | 主动遗忘 | 被动过期 | 人类更智能 |
| **情感标记** | 强烈 | 无/弱 | 人类更丰富 |
| **创造性整合** | 强 | 弱 | 人类更灵活 |
| **并行处理** | 有限 | 强 | Agent 更优 |
| **持久性** | 可能退化 | 可精确控制 | Agent 更可靠 |

### 7.2.3 记忆容量对比

| 记忆类型 | 人类容量 | Agent 容量 | 扩展策略 |
|:---|:---|:---|:---|
| 短期记忆 | 7±2 组块 | 可配置（如 10-100 条） | 增大窗口/摘要压缩 |
| 工作记忆 | 4 个组块 | 可配置状态变量 | 分层管理 |
| 长期记忆 | 理论无限 | 受存储限制 | 分布式存储、压缩 |

### 7.2.4 编码方式对比

```python
# 人类记忆编码方式与 Agent 实现的对比
ENCODING_COMPARISON = {
    "视觉编码": {
        "human": "视网膜成像，特征提取",
        "agent": "图像嵌入向量 (CLIP, ResNet)",
        "example": "处理用户上传的图片"
    },
    "听觉编码": {
        "human": "声波转换为神经信号",
        "agent": "音频转文本 (Whisper)",
        "example": "语音助手处理语音输入"
    },
    "语义编码": {
        "human": "理解含义后存储",
        "agent": "文本嵌入 (BERT, GPT embeddings)",
        "example": "理解用户查询的意图"
    },
    "情境编码": {
        "human": "关联时间、地点、情感",
        "agent": "元数据存储 (时间戳、上下文)",
        "example": "记住对话发生的场景"
    },
    "程序编码": {
        "human": "通过重复练习自动化",
        "agent": "Few-shot 示例学习",
        "example": "学习如何调用特定API"
    }
}
```

### 7.2.5 遗忘机制对比

| 遗忘类型 | 人类机制 | Agent 实现 | 设计考量 |
|:---|:---|:---|:---|
| **衰退** | 时间导致痕迹消退 | TTL 过期 | 平衡存储成本 |
| **干扰** | 相似信息覆盖 | 语义相似度去重 | 保持信息独特性 |
| **压抑** | 动机性遗忘 | 优先级过滤 | 用户隐私保护 |
| **提取失败** | 线索不足 | 检索失败 | 改进索引策略 |
 | **编码失败** | 注意力不集中 | 输入验证 | 提高数据质量 |

RL Agent 和 LLM Agent 在设计理念上有本质区别。下面的交互式对比展示了两者的差异：

<div data-component="AgentCapabilityMatrixV6"></div>

 ---

 ## 7.3 短期记忆与长期记忆

### 7.3.1 短期记忆的特征

短期记忆（Short-Term Memory, STM）是信息处理系统中的临时存储区域。在 Agent 系统中，短期记忆通常对应当前的对话上下文窗口。

**短期记忆的关键特征：**

1. **有限容量**：根据 Miller's Law，人类短期记忆容量为 7±2 个组块
2. **短暂持续**：信息只能保持 15-30 秒，除非主动复述
3. **编码方式**：主要以语音形式编码（听觉环路）

**Agent 短期记忆的实现：**

```python
from typing import List, Dict, Optional
from datetime import datetime
import json


class ShortTermMemory:
    """
    Agent 短期记忆实现
    用于存储当前对话的上下文信息
    """
    
    def __init__(self, capacity: int = 20):
        """
        初始化短期记忆
        
        Args:
            capacity: 最大存储容量（消息数量）
        """
        self.capacity = capacity
        self.messages: List[Dict] = []
        self.metadata: Dict = {}
    
    def add_message(self, role: str, content: str, **kwargs) -> bool:
        """
        添加消息到短期记忆
        
        Args:
            role: 消息角色 (user/assistant/system)
            content: 消息内容
            **kwargs: 额外元数据
        
        Returns:
            是否添加成功
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        # 检查容量限制
        if len(self.messages) >= self.capacity:
            # 移除最早的消息
            self.messages.pop(0)
        
        self.messages.append(message)
        
        return True
    
    def get_context(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        获取对话上下文
        
        Args:
            last_n: 获取最近N条消息，None表示全部
        
        Returns:
            消息列表
        """
        if last_n is None:
            return self.messages.copy()
        
        return self.messages[-last_n:]
    
    def clear(self):
        """清空短期记忆"""
        self.messages.clear()
        self.metadata.clear()
    
    def get_token_count(self, tokenizer=None) -> int:
        """
        估算当前记忆的token数量
        
        Args:
            tokenizer: 分词器（可选）
        
        Returns:
            估算的token数量
        """
        total_text = " ".join([msg["content"] for msg in self.messages])
        
        if tokenizer:
            return len(tokenizer.encode(total_text))
        
        # 简单估算：1个中文字约2个token，1个英文词约1个token
        chinese_chars = sum(1 for c in total_text if '\u4e00' <= c <= '\u9fff')
        english_words = len(total_text.split()) - chinese_chars
        return chinese_chars * 2 + english_words
    
    def to_langchain_format(self) -> List[Dict]:
        """
        转换为 LangChain 格式
        
        Returns:
            LangChain 兼容的消息列表
        """
        return [
            {"type": msg["role"], "content": msg["content"]}
            for msg in self.messages
        ]
    
    def export(self) -> str:
        """导出为JSON字符串"""
        return json.dumps(self.messages, ensure_ascii=False, indent=2)
    
    def import_messages(self, json_str: str):
        """从JSON字符串导入消息"""
        self.messages = json.loads(json_str)
```

### 7.3.2 长期记忆的特征

长期记忆（Long-Term Memory, LSTM）是信息的持久存储系统。在 Agent 系统中，长期记忆通常实现为向量数据库、图数据库或传统数据库。

**长期记忆的关键特征：**

1. **无限容量**：理论上可以存储无限量的信息
2. **持久性**：信息可以保存很长时间
3. **编码方式**：主要以语义形式编码

**Agent 长期记忆的实现：**

```python
from typing import List, Dict, Optional, Any
from datetime import datetime
import hashlib
import json


class LongTermMemory:
    """
    Agent 长期记忆实现
    用于存储持久化的知识和经历
    """
    
    def __init__(self, storage_backend: str = "dict"):
        """
        初始化长期记忆
        
        Args:
            storage_backend: 存储后端 (dict/vector_db/graph_db)
        """
        self.storage_backend = storage_backend
        self.memories: Dict[str, Dict] = {}
        self.index: Dict[str, List[str]] = {}  # 标签索引
    
    def store(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        tags: List[str] = None,
        metadata: Dict = None
    ) -> str:
        """
        存储记忆到长期记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型 (episodic/semantic/procedural)
            importance: 重要性分数 (0-1)
            tags: 标签列表
            metadata: 额外元数据
        
        Returns:
            记忆ID
        """
        # 生成唯一ID
        memory_id = self._generate_id(content)
        
        # 创建记忆条目
        memory_entry = {
            "id": memory_id,
            "content": content,
            "type": memory_type,
            "importance": importance,
            "tags": tags or [],
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": None,
            "strength": 1.0
        }
        
        # 存储
        self.memories[memory_id] = memory_entry
        
        # 更新索引
        for tag in memory_entry["tags"]:
            if tag not in self.index:
                self.index[tag] = []
            self.index[tag].append(memory_id)
        
        return memory_id
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None
    ) -> List[Dict]:
        """
        检索相关记忆
        
        Args:
            query: 查询内容
            top_k: 返回最相关的K条记忆
            memory_type: 过滤记忆类型
        
        Returns:
            检索结果列表
        """
        results = []
        
        for memory_id, memory in self.memories.items():
            # 类型过滤
            if memory_type and memory["type"] != memory_type:
                continue
            
            # 计算相关性分数
            score = self._calculate_relevance(query, memory)
            
            if score > 0:
                results.append({
                    "memory": memory,
                    "score": score
                })
                
                # 更新访问记录
                memory["access_count"] += 1
                memory["last_accessed"] = datetime.now().isoformat()
        
        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def forget(
        self,
        strategy: str = "strength_threshold",
        threshold: float = 0.3
    ) -> int:
        """
        遗忘低重要性记忆
        
        Args:
            strategy: 遗忘策略 (strength_threshold/oldest/least_accessed)
            threshold: 强度阈值
        
        Returns:
            遗忘的记忆数量
        """
        to_forget = []
        
        if strategy == "strength_threshold":
            for memory_id, memory in self.memories.items():
                if memory["strength"] < threshold:
                    to_forget.append(memory_id)
        
        elif strategy == "oldest":
            # 按创建时间排序，删除最旧的
            sorted_memories = sorted(
                self.memories.items(),
                key=lambda x: x[1]["created_at"]
            )
            to_forget = [mid for mid, _ in sorted_memories[:len(sorted_memories)//4]]
        
        elif strategy == "least_accessed":
            # 按访问次数排序，删除最少访问的
            sorted_memories = sorted(
                self.memories.items(),
                key=lambda x: x[1]["access_count"]
            )
            to_forget = [mid for mid, _ in sorted_memories[:len(sorted_memories)//4]]
        
        # 执行删除
        for memory_id in to_forget:
            self._delete_memory(memory_id)
        
        return len(to_forget)
    
    def update_strength(self, memory_id: str, decay_rate: float = 0.1):
        """
        更新记忆强度（衰减机制）
        
        Args:
            memory_id: 记忆ID
            decay_rate: 衰减速率
        """
        if memory_id not in self.memories:
            return
        
        memory = self.memories[memory_id]
        
        # 计算衰减
        hours_since_creation = self._hours_since(memory["created_at"])
        hours_since_access = self._hours_since(memory["last_accessed"]) if memory["last_accessed"] else hours_since_creation
        
        # 基础衰减
        time_decay = math.exp(-decay_rate * hours_since_creation)
        
        # 访问加成
        access_boost = 1.0 + 0.1 * min(memory["access_count"], 10)
        
        # 更新强度
        memory["strength"] = memory["importance"] * time_decay * access_boost
        memory["strength"] = max(0, min(1, memory["strength"]))
    
    def get_statistics(self) -> Dict:
        """获取记忆统计信息"""
        if not self.memories:
            return {"total": 0}
        
        types = {}
        strengths = []
        
        for memory in self.memories.values():
            # 统计类型分布
            mem_type = memory["type"]
            types[mem_type] = types.get(mem_type, 0) + 1
            
            # 收集强度值
            strengths.append(memory["strength"])
        
        return {
            "total": len(self.memories),
            "by_type": types,
            "avg_strength": sum(strengths) / len(strengths),
            "min_strength": min(strengths),
            "max_strength": max(strengths)
        }
    
    def _generate_id(self, content: str) -> str:
        """生成唯一ID"""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _calculate_relevance(self, query: str, memory: Dict) -> float:
        """计算查询与记忆的相关性"""
        # 简化的关键词匹配
        query_chars = set(query)
        content_chars = set(memory["content"])
        
        intersection = query_chars & content_chars
        union = query_chars | content_chars
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _delete_memory(self, memory_id: str):
        """删除记忆"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            
            # 从索引中移除
            for tag in memory["tags"]:
                if tag in self.index and memory_id in self.index[tag]:
                    self.index[tag].remove(memory_id)
            
            # 删除记忆
            del self.memories[memory_id]
    
    def _hours_since(self, iso_timestamp: str) -> float:
        """计算自给定时间以来的小时数"""
        if not iso_timestamp:
            return 0
        
        timestamp = datetime.fromisoformat(iso_timestamp)
        delta = datetime.now() - timestamp
        return delta.total_seconds() / 3600


import math
```

### 7.3.3 短期与长期记忆的协作

短期记忆和长期记忆在 Agent 系统中需要紧密协作。典型的工作流程包括：

1. **输入处理**：新信息首先存入短期记忆
2. **重要性评估**：评估信息的重要性
3. **记忆巩固**：重要信息从短期记忆转移到长期记忆
4. **上下文增强**：检索长期记忆来增强当前上下文

```python
class MemoryConsolidationSystem:
    """
    记忆巩固系统
    管理短期记忆和长期记忆之间的信息流动
    """
    
    def __init__(self):
        self.short_term = ShortTermMemory(capacity=50)
        self.long_term = LongTermMemory()
        self.consolidation_threshold = 0.6  # 巩固阈值
    
    def process_input(self, role: str, content: str):
        """
        处理输入信息
        """
        # 添加到短期记忆
        self.short_term.add_message(role, content)
        
        # 评估是否需要巩固
        importance = self._evaluate_importance(content)
        
        if importance >= self.consolidation_threshold:
            # 巩固到长期记忆
            self.consolidate_to_long_term(content, importance)
    
    def consolidate_to_long_term(self, content: str, importance: float):
        """
        将信息巩固到长期记忆
        """
        # 提取标签
        tags = self._extract_tags(content)
        
        # 判断记忆类型
        memory_type = self._determine_memory_type(content)
        
        # 存储到长期记忆
        self.long_term.store(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags
        )
    
    def get_enhanced_context(self, query: str) -> List[Dict]:
        """
        获取增强的上下文
        结合短期记忆和从长期记忆检索的相关信息
        """
        # 获取短期记忆
        context = self.short_term.get_context()
        
        # 检索长期记忆
        relevant_memories = self.long_term.retrieve(query, top_k=3)
        
        # 整合
        if relevant_memories:
            # 添加相关记忆作为上下文
            memory_context = [
                {"type": "system", "content": f"相关记忆: {m['memory']['content']}"}
                for m in relevant_memories
            ]
            context = memory_context + context
        
        return context
    
    def _evaluate_importance(self, content: str) -> float:
        """
        评估内容重要性
        """
        importance = 0.3  # 基础分数
        
        # 包含关键动词
        if any(word in content for word in ["记住", "重要", "注意", "关键"]):
            importance += 0.3
        
        # 包含问题
        if "?" in content or "？" in content:
            importance += 0.2
        
        # 内容长度
        if len(content) > 100:
            importance += 0.1
        
        return min(importance, 1.0)
    
    def _extract_tags(self, content: str) -> List[str]:
        """提取标签"""
        tags = []
        
        # 基于关键词提取标签
        keyword_tags = {
            "代码": ["编程", "技术"],
            "函数": ["编程", "数学"],
            "用户": ["交互", "体验"],
            "数据": ["分析", "存储"]
        }
        
        for keyword, tag_list in keyword_tags.items():
            if keyword in content:
                tags.extend(tag_list)
        
        return list(set(tags))
    
    def _determine_memory_type(self, content: str) -> str:
        """判断记忆类型"""
        # 经历类
        if any(word in content for word in ["之前", "上次", "记得"]):
            return "episodic"
        
        # 知识类
        if any(word in content for word in ["定义", "原理", "方法"]):
            return "semantic"
        
        # 程序类
        if any(word in content for word in ["步骤", "流程", "如何"]):
            return "procedural"
        
        return "episodic"
```

---

## 7.4 Buffer Memory 详解

### 7.4.1 概念与原理

Buffer Memory（缓冲记忆）是最基础的记忆类型，它完整保存对话历史中的所有消息。就像人类的短期记忆一样，Buffer Memory 提供了一个完整的对话上下文窗口。

**核心特点：**
- 保存所有历史消息
- 不进行任何压缩或摘要
- 提供完整的对话上下文
- 受限于 LLM 的上下文窗口大小

### 7.4.2 实现原理

```python
from typing import List, Dict, Optional
from datetime import datetime
import json


class ConversationBufferMemory:
    """
    对话缓冲记忆的完整实现
    完整保存所有历史消息
    """
    
    def __init__(
        self,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        memory_key: str = "history",
        return_messages: bool = False
    ):
        """
        初始化缓冲记忆
        
        Args:
            human_prefix: 人类消息前缀
            ai_prefix: AI消息前缀
            memory_key: 记忆键名
            return_messages: 是否返回消息对象列表
        """
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.memory_key = memory_key
        self.return_messages = return_messages
        
        # 存储
        self.buffer: List[Dict] = []
        self.context: Optional[str] = None
    
    def save_context(self, inputs: Dict, outputs: Dict):
        """
        保存对话上下文
        
        Args:
            inputs: 输入信息 (包含 human_input)
            outputs: 输出信息 (包含 response)
        """
        human_input = inputs.get("input", inputs.get("human_input", ""))
        ai_output = outputs.get("response", outputs.get("output", ""))
        
        # 构建消息
        human_message = {
            "role": "human",
            "content": human_input,
            "timestamp": datetime.now().isoformat()
        }
        
        ai_message = {
            "role": "ai",
            "content": ai_output,
            "timestamp": datetime.now().isoformat()
        }
        
        # 添加到缓冲区
        self.buffer.append(human_message)
        self.buffer.append(ai_message)
    
    def load_memory_variables(self, inputs: Dict = None) -> Dict:
        """
        加载记忆变量
        
        Args:
            inputs: 输入上下文
        
        Returns:
            包含历史消息的字典
        """
        if self.return_messages:
            return {self.memory_key: self.buffer}
        else:
            # 转换为字符串格式
            history = self._get_string_history()
            return {self.memory_key: history}
    
    def _get_string_history(self) -> str:
        """
        获取字符串格式的历史记录
        """
        lines = []
        for message in self.buffer:
            if message["role"] == "human":
                prefix = self.human_prefix
            else:
                prefix = self.ai_prefix
            
            lines.append(f"{prefix}: {message['content']}")
        
        return "\n".join(lines)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
    
    def get_token_count(self, tokenizer=None) -> int:
        """
        估算历史记录的token数量
        """
        history = self._get_string_history()
        
        if tokenizer:
            return len(tokenizer.encode(history))
        
        # 简单估算
        return len(history) // 2  # 假设平均2字符1token
    
    def truncate_to_fit(self, max_tokens: int, tokenizer=None):
        """
        截断历史记录以适应token限制
        
        Args:
            max_tokens: 最大token数量
            tokenizer: 分词器
        """
        current_tokens = self.get_token_count(tokenizer)
        
        while current_tokens > max_tokens and len(self.buffer) > 2:
            # 移除最早的一轮对话（2条消息）
            self.buffer.pop(0)
            self.buffer.pop(0)
            current_tokens = self.get_token_count(tokenizer)
    
    def export(self) -> str:
        """导出为JSON字符串"""
        return json.dumps(self.buffer, ensure_ascii=False, indent=2)
    
    def import_buffer(self, json_str: str):
        """从JSON字符串导入"""
        self.buffer = json.loads(json_str)
    
    def __len__(self) -> int:
        """返回缓冲区消息数量"""
        return len(self.buffer)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"ConversationBufferMemory(messages={len(self.buffer)})"
```

### 7.4.3 使用示例

```python
# 使用示例
def demo_buffer_memory():
    """演示缓冲记忆的使用"""
    
    # 创建记忆实例
    memory = ConversationBufferMemory(
        human_prefix="用户",
        ai_prefix="助手",
        return_messages=True
    )
    
    # 模拟对话
    conversations = [
        ("你好，我想学习Python", "你好！很高兴帮助你学习Python。你想从哪个方面开始？"),
        ("我想学习数据结构", "好的！Python中常用的数据结构包括列表、字典、集合和元组。"),
        ("能给我一个列表的例子吗？", "当然！列表是Python中最常用的数据结构之一。"),
    ]
    
    for human_input, ai_response in conversations:
        memory.save_context(
            inputs={"input": human_input},
            outputs={"response": ai_response}
        )
    
    # 加载记忆
    memory_vars = memory.load_memory_variables()
    
    print("历史记录：")
    print(memory_vars["history"])
    
    # 获取token数量
    token_count = memory.get_token_count()
    print(f"\n估计token数量: {token_count}")
    
    return memory


# 运行示例
if __name__ == "__main__":
    memory = demo_buffer_memory()
```

### 7.4.4 优缺点分析

**优点：**
- 实现简单，易于理解
- 保留完整上下文，信息无损失
- 适合短对话场景

**缺点：**
- 受限于 LLM 上下文窗口
- 长对话时 token 消耗大
- 可能包含无关信息

### 7.4.5 最佳实践

```python
class OptimizedBufferMemory:
    """
    优化的缓冲记忆
    添加了实用的优化策略
    """
    
    def __init__(self, max_messages: int = 50):
        self.buffer = []
        self.max_messages = max_messages
        
        # 关键信息标记
        self.key_information = []
    
    def add_with_importance(self, role: str, content: str, importance: float = 0.5):
        """
        带重要性标记的消息添加
        """
        message = {
            "role": role,
            "content": content,
            "importance": importance,
            "timestamp": datetime.now().isoformat()
        }
        
        self.buffer.append(message)
        
        # 高重要性消息额外标记
        if importance > 0.7:
            self.key_information.append(len(self.buffer) - 1)
        
        # 容量管理
        self._enforce_capacity()
    
    def _enforce_capacity(self):
        """强制执行容量限制"""
        if len(self.buffer) > self.max_messages:
            # 保留高重要性消息
            to_remove = len(self.buffer) - self.max_messages
            
            # 从低重要性消息开始删除
            sorted_indices = sorted(
                range(len(self.buffer)),
                key=lambda i: self.buffer[i]["importance"]
            )
            
            for i in range(to_remove):
                idx = sorted_indices[i]
                self.buffer[idx] = None  # 标记删除
            
            # 清理
            self.buffer = [m for m in self.buffer if m is not None]
            self.key_information = [
                i for i in self.key_information
                if i < len(self.buffer)
            ]
    
    def get_filtered_context(self, query: str = None) -> List[Dict]:
        """
        获取过滤后的上下文
        根据查询相关性过滤
        """
        if query is None:
            return self.buffer
        
        # 基于查询相关性排序
        scored_messages = []
        for msg in self.buffer:
            relevance = self._calculate_relevance(query, msg["content"])
            scored_messages.append((relevance, msg))
        
        # 按相关性排序，保留最相关的
        scored_messages.sort(key=lambda x: x[0], reverse=True)
        
         return [msg for _, msg in scored_messages[:20]]
     }
     # ... 省略其他方法

不同的推理策略适用于不同的场景。下面的交互式工具可以帮助你选择最合适的推理策略：

<div data-component="ReasoningStrategySelectorV7"></div>

 ---

 ## 7.5 Window Memory 详解

### 7.5.1 概念与原理

Window Memory（窗口记忆）是 Buffer Memory 的改进版本，它只保留最近 N 轮对话，丢弃更早的历史。这模拟了人类记忆中的"近因效应"——最近发生的事情更容易被记住。

**核心特点：**
- 只保留最近 K 轮对话
- 自动丢弃早期历史
- 节省 token 消耗
- 适合长对话场景

### 7.5.2 实现原理

```python
from typing import List, Dict, Optional
from collections import deque
from datetime import datetime


class ConversationBufferWindowMemory:
    """
    对话窗口记忆的完整实现
    只保留最近K轮对话
    """
    
    def __init__(
        self,
        k: int = 10,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        memory_key: str = "history",
        return_messages: bool = False,
        input_key: str = "input",
        output_key: str = "output"
    ):
        """
        初始化窗口记忆
        
        Args:
            k: 保留的对话轮数
            human_prefix: 人类消息前缀
            ai_prefix: AI消息前缀
            memory_key: 记忆键名
            return_messages: 是否返回消息对象列表
            input_key: 输入键名
            output_key: 输出键名
        """
        self.k = k
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.input_key = input_key
        self.output_key = output_key
        
        # 使用 deque 自动管理窗口大小
        self.buffer: deque = deque(maxlen=k * 2)  # *2 因为每轮有两条消息
    
    def save_context(self, inputs: Dict, outputs: Dict):
        """
        保存对话上下文
        
        Args:
            inputs: 输入信息
            outputs: 输出信息
        """
        human_input = inputs.get(self.input_key, "")
        ai_output = outputs.get(self.output_key, "")
        
        # 构建消息对
        human_message = {
            "role": "human",
            "content": human_input,
            "timestamp": datetime.now().isoformat()
        }
        
        ai_message = {
            "role": "ai",
            "content": ai_output,
            "timestamp": datetime.now().isoformat()
        }
        
        # 添加到窗口（自动丢弃超出窗口的消息）
        self.buffer.append(human_message)
        self.buffer.append(ai_message)
    
    def load_memory_variables(self, inputs: Dict = None) -> Dict:
        """
        加载记忆变量
        """
        if self.return_messages:
            return {self.memory_key: list(self.buffer)}
        else:
            return {self.memory_key: self._get_string_history()}
    
    def _get_string_history(self) -> str:
        """
        获取字符串格式的历史记录
        """
        lines = []
        for message in self.buffer:
            if message["role"] == "human":
                prefix = self.human_prefix
            else:
                prefix = self.ai_prefix
            lines.append(f"{prefix}: {message['content']}")
        return "\n".join(lines)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
    
    def get_window_size(self) -> int:
        """获取当前窗口大小"""
        return len(self.buffer)
    
    def get_effective_context_length(self) -> int:
        """获取有效上下文长度（轮数）"""
        return len(self.buffer) // 2
    
    def set_window_size(self, new_k: int):
        """
        动态调整窗口大小
        
        Args:
            new_k: 新的窗口大小
        """
        old_messages = list(self.buffer)
        self.buffer = deque(old_messages, maxlen=new_k * 2)
    
    def peek_oldest(self) -> Optional[Dict]:
        """查看最旧的消息（不移除）"""
        if self.buffer:
            return self.buffer[0]
        return None
    
    def peek_newest(self) -> Optional[Dict]:
        """查看最新的消息（不移除）"""
        if self.buffer:
            return self.buffer[-1]
        return None
```

### 7.5.3 使用示例

```python
def demo_window_memory():
    """演示窗口记忆的使用"""
    
    # 创建窗口大小为5的记忆
    memory = ConversationBufferWindowMemory(
        k=5,
        human_prefix="用户",
        ai_prefix="助手",
        return_messages=True
    )
    
    # 模拟20轮对话
    for i in range(20):
        memory.save_context(
            inputs={"input": f"问题 {i+1}：这是第{i+1}个问题"},
            outputs={"output": f"回答 {i+1}：这是对第{i+1}个问题的回答"}
        )
    
    # 查看当前窗口
    print(f"当前窗口大小: {memory.get_window_size()}")
    print(f"有效上下文长度: {memory.get_effective_context_length()} 轮")
    
    # 加载记忆（只会返回最近5轮）
    memory_vars = memory.load_memory_variables()
    messages = memory_vars["history"]
    
    print(f"\n返回的消息数量: {len(messages)}")
    print("\n历史记录:")
    for msg in messages:
        prefix = "用户" if msg["role"] == "human" else "助手"
        print(f"{prefix}: {msg['content']}")
    
    return memory


if __name__ == "__main__":
    demo_window_memory()
```

### 7.5.4 滑动窗口算法

```python
from collections import deque
from typing import List, Tuple


class SlidingWindowMemory:
    """
    高级滑动窗口记忆
    支持多种窗口策略
    """
    
    def __init__(self, window_size: int = 10, strategy: str = "rounds"):
        """
        初始化滑动窗口记忆
        
        Args:
            window_size: 窗口大小
            strategy: 窗口策略
                - "rounds": 按对话轮数
                - "tokens": 按token数量
                - "time": 按时间
                - "importance": 按重要性
        """
        self.window_size = window_size
        self.strategy = strategy
        
        if strategy == "rounds":
            self.buffer = deque(maxlen=window_size * 2)
        else:
            self.buffer = []
        
        # 用于token计数
        self.token_counts = deque(maxlen=window_size * 2) if strategy == "tokens" else None
        
        # 用于时间窗口
        self.timestamps = deque(maxlen=window_size * 2) if strategy == "time" else None
    
    def add_message(self, role: str, content: str, importance: float = 0.5):
        """
        添加消息
        """
        message = {
            "role": role,
            "content": content,
            "importance": importance,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.strategy == "rounds":
            self.buffer.append(message)
        
        elif self.strategy == "tokens":
            token_count = self._estimate_tokens(content)
            self.buffer.append(message)
            self.token_counts.append(token_count)
            self._enforce_token_limit()
        
        elif self.strategy == "time":
            self.buffer.append(message)
            self.timestamps.append(datetime.now())
            self._enforce_time_limit()
        
        elif self.strategy == "importance":
            self.buffer.append(message)
            self._enforce_importance_limit()
    
    def _enforce_token_limit(self):
        """强制token限制"""
        total_tokens = sum(self.token_counts)
        
        while total_tokens > self.window_size and len(self.buffer) > 1:
            removed_tokens = self.token_counts.popleft()
            total_tokens -= removed_tokens
            self.buffer.popleft()
    
    def _enforce_time_limit(self):
        """强制时间限制"""
        if not self.timestamps:
            return
        
        now = datetime.now()
        cutoff = now.timestamp() - self.window_size  # window_size秒前
        
        while self.timestamps and self.timestamps[0].timestamp() < cutoff:
            self.timestamps.popleft()
            self.buffer.popleft()
    
    def _enforce_importance_limit(self):
        """强制重要性限制"""
        if len(self.buffer) <= self.window_size:
            return
        
        # 按重要性排序，保留最重要的
        sorted_buffer = sorted(
            enumerate(self.buffer),
            key=lambda x: x[1]["importance"],
            reverse=True
        )
        
        # 保留最重要的window_size条
        keep_indices = set([idx for idx, _ in sorted_buffer[:self.window_size]])
        
        self.buffer = [
            msg for idx, msg in enumerate(self.buffer)
            if idx in keep_indices
        ]
    
    def _estimate_tokens(self, text: str) -> int:
        """估算token数量"""
        # 简单估算
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        english_words = len(text.split())
        return chinese_chars * 2 + english_words
    
    def get_context(self) -> List[Dict]:
        """获取上下文"""
        return list(self.buffer)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "strategy": self.strategy,
            "window_size": self.window_size,
            "current_size": len(self.buffer),
            "messages": len(self.buffer)
        }
```

### 7.5.5 与其他记忆类型的比较

| 特性 | Buffer Memory | Window Memory | Token Buffer Memory |
|:---|:---|:---|:---|
| 保存数量 | 全部 | 最近 K 轮 | 最近 N tokens |
| Token 消耗 | 高 | 中等 | 可控 |
| 信息完整性 | 完整 | 部分丢失 | 部分丢失 |
| 适用场景 | 短对话 | 中等长度对话 | 长对话 |
| 实现复杂度 | 低 | 中等 | 中等 |

---

## 7.6 Summary Memory 详解

### 7.6.1 概念与原理

Summary Memory（摘要记忆）通过 LLM 对对话历史进行压缩，生成简洁的摘要来代替完整的历史记录。这类似于人类记忆中的"要点提取"过程。

**核心特点：**
- 使用 LLM 生成摘要
- 大幅减少 token 消耗
- 保留关键信息
- 需要额外的 LLM 调用

### 7.6.2 实现原理

```python
from typing import List, Dict, Optional
from datetime import datetime


class ConversationSummaryMemory:
    """
    对话摘要记忆的完整实现
    使用LLM压缩对话历史
    """
    
    def __init__(
        self,
        llm,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        memory_key: str = "history",
        return_messages: bool = False,
        input_key: str = "input",
        output_key: str = "output",
        summary_template: str = None
    ):
        """
        初始化摘要记忆
        
        Args:
            llm: 用于生成摘要的语言模型
            human_prefix: 人类消息前缀
            ai_prefix: AI消息前缀
            memory_key: 记忆键名
            return_messages: 是否返回消息对象
            input_key: 输入键名
            output_key: 输出键名
            summary_template: 摘要模板
        """
        self.llm = llm
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.input_key = input_key
        self.output_key = output_key
        
        # 当前对话缓冲
        self.buffer: List[Dict] = []
        
        # 摘要
        self.summary: str = ""
        
        # 历史对话（用于增量更新）
        self.moving_summary_buffer: str = ""
        
        # 摘要模板
        self.summary_template = summary_template or self._default_summary_template()
    
    def _default_summary_template(self) -> str:
        """默认摘要模板"""
        return """当前对话摘要:
{summary}

最新对话:
{new_lines}

请基于以上信息，更新对话摘要。摘要应该:
1. 保留关键信息和上下文
2. 简洁明了
3. 便于后续对话理解"""
    
    def save_context(self, inputs: Dict, outputs: Dict):
        """
        保存对话上下文
        """
        human_input = inputs.get(self.input_key, "")
        ai_output = outputs.get(self.output_key, "")
        
        # 添加到缓冲区
        self.buffer.append({
            "human": human_input,
            "ai": ai_output,
            "timestamp": datetime.now().isoformat()
        })
        
        # 当缓冲区达到一定大小时，更新摘要
        if len(self.buffer) >= 3:
            self._update_summary()
    
    def _update_summary(self):
        """更新摘要"""
        if not self.buffer:
            return
        
        # 构建新的对话行
        new_lines = []
        for exchange in self.buffer:
            new_lines.append(f"{self.human_prefix}: {exchange['human']}")
            new_lines.append(f"{self.ai_prefix}: {exchange['ai']}")
        
        new_lines_text = "\n".join(new_lines)
        
        # 使用LLM生成新摘要
        prompt = self.summary_template.format(
            summary=self.moving_summary_buffer or "无",
            new_lines=new_lines_text
        )
        
        # 调用LLM
        try:
            response = self.llm.invoke(prompt)
            self.moving_summary_buffer = response.content if hasattr(response, 'content') else str(response)
            self.summary = self.moving_summary_buffer
        except Exception as e:
            # 如果LLM调用失败，使用简单拼接
            if self.moving_summary_buffer:
                self.moving_summary_buffer += "\n" + new_lines_text
            else:
                self.moving_summary_buffer = new_lines_text
            self.summary = self.moving_summary_buffer
        
        # 清空缓冲区
        self.buffer.clear()
    
    def load_memory_variables(self, inputs: Dict = None) -> Dict:
        """
        加载记忆变量
        """
        if self.return_messages:
            # 返回摘要作为消息
            return {
                self.memory_key: [
                    {"type": "system", "content": self.summary}
                ]
            }
        else:
            return {self.memory_key: self.summary}
    
    def clear(self):
        """清空记忆"""
        self.buffer.clear()
        self.summary = ""
        self.moving_summary_buffer = ""
    
    def get_summary(self) -> str:
        """获取当前摘要"""
        return self.summary
    
    def force_update(self):
        """强制更新摘要（即使缓冲区未满）"""
        self._update_summary()
```

### 7.6.3 使用示例

```python
def demo_summary_memory():
    """演示摘要记忆的使用"""
    
    # 模拟LLM
    class MockLLM:
        def invoke(self, prompt):
            # 简单的摘要生成逻辑
            if "无" in prompt:
                summary = "用户正在学习Python编程。"
            else:
                summary = "用户正在学习Python编程，已经讨论了数据结构和列表的使用。"
            
            class Response:
                def __init__(self, content):
                    self.content = content
            return Response(summary)
    
    llm = MockLLM()
    
    # 创建摘要记忆
    memory = ConversationSummaryMemory(
        llm=llm,
        human_prefix="用户",
        ai_prefix="助手"
    )
    
    # 模拟对话
    conversations = [
        ("你好，我想学习Python", "你好！很高兴帮助你学习Python。"),
        ("我想学习数据结构", "好的！Python中常用的数据结构包括列表、字典等。"),
        ("能给我一个列表的例子吗？", "当然！列表是Python中最常用的数据结构之一。"),
        ("列表和元组有什么区别？", "列表是可变的，元组是不可变的。"),
    ]
    
    for human_input, ai_response in conversations:
        memory.save_context(
            inputs={"input": human_input},
            outputs={"output": ai_response}
        )
    
    # 获取摘要
    print("当前摘要:")
    print(memory.get_summary())
    
    return memory


if __name__ == "__main__":
    demo_summary_memory()
```

### 7.6.4 摘要策略

```python
from typing import List, Dict
from enum import Enum


class SummaryStrategy(Enum):
    """摘要策略"""
    FULL = "full"           # 完整摘要
    INCREMENTAL = "incremental"  # 增量更新
    MULTI_LEVEL = "multi_level"  # 多层次摘要


class AdvancedSummaryMemory:
    """
    高级摘要记忆
    支持多种摘要策略
    """
    
    def __init__(self, llm, strategy: SummaryStrategy = SummaryStrategy.INCREMENTAL):
        self.llm = llm
        self.strategy = strategy
        
        # 存储
        self.full_summary = ""
        self.incremental_summaries: List[str] = []
        self.key_points: List[str] = []
        
        # 对话缓冲
        self.buffer: List[Dict] = []
    
    def add_exchange(self, human_input: str, ai_output: str):
        """添加一轮对话"""
        exchange = {
            "human": human_input,
            "ai": ai_output,
            "timestamp": datetime.now().isoformat()
        }
        
        self.buffer.append(exchange)
        
        # 提取关键点
        key_point = self._extract_key_point(human_input, ai_output)
        if key_point:
            self.key_points.append(key_point)
        
        # 根据策略更新摘要
        if self.strategy == SummaryStrategy.FULL:
            self._update_full_summary()
        elif self.strategy == SummaryStrategy.INCREMENTAL:
            self._add_incremental_summary()
        elif self.strategy == SummaryStrategy.MULTI_LEVEL:
            self._update_multi_level()
    
    def _extract_key_point(self, human_input: str, ai_output: str) -> str:
        """提取关键点"""
        # 简化的关键点提取
        if "?" in human_input or "？" in human_input:
            return f"问题: {human_input[:50]}..."
        return None
    
    def _update_full_summary(self):
        """更新完整摘要"""
        # 构建所有对话
        all_exchanges = []
        for exchange in self.buffer:
            all_exchanges.append(f"用户: {exchange['human']}")
            all_exchanges.append(f"助手: {exchange['ai']}")
        
        prompt = f"""请为以下对话生成简洁的摘要:

{chr(10).join(all_exchanges)}

摘要:"""
        
        try:
            response = self.llm.invoke(prompt)
            self.full_summary = response.content if hasattr(response, 'content') else str(response)
        except:
            self.full_summary = "对话摘要生成失败"
    
    def _add_incremental_summary(self):
        """添加增量摘要"""
        if not self.buffer:
            return
        
        last_exchange = self.buffer[-1]
        
        prompt = f"""请为以下对话生成一句话摘要:

用户: {last_exchange['human']}
助手: {last_exchange['ai']}

一句话摘要:"""
        
        try:
            response = self.llm.invoke(prompt)
            summary = response.content if hasattr(response, 'content') else str(response)
            self.incremental_summaries.append(summary)
        except:
            self.incremental_summaries.append("摘要生成失败")
    
    def _update_multi_level(self):
        """更新多层次摘要"""
        # 更新关键点摘要
        if len(self.key_points) > 5:
            # 合并旧的关键点
            old_points = self.key_points[:5]
            new_points = self.key_points[5:]
            
            prompt = f"""请将以下关键点合并为简洁的摘要:

{chr(10).join(old_points)}

合并后的摘要:"""
            
            try:
                response = self.llm.invoke(prompt)
                merged = response.content if hasattr(response, 'content') else str(response)
                self.key_points = [merged] + new_points
            except:
                pass
    
    def get_context(self, include_buffer: bool = True) -> str:
        """
        获取上下文
        
        Args:
            include_buffer: 是否包含当前缓冲
        """
        parts = []
        
        # 添加摘要
        if self.full_summary:
            parts.append(f"摘要: {self.full_summary}")
        
        if self.incremental_summaries:
            parts.append(f"近期摘要: {'; '.join(self.incremental_summaries[-3:])}")
        
        if self.key_points:
            parts.append(f"关键点: {'; '.join(self.key_points[-5:])}")
        
        # 添加当前缓冲
        if include_buffer and self.buffer:
            parts.append("当前对话:")
            for exchange in self.buffer[-3:]:
                parts.append(f"用户: {exchange['human']}")
                parts.append(f"助手: {exchange['ai']}")
        
        return "\n".join(parts)
    
    def clear(self):
        """清空所有记忆"""
        self.full_summary = ""
        self.incremental_summaries.clear()
        self.key_points.clear()
        self.buffer.clear()
```

### 7.6.5 优缺点分析

**优点：**
- 大幅减少 token 消耗
- 保留关键信息
- 适合长对话场景

**缺点：**
- 需要额外 LLM 调用（成本和延迟）
- 可能丢失细节
- 摘要质量取决于 LLM 能力

---

## 7.7 Vector Memory 详解

### 7.7.1 概念与原理

Vector Memory（向量记忆）使用向量嵌入和相似度搜索来检索相关的历史信息。这是最强大的记忆类型之一，因为它可以基于语义相似度找到相关的历史信息，而不仅仅是最近的信息。

**核心特点：**
- 基于语义相似度检索
- 可以找到跨时间的相关信息
- 需要向量数据库支持
- 适合知识密集型应用

### 7.7.2 完整实现

```python
"""
Vector Memory 完整实现
基于向量嵌入的记忆系统
"""

import math
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class EmbeddingModel:
    """
    嵌入模型抽象类
    用于将文本转换为向量表示
    """
    
    def embed(self, text: str) -> List[float]:
        """
        将文本转换为嵌入向量
        
        Args:
            text: 输入文本
        
        Returns:
            嵌入向量
        """
        raise NotImplementedError


class SimpleEmbeddingModel(EmbeddingModel):
    """
    简单的嵌入模型（用于演示）
    实际应用中应使用 OpenAI、Sentence Transformers 等
    """
    
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
    
    def embed(self, text: str) -> List[float]:
        """
        生成伪嵌入向量
        基于文本的哈希值
        """
        # 使用哈希生成伪随机向量
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # 扩展到所需维度
        vector = []
        for i in range(self.dimension):
            byte_idx = i % len(hash_bytes)
            vector.append((hash_bytes[byte_idx] / 255.0) * 2 - 1)
        
        # 归一化
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        
        return vector


class DistanceMetric(Enum):
    """距离度量"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


@dataclass
class VectorDocument:
    """向量文档"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class VectorStore:
    """
    向量存储实现
    支持基本的向量存储和检索
    """
    
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.embedding_model = embedding_model or SimpleEmbeddingModel()
        self.documents: Dict[str, VectorDocument] = {}
        self.index: List[str] = []  # 文档ID列表
    
    def add(
        self,
        content: str,
        metadata: Dict = None,
        doc_id: str = None
    ) -> str:
        """
        添加文档到向量存储
        
        Args:
            content: 文档内容
            metadata: 元数据
            doc_id: 文档ID（可选）
        
        Returns:
            文档ID
        """
        # 生成文档ID
        if doc_id is None:
            doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # 生成嵌入向量
        embedding = self.embedding_model.embed(content)
        
        # 创建文档
        doc = VectorDocument(
            id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            created_at=datetime.now()
        )
        
        # 存储
        self.documents[doc_id] = doc
        self.index.append(doc_id)
        
        return doc_id
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        distance_metric: DistanceMetric = DistanceMetric.COSINE
    ) -> List[Dict]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回最相关的K个文档
            distance_metric: 距离度量
        
        Returns:
            搜索结果列表
        """
        # 生成查询嵌入
        query_embedding = self.embedding_model.embed(query)
        
        # 计算相似度
        results = []
        for doc_id, doc in self.documents.items():
            score = self._calculate_similarity(
                query_embedding,
                doc.embedding,
                distance_metric
            )
            
            results.append({
                "id": doc_id,
                "content": doc.content,
                "metadata": doc.metadata,
                "score": score,
                "created_at": doc.created_at.isoformat()
            })
        
        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def _calculate_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
        metric: DistanceMetric
    ) -> float:
        """
        计算两个向量之间的相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            metric: 距离度量
        
        Returns:
            相似度分数
        """
        if metric == DistanceMetric.COSINE:
            return self._cosine_similarity(vec1, vec2)
        elif metric == DistanceMetric.EUCLIDEAN:
            return self._euclidean_similarity(vec1, vec2)
        elif metric == DistanceMetric.DOT_PRODUCT:
            return self._dot_product_similarity(vec1, vec2)
        else:
            return self._cosine_similarity(vec1, vec2)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算余弦相似度
        
        cosine_similarity = (A·B) / (||A|| × ||B||)
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        # 点积
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # 范数
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _euclidean_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算欧氏距离相似度
        similarity = 1 / (1 + distance)
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
        
        return 1.0 / (1.0 + distance)
    
    def _dot_product_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算点积相似度
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        return sum(a * b for a, b in zip(vec1, vec2))
    
    def delete(self, doc_id: str) -> bool:
        """
        删除文档
        
        Args:
            doc_id: 文档ID
        
        Returns:
            是否删除成功
        """
        if doc_id in self.documents:
            del self.documents[doc_id]
            if doc_id in self.index:
                self.index.remove(doc_id)
            return True
        return False
    
    def get_all_documents(self) -> List[VectorDocument]:
        """获取所有文档"""
        return list(self.documents.values())
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_documents": len(self.documents),
            "dimension": self.embedding_model.dimension if hasattr(self.embedding_model, 'dimension') else "unknown"
        }


class VectorStoreRetrieverMemory:
    """
    向量存储检索记忆
    基于向量相似度的记忆检索系统
    """
    
    def __init__(
        self,
        vector_store: VectorStore = None,
        memory_key: str = "relevant_history",
        retrieval_query: str = None,
        top_k: int = 3
    ):
        """
        初始化向量记忆
        
        Args:
            vector_store: 向量存储
            memory_key: 记忆键名
            retrieval_query: 检索查询模板
            top_k: 返回最相关的K条记忆
        """
        self.vector_store = vector_store or VectorStore()
        self.memory_key = memory_key
        self.retrieval_query = retrieval_query
        self.top_k = top_k
        
        # 对话历史
        self.chat_history: List[Dict] = []
    
    def save_context(self, inputs: Dict, outputs: Dict):
        """
        保存对话上下文到向量存储
        """
        human_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")
        
        # 构建存储内容
        content = f"用户: {human_input}\n助手: {ai_output}"
        
        # 元数据
        metadata = {
            "human_input": human_input,
            "ai_output": ai_output,
            "timestamp": datetime.now().isoformat()
        }
        
        # 存储到向量数据库
        self.vector_store.add(
            content=content,
            metadata=metadata
        )
        
        # 同时更新对话历史
        self.chat_history.append({
            "human": human_input,
            "ai": ai_output,
            "timestamp": datetime.now().isoformat()
        })
    
    def load_memory_variables(self, inputs: Dict = None) -> Dict:
        """
        加载记忆变量
        基于当前输入检索相关历史
        """
        # 获取查询
        query = ""
        if inputs:
            query = inputs.get("input", "")
        
        if not query and self.retrieval_query:
            query = self.retrieval_query
        elif not query and self.chat_history:
            # 使用最近的输入作为查询
            query = self.chat_history[-1].get("human", "")
        
        # 检索相关记忆
        if query:
            relevant_docs = self.vector_store.search(
                query=query,
                top_k=self.top_k
            )
            
            # 格式化结果
            relevant_history = []
            for doc in relevant_docs:
                relevant_history.append(doc["content"])
            
            return {self.memory_key: "\n\n".join(relevant_history)}
        
        return {self.memory_key: ""}
    
    def clear(self):
        """清空记忆"""
        # 清空向量存储
        self.vector_store.documents.clear()
        self.vector_store.index.clear()
        
        # 清空对话历史
        self.chat_history.clear()
    
    def get_relevant_memories(self, query: str) -> List[Dict]:
        """
        获取相关记忆（用于调试）
        """
        return self.vector_store.search(query=query, top_k=self.top_k)
```

### 7.7.3 高级向量记忆实现

```python
"""
高级向量记忆实现
支持过滤、元数据搜索、混合检索等
"""

from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
import re


class AdvancedVectorMemory:
    """
    高级向量记忆
    支持过滤、元数据搜索、混合检索
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel = None,
        default_top_k: int = 5
    ):
        self.embedding_model = embedding_model or SimpleEmbeddingModel()
        self.default_top_k = default_top_k
        
        # 主存储
        self.documents: Dict[str, VectorDocument] = {}
        
        # 索引
        self.tag_index: Dict[str, Set[str]] = {}  # 标签索引
        self.time_index: Dict[str, List[str]] = {}  # 时间索引（按天）
        self.type_index: Dict[str, Set[str]] = {}  # 类型索引
    
    def add_with_metadata(
        self,
        content: str,
        tags: List[str] = None,
        memory_type: str = "general",
        importance: float = 0.5,
        custom_metadata: Dict = None
    ) -> str:
        """
        添加带元数据的文档
        
        Args:
            content: 内容
            tags: 标签
            memory_type: 记忆类型
            importance: 重要性
            custom_metadata: 自定义元数据
        
        Returns:
            文档ID
        """
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # 生成嵌入
        embedding = self.embedding_model.embed(content)
        
        # 构建元数据
        metadata = {
            "tags": tags or [],
            "type": memory_type,
            "importance": importance,
            "created_at": datetime.now().isoformat(),
            **(custom_metadata or {})
        }
        
        # 创建文档
        doc = VectorDocument(
            id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata
        )
        
        # 存储
        self.documents[doc_id] = doc
        
        # 更新索引
        self._update_indices(doc_id, metadata)
        
        return doc_id
    
    def search_with_filters(
        self,
        query: str,
        top_k: int = None,
        tags: List[str] = None,
        memory_type: str = None,
        min_importance: float = None,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> List[Dict]:
        """
        带过滤条件的搜索
        
        Args:
            query: 查询
            top_k: 返回数量
            tags: 标签过滤
            memory_type: 类型过滤
            min_importance: 最小重要性
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            搜索结果
        """
        if top_k is None:
            top_k = self.default_top_k
        
        # 生成查询嵌入
        query_embedding = self.embedding_model.embed(query)
        
        # 候选文档
        candidates = set(self.documents.keys())
        
        # 应用过滤
        if tags:
            tag_candidates = set()
            for tag in tags:
                if tag in self.tag_index:
                    tag_candidates.update(self.tag_index[tag])
            candidates &= tag_candidates
        
        if memory_type:
            if memory_type in self.type_index:
                candidates &= self.type_index[memory_type]
            else:
                candidates = set()
        
        # 计算相似度并应用过滤
        results = []
        for doc_id in candidates:
            doc = self.documents[doc_id]
            metadata = doc.metadata
            
            # 时间过滤
            if start_time or end_time:
                doc_time = datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat()))
                if start_time and doc_time < start_time:
                    continue
                if end_time and doc_time > end_time:
                    continue
            
            # 重要性过滤
            if min_importance is not None:
                if metadata.get("importance", 0) < min_importance:
                    continue
            
            # 计算相似度
            score = self._cosine_similarity(query_embedding, doc.embedding)
            
            results.append({
                "id": doc_id,
                "content": doc.content,
                "metadata": metadata,
                "score": score
            })
        
        # 排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def _update_indices(self, doc_id: str, metadata: Dict):
        """更新索引"""
        # 标签索引
        for tag in metadata.get("tags", []):
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(doc_id)
        
        # 时间索引
        created_at = metadata.get("created_at", datetime.now().isoformat())
        day = created_at[:10]  # YYYY-MM-DD
        if day not in self.time_index:
            self.time_index[day] = []
        self.time_index[day].append(doc_id)
        
        # 类型索引
        memory_type = metadata.get("type", "general")
        if memory_type not in self.type_index:
            self.type_index[memory_type] = set()
        self.type_index[memory_type].add(doc_id)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def hybrid_search(
        self,
        query: str,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
        top_k: int = 5
    ) -> List[Dict]:
        """
        混合搜索：结合关键词和语义搜索
        
        Args:
            query: 查询
            keyword_weight: 关键词权重
            semantic_weight: 语义权重
            top_k: 返回数量
        
        Returns:
            搜索结果
        """
        # 语义搜索
        query_embedding = self.embedding_model.embed(query)
        semantic_results = {}
        
        for doc_id, doc in self.documents.items():
            score = self._cosine_similarity(query_embedding, doc.embedding)
            semantic_results[doc_id] = score
        
        # 关键词搜索
        keyword_results = {}
        query_chars = set(query)
        
        for doc_id, doc in self.documents.items():
            content_chars = set(doc.content)
            intersection = query_chars & content_chars
            union = query_chars | content_chars
            score = len(intersection) / len(union) if union else 0
            keyword_results[doc_id] = score
        
        # 融合结果
        combined_results = {}
        for doc_id in self.documents.keys():
            semantic_score = semantic_results.get(doc_id, 0)
            keyword_score = keyword_results.get(doc_id, 0)
            
            combined_score = (
                semantic_weight * semantic_score +
                keyword_weight * keyword_score
            )
            
            combined_results[doc_id] = combined_score
        
        # 排序
        sorted_results = sorted(
            combined_results.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 构建结果
        results = []
        for doc_id, score in sorted_results[:top_k]:
            doc = self.documents[doc_id]
            results.append({
                "id": doc_id,
                "content": doc.content,
                "metadata": doc.metadata,
                "score": score
            })
        
        return results
    
    def get_memory_stats(self) -> Dict:
        """获取记忆统计"""
        if not self.documents:
            return {"total": 0}
        
        types = {}
        tags = {}
        
        for doc in self.documents.values():
            # 统计类型
            mem_type = doc.metadata.get("type", "general")
            types[mem_type] = types.get(mem_type, 0) + 1
            
            # 统计标签
            for tag in doc.metadata.get("tags", []):
                tags[tag] = tags.get(tag, 0) + 1
        
        return {
            "total": len(self.documents),
            "by_type": types,
            "top_tags": dict(sorted(tags.items(), key=lambda x: x[1], reverse=True)[:10])
        }
    
    def delete_by_filter(
        self,
        tags: List[str] = None,
        memory_type: str = None,
        older_than: datetime = None
    ) -> int:
        """
        按条件删除记忆
        
        Returns:
            删除的记忆数量
        """
        to_delete = []
        
        for doc_id, doc in self.documents.items():
            metadata = doc.metadata
            
            # 检查标签
            if tags:
                doc_tags = set(metadata.get("tags", []))
                if not set(tags) & doc_tags:
                    continue
            
            # 检查类型
            if memory_type and metadata.get("type") != memory_type:
                continue
            
            # 检查时间
            if older_than:
                created_at = datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat()))
                if created_at > older_than:
                    continue
            
            to_delete.append(doc_id)
        
        # 删除
        for doc_id in to_delete:
            self._delete_document(doc_id)
        
        return len(to_delete)
    
    def _delete_document(self, doc_id: str):
        """删除文档并更新索引"""
        if doc_id not in self.documents:
            return
        
        doc = self.documents[doc_id]
        metadata = doc.metadata
        
        # 从标签索引中移除
        for tag in metadata.get("tags", []):
            if tag in self.tag_index and doc_id in self.tag_index[tag]:
                self.tag_index[tag].remove(doc_id)
        
        # 从时间索引中移除
        created_at = metadata.get("created_at", "")
        day = created_at[:10] if created_at else ""
        if day in self.time_index and doc_id in self.time_index[day]:
            self.time_index[day].remove(doc_id)
        
        # 从类型索引中移除
        mem_type = metadata.get("type", "general")
        if mem_type in self.type_index and doc_id in self.type_index[mem_type]:
            self.type_index[mem_type].remove(doc_id)
        
        # 删除文档
        del self.documents[doc_id]
```

### 7.7.4 使用示例

```python
def demo_vector_memory():
    """演示向量记忆的使用"""
    
    # 创建向量记忆
    memory = AdvancedVectorMemory(default_top_k=3)
    
    # 添加记忆
    memories = [
        {
            "content": "用户喜欢使用Python进行数据分析",
            "tags": ["python", "数据分析"],
            "type": "preference",
            "importance": 0.8
        },
        {
            "content": "用户之前问过如何使用pandas库",
            "tags": ["python", "pandas"],
            "type": "history",
            "importance": 0.6
        },
        {
            "content": "用户正在学习机器学习基础知识",
            "tags": ["机器学习", "学习"],
            "type": "context",
            "importance": 0.7
        },
        {
            "content": "用户使用macOS操作系统",
            "tags": ["系统", "macOS"],
            "type": "preference",
            "importance": 0.5
        },
    ]
    
    for mem in memories:
        memory.add_with_metadata(**mem)
    
    # 搜索相关记忆
    print("搜索: Python数据分析")
    results = memory.search_with_filters(
        query="Python数据分析",
        top_k=2
    )
    for r in results:
        print(f"  - {r['content']} (分数: {r['score']:.3f})")
    
    # 带过滤的搜索
    print("\n搜索: 编程语言 (只搜索preference类型)")
    results = memory.search_with_filters(
        query="编程语言",
        memory_type="preference",
        top_k=2
    )
    for r in results:
        print(f"  - {r['content']} (分数: {r['score']:.3f})")
    
    # 混合搜索
    print("\n混合搜索: 机器学习")
    results = memory.hybrid_search(
        query="机器学习",
        top_k=2
    )
    for r in results:
        print(f"  - {r['content']} (分数: {r['score']:.3f})")
    
    # 统计信息
    print("\n统计信息:")
    print(memory.get_memory_stats())
    
     return memory


 if __name__ == "__main__":
     demo_vector_memory()
 ```

记忆检索的质量直接影响 Agent 的表现。下面的交互式工具展示了多种检索策略的比较和选择：

<div data-component="MemoryRetrievalStrategies"></div>

 ---

 ## 7.8 记忆衰减机制

### 7.8.1 衰减模型设计

```python
"""
记忆衰减机制实现
模拟人类记忆的遗忘曲线
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class DecayFunction(Enum):
    """衰减函数类型"""
    EXPONENTIAL = "exponential"  # 指数衰减
    LINEAR = "linear"            # 线性衰减
    LOGARITHMIC = "logarithmic"  # 对数衰减
    POWER = "power"              # 幂函数衰减


@dataclass
class MemoryEntry:
    """记忆条目"""
    id: str
    content: str
    created_at: datetime
    last_accessed: datetime
    access_count: int
    importance: float  # 0-1
    strength: float    # 0-1 当前强度
    tags: List[str]


class MemoryDecaySystem:
    """
    记忆衰减系统
    管理记忆的强度和遗忘
    """
    
    def __init__(
        self,
        decay_function: DecayFunction = DecayFunction.EXPONENTIAL,
        decay_rate: float = 0.1,
        min_strength: float = 0.1,
        access_boost: float = 0.2
    ):
        """
        初始化衰减系统
        
        Args:
            decay_function: 衰减函数类型
            decay_rate: 衰减速率
            min_strength: 最小强度（低于此值会被遗忘）
            access_boost: 每次访问的强度提升
        """
        self.decay_function = decay_function
        self.decay_rate = decay_rate
        self.min_strength = min_strength
        self.access_boost = access_boost
        
        # 记忆存储
        self.memories: Dict[str, MemoryEntry] = {}
    
    def add_memory(
        self,
        content: str,
        importance: float = 0.5,
        tags: List[str] = None
    ) -> str:
        """
        添加新记忆
        """
        import hashlib
        memory_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        now = datetime.now()
        
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            created_at=now,
            last_accessed=now,
            access_count=0,
            importance=importance,
            strength=1.0,  # 初始强度为1
            tags=tags or []
        )
        
        self.memories[memory_id] = entry
        return memory_id
    
    def access_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        访问记忆（增强强度）
        """
        if memory_id not in self.memories:
            return None
        
        entry = self.memories[memory_id]
        entry.last_accessed = datetime.now()
        entry.access_count += 1
        
        # 访问增强
        entry.strength = min(1.0, entry.strength + self.access_boost)
        
        return entry
    
    def calculate_strength(self, memory_id: str) -> float:
        """
        计算记忆当前强度
        """
        if memory_id not in self.memories:
            return 0.0
        
        entry = self.memories[memory_id]
        
        # 计算时间衰减
        time_elapsed = (datetime.now() - entry.created_at).total_seconds() / 3600  # 小时
        
        if self.decay_function == DecayFunction.EXPONENTIAL:
            time_decay = math.exp(-self.decay_rate * time_elapsed)
        elif self.decay_function == DecayFunction.LINEAR:
            time_decay = max(0, 1 - self.decay_rate * time_elapsed / 100)
        elif self.decay_function == DecayFunction.LOGARITHMIC:
            time_decay = 1 / (1 + self.decay_rate * math.log(1 + time_elapsed))
        elif self.decay_function == DecayFunction.POWER:
            time_decay = 1 / (1 + time_elapsed) ** self.decay_rate
        else:
            time_decay = math.exp(-self.decay_rate * time_elapsed)
        
        # 访问次数加成
        access_boost = 1 + 0.1 * min(entry.access_count, 10)
        
        # 重要性加成
        importance_boost = 0.5 + 0.5 * entry.importance
        
        # 计算最终强度
        strength = time_decay * access_boost * importance_boost
        
        return max(0, min(1, strength))
    
    def update_all_strengths(self):
        """更新所有记忆的强度"""
        for memory_id in self.memories:
            self.memories[memory_id].strength = self.calculate_strength(memory_id)
    
    def forget_weak_memories(self, threshold: float = None) -> List[str]:
        """
        遗忘弱记忆
        
        Args:
            threshold: 强度阈值
        
        Returns:
            被遗忘的记忆ID列表
        """
        if threshold is None:
            threshold = self.min_strength
        
        to_forget = []
        
        for memory_id, entry in self.memories.items():
            strength = self.calculate_strength(memory_id)
            if strength < threshold:
                to_forget.append(memory_id)
        
        # 执行遗忘
        for memory_id in to_forget:
            del self.memories[memory_id]
        
        return to_forget
    
    def get_optimal_review_time(self, memory_id: str, target_strength: float = 0.7) -> float:
        """
        计算最佳复习时间
        
        Args:
            memory_id: 记忆ID
            target_strength: 目标强度
        
        Returns:
            建议的复习间隔（小时）
        """
        if memory_id not in self.memories:
            return 0
        
        entry = self.memories[memory_id]
        
        # 反解衰减公式
        if self.decay_function == DecayFunction.EXPONENTIAL:
            # target_strength = e^(-rate * t)
            # t = -ln(target_strength) / rate
            if target_strength > 0 and target_strength < 1:
                optimal_hours = -math.log(target_strength) / self.decay_rate
                return optimal_hours
        
        return 24  # 默认24小时
    
    def get_memory_stats(self) -> Dict:
        """获取记忆统计"""
        if not self.memories:
            return {"total": 0}
        
        strengths = [self.calculate_strength(mid) for mid in self.memories]
        
        return {
            "total": len(self.memories),
            "avg_strength": sum(strengths) / len(strengths),
            "min_strength": min(strengths),
            "max_strength": max(strengths),
            "weak_memories": sum(1 for s in strengths if s < self.min_strength)
        }
```

### 7.8.2 自适应衰减

```python
class AdaptiveDecaySystem:
    """
    自适应衰减系统
    根据信息特性动态调整衰减参数
    """
    
    def __init__(self):
        # 不同类型的衰减配置
        self.decay_configs = {
            "critical": {"rate": 0.01, "min_strength": 0.5},    # 关键信息：慢衰减
            "important": {"rate": 0.05, "min_strength": 0.3},   # 重要信息
            "normal": {"rate": 0.1, "min_strength": 0.1},       # 普通信息
            "trivial": {"rate": 0.2, "min_strength": 0.05},     # 琐碎信息：快衰减
        }
        
        self.memories = {}
    
    def classify_importance(self, content: str, context: Dict = None) -> str:
        """
        自动分类信息重要性
        """
        # 基于内容的分类
        if any(word in content for word in ["重要", "关键", "必须", "紧急"]):
            return "critical"
        
        if any(word in content for word in ["注意", "记住", "学习"]):
            return "important"
        
        if any(word in content for word in ["可能", "也许", "或许"]):
            return "trivial"
        
        return "normal"
    
    def add_memory(self, content: str, context: Dict = None) -> str:
        """
        添加记忆（自动分类）
        """
        import hashlib
        memory_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        importance = self.classify_importance(content, context)
        config = self.decay_configs[importance]
        
        self.memories[memory_id] = {
            "content": content,
            "importance": importance,
            "created_at": datetime.now(),
            "strength": 1.0,
            "config": config
        }
        
        return memory_id
    
    def calculate_strength(self, memory_id: str) -> float:
        """计算记忆强度"""
        if memory_id not in self.memories:
            return 0
        
        entry = self.memories[memory_id]
        config = entry["config"]
        
        hours_elapsed = (datetime.now() - entry["created_at"]).total_seconds() / 3600
        
        # 使用对应的衰减率
        strength = math.exp(-config["rate"] * hours_elapsed)
        
        return max(0, min(1, strength))
```

---

## 7.9 LangChain Memory 实战

### 7.9.1 LangChain Memory 模块概览

LangChain 提供了丰富的 Memory 模块，用于管理对话历史和上下文。主要类型包括：

```python
"""
LangChain Memory 模块完整示例
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


# ========== 1. ConversationBufferMemory ==========
class ConversationBufferMemory:
    """
    对话缓冲记忆
    完整保存所有历史消息
    """
    
    def __init__(
        self,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        memory_key: str = "history",
        return_messages: bool = False
    ):
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.buffer: List[Dict] = []
    
    def save_context(self, inputs: Dict, outputs: Dict):
        """保存对话上下文"""
        human_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")
        
        self.buffer.append({"role": "human", "content": human_input})
        self.buffer.append({"role": "ai", "content": ai_output})
    
    def load_memory_variables(self, inputs: Dict = None) -> Dict:
        """加载记忆变量"""
        if self.return_messages:
            return {self.memory_key: self.buffer}
        else:
            history = "\n".join([
                f"{self.human_prefix}: {m['content']}" if m['role'] == 'human'
                else f"{self.ai_prefix}: {m['content']}"
                for m in self.buffer
            ])
            return {self.memory_key: history}
    
    def clear(self):
        """清空记忆"""
        self.buffer.clear()


# ========== 2. ConversationBufferWindowMemory ==========
class ConversationBufferWindowMemory:
    """
    对话窗口记忆
    只保留最近K轮对话
    """
    
    def __init__(
        self,
        k: int = 10,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        memory_key: str = "history",
        return_messages: bool = False
    ):
        self.k = k
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.buffer: List[Dict] = []
    
    def save_context(self, inputs: Dict, outputs: Dict):
        """保存对话上下文"""
        human_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")
        
        self.buffer.append({"role": "human", "content": human_input})
        self.buffer.append({"role": "ai", "content": ai_output})
        
        # 保持窗口大小
        max_messages = self.k * 2
        if len(self.buffer) > max_messages:
            self.buffer = self.buffer[-max_messages:]
    
    def load_memory_variables(self, inputs: Dict = None) -> Dict:
        """加载记忆变量"""
        if self.return_messages:
            return {self.memory_key: self.buffer}
        else:
            history = "\n".join([
                f"{self.human_prefix}: {m['content']}" if m['role'] == 'human'
                else f"{self.ai_prefix}: {m['content']}"
                for m in self.buffer
            ])
            return {self.memory_key: history}
    
    def clear(self):
        """清空记忆"""
        self.buffer.clear()


# ========== 3. ConversationSummaryMemory ==========
class ConversationSummaryMemory:
    """
    对话摘要记忆
    使用LLM压缩对话历史
    """
    
    def __init__(
        self,
        llm,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        memory_key: str = "history",
        return_messages: bool = False
    ):
        self.llm = llm
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.summary = ""
        self.buffer: List[Dict] = []
    
    def save_context(self, inputs: Dict, outputs: Dict):
        """保存对话上下文"""
        human_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")
        
        self.buffer.append({"human": human_input, "ai": ai_output})
        
        if len(self.buffer) >= 3:
            self._update_summary()
    
    def _update_summary(self):
        """更新摘要"""
        if not self.buffer:
            return
        
        # 构建对话文本
        dialogue = []
        for exchange in self.buffer:
            dialogue.append(f"{self.human_prefix}: {exchange['human']}")
            dialogue.append(f"{self.ai_prefix}: {exchange['ai']}")
        
        dialogue_text = "\n".join(dialogue)
        
        prompt = f"""请为以下对话生成简洁的摘要:

{self.summary if self.summary else '无之前的摘要'}

最新对话:
{dialogue_text}

请更新摘要:"""
        
        try:
            response = self.llm.invoke(prompt)
            self.summary = response.content if hasattr(response, 'content') else str(response)
        except:
            # 简单拼接作为后备
            self.summary += "\n" + dialogue_text[:200]
        
        self.buffer.clear()
    
    def load_memory_variables(self, inputs: Dict = None) -> Dict:
        """加载记忆变量"""
        if self.return_messages:
            return {self.memory_key: [{"type": "system", "content": self.summary}]}
        else:
            return {self.memory_key: self.summary}
    
    def clear(self):
        """清空记忆"""
        self.summary = ""
        self.buffer.clear()


# ========== 4. ConversationTokenBufferMemory ==========
class ConversationTokenBufferMemory:
    """
    Token限制记忆
    根据token数量截断历史
    """
    
    def __init__(
        self,
        max_token_limit: int = 2000,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        memory_key: str = "history",
        return_messages: bool = False
    ):
        self.max_token_limit = max_token_limit
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.buffer: List[Dict] = []
        self.token_count = 0
    
    def save_context(self, inputs: Dict, outputs: Dict):
        """保存对话上下文"""
        human_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")
        
        # 估算token数量
        human_tokens = len(human_input) // 2  # 简化估算
        ai_tokens = len(ai_output) // 2
        
        self.buffer.append({
            "role": "human",
            "content": human_input,
            "tokens": human_tokens
        })
        self.buffer.append({
            "role": "ai",
            "content": ai_output,
            "tokens": ai_tokens
        })
        
        self.token_count += human_tokens + ai_tokens
        
        # 截断超出限制
        self._truncate_to_fit()
    
    def _truncate_to_fit(self):
        """截断以适应token限制"""
        while self.token_count > self.max_token_limit and len(self.buffer) > 2:
            removed = self.buffer.pop(0)
            self.token_count -= removed.get("tokens", 0)
    
    def load_memory_variables(self, inputs: Dict = None) -> Dict:
        """加载记忆变量"""
        if self.return_messages:
            messages = [{"role": m["role"], "content": m["content"]} for m in self.buffer]
            return {self.memory_key: messages}
        else:
            history = "\n".join([
                f"{self.human_prefix}: {m['content']}" if m['role'] == 'human'
                else f"{self.ai_prefix}: {m['content']}"
                for m in self.buffer
            ])
            return {self.memory_key: history}
    
    def clear(self):
        """清空记忆"""
        self.buffer.clear()
        self.token_count = 0


# ========== 5. VectorStoreRetrieverMemory ==========
class VectorStoreRetrieverMemory:
    """
    向量存储检索记忆
    基于语义相似度检索
    """
    
    def __init__(
        self,
        vector_store,
        memory_key: str = "history",
        top_k: int = 3
    ):
        self.vector_store = vector_store
        self.memory_key = memory_key
        self.top_k = top_k
        self.chat_history: List[Dict] = []
    
    def save_context(self, inputs: Dict, outputs: Dict):
        """保存对话上下文"""
        human_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")
        
        # 存储到向量数据库
        content = f"用户: {human_input}\n助手: {ai_output}"
        self.vector_store.add(content)
        
        # 更新历史
        self.chat_history.append({
            "human": human_input,
            "ai": ai_output
        })
    
    def load_memory_variables(self, inputs: Dict = None) -> Dict:
        """加载记忆变量"""
        query = inputs.get("input", "") if inputs else ""
        
        if query:
            results = self.vector_store.search(query, top_k=self.top_k)
            history = "\n\n".join(results) if results else ""
        else:
            history = ""
        
        return {self.memory_key: history}
    
    def clear(self):
        """清空记忆"""
        self.chat_history.clear()
```

### 7.9.2 组合记忆示例

```python
class CombinedMemory:
    """
    组合记忆
    结合多种记忆类型
    """
    
    def __init__(
        self,
        primary_memory,
        secondary_memory=None,
        memory_key: str = "history"
    ):
        self.primary_memory = primary_memory
        self.secondary_memory = secondary_memory
        self.memory_key = memory_key
    
    def save_context(self, inputs: Dict, outputs: Dict):
        """保存到所有记忆"""
        self.primary_memory.save_context(inputs, outputs)
        if self.secondary_memory:
            self.secondary_memory.save_context(inputs, outputs)
    
    def load_memory_variables(self, inputs: Dict = None) -> Dict:
        """加载所有记忆变量"""
        primary_vars = self.primary_memory.load_memory_variables(inputs)
        
        if self.secondary_memory:
            secondary_vars = self.secondary_memory.load_memory_variables(inputs)
            # 合并
            for key, value in secondary_vars.items():
                if key in primary_vars:
                    if isinstance(primary_vars[key], str):
                        primary_vars[key] = primary_vars[key] + "\n" + str(value)
                    elif isinstance(primary_vars[key], list):
                        primary_vars[key] = primary_vars[key] + value
        
        return primary_vars
    
    def clear(self):
        """清空所有记忆"""
        self.primary_memory.clear()
        if self.secondary_memory:
            self.secondary_memory.clear()


class ChainWithMemory:
    """
    带记忆的对话链
    """
    
    def __init__(self, llm, memory: CombinedMemory, prompt_template: str = None):
        self.llm = llm
        self.memory = memory
        self.prompt_template = prompt_template or "{history}\n\n用户: {input}\n助手:"
    
    def run(self, user_input: str) -> str:
        """
        运行对话
        """
        # 加载记忆
        memory_vars = self.memory.load_memory_variables({"input": user_input})
        history = memory_vars.get("history", "")
        
        # 构建提示
        prompt = self.prompt_template.format(
            history=history,
            input=user_input
        )
        
        # 调用LLM
        try:
            response = self.llm.invoke(prompt)
            ai_output = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            ai_output = f"抱歉，发生了错误: {str(e)}"
        
        # 保存上下文
        self.memory.save_context(
            {"input": user_input},
            {"output": ai_output}
        )
        
        return ai_output
    
    def predict(self, **kwargs) -> str:
        """预测方法（兼容LangChain接口）"""
        return self.run(kwargs.get("input", ""))


# ========== 使用示例 ==========
def demo_langchain_memory():
    """演示LangChain Memory的使用"""
    
    # 1. 基本缓冲记忆
    print("=== 1. ConversationBufferMemory ===")
    buffer_memory = ConversationBufferMemory(
        human_prefix="用户",
        ai_prefix="助手",
        return_messages=True
    )
    
    # 模拟对话
    conversations = [
        ("你好", "你好！有什么可以帮助你的吗？"),
        ("我想学习Python", "Python是一门非常好的编程语言！"),
        ("有什么推荐的学习资源吗？", "推荐官方文档和一些在线课程。"),
    ]
    
    for human_input, ai_output in conversations:
        buffer_memory.save_context(
            {"input": human_input},
            {"output": ai_output}
        )
    
    vars = buffer_memory.load_memory_variables()
    print(f"历史记录数量: {len(vars['history'])}")
    
    # 2. 窗口记忆
    print("\n=== 2. ConversationBufferWindowMemory ===")
    window_memory = ConversationBufferWindowMemory(k=2)
    
    for i in range(5):
        window_memory.save_context(
            {"input": f"问题{i+1}"},
            {"output": f"回答{i+1}"}
        )
    
    vars = window_memory.load_memory_variables()
    print(f"保留的消息数: {len(vars['history'])}")  # 应该是4（2轮）
    
    # 3. Token限制记忆
    print("\n=== 3. ConversationTokenBufferMemory ===")
    token_memory = ConversationTokenBufferMemory(max_token_limit=50)
    
    for i in range(10):
        token_memory.save_context(
            {"input": f"这是一个比较长的问题编号{i+1}，包含更多内容"},
            {"output": f"这是一个比较长的回答编号{i+1}，包含详细解释"}
        )
    
    vars = token_memory.load_memory_variables()
    print(f"Token数: {token_memory.token_count}")
    
    print("\n所有演示完成！")
    
    return buffer_memory, window_memory, token_memory


if __name__ == "__main__":
    demo_langchain_memory()
```

### 7.9.3 记忆管理器

```python
class MemoryManager:
    """
    记忆管理器
    统一管理多种记忆类型
    """
    
    def __init__(self):
        self.memories = {}
        self.default_memory = None
    
    def register(
        self,
        name: str,
        memory,
        is_default: bool = False
    ):
        """
        注册记忆
        """
        self.memories[name] = memory
        if is_default:
            self.default_memory = memory
    
    def get(self, name: str = None):
        """
        获取记忆
        """
        if name is None:
            return self.default_memory
        return self.memories.get(name)
    
    def save_context(self, inputs: Dict, outputs: Dict, memory_name: str = None):
        """
        保存上下文到指定记忆
        """
        memory = self.get(memory_name)
        if memory:
            memory.save_context(inputs, outputs)
    
    def load_memory_variables(
        self,
        inputs: Dict = None,
        memory_name: str = None
    ) -> Dict:
        """
        加载记忆变量
        """
        memory = self.get(memory_name)
        if memory:
            return memory.load_memory_variables(inputs)
        return {}
    
    def clear(self, memory_name: str = None):
        """
        清空记忆
        """
        if memory_name:
            memory = self.get(memory_name)
            if memory:
                memory.clear()
        else:
            for memory in self.memories.values():
                memory.clear()
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        """
        stats = {}
        for name, memory in self.memories.items():
            if hasattr(memory, 'buffer'):
                stats[name] = {
                    "type": type(memory).__name__,
                    "size": len(memory.buffer)
                }
            else:
                stats[name] = {
                    "type": type(memory).__name__,
                    "size": 0
                }
        return stats


# 使用示例
def demo_memory_manager():
    """演示记忆管理器的使用"""
    
    manager = MemoryManager()
    
    # 注册不同类型的记忆
    manager.register(
        "buffer",
        ConversationBufferMemory(return_messages=True),
        is_default=True
    )
    
    manager.register(
        "window",
        ConversationBufferWindowMemory(k=5)
    )
    
    manager.register(
        "token",
        ConversationTokenBufferMemory(max_token_limit=1000)
    )
    
    # 使用默认记忆
    manager.save_context(
        {"input": "你好"},
        {"output": "你好！"}
    )
    
    # 使用指定记忆
    manager.save_context(
        {"input": "你好"},
        {"output": "你好！"},
        memory_name="window"
    )
    
    # 查看统计
    print("记忆统计:")
    for name, stats in manager.get_stats().items():
        print(f"  {name}: {stats}")
    
    return manager


if __name__ == "__main__":
    demo_memory_manager()
```

---

## 7.10 Mem0 框架详解

### 7.10.1 Mem0 概述

Mem0 是一个生产级的 AI Agent 记忆框架，提供了完整的记忆管理解决方案。它支持：

- 多用户记忆隔离
- 自动记忆提取和存储
- 语义搜索和检索
- 记忆更新和遗忘
- 支持多种存储后端

### 7.10.2 核心架构

```python
"""
Mem0 核心架构实现
生产级记忆管理系统
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import hashlib
import json


class Mem0Config:
    """Mem0配置"""
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        vector_store: str = "in_memory",
        enable_graph: bool = False
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vector_store = vector_store
        self.enable_graph = enable_graph


class MemoryEntry:
    """记忆条目"""
    
    def __init__(
        self,
        id: str,
        content: str,
        user_id: str,
        agent_id: str = None,
        metadata: Dict = None,
        created_at: datetime = None,
        updated_at: datetime = None
    ):
        self.id = id
        self.content = content
        self.user_id = user_id
        self.agent_id = agent_id
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class Mem0Memory:
    """
    Mem0 记忆系统核心实现
    """
    
    def __init__(self, config: Mem0Config = None):
        self.config = config or Mem0Config()
        
        # 存储
        self.memories: Dict[str, MemoryEntry] = {}
        
        # 索引
        self.user_index: Dict[str, List[str]] = {}  # user_id -> memory_ids
        self.agent_index: Dict[str, List[str]] = {}  # agent_id -> memory_ids
        
        # 向量存储（简化实现）
        self.vector_store = {}
    
    def add(
        self,
        content: str,
        user_id: str,
        agent_id: str = None,
        metadata: Dict = None
    ) -> Dict:
        """
        添加记忆
        
        Args:
            content: 记忆内容
            user_id: 用户ID
            agent_id: Agent ID（可选）
            metadata: 元数据
        
        Returns:
            添加的记忆信息
        """
        # 生成唯一ID
        memory_id = self._generate_id(content, user_id)
        
        # 创建记忆条目
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            user_id=user_id,
            agent_id=agent_id,
            metadata=metadata or {}
        )
        
        # 存储
        self.memories[memory_id] = entry
        
        # 更新索引
        if user_id not in self.user_index:
            self.user_index[user_id] = []
        self.user_index[user_id].append(memory_id)
        
        if agent_id:
            if agent_id not in self.agent_index:
                self.agent_index[agent_id] = []
            self.agent_index[agent_id].append(memory_id)
        
        # 生成嵌入并存储
        embedding = self._embed(content)
        self.vector_store[memory_id] = {
            "embedding": embedding,
            "content": content
        }
        
        return {
            "id": memory_id,
            "content": content,
            "user_id": user_id,
            "agent_id": agent_id,
            "metadata": metadata,
            "created_at": entry.created_at.isoformat()
        }
    
    def search(
        self,
        query: str,
        user_id: str = None,
        agent_id: str = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        搜索记忆
        
        Args:
            query: 查询内容
            user_id: 用户ID过滤
            agent_id: Agent ID过滤
            top_k: 返回数量
        
        Returns:
            搜索结果
        """
        # 生成查询嵌入
        query_embedding = self._embed(query)
        
        # 获取候选记忆ID
        candidate_ids = set(self.memories.keys())
        
        if user_id and user_id in self.user_index:
            candidate_ids &= set(self.user_index[user_id])
        
        if agent_id and agent_id in self.agent_index:
            candidate_ids &= set(self.agent_index[agent_id])
        
        # 计算相似度
        results = []
        for memory_id in candidate_ids:
            if memory_id in self.vector_store:
                stored = self.vector_store[memory_id]
                score = self._cosine_similarity(query_embedding, stored["embedding"])
                
                entry = self.memories[memory_id]
                results.append({
                    "id": memory_id,
                    "content": entry.content,
                    "score": score,
                    "user_id": entry.user_id,
                    "agent_id": entry.agent_id,
                    "metadata": entry.metadata,
                    "created_at": entry.created_at.isoformat()
                })
        
        # 排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def get(self, memory_id: str) -> Optional[Dict]:
        """获取单条记忆"""
        if memory_id not in self.memories:
            return None
        
        entry = self.memories[memory_id]
        return entry.to_dict()
    
    def update(
        self,
        memory_id: str,
        content: str = None,
        metadata: Dict = None
    ) -> Optional[Dict]:
        """
        更新记忆
        
        Args:
            memory_id: 记忆ID
            content: 新内容
            metadata: 新元数据
        
        Returns:
            更新后的记忆信息
        """
        if memory_id not in self.memories:
            return None
        
        entry = self.memories[memory_id]
        
        if content is not None:
            entry.content = content
            # 更新嵌入
            embedding = self._embed(content)
            self.vector_store[memory_id] = {
                "embedding": embedding,
                "content": content
            }
        
        if metadata is not None:
            entry.metadata.update(metadata)
        
        entry.updated_at = datetime.now()
        
        return entry.to_dict()
    
    def delete(self, memory_id: str) -> bool:
        """
        删除记忆
        """
        if memory_id not in self.memories:
            return False
        
        entry = self.memories[memory_id]
        
        # 从索引中移除
        if entry.user_id in self.user_index:
            if memory_id in self.user_index[entry.user_id]:
                self.user_index[entry.user_id].remove(memory_id)
        
        if entry.agent_id and entry.agent_id in self.agent_index:
            if memory_id in self.agent_index[entry.agent_id]:
                self.agent_index[entry.agent_id].remove(memory_id)
        
        # 删除记忆
        del self.memories[memory_id]
        
        # 删除向量
        if memory_id in self.vector_store:
            del self.vector_store[memory_id]
        
        return True
    
    def get_all(
        self,
        user_id: str = None,
        agent_id: str = None
    ) -> List[Dict]:
        """获取所有记忆"""
        results = []
        
        for memory_id, entry in self.memories.items():
            if user_id and entry.user_id != user_id:
                continue
            if agent_id and entry.agent_id != agent_id:
                continue
            
            results.append(entry.to_dict())
        
        return results
    
    def _generate_id(self, content: str, user_id: str) -> str:
        """生成唯一ID"""
        hash_input = f"{content}:{user_id}:{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _embed(self, text: str) -> List[float]:
        """生成嵌入向量（简化实现）"""
        # 实际应用中使用真正的嵌入模型
        import random
        random.seed(hash(text))
        return [random.random() for _ in range(128)]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_stats(self, user_id: str = None) -> Dict:
        """获取统计信息"""
        memories = list(self.memories.values())
        
        if user_id:
            memories = [m for m in memories if m.user_id == user_id]
        
        return {
            "total": len(memories),
            "users": len(set(m.user_id for m in memories)),
            "agents": len(set(m.agent_id for m in memories if m.agent_id))
        }
```

### 7.10.3 使用示例

```python
def demo_mem0():
    """演示 Mem0 的使用"""
    
    # 创建 Mem0 实例
    config = Mem0Config(
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini"
    )
    
    memory = Mem0Memory(config)
    
    # 1. 添加记忆
    print("=== 1. 添加记忆 ===")
    
    memories_to_add = [
        ("我喜欢用Python进行数据分析", "user_001"),
        ("我正在学习机器学习", "user_001"),
        ("我使用macOS操作系统", "user_001"),
        ("我是一名软件工程师", "user_002"),
        ("我主要使用Java", "user_002"),
    ]
    
    for content, user_id in memories_to_add:
        result = memory.add(content, user_id)
        print(f"添加: {result['content'][:30]}...")
    
    # 2. 搜索记忆
    print("\n=== 2. 搜索记忆 ===")
    
    # 搜索Python相关
    results = memory.search("Python编程", user_id="user_001")
    print(f"\n用户001的Python相关记忆:")
    for r in results:
        print(f"  - {r['content']} (分数: {r['score']:.3f})")
    
    # 3. 更新记忆
    print("\n=== 3. 更新记忆 ===")
    
    # 获取用户001的所有记忆
    user_memories = memory.get_all(user_id="user_001")
    if user_memories:
        first_memory = user_memories[0]
        print(f"更新前: {first_memory['content']}")
        
        memory.update(
            first_memory['id'],
            content="我非常喜欢用Python进行数据分析和机器学习",
            metadata={"updated": True}
        )
        
        updated = memory.get(first_memory['id'])
        print(f"更新后: {updated['content']}")
    
    # 4. 删除记忆
    print("\n=== 4. 删除记忆 ===")
    
    user_memories = memory.get_all(user_id="user_001")
    if len(user_memories) > 1:
        to_delete = user_memories[-1]
        print(f"删除: {to_delete['content']}")
        memory.delete(to_delete['id'])
    
    # 5. 统计信息
    print("\n=== 5. 统计信息 ===")
    print(memory.get_stats())
    
    return memory


if __name__ == "__main__":
    demo_mem0()
```

### 7.10.4 生产环境配置

```python
class ProductionMem0Config:
    """生产环境配置"""
    
    # 向量数据库配置
    VECTOR_DB = {
        "provider": "qdrant",  # 或 "chroma", "pinecone", "weaviate"
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "agent_memories"
        }
    }
    
    # 嵌入模型配置
    EMBEDDING = {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536
    }
    
    # LLM配置
    LLM = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.1
    }
    
    # 缓存配置
    CACHE = {
        "enabled": True,
        "ttl": 3600,  # 1小时
        "max_size": 10000
    }
    
    # 安全配置
    SECURITY = {
        "enable_encryption": True,
        "max_memory_per_user": 10000,
        "enable_audit_log": True
    }
```

### 7.10.5 Mem0 vs 其他方案对比

| 特性 | Mem0 | LangChain Memory | 自定义实现 |
|:---|:---|:---|:---|
| **多用户支持** | 原生支持 | 需要手动实现 | 需要手动实现 |
| **记忆提取** | 自动 | 手动 | 手动 |
| **向量搜索** | 内置 | 需要额外配置 | 需要手动实现 |
| **生产就绪** | 是 | 部分 | 需要大量工作 |
| **可扩展性** | 高 | 中 | 取决于实现 |
| **学习曲线** | 低 | 中 | 高 |

---

## 7.11 本章小结

### 核心知识点

| 知识点 | 核心要点 | 应用场景 |
|:---|:---|:---|
| **认知科学基础** | 人类记忆模型为 Agent 记忆设计提供理论指导 | 理解记忆系统设计原理 |
| **短期记忆** | 临时存储当前对话上下文，容量有限 | 多轮对话管理 |
| **长期记忆** | 持久存储知识和经历，支持语义检索 | 知识问答、个性化 |
| **Buffer Memory** | 完整保存所有历史消息 | 短对话场景 |
| **Window Memory** | 只保留最近 K 轮对话 | 中等长度对话 |
| **Summary Memory** | 使用 LLM 压缩对话历史 | 长对话、token优化 |
| **Vector Memory** | 基于语义相似度检索 | 知识密集型应用 |
| **记忆衰减** | 模拟遗忘曲线，自动清理弱记忆 | 长期记忆管理 |
| **LangChain Memory** | 丰富的 Memory 模块生态系统 | 快速原型开发 |
| **Mem0** | 生产级记忆框架，支持多用户 | 企业级应用 |

### 记忆类型选择指南

```
对话长度 < 10 轮
└── 使用 Buffer Memory

对话长度 10-50 轮
└── 使用 Window Memory (k=10-20)

对话长度 > 50 轮
└── 使用 Summary Memory
└── 或 Token Buffer Memory

需要长期记忆
└── 使用 Vector Memory
└── 或 Mem0 框架

生产环境
└── 使用 Mem0
└── 或自定义 + 向量数据库
```

### 最佳实践总结

1. **分层设计**：结合短期记忆和长期记忆
2. **按需检索**：使用向量记忆按语义检索
3. **自动衰减**：实现记忆衰减机制，自动清理弱记忆
4. **多用户隔离**：生产环境使用 Mem0 或类似框架
 5. **监控调优**：持续监控记忆系统性能，优化参数

Agent 规划是复杂任务解决的关键能力。下面的交互式可视化展示了 Agent 如何构建规划树来解决复杂问题：

<div data-component="AgentPlanningDemoV7"></div>

---

## 7.12 思考题

### 概念理解题

1. **记忆分类**：请解释人类记忆系统中情景记忆和语义记忆的区别，并说明在 Agent 系统中如何分别实现这两种记忆。

2. **遗忘曲线**：Ebbinghaus 遗忘曲线揭示了记忆衰减的规律。请写出遗忘曲线的数学公式，并解释各个参数的含义。

3. **工作记忆**：Baddeley 的工作记忆模型包含哪些组件？每个组件在 Agent 系统中有什么对应实现？

### 设计题

4. **记忆系统设计**：设计一个智能客服系统的记忆架构，需要支持：
   - 多轮对话上下文保持
   - 用户偏好记忆
   - 历史问题检索
   - 长期知识积累

5. **衰减策略**：设计一个自适应的记忆衰减策略，能够根据信息类型（用户偏好、临时信息、关键知识）自动调整衰减速度。

6. **混合记忆**：设计一个结合 Buffer Memory、Summary Memory 和 Vector Memory 的混合记忆系统，说明每种记忆类型的职责和协作方式。

### 实践题

7. **代码实现**：实现一个完整的 MemoryManager 类，支持以下功能：
   - 注册和管理多种记忆类型
   - 统一的记忆保存和加载接口
   - 记忆统计和监控

8. **性能优化**：针对长对话场景，设计一个记忆优化方案，目标是在保持信息完整性的同时，将 token 消耗降低 50% 以上。

### 开放性问题

9. **未来方向**：你认为 Agent 记忆系统的下一个重要发展方向是什么？请从技术、应用、伦理等角度进行分析。

10. **跨 Agent 记忆**：如果需要在多个 Agent 之间共享记忆，你会如何设计记忆共享机制？需要考虑哪些安全和隐私问题？
