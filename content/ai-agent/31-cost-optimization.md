---
title: "第31章：成本优化"
description: "深入解析 AI Agent 系统的成本优化策略：成本构成分析、智能模型路由、语义缓存、Prompt 压缩与 ROI 评估"
updated: "2025-06-15"
---


下面的交互式演示展示了成本优化策略：

<div data-component="CostOptimizationStrategies"></div>

# 第31章：成本优化

> **学习目标**：
> - 理解 AI Agent 系统的成本构成与计算公式
> - 掌握 SmartModelRouter 智能模型路由的实现
> - 理解 SemanticCache 语义缓存的工作原理与实现
> - 掌握 Prompt 压缩的核心技术
> - 能够进行 Agent 系统的 ROI 分析与优化
> - 建立完整的成本监控与告警体系

## 31.1 成本构成与计算

### 31.1.1 AI Agent 系统成本模型

```
AI Agent 系统总成本 = LLM API 成本 + 基础设施成本 + 人工成本

LLM API 成本 = 输入 Token 成本 + 输出 Token 成本 + 功能附加成本

其中：
  输入 Token 成本 = Token 数量 × 单价(输入)
  输出 Token 成本 = Token 数量 × 单价(输出)
  功能附加成本 = 工具调用 + 嵌入生成 + 函数执行
```

### 31.1.2 成本计算实现

```python
# AI Agent 成本计算与追踪系统
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from enum import Enum
import json


class ModelProvider(Enum):
    """模型提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


@dataclass
class ModelPricing:
    """模型定价信息"""
    model_name: str
    provider: ModelProvider
    input_price_per_1k: float     # 每1K输入Token的价格（美元）
    output_price_per_1k: float    # 每1K输出Token的价格（美元）
    max_context: int              # 最大上下文长度
    embedding_price_per_1k: float = 0.0  # 嵌入价格
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        embedding_tokens: int = 0
    ) -> float:
        """计算调用成本（美元）"""
        input_cost = (input_tokens / 1000) * self.input_price_per_1k
        output_cost = (output_tokens / 1000) * self.output_price_per_1k
        embedding_cost = (
            embedding_tokens / 1000
        ) * self.embedding_price_per_1k
        
        return input_cost + output_cost + embedding_cost


# 主流模型定价表（2025年数据）
MODEL_PRICING_TABLE: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(
        model_name="gpt-4o",
        provider=ModelProvider.OPENAI,
        input_price_per_1k=0.0025,
        output_price_per_1k=0.01,
        max_context=128000
    ),
    "gpt-4o-mini": ModelPricing(
        model_name="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        input_price_per_1k=0.00015,
        output_price_per_1k=0.0006,
        max_context=128000
    ),
    "gpt-3.5-turbo": ModelPricing(
        model_name="gpt-3.5-turbo",
        provider=ModelProvider.OPENAI,
        input_price_per_1k=0.0005,
        output_price_per_1k=0.0015,
        max_context=16385
    ),
    "claude-3-5-sonnet": ModelPricing(
        model_name="claude-3-5-sonnet",
        provider=ModelProvider.ANTHROPIC,
        input_price_per_1k=0.003,
        output_price_per_1k=0.015,
        max_context=200000
    ),
    "claude-3-haiku": ModelPricing(
        model_name="claude-3-haiku",
        provider=ModelProvider.ANTHROPIC,
        input_price_per_1k=0.00025,
        output_price_per_1k=0.00125,
        max_context=200000
    ),
    "gemini-2.0-flash": ModelPricing(
        model_name="gemini-2.0-flash",
        provider=ModelProvider.GOOGLE,
        input_price_per_1k=0.0001,
        output_price_per_1k=0.0004,
        max_context=1000000
    ),
    "deepseek-v3": ModelPricing(
        model_name="deepseek-v3",
        provider=ModelProvider.LOCAL,
        input_price_per_1k=0.00014,
        output_price_per_1k=0.00028,
        max_context=64000
    )
}


@dataclass
class LLMCallRecord:
    """LLM 调用记录"""
    call_id: str
    model: str
    input_tokens: int
    output_tokens: int
    tool_calls: int = 0
    latency_ms: float = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    metadata: dict = field(default_factory=dict)
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    def get_cost(self) -> float:
        """计算本次调用成本"""
        pricing = MODEL_PRICING_TABLE.get(self.model)
        if not pricing:
            return 0.0
        return pricing.calculate_cost(self.input_tokens, self.output_tokens)


class CostTracker:
    """成本追踪器 - 追踪所有 LLM 调用的成本"""
    
    def __init__(self):
        self._records: list[LLMCallRecord] = []
        self._daily_costs: dict[str, float] = {}
        self._model_costs: dict[str, float] = {}
        self._total_cost: float = 0.0
        self._budget_limit: float | None = None
    
    def set_budget_limit(self, limit: float) -> None:
        """设置预算上限（美元）"""
        self._budget_limit = limit
    
    def record_call(self, record: LLMCallRecord) -> None:
        """记录 LLM 调用"""
        self._records.append(record)
        
        cost = record.get_cost()
        self._total_cost += cost
        
        # 按日统计
        day = record.timestamp[:10]
        self._daily_costs[day] = self._daily_costs.get(day, 0) + cost
        
        # 按模型统计
        self._model_costs[record.model] = (
            self._model_costs.get(record.model, 0) + cost
        )
    
    def check_budget(self) -> tuple[bool, str]:
        """检查预算"""
        if self._budget_limit is None:
            return True, "未设置预算限制"
        
        if self._total_cost >= self._budget_limit:
            return False, (
                f"预算超限: ${self._total_cost:.4f} / "
                f"${self._budget_limit:.2f}"
            )
        
        remaining = self._budget_limit - self._total_cost
        return True, f"剩余预算: ${remaining:.4f}"
    
    def get_daily_summary(self, days: int = 7) -> dict:
        """获取每日成本摘要"""
        summary = {}
        today = datetime.now()
        
        for i in range(days):
            day = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            summary[day] = {
                "cost": round(self._daily_costs.get(day, 0), 6),
                "calls": sum(
                    1 for r in self._records
                    if r.timestamp.startswith(day)
                )
            }
        
        return summary
    
    def get_model_summary(self) -> dict:
        """获取模型成本摘要"""
        summary = {}
        
        for model, cost in self._model_costs.items():
            calls = [r for r in self._records if r.model == model]
            total_tokens = sum(r.total_tokens for r in calls)
            
            summary[model] = {
                "cost": round(cost, 6),
                "calls": len(calls),
                "avg_tokens": (
                    total_tokens // len(calls) if calls else 0
                ),
                "percentage": round(
                    cost / self._total_cost * 100, 2
                ) if self._total_cost > 0 else 0
            }
        
        return summary
    
    def get_optimization_suggestions(self) -> list[str]:
        """基于成本数据生成优化建议"""
        suggestions = []
        
        # 检查是否有过度使用高端模型
        for model, cost in self._model_costs.items():
            if cost > self._total_cost * 0.5:
                pricing = MODEL_PRICING_TABLE.get(model)
                if pricing and pricing.input_price_per_1k > 0.002:
                    suggestions.append(
                        f"模型 {model} 占总成本 {cost/self._total_cost*100:.1f}%，"
                        f"考虑对简单任务使用更经济的模型"
                    )
        
        # 检查平均 Token 使用
        avg_input = sum(
            r.input_tokens for r in self._records
        ) / len(self._records) if self._records else 0
        
        if avg_input > 4000:
            suggestions.append(
                f"平均输入 Token 数 {avg_input:.0f}，"
                f"考虑使用 Prompt 压缩或语义缓存"
            )
        
        # 检查重复调用
        unique_inputs = set()
        duplicate_count = 0
        for r in self._records:
            input_hash = f"{r.model}:{r.input_tokens}"
            if input_hash in unique_inputs:
                duplicate_count += 1
            unique_inputs.add(input_hash)
        
        if duplicate_count > len(self._records) * 0.1:
            suggestions.append(
                f"检测到 {duplicate_count} 次潜在重复调用，"
                f"建议启用语义缓存"
            )
        
        return suggestions
    
    def export_report(self) -> dict:
        """导出成本报告"""
        return {
            "total_cost": round(self._total_cost, 6),
            "budget_limit": self._budget_limit,
            "total_calls": len(self._records),
            "daily_summary": self.get_daily_summary(),
            "model_summary": self.get_model_summary(),
            "suggestions": self.get_optimization_suggestions()
        }
```

## 31.2 SmartModelRouter 智能模型路由

### 31.2.1 路由器设计

```python
# 智能模型路由器 - 根据任务复杂度选择最优模型
import asyncio
from dataclasses import dataclass
from typing import Any
from enum import Enum
import re


class TaskComplexity(Enum):
    """任务复杂度"""
    TRIVIAL = "trivial"       # 简单（如格式转换）
    SIMPLE = "simple"         # 基础（如问答）
    MODERATE = "moderate"     # 中等（如分析）
    COMPLEX = "complex"       # 复杂（如推理）
    EXPERT = "expert"         # 专家级（如代码生成）


class QualityRequirement(Enum):
    """质量要求"""
    LOW = "low"               # 低质量可接受
    MEDIUM = "medium"         # 中等质量
    HIGH = "high"             # 高质量
    CRITICAL = "critical"     # 关键质量


@dataclass
class RoutingDecision:
    """路由决策结果"""
    model: str                       # 选择的模型
    complexity: TaskComplexity        # 任务复杂度评估
    reason: str                      # 选择理由
    estimated_cost: float            # 预估成本
    fallback_models: list[str]       # 备选模型
    confidence: float                # 路由置信度


class SmartModelRouter:
    """智能模型路由器
    
    核心策略：
    1. 基于任务复杂度的模型选择
    2. 基于质量要求的模型选择
    3. 基于成本约束的模型选择
    4. 基于历史性能的动态调整
    """
    
    def __init__(self, cost_tracker: CostTracker = None):
        self.cost_tracker = cost_tracker or CostTracker()
        
        # 模型能力矩阵
        self._model_capabilities: dict[str, dict] = {
            "gpt-4o": {
                "complexity_score": 0.95,
                "quality_score": 0.95,
                "speed_score": 0.7,
                "cost_score": 0.4
            },
            "gpt-4o-mini": {
                "complexity_score": 0.7,
                "quality_score": 0.75,
                "speed_score": 0.9,
                "cost_score": 0.85
            },
            "claude-3-5-sonnet": {
                "complexity_score": 0.93,
                "quality_score": 0.96,
                "speed_score": 0.75,
                "cost_score": 0.35
            },
            "claude-3-haiku": {
                "complexity_score": 0.6,
                "quality_score": 0.7,
                "speed_score": 0.95,
                "cost_score": 0.9
            },
            "gemini-2.0-flash": {
                "complexity_score": 0.75,
                "quality_score": 0.8,
                "speed_score": 0.95,
                "cost_score": 0.95
            }
        }
        
        # 路由规则配置
        self._routing_rules: list[dict] = [
            {
                "condition": lambda q: q == TaskComplexity.TRIVIAL,
                "model": "gemini-2.0-flash",
                "reason": "简单任务使用最经济模型"
            },
            {
                "condition": lambda q: q == TaskComplexity.SIMPLE,
                "model": "gpt-4o-mini",
                "reason": "基础任务使用性价比模型"
            },
            {
                "condition": lambda q: q == TaskComplexity.MODERATE,
                "model": "gpt-4o-mini",
                "reason": "中等任务使用平衡模型"
            },
            {
                "condition": lambda q: q == TaskComplexity.COMPLEX,
                "model": "gpt-4o",
                "reason": "复杂任务使用高质量模型"
            },
            {
                "condition": lambda q: q == TaskComplexity.EXPERT,
                "model": "gpt-4o",
                "reason": "专家级任务使用最强模型"
            }
        ]
        
        # 质量覆盖规则
        self._quality_overrides: dict[QualityRequirement, str] = {
            QualityRequirement.LOW: "gemini-2.0-flash",
            QualityRequirement.MEDIUM: "gpt-4o-mini",
            QualityRequirement.HIGH: "gpt-4o",
            QualityRequirement.CRITICAL: "gpt-4o"
        }
    
    def assess_complexity(
        self,
        task_description: str,
        input_tokens: int = 0,
        has_tools: bool = False,
        requires_reasoning: bool = False
    ) -> TaskComplexity:
        """评估任务复杂度"""
        score = 0
        
        # 基于关键词评估
        complexity_keywords = {
            "analyze": 2, "analyze": 2, "reason": 3,
            "complex": 3, "multi-step": 3, "chain": 2,
            "code": 2, "debug": 2, "review": 2,
            "generate": 2, "create": 2, "write": 1,
            "translate": 1, "summarize": 1, "classify": 1,
            "extract": 1, "format": 0, "convert": 0,
            "math": 3, "proof": 4, "algorithm": 3
        }
        
        task_lower = task_description.lower()
        for keyword, weight in complexity_keywords.items():
            if keyword in task_lower:
                score += weight
        
        # 基于 Token 数量
        if input_tokens > 10000:
            score += 2
        elif input_tokens > 5000:
            score += 1
        
        # 工具调用增加复杂度
        if has_tools:
            score += 1
        
        # 推理要求增加复杂度
        if requires_reasoning:
            score += 2
        
        # 映射到复杂度等级
        if score <= 1:
            return TaskComplexity.TRIVIAL
        elif score <= 3:
            return TaskComplexity.SIMPLE
        elif score <= 5:
            return TaskComplexity.MODERATE
        elif score <= 8:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.EXPERT
    
    def route(
        self,
        task_description: str,
        quality: QualityRequirement = QualityRequirement.MEDIUM,
        max_cost: float | None = None,
        input_tokens: int = 0,
        **kwargs
    ) -> RoutingDecision:
        """执行智能路由"""
        # 评估复杂度
        complexity = self.assess_complexity(
            task_description, input_tokens, **kwargs
        )
        
        # 基于质量要求选择模型
        model = self._quality_overrides.get(quality, "gpt-4o-mini")
        
        # 基于复杂度调整
        for rule in self._routing_rules:
            if rule["condition"](complexity):
                # 如果复杂度要求更高的模型，升级
                rule_model = rule["model"]
                rule_score = self._model_capabilities.get(
                    rule_model, {}
                ).get("complexity_score", 0)
                current_score = self._model_capabilities.get(
                    model, {}
                ).get("complexity_score", 0)
                
                if rule_score > current_score:
                    model = rule_model
                    reason = rule["reason"]
                break
        
        # 成本约束检查
        pricing = MODEL_PRICING_TABLE.get(model)
        estimated_cost = 0
        if pricing:
            estimated_cost = pricing.calculate_cost(input_tokens, 500)
        
        if max_cost and estimated_cost > max_cost:
            # 寻找更便宜的模型
            model, estimated_cost = self._find_cheaper_model(
                input_tokens, max_cost, quality
            )
            reason = f"成本约束: 最大 ${max_cost}"
        
        # 生成备选模型
        fallback = self._get_fallbacks(model, quality)
        
        # 计算置信度
        confidence = self._calculate_confidence(
            model, complexity, quality
        )
        
        return RoutingDecision(
            model=model,
            complexity=complexity,
            reason=reason if 'reason' in dir() else "基于质量和复杂度的综合评估",
            estimated_cost=estimated_cost,
            fallback_models=fallback,
            confidence=confidence
        )
    
    def _find_cheaper_model(
        self,
        input_tokens: int,
        max_cost: float,
        quality: QualityRequirement
    ) -> tuple[str, float]:
        """在成本约束内寻找合适的模型"""
        candidates = []
        
        for model_name, pricing in MODEL_PRICING_TABLE.items():
            cost = pricing.calculate_cost(input_tokens, 500)
            if cost <= max_cost:
                cap = self._model_capabilities.get(model_name, {})
                quality_score = cap.get("quality_score", 0)
                candidates.append((model_name, cost, quality_score))
        
        if candidates:
            # 按质量分数排序，选择最高的
            candidates.sort(key=lambda x: -x[2])
            return candidates[0][0], candidates[0][1]
        
        # 如果没有符合成本约束的，返回最便宜的
        cheapest = min(
            MODEL_PRICING_TABLE.items(),
            key=lambda x: x[1].calculate_cost(input_tokens, 500)
        )
        return (
            cheapest[0],
            cheapest[1].calculate_cost(input_tokens, 500)
        )
    
    def _get_fallbacks(
        self, primary_model: str, quality: QualityRequirement
    ) -> list[str]:
        """获取备选模型"""
        all_models = list(MODEL_PRICING_TABLE.keys())
        fallbacks = [
            m for m in all_models
            if m != primary_model
        ]
        
        # 按能力排序
        fallbacks.sort(
            key=lambda m: self._model_capabilities.get(
                m, {}
            ).get("complexity_score", 0),
            reverse=True
        )
        
        return fallbacks[:3]
    
    def _calculate_confidence(
        self,
        model: str,
        complexity: TaskComplexity,
        quality: QualityRequirement
    ) -> float:
        """计算路由置信度"""
        cap = self._model_capabilities.get(model, {})
        
        # 复杂度匹配度
        complexity_match = 1 - abs(
            cap.get("complexity_score", 0) -
            self._complexity_to_score(complexity)
        )
        
        # 质量匹配度
        quality_match = 1 - abs(
            cap.get("quality_score", 0) -
            self._quality_to_score(quality)
        )
        
        return round((complexity_match + quality_match) / 2, 3)
    
    def _complexity_to_score(self, c: TaskComplexity) -> float:
        return {
            TaskComplexity.TRIVIAL: 0.1,
            TaskComplexity.SIMPLE: 0.3,
            TaskComplexity.MODERATE: 0.5,
            TaskComplexity.COMPLEX: 0.7,
            TaskComplexity.EXPERT: 0.9
        }.get(c, 0.5)
    
    def _quality_to_score(self, q: QualityRequirement) -> float:
        return {
            QualityRequirement.LOW: 0.3,
            QualityRequirement.MEDIUM: 0.6,
            QualityRequirement.HIGH: 0.85,
            QualityRequirement.CRITICAL: 0.95
        }.get(q, 0.6)
    
    def record_performance(
        self,
        model: str,
        latency_ms: float,
        quality_score: float,
        cost: float
    ) -> None:
        """记录模型性能数据用于动态调整"""
        # 更新能力矩阵
        if model in self._model_capabilities:
            cap = self._model_capabilities[model]
            # 指数移动平均
            alpha = 0.3
            cap["speed_score"] = (
                alpha * (1 - latency_ms / 10000) +
                (1 - alpha) * cap.get("speed_score", 0.5)
            )
```

### 31.2.2 路由策略配置

```python
# 路由策略配置系统
@dataclass
class RoutingPolicy:
    """路由策略配置"""
    name: str
    description: str
    default_model: str = "gpt-4o-mini"
    max_cost_per_request: float = 0.01  # 美元
    enable_caching: bool = True
    enable_compression: bool = True
    fallback_enabled: bool = True
    # 任务类型到模型的映射
    task_model_map: dict[str, str] = field(default_factory=dict)
    # 质量到模型的映射
    quality_model_map: dict[str, str] = field(default_factory=dict)


class RoutingPolicyManager:
    """路由策略管理器"""
    
    def __init__(self):
        self._policies: dict[str, RoutingPolicy] = {}
        self._active_policy: str | None = None
        
        # 预定义策略
        self._register_default_policies()
    
    def _register_default_policies(self) -> None:
        """注册默认策略"""
        # 经济策略 - 优先成本
        self._policies["cost_optimized"] = RoutingPolicy(
            name="cost_optimized",
            description="成本优先策略，适合非关键任务",
            default_model="gemini-2.0-flash",
            max_cost_per_request=0.001,
            task_model_map={
                "chat": "gemini-2.0-flash",
                "summary": "gemini-2.0-flash",
                "classification": "gpt-4o-mini",
                "analysis": "gpt-4o-mini",
                "code_generation": "gpt-4o-mini"
            }
        )
        
        # 质量策略 - 优先质量
        self._policies["quality_first"] = RoutingPolicy(
            name="quality_first",
            description="质量优先策略，适合关键任务",
            default_model="gpt-4o",
            max_cost_per_request=0.1,
            task_model_map={
                "chat": "gpt-4o",
                "summary": "gpt-4o",
                "classification": "gpt-4o",
                "analysis": "gpt-4o",
                "code_generation": "gpt-4o"
            }
        )
        
        # 平衡策略 - 成本与质量平衡
        self._policies["balanced"] = RoutingPolicy(
            name="balanced",
            description="平衡策略，兼顾成本与质量",
            default_model="gpt-4o-mini",
            max_cost_per_request=0.02,
            task_model_map={
                "chat": "gpt-4o-mini",
                "summary": "gemini-2.0-flash",
                "classification": "gpt-4o-mini",
                "analysis": "gpt-4o",
                "code_generation": "gpt-4o"
            }
        )
    
    def set_active_policy(self, policy_name: str) -> None:
        """设置活跃策略"""
        if policy_name not in self._policies:
            raise ValueError(f"策略 '{policy_name}' 不存在")
        self._active_policy = policy_name
    
    def get_active_policy(self) -> RoutingPolicy:
        """获取当前活跃策略"""
        if not self._active_policy:
            return self._policies["balanced"]
        return self._policies[self._active_policy]
    
    def register_policy(self, policy: RoutingPolicy) -> None:
        """注册自定义策略"""
        self._policies[policy.name] = policy
```

## 31.3 SemanticCache 语义缓存

### 31.3.1 语义缓存引擎

```python
# 语义缓存引擎 - 基于语义相似度的 LLM 响应缓存
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str                          # 原始查询的哈希
    query: str                        # 原始查询
    response: str                     # LLM 响应
    embedding: list[float] | None = None  # 查询嵌入向量
    model: str = ""                   # 使用的模型
    hit_count: int = 0                # 命中次数
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    ttl: float = 3600                 # 生存时间（秒）
    metadata: dict = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """检查是否已过期"""
        return time.time() - self.created_at > self.ttl
    
    def access(self) -> None:
        """记录访问"""
        self.hit_count += 1
        self.last_accessed = time.time()


class SemanticCache:
    """语义缓存 - 基于嵌入相似度的智能缓存
    
    核心特性：
    1. 语义相似度匹配（而非精确匹配）
    2. 可配置的相似度阈值
    3. TTL 过期机制
    4. LRU 淘汰策略
    5. 缓存统计与分析
    """
    
    def __init__(
        self,
        embedding_fn: Callable = None,
        similarity_threshold: float = 0.85,
        max_size: int = 10000,
        default_ttl: float = 3600
    ):
        self._cache: dict[str, CacheEntry] = {}
        self._embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        # 统计
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def _compute_key(self, query: str, model: str) -> str:
        """计算缓存键"""
        content = f"{model}:{query}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _compute_similarity(
        self, emb1: list[float], emb2: list[float]
    ) -> float:
        """计算余弦相似度"""
        if not emb1 or not emb2:
            return 0.0
        
        arr1 = np.array(emb1)
        arr2 = np.array(emb2)
        
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    async def get(
        self,
        query: str,
        model: str = ""
    ) -> str | None:
        """获取缓存的响应"""
        # 精确匹配
        key = self._compute_key(query, model)
        entry = self._cache.get(key)
        
        if entry and not entry.is_expired:
            entry.access()
            self._hits += 1
            return entry.response
        
        # 语义匹配
        if self._embedding_fn:
            query_embedding = await self._embedding_fn(query)
            best_match = self._find_semantic_match(
                query_embedding, model
            )
            
            if best_match:
                best_match.access()
                self._hits += 1
                return best_match.response
        
        self._misses += 1
        return None
    
    async def set(
        self,
        query: str,
        response: str,
        model: str = "",
        ttl: float | None = None
    ) -> None:
        """存储缓存条目"""
        key = self._compute_key(query, model)
        
        # 计算嵌入
        embedding = None
        if self._embedding_fn:
            embedding = await self._embedding_fn(query)
        
        entry = CacheEntry(
            key=key,
            query=query,
            response=response,
            embedding=embedding,
            model=model,
            ttl=ttl or self.default_ttl
        )
        
        self._cache[key] = entry
        
        # 检查容量，淘汰过期和低频条目
        if len(self._cache) > self.max_size:
            self._evict()
    
    def _find_semantic_match(
        self,
        query_embedding: list[float],
        model: str
    ) -> CacheEntry | None:
        """查找语义最相似的缓存条目"""
        best_match = None
        best_similarity = 0
        
        for entry in self._cache.values():
            if entry.is_expired:
                continue
            if model and entry.model != model:
                continue
            if not entry.embedding:
                continue
            
            similarity = self._compute_similarity(
                query_embedding, entry.embedding
            )
            
            if (
                similarity >= self.similarity_threshold
                and similarity > best_similarity
            ):
                best_similarity = similarity
                best_match = entry
        
        return best_match
    
    def _evict(self) -> None:
        """LRU 淘汰策略"""
        if len(self._cache) <= self.max_size:
            return
        
        # 按访问时间排序，移除最久未访问的
        sorted_entries = sorted(
            self._cache.values(),
            key=lambda e: e.last_accessed
        )
        
        to_remove = len(self._cache) - self.max_size + 100
        for entry in sorted_entries[:to_remove]:
            del self._cache[entry.key]
            self._evictions += 1
    
    def clear_expired(self) -> int:
        """清除过期条目"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> dict:
        """获取缓存统计"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        
        # 计算节省的成本
        saved_tokens = sum(
            entry.hit_count * len(entry.query.split())
            for entry in self._cache.values()
        )
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "evictions": self._evictions,
            "estimated_tokens_saved": saved_tokens,
            "estimated_cost_saved": round(
                saved_tokens * 0.00001, 4
            )
        }
    
    def get_top_queries(self, top_n: int = 10) -> list[dict]:
        """获取最常命中的查询"""
        entries = sorted(
            self._cache.values(),
            key=lambda e: e.hit_count,
            reverse=True
        )[:top_n]
        
        return [
            {
                "query": e.query[:50],
                "hit_count": e.hit_count,
                "model": e.model
            }
            for e in entries
        ]
```

### 31.3.2 语义缓存与 LLM 集成

```python
# 语义缓存与 LLM 调用的集成
class CachedLLMClient:
    """带语义缓存的 LLM 客户端"""
    
    def __init__(
        self,
        llm_client: Any,
        embedding_client: Any,
        cache_config: dict = None
    ):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        
        config = cache_config or {}
        self.cache = SemanticCache(
            embedding_fn=self._get_embedding,
            similarity_threshold=config.get("threshold", 0.85),
            max_size=config.get("max_size", 10000),
            default_ttl=config.get("ttl", 3600)
        )
        
        self.cost_tracker = CostTracker()
    
    async def _get_embedding(self, text: str) -> list[float]:
        """获取文本嵌入"""
        try:
            response = await self.embedding_client.embed(text)
            return response
        except Exception:
            return []
    
    async def chat(
        self,
        messages: list[dict],
        model: str = "gpt-4o-mini",
        use_cache: bool = True,
        **kwargs
    ) -> dict:
        """带缓存的 Chat 调用"""
        # 构造查询键
        query = messages[-1].get("content", "") if messages else ""
        
        # 检查缓存
        if use_cache:
            cached = await self.cache.get(query, model)
            if cached:
                return {
                    "content": cached,
                    "cached": True,
                    "model": model
                }
        
        # 调用 LLM
        start_time = time.time()
        response = await self.llm_client.chat(
            messages=messages, model=model, **kwargs
        )
        latency = (time.time() - start_time) * 1000
        
        # 记录成本
        self.cost_tracker.record_call(LLMCallRecord(
            call_id=hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
            model=model,
            input_tokens=response.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=response.get("usage", {}).get("completion_tokens", 0),
            latency_ms=latency
        ))
        
        # 存入缓存
        content = response.get("content", "")
        if use_cache and content:
            await self.cache.set(query, content, model)
        
        return {
            "content": content,
            "cached": False,
            "model": model,
            "usage": response.get("usage", {})
        }
```

## 31.4 Prompt 压缩

### 31.4.1 压缩算法实现

```python
# Prompt 压缩引擎
from typing import Any
import re


class PromptCompressor:
    """Prompt 压缩器 - 减少输入 Token 消耗
    
    压缩策略：
    1. 冗余内容移除
    2. 重复信息合并
    3. 长文本摘要
    4. 格式优化
    5. 上下文裁剪
    """
    
    def __init__(self, target_ratio: float = 0.5):
        """
        Args:
            target_ratio: 目标压缩比（0.5 = 压缩到原来的一半）
        """
        self.target_ratio = target_ratio
        self._compression_stats = {
            "total_compressions": 0,
            "total_tokens_saved": 0,
            "avg_compression_ratio": 0
        }
    
    def compress(
        self,
        prompt: str,
        strategy: str = "auto"
    ) -> dict[str, Any]:
        """压缩 Prompt
        
        Args:
            prompt: 原始 Prompt
            strategy: 压缩策略（auto|aggressive|conservative|format）
        
        Returns:
            压缩结果
        """
        original_tokens = self._estimate_tokens(prompt)
        
        if strategy == "auto":
            compressed = self._auto_compress(prompt)
        elif strategy == "aggressive":
            compressed = self._aggressive_compress(prompt)
        elif strategy == "conservative":
            compressed = self._conservative_compress(prompt)
        elif strategy == "format":
            compressed = self._format_compress(prompt)
        else:
            compressed = prompt
        
        compressed_tokens = self._estimate_tokens(compressed)
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1
        
        # 更新统计
        self._compression_stats["total_compressions"] += 1
        self._compression_stats["total_tokens_saved"] += (
            original_tokens - compressed_tokens
        )
        n = self._compression_stats["total_compressions"]
        self._compression_stats["avg_compression_ratio"] = (
            (self._compression_stats["avg_compression_ratio"] * (n - 1) + ratio) / n
        )
        
        return {
            "original": prompt,
            "compressed": compressed,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": round(ratio, 3),
            "tokens_saved": original_tokens - compressed_tokens
        }
    
    def _auto_compress(self, prompt: str) -> str:
        """自动压缩 - 根据内容选择策略"""
        # 检查是否主要是格式化内容
        format_ratio = self._format_content_ratio(prompt)
        if format_ratio > 0.3:
            return self._format_compress(prompt)
        
        # 检查重复度
        duplicate_ratio = self._duplicate_ratio(prompt)
        if duplicate_ratio > 0.2:
            return self._aggressive_compress(prompt)
        
        # 默认保守压缩
        return self._conservative_compress(prompt)
    
    def _conservative_compress(self, prompt: str) -> str:
        """保守压缩 - 保留所有语义信息"""
        lines = prompt.split("\n")
        compressed_lines = []
        
        prev_line = ""
        for line in lines:
            stripped = line.strip()
            
            # 跳过空行（保留最多一个连续空行）
            if not stripped:
                if prev_line:
                    compressed_lines.append("")
                continue
            
            # 去除多余空格
            stripped = re.sub(r'\s+', ' ', stripped)
            
            # 去除多余标点
            stripped = re.sub(r'\.{3,}', '...', stripped)
            stripped = re.sub(r'!{2,}', '!', stripped)
            stripped = re.sub(r'\?{2,}', '?', stripped)
            
            compressed_lines.append(stripped)
            prev_line = stripped
        
        return "\n".join(compressed_lines)
    
    def _aggressive_compress(self, prompt: str) -> str:
        """激进压缩 - 最大化压缩比"""
        # 1. 移除注释和说明
        text = re.sub(r'#.*?\n', '\n', prompt)
        text = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
        text = re.sub(r"'''.*?'''", '', text, flags=re.DOTALL)
        
        # 2. 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # 3. 合并重复的句子
        sentences = re.split(r'(?<=[.!?])\s+', text)
        unique_sentences = list(dict.fromkeys(sentences))
        text = ' '.join(unique_sentences)
        
        # 4. 移除常见填充词
        fillers = [
            r'\bplease\b', r'\bkindly\b', r'\bactually\b',
            r'\bbasically\b', r'\bessentially\b', r'\bjust\b',
            r'\breally\b', r'\bvery\b', r'\bquite\b'
        ]
        for filler in fillers:
            text = re.sub(filler, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _format_compress(self, prompt: str) -> str:
        """格式压缩 - 优化格式化内容"""
        lines = prompt.split("\n")
        compressed = []
        
        for line in lines:
            # 压缩 markdown 表格
            if '|' in line and line.strip().startswith('|'):
                # 保留表格结构但压缩空格
                cells = [c.strip() for c in line.split('|')]
                line = ' | '.join(c for c in cells if c)
                if line:
                    line = f"| {line} |"
            
            # 压缩列表
            elif re.match(r'^\s*[-*+]\s', line):
                stripped = re.sub(r'^\s*[-*+]\s+', '- ', line)
                line = stripped
            
            # 压缩标题
            elif re.match(r'^#{1,6}\s', line):
                line = re.sub(r'\s+', ' ', line).strip()
            
            if line.strip():
                compressed.append(line)
        
        return '\n'.join(compressed)
    
    def _format_content_ratio(self, text: str) -> float:
        """计算格式化内容占比"""
        format_patterns = [
            r'\|.*\|',      # 表格行
            r'^#{1,6}\s',    # Markdown 标题
            r'^[-*+]\s',     # 列表项
            r'```',          # 代码块
        ]
        
        format_lines = 0
        for line in text.split('\n'):
            for pattern in format_patterns:
                if re.match(pattern, line.strip()):
                    format_lines += 1
                    break
        
        total_lines = len(text.split('\n'))
        return format_lines / total_lines if total_lines > 0 else 0
    
    def _duplicate_ratio(self, text: str) -> float:
        """计算重复内容占比"""
        sentences = re.split(r'[.!?]\s+', text)
        if len(sentences) <= 1:
            return 0
        
        unique = set(sentences)
        return 1 - len(unique) / len(sentences)
    
    def _estimate_tokens(self, text: str) -> int:
        """估算 Token 数量"""
        # 简单估算：英文约 4 字符/token，中文约 2 字符/token
        english_chars = len(re.findall(r'[a-zA-Z0-9]', text))
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - english_chars - chinese_chars
        
        return (
            english_chars // 4 +
            chinese_chars // 2 +
            other_chars // 3
        )
    
    def get_stats(self) -> dict:
        """获取压缩统计"""
        return dict(self._compression_stats)
```

### 31.4.2 上下文窗口优化

```python
# 上下文窗口优化器
class ContextWindowOptimizer:
    """上下文窗口优化器 - 智能管理 LLM 上下文"""
    
    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.compressor = PromptCompressor()
    
    def optimize_context(
        self,
        system_prompt: str,
        conversation_history: list[dict],
        current_query: str,
        tools: list[dict] = None
    ) -> dict[str, Any]:
        """优化上下文窗口使用"""
        
        # 估算当前 Token 使用
        system_tokens = self._estimate_tokens(system_prompt)
        history_tokens = sum(
            self._estimate_tokens(m.get("content", ""))
            for m in conversation_history
        )
        query_tokens = self._estimate_tokens(current_query)
        tools_tokens = (
            self._estimate_tokens(json.dumps(tools))
            if tools else 0
        )
        
        total = system_tokens + history_tokens + query_tokens + tools_tokens
        
        if total <= self.max_tokens:
            return {
                "system": system_prompt,
                "history": conversation_history,
                "query": current_query,
                "tools": tools,
                "tokens_used": total,
                "tokens_remaining": self.max_tokens - total
            }
        
        # 需要优化
        remaining_budget = self.max_tokens - system_tokens - query_tokens - tools_tokens
        
        if remaining_budget < 0:
            # 系统提示+查询已经超限，需要压缩系统提示
            result = self.compressor.compress(system_prompt, "aggressive")
            system_prompt = result["compressed"]
            remaining_budget = self.max_tokens - self._estimate_tokens(system_prompt) - query_tokens - tools_tokens
        
        # 优化对话历史
        optimized_history = self._optimize_history(
            conversation_history, remaining_budget
        )
        
        return {
            "system": system_prompt,
            "history": optimized_history,
            "query": current_query,
            "tools": tools,
            "tokens_used": self._estimate_tokens(system_prompt) + sum(
                self._estimate_tokens(m.get("content", ""))
                for m in optimized_history
            ) + query_tokens + tools_tokens,
            "tokens_remaining": self.max_tokens - (
                self._estimate_tokens(system_prompt) + sum(
                    self._estimate_tokens(m.get("content", ""))
                    for m in optimized_history
                ) + query_tokens + tools_tokens
            ),
            "optimized": True
        }
    
    def _optimize_history(
        self,
        history: list[dict],
        budget: int
    ) -> list[dict]:
        """优化对话历史"""
        if not history:
            return []
        
        # 策略1：保留最近的消息
        optimized = []
        tokens_used = 0
        
        for msg in reversed(history):
            msg_tokens = self._estimate_tokens(msg.get("content", ""))
            if tokens_used + msg_tokens <= budget:
                optimized.insert(0, msg)
                tokens_used += msg_tokens
            else:
                break
        
        # 如果还有预算，添加早期的关键消息
        if len(optimized) < len(history) and tokens_used < budget * 0.8:
            for msg in history:
                if msg not in optimized:
                    msg_tokens = self._estimate_tokens(
                        msg.get("content", "")
                    )
                    if tokens_used + msg_tokens <= budget:
                        optimized.insert(0, msg)
                        tokens_used += msg_tokens
        
        # 按时间排序
        optimized.sort(
            key=lambda m: m.get("timestamp", ""),
            reverse=False
        )
        
        return optimized
    
    def _estimate_tokens(self, text: str) -> int:
        """估算 Token 数"""
        if not text:
            return 0
        return len(text) // 4
```

## 31.5 ROI 分析

### 31.5.1 ROI 计算框架

```python
# Agent 系统 ROI 分析框架
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime


@dataclass
class CostBreakdown:
    """成本分解"""
    llm_api_cost: float = 0.0          # LLM API 费用
    infrastructure_cost: float = 0.0   # 基础设施费用
    development_cost: float = 0.0      # 开发成本
    maintenance_cost: float = 0.0      # 维护成本
    human_cost: float = 0.0           # 人工成本
    
    @property
    def total_cost(self) -> float:
        return (
            self.llm_api_cost +
            self.infrastructure_cost +
            self.development_cost +
            self.maintenance_cost +
            self.human_cost
        )


@dataclass
class BenefitMetrics:
    """效益指标"""
    time_saved_hours: float = 0.0      # 节省的工时
    error_reduction_pct: float = 0.0   # 错误减少百分比
    throughput_increase_pct: float = 0.0  # 吞吐量提升百分比
    revenue_impact: float = 0.0       # 收入影响
    cost_savings: float = 0.0         # 成本节省
    customer_satisfaction: float = 0.0  # 客户满意度提升
    
    @property
    def total_benefit(self) -> float:
        """计算总效益（美元）"""
        hourly_rate = 50  # 假设平均工时成本
        time_benefit = self.time_saved_hours * hourly_rate
        return (
            time_benefit +
            self.cost_savings +
            self.revenue_impact
        )


class ROIAnalyzer:
    """ROI 分析器"""
    
    def __init__(self):
        self._scenarios: list[dict] = []
    
    def add_scenario(
        self,
        name: str,
        costs: CostBreakdown,
        benefits: BenefitMetrics,
        period_months: int = 12
    ) -> None:
        """添加分析场景"""
        self._scenarios.append({
            "name": name,
            "costs": costs,
            "benefits": benefits,
            "period_months": period_months,
            "analyzed_at": datetime.now().isoformat()
        })
    
    def calculate_roi(self, scenario_name: str) -> dict:
        """计算 ROI"""
        scenario = next(
            (s for s in self._scenarios if s["name"] == scenario_name),
            None
        )
        
        if not scenario:
            raise ValueError(f"场景 '{scenario_name}' 未找到")
        
        costs = scenario["costs"]
        benefits = scenario["benefits"]
        period = scenario["period_months"]
        
        total_cost = costs.total_cost
        total_benefit = benefits.total_benefit
        
        # ROI = (收益 - 成本) / 成本 × 100%
        roi = (
            (total_benefit - total_cost) / total_cost * 100
            if total_cost > 0 else 0
        )
        
        # 回本周期（月）
        monthly_benefit = total_benefit / period if period > 0 else 0
        monthly_cost = total_cost / period if period > 0 else 0
        payback_months = (
            total_cost / monthly_benefit
            if monthly_benefit > 0 else float('inf')
        )
        
        # 月度 ROI
        monthly_roi = (
            (monthly_benefit - monthly_cost) / monthly_cost * 100
            if monthly_cost > 0 else 0
        )
        
        return {
            "scenario": scenario_name,
            "period_months": period,
            "total_cost": round(total_cost, 2),
            "total_benefit": round(total_benefit, 2),
            "net_benefit": round(total_benefit - total_cost, 2),
            "roi_percentage": round(roi, 2),
            "payback_months": round(payback_months, 1),
            "monthly_cost": round(monthly_cost, 2),
            "monthly_benefit": round(monthly_benefit, 2),
            "monthly_roi": round(monthly_roi, 2),
            "cost_breakdown": {
                "llm_api": costs.llm_api_cost,
                "infrastructure": costs.infrastructure_cost,
                "development": costs.development_cost,
                "maintenance": costs.maintenance_cost,
                "human": costs.human_cost
            }
        }
    
    def compare_scenarios(self) -> list[dict]:
        """比较所有场景"""
        results = []
        for scenario in self._scenarios:
            result = self.calculate_roi(scenario["name"])
            results.append(result)
        
        # 按 ROI 排序
        results.sort(key=lambda x: x["roi_percentage"], reverse=True)
        return results
    
    def generate_report(self) -> str:
        """生成 ROI 分析报告"""
        results = self.compare_scenarios()
        
        lines = [
            "=" * 60,
            "AI Agent 系统 ROI 分析报告",
            "=" * 60,
            ""
        ]
        
        for r in results:
            lines.extend([
                f"场景: {r['scenario']}",
                f"  周期: {r['period_months']} 个月",
                f"  总成本: ${r['total_cost']:,.2f}",
                f"  总收益: ${r['total_benefit']:,.2f}",
                f"  净收益: ${r['net_benefit']:,.2f}",
                f"  ROI: {r['roi_percentage']:.1f}%",
                f"  回本周期: {r['payback_months']:.1f} 个月",
                ""
            ])
        
        # 成本构成分析
        lines.extend([
            "-" * 60,
            "成本构成分析",
            "-" * 60
        ])
        
        if results:
            best = results[0]
            breakdown = best["cost_breakdown"]
            total = best["total_cost"]
            
            for category, amount in breakdown.items():
                pct = amount / total * 100 if total > 0 else 0
                lines.append(
                    f"  {category}: ${amount:,.2f} ({pct:.1f}%)"
                )
        
        return "\n".join(lines)


# 使用示例
def demo_roi_analysis():
    """演示 ROI 分析"""
    analyzer = ROIAnalyzer()
    
    # 场景1：智能客服
    analyzer.add_scenario(
        name="智能客服 Agent",
        costs=CostBreakdown(
            llm_api_cost=12000,        # 年 API 费用
            infrastructure_cost=5000,   # 服务器费用
            development_cost=30000,     # 开发成本
            maintenance_cost=6000,      # 维护成本
            human_cost=0
        ),
        benefits=BenefitMetrics(
            time_saved_hours=2000,      # 年节省工时
            error_reduction_pct=30,
            throughput_increase_pct=50,
            revenue_impact=50000,
            cost_savings=30000
        ),
        period_months=12
    )
    
    # 场景2：代码审查
    analyzer.add_scenario(
        name="代码审查 Agent",
        costs=CostBreakdown(
            llm_api_cost=8000,
            infrastructure_cost=3000,
            development_cost=20000,
            maintenance_cost=4000,
            human_cost=0
        ),
        benefits=BenefitMetrics(
            time_saved_hours=1500,
            error_reduction_pct=40,
            throughput_increase_pct=60,
            revenue_impact=0,
            cost_savings=40000
        ),
        period_months=12
    )
    
    # 生成报告
    report = analyzer.generate_report()
    print(report)
    
    # 比较场景
    comparison = analyzer.compare_scenarios()
    print(f"\n最优方案: {comparison[0]['scenario']}")
    print(f"ROI: {comparison[0]['roi_percentage']:.1f}%")
```

## 31.6 成本监控与告警

### 31.6.1 成本监控系统

```python
# 成本监控与告警系统
from enum import Enum
import asyncio


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class CostMonitor:
    """成本监控器"""
    
    def __init__(self, cost_tracker: CostTracker):
        self.tracker = cost_tracker
        self._alerts: list[dict] = []
        self._thresholds: dict[str, dict] = {}
    
    def set_threshold(
        self,
        metric: str,
        warning: float,
        critical: float,
        period: str = "daily"
    ) -> None:
        """设置告警阈值"""
        self._thresholds[metric] = {
            "warning": warning,
            "critical": critical,
            "period": period
        }
    
    def check_thresholds(self) -> list[dict]:
        """检查所有阈值"""
        alerts = []
        
        for metric, config in self._thresholds.items():
            value = self._get_metric_value(metric, config["period"])
            
            if value >= config["critical"]:
                alerts.append({
                    "level": AlertLevel.CRITICAL,
                    "metric": metric,
                    "value": value,
                    "threshold": config["critical"],
                    "message": f"严重: {metric} 超过临界值"
                })
            elif value >= config["warning"]:
                alerts.append({
                    "level": AlertLevel.WARNING,
                    "metric": metric,
                    "value": value,
                    "threshold": config["warning"],
                    "message": f"警告: {metric} 超过预警值"
                })
        
        self._alerts.extend(alerts)
        return alerts
    
    def _get_metric_value(
        self, metric: str, period: str
    ) -> float:
        """获取指标当前值"""
        if metric == "daily_cost":
            today = datetime.now().strftime("%Y-%m-%d")
            return self.tracker._daily_costs.get(today, 0)
        elif metric == "total_cost":
            return self.tracker._total_cost
        elif metric == "call_count":
            return len(self.tracker._records)
        return 0
    
    def get_alert_history(self, limit: int = 50) -> list[dict]:
        """获取告警历史"""
        return self._alerts[-limit:]
    
    def generate_cost_report(self) -> str:
        """生成成本报告"""
        report = self.tracker.export_report()
        
        lines = [
            "=" * 50,
            "AI Agent 成本监控报告",
            "=" * 50,
            f"总成本: ${report['total_cost']:.4f}",
            f"总调用次数: {report['total_calls']}",
            ""
        ]
        
        # 模型成本分布
        lines.append("模型成本分布:")
        for model, stats in report['model_summary'].items():
            lines.append(
                f"  {model}: ${stats['cost']:.4f} "
                f"({stats['percentage']}%) "
                f"[{stats['calls']} 次调用]"
            )
        
        # 优化建议
        if report['suggestions']:
            lines.extend(["", "优化建议:"])
            for i, suggestion in enumerate(report['suggestions'], 1):
                lines.append(f"  {i}. {suggestion}")
        
        # 告警
        alerts = self.check_thresholds()
        if alerts:
            lines.extend(["", "当前告警:"])
            for alert in alerts:
                lines.append(
                    f"  [{alert['level'].value}] {alert['message']}"
                )
        
        return "\n".join(lines)
```

### 31.6.2 自动化成本优化

```python
# 自动化成本优化器
class AutoCostOptimizer:
    """自动化成本优化器"""
    
    def __init__(
        self,
        router: SmartModelRouter,
        cache: SemanticCache,
        compressor: PromptCompressor,
        cost_tracker: CostTracker
    ):
        self.router = router
        self.cache = cache
        self.compressor = compressor
        self.tracker = cost_tracker
    
    async def optimize_request(
        self,
        messages: list[dict],
        model: str = "gpt-4o-mini",
        **kwargs
    ) -> dict:
        """优化 LLM 请求"""
        optimization_log = []
        
        # 1. 检查缓存
        query = messages[-1].get("content", "")
        cached = await self.cache.get(query, model)
        if cached:
            optimization_log.append("缓存命中，跳过 LLM 调用")
            return {
                "response": cached,
                "from_cache": True,
                "optimization_log": optimization_log
            }
        
        # 2. 评估任务复杂度并路由
        routing = self.router.route(
            query,
            input_tokens=sum(
                len(m.get("content", "")) // 4 for m in messages
            )
        )
        
        if routing.model != model:
            optimization_log.append(
                f"模型路由: {model} -> {routing.model} "
                f"({routing.reason})"
            )
            model = routing.model
        
        # 3. 压缩 Prompt
        system_msg = messages[0] if messages else {}
        if system_msg.get("role") == "system":
            result = self.compressor.compress(
                system_msg["content"], "conservative"
            )
            if result["tokens_saved"] > 0:
                optimization_log.append(
                    f"Prompt 压缩: 节省 {result['tokens_saved']} tokens"
                )
        
        # 4. 调用 LLM
        # (实际调用代码省略)
        
        return {
            "model": model,
            "from_cache": False,
            "optimization_log": optimization_log,
            "routing_decision": routing
        }
    
    def get_optimization_summary(self) -> dict:
        """获取优化摘要"""
        cache_stats = self.cache.get_stats()
        compression_stats = self.compressor.get_stats()
        
        return {
            "cache": cache_stats,
            "compression": compression_stats,
            "total_savings": {
                "tokens": (
                    cache_stats.get("estimated_tokens_saved", 0) +
                    compression_stats.get("total_tokens_saved", 0)
                ),
                "cost": cache_stats.get("estimated_cost_saved", 0)
            }
        }
```

## 31.7 本章小结

本章系统性地介绍了 AI Agent 系统的成本优化策略：

1. **成本构成**：LLM API 成本是主要支出，包括输入/输出 Token 费用、嵌入费用等。

2. **SmartModelRouter**：根据任务复杂度和质量要求智能选择最优模型，平衡成本与质量。

3. **SemanticCache**：基于语义相似度的缓存机制，避免重复调用 LLM，显著降低成本。

4. **Prompt 压缩**：通过冗余移除、格式优化等策略减少输入 Token 消耗。

5. **ROI 分析**：量化评估 Agent 系统的投资回报率，指导资源分配决策。

## 31.8 思考题

1. 设计一个动态定价策略，根据用户请求的紧急程度和复杂度自动调整模型选择。

2. 语义缓存的相似度阈值如何设置？阈值过高和过低各有什么问题？

3. 在多租户系统中，如何实现租户级别的成本隔离和预算控制？

4. 设计一个 A/B 测试框架，用于评估不同模型路由策略的成本效益。

5. Prompt 压缩可能会影响 LLM 的输出质量，如何在压缩率和质量之间找到平衡点？

6. 讨论 LLM API 价格变动对成本优化策略的影响，如何设计自适应的优化系统？

7. 设计一个分布式成本追踪系统，支持多节点、多模型的实时成本监控。

8. 在边缘计算场景下，如何平衡本地模型推理成本与云端 API 调用成本？
