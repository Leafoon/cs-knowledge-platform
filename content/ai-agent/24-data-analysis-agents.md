---
title: "第24章：数据分析 Agent"
description: "深入解析数据分析 Agent 的架构设计、数据处理流程、可视化生成及报告撰写"
updated: "2026-06-15"
---

# 第24章：数据分析 Agent

 > **学习目标**：
 > - 掌握数据分析 Agent 的核心架构和工作流程
 > - 理解数据加载、探索、分析、可视化的完整流程
 > - 掌握 Data Interpreter 模式的设计与实现
 > - 了解自动化报告撰写的技术方案
 > - 能够设计和实现一个完整的数据分析 Agent

 下面的交互式演示展示了数据分析 Agent 的完整流程：

 <div data-component="DataAnalysisPipeline"></div>

 ## 24.1 数据分析 Agent 概述

### 24.1.1 什么是数据分析 Agent

数据分析 Agent 是一种能够自动执行数据分析任务的 AI 系统。它能够理解用户的分析需求，自动加载数据、进行探索性分析、生成可视化图表，并撰写分析报告。

**数据分析 Agent 的核心能力**：

| 能力 | 说明 | 示例 |
|------|------|------|
| 数据理解 | 理解数据结构和含义 | 识别字段类型、数据质量 |
| 数据清洗 | 处理缺失值、异常值 | 填充缺失值、删除异常行 |
| 探索分析 | 统计分析、趋势发现 | 描述统计、相关性分析 |
| 可视化 | 生成图表和仪表板 | 柱状图、散点图、热力图 |
| 洞察发现 | 识别数据中的模式和趋势 | 聚类分析、异常检测 |
| 报告撰写 | 生成分析报告 | 自动化报告、交互式仪表板 |

> **关键价值**：数据分析 Agent 能够将数据分析师从繁琐的数据处理工作中解放出来，让他们专注于更高价值的洞察发现和决策支持。

### 24.1.2 数据分析 Agent 的应用场景

```
┌─────────────────────────────────────────────────────────────┐
│                数据分析 Agent 应用场景                       │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 业务分析    │  │ 财务分析    │  │ 市场分析    │        │
│  │ Business    │  │ Finance     │  │ Marketing   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 运营分析    │  │ 用户分析    │  │ 风险分析    │        │
│  │ Operations  │  │ User        │  │ Risk        │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 销售分析    │  │ 产品分析    │  │ 竞品分析    │        │
│  │ Sales       │  │ Product     │  │ Competitor  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 24.1.3 数据分析 Agent 的发展历程

| 阶段 | 时间 | 技术 | 特点 |
|------|------|------|------|
| SQL 查询 | 2015-2018 | 自然语言→SQL | 简单查询 |
| 笔记本集成 | 2018-2022 | Jupyter + AI | 代码生成 |
| 自动分析 | 2022-2024 | LLM + Pandas | 全流程自动化 |
| 智能洞察 | 2024-2026 | 多模态分析 | 深度洞察 |

## 24.2 核心架构设计

### 24.2.1 DataAnalysisAgent 架构

```
┌─────────────────────────────────────────────────────────────┐
│              DataAnalysisAgent 核心架构                      │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    用户接口层                           │  │
│  │              (自然语言分析需求)                         │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   任务理解层                           │  │
│  │          (需求分析、数据源识别)                        │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   数据处理层                           │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │  │
│  │  │ 数据    │  │ 数据    │  │ 数据    │  │ 数据    │ │  │
│  │  │ 加载    │  │ 清洗    │  │ 转换    │  │ 存储    │ │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   分析引擎层                           │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │  │
│  │  │ 统计    │  │ 机器    │  │ 深度    │  │ 自定义  │ │  │
│  │  │ 分析    │  │ 学习    │  │ 学习    │  │ 分析    │ │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   可视化层                             │  │
│  │          (图表生成、仪表板构建)                        │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   报告层                              │  │
│  │          (洞察总结、报告撰写)                          │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 24.2.2 任务理解器

```python
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import re

class AnalysisType(Enum):
    """分析类型"""
    DESCRIPTIVE = "descriptive"        # 描述性统计
    EXPLORATORY = "exploratory"        # 探索性分析
    DIAGNOSTIC = "diagnostic"          # 诊断性分析
    PREDICTIVE = "predictive"          # 预测性分析
    PRESCRIPTIVE = "prescriptive"      # 处方性分析
    COMPARISON = "comparison"          # 对比分析
    TREND = "trend"                    # 趋势分析
    DISTRIBUTION = "distribution"      # 分布分析
    CORRELATION = "correlation"        # 相关性分析
    CLUSTER = "cluster"                # 聚类分析
    ANOMALY = "anomaly"                # 异常检测

@dataclass
class AnalysisRequest:
    """分析请求"""
    original_text: str
    analysis_type: AnalysisType = AnalysisType.DESCRIPTIVE
    data_source: str = ""
    target_columns: list[str] = field(default_factory=list)
    filters: dict = field(default_factory=dict)
    group_by: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    time_range: tuple = (None, None)
    visualization_type: str = ""
    output_format: str = "report"  # report, dashboard, notebook

class TaskUnderstanding:
    """任务理解器"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.analysis_patterns = self._init_analysis_patterns()
    
    def _init_analysis_patterns(self) -> dict[AnalysisType, list[str]]:
        """初始化分析模式"""
        return {
            AnalysisType.DESCRIPTIVE: [
                r"统计|描述|summary|describe|概况|概览",
                r"均值|平均|中位数|标准差|最大值|最小值",
            ],
            AnalysisType.EXPLORATORY: [
                r"探索|探索性|exploratory|EDA",
                r"发现|洞察|insight|patterns",
            ],
            AnalysisType.DIAGNOSTIC: [
                r"诊断|为什么|原因|cause|reason",
                r"分析原因|根因|root cause",
            ],
            AnalysisType.PREDICTIVE: [
                r"预测|forecast|predict|未来",
                r"趋势|trend|走向",
            ],
            AnalysisType.COMPARISON: [
                r"对比|比较|compare|差异|difference",
                r"哪个更好|排名|rank",
            ],
            AnalysisType.TREND: [
                r"趋势|trend|变化|change|增长|growth",
                r"时间序列|time series|演变",
            ],
            AnalysisType.DISTRIBUTION: [
                r"分布|distribution|频率|frequency",
                r"直方图|histogram|箱线图|box",
            ],
            AnalysisType.CORRELATION: [
                r"相关|correlation|关系|relationship",
                r"影响|influence|因果",
            ],
            AnalysisType.CLUSTER: [
                r"聚类|cluster|分组|group|分类|classify",
                r"相似|similar|segments",
            ],
            AnalysisType.ANOMALY: [
                r"异常|anomaly|outlier|离群",
                r"异常值|不正常|异常检测",
            ],
        }
    
    async def understand(self, request_text: str) -> AnalysisRequest:
        """理解分析请求"""
        if self.llm_client:
            return await self._understand_with_llm(request_text)
        
        return self._understand_with_patterns(request_text)
    
    def _understand_with_patterns(self, text: str) -> AnalysisRequest:
        """使用模式匹配理解请求"""
        # 识别分析类型
        analysis_type = self._identify_analysis_type(text)
        
        # 提取数据源
        data_source = self._extract_data_source(text)
        
        # 提取目标列
        target_columns = self._extract_columns(text)
        
        # 提取时间范围
        time_range = self._extract_time_range(text)
        
        # 提取可视化类型
        viz_type = self._extract_visualization_type(text)
        
        return AnalysisRequest(
            original_text=text,
            analysis_type=analysis_type,
            data_source=data_source,
            target_columns=target_columns,
            time_range=time_range,
            visualization_type=viz_type
        )
    
    def _identify_analysis_type(self, text: str) -> AnalysisType:
        """识别分析类型"""
        text_lower = text.lower()
        
        scores = {}
        for analysis_type, patterns in self.analysis_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            scores[analysis_type] = score
        
        if max(scores.values()) == 0:
            return AnalysisType.DESCRIPTIVE
        
        return max(scores, key=scores.get)
    
    def _extract_data_source(self, text: str) -> str:
        """提取数据源"""
        # 尝试匹配文件路径
        file_match = re.search(r'[/\\]?[\w\-./]+\.(?:csv|xlsx|json|parquet|sql)', text)
        if file_match:
            return file_match.group(0)
        
        # 尝试匹配数据库
        db_match = re.search(r'(?:数据库|database|表|table)\s*[:：]?\s*(\w+)', text)
        if db_match:
            return db_match.group(1)
        
        return ""
    
    def _extract_columns(self, text: str) -> list[str]:
        """提取目标列"""
        columns = []
        
        # 匹配引号内的列名
        quoted = re.findall(r'[""\'](.*?)[""\']|[""](.*?)[""]', text)
        for match in quoted:
            col = match[0] or match[1]
            if col and len(col) < 50:
                columns.append(col)
        
        # 匹配常见的列名模式
        col_patterns = [
            r'(?:列|字段|column)\s*[:：]?\s*(\w+)',
            r'(?:分析|统计)\s*(\w+)',
        ]
        
        for pattern in col_patterns:
            matches = re.findall(pattern, text)
            columns.extend(matches)
        
        return list(set(columns))
    
    def _extract_time_range(self, text: str) -> tuple:
        """提取时间范围"""
        # 匹配日期范围
        date_range = re.findall(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', text)
        if len(date_range) >= 2:
            return (date_range[0], date_range[-1])
        
        # 匹配相对时间
        if "最近" in text or "last" in text.lower():
            return ("last_30_days", None)
        
        return (None, None)
    
    def _extract_visualization_type(self, text: str) -> str:
        """提取可视化类型"""
        viz_patterns = {
            "bar": r"柱状图|条形图|bar|column",
            "line": r"折线图|趋势图|line|trend",
            "pie": r"饼图|pie|占比|比例",
            "scatter": r"散点图|scatter|相关",
            "histogram": r"直方图|histogram|分布",
            "heatmap": r"热力图|heatmap|相关矩阵",
            "box": r"箱线图|box|盒须图",
            "treemap": r"树图|treemap|层次",
        }
        
        text_lower = text.lower()
        for viz_type, pattern in viz_patterns.items():
            if re.search(pattern, text_lower):
                return viz_type
        
        return ""
    
    async def _understand_with_llm(self, text: str) -> AnalysisRequest:
        """使用 LLM 理解请求"""
        # 在实际实现中，这里会调用 LLM API
        return self._understand_with_patterns(text)

# 使用示例
understanding = TaskUnderstanding()
request = understanding._understand_with_patterns(
    "分析 sales.csv 中最近30天的销售趋势，绘制折线图"
)
print(f"分析类型: {request.analysis_type.value}")
print(f"数据源: {request.data_source}")
print(f"可视化: {request.visualization_type}")
```

## 24.3 数据加载与处理

### 24.3.1 数据加载器

```python
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import pandas as pd
from io import StringIO

@dataclass
class DataSource:
    """数据源"""
    source_type: str  # file, database, api, url
    path: str
    format: str = ""
    encoding: str = "utf-8"
    options: dict = field(default_factory=dict)

class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        self.loaders = {
            "csv": self._load_csv,
            "xlsx": self._load_excel,
            "json": self._load_json,
            "parquet": self._load_parquet,
            "sql": self._load_sql,
            "api": self._load_api,
        }
    
    async def load(self, source: DataSource) -> pd.DataFrame:
        """加载数据"""
        # 检测格式
        if not source.format:
            source.format = self._detect_format(source.path)
        
        # 获取加载器
        loader = self.loaders.get(source.format)
        if not loader:
            raise ValueError(f"不支持的数据格式: {source.format}")
        
        # 加载数据
        return await loader(source)
    
    def _detect_format(self, path: str) -> str:
        """检测数据格式"""
        ext = Path(path).suffix.lower()
        format_map = {
            ".csv": "csv",
            ".xlsx": "xlsx",
            ".xls": "xlsx",
            ".json": "json",
            ".jsonl": "json",
            ".parquet": "parquet",
            ".pq": "parquet",
            ".sql": "sql",
        }
        return format_map.get(ext, "csv")
    
    async def _load_csv(self, source: DataSource) -> pd.DataFrame:
        """加载 CSV"""
        try:
            return pd.read_csv(
                source.path,
                encoding=source.encoding,
                **source.options
            )
        except UnicodeDecodeError:
            # 尝试其他编码
            for encoding in ["gbk", "latin1", "utf-8-sig"]:
                try:
                    return pd.read_csv(
                        source.path,
                        encoding=encoding,
                        **source.options
                    )
                except:
                    continue
            raise
    
    async def _load_excel(self, source: DataSource) -> pd.DataFrame:
        """加载 Excel"""
        return pd.read_excel(
            source.path,
            **source.options
        )
    
    async def _load_json(self, source: DataSource) -> pd.DataFrame:
        """加载 JSON"""
        return pd.read_json(
            source.path,
            encoding=source.encoding,
            **source.options
        )
    
    async def _load_parquet(self, source: DataSource) -> pd.DataFrame:
        """加载 Parquet"""
        return pd.read_parquet(
            source.path,
            **source.options
        )
    
    async def _load_sql(self, source: DataSource) -> pd.DataFrame:
        """加载 SQL"""
        import sqlite3
        
        conn = sqlite3.connect(source.path)
        query = source.options.get("query", "SELECT * FROM data")
        
        try:
            return pd.read_sql(query, conn)
        finally:
            conn.close()
    
    async def _load_api(self, source: DataSource) -> pd.DataFrame:
        """加载 API"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(source.path) as response:
                data = await response.json()
                return pd.DataFrame(data)

# 使用示例
loader = DataLoader()

async def load_example():
    # 加载 CSV
    source = DataSource(source_type="file", path="sales.csv")
    df = await loader.load(source)
    print(f"加载了 {len(df)} 行数据")
    
    # 加载 Excel
    source = DataSource(source_type="file", path="report.xlsx", options={"sheet_name": "Sheet1"})
    df = await loader.load(source)
    print(f"加载了 {len(df)} 行数据")

# asyncio.run(load_example())
```

### 24.3.2 数据清洗器

```python
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd
import numpy as np

@dataclass
class DataQualityReport:
    """数据质量报告"""
    total_rows: int = 0
    total_columns: int = 0
    missing_values: dict = field(default_factory=dict)
    missing_percentage: dict = field(default_factory=dict)
    duplicate_rows: int = 0
    data_types: dict = field(default_factory=dict)
    unique_values: dict = field(default_factory=dict)
    quality_score: float = 0.0

class DataCleaner:
    """数据清洗器"""
    
    def __init__(self):
        self.cleaning_strategies = {
            "drop_duplicates": self._drop_duplicates,
            "fill_missing_mean": self._fill_missing_mean,
            "fill_missing_median": self._fill_missing_median,
            "fill_missing_mode": self._fill_missing_mode,
            "fill_missing_forward": self._fill_missing_forward,
            "fill_missing_backward": self._fill_missing_backward,
            "drop_missing": self._drop_missing,
            "clip_outliers": self._clip_outliers,
            "remove_outliers": self._remove_outliers,
            "normalize": self._normalize,
            "standardize": self._standardize,
        }
    
    async def analyze_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """分析数据质量"""
        report = DataQualityReport()
        
        report.total_rows = len(df)
        report.total_columns = len(df.columns)
        
        # 缺失值分析
        missing = df.isnull().sum()
        report.missing_values = missing[missing > 0].to_dict()
        report.missing_percentage = (missing / len(df) * 100).to_dict()
        
        # 重复行分析
        report.duplicate_rows = df.duplicated().sum()
        
        # 数据类型
        report.data_types = df.dtypes.astype(str).to_dict()
        
        # 唯一值数量
        report.unique_values = df.nunique().to_dict()
        
        # 计算质量分数
        missing_penalty = sum(report.missing_percentage.values()) / len(df.columns)
        duplicate_penalty = report.duplicate_rows / len(df) * 100
        report.quality_score = max(0, 100 - missing_penalty - duplicate_penalty)
        
        return report
    
    async def clean(self, df: pd.DataFrame, 
                    strategies: list[str]) -> pd.DataFrame:
        """执行清洗"""
        cleaned_df = df.copy()
        
        for strategy in strategies:
            if strategy in self.cleaning_strategies:
                cleaned_df = self.cleaning_strategies[strategy](cleaned_df)
        
        return cleaned_df
    
    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除重复行"""
        return df.drop_duplicates()
    
    def _fill_missing_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        """用均值填充缺失值"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    
    def _fill_missing_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """用中位数填充缺失值"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df
    
    def _fill_missing_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """用众数填充缺失值"""
        for col in df.columns:
            if df[col].isnull().any():
                mode = df[col].mode()
                if not mode.empty:
                    df[col] = df[col].fillna(mode.iloc[0])
        return df
    
    def _fill_missing_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """前向填充"""
        return df.fillna(method='ffill')
    
    def _fill_missing_backward(self, df: pd.DataFrame) -> pd.DataFrame:
        """后向填充"""
        return df.fillna(method='bfill')
    
    def _drop_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除含缺失值的行"""
        return df.dropna()
    
    def _clip_outliers(self, df: pd.DataFrame, 
                       lower_percentile: float = 0.01,
                       upper_percentile: float = 0.99) -> pd.DataFrame:
        """裁剪异常值"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            lower = df[col].quantile(lower_percentile)
            upper = df[col].quantile(upper_percentile)
            df[col] = df[col].clip(lower, upper)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, 
                        method: str = "iqr") -> pd.DataFrame:
        """删除异常值"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == "iqr":
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        
        elif method == "zscore":
            from scipy import stats
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                df = df[z_scores < 3]
        
        return df
    
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """归一化（Min-Max）"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
        
        return df
    
    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化（Z-score）"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
        
        return df

# 使用示例
cleaner = DataCleaner()

async def clean_example():
    # 创建示例数据
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Alice"],
        "age": [25, 30, None, 25],
        "salary": [50000, 60000, 70000, 50000]
    })
    
    # 分析数据质量
    report = await cleaner.analyze_quality(df)
    print(f"质量分数: {report.quality_score:.1f}")
    print(f"缺失值: {report.missing_values}")
    
    # 清洗数据
    cleaned_df = await cleaner.clean(df, [
        "drop_duplicates",
        "fill_missing_mean"
    ])
    print(f"清洗后行数: {len(cleaned_df)}")

# asyncio.run(clean_example())
```

## 24.4 探索性分析

### 24.4.1 描述性统计

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any

@dataclass
class DescriptiveStats:
    """描述性统计结果"""
    count: dict = field(default_factory=dict)
    mean: dict = field(default_factory=dict)
    std: dict = field(default_factory=dict)
    min: dict = field(default_factory=dict)
    q1: dict = field(default_factory=dict)
    median: dict = field(default_factory=dict)
    q3: dict = field(default_factory=dict)
    max: dict = field(default_factory=dict)
    skewness: dict = field(default_factory=dict)
    kurtosis: dict = field(default_factory=dict)

class DescriptiveAnalyzer:
    """描述性统计分析器"""
    
    async def analyze(self, df: pd.DataFrame, 
                      columns: list[str] = None) -> DescriptiveStats:
        """执行描述性统计"""
        stats = DescriptiveStats()
        
        # 选择要分析的列
        if columns:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])
        
        # 计算统计量
        stats.count = numeric_df.count().to_dict()
        stats.mean = numeric_df.mean().to_dict()
        stats.std = numeric_df.std().to_dict()
        stats.min = numeric_df.min().to_dict()
        stats.q1 = numeric_df.quantile(0.25).to_dict()
        stats.median = numeric_df.median().to_dict()
        stats.q3 = numeric_df.quantile(0.75).to_dict()
        stats.max = numeric_df.max().to_dict()
        stats.skewness = numeric_df.skew().to_dict()
        stats.kurtosis = numeric_df.kurtosis().to_dict()
        
        return stats
    
    def format_report(self, stats: DescriptiveStats) -> str:
        """格式化报告"""
        lines = ["## 描述性统计报告\n"]
        
        for col in stats.mean.keys():
            lines.append(f"### {col}")
            lines.append(f"- 计数: {stats.count[col]}")
            lines.append(f"- 均值: {stats.mean[col]:.2f}")
            lines.append(f"- 标准差: {stats.std[col]:.2f}")
            lines.append(f"- 最小值: {stats.min[col]:.2f}")
            lines.append(f"- Q1: {stats.q1[col]:.2f}")
            lines.append(f"- 中位数: {stats.median[col]:.2f}")
            lines.append(f"- Q3: {stats.q3[col]:.2f}")
            lines.append(f"- 最大值: {stats.max[col]:.2f}")
            lines.append(f"- 偏度: {stats.skewness[col]:.2f}")
            lines.append(f"- 峰度: {stats.kurtosis[col]:.2f}")
            lines.append("")
        
        return "\n".join(lines)

# 使用示例
analyzer = DescriptiveAnalyzer()

async def descriptive_example():
    df = pd.DataFrame({
        "sales": [100, 200, 150, 300, 250],
        "profit": [20, 40, 30, 60, 50],
        "quantity": [10, 20, 15, 30, 25]
    })
    
    stats = await analyzer.analyze(df)
    report = analyzer.format_report(stats)
    print(report)

# asyncio.run(descriptive_example())
```

### 24.4.2 相关性分析

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any

@dataclass
class CorrelationResult:
    """相关性分析结果"""
    method: str
    matrix: pd.DataFrame = None
    p_values: pd.DataFrame = None
    significant_pairs: list = field(default_factory=list)

class CorrelationAnalyzer:
    """相关性分析器"""
    
    async def analyze(self, df: pd.DataFrame,
                      method: str = "pearson",
                      columns: list[str] = None) -> CorrelationResult:
        """执行相关性分析"""
        # 选择列
        if columns:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])
        
        # 计算相关系数
        corr_matrix = numeric_df.corr(method=method)
        
        # 计算 p 值
        p_values = self._calculate_p_values(numeric_df, method)
        
        # 找出显著相关对
        significant_pairs = self._find_significant_pairs(
            corr_matrix, p_values, threshold=0.05
        )
        
        return CorrelationResult(
            method=method,
            matrix=corr_matrix,
            p_values=p_values,
            significant_pairs=significant_pairs
        )
    
    def _calculate_p_values(self, df: pd.DataFrame, 
                           method: str) -> pd.DataFrame:
        """计算 p 值"""
        from scipy import stats
        
        cols = df.columns
        n_cols = len(cols)
        p_values = pd.DataFrame(
            np.ones((n_cols, n_cols)),
            index=cols, columns=cols
        )
        
        for i in range(n_cols):
            for j in range(i+1, n_cols):
                if method == "pearson":
                    _, p_val = stats.pearsonr(
                        df[cols[i]].dropna(),
                        df[cols[j]].dropna()
                    )
                elif method == "spearman":
                    _, p_val = stats.spearmanr(
                        df[cols[i]].dropna(),
                        df[cols[j]].dropna()
                    )
                else:
                    p_val = 1.0
                
                p_values.iloc[i, j] = p_val
                p_values.iloc[j, i] = p_val
        
        return p_values
    
    def _find_significant_pairs(self, corr_matrix: pd.DataFrame,
                                p_values: pd.DataFrame,
                                threshold: float = 0.05) -> list[dict]:
        """找出显著相关的变量对"""
        pairs = []
        cols = corr_matrix.columns
        
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                if p_values.iloc[i, j] < threshold:
                    pairs.append({
                        "var1": cols[i],
                        "var2": cols[j],
                        "correlation": corr_matrix.iloc[i, j],
                        "p_value": p_values.iloc[i, j]
                    })
        
        # 按相关系数绝对值排序
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return pairs
    
    def format_report(self, result: CorrelationResult) -> str:
        """格式化报告"""
        lines = [f"## 相关性分析报告 ({result.method})\n"]
        
        if result.matrix is not None:
            lines.append("### 相关系数矩阵\n")
            lines.append(result.matrix.to_markdown())
            lines.append("")
        
        if result.significant_pairs:
            lines.append("### 显著相关变量对\n")
            lines.append("| 变量1 | 变量2 | 相关系数 | p值 |")
            lines.append("|-------|-------|----------|-----|")
            for pair in result.significant_pairs[:10]:
                lines.append(
                    f"| {pair['var1']} | {pair['var2']} | "
                    f"{pair['correlation']:.3f} | {pair['p_value']:.4f} |"
                )
        
        return "\n".join(lines)

# 使用示例
corr_analyzer = CorrelationAnalyzer()

async def correlation_example():
    df = pd.DataFrame({
        "advertising": [100, 200, 300, 400, 500],
        "sales": [1000, 1500, 2000, 2500, 3000],
        "profit": [200, 300, 400, 500, 600]
    })
    
    result = await corr_analyzer.analyze(df)
    report = corr_analyzer.format_report(result)
    print(report)

# asyncio.run(correlation_example())
```

### 24.4.3 分布分析

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any

@dataclass
class DistributionStats:
    """分布统计结果"""
    column: str
    histogram: dict = field(default_factory=dict)
    normality_test: dict = field(default_factory=dict)
    percentiles: dict = field(default_factory=dict)
    outliers: list = field(default_factory=list)

class DistributionAnalyzer:
    """分布分析器"""
    
    async def analyze(self, df: pd.DataFrame,
                      column: str) -> DistributionStats:
        """分析单列分布"""
        stats = DistributionStats(column=column)
        
        series = df[column].dropna()
        
        # 直方图数据
        hist, bin_edges = np.histogram(series, bins=20)
        stats.histogram = {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist()
        }
        
        # 正态性检验
        from scipy import stats as scipy_stats
        if len(series) >= 8:
            stat, p_value = scipy_stats.shapiro(series)
            stats.normality_test = {
                "statistic": stat,
                "p_value": p_value,
                "is_normal": p_value > 0.05
            }
        
        # 百分位数
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        stats.percentiles = {
            f"p{p}": np.percentile(series, p) for p in percentiles
        }
        
        # 异常值检测
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        stats.outliers = series[
            (series < lower) | (series > upper)
        ].tolist()
        
        return stats
    
    def format_report(self, stats: DistributionStats) -> str:
        """格式化报告"""
        lines = [f"## 分布分析报告: {stats.column}\n"]
        
        # 直方图
        lines.append("### 直方图数据\n")
        lines.append("| 区间 | 频数 |")
        lines.append("|------|------|")
        for i, count in enumerate(stats.histogram.get("counts", [])):
            edges = stats.histogram.get("bin_edges", [])
            if i < len(edges) - 1:
                lines.append(f"| {edges[i]:.2f} - {edges[i+1]:.2f} | {count} |")
        
        # 正态性检验
        if stats.normality_test:
            lines.append("\n### 正态性检验\n")
            nt = stats.normality_test
            lines.append(f"- 统计量: {nt['statistic']:.4f}")
            lines.append(f"- p值: {nt['p_value']:.4f}")
            lines.append(f"- 是否正态: {'是' if nt['is_normal'] else '否'}")
        
        # 百分位数
        lines.append("\n### 百分位数\n")
        for key, value in stats.percentiles.items():
            lines.append(f"- {key}: {value:.2f}")
        
        # 异常值
        if stats.outliers:
            lines.append(f"\n### 异常值 ({len(stats.outliers)} 个)\n")
            lines.append(f"- 值: {stats.outliers[:10]}")
        
        return "\n".join(lines)

# 使用示例
dist_analyzer = DistributionAnalyzer()

async def distribution_example():
    df = pd.DataFrame({
        "age": np.random.normal(35, 10, 1000)
    })
    
    stats = await dist_analyzer.analyze(df, "age")
    report = dist_analyzer.format_report(stats)
    print(report)

# asyncio.run(distribution_example())
```

## 24.5 可视化生成

### 24.5.1 图表生成器

```python
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd
import numpy as np

@dataclass
class ChartConfig:
    """图表配置"""
    chart_type: str
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    width: int = 800
    height: int = 600
    theme: str = "default"
    colors: list = field(default_factory=list)
    options: dict = field(default_factory=dict)

class ChartGenerator:
    """图表生成器"""
    
    def __init__(self):
        self.chart_types = {
            "bar": self._create_bar_chart,
            "line": self._create_line_chart,
            "pie": self._create_pie_chart,
            "scatter": self._create_scatter_chart,
            "histogram": self._create_histogram,
            "heatmap": self._create_heatmap,
            "box": self._create_box_chart,
            "treemap": self._create_treemap,
        }
    
    async def generate(self, df: pd.DataFrame, 
                       config: ChartConfig) -> Any:
        """生成图表"""
        chart_func = self.chart_types.get(config.chart_type)
        if not chart_func:
            raise ValueError(f"不支持的图表类型: {config.chart_type}")
        
        return await chart_func(df, config)
    
    async def _create_bar_chart(self, df: pd.DataFrame,
                                config: ChartConfig) -> dict:
        """创建柱状图"""
        # 使用 plotly
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[
                go.Bar(
                    x=df[config.options.get("x_col", df.columns[0])],
                    y=df[config.options.get("y_col", df.columns[1])],
                    name=config.title
                )
            ])
            
            fig.update_layout(
                title=config.title,
                xaxis_title=config.x_label or df.columns[0],
                yaxis_title=config.y_label or df.columns[1],
                width=config.width,
                height=config.height
            )
            
            return fig
            
        except ImportError:
            # 回退到 matplotlib
            return self._create_matplotlib_bar(df, config)
    
    def _create_matplotlib_bar(self, df: pd.DataFrame,
                               config: ChartConfig) -> dict:
        """使用 matplotlib 创建柱状图"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
            
            x_col = config.options.get("x_col", df.columns[0])
            y_col = config.options.get("y_col", df.columns[1])
            
            ax.bar(df[x_col], df[y_col])
            ax.set_title(config.title)
            ax.set_xlabel(config.x_label or x_col)
            ax.set_ylabel(config.y_label or y_col)
            
            plt.tight_layout()
            
            return {"figure": fig, "type": "matplotlib"}
            
        except ImportError:
            return {"error": "请安装 matplotlib"}
    
    async def _create_line_chart(self, df: pd.DataFrame,
                                 config: ChartConfig) -> dict:
        """创建折线图"""
        try:
            import plotly.graph_objects as go
            
            x_col = config.options.get("x_col", df.columns[0])
            y_cols = config.options.get("y_cols", [df.columns[1]])
            
            fig = go.Figure()
            
            for y_col in y_cols:
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode='lines+markers',
                    name=y_col
                ))
            
            fig.update_layout(
                title=config.title,
                xaxis_title=config.x_label or x_col,
                yaxis_title=config.y_label,
                width=config.width,
                height=config.height
            )
            
            return fig
            
        except ImportError:
            return {"error": "请安装 plotly"}
    
    async def _create_pie_chart(self, df: pd.DataFrame,
                                config: ChartConfig) -> dict:
        """创建饼图"""
        try:
            import plotly.graph_objects as go
            
            labels_col = config.options.get("labels_col", df.columns[0])
            values_col = config.options.get("values_col", df.columns[1])
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=df[labels_col],
                    values=df[values_col],
                    hole=0.3 if config.options.get("donut") else 0
                )
            ])
            
            fig.update_layout(
                title=config.title,
                width=config.width,
                height=config.height
            )
            
            return fig
            
        except ImportError:
            return {"error": "请安装 plotly"}
    
    async def _create_scatter_chart(self, df: pd.DataFrame,
                                    config: ChartConfig) -> dict:
        """创建散点图"""
        try:
            import plotly.graph_objects as go
            
            x_col = config.options.get("x_col", df.columns[0])
            y_col = config.options.get("y_col", df.columns[1])
            color_col = config.options.get("color_col")
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=df[color_col] if color_col else None,
                        colorscale='Viridis' if color_col else None
                    )
                )
            ])
            
            fig.update_layout(
                title=config.title,
                xaxis_title=config.x_label or x_col,
                yaxis_title=config.y_label or y_col,
                width=config.width,
                height=config.height
            )
            
            return fig
            
        except ImportError:
            return {"error": "请安装 plotly"}
    
    async def _create_histogram(self, df: pd.DataFrame,
                                config: ChartConfig) -> dict:
        """创建直方图"""
        try:
            import plotly.graph_objects as go
            
            col = config.options.get("column", df.columns[0])
            bins = config.options.get("bins", 30)
            
            fig = go.Figure(data=[
                go.Histogram(
                    x=df[col],
                    nbinsx=bins,
                    name=config.title
                )
            ])
            
            fig.update_layout(
                title=config.title,
                xaxis_title=config.x_label or col,
                yaxis_title=config.y_label or "频数",
                width=config.width,
                height=config.height
            )
            
            return fig
            
        except ImportError:
            return {"error": "请安装 plotly"}
    
    async def _create_heatmap(self, df: pd.DataFrame,
                              config: ChartConfig) -> dict:
        """创建热力图"""
        try:
            import plotly.graph_objects as go
            
            # 计算相关系数矩阵
            corr_matrix = df.select_dtypes(include=[np.number]).corr()
            
            fig = go.Figure(data=[
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu_r',
                    zmin=-1, zmax=1
                )
            ])
            
            fig.update_layout(
                title=config.title or "相关系数热力图",
                width=config.width,
                height=config.height
            )
            
            return fig
            
        except ImportError:
            return {"error": "请安装 plotly"}
    
    async def _create_box_chart(self, df: pd.DataFrame,
                                config: ChartConfig) -> dict:
        """创建箱线图"""
        try:
            import plotly.graph_objects as go
            
            cols = config.options.get("columns", df.select_dtypes(include=[np.number]).columns.tolist())
            
            fig = go.Figure()
            
            for col in cols:
                fig.add_trace(go.Box(
                    y=df[col],
                    name=col
                ))
            
            fig.update_layout(
                title=config.title,
                yaxis_title=config.y_label,
                width=config.width,
                height=config.height
            )
            
            return fig
            
        except ImportError:
            return {"error": "请安装 plotly"}
    
    async def _create_treemap(self, df: pd.DataFrame,
                              config: ChartConfig) -> dict:
        """创建树图"""
        try:
            import plotly.graph_objects as go
            
            labels_col = config.options.get("labels_col", df.columns[0])
            values_col = config.options.get("values_col", df.columns[1])
            parents_col = config.options.get("parents_col")
            
            fig = go.Figure(go.Treemap(
                labels=df[labels_col],
                parents=df[parents_col] if parents_col else "",
                values=df[values_col]
            ))
            
            fig.update_layout(
                title=config.title,
                width=config.width,
                height=config.height
            )
            
            return fig
            
        except ImportError:
            return {"error": "请安装 plotly"}

# 使用示例
chart_gen = ChartGenerator()

async def chart_example():
    df = pd.DataFrame({
        "月份": ["1月", "2月", "3月", "4月", "5月"],
        "销售额": [100, 150, 200, 250, 300],
        "利润": [20, 30, 40, 50, 60]
    })
    
    # 生成柱状图
    config = ChartConfig(
        chart_type="bar",
        title="月度销售趋势",
        x_label="月份",
        y_label="销售额"
    )
    fig = await chart_gen.generate(df, config)
    print(f"生成图表: {type(fig)}")

# asyncio.run(chart_example())
```

### 24.5.2 仪表板构建器

```python
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd

@dataclass
class Widget:
    """仪表板组件"""
    widget_type: str
    title: str
    data: Any = None
    config: dict = field(default_factory=dict)
    position: tuple = (0, 0, 1, 1)  # row, col, rowspan, colspan

class DashboardBuilder:
    """仪表板构建器"""
    
    def __init__(self):
        self.widgets: list[Widget] = []
        self.layout = {"rows": 4, "cols": 3}
    
    def add_widget(self, widget: Widget):
        """添加组件"""
        self.widgets.append(widget)
    
    async def build(self) -> dict:
        """构建仪表板"""
        return {
            "layout": self.layout,
            "widgets": [
                {
                    "type": w.widget_type,
                    "title": w.title,
                    "data": w.data,
                    "config": w.config,
                    "position": w.position
                }
                for w in self.widgets
            ]
        }
    
    def add_metric_card(self, title: str, value: Any, 
                       position: tuple = None):
        """添加指标卡片"""
        self.widgets.append(Widget(
            widget_type="metric",
            title=title,
            data={"value": value},
            position=position or (len(self.widgets), 0, 1, 1)
        ))
    
    def add_chart(self, title: str, chart_type: str,
                 data: pd.DataFrame, position: tuple = None):
        """添加图表"""
        self.widgets.append(Widget(
            widget_type="chart",
            title=title,
            data=data.to_dict() if isinstance(data, pd.DataFrame) else data,
            config={"chart_type": chart_type},
            position=position or (len(self.widgets), 0, 1, 1)
        ))
    
    def add_table(self, title: str, data: pd.DataFrame,
                 position: tuple = None):
        """添加表格"""
        self.widgets.append(Widget(
            widget_type="table",
            title=title,
            data=data.to_dict() if isinstance(data, pd.DataFrame) else data,
            position=position or (len(self.widgets), 0, 1, 1)
        ))
    
    def add_text(self, title: str, content: str,
                position: tuple = None):
        """添加文本"""
        self.widgets.append(Widget(
            widget_type="text",
            title=title,
            data={"content": content},
            position=position or (len(self.widgets), 0, 1, 1)
        ))

# 使用示例
async def dashboard_example():
    builder = DashboardBuilder()
    
    # 添加指标卡片
    builder.add_metric_card("总销售额", "¥1,000,000")
    builder.add_metric_card("订单数", "5,000")
    builder.add_metric_card("平均订单金额", "¥200")
    
    # 添加图表
    df_sales = pd.DataFrame({
        "月份": ["1月", "2月", "3月"],
        "销售额": [100, 150, 200]
    })
    builder.add_chart("销售趋势", "line", df_sales)
    
    # 添加文本
    builder.add_text("分析说明", "本月销售额同比增长20%")
    
    # 构建仪表板
    dashboard = await builder.build()
    print(f"仪表板包含 {len(dashboard['widgets'])} 个组件")

# asyncio.run(dashboard_example())
```

## 24.6 Data Interpreter 模式

### 24.6.1 概述

Data Interpreter 是一种高级数据分析模式，能够自主执行完整的数据分析流程，包括数据理解、清洗、分析、可视化和报告生成。

**Data Interpreter 的核心特点**：

| 特点 | 说明 |
|------|------|
| 自主性 | 无需人工干预，自主完成分析 |
| 迭代性 | 根据中间结果调整分析策略 |
| 可解释性 | 生成详细的分析过程说明 |
| 适应性 | 能够处理不同类型的数据和需求 |

### 24.6.2 Data Interpreter 实现

```python
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd

@dataclass
class AnalysisStep:
    """分析步骤"""
    step_id: str
    name: str
    action: str
    parameters: dict = field(default_factory=dict)
    result: Any = None
    status: str = "pending"
    error: str = ""

@dataclass
class AnalysisState:
    """分析状态"""
    task: str
    data: pd.DataFrame = None
    original_data: pd.DataFrame = None
    steps: list[AnalysisStep] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    charts: list[Any] = field(default_factory=list)
    report: str = ""
    current_step: int = 0

class DataInterpreter:
    """Data Interpreter"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.descriptive_analyzer = DescriptiveAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.distribution_analyzer = DistributionAnalyzer()
        self.chart_generator = ChartGenerator()
    
    async def analyze(self, task: str, 
                      data_source: str = None,
                      data: pd.DataFrame = None) -> AnalysisState:
        """执行完整分析"""
        state = AnalysisState(task=task)
        
        # 1. 加载数据
        state = await self._load_data(state, data_source, data)
        
        if state.data is None:
            state.report = "无法加载数据"
            return state
        
        # 2. 数据理解
        state = await self._understand_data(state)
        
        # 3. 数据清洗
        state = await self._clean_data(state)
        
        # 4. 探索性分析
        state = await self._exploratory_analysis(state)
        
        # 5. 生成可视化
        state = await self._generate_visualizations(state)
        
        # 6. 生成报告
        state = await self._generate_report(state)
        
        return state
    
    async def _load_data(self, state: AnalysisState,
                        data_source: str = None,
                        data: pd.DataFrame = None) -> AnalysisState:
        """加载数据"""
        step = AnalysisStep(
            step_id="load_data",
            name="数据加载",
            action="load"
        )
        
        try:
            if data is not None:
                state.data = data.copy()
                state.original_data = data.copy()
            elif data_source:
                source = DataSource(source_type="file", path=data_source)
                state.data = await self.data_loader.load(source)
                state.original_data = state.data.copy()
            
            step.status = "completed"
            step.result = {
                "rows": len(state.data) if state.data is not None else 0,
                "columns": len(state.data.columns) if state.data is not None else 0
            }
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
        
        state.steps.append(step)
        return state
    
    async def _understand_data(self, state: AnalysisState) -> AnalysisState:
        """理解数据"""
        step = AnalysisStep(
            step_id="understand_data",
            name="数据理解",
            action="analyze"
        )
        
        try:
            df = state.data
            
            # 基本信息
            step.result = {
                "shape": df.shape,
                "dtypes": df.dtypes.astype(str).to_dict(),
                "head": df.head().to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            # 识别关键列
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            step.result["numeric_columns"] = numeric_cols
            step.result["categorical_columns"] = categorical_cols
            
            # 添加洞察
            state.insights.append(
                f"数据包含 {len(df)} 行，{len(df.columns)} 列"
            )
            state.insights.append(
                f"数值型列: {', '.join(numeric_cols[:5])}"
            )
            state.insights.append(
                f"分类型列: {', '.join(categorical_cols[:5])}"
            )
            
            # 检查缺失值
            missing_pct = df.isnull().mean() * 100
            high_missing = missing_pct[missing_pct > 50].index.tolist()
            if high_missing:
                state.insights.append(
                    f"高缺失率列 (>50%): {', '.join(high_missing)}"
                )
            
            step.status = "completed"
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
        
        state.steps.append(step)
        return state
    
    async def _clean_data(self, state: AnalysisState) -> AnalysisState:
        """清洗数据"""
        step = AnalysisStep(
            step_id="clean_data",
            name="数据清洗",
            action="clean"
        )
        
        try:
            df = state.data
            
            # 确定清洗策略
            strategies = []
            
            # 删除重复行
            if df.duplicated().sum() > 0:
                strategies.append("drop_duplicates")
                state.insights.append(
                    f"删除了 {df.duplicated().sum()} 行重复数据"
                )
            
            # 处理缺失值
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                # 数值型用中位数填充
                numeric_missing = [c for c in missing_cols if df[c].dtype in ['int64', 'float64']]
                if numeric_missing:
                    strategies.append("fill_missing_median")
                    state.insights.append(
                        f"数值列缺失值用中位数填充: {', '.join(numeric_missing[:3])}"
                    )
                
                # 分类型用众数填充
                categorical_missing = [c for c in missing_cols if df[c].dtype not in ['int64', 'float64']]
                if categorical_missing:
                    strategies.append("fill_missing_mode")
                    state.insights.append(
                        f"分类列缺失值用众数填充: {', '.join(categorical_missing[:3])}"
                    )
            
            # 执行清洗
            if strategies:
                state.data = await self.data_cleaner.clean(df, strategies)
            
            step.status = "completed"
            step.result = {
                "rows_before": len(df),
                "rows_after": len(state.data),
                "strategies_applied": strategies
            }
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
        
        state.steps.append(step)
        return state
    
    async def _exploratory_analysis(self, state: AnalysisState) -> AnalysisState:
        """探索性分析"""
        step = AnalysisStep(
            step_id="exploratory_analysis",
            name="探索性分析",
            action="analyze"
        )
        
        try:
            df = state.data
            
            # 描述性统计
            desc_stats = await self.descriptive_analyzer.analyze(df)
            step.result["descriptive_stats"] = {
                "mean": desc_stats.mean,
                "std": desc_stats.std
            }
            
            # 相关性分析（仅数值列）
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                corr_result = await self.correlation_analyzer.analyze(
                    df, columns=numeric_cols[:5]  # 限制列数
                )
                if corr_result.significant_pairs:
                    top_corr = corr_result.significant_pairs[0]
                    state.insights.append(
                        f"最强相关: {top_corr['var1']} 与 {top_corr['var2']} "
                        f"(r={top_corr['correlation']:.3f})"
                    )
            
            # 分布分析（关键数值列）
            for col in numeric_cols[:3]:
                dist_stats = await self.distribution_analyzer.analyze(df, col)
                if dist_stats.normality_test:
                    is_normal = dist_stats.normality_test.get("is_normal", False)
                    state.insights.append(
                        f"{col} 分布{'近似正态' if is_normal else '非正态'}"
                    )
            
            step.status = "completed"
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
        
        state.steps.append(step)
        return state
    
    async def _generate_visualizations(self, state: AnalysisState) -> AnalysisState:
        """生成可视化"""
        step = AnalysisStep(
            step_id="generate_visualizations",
            name="生成可视化",
            action="visualize"
        )
        
        try:
            df = state.data
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # 生成柱状图（如果分类列存在）
            if categorical_cols and numeric_cols:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                
                # 聚合数据
                agg_df = df.groupby(cat_col)[num_col].sum().reset_index()
                agg_df = agg_df.sort_values(num_col, ascending=False).head(10)
                
                config = ChartConfig(
                    chart_type="bar",
                    title=f"{num_col} by {cat_col}",
                    options={"x_col": cat_col, "y_col": num_col}
                )
                chart = await self.chart_generator.generate(agg_df, config)
                state.charts.append(chart)
            
            # 生成折线图（如果有时间相关列）
            date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
            if date_cols and numeric_cols:
                date_col = date_cols[0]
                num_col = numeric_cols[0]
                
                time_df = df.groupby(date_col)[num_col].sum().reset_index()
                
                config = ChartConfig(
                    chart_type="line",
                    title=f"{num_col} over Time",
                    options={"x_col": date_col, "y_cols": [num_col]}
                )
                chart = await self.chart_generator.generate(time_df, config)
                state.charts.append(chart)
            
            # 生成热力图（相关系数）
            if len(numeric_cols) >= 2:
                config = ChartConfig(
                    chart_type="heatmap",
                    title="Correlation Matrix"
                )
                chart = await self.chart_generator.generate(df, config)
                state.charts.append(chart)
            
            step.status = "completed"
            step.result = {"charts_generated": len(state.charts)}
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
        
        state.steps.append(step)
        return state
    
    async def _generate_report(self, state: AnalysisState) -> AnalysisState:
        """生成报告"""
        step = AnalysisStep(
            step_id="generate_report",
            name="生成报告",
            action="report"
        )
        
        try:
            # 构建报告
            report_lines = [
                "# 数据分析报告\n",
                f"## 任务\n{state.task}\n",
                f"## 数据概览\n",
                f"- 数据行数: {len(state.data)}",
                f"- 数据列数: {len(state.data.columns)}",
                f"- 列名: {', '.join(state.data.columns.tolist()[:10])}",
                "",
                "## 分析洞察\n"
            ]
            
            for i, insight in enumerate(state.insights, 1):
                report_lines.append(f"{i}. {insight}")
            
            report_lines.extend([
                "",
                "## 分析步骤\n"
            ])
            
            for step_item in state.steps:
                status_icon = "✅" if step_item.status == "completed" else "❌"
                report_lines.append(f"- {status_icon} {step_item.name}")
            
            report_lines.extend([
                "",
                "## 可视化\n",
                f"生成了 {len(state.charts)} 个图表",
                "",
                "## 建议\n",
                "1. 进一步分析高相关性变量的关系",
                "2. 检查异常值的业务含义",
                "3. 考虑添加更多特征进行深入分析"
            ])
            
            state.report = "\n".join(report_lines)
            step.status = "completed"
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
        
        state.steps.append(step)
        return state

# 使用示例
interpreter = DataInterpreter()

async def interpreter_example():
    # 创建示例数据
    df = pd.DataFrame({
        "日期": pd.date_range("2024-01-01", periods=30),
        "销售额": np.random.randint(100, 1000, 30),
        "利润": np.random.randint(20, 200, 30),
        "数量": np.random.randint(10, 100, 30),
        "地区": np.random.choice(["北京", "上海", "广州"], 30)
    })
    
    # 执行分析
    state = await interpreter.analyze(
        task="分析过去30天的销售数据",
        data=df
    )
    
    print(state.report)

# asyncio.run(interpreter_example())
```

## 24.7 报告撰写

### 24.7.1 自动化报告生成

```python
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime

@dataclass
class ReportSection:
    """报告章节"""
    title: str
    content: str
    level: int = 2  # 1: 主标题, 2: 二级标题, 3: 三级标题
    charts: list = field(default_factory=list)
    tables: list = field(default_factory=list)

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    async def generate(self, state: AnalysisState,
                      title: str = "数据分析报告") -> str:
        """生成完整报告"""
        sections = []
        
        # 1. 执行摘要
        sections.append(self._create_executive_summary(state))
        
        # 2. 数据概览
        sections.append(self._create_data_overview(state))
        
        # 3. 分析方法
        sections.append(self._create_methodology(state))
        
        # 4. 主要发现
        sections.append(self._create_findings(state))
        
        # 5. 可视化展示
        sections.append(self._create_visualizations_section(state))
        
        # 6. 建议与结论
        sections.append(self._create_recommendations(state))
        
        # 组装报告
        return self._assemble_report(title, sections, state)
    
    def _create_executive_summary(self, state: AnalysisState) -> ReportSection:
        """创建执行摘要"""
        content = f"""
本次分析针对 {state.task} 进行了全面的数据探索和分析。

**主要发现**：
"""
        for insight in state.insights[:5]:
            content += f"- {insight}\n"
        
        content += f"""
**数据规模**：{len(state.data)} 行记录，{len(state.data.columns)} 个变量
**分析步骤**：共执行 {len(state.steps)} 个分析步骤
"""
        
        return ReportSection(
            title="执行摘要",
            content=content,
            level=1
        )
    
    def _create_data_overview(self, state: AnalysisState) -> ReportSection:
        """创建数据概览"""
        df = state.data
        
        content = f"""
## 数据来源

数据包含 {len(df)} 行记录和 {len(df.columns)} 个变量。

## 变量说明

### 数值型变量
"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols[:10]:
            content += f"- {col}: 均值={df[col].mean():.2f}, 标准差={df[col].std():.2f}\n"
        
        content += "\n### 分类型变量\n"
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols[:10]:
            unique_count = df[col].nunique()
            content += f"- {col}: {unique_count} 个唯一值\n"
        
        return ReportSection(
            title="数据概览",
            content=content
        )
    
    def _create_methodology(self, state: AnalysisState) -> ReportSection:
        """创建方法论"""
        content = "## 分析方法\n\n"
        content += "本次分析采用了以下方法：\n\n"
        
        for step in state.steps:
            content += f"### {step.name}\n"
            content += f"- 状态: {'完成' if step.status == 'completed' else '失败'}\n"
            if step.result:
                content += f"- 结果: {step.result}\n"
            content += "\n"
        
        return ReportSection(
            title="分析方法",
            content=content
        )
    
    def _create_findings(self, state: AnalysisState) -> ReportSection:
        """创建主要发现"""
        content = "## 主要发现\n\n"
        
        for i, insight in enumerate(state.insights, 1):
            content += f"{i}. {insight}\n"
        
        return ReportSection(
            title="主要发现",
            content=content
        )
    
    def _create_visualizations_section(self, state: AnalysisState) -> ReportSection:
        """创建可视化章节"""
        content = f"## 可视化展示\n\n"
        content += f"本次分析生成了 {len(state.charts)} 个可视化图表。\n"
        content += "图表展示了数据的关键趋势和模式。\n"
        
        return ReportSection(
            title="可视化展示",
            content=content,
            charts=state.charts
        )
    
    def _create_recommendations(self, state: AnalysisState) -> ReportSection:
        """创建建议与结论"""
        content = """
## 建议

基于分析结果，提出以下建议：

1. **深入分析**：对发现的强相关性变量进行更深入的因果分析
2. **数据质量**：持续监控数据质量，确保分析结果的可靠性
3. **业务应用**：将分析洞察转化为具体的业务行动

## 结论

本次分析成功揭示了数据中的关键模式和趋势。建议业务团队根据分析结果采取相应行动，并持续跟踪关键指标的变化。
"""
        
        return ReportSection(
            title="建议与结论",
            content=content
        )
    
    def _assemble_report(self, title: str, 
                        sections: list[ReportSection],
                        state: AnalysisState) -> str:
        """组装报告"""
        report = f"# {title}\n\n"
        report += f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "---\n\n"
        
        for section in sections:
            report += section.content + "\n\n"
        
        report += "---\n\n"
        report += "*本报告由数据分析 Agent 自动生成*\n"
        
        return report

# 使用示例
report_gen = ReportGenerator()

async def report_example(state: AnalysisState):
    report = await report_gen.generate(state, title="销售数据分析报告")
    print(report)

# asyncio.run(report_example(interpreter_state))
```

## 24.8 本章小结

本章深入探讨了数据分析 Agent 的核心概念和实现：

1. **核心架构**：数据分析 Agent 的分层架构（任务理解→数据处理→分析引擎→可视化→报告）
2. **任务理解**：需求分析、分析类型识别、数据源识别
3. **数据加载**：支持 CSV、Excel、JSON、Parquet 等多种格式
4. **数据清洗**：缺失值处理、异常值检测、数据标准化
5. **探索性分析**：描述性统计、相关性分析、分布分析
6. **可视化生成**：柱状图、折线图、饼图、散点图、热力图等
7. **Data Interpreter**：自主执行完整分析流程的高级模式
8. **报告撰写**：自动化报告生成、结构化输出

## 24.9 思考题

1. 如何设计一个能够处理 TB 级大数据的数据分析 Agent？
2. Data Interpreter 如何处理分析过程中的错误和异常？
3. 如何评估数据分析 Agent 生成的洞察的质量？
4. 可视化选择有哪些最佳实践？如何自动选择合适的图表类型？
5. 如何设计一个支持交互式分析的 Agent？
6. 数据分析 Agent 如何处理非结构化数据（如文本、图像）？
7. 如何保证数据分析结果的可重复性和可验证性？
