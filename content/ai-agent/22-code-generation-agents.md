---
title: "第22章：代码生成 Agent"
description: "深入解析代码生成 Agent 的架构设计、SWE-bench 评测、主流产品原理及未来趋势"
updated: "2026-06-15"
---

# 第22章：代码生成 Agent

 > **学习目标**：
 > - 掌握代码生成 Agent 的完整架构流程（理解→定位→设计→生成→测试→修复）
 > - 深入了解 SWE-bench 评测基准及其排行榜
 > - 理解代码生成 Agent 面临的主要挑战与应对策略
 > - 掌握 Cursor、Devin 等主流代码 Agent 的核心原理
 > - 能够设计和实现一个完整的代码生成 Agent

 下面的交互式演示展示了代码生成 Agent 的完整流程：

 <div data-component="CodeGenPipeline"></div>

 ## 22.1 代码生成 Agent 概述

### 22.1.1 什么是代码生成 Agent

代码生成 Agent 是一种专门用于辅助或自动化软件开发的 AI 系统。它能够理解自然语言需求、分析现有代码库、生成代码、运行测试并修复问题。

**代码生成 Agent 的核心能力**：

| 能力 | 说明 | 示例 |
|------|------|------|
| 代码理解 | 理解现有代码库的结构和逻辑 | 解释函数功能、追踪调用链 |
| 代码生成 | 根据需求生成新代码 | 实现新功能、编写算法 |
| 代码修改 | 修改现有代码以修复问题 | 修复 bug、重构代码 |
| 测试生成 | 生成单元测试和集成测试 | 编写测试用例、验证正确性 |
| 调试修复 | 分析错误并提供修复方案 | 定位 bug、提出修复建议 |
| 代码审查 | 评估代码质量并提出改进建议 | 安全审查、性能优化 |

> **关键洞察**：代码生成 Agent 的核心价值不仅在于"写代码"，更在于"理解需求→分析代码→生成方案→验证结果"的完整闭环能力。

### 22.1.2 代码 Agent 与传统代码补全的区别

| 维度 | 传统代码补全 | 代码生成 Agent |
|------|------------|---------------|
| 输入 | 当前代码上下文 | 自然语言需求 + 代码库 |
| 范围 | 当前文件 | 整个代码库 |
| 输出 | 代码片段 | 完整功能实现 |
| 交互 | 单次请求-响应 | 多轮对话与迭代 |
| 能力 | 语法补全 | 理解、推理、规划 |
| 测试 | 无 | 自动测试与验证 |

### 22.1.3 代码 Agent 的发展历程

```
┌─────────────────────────────────────────────────────────────┐
│                    代码 Agent 发展历程                        │
│                                                             │
│  2021 │ Codex (GitHub Copilot)                              │
│       │ → 基于 GPT-3 的代码生成                              │
│       │                                                     │
│  2022 │ ChatGPT + Code Interpreter                          │
│       │ → 代码生成 + 交互式执行                              │
│       │                                                     │
│  2023 │ GPT-4 + GitHub Copilot X                            │
│       │ → 多文件理解 + 对话式编程                             │
│       │                                                     │
│  2024 │ Devin, Cursor, SWE-bench                            │
│       │ → 自主代码 Agent + 评测基准                           │
│       │                                                     │
│  2025 │ 多 Agent 协作 + 全栈开发                             │
│       │ → 端到端软件工程                                     │
│       │                                                     │
│  2026 │ Agent-native IDE + 自主开发                          │
│       │ → 软件工程自动化                                     │
└─────────────────────────────────────────────────────────────┘
```

## 22.2 代码 Agent 架构

### 22.2.1 核心架构流程

代码生成 Agent 的核心流程可以概括为六个阶段：

```
┌─────────────────────────────────────────────────────────────┐
│                  代码 Agent 核心架构                          │
│                                                             │
│  ┌─────────┐                                                │
│  │ 理解    │ ← 分析需求、理解上下文                           │
│  │Understand│                                               │
│  └────┬────┘                                                │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐                                                │
│  │ 定位    │ ← 在代码库中定位相关文件                          │
│  │Locate   │                                                │
│  └────┬────┘                                                │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐                                                │
│  │ 设计    │ ← 设计解决方案                                  │
│  │Design   │                                                │
│  └────┬────┘                                                │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐                                                │
│  │ 生成    │ ← 生成代码                                      │
│  │Generate │                                                │
│  └────┬────┘                                                │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐                                                │
│  │ 测试    │ ← 运行测试验证                                  │
│  │Test     │                                                │
│  └────┬────┘                                                │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐                                                │
│  │ 修复    │ ← 根据测试结果修复                               │
│  │Fix      │                                                │
│  └─────────┘                                                │
└─────────────────────────────────────────────────────────────┘
```

### 22.2.2 理解阶段（Understanding）

理解阶段是代码 Agent 的第一步，需要分析用户需求和现有代码库。

```python
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import re

class RequirementType(Enum):
    """需求类型"""
    NEW_FEATURE = "new_feature"          # 新功能
    BUG_FIX = "bug_fix"                  # Bug 修复
    REFACTOR = "refactor"                # 重构
    OPTIMIZATION = "optimization"        # 性能优化
    DOCUMENTATION = "documentation"      # 文档
    TEST = "test"                        # 测试
    SECURITY = "security"                # 安全修复

@dataclass
class CodeContext:
    """代码上下文"""
    file_path: str
    content: str
    language: str = ""
    relevance_score: float = 0.0
    metadata: dict = field(default_factory=dict)

@dataclass
class Requirement:
    """需求分析结果"""
    original_text: str
    requirement_type: RequirementType
    description: str
    affected_files: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    priority: int = 1

class RequirementAnalyzer:
    """需求分析器"""
    
    def __init__(self):
        self.patterns = self._init_patterns()
    
    def _init_patterns(self) -> dict[RequirementType, list[str]]:
        """初始化需求模式匹配"""
        return {
            RequirementType.BUG_FIX: [
                r"修复|fix|bug|错误|error|异常|exception",
                r"崩溃|crash|失败|fail|不工作|not work",
            ],
            RequirementType.NEW_FEATURE: [
                r"添加|add|实现|implement|新增|create",
                r"功能|feature|需求|requirement|支持|support",
            ],
            RequirementType.REFACTOR: [
                r"重构|refactor|优化|optimize|改进|improve",
                r"简化|simplify|整理|clean|整理|reorganize",
            ],
            RequirementType.OPTIMIZATION: [
                r"性能|performance|速度|speed|加速|faster",
                r"内存|memory|优化|optimize|效率|efficiency",
            ],
            RequirementType.TEST: [
                r"测试|test|用例|case|覆盖|coverage",
                r"单元测试|unit test|集成测试|integration test",
            ],
            RequirementType.DOCUMENTATION: [
                r"文档|doc|注释|comment|说明|description",
                r"README|wiki|帮助|help",
            ],
        }
    
    def analyze(self, text: str) -> Requirement:
        """分析需求文本"""
        # 识别需求类型
        req_type = self._identify_type(text)
        
        # 提取描述
        description = self._extract_description(text)
        
        # 提取约束条件
        constraints = self._extract_constraints(text)
        
        return Requirement(
            original_text=text,
            requirement_type=req_type,
            description=description,
            constraints=constraints
        )
    
    def _identify_type(self, text: str) -> RequirementType:
        """识别需求类型"""
        text_lower = text.lower()
        
        scores = {}
        for req_type, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            scores[req_type] = score
        
        if max(scores.values()) == 0:
            return RequirementType.NEW_FEATURE
        
        return max(scores, key=scores.get)
    
    def _extract_description(self, text: str) -> str:
        """提取需求描述"""
        # 移除常见的前缀
        prefixes = [
            r"请帮我|请|please|I need to|I want to|我想",
            r"应该|should|需要|need to|必须|must",
        ]
        
        description = text
        for prefix in prefixes:
            description = re.sub(prefix, "", description, flags=re.IGNORECASE)
        
        return description.strip()
    
    def _extract_constraints(self, text: str) -> list[str]:
        """提取约束条件"""
        constraints = []
        
        # 性能约束
        if re.search(r"时间复杂度|time complexity", text, re.IGNORECASE):
            constraints.append("需要考虑时间复杂度")
        
        if re.search(r"空间复杂度|space complexity", text, re.IGNORECASE):
            constraints.append("需要考虑空间复杂度")
        
        # 兼容性约束
        if re.search(r"向后兼容|backward compatible", text, re.IGNORECASE):
            constraints.append("必须向后兼容")
        
        # 安全约束
        if re.search(r"安全|security|加密|encrypt", text, re.IGNORECASE):
            constraints.append("需要考虑安全性")
        
        return constraints

# 使用示例
analyzer = RequirementAnalyzer()
req = analyzer.analyze("请帮我修复登录模块的密码验证bug，需要考虑安全性")
print(f"需求类型: {req.requirement_type.value}")
print(f"描述: {req.description}")
print(f"约束: {req.constraints}")
```

### 22.2.3 定位阶段（Locating）

定位阶段需要在代码库中找到与需求相关的文件和代码段。

```python
from pathlib import Path
from typing import Any
import os

class CodeLocator:
    """代码定位器"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.index: dict[str, Any] = {}
    
    def build_index(self):
        """构建代码索引"""
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and self._is_code_file(file_path):
                self._index_file(file_path)
    
    def _is_code_file(self, path: Path) -> bool:
        """检查是否为代码文件"""
        code_extensions = {
            ".py", ".js", ".ts", ".jsx", ".tsx",
            ".java", ".go", ".rs", ".cpp", ".c",
            ".h", ".hpp", ".cs", ".rb", ".php",
        }
        return path.suffix.lower() in code_extensions
    
    def _index_file(self, file_path: Path):
        """索引文件"""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            rel_path = str(file_path.relative_to(self.repo_path))
            
            self.index[rel_path] = {
                "path": str(file_path),
                "content": content,
                "language": file_path.suffix.lstrip("."),
                "size": len(content),
                "lines": content.count("\n") + 1,
                "functions": self._extract_functions(content, file_path.suffix),
                "classes": self._extract_classes(content, file_path.suffix),
                "imports": self._extract_imports(content, file_path.suffix),
            }
        except Exception as e:
            print(f"索引文件失败 {file_path}: {e}")
    
    def _extract_functions(self, content: str, ext: str) -> list[dict]:
        """提取函数定义"""
        functions = []
        
        if ext in [".py"]:
            # Python 函数
            pattern = r"(?:def|async\s+def)\s+(\w+)\s*\(([^)]*)\)"
            for match in re.finditer(pattern, content):
                functions.append({
                    "name": match.group(1),
                    "params": match.group(2),
                    "line": content[:match.start()].count("\n") + 1
                })
        elif ext in [".js", ".ts", ".jsx", ".tsx"]:
            # JavaScript/TypeScript 函数
            pattern = r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))"
            for match in re.finditer(pattern, content):
                name = match.group(1) or match.group(2)
                functions.append({
                    "name": name,
                    "line": content[:match.start()].count("\n") + 1
                })
        
        return functions
    
    def _extract_classes(self, content: str, ext: str) -> list[dict]:
        """提取类定义"""
        classes = []
        
        if ext in [".py"]:
            pattern = r"class\s+(\w+)(?:\(([^)]*)\))?:"
            for match in re.finditer(pattern, content):
                classes.append({
                    "name": match.group(1),
                    "bases": match.group(2),
                    "line": content[:match.start()].count("\n") + 1
                })
        elif ext in [".js", ".ts", ".jsx", ".tsx"]:
            pattern = r"class\s+(\w+)(?:\s+extends\s+(\w+))?"
            for match in re.finditer(pattern, content):
                classes.append({
                    "name": match.group(1),
                    "bases": match.group(2),
                    "line": content[:match.start()].count("\n") + 1
                })
        
        return classes
    
    def _extract_imports(self, content: str, ext: str) -> list[str]:
        """提取导入语句"""
        imports = []
        
        if ext in [".py"]:
            pattern = r"(?:from\s+(\S+)\s+import|import\s+(\S+))"
            for match in re.finditer(pattern, content):
                imports.append(match.group(1) or match.group(2))
        elif ext in [".js", ".ts", ".jsx", ".tsx"]:
            pattern = r"import\s+.*?from\s+['\"]([^'\"]+)['\"]"
            for match in re.finditer(pattern, content):
                imports.append(match.group(1))
        
        return imports
    
    def find_relevant_files(self, requirement: Requirement, 
                           top_k: int = 10) -> list[CodeContext]:
        """查找相关文件"""
        results = []
        
        for path, info in self.index.items():
            score = self._calculate_relevance(path, info, requirement)
            if score > 0:
                results.append(CodeContext(
                    file_path=path,
                    content=info["content"],
                    language=info["language"],
                    relevance_score=score,
                    metadata=info
                ))
        
        # 按相关性排序
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]
    
    def _calculate_relevance(self, path: str, info: dict,
                            requirement: Requirement) -> float:
        """计算文件相关性"""
        score = 0.0
        desc = requirement.description.lower()
        
        # 文件名匹配
        path_parts = Path(path).parts
        for part in path_parts:
            if part.lower() in desc:
                score += 2.0
        
        # 函数名匹配
        for func in info.get("functions", []):
            if func["name"].lower() in desc:
                score += 3.0
        
        # 类名匹配
        for cls in info.get("classes", []):
            if cls["name"].lower() in desc:
                score += 3.0
        
        # 内容关键词匹配
        content_lower = info["content"].lower()
        keywords = desc.split()
        for keyword in keywords:
            if len(keyword) > 2 and keyword in content_lower:
                score += 0.5
        
        # 文件大小调整（避免过大或过小的文件）
        size = info.get("size", 0)
        if size > 100000:
            score *= 0.5
        elif size < 100:
            score *= 0.3
        
        return score

# 使用示例
locator = CodeLocator("/path/to/repo")
locator.build_index()
relevant_files = locator.find_relevant_files(req)
for f in relevant_files[:5]:
    print(f"{f.file_path} (相关性: {f.relevance_score:.2f})")
```

### 22.2.4 设计阶段（Designing）

设计阶段需要根据需求和代码上下文，设计解决方案。

```python
from dataclasses import dataclass, field
from enum import Enum

class SolutionApproach(Enum):
    """解决方案方法"""
    MINIMAL_CHANGE = "minimal_change"      # 最小化修改
    FULL_REWRITE = "full_rewrite"          # 完全重写
    ADDITIVE = "additive"                  # 添加新代码
    REFACOR = "refactor"                   # 重构现有代码

@dataclass
class CodeChange:
    """代码变更"""
    file_path: str
    change_type: str  # "add", "modify", "delete"
    start_line: int = 0
    end_line: int = 0
    old_code: str = ""
    new_code: str = ""
    description: str = ""

@dataclass
class SolutionDesign:
    """解决方案设计"""
    approach: SolutionApproach
    summary: str
    changes: list[CodeChange] = field(default_factory=list)
    testing_strategy: str = ""
    risk_assessment: str = ""
    alternatives: list[str] = field(default_factory=list)

class SolutionDesigner:
    """解决方案设计器"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    async def design(self, requirement: Requirement,
                     context_files: list[CodeContext]) -> SolutionDesign:
        """设计解决方案"""
        # 构建上下文
        context = self._build_context(requirement, context_files)
        
        # 生成设计方案
        if self.llm_client:
            design = await self._generate_with_llm(context)
        else:
            design = self._generate_template(requirement, context_files)
        
        return design
    
    def _build_context(self, requirement: Requirement,
                      context_files: list[CodeContext]) -> str:
        """构建上下文"""
        parts = [
            f"需求类型: {requirement.requirement_type.value}",
            f"需求描述: {requirement.description}",
            f"约束条件: {', '.join(requirement.constraints) if requirement.constraints else '无'}",
            "\n相关代码文件:\n"
        ]
        
        for f in context_files[:5]:
            # 截取关键部分
            content_preview = f.content[:2000]
            if len(f.content) > 2000:
                content_preview += "\n... (已截断)"
            
            parts.append(f"### {f.file_path}\n```{f.language}\n{content_preview}\n```\n")
        
        return "\n".join(parts)
    
    def _generate_template(self, requirement: Requirement,
                          context_files: list[CodeContext]) -> SolutionDesign:
        """生成模板化的设计方案"""
        if requirement.requirement_type == RequirementType.BUG_FIX:
            return SolutionDesign(
                approach=SolutionApproach.MINIMAL_CHANGE,
                summary=f"修复 {requirement.description}",
                changes=[
                    CodeChange(
                        file_path=context_files[0].file_path if context_files else "unknown",
                        change_type="modify",
                        description="修复相关代码"
                    )
                ],
                testing_strategy="编写单元测试验证修复",
                risk_assessment="低风险，修改范围可控"
            )
        elif requirement.requirement_type == RequirementType.NEW_FEATURE:
            return SolutionDesign(
                approach=SolutionApproach.ADDITIVE,
                summary=f"实现 {requirement.description}",
                changes=[
                    CodeChange(
                        file_path="new_module.py",
                        change_type="add",
                        new_code="# 新功能代码",
                        description="添加新功能模块"
                    )
                ],
                testing_strategy="编写功能测试和集成测试",
                risk_assessment="中等风险，需要确保与现有代码兼容"
            )
        else:
            return SolutionDesign(
                approach=SolutionApproach.MINIMAL_CHANGE,
                summary=f"处理 {requirement.description}",
                changes=[],
                testing_strategy="根据变更类型确定测试策略",
                risk_assessment="需要评估"
            )
    
    async def _generate_with_llm(self, context: str) -> SolutionDesign:
        """使用 LLM 生成设计方案"""
        # 在实际实现中，这里会调用 LLM API
        # 这里简化为模板
        return SolutionDesign(
            approach=SolutionApproach.MINIMAL_CHANGE,
            summary="基于 LLM 分析的设计方案",
            changes=[],
            testing_strategy="LLM 建议的测试策略",
            risk_assessment="LLM 评估的风险"
        )

# 使用示例
designer = SolutionDesigner()
design = designer._generate_template(req, relevant_files)
print(f"方法: {design.approach.value}")
print(f"摘要: {design.summary}")
print(f"变更数: {len(design.changes)}")
```

### 22.2.5 生成阶段（Generating）

生成阶段根据设计方案生成具体的代码。

```python
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class GeneratedCode:
    """生成的代码"""
    file_path: str
    content: str
    language: str
    line_count: int = 0
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

class CodeGenerator:
    """代码生成器"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.code_templates: dict[str, str] = self._init_templates()
    
    def _init_templates(self) -> dict[str, str]:
        """初始化代码模板"""
        return {
            "python_function": '''def {function_name}({params}):
    """{docstring}"""
    {body}
''',
            "python_class": '''class {class_name}:
    """{docstring}"""
    
    def __init__(self{init_params}):
        {init_body}
    
    {methods}
''',
            "python_test": '''import pytest
from {module} import {class_or_func}


class Test{class_or_func}:
    """测试 {class_or_func}"""
    
    def test_{method_name}(self):
        """测试 {method_name} 方法"""
        # Arrange
        {setup}
        
        # Act
        {action}
        
        # Assert
        {assertion}
''',
            "javascript_function": '''/**
 * {description}
 * @param {params}
 * @returns {returns}
 */
function {function_name}({params}) {{
    {body}
}}
''',
        }
    
    async def generate(self, design: SolutionDesign,
                      context_files: list[CodeContext]) -> list[GeneratedCode]:
        """根据设计方案生成代码"""
        generated_codes = []
        
        for change in design.changes:
            if change.change_type == "add":
                code = await self._generate_new_code(change, context_files)
            elif change.change_type == "modify":
                code = await self._generate_modified_code(change, context_files)
            else:
                continue
            
            if code:
                generated_codes.append(code)
        
        return generated_codes
    
    async def _generate_new_code(self, change: CodeChange,
                                context_files: list[CodeContext]) -> Optional[GeneratedCode]:
        """生成新代码"""
        if self.llm_client:
            return await self._generate_with_llm(change, context_files)
        
        # 模板生成
        language = self._detect_language(change.file_path)
        
        if language == "python":
            content = self._generate_python_code(change)
        elif language in ["javascript", "typescript"]:
            content = self._generate_javascript_code(change)
        else:
            content = f"# Generated code for {change.description}\n# TODO: Implement\n"
        
        return GeneratedCode(
            file_path=change.file_path,
            content=content,
            language=language,
            line_count=content.count("\n") + 1,
            description=change.description
        )
    
    async def _generate_modified_code(self, change: CodeChange,
                                     context_files: list[CodeContext]) -> Optional[GeneratedCode]:
        """生成修改后的代码"""
        # 查找原始文件
        original_content = ""
        for f in context_files:
            if f.file_path == change.file_path:
                original_content = f.content
                break
        
        if not original_content:
            return None
        
        # 应用变更
        lines = original_content.split("\n")
        
        if change.old_code and change.new_code:
            # 查找并替换
            old_lines = change.old_code.split("\n")
            for i in range(len(lines)):
                if lines[i:i+len(old_lines)] == old_lines:
                    lines[i:i+len(old_lines)] = change.new_code.split("\n")
                    break
        
        new_content = "\n".join(lines)
        
        return GeneratedCode(
            file_path=change.file_path,
            content=new_content,
            language=self._detect_language(change.file_path),
            line_count=new_content.count("\n") + 1,
            description=change.description
        )
    
    def _detect_language(self, file_path: str) -> str:
        """检测编程语言"""
        ext = Path(file_path).suffix.lower()
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "jsx",
            ".tsx": "tsx",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
        }
        return lang_map.get(ext, "unknown")
    
    def _generate_python_code(self, change: CodeChange) -> str:
        """生成 Python 代码"""
        # 根据描述推断函数名
        func_name = self._infer_function_name(change.description)
        
        template = self.code_templates["python_function"]
        return template.format(
            function_name=func_name,
            params="...",
            docstring=change.description,
            body="    pass  # TODO: Implement"
        )
    
    def _generate_javascript_code(self, change: CodeChange) -> str:
        """生成 JavaScript 代码"""
        func_name = self._infer_function_name(change.description)
        
        template = self.code_templates["javascript_function"]
        return template.format(
            function_name=func_name,
            description=change.description,
            params="...",
            returns="...",
            body="// TODO: Implement"
        )
    
    def _infer_function_name(self, description: str) -> str:
        """从描述推断函数名"""
        # 简单的启发式方法
        words = re.findall(r'\w+', description.lower())
        
        # 过滤常见停用词
        stop_words = {"the", "a", "an", "is", "are", "was", "were",
                      "to", "of", "in", "for", "on", "with", "by"}
        meaningful_words = [w for w in words if w not in stop_words]
        
        if not meaningful_words:
            return "generated_function"
        
        # 组合为函数名
        return "_".join(meaningful_words[:3])

class CodeFormatter:
    """代码格式化器"""
    
    def __init__(self):
        self.formatters = {
            "python": self._format_python,
            "javascript": self._format_javascript,
            "typescript": self._format_typescript,
        }
    
    def format(self, code: GeneratedCode) -> GeneratedCode:
        """格式化代码"""
        formatter = self.formatters.get(code.language)
        if formatter:
            code.content = formatter(code.content)
        return code
    
    def _format_python(self, content: str) -> str:
        """格式化 Python 代码"""
        # 在实际实现中，这里会调用 black 或 autopep8
        # 这里简化为基础格式化
        lines = content.split("\n")
        formatted_lines = []
        
        for line in lines:
            # 移除尾部空白
            line = line.rstrip()
            formatted_lines.append(line)
        
        return "\n".join(formatted_lines)
    
    def _format_javascript(self, content: str) -> str:
        """格式化 JavaScript 代码"""
        # 在实际实现中，这里会调用 prettier
        return content
    
    def _format_typescript(self, content: str) -> str:
        """格式化 TypeScript 代码"""
        return self._format_javascript(content)

# 使用示例
generator = CodeGenerator()
designer = SolutionDesigner()

# 设计解决方案
design = designer._generate_template(req, relevant_files)

# 生成代码
generated_codes = asyncio.run(generator.generate(design, relevant_files))

# 格式化代码
formatter = CodeFormatter()
for code in generated_codes:
    formatted = formatter.format(code)
    print(f"生成文件: {formatted.file_path}")
    print(f"行数: {formatted.line_count}")
```

### 22.2.6 测试阶段（Testing）

测试阶段验证生成代码的正确性。

```python
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"

@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    status: TestStatus
    duration: float = 0.0
    message: str = ""
    assertions: list[dict] = field(default_factory=list)
    coverage: float = 0.0

@dataclass
class TestSuite:
    """测试套件"""
    name: str
    results: list[TestResult] = field(default_factory=list)
    total_duration: float = 0.0
    
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)
    
    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)
    
    @property
    def total(self) -> int:
        return len(self.results)
    
    @property
    def success_rate(self) -> float:
        return self.passed / max(self.total, 1)

class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.runners = {
            "python": self._run_python_tests,
            "javascript": self._run_javascript_tests,
            "typescript": self._run_typescript_tests,
        }
    
    async def run(self, code: GeneratedCode,
                  test_code: str = None) -> TestSuite:
        """运行测试"""
        suite = TestSuite(name=f"Test {code.file_path}")
        
        # 运行单元测试
        if test_code:
            result = await self._run_test_code(test_code, code.language)
            suite.results.append(result)
        
        # 运行现有测试
        existing_results = await self._run_existing_tests(code.file_path)
        suite.results.extend(existing_results)
        
        return suite
    
    async def _run_test_code(self, test_code: str, 
                            language: str) -> TestResult:
        """运行测试代码"""
        runner = self.runners.get(language, self._run_python_tests)
        return await runner(test_code)
    
    async def _run_python_tests(self, code: str) -> TestResult:
        """运行 Python 测试"""
        start_time = time.time()
        
        try:
            # 写入临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', 
                                           delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            # 运行 pytest
            result = subprocess.run(
                ["python", "-m", "pytest", temp_path, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return TestResult(
                    test_name="pytest",
                    status=TestStatus.PASSED,
                    duration=duration,
                    message=result.stdout
                )
            else:
                return TestResult(
                    test_name="pytest",
                    status=TestStatus.FAILED,
                    duration=duration,
                    message=result.stdout + result.stderr
                )
                
        except subprocess.TimeoutExpired:
            return TestResult(
                test_name="pytest",
                status=TestStatus.ERROR,
                duration=time.time() - start_time,
                message="测试超时"
            )
        except Exception as e:
            return TestResult(
                test_name="pytest",
                status=TestStatus.ERROR,
                duration=time.time() - start_time,
                message=str(e)
            )
    
    async def _run_javascript_tests(self, code: str) -> TestResult:
        """运行 JavaScript 测试"""
        # 在实际实现中，这里会运行 jest 或 mocha
        return TestResult(
            test_name="jest",
            status=TestStatus.PASSED,
            message="JavaScript 测试通过"
        )
    
    async def _run_typescript_tests(self, code: str) -> TestResult:
        """运行 TypeScript 测试"""
        return await self._run_javascript_tests(code)
    
    async def _run_existing_tests(self, file_path: str) -> list[TestResult]:
        """运行现有测试"""
        results = []
        
        # 查找相关测试文件
        test_patterns = [
            f"test_{Path(file_path).stem}.py",
            f"{Path(file_path).stem}_test.py",
            f"{Path(file_path).stem}.test.js",
            f"{Path(file_path).stem}.test.ts",
        ]
        
        # 在实际实现中，这里会搜索文件系统并运行测试
        return results

class TestGenerator:
    """测试生成器"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    async def generate(self, code: GeneratedCode,
                      requirement: Requirement) -> str:
        """生成测试代码"""
        if self.llm_client:
            return await self._generate_with_llm(code, requirement)
        
        return self._generate_template(code, requirement)
    
    def _generate_template(self, code: GeneratedCode,
                          requirement: Requirement) -> str:
        """生成模板化测试"""
        if code.language == "python":
            return self._generate_python_test(code, requirement)
        elif code.language in ["javascript", "typescript"]:
            return self._generate_javascript_test(code, requirement)
        return ""
    
    def _generate_python_test(self, code: GeneratedCode,
                             requirement: Requirement) -> str:
        """生成 Python 测试"""
        # 从代码中提取函数名
        func_names = re.findall(r"def\s+(\w+)\s*\(", code.content)
        
        if not func_names:
            return self._generate_basic_python_test(code)
        
        test_code = '''import pytest
'''
        
        for func_name in func_names:
            if func_name.startswith("_"):
                continue
            
            test_code += f'''
class Test{func_name.title()}:
    """测试 {func_name}"""
    
    def test_{func_name}_success(self):
        """测试正常情况"""
        # Arrange
        # TODO: 设置测试数据
        
        # Act
        # result = {func_name}(...)
        
        # Assert
        # assert result == expected
        pass
    
    def test_{func_name}_edge_case(self):
        """测试边界情况"""
        # TODO: 测试边界条件
        pass
    
    def test_{func_name}_error(self):
        """测试错误情况"""
        # TODO: 测试错误处理
        pass
'''
        
        return test_code
    
    def _generate_basic_python_test(self, code: GeneratedCode) -> str:
        """生成基础 Python 测试"""
        return '''import pytest


def test_placeholder():
    """占位测试"""
    assert True
'''
    
    def _generate_javascript_test(self, code: GeneratedCode,
                                 requirement: Requirement) -> str:
        """生成 JavaScript 测试"""
        return '''describe("Generated Code Tests", () => {
    test("placeholder test", () => {
        expect(true).toBe(true);
    });
});
'''
    
    async def _generate_with_llm(self, code: GeneratedCode,
                                requirement: Requirement) -> str:
        """使用 LLM 生成测试"""
        # 在实际实现中，这里会调用 LLM API
        return self._generate_template(code, requirement)

# 使用示例
test_generator = TestGenerator()
test_runner = TestRunner()

# 生成测试代码
test_code = asyncio.run(test_generator.generate(generated_codes[0], req))
print(f"生成测试:\n{test_code[:500]}...")

# 运行测试
suite = asyncio.run(test_runner.run(generated_codes[0], test_code))
print(f"测试结果: {suite.passed}/{suite.total} 通过")
```

### 22.2.7 修复阶段（Fixing）

修复阶段根据测试结果修复代码问题。

```python
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class BugReport:
    """Bug 报告"""
    test_name: str
    error_message: str
    stack_trace: str = ""
    file_path: str = ""
    line_number: int = 0
    severity: str = "medium"

@dataclass
class Fix:
    """修复方案"""
    bug_report: BugReport
    description: str
    changes: list[CodeChange] = field(default_factory=list)
    confidence: float = 0.0
    risk_level: str = "low"

class BugAnalyzer:
    """Bug 分析器"""
    
    def __init__(self):
        self.error_patterns = self._init_error_patterns()
    
    def _init_error_patterns(self) -> dict[str, str]:
        """初始化错误模式"""
        return {
            "ImportError": "导入错误，检查模块路径",
            "ModuleNotFoundError": "模块未找到，检查依赖安装",
            "AttributeError": "属性错误，检查对象属性",
            "TypeError": "类型错误，检查参数类型",
            "ValueError": "值错误，检查参数值",
            "KeyError": "键错误，检查字典键",
            "IndexError": "索引错误，检查列表索引",
            "FileNotFoundError": "文件未找到，检查文件路径",
            "PermissionError": "权限错误，检查文件权限",
            "ConnectionError": "连接错误，检查网络连接",
            "TimeoutError": "超时错误，检查超时设置",
            "SyntaxError": "语法错误，检查代码语法",
            "IndentationError": "缩进错误，检查代码缩进",
        }
    
    def analyze(self, test_results: TestSuite,
               generated_code: GeneratedCode) -> list[BugReport]:
        """分析测试结果，生成 Bug 报告"""
        bug_reports = []
        
        for result in test_results.results:
            if result.status == TestStatus.FAILED:
                bug_report = self._create_bug_report(result, generated_code)
                bug_reports.append(bug_report)
            elif result.status == TestStatus.ERROR:
                bug_report = self._create_error_report(result, generated_code)
                bug_reports.append(bug_report)
        
        return bug_reports
    
    def _create_bug_report(self, result: TestResult,
                          generated_code: GeneratedCode) -> BugReport:
        """创建 Bug 报告"""
        # 解析错误信息
        error_type = self._extract_error_type(result.message)
        line_number = self._extract_line_number(result.message)
        
        return BugReport(
            test_name=result.test_name,
            error_message=result.message,
            file_path=generated_code.file_path,
            line_number=line_number,
            severity="medium"
        )
    
    def _create_error_report(self, result: TestResult,
                           generated_code: GeneratedCode) -> BugReport:
        """创建错误报告"""
        return BugReport(
            test_name=result.test_name,
            error_message=result.message,
            file_path=generated_code.file_path,
            severity="high"
        )
    
    def _extract_error_type(self, message: str) -> str:
        """提取错误类型"""
        for error_type in self.error_patterns:
            if error_type in message:
                return error_type
        return "UnknownError"
    
    def _extract_line_number(self, message: str) -> int:
        """提取行号"""
        # 尝试匹配常见的行号模式
        patterns = [
            r"line\s+(\d+)",
            r"Line\s+(\d+)",
            r":(\d+):\d+",
            r"\((\d+)\)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                return int(match.group(1))
        
        return 0

class CodeFixer:
    """代码修复器"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.fix_strategies = self._init_fix_strategies()
    
    def _init_fix_strategies(self) -> dict[str, callable]:
        """初始化修复策略"""
        return {
            "ImportError": self._fix_import_error,
            "ModuleNotFoundError": self._fix_module_not_found,
            "AttributeError": self._fix_attribute_error,
            "TypeError": self._fix_type_error,
            "ValueError": self._fix_value_error,
            "KeyError": self._fix_key_error,
            "SyntaxError": self._fix_syntax_error,
        }
    
    async def fix(self, bug_reports: list[BugReport],
                 generated_code: GeneratedCode) -> tuple[GeneratedCode, list[Fix]]:
        """修复代码"""
        fixes = []
        current_code = generated_code
        
        for bug_report in bug_reports:
            fix = await self._apply_fix(bug_report, current_code)
            if fix:
                fixes.append(fix)
                # 应用修复
                current_code = self._apply_changes(current_code, fix.changes)
        
        return current_code, fixes
    
    async def _apply_fix(self, bug_report: BugReport,
                        code: GeneratedCode) -> Optional[Fix]:
        """应用单个修复"""
        # 获取错误类型
        error_type = self._identify_error_type(bug_report.error_message)
        
        # 选择修复策略
        strategy = self.fix_strategies.get(error_type)
        
        if strategy:
            return strategy(bug_report, code)
        
        # 使用 LLM 修复
        if self.llm_client:
            return await self._fix_with_llm(bug_report, code)
        
        return None
    
    def _identify_error_type(self, error_message: str) -> str:
        """识别错误类型"""
        analyzer = BugAnalyzer()
        return analyzer._extract_error_type(error_message)
    
    def _fix_import_error(self, bug_report: BugReport,
                         code: GeneratedCode) -> Fix:
        """修复导入错误"""
        # 解析错误信息
        match = re.search(r"cannot import name '(\w+)' from '([^']+)'", 
                         bug_report.error_message)
        
        if match:
            name = match.group(1)
            module = match.group(2)
            
            # 生成修复
            new_import = f"from {module} import {name}"
            
            return Fix(
                bug_report=bug_report,
                description=f"修复导入错误: {name}",
                changes=[
                    CodeChange(
                        file_path=code.file_path,
                        change_type="add",
                        new_code=new_import,
                        description=f"添加正确的导入语句"
                    )
                ],
                confidence=0.8,
                risk_level="low"
            )
        
        return Fix(
            bug_report=bug_report,
            description="修复导入错误",
            changes=[],
            confidence=0.5,
            risk_level="medium"
        )
    
    def _fix_module_not_found(self, bug_report: BugReport,
                             code: GeneratedCode) -> Fix:
        """修复模块未找到错误"""
        return Fix(
            bug_report=bug_report,
            description="安装缺失的模块",
            changes=[
                CodeChange(
                    file_path="requirements.txt",
                    change_type="add",
                    new_code="# 需要安装缺失的模块\n# pip install <module_name>",
                    description="添加依赖说明"
                )
            ],
            confidence=0.9,
            risk_level="low"
        )
    
    def _fix_attribute_error(self, bug_report: BugReport,
                            code: GeneratedCode) -> Fix:
        """修复属性错误"""
        return Fix(
            bug_report=bug_report,
            description="修复属性访问错误",
            changes=[],
            confidence=0.6,
            risk_level="medium"
        )
    
    def _fix_type_error(self, bug_report: BugReport,
                       code: GeneratedCode) -> Fix:
        """修复类型错误"""
        return Fix(
            bug_report=bug_report,
            description="修复类型错误",
            changes=[],
            confidence=0.6,
            risk_level="medium"
        )
    
    def _fix_value_error(self, bug_report: BugReport,
                        code: GeneratedCode) -> Fix:
        """修复值错误"""
        return Fix(
            bug_report=bug_report,
            description="修复值错误",
            changes=[],
            confidence=0.6,
            risk_level="medium"
        )
    
    def _fix_key_error(self, bug_report: BugReport,
                      code: GeneratedCode) -> Fix:
        """修复键错误"""
        return Fix(
            bug_report=bug_report,
            description="修复字典键错误",
            changes=[],
            confidence=0.6,
            risk_level="low"
        )
    
    def _fix_syntax_error(self, bug_report: BugReport,
                         code: GeneratedCode) -> Fix:
        """修复语法错误"""
        return Fix(
            bug_report=bug_report,
            description="修复语法错误",
            changes=[],
            confidence=0.7,
            risk_level="low"
        )
    
    async def _fix_with_llm(self, bug_report: BugReport,
                           code: GeneratedCode) -> Optional[Fix]:
        """使用 LLM 修复"""
        # 在实际实现中，这里会调用 LLM API
        return Fix(
            bug_report=bug_report,
            description="LLM 建议的修复",
            changes=[],
            confidence=0.7,
            risk_level="medium"
        )
    
    def _apply_changes(self, code: GeneratedCode,
                      changes: list[CodeChange]) -> GeneratedCode:
        """应用代码变更"""
        new_content = code.content
        
        for change in changes:
            if change.change_type == "add":
                new_content = new_content + "\n" + change.new_code
            elif change.change_type == "modify" and change.old_code:
                new_content = new_content.replace(change.old_code, change.new_code)
        
        return GeneratedCode(
            file_path=code.file_path,
            content=new_content,
            language=code.language,
            line_count=new_content.count("\n") + 1,
            description=code.description
        )

# 使用示例
bug_analyzer = BugAnalyzer()
code_fixer = CodeFixer()

# 分析 Bug
bug_reports = bug_analyzer.analyze(suite, generated_codes[0])
print(f"发现 {len(bug_reports)} 个问题")

# 修复代码
if bug_reports:
    fixed_code, fixes = asyncio.run(
        code_fixer.fix(bug_reports, generated_codes[0])
    )
    print(f"应用了 {len(fixes)} 个修复")
```

## 22.3 SWE-bench 评测基准

### 22.3.1 SWE-bench 概述

SWE-bench 是一个用于评估 AI 系统解决真实世界 GitHub Issues 能力的基准测试集。它由普林斯顿大学的研究团队创建，包含来自 12 个流行 Python 仓库的真实问题。

**SWE-bench 核心特点**：

| 特点 | 说明 |
|------|------|
| 真实性 | 基于真实 GitHub Issues |
| 复杂性 | 需要理解代码库上下文 |
| 多样性 | 覆盖 bug 修复、功能添加、重构等 |
| 可验证性 | 自动化测试验证修复正确性 |

### 22.3.2 SWE-bench 数据集结构

```python
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class SWEBenchInstance:
    """SWE-bench 实例"""
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str = ""
    patch: str = ""  # 人工编写的参考补丁
    test_patch: str = ""  # 测试补丁
    version: str = ""
    fail_to_pass: str = ""  # 修复前失败的测试
    pass_to_pass: str = ""  # 修复前已通过的测试
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "hints_text": self.hints_text,
            "patch": self.patch,
            "test_patch": self.test_patch,
            "version": self.version,
            "fail_to_pass": self.fail_to_pass,
            "pass_to_pass": self.pass_to_pass
        }

class SWEBenchDataset:
    """SWE-bench 数据集"""
    
    def __init__(self):
        self.instances: list[SWEBenchInstance] = []
        self.repos: dict[str, list[SWEBenchInstance]] = {}
    
    def load(self, file_path: str):
        """加载数据集"""
        import json
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        for item in data:
            instance = SWEBenchInstance(
                instance_id=item["instance_id"],
                repo=item["repo"],
                base_commit=item["base_commit"],
                problem_statement=item["problem_statement"],
                hints_text=item.get("hints_text", ""),
                patch=item.get("patch", ""),
                test_patch=item.get("test_patch", ""),
                version=item.get("version", ""),
                fail_to_pass=item.get("fail_to_pass", ""),
                pass_to_pass=item.get("pass_to_pass", "")
            )
            self.instances.append(instance)
            
            # 按仓库分组
            if instance.repo not in self.repos:
                self.repos[instance.repo] = []
            self.repos[instance.repo].append(instance)
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            "total_instances": len(self.instances),
            "total_repos": len(self.repos),
            "repos_distribution": {
                repo: len(instances) 
                for repo, instances in self.repos.items()
            }
        }
    
    def filter_by_repo(self, repo: str) -> list[SWEBenchInstance]:
        """按仓库过滤"""
        return self.repos.get(repo, [])
    
    def filter_by_difficulty(self, max_files: int = None) -> list[SWEBenchInstance]:
        """按难度过滤"""
        # 基于补丁大小估计难度
        if max_files is None:
            return self.instances
        
        filtered = []
        for instance in self.instances:
            # 简单估算：补丁行数
            patch_lines = len(instance.patch.split("\n"))
            if patch_lines <= max_files * 50:  # 假设每个文件平均50行
                filtered.append(instance)
        
        return filtered

# 使用示例
dataset = SWEBenchDataset()
# dataset.load("swe-bench.json")

# 获取统计信息
# stats = dataset.get_statistics()
# print(f"总实例数: {stats['total_instances']}")
# print(f"仓库数: {stats['total_repos']}")
```

### 22.3.3 SWE-bench 排行榜

SWE-bench 排行榜展示了各种 AI 系统在解决真实 GitHub Issues 方面的表现。

| 排名 | 系统 | 开发者 | 解决率 | 方法 |
|------|------|--------|--------|------|
| 1 | Agentless | OpenAI | 55.4% | 无 Agent 方法 |
| 2 | SWE-Agent | Princeton | 45.0% | Agent + 工具 |
| 3 | Aider | Aider | 43.5% | 对话式编程 |
| 4 | AutoCodeRover | NUS | 37.2% | 程序分析 + Agent |
| 5 | CodeR | MSRA | 33.2% | 检索 + Agent |
| 6 | OpenHands | All Hands | 31.5% | 开源 Agent |
| 7 | AppMap Navie | AppMap | 28.9% | 索引 + Agent |
| 8 | AgentMoat | AgentMoat | 25.4% | 多 Agent |

> **注意**：排行榜数据会随时间更新，以上数据为示例。

### 22.3.4 SWE-bench 评测实现

```python
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

class SWEBenchEvaluator:
    """SWE-bench 评测器"""
    
    def __init__(self, dataset_path: str):
        self.dataset = SWEBenchDataset()
        self.dataset.load(dataset_path)
        self.results: list[dict] = []
    
    async def evaluate(self, agent_fn, instance_id: str = None) -> dict:
        """评测 Agent"""
        instances = self.dataset.instances
        
        if instance_id:
            instances = [i for i in instances if i.instance_id == instance_id]
        
        total = len(instances)
        resolved = 0
        
        for instance in instances:
            print(f"评测 {instance.instance_id}...")
            
            try:
                success = await self._evaluate_instance(agent_fn, instance)
                if success:
                    resolved += 1
                
                self.results.append({
                    "instance_id": instance.instance_id,
                    "success": success,
                    "repo": instance.repo
                })
            except Exception as e:
                print(f"评测失败: {e}")
                self.results.append({
                    "instance_id": instance.instance_id,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "total": total,
            "resolved": resolved,
            "resolve_rate": resolved / max(total, 1)
        }
    
    async def _evaluate_instance(self, agent_fn, 
                                instance: SWEBenchInstance) -> bool:
        """评测单个实例"""
        # 1. 克隆仓库并切换到指定提交
        repo_dir = await self._setup_repository(instance)
        
        if not repo_dir:
            return False
        
        try:
            # 2. 让 Agent 解决问题
            patch = await agent_fn(
                problem_statement=instance.problem_statement,
                repo_path=repo_dir,
                hints=instance.hints_text
            )
            
            if not patch:
                return False
            
            # 3. 应用 Agent 生成的补丁
            success = await self._apply_patch(repo_dir, patch)
            
            if not success:
                return False
            
            # 4. 应用测试补丁
            if instance.test_patch:
                await self._apply_patch(repo_dir, instance.test_patch)
            
            # 5. 运行测试
            return await self._run_tests(repo_dir, instance)
            
        finally:
            # 清理
            import shutil
            shutil.rmtree(repo_dir, ignore_errors=True)
    
    async def _setup_repository(self, instance: SWEBenchInstance) -> Optional[str]:
        """设置仓库环境"""
        try:
            # 创建临时目录
            repo_dir = tempfile.mkdtemp()
            
            # 克隆仓库
            repo_url = f"https://github.com/{instance.repo}.git"
            subprocess.run(
                ["git", "clone", repo_url, repo_dir],
                check=True,
                capture_output=True,
                timeout=120
            )
            
            # 切换到指定提交
            subprocess.run(
                ["git", "checkout", instance.base_commit],
                cwd=repo_dir,
                check=True,
                capture_output=True
            )
            
            return repo_dir
            
        except Exception as e:
            print(f"设置仓库失败: {e}")
            return None
    
    async def _apply_patch(self, repo_dir: str, patch: str) -> bool:
        """应用补丁"""
        try:
            # 写入补丁文件
            patch_file = os.path.join(repo_dir, "fix.patch")
            with open(patch_file, "w") as f:
                f.write(patch)
            
            # 应用补丁
            result = subprocess.run(
                ["git", "apply", "fix.patch"],
                cwd=repo_dir,
                capture_output=True,
                text=True
            )
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"应用补丁失败: {e}")
            return False
    
    async def _run_tests(self, repo_dir: str, 
                        instance: SWEBenchInstance) -> bool:
        """运行测试"""
        try:
            # 解析需要运行的测试
            fail_to_pass = instance.fail_to_pass.split(",")
            
            # 运行测试
            result = subprocess.run(
                ["python", "-m", "pytest", "-x", "--tb=short"] + fail_to_pass,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"运行测试失败: {e}")
            return False
    
    def get_report(self) -> dict:
        """生成评测报告"""
        total = len(self.results)
        resolved = sum(1 for r in self.results if r.get("success"))
        
        # 按仓库统计
        by_repo = {}
        for r in self.results:
            repo = r.get("repo", "unknown")
            if repo not in by_repo:
                by_repo[repo] = {"total": 0, "resolved": 0}
            by_repo[repo]["total"] += 1
            if r.get("success"):
                by_repo[repo]["resolved"] += 1
        
        return {
            "total": total,
            "resolved": resolved,
            "resolve_rate": resolved / max(total, 1),
            "by_repo": by_repo,
            "details": self.results
        }

# 使用示例
async def my_agent(problem_statement: str, repo_path: str, 
                   hints: str) -> str:
    """我的代码生成 Agent"""
    # 在实际实现中，这里会调用 LLM 生成补丁
    # 这里返回空补丁作为示例
    return ""

# evaluator = SWEBenchEvaluator("swe-bench.json")
# report = await evaluator.evaluate(my_agent)
# print(f"解决率: {report['resolve_rate']:.1%}")
```

### 22.3.5 SWE-bench Lite

SWE-bench Lite 是 SWE-bench 的精简版本，包含 300 个精选实例，更适合快速评测。

```python
class SWEBenchLite:
    """SWE-bench Lite 评测集"""
    
    # 精选的 300 个实例 ID
    SELECTED_INSTANCES = [
        # django
        "django__django-11099",
        "django__django-11049",
        "django__django-11583",
        # flask
        "pallets__flask-4992",
        "pallets__flask-5063",
        # sympy
        "sympy__sympy-20590",
        "sympy__sympy-21050",
        # scikit-learn
        "scikit-learn__scikit-learn-13496",
        # ... 更多实例
    ]
    
    # 按难度分类
    DIFFICULTY_LEVELS = {
        "easy": [],      # 修改行数 < 20
        "medium": [],    # 修改行数 20-100
        "hard": [],      # 修改行数 > 100
    }
    
    @classmethod
    def get_instances(cls, difficulty: str = None) -> list[str]:
        """获取实例列表"""
        if difficulty:
            return cls.DIFFICULTY_LEVELS.get(difficulty, [])
        return cls.SELECTED_INSTANCES
    
    @classmethod
    def get_statistics(cls) -> dict:
        """获取统计信息"""
        return {
            "total": len(cls.SELECTED_INSTANCES),
            "by_difficulty": {
                k: len(v) for k, v in cls.DIFFICULTY_LEVELS.items()
            }
        }
```

## 22.4 挑战与应对

### 22.4.1 核心挑战

| 挑战 | 说明 | 影响 |
|------|------|------|
| 代码理解 | 理解大型代码库的结构和逻辑 | 无法正确定位问题 |
| 上下文限制 | LLM 上下文窗口有限 | 遗漏关键信息 |
| 幻觉问题 | 生成不存在的函数或API | 代码无法运行 |
| 一致性 | 生成代码与现有风格不一致 | 可维护性差 |
| 复杂逻辑 | 处理复杂的业务逻辑 | 逻辑错误 |
| 依赖管理 | 处理外部依赖和版本 | 兼容性问题 |
| 测试覆盖 | 生成充分的测试用例 | 验证不充分 |

### 22.4.2 应对策略

```python
class ChallengeMitigator:
    """挑战应对策略"""
    
    def __init__(self):
        self.strategies = {
            "code_understanding": self._mitigate_code_understanding,
            "context_limit": self._mitigate_context_limit,
            "hallucination": self._mitigate_hallucination,
            "consistency": self._mitigate_consistency,
            "complex_logic": self._mitigate_complex_logic,
            "dependency": self._mitigate_dependency,
            "test_coverage": self._mitigate_test_coverage,
        }
    
    def _mitigate_code_understanding(self) -> dict:
        """应对代码理解挑战"""
        return {
            "strategy": "多层代码分析",
            "techniques": [
                "AST 解析提取代码结构",
                "类型推断理解数据流",
                "调用图分析追踪依赖",
                "注释和文档语义理解",
            ],
            "implementation": """
            # 使用 AST 分析代码结构
            import ast
            
            class CodeAnalyzer(ast.NodeVisitor):
                def __init__(self):
                    self.functions = []
                    self.classes = []
                    self.imports = []
                
                def visit_FunctionDef(self, node):
                    self.functions.append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "decorators": [d.id for d in node.decorator_list if hasattr(d, 'id')]
                    })
                    self.generic_visit(node)
                
                def visit_ClassDef(self, node):
                    self.classes.append({
                        "name": node.name,
                        "bases": [b.id if hasattr(b, 'id') else str(b) for b in node.bases]
                    })
                    self.generic_visit(node)
            """
        }
    
    def _mitigate_context_limit(self) -> dict:
        """应对上下文限制挑战"""
        return {
            "strategy": "智能上下文管理",
            "techniques": [
                "检索相关代码片段",
                "摘要化无关代码",
                "分层上下文（全局/文件/函数）",
                "动态上下文窗口调整",
            ],
            "implementation": """
            class ContextManager:
                def __init__(self, max_tokens=4000):
                    self.max_tokens = max_tokens
                    self.context_layers = {
                        "global": [],      # 全局上下文
                        "file": [],        # 文件级上下文
                        "function": [],    # 函数级上下文
                    }
                
                def build_context(self, requirement, code_files):
                    context = []
                    current_tokens = 0
                    
                    # 1. 添加全局上下文（项目结构）
                    global_context = self._build_global_context(code_files)
                    context.append(global_context)
                    current_tokens += self._estimate_tokens(global_context)
                    
                    # 2. 添加相关文件上下文
                    relevant_files = self._find_relevant_files(requirement, code_files)
                    for file in relevant_files:
                        if current_tokens >= self.max_tokens:
                            break
                        file_context = self._build_file_context(file)
                        context.append(file_context)
                        current_tokens += self._estimate_tokens(file_context)
                    
                    return "\\n".join(context)
            """
        }
    
    def _mitigate_hallucination(self) -> dict:
        """应对幻觉挑战"""
        return {
            "策略": "验证与约束",
            "技术": [
                "代码搜索验证 API 存在性",
                "类型检查确保正确调用",
                "静态分析检测未定义变量",
                "单元测试验证功能正确性",
            ]
        }
    
    def _mitigate_consistency(self) -> dict:
        """应对一致性挑战"""
        return {
            "策略": "风格学习与强制",
            "技术": [
                "分析现有代码风格",
                "提取命名约定",
                "遵循代码格式化规则",
                "代码审查检查一致性",
            ]
        }
    
    def _mitigate_complex_logic(self) -> dict:
        """应对复杂逻辑挑战"""
        return {
            "策略": "分而治之",
            "技术": [
                "将复杂逻辑分解为小函数",
                "逐步验证每个步骤",
                "使用类型系统约束",
                "编写详细的测试用例",
            ]
        }
    
    def _mitigate_dependency(self) -> dict:
        """应对依赖挑战"""
        return {
            "策略": "依赖分析与管理",
            "技术": [
                "分析现有依赖",
                "检查版本兼容性",
                "使用虚拟环境隔离",
                "编写安装说明",
            ]
        }
    
    def _mitigate_test_coverage(self) -> dict:
        """应对测试覆盖挑战"""
        return {
            "策略": "系统化测试生成",
            "技术": [
                "基于代码结构生成测试",
                "边界条件测试",
                "异常路径测试",
                "集成测试验证",
            ]
        }

# 使用示例
mitigator = ChallengeMitigator()
for challenge, handler in mitigator.strategies.items():
    strategy = handler()
    print(f"\n挑战: {challenge}")
    print(f"策略: {strategy.get('strategy', strategy.get('策略', 'N/A'))}")
```

## 22.5 主流代码 Agent 产品原理

### 22.5.1 Cursor 原理

Cursor 是一款 AI-first 的代码编辑器，集成了强大的代码生成能力。

**Cursor 核心架构**：

| 组件 | 功能 |
|------|------|
| Tab 补全 | 基于上下文的智能代码补全 |
| Cmd+K | 内联代码生成和编辑 |
| Chat | 对话式代码生成和问答 |
| Composer | 多文件编辑和重构 |
| 符号索引 | 理解代码库结构和关系 |

**Cursor 的关键技术**：

```python
class CursorArchitecture:
    """Cursor 架构分析"""
    
    def __init__(self):
        self.components = {
            "tab_completion": TabCompletionEngine(),
            "inline_edit": InlineEditEngine(),
            "chat": ChatEngine(),
            "composer": ComposerEngine(),
            "indexing": SymbolIndexEngine(),
        }
    
    def process_request(self, request_type: str, context: dict) -> str:
        """处理用户请求"""
        engine = self.components.get(request_type)
        if engine:
            return engine.process(context)
        return ""

class TabCompletionEngine:
    """Tab 补全引擎"""
    
    def process(self, context: dict) -> str:
        """处理 Tab 补全"""
        # 1. 获取当前文件上下文
        file_content = context.get("file_content", "")
        cursor_position = context.get("cursor_position", 0)
        
        # 2. 分析代码结构
        code_context = self._analyze_code(file_content, cursor_position)
        
        # 3. 检索相关代码
        relevant_code = self._retrieve_relevant_code(code_context)
        
        # 4. 生成补全建议
        suggestion = self._generate_completion(code_context, relevant_code)
        
        return suggestion
    
    def _analyze_code(self, content: str, position: int) -> dict:
        """分析代码"""
        # 分析光标位置前的代码
        prefix = content[:position]
        suffix = content[position:]
        
        return {
            "prefix": prefix,
            "suffix": suffix,
            "current_line": prefix.split("\n")[-1],
            "indentation": len(prefix.split("\n")[-1]) - len(prefix.split("\n")[-1].lstrip()),
        }
    
    def _retrieve_relevant_code(self, code_context: dict) -> list[str]:
        """检索相关代码"""
        # 在实际实现中，这里会使用向量检索
        # 这里简化为返回空列表
        return []
    
    def _generate_completion(self, code_context: dict, 
                            relevant_code: list[str]) -> str:
        """生成补全"""
        # 在实际实现中，这里会调用 LLM
        # 这里返回简单的补全示例
        return "# TODO: Generated completion"

class InlineEditEngine:
    """内联编辑引擎（Cmd+K）"""
    
    def process(self, context: dict) -> str:
        """处理内联编辑"""
        # 1. 获取用户指令
        instruction = context.get("instruction", "")
        
        # 2. 获取选中的代码
        selected_code = context.get("selected_code", "")
        
        # 3. 分析上下文
        file_context = context.get("file_context", "")
        
        # 4. 生成编辑后的代码
        edited_code = self._generate_edit(instruction, selected_code, file_context)
        
        return edited_code
    
    def _generate_edit(self, instruction: str, code: str,
                      context: str) -> str:
        """生成编辑后的代码"""
        # 在实际实现中，这里会调用 LLM
        # 这里返回简单的示例
        return f"# Edited based on: {instruction}\n{code}"

class ChatEngine:
    """对话引擎"""
    
    def process(self, context: dict) -> str:
        """处理对话"""
        messages = context.get("messages", [])
        codebase_context = context.get("codebase_context", "")
        
        # 在实际实现中，这里会调用 LLM
        return "Chat response based on context..."

class ComposerEngine:
    """Composer 引擎（多文件编辑）"""
    
    def process(self, context: dict) -> dict:
        """处理多文件编辑"""
        instruction = context.get("instruction", "")
        files = context.get("files", [])
        
        # 1. 分析所有相关文件
        file_analyses = [self._analyze_file(f) for f in files]
        
        # 2. 生成编辑计划
        edit_plan = self._generate_edit_plan(instruction, file_analyses)
        
        # 3. 生成每个文件的编辑
        file_edits = {}
        for file_path, edit in edit_plan.items():
            file_edits[file_path] = self._generate_file_edit(edit)
        
        return file_edits
    
    def _analyze_file(self, file_info: dict) -> dict:
        """分析文件"""
        return {
            "path": file_info.get("path"),
            "content": file_info.get("content"),
            "language": file_info.get("language"),
        }
    
    def _generate_edit_plan(self, instruction: str,
                           file_analyses: list[dict]) -> dict:
        """生成编辑计划"""
        # 在实际实现中，这里会调用 LLM
        return {}
    
    def _generate_file_edit(self, edit: dict) -> str:
        """生成文件编辑"""
        return ""

class SymbolIndexEngine:
    """符号索引引擎"""
    
    def __init__(self):
        self.index: dict[str, Any] = {}
    
    def build_index(self, repo_path: str):
        """构建代码索引"""
        # 1. 遍历代码文件
        for file_path in Path(repo_path).rglob("*"):
            if file_path.is_file() and self._is_code_file(file_path):
                self._index_file(file_path)
        
        # 2. 建立符号关系
        self._build_relationships()
    
    def _is_code_file(self, path: Path) -> bool:
        """检查是否为代码文件"""
        code_extensions = {".py", ".js", ".ts", ".tsx", ".jsx"}
        return path.suffix.lower() in code_extensions
    
    def _index_file(self, file_path: Path):
        """索引文件"""
        # 提取符号（函数、类、变量等）
        content = file_path.read_text()
        symbols = self._extract_symbols(content, file_path.suffix)
        
        self.index[str(file_path)] = {
            "symbols": symbols,
            "imports": self._extract_imports(content, file_path.suffix),
        }
    
    def _extract_symbols(self, content: str, ext: str) -> list[dict]:
        """提取符号"""
        symbols = []
        
        if ext == ".py":
            # Python 符号
            pattern = r"(?:def|class|async\s+def)\s+(\w+)"
            for match in re.finditer(pattern, content):
                symbols.append({
                    "name": match.group(1),
                    "type": "function" if "def" in content[match.start():match.end()] else "class",
                    "line": content[:match.start()].count("\n") + 1
                })
        
        return symbols
    
    def _extract_imports(self, content: str, ext: str) -> list[str]:
        """提取导入"""
        imports = []
        
        if ext == ".py":
            pattern = r"(?:from\s+(\S+)\s+import|import\s+(\S+))"
            for match in re.finditer(pattern, content):
                imports.append(match.group(1) or match.group(2))
        
        return imports
    
    def _build_relationships(self):
        """建立符号关系"""
        # 在实际实现中，这里会建立调用关系、继承关系等
        pass

# Cursor 使用示例
cursor = CursorArchitecture()

# Tab 补全
completion = cursor.process_request("tab_completion", {
    "file_content": "def calculate_sum(numbers):\n    ",
    "cursor_position": 30
})
print(f"补全建议: {completion}")
```

### 22.5.2 Devin 原理

Devin 是全球第一个 AI 软件工程师，能够独立完成复杂的开发任务。

**Devin 核心架构**：

| 组件 | 功能 |
|------|------|
| 规划器 | 分解任务、制定执行计划 |
| 编辑器 | 编写和修改代码 |
| 终端 | 执行命令、运行测试 |
| 浏览器 | 查阅文档、搜索信息 |
| 记忆 | 长期记忆和上下文管理 |

**Devin 的工作流程**：

```python
class DevinArchitecture:
    """Devin 架构分析"""
    
    def __init__(self):
        self.planner = TaskPlanner()
        self.editor = CodeEditor()
        self.terminal = Terminal()
        self.browser = Browser()
        self.memory = MemoryManager()
    
    async def execute_task(self, task: str) -> dict:
        """执行任务"""
        # 1. 规划阶段
        plan = await self.planner.create_plan(task)
        
        # 2. 执行阶段
        results = []
        for step in plan.steps:
            result = await self._execute_step(step)
            results.append(result)
            
            # 记录执行结果
            self.memory.store(step, result)
        
        # 3. 验证阶段
        verification = await self._verify_results(results)
        
        return {
            "plan": plan,
            "results": results,
            "verification": verification
        }
    
    async def _execute_step(self, step: dict) -> dict:
        """执行单个步骤"""
        action = step.get("action")
        
        if action == "write_code":
            return await self._write_code(step)
        elif action == "run_command":
            return await self._run_command(step)
        elif action == "search_docs":
            return await self._search_docs(step)
        elif action == "run_tests":
            return await self._run_tests(step)
        
        return {"status": "unknown_action"}
    
    async def _write_code(self, step: dict) -> dict:
        """编写代码"""
        file_path = step.get("file_path")
        content = step.get("content")
        
        # 使用编辑器写入代码
        result = await self.editor.write(file_path, content)
        
        return {"status": "success", "file": file_path}
    
    async def _run_command(self, step: dict) -> dict:
        """运行命令"""
        command = step.get("command")
        
        # 使用终端执行命令
        result = await self.terminal.execute(command)
        
        return {
            "status": "success" if result.exit_code == 0 else "failed",
            "output": result.output,
            "error": result.error
        }
    
    async def _search_docs(self, step: dict) -> dict:
        """搜索文档"""
        query = step.get("query")
        
        # 使用浏览器搜索文档
        results = await self.browser.search(query)
        
        return {"status": "success", "results": results}
    
    async def _run_tests(self, step: dict) -> dict:
        """运行测试"""
        test_path = step.get("test_path")
        
        # 运行测试
        result = await self.terminal.execute(f"pytest {test_path} -v")
        
        return {
            "status": "success" if result.exit_code == 0 else "failed",
            "output": result.output,
            "passed": result.exit_code == 0
        }
    
    async def _verify_results(self, results: list[dict]) -> dict:
        """验证结果"""
        all_passed = all(r.get("status") == "success" for r in results)
        
        return {
            "all_passed": all_passed,
            "summary": f"执行了 {len(results)} 个步骤"
        }

class TaskPlanner:
    """任务规划器"""
    
    async def create_plan(self, task: str) -> dict:
        """创建执行计划"""
        # 在实际实现中，这里会调用 LLM 进行规划
        return {
            "task": task,
            "steps": [
                {"action": "search_docs", "query": "相关文档"},
                {"action": "write_code", "file_path": "main.py", "content": "# TODO"},
                {"action": "run_tests", "test_path": "test_main.py"},
            ]
        }

class CodeEditor:
    """代码编辑器"""
    
    async def write(self, file_path: str, content: str) -> dict:
        """写入文件"""
        try:
            with open(file_path, "w") as f:
                f.write(content)
            return {"status": "success"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

class Terminal:
    """终端"""
    
    async def execute(self, command: str) -> dict:
        """执行命令"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )
            return {
                "exit_code": result.returncode,
                "output": result.stdout,
                "error": result.stderr
            }
        except Exception as e:
            return {
                "exit_code": -1,
                "output": "",
                "error": str(e)
            }

class Browser:
    """浏览器"""
    
    async def search(self, query: str) -> list[dict]:
        """搜索"""
        # 在实际实现中，这里会使用浏览器搜索
        return []

class MemoryManager:
    """记忆管理器"""
    
    def __init__(self):
        self.memories: list[dict] = []
    
    def store(self, step: dict, result: dict):
        """存储记忆"""
        self.memories.append({
            "step": step,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

# Devin 使用示例
devin = DevinArchitecture()
# result = asyncio.run(devin.execute_task("实现一个 REST API"))
```

### 22.5.3 其他主流代码 Agent

| 产品 | 开发者 | 核心特点 |
|------|--------|---------|
| GitHub Copilot | GitHub/Microsoft | IDE 集成、代码补全、Chat |
| CodeWhisperer | AWS | 安全扫描、AWS 集成 |
| Cody | Sourcegraph | 代码搜索、上下文理解 |
| Tabnine | Tabnine | 本地模型、隐私保护 |
| Aider | Aider | 对话式编程、Git 集成 |
| OpenHands | All Hands | 开源、可扩展 |

## 22.6 本章小结

本章深入探讨了代码生成 Agent 的核心概念和实现：

1. **核心架构**：代码 Agent 的六阶段流程（理解→定位→设计→生成→测试→修复）
2. **理解阶段**：需求分析、类型识别、约束提取
3. **定位阶段**：代码索引、相关性排序、上下文构建
4. **设计阶段**：方案设计、变更规划、风险评估
5. **生成阶段**：模板生成、LLM 生成、代码格式化
6. **测试阶段**：测试生成、自动化运行、结果分析
7. **修复阶段**：Bug 分析、策略选择、代码修复
8. **SWE-bench**：真实世界评测基准、排行榜、评测实现
9. **挑战与应对**：代码理解、上下文限制、幻觉问题等
10. **主流产品**：Cursor、Devin 等产品的架构和原理

## 22.7 思考题

1. 如何设计一个支持多种编程语言的代码生成 Agent？需要考虑哪些语言特性？
2. SWE-bench 评测有哪些局限性？如何设计更好的评测基准？
3. 如何平衡代码生成的质量和速度？在实际应用中如何权衡？
4. Cursor 的 Tab 补全如何实现低延迟？需要哪些技术优化？
5. Devin 的自主执行模式存在哪些风险？如何设计安全机制？
6. 如何评估代码生成 Agent 生成代码的质量？需要哪些指标？
7. 代码生成 Agent 如何处理遗留代码？需要哪些特殊策略？
