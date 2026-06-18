---
title: "第16章：AutoGen 多智能体框架"
description: "深入 Microsoft AutoGen 框架：ConversableAgent、Agent 会话协议、代码执行器、GroupChat 与嵌套对话。"
date: "2026-06-11"
---

 # 第16章：AutoGen 多智能体框架

 AutoGen 是 Microsoft Research 推出的开源多 Agent 对话框架，通过让多个 AI Agent 进行对话协作来解决复杂问题。本章深入讲解 AutoGen 的核心架构、对话机制和生产实践。

 下面的交互式演示展示了多 Agent 对话的过程：

 <div data-component="AutoGenConversation"></div>

 ## 什么是 AutoGen？

AutoGen 的核心理念是**对话驱动**。它将多 Agent 协作模拟为一场对话，Agent 之间通过消息传递来协调工作。

**AutoGen 的独特之处**：

1. **对话即协作**：Agent 之间通过自然语言对话来协作，就像人类团队开会讨论一样。

2. **灵活的控制流**：支持自动模式（Agent 自主决定何时停止）和手动模式（人类随时介入）。

3. **代码执行能力**：内置代码执行器，Agent 可以编写和运行代码来解决问题。

4. **可组合性**：可以轻松组合不同的 Agent 来创建复杂的系统。

## AutoGen 的核心概念

**ConversableAgent**：AutoGen 的核心类，所有 Agent 都继承自它。它封装了：
- LLM 调用
- 消息管理
- 工具执行
- 对话历史

**GroupChat**：多 Agent 群聊机制，允许多个 Agent 在同一个对话中交流。

**GroupChatManager**：群聊管理器，负责协调群聊中的发言顺序和对话流程。

**UserProxyAgent**：用户代理，代表人类用户参与对话。

## AutoGen vs 其他框架

| 特性 | AutoGen | CrewAI | LangGraph |
|:---|:---|:---|:---|
| **核心理念** | 对话驱动 | 角色扮演 | 图编排 |
| **学习曲线** | 中 | 低 | 高 |
| **代码执行** | 内置支持 | 需要自定义 | 需要自定义 |
| **适用场景** | 代码生成、分析 | 团队协作 | 复杂流程 |

**选择建议**：
- 如果你需要代码生成和执行能力，选择 AutoGen
- 如果你需要模拟团队协作，选择 CrewAI
- 如果你需要复杂的流程控制，选择 LangGraph

---

## 16.1 AutoGen 架构设计

### 16.1.1 整体架构

AutoGen 的架构围绕对话机制设计：

```
┌─────────────────────────────────────────────────────────────┐
│                    AutoGen 架构                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │Conversable │    │  GroupChat   │    │  Code       │     │
│  │  Agent     │◀──▶│  Manager     │◀──▶│  Executor   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Message   │    │   Speaker   │    │   Tool      │     │
│  │   Protocol  │    │  Selection  │    │  Registry   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│                    ┌─────────────┐                          │
│                    │    LLM      │                          │
│                    │  Interface  │                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

**ConversableAgent**：AutoGen 的核心组件，所有 Agent 都继承自它。它封装了 LLM 调用、消息管理、工具执行等核心功能。

**GroupChat**：多 Agent 群聊机制，允许多个 Agent 在同一个对话中交流。群聊支持多种发言者选择策略。

**GroupChatManager**：群聊管理器，负责协调群聊中的发言顺序和对话流程。

**Code Executor**：代码执行器，支持在 Docker 沙箱中安全执行代码。

**Message Protocol**：消息协议，定义了 Agent 之间通信的标准格式。

**Speaker Selection**：发言者选择机制，决定下一个发言的 Agent。

**Tool Registry**：工具注册表，管理 Agent 可以使用的工具。

```
┌─────────────────────────────────────────────────────────────┐
│                    AutoGen 架构                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │Conversable │    │  GroupChat   │    │  Code       │     │
│  │  Agent     │◀──▶│  Manager     │◀──▶│  Executor   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Message   │    │   Speaker   │    │   Tool      │     │
│  │   Protocol  │    │  Selection  │    │  Registry   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│                    ┌─────────────┐                          │
│                    │    LLM      │                          │
│                    │  Interface  │                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### 16.1.2 核心组件

```python
from autogen import (
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
    register_function
)
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class AutoGenConfig:
    """AutoGen 配置"""
    model: str = "gpt-4o"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120
    
class AutoGenManager:
    """AutoGen 管理器"""
    
    def __init__(self, config: AutoGenConfig):
        self.config = config
        self.agents: Dict[str, ConversableAgent] = {}
        self.conversations: List[Dict[str, Any]] = []
    
    def create_agent(
        self,
        name: str,
        system_message: str,
        human_input_mode: str = "NEVER",
        code_execution: bool = False,
        tools: List[Callable] = None,
        **kwargs
    ) -> ConversableAgent:
        """创建 Agent"""
        llm_config = {
            "config_list": [{
                "model": self.config.model,
                "api_key": self.config.api_key
            }],
            "temperature": self.config.temperature,
            "timeout": self.config.timeout
        }
        
        agent_config = {
            "name": name,
            "system_message": system_message,
            "llm_config": llm_config,
            "human_input_mode": human_input_mode,
            "max_consecutive_auto_reply": kwargs.get("max_replies", 10),
            "is_termination_msg": kwargs.get("termination_check", lambda x: "terminate" in x.get("content", "").lower()),
        }
        
        if code_execution:
            from autogen.coding import DockerCommandLineCodeExecutor
            import tempfile
            
            code_executor = DockerCommandLineCodeExecutor(
                image="python:3.11-slim",
                timeout=60,
                work_dir=tempfile.mkdtemp()
            )
            agent_config["code_execution_config"] = {
                "executor": code_executor,
                "work_dir": tempfile.mkdtemp()
            }
        
        agent = ConversableAgent(**agent_config)
        
        # 注册工具
        if tools:
            for tool in tools:
                register_function(
                    tool,
                    caller=agent,
                    executor=agent,
                    description=tool.__doc__ or tool.__name__
                )
        
        self.agents[name] = agent
        return agent
    
    def create_user_proxy(
        self,
        name: str = "user",
        human_input_mode: str = "ALWAYS"
    ) -> UserProxyAgent:
        """创建用户代理"""
        proxy = UserProxyAgent(
            name=name,
            human_input_mode=human_input_mode,
            code_execution_config=False,
            default_auto_reply="",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("terminate")
        )
        self.agents[name] = proxy
        return proxy
    
    async def run_conversation(
        self,
        initiator: ConversableAgent,
        recipient: ConversableAgent,
        message: str,
        max_turns: int = 10
    ) -> Dict[str, Any]:
        """运行对话"""
        result = await initiator.a_initiate_chat(
            recipient,
            message=message,
            max_turns=max_turns,
            summary_method="last_msg"
        )
        
        conversation = {
            "initiator": initiator.name,
            "recipient": recipient.name,
            "message": message,
            "chat_history": result.chat_history,
            "summary": result.summary,
            "cost": result.cost
        }
        
        self.conversations.append(conversation)
        return conversation
```

---

## 16.2 ConversableAgent 深入解析

### 16.2.1 消息处理机制

```python
class MessageProcessor:
    """消息处理器"""
    
    def __init__(self, agent: ConversableAgent):
        self.agent = agent
        self.message_queue: List[Dict[str, Any]] = []
        self.processed_messages: List[Dict[str, Any]] = []
    
    async def process_message(self, message: Dict[str, Any]) -> Optional[str]:
        """处理消息"""
        # 预处理
        processed = self._preprocess(message)
        
        # 生成回复
        reply = await self._generate_reply(processed)
        
        # 后处理
        if reply:
            reply = self._postprocess(reply)
        
        # 记录
        self.processed_messages.append({
            "input": message,
            "output": reply,
            "timestamp": time.time()
        })
        
        return reply
    
    def _preprocess(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """预处理消息"""
        processed = message.copy()
        
        # 添加上下文信息
        if self.agent.chat_messages:
            recent_messages = list(self.agent.chat_messages.values())[-3:]
            processed["context"] = recent_messages
        
        # 添加角色信息
        processed["sender_role"] = self.agent.name
        
        return processed
    
    async def _generate_reply(self, message: Dict[str, Any]) -> Optional[str]:
        """生成回复"""
        # 检查是否应该回复
        if not self._should_reply(message):
            return None
        
        # 调用 LLM 生成回复
        reply = await self.agent.a_generate_reply(message)
        
        return reply
    
    def _should_reply(self, message: Dict[str, Any]) -> bool:
        """判断是否应该回复"""
        # 检查终止条件
        if self.agent.is_termination_msg(message):
            return False
        
        # 检查回复次数限制
        if self._exceeded_reply_limit():
            return False
        
        return True
    
    def _exceeded_reply_limit(self) -> bool:
        """检查是否超过回复限制"""
        if not self.agent.max_consecutive_auto_reply:
            return False
        
        consecutive_replies = 0
        for msg in reversed(self.processed_messages):
            if msg["output"] is not None:
                consecutive_replies += 1
            else:
                break
        
        return consecutive_replies >= self.agent.max_consecutive_auto_reply
    
    def _postprocess(self, reply: str) -> str:
        """后处理回复"""
        # 添加签名
        if self.agent.name:
            reply = f"[{self.agent.name}] {reply}"
        
        return reply
```

### 16.2.2 代码执行能力

```python
from autogen.coding import (
    CodeExecutor,
    CodeBlock,
    DockerCommandLineCodeExecutor,
    LocalCommandLineCodeExecutor
)
import tempfile
import os

class EnhancedCodeExecutor:
    """增强型代码执行器"""
    
    def __init__(self, use_docker: bool = True):
        if use_docker:
            self.executor = DockerCommandLineCodeExecutor(
                image="python:3.11-slim",
                timeout=60,
                work_dir=tempfile.mkdtemp()
            )
        else:
            self.executor = LocalCommandLineCodeExecutor(
                work_dir=tempfile.mkdtemp()
            )
    
    async def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """执行代码"""
        code_block = CodeBlock(code=code, language=language)
        
        try:
            result = await self.executor.execute_code_blocks([code_block])
            
            return {
                "success": True,
                "output": result.output,
                "exit_code": result.exit_code,
                "code": code
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "code": code
            }
    
    def create_code_agent(self, name: str, system_message: str) -> ConversableAgent:
        """创建代码执行 Agent"""
        return ConversableAgent(
            name=name,
            system_message=system_message,
            llm_config={
                "config_list": [{
                    "model": "gpt-4o",
                    "api_key": os.getenv("OPENAI_API_KEY")
                }]
            },
            code_execution_config={
                "executor": self.executor,
                "work_dir": tempfile.mkdtemp(),
                "timeout": 120
            },
            human_input_mode="NEVER"
        )

class CodeReviewAgent:
    """代码审查 Agent"""
    
    def __init__(self, executor: EnhancedCodeExecutor):
        self.executor = executor
    
    async def review_and_execute(self, code: str) -> Dict[str, Any]:
        """审查并执行代码"""
        # 代码审查
        review = await self._review_code(code)
        
        if not review["approved"]:
            return {
                "success": False,
                "reason": "代码审查未通过",
                "issues": review["issues"]
            }
        
        # 执行代码
        result = await self.executor.execute_code(code)
        
        return {
            **result,
            "review": review
        }
    
    async def _review_code(self, code: str) -> Dict[str, Any]:
        """审查代码"""
        issues = []
        
        # 安全检查
        if self._has_security_issues(code):
            issues.append("检测到潜在的安全问题")
        
        # 质量检查
        quality_issues = self._check_code_quality(code)
        issues.extend(quality_issues)
        
        return {
            "approved": len(issues) == 0,
            "issues": issues,
            "score": max(0, 100 - len(issues) * 20)
        }
    
    def _has_security_issues(self, code: str) -> bool:
        """检查安全问题"""
        dangerous_patterns = [
            "import os",
            "import subprocess",
            "exec(",
            "eval(",
            "os.system(",
            "open(",
        ]
        
        return any(pattern in code for pattern in dangerous_patterns)
    
    def _check_code_quality(self, code: str) -> List[str]:
        """检查代码质量"""
        issues = []
        
        if len(code) > 1000:
            issues.append("代码过长，建议拆分")
        
        if code.count("\n") < 3:
            issues.append("代码缺少必要的注释")
        
        return issues
```

---

## 16.3 GroupChat 群聊系统

### 16.3.1 高级群聊配置

```python
class AdvancedGroupChat:
    """高级群聊配置"""
    
    def __init__(self):
        self.agents: List[ConversableAgent] = []
        self.group_chat: Optional[GroupChat] = None
        self.manager: Optional[GroupChatManager] = None
    
    def setup_team(
        self,
        team_config: Dict[str, Any],
        llm_config: Dict[str, Any]
    ):
        """设置团队"""
        # 创建 Agent
        for agent_config in team_config["agents"]:
            agent = ConversableAgent(
                name=agent_config["name"],
                system_message=agent_config["system_message"],
                llm_config=llm_config,
                human_input_mode=agent_config.get("human_input_mode", "NEVER"),
                max_consecutive_auto_reply=agent_config.get("max_replies", 5)
            )
            self.agents.append(agent)
        
        # 创建群聊
        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],
            max_round=team_config.get("max_round", 15),
            speaker_selection_method=team_config.get("selection_method", "auto"),
            speaker_selection_silent=team_config.get("silent_selection", False),
            allow_repeat_speaker=team_config.get("allow_repeat", False)
        )
        
        # 创建管理器
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=llm_config
        )
    
    async def run_discussion(
        self,
        topic: str,
        max_rounds: int = None
    ) -> Dict[str, Any]:
        """运行讨论"""
        if not self.manager:
            raise ValueError("群聊未初始化")
        
        # 启动讨论
        initiator = self.agents[0]
        result = await initiator.a_initiate_chat(
            self.manager,
            message=topic,
            max_turns=max_rounds or self.group_chat.max_round
        )
        
        return {
            "topic": topic,
            "participants": [agent.name for agent in self.agents],
            "chat_history": result.chat_history,
            "summary": result.summary,
            "rounds": len(result.chat_history)
        }

class SpeakerSelectionStrategy:
    """发言者选择策略"""
    
    @staticmethod
    def create_custom_strategy(
        agents: List[ConversableAgent],
        strategy_type: str = "round_robin"
    ) -> Callable:
        """创建自定义策略"""
        if strategy_type == "round_robin":
            return SpeakerSelectionStrategy._round_robin(agents)
        elif strategy_type == "expertise_based":
            return SpeakerSelectionStrategy._expertise_based(agents)
        elif strategy_type == "load_balanced":
            return SpeakerSelectionStrategy._load_balanced(agents)
        else:
            return SpeakerSelectionStrategy._auto
    
    @staticmethod
    def _round_robin(agents: List[ConversableAgent]) -> Callable:
        """轮询策略"""
        current_index = [0]
        
        def select_speaker(speaker, messenger, messages):
            speaker = agents[current_index[0] % len(agents)]
            current_index[0] += 1
            return speaker
        
        return select_speaker
    
    @staticmethod
    def _expertise_based(agents: List[ConversableAgent]) -> Callable:
        """基于专业能力的策略"""
        expertise_map = {}
        
        for agent in agents:
            # 从系统消息中提取专业领域
            expertise_map[agent.name] = agent.system_message.lower()
        
        def select_speaker(speaker, messenger, messages):
            last_message = messages[-1]["content"].lower()
            
            # 根据消息内容选择最合适的发言者
            best_match = None
            best_score = -1
            
            for agent in agents:
                score = sum(1 for word in expertise_map[agent.name].split() 
                           if word in last_message)
                if score > best_score:
                    best_score = score
                    best_match = agent
            
            return best_match or agents[0]
        
        return select_speaker
    
    @staticmethod
    def _load_balanced(agents: List[ConversableAgent]) -> Callable:
        """负载均衡策略"""
        speak_count = {agent.name: 0 for agent in agents}
        
        def select_speaker(speaker, messenger, messages):
            # 选择发言次数最少的 Agent
            min_speaker = min(agents, key=lambda a: speak_count[a.name])
            speak_count[min_speaker.name] += 1
            return min_speaker
        
        return select_speaker
    
    @staticmethod
    def _auto(speaker, messenger, messages):
        """自动策略（AutoGen 默认）"""
        return None  # 让 AutoGen 自动决定
```

### 16.3.2 群聊监控和管理

```python
class GroupChatMonitor:
    """群聊监控器"""
    
    def __init__(self, group_chat: GroupChat):
        self.group_chat = group_chat
        self.conversation_log: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
    
    def log_message(self, speaker: str, message: str, timestamp: float):
        """记录消息"""
        self.conversation_log.append({
            "speaker": speaker,
            "message": message,
            "timestamp": timestamp,
            "word_count": len(message.split())
        })
        
        self._update_metrics()
    
    def _update_metrics(self):
        """更新指标"""
        if not self.conversation_log:
            return
        
        # 计算每个 Agent 的发言统计
        speaker_stats = {}
        for log in self.conversation_log:
            speaker = log["speaker"]
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "message_count": 0,
                    "total_words": 0,
                    "first_speak": log["timestamp"],
                    "last_speak": log["timestamp"]
                }
            
            speaker_stats[speaker]["message_count"] += 1
            speaker_stats[speaker]["total_words"] += log["word_count"]
            speaker_stats[speaker]["last_speak"] = log["timestamp"]
        
        self.metrics = {
            "total_messages": len(self.conversation_log),
            "total_words": sum(log["word_count"] for log in self.conversation_log),
            "speaker_stats": speaker_stats,
            "average_message_length": sum(log["word_count"] for log in self.conversation_log) / len(self.conversation_log),
            "duration": self.conversation_log[-1]["timestamp"] - self.conversation_log[0]["timestamp"] if len(self.conversation_log) > 1 else 0
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        return {
            "metrics": self.metrics,
            "recent_messages": self.conversation_log[-5:],
            "top_speakers": self._get_top_speakers(3)
        }
    
    def _get_top_speakers(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """获取最活跃的发言者"""
        if not self.metrics.get("speaker_stats"):
            return []
        
        sorted_speakers = sorted(
            self.metrics["speaker_stats"].items(),
            key=lambda x: x[1]["message_count"],
            reverse=True
        )[:top_k]
        
        return [
            {"name": name, **stats}
            for name, stats in sorted_speakers
        ]
    
    def detect_patterns(self) -> List[Dict[str, Any]]:
        """检测对话模式"""
        patterns = []
        
        # 检测对话循环
        if self._detect_loop():
            patterns.append({
                "type": "loop",
                "description": "检测到对话循环",
                "severity": "warning"
            })
        
        # 检测冷场
        if self._detect_silence():
            patterns.append({
                "type": "silence",
                "description": "检测到长时间沉默",
                "severity": "info"
            })
        
        return patterns
    
    def _detect_loop(self) -> bool:
        """检测对话循环"""
        if len(self.conversation_log) < 6:
            return False
        
        # 检查最近 6 条消息是否重复
        recent = [log["speaker"] for log in self.conversation_log[-6:]]
        return recent[:3] == recent[3:]
    
    def _detect_silence(self) -> bool:
        """检测沉默"""
        if len(self.conversation_log) < 2:
            return False
        
        # 检查最近两条消息的时间间隔
        last_two = self.conversation_log[-2:]
        time_diff = last_two[1]["timestamp"] - last_two[0]["timestamp"]
        
        return time_diff > 60  # 超过 60 秒
```

---

## 16.4 工具集成系统

### 16.4.1 工具注册和管理

```python
from autogen import register_function
from typing import Callable, Any
import inspect

class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_agents: Dict[str, List[str]] = {}
    
    def register_tool(
        self,
        func: Callable,
        caller: ConversableAgent,
        executor: ConversableAgent,
        description: str = None
    ):
        """注册工具"""
        tool_name = func.__name__
        
        # 自动生成描述
        if not description:
            description = self._generate_description(func)
        
        # 注册函数
        register_function(
            func,
            caller=caller,
            executor=executor,
            description=description
        )
        
        # 记录工具信息
        self.tools[tool_name] = {
            "function": func,
            "description": description,
            "parameters": self._extract_parameters(func),
            "caller": caller.name,
            "executor": executor.name
        }
        
        # 更新 Agent 工具映射
        for agent_name in [caller.name, executor.name]:
            if agent_name not in self.tool_agents:
                self.tool_agents[agent_name] = []
            self.tool_agents[agent_name].append(tool_name)
    
    def _generate_description(self, func: Callable) -> str:
        """生成工具描述"""
        docstring = inspect.getdoc(func)
        if docstring:
            return docstring.split("\n")[0]
        
        # 基于函数名和参数生成描述
        params = list(inspect.signature(func).parameters.keys())
        return f"执行 {func.__name__} 操作，参数: {', '.join(params)}"
    
    def _extract_parameters(self, func: Callable) -> List[Dict[str, str]]:
        """提取参数信息"""
        params = []
        sig = inspect.signature(func)
        
        for name, param in sig.parameters.items():
            param_info = {
                "name": name,
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                "default": str(param.default) if param.default != inspect.Parameter.empty else None
            }
            params.append(param_info)
        
        return params
    
    def get_tools_for_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        """获取 Agent 可用工具"""
        tool_names = self.tool_agents.get(agent_name, [])
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """获取所有工具"""
        return list(self.tools.values())

class ToolExecutor:
    """工具执行器"""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.execution_log: List[Dict[str, Any]] = []
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        agent_name: str
    ) -> Dict[str, Any]:
        """执行工具"""
        if tool_name not in self.registry.tools:
            return {"error": f"工具不存在: {tool_name}"}
        
        tool_info = self.registry.tools[tool_name]
        func = tool_info["function"]
        
        start_time = time.time()
        
        try:
            # 执行函数
            result = await func(**arguments) if inspect.iscoroutinefunction(func) else func(**arguments)
            
            execution_time = time.time() - start_time
            
            # 记录执行日志
            log_entry = {
                "tool": tool_name,
                "agent": agent_name,
                "arguments": arguments,
                "result": str(result)[:500],
                "success": True,
                "execution_time": execution_time
            }
            self.execution_log.append(log_entry)
            
            return {"success": True, "result": result, "execution_time": execution_time}
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            log_entry = {
                "tool": tool_name,
                "agent": agent_name,
                "arguments": arguments,
                "error": str(e),
                "success": False,
                "execution_time": execution_time
            }
            self.execution_log.append(log_entry)
            
            return {"success": False, "error": str(e), "execution_time": execution_time}
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.execution_log:
            return {}
        
        successful = sum(1 for log in self.execution_log if log["success"])
        total = len(self.execution_log)
        
        tool_usage = {}
        for log in self.execution_log:
            tool = log["tool"]
            if tool not in tool_usage:
                tool_usage[tool] = {"success": 0, "failed": 0}
            
            if log["success"]:
                tool_usage[tool]["success"] += 1
            else:
                tool_usage[tool]["failed"] += 1
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": successful / total if total > 0 else 0,
            "tool_usage": tool_usage,
            "average_execution_time": sum(log["execution_time"] for log in self.execution_log) / total
        }
```

### 16.4.2 常用工具示例

```python
import requests
import json
from typing import Optional

class WebSearchTool:
    """网页搜索工具"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    async def search(self, query: str, num_results: int = 5) -> str:
        """执行搜索"""
        # 这里可以集成各种搜索 API
        # 例如：Google Search, Bing, DuckDuckGo
        
        # 模拟搜索结果
        results = [
            {"title": f"Result {i+1}", "url": f"https://example.com/{i}", "snippet": f"Snippet for {query}"}
            for i in range(num_results)
        ]
        
        formatted = "\n".join([
            f"{i+1}. {r['title']}\n   {r['url']}\n   {r['snippet']}"
            for i, r in enumerate(results)
        ])
        
        return formatted

class FileOperationTool:
    """文件操作工具"""
    
    def read_file(self, file_path: str) -> str:
        """读取文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"读取文件失败: {e}"
    
    def write_file(self, file_path: str, content: str) -> str:
        """写入文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"文件已写入: {file_path}"
        except Exception as e:
            return f"写入文件失败: {e}"
    
    def list_files(self, directory: str) -> str:
        """列出文件"""
        import os
        try:
            files = os.listdir(directory)
            return "\n".join(files)
        except Exception as e:
            return f"列出文件失败: {e}"

class DataAnalysisTool:
    """数据分析工具"""
    
    def analyze_csv(self, file_path: str) -> str:
        """分析 CSV 文件"""
        import pandas as pd
        
        try:
            df = pd.read_csv(file_path)
            
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
        except Exception as e:
            return f"分析失败: {e}"

class APICallTool:
    """API 调用工具"""
    
    def __init__(self, base_url: str, headers: Dict[str, str] = None):
        self.base_url = base_url
        self.headers = headers or {}
    
    async def get(self, endpoint: str, params: Dict[str, Any] = None) -> str:
        """GET 请求"""
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                params=params,
                timeout=10
            )
            return json.dumps(response.json(), indent=2, ensure_ascii=False)
        except Exception as e:
            return f"请求失败: {e}"
    
    async def post(self, endpoint: str, data: Dict[str, Any] = None) -> str:
        """POST 请求"""
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                json=data,
                timeout=10
            )
            return json.dumps(response.json(), indent=2, ensure_ascii=False)
        except Exception as e:
            return f"请求失败: {e}"
```

---

## 16.5 状态管理和记忆

### 16.5.1 对话状态管理

```python
class ConversationState:
    """对话状态管理"""
    
    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.shared_memory: Dict[str, Any] = {}
    
    def update_state(self, key: str, value: Any):
        """更新状态"""
        self.state[key] = value
        self.history.append({
            "action": "update_state",
            "key": key,
            "value": str(value)[:100],
            "timestamp": time.time()
        })
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """获取状态"""
        return self.state.get(key, default)
    
    def set_shared_memory(self, key: str, value: Any):
        """设置共享内存"""
        self.shared_memory[key] = value
    
    def get_shared_memory(self, key: str, default: Any = None) -> Any:
        """获取共享内存"""
        return self.shared_memory.get(key, default)
    
    def export_state(self) -> Dict[str, Any]:
        """导出状态"""
        return {
            "state": self.state.copy(),
            "shared_memory": self.shared_memory.copy(),
            "history": self.history[-10:]  # 最近 10 条历史
        }
    
    def import_state(self, state_data: Dict[str, Any]):
        """导入状态"""
        self.state.update(state_data.get("state", {}))
        self.shared_memory.update(state_data.get("shared_memory", {}))

class StatefulAgent:
    """有状态的 Agent"""
    
    def __init__(self, agent: ConversableAgent):
        self.agent = agent
        self.state = ConversationState()
        self.task_memory: List[Dict[str, Any]] = []
    
    async def run_with_state(self, task: str) -> Dict[str, Any]:
        """带状态运行任务"""
        # 记录任务开始
        self.state.update_state("current_task", task)
        self.state.update_state("task_start_time", time.time())
        
        try:
            # 执行任务
            result = await self.agent.a_generate_reply(
                {"content": task, "role": "user"}
            )
            
            # 记录任务完成
            task_record = {
                "task": task,
                "result": result,
                "timestamp": time.time(),
                "status": "completed"
            }
            self.task_memory.append(task_record)
            
            self.state.update_state("current_task", None)
            self.state.update_state("last_result", result)
            
            return {"success": True, "result": result}
        
        except Exception as e:
            task_record = {
                "task": task,
                "error": str(e),
                "timestamp": time.time(),
                "status": "failed"
            }
            self.task_memory.append(task_record)
            
            return {"success": False, "error": str(e)}
    
    def get_task_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取任务历史"""
        return self.task_memory[-limit:]
```

---

## 16.6 嵌套对话

### 16.6.1 嵌套对话模式

```python
class NestedConversationManager:
    """嵌套对话管理器"""
    
    def __init__(self):
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.conversation_tree: Dict[str, List[str]] = {}
    
    async def start_nested_conversation(
        self,
        parent_id: str,
        agents: List[ConversableAgent],
        topic: str,
        max_rounds: int = 5
    ) -> Dict[str, Any]:
        """启动嵌套对话"""
        conversation_id = f"{parent_id}_nested_{len(self.conversations)}"
        
        # 创建群聊
        group_chat = GroupChat(
            agents=agents,
            messages=[],
            max_round=max_rounds,
            speaker_selection_method="auto"
        )
        
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=agents[0].llm_config
        )
        
        # 启动对话
        initiator = agents[0]
        result = await initiator.a_initiate_chat(
            manager,
            message=topic,
            max_turns=max_rounds
        )
        
        # 记录对话
        self.conversations[conversation_id] = result.chat_history
        self.conversation_tree[conversation_id] = {
            "parent": parent_id,
            "topic": topic,
            "participants": [a.name for a in agents]
        }
        
        return {
            "conversation_id": conversation_id,
            "summary": result.summary,
            "history": result.chat_history
        }
    
    def get_conversation_hierarchy(self) -> Dict[str, Any]:
        """获取对话层次结构"""
        hierarchy = {}
        
        for conv_id, info in self.conversation_tree.items():
            parent = info["parent"]
            if parent not in hierarchy:
                hierarchy[parent] = []
            hierarchy[parent].append({
                "id": conv_id,
                "topic": info["topic"],
                "participants": info["participants"]
            })
        
        return hierarchy
    
    def merge_conversation_results(self, conversation_id: str) -> str:
        """合并对话结果"""
        if conversation_id not in self.conversations:
            return ""
        
        history = self.conversations[conversation_id]
        merged = []
        
        for msg in history:
            if "content" in msg:
                merged.append(f"{msg.get('name', 'Unknown')}: {msg['content'][:200]}")
        
        return "\n".join(merged)
```

---

## 16.7 生产实践案例

### 16.7.1 代码审查系统

```python
class CodeReviewSystem:
    """代码审查系统"""
    
    def __init__(self):
        self.review_agents = self._create_review_agents()
        self.code_executor = EnhancedCodeExecutor(use_docker=True)
    
    def _create_review_agents(self) -> Dict[str, ConversableAgent]:
        """创建审查 Agent"""
        config_list = [{"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}]
        llm_config = {"config_list": config_list}
        
        agents = {
            "security_reviewer": ConversableAgent(
                name="security_reviewer",
                system_message="""你是安全审查专家，专注于检查代码中的安全漏洞。
                你需要检查：
                1. SQL 注入风险
                2. XSS 漏洞
                3. 权限问题
                4. 敏感信息泄露
                5. 不安全的依赖""",
                llm_config=llm_config,
                human_input_mode="NEVER"
            ),
            
            "performance_reviewer": ConversableAgent(
                name="performance_reviewer",
                system_message="""你是性能审查专家，专注于检查代码的性能问题。
                你需要检查：
                1. 时间复杂度
                2. 空间复杂度
                3. 数据库查询优化
                4. 缓存使用
                5. 并发处理""",
                llm_config=llm_config,
                human_input_mode="NEVER"
            ),
            
            "code_quality_reviewer": ConversableAgent(
                name="code_quality_reviewer",
                system_message="""你是代码质量审查专家，专注于检查代码质量。
                你需要检查：
                1. 代码规范
                2. 命名约定
                3. 注释质量
                4. 代码复用
                5. 错误处理""",
                llm_config=llm_config,
                human_input_mode="NEVER"
            ),
            
            "lead_reviewer": ConversableAgent(
                name="lead_reviewer",
                system_message="""你是首席审查员，负责协调所有审查员的工作。
                你需要：
                1. 分配审查任务
                2. 汇总审查结果
                3. 做出最终决定
                4. 提供改进建议""",
                llm_config=llm_config,
                human_input_mode="NEVER"
            )
        }
        
        return agents
    
    async def review_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """审查代码"""
        # 执行代码测试
        test_result = await self.code_executor.execute_code(code)
        
        # 创建审查群聊
        group_chat = GroupChat(
            agents=list(self.review_agents.values()),
            messages=[],
            max_round=10,
            speaker_selection_method="auto"
        )
        
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=self.review_agents["lead_reviewer"].llm_config
        )
        
        # 启动审查
        review_message = f"""
请审查以下代码：

语言：{language}
代码：
```
{code}
```

代码执行结果：
{json.dumps(test_result, indent=2)}

请从安全、性能和代码质量三个方面进行审查。
"""
        
        result = await self.review_agents["lead_reviewer"].a_initiate_chat(
            manager,
            message=review_message,
            max_turns=10
        )
        
        return {
            "code": code,
            "test_result": test_result,
            "review_summary": result.summary,
            "review_history": result.chat_history
        }
```

### 16.7.2 研究分析系统

```python
class ResearchAnalysisSystem:
    """研究分析系统"""
    
    def __init__(self):
        self.research_agents = self._create_research_agents()
    
    def _create_research_agents(self) -> Dict[str, ConversableAgent]:
        """创建研究 Agent"""
        config_list = [{"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}]
        llm_config = {"config_list": config_list}
        
        agents = {
            "data_collector": ConversableAgent(
                name="data_collector",
                system_message="""你是数据收集专家，负责收集和整理研究数据。
                你需要：
                1. 搜索相关资料
                2. 提取关键信息
                3. 整理数据格式
                4. 验证数据准确性""",
                llm_config=llm_config,
                human_input_mode="NEVER"
            ),
            
            "analyst": ConversableAgent(
                name="analyst",
                system_message="""你是数据分析专家，负责分析收集到的数据。
                你需要：
                1. 识别数据模式
                2. 进行统计分析
                3. 发现关键洞察
                4. 生成分析报告""",
                llm_config=llm_config,
                human_input_mode="NEVER"
            ),
            
            "report_writer": ConversableAgent(
                name="report_writer",
                system_message="""你是报告撰写专家，负责撰写研究报告。
                你需要：
                1. 组织报告结构
                2. 撰写清晰内容
                3. 添加数据可视化
                4. 提出建议""",
                llm_config=llm_config,
                human_input_mode="NEVER"
            ),
            
            "reviewer": ConversableAgent(
                name="reviewer",
                system_message="""你是审查专家，负责审查研究报告。
                你需要：
                1. 检查事实准确性
                2. 评估逻辑连贯性
                3. 提供改进建议
                4. 确保质量标准""",
                llm_config=llm_config,
                human_input_mode="NEVER"
            )
        }
        
        return agents
    
    async def conduct_research(self, topic: str) -> Dict[str, Any]:
        """进行研究"""
        # 创建研究群聊
        group_chat = GroupChat(
            agents=list(self.research_agents.values()),
            messages=[],
            max_round=15,
            speaker_selection_method="auto"
        )
        
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=self.research_agents["data_collector"].llm_config
        )
        
        # 启动研究
        research_message = f"""
请对以下主题进行深入研究：

主题：{topic}

请按照以下步骤进行：
1. 数据收集：收集相关数据和资料
2. 数据分析：分析数据并发现洞察
3. 报告撰写：撰写详细的研究报告
4. 审查优化：审查并优化报告质量

最终输出一份完整的研究报告。
"""
        
        result = await self.research_agents["data_collector"].a_initiate_chat(
            manager,
            message=research_message,
            max_turns=15
        )
        
        return {
            "topic": topic,
            "research_summary": result.summary,
            "research_history": result.chat_history,
            "participants": list(self.research_agents.keys())
        }
```

---

## 16.8 AutoGen vs LangGraph vs CrewAI

### 16.8.1 特性对比

| 特性 | AutoGen | LangGraph | CrewAI |
|:---|:---|:---|:---|
| **核心理念** | 对话驱动 | 图编排 | 角色扮演 |
| **学习曲线** | 中 | 中高 | 低 |
| **配置复杂度** | 中等 | 复杂 | 简单 |
| **灵活性** | 高 | 很高 | 中 |
| **代码执行** | 内置 Docker | 需要自定义 | 需要自定义 |
| **状态管理** | 对话历史 | 自定义状态 | 内置记忆 |
| **适用场景** | 多 Agent 对话 | 复杂工作流 | 团队协作 |
| **生产就绪** | 是 | 是 | 是 |
| **社区支持** | 活跃 | 活跃 | 活跃 |

### 16.8.2 选择建议

```python
class FrameworkSelector:
    """框架选择器"""
    
    @staticmethod
    def recommend(requirements: Dict[str, Any]) -> str:
        """推荐框架"""
        
        # 分析需求
        use_case = requirements.get("use_case", "general")
        complexity = requirements.get("complexity", "medium")
        team_size = requirements.get("team_size", 1)
        
        # 推荐逻辑
        if use_case == "code_generation" or use_case == "code_review":
            return "AutoGen"
        
        elif use_case == "complex_workflow" or complexity == "high":
            return "LangGraph"
        
        elif use_case == "content_creation" or team_size > 1:
            return "CrewAI"
        
        else:
            return "AutoGen"  # 默认推荐 AutoGen
    
    @staticmethod
    def get_examples() -> Dict[str, List[str]]:
        """获取示例用例"""
        return {
            "AutoGen": [
                "代码生成和审查",
                "数据分析对话",
                "问题解决讨论",
                "知识问答系统",
                "多轮对话系统"
            ],
            "LangGraph": [
                "复杂工作流编排",
                "多步骤决策流程",
                "状态机应用",
                "需要回溯的复杂任务",
                "实时数据处理"
            ],
            "CrewAI": [
                "内容创作团队",
                "研究分析团队",
                "产品开发团队",
                "市场营销团队",
                "客户支持团队"
            ]
        }
```

---

## 16.9 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| **AutoGen 架构** | ConversableAgent → GroupChat → CodeExecutor → Tools |
| **对话机制** | 消息处理、自动回复、终止条件 |
| **GroupChat** | 多 Agent 群聊、发言者选择策略、群聊监控 |
| **代码执行** | Docker 沙箱、代码审查、安全执行 |
| **工具集成** | 工具注册、工具执行、工具统计 |
| **状态管理** | 对话状态、共享内存、任务记忆 |
| **嵌套对话** | 层次结构、对话合并、结果整合 |
| **生产实践** | 代码审查系统、研究分析系统 |

---

## 16.9 AutoGen 高级特性

### 16.9.1 自定义 Agent 类型

```python
from autogen import ConversableAgent
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import json

@dataclass
class AgentConfig:
    """Agent 配置"""
    name: str
    system_message: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: List[Dict] = None
    human_input_mode: str = "NEVER"

class SpecializedAgents:
    """专业化 Agent 集合"""
    
    def __init__(self, llm_config: Dict):
        self.llm_config = llm_config
    
    def create_code_reviewer(self) -> ConversableAgent:
        """创建代码审查 Agent"""
        return ConversableAgent(
            name="code_reviewer",
            system_message="""你是一位资深的代码审查专家。你的职责是：
            1. 检查代码质量和可读性
            2. 识别潜在的 bug 和安全漏洞
            3. 评估代码的性能和可维护性
            4. 提供具体的改进建议
            
            请提供详细、具体的审查意见，包括代码行号和改进建议。""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5
        )
    
    def create_data_analyst(self) -> ConversableAgent:
        """创建数据分析 Agent"""
        return ConversableAgent(
            name="data_analyst",
            system_message="""你是一位专业的数据分析师。你的职责是：
            1. 分析数据并发现趋势和模式
            2. 生成统计摘要和可视化建议
            3. 提供数据驱动的洞察
            4. 识别数据质量问题
            
            请提供清晰、有见地的分析结果。""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=8
        )
    
    def create_researcher(self) -> ConversableAgent:
        """创建研究员 Agent"""
        return ConversableAgent(
            name="researcher",
            system_message="""你是一位专注的研究员。你的职责是：
            1. 收集和整理相关信息
            2. 分析研究问题
            3. 提供基于证据的结论
            4. 识别研究的局限性
            
            请提供全面、客观的研究结果。""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10
        )
    
    def create_writer(self) -> ConversableAgent:
        """创建写手 Agent"""
        return ConversableAgent(
            name="writer",
            system_message="""你是一位专业的技术写手。你的职责是：
            1. 将复杂概念转化为易懂的内容
            2. 撰写清晰、结构化的文档
            3. 确保内容准确性和一致性
            4. 适应不同的目标受众
            
            请提供高质量、易读的内容。""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=6
        )
    
    def create_planner(self) -> ConversableAgent:
        """创建规划 Agent"""
        return ConversableAgent(
            name="planner",
            system_message="""你是一位经验丰富的项目规划师。你的职责是：
            1. 分解复杂任务为可管理的子任务
            2. 制定时间表和里程碑
            3. 识别依赖关系和风险
            4. 分配资源和责任
            
            请提供详细、可行的计划。""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5
        )
    
    def create_critic(self) -> ConversableAgent:
        """创建评审 Agent"""
        return ConversableAgent(
            name="critic",
            system_message="""你是一位严格的评审专家。你的职责是：
            1. 评估工作成果的质量
            2. 识别问题和不足
            3. 提供建设性的反馈
            4. 确保符合标准和要求
            
            请提供客观、详细的评审意见。""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=4
        )

class AgentFactory:
    """Agent 工厂"""
    
    def __init__(self, llm_config: Dict):
        self.llm_config = llm_config
        self.specialized_agents = SpecializedAgents(llm_config)
        self.custom_agents: Dict[str, Callable] = {}
    
    def register_custom_agent(self, name: str, factory_func: Callable):
        """注册自定义 Agent"""
        self.custom_agents[name] = factory_func
    
    def create_agent(self, agent_type: str, **kwargs) -> ConversableAgent:
        """创建 Agent"""
        # 检查是否为预定义类型
        predefined_creators = {
            "code_reviewer": self.specialized_agents.create_code_reviewer,
            "data_analyst": self.specialized_agents.create_data_analyst,
            "researcher": self.specialized_agents.create_researcher,
            "writer": self.specialized_agents.create_writer,
            "planner": self.specialized_agents.create_planner,
            "critic": self.specialized_agents.create_critic,
        }
        
        if agent_type in predefined_creators:
            return predefined_creators[agent_type]()
        
        # 检查是否为自定义类型
        if agent_type in self.custom_agents:
            return self.custom_agents[agent_type](**kwargs)
        
        # 创建通用 Agent
        return ConversableAgent(
            name=kwargs.get("name", agent_type),
            system_message=kwargs.get("system_message", f"你是一个{agent_type}专家。"),
            llm_config=self.llm_config,
            human_input_mode=kwargs.get("human_input_mode", "NEVER")
        )
    
    def create_team(self, team_config: List[Dict[str, Any]]) -> List[ConversableAgent]:
        """创建团队"""
        agents = []
        
        for agent_config in team_config:
            agent = self.create_agent(
                agent_type=agent_config["type"],
                **agent_config.get("config", {})
            )
            agents.append(agent)
        
        return agents
```

### 16.9.2 对话流程控制

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class FlowControlMode(Enum):
    """流程控制模式"""
    AUTO = "auto"
    MANUAL = "manual"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"

@dataclass
class ConversationFlow:
    """对话流程"""
    name: str
    steps: List[Dict[str, Any]]
    mode: FlowControlMode
    max_rounds: int = 10

class ConversationFlowManager:
    """对话流程管理器"""
    
    def __init__(self):
        self.flows: Dict[str, ConversationFlow] = {}
        self.current_flow: Optional[str] = None
        self.flow_history: List[Dict] = []
    
    def create_flow(self, flow_config: Dict[str, Any]) -> str:
        """创建流程"""
        flow = ConversationFlow(
            name=flow_config["name"],
            steps=flow_config["steps"],
            mode=FlowControlMode(flow_config.get("mode", "auto")),
            max_rounds=flow_config.get("max_rounds", 10)
        )
        
        self.flows[flow.name] = flow
        return flow.name
    
    def get_next_step(self, flow_name: str, current_step: int, context: Dict) -> Optional[Dict]:
        """获取下一步"""
        flow = self.flows.get(flow_name)
        if not flow:
            return None
        
        if current_step >= len(flow.steps) - 1:
            return None
        
        next_step = flow.steps[current_step + 1]
        
        # 检查条件
        if "condition" in next_step:
            if not self._evaluate_condition(next_step["condition"], context):
                # 跳过此步骤，查找下一步
                return self.get_next_step(flow_name, current_step + 1, context)
        
        return next_step
    
    def _evaluate_condition(self, condition: Dict, context: Dict) -> bool:
        """评估条件"""
        condition_type = condition.get("type", "always")
        
        if condition_type == "always":
            return True
        
        elif condition_type == "variable":
            var_name = condition.get("variable")
            expected_value = condition.get("value")
            return context.get(var_name) == expected_value
        
        elif condition_type == "function":
            func_name = condition.get("function")
            # 这里可以执行自定义函数
            return True
        
        return True
    
    def record_flow_execution(self, flow_name: str, step: int, result: Any):
        """记录流程执行"""
        self.flow_history.append({
            "flow": flow_name,
            "step": step,
            "result": str(result)[:200],
            "timestamp": time.time()
        })
    
    def get_flow_status(self, flow_name: str) -> Dict[str, Any]:
        """获取流程状态"""
        flow = self.flows.get(flow_name)
        if not flow:
            return {"error": "Flow not found"}
        
        relevant_history = [
            h for h in self.flow_history if h["flow"] == flow_name
        ]
        
        return {
            "flow_name": flow_name,
            "total_steps": len(flow.steps),
            "completed_steps": len(relevant_history),
            "status": "completed" if len(relevant_history) >= len(flow.steps) else "in_progress"
        }
    
    def generate_flow_diagram(self, flow_name: str) -> str:
        """生成流程图"""
        flow = self.flows.get(flow_name)
        if not flow:
            return "Flow not found"
        
        diagram = f"Flow: {flow.name}\n"
        diagram += "=" * 30 + "\n\n"
        
        for i, step in enumerate(flow.steps):
            diagram += f"Step {i+1}: {step.get('name', 'Unnamed')}\n"
            diagram += f"  Agent: {step.get('agent', 'N/A')}\n"
            diagram += f"  Task: {step.get('task', 'N/A')}\n"
            if "condition" in step:
                diagram += f"  Condition: {step['condition']}\n"
            diagram += "\n"
        
        return diagram
```

---

## 16.10 AutoGen 性能优化

### 16.10.1 性能分析和优化

```python
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
import asyncio
import time

@dataclass
class PerformanceMetrics:
    """性能指标"""
    conversation_count: int = 0
    total_messages: int = 0
    avg_response_time: float = 0
    total_tokens: int = 0
    error_count: int = 0

class AutoGenPerformanceOptimizer:
    """AutoGen 性能优化器"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.response_times: List[float] = []
        self.optimization_strategies: Dict[str, Callable] = {}
        
        self._register_strategies()
    
    def _register_strategies(self):
        """注册优化策略"""
        self.optimization_strategies = {
            "response_caching": self._optimize_with_caching,
            "batch_processing": self._optimize_with_batching,
            "parallel_execution": self._optimize_with_parallel,
            "model_selection": self._optimize_with_model_selection
        }
    
    async def _optimize_with_caching(self, messages: List[Dict], context: Dict) -> Dict:
        """使用缓存优化"""
        # 实现响应缓存
        cache_key = self._generate_cache_key(messages)
        
        if cache_key in context.get("cache", {}):
            return {"cached": True, "response": context["cache"][cache_key]}
        
        return {"cached": False}
    
    async def _optimize_with_batching(self, messages: List[Dict], context: Dict) -> Dict:
        """使用批处理优化"""
        # 实现请求批处理
        batch_size = context.get("batch_size", 5)
        
        if len(messages) > batch_size:
            # 分批处理
            batches = [messages[i:i+batch_size] for i in range(0, len(messages), batch_size)]
            return {"batched": True, "batches": len(batches)}
        
        return {"batched": False}
    
    async def _optimize_with_parallel(self, messages: List[Dict], context: Dict) -> Dict:
        """使用并行优化"""
        # 实现并行处理
        parallel_threshold = context.get("parallel_threshold", 3)
        
        if len(messages) > parallel_threshold:
            return {"parallel": True, "tasks": len(messages)}
        
        return {"parallel": False}
    
    async def _optimize_with_model_selection(self, messages: List[Dict], context: Dict) -> Dict:
        """使用模型选择优化"""
        # 根据任务复杂度选择模型
        complexity = self._estimate_complexity(messages)
        
        if complexity == "simple":
            recommended_model = "gpt-3.5-turbo"
        elif complexity == "moderate":
            recommended_model = "gpt-4"
        else:
            recommended_model = "gpt-4-turbo"
        
        return {"recommended_model": recommended_model, "complexity": complexity}
    
    def _generate_cache_key(self, messages: List[Dict]) -> str:
        """生成缓存键"""
        import hashlib
        import json
        
        # 使用消息内容生成键
        content = json.dumps([m.get("content", "") for m in messages[-3:]])
        return hashlib.md5(content.encode()).hexdigest()
    
    def _estimate_complexity(self, messages: List[Dict]) -> str:
        """估算复杂度"""
        # 基于消息长度和内容估算
        total_length = sum(len(m.get("content", "")) for m in messages)
        
        if total_length < 500:
            return "simple"
        elif total_length < 2000:
            return "moderate"
        else:
            return "complex"
    
    def record_response_time(self, response_time: float):
        """记录响应时间"""
        self.response_times.append(response_time)
        self.metrics.avg_response_time = sum(self.response_times) / len(self.response_times)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            "metrics": {
                "conversation_count": self.metrics.conversation_count,
                "total_messages": self.metrics.total_messages,
                "avg_response_time": self.metrics.avg_response_time,
                "total_tokens": self.metrics.total_tokens,
                "error_count": self.metrics.error_count
            },
            "response_time_distribution": {
                "min": min(self.response_times) if self.response_times else 0,
                "max": max(self.response_times) if self.response_times else 0,
                "avg": self.metrics.avg_response_time,
                "p95": sorted(self.response_times)[int(len(self.response_times) * 0.95)] if self.response_times else 0
            },
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if self.metrics.avg_response_time > 5:
            recommendations.append("响应时间较长，建议使用缓存或更快的模型")
        
        if self.metrics.error_count > 10:
            recommendations.append("错误率较高，建议检查错误处理逻辑")
        
        return recommendations
```

---

## 16.11 AutoGen 安全实践

### 16.11.1 安全配置和最佳实践

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re

@dataclass
class AutoGenSecurityConfig:
    """AutoGen 安全配置"""
    max_message_length: int = 4000
    enable_injection_detection: bool = True
    blocked_patterns: List[str] = None
    enable_output_filtering: bool = True
    allowed_tools: List[str] = None
    max_conversations_per_hour: int = 100
    
    def __post_init__(self):
        if self.blocked_patterns is None:
            self.blocked_patterns = [
                r"ignore.*instructions",
                r"忽略.*指令",
                r"system prompt",
                r"reveal.*secret"
            ]

class AutoGenSecurityManager:
    """AutoGen 安全管理器"""
    
    def __init__(self, config: AutoGenSecurityConfig = None):
        self.config = config or AutoGenSecurityConfig()
        self.security_events: List[Dict] = []
        self.conversation_counts: Dict[str, int] = {}
    
    def validate_message(self, message: str, sender: str) -> Dict[str, Any]:
        """验证消息"""
        issues = []
        
        # 检查长度
        if len(message) > self.config.max_message_length:
            issues.append(f"Message too long: {len(message)}")
        
        # 检查注入模式
        if self.config.enable_injection_detection:
            for pattern in self.config.blocked_patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    issues.append(f"Potential injection detected: {pattern}")
                    self._log_security_event("injection_attempt", {
                        "sender": sender,
                        "pattern": pattern,
                        "message_preview": message[:100]
                    })
        
        # 检查速率限制
        if not self._check_rate_limit(sender):
            issues.append("Rate limit exceeded")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def filter_output(self, output: str) -> str:
        """过滤输出"""
        if not self.config.enable_output_filtering:
            return output
        
        filtered = output
        
        # 移除 API 密钥
        filtered = re.sub(r'sk-[A-Za-z0-9]{20,}', '***API_KEY***', filtered)
        
        # 移除邮箱
        filtered = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '***EMAIL***', filtered)
        
        return filtered
    
    def validate_tool_call(self, tool_name: str, tool_input: Any, agent_name: str) -> Dict[str, Any]:
        """验证工具调用"""
        # 检查工具是否允许
        if self.config.allowed_tools and tool_name not in self.config.allowed_tools:
            return {
                "allowed": False,
                "reason": f"Tool {tool_name} not allowed"
            }
        
        # 记录工具调用
        self._log_security_event("tool_call", {
            "agent": agent_name,
            "tool": tool_name,
            "input_preview": str(tool_input)[:100]
        })
        
        return {"allowed": True}
    
    def _check_rate_limit(self, agent_name: str) -> bool:
        """检查速率限制"""
        current_hour = int(time.time() // 3600)
        key = f"{agent_name}:{current_hour}"
        
        current_count = self.conversation_counts.get(key, 0)
        if current_count >= self.config.max_conversations_per_hour:
            return False
        
        self.conversation_counts[key] = current_count + 1
        return True
    
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
            "recent_events": self.security_events[-10:],
            "rate_limit_status": self.conversation_counts
        }
```

---

## 16.12 AutoGen 生产案例

### 16.12.1 企业级 AutoGen 应用

```python
class EnterpriseAutoGenCase:
    """企业级 AutoGen 案例"""
    
    def __init__(self):
        self.case_study = {
            "company": "某大型科技公司",
            "use_case": "智能客服系统",
            "scale": {
                "daily_conversations": 50000,
                "agents": 20,
                "response_time_sla": "2秒"
            }
        }
    
    async def implement_solution(self) -> Dict[str, Any]:
        """实施解决方案"""
        solution = {
            "architecture": {
                "components": [
                    {
                        "name": "Intent Router",
                        "agent_type": "conversable",
                        "responsibility": "识别用户意图并路由到相应 Agent"
                    },
                    {
                        "name": "Technical Support Agent",
                        "agent_type": "code_reviewer",
                        "responsibility": "处理技术问题"
                    },
                    {
                        "name": "Billing Agent",
                        "agent_type": "data_analyst",
                        "responsibility": "处理账单问题"
                    },
                    {
                        "name": "Escalation Manager",
                        "agent_type": "planner",
                        "responsibility": "处理需要人工介入的情况"
                    }
                ],
                "workflow": "Intent Recognition → Agent Routing → Response Generation → Quality Check"
            },
            "optimizations": [
                {
                    "type": "caching",
                    "description": "缓存常见问题的回答",
                    "impact": "减少 30% 的 LLM 调用"
                },
                {
                    "type": "parallel_processing",
                    "description": "并行处理多个对话",
                    "impact": "提高 50% 的吞吐量"
                },
                {
                    "type": "model_selection",
                    "description": "根据问题复杂度选择模型",
                    "impact": "降低 40% 的成本"
                }
            ],
            "results": {
                "response_time": "1.5秒 (满足 2秒 SLA)",
                "accuracy": "95%",
                "customer_satisfaction": "4.8/5",
                "cost_reduction": "35%"
            }
        }
        
        return solution
    
    def get_best_practices(self) -> List[Dict[str, str]]:
        """获取最佳实践"""
        return [
            {
                "practice": "明确的 Agent 职责",
                "description": "每个 Agent 专注于特定领域",
                "benefit": "提高响应质量和效率"
            },
            {
                "practice": "有效的意图路由",
                "description": "准确识别用户意图并路由到正确的 Agent",
                "benefit": "减少处理时间"
            },
            {
                "practice": "持续的性能监控",
                "description": "实时监控系统性能和用户满意度",
                "benefit": "及时发现和解决问题"
            },
            {
                "practice": "优雅的降级策略",
                "description": "当 Agent 无法处理时，提供备用方案",
                "benefit": "保证用户体验"
            }
        ]
```

---

## 16.13 AutoGen 调试和监控

### 16.13.1 调试工具

```python
from typing import Dict, List, Any
from dataclasses import dataclass
import time

@dataclass
class DebugEvent:
    """调试事件"""
    event_type: str
    timestamp: float
    agent: str
    message: str
    details: Dict[str, Any]
    level: str

class AutoGenDebugger:
    """AutoGen 调试器"""
    
    def __init__(self):
        self.events: List[DebugEvent] = []
        self.conversation_logs: List[Dict] = []
        self.breakpoints: Dict[str, bool] = {}
    
    def log_event(self, event_type: str, agent: str, message: str, details: Dict = None, level: str = "info"):
        """记录事件"""
        event = DebugEvent(
            event_type=event_type,
            timestamp=time.time(),
            agent=agent,
            message=message,
            details=details or {},
            level=level
        )
        
        self.events.append(event)
        
        # 检查断点
        if agent in self.breakpoints and self.breakpoints[agent]:
            print(f"\n=== Breakpoint hit: {agent} ===")
            print(f"Event: {event_type}")
            print(f"Message: {message[:200]}")
    
    def log_conversation(self, sender: str, receiver: str, message: str):
        """记录对话"""
        self.conversation_logs.append({
            "sender": sender,
            "receiver": receiver,
            "message": message[:500],
            "timestamp": time.time()
        })
    
    def set_breakpoint(self, agent_name: str):
        """设置断点"""
        self.breakpoints[agent_name] = True
    
    def remove_breakpoint(self, agent_name: str):
        """移除断点"""
        self.breakpoints.pop(agent_name, None)
    
    def get_event_log(self, level: str = None, limit: int = 100) -> List[DebugEvent]:
        """获取事件日志"""
        filtered = self.events
        
        if level:
            filtered = [e for e in filtered if e.level == level]
        
        return filtered[-limit:]
    
    def get_conversation_flow(self, limit: int = 50) -> List[Dict]:
        """获取对话流"""
        return self.conversation_logs[-limit:]
    
    def generate_debug_report(self) -> Dict[str, Any]:
        """生成调试报告"""
        return {
            "total_events": len(self.events),
            "events_by_level": {
                level: sum(1 for e in self.events if e.level == level)
                for level in ["info", "warning", "error"]
            },
            "conversation_count": len(self.conversation_logs),
            "active_breakpoints": list(self.breakpoints.keys()),
            "recent_events": [
                {
                    "type": e.event_type,
                    "agent": e.agent,
                    "message": e.message[:100]
                }
                for e in self.events[-10:]
            ]
        }
```

---

## 16.14 AutoGen 最佳实践

### 16.14.1 设计原则

```python
class AutoGenBestPractices:
    """AutoGen 最佳实践"""
    
    @staticmethod
    def get_design_principles() -> List[Dict[str, str]]:
        """获取设计原则"""
        return [
            {
                "principle": "单一职责",
                "description": "每个 Agent 只负责一个特定任务",
                "implementation": "为每个 Agent 定义清晰的角色和职责"
            },
            {
                "principle": "松耦合",
                "description": "Agent 之间通过消息通信，减少直接依赖",
                "implementation": "使用消息传递而非直接调用"
            },
            {
                "principle": "可观测性",
                "description": "记录所有对话和操作",
                "implementation": "实现详细的日志和监控"
            },
            {
                "principle": "容错性",
                "description": "优雅处理错误和异常",
                "implementation": "实现重试机制和降级策略"
            },
            {
                "principle": "可测试性",
                "description": "便于单元测试和集成测试",
                "implementation": "使用依赖注入和模拟对象"
            }
        ]
    
    @staticmethod
    def get_coding_standards() -> List[Dict[str, str]]:
        """获取编码规范"""
        return [
            {
                "category": "Agent 定义",
                "standards": [
                    "使用描述性的 system_message",
                    "设置合理的 max_consecutive_auto_reply",
                    "配置适当的 human_input_mode"
                ]
            },
            {
                "category": "对话管理",
                "standards": [
                    "使用结构化的消息格式",
                    "实现消息验证",
                    "记录对话历史"
                ]
            },
            {
                "category": "错误处理",
                "standards": [
                    "捕获具体的异常类型",
                    "提供有意义的错误信息",
                    "实现优雅的降级"
                ]
            },
            {
                "category": "性能优化",
                "standards": [
                    "使用异步操作",
                    "实现响应缓存",
                    "避免不必要的 LLM 调用"
                ]
            }
        ]
```

---

## 16.15 本章小结（更新版）

| 知识点 | 核心要点 |
|:---|:---|
| **AutoGen 架构** | ConversableAgent → GroupChat → CodeExecutor → Tools |
| **对话机制** | 消息处理、自动回复、终止条件 |
| **GroupChat** | 多 Agent 群聊、发言者选择策略、群聊监控 |
| **代码执行** | Docker 沙箱、代码审查、安全执行 |
| **工具集成** | 工具注册、工具执行、工具统计 |
| **状态管理** | 对话状态、共享内存、任务记忆 |
| **嵌套对话** | 层次结构、对话合并、结果整合 |
| **生产实践** | 代码审查系统、研究分析系统 |
| **自定义 Agent** | 专业化 Agent、Agent 工厂、团队组建 |
| **流程控制** | 对话流程、条件分支、并行执行 |
| **性能优化** | 缓存、批处理、并行、模型选择 |
| **安全实践** | 输入验证、输出过滤、速率限制 |
| **调试监控** | 事件日志、对话记录、断点调试 |
| **最佳实践** | 单一职责、松耦合、可观测性 |

---

## 16.16 AutoGen 工具生态

### 16.16.1 内置工具集

```python
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class ToolInfo:
    """工具信息"""
    name: str
    description: str
    category: str
    parameters: Dict[str, Any]
    examples: List[str]

class AutoGenToolkit:
    """AutoGen 工具集"""
    
    def __init__(self):
        self.tools: Dict[str, ToolInfo] = {}
        self.tool_categories: Dict[str, List[str]] = {}
        
        self._load_builtin_tools()
    
    def _load_builtin_tools(self):
        """加载内置工具"""
        builtin_tools = [
            ToolInfo(
                name="code_executor",
                description="执行 Python 代码",
                category="code",
                parameters={"code": "str", "timeout": "int"},
                examples=["执行数据分析脚本", "运行算法"]
            ),
            ToolInfo(
                name="web_search",
                description="网络搜索",
                category="research",
                parameters={"query": "str", "num_results": "int"},
                examples=["搜索最新信息", "查找资料"]
            ),
            ToolInfo(
                name="file_reader",
                description="读取文件内容",
                category="file",
                parameters={"file_path": "str"},
                examples=["读取配置文件", "读取数据文件"]
            ),
            ToolInfo(
                name="file_writer",
                description="写入文件内容",
                category="file",
                parameters={"file_path": "str", "content": "str"},
                examples=["保存分析结果", "生成报告"]
            ),
            ToolInfo(
                name="database_query",
                description="执行数据库查询",
                category="data",
                parameters={"query": "str", "database": "str"},
                examples=["查询用户数据", "统计分析"]
            ),
            ToolInfo(
                name="api_caller",
                description="调用外部 API",
                category="integration",
                parameters={"url": "str", "method": "str", "data": "dict"},
                examples=["调用天气 API", "调用支付接口"]
            ),
            ToolInfo(
                name="image_analyzer",
                description="分析图片内容",
                category="vision",
                parameters={"image_path": "str"},
                examples=["识别图片中的物体", "分析图表"]
            ),
            ToolInfo(
                name="document_parser",
                description="解析文档内容",
                category="document",
                parameters={"file_path": "str", "format": "str"},
                examples=["解析 PDF 文档", "解析 Word 文档"]
            ),
        ]
        
        for tool in builtin_tools:
            self.tools[tool.name] = tool
            
            if tool.category not in self.tool_categories:
                self.tool_categories[tool.category] = []
            self.tool_categories[tool.category].append(tool.name)
    
    def get_tool(self, tool_name: str) -> Optional[ToolInfo]:
        """获取工具"""
        return self.tools.get(tool_name)
    
    def get_tools_by_category(self, category: str) -> List[ToolInfo]:
        """按类别获取工具"""
        tool_names = self.tool_categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def search_tools(self, query: str) -> List[ToolInfo]:
        """搜索工具"""
        results = []
        
        for tool in self.tools.values():
            if (query.lower() in tool.name.lower() or
                query.lower() in tool.description.lower()):
                results.append(tool)
        
        return results
    
    def get_tool_recommendations(self, task_description: str) -> List[ToolInfo]:
        """获取工具推荐"""
        recommendations = []
        
        task_keywords = task_description.lower().split()
        
        for tool in self.tools.values():
            score = 0
            for keyword in task_keywords:
                if keyword in tool.name.lower() or keyword in tool.description.lower():
                    score += 1
            
            if score > 0:
                recommendations.append((tool, score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return [tool for tool, score in recommendations[:5]]
```

---

## 16.17 AutoGen 测试框架

### 16.17.1 测试工具

```python
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
import asyncio
import time

@dataclass
class TestCase:
    """测试用例"""
    name: str
    description: str
    input_data: Dict[str, Any]
    expected_output: Any
    timeout: int = 30

class AutoGenTestSuite:
    """AutoGen 测试套件"""
    
    def __init__(self, agents: List[ConversableAgent]):
        self.agents = agents
        self.test_results: List[Dict] = []
    
    async def run_test(self, test_case: TestCase) -> Dict[str, Any]:
        """运行单个测试"""
        start_time = time.time()
        
        try:
            # 执行测试
            result = await asyncio.wait_for(
                self._execute_test(test_case),
                timeout=test_case.timeout
            )
            
            execution_time = time.time() - start_time
            
            # 验证结果
            passed = self._validate_result(result, test_case.expected_output)
            
            return {
                "test_name": test_case.name,
                "passed": passed,
                "execution_time": execution_time,
                "result": result,
                "expected": test_case.expected_output
            }
        
        except asyncio.TimeoutError:
            return {
                "test_name": test_case.name,
                "passed": False,
                "error": "Timeout",
                "execution_time": time.time() - start_time
            }
        except Exception as e:
            return {
                "test_name": test_case.name,
                "passed": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    async def _execute_test(self, test_case: TestCase) -> Any:
        """执行测试"""
        # 这里可以根据测试用例执行不同的测试逻辑
        # 简化实现
        return "test_result"
    
    def _validate_result(self, result: Any, expected: Any) -> bool:
        """验证结果"""
        if expected is None:
            return True
        
        return result == expected
    
    async def run_all_tests(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """运行所有测试"""
        results = []
        
        for test_case in test_cases:
            result = await self.run_test(test_case)
            results.append(result)
        
        return {
            "total": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "failed": sum(1 for r in results if not r["passed"]),
            "results": results
        }
    
    def generate_test_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        return {
            "total_tests": len(self.test_results),
            "passed": sum(1 for r in self.test_results if r.get("passed")),
            "failed": sum(1 for r in self.test_results if not r.get("passed")),
            "avg_execution_time": sum(r.get("execution_time", 0) for r in self.test_results) / max(1, len(self.test_results)),
            "results": self.test_results
        }

class ConversationTestSuite:
    """对话测试套件"""
    
    def __init__(self, agent: ConversableAgent):
        self.agent = agent
        self.conversation_history: List[Dict] = []
    
    async def test_conversation(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """测试对话"""
        results = []
        
        for message in messages:
            start_time = time.time()
            
            try:
                response = await self.agent.a_generate_reply(message)
                execution_time = time.time() - start_time
                
                results.append({
                    "input": message.get("content", ""),
                    "output": response,
                    "execution_time": execution_time,
                    "success": True
                })
            
            except Exception as e:
                execution_time = time.time() - start_time
                results.append({
                    "input": message.get("content", ""),
                    "error": str(e),
                    "execution_time": execution_time,
                    "success": False
                })
        
        return {
            "total_messages": len(messages),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "avg_response_time": sum(r["execution_time"] for r in results) / max(1, len(results)),
            "results": results
        }
```

---

## 16.18 AutoGen 案例研究

### 16.18.1 大规模 AutoGen 部署

```python
class LargeScaleAutoGenDeployment:
    """大规模 AutoGen 部署案例"""
    
    def __init__(self):
        self.deployment_config = {
            "infrastructure": {
                "compute": "Kubernetes",
                "messaging": "Kafka",
                "storage": "PostgreSQL + Redis",
                "monitoring": "Prometheus + Grafana"
            },
            "agents": {
                "total": 50,
                "types": ["researcher", "analyst", "writer", "reviewer", "coordinator"]
            },
            "performance": {
                "concurrent_conversations": 1000,
                "avg_response_time": "2s",
                "availability": "99.99%"
            }
        }
    
    async def deploy(self) -> Dict[str, Any]:
        """执行部署"""
        deployment_steps = [
            {"step": "Infrastructure Setup", "status": "completed", "duration": "2 days"},
            {"step": "Agent Development", "status": "completed", "duration": "2 weeks"},
            {"step": "Integration Testing", "status": "completed", "duration": "1 week"},
            {"step": "Performance Testing", "status": "completed", "duration": "3 days"},
            {"step": "Production Deployment", "status": "completed", "duration": "1 day"},
            {"step": "Monitoring Setup", "status": "completed", "duration": "2 days"}
        ]
        
        return {
            "deployment_id": f"deploy_{int(time.time())}",
            "status": "completed",
            "steps": deployment_steps,
            "total_duration": "3.5 weeks",
            "results": {
                "concurrent_users": 10000,
                "response_time_p95": "2.5s",
                "error_rate": "0.1%",
                "cost_per_conversation": "$0.02"
            }
        }
    
    def get_architecture_overview(self) -> Dict[str, Any]:
        """获取架构概览"""
        return {
            "layers": {
                "presentation": "Web UI + Mobile App",
                "api": "FastAPI Gateway",
                "agents": "AutoGen Agent Pool",
                "services": "Microservices",
                "data": "PostgreSQL + Redis + S3"
            },
            "communication": {
                "sync": "REST API",
                "async": "Kafka",
                "real_time": "WebSocket"
            },
            "scalability": {
                "horizontal": "Kubernetes HPA",
                "vertical": "Resource limits",
                "cache": "Redis Cluster"
            }
        }
```

---

## 16.19 AutoGen 未来发展方向

### 16.19.1 技术趋势

```python
class AutoGenFutureTrends:
    """AutoGen 未来发展趋势"""
    
    @staticmethod
    def get发展趋势() -> List[Dict[str, str]]:
        """获取发展趋势"""
        return [
            {
                "trend": "多模态 Agent",
                "description": "支持文本、图像、音频等多种模态的 Agent",
                "impact": "扩展 Agent 的应用场景",
                "timeline": "2024-2025"
            },
            {
                "trend": "自主学习 Agent",
                "description": "能够从交互中自主学习和改进的 Agent",
                "impact": "提高 Agent 的适应能力",
                "timeline": "2025-2026"
            },
            {
                "trend": "Agent 协作网络",
                "description": "大规模 Agent 协作的网络化架构",
                "impact": "解决更复杂的问题",
                "timeline": "2024-2025"
            },
            {
                "trend": "安全和对齐",
                "description": "更强大的安全机制和价值观对齐",
                "impact": "提高 Agent 的可信度",
                "timeline": "持续发展"
            },
            {
                "trend": "边缘计算集成",
                "description": "在边缘设备上运行轻量级 Agent",
                "impact": "降低延迟，保护隐私",
                "timeline": "2025-2026"
            }
        ]
    
    @staticmethod
    def get_research方向() -> List[Dict[str, str]]:
        """获取研究方向"""
        return [
            {
                "direction": "长期记忆",
                "description": "更有效的长期记忆存储和检索机制",
                "challenge": "记忆的可靠性和效率"
            },
            {
                "direction": "推理能力",
                "description": "增强 Agent 的推理和规划能力",
                "challenge": "复杂问题的分解和求解"
            },
            {
                "direction": "多 Agent 协调",
                "description": "大规模 Agent 的高效协调机制",
                "challenge": "通信开销和一致性"
            },
            {
                "direction": "可解释性",
                "description": "提高 Agent 决策的可解释性",
                "challenge": "在性能和可解释性之间平衡"
            }
        ]
```

---

## 16.20 本章小结（最终版）

| 知识点 | 核心要点 |
|:---|:---|
| **AutoGen 架构** | ConversableAgent → GroupChat → CodeExecutor → Tools |
| **对话机制** | 消息处理、自动回复、终止条件 |
| **GroupChat** | 多 Agent 群聊、发言者选择策略、群聊监控 |
| **代码执行** | Docker 沙箱、代码审查、安全执行 |
| **工具集成** | 工具注册、工具执行、工具统计 |
| **状态管理** | 对话状态、共享内存、任务记忆 |
| **嵌套对话** | 层次结构、对话合并、结果整合 |
| **生产实践** | 代码审查系统、研究分析系统 |
| **自定义 Agent** | 专业化 Agent、Agent 工厂、团队组建 |
| **流程控制** | 对话流程、条件分支、并行执行 |
| **性能优化** | 缓存、批处理、并行、模型选择 |
| **安全实践** | 输入验证、输出过滤、速率限制 |
| **调试监控** | 事件日志、对话记录、断点调试 |
| **工具生态** | 内置工具、自定义工具、工具推荐 |
| **测试框架** | 单元测试、集成测试、对话测试 |
| **最佳实践** | 单一职责、松耦合、可观测性 |

---

## 16.21 常见问题和解决方案

### 16.21.1 AutoGen 问题排查

```python
from typing import Dict, List, Any

class AutoGenTroubleshooting:
    """AutoGen 问题排查指南"""
    
    @staticmethod
    def get_common_issues() -> List[Dict[str, Any]]:
        """获取常见问题"""
        return [
            {
                "issue": "Agent 无法生成回复",
                "symptoms": ["Agent 返回空响应", "对话陷入循环"],
                "causes": [
                    "system_message 配置不当",
                    "LLM 连接问题",
                    "消息格式错误",
                    "终止条件设置不当"
                ],
                "solutions": [
                    "检查 system_message 是否清晰",
                    "验证 LLM 配置和 API 密钥",
                    "确保消息格式正确",
                    "调整 max_consecutive_auto_reply"
                ]
            },
            {
                "issue": "GroupChat 混乱",
                "symptoms": ["多个 Agent 同时发言", "对话无序"],
                "causes": [
                    "speaker_selection_method 配置不当",
                    "Agent 角色重叠",
                    "缺少对话管理"
                ],
                "solutions": [
                    "使用 round_robin 或 manual 策略",
                    "明确每个 Agent 的职责",
                    "添加 GroupChatManager 进行协调"
                ]
            },
            {
                "issue": "代码执行失败",
                "symptoms": ["代码无法运行", "沙箱错误"],
                "causes": [
                    "Docker 未安装或未运行",
                    "代码语法错误",
                    "依赖缺失",
                    "权限问题"
                ],
                "solutions": [
                    "确保 Docker 已安装并运行",
                    "检查代码语法",
                    "在代码中安装必要依赖",
                    "检查文件权限"
                ]
            },
            {
                "issue": "性能问题",
                "symptoms": ["响应时间长", "吞吐量低"],
                "causes": [
                    "LLM 调用频繁",
                    "消息处理瓶颈",
                    "资源不足"
                ],
                "solutions": [
                    "实现响应缓存",
                    "优化消息处理逻辑",
                    "增加计算资源",
                    "使用更快的模型"
                ]
            }
        ]
    
    @staticmethod
    def get_debugging_tips() -> List[Dict[str, str]]:
        """获取调试技巧"""
        return [
            {
                "tip": "启用详细日志",
                "command": "设置 verbose=True",
                "description": "查看详细的对话过程"
            },
            {
                "tip": "使用断点调试",
                "command": "在关键位置设置断点",
                "description": "检查 Agent 状态和消息"
            },
            {
                "tip": "检查消息格式",
                "command": "打印消息内容",
                "description": "确保消息格式正确"
            },
            {
                "tip": "验证 LLM 配置",
                "command": "测试 LLM 连接",
                "description": "确保 LLM 服务可用"
            },
            {
                "tip": "监控资源使用",
                "command": "使用 top/htop 监控",
                "description": "检查 CPU 和内存使用"
            }
        ]
    
    @staticmethod
    def get_performance_checklist() -> List[Dict[str, str]]:
        """获取性能检查清单"""
        return [
            {"check": "LLM 模型选择", "status": "根据任务复杂度选择合适的模型"},
            {"check": "缓存策略", "status": "实现响应缓存减少重复调用"},
            {"check": "消息批处理", "status": "合并多个请求批量处理"},
            {"check": "并发控制", "status": "限制并发请求数量"},
            {"check": "资源监控", "status": "监控 CPU、内存和网络使用"},
            {"check": "错误处理", "status": "实现优雅的错误处理和重试"}
        ]
```

---

## 16.22 AutoGen 社区和资源

### 16.22.1 学习资源

```python
class AutoGenResources:
    """AutoGen 学习资源"""
    
    @staticmethod
    def get_official_resources() -> Dict[str, str]:
        """获取官方资源"""
        return {
            "documentation": "https://microsoft.github.io/autogen/",
            "github": "https://github.com/microsoft/autogen",
            "examples": "https://github.com/microsoft/autogen/tree/main/examples",
            "tutorials": "https://microsoft.github.io/autogen/docs/Tutorials",
            "api_reference": "https://microsoft.github.io/autogen/docs/reference",
            "discord": "https://discord.gg/pAbnRBQzTW",
            "forum": "https://github.com/microsoft/autogen/discussions"
        }
    
    @staticmethod
    def get_recommended_reading() -> List[Dict[str, str]]:
        """获取推荐阅读"""
        return [
            {
                "title": "Getting Started with AutoGen",
                "type": "tutorial",
                "description": "AutoGen 入门教程"
            },
            {
                "title": "Building Multi-Agent Systems",
                "type": "guide",
                "description": "构建多 Agent 系统指南"
            },
            {
                "title": "Advanced Agent Patterns",
                "type": "advanced",
                "description": "高级 Agent 模式"
            },
            {
                "title": "Production Deployment Guide",
                "type": "production",
                "description": "生产环境部署指南"
            }
        ]
    
    @staticmethod
    def get_community_projects() -> List[Dict[str, str]]:
        """获取社区项目"""
        return [
            {
                "name": "AutoGen Studio",
                "description": "可视化 AutoGen 界面",
                "url": "https://github.com/microsoft/autogen/tree/main/autogenstudio"
            },
            {
                "name": "AutoGen Extensions",
                "description": "AutoGen 扩展库",
                "url": "https://github.com/microsoft/autogen/tree/main/extensions"
            },
            {
                "name": "AutoGen Examples",
                "description": "示例项目集合",
                "url": "https://github.com/microsoft/autogen/tree/main/examples"
            }
        ]
```

---

## 16.23 本章小结（完整版）

| 知识点 | 核心要点 |
|:---|:---|
| **AutoGen 架构** | ConversableAgent → GroupChat → CodeExecutor → Tools |
| **对话机制** | 消息处理、自动回复、终止条件 |
| **GroupChat** | 多 Agent 群聊、发言者选择策略、群聊监控 |
| **代码执行** | Docker 沙箱、代码审查、安全执行 |
| **工具集成** | 工具注册、工具执行、工具统计 |
| **状态管理** | 对话状态、共享内存、任务记忆 |
| **嵌套对话** | 层次结构、对话合并、结果整合 |
| **生产实践** | 代码审查系统、研究分析系统 |
| **自定义 Agent** | 专业化 Agent、Agent 工厂、团队组建 |
| **流程控制** | 对话流程、条件分支、并行执行 |
| **性能优化** | 缓存、批处理、并行、模型选择 |
| **安全实践** | 输入验证、输出过滤、速率限制 |
| **调试监控** | 事件日志、对话记录、断点调试 |
| **工具生态** | 内置工具、自定义工具、工具推荐 |
| **测试框架** | 单元测试、集成测试、对话测试 |
| **问题排查** | 常见问题、调试技巧、性能检查 |
| **社区资源** | 官方文档、学习资源、社区项目 |

> **下一章预告**
>
> 在第 17 章中，我们将学习 CrewAI。
