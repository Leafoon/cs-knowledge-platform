---
title: "第13章：Agent 架构设计 — 单体到分布式"
description: "系统梳理 Agent 架构设计模式：单 Agent 循环、状态机 Agent、事件驱动 Agent、分层 Agent、多 Agent 主从/对等架构与微服务架构。"
date: "2026-06-11"
---

 # 第13章：Agent 架构设计 — 单体到分布式

Plan-and-Execute 是一种常见的 Agent 架构模式。下面的交互式演示展示了完整的执行流程：

<div data-component="PlanExecuteFlow"></div>

不同的 Agent 实现方式各有优劣。下面的交互式对比可以帮助你选择合适的方案：

<div data-component="AgentArchitectureComparisonV9"></div>

不同的推理策略适用于不同的场景。下面的交互式工具可以帮助你选择最合适的推理策略：

<div data-component="ReasoningStrategySelectorV11"></div>

 > **下一章预告**
 >
 > 在第 14 章中，我们将深入 LangChain/LangGraph Agent 实战，通过具体案例学习如何构建高效的Agent系统。
