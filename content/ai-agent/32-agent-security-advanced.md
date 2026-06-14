---
title: "第32章：Agent 安全工程进阶"
description: "深入 Agent 安全工程：间接提示注入、工具链攻击面分析、权限最小化、审计日志、合规框架与安全开发生命周期。"
date: "2026-06-11"
---

# 第32章：Agent 安全工程进阶

---

## 32.1 间接提示注入

```python
class IndirectInjectionDefense:
    def sanitize(self, content):
        import re
        patterns = [r'ignore\s+(previous|all)\s+instructions', r'忽略.*指令', r'system\s*prompt']
        for p in patterns:
            if re.search(p, content, re.IGNORECASE):
                return "[内容安全检查未通过，已过滤]"
        return content
```

---

## 32.2 审计日志

```python
import json
from datetime import datetime

class AgentAuditLogger:
    def log_tool_call(self, agent_id, tool_name, args, result):
        import logging
        logger = logging.getLogger("agent_audit")
        logger.info(json.dumps({"event": "tool_call", "agent_id": agent_id,
                               "tool": tool_name, "timestamp": datetime.now().isoformat()}))
```

---

## 32.3 合规框架

| 框架 | Agent 相关要求 |
|:---|:---|
| SOC2 | 访问控制、审计日志、变更管理 |
| GDPR | 数据最小化、用户同意、删除权 |
| HIPAA | 健康数据保护、访问审计 |

---

## 32.4 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| 间接注入 | 外部数据可能包含恶意指令 |
| 审计日志 | 记录所有 Agent 行为 |
| 合规框架 | SOC2/GDPR/HIPAA |

> **下一章预告**
>
> 在第 33 章中，我们将学习企业级 Agent 实战案例。
