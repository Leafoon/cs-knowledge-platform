---
title: "第25章：安全与对齐 — Agent 的安全护栏"
description: "全面掌握 Agent 安全体系：提示注入防御、工具权限控制、输出过滤、沙箱执行、Constitutional AI 与红队测试。"
date: "2026-06-11"
---

# 第25章：安全与对齐 — Agent 的安全护栏

---

## 25.1 安全威胁模型

| 威胁类型 | 描述 | 风险等级 |
|:---|:---|:---|
| 提示注入 | 恶意输入覆盖系统指令 | 高 |
| 间接注入 | 通过工具返回值注入 | 很高 |
| 工具滥用 | Agent 调用危险工具 | 高 |
| 数据泄露 | Agent 输出敏感信息 | 高 |

---

## 25.2 多层防御

```python
class InputGuard:
    def __init__(self):
        self.patterns = [r"忽略.*指令", r"ignore.*instruction", r"你现在是"]
    def check(self, user_input):
        import re
        for p in self.patterns:
            if re.search(p, user_input, re.IGNORECASE): return False, "检测到注入"
        return True, "安全"
```

---

## 25.3 工具层防御

```python
class ToolGuard:
    def __init__(self):
        self.dangerous = {"delete_file", "drop_table", "send_email"}
        self.require_approval = {"write_file", "execute_code"}
    def check(self, tool_name):
        if tool_name in self.dangerous: return "blocked"
        if tool_name in self.require_approval: return "require_approval"
        return "allowed"
```

---

## 25.4 输出层防御

```python
class OutputGuard:
    def check(self, output):
        import re
        for p in [r"api[_\s]*key", r"password", r"secret", r"token"]:
            if re.search(p, output, re.IGNORECASE): return False, "可能包含敏感信息"
        return True, "安全"
```

---

## 25.5 沙箱执行

```python
class SandboxExecutor:
    def execute(self, code, timeout=30):
        import subprocess, tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code); temp_path = f.name
        try:
            result = subprocess.run(["python3", temp_path], capture_output=True, text=True, timeout=timeout)
            return {"status": "success" if result.returncode == 0 else "error", "output": result.stdout}
        finally: os.unlink(temp_path)
```

---

## 25.6 Constitutional AI

```python
CONSTITUTIONAL_PRINCIPLES = [
    "Agent 不应生成有害、不道德或非法的内容",
    "Agent 不应泄露用户的私人信息",
    "Agent 不应执行可能损害系统的操作",
    "Agent 应在不确定时请求人类确认",
]
```

---

## 25.7 红队测试

```python
class RedTeamTester:
    def run_tests(self, agent, test_cases):
        results = {"passed": 0, "failed": 0}
        for case in test_cases:
            response = agent.run(case["input"])
            passed = self._check_refused(response) if case["should_refuse"] else True
            if passed: results["passed"] += 1
            else: results["failed"] += 1
        results["success_rate"] = results["passed"] / len(test_cases)
        return results
```

---

## 25.8 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| 威胁模型 | 注入、滥用、泄露 |
| 多层防御 | 输入→工具→输出→沙箱 |
| 沙箱 | 代码隔离 |
| Constitutional AI | 基于原则的自我约束 |

> **下一章预告**
>
> 在第 26 章中，我们将学习 Agent 评测。
