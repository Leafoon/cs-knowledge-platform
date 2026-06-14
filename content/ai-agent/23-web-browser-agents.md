---
title: "第23章：Web 浏览器 Agent — 自动化操作"
description: "掌握 Web Agent 的设计与实现：Playwright 页面理解、DOM 操作、视觉定位、表单填写、导航规划与 WebArena 评测。"
date: "2026-06-11"
---

# 第23章：Web 浏览器 Agent — 自动化操作

---

## 23.1 Web Agent 架构

```python
from playwright.async_api import async_playwright

class WebAgent:
    def __init__(self, llm):
        self.llm = llm
        self.browser = None
        self.page = None

    async def start(self):
        pw = await async_playwright().start()
        self.browser = await pw.chromium.launch(headless=False)
        self.page = await self.browser.new_page()

    async def run_task(self, task):
        for step in range(10):
            page_info = await self.get_page_content()
            response = self.llm.invoke([HumanMessage(content=f"页面：{page_info['title']}\n任务：{task}\n下一步操作：")])
            action = response.content.strip()
            if "done" in action.lower(): return action
            await self.execute_action(action)
```

---

## 23.2 视觉 Web Agent

```python
class VisualWebAgent:
    async def run_task(self, task):
        for step in range(10):
            screenshot = await self.page.screenshot()
            import base64
            screenshot_b64 = base64.b64encode(screenshot).decode()
            response = self.llm.invoke([HumanMessage(content=[
                {"type": "text", "text": f"任务：{task}\n请描述页面并决定操作。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
            ])])
```

---

## 23.3 框架对比

| 框架 | 特点 | 适用场景 |
|:---|:---|:---|
| Playwright | 全功能 | 通用 Web |
| Browser Use | LLM 驱动 | AI Agent |
| Selenium | 经典 | 兼容旧系统 |

---

## 23.4 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| Web Agent | 自主操作浏览器 |
| DOM 操作 | 点击、输入、导航 |
| 视觉 Agent | 基于截图的多模态理解 |

> **下一章预告**
>
> 在第 24 章中，我们将学习数据分析 Agent。
