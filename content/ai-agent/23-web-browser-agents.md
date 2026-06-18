---
title: "第23章：Web 浏览器 Agent — 自动化操作"
description: "掌握 Web Agent 的设计与实现：Playwright 页面理解、DOM 操作、视觉定位、表单填写、导航规划与 WebArena 评测。"
date: "2026-06-11"
---

# 第23章：Web 浏览器 Agent — 自动化操作

 Web Agent 能够像人类一样自主操作浏览器，完成网页浏览、表单填写、数据提取等任务。本章深入讲解 Web Agent 的架构设计、核心技术实现和生产实践。

 下面的交互式演示展示了 Web Agent 的工作流程：

 <div data-component="WebAgentFlow"></div>

 ## 什么是 Web Agent？

Web Agent 是一种能够自主操作浏览器的 AI 系统。与传统的浏览器自动化工具（如 Selenium）不同，Web Agent 具有以下特点：

**1. 自主决策能力**
传统自动化需要预先编写脚本，而 Web Agent 可以根据任务目标自主决定操作步骤。例如，用户说"帮我在淘宝上搜索 iPhone 并找到最便宜的"，Agent 会自动完成搜索、筛选、比较等一系列操作。

**2. 理解能力**
Web Agent 不仅仅是点击按钮和输入文字，它能够理解页面内容，包括：
- 文本内容和语义
- 页面布局和结构
- 交互元素的功能
- 视觉信息（如图片、图表）

**3. 适应能力**
当页面结构发生变化时，Web Agent 能够自动调整策略。例如，如果按钮的 ID 改变了，它可以尝试通过文本、位置或其他属性找到按钮。

**4. 错误恢复能力**
操作失败时，Web Agent 能够自动重试或尝试替代方案。

## Web Agent 的应用场景

Web Agent 在以下场景中非常有价值：

**数据采集**：从各种网站采集结构化数据，如电商价格、新闻内容、招聘信息等。

**表单自动化**：自动填写复杂的表单，如申请表、注册表、订单表等。

**测试自动化**：进行端到端的功能测试，模拟真实用户操作。

**流程自动化**：自动化重复性的网页操作，如批量发布内容、数据同步等。

**辅助浏览**：帮助用户完成复杂的网页任务，如比价、预订、搜索等。

---

## 23.1 Web Agent 架构设计

### 23.1.1 整体架构

一个完整的 Web Agent 系统通常包含以下几个核心组件：

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Agent 架构                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Task      │    │   Planner   │    │   Executor  │     │
│  │   Parser    │───▶│   (LLM)     │───▶│  (Browser)  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Goal       │    │   Action    │    │  DOM/Screenshot│   │
│  │  Extractor  │    │   Sequence  │    │  Understanding │  │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│                    ┌─────────────┐                          │
│                    │  Memory &   │                          │
│                    │  State      │                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

**Task Parser（任务解析器）**：将用户的自然语言任务转换为结构化的任务描述。例如，"帮我在淘宝搜索 iPhone"会被解析为：目标网站=淘宝，操作=搜索，关键词=iPhone。

**Planner（规划器）**：使用 LLM 根据当前页面状态和任务目标，规划下一步操作。这是 Web Agent 的"大脑"，负责决策。

**Executor（执行器）**：实际执行浏览器操作，如点击、输入、滚动等。通常使用 Playwright 或 Selenium 实现。

**DOM/Screenshot Understanding（页面理解）**：解析页面内容，提取有用信息，如可点击的元素、文本内容、图片等。

**Memory & State（记忆和状态）**：维护任务执行过程中的状态信息，包括已完成的步骤、遇到的问题等。

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Agent 架构                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Task      │    │   Planner   │    │   Executor  │     │
│  │   Parser    │───▶│   (LLM)     │───▶│  (Browser)  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Goal       │    │   Action    │    │  DOM/Screenshot│   │
│  │  Extractor  │    │   Sequence  │    │  Understanding │  │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│                    ┌─────────────┐                          │
│                    │  Memory &   │                          │
│                    │  State      │                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### 23.1.2 核心组件

```python
from playwright.async_api import async_playwright, Page, Browser
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import base64
import json

@dataclass
class WebAction:
    """网页操作定义"""
    action_type: str  # click, type, select, scroll, navigate, wait
    selector: Optional[str] = None
    value: Optional[str] = None
    url: Optional[str] = None
    timeout: int = 5000

@dataclass
class PageState:
    """页面状态"""
    url: str
    title: str
    text_content: str
    interactive_elements: List[Dict[str, Any]]
    screenshot: Optional[str] = None  # base64 encoded

class WebAgent:
    """Web Agent 核心类"""
    
    def __init__(self, llm, config: Optional[Dict] = None):
        self.llm = llm
        self.config = config or {}
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.action_history: List[WebAction] = []
        self.state_history: List[PageState] = []
    
    async def start(self, headless: bool = True):
        """启动浏览器"""
        pw = await async_playwright().start()
        self.browser = await pw.chromium.launch(
            headless=headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage'
            ]
        )
        context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        self.page = await context.new_page()
    
    async def stop(self):
        """关闭浏览器"""
        if self.browser:
            await self.browser.close()
    
    async def navigate(self, url: str) -> PageState:
        """导航到指定 URL"""
        await self.page.goto(url, wait_until='networkidle')
        return await self.get_page_state()
    
    async def get_page_state(self) -> PageState:
        """获取当前页面状态"""
        url = self.page.url
        title = await self.page.title()
        
        # 获取文本内容
        text_content = await self.page.evaluate('''() => {
            const walker = document.createTreeWalker(
                document.body,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );
            let text = '';
            let node;
            while (node = walker.nextNode()) {
                const parent = node.parentElement;
                if (parent && 
                    !['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(parent.tagName) &&
                    parent.offsetParent !== null) {
                    text += node.textContent.trim() + ' ';
                }
            }
            return text.replace(/\\s+/g, ' ').trim();
        }''')
        
        # 获取交互元素
        interactive_elements = await self.page.evaluate('''() => {
            const selectors = 'a, button, input, select, textarea, [role="button"], [onclick]';
            const elements = document.querySelectorAll(selectors);
            const result = [];
            
            elements.forEach((el, index) => {
                if (el.offsetParent !== null) {  // 可见元素
                    const rect = el.getBoundingClientRect();
                    result.push({
                        index: index,
                        tag: el.tagName.toLowerCase(),
                        type: el.type || '',
                        text: (el.innerText || el.value || el.placeholder || '').substring(0, 100),
                        id: el.id || '',
                        name: el.name || '',
                        href: el.href || '',
                        x: Math.round(rect.x + rect.width / 2),
                        y: Math.round(rect.y + rect.height / 2),
                        visible: true
                    });
                }
            });
            
            return result;
        }''')
        
        state = PageState(
            url=url,
            title=title,
            text_content=text_content[:5000],  # 限制长度
            interactive_elements=interactive_elements
        )
        
        self.state_history.append(state)
        return state
    
    async def take_screenshot(self) -> str:
        """获取页面截图"""
        screenshot = await self.page.screenshot(type='png')
        return base64.b64encode(screenshot).decode()
    
    async def execute_action(self, action: WebAction) -> bool:
        """执行操作"""
        try:
            if action.action_type == 'click':
                await self.page.click(action.selector, timeout=action.timeout)
            
            elif action.action_type == 'type':
                await self.page.fill(action.selector, action.value, timeout=action.timeout)
            
            elif action.action_type == 'select':
                await self.page.select_option(action.selector, action.value, timeout=action.timeout)
            
            elif action.action_type == 'scroll':
                direction = action.value or 'down'
                if direction == 'down':
                    await self.page.mouse.wheel(0, 500)
                else:
                    await self.page.mouse.wheel(0, -500)
            
            elif action.action_type == 'navigate':
                await self.navigate(action.url)
            
            elif action.action_type == 'wait':
                await self.page.wait_for_timeout(int(action.value or 1000))
            
            self.action_history.append(action)
            return True
            
        except Exception as e:
            print(f"Action failed: {e}")
            return False
    
    async def run_task(self, task: str, max_steps: int = 15) -> Dict[str, Any]:
        """执行任务"""
        for step in range(max_steps):
            # 获取当前页面状态
            page_state = await self.get_page_state()
            
            # 构建 prompt
            prompt = self._build_prompt(task, page_state, step)
            
            # 调用 LLM 决策
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            action_plan = self._parse_response(response.content)
            
            # 检查是否完成
            if action_plan.get('done'):
                return {
                    'success': True,
                    'result': action_plan.get('result', ''),
                    'steps': step + 1,
                    'action_history': self.action_history
                }
            
            # 执行操作
            action = WebAction(
                action_type=action_plan['action'],
                selector=action_plan.get('selector'),
                value=action_plan.get('value'),
                url=action_plan.get('url')
            )
            
            success = await self.execute_action(action)
            if not success:
                continue
            
            # 等待页面更新
            await self.page.wait_for_timeout(1000)
        
        return {
            'success': False,
            'error': 'Max steps exceeded',
            'steps': max_steps,
            'action_history': self.action_history
        }
    
    def _build_prompt(self, task: str, state: PageState, step: int) -> str:
        """构建 LLM prompt"""
        elements_desc = "\n".join([
            f"[{el['index']}] <{el['tag']}> {el['text'][:50]}"
            for el in state.interactive_elements[:15]
        ])
        
        return f"""
你是一个 Web 浏览器 Agent，需要完成以下任务：

任务：{task}

当前页面：
- URL: {state.url}
- 标题: {state.title}
- 页面文本: {state.text_content[:1000]}

可用交互元素：
{elements_desc}

已执行步骤: {step}

请根据当前页面状态，决定下一步操作。返回 JSON 格式：
{{
    "action": "click|type|select|scroll|navigate|wait|done",
    "selector": "CSS选择器或元素索引",
    "value": "输入值或选项",
    "reason": "决策理由",
    "done": false,
    "result": "任务完成时的结果"
}}
"""
    
    def _parse_response(self, response: str) -> Dict:
        """解析 LLM 响应"""
        try:
            # 尝试提取 JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # 默认返回
        return {'action': 'wait', 'value': '1000', 'reason': '无法解析响应'}
```

---

## 23.2 页面理解技术

### 23.2.1 DOM 结构分析

```python
class DOMAnalyzer:
    """DOM 结构分析器"""
    
    def __init__(self, page: Page):
        self.page = page
    
    async def analyze_page_structure(self) -> Dict[str, Any]:
        """分析页面结构"""
        return await self.page.evaluate('''() => {
            const analyze = (element, depth = 0) => {
                if (!element || depth > 3) return null;
                
                const info = {
                    tag: element.tagName?.toLowerCase(),
                    id: element.id || undefined,
                    className: element.className || undefined,
                    role: element.getAttribute('role') || undefined,
                    text: element.innerText?.substring(0, 100) || undefined,
                    children: []
                };
                
                if (element.children) {
                    Array.from(element.children).forEach(child => {
                        const childInfo = analyze(child, depth + 1);
                        if (childInfo) info.children.push(childInfo);
                    });
                }
                
                return info;
            };
            
            return analyze(document.body);
        }''')
    
    async def extract_semantic_structure(self) -> Dict[str, Any]:
        """提取语义结构"""
        return await self.page.evaluate('''() => {
            const structure = {
                navigation: [],
                main_content: [],
                forms: [],
                tables: [],
                links: []
            };
            
            // 导航
            document.querySelectorAll('nav, [role="navigation"]').forEach(el => {
                structure.navigation.push({
                    text: el.innerText.substring(0, 200),
                    links: Array.from(el.querySelectorAll('a')).map(a => ({
                        text: a.innerText,
                        href: a.href
                    }))
                });
            });
            
            // 主要内容
            document.querySelectorAll('main, article, [role="main"]').forEach(el => {
                structure.main_content.push({
                    text: el.innerText.substring(0, 500),
                    headings: Array.from(el.querySelectorAll('h1, h2, h3')).map(h => h.innerText)
                });
            });
            
            // 表单
            document.querySelectorAll('form').forEach(form => {
                structure.forms.push({
                    action: form.action,
                    method: form.method,
                    fields: Array.from(form.querySelectorAll('input, select, textarea')).map(field => ({
                        name: field.name,
                        type: field.type,
                        placeholder: field.placeholder,
                        required: field.required
                    }))
                });
            });
            
            // 表格
            document.querySelectorAll('table').forEach(table => {
                const headers = Array.from(table.querySelectorAll('th')).map(th => th.innerText);
                const rows = Array.from(table.querySelectorAll('tbody tr')).slice(0, 5).map(row =>
                    Array.from(row.querySelectorAll('td')).map(td => td.innerText)
                );
                structure.tables.push({ headers, rows });
            });
            
            return structure;
        }''')
    
    async def find_interactive_elements(self) -> List[Dict[str, Any]]:
        """查找所有可交互元素"""
        return await self.page.evaluate('''() => {
            const elements = [];
            const selectors = [
                'a[href]',
                'button',
                'input:not([type="hidden"])',
                'select',
                'textarea',
                '[role="button"]',
                '[role="link"]',
                '[role="tab"]',
                '[onclick]',
                '[tabindex]'
            ].join(', ');
            
            document.querySelectorAll(selectors).forEach((el, index) => {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                
                if (rect.width > 0 && rect.height > 0 && 
                    style.display !== 'none' && style.visibility !== 'hidden') {
                    
                    elements.push({
                        index: index,
                        tag: el.tagName.toLowerCase(),
                        type: el.type || '',
                        text: (el.innerText || el.value || el.placeholder || '').substring(0, 100),
                        id: el.id || '',
                        name: el.name || '',
                        href: el.href || '',
                        ariaLabel: el.getAttribute('aria-label') || '',
                        role: el.getAttribute('role') || '',
                        x: Math.round(rect.x + rect.width / 2),
                        y: Math.round(rect.y + rect.height / 2),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height),
                        clickable: el.tagName === 'A' || el.tagName === 'BUTTON' || el.onclick !== null
                    });
                }
            });
            
            return elements;
        }''')
```

### 23.2.2 智能元素定位

```python
class SmartLocator:
    """智能元素定位器"""
    
    def __init__(self, page: Page):
        self.page = page
    
    async def find_element(self, description: str) -> Optional[str]:
        """根据描述查找元素"""
        # 尝试多种定位策略
        strategies = [
            self._by_text,
            self._by_role,
            self._by_aria_label,
            self._by_placeholder,
            self._by_id,
            self._by_css
        ]
        
        for strategy in strategies:
            selector = await strategy(description)
            if selector:
                return selector
        
        return None
    
    async def _by_text(self, text: str) -> Optional[str]:
        """通过文本查找"""
        return await self.page.evaluate(f'''() => {{
            const elements = document.querySelectorAll('a, button, span, div, p');
            for (const el of elements) {{
                if (el.innerText.includes('{text}') && el.offsetParent !== null) {{
                    return el.id ? `#${{el.id}}` : null;
                }}
            }}
            return null;
        }}''')
    
    async def _by_role(self, role: str) -> Optional[str]:
        """通过角色查找"""
        return f'[role="{role}"]'
    
    async def _by_aria_label(self, label: str) -> Optional[str]:
        """通过 aria-label 查找"""
        return f'[aria-label*="{label}"]'
    
    async def _by_placeholder(self, placeholder: str) -> Optional[str]:
        """通过 placeholder 查找"""
        return f'[placeholder*="{placeholder}"]'
    
    async def _by_id(self, id_part: str) -> Optional[str]:
        """通过 ID 查找"""
        return f'[id*="{id_part}"]'
    
    async def _by_css(self, description: str) -> Optional[str]:
        """尝试 CSS 选择器"""
        # 将自然语言转换为 CSS 选择器
        mappings = {
            '按钮': 'button',
            '链接': 'a',
            '输入框': 'input',
            '下拉菜单': 'select',
            '文本域': 'textarea'
        }
        
        for keyword, selector in mappings.items():
            if keyword in description:
                return selector
        
        return None
    
    async def get_element_info(self, selector: str) -> Dict[str, Any]:
        """获取元素详细信息"""
        return await self.page.evaluate(f'''(selector) => {{
            const el = document.querySelector(selector);
            if (!el) return null;
            
            const rect = el.getBoundingClientRect();
            const style = window.getComputedStyle(el);
            
            return {{
                tag: el.tagName.toLowerCase(),
                id: el.id,
                className: el.className,
                text: el.innerText?.substring(0, 200),
                value: el.value,
                disabled: el.disabled,
                visible: rect.width > 0 && rect.height > 0 && style.display !== 'none',
                x: Math.round(rect.x + rect.width / 2),
                y: Math.round(rect.y + rect.height / 2)
            }};
        }}''', selector)
```

---

## 23.3 视觉理解

### 23.3.1 多模态页面理解

```python
class VisualPageUnderstanding:
    """视觉页面理解"""
    
    def __init__(self, vision_model):
        self.vision_model = vision_model
    
    async def analyze_screenshot(self, screenshot_b64: str, task: str) -> Dict[str, Any]:
        """分析页面截图"""
        prompt = f"""
请分析这个网页截图，完成以下任务：
任务：{task}

请提供：
1. 页面布局描述
2. 关键元素位置（坐标）
3. 推荐的操作序列
4. 潜在的挑战

以 JSON 格式返回：
{{
    "layout_description": "页面布局描述",
    "key_elements": [
        {{
            "description": "元素描述",
            "x": 100,
            "y": 200,
            "type": "button|link|input|text"
        }}
    ],
    "recommended_actions": [
        {{
            "action": "click|type|scroll",
            "target": "元素描述",
            "value": "输入值（如果适用）"
        }}
    ],
    "challenges": ["潜在挑战列表"]
}}
"""
        
        response = await self.vision_model.ainvoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
            ])
        ])
        
        return self._parse_json_response(response.content)
    
    async def locate_element_by_visual(self, screenshot_b64: str, element_description: str) -> Optional[Dict]:
        """通过视觉定位元素"""
        prompt = f"""
请在截图中找到以下元素：{element_description}

返回该元素的中心坐标 (x, y)，格式为 JSON：
{{"x": 100, "y": 200, "confidence": 0.95}}
"""
        
        response = await self.vision_model.ainvoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
            ])
        ])
        
        return self._parse_json_response(response.content)
    
    def _parse_json_response(self, text: str) -> Dict:
        """解析 JSON 响应"""
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {}
```

### 23.3.2 视觉-文本融合

```python
class MultimodalAgent:
    """多模态 Web Agent"""
    
    def __init__(self, llm, vision_model):
        self.llm = llm
        self.vision_model = vision_model
    
    async def decide_action(self, page_state: PageState, screenshot: str, task: str) -> Dict:
        """融合视觉和文本信息决策"""
        
        # 文本分析
        text_analysis = await self._analyze_text(page_state, task)
        
        # 视觉分析
        visual_analysis = await self._analyze_visual(screenshot, task)
        
        # 融合决策
        combined_prompt = f"""
基于文本和视觉分析，决定下一步操作。

文本分析：
{json.dumps(text_analysis, ensure_ascii=False, indent=2)}

视觉分析：
{json.dumps(visual_analysis, ensure_ascii=False, indent=2)}

任务：{task}

请综合以上分析，返回最终决策：
{{
    "action": "click|type|scroll|navigate|done",
    "selector": "CSS选择器",
    "value": "输入值",
    "reason": "决策理由",
    "confidence": 0.9
}}
"""
        
        response = await self.llm.ainvoke([HumanMessage(content=combined_prompt)])
        return self._parse_json_response(response.content)
    
    async def _analyze_text(self, state: PageState, task: str) -> Dict:
        """分析文本信息"""
        prompt = f"""
分析页面文本，找出与任务相关的信息。

页面标题：{state.title}
页面文本：{state.text_content[:2000]}

任务：{task}

返回 JSON：
{{
    "relevant_info": ["相关信息列表"],
    "suggested_elements": ["建议交互的元素"],
    "potential_issues": ["潜在问题"]
}}
"""
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return self._parse_json_response(response.content)
    
    async def _analyze_visual(self, screenshot: str, task: str) -> Dict:
        """分析视觉信息"""
        prompt = f"""
分析页面截图，找出关键视觉元素。

任务：{task}

返回 JSON：
{{
    "layout": "页面布局描述",
    "key_elements": ["关键视觉元素列表"],
    "visual_cues": ["视觉提示列表"]
}}
"""
        response = await self.vision_model.ainvoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot}"}}
            ])
        ])
        return self._parse_json_response(response.content)
```

---

## 23.4 导航策略

### 23.4.1 任务规划

```python
class TaskPlanner:
    """任务规划器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def plan_task(self, task: str, initial_state: PageState) -> List[Dict]:
        """规划任务执行步骤"""
        prompt = f"""
为以下任务制定执行计划：

任务：{task}

初始页面状态：
- URL: {initial_state.url}
- 标题: {initial_state.title}

请返回步骤计划：
{{
    "goal": "最终目标",
    "sub_goals": ["子目标列表"],
    "steps": [
        {{
            "step": 1,
            "action": "操作类型",
            "target": "目标元素",
            "expected_result": "预期结果"
        }}
    ],
    "fallback_strategies": ["备选策略"]
}}
"""
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return self._parse_json_response(response.content).get('steps', [])
    
    async def replan(self, current_state: PageState, original_plan: List, error: str) -> List[Dict]:
        """重新规划"""
        prompt = f"""
任务执行遇到问题，需要重新规划。

原始计划：
{json.dumps(original_plan, ensure_ascii=False, indent=2)}

当前状态：
- URL: {current_state.url}
- 页面内容: {current_state.text_content[:1000]}

错误信息：{error}

请提供修正后的计划。
"""
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return self._parse_json_response(response.content).get('steps', [])
```

### 23.4.2 智能导航

```python
class SmartNavigator:
    """智能导航器"""
    
    def __init__(self, page: Page):
        self.page = page
    
    async def navigate_to_goal(self, goal_url: str) -> bool:
        """导航到目标 URL"""
        current_url = self.page.url
        
        # 检查是否已经在目标页面
        if goal_url in current_url:
            return True
        
        # 尝试直接导航
        try:
            await self.page.goto(goal_url, wait_until='networkidle', timeout=10000)
            return True
        except:
            pass
        
        # 尝试通过搜索找到目标
        return await self._navigate_via_search(goal_url)
    
    async def _navigate_via_search(self, goal_url: str) -> bool:
        """通过搜索导航"""
        # 查找搜索框
        search_selectors = [
            'input[type="search"]',
            'input[name="q"]',
            'input[name="search"]',
            'input[placeholder*="搜索"]',
            'input[placeholder*="search"]'
        ]
        
        for selector in search_selectors:
            search_input = await self.page.query_selector(selector)
            if search_input:
                await search_input.fill(goal_url)
                await search_input.press('Enter')
                await self.page.wait_for_load_state('networkidle')
                return True
        
        return False
    
    async def handle_popups(self):
        """处理弹窗"""
        try:
            # 关闭常见的弹窗
            close_selectors = [
                'button[aria-label="Close"]',
                'button[aria-label="关闭"]',
                '.modal-close',
                '[data-dismiss="modal"]'
            ]
            
            for selector in close_selectors:
                close_button = await self.page.query_selector(selector)
                if close_button:
                    await close_button.click()
                    await self.page.wait_for_timeout(500)
        except:
            pass
    
    async def wait_for_content(self, content_type: str = 'networkidle', timeout: int = 10000):
        """等待内容加载"""
        try:
            await self.page.wait_for_load_state(content_type, timeout=timeout)
        except:
            await self.page.wait_for_timeout(2000)
```

---

## 23.5 错误处理和恢复

### 23.5.1 错误检测

```python
class ErrorDetector:
    """错误检测器"""
    
    def __init__(self, page: Page):
        self.page = page
    
    async def detect_errors(self) -> List[Dict[str, str]]:
        """检测页面错误"""
        errors = []
        
        # 检查 HTTP 错误页面
        http_errors = await self._check_http_errors()
        errors.extend(http_errors)
        
        # 检查 JavaScript 错误
        js_errors = await self._check_js_errors()
        errors.extend(js_errors)
        
        # 检查页面内容错误
        content_errors = await self._check_content_errors()
        errors.extend(content_errors)
        
        return errors
    
    async def _check_http_errors(self) -> List[Dict]:
        """检查 HTTP 错误"""
        title = await self.page.title()
        error_patterns = ['404', '500', 'Error', 'Not Found', 'Forbidden']
        
        for pattern in error_patterns:
            if pattern.lower() in title.lower():
                return [{'type': 'http_error', 'message': f'HTTP Error: {title}'}]
        
        return []
    
    async def _check_js_errors(self) -> List[Dict]:
        """检查 JavaScript 错误"""
        errors = await self.page.evaluate('''() => {
            const errors = [];
            
            // 检查控制台错误
            if (window.console && window.console.errors) {
                errors.push(...window.console.errors.map(e => e.message));
            }
            
            // 检查未捕获的异常
            window.onerror = function(msg) {
                errors.push(msg);
            };
            
            return errors;
        }''')
        
        return [{'type': 'js_error', 'message': err} for err in errors]
    
    async def _check_content_errors(self) -> List[Dict]:
        """检查内容错误"""
        text = await self.page.inner_text('body')
        
        error_patterns = [
            '加载失败',
            '加载失败',
            '网络错误',
            '请稍后重试',
            '系统错误'
        ]
        
        for pattern in error_patterns:
            if pattern in text:
                return [{'type': 'content_error', 'message': f'Content Error: {pattern}'}]
        
        return []
```

### 23.5.2 恢复策略

```python
class RecoveryManager:
    """恢复管理器"""
    
    def __init__(self, agent: WebAgent):
        self.agent = agent
        self.max_retries = 3
    
    async def handle_error(self, error: Dict, task: str) -> bool:
        """处理错误并尝试恢复"""
        error_type = error['type']
        
        if error_type == 'http_error':
            return await self._handle_http_error(error)
        
        elif error_type == 'timeout':
            return await self._handle_timeout(error)
        
        elif error_type == 'element_not_found':
            return await self._handle_element_not_found(error, task)
        
        return False
    
    async def _handle_http_error(self, error: Dict) -> bool:
        """处理 HTTP 错误"""
        # 尝试刷新页面
        try:
            await self.agent.page.reload(wait_until='networkidle')
            return True
        except:
            return False
    
    async def _handle_timeout(self, error: Dict) -> bool:
        """处理超时"""
        # 等待一段时间后重试
        await self.agent.page.wait_for_timeout(3000)
        return True
    
    async def _handle_element_not_found(self, error: Dict, task: str) -> bool:
        """处理元素未找到"""
        # 尝试滚动页面
        await self.agent.page.mouse.wheel(0, 500)
        await self.agent.page.wait_for_timeout(1000)
        return True
    
    async def rollback(self, steps: int = 1):
        """回滚操作"""
        for _ in range(steps):
            if self.agent.action_history:
                last_action = self.agent.action_history.pop()
                if last_action.action_type == 'click':
                    await self.agent.page.go_back()
                    await self.agent.page.wait_for_load_state('networkidle')
```

---

## 23.6 生产实践案例

### 23.6.1 电商数据采集

```python
class EcommerceScraper:
    """电商数据采集 Agent"""
    
    def __init__(self, llm):
        self.agent = WebAgent(llm)
        self.llm = llm
    
    async def scrape_products(self, url: str, max_products: int = 50) -> List[Dict]:
        """采集商品数据"""
        await self.agent.start(headless=True)
        
        try:
            await self.agent.navigate(url)
            
            products = []
            page_num = 1
            
            while len(products) < max_products:
                # 提取当前页面商品
                page_products = await self._extract_products()
                products.extend(page_products)
                
                # 检查是否有下一页
                has_next = await self._has_next_page()
                if not has_next:
                    break
                
                # 翻页
                await self._go_next_page()
                page_num += 1
            
            return products[:max_products]
        
        finally:
            await self.agent.stop()
    
    async def _extract_products(self) -> List[Dict]:
        """提取商品信息"""
        return await self.agent.page.evaluate('''() => {
            const products = [];
            
            // 根据实际页面结构调整选择器
            const productCards = document.querySelectorAll('.product-card, .item, [data-product-id]');
            
            productCards.forEach(card => {
                products.push({
                    name: card.querySelector('.product-name, .title, h3')?.innerText || '',
                    price: card.querySelector('.price, .current-price')?.innerText || '',
                    rating: card.querySelector('.rating, .stars')?.innerText || '',
                    reviews: card.querySelector('.reviews, .count')?.innerText || '',
                    url: card.querySelector('a')?.href || ''
                });
            });
            
            return products;
        }''')
    
    async def _has_next_page(self) -> bool:
        """检查是否有下一页"""
        next_button = await self.agent.page.query_selector('.next, [aria-label="Next"], .pagination .next')
        return next_button is not None
    
    async def _go_next_page(self):
        """翻到下一页"""
        next_button = await self.agent.page.query_selector('.next, [aria-label="Next"], .pagination .next')
        if next_button:
            await next_button.click()
            await self.agent.page.wait_for_load_state('networkidle')
```

### 23.6.2 表单自动填写

```python
class FormAutoFiller:
    """表单自动填写 Agent"""
    
    def __init__(self, llm):
        self.agent = WebAgent(llm)
        self.llm = llm
    
    async def fill_form(self, url: str, form_data: Dict[str, str]) -> bool:
        """自动填写表单"""
        await self.agent.start(headless=False)
        
        try:
            await self.agent.navigate(url)
            
            # 分析表单结构
            form_structure = await self._analyze_form()
            
            # 填写表单
            for field_name, value in form_data.items():
                field = self._find_field(form_structure, field_name)
                if field:
                    await self._fill_field(field, value)
            
            # 提交表单
            success = await self._submit_form()
            
            return success
        
        finally:
            await self.agent.stop()
    
    async def _analyze_form(self) -> Dict:
        """分析表单结构"""
        return await self.agent.page.evaluate('''() => {
            const forms = document.querySelectorAll('form');
            const result = [];
            
            forms.forEach((form, formIndex) => {
                const fields = [];
                
                form.querySelectorAll('input, select, textarea').forEach(field => {
                    fields.push({
                        name: field.name,
                        type: field.type,
                        id: field.id,
                        placeholder: field.placeholder,
                        required: field.required,
                        label: field.labels?.[0]?.innerText || ''
                    });
                });
                
                result.push({
                    index: formIndex,
                    action: form.action,
                    method: form.method,
                    fields: fields
                });
            });
            
            return result;
        }''')
    
    def _find_field(self, form_structure: Dict, field_name: str) -> Optional[Dict]:
        """查找字段"""
        for form in form_structure:
            for field in form['fields']:
                if (field_name.lower() in field['name'].lower() or
                    field_name.lower() in field['label'].lower() or
                    field_name.lower() in field['placeholder'].lower()):
                    return field
        return None
    
    async def _fill_field(self, field: Dict, value: str):
        """填写字段"""
        selector = f'#{field["id"]}' if field['id'] else f'[name="{field["name"]}"]'
        
        if field['type'] == 'select':
            await self.agent.page.select_option(selector, value)
        else:
            await self.agent.page.fill(selector, value)
    
    async def _submit_form(self) -> bool:
        """提交表单"""
        try:
            submit_button = await self.agent.page.query_selector('button[type="submit"], input[type="submit"]')
            if submit_button:
                await submit_button.click()
                await self.agent.page.wait_for_load_state('networkidle')
                return True
            return False
        except:
            return False
```

---

## 23.7 WebArena 评测详解

### 23.7.1 评测环境

| 环境 | 任务类型 | 任务数 | 难度 | 最佳 Agent 准确率 |
|:---|:---|:---|:---|:---|
| **Reddit** | 帖子创建、评论、投票 | 100 | 中 | ~30% |
| **GitLab** | 代码提交、Issue 管理 | 100 | 高 | ~25% |
| **Shopping** | 商品搜索、购买流程 | 100 | 中 | ~35% |
| **Map** | 地点搜索、路线规划 | 100 | 低 | ~40% |
| **CMS** | 内容创建、编辑 | 100 | 中 | ~28% |
| **Terminal** | 命令行操作 | 50 | 高 | ~20% |

### 23.7.2 评测指标

```python
class WebArenaEvaluator:
    """WebArena 评测器"""
    
    def __init__(self):
        self.metrics = {
            'success_rate': 0,
            'avg_steps': 0,
            'avg_time': 0,
            'error_rate': 0
        }
    
    async def evaluate(self, agent, tasks: List[Dict]) -> Dict:
        """评估 Agent 性能"""
        results = []
        
        for task in tasks:
            start_time = time.time()
            
            try:
                result = await agent.run_task(task['query'])
                elapsed_time = time.time() - start_time
                
                results.append({
                    'task_id': task['id'],
                    'success': result['success'],
                    'steps': result['steps'],
                    'time': elapsed_time,
                    'error': result.get('error')
                })
            except Exception as e:
                results.append({
                    'task_id': task['id'],
                    'success': False,
                    'steps': 0,
                    'time': time.time() - start_time,
                    'error': str(e)
                })
        
        return self._calculate_metrics(results)
    
    def _calculate_metrics(self, results: List[Dict]) -> Dict:
        """计算评测指标"""
        total = len(results)
        successes = sum(1 for r in results if r['success'])
        
        return {
            'success_rate': successes / total if total > 0 else 0,
            'avg_steps': sum(r['steps'] for r in results) / total if total > 0 else 0,
            'avg_time': sum(r['time'] for r in results) / total if total > 0 else 0,
            'error_rate': sum(1 for r in results if r['error']) / total if total > 0 else 0,
            'total_tasks': total,
            'successful_tasks': successes
        }
```

---

## 23.8 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| **Web Agent 架构** | 任务解析 → 规划 → 执行 → 状态跟踪 |
| **页面理解** | DOM 分析、语义结构提取、交互元素识别 |
| **视觉理解** | 多模态分析、截图理解、视觉定位 |
| **导航策略** | 任务规划、智能导航、弹窗处理 |
| **错误处理** | 错误检测、恢复策略、操作回滚 |
| **框架选择** | Playwright（推荐）、Selenium、Puppeteer |
| **生产实践** | 电商采集、表单填写、数据提取 |
| **评测基准** | WebArena 提供标准化评测环境 |

---

## 23.9 高级页面理解技术

### 23.9.1 语义理解

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re

@dataclass
class SemanticElement:
    """语义元素"""
    element_type: str  # navigation, content, form, interactive
    purpose: str
    importance: float  # 0-1
    accessibility: Dict[str, Any]
    relationships: List[str]

class SemanticPageAnalyzer:
    """语义页面分析器"""
    
    def __init__(self, page):
        self.page = page
    
    async def analyze_semantics(self) -> Dict[str, Any]:
        """分析页面语义"""
        return await self.page.evaluate('''() => {
            const analysis = {
                structure: {},
                semantics: {},
                accessibility: {},
                interactions: []
            };
            
            // 分析页面结构
            analysis.structure = {
                hasHeader: !!document.querySelector('header, [role="banner"]'),
                hasNav: !!document.querySelector('nav, [role="navigation"]'),
                hasMain: !!document.querySelector('main, [role="main"]'),
                hasAside: !!document.querySelector('aside, [role="complementary"]'),
                hasFooter: !!document.querySelector('footer, [role="contentinfo"]'),
                hasForms: document.querySelectorAll('form').length,
                hasTables: document.querySelectorAll('table').length,
                hasMedia: document.querySelectorAll('img, video, audio').length
            };
            
            // 分析语义
            const semanticElements = document.querySelectorAll('[role], aria-label, aria-describedby]');
            analysis.semantics = {
                roles: {},
                landmarks: [],
                ariaLabels: []
            };
            
            semanticElements.forEach(el => {
                const role = el.getAttribute('role');
                if (role) {
                    analysis.semantics.roles[role] = (analysis.semantics.roles[role] || 0) + 1;
                }
                
                if (['banner', 'navigation', 'main', 'complementary', 'contentinfo'].includes(role)) {
                    analysis.semantics.landmarks.push({
                        role: role,
                        text: el.innerText?.substring(0, 100)
                    });
                }
                
                const ariaLabel = el.getAttribute('aria-label');
                if (ariaLabel) {
                    analysis.semantics.ariaLabels.push(ariaLabel);
                }
            });
            
            // 分析可访问性
            analysis.accessibility = {
                imagesWithoutAlt: document.querySelectorAll('img:not([alt])').length,
                inputsWithoutLabel: document.querySelectorAll('input:not([aria-label]):not([id])').length,
                colorContrastIssues: 0,  // 需要额外计算
                keyboardNavigation: !!document.querySelector('[tabindex]')
            };
            
            // 分析交互元素
            const interactiveElements = document.querySelectorAll('button, a, input, select, textarea, [role="button"]');
            analysis.interactions = Array.from(interactiveElements).slice(0, 20).map(el => ({
                tag: el.tagName.toLowerCase(),
                text: el.innerText?.substring(0, 50) || el.value || '',
                type: el.type || '',
                ariaLabel: el.getAttribute('aria-label') || '',
                tabindex: el.getAttribute('tabindex'),
                visible: el.offsetParent !== null
            }));
            
            return analysis;
        }''')
    
    async def extract_content_hierarchy(self) -> Dict[str, Any]:
        """提取内容层次结构"""
        return await self.page.evaluate('''() => {
            const extractHierarchy = (element, depth = 0) => {
                if (!element || depth > 5) return null;
                
                const info = {
                    tag: element.tagName?.toLowerCase(),
                    text: element.innerText?.substring(0, 200),
                    children: [],
                    metadata: {
                        id: element.id,
                        className: element.className,
                        role: element.getAttribute('role'),
                        ariaLabel: element.getAttribute('aria-label')
                    }
                };
                
                // 提取子元素
                if (element.children) {
                    Array.from(element.children).forEach(child => {
                        const childInfo = extractHierarchy(child, depth + 1);
                        if (childInfo) info.children.push(childInfo);
                    });
                }
                
                return info;
            };
            
            return extractHierarchy(document.body);
        }''')
    
    async def identify_actionable_elements(self) -> List[Dict[str, Any]]:
        """识别可操作元素"""
        return await self.page.evaluate('''() => {
            const actionable = [];
            
            // 按钮
            document.querySelectorAll('button, [role="button"]').forEach(el => {
                if (el.offsetParent !== null) {
                    actionable.push({
                        type: 'button',
                        text: el.innerText?.substring(0, 50),
                        id: el.id,
                        ariaLabel: el.getAttribute('aria-label'),
                        disabled: el.disabled,
                        selector: el.id ? `#${el.id}` : null
                    });
                }
            });
            
            // 链接
            document.querySelectorAll('a[href]').forEach(el => {
                if (el.offsetParent !== null) {
                    actionable.push({
                        type: 'link',
                        text: el.innerText?.substring(0, 50),
                        href: el.href,
                        target: el.target,
                        selector: el.id ? `#${el.id}` : null
                    });
                }
            });
            
            // 输入框
            document.querySelectorAll('input, textarea, select').forEach(el => {
                if (el.offsetParent !== null && el.type !== 'hidden') {
                    actionable.push({
                        type: 'input',
                        inputType: el.type,
                        name: el.name,
                        placeholder: el.placeholder,
                        required: el.required,
                        selector: el.id ? `#${el.id}` : (el.name ? `[name="${el.name}"]` : null)
                    });
                }
            });
            
            return actionable;
        }''')
```

### 23.9.2 智能表单填写

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class FormField:
    """表单字段"""
    name: str
    type: str
    selector: str
    required: bool
    label: str = ""
    placeholder: str = ""
    options: List[str] = None
    validation_rules: Dict[str, Any] = None

class IntelligentFormFiller:
    """智能表单填写器"""
    
    def __init__(self, page, llm):
        self.page = page
        self.llm = llm
    
    async def analyze_form(self, form_selector: str = None) -> List[FormField]:
        """分析表单结构"""
        if form_selector:
            form_html = await self.page.evaluate(f'''() => {{
                const form = document.querySelector('{form_selector}');
                return form ? form.outerHTML : '';
            }}''')
        else:
            form_html = await self.page.evaluate('''() => {
                const forms = document.querySelectorAll('form');
                return Array.from(forms).map(f => f.outerHTML).join('\\n');
            }''')
        
        # 使用 LLM 分析表单结构
        analysis_prompt = f"""
分析以下 HTML 表单，提取所有字段信息：

{form_html[:3000]}

请返回 JSON 格式的字段列表：
[
    {{
        "name": "字段名",
        "type": "text|email|password|select|checkbox|radio|textarea",
        "selector": "CSS 选择器",
        "required": true/false,
        "label": "标签文本",
        "placeholder": "占位符文本",
        "options": ["选项1", "选项2"]  // 仅用于 select
    }}
]
"""
        
        response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
        
        try:
            # 提取 JSON
            json_match = re.search(r'\[[\s\S]*\]', response.content)
            if json_match:
                fields_data = json.loads(json_match.group())
                return [FormField(**field) for field in fields_data]
        except:
            pass
        
        return []
    
    async def fill_form(self, form_data: Dict[str, str], form_selector: str = None) -> Dict[str, Any]:
        """智能填写表单"""
        # 分析表单结构
        fields = await self.analyze_form(form_selector)
        
        results = {
            "total_fields": len(fields),
            "filled": 0,
            "skipped": 0,
            "failed": 0,
            "details": []
        }
        
        for field in fields:
            # 查找对应的输入值
            value = self._find_matching_value(field, form_data)
            
            if value is None and field.required:
                # 使用 LLM 生成合理的值
                value = await self._generate_value_for_field(field, form_data)
            
            if value is None:
                results["skipped"] += 1
                results["details"].append({
                    "field": field.name,
                    "status": "skipped",
                    "reason": "no matching value"
                })
                continue
            
            # 填写字段
            success = await self._fill_field(field, value)
            
            if success:
                results["filled"] += 1
                results["details"].append({
                    "field": field.name,
                    "status": "filled",
                    "value": value[:50] + "..." if len(str(value)) > 50 else value
                })
            else:
                results["failed"] += 1
                results["details"].append({
                    "field": field.name,
                    "status": "failed",
                    "error": "fill operation failed"
                })
        
        return results
    
    def _find_matching_value(self, field: FormField, form_data: Dict[str, str]) -> Optional[str]:
        """查找匹配的值"""
        # 直接匹配字段名
        if field.name in form_data:
            return form_data[field.name]
        
        # 匹配标签
        if field.label:
            for key, value in form_data.items():
                if field.label.lower() in key.lower() or key.lower() in field.label.lower():
                    return value
        
        # 匹配占位符
        if field.placeholder:
            for key, value in form_data.items():
                if field.placeholder.lower() in value.lower():
                    return value
        
        return None
    
    async def _generate_value_for_field(self, field: FormField, context: Dict[str, str]) -> Optional[str]:
        """使用 LLM 生成字段值"""
        prompt = f"""
根据以下上下文，为表单字段生成合适的值：

字段信息：
- 名称: {field.name}
- 类型: {field.type}
- 标签: {field.label}
- 占位符: {field.placeholder}
- 必填: {field.required}
- 选项: {field.options}

上下文数据：
{json.dumps(context, ensure_ascii=False, indent=2)[:1000]}

请生成一个合适的值（只返回值本身，不要包含其他内容）：
"""
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
    
    async def _fill_field(self, field: FormField, value: str) -> bool:
        """填写字段"""
        try:
            if field.type in ['text', 'email', 'password', 'number', 'tel', 'url']:
                await self.page.fill(field.selector, value)
            
            elif field.type == 'textarea':
                await self.page.fill(field.selector, value)
            
            elif field.type == 'select':
                await self.page.select_option(field.selector, value)
            
            elif field.type == 'checkbox':
                if value.lower() in ['true', '1', 'yes', 'on']:
                    await self.page.check(field.selector)
                else:
                    await self.page.uncheck(field.selector)
            
            elif field.type == 'radio':
                await self.page.check(field.selector)
            
            # 等待输入生效
            await self.page.wait_for_timeout(300)
            
            return True
        
        except Exception as e:
            print(f"Failed to fill field {field.name}: {e}")
            return False
    
    async def validate_form(self, form_selector: str = None) -> Dict[str, Any]:
        """验证表单填写"""
        validation_result = await self.page.evaluate('''(formSelector) => {
            const form = formSelector ? document.querySelector(formSelector) : document.querySelector('form');
            if (!form) return { valid: false, error: 'Form not found' };
            
            const errors = [];
            
            // 检查必填字段
            form.querySelectorAll('[required]').forEach(field => {
                if (!field.value || field.value.trim() === '') {
                    errors.push({
                        field: field.name || field.id,
                        error: 'Required field is empty'
                    });
                }
            });
            
            // 检查邮箱格式
            form.querySelectorAll('[type="email"]').forEach(field => {
                if (field.value && !field.value.match(/^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/)) {
                    errors.push({
                        field: field.name,
                        error: 'Invalid email format'
                    });
                }
            });
            
            // 检查 URL 格式
            form.querySelectorAll('[type="url"]').forEach(field => {
                if (field.value && !field.value.match(/^https?:\\/\\/.+/)) {
                    errors.push({
                        field: field.name,
                        error: 'Invalid URL format'
                    });
                }
            });
            
            return {
                valid: errors.length === 0,
                errors: errors
            };
        }''', form_selector)
        
        return validation_result
```

---

## 23.10 验证码处理

### 23.10.1 验证码识别和解决

```python
from typing import Dict, Any, Optional
from enum import Enum

class CaptchaType(Enum):
    """验证码类型"""
    TEXT = "text"  # 文本验证码
    IMAGE = "image"  # 图片验证码
    RECAPTCHA = "recaptcha"  # Google reCAPTCHA
    HCAPTCHA = "hcaptcha"  # hCaptcha
    FUNCAPTCHA = "funcaptcha"  # FunCaptcha
    SLIDER = "slider"  # 滑块验证码
    PUZZLE = "puzzle"  # 拼图验证码

class CaptchaHandler:
    """验证码处理器"""
    
    def __init__(self, llm, ocr_engine=None):
        self.llm = llm
        self.ocr = ocr_engine
        self.solvers = {
            CaptchaType.TEXT: self._solve_text_captcha,
            CaptchaType.IMAGE: self._solve_image_captcha,
            CaptchaType.SLIDER: self._solve_slider_captcha,
        }
    
    async def detect_captcha(self, page) -> Optional[CaptchaType]:
        """检测验证码类型"""
        captcha_info = await page.evaluate('''() => {
            const info = { detected: false, type: null, selectors: [] };
            
            // 检测 reCAPTCHA
            if (document.querySelector('.g-recaptcha, [data-sitekey]')) {
                info.detected = true;
                info.type = 'recaptcha';
                info.selectors.push('.g-recaptcha');
            }
            
            // 检测 hCaptcha
            if (document.querySelector('.h-captcha, [data-hcaptcha-widget]')) {
                info.detected = true;
                info.type = 'hcaptcha';
                info.selectors.push('.h-captcha');
            }
            
            // 检测图片验证码
            const captchaImages = document.querySelectorAll('img[src*="captcha"], img[alt*="captcha"], .captcha-image');
            if (captchaImages.length > 0) {
                info.detected = true;
                info.type = 'image';
                info.selectors.push(...Array.from(captchaImages).map(img => img.className || 'img[src*="captcha"]'));
            }
            
            // 检测文本验证码
            const captchaInputs = document.querySelectorAll('input[name*="captcha"], input[placeholder*="captcha"], input[placeholder*="验证码"]');
            if (captchaInputs.length > 0) {
                info.detected = true;
                info.type = 'text';
                info.selectors.push(...Array.from(captchaInputs).map(input => input.name ? `input[name="${input.name}"]` : 'input[placeholder*="验证码"]'));
            }
            
            // 检测滑块验证码
            if (document.querySelector('.slider-captcha, .slide-verify, [class*="slider"]')) {
                info.detected = true;
                info.type = 'slider';
                info.selectors.push('.slider-captcha');
            }
            
            return info;
        }''')
        
        if captcha_info.get('detected'):
            return CaptchaType(captcha_info['type'])
        
        return None
    
    async def solve_captcha(self, page, captcha_type: CaptchaType) -> Dict[str, Any]:
        """解决验证码"""
        solver = self.solvers.get(captcha_type)
        if not solver:
            return {"success": False, "error": f"No solver for {captcha_type}"}
        
        return await solver(page)
    
    async def _solve_text_captcha(self, page) -> Dict[str, Any]:
        """解决文本验证码"""
        try:
            # 获取验证码图片
            captcha_img = await page.query_selector('img[src*="captcha"], img[alt*="captcha"]')
            if not captcha_img:
                return {"success": False, "error": "Captcha image not found"}
            
            # 截图
            screenshot = await captcha_img.screenshot()
            
            # OCR 识别
            if self.ocr:
                text = self.ocr.recognize(screenshot)
            else:
                # 使用 LLM 识别
                import base64
                img_b64 = base64.b64encode(screenshot).decode()
                text = await self._recognize_with_llm(img_b64)
            
            if text:
                # 填入验证码
                input_field = await page.query_selector('input[name*="captcha"], input[placeholder*="验证码"]')
                if input_field:
                    await input_field.fill(text)
                    return {"success": True, "text": text}
            
            return {"success": False, "error": "Failed to recognize captcha"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _solve_image_captcha(self, page) -> Dict[str, Any]:
        """解决图片验证码"""
        return await self._solve_text_captcha(page)
    
    async def _solve_slider_captcha(self, page) -> Dict[str, Any]:
        """解决滑块验证码"""
        try:
            # 获取滑块元素
            slider = await page.query_selector('.slider-button, .slide-verify-slider, [class*="slider"] button')
            if not slider:
                return {"success": False, "error": "Slider not found"}
            
            # 获取滑块轨道
            track = await page.query_selector('.slider-track, .slide-verify-track, [class*="slider"] .track')
            if not track:
                return {"success": False, "error": "Slider track not found"}
            
            # 获取位置信息
            slider_box = await slider.bounding_box()
            track_box = await track.bounding_box()
            
            if not slider_box or not track_box:
                return {"success": False, "error": "Failed to get element positions"}
            
            # 模拟滑动
            start_x = slider_box['x'] + slider_box['width'] / 2
            start_y = slider_box['y'] + slider_box['height'] / 2
            end_x = track_box['x'] + track_box['width'] * 0.8  # 滑动到 80% 位置
            
            await page.mouse.move(start_x, start_y)
            await page.mouse.down()
            
            # 模拟人类滑动
            steps = 20
            for i in range(steps):
                progress = (i + 1) / steps
                current_x = start_x + (end_x - start_x) * progress
                await page.mouse.move(current_x, start_y + random.uniform(-2, 2))
                await page.wait_for_timeout(random.randint(10, 30))
            
            await page.mouse.up()
            
            return {"success": True, "distance": end_x - start_x}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _recognize_with_llm(self, image_b64: str) -> Optional[str]:
        """使用 LLM 识别验证码"""
        prompt = f"""
请识别这张验证码图片中的文字/数字。

这是一个验证码图片，请只返回识别出的文字，不要包含任何其他内容。

验证码图片（base64）：
{image_b64[:500]}...
"""
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        # 提取识别结果
        text = response.content.strip()
        
        # 清理结果（只保留字母和数字）
        text = re.sub(r'[^a-zA-Z0-9]', '', text)
        
        return text if text else None
```

---

## 23.11 反爬虫策略

### 23.11.1 反检测技术

```python
from typing import Dict, List, Any
import random
import asyncio

class AntiDetection:
    """反检测技术"""
    
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        ]
    
    async def apply_stealth(self, page):
        """应用隐身模式"""
        # 设置随机 User-Agent
        user_agent = random.choice(self.user_agents)
        await page.set_extra_http_headers({"User-Agent": user_agent})
        
        # 修改 navigator 属性
        await page.evaluate('''() => {
            // 修改 webdriver 属性
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false
            });
            
            // 修改 plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            
            // 修改 languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
            
            // 修改 platform
            Object.defineProperty(navigator, 'platform', {
                get: () => 'Win32'
            });
            
            // 修改 chrome
            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {}
            };
            
            // 修改权限
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        }''')
        
        # 添加 stealth 插件
        await page.add_init_script("""
            // 隐藏 webdriver 标志
            delete navigator.__proto__.webdriver;
            
            // 修改 chrome runtime
            window.chrome = {
                runtime: {}
            };
            
            // 修改 permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        """)
    
    async def random_delay(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """随机延迟"""
        delay = random.uniform(min_seconds, max_seconds)
        await asyncio.sleep(delay)
    
    async def simulate_human_behavior(self, page):
        """模拟人类行为"""
        # 随机鼠标移动
        await page.mouse.move(
            random.randint(100, 500),
            random.randint(100, 300)
        )
        
        # 随机滚动
        await page.mouse.wheel(0, random.randint(-100, 100))
        
        # 随机延迟
        await self.random_delay(0.5, 1.5)

class ProxyManager:
    """代理管理器"""
    
    def __init__(self):
        self.proxies: List[Dict[str, str]] = []
        self.current_proxy_index = 0
        self.failed_proxies: List[str] = []
    
    def add_proxy(self, proxy: Dict[str, str]):
        """添加代理"""
        self.proxies.append(proxy)
    
    def load_proxies_from_file(self, file_path: str):
        """从文件加载代理"""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        parts = line.split(':')
                        if len(parts) == 2:
                            self.proxies.append({
                                "server": f"http://{parts[0]}:{parts[1]}",
                                "username": "",
                                "password": ""
                            })
                        elif len(parts) == 4:
                            self.proxies.append({
                                "server": f"http://{parts[0]}:{parts[1]}",
                                "username": parts[2],
                                "password": parts[3]
                            })
        except Exception as e:
            print(f"Failed to load proxies: {e}")
    
    def get_next_proxy(self) -> Optional[Dict[str, str]]:
        """获取下一个代理"""
        available_proxies = [
            p for p in self.proxies
            if p["server"] not in self.failed_proxies
        ]
        
        if not available_proxies:
            return None
        
        proxy = available_proxies[self.current_proxy_index % len(available_proxies)]
        self.current_proxy_index += 1
        
        return proxy
    
    def mark_proxy_failed(self, proxy: Dict[str, str]):
        """标记代理失败"""
        self.failed_proxies.append(proxy["server"])
        
        # 限制失败列表大小
        if len(self.failed_proxies) > 100:
            self.failed_proxies = self.failed_proxies[-50:]
    
    async def create_browser_with_proxy(self, playwright, proxy: Dict[str, str]):
        """使用代理创建浏览器"""
        browser = await playwright.chromium.launch(
            headless=True,
            proxy=proxy
        )
        
        return browser

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.request_times: List[float] = []
    
    async def wait_if_needed(self):
        """如果需要则等待"""
        now = time.time()
        
        # 清理旧请求记录
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # 检查是否超过限制
        if len(self.request_times) >= self.requests_per_minute:
            # 计算需要等待的时间
            oldest_request = self.request_times[0]
            wait_time = 60 - (now - oldest_request)
            
            if wait_time > 0:
                print(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
        
        # 记录请求时间
        self.request_times.append(time.time())
```

---

## 23.12 并发控制

### 23.12.1 并发任务管理

```python
import asyncio
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ConcurrentTask:
    """并发任务"""
    task_id: str
    url: str
    function: Callable
    status: TaskStatus
    result: Any = None
    error: str = None
    start_time: float = None
    end_time: float = None

class ConcurrentTaskManager:
    """并发任务管理器"""
    
    def __init__(self, max_concurrent: int = 5, max_tasks_per_minute: int = 30):
        self.max_concurrent = max_concurrent
        self.max_tasks_per_minute = max_tasks_per_minute
        self.tasks: List[ConcurrentTask] = []
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = RateLimiter(max_tasks_per_minute)
        self.running_tasks = 0
    
    async def add_task(self, url: str, function: Callable, task_id: str = None) -> str:
        """添加任务"""
        if not task_id:
            task_id = f"task_{len(self.tasks) + 1}_{int(time.time())}"
        
        task = ConcurrentTask(
            task_id=task_id,
            url=url,
            function=function,
            status=TaskStatus.PENDING
        )
        
        self.tasks.append(task)
        return task_id
    
    async def run_all_tasks(self) -> Dict[str, Any]:
        """运行所有任务"""
        start_time = time.time()
        
        # 创建任务协程
        coroutines = [self._run_task(task) for task in self.tasks if task.status == TaskStatus.PENDING]
        
        # 并发执行
        await asyncio.gather(*coroutines, return_exceptions=True)
        
        # 统计结果
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        
        return {
            "total_tasks": len(self.tasks),
            "completed": completed,
            "failed": failed,
            "duration": time.time() - start_time,
            "results": [t.result for t in self.tasks if t.status == TaskStatus.COMPLETED]
        }
    
    async def _run_task(self, task: ConcurrentTask):
        """运行单个任务"""
        async with self.semaphore:
            # 等待速率限制
            await self.rate_limiter.wait_if_needed()
            
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()
            self.running_tasks += 1
            
            try:
                result = await task.function(task.url)
                task.result = result
                task.status = TaskStatus.COMPLETED
            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED
            finally:
                task.end_time = time.time()
                self.running_tasks -= 1
    
    def get_task_status(self, task_id: str) -> Optional[ConcurrentTask]:
        """获取任务状态"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_tasks": len(self.tasks),
            "pending": sum(1 for t in self.tasks if t.status == TaskStatus.PENDING),
            "running": sum(1 for t in self.tasks if t.status == TaskStatus.RUNNING),
            "completed": sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self.tasks if t.status == TaskStatus.FAILED),
            "currently_running": self.running_tasks
        }
    
    def cancel_all_tasks(self):
        """取消所有任务"""
        for task in self.tasks:
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                task.status = TaskStatus.CANCELLED
```

---

## 23.13 数据提取和解析

### 23.13.1 智能数据提取

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re
import json

@dataclass
class ExtractedData:
    """提取的数据"""
    data_type: str
    content: Any
    confidence: float
    metadata: Dict[str, Any]

class IntelligentDataExtractor:
    """智能数据提取器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def extract_structured_data(self, page, data_type: str) -> List[ExtractedData]:
        """提取结构化数据"""
        # 获取页面内容
        page_content = await page.evaluate('''() => {
            return {
                html: document.body.innerHTML.substring(0, 10000),
                text: document.body.innerText.substring(0, 5000),
                url: window.location.href,
                title: document.title
            };
        }''')
        
        # 使用 LLM 提取数据
        prompt = f"""
从以下网页内容中提取 {data_type} 类型的结构化数据。

网页 URL: {page_content['url']}
网页标题: {page_content['title']}

网页文本内容:
{page_content['text'][:3000]}

请提取所有 {data_type} 相关的数据，并以 JSON 格式返回：
[
    {{
        "data": {{...}},
        "confidence": 0.95,
        "metadata": {{...}}
    }}
]
"""
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        try:
            # 提取 JSON
            json_match = re.search(r'\[[\s\S]*\]', response.content)
            if json_match:
                data_list = json.loads(json_match.group())
                return [
                    ExtractedData(
                        data_type=data_type,
                        content=item.get("data"),
                        confidence=item.get("confidence", 0.8),
                        metadata=item.get("metadata", {})
                    )
                    for item in data_list
                ]
        except:
            pass
        
        return []
    
    async def extract_table_data(self, page, table_selector: str = None) -> List[Dict[str, Any]]:
        """提取表格数据"""
        if table_selector:
            table_html = await page.evaluate(f'''() => {{
                const table = document.querySelector('{table_selector}');
                return table ? table.outerHTML : '';
            }}''')
        else:
            table_html = await page.evaluate('''() => {
                const tables = document.querySelectorAll('table');
                return Array.from(tables).map(t => t.outerHTML).join('\\n');
            }''')
        
        # 解析表格
        return self._parse_html_tables(table_html)
    
    def _parse_html_tables(self, html: str) -> List[Dict[str, Any]]:
        """解析 HTML 表格"""
        tables = []
        
        # 使用正则表达式提取表格
        table_pattern = r'<table[^>]*>(.*?)</table>'
        tables_html = re.findall(table_pattern, html, re.DOTALL)
        
        for table_html in tables_html:
            # 提取表头
            header_pattern = r'<th[^>]*>(.*?)</th>'
            headers = re.findall(header_pattern, table_html, re.DOTALL)
            headers = [re.sub(r'<[^>]+>', '', h).strip() for h in headers]
            
            # 提取行
            row_pattern = r'<tr[^>]*>(.*?)</tr>'
            rows_html = re.findall(row_pattern, table_html, re.DOTALL)
            
            rows = []
            for row_html in rows_html:
                if '<th>' in row_html:  # 跳过表头行
                    continue
                
                cell_pattern = r'<td[^>]*>(.*?)</td>'
                cells = re.findall(cell_pattern, row_html, re.DOTALL)
                cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
                
                if cells and len(cells) == len(headers):
                    row_dict = {headers[i]: cells[i] for i in range(len(headers))}
                    rows.append(row_dict)
            
            if rows:
                tables.append({
                    "headers": headers,
                    "rows": rows,
                    "row_count": len(rows)
                })
        
        return tables
    
    async def extract_links(self, page, filter_pattern: str = None) -> List[Dict[str, str]]:
        """提取链接"""
        links = await page.evaluate('''() => {
            const links = [];
            document.querySelectorAll('a[href]').forEach(a => {
                if (a.href && !a.href.startsWith('javascript:')) {
                    links.push({
                        url: a.href,
                        text: a.innerText?.trim() || '',
                        title: a.title || '',
                        target: a.target || '_self'
                    });
                }
            });
            return links;
        }''')
        
        if filter_pattern:
            links = [l for l in links if re.search(filter_pattern, l['url'])]
        
        return links
    
    async def extract_images(self, page) -> List[Dict[str, str]]:
        """提取图片"""
        return await page.evaluate('''() => {
            const images = [];
            document.querySelectorAll('img').forEach(img => {
                if (img.src) {
                    images.push({
                        url: img.src,
                        alt: img.alt || '',
                        title: img.title || '',
                        width: img.naturalWidth,
                        height: img.naturalHeight
                    });
                }
            });
            return images;
        }''')
```

---

## 23.14 Web Agent 框架对比

### 23.14.1 主流框架比较

| 框架 | 语言 | 主要特点 | 适用场景 | 学习曲线 |
|:---|:---|:---|:---|:---|
| **Playwright** | Python/JS | 全功能、多浏览器、自动等待 | 通用 Web 自动化 | 中 |
| **Selenium** | 多语言 | 经典、社区大、兼容性好 | 传统 Web 测试 | 中 |
| **Puppeteer** | JS | Chrome 专用、性能好 | Chrome 自动化 | 低 |
| **Browser Use** | Python | LLM 驱动、简单易用 | AI Agent 集成 | 低 |
| **Skyvern** | Python | 视觉驱动、OCR | 复杂页面操作 | 中 |
| **Scrapy** | Python | 爬虫框架、异步 | 大规模数据采集 | 高 |

### 23.14.2 框架选择建议

```python
class FrameworkSelector:
    """框架选择器"""
    
    @staticmethod
    def recommend(use_case: str, requirements: Dict[str, Any] = None) -> str:
        """推荐框架"""
        requirements = requirements or {}
        
        recommendations = {
            "web_scraping": {
                "simple": "Browser Use",
                "complex": "Playwright",
                "large_scale": "Scrapy + Playwright"
            },
            "form_automation": {
                "simple": "Browser Use",
                "complex": "Playwright",
                "legacy": "Selenium"
            },
            "testing": {
                "unit": "Playwright",
                "integration": "Playwright",
                "e2e": "Playwright"
            },
            "ai_agent": {
                "simple": "Browser Use",
                "advanced": "Playwright + LLM",
                "visual": "Skyvern"
            }
        }
        
        if use_case in recommendations:
            complexity = requirements.get("complexity", "simple")
            return recommendations[use_case].get(complexity, "Playwright")
        
        return "Playwright"  # 默认推荐
```

---

## 23.15 生产环境案例

### 23.15.1 大规模电商数据采集

```python
class EcommerceDataCollectionCase:
    """电商数据采集案例"""
    
    def __init__(self):
        self.case_study = {
            "company": "某数据分析公司",
            "scale": {
                "daily_products": 1_000_000,
                "websites": 500,
                "data_points_per_product": 50
            },
            "challenges": [
                "大规模并发采集",
                "反爬虫策略",
                "数据质量保证",
                "实时数据处理"
            ]
        }
    
    async def implement_solution(self) -> Dict[str, Any]:
        """实施解决方案"""
        solution = {
            "architecture": {
                "task_manager": "Celery + Redis",
                "browser_pool": "Playwright Browser Pool",
                "proxy_pool": "自建代理池 + 第三方代理",
                "data_storage": "MongoDB + Elasticsearch",
                "monitoring": "Prometheus + Grafana"
            },
            "strategies": [
                {
                    "name": "智能任务调度",
                    "description": "根据网站优先级和反爬虫强度动态调度",
                    "implementation": "使用 Celery Beat 定时任务，根据历史成功率调整频率"
                },
                {
                    "name": "代理轮换",
                    "description": "自动轮换代理避免封禁",
                    "implementation": "维护代理池，自动检测代理可用性，失败自动切换"
                },
                {
                    "name": "浏览器指纹伪装",
                    "description": "模拟真实用户浏览器环境",
                    "implementation": "使用 Playwright stealth 插件，随机化浏览器指纹"
                },
                {
                    "name": "数据质量验证",
                    "description": "自动验证提取数据的完整性",
                    "implementation": "使用 LLM 验证数据合理性，异常数据自动标记"
                }
            ],
            "results": {
                "daily_products": "1,000,000+",
                "success_rate": "95%",
                "data_quality": "98%",
                "cost_per_product": "$0.001"
            }
        }
        
        return solution
```

---

## 23.16 常见问题和解决方案

### 23.16.1 Web Agent 问题排查

```python
class WebAgentTroubleshooting:
    """Web Agent 问题排查指南"""
    
    @staticmethod
    def get_common_issues() -> List[Dict[str, Any]]:
        """获取常见问题"""
        return [
            {
                "issue": "元素无法点击",
                "symptoms": ["Timeout waiting for element", "Element is not visible"],
                "causes": [
                    "元素未加载完成",
                    "元素被遮挡",
                    "元素不在视口中",
                    "动态加载的元素"
                ],
                "solutions": [
                    "增加等待时间",
                    "使用 page.wait_for_selector()",
                    "滚动到元素位置",
                    "使用 page.wait_for_load_state()"
                ]
            },
            {
                "issue": "页面加载超时",
                "symptoms": ["Timeout exceeded", "Navigation timeout"],
                "causes": [
                    "网络延迟",
                    "服务器响应慢",
                    "页面资源过多",
                    "反爬虫检测"
                ],
                "solutions": [
                    "增加超时时间",
                    "使用 headless 模式",
                    "拦截不必要的资源",
                    "使用代理或 VPN"
                ]
            },
            {
                "issue": "验证码频繁出现",
                "symptoms": ["CAPTCHA required", "Access denied"],
                "causes": [
                    "请求频率过高",
                    "浏览器指纹异常",
                    "IP 被标记"
                ],
                "solutions": [
                    "降低请求频率",
                    "使用代理轮换",
                    "应用隐身模式",
                    "使用验证码解决服务"
                ]
            },
            {
                "issue": "数据提取不完整",
                "symptoms": ["Missing fields", "Empty results"],
                "causes": [
                    "选择器错误",
                    "页面结构变化",
                    "动态内容未加载"
                ],
                "solutions": [
                    "更新选择器",
                    "使用更稳定的选择器",
                    "等待动态内容加载",
                    "使用多种选择器策略"
                ]
            }
        ]
    
    @staticmethod
    def get_debugging_tips() -> List[Dict[str, str]]:
        """获取调试技巧"""
        return [
            {"tip": "使用 headed 模式调试", "command": "headless=False"},
            {"tip": "截图保存页面状态", "command": "await page.screenshot()"},
            {"tip": "记录网络请求", "command": "page.on('response', handler)"},
            {"tip": "调试选择器", "command": "await page.query_selector('selector')"},
            {"tip": "检查元素状态", "command": "element.is_visible(), element.is_enabled()"},
            {"tip": "使用慢动作模式", "command": "playwright.slow_mo=1000"},
            {"tip": "保存页面 HTML", "command": "await page.content()"},
            {"tip": "录制操作脚本", "command": "playwright codegen"}
        ]
```

---

## 23.17 本章小结（更新版）

| 知识点 | 核心要点 |
|:---|:---|
| **Web Agent 架构** | 任务解析 → 规划 → 执行 → 状态跟踪 |
| **页面理解** | DOM 分析、语义结构提取、交互元素识别 |
| **视觉理解** | 多模态分析、截图理解、视觉定位 |
| **导航策略** | 任务规划、智能导航、弹窗处理 |
| **错误处理** | 错误检测、恢复策略、操作回滚 |
| **框架选择** | Playwright（推荐）、Selenium、Puppeteer |
| **生产实践** | 电商采集、表单填写、数据提取 |
| **评测基准** | WebArena 提供标准化评测环境 |
| **语义理解** | 页面结构分析、可访问性检查 |
| **智能表单** | 自动识别字段、智能填写、验证 |
| **验证码处理** | 识别类型、自动解决、滑块验证 |
| **反爬虫** | 随机延迟、指纹伪装、代理轮换 |
| **并发控制** | 任务队列、速率限制、资源管理 |
| **数据提取** | 结构化数据、表格、链接、图片 |

---

## 23.18 Web Agent 最佳实践

### 23.18.1 设计原则

```python
class WebAgentBestPractices:
    """Web Agent 最佳实践"""
    
    @staticmethod
    def get_design_principles() -> List[Dict[str, str]]:
        """获取设计原则"""
        return [
            {
                "principle": "幂等性",
                "description": "确保相同的输入产生相同的结果，避免副作用",
                "implementation": "使用唯一标识符，检查状态再操作"
            },
            {
                "principle": "容错性",
                "description": "优雅处理各种异常情况",
                "implementation": "实现重试机制、超时处理、错误恢复"
            },
            {
                "principle": "可观测性",
                "description": "记录详细的执行日志和指标",
                "implementation": "使用结构化日志、分布式追踪"
            },
            {
                "principle": "资源管理",
                "description": "合理使用系统资源，避免内存泄漏",
                "implementation": "及时关闭浏览器、限制并发数"
            },
            {
                "principle": "安全性",
                "description": "保护敏感数据，防止恶意利用",
                "implementation": "输入验证、输出过滤、访问控制"
            }
        ]
    
    @staticmethod
    def get_coding_standards() -> List[Dict[str, str]]:
        """获取编码规范"""
        return [
            {
                "category": "选择器",
                "standards": [
                    "优先使用 data-testid 属性",
                    "避免使用脆弱的选择器（如 nth-child）",
                    "使用 role 属性定位语义元素",
                    "为选择器添加注释说明用途"
                ]
            },
            {
                "category": "等待策略",
                "standards": [
                    "使用显式等待而非隐式等待",
                    "优先使用 page.wait_for_selector()",
                    "避免使用固定延时（sleep）",
                    "设置合理的超时时间"
                ]
            },
            {
                "category": "错误处理",
                "standards": [
                    "捕获具体的异常类型",
                    "记录详细的错误信息",
                    "实现优雅的降级策略",
                    "避免捕获所有异常"
                ]
            },
            {
                "category": "代码组织",
                "standards": [
                    "将页面操作封装为 Page Object",
                    "分离测试数据和测试逻辑",
                    "使用配置文件管理环境变量",
                    "编写清晰的文档字符串"
                ]
            }
        ]
    
    @staticmethod
    def get_performance_tips() -> List[Dict[str, str]]:
        """获取性能优化建议"""
        return [
            {
                "tip": "使用无头模式",
                "description": "在不需要可视化时使用 headless=True",
                "benefit": "减少资源消耗，提高执行速度"
            },
            {
                "tip": "拦截不必要的资源",
                "description": "阻止加载图片、字体等不需要的资源",
                "benefit": "减少网络请求，加快页面加载"
            },
            {
                "tip": "复用浏览器实例",
                "description": "在多个任务间复用同一个浏览器实例",
                "benefit": "减少启动开销，提高效率"
            },
            {
                "tip": "并行执行任务",
                "description": "使用异步和并发同时处理多个页面",
                "benefit": "提高吞吐量，减少总时间"
            },
            {
                "tip": "缓存页面状态",
                "description": "缓存页面内容避免重复加载",
                "benefit": "减少网络请求，加快响应速度"
            }
        ]
    
    @staticmethod
    def get_security_practices() -> List[Dict[str, str]]:
        """获取安全实践"""
        return [
            {
                "practice": "输入验证",
                "description": "验证所有用户输入",
                "implementation": "使用正则表达式、类型检查"
            },
            {
                "practice": "敏感数据保护",
                "description": "不记录敏感信息（密码、token）",
                "implementation": "日志脱敏、内存清理"
            },
            {
                "practice": "访问控制",
                "description": "限制 Agent 的访问范围",
                "implementation": "使用沙箱、网络隔离"
            },
            {
                "practice": "速率限制",
                "description": "防止过度请求导致封禁",
                "implementation": "实现请求队列、延时控制"
            },
            {
                "practice": "日志审计",
                "description": "记录所有操作用于审计",
                "implementation": "结构化日志、操作追踪"
            }
        ]
```

### 23.18.2 常见模式和模板

```python
class WebAgentPatterns:
    """Web Agent 常见模式"""
    
    @staticmethod
    def get_scraping_pattern() -> Dict[str, Any]:
        """获取数据采集模式"""
        return {
            "name": "数据采集模式",
            "description": "从网站提取结构化数据",
            "steps": [
                "1. 导航到目标页面",
                "2. 等待页面加载完成",
                "3. 提取页面内容",
                "4. 解析数据结构",
                "5. 验证数据完整性",
                "6. 保存到存储"
            ],
            "error_handling": [
                "页面加载超时：重试或跳过",
                "选择器未找到：更新选择器或报告",
                "数据格式异常：记录并跳过",
                "反爬虫检测：切换代理或等待"
            ]
        }
    
    @staticmethod
    def get_form_filling_pattern() -> Dict[str, Any]:
        """获取表单填写模式"""
        return {
            "name": "表单填写模式",
            "description": "自动填写网页表单",
            "steps": [
                "1. 分析表单结构",
                "2. 识别字段类型",
                "3. 准备填写数据",
                "4. 逐个填写字段",
                "5. 验证填写结果",
                "6. 提交表单"
            ],
            "validation": [
                "必填字段检查",
                "格式验证（邮箱、电话）",
                "业务规则验证",
                "重复提交检查"
            ]
        }
    
    @staticmethod
    def get_navigation_pattern() -> Dict[str, Any]:
        """获取导航模式"""
        return {
            "name": "智能导航模式",
            "description": "在网站中导航到目标页面",
            "strategies": [
                "直接 URL 导航",
                "搜索框搜索",
                "菜单点击导航",
                "链接点击导航",
                "面包屑导航"
            ],
            "fallback": [
                "页面不存在：尝试搜索",
                "链接失效：检查替代链接",
                "需要登录：使用预设账号",
                "访问被拒：切换代理"
            ]
        }
    
    @staticmethod
    def get_monitoring_pattern() -> Dict[str, Any]:
        """获取监控模式"""
        return {
            "name": "网站监控模式",
            "description": "监控网站变化",
            "features": [
                "内容变化检测",
                "价格变动监控",
                "新页面发现",
                "异常告警"
            ],
            "implementation": [
                "定期访问目标页面",
                "对比历史快照",
                "计算变化差异",
                "发送告警通知"
            ]
        }
```

---

## 23.19 案例研究：大规模 Web 数据采集平台

### 23.19.1 平台架构

```python
class LargeScaleWebScrapingPlatform:
    """大规模 Web 数据采集平台"""
    
    def __init__(self):
        self.architecture = {
            "components": {
                "task_scheduler": {
                    "technology": "Celery + Redis",
                    "responsibility": "任务调度和队列管理",
                    "scaling": "水平扩展 Worker"
                },
                "browser_pool": {
                    "technology": "Playwright + Browser Pool",
                    "responsibility": "浏览器实例管理",
                    "scaling": "动态调整池大小"
                },
                "proxy_manager": {
                    "technology": "自建代理池 + 第三方代理",
                    "responsibility": "代理轮换和健康检查",
                    "scaling": "多区域代理部署"
                },
                "data_pipeline": {
                    "technology": "Apache Kafka + Spark",
                    "responsibility": "数据流处理和转换",
                    "scaling": "分区并行处理"
                },
                "storage": {
                    "technology": "MongoDB + Elasticsearch + S3",
                    "responsibility": "数据存储和检索",
                    "scaling": "分片和副本集"
                },
                "monitoring": {
                    "technology": "Prometheus + Grafana",
                    "responsibility": "系统监控和告警",
                    "scaling": "联邦监控"
                }
            },
            "workflows": {
                "scraping_workflow": [
                    "1. 任务入队",
                    "2. 调度器分配任务",
                    "3. Worker 获取代理",
                    "4. 启动浏览器实例",
                    "5. 执行采集任务",
                    "6. 提取数据",
                    "7. 数据验证",
                    "8. 数据存储",
                    "9. 任务完成通知"
                ],
                "error_recovery": [
                    "1. 检测错误类型",
                    "2. 记录错误详情",
                    "3. 重试策略判断",
                    "4. 代理切换",
                    "5. 任务重新入队",
                    "6. 失败任务报告"
                ]
            }
        }
    
    async def deploy(self) -> Dict[str, Any]:
        """部署平台"""
        deployment_config = {
            "kubernetes": {
                "namespaces": ["scraping", "monitoring"],
                "deployments": [
                    {"name": "task-scheduler", "replicas": 3},
                    {"name": "browser-worker", "replicas": 10},
                    {"name": "data-processor", "replicas": 5},
                    {"name": "api-server", "replicas": 3}
                ],
                "services": [
                    {"name": "redis", "type": "ClusterIP"},
                    {"name": "mongodb", "type": "ClusterIP"},
                    {"name": "kafka", "type": "ClusterIP"}
                ]
            },
            "monitoring": {
                "metrics": [
                    "scraping_tasks_total",
                    "scraping_success_rate",
                    "scraping_duration_seconds",
                    "proxy_usage_ratio",
                    "browser_pool_size"
                ],
                "alerts": [
                    {"metric": "scraping_success_rate < 0.9", "severity": "warning"},
                    {"metric": "scraping_duration_seconds > 300", "severity": "warning"},
                    {"metric": "proxy_usage_ratio > 0.8", "severity": "info"}
                ]
            }
        }
        
        return deployment_config
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "daily_metrics": {
                "total_tasks": 1_000_000,
                "success_rate": 0.95,
                "avg_duration": 45,  # seconds
                "data_extracted": 50_000_000,  # records
                "cost_per_task": 0.001  # USD
            },
            "resource_usage": {
                "browser_instances": 50,
                "proxy_ips": 1000,
                "storage_tb": 10,
                "bandwidth_gbps": 100
            },
            "quality_metrics": {
                "data_accuracy": 0.98,
                "completeness": 0.95,
                "freshness_hours": 24
            }
        }
```

---

## 23.20 本章小结（最终版）

| 知识点 | 核心要点 |
|:---|:---|
| **Web Agent 架构** | 任务解析 → 规划 → 执行 → 状态跟踪 |
| **页面理解** | DOM 分析、语义结构提取、交互元素识别 |
| **视觉理解** | 多模态分析、截图理解、视觉定位 |
| **导航策略** | 任务规划、智能导航、弹窗处理 |
| **错误处理** | 错误检测、恢复策略、操作回滚 |
| **框架选择** | Playwright（推荐）、Selenium、Puppeteer |
| **生产实践** | 电商采集、表单填写、数据提取 |
| **评测基准** | WebArena 提供标准化评测环境 |
| **语义理解** | 页面结构分析、可访问性检查 |
| **智能表单** | 自动识别字段、智能填写、验证 |
| **验证码处理** | 识别类型、自动解决、滑块验证 |
| **反爬虫** | 随机延迟、指纹伪装、代理轮换 |
| **并发控制** | 任务队列、速率限制、资源管理 |
| **数据提取** | 结构化数据、表格、链接、图片 |
| **最佳实践** | 幂等性、容错性、可观测性、安全性 |
| **性能优化** | 无头模式、资源拦截、实例复用 |
| **生产部署** | 大规模架构、监控告警、质量保证 |

> **下一章预告**
>
> 在第 24 章中，我们将学习数据分析 Agent。
