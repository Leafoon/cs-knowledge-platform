> **本章目标**：深入理解 Tool Calling 与 Function Calling 机制，掌握工具定义、绑定、执行、错误处理以及 Agent 工具集成，构建具备外部能力的 LLM 应用。

---

## 本章导览

本章聚焦于赋予 LLM 调用外部工具的能力，这是构建实用 Agent 的核心技术：

- **Tool Calling 基础**：理解工具调用的完整生命周期，从定义到执行
- **Function Calling**：使用 OpenAI Function Calling、Anthropic Tool Use 等原生 API
- **工具定义与绑定**：使用 `@tool` 装饰器、`StructuredTool`、`BaseTool` 定义工具
- **工具执行与编排**：实现工具调用链、并行工具调用、条件工具选择
- **错误处理与安全**：工具执行失败处理、权限控制、危险操作保护
- **Agent 工具集成**：将工具集成到 ReAct、OpenAI Functions Agent 等架构

掌握这些技术将让你的 LLM 应用能够查询数据库、调用 API、执行代码、操作文件系统等。

---

## 8.1 Tool Calling 完整生命周期

### 8.1.1 什么是 Tool Calling？

Tool Calling 是让 LLM 决定何时、如何调用外部工具的机制。完整流程包括：

1. **工具定义**：描述工具的功能、参数、返回值
2. **工具绑定**：将工具附加到 LLM 模型
3. **LLM 决策**：模型根据用户输入决定是否调用工具
4. **参数提取**：模型生成工具调用的参数
5. **工具执行**：实际执行工具函数
6. **结果返回**：将工具执行结果返回给 LLM
7. **最终响应**：LLM 基于工具结果生成用户回复

### 8.1.2 第一个工具：天气查询

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 1. 定义工具
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: The name of the city to get weather for.
    """
    # 模拟天气查询
    weather_data = {
        "Beijing": "Sunny, 15°C",
        "Shanghai": "Rainy, 18°C",
        "Shenzhen": "Cloudy, 22°C"
    }
    return weather_data.get(city, f"Weather data not available for {city}")

# 2. 绑定工具到模型
model = ChatOpenAI(model="gpt-4")
model_with_tools = model.bind_tools([get_weather])

# 3. 调用
response = model_with_tools.invoke("What's the weather in Beijing?")

print(response.content)
# ""  (内容为空，因为模型选择调用工具)

print(response.tool_calls)
# [{'name': 'get_weather', 'args': {'city': 'Beijing'}, 'id': 'call_abc123'}]

# 4. 执行工具
tool_call = response.tool_calls[0]
tool_result = get_weather.invoke(tool_call["args"])
print(tool_result)
# "Sunny, 15°C"

# 5. 将结果返回给 LLM
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

messages = [
    HumanMessage(content="What's the weather in Beijing?"),
    AIMessage(content="", tool_calls=response.tool_calls),
    ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
]

final_response = model.invoke(messages)
print(final_response.content)
# "The weather in Beijing is sunny with a temperature of 15°C."
```

### 8.1.3 自动工具调用（invoke_tools）

LangChain 提供了自动执行工具的辅助函数。

```python
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# 创建工具链
tools = [multiply, add]
model_with_tools = model.bind_tools(tools)

# 使用 RunnableLambda 自动执行
from langchain_core.runnables import RunnableLambda

def call_tools(msg):
    """自动执行所有工具调用"""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls
    
    return [
        ToolMessage(
            content=str(tool_map[tc["name"]].invoke(tc["args"])),
            tool_call_id=tc["id"]
        )
        for tc in tool_calls
    ]

chain = model_with_tools | RunnableLambda(call_tools)

# 测试
result = chain.invoke("What is 23 times 7?")
print(result)
# [ToolMessage(content='161', tool_call_id='call_xyz')]
```

### 8.1.4 完整对话链

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools."),
    MessagesPlaceholder(variable_name="messages"),
])

# 构建链
chain = prompt | model_with_tools

# 使用
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

messages = [HumanMessage(content="What is 15 + 27?")]

# 第一轮：LLM 决定调用工具
response = chain.invoke({"messages": messages})
messages.append(response)

# 执行工具
if response.tool_calls:
    for tool_call in response.tool_calls:
        tool_result = add.invoke(tool_call["args"])
        messages.append(ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"]
        ))

# 第二轮：LLM 基于工具结果回复
final_response = chain.invoke({"messages": messages})
print(final_response.content)
# "15 + 27 equals 42."
```

<div data-component="ToolCallingFlow"></div>

---

## 8.2 Function Calling 深度解析

### 8.2.1 OpenAI Function Calling

OpenAI 的 Function Calling 是原生支持的工具调用机制。

```python
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function

# 定义工具
@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the database for relevant information.
    
    Args:
        query: The search query string.
        limit: Maximum number of results to return.
    """
    return f"Found {limit} results for: {query}"

# 转换为 OpenAI Function 格式
openai_function = convert_to_openai_function(search_database)

print(openai_function)
# {
#     'name': 'search_database',
#     'description': 'Search the database for relevant information.',
#     'parameters': {
#         'type': 'object',
#         'properties': {
#             'query': {'type': 'string', 'description': 'The search query string.'},
#             'limit': {'type': 'integer', 'description': 'Maximum number of results.', 'default': 10}
#         },
#         'required': ['query']
#     }
# }

# 绑定函数
model = ChatOpenAI(model="gpt-4")
model_with_functions = model.bind_functions([search_database])

response = model_with_functions.invoke("Search for LangChain tutorials")
print(response.additional_kwargs["function_call"])
# {'name': 'search_database', 'arguments': '{"query": "LangChain tutorials", "limit": 10}'}
```

### 8.2.2 Anthropic Tool Use

Anthropic Claude 也原生支持工具调用。

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

@tool
def get_stock_price(symbol: str) -> float:
    """Get the current stock price for a given symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL').
    """
    # 模拟股价查询
    prices = {"AAPL": 175.50, "GOOGL": 140.20, "MSFT": 380.75}
    return prices.get(symbol, 0.0)

model_with_tools = model.bind_tools([get_stock_price])

response = model_with_tools.invoke("What's the price of Apple stock?")
print(response.tool_calls)
# [{'name': 'get_stock_price', 'args': {'symbol': 'AAPL'}, 'id': 'toolu_123'}]
```

### 8.2.3 强制工具调用（tool_choice）

可以强制模型必须调用特定工具。

```python
# 强制调用任意工具
model_with_tools = model.bind_tools(
    [get_weather, get_stock_price],
    tool_choice="any"  # 必须调用某个工具
)

# 强制调用特定工具
model_with_tools = model.bind_tools(
    [get_weather],
    tool_choice="get_weather"  # 必须调用 get_weather
)

# 禁止调用工具
model_with_tools = model.bind_tools(
    [get_weather],
    tool_choice="none"  # 不允许调用工具
)

# 自动决定（默认）
model_with_tools = model.bind_tools(
    [get_weather],
    tool_choice="auto"  # 让模型自己决定
)
```

### 8.2.4 并行工具调用

现代 LLM 支持在一次响应中调用多个工具。

```python
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny"

@tool
def get_time(timezone: str) -> str:
    """Get current time in a timezone."""
    return f"Current time in {timezone}: 14:30"

model_with_tools = model.bind_tools([get_weather, get_time])

response = model_with_tools.invoke(
    "What's the weather in Beijing and the current time in Asia/Shanghai?"
)

print(len(response.tool_calls))
# 2

for tool_call in response.tool_calls:
    print(f"{tool_call['name']}: {tool_call['args']}")
# get_weather: {'city': 'Beijing'}
# get_time: {'timezone': 'Asia/Shanghai'}
```

<div data-component="FunctionSchemaBuilder"></div>

---

## 8.3 工具定义的多种方式

### 8.3.1 使用 @tool 装饰器

最简单的工具定义方式。

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A valid Python mathematical expression.
    """
    try:
        return eval(expression)
    except Exception as e:
        return f"Error: {str(e)}"

# 查看工具属性
print(calculator.name)  # calculator
print(calculator.description)  # Evaluate a mathematical expression.
print(calculator.args_schema)  # Pydantic model
```

### 8.3.2 StructuredTool.from_function

从现有函数创建工具。

```python
from langchain_core.tools import StructuredTool

def reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]

reverse_tool = StructuredTool.from_function(
    func=reverse_string,
    name="reverse_string",
    description="Reverse the characters in a string"
)

result = reverse_tool.invoke({"text": "hello"})
print(result)  # "olleh"
```

### 8.3.3 继承 BaseTool

完全自定义工具行为。

```python
from langchain_core.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class DatabaseQueryInput(BaseModel):
    """Input for database query tool."""
    sql: str = Field(description="SQL query to execute")
    database: str = Field(default="main", description="Database name")

class DatabaseQueryTool(BaseTool):
    name: str = "database_query"
    description: str = "Execute a SQL query on the database"
    args_schema: Type[BaseModel] = DatabaseQueryInput
    
    def _run(self, sql: str, database: str = "main") -> str:
        """Execute the query."""
        # 模拟数据库查询
        return f"Executed '{sql}' on {database} database"
    
    async def _arun(self, sql: str, database: str = "main") -> str:
        """Async version."""
        return self._run(sql, database)

# 使用
db_tool = DatabaseQueryTool()
result = db_tool.invoke({"sql": "SELECT * FROM users", "database": "production"})
print(result)
```

### 8.3.4 带状态的工具

工具可以维护内部状态。

```python
from langchain_core.tools import BaseTool
from typing import List

class CounterTool(BaseTool):
    name: str = "counter"
    description: str = "Increment and get counter value"
    
    # 内部状态
    count: int = 0
    history: List[int] = []
    
    def _run(self, action: str = "increment") -> str:
        if action == "increment":
            self.count += 1
            self.history.append(self.count)
            return f"Counter: {self.count}"
        elif action == "reset":
            self.count = 0
            self.history = []
            return "Counter reset"
        elif action == "history":
            return f"History: {self.history}"
        return "Unknown action"

counter = CounterTool()
print(counter.invoke({"action": "increment"}))  # Counter: 1
print(counter.invoke({"action": "increment"}))  # Counter: 2
print(counter.invoke({"action": "history"}))    # History: [1, 2]
```

---

## 8.4 工具执行与编排

### 8.4.1 条件工具选择

根据用户意图选择合适的工具。

```python
from langchain_core.prompts import ChatPromptTemplate

@tool
def calculator(expression: str) -> float:
    """Calculate mathematical expressions."""
    return eval(expression)

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"

tools = [calculator, search_web, send_email]

# 创建智能路由
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the appropriate tool to answer the user's question."),
    ("human", "{input}")
])

model_with_tools = model.bind_tools(tools)
chain = prompt | model_with_tools

# 测试不同类型的查询
queries = [
    "What is 25 * 37?",
    "Search for LangChain documentation",
    "Send email to john@example.com about meeting"
]

for query in queries:
    response = chain.invoke({"input": query})
    if response.tool_calls:
        print(f"Query: {query}")
        print(f"Tool: {response.tool_calls[0]['name']}")
        print()
```

### 8.4.2 工具链组合

一个工具的输出作为另一个工具的输入。

```python
@tool
def extract_keywords(text: str) -> str:
    """Extract keywords from text."""
    # 简化示例
    words = text.split()
    return ", ".join(words[:3])

@tool
def search_documents(keywords: str) -> str:
    """Search documents by keywords."""
    return f"Found documents for: {keywords}"

# 手动链式调用
text = "LangChain is a framework for building LLM applications"
keywords = extract_keywords.invoke({"text": text})
results = search_documents.invoke({"keywords": keywords})

print(results)
# "Found documents for: LangChain, is, a"
```

### 8.4.3 工具执行超时与重试

```python
import time
from langchain_core.tools import tool

@tool
def slow_api_call(endpoint: str) -> str:
    """Call a slow external API.
    
    Args:
        endpoint: API endpoint to call.
    """
    time.sleep(5)  # 模拟慢速 API
    return f"Response from {endpoint}"

# 添加超时装饰器
from functools import wraps
import signal

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Function timed out after {seconds}s")
            
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator

@tool
@timeout(3)
def fast_api_call(endpoint: str) -> str:
    """Call API with 3s timeout."""
    time.sleep(1)
    return f"Fast response from {endpoint}"

# 使用
try:
    result = fast_api_call.invoke({"endpoint": "/data"})
    print(result)
except TimeoutError as e:
    print(f"Tool execution failed: {e}")
```

<div data-component="ToolExecutionTimeline"></div>

---

## 8.5 错误处理与安全

### 8.5.1 工具执行错误处理

```python
from langchain_core.tools import tool
from typing import Union

@tool
def divide(a: float, b: float) -> Union[float, str]:
    """Divide two numbers.
    
    Args:
        a: Numerator.
        b: Denominator.
    """
    try:
        if b == 0:
            return "Error: Division by zero"
        return a / b
    except Exception as e:
        return f"Error: {str(e)}"

# 测试
print(divide.invoke({"a": 10, "b": 2}))   # 5.0
print(divide.invoke({"a": 10, "b": 0}))   # "Error: Division by zero"
```

### 8.5.2 工具权限控制

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class ProtectedTool(BaseTool):
    name: str = "delete_file"
    description: str = "Delete a file (requires admin privileges)"
    
    # 权限配置
    required_role: str = "admin"
    current_user_role: str = "user"
    
    def _run(self, filename: str) -> str:
        # 检查权限
        if self.current_user_role != self.required_role:
            return f"Permission denied: requires {self.required_role} role"
        
        # 执行操作
        return f"Deleted file: {filename}"

# 使用
tool = ProtectedTool(current_user_role="user")
print(tool.invoke({"filename": "data.txt"}))
# "Permission denied: requires admin role"

tool = ProtectedTool(current_user_role="admin")
print(tool.invoke({"filename": "data.txt"}))
# "Deleted file: data.txt"
```

### 8.5.3 危险操作确认

```python
from langchain_core.tools import tool

@tool
def execute_system_command(command: str) -> str:
    """Execute a system command (DANGEROUS).
    
    Args:
        command: Shell command to execute.
    """
    # 危险命令黑名单
    dangerous_commands = ["rm -rf", "format", "dd if=", ":(){:|:&};:"]
    
    for dangerous in dangerous_commands:
        if dangerous in command.lower():
            return f"BLOCKED: Command contains dangerous pattern '{dangerous}'"
    
    # 需要用户确认
    print(f"⚠️  About to execute: {command}")
    confirmation = input("Type 'yes' to confirm: ")
    
    if confirmation.lower() != 'yes':
        return "Command cancelled by user"
    
    # 执行命令（示例中不实际执行）
    return f"Would execute: {command}"
```

### 8.5.4 工具输入验证

```python
from pydantic import BaseModel, Field, field_validator

class EmailInput(BaseModel):
    """Validated email input."""
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body")
    
    @field_validator('to')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v or '.' not in v:
            raise ValueError('Invalid email address format')
        return v
    
    @field_validator('subject')
    @classmethod
    def validate_subject(cls, v: str) -> str:
        if len(v) > 100:
            raise ValueError('Subject too long (max 100 characters)')
        return v

@tool(args_schema=EmailInput)
def send_email_validated(to: str, subject: str, body: str) -> str:
    """Send a validated email."""
    return f"Email sent to {to}: {subject}"

# 测试
try:
    result = send_email_validated.invoke({
        "to": "invalid-email",
        "subject": "Test",
        "body": "Hello"
    })
except ValueError as e:
    print(f"Validation error: {e}")
```

---

## 8.6 Agent 工具集成

### 8.6.1 ReAct Agent 与工具

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain import hub

# 定义工具集
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

@tool
def get_current_date() -> str:
    """Returns the current date."""
    from datetime import date
    return str(date.today())

tools = [get_word_length, get_current_date]

# 加载 ReAct 提示
prompt = hub.pull("hwchase17/react")

# 创建 agent
model = ChatOpenAI(model="gpt-4", temperature=0)
agent = create_react_agent(model, tools, prompt)

# 创建 executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# 使用
result = agent_executor.invoke({
    "input": "What is the length of the word 'LangChain'?"
})

print(result["output"])
# "The length of the word 'LangChain' is 9."
```

### 8.6.2 OpenAI Functions Agent

```python
from langchain.agents import create_openai_functions_agent

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information."""
    return f"Wikipedia result for: {query}"

@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

tools = [search_wikipedia, calculate]

# 创建提示
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 创建 agent
model = ChatOpenAI(model="gpt-4", temperature=0)
agent = create_openai_functions_agent(model, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# 使用
result = agent_executor.invoke({
    "input": "What is 15 * 23, and then search Wikipedia for that number?"
})
```

### 8.6.3 自定义工具包（Toolkit）

```python
from langchain_core.tools import BaseTool
from typing import List

class MathToolkit:
    """A collection of mathematical tools."""
    
    @staticmethod
    def get_tools() -> List[BaseTool]:
        @tool
        def add(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b
        
        @tool
        def subtract(a: float, b: float) -> float:
            """Subtract b from a."""
            return a - b
        
        @tool
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers."""
            return a * b
        
        @tool
        def divide(a: float, b: float) -> float:
            """Divide a by b."""
            if b == 0:
                raise ValueError("Division by zero")
            return a / b
        
        return [add, subtract, multiply, divide]

# 使用
math_tools = MathToolkit.get_tools()
print(f"Loaded {len(math_tools)} math tools")

model_with_tools = model.bind_tools(math_tools)
```

---

## 本章小结

本章深入学习了 Tool Calling 与 Function Calling：

✅ **完整生命周期**：工具定义、绑定、LLM 决策、参数提取、执行、结果返回  
✅ **Function Calling**：OpenAI、Anthropic 原生 API，tool_choice 控制  
✅ **工具定义**：@tool 装饰器、StructuredTool、BaseTool 继承  
✅ **执行与编排**：条件选择、工具链、并行调用、超时重试  
✅ **安全与容错**：错误处理、权限控制、输入验证、危险操作保护  
✅ **Agent 集成**：ReAct、OpenAI Functions Agent、自定义工具包

这些技术是构建实用 Agent 的基础，让 LLM 能够与外部世界交互。

---

## 扩展阅读

- [Tools](https://python.langchain.com/docs/modules/agents/tools/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Agents](https://python.langchain.com/docs/modules/agents/)
- [Custom Tools](https://python.langchain.com/docs/modules/agents/tools/custom_tools)
