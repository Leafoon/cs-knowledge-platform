# Chapter 25: LangServe 基础

## 本章概览

LangServe 是 LangChain 官方的生产部署解决方案，帮助你将构建好的链快速部署为 **RESTful API**。基于 FastAPI 构建，LangServe 提供自动生成的 OpenAPI 文档、内置 Playground、流式响应、批处理等企业级特性。本章将学习如何使用 LangServe 将 LangChain 应用从开发环境无缝迁移到生产环境。

**本章重点**：
- LangServe 核心概念与架构
- 第一个 LangServe 应用：从安装到部署
- 支持的端点：/invoke、/batch、/stream、/playground
- 客户端调用：RemoteRunnable、HTTP 请求
- 配置化部署：运行时参数与多版本管理

---

## 25.1 LangServe 概览

### 25.1.1 为什么需要 LangServe？

<div data-component="LangServeArchitecture"></div>

**开发与生产的鸿沟**：

```python
# 开发环境：直接调用
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm

result = chain.invoke({"text": "Hello"})
print(result.content)  # "Bonjour"
```

**生产环境需求**：

| 需求 | 开发环境 | 生产环境 |
|------|---------|---------|
| **调用方式** | 本地 Python 代码 | HTTP API（跨语言） |
| **并发处理** | 单线程 | 多线程/异步 |
| **API 文档** | 无 | 自动生成 OpenAPI |
| **监控追踪** | print() | 结构化日志 + Tracing |
| **部署管理** | 手动运行 | 容器化 + 编排 |

**LangServe 解决方案**：

✅ **一行代码部署**：`add_routes(app, chain, path="/translate")`  
✅ **自动 API 文档**：访问 `/docs` 即可查看完整 OpenAPI 规范  
✅ **内置 Playground**：访问 `/translate/playground` 在线测试  
✅ **流式响应**：支持 SSE（Server-Sent Events）实时输出  
✅ **批处理优化**：自动聚合多个请求提升吞吐  

### 25.1.2 核心功能：REST API + Playground

**LangServe 提供的端点**：

```
/translate/invoke        → 单次调用（同步）
/translate/batch         → 批量调用
/translate/stream        → 流式输出（SSE）
/translate/stream_events → 事件流（细粒度）
/translate/playground    → 交互式 UI
/translate/input_schema  → 输入 Schema（JSON）
/translate/output_schema → 输出 Schema（JSON）
```

### 25.1.3 与 FastAPI 的关系

LangServe 基于 FastAPI 构建，所以你可以：

```python
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI()  # 标准 FastAPI 应用

# 添加 LangChain 路由
add_routes(app, chain, path="/translate")

# 添加其他 FastAPI 路由
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# 中间件、认证、CORS 等都可以正常使用
```

---

## 25.2 第一个 LangServe 应用

### 25.2.1 安装 langserve

```bash
# 安装 LangServe（包含 FastAPI 和 uvicorn）
pip install "langserve[all]"

# 或者分开安装
pip install langserve fastapi uvicorn
```

**依赖说明**：

- `langserve`：核心库
- `fastapi`：Web 框架
- `uvicorn`：ASGI 服务器（生产级）
- `sse-starlette`（可选）：流式响应支持

### 25.2.2 add_routes()：注册链

**最小示例**：

```python
# server.py
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes

# 创建链
prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm

# 创建 FastAPI 应用
app = FastAPI(
    title="Translation API",
    version="1.0",
    description="LangChain-powered translation service"
)

# 注册链
add_routes(
    app,
    chain,
    path="/translate",
    enabled_endpoints=["invoke", "batch", "stream", "playground"],
)

# 就这么简单！
```

### 25.2.3 启动服务（uvicorn）

```bash
# 开发模式（自动重载）
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# 生产模式（多 worker）
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

**启动日志**：

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345]
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 25.2.4 访问 /docs（OpenAPI）

打开浏览器访问 [http://localhost:8000/docs](http://localhost:8000/docs)

**你会看到**：

```
Translation API
Version: 1.0

Endpoints:
  POST /translate/invoke
  POST /translate/batch
  POST /translate/stream
  GET  /translate/playground
  GET  /translate/input_schema
  GET  /translate/output_schema
```

**在 Swagger UI 中测试**：

1. 展开 `POST /translate/invoke`
2. 点击 **Try it out**
3. 输入 JSON：
   ```json
   {
     "input": {
       "text": "Hello, world!"
     }
   }
   ```
4. 点击 **Execute**
5. 查看响应：
   ```json
   {
     "output": {
       "content": "Bonjour, le monde !",
       ...
     }
   }
   ```

---

## 25.3 支持的端点

### 25.3.1 /invoke：单次调用

<div data-component="EndpointExplorer"></div>

**请求示例**：

```python
import requests

url = "http://localhost:8000/translate/invoke"
payload = {
    "input": {"text": "Good morning"}
}

response = requests.post(url, json=payload)
print(response.json())
```

**响应**：

```json
{
  "output": {
    "content": "Bonjour",
    "additional_kwargs": {},
    "type": "ai",
    "example": false
  },
  "callback_events": [],
  "metadata": {
    "run_id": "abc-123-def-456"
  }
}
```

### 25.3.2 /batch：批量调用

**请求示例**：

```python
url = "http://localhost:8000/translate/batch"
payload = {
    "inputs": [
        {"text": "Hello"},
        {"text": "Goodbye"},
        {"text": "Thank you"}
    ]
}

response = requests.post(url, json=payload)
results = response.json()

for result in results["output"]:
    print(result["content"])
```

**输出**：

```
Bonjour
Au revoir
Merci
```

**优势**：

- ✅ 一次 HTTP 请求处理多个翻译
- ✅ 减少网络开销
- ✅ 后端可以批量调用 LLM（更高吞吐）

### 25.3.3 /stream：流式输出

**请求示例**：

```python
import requests

url = "http://localhost:8000/translate/stream"
payload = {
    "input": {"text": "The weather is nice today"}
}

with requests.post(url, json=payload, stream=True) as response:
    for line in response.iter_lines():
        if line:
            # Server-Sent Events 格式
            if line.startswith(b"data:"):
                data = line[5:].decode('utf-8')
                print(data, end="", flush=True)
```

**输出**（逐字符流式）：

```
Il 
fait 
beau 
aujourd
'
hui
.
```

**适用场景**：

- 聊天机器人（实时显示回复）
- 长文档摘要（边生成边展示）
- 代码生成（增强用户体验）

### 25.3.4 /stream_events：事件流

**更细粒度的流式输出**：

```python
url = "http://localhost:8000/translate/stream_events"
payload = {
    "input": {"text": "Hello"},
    "version": "v1"  # 事件流版本
}

with requests.post(url, json=payload, stream=True) as response:
    for line in response.iter_lines():
        if line:
            event = json.loads(line[5:])  # 移除 "data:"
            
            # 事件类型
            if event["event"] == "on_chain_start":
                print(f"Chain started: {event['name']}")
            elif event["event"] == "on_llm_stream":
                print(event["data"]["chunk"], end="", flush=True)
            elif event["event"] == "on_chain_end":
                print("\nChain completed")
```

### 25.3.5 /playground：交互式 UI

访问 [http://localhost:8000/translate/playground](http://localhost:8000/translate/playground)

**Playground 功能**：

- ✅ 在线输入测试数据
- ✅ 查看流式输出
- ✅ 查看完整 trace
- ✅ 调整链配置（如 temperature）
- ✅ 分享测试链接

**截图示例**：

```
┌─────────────────────────────────────┐
│ Translation API Playground          │
├─────────────────────────────────────┤
│ Input:                              │
│ {                                   │
│   "text": "Hello, world!"           │
│ }                                   │
│                                     │
│ [Run]  [Stream]  [Clear]            │
├─────────────────────────────────────┤
│ Output:                             │
│ Bonjour, le monde !                 │
└─────────────────────────────────────┘
```

---

## 25.4 客户端调用

### 25.4.1 RemoteRunnable：Python 客户端

<div data-component="RemoteRunnableDemo"></div>

**最方便的调用方式**：

```python
from langserve import RemoteRunnable

# 创建远程链
remote_chain = RemoteRunnable("http://localhost:8000/translate")

# 像本地链一样使用
result = remote_chain.invoke({"text": "Hello"})
print(result.content)  # "Bonjour"

# 批量调用
results = remote_chain.batch([
    {"text": "Hello"},
    {"text": "Goodbye"}
])

# 流式调用
for chunk in remote_chain.stream({"text": "Good morning"}):
    print(chunk.content, end="", flush=True)
```

**优势**：

- ✅ API 与本地链完全一致
- ✅ 支持 invoke、batch、stream、astream 等所有方法
- ✅ 自动处理 HTTP 请求/响应
- ✅ 可以在 LCEL 中组合使用

**在 LCEL 中使用**：

```python
from langserve import RemoteRunnable
from langchain_core.output_parsers import StrOutputParser

# 远程翻译链
remote_translate = RemoteRunnable("http://localhost:8000/translate")

# 本地后处理
chain = remote_translate | StrOutputParser() | (lambda x: x.upper())

result = chain.invoke({"text": "Hello"})
print(result)  # "BONJOUR"
```

### 25.4.2 HTTP 请求示例（curl、requests）

**curl 命令**：

```bash
# Invoke
curl -X POST "http://localhost:8000/translate/invoke" \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "Hello"}}'

# Stream（需要处理 SSE）
curl -X POST "http://localhost:8000/translate/stream" \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "Hello"}}' \
  --no-buffer
```

**Python requests**：

```python
import requests

# Invoke
response = requests.post(
    "http://localhost:8000/translate/invoke",
    json={"input": {"text": "Hello"}}
)
print(response.json()["output"]["content"])

# Batch
response = requests.post(
    "http://localhost:8000/translate/batch",
    json={"inputs": [{"text": "Hello"}, {"text": "Goodbye"}]}
)
for result in response.json()["output"]:
    print(result["content"])
```

### 25.4.3 JavaScript/TypeScript 客户端

```typescript
// 使用 fetch API
async function translateText(text: string): Promise<string> {
  const response = await fetch('http://localhost:8000/translate/invoke', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      input: { text }
    }),
  });

  const data = await response.json();
  return data.output.content;
}

// 使用
const result = await translateText("Hello");
console.log(result);  // "Bonjour"
```

**流式调用（EventSource）**：

```typescript
const eventSource = new EventSource(
  'http://localhost:8000/translate/stream?input={"text":"Hello"}'
);

eventSource.onmessage = (event) => {
  const chunk = JSON.parse(event.data);
  console.log(chunk);
};
```

---

## 25.5 配置化部署

### 25.5.1 ConfigurableField 暴露

**问题场景**：不同用户想使用不同模型

```python
# ❌ 硬编码模型
llm = ChatOpenAI(model="gpt-4")

# ✅ 可配置模型
from langchain_core.runnables import ConfigurableField

llm = ChatOpenAI(model="gpt-4").configurable_fields(
    model=ConfigurableField(
        id="model_name",
        name="Model Name",
        description="The OpenAI model to use",
    )
)

chain = prompt | llm

# 部署
add_routes(app, chain, path="/translate")
```

### 25.5.2 运行时参数传递

**客户端调用时指定模型**：

```python
from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/translate")

# 使用 GPT-4
result1 = remote_chain.invoke(
    {"text": "Hello"},
    config={"configurable": {"model_name": "gpt-4"}}
)

# 使用 GPT-3.5（更便宜）
result2 = remote_chain.invoke(
    {"text": "Hello"},
    config={"configurable": {"model_name": "gpt-3.5-turbo"}}
)
```

**HTTP 请求传递配置**：

```python
import requests

response = requests.post(
    "http://localhost:8000/translate/invoke",
    json={
        "input": {"text": "Hello"},
        "config": {
            "configurable": {
                "model_name": "gpt-3.5-turbo"
            }
        }
    }
)
```

### 25.5.3 多版本模型切换

**高级配置：暴露多个参数**

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import ConfigurableField

llm = ChatOpenAI(model="gpt-4", temperature=0.7).configurable_fields(
    model=ConfigurableField(
        id="model_name",
        name="Model",
        description="The LLM model to use",
    ),
    temperature=ConfigurableField(
        id="llm_temperature",
        name="Temperature",
        description="Sampling temperature (0-2)",
    ),
)

chain = prompt | llm

# 客户端可以自由组合
result = remote_chain.invoke(
    {"text": "Hello"},
    config={
        "configurable": {
            "model_name": "gpt-3.5-turbo",
            "llm_temperature": 0.3
        }
    }
)
```

**ConfigurableAlternatives：预设配置**

```python
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4").configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="gpt4",
    gpt35=ChatOpenAI(model="gpt-3.5-turbo"),
    claude=ChatAnthropic(model="claude-3-opus-20240229"),
)

# 客户端选择
remote_chain.invoke(
    {"text": "Hello"},
    config={"configurable": {"llm": "gpt35"}}  # 切换到 GPT-3.5
)
```

---

## 25.6 最佳实践

### 25.6.1 错误处理

```python
from fastapi import HTTPException

@app.exception_handler(Exception)
async def custom_exception_handler(request, exc):
    """统一错误处理"""
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "path": str(request.url)
        }
    )
```

### 25.6.2 添加认证

```python
from fastapi import Header, HTTPException

async def verify_token(authorization: str = Header(None)):
    """API Key 认证"""
    if authorization != "Bearer YOUR_SECRET_KEY":
        raise HTTPException(status_code=401, detail="Unauthorized")

# 添加依赖
add_routes(
    app,
    chain,
    path="/translate",
    dependencies=[Depends(verify_token)]
)
```

### 25.6.3 CORS 配置

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 25.6.4 性能优化

```python
# 启用压缩
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 连接池
llm = ChatOpenAI(
    model="gpt-4",
    max_retries=3,
    request_timeout=60,
)
```

---

## 本章总结

**核心收获**：

1. ✅ **LangServe = FastAPI + LangChain**：一行代码将链部署为 REST API
2. ✅ **自动生成文档**：访问 `/docs` 查看完整 OpenAPI 规范
3. ✅ **多种端点**：invoke（单次）、batch（批量）、stream（流式）、playground（在线测试）
4. ✅ **跨语言调用**：Python、JavaScript、curl 都可以调用
5. ✅ **配置化部署**：运行时切换模型、参数，无需重启服务

**LangServe 优势总结**：

| 特性 | 传统方式 | LangServe |
|------|---------|-----------|
| **部署难度** | 需手动编写 API | 一行代码 |
| **API 文档** | 手动编写 | 自动生成 |
| **在线测试** | 需自建 UI | 内置 Playground |
| **流式输出** | 复杂实现 | 开箱即用 |
| **批处理** | 手动优化 | 自动聚合 |

**下一章预告**：
Chapter 26 将学习 **LangServe 高级特性**，掌握认证授权、速率限制、监控日志等生产环境必备功能。

---

## 练习题

### 基础练习

1. **部署第一个 API**：将一个简单的翻译链部署为 LangServe API。

2. **测试所有端点**：使用 Swagger UI 测试 /invoke、/batch、/stream 端点。

3. **RemoteRunnable 调用**：使用 RemoteRunnable 调用部署的 API。

### 进阶练习

4. **配置化模型**：使用 ConfigurableField 暴露模型参数，支持运行时切换 GPT-4 和 GPT-3.5。

5. **添加认证**：为 API 添加 API Key 认证中间件。

6. **JavaScript 客户端**：使用 fetch API 调用 LangServe 端点。

### 挑战练习

7. **多链部署**：在同一个 FastAPI 应用中部署多个不同功能的链（翻译、摘要、问答）。

8. **流式前端**：实现一个网页，展示 LangServe 流式输出（使用 EventSource）。

9. **性能对比**：对比 invoke 和 batch 端点的吞吐量差异。

---

## 扩展阅读

- [LangServe Documentation](https://python.langchain.com/docs/langserve)
- [LangServe GitHub Repository](https://github.com/langchain-ai/langserve)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Deploying LangChain Apps Tutorial](https://blog.langchain.dev/deploying-langchain-apps/)
