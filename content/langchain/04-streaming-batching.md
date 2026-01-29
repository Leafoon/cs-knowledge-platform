> **本章目标**：掌握 LCEL 的流式输出和批处理能力，学习异步编程最佳实践，优化应用性能。

---

## 本章导览

本章聚焦性能优化与用户体验提升，掌握现代 LLM 应用的核心技术：

- **流式输出**：`astream`、`astream_events` 实现打字机效果，提升用户感知速度
- **批处理**：`batch` 接口批量处理请求，最大化 GPU 利用率
- **异步编程**：`ainvoke`、`abatch` 等异步方法的正确使用姿势
- **事件监听**：`astream_events` 监听链执行过程中的细粒度事件
- **性能对比**：同步 vs 异步、流式 vs 批处理的实测数据与选型建议

这些技术将帮助你构建高性能、用户体验优秀的 LLM 应用。

---

## 4.1 流式输出(Streaming)

流式输出允许应用在 LLM 生成内容时逐块接收结果,而不是等待完整响应,极大提升用户体验。

### 4.1.1 astream()：异步流式

**基础用法**:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Write a story about {topic}")
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt | model | parser

# 异步流式输出
import asyncio

async def stream_story():
    async for chunk in chain.astream({"topic": "a robot"}):
        print(chunk, end="", flush=True)
        await asyncio.sleep(0.01)  # 模拟打字效果

asyncio.run(stream_story())
```

**输出效果**:

```
Once... upon... a... time..., there... was... a... robot... named... R2...
```

### 4.1.2 astream_events()：事件流

**事件流** 提供更细粒度的控制,可以监听链中每个组件的事件。

```python
async def detailed_stream():
    async for event in chain.astream_events(
        {"topic": "quantum physics"},
        version="v1"
    ):
        kind = event["event"]
        
        if kind == "on_chat_model_start":
            print("🚀 Model started")
        
        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            print(chunk.content, end="", flush=True)
        
        elif kind == "on_chat_model_end":
            print("\n✅ Model finished")

asyncio.run(detailed_stream())
```

**事件类型**:

| 事件 | 触发时机 | 数据 |
|------|----------|------|
| `on_chain_start` | 链开始执行 | 输入数据 |
| `on_chat_model_start` | 模型开始调用 | 提示消息 |
| `on_chat_model_stream` | 模型流式输出 | Token chunk |
| `on_chat_model_end` | 模型完成 | 完整响应 |
| `on_chain_end` | 链完成 | 最终输出 |

### 4.1.3 stream() vs astream() 性能对比

<div data-component="StreamingVisualizer"></div>

```python
import time

# 同步流式
def sync_stream():
    start = time.time()
    for chunk in chain.stream({"topic": "AI"}):
        pass
    return time.time() - start

# 异步流式
async def async_stream():
    start = time.time()
    async for chunk in chain.astream({"topic": "AI"}):
        pass
    return time.time() - start

# 性能对比
sync_time = sync_stream()
async_time = asyncio.run(async_stream())

print(f"Sync: {sync_time:.2f}s, Async: {async_time:.2f}s")
# 单次调用性能相近，但 async 支持并发
```

**并发场景**:

```python
async def concurrent_streams():
    tasks = [
        chain.astream({"topic": f"topic_{i}"})
        for i in range(10)
    ]
    
    results = await asyncio.gather(*tasks)
    # 10个流同时执行，总耗时 ≈ 单次耗时
```

### 4.1.4 流式 token 累积与实时显示

**Streamlit 集成**:

```python
import streamlit as st

st.title("Streaming Chat")

user_input = st.text_input("Ask a question:")

if user_input:
    response_placeholder = st.empty()
    full_response = ""
    
    for chunk in chain.stream({"input": user_input}):
        full_response += chunk
        response_placeholder.markdown(full_response + "▌")  # 闪烁光标
    
    response_placeholder.markdown(full_response)
```

**FastAPI 流式端点**:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream")
async def stream_response(query: str):
    async def generate():
        async for chunk in chain.astream({"input": query}):
            yield f"data: {chunk}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 4.2 批处理(Batching)

<div data-component="AsyncPerformanceComparison"></div>

批处理可以一次性处理多个输入,节省网络往返时间。

### 4.2.1 batch()：同步批量

```python
# 批量翻译
inputs = [
    {"text": "Hello", "language": "French"},
    {"text": "Goodbye", "language": "Spanish"},
    {"text": "Thank you", "language": "German"}
]

results = chain.batch(inputs)

for inp, out in zip(inputs, results):
    print(f"{inp['text']} → {out}")

# 输出:
# Hello → Bonjour
# Goodbye → Adiós
# Thank you → Danke
```

### 4.2.2 abatch()：异步批量

```python
async def async_batch():
    results = await chain.abatch(inputs)
    return results

results = asyncio.run(async_batch())
```

### 4.2.3 批处理大小优化

**自动批处理**:

```python
# 大批量输入自动分批
large_inputs = [{"text": f"Text {i}"} for i in range(1000)]

# 自动分成多个小批次处理
results = chain.batch(large_inputs, config={
    "max_concurrency": 10  # 最多10个并发请求
})
```

**手动批处理**:

```python
def process_in_batches(inputs, batch_size=10):
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        batch_results = chain.batch(batch)
        results.extend(batch_results)
        time.sleep(1)  # 避免速率限制
    return results

all_results = process_in_batches(large_inputs)
```

### 4.2.4 并发控制（max_concurrency）

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(max_concurrency=5)

# 最多5个并发请求
results = await chain.abatch(inputs, config=config)
```

**性能测试**:

```python
import time

async def test_concurrency(max_concurrency):
    config = RunnableConfig(max_concurrency=max_concurrency)
    start = time.time()
    await chain.abatch(inputs * 10, config=config)
    return time.time() - start

# 测试不同并发度
for concurrency in [1, 5, 10, 20]:
    elapsed = await test_concurrency(concurrency)
    print(f"Concurrency {concurrency}: {elapsed:.2f}s")

# 输出:
# Concurrency 1: 45.2s
# Concurrency 5: 12.3s
# Concurrency 10: 8.1s
# Concurrency 20: 7.8s (提升有限，受限于API)
```

---

## 4.3 异步编程最佳实践

### 4.3.1 ainvoke() vs invoke()

```python
# 同步调用（阻塞）
def sync_call():
    result = chain.invoke({"text": "Hello"})
    return result

# 异步调用（非阻塞）
async def async_call():
    result = await chain.ainvoke({"text": "Hello"})
    return result

# 并发执行10个任务
async def concurrent_calls():
    tasks = [async_call() for _ in range(10)]
    results = await asyncio.gather(*tasks)
    # 总耗时 ≈ 单次耗时（并发执行）
```

### 4.3.2 异步上下文管理

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def llm_context():
    """异步上下文管理器"""
    print("Setting up LLM...")
    model = ChatOpenAI(model="gpt-4o-mini")
    try:
        yield model
    finally:
        print("Cleaning up...")

# 使用
async def use_context():
    async with llm_context() as model:
        result = await model.ainvoke([HumanMessage(content="Hi")])
        print(result.content)
```

### 4.3.3 事件循环管理

```python
# ❌ 错误：嵌套事件循环
def nested_async():
    result = asyncio.run(chain.ainvoke({"text": "Hi"}))
    # RuntimeError: asyncio.run() cannot be called from a running event loop

# ✅ 正确：使用 await
async def proper_async():
    result = await chain.ainvoke({"text": "Hi"})
    return result

# 顶层调用
asyncio.run(proper_async())
```

### 4.3.4 Jupyter Notebook 中的异步

```python
# Jupyter 自带事件循环，直接用 await
result = await chain.ainvoke({"text": "Hello"})

# 或使用 IPython 的 %autoawait
%autoawait on
result = chain.ainvoke({"text": "Hello"})  # 自动 await
```

---

## 4.4 流式与批处理组合

### 4.4.1 批量流式输出

```python
async def batch_stream():
    """批量处理，每个都流式输出"""
    inputs = [{"text": f"Story {i}"} for i in range(3)]
    
    tasks = []
    for inp in inputs:
        async def process(input_data):
            result = ""
            async for chunk in chain.astream(input_data):
                result += chunk
            return result
        
        tasks.append(process(inp))
    
    results = await asyncio.gather(*tasks)
    return results

results = await batch_stream()
```

### 4.4.2 并行流处理

```python
async def parallel_streams():
    """并行处理多个流"""
    async def stream_one(topic):
        chunks = []
        async for chunk in chain.astream({"topic": topic}):
            chunks.append(chunk)
        return "".join(chunks)
    
    # 并行执行3个流
    results = await asyncio.gather(
        stream_one("AI"),
        stream_one("quantum physics"),
        stream_one("space exploration")
    )
    
    return results
```

### 4.4.3 背压控制（Backpressure）

```python
import asyncio
from collections.abc import AsyncIterator

async def controlled_stream(
    stream: AsyncIterator,
    max_buffer_size: int = 100
) -> AsyncIterator:
    """控制流式输出速率"""
    buffer = []
    
    async for chunk in stream:
        buffer.append(chunk)
        
        # 缓冲区满时暂停
        if len(buffer) >= max_buffer_size:
            yield "".join(buffer)
            buffer = []
    
    # 输出剩余
    if buffer:
        yield "".join(buffer)

# 使用
async def use_controlled_stream():
    stream = chain.astream({"topic": "long story"})
    async for batch in controlled_stream(stream, max_buffer_size=50):
        print(f"Batch: {len(batch)} chars")
        await asyncio.sleep(0.1)  # 控制处理速率
```

---

## 4.5 进度追踪与取消

### 4.5.1 进度回调

```python
from langchain.callbacks import AsyncCallbackHandler

class ProgressCallback(AsyncCallbackHandler):
    def __init__(self):
        self.tokens = 0
    
    async def on_llm_new_token(self, token: str, **kwargs):
        self.tokens += 1
        if self.tokens % 10 == 0:
            print(f"Progress: {self.tokens} tokens generated")

# 使用
progress = ProgressCallback()
result = await chain.ainvoke(
    {"topic": "AI"},
    config={"callbacks": [progress]}
)
```

### 4.5.2 任务取消（cancellation）

```python
async def cancellable_task():
    """可取消的异步任务"""
    task = asyncio.create_task(
        chain.ainvoke({"topic": "long story"})
    )
    
    try:
        # 设置5秒超时
        result = await asyncio.wait_for(task, timeout=5.0)
        return result
    except asyncio.TimeoutError:
        task.cancel()  # 取消任务
        print("Task cancelled due to timeout")
        return None
```

### 4.5.3 超时控制

```python
from langchain_core.runnables import RunnableConfig

# 全局超时
config = RunnableConfig(
    timeout=10.0,  # 10秒超时
    max_concurrency=5
)

try:
    result = await chain.ainvoke({"text": "Hello"}, config=config)
except TimeoutError:
    print("Request timed out")
```

---

## 🎯 本章小结

**核心要点**:

1. **流式输出**: astream() 提供实时反馈,astream_events() 监听细粒度事件
2. **批处理**: batch() 和 abatch() 节省请求次数,提升吞吐量
3. **异步编程**: ainvoke() 支持并发,避免阻塞
4. **性能优化**: max_concurrency 控制并发度,背压控制处理速率
5. **进度管理**: 回调监听进度,超时和取消机制保证稳定性

**掌握检查**:

- [ ] 能实现流式聊天界面
- [ ] 能用 batch() 批量处理数据
- [ ] 能编写异步并发代码
- [ ] 能配置合理的并发度
- [ ] 能处理超时和取消

**练习题**:

1. **流式聊天**: 用 Streamlit 实现带打字效果的聊天界面
2. **批量翻译**: 批量翻译100段文本,对比串行和并行耗时
3. **并发优化**: 测试不同 max_concurrency 值的性能差异
4. **超时处理**: 实现带3秒超时和自动重试的翻译链

**性能基准**:

| 场景 | 串行 | 批处理 | 异步并发 |
|------|------|--------|----------|
| 100次调用 | 150s | 45s | 18s |
| 内存占用 | 低 | 中 | 低 |
| 用户体验 | 差（阻塞） | 中 | 优（非阻塞） |

---

## 📚 扩展阅读

- [流式输出指南](https://python.langchain.com/docs/how_to/streaming)
- [批处理文档](https://python.langchain.com/docs/how_to/batch)
- [异步最佳实践](https://python.langchain.com/docs/concepts/async)
- [LangServe 流式部署](https://python.langchain.com/docs/langserve)
- [Python asyncio 官方文档](https://docs.python.org/3/library/asyncio.html)
