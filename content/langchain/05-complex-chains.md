> **本章目标**：掌握顺序链、并行链、路由链、Map-Reduce 等高级编排模式，学会构建复杂多步骤 LLM 应用，并理解链嵌套与递归的设计原则。

---

## 本章导览

本章深入探讨 LangChain 的高级链编排技术，这些模式是构建企业级复杂应用的基石：

- **顺序链模式**：掌握多步骤串行处理流程，实现中间结果传递与转换
- **并行链模式**：通过 RunnableParallel 实现任务并发执行，提升整体性能
- **路由链模式**：基于条件、语义或向量相似度实现动态路由，构建智能分发系统
- **Map-Reduce 模式**：处理大规模文档集合，实现分布式计算范式
- **链嵌套与递归**：理解链作为组件的组合能力，掌握递归调用的控制技巧

通过本章学习，你将能够设计和实现生产级的复杂 LLM 应用架构。

---

## 5.1 顺序链（Sequential Chain）

### 5.1.1 多步骤处理流程

顺序链是最直观的编排模式，它将多个处理步骤串联起来，前一步的输出作为后一步的输入。在 LCEL 中，通过管道操作符 `|` 即可实现顺序链。

**基础示例**：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Step 1: 翻译
translate_prompt = ChatPromptTemplate.from_template(
    "Translate the following text to {language}:\n\n{text}"
)

# Step 2: 摘要
summary_prompt = ChatPromptTemplate.from_template(
    "Summarize the following text in 50 words:\n\n{translated_text}"
)

model = ChatOpenAI(model="gpt-4")

# 构建顺序链
sequential_chain = (
    translate_prompt
    | model
    | StrOutputParser()
    | (lambda translated_text: {"translated_text": translated_text})  # 转换为字典
    | summary_prompt
    | model
    | StrOutputParser()
)

# 执行
result = sequential_chain.invoke({
    "text": "Artificial Intelligence is transforming every industry...",
    "language": "Chinese"
})

print(result)
# 输出：人工智能正在改变各行各业...（摘要）
```

### 5.1.2 中间结果传递

在复杂流程中，我们经常需要保留中间步骤的结果，并在后续步骤中使用。使用 `RunnablePassthrough` 可以实现这一目标。

```python
from langchain_core.runnables import RunnablePassthrough

# 保留原始输入 + 添加新字段
chain = (
    RunnablePassthrough.assign(
        translation=translate_prompt | model | StrOutputParser()
    )
    | RunnablePassthrough.assign(
        summary=summary_prompt | model | StrOutputParser()
    )
)

result = chain.invoke({
    "text": "Machine learning is a subset of AI...",
    "language": "French"
})

print(result)
# {
#     "text": "Machine learning is a subset of AI...",
#     "language": "French",
#     "translation": "L'apprentissage automatique est un sous-ensemble de l'IA...",
#     "summary": "L'IA comprend l'apprentissage automatique..."
# }
```

### 5.1.3 TransformChain 自定义变换

对于非 LLM 的数据转换步骤，可以使用 `TransformChain` 或 Lambda 函数。

```python
from langchain.chains import TransformChain

def transform_inputs(inputs):
    """自定义转换函数"""
    text = inputs["text"]
    # 预处理：移除 HTML 标签、规范化空白符等
    cleaned_text = text.strip().replace("\n\n", "\n")
    return {"cleaned_text": cleaned_text}

transform_chain = TransformChain(
    input_variables=["text"],
    output_variables=["cleaned_text"],
    transform=transform_inputs
)

# 集成到顺序链
full_chain = (
    transform_chain
    | ChatPromptTemplate.from_template("Analyze: {cleaned_text}")
    | model
    | StrOutputParser()
)
```

**最佳实践**：
- 使用 `assign()` 保留上下文，避免丢失关键信息
- 为每个步骤添加清晰的命名（如 `translation`、`summary`）
- 在 Lambda 中进行简单转换，复杂逻辑使用 TransformChain

<div data-component="ChainOrchestrationDiagram"></div>

---

## 5.2 并行链（Parallel Chain）

### 5.2.1 RunnableParallel 详解

`RunnableParallel` 允许同时执行多个独立的链，并将结果合并为一个字典。这对于需要多视角分析或多任务处理的场景非常有用。

**语法形式**：

```python
from langchain_core.runnables import RunnableParallel

# 方式 1：字典语法（推荐）
parallel_chain = {
    "translation": translate_chain,
    "sentiment": sentiment_chain,
    "entities": entity_extraction_chain
}

# 方式 2：显式 RunnableParallel
parallel_chain = RunnableParallel(
    translation=translate_chain,
    sentiment=sentiment_chain,
    entities=entity_extraction_chain
)
```

**完整示例：多维度文本分析**：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4")

# 定义三个独立的分析链
translation_chain = (
    ChatPromptTemplate.from_template("Translate to Chinese: {text}")
    | model
    | StrOutputParser()
)

sentiment_chain = (
    ChatPromptTemplate.from_template(
        "Analyze sentiment (positive/negative/neutral): {text}"
    )
    | model
    | StrOutputParser()
)

key_points_chain = (
    ChatPromptTemplate.from_template("Extract 3 key points from: {text}")
    | model
    | StrOutputParser()
)

# 并行执行
parallel_analysis = RunnableParallel(
    translation=translation_chain,
    sentiment=sentiment_chain,
    key_points=key_points_chain
)

result = parallel_analysis.invoke({
    "text": "The new AI model shows impressive performance..."
})

print(result)
# {
#     "translation": "新的 AI 模型展现出令人印象深刻的性能...",
#     "sentiment": "positive",
#     "key_points": "1. High performance\n2. New model\n3. AI advancement"
# }
```

### 5.2.2 结果聚合策略

并行链执行后，通常需要将结果聚合为最终输出。

```python
# 聚合器：将并行结果合并为报告
aggregation_chain = (
    parallel_analysis
    | ChatPromptTemplate.from_template(
        \"\"\"
        Based on the following analysis:
        
        Translation: {translation}
        Sentiment: {sentiment}
        Key Points: {key_points}
        
        Generate a comprehensive report.
        \"\"\"
    )
    | model
    | StrOutputParser()
)

report = aggregation_chain.invoke({"text": "..."})
```

### 5.2.3 部分失败处理

在并行执行中，某个分支失败不应影响其他分支。使用 `try_except` 包装或配置 `return_exceptions=True`。

```python
from langchain_core.runnables import RunnableParallel

def safe_chain(chain, default="Error occurred"):
    \"\"\"包装链以处理异常\"\"\"
    def wrapped(inputs):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            return f"{default}: {str(e)}"
    return wrapped

safe_parallel = RunnableParallel(
    translation=safe_chain(translation_chain, "Translation failed"),
    sentiment=safe_chain(sentiment_chain, "Sentiment analysis failed"),
    key_points=safe_chain(key_points_chain, "Extraction failed")
)
```

**性能优化**：
- 并行链的执行时间取决于最慢的分支
- 使用异步 `abatch()` 进一步提升并发性能
- 监控各分支的延迟，优化慢链

---

## 5.3 路由链（Router Chain）

### 5.3.1 基于条件的动态路由

路由链根据输入特征将请求分发到不同的处理链。最简单的方式是使用 `RunnableBranch`。

```python
from langchain_core.runnables import RunnableBranch

# 根据语言类型路由到不同的翻译链
router_chain = RunnableBranch(
    (
        lambda x: x["language"] == "technical",
        technical_translation_chain  # 技术文档翻译
    ),
    (
        lambda x: x["language"] == "literary",
        literary_translation_chain   # 文学作品翻译
    ),
    general_translation_chain  # 默认链
)

result = router_chain.invoke({
    "text": "The algorithm complexity is O(n log n)",
    "language": "technical"
})
```

### 5.3.2 LLMRouterChain（语义路由）

使用 LLM 根据语义内容进行智能路由。

```python
from langchain.chains.router import LLMRouterChain, MultiPromptChain
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

# 定义目标链
physics_chain = ChatPromptTemplate.from_template(
    "You are a physics expert. Answer: {input}"
) | model | StrOutputParser()

math_chain = ChatPromptTemplate.from_template(
    "You are a math expert. Answer: {input}"
) | model | StrOutputParser()

history_chain = ChatPromptTemplate.from_template(
    "You are a history expert. Answer: {input}"
) | model | StrOutputParser()

# 路由配置
destination_chains = {
    "physics": physics_chain,
    "math": math_chain,
    "history": history_chain
}

destinations = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics"
    },
    {
        "name": "math",
        "description": "Good for solving math problems"
    },
    {
        "name": "history",
        "description": "Good for historical questions"
    }
]

# 构建路由链（Legacy API，新版本建议用 RunnableBranch）
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations="\\n".join([f"{d['name']}: {d['description']}" for d in destinations])
)

router_prompt = ChatPromptTemplate.from_template(router_template)
router_chain = router_prompt | model | RouterOutputParser()
```

### 5.3.3 EmbeddingRouterChain（向量路由）

基于向量相似度进行路由，适合大规模分类场景。

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 创建路由索引
embeddings = OpenAIEmbeddings()

route_embeddings = embeddings.embed_documents([
    "Physics and quantum mechanics",
    "Calculus and algebra",
    "World War II and ancient Rome"
])

# 构建向量索引
vectorstore = FAISS.from_embeddings(
    text_embeddings=list(zip(
        ["physics", "math", "history"],
        route_embeddings
    )),
    embedding=embeddings
)

# 路由函数
def route_by_embedding(query):
    results = vectorstore.similarity_search(query, k=1)
    return destination_chains[results[0].page_content]

# 使用
query = "What is the derivative of x^2?"
selected_chain = route_by_embedding(query)
result = selected_chain.invoke({"input": query})
```

### 5.3.4 自定义路由逻辑

对于复杂业务规则，自定义路由函数提供最大灵活性。

```python
def custom_router(inputs):
    \"\"\"基于多个条件的复杂路由\"\"\"
    text = inputs["text"]
    user_type = inputs.get("user_type", "free")
    
    # 规则 1：VIP 用户使用高级模型
    if user_type == "vip":
        return gpt4_chain
    
    # 规则 2：长文本使用摘要模型
    if len(text) > 5000:
        return long_document_chain
    
    # 规则 3：包含代码的使用代码专用模型
    if "```" in text or "def " in text:
        return code_chain
    
    # 默认
    return standard_chain

# 集成到 LCEL
routed_chain = custom_router | model | StrOutputParser()
```

<div data-component="RouterDecisionTree"></div>

---

## 5.4 Map-Reduce 模式

### 5.4.1 文档批量处理

Map-Reduce 是处理大规模文档集合的经典模式。Map 阶段对每个文档独立处理，Reduce 阶段合并结果。

**应用场景**：
- 长文档摘要（拆分成多个片段分别摘要，再合并）
- 批量问答（对多个文档分别提问，再聚合答案）
- 并行翻译（分段翻译，保持一致性）

### 5.4.2 Map 阶段：并行转换

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel

# 加载长文档
long_document = open("long_article.txt").read()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)
chunks = text_splitter.split_text(long_document)

# Map 阶段：为每个 chunk 生成摘要
map_prompt = ChatPromptTemplate.from_template(
    "Summarize the following text:\\n\\n{text}"
)

map_chain = map_prompt | model | StrOutputParser()

# 并行处理所有 chunks
map_results = map_chain.batch([{"text": chunk} for chunk in chunks])

print(f"Generated {len(map_results)} summaries")
```

### 5.4.3 Reduce 阶段：结果合并

```python
# Reduce 阶段：合并所有摘要
reduce_prompt = ChatPromptTemplate.from_template(
    \"\"\"
    Combine the following summaries into a single comprehensive summary:
    
    {summaries}
    
    Final summary:
    \"\"\"
)

reduce_chain = reduce_prompt | model | StrOutputParser()

# 合并
combined_summaries = "\\n\\n".join(map_results)
final_summary = reduce_chain.invoke({"summaries": combined_summaries})

print(final_summary)
```

### 5.4.4 应用场景：长文本摘要

**完整的 Map-Reduce 摘要流程**：

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_core.documents import Document

# 将 chunks 转换为 Document 对象
documents = [Document(page_content=chunk) for chunk in chunks]

# Map Chain
map_chain = (
    ChatPromptTemplate.from_template("Summarize: {page_content}")
    | model
    | StrOutputParser()
)

# Reduce Chain
reduce_chain = (
    ChatPromptTemplate.from_template(
        "Combine these summaries:\\n\\n{summaries}"
    )
    | model
    | StrOutputParser()
)

# 组合（使用 LCEL 方式）
def map_reduce_summarize(docs):
    # Map
    summaries = map_chain.batch([{"page_content": doc.page_content} for doc in docs])
    
    # Reduce
    combined = "\\n\\n".join(summaries)
    final = reduce_chain.invoke({"summaries": combined})
    
    return final

result = map_reduce_summarize(documents)
```

**性能优化**：
- 使用 `abatch()` 进行异步并行处理
- 合理设置 `chunk_size` 和 `chunk_overlap`
- 考虑使用流式输出减少等待时间

<div data-component="MapReduceVisualizer"></div>

---

## 5.5 链嵌套与递归

### 5.5.1 链作为链的组件

LCEL 的强大之处在于链本身也是 Runnable，可以作为更大链的组件。

```python
# 基础链：翻译
translation_chain = (
    ChatPromptTemplate.from_template("Translate to {lang}: {text}")
    | model
    | StrOutputParser()
)

# 嵌套链：先翻译，再摘要
nested_chain = (
    translation_chain
    | (lambda translated: {"text": translated})
    | ChatPromptTemplate.from_template("Summarize: {text}")
    | model
    | StrOutputParser()
)

# 更深层嵌套：翻译 -> 摘要 -> 关键词提取
deep_nested_chain = (
    nested_chain
    | (lambda summary: {"text": summary})
    | ChatPromptTemplate.from_template("Extract keywords: {text}")
    | model
    | StrOutputParser()
)
```

### 5.5.2 递归调用控制

某些任务需要递归处理，例如迭代优化输出。

```python
def iterative_refinement_chain(text, max_iterations=3):
    \"\"\"迭代改进文本质量\"\"\"
    refinement_chain = (
        ChatPromptTemplate.from_template(
            "Improve the following text:\\n\\n{text}\\n\\nImproved version:"
        )
        | model
        | StrOutputParser()
    )
    
    current_text = text
    for i in range(max_iterations):
        print(f"Iteration {i+1}...")
        current_text = refinement_chain.invoke({"text": current_text})
    
    return current_text

# 使用
original = "AI is good"
refined = iterative_refinement_chain(original, max_iterations=3)
print(refined)
# Iteration 1...
# Iteration 2...
# Iteration 3...
# "Artificial Intelligence represents a transformative technology..."
```

### 5.5.3 最大深度限制

递归必须有终止条件，避免无限循环。

```python
def recursive_solver(problem, depth=0, max_depth=5):
    \"\"\"递归问题求解器\"\"\"
    if depth >= max_depth:
        return "Max depth reached, stopping recursion"
    
    solve_chain = (
        ChatPromptTemplate.from_template(
            \"\"\"
            Solve this problem: {problem}
            
            If you need to break it into sub-problems, return them as JSON:
            {{"sub_problems": ["sub1", "sub2"]}}
            
            Otherwise, return the solution:
            {{"solution": "..."}}
            \"\"\"
        )
        | model
        | StrOutputParser()
    )
    
    result = solve_chain.invoke({"problem": problem})
    
    # 解析结果
    import json
    try:
        parsed = json.loads(result)
        if "sub_problems" in parsed:
            # 递归处理子问题
            sub_solutions = [
                recursive_solver(sub, depth+1, max_depth)
                for sub in parsed["sub_problems"]
            ]
            return f"Solved by breaking into: {sub_solutions}"
        else:
            return parsed["solution"]
    except:
        return result

# 使用
solution = recursive_solver("How to build a web app?")
```

**最佳实践**：
- 始终设置 `max_depth` 限制
- 添加日志记录递归深度
- 考虑使用尾递归优化（Python 不原生支持，需手动转循环）
- 监控递归调用的成本（Token 消耗）

---

## 5.6 实战案例：智能客服路由系统

综合运用顺序链、并行链、路由链构建一个生产级客服系统。

```python
from langchain_core.runnables import RunnableParallel, RunnableBranch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4")

# 1. 意图识别链
intent_chain = (
    ChatPromptTemplate.from_template(
        \"\"\"
        Classify the intent of this customer message:
        - billing: billing and payment issues
        - technical: technical support
        - general: general inquiries
        
        Message: {message}
        
        Return only the intent name.
        \"\"\"
    )
    | model
    | StrOutputParser()
)

# 2. 三个专门的处理链
billing_chain = (
    ChatPromptTemplate.from_template(
        "As a billing specialist, respond to: {message}"
    )
    | model
    | StrOutputParser()
)

technical_chain = (
    ChatPromptTemplate.from_template(
        "As a technical support engineer, respond to: {message}"
    )
    | model
    | StrOutputParser()
)

general_chain = (
    ChatPromptTemplate.from_template(
        "As a customer service agent, respond to: {message}"
    )
    | model
    | StrOutputParser()
)

# 3. 情感分析链（并行）
sentiment_chain = (
    ChatPromptTemplate.from_template(
        "Analyze sentiment (1-10): {message}"
    )
    | model
    | StrOutputParser()
)

# 4. 组合系统
customer_service_system = (
    # 步骤 1：并行执行意图识别和情感分析
    RunnableParallel(
        intent=intent_chain,
        sentiment=sentiment_chain,
        message=lambda x: x["message"]
    )
    # 步骤 2：根据意图路由
    | RunnableBranch(
        (lambda x: "billing" in x["intent"].lower(), billing_chain),
        (lambda x: "technical" in x["intent"].lower(), technical_chain),
        general_chain
    )
)

# 测试
response = customer_service_system.invoke({
    "message": "I was charged twice for my subscription!"
})

print(response)
# "I apologize for the inconvenience. Let me check your billing records..."
```

---

## 5.7 性能监控与调优

### 5.7.1 链执行时间分析

```python
import time
from functools import wraps

def timing_wrapper(chain):
    \"\"\"为链添加计时功能\"\"\"
    def timed_invoke(inputs):
        start = time.time()
        result = chain.invoke(inputs)
        elapsed = time.time() - start
        print(f"Chain executed in {elapsed:.2f}s")
        return result
    return timed_invoke

# 使用
timed_chain = timing_wrapper(sequential_chain)
result = timed_chain({"text": "..."})
# Chain executed in 2.34s
```

### 5.7.2 并行优化

```python
# 对比：顺序 vs 并行
import asyncio

# 顺序执行（慢）
sequential_result = {
    "translation": translation_chain.invoke(input),
    "sentiment": sentiment_chain.invoke(input),
    "summary": summary_chain.invoke(input)
}

# 并行执行（快）
parallel_result = RunnableParallel(
    translation=translation_chain,
    sentiment=sentiment_chain,
    summary=summary_chain
).invoke(input)

# 异步并行（最快）
async def async_parallel():
    return await RunnableParallel(
        translation=translation_chain,
        sentiment=sentiment_chain,
        summary=summary_chain
    ).ainvoke(input)

result = asyncio.run(async_parallel())
```

---

## 本章小结

本章深入学习了 LangChain 的高级链编排模式：

✅ **顺序链**：掌握了多步骤串行处理和中间结果传递  
✅ **并行链**：学会使用 RunnableParallel 提升性能  
✅ **路由链**：实现了基于条件、语义和向量的智能路由  
✅ **Map-Reduce**：理解了分布式处理大规模文档的模式  
✅ **嵌套与递归**：掌握了复杂链组合和递归控制

这些模式是构建企业级 LLM 应用的核心技术，灵活组合它们可以解决绝大多数复杂业务场景。

---

## 扩展阅读

- [LangChain Expression Language 官方文档](https://python.langchain.com/docs/expression_language/)
- [Runnable Interface](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.Runnable.html)
- [Map-Reduce Chains](https://python.langchain.com/docs/modules/chains/document/)
- [Router Chains](https://python.langchain.com/docs/modules/chains/foundational/router)
