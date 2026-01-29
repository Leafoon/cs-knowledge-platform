> **本章目标**：精通 Few-Shot 提示、Chat Prompt Templates、提示组合与复用、LangChain Hub 以及动态提示生成等高级技术，构建高质量的提示词系统。

---

## 本章导览

本章深入提示工程的核心技术，这是决定 LLM 应用质量的关键因素：

- **Few-Shot Learning**：掌握示例选择器的设计，实现基于相似度、长度、多样性的动态示例注入
- **Chat Prompt 管理**：系统化管理多轮对话中的角色、历史和上下文
- **提示复用**：通过模块化设计实现提示的继承、组合与版本管理
- **LangChain Hub**：利用社区提示资源，快速原型开发与协作
- **动态生成**：使用 LLM 生成提示词，实现 Meta-Prompting 和自适应提示

掌握这些技术将极大提升你的 LLM 应用性能和可维护性。

---

## 6.1 Few-Shot Prompting

### 6.1.1 FewShotPromptTemplate 基础

Few-Shot Learning 通过在提示中包含少量示例，帮助 LLM 理解任务模式并生成更准确的输出。

**基础用法**：

```python
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts import PromptTemplate

# 定义示例
examples = [
    {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris."
    },
    {
        "question": "What is 2 + 2?",
        "answer": "2 + 2 equals 4."
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "answer": "Romeo and Juliet was written by William Shakespeare."
    }
]

# 定义示例模板
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\\nAnswer: {answer}"
)

# 创建 Few-Shot 提示模板
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Answer the question following these examples:",
    suffix="Question: {input}\\nAnswer:",
    input_variables=["input"]
)

# 使用
print(few_shot_prompt.format(input="What is the largest planet?"))
```

**输出**：
```
Answer the question following these examples:

Question: What is the capital of France?
Answer: The capital of France is Paris.

Question: What is 2 + 2?
Answer: 2 + 2 equals 4.

Question: Who wrote Romeo and Juliet?
Answer: Romeo and Juliet was written by William Shakespeare.

Question: What is the largest planet?
Answer:
```

### 6.1.2 ExampleSelector：动态示例选择

手动指定示例在大规模应用中不现实。`ExampleSelector` 允许根据输入动态选择最相关的示例。

**接口定义**：

```python
from langchain.prompts.example_selector.base import BaseExampleSelector

class CustomExampleSelector(BaseExampleSelector):
    def add_example(self, example):
        """添加新示例到存储"""
        pass
    
    def select_examples(self, input_variables):
        \"\"\"根据输入选择示例\"\"\"
        pass
```

### 6.1.3 SemanticSimilarityExampleSelector：相似度选择

基于向量相似度选择与输入最相关的示例。

```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 准备大量示例
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "hot", "output": "cold"},
    {"input": "fast", "output": "slow"},
    {"input": "light", "output": "dark"},
]

# 创建向量存储
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=2  # 选择最相似的 2 个示例
)

# 使用选择器
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\\nOutput: {output}"
    ),
    prefix="Give the antonym of the word:",
    suffix="Input: {adjective}\\nOutput:",
    input_variables=["adjective"]
)

# 测试
print(few_shot_prompt.format(adjective="bright"))
# 会自动选择 "light"→"dark" 和 "happy"→"sad" 等相似示例
```

### 6.1.4 MaxMarginalRelevanceExampleSelector：多样性平衡

`MMR` 在相似度和多样性之间找平衡，避免选择过于相似的示例。

```python
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector

example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=3,
    fetch_k=10  # 先召回 10 个，再选 3 个多样化的
)

# lambda 参数控制相似度 vs 多样性权重
# lambda=1.0: 纯相似度
# lambda=0.0: 纯多样性
example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=3,
    lambda_mult=0.5  # 平衡设置
)
```

### 6.1.5 LengthBasedExampleSelector：长度控制

根据输入长度动态调整示例数量，避免超出 token 限制。

```python
from langchain.prompts.example_selector import LengthBasedExampleSelector

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=100,  # 最大 token 数
    get_text_length=lambda x: len(x.split())  # 自定义长度计算
)

# 短输入会包含更多示例，长输入会减少示例
```

**最佳实践**：
- 使用 `SemanticSimilarityExampleSelector` 提升相关性
- 使用 `MaxMarginalRelevanceExampleSelector` 增加多样性
- 使用 `LengthBasedExampleSelector` 控制成本
- 组合多个选择器实现复杂逻辑

<div data-component="FewShotExampleSelector"></div>

---

## 6.2 Chat Prompt Templates

### 6.2.1 消息角色（system、user、assistant）

Chat 模型使用结构化消息，每条消息都有角色和内容。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 方式 1：显式创建消息
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris."),
    HumanMessage(content="What about Germany?")
]

# 方式 2：使用 ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in {topic}."),
    ("human", "What is {question}?"),
])

formatted = chat_prompt.format_messages(
    topic="geography",
    question="the capital of France"
)
```

### 6.2.2 MessagesPlaceholder：动态消息注入

`MessagesPlaceholder` 允许在模板中插入可变数量的历史消息。

```python
from langchain_core.prompts import MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chatbot having a conversation with a human."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 使用
from langchain_core.messages import HumanMessage, AIMessage

chat_history = [
    HumanMessage(content="Hi, I'm Alice"),
    AIMessage(content="Hello Alice! How can I help you?"),
    HumanMessage(content="What's my name?")
]

formatted = chat_prompt.format_messages(
    chat_history=chat_history,
    input="Remind me what we talked about"
)
```

### 6.2.3 对话历史管理

结合记忆系统管理长对话。

```python
from langchain.memory import ConversationBufferMemory

# 创建记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 提示模板
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 构建链
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4")

chain = chat_prompt | model | StrOutputParser()

# 使用（手动管理记忆）
def chat(user_input):
    # 获取历史
    history = memory.load_memory_variables({})["chat_history"]
    
    # 执行
    response = chain.invoke({
        "chat_history": history,
        "input": user_input
    })
    
    # 保存到记忆
    memory.save_context(
        {"input": user_input},
        {"output": response}
    )
    
    return response

# 对话
print(chat("Hi, I'm Bob"))
print(chat("What's my name?"))
```

### 6.2.4 角色扮演提示

使用 `system` 消息定义角色人设。

```python
personas = {
    "expert": "You are an expert in {field} with 20 years of experience. Provide detailed, technical answers.",
    "beginner": "You are helping someone learn {field} for the first time. Use simple language and examples.",
    "creative": "You are a creative writer. Answer questions with imaginative stories and metaphors."
}

def create_persona_prompt(persona_type, field):
    return ChatPromptTemplate.from_messages([
        ("system", personas[persona_type]),
        ("human", "{question}")
    ])

# 使用
expert_prompt = create_persona_prompt("expert", "quantum physics")
response = (expert_prompt | model | StrOutputParser()).invoke({
    "field": "quantum physics",
    "question": "Explain quantum entanglement"
})
```

---

## 6.3 Prompt 组合与复用

### 6.3.1 PipelinePromptTemplate：模块化提示

将复杂提示拆分为可复用的模块。

```python
from langchain.prompts.pipeline import PipelinePromptTemplate

# 模块 1：角色定义
role_template = PromptTemplate.from_template(
    \"\"\"You are a {role} with expertise in {domain}.\"\"\"
)

# 模块 2：任务说明
task_template = PromptTemplate.from_template(
    \"\"\"Task: {task}
    Requirements:
    - {requirement1}
    - {requirement2}
    \"\"\"
)

# 模块 3：输出格式
format_template = PromptTemplate.from_template(
    \"\"\"Output format:
    ```{output_format}```
    \"\"\"
)

# 组合
final_template = PromptTemplate.from_template(
    \"\"\"{role}

{task}

{format}

Input: {input}
Output:\"\"\"
)

pipeline_prompt = PipelinePromptTemplate(
    final_prompt=final_template,
    pipeline_prompts=[
        ("role", role_template),
        ("task", task_template),
        ("format", format_template)
    ]
)

# 使用
result = pipeline_prompt.format(
    role="Data Scientist",
    domain="machine learning",
    task="Analyze this dataset",
    requirement1="Clean the data",
    requirement2="Identify outliers",
    output_format="JSON",
    input="sales_data.csv"
)
```

### 6.3.2 提示继承与覆盖

创建基础提示模板，不同场景继承并覆盖部分内容。

```python
# 基础提示
base_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant."),
    ("human", "{input}")
])

# 继承并添加
specialized_prompt = base_prompt + ChatPromptTemplate.from_messages([
    ("system", "You specialize in {specialty}.")
])

# 使用
formatted = specialized_prompt.format_messages(
    input="Solve this problem",
    specialty="mathematics"
)
```

### 6.3.3 多语言提示模板

```python
prompts = {
    "en": ChatPromptTemplate.from_template("Translate to French: {text}"),
    "zh": ChatPromptTemplate.from_template("翻译成法语：{text}"),
    "es": ChatPromptTemplate.from_template("Traducir al francés: {text}")
}

def get_prompt(language):
    return prompts.get(language, prompts["en"])

# 使用
prompt = get_prompt("zh")
```

<div data-component="PromptComposer"></div>

---

## 6.4 LangChain Hub

### 6.4.1 Hub 提示浏览与搜索

LangChain Hub 是社区共享提示词的仓库，类似 Hugging Face Hub。

**浏览 Hub**：
- 网站：https://smith.langchain.com/hub
- 按类别浏览：QA、Summarization、Code、Creative Writing
- 查看评分和使用统计

### 6.4.2 hub.pull()：加载提示

```python
from langchain import hub

# 拉取热门提示
prompt = hub.pull("rlm/rag-prompt")

# 查看内容
print(prompt)

# 直接使用
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4")

chain = prompt | model | StrOutputParser()

result = chain.invoke({
    "context": "LangChain is a framework...",
    "question": "What is LangChain?"
})
```

**常用提示**：
- `rlm/rag-prompt`：RAG 问答提示
- `hwchase17/openai-functions-agent`：Function Calling Agent
- `langchain-ai/sql-query-system`：SQL 生成提示

### 6.4.3 hub.push()：上传提示

```python
# 创建自己的提示
my_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}."),
    ("human", "{question}")
])

# 上传到 Hub
hub.push("your-username/expert-qa", my_prompt)

# 添加描述和标签
hub.push(
    "your-username/expert-qa",
    my_prompt,
    description="Expert Q&A prompt with domain specialization",
    tags=["qa", "expert", "customizable"]
)
```

### 6.4.4 版本管理与协作

```python
# 拉取特定版本
prompt = hub.pull("rlm/rag-prompt", version="v2")

# 查看版本历史
versions = hub.list_versions("rlm/rag-prompt")

# Fork 并修改
prompt = hub.pull("rlm/rag-prompt")
modified_prompt = prompt + ChatPromptTemplate.from_template("...")
hub.push("your-username/rag-prompt-improved", modified_prompt)
```

**最佳实践**：
- 先搜索 Hub，避免重复造轮子
- Fork 现有提示并改进
- 为自己的提示添加详细文档
- 使用语义化版本号（v1.0.0, v1.1.0）

<div data-component="HubBrowser"></div>

---

## 6.5 动态提示生成

### 6.5.1 基于上下文的提示调整

根据用户输入动态调整提示。

```python
def adaptive_prompt(user_level, topic):
    \"\"\"根据用户水平调整提示\"\"\"
    difficulty_levels = {
        "beginner": "Explain {topic} using simple language and everyday examples.",
        "intermediate": "Explain {topic} with technical details and examples.",
        "expert": "Provide an in-depth analysis of {topic} with advanced concepts."
    }
    
    template = difficulty_levels.get(user_level, difficulty_levels["intermediate"])
    return ChatPromptTemplate.from_template(template)

# 使用
prompt = adaptive_prompt("beginner", "quantum computing")
```

### 6.5.2 LLM 生成提示（Meta-Prompting）

使用 LLM 生成优化的提示词。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI(model="gpt-4")

# Meta-prompt：生成提示的提示
meta_prompt = ChatPromptTemplate.from_template(
    \"\"\"Generate an optimal prompt for the following task:

Task: {task}
Target audience: {audience}
Desired output format: {format}

Create a detailed, effective prompt that will get high-quality results from an LLM.\"\"\"
)

# 生成提示
generated_prompt_text = (meta_prompt | model | StrOutputParser()).invoke({
    "task": "Summarize scientific papers",
    "audience": "Researchers",
    "format": "Structured abstract with sections"
})

print(generated_prompt_text)

# 使用生成的提示
generated_prompt = ChatPromptTemplate.from_template(generated_prompt_text)
```

### 6.5.3 A/B 测试提示变体

```python
import random

prompts_variants = [
    ChatPromptTemplate.from_template("Summarize: {text}"),
    ChatPromptTemplate.from_template("Provide a brief summary of: {text}"),
    ChatPromptTemplate.from_template("TL;DR: {text}")
]

def ab_test_prompt(text):
    \"\"\"随机选择提示变体\"\"\"
    prompt = random.choice(prompts_variants)
    return (prompt | model | StrOutputParser()).invoke({"text": text})

# 收集数据
results = {}
for i in range(100):
    variant = random.randint(0, 2)
    response = (prompts_variants[variant] | model | StrOutputParser()).invoke(...)
    
    # 记录用户反馈
    results.setdefault(variant, []).append(user_rating)

# 分析最优变体
best_variant = max(results, key=lambda k: sum(results[k]) / len(results[k]))
```

---

## 6.6 实战案例：多语言客服提示系统

综合运用本章技术构建生产级提示系统。

```python
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS

# 1. 加载基础提示
base_prompt = hub.pull("customer-service/multilingual")

# 2. 准备 Few-Shot 示例
examples = [
    {
        "customer": "My order hasn't arrived yet",
        "agent": "I apologize for the delay. Let me check your order status..."
    },
    {
        "customer": "产品质量有问题",
        "agent": "非常抱歉给您带来不便。请告诉我具体情况..."
    },
    # ... 更多示例
]

# 3. 创建示例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=3
)

# 4. 构建完整提示
final_prompt = ChatPromptTemplate.from_messages([
    ("system", base_prompt.template),
    ("system", "Here are some examples of good responses:"),
    MessagesPlaceholder(variable_name="examples"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 5. 创建链
model = ChatOpenAI(model="gpt-4")

def customer_service_chain(user_input, chat_history):
    # 选择示例
    selected_examples = example_selector.select_examples({"customer": user_input})
    
    # 格式化示例为消息
    example_messages = []
    for ex in selected_examples:
        example_messages.append(("human", ex["customer"]))
        example_messages.append(("ai", ex["agent"]))
    
    # 执行
    response = (final_prompt | model | StrOutputParser()).invoke({
        "examples": example_messages,
        "chat_history": chat_history,
        "input": user_input
    })
    
    return response
```

---

## 本章小结

本章深入学习了高级提示工程技术：

✅ **Few-Shot Learning**：掌握了多种示例选择策略（相似度、多样性、长度）  
✅ **Chat Prompts**：系统化管理多轮对话的角色和历史  
✅ **提示复用**：通过模块化设计实现组合与继承  
✅ **LangChain Hub**：利用社区资源加速开发  
✅ **动态生成**：使用 Meta-Prompting 和 A/B 测试优化提示

这些技术是构建高质量 LLM 应用的关键，直接影响输出质量、成本和用户体验。

---

## 扩展阅读

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain Hub](https://smith.langchain.com/hub)
- [Few-Shot Prompting](https://python.langchain.com/docs/modules/model_io/prompts/few_shot_examples)
- [Example Selectors](https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/)
