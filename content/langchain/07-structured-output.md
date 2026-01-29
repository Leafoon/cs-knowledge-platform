> **本章目标**：掌握 Output Parsers、Structured Output、Pydantic 模型定义、自定义解析器开发以及输出验证与后处理技术，实现类型安全的 LLM 输出处理。

---

## 本章导览

本章聚焦于 LLM 输出的结构化处理，这是构建可靠应用的关键环节：

- **Output Parsers 体系**：深入理解 PydanticOutputParser、JsonOutputParser 等内置解析器的设计与使用
- **原生结构化输出**：使用 `with_structured_output()` 和 Function Calling 实现更可靠的结构化输出
- **复杂类型处理**：解析嵌套对象、列表、枚举等复杂数据结构
- **自定义解析器**：基于 BaseOutputParser 开发满足特定需求的解析器
- **验证与容错**：使用 Pydantic Validator 和 OutputFixingParser 提升鲁棒性

掌握这些技术将让你的应用输出更可预测、更易处理、更适合生产环境。

---

## 7.1 Output Parsers 深度解析

### 7.1.1 Pydantic OutputParser 完整指南

`PydanticOutputParser` 是最常用的输出解析器，它将 LLM 的文本输出解析为 Pydantic 模型。

**基础用法**：

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 1. 定义数据模型
class Person(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    email: str = Field(description="The person's email address")
    occupation: str = Field(description="The person's job title")

# 2. 创建解析器
parser = PydanticOutputParser(pydantic_object=Person)

# 3. 构建提示（包含格式说明）
prompt = PromptTemplate(
    template="Extract information about the person.\\n{format_instructions}\\n{query}\\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. 创建链
model = ChatOpenAI(model="gpt-4")
chain = prompt | model | parser

# 5. 使用
result = chain.invoke({
    "query": "John Doe is a 30-year-old software engineer. His email is john@example.com"
})

print(result)
# Person(name='John Doe', age=30, email='john@example.com', occupation='Software Engineer')

print(type(result))
# <class '__main__.Person'>
```

### 7.1.2 自动生成格式说明（get_format_instructions）

`get_format_instructions()` 自动生成提示模型输出格式的说明。

```python
format_instructions = parser.get_format_instructions()
print(format_instructions)
```

**输出**：
```
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```json
{
  "properties": {
    "name": {"type": "string", "description": "The person's full name"},
    "age": {"type": "integer", "description": "The person's age in years"},
    "email": {"type": "string", "description": "The person's email address"},
    "occupation": {"type": "string", "description": "The person's job title"}
  },
  "required": ["name", "age", "email", "occupation"]
}
```
```

### 7.1.3 解析失败处理（OutputFixingParser）

当 LLM 输出格式不正确时，`OutputFixingParser` 可以自动修复。

```python
from langchain.output_parsers import OutputFixingParser

# 原始解析器
base_parser = PydanticOutputParser(pydantic_object=Person)

# 包装为自修复解析器
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-4")
)

# 故意提供错误格式的输出
bad_output = '''
{
    "name": "Jane Doe",
    "age": "twenty-five",  // 错误：应该是整数
    "email": "jane@example.com"
    // 缺少 occupation 字段
}
'''

try:
    # 直接解析会失败
    result = base_parser.parse(bad_output)
except Exception as e:
    print(f"Base parser failed: {e}")
    
    # 使用修复解析器
    result = fixing_parser.parse(bad_output)
    print(result)
    # Person(name='Jane Doe', age=25, email='jane@example.com', occupation='Unknown')
```

### 7.1.4 重试解析器（RetryOutputParser）

`RetryOutputParser` 在解析失败时重新调用 LLM。

```python
from langchain.output_parsers import RetryWithErrorOutputParser

base_parser = PydanticOutputParser(pydantic_object=Person)

retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-4")
)

# 使用
completion = model.invoke(prompt.format(query="..."))

try:
    result = base_parser.parse(completion.content)
except Exception as e:
    # 重试时会将错误信息发送给 LLM
    result = retry_parser.parse_with_prompt(
        completion.content,
        prompt.format(query="...")
    )
```

<div data-component="OutputParserFlow"></div>

---

## 7.2 Structured Output

### 7.2.1 with_structured_output()：原生结构化

现代 LLM（如 GPT-4、Claude 3）原生支持结构化输出，比传统解析器更可靠。

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    email: str
    occupation: str

model = ChatOpenAI(model="gpt-4")

# 使用 with_structured_output
structured_model = model.with_structured_output(Person)

# 直接调用，无需解析器
result = structured_model.invoke(
    "John Doe is a 30-year-old software engineer. His email is john@example.com"
)

print(result)
# Person(name='John Doe', age=30, email='john@example.com', occupation='Software Engineer')
```

### 7.2.2 JSON Mode（OpenAI）

OpenAI 的 JSON Mode 强制模型输出有效 JSON。

```python
model = ChatOpenAI(
    model="gpt-4-turbo-preview",
    model_kwargs={"response_format": {"type": "json_object"}}
)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that outputs in JSON."),
    ("human", "{input}")
])

chain = prompt | model

result = chain.invoke({
    "input": "List 3 colors in JSON format with 'colors' key"
})

print(result.content)
# {"colors": ["red", "blue", "green"]}

import json
parsed = json.loads(result.content)
print(parsed["colors"])
```

### 7.2.3 Function Calling 集成

使用 OpenAI Function Calling 实现结构化输出。

```python
from langchain_core.utils.function_calling import convert_to_openai_function

# 定义模型
class Person(BaseModel):
    \"\"\"Information about a person.\"\"\"
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age")
    occupation: str = Field(description="The person's job")

# 转换为 OpenAI Function
function = convert_to_openai_function(Person)

# 绑定到模型
model = ChatOpenAI(model="gpt-4").bind_functions([function])

result = model.invoke("Tell me about John, a 30-year-old engineer")

print(result.additional_kwargs["function_call"])
# {
#   "name": "Person",
#   "arguments": '{"name": "John", "age": 30, "occupation": "engineer"}'
# }
```

### 7.2.4 Pydantic 模型定义最佳实践

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    \"\"\"A task to be completed.\"\"\"
    
    title: str = Field(description="Brief title of the task")
    description: Optional[str] = Field(default=None, description="Detailed description")
    priority: Priority = Field(default=Priority.MEDIUM, description="Task priority")
    tags: List[str] = Field(default_factory=list, description="Category tags")
    estimated_hours: float = Field(gt=0, le=40, description="Estimated time in hours")
    
    @field_validator('title')
    @classmethod
    def title_must_be_capitalized(cls, v: str) -> str:
        if not v[0].isupper():
            raise ValueError('Title must start with capital letter')
        return v
    
    @field_validator('tags')
    @classmethod
    def tags_must_be_lowercase(cls, v: List[str]) -> List[str]:
        return [tag.lower() for tag in v]

# 使用
structured_model = ChatOpenAI(model="gpt-4").with_structured_output(Task)

result = structured_model.invoke(
    "Create a task: Fix login bug, high priority, should take 3 hours"
)

print(result)
# Task(
#     title='Fix login bug',
#     description=None,
#     priority=<Priority.HIGH: 'high'>,
#     tags=[],
#     estimated_hours=3.0
# )
```

---

## 7.3 复杂数据类型解析

### 7.3.1 嵌套对象（Nested Objects）

```python
from typing import List

class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str

class Company(BaseModel):
    name: str
    founded_year: int
    address: Address

class Employee(BaseModel):
    name: str
    position: str
    company: Company
    skills: List[str]

# 使用
parser = PydanticOutputParser(pydantic_object=Employee)

prompt = PromptTemplate(
    template=\"\"\"Extract employee information:
{format_instructions}

Text: {text}
\"\"\",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model | parser

result = chain.invoke({
    "text": \"\"\"
    Alice works as Senior Developer at TechCorp, which was founded in 2010 
    and located at 123 Tech Street, San Francisco, USA, 94105.
    She knows Python, JavaScript, and Docker.
    \"\"\"
})

print(result.company.address.city)  # San Francisco
print(result.skills)  # ['Python', 'JavaScript', 'Docker']
```

### 7.3.2 列表与数组

```python
class Article(BaseModel):
    title: str
    authors: List[str]
    keywords: List[str]
    publish_year: int

class Bibliography(BaseModel):
    articles: List[Article]

parser = PydanticOutputParser(pydantic_object=Bibliography)

# 提取多篇文章
result = chain.invoke({
    "text": \"\"\"
    1. "AI in Medicine" by Smith, Johnson (2023), keywords: AI, healthcare
    2. "Deep Learning Basics" by Lee (2022), keywords: neural networks, ML
    \"\"\"
})

print(len(result.articles))  # 2
print(result.articles[0].authors)  # ['Smith', 'Johnson']
```

### 7.3.3 枚举类型（Enum）

```python
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class Review(BaseModel):
    text: str
    sentiment: Sentiment
    rating: int = Field(ge=1, le=5)

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    "The product is amazing! I love it. 5 stars!"
)

print(result.sentiment)  # Sentiment.POSITIVE
print(result.rating)  # 5
```

### 7.3.4 可选字段与默认值

```python
from typing import Optional

class BlogPost(BaseModel):
    title: str
    content: str
    author: str = "Anonymous"  # 默认值
    published: bool = False
    views: Optional[int] = None  # 可选字段
    tags: List[str] = Field(default_factory=list)  # 空列表默认
    
result = structured_model.invoke("Write a post titled 'Hello World'")

print(result.author)  # Anonymous
print(result.published)  # False
print(result.views)  # None
```

<div data-component="StructuredOutputBuilder"></div>

---

## 7.4 自定义 Output Parser

### 7.4.1 继承 BaseOutputParser

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import List

class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    \"\"\"Parse comma-separated list.\"\"\"
    
    def parse(self, text: str) -> List[str]:
        \"\"\"Parse the output of an LLM call.\"\"\"
        return [item.strip() for item in text.strip().split(',')]
    
    def get_format_instructions(self) -> str:
        return "Your response should be a comma-separated list."

# 使用
parser = CommaSeparatedListOutputParser()

prompt = PromptTemplate(
    template="List 5 {subject}.\\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model | parser

result = chain.invoke({"subject": "programming languages"})
print(result)
# ['Python', 'JavaScript', 'Java', 'C++', 'Go']
```

### 7.4.2 parse() 方法实现

```python
class KeyValueParser(BaseOutputParser[dict]):
    \"\"\"Parse key:value format.\"\"\"
    
    def parse(self, text: str) -> dict:
        lines = text.strip().split('\\n')
        result = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
        
        return result
    
    def get_format_instructions(self) -> str:
        return \"\"\"Format your response as:
key1: value1
key2: value2
...\"\"\"

# 使用
parser = KeyValueParser()
output = model.invoke("List 3 countries and capitals in key:value format")
result = parser.parse(output.content)
print(result)
# {'France': 'Paris', 'Japan': 'Tokyo', 'Brazil': 'Brasilia'}
```

### 7.4.3 正则表达式解析

```python
import re

class EmailExtractor(BaseOutputParser[List[str]]):
    \"\"\"Extract all email addresses.\"\"\"
    
    def parse(self, text: str) -> List[str]:
        email_pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
        return re.findall(email_pattern, text)
    
    def get_format_instructions(self) -> str:
        return "Include email addresses in your response."

parser = EmailExtractor()
text = "Contact us at support@example.com or sales@company.org"
emails = parser.parse(text)
print(emails)
# ['support@example.com', 'sales@company.org']
```

### 7.4.4 多格式兼容解析器

```python
import json
import yaml

class FlexibleFormatParser(BaseOutputParser[dict]):
    \"\"\"Parse JSON or YAML format.\"\"\"
    
    def parse(self, text: str) -> dict:
        # 尝试 JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 尝试 YAML
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError:
            pass
        
        # 尝试 key:value 格式
        result = {}
        for line in text.strip().split('\\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
        
        return result
```

<div data-component="ParsingErrorDemo"></div>

---

## 7.5 输出验证与后处理

### 7.5.1 Pydantic Validator

```python
from pydantic import field_validator, model_validator

class Product(BaseModel):
    name: str
    price: float
    quantity: int
    discount_percent: float = 0.0
    
    @field_validator('price')
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('Price must be positive')
        return v
    
    @field_validator('discount_percent')
    @classmethod
    def discount_in_range(cls, v: float) -> float:
        if not 0 <= v <= 100:
            raise ValueError('Discount must be between 0 and 100')
        return v
    
    @model_validator(mode='after')
    def check_discounted_price(self):
        final_price = self.price * (1 - self.discount_percent / 100)
        if final_price < 0:
            raise ValueError('Final price cannot be negative')
        return self
    
    @property
    def final_price(self) -> float:
        return self.price * (1 - self.discount_percent / 100)

# 使用
try:
    product = Product(name="Laptop", price=-100, quantity=5)
except ValueError as e:
    print(e)  # Price must be positive
```

### 7.5.2 数据清洗与标准化

```python
class UserProfile(BaseModel):
    username: str
    email: str
    phone: Optional[str] = None
    
    @field_validator('username')
    @classmethod
    def normalize_username(cls, v: str) -> str:
        return v.lower().strip()
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.lower().strip()
        if '@' not in v:
            raise ValueError('Invalid email')
        return v
    
    @field_validator('phone')
    @classmethod
    def normalize_phone(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        # 移除非数字字符
        return ''.join(filter(str.isdigit, v))
```

### 7.5.3 业务规则校验

```python
from datetime import date

class Booking(BaseModel):
    guest_name: str
    checkin: date
    checkout: date
    rooms: int
    
    @model_validator(mode='after')
    def validate_dates(self):
        if self.checkout <= self.checkin:
            raise ValueError('Checkout must be after checkin')
        
        nights = (self.checkout - self.checkin).days
        if nights > 30:
            raise ValueError('Maximum stay is 30 nights')
        
        return self
    
    @field_validator('rooms')
    @classmethod
    def validate_rooms(cls, v: int) -> int:
        if v < 1 or v > 10:
            raise ValueError('Rooms must be between 1 and 10')
        return v
```

---

## 7.6 实战案例：简历解析系统

综合运用本章技术构建生产级简历解析系统。

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import date

class Education(BaseModel):
    degree: str
    institution: str
    graduation_year: int
    gpa: Optional[float] = None

class Experience(BaseModel):
    title: str
    company: str
    start_date: str
    end_date: Optional[str] = None
    description: str

class Resume(BaseModel):
    \"\"\"Structured resume data.\"\"\"
    
    name: str = Field(description="Full name")
    email: str = Field(description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    
    summary: str = Field(description="Professional summary")
    skills: List[str] = Field(description="Technical skills")
    
    education: List[Education] = Field(description="Educational background")
    experience: List[Experience] = Field(description="Work experience")
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v.lower()
    
    @field_validator('skills')
    @classmethod
    def deduplicate_skills(cls, v: List[str]) -> List[str]:
        return list(set(v))

# 创建解析链
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")
structured_model = model.with_structured_output(Resume)

# 解析简历
resume_text = \"\"\"
John Smith
Email: john.smith@email.com
Phone: +1-555-0123

SUMMARY
Experienced software engineer with 5 years in full-stack development.

SKILLS
Python, JavaScript, React, Node.js, Docker, AWS

EDUCATION
- Bachelor of Computer Science, MIT, 2018, GPA: 3.8
- Master of Software Engineering, Stanford, 2020

EXPERIENCE
Senior Developer at TechCorp
Jan 2021 - Present
Leading development of cloud-native applications using microservices architecture.

Software Engineer at StartupXYZ
Jun 2018 - Dec 2020
Developed web applications using React and Node.js.
\"\"\"

result = structured_model.invoke(f"Parse this resume:\\n\\n{resume_text}")

print(result.name)  # John Smith
print(result.skills)  # ['Python', 'JavaScript', 'React', ...]
print(len(result.experience))  # 2
print(result.education[0].institution)  # MIT
```

---

## 本章小结

本章深入学习了结构化输出与解析技术：

✅ **Output Parsers**：掌握了 PydanticOutputParser、OutputFixingParser、RetryOutputParser  
✅ **原生结构化**：使用 `with_structured_output()` 和 Function Calling  
✅ **复杂类型**：处理嵌套对象、列表、枚举等数据结构  
✅ **自定义解析器**：基于 BaseOutputParser 实现特定需求  
✅ **验证与容错**：使用 Pydantic Validator 提升可靠性

这些技术是构建生产级 LLM 应用的基础，确保输出可预测、易处理、类型安全。

---

## 扩展阅读

- [Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Structured Output](https://python.langchain.com/docs/modules/model_io/chat/structured_output)
- [Function Calling](https://platform.openai.com/docs/guides/function-calling)
