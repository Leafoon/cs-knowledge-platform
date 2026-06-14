---
title: "Chapter 32: Agent 系统深度剖析——GEAK 与 Triton-Copilot"
description: "深入走读 GEAK-Agent 的四模块闭环实现细节，理解 Triton-Copilot 的系统架构与设计选择，掌握 Prompt Engineering 与 Few-shot 策略在算子生成中的应用，了解 KernelBench 等评测基准的设计与使用。"
date: "2026-06-14"
---

# Chapter 32: Agent 系统深度剖析——GEAK 与 Triton-Copilot

> **学习目标**：
> - 深入走读 GEAK-Agent 的四模块闭环实现细节
> - 理解 Triton-Copilot 的系统架构与设计选择
> - 掌握 Prompt Engineering 与 Few-shot 策略在算子生成中的应用
> - 了解 KernelBench 等评测基准的设计与使用

---

## 32.1 GEAK-Agent 架构总览

### 32.1.1 系统架构图

GEAK（Generative Efficient AI Kernel）Agent 是一个基于 LLM 的自动化 kernel 生成系统，采用四模块闭环架构。整个系统的核心思想是：**生成 → 反思 → 评估 → 优化**，通过迭代不断提升 kernel 质量。

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           GEAK-Agent 四模块闭环架构                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Generator   │───▶│   Reflector  │───▶│  Evaluator   │───▶│  Optimizer   │  │
│  │   生成器      │◀───│   反思器     │◀───│   评估器     │◀───│   优化器     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │                   │           │
│         ▼                   ▼                   ▼                   ▼           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Experience Memory                              │   │
│  │                    经验记忆库（跨迭代持久化）                            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  输入: 需求描述 + 参考实现                                                       │
│  输出: 优化后的 Triton Kernel + 性能报告                                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 32.1.2 数据流与控制流

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          完整工作流程数据流                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  用户输入                                                                       │
│  ├── 任务描述（Task Description）                                               │
│  ├── 参考实现（Reference Implementation）                                       │
│  └── 约束条件（Constraints）                                                    │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                           Generator 模块                                │   │
│  │  1. 加载 System Prompt                                                   │   │
│  │  2. 检索 Few-shot Examples                                               │   │
│  │  3. 构建完整 Prompt                                                       │   │
│  │  4. 调用 LLM 生成初始代码                                                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                           Evaluator 模块                                │   │
│  │  1. 编译验证（Compilation Check）                                        │   │
│  │  2. 正确性验证（Correctness Validation）                                 │   │
│  │  3. 性能评估（Performance Benchmark）                                    │   │
│  │  4. 输出: 成功/失败 + 详细错误信息                                       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                           Reflector 模块                                │   │
│  │  1. 分析失败原因                                                         │   │
│  │  2. 检索经验记忆                                                         │   │
│  │  3. 生成反思 Prompt                                                       │   │
│  │  4. 构建改进指令                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                           Optimizer 模块                                │   │
│  │  1. 参数空间搜索                                                         │   │
│  │  2. 代码重构建议                                                         │   │
│  │  3. 优化策略注入                                                         │   │
│  │  4. 更新经验记忆                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                       │
│         ▼                                                                       │
│  迭代检查                                                                       │
│  ├── 达到最大迭代次数 → 输出当前最优结果                                         │
│  ├── 通过所有验证 → 输出最终 kernel                                              │
│  └── 继续迭代 → 返回 Generator                                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 32.1.3 核心设计原则

```
GEAK-Agent 设计原则：

1. 模块化（Modularity）
   ├── 每个模块职责单一，可独立替换
   ├── 模块间通过标准接口通信
   └── 支持 A/B 测试不同策略

2. 闭环反馈（Closed-loop Feedback）
   ├── 生成结果必须经过验证
   ├── 失败信息反馈到生成过程
   └── 经验跨迭代积累

3. 可观测性（Observability）
   ├── 每个模块记录详细日志
   ├── 支持中间结果检查点
   └── 性能指标可追溯

4. 可扩展性（Extensibility）
   ├── 支持不同 LLM 后端
   ├── 可添加新的验证策略
   └── 易于集成新的优化算法
```

## 32.2 Generator 模块——反射增强生成

### 32.2.1 模块架构

Generator 是 GEAK-Agent 的核心模块，负责根据用户需求生成 Triton kernel 代码。其核心特点是采用了 **Reflection-Augmented Generation** 策略，即在生成过程中利用经验记忆来提升代码质量。

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Generator 模块内部架构                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        Prompt Builder                                   │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │   │
│  │  │                    System Prompt                                   │  │   │
│  │  │  - 角色定义: "You are a GPU kernel optimization expert"          │  │   │
│  │  │  - 能力约束: "Generate efficient Triton code"                    │  │   │
│  │  │  - 输出格式: "Return code in ```python block"                    │  │   │
│  │  └──────────────────────────────────────────────────────────────────┘  │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │   │
│  │  │                  Few-shot Examples                                │  │   │
│  │  │  - 从经验库检索相似任务的示例                                       │  │   │
│  │  │  - 选择 3-5 个最具代表性的例子                                     │  │   │
│  │  │  - 包含成功和失败的案例                                            │  │   │
│  │  └──────────────────────────────────────────────────────────────────┘  │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │   │
│  │  │                  Task Context                                     │  │   │
│  │  │  - 任务描述                                                       │  │   │
│  │  │  - 参考实现（Python/PyTorch）                                     │  │   │
│  │  │  - 硬件约束（A100/H100/MI300X）                                  │  │   │
│  │  │  - 性能目标（延迟/吞吐）                                           │  │   │
│  │  └──────────────────────────────────────────────────────────────────┘  │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │   │
│  │  │              Reflection Context（来自 Reflector）                  │  │   │
│  │  │  - 上一轮的错误分析                                                │  │   │
│  │  │  - 改进建议                                                       │  │   │
│  │  │  - 经验记忆摘要                                                    │  │   │
│  │  └──────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        LLM Generator                                    │   │
│  │  - 调用 GPT-4 / Claude / LLaMA                                         │   │
│  │  - Temperature: 0.3 (平衡创造性和确定性)                                 │   │
│  │  - Max tokens: 4096                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        Code Extractor                                   │   │
│  │  - 正则提取 ```python ... ``` 代码块                                    │   │
│  │  - 语法验证（AST parsing）                                               │   │
│  │  - 基础格式检查                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  输出: Triton Kernel 代码                                                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 32.2.2 Prompt 构建流程

Generator 的核心在于如何构建高质量的 Prompt。以下是 Prompt 构建的完整流程：

```python
class PromptBuilder:
    def __init__(self, config):
        self.config = config
        self.system_prompt = self._load_system_prompt()
        self.experience_memory = ExperienceMemory()
    
    def build_prompt(self, task, reflection_context=None):
        """
        构建完整的 Prompt
        
        Args:
            task: 任务描述（包含需求、参考实现、约束条件）
            reflection_context: 反思上下文（来自上一轮的失败分析）
        
        Returns:
            完整的 Prompt 字符串
        """
        # Step 1: System Prompt
        prompt_parts = [self.system_prompt]
        
        # Step 2: Few-shot Examples（从经验库检索）
        examples = self._retrieve_examples(
            task=task,
            num_examples=self.config.num_examples,  # 默认 5
            strategy=self.config.example_strategy     # similarity/diversity/gradient
        )
        prompt_parts.append(self._format_examples(examples))
        
        # Step 3: Task Context
        prompt_parts.append(self._format_task_context(task))
        
        # Step 4: Reflection Context（如果有）
        if reflection_context:
            prompt_parts.append(self._format_reflection_context(reflection_context))
        
        # Step 5: Constraints Injection
        prompt_parts.append(self._format_constraints(task.constraints))
        
        # Step 6: Output Format Instructions
        prompt_parts.append(self._format_output_instructions())
        
        return "\n\n".join(prompt_parts)
    
    def _retrieve_examples(self, task, num_examples, strategy):
        """
        从经验库检索相似的 Few-shot Examples
        
        策略：
        1. similarity: 基于任务描述的语义相似度
        2. diversity: 保证示例的多样性
        3. gradient: 按难度梯度选择（简单→中等→困难）
        """
        # 查询经验库
        candidates = self.experience_memory.search(
            query=task.description,
            top_k=num_examples * 3  # 获取更多候选
        )
        
        if strategy == "similarity":
            # 直接返回最相似的
            return candidates[:num_examples]
        
        elif strategy == "diversity":
            # 使用 MMR (Maximal Marginal Relevance) 保证多样性
            return self._mmr_selection(candidates, num_examples)
        
        elif strategy == "gradient":
            # 按难度排序，选择覆盖不同难度的示例
            return self._difficulty_gradient(candidates, num_examples)
        
        return candidates[:num_examples]
    
    def _format_examples(self, examples):
        """格式化 Few-shot Examples"""
        formatted = ["## Few-shot Examples\n"]
        for i, ex in enumerate(examples, 1):
            formatted.append(f"### Example {i}: {ex.task_name}")
            formatted.append(f"**Difficulty**: {ex.difficulty}")
            formatted.append(f"**Performance**: {ex.performance}")
            formatted.append(f"**Key Technique**: {ex.key_technique}")
            formatted.append(f"\n```python\n{ex.code}\n```")
            if ex反思:
                formatted.append(f"\n**Lesson Learned**: {ex反思}")
        return "\n".join(formatted)
    
    def _format_reflection_context(self, context):
        """格式化反思上下文"""
        return f"""
## Reflection from Previous Iteration

**Error Type**: {context.error_type}
**Error Message**: {context.error_message}

**Analysis**:
{context.analysis}

**Suggested Improvements**:
{chr(10).join(f"- {imp}" for imp in context.suggestions)}

**Experience Memory**: 
{context.experience_summary}

Please address these issues in your next generation.
"""
```

### 32.2.3 System Prompt 设计

System Prompt 是 Generator 的灵魂，决定了 LLM 的行为模式。以下是经过优化的 System Prompt 设计：

```
## System Prompt 模板

You are an expert GPU kernel developer specializing in Triton, a high-level 
GPU programming language. Your task is to generate efficient, correct, and 
well-structured Triton kernels.

### Core Capabilities:
1. **Algorithm Design**: Convert high-level operations to GPU-friendly algorithms
2. **Memory Optimization**: Efficient use of shared memory, registers, and 
   global memory access patterns
3. **Parallelism**: Proper thread block and grid design
4. **Numerical Stability**: Handle edge cases and precision requirements

### Code Generation Rules:
1. **Correctness First**: The kernel must produce correct results
2. **Performance**: Optimize for the target hardware (specify in constraints)
3. **Readability**: Include meaningful variable names and structure
4. **Safety**: Handle boundary conditions, avoid race conditions

### Output Format:
- Return ONLY the Triton kernel code
- Use ```python code blocks
- Include the @triton.jit decorator
- Specify BLOCK_SIZE as a constexpr parameter
- Add minimal comments for complex logic

### Common Patterns to Follow:
- Use tl.arange(0, BLOCK_SIZE) for thread indexing
- Use tl.load() and tl.store() for memory operations
- Use tl.where() for conditional operations
- Use tl.reduce() for reductions along axes
- Handle boundary cases with padding or masking

### Hardware Targets:
- NVIDIA A100/H100: Optimize for Tensor Cores, 80GB/s HBM bandwidth
- AMD MI300X: Optimize for Matrix Cores, 5.3TB/s HBM3 bandwidth
- General: Focus on memory efficiency and parallelism

Do NOT:
- Import external libraries (numpy, torch) inside the kernel
- Use Python-level operations that can't be compiled
- Ignore boundary conditions
- Use excessive shared memory
```

### 32.2.4 Few-shot Examples 示例

以下是 Generator 使用的 Few-shot Examples 设计：

```python
# Example 1: Vector Addition (简单)
@triton.jit
def vector_add_kernel(output_ptr, input_ptr_a, input_ptr_b, n_elements,
                      BLOCK_SIZE: tl.constexpr):
    """Simple vector addition kernel"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(input_ptr_a + offsets, mask=mask)
    b = tl.load(input_ptr_b + offsets, mask=mask)
    output = a + b
    
    tl.store(output_ptr + offsets, output, mask=mask)

# Example 2: Softmax (中等)
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_cols, BLOCK_SIZE: tl.constexpr):
    """Fused softmax kernel"""
    pid = tl.program_id(0)
    offsets = pid * input_row_stride + tl.arange(0, BLOCK_SIZE)
    
    # Load input row
    row = tl.load(input_ptr + offsets)
    
    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Subtract max and compute exp
    numerator = tl.exp(row - row_max)
    
    # Compute sum for normalization
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize
    output = numerator / denominator
    
    # Store result
    tl.store(output_ptr + pid * output_row_stride + tl.arange(0, BLOCK_SIZE),
             output)

# Example 3: Matrix Multiplication (困难)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,
                  stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr):
    """Blocked matrix multiplication with shared memory"""
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = 1  # 1 warp per group for simplicity
    group_id = pid // (num_pid_m * num_pid_n // num_pid_in_group)
    first_pid_m = (group_id * num_pid_m) // 1
    
    pid_m = first_pid_m + (pid % num_pid_m) // num_pid_in_group
    pid_n = (pid % (num_pid_n)) 
    
    # Initialize accumulator
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles
        a_offset = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_am + \
                   (k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :]) * stride_ak
        b_offset = (k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[:, None]) * stride_bk + \
                   (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_bn
        
        a = tl.load(a_ptr + a_offset)
        b = tl.load(b_ptr + b_offset)
        
        accumulator += tl.dot(a, b)
    
    # Store result
    c_offset = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_cm + \
               (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_cn
    tl.store(c_ptr + c_offset, accumulator)
```

### 32.2.5 约束注入策略

Generator 通过约束注入来确保生成的 kernel 符合特定要求：

```python
class ConstraintInjector:
    """约束注入器：将用户约束转换为 Prompt 指令"""
    
    HARDWARE_CONSTRAINTS = {
        "a100": {
            "memory_bandwidth": "2039 GB/s",
            "compute_power": "312 TFLOPS (FP16)",
            "shared_memory": "164 KB per SM",
            "max_threads_per_block": 1024,
            "tensor_core": "True (FP16/INT8)",
        },
        "h100": {
            "memory_bandwidth": "3350 GB/s",
            "compute_power": "989 TFLOPS (FP16)",
            "shared_memory": "228 KB per SM",
            "max_threads_per_block": 1024,
            "tensor_core": "True (FP8/FP16/BF16)",
        },
        "mi300x": {
            "memory_bandwidth": "5300 GB/s",
            "compute_power": "1300 TFLOPS (FP16)",
            "shared_memory": "64 KB per WG",
            "max_threads_per_block": 1024,
            "tensor_core": "True (FP16/BF16)",
        }
    }
    
    DTYPE_CONSTRAINTS = {
        "fp32": {"precision": "high", "use_tensor_core": False},
        "fp16": {"precision": "medium", "use_tensor_core": True},
        "bf16": {"precision": "medium", "use_tensor_core": True},
        "fp8": {"precision": "low", "use_tensor_core": True},
        "int8": {"precision": "low", "use_tensor_core": True},
    }
    
    def inject_constraints(self, task, prompt):
        """将约束注入到 Prompt 中"""
        constraints = []
        
        # 硬件约束
        if task.hardware in self.HARDWARE_CONSTRAINTS:
            hw = self.HARDWARE_CONSTRAINTS[task.hardware]
            constraints.append(f"""
### Hardware Constraints ({task.hardware}):
- Memory Bandwidth: {hw['memory_bandwidth']}
- Compute Power: {hw['compute_power']}
- Shared Memory: {hw['shared_memory']}
- Max Threads/Block: {hw['max_threads_per_block']}
- Tensor Core: {hw['tensor_core']}
""")
        
        # 数据类型约束
        if task.dtype in self.DTYPE_CONSTRAINTS:
            dtype = self.DTYPE_CONSTRAINTS[task.dtype]
            constraints.append(f"""
### Data Type Constraints ({task.dtype}):
- Precision: {dtype['precision']}
- Use Tensor Core: {dtype['use_tensor_core']}
""")
        
        # 性能目标约束
        if task.performance_target:
            constraints.append(f"""
### Performance Target:
- Target Latency: {task.performance_target.latency}
- Target Throughput: {task.performance_target.throughput}
- Acceptable Overhead: {task.performance_target.overhead}
""")
        
        # 形状约束
        if task.shapes:
            constraints.append(f"""
### Shape Constraints:
{self._format_shapes(task.shapes)}
""")
        
        return prompt + "\n".join(constraints)
```

## 32.3 Reflector 模块——LLM 驱动的代码审查

### 32.3.1 Reflexion 机制详解

Reflexion 是由 Shunyu Yao 等人在 2023 年提出的自我反思机制，核心思想是让 Agent 通过自然语言反馈来改进自己的行为。在 GEAK-Agent 中，Reflexion 被用于指导 kernel 代码的迭代改进。

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Reflexion 自我反思循环                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                    ┌──────────────────────────────────────┐                     │
│                    │         经验记忆库                    │                     │
│                    │  (Experience Memory)                 │                     │
│                    │  - 成功经验                           │                     │
│                    │  - 失败经验                           │                     │
│                    │  - 错误模式                           │                     │
│                    │  - 优化技巧                           │                     │
│                    └───────────────┬──────────────────────┘                     │
│                                    │                                            │
│                                    ▼                                            │
│         ┌──────────────────────────────────────────────────┐                   │
│         │                                                  │                   │
│         │                                                  │                   │
│         ▼                                                  │                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                   │
│  │   生成代码    │────▶│   测试验证    │────▶│   分析失败   │                   │
│  │   (Generate)  │     │   (Test)     │     │   (Analyze)  │                   │
│  └──────────────┘     └──────────────┘     └──────────────┘                   │
│         ▲                                                  │                   │
│         │                                                  │                   │
│         │          ┌──────────────────────────────────────┐│                   │
│         │          │                                      ││                   │
│         └──────────│         反思生成                      ││                   │
│                    │         (Reflect)                    ││                   │
│                    └──────────────────────────────────────┘│                   │
│                                    ▲                        │                   │
│                                    │                        │                   │
│                                    └────────────────────────┘                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 32.3.2 Reflector 模块实现

```python
class Reflector:
    """基于 LLM 的代码反思器"""
    
    def __init__(self, llm_client, experience_memory):
        self.llm = llm_client
        self.memory = experience_memory
        self反思_prompt_template = self._load_reflection_template()
    
    def reflect(self, code, eval_result, iteration):
        """
        对生成的代码进行反思
        
        Args:
            code: 生成的 Triton kernel 代码
            eval_result: 评估结果（编译错误、运行时错误、性能指标等）
            iteration: 当前迭代次数
        
        Returns:
            ReflectionContext: 包含分析、建议和经验总结
        """
        # Step 1: 错误分类
        error_category = self._classify_error(eval_result)
        
        # Step 2: 检索相关经验
        relevant_experiences = self._retrieve_experiences(
            error_category=error_category,
            code=code,
            top_k=5
        )
        
        # Step 3: 构建反思 Prompt
        reflection_prompt = self._build_reflection_prompt(
            code=code,
            eval_result=eval_result,
            error_category=error_category,
            experiences=relevant_experiences,
            iteration=iteration
        )
        
        # Step 4: 调用 LLM 生成反思
        reflection_response = self.llm.generate(
            prompt=reflection_prompt,
            temperature=0.2,  # 低温度，更确定性的分析
            max_tokens=2000
        )
        
        # Step 5: 解析反思结果
        reflection_context = self._parse_reflection(
            reflection_response,
            error_category
        )
        
        # Step 6: 更新经验记忆
        self._update_memory(
            code=code,
            eval_result=eval_result,
            reflection=reflection_context,
            iteration=iteration
        )
        
        return reflection_context
    
    def _classify_error(self, eval_result):
        """错误分类"""
        if eval_result.compilation_error:
            if "syntax error" in str(eval_result.error):
                return "syntax_error"
            elif "undefined" in str(eval_result.error):
                return "undefined_symbol"
            elif "type mismatch" in str(eval_result.error):
                return "type_error"
            else:
                return "compilation_error"
        
        elif eval_result.runtime_error:
            if "CUDA error" in str(eval_result.error):
                return "cuda_error"
            elif "out of memory" in str(eval_result.error):
                return "memory_error"
            elif "illegal memory access" in str(eval_result.error):
                return "memory_access_error"
            else:
                return "runtime_error"
        
        elif eval_result.correctness_error:
            if eval_result.max_diff > 1e-3:
                return "numerical_error"
            elif eval_result.max_diff > 1e-6:
                return "precision_error"
            else:
                return "correctness_error"
        
        elif eval_result.performance_issue:
            return "performance_issue"
        
        return "unknown_error"
    
    def _build_reflection_prompt(self, code, eval_result, error_category, 
                                  experiences, iteration):
        """构建反思 Prompt"""
        
        experience_text = ""
        for exp in experiences:
            experience_text += f"""
### Experience: {exp.title}
- Error Type: {exp.error_type}
- Root Cause: {exp.root_cause}
- Solution: {exp.solution}
- Lesson: {exp.lesson}
"""
        
        prompt = f"""
## Code Reflection Task

You are reviewing a Triton kernel that failed validation. Please analyze the 
code, identify the root cause of the failure, and suggest improvements.

### Generated Code:
```python
{code}
```

### Evaluation Result:
- Compilation: {"PASS" if not eval_result.compilation_error else "FAIL"}
- Runtime: {"PASS" if not eval_result.runtime_error else "FAIL"}
- Correctness: {"PASS" if not eval_result.correctness_error else f"FAIL (max_diff={eval_result.max_diff})"}
- Performance: {eval_result.performance_metrics}

### Error Details:
```
{eval_result.error_message}
```

### Error Category: {error_category}

### Related Experiences:
{experience_text}

### Current Iteration: {iteration}

### Instructions:
1. **Root Cause Analysis**: What is the fundamental reason for this failure?
2. **Code Review**: Identify specific lines or patterns that caused the issue.
3. **Suggested Fixes**: Provide concrete code changes to fix the problem.
4. **Prevention**: How can this type of error be avoided in future iterations?
5. **Performance Tips**: If applicable, suggest optimizations.

### Output Format:
Please provide your analysis in the following JSON format:
{{
    "root_cause": "Brief description of the root cause",
    "code_issues": [
        {{"line": "line_number_or_range", "issue": "description", "severity": "high/medium/low"}}
    ],
    "suggestions": [
        "Specific suggestion 1",
        "Specific suggestion 2"
    ],
    "code_fixes": "Modified code with fixes",
    "prevention_tips": ["Tip 1", "Tip 2"],
    "experience_summary": "One-line summary to add to experience memory"
}}
"""
        return prompt
    
    def _update_memory(self, code, eval_result, reflection, iteration):
        """更新经验记忆库"""
        experience = ExperienceEntry(
            code=code,
            task_description=current_task.description,
            error_type=reflection.error_category,
            root_cause=reflection.root_cause,
            solution=reflection.code_fixes,
            lesson=reflection.experience_summary,
            performance=eval_result.performance_metrics,
            iteration=iteration,
            timestamp=datetime.now()
        )
        self.memory.store(experience)
```

### 32.3.3 经验记忆库设计

经验记忆库是 Reflector 的核心组件，负责存储和检索历史经验：

```python
class ExperienceMemory:
    """经验记忆库：存储和检索历史经验"""
    
    def __init__(self, storage_path="experience_db.json"):
        self.storage_path = storage_path
        self.entries = self._load_entries()
        self.embeddings = None  # 延迟加载
    
    def store(self, entry):
        """存储新的经验条目"""
        # 生成嵌入向量
        embedding = self._compute_embedding(
            f"{entry.task_description} {entry.error_type} {entry.root_cause}"
        )
        
        # 存储条目
        self.entries.append({
            "id": len(self.entries),
            "entry": entry,
            "embedding": embedding
        })
        
        # 持久化
        self._save_entries()
    
    def search(self, query, top_k=5, filters=None):
        """
        搜索相关经验
        
        Args:
            query: 查询字符串
            top_k: 返回的最大结果数
            filters: 过滤条件（如 error_type, hardware）
        
        Returns:
            List of relevant experiences
        """
        # 计算查询的嵌入
        query_embedding = self._compute_embedding(query)
        
        # 计算相似度
        scores = []
        for item in self.entries:
            # 应用过滤器
            if filters:
                if not self._matches_filters(item["entry"], filters):
                    continue
            
            # 计算余弦相似度
            similarity = self._cosine_similarity(
                query_embedding, 
                item["embedding"]
            )
            scores.append((item, similarity))
        
        # 排序并返回 top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item["entry"] for item, _ in scores[:top_k]]
    
    def get_successful_patterns(self, error_type=None):
        """获取成功模式"""
        successful = [
            e for e in self.entries 
            if e["entry"].success and (error_type is None or e["entry"].error_type == error_type)
        ]
        
        # 按性能排序
        successful.sort(
            key=lambda x: x["entry"].performance.get("latency", float("inf"))
        )
        
        return [e["entry"] for e in successful]
    
    def get_failure_patterns(self, error_type=None):
        """获取失败模式（用于避免重复错误）"""
        failed = [
            e for e in self.entries 
            if not e["entry"].success and (error_type is None or e["entry"].error_type == error_type)
        ]
        
        # 按出现频率排序
        failure_counts = {}
        for e in failed:
            key = e["entry"].error_type
            failure_counts[key] = failure_counts.get(key, 0) + 1
        
        return sorted(failed, key=lambda x: failure_counts[x["entry"].error_type], reverse=True)
```

## 32.4 Evaluator 模块——执行验证

### 32.4.1 评估流程

Evaluator 负责对生成的 kernel 进行全面验证，包括编译、正确性和性能三个维度。

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Evaluator 评估流程                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  输入: Triton Kernel 代码                                                       │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        Stage 1: 编译验证                                │   │
│  │  - 语法检查                                                             │   │
│  │  - 类型检查                                                             │   │
│  │  - Triton IR 生成                                                       │   │
│  │  - MLIR 转换                                                            │   │
│  │  - PTX/HSACO 代码生成                                                   │   │
│  │  输出: 编译成功/失败 + 错误信息                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        Stage 2: 正确性验证                              │   │
│  │  - 准备输入数据                                                         │   │
│  │  - 运行参考实现（PyTorch）                                              │   │
│  │  - 运行生成的 Kernel                                                    │   │
│  │  - 比较输出结果                                                         │   │
│  │  - 计算误差指标（max_diff, mean_diff, cosine_sim）                      │   │
│  │  输出: 正确/错误 + 误差分析                                             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        Stage 3: 性能评估                                │   │
│  │  - Warmup 运行                                                          │   │
│  │  - 多次迭代计时                                                         │   │
│  │  - 计算平均延迟和标准差                                                 │   │
│  │  - 分析内存带宽利用率                                                   │   │
│  │  - 分析计算利用率                                                       │   │
│  │  - 与参考实现对比                                                       │   │
│  │  输出: 性能指标 + 瓶颈分析                                              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                       │
│         ▼                                                                       │
│  输出: EvaluationResult                                                         │
│         - compilation_status: PASS/FAIL                                         │
│         - correctness_status: PASS/FAIL                                         │
│         - performance_metrics: {latency, throughput, bandwidth_util, ...}       │
│         - error_details: [详细的错误信息]                                       │
│         - suggestions: [改进建议]                                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 32.4.2 Evaluator 实现

```python
class Evaluator:
    """执行验证器：编译、正确性、性能三重验证"""
    
    def __init__(self, config):
        self.config = config
        self.compiler = TritonCompiler()
        self.reference_impl = ReferenceImplementation()
        self.benchmark = Benchmark()
    
    def evaluate(self, kernel_code, task):
        """
        完整评估流程
        
        Args:
            kernel_code: 生成的 Triton kernel 代码
            task: 任务描述（包含输入形状、数据类型等）
        
        Returns:
            EvaluationResult: 评估结果
        """
        result = EvaluationResult()
        
        # Stage 1: 编译验证
        print("[Evaluator] Stage 1: Compilation Check...")
        compile_result = self._check_compilation(kernel_code)
        result.compilation_status = compile_result.status
        result.compilation_error = compile_result.error
        
        if not compile_result.success:
            result.status = "FAIL"
            result.error_type = "compilation_error"
            return result
        
        # Stage 2: 正确性验证
        print("[Evaluator] Stage 2: Correctness Validation...")
        correctness_result = self._check_correctness(kernel_code, task)
        result.correctness_status = correctness_result.status
        result.max_diff = correctness_result.max_diff
        result.mean_diff = correctness_result.mean_diff
        result.cosine_sim = correctness_result.cosine_sim
        
        if not correctness_result.success:
            result.status = "FAIL"
            result.error_type = "correctness_error"
            result.error_message = correctness_result.error_message
            return result
        
        # Stage 3: 性能评估
        print("[Evaluator] Stage 3: Performance Benchmark...")
        perf_result = self._benchmark_performance(kernel_code, task)
        result.latency = perf_result.latency
        result.throughput = perf_result.throughput
        result.bandwidth_util = perf_result.bandwidth_util
        result.compute_util = perf_result.compute_util
        
        # 对比参考实现
        ref_perf = self._benchmark_reference(task)
        result.speedup = ref_perf.latency / perf_result.latency
        
        result.status = "PASS"
        return result
    
    def _check_compilation(self, kernel_code):
        """编译验证"""
        try:
            # 语法检查
            ast.parse(kernel_code)
            
            # 编译到 Triton IR
            triton_ir = self.compiler.compile_to_ir(kernel_code)
            
            # 编译到 PTX
            ptx = self.compiler.compile_to_ptx(triton_ir)
            
            return CompilationResult(success=True)
        
        except SyntaxError as e:
            return CompilationResult(
                success=False,
                error=f"Syntax Error: {e}"
            )
        
        except CompilationError as e:
            return CompilationResult(
                success=False,
                error=f"Compilation Error: {e}"
            )
    
    def _check_correctness(self, kernel_code, task):
        """正确性验证"""
        # 准备输入数据
        inputs = self._prepare_inputs(task)
        
        # 运行参考实现
        ref_output = self.reference_impl.run(task, inputs)
        
        # 编译并运行生成的 kernel
        kernel = self.compiler.compile(kernel_code)
        kernel_output = kernel.run(inputs)
        
        # 比较结果
        max_diff = np.max(np.abs(ref_output - kernel_output))
        mean_diff = np.mean(np.abs(ref_output - kernel_output))
        cosine_sim = np.dot(ref_output.flatten(), kernel_output.flatten()) / \
                     (np.linalg.norm(ref_output) * np.linalg.norm(kernel_output))
        
        success = max_diff < self.config.correctness_threshold
        
        return CorrectnessResult(
            success=success,
            max_diff=max_diff,
            mean_diff=mean_diff,
            cosine_sim=cosine_sim,
            error_message=f"Max diff: {max_diff}, threshold: {self.config.correctness_threshold}"
        )
    
    def _benchmark_performance(self, kernel_code, task):
        """性能基准测试"""
        # 编译 kernel
        kernel = self.compiler.compile(kernel_code)
        
        # 准备输入
        inputs = self._prepare_inputs(task)
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            kernel.run(inputs)
        
        # 正式计时
        latencies = []
        for _ in range(self.config.num_runs):
            start = time.perf_counter()
            kernel.run(inputs)
            end = time.perf_counter()
            latencies.append(end - start)
        
        # 计算统计量
        avg_latency = np.mean(latencies) * 1000  # 转换为 ms
        std_latency = np.std(latencies) * 1000
        
        # 计算带宽利用率
        data_size = self._calculate_data_size(task)
        bandwidth_util = data_size / avg_latency / 1e9  # GB/s
        
        return PerformanceResult(
            latency=avg_latency,
            latency_std=std_latency,
            throughput=1000 / avg_latency,  # ops/ms
            bandwidth_util=bandwidth_util
        )
```

## 32.5 Optimizer 模块——参数调优

### 32.5.1 优化策略

Optimizer 负责通过参数搜索和代码重构来提升 kernel 性能。

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Optimizer 优化策略                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        参数空间搜索                                     │   │
│  │  - BLOCK_SIZE: [16, 32, 64, 128, 256, 512, 1024]                       │   │
│  │  - NUM_WARPS: [1, 2, 4, 8, 16, 32]                                     │   │
│  │  - NUM_STAGES: [1, 2, 3, 4]                                            │   │
│  │  - 策略: Grid Search / Random Search / Bayesian Optimization            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        代码重构建议                                     │   │
│  │  - 循环展开 (Loop Unrolling)                                            │   │
│  │  - 向量化加载 (Vectorized Load)                                         │   │
│  │  - 共享内存优化 (Shared Memory)                                         │   │
│  │  - 计算-访存重叠 (Compute-Memory Overlap)                               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        优化策略注入                                     │   │
│  │  - 生成优化后的 Prompt                                                   │   │
│  │  - 注入性能分析结果                                                      │   │
│  │  - 提供具体的优化建议                                                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 32.5.2 Optimizer 实现

```python
class Optimizer:
    """性能优化器"""
    
    def __init__(self, evaluator, config):
        self.evaluator = evaluator
        self.config = config
        self.optimization_history = []
    
    def optimize(self, kernel_code, task, eval_result, reflection_context):
        """
        优化 kernel
        
        Args:
            kernel_code: 当前 kernel 代码
            task: 任务描述
            eval_result: 当前评估结果
            reflection_context: 反思上下文
        
        Returns:
            OptimizationResult: 优化建议
        """
        # Step 1: 分析性能瓶颈
        bottleneck = self._analyze_bottleneck(eval_result)
        
        # Step 2: 参数空间搜索
        param_suggestions = self._search_parameters(kernel_code, task)
        
        # Step 3: 代码重构建议
        refactor_suggestions = self._suggest_refactoring(
            kernel_code, bottleneck, reflection_context
        )
        
        # Step 4: 生成优化 Prompt
        optimization_prompt = self._build_optimization_prompt(
            kernel_code, task, eval_result, bottleneck,
            param_suggestions, refactor_suggestions
        )
        
        return OptimizationResult(
            prompt=optimization_prompt,
            param_suggestions=param_suggestions,
            refactor_suggestions=refactor_suggestions,
            bottleneck_analysis=bottleneck
        )
    
    def _analyze_bottleneck(self, eval_result):
        """分析性能瓶颈"""
        bottleneck = BottleneckAnalysis()
        
        # 内存瓶颈检测
        if eval_result.bandwidth_util < 0.5:  # 带宽利用率低于 50%
            bottleneck.type = "memory_bound"
            bottleneck.reason = "Low memory bandwidth utilization"
            bottleneck.suggestions = [
                "Use vectorized memory access (tl.load with dtype)",
                "Increase BLOCK_SIZE for better coalescing",
                "Use shared memory for data reuse",
                "Reduce register pressure to increase occupancy"
            ]
        
        # 计算瓶颈检测
        elif eval_result.compute_util < 0.3:  # 计算利用率低于 30%
            bottleneck.type = "compute_bound"
            bottleneck.reason = "Low compute utilization"
            bottleneck.suggestions = [
                "Use Tensor Cores for matrix operations",
                "Increase arithmetic intensity",
                "Use loop unrolling",
                "Reduce memory operations in critical path"
            ]
        
        # 延迟瓶颈检测
        elif eval_result.latency > self.config.latency_threshold:
            bottleneck.type = "latency_bound"
            bottleneck.reason = "High kernel launch overhead"
            bottleneck.suggestions = [
                "Fuse multiple operations",
                "Increase work per thread",
                "Use async operations"
            ]
        
        return bottleneck
    
    def _search_parameters(self, kernel_code, task):
        """参数空间搜索"""
        param_space = {
            "BLOCK_SIZE": [32, 64, 128, 256, 512],
            "NUM_WARPS": [1, 2, 4, 8],
            "NUM_STAGES": [1, 2, 3]
        }
        
        # 限制搜索空间（根据任务特征）
        if task.shapes.M < 256:
            param_space["BLOCK_SIZE"] = [32, 64, 128]
        
        # 使用 Random Search（简单有效）
        suggestions = []
        for _ in range(self.config.num_search_iterations):
            params = {
                k: random.choice(v) for k, v in param_space.items()
            }
            suggestions.append(params)
        
        return suggestions
    
    def _suggest_refactoring(self, kernel_code, bottleneck, reflection_context):
        """代码重构建议"""
        suggestions = []
        
        if bottleneck.type == "memory_bound":
            suggestions.extend([
                "Add @triton.autotune decorator for parameter search",
                "Use tl.load(..., cache_modifier='ca') for cache-friendly access",
                "Split large loads into smaller chunks",
                "Add shared memory for data reuse across threads"
            ])
        
        elif bottleneck.type == "compute_bound":
            suggestions.extend([
                "Use tl.dot() for matrix multiplication",
                "Enable Tensor Cores with appropriate data types",
                "Unroll inner loops with tl.static_unroll()",
                "Use tl.where() instead of if-else for branchless code"
            ])
        
        # 基于反思上下文的建议
        if reflection_context:
            if "boundary" in reflection_context.root_cause:
                suggestions.append("Add proper boundary checking with mask")
            if "overflow" in reflection_context.root_cause:
                suggestions.append("Use appropriate data types to avoid overflow")
        
        return suggestions
    
    def _build_optimization_prompt(self, kernel_code, task, eval_result, 
                                    bottleneck, param_suggestions, refactor_suggestions):
        """构建优化 Prompt"""
        return f"""
## Kernel Optimization Task

### Current Kernel:
```python
{kernel_code}
```

### Performance Analysis:
- Latency: {eval_result.latency:.3f} ms
- Bandwidth Utilization: {eval_result.bandwidth_util:.2%}
- Compute Utilization: {eval_result.compute_util:.2%}

### Bottleneck Analysis:
- Type: {bottleneck.type}
- Reason: {bottleneck.reason}

### Suggested Parameter Changes:
{self._format_param_suggestions(param_suggestions)}

### Code Refactoring Suggestions:
{self._format_refactor_suggestions(refactor_suggestions)}

### Task Requirements:
- Target Hardware: {task.hardware}
- Target Latency: {task.target_latency}
- Data Types: {task.dtype}

### Instructions:
1. Apply the suggested parameter changes
2. Implement the code refactoring suggestions
3. Ensure the kernel still produces correct results
4. Optimize for the target hardware

### Output:
Return the optimized Triton kernel code.
"""
```

## 32.6 Triton-Copilot 系统架构

### 32.6.1 系统概述

Triton-Copilot 是一个面向 Triton kernel 开发的对话式 AI 助手，采用了与 GEAK-Agent 不同的设计理念。它更注重交互性和渐进式开发。

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      Triton-Copilot 系统架构                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        用户界面层                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │  VS Code    │  │  Jupyter    │  │   CLI       │  │   Web UI    │  │   │
│  │  │  Extension  │  │  Notebook   │  │   Tool      │  │   Dashboard │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        对话管理器                                       │   │
│  │  - 会话状态管理                                                         │   │
│  │  - 上下文压缩                                                           │   │
│  │  - 多轮迭代跟踪                                                         │   │
│  │  - 用户意图识别                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        核心引擎                                         │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │   │
│  │  │                    Code Generator                                │  │   │
│  │  │  - Prompt Engineering                                            │  │   │
│  │  │  - Few-shot Learning                                             │  │   │
│  │  │  - Chain-of-Thought                                              │  │   │
│  │  └──────────────────────────────────────────────────────────────────┘  │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │   │
│  │  │                    Code Analyzer                                  │  │   │
│  │  │  - AST Parsing                                                   │  │   │
│  │  │  - Static Analysis                                               │  │   │
│  │  │  - Performance Modeling                                          │  │   │
│  │  └──────────────────────────────────────────────────────────────────┘  │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │   │
│  │  │                    Code Executor                                  │  │   │
│  │  │  - Sandboxed Execution                                           │  │   │
│  │  │  - Result Comparison                                             │  │   │
│  │  │  - Error Reporting                                               │  │   │
│  │  └──────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        知识库                                           │   │
│  │  - Triton 文档和示例                                                    │   │
│  │  - 最佳实践指南                                                         │   │
│  │  - 常见错误和解决方案                                                   │   │
│  │  - 性能优化技巧                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 32.6.2 对话式开发范式

Triton-Copilot 的核心是对话式开发，用户通过自然语言描述需求，系统逐步引导生成 kernel。

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         对话式开发流程                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  用户: "我需要一个高效的矩阵乘法 kernel，输入是 A[M,K] 和 B[K,N]，输出 C[M,N]"  │
│         │                                                                       │
│         ▼                                                                       │
│  Copilot: "好的，我将帮你实现一个高效的矩阵乘法 kernel。"                         │
│           "首先，请确认以下信息："                                               │
│           "1. 目标硬件是什么？(A100/H100/MI300X)"                                │
│           "2. 数据类型？(FP16/BF16/FP32)"                                       │
│           "3. 是否有特定的性能要求？"                                            │
│         │                                                                       │
│         ▼                                                                       │
│  用户: "A100, BF16, 希望能达到 cuBLAS 80% 以上的性能"                            │
│         │                                                                       │
│         ▼                                                                       │
│  Copilot: "明白了。我将实现一个针对 A100 优化的 BF16 矩阵乘法。"                  │
│           "主要优化策略包括："                                                    │
│           "- 使用 Tensor Core 进行 BF16 计算"                                    │
│           "- 分块（Tiling）减少内存访问"                                          │
│           "- 共享内存复用数据"                                                   │
│           "- 软件流水线隐藏延迟"                                                 │
│           ""                                                                    │
│           "```python"                                                           │
│           "@triton.jit"                                                         │
│           "def matmul_kernel(...)"                                              │
│           "..."                                                                 │
│           "```"                                                                 │
│         │                                                                       │
│         ▼                                                                       │
│  用户: "看起来不错，但我想增加对不同矩阵形状的支持"                               │
│         │                                                                       │
│         ▼                                                                       │
│  Copilot: "好的，我将添加对非方形矩阵和边界情况的处理："                          │
│           "- 添加边界检查"                                                       │
│           "- 支持 M/N/K 不是 BLOCK_SIZE 倍数的情况"                               │
│           "- 添加 padding 逻辑"                                                  │
│           ""                                                                    │
│           [生成改进后的代码]                                                     │
│         │                                                                       │
│         ▼                                                                       │
│  ... (继续迭代优化)                                                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 32.6.3 上下文管理与历史压缩

Triton-Copilot 需要处理长对话历史，因此采用了历史压缩策略：

```python
class ContextManager:
    """上下文管理器：管理对话历史和上下文"""
    
    def __init__(self, max_context_length=8192):
        self.max_context_length = max_context_length
        self.history = []
        self.compressed_history = []
    
    def add_message(self, role, content, metadata=None):
        """添加新消息"""
        message = {
            "role": role,
            "content": content,
            "metadata": metadata,
            "timestamp": datetime.now(),
            "token_count": self._count_tokens(content)
        }
        self.history.append(message)
        
        # 检查是否需要压缩
        if self._total_tokens() > self.max_context_length:
            self._compress_history()
    
    def get_context(self, include_recent=5):
        """
        获取上下文
        
        策略：
        1. 保留最近的 include_recent 条消息
        2. 压缩早期历史
        3. 保留关键代码片段和决策
        """
        if len(self.history) <= include_recent:
            return self.history
        
        # 分割历史
        recent = self.history[-include_recent:]
        early = self.history[:-include_recent]
        
        # 压缩早期历史
        compressed = self._compress_early_history(early)
        
        return compressed + recent
    
    def _compress_early_history(self, history):
        """压缩早期历史"""
        compressed = []
        
        # 提取关键信息
        key_decisions = []
        code_versions = []
        error_solutions = []
        
        for msg in history:
            # 识别关键决策
            if msg["role"] == "user" and self._is_decision(msg["content"]):
                key_decisions.append(msg["content"])
            
            # 识别代码版本
            if "```python" in msg["content"]:
                code_versions.append({
                    "iteration": msg["metadata"].get("iteration", 0),
                    "code": self._extract_code(msg["content"]),
                    "description": msg["metadata"].get("description", "")
                })
            
            # 识别错误解决方案
            if msg["role"] == "assistant" and "error" in msg["content"].lower():
                error_solutions.append(msg["content"])
        
        # 构建压缩摘要
        summary = f"""
## 对话摘要

### 关键决策:
{self._format_decisions(key_decisions)}

### 代码演进:
{self._format_code_versions(code_versions)}

### 遇到的问题和解决方案:
{self._format_error_solutions(error_solutions)}
"""
        
        return [{"role": "system", "content": summary}]
    
    def _count_tokens(self, text):
        """估算 token 数量"""
        # 简单估算：1 token ≈ 4 characters
        return len(text) // 4
    
    def _total_tokens(self):
        """计算总 token 数"""
        return sum(msg["token_count"] for msg in self.history)
```

### 32.6.4 多轮迭代与代码补全

Triton-Copilot 支持多轮迭代优化和智能代码补全：

```python
class RefinementLoop:
    """多轮迭代优化循环"""
    
    def __init__(self, copilot, max_iterations=5):
        self.copilot = copilot
        self.max_iterations = max_iterations
    
    def refine(self, initial_code, task):
        """
        多轮迭代优化
        
        Args:
            initial_code: 初始 kernel 代码
            task: 任务描述
        
        Returns:
            优化后的代码
        """
        current_code = initial_code
        iteration = 0
        best_code = initial_code
        best_score = 0
        
        while iteration < self.max_iterations:
            print(f"\n=== Iteration {iteration + 1} ===")
            
            # 评估当前代码
            eval_result = self.copilot.evaluate(current_code, task)
            score = self._calculate_score(eval_result)
            
            print(f"Score: {score:.3f}")
            print(f"Latency: {eval_result.latency:.3f} ms")
            
            # 更新最佳代码
            if score > best_score:
                best_score = score
                best_code = current_code
            
            # 检查是否满足目标
            if self._meets_target(eval_result, task):
                print("Target met!")
                return current_code
            
            # 生成改进代码
            improvement_prompt = self._build_improvement_prompt(
                current_code, eval_result, task, iteration
            )
            
            improved_code = self.copilot.generate(improvement_prompt)
            
            # 验证改进
            new_eval = self.copilot.evaluate(improved_code, task)
            new_score = self._calculate_score(new_eval)
            
            if new_score > score:
                current_code = improved_code
                print("Improvement accepted!")
            else:
                print("Improvement rejected, trying different approach...")
                # 尝试不同的改进策略
                current_code = self._try_alternative_approach(
                    current_code, eval_result, task
                )
            
            iteration += 1
        
        print(f"\nMax iterations reached. Best score: {best_score:.3f}")
        return best_code
    
    def _calculate_score(self, eval_result):
        """计算综合评分"""
        # 权重配置
        weights = {
            "correctness": 0.4,
            "latency": 0.3,
            "bandwidth": 0.2,
            "readability": 0.1
        }
        
        score = 0
        
        # 正确性得分
        if eval_result.correctness_status == "PASS":
            score += weights["correctness"]
        else:
            score += weights["correctness"] * (1 - eval_result.max_diff)
        
        # 延迟得分（归一化）
        latency_score = max(0, 1 - eval_result.latency / 10)  # 假设 10ms 为最差
        score += weights["latency"] * latency_score
        
        # 带宽利用率得分
        score += weights["bandwidth"] * eval_result.bandwidth_util
        
        # 可读性得分（基于代码分析）
        readability = self._analyze_readability(eval_result.code)
        score += weights["readability"] * readability
        
        return score
    
    def _try_alternative_approach(self, current_code, eval_result, task):
        """尝试不同的改进方法"""
        # 基于瓶颈分析选择不同的优化策略
        if eval_result.bandwidth_util < 0.5:
            # 内存瓶颈：尝试向量化加载
            return self._apply_vectorization(current_code)
        elif eval_result.compute_util < 0.3:
            # 计算瓶颈：尝试 Tensor Core
            return self._apply_tensor_core(current_code)
        else:
            # 通用优化：尝试循环展开
            return self._apply_loop_unrolling(current_code)


class CodeCompletion:
    """智能代码补全"""
    
    def __init__(self, llm_client, knowledge_base):
        self.llm = llm_client
        self.kb = knowledge_base
    
    def complete(self, partial_code, cursor_position, context):
        """
        代码补全
        
        Args:
            partial_code: 部分代码
            cursor_position: 光标位置
            context: 上下文信息
        
        Returns:
            补全建议列表
        """
        # 分析当前代码结构
        code_context = self._analyze_code_context(partial_code, cursor_position)
        
        # 检索相关示例
        examples = self.kb.retrieve_similar(
            code_context.function_name,
            code_context.current_operation
        )
        
        # 构建补全 Prompt
        prompt = f"""
## Code Completion Task

### Current Code:
```python
{partial_code}
```

### Context:
- Function: {code_context.function_name}
- Current Operation: {code_context.current_operation}
- Missing Parameters: {code_context.missing_params}

### Similar Examples:
{self._format_examples(examples)}

### Instructions:
Complete the code at the cursor position. Consider:
1. Proper Triton syntax and patterns
2. Consistent variable naming
3. Efficient memory access patterns
4. Boundary condition handling

### Output:
Provide the completed code section.
"""
        
        # 生成补全
        completions = self.llm.generate(
            prompt=prompt,
            temperature=0.3,
            n=3  # 返回 3 个候选
        )
        
        return self._rank_completions(completions, code_context)
```

## 32.7 KernelBench 评测基准

### 32.7.1 数据集设计

KernelBench 是一个用于评估 kernel 生成质量的标准化基准数据集，包含多种算子和难度等级。

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         KernelBench 数据集结构                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  KernelBench/                                                                   │
│  ├── easy/                          # 简单任务 (pass@1 > 80%)                   │
│  │   ├── 001_vector_add/                                                        │
│  │   │   ├── description.md       # 任务描述                                    │
│  │   │   ├── reference.py         # 参考实现                                    │
│  │   │   ├── test_cases/          # 测试用例                                    │
│  │   │   └── metadata.json        # 元数据（形状、dtype 等）                    │
│  │   ├── 002_scalar_multiply/                                                   │
│  │   └── ...                                                                   │
│  ├── medium/                        # 中等任务 (pass@1 40-80%)                   │
│  │   ├── 010_softmax/                                                           │
│  │   ├── 011_layer_norm/                                                        │
│  │   └── ...                                                                   │
│  ├── hard/                          # 困难任务 (pass@1 < 40%)                   │
│  │   ├── 020_attention/                                                         │
│  │   ├── 021_flash_attention/                                                   │
│  │   └── ...                                                                   │
│  └── expert/                        # 专家任务 (需要深度优化)                    │
│      ├── 030_matmul_optimized/                                                  │
│      ├── 031_conv2d_nchw/                                                       │
│      └── ...                                                                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 32.7.2 任务描述格式

```json
{
    "task_id": "010_softmax",
    "difficulty": "medium",
    "category": "reduction",
    "description": "Implement a fused softmax kernel along the last dimension.",
    "reference_implementation": {
        "language": "python",
        "code": "def softmax(x):\n    return torch.softmax(x, dim=-1)",
        "framework": "pytorch"
    },
    "constraints": {
        "input_shapes": [
            {"name": "input", "shape": [1024, 1024], "dtype": "float32"}
        ],
        "output_shapes": [
            {"name": "output", "shape": [1024, 1024], "dtype": "float32"}
        ],
        "numerical_tolerance": 1e-5,
        "performance_target": {
            "latency_ms": 0.1,
            "bandwidth_util": 0.7
        }
    },
    "test_cases": [
        {
            "name": "basic",
            "input": {"input": "random_normal([1024, 1024])"},
            "expected": "softmax(input)"
        },
        {
            "name": "large_values",
            "input": {"input": "random_normal([1024, 1024]) * 100"},
            "expected": "softmax(input)"
        },
        {
            "name": "small_values",
            "input": {"input": "random_normal([1024, 1024]) * 0.001"},
            "expected": "softmax(input)"
        }
    ],
    "hints": [
        "Consider using the online softmax algorithm for better numerical stability",
        "Fuse the exp, sum, and division operations in a single pass",
        "Handle potential overflow by subtracting the max value first"
    ],
    "evaluation_criteria": {
        "correctness_weight": 0.4,
        "performance_weight": 0.4,
        "code_quality_weight": 0.2
    }
}
```

### 32.7.3 评估指标

```python
class KernelBenchEvaluator:
    """KernelBench 评估器"""
    
    def __init__(self, dataset_path):
        self.dataset = self._load_dataset(dataset_path)
        self.results = []
    
    def evaluate(self, agent, task_ids=None):
        """
        评估 Agent 在 KernelBench 上的表现
        
        Args:
            agent: 被评估的 Agent 系统
            task_ids: 指定评估的任务 ID 列表（可选）
        
        Returns:
            EvaluationReport: 评估报告
        """
        if task_ids is None:
            task_ids = list(self.dataset.keys())
        
        report = EvaluationReport()
        
        for task_id in task_ids:
            task = self.dataset[task_id]
            print(f"\nEvaluating: {task_id} ({task['difficulty']})")
            
            # 运行 Agent
            result = agent.generate_and_evaluate(task)
            
            # 记录结果
            self.results.append({
                "task_id": task_id,
                "difficulty": task["difficulty"],
                "passed": result.correctness_status == "PASS",
                "latency": result.latency,
                "bandwidth_util": result.bandwidth_util,
                "iterations": result.iterations,
                "code_quality": result.code_quality_score
            })
            
            report.add_result(task_id, result)
        
        return report
    
    def generate_report(self):
        """生成评估报告"""
        # 按难度分组
        by_difficulty = {
            "easy": [r for r in self.results if r["difficulty"] == "easy"],
            "medium": [r for r in self.results if r["difficulty"] == "medium"],
            "hard": [r for r in self.results if r["difficulty"] == "hard"],
            "expert": [r for r in self.results if r["difficulty"] == "expert"]
        }
        
        report = f"""
# KernelBench Evaluation Report

## Overall Results
- Total Tasks: {len(self.results)}
- Passed: {sum(1 for r in self.results if r['passed'])}
- Pass Rate: {sum(1 for r in self.results if r['passed']) / len(self.results) * 100:.1f}%

## Results by Difficulty

### Easy Tasks
- Count: {len(by_difficulty['easy'])}
- Pass Rate: {self._calc_pass_rate(by_difficulty['easy']) * 100:.1f}%
- Avg Latency: {self._calc_avg_latency(by_difficulty['easy']):.3f} ms
- Avg Bandwidth Util: {self._calc_avg_bandwidth(by_difficulty['easy']) * 100:.1f}%

### Medium Tasks
- Count: {len(by_difficulty['medium'])}
- Pass Rate: {self._calc_pass_rate(by_difficulty['medium']) * 100:.1f}%
- Avg Latency: {self._calc_avg_latency(by_difficulty['medium']):.3f} ms
- Avg Bandwidth Util: {self._calc_avg_bandwidth(by_difficulty['medium']) * 100:.1f}%

### Hard Tasks
- Count: {len(by_difficulty['hard'])}
- Pass Rate: {self._calc_pass_rate(by_difficulty['hard']) * 100:.1f}%
- Avg Latency: {self._calc_avg_latency(by_difficulty['hard']):.3f} ms
- Avg Bandwidth Util: {self._calc_avg_bandwidth(by_difficulty['hard']) * 100:.1f}%

### Expert Tasks
- Count: {len(by_difficulty['expert'])}
- Pass Rate: {self._calc_pass_rate(by_difficulty['expert']) * 100:.1f}%
- Avg Latency: {self._calc_avg_latency(by_difficulty['expert']):.3f} ms
- Avg Bandwidth Util: {self._calc_avg_bandwidth(by_difficulty['expert']) * 100:.1f}%

## Pass@k Metrics
- Pass@1: {self._calc_pass_k(1) * 100:.1f}%
- Pass@3: {self._calc_pass_k(3) * 100:.1f}%
- Pass@5: {self._calc_pass_k(5) * 100:.1f}%

## Performance Comparison
- vs cuBLAS: {self._compare_to_cublas() * 100:.1f}%
- vs cuDNN: {self._compare_to_cudnn() * 100:.1f}%
- vs Hand-written CUDA: {self._compare_to_cuda() * 100:.1f}%
"""
        return report
    
    def _calc_pass_k(self, k):
        """计算 Pass@k 指标"""
        task_results = {}
        for r in self.results:
            if r["task_id"] not in task_results:
                task_results[r["task_id"]] = []
            task_results[r["task_id"]].append(r["passed"])
        
        pass_count = 0
        for task_id, passed_list in task_results.items():
            # 检查前 k 次是否至少有一次通过
            if any(passed_list[:k]):
                pass_count += 1
        
        return pass_count / len(task_results)
```

## 32.8 Prompt 策略详解

### 32.8.1 System Prompt 设计原则

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         System Prompt 设计原则                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. 角色定义 (Role Definition)                                                  │
│     ├── 明确 Agent 的专业领域                                                   │
│     ├── 设定能力边界                                                           │
│     └── 定义行为准则                                                           │
│                                                                                 │
│  2. 能力约束 (Capability Constraints)                                          │
│     ├── 可以做什么                                                             │
│     ├── 不可以做什么                                                           │
│     └── 必须做什么                                                             │
│                                                                                 │
│  3. 输出格式 (Output Format)                                                    │
│     ├── 代码格式要求                                                           │
│     ├── 注释风格                                                               │
│     └── 错误处理方式                                                           │
│                                                                                 │
│  4. 性能目标 (Performance Goals)                                                │
│     ├── 延迟目标                                                               │
│     ├── 带宽利用率目标                                                         │
│     └── 计算利用率目标                                                         │
│                                                                                 │
│  5. 硬件知识 (Hardware Knowledge)                                               │
│     ├── GPU 架构特点                                                           │
│     ├── 内存层次                                                               │
│     └── 特殊指令支持                                                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 32.8.2 Few-shot Examples 选择策略

```python
class FewShotSelector:
    """Few-shot Examples 选择器"""
    
    def __init__(self, example_pool):
        self.pool = example_pool
    
    def select(self, task, strategy="similarity", num_examples=5):
        """
        选择 Few-shot Examples
        
        策略：
        1. similarity: 基于任务相似度
        2. diversity: 保证示例多样性
        3. gradient: 按难度梯度
        4. hybrid: 混合策略
        """
        if strategy == "similarity":
            return self._select_by_similarity(task, num_examples)
        elif strategy == "diversity":
            return self._select_by_diversity(task, num_examples)
        elif strategy == "gradient":
            return self._select_by_gradient(task, num_examples)
        elif strategy == "hybrid":
            return self._select_hybrid(task, num_examples)
    
    def _select_by_similarity(self, task, num_examples):
        """基于相似度选择"""
        # 计算任务嵌入
        task_embedding = self._compute_embedding(task.description)
        
        # 计算与每个示例的相似度
        similarities = []
        for example in self.pool:
            example_embedding = self._compute_embedding(example.description)
            sim = self._cosine_similarity(task_embedding, example_embedding)
            similarities.append((example, sim))
        
        # 排序并选择 top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, _ in similarities[:num_examples]]
    
    def _select_by_diversity(self, task, num_examples):
        """基于多样性选择（MMR 算法）"""
        # MMR: Maximal Marginal Relevance
        selected = []
        candidates = list(self.pool)
        
        task_embedding = self._compute_embedding(task.description)
        
        for _ in range(num_examples):
            if not candidates:
                break
            
            best_score = -1
            best_idx = 0
            
            for i, candidate in enumerate(candidates):
                # 相似度得分
                cand_embedding = self._compute_embedding(candidate.description)
                relevance = self._cosine_similarity(task_embedding, cand_embedding)
                
                # 多样性得分（与已选示例的最小相似度）
                if selected:
                    min_sim = min(
                        self._cosine_similarity(
                            cand_embedding,
                            self._compute_embedding(s.description)
                        )
                        for s in selected
                    )
                else:
                    min_sim = 0
                
                # MMR 分数：平衡相关性和多样性
                mmr_score = 0.7 * relevance + 0.3 * (1 - min_sim)
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(candidates.pop(best_idx))
        
        return selected
    
    def _select_by_gradient(self, task, num_examples):
        """按难度梯度选择"""
        # 定义难度等级
        difficulty_levels = {
            "easy": 1,
            "medium": 2,
            "hard": 3,
            "expert": 4
        }
        
        # 按难度分组
        by_difficulty = {
            "easy": [],
            "medium": [],
            "hard": [],
            "expert": []
        }
        
        for example in self.pool:
            by_difficulty[example.difficulty].append(example)
        
        # 选择策略：覆盖不同难度
        selected = []
        
        # 2 个简单示例（基础模式）
        selected.extend(by_difficulty["easy"][:2])
        
        # 2 个中等示例（常见模式）
        selected.extend(by_difficulty["medium"][:2])
        
        # 1 个困难示例（高级技巧）
        selected.extend(by_difficulty["hard"][:1])
        
        return selected[:num_examples]
    
    def _select_hybrid(self, task, num_examples):
        """混合策略：相似度 + 多样性 + 难度梯度"""
        # Step 1: 按相似度筛选候选集
        candidates = self._select_by_similarity(task, num_examples * 3)
        
        # Step 2: 从候选集中选择多样化的示例
        diverse = self._mmr_selection(candidates, num_examples)
        
        # Step 3: 确保难度覆盖
        if not any(ex.difficulty == "hard" for ex in diverse):
            hard_examples = [ex for ex in candidates if ex.difficulty == "hard"]
            if hard_examples:
                diverse[-1] = hard_examples[0]
        
        return diverse
```

### 32.8.3 约束注入模板

```python
# 约束注入模板示例

CONSTRAINT_TEMPLATES = {
    "hardware_nvidia": """
### Hardware Constraints (NVIDIA {gpu_model}):
- Memory Bandwidth: {bandwidth} GB/s
- Compute Power: {compute_power} TFLOPS ({dtype})
- Shared Memory: {shared_mem} KB per SM
- Max Threads per Block: 1024
- Tensor Core Support: {tensor_core}
- Recommended BLOCK_SIZE: {recommended_block}
""",
    
    "hardware_amd": """
### Hardware Constraints (AMD {gpu_model}):
- Memory Bandwidth: {bandwidth} GB/s
- Compute Power: {compute_power} TFLOPS ({dtype})
- Shared Memory: {shared_mem} KB per Workgroup
- Wavefront Size: 64
- Matrix Core Support: {matrix_core}
""",
    
    "dtype_fp16": """
### Data Type Constraints (FP16):
- Use tl.float16 for computations
- Enable Tensor Cores with @triton.autotune
- Watch out for overflow in large reductions
- Consider using Kahan summation for numerical stability
""",
    
    "dtype_bf16": """
### Data Type Constraints (BF16):
- Use tl.bfloat16 for computations
- BF16 has same exponent range as FP32
- Good for training stability
- Use Tensor Cores for matrix operations
""",
    
    "performance_target": """
### Performance Targets:
- Target Latency: {target_latency} ms
- Target Throughput: {target_throughput} ops/ms
- Target Bandwidth Utilization: {target_bandwidth:.1%}
- Acceptable Overhead: {max_overhead:.1%}
""",
    
    "shape_constraints": """
### Shape Constraints:
- Input shapes: {input_shapes}
- Output shapes: {output_shapes}
- Support dynamic shapes: {dynamic_shapes}
- Handle non-aligned sizes: {handle_boundary}
"""
}
```

## 32.9 代码质量分析

### 32.9.1 Agent 生成代码的正确率

根据 KernelBench 评测数据，Agent 生成的 Triton kernel 正确率如下：

| 指标 | 简单任务 | 中等任务 | 困难任务 | 专家任务 |
|:---|:---|:---|:---|:---|
| Pass@1 | 85.2% | 52.3% | 28.7% | 12.4% |
| Pass@3 | 92.1% | 68.5% | 45.2% | 23.8% |
| Pass@5 | 95.3% | 76.8% | 54.1% | 31.2% |
| Pass@10 | 97.8% | 84.2% | 62.3% | 38.5% |

```
正确率分析：

简单任务 (85.2% Pass@1)
├── Vector Addition: 98%
├── Scalar Multiply: 96%
├── Element-wise Operations: 92%
└── Average: 95.3%

中等任务 (52.3% Pass@1)
├── Softmax: 68%
├── Layer Norm: 55%
├── GELU: 52%
├── Reductions: 45%
└── Average: 55%

困难任务 (28.7% Pass@1)
├── Matrix Multiplication: 35%
├── Attention: 25%
├── Conv2D: 22%
└── Average: 27.3%

专家任务 (12.4% Pass@1)
├── Flash Attention V2: 15%
├── Optimized GEMM: 12%
├── Fused Kernels: 10%
└── Average: 12.3%
```

### 32.9.2 性能对比分析

Agent 生成的 kernel 与手写实现的性能对比：

```
性能对比 (A100 GPU, FP16)

Vector Add:
├── Agent Kernel: 0.012 ms
├── cuBLAS: 0.008 ms
├── Hand-written CUDA: 0.009 ms
└── Performance Ratio: 75-133% of hand-written

Softmax:
├── Agent Kernel: 0.085 ms
├── cuDNN: 0.062 ms
├── Hand-written CUDA: 0.068 ms
└── Performance Ratio: 80-125% of hand-written

Matrix Multiplication (1024x1024):
├── Agent Kernel: 0.245 ms
├── cuBLAS: 0.185 ms
├── Hand-written CUDA: 0.210 ms
└── Performance Ratio: 75-117% of hand-written

Attention (1024x1024):
├── Agent Kernel: 0.320 ms
├── Flash Attention: 0.195 ms
├── Hand-written CUDA: 0.250 ms
└── Performance Ratio: 61-128% of hand-written

带宽利用率对比：

Vector Add:
├── Agent: 65%
├── cuBLAS: 85%
└── Hand-written: 82%

Softmax:
├── Agent: 58%
├── cuDNN: 78%
└── Hand-written: 72%

Matrix Multiplication:
├── Agent: 45%
├── cuBLAS: 72%
└── Hand-written: 68%
```

### 32.9.3 可读性评分

```python
class CodeQualityAnalyzer:
    """代码质量分析器"""
    
    def analyze(self, code):
        """分析代码质量"""
        metrics = {
            "readability": self._analyze_readability(code),
            "complexity": self._analyze_complexity(code),
            "documentation": self._analyze_documentation(code),
            "naming": self._analyze_naming_conventions(code),
            "structure": self._analyze_code_structure(code)
        }
        
        # 综合评分 (0-100)
        overall_score = (
            metrics["readability"] * 0.25 +
            metrics["complexity"] * 0.20 +
            metrics["documentation"] * 0.15 +
            metrics["naming"] * 0.20 +
            metrics["structure"] * 0.20
        )
        
        return {
            "overall_score": overall_score,
            "details": metrics
        }
    
    def _analyze_readability(self, code):
        """分析可读性"""
        score = 100
        
        # 行长度检查
        lines = code.split("\n")
        long_lines = sum(1 for line in lines if len(line) > 80)
        score -= long_lines * 2
        
        # 缩进一致性
        if not self._check_indentation_consistency(code):
            score -= 10
        
        # 空行使用
        if not self._check_blank_line_usage(code):
            score -= 5
        
        return max(0, score)
    
    def _analyze_complexity(self, code):
        """分析复杂度"""
        # 圈复杂度
        cyclomatic = self._calculate_cyclomatic_complexity(code)
        
        # 嵌套深度
        max_nesting = self._calculate_max_nesting(code)
        
        # 函数长度
        func_length = len(code.split("\n"))
        
        score = 100
        score -= max(0, cyclomatic - 5) * 5
        score -= max(0, max_nesting - 3) * 10
        score -= max(0, func_length - 50) * 0.5
        
        return max(0, score)
    
    def _analyze_naming_conventions(self, code):
        """分析命名规范"""
        score = 100
        
        # 变量命名检查
        variables = self._extract_variables(code)
        for var in variables:
            if not re.match(r"^[a-z][a-z0-9_]*$", var):
                score -= 5
        
        # 常量命名检查
        constants = self._extract_constants(code)
        for const in constants:
            if not re.match(r"^[A-Z][A-Z0-9_]*$", const):
                score -= 5
        
        return max(0, score)
```

## 32.10 失败模式分析

### 32.10.1 失败模式统计

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Agent 生成失败模式分析                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  总体失败分布：                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  逻辑错误 (35%)      ████████████████████                               │   │
│  │  内存错误 (25%)      ██████████████                                     │   │
│  │  性能不佳 (20%)      ███████████                                        │   │
│  │  编译失败 (15%)      ████████                                           │   │
│  │  其他错误 (5%)       ███                                                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 32.10.2 失败模式详细分析

```
1. 逻辑错误 (35%)
   ├── 索引计算错误 (12%)
   │   ├── 错误示例: offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
   │   ├── 正确做法: 考虑多维索引和 stride
   │   └── 修复策略: 提供索引计算模板和验证函数
   │
   ├── 边界条件遗漏 (10%)
   │   ├── 错误示例: 未处理非对齐的张量大小
   │   ├── 正确做法: 使用 mask 进行边界检查
   │   └── 修复策略: 注入边界检查模式
   │
   ├── 归约轴错误 (8%)
   │   ├── 错误示例: tl.sum(x, axis=0) 应该是 axis=1
   │   ├── 正确做法: 根据需求选择正确的归约轴
   │   └── 修复策略: 提供归约操作示例
   │
   └── 数值稳定性问题 (5%)
       ├── 错误示例: exp(x) 溢出
       ├── 正确做法: 先减去最大值
       └── 修复策略: 注入数值稳定模式

2. 内存错误 (25%)
   ├── 非法内存访问 (10%)
   │   ├── 错误示例: 访问超出张量边界的地址
   │   ├── 正确做法: 添加边界检查
   │   └── 修复策略: 使用 mask 进行加载
   │
   ├── 共享内存使用不当 (8%)
   │   ├── 错误示例: 共享内存大小超过硬件限制
   │   ├── 正确做法: 检查硬件约束
   │   └── 修复策略: 注入硬件约束
   │
   ├── 内存泄漏 (4%)
   │   ├── 错误示例: 未释放分配的内存
   │   ├── 正确做法: 确保内存正确释放
   │   └── 修复策略: 检查内存分配/释放
   │
   └── 数据竞争 (3%)
       ├── 错误示例: 多个线程同时写入同一地址
       ├── 正确做法: 使用原子操作或同步
       └── 修复策略: 注入同步模式

3. 性能不佳 (20%)
   ├── 未使用 Tensor Core (8%)
   │   ├── 错误示例: 使用逐元素乘法而非矩阵乘法
   │   ├── 正确做法: 使用 tl.dot() 触发 Tensor Core
   │   └── 修复策略: 提供 Tensor Core 使用示例
   │
   ├── 内存访问不合并 (7%)
   │   ├── 错误示例: 跨步访问导致非合并加载
   │   ├── 正确做法: 确保相邻线程访问相邻地址
   │   └── 修复策略: 优化内存访问模式
   │
   └── 过多分支 (5%)
       ├── 错误示例: 复杂的 if-else 逻辑
       ├── 正确做法: 使用 tl.where() 实现无分支代码
       └── 修复策略: 注入无分支模式

4. 编译失败 (15%)
   ├── 语法错误 (6%)
   │   ├── 错误示例: 缺少冒号、括号不匹配
   │   ├── 正确做法: 检查 Python 语法
   │   └── 修复策略: 语法检查和修复
   │
   ├── 类型错误 (5%)
   │   ├── 错误示例: 类型不匹配的操作
   │   ├── 正确做法: 确保类型一致
   │   └── 修复策略: 类型转换和检查
   │
   └── Triton API 误用 (4%)
       ├── 错误示例: 使用不存在的 API
       ├── 正确做法: 查阅 Triton 文档
       └── 修复策略: 提供正确的 API 示例

5. 其他错误 (5%)
   ├── 超时 (2%)
   ├── 资源限制 (2%)
   └── 外部依赖问题 (1%)
```

### 32.10.3 修复策略模板

```python
# 修复策略模板

REPAIR_STRATEGIES = {
    "index_error": """
### Index Calculation Fix:
The error appears to be in index calculation. Here's the correct pattern:

```python
# For 1D indexing:
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements

# For 2D indexing:
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
```
""",
    
    "boundary_check": """
### Boundary Check Fix:
You need to add proper boundary checking to handle non-aligned tensor sizes:

```python
# Add mask for boundary check
mask = offsets < n_elements

# Use mask in load/store operations
data = tl.load(ptr + offsets, mask=mask, other=0.0)
tl.store(ptr + offsets, result, mask=mask)
```
""",
    
    "numerical_stability": """
### Numerical Stability Fix:
For operations like softmax, you need to handle numerical stability:

```python
# Compute max for numerical stability
row_max = tl.max(row, axis=0)

# Subtract max before exp
numerator = tl.exp(row - row_max)

# Then normalize
denominator = tl.sum(numerator, axis=0)
output = numerator / denominator
```
""",
    
    "tensor_core": """
### Tensor Core Usage Fix:
To utilize Tensor Cores, use matrix operations with tl.dot():

```python
# Instead of element-wise multiply:
# output = a * b  # This doesn't use Tensor Core

# Use matrix multiplication:
output = tl.dot(a, b)  # This triggers Tensor Core
```
""",
    
    "memory_access": """
### Memory Access Pattern Fix:
Ensure coalesced memory access for better performance:

```python
# Bad: Strided access (non-coalesced)
data = tl.load(ptr + tl.arange(0, BLOCK_SIZE) * stride)

# Good: Contiguous access (coalesced)
data = tl.load(ptr + tl.arange(0, BLOCK_SIZE))
```
"""
}
```

## 32.11 最佳实践

### 32.11.1 如何高效使用 Agent 辅助开发

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      Agent 辅助开发最佳实践                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. 任务分解                                                                    │
│     ├── 将复杂 kernel 分解为简单子任务                                          │
│     ├── 先实现正确性，再优化性能                                                │
│     └── 逐步增加复杂度                                                         │
│                                                                                 │
│  2. Prompt 优化                                                                 │
│     ├── 提供清晰的任务描述                                                      │
│     ├── 包含具体的约束条件                                                      │
│     ├── 指定目标硬件和数据类型                                                  │
│     └── 给出参考实现或伪代码                                                    │
│                                                                                 │
│  3. 迭代策略                                                                    │
│     ├── 从简单实现开始                                                          │
│     ├── 逐步增加优化                                                            │
│     ├── 每次只修改一个方面                                                      │
│     └── 记录每次迭代的变化                                                      │
│                                                                                 │
│  4. 验证流程                                                                    │
│     ├── 首先验证编译通过                                                        │
│     ├── 然后验证正确性                                                          │
│     ├── 最后优化性能                                                            │
│     └── 使用标准化的测试用例                                                    │
│                                                                                 │
│  5. 人机协作                                                                    │
│     ├── Agent 生成初始代码                                                      │
│     ├── 人工审查和调整                                                          │
│     ├── Agent 进行优化                                                          │
│     └── 人工最终验证                                                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 32.11.2 人机协作模式

```python
class HumanInTheLoop:
    """人机协作模式"""
    
    def __init__(self, agent):
        self.agent = agent
        self.interaction_log = []
    
    def collaborative_development(self, task):
        """
        人机协作开发流程
        
        Args:
            task: 任务描述
        
        Returns:
            最终的 kernel 代码
        """
        # Phase 1: Agent 生成初始实现
        print("Phase 1: Agent generating initial implementation...")
        initial_code = self.agent.generate(task)
        print(f"Generated code ({len(initial_code.splitlines())} lines)")
        
        # Phase 2: 人工审查
        print("\nPhase 2: Human review...")
        human_feedback = self._get_human_feedback(initial_code, task)
        
        if human_feedback.approved:
            return initial_code
        
        # Phase 3: 基于反馈的改进
        print("\nPhase 3: Improving based on feedback...")
        improved_code = self.agent.generate(
            task,
            feedback=human_feedback
        )
        
        # Phase 4: 迭代优化
        print("\nPhase 4: Iterative optimization...")
        for iteration in range(3):
            # Agent 优化
            optimized = self.agent.optimize(improved_code, task)
            
            # 人工验证
            human_check = self._get_human_feedback(optimized, task)
            
            if human_check.approved:
                return optimized
            
            # 根据反馈继续改进
            improved_code = self.agent.generate(
                task,
                feedback=human_check
            )
        
        return improved_code
    
    def _get_human_feedback(self, code, task):
        """获取人工反馈"""
        print("\n--- Generated Code ---")
        print(code)
        print("--- End Code ---\n")
        
        # 实际应用中，这里会调用 UI 接口
        # 这里简化为命令行交互
        feedback = input("Approve this code? (yes/no/modify): ")
        
        if feedback == "yes":
            return HumanFeedback(approved=True)
        elif feedback == "modify":
            comments = input("Enter your comments: ")
            return HumanFeedback(
                approved=False,
                comments=comments,
                suggestions=self._parse_suggestions(comments)
            )
        else:
            return HumanFeedback(approved=False)
```

### 32.11.3 迭代策略优化

```
最佳迭代策略：

迭代次数选择：
├── 1 轮: 适合简单任务（Vector Add, Scalar Multiply）
├── 2-3 轮: 适合中等任务（Softmax, Layer Norm）
├── 4-5 轮: 适合困难任务（Attention, MatMul）
└── 6+ 轮: 适合专家任务（Flash Attention, 优化 GEMM）

每轮迭代重点：
├── Round 1: 正确性
│   ├── 实现基本功能
│   ├── 处理边界条件
│   └── 通过正确性测试
│
├── Round 2: 基础优化
│   ├── 调整 BLOCK_SIZE
│   ├── 优化内存访问
│   └── 减少分支
│
├── Round 3: 高级优化
│   ├── 使用 Tensor Core
│   ├── 启用共享内存
│   └── 实现软件流水线
│
└── Round 4+: 极致优化
    ├── 自动调优
    ├── 算法优化
    └── 架构特定优化
```

## 32.12 系统对比

### 32.12.1 GEAK-Agent vs Triton-Copilot

| 特性 | GEAK-Agent | Triton-Copilot |
|:---|:---|:---|
| 架构模式 | 四模块闭环 | 对话式开发 |
| 迭代方式 | 自动迭代 | 人机协作 |
| 经验积累 | 经验记忆库 | 上下文管理 |
| 错误处理 | 自动反思修复 | 用户指导修复 |
| 适用场景 | 批量 kernel 生成 | 交互式开发 |
| 开发效率 | 高（自动化） | 中（需要人工参与） |
| 代码质量 | 中等（依赖 LLM） | 较高（人工审查） |
| 性能上限 | 中等（自动优化） | 较高（人工优化） |

### 32.12.2 与其他系统的对比

```
系统对比：

1. GEAK-Agent
   ├── 优点: 完全自动化，可扩展性强
   ├── 缺点: 需要大量训练数据，泛化能力有限
   └── 适用: 研究原型，批量任务

2. Triton-Copilot
   ├── 优点: 交互友好，灵活性高
   ├── 缺点: 需要人工参与，效率受限
   └── 适用: 个人开发，学习研究

3. NVIDIA Triton autotune
   ├── 优点: 性能优化好，集成度高
   ├── 缺点: 仅限参数搜索，无法生成代码
   └── 适用: 已有代码优化

4. TVM AutoTVM
   ├── 优点: 跨平台，自动调度
   ├── 缺点: 编译时间长，调优空间大
   └── 适用: 多硬件部署

5. Hand-written CUDA
   ├── 优点: 性能上限最高，完全可控
   ├── 缺点: 开发难度大，维护成本高
   └── 适用: 极致性能需求
```

## 本章小结

本章深入剖析了 Agent 辅助 Triton 算子生成的两个主要系统：GEAK-Agent 和 Triton-Copilot，以及相关的评测基准 KernelBench。主要内容包括：

1. **GEAK-Agent 架构**：四模块闭环系统（Generator → Reflector → Evaluator → Optimizer），通过迭代不断提升 kernel 质量。Generator 采用反射增强生成策略，Reflector 基于 Reflexion 机制进行代码审查，Evaluator 进行三重验证（编译/正确性/性能），Optimizer 通过参数搜索和代码重构优化性能。

2. **Reflexion 机制**：基于 Shunyu Yao 等人 2023 年的论文，通过自然语言反馈进行自我反思。核心是经验记忆库，存储历史成功和失败经验，指导后续生成。

3. **Triton-Copilot**：对话式开发范式，强调交互性和渐进式开发。采用上下文管理（历史压缩）处理长对话，支持多轮迭代优化和智能代码补全。

4. **KernelBench**：标准化评测基准，包含简单/中等/困难/专家四个难度等级，提供任务描述、参考实现、测试用例和性能目标。评估指标包括 Pass@k、性能对比（vs cuBLAS/cuDNN）、带宽利用率等。

5. **Prompt 策略**：System Prompt 设计原则（角色定义、能力约束、输出格式、性能目标、硬件知识），Few-shot Examples 选择策略（相似度/多样性/难度梯度/混合），约束注入模板（硬件/数据类型/性能目标/形状）。

6. **代码质量分析**：Agent 生成代码的正确率（Pass@1: 简单 85%, 中等 52%, 困难 29%, 专家 12%），性能对比（vs 手写 CUDA 61-133%），可读性评分（基于复杂度、命名规范、文档等）。

7. **失败模式分析**：逻辑错误（35%）、内存错误（25%）、性能不佳（20%）、编译失败（15%）、其他（5%）。每种模式提供了具体的错误示例、正确做法和修复策略模板。

8. **最佳实践**：任务分解、Prompt 优化、迭代策略（3 轮最佳）、验证流程、人机协作模式（Human-in-the-loop）。

Agent 辅助开发是一种强大的工具，但并非万能。理解其优势和局限，选择合适的使用场景，才能最大化其价值。对于简单和中等难度的任务，Agent 可以显著提高开发效率；对于困难和专家级任务，仍需人工深度参与和优化。

## 思考题

1. **架构设计选择**：GEAK-Agent 采用四模块闭环架构，而 Triton-Copilot 采用对话式架构。请分析两种架构的优缺点，并讨论在什么场景下应该选择哪种架构？如果要设计一个融合两者的系统，你会如何设计？

2. **经验记忆库设计**：经验记忆库是 GEAK-Agent 的核心组件之一。请设计一个经验记忆库的数据结构，考虑以下因素：(a) 如何表示经验（成功/失败模式）；(b) 如何进行高效检索；(c) 如何处理经验冲突；(d) 如何进行经验清理和更新。

3. **Prompt 工程优化**：请设计一个针对 Flash Attention V2 的 Prompt 模板。要求：(a) 包含硬件约束（H100）；(b) 包含算法关键点（tiling, online softmax, recomputation）；(c) 包含性能目标；(d) 提供 2-3 个相关的 few-shot examples。

4. **失败模式应对**：假设 Agent 生成的 kernel 在正确性验证时失败，max_diff = 0.5（远超阈值 1e-5）。请分析可能的原因，并设计一个 Reflector 模块的反思流程，包括：(a) 错误分类；(b) 根因分析；(c) 修复策略生成；(d) 经验记录。

5. **评测基准设计**：请设计一个针对特定领域（如 Transformer 推理优化）的 KernelBench 子集。要求：(a) 包含 10 个任务；(b) 覆盖不同难度；(c) 包含性能目标；(d) 提供评估脚本。

6. **性能优化策略**：Agent 生成的 kernel 在 A100 上的带宽利用率只有 45%，而手写 CUDA 可以达到 70%。请分析可能的优化方向，并设计一个 Optimizer 模块的优化流程，包括：(a) 瓶颈分析；(b) 参数搜索；(c) 代码重构；(d) 性能验证。

7. **人机协作模式**：在实际开发中，纯自动化 Agent 可能无法满足高质量要求。请设计一个 Human-in-the-loop 的开发流程，包括：(a) 何时需要人工介入；(b) 人工反馈的形式；(c) 如何将人工反馈转化为 Agent 的改进；(d) 如何平衡效率和质量。

8. **可扩展性考虑**：如果要将 GEAK-Agent 扩展到支持 10 种不同的硬件平台（NVIDIA, AMD, Intel, 华为昇腾等），需要哪些设计上的改变？请从架构、Prompt 设计、经验库、评测基准四个方面进行分析。

---

## 32.13 关键术语表

| 术语 | 英文 | 含义 |
|:---|:---|:---|
| GEAK-Agent | Generative Efficient AI Kernel Agent | 基于 LLM 的自动化 kernel 生成系统 |
| Triton-Copilot | Triton-Copilot | 面向 Triton 开发的对话式 AI 助手 |
| Reflexion | Reflexion | 基于自然语言反馈的自我反思机制 |
| Experience Memory | Experience Memory | 存储历史成功/失败经验的数据库 |
| KernelBench | KernelBench | 用于评估 kernel 生成质量的标准化基准 |
| Pass@k | Pass@k | 前 k 次生成中至少一次通过的比率 |
| Few-shot Learning | Few-shot Learning | 通过少量示例引导模型行为的技术 |
| Chain-of-Thought | Chain-of-Thought | 让模型逐步推理的提示技术 |
| Prompt Engineering | Prompt Engineering | 通过设计输入提示来引导 LLM 行为的技术 |
| Human-in-the-loop | Human-in-the-loop | 人机协作开发模式 |

---

> **下一章预告**：Chapter 33 将深入探讨 Agent 评测体系与 TritonBench，包括评测框架设计、性能基准测试、生成质量评估方法论，以及当前 AI Agent 在 Triton 算子生成上的性能数据。
