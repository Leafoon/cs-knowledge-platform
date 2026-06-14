> **学习目标**：
> - 理解 TVM、XLA、MLIR 三大深度学习编译器生态的历史背景与设计哲学
> - 掌握三者在架构设计、IR 层次、优化策略上的核心差异
> - 能够对比三者在硬件覆盖、动态形状支持、前端框架兼容性等方面的能力
> - 理解搜索驱动优化（TVM）与规则驱动优化（XLA）的本质区别
> - 能够根据实际场景选择合适的编译器方案

---

## 36.1 三大编译器生态概述

### 36.1.1 TVM：学术驱动的开源编译器

TVM 起源于华盛顿大学陈天奇团队的研究项目，2018 年在 OSDI 会议上发表论文 *"TVM: An Automated End-to-End Optimizing Compiler for Deep Learning"* 后引起广泛关注。TVM 的核心愿景是弥合深度学习框架与多样化硬件后端之间的鸿沟。

TVM 的关键里程碑：

| 时间 | 事件 | 意义 |
|------|------|------|
| 2017 | TVM 项目启动 | 陈天奇在华盛顿大学开始研究 |
| 2018 | OSDI 论文发表 | 学术界首次系统化提出端到端 DL 编译器 |
| 2019 | Apache 孵化 | 进入 Apache 基金会，社区治理模式确立 |
| 2020 | AutoTVM / Ansor | 自动调度搜索技术成熟 |
| 2022 | MetaSchedule | 统一的搜索框架，替代 AutoTVM |
| 2023 | TVM Unity (Relax) | 统一三层 IR，消除语义鸿沟 |

TVM 的设计哲学可以概括为三个关键词：**表达与解耦（Decouple）**、**搜索驱动（Search-based）**、**硬件无关（Hardware-agnostic）**。

```python
# TVM 的核心理念：将计算描述与调度策略分离
# 这段代码展示了 TVM 的基本工作流
import tvm
from tvm import te

# 第一步：定义计算（WHAT）
A = te.placeholder((1024, 1024), name='A', dtype='float32')
B = te.placeholder((1024, 1024), name='B', dtype='float32')
k = te.reduce_axis((0, 1024), name='k')
C = te.compute((1024, 1024), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')

# 第二步：定义调度（HOW）—— 这部分可以由 AutoTVM/MetaSchedule 自动搜索
s = te.create_schedule(C.op)
```

### 36.1.2 XLA：Google 的生产级编译器

XLA（Accelerated Linear Algebra）是 Google 于 2017 年开源的深度学习编译器，最初为 TensorFlow 服务，后来成为 JAX 的核心编译后端。XLA 的设计目标是为 Google 内部的大规模 TPU 集群提供高效的编译优化。

XLA 的发展与 Google 的硬件战略紧密绑定：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
TensorFlow (2015)
    ↓ 需要高效的 TPU 编译
XLA (2017) —— 首版聚焦 TPU 后端
    ↓ JAX 需要函数式编译
JAX + XLA (2018) —— jit 编译成为 JAX 核心特性
    ↓ PyTorch 生态扩展
torch_xla (2019) —— PyTorch 通过 XLA 支持 TPU
    ↓ OpenXLA 基金会
OpenXLA (2023) —— 独立基金会，多厂商参与
```

XLA 的核心设计哲学是 **StableHLO（Stable High Level Optimizer）**——通过稳定的 HLO IR 定义，实现前端框架与硬件后端的解耦。

### 36.1.3 MLIR：编译器基础设施的新范式

MLIR（Multi-Level Intermediate Representation）是 Google 主导、LLVM 社区孵化的编译器基础设施项目，由 Chris Lattner（LLVM/Swift/Clang 之父）于 2019 年发起。MLIR 不是专门为深度学习设计的，而是提供了一套通用的 **Dialect（方言）** 系统来支持任意领域的编译。

MLIR 的关键里程碑：

| 时间 | 事件 | 意义 |
|------|------|------|
| 2019 | MLIR RFC 提出 | Chris Lattner 加入 Google 后发起 |
| 2020 | 进入 LLVM 社区 | 成为 LLVM 项目的官方子项目 |
| 2021 | Linalg Dialect 成熟 | 张量计算的标准化表示 |
| 2022 | StableHLO 定义 | XLA 的稳定 IR 基于 MLIR Dialect |
| 2023 | 广泛工业采用 | Google、Intel、AMD、Qualcomm 等均基于 MLIR 构建编译栈 |

MLIR 的设计哲学与 TVM/XLA 截然不同：**不定义固定的 IR，而是提供构建 IR 的基础设施**。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
传统编译器（如 LLVM）：
  源代码 → 固定 IR（如 LLVM IR）→ 目标代码
  问题：高层语义在 lowering 过程中丢失

MLIR：
  源代码 → Dialect A（高层语义）
         → Dialect B（中层优化）
         → Dialect C（低层代码生成）
  优势：每个 Dialect 保留领域特定语义，渐进式 lowering
```

### 36.1.4 三者定位对比

| 维度 | TVM | XLA | MLIR |
|------|-----|-----|------|
| **发起方** | 华盛顿大学 / Apache | Google | Google / LLVM |
| **核心定位** | 端到端 DL 编译器 | DL 编译器（TPU 优先） | 通用编译器基础设施 |
| **目标用户** | 硬件厂商、AI 公司 | Google 内部 + JAX 用户 | 编译器开发者 |
| **优化范式** | 搜索驱动 | 规则驱动 | 声明式转换 |
| **硬件策略** | 广覆盖 | TPU 深度优化 | 通过 Dialect 扩展 |
| **开源治理** | Apache Foundation | OpenXLA Foundation | LLVM Foundation |

<div data-component="CompilerEcosystemOverview"></div>

---

## 36.2 架构设计对比

### 36.2.1 TVM 的分层架构

TVM 采用经典的三层 IR 架构，每一层负责不同粒度的优化：

```
┌─────────────────────────────────────────────────────────────────┐
│                        TVM 编译栈                                │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Layer 1: Relay IR（计算图级）                              │  │
│  │  - 算子融合 (FuseOps)                                     │  │
│  │  - 常量折叠 (FoldConstant)                                │  │
│  │  - 部分求导 (PartialEvaluate)                             │  │
│  │  源码：src/relay/transforms/                              │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│                           ↓ LowerRelayToTensorIR               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Layer 2: TE / TIR（算子级）                                │  │
│  │  - 循环变换 (split/reorder/vectorize)                      │  │
│  │  - 内存管理 (compute_at/compute_root)                      │  │
│  │  - 并行化 (bind to thread axes)                            │  │
│  │  源码：src/tir/                                            │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│                           ↓ CodeGen                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Layer 3: 目标代码生成                                      │  │
│  │  - LLVM IR (CPU)      src/codegen/llvm/                    │  │
│  │  - CUDA Source (GPU)  src/codegen/source/                   │  │
│  │  - Vulkan SPIR-V      src/codegen/spirv/                   │  │
│  │  - WASM               src/codegen/wasm/                    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

TVM 的 Relay IR 定义在 `include/tvm/relay/` 目录下，核心数据结构包括：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// include/tvm/relay/expr.h —— Relay 表达式定义
class ExprNode : public RelayExprNode {
 public:
  /*! \brief 类型注解 */
  Type checked_type_;
  // ...
};

// include/tvm/relay/op.h —— 算子注册
// 每个 Relay 算子通过 OpRegistry 注册
TVM_REGISTER_OP("relay.op.nn.conv2d")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "Input data.")
    .add_argument("weight", "Tensor", "Input weight.");
```

实际的 Relay IR 表示示例：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relay

# 定义一个简单的卷积网络
data = relay.var("data", shape=(1, 3, 224, 224), dtype="float32")
weight = relay.var("weight", shape=(64, 3, 7, 7), dtype="float32")

# Relay IR 表示：每个操作是一个 Expr 节点
conv = relay.nn.conv2d(data, weight, strides=(2, 2), padding=(3, 3))
bn_gamma = relay.var("bn_gamma", shape=(64,), dtype="float32")
bn_beta = relay.var("bn_beta", shape=(64,), dtype="float32")
bn_mean = relay.var("bn_mean", shape=(64,), dtype="float32")
bn_var = relay.var("bn_var", shape=(64,), dtype="float32")
bn = relay.nn.batch_norm(conv, bn_gamma, bn_beta, bn_mean, bn_var)
relu = relay.nn.relu(bn[0])

# 整个计算图是一个嵌套的 Expr 树
func = relay.Function(relay.analysis.free_vars(relu), relu)
mod = tvm.IRModule.from_expr(func)

# 查看融合后的 IR
from tvm.relay import transform
seq = tvm.transform.Sequential([
    transform.FuseOps(fuse_opt_level=2),
])
with tvm.transform.PassContext(opt_level=3):
    mod = seq(mod)
print(mod)
```

### 36.2.2 XLA 的 HLO 架构

XLA 以 **HLO（High Level Optimizer）** IR 为核心，所有计算都被表示为 HLO 指令的有向无环图（DAG）。HLO IR 定义在 `tensorflow/compiler/xla/service/hlo_instruction.h` 中。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
┌─────────────────────────────────────────────────────────────────┐
│                        XLA 编译栈                                │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  前端：TensorFlow Graph / JAX jaxpr / StableHLO           │  │
│  │  源码：tensorflow/compiler/xla/client/                     │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│                           ↓ GraphCompiler                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  HLO IR（核心中间表示）                                     │  │
│  │  - 每个 HloInstruction 表示一个原子操作                     │  │
│  │  - HloModule 包含多个 HloComputation                       │  │
│  │  源码：tensorflow/compiler/xla/service/hlo_instruction.h   │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│                           ↓ Optimization Passes                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  优化 Pass Pipeline                                        │  │
│  │  - AlgebraicSimplifier   (代数化简)                        │  │
│  │  - InstructionFusion     (指令融合)                        │  │
│  │  - LayoutAssignment      (内存布局)                        │  │
│  │  - HLO CSE               (公共子表达式消除)                 │  │
│  │  源码：tensorflow/compiler/xla/service/*.cc                │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│                           ↓ CodeGen                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  后端代码生成                                               │  │
│  │  - TPU:  tensorflow/compiler/xla/service/tpu/              │  │
│  │  - GPU:  tensorflow/compiler/xla/service/gpu/              │  │
│  │  - CPU:  tensorflow/compiler/xla/service/cpu/              │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

HLO IR 的核心数据结构：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// tensorflow/compiler/xla/service/hlo_instruction.h
class HloInstruction {
 public:
  // 获取操作码
  HloOpcode opcode() const { return opcode_; }

  // 获取操作数
  const HloInstruction* operand(int64_t i) const;

  // 获取形状
  const Shape& shape() const { return shape_; }

  // 获取融合指令中的内部指令
  HloComputation* fused_instructions_computation() const;

 private:
  HloOpcode opcode_;
  Shape shape_;
  std::vector<HloInstruction*> operands_;
  // ...
};

// HLO 操作码枚举（部分）
enum class HloOpcode {
  kAdd, kMultiply, kConvolution, kDot, kReshape,
  kBroadcast, kReduce, kCustomCall, kFusion,
  // ... 约 200 种操作码
};
```

一个简单的矩阵乘法在 HLO 中的表示：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 使用 JAX 创建 HLO 计算
import jax
import jax.numpy as jnp

def matmul_fn(x, y):
    return jnp.dot(x, y)

# 编译为 HLO
x = jnp.ones((128, 256), dtype=jnp.float32)
y = jnp.ones((256, 512), dtype=jnp.float32)

# 获取 XLA 编译后的 HLO IR
compiled = jax.jit(matmul_fn).lower(x, y)
hlo_ir = compiled.compiler_ir(dialect="hlo")
print(hlo_ir)
# 输出类似：
# HloModule matmul_fn
# ENTRY %matmul_fn.4 (Arg_0.1: f32[128,256], Arg_1.2: f32[256,512]) -> f32[128,512] {
#   %Arg_0.1 = f32[128,256]{1,0} parameter(0)
#   %Arg_1.2 = f32[256,512]{1,0} parameter(1)
#   %dot.3 = f32[128,512]{1,0} dot(f32[128,256]{1,0} %Arg_0.1, f32[256,512]{1,0} %Arg_1.2),
#     lhs_contracting_dims={1}, rhs_contracting_dims={0}
# }
```

### 36.2.3 MLIR 的 Dialect 架构

MLIR 的核心创新是 **Dialect（方言）** 系统。每个 Dialect 定义了一组类型、操作和属性，编译过程就是将高层 Dialect 逐步 lower 到低层 Dialect。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
┌─────────────────────────────────────────────────────────────────┐
│                        MLIR 编译栈                               │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  高层 Dialect                                               │  │
│  │  - tosa/ (Tensor Operator Set Architecture)                │  │
│  │  - mhlo/ (StableHLO 的 MLIR 表示)                          │  │
│  │  源码：mlir/include/mlir/Dialect/TOSA/IR/TosaOps.td        │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│                           ↓ Dialect Conversion                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  中层 Dialect                                               │  │
│  │  - linalg/ (Linear Algebra dialect)                        │  │
│  │  - tensor/ (Tensor operations)                             │  │
│  │  - scf/   (Structured Control Flow)                        │  │
│  │  源码：mlir/include/mlir/Dialect/Linalg/IR/LinalgOps.td   │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│                           ↓ Bufferization + Tiling             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  低层 Dialect                                               │  │
│  │  - memref/ (Memory references)                             │  │
│  │  - vector/ (Vector operations)                             │  │
│  │  - arith/  (Arithmetic operations)                         │  │
│  │  - cf/     (Control flow)                                  │  │
│  │  - func/   (Functions)                                     │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│                           ↓ Lower to LLVM / SPIR-V            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  目标代码                                                   │  │
│  │  - llvm/ (LLVM IR dialect → 机器码)                        │  │
│  │  - gpu/  (GPU dialect → SPIR-V / NVVM)                    │  │
│  │  - amdgpu/ (AMD GPU dialect)                               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

MLIR 的 Dialect 定义使用 TableGen（`.td`）文件：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```tablegen
// mlir/include/mlir/Dialect/Linalg/IR/LinalgOps.td
// Linalg dialect 的算子定义（简化版）

def Linalg_Dialect : Dialect {
  let name = "linalg";
  let cppNamespace = "::mlir::linalg";
  let summary = "Linear algebra operations dialect";
}

// matmul 操作定义
def Linalg_MatmulOp : Linalg_Op<"matmul", [
    AttrSizedOperandSegments,
    LinalgStructuredOpInterface
  ]> {
  let summary = "Matrix multiplication operation";
  let arguments = (ins
    AnyRankedTensor:$inputs,
    AnyRankedTensor:$outputs
  );
  let results = (outs AnyRankedTensor:$result_tensors);
}
```

实际的 MLIR IR 表示示例（Linalg dialect 中的矩阵乘法）：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```mlir
// 使用 Linalg dialect 表示矩阵乘法
func.func @matmul(%A: tensor<128x256xf32>,
                   %B: tensor<256x512xf32>,
                   %C: tensor<128x512xf32>) -> tensor<128x512xf32> {
  %result = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x512xf32>)
                          outs(%C : tensor<128x512xf32>) -> tensor<128x512xf32>
  return %result : tensor<128x512xf32>
}

// lower 到 scf + memref 后：
func.func @matmul_lowered(%A: memref<128x256xf32>,
                           %B: memref<256x512xf32>,
                           %C: memref<128x512xf32>) {
  scf.for %i = 0 to 128 {
    scf.for %j = 0 to 512 {
      scf.for %k = 0 to 256 {
        %a = memref.load %A[%i, %k] : memref<128x256xf32>
        %b = memref.load %B[%k, %j] : memref<256x512xf32>
        %c = memref.load %C[%i, %j] : memref<128x512xf32>
        %prod = arith.mulf %a, %b : f32
        %sum = arith.addf %c, %prod : f32
        memref.store %sum, %C[%i, %j] : memref<128x512xf32>
      }
    }
  }
  return
}
```

### 36.2.4 架构对比总结



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
TVM 架构：
  Relay（图级）──→ TE/TIR（算子级）──→ CodeGen
  固定三层 IR，层间通过 Pass 转换

XLA 架构：
  Frontend ──→ HLO（唯一核心 IR）──→ Backend
  单一 IR + 规则化 Pass Pipeline

MLIR 架构：
  Dialect A ──→ Dialect B ──→ Dialect C ──→ ...
  可扩展的 Dialect 系统，渐进式 lowering
```

| 特性 | TVM | XLA | MLIR |
|------|-----|-----|------|
| **IR 数量** | 3 层（Relay/TE/TIR） | 1 层（HLO） | N 层（Dialect 链） |
| **可扩展性** | 新硬件需新增 CodeGen | 新硬件需新增 Backend | 新增 Dialect 即可 |
| **语义保留** | 层间信息丢失 | HLO 内信息保留 | 每层 Dialect 保留语义 |
| **学习曲线** | 中等 | 较低 | 较高 |

---

## 36.3 IR 层次差异与设计理念

### 36.3.1 TVM 的三层 IR 体系

TVM 的三层 IR 各有明确的职责边界：

**第一层：Relay IR（计算图级）**

Relay IR 位于 `include/tvm/relay/` 和 `src/relay/`，负责图级优化。Relay 的核心数据类型包括 `Expr`（表达式）、`Type`（类型）、`Function`（函数）。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Relay IR 中表示一个 conv2d 操作
import tvm
from tvm import relay

# 创建输入变量
data = relay.var("data", shape=(1, 3, 224, 224))
weight = relay.var("weight", shape=(64, 3, 7, 7))

# conv2d 在 Relay 中是一个 Call 节点
conv = relay.nn.conv2d(
    data, weight,
    strides=(2, 2),
    padding=(3, 3, 3, 3),
    dilation=(1, 1),
    groups=1,
    channels=64,
    kernel_size=(7, 7),
    data_layout="NCHW",
    kernel_layout="OIHW",
    out_dtype="float32"
)

# Relay IR 结构：
# CallNode {
#   op: Op("nn.conv2d")
#   args: [VarNode("data"), VarNode("weight")]
#   attrs: Conv2DAttrs { strides=[2,2], padding=[3,3,3,3], ... }
# }
```

**第二层：TE（Tensor Expression，张量表达式）**

TE 位于 `include/tvm/te/` 和 `src/te/`，负责算子级的计算描述与调度。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# TE 中表示 conv2d 的计算
from tvm import te

# 输入张量
Input = te.placeholder((1, 3, 224, 224), name="Input", dtype="float32")
Filter = te.placeholder((64, 3, 7, 7), name="Filter", dtype="float32")

# 输出尺寸计算
OH = (224 + 2*3 - 7) // 2 + 1  # = 112
OW = (224 + 2*3 - 7) // 2 + 1  # = 112

# 卷积计算的 TE 表示
rc = te.reduce_axis((0, 3), name="rc")
rh = te.reduce_axis((0, 7), name="rh")
rw = te.reduce_axis((0, 7), name="rw")

Output = te.compute(
    (1, 64, OH, OW),
    lambda n, c, h, w: te.sum(
        Input[n, rc, h*2 + rh, w*2 + rw] * Filter[c, rc, rh, rw],
        axis=[rc, rh, rw]
    ),
    name="Conv2D"
)
```

**第三层：TIR（Tensor IR，低层循环 IR）**

TIR 位于 `include/tvm/tir/` 和 `src/tir/`，是最接近目标硬件的表示。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# TIR 是 TE lower 后的结果，表示为显式的循环结构
# 可以通过以下方式查看 TIR：
from tvm import tir

# TE → TIR 的 lowering
s = te.create_schedule(Output.op)
func = tvm.build(s, [Input, Filter, Output], target="llvm", name="conv2d")

# TIR 表示（伪代码）：
# for (n, 0, 1):
#   for (c, 0, 64):
#     for (h, 0, 112):
#       for (w, 0, 112):
#         Conv2D[n, c, h, w] = 0f
#         for (rc, 0, 3):
#           for (rh, 0, 7):
#             for (rw, 0, 7):
#               Conv2D[n, c, h, w] += Input[n, rc, h*2+rh, w*2+rw] * Filter[c, rc, rh, rw]
```

### 36.3.2 XLA 的 HLO IR 层次

XLA 只有一层核心 IR——HLO，但 HLO 内部有不同的优化阶段。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# XLA 中 conv2d 的 HLO 表示
# 使用 JAX 来查看 HLO IR
import jax
import jax.numpy as jnp

def conv2d_fn(input, weight):
    # 使用 lax.conv_general_dilated 对应 XLA 的 Convolution HLO 指令
    return jax.lax.conv_general_dilated(
        input, weight,
        window_strides=(2, 2),
        padding=((3, 3), (3, 3)),
        dimension_numbers=("NCHW", "OIHW", "NCHW")
    )

# 编译获取 HLO
input_shape = jax.ShapeDtypeStruct((1, 3, 224, 224), jnp.float32)
weight_shape = jax.ShapeDtypeStruct((64, 3, 7, 7), jnp.float32)
compiled = jax.jit(conv2d_fn).lower(input_shape, weight_shape)
hlo = compiled.compiler_ir(dialect="hlo")
print(hlo)
```

HLO 的 `Convolution` 指令在 proto 中的定义：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```protobuf
// tensorflow/compiler/xla/xla_data.proto
message ConvolutionDimensionNumbers {
  // 输入的空间维度索引
  int64 input_spatial_dimensions = 6;
  // kernel 的空间维度索引
  int64 kernel_spatial_dimensions = 7;
  // 输出的空间维度索引
  int64 output_spatial_dimensions = 8;
  // 输入的 batch 维度
  int64 input_batch_dimension = 4;
  // 输入的 feature 维度
  int64 input_feature_dimension = 5;
  // kernel 的 input feature 维度
  int64 kernel_input_feature_dimension = 9;
  // kernel 的 output feature 维度
  int64 kernel_output_feature_dimension = 10;
  // 输出的 batch 维度
  int64 output_batch_dimension = 11;
  // 输出的 feature 维度
  int64 output_feature_dimension = 12;
}

// tensorflow/compiler/xla/service/hlo_instruction.cc
// HloInstruction 中 Convolution 的创建
StatusOr<HloInstruction*> HloInstruction::CreateConvolve(
    const Shape& shape,
    HloInstruction* lhs,
    HloInstruction* rhs,
    const Window& window,
    const ConvolutionDimensionNumbers& dnums,
    PrecisionConfig precision_config,
    /*preferred_element_type=*/std::optional<PrimitiveType>) {
  // ...
}
```

### 36.3.3 MLIR 的 Dialect 层次

MLIR 中表示 conv2d 涉及多个 Dialect，展示了渐进式 lowering 的过程。

**TOSA Dialect 层（高层语义）：**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```mlir
// 使用 TOSA dialect 表示 conv2d
// 源码参考：mlir/include/mlir/Dialect/TOSA/IR/TosaOps.td
func.func @conv2d_tosa(%input: tensor<1x224x224x3xf32>,
                        %weight: tensor<64x7x7x3xf32>,
                        %bias: tensor<64xf32>) -> tensor<1x112x112x64xf32> {
  // TOSA 的 conv2d 操作，保留了卷积的高层语义
  %conv = "tosa.conv2d"(%input, %weight, %bias) {
    dilation = array<i64: 1, 1>,
    pad = array<i64: 3, 3, 3, 3>,
    stride = array<i64: 2, 2>
  } : (tensor<1x224x224x3xf32>, tensor<64x7x7x3xf32>, tensor<64xf32>)
       -> tensor<1x112x112x64xf32>
  return %conv : tensor<1x112x112x64xf32>
}
```

**Linalg Dialect 层（中层优化）：**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```mlir
// TOSA → Linalg lowering 后
// 源码参考：mlir/include/mlir/Dialect/Linalg/IR/LinalgOps.td
func.func @conv2d_linalg(%input: tensor<1x3x224x224xf32>,
                          %filter: tensor<64x3x7x7xf32>,
                          %output: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> {
  // 使用 linalg.conv_2d_nhwc_hwcf 表示卷积
  // 这是 Linalg dialect 中标准化的卷积操作
  %result = linalg.conv_2d_nhwc_hwcf
    ins(%input, %filter : tensor<1x224x224x3xf32>, tensor<7x7x3x64xf32>)
    outs(%output : tensor<1x112x112x64xf32>) {
      ^bb0(%in: f32, %inf: f32, %out: f32):
        %mul = arith.mulf %in, %inf : f32
        %add = arith.addf %out, %mul : f32
        linalg.yield %add : f32
    } -> tensor<1x112x112x64xf32>
  return %result : tensor<1x112x112x64xf32>
}
```

**Arith/SCF Dialect 层（低层循环）：**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```mlir
// Linalg → SCF + Arith lowering 后
func.func @conv2d_loops(%input: memref<1x3x224x224xf32>,
                         %filter: memref<64x3x7x7xf32>,
                         %output: memref<1x64x112x112xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c7 = arith.constant 7 : index
  %c64 = arith.constant 64 : index
  %c112 = arith.constant 112 : index
  %c224 = arith.constant 224 : index

  scf.for %n = %c0 to %c1 step %c1 {
    scf.for %oc = %c0 to %c64 step %c1 {
      scf.for %oh = %c0 to %c112 step %c1 {
        scf.for %ow = %c0 to %c112 step %c1 {
          scf.for %ic = %c0 to %c3 step %c1 {
            scf.for %kh = %c0 to %c7 step %c1 {
              scf.for %kw = %c0 to %c7 step %c1 {
                // 计算输入索引（考虑 stride=2, pad=3）
                %ih = arith.muli %oh, %c2 : index
                %ih_pad = arith.addi %ih, %kh : index
                %ih_final = arith.subi %ih_pad, %c3 : index
                %iw = arith.muli %ow, %c2 : index
                %iw_pad = arith.addi %iw, %kw : index
                %iw_final = arith.subi %iw_pad, %c3 : index

                // 边界检查 + 加载
                %in_val = memref.load %input[%n, %ic, %ih_final, %iw_final]
                    : memref<1x3x224x224xf32>
                %w_val = memref.load %filter[%oc, %ic, %kh, %kw]
                    : memref<64x3x7x7xf32>
                %out_val = memref.load %output[%n, %oc, %oh, %ow]
                    : memref<1x64x112x112xf32>

                // 乘加
                %mul = arith.mulf %in_val, %w_val : f32
                %acc = arith.addf %out_val, %mul : f32
                memref.store %acc, %output[%n, %oc, %oh, %ow]
                    : memref<1x64x112x112xf32>
              }
            }
          }
        }
      }
    }
  }
  return
}
```

### 36.3.4 IR 设计理念对比

| 维度 | TVM Relay | XLA HLO | MLIR Linalg |
|------|-----------|---------|-------------|
| **粒度** | 图级算子 | 原子操作 | 结构化操作 |
| **语义层次** | 高层（nn.conv2d） | 中层（Convolution） | 可配置 |
| **类型系统** | Relay Type | Shape proto | MLIR Type |
| **控制流** | If/Loop 节点 | While/Conditional | SCF dialect |
| **可扩展性** | 通过 Op 注册 | 通过 CustomCall | 通过 Dialect |



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
语义保留度对比（从高到低）：

TVM Relay:    ████████████  （高层语义丰富，但层次固定）
MLIR Linalg:  ████████████  （每层保留结构化语义）
XLA HLO:      ██████████    （原子操作，语义相对扁平）
MLIR Arith:   ██████        （低层算术，语义最少）
TVM TIR:      ██████        （低层循环，语义最少）
```

---

## 36.4 优化策略对比：搜索驱动 vs 规则驱动

### 36.4.1 TVM 的搜索驱动优化

TVM 的核心创新是 **基于搜索的自动优化**。TVM 不依赖手工编写的优化规则，而是定义一个搜索空间，然后用机器学习算法搜索最优配置。

**AutoTVM 的搜索空间定义：**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# python/tvm/auto_scheduler/compute_dag.py
# AutoTVM 的 Schedule Template 定义
import tvm
from tvm import te, auto_scheduler

# 定义计算任务
@auto_scheduler.register_workload
def matmul(M, N, K, dtype):
    A = te.placeholder((M, K), name='A', dtype=dtype)
    B = te.placeholder((K, N), name='B', dtype=dtype)
    k = te.reduce_axis((0, K), name='k')
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    return [A, B, C]

# 搜索空间由 ScheduleRules 和 SearchStrategy 定义
# python/tvm/meta_schedule/space_generator/
@auto_scheduler.register_workload
def conv2d_nchw(N, C, H, W, KH, KW, OC, stride, padding):
    Input = te.placeholder((N, C, H, W), name="Input")
    Filter = te.placeholder((OC, C, KH, KW), name="Filter")
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    rc = te.reduce_axis((0, C), name="rc")
    rh = te.reduce_axis((0, KH), name="rh")
    rw = te.reduce_axis((0, KW), name="rw")
    Output = te.compute(
        (N, OC, OH, OW),
        lambda n, oc, oh, ow: te.sum(
            Input[n, rc, oh * stride + rh - padding, ow * stride + rw - padding] *
            Filter[oc, rc, rh, rw],
            axis=[rc, rh, rw]
        ),
        name="Output"
    )
    return [Input, Filter, Output]
```

**MetaSchedule 的搜索策略：**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# python/tvm/meta_schedule/search_strategy/
# MetaSchedule 提供了多种搜索策略

from tvm import meta_schedule as ms

# 定义搜索空间生成器
space_generator = ms.space_generator.ReplayTrace()
# 或使用预定义的规则
space_generator = ms.space_generator.ScheduleRule([
    ms.schedule_rule.AutoInline(
        into_producer=True,
        into_consumer=True,
        inline_const_tensor=True
    ),
    ms.schedule_rule.MultiLevelTiling(
        structure="SSRSRS",
        tile_binds=["blockIdx.x", "threadIdx.x"],
        max_innermost_factor=4,
        vector_load_lens=[1, 2, 4],
        reuse_read=ms.schedule_rule.ReuseType(
            req="may",
            levels=[1, 2],
            scope="shared"
        ),
    ),
])

# 搜索策略：代价模型引导的搜索
search_strategy = ms.search_strategy.ReplayFunc()

# 搜索过程
sch = ms.tune_tir(
    mod=matmul_mod,
    target="cuda",
    space_generator=space_generator,
    search_strategy=search_strategy,
    max_trials_global=1000,
    num_trials_per_iter=64,
)
```

**MetaSchedule 的代价模型：**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# python/tvm/meta_schedule/cost_model/
# 代价模型用于预测调度配置的性能

# MetaSchedule 使用多种代价模型
cost_model = ms.cost_model.XGBModel(
    # 特征提取器
    extractor=ms.feature_extractor.PerStoreFeature(),
    # 搜索空间的维度
    space=space_generator,
    # 训练参数
    learning_rate=0.1,
    max_bin=256,
)

# 搜索循环的核心逻辑（简化）：
# 1. 从搜索空间采样一批候选调度
# 2. 用代价模型预测每个候选的性能
# 3. 选择 top-K 候选进行实际测量
# 4. 用测量结果更新代价模型
# 5. 重复直到达到收敛或预算用完
```

### 36.4.2 XLA 的规则驱动优化

XLA 采用完全不同的策略——**基于规则的确定性优化**。每个优化 Pass 都是手工编写的，应用固定的优化规则。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// tensorflow/compiler/xla/service/algebraic_simplifier.cc
// 代数化简 Pass：应用代数恒等式进行化简

Status AlgebraicSimplifierVisitor::HandleDot(HloInstruction* dot) {
  // 规则1：单位矩阵消除 A @ I = A
  if (IsIdentityMatrix(dot->operand(1))) {
    return ReplaceInstruction(dot, dot->operand(0));
  }
  // 规则2：零矩阵消除 A @ 0 = 0
  if (IsAllZeros(dot->operand(1))) {
    return ReplaceInstruction(dot, MakeZerosLike(dot));
  }
  // 规则3：转置合并 (A @ B)^T = B^T @ A^T
  if (/* ... */) {
    // ... transform
  }
  return OkStatus();
}
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// tensorflow/compiler/xla/service/instruction_fusion.cc
// 指令融合 Pass：决定哪些 HLO 指令可以融合

bool InstructionFusion::ShouldFuse(HloInstruction* consumer,
                                    int64_t operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);

  // 规则：如果融合后不会增加内存带宽消耗，则融合
  // 具体判断逻辑：
  // 1. producer 只有一个 consumer（避免重复计算）
  // 2. 融合后的计算量不超过阈值
  // 3. 不跨越控制流边界

  if (producer->user_count() > 1) {
    // 多个 consumer 时，只在第一个 consumer 处融合
    if (producer->users().front() != consumer) {
      return false;
    }
  }

  // 检查是否是 element-wise 操作（容易融合）
  if (IsElementwise(producer->opcode())) {
    return true;
  }

  // 检查是否是 reduce 操作（可能需要特殊处理）
  if (producer->opcode() == HloOpcode::kReduce) {
    return ShouldFuseReduce(consumer, producer);
  }

  return false;
}
```

**XLA 的 GPU 融合 Pass：**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// tensorflow/compiler/xla/service/gpu/gpu_instruction_fusion.cc
// GPU 特有的融合策略

class GpuInstructionFusion : public InstructionFusion {
 public:
  bool ShouldFuse(HloInstruction* consumer, int64_t operand_index) override {
    HloInstruction* producer = consumer->mutable_operand(operand_index);

    // GPU 特有规则：避免产生过大的 kernel
    // CUDA kernel 的寄存器数量有限
    int64_t fused_size = GetFusedSize(producer, consumer);
    if (fused_size > kMaxKernelSize) {
      return false;
    }

    // GPU 特有规则：考虑 shared memory 使用
    // 如果融合后需要大量 shared memory，可能降低 occupancy
    if (RequiresSharedMemory(producer, consumer) &&
        GetSharedMemoryUsage() > kMaxSharedMemory) {
      return false;
    }

    return InstructionFusion::ShouldFuse(consumer, operand_index);
  }
};
```

### 36.4.3 MLIR 的声明式转换

MLIR 采用 **声明式转换（Declarative Transformation）**，通过 `td/` 文件定义模式匹配和重写规则。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```tablegen
// mlir/include/mlir/Dialect/Linalg/IR/LinalgTransformOps.td
// 使用 TableGen 定义 Linalg 的 tiling 规则

def TileOp : Linalg_TransformOp<"tile", [
    AttrSizedOperandSegments
  ]> {
  let summary = "Tile a linalg operation";
  let arguments = (ins
    AnyOpType:$target,
    DefaultValuedOptionalAttr<I64ArrayAttr, "{}">:$sizes,
    DefaultValuedOptionalAttr<I64ArrayAttr, "{}">:$interchange,
    DefaultValuedOptionalAttr<StrAttr, "">:$pad
  );
  let results = (outs AnyOpType:$tiled_op);
}
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```mlir
// MLIR 的 transform dialect：声明式优化编排
// 使用 transform dialect 编写优化策略

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // 找到所有 linalg.matmul 操作
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
    : (!transform.any_op) -> !transform.any_op

  // 对 matmul 进行 tiling
  %tiled, %loops:3 = transform.structured.tile %matmul [32, 32, 8]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                               !transform.any_op, !transform.any_op)

  // 对内层循环进行 vectorization
  transform.structured.vectorize %tiled

  // 将外层循环映射到 GPU 线程
  transform.gpu.map_forall_to_threads %loops#0 block_dims [4, 8, 1]
}
```

### 36.4.4 优化策略的本质区别

| 维度 | TVM (搜索驱动) | XLA (规则驱动) | MLIR (声明式) |
|------|---------------|---------------|--------------|
| **优化决策** | ML 模型预测 | 手写规则 | 声明式 pattern |
| **可确定性** | 非确定性 | 确定性 | 确定性 |
| **新硬件适配** | 自动搜索 | 需编写规则 | 需定义 Dialect |
| **优化质量** | 接近最优 | 依赖规则质量 | 依赖 pattern 覆盖 |
| **编译时间** | 搜索耗时 | 快速 | 中等 |
| **可调试性** | 较难 | 容易 | 中等 |



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
优化流程对比：

TVM:
  定义搜索空间 → 采样候选 → 评估性能 → 更新模型 → 重复
  关键：搜索空间设计 + 代价模型精度

XLA:
  读取 HLO IR → 应用 Pass 1 → 应用 Pass 2 → ... → 输出
  关键：Pass 顺序 + 规则正确性

MLIR:
  定义 Pattern → 匹配 Dialect → 应用 Rewrite → lower → 重复
  关键：Pattern 设计 + Dialect 层次
```

---

## 36.5 硬件覆盖范围对比

### 36.5.1 TVM 的硬件后端

TVM 通过 CodeGen 模块支持多种硬件后端：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
python/tvm/codegen.py
    ↓
src/codegen/
├── llvm/              # CPU 后端（LLVM IR）
│   ├── codegen_llvm.h
│   ├── codegen_llvm.cc
│   └── llvm_module.cc
├── source/            # 源代码生成
│   ├── codegen_c.h
│   ├── codegen_c.cc
│   ├── codegen_cuda.h
│   ├── codegen_cuda.cc
│   ├── codegen_opencl.h
│   ├── codegen_opencl.cc
│   └── codegen_vulkan.cc
├── spirv/             # Vulkan SPIR-V
│   ├── codegen_spirv.h
│   └── codegen_spirv.cc
├── wasm/              # WebAssembly
│   └── codegen_wasm.cc
└── rocm/              # AMD GPU (ROCm)
    └── codegen_rocm.cc
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# TVM 支持的目标 target 示例
import tvm
from tvm import relay

# CPU (LLVM)
target_cpu = tvm.target.Target("llvm -mcpu=skylake")

# NVIDIA GPU (CUDA)
target_cuda = tvm.target.Target("cuda")

# AMD GPU (ROCm)
target_rocm = tvm.target.Target("rocm")

# Vulkan (跨平台 GPU)
target_vulkan = tvm.target.Target("vulkan")

# OpenCL (跨平台 GPU)
target_opencl = tvm.target.Target("opencl")

# WebAssembly (浏览器)
target_wasm = tvm.target.Target("wasm")

# ARM CPU (嵌入式)
target_arm = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")

# RISC-V (MCU)
target_riscv = tvm.target.Target("llvm -mtriple=riscv32-unknown-elf")

# VTA (FPGA 加速器)
target_vta = tvm.target.Target("ext_dev", host="llvm")
```

### 36.5.2 XLA 的硬件后端

XLA 的后端主要集中在 Google 自家硬件：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
tensorflow/compiler/xla/service/
├── cpu/               # CPU 后端
│   ├── cpu_compiler.cc
│   ├── cpu_executable.cc
│   └── ir_emitter.cc
├── gpu/               # NVIDIA GPU (CUDA) / AMD GPU (ROCm)
│   ├── gpu_compiler.cc
│   ├── gpu_executable.cc
│   ├── ir_emitter_unnested.cc
│   └── cudnn_fused_conv_rewriter.cc
└── tpu/               # Google TPU
    ├── tpu_compiler.cc
    ├── tpu_executable.cc
    └── tpu_ir_emitter.cc
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# XLA 的硬件后端使用方式
# 通过 JAX 指定后端
import jax
import jax.numpy as jnp

# 默认使用 GPU（如果可用）
x = jnp.ones((128, 128))

# 使用 TPU
# 需要在 TPU 环境中运行
# jax.devices() 会返回 TPU 设备

# 使用 CPU
with jax.default_device(jax.devices("cpu")[0]):
    result = jnp.dot(x, x)

# XLA 编译到 GPU 时的关键优化 Pass
# tensorflow/compiler/xla/service/gpu/
# - cudnn_fused_conv_rewriter.cc: 利用 cuDNN 的融合卷积
# - gpu_hlo_schedule.cc: HLO 指令的调度优化
# - gemm_rewriter.cc: 矩阵乘法优化（使用 cuBLAS）
# - convolution_thunk.cc: 卷积 kernel 选择
```

### 36.5.3 MLIR 的硬件后端

MLIR 通过 Dialect 系统支持硬件扩展：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
mlir/include/mlir/Dialect/
├── GPU/               # GPU 抽象层
│   └── IR/GPUDialect.td
├── SPIRV/             # Vulkan SPIR-V
│   └── IR/SPIRVOps.td
├── NVVM/              # NVIDIA GPU (NVVM/CUDA)
│   └── IR/NVVMDialect.td
├── ROCDL/             # AMD GPU (ROCm)
│   └── IR/ROCDLOps.td
├── LLVM/              # CPU (LLVM IR)
│   └── IR/LLVMDialect.td
├── AMX/               # Intel AMX 指令
│   └── IR/AMXDialect.td
├── X86Vector/         # x86 SIMD 指令
│   └── IR/X86VectorOps.td
└── ArmNeon/           # ARM NEON 指令
    └── IR/ArmNeon.td
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```mlir
// MLIR GPU dialect 示例
func.func @gpu_example(%arg0: memref<128x128xf32>,
                        %arg1: memref<128x128xf32>,
                        %arg2: memref<128x128xf32>) {
  // 在 GPU 上启动 kernel
  gpu.launch_func @kernels::@matmul
      blocks in (%c1, %c1, %c1) threads in (%c128, %c128, %c1)
      args(%arg0 : memref<128x128xf32>,
           %arg1 : memref<128x128xf32>,
           %arg2 : memref<128x128xf32>)
  return
}

gpu.module @kernels {
  gpu.func @matmul(%arg0: memref<128x128xf32>,
                    %arg1: memref<128x128xf32>,
                    %arg2: memref<128x128xf32>)
      workgroup(%shared : memref<32x32xf32, 3>)
      private(%private : memref<4x4xf32, 5>) {
    // GPU kernel 实现
    %tidx = gpu.thread_id x
    %tidy = gpu.thread_id y
    // ... 计算逻辑
    gpu.return
  }
}
```

### 36.5.4 硬件覆盖范围对比表

| 硬件后端 | TVM | XLA | MLIR |
|---------|-----|-----|------|
| **x86 CPU** | LLVM | 自有 CPU backend | LLVM dialect |
| **ARM CPU** | LLVM | 部分支持 | LLVM dialect |
| **NVIDIA GPU** | CUDA CodeGen | GPU backend | NVVM dialect |
| **AMD GPU** | ROCm CodeGen | GPU backend | ROCDL dialect |
| **Google TPU** | 无 | TPU backend (primary) | 通过 XLA |
| **Vulkan** | SPIR-V CodeGen | 无 | SPIRV dialect |
| **OpenCL** | OpenCL CodeGen | 无 | 需扩展 |
| **WebAssembly** | WASM CodeGen | 无 | 需扩展 |
| **FPGA** | VTA | 无 | 需扩展 |
| **MCU** | microTVM | 无 | 需扩展 |
| **Apple Metal** | 部分支持 | 无 | 需扩展 |
| **Intel GPU** | 部分支持 | 无 | oneAPI dialect |



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
硬件支持广度：

TVM:  ████████████████████  （最广泛，覆盖从 MCU 到 FPGA）
MLIR: ██████████████████    （通过 Dialect 扩展，潜力大）
XLA:  ██████████            （TPU 深度优化，其他硬件覆盖有限）
```

---

## 36.6 动态形状支持对比

### 36.6.1 TVM 的动态形状支持

TVM 通过多种机制支持动态形状：

**Relay 中的动态形状（Any 类型）：**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# python/tvm/relay/frontend/
# 导入模型时处理动态维度
import tvm
from tvm import relay

# 使用 "Any" 表示动态维度
data = relay.var("data", shape=(relay.Any(), 3, 224, 224), dtype="float32")
weight = relay.var("weight", shape=(64, 3, 7, 7), dtype="float32")

# 动态 batch 的卷积
conv = relay.nn.conv2d(data, weight, strides=(2, 2), padding=(3, 3))

# Relay 的 shape 函数推断
# src/relay/transforms/dynamic_to_static.cc
# DynamicToStatic pass 尝试将动态形状转换为静态
```

**Relax 中的符号形状（Symbolic Shape）：**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Relax 中原生支持符号形状
from tvm import relax

# 使用符号变量表示动态维度
@tvm.script.ir_module
class DynamicModel:
    @R.function
    def main(
        x: R.Tensor(("batch", "seq_len", 768), dtype="float32"),
        w: R.Tensor((768, 3072), dtype="float32")
    ) -> R.Tensor(("batch", "seq_len", 3072), dtype="float32"):
        # 符号形状在计算图中自动传播
        lv0 = R.matmul(x, w)
        lv1 = R.nn.gelu(lv0)
        return lv1
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# tvm/src/relax/backend/vm/vm_shape_lower.cc
# Relax VM 的形状推断机制
# 符号形状在运行时通过 ShapeTuple 传递

# 使用 Relax 编译动态形状模型
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_fx

# 从 PyTorch 模型导入（支持动态形状）
# python/tvm/relax/frontend/torch/
# 参考源码中的 dynamic shape 处理逻辑
```

### 36.6.2 XLA 的动态形状支持

XLA 最初设计为纯静态形状编译器，后来逐步增加了动态形状支持。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# XLA 的动态维度通过 DynamicDimension 表示
# tensorflow/compiler/xla/service/dynamic_dimension_inference.cc

# 使用 JAX 的动态形状
import jax
import jax.numpy as jnp

# JAX 中使用 jax.jit 处理不同形状的输入
@jax.jit
def dynamic_fn(x):
    # x 的形状可以是任意的
    return jnp.sum(x, axis=-1)

# 不同形状的输入会触发 XLA 的 recompilation
a = jnp.ones((128,))
b = jnp.ones((256,))
# 这两次调用可能触发两次 XLA 编译
print(dynamic_fn(a))
print(dynamic_fn(b))
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// tensorflow/compiler/xla/service/dynamic_dimension_inference.cc
// 动态维度推断 Pass

Status DynamicDimensionInference::InferDynamicDimensions(
    HloModule* module) {
  // 遍历所有 HLO 指令
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      switch (instruction->opcode()) {
        case HloOpcode::kReshape:
          // 动态 reshape 需要推断输出的动态维度
          TF_RETURN_IF_ERROR(InferDynamicReshape(instruction));
          break;
        case HloOpcode::kDynamicSlice:
          // DynamicSlice 的大小参数可能是动态的
          TF_RETURN_IF_ERROR(InferDynamicSlice(instruction));
          break;
        case HloOpcode::kDynamicUpdateSlice:
          TF_RETURN_IF_ERROR(InferDynamicUpdateSlice(instruction));
          break;
        // ...
      }
    }
  }
  return OkStatus();
}
```

### 36.6.3 MLIR 的动态形状支持

MLIR 在类型系统层面原生支持动态维度。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```mlir
// MLIR 的 ShapedType 支持动态维度
// mlir/include/mlir/IR/OpBase.td 中定义了 kDynamic

// 静态形状
%static = tensor<128x256xf32>

// 动态形状：使用 ? 表示动态维度
%dynamic_batch = tensor<?x256xf32>
%dynamic_both = tensor<?x?xf32>

// 混合静态/动态形状
%mixed = tensor<128x?xf32>

// 在函数签名中使用动态形状
func.func @dynamic_matmul(
    %A: tensor<?x256xf32>,    // batch 维度动态
    %B: tensor<256x512xf32>   // 完全静态
) -> tensor<?x512xf32> {
  // linalg 操作自动支持动态形状
  %C = tensor.empty : tensor<?x512xf32>
  %result = linalg.matmul ins(%A, %B : tensor<?x256xf32>, tensor<256x512xf32>)
                          outs(%C : tensor<?x512xf32>) -> tensor<?x512xf32>
  return %result : tensor<?x512xf32>
}
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// mlir/include/mlir/IR/BuiltinTypes.h
// ShapedType 中动态维度的定义

class ShapedType : public Type {
 public:
  // 动态维度的哨兵值
  static constexpr int64_t kDynamic = std::numeric_limits<int64_t>::min();

  // 检查某个维度是否是动态的
  bool isDynamicDim(unsigned dim) const {
    return getShape()[dim] == kDynamic;
  }

  // 获取动态维度的数量
  int64_t getNumDynamicDims() const {
    return llvm::count(getShape(), kDynamic);
  }

  // 判断类型是否包含动态维度
  bool hasStaticShape() const {
    return getNumDynamicDims() == 0;
  }
};
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```mlir
// MLIR 中使用 tensor.dim 获取动态维度
func.func @compute_with_dynamic(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // 获取动态维度
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %input, %c0 : tensor<?x?xf32>  // batch size
  %dim1 = tensor.dim %input, %c1 : tensor<?x?xf32>  // feature dim

  // 使用动态维度创建输出张量
  %output = tensor.empty [%dim0, %dim1] : tensor<?x?xf32>

  // linalg 操作在动态形状上正常工作
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<?x?xf32>)
    outs(%output : tensor<?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %cst = arith.constant 0.0 : f32
        %relu = arith.maximumf %in, %cst : f32
        linalg.yield %relu : f32
    } -> tensor<?x?xf32>

  return %result : tensor<?x?xf32>
}
```

### 36.6.4 动态形状支持对比

| 维度 | TVM | XLA | MLIR |
|------|-----|-----|------|
| **类型支持** | Any (Relay) / Symbolic (Relax) | DynamicDimension | ShapedType::kDynamic |
| **编译策略** | 参数化 kernel / 多版本 | Recompilation / Dynamic Dim | 参数化代码 |
| **形状推断** | ShapeFunc | DynamicDimensionInference | tensor.dim |
| **内存规划** | 运行时分配 | 运行时分配 | 运行时分配 |
| **优化影响** | 部分优化失效 | 部分优化失效 | 部分优化失效 |

---

## 36.7 前端框架支持对比

### 36.7.1 TVM 的前端支持

TVM 支持最广泛的前端框架导入：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
python/tvm/relay/frontend/
├── pytorch.py         # PyTorch 模型导入
├── tensorflow.py      # TensorFlow 模型导入
├── onnx.py            # ONNX 模型导入
├── keras.py           # Keras 模型导入
├── tflite.py          # TensorFlow Lite 模型导入
├── caffe.py           # Caffe 模型导入
├── coreml.py          # Core ML 模型导入
├── darknet.py         # DarkNet (YOLO) 模型导入
├── paddlepaddle.py    # PaddlePaddle 模型导入
├── mxnet.py           # Apache MXNet 模型导入
└── from_paddle.py     # PaddlePaddle 新版导入
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# TVM 导入 PyTorch 模型的示例
# python/tvm/relay/frontend/pytorch.py
import tvm
from tvm import relay
import torch
import torchvision

# 加载预训练的 ResNet-18
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# 创建示例输入
input_shape = (1, 3, 224, 224)
input_data = torch.randn(input_shape)

# 通过 PyTorch 的 trace 导出计算图
trace = torch.jit.trace(model, input_data)

# 使用 TVM 的 PyTorch frontend 导入
# 核心函数：from_pytorch() in python/tvm/relay/frontend/pytorch.py
mod, params = relay.frontend.from_pytorch(
    trace,
    input_info=[("input", input_shape)],
    default_dtype="float32"
)

# 导入后可以查看 Relay IR
print(mod)
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# TVM 导入 ONNX 模型
# python/tvm/relay/frontend/onnx.py
import onnx

# 加载 ONNX 模型
onnx_model = onnx.load("resnet18.onnx")

# 使用 TVM 的 ONNX frontend 导入
# 核心函数：from_onnx() in python/tvm/relay/frontend/onnx.py
mod, params = relay.frontend.from_onnx(
    onnx_model,
    shape={"input": (1, 3, 224, 224)},
    dtype="float32"
)
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# TVM 导入 TensorFlow 模型
# python/tvm/relay/frontend/tensorflow.py
import tensorflow as tf

# 加载 SavedModel
model = tf.saved_model.load("resnet18_saved_model")

# 使用 TVM 的 TensorFlow frontend 导入
# 核心函数：from_tensorflow() in python/tvm/relay/frontend/tensorflow.py
mod, params = relay.frontend.from_tensorflow(
    model,
    layout="NCHW",
    outputs=None  # 自动检测输出节点
)
```

### 36.7.2 XLA 的前端支持

XLA 主要服务 TensorFlow 和 JAX 生态：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# XLA 的前端架构
# tensorflow/compiler/xla/client/
# ├── xla_builder.h      # 构建 HLO 计算图的 API
# ├── xla_computation.h  # 编译后的计算
# └── local_client.h     # 本地执行客户端

# JAX 是 XLA 的最自然前端
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

# JAX 函数通过 jit 编译到 XLA
@jit
def predict(params, x):
    for w, b in params:
        x = jnp.tanh(jnp.dot(x, w) + b)
    return x

# jit 装饰器触发 XLA 编译
# 1. JAX 将 Python 函数 trace 为 jaxpr
# 2. jaxpr 转换为 XLA HLO
# 3. XLA 对 HLO 进行优化
# 4. 生成目标硬件代码

# 查看 HLO IR
params = [(jnp.ones((784, 256)), jnp.ones(256)),
          (jnp.ones((256, 10)), jnp.ones(10))]
x = jnp.ones((1, 784))

compiled = jit(predict).lower(params, x)
print(compiled.compiler_ir(dialect="hlo"))
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# PyTorch 通过 torch_xla 使用 XLA
# torch_xla 是 PyTorch 的 XLA 后端
import torch
import torch_xla
import torch_xla.core.xla_model as xm

# 创建 XLA 设备
device = xm.xla_device()

# 模型和数据放在 XLA 设备上
model = torch.nn.Linear(784, 10).to(device)
x = torch.randn(32, 784).to(device)

# 前向传播自动通过 XLA 编译
output = model(x)

# 需要手动标记 step，触发 XLA 编译和执行
xm.mark_step()
```

### 36.7.3 MLIR 的前端支持

MLIR 通过各种 Dialect 支持不同前端：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
# MLIR 的前端支持
# tensorflow/compiler/mlir/
├── lite/              # TensorFlow Lite → MLIR
│   ├── tfl_to_tosa.cc
│   └── tfl_to_mlir.cc
├── tfx/               # TensorFlow Extended → MLIR
│   └── ...
├── torch-mlir/        # PyTorch → MLIR（社区项目）
│   ├── python/torch_mlir/
│   │   ├── compiler.py
│   │   └── irbuilder.py
│   └── lib/Conversion/TorchToTosa/
│       └── TorchToTosa.cpp
└── onnx-mlir/         # ONNX → MLIR（社区项目）
    └── src/Conversion/ONNXToTOSA/
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# PyTorch → MLIR (通过 torch-mlir)
# 社区项目：https://github.com/llvm/torch-mlir
import torch
import torch_mlir

# 创建 PyTorch 模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(784, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

model = SimpleModel()
example_input = torch.randn(1, 784)

# 通过 torch-mlir 编译到 MLIR
mlir_module = torch_mlir.compile(
    model,
    example_input,
    output_type=torch_mlir.OutputType.TOSA,  # 输出 TOSA dialect
)

print(mlir_module)
# 输出：
# module attributes {torch.debug_module_name = "SimpleModel"} {
#   func.func @forward(%arg0: tensor<1x784xf32>) -> tensor<1x10xf32> {
#     %0 = "tosa.const"() {value = dense<...> : tensor<10x784xf32>} : () -> tensor<10x784xf32>
#     %1 = "tosa.fully_connected"(%arg0, %0, ...) : ...
#     %2 = "tosa.clamp"(%1, ...) : ...
#     return %2 : tensor<1x10xf32>
#   }
# }
```

### 36.7.4 前端支持对比

| 框架 | TVM | XLA | MLIR |
|------|-----|-----|------|
| **PyTorch** | relay.frontend.from_pytorch | torch_xla | torch-mlir |
| **TensorFlow** | relay.frontend.from_tensorflow | 原生支持 | tf-mlir |
| **JAX** | relay.frontend.from_jax | 原生支持 | jax-mlir |
| **ONNX** | relay.frontend.from_onnx | 需转换 | onnx-mlir |
| **TFLite** | relay.frontend.from_tflite | 部分支持 | tfl-to-tosa |
| **Keras** | relay.frontend.from_keras | 原生支持 | tf-mlir |
| **PaddlePaddle** | relay.frontend.from_paddle | 无 | 无 |
| **CoreML** | relay.frontend.from_coreml | 无 | 无 |

---

## 36.8 社区生态与商业模式

### 36.8.1 TVM 的社区治理

TVM 是 Apache 软件基金会的孵化项目，采用完全开放的社区治理模式。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
TVM 社区组织架构：

┌─────────────────────────────────────────────────┐
│              Apache TVM PMC                      │
│  (Project Management Committee)                  │
│  - 决策重大项目方向                                │
│  - 审批 Committer 提名                            │
├─────────────────────────────────────────────────┤
│              Committers                          │
│  - 代码审查与合并                                  │
│  - 来自多家公司和学术机构                          │
├─────────────────────────────────────────────────┤
│              Contributors                        │
│  - 提交 PR、报告 Issue                            │
│  - 开放社区，任何人可参与                          │
└─────────────────────────────────────────────────┘

主要贡献公司：
- OctoML（商业化公司）
- 华为（昇腾生态）
- AMD（ROCm 支持）
- Intel（oneAPI 集成）
- 阿里巴巴（移动端优化）
- 字节跳动（大规模部署）
```

### 36.8.2 XLA 的治理模式

XLA 是 Google 主导的项目，虽然开源但决策权集中在 Google。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
XLA 的治理架构：

┌─────────────────────────────────────────────────┐
│              Google TPU Team                     │
│  - XLA 的核心开发团队                              │
│  - TPU 后端的主要维护者                            │
├─────────────────────────────────────────────────┤
│              OpenXLA Foundation (2023)           │
│  - AMD、Intel、NVIDIA 等参与                      │
│  - 推动 XLA 的开放治理                            │
├─────────────────────────────────────────────────┤
│              社区贡献者                            │
│  - GPU 后端优化                                   │
│  - 新硬件支持                                    │
└─────────────────────────────────────────────────┘

XLA 的使用方：
- Google（内部 TPU 集群）
- Google Cloud TPU 用户
- JAX 用户社区
- PyTorch/TPU 用户（通过 torch_xla）
```

### 36.8.3 MLIR 的社区生态

MLIR 是 LLVM 社区的一部分，拥有广泛的行业支持。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
MLIR 的行业采纳：

┌─────────────────────────────────────────────────┐
│              LLVM Foundation                     │
│  - MLIR 作为 LLVM 项目的官方子项目                 │
│  - 继承 LLVM 的开放治理模式                       │
├─────────────────────────────────────────────────┤
│              主要贡献方                            │
│  - Google：发起者，主导 Dialect 设计              │
│  - Intel：oneAPI/DPC++ 编译栈                    │
│  - AMD：ROCm 编译栈                              │
│  - Qualcomm：移动端 AI 编译器                     │
│  - Samsung：Exynos NPU 编译器                    │
│  - Cerebras：Wafer-Scale 编译器                  │
│  - SiFive：RISC-V 编译器                         │
├─────────────────────────────────────────────────┤
│              基于 MLIR 的项目                      │
│  - IREE：端到端 ML 编译器运行时                    │
│  - Triton：GPU kernel 编译器                     │
│  - npcomp / torch-mlir：PyTorch 编译             │
│  - XLA：HLO 基于 MLIR Dialect 重写               │
└─────────────────────────────────────────────────┘
```

### 36.8.4 生态对比

| 维度 | TVM | XLA | MLIR |
|------|-----|-----|------|
| **治理模型** | Apache PMC（开放） | Google/OpenXLA（半开放） | LLVM Foundation（开放） |
| **主要贡献方** | 多家公司+学术界 | Google 为主 | Google + 多家公司 |
| **商业化** | OctoML 等 | Google Cloud | 无直接商业公司 |
| **学术影响** | 高（OSDI/MLSys 论文） | 中（Google 内部） | 高（PLDI/CGO 论文） |
| **工业采用** | 中等（多家公司） | 广泛（Google 内部） | 广泛（多家公司） |
| **社区活跃度** | 高 | 中 | 高 |

---

## 36.9 综合对比表格与适用场景推荐

### 36.9.1 全维度对比表

| 对比维度 | TVM | XLA | MLIR |
|---------|-----|-----|------|
| **核心定位** | 端到端 DL 编译器 | DL 编译器 (TPU 优先) | 通用编译器基础设施 |
| **设计哲学** | 搜索驱动 + 硬件无关 | 规则驱动 + TPU 深度优化 | Dialect 系统 + 渐进 lowering |
| **IR 层次** | Relay → TE → TIR | HLO (单一) | N 层 Dialect 链 |
| **优化策略** | AutoTVM / MetaSchedule | 手写 Pass Pipeline | 声明式 Pattern |
| **硬件覆盖** | CPU/GPU/FPGA/MCU/TPU | TPU/GPU/CPU | 通过 Dialect 无限扩展 |
| **动态形状** | Any / Symbolic | DynamicDimension | ShapedType::kDynamic |
| **前端支持** | PyTorch/TF/ONNX/... | TF/JAX/PyTorch(xla) | 通过 Dialect/工具 |
| **编译速度** | 慢（搜索） | 快（规则） | 中等 |
| **优化质量** | 接近最优 | 依赖规则 | 依赖 Pattern |
| **学习曲线** | 中等 | 较低 | 较高 |
| **可扩展性** | 中等（需实现 CodeGen） | 较低（需改 backend） | 高（新增 Dialect） |
| **社区活跃度** | 高 | 中 | 高 |
| **治理模型** | Apache 开放 | Google/OpenXLA | LLVM 开放 |

### 36.9.2 选型决策流程



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
                          ┌─────────────────┐
                          │  你的需求是什么？ │
                          └────────┬────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ↓              ↓              ↓
            ┌──────────┐  ┌──────────┐  ┌──────────┐
            │ 部署模型  │  │ 构建编译器│  │ 研究优化  │
            │ 到生产环境│  │ 工具链    │  │ 算法      │
            └────┬─────┘  └────┬─────┘  └────┬─────┘
                 │             │             │
         ┌───────┴───────┐    │      ┌──────┴──────┐
         ↓               ↓    ↓      ↓             ↓
    ┌─────────┐    ┌─────────┐ ┌─────────┐  ┌─────────┐
    │ 目标硬件 │    │ 需要    │ │ MLIR    │  │ TVM     │
    │ 是什么？ │    │ 最大灵活│ │ (构建新 │  │ (搜索   │
    │         │    │ 性？    │ │  Dialect)│  │  空间)  │
    └────┬────┘    └────┬────┘ └─────────┘  └─────────┘
         │              │
    ┌────┴────┐    ┌────┴────┐
    ↓         ↓    ↓         ↓
┌───────┐ ┌───────┐ ┌───────┐
│Google │ │其他   │ │是     │
│TPU/GPU│ │硬件   │ │       │
└───┬───┘ └───┬───┘ └───┬───┘
    ↓         ↓         ↓
┌───────┐ ┌───────┐ ┌───────┐
│XLA/JAX│ │TVM    │ │MLIR   │
│       │ │       │ │       │
└───────┘ └───────┘ └───────┘
```

### 36.9.3 场景化推荐

**场景一：学术研究**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
需求：快速实验新的优化算法，需要灵活的搜索空间
推荐：TVM (MetaSchedule)
理由：
- 搜索空间定义灵活
- 社区活跃，论文引用方便
- Python API 友好，快速原型开发
```

**场景二：Google Cloud TPU 生产部署**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
需求：在 TPU 集群上大规模部署模型
推荐：XLA (通过 JAX)
理由：
- TPU 的深度优化
- JAX + XLA 是 TPU 的原生方案
- Google 内部大规模验证
```

**场景三：边缘设备部署**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
需求：将模型部署到 ARM MCU 或 FPGA
推荐：TVM (microTVM / VTA)
理由：
- microTVM 支持 MCU 后端
- VTA 支持 FPGA 加速
- 模型量化和优化工具成熟
```

**场景四：构建新的 AI 加速器编译器**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
需求：为自研 NPU/AI 芯片构建编译器工具链
推荐：MLIR
理由：
- Dialect 系统可定制硬件特定 IR
- 渐进式 lowering 适合复杂的编译流程
- LLVM 生态的工具链支持
- 行业共识：越来越多芯片公司选择 MLIR
```

**场景五：多硬件推理平台**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
需求：一个推理引擎支持多种硬件
推荐：TVM + MLIR 混合
理由：
- TVM 的硬件覆盖最广
- MLIR 的 Dialect 可以扩展到新硬件
- IREE 项目结合了两者的优势
```

### 36.9.4 性能基准参考

以下是不同场景下的典型性能表现（相对值）：

| 场景 | TVM | XLA | 手写 CUDA |
|------|-----|-----|----------|
| **ResNet-50 (GPU)** | 95% | 98% | 100% |
| **BERT-Large (TPU)** | N/A | 100% | N/A |
| **MobileNet (ARM)** | 98% | 90% | 100% |
| **Transformer (A100)** | 96% | 97% | 100% |
| **自定义算子** | 85% | 80% | 100% |



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
性能接近度（相对于手写 CUDA/汇编）：

GPU 云推理：
  XLA:  ████████████████████  98%
  TVM:  ███████████████████   95%

TPU 推理：
  XLA:  ████████████████████  100%
  TVM:  N/A

边缘设备：
  TVM:  ████████████████████  98%
  XLA:  ██████████████████    90%

新硬件适配：
  TVM:  █████████████████     85%
  XLA:  ████████████████      80%
```

### 36.9.5 未来趋势



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
┌─────────────────────────────────────────────────────────────────┐
│                        发展趋势                                  │
│                                                                 │
│  1. MLIR 成为编译器基础设施的事实标准                              │
│     - XLA 的 HLO 正在迁移为 MLIR Dialect (StableHLO)            │
│     - TVM Unity 的底层也在借鉴 MLIR 的设计思想                    │
│     - 新的 AI 芯片编译器几乎都基于 MLIR                           │
│                                                                 │
│  2. 搜索驱动与规则驱动的融合                                      │
│     - XLA 开始引入 AutoTuning                                    │
│     - TVM 在 MetaSchedule 中融合了规则和搜索                     │
│     - MLIR 的 transform dialect 支持混合策略                     │
│                                                                 │
│  3. 动态形状成为标配                                              │
│     - LLM 推理的序列长度天然是动态的                              │
│     - 三大编译器都在加强动态形状支持                               │
│                                                                 │
│  4. 编译器即服务（Compiler-as-a-Service）                         │
│     - TVM 的 RPC 机制                                            │
│     - XLA 的 AOT 编译                                            │
│     - MLIR 的增量编译能力                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 36.99 文字内容强化：TVM 与 XLA/MLIR 对比 的工程化理解

这一节用于把前文的 API、IR、Pass、Runtime 和部署片段串联为更完整的工程叙事。
很多学习者第一次阅读 TVM 文档时会觉得示例代码很多，但真正上线时仍然不知道如何判断方案是否可靠。
原因在于 TVM 不是单个推理库，而是一条从模型语义到硬件代码的编译链路。
链路越长，越需要把每一步的业务目标、内部机制、适用边界和失败模式说清楚。

### 36.99.1 代码解读的阅读方法

1. 阅读本章代码时，首先要判断它是在构建模型、转换 IR、配置 target、选择 executor、执行 tuning，还是加载 runtime artifact。
2. 如果片段位于前端导入阶段，重点不是性能，而是语义是否完整保留下来。
3. 如果片段位于 Relay、Relax 或 TIR 优化阶段，重点是 pass 是否改变了张量形状、数据类型、布局和算子边界。
4. 如果片段位于运行时阶段，重点是参数、输入、设备上下文和编译产物是否一一对应。
5. 不要把示例中的单个函数调用理解成黑盒魔法，它通常只是封装了多层 IR 变换。
6. 调试时应把模型切成前端、IR、调度、代码生成和运行时五个观察面。
7. 性能分析时应把端到端耗时、单算子耗时、数据搬运耗时和编译耗时分开记录。
8. 数值验证时应同时比较最大误差、平均误差、相对误差和业务指标。
9. 部署验证时应记录 TVM 版本、LLVM 版本、Python 包版本、目标硬件型号和驱动版本。
10. 当代码可以运行但结果不稳定时，优先怀疑输入预处理、随机数、线程调度和未固定的编译参数。

- 围绕“搜索驱动优化与规则驱动优化的长期差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“多级 IR、方言生态和硬件接入方式的不同”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“模型部署、编译缓存和动态形状支持的工程差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“企业选型、团队能力和维护成本的判断标准”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 36.99.2 业务意义

1. TVM 与 XLA/MLIR 对比 的业务价值不只是让模型跑得更快，而是让同一个模型可以在不同成本、功耗和延迟约束下交付。
2. 在服务器场景中，核心指标通常是吞吐、P95/P99 延迟、资源利用率和多租户隔离。
3. 在移动端场景中，核心指标通常是首帧时间、持续发热、内存峰值和包体大小。
4. 在嵌入式场景中，核心指标通常是 Flash 占用、静态内存、实时性和掉电恢复能力。
5. 在云端批处理场景中，编译时间可以接受更长，但调优记录和缓存复用变得非常重要。
6. 在在线服务场景中，编译产物需要可回滚、可审计、可灰度，而不能只在开发机上验证。
7. 业务方关心的是 SLA、成本和稳定性，编译器工程师关心的是 IR 正确性、优化空间和后端能力。
8. 优秀的 TVM 项目需要把这两类语言翻译成共同的指标体系。
9. 当优化收益只有少量百分点时，应评估它是否值得引入新的维护复杂度。
10. 当优化收益很大但只在少数输入上成立时，应评估输入分布变化后的风险。

- 围绕“搜索驱动优化与规则驱动优化的长期差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“多级 IR、方言生态和硬件接入方式的不同”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“模型部署、编译缓存和动态形状支持的工程差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“企业选型、团队能力和维护成本的判断标准”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 36.99.3 TVM 内部机制

1. TVM 的核心机制是分层表示和逐层降低，高层 IR 保留模型语义，低层 IR 暴露循环、内存和并行结构。
2. Relay 更适合表达静态计算图和传统深度学习算子。
3. Relax 更强调动态形状、函数式组合和跨层优化。
4. TIR 是最终性能调优的关键，因为它决定循环嵌套、内存作用域、向量化和线程映射。
5. PassContext 会影响优化级别、禁用或启用的 pass，以及某些后端特定配置。
6. Target 不只是字符串，它包含硬件特征、指令集、运行时约定和 codegen 选择。
7. Executor 决定模型入口如何组织，Runtime 决定编译产物如何加载和调用。
8. 参数绑定会影响常量折叠、内存规划和代码体积。
9. Layout rewrite 会影响算子融合和后端库调用。
10. AutoTVM 与 MetaSchedule 的调优记录会把搜索结果沉淀为可复用资产。

- 围绕“搜索驱动优化与规则驱动优化的长期差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“多级 IR、方言生态和硬件接入方式的不同”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“模型部署、编译缓存和动态形状支持的工程差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“企业选型、团队能力和维护成本的判断标准”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 36.99.4 适用场景

1. 当模型结构相对稳定、目标硬件明确、性能收益可以通过基准测试确认时，TVM 与 XLA/MLIR 对比 相关技术最容易发挥价值。
2. 当团队需要支持多种硬件后端时，TVM 的统一 IR 和 Target 抽象可以降低重复适配成本。
3. 当模型中存在框架运行时开销、算子融合机会或布局转换冗余时，编译优化通常能带来明显收益。
4. 当部署环境不能依赖完整 Python 栈时，AOT、CRT 或导出后的 runtime artifact 更有意义。
5. 当硬件厂商提供高性能库但模型图需要复杂切分时，BYOC 和外部 codegen 是常见选择。
6. 当输入形状变化频繁时，应提前设计 shape 策略，而不是在上线前才补动态形状支持。
7. 当模型版本迭代频繁时，应把编译、调优、验证和发布纳入 CI/CD。
8. 当业务对精度非常敏感时，应把优化收益和数值回归一起评估。
9. 当系统存在多模型串联时，应评估端到端 pipeline，而不是只优化单个模型。
10. 当部署设备数量很大时，编译产物的一致性和可追踪性比单次实验性能更重要。

- 围绕“搜索驱动优化与规则驱动优化的长期差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“多级 IR、方言生态和硬件接入方式的不同”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“模型部署、编译缓存和动态形状支持的工程差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“企业选型、团队能力和维护成本的判断标准”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 36.99.5 限制条件

1. TVM 并不能自动解决所有性能问题，尤其不能替代对模型结构、硬件层次和数据分布的理解。
2. 如果前端导入阶段已经丢失语义，后续 pass 很难恢复原始意图。
3. 如果目标硬件缺少成熟 codegen 或 runtime 支持，理论上的 IR 优化收益可能无法兑现。
4. 如果模型包含大量小算子，调度优化可能被调用开销和内存同步抵消。
5. 如果输入分布与校准集或 benchmark 数据差异较大，性能和精度结论都可能失真。
6. 如果编译参数没有固定，调优结果难以复现。
7. 如果只看平均延迟，可能忽略 P99 抖动和内存峰值。
8. 如果只在开发机验证，可能忽略生产驱动、内核版本和设备温控策略。
9. 如果项目缺少 IR dump 和 artifact 存档，出现线上问题时很难追溯。
10. 如果团队没有维护编译工具链的能力，过度定制会形成长期负担。

- 围绕“搜索驱动优化与规则驱动优化的长期差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“多级 IR、方言生态和硬件接入方式的不同”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“模型部署、编译缓存和动态形状支持的工程差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“企业选型、团队能力和维护成本的判断标准”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 36.99.6 工程经验

1. 第一条经验是先建立可靠 baseline，再做任何复杂优化。
2. baseline 应包含原框架结果、TVM 未调优结果、TVM 调优结果和目标平台参考结果。
3. 第二条经验是每次只改变一个变量，例如 target、layout、executor、batch 或调优记录。
4. 第三条经验是把编译日志、IR dump、调优数据库和运行时产物一起归档。
5. 第四条经验是为每个模型维护最小可复现输入，便于定位前端导入和数值误差。
6. 第五条经验是把 shape、dtype、layout 写入模型契约，而不是散落在脚本里。
7. 第六条经验是使用真实业务样本做最终验证，因为随机输入无法覆盖预处理和分布偏移。
8. 第七条经验是把冷启动和热启动分开统计。
9. 第八条经验是为不同硬件维护独立调优记录，不要假设一个 schedule 可以跨平台复用。
10. 第九条经验是对编译产物做哈希和版本标记，方便灰度和回滚。
11. 第十条经验是让模型工程师、系统工程师和硬件工程师共享同一份性能报告。

- 围绕“搜索驱动优化与规则驱动优化的长期差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“多级 IR、方言生态和硬件接入方式的不同”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“模型部署、编译缓存和动态形状支持的工程差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“企业选型、团队能力和维护成本的判断标准”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 36.99.7 常见误区

1. 误区一是认为 TVM build 成功就等于模型可以生产上线。
2. 误区二是认为单次 benchmark 的最小值代表真实线上延迟。
3. 误区三是把调优时间算进推理收益，或者完全忽略调优成本。
4. 误区四是只关注算子融合数量，却不检查内存带宽和 cache 行为。
5. 误区五是看到 INT8 就默认更快，却忽略硬件是否有高效低精度指令。
6. 误区六是看到动态形状支持就默认所有输入尺寸都高效。
7. 误区七是把外部库调用当成万能方案，却忽略数据布局转换成本。
8. 误区八是把示例代码复制到生产环境，却没有补齐错误处理和 artifact 管理。
9. 误区九是只比较平均精度，不检查关键类别或长尾输入。
10. 误区十是把编译器问题、模型问题和硬件问题混在一起排查。

- 围绕“搜索驱动优化与规则驱动优化的长期差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“多级 IR、方言生态和硬件接入方式的不同”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“模型部署、编译缓存和动态形状支持的工程差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“企业选型、团队能力和维护成本的判断标准”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 36.99.8 生产部署注意事项

1. 生产部署前应冻结模型文件、输入预处理、编译配置和目标硬件描述。
2. 编译产物应包含可读元数据，例如模型版本、TVM commit、target、executor、runtime 和调优数据库版本。
3. 发布流程应支持灰度、回滚和双跑验证。
4. 线上监控应覆盖延迟、错误率、内存峰值、设备温度和业务指标。
5. 对于多线程 CPU 部署，应固定线程数并观察 NUMA、亲和性和其他服务的干扰。
6. 对于 GPU 部署，应区分 host 计时和 device 计时，并处理异步执行带来的误判。
7. 对于移动端部署，应关注长时间运行后的降频，而不是只看前几次推理。
8. 对于嵌入式部署，应把静态内存和栈空间作为硬约束。
9. 对于远程 RPC 测试，应把网络传输时间从设备执行时间中剥离。
10. 对于安全敏感业务，应限制模型产物和日志中的数据泄露风险。

- 围绕“搜索驱动优化与规则驱动优化的长期差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“多级 IR、方言生态和硬件接入方式的不同”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“模型部署、编译缓存和动态形状支持的工程差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“企业选型、团队能力和维护成本的判断标准”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 36.99.9 与同类系统对比

1. 与 TensorRT 相比，TVM 的硬件覆盖更开放，TensorRT 在 NVIDIA GPU 上的工程成熟度更高。
2. 与 XLA 相比，TVM 更强调可调度性和多后端扩展，XLA 更强调与框架图执行的深度集成。
3. 与 MLIR 相比，TVM 更像面向深度学习部署的完整编译器，MLIR 更像可构建编译器的基础设施。
4. 与 ONNX Runtime 相比，TVM 更关注提前编译和内核生成，ONNX Runtime 更强调运行时图优化和 execution provider 生态。
5. 与 Triton 相比，TVM 覆盖端到端模型编译，Triton 更适合手写或自动生成 GPU kernel。
6. 与 TFLite 相比，TVM 的后端扩展更灵活，TFLite 在移动端生态和模型格式上更标准化。
7. 与厂商 NPU SDK 相比，TVM 更中立，厂商 SDK 往往能访问更底层的私有能力。
8. 选择系统时不应只看峰值性能，还应看调试成本、团队经验、社区活跃度和长期维护风险。
9. 如果项目只部署到单一成熟硬件，专用推理引擎可能更省事。
10. 如果项目需要跨硬件、跨模型长期演进，TVM 的编译器化路线更有战略价值。

- 围绕“搜索驱动优化与规则驱动优化的长期差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“多级 IR、方言生态和硬件接入方式的不同”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“模型部署、编译缓存和动态形状支持的工程差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“企业选型、团队能力和维护成本的判断标准”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 36.99.10 章节复盘

1. 回到本章，TVM 与 XLA/MLIR 对比 的关键不是记住所有 API，而是理解为什么这些 API 会出现在编译链路的这个位置。
2. 当你看到一段代码时，应能说出它改变了模型语义、调度空间、内存布局、运行时入口还是部署产物。
3. 当你看到一个性能数字时，应能说出它的测试输入、硬件状态、计时方法和误差范围。
4. 当你看到一个优化 pass 时，应能说出它依赖的前置假设和可能破坏的边界条件。
5. 当你准备上线时，应能说出失败后如何回滚、如何复现、如何定位和如何与业务方沟通影响。
6. 这套思维比单个示例更重要，因为 TVM 的 API 会演进，但编译部署的工程约束长期稳定。
7. 后续学习中，可以把每一章都转化为一张决策表：何时使用、收益来自哪里、风险是什么、如何验证。
8. 只有把代码、机制和工程策略放在一起，TVM 才不只是工具箱，而是可运行的生产系统。
9. 因此，本章新增的文字说明应作为阅读代码段的上下文，而不是替代对原始代码的逐行理解。
10. 如果遇到与示例不一致的实际项目，应优先回到模型约束和目标硬件，而不是机械套用章节流程。

- 围绕“搜索驱动优化与规则驱动优化的长期差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“多级 IR、方言生态和硬件接入方式的不同”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“模型部署、编译缓存和动态形状支持的工程差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“企业选型、团队能力和维护成本的判断标准”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。


## 36.10 本章小结

### 36.10.1 核心差异回顾

本章深入对比了 TVM、XLA、MLIR 三大深度学习编译器生态。以下是关键差异的总结：

**架构层面**：TVM 采用固定的三层 IR（Relay → TE → TIR），XLA 以单一 HLO IR 为核心，MLIR 提供可扩展的 Dialect 系统。三者代表了编译器架构的三种不同哲学——固定层次、单一核心、可扩展层次。

**优化策略**：TVM 的搜索驱动方法（MetaSchedule）在理论上可以逼近最优，但需要实际运行测量；XLA 的规则驱动方法快速确定性，但依赖专家知识；MLIR 的声明式转换兼顾了灵活性和可维护性。

**硬件支持**：TVM 的硬件覆盖最广，从 MCU 到 FPGA 均有支持；XLA 在 TPU 上的优化最深；MLIR 通过 Dialect 系统提供了最大的可扩展性。

**工程实践**：选择哪个编译器取决于具体场景。学术研究偏好 TVM 的灵活性，Google Cloud 生态绑定 XLA/JAX，构建新硬件编译器则首选 MLIR。

### 36.10.2 一句话总结

> TVM 擅长 **"自动找到最优解"**，XLA 擅长 **"在 Google 硬件上高效执行"**，MLIR 擅长 **"构建任意编译器"**。三者不是替代关系，而是在不同维度上各有所长。

### 36.10.3 Pass Pipeline 对比

理解三者的优化 Pass Pipeline 有助于深入理解各自的优化策略：

**TVM 的 Relay Pass Pipeline：**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# src/relay/transforms/ 中定义的主要 Pass
# python/tvm/relay/transform.py
from tvm.relay import transform

# TVM 的典型 Pass Pipeline
relay_passes = tvm.transform.Sequential([
    # 1. 类型推断
    transform.InferType(),
    # 2. 常量折叠
    transform.FoldConstant(),
    # 3. 算子融合
    transform.FuseOps(fuse_opt_level=2),
    # 4. 部分求值
    transform.PartialEvaluate(),
    # 5. 去除无用代码
    transform.DeadCodeElimination(),
    # 6. 内存优化
    transform.ToANormalForm(),
    # 7. 布局变换
    transform.AlterOpLayout(),
], opt_level=3)
```

**XLA 的 Pass Pipeline：**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// tensorflow/compiler/xla/service/hlo_pass_pipeline.cc
// CPU/GPU 后端的典型 Pass 序列
void CpuCompiler::AddHloPasses(HloPassPipeline* pipeline) {
  // 1. 布局赋值
  pipeline->AddPass<LayoutAssignment>(&instruction_sequence);
  // 2. 代数化简
  pipeline->AddPass<AlgebraicSimplifier>(options);
  // 3. 常量折叠
  pipeline->AddPass<HloConstantFolding>();
  // 4. 公共子表达式消除
  pipeline->AddPass<HloCSE>();
  // 5. 指令融合
  pipeline->AddPass<CpuInstructionFusion>();
  // 6. 内存优化
  pipeline->AddPass<BufferLiveness>();
  pipeline->AddPass<MemoryOptimization>();
}
```

**MLIR 的优化 Pipeline：**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```mlir
// MLIR 的优化 pipeline 通过 Pass Manager 编排
// 使用 mlir-opt 工具可以查看各种优化效果

// 1. TOSA → Linalg lowering
// mlir-opt --tosa-to-linalg input.mlir

// 2. Linalg Tiling + Vectorization
// mlir-opt --linalg-tile-and-fuse --linalg-vectorize input.mlir

// 3. Bufferization（tensor → memref）
// mlir-opt --one-shot-bufferize input.mlir

// 4. SCF → CF lowering
// mlir-opt --lower-scf-to-cf input.mlir

// 5. 最终 lower 到 LLVM IR
// mlir-opt --convert-linalg-to-loops --convert-scf-to-cf \
//          --convert-linalg-to-llvm --convert-memref-to-llvm \
//          --convert-func-to-llvm --reconcile-unrealized-casts \
//          input.mlir
```

### 36.10.4 内存管理策略对比



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
┌─────────────────────────────────────────────────────────────────┐
│                    内存管理策略对比                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TVM:                                                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Workspace Memory (静态规划 + 动态分配)                    │  │
│  │  - Relay 层：通过 StorageRewrite 分析内存复用              │  │
│  │  - TIR 层：通过 buffer 批注指定内存位置                    │  │
│  │  - Runtime：Workspace pool 避免频繁 malloc               │  │
│  │  源码：src/relay/backend/graph_executor_memory.cc         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  XLA:                                                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Buffer Assignment (编译时静态分配)                         │  │
│  │  - BufferAssignment 分析每个 HLO 值的活跃范围              │  │
│  │  - 对不重叠的 buffer 进行复用                              │  │
│  │  - 生成显式的 memory deallocate 指令                      │  │
│  │  源码：tensorflow/compiler/xla/service/buffer_assignment.cc│  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  MLIR:                                                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Bufferization (tensor → memref 转换)                      │  │
│  │  - One-Shot Bufferize 分析 tensor 的使用模式               │  │
│  │  - 自动插入 buffer allocation / deallocation              │  │
│  │  - 支持用户指定的 bufferization 策略                       │  │
│  │  源码：mlir/lib/Dialect/Bufferization/                     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 36.10.5 算子融合策略深度对比

**TVM 的融合策略：**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# src/relay/transforms/fuse_ops.cc
# TVM 的算子融合基于规则，但融合的 kernel 可以通过搜索优化
from tvm.relay import transform

# fuse_opt_level = 0: 不融合
# fuse_opt_level = 1: 仅融合简单的逐元素操作
# fuse_opt_level = 2: 融合 injective + reduce + broadcast（默认）
# fuse_opt_level = 3: 激进融合（包括复杂的计算模式）
mod_fused = transform.FuseOps(fuse_opt_level=2)(mod)
```

**XLA 的融合策略：**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// tensorflow/compiler/xla/service/instruction_fusion.cc
// XLA 的融合决策是确定性的规则
enum class FusionKind {
  kLoop,      // 循环融合：多个 element-wise 操作
  kInput,     // 输入融合：一个 producer 被融合到 consumer
  kOutput,    // 输出融合：多个 consumer 共享 producer
  kCustom,    // 自定义融合：后端特定的融合模式
  kCrossLoop, // 跨循环融合
};
```

**MLIR 的融合策略：**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```mlir
// MLIR 的 Linalg dialect 支持结构化融合
// 通过 tiling 和 fusion 实现

// 原始计算：matmul + relu
func.func @matmul_relu(%A: tensor<128x256xf32>,
                        %B: tensor<256x512xf32>,
                        %bias: tensor<512xf32>) -> tensor<128x512xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x512xf32>
  %mm = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x512xf32>)
                      outs(%empty : tensor<128x512xf32>) -> tensor<128x512xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm, %bias : tensor<128x512xf32>, tensor<512xf32>)
    outs(%empty : tensor<128x512xf32>) {
      ^bb0(%in: f32, %b: f32, %out: f32):
        %add = arith.addf %in, %b : f32
        %relu = arith.maximumf %add, %cst : f32
        linalg.yield %relu : f32
    } -> tensor<128x512xf32>
  return %result : tensor<128x512xf32>
}
```

### 36.10.6 编译时间对比

| 阶段 | TVM | XLA | MLIR |
|------|-----|-----|------|
| **前端导入** | 快（秒级） | 快（秒级） | 快（秒级） |
| **图优化** | 中等（秒级） | 快（秒级） | 中等（秒级） |
| **算子编译** | 慢（分钟~小时，含搜索） | 快（秒~分钟） | 中等（秒~分钟） |
| **代码生成** | 中等（秒~分钟） | 快（秒级） | 中等（秒级） |
| **总编译时间** | 高（可离线） | 低 | 中等 |

### 36.10.7 进一步学习建议

| 方向 | 推荐资源 |
|------|---------|
| TVM 深入 | Apache TVM 官方文档 + MetaSchedule 论文 (MLSys 2022) |
| XLA 深入 | XLA 源码 + JAX 文档 + StableHLO 规范 |
| MLIR 深入 | MLIR 官方文档 + Dialect 设计教程 |
| 编译器理论 | *Compilers: Principles, Techniques, and Tools* (龙书) |
| DL 系统 | *Machine Learning Systems* (MLSys 会议论文) |
| TVM 调优 | AutoTVM 论文 (OSDI 2018) + Ansor 论文 (OSDI 2020) |
| MLIR Dialect 设计 | LLVM Dev Meeting tutorial + mlir.llvm.org/docs |
| XLA 优化 | XLA: Optimizing Compiler for Machine Learning 论文 |

---

> **下一章预告**：第 37 章将通过真实案例展示 TVM 在生产环境中的部署实践。

**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
