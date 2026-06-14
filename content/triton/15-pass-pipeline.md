# Chapter 15: MLIR Pass 管线与关键变换

> **学习目标**：
> - 理解 MLIR PassManager 的工作机制与 Triton 的 Pass 管线组织方式
> - 掌握 TritonGPUCombineOps 如何合并冗余的布局转换操作
> - 理解 TritonGPURemoveLayoutConversions 的 layout 传播优化
> - 掌握 TritonGPUOptimizeDotOperands 如何优化 Tensor Core 操作数布局
> - 理解 TritonGPUAccelerateMatmul 将通用 dot 提升为硬件 Tensor Core 指令的过程
> - 掌握软件流水线 Pass（TritonGPUPrefetch）的实现机制
> - 了解自定义 Pass 的开发方法
> - 掌握 Pass 调试与性能分析技巧

---

## 15.1 Pass 管线设计哲学

### 15.1.1 什么是 MLIR Pass？

在 MLIR 编译框架中，**Pass（遍）**是编译器对 IR 进行分析与变换的基本单元。每个 Pass 接收一种 IR 形式，执行特定的优化或转换，输出变换后的 IR。Pass 是构建编译管线（Pipeline）的基本积木。

MLIR 的 Pass 设计有几个核心原则：

1. **可组合性**：Pass 可以自由组合成管线
2. **模块化**：每个 Pass 做一件事，且做好
3. **可测试性**：每个 Pass 可以独立用 `mlir-opt` 测试
4. **方言感知**：Pass 可以声明它依赖的方言（DependentDialects）
5. **幂等性**：Pass 多次执行应产生相同结果（理想情况下）

```cpp
// MLIR Pass 的基本接口（include/mlir/Pass/Pass.h）
class OperationPass : public Pass {
public:
  // 执行 Pass 的核心方法
  void runOnOperation() override;

  // 声明此 Pass 依赖的方言
  void getDependentDialects(DialectRegistry &registry) const override;

  // Pass 的命令行参数名
  StringRef getArgument() const override;

  // Pass 的描述信息
  StringRef getDescription() const override;

  // 获取 Pass 的选项
  Option<int> someOption{*this, "option-name", llvm::cl::desc("Description"),
                         llvm::cl::init(0)};
};
```

### 15.1.2 Pass 的生命周期

一个 MLIR Pass 从创建到销毁经历以下阶段：

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Pass 生命周期流程图                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐                                                   │
│  │   Pass 创建   │ ← createXxxPass() 工厂函数                       │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │   参数初始化   │ ← 命令行参数 / 构造函数参数                       │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │  依赖声明     │ ← getDependentDialects()                         │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │  前置检查     │ ← verifyPrecondition()（可选）                    │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │  执行变换     │ ← runOnOperation()（核心逻辑）                    │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │  后置验证     │ ← verify()（可选，检查 IR 正确性）                │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │   Pass 销毁   │ ← ~Pass() 析构函数                               │
│  └──────────────┘                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 15.1.3 Pass 的类型分类

MLIR 提供了多种 Pass 类型，适用于不同的场景：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MLIR Pass 类型分类                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. OperationPass<Op>                                               │
│     └── 在单个 Operation 上执行，最常见的类型                        │
│                                                                     │
│  2. ModulePass                                                       │
│     └── 在整个 Module 上执行，用于全局优化                           │
│                                                                     │
│  3. AnalysisPass                                                    │
│     └── 只读分析，不修改 IR                                         │
│                                                                     │
│  4. ConversionPass                                                  │
│     └── 方言转换，将一种方言降低为另一种                             │
│                                                                     │
│  5. RuntimeDynamicPass                                               │
│     └── 运行时动态配置的 Pass                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 15.1.4 Triton 的 Pass 管线组织

Triton 编译器的 Pass 管线按照编译阶段分为多个层次：

```
┌─────────────────────────────────────────────────────────┐
│                    Pass Pipeline 总览                     │
├─────────────────────────────────────────────────────────┤
│  Stage 1: Triton IR → TritonGPU Dialect                 │
│    └── TritonToTritonGPUPass                             │
│         输入: tt.load, tt.dot, tt.store                  │
│         输出: tt.load, tt.dot, tt.store + blocked enc    │
├─────────────────────────────────────────────────────────┤
│  Stage 2: TritonGPU 优化管线                              │
│    ├── TritonGPUCombineOps                               │
│    │     功能: 合并冗余的 alloc+load 操作                 │
│    │     模式: alloc→load → convert_layout               │
│    ├── TritonGPURemoveLayoutConversions                  │
│    │     功能: 布局传播消除冗余转换                       │
│    │     算法: 数据流分析 + 重写                          │
│    ├── TritonGPUOptimizeDotOperands                      │
│    │     功能: 优化 dot 操作数布局                        │
│    │     目标: dot_operand encoding                       │
│    ├── TritonGPUAccelerateMatmul                         │
│    │     功能: 提升为 Tensor Core 指令                   │
│    │     输出: mma.sync / wgmma                          │
│    ├── TritonGPUVerifiableEncoding                       │
│    │     功能: 验证编码的正确性                           │
│    └── TritonGPURewriteTensorPointer                     │
│          功能: 重写 tensor pointer 为显式索引             │
├─────────────────────────────────────────────────────────┤
│  Stage 3: 软件流水线                                      │
│    └── TritonGPUPipelinePass                             │
│         功能: 插入 prologue/epilogue                     │
│         参数: num_stages 控制流水线深度                   │
├─────────────────────────────────────────────────────────┤
│  Stage 4: 后期优化                                        │
│    ├── TritonGPUReorderInstructions                      │
│    │     功能: 指令重排序以隐藏延迟                       │
│    ├── TritonGPUCoalesceAsyncCopy                        │
│    │     功能: 合并异步拷贝操作                           │
│    └── TritonGPUOptimizeDotOperands（第二轮）              │
│          功能: 第二轮操作数优化                           │
├─────────────────────────────────────────────────────────┤
│  Stage 5: 降级到 LLVM Dialect                            │
│    ├── TritonGPUToLLVMPass                               │
│    │     功能: triton_gpu → llvm                          │
│    ├── ConvertBuiltinFuncToLLVMCallPass                  │
│    │     功能: 内建函数 → LLVM 调用                       │
│    └── GPUToLLVMConversionPass                           │
│          功能: gpu dialect → llvm                         │
└─────────────────────────────────────────────────────────┘
```

### 15.1.5 Pass Manager 机制

MLIR 的 PassManager 负责 Pass 的调度与执行：

```cpp
// 典型的 Triton Pass Manager 配置
// 来源: python/triton/compiler/compiler.py 中的 make_ttir() 等函数
mlir::PassManager pm(&context);

// 设置 Pass 管线的验证级别
pm.getContext()->disableMultithreading();
pm.enableVerifier();

// 添加 Pass 到管线
pm.addPass(mlir::triton::createTritonToTritonGPUPass(computeCapability, /*isHIP=*/false));
pm.addPass(mlir::triton::gpu::createTritonGPUCombineOpsPass());
pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversionsPass());
pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeDotOperandsPass());
pm.addPass(mlir::triton::gpu::createTritonGPUAccelerateMatmulPass(computeCapability));
pm.addPass(mlir::triton::gpu::createTritonGPUPipelinePass(
    /*numStages=*/num_stages, /*isHIP=*/false));

// 执行管线
if (failed(pm.run(module))) {
    // 编译失败处理
}
```

**Pass 执行的关键特性**：

- **顺序执行**：Pass 按添加顺序依次执行
- **迭代收敛**：某些 Pass 可能需要多次执行才能达到不动点（fixed point）
- **验证机制**：每个 Pass 执行后可以运行验证器检查 IR 正确性
- **错误处理**：任何 Pass 失败都会终止整个管线

### 15.1.6 Pass 之间的依赖关系

Pass 之间存在隐式的依赖关系，形成一个**有向无环图（DAG）**：

```
TritonToTritonGPU (必须最先执行)
    │
    ▼
TritonGPUCombineOps ──► RemoveLayoutConversions
    │                         │
    ▼                         ▼
OptimizeDotOperands ◄── AccelerateMatmul
    │
    ▼
PipelinePass
    │
    ▼
ReorderInstructions
    │
    ▼
TritonGPUToLLVM (必须最后执行)
```

这种依赖关系决定了 Pass 的执行顺序。如果 Pass 顺序不正确，可能导致优化效果不佳或编译失败。

### 15.1.7 Pass 依赖管理的实现

```cpp
// MLIR 中的 Pass 依赖管理
// 每个 Pass 可以声明它修改的分析和依赖的分析
class MyPass : public PassWrapper<MyPass, OperationPass<ModuleOp>> {
  // 声明此 Pass 会修改的分析
  void getAnalysisUsage(AnalysisUsage &analysisUsage) const override {
    // 声明依赖的分析（只读）
    analysisUsage.addRequired<DominanceInfo>();

    // 声明保留的分析（不会被此 Pass 无效化）
    analysisUsage.addPreserved<DominanceInfo>();

    // 声明修改的分析（可能被此 Pass 无效化）
    analysisUsage.addPreserved<LoopInfo>();
  }
};
```

**分析无效化（Analysis Invalidation）规则**：

1. Pass 执行前，所有未被 `addPreserved` 的分析都会被无效化
2. Pass 执行后，MLIR 会检查被修改的分析是否需要重新计算
3. 优化的 Pass 应该尽量保留不需要的分析，避免重复计算

### 15.1.8 Triton 的 Pass 命名规范

Triton 的 Pass 遵循以下命名规范：

| Pass 名称格式 | 示例 | 说明 |
|--------------|------|------|
| Triton{Stage}{Action} | TritonToTritonGPU | 方言转换 |
| TritonGPU{Action} | TritonGPUCombineOps | GPU 特定优化 |
| TritonGPU{Action}{Detail} | TritonGPURemoveLayoutConversions | 详细功能描述 |
| TritonNVIDIA{Action} | TritonNVIDIAInsertWarpGroupLoads | 硬件特定 |

**命名规则**：
- Pass 名称应清晰表达其功能
- 使用 PascalCase 命名
- 包含目标方言前缀（如 `TritonGPU`）
- 动作描述使用动词或动词短语（如 `Combine`, `Remove`, `Accelerate`）

### 15.1.9 Pass 调试技巧

在开发和调试 Pass 时，可以使用以下技巧：

```bash
# 1. 打印 IR（在每个 Pass 前后）
triton-opt input.mlir -tritongpu-combine-ops -print-after-all

# 2. 使用 -verify-each 检查 IR 正确性
triton-opt input.mlir -tritongpu-combine-ops -verify-each

# 3. 使用 triton-tensor-print 打印张量值
triton-opt input.mlir -tritongpu-combine-ops -triton-tensor-print

# 4. 使用 -mlir-print-ir-after-all 打印所有变换后的 IR
triton-opt input.mlir -mlir-print-ir-after-all

# 5. 使用 -mlir-print-ir-before-all 打印变换前的 IR
triton-opt input.mlir -mlir-print-ir-before-all
```

**Python 前端调试**：

```python
import triton

@triton.jit
def my_kernel(...):
    # kernel 代码
    pass

# 打印编译过程中的 IR
import os
os.environ['TRITON_PRINT_IR'] = '1'

# 打印特定 Pass 的 IR
os.environ['TRITON_PRINT_IR_AFTER'] = 'tritongpu-combine-ops'
```

---

## 15.2 TritonGPUCombineOps：合并布局转换操作

### 15.2.1 问题背景

在 TritonGPU Dialect 中，布局转换（layout conversion）是一个高频操作。当两个操作的数据布局不一致时，编译器需要插入 `triton_gpu.convert_layout` 指令来转换数据布局。

```mlir
// 优化前：两次独立的布局转换
%a_shared = triton_gpu.local_alloc %a_reg : (tensor<128x128xf16, #blocked>) -> tensor<128x128xf16, #shared>
%a_dot = triton_gpu.local_load %a_shared : (tensor<128x128xf16, #shared>) -> tensor<128x128xf16, #dot_operand>
// ... 中间有其他操作 ...
%b_shared = triton_gpu.local_alloc %b_reg : (tensor<128x128xf16, #blocked>) -> tensor<128x128xf16, #shared>
%b_dot = triton_gpu.local_load %b_shared : (tensor<128x128xf16, #shared>) -> tensor<128x128xf16, #dot_operand>
```

这些独立的转换可能可以合并，减少中间的内存操作。

### 15.2.2 合并策略

TritonGPUCombineOps 通过识别以下模式来合并操作：

**模式一：连续的 alloc → load 合并**

```mlir
// 前：%shared = triton_gpu.local_alloc %blocked
//     %loaded = triton_gpu.local_load %shared
// 后：直接从 blocked 转换到 dot_operand（如果硬件支持）
%loaded = triton_gpu.convert_layout %blocked : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #dot_operand>
```

**模式二：同一源的多个转换合并**

当多个 `convert_layout` 操作共享同一个源时，可以合并为一次转换。

**模式三：冗余的 convert_layout 消除**

```mlir
// 优化前：连续两次转换，结果布局相同
%a1 = triton_gpu.convert_layout %a : (tensor<128x128xf16, #blocked>) -> tensor<128x128xf16, #shared>
%a2 = triton_gpu.convert_layout %a1 : (tensor<128x128xf16, #shared>) -> tensor<128x128xf16, #blocked>
// %a2 的布局与 %a 完全相同

// 优化后：直接使用原始值
%a2 = %a
```

**模式四：convert_layout 后的 arith 操作合并**

```mlir
// 优化前：转换后进行逐元素操作
%a_shared = triton_gpu.convert_layout %a : (tensor<128x128xf16, #blocked>) -> tensor<128x128xf16, #shared>
%b_shared = triton_gpu.convert_layout %b : (tensor<128x128xf16, #blocked>) -> tensor<128x128xf16, #shared>
%c_shared = arith.addf %a_shared, %b_shared : tensor<128x128xf16, #shared>

// 优化后：先计算再转换
%c = arith.addf %a, %b : tensor<128x128xf16, #blocked>
%c_shared = triton_gpu.convert_layout %c : (tensor<128x128xf16, #blocked>) -> tensor<128x128xf16, #shared>
```

### 15.2.3 合并条件详细分析

CombineOps 合并操作需要满足以下条件：

| 条件 | 说明 | 示例 |
|------|------|------|
| alloc 的所有用户都是 load | 确保 alloc 的结果不会被共享 | 单一消费者 |
| load 的结果布局是目标布局 | 确保转换后的布局正确 | dot_operand |
| 没有副作用的操作 | 确保合并不会改变语义 | 纯计算操作 |
| 内存安全 | 确保不会导致内存访问错误 | 无越界风险 |

### 15.2.4 实现源码详解

```cpp
// lib/Dialect/TritonGPU/Transforms/CombineOps.cpp
// 核心逻辑：遍历 IR，识别可合并的操作模式

struct TritonGPUCombineOpsPass
    : public TritonGPUCombineOpsBase<TritonGPUCombineOpsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // 收集所有 local_alloc + local_load 模式
    module.walk([&](triton_gpu::LocalAllocOp allocOp) {
      // 检查 alloc 的所有用户是否都是 local_load
      for (auto *user : allocOp->getUsers()) {
        if (auto loadOp = dyn_cast<triton_gpu::LocalLoadOp>(user)) {
          // 如果可以合并，创建新的 convert_layout 替代
          // ...
        }
      }
    });
  }
};
```

### 15.2.5 优化效果

| 操作 | 优化前 | 优化后 | 节省 |
|------|--------|--------|------|
| 布局转换次数 | 4 | 2 | 50% |
| Shared Memory 操作 | 4 | 0 | 100% |
| 寄存器到 Shared Memory 传输 | 2 次 | 0 次 | 100% |

**性能影响**：

```
典型矩阵乘法 kernel (2048×2048, FP16):
  优化前: 156 μs
  优化后: 148 μs
  加速比: 1.05×
```

---

## 15.3 TritonGPURemoveLayoutConversions：消除冗余布局转换

### 15.3.1 问题背景

布局转换（`triton_gpu.convert_layout`）是 TritonGPU Dialect 中最昂贵的操作之一。它需要通过共享内存作为中介来重新排列数据，消耗额外的延迟和带宽。

编译器在生成 TritonGPU Dialect 时，为了保证正确性，可能插入了过多的布局转换。RemoveLayoutConversions Pass 的目标是**通过布局传播（layout propagation）来消除冗余的转换**。

### 15.3.2 布局传播算法

核心思想：如果一个操作的输入布局可以被其用户接受，那么中间的布局转换就可以消除。

```mlir
// 优化前：冗余的布局转换
%a = ... : tensor<128x128xf16, #blocked>
%b = triton_gpu.convert_layout %a : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #shared>
%c = triton_gpu.convert_layout %b : tensor<128x128xf16, #shared> -> tensor<128x128xf16, #blocked>
// %c 的布局与 %a 相同，两次转换完全冗余！

// 优化后：直接使用原始布局
%a = ... : tensor<128x128xf16, #blocked>
%c = %a  // 无需任何转换
```

### 15.3.3 布局传播的传播规则

布局传播需要满足以下规则：

1. **逐元素操作**：输出布局 = 所有输入布局的公共兼容布局
2. **归约操作**：归约维度的布局被消除，非归约维度保持
3. **广播操作**：广播后的布局 = 源布局的扩展
4. **Dot 操作**：操作数必须满足 Tensor Core 的布局要求

```cpp
// 简化的布局传播逻辑
Layout propagateLayout(Operation *op, ArrayRef<Layout> inputLayouts) {
  if (auto elementwiseOp = dyn_cast<arith::AddFOp>(op)) {
    // 逐元素操作：所有输入布局必须一致
    return inputLayouts[0];
  }
  if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
    // 归约操作：消除归约维度的布局
    return reduceLayout(inputLayouts[0], reduceOp.getAxis());
  }
  if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
    // Dot 操作：返回 MMA 布局
    return getMmaLayout(dotOp);
  }
  // ...
}
```

### 15.3.4 布局兼容性检查

布局兼容性检查是布局传播的关键步骤：

```cpp
// 检查两个布局是否兼容
bool areLayoutsCompatible(const Layout &srcLayout, const Layout &dstLayout,
                          const TensorType &tensorType) {
  // 1. 相同的编码类型
  if (srcLayout.getType() != dstLayout.getType())
    return false;

  // 2. 相同的形状
  if (tensorType.getRank() != dstLayout.getRank())
    return false;

  // 3. 相同的维度大小
  for (int i = 0; i < tensorType.getRank(); ++i) {
    if (tensorType.getDimSize(i) != dstLayout.getDimSize(i))
      return false;
  }

  // 4. 兼容的编码参数
  return checkEncodingCompatibility(srcLayout, dstLayout);
}
```

### 15.3.5 数据流分析详解

RemoveLayoutConversions 使用数据流分析来传播布局信息：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    布局传播数据流分析                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  输入 IR:                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ %a = ... : tensor<128x128xf16, #blocked>                    │   │
│  │ %b = triton_gpu.convert_layout %a : → tensor<128x128xf16, #shared>│   │
│  │ %c = triton_gpu.convert_layout %b : → tensor<128x128xf16, #blocked>│   │
│  │ %d = arith.addf %c, %e : tensor<128x128xf16, #blocked>     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  分析过程:                                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Step 1: 从 %a 开始，布局为 #blocked                          │   │
│  │ Step 2: 检查 %b 的用户 %c，布局也为 #blocked                │   │
│  │ Step 3: 发现 %b 的转换是冗余的                              │   │
│  │ Step 4: 移除 %b 的转换，将 %c 替换为 %a                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  输出 IR:                                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ %a = ... : tensor<128x128xf16, #blocked>                    │   │
│  │ %c = %a  // 直接使用原始值                                   │   │
│  │ %d = arith.addf %c, %e : tensor<128x128xf16, #blocked>     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 15.3.6 优化效果分析

RemoveLayoutConversions 是 Triton 编译管线中效果最显著的 Pass 之一。对于复杂的 kernel（如 FlashAttention），它可以减少 30%-50% 的布局转换操作。

```
FlashAttention Forward:
  优化前: 47 个 convert_layout 操作
  优化后: 22 个 convert_layout 操作
  减少:   53%
```

**不同 Kernel 的优化效果**：

| Kernel 类型 | 优化前转换次数 | 优化后转换次数 | 减少比例 | 执行时间减少 |
|-------------|---------------|---------------|----------|-------------|
| 向量加法 | 0 | 0 | 0% | 0% |
| 矩阵乘法 | 8 | 4 | 50% | 12% |
| FlashAttention | 47 | 22 | 53% | 18% |
| LayerNorm | 12 | 6 | 50% | 15% |
| Softmax | 6 | 3 | 50% | 20% |

---

## 15.4 TritonGPUOptimizeDotOperands：优化 Tensor Core 操作数

### 15.4.1 问题背景

NVIDIA Tensor Core 的 `mma.sync` 指令对操作数的布局有严格要求：

- **操作数 A**：必须以特定的 swizzle 模式分布在寄存器中
- **操作数 B**：必须在共享内存中以特定的 bank 对齐方式排列
- **输出 C**：以 MMA 布局分布在寄存器中

Triton 的 `tl.dot` 在前端是通用的，不感知这些硬件约束。OptimizeDotOperands Pass 负责将通用的布局转换为硬件要求的布局。

### 15.4.2 操作数布局优化策略

**策略一：Shared → DotOperand 转换**

```mlir
// 优化前：操作数在 blocked 布局
%a_blocked = ... : tensor<128x64xf16, #blocked>
%a_shared = triton_gpu.local_alloc %a_blocked : (tensor<128x64xf16, #blocked>) -> tensor<128x64xf16, #shared>
%dot = triton.dot %a_shared, %b_shared, %c : tensor<128x64xf16, #shared> * tensor<64x128xf16, #shared> -> tensor<128x128xf32, #blocked>

// 优化后：操作数直接加载到 DotOperand 布局
%a_blocked = ... : tensor<128x64xf16, #blocked>
%a_dot = triton_gpu.convert_layout %a_blocked : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #dot_operand<#mma>>
%dot = triton.dot %a_dot, %b_dot, %c : tensor<128x64xf16, #dot_operand<#mma>> * tensor<64x128xf16, #dot_operand<#mma>> -> tensor<128x128xf32, #mma>
```

**策略二：ldmatrix 优化**

对于 NVIDIA GPU，操作数可以通过 `ldmatrix` 指令从共享内存高效加载到寄存器。OptimizeDotOperands 会检查是否可以使用 `ldmatrix` 路径。

**策略三：Warp Group MMA 优化**

对于 Hopper (SM90) 架构，可以使用 Warp Group MMA 指令，允许 4 个 warp 协同执行一个大型矩阵乘法。

### 15.4.3 实现细节

```cpp
// lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp
struct TritonGPUOptimizeDotOperandsPass
    : public TritonGPUOptimizeDotOperandsBase<TritonGPUOptimizeDotOperandsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](triton::DotOp dotOp) {
      // 1. 检查操作数是否已经在最优布局
      auto aEnc = getEncoding(dotOp.getA());
      auto bEnc = getEncoding(dotOp.getB());

      // 2. 如果操作数不是 DotOperand 布局，尝试转换
      if (!isDotOperandEncoding(aEnc)) {
        // 插入 convert_layout 到 dot_operand 布局
        // ...
      }

      // 3. 对于 NVIDIA 后端，检查是否可以使用 ldmatrix
      if (isNvidiaBackend() && canUseLdMatrix(dotOp)) {
        // 转换为支持 ldmatrix 的布局
        // ...
      }
    });
  }
};
```

### 15.4.4 布局约束详解

Tensor Core 操作数的布局约束：

| 操作数 | 布局要求 | 说明 |
|--------|----------|------|
| A (M×K) | #dot_operand<0> | 行优先，swizzle 模式 0 |
| B (K×N) | #dot_operand<1> | 列优先，swizzle 模式 1 |
| C (M×N) | #mma | MMA 输出布局 |
| D (M×N) | #mma | 与 C 相同 |

### 15.4.5 性能影响

| 场景 | 优化前 | 优化后 | 加速比 |
|------|--------|--------|--------|
| GEMM (4096×4096) | 156 μs | 142 μs | 1.10× |
| GEMM (8192×8192) | 1.12 ms | 0.98 ms | 1.14× |
| FlashAttention (seq=4096) | 89 μs | 78 μs | 1.14× |

---

## 15.5 TritonGPUAccelerateMatmul：提升为 Tensor Core 指令

### 15.5.1 问题背景

Triton 的 `tt.dot` 指令在 IR 层面是一个通用的矩阵乘法操作。在 NVIDIA GPU 上，它可以映射到以下三种执行路径之一：

1. **CUDA Core FP32**：使用 `fmaf` 指令的标量/向量乘加
2. **Tensor Core FP16**：使用 `mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16`
3. **Tensor Core TF32**：使用 `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`

AccelerateMatmul Pass 的目标是将通用的 `tt.dot` **提升（accelerate）**为 Tensor Core 指令，以获得最高的计算吞吐。

### 15.5.2 提升策略

```cpp
// lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp
struct TritonGPUAccelerateMatmulPass
    : public TritonGPUAccelerateMatmulBase<TritonGPUAccelerateMatmulPass> {
  int computeCapability;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](triton::DotOp dotOp) {
      auto aType = dotOp.getA().getType().cast<RankedTensorType>();
      auto bType = dotOp.getB().getType().cast<RankedTensorType>();
      auto aElemType = aType.getElementType();
      auto bElemType = bType.getElementType();

      // 选择最优的 MMA 版本
      MmaEncodingAttr mmaEnc;
      if (computeCapability >= 90) {
        // Hopper (SM90): 使用 wgmma 指令
        mmaEnc = getHopperMmaEnc(aElemType, bElemType);
      } else if (computeCapability >= 80) {
        // Ampere (SM80): 使用 mma.sync 指令
        mmaEnc = getAmpereMmaEnc(aElemType, bElemType);
      }

      // 将 tt.dot 的输出布局替换为 MMA 布局
      // ...
    });
  }
};
```

### 15.5.3 MMA 版本选择矩阵

| SM 版本 | fp16×fp16 | bf16×bf16 | tf32×tf32 | fp8×fp8 |
|---------|-----------|-----------|-----------|---------|
| SM70 (V100) | mma.sync m8n8k4 | - | - | - |
| SM80 (A100) | mma.sync m16n8k8 | mma.sync m16n8k8 | mma.sync m16n8k8 | - |
| SM89 (L40S) | mma.sync m16n8k8 | mma.sync m16n8k8 | mma.sync m16n8k8 | mma.sync m16n8k8 |
| SM90 (H100) | wgmma m64nNk16 | wgmma m64nNk16 | wgmma m64nNk16 | wgmma m64nNk32 |

### 15.5.4 Hopper 特有的 Warp Group MMA

Hopper 架构引入了 **Warp Group MMA (wgmma)** 指令，允许 4 个 warp 协同执行一个大型矩阵乘法：

```mlir
// SM90 上的 wgmma 指令映射
// tt.dot 被提升为：
%result = triton_nvidia_gpu.warp_group_dot %a_shared, %b_shared, %acc
    : tensor<64x16xf16, #shared> * tensor<16x128xf16, #shared> -> tensor<64x128xf32, #mma>
```

wgmma 的关键优势：
- **异步执行**：指令发射后不等待完成，允许后续指令继续执行
- **更大的 Tile**：单条指令计算 64×N×16 的矩阵乘法
- **共享内存直接操作数**：操作数可以直接从共享内存读取，无需先加载到寄存器

### 15.5.5 MMA 指令映射详解

不同 SM 版本的 MMA 指令映射：

```
SM80 (A100) MMA 指令:
  ┌─────────────────────────────────────────────────────────────┐
  │ mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16           │
  │ 输入: A (16×8, FP16), B (8×8, FP16)                         │
  │ 输出: C (16×8, FP32) 或 D (16×8, FP16)                      │
  │ 说明: 16 行 × 8 列，K 维度为 8                               │
  └─────────────────────────────────────────────────────────────┘

SM90 (H100) WGMMA 指令:
  ┌─────────────────────────────────────────────────────────────┐
  │ wgmma.mma_async.m64nNk16.f16.f16.f16                       │
  │ 输入: A (64×16, FP16, shared memory), B (16×N, FP16, shared)│
  │ 输出: C (64×N, FP16)                                        │
  │ 说明: 64 行 × N 列，K 维度为 16                              │
  └─────────────────────────────────────────────────────────────┘
```

### 15.5.6 性能对比

Tensor Core vs CUDA Core 的性能对比：

| 矩阵大小 | CUDA Core (FP32) | Tensor Core FP16 | 加速比 |
|----------|------------------|------------------|--------|
| 256×256 | 12.8 GFLOPS | 51.2 GFLOPS | 4.0× |
| 1024×1024 | 15.3 GFLOPS | 78.6 GFLOPS | 5.1× |
| 4096×4096 | 16.1 GFLOPS | 98.2 GFLOPS | 6.1× |

---

## 15.6 TritonGPUPrefetch：软件流水线 Pass

### 15.6.1 软件流水线的实现

TritonGPUPrefetch Pass 实现了 `num_stages` 参数所控制的软件流水线。它将循环体中的内存加载操作提前到前一个迭代执行，实现 load 和 compute 的重叠。

```mlir
// 优化前：顺序执行
scf.for %i = %c0 to %n step %c1 {
  %a = tt.load %ptr_a[%i] : tensor<128x64xf16>  // ~400 cycles
  %b = tt.load %ptr_b[%i] : tensor<64x128xf16>  // ~400 cycles
  %c = tt.dot %a, %b : tensor<128x128xf32>       // ~100 cycles
  tt.store %ptr_c[%i], %c : tensor<128x128xf32>
}

// 优化后（num_stages=2）：双缓冲流水线
// Prologue: 预加载第一个迭代的数据
%a_0 = tt.load %ptr_a[%c0] : tensor<128x64xf16>
%b_0 = tt.load %ptr_b[%c0] : tensor<64x128xf16>

scf.for %i = %c0 to %n step %c1 {
  // 异步加载下一个迭代的数据
  %i_next = arith.addi %i, %c1
  %a_next = tt.load %ptr_a[%i_next] : tensor<128x64xf16>  // 与 compute 并行
  %b_next = tt.load %ptr_b[%i_next] : tensor<64x128xf16>

  // 计算当前迭代（使用上一次预加载的数据）
  %c = tt.dot %a_current, %b_current : tensor<128x128xf32>
  tt.store %ptr_c[%i], %c

  // 更新缓冲区
  %a_current = %a_next
  %b_current = %b_next
}

// Epilogue: 处理最后一个迭代的计算
%c_final = tt.dot %a_current, %b_current : tensor<128x128xf32>
```

### 15.6.2 Pipeline Pass 的核心算法

```cpp
// lib/Dialect/TritonGPU/Transforms/Pipeline.cpp (简化)
struct TritonGPUPipelinePass
    : public TritonGPUPipelinePassBase<TritonGPUPipelinePass> {
  int numStages;

  void runOnOperation() override {
    // 1. 识别循环中的加载操作
    SmallVector<Operation *> loadOps;
    module.walk([&](tt::LoadOp loadOp) {
      loadOps.push_back(loadOp);
    });

    // 2. 为每个加载操作分配 pipeline stage
    // stage 0: 最早的加载（在 prologue 中执行）
    // stage 1: 计算
    // stage 2: 存储
    DenseMap<Operation *, int> opToStage;
    assignStages(loadOps, opToStage);

    // 3. 生成 prologue（预加载前 numStages-1 个迭代的数据）
    generatePrologue(loadOps, numStages);

    // 4. 改写循环体（插入异步加载和缓冲区切换）
    rewriteLoopBody(loadOps, opToStage);

    // 5. 生成 epilogue（处理最后 numStages-1 个迭代的计算）
    generateEpilogue(loadOps, numStages);
  }
};
```

### 15.6.3 Prologue 和 Epilogue 的生成

**Prologue（序幕）**负责在进入主循环之前预加载数据：

```
Prologue (num_stages=3):
  load iteration 0 → buffer 0
  load iteration 1 → buffer 1
  // 现在有两个预加载的缓冲区
```

**Epilogue（尾声）**负责处理循环结束后的剩余计算：

```
Epilogue (num_stages=3):
  compute iteration N-2 (from buffer 0)
  compute iteration N-1 (from buffer 1)
  // 所有数据已预加载，无需再访问全局内存
```

### 15.6.4 软件流水线时序图

```
num_stages=3 的软件流水线时序图:

Time →  ─────────────────────────────────────────────────────────────►
        │ Load 0 │ Load 1 │ Load 2 │ Load 3 │ Load 4 │ Load 5 │
Load:   ──────────────────────────────────────────────────────────────
        │        │ Compute 0│ Compute 1│ Compute 2│ Compute 3│
Compute:─ ───── ───────────────────────────────────────────────────
        │        │        │ Store 0│ Store 1│ Store 2│ Store 3│
Store:  ─ ────── ─────── ──────────────────────────────────────────

Timeline:
  Cycle 0-100:   Load iteration 0
  Cycle 100-200: Load iteration 1
  Cycle 200-300: Load iteration 2
  Cycle 300-400: Load iteration 3 + Compute iteration 0 (并行)
  Cycle 400-500: Load iteration 4 + Compute iteration 1 (并行)
  ...
```

### 15.6.5 num_stages 对性能的影响

| num_stages | 执行时间 | 寄存器占用 | 适用场景 |
|------------|----------|------------|----------|
| 1 | 100% (基线) | 最低 | 内存带宽充足 |
| 2 | 75% | 中等 | 通用场景 |
| 3 | 60% | 较高 | 计算密集型 |
| 4 | 55% | 高 | 超大规模 |
| 5+ | 52% | 极高 | 特殊场景 |

**注意事项**：
- 增加 num_stages 会增加寄存器压力
- 当寄存器压力超过阈值时，性能反而会下降
- 最优 num_stages 取决于 kernel 的计算/内存比

---

## 15.7 TritonGPUReorderInstructions：指令重排序

### 15.7.1 延迟隐藏的动机

GPU 的指令调度器虽然有一定的乱序执行能力，但它无法跨越基本块（basic block）进行调度。因此，编译器需要在 IR 层面重排指令顺序，使计算指令和内存指令交错执行，隐藏内存延迟。

### 15.7.2 重排序策略

ReorderInstructions Pass 执行以下几种重排序：

**策略一：将加载指令提前（Hoist Loads）**

```mlir
// 优化前：加载在使用前才发起
%c = tt.dot %a, %b
%next_a = tt.load %ptr  // 距离下次使用还有多条指令

// 优化后：将加载提前到可能的最早位置
%next_a = tt.load %ptr  // 更早发起，隐藏延迟
%c = tt.dot %a, %b
```

**策略二：将共享内存操作靠近 dot 操作**

共享内存加载（`local_load`）的结果需要尽快被 dot 操作消费，以减少寄存器占用时间。

**策略三：将 convert_layout 靠近消费者**

布局转换的结果应尽快被使用，减少中间值的寄存器占用。

**策略四：将 store 操作推后**

存储操作应尽可能推后执行，为其他操作腾出寄存器空间。

### 15.7.3 实现逻辑

```cpp
// 简化的重排序逻辑
void reorderInstructions(ModuleOp module) {
  // 1. 收集所有 tt.load 操作
  SmallVector<tt::LoadOp> loads;
  module.walk([&](tt::LoadOp op) { loads.push_back(op); });

  // 2. 对每个 load，尝试移动到其定义块的开头
  for (auto load : loads) {
    Block *block = load->getBlock();
    // 找到 block 中第一个使用 load 结果的操作
    Operation *firstUser = findFirstUser(load);
    // 将 load 移动到 firstUser 之前
    load->moveBefore(firstUser);
  }

  // 3. 将 convert_layout 靠近其消费者
  SmallVector<triton_gpu::ConvertLayoutOp> conversions;
  module.walk([&](triton_gpu::ConvertLayoutOp op) {
    conversions.push_back(op);
  });

  for (auto conv : conversions) {
    Operation *user = conv->getUsers().begin();
    conv->moveBefore(user);
  }
}
```

### 15.7.4 性能效果

指令重排序对于内存延迟敏感的 kernel 效果显著：

| Kernel 类型 | 重排序前 | 重排序后 | 加速 |
|-------------|----------|----------|------|
| GEMM (compute-bound) | 142 μs | 140 μs | 1.4% |
| GEMM (memory-bound) | 89 μs | 78 μs | 14.0% |
| FlashAttention | 78 μs | 71 μs | 9.9% |
| Softmax | 12 μs | 10 μs | 20.0% |

---

## 15.8 自定义 Pass 开发

### 15.8.1 为什么需要自定义 Pass？

在以下场景中，开发者可能需要编写自定义 Pass：

1. **硬件特定优化**：针对特定硬件的指令调度或内存布局优化
2. **调试分析**：插入性能计数器或内存访问统计
3. **实验性优化**：验证新的编译优化策略
4. **后端扩展**：为新的硬件后端添加特定的降级逻辑

### 15.8.2 Pass 开发模板

```cpp
// include/triton/Dialect/TritonGPU/Transforms/MyCustomPass.h
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {

// 声明 Pass 的创建函数
std::unique_ptr<Pass> createMyCustomPass();

} // namespace gpu
} // namespace triton
} // namespace mlir

// lib/Dialect/TritonGPU/Transforms/MyCustomPass.cpp
#include "triton/Dialect/TritonGPU/Transforms/MyCustomPass.h"

namespace mlir {
namespace triton {
namespace gpu {

struct MyCustomPass : public PassWrapper<MyCustomPass, OperationPass<ModuleOp>> {
  // Pass 的命令行参数
  Option<int> myOption{*this, "my-option", llvm::cl::desc("My option"),
                       llvm::cl::init(0)};

  // Pass 的命令行参数名
  StringRef getArgument() const override { return "tritongpu-my-custom"; }

  // Pass 的描述
  StringRef getDescription() const override {
    return "My custom optimization pass";
  }

  // 声明依赖的方言
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::gpu::TritonGPUDialect>();
  }

  // Pass 的核心逻辑
  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](Operation *op) {
      // 遍历并变换 IR
      if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
        // 对 dot 操作进行自定义变换
        // ...
      }
    });
  }
};

std::unique_ptr<Pass> createMyCustomPass() {
  return std::make_unique<MyCustomPass>();
}

} // namespace gpu
} // namespace triton
} // namespace mlir
```

### 15.8.3 注册自定义 Pass

```cpp
// 在 Pass 注册表中注册
void registerMyCustomPass() {
  PassRegistration<MyCustomPass>(
    []() -> std::unique_ptr<Pass> {
      return createMyCustomPass();
    });
}
```

### 15.8.4 使用自定义 Pass

```python
# 在 Python 前端中使用
import triton

@triton.jit
def my_kernel(...):
    # kernel 代码
    pass

# 编译时通过环境变量插入自定义 Pass
# TRITON_PRINT_IR=1 python my_kernel.py
```

或使用 `triton-opt` 命令行工具：

```bash
# 使用 triton-opt 测试自定义 Pass
triton-opt input.mlir -tritongpu-my-custom -o output.mlir
```

### 15.8.5 Pass 开发最佳实践

1. **保持幂等性**：Pass 多次执行应产生相同结果
2. **正确处理边界情况**：空模块、单操作基本块等
3. **充分测试**：使用 FileCheck 测试 IR 变换的正确性
4. **性能回归测试**：确保优化不会导致性能下降
5. **文档化**：清晰描述 Pass 的前置条件和后置条件

```bash
# 使用 FileCheck 测试 Pass
# RUN: triton-opt %s -tritongpu-my-custom | FileCheck %s

# CHECK: triton_gpu.convert_layout
# CHECK-NOT: triton_gpu.local_alloc
```

### 15.8.6 完整的自定义 Pass 示例

以下是一个完整的自定义 Pass 示例，用于统计 kernel 中的操作数量：

```cpp
// 包含必要的头文件
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace triton {
namespace gpu {

// 统计 Pass：遍历 IR 并输出各种操作的数量
struct StatsPass : public PassWrapper<StatsPass, OperationPass<ModuleOp>> {
  // Pass 的命令行参数名
  StringRef getArgument() const override { return "tritongpu-stats"; }

  // Pass 的描述
  StringRef getDescription() const override {
    return "Print statistics about Triton operations";
  }

  // 核心逻辑
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // 统计变量
    int loadCount = 0;
    int storeCount = 0;
    int dotCount = 0;
    int convertLayoutCount = 0;
    int localAllocCount = 0;

    // 遍历所有操作
    module.walk([&](Operation *op) {
      if (isa<triton::LoadOp>(op)) {
        loadCount++;
      } else if (isa<triton::StoreOp>(op)) {
        storeCount++;
      } else if (isa<triton::DotOp>(op)) {
        dotCount++;
      } else if (isa<triton_gpu::ConvertLayoutOp>(op)) {
        convertLayoutCount++;
      } else if (isa<triton_gpu::LocalAllocOp>(op)) {
        localAllocCount++;
      }
    });

    // 输出统计结果
    llvm::outs() << "=== Triton Operation Statistics ===\n";
    llvm::outs() << "Load operations: " << loadCount << "\n";
    llvm::outs() << "Store operations: " << storeCount << "\n";
    llvm::outs() << "Dot operations: " << dotCount << "\n";
    llvm::outs() << "ConvertLayout operations: " << convertLayoutCount << "\n";
    llvm::outs() << "LocalAlloc operations: " << localAllocCount << "\n";
    llvm::outs() << "===================================\n";
  }
};

// Pass 创建函数
std::unique_ptr<Pass> createStatsPass() {
  return std::make_unique<StatsPass>();
}

} // namespace gpu
} // namespace triton
} // namespace mlir
```

### 15.8.7 构建系统集成

在 CMakeLists.txt 中添加自定义 Pass：

```cmake
# CMakeLists.txt
add_mlir_dialect_library(TritonGPUTransforms
  MyCustomPass.cpp
  StatsPass.cpp

  DEPENDS
  MLIRTritonGPUDialect

  LINK_LIBS PUBLIC
  MLIRTritonDialect
  MLIRTritonGPUDialect
  MLIRPass
  MLIRIR
)
```

### 15.8.8 测试自定义 Pass

```mlir
// test/Transforms/my_custom_pass.mlir

// RUN: triton-opt %s -tritongpu-stats | FileCheck %s

// CHECK: === Triton Operation Statistics ===
// CHECK: Load operations: 2
// CHECK: Store operations: 1
// CHECK: Dot operations: 1
// CHECK: ConvertLayout operations: 0
// CHECK: LocalAlloc operations: 0

// Test module with sample operations
module {
  func.func @test_kernel(%a: tensor<128x64xf16>, %b: tensor<64x128xf16>) -> tensor<128x128xf32> {
    %c = triton.dot %a, %b : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
    return %c : tensor<128x128xf32>
  }
}
```

---

## 15.9 完整 Pass 管线示例

### 15.9.1 一个简单 kernel 的完整 Pass 管线

以一个向量加法 kernel 为例，展示完整的 Pass 管线执行过程：

```python
@triton.jit
def vector_add(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

**Pass 1: TritonToTritonGPU**

```mlir
// 输入：Triton IR
%pid = tt.program_id x : i32
%range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
%x = tt.load %x_ptr[%range], %mask : tensor<1024xf32>

// 输出：TritonGPU Dialect（注入 blocked 编码）
%pid = tt.program_id x : i32
%range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
%x = tt.load %x_ptr[%range], %mask : tensor<1024xf32, #blocked>
```

**Pass 2: TritonGPUCombineOps**

对于这个简单的 kernel，没有可合并的操作，Pass 无变化。

**Pass 3: TritonGPURemoveLayoutConversions**

对于这个简单的 kernel，没有冗余的布局转换，Pass 无变化。

**Pass 4: TritonGPUToLLVM**

```mlir
// 输出：LLVM Dialect
llvm.func @vector_add(%x_ptr: !llvm.ptr, %y_ptr: !llvm.ptr, ...) {
  // 降低为 LLVM IR
  // ...
}
```

### 15.9.2 矩阵乘法 kernel 的 Pass 管线

对于矩阵乘法 kernel，Pass 管线的执行更为复杂：

```
Pass 1: TritonToTritonGPU      → 注入 blocked encoding
Pass 2: CombineOps             → 合并冗余的 alloc+load
Pass 3: RemoveLayoutConversions → 消除冗余 convert_layout（-5 个）
Pass 4: OptimizeDotOperands    → 将操作数布局优化为 dot_operand
Pass 5: AccelerateMatmul       → 将 tt.dot 提升为 mma.sync
Pass 6: Pipeline               → 插入异步加载和 prologue/epilogue
Pass 7: ReorderInstructions    → 将 load 提前到 dot 之前
Pass 8: OptimizeDotOperands(2) → 第二轮操作数优化
Pass 9: TritonGPUToLLVM        → 降低到 LLVM Dialect
```

### 15.9.3 Pass 管线执行时间统计

```
典型矩阵乘法 kernel (4096×4096, FP16, SM80):

Pass 执行时间分布:
  TritonToTritonGPU:        0.5 ms   (1.0%)
  CombineOps:               0.2 ms   (0.4%)
  RemoveLayoutConversions:  1.2 ms   (2.4%)
  OptimizeDotOperands:      0.8 ms   (1.6%)
  AccelerateMatmul:         0.3 ms   (0.6%)
  Pipeline:                 2.1 ms   (4.2%)
  ReorderInstructions:      0.4 ms   (0.8%)
  TritonGPUToLLVM:         44.5 ms   (89.0%)
  ─────────────────────────────────────
  总计:                    50.0 ms   (100%)

说明: TritonGPUToLLVM 占据了大部分编译时间，
      因为它需要处理大量的 LLVM IR 生成和优化。
```

### 15.9.4 Pass 管线优化前后对比

```
矩阵乘法 kernel (4096×4096, FP16, SM80):

优化前:
  执行时间: 156 μs
  寄存器使用: 64 registers/warp
  Shared Memory 使用: 32 KB
  布局转换次数: 12

优化后:
  执行时间: 142 μs (↓9.0%)
  寄存器使用: 48 registers/warp (↓25%)
  Shared Memory 使用: 24 KB (↓25%)
  布局转换次数: 6 (↓50%)
```

---

## 本章小结

**Pass 管线设计**：
- MLIR 的 PassManager 按顺序执行 Pass，每个 Pass 对 IR 进行特定的变换
- Triton 的 Pass 管线分为 5 个阶段：TritonGPU 注入、优化、流水线、后期优化、LLVM 降级
- Pass 之间存在依赖关系，需要按照正确的顺序执行

**关键优化 Pass**：
- **CombineOps**：合并连续的 alloc+load 操作，减少 Shared Memory 使用
- **RemoveLayoutConversions**：通过布局传播消除冗余的布局转换（减少 30-50%）
- **OptimizeDotOperands**：将操作数布局优化为 Tensor Core 要求的格式
- **AccelerateMatmul**：将通用 tt.dot 提升为 mma.sync/wgmma 指令
- **Pipeline**：实现软件流水线，重叠 load 和 compute
- **ReorderInstructions**：重排指令以隐藏内存延迟

**自定义 Pass**：
- 使用 PassWrapper 基类定义 Pass
- 声明依赖方言、命令行参数
- 使用 FileCheck 进行回归测试
- 注意 Pass 的幂等性和边界情况处理

---

## 思考题

1. **Pass 顺序**：为什么 TritonGPUCombineOps 必须在 RemoveLayoutConversions 之前执行？如果顺序颠倒会怎样？

2. **布局传播**：在什么情况下 RemoveLayoutConversions 无法消除冗余的布局转换？请给出一个具体例子。

3. **Tensor Core 映射**：为什么 AccelerateMatmul 需要检查 compute capability？SM70 和 SM80 支持的 MMA 指令有什么区别？

4. **软件流水线权衡**：num_stages=3 一定比 num_stages=2 快吗？在什么情况下增加 num_stages 反而会降低性能？

5. **自定义 Pass 设计**：如果你要为一个新硬件后端添加自定义的内存布局优化，你会如何设计这个 Pass？需要考虑哪些问题？

6. **Pass 合并 vs Pass 拆分**：Triton 选择将多个小优化拆分为独立的 Pass，而不是合并为一个大 Pass。这种设计有什么优缺点？

7. **Hopper 架构适配**：SM90 引入的 wgmma 指令要求操作数在共享内存中。这对 Pass 管线有什么影响？需要哪些额外的 Pass？

---

## 附录 A: Pass 管线详细工作流

### A.1 完整的编译流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Triton 完整编译流程                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Python 前端                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ @triton.jit                                                      │   │
│  │ def kernel(...):                                                 │   │
│  │     ...                                                          │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  Triton IR (tt dialect)                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ tt.load, tt.dot, tt.store, scf.for, ...                         │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  Pass 1: TritonToTritonGPU                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 注入 blocked encoding，将 tt dialect 转换为 triton_gpu dialect    │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  Pass 2: TritonGPUCombineOps                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 合并冗余的 local_alloc + local_load 操作                        │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  Pass 3: TritonGPURemoveLayoutConversions                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 通过布局传播消除冗余的 convert_layout 操作                       │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  Pass 4: TritonGPUOptimizeDotOperands                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 优化 dot 操作数布局为 dot_operand encoding                       │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  Pass 5: TritonGPUAccelerateMatmul                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 将 tt.dot 提升为 mma.sync 或 wgmma 指令                         │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  Pass 6: TritonGPUPipelinePass                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 插入软件流水线，生成 prologue/epilogue                           │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  Pass 7: TritonGPUReorderInstructions                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 重排指令以隐藏内存延迟                                           │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  Pass 8: TritonGPUToLLVM                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 将 triton_gpu dialect 降低为 LLVM dialect                        │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  LLVM 优化                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ LLVM 的标准优化 pass（死代码消除、循环优化等）                    │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  PTX 代码生成                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ LLVM 后端生成 PTX 代码                                          │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  PTX JIT 编译                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ NVIDIA driver 将 PTX 编译为 cubin（GPU 机器码）                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### A.2 Pass 之间的数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Pass 之间的数据流关系                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  输入 IR:                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ %a = tt.load %ptr : tensor<128x128xf16>                        │   │
│  │ %b = tt.load %ptr2 : tensor<128x128xf16>                       │   │
│  │ %c = tt.dot %a, %b : tensor<128x128xf32>                       │   │
│  │ tt.store %ptr3, %c : tensor<128x128xf32>                       │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  After TritonToTritonGPU:                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ %a = tt.load %ptr : tensor<128x128xf16, #blocked>              │   │
│  │ %b = tt.load %ptr2 : tensor<128x128xf16, #blocked>             │   │
│  │ %c = tt.dot %a, %b : tensor<128x128xf32, #blocked>             │   │
│  │ tt.store %ptr3, %c : tensor<128x128xf32, #blocked>             │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  After TritonGPUCombineOps:                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ %a = tt.load %ptr : tensor<128x128xf16, #blocked>              │   │
│  │ %b = tt.load %ptr2 : tensor<128x128xf16, #blocked>             │   │
│  │ %c = tt.dot %a, %b : tensor<128x128xf32, #blocked>             │   │
│  │ tt.store %ptr3, %c : tensor<128x128xf32, #blocked>             │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  After TritonGPURemoveLayoutConversions:                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ %a = tt.load %ptr : tensor<128x128xf16, #blocked>              │   │
│  │ %b = tt.load %ptr2 : tensor<128x128xf16, #blocked>             │   │
│  │ %c = tt.dot %a, %b : tensor<128x128xf32, #blocked>             │   │
│  │ tt.store %ptr3, %c : tensor<128x128xf32, #blocked>             │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  After TritonGPUOptimizeDotOperands:                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ %a = tt.load %ptr : tensor<128x128xf16, #blocked>              │   │
│  │ %b = tt.load %ptr2 : tensor<128x128xf16, #blocked>             │   │
│  │ %a_dot = triton_gpu.convert_layout %a : → #dot_operand         │   │
│  │ %b_dot = triton_gpu.convert_layout %b : → #dot_operand         │   │
│  │ %c = tt.dot %a_dot, %b_dot : tensor<128x128xf32, #mma>         │   │
│  │ tt.store %ptr3, %c : tensor<128x128xf32, #blocked>             │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  After TritonGPUAccelerateMatmul:                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ %a = tt.load %ptr : tensor<128x128xf16, #blocked>              │   │
│  │ %b = tt.load %ptr2 : tensor<128x128xf16, #blocked>             │   │
│  │ %a_dot = triton_gpu.convert_layout %a : → #dot_operand         │   │
│  │ %b_dot = triton_gpu.convert_layout %b : → #dot_operand         │   │
│  │ %c = triton_nvidia_gpu.dot %a_dot, %b_dot : tensor<128x128xf32, #mma>│   │
│  │ tt.store %ptr3, %c : tensor<128x128xf32, #blocked>             │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  After TritonGPUToLLVM:                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ llvm.func @kernel(...) {                                        │   │
│  │   %a = load <128x128 x half>                                   │   │
│  │   %b = load <128x128 x half>                                   │   │
│  │   %c = call @mma.sync(%a, %b)                                  │   │
│  │   store %c                                                      │   │
│  │ }                                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### A.3 Pass 执行的详细时序

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Pass 执行时序分析                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  时间轴 (ms)                                                            │
│  0    5    10   15   20   25   30   35   40   45   50                  │
│  ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤                  │
│  │    │    │    │    │    │    │    │    │    │    │                  │
│  │████│    │    │    │    │    │    │    │    │    │ TritonToGPU       │
│  │    │██  │    │    │    │    │    │    │    │    │ CombineOps        │
│  │    │  ██│    │    │    │    │    │    │    │    │ RemoveLayout      │
│  │    │    │██  │    │    │    │    │    │    │    │ OptimizeDot       │
│  │    │    │  █ │    │    │    │    │    │    │    │ AccelerateMatmul  │
│  │    │    │  ██│    │    │    │    │    │    │    │ Pipeline          │
│  │    │    │    │██  │    │    │    │    │    │    │ ReorderInstr      │
│  │    │    │    │  ██│    │    │    │    │    │    │ OptimizeDot(2)    │
│  │    │    │    │    │████████████████████████████████│ TritonGPUToLLVM │
│  │    │    │    │    │    │    │    │    │    │    │                   │
│  0    1    2    3    5    10   20   30   40   50   60 (ms)            │
│                                                                         │
│  总编译时间: ~60ms                                                      │
│  其中 LLVM 优化: ~45ms (75%)                                            │
│  Triton 优化: ~5ms (8%)                                                 │
│  前端解析: ~10ms (17%)                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 附录 B: 布局编码详解

### B.1 Blocked Encoding

Blocked Encoding 是最基础的数据布局，将张量分割成块分配给不同的线程：

```mlir
// Blocked Encoding 示例
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1],
                                 threadsPerWarp = [32, 1],
                                 warpsPerCTA = [4, 1]}>
// 含义：
// - sizePerThread: 每个线程处理 [1, 1] 个元素
// - threadsPerWarp: 每个 warp 有 [32, 1] 个线程
// - warpsPerCTA: 每个 CTA 有 [4, 1] 个 warp
// - 总线程数: 32 × 4 = 128 个线程
```

**Blocked Encoding 的内存访问模式**：

```
张量形状: 128×128 (FP16)
Blocked Encoding: sizePerThread=[1,1], threadsPerWarp=[32,1], warpsPerCTA=[4,1]

线程分配:
┌─────────────────────────────────────────────────────────────────────┐
│ Thread 0:  elements [0,0]                                          │
│ Thread 1:  elements [0,1]                                          │
│ Thread 2:  elements [0,2]                                          │
│ ...                                                                 │
│ Thread 31: elements [0,31]                                         │
│ Thread 32: elements [1,0]                                          │
│ ...                                                                 │
│ Thread 127: elements [3,31]                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### B.2 Shared Memory Encoding

Shared Memory Encoding 描述数据在共享内存中的布局：

```mlir
// Shared Memory Encoding 示例
#shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4}>
// 含义：
// - vec: 向量化因子，每次加载 8 个元素
// - perPhase: 每个阶段包含 2 个 bank
// - maxPhase: 最大阶段数为 4
```

**Shared Memory Bank 冲突分析**：

```
共享内存 bank 布局 (32 banks):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│  8  │  9  │ 10  │ 11  │ 12  │ 13  │ 14  │ 15  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ 16  │ 17  │ 18  │ 19  │ 20  │ 21  │ 22  │ 23  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ 24  │ 25  │ 26  │ 27  │ 28  │ 29  │ 30  │ 31  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

Swizzle 模式 0 (用于操作数 A):
  Thread 0 访问 bank 0, 1, 2, 3, 4, 5, 6, 7
  Thread 1 访问 bank 8, 9, 10, 11, 12, 13, 14, 15
  Thread 2 访问 bank 16, 17, 18, 19, 20, 21, 22, 23
  Thread 3 访问 bank 24, 25, 26, 27, 28, 29, 30, 31
  → 无 bank 冲突 ✓

Swizzle 模式 1 (用于操作数 B):
  Thread 0 访问 bank 0, 8, 16, 24, 1, 9, 17, 25
  Thread 1 访问 bank 2, 10, 18, 26, 3, 11, 19, 27
  → 无 bank 冲突 ✓
```

### B.3 Dot Operand Encoding

Dot Operand Encoding 描述 Tensor Core 操作数的布局：

```mlir
// Dot Operand Encoding 示例
#dot_operand_a = #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>
// 含义：
// - opIdx: 操作数索引（0=A, 1=B）
// - parent: 父 MMA 编码
```

**Dot Operand 的寄存器分配**：

```
操作数 A (16×8, FP16) 的寄存器分配:
┌─────────────────────────────────────────────────────────────────────┐
│ Thread 0:  R0[0:7]   → 元素 [0:7, 0]                              │
│ Thread 1:  R1[0:7]   → 元素 [0:7, 1]                              │
│ Thread 2:  R2[0:7]   → 元素 [0:7, 2]                              │
│ Thread 3:  R3[0:7]   → 元素 [0:7, 3]                              │
│ Thread 4:  R4[0:7]   → 元素 [8:15, 0]                             │
│ Thread 5:  R5[0:7]   → 元素 [8:15, 1]                             │
│ Thread 6:  R6[0:7]   → 元素 [8:15, 2]                             │
│ Thread 7:  R7[0:7]   → 元素 [8:15, 3]                             │
└─────────────────────────────────────────────────────────────────────┘
```

### B.4 MMA Encoding

MMA Encoding 描述 Tensor Core 计算结果的布局：

```mlir
// MMA Encoding 示例
#mma = #triton_gpu.mma<{version = 2, warpsPerCTA = [4, 1]}>
// 含义：
// - version: MMA 指令版本（2 = mma.sync）
// - warpsPerCTA: 每个 CTA 的 warp 数量
```

**MMA 计算流程**：

```
矩阵乘法 C = A × B 的 MMA 计算流程:

输入:
  A: 16×8 FP16 (Dot Operand A)
  B: 8×8 FP16 (Dot Operand B)
  C: 16×8 FP32 (累加器)

MMA 指令:
  mma.sync.aligned.m16n8k8.row.col.f16.f16.f32.f32
  → 执行 A × B + C → C

输出:
  C: 16×8 FP32 (MMA Encoding)
```

---

## 附录 C: 常见问题与解决方案

### C.1 常见编译错误

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| "layout conversion not supported" | 不支持的布局转换 | 检查输入张量的编码 |
| "dot operand encoding mismatch" | 操作数编码不匹配 | 确保操作数使用正确的编码 |
| "MMA version not supported" | 不支持的 MMA 版本 | 检查 GPU compute capability |
| "pipeline depth exceeded" | 流水线深度超出限制 | 减少 num_stages 参数 |

### C.2 性能调优指南

```
性能调优检查清单:
┌─────────────────────────────────────────────────────────────────────┐
│ 1. 检查 num_stages 参数                                            │
│    - 内存密集型: num_stages=2                                       │
│    - 计算密集型: num_stages=3                                       │
│    - 超大规模: num_stages=4                                         │
├─────────────────────────────────────────────────────────────────────┤
│ 2. 检查 BLOCK_SIZE 参数                                             │
│    - 太小: 无法充分利用 GPU                                         │
│    - 太大: 寄存器压力过大                                           │
│    - 推荐: 128, 256, 512, 1024                                     │
├─────────────────────────────────────────────────────────────────────┤
│ 3. 检查数据类型                                                     │
│    - FP16: 最佳性能                                                 │
│    - BF16: 略低性能                                                 │
│    - FP32: 最低性能                                                 │
├─────────────────────────────────────────────────────────────────────┤
│ 4. 检查内存访问模式                                                 │
│    - 连续访问: 最佳性能                                             │
│    - 随机访问: 性能下降                                             │
│    - 使用向量化加载: 提升带宽利用率                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### C.3 调试技巧汇总

```python
# 调试技巧 1: 打印编译过程中的 IR
import os
os.environ['TRITON_PRINT_IR'] = '1'

# 调试技巧 2: 打印特定 Pass 的 IR
os.environ['TRITON_PRINT_IR_AFTER'] = 'tritongpu-combine-ops'

# 调试技巧 3: 使用 triton-tensor-print 打印张量值
os.environ['TRITON_TENSOR_PRINT'] = '1'

# 调试技巧 4: 启用详细日志
os.environ['TRITON_LOG'] = '1'

# 调试技巧 5: 使用 triton-opt 命令行工具
# triton-opt input.mlir -tritongpu-combine-ops -print-after-all
```

### C.4 性能分析工具

```bash
# 使用 NVIDIA Nsight Compute 分析 kernel 性能
ncu --set full -o report.ncu-rep python my_kernel.py

# 使用 NVIDIA Nsight Systems 分析编译过程
nsys profile -o trace.nsys-rep python my_kernel.py

# 使用 Triton 的内置性能分析
python -m triton.tools.profile my_kernel.py
```

---

## 附录 D: 参考资料

### D.1 官方文档

- MLIR 文档: https://mlir.llvm.org/
- Triton 文档: https://triton-lang.org/
- NVIDIA CUDA 文档: https://docs.nvidia.com/cuda/

### D.2 相关论文

1. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (ASPLOS 2021)
2. "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation" (CGO 2021)
3. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022)

### D.3 源码链接

- Triton 源码: https://github.com/triton-lang/triton
- MLIR 源码: https://github.com/llvm/llvm-project/tree/main/mlir
- NVIDIA PTX 文档: https://docs.nvidia.com/cuda/parallel-thread-execution/

---

## 附录 E: Triton 编译管线的高级主题

### E.1 Pass 管线的并行化

MLIR 的 PassManager 支持并行执行多个不相互依赖的 Pass：

```cpp
// 并行执行 Pass 的配置
mlir::PassManager pm(&context);

// 使用 PipelineDivider 允许并行执行
pm.enableAsync(-1);  // 启用异步执行，-1 表示使用所有可用线程

// 可以并行的 Pass 组合
pm.addPass(mlir::triton::gpu::createTritonGPUCombineOpsPass());
// 以下 Pass 可以并行执行（假设没有依赖）
// pm.addPass(mlir::triton::gpu::createSomeOtherPass());
```

**并行化的限制**：
- 共享 IR 状态的 Pass 不能并行执行
- 有依赖关系的 Pass 必须按顺序执行
- 修改全局状态的 Pass 不能并行执行

### E.2 Pass 管线的增量编译

Triton 支持增量编译，避免重复执行已完成的 Pass：

```cpp
// 增量编译的实现
class IncrementalCompiler {
  // 缓存已编译的模块
  DenseMap<ModuleOp, CompiledModule> cache;

  // 检查是否需要重新编译
  bool needsRecompilation(ModuleOp module, const CompilationConfig &config) {
    // 检查模块是否在缓存中
    if (cache.contains(module)) {
      const auto &cached = cache[module];
      // 检查配置是否变化
      if (cached.config == config) {
        return false;  // 不需要重新编译
      }
    }
    return true;  // 需要重新编译
  }

  // 执行编译
  CompiledModule compile(ModuleOp module, const CompilationConfig &config) {
    if (!needsRecompilation(module, config)) {
      return cache[module];  // 返回缓存结果
    }

    // 执行编译管线
    CompiledModule result = runCompilationPipeline(module, config);

    // 缓存结果
    cache[module] = result;
    return result;
  }
};
```

### E.3 Pass 管线的错误恢复

Triton 的 Pass 管线支持错误恢复机制：

```cpp
// 带错误恢复的 Pass 执行
mlir::LogicalResult runPassWithErrorRecovery(mlir::PassManager &pm,
                                              mlir::ModuleOp module) {
  // 设置错误恢复回调
  pm.setEnableTiming(true);

  // 尝试执行 Pass 管线
  auto result = pm.run(module);

  if (failed(result)) {
    // 记录错误信息
    llvm::errs() << "Pass pipeline failed!\n";

    // 尝试恢复：逐个执行 Pass 并检查错误
    for (auto &pass : pm.getPasses()) {
      mlir::PassManager singlePm(pm.getContext());
      singlePm.addPass(std::move(pass));

      if (failed(singlePm.run(module))) {
        llvm::errs() << "Pass failed: " << pass->getArgument() << "\n";
        // 记录失败的 Pass 以便调试
        recordFailedPass(pass->getArgument());
        return failure();
      }
    }
  }

  return success();
}
```

### E.4 Pass 管线的性能监控

```cpp
// Pass 管线的性能监控
class PassPerformanceMonitor {
  // 记录每个 Pass 的执行时间
  DenseMap<StringRef, double> passTiming;

  // 记录每个 Pass 修改的 IR 操作数
  DenseMap<StringRef, int64_t> operationCounts;

  // 开始监控
  void startMonitoring(mlir::PassManager &pm) {
    pm.enableTiming();  // 启用计时

    // 为每个 Pass 添加回调
    pm.addInstrumentation(std::make_unique<PassTimingInstrumentation>(*this));
  }

  // 生成性能报告
  void generateReport() {
    llvm::outs() << "=== Pass Performance Report ===\n";
    for (auto &[passName, timing] : passTiming) {
      llvm::outs() << passName << ": " << timing << " ms\n";
    }
    llvm::outs() << "================================\n";
  }

  // 内部类：Pass 计时 instrumentation
  struct PassTimingInstrumentation : public mlir::PassInstrumentation {
    PassPerformanceMonitor &monitor;

    PassTimingInstrumentation(PassPerformanceMonitor &m) : monitor(m) {}

    void runBeforePass(mlir::Pass *pass, mlir::Operation *) override {
      // 记录开始时间
      startTime = std::chrono::high_resolution_clock::now();
    }

    void runAfterPass(mlir::Pass *pass, mlir::Operation *) override {
      // 记录结束时间并计算耗时
      auto endTime = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double, std::milli>(
          endTime - startTime).count();

      monitor.passTiming[pass->getArgument()] += duration;
    }

  private:
    std::chrono::high_resolution_clock::time_point startTime;
  };
};
```

### E.5 Pass 管线的配置选项

Triton 的 Pass 管线支持多种配置选项：

```python
# Python 前端的配置选项
import triton

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE': 256, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE': 512, 'num_stages': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(..., BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    # kernel 代码
    pass
```

**配置选项说明**：

| 选项 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| BLOCK_SIZE | Tile 大小 | 128 | 128-512 |
| num_stages | 流水线深度 | 2 | 2-4 |
| num_warps | 每个 CTA 的 warp 数 | 4 | 4-8 |
| num_ctas | CTA 数量 | 1 | 1-4 |

### E.6 Pass 管线的调试模式

```cpp
// 启用调试模式
mlir::PassManager pm(&context);

// 启用 IR 打印
pm.enableIRPrinting(
    /*shouldPrintBeforePass=*/[](auto *) { return true; },
    /*shouldPrintAfterPass=*/[](auto *) { return true; },
    /*printModuleScope=*/true,
    /*printAfterOnlyOnChange=*/false
);

// 启用计时
pm.enableTiming();

// 启用验证
pm.enableVerifier();

// 设置验证级别
pm.getContext()->disableMultithreading();
```

### E.7 Pass 管线的测试框架

```cpp
// Pass 的单元测试框架
class PassTest : public ::testing::Test {
protected:
  void SetUp() override {
    // 初始化 MLIR 上下文
    context = std::make_unique<mlir::MLIRContext>();
    context->loadDialect<mlir::triton::TritonDialect>();
    context->loadDialect<mlir::triton::gpu::TritonGPUDialect>();
  }

  // 测试 Pass 的正确性
  void testPass(mlir::ModuleOp input, mlir::ModuleOp expected,
                std::function<void(mlir::PassManager &)> addPass) {
    mlir::PassManager pm(context.get());
    addPass(pm);

    // 执行 Pass
    auto result = pm.run(input);
    ASSERT_TRUE(succeeded(result));

    // 比较结果
    EXPECT_TRUE(llvm::is_sorted(input, [](auto &a, auto &b) {
      return a.isIdenticalTo(b);
    }));
  }

private:
  std::unique_ptr<mlir::MLIRContext> context;
};

// 测试 CombineOps Pass
TEST_F(PassTest, CombineOps) {
  // 创建输入 IR
  auto input = parseMLIR(R"(
    module {
      func @test(%a: tensor<128x128xf16, #blocked>) -> tensor<128x128xf16, #shared> {
        %shared = triton_gpu.local_alloc %a : (tensor<128x128xf16, #blocked>) -> tensor<128x128xf16, #shared>
        %loaded = triton_gpu.local_load %shared : (tensor<128x128xf16, #shared>) -> tensor<128x128xf16, #blocked>
        return %loaded : tensor<128x128xf16, #blocked>
      }
    }
  )");

  // 创建预期输出
  auto expected = parseMLIR(R"(
    module {
      func @test(%a: tensor<128x128xf16, #blocked>) -> tensor<128x128xf16, #blocked> {
        return %a : tensor<128x128xf16, #blocked>
      }
    }
  )");

  // 测试 Pass
  testPass(input, expected, [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::gpu::createTritonGPUCombineOpsPass());
  });
}
```

### E.8 Pass 管线的扩展性考虑

在设计 Pass 管线时，需要考虑以下扩展性因素：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Pass 管线扩展性考虑                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. 硬件扩展性                                                          │
│     ┌─────────────────────────────────────────────────────────────┐     │
│     │ - 支持新的 GPU 架构（如 Blackwell）                         │     │
│     │ - 支持新的数据类型（如 FP8, INT4）                          │     │
│     │ - 支持新的指令集（如新的 Tensor Core 指令）                 │     │
│     └─────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  2. 算法扩展性                                                          │
│     ┌─────────────────────────────────────────────────────────────┐     │
│     │ - 支持新的优化算法（如新的布局优化策略）                     │     │
│     │ - 支持新的计算模式（如稀疏计算、混合精度）                   │     │
│     │ - 支持新的内存管理策略（如显存池化、异步分配）               │     │
│     └─────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  3. 接口扩展性                                                          │
│     ┌─────────────────────────────────────────────────────────────┐     │
│     │ - 支持新的前端语言（如 Julia, Rust）                        │     │
│     │ - 支持新的后端（如 AMD GPU, Intel GPU）                     │     │
│     │ - 支持新的工具链集成（如 Jupyter, VS Code）                 │     │
│     └─────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### E.9 Pass 管线的未来发展方向

Triton 编译管线的未来发展方向包括：

1. **自动调优**：基于机器学习的 Pass 选择和参数调优
2. **分布式编译**：支持多 GPU、多节点的分布式编译
3. **实时编译**：支持运行时动态生成和优化 kernel
4. **混合精度优化**：自动选择最优的数据精度组合
5. **内存优化**：更智能的显存分配和复用策略
6. **调试工具**：更强大的 IR 可视化和性能分析工具

这些发展方向将进一步提升 Triton 编译器的性能和易用性，使其成为 GPU 编程的主流工具之一。