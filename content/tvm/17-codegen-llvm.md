> **学习目标**：
> - 理解 TVM 中 LLVM 代码生成后端的架构设计
> - 掌握 TIR 到 LLVM IR 的映射规则
> - 理解向量化指令生成（SSE/AVX/NEON）的实现机制
> - 掌握 CPU 调优策略与 LLVM 优化管线的集成
> - 了解 LLVM CodeGen 的源码结构与扩展点

---

## 17.1 LLVM 在 TVM 中的角色

### 17.1.1 为什么选择 LLVM

LLVM 是 TVM 最重要的 CPU 后端，也是许多其他后端的基础。选择 LLVM 的原因：

1. **成熟的优化管线**：LLVM 提供了数百个经过验证的优化 Pass
2. **广泛的硬件支持**：x86、ARM、RISC-V、PowerPC 等主流 CPU 架构
3. **向量化能力**：自动向量化和显式 SIMD 指令支持（SSE/AVX/NEON）
4. **JIT 编译**：支持运行时即时编译，减少部署开销
5. **社区活跃**：持续的更新和优化

```
┌─────────────────────────────────────────────────────┐
│                    TVM 编译流程                       │
│                                                     │
│  TIR Program → LLVM CodeGen → LLVM IR → 机器码      │
│       │              │           │          │        │
│       │         TIR→IR映射    LLVM Pass   目标代码   │
│       │         指令选择      管线优化     生成       │
└─────────────────────────────────────────────────────┘
```

### 17.1.2 LLVM CodeGen 的源码结构

TVM 的 LLVM CodeGen 实现位于以下目录：

```
src/target/llvm/
├── codegen_llvm.cc           # 核心：TIR→LLVM IR 翻译
├── codegen_llvm.h            # CodeGenLLVM 类定义
├── llvm_module.cc            # LLVM Module 封装
├── llvm_common.cc            # LLVM 通用工具函数
├── llvm_node.cc              # LLVM 节点注册
├── codegen_x86_64.cc         # x86-64 特定优化
├── codegen_aarch64.cc        # AArch64 特定优化
├── codegen_arm.cc            # ARM 32-bit 特定优化
├── codegen_blob.cc           # Blob 代码生成
├── codegen_params.cc         # 常量参数处理
└── codegen_cpu.cc            # CPU 通用代码生成

include/tvm/target/
└── codegen.h                 # CodeGen 接口定义

src/target/
├── codegen.cc                # CodeGen 注册与管理
└── target.cc                 # Target 系统
```

### 17.1.3 CodeGenLLVM 类层次

```cpp
// src/target/llvm/codegen_llvm.h
class CodeGenLLVM : public codegen::CodegenCBase {
 public:
  // 初始化
  void Init(const std::string& module_name,
            const LLVMTarget& target,
            bool system_lib);

  // 主入口：编译 TIR 函数到 LLVM IR
  virtual llvm::Function* AddFunction(const tir::PrimFunc& f);

  // 编译 TIR 表达式
  llvm::Value* CreateLLVMValue(const tir::PrimExpr& expr);

  // 编译 TIR 语句
  void VisitStmt(const tir::Stmt& stmt) override;

  // 编译 TIR 表达式
  void VisitExpr(const tir::PrimExpr& expr) override;

 protected:
  // LLVM 核心对象
  llvm::LLVMContext* ctx_;
  llvm::Module* module_;
  llvm::IRBuilder<>* builder_;
  llvm::TargetMachine* target_machine_;

  // 符号表
  std::unordered_map<const tir::VarNode*, llvm::Value*> var_map_;
  std::unordered_map<std::string, llvm::GlobalVariable*> global_var_map_;

  // 类型映射
  llvm::Type* LLVMType(const DataType& dtype);

  // 向量化支持
  llvm::Value* CreateVecBroadcast(llvm::Value* value, int lanes);
  llvm::Value* CreateVecLoad(llvm::Value* buffer, llvm::Value* index);
  void CreateVecStore(llvm::Value* buffer, llvm::Value* index, llvm::Value* value);
};

// x86-64 特化的 CodeGen
class CodeGenX86_64 : public CodeGenLLVM {
  // 使用 AVX/SSE 指令优化
};

// AArch64 特化的 CodeGen
class CodeGenAArch64 : public CodeGenLLVM {
  // 使用 NEON 指令优化
};
```

---

## 17.2 TIR 到 LLVM IR 的映射

### 17.2.1 数据类型映射

TIR 的数据类型直接映射到 LLVM 的类型系统：

| TIR DataType | LLVM Type | 说明 |
|-------------|-----------|------|
| `int8` | `i8` | 8 位整数 |
| `int16` | `i16` | 16 位整数 |
| `int32` | `i32` | 32 位整数 |
| `int64` | `i64` | 64 位整数 |
| `float16` | `half` | 半精度浮点 |
| `float32` | `float` | 单精度浮点 |
| `float64` | `double` | 双精度浮点 |
| `bool` | `i1` | 布尔值 |
| `int8x4` | `<4 x i8>` | 4 路 int8 向量 |
| `float32x4` | `<4 x float>` | 4 路 float32 向量 |

```cpp
// src/target/llvm/codegen_llvm.cc
llvm::Type* CodeGenLLVM::LLVMType(const DataType& dtype) {
  if (dtype.is_vector()) {
    llvm::Type* element_type = LLVMType(dtype.with_lanes(1));
    return llvm::VectorType::get(element_type, dtype.lanes());
  }

  switch (dtype.code()) {
    case DataType::kInt:
      return llvm::Type::getIntNTy(*ctx_, dtype.bits());
    case DataType::kUInt:
      return llvm::Type::getIntNTy(*ctx_, dtype.bits());
    case DataType::kFloat:
      switch (dtype.bits()) {
        case 16: return llvm::Type::getHalfTy(*ctx_);
        case 32: return llvm::Type::getFloatTy(*ctx_);
        case 64: return llvm::Type::getDoubleTy(*ctx_);
      }
    case DataType::kBFloat:
      return llvm::Type::getBFloatTy(*ctx_);
  }
  LOG(FATAL) << "Cannot convert type " << dtype;
  return nullptr;
}
```

### 17.2.2 Buffer 映射

TIR 的 Buffer 被映射为 LLVM 的指针参数：

```cpp
// TIR 的 Buffer 参数映射到 LLVM 函数参数
llvm::Function* CodeGenLLVM::AddFunction(const tir::PrimFunc& f) {
  // 1. 构建函数参数类型列表
  std::vector<llvm::Type*> arg_types;

  for (const auto& param : f->params) {
    tir::Buffer buf = f->buf_map[param];
    if (buf.defined()) {
      // Buffer 参数映射为指向元素类型的指针
      llvm::Type* elem_type = LLVMType(buf->dtype);
      arg_types.push_back(llvm::PointerType::get(elem_type, 0));
    } else {
      // 标量参数直接映射
      arg_types.push_back(LLVMType(param->dtype));
    }
  }

  // 2. 创建 LLVM 函数
  llvm::FunctionType* func_type =
      llvm::FunctionType::get(llvm::Type::getVoidTy(*ctx_), arg_types, false);

  llvm::Function* function = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, f->name, module_);

  // 3. 设置参数名称并加入符号表
  auto it = function->arg_begin();
  for (size_t i = 0; i < f->params.size(); i++, ++it) {
    it->setName(f->params[i]->name_hint);
    var_map_[f->params[i].get()] = &(*it);
  }

  return function;
}
```

### 17.2.3 循环映射

TIR 的循环语句映射为 LLVM 的基本块和分支指令：

```
TIR: for i in range(0, N):
         body()

LLVM:
  loop_header:
    %i = phi i32 [0, %entry], [%i_next, %loop_body]
    %cond = icmp slt i32 %i, N
    br i1 %cond, label %loop_body, label %loop_end

  loop_body:
    ; body code
    %i_next = add i32 %i, 1
    br label %loop_header

  loop_end:
    ; continue
```

```cpp
void CodeGenLLVM::VisitStmt(const tir::ForNode* op) {
  // 获取循环边界
  llvm::Value* begin = CreateLLVMValue(op->min);
  llvm::Value* end = CreateLLVMValue(op->min + op->extent);

  // 创建基本块
  llvm::BasicBlock* preheader = builder_->GetInsertBlock();
  llvm::BasicBlock* loop_header = llvm::BasicBlock::Create(
      *ctx_, "loop_header", function_);
  llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(
      *ctx_, "loop_body", function_);
  llvm::BasicBlock* loop_end = llvm::BasicBlock::Create(
      *ctx_, "loop_end", function_);

  // 跳转到循环头
  builder_->CreateBr(loop_header);

  // 循环头：phi 节点 + 条件判断
  builder_->SetInsertPoint(loop_header);
  llvm::PHINode* loop_var = builder_->CreatePHI(
      llvm::Type::getInt32Ty(*ctx_), 2, op->loop_var->name_hint);
  loop_var->addIncoming(begin, preheader);

  llvm::Value* cond = builder_->CreateICmpSLT(loop_var, end);
  builder_->CreateCondBr(cond, loop_body, loop_end);

  // 循环体：执行 body
  builder_->SetInsertPoint(loop_body);
  var_map_[op->loop_var.get()] = loop_var;
  VisitStmt(op->body);

  // 循环递增
  llvm::Value* next = builder_->CreateAdd(
      loop_var, llvm::ConstantInt::get(loop_var->getType(), 1));
  loop_var->addIncoming(next, builder_->GetInsertBlock());
  builder_->CreateBr(loop_header);

  // 循环结束
  builder_->SetInsertPoint(loop_end);
}
```

### 17.2.4 Buffer 读写映射

TIR 的 Buffer Load/Store 映射为 LLVM 的 load/store 指令：

```cpp
// Buffer Load
void CodeGenLLVM::VisitExpr(const tir::BufferLoadNode* op) {
  // 计算内存地址
  llvm::Value* buffer = GetVar(op->buffer->data.get());
  llvm::Value* index = CreateLLVMValue(op->indices[0]);

  // 对于多维索引，计算线性偏移
  for (size_t i = 1; i < op->indices.size(); i++) {
    llvm::Value* stride = CreateLLVMValue(op->buffer->strides[i]);
    llvm::Value* offset = CreateLLVMValue(op->indices[i]);
    index = builder_->CreateAdd(index, builder_->CreateMul(offset, stride));
  }

  // 计算指针偏移
  llvm::Value* ptr = builder_->CreateGEP(
      LLVMType(op->buffer->dtype), buffer, index);

  // 生成 load 指令
  llvm::Value* value = builder_->CreateAlignedLoad(
      LLVMType(op->dtype), ptr,
      llvm::Align(op->dtype.bytes()));

  // 设置对齐信息
  SetVar(op.get(), value);
}

// Buffer Store
void CodeGenLLVM::VisitStmt(const tir::BufferStoreNode* op) {
  llvm::Value* buffer = GetVar(op->buffer->data.get());
  llvm::Value* index = CreateLLVMValue(op->indices[0]);
  llvm::Value* value = CreateLLVMValue(op->value);

  // 计算线性偏移（同上）
  for (size_t i = 1; i < op->indices.size(); i++) {
    // ...
  }

  llvm::Value* ptr = builder_->CreateGEP(
      LLVMType(op->buffer->dtype), buffer, index);

  // 生成 store 指令
  builder_->CreateAlignedStore(value, ptr,
      llvm::Align(op->buffer->dtype.bytes()));
}
```

### 17.2.5 完整的 TIR→LLVM IR 映射表

| TIR 节点 | LLVM IR | 说明 |
|---------|---------|------|
| `For` | `br` + `phi` + `icmp` | 循环头/体/尾基本块 |
| `IfThenElse` | `br i1 cond, then, else` | 条件分支 |
| `BufferLoad` | `load` | 内存读取 |
| `BufferStore` | `store` | 内存写入 |
| `Add` | `add` / `fadd` | 整数/浮点加法 |
| `Mul` | `mul` / `fmul` | 整数/浮点乘法 |
| `Div` | `sdiv` / `fdiv` | 整数/浮点除法 |
| `Call("sqrt")` | `call @sqrtf` | 函数调用 |
| `Let` | `alloca` + `store` | 局部变量 |
| `Allocate` | `alloca` | 局部内存分配 |
| `AttrStmt` | metadata | 循环注释/pragma |
| `IntImm` | `i32 42` | 整数常量 |
| `FloatImm` | `float 3.14` | 浮点常量 |

---

## 17.3 向量化指令生成

### 17.3.1 SIMD 向量化概述

现代 CPU 支持 SIMD（Single Instruction, Multiple Data）指令集，可以一次处理多个数据元素：

| 指令集 | 寄存器宽度 | float32 路数 | 架构 |
|--------|-----------|-------------|------|
| SSE | 128-bit | 4 | x86/x86-64 |
| AVX2 | 256-bit | 8 | x86-64 |
| AVX-512 | 512-bit | 16 | x86-64 (服务器) |
| NEON | 128-bit | 4 | ARM/AArch64 |
| SVE | 128-2048-bit | 4-64 | AArch64 (可变长度) |

TVM 通过两种方式利用 SIMD 指令：

1. **显式向量化**：在 TIR 中使用 `vectorize` 标注，直接生成向量指令
2. **LLVM 自动向量化**：依赖 LLVM 的 SLP/Loop Vectorizer 自动向量化

### 17.3.2 TIR 向量化的 LLVM 代码生成

当 TIR 中包含向量化循环时，CodeGenLLVM 直接生成向量类型的 LLVM IR：

```cpp
// 向量化循环的代码生成
void CodeGenLLVM::VisitStmt(const tir::ForNode* op) {
  if (op->kind == tir::ForKind::kVectorized) {
    // 向量化循环：生成向量类型的运算
    int vector_width = op->extent.as<IntImmNode>()->value;
    llvm::Type* vec_type = llvm::VectorType::get(
        LLVMType(op->loop_var->dtype), vector_width, false);

    // 生成 SIMD 指令
    for (int i = 0; i < vector_width; i++) {
      // 使用 shufflevector / extractelement 操作
    }
  } else {
    // 普通循环：标准的 LLVM IR 生成
    GenerateStandardLoop(op);
  }
}
```

**TIR 向量化示例**：

```python
# TIR 中的向量化循环
@T.prim_func
def vector_add(
    A: T.Buffer[(1024,), "float32"],
    B: T.Buffer[(1024,), "float32"],
    C: T.Buffer[(1024,), "float32"],
) -> None:
    for i in T.serial(256):  # 1024 / 4 = 256 次迭代
        with T.block("C"):
            vi = T.axis.spatial(256, i)
            T.reads(A[vi * 4 : vi * 4 + 4], B[vi * 4 : vi * 4 + 4])
            T.writes(C[vi * 4 : vi * 4 + 4])
            # 向量化加载
            A_vec = T.tvm_load_vector(
                A.data, vi * 4, 4, dtype="float32x4")
            B_vec = T.tvm_load_vector(
                B.data, vi * 4, 4, dtype="float32x4")
            C_vec = A_vec + B_vec
            T.tvm_store_vector(
                C.data, vi * 4, C_vec, 4, dtype="float32x4")
```

**生成的 LLVM IR**：

```llvm
; 循环体（向量化版本）
loop_body:
  %i = phi i32 [0, %entry], [%i_next, %loop_body]

  ; 向量加载 A[i:i+4]
  %ptr_a = getelementptr float, float* %A, i32 %i
  %a_vec = load <4 x float>, <4 x float>* %ptr_a, align 16

  ; 向量加载 B[i:i+4]
  %ptr_b = getelementptr float, float* %B, i32 %i
  %b_vec = load <4 x float>, <4 x float>* %ptr_b, align 16

  ; 向量加法（单条 SIMD 指令）
  %c_vec = fadd <4 x float> %a_vec, %b_vec

  ; 向量存储 C[i:i+4]
  %ptr_c = getelementptr float, float* %C, i32 %i
  store <4 x float> %c_vec, <4 x float>* %ptr_c, align 16

  ; 循环递增
  %i_next = add i32 %i, 4
  %cond = icmp slt i32 %i_next, 1024
  br i1 %cond, label %loop_body, label %loop_end
```

### 17.3.3 x86-64 特定优化（SSE/AVX）

`CodeGenX86_64` 类在基础的 `CodeGenLLVM` 上增加了 x86-64 特定的优化：

```cpp
// src/target/llvm/codegen_x86_64.cc
class CodeGenX86_64 : public CodeGenLLVM {
 protected:
  // 使用 AVX intrinsics 进行向量操作
  llvm::Value* CreateVecAdd(llvm::Value* a, llvm::Value* b) override {
    int lanes = a->getType()->getVectorNumElements();
    if (lanes >= 8 && target_->GetFeature("avx2")) {
      // 使用 AVX2 指令
      return builder_->CreateCall(
          GetIntrinsic("llvm.x86.avx2.add.ps.256"), {a, b});
    } else if (lanes >= 4 && target_->GetFeature("sse")) {
      // 使用 SSE 指令
      return builder_->CreateCall(
          GetIntrinsic("llvm.x86.sse.add.ps.128"), {a, b});
    }
    return CodeGenLLVM::CreateVecAdd(a, b);
  }

  // 使用 FMA 指令（融合乘加）
  llvm::Value* CreateFMA(llvm::Value* a, llvm::Value* b,
                          llvm::Value* c) override {
    if (target_->GetFeature("fma")) {
      return builder_->CreateCall(
          GetIntrinsic("llvm.fma.f32"), {a, b, c});
    }
    return builder_->CreateFAdd(builder_->CreateFMul(a, b), c);
  }

  // 设置 CPU 特性
  void InitTarget() override {
    std::string cpu = target_->GetAttr<String>("mcpu");
    std::string features = target_->GetAttr<String>("mattr");

    // 设置 LLVM target triple
    module_->setTargetTriple("x86_64-unknown-linux-gnu");

    // 设置 CPU 特性
    target_machine_ = GetTargetMachine(cpu, features);
  }
};
```

**AVX-512 的特殊处理**：

```cpp
// AVX-512 向量化（512-bit = 16 个 float32）
void CodeGenX86_64::GenerateAVX512Loop(const tir::ForNode* op) {
  int vector_width = 16;  // 512 / 32 = 16

  // 生成 512-bit 向量类型的 LLVM IR
  llvm::Type* vec_type = llvm::VectorType::get(
      llvm::Type::getFloatTy(*ctx_), vector_width, false);

  // 使用掩码操作处理尾部元素
  llvm::Value* mask = builder_->CreateICmpULT(
      loop_var, llvm::ConstantInt::get(i32_ty, remainder));

  // 生成掩码加载/存储
  // call <16 x float> @llvm.masked.load.v16f32(...)
  // call void @llvm.masked.store.v16f32(...)
}
```

### 17.3.4 AArch64 特定优化（NEON）

```cpp
// src/target/llvm/codegen_aarch64.cc
class CodeGenAArch64 : public CodeGenLLVM {
 protected:
  // 使用 NEON intrinsics
  llvm::Value* CreateVecAdd(llvm::Value* a, llvm::Value* b) override {
    int lanes = a->getType()->getVectorNumElements();
    if (lanes == 4 && target_->GetFeature("neon")) {
      // 使用 NEON 的 128-bit 向量加法
      return builder_->CreateCall(
          GetIntrinsic("llvm.aarch64.neon.add.v4f32"), {a, b});
    }
    return CodeGenLLVM::CreateVecAdd(a, b);
  }

  // NEON 特有的向量乘累加
  llvm::Value* CreateVMLA(llvm::Value* acc, llvm::Value* a,
                           llvm::Value* b) override {
    if (target_->GetFeature("neon")) {
      return builder_->CreateCall(
          GetIntrinsic("llvm.fma.v4f32"), {a, b, acc});
    }
    return builder_->CreateFAdd(acc, builder_->CreateFMul(a, b));
  }

  // SVE（可变长度向量）支持
  void GenerateSVELoop(const tir::ForNode* op) {
    // SVE 使用可变长度向量，通过 vscale 参数确定
    // <vscale x 4 x float> 类型
    llvm::Type* sve_type = llvm::ScalableVectorType::get(
        llvm::Type::getFloatTy(*ctx_), 4);
    // ...
  }
};
```

### 17.3.5 向量化的约束与挑战

向量化面临的主要挑战：

**对齐约束**：向量加载/存储要求内存地址对齐到向量宽度

```cpp
// 检查对齐
void CodeGenLLVM::EmitVectorLoad(llvm::Value* buffer,
                                  llvm::Value* index,
                                  int vector_width) {
  // 计算对齐
  unsigned alignment = vector_width * dtype.bytes();

  // 如果不能保证对齐，使用非对齐加载
  if (!IsAligned(index, alignment)) {
    // 使用普通 load（非对齐），性能较低
    load->setAlignment(llvm::Align(1));
  } else {
    load->setAlignment(llvm::Align(alignment));
  }
}
```

**尾部元素处理**：当循环长度不能被向量宽度整除时，需要处理尾部元素

```
循环长度 N = 1023，向量宽度 W = 8
主体循环：1023 / 8 = 127 次迭代，处理 1016 个元素
尾部处理：1023 - 1016 = 7 个元素，用标量处理
```

---

## 17.4 LLVM 优化管线集成

### 17.4.1 LLVM Pass 管线

TVM 生成 LLVM IR 后，会运行 LLVM 的标准优化管线：

```cpp
// src/target/llvm/llvm_module.cc
void LLVMModuleNode::Optimize() {
  // 创建 LLVM Pass Manager
  llvm::legacy::PassManager pass_manager;

  // 设置优化级别
  llvm::PassBuilder pass_builder;
  llvm::OptimizationLevel opt_level = llvm::OptimizationLevel::O3;

  // 注册标准 Pass 管线
  pass_builder.buildPerModuleDefaultPipeline(
      opt_level, llvm::PassBuilder::ThinLTOPreLink,
      pass_manager);

  // 运行优化
  pass_manager.run(*module_);
}
```

### 17.4.2 TVM 特定的 LLVM Pass

除了 LLVM 的标准 Pass，TVM 还注册了一些自定义 Pass：

```cpp
// TVM 的 LLVM Pass 注册
class TVMAddLikelyBranchWeights : public llvm::FunctionPass {
  static char ID;
  TVMAddLikelyBranchWeights() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function& F) override {
    // 为循环的分支添加 likely 权重
    // 提示 CPU 分支预测器正确预测
    for (auto& BB : F) {
      if (IsLoopHeader(BB)) {
        AddBranchWeight(&BB, /*likely=*/100, /*unlikely=*/1);
      }
    }
    return true;
  }
};
```

### 17.4.3 目标特化

TVM 根据 Target 配置设置 LLVM 的目标特化参数：

```python
# Target 配置示例
target = tvm.target.Target("llvm -mcpu=skylake -mattr=+avx2,+fma")

# 对应的 LLVM 参数
# -mtriple=x86_64-unknown-linux-gnu
# -mcpu=skylake
# -mattr=+avx2,+fma
```

```cpp
void CodeGenLLVM::InitTarget(const Target& target) {
  // 解析 Target 属性
  std::string triple = target->GetAttr<String>("mtriple");
  std::string cpu = target->GetAttr<String>("mcpu");
  std::string attrs = target->GetAttr<String>("mattr");

  // 设置 LLVM 模块属性
  module_->setTargetTriple(triple);

  // 创建 TargetMachine
  auto target_ref = llvm::TargetRegistry::lookupTarget(triple);
  target_machine_ = target_ref->createTargetMachine(
      triple, cpu, attrs,
      llvm::TargetOptions(),
      llvm::Reloc::PIC_,
      llvm::CodeModel::Small);
}
```

### 17.4.4 LLVM 优化 Pass 的效果

LLVM 的优化 Pass 管线对 TVM 生成的代码质量有显著影响：

| LLVM Pass | 效果 | TVM 中的作用 |
|-----------|------|-------------|
| SLP Vectorizer | 自动向量化 | 增强 SIMD 利用率 |
| Loop Vectorizer | 循环向量化 | 自动向量化简单循环 |
| Loop Unroll | 循环展开 | 减少分支开销 |
| GVN | 全局值编号 | 消除冗余计算 |
| LICM | 循环不变代码外提 | 减少循环内计算 |
| Mem2Reg | 内存到寄存器提升 | 消除不必要的内存访问 |
| InstCombine | 指令合并 | 简化表达式 |
| DeadCodeElim | 死代码消除 | 移除无用代码 |

---

## 17.5 CPU 运行时支持

### 17.5.1 多线程并行

TVM 使用 OpenMP 或自定义的线程池来实现 CPU 多线程并行：

```cpp
// TIR 中的并行循环
// for i in T.parallel(0, N):
//     body()

// 生成的 LLVM IR（使用 TVM 的并行运行时）
void CodeGenLLVM::VisitStmt(const tir::ForNode* op) {
  if (op->kind == tir::ForKind::kParallel) {
    // 调用 TVM 的并行 for 运行时
    llvm::Value* begin = CreateLLVMValue(op->min);
    llvm::Value* extent = CreateLLVMValue(op->extent);

    // 生成对 TVMParallelLaunch 的调用
    llvm::Function* parallel_launch = GetRuntimeFunction("TVMBackendParallelLaunch");

    // 创建 lambda 函数作为并行体
    llvm::Function* parallel_body = CreateParallelBody(op->body);

    builder_->CreateCall(parallel_launch, {parallel_body, begin, extent});
  }
}
```

```cpp
// src/runtime/threading_backend.cc
// TVM 的线程池实现
class ThreadPool {
 public:
  void Launch(FTVMParallelLambda flambda, void* cdata, int num_tasks) {
    // 将任务分发到线程池中的工作线程
    for (int i = 0; i < num_workers_; i++) {
      workers_[i]->Submit(flambda, cdata, i, num_tasks);
    }
    // 等待所有任务完成
    WaitForAll();
  }
};
```

### 17.5.2 调用约定

TVM 生成的函数遵循特定的调用约定：

```cpp
// TVM 的函数调用约定
// 参数传递：
//   - 前 6 个参数通过寄存器传递（x86-64 ABI）
//   - 大型 buffer 通过指针传递
//   - 返回值通过寄存器返回

// 内存管理：
//   - 局部变量通过 alloca 分配
//   - 临时 buffer 通过 TVMBackendAllocWorkspace 分配
//   - 通过 TVMBackendFreeWorkspace 释放
```

### 17.5.3 常量池管理

大型常量（如模型权重）通过常量池管理，避免重复嵌入到代码中：

```cpp
// src/target/llvm/codegen_params.cc
class ConstantPool {
 public:
  // 将常量数据注册到常量池
  llvm::GlobalVariable* RegisterConstant(const std::string& name,
                                          const void* data,
                                          size_t size) {
    // 创建 LLVM 全局变量
    llvm::Constant* init = CreateConstantData(data, size);
    llvm::GlobalVariable* global = new llvm::GlobalVariable(
        *module_, init->getType(), true,
        llvm::GlobalValue::PrivateLinkage, init, name);

    // 设置对齐
    global->setAlignment(llvm::Align(64));  // 缓存行对齐

    return global;
  }
};
```

---

## 17.6 JIT 编译与代码缓存

### 17.6.1 LLVM JIT 编译

TVM 支持通过 LLVM 的 ORC JIT 进行运行时编译：

```cpp
// src/target/llvm/llvm_module.cc
class LLVMJIT {
 public:
  void Compile(llvm::Module* module) {
    // 初始化 JIT
    auto jit = llvm::orc::LLJITBuilder()
        .setDataLayout(module->getDataLayout())
        .create();

    // 添加模块
    jit->addIRModule(
        llvm::orc::ThreadSafeModule(std::move(module), ctx_));

    // 查找函数
    auto func = jit->lookup("main");
    auto* func_ptr = func->toPtr<void(*)(float*, float*, float*)>();

    // 调用编译后的函数
    func_ptr(A, B, C);
  }
};
```

### 17.6.2 代码缓存

TVM 支持将编译后的目标代码缓存到磁盘，避免重复编译：

```python
# 编译并缓存
lib = tvm.build(sch.mod, target="llvm")
lib.export_library("compiled_lib.tar")

# 从缓存加载
lib = tvm.runtime.load_module("compiled_lib.tar")
```

---

## 17.7 源码走读：关键函数

### 17.7.1 CodeGenLLVM::AddFunction

```cpp
// src/target/llvm/codegen_llvm.cc
llvm::Function* CodeGenLLVM::AddFunction(const tir::PrimFunc& f) {
  // 1. 准备阶段
  this->Init(function_name, target, system_lib);

  // 2. 构建函数签名
  std::vector<llvm::Type*> arg_types = BuildArgTypes(f);
  llvm::FunctionType* func_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(*ctx_), arg_types, false);

  // 3. 创建函数
  llvm::Function* function = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, f->name, module_);

  // 4. 设置参数
  SetupFunctionArgs(function, f);

  // 5. 创建入口基本块
  llvm::BasicBlock* entry = llvm::BasicBlock::Create(
      *ctx_, "entry", function);
  builder_->SetInsertPoint(entry);

  // 6. 分配局部变量
  AllocateLocalVars(f);

  // 7. 生成函数体
  VisitStmt(f->body);

  // 8. 添加返回语句
  builder_->CreateRetVoid();

  // 9. 验证函数
  llvm::verifyFunction(*function);

  return function;
}
```

### 17.7.2 CodeGenLLVM::VisitExpr（二元运算）

```cpp
void CodeGenLLVM::VisitExpr(const tir::AddNode* op) {
  llvm::Value* a = CreateLLVMValue(op->a);
  llvm::Value* b = CreateLLVMValue(op->b);

  if (op->dtype.is_float()) {
    SetVar(op, builder_->CreateFAdd(a, b, "fadd"));
  } else {
    SetVar(op, builder_->CreateAdd(a, b, "add"));
  }
}

void CodeGenLLVM::VisitExpr(const tir::MulNode* op) {
  llvm::Value* a = CreateLLVMValue(op->a);
  llvm::Value* b = CreateLLVMValue(op->b);

  if (op->dtype.is_float()) {
    SetVar(op, builder_->CreateFMul(a, b, "fmul"));
  } else {
    SetVar(op, builder_->CreateMul(a, b, "mul"));
  }
}

void CodeGenLLVM::VisitExpr(const tir::DivNode* op) {
  llvm::Value* a = CreateLLVMValue(op->a);
  llvm::Value* b = CreateLLVMValue(op->b);

  if (op->dtype.is_float()) {
    SetVar(op, builder_->CreateFDiv(a, b, "fdiv"));
  } else if (op->dtype.is_int()) {
    SetVar(op, builder_->CreateSDiv(a, b, "sdiv"));
  } else {
    SetVar(op, builder_->CreateUDiv(a, b, "udiv"));
  }
}
```

### 17.7.3 CodeGenLLVM::VisitExpr（特殊函数）

```cpp
void CodeGenLLVM::VisitExpr(const tir::CallNode* op) {
  // 内置数学函数
  if (op->op.same_as(tir::builtin::tvm_if_then_else())) {
    // 条件表达式
    llvm::Value* cond = CreateLLVMValue(op->args[0]);
    llvm::Value* true_val = CreateLLVMValue(op->args[1]);
    llvm::Value* false_val = CreateLLVMValue(op->args[2]);
    SetVar(op, builder_->CreateSelect(cond, true_val, false_val));
    return;
  }

  // 数学函数调用
  if (op->op.same_as(tir::builtin::call_llvm_pure_intrin())) {
    std::string intrin_name = op->args[0].as<StringImmNode>()->value;
    llvm::Function* intrin = GetIntrinsic(intrin_name);
    std::vector<llvm::Value*> args;
    for (size_t i = 1; i < op->args.size(); i++) {
      args.push_back(CreateLLVMValue(op->args[i]));
    }
    SetVar(op, builder_->CreateCall(intrin, args));
    return;
  }

  // TVM 运行时函数调用
  if (op->op.same_as(tir::builtin::tvm_call_packed())) {
    GeneratePackedCall(op);
    return;
  }

  LOG(FATAL) << "Unsupported call: " << op->op;
}
```

---

## 17.8 性能对比与最佳实践

### 17.8.1 不同 CPU 架构的优化策略

| 架构 | 关键优化 | TVM Target 配置 |
|------|---------|-----------------|
| x86-64 (Skylake) | AVX2, FMA, BMI2 | `llvm -mcpu=skylake` |
| x86-64 (Zen3) | AVX2, FMA | `llvm -mcpu=znver3` |
| AArch64 (Cortex-A78) | NEON, DotProd | `llvm -mcpu=cortex-a78` |
| AArch64 (Apple M1) | NEON, Apple-specific | `llvm -mcpu=apple-m1` |
| RISC-V (SiFive) | RV64GCV (V 扩展) | `llvm -mcpu=sifive-p670` |

### 17.8.2 向量化宽度选择

选择合适的向量化宽度需要考虑目标硬件和数据类型：

```python
# 根据目标自动选择向量化宽度
def get_vector_width(target, dtype):
    """根据目标硬件和数据类型选择向量化宽度"""
    if "avx512" in target.attrs.get("mattr", ""):
        if dtype == "float32": return 16  # 512 / 32
        if dtype == "float64": return 8   # 512 / 64
        if dtype == "int8": return 64     # 512 / 8
    elif "avx2" in target.attrs.get("mattr", ""):
        if dtype == "float32": return 8   # 256 / 32
        if dtype == "float64": return 4   # 256 / 64
    elif "neon" in target.attrs.get("mattr", ""):
        if dtype == "float32": return 4   # 128 / 32
        if dtype == "float64": return 2   # 128 / 64
    return 1  # 无向量化
```

### 17.8.3 缓存友好的数据布局

TVM 支持通过布局变换优化 CPU 缓存利用率：

```python
# NHWC 布局通常比 NCHW 更适合 CPU 向量化
# 因为相邻的内存地址可以被向量加载指令一起加载
layout_transform = relay.layout_transform(data, "NCHW", "NHWC")
```

---

## 17.9 TIR 特殊节点的 LLVM IR 生成

### 17.9.1 Allocate 节点

TIR 的 `Allocate` 节点对应局部内存分配，在 LLVM 中映射为 `alloca` 指令：

```cpp
void CodeGenLLVM::VisitStmt(const tir::AllocateNode* op) {
  // 计算分配大小
  int64_t constant_size = op->constant_allocation_size();
  llvm::Type* elem_type = LLVMType(op->dtype);

  llvm::Value* alloca_result;
  if (constant_size > 0) {
    // 常量大小：直接使用 alloca
    alloca_result = builder_->CreateAlloca(
        elem_type,
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx_),
                               constant_size));
  } else {
    // 动态大小：使用 alloca + 动态大小
    llvm::Value* size = CreateLLVMValue(op->extents[0]);
    for (size_t i = 1; i < op->extents.size(); i++) {
      size = builder_->CreateMul(size, CreateLLVMValue(op->extents[i]));
    }
    alloca_result = builder_->CreateAlloca(elem_type, size);
  }

  // 设置对齐
  alloca_result->setAlignment(llvm::Align(op->dtype.bytes()));

  // 注册到符号表
  var_map_[op->buffer_var.get()] = alloca_result;

  // 生成初始化代码（如果有）
  if (op->init_value.defined()) {
    GenerateBufferInit(alloca_result, op->extents, op->init_value);
  }

  // 生成使用代码
  VisitStmt(op->body);
}
```

### 17.9.2 IfThenElse 节点

条件分支在 LLVM 中映射为条件跳转指令：

```cpp
void CodeGenLLVM::VisitStmt(const tir::IfThenElseNode* op) {
  llvm::Value* cond = CreateLLVMValue(op->condition);

  // 创建基本块
  llvm::Function* function = builder_->GetInsertBlock()->getParent();
  llvm::BasicBlock* then_bb = llvm::BasicBlock::Create(
      *ctx_, "then", function);
  llvm::BasicBlock* else_bb = llvm::BasicBlock::Create(
      *ctx_, "else", function);
  llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(
      *ctx_, "if_merge", function);

  // 条件跳转
  builder_->CreateCondBr(cond, then_bb, else_bb);

  // Then 分支
  builder_->SetInsertPoint(then_bb);
  VisitStmt(op->then_case);
  builder_->CreateBr(merge_bb);

  // Else 分支
  builder_->SetInsertPoint(else_bb);
  if (op->else_case.defined()) {
    VisitStmt(op->else_case);
  }
  builder_->CreateBr(merge_bb);

  // 合并点
  builder_->SetInsertPoint(merge_bb);
}
```

### 17.9.3 Let 节点

`Let` 绑定在 LLVM 中通过 `alloca` + `store` + `load` 实现：

```cpp
void CodeGenLLVM::VisitStmt(const tir::LetNode* op) {
  // 计算值
  llvm::Value* value = CreateLLVMValue(op->value);

  // 分配局部变量
  llvm::AllocaInst* alloca = builder_->CreateAlloca(
      LLVMType(op->var->dtype), nullptr, op->var->name_hint);
  alloca->setAlignment(llvm::Align(op->var->dtype.bytes()));

  // 存储值
  builder_->CreateStore(value, alloca);

  // 注册到符号表
  var_map_[op->var.get()] = alloca;

  // 生成后续代码
  VisitStmt(op->body);
}
```

### 17.9.4 Assert 节点

`Assert` 在调试模式下生成断言检查，在发布模式下被忽略：

```cpp
void CodeGenLLVM::VisitStmt(const tir::AssertStmtNode* op) {
  if (enable_assert_) {
    // 生成条件检查
    llvm::Value* cond = CreateLLVMValue(op->condition);

    // 创建失败分支
    llvm::Function* function = builder_->GetInsertBlock()->getParent();
    llvm::BasicBlock* assert_fail = llvm::BasicBlock::Create(
        *ctx_, "assert_fail", function);
    llvm::BasicBlock* assert_pass = llvm::BasicBlock::Create(
        *ctx_, "assert_pass", function);

    builder_->CreateCondBr(cond, assert_pass, assert_fail);

    // 失败分支：调用错误处理
    builder_->SetInsertPoint(assert_fail);
    llvm::Function* abort_func = GetRuntimeFunction("TVMAPISetLastError");
    std::string error_msg = op->message.as<StringImmNode>()->value;
    builder_->CreateCall(abort_func, {
        builder_->CreateGlobalStringPtr(error_msg)});
    builder_->CreateUnreachable();

    // 通过分支
    builder_->SetInsertPoint(assert_pass);
  }

  VisitStmt(op->body);
}
```

### 17.9.5 AttrStmt 节点

`AttrStmt` 用于传递元数据信息，不生成实际代码：

```cpp
void CodeGenLLVM::VisitStmt(const tir::AttrStmtNode* op) {
  if (op->attr_key == tir::attr::thread_extent) {
    // 并行循环的线程范围注释
    // 用于后续的并行化优化
    thread_extents_[op->node.as<tir::IterVarNode>()->var->name_hint] =
        op->value.as<IntImmNode>()->value;
  } else if (op->attr_key == tir::attr::storage_alignment) {
    // 内存对齐注释
    storage_alignments_[op->node.as<tir::BufferNode>()->data->name_hint] =
        op->value.as<IntImmNode>()->value;
  }
  // 其他属性不影响代码生成
  VisitStmt(op->body);
}
```

---

## 17.10 LLVM 内置函数（Intrinsics）映射

### 17.10.1 数学函数映射

TVM 的数学函数调用映射到 LLVM 内置函数或 libc 函数：

| TVM 函数 | LLVM Intrinsic / libc 函数 | 说明 |
|---------|---------------------------|------|
| `T.sqrt(x)` | `@llvm.sqrt.f32` | 平方根 |
| `T.exp(x)` | `@llvm.exp.f32` | 指数函数 |
| `T.log(x)` | `@llvm.log.f32` | 对数函数 |
| `T.sin(x)` | `@llvm.sin.f32` | 正弦函数 |
| `T.cos(x)` | `@llvm.cos.f32` | 余弦函数 |
| `T.fma(a,b,c)` | `@llvm.fma.f32` | 融合乘加 |
| `T.abs(x)` | `@llvm.fabs.f32` | 绝对值 |
| `T.floor(x)` | `@llvm.floor.f32` | 向下取整 |
| `T.ceil(x)` | `@llvm.ceil.f32` | 向上取整 |
| `T.round(x)` | `@llvm.round.f32` | 四舍五入 |
| `T.pow(x,y)` | `@llvm.pow.f32` | 幂函数 |
| `T.clamp(x,a,b)` | `@llvm.minnum/@llvm.maxnum` | 裁剪 |

### 17.10.2 位操作函数

| TVM 函数 | LLVM IR | 说明 |
|---------|---------|------|
| `T.shift_left(x,n)` | `shl` | 左移 |
| `T.shift_right(x,n)` | `lshr` / `ashr` | 右移（逻辑/算术） |
| `T.bitwise_and(a,b)` | `and` | 按位与 |
| `T.bitwise_or(a,b)` | `or` | 按位或 |
| `T.bitwise_xor(a,b)` | `xor` | 按位异或 |
| `T.popcount(x)` | `@llvm.ctpop.i32` | 置位计数 |

### 17.10.3 SIMD Intrinsics

```cpp
// 向量化的 LLVM 内置函数
llvm::Function* CodeGenLLVM::GetSIMDIntrinsic(const std::string& name,
                                                llvm::Type* vec_type) {
  int lanes = vec_type->getVectorNumElements();
  llvm::Type* elem_type = vec_type->getScalarType();

  std::string intrinsic_name;
  if (elem_type->isFloatTy()) {
    intrinsic_name = "llvm.f" + name + ".f32" + "v" + std::to_string(lanes);
  } else if (elem_type->isIntegerTy()) {
    intrinsic_name = "llvm." + name + ".i" +
                     std::to_string(elem_type->getIntegerBitWidth()) +
                     "v" + std::to_string(lanes);
  }

  return llvm::Intrinsic::getDeclaration(
      module_, llvm::Intrinsic::getIntrinsicID(intrinsic_name));
}
```

---

## 17.11 LLVM Module 链接与优化

### 17.11.1 Module 链接

TVM 支持将多个 LLVM Module 链接在一起：

```cpp
// src/target/llvm/llvm_module.cc
class LinkedLLVMModule {
 public:
  void AddModule(std::unique_ptr<llvm::Module> module) {
    modules_.push_back(std::move(module));
  }

  void Link() {
    // 使用 LLVM 的 Linker 链接模块
    for (size_t i = 1; i < modules_.size(); i++) {
      llvm::Linker::linkModules(
          *modules_[0], std::move(modules_[i]));
    }
  }

  void Optimize() {
    // 对链接后的模块进行全局优化
    llvm::legacy::PassManager pm;
    pm.add(llvm::createGlobalDCEPass());       // 全局死代码消除
    pm.add(llvm::createConstantMergePass());   // 常量合并
    pm.add(llvm::createFunctionInliningPass()); // 函数内联
    pm.run(*modules_[0]);
  }

 private:
  std::vector<std::unique_ptr<llvm::Module>> modules_;
};
```

### 17.11.2 LTO（链接时优化）

```cpp
// 启用 LTO 优化
void CodeGenLLVM::EnableLTO() {
  // 设置模块标志
  module_->addModuleFlag(llvm::Module::Error, "LTOPostLink", 1);

  // 使用 ThinLTO
  module_->addModuleFlag(llvm::Module::Error, "ThinLTO", 0);
  module_->addModuleFlag(llvm::Module::Error, "EnableSplitLTOUnit", 1);
}
```

### 17.11.3 目标特化的优化 Pass

```cpp
// x86-64 特有的优化 Pass
void CodeGenX86_64::AddTargetPasses(llvm::PassManagerBase& pm) {
  // 使用 x86 特有的优化
  pm.add(llvm::createX86InstrInfoPass());
  pm.add(llvm::createX86OptimizeLEAPass());

  // 启用 BMI/BMI2 指令
  if (target_->GetFeature("bmi")) {
    pm.add(llvm::createX86FixupBWInstsPass());
  }
}

// AArch64 特有的优化 Pass
void CodeGenAArch64::AddTargetPasses(llvm::PassManagerBase& pm) {
  // 使用 ARM 特有的优化
  pm.add(llvm::createAArch64ConditionOptimizerPass());
  pm.add(llvm::createAArch64DeadRegisterDefinitionsPass());
}
```

---

## 17.12 调试与性能分析

### 17.12.1 LLVM IR Dump

TVM 支持在编译过程中 dump LLVM IR 用于调试：

```python
# 启用 LLVM IR dump
with tvm.transform.PassContext(
    config={"tir.add_lower_pass": [
        (2, tvm.tir.transform.PrintIR("llvm_ir_dump"))
    ]}
):
    lib = tvm.build(sch.mod, target="llvm")
```

```cpp
// 手动 dump LLVM IR
void CodeGenLLVM::DumpIR(const std::string& filename) {
  std::error_code ec;
  llvm::raw_fd_ostream os(filename, ec);
  module_->print(os, nullptr);
}
```

### 17.12.2 性能计数器

```cpp
// LLVM 支持的性能计数器
enum class PerfCounter {
  kInstructionCount,     // 指令数
  kCycleCount,           // 周期数
  kCacheMiss,            // 缓存未命中
  kBranchMisprediction,  // 分支预测失败
};

void CodeGenLLVM::InsertPerfCounter(PerfCounter counter) {
  // 使用 LLVM 的性能计数器 intrinsic
  switch (counter) {
    case kInstructionCount:
      builder_->CreateCall(
          GetIntrinsic("llvm.readcyclecounter"), {});
      break;
    // ...
  }
}
```

### 17.12.3 常见编译问题排查

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| 段错误 | Buffer 越界 | 检查 Allocate 大小 |
| 数值错误 | 浮点精度 | 使用 `-ffast-math` 的替代方案 |
| 性能不佳 | 未向量化 | 检查对齐和循环结构 |
| 编译时间长 | 函数过大 | 拆分为多个小函数 |
| 链接错误 | 符号未定义 | 检查外部函数声明 |

---

## 17.13 LLVM 代码生成的高级话题

### 17.13.1 函数内联策略

TVM 使用 LLVM 的函数内联来减少函数调用开销：

```cpp
// 函数内联策略配置
void CodeGenLLVM::SetInliningStrategy(const Target& target) {
  if (target->GetAttr<String>("mcpu") == "skylake") {
    // Skylake: 激进内联
    module_->addModuleFlag(llvm::Module::Override,
                           "InlineThreshold", 500);
  } else if (target->GetAttr<String>("mcpu") == "cortex-a55") {
    // Cortex-A55: 保守内联（小缓存）
    module_->addModuleFlag(llvm::Module::Override,
                           "InlineThreshold", 100);
  }
}
```

### 17.13.2 循环展开策略

```cpp
// 自动循环展开
void CodeGenLLVM::ApplyLoopUnrolling(const tir::ForNode* op) {
  int64_t extent = op->extent.as<IntImmNode>()->value;

  // 展开策略：
  // 1. 小循环（< 8 次）：完全展开
  // 2. 中等循环（8-64 次）：部分展开（4 或 8 次）
  // 3. 大循环（> 64 次）：不展开

  if (extent <= 8) {
    // 完全展开
    for (int64_t i = 0; i < extent; i++) {
      VisitStmt(Substitute(op->body, {{op->loop_var, i}}));
    }
  } else if (extent <= 64) {
    // 部分展开
    int unroll_factor = 4;
    PrintIndent();
    stream << "#pragma unroll " << unroll_factor << "\n";
    VisitStmt_(op);
  } else {
    // 不展开
    VisitStmt_(op);
  }
}
```

### 17.13.3 内存预取

```cpp
// 插入内存预取指令
void CodeGenLLVM::InsertPrefetch(const tir::PrefetchNode* op) {
  llvm::Value* addr = CreateLLVMValue(op->addr);
  llvm::Type* i8_ptr = llvm::Type::getInt8PtrTy(*ctx_);

  // 使用 LLVM 预取 intrinsic
  // @llvm.prefetch(i8* <addr>, i32 <rw>, i32 <locality>, i32 <cache_type>)
  llvm::Function* prefetch = llvm::Intrinsic::getDeclaration(
      module_, llvm::Intrinsic::prefetch);

  builder_->CreateCall(prefetch, {
      builder_->CreateBitCast(addr, i8_ptr),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx_), 0),  // read
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx_), 3),  // high locality
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx_), 1),  // data cache
  });
}
```

### 17.13.4 SIMD Intrinsics 详细映射

| TVM 操作 | x86 SSE | x86 AVX2 | ARM NEON |
|---------|---------|----------|----------|
| 向量加法 (f32x4) | `_mm_add_ps` | `_mm256_add_ps` | `vaddq_f32` |
| 向量乘法 (f32x4) | `_mm_mul_ps` | `_mm256_mul_ps` | `vmulq_f32` |
| 融合乘加 (f32x4) | `_mm_fmadd_ps` | `_mm256_fmadd_ps` | `vfmaq_f32` |
| 向量加载 (f32x4) | `_mm_load_ps` | `_mm256_load_ps` | `vld1q_f32` |
| 向量存储 (f32x4) | `_mm_store_ps` | `_mm256_store_ps` | `vst1q_f32` |
| 水平求和 (f32x4) | `_mm_hadd_ps` | `_mm256_hadd_ps` | `vaddvq_f32` |
| 打包 (f32x4→f32x8) | - | `_mm256_insertf128_ps` | - |
| 解包 (f32x8→f32x4) | - | `_mm256_extractf128_ps` | - |

### 17.13.5 TVM 向量操作的 LLVM IR 生成

```cpp
// 向量广播：将标量复制到向量的每个元素
llvm::Value* CodeGenLLVM::CreateVecBroadcast(llvm::Value* value,
                                               int lanes) {
  llvm::Type* vec_type = llvm::VectorType::get(value->getType(), lanes);
  llvm::Value* undef = llvm::UndefValue::get(vec_type);

  // 使用 insertelement 构建向量
  llvm::Value* result = undef;
  for (int i = 0; i < lanes; i++) {
    result = builder_->CreateInsertElement(
        result, value,
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx_), i));
  }
  return result;
}

// 向量归约：将向量的所有元素归约为标量
llvm::Value* CodeGenLLVM::CreateVecReduce(llvm::Value* vec,
                                            const std::string& op) {
  int lanes = vec->getType()->getVectorNumElements();

  if (op == "add") {
    // 树形归约
    for (int stride = lanes / 2; stride > 0; stride /= 2) {
      llvm::Value* shuffled = builder_->CreateShuffleVector(
          vec, llvm::UndefValue::get(vec->getType()),
          CreateShuffleMask(stride, lanes));
      vec = builder_->CreateFAdd(vec, shuffled);
    }
    return builder_->CreateExtractElement(
        vec, llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx_), 0));
  }

  if (op == "max") {
    // 树形最大值归约
    for (int stride = lanes / 2; stride > 0; stride /= 2) {
      llvm::Value* shuffled = builder_->CreateShuffleVector(
          vec, llvm::UndefValue::get(vec->getType()),
          CreateShuffleMask(stride, lanes));
      vec = builder_->CreateMaxNum(vec, shuffled);
    }
    return builder_->CreateExtractElement(
        vec, llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx_), 0));
  }

  LOG(FATAL) << "Unsupported reduce op: " << op;
  return nullptr;
}
```

### 17.13.6 条件向量化

当循环体中包含条件语句时，LLVM 使用掩码向量化：

```llvm
; 条件向量化的 LLVM IR 示例
; if (A[i] > 0) B[i] = A[i]; else B[i] = 0;

loop_body:
  %a_vec = load <4 x float>, <4 x float>* %ptr_a
  %mask = fcmp ogt <4 x float> %a_vec, zeroinitializer
  %b_vec = select <4 x i1> %mask, <4 x float> %a_vec, <4 x float> zeroinitializer
  store <4 x float> %b_vec, <4 x float>* %ptr_b
```

### 17.13.7 尾部向量化处理

当循环长度不能被向量宽度整除时：

```cpp
// 生成尾部处理代码
void CodeGenLLVM::GenerateVectorizedWithTail(
    const tir::ForNode* op, int vector_width) {
  int64_t total = op->extent.as<IntImmNode>()->value;
  int64_t main_iters = (total / vector_width) * vector_width;
  int64_t tail_iters = total - main_iters;

  // 主循环：向量化处理
  tir::For main_loop = tir::For(
      op->loop_var, 0, main_iters, vector_width,
      tir::ForKind::kVectorized, op->body);
  VisitStmt(main_loop);

  // 尾部循环：标量处理
  if (tail_iters > 0) {
    tir::For tail_loop = tir::For(
        tir::Var(op->loop_var->name_hint + "_tail"),
        main_iters, tail_iters, 1,
        tir::ForKind::kSerial, op->body);
    VisitStmt(tail_loop);
  }
}
```

### 17.13.8 LLVM Pass 管线的自定义配置

```python
# 自定义 LLVM Pass 管线
target = tvm.target.Target({
    "kind": "llvm",
    "mcpu": "skylake",
    # 自定义 LLVM Pass 选项
    "opt-level": 3,
    "disable-vector-combine": False,
    "enable-loop-interchange": True,
    "enable-loop-distribute": True,
})
```

```cpp
// TVM 中的 LLVM Pass 配置
void CodeGenLLVM::ConfigurePasses(const Target& target) {
  // 设置优化级别
  opt_level_ = target->GetAttr<Integer>("opt-level").value_or(3);

  // 禁用特定 Pass
  if (target->GetAttr<Bool>("disable-vector-combine").value_or(false)) {
    disabled_passes_.insert("vector-combine");
  }

  // 启用实验性 Pass
  if (target->GetAttr<Bool>("enable-loop-interchange").value_or(false)) {
    enabled_passes_.insert("loop-interchange");
  }
}
```

### 17.13.9 调试信息生成

```cpp
// 生成 DWARF 调试信息
void CodeGenLLVM::GenerateDebugInfo(const tir::PrimFunc& func) {
  // 创建编译单元
  llvm::DIFile* file = dibuilder_->createFile(
      func->source_file, "/");

  llvm::DISubprogram* sp = dibuilder_->createFunction(
      file, func->name, func->name, file,
      func->source_line, nullptr, false, true);

  // 为每条指令添加调试位置
  for (auto& bb : function_->getBasicBlockList()) {
    for (auto& inst : bb.getInstList()) {
      inst.setDebugLoc(llvm::DILocation::get(
          *ctx_, inst.getSourceLocation(), 0, sp));
    }
  }
}
```

---

## 17.14 LLVM 代码生成的高级话题

### 17.14.1 函数内联策略

TVM 使用 LLVM 的函数内联来减少函数调用开销：

```cpp
// 函数内联策略配置
void CodeGenLLVM::SetInliningStrategy(const Target& target) {
  if (target->GetAttr<String>("mcpu") == "skylake") {
    // Skylake: 激进内联
    module_->addModuleFlag(llvm::Module::Override,
                           "InlineThreshold", 500);
  } else if (target->GetAttr<String>("mcpu") == "cortex-a55") {
    // Cortex-A55: 保守内联（小缓存）
    module_->addModuleFlag(llvm::Module::Override,
                           "InlineThreshold", 100);
  }
}
```

### 17.14.2 循环展开策略

```cpp
// 自动循环展开
void CodeGenLLVM::ApplyLoopUnrolling(const tir::ForNode* op) {
  int64_t extent = op->extent.as<IntImmNode>()->value;

  // 展开策略：
  // 1. 小循环（< 8 次）：完全展开
  // 2. 中等循环（8-64 次）：部分展开（4 或 8 次）
  // 3. 大循环（> 64 次）：不展开

  if (extent <= 8) {
    // 完全展开
    for (int64_t i = 0; i < extent; i++) {
      VisitStmt(Substitute(op->body, {{op->loop_var, i}}));
    }
  } else if (extent <= 64) {
    // 部分展开
    int unroll_factor = 4;
    PrintIndent();
    stream << "#pragma unroll " << unroll_factor << "\n";
    VisitStmt_(op);
  } else {
    // 不展开
    VisitStmt_(op);
  }
}
```

### 17.14.3 内存预取

```cpp
// 插入内存预取指令
void CodeGenLLVM::InsertPrefetch(const tir::PrefetchNode* op) {
  llvm::Value* addr = CreateLLVMValue(op->addr);
  llvm::Type* i8_ptr = llvm::Type::getInt8PtrTy(*ctx_);

  // 使用 LLVM 预取 intrinsic
  // @llvm.prefetch(i8* <addr>, i32 <rw>, i32 <locality>, i32 <cache_type>)
  llvm::Function* prefetch = llvm::Intrinsic::getDeclaration(
      module_, llvm::Intrinsic::prefetch);

  builder_->CreateCall(prefetch, {
      builder_->CreateBitCast(addr, i8_ptr),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx_), 0),  // read
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx_), 3),  // high locality
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx_), 1),  // data cache
  });
}
```

### 17.14.4 SIMD Intrinsics 详细映射

| TVM 操作 | x86 SSE | x86 AVX2 | ARM NEON |
|---------|---------|----------|----------|
| 向量加法 (f32x4) | `_mm_add_ps` | `_mm256_add_ps` | `vaddq_f32` |
| 向量乘法 (f32x4) | `_mm_mul_ps` | `_mm256_mul_ps` | `vmulq_f32` |
| 融合乘加 (f32x4) | `_mm_fmadd_ps` | `_mm256_fmadd_ps` | `vfmaq_f32` |
| 向量加载 (f32x4) | `_mm_load_ps` | `_mm256_load_ps` | `vld1q_f32` |
| 向量存储 (f32x4) | `_mm_store_ps` | `_mm256_store_ps` | `vst1q_f32` |
| 水平求和 (f32x4) | `_mm_hadd_ps` | `_mm256_hadd_ps` | `vaddvq_f32` |
| 打包 (f32x4→f32x8) | - | `_mm256_insertf128_ps` | - |
| 解包 (f32x8→f32x4) | - | `_mm256_extractf128_ps` | - |

### 17.14.5 TVM 向量操作的 LLVM IR 生成

```cpp
// 向量广播：将标量复制到向量的每个元素
llvm::Value* CodeGenLLVM::CreateVecBroadcast(llvm::Value* value,
                                               int lanes) {
  llvm::Type* vec_type = llvm::VectorType::get(value->getType(), lanes);
  llvm::Value* undef = llvm::UndefValue::get(vec_type);

  // 使用 insertelement 构建向量
  llvm::Value* result = undef;
  for (int i = 0; i < lanes; i++) {
    result = builder_->CreateInsertElement(
        result, value,
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx_), i));
  }
  return result;
}

// 向量归约：将向量的所有元素归约为标量
llvm::Value* CodeGenLLVM::CreateVecReduce(llvm::Value* vec,
                                            const std::string& op) {
  int lanes = vec->getType()->getVectorNumElements();

  if (op == "add") {
    // 树形归约
    for (int stride = lanes / 2; stride > 0; stride /= 2) {
      llvm::Value* shuffled = builder_->CreateShuffleVector(
          vec, llvm::UndefValue::get(vec->getType()),
          CreateShuffleMask(stride, lanes));
      vec = builder_->CreateFAdd(vec, shuffled);
    }
    return builder_->CreateExtractElement(
        vec, llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx_), 0));
  }

  if (op == "max") {
    // 树形最大值归约
    for (int stride = lanes / 2; stride > 0; stride /= 2) {
      llvm::Value* shuffled = builder_->CreateShuffleVector(
          vec, llvm::UndefValue::get(vec->getType()),
          CreateShuffleMask(stride, lanes));
      vec = builder_->CreateMaxNum(vec, shuffled);
    }
    return builder_->CreateExtractElement(
        vec, llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx_), 0));
  }

  LOG(FATAL) << "Unsupported reduce op: " << op;
  return nullptr;
}
```

### 17.14.6 条件向量化

当循环体中包含条件语句时，LLVM 使用掩码向量化：

```llvm
; 条件向量化的 LLVM IR 示例
; if (A[i] > 0) B[i] = A[i]; else B[i] = 0;

loop_body:
  %a_vec = load <4 x float>, <4 x float>* %ptr_a
  %mask = fcmp ogt <4 x float> %a_vec, zeroinitializer
  %b_vec = select <4 x i1> %mask, <4 x float> %a_vec, <4 x float> zeroinitializer
  store <4 x float> %b_vec, <4 x float>* %ptr_b
```

### 17.14.7 尾部向量化处理

当循环长度不能被向量宽度整除时：

```cpp
// 生成尾部处理代码
void CodeGenLLVM::GenerateVectorizedWithTail(
    const tir::ForNode* op, int vector_width) {
  int64_t total = op->extent.as<IntImmNode>()->value;
  int64_t main_iters = (total / vector_width) * vector_width;
  int64_t tail_iters = total - main_iters;

  // 主循环：向量化处理
  tir::For main_loop = tir::For(
      op->loop_var, 0, main_iters, vector_width,
      tir::ForKind::kVectorized, op->body);
  VisitStmt(main_loop);

  // 尾部循环：标量处理
  if (tail_iters > 0) {
    tir::For tail_loop = tir::For(
        tir::Var(op->loop_var->name_hint + "_tail"),
        main_iters, tail_iters, 1,
        tir::ForKind::kSerial, op->body);
    VisitStmt(tail_loop);
  }
}
```

### 17.14.8 LLVM Pass 管线的自定义配置

```python
# 自定义 LLVM Pass 管线
target = tvm.target.Target({
    "kind": "llvm",
    "mcpu": "skylake",
    # 自定义 LLVM Pass 选项
    "opt-level": 3,
    "disable-vector-combine": False,
    "enable-loop-interchange": True,
    "enable-loop-distribute": True,
})
```

```cpp
// TVM 中的 LLVM Pass 配置
void CodeGenLLVM::ConfigurePasses(const Target& target) {
  // 设置优化级别
  opt_level_ = target->GetAttr<Integer>("opt-level").value_or(3);

  // 禁用特定 Pass
  if (target->GetAttr<Bool>("disable-vector-combine").value_or(false)) {
    disabled_passes_.insert("vector-combine");
  }

  // 启用实验性 Pass
  if (target->GetAttr<Bool>("enable-loop-interchange").value_or(false)) {
    enabled_passes_.insert("loop-interchange");
  }
}
```

### 17.14.9 调试信息生成

```cpp
// 生成 DWARF 调试信息
void CodeGenLLVM::GenerateDebugInfo(const tir::PrimFunc& func) {
  // 创建编译单元
  llvm::DIFile* file = dibuilder_->createFile(
      func->source_file, "/");

  llvm::DISubprogram* sp = dibuilder_->createFunction(
      file, func->name, func->name, file,
      func->source_line, nullptr, false, true);

  // 为每条指令添加调试位置
  for (auto& bb : function_->getBasicBlockList()) {
    for (auto& inst : bb.getInstList()) {
      inst.setDebugLoc(llvm::DILocation::get(
          *ctx_, inst.getSourceLocation(), 0, sp));
    }
  }
}
```

---

## 17.15 LLVM 后端的常见陷阱与最佳实践

### 17.15.1 常见陷阱

**陷阱一：类型不匹配**

```cpp
// 错误：int32 和 float32 混用
llvm::Value* a = CreateLLVMValue(int_expr);
llvm::Value* b = CreateLLVMValue(float_expr);
builder_->CreateAdd(a, b);  // 类型不匹配！

// 正确：先进行类型转换
llvm::Value* a_float = builder_->CreateSIToFP(a, float_type);
builder_->CreateFAdd(a_float, b);
```

**陷阱二：对齐假设错误**

```cpp
// 错误：假设所有访问都是对齐的
builder_->CreateAlignedLoad(float_type, ptr, llvm::Align(16));

// 正确：根据实际情况设置对齐
builder_->CreateAlignedLoad(float_type, ptr, llvm::Align(4));  // float32
```

**陷阱三：未处理的循环边界**

```cpp
// 错误：假设循环长度是向量宽度的倍数
// 如果 N=1023, vector_width=8, 则会越界！

// 正确：添加边界检查
if (N % vector_width != 0) {
  // 生成尾部处理代码
  GenerateVectorizedWithTail(op, vector_width);
}
```

### 17.15.2 最佳实践

**实践一：缓存 LLVM 值**

```cpp
// 避免重复计算相同的表达式
llvm::Value* CodeGenLLVM::CreateLLVMValue(const tir::PrimExpr& expr) {
  // 检查缓存
  auto it = value_cache_.find(expr.get());
  if (it != value_cache_.end()) {
    return it->second;
  }

  // 计算并缓存
  llvm::Value* value = ComputeLLVMValue(expr);
  value_cache_[expr.get()] = value;
  return value;
}
```

**实践二：合理使用 LLVM Pass**

```cpp
// 不要禁用所有优化
// 错误：opt_level = 0

// 根据场景选择优化级别
// 调试：opt_level = 0
// 开发：opt_level = 2
// 生产：opt_level = 3
```

**实践三：使用正确的内存模型**

```cpp
// 正确设置内存操作的属性
llvm::LoadInst* load = builder_->CreateLoad(type, ptr);
load->setAlignment(llvm::Align(alignment));
load->setVolatile(false);  // 非易失性
load->setOrdering(llvm::AtomicOrdering::NotAtomic);  // 非原子

// 对于并行访问
load->setOrdering(llvm::AtomicOrdering::Monotonic);
```

### 17.15.3 LLVM Target Machine 配置

```cpp
// LLVM Target Machine 的详细配置
llvm::TargetMachine* CodeGenLLVM::CreateTargetMachine(
    const Target& target) {
  // 获取目标三元组
  std::string triple = target->GetAttr<String>("mtriple").value();

  // 获取 CPU 型号
  std::string cpu = target->GetAttr<String>("mcpu").value_or("generic");

  // 获取特性字符串
  std::string features = target->GetAttr<String>("mattr").value_or("");

  // 解析目标
  const llvm::Target* llvm_target =
      llvm::TargetRegistry::lookupTarget(triple);

  // 配置选项
  llvm::TargetOptions options;
  options.FloatABIType = llvm::FloatABI::Hard;
  options.NoInfsFPMath = false;
  options.NoNaNsFPMath = false;
  options.NoSignedZerosFPMath = false;
  options.UnsafeFPMath = false;

  // 代码模型
  llvm::CodeModel::Model code_model = llvm::CodeModel::Small;

  // 重定位模型
  llvm::Reloc::Model reloc_model = llvm::Reloc::PIC_;

  // 创建 Target Machine
  return llvm_target->createTargetMachine(
      triple, cpu, features, options, reloc_model, code_model);
}
```

### 17.15.4 LLVM 数据布局

```cpp
// 数据布局配置
void CodeGenLLVM::ConfigureDataLayout(const Target& target) {
  std::string triple = target->GetAttr<String>("mtriple").value();

  // x86-64 数据布局
  if (triple.find("x86_64") != std::string::npos) {
    // e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128
    module_->setDataLayout("e-m:e-p270:32:32-p271:32:32-p272:64:64"
                           "-i64:64-f80:128-n8:16:32:64-S128");
  }
  // AArch64 数据布局
  else if (triple.find("aarch64") != std::string::npos) {
    // e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128
    module_->setDataLayout("e-m:e-i8:8:32-i16:16:32-i64:64"
                           "-i128:128-n32:64-S128");
  }
  // RISC-V 数据布局
  else if (triple.find("riscv") != std::string::npos) {
    // e-m:e-p:64:64-i64:64-i128:128-n64-S128
    module_->setDataLayout("e-m:e-p:64:64-i64:64-i128:128-n64-S128");
  }
}
```

### 17.15.5 CPU 特性检测与使用

```cpp
// 运行时 CPU 特性检测
class CPUFeatureDetector {
 public:
  struct Features {
    bool sse;
    bool sse2;
    bool sse3;
    bool ssse3;
    bool sse4_1;
    bool sse4_2;
    bool avx;
    bool avx2;
    bool avx512f;
    bool fma;
    bool neon;
    bool sve;
  };

  static Features Detect() {
    Features feat = {false};

    #if defined(__x86_64__) || defined(_M_X64)
    // 使用 CPUID 指令检测
    int info[4];
    __cpuid(info, 1);
    feat.sse   = (info[3] & (1 << 25)) != 0;
    feat.sse2  = (info[3] & (1 << 26)) != 0;
    feat.sse3  = (info[2] & (1 << 0)) != 0;
    feat.ssse3 = (info[2] & (1 << 9)) != 0;
    feat.sse4_1 = (info[2] & (1 << 19)) != 0;
    feat.sse4_2 = (info[2] & (1 << 20)) != 0;
    feat.avx   = (info[2] & (1 << 28)) != 0;
    feat.fma   = (info[2] & (1 << 12)) != 0;

    __cpuidex(info, 7, 0);
    feat.avx2  = (info[1] & (1 << 5)) != 0;
    feat.avx512f = (info[1] & (1 << 16)) != 0;
    #elif defined(__aarch64__)
    feat.neon = true;
    // 检测 SVE
    feat.sve = false;  // 需要运行时检测
    #endif

    return feat;
  }
};
```

---

## 17.14 本章小结

本章详细介绍了 TVM 的 LLVM 代码生成后端的架构与实现：

| 概念 | 作用 | 关键源码 |
|------|------|---------|
| CodeGenLLVM | TIR→LLVM IR 翻译 | `src/target/llvm/codegen_llvm.cc` |
| CodeGenX86_64 | x86-64 特化 | `src/target/llvm/codegen_x86_64.cc` |
| CodeGenAArch64 | AArch64 特化 | `src/target/llvm/codegen_aarch64.cc` |
| 数据类型映射 | TIR dtype→LLVM Type | `codegen_llvm.cc:LLVMType()` |
| 向量化 | SIMD 指令生成 | `codegen_llvm.cc:CreateVec*()` |
| LLVM Pass | 代码优化 | `llvm_module.cc:Optimize()` |
| JIT 编译 | 运行时编译 | `llvm_module.cc:LLVMJIT` |

**核心洞察**：

1. **TIR 到 LLVM IR 的映射是直接的**：循环→基本块+phi，Buffer→指针，算术→对应的 LLVM 指令
2. **向量化是 CPU 性能的关键**：通过显式向量化和 LLVM 自动向量化两种方式利用 SIMD
3. **目标特化很重要**：不同的 CPU 架构需要不同的向量化策略和指令选择
4. **LLVM 的优化管线是免费的**：TVM 生成的 LLVM IR 可以受益于 LLVM 数十年积累的优化技术

<div data-component="LLVMCodeGenPipeline"></div>

---

## 延伸阅读

1. **LLVM 官方文档**：https://llvm.org/docs/
2. **TVM LLVM CodeGen 源码**：`src/target/llvm/codegen_llvm.cc`
3. **LLVM Programmer's Manual**：https://llvm.org/docs/ProgrammersManual.html
4. **SIMD 指令集参考**：Intel Intrinsics Guide, ARM NEON Programmer's Guide

---

## 17.99 文字内容强化：LLVM CodeGen 的工程化阅读补充

LLVM 后端看似只是把 TIR 翻译成另一种 IR，但真正的难点在于类型、地址、调用约定和目标机器信息如何稳定贯通。

### 17.99.1 代码解读：从片段回到主流程

原有代码块里的 LLVMType、VisitExpr 和 VisitStmt 代表三层翻译：类型、表达式、语句。
控制流通常从 Build 函数进入 CodeGenLLVM，逐个 PrimFunc 生成 LLVM Function。
工程意义在于把 TVM 自己的 TIR 约束转换为 LLVM 能继续优化的 SSA 形式。
代码块中的变量名、函数名和类名不应孤立记忆，而应放回编译流水线中理解。
读者可以先判断代码块处在构建期、优化期、代码生成期还是运行期。
构建期代码通常负责收集信息，优化期代码负责改写 IR，代码生成期代码负责降低表示，运行期代码负责执行与资源管理。
一旦阶段判断正确，许多看似相似的数据结构就能区分出职责边界。

### 17.99.2 源码阅读路径

阅读 apache/tvm 源码时，建议按下面顺序推进，而不是直接在全仓库搜索 LLVM CodeGen。
第 1 步，阅读 `src/target/llvm/codegen_llvm.cc`，目标是确认这一层暴露的主要接口和被谁调用。
第 2 步，阅读 `src/target/llvm/codegen_cpu.cc`，目标是确认这一层暴露的主要接口和被谁调用。
第 3 步，阅读 `src/target/llvm/llvm_module.cc`，目标是确认这一层暴露的主要接口和被谁调用。
第 4 步，阅读 `src/tir/transforms/`，目标是确认这一层暴露的主要接口和被谁调用。
第 5 步，阅读 `include/tvm/target/codegen.h`，目标是确认这一层暴露的主要接口和被谁调用。
完成主路径后，再阅读相邻测试目录，测试通常比注释更清楚地展示了设计者希望维持的不变量。
如果遇到注册表入口，应记录注册名、C++ 实现函数、Python 包装函数和最终用户 API 四个位置。
如果遇到 Pass，应记录 Pass 的输入 IR、输出 IR、启用条件和在默认流水线中的相对顺序。
如果遇到运行时模块，应记录它的创建时机、序列化格式、加载入口和资源释放位置。

### 17.99.3 为什么这样设计

LLVM 后端复用 LLVM 生态，是因为 CPU 指令选择、寄存器分配和平台 ABI 极其复杂，TVM 只需要负责生成结构良好的 LLVM IR。
这种设计把变化频繁的硬件细节放在可替换层，把稳定的编译流程保留在公共层。
因此，当新增后端、调整调度策略或替换运行时实现时，系统不需要推翻已有抽象。
代价是调用路径会更长，读源码时会看到更多注册、转发和包装对象。
但这些额外层次换来的是跨语言、跨设备和跨部署场景的一致性。
判断一个设计是否合理，可以看它是否让常见路径足够简单，同时让少见路径仍有扩展空间。

### 17.99.4 逐行阅读提示与工程理解清单

1. TIR 到 LLVM IR 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
2. 阅读 类型映射 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
3. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
4. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
5. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
6. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
7. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
8. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
9. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
10. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
11. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
12. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
13. 工程上，稳定的边界往往比复杂的局部优化更重要。
14. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
15. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
16. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
17. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
18. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
19. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
20. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
21. ABI 边界 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
22. 阅读 JIT 模块 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
23. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
24. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
25. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
26. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
27. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
28. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
29. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
30. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
31. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
32. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
33. 工程上，稳定的边界往往比复杂的局部优化更重要。
34. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
35. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
36. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
37. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
38. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
39. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
40. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
41. TIR 到 LLVM IR 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
42. 阅读 类型映射 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
43. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
44. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
45. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
46. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
47. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
48. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
49. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
50. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
51. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
52. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
53. 工程上，稳定的边界往往比复杂的局部优化更重要。
54. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
55. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
56. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
57. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
58. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
59. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
60. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
61. ABI 边界 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
62. 阅读 JIT 模块 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
63. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
64. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
65. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
66. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
67. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
68. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
69. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
70. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
71. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
72. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
73. 工程上，稳定的边界往往比复杂的局部优化更重要。
74. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
75. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
76. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
77. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
78. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
79. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
80. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
81. TIR 到 LLVM IR 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
82. 阅读 类型映射 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
83. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
84. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
85. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
86. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
87. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
88. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
89. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
90. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
91. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
92. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
93. 工程上，稳定的边界往往比复杂的局部优化更重要。
94. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
95. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
96. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
97. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
98. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
99. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
100. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
101. ABI 边界 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
102. 阅读 JIT 模块 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
103. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
104. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
105. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
106. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
107. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
108. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
109. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
110. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
111. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
112. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
113. 工程上，稳定的边界往往比复杂的局部优化更重要。
114. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
115. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
116. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
117. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
118. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
119. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
120. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
121. TIR 到 LLVM IR 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
122. 阅读 类型映射 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
123. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
124. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
125. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
126. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
127. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
128. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
129. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
130. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
131. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
132. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
133. 工程上，稳定的边界往往比复杂的局部优化更重要。
134. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
135. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
136. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
137. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
138. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
139. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
140. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
141. ABI 边界 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
142. 阅读 JIT 模块 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
143. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
144. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
145. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
146. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
147. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
148. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
149. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
150. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
151. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
152. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
153. 工程上，稳定的边界往往比复杂的局部优化更重要。
154. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
155. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
156. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
157. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
158. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
159. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
160. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
161. TIR 到 LLVM IR 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
162. 阅读 类型映射 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
163. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
164. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
165. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
166. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
167. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
168. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
169. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
170. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
171. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
172. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
173. 工程上，稳定的边界往往比复杂的局部优化更重要。
174. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
175. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
176. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
177. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
178. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
179. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
180. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
181. ABI 边界 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
182. 阅读 JIT 模块 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
183. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
184. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
185. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
186. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
187. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
188. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
189. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
190. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
191. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
192. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
193. 工程上，稳定的边界往往比复杂的局部优化更重要。
194. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
195. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
196. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
197. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
198. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
199. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
200. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
201. TIR 到 LLVM IR 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
202. 阅读 类型映射 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
203. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
204. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
205. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
206. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
207. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
208. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
209. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
210. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
211. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
212. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
213. 工程上，稳定的边界往往比复杂的局部优化更重要。
214. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
215. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
216. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
217. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
218. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
219. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
220. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
221. ABI 边界 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
222. 阅读 JIT 模块 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
223. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
224. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
225. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
226. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
227. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
228. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
229. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
230. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
231. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
232. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
233. 工程上，稳定的边界往往比复杂的局部优化更重要。
234. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
235. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
236. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
237. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
238. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
239. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
240. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
241. TIR 到 LLVM IR 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
242. 阅读 类型映射 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
243. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
244. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
245. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
246. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
247. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
248. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
249. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
250. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
251. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
252. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
253. 工程上，稳定的边界往往比复杂的局部优化更重要。
254. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
255. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
256. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
257. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
258. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
259. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
260. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
261. ABI 边界 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
262. 阅读 JIT 模块 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
263. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
264. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
265. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
266. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
267. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
268. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
269. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
270. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
271. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
272. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
273. 工程上，稳定的边界往往比复杂的局部优化更重要。
274. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
275. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
276. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
277. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
278. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
279. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
280. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
281. TIR 到 LLVM IR 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
282. 阅读 类型映射 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
283. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
284. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
285. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
286. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
287. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
288. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
289. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
290. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
291. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
292. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
293. 工程上，稳定的边界往往比复杂的局部优化更重要。
294. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
295. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
296. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
297. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
298. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
299. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
300. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
301. ABI 边界 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
302. 阅读 JIT 模块 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
303. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
304. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
305. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
306. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
307. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
308. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
309. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
310. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
311. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
312. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
313. 工程上，稳定的边界往往比复杂的局部优化更重要。
314. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
315. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
316. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
317. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
318. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
319. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
320. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
321. TIR 到 LLVM IR 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
322. 阅读 类型映射 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
323. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
324. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
325. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
326. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
327. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
328. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
329. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
330. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
331. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
332. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
333. 工程上，稳定的边界往往比复杂的局部优化更重要。
334. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
335. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
336. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
337. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
338. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
339. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
340. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
341. ABI 边界 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
342. 阅读 JIT 模块 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
343. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
344. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
345. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
346. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
347. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
348. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
349. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
350. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
351. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
352. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
353. 工程上，稳定的边界往往比复杂的局部优化更重要。
354. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
355. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
356. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
357. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
358. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
359. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
360. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
361. TIR 到 LLVM IR 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
362. 阅读 类型映射 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
363. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
364. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
365. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
366. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
367. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
368. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
369. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
370. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
371. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
372. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
373. 工程上，稳定的边界往往比复杂的局部优化更重要。
374. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
375. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
376. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
377. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
378. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
379. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
380. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
381. ABI 边界 的第一层理解，是把它看成 CPU 代码生成后端 中连接抽象语义和工程实现的接口。
382. 阅读 JIT 模块 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
383. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
384. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
385. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
386. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
387. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
388. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
389. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。

### 17.99.5 小结：把本章放回 TVM 全链路

LLVM CodeGen 的学习重点不是记住每个函数名，而是理解它在 TVM 全链路中承担的边界职责。
当读者能够说清楚输入从哪里来、状态在哪里保存、输出被谁消费，就已经掌握了源码阅读的主线。
后续遇到性能、兼容性或部署问题时，可以沿着这条主线逐层排查，而不是在全仓库中盲目搜索。

