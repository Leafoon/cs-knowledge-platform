> **学习目标**：
> - 理解 PackedFunc 的类型擦除调用机制与设计动机
> - 掌握 TVM 的函数注册系统与 Module 体系
> - 理解 RPC Server/Client 的架构与协议
> - 掌握跨设备调试与远程执行技术
> - 理解 FFI（Foreign Function Interface）在 TVM 中的角色
> - 掌握 PackedFunc 的性能优化技巧与常见陷阱
> - 能够独立实现自定义 PackedFunc、Module 和 RPC 传输层

---

## 24.1 PackedFunc 概述

### 24.1.1 什么是 PackedFunc

PackedFunc 是 TVM 的**核心函数调用接口**，它实现了一种**类型擦除（Type-erased）**的通用函数调用机制。所有 TVM 组件——算子、Pass、优化器、运行时——都通过 PackedFunc 接口交互。

源码位置：`include/tvm/runtime/packed_func.h`

```cpp
// PackedFunc 的本质：接受任意数量、任意类型的参数，返回任意类型
using PackedFunc = TypedPackedFunc<void(TVMArgs, TVMRetValue*)>;
```

PackedFunc 的核心思想可以用一句话概括：**将函数签名统一化，通过运行时类型检查替代编译时类型检查**。这种设计使得 TVM 能够在不同语言和设备之间无缝传递函数调用。

<div data-component="PackedFuncConceptDiagram"></div>

### 24.1.2 设计动机

TVM 需要在多种语言（Python、C++、Rust、Java）和多种设备（CPU、GPU、DSP）之间传递函数调用。传统的类型安全函数签名无法满足这种需求，因此 TVM 设计了 PackedFunc：

```
传统方式（类型安全）：
  void conv2d(float* data, float* kernel, int M, int N, int K);
  → 每种函数签名需要单独的 FFI 绑定
  → 组合爆炸

PackedFunc 方式（类型擦除）：
  void func(TVMArgs args, TVMRetValue* ret);
  → 统一的调用接口
  → 运行时类型检查
```

传统 FFI 方式的组合爆炸问题：

| 参数数量 | 每种参数类型组合 | 需要的 FFI 绑定数 |
|---------|----------------|------------------|
| 2 | 5 种基本类型 | 25 |
| 3 | 5 种基本类型 | 125 |
| 4 | 5 种基本类型 | 625 |
| 5 | 5 种基本类型 | 3125 |

使用 PackedFunc 后，只需要**一个统一的调用接口**：

$$
\text{FFI 绑定数} = 1 \quad \text{（而非 } T^n \text{，其中 } T \text{ 为类型数，} n \text{ 为参数数量）}
$$

### 24.1.3 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `PackedFunc` | `include/tvm/runtime/packed_func.h` | 函数接口定义 |
| `TVMArgs` | 同上 | 参数打包容器 |
| `TVMRetValue` | 同上 | 返回值容器 |
| `TVMArgValue` | 同上 | 单个参数的类型擦除视图 |
| `Registry` | `include/tvm/runtime/registry.h` | 全局函数注册表 |
| `Module` | `include/tvm/runtime/module.h` | 模块系统 |
| `TVMValue` | `include/tvm/runtime/packed_func.h` | 类型擦除的值联合体 |
| `TVMPODValue_` | 同上 | POD 值基类 |
| `TVMRetValue` | 同上 | 返回值容器，支持移动语义 |

<div data-component="PackedFuncArchitecture"></div>

### 24.1.4 PackedFunc 的生命周期

```
PackedFunc 的生命周期：

  创建阶段
  ┌─────────────────────────────────────────────┐
  │ 1. 定义函数体（lambda / std::function）       │
  │ 2. 通过 TVM_REGISTER_GLOBAL 注册到全局注册表   │
  │ 3. 或通过 Registry::Register 动态注册          │
  └─────────────────────────────────────────────┘
                    │
                    ▼
  查找阶段
  ┌─────────────────────────────────────────────┐
  │ 1. 通过名称从 Registry 查找                   │
  │ 2. 返回 PackedFunc 指针                      │
  │ 3. 可选：通过 Module.GetFunction 查找          │
  └─────────────────────────────────────────────┘
                    │
                    ▼
  调用阶段
  ┌─────────────────────────────────────────────┐
  │ 1. 构造 TVMArgs（打包参数）                    │
  │ 2. 调用 PackedFunc::operator()               │
  │ 3. 获取 TVMRetValue（解包返回值）              │
  └─────────────────────────────────────────────┘
                    │
                    ▼
  销毁阶段
  ┌─────────────────────────────────────────────┐
  │ 1. PackedFunc 引用计数归零                    │
  │ 2. 释放函数体和捕获的资源                      │
  │ 3. 从注册表中可选移除                          │
  └─────────────────────────────────────────────┘
```

### 24.1.5 PackedFunc 与 std::function 的对比

| 特性 | `std::function` | `PackedFunc` |
|------|----------------|--------------|
| 类型安全 | 编译时检查 | 运行时检查 |
| 参数类型 | 固定签名 | 任意类型 |
| 跨语言支持 | 不支持 | 原生支持 |
| 序列化 | 不支持 | 支持（通过 RPC） |
| 性能开销 | ~5 ns | ~10 ns |
| 注册机制 | 无 | 全局注册表 |
| 适用场景 | C++ 内部调用 | 跨语言/跨设备调用 |

---

## 24.2 PackedFunc 的内部实现

### 24.2.1 TVMValue：类型擦除的值

```cpp
// include/tvm/runtime/packed_func.h
// TVMValue 是一个联合体，可以存储任意基本类型
typedef union {
  int64_t v_int64;
  double v_float64;
  void* v_handle;
  const char* v_str;
  DLTensor* v_tensor;
  TVMContext v_ctx;
  // ... 其他类型
} TVMValue;
```

TVMValue 的内存布局：

```
TVMValue 联合体内存布局（8 bytes）：

  ┌────────────────────────────────────────┐
  │          8 字节共享内存空间              │
  ├────────────────────────────────────────┤
  │ v_int64:    [========================] │  int64_t
  │ v_float64:  [========================] │  double
  │ v_handle:   [========================] │  void*
  │ v_str:      [========================] │  const char*
  │ v_tensor:   [========================] │  DLTensor*
  │ v_ctx:      [========================] │  TVMContext
  └────────────────────────────────────────┘

  注意：所有字段共享同一块内存，
  实际存储的类型由 type_code 决定
```

### 24.2.2 TVMArgs：参数容器

```cpp
class TVMArgs {
 public:
  TVMValue* values;    // 值数组
  int* type_codes;     // 类型代码数组
  int num_args;        // 参数数量

  // 按索引访问参数
  TVMArgValue operator[](int i) const {
    return TVMArgValue(values[i], type_codes[i]);
  }

  // 类型安全的访问
  int64_t IntValue(int i) const;
  double FloatValue(int i) const;
  void* HandleValue(int i) const;
  std::string StringValue(int i) const;
  NDArray NDArrayValue(int i) const;
};
```

TVMArgs 的内存布局：

```
TVMArgs 内存布局：

  values 数组                    type_codes 数组
  ┌──────────────┐              ┌──────────────┐
  │ TVMValue[0]  │ ──────────── │ type_code[0] │  kDLInt
  │ TVMValue[1]  │ ──────────── │ type_code[1] │  kTVMStr
  │ TVMValue[2]  │ ──────────── │ type_code[2] │  kDLTensor
  │ TVMValue[3]  │ ──────────── │ type_code[3] │  kTVMPackedFuncHandle
  └──────────────┘              └──────────────┘
         │                              │
         └──────────┬───────────────────┘
                    │
              num_args = 4
```

### 24.2.3 类型代码（Type Codes）

每个参数都附带一个类型代码，用于运行时类型检查：

```cpp
// include/tvm/runtime/c_runtime_api.h
enum TVMTypeCode {
  kDLInt = 0,           // 整数
  kDLUInt = 1,          // 无符号整数
  kDLFloat = 2,         // 浮点数
  kTVMObjectHandle = 3, // TVM 对象
  kTVMObjectRValueRef = 4,
  kTVMNullptr = 5,
  kTVMDataType = 6,     // DLDataType
  kDLTensor = 7,        // DLTensor
  kTVMContext = 8,      // DLDevice
  kTVMPackedFuncHandle = 9, // PackedFunc
  kTVMStr = 10,         // 字符串
  kTVMBytes = 11,       // 字节数组
  kTVMNDArrayHandle = 12,  // NDArray
  kTVMObjectHandle = 13,   // Object
};
```

类型代码的完整映射表：

| 类型代码 | 常量名 | C++ 类型 | Python 类型 | 用途 |
|---------|--------|---------|-------------|------|
| 0 | `kDLInt` | `int64_t` | `int` | 整数参数 |
| 1 | `kDLUInt` | `uint64_t` | `int` | 无符号整数 |
| 2 | `kDLFloat` | `double` | `float` | 浮点参数 |
| 3 | `kTVMObjectHandle` | `Object*` | `Object` | TVM 对象 |
| 5 | `kTVMNullptr` | `nullptr` | `None` | 空值 |
| 6 | `kTVMDataType` | `DLDataType` | `DataType` | 数据类型描述 |
| 7 | `kDLTensor` | `DLTensor*` | `Tensor` | 张量 |
| 8 | `kTVMContext` | `DLDevice` | `Device` | 设备上下文 |
| 9 | `kTVMPackedFuncHandle` | `PackedFunc*` | `PackedFunc` | 函数句柄 |
| 10 | `kTVMStr` | `const char*` | `str` | 字符串 |
| 11 | `kTVMBytes` | `const char*` | `bytes` | 字节数组 |
| 12 | `kTVMNDArrayHandle` | `NDArray*` | `nd.NDArray` | NDArray |
| 13 | `kTVMObjectHandle` | `Object*` | `Object` | 通用对象 |

<div data-component="TypeCodeMappingDiagram"></div>

### 24.2.4 TVMRetValue：返回值容器

```cpp
class TVMRetValue {
 public:
  TVMValue value_;
  int type_code_;

  // 设置返回值
  TVMRetValue& operator=(int64_t value) {
    value_.v_int64 = value;
    type_code_ = kDLInt;
    return *this;
  }

  TVMRetValue& operator=(const std::string& value) {
    // 字符串需要拷贝
    value_.v_str = strdup(value.c_str());
    type_code_ = kTVMStr;
    return *this;
  }

  TVMRetValue& operator=(const PackedFunc& value) {
    value_.v_handle = new PackedFunc(value);
    type_code_ = kTVMPackedFuncHandle;
    return *this;
  }

  TVMRetValue& operator=(NDArray value) {
    // NDArray 是引用传递
    value_.v_handle = new NDArray(value);
    type_code_ = kTVMNDArrayHandle;
    return *this;
  }
};
```

TVMRetValue 的赋值语义：

| 类型 | 赋值方式 | 内存管理 |
|------|---------|---------|
| `int64_t` | 值拷贝 | 无需释放 |
| `double` | 值拷贝 | 无需释放 |
| `std::string` | `strdup` 拷贝 | 需要 `free` |
| `PackedFunc` | `new` 分配 | 需要 `delete` |
| `NDArray` | 引用计数 | 自动管理 |
| `Object` | 引用计数 | 自动管理 |

### 24.2.5 类型转换的实现

```cpp
// TVMArgValue 提供类型安全的转换
class TVMArgValue : public TVMPODValue_ {
 public:
  // 隐式转换为 int
  operator int() const {
    CHECK(type_code_ == kDLInt || type_code_ == kDLUInt)
        << "Expected int, got " << type_code_;
    return static_cast<int>(value_.v_int64);
  }

  // 隐式转换为 NDArray
  operator NDArray() const {
    if (type_code_ == kTVMNullptr) return NDArray();
    CHECK(type_code_ == kTVMNDArrayHandle || type_code_ == kDLTensor);
    return *static_cast<NDArray*>(value_.v_handle);
  }

  // 隐式转换为 PackedFunc
  operator PackedFunc() const {
    if (type_code_ == kTVMNullptr) return PackedFunc();
    CHECK(type_code_ == kTVMPackedFuncHandle);
    return *static_cast<PackedFunc*>(value_.v_handle);
  }
};
```

类型转换的完整流程：

```
类型转换流程：

  TVMArgValue
       │
       ▼
  ┌─────────────────────────────────────┐
  │ 检查 type_code_ 是否匹配目标类型     │
  │ CHECK(type_code_ == expected_type)  │
  └─────────────────────────────────────┘
       │
       ├── 匹配 → 直接转换
       │
       └── 不匹配 → 抛出错误
              │
              ▼
       ┌─────────────────────────────────────┐
       │ 错误信息：                           │
       │ "Expected <type>, got <actual_type>" │
       └─────────────────────────────────────┘
```

### 24.2.6 TVMPODValue_ 基类

```cpp
// include/tvm/runtime/packed_func.h
class TVMPODValue_ {
 public:
  // 基本类型的转换
  operator int() const {
    return static_cast<int>(value_.v_int64);
  }
  operator int64_t() const {
    return value_.v_int64;
  }
  operator double() const {
    return value_.v_float64;
  }
  operator float() const {
    return static_cast<float>(value_.v_float64);
  }
  operator bool() const {
    return value_.v_int64 != 0;
  }
  operator DLDevice() const {
    return value_.v_ctx;
  }
  operator DLDataType() const {
    return value_.v_type;
  }

 protected:
  TVMValue value_;
  int type_code_;
};
```

<div data-component="TypeConversionDiagram"></div>

### 24.2.7 PackedFunc 的调用开销分析

PackedFunc 的单次调用开销：

$$
T_{\text{packed}} = T_{\text{pack}} + T_{\text{dispatch}} + T_{\text{unpack}} + T_{\text{body}}
$$

其中：
- $T_{\text{pack}}$：参数打包时间（~2 ns）
- $T_{\text{dispatch}}$：函数分发时间（~3 ns）
- $T_{\text{unpack}}$：参数解包时间（~2 ns）
- $T_{\text{body}}$：函数体执行时间

对于简单的算术运算，$T_{\text{body}}$ 可能只有 1-2 ns，而打包/解包开销占主导。但对于实际的计算任务（如矩阵乘法），$T_{\text{body}}$ 通常在毫秒级别，打包/解包开销可忽略。

---

## 24.3 函数注册系统

### 24.3.1 全局注册表

TVM 维护一个**全局函数注册表**，所有通过 PackedFunc 暴露的函数都可以通过名称查找：

```cpp
// include/tvm/runtime/registry.h
class Registry {
 public:
  // 注册函数
  Registry& set_body(PackedFunc f);

  // 注册函数（模板版本，自动包装）
  template <typename F>
  Registry& set_body(F f);

  // 全局查找
  static const PackedFunc* Get(const std::string& name);

  // 列出所有注册的函数
  static std::vector<std::string> ListNames();

 private:
  static std::unordered_map<std::string, Registry*>& Global();
};
```

注册表的内部数据结构：

```
全局注册表（Registry::Global()）：

  std::unordered_map<std::string, Registry*>
  ┌─────────────────────────────────────────────────────────────┐
  │ "relay._transform.FuseOps" ──→ Registry{ PackedFunc }      │
  │ "target._ffi_api.TargetKindGet" ──→ Registry{ PackedFunc } │
  │ "runtime._ffi_api.GraphExecutorCreate" ──→ Registry{ ... } │
  │ "tvm.contrib.graph_executor.create" ──→ Registry{ ... }    │
  │ "runtime.cuda.mem_get_info" ──→ Registry{ PackedFunc }     │
  │ ...                                                         │
  └─────────────────────────────────────────────────────────────┘
```

### 24.3.2 注册宏

```cpp
// C++ 端注册函数
TVM_REGISTER_GLOBAL("my_module.my_function")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  int64_t a = args[0];
  int64_t b = args[1];
  *ret = a + b;
});

// Python 端调用
import tvm
result = tvm.get_global_func("my_module.my_function")(3, 4)  # 返回 7
```

`TVM_REGISTER_GLOBAL` 宏展开后的代码：

```cpp
// TVM_REGISTER_GLOBAL 宏的实际展开
static TVM_FUNC_REG_VAR_DEF_##__LINE__ __attribute__((used)) = \
  TVM_FUNC_REG_VAR_DEF_##__LINE__##_body()

// 简化理解：
// 1. 创建一个静态 Registry 对象
// 2. 在程序启动时自动执行注册
// 3. 注册的函数在整个程序生命周期内有效
```

### 24.3.3 注册表的使用场景

| 注册函数名 | 用途 | 源文件 |
|-----------|------|--------|
| `relay._transform.FuseOps` | Relay Pass 注册 | `src/relay/transforms/fuse_ops.cc` |
| `target._ffi_api.TargetKindGet` | Target 系统注册 | `src/target/target_kind.cc` |
| `runtime._ffi_api.GraphExecutorCreate` | 运行时模块注册 | `src/runtime/graph_executor/graph_executor.cc` |
| `tvm.contrib.graph_executor.create` | Python 端 GraphExecutor | `python/tvm/contrib/graph_executor.py` |
| `runtime.cuda.mem_get_info` | CUDA 内存查询 | `src/runtime/cuda/cuda_device_api.cc` |
| `runtime.ndarray.alloc` | NDArray 分配 | `src/runtime/ndarray.cc` |
| `runtime.DetectDeviceType` | 设备类型检测 | `src/runtime/runtime_base.cc` |

### 24.3.4 条件注册

```cpp
// 仅在启用 CUDA 时注册
#ifdef TVM_CUDA_RUNTIME
TVM_REGISTER_GLOBAL("runtime.cuda.mem_get_info")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  // CUDA 特定实现
});
#endif
```

条件注册的完整示例：

```cpp
// src/runtime/cuda/cuda_device_api.cc
#ifdef TVM_CUDA_RUNTIME

TVM_REGISTER_GLOBAL("runtime.cuda.mem_get_info")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  size_t free_mem, total_mem;
  CUDA_CALL(cudaMemGetInfo(&free_mem, &total_mem));
  // 返回 [free_mem, total_mem]
  TVMRetValue* rv = static_cast<TVMRetValue*>(ret);
  rv[0] = static_cast<int64_t>(free_mem);
  rv[1] = static_cast<int64_t>(total_mem);
});

TVM_REGISTER_GLOBAL("runtime.cuda.device_count")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  int count;
  CUDA_CALL(cudaGetDeviceCount(&count));
  *ret = count;
});

#endif  // TVM_CUDA_RUNTIME
```

### 24.3.5 注册表的线程安全

```cpp
// include/tvm/runtime/registry.h
class Registry {
 public:
  // 注册表是线程安全的
  // 使用读写锁保护
  static const PackedFunc* Get(const std::string& name) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = Global().find(name);
    if (it != Global().end()) {
      return it->second->func_;
    }
    return nullptr;
  }

 private:
  static std::shared_mutex mutex_;
};
```

### 24.3.6 注册表的命名约定

TVM 使用**分层命名**来组织注册的函数：

```
命名约定：

  <module>.<submodule>.<function>

  示例：
  relay._transform.FuseOps          → Relay 变换 Pass
  target._ffi_api.TargetKindGet     → Target FFI API
  runtime._ffi_api.GraphExecutorCreate → 运行时 FFI API
  tvm.contrib.graph_executor.create → Python 封装
  tir.Schedule.Apply                 → TIR 调度操作
  meta_schedule.SearchStrategy       → MetaSchedule 搜索策略
```

### 24.3.7 注册表的遍历与调试

```python
import tvm

# 列出所有注册的函数
all_funcs = tvm.get_global_func("runtime.ListGlobalFuncNames")()
print(f"Total registered functions: {len(all_funcs)}")

# 按前缀过滤
relay_funcs = [f for f in all_funcs if f.startswith("relay.")]
print(f"Relay functions: {len(relay_funcs)}")

# 查找特定函数
func = tvm.get_global_func("relay._transform.FuseOps")
print(f"Found: {func}")
```

---

## 24.4 Module 系统

### 24.4.1 Module 的设计

Module 是 TVM 的**代码组织单元**，每个编译产物（如 LLVM 库、CUDA 库）被封装为一个 Module。Module 通过 PackedFunc 接口暴露其功能。

```cpp
// include/tvm/runtime/module.h
class Module : public ObjectRef {
 public:
  // 获取模块中的函数
  PackedFunc GetFunction(const std::string& name, bool query_import = false);

  // 导入子模块
  void Import(Module module);

  // 模块类型
  const char* type_key() const;

  // 获取源码（可选）
  std::string GetSource(const std::string& format = "") const;
};

class ModuleNode : public Object {
 public:
  virtual PackedFunc GetFunction(const std::string& name,
                                  const ObjectPtr<Object>& sptr_to_self) = 0;
  virtual const char* type_key() const = 0;
  virtual void SaveToFile(const std::string& file_name,
                           const std::string& format) {}
  virtual std::string GetSource(const std::string& format) { return ""; }

 protected:
  // 子模块导入列表
  std::vector<Module> imports_;
};
```

Module 的类层次结构：

```
Module 类层次结构：

  ObjectRef
    │
    └── Module
          │
          └── ModuleNode (抽象基类)
                │
                ├── DSOModule
                ├── CUDAModule
                ├── LLVMModule
                ├── SourceModule
                ├── GraphExecutorFactory
                ├── RPCModule
                └── CompositeModule
```

### 24.4.2 内置 Module 类型

| Module 类型 | 位置 | 说明 | 文件格式 |
|------------|------|------|---------|
| `DSOModule` | `src/runtime/dso_module.cc` | 动态链接库 | `.so`, `.dylib`, `.dll` |
| `CUDAModule` | `src/runtime/cuda/cuda_module.cc` | CUDA PTX/CUBIN | `.ptx`, `.cubin` |
| `LLVMModule` | `src/target/llvm/llvm_module.cc` | LLVM 编译的机器码 | `.o`, `.so` |
| `SourceModule` | `src/runtime/source_module.cc` | 源码模块 | `.c`, `.cu` |
| `GraphExecutorFactory` | `src/runtime/graph_executor/` | 图执行器工厂 | - |
| `RPCModule` | `src/runtime/rpc/` | RPC 远程模块 | - |
| `CompositeModule` | `src/runtime/module.cc` | 组合模块 | - |
| `VMModule` | `src/runtime/vm/` | 虚拟机模块 | - |

<div data-component="ModuleHierarchyDiagram"></div>

### 24.4.3 Module 的创建与使用

```python
import tvm
from tvm import runtime

# 编译得到 Module
lib = relay.build(mod, target="llvm")
# lib 是一个 GraphExecutorFactory Module

# 获取函数
default_func = lib["default"]  # 返回 PackedFunc
# 调用函数创建 GraphExecutor
gmod = default_func(dev)  # dev = tvm.cuda(0)

# Module 的嵌套导入
# GraphExecutorFactory 内部 import 了 DSO Module（包含编译的算子）
```

### 24.4.4 Module 的序列化

```python
# 保存 Module 到文件
lib.export_library("model.tar")

# 加载 Module
loaded_lib = runtime.load_module("model.tar")

# .tar 内部结构：
# model.tar/
#   ├── lib.so          # 编译的机器码
#   ├── graph.json      # 图结构
#   ├── params.bin      # 模型参数
#   └── metadata.json   # 元信息
```

### 24.4.5 Module 的导入机制

Module 支持**嵌套导入**，形成一个模块树：

```
Module 导入树：

  GraphExecutorFactory (根模块)
    │
    ├── import → DSOModule (lib.so)
    │              │
    │              ├── import → CUDAModule (cuda_kernels.ptx)
    │              │
    │              └── import → LLVMModule (cpu_ops.o)
    │
    └── import → SourceModule (util.c)
```

```cpp
// Module 导入的实现
void ModuleNode::Import(Module other) {
  // 检查循环导入
  for (auto& m : imports_) {
    CHECK_NE(m.operator->(), other.operator->())
        << "Attempt to cyclically import";
  }
  imports_.push_back(other);
}

// GetFunction 的查找顺序
PackedFunc ModuleNode::GetFunction(const std::string& name,
                                    bool query_import) {
  // 1. 先在当前模块查找
  PackedFunc f = GetFunction(name, sptr_to_self);
  if (f != nullptr) return f;

  // 2. 如果 query_import 为 true，递归查找导入的模块
  if (query_import) {
    for (auto& m : imports_) {
      f = m.GetFunction(name, true);
      if (f != nullptr) return f;
    }
  }
  return nullptr;
}
```

### 24.4.6 DSOModule 的实现

```cpp
// src/runtime/dso_module.cc
class DSOModuleNode final : public ModuleNode {
 public:
  // 从文件加载动态库
  static Module Create(const std::string& file_name) {
    auto n = make_object<DSOModuleNode>();
    n->lib_handle_ = dlopen(file_name.c_str(), RTLD_LAZY);
    CHECK(n->lib_handle_ != nullptr)
        << "Failed to load library: " << dlerror();
    n->file_name_ = file_name;
    return Module(n);
  }

  PackedFunc GetFunction(const std::string& name,
                          const ObjectPtr<Object>& sptr_to_self) override {
    // 从动态库中查找符号
    TVMPackedFunc pf;
    pf->name_ = name;
    pf->body_ = reinterpret_cast<TVMPackedCFunc>(
        dlsym(lib_handle_, name.c_str()));
    CHECK(pf->body_ != nullptr)
        << "Symbol " << name << " not found";
    return PackedFunc(pf);
  }

 private:
  void* lib_handle_;
  std::string file_name_;
};
```

### 24.4.7 CUDAModule 的实现

```cpp
// src/runtime/cuda/cuda_module.cc
class CUDAModuleNode : public ModuleNode {
 public:
  CUDAModuleNode(std::string data, std::string fmt,
                 std::unordered_map<std::string, FunctionInfo> fmap,
                 std::string source)
      : data_(data), fmt_(fmt), fmap_(fmap), source_(source) {}

  PackedFunc GetFunction(const std::string& name,
                          const ObjectPtr<Object>& sptr_to_self) override {
    // 查找函数信息
    auto it = fmap_.find(name);
    CHECK(it != fmap_.end()) << "Function " << name << " not found";

    // 获取 CUDA 函数句柄
    CUfunction func = GetFunction_(name);

    // 返回一个 PackedFunc，封装 CUDA kernel 调用
    auto f = [this, func](TVMArgs args, TVMRetValue* rv) {
      // 设置 kernel 参数
      void* kernel_args[args.num_args];
      for (int i = 0; i < args.num_args; i++) {
        kernel_args[i] = args.values[i].v_handle;
      }
      // 启动 kernel
      cuLaunchKernel(func, ...);
    };
    return PackedFunc(f);
  }

 private:
  std::string data_;  // PTX 或 CUBIN 数据
  std::string fmt_;   // "ptx" 或 "cubin"
  // ...
};
```

---

## 24.5 FFI（Foreign Function Interface）

### 24.5.1 TVM 的 FFI 设计

TVM 的 FFI 允许 Python、Java、Rust 等语言调用 C++ 实现的 PackedFunc，反之亦然：

```
Python 代码
  │  tvm.get_global_func("relay.build")
  ▼
FFI 层（ctypes / Cython / FFI）
  │  TVMFuncCall()
  ▼
C++ Registry
  │  Registry::Get("relay.build")
  ▼
PackedFunc 实现
```

### 24.5.2 Python FFI 实现

```python
# python/tvm/_ffi/_ctypes/packed_func.py
class PackedFunc:
    """Python 包装的 PackedFunc"""

    def __init__(self, handle):
        self.handle = handle

    def __call__(self, *args):
        # 1. 将 Python 参数转换为 TVMArgs
        tvm_args = convert_to_tvm_args(args)

        # 2. 调用 C++ 函数
        ret_val = TVMRetValue()
        check_call(_LIB.TVMFuncCall(
            self.handle,
            tvm_args.values,
            tvm_args.type_codes,
            tvm_args.num_args,
            ctypes.byref(ret_val)
        ))

        # 3. 将返回值转换为 Python 对象
        return convert_from_tvm_ret(ret_val)
```

### 24.5.3 Cython FFI（高性能路径）

对于性能关键的调用，TVM 使用 Cython 实现直接调用：

```cython
# python/tvm/_ffi/_cython/packed_func.pyc
cdef class PackedFunc:
    cdef TVMFuncHandle chandle

    def __call__(self, *args):
        cdef TVMValue[10] values
        cdef int type_codes[10]
        cdef int num_args = len(args)

        # 直接在 Cython 层转换参数，避免 Python 开销
        for i in range(num_args):
            convert_arg(args[i], &values[i], &type_codes[i])

        # 直接调用 C 函数
        cdef TVMRetValue ret
        CHECK_CALL(TVMFuncCall(self.chandle, values, type_codes,
                               num_args, &ret))
        return convert_ret(&ret)
```

### 24.5.4 FFI 的性能开销

| 调用路径 | 典型延迟 | 相对开销 |
|---------|---------|---------|
| C++ → C++ PackedFunc | ~10 ns | 1x |
| Python (Cython) → C++ PackedFunc | ~200 ns | 20x |
| Python (ctypes) → C++ PackedFunc | ~1 μs | 100x |
| RPC → 远程 PackedFunc | ~100 μs | 10000x |

对于单次算子调用（如矩阵乘法），FFI 开销可忽略；但对于大量细粒度调用，应使用批量化接口。

### 24.5.5 Python 参数类型映射

| Python 类型 | TVM 类型代码 | 转换方式 |
|------------|-------------|---------|
| `int` | `kDLInt` | 直接转换 |
| `float` | `kDLFloat` | 直接转换 |
| `str` | `kTVMStr` | 编码为 UTF-8 |
| `bytes` | `kTVMBytes` | 直接传递 |
| `None` | `kTVMNullptr` | 特殊处理 |
| `tvm.nd.NDArray` | `kTVMNDArrayHandle` | 传递句柄 |
| `tvm.runtime.PackedFunc` | `kTVMPackedFuncHandle` | 传递句柄 |
| `tvm.runtime.Object` | `kTVMObjectHandle` | 传递句柄 |
| `tvm.Device` | `kTVMContext` | 直接转换 |
| `tvm.DataType` | `kTVMDataType` | 直接转换 |

<div data-component="FFITypeMappingDiagram"></div>

### 24.5.6 TVMFuncRegister：C++ 注册 Python 回调

```cpp
// include/tvm/runtime/c_runtime_api.h
TVM_DLL int TVMFuncRegisterGlobal(
    const char* name,
    TVMPackedFunc func,
    int override);
```

```python
# Python 端注册回调到 C++
@tvm.register_func("my_python_callback")
def my_callback(x, y):
    return x + y

# C++ 端可以调用这个 Python 函数
# 通过 Registry::Get("my_python_callback")
```

### 24.5.7 FFI 的内存管理

```
FFI 内存管理：

  Python 参数 → C++ 参数
  ┌─────────────────────────────────────────────────────┐
  │ 1. Python 对象引用计数 +1（防止 GC）                   │
  │ 2. 转换为 TVMValue（值拷贝或句柄传递）                  │
  │ 3. 调用完成后，Python 对象引用计数 -1                   │
  └─────────────────────────────────────────────────────┘

  C++ 返回值 → Python 返回值
  ┌─────────────────────────────────────────────────────┐
  │ 1. TVMRetValue 包含返回值                             │
  │ 2. 转换为 Python 对象（NDArray 等使用引用计数）         │
  │ 3. TVMRetValue 清理（释放临时内存）                     │
  └─────────────────────────────────────────────────────┘
```

---

## 24.6 RPC 机制

### 24.6.1 RPC 的设计动机

TVM 的 RPC（Remote Procedure Call）机制允许在**开发机上编写代码**，在**远程设备上执行**。这对于以下场景至关重要：

1. **嵌入式设备调试**：在 PC 上编写 TVM 代码，在手机/开发板上执行
2. **多设备测试**：一台 PC 控制多台 GPU 服务器
3. **自动调优**：AutoTVM/MetaSchedule 需要在目标设备上评估性能
4. **资源隔离**：编译在 CPU 上完成，执行在 GPU 上完成

源码位置：`src/runtime/rpc/`

### 24.6.2 RPC 架构

```
RPC Client（开发机）                RPC Server（目标设备）
┌─────────────────┐               ┌─────────────────┐
│  Python 代码     │               │  RPC Server 进程 │
│  tvm.rpc.connect │──── 网络 ────→│  监听端口        │
│                  │               │                  │
│  调用远程函数     │─── 请求 ────→│  执行函数        │
│  接收返回值      │←── 响应 ────│  返回结果        │
│                  │               │                  │
│  传输数据        │─── 数据流 ──→│  接收数据        │
└─────────────────┘               └─────────────────┘
```

### 24.6.3 关键源文件

| 文件 | 职责 |
|------|------|
| `src/runtime/rpc/rpc_endpoint.cc` | RPC 端点（协议层） |
| `src/runtime/rpc/rpc_server.cc` | RPC Server 实现 |
| `src/runtime/rpc/rpc_client.cc` | RPC Client 实现 |
| `src/runtime/rpc/rpc_session.cc` | RPC Session 管理 |
| `src/runtime/rpc/rpc_socket_impl.cc` | Socket 传输层 |
| `src/runtime/rpc/rpc_channel.h` | 传输层抽象接口 |
| `python/tvm/rpc/client.py` | Python Client 封装 |
| `python/tvm/rpc/server.py` | Python Server 封装 |
| `python/tvm/rpc/tracker.py` | Tracker 封装 |

<div data-component="RPCArchitectureDiagram"></div>

### 24.6.4 RPC 调用的完整流程

```
RPC 调用完整流程：

  Client 端                              Server 端
  ┌─────────────────────┐               ┌─────────────────────┐
  │ 1. 构造 TVMArgs     │               │                     │
  │ 2. 序列化参数        │               │                     │
  │ 3. 发送 CALL_FUNC   │───────────────→│ 4. 接收 CALL_FUNC   │
  │    消息             │               │ 5. 反序列化参数      │
  │                     │               │ 6. 查找 PackedFunc  │
  │                     │               │ 7. 执行函数         │
  │                     │               │ 8. 序列化返回值      │
  │                     │←──────────────│ 9. 发送 RET_VALUE   │
  │ 10. 接收 RET_VALUE  │               │                     │
  │ 11. 反序列化返回值   │               │                     │
  │ 12. 返回结果        │               │                     │
  └─────────────────────┘               └─────────────────────┘
```

---

## 24.7 RPC Server

### 24.7.1 启动 RPC Server

```python
# 方式一：Python 启动
from tvm.rpc import Server

server = Server(host="0.0.0.0", port=9090, key="my_device")
server.start()  # 阻塞等待连接

# 方式二：命令行启动
# python -m tvm.exec.rpc_server --host 0.0.0.0 --port 9090 --key my_device

# 方式三：Tracker 模式（多设备管理）
server = Server(host="0.0.0.0", port=9090, key="my_gpu",
                tracker_addr=("tracker_host", 9190))
```

### 24.7.2 Server 的初始化

```cpp
// src/runtime/rpc/rpc_server.cc
class RPCServer {
 public:
  RPCServer(int port, const std::string& key) {
    // 1. 创建 socket 监听
    listen_socket_ = CreateTCPSocket(port);

    // 2. 注册本地函数到 RPC
    // 所有注册的 PackedFunc 都可以通过 RPC 调用
    this->RegisterServerFunctions();
  }

  void Run() {
    while (true) {
      // 3. 接受连接
      auto conn = listen_socket_->Accept();

      // 4. 创建 RPC Session
      auto session = std::make_shared<RPCSession>(conn);

      // 5. 处理请求循环
      session->ServerLoop();
    }
  }

  void RegisterServerFunctions() {
    // 注册关键的运行时函数
    // runtime.GraphExecutorCreate
    // runtime.NDArrayCreate
    // runtime.DeviceAllocData
    // ...
  }
};
```

### 24.7.3 安全考虑

```python
# RPC Server 应仅在受信任的网络中运行
# 支持 key 认证
server = Server(host="0.0.0.0", port=9090, key="secret_key")

# Client 连接时需要提供 key
conn = client.connect("server_ip", 9090, key="secret_key")
```

### 24.7.4 Server 的生命周期状态

```
RPC Server 状态机：

  ┌──────────┐
  │  INIT    │  初始化 socket、注册函数
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │ LISTENING│  等待连接
  └────┬─────┘
       │ Accept()
       ▼
  ┌──────────┐
  │CONNECTED │  处理单个连接
  └────┬─────┘
       │
       ├──→ 处理请求 → 继续循环
       │
       └──→ 连接断开 → 回到 LISTENING
```

### 24.7.5 Server 的错误处理

```cpp
// src/runtime/rpc/rpc_server.cc
void RPCSession::ServerLoop() {
  try {
    while (true) {
      // 读取消息类型
      int msg_type = RecvMsgType();

      switch (msg_type) {
        case kCallFunc: {
          // 处理函数调用
          HandleFuncCall();
          break;
        }
        case kCopyToRemote: {
          // 处理数据上传
          HandleCopyToRemote();
          break;
        }
        case kShutdown: {
          // 关闭连接
          return;
        }
        default: {
          LOG(ERROR) << "Unknown message type: " << msg_type;
          SendError("Unknown message type");
          break;
        }
      }
    }
  } catch (const std::exception& e) {
    // 发送异常信息到 Client
    SendException(e.what());
  }
}
```

### 24.7.6 Server 的并发模型

```
并发模型：

  单线程模型（默认）：
  ┌─────────────────────────────────────┐
  │  Server 主线程                       │
  │    ├── Accept() 接受连接              │
  │    ├── 处理请求                       │
  │    └── 返回结果                       │
  │  （一次只能处理一个连接）              │
  └─────────────────────────────────────┘

  多线程模型：
  ┌─────────────────────────────────────┐
  │  Server 主线程                       │
  │    ├── Accept() 接受连接              │
  │    └── 创建工作线程                   │
  │         ├── Thread 1: 处理连接 1      │
  │         ├── Thread 2: 处理连接 2      │
  │         └── Thread 3: 处理连接 3      │
  └─────────────────────────────────────┘
```

---

## 24.8 RPC Client

### 24.8.1 连接到 RPC Server

```python
from tvm import rpc

# 直接连接
conn = rpc.connect("192.168.1.100", 9090)

# 通过 Tracker 连接
conn = rpc.connect_tracker("tracker_host:9190").request("my_gpu")

# 连接后可以像本地一样使用远程设备
remote_dev = conn.gpu(0)  # 远程 GPU
remote_cpu = conn.cpu(0)  # 远程 CPU
```

### 24.8.2 远程函数调用

```python
# 获取远程函数
remote_func = conn.get_function("runtime.GraphExecutorCreate")

# 调用远程函数（通过 RPC 传输参数和返回值）
result = remote_func(graph_json, lib, remote_dev)

# 传输数据到远程
local_arr = tvm.nd.array(np.random.randn(10, 20))
remote_arr = conn.upload(local_arr)  # 通过网络传输

# 从远程下载数据
local_arr = remote_arr.download()  # 通过网络传输
```

### 24.8.3 远程编译与执行

```python
# 完整的远程编译执行流程
import tvm
from tvm import relay, rpc
import numpy as np

# 1. 连接到远程设备
conn = rpc.connect("192.168.1.100", 9090)
remote_dev = conn.gpu(0)

# 2. 在本地编译
mod, params = relay.frontend.from_onnx(model)
with tvm.target.Target("cuda"):
    lib = relay.build(mod, target="cuda", params=params)

# 3. 上传编译产物到远程
remote_lib = conn.upload(lib)

# 4. 在远程创建执行器
remote_gmod = conn.graph_executor.create(
    graph_json, remote_lib, remote_dev
)

# 5. 上传输入数据
input_data = np.random.randn(1, 3, 224, 224).astype("float32")
remote_input = conn.upload(tvm.nd.array(input_data, remote_dev))

# 6. 执行推理
remote_gmod.set_input("input", remote_input)
remote_gmod.run()

# 7. 下载输出
output = remote_gmod.get_output(0).download().numpy()
```

<div data-component="RemoteExecutionFlow"></div>

### 24.8.4 RemoteModule 的使用

```python
# RemoteModule 提供更高层的抽象
from tvm import rpc

conn = rpc.connect("192.168.1.100", 9090)

# 直接在远程加载模块
remote_lib = conn.load_module("model.tar")

# 创建远程执行器
remote_gmod = conn.graph_executor.create(
    graph_json, remote_lib, conn.gpu(0)
)

# 使用 RemoteModule 的便利方法
remote_mod = rpc.create_remote_module(
    "192.168.1.100:9090",
    lib,
    session_timeout=60
)
```

### 24.8.5 RPC 连接池

```python
# 连接池管理（用于高并发场景）
from tvm.rpc import RPCSessionPool

pool = RPCSessionPool(
    host="192.168.1.100",
    port=9090,
    max_connections=10
)

# 从池中获取连接
with pool.get_connection() as conn:
    remote_func = conn.get_function("my_func")
    result = remote_func(args)
```

---

## 24.9 RPC 协议

### 24.9.1 消息格式

RPC 协议使用**二进制消息格式**，包含消息头和消息体：

```
消息格式：
  ┌─────────────────────────────────┐
  │ 消息类型 (1 byte)               │
  │ 消息长度 (4 bytes, big-endian)  │
  │ 消息体 (变长)                    │
  └─────────────────────────────────┘

消息类型：
  CALL_FUNC    = 1   // 调用函数
  RET_VALUE    = 2   // 返回值
  EXCEPTION    = 3   // 异常
  COPY_TO_REMOTE = 4 // 上传数据
  COPY_FROM_REMOTE = 5 // 下载数据
  SESSION_INIT = 6   // 会话初始化
  SHUTDOWN     = 7   // 关闭连接
```

### 24.9.2 参数编码

```cpp
// src/runtime/rpc/rpc_session.cc
void RPCSession::SendPackedArgs(const TVMArgs& args) {
  // 1. 发送参数数量
  SendInt(args.num_args);

  // 2. 发送类型代码数组
  for (int i = 0; i < args.num_args; i++) {
    SendInt(args.type_codes[i]);
  }

  // 3. 发送参数值
  for (int i = 0; i < args.num_args; i++) {
    switch (args.type_codes[i]) {
      case kDLInt:
        SendInt(args.values[i].v_int64);
        break;
      case kDLFloat:
        SendFloat(args.values[i].v_float64);
        break;
      case kTVMStr:
        SendString(args.values[i].v_str);
        break;
      case kTVMNDArrayHandle:
        SendNDArray(args.values[i].v_handle);
        break;
      // ... 其他类型
    }
  }
}
```

### 24.9.3 NDArray 传输协议

NDArray 的传输分为**元信息**和**数据**两部分：

```
NDArray 传输格式：
  ┌──────────────────────────┐
  │ ndim (int32)             │  元信息
  │ dtype (int32)            │
  │ shape[0..ndim] (int64[]) │
  │ device_type (int32)      │
  │ device_id (int32)        │
  ├──────────────────────────┤
  │ data (raw bytes)         │  数据
  └──────────────────────────┘
```

NDArray 传输的详细步骤：

```
NDArray 上传流程：

  Client 端                              Server 端
  ┌─────────────────────┐               ┌─────────────────────┐
  │ 1. 获取 NDArray 元信息│               │                     │
  │    - ndim, dtype     │               │                     │
  │    - shape, device   │               │                     │
  │                     │               │                     │
  │ 2. 发送 COPY_TO_    │───────────────→│ 3. 接收消息类型      │
  │    REMOTE 消息       │               │                     │
  │                     │               │                     │
  │ 4. 发送元信息        │───────────────→│ 5. 解析元信息        │
  │                     │               │                     │
  │ 6. 发送数据          │───────────────→│ 7. 分配内存并接收    │
  │                     │               │                     │
  │                     │←──────────────│ 8. 返回 NDArray 句柄 │
  └─────────────────────┘               └─────────────────────┘
```

### 24.9.4 会话初始化协议

```
会话初始化流程：

  Client                              Server
  ┌─────────────────┐               ┌─────────────────┐
  │ 1. 发送 key     │───────────────→│ 2. 验证 key     │
  │                 │               │                 │
  │                 │←──────────────│ 3. 发送确认      │
  │ 4. 接收确认     │               │                 │
  │                 │               │                 │
  │ 5. 交换版本信息  │←─────────────→│ 6. 版本兼容检查  │
  │                 │               │                 │
  │ 7. 会话建立     │               │ 8. 进入请求循环  │
  └─────────────────┘               └─────────────────┘
```

### 24.9.5 错误处理协议

```
错误处理流程：

  Client                              Server
  ┌─────────────────┐               ┌─────────────────┐
  │ 1. 发送 CALL_FUNC│──────────────→│ 2. 执行函数     │
  │                 │               │    发生异常      │
  │                 │               │                 │
  │                 │←──────────────│ 3. 发送 EXCEPTION│
  │ 4. 接收异常     │               │    消息         │
  │                 │               │                 │
  │ 5. 重新抛出异常  │               │                 │
  └─────────────────┘               └─────────────────┘

异常消息格式：
  ┌──────────────────────────┐
  │ error_code (int32)       │
  │ error_msg_length (int32) │
  │ error_msg (string)       │
  └──────────────────────────┘
```

---

## 24.10 RPC Tracker：多设备管理

### 24.10.1 Tracker 的角色

RPC Tracker 是一个**中央调度器**，管理多个 RPC Server 的注册和分配：

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Server A │     │ Server B │     │ Server C │
│ (GPU 0)  │     │ (GPU 1)  │     │ (CPU)    │
└────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │
     └────────────────┼────────────────┘
                      │
                ┌─────┴─────┐
                │  Tracker  │
                │ (port 9190)│
                └─────┬─────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
    ┌─────┴─────┐ ┌──┴───┐ ┌────┴────┐
    │ Client 1  │ │Client│ │Client 3 │
    └───────────┘ └──────┘ └─────────┘
```

### 24.10.2 Tracker 的使用

```python
from tvm.rpc import Tracker, connect_tracker

# 启动 Tracker
tracker = Tracker(host="0.0.0.0", port=9190)
tracker.start()

# Server 注册到 Tracker
server = Server(host="0.0.0.0", port=9090, key="my_gpu",
                tracker_addr=("localhost", 9190))

# Client 通过 Tracker 请求设备
tracker_client = connect_tracker("localhost", 9190)
conn = tracker_client.request("my_gpu", session_timeout=10)
```

### 24.10.3 设备池管理

Tracker 支持设备池的自动管理：

```python
# 查看可用设备
tracker = connect_tracker("localhost", 9190)
summary = tracker.summary()
print(summary)
# 输出：
# Server my_gpu: 1 GPU, 2 CPUs, load=0.3
# Server my_phone: 1 ARM CPU, load=0.0

# 按优先级请求
conn = tracker.request("my_gpu", priority=10, max_count=1)
```

### 24.10.4 Tracker 的内部数据结构

```cpp
// src/runtime/rpc/rpc_tracker.cc
class RPCTracker {
 public:
  // 服务器信息
  struct ServerInfo {
    std::string key;           // 设备标识
    std::string host;          // 服务器地址
    int port;                  // 服务器端口
    int max_connections;       // 最大连接数
    int current_connections;   // 当前连接数
    double load;               // 负载
    std::vector<std::string> devices;  // 设备列表
  };

  // 待处理的请求
  struct PendingRequest {
    std::string key;           // 请求的设备标识
    int priority;              // 优先级
    std::promise<Connection> promise;  // 结果
  };

 private:
  std::unordered_map<std::string, ServerInfo> servers_;
  std::priority_queue<PendingRequest> pending_requests_;
  std::mutex mutex_;
};
```

### 24.10.5 Tracker 的负载均衡

```
负载均衡策略：

  1. 最小负载优先（默认）：
     选择 current_connections / max_connections 最小的服务器

  2. 优先级调度：
     高优先级请求优先分配资源

  3. 设备亲和性：
     优先分配到特定类型的设备（如 GPU）

  负载计算公式：
  load = current_connections / max_connections

  选择策略：
  selected_server = argmin(load) for all servers with key match
```

<div data-component="TrackerLoadBalancing"></div>

---

## 24.11 RPC 在自动调优中的应用

### 24.11.1 AutoTVM 的 RPC 使用

AutoTVM 使用 RPC 在目标设备上评估调度策略的性能：

```python
import tvm
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner

# 定义调优任务
@autotvm.template("matmul")
def matmul_template(N, M, K):
    A = te.placeholder((N, K), name="A")
    B = te.placeholder((K, M), name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))

    s = te.create_schedule(C.op)
    # 定义搜索空间
    cfg = autotvm.get_config()
    cfg.define_knob("tile_x", [16, 32, 64, 128])
    cfg.define_knob("tile_y", [16, 32, 64, 128])
    # ...
    return s, [A, B, C]

# 使用 RPC 连接目标设备进行调优
measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.RPCRunner(
        key="my_gpu",
        host="192.168.1.100",
        port=9090,
        number=10,
        repeat=3,
        min_repeat_ms=100,
    )
)

# 执行调优
tuner = XGBTuner(tasks[0], loss_type="rank")
tuner.tune(n_trial=1000, measure_option=measure_option)
```

### 24.11.2 MetaSchedule 的 RPC 集成

```python
from tvm import meta_schedule as ms

# MetaSchedule 使用 RPC 进行远程评估
database = ms.tune_relay(
    mod=mod,
    target="cuda",
    config=ms.TuneConfig(
        max_trials_global=1000,
        strategy="replay-trace",
    ),
    work_dir="./tune_logs",
    # RPC 配置
    runner=ms.runner.RPCRunner(
        rpc_config=ms.runner.RPCConfig(
            tracker_host="192.168.1.100",
            tracker_port=9190,
            tracker_key="my_gpu",
        ),
    ),
)
```

<div data-component="AutoTuningRPCTiming"></div>

### 24.11.3 调优过程中的数据流

```
AutoTVM 调优数据流：

  开发机                              目标设备
  ┌─────────────────────┐            ┌─────────────────────┐
  │ 1. 生成调度配置      │            │                     │
  │    (Search Space)   │            │                     │
  │                     │            │                     │
  │ 2. 编译为 Module    │            │                     │
  │                     │            │                     │
  │ 3. 上传 Module      │────────────→│ 4. 加载 Module      │
  │    通过 RPC         │            │                     │
  │                     │            │ 5. 执行 benchmark   │
  │                     │            │    (多次运行取平均)   │
  │                     │            │                     │
  │                     │←────────────│ 6. 返回执行时间     │
  │ 7. 记录性能数据     │            │                     │
  │                     │            │                     │
  │ 8. 更新搜索策略     │            │                     │
  │    (XGBoost)       │            │                     │
  │                     │            │                     │
  │ 9. 重复 1-8        │            │                     │
  └─────────────────────┘            └─────────────────────┘

性能指标：
  latency = mean(execution_times)
  overhead = compile_time + upload_time
  efficiency = compute_time / total_time
```

### 24.11.4 调优配置参数详解

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `number` | 每次测量的运行次数 | 10 | 10-100 |
| `repeat` | 重复测量次数 | 1 | 3-5 |
| `min_repeat_ms` | 最小重复时间（ms） | 100 | 100-1000 |
| `timeout` | 单次测量超时（秒） | 10 | 10-60 |
| `cooldown_interval` | 测量间隔（秒） | 0 | 0-1 |

---

## 24.12 跨设备调试技巧

### 24.12.1 远程打印

```python
# 在远程设备上打印调试信息
conn = rpc.connect("192.168.1.100", 9090)

# 注册远程打印函数
@tvm.register_func("debug.print_remote")
def print_remote(msg):
    print(f"[Remote] {msg}")

# 在远程调用
remote_func = conn.get_function("debug.print_remote")
remote_func("Hello from remote!")
```

### 24.12.2 远程性能分析

```python
# 在远程设备上运行性能分析
remote_prof = conn.get_function("runtime.profiling.Profile")

# 上传要分析的模块
remote_lib = conn.upload(lib)

# 执行性能分析
profile_result = remote_prof(remote_lib, remote_dev)
print(profile_result)
```

### 24.12.3 远程文件系统

```python
# RPC 支持远程文件操作
conn = rpc.connect("192.168.1.100", 9090)

# 上传文件到远程
conn.upload_file("local_model.tar", "/remote/path/model.tar")

# 在远程加载
remote_lib = conn.load_module("/remote/path/model.tar")
```

### 24.12.4 远程断点调试

```python
# 远程断点调试（需要编译时启用调试符号）
conn = rpc.connect("192.168.1.100", 9090)

# 注册调试回调
@tvm.register_func("debug.breakpoint")
def debug_breakpoint(func_name, args):
    print(f"Breakpoint hit at {func_name}")
    print(f"Args: {args}")
    # 可以在这里检查变量、修改状态等

# 在远程代码中插入断点
remote_func = conn.get_function("my_kernel")
remote_func(args)  # 执行到断点时会调用回调
```

### 24.12.5 远程日志收集

```python
# 收集远程设备的日志
conn = rpc.connect("192.168.1.100", 9090)

# 设置日志级别
conn.get_function("runtime.SetLogLevel")(3)  # INFO 级别

# 执行操作
remote_gmod.run()

# 获取日志
logs = conn.get_function("runtime.GetLogs")()
print(logs)
```

---

## 24.13 高级话题

### 24.13.1 自定义 RPC 传输层

TVM 的 RPC 传输层是可扩展的，用户可以实现自定义的传输协议：

```cpp
// 实现自定义传输层（如 USB、蓝牙）
class CustomRPCChannel : public RPCChannel {
 public:
  void Send(const void* data, size_t size) override {
    // 通过自定义协议发送数据
    custom_protocol_send(data, size);
  }

  size_t Recv(void* data, size_t size) override {
    // 通过自定义协议接收数据
    return custom_protocol_recv(data, size);
  }
};

// 使用自定义传输层创建 RPC Session
auto session = RPCSession::Create(std::make_shared<CustomRPCChannel>());
```

### 24.13.2 RPC 的安全性增强

```python
# 使用 TLS 加密 RPC 连接（需要编译时启用）
from tvm.rpc import TLSContext

tls_ctx = TLSContext(
    server_cert="server.pem",
    server_key="server_key.pem",
    client_cert="client.pem",
)

server = Server(host="0.0.0.0", port=9090, key="my_device",
                tls_context=tls_ctx)
```

### 24.13.3 多线程 RPC

```python
# RPC Server 可以处理多个并发连接
import threading

def worker(conn):
    gmod = conn.graph_executor.create(...)
    gmod.run()
    output = gmod.get_output(0).download()

# 为每个连接创建一个线程
for conn in connections:
    threading.Thread(target=worker, args=(conn,)).start()
```

### 24.13.4 RPC 的超时与重试机制

```python
# RPC 连接超时配置
from tvm.rpc import connect

# 设置连接超时
conn = connect("192.168.1.100", 9090, session_timeout=30)

# 设置调用超时
remote_func = conn.get_function("my_func")
try:
    result = remote_func(args, timeout=10)
except tvm.RPCTimeoutError:
    print("Remote call timed out")
    # 重试逻辑
    result = remote_func(args, timeout=30)
```

### 24.13.5 RPC 的性能优化

```
RPC 性能优化技巧：

  1. 批量化调用：
     - 不要逐个调用远程函数
     - 将多个操作打包成一个 PackedFunc

  2. 数据传输优化：
     - 使用压缩传输大 NDArray
     - 避免频繁的小数据传输
     - 使用 zero-copy 传输（如共享内存）

  3. 连接复用：
     - 使用连接池
     - 避免频繁建立/断开连接

  4. 异步调用：
     - 使用异步 API 并行执行
     - 重叠计算和数据传输
```

### 24.13.6 RPC 的扩展：gRPC 后端

```cpp
// 实现 gRPC 传输层
class GRPCChannel : public RPCChannel {
 public:
  GRPCChannel(const std::string& server_address)
      : stub_(grpc::NewStub(grpc::CreateChannel(
            server_address, grpc::InsecureChannelCredentials()))) {}

  void Send(const void* data, size_t size) override {
    rpc::DataRequest request;
    request.set_data(data, size);
    rpc::DataResponse response;
    grpc::ClientContext context;
    stub_->SendData(&context, request, &response);
  }

  size_t Recv(void* data, size_t size) override {
    rpc::DataRequest request;
    rpc::DataResponse response;
    grpc::ClientContext context;
    stub_->RecvData(&context, request, &response);
    memcpy(data, response.data().data(), response.data().size());
    return response.data().size();
  }

 private:
  std::unique_ptr<rpc::RPCService::Stub> stub_;
};
```

---

## 24.14 PackedFunc 的设计模式

### 24.14.1 工厂模式

```cpp
// 使用 PackedFunc 实现工厂模式
TVM_REGISTER_GLOBAL("runtime.GraphExecutorFactory")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string graph_json = args[0];
  Module lib = args[1];
  Device dev = args[2];

  // 创建 GraphExecutor
  auto exec = GraphExecutor::Create(graph_json, lib, dev);
  *ret = exec;
});
```

### 24.14.2 策略模式

```cpp
// 使用 PackedFunc 实现策略模式
class Optimizer {
 public:
  void SetStrategy(PackedFunc strategy) {
    strategy_ = strategy;
  }

  void Optimize(Module mod) {
    // 调用策略函数
    strategy_(mod);
  }

 private:
  PackedFunc strategy_;
};

// 注册不同的优化策略
TVM_REGISTER_GLOBAL("optimizer.level0")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  Module mod = args[0];
  // Level 0 优化
});

TVM_REGISTER_GLOBAL("optimizer.level1")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  Module mod = args[0];
  // Level 1 优化
});
```

### 24.14.3 观察者模式

```cpp
// 使用 PackedFunc 实现观察者模式
class EventEmitter {
 public:
  void On(const std::string& event, PackedFunc callback) {
    listeners_[event].push_back(callback);
  }

  void Emit(const std::string& event, TVMArgs args) {
    for (auto& listener : listeners_[event]) {
      listener(args, nullptr);
    }
  }

 private:
  std::unordered_map<std::string, std::vector<PackedFunc>> listeners_;
};

// 使用示例
EventEmitter emitter;
emitter.On("compile_start", [](TVMArgs args, TVMRetValue*) {
  std::cout << "Compilation started" << std::endl;
});
emitter.Emit("compile_start", TVMArgs({}, {}, 0));
```

### 24.14.4 装饰器模式

```cpp
// 使用 PackedFunc 实现装饰器模式
PackedFunc WithLogging(PackedFunc func, const std::string& name) {
  return PackedFunc([func, name](TVMArgs args, TVMRetValue* ret) {
    std::cout << "[LOG] Calling " << name << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    func(args, ret);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "[LOG] " << name << " took " << duration.count() << " μs" << std::endl;
  });
}

// 使用示例
auto original_func = Registry::Get("my_func")->body();
auto logged_func = WithLogging(original_func, "my_func");
```

---

## 24.15 PackedFunc 在 TVM 编译流水线中的应用

### 24.15.1 Pass 注册

```cpp
// src/relay/transforms/fuse_ops.cc
TVM_REGISTER_GLOBAL("relay._transform.FuseOps")
.set_body_typed([](int fuse_opt_level) {
  return CreateFunctionPass(
    [=](const Function& f, const IRModule& m, PassContext ctx) {
      return FuseOps(f, fuse_opt_level);
    },
    0,
    "FuseOps",
    {"InferType"});
});
```

### 24.15.2 算子注册

```cpp
// src/relay/op/nn/convolution.cc
TVM_REGISTER_GLOBAL("relay.op.nn.conv2d")
.set_body_typed([](Expr data, Expr weight, ...) {
  return MakeConv2D(data, weight, ...);
});
```

### 24.15.3 调度原语注册

```cpp
// src/tir/schedule/schedule.cc
TVM_REGISTER_GLOBAL("tir.Schedule.Split")
.set_body_typed([](Schedule self, StmtSRef loop,
                    const Array<Expr>& factors) {
  return self->Split(loop, factors);
});
```

### 24.15.4 编译流水线中的 PackedFunc 调用链

```
编译流水线中的 PackedFunc 调用链：

  relay.build(mod, target)
    │
    ├── TVM_REGISTER_GLOBAL("relay.build")
    │     └── 调用 relay._transform.FuseOps
    │           └── TVM_REGISTER_GLOBAL("relay._transform.FuseOps")
    │
    ├── TVM_REGISTER_GLOBAL("relay._transform.ToANormalForm")
    │
    ├── TVM_REGISTER_GLOBAL("relay._transform.InferType")
    │
    ├── TVM_REGISTER_GLOBAL("relay.backend.CompileEngine")
    │     └── 调用 tir.Schedule.Apply
    │           └── TVM_REGISTER_GLOBAL("tir.Schedule.Apply")
    │
    └── 返回编译后的 Module
```

---

## 24.16 常见陷阱与最佳实践

### 24.16.1 常见陷阱

#### 陷阱 1：类型转换错误

```cpp
// ❌ 错误：未检查类型就转换
TVM_REGISTER_GLOBAL("my_func")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  int a = args[0];  // 如果 args[0] 不是 int，会崩溃
});

// ✅ 正确：先检查类型
TVM_REGISTER_GLOBAL("my_func")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  CHECK(args[0].type_code() == kDLInt) << "Expected int";
  int a = args[0];
});
```

#### 陷阱 2：生命周期管理

```cpp
// ❌ 错误：返回局部变量的引用
TVM_REGISTER_GLOBAL("my_func")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string local_str = "hello";
  *ret = local_str;  // local_str 在函数返回后被销毁
});

// ✅ 正确：使用 TVMRetValue 的值语义
TVM_REGISTER_GLOBAL("my_func")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = std::string("hello");  // TVMRetValue 会拷贝字符串
});
```

#### 陷阱 3：NDArray 设备不匹配

```python
# ❌ 错误：在 CPU 上创建 NDArray，传给 GPU 函数
arr = tvm.nd.array(np.zeros(10))  # CPU 上
gpu_func(arr)  # 可能崩溃或产生未定义行为

# ✅ 正确：确保设备匹配
arr = tvm.nd.array(np.zeros(10), tvm.cuda(0))  # GPU 上
gpu_func(arr)
```

#### 陷阱 4：RPC 连接泄漏

```python
# ❌ 错误：连接未关闭
conn = rpc.connect("192.168.1.100", 9090)
# ... 使用连接 ...
# 忘记关闭连接

# ✅ 正确：使用 with 语句
with rpc.connect("192.168.1.100", 9090) as conn:
    # ... 使用连接 ...
    pass  # 自动关闭
```

#### 陷阱 5：序列化失败

```python
# ❌ 错误：尝试序列化不支持的类型
@tvm.register_func("my_func")
def my_func():
    return lambda x: x  # lambda 无法序列化

# ✅ 正确：只返回可序列化的类型
@tvm.register_func("my_func")
def my_func():
    return 42  # 基本类型可以序列化
```

### 24.16.2 最佳实践

#### 实践 1：使用 TypedPackedFunc

```cpp
// ❌ 使用原始 PackedFunc
PackedFunc my_func = [](TVMArgs args, TVMRetValue* ret) {
  int a = args[0];
  int b = args[1];
  *ret = a + b;
};

// ✅ 使用 TypedPackedFunc（编译时类型检查）
TypedPackedFunc<int(int, int)> my_func = [](int a, int b) {
  return a + b;
};
```

#### 实践 2：批量操作

```python
# ❌ 错误：多次小数据传输
for i in range(100):
    remote_arr = conn.upload(local_arrs[i])

# ✅ 正确：批量传输
batch_arr = np.stack(local_arrs)
remote_batch = conn.upload(tvm.nd.array(batch_arr))
```

#### 实践 3：错误处理

```python
# ✅ 健壮的错误处理
try:
    result = remote_func(args)
except tvm.RPCError as e:
    print(f"RPC error: {e}")
    # 重试或回退到本地执行
except tvm.RPCTimeoutError as e:
    print(f"RPC timeout: {e}")
    # 增加超时时间或检查网络
```

#### 实践 4：性能监控

```python
# ✅ 监控 RPC 性能
import time

start = time.time()
remote_gmod.run()
end = time.time()

print(f"Remote execution time: {end - start:.3f} s")
```

---

## 24.17 练习题

### 练习 1：实现自定义 PackedFunc

**题目**：实现一个 PackedFunc，接受两个 NDArray 作为输入，返回它们的点积。

```cpp
// 提示：使用 TVM_REGISTER_GLOBAL
TVM_REGISTER_GLOBAL("exercise.dot_product")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  NDArray a = args[0];
  NDArray b = args[1];
  // TODO: 实现点积计算
  // 1. 检查 NDArray 的形状是否兼容
  // 2. 执行点积
  // 3. 将结果设置为返回值
});
```

**要求**：
1. 处理形状不匹配的错误
2. 支持 CPU 和 GPU 设备
3. 返回一个标量 NDArray

### 练习 2：实现自定义 Module

**题目**：实现一个 `CustomModule`，封装一个简单的数学函数库。

```cpp
// 提示：继承 ModuleNode
class CustomModuleNode : public ModuleNode {
 public:
  PackedFunc GetFunction(const std::string& name,
                          const ObjectPtr<Object>& sptr_to_self) override {
    // TODO: 根据 name 返回对应的 PackedFunc
    // 支持的函数：
    // - "add": 加法
    // - "multiply": 乘法
    // - "power": 幂运算
  }

  const char* type_key() const override {
    return "custom";
  }
};
```

**要求**：
1. 实现至少 3 个数学函数
2. 支持整数和浮点数参数
3. 处理未知函数名的错误

### 练习 3：实现 RPC 传输层

**题目**：实现一个基于 Unix Domain Socket 的 RPC 传输层。

```cpp
// 提示：继承 RPCChannel
class UnixSocketChannel : public RPCChannel {
 public:
  UnixSocketChannel(const std::string& socket_path) {
    // TODO: 创建 Unix Domain Socket
  }

  void Send(const void* data, size_t size) override {
    // TODO: 通过 socket 发送数据
  }

  size_t Recv(void* data, size_t size) override {
    // TODO: 通过 socket 接收数据
  }
};
```

**要求**：
1. 处理连接断开和重连
2. 实现超时机制
3. 支持大数据分片传输

### 练习 4：远程性能分析

**题目**：编写一个脚本，使用 RPC 在远程设备上运行性能分析，并生成报告。

```python
def profile_remote(host, port, lib, dev):
    """
    在远程设备上运行性能分析

    参数：
        host: 远程主机地址
        port: 端口号
        lib: 编译后的 Module
        dev: 远程设备

    返回：
        包含性能指标的字典
    """
    # TODO:
    # 1. 连接到远程设备
    # 2. 上传编译产物
    # 3. 创建 GraphExecutor
    # 4. 运行性能分析
    # 5. 收集并返回结果
    pass
```

**要求**：
1. 测量每次推理的延迟
2. 计算平均值、标准差、P50/P90/P99
3. 输出格式化的报告

### 练习 5：实现函数注册表的热更新

**题目**：实现一个支持热更新的函数注册表，可以在运行时动态替换函数实现。

```cpp
class HotReloadRegistry {
 public:
  // 注册函数
  void Register(const std::string& name, PackedFunc func);

  // 热更新函数
  void HotUpdate(const std::string& name, PackedFunc new_func);

  // 获取函数
  PackedFunc Get(const std::string& name);

 private:
  std::unordered_map<std::string, PackedFunc> funcs_;
  std::shared_mutex mutex_;
};
```

**要求**：
1. 支持线程安全的热更新
2. 保留旧版本的回滚能力
3. 记录更新历史

---

## 24.18 本章小结

本章深入分析了 TVM 的 PackedFunc 与 RPC 机制：

1. **PackedFunc**：类型擦除的函数调用接口，是 TVM 组件间交互的基础
2. **函数注册**：全局注册表，支持 Python/C++ 跨语言调用
3. **Module 系统**：代码组织单元，支持嵌套导入和序列化
4. **FFI**：跨语言函数接口，支持 Python、Cython、ctypes 等多种路径
5. **RPC Server/Client**：远程过程调用，支持跨设备编译和执行
6. **RPC Tracker**：多设备调度器，支持自动调优
7. **设计模式**：PackedFunc 支持工厂、策略、观察者等多种设计模式

关键设计原则：

```
抽象层次          设计目标
──────────────────────────────
PackedFunc       统一的跨语言函数接口
Registry         全局函数查找
Module           代码组织与序列化
FFI              跨语言类型转换
RPC              跨设备透明调用
Tracker          多设备资源管理
```

核心公式回顾：

$$
\text{FFI 绑定数} = 1 \quad \text{（统一接口）}
$$

$$
T_{\text{packed}} = T_{\text{pack}} + T_{\text{dispatch}} + T_{\text{unpack}} + T_{\text{body}}
$$

$$
\text{负载} = \frac{\text{当前连接数}}{\text{最大连接数}}
$$

下一章我们将深入 Target 系统与硬件描述。

---

## 24.99 文字内容强化：PackedFunc 与 RPC 的工程化阅读补充

PackedFunc 与 RPC 是 TVM 工程化能力的粘合层，表面上是函数调用，实质上连接了语言边界、模块边界和设备边界。

### 24.99.1 代码解读：从片段回到主流程

原有 PackedFunc 代码块要从参数打包、函数查表、调用分派和返回值解包四步阅读。
控制流在 C++ 注册表与 Python FFI 之间来回穿越，但统一协议让调用点保持简洁。
工程意义在于 TVM 可以用少量绑定暴露大量内部能力。
代码块中的变量名、函数名和类名不应孤立记忆，而应放回编译流水线中理解。
读者可以先判断代码块处在构建期、优化期、代码生成期还是运行期。
构建期代码通常负责收集信息，优化期代码负责改写 IR，代码生成期代码负责降低表示，运行期代码负责执行与资源管理。
一旦阶段判断正确，许多看似相似的数据结构就能区分出职责边界。

### 24.99.2 源码阅读路径

阅读 apache/tvm 源码时，建议按下面顺序推进，而不是直接在全仓库搜索 PackedFunc 与 RPC。
第 1 步，阅读 `include/tvm/runtime/packed_func.h`，目标是确认这一层暴露的主要接口和被谁调用。
第 2 步，阅读 `src/runtime/registry.cc`，目标是确认这一层暴露的主要接口和被谁调用。
第 3 步，阅读 `src/runtime/module.cc`，目标是确认这一层暴露的主要接口和被谁调用。
第 4 步，阅读 `src/runtime/rpc/`，目标是确认这一层暴露的主要接口和被谁调用。
第 5 步，阅读 `python/tvm/rpc/`，目标是确认这一层暴露的主要接口和被谁调用。
完成主路径后，再阅读相邻测试目录，测试通常比注释更清楚地展示了设计者希望维持的不变量。
如果遇到注册表入口，应记录注册名、C++ 实现函数、Python 包装函数和最终用户 API 四个位置。
如果遇到 Pass，应记录 Pass 的输入 IR、输出 IR、启用条件和在默认流水线中的相对顺序。
如果遇到运行时模块，应记录它的创建时机、序列化格式、加载入口和资源释放位置。

### 24.99.4 逐行阅读提示与工程理解清单

1. 类型擦除 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
2. 阅读 注册表 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
21. Module 导出 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
22. 阅读 RPC 会话 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
41. 类型擦除 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
42. 阅读 注册表 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
61. Module 导出 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
62. 阅读 RPC 会话 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
81. 类型擦除 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
82. 阅读 注册表 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
101. Module 导出 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
102. 阅读 RPC 会话 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
121. 类型擦除 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
122. 阅读 注册表 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
141. Module 导出 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
142. 阅读 RPC 会话 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
161. 类型擦除 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
162. 阅读 注册表 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
181. Module 导出 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
182. 阅读 RPC 会话 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
201. 类型擦除 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
202. 阅读 注册表 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
221. Module 导出 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
222. 阅读 RPC 会话 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
241. 类型擦除 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
242. 阅读 注册表 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
261. Module 导出 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
262. 阅读 RPC 会话 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
281. 类型擦除 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
282. 阅读 注册表 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
301. Module 导出 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
302. 阅读 RPC 会话 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
321. 类型擦除 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
322. 阅读 注册表 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
341. Module 导出 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
342. 阅读 RPC 会话 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
361. 类型擦除 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
362. 阅读 注册表 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
381. Module 导出 的第一层理解，是把它看成 跨语言与跨设备调用基础设施 中连接抽象语义和工程实现的接口。
382. 阅读 RPC 会话 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
383. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
384. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
385. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
386. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
387. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
388. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
389. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
390. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
391. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
392. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
393. 工程上，稳定的边界往往比复杂的局部优化更重要。
394. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
395. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
396. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
397. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
398. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。

### 24.99.5 小结：把本章放回 TVM 全链路

PackedFunc 与 RPC 的学习重点不是记住每个函数名，而是理解它在 TVM 全链路中承担的边界职责。
当读者能够说清楚输入从哪里来、状态在哪里保存、输出被谁消费，就已经掌握了源码阅读的主线。
后续遇到性能、兼容性或部署问题时，可以沿着这条主线逐层排查，而不是在全仓库中盲目搜索。

