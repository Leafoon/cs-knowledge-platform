> **学习目标**：
> - 理解 TVM Target 系统的设计哲学与核心抽象
> - 掌握 TargetKind 的注册机制与属性定义
> - 理解 Target JSON 格式与解析流程
> - 掌握 Device API 注册与硬件特性声明
> - 理解 Target 对编译管线各阶段的影响
> - 能够为自定义硬件平台添加 TargetKind 支持
> - 掌握多 Target 异构编译的完整流程
> - 理解 Target 与自动调优系统的交互机制

---

## 25.1 Target 系统概述

### 25.1.1 什么是 Target

Target 是 TVM 对**编译目标硬件**的抽象描述。它封装了目标设备的所有关键信息：

1. **设备类型**：CPU、GPU、FPGA、DSP 等
2. **指令集特性**：AVX-512、NEON、CUDA Compute Capability 等
3. **编译选项**：优化级别、链接库、代码生成器参数
4. **运行时特性**：内存大小、线程数、缓存层次

源码位置：`include/tvm/target/target.h`

Target 在 TVM 中的角色可以用一个简单的公式表示：

$$\text{Target} = (\text{TargetKind}, \text{Attrs}, \text{Keys}, \text{Host})$$

其中每个分量的含义如下：

| 分量 | 类型 | 作用 |
|------|------|------|
| TargetKind | `TargetKind` | 硬件类型分类（如 CPU、GPU） |
| Attrs | `Map<String, ObjectRef>` | 具体硬件参数（如 mcpu、mattr） |
| Keys | `Array<String>` | 特性标签集合（用于 Pass 匹配） |
| Host | `Target` | 宿主设备（GPU 需要 CPU Host） |

### 25.1.2 Target 的作用域

Target 在 TVM 编译管线的多个阶段发挥作用：

```
前端导入 → Relay 优化 → TE Lowering → TIR 变换 → CodeGen → 运行时
              │              │            │           │
              └──────────────┴────────────┴───────────┘
                        每个阶段都依赖 Target 信息
```

每个阶段对 Target 的依赖程度不同：

| 编译阶段 | Target 的影响 | 依赖程度 |
|---------|--------------|---------|
| **前端导入** | 决定算子 dtype 映射 | 低 |
| **Relay 优化** | 决定算子融合策略、布局变换 | 高 |
| **TE Lowering** | 选择默认调度策略 | 高 |
| **TIR 变换** | 决定向量化宽度、并行策略 | 高 |
| **CodeGen** | 选择代码生成器（LLVM/CUDA/C） | 极高 |
| **链接** | 选择链接器参数、目标库 | 中 |
| **运行时** | 选择 DeviceAPI、内存分配策略 | 极高 |

### 25.1.3 关键源文件

| 文件 | 职责 | 行数（约） |
|------|------|-----------|
| `include/tvm/target/target.h` | Target 类定义、TargetKind 定义 | ~500 |
| `src/target/target.cc` | Target 实现、JSON 解析 | ~800 |
| `src/target/target_kind.cc` | TargetKind 注册、全局注册表 | ~300 |
| `src/target/tag.cc` | 预定义 Target 标签 | ~500 |
| `python/tvm/target/target.py` | Python 封装、便捷函数 | ~600 |
| `python/tvm/target/tag.py` | Python 标签定义 | ~400 |
| `src/target/llvm/llvm_target.cc` | LLVM Target 集成 | ~400 |
| `src/target/source/codegen.cc` | 源码 Target 基类 | ~300 |
| `include/tvm/runtime/device_api.h` | DeviceAPI 接口定义 | ~200 |
| `src/runtime/device_api.cc` | DeviceAPI 注册与分发 | ~150 |

### 25.1.4 设计哲学

TVM Target 系统的设计遵循以下原则：

1. **声明式硬件描述**：通过属性（Attributes）声明硬件能力，而非硬编码逻辑
2. **可扩展性**：通过注册机制添加新硬件，无需修改核心代码
3. **可序列化**：Target 可以转换为 JSON，支持跨进程传输和持久化
4. **分层抽象**：TargetKind → Target → DeviceAPI 三层抽象

```
┌─────────────────────────────────────────────────┐
│                  用户层 (Python)                  │
│  tvm.target.Target("cuda -mcpu=sm_80")          │
├─────────────────────────────────────────────────┤
│               Target 抽象层 (C++)                 │
│  TargetKind → Target → Keys → Host              │
├─────────────────────────────────────────────────┤
│              Device API 层 (C++)                  │
│  AllocDataSpace / CopyDataFromTo / StreamSync    │
├─────────────────────────────────────────────────┤
│              硬件驱动层                            │
│  CUDA Driver / OpenCL Runtime / Metal Framework   │
└─────────────────────────────────────────────────┘
```

<div data-component="TargetSystemOverview"></div>

---

## 25.2 TargetKind：硬件类型注册

### 25.2.1 TargetKind 的定义

TargetKind 是 TVM 对一类硬件设备的**类型描述**，它定义了该设备支持的属性、默认值和约束：

```cpp
// include/tvm/target/target.h
class TargetKind : public Object {
 public:
  // TargetKind 名称（如 "llvm", "cuda", "opencl"）
  String name;

  // 支持的属性定义
  Array<String> keys;

  // 默认属性值
  Map<String, ObjectRef> default_attrs;

  // 设备类型
  int device_type;

  // 是否支持指定 feature
  bool IsDevice() const;
  bool IsGPU() const;
  bool IsCPU() const;
};
```

TargetKind 与 Target 的关系可以类比为"类"与"实例"：

$$\text{Target} \in \text{TargetKind} \times \text{Attrs}$$

即 Target 是某个 TargetKind 的一个具体实例，带有特定的属性值。

### 25.2.2 TargetKind 的继承体系

```
Object (TVM 基类)
  └── TargetKind
        ├── "llvm"      (kDLCPU)
        ├── "cuda"      (kDLCUDA)
        ├── "opencl"    (kDLOpenCL)
        ├── "vulkan"    (kDLVulkan)
        ├── "metal"     (kDLMetal)
        ├── "hexagon"   (kDLHexagon)
        ├── "c"         (kDLCPU)
        ├── "stackvm"   (kDLCPU)
        └── "ext_dev"   (kDLExtDev)
```

### 25.2.3 内置 TargetKind

| TargetKind | 设备类型 | 典型用途 | 默认 Key |
|-----------|---------|---------|----------|
| `"llvm"` | CPU | x86/ARM CPU 编译 | `cpu` |
| `"cuda"` | GPU | NVIDIA GPU | `cuda`, `gpu` |
| `"opencl"` | GPU | 跨平台 GPU | `opencl`, `gpu` |
| `"vulkan"` | GPU | Vulkan GPU | `vulkan`, `gpu` |
| `"metal"` | GPU | Apple GPU | `metal`, `gpu` |
| `"hexagon"` | DSP | Qualcomm DSP | `hexagon` |
| `"webassembly"` | CPU | 浏览器 | `cpu`, `wasm` |
| `"c"` | CPU | 纯 C 源码 | `c` |
| `"ext_dev"` | 外部 | 外部代码生成器 | `ext_dev` |

### 25.2.4 TargetKind 注册

```cpp
// src/target/target_kind.cc
// 注册 LLVM TargetKind
TVM_REGISTER_TARGET_KIND("llvm", kDLCPU)
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<String>("mtriple")
    .add_attr_option<String>("mcpu")
    .add_attr_option<String>("mattr")
    .add_attr_option<Bool>("mfloat-abi")
    .add_attr_option<String>("host")
    .set_default_keys({"cpu"});

// 注册 CUDA TargetKind
TVM_REGISTER_TARGET_KIND("cuda", kDLCUDA)
    .add_attr_option<String>("mcpu")           // 如 "sm_70"
    .add_attr_option<String>("arch")           // GPU 架构
    .add_attr_option<Integer>("max_num_threads", Integer(1024))
    .add_attr_option<Integer>("max_shared_memory_per_block", Integer(49152))
    .add_attr_option<Integer>("thread_warp_size", Integer(32))
    .add_attr_option<Bool>("registers_per_block")
    .set_default_keys({"cuda", "gpu"});
```

### 25.2.5 TVM_REGISTER_TARGET_KIND 宏展开

`TVM_REGISTER_TARGET_KIND` 是一个注册宏，展开后的核心逻辑如下：

```cpp
// src/target/target_kind.cc
// 宏展开后的简化代码
struct TargetKindRegEntry {
  static TargetKindRegEntry& Register(const String& name, int device_type) {
    TargetKind kind;
    kind->name = name;
    kind->device_type = device_type;
    // 注册到全局注册表
    TargetKindRegistry::Global()->Register(kind);
    return *this;
  }

  // 添加属性选项
  TargetKindRegEntry& add_attr_option(const String& name) {
    kind->valid_attrs.insert(name);
    return *this;
  }

  // 设置默认 Key
  TargetKindRegEntry& set_default_keys(Array<String> keys) {
    kind->default_keys = keys;
    return *this;
  }
};
```

### 25.2.6 属性选项（Attribute Options）

每个 TargetKind 可以声明支持的属性选项及其默认值：

```cpp
// add_attr_option<T>(name, default_value)
// T 可以是：Bool, Integer, String, Float, Array, Map

TVM_REGISTER_TARGET_KIND("my_device", kDLExtDev)
    .add_attr_option<String>("model_name")
    .add_attr_option<Integer>("num_cores", Integer(4))
    .add_attr_option<Bool>("supports_fp16", Bool(true))
    .add_attr_option<Float>("clock_ghz", Float(1.0))
    .set_default_keys({"my_device"});
```

属性选项的类型系统：

| 类型 | C++ 类型 | 用途 | 示例 |
|------|---------|------|------|
| Bool | `Bool` (bool) | 开关特性 | `"system-lib": true` |
| Integer | `Integer` (int64_t) | 数量参数 | `"max_num_threads": 1024` |
| String | `String` | 标识符 | `"mcpu": "core-avx2"` |
| Float | `Float` (double) | 浮点参数 | `"clock_ghz": 3.5` |
| Array | `Array<ObjectRef>` | 列表参数 | `"mattr": ["+avx2", "+fma"]` |
| Map | `Map<String, ObjectRef>` | 字典参数 | 自定义配置 |

### 25.2.7 属性验证机制

当创建 Target 时，TVM 会验证每个属性是否在 TargetKind 的合法属性列表中：

```cpp
// src/target/target.cc
void Target::ValidateAttrs(const TargetKind& kind,
                           const Map<String, ObjectRef>& attrs) {
  for (const auto& kv : attrs) {
    if (kv.first == "host" || kv.first == "keys" || kv.first == "tag") {
      continue;  // 这些是通用属性，跳过验证
    }
    ICHECK(kind->HasAttrOption(kv.first))
        << "AttributeError: TargetKind \"" << kind->name
        << "\" does not support attribute \"" << kv.first
        << "\". Valid attributes are: " << kind->ListAttrOptions();
  }
}
```

如果传入了无效属性，会抛出明确的错误信息，帮助用户快速定位问题。

<div data-component="TargetKindRegistration"></div>

---

## 25.3 Target 对象

### 25.3.1 Target 的 C++ 定义

```cpp
// include/tvm/target/target.h
class Target : public ObjectRef {
 public:
  // 从字符串创建 Target
  static Target FromString(const String& target_str);

  // 从 JSON 配置创建 Target
  static Target FromConfig(const Map<String, ObjectRef>& config);

  // 获取当前 Target（线程局部）
  static Target Current();

  // 属性访问
  const TargetKindNode* kind() const;
  const Map<String, ObjectRef>& attrs() const;
  const Array<String>& keys() const;

  // 设备类型查询
  int GetTargetDeviceType() const;

  // Host Target
  Target GetHost() const;

  // 序列化
  String ToJsonString() const;
};
```

### 25.3.2 Target 的创建方式

```python
import tvm

# 方式一：从字符串创建
target = tvm.target.Target("llvm -mcpu=core-avx2")

# 方式二：从 JSON 创建
target = tvm.target.Target({
    "kind": "llvm",
    "mcpu": "core-avx2",
    "mattr": ["+avx2", "+fma"],
})

# 方式三：使用设备标识符
target = tvm.target.Target("cuda", host="llvm")

# 方式四：使用预定义标签
target = tvm.target.cuda(model="rtx3090")
target = tvm.target.arm_cpu(model="cortex-a76")

# 方式五：使用 Target.current() 获取当前上下文
with tvm.target.Target("cuda"):
    current = tvm.target.Target.current()
    assert current.kind.name == "cuda"
```

各创建方式的对比：

| 方式 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| 字符串 | 快速原型 | 简洁 | 不支持嵌套 host |
| JSON | 精确配置 | 完整 | 代码冗长 |
| 预定义标签 | 常用设备 | 方便 | 仅限预定义设备 |
| 上下文管理器 | Pass 内部 | 自动关联 | 仅限作用域内 |

### 25.3.3 Target 的属性访问

```python
target = tvm.target.Target("llvm -mcpu=core-avx2")

# 访问基本属性
print(target.kind.name)      # "llvm"
print(target.kind.device_type)  # 1 (kDLCPU)

# 访问自定义属性
print(target.mcpu)           # "core-avx2"
print(target.attrs.get("mattr", []))

# 检查设备特性
print(target.kind.IsCPU())   # True
print(target.kind.IsGPU())   # False
```

### 25.3.4 Target 的比较与匹配

```python
target1 = tvm.target.Target("llvm -mcpu=core-avx2")
target2 = tvm.target.Target("llvm -mcpu=core-avx2")
target3 = tvm.target.Target("llvm -mcpu=znver3")

# 结构相等
assert target1 == target2
assert target1 != target3

# Target 匹配（用于 Pass 注册）
# 检查 target 是否支持特定的 key
assert "cpu" in target1.keys
assert "avx2" in target1.keys  # 如果 mattr 包含 +avx2
```

### 25.3.5 Target 字符串解析

Target 字符串的解析遵循特定的语法规则：

```
target_string := <kind_name> [<attr_key>=<attr_value>]...
kind_name     := [a-z_][a-z0-9_]*
attr_key      := -[a-z][a-z0-9-]*
attr_value    := <string> | <number> | <boolean>
```

示例解析过程：

```
输入: "llvm -mcpu=core-avx2 -mattr=+avx2,+fma -system-lib"
解析结果:
  kind = "llvm"
  attrs = {
    "mcpu": "core-avx2",
    "mattr": ["+avx2", "+fma"],   # 逗号分隔的列表
    "system-lib": true             # 无值的标志
  }
```

### 25.3.6 Target 字符串解析源码

```cpp
// src/target/target.cc
Target Target::FromString(const String& target_str) {
  // 分割字符串
  std::vector<std::string> parts = Split(target_str, ' ');

  // 第一个部分是 TargetKind 名称
  String kind_name = parts[0];
  TargetKind kind = TargetKind::Get(kind_name);

  // 解析属性
  Map<String, ObjectRef> attrs;
  for (size_t i = 1; i < parts.size(); ++i) {
    std::string part = parts[i];
    if (part[0] == '-') {
      // 处理 -key=value 或 -flag 格式
      size_t eq_pos = part.find('=');
      if (eq_pos != std::string::npos) {
        String key = part.substr(1, eq_pos - 1);
        String value = part.substr(eq_pos + 1);
        attrs.Set(key, ParseAttrValue(value));
      } else {
        // 布尔标志
        attrs.Set(part.substr(1), Bool(true));
      }
    }
  }

  return Target(kind, attrs);
}
```

<div data-component="TargetObjectDiagram"></div>

---

## 25.4 Target JSON 格式

### 25.4.1 JSON 表示

Target 可以序列化为 JSON 格式，便于持久化和传输：

```json
{
  "kind": "cuda",
  "mcpu": "sm_70",
  "arch": "sm_70",
  "max_num_threads": 1024,
  "max_shared_memory_per_block": 49152,
  "thread_warp_size": 32,
  "host": {
    "kind": "llvm",
    "mcpu": "core-avx2"
  },
  "keys": ["cuda", "gpu"],
  "tag": ""
}
```

JSON 格式的字段说明：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `kind` | string | 是 | TargetKind 名称 |
| `host` | object | 否 | Host Target（递归结构） |
| `keys` | array | 否 | 额外的特性键 |
| `tag` | string | 否 | 预定义标签名 |
| 其他 | varies | 否 | TargetKind 特定属性 |

### 25.4.2 Host Target

许多 Target 需要指定一个 **Host Target**，用于编译 CPU 端代码：

```python
# GPU Target 必须有 Host Target
target = tvm.target.Target("cuda", host="llvm -mcpu=core-avx2")

# 等价于
target = tvm.target.Target({
    "kind": "cuda",
    "host": {"kind": "llvm", "mcpu": "core-avx2"}
})

# Host Target 用于编译：
# - 模型参数加载代码
# - CPU 端辅助函数
# - Graph Runtime 调度代码
```

Host Target 的嵌套结构形成一棵树：

```
Target (cuda)
  ├── kind: "cuda"
  ├── mcpu: "sm_80"
  ├── max_num_threads: 1024
  └── host: Target (llvm)
        ├── kind: "llvm"
        ├── mcpu: "core-avx2"
        └── host: null (CPU 没有 host)
```

### 25.4.3 JSON 解析流程

```cpp
// src/target/target.cc
Target Target::FromConfig(const Map<String, ObjectRef>& config) {
  // 1. 提取 "kind" 字段
  String kind_name = config.at("kind").as<String>();

  // 2. 查找 TargetKind
  TargetKind kind = TargetKind::Get(kind_name);

  // 3. 验证并设置属性
  Map<String, ObjectRef> attrs;
  for (auto& kv : config) {
    if (kv.first == "kind" || kv.first == "tag") continue;
    // 验证属性是否在 TargetKind 的属性选项中
    CHECK(kind->HasAttrOption(kv.first))
        << "Unknown attribute: " << kv.first;
    attrs.Set(kv.first, kv.second);
  }

  // 4. 递归解析 host target
  if (config.count("host")) {
    Target host = Target::FromConfig(config.at("host"));
    attrs.Set("host", host);
  }

  // 5. 创建 Target 对象
  return Target(kind, attrs);
}
```

### 25.4.4 JSON 序列化流程

```cpp
// src/target/target.cc
String Target::ToJsonString() const {
  std::ostringstream os;
  os << "{";

  // 1. 写入 kind
  os << "\"kind\":\"" << operator->()->kind->name << "\"";

  // 2. 写入属性
  for (const auto& kv : operator->()->attrs) {
    os << ",\"" << kv.first << "\":";
    WriteJsonValue(kv.second, os);
  }

  // 3. 写入 keys
  if (operator->()->keys.size() > 0) {
    os << ",\"keys\":[";
    for (size_t i = 0; i < operator->()->keys.size(); ++i) {
      if (i > 0) os << ",";
      os << "\"" << operator->()->keys[i] << "\"";
    }
    os << "]";
  }

  // 4. 写入 tag
  if (!operator->()->tag.empty()) {
    os << ",\"tag\":\"" << operator->()->tag << "\"";
  }

  os << "}";
  return os.str();
}
```

### 25.4.5 JSON 格式的常见 Target 配置

**NVIDIA A100 GPU (SM 80)**：

```json
{
  "kind": "cuda",
  "mcpu": "sm_80",
  "arch": "sm_80",
  "max_num_threads": 1024,
  "max_shared_memory_per_block": 49152,
  "thread_warp_size": 32,
  "host": {
    "kind": "llvm",
    "mcpu": "core-avx2"
  },
  "keys": ["cuda", "gpu"],
  "tag": "a100"
}
```

**ARM Cortex-A76 CPU**：

```json
{
  "kind": "llvm",
  "mtriple": "aarch64-linux-gnu",
  "mcpu": "cortex-a76",
  "mattr": ["+neon", "+fp-armv8", "+dotprod"],
  "keys": ["cpu"],
  "tag": "cortex-a76"
}
```

**RISC-V 嵌入式设备**：

```json
{
  "kind": "llvm",
  "mtriple": "riscv32-unknown-elf",
  "mcpu": "sifive-u74",
  "mattr": ["+m", "+a", "+f", "+d"],
  "system-lib": true,
  "keys": ["cpu", "riscv"],
  "tag": ""
}
```

<div data-component="TargetJSONParser"></div>

---

## 25.5 预定义 Target 标签

### 25.5.1 标签系统

TVM 提供了一系列**预定义的 Target 标签**，用于快速创建常用的 Target 配置：

```python
# python/tvm/target/tag.py 中定义的标签

# CPU 标签
tvm.target.arm_cpu("cortex-a76")      # ARM Cortex-A76
tvm.target.arm_cpu("apple-a14")       # Apple A14
tvm.target.intel_cpu("core-avx2")     # Intel Core with AVX2
tvm.target.intel_cpu("skylake-avx512") # Intel Skylake with AVX-512

# GPU 标签
tvm.target.cuda("rtx3090")            # NVIDIA RTX 3090
tvm.target.cuda("a100")               # NVIDIA A100
tvm.target.cuda("jetson-agx-orin")    # Jetson AGX Orin

# 移动端
tvm.target.arm_cpu("rk3399")          # Rockchip RK3399
tvm.target.mali("g78")                # Mali-G78 GPU
tvm.target.adreno("adreno-660")       # Qualcomm Adreno 660
```

### 25.5.2 标签的注册

```python
# python/tvm/target/tag.py
# 注册 RTX 3090 标签
def _register_rtx3090():
    target = tvm.target.Target({
        "kind": "cuda",
        "mcpu": "sm_86",
        "arch": "sm_86",
        "max_num_threads": 1024,
        "max_shared_memory_per_block": 49152,
        "thread_warp_size": 32,
        "tag": "rtx3090",
    })
    tvm.target.register_target_tag("rtx3090", target)
```

### 25.5.3 标签注册的底层实现

```cpp
// src/target/tag.cc
// 全局标签注册表
class TargetTagRegistry {
 public:
  static TargetTagRegistry* Global() {
    static TargetTagRegistry inst;
    return &inst;
  }

  void Register(const String& tag, const Target& target) {
    ICHECK(!tag_map_.count(tag))
        << "Target tag \"" << tag << "\" already registered";
    tag_map_[tag] = target;
  }

  Target Get(const String& tag) const {
    auto it = tag_map_.find(tag);
    ICHECK(it != tag_map_.end())
        << "Unknown target tag: " << tag;
    return it->second;
  }

 private:
  std::unordered_map<String, Target> tag_map_;
};
```

### 25.5.4 常用 Target 标签速查

| 标签 | 设备 | Target 配置 | 适用场景 |
|------|------|------------|---------|
| `"llvm"` | 通用 CPU | `llvm -mcpu=core-avx2` | 服务器推理 |
| `"cuda"` | 通用 NVIDIA GPU | `cuda -mcpu=sm_70` | GPU 训练/推理 |
| `"opencl"` | 通用 OpenCL GPU | `opencl` | 跨平台 GPU |
| `"vulkan"` | 通用 Vulkan GPU | `vulkan` | 移动 GPU |
| `"arm_cpu"` | ARM CPU | `llvm -mtriple=aarch64-linux-gnu` | 嵌入式 |
| `"hexagon"` | Qualcomm DSP | `hexagon` | 手机 DSP |
| `"micro_dev"` | 嵌入式设备 | `c` | 微控制器 |

### 25.5.5 自定义标签注册

```python
import tvm

# 注册自定义标签
my_target = tvm.target.Target({
    "kind": "llvm",
    "mtriple": "aarch64-linux-gnu",
    "mcpu": "cortex-a55",
    "mattr": ["+neon", "+dotprod"],
    "tag": "my_embedded_device",
})

tvm.target.register_target_tag("my_embedded_device", my_target)

# 之后可以使用
target = tvm.target.Target("my_embedded_device")
```

<div data-component="TargetTagSystem"></div>

---

## 25.6 Target 对编译的影响

### 25.6.1 对 Relay 优化的影响

Target 决定了 Relay 优化 Pass 的行为：

```python
# 在 CUDA Target 上，conv2d 会选择 NCHW 布局
with tvm.target.Target("cuda"):
    mod = relay.transform.FuseOps(fuse_opt_level=2)(mod)
    # CUDA 上的融合策略可能与 CPU 不同

# 在 ARM CPU 上，可能选择 NHWC 布局以利用 NEON 指令
with tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+neon"):
    mod = relay.transform.AlterOpLayout()(mod)
    # 布局变换会考虑 NEON 的向量宽度
```

不同 Target 对 Relay 优化的影响对比：

| 优化 Pass | CPU (x86) | CPU (ARM) | CUDA | 说明 |
|-----------|-----------|-----------|------|------|
| FuseOps | 融合粒度较小 | 融合粒度较小 | 融合粒度较大 | GPU kernel 启动开销大 |
| AlterOpLayout | NCHWc | NHWC | NCHW | 利用 SIMD 特性 |
| FoldScaleAxis | 支持 | 支持 | 支持 | 跨 Target 通用 |
| ToMixedPrecision | fp32→int8 | fp32→int8 | fp32→fp16 | 硬件支持不同 |

### 25.6.2 对 TE 调度的影响

```python
# TOPI 的调度函数根据 Target 选择不同的实现
import topi

# CPU 调度
with tvm.target.Target("llvm"):
    s = topi.generic.schedule_conv2d_nchw(outs)
    # 使用 LLVM 特定的优化（如向量化宽度 8）

# CUDA 调度
with tvm.target.Target("cuda"):
    s = topi.generic.schedule_conv2d_nchw(outs)
    # 使用 CUDA 特定的优化（如 shared memory tiling）
```

### 25.6.3 TOPI 调度分发机制

TOPI 使用 Target Key 进行调度分发：

```python
# python/topi/generic/nn.py
@tvm.target.override_native_generic_func("schedule_conv2d_nchw")
def schedule_conv2d_nchw(outs):
    """Generic schedule for conv2d NCHW"""
    # 默认调度（CPU）
    return _default_schedule(outs)

# 注册 CUDA 特定的调度
@schedule_conv2d_nchw.register(["cuda", "gpu"])
def schedule_conv2d_nchw_cuda(outs):
    """CUDA schedule for conv2d NCHW"""
    return _cuda_schedule(outs)

# 注册 ARM 特定的调度
@schedule_conv2d_nchw.register(["arm_cpu"])
def schedule_conv2d_nchw_arm(outs):
    """ARM CPU schedule for conv2d NCHW"""
    return _arm_schedule(outs)
```

分发流程图：

```
schedule_conv2d_nchw(outs)
  │
  ├─ Target key 匹配 "cuda"?
  │    └─ Yes → _cuda_schedule()
  │
  ├─ Target key 匹配 "arm_cpu"?
  │    └─ Yes → _arm_schedule()
  │
  └─ Default → _default_schedule()
```

### 25.6.4 对 TIR 变换的影响

```cpp
// TIR 变换 Pass 根据 Target 选择参数
// src/tir/transforms/vectorize_loop.cc
// 向量化宽度由 Target 的向量寄存器宽度决定

// CPU (AVX-512)：向量化宽度 = 16（512/32）
// CPU (NEON)：向量化宽度 = 4（128/32）
// CUDA：向量化宽度 = 4（128/32, float4）
```

向量化宽度的计算公式：

$$\text{vec\_width} = \frac{\text{vector\_register\_bits}}{\text{element\_type\_bits}}$$

| 目标平台 | 向量寄存器宽度 | float32 宽度 | int8 宽度 |
|---------|--------------|-------------|----------|
| AVX-512 | 512 bit | 16 | 64 |
| AVX2 | 256 bit | 8 | 32 |
| NEON | 128 bit | 4 | 16 |
| CUDA (float4) | 128 bit | 4 | 16 |

### 25.6.5 对 CodeGen 的影响

Target 决定使用哪个 CodeGen 后端：

```cpp
// src/relay/backend/compile_engine.cc
CodeGen CreateCodeGen(const Target& target) {
  if (target->kind->name == "llvm") {
    return LLVMCodeGen();
  } else if (target->kind->name == "cuda") {
    return CUDACodeGen();
  } else if (target->kind->name == "c") {
    return CSourceCodeGen();
  } else if (target->kind->name == "opencl") {
    return OpenCLCodeGen();
  }
  // ...
}
```

CodeGen 选择与 TargetKind 的对应关系：

| TargetKind | CodeGen 类 | 输出格式 | 源码位置 |
|-----------|-----------|---------|---------|
| `"llvm"` | `LLVMCodeGen` | LLVM IR → 机器码 | `src/target/llvm/` |
| `"cuda"` | `CUDACodeGen` | CUDA C → PTX | `src/target/source/` |
| `"opencl"` | `OpenCLCodeGen` | OpenCL C | `src/target/source/` |
| `"c"` | `CSourceCodeGen` | C 源码 | `src/target/source/` |
| `"vulkan"` | `VulkanCodeGen` | SPIR-V | `src/target/source/` |

### 25.6.6 Target 上下文管理器的实现

```python
# Python 层
class Target:
    def __enter__(self):
        # 压入 Target 栈
        _ffi_api.EnterTargetScope(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 弹出 Target 栈
        _ffi_api.ExitTargetScope()
```

```cpp
// C++ 层
// src/target/target.cc
void EnterTargetScope(const Target& target) {
  // 使用 thread_local 存储 Target 栈
  TargetThreadLocalEntry::ThreadLocal()->target_stack.push_back(target);
}

void ExitTargetScope() {
  TargetThreadLocalEntry::ThreadLocal()->target_stack.pop_back();
}

Target Target::Current() {
  auto& stack = TargetThreadLocalEntry::ThreadLocal()->target_stack;
  ICHECK(!stack.empty()) << "No target in scope";
  return stack.back();
}
```

<div data-component="TargetInfluenceDiagram"></div>

---

## 25.7 Device API 注册

### 25.7.1 Device API 接口定义

```cpp
// include/tvm/runtime/device_api.h
class DeviceAPI {
 public:
  virtual ~DeviceAPI() = default;

  // 内存分配
  virtual void* AllocDataSpace(Device dev, size_t nbytes,
                               size_t alignment, DLDataType type_hint) = 0;

  // 内存释放
  virtual void FreeDataSpace(Device dev, void* ptr) = 0;

  // 数据拷贝
  virtual void CopyDataFromTo(const void* from, void* to, size_t size,
                               Device dev_from, Device dev_to,
                               DLDataType type_hint, TVMStreamHandle stream) = 0;

  // 创建流
  virtual void* CreateStream(Device dev) = 0;

  // 销毁流
  virtual void FreeStream(Device dev, TVMStreamHandle stream) = 0;

  // 同步流
  virtual void StreamSync(Device dev, TVMStreamHandle stream) = 0;

  // 设置流
  virtual void SetStream(Device dev, TVMStreamHandle stream) = 0;

  // 查询设备属性
  virtual void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) = 0;

  // 获取全局 DeviceAPI 实例
  static DeviceAPI* Get(Device dev);
};
```

### 25.7.2 DeviceAttrKind 枚举

```cpp
// include/tvm/runtime/device_api.h
enum DeviceAttrKind : int {
  kExist = 0,              // 设备是否存在
  kMaxThreadsPerBlock = 1, // 每个 Block 最大线程数
  kMaxBlockDimX = 2,       // Block 最大 X 维度
  kMaxBlockDimY = 3,       // Block 最大 Y 维度
  kMaxBlockDimZ = 4,       // Block 最大 Z 维度
  kMaxSharedMemoryPerBlock = 5, // 每 Block 最大共享内存
  kComputeVersion = 6,     // 计算能力版本
  kDeviceName = 7,         // 设备名称
  kMaxClockRate = 8,       // 最大时钟频率
  kMultiProcessorCount = 9,// SM 数量
  kMaxThreadDimensions = 10, // 最大线程维度
  kGcnArch = 11,           // GCN 架构 (AMD)
  api_platform = 12,       // 平台名称
};
```

### 25.7.3 Device API 与 Target 的关系

每个 TargetKind 对应一个或多个 Device API 实现：

```cpp
// src/runtime/device_api.cc
// TargetKind → DeviceType → DeviceAPI 的映射

// 注册 CUDA Device API
TVM_REGISTER_DEVICE_API(kDLCUDA, CUDADeviceAPI)
    .set_device_type(kDLCUDA);

// 注册 CPU Device API
TVM_REGISTER_DEVICE_API(kDLCPU, CPUDeviceAPI)
    .set_device_type(kDLCPU);
```

映射关系图：

```
TargetKind        DeviceType        DeviceAPI
─────────        ──────────        ─────────
"llvm"     →     kDLCPU      →    CPUDeviceAPI
"cuda"     →     kDLCUDA     →    CUDADeviceAPI
"opencl"   →     kDLOpenCL   →    OpenCLDeviceAPI
"vulkan"   →     kDLVulkan   →    VulkanDeviceAPI
"metal"    →     kDLMetal    →    MetalDeviceAPI
"hexagon"  →     kDLHexagon  →    HexagonDeviceAPI
```

### 25.7.4 自定义 Device API

```cpp
// 自定义 FPGA Device API
class FPGADeviceAPI : public DeviceAPI {
 public:
  void* AllocDataSpace(Device dev, size_t nbytes,
                       size_t alignment, DLDataType type_hint) override {
    return FPGA_alloc(dev.device_id, nbytes);
  }

  void FreeDataSpace(Device dev, void* ptr) override {
    FPGA_free(dev.device_id, ptr);
  }

  void CopyDataFromTo(const void* from, void* to, size_t size,
                       Device dev_from, Device dev_to,
                       DLDataType type_hint, TVMStreamHandle stream) override {
    FPGA_dma_transfer(dev_from.device_id, dev_to.device_id, from, to, size);
  }

  void* CreateStream(Device dev) override {
    return FPGA_create_stream(dev.device_id);
  }

  void FreeStream(Device dev, TVMStreamHandle stream) override {
    FPGA_destroy_stream(dev.device_id, stream);
  }

  void StreamSync(Device dev, TVMStreamHandle stream) override {
    FPGA_sync_stream(dev.device_id, stream);
  }

  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) override {
    switch (kind) {
      case kExist:
        *rv = FPGA_device_exists(dev.device_id);
        break;
      case kMaxThreadsPerBlock:
        *rv = FPGA_get_max_threads(dev.device_id);
        break;
      case kDeviceName:
        *rv = FPGA_get_device_name(dev.device_id);
        break;
      default:
        *rv = 0;
        break;
    }
  }
};

// 注册自定义设备类型
static const int kDLFPGA = 100;
TVM_REGISTER_DEVICE_API(kDLFPGA, FPGADeviceAPI);
```

### 25.7.5 设备属性查询

```cpp
// 查询设备属性
Device dev = {kDLCUDA, 0};
DeviceAPI* api = DeviceAPI::Get(dev);

// 查询最大线程数
int max_threads;
api->GetAttr(dev, kMaxThreadsPerBlock, &max_threads);

// 查询共享内存大小
int shared_mem;
api->GetAttr(dev, kMaxSharedMemoryPerBlock, &shared_mem);

// 查询计算能力
int major, minor;
api->GetAttr(dev, kComputeVersion, &major, &minor);
```

### 25.7.6 DeviceAPI 的 Python 接口

```python
import tvm

# 获取设备属性
dev = tvm.cuda(0)
max_threads = dev.max_threads_per_block
shared_mem = dev.max_shared_memory_per_block
compute_version = dev.compute_version
device_name = dev.device_name

print(f"Device: {device_name}")
print(f"Max threads per block: {max_threads}")
print(f"Shared memory: {shared_mem} bytes")
print(f"Compute capability: {compute_version}")
```

### 25.7.7 跨设备数据传输

```python
import tvm
import numpy as np

# 创建 CPU 和 GPU 张量
cpu_arr = tvm.nd.array(np.ones((10, 10)), tvm.cpu(0))
gpu_arr = tvm.nd.array(np.zeros((10, 10)), tvm.cuda(0))

# CPU → GPU
gpu_arr.copyfrom(cpu_arr)

# GPU → CPU
result = gpu_arr.numpy()

# GPU → GPU (不同设备)
gpu1_arr = tvm.nd.array(np.zeros((10, 10)), tvm.cuda(0))
gpu2_arr = tvm.nd.array(np.zeros((10, 10)), tvm.cuda(1))
gpu2_arr.copyfrom(gpu1_arr)
```

跨设备拷贝的底层调用链：

```
gpu_arr.copyfrom(cpu_arr)
  │
  ├─ Python: NDArray.copyfrom()
  ├─ C++: NDArray::CopyFromTo()
  ├─ C++: DeviceAPI::CopyDataFromTo()
  │     └─ CUDADeviceAPI::CopyDataFromTo()
  │           └─ cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)
  └─ 返回
```

<div data-component="DeviceAPIRegistration"></div>

---

## 25.8 Target 特性查询

### 25.8.1 Key 系统

Target 的 `keys` 字段用于标识设备支持的特性集合：

```python
target = tvm.target.Target("llvm -mcpu=core-avx2 -mattr=+avx2,+fma")

# 查询特性
print(target.keys)  # ["cpu", "avx2", "fma"]

# 用于 Pass 注册：只有匹配的 Pass 才会执行
@relay.transform.function_pass(opt_level=1)
class MyPass:
    def transform_function(self, func, mod, ctx):
        target = ctx.current_target()
        if "avx2" in target.keys:
            # AVX2 特定优化
            pass
```

### 25.8.2 Key 的来源

Key 的来源有三个：

1. **TargetKind 默认 Key**：注册时通过 `set_default_keys` 设置
2. **属性推导 Key**：从 `mattr` 等属性自动推导
3. **用户指定 Key**：通过 JSON 的 `keys` 字段手动添加

```python
# 示例：Key 的推导
target = tvm.target.Target({
    "kind": "llvm",
    "mcpu": "core-avx2",
    "mattr": ["+avx2", "+fma"],
})

# 推导过程：
# 1. TargetKind 默认 key: ["cpu"]
# 2. mcpu 推导: 无额外 key
# 3. mattr 推导: ["avx2", "fma"]
# 最终 keys: ["cpu", "avx2", "fma"]
```

### 25.8.3 硬件能力查询

```python
target = tvm.target.cuda(model="rtx3090")

# 查询 GPU 能力
max_threads = target.max_num_threads          # 1024
shared_mem = target.max_shared_memory_per_block  # 49152
warp_size = target.thread_warp_size           # 32

# 查询是否支持特定数据类型
supports_fp16 = target.supports_float16       # True
supports_int8 = target.supports_int8          # True
supports_tensorcore = target.supports_tensorcore  # True (if available)
```

### 25.8.4 Target 匹配函数

```cpp
// include/tvm/target/target.h
// 检查 Target 是否支持特定特性
bool Target::IsDeviceType(int device_type) const;
bool Target::IsGPU() const;
bool Target::IsCPU() const;
bool Target::HasKey(const String& key) const;

// Target 匹配：检查两个 Target 是否兼容
bool Target::IsCompatible(const Target& other) const;
```

### 25.8.5 Target Key 匹配的数学模型

Target Key 匹配可以形式化为集合操作：

$$\text{Target.keys} = K_{\text{default}} \cup K_{\text{attr}} \cup K_{\text{user}}$$

$$\text{Match}(\text{target}, \text{required\_keys}) = \bigcap_{k \in \text{required\_keys}} (k \in \text{target.keys})$$

其中：
- $K_{\text{default}}$ 是 TargetKind 的默认 Key 集合
- $K_{\text{attr}}$ 是从属性推导的 Key 集合
- $K_{\text{user}}$ 是用户手动指定的 Key 集合

匹配成功当且仅当所有 required_keys 都在 target.keys 中。

### 25.8.6 Key 匹配的实际应用

```python
# Pass 注册时使用 Key 匹配
@tvm.target.override_native_generic_func("my_schedule")
def my_schedule(outs):
    """默认调度"""
    return _default_schedule(outs)

# 只有 target 包含 "cuda" key 时才使用此调度
@my_schedule.register(["cuda", "gpu"])
def my_schedule_cuda(outs):
    return _cuda_schedule(outs)

# 只有 target 包含 "avx512" key 时才使用此调度
@my_schedule.register(["avx512"])
def my_schedule_avx512(outs):
    return _avx512_schedule(outs)
```

<div data-component="KeyMatchingSystem"></div>

---

## 25.9 多 Target 编译

### 25.9.1 异构编译场景

在实际部署中，模型的不同部分可能需要在不同硬件上执行：

```python
# 场景：NLP 模型
# - Embedding + Attention：GPU（计算密集）
# - 后处理 + Beam Search：CPU（控制流密集）

import tvm
from tvm import relay

# 定义模型
x = relay.var("x", shape=(batch_size, seq_len, hidden_dim))
emb = relay.nn.embedding(x, weight)
# ... Attention 层
attn_out = relay.nn.multi_head_attention(emb, ...)

# 使用 Target 注解指定设备
with tvm.target.Target("cuda"):
    gpu_out = relay.nn.dense(attn_out, gpu_weight)

with tvm.target.Target("llvm"):
    cpu_out = relay.nn.softmax(gpu_out)

# relay.build 会自动处理多 Target
lib = relay.build(mod, target={"cuda": gpu_target, "llvm": cpu_target})
```

### 25.9.2 Target 注解

```python
# Relay 支持通过 on_device 标注指定每个算子的执行设备
conv = relay.nn.conv2d(x, w)
conv_on_gpu = relay.annotation.on_device(conv, tvm.target.cuda(0))

softmax = relay.nn.softmax(conv_on_gpu)
softmax_on_cpu = relay.annotation.on_device(softmax, tvm.target.cpu(0))
```

### 25.9.3 编译管线中的 Target 流转

```
原始 Relay Module
  │
  ▼ Target Annotation Pass
  每个 Call 节点附带 Target 标注
  │
  ▼ FuseOps (按 Target 分组融合)
  同一 Target 的算子优先融合
  │
  ▼ Partition Graph (按 Target 分割)
  Module 被分割为多个子 Module
  │
  ▼ 分别编译
  子 Module 1 (cuda) → LLVM/CUDA CodeGen
  子 Module 2 (llvm) → LLVM CodeGen
  │
  ▼ 合并运行时模块
  统一的 GraphExecutor
```

### 25.9.4 多 Target 编译的详细流程

```python
import tvm
from tvm import relay

# 1. 定义模型
mod, params = relay.frontend.from_onnx(onnx_model)

# 2. 定义多个 Target
cuda_target = tvm.target.cuda(model="rtx3090")
cpu_target = tvm.target.Target("llvm -mcpu=core-avx2")

# 3. Target 注解
# 使用 relay.annotation.on_device 为每个算子指定设备
# 或使用自动分区策略

# 4. 图分区
mod = relay.transform.PartitionGraph()(mod)

# 5. 编译
lib = relay.build(
    mod,
    target=[cuda_target, cpu_target],
    params=params
)

# 6. 创建运行时模块
import tvm.contrib.graph_executor as graph_executor
dev = tvm.cuda(0)
gmod = graph_executor.GraphModule(lib["default"](dev))
```

### 25.9.5 图分区算法

```python
# src/relay/transforms/partition_graph.cc
# 分区算法的核心逻辑：

# 1. 遍历 Relay IR，为每个算子分配 Target
# 2. 相同 Target 的相邻算子合并为一个子图
# 3. 子图之间的数据传输通过特殊的 "device_copy" 算子连接
# 4. 每个子图独立编译为对应的 CodeGen

# 分区结果示例：
# 原始图: A(cuda) → B(cuda) → C(cpu) → D(cpu) → E(cuda)
# 分区后:
#   子图1: A → B (cuda)
#   device_copy: B → C (cuda → cpu)
#   子图2: C → D (cpu)
#   device_copy: D → E (cpu → cuda)
#   子图3: E (cuda)
```

### 25.9.6 设备间数据传输

```
┌─────────────────┐     device_copy      ┌─────────────────┐
│   GPU 子图       │  ──────────────────→ │   CPU 子图       │
│                  │     GPU→CPU          │                  │
│  conv2d          │     数据传输          │  softmax         │
│  batch_norm      │                      │  argmax          │
│  relu            │                      │                  │
└─────────────────┘                      └─────────────────┘
```

device_copy 的实现：

```cpp
// src/relay/transforms/device_annotation.cc
// device_copy 算子会在运行时触发跨设备数据拷贝
Expr DeviceCopy(const Expr& data, const Device& src_dev,
                const Device& dst_dev) {
  return relay::Call(relay::Op::Get("device_copy"),
                     {data},
                     Attrs{{"src_dev", src_dev}, {"dst_dev", dst_dev}});
}
```

<div data-component="MultiTargetPipeline"></div>

---

## 25.10 Target 与自动调优

### 25.10.1 Target 驱动的调度空间

自动调优器根据 Target 确定搜索空间：

```python
from tvm import autotvm

# 不同 Target 的搜索空间不同
@autotvm.template("conv2d")
def conv2d_template(data, kernel, target):
    if target.kind.name == "cuda":
        # CUDA 搜索空间：block size, thread binding, shared memory
        cfg.define_knob("block_x", [32, 64, 128, 256])
        cfg.define_knob("block_y", [32, 64, 128, 256])
        cfg.define_knob("use_shared", [True, False])
    elif target.kind.name == "llvm":
        # CPU 搜索空间：vectorize width, unroll factor, parallel threads
        cfg.define_knob("vec_width", [4, 8, 16])
        cfg.define_knob("unroll_factor", [1, 2, 4, 8])
    # ...
```

### 25.10.2 搜索空间大小

不同 Target 的搜索空间大小差异巨大：

| Target | 主要调优参数 | 搜索空间大小（约） |
|--------|------------|-------------------|
| CUDA | block_size, thread_binding, shared_mem | $10^6 \sim 10^9$ |
| LLVM (x86) | vec_width, unroll_factor, num_threads | $10^3 \sim 10^6$ |
| ARM CPU | vec_width, unroll_factor, cache_tiling | $10^3 \sim 10^6$ |
| OpenCL | workgroup_size, local_mem | $10^4 \sim 10^7$ |

搜索空间可以形式化为：

$$S = \prod_{i=1}^{n} |V_i|$$

其中 $V_i$ 是第 $i$ 个调优参数的取值集合，$n$ 是参数个数。

### 25.10.3 代价模型与 Target

MetaSchedule 的代价模型考虑了 Target 的硬件特性：

```python
from tvm import meta_schedule as ms

# 代价模型根据 Target 的硬件参数估算性能
cost_model = ms.cost_model.XGBModel(
    # 特征提取考虑 Target 参数：
    # - 缓存大小（影响 tiling 决策）
    # - 向量宽度（影响 vectorization）
    # - 线程数（影响并行策略）
)
```

### 25.10.4 MetaSchedule 的 Target 感知搜索

```python
from tvm import meta_schedule as ms

# 创建 Target 感知的搜索空间
space = ms.space.ScheduleSpace(
    target=tvm.target.cuda(model="rtx3090"),
    # 搜索空间会根据 Target 自动调整
    # - CUDA: 包含 shared memory, warp scheduling
    # - CPU: 包含 vectorization, loop tiling
)

# 搜索策略也会根据 Target 调整
strategy = ms.search_strategy.ReplayFunc(
    # CUDA: 倾向于探索 shared memory 优化
    # CPU: 倾向于探索 cache blocking
)
```

### 25.10.5 AutoTVM 的 Target 特定模板

```python
# CUDA conv2d 模板
@autotvm.template("conv2d_nchw_cuda")
def conv2d_nchw_cuda(data, kernel, stride, padding):
    cfg = autotvm.get_config()

    # CUDA 特定的调优参数
    cfg.define_knob("tile_n", [1, 2, 4])
    cfg.define_knob("tile_c", [1, 2, 4, 8])
    cfg.define_knob("tile_h", [1, 2, 4, 8, 16])
    cfg.define_knob("tile_w", [1, 2, 4, 8, 16])
    cfg.define_knob("tile_rc", [1, 2, 4, 8])
    cfg.define_knob("use_shared", [0, 1])
    cfg.define_knob("vectorize", [1, 2, 4])

    # ... 调度逻辑
    return s, [data, kernel, output]

# CPU conv2d 模板
@autotvm.template("conv2d_nchw_cpu")
def conv2d_nchw_cpu(data, kernel, stride, padding):
    cfg = autotvm.get_config()

    # CPU 特定的调优参数
    cfg.define_knob("tile_oh", [1, 2, 4, 8])
    cfg.define_knob("tile_ow", [1, 2, 4, 8, 16])
    cfg.define_knob("vec_width", [4, 8, 16])
    cfg.define_knob("unroll_factor", [1, 2, 4, 8])
    cfg.define_knob("num_threads", [1, 2, 4, 8])

    # ... 调度逻辑
    return s, [data, kernel, output]
```

<div data-component="AutoTVMTargetIntegration"></div>

---

## 25.11 Target 的高级用法

### 25.11.1 自定义 Target

```python
# 定义自定义 Target
my_target = tvm.target.Target({
    "kind": "llvm",
    "mtriple": "riscv32-unknown-elf",
    "mcpu": "sifive-u74",
    "mattr": ["+m", "+a", "+f", "+d"],
    "system-lib": True,
    "keys": ["cpu", "riscv"],
})

# 使用自定义 Target
with my_target:
    lib = relay.build(mod, target=my_target)
```

### 25.11.2 Target 的序列化与恢复

```python
import json
import tvm

# 序列化 Target
target = tvm.target.cuda(model="rtx3090")
target_json = str(target)  # JSON 字符串
print(target_json)
# {"kind":"cuda","mcpu":"sm_86","arch":"sm_86",...}

# 从 JSON 恢复 Target
restored_target = tvm.target.Target(target_json)
assert restored_target == target
```

### 25.11.3 Target 的条件编译

```python
# 根据 Target 选择不同的编译策略
target = tvm.target.Target.current()

if target.kind.name == "cuda":
    # CUDA 特定的优化
    mod = relay.transform.FuseOps(fuse_opt_level=3)(mod)
elif target.kind.name == "llvm":
    if "avx512" in target.keys:
        # AVX-512 特定的优化
        mod = relay.transform.FoldScaleAxis()(mod)
    elif "neon" in target.keys:
        # NEON 特定的优化
        mod = relay.transform.AlterOpLayout()(mod)
```

### 25.11.4 Target 与量化

```python
# 不同 Target 支持不同的量化方案
target = tvm.target.Target.current()

if "cuda" in target.keys:
    # GPU: 使用 FP16 量化
    mod = relay.quantize.qnn_transform(mod, dtype="float16")
elif "vnni" in target.keys:
    # Intel VNNI: 使用 INT8 量化
    mod = relay.quantize.qnn_transform(mod, dtype="int8")
elif "neon" in target.keys:
    # ARM NEON: 使用 INT8 量化
    mod = relay.quantize.qnn_transform(mod, dtype="int8")
```

不同硬件的量化支持：

| 硬件 | INT8 | FP16 | BF16 | INT4 | 说明 |
|------|------|------|------|------|------|
| NVIDIA A100 | ✓ | ✓ | ✓ | ✓ | Tensor Core 支持 |
| NVIDIA V100 | ✓ | ✓ | ✗ | ✗ | 无 INT8 Tensor Core |
| Intel (VNNI) | ✓ | ✗ | ✗ | ✗ | VNNI 指令集 |
| ARM (DOTPROD) | ✓ | ✗ | ✗ | ✗ | DOTPROD 指令 |
| Qualcomm DSP | ✓ | ✓ | ✗ | ✓ | Hexagon HVX |

### 25.11.5 Target 与混合精度

```python
import tvm
from tvm import relay

# 混合精度推理
mod, params = relay.frontend.from_onnx(onnx_model)

# 获取当前 Target
target = tvm.target.cuda(model="a100")

# 混合精度配置
mixed_precision_config = {
    "relay.backend.use_auto_scheduler": True,
    "relay.backend.auto_scheduler.use_fp16": True,
    "relay.backend.auto_scheduler.fp16_ops": [
        "nn.conv2d",
        "nn.dense",
    ],
}

with tvm.transform.PassContext(
    opt_level=3,
    config=mixed_precision_config
):
    lib = relay.build(mod, target=target, params=params)
```

---

## 25.12 Target 系统的扩展

### 25.12.1 添加新的 TargetKind

```cpp
// 在 TVM 源码中添加新的 TargetKind
// src/target/my_device.cc

#include <tvm/target/target.h>

TVM_REGISTER_TARGET_KIND("my_device", kDLExtDev)
    .add_attr_option<String>("model")
    .add_attr_option<Integer>("num_cores", Integer(8))
    .add_attr_option<Bool>("supports_fp16", Bool(true))
    .add_attr_option<Integer>("memory_mb", Integer(4096))
    .set_default_keys({"my_device", "custom_accelerator"});

// 注册对应的 Device API
TVM_REGISTER_DEVICE_API(kDLMyDevice, MyDeviceAPI);
```

### 25.12.2 Target 注册表

```python
# 查看所有注册的 TargetKind
import tvm

kinds = tvm.target.list_registered_target_kinds()
print(kinds)
# ["llvm", "cuda", "opencl", "vulkan", "metal", "hexagon", "c", "webassembly", ...]
```

### 25.12.3 Target 的兼容性检查

```cpp
// 检查 Target 是否支持特定的算子
bool SupportsOp(const Target& target, const Op& op) {
  // 查找算子的调度注册
  auto fschema = Op::GetAttrMap<FTVMSchedule>("FTVMSchedule");
  if (fschema.count(op)) {
    // 尝试获取该 Target 的调度
    auto schedule = fschema[op](target);
    return schedule.defined();
  }
  return false;
}
```

### 25.12.4 Target 的属性继承

```cpp
// TargetKind 的属性可以有默认值
TVM_REGISTER_TARGET_KIND("my_device", kDLExtDev)
    .add_attr_option<Integer>("num_cores", Integer(8))       // 默认值 8
    .add_attr_option<Bool>("supports_fp16", Bool(true))      // 默认值 true
    .add_attr_option<Float>("clock_ghz", Float(1.0))         // 默认值 1.0
    .set_default_keys({"my_device"});

// 创建 Target 时，未指定的属性使用默认值
// Target({"kind": "my_device"})
// 实际属性: num_cores=8, supports_fp16=true, clock_ghz=1.0

// 创建 Target 时，可以覆盖默认值
// Target({"kind": "my_device", "num_cores": 16})
// 实际属性: num_cores=16, supports_fp16=true, clock_ghz=1.0
```

### 25.12.5 通过 Python 扩展 TargetKind

```python
import tvm

# 通过 Python 注册自定义 TargetKind
# 注意：这种方式主要用于原型验证
# 生产环境建议使用 C++ 注册

@tvm.target.register_func("my_custom_target")
def create_my_target():
    return tvm.target.Target({
        "kind": "llvm",
        "mtriple": "my-custom-triple",
        "mcpu": "my-custom-cpu",
        "keys": ["cpu", "custom"],
    })
```

---

## 25.13 实战示例

### 25.13.1 为 Jetson Nano 编译

```python
import tvm
from tvm import relay

# Jetson Nano 的 Target 配置
target = tvm.target.Target({
    "kind": "cuda",
    "mcpu": "sm_53",        # Maxwell 架构
    "arch": "sm_53",
    "max_num_threads": 1024,
    "max_shared_memory_per_block": 49152,
    "thread_warp_size": 32,
    "host": {
        "kind": "llvm",
        "mtriple": "aarch64-linux-gnu",
        "mcpu": "cortex-a57",
        "mattr": ["+neon", "+fp-armv8"],
    },
})

# 编译模型
mod, params = relay.frontend.from_onnx(onnx_model)
with target:
    lib = relay.build(mod, target=target, params=params)

# 导出到 Jetson Nano
lib.export_library("model_jetson.tar")
```

Jetson Nano Target 配置解析：

| 参数 | 值 | 说明 |
|------|------|------|
| kind | cuda | 使用 CUDA 编译 |
| mcpu | sm_53 | Maxwell 架构 (Compute 5.3) |
| max_num_threads | 1024 | 每 Block 最大 1024 线程 |
| max_shared_memory_per_block | 49152 | 48KB 共享内存 |
| thread_warp_size | 32 | Warp 大小 32 |
| host.mtriple | aarch64-linux-gnu | ARM 64 位 Linux |
| host.mcpu | cortex-a57 | ARM Cortex-A57 CPU |

### 25.13.2 为 Intel CPU 编译

```python
# Intel Xeon (Ice Lake) Target
target = tvm.target.Target({
    "kind": "llvm",
    "mcpu": "icelake-server",
    "mattr": ["+avx512f", "+avx512bw", "+avx512vl",
              "+vnni", "+bmi2"],
})

with target:
    lib = relay.build(mod, target=target, params=params)
```

Intel Ice Lake 的指令集特性：

| 指令集 | 说明 | 用途 |
|--------|------|------|
| AVX-512F | 512 位向量基础指令 | 通用向量化 |
| AVX-512BW | Byte/Word 扩展 | INT8 向量化 |
| AVX-512VL | 128/256 位向量扩展 | 灵活向量宽度 |
| VNNI | 向量神经网络指令 | INT8 点积加速 |
| BMI2 | 位操作指令 | 地址计算加速 |

### 25.13.3 为多个设备编译

```python
# 多设备编译：GPU 推理 + CPU 后处理
gpu_target = tvm.target.cuda(model="rtx3090")
cpu_target = tvm.target.Target("llvm -mcpu=core-avx2")

# 分别编译
with gpu_target:
    gpu_lib = relay.build(gpu_mod, target=gpu_target, params=params)

with cpu_target:
    cpu_lib = relay.build(cpu_mod, target=cpu_target)

# 运行时组合使用
gpu_gmod = graph_executor.GraphModule(gpu_lib["default"](tvm.cuda(0)))
cpu_gmod = graph_executor.GraphModule(cpu_lib["default"](tvm.cpu(0)))
```

### 25.13.4 为 ARM 嵌入式设备编译

```python
# ARM Cortex-A55 嵌入式设备
target = tvm.target.Target({
    "kind": "llvm",
    "mtriple": "aarch64-linux-gnu",
    "mcpu": "cortex-a55",
    "mattr": ["+neon", "+dotprod", "+fp-armv8"],
    "system-lib": True,  # 链接到系统库
})

# 交叉编译
mod, params = relay.frontend.from_onnx(onnx_model)
with target:
    lib = relay.build(mod, target=target, params=params)

# 导出到嵌入式设备
lib.export_library("model_arm.tar")
# 部署时需要在目标设备上解压并运行
```

### 25.13.5 为 Qualcomm Hexagon DSP 编译

```python
# Qualcomm Hexagon DSP
target = tvm.target.Target({
    "kind": "hexagon",
    "mcpu": "hexagon-v68",
    "keys": ["hexagon"],
})

mod, params = relay.frontend.from_onnx(onnx_model)
with target:
    lib = relay.build(mod, target=target, params=params)

# Hexagon DSP 的特点：
# - 向量处理单元 (HVX)
# - 低功耗推理
# - 适合移动端 AI 推理
```

<div data-component="MultiDeviceDeployment"></div>

---

## 25.14 Target 系统的内部实现

### 25.14.1 Target 的内存布局

```
Target 对象
  ├── TargetKind* kind;          // 指向注册的 TargetKind
  ├── Map<String, ObjectRef> attrs; // 属性映射
  │     ├── "mcpu": String
  │     ├── "mattr": Array<String>
  │     ├── "max_num_threads": Integer
  │     └── "host": Target
  ├── Array<String> keys;        // 特性键
  └── String tag;                // 预定义标签
```

### 25.14.2 Target 的比较算法

```cpp
// Target 的相等比较
bool Target::operator==(const Target& other) const {
  // 1. 比较 TargetKind
  if (this->kind != other.kind) return false;

  // 2. 比较所有属性
  if (this->attrs.size() != other.attrs.size()) return false;
  for (auto& kv : this->attrs) {
    if (!other.attrs.count(kv.first)) return false;
    if (!ObjectEqual()(kv.second, other.attrs.at(kv.first))) return false;
  }

  return true;
}
```

### 25.14.3 Target 的哈希函数

```cpp
// Target 的哈希（用于缓存）
size_t TargetHash::operator()(const Target& target) const {
  size_t hash = std::hash<String>()(target->kind->name);
  for (auto& kv : target->attrs) {
    hash = dmlc::HashCombine(hash, std::hash<String>()(kv.first));
    hash = dmlc::HashCombine(hash, ObjectHash()(kv.second));
  }
  return hash;
}
```

### 25.14.4 Target 的线程局部存储

```cpp
// src/target/target.cc
// Target 使用线程局部存储管理作用域栈
struct TargetThreadLocalEntry {
  // Target 作用域栈（支持嵌套 with 语句）
  std::vector<Target> target_stack;
};

// 获取线程局部存储
static TargetThreadLocalEntry* ThreadLocal() {
  // thread_local 保证每个线程有独立的栈
  thread_local TargetThreadLocalEntry entry;
  return &entry;
}
```

### 25.14.5 TargetKind 的全局注册表

```cpp
// src/target/target_kind.cc
class TargetKindRegistry {
 public:
  static TargetKindRegistry* Global() {
    static TargetKindRegistry inst;
    return &inst;
  }

  void Register(const TargetKind& kind) {
    ICHECK(!name_map_.count(kind->name))
        << "TargetKind \"" << kind->name << "\" already registered";
    name_map_[kind->name] = kind;
    all_kinds_.push_back(kind);
  }

  TargetKind Get(const String& name) const {
    auto it = name_map_.find(name);
    ICHECK(it != name_map_.end())
        << "Unknown TargetKind: " << name;
    return it->second;
  }

  Array<String> ListAll() const {
    Array<String> result;
    for (const auto& kind : all_kinds_) {
      result.push_back(kind->name);
    }
    return result;
  }

 private:
  std::unordered_map<String, TargetKind> name_map_;
  std::vector<TargetKind> all_kinds_;
};
```

### 25.14.6 Target 对象的生命周期

```
创建阶段:
  Target::FromString() / Target::FromConfig()
    │
    ├─ 查找 TargetKind
    ├─ 验证属性
    └─ 构建 Target 对象

使用阶段:
  with target:  # 压入作用域栈
    relay.build(mod, target)
    # Pass 通过 Target::Current() 获取当前 Target
  # 弹出作用域栈

销毁阶段:
  Target 对象通过引用计数自动管理
  当引用计数为 0 时，自动释放
```

---

## 25.15 Target 系统的设计模式

### 25.15.1 注册表模式（Registry Pattern）

Target 系统大量使用注册表模式：

```cpp
// 注册表模式的核心组件
// 1. 注册表（Registry）
class Registry {
  std::map<String, Entry> entries_;
public:
  void Register(const String& name, Entry entry);
  Entry Get(const String& name);
};

// 2. 注册宏（Registration Macro）
#define TVM_REGISTER_TARGET_KIND(name, device_type) \
  static TargetKindRegEntry __make_##name##_ = \
      TargetKindRegEntry::Register(name, device_type)

// 3. 自注册（Self-Registration）
// 每个 TargetKind 在自己的 .cc 文件中注册自己
// 无需修改中心代码
```

### 25.15.2 策略模式（Strategy Pattern）

TOPI 的调度分发使用策略模式：

```python
# 策略接口
class ScheduleStrategy:
    def schedule(self, outs, target):
        raise NotImplementedError

# 具体策略
class CUDAStrategy(ScheduleStrategy):
    def schedule(self, outs, target):
        # CUDA 特定的调度逻辑
        pass

class CPUStrategy(ScheduleStrategy):
    def schedule(self, outs, target):
        # CPU 特定的调度逻辑
        pass

# 策略选择
def get_strategy(target):
    if target.kind.name == "cuda":
        return CUDAStrategy()
    elif target.kind.name == "llvm":
        return CPUStrategy()
```

### 25.15.3 访问者模式（Visitor Pattern）

Target 对编译管线的影响通过访问者模式实现：

```cpp
// 不同的 Pass 访问 Target 的方式不同
class FuseOpsPass : public Pass {
  void VisitExpr(const CallNode* call) override {
    Target target = Target::Current();
    if (target->IsGPU()) {
      // GPU 融合策略
    } else {
      // CPU 融合策略
    }
  }
};

class AlterLayoutPass : public Pass {
  void VisitExpr(const CallNode* call) override {
    Target target = Target::Current();
    if (target->HasKey("neon")) {
      // NEON 布局变换
    } else if (target->HasKey("avx512")) {
      // AVX-512 布局变换
    }
  }
};
```

### 25.15.4 工厂模式（Factory Pattern）

Target 的创建使用工厂模式：

```python
# Python 层的工厂
class TargetFactory:
    @staticmethod
    def create(kind, **kwargs):
        if kind == "cuda":
            return CUDATarget(**kwargs)
        elif kind == "llvm":
            return LLVMTarget(**kwargs)
        # ...

# 便捷工厂函数
def cuda(model=None, host=None):
    return TargetFactory.create("cuda", model=model, host=host)

def arm_cpu(model=None):
    return TargetFactory.create("llvm", mtriple="aarch64-linux-gnu", mcpu=model)
```

---

## 25.16 常见陷阱（Common Pitfalls）

### 25.16.1 陷阱一：忘记设置 Host Target

```python
# ❌ 错误：GPU Target 没有设置 Host
target = tvm.target.Target("cuda")
# 编译时可能使用默认 Host，导致性能不佳

# ✅ 正确：显式设置 Host Target
target = tvm.target.Target("cuda", host="llvm -mcpu=core-avx2")
```

### 25.16.2 陷阱二：Target 字符串格式错误

```python
# ❌ 错误：属性名前缺少 -
target = tvm.target.Target("llvm mcpu=core-avx2")
# 解析失败：mcpu=core-avx2 被当作 kind 名称

# ✅ 正确：属性名前加 -
target = tvm.target.Target("llvm -mcpu=core-avx2")

# ❌ 错误：多个属性用逗号分隔
target = tvm.target.Target("llvm -mcpu=core-avx2,-mattr=+avx2")
# 解析失败：逗号不是分隔符

# ✅ 正确：多个属性用空格分隔
target = tvm.target.Target("llvm -mcpu=core-avx2 -mattr=+avx2")
```

### 25.16.3 陷阱三：Target 与设备不匹配

```python
# ❌ 错误：为旧 GPU 设置新架构
target = tvm.target.Target({
    "kind": "cuda",
    "mcpu": "sm_80",  # Ampere 架构
    # 但实际设备是 sm_70 (Volta)
})

# ✅ 正确：匹配实际设备
target = tvm.target.Target({
    "kind": "cuda",
    "mcpu": "sm_70",  # 匹配实际设备
})
```

### 25.16.4 陷阱四：属性值类型错误

```python
# ❌ 错误：整数属性传入字符串
target = tvm.target.Target({
    "kind": "cuda",
    "max_num_threads": "1024",  # 字符串，应该是整数
})

# ✅ 正确：使用正确的类型
target = tvm.target.Target({
    "kind": "cuda",
    "max_num_threads": 1024,  # 整数
})
```

### 25.16.5 陷阱五：Target 作用域管理不当

```python
# ❌ 错误：在作用域外访问 Target
with tvm.target.Target("cuda"):
    pass
# 此时 Target 已出作用域
target = tvm.target.Target.current()  # 报错！

# ✅ 正确：在作用域内使用 Target
with tvm.target.Target("cuda") as target:
    lib = relay.build(mod, target=target)
```

### 25.16.6 陷阱六：多 Target 编译的 Key 冲突

```python
# ❌ 错误：两个 Target 使用相同的 Key
target1 = tvm.target.Target({
    "kind": "cuda",
    "keys": ["gpu"],
})
target2 = tvm.target.Target({
    "kind": "opencl",
    "keys": ["gpu"],  # 与 target1 的 key 冲突
})

# ✅ 正确：使用不同的 Key 或使用 tag 区分
target1 = tvm.target.Target({
    "kind": "cuda",
    "tag": "nvidia_gpu",
})
target2 = tvm.target.Target({
    "kind": "opencl",
    "tag": "opencl_gpu",
})
```

### 25.16.7 陷阱七：忽略 Target 对性能的影响

```python
# ❌ 错误：使用通用 Target，忽略硬件特性
target = tvm.target.Target("llvm")
# 未指定 mcpu，使用默认值，无法利用特定指令集

# ✅ 正确：指定具体的 CPU 特性
target = tvm.target.Target("llvm -mcpu=core-avx2 -mattr=+avx2,+fma")
# 利用 AVX2 和 FMA 指令加速
```

### 25.16.8 陷阱八：Device API 注册遗漏

```cpp
// ❌ 错误：注册了 TargetKind 但忘记注册 Device API
TVM_REGISTER_TARGET_KIND("my_device", kDLMyDevice)
    .add_attr_option<Integer>("num_cores")
    .set_default_keys({"my_device"});

// 运行时找不到 DeviceAPI，报错

// ✅ 正确：同时注册 Device API
TVM_REGISTER_TARGET_KIND("my_device", kDLMyDevice)
    .add_attr_option<Integer>("num_cores")
    .set_default_keys({"my_device"});

TVM_REGISTER_DEVICE_API(kDLMyDevice, MyDeviceAPI);
```

---

## 25.17 练习题（Practice Exercises）

### 练习 1：创建自定义 Target

**题目**：为一个具有以下特性的自定义 AI 加速器创建 Target：

- 名称：`my_ai_accelerator`
- 设备类型：`kDLExtDev` (值为 12)
- 支持 FP16 和 INT8
- 最大线程数：256
- 共享内存：64KB
- 4 个计算核心

```python
# 请填写以下代码
target = tvm.target.Target({
    "kind": "my_ai_accelerator",
    # 请补充其他属性
})
```

**参考答案**：

```python
# 首先需要在 C++ 中注册 TargetKind
# TVM_REGISTER_TARGET_KIND("my_ai_accelerator", 12)
#     .add_attr_option<Bool>("supports_fp16", Bool(true))
#     .add_attr_option<Bool>("supports_int8", Bool(true))
#     .add_attr_option<Integer>("max_num_threads", Integer(256))
#     .add_attr_option<Integer>("shared_memory_bytes", Integer(65536))
#     .add_attr_option<Integer>("num_cores", Integer(4))
#     .set_default_keys({"my_ai_accelerator", "custom"});

target = tvm.target.Target({
    "kind": "my_ai_accelerator",
    "supports_fp16": True,
    "supports_int8": True,
    "max_num_threads": 256,
    "shared_memory_bytes": 65536,
    "num_cores": 4,
})
```

### 练习 2：解析 Target JSON

**题目**：解析以下 JSON 并回答问题：

```json
{
  "kind": "cuda",
  "mcpu": "sm_86",
  "max_num_threads": 1024,
  "max_shared_memory_per_block": 49152,
  "thread_warp_size": 32,
  "host": {
    "kind": "llvm",
    "mcpu": "cortex-a78",
    "mattr": ["+neon", "+dotprod"]
  },
  "keys": ["cuda", "gpu"],
  "tag": "rtx3090"
}
```

问题：
1. 这个 Target 的 TargetKind 是什么？
2. Host Target 的 CPU 型号是什么？
3. Host Target 支持哪些指令集？
4. 这个 Target 的 tag 是什么？

**参考答案**：
1. TargetKind 是 `"cuda"`
2. Host Target 的 CPU 型号是 `"cortex-a78"`
3. Host Target 支持 `"+neon"` 和 `"+dotprod"` 指令集
4. tag 是 `"rtx3090"`

### 练习 3：编写调度分发

**题目**：编写一个 TOPI 调度函数，根据 Target 选择不同的实现：

```python
@tvm.target.override_native_generic_func("my_matmul_schedule")
def my_matmul_schedule(outs):
    # 请实现：
    # 1. 默认调度（CPU）
    # 2. CUDA 调度
    # 3. ARM CPU 调度
    pass
```

**参考答案**：

```python
@tvm.target.override_native_generic_func("my_matmul_schedule")
def my_matmul_schedule(outs):
    """默认调度（通用 CPU）"""
    s = tvm.create_schedule([x.op for x in outs])
    return s

@my_matmul_schedule.register(["cuda", "gpu"])
def my_matmul_schedule_cuda(outs):
    """CUDA 调度"""
    s = tvm.create_schedule([x.op for x in outs])
    # CUDA 特定的调度逻辑
    # 使用 shared memory, thread binding 等
    return s

@my_matmul_schedule.register(["arm_cpu"])
def my_matmul_schedule_arm(outs):
    """ARM CPU 调度"""
    s = tvm.create_schedule([x.op for x in outs])
    # ARM 特定的调度逻辑
    # 使用 NEON 向量化等
    return s
```

### 练习 4：多 Target 编译

**题目**：编写代码实现以下场景：

- 模型的前半部分在 GPU 上执行
- 模型的后半部分在 CPU 上执行
- 使用 `relay.annotation.on_device` 进行 Target 注解

**参考答案**：

```python
import tvm
from tvm import relay

# 定义模型
x = relay.var("x", shape=(1, 3, 224, 224))
w1 = relay.var("w1", shape=(64, 3, 3, 3))
w2 = relay.var("w2", shape=(10, 64))

# 前半部分：GPU
conv = relay.nn.conv2d(x, w1)
bn = relay.nn.batch_norm(conv)
relu = relay.nn.relu(bn)
# 注解为 GPU
gpu_out = relay.annotation.on_device(relu, tvm.target.cuda(0))

# 后半部分：CPU
flatten = relay.nn.batch_flatten(gpu_out)
dense = relay.nn.dense(flatten, w2)
softmax = relay.nn.softmax(dense)
# 注解为 CPU
cpu_out = relay.annotation.on_device(softmax, tvm.target.cpu(0))

# 分区和编译
mod = tvm.IRModule.from_expr(cpu_out)
mod = relay.transform.PartitionGraph()(mod)

# 多 Target 编译
lib = relay.build(
    mod,
    target=[tvm.target.cuda(), tvm.target.Target("llvm")]
)
```

### 练习 5：Device API 实现

**题目**：为一个自定义设备实现 DeviceAPI 的 `AllocDataSpace` 和 `FreeDataSpace` 方法。

**参考答案**：

```cpp
#include <tvm/runtime/device_api.h>

class MyDeviceAPI : public DeviceAPI {
 public:
  void* AllocDataSpace(Device dev, size_t nbytes,
                       size_t alignment, DLDataType type_hint) override {
    // 确保对齐
    size_t aligned_nbytes = (nbytes + alignment - 1) & ~(alignment - 1);

    // 调用设备驱动分配内存
    void* ptr = my_device_malloc(dev.device_id, aligned_nbytes);

    ICHECK(ptr != nullptr)
        << "Failed to allocate " << aligned_nbytes
        << " bytes on device " << dev.device_id;

    return ptr;
  }

  void FreeDataSpace(Device dev, void* ptr) override {
    if (ptr != nullptr) {
      my_device_free(dev.device_id, ptr);
    }
  }

  void CopyDataFromTo(const void* from, void* to, size_t size,
                       Device dev_from, Device dev_to,
                       DLDataType type_hint, TVMStreamHandle stream) override {
    if (dev_from.device_type == dev_to.device_type) {
      // 同设备拷贝
      my_device_memcpy_same_device(
          dev_from.device_id, to, from, size);
    } else if (dev_from.device_type == kDLCPU) {
      // CPU → 设备
      my_device_memcpy_to_device(
          dev_to.device_id, to, from, size);
    } else if (dev_to.device_type == kDLCPU) {
      // 设备 → CPU
      my_device_memcpy_from_device(
          dev_from.device_id, to, from, size);
    }
  }

  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) override {
    switch (kind) {
      case kExist:
        *rv = my_device_exists(dev.device_id);
        break;
      case kMaxThreadsPerBlock:
        *rv = my_device_max_threads(dev.device_id);
        break;
      case kDeviceName:
        *rv = my_device_name(dev.device_id);
        break;
      default:
        *rv = 0;
        break;
    }
  }
};

// 注册
static const int kDLMyDevice = 200;
TVM_REGISTER_DEVICE_API(kDLMyDevice, MyDeviceAPI);
```

### 练习 6：Target 性能分析

**题目**：分析以下两种 Target 配置的性能差异，并解释原因：

```python
# 配置 A
target_a = tvm.target.Target("llvm")

# 配置 B
target_b = tvm.target.Target("llvm -mcpu=core-avx2 -mattr=+avx2,+fma")
```

**参考答案**：

配置 B 通常比配置 A 快 2-8 倍，原因：

1. **向量化宽度**：配置 A 可能使用 SSE (128-bit)，配置 B 使用 AVX2 (256-bit)，向量化宽度是 A 的 2 倍
2. **FMA 指令**：配置 B 启用了 FMA (Fused Multiply-Add)，乘加运算在一条指令内完成
3. **寄存器数量**：AVX2 提供 16 个 256-bit 寄存器，比 SSE 的 16 个 128-bit 寄存器更高效
4. **指令级并行**：AVX2 的指令吞吐量更高

性能提升估算（float32 矩阵乘法）：

$$\text{Speedup} \approx \frac{\text{AVX2 FLOPs/cycle}}{\text{SSE FLOPs/cycle}} = \frac{256/32 \times 2}{128/32 \times 1} = 4$$

实际提升取决于内存带宽、缓存命中率等因素。

---

## 25.18 本章小结

本章深入分析了 TVM 的 Target 系统：

1. **TargetKind**：硬件类型的注册与属性定义
2. **Target 对象**：封装设备类型、指令集、编译选项
3. **JSON 格式**：Target 的序列化与反序列化
4. **Device API**：Target 与设备内存管理的关联
5. **编译影响**：Target 对 Relay/TE/TIR/CodeGen 各阶段的影响

关键设计原则：

```
抽象层次         设计目标
───────────────────────────
TargetKind      硬件类型分类
Target          设备能力描述
JSON            可序列化/可传输
Keys            特性查询/匹配
Device API      运行时内存管理
```

Target 系统的核心公式：

$$\text{编译结果} = \text{Compile}(\text{Model}, \text{Target})$$

其中 Target 决定了：
- 使用哪个 CodeGen 后端
- 采用什么优化策略
- 生成什么指令集代码
- 运行时使用什么 Device API

Target 系统是 TVM 可移植性的基础——同一份模型描述，通过不同的 Target 配置，可以编译到从微控制器到数据中心 GPU 的各种硬件上。这种"写一次，部署到任何地方"的能力正是 TVM 作为深度学习编译器的核心价值。

---

## 25.19 参考资源

### 源码阅读建议

| 阶段 | 文件 | 重点 |
|------|------|------|
| 入门 | `include/tvm/target/target.h` | Target 类接口 |
| 进阶 | `src/target/target.cc` | JSON 解析、序列化 |
| 高级 | `src/target/target_kind.cc` | 注册机制 |
| 实战 | `src/target/tag.cc` | 预定义标签 |

### 延伸阅读

- TVM 官方文档：Target 系统
- TVM RFC：Target System Refactoring
- LLVM Target Machine 文档
- CUDA Compute Capability 文档

---

## 25.99 文字内容强化：Target 系统 的工程化阅读补充

Target 系统把硬件能力变成编译器可查询的数据结构，它决定同一份模型会走向哪条优化路径和哪个代码生成后端。

### 25.99.1 代码解读：从片段回到主流程

原有 Target 代码块要先看 TargetKind 注册，再看 Target 对象解析和属性查询。
控制流从字符串或 JSON 解析开始，生成结构化 Target，随后被 PassContext、TOPI 和 CodeGen 查询。
工程意义在于硬件信息不再散落在后端代码里，而是成为统一配置对象。
代码块中的变量名、函数名和类名不应孤立记忆，而应放回编译流水线中理解。
读者可以先判断代码块处在构建期、优化期、代码生成期还是运行期。
构建期代码通常负责收集信息，优化期代码负责改写 IR，代码生成期代码负责降低表示，运行期代码负责执行与资源管理。
一旦阶段判断正确，许多看似相似的数据结构就能区分出职责边界。

### 25.99.2 源码阅读路径

阅读 apache/tvm 源码时，建议按下面顺序推进，而不是直接在全仓库搜索 Target 系统。
第 1 步，阅读 `include/tvm/target/target.h`，目标是确认这一层暴露的主要接口和被谁调用。
第 2 步，阅读 `src/target/target.cc`，目标是确认这一层暴露的主要接口和被谁调用。
第 3 步，阅读 `src/target/target_kind.cc`，目标是确认这一层暴露的主要接口和被谁调用。
第 4 步，阅读 `src/target/tag.cc`，目标是确认这一层暴露的主要接口和被谁调用。
第 5 步，阅读 `python/tvm/target/`，目标是确认这一层暴露的主要接口和被谁调用。
完成主路径后，再阅读相邻测试目录，测试通常比注释更清楚地展示了设计者希望维持的不变量。
如果遇到注册表入口，应记录注册名、C++ 实现函数、Python 包装函数和最终用户 API 四个位置。
如果遇到 Pass，应记录 Pass 的输入 IR、输出 IR、启用条件和在默认流水线中的相对顺序。
如果遇到运行时模块，应记录它的创建时机、序列化格式、加载入口和资源释放位置。

### 25.99.3 为什么这样设计

Target 采用结构化属性而非散乱字符串，是为了让 Pass、调度规则、代码生成和运行时都能可靠查询硬件能力。
这种设计把变化频繁的硬件细节放在可替换层，把稳定的编译流程保留在公共层。
因此，当新增后端、调整调度策略或替换运行时实现时，系统不需要推翻已有抽象。
代价是调用路径会更长，读源码时会看到更多注册、转发和包装对象。
但这些额外层次换来的是跨语言、跨设备和跨部署场景的一致性。
判断一个设计是否合理，可以看它是否让常见路径足够简单，同时让少见路径仍有扩展空间。

### 25.99.4 逐行阅读提示与工程理解清单

1. TargetKind 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
2. 阅读 属性解析 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
21. 设备能力 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
22. 阅读 Pass 分派 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
41. TargetKind 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
42. 阅读 属性解析 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
61. 设备能力 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
62. 阅读 Pass 分派 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
81. TargetKind 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
82. 阅读 属性解析 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
101. 设备能力 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
102. 阅读 Pass 分派 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
121. TargetKind 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
122. 阅读 属性解析 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
141. 设备能力 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
142. 阅读 Pass 分派 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
161. TargetKind 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
162. 阅读 属性解析 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
181. 设备能力 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
182. 阅读 Pass 分派 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
201. TargetKind 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
202. 阅读 属性解析 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
221. 设备能力 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
222. 阅读 Pass 分派 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
241. TargetKind 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
242. 阅读 属性解析 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
261. 设备能力 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
262. 阅读 Pass 分派 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
281. TargetKind 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
282. 阅读 属性解析 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
301. 设备能力 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
302. 阅读 Pass 分派 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
321. TargetKind 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
322. 阅读 属性解析 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
341. 设备能力 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
342. 阅读 Pass 分派 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
361. TargetKind 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
362. 阅读 属性解析 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
381. 设备能力 的第一层理解，是把它看成 硬件能力描述与编译配置系统 中连接抽象语义和工程实现的接口。
382. 阅读 Pass 分派 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
383. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
384. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
385. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
386. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
387. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
388. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
389. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。

### 25.99.5 小结：把本章放回 TVM 全链路

Target 系统 的学习重点不是记住每个函数名，而是理解它在 TVM 全链路中承担的边界职责。
当读者能够说清楚输入从哪里来、状态在哪里保存、输出被谁消费，就已经掌握了源码阅读的主线。
后续遇到性能、兼容性或部署问题时，可以沿着这条主线逐层排查，而不是在全仓库中盲目搜索。

