"use client";

import { useState } from "react";

const errors = [
  {
    id: "shape",
    name: "Shape 不匹配",
    icon: "📐",
    severity: "error",
    msg: "matmul: shapes (128,64) and (32,10) not aligned",
    cause: "矩阵乘法维度不兼容: A 的列数 ≠ B 的行数",
    fix: "检查 input shape，确保 matmul 维度对齐\nA: (M, K) × B: (K, N) → (M, N)",
    code: `# 错误
A = relay.var("A", shape=(128, 64))
B = relay.var("B", shape=(32, 10))
relay.nn.dense(A, B)  # 64 ≠ 32!

# 修复
B = relay.var("B", shape=(64, 10))`,
  },
  {
    id: "dtype",
    name: "数据类型错误",
    icon: "🔢",
    severity: "error",
    msg: "Cannot add float32 and int32 tensors",
    cause: "混合了不同 dtype 的张量进行运算",
    fix: "使用 cast 统一类型\nrelay.cast(x, dtype='float32')",
    code: `# 错误
x = relay.var("x", dtype="float32")
y = relay.var("y", dtype="int32")
relay.add(x, y)  # 类型不匹配!

# 修复
y_f = relay.cast(y, dtype="float32")
relay.add(x, y_f)`,
  },
  {
    id: "device",
    name: "设备不匹配",
    icon: "💻",
    severity: "error",
    msg: "Cannot copy tensor from gpu:0 to cpu:0 implicitly",
    cause: "尝试在不同设备上的张量间运算",
    fix: "显式拷贝到同一设备\ntvm.nd.array(data, target_device)",
    code: `# 错误
a = tvm.nd.array(data, tvm.gpu())  # GPU
b = tvm.nd.array(data, tvm.cpu())  # CPU
f(a, b)  # 设备不匹配!

# 修复
b_gpu = b.copyto(tvm.gpu())
f(a, b_gpu)`,
  },
  {
    id: "compile",
    name: "编译错误",
    icon: "⚙️",
    severity: "error",
    msg: "TIR compilation failed: unsupported op 'custom_op'",
    cause: "Target 不支持某个算子或调度",
    fix: "注册自定义算子或降级到支持的算子\ntvm.target.register_func()",
    code: `# 错误: CUDA 后端不支持该算子
target = "cuda"
lib = tvm.build(mod, target)  # 失败!

# 修复: 注册 fallback
@tvm.target.register_func
def custom_op_lower(expr, target):
    # 提供降级实现
    return relay.nn.dense(expr)`,
  },
  {
    id: "runtime",
    name: "运行时错误",
    icon: "💥",
    severity: "error",
    msg: "Check failed: size == expected: buffer size mismatch",
    cause: "分配的 buffer 大小与实际使用不匹配",
    fix: "检查 tensor shape 和 buffer 分配\n确保 allocate 的大小足够",
    code: `# 错误: buffer 大小不足
A = tvm.nd.empty((64,), "float32")  # 64 元素
B = tvm.nd.empty((128,), "float32")  # 128 元素
f(A, B)  # 函数期望 A 有 128 元素

# 修复
A = tvm.nd.empty((128,), "float32")`,
  },
];

export function RuntimeErrorGuide() {
  const [active, setActive] = useState("shape");

  const error = errors.find((e) => e.id === active)!;

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">运行时错误指南</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">5 种常见运行时错误及修复方案</p>

      <div className="flex gap-2 mb-6 flex-wrap">
        {errors.map((e) => (
          <button
            key={e.id}
            onClick={() => setActive(e.id)}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg border-2 transition-all duration-300 ${
              active === e.id
                ? "border-red-400 bg-red-50 dark:bg-red-900/30"
                : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-indigo-300"
            }`}
          >
            <span>{e.icon}</span>
            <span className="text-xs font-bold text-slate-700 dark:text-slate-200">{e.name}</span>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-5">
        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <div className="bg-red-50 dark:bg-red-900/30 rounded-lg p-3 mb-3 font-mono text-xs text-red-600 dark:text-red-400">
            ❌ {error.msg}
          </div>
          <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-2">原因分析</h4>
          <p className="text-sm text-slate-600 dark:text-slate-300 mb-3">{error.cause}</p>
          <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-2">修复方案</h4>
          <p className="text-sm text-slate-600 dark:text-slate-300 whitespace-pre-wrap">{error.fix}</p>
        </div>

        <div className="bg-slate-900 dark:bg-slate-950 rounded-xl p-4 font-mono text-xs leading-relaxed text-green-400 overflow-x-auto">
          <pre>{error.code}</pre>
        </div>
      </div>

      <div className="bg-indigo-50 dark:bg-indigo-900/30 rounded-lg p-3 flex items-center gap-2">
        <span className="text-lg">💡</span>
        <p className="text-xs text-slate-600 dark:text-slate-300">
          使用 <code className="bg-indigo-200 dark:bg-indigo-800 px-1 rounded">tvm.transform.PassContext(trace=True)</code> 可以在编译时捕获大部分错误，避免运行时才发现。
        </p>
      </div>
    </div>
  );
}
