"use client";

import { useState } from "react";

const parts = [
  {
    id: "cache",
    name: "Compilation Cache",
    icon: "🗂️",
    color: "from-blue-500 to-indigo-500",
    desc: "缓存已编译的子图，避免重复编译",
    details: [
      "以 (subgraph_hash, target) 为 key",
      "命中缓存直接返回编译结果",
      "支持磁盘持久化",
      "减少热启动时间 50%+",
    ],
    code: `# 缓存机制
cache = CompilationCache()
key = hash(subgraph) + target
if cache.contains(key):
    return cache.lookup(key)  # 命中
else:
    lib = compile(subgraph)
    cache.insert(key, lib)
    return lib`,
  },
  {
    id: "workspace",
    name: "Workspace",
    icon: "📁",
    color: "from-indigo-500 to-purple-500",
    desc: "管理编译中间产物和工作目录",
    details: [
      "存储 TIR、TE 中间表示",
      "管理临时文件和日志",
      "支持并行编译隔离",
      "自动清理过期产物",
    ],
    code: `# 工作空间管理
workspace = Workspace("/tmp/tvm_ws/")
workspace.save_ir("step3_tir.txt", tir_mod)
workspace.save_schedule("step3_sch.txt", sch)
# 并行编译每个子图在独立目录
sub_ws = workspace.create_sub("subgraph_0")`,
  },
  {
    id: "codegen",
    name: "Code Generator",
    icon: "⚡",
    color: "from-purple-500 to-pink-500",
    desc: "将 TIR 翻译为目标平台代码",
    details: [
      "LLVM IR 代码生成",
      "CUDA kernel 生成",
      "C 源码生成 (MCU)",
      "SPIR-V 生成 (Vulkan)",
    ],
    code: `# 代码生成
codegen = CodeGenFactory.create(target)
# TIR → LLVM IR
llvm_ir = codegen.emit(tir_mod)
# 链接为动态库
lib = codegen.link(llvm_ir)
# 输出: libmodel.so`,
  },
];

export function CompileEngineDiagram() {
  const [active, setActive] = useState("cache");

  const part = parts.find((p) => p.id === active)!;

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">编译引擎</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">Cache + Workspace + CodeGen 三大核心组件</p>

      <div className="flex items-center justify-center gap-4 mb-6">
        {parts.map((p) => (
          <button
            key={p.id}
            onClick={() => setActive(p.id)}
            className={`flex flex-col items-center px-6 py-4 rounded-xl border-2 transition-all duration-300 ${
              active === p.id
                ? "border-indigo-500 bg-indigo-100 dark:bg-indigo-900/40 shadow-lg scale-105"
                : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-indigo-300"
            }`}
          >
            <span className="text-3xl mb-2">{p.icon}</span>
            <span className="text-sm font-bold text-slate-700 dark:text-slate-200">{p.name}</span>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-2">{part.icon} {part.name}</h4>
          <p className="text-sm text-slate-500 dark:text-slate-400 mb-3">{part.desc}</p>
          <ul className="space-y-1.5">
            {part.details.map((d, i) => (
              <li key={i} className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                {d}
              </li>
            ))}
          </ul>
        </div>
        <div className="bg-slate-900 dark:bg-slate-950 rounded-xl p-4 font-mono text-xs leading-relaxed text-green-400 overflow-x-auto">
          <pre>{part.code}</pre>
        </div>
      </div>

      <div className="mt-5 bg-white/60 dark:bg-slate-800/60 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
        <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-2">引擎协作流程</h4>
        <div className="flex items-center justify-center gap-2">
          {["IR 输入", "Cache 查询", "Workspace 准备", "CodeGen 生成", "输出库"].map((step, i) => (
            <div key={i} className="flex items-center">
              <span className="px-3 py-1.5 rounded-lg text-xs font-medium bg-gradient-to-r from-indigo-500 to-purple-500 text-white">
                {step}
              </span>
              {i < 4 && <span className="mx-1 text-indigo-400">→</span>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
