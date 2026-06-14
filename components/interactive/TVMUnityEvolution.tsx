"use client";

import { useState } from "react";

const versions = [
  {
    id: "v0.8",
    version: "v0.8",
    title: "传统架构",
    year: "2020",
    components: [
      { name: "Relay", desc: "高层 IR，图级别优化", color: "from-blue-500 to-indigo-500" },
      { name: "TE", desc: "张量表达式，调度描述", color: "from-indigo-500 to-purple-500" },
      { name: "TIR", desc: "低层 IR，循环表示", color: "from-purple-500 to-pink-500" },
      { name: "Codegen", desc: "代码生成 (LLVM/CUDA)", color: "from-pink-500 to-rose-500" },
    ],
    issues: ["Pass 间耦合紧密", "前端/后端分离不彻底", "扩展新硬件困难"],
    code: `# v0.8: 独立的 Relay 和 TE
relay_mod = relay.from_pyfunc(model)
te_sch = topi.nn.conv2d_schedule(target)
# 手动连接两个层`,
  },
  {
    id: "v0.12",
    version: "v0.12",
    title: "统一 IR",
    year: "2022",
    components: [
      { name: "Relax", desc: "新高层 IR，支持动态 shape", color: "from-blue-500 to-indigo-500" },
      { name: "TIR (统一)", desc: "统一的低层 IR", color: "from-indigo-500 to-purple-500" },
      { name: "MetaSchedule", desc: "自动搜索调度", color: "from-purple-500 to-pink-500" },
      { name: "Codegen", desc: "统一代码生成", color: "from-pink-500 to-rose-500" },
    ],
    issues: ["部分兼容旧 API", "迁移成本中等"],
    code: `# v0.12: MetaSchedule 自动搜索
from tvm import meta_schedule
sch = meta_schedule.tune(mod, target)
# 自动找到最优 schedule`,
  },
  {
    id: "v0.14",
    version: "v0.14",
    title: "Unity 架构",
    year: "2024",
    components: [
      { name: "Relax (成熟)", desc: "完整的高层 IR 框架", color: "from-blue-500 to-indigo-500" },
      { name: "TIR (统一)", desc: "统一 IR 栈", color: "from-indigo-500 to-purple-500" },
      { name: "Unity Pass", desc: "端到端优化管线", color: "from-purple-500 to-pink-500" },
      { name: "统一 Codegen", desc: "支持所有后端", color: "from-pink-500 to-rose-500" },
    ],
    issues: ["完全统一的 IR 栈", "端到端优化", "新硬件接入标准化"],
    code: `# v0.14: Unity 统一架构
from tvm import relax
mod = relax.frontend.from_onnx(model)
mod = relax.get_pipeline("default")(mod)
lib = tvm.build(mod, target)`,
  },
];

export function TVMUnityEvolution() {
  const [active, setActive] = useState("v0.14");

  const ver = versions.find((v) => v.id === active)!;

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">TVM Unity 演进</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">v0.8 → v0.12 → v0.14 的架构变化</p>

      <div className="flex items-center justify-center gap-2 mb-6">
        {versions.map((v, i) => (
          <div key={v.id} className="flex items-center">
            <button
              onClick={() => setActive(v.id)}
              className={`px-5 py-3 rounded-xl border-2 transition-all duration-300 ${
                active === v.id
                  ? "border-indigo-500 bg-indigo-100 dark:bg-indigo-900/40 shadow-lg"
                  : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-indigo-300"
              }`}
            >
              <div className="text-sm font-bold text-slate-700 dark:text-slate-200">{v.version}</div>
              <div className="text-xs text-slate-500 dark:text-slate-400">{v.title}</div>
              <div className="text-[10px] text-slate-400 dark:text-slate-500">{v.year}</div>
            </button>
            {i < versions.length - 1 && <span className="mx-2 text-indigo-400 text-lg">→</span>}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-3">
          <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-3">架构组件</h4>
            <div className="space-y-2">
              {ver.components.map((c, i) => (
                <div key={i} className="flex items-center gap-3">
                  <div className={`w-2 h-8 rounded bg-gradient-to-b ${c.color}`} />
                  <div>
                    <div className="text-xs font-bold text-slate-700 dark:text-slate-200">{c.name}</div>
                    <div className="text-[10px] text-slate-500 dark:text-slate-400">{c.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-2">特性/改进</h4>
            <ul className="space-y-1">
              {ver.issues.map((iss, i) => (
                <li key={i} className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  {iss}
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="bg-slate-900 dark:bg-slate-950 rounded-xl p-4 font-mono text-xs leading-relaxed text-green-400 overflow-x-auto">
          <div className="text-slate-500 mb-2"># {ver.version} API 示例</div>
          <pre>{ver.code}</pre>
        </div>
      </div>
    </div>
  );
}
