"use client";

import { useState } from "react";

const pipelineStages = [
  { id: "relay", name: "Relay IR", desc: "高层计算图表示", color: "from-blue-500 to-indigo-500" },
  { id: "relay_opt", name: "Relay 优化", desc: "算子融合、常量折叠", color: "from-indigo-500 to-purple-500" },
  { id: "engine", name: "编译引擎", desc: "Relay→TE 转换和编译", color: "from-purple-500 to-pink-500", highlight: true },
  { id: "te", name: "TE Schedule", desc: "张量表达式调度", color: "from-pink-500 to-rose-500" },
  { id: "tir", name: "TIR", desc: "底层循环 IR", color: "from-rose-500 to-red-500" },
  { id: "codegen", name: "Codegen", desc: "目标代码生成", color: "from-red-500 to-orange-500" },
];

export function CompileEnginePosition() {
  const [selected, setSelected] = useState("engine");

  const stage = pipelineStages.find((s) => s.id === selected)!;

  const stageDetails: Record<string, string> = {
    relay: "Relay IR 是 TVM 的高层计算图表示\n包含算子节点和数据流边\n负责图级别的优化 (融合、折叠等)",
    relay_opt: "应用 Relay Pass 进行图优化\n- Operator Fusion\n- Constant Folding\n- Dead Code Elimination\n- Layout Transform",
    engine: "编译引擎是核心枢纽\n- 接收优化后的 Relay 子图\n- 调用 TOPI/TE 生成 schedule\n- 管理编译缓存\n- 协调 Workspace 和 CodeGen",
    te: "张量表达式 (TE) 调度层\n描述循环结构、tiling、向量化\nAutoTVM/MetaSchedule 在此搜索",
    tir: "TIR 是最终的低层表示\n精确的循环、内存访问模式\n直接映射到硬件指令",
    codegen: "代码生成层\nTIR → LLVM IR / CUDA / C\n链接为目标平台的可执行库",
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">编译引擎位置</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">编译引擎在 Relay→TE 管线中的位置</p>

      <div className="flex items-center justify-center gap-1 mb-6 flex-wrap">
        {pipelineStages.map((s, i) => (
          <div key={s.id} className="flex items-center">
            <button
              onClick={() => setSelected(s.id)}
              className={`px-4 py-3 rounded-xl border-2 transition-all duration-300 min-w-[90px] ${
                selected === s.id
                  ? s.highlight
                    ? "border-pink-500 bg-pink-100 dark:bg-pink-900/40 shadow-lg scale-110 ring-2 ring-pink-300 dark:ring-pink-600"
                    : "border-indigo-500 bg-indigo-100 dark:bg-indigo-900/40 shadow-lg scale-105"
                  : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-indigo-300"
              }`}
            >
              <div className="text-xs font-bold text-slate-700 dark:text-slate-200 text-center">{s.name}</div>
              <div className="text-[10px] text-slate-500 dark:text-slate-400 text-center">{s.desc}</div>
              {s.highlight && (
                <div className="text-[10px] text-pink-500 font-bold text-center mt-0.5">★ 核心</div>
              )}
            </button>
            {i < pipelineStages.length - 1 && (
              <span className="mx-0.5 text-indigo-400 text-sm">→</span>
            )}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-2">{stage.name}</h4>
          <pre className="text-xs text-slate-600 dark:text-slate-300 whitespace-pre-wrap font-mono leading-relaxed">
            {stageDetails[selected]}
          </pre>
        </div>
        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-3">引擎职责</h4>
          <div className="space-y-2">
            {[
              { icon: "🔄", title: "Relay→TE 转换", desc: "将图级算子降级为 TE 表达式" },
              { icon: "🔍", title: "Schedule 生成", desc: "调用 TOPI 获取默认或调优后的 schedule" },
              { icon: "🗂️", title: "缓存管理", desc: "避免相同子图的重复编译" },
              { icon: "⚙️", title: "调度 TIR 优化", desc: "应用 TIR 级别的 pass 优化" },
            ].map((r, i) => (
              <div key={i} className="flex items-center gap-3">
                <span>{r.icon}</span>
                <div>
                  <div className="text-xs font-bold text-slate-700 dark:text-slate-200">{r.title}</div>
                  <div className="text-[10px] text-slate-500 dark:text-slate-400">{r.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
