"use client";

import React, { useState } from "react";

interface Stage {
  id: string;
  label: string;
  detail: string;
}

const stages: Stage[] = [
  { id: "input", label: "输入数据", detail: "加载输入张量，验证 shape 和 dtype" },
  { id: "load", label: "加载模型", detail: "从 .so / .json 加载编译产物，构建执行图" },
  { id: "pre", label: "Pre-Exec Hook", detail: "执行前回调：记录时间戳、dump 输入张量" },
  { id: "kernel", label: "算子执行", detail: "逐节点执行 TIR kernel，中间结果写入临时 buffer" },
  { id: "dump", label: "中间输出 Dump", detail: "将每层输出保存到 debug_output/ 目录" },
  { id: "post", label: "Post-Exec Hook", detail: "执行后回调：对比输出、记录内存使用" },
  { id: "verify", label: "验证输出", detail: "与 golden reference 对比，输出 diff 报告" },
  { id: "output", label: "最终输出", detail: "返回推理结果" },
];

export function DebugExecutorDiagram() {
  const [active, setActive] = useState<string | null>(null);

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-indigo-950/80 via-purple-950/60 to-blue-950/80 backdrop-blur rounded-xl border border-indigo-700/30 shadow-lg my-6">
      <h3 className="text-xl font-bold mb-2 text-indigo-200">DebugExecutor 架构图</h3>
      <p className="text-sm text-indigo-300/70 mb-4">带中间输出 dump 的调试执行器，在每个算子节点插入检查点。</p>

      <div className="flex flex-col items-center gap-1">
        {stages.map((s, i) => (
          <React.Fragment key={s.id}>
            <button
              onClick={() => setActive(active === s.id ? null : s.id)}
              className={`w-full max-w-md px-4 py-2.5 rounded-lg border text-left transition-all ${
                active === s.id
                  ? "bg-indigo-600/60 border-indigo-400 shadow-md shadow-indigo-500/20"
                  : s.id === "dump"
                  ? "bg-purple-900/40 border-purple-600/40 hover:bg-purple-800/40"
                  : "bg-indigo-900/30 border-indigo-700/30 hover:bg-indigo-800/40"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className={`text-sm font-semibold ${active === s.id ? "text-white" : "text-indigo-200"}`}>
                  {s.label}
                </span>
                {s.id === "dump" && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-600/40 text-purple-200">Debug</span>
                )}
              </div>
              {active === s.id && (
                <div className="text-xs text-indigo-200/70 mt-1">{s.detail}</div>
              )}
            </button>
            {i < stages.length - 1 && (
              <div className="flex flex-col items-center">
                <div className="w-0.5 h-3 bg-indigo-600/50" />
                <div className="text-indigo-500 text-xs">▼</div>
              </div>
            )}
          </React.Fragment>
        ))}
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3 text-xs">
        <div className="bg-indigo-900/30 rounded-lg p-3 border border-indigo-800/30">
          <div className="text-indigo-400 font-semibold mb-1">📁 Dump 输出结构</div>
          <pre className="text-indigo-300/70 font-mono text-[11px]">
{`debug_output/
├── layer_0_input.bin
├── layer_0_output.bin
├── layer_1_output.bin
└── ...`}
          </pre>
        </div>
        <div className="bg-indigo-900/30 rounded-lg p-3 border border-indigo-800/30">
          <div className="text-indigo-400 font-semibold mb-1">⚙️ 启用方式</div>
          <pre className="text-indigo-300/70 font-mono text-[11px]">
{`from tvm.contrib.debug_executor\
import create

lib = tvm.runtime.load_module(
    "model.so"
)
debug_exec = create(
    graph, lib, ctx
)`}
          </pre>
        </div>
        <div className="bg-indigo-900/30 rounded-lg p-3 border border-indigo-800/30">
          <div className="text-indigo-400 font-semibold mb-1">⚡ vs 普通 Executor</div>
          <div className="text-indigo-300/70 space-y-1">
            <div>普通: <span className="text-green-400">快</span>，无中间检查</div>
            <div>Debug: <span className="text-yellow-400">慢 2-5x</span>，完整 dump</div>
            <div>Debug: 内存占用 <span className="text-red-400">+50-200%</span></div>
          </div>
        </div>
      </div>
    </div>
  );
};


