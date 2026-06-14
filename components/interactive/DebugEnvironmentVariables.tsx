"use client";

import React, { useState } from "react";

interface EnvVar {
  name: string;
  values: string;
  description: string;
  example: string;
  category: string;
}

const envVars: EnvVar[] = [
  { name: "TVM_LOG_DEBUG", values: "0-4", description: "控制调试日志详细程度", example: "TVM_LOG_DEBUG=3 python model.py", category: "日志" },
  { name: "TVM_DUMP_LOWER_IR", values: "0|1", description: "dump 降低后的 TIR IR 到文件", example: "TVM_DUMP_LOWER_IR=1 python model.py", category: "IR" },
  { name: "TVM_TRACE_EXECUTION", values: "0|1", description: "跟踪执行路径并输出每个算子的调用栈", example: "TVM_TRACE_EXECUTION=1 ./run_model", category: "执行" },
  { name: "TVM_DEBUG_RUNTIME", values: "0|1", description: "启用运行时调试检查，如越界访问检测", example: "TVM_DEBUG_RUNTIME=1 python infer.py", category: "运行时" },
  { name: "TVM_GRAPH_DEBUG", values: "0|1", description: "输出计算图结构的调试信息", example: "TVM_GRAPH_DEBUG=1 python compile.py", category: "图" },
  { name: "TVM_OPENCL_GPU", values: "0|1", description: "强制使用 OpenCL GPU 后端", example: "TVM_OPENCL_GPU=1 python test_opencl.py", category: "后端" },
  { name: "TVM_CUDA_USE_NVRTC", values: "0|1", description: "使用 NVRTC 替代 NVCC 编译 CUDA 内核", example: "TVM_CUDA_USE_NVRTC=1 python cuda_test.py", category: "后端" },
  { name: "TVM_AUTOTVM_USE_MEASURE", values: "0|1", description: "AutoTVM 测量模式开关", example: "TVM_AUTOTVM_USE_MEASURE=1 python tune.py", category: "调优" },
];

const categories = [...new Set(envVars.map((e) => e.category))];

export function DebugEnvironmentVariables() {
  const [filter, setFilter] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);

  const filtered = envVars.filter(
    (e) =>
      (!filter || e.category === filter) &&
      (!search || e.name.toLowerCase().includes(search.toLowerCase()))
  );

  const copy = (text: string, idx: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 1500);
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-indigo-950/80 via-purple-950/60 to-blue-950/80 backdrop-blur rounded-xl border border-indigo-700/30 shadow-lg my-6">
      <h3 className="text-xl font-bold mb-2 text-indigo-200">TVM 调试环境变量</h3>
      <p className="text-sm text-indigo-300/70 mb-4">配置环境变量来控制 TVM 编译器的调试行为与输出。</p>

      <div className="flex flex-wrap gap-2 mb-3">
        <button
          onClick={() => setFilter(null)}
          className={`px-2.5 py-1 text-xs rounded-lg border ${!filter ? "bg-indigo-600 text-white border-indigo-400" : "bg-indigo-900/40 text-indigo-300 border-indigo-700/50 hover:bg-indigo-800/60"}`}
        >
          全部
        </button>
        {categories.map((c) => (
          <button
            key={c}
            onClick={() => setFilter(c)}
            className={`px-2.5 py-1 text-xs rounded-lg border ${filter === c ? "bg-indigo-600 text-white border-indigo-400" : "bg-indigo-900/40 text-indigo-300 border-indigo-700/50 hover:bg-indigo-800/60"}`}
          >
            {c}
          </button>
        ))}
        <input
          type="text"
          placeholder="搜索变量名..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="ml-auto px-3 py-1 text-xs rounded-lg bg-indigo-950/60 border border-indigo-700/40 text-indigo-200 placeholder-indigo-500 focus:outline-none focus:border-indigo-500"
        />
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-indigo-700/40">
              <th className="text-left py-2 px-3 text-indigo-400 font-medium">变量名</th>
              <th className="text-left py-2 px-3 text-indigo-400 font-medium">可选值</th>
              <th className="text-left py-2 px-3 text-indigo-400 font-medium">说明</th>
              <th className="text-left py-2 px-3 text-indigo-400 font-medium">示例</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((e, i) => (
              <tr key={i} className="border-b border-indigo-800/20 hover:bg-indigo-900/20 transition-colors">
                <td className="py-2 px-3 font-mono text-cyan-300 text-xs">{e.name}</td>
                <td className="py-2 px-3 text-indigo-300">{e.values}</td>
                <td className="py-2 px-3 text-indigo-200/70">{e.description}</td>
                <td className="py-2 px-3">
                  <div className="flex items-center gap-1">
                    <code className="text-[11px] text-purple-300 bg-indigo-950/60 px-1.5 py-0.5 rounded">{e.example}</code>
                    <button
                      onClick={() => copy(e.example, i)}
                      className="text-indigo-500 hover:text-indigo-300 text-xs"
                    >
                      {copiedIdx === i ? "✓" : "📋"}
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};


