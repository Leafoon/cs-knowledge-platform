"use client";

import { useState } from "react";

const files = [
  {
    name: "lib.so",
    icon: "⚙️",
    color: "from-violet-500 to-purple-600",
    desc: "编译后的算子库 (共享库)",
    content: "包含编译后的算子实现\n- CUDA kernel functions\n- LLVM compiled functions\n- 所有 PackedFunc 实现",
    size: "~2-50 MB",
  },
  {
    name: "mod.json",
    icon: "📋",
    color: "from-indigo-500 to-blue-600",
    desc: "模型结构描述 (Graph JSON)",
    content: "计算图的 JSON 表示\n- nodes: 算子节点列表\n- arg_nodes: 输入节点\n- heads: 输出节点\n- 属性信息",
    size: "~1-100 KB",
  },
  {
    name: "param.bin",
    icon: "📊",
    color: "from-blue-500 to-cyan-600",
    desc: "模型参数 (权重二进制)",
    content: "所有参数的二进制打包\n- NDArray 序列化格式\n- 每个 tensor: shape + dtype + data\n- 使用 DLTensor 序列化协议",
    size: "~10 MB - 1 GB",
  },
];

export function SerializationDiagram() {
  const [activeFile, setActiveFile] = useState<number | null>(null);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
        模型序列化: 三文件格式
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400 mb-6">
        TVM 编译输出由三个文件组成，分别存储代码、结构和参数
      </p>

      <div className="grid grid-cols-3 gap-4 mb-6">
        {files.map((f, i) => (
          <button
            key={f.name}
            onClick={() => setActiveFile(activeFile === i ? null : i)}
            className={`text-center transition-all ${
              activeFile === i ? "scale-105" : "hover:scale-105"
            }`}
          >
            <div className={`bg-gradient-to-br ${f.color} rounded-xl p-5 text-white shadow-lg`}>
              <div className="text-3xl mb-2">{f.icon}</div>
              <div className="font-bold text-lg">{f.name}</div>
              <div className="text-xs opacity-70 mt-1">{f.size}</div>
            </div>
          </button>
        ))}
      </div>

      {activeFile !== null && (
        <div className="bg-white dark:bg-slate-800 rounded-xl p-5 border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-3 mb-3">
            <span className="text-2xl">{files[activeFile].icon}</span>
            <div>
              <h4 className="font-bold text-slate-800 dark:text-slate-100">
                {files[activeFile].name}
              </h4>
              <p className="text-xs text-slate-500">{files[activeFile].desc}</p>
            </div>
            <span className="ml-auto px-2 py-1 bg-slate-100 dark:bg-slate-700 rounded text-xs text-slate-600 dark:text-slate-400">
              {files[activeFile].size}
            </span>
          </div>
          <pre className="bg-slate-900 text-green-400 p-3 rounded-lg text-xs font-mono whitespace-pre-wrap">
            {files[activeFile].content}
          </pre>
        </div>
      )}

      <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
        <pre className="text-xs text-amber-700 dark:text-amber-300 font-mono">
{`# 加载三文件
lib = tvm.runtime.load_module("lib.so")
graph = open("mod.json").read()
params = bytearray(open("param.bin").read())

# 创建执行器
module = graph_executor.create(graph, lib, device)
module.load_params(params)`}
        </pre>
      </div>
    </div>
  );
}
