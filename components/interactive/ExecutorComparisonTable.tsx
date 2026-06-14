"use client";

import { useState } from "react";

const executors = [
  {
    name: "GraphExecutor",
    color: "from-violet-500 to-purple-600",
    pros: ["简单高效", "内存占用低", "适合固定 shape"],
    cons: ["不支持动态 shape", "不支持控制流"],
    useCase: "CNN 推理、固定输入 shape",
    runtime: "按拓扑序逐节点执行",
  },
  {
    name: "Virtual Machine",
    color: "from-indigo-500 to-blue-600",
    pros: ["支持动态 shape", "支持控制流", "更灵活"],
    cons: ["运行时开销较大", "内存管理复杂"],
    useCase: "NLP 模型、动态图",
    runtime: "字节码解释执行",
  },
  {
    name: "AOT Executor",
    color: "from-blue-500 to-cyan-600",
    pros: ["无运行时依赖", "最小二进制", "嵌入式友好"],
    cons: ["不支持动态 shape", "调试困难"],
    useCase: "MCU 部署、边缘设备",
    runtime: "生成 C 代码直接编译",
  },
];

export function ExecutorComparisonTable() {
  const [selected, setSelected] = useState<number | null>(null);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
        执行器对比
      </h3>

      <div className="grid grid-cols-3 gap-4 mb-6">
        {executors.map((ex, i) => (
          <button
            key={ex.name}
            onClick={() => setSelected(selected === i ? null : i)}
            className={`text-center p-4 rounded-xl transition-all ${
              selected === i
                ? "ring-2 ring-indigo-400 ring-offset-2 dark:ring-offset-slate-900 scale-105"
                : "hover:scale-105"
            }`}
          >
            <div className={`bg-gradient-to-r ${ex.color} text-white px-4 py-3 rounded-xl shadow-lg`}>
              <div className="font-bold">{ex.name}</div>
            </div>
          </button>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b-2 border-slate-300 dark:border-slate-600">
              <th className="text-left p-3 text-slate-700 dark:text-slate-300 font-bold">特性</th>
              {executors.map((ex) => (
                <th key={ex.name} className="p-3 text-center">
                  <span className={`px-2 py-1 rounded text-xs font-bold text-white bg-gradient-to-r ${ex.color}`}>
                    {ex.name}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-slate-200 dark:border-slate-700">
              <td className="p-3 font-bold text-slate-700 dark:text-slate-300">动态 Shape</td>
              <td className="p-3 text-center text-red-500">✗</td>
              <td className="p-3 text-center text-green-500">✓</td>
              <td className="p-3 text-center text-red-500">✗</td>
            </tr>
            <tr className="border-b border-slate-200 dark:border-slate-700">
              <td className="p-3 font-bold text-slate-700 dark:text-slate-300">控制流</td>
              <td className="p-3 text-center text-red-500">✗</td>
              <td className="p-3 text-center text-green-500">✓</td>
              <td className="p-3 text-center text-green-500">✓</td>
            </tr>
            <tr className="border-b border-slate-200 dark:border-slate-700">
              <td className="p-3 font-bold text-slate-700 dark:text-slate-300">运行时开销</td>
              <td className="p-3 text-center text-green-500">低</td>
              <td className="p-3 text-center text-amber-500">中</td>
              <td className="p-3 text-center text-green-500">最低</td>
            </tr>
            <tr className="border-b border-slate-200 dark:border-slate-700">
              <td className="p-3 font-bold text-slate-700 dark:text-slate-300">执行方式</td>
              <td className="p-3 text-center text-slate-600 dark:text-slate-400">拓扑序</td>
              <td className="p-3 text-center text-slate-600 dark:text-slate-400">字节码</td>
              <td className="p-3 text-center text-slate-600 dark:text-slate-400">C 代码</td>
            </tr>
          </tbody>
        </table>
      </div>

      {selected !== null && (
        <div className="mt-4 p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
          <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-2">
            {executors[selected].name} 详情
          </h4>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs font-bold text-green-600 dark:text-green-400 mb-1">优点</p>
              <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                {executors[selected].pros.map((p) => <li key={p}>✓ {p}</li>)}
              </ul>
            </div>
            <div>
              <p className="text-xs font-bold text-red-600 dark:text-red-400 mb-1">缺点</p>
              <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                {executors[selected].cons.map((c) => <li key={c}>✗ {c}</li>)}
              </ul>
            </div>
          </div>
          <p className="text-xs text-indigo-600 dark:text-indigo-400 mt-2">
            <strong>适用场景:</strong> {executors[selected].useCase}
          </p>
        </div>
      )}
    </div>
  );
}
