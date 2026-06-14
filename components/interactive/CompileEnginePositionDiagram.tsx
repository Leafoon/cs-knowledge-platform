"use client";

import { useState } from "react";

const layers = [
    {
        id: "frontend",
        label: "前端 (Frontend)",
        icon: "📐",
        items: ["PyTorch / ONNX / TensorFlow", "Relay IR", "计算图优化"],
        color: "from-blue-500 to-indigo-500",
        bgColor: "bg-blue-50 dark:bg-blue-950/30",
    },
    {
        id: "engine",
        label: "编译引擎 (Compile Engine)",
        icon: "⚙️",
        items: ["TE 调度", "AutoTVM / AutoSchedule", "TIR Lowering"],
        color: "from-indigo-500 to-purple-500",
        bgColor: "bg-indigo-50 dark:bg-indigo-950/30",
        highlight: true,
    },
    {
        id: "codegen",
        label: "代码生成 (CodeGen)",
        icon: "🏗️",
        items: ["LLVM IR 生成", "CUDA 代码生成", "C 代码生成"],
        color: "from-purple-500 to-pink-500",
        bgColor: "bg-purple-50 dark:bg-purple-950/30",
    },
    {
        id: "runtime",
        label: "运行时 (Runtime)",
        icon: "🚀",
        items: ["设备驱动 (CPU/GPU)", "内存管理", "算子执行"],
        color: "from-pink-500 to-rose-500",
        bgColor: "bg-pink-50 dark:bg-pink-950/30",
    },
];

export function CompileEnginePositionDiagram() {
    const [activeLayer, setActiveLayer] = useState("engine");

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                编译引擎在 TVM 栈中的位置
            </h2>
            <p className="text-center text-sm text-slate-500 dark:text-slate-400 mb-6">
                TVM 技术栈: 编译引擎是连接前端与后端的核心层
            </p>

            {/* Stack visualization */}
            <div className="max-w-2xl mx-auto space-y-2 mb-6">
                {layers.map((layer, i) => (
                    <button
                        key={layer.id}
                        onClick={() => setActiveLayer(layer.id)}
                        className={`w-full text-left p-4 rounded-xl border-2 transition-all duration-300 ${
                            activeLayer === layer.id
                                ? "border-indigo-500 shadow-lg scale-[1.02]"
                                : "border-slate-200 dark:border-slate-700 hover:border-indigo-300"
                        } ${layer.bgColor} ${layer.highlight ? "ring-2 ring-indigo-300 dark:ring-indigo-700" : ""}`}
                    >
                        <div className="flex items-center gap-3">
                            <span className="text-2xl">{layer.icon}</span>
                            <div className="flex-1">
                                <div className="flex items-center gap-2">
                                    <span className="font-bold text-sm text-slate-700 dark:text-slate-200">{layer.label}</span>
                                    {layer.highlight && (
                                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-500 text-white">核心</span>
                                    )}
                                </div>
                                <div className="flex flex-wrap gap-2 mt-1.5">
                                    {layer.items.map((item, j) => (
                                        <span key={j} className="px-2 py-0.5 rounded bg-white/60 dark:bg-slate-800/60 text-xs text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
                                            {item}
                                        </span>
                                    ))}
                                </div>
                            </div>
                            <span className="text-slate-400 text-sm">
                                {activeLayer === layer.id ? "▼" : "▶"}
                            </span>
                        </div>
                    </button>
                ))}
            </div>

            {/* Layer detail */}
            {(() => {
                const layer = layers.find((l) => l.id === activeLayer)!;
                return (
                    <div className={`bg-gradient-to-r ${layer.color} rounded-xl p-5 text-white`}>
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-xl">{layer.icon}</span>
                            <h3 className="font-bold">{layer.label}</h3>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                            {layer.items.map((item, i) => (
                                <div key={i} className="bg-white/20 rounded-lg p-2 text-sm">
                                    {item}
                                </div>
                            ))}
                        </div>
                        {layer.highlight && (
                            <p className="mt-3 text-sm opacity-90">
                                编译引擎负责将高级 IR 转换为目标代码，协调调度优化、内存规划和代码生成。
                            </p>
                        )}
                    </div>
                );
            })()}

            {/* Side arrows */}
            <div className="mt-4 flex justify-center gap-6 text-xs text-slate-500 dark:text-slate-400">
                <div className="flex items-center gap-1">
                    <span>⬆️</span> 模型输入 (前端导入)
                </div>
                <div className="flex items-center gap-1">
                    <span>⬇️</span> 部署输出 (运行时执行)
                </div>
            </div>
        </div>
    );
}
