"use client";

import { useState } from "react";

const layers = [
    {
        name: "算子级优化",
        color: "from-indigo-500 to-blue-500",
        techniques: [
            { name: "循环分块", desc: "提升缓存利用率", impact: "高" },
            { name: "向量化", desc: "利用SIMD指令", impact: "高" },
            { name: "循环展开", desc: "减少循环开销", impact: "中" },
            { name: "内存预取", desc: "隐藏内存延迟", impact: "中" },
        ],
    },
    {
        name: "图级优化",
        color: "from-purple-500 to-indigo-500",
        techniques: [
            { name: "算子融合", desc: "消除中间张量", impact: "高" },
            { name: "常量折叠", desc: "编译期计算", impact: "中" },
            { name: "死代码消除", desc: "移除无用计算", impact: "低" },
            { name: "布局优化", desc: "选择最优数据布局", impact: "高" },
        ],
    },
    {
        name: "调度级优化",
        color: "from-pink-500 to-purple-500",
        techniques: [
            { name: "自动调优", desc: "搜索最优调度参数", impact: "高" },
            { name: "内存作用域", desc: "共享内存/寄存器分配", impact: "高" },
            { name: "并行策略", desc: "线程/块绑定", impact: "高" },
            { name: "流水线", desc: "计算与访存重叠", impact: "中" },
        ],
    },
];

export function PerformanceOptimizationDiagram() {
    const [activeLayer, setActiveLayer] = useState(0);

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">性能优化层次</h3>

            <div className="grid grid-cols-3 gap-3 mb-6">
                {layers.map((layer, i) => (
                    <button
                        key={i}
                        onClick={() => setActiveLayer(i)}
                        className={`p-4 rounded-xl text-center transition-all ${
                            activeLayer === i
                                ? `bg-gradient-to-r ${layer.color} text-white shadow-lg`
                                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:border-indigo-300"
                        }`}
                    >
                        <div className="text-lg font-bold">{layer.name}</div>
                        <div className={`text-xs mt-1 ${activeLayer === i ? "text-white/80" : "text-slate-500"}`}>
                            {layer.techniques.length} 项技术
                        </div>
                    </button>
                ))}
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {layers[activeLayer].techniques.map((tech, i) => (
                        <div key={i} className="flex items-start gap-3 p-3 bg-slate-50 dark:bg-slate-900/50 rounded-lg">
                            <div className={`w-8 h-8 rounded-lg bg-gradient-to-r ${layers[activeLayer].color} flex items-center justify-center text-white text-xs font-bold shrink-0`}>
                                {i + 1}
                            </div>
                            <div>
                                <h5 className="text-sm font-bold text-slate-800 dark:text-slate-100">{tech.name}</h5>
                                <p className="text-xs text-slate-500 dark:text-slate-400">{tech.desc}</p>
                                <span className={`inline-block mt-1 text-xs font-medium ${
                                    tech.impact === "高" ? "text-emerald-600 dark:text-emerald-400" : tech.impact === "中" ? "text-amber-600 dark:text-amber-400" : "text-slate-500"
                                }`}>
                                    影响: {tech.impact}
                                </span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg">
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    💡 三个层次<strong className="text-indigo-600 dark:text-indigo-400">自底向上</strong>协同工作：算子级优化单个计算，图级优化全局结构，调度级优化硬件映射。
                </p>
            </div>
        </div>
    );
}
