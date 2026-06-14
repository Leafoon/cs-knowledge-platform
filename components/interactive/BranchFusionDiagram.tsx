"use client";

import { useState } from "react";

type FusionType = "horizontal" | "vertical" | "chunk";

const fusionTypes: Record<FusionType, { title: string; desc: string; before: string[]; after: string[]; benefit: string; color: string }> = {
    horizontal: {
        title: "水平融合",
        desc: "将多个独立的、输入相同的小算子合并为一个大算子",
        before: ["输入X → Conv1 → 输出A", "输入X → Conv2 → 输出B", "输入X → Conv3 → 输出C"],
        after: ["输入X → FusedConv(A,B,C) → 三路输出"],
        benefit: "减少内核启动开销和内存读取次数",
        color: "from-blue-500 to-indigo-500",
    },
    vertical: {
        title: "垂直融合",
        desc: "将串联的逐元素算子融合为一个算子，消除中间张量",
        before: ["输入X → Relu → 中间1", "中间1 → Add → 中间2", "中间2 → Sigmoid → 输出"],
        after: ["输入X → Fused(Relu+Add+Sigmoid) → 输出"],
        benefit: "消除中间张量的内存分配和读写",
        color: "from-purple-500 to-indigo-500",
    },
    chunk: {
        title: "Chunk融合",
        desc: "将大张量计算拆分为多个chunk并行执行，融合相邻chunk的计算",
        before: ["Chunk1: [0,32) → Op1 → Op2", "Chunk2: [32,64) → Op1 → Op2", "Chunk3: [64,96) → Op1 → Op2"],
        after: ["Chunk1-3并行: 每个chunk内 Op1+Op2 融合"],
        benefit: "提升并行度，减少中间缓冲区大小",
        color: "from-amber-500 to-orange-500",
    },
};

export function BranchFusionDiagram() {
    const [active, setActive] = useState<FusionType>("horizontal");
    const info = fusionTypes[active];

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">分支融合策略</h3>

            <div className="flex justify-center gap-3 mb-6 flex-wrap">
                {(Object.keys(fusionTypes) as FusionType[]).map((key) => (
                    <button
                        key={key}
                        onClick={() => setActive(key)}
                        className={`px-5 py-2 rounded-full text-sm font-bold transition-all ${
                            active === key
                                ? `bg-gradient-to-r ${fusionTypes[key].color} text-white shadow-lg`
                                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:border-indigo-300"
                        }`}
                    >
                        {fusionTypes[key].title}
                    </button>
                ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                    <h4 className="text-sm font-bold text-red-600 dark:text-red-400 mb-3">❌ 融合前</h4>
                    <div className="space-y-2">
                        {info.before.map((item, i) => (
                            <div key={i} className="flex items-center gap-1 flex-wrap">
                                {item.split(" → ").map((node, j, arr) => (
                                    <span key={j} className="flex items-center gap-1">
                                        <span className="px-2 py-1 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-xs font-mono text-red-700 dark:text-red-300">
                                            {node}
                                        </span>
                                        {j < arr.length - 1 && <span className="text-red-400">→</span>}
                                    </span>
                                ))}
                            </div>
                        ))}
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                    <h4 className="text-sm font-bold text-emerald-600 dark:text-emerald-400 mb-3">✅ 融合后</h4>
                    <div className="space-y-2">
                        {info.after.map((item, i) => (
                            <div key={i} className="flex items-center gap-1 flex-wrap">
                                {item.split(" → ").map((node, j, arr) => (
                                    <span key={j} className="flex items-center gap-1">
                                        <span className={`px-2 py-1 rounded-lg text-xs font-mono ${
                                            node.includes("Fused") || node.includes("融合")
                                                ? "bg-emerald-100 dark:bg-emerald-900/30 border border-emerald-300 dark:border-emerald-700 text-emerald-700 dark:text-emerald-300 font-bold"
                                                : "bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-300"
                                        }`}>
                                            {node}
                                        </span>
                                        {j < arr.length - 1 && <span className="text-emerald-400">→</span>}
                                    </span>
                                ))}
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className={`bg-gradient-to-r ${info.color} rounded-xl p-4 text-white text-sm`}>
                <span className="font-bold">💡 优化收益：</span>{info.benefit}
            </div>

            <div className="mt-3 bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg">
                <p className="text-sm text-slate-600 dark:text-slate-300">{info.desc}</p>
            </div>
        </div>
    );
}
