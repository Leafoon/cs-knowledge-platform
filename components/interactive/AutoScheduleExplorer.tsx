"use client";

import { useState } from "react";

type Axis = "tile" | "reorder" | "vectorize";

interface ScheduleOption {
    id: string;
    name: string;
    desc: string;
    impact: string;
    example: string;
}

const axes: Record<Axis, { title: string; icon: string; options: ScheduleOption[] }> = {
    tile: {
        title: "分块 (Tiling)",
        icon: "🔲",
        options: [
            { id: "t1", name: "32×32 分块", desc: "将循环拆分为 32×32 的小块", impact: "提升 L1 缓存命中率", example: "T.tile(xi, 32, yi, 32)" },
            { id: "t2", name: "64×64 分块", desc: "更大的分块适合 L2 缓存", impact: "减少循环开销", example: "T.tile(xi, 64, yi, 64)" },
            { id: "t3", name: "多级分块", desc: "L1: 16×16, L2: 64×64", impact: "适配多级缓存层次", example: "T.tile(..., 64).tile(..., 16)" },
        ],
    },
    reorder: {
        title: "重排 (Reordering)",
        icon: "🔄",
        options: [
            { id: "r1", name: "行优先遍历", desc: "按行顺序访问内存", impact: "空间局部性好", example: "T.reorder(i, j)" },
            { id: "r2", name: "列优先遍历", desc: "按列顺序访问内存", impact: "适合列主序存储", example: "T.reorder(j, i)" },
            { id: "r3", name: "Z字形遍历", desc: "空间填充曲线方式", impact: "2D 局部性优化", example: "T.reorder(..., z_order)" },
        ],
    },
    vectorize: {
        title: "向量化 (Vectorize)",
        icon: "📊",
        options: [
            { id: "v1", name: "SSE/128-bit", desc: "4个float32并行计算", impact: "4x 理论加速", example: "T.vectorize(j, 4)" },
            { id: "v2", name: "AVX/256-bit", desc: "8个float32并行计算", impact: "8x 理论加速", example: "T.vectorize(j, 8)" },
            { id: "v3", name: "AVX-512/512-bit", desc: "16个float32并行计算", impact: "16x 理论加速", example: "T.vectorize(j, 16)" },
        ],
    },
};

export function AutoScheduleExplorer() {
    const [activeAxis, setActiveAxis] = useState<Axis>("tile");
    const [selected, setSelected] = useState<Record<Axis, string>>({ tile: "t1", reorder: "r1", vectorize: "v1" });
    const axis = axes[activeAxis];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                自动调度搜索空间
            </h2>
            <p className="text-center text-sm text-slate-500 dark:text-slate-400 mb-6">
                AutoSchedule 在三个维度上搜索最优调度策略
            </p>

            {/* Axis selector */}
            <div className="flex justify-center gap-4 mb-6 flex-wrap">
                {(Object.keys(axes) as Axis[]).map((key) => (
                    <button
                        key={key}
                        onClick={() => setActiveAxis(key)}
                        className={`flex items-center gap-2 px-5 py-3 rounded-xl text-sm font-bold transition-all duration-300 ${
                            activeAxis === key
                                ? "bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-lg scale-105"
                                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:border-indigo-300"
                        }`}
                    >
                        <span className="text-lg">{axes[key].icon}</span>
                        {axes[key].title}
                    </button>
                ))}
            </div>

            {/* Options grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                {axis.options.map((opt) => (
                    <button
                        key={opt.id}
                        onClick={() => setSelected((prev) => ({ ...prev, [activeAxis]: opt.id }))}
                        className={`text-left p-4 rounded-xl border-2 transition-all duration-300 ${
                            selected[activeAxis] === opt.id
                                ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-900/30 shadow-md"
                                : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-indigo-300"
                        }`}
                    >
                        <div className="font-bold text-sm text-slate-700 dark:text-slate-200 mb-1">{opt.name}</div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-2">{opt.desc}</div>
                        <div className="text-xs text-indigo-600 dark:text-indigo-400 font-medium">💡 {opt.impact}</div>
                        <div className="mt-2 bg-slate-100 dark:bg-slate-900 rounded px-2 py-1 font-mono text-xs text-slate-600 dark:text-slate-400">
                            {opt.example}
                        </div>
                    </button>
                ))}
            </div>

            {/* Search space visualization */}
            <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
                <h3 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-3">搜索空间组合</h3>
                <div className="flex items-center justify-center gap-3 flex-wrap">
                    {(Object.keys(axes) as Axis[]).map((key) => {
                        const opt = axes[key].options.find((o) => o.id === selected[key]);
                        return (
                            <div key={key} className="flex items-center gap-2">
                                <span className="text-lg">{axes[key].icon}</span>
                                <span className="px-3 py-1 rounded-full bg-indigo-100 dark:bg-indigo-900/40 text-xs font-medium text-indigo-700 dark:text-indigo-300">
                                    {opt?.name}
                                </span>
                            </div>
                        );
                    })}
                    <span className="text-slate-400 mx-2">→</span>
                    <span className="px-3 py-1 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 text-white text-xs font-bold">
                        调度方案 #{Object.values(selected).join("-")}
                    </span>
                </div>
                <p className="text-center text-xs text-slate-400 mt-3">
                    总搜索空间: {axes.tile.options.length} × {axes.reorder.options.length} × {axes.vectorize.options.length} = {axes.tile.options.length * axes.reorder.options.length * axes.vectorize.options.length} 种组合
                </p>
            </div>
        </div>
    );
}
