"use client";

import { useState } from "react";

const stages = [
    { id: "A", name: "Input", deps: [], color: "from-blue-500 to-indigo-500" },
    { id: "B", name: "Conv2d", deps: ["A"], color: "from-indigo-500 to-purple-500" },
    { id: "C", name: "BatchNorm", deps: ["B"], color: "from-purple-500 to-pink-500" },
    { id: "D", name: "Relu", deps: ["C"], color: "from-pink-500 to-red-500" },
    { id: "E", name: "Pooling", deps: ["D"], color: "from-amber-500 to-orange-500" },
    { id: "F", name: "Linear", deps: ["D"], color: "from-emerald-500 to-teal-500" },
    { id: "G", name: "Output", deps: ["E", "F"], color: "from-indigo-500 to-blue-500" },
];

export function DependencyAnalyzer() {
    const [selected, setSelected] = useState<string | null>(null);

    const getDependents = (id: string): string[] => {
        const direct = stages.filter((s) => s.deps.includes(id)).map((s) => s.id);
        return [...direct, ...direct.flatMap(getDependents)];
    };

    const getAncestors = (id: string): string[] => {
        const stage = stages.find((s) => s.id === id);
        if (!stage) return [];
        const direct = stage.deps;
        return [...direct, ...direct.flatMap(getAncestors)];
    };

    const highlightDeps = selected ? getAncestors(selected) : [];
    const highlightDown = selected ? getDependents(selected) : [];
    const allHighlight = selected ? [selected, ...highlightDeps, ...highlightDown] : [];

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">依赖分析 DAG</h3>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="flex flex-wrap gap-3 justify-center mb-4">
                    {stages.map((s) => (
                        <button
                            key={s.id}
                            onClick={() => setSelected(selected === s.id ? null : s.id)}
                            className={`px-4 py-3 rounded-xl text-sm font-bold transition-all ${
                                selected === s.id
                                    ? `bg-gradient-to-r ${s.color} text-white shadow-lg scale-105`
                                    : allHighlight.includes(s.id)
                                        ? "bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 border-2 border-indigo-300 dark:border-indigo-700"
                                        : "bg-slate-50 dark:bg-slate-700 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-600"
                            }`}
                        >
                            <div className="text-xs opacity-60">{s.id}</div>
                            {s.name}
                        </button>
                    ))}
                </div>

                {selected && (
                    <div className="text-center text-sm text-slate-600 dark:text-slate-400">
                        <span className="text-indigo-600 dark:text-indigo-400 font-bold">{selected}</span> 依赖{" "}
                        <span className="text-purple-600 dark:text-purple-400">{highlightDeps.join(", ") || "无"}</span>
                        {" → "}被依赖{" "}
                        <span className="text-emerald-600 dark:text-emerald-400">{highlightDown.join(", ") || "无"}</span>
                    </div>
                )}
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                <h4 className="text-base font-bold text-indigo-600 dark:text-indigo-400 mb-3">依赖关系表</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {stages.map((s) => (
                        <div key={s.id} className="flex items-center gap-2 text-sm p-2 bg-slate-50 dark:bg-slate-900/50 rounded">
                            <span className="font-mono font-bold text-indigo-600 dark:text-indigo-400 w-4">{s.id}</span>
                            <span className="text-slate-700 dark:text-slate-300">{s.name}</span>
                            <span className="text-slate-400">←</span>
                            <span className="text-xs text-slate-500">{s.deps.length > 0 ? s.deps.join(", ") : "无依赖"}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
