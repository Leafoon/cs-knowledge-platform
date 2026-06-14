"use client";

import { useState } from "react";

const splitConfigs = [
    { outer: 32, inner: 32, total: 1024, label: "factor=32", desc: "将1024次迭代分裂为32×32的两层循环" },
    { outer: 16, inner: 64, total: 1024, label: "factor=64", desc: "内层64次，外层16次" },
    { outer: 64, inner: 16, total: 1024, label: "factor=16", desc: "内层16次，外层64次" },
];

export function SplitVisualization() {
    const [selected, setSelected] = useState(0);
    const [hoverOuter, setHoverOuter] = useState<number | null>(null);
    const config = splitConfigs[selected];

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">循环分裂可视化</h3>

            <div className="flex gap-2 mb-6">
                {splitConfigs.map((cfg, i) => (
                    <button
                        key={i}
                        onClick={() => setSelected(i)}
                        className={`px-4 py-2 rounded-lg text-sm font-mono font-medium transition-all ${
                            selected === i
                                ? "bg-indigo-600 text-white shadow-lg"
                                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:border-indigo-300"
                        }`}
                    >
                        {cfg.label}
                    </button>
                ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                    <h4 className="text-base font-bold text-indigo-600 dark:text-indigo-400 mb-3">原始循环</h4>
                    <div className="bg-slate-100 dark:bg-slate-900 rounded-lg p-3 mb-3">
                        <code className="text-sm font-mono text-slate-700 dark:text-slate-300">
                            for <span className="text-amber-600 dark:text-amber-300">i</span> in range({config.total}):
                        </code>
                    </div>
                    <div className="flex flex-wrap gap-0.5">
                        {Array.from({ length: 32 }, (_, v) => (
                            <div key={v} className="w-5 h-5 bg-indigo-200 dark:bg-indigo-800/60 rounded text-xs flex items-center justify-center text-indigo-700 dark:text-indigo-300">
                                {v}
                            </div>
                        ))}
                        <div className="w-5 h-5 flex items-center justify-center text-slate-400 text-xs">...</div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                    <h4 className="text-base font-bold text-purple-600 dark:text-purple-400 mb-3">分裂后</h4>
                    <div className="bg-slate-100 dark:bg-slate-900 rounded-lg p-3 mb-3 space-y-1">
                        <code className="text-sm font-mono text-slate-700 dark:text-slate-300 block">
                            for <span className="text-blue-600 dark:text-blue-300">i_outer</span> in range({config.outer}):
                        </code>
                        <code className="text-sm font-mono text-slate-700 dark:text-slate-300 block pl-4">
                            for <span className="text-emerald-600 dark:text-emerald-300">i_inner</span> in range({config.inner}):
                        </code>
                        <code className="text-sm font-mono text-slate-700 dark:text-slate-300 block pl-8">
                            <span className="text-amber-600 dark:text-amber-300">i</span> = i_outer × {config.inner} + i_inner
                        </code>
                    </div>
                    <div className="space-y-1">
                        {Array.from({ length: Math.min(config.outer, 8) }, (_, outer) => (
                            <div
                                key={outer}
                                className="flex items-center gap-1 cursor-pointer"
                                onMouseEnter={() => setHoverOuter(outer)}
                                onMouseLeave={() => setHoverOuter(null)}
                            >
                                <span className="text-xs text-blue-600 dark:text-blue-300 w-8 font-mono">{outer}:</span>
                                <div className="flex gap-0.5 flex-wrap">
                                    {Array.from({ length: Math.min(config.inner, 16) }, (_, inner) => (
                                        <div
                                            key={inner}
                                            className={`w-4 h-4 rounded text-xs flex items-center justify-center transition-all ${
                                                hoverOuter === outer
                                                    ? "bg-indigo-500 text-white scale-110"
                                                    : "bg-slate-200 dark:bg-slate-700 text-slate-500 dark:text-slate-400"
                                            }`}
                                        >
                                            {inner}
                                        </div>
                                    ))}
                                    {config.inner > 16 && <span className="text-slate-400 text-xs">...</span>}
                                </div>
                            </div>
                        ))}
                        {config.outer > 8 && <div className="text-xs text-slate-400 pl-8">... 共{config.outer}个外层迭代</div>}
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg">
                <p className="text-sm text-slate-600 dark:text-slate-300 font-mono">
                    i = i_outer × {config.inner} + i_inner &nbsp; (0 ≤ i_outer &lt; {config.outer}, 0 ≤ i_inner &lt; {config.inner})
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">{config.desc}</p>
            </div>
        </div>
    );
}
