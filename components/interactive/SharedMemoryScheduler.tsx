"use client";

import { useState } from "react";

const phases = [
    { name: "数据分块", desc: "将大矩阵切分为适合共享内存的小块(tile)", code: "tile_M, tile_K = 64, 32", color: "bg-blue-500" },
    { name: "加载到共享内存", desc: "协作加载数据块到 shared memory", code: 'A_shared = allocate(..., scope="shared")\nA_shared[...] = A_global[...]', color: "bg-purple-500" },
    { name: "块内计算", desc: "在共享内存上执行计算，减少全局内存访问", code: "for i in range(tile_M):\n  for k in range(tile_K):\n    C += A_shared[i,k] * B[k,j]", color: "bg-emerald-500" },
    { name: "写回结果", desc: "将计算结果写回全局内存", code: "C_global[o:o+tile] = C_local[...]", color: "bg-indigo-500" },
];

export function SharedMemoryScheduler() {
    const [phase, setPhase] = useState(0);
    const N = 8;
    const tileSize = 4;

    const renderGrid = () => (
        <div className="grid grid-cols-8 gap-0.5 w-fit mx-auto">
            {Array.from({ length: N * N }, (_, i) => {
                const r = Math.floor(i / N);
                const c = i % N;
                const inTile = r < tileSize && c < tileSize;
                let bg = "bg-slate-200 dark:bg-slate-700";
                if (phase >= 3 && inTile) bg = "bg-indigo-500";
                else if (phase >= 2 && inTile) bg = "bg-emerald-500";
                else if (phase >= 1 && inTile) bg = "bg-purple-500";
                return (
                    <div key={i} className={`w-8 h-8 rounded-sm ${bg} flex items-center justify-center text-xs font-mono text-white/70 transition-colors`}>
                        {r},{c}
                    </div>
                );
            })}
        </div>
    );

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">共享内存调度</h3>

            <div className="flex gap-1 mb-6 overflow-x-auto pb-2">
                {phases.map((p, i) => (
                    <button
                        key={i}
                        onClick={() => setPhase(i)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-all ${
                            phase === i
                                ? "bg-indigo-600 text-white shadow-lg"
                                : phase > i
                                    ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-700"
                                    : "bg-white dark:bg-slate-800 text-slate-500 border border-slate-200 dark:border-slate-700"
                        }`}
                    >
                        {p.name}
                    </button>
                ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                    <h4 className="text-base font-bold text-indigo-600 dark:text-indigo-400 mb-3">矩阵视图</h4>
                    {renderGrid()}
                    <div className="flex items-center gap-3 mt-3 text-xs text-slate-500 flex-wrap">
                        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-slate-300 dark:bg-slate-700 rounded-sm" /> 全局</span>
                        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-purple-500 rounded-sm" /> 共享</span>
                        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-emerald-500 rounded-sm" /> 计算</span>
                        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-indigo-500 rounded-sm" /> 写回</span>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                    <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-2">{phases[phase].name}</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">{phases[phase].desc}</p>
                    <pre className="bg-slate-100 dark:bg-slate-900 rounded-lg p-3 text-xs font-mono text-emerald-700 dark:text-emerald-400 overflow-x-auto whitespace-pre-wrap">
                        {phases[phase].code}
                    </pre>
                </div>
            </div>
        </div>
    );
}
