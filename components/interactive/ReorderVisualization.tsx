"use client";

import { useState } from "react";

const loopOrders = [
    { name: "ijk（默认）", order: [0, 1, 2], desc: "按行优先遍历，对B矩阵不友好", cache: "低", perf: "1x" },
    { name: "ikj", order: [0, 2, 1], desc: "中间维度提前，改善B的局部性", cache: "中", perf: "1.5x" },
    { name: "jik", order: [1, 0, 2], desc: "按列主序遍历，对A矩阵友好", cache: "中", perf: "1.3x" },
];

const labels = ["i", "j", "k"];

export function ReorderVisualization() {
    const [selected, setSelected] = useState(0);
    const order = loopOrders[selected].order;

    const generateGrid = (ord: number[]) => {
        const N = 4;
        const cells: { row: number; col: number; step: number }[] = [];
        let step = 0;
        for (let a = 0; a < N; a++) {
            for (let b = 0; b < N; b++) {
                for (let c = 0; c < N; c++) {
                    const idx = [0, 0, 0];
                    idx[ord[0]] = a;
                    idx[ord[1]] = b;
                    idx[ord[2]] = c;
                    cells.push({ row: idx[0], col: idx[1], step });
                    step++;
                }
            }
        }
        return cells;
    };

    const cells = generateGrid(order);

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">循环重排可视化</h3>

            <div className="flex gap-2 mb-6">
                {loopOrders.map((lo, i) => (
                    <button
                        key={i}
                        onClick={() => setSelected(i)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                            selected === i
                                ? "bg-indigo-600 text-white shadow-lg"
                                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:border-indigo-300"
                        }`}
                    >
                        {lo.name}
                    </button>
                ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                    <h4 className="text-base font-bold text-indigo-600 dark:text-indigo-400 mb-3">访问顺序</h4>
                    <div className="grid grid-cols-4 gap-1 mb-3">
                        {Array.from({ length: 16 }, (_, i) => {
                            const r = Math.floor(i / 4);
                            const c = i % 4;
                            const cell = cells.find((cl) => cl.row === r && cl.col === c);
                            const stepNum = cell?.step ?? 0;
                            const intensity = stepNum / 63;
                            return (
                                <div
                                    key={i}
                                    className="aspect-square rounded flex items-center justify-center text-xs font-mono transition-all"
                                    style={{
                                        backgroundColor: `rgba(99, 102, 241, ${0.1 + intensity * 0.6})`,
                                        color: intensity > 0.5 ? "white" : "#818cf8",
                                    }}
                                >
                                    {stepNum}
                                </div>
                            );
                        })}
                    </div>
                    <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                        <span>早</span>
                        <div className="flex-1 h-2 rounded-full bg-gradient-to-r from-indigo-200 to-indigo-600 dark:from-indigo-900 dark:to-indigo-400" />
                        <span>晚</span>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                    <h4 className="text-base font-bold text-purple-600 dark:text-purple-400 mb-3">循环嵌套</h4>
                    <div className="bg-slate-100 dark:bg-slate-900 rounded-lg p-4 font-mono text-sm space-y-1 mb-4">
                        <div>
                            <span className="text-blue-600 dark:text-blue-400">for</span>{" "}
                            <span className="text-amber-600 dark:text-amber-300">{labels[order[0]]}</span>
                            <span className="text-slate-500"> in range(N):</span>
                        </div>
                        <div className="pl-4">
                            <span className="text-blue-600 dark:text-blue-400">for</span>{" "}
                            <span className="text-amber-600 dark:text-amber-300">{labels[order[1]]}</span>
                            <span className="text-slate-500"> in range(N):</span>
                        </div>
                        <div className="pl-8">
                            <span className="text-blue-600 dark:text-blue-400">for</span>{" "}
                            <span className="text-amber-600 dark:text-amber-300">{labels[order[2]]}</span>
                            <span className="text-slate-500"> in range(N):</span>
                        </div>
                        <div className="pl-12 text-emerald-600 dark:text-emerald-400">C[i][j] += A[i][k] * B[k][j]</div>
                    </div>
                    <div className="space-y-2 text-sm">
                        <div className="flex justify-between"><span className="text-slate-500">缓存友好度</span><span className="text-indigo-600 dark:text-indigo-400 font-medium">{loopOrders[selected].cache}</span></div>
                        <div className="flex justify-between"><span className="text-slate-500">相对性能</span><span className="text-purple-600 dark:text-purple-400 font-medium">{loopOrders[selected].perf}</span></div>
                    </div>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-3 bg-slate-50 dark:bg-slate-900/50 rounded-lg p-2">{loopOrders[selected].desc}</p>
                </div>
            </div>
        </div>
    );
}
