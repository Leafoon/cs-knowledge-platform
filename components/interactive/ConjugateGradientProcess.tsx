"use client";

import { useState } from "react";

export function ConjugateGradientProcess() {
    const [iteration, setIteration] = useState(0);
    const maxIter = 5;

    const residuals = [1.0, 0.5, 0.25, 0.125, 0.06, 0.03];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-slate-900 dark:to-cyan-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    共轭梯度迭代过程
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">求解: Fx = g</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                        <strong>F:</strong> Fisher Information Matrix
                    </div>
                    <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                        <strong>g:</strong> 策略梯度
                    </div>
                    <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                        <strong>x:</strong> 自然梯度方向
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-bold">迭代: {iteration} / {maxIter}</h4>
                    <div className="flex gap-2">
                        <button
                            onClick={() => setIteration(Math.max(0, iteration - 1))}
                            disabled={iteration === 0}
                            className="px-4 py-2 bg-cyan-100 rounded disabled:opacity-50"
                        >
                            ← 上一步
                        </button>
                        <button
                            onClick={() => setIteration(Math.min(maxIter, iteration + 1))}
                            disabled={iteration === maxIter}
                            className="px-4 py-2 bg-cyan-600 text-white rounded disabled:opacity-50"
                        >
                            下一步 →
                        </button>
                        <button
                            onClick={() => setIteration(0)}
                            className="px-4 py-2 bg-slate-200 rounded"
                        >
                            重置
                        </button>
                    </div>
                </div>

                <div className="space-y-3">
                    {Array.from({ length: iteration + 1 }).map((_, i) => (
                        <div key={i} className={`p-3 rounded ${i === iteration ? 'bg-cyan-100 dark:bg-cyan-900/30 border-2 border-cyan-500' : 'bg-slate-50 dark:bg-slate-700'}`}>
                            <div className="flex items-center justify-between">
                                <span className="font-mono text-sm">迭代 {i}</span>
                                <span className="text-sm">残差: {residuals[i].toFixed(3)}</span>
                            </div>
                            {i === iteration && (
                                <div className="mt-2 text-xs text-cyan-700 dark:text-cyan-400">
                                    {i === 0 && "初始化 x₀=0, r₀=b, p₀=r₀"}
                                    {i > 0 && "计算 α, 更新 x, r, β, p"}
                                </div>
                            )}
                        </div>
                    ))}
                </div>

                <div className="mt-4 h-32 bg-slate-50 dark:bg-slate-700 rounded">
                    <svg width="100%" height="100%" viewBox="0 0 400 100">
                        {residuals.slice(0, iteration + 1).map((r, i) => {
                            const x = (i / maxIter) * 400;
                            const y = 100 - r * 80;
                            return (
                                <g key={i}>
                                    {i > 0 && (
                                        <line
                                            x1={(i - 1) / maxIter * 400}
                                            y1={100 - residuals[i - 1] * 80}
                                            x2={x}
                                            y2={y}
                                            stroke="rgb(6, 182, 212)"
                                            strokeWidth="2"
                                        />
                                    )}
                                    <circle cx={x} cy={y} r="4" fill="rgb(6, 182, 212)" />
                                </g>
                            );
                        })}
                        <text x="200" y="95" textAnchor="middle" className="text-xs fill-current">残差收敛</text>
                    </svg>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">共轭梯度优势</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong>✅ 无需求逆</strong><br />
                        避免 O(n³) 复杂度
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong>✅ 快速收敛</strong><br />
                        通常 10-20 次迭代
                    </div>
                </div>
            </div>
        </div>
    );
}
