"use client";

import { useState } from "react";

export function BaselineEffect() {
    const [useBaseline, setUseBaseline] = useState(false);

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-slate-900 dark:to-cyan-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Baseline 效果演示
                </h3>
            </div>

            <div className="flex justify-center gap-4 mb-6">
                <button onClick={() => setUseBaseline(false)} className={`px-6 py-2 rounded-lg font-bold ${!useBaseline ? "bg-red-600 text-white" : "bg-red-100 text-red-700"}`}>
                    无 Baseline
                </button>
                <button onClick={() => setUseBaseline(true)} className={`px-6 py-2 rounded-lg font-bold ${useBaseline ? "bg-green-600 text-white" : "bg-green-100 text-green-700"}`}>
                    有 Baseline
                </button>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">更新公式</h4>
                {!useBaseline ? (
                    <div className="space-y-4">
                        <div className="font-mono text-sm bg-red-50 dark:bg-red-900/20 p-4 rounded">
                            θ ← θ + α ∇log π(a|s;θ) · G<sub>t</sub>
                        </div>
                        <div className="text-sm text-red-700 dark:text-red-400">
                            <strong>问题:</strong> G<sub>t</sub> 方差很大
                        </div>
                    </div>
                ) : (
                    <div className="space-y-4">
                        <div className="font-mono text-sm bg-green-50 dark:bg-green-900/20 p-4 rounded">
                            θ ← θ + α ∇log π(a|s;θ) · (G<sub>t</sub> - V(s<sub>t</sub>))
                        </div>
                        <div className="text-sm text-green-700 dark:text-green-400">
                            <strong>优势:</strong> 使用 Advantage = Q - V，方差降低 30-50%
                        </div>
                    </div>
                )}
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">关键性质</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong>无偏性:</strong> E[∇log π · b(s)] = 0（不改变期望）
                    </div>
                    <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong>降方差:</strong> Var[G - b] &lt; Var[G]
                    </div>
                    <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                        <strong>最优选择:</strong> b<sup>*</sup>(s) = V(s)
                    </div>
                </div>
            </div>
        </div>
    );
}
