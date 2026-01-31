"use client";

import { useState } from "react";

export function CountBasedBonus() {
    const [visits, setVisits] = useState([1, 5, 10, 50, 100]);

    const getBonus = (n: number, beta: number = 1.0) => beta / Math.sqrt(n);

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-slate-900 dark:to-green-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Count-Based 奖励 Bonus
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">公式</h4>
                <div className="font-mono text-center p-4 bg-green-50 dark:bg-green-900/20 rounded border-2 border-green-500">
                    r<sup>+</sup> = r + β / √N(s)
                </div>
                <div className="mt-4 text-sm text-center text-slate-600 dark:text-slate-400">
                    访问次数越少 → bonus 越大 → 更鼓励探索
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Bonus 随访问次数变化</h4>
                <div className="space-y-3">
                    {visits.map((n, i) => {
                        const bonus = getBonus(n);
                        const maxBonus = getBonus(1);
                        const width = (bonus / maxBonus) * 100;

                        return (
                            <div key={i}>
                                <div className="flex items-center justify-between mb-1 text-sm">
                                    <span>访问 {n} 次</span>
                                    <span className="font-mono">bonus = {bonus.toFixed(3)}</span>
                                </div>
                                <div className="h-8 bg-slate-100 dark:bg-slate-700 rounded overflow-hidden">
                                    <div
                                        className={`h-full ${i === 0 ? 'bg-green-500' : i === 1 ? 'bg-green-400' : i === 2 ? 'bg-green-300' : 'bg-green-200'} flex items-center px-3 text-white font-bold text-sm`}
                                        style={{ width: `${width}%` }}
                                    >
                                        {bonus.toFixed(3)}
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-green-600 mb-4">优势</h4>
                    <div className="space-y-2 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            ✅ 简单直观
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            ✅ 自动鼓励访问新状态
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            ✅ 适用于表格型 RL
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-red-600 mb-4">局限性</h4>
                    <div className="space-y-2 text-sm">
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            ❌ 高维状态空间失效
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            ❌ 连续状态难以计数
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            ❌ 需要状态哈希或密度模型
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
