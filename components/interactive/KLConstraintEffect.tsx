"use client";

import { useState } from "react";

export function KLConstraintEffect() {
    const [kl, setKL] = useState(0.01);

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-slate-900 dark:to-purple-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    KL 约束的作用
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">调整 KL 上限 δ</h4>
                <div className="flex items-center gap-4">
                    <span className="text-sm font-medium">δ = {kl.toFixed(3)}</span>
                    <input
                        type="range"
                        min="1"
                        max="50"
                        value={kl * 1000}
                        onChange={(e) => setKL(parseInt(e.target.value) / 1000)}
                        className="flex-1 h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex gap-2">
                        <button onClick={() => setKL(0.001)} className="px-3 py-1 text-xs bg-purple-100 rounded">严格</button>
                        <button onClick={() => setKL(0.01)} className="px-3 py-1 text-xs bg-purple-600 text-white rounded">默认</button>
                        <button onClick={() => setKL(0.05)} className="px-3 py-1 text-xs bg-purple-100 rounded">宽松</button>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className={`bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg ${kl < 0.005 ? 'border-4 border-purple-500' : ''}`}>
                    <h4 className="text-lg font-bold text-purple-600 mb-4">δ 很小 ({'<'} 0.005)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>✅ 非常稳定</strong><br />
                            策略几乎不变
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>❌ 学习极慢</strong><br />
                            每步进展微小
                        </div>
                    </div>
                </div>

                <div className={`bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg ${kl >= 0.005 && kl <= 0.02 ? 'border-4 border-green-500' : ''}`}>
                    <h4 className="text-lg font-bold text-green-600 mb-4">δ 适中 (0.01)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>✅ 平衡</strong><br />
                            稳定 + 合理速度
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>💡 推荐值</strong><br />
                            大多数任务适用
                        </div>
                    </div>
                </div>

                <div className={`bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg ${kl > 0.02 ? 'border-4 border-purple-500' : ''}`}>
                    <h4 className="text-lg font-bold text-orange-600 mb-4">δ 很大 ({'>'} 0.02)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>✅ 学习快</strong><br />
                            策略变化大
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>❌ 可能不稳定</strong><br />
                            失去单调改进保证
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">KL 散度公式</h4>
                <div className="font-mono text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded">
                    D<sub>KL</sub>(π<sub>old</sub> || π<sub>new</sub>) = E<sub>a~π<sub>old</sub></sub>[log(π<sub>old</sub>(a|s) / π<sub>new</sub>(a|s))]
                </div>
            </div>
        </div>
    );
}
