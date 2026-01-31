"use client";

import { useState } from "react";

export function MaxEntropyFramework() {
    const [alpha, setAlpha] = useState(0.2);

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-slate-900 dark:to-emerald-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    最大熵框架
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">最大熵目标函数</h4>
                <div className="font-mono text-center p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded border-2 border-emerald-500">
                    J(π) = E[Σ(r<sub>t</sub> + α H(π(·|s<sub>t</sub>)))]
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">调整温度参数 α</h4>
                <div className="flex items-center gap-4">
                    <span className="text-sm font-medium">α = {alpha.toFixed(2)}</span>
                    <input
                        type="range"
                        min="0"
                        max="100"
                        value={alpha * 100}
                        onChange={(e) => setAlpha(parseInt(e.target.value) / 100)}
                        className="flex-1 h-2 bg-emerald-200 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex gap-2">
                        <button onClick={() => setAlpha(0)} className="px-3 py-1 text-xs bg-emerald-100 rounded">0 (贪婪)</button>
                        <button onClick={() => setAlpha(0.2)} className="px-3 py-1 text-xs bg-emerald-600 text-white rounded">0.2</button>
                        <button onClick={() => setAlpha(1)} className="px-3 py-1 text-xs bg-emerald-100 rounded">1.0 (高探索)</button>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div className={`bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg ${alpha < 0.05 ? 'border-4 border-emerald-500' : ''}`}>
                    <h4 className="text-lg font-bold text-blue-600 mb-4">α ≈ 0 (低熵)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>行为:</strong> 接近确定性策略
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>探索:</strong> 极少探索
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>风险:</strong> 可能陷入局部最优
                        </div>
                    </div>
                </div>

                <div className={`bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg ${alpha >= 0.05 && alpha <= 0.4 ? 'border-4 border-green-500' : ''}`}>
                    <h4 className="text-lg font-bold text-green-600 mb-4">α ≈ 0.2 (平衡)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>行为:</strong> 平衡探索与利用
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>探索:</strong> 适度探索
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>💡 推荐值</strong>
                        </div>
                    </div>
                </div>

                <div className={`bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg ${alpha > 0.4 ? 'border-4 border-emerald-500' : ''}`}>
                    <h4 className="text-lg font-bold text-orange-600 mb-4">α {'>'} 0.5 (高熵)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>行为:</strong> 接近均匀分布
                        </div>
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>探索:</strong> 大量探索
                        </div>
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>风险:</strong> 学习缓慢
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">最大熵的优势</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded">
                        <strong className="text-emerald-700 dark:text-emerald-400">✅ 内在探索</strong><br />
                        熵 bonus 自动鼓励探索
                    </div>
                    <div className="p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded">
                        <strong className="text-emerald-700 dark:text-emerald-400">✅ 鲁棒性</strong><br />
                        学习多种解决方案
                    </div>
                    <div className="p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded">
                        <strong className="text-emerald-700 dark:text-emerald-400">✅ 泛化性</strong><br />
                        更广泛的探索空间
                    </div>
                    <div className="p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded">
                        <strong className="text-emerald-700 dark:text-emerald-400">✅ 自适应</strong><br />
                        探索自动衰减
                    </div>
                </div>
            </div>
        </div>
    );
}
