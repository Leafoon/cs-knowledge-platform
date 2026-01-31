"use client";

export function SoftBellmanEquation() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-sky-50 to-blue-50 dark:from-slate-900 dark:to-sky-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Soft Bellman 方程
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-blue-600 mb-4">标准 Bellman 方程</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <div className="font-mono text-xs">
                                Q(s,a) = r + γ E[Q(s',a')]
                            </div>
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <div className="font-mono text-xs">
                                V(s) = E<sub>a~π</sub>[Q(s,a)]
                            </div>
                        </div>
                        <div className="p-3 bg-slate-50 dark:bg-slate-700 rounded">
                            <strong>特点:</strong> 只考虑奖励
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-sky-500">
                    <h4 className="text-lg font-bold text-sky-600 mb-4">Soft Bellman 方程</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-sky-50 dark:bg-sky-900/20 rounded">
                            <div className="font-mono text-xs">
                                Q<sup>soft</sup>(s,a) = r + γ E[Q<sup>soft</sup>(s',a') - α log π(a'|s')]
                            </div>
                        </div>
                        <div className="p-3 bg-sky-50 dark:bg-sky-900/20 rounded">
                            <div className="font-mono text-xs">
                                V<sup>soft</sup>(s) = E<sub>a~π</sub>[Q<sup>soft</sup>(s,a) - α log π(a|s)]
                            </div>
                        </div>
                        <div className="p-3 bg-emerald-50 dark:bg-emerald-900/20 rounded">
                            <strong>特点:</strong> 奖励 + 熵
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Soft V-function (LogSumExp)</h4>
                <div className="p-4 bg-sky-50 dark:bg-sky-900/20 rounded mb-4">
                    <div className="font-mono text-center">
                        V<sup>soft</sup>(s) = α log Σ<sub>a</sub> exp(Q<sup>soft</sup>(s,a) / α)
                    </div>
                </div>
                <div className="text-sm text-center text-slate-600 dark:text-slate-400">
                    连续动作: V<sup>soft</sup>(s) = α log ∫ exp(Q<sup>soft</sup>(s,a) / α) da
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">关键性质</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong>✅ 单调改进</strong><br />
                        Soft policy iteration 保证收敛
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong>✅ 最优策略</strong><br />
                        π*(a|s) ∝ exp(Q*(s,a)/α)
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong>✅ 平滑性</strong><br />
                        LogSumExp 是 max 的平滑近似
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong>✅ 包含熵</strong><br />
                        隐式鼓励探索
                    </div>
                </div>
            </div>
        </div>
    );
}
