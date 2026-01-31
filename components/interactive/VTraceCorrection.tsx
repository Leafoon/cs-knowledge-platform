"use client";

export function VTraceCorrection() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-slate-900 dark:to-purple-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    V-trace 修正机制
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">V-trace 目标函数</h4>
                <div className="font-mono text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded border-2 border-purple-500 text-sm">
                    v<sub>s</sub> = V(s) + Σ γ<sup>t-s</sup> (∏ c<sub>i</sub>) δ<sub>t</sub>
                </div>
                <div className="mt-4 text-sm text-center text-slate-600 dark:text-slate-400">
                    其中 δ<sub>t</sub> = ρ<sub>t</sub>(r<sub>t</sub> + γV(s<sub>t+1</sub>) - V(s<sub>t</sub>))
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-purple-600 mb-4">ρ<sub>t</sub> (TD 误差权重)</h4>
                    <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded mb-3">
                        <div className="font-mono text-sm text-center">
                            ρ<sub>t</sub> = min(ρ̄, π(a<sub>t</sub>|s<sub>t</sub>) / μ(a<sub>t</sub>|s<sub>t</sub>))
                        </div>
                    </div>
                    <div className="space-y-2 text-sm">
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>作用:</strong> 修正 TD 误差
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>截断 ρ̄:</strong> 通常 1.0
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>效果:</strong> 防止过高权重
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-pink-600 mb-4">c<sub>i</sub> (Trace 权重)</h4>
                    <div className="p-4 bg-pink-50 dark:bg-pink-900/20 rounded mb-3">
                        <div className="font-mono text-sm text-center">
                            c<sub>i</sub> = min(c̄, π(a<sub>i</sub>|s<sub>i</sub>) / μ(a<sub>i</sub>|s<sub>i</sub>))
                        </div>
                    </div>
                    <div className="space-y-2 text-sm">
                        <div className="p-3 bg-pink-50 dark:bg-pink-900/20 rounded">
                            <strong>作用:</strong> 控制 trace 长度
                        </div>
                        <div className="p-3 bg-pink-50 dark:bg-pink-900/20 rounded">
                            <strong>截断 c̄:</strong> 通常 1.0
                        </div>
                        <div className="p-3 bg-pink-50 dark:bg-pink-900/20 rounded">
                            <strong>效果:</strong> 稳定长期估计
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">为什么需要 V-trace？</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded">
                        <strong className="text-red-700 dark:text-red-400">问题: 策略滞后</strong><br />
                        Actor 的策略 μ 比 Learner 的策略 π 旧
                    </div>
                    <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                        <strong className="text-yellow-700 dark:text-yellow-400">风险: 方差爆炸</strong><br />
                        重要性权重 π/μ 可能非常大
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">解决: 截断权重</strong><br />
                        min(ρ̄, π/μ) 限制最大权重
                    </div>
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong className="text-blue-700 dark:text-blue-400">结果: 稳定训练</strong><br />
                        即使 μ 和 π 差异较大也能有效学习
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">V-trace vs 标准重要性采样</h4>
                <div className="space-y-4">
                    <div className="p-4 bg-slate-50 dark:bg-slate-700 rounded">
                        <strong>标准 IS:</strong>
                        <div className="font-mono text-sm mt-2">权重 = π/μ (无截断)</div>
                        <div className="text-sm mt-2 text-red-600">❌ 方差可能非常大</div>
                    </div>
                    <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded border-2 border-purple-500">
                        <strong>V-trace:</strong>
                        <div className="font-mono text-sm mt-2">权重 = min(ρ̄, π/μ)</div>
                        <div className="text-sm mt-2 text-green-600">✅ 方差受控，更稳定</div>
                    </div>
                </div>
            </div>
        </div>
    );
}
