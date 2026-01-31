"use client";

export function PolicyGradientTheorem() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-violet-50 to-purple-50 dark:from-slate-900 dark:to-violet-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    策略梯度定理推导
                </h3>
            </div>

            <div className="space-y-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold mb-4">1. 目标函数</h4>
                    <div className="font-mono text-sm bg-violet-50 dark:bg-violet-900/20 p-4 rounded">
                        J(θ) = E<sub>τ~π<sub>θ</sub></sub>[G(τ)] = E<sub>τ</sub>[Σr<sub>t</sub>]
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold mb-4">2. 轨迹概率</h4>
                    <div className="font-mono text-sm bg-violet-50 dark:bg-violet-900/20 p-4 rounded">
                        P(τ|θ) = μ(s₀) ∏π(a<sub>t</sub>|s<sub>t</sub>;θ) P(s'|s,a)
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold mb-4">3. Log-Derivative Trick</h4>
                    <div className="font-mono text-sm bg-violet-50 dark:bg-violet-900/20 p-4 rounded space-y-2">
                        <div>∇P(τ|θ) = P(τ|θ) ∇log P(τ|θ)</div>
                        <div className="mt-2">∇log P(τ|θ) = Σ ∇log π(a<sub>t</sub>|s<sub>t</sub>;θ)</div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-violet-500">
                    <h4 className="text-lg font-bold mb-4">4. 最终结果 ✨</h4>
                    <div className="font-mono text-lg bg-violet-50 dark:bg-violet-900/20 p-4 rounded">
                        ∇<sub>θ</sub>J(θ) = E[Σ ∇<sub>θ</sub>log π(a<sub>t</sub>|s<sub>t</sub>;θ) · G<sub>t</sub>]
                    </div>
                </div>

                <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-6 border-l-4 border-green-500">
                    <h4 className="text-lg font-bold text-green-800 dark:text-green-300 mb-3">
                        关键洞察
                    </h4>
                    <ul className="space-y-2 text-sm text-green-700 dark:text-green-400">
                        <li>• 梯度<strong>不依赖</strong>环境动态 P(s'|s,a)</li>
                        <li>• 无需模型（model-free）</li>
                        <li>• 只需对策略求梯度</li>
                        <li>• 可用蒙特卡洛采样估计</li>
                    </ul>
                </div>
            </div>
        </div>
    );
}
