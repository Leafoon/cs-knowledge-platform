"use client";

export function RewardRecovery() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-cyan-50 to-teal-50 dark:from-slate-900 dark:to-cyan-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    奖励函数恢复过程
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">MaxEnt IRL 恢复流程</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                        <strong>输入:</strong> 专家演示轨迹 D = τ₁, τ₂, ..., τₙ
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                        <strong>步骤 1:</strong> 计算专家特征期望 μₑ = E[φ(s,a)]
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                        <strong>步骤 2:</strong> 初始化奖励参数 θ = 0
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                        <strong>步骤 3:</strong> 用 rθ = θᵀφ(s,a) 执行 RL
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                        <strong>步骤 4:</strong> 计算学习策略特征期望 μπ
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                        <strong>步骤 5:</strong> 梯度更新 θ ← θ + α(μₑ - μπ)
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded border-2 border-green-500">
                        <strong>输出:</strong> 恢复的奖励函数 Rθ
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-cyan-600 mb-4">特征匹配目标</h4>
                    <div className="text-center p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                        <div className="font-mono mb-2">μₑ ≈ μπ</div>
                        <div className="text-xs text-slate-600 dark:text-slate-400">
                            专家与学习策略的特征期望匹配
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-teal-600 mb-4">收敛条件</h4>
                    <div className="text-center p-4 bg-teal-50 dark:bg-teal-900/20 rounded">
                        <div className="font-mono mb-2">‖μₑ - μπ‖ &lt; ε</div>
                        <div className="text-xs text-slate-600 dark:text-slate-400">
                            特征期望差异小于阈值
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
