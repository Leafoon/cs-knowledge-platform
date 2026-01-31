"use client";

export function SACArchitecture() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    SAC 架构
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-indigo-500">
                    <div className="text-center mb-4">
                        <div className="text-4xl mb-2">🎭</div>
                        <h4 className="text-xl font-bold text-indigo-600">Actor π<sub>θ</sub></h4>
                    </div>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>类型:</strong> Squashed Gaussian
                        </div>
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>输入:</strong> 状态 s
                        </div>
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>输出:</strong> μ, σ
                        </div>
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>采样:</strong> a = tanh(μ + σ·ε)
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-purple-500">
                    <div className="text-center mb-4">
                        <div className="text-4xl mb-2">📊</div>
                        <h4 className="text-xl font-bold text-purple-600">Twin Critics</h4>
                    </div>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>Q<sub>φ₁</sub>, Q<sub>φ₂</sub>:</strong> 双 Q 网络
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>输入:</strong> (s, a)
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>输出:</strong> Q 值
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>目标:</strong> min(Q₁, Q₂)
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-pink-500">
                    <div className="text-center mb-4">
                        <div className="text-4xl mb-2">🌡️</div>
                        <h4 className="text-xl font-bold text-pink-600">Temperature α</h4>
                    </div>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-pink-50 dark:bg-pink-900/20 rounded">
                            <strong>log α:</strong> 可学习参数
                        </div>
                        <div className="p-3 bg-pink-50 dark:bg-pink-900/20 rounded">
                            <strong>作用:</strong> 控制探索程度
                        </div>
                        <div className="p-3 bg-pink-50 dark:bg-pink-900/20 rounded">
                            <strong>目标:</strong> H(π) ≥ H̄
                        </div>
                        <div className="p-3 bg-pink-50 dark:bg-pink-900/20 rounded">
                            <strong>自动调整</strong>
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">SAC 训练流程</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-indigo-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
                        <div>采样动作: a, log π(a|s) = μ + σ·ε (reparameterization)</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
                        <div>更新 Critics: y = r + γ(min(Q₁', Q₂')(s',a') - α log π(a'|s'))</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-indigo-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
                        <div>更新 Actor: 最大化 E[min(Q₁,Q₂)(s,a) - α log π(a|s)]</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-pink-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
                        <div>更新 α: 最小化 E[-α(log π(a|s) + H̄)]</div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">SAC 核心技巧</h4>
                <div className="grid grid-cols-3 gap-4 text-sm text-center">
                    <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                        <div className="text-2xl mb-2">🔄</div>
                        <div className="font-bold">Reparameterization</div>
                        <div className="text-xs mt-2">使策略可微</div>
                    </div>
                    <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded">
                        <div className="text-2xl mb-2">👥</div>
                        <div className="font-bold">Twin Critics</div>
                        <div className="text-xs mt-2">减少过高估计</div>
                    </div>
                    <div className="p-4 bg-pink-50 dark:bg-pink-900/20 rounded">
                        <div className="text-2xl mb-2">🌡️</div>
                        <div className="font-bold">Auto α</div>
                        <div className="text-xs mt-2">自适应探索</div>
                    </div>
                </div>
            </div>
        </div>
    );
}
