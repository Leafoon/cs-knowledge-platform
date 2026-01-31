"use client";

export function ActorCriticArchitecture() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Actor-Critic 架构
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-indigo-500">
                    <div className="text-center mb-4">
                        <div className="text-4xl mb-2">🎭</div>
                        <h4 className="text-xl font-bold text-indigo-600">Actor (策略网络)</h4>
                    </div>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>输入:</strong> 状态 s
                        </div>
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>输出:</strong> π(a|s;θ)
                        </div>
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>作用:</strong> 决策做什么
                        </div>
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>更新:</strong> θ ← θ + αδ∇logπ
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-purple-500">
                    <div className="text-center mb-4">
                        <div className="text-4xl mb-2">🎓</div>
                        <h4 className="text-xl font-bold text-purple-600">Critic (价值网络)</h4>
                    </div>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>输入:</strong> 状态 s
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>输出:</strong> V(s;w)
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>作用:</strong> 评估做得如何
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>更新:</strong> w ← w + αδ∇V
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">TD Error 连接两者</h4>
                <div className="font-mono text-center text-lg p-4 bg-green-50 dark:bg-green-900/20 rounded">
                    δ = r + γV(s';w) - V(s;w)
                </div>
                <div className="mt-4 text-sm text-center text-slate-600 dark:text-slate-400">
                    δ 是 Advantage A(s,a) 的无偏估计，同时用于更新 Actor 和 Critic
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 Actor-Critic 结合策略梯度和价值函数，实现在线学习
            </div>
        </div>
    );
}
