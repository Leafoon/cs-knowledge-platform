"use client";

export function MAPPOArchitecture() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-violet-50 to-purple-50 dark:from-slate-900 dark:to-violet-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    MAPPO 架构
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Multi-Agent PPO
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-violet-500">
                    <div className="text-center mb-4">
                        <div className="text-4xl mb-2">🎭</div>
                        <h4 className="text-xl font-bold text-violet-600">Actor（分散）</h4>
                    </div>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                            <strong>输入:</strong> 局部观测 o^i
                        </div>
                        <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                            <strong>输出:</strong> 动作 a^i
                        </div>
                        <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                            <strong>参数:</strong> 可共享
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-purple-500">
                    <div className="text-center mb-4">
                        <div className="text-4xl mb-2">📊</div>
                        <h4 className="text-xl font-bold text-purple-600">Critic（集中）</h4>
                    </div>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>输入:</strong> 全局状态 s
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>输出:</strong> 价值 V(s)
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>训练:</strong> 仅训练时使用
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">PPO更新</h4>
                <div className="text-center p-4 bg-violet-50 dark:bg-violet-900/20 rounded">
                    <div className="font-mono text-sm mb-2">
                        L = min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                        r_t = π_new / π_old
                    </div>
                </div>
            </div>
        </div>
    );
}
