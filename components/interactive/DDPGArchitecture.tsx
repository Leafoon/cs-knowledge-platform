"use client";

export function DDPGArchitecture() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-slate-900 dark:to-cyan-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    DDPG 架构
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-cyan-500">
                    <div className="text-center mb-4">
                        <div className="text-4xl mb-2">🎯</div>
                        <h4 className="text-xl font-bold text-cyan-600">Actor μ(s;θ<sup>μ</sup>)</h4>
                    </div>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                            <strong>作用:</strong> 确定性策略
                        </div>
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                            <strong>输入:</strong> 状态 s
                        </div>
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                            <strong>输出:</strong> 动作 a = μ(s)
                        </div>
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                            <strong>目标:</strong> 最大化 Q(s, μ(s))
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-blue-500">
                    <div className="text-center mb-4">
                        <div className="text-4xl mb-2">📊</div>
                        <h4 className="text-xl font-bold text-blue-600">Critic Q(s,a;θ<sup>Q</sup>)</h4>
                    </div>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>作用:</strong> 评估动作价值
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>输入:</strong> 状态 s + 动作 a
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>输出:</strong> Q 值
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>目标:</strong> TD 目标 r + γQ'(s',μ'(s'))
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">训练流程</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-cyan-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
                        <div>探索: a = μ(s) + 噪声</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-cyan-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
                        <div>存储经验 (s,a,r,s') 到 Replay Buffer</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
                        <div>Critic: 最小化 (Q(s,a) - y)<sup>2</sup>, y = r + γQ'(s',μ'(s'))</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-cyan-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
                        <div>Actor: 最大化 Q(s, μ(s))</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold">5</div>
                        <div>软更新目标网络: θ' ← τθ + (1-τ)θ'</div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">DDPG 核心技巧</h4>
                <div className="grid grid-cols-3 gap-4 text-sm text-center">
                    <div className="p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                        <div className="text-2xl mb-2">🎲</div>
                        <div className="font-bold">Experience Replay</div>
                        <div className="text-xs mt-2">打破样本相关性</div>
                    </div>
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <div className="text-2xl mb-2">🎯</div>
                        <div className="font-bold">Target Networks</div>
                        <div className="text-xs mt-2">软更新 τ=0.005</div>
                    </div>
                    <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded">
                        <div className="text-2xl mb-2">🔊</div>
                        <div className="font-bold">OU Noise</div>
                        <div className="text-xs mt-2">时间相关探索</div>
                    </div>
                </div>
            </div>
        </div>
    );
}
