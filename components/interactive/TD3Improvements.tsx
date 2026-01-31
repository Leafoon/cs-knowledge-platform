"use client";

export function TD3Improvements() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-emerald-50 to-green-50 dark:from-slate-900 dark:to-emerald-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    TD3 三大改进
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">Twin Delayed DDPG</p>
            </div>

            <div className="space-y-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-l-4 border-emerald-500">
                    <div className="flex items-start gap-4">
                        <div className="flex-shrink-0 w-12 h-12 bg-emerald-500 text-white rounded-xl flex items-center justify-center text-2xl font-bold">
                            1
                        </div>
                        <div className="flex-1">
                            <h4 className="text-lg font-bold text-emerald-600 mb-3">Clipped Double Q-learning</h4>
                            <div className="space-y-3 text-sm">
                                <div className="p-3 bg-emerald-50 dark:bg-emerald-900/20 rounded">
                                    <strong>问题:</strong> 单个 Q 网络过度估计
                                </div>
                                <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                                    <strong>解决:</strong> 使用两个 Critic，取<strong>最小值</strong>
                                </div>
                                <div className="font-mono text-xs p-3 bg-slate-100 dark:bg-slate-700 rounded">
                                    y = r + γ min(Q<sub>1</sub>', Q<sub>2</sub>')(s', μ'(s'))
                                </div>
                                <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                                    <strong>效果:</strong> 保守估计，降低过高估计
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-l-4 border-blue-500">
                    <div className="flex items-start gap-4">
                        <div className="flex-shrink-0 w-12 h-12 bg-blue-500 text-white rounded-xl flex items-center justify-center text-2xl font-bold">
                            2
                        </div>
                        <div className="flex-1">
                            <h4 className="text-lg font-bold text-blue-600 mb-3">Delayed Policy Updates</h4>
                            <div className="space-y-3 text-sm">
                                <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                                    <strong>问题:</strong> 频繁更新 Actor 导致不稳定
                                </div>
                                <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                                    <strong>解决:</strong> 每 d 次 Critic 更新才更新 1 次 Actor (d=2)
                                </div>
                                <div className="font-mono text-xs p-3 bg-slate-100 dark:bg-slate-700 rounded">
                                    if step % policy_delay == 0:<br />
                                    &nbsp;&nbsp;update_actor()
                                </div>
                                <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                                    <strong>效果:</strong> Critic 先收敛，引导 Actor 更新
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-l-4 border-purple-500">
                    <div className="flex items-start gap-4">
                        <div className="flex-shrink-0 w-12 h-12 bg-purple-500 text-white rounded-xl flex items-center justify-center text-2xl font-bold">
                            3
                        </div>
                        <div className="flex-1">
                            <h4 className="text-lg font-bold text-purple-600 mb-3">Target Policy Smoothing</h4>
                            <div className="space-y-3 text-sm">
                                <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                                    <strong>问题:</strong> 确定性目标策略可能过拟合
                                </div>
                                <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                                    <strong>解决:</strong> 目标策略添加<strong>裁剪的噪声</strong>
                                </div>
                                <div className="font-mono text-xs p-3 bg-slate-100 dark:bg-slate-700 rounded">
                                    noise = clip(N(0, σ), -c, c)<br />
                                    ã = μ'(s') + noise
                                </div>
                                <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                                    <strong>效果:</strong> 平滑 Q 值表面，减少过拟合
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-center">TD3 vs DDPG 性能对比</h4>
                <div className="grid grid-cols-2 gap-4 text-sm text-center">
                    <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded">
                        <div className="font-bold text-red-600">DDPG</div>
                        <div className="mt-2">• 过度估计 Q 值<br />• 训练不稳定<br />• 性能波动大</div>
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded border-2 border-green-500">
                        <div className="font-bold text-green-600">TD3</div>
                        <div className="mt-2">• 保守 Q 估计<br />• 更稳定收敛<br />• 性能更优</div>
                    </div>
                </div>
            </div>
        </div>
    );
}
