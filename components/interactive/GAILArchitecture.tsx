"use client";

export function GAILArchitecture() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-rose-50 to-red-50 dark:from-slate-900 dark:to-rose-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    GAIL 架构
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Generative Adversarial Imitation Learning
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-rose-500">
                    <div className="text-center mb-4">
                        <div className="text-4xl mb-2">🎭</div>
                        <h4 className="text-xl font-bold text-rose-600">判别器 D</h4>
                    </div>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-rose-50 dark:bg-rose-900/20 rounded">
                            <strong>输入:</strong> (s, a) 对
                        </div>
                        <div className="p-3 bg-rose-50 dark:bg-rose-900/20 rounded">
                            <strong>输出:</strong> D(s,a) ∈ [0,1]
                        </div>
                        <div className="p-3 bg-rose-50 dark:bg-rose-900/20 rounded">
                            <strong>目标:</strong> 区分专家 vs 策略
                        </div>
                        <div className="p-3 bg-rose-50 dark:bg-rose-900/20 rounded">
                            <strong>训练:</strong> 最大化分类准确率
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-red-500">
                    <div className="text-center mb-4">
                        <div className="text-4xl mb-2">🎯</div>
                        <h4 className="text-xl font-bold text-red-600">生成器 π</h4>
                    </div>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>输入:</strong> 状态 s
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>输出:</strong> 动作 a
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>目标:</strong> 生成类似专家的轨迹
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>训练:</strong> 欺骗判别器
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">GAIL 训练循环</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-gradient-to-r from-rose-50 to-red-50 dark:from-rose-900/20 dark:to-red-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-rose-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
                        <div><strong>策略生成:</strong> 用当前策略 π 收集轨迹</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-rose-50 to-red-50 dark:from-rose-900/20 dark:to-red-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-red-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
                        <div><strong>训练判别器:</strong> 区分专家数据 vs 策略数据</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-rose-50 to-red-50 dark:from-rose-900/20 dark:to-red-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-rose-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
                        <div><strong>计算奖励:</strong> r(s,a) = log D(s,a)</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-rose-50 to-red-50 dark:from-rose-900/20 dark:to-red-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-red-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
                        <div><strong>训练策略:</strong> 用 PPO/TRPO 最大化 GAIL 奖励</div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">GAIL 奖励公式</h4>
                <div className="font-mono text-center p-4 bg-pink-50 dark:bg-pink-900/20 rounded border-2 border-pink-500">
                    r(s, a) = log D(s, a)
                </div>
                <div className="mt-4 text-sm text-center text-slate-600 dark:text-slate-400">
                    判别器输出越高 → 更像专家 → 奖励越大
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">GAIL 的优势</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 无需奖励</strong><br />
                        不需要手工设计奖励函数
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 样本高效</strong><br />
                        比行为克隆更高效
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 稳定训练</strong><br />
                        结合 TRPO/PPO 稳定性好
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 理论保证</strong><br />
                        等价于最小化 JSD 散度
                    </div>
                </div>
            </div>
        </div>
    );
}
