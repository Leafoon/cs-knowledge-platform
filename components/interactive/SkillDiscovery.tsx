"use client";

export function SkillDiscovery() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-violet-50 to-fuchsia-50 dark:from-slate-900 dark:to-violet-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    技能发现 (DIAYN)
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Diversity is All You Need
                </p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">目标函数</h4>
                <div className="font-mono text-center p-4 bg-violet-50 dark:bg-violet-900/20 rounded border-2 border-violet-500">
                    max I(Z; S) - I(Z; A|S)
                </div>
                <div className="mt-4 text-sm text-center text-slate-600 dark:text-slate-400">
                    技能应对状态有影响，但对动作影响小
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-violet-500">
                    <h4 className="text-lg font-bold text-violet-600 mb-4">I(Z; S) (可区分性)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                            <strong>含义:</strong> 技能对状态有影响
                        </div>
                        <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                            <strong>训练:</strong> 判别器 q(z|s)
                        </div>
                        <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                            <strong>效果:</strong> 不同技能访问不同状态
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-fuchsia-600 mb-4">I(Z; A|S) (探索性)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-fuchsia-50 dark:bg-fuchsia-900/20 rounded">
                            <strong>含义:</strong> 技能对动作影响小
                        </div>
                        <div className="p-3 bg-fuchsia-50 dark:bg-fuchsia-900/20 rounded">
                            <strong>训练:</strong> 最大化熵
                        </div>
                        <div className="p-3 bg-fuchsia-50 dark:bg-fuchsia-900/20 rounded">
                            <strong>效果:</strong> 鼓励探索性行为
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">DIAYN 训练流程</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-gradient-to-r from-violet-50 to-fuchsia-50 dark:from-violet-900/20 dark:to-fuchsia-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-violet-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
                        <div><strong>采样技能:</strong> z ~ p(z) (例如均匀分布)</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-violet-50 to-fuchsia-50 dark:from-violet-900/20 dark:to-fuchsia-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-fuchsia-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
                        <div><strong>执行技能:</strong> 用 π<sub>z</sub> 收集轨迹</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-violet-50 to-fuchsia-50 dark:from-violet-900/20 dark:to-fuchsia-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-violet-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
                        <div><strong>训练判别器:</strong> q<sub>φ</sub>(z|s) 预测技能</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-violet-50 to-fuchsia-50 dark:from-violet-900/20 dark:to-fuchsia-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-fuchsia-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
                        <div><strong>训练策略:</strong> π<sub>z</sub> 最大化伪奖励</div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">伪奖励</h4>
                <div className="font-mono text-center p-4 bg-pink-50 dark:bg-pink-900/20 rounded border-2 border-pink-500">
                    r(s, a, z) = log q<sub>φ</sub>(z|s) - log p(z)
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">应用场景</h4>
                <div className="grid grid-cols-3 gap-4 text-sm text-center">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <div className="text-2xl mb-2">🎓</div>
                        <div className="font-bold">预训练</div>
                        <div className="text-xs mt-2">先学习技能<br />再用于下游任务</div>
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <div className="text-2xl mb-2">🔄</div>
                        <div className="font-bold">迁移学习</div>
                        <div className="text-xs mt-2">技能可复用<br />于新任务</div>
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <div className="text-2xl mb-2">🔍</div>
                        <div className="font-bold">探索</div>
                        <div className="text-xs mt-2">技能作为<br />探索策略</div>
                    </div>
                </div>
            </div>
        </div>
    );
}
