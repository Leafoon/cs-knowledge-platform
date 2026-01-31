"use client";

export function PrioritizedReplayWeighting() {
    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-rose-50 to-red-50 dark:from-slate-900 dark:to-rose-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Prioritized Replay 优先级采样
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-rose-600 mb-4">1. 优先级</h4>
                    <div className="font-mono text-sm bg-rose-50 dark:bg-rose-900/20 p-4 rounded">
                        p<sub>i</sub> = |δ<sub>i</sub>| + ε
                    </div>
                    <div className="text-sm mt-3">
                        基于 TD error 大小
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-orange-600 mb-4">2. 采样概率</h4>
                    <div className="font-mono text-sm bg-orange-50 dark:bg-orange-900/20 p-4 rounded">
                        P(i) = p<sub>i</sub><sup>α</sup> / Σp<sub>k</sub><sup>α</sup>
                    </div>
                    <div className="text-sm mt-3">
                        α 控制优先级强度
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-purple-600 mb-4">3. IS权重</h4>
                    <div className="font-mono text-sm bg-purple-50 dark:bg-purple-900/20 p-4 rounded">
                        w<sub>i</sub> = (N·P(i))<sup>-β</sup>
                    </div>
                    <div className="text-sm mt-3">
                        消除采样偏差
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">核心思想</h4>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    <strong>动机:</strong> 高 TD error 的转移更有信息量，应该更频繁地被采样学习<br />
                    <strong>效果:</strong> 显著提高样本效率，特别是在稀疏奖励任务中<br />
                    <strong>成本:</strong> 需要维护优先级队列和重要性采样权重
                </p>
            </div>
        </div>
    );
}
