"use client";

export function DistributionShift() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-yellow-50 to-orange-50 dark:from-slate-900 dark:to-yellow-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    分布漂移可视化
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">错误累积过程</h4>
                <div className="space-y-4">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded border-2 border-green-500">
                        <strong>时刻 0:</strong> 策略完美匹配<br />
                        <div className="text-sm mt-2">状态 s₀ ∈ 专家分布 ✓</div>
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded border-2 border-yellow-500">
                        <strong>时刻 1:</strong> 小错误出现<br />
                        <div className="text-sm mt-2">策略偏差 ε → 到达 s'₁ ≈ s₁</div>
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded border-2 border-orange-500">
                        <strong>时刻 2:</strong> 偏离专家轨迹<br />
                        <div className="text-sm mt-2">状态 s₂ ∉ 专家数据（未见过）</div>
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded border-2 border-red-500">
                        <strong>时刻 T:</strong> 完全失控<br />
                        <div className="text-sm mt-2">累积误差 = O(εT²)</div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-yellow-600 mb-4">训练 vs 测试分布</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>训练:</strong> p_expert(s)
                        </div>
                        <div className="text-center text-2xl">≠</div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>测试:</strong> p_π(s)
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-orange-600 mb-4">解决方案</h4>
                    <div className="space-y-2 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            ✅ DAgger（迭代聚合）
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            ✅ 数据增强
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            ✅ 噪声注入
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
