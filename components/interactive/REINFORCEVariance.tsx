"use client";

export function REINFORCEVariance() {
    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-orange-50 to-red-50 dark:from-slate-900 dark:to-orange-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    REINFORCE 方差问题
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-red-600 mb-4">高方差来源</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>1. 长 Episode:</strong> 累积大量随机性
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>2. 奖励随机:</strong> 环境本身的随机性
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>3. 策略随机:</strong> 动作采样的随机性
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-green-600 mb-4">降低方差方法</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>1. Baseline:</strong> 减去状态价值 V(s)
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>2. Advantage:</strong> A(s,a) = Q(s,a) - V(s)
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>3. 标准化:</strong> (G - mean) / std
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">方差影响</h4>
                <div className="grid grid-cols-3 gap-4 text-sm text-center">
                    <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded">
                        <div className="text-2xl mb-2">📈</div>
                        <div className="font-bold">高方差</div>
                        <div className="text-xs mt-2">学习不稳定</div>
                    </div>
                    <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                        <div className="text-2xl mb-2">🐌</div>
                        <div className="font-bold">需要更多样本</div>
                        <div className="text-xs mt-2">收敛缓慢</div>
                    </div>
                    <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded">
                        <div className="text-2xl mb-2">💥</div>
                        <div className="font-bold">可能发散</div>
                        <div className="text-xs mt-2">训练失败</div>
                    </div>
                </div>
            </div>
        </div>
    );
}
