"use client";

export function RatioClippingEffect() {
    const epsilon = 0.2;

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-amber-50 to-orange-50 dark:from-slate-900 dark:to-amber-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    比率裁剪边界
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">裁剪区域 (ε = {epsilon})</h4>
                <div className="relative h-40 bg-slate-50 dark:bg-slate-700 rounded flex items-center justify-center">
                    <div className="absolute inset-0 flex items-center">
                        <div className="w-full px-8">
                            {/* 裁剪区域 */}
                            <div className="relative h-16 bg-green-100 dark:bg-green-900/30 border-2 border-green-500 rounded"
                                style={{
                                    marginLeft: `${((1 - epsilon) / 1.6) * 100}%`,
                                    width: `${(2 * epsilon / 1.6) * 100}%`
                                }}>
                                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                                    <div className="text-sm font-bold text-green-700 dark:text-green-300">允许范围</div>
                                </div>
                            </div>

                            {/* 刻度 */}
                            <div className="relative h-8 mt-2">
                                <div className="absolute" style={{ left: `${((1 - epsilon) / 1.6) * 100}%` }}>
                                    <div className="text-xs text-center -ml-4">1-ε<br />{(1 - epsilon).toFixed(2)}</div>
                                </div>
                                <div className="absolute" style={{ left: `${(1 / 1.6) * 100}%` }}>
                                    <div className="text-xs text-center -ml-2">1.0</div>
                                </div>
                                <div className="absolute" style={{ left: `${((1 + epsilon) / 1.6) * 100}%` }}>
                                    <div className="text-xs text-center -ml-4">1+ε<br />{(1 + epsilon).toFixed(2)}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-red-600 mb-4">r &lt; 1-ε</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            策略变化<strong>过大</strong>（减少概率太多）
                        </div>
                        <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded">
                            <strong>裁剪到 1-ε</strong>
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            限制减少幅度
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-green-500">
                    <h4 className="text-lg font-bold text-green-600 mb-4">1-ε ≤ r ≤ 1+ε</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            策略变化<strong>适中</strong>
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>不裁剪</strong>
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            正常优化
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-red-600 mb-4">r &gt; 1+ε</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            策略变化<strong>过大</strong>（增加概率太多）
                        </div>
                        <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded">
                            <strong>裁剪到 1+ε</strong>
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            限制增加幅度
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">比率公式</h4>
                <div className="font-mono text-center p-4 bg-amber-50 dark:bg-amber-900/20 rounded">
                    r = π<sub>θ</sub>(a|s) / π<sub>θ<sub>old</sub></sub>(a|s)
                </div>
            </div>
        </div>
    );
}
