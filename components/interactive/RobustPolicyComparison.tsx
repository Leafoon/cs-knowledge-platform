"use client";

export function RobustPolicyComparison() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-violet-50 dark:from-slate-900 dark:to-purple-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    鲁棒策略对比
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-xl font-bold text-blue-600 mb-4">标准策略</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>训练:</strong> 固定环境
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>性能:</strong> 对扰动敏感 ✗
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-purple-500">
                    <h4 className="text-xl font-bold text-purple-600 mb-4">鲁棒策略</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>训练:</strong> Domain Randomization
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>性能:</strong> 对扰动鲁棒 ✓
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">对抗训练</h4>
                <div className="text-center p-4 bg-violet-50 dark:bg-violet-900/20 rounded font-mono text-sm">
                    max_π min_ω J(π, ω)
                </div>
                <div className="text-center mt-2 text-xs text-slate-600 dark:text-slate-400">
                    在最坏情况下优化
                </div>
            </div>
        </div>
    );
}
