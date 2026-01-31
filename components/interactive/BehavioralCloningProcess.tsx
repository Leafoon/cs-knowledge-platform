"use client";

export function BehavioralCloningProcess() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-slate-900 dark:to-blue-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    行为克隆过程
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">BC 训练流程</h4>
                <div className="space-y-3">
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong>步骤 1: 收集专家数据</strong>
                        <div className="text-sm mt-2">D = (s₁, a₁), (s₂, a₂), ..., (sₙ, aₙ)</div>
                    </div>
                    <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                        <strong>步骤 2: 监督学习</strong>
                        <div className="text-sm mt-2">最小化 L(θ) = -Σ log πθ(a|s)</div>
                    </div>
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong>步骤 3: 策略部署</strong>
                        <div className="text-sm mt-2">用学习的 πθ 执行任务</div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-green-600 mb-4">优势</h4>
                    <div className="space-y-2 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            ✅ 简单易实现
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            ✅ 无需环境交互
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            ✅ 快速训练
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-red-600 mb-4">局限</h4>
                    <div className="space-y-2 text-sm">
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            ❌ 分布漂移
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            ❌ 误差累积 O(εT²)
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            ❌ 缺乏泛化
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
