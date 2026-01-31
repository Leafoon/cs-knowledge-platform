"use client";

export function TaskDistributionSampling() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-cyan-50 to-teal-50 dark:from-slate-900 dark:to-cyan-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    任务分布采样
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">任务分布 p(T)</h4>
                <div className="grid grid-cols-3 gap-4 text-sm">
                    <div className="p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded text-center">
                        <div className="text-2xl mb-2">📦</div>
                        <strong>任务 T₁</strong>
                        <div className="text-xs mt-2">参数: w=1kg</div>
                    </div>
                    <div className="p-4 bg-teal-50 dark:bg-teal-900/20 rounded text-center">
                        <div className="text-2xl mb-2">📦</div>
                        <strong>任务 T₂</strong>
                        <div className="text-xs mt-2">参数: w=2kg</div>
                    </div>
                    <div className="p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded text-center">
                        <div className="text-2xl mb-2">📦</div>
                        <strong>任务 T₃</strong>
                        <div className="text-xs mt-2">参数: w=3kg</div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">元训练流程</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                        <strong>步骤 1:</strong> 从分布采样任务 Tᵢ ~ p(T)
                    </div>
                    <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                        <strong>步骤 2:</strong> 在 Tᵢ 上收集数据并适应
                    </div>
                    <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                        <strong>步骤 3:</strong> 评估适应后的性能
                    </div>
                    <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                        <strong>步骤 4:</strong> 更新元参数
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">测试</h4>
                <div className="text-center p-4 bg-teal-50 dark:bg-teal-900/20 rounded">
                    <div className="text-sm">新任务 T_new ~ p(T)</div>
                    <div className="text-xs mt-2">期望：快速适应（Few-Shot）</div>
                </div>
            </div>
        </div>
    );
}
