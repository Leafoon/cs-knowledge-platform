"use client";

export function TransferLearningFlow() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-slate-900 dark:to-green-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    迁移学习流程
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">迁移学习步骤</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
                        <div><strong>源任务训练:</strong> 在源任务上训练策略 π_source</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-emerald-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
                        <div><strong>特征提取:</strong> 复用编码器权重</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
                        <div><strong>Fine-tuning:</strong> 在目标任务上微调</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-emerald-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
                        <div><strong>评估:</strong> 在目标任务上测试性能</div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-green-600 mb-4">源任务</h4>
                    <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <div className="text-sm">已训练的策略 πₛₒᵤᵣcₑ</div>
                        <div className="text-xs mt-2">大量数据、完整训练</div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-emerald-600 mb-4">目标任务</h4>
                    <div className="text-center p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded">
                        <div className="text-sm">微调的策略 πₜₐᵣ_ggₑₜ</div>
                        <div className="text-xs mt-2">少量数据、快速适应</div>
                    </div>
                </div>
            </div>
        </div>
    );
}
