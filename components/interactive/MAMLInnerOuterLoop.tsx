"use client";

export function MAMLInnerOuterLoop() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-rose-50 to-red-50 dark:from-slate-900 dark:to-rose-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    MAML 内外循环
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">内循环（Task Adaptation）</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-rose-50 dark:bg-rose-900/20 rounded">
                        <strong>输入:</strong> 元参数 θ, 任务数据 D_train
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                        <strong>梯度步:</strong> θ&apos; = θ - α∇L(θ)
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-3 bg-rose-50 dark:bg-rose-900/20 rounded border-2 border-rose-500">
                        <strong>输出:</strong> 适应后参数 θ&apos;
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">外循环（Meta-Update）</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                        <strong>采样:</strong> 批次任务 (T₁, T₂, ..., Tₙ)
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-3 bg-rose-50 dark:bg-rose-900/20 rounded">
                        <strong>内循环:</strong> 对每个任务计算 θ&apos;ᵢ
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                        <strong>评估:</strong> 在测试集上计算损失 L(θ&apos;ᵢ)
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-3 bg-rose-50 dark:bg-rose-900/20 rounded border-2 border-red-500">
                        <strong>元更新:</strong> θ ← θ - β∇Σ L(θ&apos;ᵢ)
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">关键参数</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-rose-50 dark:bg-rose-900/20 rounded">
                        <strong>α (inner_lr):</strong> 内循环学习率
                    </div>
                    <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded">
                        <strong>β (outer_lr):</strong> 外循环学习率（元学习率）
                    </div>
                </div>
            </div>
        </div>
    );
}
