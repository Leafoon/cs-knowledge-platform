"use client";

export function MetaLearningConcept() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-slate-900 dark:to-purple-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    元学习概念图
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Learning to Learn
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-xl font-bold text-blue-600 mb-4">传统 RL</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>每个任务:</strong> 从头学习
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>数据需求:</strong> 大量交互
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>适应速度:</strong> 慢
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-purple-500">
                    <h4 className="text-xl font-bold text-purple-600 mb-4">元强化学习</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>任务分布:</strong> 学习如何学习
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>数据需求:</strong> 少量交互
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>适应速度:</strong> 快（Few-Shot）
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">元学习目标</h4>
                <div className="text-center p-4 bg-pink-50 dark:bg-pink-900/20 rounded font-mono">
                    学习参数 θ，使得少量梯度步就能适应新任务
                </div>
            </div>
        </div>
    );
}
