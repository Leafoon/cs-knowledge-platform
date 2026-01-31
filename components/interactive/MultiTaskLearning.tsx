"use client";

export function MultiTaskLearning() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-slate-900 dark:to-blue-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    多任务学习架构
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">共享表示学习</h4>
                <div className="space-y-3">
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong>输入:</strong> 状态 s
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded border-2 border-indigo-500">
                        <strong>共享编码器</strong><br />
                        <div className="text-sm mt-2">学习通用特征表示</div>
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="grid grid-cols-3 gap-3">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded text-center">
                            <strong>任务 1 头</strong>
                        </div>
                        <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded text-center">
                            <strong>任务 2 头</strong>
                        </div>
                        <div className="p-3 bg-pink-50 dark:bg-pink-900/20 rounded text-center">
                            <strong>任务 3 头</strong>
                        </div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-green-600 mb-4">优势</h4>
                    <div className="space-y-2 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            ✅ 知识共享
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            ✅ 样本效率高
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            ✅ 泛化能力强
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-red-600 mb-4">挑战</h4>
                    <div className="space-y-2 text-sm">
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            ❌ 任务干扰
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            ❌ 负迁移
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            ❌ 权衡困难
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
