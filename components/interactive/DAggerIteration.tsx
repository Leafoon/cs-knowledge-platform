"use client";

export function DAggerIteration() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-emerald-50 to-green-50 dark:from-slate-900 dark:to-emerald-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    DAgger 迭代流程
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Dataset Aggregation
                </p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">DAgger 循环</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-gradient-to-r from-emerald-50 to-green-50 dark:from-emerald-900/20 dark:to-green-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-emerald-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
                        <div><strong>策略收集:</strong> 用当前策略 π 收集轨迹</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-emerald-50 to-green-50 dark:from-emerald-900/20 dark:to-green-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
                        <div><strong>专家标注:</strong> 对每个状态 s，查询专家动作 a*</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-emerald-50 to-green-50 dark:from-emerald-900/20 dark:to-green-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-emerald-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
                        <div><strong>数据聚合:</strong> 添加 (s, a*) 到数据集 D</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-emerald-50 to-green-50 dark:from-emerald-900/20 dark:to-green-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
                        <div><strong>重新训练:</strong> 用聚合数据集 D 训练新策略</div>
                    </div>
                    <div className="flex justify-center text-2xl font-bold text-green-600">
                        ↻ 重复迭代
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-emerald-600 mb-4">vs Behavioral Cloning</h4>
                    <div className="space-y-2 text-sm">
                        <div className="p-3 bg-emerald-50 dark:bg-emerald-900/20 rounded">
                            <strong>BC:</strong> 仅用专家轨迹训练
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded border-2 border-green-500">
                            <strong>DAgger:</strong> 用策略访问的状态训练
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-green-600 mb-4">性能保证</h4>
                    <div className="text-center">
                        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                            <div className="font-mono mb-2">误差 = O(εT)</div>
                            <div className="text-xs text-slate-600 dark:text-slate-400">
                                vs BC 的 O(εT²)
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">关键优势</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 缓解分布漂移</strong><br />
                        策略访问的状态被加入数据集
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 理论保证</strong><br />
                        误差线性而非二次
                    </div>
                </div>
            </div>
        </div>
    );
}
