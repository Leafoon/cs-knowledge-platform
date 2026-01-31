"use client";

export function DuelingDQNDecomposition() {
    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-amber-50 to-yellow-50 dark:from-slate-900 dark:to-amber-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Dueling DQN 架构分解
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Q 值分解</h4>
                <div className="font-mono text-center text-lg p-4 bg-amber-50 dark:bg-amber-900/20 rounded">
                    Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-blue-600 mb-4">V(s) - 状态价值</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>含义:</strong> 处于状态 s 的价值
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>作用:</strong> 评估状态本身的好坏
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>独立于:</strong> 具体动作选择
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-green-600 mb-4">A(s,a) - 优势函数</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>含义:</strong> 选择 a 相对平均的优势
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>作用:</strong> 区分不同动作的相对好坏
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>特性:</strong> mean(A) = 0 (约束)
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 Dueling 架构在动作价值相近时更稳定，更快学习状态价值
            </div>
        </div>
    );
}
