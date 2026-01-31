"use client";

export function CTDEArchitecture() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    CTDE 架构图
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Centralized Training Decentralized Execution
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-indigo-500">
                    <h4 className="text-xl font-bold text-indigo-600 mb-4">训练阶段（集中）</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>Critic输入:</strong> 全局状态s + 所有动作a
                        </div>
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>信息:</strong> 完整信息可用
                        </div>
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>优势:</strong> 更好的价值估计
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-purple-500">
                    <h4 className="text-xl font-bold text-purple-600 mb-4">执行阶段（分散）</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>Actor输入:</strong> 局部观测oⁱ
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>信息:</strong> 仅局部可用
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>优势:</strong> 满足分散约束
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">CTDE 原理</h4>
                <div className="text-center p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                    <div className="text-sm">训练时利用全局信息 + 执行时满足分散约束</div>
                </div>
            </div>
        </div>
    );
}
