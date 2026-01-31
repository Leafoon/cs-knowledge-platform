"use client";

export function CQLObjective() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-amber-50 to-yellow-50 dark:from-slate-900 dark:to-amber-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    CQL 目标函数
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Conservative Q-Learning
                </p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">CQL 损失</h4>
                <div className="font-mono text-center p-4 bg-amber-50 dark:bg-amber-900/20 rounded border-2 border-amber-500 mb-4">
                    L = α · E[log Σ exp(Q(s,a))] - E[Q(s,a)]_data + L_TD
                </div>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                        <strong>项 1:</strong> 降低所有动作的 Q 值（pessimism）
                    </div>
                    <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong>项 2:</strong> 提升数据内动作的 Q 值
                    </div>
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong>项 3:</strong> 标准 TD 误差
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-amber-600 mb-4">保守估计</h4>
                    <div className="text-center p-4 bg-amber-50 dark:bg-amber-900/20 rounded">
                        <div className="font-mono mb-2">Q_CQL ≤ Q*</div>
                        <div className="text-xs text-slate-600 dark:text-slate-400">
                            学习真实 Q 函数的下界
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-yellow-600 mb-4">效果</h4>
                    <div className="text-center p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                        <div className="text-sm">避免 OOD 动作的 Q 值高估</div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">CQL 优势</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 理论保证</strong><br />
                        Q 值下界保证
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 简单实现</strong><br />
                        在标准 Q-learning 上添加正则化
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 鲁棒性</strong><br />
                        对数据质量不敏感
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ SOTA 性能</strong><br />
                        D4RL benchmark 表现优异
                    </div>
                </div>
            </div>
        </div>
    );
}
