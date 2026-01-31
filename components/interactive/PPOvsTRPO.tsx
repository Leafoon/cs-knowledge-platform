"use client";

export function PPOvsTRPO() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-teal-50 to-cyan-50 dark:from-slate-900 dark:to-teal-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    PPO vs TRPO 对比
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-blue-600 mb-4">TRPO</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>约束方式:</strong> KL 散度硬约束
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>优化:</strong> 共轭梯度 + Line Search
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>实现:</strong> ~500 行代码
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>速度:</strong> 慢 (~10-20倍计算)
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>优势:</strong> 理论保证单调改进
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>劣势:</strong> 实现复杂，调试困难
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-teal-500">
                    <h4 className="text-lg font-bold text-teal-600 mb-4">PPO ⭐</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                            <strong>约束方式:</strong> Clipping 软约束
                        </div>
                        <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                            <strong>优化:</strong> 普通 SGD
                        </div>
                        <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                            <strong>实现:</strong> ~150 行代码
                        </div>
                        <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                            <strong>速度:</strong> 快 (一阶优化)
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>优势:</strong> 简单、快速、实用
                        </div>
                        <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                            <strong>权衡:</strong> 无严格理论保证
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">性能对比</h4>
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b-2 border-slate-200 dark:border-slate-700">
                            <th className="text-left p-3">指标</th>
                            <th className="text-center p-3">TRPO</th>
                            <th className="text-center p-3">PPO</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3">样本效率</td>
                            <td className="text-center p-3">⭐⭐⭐⭐</td>
                            <td className="text-center p-3 text-teal-600 font-bold">⭐⭐⭐⭐⭐</td>
                        </tr>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3">计算效率</td>
                            <td className="text-center p-3">⭐⭐</td>
                            <td className="text-center p-3 text-teal-600 font-bold">⭐⭐⭐⭐⭐</td>
                        </tr>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3">实现难度</td>
                            <td className="text-center p-3">困难</td>
                            <td className="text-center p-3 text-teal-600 font-bold">简单</td>
                        </tr>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3">稳定性</td>
                            <td className="text-center p-3">⭐⭐⭐⭐⭐</td>
                            <td className="text-center p-3">⭐⭐⭐⭐</td>
                        </tr>
                        <tr>
                            <td className="p-3">应用广泛度</td>
                            <td className="text-center p-3">⭐⭐</td>
                            <td className="text-center p-3 text-teal-600 font-bold">⭐⭐⭐⭐⭐</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 PPO 保留了 TRPO 的核心思想，但大幅简化实现，成为现代 RL 的主力算法
            </div>
        </div>
    );
}
