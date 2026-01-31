"use client";

export function AdvantageEstimation() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-slate-900 dark:to-emerald-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Advantage 函数估计
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">核心公式</h4>
                <div className="font-mono text-center text-lg p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded border-2 border-emerald-500">
                    A(s,a) = Q(s,a) - V(s)
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-blue-600 mb-4">Q(s,a)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>动作价值</strong><br />
                            在状态 s 选择动作 a 的价值
                        </div>
                        <div className="text-center text-2xl font-bold">8</div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-purple-600 mb-4">V(s)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>状态价值</strong><br />
                            状态 s 的平均价值
                        </div>
                        <div className="text-center text-2xl font-bold">5</div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-emerald-500">
                    <h4 className="text-lg font-bold text-emerald-600 mb-4">A(s,a)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-emerald-50 dark:bg-emerald-900/20 rounded">
                            <strong>优势函数</strong><br />
                            相对于平均的优势
                        </div>
                        <div className="text-center text-2xl font-bold text-emerald-600">+3</div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">为什么使用 Advantage？</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 降低方差</strong><br />
                        中心化价值，减少震荡
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 保持期望</strong><br />
                        E[A(s,a)] = 0 不改变梯度期望
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 关注相对差异</strong><br />
                        只看动作间的相对好坏
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 更稳定训练</strong><br />
                        方差降低 30-50%
                    </div>
                </div>
            </div>
        </div>
    );
}
