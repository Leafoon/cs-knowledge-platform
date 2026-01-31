"use client";

export function ValueDecompositionComparison() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-emerald-50 to-green-50 dark:from-slate-900 dark:to-emerald-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    价值分解方法对比
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-xl font-bold text-emerald-600 mb-4">VDN</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-emerald-50 dark:bg-emerald-900/20 rounded">
                            <strong>公式:</strong>
                            <div className="font-mono mt-2">Q_tot = Σ Q^i</div>
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>优势:</strong> 简单
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>局限:</strong> 仅可加性
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-green-500">
                    <h4 className="text-xl font-bold text-green-600 mb-4">QMIX</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>公式:</strong>
                            <div className="font-mono mt-2">Q_tot = f_mix(Q¹,...,Qᴺ; s)</div>
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>单调性:</strong> ∂Q_tot/∂Q^i ≥ 0
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>优势:</strong> 更强表达力
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">分散执行</h4>
                <div className="text-center p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded">
                    <div className="text-sm">argmax_a^i Q^i = argmax_a^i Q_tot（单调性保证）</div>
                </div>
            </div>
        </div>
    );
}
