"use client";

export function ParetoFrontVisualization() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-amber-50 to-yellow-50 dark:from-slate-900 dark:to-amber-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Pareto Front 可视化
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">双目标空间</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-3">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded border-2 border-green-500">
                            <strong>Pareto 最优</strong><br />
                            <div className="text-sm mt-2">策略 A: (安全=9, 速度=5)</div>
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded border-2 border-green-500">
                            <strong>Pareto 最优</strong><br />
                            <div className="text-sm mt-2">策略 B: (安全=7, 速度=8)</div>
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded border-2 border-green-500">
                            <strong>Pareto 最优</strong><br />
                            <div className="text-sm mt-2">策略 C: (安全=5, 速度=9)</div>
                        </div>
                    </div>

                    <div className="space-y-3">
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded border-2 border-red-500">
                            <strong>被支配</strong><br />
                            <div className="text-sm mt-2">策略 D: (安全=6, 速度=6)</div>
                            <div className="text-xs mt-1 text-red-600">被 B 支配</div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">Pareto 支配定义</h4>
                <div className="text-center p-4 bg-amber-50 dark:bg-amber-900/20 rounded">
                    <div className="font-mono mb-2">策略 A 支配 策略 B ⟺</div>
                    <div className="text-sm">A 在所有目标上 ≥ B，且至少一个目标 &gt; B</div>
                </div>
            </div>
        </div>
    );
}
