"use client";

export function MeanFieldApproximation() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-orange-50 to-amber-50 dark:from-slate-900 dark:to-orange-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    平均场近似
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Mean Field Approximation
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-xl font-bold text-red-600 mb-4">朴素方法</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>Q函数:</strong> Q(s, a¹, ..., aᴺ)
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>复杂度:</strong> O(N × |A|ᴺ)
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>问题:</strong> 指数增长 ✗
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-orange-500">
                    <h4 className="text-xl font-bold text-orange-600 mb-4">平均场方法</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>Q函数:</strong> Q(s, a^i, ā)
                        </div>
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>平均动作:</strong> ā = (1/N)Σaʲ
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>复杂度:</strong> O(N × |A|) ✓
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">应用场景</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded">
                        <strong>交通仿真:</strong> 数千辆车
                    </div>
                    <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded">
                        <strong>人群仿真:</strong> 大规模人群
                    </div>
                </div>
            </div>
        </div>
    );
}
