"use client";

export function ScalarizationComparison() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-sky-50 to-cyan-50 dark:from-slate-900 dark:to-sky-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Scalarization 方法对比
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-xl font-bold text-sky-600 mb-4">线性加权</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-sky-50 dark:bg-sky-900/20 rounded">
                            <strong>公式:</strong><br />
                            <div className="font-mono mt-2">r = w₁r₁ + w₂r₂</div>
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>优势:</strong> 简单、易实现
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>局限:</strong> 仅找到凸 Pareto Front
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-xl font-bold text-cyan-600 mb-4">Chebyshev</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                            <strong>公式:</strong><br />
                            <div className="font-mono mt-2">r = min wᵢ(rᵢ - z*ᵢ)</div>
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>优势:</strong> 找到非凸部分
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>优势:</strong> 更均衡权衡
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">选择建议</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-sky-50 dark:bg-sky-900/20 rounded">
                        <strong>线性加权:</strong> 简单任务、凸 Pareto
                    </div>
                    <div className="p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                        <strong>Chebyshev:</strong> 复杂任务、非凸 Pareto
                    </div>
                </div>
            </div>
        </div>
    );
}
