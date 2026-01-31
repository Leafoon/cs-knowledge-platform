"use client";

export function TrustRegionVisualization() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-slate-900 dark:to-blue-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Trust Region 可视化
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-red-600 mb-4">❌ 无约束优化</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            步长过大 → 性能崩溃
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            步长过小 → 学习缓慢
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            难以选择合适的学习率
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-green-500">
                    <h4 className="text-lg font-bold text-green-600 mb-4">✅ Trust Region</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            KL 约束限制策略变化
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            保证单调改进
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            稳定、可预测的训练
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">TRPO 优化问题</h4>
                <div className="space-y-4">
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded border-l-4 border-blue-500">
                        <strong>目标:</strong>
                        <div className="font-mono text-sm mt-2">
                            max E[(π<sub>new</sub>/π<sub>old</sub>) · A]
                        </div>
                    </div>
                    <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded border-l-4 border-indigo-500">
                        <strong>约束:</strong>
                        <div className="font-mono text-sm mt-2">
                            E[D<sub>KL</sub>(π<sub>old</sub> || π<sub>new</sub>)] ≤ δ
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">Trust Region 示意图</h4>
                <div className="relative h-48 bg-slate-50 dark:bg-slate-700 rounded">
                    <svg width="100%" height="100%" viewBox="0 0 400 200">
                        {/* 当前策略 */}
                        <circle cx="200" cy="100" r="5" fill="rgb(59, 130, 246)" />
                        <text x="200" y="90" textAnchor="middle" className="text-xs fill-current">π<tspan fontSize="8" dy="2">old</tspan></text>

                        {/* Trust Region */}
                        <circle cx="200" cy="100" r="60" fill="none" stroke="rgb(99, 102, 241)" strokeWidth="2" strokeDasharray="5,5" />
                        <text x="200" y="170" textAnchor="middle" className="text-xs fill-current">Trust Region (KL ≤ δ)</text>

                        {/* 安全更新 */}
                        <circle cx="230" cy="110" r="4" fill="rgb(34, 197, 94)" />
                        <text x="230" y="128" textAnchor="middle" className="text-xs fill-current text-green-600">✓ 安全</text>

                        {/* 危险更新 */}
                        <circle cx="290" cy="80" r="4" fill="rgb(239, 68, 68)" />
                        <text x="290" y="68" textAnchor="middle" className="text-xs fill-current text-red-600">✗ 危险</text>
                    </svg>
                </div>
            </div>
        </div>
    );
}
