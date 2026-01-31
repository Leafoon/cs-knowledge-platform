"use client";

export function NaturalGradientVisualization() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-amber-50 to-yellow-50 dark:from-slate-900 dark:to-amber-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    自然梯度 vs 普通梯度
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-blue-600 mb-4">普通梯度 ∇<sub>θ</sub> J</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>空间:</strong> 参数空间（欧几里得）
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>方向:</strong> ∂J/∂θ
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>度量:</strong> ‖·‖₂
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>问题:</strong> 依赖参数化
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-amber-500">
                    <h4 className="text-lg font-bold text-amber-600 mb-4">自然梯度 ∇̃<sub>θ</sub> J</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded">
                            <strong>空间:</strong> 策略空间（黎曼）
                        </div>
                        <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded">
                            <strong>方向:</strong> F<sup>-1</sup>∇<sub>θ</sub> J
                        </div>
                        <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded">
                            <strong>度量:</strong> D<sub>KL</sub>
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>优势:</strong> 参数化不变
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">梯度方向对比</h4>
                <div className="relative h-64 bg-slate-50 dark:bg-slate-700 rounded">
                    <svg width="100%" height="100%" viewBox="0 0 400 250">
                        {/* 等高线 (策略空间) */}
                        <ellipse cx="200" cy="125" rx="150" ry="80" fill="none" stroke="rgb(34, 197, 94)" strokeWidth="2" opacity="0.3" />
                        <ellipse cx="200" cy="125" rx="110" ry="60" fill="none" stroke="rgb(34, 197, 94)" strokeWidth="2" opacity="0.3" />
                        <ellipse cx="200" cy="125" rx="70" ry="40" fill="none" stroke="rgb(34, 197, 94)" strokeWidth="2" opacity="0.3" />

                        {/* 起点 */}
                        <circle cx="120" cy="160" r="5" fill="rgb(59, 130, 246)" />
                        <text x="120" y="180" textAnchor="middle" className="text-xs fill-current">θ<tspan fontSize="8" dy="2">0</tspan></text>

                        {/* 普通梯度方向 */}
                        <line x1="120" y1="160" x2="160" y2="140" stroke="rgb(59, 130, 246)" strokeWidth="3" markerEnd="url(#arrowblue)" />
                        <text x="170" y="135" className="text-xs fill-blue-600">普通梯度</text>

                        {/* 自然梯度方向 */}
                        <line x1="120" y1="160" x2="165" y2="125" stroke="rgb(245, 158, 11)" strokeWidth="3" markerEnd="url(#arroworange)" />
                        <text x="175" y="120" className="text-xs fill-amber-600">自然梯度</text>

                        {/* 最优点 */}
                        <circle cx="200" cy="125" r="6" fill="rgb(34, 197, 94)" />
                        <text x="200" y="115" textAnchor="middle" className="text-xs fill-green-600">最优</text>

                        {/* 箭头定义 */}
                        <defs>
                            <marker id="arrowblue" markerWidth="10" markerHeight="10" refX="5" refY="3" orient="auto" markerUnits="strokeWidth">
                                <path d="M0,0 L0,6 L9,3 z" fill="rgb(59, 130, 246)" />
                            </marker>
                            <marker id="arroworange" markerWidth="10" markerHeight="10" refX="5" refY="3" orient="auto" markerUnits="strokeWidth">
                                <path d="M0,0 L0,6 L9,3 z" fill="rgb(245, 158, 11)" />
                            </marker>
                        </defs>
                    </svg>
                </div>
                <div className="mt-3 text-sm text-center text-slate-600 dark:text-slate-400">
                    自然梯度更直接指向策略空间的最优解
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">更新公式对比</h4>
                <div className="space-y-4">
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <div className="font-bold mb-2">普通梯度下降:</div>
                        <div className="font-mono text-xs">
                            θ ← θ + α∇<sub>θ</sub> J(θ)
                        </div>
                    </div>
                    <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded border-2 border-amber-500">
                        <div className="font-bold mb-2">自然梯度下降:</div>
                        <div className="font-mono text-xs">
                            θ ← θ + αF<sup>-1</sup>∇<sub>θ</sub> J(θ)
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
