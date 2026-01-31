"use client";

export function DecisionTransformerArchitecture() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-sky-50 to-cyan-50 dark:from-slate-900 dark:to-sky-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Decision Transformer 架构
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">序列建模视角</h4>
                <div className="p-4 bg-sky-50 dark:bg-sky-900/20 rounded mb-4">
                    <div className="font-mono text-center">
                        (R̂₁, s₁, a₁, R̂₂, s₂, a₂, ..., R̂ₜ, sₜ, aₜ)
                    </div>
                </div>
                <div className="text-sm text-center text-slate-600 dark:text-slate-400">
                    R̂ₜ = return-to-go = Σ rₜ'
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">架构组件</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-gradient-to-r from-sky-50 to-cyan-50 dark:from-sky-900/20 dark:to-cyan-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-sky-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
                        <div><strong>嵌入层:</strong> MLP(R̂), MLP(s), MLP(a) → 隐藏维度</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-sky-50 to-cyan-50 dark:from-sky-900/20 dark:to-cyan-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-cyan-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
                        <div><strong>位置编码:</strong> 添加时间步信息</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-sky-50 to-cyan-50 dark:from-sky-900/20 dark:to-cyan-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-sky-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
                        <div><strong>Transformer:</strong> 自注意力处理序列</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-sky-50 to-cyan-50 dark:from-sky-900/20 dark:to-cyan-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-cyan-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
                        <div><strong>预测头:</strong> 输出下一个动作 aₜ</div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Return-Conditioned Policy</h4>
                <div className="text-center p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                    <div className="font-mono mb-2">aₜ = Transformer(sₜ, aₜ₋₁, R̂ₜ, ...)</div>
                    <div className="text-sm mt-3 text-slate-600 dark:text-slate-400">
                        条件在期望回报上 → 测试时设置高 return → 生成高回报轨迹
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-sky-600 mb-4">优势</h4>
                    <div className="space-y-2 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            ✅ 无需 RL 特定技巧
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            ✅ 简单稳定训练
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            ✅ 利用 Transformer 能力
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-cyan-600 mb-4">局限</h4>
                    <div className="space-y-2 text-sm">
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            ❌ 依赖数据质量
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            ❌ 计算成本高
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            ❌ 难以超越数据最优
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
