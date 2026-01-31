"use client";

export function ParameterSpaceVsPolicySpace() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-cyan-50 to-teal-50 dark:from-slate-900 dark:to-cyan-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    参数空间 vs 策略空间
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-cyan-600 mb-4">参数空间 θ</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                            <strong>空间:</strong> 欧几里得空间 ℝ<sup>n</sup>
                        </div>
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                            <strong>度量:</strong> ‖Δθ‖₂ (L2 范数)
                        </div>
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                            <strong>梯度:</strong> ∇<sub>θ</sub> J(θ)
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>问题:</strong> 相同Δθ → 不同策略变化
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-teal-500">
                    <h4 className="text-lg font-bold text-teal-600 mb-4">策略空间 π</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                            <strong>空间:</strong> 概率分布空间
                        </div>
                        <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                            <strong>度量:</strong> D<sub>KL</sub>(π‖π') (KL 散度)
                        </div>
                        <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                            <strong>梯度:</strong> F<sup>-1</sup>∇<sub>θ</sub> J(θ)
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>优势:</strong> 参数化不变
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">问题示例</h4>
                <div className="space-y-4 text-sm">
                    <div className="p-4 bg-slate-50 dark:bg-slate-700 rounded">
                        <strong>场景:</strong> Softmax 策略 π(a|s) ∝ exp(θ<sup>T</sup>φ(s,a))
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>‖θ‖ 很大 (后期训练):</strong><br />
                            π 接近确定性<br />
                            Δθ → 策略<strong>剧烈</strong>变化
                        </div>
                        <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                            <strong>‖θ‖ 很小 (早期训练):</strong><br />
                            π 接近均匀分布<br />
                            Δθ → 策略<strong>微小</strong>变化
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">自然梯度的优势</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-teal-50 dark:bg-teal-900/20 rounded">
                        <strong>✅ 参数化不变</strong><br />
                        不依赖神经网络结构
                    </div>
                    <div className="p-4 bg-teal-50 dark:bg-teal-900/20 rounded">
                        <strong>✅ 策略空间最速下降</strong><br />
                        在策略分布上优化
                    </div>
                    <div className="p-4 bg-teal-50 dark:bg-teal-900/20 rounded">
                        <strong>✅ 更好的收敛性</strong><br />
                        自适应步长
                    </div>
                    <div className="p-4 bg-teal-50 dark:bg-teal-900/20 rounded">
                        <strong>✅ 避免协变量偏移</strong><br />
                        重新参数化不影响更新
                    </div>
                </div>
            </div>
        </div>
    );
}
