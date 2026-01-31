"use client";

export function FisherInformationMatrix() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-violet-50 to-indigo-50 dark:from-slate-900 dark:to-violet-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Fisher Information Matrix
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Fisher 矩阵定义</h4>
                <div className="font-mono text-center p-4 bg-violet-50 dark:bg-violet-900/20 rounded border-2 border-violet-500">
                    F(θ) = E<sub>s,a~π</sub>[∇<sub>θ</sub> log π(a|s) · ∇<sub>θ</sub> log π(a|s)<sup>T</sup>]
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-violet-600 mb-4">几何解释</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                            <strong>黎曼度量:</strong> 定义策略空间的距离
                        </div>
                        <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                            <strong>局部曲率:</strong> 度量参数空间的弯曲程度
                        </div>
                        <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                            <strong>KL 散度:</strong> D<sub>KL</sub> ≈ ½ΔθF(θ)Δθ
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-indigo-600 mb-4">性质</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>对称正定:</strong> F = F<sup>T</sup>, F {'>'} 0
                        </div>
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>Hessian:</strong> F = ∇²<sub>θ</sub> D<sub>KL</sub>
                        </div>
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>信息几何:</strong> 策略流形的度量张量
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">自然梯度</h4>
                <div className="space-y-4">
                    <div className="p-4 bg-slate-50 dark:bg-slate-700 rounded">
                        <div className="font-mono text-center mb-2">
                            ∇̃<sub>θ</sub> J = F<sup>-1</sup> ∇<sub>θ</sub> J
                        </div>
                        <div className="text-sm text-center text-slate-600 dark:text-slate-400">
                            用 Fisher 矩阵的逆预条件普通梯度
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                            <strong>普通梯度:</strong><br />
                            参数空间最速下降
                        </div>
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded border-2 border-indigo-500">
                            <strong>自然梯度:</strong><br />
                            策略空间最速下降
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">计算挑战与解决方案</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded">
                        <strong>❌ 直接求逆:</strong><br />
                        O(n³) 复杂度<br />
                        n = 参数数量
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong>✅ 共轭梯度法:</strong><br />
                        O(kn) 复杂度<br />
                        k = 迭代次数 ≪ n
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong>✅ K-FAC:</strong><br />
                        Kronecker 分解<br />
                        利用网络结构
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong>✅ 对角近似:</strong><br />
                        只用 diag(F)<br />
                        类似 Adam
                    </div>
                </div>
            </div>
        </div>
    );
}
