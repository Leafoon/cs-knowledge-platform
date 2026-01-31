"use client";

export function ICMArchitecture() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-cyan-50 to-sky-50 dark:from-slate-900 dark:to-cyan-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    ICM 架构
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Intrinsic Curiosity Module
                </p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">三个组件</h4>
                <div className="space-y-4">
                    <div className="p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                        <div className="flex items-center gap-3 mb-2">
                            <div className="w-10 h-10 bg-cyan-500 text-white rounded-full flex items-center justify-center font-bold">φ</div>
                            <strong>Feature Encoder</strong>
                        </div>
                        <div className="text-sm ml-13">
                            φ(s) → 特征表示<br />
                            过滤无关信息（背景噪声等）
                        </div>
                    </div>

                    <div className="p-4 bg-sky-50 dark:bg-sky-900/20 rounded">
                        <div className="flex items-center gap-3 mb-2">
                            <div className="w-10 h-10 bg-sky-500 text-white rounded-full flex items-center justify-center font-bold">I</div>
                            <strong>Inverse Model</strong>
                        </div>
                        <div className="text-sm ml-13">
                            g(φ<sub>t</sub>, φ<sub>t+1</sub>) → â<sub>t</sub><br />
                            预测导致转移的动作
                        </div>
                    </div>

                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <div className="flex items-center gap-3 mb-2">
                            <div className="w-10 h-10 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">F</div>
                            <strong>Forward Model</strong>
                        </div>
                        <div className="text-sm ml-13">
                            f(φ<sub>t</sub>, a<sub>t</sub>) → φ̂<sub>t+1</sub><br />
                            预测下一特征
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">内在奖励</h4>
                <div className="font-mono text-center p-4 bg-pink-50 dark:bg-pink-900/20 rounded border-2 border-pink-500">
                    r<sub>intrinsic</sub> = η/2 × ‖φ̂<sub>t+1</sub> - φ<sub>t+1</sub>‖²
                </div>
                <div className="mt-4 text-sm text-center text-slate-600 dark:text-slate-400">
                    预测误差 = 新颖性 = 探索奖励
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">训练损失</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-sky-50 dark:bg-sky-900/20 rounded">
                        <strong>Inverse Model Loss:</strong><br />
                        𝓛<sub>I</sub> = ‖â<sub>t</sub> - a<sub>t</sub>‖²
                    </div>
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong>Forward Model Loss:</strong><br />
                        𝓛<sub>F</sub> = ‖φ̂<sub>t+1</sub> - φ<sub>t+1</sub>‖²
                    </div>
                    <div className="p-3 bg-slate-50 dark:bg-slate-700 rounded">
                        <strong>总损失:</strong><br />
                        𝓛 = λ<sub>I</sub> 𝓛<sub>I</sub> + (1-β<sub>I</sub>) 𝓛<sub>F</sub>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">关键优势</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 学习相关特征</strong><br />
                        Inverse model 强制 φ 只编码动作相关信息
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 过滤随机性</strong><br />
                        无法控制的噪声不会被编码
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 端到端学习</strong><br />
                        与策略一起训练
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 稀疏奖励有效</strong><br />
                        在Montezuma's Revenge等任务表现优异
                    </div>
                </div>
            </div>
        </div>
    );
}
