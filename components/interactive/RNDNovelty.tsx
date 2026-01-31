"use client";

export function RNDNovelty() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-violet-50 dark:from-slate-900 dark:to-purple-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    RND 新颖性检测
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Random Network Distillation
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-purple-600 mb-4">Target Network (固定)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>初始化:</strong> 随机
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>训练:</strong> 不训练（冻结）
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>输出:</strong> f(s) ∈ ℝ<sup>d</sup>
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-violet-500">
                    <h4 className="text-lg font-bold text-violet-600 mb-4">Predictor Network (训练)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                            <strong>初始化:</strong> 随机
                        </div>
                        <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                            <strong>训练:</strong> 拟合 target
                        </div>
                        <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                            <strong>输出:</strong> f̂(s) ∈ ℝ<sup>d</sup>
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">内在奖励公式</h4>
                <div className="font-mono text-center p-4 bg-violet-50 dark:bg-violet-900/20 rounded border-2 border-violet-500">
                    r<sub>intrinsic</sub> = ‖f̂(s) - f(s)‖²
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">为什么有效？</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong className="text-blue-700 dark:text-blue-400">常见状态</strong><br />
                        • Predictor 见过多次<br />
                        • 预测准确<br />
                        • 误差<strong>小</strong> → bonus 小
                    </div>
                    <div className="p-4 bg-pink-50 dark:bg-pink-900/20 rounded border-2 border-pink-500">
                        <strong className="text-pink-700 dark:text-pink-400">新颖状态</strong><br />
                        • Predictor 没见过<br />
                        • 预测不准<br />
                        • 误差<strong>大</strong> → bonus 大
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">RND vs ICM</h4>
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b-2 border-slate-200 dark:border-slate-700">
                            <th className="text-left p-3">特性</th>
                            <th className="text-center p-3">ICM</th>
                            <th className="text-center p-3">RND</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3">需要动作</td>
                            <td className="text-center p-3">✅</td>
                            <td className="text-center p-3 text-green-600 font-bold">❌ 更简单</td>
                        </tr>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3">特征学习</td>
                            <td className="text-center p-3">学习</td>
                            <td className="text-center p-3">随机</td>
                        </tr>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3">实现复杂度</td>
                            <td className="text-center p-3">高</td>
                            <td className="text-center p-3 text-green-600 font-bold">低</td>
                        </tr>
                        <tr>
                            <td className="p-3">效果</td>
                            <td className="text-center p-3">优秀</td>
                            <td className="text-center p-3 text-green-600 font-bold">优秀</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    );
}
