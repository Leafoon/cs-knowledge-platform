"use client";

export function ModelBasedVsModelFree() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-slate-900 dark:to-cyan-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Model-Based vs Model-Free
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-xl font-bold text-blue-600 mb-4">Model-Free RL</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>学习:</strong> 直接学习策略 π 或价值 Q
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>数据:</strong> 仅使用真实环境交互
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>✅ 优势:</strong> 简单、渐近性能高
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>❌ 劣势:</strong> 样本效率低
                        </div>
                        <div className="p-3 bg-slate-50 dark:bg-slate-700 rounded">
                            <strong>示例:</strong> DQN, PPO, SAC
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-cyan-500">
                    <h4 className="text-xl font-bold text-cyan-600 mb-4">Model-Based RL</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                            <strong>学习:</strong> 学习环境模型 P(s'|s,a)
                        </div>
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                            <strong>数据:</strong> 真实 + 想象中的轨迹
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>✅ 优势:</strong> 样本效率高、可规划
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>❌ 劣势:</strong> 模型误差累积
                        </div>
                        <div className="p-3 bg-slate-50 dark:bg-slate-700 rounded">
                            <strong>示例:</strong> Dyna, MBPO, Dreamer
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">性能对比</h4>
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b-2 border-slate-200 dark:border-slate-700">
                            <th className="text-left p-3">维度</th>
                            <th className="text-center p-3">Model-Free</th>
                            <th className="text-center p-3">Model-Based</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3">样本效率</td>
                            <td className="text-center p-3">⭐⭐</td>
                            <td className="text-center p-3 text-cyan-600 font-bold">⭐⭐⭐⭐⭐</td>
                        </tr>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3">渐近性能</td>
                            <td className="text-center p-3 text-blue-600 font-bold">⭐⭐⭐⭐⭐</td>
                            <td className="text-center p-3">⭐⭐⭐⭐</td>
                        </tr>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3">计算复杂度</td>
                            <td className="text-center p-3 text-blue-600 font-bold">低</td>
                            <td className="text-center p-3">高</td>
                        </tr>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3">泛化能力</td>
                            <td className="text-center p-3">⭐⭐⭐</td>
                            <td className="text-center p-3 text-cyan-600 font-bold">⭐⭐⭐⭐</td>
                        </tr>
                        <tr>
                            <td className="p-3">实现难度</td>
                            <td className="text-center p-3 text-blue-600 font-bold">简单</td>
                            <td className="text-center p-3">复杂</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">适用场景</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong className="text-blue-700 dark:text-blue-400">Model-Free 适用:</strong><br />
                        • 样本便宜（模拟器）<br />
                        • 环境复杂难建模<br />
                        • 追求最终性能
                    </div>
                    <div className="p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                        <strong className="text-cyan-700 dark:text-cyan-400">Model-Based 适用:</strong><br />
                        • 样本昂贵（机器人）<br />
                        • 环境相对规律<br />
                        • 需要快速学习
                    </div>
                </div>
            </div>
        </div>
    );
}
