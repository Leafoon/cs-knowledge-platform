"use client";

export function IMPALAFlow() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-slate-900 dark:to-green-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    IMPALA 数据流
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Importance Weighted Actor-Learner Architecture
                </p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">异步 Actor-Learner 模式</h4>
                <div className="space-y-4">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <div className="flex items-center gap-3 mb-2">
                            <div className="w-10 h-10 bg-green-500 text-white rounded-full flex items-center justify-center font-bold">A</div>
                            <strong>Actors (异步执行)</strong>
                        </div>
                        <div className="text-sm ml-13 space-y-2">
                            <div>• 使用 <strong>旧策略 μ</strong> 收集数据</div>
                            <div>• 不等待 Learner 更新</div>
                            <div>• 发送 <strong>轨迹片段</strong> (例如20步)</div>
                            <div>• 包含 log μ(a|s) 用于重要性权重</div>
                        </div>
                    </div>

                    <div className="flex justify-center">
                        <div className="text-2xl">↓</div>
                    </div>

                    <div className="p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded">
                        <div className="flex items-center gap-3 mb-2">
                            <div className="w-10 h-10 bg-emerald-500 text-white rounded-full flex items-center justify-center font-bold">L</div>
                            <strong>Learner (GPU 训练)</strong>
                        </div>
                        <div className="text-sm ml-13 space-y-2">
                            <div>• 接收多个 Actors 的轨迹</div>
                            <div>• 使用 <strong>V-trace</strong> 修正 off-policy</div>
                            <div>• 更新策略 π (比 μ 更新)</div>
                            <div>• 广播新参数给 Actors</div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-red-600 mb-4">❌ 无 V-trace</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>问题:</strong> 策略滞后 (μ ≠ π)
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>后果:</strong> 高偏差或高方差
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>结果:</strong> 训练不稳定
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-green-500">
                    <h4 className="text-lg font-bold text-green-600 mb-4">✅ 有 V-trace</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>方法:</strong> 重要性权重修正
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>截断:</strong> 避免高方差
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>结果:</strong> 稳定训练
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">IMPALA vs Ape-X</h4>
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b-2 border-slate-200 dark:border-slate-700">
                            <th className="text-left p-3">特性</th>
                            <th className="text-center p-3">Ape-X</th>
                            <th className="text-center p-3">IMPALA</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3">基础算法</td>
                            <td className="text-center p-3">DQN (off-policy)</td>
                            <td className="text-center p-3 text-green-600 font-bold">Actor-Critic</td>
                        </tr>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3">Replay Buffer</td>
                            <td className="text-center p-3">✅ 有 (PER)</td>
                            <td className="text-center p-3">❌ 无</td>
                        </tr>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3">修正方法</td>
                            <td className="text-center p-3">优先级采样</td>
                            <td className="text-center p-3 text-green-600 font-bold">V-trace</td>
                        </tr>
                        <tr>
                            <td className="p-3">适用任务</td>
                            <td className="text-center p-3">Atari</td>
                            <td className="text-center p-3 text-green-600 font-bold">Atari, 3D 环境</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    );
}
