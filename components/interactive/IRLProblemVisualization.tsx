"use client";

export function IRLProblemVisualization() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-slate-900 dark:to-purple-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    IRL 问题可视化
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-xl font-bold text-blue-600 mb-4">正向 RL</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded flex items-center gap-2">
                            <strong>输入:</strong> 奖励函数 R(s,a)
                        </div>
                        <div className="flex justify-center text-4xl">↓</div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded flex items-center gap-2">
                            <strong>学习:</strong> 策略 π(a|s)
                        </div>
                        <div className="flex justify-center text-4xl">↓</div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>输出:</strong> 最优策略 π*
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-purple-500">
                    <h4 className="text-xl font-bold text-purple-600 mb-4">逆向 RL (IRL)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded flex items-center gap-2">
                            <strong>输入:</strong> 专家演示 D
                        </div>
                        <div className="flex justify-center text-4xl">↓</div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded flex items-center gap-2">
                            <strong>推断:</strong> 奖励函数 R(s,a)
                        </div>
                        <div className="flex justify-center text-4xl">↓</div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>输出:</strong> 学习的奖励 R
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">为什么需要 IRL？</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded">
                        <strong className="text-red-700 dark:text-red-400">❌ 奖励难设计</strong><br/>
                        复杂任务的奖励函数很难手工编写
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 专家易获得</strong><br/>
                        专家演示比奖励函数更容易收集
                    </div>
                    <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded">
                        <strong className="text-red-700 dark:text-red-400">❌ 意图不明</strong><br/>
                        难以形式化表达任务目标
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 理解目标</strong><br/>
                        IRL 学习任务的真正目标
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">应用示例</h4>
                <div className="space-y-2 text-sm">
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong>🚗 自动驾驶:</strong> 从人类驾驶数据学习舒适性、安全性偏好
                    </div>
                    <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                        <strong>🤖 机器人:</strong> 从人类操作学习抓取、组装等任务目标
                    </div>
                    <div className="p-3 bg-pink-50 dark:bg-pink-900/20 rounded">
                        <strong>🎮 游戏 AI:</strong> 从职业玩家录像学习游戏策略
                    </div>
                </div>
            </div>
        </div>
    );
}
