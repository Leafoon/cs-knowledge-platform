"use client";

export function OptionsFramework() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Options 框架
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">
                    Option 定义 o = ⟨I, π, β⟩
                </h4>
                <div className="space-y-4">
                    <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                        <strong className="text-indigo-700 dark:text-indigo-400">I (Initiation Set)</strong><br />
                        <div className="text-sm mt-2">
                            可以启动 option 的状态集合：I ⊆ S
                        </div>
                    </div>
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong className="text-blue-700 dark:text-blue-400">π (Option Policy)</strong><br />
                        <div className="text-sm mt-2">
                            option 的内部策略：π: S×A → [0,1]
                        </div>
                    </div>
                    <div className="p-4 bg-sky-50 dark:bg-sky-900/20 rounded">
                        <strong className="text-sky-700 dark:text-sky-400">β (Termination)</strong><br />
                        <div className="text-sm mt-2">
                            在状态 s 终止的概率：β: S → [0,1]
                        </div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-indigo-600 mb-4">示例：导航到门口</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>I:</strong> 任何室内状态
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>π:</strong> 朝门移动的策略
                        </div>
                        <div className="p-3 bg-sky-50 dark:bg-sky-900/20 rounded">
                            <strong>β(s):</strong> 1 if s在门口, else 0
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-blue-600 mb-4">Semi-MDP</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>状态:</strong> s ∈ S
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>动作:</strong> o ∈ O (options)
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>转移:</strong> 执行 option 直到终止
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>奖励:</strong> 累积奖励
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Option-Critic 算法</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-slate-50 dark:bg-slate-700 rounded">
                        <strong>High-level 策略:</strong>
                        <div className="font-mono text-center mt-2">
                            π<sub>Ω</sub>(o|s) = softmax(Q<sub>Ω</sub>(s,o))
                        </div>
                    </div>
                    <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                        <strong>Option 策略梯度:</strong>
                        <div className="font-mono text-center mt-2">
                            ∇<sub>θ<sub>o</sub></sub> J = 𝔼[∇ log π<sub>o</sub>(a|s) Q<sub>U</sub>(s,o,a)]
                        </div>
                    </div>
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong>Termination 梯度:</strong>
                        <div className="font-mono text-center mt-2">
                            ∇<sub>θ<sub>β</sub></sub> J = 𝔼[∇ β<sub>o</sub>(s) A<sub>Ω</sub>(s,o)]
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">Options 的优势</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 时间抽象</strong><br />
                        宏动作简化决策
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 技能复用</strong><br />
                        option 可迁移到新任务
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 加速探索</strong><br />
                        大步跳跃覆盖更多状态
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 可学习</strong><br />
                        Option-Critic 端到端学习
                    </div>
                </div>
            </div>
        </div>
    );
}
