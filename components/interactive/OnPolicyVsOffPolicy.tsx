"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function OnPolicyVsOffPolicy() {
    const [activeTab, setActiveTab] = useState<"onpolicy" | "offpolicy">("onpolicy");

    const onPolicyFeatures = [
        { icon: "🎯", title: "学习目标", content: "学习当前执行的策略 π" },
        { icon: "🔄", title: "数据来源", content: "使用 π 生成的数据" },
        { icon: "📈", title: "探索策略", content: "ε-greedy（π 本身带探索）" },
        { icon: "✅", title: "收敛性", content: "GLIE 条件下收敛" },
        { icon: "📊", title: "方差", content: "较低（无重要性采样）" },
        { icon: "⚡", title: "样本效率", content: "数据仅用一次" },
    ];

    const offPolicyFeatures = [
        { icon: "🔀", title: "学习目标", content: "学习目标策略 π（贪心）" },
        { icon: "🎲", title: "数据来源", content: "使用行为策略 b 生成" },
        { icon: "🔍", title: "探索策略", content: "b 负责探索，π 负责利用" },
        { icon: "⚖️", title: "收敛性", content: "需要重要性采样修正" },
        { icon: "📉", title: "方差", content: "较高（重要性采样比）" },
        { icon: "♻️", title: "样本效率", content: "数据可复用" },
    ];

    const onPolicyAlgorithms = [
        { name: "MC ε-greedy", description: "蒙特卡洛控制 + ε探索" },
        { name: "SARSA", description: "On-policy TD 控制" },
        { name: "A2C / A3C", description: "Advantage Actor-Critic" },
        { name: "PPO", description: "Proximal Policy Optimization" },
    ];

    const offPolicyAlgorithms = [
        { name: "MC IS", description: "重要性采样蒙特卡洛" },
        { name: "Q-learning", description: "Off-policy TD 控制" },
        { name: "DQN", description: "Deep Q-Network" },
        { name: "SAC", description: "Soft Actor-Critic" },
        { name: "DDPG", description: "Deep Deterministic PG" },
    ];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-sky-50 to-indigo-50 dark:from-slate-900 dark:to-sky-950 rounded-2xl shadow-xl">
            <div className="text-center mb-8">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    On-policy vs Off-policy
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    两种学习范式的完整对比
                </p>
            </div>

            {/* Tab 切换 */}
            <div className="flex justify-center gap-4 mb-8">
                <button
                    onClick={() => setActiveTab("onpolicy")}
                    className={`px-8 py-3 rounded-xl font-bold text-lg transition-all ${activeTab === "onpolicy"
                            ? "bg-sky-600 text-white shadow-lg scale-105"
                            : "bg-sky-100 text-sky-700 dark:bg-sky-900/30 dark:text-sky-300"
                        }`}
                >
                    On-policy
                </button>
                <button
                    onClick={() => setActiveTab("offpolicy")}
                    className={`px-8 py-3 rounded-xl font-bold text-lg transition-all ${activeTab === "offpolicy"
                            ? "bg-indigo-600 text-white shadow-lg scale-105"
                            : "bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300"
                        }`}
                >
                    Off-policy
                </button>
            </div>

            {/* 特性网格 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-8">
                <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">
                    核心特性
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {(activeTab === "onpolicy" ? onPolicyFeatures : offPolicyFeatures).map((feature, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: idx * 0.1 }}
                            className={`p-4 rounded-lg border-2 ${activeTab === "onpolicy"
                                    ? "border-sky-300 bg-sky-50 dark:bg-sky-900/20 dark:border-sky-600"
                                    : "border-indigo-300 bg-indigo-50 dark:bg-indigo-900/20 dark:border-indigo-600"
                                }`}
                        >
                            <div className="text-3xl mb-2">{feature.icon}</div>
                            <div className="font-bold text-slate-800 dark:text-slate-100 mb-1">
                                {feature.title}
                            </div>
                            <div className="text-sm text-slate-600 dark:text-slate-400">
                                {feature.content}
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>

            {/* 典型算法 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-8">
                <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">
                    典型算法
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {(activeTab === "onpolicy" ? onPolicyAlgorithms : offPolicyAlgorithms).map((algo, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.1 }}
                            className={`p-4 rounded-lg border-l-4 ${activeTab === "onpolicy"
                                    ? "border-sky-500 bg-sky-50/50 dark:bg-sky-900/10"
                                    : "border-indigo-500 bg-indigo-50/50 dark:bg-indigo-900/10"
                                }`}
                        >
                            <div className="font-bold text-slate-800 dark:text-slate-100">
                                {algo.name}
                            </div>
                            <div className="text-sm text-slate-600 dark:text-slate-400">
                                {algo.description}
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>

            {/* 流程图 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">
                    学习流程对比
                </h4>
                {activeTab === "onpolicy" ? (
                    <div className="flex flex-col items-center space-y-4">
                        <div className="flex items-center gap-4">
                            <div className="w-40 h-20 bg-sky-500 rounded-lg flex items-center justify-center text-white font-bold">
                                策略 π<br />(ε-greedy)
                            </div>
                            <div className="text-2xl">→</div>
                            <div className="w-40 h-20 bg-green-500 rounded-lg flex items-center justify-center text-white font-bold">
                                生成数据
                            </div>
                            <div className="text-2xl">→</div>
                            <div className="w-40 h-20 bg-blue-500 rounded-lg flex items-center justify-center text-white font-bold">
                                更新 π
                            </div>
                        </div>
                        <div className="text-sm text-slate-600 dark:text-slate-400 max-w-2xl text-center">
                            On-policy：使用当前策略 π 生成数据，学习并改进 π。
                            探索由 π 自身保证（如 ε-greedy）。
                        </div>
                    </div>
                ) : (
                    <div className="flex flex-col items-center space-y-4">
                        <div className="flex flex-col items-center gap-4">
                            <div className="flex items-center gap-4">
                                <div className="w-40 h-20 bg-indigo-500 rounded-lg flex items-center justify-center text-white font-bold">
                                    行为策略 b<br />(探索)
                                </div>
                                <div className="text-2xl">→</div>
                                <div className="w-40 h-20 bg-green-500 rounded-lg flex items-center justify-center text-white font-bold">
                                    生成数据
                                </div>
                            </div>
                            <div className="text-2xl">↓ 重要性采样</div>
                            <div className="w-40 h-20 bg-purple-500 rounded-lg flex items-center justify-center text-white font-bold">
                                目标策略 π<br />(贪心)
                            </div>
                        </div>
                        <div className="text-sm text-slate-600 dark:text-slate-400 max-w-2xl text-center">
                            Off-policy：使用行为策略 b 生成数据，通过重要性采样学习目标策略 π。
                            b 负责探索，π 可以是确定性贪心策略。
                        </div>
                    </div>
                )}
            </div>

            {/* 对比总结 */}
            <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-sky-50 dark:bg-sky-900/20 rounded-lg p-6 border-2 border-sky-300 dark:border-sky-600">
                    <h5 className="font-bold text-sky-800 dark:text-sky-300 mb-3 text-lg">
                        ✅ On-policy 优势
                    </h5>
                    <ul className="space-y-2 text-sm text-sky-700 dark:text-sky-400">
                        <li>• 简单直观，易于实现</li>
                        <li>• 无重要性采样，方差较低</li>
                        <li>• 理论收敛性强（GLIE）</li>
                        <li>• 适合在线学习</li>
                    </ul>
                    <h5 className="font-bold text-sky-800 dark:text-sky-300 mb-2 mt-4">
                        ⚠️ On-policy 劣势
                    </h5>
                    <ul className="space-y-2 text-sm text-sky-700 dark:text-sky-400">
                        <li>• 数据利用率低（仅用一次）</li>
                        <li>• 难以从旧数据学习</li>
                        <li>• 探索与利用难以平衡</li>
                    </ul>
                </div>

                <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6 border-2 border-indigo-300 dark:border-indigo-600">
                    <h5 className="font-bold text-indigo-800 dark:text-indigo-300 mb-3 text-lg">
                        ✅ Off-policy 优势
                    </h5>
                    <ul className="space-y-2 text-sm text-indigo-700 dark:text-indigo-400">
                        <li>• 数据复用（Experience Replay）</li>
                        <li>• 从任意数据学习（人类专家）</li>
                        <li>• 探索与利用分离</li>
                        <li>• 样本效率更高</li>
                    </ul>
                    <h5 className="font-bold text-indigo-800 dark:text-indigo-300 mb-2 mt-4">
                        ⚠️ Off-policy 劣势
                    </h5>
                    <ul className="space-y-2 text-sm text-indigo-700 dark:text-indigo-400">
                        <li>• 重要性采样方差可能很大</li>
                        <li>• 实现复杂度更高</li>
                        <li>• 收敛性较难保证</li>
                    </ul>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 提示：现代深度 RL 中，Off-policy 方法（DQN, SAC）更流行，因为可以使用 Experience Replay
            </div>
        </div>
    );
}
