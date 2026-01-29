"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function RLEcosystemMap() {
    const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

    const categories = [
        {
            id: "algorithms",
            name: "核心算法",
            color: "#6366f1",
            items: [
                { name: "DQN", desc: "深度Q网络" },
                { name: "PPO", desc: "近端策略优化" },
                { name: "SAC", desc: "软演员-评论家" },
                { name: "TD3", desc: "双延迟DDPG" },
            ]
        },
        {
            id: "methods",
            name: "学习范式",
            color: "#8b5cf6",
            items: [
                { name: "Value-Based", desc: "基于价值" },
                { name: "Policy-Based", desc: "基于策略" },
                { name: "Actor-Critic", desc: "演员-评论家" },
                { name: "Model-Based", desc: "基于模型" },
            ]
        },
        {
            id: "applications",
            name: "应用领域",
            color: "#ec4899",
            items: [
                { name: "游戏AI", desc: "AlphaGo, OpenAI Five" },
                { name: "机器人", desc: "控制与操作" },
                { name: "LLM对齐", desc: "RLHF, DPO" },
                { name: "自动驾驶", desc: "决策规划" },
            ]
        },
        {
            id: "environments",
            name: "环境平台",
            color: "#10b981",
            items: [
                { name: "Gymnasium", desc: "标准RL环境" },
                { name: "MuJoCo", desc: "物理仿真" },
                { name: "Atari", desc: "视觉游戏" },
                { name: "Procgen", desc: "泛化测试" },
            ]
        },
        {
            id: "frameworks",
            name: "开发框架",
            color: "#f59e0b",
            items: [
                { name: "Stable-Baselines3", desc: "PyTorch实现" },
                { name: "RLlib", desc: "Ray分布式" },
                { name: "CleanRL", desc: "简洁实现" },
                { name: "Acme", desc: "DeepMind框架" },
            ]
        },
        {
            id: "frontiers",
            name: "前沿方向",
            color: "#06b6d4",
            items: [
                { name: "Offline RL", desc: "离线强化学习" },
                { name: "Multi-Agent", desc: "多智能体" },
                { name: "Meta-RL", desc: "元强化学习" },
                { name: "Safe RL", desc: "安全强化学习" },
            ]
        },
    ];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
            <div className="text-center mb-8">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    强化学习生态全景图
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    点击类别查看详细信息
                </p>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
                {categories.map((category) => (
                    <motion.button
                        key={category.id}
                        onClick={() => setSelectedCategory(
                            selectedCategory === category.id ? null : category.id
                        )}
                        className="relative p-6 rounded-xl border-2 transition-all duration-300 hover:scale-105"
                        style={{
                            borderColor: selectedCategory === category.id ? category.color : "#e2e8f0",
                            backgroundColor: selectedCategory === category.id
                                ? `${category.color}15`
                                : "white",
                        }}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                    >
                        <div
                            className="w-12 h-12 rounded-full mx-auto mb-3 flex items-center justify-center text-white font-bold text-xl"
                            style={{ backgroundColor: category.color }}
                        >
                            {category.name.charAt(0)}
                        </div>
                        <div className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                            {category.name}
                        </div>
                    </motion.button>
                ))}
            </div>

            {selectedCategory && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg"
                >
                    <h4 className="text-xl font-bold mb-4 text-slate-800 dark:text-slate-100">
                        {categories.find(c => c.id === selectedCategory)?.name}
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {categories
                            .find(c => c.id === selectedCategory)
                            ?.items.map((item, idx) => (
                                <motion.div
                                    key={idx}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: idx * 0.1 }}
                                    className="p-4 rounded-lg bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600"
                                >
                                    <div className="font-semibold text-slate-800 dark:text-slate-100 mb-1">
                                        {item.name}
                                    </div>
                                    <div className="text-sm text-slate-600 dark:text-slate-400">
                                        {item.desc}
                                    </div>
                                </motion.div>
                            ))}
                    </div>
                </motion.div>
            )}

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 提示：强化学习是一个庞大的生态系统，涵盖理论、算法、应用和工具
            </div>
        </div>
    );
}
