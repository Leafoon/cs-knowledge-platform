"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

export function RLEcosystemMap() {
    const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

    const categories = [
        {
            id: "model_free",
            name: "Model-Free RL",
            desc: "不学习环境模型，直接由经验学习价值或策略。",
            color: "blue",
            children: [
                { id: "q_learning", name: "Q-Learning", type: "Value-based" },
                { id: "dqn", name: "DQN", type: "Value-based" },
                { id: "policy_gradient", name: "Policy Gradient", type: "Policy-based" },
                { id: "ppo", name: "PPO", type: "Policy-based" },
                { id: "sac", name: "SAC", type: "Actor-Critic" },
            ]
        },
        {
            id: "model_based",
            name: "Model-Based RL",
            desc: "学习环境动态模型 P(s'|s,a)，并在模型中规划。",
            color: "purple",
            children: [
                { id: "dyna", name: "Dyna-Q", type: "Hybrid" },
                { id: "alphazero", name: "AlphaZero", type: "MCTS" },
                { id: "dreamer", name: "Dreamer", type: "Latent Dynamics" },
                { id: "mbpo", name: "MBPO", type: "Dyna-style" },
            ]
        },
        {
            id: "advanced",
            name: "Extension & Frontiers",
            desc: "解决特定挑战的高级方向。",
            color: "green",
            children: [
                { id: "offline", name: "Offline RL", type: "Sample Efficiency" },
                { id: "marl", name: "Multi-Agent RL", type: "Interaction" },
                { id: "meta", name: "Meta-RL", type: "Generalization" },
                { id: "rlhf", name: "RLHF", type: "Alignment" },
            ]
        }
    ];

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-2xl shadow-xl border border-slate-200 dark:border-slate-800">
            <h3 className="text-2xl font-bold text-center mb-8 text-slate-800 dark:text-slate-100">
                强化学习技术生态图谱
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {categories.map((cat) => (
                    <motion.div
                        key={cat.id}
                        className={`relative p-6 rounded-xl border-2 cursor-pointer transition-all ${selectedCategory === cat.id
                                ? `bg-${cat.color}-50 dark:bg-${cat.color}-900/20 border-${cat.color}-500 shadow-lg scale-105 z-10`
                                : `bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:border-${cat.color}-300`
                            }`}
                        onClick={() => setSelectedCategory(cat.id === selectedCategory ? null : cat.id)}
                        layout
                    >
                        <div className={`text-${cat.color}-600 dark:text-${cat.color}-400 font-bold text-xl mb-2 flex items-center justify-between`}>
                            {cat.name}
                            <motion.span
                                animate={{ rotate: selectedCategory === cat.id ? 180 : 0 }}
                                className="text-sm opacity-50"
                            >
                                ▼
                            </motion.span>
                        </div>
                        <p className="text-sm text-slate-500 dark:text-slate-400 mb-4 h-10 line-clamp-2">
                            {cat.desc}
                        </p>

                        <div className="space-y-2">
                            {cat.children.map((child) => (
                                <motion.div
                                    key={child.id}
                                    className={`p-2 rounded-lg text-sm border flex justify-between items-center ${selectedCategory === cat.id
                                            ? "bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700"
                                            : "opacity-60 grayscale bg-slate-100 dark:bg-slate-800"
                                        }`}
                                >
                                    <span className="font-bold text-slate-700 dark:text-slate-300">{child.name}</span>
                                    {selectedCategory === cat.id && (
                                        <span className={`text-[10px] px-2 py-0.5 rounded-full bg-${cat.color}-100 dark:bg-${cat.color}-900/50 text-${cat.color}-700 dark:text-${cat.color}-300`}>
                                            {child.type}
                                        </span>
                                    )}
                                </motion.div>
                            ))}
                        </div>
                    </motion.div>
                ))}
            </div>

            <div className="mt-8 p-4 bg-slate-50 dark:bg-slate-800 rounded-lg text-center text-sm text-slate-500">
                点击上方卡片查看详细算法分类。学习路径建议：
                <span className="font-bold text-blue-600 dark:text-blue-400 mx-2">Model-Free (DQN/PPO)</span>
                →
                <span className="font-bold text-purple-600 dark:text-purple-400 mx-2">Model-Based</span>
                →
                <span className="font-bold text-green-600 dark:text-green-400 mx-2">Advanced</span>
            </div>
        </div>
    );
}
