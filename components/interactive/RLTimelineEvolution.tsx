"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function RLTimelineEvolution() {
    const [selectedEra, setSelectedEra] = useState<number | null>(null);

    const timeline = [
        {
            era: "1950s-1980s",
            title: "理论奠基",
            color: "#6366f1",
            icon: "📚",
            milestones: [
                { year: "1957", event: "Bellman 提出动态规划", impact: "奠定 RL 数学基础" },
                { year: "1972", event: "Klopf 提出 Hedonistic Neuron", impact: "神经科学启发" },
                { year: "1988", event: "Sutton 提出 TD(λ)", impact: "时序差分学习" },
            ]
        },
        {
            era: "1989-2012",
            title: "表格方法",
            color: "#8b5cf6",
            icon: "📊",
            milestones: [
                { year: "1989", event: "Watkins 提出 Q-learning", impact: "Off-policy 学习突破" },
                { year: "1992", event: "Tesauro 的 TD-Gammon", impact: "西洋双陆棋大师级" },
                { year: "1996", event: "Sutton & Barto 第1版教材", impact: "RL 标准教材诞生" },
            ]
        },
        {
            era: "2013-2015",
            title: "深度革命",
            color: "#ec4899",
            icon: "🚀",
            milestones: [
                { year: "2013", event: "DeepMind 提出 DQN", impact: "深度学习 + RL" },
                { year: "2015", event: "DQN Nature 论文", impact: "Atari 人类水平" },
                { year: "2015", event: "TRPO 算法", impact: "策略优化理论保证" },
            ]
        },
        {
            era: "2016-2019",
            title: "策略优化",
            color: "#10b981",
            icon: "🎯",
            milestones: [
                { year: "2016", event: "AlphaGo 击败李世石", impact: "围棋 AI 里程碑" },
                { year: "2017", event: "PPO 算法", impact: "工业界标准算法" },
                { year: "2018", event: "OpenAI Five (Dota 2)", impact: "复杂多智能体" },
                { year: "2019", event: "AlphaStar (星际2)", impact: "实时策略游戏" },
            ]
        },
        {
            era: "2020-2023",
            title: "LLM 对齐",
            color: "#f59e0b",
            icon: "💬",
            milestones: [
                { year: "2020", event: "GPT-3 + RLHF 探索", impact: "语言模型对齐" },
                { year: "2022", event: "InstructGPT 论文", impact: "RLHF 方法论" },
                { year: "2022.11", event: "ChatGPT 发布", impact: "RLHF 大规模应用" },
                { year: "2023", event: "DPO 算法", impact: "简化 RLHF 流程" },
            ]
        },
        {
            era: "2024-至今",
            title: "推理时代",
            color: "#06b6d4",
            icon: "🧠",
            milestones: [
                { year: "2024", event: "OpenAI o1 模型", impact: "Reasoning-Time RL" },
                { year: "2024", event: "Process Reward 研究", impact: "过程监督" },
                { year: "2025", event: "Multi-Agent 突破", impact: "协作与竞争" },
            ]
        },
    ];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
            <div className="text-center mb-8">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    强化学习发展时间线
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    从理论奠基到 LLM 对齐的70年历程
                </p>
            </div>

            {/* 时间线 */}
            <div className="relative">
                {/* 中心线 */}
                <div className="absolute left-1/2 transform -translate-x-1/2 w-1 h-full bg-gradient-to-b from-indigo-500 via-purple-500 to-cyan-500 opacity-30" />

                <div className="space-y-12">
                    {timeline.map((era, idx) => (
                        <motion.div
                            key={idx}
                            className="relative"
                            initial={{ opacity: 0, y: 50 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: idx * 0.1 }}
                        >
                            {/* 时间节点 */}
                            <div className="flex items-center justify-center mb-4">
                                <motion.button
                                    onClick={() => setSelectedEra(selectedEra === idx ? null : idx)}
                                    className="relative z-10 px-6 py-3 rounded-full font-bold text-white shadow-lg transition-all duration-300 hover:scale-110"
                                    style={{ backgroundColor: era.color }}
                                    whileHover={{ scale: 1.1 }}
                                    whileTap={{ scale: 0.95 }}
                                >
                                    <span className="text-2xl mr-2">{era.icon}</span>
                                    <span className="text-sm">{era.era}</span>
                                    <div className="text-lg font-black">{era.title}</div>
                                </motion.button>
                            </div>

                            {/* 详细信息 */}
                            {selectedEra === idx && (
                                <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: "auto" }}
                                    exit={{ opacity: 0, height: 0 }}
                                    className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mx-auto max-w-4xl"
                                >
                                    <div className="space-y-4">
                                        {era.milestones.map((milestone, midx) => (
                                            <motion.div
                                                key={midx}
                                                initial={{ opacity: 0, x: -20 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                transition={{ delay: midx * 0.1 }}
                                                className="flex gap-4 p-4 rounded-lg bg-slate-50 dark:bg-slate-700 border-l-4"
                                                style={{ borderColor: era.color }}
                                            >
                                                <div
                                                    className="flex-shrink-0 w-16 h-16 rounded-full flex items-center justify-center text-white font-bold text-sm"
                                                    style={{ backgroundColor: era.color }}
                                                >
                                                    {milestone.year}
                                                </div>
                                                <div className="flex-1">
                                                    <div className="font-bold text-slate-800 dark:text-slate-100 mb-1">
                                                        {milestone.event}
                                                    </div>
                                                    <div className="text-sm text-slate-600 dark:text-slate-400">
                                                        💡 {milestone.impact}
                                                    </div>
                                                </div>
                                            </motion.div>
                                        ))}
                                    </div>
                                </motion.div>
                            )}
                        </motion.div>
                    ))}
                </div>
            </div>

            {/* 图例 */}
            <div className="mt-8 grid grid-cols-2 md:grid-cols-3 gap-3">
                {timeline.map((era, idx) => (
                    <div
                        key={idx}
                        className="flex items-center gap-2 p-2 rounded-lg bg-white dark:bg-slate-800"
                    >
                        <div
                            className="w-4 h-4 rounded-full"
                            style={{ backgroundColor: era.color }}
                        />
                        <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                            {era.title}
                        </span>
                    </div>
                ))}
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 提示：点击时间节点查看该时期的重要里程碑
            </div>
        </div>
    );
}
