"use client";

import { motion } from "framer-motion";

export function RLTimelineEvolution() {
    const milestones = [
        { year: "1957", name: "Dynamic Programming", author: "Bellman", desc: "MDP 理论基础，Bellman 方程提出。", color: "slate" },
        { year: "1989", name: "Q-Learning", author: "Watkins", desc: "第一个收敛的 Model-Free Off-policy 算法。", color: "blue" },
        { year: "1992", name: "TD-Gammon", author: "Tesauro", desc: "神经网络 + TD Learning 首次在西洋双陆棋达到人类水平。", color: "blue" },
        { year: "2013", name: "DQN", author: "DeepMind", desc: "Deep RL 爆发。CNN + Q-Learning 攻克 Atari 游戏。", color: "purple" },
        { year: "2016", name: "AlphaGo", author: "DeepMind", desc: "MCTS + RL 击败围棋世界冠军，里程碑时刻。", color: "purple" },
        { year: "2017", name: "PPO & Transformer", author: "OpenAI / Google", desc: "最流行的 Policy Gradient 算法 PPO 诞生。", color: "green" },
        { year: "2022", name: "InstructGPT / RLHF", author: "OpenAI", desc: "RL 用于对齐大语言模型，ChatGPT 诞生。", color: "orange" },
        { year: "2024+", name: "Reasoning & Agent", author: "Community", desc: "o1-like 推理、具身智能、World Models。", color: "red" },
    ];

    return (
        <div className="w-full max-w-3xl mx-auto p-8 relative">
            <h3 className="text-2xl font-bold text-center mb-12 text-slate-800 dark:text-slate-100">
                强化学习演进史 (The History of RL)
            </h3>

            {/* Center Line */}
            <div className="absolute left-1/2 top-24 bottom-10 w-1 bg-gradient-to-b from-slate-200 via-blue-200 to-orange-200 dark:from-slate-700 dark:via-blue-900 dark:to-orange-900 -translate-x-1/2 rounded-full" />

            <div className="space-y-24">
                {milestones.map((item, index) => {
                    const isEven = index % 2 === 0;
                    return (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true, margin: "-50px" }}
                            className={`flex items-center justify-between ${isEven ? "flex-row" : "flex-row-reverse"}`}
                        >
                            {/* Text Content */}
                            <div className={`w-[42%] ${isEven ? "text-right" : "text-left"}`}>
                                <div className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">
                                    {item.name}
                                </div>
                                <div className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">
                                    {item.author}
                                </div>
                                <div className="text-sm text-slate-600 dark:text-slate-400 bg-white dark:bg-slate-800 p-3 rounded-lg shadow-sm border border-slate-100 dark:border-slate-700">
                                    {item.desc}
                                </div>
                            </div>

                            {/* Center Dot */}
                            <div className="relative z-10 w-4 h-4 rounded-full bg-white dark:bg-slate-900 border-4 border-slate-300 dark:border-slate-600 shadow-xl shrink-0" />

                            {/* Year Watermark (On the opposite side) */}
                            <div className={`w-[42%] flex flex-col justify-center ${isEven ? "items-start" : "items-end"}`}>
                                <div className={`text-4xl md:text-5xl font-black text-${item.color}-500/10 dark:text-${item.color}-500/20`}>
                                    {item.year}
                                </div>
                            </div>
                        </motion.div>
                    );
                })}
            </div>
        </div>
    );
}
