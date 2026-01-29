"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function GPIFramework() {
    const [highlightPhase, setHighlightPhase] = useState<"evaluation" | "improvement" | "both" | null>(null);

    const algorithms = [
        { name: "策略迭代", evaluation: "完全收敛", improvement: "贪心", color: "#8b5cf6" },
        { name: "价值迭代", evaluation: "单步更新", improvement: "贪心", color: "#3b82f6" },
        { name: "Monte Carlo", evaluation: "完整episode", improvement: "ε-greedy", color: "#10b981" },
        { name: "TD学习", evaluation: "Bootstrap", improvement: "ε-greedy", color: "#f59e0b" },
        { name: "Q-learning", evaluation: "TD(0)", improvement: "max Q", color: "#ef4444" },
        { name: "Actor-Critic", evaluation: "TD(0)", improvement: "策略梯度", color: "#ec4899" },
    ];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-8">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    广义策略迭代（GPI）框架
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    理解所有 RL 算法的统一视角
                </p>
            </div>

            {/* 核心图示 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-8 shadow-lg mb-8">
                <div className="relative flex items-center justify-center">
                    {/* 策略评估 */}
                    <motion.div
                        className="w-48 h-32 rounded-xl flex flex-col items-center justify-center cursor-pointer border-4"
                        style={{
                            backgroundColor: highlightPhase === "evaluation" || highlightPhase === "both"
                                ? "#3b82f6"
                                : "#e0e7ff",
                            borderColor: highlightPhase === "evaluation" || highlightPhase === "both"
                                ? "#1d4ed8"
                                : "#c7d2fe",
                        }}
                        onMouseEnter={() => setHighlightPhase("evaluation")}
                        onMouseLeave={() => setHighlightPhase(null)}
                        whileHover={{ scale: 1.05 }}
                    >
                        <div className={`text-2xl mb-2 ${highlightPhase === "evaluation" || highlightPhase === "both"
                                ? "text-white"
                                : "text-indigo-700"
                            }`}>
                            📊
                        </div>
                        <div className={`text-lg font-bold ${highlightPhase === "evaluation" || highlightPhase === "both"
                                ? "text-white"
                                : "text-indigo-700"
                            }`}>
                            策略评估
                        </div>
                        <div className={`text-sm ${highlightPhase === "evaluation" || highlightPhase === "both"
                                ? "text-indigo-100"
                                : "text-indigo-600"
                            }`}>
                            π → V^π
                        </div>
                    </motion.div>

                    {/* 箭头 (评估→改进) */}
                    <div className="mx-12 flex flex-col items-center">
                        <motion.div
                            animate={{
                                x: highlightPhase === "improvement" || highlightPhase === "both" ? [0, 20, 0] : 0,
                            }}
                            transition={{ duration: 1, repeat: Infinity }}
                        >
                            <svg width="100" height="40" viewBox="0 0 100 40">
                                <defs>
                                    <marker
                                        id="arrowhead1"
                                        markerWidth="10"
                                        markerHeight="10"
                                        refX="9"
                                        refY="3"
                                        orient="auto"
                                    >
                                        <polygon points="0 0, 10 3, 0 6" fill="#10b981" />
                                    </marker>
                                </defs>
                                <line
                                    x1="0"
                                    y1="20"
                                    x2="90"
                                    y2="20"
                                    stroke="#10b981"
                                    strokeWidth="3"
                                    markerEnd="url(#arrowhead1)"
                                />
                                <text x="50" y="12" textAnchor="middle" fill="#10b981" fontSize="12" fontWeight="bold">
                                    V更新
                                </text>
                            </svg>
                        </motion.div>
                    </div>

                    {/* 策略改进 */}
                    <motion.div
                        className="w-48 h-32 rounded-xl flex flex-col items-center justify-center cursor-pointer border-4"
                        style={{
                            backgroundColor: highlightPhase === "improvement" || highlightPhase === "both"
                                ? "#10b981"
                                : "#d1fae5",
                            borderColor: highlightPhase === "improvement" || highlightPhase === "both"
                                ? "#059669"
                                : "#a7f3d0",
                        }}
                        onMouseEnter={() => setHighlightPhase("improvement")}
                        onMouseLeave={() => setHighlightPhase(null)}
                        whileHover={{ scale: 1.05 }}
                    >
                        <div className={`text-2xl mb-2 ${highlightPhase === "improvement" || highlightPhase === "both"
                                ? "text-white"
                                : "text-green-700"
                            }`}>
                            🎯
                        </div>
                        <div className={`text-lg font-bold ${highlightPhase === "improvement" || highlightPhase === "both"
                                ? "text-white"
                                : "text-green-700"
                            }`}>
                            策略改进
                        </div>
                        <div className={`text-sm ${highlightPhase === "improvement" || highlightPhase === "both"
                                ? "text-green-100"
                                : "text-green-600"
                            }`}>
                            V^π → π'
                        </div>
                    </motion.div>
                </div>

                {/* 反向箭头 (改进→评估) */}
                <div className="flex justify-center mt-8">
                    <motion.div
                        animate={{
                            x: highlightPhase === "evaluation" || highlightPhase === "both" ? [0, -20, 0] : 0,
                        }}
                        transition={{ duration: 1, repeat: Infinity }}
                    >
                        <svg width="400" height="60" viewBox="0 0 400 60">
                            <defs>
                                <marker
                                    id="arrowhead2"
                                    markerWidth="10"
                                    markerHeight="10"
                                    refX="0"
                                    refY="3"
                                    orient="auto"
                                >
                                    <polygon points="10 0, 0 3, 10 6" fill="#6366f1" />
                                </marker>
                            </defs>
                            <path
                                d="M 380 30 Q 200 60 20 30"
                                stroke="#6366f1"
                                strokeWidth="3"
                                fill="none"
                                markerStart="url(#arrowhead2)"
                            />
                            <text x="200" y="70" textAnchor="middle" fill="#6366f1" fontSize="12" fontWeight="bold">
                                π更新
                            </text>
                        </svg>
                    </motion.div>
                </div>
            </div>

            {/* 算法对比表格 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">
                    GPI 算法族
                </h4>
                <div className="space-y-3">
                    {algorithms.map((algo, idx) => (
                        <motion.div
                            key={idx}
                            className="grid grid-cols-3 gap-4 p-4 rounded-lg border-2 border-slate-200 dark:border-slate-600"
                            style={{ borderLeftColor: algo.color, borderLeftWidth: '6px' }}
                            whileHover={{ scale: 1.02, boxShadow: "0 4px 12px rgba(0,0,0,0.1)" }}
                        >
                            <div className="font-bold text-slate-800 dark:text-slate-100">
                                {algo.name}
                            </div>
                            <div className="text-sm text-slate-600 dark:text-slate-400">
                                <span className="font-semibold">评估: </span>
                                {algo.evaluation}
                            </div>
                            <div className="text-sm text-slate-600 dark:text-slate-400">
                                <span className="font-semibold">改进: </span>
                                {algo.improvement}
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>

            {/* 核心要点 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border-l-4 border-blue-500">
                    <h5 className="font-bold text-blue-800 dark:text-blue-300 mb-2">
                        📊 评估（Evaluation）
                    </h5>
                    <p className="text-sm text-blue-700 dark:text-blue-400">
                        使价值函数 V 更接近当前策略 π 的真实价值 V^π。
                        可以是完全收敛、单步更新或采样估计。
                    </p>
                </div>
                <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border-l-4 border-green-500">
                    <h5 className="font-bold text-green-800 dark:text-green-300 mb-2">
                        🎯 改进（Improvement）
                    </h5>
                    <p className="text-sm text-green-700 dark:text-green-400">
                        基于当前价值函数 V，使策略 π 变得更贪心。
                        可以是完全贪心、ε-greedy 或策略梯度。
                    </p>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 提示：GPI 是所有 RL 算法的统一框架，评估和改进相互竞争但最终收敛
            </div>
        </div>
    );
}
