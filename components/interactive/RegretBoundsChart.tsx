"use client";

import { useState } from "react";
import { motion } from "framer-motion";

interface RegretData {
    algorithm: string;
    bound: string;
    complexity: string;
    color: string;
}

export function RegretBoundsChart() {
    const [selectedMetric, setSelectedMetric] = useState<"regret" | "sample">("regret");
    const [T, setT] = useState(10000);

    const algorithms: RegretData[] = [
        {
            algorithm: "ε-greedy",
            bound: "O(√(KT log T))",
            complexity: "suboptimal",
            color: "gray"
        },
        {
            algorithm: "UCB",
            bound: "O(K log T / Δ)",
            complexity: "optimal",
            color: "blue"
        },
        {
            algorithm: "Thompson Sampling",
            bound: "O(√(KT log T))",
            complexity: "near-optimal",
            color: "green"
        },
        {
            algorithm: "UCB-VI (MDP)",
            bound: "O(√(H³SAT))",
            complexity: "minimax-optimal",
            color: "purple"
        }
    ];

    // 计算遗憾值
    const calculateRegret = (algo: RegretData, t: number) => {
        const K = 10; // 10臂bandit
        const Delta = 0.2; // gap
        const H = 20; // horizon
        const S = 100; // states
        const A = 10; // actions

        switch (algo.algorithm) {
            case "ε-greedy":
                return Math.sqrt(K * t * Math.log(t));
            case "UCB":
                return (K * Math.log(t)) / Delta;
            case "Thompson Sampling":
                return Math.sqrt(K * t * Math.log(t)) * 0.9;
            case "UCB-VI (MDP)":
                return Math.sqrt(H ** 3 * S * A * t);
            default:
                return 0;
        }
    };

    const timePoints = Array.from({ length: 50 }, (_, i) => Math.floor((T / 50) * (i + 1)));

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-slate-900 dark:to-purple-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    遗憾界对比
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Online Learning & Exploration-Exploitation Tradeoff
                </p>
            </div>

            {/* 时间范围滑块 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg mb-6">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                        时间步数 T
                    </span>
                    <span className="text-lg font-bold text-purple-600 dark:text-purple-400">
                        {T.toLocaleString()}
                    </span>
                </div>
                <input
                    type="range"
                    min="1000"
                    max="100000"
                    step="1000"
                    value={T}
                    onChange={(e) => setT(Number(e.target.value))}
                    className="w-full"
                />
            </div>

            {/* 遗憾曲线 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    累积遗憾随时间增长
                </h4>

                <div className="relative h-80 bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                    <svg className="w-full h-full">
                        {/* 坐标轴 */}
                        <line x1="50" y1="10" x2="50" y2="280" stroke="currentColor" strokeWidth="2" className="text-gray-400" />
                        <line x1="50" y1="280" x2="750" y2="280" stroke="currentColor" strokeWidth="2" className="text-gray-400" />

                        {/* 轴标签 */}
                        <text x="10" y="15" className="text-xs fill-current text-gray-600 dark:text-gray-400">Regret</text>
                        <text x="700" y="300" className="text-xs fill-current text-gray-600 dark:text-gray-400">Time</text>

                        {/* 算法曲线 */}
                        {algorithms.map((algo, idx) => {
                            const maxRegret = Math.max(...algorithms.map(a => calculateRegret(a, T)));

                            const pathData = timePoints.map((t, i) => {
                                const x = 50 + (t / T) * 700;
                                const regret = calculateRegret(algo, t);
                                const y = 280 - (regret / maxRegret) * 260;
                                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                            }).join(' ');

                            return (
                                <motion.g key={algo.algorithm}>
                                    <motion.path
                                        d={pathData}
                                        fill="none"
                                        stroke="currentColor"
                                        strokeWidth="3"
                                        className={`text-${algo.color}-600`}
                                        initial={{ pathLength: 0, opacity: 0 }}
                                        animate={{ pathLength: 1, opacity: 1 }}
                                        transition={{ duration: 1, delay: idx * 0.2 }}
                                    />
                                </motion.g>
                            );
                        })}
                    </svg>
                </div>

                {/* 图例 */}
                <div className="grid grid-cols-4 gap-3 mt-4">
                    {algorithms.map((algo) => (
                        <div key={algo.algorithm} className="flex items-center gap-2">
                            <div className={`w-4 h-4 rounded bg-${algo.color}-600`}></div>
                            <div>
                                <div className="text-xs font-semibold text-slate-800 dark:text-slate-100">
                                    {algo.algorithm}
                                </div>
                                <div className="text-xs text-slate-500 dark:text-slate-400">
                                    {algo.bound}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* 理论界对比表 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    理论复杂度对比
                </h4>

                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b-2 border-gray-300 dark:border-gray-700">
                                <th className="text-left py-2 px-4 text-slate-700 dark:text-slate-300">算法</th>
                                <th className="text-center py-2 px-4 text-slate-700 dark:text-slate-300">遗憾界</th>
                                <th className="text-center py-2 px-4 text-slate-700 dark:text-slate-300">T={T}时遗憾</th>
                                <th className="text-center py-2 px-4 text-slate-700 dark:text-slate-300">最优性</th>
                            </tr>
                        </thead>
                        <tbody>
                            {algorithms.map((algo, idx) => {
                                const regret = calculateRegret(algo, T);
                                return (
                                    <motion.tr
                                        key={idx}
                                        className="border-b border-gray-200 dark:border-gray-700"
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: idx * 0.1 }}
                                    >
                                        <td className="py-3 px-4 font-semibold text-slate-800 dark:text-slate-100">
                                            {algo.algorithm}
                                        </td>
                                        <td className="text-center py-3 px-4 font-mono text-xs">
                                            {algo.bound}
                                        </td>
                                        <td className="text-center py-3 px-4 font-bold text-purple-600 dark:text-purple-400">
                                            {regret.toFixed(0)}
                                        </td>
                                        <td className="text-center py-3 px-4">
                                            <span className={`px-2 py-1 rounded text-xs font-semibold ${algo.complexity === "optimal"
                                                    ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                                                    : algo.complexity === "minimax-optimal"
                                                        ? "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400"
                                                        : algo.complexity === "near-optimal"
                                                            ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400"
                                                            : "bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400"
                                                }`}>
                                                {algo.complexity}
                                            </span>
                                        </td>
                                    </motion.tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* PAC vs Regret对比 */}
            <div className="grid grid-cols-2 gap-4">
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-300 dark:border-blue-700">
                    <h5 className="font-semibold text-blue-700 dark:text-blue-400 mb-2">
                        PAC学习框架
                    </h5>
                    <div className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                        <div>• 目标: 以高概率找到ε-最优策略</div>
                        <div>• 样本复杂度: Õ(S²A/(ε²(1−γ)³))</div>
                        <div>• 适用: 离线学习</div>
                    </div>
                </div>

                <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border border-purple-300 dark:border-purple-700">
                    <h5 className="font-semibold text-purple-700 dark:text-purple-400 mb-2">
                        遗憾最小化
                    </h5>
                    <div className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                        <div>• 目标: 最小化累积遗憾</div>
                        <div>• 遗憾界: Õ(√(H³SAT))</div>
                        <div>• 适用: 在线学习</div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-purple-100 dark:bg-purple-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                <strong>Minimax最优</strong>: UCB-VI在episodic MDP上达到Õ(√(H³SAT))遗憾界，无法改进（除log因子）
            </div>
        </div>
    );
}
