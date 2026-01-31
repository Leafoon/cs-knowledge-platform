"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function LearningCurveComparison() {
    const [showConfidence, setShowConfidence] = useState(true);
    const [numSeeds, setNumSeeds] = useState(5);

    // Simulate learning curves for different algorithms
    const generateLearningCurve = (algorithm: string, seed: number) => {
        const points = [];
        const noise = 0.1 + seed * 0.02;

        for (let episode = 0; episode < 500; episode++) {
            let baseReward = 0;

            if (algorithm === "PPO") {
                baseReward = Math.min(200, 50 + episode * 0.6 - Math.exp(-episode / 100) * 30);
            } else if (algorithm === "DQN") {
                baseReward = Math.min(180, 30 + episode * 0.5 - Math.exp(-episode / 150) * 40);
            } else if (algorithm === "A2C") {
                baseReward = Math.min(160, 40 + episode * 0.45 - Math.exp(-episode / 120) * 35);
            }

            points.push(baseReward + (Math.random() - 0.5) * 40 * noise);
        }

        return points;
    };

    const algorithms = ["PPO", "DQN", "A2C"];
    const colors = ["blue", "red", "green"];

    // Generate multiple seeds
    const allCurves = algorithms.map(algo => {
        const seeds = Array.from({ length: numSeeds }, (_, i) =>
            generateLearningCurve(algo, i)
        );

        // Compute mean and std
        const numEpisodes = seeds[0].length;
        const mean = Array.from({ length: numEpisodes }, (_, episode) => {
            const values = seeds.map(s => s[episode]);
            return values.reduce((a, b) => a + b) / values.length;
        });

        const std = Array.from({ length: numEpisodes }, (_, episode) => {
            const values = seeds.map(s => s[episode]);
            const m = mean[episode];
            const variance = values.reduce((acc, v) => acc + (v - m) ** 2, 0) / values.length;
            return Math.sqrt(variance);
        });

        return { algorithm: algo, mean, std };
    });

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-green-50 to-blue-50 dark:from-slate-900 dark:to-green-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    学习曲线对比
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Multi-Algorithm Learning Curve Comparison
                </p>
            </div>

            {/* Controls */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg mb-6">
                <div className="flex items-center justify-between">
                    <div>
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={showConfidence}
                                onChange={(e) => setShowConfidence(e.target.checked)}
                                className="w-4 h-4"
                            />
                            <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                                显示置信区间（±1 std）
                            </span>
                        </label>
                    </div>

                    <div className="flex items-center gap-3">
                        <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                            随机种子数:
                        </span>
                        <select
                            value={numSeeds}
                            onChange={(e) => setNumSeeds(Number(e.target.value))}
                            className="px-3 py-1 border rounded-lg dark:bg-slate-700 dark:border-slate-600"
                        >
                            {[3, 5, 10].map(n => (
                                <option key={n} value={n}>{n}</option>
                            ))}
                        </select>
                    </div>
                </div>
            </div>

            {/* Learning Curves */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    训练性能曲线（{numSeeds}次运行均值）
                </h4>

                <svg width="100%" height="400" viewBox="0 0 800 400" className="mx-auto">
                    {/* Grid */}
                    {Array.from({ length: 5 }).map((_, i) => (
                        <line
                            key={`h-${i}`}
                            x1="80"
                            y1={50 + i * 75}
                            x2="750"
                            y2={50 + i * 75}
                            stroke="currentColor"
                            className="text-gray-300 dark:text-gray-700"
                            strokeWidth="1"
                            opacity="0.3"
                        />
                    ))}

                    {/* Axes */}
                    <line x1="80" y1="350" x2="750" y2="350" stroke="currentColor" className="text-gray-600" strokeWidth="2" />
                    <line x1="80" y1="50" x2="80" y2="350" stroke="currentColor" className="text-gray-600" strokeWidth="2" />

                    {/* Curves */}
                    {allCurves.map((curve, algoIdx) => {
                        const color = colors[algoIdx];
                        const points = curve.mean.filter((_, i) => i % 5 === 0); // Sample every 5
                        const stds = curve.std.filter((_, i) => i % 5 === 0);

                        return (
                            <g key={curve.algorithm}>
                                {/* Confidence band */}
                                {showConfidence && (
                                    <path
                                        d={`
                                            M ${points.map((p, i) => {
                                            const x = 80 + (i / (points.length - 1)) * 670;
                                            const y = 350 - ((p + stds[i]) / 200) * 300;
                                            return `${x},${y}`;
                                        }).join(' L ')}
                                            L ${points.map((p, i) => {
                                            const idx = points.length - 1 - i;
                                            const x = 80 + (idx / (points.length - 1)) * 670;
                                            const y = 350 - ((points[idx] - stds[idx]) / 200) * 300;
                                            return `${x},${y}`;
                                        }).join(' L ')}
                                            Z
                                        `}
                                        className={`fill-${color}-200 dark:fill-${color}-900`}
                                        opacity="0.3"
                                    />
                                )}

                                {/* Mean curve */}
                                <motion.path
                                    d={`M ${points.map((p, i) => {
                                        const x = 80 + (i / (points.length - 1)) * 670;
                                        const y = 350 - (p / 200) * 300;
                                        return `${i === 0 ? '' : 'L '}${x},${y}`;
                                    }).join(' ')}`}
                                    stroke="currentColor"
                                    className={`text-${color}-600 dark:text-${color}-400`}
                                    strokeWidth="3"
                                    fill="none"
                                    initial={{ pathLength: 0 }}
                                    animate={{ pathLength: 1 }}
                                    transition={{ duration: 1.5, delay: algoIdx * 0.2 }}
                                />
                            </g>
                        );
                    })}

                    {/* Labels */}
                    <text x="415" y="385" textAnchor="middle" className="fill-current text-slate-600 dark:text-slate-400 text-sm font-semibold">
                        Episode
                    </text>
                    <text x="40" y="200" textAnchor="middle" className="fill-current text-slate-600 dark:text-slate-400 text-sm font-semibold" transform="rotate(-90 40 200)">
                        Reward
                    </text>

                    {/* Y-axis labels */}
                    {[0, 50, 100, 150, 200].map((val, i) => (
                        <text
                            key={val}
                            x="70"
                            y={350 - i * 75 + 5}
                            textAnchor="end"
                            className="fill-current text-slate-500 dark:text-slate-400 text-xs"
                        >
                            {val}
                        </text>
                    ))}
                </svg>

                {/* Legend */}
                <div className="flex justify-center gap-6 mt-4">
                    {algorithms.map((algo, idx) => (
                        <div key={algo} className="flex items-center gap-2">
                            <div className={`w-8 h-1 bg-${colors[idx]}-600 rounded`}></div>
                            <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                                {algo}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Statistics Table */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    性能统计
                </h4>

                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b dark:border-slate-700">
                                <th className="text-left py-2 px-3 font-semibold text-slate-700 dark:text-slate-300">
                                    Algorithm
                                </th>
                                <th className="text-center py-2 px-3 font-semibold text-slate-700 dark:text-slate-300">
                                    Final Mean
                                </th>
                                <th className="text-center py-2 px-3 font-semibold text-slate-700 dark:text-slate-300">
                                    Final Std
                                </th>
                                <th className="text-center py-2 px-3 font-semibold text-slate-700 dark:text-slate-300">
                                    AUC
                                </th>
                                <th className="text-center py-2 px-3 font-semibold text-slate-700 dark:text-slate-300">
                                    Convergence
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            {allCurves.map((curve, idx) => {
                                const finalMean = curve.mean.slice(-50).reduce((a, b) => a + b) / 50;
                                const finalStd = curve.std.slice(-50).reduce((a, b) => a + b) / 50;
                                const auc = curve.mean.reduce((a, b) => a + b) / curve.mean.length;
                                const convergence = curve.mean.findIndex(r => r > 150);

                                return (
                                    <tr key={curve.algorithm} className="border-b dark:border-slate-700 hover:bg-gray-50 dark:hover:bg-slate-700">
                                        <td className={`py-3 px-3 font-semibold text-${colors[idx]}-600 dark:text-${colors[idx]}-400`}>
                                            {curve.algorithm}
                                        </td>
                                        <td className="text-center py-3 px-3 text-slate-700 dark:text-slate-300">
                                            {finalMean.toFixed(1)}
                                        </td>
                                        <td className="text-center py-3 px-3 text-slate-700 dark:text-slate-300">
                                            {finalStd.toFixed(1)}
                                        </td>
                                        <td className="text-center py-3 px-3 text-slate-700 dark:text-slate-300">
                                            {auc.toFixed(1)}
                                        </td>
                                        <td className="text-center py-3 px-3 text-slate-700 dark:text-slate-300">
                                            ~{convergence > 0 ? convergence : "N/A"}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>

            <div className="mt-6 bg-blue-100 dark:bg-blue-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                📊 <strong>建议</strong>: 使用至少5个随机种子 + 报告均值±标准差或中位数+IQR
            </div>
        </div>
    );
}
