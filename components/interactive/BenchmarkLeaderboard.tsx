"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function BenchmarkLeaderboard() {
    const [benchmark, setBenchmark] = useState<"atari" | "mujoco" | "procgen">("atari");
    const [sortBy, setSortBy] = useState<"score" | "efficiency">("score");

    const benchmarks = {
        atari: [
            { algo: "Rainbow DQN", score: 223, humanNorm: 152, sampleEfficiency: 75, year: 2018 },
            { algo: "PPO", score: 189, humanNorm: 112, sampleEfficiency: 85, year: 2017 },
            { algo: "DQN", score: 145, humanNorm: 87, sampleEfficiency: 65, year: 2015 },
            { algo: "A3C", score: 132, humanNorm: 78, sampleEfficiency: 80, year: 2016 },
            { algo: "IMPALA", score: 167, humanNorm: 98, sampleEfficiency: 88, year: 2018 }
        ],
        mujoco: [
            { algo: "SAC", score: 4500, humanNorm: 0, sampleEfficiency: 92, year: 2018 },
            { algo: "TD3", score: 4200, humanNorm: 0, sampleEfficiency: 88, year: 2018 },
            { algo: "PPO", score: 3800, humanNorm: 0, sampleEfficiency: 75, year: 2017 },
            { algo: "DDPG", score: 3200, humanNorm: 0, sampleEfficiency: 70, year: 2015 },
            { algo: "TRPO", score: 3500, humanNorm: 0, sampleEfficiency: 65, year: 2015 }
        ],
        procgen: [
            { algo: "PPG", score: 68, humanNorm: 0, sampleEfficiency: 55, year: 2020 },
            { algo: "PPO", score: 58, humanNorm: 0, sampleEfficiency: 70, year: 2017 },
            { algo: "IMPALA", score: 52, humanNorm: 0, sampleEfficiency: 80, year: 2018 },
            { algo: "Rainbow", score: 45, humanNorm: 0, sampleEfficiency: 60, year: 2018 },
            { algo: "UCB-DrAC", score: 72, humanNorm: 0, sampleEfficiency: 65, year: 2021 }
        ]
    };

    const currentData = benchmarks[benchmark];
    const sortedData = [...currentData].sort((a, b) => {
        if (sortBy === "score") {
            return b.score - a.score;
        } else {
            return b.sampleEfficiency - a.sampleEfficiency;
        }
    });

    const benchmarkInfo = {
        atari: { name: "Atari 2600", metric: "Average Score", desc: "57 games, 200M frames" },
        mujoco: { name: "MuJoCo", metric: "HalfCheetah", desc: "Continuous control" },
        procgen: { name: "Procgen", metric: "Generalization", desc: "16 games, zero-shot" }
    };

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-yellow-50 to-orange-50 dark:from-slate-900 dark:to-yellow-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Benchmark 排行榜
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    RL Algorithm Performance Leaderboard
                </p>
            </div>

            {/* Benchmark Selection */}
            <div className="flex gap-3 justify-center mb-6">
                {(Object.keys(benchmarks) as Array<keyof typeof benchmarks>).map((b) => (
                    <button
                        key={b}
                        onClick={() => setBenchmark(b)}
                        className={`px-6 py-3 rounded-xl font-semibold transition ${benchmark === b
                                ? "bg-orange-600 text-white shadow-lg"
                                : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                            }`}
                    >
                        <div className="text-sm">{benchmarkInfo[b].name}</div>
                        <div className="text-xs opacity-75">{benchmarkInfo[b].desc}</div>
                    </button>
                ))}
            </div>

            {/* Sort Control */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg mb-6 flex items-center justify-between">
                <div className="text-sm text-slate-600 dark:text-slate-400">
                    Metric: <strong className="text-slate-800 dark:text-slate-100">{benchmarkInfo[benchmark].metric}</strong>
                </div>

                <div className="flex items-center gap-3">
                    <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                        排序:
                    </span>
                    <select
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value as any)}
                        className="px-3 py-1 border rounded-lg dark:bg-slate-700 dark:border-slate-600"
                    >
                        <option value="score">Performance</option>
                        <option value="efficiency">Sample Efficiency</option>
                    </select>
                </div>
            </div>

            {/* Leaderboard */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="space-y-3">
                    {sortedData.map((entry, idx) => {
                        const maxScore = Math.max(...currentData.map(e => e.score));
                        const scorePercentage = (entry.score / maxScore) * 100;

                        return (
                            <motion.div
                                key={entry.algo}
                                className="p-4 bg-gradient-to-r from-gray-50 to-white dark:from-slate-700 dark:to-slate-800 rounded-lg border-l-4 border-orange-500"
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: idx * 0.1 }}
                            >
                                <div className="flex items-center justify-between mb-2">
                                    <div className="flex items-center gap-3">
                                        {/* Rank */}
                                        <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold ${idx === 0
                                                ? "bg-yellow-400 text-yellow-900"
                                                : idx === 1
                                                    ? "bg-gray-300 text-gray-700"
                                                    : idx === 2
                                                        ? "bg-orange-400 text-orange-900"
                                                        : "bg-gray-200 text-gray-600"
                                            }`}>
                                            {idx + 1}
                                        </div>

                                        {/* Algorithm */}
                                        <div>
                                            <div className="font-bold text-slate-800 dark:text-slate-100">
                                                {entry.algo}
                                            </div>
                                            <div className="text-xs text-slate-500 dark:text-slate-400">
                                                Published: {entry.year}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-6">
                                        {/* Score */}
                                        <div className="text-right">
                                            <div className="text-xs text-slate-500 dark:text-slate-400">
                                                Score
                                            </div>
                                            <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                                                {entry.score}
                                            </div>
                                            {benchmark === "atari" && (
                                                <div className="text-xs text-slate-500 dark:text-slate-400">
                                                    {entry.humanNorm}% human
                                                </div>
                                            )}
                                        </div>

                                        {/* Efficiency */}
                                        <div className="text-right">
                                            <div className="text-xs text-slate-500 dark:text-slate-400">
                                                Efficiency
                                            </div>
                                            <div className="text-lg font-semibold text-green-600 dark:text-green-400">
                                                {entry.sampleEfficiency}%
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Progress bar */}
                                <div className="relative h-2 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden">
                                    <motion.div
                                        className="absolute left-0 top-0 bottom-0 bg-gradient-to-r from-orange-500 to-yellow-500"
                                        initial={{ width: 0 }}
                                        animate={{ width: `${scorePercentage}%` }}
                                        transition={{ duration: 0.8, delay: idx * 0.1 }}
                                    />
                                </div>
                            </motion.div>
                        );
                    })}
                </div>
            </div>

            {/* Benchmark Standards */}
            <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 p-4 rounded-lg">
                <h5 className="font-semibold text-slate-700 dark:text-slate-300 mb-3">
                    Benchmark标准
                </h5>
                <div className="grid grid-cols-3 gap-3 text-xs">
                    <div className="p-3 bg-white dark:bg-slate-800 rounded-lg">
                        <div className="font-semibold text-blue-600 dark:text-blue-400 mb-1">
                            Atari 2600
                        </div>
                        <div className="text-slate-600 dark:text-slate-400">
                            57游戏 · 人类归一化
                        </div>
                    </div>
                    <div className="p-3 bg-white dark:bg-slate-800 rounded-lg">
                        <div className="font-semibold text-purple-600 dark:text-purple-400 mb-1">
                            MuJoCo
                        </div>
                        <div className="text-slate-600 dark:text-slate-400">
                            连续控制 · 物理模拟
                        </div>
                    </div>
                    <div className="p-3 bg-white dark:bg-slate-800 rounded-lg">
                        <div className="font-semibold text-green-600 dark:text-green-400 mb-1">
                            Procgen
                        </div>
                        <div className="text-slate-600 dark:text-slate-400">
                            泛化能力 · Zero-shot
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
