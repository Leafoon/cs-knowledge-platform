"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function ExplorationStrategiesComparison() {
    const [selectedStrategy, setSelectedStrategy] = useState<"epsilon" | "ucb" | "ts">("ucb");
    const [step, setStep] = useState(0);
    const [isAnimating, setIsAnimating] = useState(false);

    // 10-armed bandit
    const numArms = 10;
    const trueRewards = [0.3, 0.5, 0.7, 0.2, 0.9, 0.4, 0.6, 0.1, 0.8, 0.35];
    const [empiricalRewards, setEmpiricalRewards] = useState(Array(numArms).fill(0));
    const [pullCounts, setPullCounts] = useState(Array(numArms).fill(0));

    // 策略详情
    const strategies = {
        epsilon: {
            name: "ε-greedy",
            formula: "a = argmax Q̂(a) w.p. 1-ε, random w.p. ε",
            regret: "O(√(KT log T))",
            color: "gray"
        },
        ucb: {
            name: "UCB",
            formula: "a = argmax [Q̂(a) + √(2 log t / N(a))]",
            regret: "O(K log T / Δ)",
            color: "blue"
        },
        ts: {
            name: "Thompson Sampling",
            formula: "a ~ P(a is optimal | data)",
            regret: "O(√(KT log T))",
            color: "green"
        }
    };

    const selectArm = () => {
        const t = step + 1;
        let selectedArm = 0;

        if (selectedStrategy === "epsilon") {
            // ε-greedy
            const epsilon = 0.1;
            if (Math.random() < epsilon) {
                selectedArm = Math.floor(Math.random() * numArms);
            } else {
                selectedArm = empiricalRewards.indexOf(Math.max(...empiricalRewards));
            }
        } else if (selectedStrategy === "ucb") {
            // UCB
            const ucbValues = empiricalRewards.map((reward, i) => {
                const count = pullCounts[i] || 1;
                const bonus = Math.sqrt((2 * Math.log(t)) / count);
                return reward + bonus;
            });
            selectedArm = ucbValues.indexOf(Math.max(...ucbValues));
        } else {
            // Thompson Sampling (简化版)
            const samples = empiricalRewards.map((reward, i) => {
                const alpha = pullCounts[i] * reward + 1;
                const beta = pullCounts[i] * (1 - reward) + 1;
                // Beta分布采样（简化）
                return Math.random() * alpha / (alpha + beta);
            });
            selectedArm = samples.indexOf(Math.max(...samples));
        }

        // 更新统计
        const newCounts = [...pullCounts];
        newCounts[selectedArm]++;

        const newRewards = [...empiricalRewards];
        const reward = trueRewards[selectedArm] + (Math.random() - 0.5) * 0.2;
        newRewards[selectedArm] = (
            (newRewards[selectedArm] * pullCounts[selectedArm] + reward) /
            newCounts[selectedArm]
        );

        setPullCounts(newCounts);
        setEmpiricalRewards(newRewards);
        setStep(step + 1);
    };

    const reset = () => {
        setStep(0);
        setEmpiricalRewards(Array(numArms).fill(0));
        setPullCounts(Array(numArms).fill(0));
    };

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-green-50 to-teal-50 dark:from-slate-900 dark:to-green-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    探索策略对比
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Multi-Armed Bandit: Exploration vs Exploitation
                </p>
            </div>

            {/* 策略选择 */}
            <div className="flex gap-4 justify-center mb-6">
                {Object.entries(strategies).map(([key, strategy]) => (
                    <button
                        key={key}
                        onClick={() => {
                            setSelectedStrategy(key as any);
                            reset();
                        }}
                        className={`px-6 py-3 rounded-xl font-semibold transition ${selectedStrategy === key
                                ? `bg-${strategy.color}-600 text-white`
                                : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                            }`}
                    >
                        <div>{strategy.name}</div>
                        <div className="text-xs opacity-75">{strategy.regret}</div>
                    </button>
                ))}
            </div>

            {/* 当前策略说明 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg mb-6">
                <div className="text-center">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">公式</div>
                    <div className="font-mono text-sm font-semibold text-slate-800 dark:text-slate-100">
                        {strategies[selectedStrategy].formula}
                    </div>
                </div>
            </div>

            {/* Bandit可视化 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100">
                        10-Armed Bandit
                    </h4>
                    <div className="text-sm text-slate-600 dark:text-slate-400">
                        步数: <strong>{step}</strong>
                    </div>
                </div>

                <div className="grid grid-cols-10 gap-2 mb-4">
                    {trueRewards.map((reward, i) => {
                        const isOptimal = reward === Math.max(...trueRewards);
                        const count = pullCounts[i];
                        const empirical = empiricalRewards[i];

                        return (
                            <div key={i} className="text-center">
                                {/* 臂柱状图 */}
                                <div className="relative h-40 bg-gray-200 dark:bg-gray-700 rounded-lg overflow-hidden">
                                    {/* 真实值 */}
                                    <div
                                        className="absolute bottom-0 w-full bg-blue-300 dark:bg-blue-600 opacity-50"
                                        style={{ height: `${reward * 100}%` }}
                                    />
                                    {/* 估计值 */}
                                    <motion.div
                                        className={`absolute bottom-0 w-full ${isOptimal
                                                ? "bg-green-500"
                                                : "bg-orange-500"
                                            }`}
                                        initial={{ height: 0 }}
                                        animate={{ height: `${empirical * 100}%` }}
                                        transition={{ duration: 0.3 }}
                                    />
                                    {/* 最优标记 */}
                                    {isOptimal && (
                                        <div className="absolute top-1 left-1 text-xs bg-green-700 text-white px-1 rounded">
                                            ★
                                        </div>
                                    )}
                                </div>

                                {/* 拉取次数 */}
                                <div className="mt-2 text-xs font-semibold text-slate-600 dark:text-slate-400">
                                    N={count}
                                </div>
                            </div>
                        );
                    })}
                </div>

                {/* 图例 */}
                <div className="flex gap-6 justify-center text-xs">
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-blue-300 dark:bg-blue-600 opacity-50"></div>
                        <span className="text-slate-600 dark:text-slate-400">真实奖励</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-orange-500"></div>
                        <span className="text-slate-600 dark:text-slate-400">估计奖励</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-green-500"></div>
                        <span className="text-slate-600 dark:text-slate-400">最优臂</span>
                    </div>
                </div>
            </div>

            {/* 控制面板 */}
            <div className="flex gap-3 justify-center mb-6">
                <button
                    onClick={selectArm}
                    className="px-8 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition shadow-lg"
                >
                    拉一次
                </button>
                <button
                    onClick={() => {
                        for (let i = 0; i < 10; i++) selectArm();
                    }}
                    className="px-8 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition shadow-lg"
                >
                    拉10次
                </button>
                <button
                    onClick={reset}
                    className="px-8 py-3 bg-gray-600 text-white rounded-lg font-semibold hover:bg-gray-700 transition shadow-lg"
                >
                    重置
                </button>
            </div>

            {/* 性能指标 */}
            <div className="grid grid-cols-3 gap-4">
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                    <div className="text-sm text-blue-700 dark:text-blue-400 mb-1">最优臂拉取率</div>
                    <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                        {step > 0
                            ? ((pullCounts[trueRewards.indexOf(Math.max(...trueRewards))] / step) * 100).toFixed(1)
                            : "0.0"}%
                    </div>
                </div>

                <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                    <div className="text-sm text-green-700 dark:text-green-400 mb-1">累积奖励</div>
                    <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                        {pullCounts.reduce((sum, count, i) => sum + count * empiricalRewards[i], 0).toFixed(1)}
                    </div>
                </div>

                <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
                    <div className="text-sm text-orange-700 dark:text-orange-400 mb-1">遗憾</div>
                    <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                        {(step * Math.max(...trueRewards) -
                            pullCounts.reduce((sum, count, i) => sum + count * trueRewards[i], 0)).toFixed(1)}
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-green-100 dark:bg-green-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                💡 <strong>原则</strong>: 探索未知可能性 vs 利用已知最优 — UCB通过置信区间自动平衡
            </div>
        </div>
    );
}
