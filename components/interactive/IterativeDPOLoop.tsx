"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function IterativeDPOLoop() {
    const [iteration, setIteration] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    const maxIterations = 5;

    // 模拟指标
    const metrics = Array.from({ length: maxIterations + 1 }, (_, i) => ({
        iteration: i,
        winRate: 0.5 + 0.08 * i,
        rewardMargin: 0.5 + 0.4 * i,
        klDiv: 0.1 * i,
        policyQuality: 60 + 7 * i
    }));

    const currentMetrics = metrics[iteration];

    const handlePlay = () => {
        if (isPlaying) {
            setIsPlaying(false);
        } else {
            setIsPlaying(true);
            const interval = setInterval(() => {
                setIteration(prev => {
                    if (prev >= maxIterations) {
                        setIsPlaying(false);
                        clearInterval(interval);
                        return prev;
                    }
                    return prev + 1;
                });
            }, 1500);
        }
    };

    const handleReset = () => {
        setIsPlaying(false);
        setIteration(0);
    };

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-slate-900 dark:to-cyan-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    迭代DPO自我改进循环
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    多轮DPO训练实现持续提升
                </p>
            </div>

            {/* 控制按钮 */}
            <div className="flex items-center justify-between mb-6">
                <div className="flex gap-3">
                    <button
                        onClick={handlePlay}
                        className={`px-4 py-2 rounded-lg font-semibold transition ${isPlaying
                            ? "bg-orange-500 text-white hover:bg-orange-600"
                            : "bg-cyan-600 text-white hover:bg-cyan-700"
                            }`}
                    >
                        {isPlaying ? "⏸ 暂停" : "▶ 开始迭代"}
                    </button>
                    <button
                        onClick={handleReset}
                        className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-semibold hover:bg-gray-300 dark:hover:bg-gray-600 transition"
                    >
                        🔄 重置
                    </button>
                </div>

                <div className="text-sm text-slate-600 dark:text-slate-400">
                    当前轮次: <strong className="text-cyan-600 dark:text-cyan-400">第 {iteration} 轮</strong>
                </div>
            </div>

            {/* 迭代流程 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    迭代 {iteration} - 训练流程
                </h4>

                <div className="grid grid-cols-4 gap-3 mb-4">
                    {[
                        { step: 1, name: "生成样本", icon: "🎲", desc: "π₍ᵢ₎生成回复" },
                        { step: 2, name: "收集偏好", icon: "⭐", desc: "人工/AI标注" },
                        { step: 3, name: "DPO训练", icon: "🔧", desc: "优化策略" },
                        { step: 4, name: "更新参考", icon: "🔄", desc: "π_ref ← π₍ᵢ₊₁₎" }
                    ].map((item) => (
                        <motion.div
                            key={item.step}
                            animate={{
                                scale: iteration > 0 ? 1 : 0.95,
                                opacity: iteration > 0 ? 1 : 0.5
                            }}
                            className={`p-3 rounded-lg border-2 ${iteration > 0
                                ? "border-cyan-500 bg-cyan-50 dark:bg-cyan-900/20"
                                : "border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-800"
                                }`}
                        >
                            <div className="text-2xl text-center mb-1">{item.icon}</div>
                            <div className="text-xs font-semibold text-center text-slate-800 dark:text-slate-100 mb-1">
                                {item.name}
                            </div>
                            <div className="text-xs text-center text-slate-600 dark:text-slate-400">
                                {item.desc}
                            </div>
                        </motion.div>
                    ))}
                </div>

                {/* 迭代历史时间线 */}
                <div className="relative">
                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-cyan-500 to-blue-500 rounded-full"></div>

                    <div className="space-y-3 pl-8">
                        {metrics.slice(0, iteration + 1).map((m, idx) => (
                            <motion.div
                                key={idx}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: idx * 0.1 }}
                                className="relative"
                            >
                                <div className="absolute -left-9 top-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-cyan-600 border-2 border-white dark:border-slate-900"></div>

                                <div className={`p-3 rounded-lg ${idx === iteration
                                    ? "bg-cyan-100 dark:bg-cyan-900/30 border-2 border-cyan-500"
                                    : "bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700"
                                    }`}>
                                    <div className="flex items-center justify-between">
                                        <span className="font-semibold text-slate-800 dark:text-slate-100">
                                            Round {m.iteration}
                                        </span>
                                        <div className="flex gap-4 text-sm">
                                            <span className="text-green-600 dark:text-green-400">
                                                胜率: {(m.winRate * 100).toFixed(0)}%
                                            </span>
                                            <span className="text-purple-600 dark:text-purple-400">
                                                Margin: {m.rewardMargin.toFixed(2)}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </div>

            {/* 指标面板 */}
            <div className="grid grid-cols-4 gap-4 mb-6">
                <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-2 border-green-500">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">胜率</div>
                    <div className="text-3xl font-bold text-green-600 dark:text-green-400">
                        {(currentMetrics.winRate * 100).toFixed(0)}%
                    </div>
                    <div className="mt-2 h-2 bg-green-200 dark:bg-green-900 rounded-full overflow-hidden">
                        <motion.div
                            className="h-full bg-green-600"
                            animate={{ width: `${currentMetrics.winRate * 100}%` }}
                            transition={{ duration: 0.5 }}
                        />
                    </div>
                </div>

                <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border-2 border-purple-500">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">奖励Margin</div>
                    <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                        {currentMetrics.rewardMargin.toFixed(2)}
                    </div>
                    <div className="mt-2 h-2 bg-purple-200 dark:bg-purple-900 rounded-full overflow-hidden">
                        <motion.div
                            className="h-full bg-purple-600"
                            animate={{ width: `${(currentMetrics.rewardMargin / 2.5) * 100}%` }}
                            transition={{ duration: 0.5 }}
                        />
                    </div>
                </div>

                <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border-2 border-orange-500">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">KL散度</div>
                    <div className="text-3xl font-bold text-orange-600 dark:text-orange-400">
                        {currentMetrics.klDiv.toFixed(2)}
                    </div>
                    <div className="mt-2 h-2 bg-orange-200 dark:bg-orange-900 rounded-full overflow-hidden">
                        <motion.div
                            className="h-full bg-orange-600"
                            animate={{ width: `${(currentMetrics.klDiv / 0.5) * 100}%` }}
                            transition={{ duration: 0.5 }}
                        />
                    </div>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-2 border-blue-500">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">策略质量</div>
                    <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                        {currentMetrics.policyQuality.toFixed(0)}
                    </div>
                    <div className="mt-2 h-2 bg-blue-200 dark:bg-blue-900 rounded-full overflow-hidden">
                        <motion.div
                            className="h-full bg-blue-600"
                            animate={{ width: `${currentMetrics.policyQuality}%` }}
                            transition={{ duration: 0.5 }}
                        />
                    </div>
                </div>
            </div>

            {/* 指标趋势图 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">性能趋势</h4>

                <div className="h-48 bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                    <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                        {/* 网格 */}
                        {[0, 25, 50, 75, 100].map(y => (
                            <line
                                key={y}
                                x1="0" y1={y} x2="100" y2={y}
                                stroke="#e5e7eb"
                                strokeWidth="0.5"
                            />
                        ))}

                        {/* 胜率曲线 */}
                        <path
                            d={metrics.slice(0, iteration + 1).map((m, i) => {
                                const x = (i / maxIterations) * 100;
                                const y = 100 - (m.winRate * 100);
                                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                            }).join(' ')}
                            stroke="#22c55e"
                            strokeWidth="2"
                            fill="none"
                        />

                        {/* Margin曲线 */}
                        <path
                            d={metrics.slice(0, iteration + 1).map((m, i) => {
                                const x = (i / maxIterations) * 100;
                                const y = 100 - ((m.rewardMargin / 2.5) * 100);
                                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                            }).join(' ')}
                            stroke="#a855f7"
                            strokeWidth="2"
                            fill="none"
                        />
                    </svg>
                </div>

                <div className="flex justify-center gap-6 mt-4 text-sm">
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-1 bg-green-600 rounded"></div>
                        <span className="text-slate-600 dark:text-slate-400">胜率</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-1 bg-purple-600 rounded"></div>
                        <span className="text-slate-600 dark:text-slate-400">奖励Margin</span>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-cyan-100 dark:bg-cyan-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                💡 每轮迭代都将当前策略作为新的参考模型，实现持续自我改进
            </div>
        </div>
    );
}
