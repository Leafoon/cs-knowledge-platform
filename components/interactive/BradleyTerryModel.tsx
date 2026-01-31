"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function BradleyTerryModel() {
    const [rewardA, setRewardA] = useState(2.5);
    const [rewardB, setRewardB] = useState(1.5);

    // Bradley-Terry概率
    const probAWins = 1 / (1 + Math.exp(-(rewardA - rewardB)));
    const probBWins = 1 - probAWins;

    // Sigmoid函数
    const sigmoidPoints = [];
    for (let diff = -5; diff <= 5; diff += 0.1) {
        const prob = 1 / (1 + Math.exp(-diff));
        sigmoidPoints.push({ diff, prob });
    }

    const currentDiff = rewardA - rewardB;

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-slate-900 dark:to-purple-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Bradley-Terry 偏好模型
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    从成对比较中学习奖励函数
                </p>
            </div>

            {/* 公式 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">模型公式</h4>
                <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg font-mono text-center">
                    <div className="text-lg mb-2">P(A ≻ B | x) = σ(r(x, A) - r(x, B))</div>
                    <div className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                        其中 σ(z) = 1 / (1 + e<sup>-z</sup>)
                    </div>
                </div>
            </div>

            {/* 交互式调整 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">调整奖励分数</h4>

                <div className="space-y-6">
                    {/* 回复A */}
                    <div>
                        <div className="flex justify-between mb-2">
                            <span className="font-semibold text-blue-600 dark:text-blue-400">回复 A 奖励</span>
                            <span className="font-mono text-blue-600 dark:text-blue-400">{rewardA.toFixed(2)}</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="5"
                            step="0.1"
                            value={rewardA}
                            onChange={(e) => setRewardA(parseFloat(e.target.value))}
                            className="w-full h-3 bg-blue-200 rounded-lg appearance-none cursor-pointer dark:bg-blue-900"
                        />
                    </div>

                    {/* 回复B */}
                    <div>
                        <div className="flex justify-between mb-2">
                            <span className="font-semibold text-red-600 dark:text-red-400">回复 B 奖励</span>
                            <span className="font-mono text-red-600 dark:text-red-400">{rewardB.toFixed(2)}</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="5"
                            step="0.1"
                            value={rewardB}
                            onChange={(e) => setRewardB(parseFloat(e.target.value))}
                            className="w-full h-3 bg-red-200 rounded-lg appearance-none cursor-pointer dark:bg-red-900"
                        />
                    </div>
                </div>
            </div>

            {/* 可视化结果 */}
            <div className="grid grid-cols-2 gap-6 mb-6">
                {/* 奖励差异 */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">奖励差异</h4>
                    <div className="text-center">
                        <div className="text-5xl font-bold text-purple-600 dark:text-purple-400 mb-2">
                            {currentDiff > 0 ? "+" : ""}{currentDiff.toFixed(2)}
                        </div>
                        <div className="text-sm text-slate-600 dark:text-slate-400">
                            r(A) - r(B)
                        </div>
                    </div>

                    <div className="mt-4 h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                        <div
                            className={`h-full ${currentDiff > 0 ? "bg-blue-500" : "bg-red-500"}`}
                            style={{ width: `${Math.abs(currentDiff) / 5 * 100}%` }}
                        />
                    </div>
                </div>

                {/* 偏好概率 */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">偏好概率</h4>

                    <div className="space-y-4">
                        <div>
                            <div className="flex justify-between text-sm mb-2">
                                <span className="text-blue-600 dark:text-blue-400 font-semibold">P(A ≻ B)</span>
                                <span className="font-mono">{(probAWins * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                <motion.div
                                    className="h-full bg-blue-500"
                                    animate={{ width: `${probAWins * 100}%` }}
                                    transition={{ duration: 0.3 }}
                                />
                            </div>
                        </div>

                        <div>
                            <div className="flex justify-between text-sm mb-2">
                                <span className="text-red-600 dark:text-red-400 font-semibold">P(B ≻ A)</span>
                                <span className="font-mono">{(probBWins * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                <motion.div
                                    className="h-full bg-red-500"
                                    animate={{ width: `${probBWins * 100}%` }}
                                    transition={{ duration: 0.3 }}
                                />
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Sigmoid曲线 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">Sigmoid 函数曲线</h4>

                <div className="relative h-64 bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                    <svg className="w-full h-full" viewBox="-5 -0.1 10 1.2" preserveAspectRatio="none">
                        {/* 网格线 */}
                        <line x1="-5" y1="0.5" x2="5" y2="0.5" stroke="#94a3b8" strokeWidth="0.02" strokeDasharray="0.1,0.1" />
                        <line x1="0" y1="0" x2="0" y2="1" stroke="#94a3b8" strokeWidth="0.02" />

                        {/* Sigmoid曲线 */}
                        <path
                            d={sigmoidPoints.map((p, i) =>
                                `${i === 0 ? 'M' : 'L'} ${p.diff} ${1 - p.prob}`
                            ).join(' ')}
                            stroke="#8b5cf6"
                            strokeWidth="0.05"
                            fill="none"
                        />

                        {/* 当前点 */}
                        <circle
                            cx={currentDiff}
                            cy={1 - probAWins}
                            r="0.1"
                            fill="#ef4444"
                            stroke="white"
                            strokeWidth="0.02"
                        />
                    </svg>

                    {/* 坐标轴标签 */}
                    <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 text-xs text-slate-600 dark:text-slate-400">
                        r(A) - r(B)
                    </div>
                    <div className="absolute top-1/2 right-2 transform -translate-y-1/2 text-xs text-slate-600 dark:text-slate-400">
                        P(A ≻ B)
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-purple-100 dark:bg-purple-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                💡 奖励差异越大，偏好越明显（概率接近0或1）
            </div>
        </div>
    );
}
