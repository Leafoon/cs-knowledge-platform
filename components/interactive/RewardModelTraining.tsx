"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function RewardModelTraining() {
    const [epoch, setEpoch] = useState(0);
    const [isTraining, setIsTraining] = useState(false);

    // 模拟训练指标
    const maxEpochs = 10;
    const loss = 2.5 * Math.exp(-epoch * 0.3) + 0.1;
    const accuracy = 0.5 + 0.45 * (1 - Math.exp(-epoch * 0.4));

    const handleTrain = () => {
        if (isTraining) {
            setIsTraining(false);
        } else {
            setIsTraining(true);
            const interval = setInterval(() => {
                setEpoch(prev => {
                    if (prev >= maxEpochs) {
                        setIsTraining(false);
                        clearInterval(interval);
                        return prev;
                    }
                    return prev + 1;
                });
            }, 800);
        }
    };

    const handleReset = () => {
        setIsTraining(false);
        setEpoch(0);
    };

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    奖励模型训练过程
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    从偏好对学习人类价值观
                </p>
            </div>

            {/* 训练架构 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">模型架构</h4>

                <div className="flex items-center justify-between gap-4">
                    <div className="flex-1 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-2 border-blue-300 dark:border-blue-700">
                        <div className="text-center">
                            <div className="text-3xl mb-2">📝</div>
                            <div className="font-semibold text-slate-800 dark:text-slate-100">输入</div>
                            <div className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                                Prompt + Response
                            </div>
                        </div>
                    </div>

                    <div className="text-2xl text-gray-400">→</div>

                    <div className="flex-1 bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border-2 border-purple-300 dark:border-purple-700">
                        <div className="text-center">
                            <div className="text-3xl mb-2">🧠</div>
                            <div className="font-semibold text-slate-800 dark:text-slate-100">Transformer</div>
                            <div className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                                (基于SFT初始化)
                            </div>
                        </div>
                    </div>

                    <div className="text-2xl text-gray-400">→</div>

                    <div className="flex-1 bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-2 border-green-300 dark:border-green-700">
                        <div className="text-center">
                            <div className="text-3xl mb-2">⭐</div>
                            <div className="font-semibold text-slate-800 dark:text-slate-100">奖励分数</div>
                            <div className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                                标量值 r ∈ ℝ
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* 训练数据示例 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">训练数据（偏好对）</h4>

                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-2 border-green-500">
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-xl">✅</span>
                            <span className="font-semibold text-green-700 dark:text-green-400">Chosen（优选）</span>
                        </div>
                        <div className="text-sm text-slate-700 dark:text-slate-300 italic">
                            "强化学习是一种机器学习方法，智能体通过与环境交互学习策略。
                            主要组成包括状态、动作和奖励..."
                        </div>
                        <div className="mt-3 text-center">
                            <div className="inline-block bg-green-600 text-white px-3 py-1 rounded-full text-sm font-semibold">
                                r<sub>chosen</sub> = 3.2
                            </div>
                        </div>
                    </div>

                    <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border-2 border-red-500">
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-xl">❌</span>
                            <span className="font-semibold text-red-700 dark:text-red-400">Rejected（拒绝）</span>
                        </div>
                        <div className="text-sm text-slate-700 dark:text-slate-300 italic">
                            "强化学习就是一种学习方式。"
                        </div>
                        <div className="mt-3 text-center">
                            <div className="inline-block bg-red-600 text-white px-3 py-1 rounded-full text-sm font-semibold">
                                r<sub>rejected</sub> = 0.8
                            </div>
                        </div>
                    </div>
                </div>

                <div className="mt-4 bg-purple-100 dark:bg-purple-900/30 p-3 rounded-lg text-center font-mono text-sm">
                    Loss = -log σ(r<sub>chosen</sub> - r<sub>rejected</sub>) = -log σ(2.4) ≈ 0.08
                </div>
            </div>

            {/* 训练控制 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100">训练进度</h4>
                    <div className="flex gap-3">
                        <button
                            onClick={handleTrain}
                            className={`px-4 py-2 rounded-lg font-semibold transition ${isTraining
                                    ? "bg-orange-500 text-white hover:bg-orange-600"
                                    : "bg-indigo-600 text-white hover:bg-indigo-700"
                                }`}
                        >
                            {isTraining ? "⏸ 暂停" : "▶ 开始训练"}
                        </button>
                        <button
                            onClick={handleReset}
                            className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-semibold hover:bg-gray-300 dark:hover:bg-gray-600 transition"
                        >
                            🔄 重置
                        </button>
                    </div>
                </div>

                <div className="mb-4">
                    <div className="flex justify-between text-sm mb-2">
                        <span className="text-slate-600 dark:text-slate-400">Epoch {epoch}/{maxEpochs}</span>
                        <span className="font-semibold text-indigo-600 dark:text-indigo-400">
                            {Math.round((epoch / maxEpochs) * 100)}%
                        </span>
                    </div>
                    <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                        <motion.div
                            className="h-full bg-indigo-600"
                            animate={{ width: `${(epoch / maxEpochs) * 100}%` }}
                            transition={{ duration: 0.3 }}
                        />
                    </div>
                </div>

                {/* 指标 */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
                        <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Loss (损失)</div>
                        <div className="text-3xl font-bold text-red-600 dark:text-red-400">
                            {loss.toFixed(3)}
                        </div>
                        <div className="mt-2 h-2 bg-red-200 dark:bg-red-900 rounded-full overflow-hidden">
                            <motion.div
                                className="h-full bg-red-600"
                                animate={{ width: `${Math.max(0, 100 - loss * 30)}%` }}
                                transition={{ duration: 0.3 }}
                            />
                        </div>
                    </div>

                    <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                        <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">Accuracy (准确率)</div>
                        <div className="text-3xl font-bold text-green-600 dark:text-green-400">
                            {(accuracy * 100).toFixed(1)}%
                        </div>
                        <div className="mt-2 h-2 bg-green-200 dark:bg-green-900 rounded-full overflow-hidden">
                            <motion.div
                                className="h-full bg-green-600"
                                animate={{ width: `${accuracy * 100}%` }}
                                transition={{ duration: 0.3 }}
                            />
                        </div>
                    </div>
                </div>
            </div>

            {/* 训练曲线 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">训练曲线</h4>

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

                        {/* Loss曲线 */}
                        <path
                            d={Array.from({ length: epoch + 1 }, (_, i) => {
                                const l = 2.5 * Math.exp(-i * 0.3) + 0.1;
                                const x = (i / maxEpochs) * 100;
                                const y = 100 - (l / 3 * 100);
                                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                            }).join(' ')}
                            stroke="#ef4444"
                            strokeWidth="2"
                            fill="none"
                        />

                        {/* Accuracy曲线 */}
                        <path
                            d={Array.from({ length: epoch + 1 }, (_, i) => {
                                const acc = 0.5 + 0.45 * (1 - Math.exp(-i * 0.4));
                                const x = (i / maxEpochs) * 100;
                                const y = 100 - (acc * 100);
                                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                            }).join(' ')}
                            stroke="#22c55e"
                            strokeWidth="2"
                            fill="none"
                        />
                    </svg>
                </div>

                <div className="flex justify-center gap-6 mt-4 text-sm">
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-1 bg-red-600 rounded"></div>
                        <span className="text-slate-600 dark:text-slate-400">Loss</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-1 bg-green-600 rounded"></div>
                        <span className="text-slate-600 dark:text-slate-400">Accuracy</span>
                    </div>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-600 dark:text-slate-400">
                💡 奖励模型学习人类偏好：chosen奖励 &gt; rejected奖励
            </div>
        </div>
    );
}
