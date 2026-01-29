"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

export function BellmanEquationDerivation() {
    const [currentStep, setCurrentStep] = useState(0);

    const derivationSteps = [
        {
            title: "起点：价值函数定义",
            equation: "V^π(s) = 𝔼_π[G_t | S_t = s]",
            explanation: "状态价值函数定义为从状态 s 开始，遵循策略 π，期望获得的累积折扣奖励"
        },
        {
            title: "展开 Return",
            equation: "V^π(s) = 𝔼_π[R_{t+1} + γG_{t+1} | S_t = s]",
            explanation: "将 Return G_t 展开为即时奖励 R_{t+1} 加上折扣后的未来 Return γG_{t+1}"
        },
        {
            title: "期望的线性性",
            equation: "V^π(s) = 𝔼_π[R_{t+1} | S_t = s] + γ𝔼_π[G_{t+1} | S_t = s]",
            explanation: "利用期望的线性性质，将期望分解为两部分"
        },
        {
            title: "对动作求和",
            equation: "V^π(s) = Σ_a π(a|s) 𝔼[R_{t+1} + γG_{t+1} | S_t=s, A_t=a]",
            explanation: "根据全期望公式，对所有可能的动作求和，权重为策略概率 π(a|s)"
        },
        {
            title: "对下一状态求和",
            equation: "V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) 𝔼[R_{t+1} + γG_{t+1} | S_t=s, A_t=a, S_{t+1}=s']",
            explanation: "再次使用全期望公式，对所有可能的下一状态 s' 求和"
        },
        {
            title: "马尔可夫性质",
            equation: "V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ𝔼[G_{t+1} | S_{t+1}=s']]",
            explanation: "利用马尔可夫性质，未来只依赖于 S_{t+1}，与 S_t 和 A_t 无关"
        },
        {
            title: "递归定义",
            equation: "V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]",
            explanation: "识别出 𝔼[G_{t+1} | S_{t+1}=s'] = V^π(s')，得到 Bellman 期望方程！"
        },
    ];

    const nextStep = () => {
        if (currentStep < derivationSteps.length - 1) {
            setCurrentStep(currentStep + 1);
        }
    };

    const prevStep = () => {
        if (currentStep > 0) {
            setCurrentStep(currentStep - 1);
        }
    };

    const reset = () => {
        setCurrentStep(0);
    };

    return (
        <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Bellman 方程推导
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    从价值函数定义到递归形式
                </p>
            </div>

            {/* 进度条 */}
            <div className="mb-8">
                <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-semibold text-slate-600 dark:text-slate-400">
                        步骤 {currentStep + 1} / {derivationSteps.length}
                    </span>
                    <span className="text-sm text-slate-500 dark:text-slate-400">
                        {Math.round(((currentStep + 1) / derivationSteps.length) * 100)}%
                    </span>
                </div>
                <div className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                    <motion.div
                        className="h-full bg-gradient-to-r from-indigo-500 to-purple-600"
                        initial={{ width: 0 }}
                        animate={{ width: `${((currentStep + 1) / derivationSteps.length) * 100}%` }}
                        transition={{ duration: 0.5 }}
                    />
                </div>
            </div>

            {/* 推导步骤 */}
            <AnimatePresence mode="wait">
                <motion.div
                    key={currentStep}
                    initial={{ opacity: 0, x: 50 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -50 }}
                    transition={{ duration: 0.3 }}
                    className="bg-white dark:bg-slate-800 rounded-xl p-8 shadow-lg mb-6"
                >
                    <div className="mb-6">
                        <div className="inline-block px-4 py-2 rounded-full bg-indigo-100 dark:bg-indigo-900 text-indigo-700 dark:text-indigo-300 font-bold text-sm mb-4">
                            步骤 {currentStep + 1}
                        </div>
                        <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
                            {derivationSteps[currentStep].title}
                        </h4>
                    </div>

                    {/* 方程 */}
                    <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-slate-700 dark:to-indigo-900 rounded-lg p-6 mb-6 border-2 border-indigo-200 dark:border-indigo-700">
                        <div className="font-mono text-lg md:text-xl text-center text-slate-800 dark:text-slate-100 font-bold">
                            {derivationSteps[currentStep].equation}
                        </div>
                    </div>

                    {/* 解释 */}
                    <div className="text-slate-700 dark:text-slate-300 leading-relaxed">
                        <div className="flex items-start gap-3">
                            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-purple-500 flex items-center justify-center text-white font-bold text-sm">
                                💡
                            </div>
                            <p className="flex-1 pt-1">
                                {derivationSteps[currentStep].explanation}
                            </p>
                        </div>
                    </div>

                    {/* 关键概念高亮 */}
                    {currentStep === 3 && (
                        <div className="mt-4 p-4 bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 rounded">
                            <div className="text-sm font-semibold text-yellow-800 dark:text-yellow-300">
                                📌 全期望公式（Law of Total Expectation）
                            </div>
                            <div className="text-sm text-yellow-700 dark:text-yellow-400 mt-1">
                                𝔼[X] = Σ_y P(Y=y) 𝔼[X | Y=y]
                            </div>
                        </div>
                    )}

                    {currentStep === 5 && (
                        <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 rounded">
                            <div className="text-sm font-semibold text-blue-800 dark:text-blue-300">
                                📌 马尔可夫性质（Markov Property）
                            </div>
                            <div className="text-sm text-blue-700 dark:text-blue-400 mt-1">
                                P(S_{t + 1} | S_t, A_t, S_{t - 1}, ...) = P(S_{t + 1} | S_t, A_t)
                            </div>
                        </div>
                    )}

                    {currentStep === 6 && (
                        <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 rounded">
                            <div className="text-sm font-semibold text-green-800 dark:text-green-300">
                                🎉 完成！Bellman 期望方程
                            </div>
                            <div className="text-sm text-green-700 dark:text-green-400 mt-1">
                                这个递归形式是动态规划、TD 学习等算法的理论基础
                            </div>
                        </div>
                    )}
                </motion.div>
            </AnimatePresence>

            {/* 控制按钮 */}
            <div className="flex justify-center gap-4">
                <button
                    onClick={prevStep}
                    disabled={currentStep === 0}
                    className="px-6 py-3 rounded-lg bg-slate-600 hover:bg-slate-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-semibold transition-colors"
                >
                    ← 上一步
                </button>
                <button
                    onClick={reset}
                    className="px-6 py-3 rounded-lg bg-purple-600 hover:bg-purple-700 text-white font-semibold transition-colors"
                >
                    🔄 重新开始
                </button>
                <button
                    onClick={nextStep}
                    disabled={currentStep === derivationSteps.length - 1}
                    className="px-6 py-3 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 disabled:cursor-not-allowed text-white font-semibold transition-colors"
                >
                    下一步 →
                </button>
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 提示：Bellman 方程将价值函数表示为递归形式，是 RL 的核心数学工具
            </div>
        </div>
    );
}
