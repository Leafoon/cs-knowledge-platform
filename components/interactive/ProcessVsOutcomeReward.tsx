"use client";

import { useState } from "react";
import { motion } from "framer-motion";

type RewardType = "outcome" | "process";

export function ProcessVsOutcomeReward() {
    const [rewardType, setRewardType] = useState<RewardType>("outcome");

    // 示例推理问题
    const problem = "Roger有5个网球。他又买了2罐网球，每罐3个。他现在有几个网球？";

    const steps = [
        { id: 1, text: "Roger开始有5个球", isCorrect: true },
        { id: 2, text: "他买了2罐网球", isCorrect: true },
        { id: 3, text: "每罐有3个球，所以 2 × 3 = 5", isCorrect: false },  // 错误！
        { id: 4, text: "总共 5 + 5 = 10个球", isCorrect: false },  // 基于错误延续
    ];

    const finalAnswer = "10";
    const correctAnswer = "11";

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-slate-900 dark:to-emerald-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    过程奖励 vs 结果奖励
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    密集信号 vs 稀疏信号：哪个更有效？
                </p>
            </div>

            {/* 选择奖励类型 */}
            <div className="grid grid-cols-2 gap-4 mb-6">
                <button
                    onClick={() => setRewardType("outcome")}
                    className={`p-4 rounded-xl border-2 transition ${rewardType === "outcome"
                            ? "border-red-500 bg-red-50 dark:bg-red-900/20"
                            : "border-gray-200 dark:border-gray-700 bg-white dark:bg-slate-800"
                        }`}
                >
                    <div className="text-xl font-bold text-red-600 dark:text-red-400 mb-1">
                        Outcome Reward
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-400">
                        只看最终答案正确性
                    </div>
                </button>

                <button
                    onClick={() => setRewardType("process")}
                    className={`p-4 rounded-xl border-2 transition ${rewardType === "process"
                            ? "border-green-500 bg-green-50 dark:bg-green-900/20"
                            : "border-gray-200 dark:border-gray-700 bg-white dark:bg-slate-800"
                        }`}
                >
                    <div className="text-xl font-bold text-green-600 dark:text-green-400 mb-1">
                        Process Reward
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-400">
                        评估每个中间步骤
                    </div>
                </button>
            </div>

            {/* 问题展示 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">示例问题</h4>
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-2 border-blue-300 dark:border-blue-700">
                    <div className="font-semibold text-blue-700 dark:text-blue-400 mb-2">问题：</div>
                    <div className="text-slate-800 dark:text-slate-100">{problem}</div>
                    <div className="mt-3 text-sm text-green-600 dark:text-green-400">
                        ✅ 正确答案：{correctAnswer}
                    </div>
                </div>
            </div>

            {/* 推理步骤 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    模型推理过程
                </h4>

                <div className="space-y-3">
                    {steps.map((step, idx) => (
                        <motion.div
                            key={step.id}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.1 }}
                            className={`p-4 rounded-lg border-2 ${rewardType === "process"
                                    ? step.isCorrect
                                        ? "border-green-500 bg-green-50 dark:bg-green-900/20"
                                        : "border-red-500 bg-red-50 dark:bg-red-900/20"
                                    : "border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-800"
                                }`}
                        >
                            <div className="flex items-start justify-between">
                                <div className="flex items-start gap-3 flex-1">
                                    <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-white ${rewardType === "process"
                                            ? step.isCorrect ? "bg-green-600" : "bg-red-600"
                                            : "bg-gray-600"
                                        }`}>
                                        {step.id}
                                    </div>
                                    <div className="flex-1">
                                        <div className="text-slate-800 dark:text-slate-100">{step.text}</div>
                                        {rewardType === "process" && (
                                            <motion.div
                                                initial={{ opacity: 0, y: -10 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                className={`mt-2 text-sm font-semibold ${step.isCorrect
                                                        ? "text-green-600 dark:text-green-400"
                                                        : "text-red-600 dark:text-red-400"
                                                    }`}
                                            >
                                                {step.isCorrect ? "✅ 步骤正确" : "❌ 步骤错误"}
                                            </motion.div>
                                        )}
                                    </div>
                                </div>

                                {rewardType === "process" && (
                                    <div className={`text-3xl ${step.isCorrect ? "text-green-600" : "text-red-600"}`}>
                                        {step.isCorrect ? "+1" : "-1"}
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    ))}
                </div>

                {/* 最终答案 */}
                <div className={`mt-4 p-4 rounded-lg border-2 ${rewardType === "outcome"
                        ? "border-red-500 bg-red-50 dark:bg-red-900/20"
                        : "border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-800"
                    }`}>
                    <div className="flex items-center justify-between">
                        <div>
                            <div className="font-semibold text-slate-800 dark:text-slate-100 mb-1">
                                最终答案：{finalAnswer}
                            </div>
                            <div className="text-sm text-red-600 dark:text-red-400">
                                ❌ 答案错误（正确答案是{correctAnswer}）
                            </div>
                        </div>
                        {rewardType === "outcome" && (
                            <div className="text-4xl font-bold text-red-600 dark:text-red-400">
                                -1
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* 奖励信号对比 */}
            <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl border-2 border-red-500">
                    <h5 className="font-semibold text-red-700 dark:text-red-400 mb-4">Outcome Reward</h5>
                    <div className="space-y-3 text-sm">
                        <div>
                            <div className="text-slate-600 dark:text-slate-400 mb-1">奖励信号</div>
                            <div className="font-mono text-2xl text-red-600 dark:text-red-400">-1</div>
                        </div>
                        <div>
                            <div className="text-slate-600 dark:text-slate-400 mb-1">信息量</div>
                            <div className="font-semibold text-slate-800 dark:text-slate-100">稀疏（仅1个信号）</div>
                        </div>
                        <div>
                            <div className="text-slate-600 dark:text-slate-400 mb-1">错误定位</div>
                            <div className="font-semibold text-red-600 dark:text-red-400">❌ 无法定位</div>
                        </div>
                    </div>
                </div>

                <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border-2 border-green-500">
                    <h5 className="font-semibold text-green-700 dark:text-green-400 mb-4">Process Reward</h5>
                    <div className="space-y-3 text-sm">
                        <div>
                            <div className="text-slate-600 dark:text-slate-400 mb-1">奖励信号</div>
                            <div className="font-mono text-2xl text-green-600 dark:text-green-400">
                                +1, +1, -1, -1
                            </div>
                        </div>
                        <div>
                            <div className="text-slate-600 dark:text-slate-400 mb-1">信息量</div>
                            <div className="font-semibold text-slate-800 dark:text-slate-100">密集（4个信号）</div>
                        </div>
                        <div>
                            <div className="text-slate-600 dark:text-slate-400 mb-1">错误定位</div>
                            <div className="font-semibold text-green-600 dark:text-green-400">✅ 精准定位步骤3</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* 性能对比 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    实验结果对比（MATH数据集）
                </h4>

                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b-2 border-gray-300 dark:border-gray-700">
                                <th className="text-left py-3 px-4 text-slate-600 dark:text-slate-400">指标</th>
                                <th className="text-center py-3 px-4 text-red-600 dark:text-red-400">Outcome</th>
                                <th className="text-center py-3 px-4 text-green-600 dark:text-green-400">Process</th>
                            </tr>
                        </thead>
                        <tbody className="text-slate-700 dark:text-slate-300">
                            <tr className="border-b border-gray-200 dark:border-gray-700">
                                <td className="py-3 px-4">准确率</td>
                                <td className="py-3 px-4 text-center">26.6%</td>
                                <td className="py-3 px-4 text-center font-bold text-green-600">78.2%</td>
                            </tr>
                            <tr className="border-b border-gray-200 dark:border-gray-700">
                                <td className="py-3 px-4">样本效率</td>
                                <td className="py-3 px-4 text-center">低</td>
                                <td className="py-3 px-4 text-center font-bold text-green-600">高（3-5x）</td>
                            </tr>
                            <tr className="border-b border-gray-200 dark:border-gray-700">
                                <td className="py-3 px-4">泛化能力</td>
                                <td className="py-3 px-4 text-center">弱</td>
                                <td className="py-3 px-4 text-center font-bold text-green-600">强</td>
                            </tr>
                            <tr>
                                <td className="py-3 px-4">标注成本</td>
                                <td className="py-3 px-4 text-center font-bold text-red-600">低</td>
                                <td className="py-3 px-4 text-center">高（需逐步标注）</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div className="mt-6 bg-teal-100 dark:bg-teal-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                💡 <strong>PRM800K论文结论</strong>: Process Reward在数学推理任务上将准确率从26.6%提升到78.2%！
            </div>
        </div>
    );
}
