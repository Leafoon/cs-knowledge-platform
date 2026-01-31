"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function ImplicitRewardVisualization() {
    const [policyLogit, setPolicyLogit] = useState(2.0);
    const [refLogit, setRefLogit] = useState(1.0);
    const beta = 0.1;

    // 隐式奖励计算
    const implicitReward = beta * (policyLogit - refLogit);

    // 示例：两个回复的对比
    const responses = [
        {
            id: "chosen",
            text: "这是一个详细且准确的回答...",
            policyLogProb: policyLogit,
            refLogProb: refLogit,
            label: "Chosen",
            color: "green"
        },
        {
            id: "rejected",
            text: "这是一个简短回答。",
            policyLogProb: 0.5,
            refLogProb: 0.8,
            label: "Rejected",
            color: "red"
        }
    ];

    const chosenReward = beta * (responses[0].policyLogProb - responses[0].refLogProb);
    const rejectedReward = beta * (responses[1].policyLogProb - responses[1].refLogProb);
    const rewardMargin = chosenReward - rejectedReward;

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-slate-900 dark:to-purple-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    隐式奖励可视化
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    DPO如何从策略中恢复奖励信号
                </p>
            </div>

            {/* 公式展示 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">隐式奖励公式</h4>
                <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg font-mono text-center mb-4">
                    <div className="text-lg mb-2">
                        r(x, y) = β × [log π<sub>θ</sub>(y|x) - log π<sub>ref</sub>(y|x)]
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                        β = {beta} (温度参数)
                    </div>
                </div>

                <div className="grid grid-cols-3 gap-4 text-sm">
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg border border-blue-300 dark:border-blue-700">
                        <div className="text-slate-600 dark:text-slate-400 mb-1">策略模型</div>
                        <div className="font-mono font-semibold text-blue-600 dark:text-blue-400">
                            π<sub>θ</sub>(y|x)
                        </div>
                        <div className="text-xs text-slate-500 dark:text-slate-500 mt-1">正在训练</div>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg border border-gray-300 dark:border-gray-700">
                        <div className="text-slate-600 dark:text-slate-400 mb-1">参考模型</div>
                        <div className="font-mono font-semibold text-gray-600 dark:text-gray-400">
                            π<sub>ref</sub>(y|x)
                        </div>
                        <div className="text-xs text-slate-500 dark:text-slate-500 mt-1">冻结（SFT）</div>
                    </div>

                    <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-lg border border-purple-300 dark:border-purple-700">
                        <div className="text-slate-600 dark:text-slate-400 mb-1">隐式奖励</div>
                        <div className="font-mono font-semibold text-purple-600 dark:text-purple-400">
                            r(x, y)
                        </div>
                        <div className="text-xs text-slate-500 dark:text-slate-500 mt-1">无需RM！</div>
                    </div>
                </div>
            </div>

            {/* 交互式调整 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">调整Log概率</h4>

                <div className="space-y-6">
                    <div>
                        <div className="flex justify-between mb-2">
                            <span className="font-semibold text-blue-600 dark:text-blue-400">
                                策略模型 log π<sub>θ</sub>
                            </span>
                            <span className="font-mono text-blue-600 dark:text-blue-400">{policyLogit.toFixed(2)}</span>
                        </div>
                        <input
                            type="range"
                            min="-2"
                            max="4"
                            step="0.1"
                            value={policyLogit}
                            onChange={(e) => setPolicyLogit(parseFloat(e.target.value))}
                            className="w-full h-3 bg-blue-200 rounded-lg appearance-none cursor-pointer dark:bg-blue-900"
                        />
                    </div>

                    <div>
                        <div className="flex justify-between mb-2">
                            <span className="font-semibold text-gray-600 dark:text-gray-400">
                                参考模型 log π<sub>ref</sub>
                            </span>
                            <span className="font-mono text-gray-600 dark:text-gray-400">{refLogit.toFixed(2)}</span>
                        </div>
                        <input
                            type="range"
                            min="-2"
                            max="4"
                            step="0.1"
                            value={refLogit}
                            onChange={(e) => setRefLogit(parseFloat(e.target.value))}
                            className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                        />
                    </div>
                </div>

                {/* 计算结果 */}
                <div className="mt-6 grid grid-cols-3 gap-4">
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg text-center">
                        <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">π差异</div>
                        <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                            {(policyLogit - refLogit).toFixed(2)}
                        </div>
                    </div>

                    <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg text-center">
                        <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">隐式奖励</div>
                        <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                            {implicitReward.toFixed(3)}
                        </div>
                    </div>

                    <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg text-center">
                        <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">学习信号</div>
                        <div className={`text-3xl font-bold ${implicitReward > 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}>
                            {implicitReward > 0 ? "↑" : "↓"}
                        </div>
                    </div>
                </div>
            </div>

            {/* 偏好对示例 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">偏好对的隐式奖励</h4>

                <div className="grid grid-cols-2 gap-4">
                    {responses.map((resp) => {
                        const reward = beta * (resp.policyLogProb - resp.refLogProb);

                        return (
                            <div
                                key={resp.id}
                                className={`p-4 rounded-lg border-2 border-${resp.color}-500 bg-${resp.color}-50 dark:bg-${resp.color}-900/20`}
                            >
                                <div className="flex items-center justify-between mb-3">
                                    <span className={`font-semibold text-${resp.color}-700 dark:text-${resp.color}-400`}>
                                        {resp.label}
                                    </span>
                                    <span className={`text-2xl font-bold text-${resp.color}-600 dark:text-${resp.color}-400`}>
                                        {reward.toFixed(3)}
                                    </span>
                                </div>

                                <div className="text-sm text-slate-700 dark:text-slate-300 italic mb-3">
                                    {resp.text}
                                </div>

                                <div className="space-y-2 text-xs">
                                    <div className="flex justify-between">
                                        <span className="text-slate-600 dark:text-slate-400">log π<sub>θ</sub>:</span>
                                        <span className="font-mono">{resp.policyLogProb.toFixed(2)}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-600 dark:text-slate-400">log π<sub>ref</sub>:</span>
                                        <span className="font-mono">{resp.refLogProb.toFixed(2)}</span>
                                    </div>
                                    <div className="flex justify-between font-semibold pt-2 border-t border-gray-300 dark:border-gray-700">
                                        <span>隐式奖励:</span>
                                        <span className={`text-${resp.color}-600 dark:text-${resp.color}-400`}>
                                            {reward.toFixed(3)}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>

                {/* 奖励Margin */}
                <div className="mt-4 bg-purple-100 dark:bg-purple-900/30 p-4 rounded-lg border-2 border-purple-500">
                    <div className="text-center">
                        <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">
                            奖励Margin（Chosen - Rejected）
                        </div>
                        <div className="text-4xl font-bold text-purple-600 dark:text-purple-400">
                            {rewardMargin > 0 ? "+" : ""}{rewardMargin.toFixed(3)}
                        </div>
                        <div className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                            {rewardMargin > 0 ? "✅ 偏好正确：chosen奖励更高" : "❌ 需要优化：chosen奖励偏低"}
                        </div>
                    </div>
                </div>
            </div>

            {/* 关键洞察 */}
            <div className="bg-indigo-100 dark:bg-indigo-900/30 p-4 rounded-lg">
                <h5 className="font-semibold text-indigo-700 dark:text-indigo-400 mb-2">🔍 关键洞察</h5>
                <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
                    <li>• DPO <strong>无需训练显式RM</strong>：奖励从策略隐式恢复</li>
                    <li>• 策略越偏离参考模型（π<sub>θ</sub> ≠ π<sub>ref</sub>），奖励信号越强</li>
                    <li>• β控制奖励尺度：β越大，KL惩罚越强</li>
                    <li>• 只要 r(chosen) &gt; r(rejected)，DPO就能学习偏好</li>
                </ul>
            </div>
        </div>
    );
}
