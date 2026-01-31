"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function QMIXMixingNetwork() {
    const [numAgents] = useState(3);
    const [agentQValues, setAgentQValues] = useState([5.0, 3.0, 4.0]);
    const [hyperWeights, setHyperWeights] = useState([0.4, 0.3, 0.3]);

    // 计算混合Q值（简化版QMIX）
    const mixedQ = agentQValues.reduce((sum, q, i) => sum + q * hyperWeights[i], 0);

    const handleQChange = (index: number, value: number) => {
        const newQValues = [...agentQValues];
        newQValues[index] = value;
        setAgentQValues(newQValues);
    };

    const handleWeightChange = (index: number, value: number) => {
        const newWeights = [...hyperWeights];
        newWeights[index] = value;
        // 归一化权重
        const sum = newWeights.reduce((a, b) => a + b, 0);
        const normalized = newWeights.map(w => w / sum);
        setHyperWeights(normalized);
    };

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-lime-50 to-green-50 dark:from-slate-900 dark:to-lime-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    QMIX Mixing Network（交互式）
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    调整智能体Q值和权重，观察混合效果
                </p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">个体Q值</h4>
                <div className="space-y-4">
                    {agentQValues.map((q, i) => (
                        <div key={i} className="space-y-2">
                            <div className="flex justify-between items-center">
                                <label className="text-sm font-semibold">
                                    智能体 {i + 1}: Q<sup>{i + 1}</sup>
                                </label>
                                <span className="text-sm font-mono bg-lime-100 dark:bg-lime-900/30 px-3 py-1 rounded">
                                    {q.toFixed(2)}
                                </span>
                            </div>
                            <input
                                type="range"
                                min="-5"
                                max="10"
                                step="0.1"
                                value={q}
                                onChange={(e) => handleQChange(i, parseFloat(e.target.value))}
                                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                            />
                        </div>
                    ))}
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">超网络权重（单调性约束: w ≥ 0）</h4>
                <div className="space-y-4">
                    {hyperWeights.map((w, i) => (
                        <div key={i} className="space-y-2">
                            <div className="flex justify-between items-center">
                                <label className="text-sm font-semibold">
                                    w<sub>{i + 1}</sub>
                                </label>
                                <span className="text-sm font-mono bg-green-100 dark:bg-green-900/30 px-3 py-1 rounded">
                                    {w.toFixed(3)}
                                </span>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.01"
                                value={w}
                                onChange={(e) => handleWeightChange(i, parseFloat(e.target.value))}
                                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                            />
                        </div>
                    ))}
                </div>
            </div>

            <motion.div
                className="bg-gradient-to-r from-emerald-500 to-green-600 rounded-xl p-8 shadow-2xl text-center"
                animate={{ scale: [1, 1.02, 1] }}
                transition={{ duration: 0.5, repeat: Infinity, repeatDelay: 1 }}
            >
                <h4 className="text-xl font-bold text-white mb-2">混合Q值 (Q<sub>tot</sub>)</h4>
                <div className="text-5xl font-bold text-white mb-4">
                    {mixedQ.toFixed(2)}
                </div>
                <div className="text-sm text-emerald-100 font-mono">
                    Q<sub>tot</sub> = Σ w<sub>i</sub> · Q<sup>i</sup>
                </div>
            </motion.div>

            <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">单调性验证</h4>
                <div className="grid grid-cols-3 gap-4 text-sm">
                    {agentQValues.map((_, i) => (
                        <div key={i} className="p-4 bg-green-50 dark:bg-green-900/20 rounded text-center">
                            <div className="font-bold text-green-700 dark:text-green-400">
                                ∂Q<sub>tot</sub>/∂Q<sup>{i + 1}</sup>
                            </div>
                            <div className="text-2xl font-mono mt-2">
                                {hyperWeights[i].toFixed(3)}
                            </div>
                            <div className="text-xs mt-1 text-green-600">
                                ≥ 0 ✓
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-600 dark:text-slate-400">
                💡 单调性保证：增加任何智能体的Q值，混合Q值也增加（保持一致性）
            </div>
        </div>
    );
}
