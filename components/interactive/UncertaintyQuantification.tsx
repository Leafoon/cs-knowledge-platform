"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function UncertaintyQuantification() {
    const [numSamples, setNumSamples] = useState(50);
    const [showEpistemic, setShowEpistemic] = useState(true);
    const [showAleatoric, setShowAleatoric] = useState(true);

    // Simulated predictions from ensemble/Bayesian network
    const generatePredictions = (n: number) => {
        const predictions = [];
        for (let i = 0; i < n; i++) {
            // 4 actions
            const probs = [
                0.6 + Math.random() * 0.25,  // Action 0 (high mean, low variance - epistemic)
                0.15 + Math.random() * 0.15, // Action 1
                0.15 + Math.random() * 0.15, // Action 2
                0.10 + Math.random() * 0.10  // Action 3
            ];
            const sum = probs.reduce((a, b) => a + b);
            predictions.push(probs.map(p => p / sum));
        }
        return predictions;
    };

    const predictions = generatePredictions(numSamples);

    // Calculate statistics
    const meanProbs = predictions[0].map((_, actionIdx) =>
        predictions.reduce((sum, pred) => sum + pred[actionIdx], 0) / predictions.length
    );

    const variances = predictions[0].map((_, actionIdx) => {
        const mean = meanProbs[actionIdx];
        return predictions.reduce((sum, pred) => sum + Math.pow(pred[actionIdx] - mean, 2), 0) / predictions.length;
    });

    const actions = ["前进", "后退", "左转", "右转"];

    // Epistemic (model uncertainty) vs Aleatoric (data uncertainty)
    const epistemicUncertainty = Math.sqrt(variances.reduce((a, b) => a + b) / variances.length);
    const aleatoricUncertainty = 0.15; // Simulated environment noise
    const totalUncertainty = Math.sqrt(epistemicUncertainty ** 2 + aleatoricUncertainty ** 2);

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-slate-900 dark:to-purple-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    不确定性量化
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Uncertainty Quantification: Epistemic vs Aleatoric
                </p>
            </div>

            {/* Sample Size Control */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg mb-6">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                        采样数量（模型/权重）
                    </span>
                    <span className="text-lg font-bold text-purple-600 dark:text-purple-400">
                        {numSamples}
                    </span>
                </div>
                <input
                    type="range"
                    min="10"
                    max="100"
                    step="10"
                    value={numSamples}
                    onChange={(e) => setNumSamples(Number(e.target.value))}
                    className="w-full"
                />
            </div>

            {/* Uncertainty Decomposition */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg">
                    <div className={`text-sm font-semibold mb-2 ${showEpistemic ? 'text-blue-700 dark:text-blue-400' : 'text-gray-400'}`}>
                        认知不确定性
                    </div>
                    <div className={`text-3xl font-bold mb-2 ${showEpistemic ? 'text-blue-600 dark:text-blue-400' : 'text-gray-400'}`}>
                        {epistemicUncertainty.toFixed(3)}
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-400 mb-3">
                        模型参数不确定性
                    </div>
                    <label className="flex items-center gap-2 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={showEpistemic}
                            onChange={(e) => setShowEpistemic(e.target.checked)}
                            className="w-4 h-4"
                        />
                        <span className="text-xs text-slate-600 dark:text-slate-400">显示</span>
                    </label>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg">
                    <div className={`text-sm font-semibold mb-2 ${showAleatoric ? 'text-green-700 dark:text-green-400' : 'text-gray-400'}`}>
                        偶然不确定性
                    </div>
                    <div className={`text-3xl font-bold mb-2 ${showAleatoric ? 'text-green-600 dark:text-green-400' : 'text-gray-400'}`}>
                        {aleatoricUncertainty.toFixed(3)}
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-400 mb-3">
                        环境随机性
                    </div>
                    <label className="flex items-center gap-2 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={showAleatoric}
                            onChange={(e) => setShowAleatoric(e.target.checked)}
                            className="w-4 h-4"
                        />
                        <span className="text-xs text-slate-600 dark:text-slate-400">显示</span>
                    </label>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg border-2 border-purple-500">
                    <div className="text-sm font-semibold text-purple-700 dark:text-purple-400 mb-2">
                        总不确定性
                    </div>
                    <div className="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-2">
                        {totalUncertainty.toFixed(3)}
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                        √(σ²_epistemic + σ²_aleatoric)
                    </div>
                </div>
            </div>

            {/* Prediction Distribution */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    预测分布（{numSamples}次采样）
                </h4>

                <div className="grid grid-cols-4 gap-4">
                    {actions.map((action, actionIdx) => {
                        const mean = meanProbs[actionIdx];
                        const variance = variances[actionIdx];
                        const std = Math.sqrt(variance);

                        return (
                            <div key={action} className="text-center">
                                <div className="text-sm font-semibold mb-3 text-slate-700 dark:text-slate-300">
                                    {action}
                                </div>

                                {/* Mean */}
                                <div className="mb-2">
                                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">
                                        均值
                                    </div>
                                    <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded relative overflow-hidden">
                                        <motion.div
                                            className="absolute bottom-0 w-full bg-purple-500"
                                            initial={{ height: 0 }}
                                            animate={{ height: `${mean * 100}%` }}
                                            transition={{ duration: 0.5 }}
                                        />
                                        <div className="absolute inset-0 flex items-center justify-center text-xs font-bold text-white">
                                            {(mean * 100).toFixed(1)}%
                                        </div>

                                        {/* Uncertainty range */}
                                        {showEpistemic && (
                                            <>
                                                <div
                                                    className="absolute w-full bg-blue-300 opacity-30"
                                                    style={{
                                                        bottom: `${Math.max(0, (mean - std) * 100)}%`,
                                                        height: `${Math.min(2 * std * 100, 100 - (mean - std) * 100)}%`
                                                    }}
                                                />
                                            </>
                                        )}
                                    </div>
                                </div>

                                {/* Variance */}
                                <div className="text-xs text-slate-600 dark:text-slate-400">
                                    σ = {std.toFixed(3)}
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Decision with Uncertainty */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    不确定性感知决策
                </h4>

                <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                        <div className="font-semibold text-green-700 dark:text-green-400 mb-2">
                            低不确定性场景
                        </div>
                        <div className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                            <div>• σ_total {"<"} 0.15</div>
                            <div>• 使用均值预测</div>
                            <div>• 贪婪策略</div>
                            <div>✅ 置信度高，执行最优动作</div>
                        </div>
                    </div>

                    <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-500">
                        <div className="font-semibold text-orange-700 dark:text-orange-400 mb-2">
                            高不确定性场景
                        </div>
                        <div className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                            <div>• σ_total ≥ {totalUncertainty.toFixed(2)}</div>
                            <div>• 保守策略</div>
                            <div>• 安全回退或探索</div>
                            <div>⚠️ 避免高风险决策</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Methods Comparison */}
            <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-4 rounded-lg">
                <h5 className="font-semibold text-slate-700 dark:text-slate-300 mb-3">
                    不确定性量化方法
                </h5>
                <div className="grid grid-cols-3 gap-3">
                    <div className="p-3 bg-white dark:bg-slate-800 rounded-lg text-xs">
                        <div className="font-semibold text-blue-600 dark:text-blue-400 mb-1">
                            贝叶斯神经网络
                        </div>
                        <div className="text-slate-600 dark:text-slate-400">
                            权重分布 + 变分推断
                        </div>
                    </div>
                    <div className="p-3 bg-white dark:bg-slate-800 rounded-lg text-xs">
                        <div className="font-semibold text-purple-600 dark:text-purple-400 mb-1">
                            Ensemble方法
                        </div>
                        <div className="text-slate-600 dark:text-slate-400">
                            多模型聚合 + 分歧度
                        </div>
                    </div>
                    <div className="p-3 bg-white dark:bg-slate-800 rounded-lg text-xs">
                        <div className="font-semibold text-pink-600 dark:text-pink-400 mb-1">
                            MC Dropout
                        </div>
                        <div className="text-slate-600 dark:text-slate-400">
                            测试时Dropout采样
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
