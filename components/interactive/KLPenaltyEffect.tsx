"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function KLPenaltyEffect() {
    const [klCoef, setKlCoef] = useState(0.05);

    // 模拟数据
    const rmScore = 8.0;
    const klDivergence = 2.5;
    const finalReward = rmScore - klCoef * klDivergence;

    // 不同KL系数的影响
    const klScenarios = [
        { coef: 0.0, name: "无惩罚", color: "red", risk: "高风险：奖励Hacking" },
        { coef: 0.01, name: "弱惩罚", color: "orange", risk: "中风险：可能过拟合" },
        { coef: 0.05, name: "平衡", color: "green", risk: "推荐：平衡奖励与稳定性" },
        { coef: 0.1, name: "强惩罚", color: "blue", risk: "保守：策略更新慢" },
        { coef: 0.5, name: "过强", color: "purple", risk: "极保守：几乎不更新" },
    ];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-slate-900 dark:to-green-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    KL 惩罚的作用
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    防止策略过度偏离参考模型
                </p>
            </div>

            {/* 公式 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">奖励计算公式</h4>
                <div className="bg-emerald-50 dark:bg-emerald-900/20 p-4 rounded-lg font-mono text-center">
                    <div className="text-lg mb-2">
                        r<sub>final</sub> = r<sub>RM</sub> - β × KL(π<sub>θ</sub> || π<sub>ref</sub>)
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-400 mt-3 space-y-1">
                        <div>r<sub>RM</sub>: 奖励模型打分</div>
                        <div>β: KL系数（可调）</div>
                        <div>KL: 策略与参考模型的KL散度</div>
                    </div>
                </div>
            </div>

            {/* 交互式调整 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">调整 KL 系数 (β)</h4>

                <div className="mb-6">
                    <div className="flex justify-between mb-2">
                        <span className="font-semibold text-green-600 dark:text-green-400">β = {klCoef.toFixed(3)}</span>
                        <span className="text-sm text-slate-600 dark:text-slate-400">
                            {klCoef === 0 ? "无惩罚" : klCoef < 0.03 ? "弱" : klCoef < 0.08 ? "平衡" : klCoef < 0.2 ? "强" : "过强"}
                        </span>
                    </div>
                    <input
                        type="range"
                        min="0"
                        max="0.5"
                        step="0.01"
                        value={klCoef}
                        onChange={(e) => setKlCoef(parseFloat(e.target.value))}
                        className="w-full h-3 bg-green-200 rounded-lg appearance-none cursor-pointer dark:bg-green-900"
                    />
                    <div className="flex justify-between text-xs text-slate-500 dark:text-slate-500 mt-1">
                        <span>0.0</span>
                        <span>0.25</span>
                        <span>0.5</span>
                    </div>
                </div>

                {/* 奖励分解 */}
                <div className="grid grid-cols-3 gap-4">
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg text-center">
                        <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">RM分数</div>
                        <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                            {rmScore.toFixed(2)}
                        </div>
                    </div>

                    <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg text-center">
                        <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">KL惩罚项</div>
                        <div className="text-3xl font-bold text-orange-600 dark:text-orange-400">
                            -{(klCoef * klDivergence).toFixed(2)}
                        </div>
                    </div>

                    <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg text-center">
                        <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">最终奖励</div>
                        <div className="text-3xl font-bold text-green-600 dark:text-green-400">
                            {finalReward.toFixed(2)}
                        </div>
                    </div>
                </div>
            </div>

            {/* 不同KL系数对比 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">不同 KL 系数对比</h4>

                <div className="space-y-3">
                    {klScenarios.map((scenario) => {
                        const reward = rmScore - scenario.coef * klDivergence;
                        const isActive = Math.abs(scenario.coef - klCoef) < 0.015;

                        return (
                            <motion.div
                                key={scenario.coef}
                                className={`p-4 rounded-lg border-2 transition ${isActive
                                    ? `border-${scenario.color}-500 bg-${scenario.color}-50 dark:bg-${scenario.color}-900/20`
                                    : "border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50"
                                    }`}
                                animate={{ scale: isActive ? 1.02 : 1 }}
                            >
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4">
                                        <div className={`w-16 text-center font-mono font-bold text-${scenario.color}-600 dark:text-${scenario.color}-400`}>
                                            β={scenario.coef.toFixed(2)}
                                        </div>
                                        <div>
                                            <div className="font-semibold text-slate-800 dark:text-slate-100">
                                                {scenario.name}
                                            </div>
                                            <div className={`text-sm ${isActive ? `text-${scenario.color}-600 dark:text-${scenario.color}-400` : "text-slate-600 dark:text-slate-400"}`}>
                                                {scenario.risk}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="text-right">
                                        <div className="text-sm text-slate-600 dark:text-slate-400">最终奖励</div>
                                        <div className={`text-2xl font-bold ${isActive ? `text-${scenario.color}-600 dark:text-${scenario.color}-400` : "text-slate-800 dark:text-slate-100"}`}>
                                            {reward.toFixed(2)}
                                        </div>
                                    </div>
                                </div>

                                {/* 奖励条 */}
                                <div className="mt-3 h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full bg-${scenario.color}-500`}
                                        style={{ width: `${(reward / rmScore) * 100}%` }}
                                    />
                                </div>
                            </motion.div>
                        );
                    })}
                </div>
            </div>

            {/* 效果说明 */}
            <div className="grid grid-cols-2 gap-4">
                <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border-2 border-red-300 dark:border-red-700">
                    <h5 className="font-semibold text-red-700 dark:text-red-400 mb-2 flex items-center gap-2">
                        <span>⚠️</span> β 过小的问题
                    </h5>
                    <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
                        <li>• 策略可能剧烈变化</li>
                        <li>• 奖励Hacking风险高</li>
                        <li>• 生成不自然的文本</li>
                        <li>• 训练不稳定</li>
                    </ul>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-2 border-blue-300 dark:border-blue-700">
                    <h5 className="font-semibold text-blue-700 dark:text-blue-400 mb-2 flex items-center gap-2">
                        <span>❄️</span> β 过大的问题
                    </h5>
                    <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
                        <li>• 策略几乎不更新</li>
                        <li>• 学习速度慢</li>
                        <li>• 浪费计算资源</li>
                        <li>• 难以改进性能</li>
                    </ul>
                </div>
            </div>

            <div className="mt-6 bg-green-100 dark:bg-green-900/30 p-4 rounded-lg text-center text-sm text-sl ate-700 dark:text-slate-300">
                💡 推荐范围：β ∈ [0.01, 0.1]，需根据具体任务调整
            </div>
        </div>
    );
}
