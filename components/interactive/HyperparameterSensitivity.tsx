"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function HyperparameterSensitivity() {
    const [hyperparameter, setHyperparameter] = useState<"lr" | "gamma" | "epsilon">("lr");
    const [value, setValue] = useState(3e-4);

    // Simulate performance based on hyperparameter value
    const computePerformance = (param: string, val: number) => {
        if (param === "lr") {
            // Learning rate: peak around 3e-4
            const optimal = 3e-4;
            const distance = Math.abs(Math.log10(val) - Math.log10(optimal));
            return Math.max(0, 100 - distance * 40);
        } else if (param === "gamma") {
            // Gamma: higher is generally better but with diminishing returns
            return 50 + val * 50;
        } else {
            // Epsilon: moderate values work best
            const optimal = 0.1;
            const distance = Math.abs(val - optimal);
            return Math.max(0, 100 - distance * 200);
        }
    };

    const currentPerformance = computePerformance(hyperparameter, value);

    // Generate sensitivity curve
    const generateCurve = (param: string) => {
        const points = [];
        let values: number[] = [];

        if (param === "lr") {
            values = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2];
        } else if (param === "gamma") {
            values = Array.from({ length: 20 }, (_, i) => 0.9 + i * 0.005);
        } else {
            values = Array.from({ length: 20 }, (_, i) => i * 0.025);
        }

        for (const v of values) {
            points.push({
                value: v,
                performance: computePerformance(param, v)
            });
        }

        return points;
    };

    const curvePoints = generateCurve(hyperparameter);

    const hyperparameters = [
        { id: "lr", name: "Learning Rate", range: [1e-5, 1e-2], optimal: 3e-4, color: "blue" },
        { id: "gamma", name: "Discount γ", range: [0.9, 0.999], optimal: 0.99, color: "purple" },
        { id: "epsilon", name: "Exploration ε", range: [0.0, 0.5], optimal: 0.1, color: "green" }
    ];

    const currentParam = hyperparameters.find(h => h.id === hyperparameter)!;

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-slate-900 dark:to-blue-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    超参数敏感性分析
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Hyperparameter Sensitivity Analysis
                </p>
            </div>

            {/* Hyperparameter Selection */}
            <div className="flex gap-3 justify-center mb-6">
                {hyperparameters.map((hp) => (
                    <button
                        key={hp.id}
                        onClick={() => {
                            setHyperparameter(hp.id as any);
                            setValue(hp.optimal);
                        }}
                        className={`px-6 py-3 rounded-xl font-semibold transition ${hyperparameter === hp.id
                                ? `bg-${hp.color}-600 text-white shadow-lg`
                                : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                            }`}
                    >
                        {hp.name}
                    </button>
                ))}
            </div>

            {/* Value Slider */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg mb-6">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                        {currentParam.name}值
                    </span>
                    <span className="text-lg font-bold text-purple-600 dark:text-purple-400">
                        {hyperparameter === "lr" ? value.toExponential(1) : value.toFixed(3)}
                    </span>
                </div>
                <input
                    type="range"
                    min={currentParam.range[0]}
                    max={currentParam.range[1]}
                    step={hyperparameter === "lr" ? 0.0001 : hyperparameter === "gamma" ? 0.001 : 0.01}
                    value={value}
                    onChange={(e) => setValue(Number(e.target.value))}
                    className="w-full"
                />
                <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 mt-1">
                    <span>{hyperparameter === "lr" ? currentParam.range[0].toExponential(0) : currentParam.range[0]}</span>
                    <span>最优: {hyperparameter === "lr" ? currentParam.optimal.toExponential(0) : currentParam.optimal}</span>
                    <span>{hyperparameter === "lr" ? currentParam.range[1].toExponential(0) : currentParam.range[1]}</span>
                </div>
            </div>

            {/* Performance Gauge */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    预测性能
                </h4>

                <div className="relative h-32 bg-gray-200 dark:bg-gray-700 rounded-lg overflow-hidden">
                    <motion.div
                        className={`absolute left-0 top-0 bottom-0 ${currentPerformance > 80
                                ? "bg-green-500"
                                : currentPerformance > 60
                                    ? "bg-yellow-500"
                                    : currentPerformance > 40
                                        ? "bg-orange-500"
                                        : "bg-red-500"
                            }`}
                        initial={{ width: 0 }}
                        animate={{ width: `${currentPerformance}%` }}
                        transition={{ duration: 0.5 }}
                    />
                    <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-4xl font-bold text-white drop-shadow-lg">
                            {currentPerformance.toFixed(1)}%
                        </span>
                    </div>
                </div>

                <div className="mt-3 text-center text-sm text-slate-600 dark:text-slate-400">
                    {currentPerformance > 80
                        ? "✅ 优秀配置 - 预期高性能"
                        : currentPerformance > 60
                            ? "⚠️ 可接受 - 但可优化"
                            : "❌ 次优配置 - 建议调整"}
                </div>
            </div>

            {/* Sensitivity Curve */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    敏感性曲线
                </h4>

                <svg width="100%" height="300" viewBox="0 0 600 300" className="mx-auto">
                    {/* Axes */}
                    <line x1="50" y1="250" x2="550" y2="250" stroke="currentColor" className="text-gray-400" strokeWidth="2" />
                    <line x1="50" y1="50" x2="50" y2="250" stroke="currentColor" className="text-gray-400" strokeWidth="2" />

                    {/* Optimal zone */}
                    <rect x="50" y="50" width="500" height="40" className="fill-green-200 dark:fill-green-900" opacity="0.3" />

                    {/* Curve */}
                    <motion.path
                        d={`M ${curvePoints.map((p, i) => {
                            const x = 50 + (i / (curvePoints.length - 1)) * 500;
                            const y = 250 - (p.performance / 100) * 200;
                            return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                        }).join(' ')}`}
                        stroke="currentColor"
                        className={`text-${currentParam.color}-600`}
                        strokeWidth="3"
                        fill="none"
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                        transition={{ duration: 1 }}
                    />

                    {/* Current value marker */}
                    {curvePoints.map((p, i) => {
                        const isClosest = Math.abs(p.value - value) < (hyperparameter === "lr" ? 0.001 : 0.01);
                        if (!isClosest) return null;

                        const x = 50 + (i / (curvePoints.length - 1)) * 500;
                        const y = 250 - (p.performance / 100) * 200;

                        return (
                            <g key={i}>
                                <circle cx={x} cy={y} r="6" className="fill-red-600" />
                                <line x1={x} y1={y} x2={x} y2="250" stroke="red" strokeWidth="2" strokeDasharray="5,5" />
                            </g>
                        );
                    })}

                    {/* Labels */}
                    <text x="300" y="290" textAnchor="middle" className="fill-current text-slate-600 dark:text-slate-400 text-sm">
                        {currentParam.name}
                    </text>
                    <text x="25" y="150" textAnchor="middle" className="fill-current text-slate-600 dark:text-slate-400 text-sm" transform="rotate(-90 25 150)">
                        Performance
                    </text>
                </svg>
            </div>

            {/* Recommendations */}
            <div className="mt-6 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 p-4 rounded-lg">
                <h5 className="font-semibold text-slate-700 dark:text-slate-300 mb-3">
                    调优建议
                </h5>
                <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="p-3 bg-white dark:bg-slate-800 rounded-lg">
                        <div className="font-semibold text-blue-600 dark:text-blue-400 mb-1">
                            网格搜索
                        </div>
                        <div className="text-slate-600 dark:text-slate-400">
                            系统遍历所有组合
                        </div>
                    </div>
                    <div className="p-3 bg-white dark:bg-slate-800 rounded-lg">
                        <div className="font-semibold text-purple-600 dark:text-purple-400 mb-1">
                            随机搜索
                        </div>
                        <div className="text-slate-600 dark:text-slate-400">
                            更高效，推荐首选
                        </div>
                    </div>
                    <div className="p-3 bg-white dark:bg-slate-800 rounded-lg">
                        <div className="font-semibold text-green-600 dark:text-green-400 mb-1">
                            贝叶斯优化
                        </div>
                        <div className="text-slate-600 dark:text-slate-400">
                            智能采样（Optuna）
                        </div>
                    </div>
                    <div className="p-3 bg-white dark:bg-slate-800 rounded-lg">
                        <div className="font-semibold text-orange-600 dark:text-orange-400 mb-1">
                            PBT
                        </div>
                        <div className="text-slate-600 dark:text-slate-400">
                            训练中动态调整
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
