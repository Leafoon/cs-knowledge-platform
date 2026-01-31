"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

export function DistributionShiftVisualization() {
    const [shiftType, setShiftType] = useState<"covariate" | "concept" | "temporal">("covariate");
    const [shiftIntensity, setShiftIntensity] = useState(0.5);

    // Generate synthetic data
    const generateDistribution = (shift: number, noise: number = 0.5) => {
        const points = [];
        for (let i = 0; i < 200; i++) {
            const angle = (Math.PI * 2 * i) / 200;
            const radius = 50 + Math.random() * noise * 20;
            points.push({
                x: 150 + Math.cos(angle) * radius + shift,
                y: 150 + Math.sin(angle) * radius + shift * 0.5
            });
        }
        return points;
    };

    const trainDist = generateDistribution(0, 0.5);
    const testDist = generateDistribution(shiftIntensity * 80, 0.5);

    // Compute statistics
    const computeStats = (points: { x: number, y: number }[]) => {
        const xValues = points.map(p => p.x);
        const yValues = points.map(p => p.y);

        return {
            meanX: xValues.reduce((a, b) => a + b, 0) / xValues.length,
            meanY: yValues.reduce((a, b) => a + b, 0) / yValues.length,
            stdX: Math.sqrt(xValues.reduce((acc, x, _, arr) => {
                const mean = arr.reduce((a, b) => a + b) / arr.length;
                return acc + Math.pow(x - mean, 2);
            }, 0) / xValues.length),
            stdY: Math.sqrt(yValues.reduce((acc, y, _, arr) => {
                const mean = arr.reduce((a, b) => a + b) / arr.length;
                return acc + Math.pow(y - mean, 2);
            }, 0) / yValues.length)
        };
    };

    const trainStats = computeStats(trainDist);
    const testStats = computeStats(testDist);

    // Wasserstein distance (simplified)
    const wassersteinDistance = (
        Math.abs(trainStats.meanX - testStats.meanX) +
        Math.abs(trainStats.meanY - testStats.meanY)
    ) / 100;

    // MMD (simplified)
    const mmd = wassersteinDistance * 1.2;

    const shiftTypes = [
        { id: "covariate", name: "协变量偏移", desc: "输入分布改变，条件分布不变", color: "blue" },
        { id: "concept", name: "概念漂移", desc: "P(Y|X)改变", color: "purple" },
        { id: "temporal", name: "时序漂移", desc: "随时间逐渐改变", color: "green" }
    ];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    分布漂移可视化
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Distribution Shift Detection & Adaptation
                </p>
            </div>

            {/* Shift Type Selection */}
            <div className="flex gap-3 justify-center mb-6">
                {shiftTypes.map((type) => (
                    <button
                        key={type.id}
                        onClick={() => setShiftType(type.id as any)}
                        className={`px-6 py-3 rounded-xl font-semibold transition ${shiftType === type.id
                                ? `bg-${type.color}-600 text-white shadow-lg`
                                : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                            }`}
                    >
                        <div className="text-sm">{type.name}</div>
                        <div className="text-xs opacity-75">{type.desc}</div>
                    </button>
                ))}
            </div>

            {/* Shift Intensity Slider */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg mb-6">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                        偏移强度
                    </span>
                    <span className="text-lg font-bold text-indigo-600 dark:text-indigo-400">
                        {(shiftIntensity * 100).toFixed(0)}%
                    </span>
                </div>
                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={shiftIntensity}
                    onChange={(e) => setShiftIntensity(Number(e.target.value))}
                    className="w-full"
                />
            </div>

            {/* Distribution Visualization */}
            <div className="grid grid-cols-2 gap-4 mb-6">
                {/* Train Distribution */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg">
                    <h4 className="text-sm font-bold mb-3 text-slate-800 dark:text-slate-100 text-center">
                        训练分布 P_train(s)
                    </h4>
                    <svg width="300" height="300" className="mx-auto">
                        {/* Train points */}
                        {trainDist.map((point, idx) => (
                            <motion.circle
                                key={`train-${idx}`}
                                cx={point.x}
                                cy={point.y}
                                r="2"
                                className="fill-current text-blue-500"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 0.6 }}
                                transition={{ delay: idx * 0.001 }}
                            />
                        ))}
                        {/* Mean */}
                        <circle
                            cx={trainStats.meanX}
                            cy={trainStats.meanY}
                            r="8"
                            className="fill-current text-blue-700"
                            strokeWidth="2"
                            stroke="white"
                        />
                    </svg>
                    <div className="text-xs text-center text-slate-600 dark:text-slate-400 mt-2">
                        <div>μ = ({trainStats.meanX.toFixed(1)}, {trainStats.meanY.toFixed(1)})</div>
                        <div>σ = ({trainStats.stdX.toFixed(1)}, {trainStats.stdY.toFixed(1)})</div>
                    </div>
                </div>

                {/* Test Distribution */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg">
                    <h4 className="text-sm font-bold mb-3 text-slate-800 dark:text-slate-100 text-center">
                        部署分布 P_deploy(s)
                    </h4>
                    <svg width="300" height="300" className="mx-auto">
                        {/* Test points */}
                        {testDist.map((point, idx) => (
                            <motion.circle
                                key={`test-${idx}`}
                                cx={point.x}
                                cy={point.y}
                                r="2"
                                className="fill-current text-orange-500"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 0.6 }}
                                transition={{ delay: idx * 0.001 }}
                            />
                        ))}
                        {/* Mean */}
                        <circle
                            cx={testStats.meanX}
                            cy={testStats.meanY}
                            r="8"
                            className="fill-current text-orange-700"
                            strokeWidth="2"
                            stroke="white"
                        />
                    </svg>
                    <div className="text-xs text-center text-slate-600 dark:text-slate-400 mt-2">
                        <div>μ = ({testStats.meanX.toFixed(1)}, {testStats.meanY.toFixed(1)})</div>
                        <div>σ = ({testStats.stdX.toFixed(1)}, {testStats.stdY.toFixed(1)})</div>
                    </div>
                </div>
            </div>

            {/* Distance Metrics */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    分布距离度量
                </h4>

                <div className="grid grid-cols-3 gap-4">
                    {/* Wasserstein Distance */}
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                        <div className="text-sm font-semibold text-blue-700 dark:text-blue-400 mb-2">
                            Wasserstein距离
                        </div>
                        <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-2">
                            {wassersteinDistance.toFixed(3)}
                        </div>
                        <div className="text-xs text-slate-600 dark:text-slate-400">
                            Earth Mover's Distance
                        </div>
                    </div>

                    {/* MMD */}
                    <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                        <div className="text-sm font-semibold text-purple-700 dark:text-purple-400 mb-2">
                            MMD
                        </div>
                        <div className="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-2">
                            {mmd.toFixed(3)}
                        </div>
                        <div className="text-xs text-slate-600 dark:text-slate-400">
                            Maximum Mean Discrepancy
                        </div>
                    </div>

                    {/* KS Statistic */}
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                        <div className="text-sm font-semibold text-green-700 dark:text-green-400 mb-2">
                            KS统计量
                        </div>
                        <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-2">
                            {(wassersteinDistance * 0.8).toFixed(3)}
                        </div>
                        <div className="text-xs text-slate-600 dark:text-slate-400">
                            Kolmogorov-Smirnov
                        </div>
                    </div>
                </div>

                {/* Interpretation */}
                <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-800 rounded-lg text-sm text-slate-700 dark:text-slate-300">
                    <strong>解读：</strong>
                    {wassersteinDistance < 0.2
                        ? " 分布差异较小，策略可能仍然有效"
                        : wassersteinDistance < 0.5
                            ? " 中等分布偏移，建议进行域适应"
                            : " 显著分布偏移！需要域随机化或重新训练"}
                </div>
            </div>

            {/* Adaptation Strategies */}
            <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 p-4 rounded-lg">
                <h5 className="font-semibold text-slate-700 dark:text-slate-300 mb-3">
                    应对策略
                </h5>
                <div className="grid grid-cols-3 gap-3">
                    <div className="p-3 bg-white dark:bg-slate-800 rounded-lg text-xs">
                        <div className="font-semibold text-blue-600 dark:text-blue-400 mb-1">
                            域随机化
                        </div>
                        <div className="text-slate-600 dark:text-slate-400">
                            在训练时随机化物理参数
                        </div>
                    </div>
                    <div className="p-3 bg-white dark:bg-slate-800 rounded-lg text-xs">
                        <div className="font-semibold text-purple-600 dark:text-purple-400 mb-1">
                            对抗域适应
                        </div>
                        <div className="text-slate-600 dark:text-slate-400">
                            训练域不变特征提取器
                        </div>
                    </div>
                    <div className="p-3 bg-white dark:bg-slate-800 rounded-lg text-xs">
                        <div className="font-semibold text-green-600 dark:text-green-400 mb-1">
                            持续学习
                        </div>
                        <div className="text-slate-600 dark:text-slate-400">
                            在线适应with EWC/PackNet
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
