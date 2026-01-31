"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

interface TrainingSession {
    id: number;
    name: string;
    gpuPower: number;
    numGPUs: number;
    hours: number;
    pue: number;
    carbonIntensity: number;
}

export function CarbonFootprintTracker() {
    const [selectedSession, setSelectedSession] = useState(0);
    const [customHours, setCustomHours] = useState(72);

    const sessions: TrainingSession[] = [
        {
            id: 1,
            name: "LLaMA-7B (LoRA)",
            gpuPower: 300,
            numGPUs: 8,
            hours: 24,
            pue: 1.2,
            carbonIntensity: 0.5
        },
        {
            id: 2,
            name: "LLaMA-7B (Full FT)",
            gpuPower: 300,
            numGPUs: 32,
            hours: 72,
            pue: 1.2,
            carbonIntensity: 0.5
        },
        {
            id: 3,
            name: "GPT-3 (估算)",
            gpuPower: 300,
            numGPUs: 10000,
            hours: 336,
            pue: 1.1,
            carbonIntensity: 0.385
        },
        {
            id: 4,
            name: "自定义配置",
            gpuPower: 300,
            numGPUs: 8,
            hours: customHours,
            pue: 1.2,
            carbonIntensity: 0.5
        }
    ];

    const current = sessions[selectedSession];

    // 计算碳排放
    const calculateEmissions = (session: TrainingSession) => {
        const gpuEnergyKwh = (session.gpuPower * session.numGPUs * session.hours) / 1000;
        const totalEnergyKwh = gpuEnergyKwh * session.pue;
        const carbonKg = totalEnergyKwh * session.carbonIntensity;
        const carMiles = carbonKg / 0.41;
        const treesNeeded = carbonKg / 21.77; // 一棵树一年约吸收21.77kg CO2

        return {
            energyKwh: totalEnergyKwh,
            carbonKg,
            carMiles,
            treesNeeded
        };
    };

    const emissions = calculateEmissions(current);

    // 数据中心对比
    const dataCenters = [
        { name: "Quebec (水电)", intensity: 0.002, color: "green" },
        { name: "Iceland", intensity: 0.015, color: "green" },
        { name: "US-Iowa (Google)", intensity: 0.220, color: "blue" },
        { name: "EU平均", intensity: 0.276, color: "yellow" },
        { name: "中国平均", intensity: 0.681, color: "red" }
    ];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-green-50 to-teal-50 dark:from-slate-900 dark:to-green-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    碳足迹追踪器
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    计算AI训练的环境影响
                </p>
            </div>

            {/* 会话选择 */}
            <div className="grid grid-cols-4 gap-3 mb-6">
                {sessions.map((session, idx) => (
                    <button
                        key={session.id}
                        onClick={() => setSelectedSession(idx)}
                        className={`p-3 rounded-xl border-2 transition ${selectedSession === idx
                                ? "border-green-500 bg-green-50 dark:bg-green-900/20"
                                : "border-gray-200 dark:border-gray-700 bg-white dark:bg-slate-800"
                            }`}
                    >
                        <div className={`text-sm font-bold ${selectedSession === idx
                                ? "text-green-600 dark:text-green-400"
                                : "text-slate-700 dark:text-slate-300"
                            }`}>
                            {session.name}
                        </div>
                    </button>
                ))}
            </div>

            {/* 自定义配置 */}
            {selectedSession === 3 && (
                <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg mb-6">
                    <h4 className="text-sm font-bold mb-3 text-slate-800 dark:text-slate-100">
                        自定义训练时长 </h4>
                    <div className="flex items-center gap-4">
                        <input
                            type="range"
                            min="1"
                            max="500"
                            value={customHours}
                            onChange={(e) => setCustomHours(Number(e.target.value))}
                            className="flex-1"
                        />
                        <div className="text-lg font-bold text-green-600 dark:text-green-400 w-24 text-right">
                            {customHours}小时
                        </div>
                    </div>
                </div>
            )}

            {/* 配置信息 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    训练配置
                </h4>

                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex justify-between">
                        <span className="text-slate-600 dark:text-slate-400">GPU功耗:</span>
                        <span className="font-semibold text-slate-800 dark:text-slate-100">
                            {current.gpuPower}W
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-slate-600 dark:text-slate-400">GPU数量:</span>
                        <span className="font-semibold text-slate-800 dark:text-slate-100">
                            {current.numGPUs}
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-slate-600 dark:text-slate-400">训练时长:</span>
                        <span className="font-semibold text-slate-800 dark:text-slate-100">
                            {current.hours}小时
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-slate-600 dark:text-slate-400">PUE:</span>
                        <span className="font-semibold text-slate-800 dark:text-slate-100">
                            {current.pue}
                        </span>
                    </div>
                    <div className="flex justify-between col-span-2">
                        <span className="text-slate-600 dark:text-slate-400">碳强度:</span>
                        <span className="font-semibold text-slate-800 dark:text-slate-100">
                            {current.carbonIntensity} kg CO₂/kWh
                        </span>
                    </div>
                </div>
            </div>

            {/* 排放结果 */}
            <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 p-6 rounded-xl shadow-lg border-2 border-blue-300 dark:border-blue-700">
                    <div className="text-sm text-blue-700 dark:text-blue-400 mb-1">总能耗</div>
                    <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.5 }}
                        className="text-4xl font-bold text-blue-600 dark:text-blue-400 mb-2"
                    >
                        {emissions.energyKwh.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </motion.div>
                    <div className="text-sm text-blue-700 dark:text-blue-400">kWh</div>
                </div>

                <div className="bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 p-6 rounded-xl shadow-lg border-2 border-red-300 dark:border-red-700">
                    <div className="text-sm text-red-700 dark:text-red-400 mb-1">碳排放</div>
                    <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.5, delay: 0.1 }}
                        className="text-4xl font-bold text-red-600 dark:text-red-400 mb-2"
                    >
                        {emissions.carbonKg.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </motion.div>
                    <div className="text-sm text-red-700 dark:text-red-400">kg CO₂</div>
                </div>
            </div>

            {/* 可视化对比 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    等效对比
                </h4>

                <div className="space-y-4">
                    {/* 汽车里程 */}
                    <div>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm text-slate-600 dark:text-slate-400">🚗 汽车行驶里程</span>
                            <span className="text-lg font-bold text-orange-600 dark:text-orange-400">
                                {emissions.carMiles.toLocaleString('en-US', { maximumFractionDigits: 0 })} 英里
                            </span>
                        </div>
                        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${Math.min((emissions.carMiles / 100000) * 100, 100)}%` }}
                                transition={{ duration: 1.5 }}
                                className="h-full bg-orange-600"
                            />
                        </div>
                    </div>

                    {/* 树木抵消 */}
                    <div>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm text-slate-600 dark:text-slate-400">🌳 需要树木吸收(一年)</span>
                            <span className="text-lg font-bold text-green-600 dark:text-green-400">
                                {emissions.treesNeeded.toLocaleString('en-US', { maximumFractionDigits: 0 })} 棵
                            </span>
                        </div>
                        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${Math.min((emissions.treesNeeded / 10000) * 100, 100)}%` }}
                                transition={{ duration: 1.5, delay: 0.2 }}
                                className="h-full bg-green-600"
                            />
                        </div>
                    </div>
                </div>
            </div>

            {/* 数据中心对比 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    不同数据中心的碳排放
                </h4>

                <div className="space-y-3">
                    {dataCenters.map((dc, idx) => {
                        const dcEmissions = calculateEmissions({
                            ...current,
                            carbonIntensity: dc.intensity
                        });

                        return (
                            <div key={idx}>
                                <div className="flex items-center justify-between mb-1">
                                    <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                                        {dc.name}
                                    </span>
                                    <span className="text-sm font-bold text-slate-600 dark:text-slate-400">
                                        {dcEmissions.carbonKg.toLocaleString('en-US', { maximumFractionDigits: 0 })} kg CO₂
                                    </span>
                                </div>
                                <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                    <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: `${(dcEmissions.carbonKg / emissions.carbonKg) * (selectedSession === 2 ? 10 : 100)}%` }}
                                        transition={{ duration: 1, delay: idx * 0.1 }}
                                        className={`h-full bg-${dc.color}-600`}
                                    />
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            <div className="mt-6 grid grid-cols-3 gap-4 text-xs">
                <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded-lg border border-green-300 dark:border-green-700">
                    <div className="font-semibold text-green-700 dark:text-green-400 mb-1">💡 最佳实践</div>
                    <div className="text-slate-600 dark:text-slate-400">
                        选择绿色能源数据中心
                    </div>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg border border-blue-300 dark:border-blue-700">
                    <div className="font-semibold text-blue-700 dark:text-blue-400 mb-1">⏰ 碳感知调度</div>
                    <div className="text-slate-600 dark:text-slate-400">
                        在低碳时段训练
                    </div>
                </div>

                <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-lg border border-purple-300 dark:border-purple-700">
                    <div className="font-semibold text-purple-700 dark:text-purple-400 mb-1">🎯 高效方法</div>
                    <div className="text-slate-600 dark:text-slate-400">
                        使用LoRA/QLoRA减少训练时间
                    </div>
                </div>
            </div>
        </div>
    );
}
