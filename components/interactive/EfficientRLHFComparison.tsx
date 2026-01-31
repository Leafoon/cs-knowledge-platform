"use client";

import { useState } from "react";
import { motion } from "framer-motion";

interface Method {
    name: string;
    params: string;
    memory: string;
    speed: string;
    performance: string;
    color: string;
}

export function EfficientRLHFComparison() {
    const [selectedMethod, setSelectedMethod] = useState<string>("LoRA");

    const methods: Method[] = [
        {
            name: "Full Fine-tuning",
            params: "100%",
            memory: "780 GB",
            speed: "1x",
            performance: "100%",
            color: "gray"
        },
        {
            name: "LoRA",
            params: "0.8%",
            memory: "17 GB",
            speed: "1.8x",
            performance: "97%",
            color: "blue"
        },
        {
            name: "QLoRA",
            params: "0.8%",
            memory: "48 GB",
            speed: "1.6x",
            performance: "95%",
            color: "purple"
        },
        {
            name: "Prefix Tuning",
            params: "0.1%",
            memory: "12 GB",
            speed: "2.5x",
            performance: "92%",
            color: "green"
        },
        {
            name: "Adapter",
            params: "5%",
            memory: "85 GB",
            speed: "1.4x",
            performance: "96%",
            color: "orange"
        }
    ];

    const currentMethod = methods.find(m => m.name === selectedMethod) || methods[1];

    // LoRA机制可视化数据
    const loraVisualization = {
        originalRank: 4096,
        loraRank: 16,
        paramReduction: 99.2
    };

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-slate-900 dark:to-blue-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    高效RLHF方法对比
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Parameter-Efficient Fine-Tuning (PEFT)
                </p>
            </div>

            {/* 方法选择 */}
            <div className="grid grid-cols-5 gap-3 mb-6">
                {methods.map((method) => (
                    <button
                        key={method.name}
                        onClick={() => setSelectedMethod(method.name)}
                        className={`p-3 rounded-xl border-2 transition ${selectedMethod === method.name
                                ? `border-${method.color}-500 bg-${method.color}-50 dark:bg-${method.color}-900/20`
                                : "border-gray-200 dark:border-gray-700 bg-white dark:bg-slate-800"
                            }`}
                    >
                        <div className={`text-sm font-bold ${selectedMethod === method.name
                                ? `text-${method.color}-600 dark:text-${method.color}-400`
                                : "text-slate-700 dark:text-slate-300"
                            }`}>
                            {method.name}
                        </div>
                    </button>
                ))}
            </div>

            {/* 当前方法详情 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    {currentMethod.name} 性能指标
                </h4>

                <div className="grid grid-cols-2 gap-6">
                    {/* 参数量 */}
                    <div>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm text-slate-600 dark:text-slate-400">可训练参数</span>
                            <span className="text-lg font-bold text-blue-600 dark:text-blue-400">
                                {currentMethod.params}
                            </span>
                        </div>
                        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: currentMethod.params }}
                                transition={{ duration: 1 }}
                                className="h-full bg-blue-600"
                            />
                        </div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                            vs 100% (Full Fine-tuning)
                        </div>
                    </div>

                    {/* 显存 */}
                    <div>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm text-slate-600 dark:text-slate-400">显存占用</span>
                            <span className="text-lg font-bold text-purple-600 dark:text-purple-400">
                                {currentMethod.memory}
                            </span>
                        </div>
                        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${(parseInt(currentMethod.memory) / 780) * 100}%` }}
                                transition={{ duration: 1 }}
                                className="h-full bg-purple-600"
                            />
                        </div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                            vs 780 GB (Full Fine-tuning)
                        </div>
                    </div>

                    {/* 速度 */}
                    <div>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm text-slate-600 dark:text-slate-400">训练速度</span>
                            <span className="text-lg font-bold text-green-600 dark:text-green-400">
                                {currentMethod.speed}
                            </span>
                        </div>
                        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${parseFloat(currentMethod.speed) * 40}%` }}
                                transition={{ duration: 1 }}
                                className="h-full bg-green-600"
                            />
                        </div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                            相对加速倍数
                        </div>
                    </div>

                    {/* 性能 */}
                    <div>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm text-slate-600 dark:text-slate-400">模型性能</span>
                            <span className="text-lg font-bold text-orange-600 dark:text-orange-400">
                                {currentMethod.performance}
                            </span>
                        </div>
                        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: currentMethod.performance }}
                                transition={{ duration: 1 }}
                                className="h-full bg-orange-600"
                            />
                        </div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                            vs Full Fine-tuning
                        </div>
                    </div>
                </div>
            </div>

            {/* LoRA机制可视化 */}
            {(selectedMethod === "LoRA" || selectedMethod === "QLoRA") && (
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                    <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                        {selectedMethod} 分解机制
                    </h4>

                    <div className="flex items-center justify-center gap-6">
                        {/* 原始矩阵 */}
                        <div className="text-center">
                            <div className="w-32 h-32 bg-gray-300 dark:bg-gray-700 rounded-lg flex items-center justify-center mb-2">
                                <div className="text-center">
                                    <div className="text-xs text-slate-600 dark:text-slate-400">W</div>
                                    <div className="text-2xl font-bold text-slate-800 dark:text-slate-100">
                                        {loraVisualization.originalRank}×{loraVisualization.originalRank}
                                    </div>
                                </div>
                            </div>
                            <div className="text-xs text-slate-600 dark:text-slate-400">
                                原始权重矩阵<br />(冻结)
                            </div>
                        </div>

                        <div className="text-3xl text-blue-600 dark:text-blue-400">=</div>

                        {/* 固定W */}
                        <div className="text-center">
                            <div className="w-32 h-32 bg-blue-100 dark:bg-blue-900/20 border-2 border-blue-500 rounded-lg flex items-center justify-center mb-2">
                                <div className="text-center">
                                    <div className="text-xs text-blue-600 dark:text-blue-400">W₀</div>
                                    <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                                        {loraVisualization.originalRank}×{loraVisualization.originalRank}
                                    </div>
                                </div>
                            </div>
                            <div className="text-xs text-slate-600 dark:text-slate-400">
                                基础权重<br />(冻结)
                            </div>
                        </div>

                        <div className="text-3xl text-blue-600 dark:text-blue-400">+</div>

                        {/* B×A */}
                        <div className="text-center">
                            <div className="flex items-center gap-2 mb-2">
                                <div className="w-14 h-32 bg-green-100 dark:bg-green-900/20 border-2 border-green-500 rounded-lg flex items-center justify-center">
                                    <div className="text-center">
                                        <div className="text-xs text-green-600 dark:text-green-400">B</div>
                                        <div className="text-sm font-bold text-green-600 dark:text-green-400 transform -rotate-90">
                                            {loraVisualization.originalRank}×{loraVisualization.loraRank}
                                        </div>
                                    </div>
                                </div>
                                <div className="text-xl text-green-600 dark:text-green-400">×</div>
                                <div className="w-14 h-14 bg-green-100 dark:bg-green-900/20 border-2 border-green-500 rounded-lg flex items-center justify-center">
                                    <div className="text-center">
                                        <div className="text-xs text-green-600 dark:text-green-400">A</div>
                                        <div className="text-xs font-bold text-green-600 dark:text-green-400">
                                            {loraVisualization.loraRank}×{loraVisualization.originalRank}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div className="text-xs text-slate-600 dark:text-slate-400">
                                低秩分解<br />(可训练)
                            </div>
                        </div>
                    </div>

                    <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg text-center">
                        <div className="text-sm text-slate-700 dark:text-slate-300">
                            参数减少: <strong className="text-blue-600 dark:text-blue-400">
                                {loraVisualization.paramReduction}%
                            </strong>
                        </div>
                        <div className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                            从 {loraVisualization.originalRank * loraVisualization.originalRank / 1e6}M
                            到 {(2 * loraVisualization.originalRank * loraVisualization.loraRank) / 1e6}M 参数
                        </div>
                    </div>
                </div>
            )}

            {/* 方法对比表 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    所有方法对比
                </h4>

                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b-2 border-gray-300 dark:border-gray-700">
                                <th className="text-left py-2 px-4 text-slate-700 dark:text-slate-300">方法</th>
                                <th className="text-center py-2 px-4 text-slate-700 dark:text-slate-300">可训练参数</th>
                                <th className="text-center py-2 px-4 text-slate-700 dark:text-slate-300">显存</th>
                                <th className="text-center py-2 px-4 text-slate-700 dark:text-slate-300">速度</th>
                                <th className="text-center py-2 px-4 text-slate-700 dark:text-slate-300">性能</th>
                            </tr>
                        </thead>
                        <tbody>
                            {methods.map((method, idx) => (
                                <tr
                                    key={idx}
                                    className={`border-b border-gray-200 dark:border-gray-700 ${method.name === selectedMethod
                                            ? "bg-blue-50 dark:bg-blue-900/20"
                                            : ""
                                        }`}
                                >
                                    <td className="py-3 px-4 font-semibold text-slate-800 dark:text-slate-100">
                                        {method.name}
                                    </td>
                                    <td className="text-center py-3 px-4">{method.params}</td>
                                    <td className="text-center py-3 px-4">{method.memory}</td>
                                    <td className="text-center py-3 px-4">{method.speed}</td>
                                    <td className="text-center py-3 px-4">{method.performance}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            <div className="mt-6 bg-blue-100 dark:bg-blue-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                💡 <strong>推荐</strong>: LoRA/QLoRA在保持97%+性能的同时，显著降低显存和参数量
            </div>
        </div>
    );
}
