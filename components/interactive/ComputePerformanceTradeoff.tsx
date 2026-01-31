"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function ComputePerformanceTradeoff() {
    const [selectedTask, setSelectedTask] = useState<"math" | "code" | "reasoning">("math");

    const tasks = {
        math: {
            name: "数学推理 (AIME)",
            color: "blue",
            data: [
                { compute: 1, accuracy: 13, cost: 1, time: "1秒" },
                { compute: 5, accuracy: 32, cost: 5, time: "5秒" },
                { compute: 10, accuracy: 45, cost: 10, time: "10秒" },
                { compute: 30, accuracy: 64, cost: 30, time: "30秒" },
                { compute: 60, accuracy: 74, cost: 60, time: "1分钟" },
                { compute: 100, accuracy: 80, cost: 100, time: "1.7分钟" }
            ]
        },
        code: {
            name: "代码生成 (HumanEval)",
            color: "green",
            data: [
                { compute: 1, accuracy: 52, cost: 1, time: "1秒" },
                { compute: 5, accuracy: 65, cost: 5, time: "5秒" },
                { compute: 10, accuracy: 72, cost: 10, time: "10秒" },
                { compute: 30, accuracy: 82, cost: 30, time: "30秒" },
                { compute: 60, accuracy: 88, cost: 60, time: "1分钟" },
                { compute: 100, accuracy: 91, cost: 100, time: "1.7分钟" }
            ]
        },
        reasoning: {
            name: "科学推理 (GPQA)",
            color: "purple",
            data: [
                { compute: 1, accuracy: 38, cost: 1, time: "1秒" },
                { compute: 5, accuracy: 51, cost: 5, time: "5秒" },
                { compute: 10, accuracy: 61, cost: 10, time: "10秒" },
                { compute: 30, accuracy: 74, cost: 30, time: "30秒" },
                { compute: 60, accuracy: 81, cost: 60, time: "1分钟" },
                { compute: 100, accuracy: 85, cost: 100, time: "1.7分钟" }
            ]
        }
    };

    const currentTask = tasks[selectedTask];
    const [computeLevel, setComputeLevel] = useState(2);
    const currentPoint = currentTask.data[computeLevel];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-amber-50 to-yellow-50 dark:from-slate-900 dark:to-amber-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    计算-性能权衡
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Test-Time Compute vs Accuracy: 值得投入更多计算吗？
                </p>
            </div>

            {/* 任务选择 */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                {(["math", "code", "reasoning"] as const).map((taskKey) => {
                    const task = tasks[taskKey];
                    return (
                        <button
                            key={taskKey}
                            onClick={() => setSelectedTask(taskKey)}
                            className={`p-4 rounded-xl border-2 transition ${selectedTask === taskKey
                                    ? `border-${task.color}-500 bg-${task.color}-50 dark:bg-${task.color}-900/20`
                                    : "border-gray-200 dark:border-gray-700 bg-white dark:bg-slate-800"
                                }`}
                        >
                            <div className={`text-lg font-bold text-${task.color}-600 dark:text-${task.color}-400`}>
                                {task.name}
                            </div>
                        </button>
                    );
                })}
            </div>

            {/* 计算预算滑块 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    调整计算预算
                </h4>

                <div>
                    <div className="flex justify-between mb-2">
                        <span className="font-semibold text-amber-600 dark:text-amber-400">
                            计算量
                        </span>
                        <span className="font-mono text-amber-600 dark:text-amber-400">
                            {currentPoint.compute}x ({currentPoint.time})
                        </span>
                    </div>
                    <input
                        type="range"
                        min="0"
                        max={currentTask.data.length - 1}
                        step="1"
                        value={computeLevel}
                        onChange={(e) => setComputeLevel(parseInt(e.target.value))}
                        className="w-full h-3 bg-amber-200 rounded-lg appearance-none cursor-pointer dark:bg-amber-900"
                    />
                    <div className="flex justify-between text-xs text-slate-500 dark:text-slate-500 mt-1">
                        {currentTask.data.map((point, idx) => (
                            <span key={idx}>{point.time}</span>
                        ))}
                    </div>
                </div>
            </div>

            {/* 当前状态 */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                <motion.div
                    animate={{ scale: [1, 1.02, 1] }}
                    transition={{ duration: 0.3 }}
                    className={`bg-${currentTask.color}-50 dark:bg-${currentTask.color}-900/20 p-6 rounded-xl border-2 border-${currentTask.color}-500`}
                >
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-2">准确率</div>
                    <div className={`text-5xl font-bold text-${currentTask.color}-600 dark:text-${currentTask.color}-400 mb-3`}>
                        {currentPoint.accuracy}%
                    </div>
                    <div className={`h-3 bg-${currentTask.color}-200 dark:bg-${currentTask.color}-900 rounded-full overflow-hidden`}>
                        <motion.div
                            className={`h-full bg-${currentTask.color}-600`}
                            animate={{ width: `${currentPoint.accuracy}%` }}
                            transition={{ duration: 0.5 }}
                        />
                    </div>
                </motion.div>

                <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl border-2 border-orange-500">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-2">成本</div>
                    <div className="text-5xl font-bold text-orange-600 dark:text-orange-400 mb-3">
                        {currentPoint.cost}x
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                        相对于baseline
                    </div>
                </div>

                <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border-2 border-green-500">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-2">ROI</div>
                    <div className="text-5xl font-bold text-green-600 dark:text-green-400 mb-3">
                        {(currentPoint.accuracy / currentPoint.cost).toFixed(1)}
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                        准确率/成本
                    </div>
                </div>
            </div>

            {/* 曲线图 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    准确率 vs 计算量
                </h4>

                <div className="h-64 bg-gray-50 dark:bg-gray-900 rounded-lg p-4 relative">
                    <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                        {/* 网格 */}
                        {[0, 25, 50, 75, 100].map(y => (
                            <line
                                key={y}
                                x1="0" y1={y} x2="100" y2={y}
                                stroke="#e5e7eb"
                                strokeWidth="0.5"
                            />
                        ))}

                        {/* 曲线 */}
                        <path
                            d={currentTask.data.map((point, i) => {
                                const x = (i / (currentTask.data.length - 1)) * 100;
                                const y = 100 - point.accuracy;
                                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                            }).join(' ')}
                            stroke={
                                currentTask.color === 'blue' ? '#3b82f6' :
                                    currentTask.color === 'green' ? '#22c55e' : '#a855f7'
                            }
                            strokeWidth="3"
                            fill="none"
                        />

                        {/* 当前点 */}
                        <circle
                            cx={(computeLevel / (currentTask.data.length - 1)) * 100}
                            cy={100 - currentPoint.accuracy}
                            r="4"
                            fill={
                                currentTask.color === 'blue' ? '#3b82f6' :
                                    currentTask.color === 'green' ? '#22c55e' : '#a855f7'
                            }
                            stroke="white"
                            strokeWidth="2"
                        />

                        {/* 所有点 */}
                        {currentTask.data.map((point, i) => (
                            <circle
                                key={i}
                                cx={(i / (currentTask.data.length - 1)) * 100}
                                cy={100 - point.accuracy}
                                r="2"
                                fill={
                                    currentTask.color === 'blue' ? '#3b82f6' :
                                        currentTask.color === 'green' ? '#22c55e' : '#a855f7'
                                }
                            />
                        ))}
                    </svg>

                    {/* 轴标签 */}
                    <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 text-xs text-slate-600 dark:text-slate-400">
                        计算量 →
                    </div>
                    <div className="absolute top-1/2 left-2 transform -translate-y-1/2 -rotate-90 text-xs text-slate-600 dark:text-slate-400">
                        ← 准确率 (%)
                    </div>
                </div>
            </div>

            {/* 数据表格 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">详细数据</h4>

                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b-2 border-gray-300 dark:border-gray-700">
                                <th className="text-left py-3 px-4 text-slate-600 dark:text-slate-400">计算量</th>
                                <th className="text-center py-3 px-4 text-slate-600 dark:text-slate-400">时间</th>
                                <th className="text-center py-3 px-4 text-slate-600 dark:text-slate-400">准确率</th>
                                <th className="text-center py-3 px-4 text-slate-600 dark:text-slate-400">提升</th>
                                <th className="text-center py-3 px-4 text-slate-600 dark:text-slate-400">ROI</th>
                            </tr>
                        </thead>
                        <tbody className="text-slate-700 dark:text-slate-300">
                            {currentTask.data.map((point, idx) => {
                                const improvement = idx > 0 ? point.accuracy - currentTask.data[0].accuracy : 0;
                                const roi = point.accuracy / point.cost;
                                const isCurrent = idx === computeLevel;

                                return (
                                    <tr
                                        key={idx}
                                        className={`border-b border-gray-200 dark:border-gray-700 ${isCurrent ? `bg-${currentTask.color}-50 dark:bg-${currentTask.color}-900/20` : ""
                                            }`}
                                    >
                                        <td className="py-3 px-4 font-mono">{point.compute}x</td>
                                        <td className="py-3 px-4 text-center">{point.time}</td>
                                        <td className="py-3 px-4 text-center font-semibold">{point.accuracy}%</td>
                                        <td className="py-3 px-4 text-center text-green-600 dark:text-green-400">
                                            {idx > 0 ? `+${improvement}%` : "-"}
                                        </td>
                                        <td className="py-3 px-4 text-center">{roi.toFixed(2)}</td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>

            <div className="mt-6 bg-amber-100 dark:bg-amber-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                💡 <strong>权衡建议</strong>: 10-30x计算量通常是ROI最优点，继续增加收益递减
            </div>
        </div>
    );
}
