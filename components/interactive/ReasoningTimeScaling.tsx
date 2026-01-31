"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function ReasoningTimeScaling() {
    const [computeBudget, setComputeBudget] = useState(10);

    // 模拟不同任务的scaling曲线
    const tasks = [
        { name: "数学竞赛(AIME)", color: "blue", alpha: 0.45, baseline: 0.13 },
        { name: "代码生成(HumanEval)", color: "green", alpha: 0.35, baseline: 0.52 },
        { name: "科学推理(GPQA)", color: "purple", alpha: 0.40, baseline: 0.38 },
    ];

    // 计算准确率: Accuracy = baseline + scaling_factor * Compute^alpha
    const getAccuracy = (task: typeof tasks[0], compute: number) => {
        const scalingFactor = (1 - task.baseline) / Math.pow(100, task.alpha);
        return Math.min(0.95, task.baseline + scalingFactor * Math.pow(compute, task.alpha));
    };

    const currentAccuracies = tasks.map(task => ({
        ...task,
        accuracy: getAccuracy(task, computeBudget)
    }));

    // 生成曲线点
    const curvePoints = Array.from({ length: 100 }, (_, i) => {
        const compute = i + 1;
        return {
            compute,
            ...Object.fromEntries(
                tasks.map(task => [task.name, getAccuracy(task, compute)])
            )
        };
    });

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-slate-900 dark:to-blue-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    推理时计算扩展曲线
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Test-Time Compute Scaling: 计算量与性能的幂律关系
                </p>
            </div>

            {/* 公式 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">Scaling Law</h4>
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg font-mono text-center mb-4">
                    <div className="text-lg mb-2">
                        Accuracy = Baseline + k × (Compute)<sup>α</sup>
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                        α ≈ 0.3-0.5 (任务相关的scaling指数)
                    </div>
                </div>

                <div className="grid grid-cols-3 gap-4 text-sm">
                    {tasks.map((task) => (
                        <div key={task.name} className={`bg-${task.color}-50 dark:bg-${task.color}-900/20 p-3 rounded-lg border border-${task.color}-300 dark:border-${task.color}-700`}>
                            <div className={`font-semibold text-${task.color}-700 dark:text-${task.color}-400 mb-1`}>
                                {task.name}
                            </div>
                            <div className="text-xs text-slate-600 dark:text-slate-400">
                                Baseline: {(task.baseline * 100).toFixed(0)}%
                            </div>
                            <div className="text-xs text-slate-600 dark:text-slate-400">
                                α = {task.alpha.toFixed(2)}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* 计算预算滑块 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">调整计算预算</h4>

                <div>
                    <div className="flex justify-between mb-2">
                        <span className="font-semibold text-cyan-600 dark:text-cyan-400">
                            计算量（相对于base）
                        </span>
                        <span className="font-mono text-cyan-600 dark:text-cyan-400">{computeBudget}x</span>
                    </div>
                    <input
                        type="range"
                        min="1"
                        max="100"
                        step="1"
                        value={computeBudget}
                        onChange={(e) => setComputeBudget(parseInt(e.target.value))}
                        className="w-full h-3 bg-cyan-200 rounded-lg appearance-none cursor-pointer dark:bg-cyan-900"
                    />
                    <div className="flex justify-between text-xs text-slate-500 dark:text-slate-500 mt-1">
                        <span>1x (立即)</span>
                        <span>50x (~30秒)</span>
                        <span>100x (~1分钟)</span>
                    </div>
                </div>
            </div>

            {/* 当前性能 */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                {currentAccuracies.map((task) => (
                    <motion.div
                        key={task.name}
                        animate={{ scale: [1, 1.02, 1] }}
                        transition={{ duration: 0.3 }}
                        className={`bg-${task.color}-50 dark:bg-${task.color}-900/20 p-4 rounded-lg border-2 border-${task.color}-500`}
                    >
                        <div className={`text-sm font-semibold text-${task.color}-700 dark:text-${task.color}-400 mb-2`}>
                            {task.name}
                        </div>
                        <div className={`text-4xl font-bold text-${task.color}-600 dark:text-${task.color}-400 mb-2`}>
                            {(task.accuracy * 100).toFixed(1)}%
                        </div>
                        <div className={`h-3 bg-${task.color}-200 dark:bg-${task.color}-900 rounded-full overflow-hidden`}>
                            <motion.div
                                className={`h-full bg-${task.color}-600`}
                                animate={{ width: `${task.accuracy * 100}%` }}
                                transition={{ duration: 0.5 }}
                            />
                        </div>
                        <div className="text-xs text-slate-600 dark:text-slate-400 mt-2">
                            提升: +{((task.accuracy - task.baseline) * 100).toFixed(1)}%
                        </div>
                    </motion.div>
                ))}
            </div>

            {/* Scaling曲线图 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">性能曲线</h4>

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
                        {tasks.map((task) => (
                            <path
                                key={task.name}
                                d={curvePoints.map((p, i) => {
                                    const x = (p.compute / 100) * 100;
                                    const y = 100 - (p[task.name] as number * 100);
                                    return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                                }).join(' ')}
                                stroke={
                                    task.color === 'blue' ? '#3b82f6' :
                                        task.color === 'green' ? '#22c55e' : '#a855f7'
                                }
                                strokeWidth="2"
                                fill="none"
                            />
                        ))}

                        {/* 当前位置标记 */}
                        {currentAccuracies.map((task) => (
                            <circle
                                key={task.name}
                                cx={computeBudget}
                                cy={100 - task.accuracy * 100}
                                r="2"
                                fill={
                                    task.color === 'blue' ? '#3b82f6' :
                                        task.color === 'green' ? '#22c55e' : '#a855f7'
                                }
                                stroke="white"
                                strokeWidth="1"
                            />
                        ))}
                    </svg>

                    {/* 轴标签 */}
                    <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 text-xs text-slate-600 dark:text-slate-400">
                        计算量 (倍数) →
                    </div>
                    <div className="absolute top-1/2 left-2 transform -translate-y-1/2 -rotate-90 text-xs text-slate-600 dark:text-slate-400">
                        ← 准确率 (%)
                    </div>
                </div>

                <div className="flex justify-center gap-6 mt-4 text-sm">
                    {tasks.map((task) => (
                        <div key={task.name} className="flex items-center gap-2">
                            <div className={`w-4 h-1 bg-${task.color}-600 rounded`}></div>
                            <span className="text-slate-600 dark:text-slate-400">{task.name}</span>
                        </div>
                    ))}
                </div>
            </div>

            <div className="mt-6 bg-cyan-100 dark:bg-cyan-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                💡 <strong>OpenAI o1启示</strong>: 推理时投入更多计算可显著提升性能，遵循幂律关系
            </div>
        </div>
    );
}
