"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function AblationStudyVisualizer() {
    const [selectedComponent, setSelectedComponent] = useState<string | null>(null);

    // PPO components and their contributions
    const components = [
        { id: "clip", name: "PPO Clip", contribution: 18, basePerf: 200 },
        { id: "value_clip", name: "Value Clipping", contribution: 8, basePerf: 200 },
        { id: "gae", name: "GAE", contribution: 25, basePerf: 200 },
        { id: "entropy", name: "Entropy Bonus", contribution: 12, basePerf: 200 },
        { id: "normalize", name: "Advantage Normalization", contribution: 6, basePerf: 200 }
    ];

    const fullPerformance = 200;

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-slate-900 dark:to-purple-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Ablation Study 可视化
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Component Contribution Analysis
                </p>
            </div>

            {/* Full vs Ablated Performance */}
            <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-2 border-green-500">
                    <div className="text-sm font-semibold text-green-700 dark:text-green-400 mb-2">
                        Full Algorithm
                    </div>
                    <div className="text-4xl font-bold text-green-600 dark:text-green-400 mb-2">
                        {fullPerformance}
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                        所有组件
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-2 border-orange-500">
                    <div className="text-sm font-semibold text-orange-700 dark:text-orange-400 mb-2">
                        {selectedComponent ? `w/o ${components.find(c => c.id === selectedComponent)?.name}` : "选择组件"}
                    </div>
                    <div className="text-4xl font-bold text-orange-600 dark:text-orange-400 mb-2">
                        {selectedComponent
                            ? fullPerformance - (components.find(c => c.id === selectedComponent)?.contribution || 0)
                            : "---"}
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                        移除一个组件
                    </div>
                </div>
            </div>

            {/* Component List */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    组件贡献度
                </h4>

                <div className="space-y-3">
                    {components
                        .sort((a, b) => b.contribution - a.contribution)
                        .map((component, idx) => {
                            const isSelected = selectedComponent === component.id;
                            const percentage = (component.contribution / fullPerformance) * 100;

                            return (
                                <motion.div
                                    key={component.id}
                                    className={`p-4 rounded-lg cursor-pointer transition ${isSelected
                                            ? "bg-purple-100 dark:bg-purple-900/30 border-2 border-purple-500"
                                            : "bg-gray-50 dark:bg-slate-700 hover:bg-gray-100 dark:hover:bg-slate-600"
                                        }`}
                                    onClick={() => setSelectedComponent(component.id)}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: idx * 0.1 }}
                                >
                                    <div className="flex items-center justify-between mb-2">
                                        <div className="font-semibold text-slate-800 dark:text-slate-100">
                                            {component.name}
                                        </div>
                                        <div className="text-lg font-bold text-purple-600 dark:text-purple-400">
                                            +{component.contribution} ({percentage.toFixed(1)}%)
                                        </div>
                                    </div>

                                    <div className="relative h-4 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden">
                                        <motion.div
                                            className="absolute left-0 top-0 bottom-0 bg-gradient-to-r from-purple-500 to-pink-500"
                                            initial={{ width: 0 }}
                                            animate={{ width: `${percentage}%` }}
                                            transition={{ duration: 0.8, delay: idx * 0.1 }}
                                        />
                                    </div>
                                </motion.div>
                            );
                        })}
                </div>
            </div>

            {/* Performance Breakdown Chart */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    性能分解
                </h4>

                <div className="flex items-end gap-2 h-64">
                    {/* Full */}
                    <div className="flex-1 flex flex-col">
                        <div className="flex-1 flex flex-col justify-end">
                            <motion.div
                                className="bg-green-500 rounded-t-lg relative"
                                initial={{ height: 0 }}
                                animate={{ height: `${(fullPerformance / fullPerformance) * 100}%` }}
                                transition={{ duration: 0.8 }}
                            >
                                <div className="absolute inset-0 flex items-center justify-center text-white font-bold">
                                    {fullPerformance}
                                </div>
                            </motion.div>
                        </div>
                        <div className="text-center text-xs font-semibold mt-2 text-slate-700 dark:text-slate-300">
                            Full
                        </div>
                    </div>

                    {/* Each ablation */}
                    {components.map((component, idx) => {
                        const ablatedPerf = fullPerformance - component.contribution;
                        const height = (ablatedPerf / fullPerformance) * 100;

                        return (
                            <div key={component.id} className="flex-1 flex flex-col">
                                <div className="flex-1 flex flex-col justify-end">
                                    <motion.div
                                        className={`rounded-t-lg relative ${selectedComponent === component.id
                                                ? "bg-orange-500"
                                                : "bg-blue-400"
                                            }`}
                                        initial={{ height: 0 }}
                                        animate={{ height: `${height}%` }}
                                        transition={{ duration: 0.8, delay: idx * 0.1 }}
                                    >
                                        <div className="absolute inset-0 flex items-center justify-center text-white font-bold text-sm">
                                            {ablatedPerf}
                                        </div>
                                    </motion.div>
                                </div>
                                <div className="text-center text-xs font-semibold mt-2 text-slate-700 dark:text-slate-300">
                                    w/o<br />{component.name.split(' ')[0]}
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Insights */}
            {selectedComponent && (
                <motion.div
                    className="bg-purple-100 dark:bg-purple-900/30 p-4 rounded-lg"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <div className="font-semibold text-purple-800 dark:text-purple-300 mb-2">
                        💡 洞察
                    </div>
                    <div className="text-sm text-slate-700 dark:text-slate-300">
                        移除 <strong>{components.find(c => c.id === selectedComponent)?.name}</strong> 导致性能下降{" "}
                        <strong className="text-purple-600 dark:text-purple-400">
                            {components.find(c => c.id === selectedComponent)?.contribution}分
                        </strong>
                        {" "}({((components.find(c => c.id === selectedComponent)?.contribution || 0) / fullPerformance * 100).toFixed(1)}%)。
                        这表明该组件对算法整体性能有重要贡献。
                    </div>
                </motion.div>
            )}

            <div className="mt-6 bg-pink-100 dark:bg-pink-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                🔬 <strong>Ablation Study</strong>: 逐个移除组件以量化其贡献度，帮助理解算法设计的重要性
            </div>
        </div>
    );
}
