"use client";

import { useState } from "react";
import { motion } from "framer-motion";

interface TaskPerformance {
    task: string;
    baseScore: number;
    alignedScore: number;
}

export function AlignmentTaxVisualization() {
    const [viewMode, setViewMode] = useState<"comparison" | "tax">("comparison");

    const tasks: TaskPerformance[] = [
        { task: "数学推理 (MATH)", baseScore: 45.2, alignedScore: 42.8 },
        { task: "代码生成 (HumanEval)", baseScore: 48.1, alignedScore: 45.3 },
        { task: "常识推理 (HellaSwag)", baseScore: 85.3, alignedScore: 84.1 },
        { task: "阅读理解 (SQuAD)", baseScore: 87.6, alignedScore: 88.2 },
        { task: "事实问答 (TriviaQA)", baseScore: 71.2, alignedScore: 70.5 },
        { task: "创造性写作", baseScore: 78.5, alignedScore: 72.1 },
    ];

    const calculateTax = (task: TaskPerformance) => {
        return task.baseScore - task.alignedScore;
    };

    const avgTax = tasks.reduce((sum, t) => sum + calculateTax(t), 0) / tasks.length;
    const maxTax = Math.max(...tasks.map(calculateTax));
    const maxTaxTask = tasks.find(t => calculateTax(t) === maxTax);

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-rose-50 to-orange-50 dark:from-slate-900 dark:to-rose-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    对齐税可视化
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Alignment Tax: 对齐带来的性能代价
                </p>
            </div>

            {/* 视图切换 */}
            <div className="flex gap-4 justify-center mb-6">
                <button
                    onClick={() => setViewMode("comparison")}
                    className={`px-6 py-2 rounded-lg font-semibold transition ${viewMode === "comparison"
                            ? "bg-rose-600 text-white"
                            : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                        }`}
                >
                    性能对比
                </button>
                <button
                    onClick={() => setViewMode("tax")}
                    className={`px-6 py-2 rounded-lg font-semibold transition ${viewMode === "tax"
                            ? "bg-rose-600 text-white"
                            : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                        }`}
                >
                    对齐税
                </button>
            </div>

            {/* 统计卡片 */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-white dark:bg-slate-800 p-4 rounded-xl shadow-lg">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">平均对齐税</div>
                    <div className="text-3xl font-bold text-rose-600 dark:text-rose-400">
                        {avgTax.toFixed(2)}%
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 p-4 rounded-xl shadow-lg">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">最大对齐税</div>
                    <div className="text-3xl font-bold text-orange-600 dark:text-orange-400">
                        {maxTax.toFixed(2)}%
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 p-4 rounded-xl shadow-lg">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">受影响最严重</div>
                    <div className="text-lg font-bold text-purple-600 dark:text-purple-400 truncate">
                        {maxTaxTask?.task.split(' ')[0]}
                    </div>
                </div>
            </div>

            {/* 可视化 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                {viewMode === "comparison" ? (
                    /* 性能对比视图 */
                    <div className="space-y-4">
                        <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                            Base Model vs Aligned Model
                        </h4>

                        {tasks.map((task, idx) => (
                            <div key={idx}>
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                                        {task.task}
                                    </span>
                                    <div className="flex items-center gap-4 text-sm">
                                        <span className="text-blue-600 dark:text-blue-400">
                                            Base: {task.baseScore.toFixed(1)}%
                                        </span>
                                        <span className="text-rose-600 dark:text-rose-400">
                                            Aligned: {task.alignedScore.toFixed(1)}%
                                        </span>
                                    </div>
                                </div>

                                <div className="relative h-8 bg-gray-200 dark:bg-gray-700 rounded-lg overflow-hidden">
                                    {/* Base Model Bar */}
                                    <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: `${task.baseScore}%` }}
                                        transition={{ duration: 1, delay: idx * 0.1 }}
                                        className="absolute top-0 left-0 h-4 bg-blue-500 opacity-70"
                                    />
                                    {/* Aligned Model Bar */}
                                    <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: `${task.alignedScore}%` }}
                                        transition={{ duration: 1, delay: idx * 0.1 + 0.2 }}
                                        className="absolute bottom-0 left-0 h-4 bg-rose-500 opacity-70"
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    /* 对齐税视图 */
                    <div className="space-y-4">
                        <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                            性能下降（Alignment Tax）
                        </h4>

                        {tasks.map((task, idx) => {
                            const tax = calculateTax(task);
                            const isBenefit = tax < 0;

                            return (
                                <div key={idx}>
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                                            {task.task}
                                        </span>
                                        <span className={`text-sm font-bold ${isBenefit
                                                ? "text-green-600 dark:text-green-400"
                                                : "text-rose-600 dark:text-rose-400"
                                            }`}>
                                            {isBenefit ? "+" : "-"}{Math.abs(tax).toFixed(2)}%
                                        </span>
                                    </div>

                                    <div className="relative h-6 bg-gray-200 dark:bg-gray-700 rounded-lg overflow-hidden">
                                        <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${(Math.abs(tax) / 10) * 100}%` }}
                                            transition={{ duration: 1, delay: idx * 0.1 }}
                                            className={`absolute top-0 left-0 h-full ${isBenefit
                                                    ? "bg-green-500"
                                                    : "bg-rose-500"
                                                }`}
                                        />
                                        <div className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-white">
                                            {isBenefit ? "性能提升" : "性能下降"}
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            {/* 原因解释 */}
            <div className="grid grid-cols-3 gap-4">
                <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border border-red-300 dark:border-red-700">
                    <h5 className="font-semibold text-red-700 dark:text-red-400 mb-2">能力退化</h5>
                    <p className="text-xs text-slate-600 dark:text-slate-400">
                        数学推理、代码生成等能力下降
                    </p>
                </div>

                <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border border-orange-300 dark:border-orange-700">
                    <h5 className="font-semibold text-orange-700 dark:text-orange-400 mb-2">过度谨慎</h5>
                    <p className="text-xs text-slate-600 dark:text-slate-400">
                        拒绝正常请求，过多免责声明
                    </p>
                </div>

                <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border border-purple-300 dark:border-purple-700">
                    <h5 className="font-semibold text-purple-700 dark:text-purple-400 mb-2">创造性受限</h5>
                    <p className="text-xs text-slate-600 dark:text-slate-400">
                        新颖性和多样性下降
                    </p>
                </div>
            </div>

            <div className="mt-6 bg-rose-100 dark:bg-rose-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                💡 <strong>权衡策略</strong>: Constitutional AI、任务特定微调、Reward Model Ensembling
            </div>
        </div>
    );
}
