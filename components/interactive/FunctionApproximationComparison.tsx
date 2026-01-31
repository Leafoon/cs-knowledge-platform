"use client";

import { useState } from "react";

export function FunctionApproximationComparison() {
    const [method, setMethod] = useState<"tabular" | "linear" | "nn">("tabular");

    const comparisons = {
        tabular: {
            name: "表格方法",
            storage: "O(|S|) 或 O(|S||A|)",
            generalization: "无泛化",
            scalability: "小规模任务",
            continuous: "不支持",
            color: "bg-red-50 dark:bg-red-900/20 border-red-500"
        },
        linear: {
            name: "线性函数逼近",
            storage: "O(d)（特征维度）",
            generalization: "基于特征相似性",
            scalability: "中等规模",
            continuous: "支持（需离散化特征）",
            color: "bg-yellow-50 dark:bg-yellow-900/20 border-yellow-500"
        },
        nn: {
            name: "神经网络",
            storage: "O(参数数量)",
            generalization: "强大",
            scalability: "大规模/高维",
            continuous: "完全支持",
            color: "bg-green-50 dark:bg-green-900/20 border-green-500"
        }
    };

    const current = comparisons[method];

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-gray-50 dark:from-slate-900 dark:to-gray-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    函数逼近方法对比
                </h3>
            </div>

            {/* 方法选择 */}
            <div className="flex justify-center gap-4 mb-6">
                {(Object.keys(comparisons) as Array<keyof typeof comparisons>).map(key => (
                    <button
                        key={key}
                        onClick={() => setMethod(key)}
                        className={`px-6 py-2 rounded-lg font-semibold transition-all ${
                            method === key
                                ? "bg-slate-700 text-white scale-105 shadow-lg"
                                : "bg-slate-200 text-slate-700 dark:bg-slate-700 dark:text-slate-300"
                        }`}
                    >
                        {comparisons[key].name}
                    </button>
                ))}
            </div>

            {/* 详细信息 */}
            <div className={`rounded-xl p-6 border-4 ${current.color}`}>
                <h4 className="text-2xl font-bold mb-4">{current.name}</h4>
                <div className="space-y-3">
                    <div><strong>存储复杂度:</strong> {current.storage}</div>
                    <div><strong>泛化能力:</strong> {current.generalization}</div>
                    <div><strong>可扩展性:</strong> {current.scalability}</div>
                    <div><strong>连续状态:</strong> {current.continuous}</div>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 神经网络是最强大的函数逼近器（Universal Approximation Theorem）
            </div>
        </div>
    );
}
