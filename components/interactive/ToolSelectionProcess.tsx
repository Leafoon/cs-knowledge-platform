"use client";

import { useState } from "react";
import { motion } from "framer-motion";

interface Tool {
    name: string;
    description: string;
    successRate: number;
    cost: number;
    estimatedValue: number;
}

export function ToolSelectionProcess() {
    const [context, setContext] = useState<"search" | "calculate" | "code">("search");

    const contexts = {
        search: {
            query: "找到2024年世界人口数量",
            tools: [
                { name: "Wikipedia", description: "搜索维基百科", successRate: 0.85, cost: 1, estimatedValue: 8.5 },
                { name: "Calculator", description: "数学计算", successRate: 0.95, cost: 0.5, estimatedValue: 2.0 },
                { name: "PythonREPL", description: "执行代码", successRate: 0.90, cost: 2, estimatedValue: 3.0 },
                { name: "WebBrowser", description: "浏览网页", successRate: 0.75, cost: 3, estimatedValue: 9.0 },
            ] as Tool[]
        },
        calculate: {
            query: "计算fibonacci(20)的值",
            tools: [
                { name: "Wikipedia", description: "搜索维基百科", successRate: 0.85, cost: 1, estimatedValue: 1.5 },
                { name: "Calculator", description: "数学计算", successRate: 0.60, cost: 0.5, estimatedValue: 4.0 },
                { name: "PythonREPL", description: "执行代码", successRate: 0.95, cost: 2, estimatedValue: 14.0 },
                { name: "WebBrowser", description: "浏览网页", successRate: 0.70, cost: 3, estimatedValue: 2.0 },
            ] as Tool[]
        },
        code: {
            query: "生成快速排序的Python实现",
            tools: [
                { name: "Wikipedia", description: "搜索维基百科", successRate: 0.80, cost: 1, estimatedValue: 3.0 },
                { name: "Calculator", description: "数学计算", successRate: 0.30, cost: 0.5, estimatedValue: 0.5 },
                { name: "PythonREPL", description: "执行代码", successRate: 0.90, cost: 2, estimatedValue: 13.5 },
                { name: "WebBrowser", description: "浏览网页", successRate: 0.75, cost: 3, estimatedValue: 8.0 },
            ] as Tool[]
        }
    };

    const currentContext = contexts[context];

    // 排序工具（按估计价值）
    const sortedTools = [...currentContext.tools].sort((a, b) => b.estimatedValue - a.estimatedValue);
    const bestTool = sortedTools[0];

    // 计算UCB值
    const computeUCB = (tool: Tool, totalVisits: number, toolVisits: number) => {
        const exploitationTerm = tool.estimatedValue;
        const explorationTerm = Math.sqrt(2 * Math.log(totalVisits) / (toolVisits + 1));
        return exploitationTerm + 2 * explorationTerm;
    };

    const totalVisits = 10;
    const toolVisits = { "Wikipedia": 3, "Calculator": 2, "PythonREPL": 4, "WebBrowser": 1 };

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-slate-900 dark:to-emerald-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    工具选择过程
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    如何在多个工具中选择最优工具？
                </p>
            </div>

            {/* 上下文选择 */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                {(["search", "calculate", "code"] as const).map((ctx) => (
                    <button
                        key={ctx}
                        onClick={() => setContext(ctx)}
                        className={`p-4 rounded-xl border-2 transition ${context === ctx
                            ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20"
                            : "border-gray-200 dark:border-gray-700 bg-white dark:bg-slate-800"
                            }`}
                    >
                        <div className="text-lg font-bold text-emerald-600 dark:text-emerald-400 capitalize">
                            {ctx === "search" ? "搜索任务" : ctx === "calculate" ? "计算任务" : "代码任务"}
                        </div>
                    </button>
                ))}
            </div>

            {/* 当前查询 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-3 text-slate-800 dark:text-slate-100">当前任务</h4>
                <div className="bg-emerald-50 dark:bg-emerald-900/20 p-4 rounded-lg border-2 border-emerald-300 dark:border-emerald-700">
                    <div className="text-slate-800 dark:text-slate-100">{currentContext.query}</div>
                </div>
            </div>

            {/* 选择策略 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">选择策略</h4>

                <div className="grid grid-cols-3 gap-4 text-sm">
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-300 dark:border-blue-700">
                        <div className="font-semibold text-blue-700 dark:text-blue-400 mb-2">贪婪策略</div>
                        <div className="text-slate-600 dark:text-slate-400 mb-2">选择估计价值最高的工具</div>
                        <div className="font-mono text-xs bg-blue-100 dark:bg-blue-900/30 p-2 rounded">
                            arg max E[V(tool)]
                        </div>
                    </div>

                    <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border border-purple-300 dark:border-purple-700">
                        <div className="font-semibold text-purple-700 dark:text-purple-400 mb-2">UCB策略</div>
                        <div className="text-slate-600 dark:text-slate-400 mb-2">平衡探索与利用</div>
                        <div className="font-mono text-xs bg-purple-100 dark:bg-purple-900/30 p-2 rounded">
                            V + c√(log N / n)
                        </div>
                    </div>

                    <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-300 dark:border-green-700">
                        <div className="font-semibold text-green-700 dark:text-green-400 mb-2">RL策略</div>
                        <div className="text-slate-600 dark:text-slate-400 mb-2">学习最优策略</div>
                        <div className="font-mono text-xs bg-green-100 dark:bg-green-900/30 p-2 rounded">
                            π*(s) = arg max Q(s,a)
                        </div>
                    </div>
                </div>
            </div>

            {/* 工具评估 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">工具评估</h4>

                <div className="space-y-4">
                    {sortedTools.map((tool, idx) => {
                        const isBest = tool.name === bestTool.name;
                        const ucb = computeUCB(tool, totalVisits, toolVisits[tool.name as keyof typeof toolVisits]);

                        return (
                            <div
                                key={tool.name}
                                className={`p-4 rounded-lg border-2 ${isBest
                                    ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20"
                                    : "border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-800"
                                    }`}
                            >
                                <div className="flex items-center justify-between mb-3">
                                    <div className="flex items-center gap-3">
                                        {isBest && <div className="text-2xl">⭐</div>}
                                        <div>
                                            <div className="font-bold text-slate-800 dark:text-slate-100">
                                                {tool.name}
                                            </div>
                                            <div className="text-sm text-slate-600 dark:text-slate-400">
                                                {tool.description}
                                            </div>
                                        </div>
                                    </div>
                                    {isBest && (
                                        <span className="bg-emerald-600 text-white px-3 py-1 rounded-full text-xs font-semibold">
                                            最佳选择
                                        </span>
                                    )}
                                </div>

                                <div className="grid grid-cols-4 gap-4 text-sm">
                                    <div>
                                        <div className="text-slate-600 dark:text-slate-400 mb-1">成功率</div>
                                        <div className="font-semibold text-slate-800 dark:text-slate-100">
                                            {(tool.successRate * 100).toFixed(0)}%
                                        </div>
                                        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full mt-1">
                                            <div
                                                className="h-full bg-blue-600 rounded-full"
                                                style={{ width: `${tool.successRate * 100}%` }}
                                            />
                                        </div>
                                    </div>

                                    <div>
                                        <div className="text-slate-600 dark:text-slate-400 mb-1">成本</div>
                                        <div className="font-semibold text-slate-800 dark:text-slate-100">
                                            {tool.cost}x
                                        </div>
                                        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full mt-1">
                                            <div
                                                className="h-full bg-orange-600 rounded-full"
                                                style={{ width: `${(tool.cost / 3) * 100}%` }}
                                            />
                                        </div>
                                    </div>

                                    <div>
                                        <div className="text-slate-600 dark:text-slate-400 mb-1">估计价值</div>
                                        <div className="font-semibold text-emerald-600 dark:text-emerald-400">
                                            {tool.estimatedValue.toFixed(1)}
                                        </div>
                                        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full mt-1">
                                            <div
                                                className="h-full bg-emerald-600 rounded-full"
                                                style={{ width: `${(tool.estimatedValue / 15) * 100}%` }}
                                            />
                                        </div>
                                    </div>

                                    <div>
                                        <div className="text-slate-600 dark:text-slate-400 mb-1">UCB值</div>
                                        <div className="font-semibold text-purple-600 dark:text-purple-400">
                                            {ucb.toFixed(1)}
                                        </div>
                                        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full mt-1">
                                            <div
                                                className="h-full bg-purple-600 rounded-full"
                                                style={{ width: `${(ucb / 20) * 100}%` }}
                                            />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* 决策过程 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">决策过程</h4>

                <div className="space-y-3 text-sm">
                    <div className="flex items-start gap-3">
                        <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold flex-shrink-0">
                            1
                        </div>
                        <div className="flex-1">
                            <div className="font-semibold text-slate-800 dark:text-slate-100">编码上下文</div>
                            <div className="text-slate-600 dark:text-slate-400">将当前任务和历史记录编码为状态向量</div>
                        </div>
                    </div>

                    <div className="flex items-start gap-3">
                        <div className="w-8 h-8 rounded-full bg-purple-600 text-white flex items-center justify-center font-bold flex-shrink-0">
                            2
                        </div>
                        <div className="flex-1">
                            <div className="font-semibold text-slate-800 dark:text-slate-100">评估工具</div>
                            <div className="text-slate-600 dark:text-slate-400">
                                考虑成功率、成本、预期价值等因素
                            </div>
                        </div>
                    </div>

                    <div className="flex items-start gap-3">
                        <div className="w-8 h-8 rounded-full bg-emerald-600 text-white flex items-center justify-center font-bold flex-shrink-0">
                            3
                        </div>
                        <div className="flex-1">
                            <div className="font-semibold text-slate-800 dark:text-slate-100">选择工具</div>
                            <div className="text-slate-600 dark:text-slate-400">
                                使用策略网络或启发式方法选择 <strong className="text-emerald-600 dark:text-emerald-400">{bestTool.name}</strong>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-start gap-3">
                        <div className="w-8 h-8 rounded-full bg-orange-600 text-white flex items-center justify-center font-bold flex-shrink-0">
                            4
                        </div>
                        <div className="flex-1">
                            <div className="font-semibold text-slate-800 dark:text-slate-100">执行并学习</div>
                            <div className="text-slate-600 dark:text-slate-400">执行工具，根据结果更新策略</div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-emerald-100 dark:bg-emerald-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                💡 <strong>RL优化</strong>: 通过策略梯度学习最优工具选择策略，平衡探索与利用
            </div>
        </div>
    );
}
