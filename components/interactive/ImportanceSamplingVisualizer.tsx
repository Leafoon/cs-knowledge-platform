"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function ImportanceSamplingVisualizer() {
    const [scenario, setScenario] = useState<"similar" | "different">("similar");
    const [method, setMethod] = useState<"ordinary" | "weighted">("weighted");

    // 模拟数据
    const episodes = [
        { id: 1, targetProb: 0.8, behaviorProb: 0.7, return: 10 },
        { id: 2, targetProb: 0.6, behaviorProb: 0.5, return: 8 },
        { id: 3, targetProb: 0.9, behaviorProb: 0.8, return: 12 },
        { id: 4, targetProb: 0.1, behaviorProb: 0.3, return: -5 },
        { id: 5, targetProb: 0.7, behaviorProb: 0.6, return: 9 },
    ];

    const differentEpisodes = [
        { id: 1, targetProb: 0.9, behaviorProb: 0.1, return: 10 },
        { id: 2, targetProb: 0.8, behaviorProb: 0.2, return: 8 },
        { id: 3, targetProb: 0.95, behaviorProb: 0.15, return: 12 },
        { id: 4, targetProb: 0.1, behaviorProb: 0.7, return: -5 },
        { id: 5, targetProb: 0.85, behaviorProb: 0.25, return: 9 },
    ];

    const data = scenario === "similar" ? episodes : differentEpisodes;

    const calculateEstimate = () => {
        if (method === "ordinary") {
            // 普通重要性采样
            const n = data.length;
            const sum = data.reduce((acc, ep) => {
                const ratio = ep.targetProb / ep.behaviorProb;
                return acc + ratio * ep.return;
            }, 0);
            return sum / n;
        } else {
            // 加权重要性采样
            const numerator = data.reduce((acc, ep) => {
                const ratio = ep.targetProb / ep.behaviorProb;
                return acc + ratio * ep.return;
            }, 0);
            const denominator = data.reduce((acc, ep) => {
                const ratio = ep.targetProb / ep.behaviorProb;
                return acc + ratio;
            }, 0);
            return numerator / denominator;
        }
    };

    const estimate = calculateEstimate();

    const getMaxRatio = () => {
        return Math.max(...data.map(ep => ep.targetProb / ep.behaviorProb));
    };

    const maxRatio = getMaxRatio();

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-violet-50 to-fuchsia-50 dark:from-slate-900 dark:to-violet-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    重要性采样可视化
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    Off-policy 学习的核心技术
                </p>
            </div>

            {/* 控制面板 */}
            <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-lg p-4">
                    <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3">
                        策略差异
                    </h4>
                    <div className="flex gap-2">
                        <button
                            onClick={() => setScenario("similar")}
                            className={`flex-1 px-4 py-2 rounded-lg font-semibold transition-colors ${scenario === "similar"
                                    ? "bg-violet-600 text-white"
                                    : "bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-300"
                                }`}
                        >
                            相似策略
                        </button>
                        <button
                            onClick={() => setScenario("different")}
                            className={`flex-1 px-4 py-2 rounded-lg font-semibold transition-colors ${scenario === "different"
                                    ? "bg-fuchsia-600 text-white"
                                    : "bg-fuchsia-100 text-fuchsia-700 dark:bg-fuchsia-900/30 dark:text-fuchsia-300"
                                }`}
                        >
                            差异策略
                        </button>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-lg p-4">
                    <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3">
                        采样方法
                    </h4>
                    <div className="flex gap-2">
                        <button
                            onClick={() => setMethod("ordinary")}
                            className={`flex-1 px-4 py-2 rounded-lg font-semibold transition-colors ${method === "ordinary"
                                    ? "bg-blue-600 text-white"
                                    : "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300"
                                }`}
                        >
                            普通 IS
                        </button>
                        <button
                            onClick={() => setMethod("weighted")}
                            className={`flex-1 px-4 py-2 rounded-lg font-semibold transition-colors ${method === "weighted"
                                    ? "bg-green-600 text-white"
                                    : "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300"
                                }`}
                        >
                            加权 IS
                        </button>
                    </div>
                </div>
            </div>

            {/* 统计信息 */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-lg p-4 text-center">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">价值估计</div>
                    <div className="text-2xl font-bold text-violet-600">{estimate.toFixed(2)}</div>
                </div>
                <div className="bg-white dark:bg-slate-800 rounded-lg p-4 text-center">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">最大比率</div>
                    <div className="text-2xl font-bold text-fuchsia-600">{maxRatio.toFixed(2)}</div>
                </div>
                <div className={`rounded-lg p-4 text-center ${maxRatio > 5
                        ? "bg-red-100 dark:bg-red-900/30"
                        : "bg-green-100 dark:bg-green-900/30"
                    }`}>
                    <div className="text-sm font-semibold mb-1">方差状态</div>
                    <div className="text-xl font-bold">
                        {maxRatio > 5 ? "⚠️ 高方差" : "✅ 正常"}
                    </div>
                </div>
            </div>

            {/* Episode 表格 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">
                    Episode 数据与重要性采样比
                </h4>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b-2 border-slate-200 dark:border-slate-600">
                                <th className="px-4 py-2 text-left">Episode</th>
                                <th className="px-4 py-2 text-center">π(τ)</th>
                                <th className="px-4 py-2 text-center">b(τ)</th>
                                <th className="px-4 py-2 text-center">ρ = π/b</th>
                                <th className="px-4 py-2 text-center">Return G</th>
                                <th className="px-4 py-2 text-center">ρ × G</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.map((ep, idx) => {
                                const ratio = ep.targetProb / ep.behaviorProb;
                                const weighted = ratio * ep.return;

                                return (
                                    <motion.tr
                                        key={ep.id}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: idx * 0.1 }}
                                        className="border-b border-slate-100 dark:border-slate-700"
                                    >
                                        <td className="px-4 py-3 font-semibold">{ep.id}</td>
                                        <td className="px-4 py-3 text-center">{ep.targetProb.toFixed(2)}</td>
                                        <td className="px-4 py-3 text-center">{ep.behaviorProb.toFixed(2)}</td>
                                        <td className={`px-4 py-3 text-center font-bold ${ratio > 3 ? "text-red-600" : "text-green-600"
                                            }`}>
                                            {ratio.toFixed(2)}
                                        </td>
                                        <td className="px-4 py-3 text-center">{ep.return}</td>
                                        <td className="px-4 py-3 text-center font-semibold">
                                            {weighted.toFixed(2)}
                                        </td>
                                    </motion.tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>

                <div className="mt-4 p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
                    <div className="text-sm font-mono text-slate-700 dark:text-slate-300">
                        {method === "ordinary" ? (
                            <>
                                <div>普通 IS: V(s) = (1/n) Σ ρᵢGᵢ</div>
                                <div className="mt-1">
                                    = (1/{data.length}) × {data.reduce((acc, ep) => {
                                        const ratio = ep.targetProb / ep.behaviorProb;
                                        return acc + ratio * ep.return;
                                    }, 0).toFixed(2)}
                                </div>
                                <div className="mt-1">= {estimate.toFixed(2)}</div>
                            </>
                        ) : (
                            <>
                                <div>加权 IS: V(s) = (Σ ρᵢGᵢ) / (Σ ρᵢ)</div>
                                <div className="mt-1">
                                    = {data.reduce((acc, ep) => {
                                        const ratio = ep.targetProb / ep.behaviorProb;
                                        return acc + ratio * ep.return;
                                    }, 0).toFixed(2)} / {data.reduce((acc, ep) => {
                                        const ratio = ep.targetProb / ep.behaviorProb;
                                        return acc + ratio;
                                    }, 0).toFixed(2)}
                                </div>
                                <div className="mt-1">= {estimate.toFixed(2)}</div>
                            </>
                        )}
                    </div>
                </div>
            </div>

            {/* 对比说明 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border-l-4 border-blue-500">
                    <h5 className="font-bold text-blue-800 dark:text-blue-300 mb-2">
                        📊 普通重要性采样
                    </h5>
                    <p className="text-sm text-blue-700 dark:text-blue-400">
                        <strong>无偏</strong>：E[V(s)] = V^π(s)<br />
                        <strong>高方差</strong>：ρ 可能很大导致方差爆炸<br />
                        <strong>实践</strong>：很少使用
                    </p>
                </div>
                <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border-l-4 border-green-500">
                    <h5 className="font-bold text-green-800 dark:text-green-300 mb-2">
                        ✅ 加权重要性采样
                    </h5>
                    <p className="text-sm text-green-700 dark:text-green-400">
                        <strong>有偏（渐近无偏）</strong>：lim E[V(s)] = V^π(s)<br />
                        <strong>低方差</strong>：权重归一化抑制极值<br />
                        <strong>实践</strong>：推荐使用
                    </p>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 提示：ρ = π(τ)/b(τ) 是重要性采样的核心，策略差异越大方差越大
            </div>
        </div>
    );
}
