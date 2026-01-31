"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";

export function CVaRRiskMeasure() {
    const [alpha, setAlpha] = useState(0.1); // 风险水平
    const [showDistribution, setShowDistribution] = useState(true);

    // 生成回报分布样本（模拟）
    const returns = useMemo(() => {
        const samples = [];
        for (let i = 0; i < 100; i++) {
            // 模拟正态分布 + 一些负尾部风险
            const normal = (Math.random() + Math.random() + Math.random() - 1.5) * 20;
            const risk = Math.random() < 0.1 ? -50 * Math.random() : 0;
            samples.push(normal + risk);
        }
        return samples.sort((a, b) => a - b);
    }, []);

    // 计算VaR和CVaR
    const varIndex = Math.floor(alpha * returns.length);
    const var_ = returns[varIndex];
    const cvar = returns.slice(0, varIndex).reduce((a, b) => a + b, 0) / varIndex;
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-amber-50 to-yellow-50 dark:from-slate-900 dark:to-amber-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    CVaR 风险度量（交互式）
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    调整风险水平α，观察VaR和CVaR变化
                </p>
            </div>

            {/* Alpha滑块 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="flex justify-between items-center mb-4">
                    <label className="text-lg font-bold">风险水平 α</label>
                    <span className="text-2xl font-mono bg-amber-100 dark:bg-amber-900/30 px-4 py-2 rounded-lg">
                        {(alpha * 100).toFixed(0)}%
                    </span>
                </div>
                <input
                    type="range"
                    min="0.01"
                    max="0.5"
                    step="0.01"
                    value={alpha}
                    onChange={(e) => setAlpha(parseFloat(e.target.value))}
                    className="w-full h-3 bg-gradient-to-r from-green-200 to-red-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-2">
                    <span>保守 1%</span>
                    <span>中等 25%</span>
                    <span>激进 50%</span>
                </div>
            </div>

            {/* 回报分布可视化 */}
            {showDistribution && (
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                    <h4 className="text-lg font-bold mb-4">回报分布</h4>
                    <div className="relative h-40 bg-gradient-to-r from-red-100 via-yellow-100 to-green-100 dark:from-red-900/20 dark:via-yellow-900/20 dark:to-green-900/20 rounded-lg p-4">
                        {returns.map((r, i) => {
                            const isInCVaR = i < varIndex;
                            const x = (i / returns.length) * 100;
                            const y = ((r - Math.min(...returns)) / (Math.max(...returns) - Math.min(...returns))) * 80 + 10;

                            return (
                                <motion.div
                                    key={i}
                                    className={`absolute w-2 h-2 rounded-full ${isInCVaR ? 'bg-red-500' : 'bg-blue-400'}`}
                                    style={{ left: `${x}%`, bottom: `${y}%` }}
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    transition={{ delay: i * 0.01 }}
                                />
                            );
                        })}
                        {/* VaR线 */}
                        <div
                            className="absolute top-0 bottom-0 w-0.5 bg-yellow-600 dark:bg-yellow-400"
                            style={{ left: `${alpha * 100}%` }}
                        >
                            <span className="absolute -top-6 -left-8 text-xs font-bold bg-yellow-600 text-white px-2 py-1 rounded">
                                VaR
                            </span>
                        </div>
                    </div>
                    <div className="flex gap-4 mt-4 text-xs">
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                            <span>最坏{(alpha * 100).toFixed(0)}%（CVaR计算范围）</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
                            <span>其余回报</span>
                        </div>
                    </div>
                </div>
            )}

            {/* 度量对比 */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <motion.div
                    className="bg-blue-500 rounded-xl p-6 text-center text-white"
                    whileHover={{ scale: 1.05 }}
                >
                    <h4 className="text-sm font-semibold mb-2">期望回报 E[R]</h4>
                    <div className="text-3xl font-bold">{mean.toFixed(1)}</div>
                    <div className="text-xs mt-2 opacity-80">风险中性</div>
                </motion.div>

                <motion.div
                    className="bg-yellow-500 rounded-xl p-6 text-center text-white"
                    whileHover={{ scale: 1.05 }}
                >
                    <h4 className="text-sm font-semibold mb-2">VaR<sub>{(alpha * 100).toFixed(0)}%</sub></h4>
                    <div className="text-3xl font-bold">{var_.toFixed(1)}</div>
                    <div className="text-xs mt-2 opacity-80">{(alpha * 100).toFixed(0)}%分位数</div>
                </motion.div>

                <motion.div
                    className="bg-red-500 rounded-xl p-6 text-center text-white"
                    whileHover={{ scale: 1.05 }}
                    animate={{ boxShadow: ["0 0 0 rgba(239, 68, 68, 0)", "0 0 20px rgba(239, 68, 68, 0.5)", "0 0 0 rgba(239, 68, 68, 0)"] }}
                    transition={{ duration: 2, repeat: Infinity }}
                >
                    <h4 className="text-sm font-semibold mb-2">CVaR<sub>{(alpha * 100).toFixed(0)}%</sub></h4>
                    <div className="text-3xl font-bold">{cvar.toFixed(1)}</div>
                    <div className="text-xs mt-2 opacity-80">最坏{(alpha * 100).toFixed(0)}%平均</div>
                </motion.div>
            </div>

            {/* 说明 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">风险解读</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong>期望回报 ({mean.toFixed(1)}):</strong> 平均情况
                    </div>
                    <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                        <strong>VaR ({var_.toFixed(1)}):</strong> 有{(alpha * 100).toFixed(0)}%概率损失超过此值
                    </div>
                    <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                        <strong>CVaR ({cvar.toFixed(1)}):</strong>最坏{(alpha * 100).toFixed(0)}%情况的平均损失
                        {cvar < var_ && <span className="ml-2 text-red-600 font-bold">（比VaR更保守）</span>}
                    </div>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-600 dark:text-slate-400">
                💡 CVaR优化适合风险规避的应用（金融、医疗、自动驾驶）
            </div>
        </div>
    );
}
