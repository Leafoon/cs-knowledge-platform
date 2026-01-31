"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

export function ConvergenceProofVisualization() {
    const [iteration, setIteration] = useState(0);
    const [isAnimating, setIsAnimating] = useState(false);
    const [selectedAlgorithm, setSelectedAlgorithm] = useState<"VI" | "QL" | "PG">("VI");

    const maxIterations = 50;
    const gamma = 0.9;

    // 模拟值函数收敛
    const generateConvergence = (algo: string) => {
        const data = [];
        for (let k = 0; k <= maxIterations; k++) {
            let error;
            if (algo === "VI") {
                // 值迭代: 指数收敛 γ^k
                error = Math.pow(gamma, k);
            } else if (algo === "QL") {
                // Q-learning: 稍慢（随机性）
                error = Math.pow(gamma, k * 0.8) * (1 + 0.1 * Math.random());
            } else {
                // 策略梯度: 1/√k
                error = 1 / Math.sqrt(k + 1);
            }
            data.push({ k, error });
        }
        return data;
    };

    const convergenceData = generateConvergence(selectedAlgorithm);
    const currentError = convergenceData[iteration].error;

    // 启动/停止动画
    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isAnimating && iteration < maxIterations) {
            interval = setInterval(() => {
                setIteration(prev => Math.min(prev + 1, maxIterations));
            }, 100);
        }
        return () => clearInterval(interval);
    }, [isAnimating, iteration]);

    const algorithms = [
        { id: "VI", name: "值迭代 (VI)", rate: "O(γᵏ)", color: "blue" },
        { id: "QL", name: "Q-learning", rate: "O(γᵏ) + noise", color: "green" },
        { id: "PG", name: "策略梯度 (PG)", rate: "O(1/√k)", color: "purple" }
    ];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-slate-900 dark:to-blue-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    收敛性证明可视化
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Contraction Mapping & Convergence Rates
                </p>
            </div>

            {/* 算法选择 */}
            <div className="flex gap-4 justify-center mb-6">
                {algorithms.map((algo) => (
                    <button
                        key={algo.id}
                        onClick={() => {
                            setSelectedAlgorithm(algo.id as any);
                            setIteration(0);
                            setIsAnimating(false);
                        }}
                        className={`px-6 py-3 rounded-xl font-semibold transition ${selectedAlgorithm === algo.id
                                ? `bg-${algo.color}-600 text-white`
                                : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                            }`}
                    >
                        <div className="text-sm">{algo.name}</div>
                        <div className="text-xs opacity-75">{algo.rate}</div>
                    </button>
                ))}
            </div>

            {/* 收敛分析 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    Banach不动点定理
                </h4>

                <div className="grid grid-cols-2 gap-6">
                    {/* 压缩性质 */}
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                        <h5 className="font-semibold text-blue-700 dark:text-blue-400 mb-2">
                            压缩映射
                        </h5>
                        <div className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
                            <div>||T V - T U||∞ ≤ γ ||V - U||∞</div>
                            <div className="text-xs text-slate-500 dark:text-slate-400">
                                γ = {gamma} {"<"} 1
                            </div>
                        </div>
                    </div>

                    {/* 不动点 */}
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                        <h5 className="font-semibold text-green-700 dark:text-green-400 mb-2">
                            唯一不动点
                        </h5>
                        <div className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
                            <div>∃! V* : T V* = V*</div>
                            <div className="text-xs text-slate-500 dark:text-slate-400">
                                Bellman最优方程
                            </div>
                        </div>
                    </div>

                    {/* 收敛速度 */}
                    <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg col-span-2">
                        <h5 className="font-semibold text-purple-700 dark:text-purple-400 mb-2">
                            收敛速度
                        </h5>
                        <div className="text-sm text-slate-700 dark:text-slate-300">
                            ||V_k - V*||∞ ≤ γᵏ ||V₀ - V*||∞
                        </div>
                        <div className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                            需要迭代次数: k = O((1/(1−γ)) log(1/ε))
                        </div>
                    </div>
                </div>
            </div>

            {/* 迭代可视化 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100">
                        迭代收敛过程
                    </h4>
                    <div className="flex items-center gap-4">
                        <div className="text-sm text-slate-600 dark:text-slate-400">
                            迭代: <strong>{iteration}</strong> / {maxIterations}
                        </div>
                        <div className="text-sm text-slate-600 dark:text-slate-400">
                            误差: <strong>{currentError.toFixed(6)}</strong>
                        </div>
                    </div>
                </div>

                {/* 误差曲线 */}
                <div className="relative h-64 bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                    <svg className="w-full h-full">
                        {/* 坐标轴 */}
                        <line x1="40" y1="10" x2="40" y2="220" stroke="currentColor" strokeWidth="2" className="text-gray-400" />
                        <line x1="40" y1="220" x2="780" y2="220" stroke="currentColor" strokeWidth="2" className="text-gray-400" />

                        {/* Y轴标签 */}
                        <text x="10" y="20" className="text-xs fill-current text-gray-600 dark:text-gray-400">1.0</text>
                        <text x="10" y="120" className="text-xs fill-current text-gray-600 dark:text-gray-400">0.5</text>
                        <text x="10" y="220" className="text-xs fill-current text-gray-600 dark:text-gray-400">0.0</text>

                        {/* X轴标签 */}
                        <text x="35" y="240" className="text-xs fill-current text-gray-600 dark:text-gray-400">0</text>
                        <text x="400" y="240" className="text-xs fill-current text-gray-600 dark:text-gray-400">{maxIterations / 2}</text>
                        <text x="760" y="240" className="text-xs fill-current text-gray-600 dark:text-gray-400">{maxIterations}</text>

                        {/* 理论界（虚线） */}
                        <path
                            d={convergenceData.slice(0, iteration + 1).map((d, i) => {
                                const x = 40 + (i / maxIterations) * 740;
                                const y = 220 - d.error * 210;
                                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                            }).join(' ')}
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeDasharray="5,5"
                            className="text-blue-400"
                        />

                        {/* 实际收敛曲线 */}
                        <motion.path
                            d={convergenceData.slice(0, iteration + 1).map((d, i) => {
                                const x = 40 + (i / maxIterations) * 740;
                                const y = 220 - d.error * 210;
                                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                            }).join(' ')}
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="3"
                            className={`text-${algorithms.find(a => a.id === selectedAlgorithm)?.color}-600`}
                            initial={{ pathLength: 0 }}
                            animate={{ pathLength: 1 }}
                            transition={{ duration: 0.5 }}
                        />

                        {/* 当前点 */}
                        <motion.circle
                            cx={40 + (iteration / maxIterations) * 740}
                            cy={220 - currentError * 210}
                            r="6"
                            className={`fill-current text-${algorithms.find(a => a.id === selectedAlgorithm)?.color}-600`}
                            animate={{ scale: [1, 1.3, 1] }}
                            transition={{ duration: 0.5, repeat: Infinity }}
                        />
                    </svg>
                </div>

                {/* 控制按钮 */}
                <div className="flex gap-3 mt-4 justify-center">
                    <button
                        onClick={() => setIsAnimating(!isAnimating)}
                        className="px-6 py-2 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition"
                    >
                        {isAnimating ? "暂停" : "播放"}
                    </button>
                    <button
                        onClick={() => {
                            setIteration(0);
                            setIsAnimating(false);
                        }}
                        className="px-6 py-2 bg-gray-600 text-white rounded-lg font-semibold hover:bg-gray-700 transition"
                    >
                        重置
                    </button>
                    <button
                        onClick={() => setIteration(Math.min(iteration + 1, maxIterations))}
                        className="px-6 py-2 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition"
                        disabled={iteration >= maxIterations}
                    >
                        单步
                    </button>
                </div>
            </div>

            {/* 数学公式 */}
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 p-4 rounded-lg text-center text-sm">
                <div className="font-semibold text-slate-700 dark:text-slate-300 mb-2">
                    收敛定理
                </div>
                <div className="text-slate-600 dark:text-slate-400">
                    对于压缩算子T，迭代序列 V_{"{k+1}"} = T V_{"{k}"} 以指数速度收敛到唯一不动点 V*
                </div>
            </div>
        </div>
    );
}
