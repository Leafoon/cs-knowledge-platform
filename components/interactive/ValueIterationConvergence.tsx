"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export function ValueIterationConvergence() {
    const [iteration, setIteration] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [convergenceData, setConvergenceData] = useState<any[]>([]);
    const [values, setValues] = useState<number[][]>([]);

    const gridSize = 4;
    const goalPos = { x: 3, y: 3 };
    const trapPos = { x: 1, y: 1 };
    const gamma = 0.9;
    const maxIterations = 50;

    useEffect(() => {
        reset();
    }, []);

    useEffect(() => {
        if (!isPlaying || iteration >= maxIterations) {
            if (iteration >= maxIterations) setIsPlaying(false);
            return;
        }

        const timer = setTimeout(() => {
            performValueIteration();
        }, 200);

        return () => clearTimeout(timer);
    }, [isPlaying, iteration, values]);

    const performValueIteration = () => {
        let maxDelta = 0;

        const newValues = values.map((row, i) =>
            row.map((oldValue, j) => {
                if (i === goalPos.x && j === goalPos.y) return 10;
                if (i === trapPos.x && j === trapPos.y) return -10;

                // 计算所有动作的 Q 值
                const qValues = [];
                for (let a = 0; a < 4; a++) {
                    const neighbors = getNeighbors(i, j, a);
                    if (neighbors) {
                        const [ni, nj] = neighbors;
                        const reward = -1;
                        qValues.push(reward + gamma * values[ni][nj]);
                    }
                }

                const newValue = qValues.length > 0 ? Math.max(...qValues) : oldValue;
                maxDelta = Math.max(maxDelta, Math.abs(newValue - oldValue));
                return newValue;
            })
        );

        setValues(newValues);
        setIteration(iteration + 1);

        // 记录收敛曲线
        setConvergenceData(prev => [
            ...prev,
            {
                iteration: iteration + 1,
                delta: maxDelta,
                maxValue: Math.max(...newValues.flat()),
                minValue: Math.min(...newValues.flat().filter(v => ![-10, 10].includes(v))),
            }
        ]);
    };

    const getNeighbors = (i: number, j: number, action: number): [number, number] | null => {
        const directions = [[-1, 0], [0, 1], [1, 0], [0, -1]];
        const [di, dj] = directions[action];
        const ni = i + di;
        const nj = j + dj;

        if (ni >= 0 && ni < gridSize && nj >= 0 && nj < gridSize) {
            return [ni, nj];
        }
        return [i, j];
    };

    const reset = () => {
        setIteration(0);
        setIsPlaying(false);
        setValues(Array(gridSize).fill(0).map(() => Array(gridSize).fill(0)));
        setConvergenceData([]);
    };

    const currentDelta = convergenceData.length > 0
        ? convergenceData[convergenceData.length - 1].delta
        : 0;

    const hasConverged = currentDelta < 0.01;

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-slate-900 dark:to-blue-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    价值迭代收敛过程
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    观察 Bellman 最优方程的迭代收敛
                </p>
            </div>

            {/* 控制面板 */}
            <div className="flex justify-center items-center gap-4 mb-6">
                <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    disabled={iteration >= maxIterations}
                    className="px-6 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-semibold transition-colors"
                >
                    {isPlaying ? "⏸ 暂停" : "▶ 播放"}
                </button>
                <button
                    onClick={() => {
                        if (iteration < maxIterations) {
                            performValueIteration();
                        }
                    }}
                    disabled={isPlaying || iteration >= maxIterations}
                    className="px-6 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-700 disabled:bg-cyan-300 text-white font-semibold transition-colors"
                >
                    ⏭ 单步
                </button>
                <button
                    onClick={reset}
                    className="px-6 py-2 rounded-lg bg-slate-600 hover:bg-slate-700 text-white font-semibold transition-colors"
                >
                    🔄 重置
                </button>
            </div>

            {/* 状态显示 */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-lg p-4 text-center">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">迭代次数</div>
                    <div className="text-2xl font-bold text-blue-600">{iteration}</div>
                </div>
                <div className="bg-white dark:bg-slate-800 rounded-lg p-4 text-center">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">最大变化 Δ</div>
                    <div className="text-2xl font-bold text-cyan-600">
                        {currentDelta.toFixed(4)}
                    </div>
                </div>
                <div className={`rounded-lg p-4 text-center ${hasConverged
                        ? "bg-green-100 dark:bg-green-900"
                        : "bg-yellow-100 dark:bg-yellow-900"
                    }`}>
                    <div className="text-sm font-semibold mb-1">状态</div>
                    <div className="text-xl font-bold">
                        {hasConverged ? "✅ 已收敛" : "⏳ 迭代中"}
                    </div>
                </div>
            </div>

            {/* GridWorld 可视化 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}>
                    {values.map((row, i) =>
                        row.map((value, j) => {
                            const isGoal = i === goalPos.x && j === goalPos.y;
                            const isTrap = i === trapPos.x && j === trapPos.y;

                            let bgColor = "#fff";
                            if (isGoal) bgColor = "#10b981";
                            else if (isTrap) bgColor = "#ef4444";
                            else if (value > 5) bgColor = "#86efac";
                            else if (value > 0) bgColor = "#fde047";
                            else if (value < 0) bgColor = "#fca5a5";

                            return (
                                <motion.div
                                    key={`${i}-${j}`}
                                    className="aspect-square rounded-lg border-2 border-slate-200 dark:border-slate-600 p-2 flex items-center justify-center"
                                    style={{ backgroundColor: bgColor }}
                                    animate={{ scale: [1, 1.05, 1] }}
                                    transition={{ duration: 0.2 }}
                                >
                                    {isGoal && <span className="text-2xl">🎯</span>}
                                    {isTrap && <span className="text-2xl">💀</span>}
                                    {!isGoal && !isTrap && (
                                        <span className="text-sm font-bold text-slate-700">
                                            {value.toFixed(1)}
                                        </span>
                                    )}
                                </motion.div>
                            );
                        })
                    )}
                </div>
            </div>

            {/* 收敛曲线 */}
            {convergenceData.length > 1 && (
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">
                        收敛曲线（Δ = max<sub>s</sub> |V<sub>k + 1</sub>(s) - V<sub>k</sub>(s)|）
                    </h4>
                    <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={convergenceData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                                dataKey="iteration"
                                label={{ value: '迭代次数', position: 'insideBottom', offset: -5 }}
                            />
                            <YAxis
                                label={{ value: 'Δ', angle: -90, position: 'insideLeft' }}
                            />
                            <Tooltip />
                            <Legend />
                            <Line
                                type="monotone"
                                dataKey="delta"
                                stroke="#3b82f6"
                                strokeWidth={2}
                                name="最大变化Δ"
                                dot={{ r: 3 }}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-4 text-center">
                        💡 收敛速度：||V_k - V*||_∞ ≤ γ^k ||V_0 - V*||_∞，速度为几何级数
                    </p>
                </div>
            )}

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 提示：价值迭代直接更新 V(s) ← max_a [r + γV(s')]，无需显式策略
            </div>
        </div>
    );
}
