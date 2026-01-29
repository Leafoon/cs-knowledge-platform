"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

export function ValueFunctionEvolution() {
    const [iteration, setIteration] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [values, setValues] = useState<number[][]>([]);

    // 5x5 GridWorld
    const gridSize = 5;
    const goalPos = { x: 4, y: 4 };
    const trapPos = { x: 1, y: 1 };
    const gamma = 0.9;

    // 初始化价值函数
    useEffect(() => {
        const initialValues = Array(gridSize).fill(0).map(() => Array(gridSize).fill(0));
        setValues(initialValues);
    }, []);

    // 价值迭代更新
    const updateValues = () => {
        const newValues = values.map((row, i) =>
            row.map((_, j) => {
                // 目标状态
                if (i === goalPos.x && j === goalPos.y) return 10;
                // 陷阱状态
                if (i === trapPos.x && j === trapPos.y) return -10;

                // Bellman 更新
                const neighbors = [
                    { x: i - 1, y: j },  // 上
                    { x: i + 1, y: j },  // 下
                    { x: i, y: j - 1 },  // 左
                    { x: i, y: j + 1 },  // 右
                ];

                let maxValue = -Infinity;
                for (const neighbor of neighbors) {
                    if (
                        neighbor.x >= 0 &&
                        neighbor.x < gridSize &&
                        neighbor.y >= 0 &&
                        neighbor.y < gridSize
                    ) {
                        const reward = -1; // 每步惩罚
                        const nextValue = values[neighbor.x][neighbor.y];
                        const actionValue = reward + gamma * nextValue;
                        maxValue = Math.max(maxValue, actionValue);
                    }
                }

                return maxValue === -Infinity ? 0 : maxValue;
            })
        );

        setValues(newValues);
        setIteration(iteration + 1);
    };

    // 自动播放
    useEffect(() => {
        if (!isPlaying) return;

        const interval = setInterval(() => {
            if (iteration < 50) {
                updateValues();
            } else {
                setIsPlaying(false);
            }
        }, 500);

        return () => clearInterval(interval);
    }, [isPlaying, iteration, values]);

    const reset = () => {
        const initialValues = Array(gridSize).fill(0).map(() => Array(gridSize).fill(0));
        setValues(initialValues);
        setIteration(0);
        setIsPlaying(false);
    };

    const getColor = (value: number) => {
        if (value >= 5) return "#10b981"; // 绿色（好）
        if (value >= 0) return "#fbbf24"; // 黄色（中等）
        if (value >= -5) return "#f97316"; // 橙色（差）
        return "#ef4444"; // 红色（很差）
    };

    return (
        <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-slate-900 dark:to-green-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    价值函数迭代演化
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    观察价值函数如何从目标状态向外传播
                </p>
            </div>

            {/* 控制面板 */}
            <div className="flex justify-center items-center gap-4 mb-6">
                <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    disabled={iteration >= 50}
                    className="px-6 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 disabled:bg-emerald-300 text-white font-semibold transition-colors"
                >
                    {isPlaying ? "⏸ 暂停" : "▶ 播放"}
                </button>
                <button
                    onClick={updateValues}
                    disabled={isPlaying || iteration >= 50}
                    className="px-6 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-semibold transition-colors"
                >
                    ⏭ 单步
                </button>
                <button
                    onClick={reset}
                    className="px-6 py-2 rounded-lg bg-slate-600 hover:bg-slate-700 text-white font-semibold transition-colors"
                >
                    🔄 重置
                </button>
                <div className="px-4 py-2 rounded-lg bg-white dark:bg-slate-800 border-2 border-emerald-500 font-mono font-bold text-slate-800 dark:text-slate-100">
                    迭代: {iteration}
                </div>
            </div>

            {/* GridWorld 可视化 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}>
                    {values.map((row, i) =>
                        row.map((value, j) => {
                            const isGoal = i === goalPos.x && j === goalPos.y;
                            const isTrap = i === trapPos.x && j === trapPos.y;

                            return (
                                <motion.div
                                    key={`${i}-${j}`}
                                    className="aspect-square rounded-lg flex flex-col items-center justify-center p-2 border-2 border-slate-200 dark:border-slate-600"
                                    style={{ backgroundColor: getColor(value) }}
                                    animate={{ scale: [1, 1.05, 1] }}
                                    transition={{ duration: 0.3 }}
                                >
                                    {isGoal && (
                                        <div className="text-2xl mb-1">🎯</div>
                                    )}
                                    {isTrap && (
                                        <div className="text-2xl mb-1">💀</div>
                                    )}
                                    <div className="text-xs font-bold text-white drop-shadow-lg">
                                        {value.toFixed(1)}
                                    </div>
                                </motion.div>
                            );
                        })
                    )}
                </div>
            </div>

            {/* 说明 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-slate-800 rounded-lg p-4">
                    <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-2">
                        📊 价值函数含义
                    </h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                        V(s) 表示从状态 s 开始，遵循最优策略，期望获得的累积折扣奖励。
                        颜色越绿表示状态越好。
                    </p>
                </div>
                <div className="bg-white dark:bg-slate-800 rounded-lg p-4">
                    <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-2">
                        🔄 Bellman 更新
                    </h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                        V(s) ← max_a [r + γ V(s')]
                        <br />
                        价值从目标状态（+10）向外传播，每步衰减 γ=0.9
                    </p>
                </div>
            </div>

            {/* 颜色图例 */}
            <div className="mt-6 flex justify-center gap-4">
                {[
                    { label: "很好 (≥5)", color: "#10b981" },
                    { label: "中等 (0-5)", color: "#fbbf24" },
                    { label: "较差 (-5-0)", color: "#f97316" },
                    { label: "很差 (<-5)", color: "#ef4444" },
                ].map((item) => (
                    <div key={item.label} className="flex items-center gap-2">
                        <div
                            className="w-4 h-4 rounded"
                            style={{ backgroundColor: item.color }}
                        />
                        <span className="text-xs text-slate-600 dark:text-slate-400">
                            {item.label}
                        </span>
                    </div>
                ))}
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 提示：价值函数收敛后，可以通过贪心选择最大价值的邻居来得到最优策略
            </div>
        </div>
    );
}
