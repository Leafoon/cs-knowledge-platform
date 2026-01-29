"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

export function PolicyIterationVisualizer() {
    const [round, setRound] = useState(0);
    const [phase, setPhase] = useState<"evaluation" | "improvement">("evaluation");
    const [isPlaying, setIsPlaying] = useState(false);
    const [values, setValues] = useState<number[][]>([]);
    const [policy, setPolicy] = useState<number[][]>([]);

    const gridSize = 4;
    const goalPos = { x: 3, y: 3 };
    const trapPos = { x: 1, y: 1 };
    const gamma = 0.9;

    // 初始化
    useEffect(() => {
        const initialValues = Array(gridSize).fill(0).map(() => Array(gridSize).fill(0));
        const initialPolicy = Array(gridSize).fill(0).map(() =>
            Array(gridSize).fill(Math.floor(Math.random() * 4))
        );
        setValues(initialValues);
        setPolicy(initialPolicy);
    }, []);

    // 自动播放
    useEffect(() => {
        if (!isPlaying || round >= 4) return;

        const timer = setTimeout(() => {
            if (phase === "evaluation") {
                // 策略评估（简化：直接计算一步）
                performPolicyEvaluation();
                setPhase("improvement");
            } else {
                // 策略改进
                performPolicyImprovement();
                setPhase("evaluation");
                setRound(round + 1);
            }
        }, 2000);

        return () => clearTimeout(timer);
    }, [isPlaying, phase, round, values, policy]);

    const performPolicyEvaluation = () => {
        const newValues = values.map((row, i) =>
            row.map((_, j) => {
                if (i === goalPos.x && j === goalPos.y) return 10;
                if (i === trapPos.x && j === trapPos.y) return -10;

                const action = policy[i][j];
                const neighbors = getNeighbors(i, j, action);
                const reward = -1;

                if (neighbors) {
                    const [ni, nj] = neighbors;
                    return reward + gamma * values[ni][nj];
                }
                return values[i][j];
            })
        );
        setValues(newValues);
    };

    const performPolicyImprovement = () => {
        const newPolicy = policy.map((row, i) =>
            row.map((_, j) => {
                if (i === goalPos.x && j === goalPos.y) return 0;
                if (i === trapPos.x && j === trapPos.y) return 0;

                // 找到最佳动作
                let bestAction = 0;
                let bestValue = -Infinity;

                for (let a = 0; a < 4; a++) {
                    const neighbors = getNeighbors(i, j, a);
                    if (neighbors) {
                        const [ni, nj] = neighbors;
                        const value = -1 + gamma * values[ni][nj];
                        if (value > bestValue) {
                            bestValue = value;
                            bestAction = a;
                        }
                    }
                }
                return bestAction;
            })
        );
        setPolicy(newPolicy);
    };

    const getNeighbors = (i: number, j: number, action: number): [number, number] | null => {
        const directions = [[-1, 0], [0, 1], [1, 0], [0, -1]]; // 上右下左
        const [di, dj] = directions[action];
        const ni = i + di;
        const nj = j + dj;

        if (ni >= 0 && ni < gridSize && nj >= 0 && nj < gridSize) {
            return [ni, nj];
        }
        return [i, j]; // 保持不动
    };

    const getArrow = (action: number) => {
        const arrows = ['↑', '→', '↓', '←'];
        return arrows[action];
    };

    const reset = () => {
        setRound(0);
        setPhase("evaluation");
        setIsPlaying(false);
        const initialValues = Array(gridSize).fill(0).map(() => Array(gridSize).fill(0));
        const initialPolicy = Array(gridSize).fill(0).map(() =>
            Array(gridSize).fill(Math.floor(Math.random() * 4))
        );
        setValues(initialValues);
        setPolicy(initialPolicy);
    };

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-slate-900 dark:to-purple-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    策略迭代可视化
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    评估-改进循环演示
                </p>
            </div>

            {/* 控制面板 */}
            <div className="flex justify-center items-center gap-4 mb-6">
                <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    disabled={round >= 4}
                    className="px-6 py-2 rounded-lg bg-purple-600 hover:bg-purple-700 disabled:bg-purple-300 text-white font-semibold transition-colors"
                >
                    {isPlaying ? "⏸ 暂停" : "▶ 播放"}
                </button>
                <button
                    onClick={reset}
                    className="px-6 py-2 rounded-lg bg-slate-600 hover:bg-slate-700 text-white font-semibold transition-colors"
                >
                    🔄 重置
                </button>
                <div className="px-4 py-2 rounded-lg bg-white dark:bg-slate-800 border-2 border-purple-500">
                    <span className="font-bold text-slate-800 dark:text-slate-100">
                        第 {round + 1} 轮
                    </span>
                </div>
                <div className={`px-4 py-2 rounded-lg border-2 ${phase === "evaluation"
                        ? "bg-blue-100 border-blue-500 text-blue-700"
                        : "bg-green-100 border-green-500 text-green-700"
                    }`}>
                    <span className="font-bold">
                        {phase === "evaluation" ? "📊 策略评估" : "🎯 策略改进"}
                    </span>
                </div>
            </div>

            {/* GridWorld 可视化 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}>
                    {values.map((row, i) =>
                        row.map((value, j) => {
                            const isGoal = i === goalPos.x && j === goalPos.y;
                            const isTrap = i === trapPos.x && j === trapPos.y;
                            const cellPolicy = policy[i][j];

                            return (
                                <motion.div
                                    key={`${i}-${j}`}
                                    className="aspect-square rounded-lg border-2 border-slate-200 dark:border-slate-600 p-2 flex flex-col items-center justify-center"
                                    style={{
                                        backgroundColor: isGoal ? "#10b981" : isTrap ? "#ef4444" : "#fff",
                                    }}
                                    animate={phase === "evaluation" ? { scale: [1, 1.05, 1] } : {}}
                                    transition={{ duration: 0.3 }}
                                >
                                    {isGoal && <div className="text-2xl mb-1">🎯</div>}
                                    {isTrap && <div className="text-2xl mb-1">💀</div>}
                                    {!isGoal && !isTrap && (
                                        <>
                                            <div className="text-xs font-bold text-slate-600 dark:text-slate-300">
                                                {value.toFixed(1)}
                                            </div>
                                            <div className="text-2xl text-blue-600">
                                                {getArrow(cellPolicy)}
                                            </div>
                                        </>
                                    )}
                                </motion.div>
                            );
                        })
                    )}
                </div>
            </div>

            {/* 说明 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-slate-800 rounded-lg p-4">
                    <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-2 flex items-center gap-2">
                        <span className="text-blue-500">📊</span> 策略评估
                    </h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                        给定策略 π，迭代计算状态价值函数 V^π，直到收敛。
                        使用 Bellman 期望方程：V(s) ← Σ π(a|s) [r + γV(s')]
                    </p>
                </div>
                <div className="bg-white dark:bg-slate-800 rounded-lg p-4">
                    <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-2 flex items-center gap-2">
                        <span className="text-green-500">🎯</span> 策略改进
                    </h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                        基于当前价值函数 V，贪心地选择最佳动作。
                        π'(s) ← argmax_a [r + γV(s')]
                    </p>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 提示：策略迭代通常在 3-5 轮内收敛到最优策略
            </div>
        </div>
    );
}
