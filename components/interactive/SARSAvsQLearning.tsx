"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { InlineMath } from "@/components/ui/Math";

export function SARSAvsQLearning() {
    const [mode, setMode] = useState<"SARSA" | "Q-Learning">("SARSA");
    const [agentPos, setAgentPos] = useState({ x: 0, y: 3 }); // Start at bottom-left (3,0) in 4x12 grid
    const [isRunning, setIsRunning] = useState(false);

    // Grid Setup: 4 rows, 12 cols
    // Start (3, 0), Goal (3, 11)
    // Cliff: (3, 1) to (3, 10)

    const ROWS = 4;
    const COLS = 12;

    const sarsaPath = [
        { x: 0, y: 3 }, { x: 0, y: 2 }, { x: 0, y: 1 }, { x: 0, y: 0 }, // Up to safe zone
        { x: 1, y: 0 }, { x: 2, y: 0 }, { x: 3, y: 0 }, { x: 4, y: 0 }, { x: 5, y: 0 }, { x: 6, y: 0 }, { x: 7, y: 0 }, { x: 8, y: 0 }, { x: 9, y: 0 }, { x: 10, y: 0 }, { x: 11, y: 0 }, // Across top
        { x: 11, y: 1 }, { x: 11, y: 2 }, { x: 11, y: 3 } // Down to goal
    ];

    const qLearningPath = [
        { x: 0, y: 3 }, // Start
        { x: 1, y: 3 }, { x: 2, y: 3 }, { x: 3, y: 3 }, { x: 4, y: 3 }, { x: 5, y: 3 }, { x: 6, y: 3 }, { x: 7, y: 3 }, { x: 8, y: 3 }, { x: 9, y: 3 }, { x: 10, y: 3 }, // The dangerous path
        { x: 11, y: 3 } // Goal
    ];

    const runSimulation = async () => {
        setIsRunning(true);
        const path = mode === "SARSA" ? sarsaPath : qLearningPath;

        for (let i = 0; i < path.length; i++) {
            setAgentPos(path[i]);
            await new Promise(r => setTimeout(r, 300));
        }
        setIsRunning(false);
    };

    const reset = () => {
        setAgentPos({ x: 0, y: 3 });
        setIsRunning(false);
    };

    return (
        <Card className="p-6 w-full bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-800">
            <div className="flex justify-between items-center mb-6">
                <h3 className="text-lg font-bold">Cliff Walking: SARSA vs Q-Learning</h3>
                <div className="flex gap-2">
                    <Button
                        variant={mode === "SARSA" ? "primary" : "secondary"}
                        onClick={() => { setMode("SARSA"); reset(); }}
                        disabled={isRunning}
                    >
                        SARSA (Safe)
                    </Button>
                    <Button
                        variant={mode === "Q-Learning" ? "primary" : "secondary"}
                        onClick={() => { setMode("Q-Learning"); reset(); }}
                        disabled={isRunning}
                    >
                        Q-Learning (Optimal)
                    </Button>
                </div>
            </div>

            {/* Grid */}
            <div className="relative border-2 border-slate-800 bg-white dark:bg-slate-800 mx-auto" style={{ width: 'fit-content' }}>
                <div className="grid grid-cols-12 gap-0">
                    {Array.from({ length: ROWS * COLS }).map((_, i) => {
                        const x = i % COLS;
                        const y = Math.floor(i / COLS);
                        const isCliff = y === 3 && x > 0 && x < 11;
                        const isGoal = y === 3 && x === 11;
                        const isStart = y === 3 && x === 0;

                        return (
                            <div
                                key={i}
                                className={`w-8 h-8 md:w-12 md:h-12 border border-slate-100 dark:border-slate-700 flex items-center justify-center text-xs
                                    ${isCliff ? "bg-red-200 dark:bg-red-900/50" : ""}
                                    ${isGoal ? "bg-green-200 dark:bg-green-900/50" : ""}
                                    ${isStart ? "bg-yellow-100 dark:bg-yellow-900/30" : ""}
                                `}
                            >
                                {isCliff && "☠️"}
                                {isGoal && "🏁"}
                                {isStart && "S"}
                            </div>
                        );
                    })}
                </div>

                {/* Agent */}
                <motion.div
                    className="absolute w-8 h-8 md:w-12 md:h-12 flex items-center justify-center text-2xl z-10 top-0 left-0"
                    animate={{
                        x: agentPos.x * (typeof window !== 'undefined' && window.innerWidth < 768 ? 32 : 48), // simplistic assumption for responsiveness
                        y: agentPos.y * (typeof window !== 'undefined' && window.innerWidth < 768 ? 32 : 48)
                    }}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                >
                    🤖
                </motion.div>
            </div>

            {/* Description */}
            <div className="mt-6 p-4 bg-slate-100 dark:bg-slate-800 rounded-lg text-sm text-slate-700 dark:text-slate-300">
                {mode === "SARSA" ? (
                    <p>
                        <strong>SARSA (On-policy):</strong>
                        考虑到在训练中会使用 <InlineMath>{"\\epsilon"}</InlineMath>-greedy 策略探索，如果走悬崖边，一旦随机选到“向下”，就会掉下去 (-100)。
                        因此 SARSA 学会了<b>远离悬崖</b>的安全路径，虽然路径更长，但训练回报更高（更少掉下去）。
                    </p>
                ) : (
                    <p>
                        <strong>Q-Learning (Off-policy):</strong>
                        直接学习<b>最优策略</b>（紧贴悬崖走）。虽然在训练中因为探索会频繁掉下悬崖，但 Q 表收敛到的策略是最短路径。
                        <span className="text-red-500 block mt-1">注意：在执行时如果不关闭 <InlineMath>{"\\epsilon"}</InlineMath> 探索，Q-Learning Agent 会经常死掉！</span>
                    </p>
                )}
            </div>

            <div className="mt-4 flex justify-center">
                <Button onClick={runSimulation} disabled={isRunning} className="w-48">
                    {isRunning ? "Running..." : "Start Episode"}
                </Button>
            </div>
        </Card>
    );
}
