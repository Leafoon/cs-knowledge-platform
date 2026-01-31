"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

export function PolicyIterationVisualizer() {
    type Stage = "eval" | "improve" | "done";
    const [iter, setIter] = useState(0);
    const [stage, setStage] = useState<Stage>("eval");
    const [policy, setPolicy] = useState<string[]>(Array(16).fill("?"));
    const [values, setValues] = useState<number[]>(Array(16).fill(0));

    // Grid: Goal at 15.
    // Optimal Policy: Arrows pointing to 15.

    // Simulated sequence of policies
    const getPolicyArrow = (idx: number, it: number) => {
        if (idx === 15) return "★";
        if (idx === 5) return "☠"; // Trap

        // Simple heuristic for visualization:
        // Iter 0: Random directions
        // Iter 1: Converging closer
        // Iter 2: Optimal

        const row = Math.floor(idx / 4);
        const col = idx % 4;

        if (it === 0) return ["↑", "↓", "←", "→"][idx % 4]; // Random-ish

        if (it >= 1) {
            // Greedy towards bottom-right (3,3)
            if (row < 3 && col < 3) return Math.random() > 0.5 ? "↓" : "→";
            if (row < 3) return "↓";
            if (col < 3) return "→";
        }
        return "✓"; // Optimal
    };

    const runStep = () => {
        if (stage === "eval") {
            // Evaluate: Update values
            const newValues = values.map((v, i) => {
                if (i === 15) return 10;
                if (i === 5) return -10;
                // Value increases with iteration
                return Math.min(10, v + (iter + 1) * 2);
            });
            setValues(newValues);
            setStage("improve");
        } else if (stage === "improve") {
            // Improve: Update policy based on values
            const newPolicy = policy.map((_, i) => getPolicyArrow(i, iter + 1));
            setPolicy(newPolicy);

            if (iter >= 2) {
                setStage("done");
            } else {
                setStage("eval");
                setIter(i => i + 1);
            }
        }
    };

    const reset = () => {
        setIter(0);
        setStage("eval");
        setPolicy(Array(16).fill("?").map((_, i) => getPolicyArrow(i, 0)));
        setValues(Array(16).fill(0));
    };

    // Init
    useEffect(() => reset(), []);

    return (
        <div className="w-full max-w-2xl mx-auto p-6 bg-slate-50 dark:bg-slate-900 rounded-xl shadow-lg border border-slate-200 dark:border-slate-800 flex flex-col items-center">
            <h3 className="text-xl font-bold mb-4 text-slate-800 dark:text-slate-100">
                策略迭代过程 (Policy Iteration)
            </h3>

            <div className="flex gap-8 w-full justify-center items-start">
                {/* Policy Grid */}
                <div className="flex flex-col items-center">
                    <div className="text-sm font-bold mb-2 text-blue-600 dark:text-blue-400">Policy π</div>
                    <div className="grid grid-cols-4 gap-1 p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                        {policy.map((arrow, i) => (
                            <motion.div
                                key={i}
                                layoutId={`p-${i}`}
                                className={`w-10 h-10 flex items-center justify-center bg-white dark:bg-slate-800 rounded shadow-sm text-lg font-bold
                                  ${stage === "improve" ? "scale-110 border-2 border-blue-500" : ""}`}
                            >
                                {arrow}
                            </motion.div>
                        ))}
                    </div>
                </div>

                {/* Flow Arrows */}
                <div className="flex flex-col justify-center h-[200px] gap-4">
                    <motion.div
                        animate={{ opacity: stage === "eval" ? 1 : 0.3, scale: stage === "eval" ? 1.2 : 1 }}
                        className="text-center"
                    >
                        <div className="text-xs text-slate-500">Evaluation</div>
                        <div className="text-2xl">➡️</div>
                    </motion.div>
                    <motion.div
                        animate={{ opacity: stage === "improve" ? 1 : 0.3, scale: stage === "improve" ? 1.2 : 1 }}
                        className="text-center"
                    >
                        <div className="text-2xl">⬅️</div>
                        <div className="text-xs text-slate-500">Improvement</div>
                    </motion.div>
                </div>

                {/* Value Grid */}
                <div className="flex flex-col items-center">
                    <div className="text-sm font-bold mb-2 text-green-600 dark:text-green-400">Value V</div>
                    <div className="grid grid-cols-4 gap-1 p-2 bg-green-100 dark:bg-green-900/30 rounded-lg">
                        {values.map((v, i) => (
                            <motion.div
                                key={i}
                                className={`w-10 h-10 flex items-center justify-center bg-white dark:bg-slate-800 rounded shadow-sm text-xs
                                    ${stage === "eval" ? "scale-110 border-2 border-green-500" : ""}`}
                            >
                                {v.toFixed(0)}
                            </motion.div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="mt-8 flex gap-4 items-center">
                <div className="text-sm font-mono text-slate-600 dark:text-slate-400">
                    Iteration: {iter} | Step: <span className="uppercase font-bold">{stage}</span>
                </div>
                <button
                    onClick={runStep}
                    disabled={stage === "done"}
                    className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50"
                >
                    {stage === "done" ? "Converged!" : "Next Step"}
                </button>
                <button onClick={reset} className="px-4 py-2 bg-slate-200 dark:bg-slate-700 rounded hover:bg-slate-300">
                    Reset
                </button>
            </div>
        </div>
    );
}
