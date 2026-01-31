"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

export function ValueFunctionEvolution() {
    // 4x4 GridWorld Values
    const [values, setValues] = useState<number[]>(Array(16).fill(0));
    const [iteration, setIteration] = useState(0);
    const [isRunning, setIsRunning] = useState(false);

    // Goal at 15 (bottom-right), Trap at 5.

    const performUpdate = () => {
        setValues(prev => {
            const next = [...prev];
            // Goal at 15
            next[15] = 1.0;
            // Trap at 5
            next[5] = -1.0;

            for (let i = 0; i < 15; i++) {
                if (i === 5) continue;
                // Simple averaging of neighbors (like Bellman update)
                const neighbors = [];
                if (i >= 4) neighbors.push(prev[i - 4]); // Up
                if (i < 12) neighbors.push(prev[i + 4]); // Down
                if (i % 4 !== 0) neighbors.push(prev[i - 1]); // Left
                if ((i + 1) % 4 !== 0) neighbors.push(prev[i + 1]); // Right

                const maxVal = Math.max(...neighbors);
                // Bellman Optimality: V(s) = 0.9 * max_neighbor
                next[i] = 0.9 * maxVal;
            }
            return next;
        });
        setIteration(it => it + 1);
    };

    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isRunning) {
            interval = setInterval(performUpdate, 200);
        }
        return () => clearInterval(interval);
    }, [isRunning]);

    const reset = () => {
        setIsRunning(false);
        setValues(Array(16).fill(0));
        setIteration(0);
    };

    const getColor = (val: number) => {
        // Red for neg, Green for pos
        if (val > 0) return `rgba(34, 197, 94, ${val})`; // Green
        if (val < 0) return `rgba(239, 68, 68, ${Math.abs(val)})`; // Red
        return "rgba(241, 245, 249, 1)"; // Slate-100
    };

    return (
        <div className="w-full max-w-xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl shadow-lg border border-slate-200 dark:border-slate-800">
            <div className="flex justify-between items-center mb-6">
                <h3 className="font-bold text-slate-800 dark:text-slate-100">
                    Value Iteration Process
                </h3>
                <div className="flex gap-2">
                    <button onClick={() => setIsRunning(!isRunning)} className="px-3 py-1 bg-blue-600 text-white rounded text-sm">
                        {isRunning ? "Pause" : "Start"}
                    </button>
                    <button onClick={reset} className="px-3 py-1 bg-slate-200 dark:bg-slate-700 rounded text-sm">
                        Reset
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-4 gap-2 mb-4">
                {values.map((v, i) => (
                    <motion.div
                        key={i}
                        className="aspect-square flex items-center justify-center rounded border border-slate-200 dark:border-slate-700 text-slate-800 font-mono text-xs font-bold transition-colors duration-200"
                        style={{ backgroundColor: getColor(v) }}
                    >
                        {i === 15 ? "GOAL" : i === 5 ? "TRAP" : v.toFixed(2)}
                    </motion.div>
                ))}
            </div>

            <div className="text-center text-sm text-slate-500">
                Iteration: {iteration}
            </div>
        </div>
    );
}
