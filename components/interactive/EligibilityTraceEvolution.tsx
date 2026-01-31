"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

const gamma = 0.9;
const lambda = 0.8;
const path = [0, 1, 2, 1, 3, 4];  // 示例路径

export function EligibilityTraceEvolution() {
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [traces, setTraces] = useState([0, 0, 0, 0, 0]);
    const [visitedStates, setVisitedStates] = useState<number[]>([]);

    useEffect(() => {
        if (!isPlaying || step >= path.length) {
            if (step >= path.length) setIsPlaying(false);
            return;
        }

        const timer = setTimeout(() => {
            const currentState = path[step];

            setTraces(prevTraces => prevTraces.map((trace, idx) => {
                if (idx === currentState) {
                    return trace * gamma * lambda + 1;  // 累积迹
                } else {
                    return trace * gamma * lambda;  // 衰减
                }
            }));

            setVisitedStates(prev => [...prev, currentState]);
            setStep(s => s + 1);
        }, 1500);

        return () => clearTimeout(timer);
    }, [isPlaying, step]);

    const reset = () => {
        setStep(0);
        setIsPlaying(false);
        setTraces([0, 0, 0, 0, 0]);
        setVisitedStates([]);
    };

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-teal-50 to-emerald-50 dark:from-slate-900 dark:to-teal-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    资格迹演化过程
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    观察资格迹如何累积和衰减
                </p>
            </div>

            {/* 控制面板 */}
            <div className="flex justify-center items-center gap-4 mb-6">
                <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    disabled={step >= path.length}
                    className="px-6 py-2 rounded-lg bg-teal-600 hover:bg-teal-700 disabled:bg-teal-300 text-white font-semibold transition-colors"
                >
                    {isPlaying ? "⏸ 暂停" : "▶ 播放"}
                </button>
                <button
                    onClick={reset}
                    className="px-6 py-2 rounded-lg bg-slate-600 hover:bg-slate-700 text-white font-semibold transition-colors"
                >
                    🔄 重置
                </button>
                <div className="px-4 py-2 rounded-lg bg-white dark:bg-slate-800 border-2 border-teal-500">
                    <span className="font-bold text-slate-800 dark:text-slate-100">
                        Step {step} / {path.length}
                    </span>
                </div>
            </div>

            {/* 状态和资格迹可视化 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="grid grid-cols-5 gap-4">
                    {traces.map((trace, idx) => (
                        <div key={idx} className="flex flex-col items-center">
                            <motion.div
                                className={`w-20 h-20 rounded-xl flex items-center justify-center border-4 ${step < path.length && path[step] === idx
                                        ? "border-teal-500 bg-teal-100 dark:bg-teal-900/30"
                                        : "border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-700"
                                    }`}
                                animate={step < path.length && path[step] === idx ? { scale: [1, 1.2, 1] } : {}}
                            >
                                <div className="text-center">
                                    <div className="text-xs text-slate-600 dark:text-slate-400">S{idx}</div>
                                    <div className="text-sm font-bold text-slate-800 dark:text-slate-100">
                                        e={trace.toFixed(2)}
                                    </div>
                                </div>
                            </motion.div>

                            {/* 资格迹柱状图 */}
                            <div className="w-full h-24 mt-2 bg-slate-100 dark:bg-slate-700 rounded relative overflow-hidden">
                                <motion.div
                                    className="absolute bottom-0 w-full bg-gradient-to-t from-teal-500 to-teal-300"
                                    initial={{ height: 0 }}
                                    animate={{ height: `${Math.min(trace * 20, 100)}%` }}
                                    transition={{ duration: 0.3 }}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* 路径显示 */}
            {visitedStates.length > 0 && (
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                    <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-3">
                        访问路径
                    </h4>
                    <div className="flex items-center gap-2 flex-wrap">
                        {visitedStates.map((state, idx) => (
                            <div key={idx} className="flex items-center">
                                <div className="px-3 py-1 rounded bg-teal-100 dark:bg-teal-900/30 font-mono font-semibold">
                                    S{state}
                                </div>
                                {idx < visitedStates.length - 1 && (
                                    <span className="mx-2 text-slate-400">→</span>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* 更新公式 */}
            <div className="bg-white dark:bg-slate-800 rounded-lg p-4">
                <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-2">
                    累积迹更新公式
                </h4>
                <div className="font-mono text-sm text-slate-600 dark:text-slate-400 space-y-1">
                    <div>e(s) ← γλe(s) + 1  <span className="text-teal-600">(if s = S<sub>t</sub>)</span></div>
                    <div>e(s) ← γλe(s)      <span className="text-slate-400">(otherwise)</span></div>
                    <div className="mt-2 text-xs">
                        γ = {gamma}, λ = {lambda}
                    </div>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 资格迹记录每个状态的"资格"，用于分配信用
            </div>
        </div>
    );
}
