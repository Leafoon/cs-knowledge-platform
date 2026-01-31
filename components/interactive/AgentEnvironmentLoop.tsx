"use client";

import { motion } from "framer-motion";
import { useState } from "react";

export function AgentEnvironmentLoop() {
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    const stages = [
        { id: 0, agent: "观察 State", env: "提供当前状态", highlight: "state" },
        { id: 1, agent: "选择 Action", env: "等待动作", highlight: "action" },
        { id: 2, agent: "等待反馈", env: "执行动作", highlight: "env" },
        { id: 3, agent: "计算 Reward & State'", env: "返回奖励和新状态", highlight: "reward" },
    ];

    const handlePlay = () => {
        if (isPlaying) return;
        setIsPlaying(true);
        const interval = setInterval(() => {
            setStep(prev => {
                const next = (prev + 1) % stages.length;
                if (next === 0) {
                    clearInterval(interval);
                    setIsPlaying(false);
                }
                return next;
            });
        }, 1500);
    };

    const currentStage = stages[step];

    return (
        <div className="w-full max-w-4xl mx-auto p-8 bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
            <div className="flex justify-between items-center mb-12">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
                    Agent-Environment 交互循环
                </h3>
                <button
                    onClick={handlePlay}
                    disabled={isPlaying}
                    className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center gap-2 shadow-md"
                >
                    {isPlaying ? (
                        <span className="flex items-center gap-1">
                            <span className="animate-spin text-lg">↻</span> 运行中...
                        </span>
                    ) : (
                        <span>▶️ 演示交互</span>
                    )}
                </button>
            </div>

            <div className="relative h-64 flex items-center justify-between px-12">
                {/* Agent Node */}
                <motion.div
                    className="w-48 h-48 bg-white dark:bg-slate-800 rounded-2xl shadow-lg border-2 border-purple-500/30 flex flex-col items-center justify-center relative z-10"
                    animate={{
                        scale: currentStage.highlight === "action" || currentStage.highlight === "state" ? 1.05 : 1,
                        borderColor: currentStage.highlight === "action" ? "rgba(168, 85, 247, 1)" : "rgba(168, 85, 247, 0.3)",
                        boxShadow: currentStage.highlight === "action" ? "0 0 30px rgba(168, 85, 247, 0.4)" : "0 4px 6px rgba(0,0,0,0.1)",
                    }}
                >
                    <div className="text-6xl mb-4">🧠</div>
                    <div className="text-xl font-bold text-purple-600 dark:text-purple-400">Agent</div>
                    <div className="text-xs text-slate-500 font-mono mt-1">Policy π(s)</div>
                </motion.div>

                {/* Connections */}
                <div className="flex-1 h-full relative mx-4">
                    {/* Top Arrow: Action */}
                    <div className="absolute top-1/3 left-0 right-0 h-[2px] bg-slate-300 dark:bg-slate-600">
                        <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-white dark:bg-slate-900 px-2 text-xs font-bold text-slate-500 uppercase tracking-widest">
                            Action ($a_t$)
                        </div>
                        {currentStage.highlight === "action" && (
                            <motion.div
                                className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-purple-500 rounded-full shadow-lg"
                                initial={{ left: "0%" }}
                                animate={{ left: "100%" }}
                                transition={{ duration: 1.0, ease: "easeInOut" }}
                            />
                        )}
                        <div className="absolute right-0 top-1/2 -translate-y-[5px] border-t-[6px] border-t-transparent border-b-[6px] border-b-transparent border-l-[10px] border-l-slate-300 dark:border-l-slate-600"></div>
                    </div>

                    {/* Bottom Arrow: Reward & State */}
                    <div className="absolute bottom-1/3 left-0 right-0 h-[2px] bg-slate-300 dark:bg-slate-600">
                        <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 bg-white dark:bg-slate-900 px-2 text-xs font-bold text-slate-500 uppercase tracking-widest whitespace-nowrap">
                            Reward {'($r_{t + 1}$)'} + State {'($s_{t + 1}$)'}
                        </div>
                        {currentStage.highlight === "reward" && (
                            <motion.div
                                className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-green-500 rounded-full shadow-lg"
                                initial={{ right: "0%" }}
                                animate={{ right: "100%" }}
                                transition={{ duration: 1.0, ease: "easeInOut" }}
                            />
                        )}
                        <div className="absolute left-0 top-1/2 -translate-y-[5px] -rotate-180 border-t-[6px] border-t-transparent border-b-[6px] border-b-transparent border-l-[10px] border-l-slate-300 dark:border-l-slate-600"></div>
                    </div>
                </div>

                {/* Environment Node */}
                <motion.div
                    className="w-48 h-48 bg-white dark:bg-slate-800 rounded-2xl shadow-lg border-2 border-green-500/30 flex flex-col items-center justify-center relative z-10"
                    animate={{
                        scale: currentStage.highlight === "env" || currentStage.highlight === "reward" ? 1.05 : 1,
                        borderColor: currentStage.highlight === "env" ? "rgba(34, 197, 94, 1)" : "rgba(34, 197, 94, 0.3)",
                        boxShadow: currentStage.highlight === "env" ? "0 0 30px rgba(34, 197, 94, 0.4)" : "0 4px 6px rgba(0,0,0,0.1)",
                    }}
                >
                    <div className="text-6xl mb-4">🌍</div>
                    <div className="text-xl font-bold text-green-600 dark:text-green-400">Environment</div>
                    <div className="text-xs text-slate-500 font-mono mt-1">Dynamics P(s'|s,a)</div>
                </motion.div>
            </div>

            {/* Status Panel */}
            <div className="mt-8 grid grid-cols-2 gap-4">
                <motion.div
                    animate={{ opacity: currentStage.highlight.includes("action") || currentStage.highlight === "state" ? 1 : 0.5 }}
                    className="p-4 bg-purple-50 dark:bg-purple-900/10 rounded-xl border border-purple-100 dark:border-purple-800"
                >
                    <div className="font-bold text-purple-700 dark:text-purple-300 mb-1">Agent Status</div>
                    <div className="text-sm text-slate-600 dark:text-slate-300">{currentStage.agent}</div>
                </motion.div>

                <motion.div
                    animate={{ opacity: currentStage.highlight.includes("env") || currentStage.highlight === "reward" ? 1 : 0.5 }}
                    className="p-4 bg-green-50 dark:bg-green-900/10 rounded-xl border border-green-100 dark:border-green-800"
                >
                    <div className="font-bold text-green-700 dark:text-green-300 mb-1">Environment Status</div>
                    <div className="text-sm text-slate-600 dark:text-slate-300">{currentStage.env}</div>
                </motion.div>
            </div>
        </div>
    );
}
