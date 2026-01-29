"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

export function AgentEnvironmentLoop() {
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [state, setState] = useState("S₀");
    const [action, setAction] = useState("—");
    const [reward, setReward] = useState("—");
    const [history, setHistory] = useState<Array<{ state: string, action: string, reward: string }>>([]);

    const states = ["S₀", "S₁", "S₂", "S₃", "S₄"];
    const actions = ["↑", "→", "↓", "←"];
    const rewards = ["+1", "+5", "-1", "+10", "0"];

    useEffect(() => {
        if (!isPlaying) return;

        const interval = setInterval(() => {
            setStep(s => {
                const newStep = (s + 1) % 4;

                if (newStep === 0) {
                    // 新的循环
                    const newState = states[Math.floor(Math.random() * states.length)];
                    const newAction = actions[Math.floor(Math.random() * actions.length)];
                    const newReward = rewards[Math.floor(Math.random() * rewards.length)];

                    setState(newState);
                    setAction(newAction);
                    setReward(newReward);

                    setHistory(prev => [...prev.slice(-4), {
                        state: newState,
                        action: newAction,
                        reward: newReward
                    }]);
                }

                return newStep;
            });
        }, 1000);

        return () => clearInterval(interval);
    }, [isPlaying]);

    const phases = [
        { name: "观察状态", desc: "Agent 接收环境状态", color: "#6366f1" },
        { name: "选择动作", desc: "Agent 根据策略选择动作", color: "#8b5cf6" },
        { name: "执行动作", desc: "环境执行动作并转移状态", color: "#ec4899" },
        { name: "获得反馈", desc: "Agent 接收奖励和新状态", color: "#10b981" },
    ];

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Agent-Environment 交互循环
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    强化学习的核心机制
                </p>
            </div>

            {/* 控制按钮 */}
            <div className="flex justify-center gap-4 mb-8">
                <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="px-6 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-semibold transition-colors"
                >
                    {isPlaying ? "⏸ 暂停" : "▶ 播放"}
                </button>
                <button
                    onClick={() => {
                        setStep(0);
                        setIsPlaying(false);
                        setHistory([]);
                    }}
                    className="px-6 py-2 rounded-lg bg-slate-600 hover:bg-slate-700 text-white font-semibold transition-colors"
                >
                    🔄 重置
                </button>
            </div>

            {/* 主循环可视化 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                {/* Agent */}
                <motion.div
                    className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-2"
                    style={{
                        borderColor: step === 1 ? "#6366f1" : "#e2e8f0"
                    }}
                    animate={{
                        scale: step === 1 ? 1.05 : 1,
                    }}
                >
                    <div className="text-center">
                        <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-3xl font-bold">
                            🤖
                        </div>
                        <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                            Agent（智能体）
                        </h4>
                        <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                                <span className="text-slate-600 dark:text-slate-400">当前状态:</span>
                                <span className="font-mono font-bold text-indigo-600">{state}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-600 dark:text-slate-400">选择动作:</span>
                                <span className="font-mono font-bold text-purple-600">{action}</span>
                            </div>
                        </div>
                    </div>
                </motion.div>

                {/* Environment */}
                <motion.div
                    className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-2"
                    style={{
                        borderColor: step === 2 || step === 3 ? "#10b981" : "#e2e8f0"
                    }}
                    animate={{
                        scale: step === 2 || step === 3 ? 1.05 : 1,
                    }}
                >
                    <div className="text-center">
                        <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center text-white text-3xl font-bold">
                            🌍
                        </div>
                        <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                            Environment（环境）
                        </h4>
                        <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                                <span className="text-slate-600 dark:text-slate-400">下一状态:</span>
                                <span className="font-mono font-bold text-green-600">{state}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-600 dark:text-slate-400">奖励:</span>
                                <span className="font-mono font-bold text-emerald-600">{reward}</span>
                            </div>
                        </div>
                    </div>
                </motion.div>
            </div>

            {/* 交互流程 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4 text-center">
                    交互流程（第 {Math.floor(history.length / 4) + 1} 轮）
                </h4>
                <div className="grid grid-cols-4 gap-2">
                    {phases.map((phase, idx) => (
                        <motion.div
                            key={idx}
                            className="p-4 rounded-lg text-center"
                            style={{
                                backgroundColor: step === idx ? `${phase.color}20` : "#f8fafc",
                                borderWidth: 2,
                                borderColor: step === idx ? phase.color : "#e2e8f0",
                            }}
                            animate={{
                                scale: step === idx ? 1.05 : 1,
                            }}
                        >
                            <div
                                className="w-8 h-8 mx-auto mb-2 rounded-full flex items-center justify-center text-white font-bold text-sm"
                                style={{ backgroundColor: phase.color }}
                            >
                                {idx + 1}
                            </div>
                            <div className="text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                                {phase.name}
                            </div>
                            <div className="text-xs text-slate-500 dark:text-slate-400">
                                {phase.desc}
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>

            {/* 历史记录 */}
            {history.length > 0 && (
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">
                        交互历史
                    </h4>
                    <div className="space-y-2">
                        {history.map((item, idx) => (
                            <motion.div
                                key={idx}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                className="flex items-center gap-4 p-3 bg-slate-50 dark:bg-slate-700 rounded-lg text-sm"
                            >
                                <span className="font-mono text-slate-600 dark:text-slate-400">
                                    t={idx}
                                </span>
                                <span className="font-mono">
                                    状态: <span className="text-indigo-600 font-bold">{item.state}</span>
                                </span>
                                <span className="font-mono">
                                    动作: <span className="text-purple-600 font-bold">{item.action}</span>
                                </span>
                                <span className="font-mono">
                                    奖励: <span className="text-emerald-600 font-bold">{item.reward}</span>
                                </span>
                            </motion.div>
                        ))}
                    </div>
                </div>
            )}

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 提示：这个循环会一直重复，直到 episode 结束（terminated 或 truncated）
            </div>
        </div>
    );
}
