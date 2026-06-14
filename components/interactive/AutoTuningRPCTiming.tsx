"use client";

import { useState, useEffect } from "react";

const phases = [
    { id: "sample", label: "采样配置", desc: "从搜索空间采样一组调度参数", icon: "🎲", duration: 500, color: "bg-blue-500" },
    { id: "compile", label: "编译内核", desc: "根据配置编译生成内核代码", icon: "⚙️", duration: 1500, color: "bg-indigo-500" },
    { id: "transfer", label: "传输到设备", desc: "通过 RPC 将编译结果发送到远程设备", icon: "📡", duration: 800, color: "bg-purple-500" },
    { id: "execute", label: "远程执行", desc: "在目标设备上运行基准测试", icon: "🚀", duration: 2000, color: "bg-pink-500" },
    { id: "collect", label: "收集结果", desc: "收集执行时间并更新成本模型", icon: "📊", duration: 300, color: "bg-rose-500" },
];

export function AutoTuningRPCTiming() {
    const [isRunning, setIsRunning] = useState(false);
    const [currentPhase, setCurrentPhase] = useState(-1);
    const [completed, setCompleted] = useState<number[]>([]);
    const [trial, setTrial] = useState(0);

    useEffect(() => {
        if (!isRunning) return;
        if (currentPhase >= phases.length) {
            setTrial((t) => t + 1);
            if (trial >= 4) {
                setIsRunning(false);
                setCurrentPhase(-1);
                return;
            }
            setCurrentPhase(0);
            setCompleted([]);
            return;
        }

        const timer = setTimeout(() => {
            setCompleted((prev) => [...prev, currentPhase]);
            setCurrentPhase((p) => p + 1);
        }, phases[currentPhase]?.duration || 1000);

        return () => clearTimeout(timer);
    }, [isRunning, currentPhase, trial]);

    const startTuning = () => {
        setIsRunning(true);
        setCurrentPhase(0);
        setCompleted([]);
        setTrial(0);
    };

    const resetTuning = () => {
        setIsRunning(false);
        setCurrentPhase(-1);
        setCompleted([]);
        setTrial(0);
    };

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                AutoTVM RPC 调优时序
            </h2>
            <p className="text-center text-sm text-slate-500 dark:text-slate-400 mb-6">
                每轮调优: 采样 → 编译 → 远程执行 → 收集
            </p>

            {/* Control buttons */}
            <div className="flex justify-center gap-4 mb-6">
                <button
                    onClick={startTuning}
                    disabled={isRunning}
                    className={`px-6 py-2 rounded-full text-sm font-bold transition-all ${
                        isRunning
                            ? "bg-slate-200 dark:bg-slate-700 text-slate-400 cursor-not-allowed"
                            : "bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-lg hover:shadow-xl"
                    }`}
                >
                    {isRunning ? "调优中..." : "开始调优"}
                </button>
                <button
                    onClick={resetTuning}
                    className="px-6 py-2 rounded-full text-sm font-bold bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:border-indigo-300"
                >
                    重置
                </button>
            </div>

            {/* Trial counter */}
            {isRunning && (
                <div className="text-center mb-4">
                    <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-indigo-100 dark:bg-indigo-900/40 text-sm font-medium text-indigo-700 dark:text-indigo-300">
                        Trial {trial + 1} / 5
                        <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
                    </span>
                </div>
            )}

            {/* Timing diagram */}
            <div className="bg-white dark:bg-slate-800/80 rounded-xl p-5 border border-slate-200 dark:border-slate-700">
                <div className="space-y-3">
                    {phases.map((p, i) => {
                        const isDone = completed.includes(i);
                        const isCurrent = currentPhase === i;
                        const widthPercent = (p.duration / 5000) * 100;

                        return (
                            <div key={p.id} className="flex items-center gap-3">
                                <div className="w-24 text-right text-xs font-medium text-slate-600 dark:text-slate-300 shrink-0">
                                    {p.icon} {p.label}
                                </div>
                                <div className="flex-1 h-8 bg-slate-100 dark:bg-slate-900 rounded-lg overflow-hidden relative">
                                    <div
                                        className={`h-full rounded-lg transition-all duration-500 ${
                                            isDone ? "bg-gradient-to-r from-green-400 to-emerald-500" : isCurrent ? p.color + " animate-pulse" : "bg-transparent"
                                        }`}
                                        style={{ width: isDone || isCurrent ? `${widthPercent}%` : "0%" }}
                                    />
                                    {isCurrent && (
                                        <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] font-mono text-white">
                                            {p.duration}ms
                                        </span>
                                    )}
                                </div>
                                <div className="w-8 text-xs text-slate-400">
                                    {isDone ? "✅" : isCurrent ? "⏳" : "⏸"}
                                </div>
                            </div>
                        );
                    })}
                </div>

                {/* Arrow labels */}
                <div className="flex justify-between mt-4 px-28 text-[10px] text-slate-400">
                    <span>← 端侧 (Host) →</span>
                    <span>← 远程 (Device) →</span>
                </div>
            </div>

            {/* Stats */}
            <div className="mt-4 grid grid-cols-3 gap-3">
                {[
                    { label: "已完成试验", value: `${trial}/5` },
                    { label: "平均耗时", value: "~5.1s/trial" },
                    { label: "最优配置", value: trial > 0 ? "tile=32, vec=8" : "待搜索" },
                ].map((s, i) => (
                    <div key={i} className="bg-white/60 dark:bg-slate-800/60 rounded-lg p-3 border border-slate-200 dark:border-slate-700 text-center">
                        <div className="text-xs text-slate-500 dark:text-slate-400">{s.label}</div>
                        <div className="text-sm font-bold text-indigo-600 dark:text-indigo-400 mt-1">{s.value}</div>
                    </div>
                ))}
            </div>
        </div>
    );
}
