"use client";

import { useState } from "react";
import { motion } from "framer-motion";

interface ExecutionStep {
    id: number;
    phase: "plan" | "execute" | "verify";
    content: string;
    status: "pending" | "running" | "completed" | "failed";
    result?: string;
}

export function MultiStepExecution() {
    const [currentStep, setCurrentStep] = useState(0);
    const [isRunning, setIsRunning] = useState(false);

    const task = "写一个Python脚本并验证其正确性";

    const steps: ExecutionStep[] = [
        { id: 1, phase: "plan", content: "分析需求：创建快速排序函数", status: "completed", result: "需求明确" },
        { id: 2, phase: "plan", content: "设计算法：分治法实现", status: "completed", result: "设计完成" },
        { id: 3, phase: "execute", content: "编写代码", status: "completed", result: "代码已生成" },
        { id: 4, phase: "execute", content: "执行测试用例", status: "running" },
        { id: 5, phase: "verify", content: "验证输出正确性", status: "pending" },
        { id: 6, phase: "verify", content: "性能测试", status: "pending" },
    ];

    const updatedSteps = steps.map((step, idx) => ({
        ...step,
        status: idx < currentStep ? "completed" as const :
            idx === currentStep ? "running" as const : "pending" as const
    }));

    const handleRun = () => {
        if (isRunning) {
            setIsRunning(false);
        } else {
            setIsRunning(true);
            const interval = setInterval(() => {
                setCurrentStep(prev => {
                    if (prev >= steps.length - 1) {
                        setIsRunning(false);
                        clearInterval(interval);
                        return prev;
                    }
                    return prev + 1;
                });
            }, 2000);
        }
    };

    const getPhaseColor = (phase: ExecutionStep["phase"]) => {
        switch (phase) {
            case "plan": return "blue";
            case "execute": return "green";
            case "verify": return "purple";
        }
    };

    const getPhaseIcon = (phase: ExecutionStep["phase"]) => {
        switch (phase) {
            case "plan": return "📋";
            case "execute": return "⚡";
            case "verify": return "✓";
        }
    };

    const getPhaseLabel = (phase: ExecutionStep["phase"]) => {
        switch (phase) {
            case "plan": return "规划";
            case "execute": return "执行";
            case "verify": return "验证";
        }
    };

    const planSteps = updatedSteps.filter(s => s.phase === "plan");
    const executeSteps = updatedSteps.filter(s => s.phase === "execute");
    const verifySteps = updatedSteps.filter(s => s.phase === "verify");

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-slate-900 dark:to-cyan-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    多步执行流程
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Plan → Execute → Verify 三阶段流程
                </p>
            </div>

            {/* 任务 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-3 text-slate-800 dark:text-slate-100">任务</h4>
                <div className="bg-cyan-50 dark:bg-cyan-900/20 p-4 rounded-lg border-2 border-cyan-300 dark:border-cyan-700">
                    <div className="text-slate-800 dark:text-slate-100">{task}</div>
                </div>
            </div>

            {/* 控制 */}
            <div className="flex items-center justify-between mb-6">
                <div className="flex gap-3">
                    <button
                        onClick={handleRun}
                        className={`px-4 py-2 rounded-lg font-semibold transition ${isRunning ? "bg-orange-500 text-white" : "bg-cyan-600 text-white hover:bg-cyan-700"
                            }`}
                    >
                        {isRunning ? "⏸ 暂停" : "▶ 开始执行"}
                    </button>
                    <button
                        onClick={() => { setIsRunning(false); setCurrentStep(0); }}
                        className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-semibold hover:bg-gray-300 dark:hover:bg-gray-600 transition"
                    >
                        🔄 重置
                    </button>
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400">
                    进度: <strong className="text-cyan-600 dark:text-cyan-400">{currentStep + 1}/{steps.length}</strong>
                </div>
            </div>

            {/* 三阶段可视化 */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                {[
                    { phase: "plan" as const, steps: planSteps },
                    { phase: "execute" as const, steps: executeSteps },
                    { phase: "verify" as const, steps: verifySteps }
                ].map(({ phase, steps: phaseSteps }) => {
                    const completedCount = phaseSteps.filter(s => s.status === "completed").length;
                    const progress = (completedCount / phaseSteps.length) * 100;

                    return (
                        <div
                            key={phase}
                            className={`bg-${getPhaseColor(phase)}-50 dark:bg-${getPhaseColor(phase)}-900/20 p-6 rounded-xl border-2 border-${getPhaseColor(phase)}-500`}
                        >
                            <div className="text-3xl text-center mb-2">{getPhaseIcon(phase)}</div>
                            <div className={`text-center font-bold text-${getPhaseColor(phase)}-700 dark:text-${getPhaseColor(phase)}-400 mb-3`}>
                                {getPhaseLabel(phase)}
                            </div>
                            <div className="text-center text-sm text-slate-600 dark:text-slate-400 mb-2">
                                {completedCount}/{phaseSteps.length} 完成
                            </div>
                            <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
                                <motion.div
                                    className={`h-full bg-${getPhaseColor(phase)}-600 rounded-full`}
                                    animate={{ width: `${progress}%` }}
                                    transition={{ duration: 0.5 }}
                                />
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* 详细步骤 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">执行详情</h4>

                <div className="space-y-3">
                    {updatedSteps.map((step, idx) => (
                        <motion.div
                            key={step.id}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.1 }}
                            className={`p-4 rounded-lg border-2 ${step.status === "completed"
                                    ? "border-green-500 bg-green-50 dark:bg-green-900/20"
                                    : step.status === "running"
                                        ? `border-${getPhaseColor(step.phase)}-500 bg-${getPhaseColor(step.phase)}-50 dark:bg-${getPhaseColor(step.phase)}-900/20`
                                        : "border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-800"
                                }`}
                        >
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3 flex-1">
                                    <div className="text-2xl">{getPhaseIcon(step.phase)}</div>
                                    <div className="flex-1">
                                        <div className="flex items-center gap-2 mb-1">
                                            <span className={`text-xs px-2 py-1 rounded-full bg-${getPhaseColor(step.phase)}-100 dark:bg-${getPhaseColor(step.phase)}-900/30 text-${getPhaseColor(step.phase)}-700 dark:text-${getPhaseColor(step.phase)}-400`}>
                                                {getPhaseLabel(step.phase)}
                                            </span>
                                            <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                                                步骤 {step.id}
                                            </span>
                                        </div>
                                        <div className="text-slate-800 dark:text-slate-100">
                                            {step.content}
                                        </div>
                                        {step.result && step.status === "completed" && (
                                            <div className="text-sm text-green-600 dark:text-green-400 mt-2">
                                                ✓ {step.result}
                                            </div>
                                        )}
                                    </div>
                                </div>

                                <div>
                                    {step.status === "completed" && (
                                        <div className="w-8 h-8 rounded-full bg-green-600 text-white flex items-center justify-center font-bold">
                                            ✓
                                        </div>
                                    )}
                                    {step.status === "running" && (
                                        <motion.div
                                            animate={{ rotate: 360 }}
                                            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                            className={`w-8 h-8 rounded-full border-4 border-${getPhaseColor(step.phase)}-600 border-t-transparent`}
                                        />
                                    )}
                                    {step.status === "pending" && (
                                        <div className="w-8 h-8 rounded-full bg-gray-300 dark:bg-gray-700 text-gray-600 dark:text-gray-400 flex items-center justify-center">
                                            ○
                                        </div>
                                    )}
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>

            <div className="mt-6 bg-cyan-100 dark:bg-cyan-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                💡 <strong>Plan-and-Execute</strong>: 先规划完整方案，再逐步执行并验证，提升任务完成质量
            </div>
        </div>
    );
}
