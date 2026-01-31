"use client";

import { useState } from "react";
import { motion } from "framer-motion";

type StepType = "thought" | "action" | "observation";

interface Step {
    id: number;
    type: StepType;
    content: string;
}

export function ReActFramework() {
    const [currentStep, setCurrentStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    const problem = "2024年诺贝尔物理学奖得主是谁？他们的主要贡献是什么？";

    const steps: Step[] = [
        { id: 1, type: "thought", content: "我需要找到2024年诺贝尔物理学奖的获得者信息" },
        { id: 2, type: "action", content: "Search[2024 Nobel Prize Physics winner]" },
        { id: 3, type: "observation", content: "John Hopfield和Geoffrey Hinton获得2024年诺贝尔物理学奖" },
        { id: 4, type: "thought", content: "已找到获得者，现在需要了解他们的主要贡献" },
        { id: 5, type: "action", content: "Search[John Hopfield Geoffrey Hinton contributions]" },
        { id: 6, type: "observation", content: "他们因在人工神经网络方面的开创性工作获奖，Hopfield网络和深度学习的奠基工作" },
        { id: 7, type: "thought", content: "已收集足够信息，可以回答问题了" },
        { id: 8, type: "action", content: "Finish[2024年诺贝尔物理学奖得主是John Hopfield和Geoffrey Hinton，他们的主要贡献是人工神经网络的开创性工作]" },
    ];

    const handlePlay = () => {
        if (isPlaying) {
            setIsPlaying(false);
        } else {
            setIsPlaying(true);
            const interval = setInterval(() => {
                setCurrentStep(prev => {
                    if (prev >= steps.length - 1) {
                        setIsPlaying(false);
                        clearInterval(interval);
                        return prev;
                    }
                    return prev + 1;
                });
            }, 1500);
        }
    };

    const getStepColor = (type: StepType) => {
        switch (type) {
            case "thought": return "blue";
            case "action": return "green";
            case "observation": return "purple";
        }
    };

    const getStepIcon = (type: StepType) => {
        switch (type) {
            case "thought": return "💭";
            case "action": return "⚡";
            case "observation": return "👁️";
        }
    };

    const getStepLabel = (type: StepType) => {
        switch (type) {
            case "thought": return "Thought（思考）";
            case "action": return "Action（行动）";
            case "observation": return "Observation（观察）";
        }
    };

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    ReAct 框架
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Reasoning + Acting：思考与行动的协同
                </p>
            </div>

            {/* 问题 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-3 text-slate-800 dark:text-slate-100">问题</h4>
                <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg border-2 border-indigo-300 dark:border-indigo-700">
                    <div className="text-slate-800 dark:text-slate-100">{problem}</div>
                </div>
            </div>

            {/* 控制 */}
            <div className="flex items-center justify-between mb-6">
                <div className="flex gap-3">
                    <button
                        onClick={handlePlay}
                        className={`px-4 py-2 rounded-lg font-semibold transition ${isPlaying ? "bg-orange-500 text-white" : "bg-indigo-600 text-white hover:bg-indigo-700"
                            }`}
                    >
                        {isPlaying ? "⏸ 暂停" : "▶ 播放"}
                    </button>
                    <button
                        onClick={() => { setIsPlaying(false); setCurrentStep(0); }}
                        className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-semibold hover:bg-gray-300 dark:hover:bg-gray-600 transition"
                    >
                        🔄 重置
                    </button>
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400">
                    步骤: <strong className="text-indigo-600 dark:text-indigo-400">{currentStep + 1}/{steps.length}</strong>
                </div>
            </div>

            {/* 流程图示 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">ReAct 循环</h4>

                <div className="flex items-center justify-center gap-4 flex-wrap">
                    {["thought", "action", "observation"].map((type, idx) => (
                        <div key={type} className="flex items-center gap-4">
                            <div className={`p-4 rounded-xl border-2 border-${getStepColor(type as StepType)}-500 bg-${getStepColor(type as StepType)}-50 dark:bg-${getStepColor(type as StepType)}-900/20`}>
                                <div className="text-3xl text-center mb-2">{getStepIcon(type as StepType)}</div>
                                <div className={`text-sm font-semibold text-center text-${getStepColor(type as StepType)}-700 dark:text-${getStepColor(type as StepType)}-400`}>
                                    {getStepLabel(type as StepType)}
                                </div>
                            </div>
                            {idx < 2 && <div className="text-3xl text-indigo-500">→</div>}
                        </div>
                    ))}
                    <div className="text-3xl text-indigo-500">↻</div>
                </div>
            </div>

            {/* 步骤执行 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">执行过程</h4>

                <div className="space-y-3">
                    {steps.slice(0, currentStep + 1).map((step, idx) => (
                        <motion.div
                            key={step.id}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.5 }}
                            className={`p-4 rounded-lg border-2 border-${getStepColor(step.type)}-500 bg-${getStepColor(step.type)}-50 dark:bg-${getStepColor(step.type)}-900/20`}
                        >
                            <div className="flex items-start gap-3">
                                <div className={`text-2xl`}>{getStepIcon(step.type)}</div>
                                <div className="flex-1">
                                    <div className={`text-sm font-semibold text-${getStepColor(step.type)}-700 dark:text-${getStepColor(step.type)}-400 mb-1`}>
                                        {getStepLabel(step.type)} {Math.ceil((idx + 1) / 3)}
                                    </div>
                                    <div className="text-slate-800 dark:text-slate-100">
                                        {step.content}
                                    </div>
                                </div>
                                {idx === currentStep && (
                                    <motion.div
                                        animate={{ opacity: [1, 0.3, 1] }}
                                        transition={{ duration: 1, repeat: Infinity }}
                                        className="text-2xl"
                                    >
                                        ⬤
                                    </motion.div>
                                )}
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>

            {/* 对比说明 */}
            <div className="mt-6 grid grid-cols-3 gap-4">
                <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border border-red-300 dark:border-red-700">
                    <h5 className="font-semibold text-red-700 dark:text-red-400 mb-2">❌ CoT Only</h5>
                    <p className="text-xs text-slate-600 dark:text-slate-400">只思考，无法与环境交互获取实时信息</p>
                </div>

                <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border border-orange-300 dark:border-orange-700">
                    <h5 className="font-semibold text-orange-700 dark:text-orange-400 mb-2">⚠️ Action Only</h5>
                    <p className="text-xs text-slate-600 dark:text-slate-400">盲目行动，缺乏推理链和错误修正</p>
                </div>

                <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-300 dark:border-green-700">
                    <h5 className="font-semibold text-green-700 dark:text-green-400 mb-2">✅ ReAct</h5>
                    <p className="text-xs text-slate-600 dark:text-slate-400">结合推理与行动，可解释且能自我修正</p>
                </div>
            </div>

            <div className="mt-6 bg-indigo-100 dark:bg-indigo-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                💡 <strong>核心优势</strong>: ReAct通过显式推理链提升可靠性，通过实时反馈支持动态调整
            </div>
        </div>
    );
}
