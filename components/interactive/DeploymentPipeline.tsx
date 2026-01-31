"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function DeploymentPipeline() {
    const [activeStage, setActiveStage] = useState<number | null>(null);

    const stages = [
        {
            id: 1,
            name: "训练 (Training)",
            description: "PyTorch/TensorFlow Training Loop",
            detail: "Policy Gradient / DQN / Actor-Critic",
            icon: "🧠",
            color: "blue"
        },
        {
            id: 2,
            name: "导出 (Export)",
            description: "ONNX / TorchScript / SavedModel",
            detail: "Graph Tracing & Serialization",
            icon: "📦",
            color: "purple"
        },
        {
            id: 3,
            name: "优化 (Optimization)",
            description: "Quantization / Pruning / Fusion",
            detail: "FP32 -> INT8, Operator Fusion",
            icon: "⚡",
            color: "yellow"
        },
        {
            id: 4,
            name: "部署 (Inference)",
            description: "Triton / ONNX Runtime / TFLite",
            detail: "High-throughput Serving",
            icon: "🚀",
            color: "green"
        }
    ];

    return (
        <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-slate-900 dark:to-blue-950 rounded-2xl shadow-xl">
            <div className="text-center mb-8">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    RL 模型部署流水线
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Model Deployment Pipeline: From Training to Production
                </p>
            </div>

            {/* Pipeline Visualization */}
            <div className="relative flex flex-col md:flex-row items-center justify-between gap-4 mb-8">
                {/* Connecting Line (Background) */}
                <div className="absolute left-1/2 top-0 bottom-0 w-2 md:w-full md:h-2 md:left-0 md:top-1/2 bg-gray-200 dark:bg-slate-700 -z-0 transform -translate-x-1/2 md:translate-x-0 md:-translate-y-1/2 rounded-full" />

                {stages.map((stage, idx) => (
                    <motion.div
                        key={stage.id}
                        className="relative z-10 w-full md:w-auto"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.2 }}
                    >
                        <button
                            onClick={() => setActiveStage(stage.id)}
                            className={`w-full md:w-48 p-4 rounded-xl border-2 transition-all duration-300 flex flex-col items-center gap-2 ${activeStage === stage.id
                                    ? `bg-white dark:bg-slate-800 border-${stage.color}-500 shadow-lg scale-105`
                                    : "bg-white dark:bg-slate-800 border-transparent hover:border-gray-300 dark:hover:border-slate-600 shadow-md"
                                }`}
                        >
                            <div className={`text-4xl p-3 rounded-full bg-${stage.color}-100 dark:bg-${stage.color}-900/30`}>
                                {stage.icon}
                            </div>
                            <div className="font-bold text-slate-800 dark:text-slate-100">
                                {stage.name}
                            </div>
                            <div className="text-xs text-slate-500 dark:text-slate-400 text-center">
                                {stage.description}
                            </div>

                            {/* Arrow for next stage (mobile: down, desktop: right) */}
                            {idx < stages.length - 1 && (
                                <div className="hidden md:block absolute -right-6 top-1/2 transform -translate-y-1/2 text-gray-400 z-0">
                                    ➜
                                </div>
                            )}
                        </button>
                    </motion.div>
                ))}
            </div>

            {/* Stage Detail View */}
            <div className="h-48">
                {activeStage && (
                    <motion.div
                        key={activeStage}
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-inner border border-gray-100 dark:border-slate-700"
                    >
                        <div className="flex items-start gap-4">
                            <div className={`text-5xl p-4 rounded-2xl bg-${stages[activeStage - 1].color}-100 dark:bg-${stages[activeStage - 1].color}-900/30`}>
                                {stages[activeStage - 1].icon}
                            </div>
                            <div>
                                <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                                    {stages[activeStage - 1].name}
                                </h4>
                                <p className="text-slate-600 dark:text-slate-400 mb-4">
                                    {stages[activeStage - 1].description}
                                </p>
                                <div className="inline-block px-3 py-1 bg-gray-100 dark:bg-slate-700 rounded text-sm text-slate-600 dark:text-slate-300 font-mono">
                                    {stages[activeStage - 1].detail}
                                </div>

                                <div className="mt-4 text-sm text-slate-500">
                                    {activeStage === 1 && "Start here: Define inputs, outputs, and loss functions."}
                                    {activeStage === 2 && "Convert dynamic graph to static graph for portability."}
                                    {activeStage === 3 && "Reduce model size and latency without significant accuracy loss."}
                                    {activeStage === 4 && "Serve requests with high availability and low latency."}
                                </div>
                            </div>
                        </div>
                    </motion.div>
                )}
                {!activeStage && (
                    <div className="h-full flex items-center justify-center text-slate-400 border-2 border-dashed border-gray-200 dark:border-slate-700 rounded-xl">
                        点击上方阶段查看详情
                    </div>
                )}
            </div>
        </div>
    );
}
