"use client";

import { useState } from "react";
import { motion } from "framer-motion";

type Step = "fetch" | "decode" | "execute" | "writeback";

const steps: { id: Step; label: string; description: string }[] = [
    {
        id: "fetch",
        label: "取指 (Fetch)",
        description: "PC → MAR → 从存储器读取指令 → MDR → IR",
    },
    {
        id: "decode",
        label: "译码 (Decode)",
        description: "指令译码器分析IR中的指令",
    },
    {
        id: "execute",
        label: "执行 (Execute)",
        description: "ALU执行算术/逻辑运算",
    },
    {
        id: "writeback",
        label: "回写 (Write Back)",
        description: "将结果写回存储器或寄存器",
    },
];

export function InstructionCycleSimulator() {
    const [currentStep, setCurrentStep] = useState<number>(0);
    const [isPlaying, setIsPlaying] = useState(false);

    const handleNext = () => {
        setCurrentStep((prev) => (prev + 1) % steps.length);
    };

    const handlePrev = () => {
        setCurrentStep((prev) => (prev === 0 ? steps.length - 1 : prev - 1));
    };

    const handleReset = () => {
        setCurrentStep(0);
        setIsPlaying(false);
    };

    return (
        <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
            <h3 className="text-xl font-semibold mb-4 text-text-primary">
                指令周期动态演示
            </h3>

            {/* SVG Diagram */}
            <div className="relative w-full h-96 bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden mb-6">
                <svg
                    viewBox="0 0 800 400"
                    className="w-full h-full"
                    xmlns="http://www.w3.org/2000/svg"
                >
                    {/* CPU Box */}
                    <motion.rect
                        x="50"
                        y="50"
                        width="300"
                        height="300"
                        fill="none"
                        stroke={currentStep === 1 || currentStep === 2 ? "#667eea" : "#cbd5e1"}
                        strokeWidth="3"
                        rx="10"
                        animate={{
                            stroke: currentStep === 1 || currentStep === 2 ? "#667eea" : "#cbd5e1",
                        }}
                    />
                    <text x="200" y="35" textAnchor="middle" className="fill-text-primary font-semibold">
                        CPU
                    </text>

                    {/* Control Unit */}
                    <motion.rect
                        x="70"
                        y="70"
                        width="120"
                        height="100"
                        fill="#f1f5f9"
                        stroke="#64748b"
                        strokeWidth="2"
                        rx="6"
                        animate={{
                            fill: currentStep === 0 || currentStep === 1 ? "#dbeafe" : "#f1f5f9",
                        }}
                    />
                    <text x="130" y="95" textAnchor="middle" className="text-sm fill-text-primary">
                        控制器
                    </text>
                    <text x="130" y="115" textAnchor="middle" className="text-xs fill-text-secondary">
                        PC
                    </text>
                    <text x="130" y="130" textAnchor="middle" className="text-xs fill-text-secondary">
                        IR
                    </text>
                    <text x="130" y="145" textAnchor="middle" className="text-xs fill-text-secondary">
                        ID
                    </text>

                    {/* ALU */}
                    <motion.rect
                        x="210"
                        y="70"
                        width="120"
                        height="100"
                        fill="#f1f5f9"
                        stroke="#64748b"
                        strokeWidth="2"
                        rx="6"
                        animate={{
                            fill: currentStep === 2 ? "#dbeafe" : "#f1f5f9",
                        }}
                    />
                    <text x="270" y="95" textAnchor="middle" className="text-sm fill-text-primary">
                        运算器
                    </text>
                    <text x="270" y="115" textAnchor="middle" className="text-xs fill-text-secondary">
                        ALU
                    </text>
                    <text x="270" y="130" textAnchor="middle" className="text-xs fill-text-secondary">
                        ACC
                    </text>

                    {/* Registers */}
                    <motion.rect
                        x="70"
                        y="200"
                        width="260"
                        height="130"
                        fill="#fef3c7"
                        stroke="#f59e0b"
                        strokeWidth="2"
                        rx="6"
                        animate={{
                            fill: currentStep === 3 ? "#fde68a" : "#fef3c7",
                        }}
                    />
                    <text x="200" y="225" textAnchor="middle" className="text-sm fill-text-primary font-semibold">
                        寄存器组
                    </text>
                    <text x="120" y="250" className="text-xs fill-text-secondary">
                        MAR
                    </text>
                    <text x="200" y="250" className="text-xs fill-text-secondary">
                        MDR
                    </text>
                    <text x="280" y="250" className="text-xs fill-text-secondary">
                        PSW
                    </text>

                    {/* Memory */}
                    <motion.rect
                        x="450"
                        y="100"
                        width="300"
                        height="200"
                        fill="#f1f5f9"
                        stroke="#64748b"
                        strokeWidth="3"
                        rx="10"
                        animate={{
                            stroke: currentStep === 0 || currentStep === 3 ? "#667eea" : "#64748b",
                        }}
                    />
                    <text x="600" y="85" textAnchor="middle" className="fill-text-primary font-semibold">
                        主存储器 (Memory)
                    </text>

                    {/* Memory cells */}
                    {[0, 1, 2, 3, 4].map((i) => (
                        <g key={i}>
                            <rect
                                x="470"
                                y={120 + i * 30}
                                width="260"
                                height="25"
                                fill={i === 0 && currentStep === 0 ? "#dbeafe" : "#ffffff"}
                                stroke="#cbd5e1"
                                strokeWidth="1"
                            />
                            <text
                                x="490"
                                y={135 + i * 30}
                                className="text-xs fill-text-secondary"
                            >
                                {`地址 ${i}`}
                            </text>
                            <text
                                x="650"
                                y={135 + i * 30}
                                className="text-xs fill-text-tertiary"
                            >
                                {i === 0 ? "指令" : "数据"}
                            </text>
                        </g>
                    ))}

                    {/* Data Flow Arrow - Animated */}
                    {currentStep === 0 && (
                        <motion.g
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ duration: 0.5 }}
                        >
                            <defs>
                                <marker
                                    id="arrowhead"
                                    markerWidth="10"
                                    markerHeight="10"
                                    refX="9"
                                    refY="3"
                                    orient="auto"
                                >
                                    <polygon points="0 0, 10 3, 0 6" fill="#667eea" />
                                </marker>
                            </defs>
                            <motion.path
                                d="M 450 150 L 350 150"
                                stroke="#667eea"
                                strokeWidth="3"
                                fill="none"
                                markerEnd="url(#arrowhead)"
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                                transition={{ duration: 1, repeat: Infinity }}
                            />
                            <text x="400" y="140" className="text-xs fill-accent-primary font-semibold">
                                读取指令
                            </text>
                        </motion.g>
                    )}

                    {currentStep === 3 && (
                        <motion.g
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ duration: 0.5 }}
                        >
                            <motion.path
                                d="M 350 200 L 450 200"
                                stroke="#10b981"
                                strokeWidth="3"
                                fill="none"
                                markerEnd="url(#arrowhead2)"
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                                transition={{ duration: 1, repeat: Infinity }}
                            />
                            <defs>
                                <marker
                                    id="arrowhead2"
                                    markerWidth="10"
                                    markerHeight="10"
                                    refX="9"
                                    refY="3"
                                    orient="auto"
                                >
                                    <polygon points="0 0, 10 3, 0 6" fill="#10b981" />
                                </marker>
                            </defs>
                            <text x="400" y="190" className="text-xs fill-green-600 font-semibold">
                                写回结果
                            </text>
                        </motion.g>
                    )}
                </svg>
            </div>

            {/* Current Step Info */}
            <div className="mb-6 p-4 bg-accent-primary/10 rounded-lg border-l-4 border-accent-primary">
                <div className="font-semibold text-accent-primary mb-1">
                    {steps[currentStep].label}
                </div>
                <div className="text-sm text-text-secondary">
                    {steps[currentStep].description}
                </div>
            </div>

            {/* Progress Indicator */}
            <div className="flex gap-2 mb-6">
                {steps.map((step, index) => (
                    <div
                        key={step.id}
                        className={`flex-1 h-2 rounded-full transition-all ${index === currentStep
                                ? "bg-accent-primary"
                                : index < currentStep
                                    ? "bg-accent-primary/50"
                                    : "bg-gray-300 dark:bg-gray-700"
                            }`}
                    />
                ))}
            </div>

            {/* Controls */}
            <div className="flex gap-3 justify-center">
                <button
                    onClick={handlePrev}
                    className="px-4 py-2 rounded-md bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 text-text-primary transition"
                >
                    ← 上一步
                </button>
                <button
                    onClick={handleReset}
                    className="px-4 py-2 rounded-md bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 text-text-primary transition"
                >
                    重置
                </button>
                <button
                    onClick={handleNext}
                    className="px-4 py-2 rounded-md bg-accent-primary hover:bg-accent-secondary text-white transition"
                >
                    下一步 →
                </button>
            </div>

            {/* Step List */}
            <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-3">
                {steps.map((step, index) => (
                    <button
                        key={step.id}
                        onClick={() => setCurrentStep(index)}
                        className={`p-3 rounded-lg text-sm text-left transition ${index === currentStep
                                ? "bg-accent-primary text-white"
                                : "bg-gray-100 dark:bg-gray-800 text-text-secondary hover:bg-gray-200 dark:hover:bg-gray-700"
                            }`}
                    >
                        <div className="font-semibold">步骤 {index + 1}</div>
                        <div className="text-xs mt-1 opacity-80">{step.label}</div>
                    </button>
                ))}
            </div>
        </div>
    );
}
