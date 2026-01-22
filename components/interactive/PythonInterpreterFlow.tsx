"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const steps = [
    {
        id: "source",
        title: "Source Code",
        description: "Human-readable Python code",
        content: `print("Hello")\na = 1 + 2`,
        color: "bg-blue-500",
        icon: "📄"
    },
    {
        id: "lexer",
        title: "Lexer (Tokenizer)",
        description: "Breaks code into tokens",
        content: `[NAME: print]\n[LPAR: (]\n[STRING: "Hello"]\n[RPAR: )]`,
        color: "bg-indigo-500",
        icon: "✂️"
    },
    {
        id: "parser",
        title: "Parser (AST)",
        description: "Builds Abstract Syntax Tree",
        content: `Module(\n  body=[\n    Expr(value=Call(...))\n    Assign(targets=...)\n  ]\n)`,
        color: "bg-purple-500",
        icon: "🌳"
    },
    {
        id: "compiler",
        title: "Compiler",
        description: "Generates Bytecode",
        content: `LOAD_NAME 0 (print)\nLOAD_CONST 0 ("Hello")\nCALL_FUNCTION 1\n...`,
        color: "bg-pink-500",
        icon: "⚙️"
    },
    {
        id: "vm",
        title: "PVM (Interpreter)",
        description: "Executes bytecode",
        content: `> Hello\n> Process finished`,
        color: "bg-green-500",
        icon: "🖥️"
    }
];

export function PythonInterpreterFlow() {
    const [currentStep, setCurrentStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isPlaying) {
            interval = setInterval(() => {
                setCurrentStep((prev) => {
                    if (prev >= steps.length - 1) {
                        setIsPlaying(false);
                        return prev;
                    }
                    return prev + 1;
                });
            }, 2000);
        }
        return () => clearInterval(interval);
    }, [isPlaying]);

    const handleStepClick = (index: number) => {
        setCurrentStep(index);
        setIsPlaying(false);
    };

    return (
        <div className="w-full max-w-4xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur-sm rounded-xl border border-border-subtle shadow-lg my-8">
            <h3 className="text-xl font-bold text-center mb-8 bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent">
                CPython Execution Pipeline
            </h3>

            {/* Progress Bar */}
            <div className="relative h-2 bg-gray-200 dark:bg-gray-700 rounded-full mb-12 mx-4">
                <motion.div
                    className="absolute top-0 left-0 h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                    initial={{ width: "0%" }}
                    animate={{ width: `${(currentStep / (steps.length - 1)) * 100}%` }}
                    transition={{ duration: 0.5 }}
                />

                {/* Steps Dots */}
                <div className="absolute top-1/2 left-0 w-full -translate-y-1/2 flex justify-between px-0">
                    {steps.map((step, index) => (
                        <button
                            key={step.id}
                            onClick={() => handleStepClick(index)}
                            className={`relative w-8 h-8 rounded-full flex items-center justify-center transition-all duration-300 z-10 ${index <= currentStep
                                    ? "bg-gradient-to-r from-blue-500 to-purple-500 text-white scale-110 shadow-lg"
                                    : "bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400"
                                }`}
                        >
                            <span className="text-xs">{step.icon}</span>
                            {/* Step Label */}
                            <span className={`absolute -bottom-8 text-xs font-semibold whitespace-nowrap transition-colors duration-300 ${index === currentStep ? "text-blue-500" : "text-text-tertiary"
                                }`}>
                                {step.title}
                            </span>
                        </button>
                    ))}
                </div>
            </div>

            {/* Content Area */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8 min-h-[300px]">
                {/* Source/Input View */}
                <div className="bg-bg-base rounded-xl p-6 border border-border-subtle relative overflow-hidden">
                    <div className="text-sm font-bold text-text-secondary mb-4 uppercase tracking-wider">
                        Current Stage: {steps[currentStep].title}
                    </div>
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={currentStep}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                            className="relative z-10"
                        >
                            <pre className="font-mono text-sm overflow-x-auto p-4 bg-bg-elevated/50 rounded-lg border border-border-subtle">
                                <code className="language-python">
                                    {steps[currentStep].content}
                                </code>
                            </pre>
                            <p className="mt-4 text-text-secondary">
                                {steps[currentStep].description}
                            </p>
                        </motion.div>
                    </AnimatePresence>

                    {/* Background Glow */}
                    <div className={`absolute -right-10 -bottom-10 w-32 h-32 ${steps[currentStep].color} opacity-10 blur-3xl rounded-full transition-colors duration-500`} />
                </div>

                {/* Info Panel / Controller */}
                <div className="flex flex-col justify-center items-center text-center space-y-6">
                    <div className="text-6xl animate-bounce-slow">
                        {steps[currentStep].icon}
                    </div>

                    <div className="space-y-2">
                        <h4 className="text-2xl font-bold text-text-primary">
                            {steps[currentStep].title}
                        </h4>
                        <p className="text-text-secondary max-w-xs mx-auto">
                            {steps[currentStep].description}
                        </p>
                    </div>

                    <div className="flex gap-4 mt-4">
                        <button
                            onClick={() => {
                                setCurrentStep(0);
                                setIsPlaying(true);
                            }}
                            className="px-6 py-2 bg-accent-primary text-white rounded-lg hover:bg-accent-primary/90 transition-colors shadow-lg flex items-center gap-2"
                        >
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            {currentStep === steps.length - 1 ? "Replay Simulation" : "Start Simulation"}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
