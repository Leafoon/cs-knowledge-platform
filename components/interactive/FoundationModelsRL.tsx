"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Scenario: Robot Assistant making coffee
const scenario = [
    { role: "user", content: "Robot, please make me a cup of coffee." },
    {
        role: "assistant",
        type: "thought",
        content: "Goal: Make coffee. \nPlan:\n1. Find mug.\n2. Find coffee machine.\n3. Place mug under machine.\n4. Press 'Start'."
    },
    { role: "assistant", type: "action", content: "Action: navigate_to(kitchen)" },
    { role: "environment", content: "Obs: You are in the kitchen. You see a coffee_machine and a cabinet." },
    {
        role: "assistant",
        type: "thought",
        content: "I see the machine. Now I need a mug. Usually mugs are in the cabinet."
    },
    { role: "assistant", type: "action", content: "Action: open(cabinet)" },
    { role: "environment", content: "Obs: Cabinet is open. You see a white_mug and a blue_plate." },
    { role: "assistant", type: "action", content: "Action: pick_up(white_mug)" },
    { role: "environment", content: "Obs: You are holding white_mug." },
    { role: "assistant", type: "action", content: "Action: place_under(white_mug, coffee_machine)" },
    { role: "environment", content: "Obs: Mug placed. Machine is ready." },
    { role: "assistant", type: "action", content: "Action: toggle(coffee_machine)" },
    { role: "environment", content: "Reward: +10. Coffee is brewing!" }
];

export function FoundationModelsRL() {
    const [step, setStep] = useState(0);
    const [history, setHistory] = useState<any[]>([]);
    const [isThinking, setIsThinking] = useState(false);

    useEffect(() => {
        if (step < scenario.length) {
            const timeout = setTimeout(() => {
                if (scenario[step].type === "thought") {
                    setIsThinking(true);
                    setTimeout(() => {
                        setIsThinking(false);
                        setHistory(prev => [...prev, scenario[step]]);
                        setStep(prev => prev + 1);
                    }, 1500); // Thinking delay
                } else {
                    setHistory(prev => [...prev, scenario[step]]);
                    setStep(prev => prev + 1);
                }
            }, 1000); // Step delay
            return () => clearTimeout(timeout);
        }
    }, [step]);

    const reset = () => {
        setHistory([]);
        setStep(0);
    };

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-slate-50 dark:bg-slate-900 rounded-2xl shadow-xl border border-slate-200 dark:border-slate-800 flex flex-col md:flex-row gap-6">
            {/* Architecture Diagram - Pure SVG for perfect alignment */}
            <div className="w-full md:w-1/3 flex flex-col items-center gap-4">
                <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100">LLM-based Agent</h3>

                <div className="w-full h-[400px] bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-2 shadow-inner flex items-center justify-center">
                    <svg viewBox="0 0 300 400" className="w-full h-full max-w-[300px]">
                        <defs>
                            <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                                <path d="M0,0 L6,3 L0,6" fill="#94a3b8" />
                            </marker>
                            <marker id="arrow-blue" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                                <path d="M0,0 L6,3 L0,6" fill="#3b82f6" />
                            </marker>
                        </defs>

                        {/* 1. Prompt / Memory Node */}
                        <g transform="translate(50, 20)">
                            <rect width="200" height="70" rx="10" fill="#dbeafe" stroke="#bfdbfe" strokeWidth="2" className="dark:fill-blue-900/30 dark:stroke-blue-700" />
                            <text x="100" y="30" textAnchor="middle" className="font-bold fill-blue-700 dark:fill-blue-300 text-sm">Prompt / Memory</text>
                            <text x="100" y="50" textAnchor="middle" className="fill-blue-600 dark:fill-blue-400 text-xs">Context Window</text>
                        </g>

                        {/* Arrow: Prompt -> LLM */}
                        <motion.path
                            d="M 150 90 L 150 120"
                            stroke="#94a3b8"
                            strokeWidth="2"
                            fill="none"
                            markerEnd="url(#arrow)"
                            initial={{ pathLength: 0 }}
                            animate={{ pathLength: 1 }}
                        />

                        {/* 2. LLM Core Node */}
                        <g transform="translate(50, 130)">
                            <motion.rect
                                width="200" height="100" rx="10"
                                fill="#f3e8ff" stroke="#d8b4fe" strokeWidth="2"
                                className="dark:fill-purple-900/30 dark:stroke-purple-600"
                                animate={{
                                    stroke: isThinking ? "#a855f7" : "#d8b4fe",
                                    strokeWidth: isThinking ? 4 : 2,
                                    filter: isThinking ? "drop-shadow(0 0 8px rgba(168, 85, 247, 0.5))" : "none"
                                }}
                            />
                            <text x="100" y="45" textAnchor="middle" className="font-bold fill-purple-700 dark:fill-purple-300 text-lg">LLM Core</text>
                            <text x="100" y="65" textAnchor="middle" className="fill-purple-600 dark:fill-purple-400 text-xs">Policy π(a|h)</text>
                            {isThinking && (
                                <text x="100" y="85" textAnchor="middle" className="fill-purple-600 dark:fill-purple-400 text-xs font-bold animate-pulse">
                                    🤔 Reasoning...
                                </text>
                            )}
                        </g>

                        {/* Arrow: LLM -> Env */}
                        <motion.path
                            d="M 150 230 L 150 280"
                            stroke="#94a3b8"
                            strokeWidth="2"
                            fill="none"
                            markerEnd="url(#arrow)"
                        />

                        {/* 3. Environment Node */}
                        <g transform="translate(50, 290)">
                            <rect width="200" height="70" rx="10" fill="#dcfce7" stroke="#bbf7d0" strokeWidth="2" className="dark:fill-green-900/30 dark:stroke-green-700" />
                            <text x="100" y="30" textAnchor="middle" className="font-bold fill-green-700 dark:fill-green-300 text-sm">Environment</text>
                            <text x="100" y="50" textAnchor="middle" className="fill-green-600 dark:fill-green-400 text-xs">Observations & Feedback</text>
                        </g>

                        {/* Feedback Loop: Env -> Prompt (Curved Arrow on Left) */}
                        <motion.path
                            d="M 50 325 C 10 325, 10 55, 50 55"
                            stroke="#94a3b8"
                            strokeWidth="2"
                            fill="none"
                            strokeDasharray="5,5"
                            markerEnd="url(#arrow)"
                        />
                        <text x="30" y="190" transform="rotate(-90 30,190)" textAnchor="middle" className="fill-slate-400 text-[10px] tracking-widest">FEEDBACK LOOP</text>

                    </svg>
                </div>
            </div>

            {/* Chat / Interaction Log */}
            <div className="w-full md:w-2/3 flex flex-col h-[500px] bg-white dark:bg-slate-950 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden">
                <div className="p-3 border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 flex justify-between items-center">
                    <span className="text-sm font-bold text-slate-600 dark:text-slate-300">Agent Interaction Log</span>
                    <button
                        onClick={reset}
                        className="text-xs px-2 py-1 bg-slate-200 dark:bg-slate-700 rounded hover:bg-slate-300 dark:hover:bg-slate-600 transition"
                    >
                        Replay
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-4 font-mono text-sm">
                    <AnimatePresence>
                        {history.map((msg, idx) => (
                            <motion.div
                                key={idx}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                className={`flex flex-col max-w-[90%] ${msg.role === "assistant" ? "ml-auto items-end" : "mr-auto items-start"
                                    }`}
                            >
                                <div className={`px-3 py-2 rounded-lg whitespace-pre-wrap ${msg.role === "user" ? "bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200" :
                                    msg.role === "environment" ? "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300" :
                                        msg.type === "thought" ? "bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 text-slate-600 dark:text-slate-400 italic" :
                                            "bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200"
                                    }`}>
                                    {msg.role === "assistant" && msg.type === "thought" && <span className="mr-1">💭</span>}
                                    {msg.role === "assistant" && msg.type !== "thought" && <span className="mr-1">🤖</span>}
                                    {msg.role === "environment" && <span className="mr-1">🌍</span>}
                                    {msg.content}
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                    {step >= scenario.length && (
                        <div className="text-center text-xs text-slate-400 py-2">
                            --- Task Completed ---
                        </div>
                    )}
                    <div className="h-4" /> {/* Spacer */}
                </div>
            </div>
        </div>
    );
}
