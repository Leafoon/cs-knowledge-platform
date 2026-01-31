"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

export function OnlineLearningArchitecture() {
    const [activeFlow, setActiveFlow] = useState(false);

    // Components of the architecture
    const components = [
        { id: "users", name: "Users", icon: "👥", x: 50, y: 50, color: "blue" },
        { id: "env", name: "Environment / App", icon: "📱", x: 250, y: 50, color: "green" },
        { id: "model_serving", name: "Model Serving", icon: "🚀", x: 450, y: 50, color: "purple" },
        { id: "queue", name: "Message Queue (Kafka/Redis)", icon: "📥", x: 250, y: 200, color: "orange" },
        { id: "trainer", name: "Online Trainer", icon: "⚙️", x: 250, y: 350, color: "red" },
        { id: "model_registry", name: "Model Registry", icon: "📦", x: 450, y: 350, color: "indigo" }
    ];

    // Data flow paths
    const flows = [
        { from: "users", to: "env", label: "Action" },
        { from: "env", to: "model_serving", label: "Request" },
        { from: "model_serving", to: "env", label: "Prediction" },
        { from: "env", to: "queue", label: "Logs (S,A,R,S')" },
        { from: "queue", to: "trainer", label: "Batch" },
        { from: "trainer", to: "model_registry", label: "Update Weights" },
        { from: "model_registry", to: "model_serving", label: "Hot Swap" }
    ];

    useEffect(() => {
        const interval = setInterval(() => {
            setActiveFlow(prev => !prev);
        }, 2000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    在线学习系统架构
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Online Learning System Architecture
                </p>

                <button
                    onClick={() => setActiveFlow(!activeFlow)}
                    className="mt-4 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-sm transition"
                >
                    {activeFlow ? "停止数据流" : "启动数据流模拟"}
                </button>
            </div>

            <div className="relative h-[450px] bg-white dark:bg-slate-800 rounded-xl shadow-inner overflow-hidden border border-gray-200 dark:border-slate-700">
                {/* SVG Connections */}
                <svg className="absolute inset-0 w-full h-full pointer-events-none">
                    <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" />
                        </marker>
                    </defs>
                    {flows.map((flow, idx) => {
                        const start = components.find(c => c.id === flow.from)!;
                        const end = components.find(c => c.id === flow.to)!;

                        // Calculate center points (assuming simplified layout for demo)
                        // This uses a predefined layout logic, which is simpler than dynamic graph computation

                        // Using explicit coordinates for cleaner visualization
                        const coords: Record<string, { x: number, y: number }> = {
                            "users": { x: 100, y: 80 },
                            "env": { x: 300, y: 80 },
                            "model_serving": { x: 500, y: 80 },
                            "queue": { x: 300, y: 230 },
                            "trainer": { x: 300, y: 380 },
                            "model_registry": { x: 500, y: 380 }
                        };

                        const p1 = coords[flow.from];
                        const p2 = coords[flow.to];

                        return (
                            <g key={`${flow.from}-${flow.to}`}>
                                <line
                                    x1={p1.x} y1={p1.y}
                                    x2={p2.x} y2={p2.y}
                                    stroke="#cbd5e1"
                                    strokeWidth="2"
                                    markerEnd="url(#arrowhead)"
                                />
                                {/* Animated flow packet */}
                                {activeFlow && (
                                    <circle r="4" fill="#6366f1">
                                        <animateMotion
                                            dur="1.5s"
                                            repeatCount="indefinite"
                                            path={`M${p1.x},${p1.y} L${p2.x},${p2.y}`}
                                            begin={`${idx * 0.5}s`}
                                        />
                                    </circle>
                                )}
                                {/* Label */}
                                <text
                                    x={(p1.x + p2.x) / 2}
                                    y={(p1.y + p2.y) / 2 - 10}
                                    textAnchor="middle"
                                    className="text-[10px] fill-slate-500 font-mono bg-white"
                                >
                                    {flow.label}
                                </text>
                            </g>
                        );
                    })}
                </svg>

                {/* Component Nodes */}
                <div className="absolute inset-0">
                    {/* Manually positioned for the specific layout */}

                    {/* Top Row */}
                    <div className="absolute top-10 left-[10%] transform -translate-x-1/2">
                        <Node {...components[0]} />
                    </div>
                    <div className="absolute top-10 left-[40%] transform -translate-x-1/2">
                        <Node {...components[1]} />
                    </div>
                    <div className="absolute top-10 left-[70%] transform -translate-x-1/2">
                        <Node {...components[2]} />
                    </div>

                    {/* Middle */}
                    <div className="absolute top-[180px] left-[40%] transform -translate-x-1/2">
                        <Node {...components[3]} />
                    </div>

                    {/* Bottom Row */}
                    <div className="absolute top-[330px] left-[40%] transform -translate-x-1/2">
                        <Node {...components[4]} />
                    </div>
                    <div className="absolute top-[330px] left-[70%] transform -translate-x-1/2">
                        <Node {...components[5]} />
                    </div>
                </div>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-4 text-xs text-slate-600 dark:text-slate-400">
                <div className="bg-white dark:bg-slate-800 p-3 rounded-lg shadow-sm">
                    <strong>1. Inference Loop:</strong> Users &rarr; App &rarr; Model Serving &rarr; App
                </div>
                <div className="bg-white dark:bg-slate-800 p-3 rounded-lg shadow-sm">
                    <strong>2. Training Loop:</strong> Logs &rarr; Queue &rarr; Trainer &rarr; Registry
                </div>
            </div>
        </div>
    );
}

function Node({ name, icon, color }: { name: string, icon: string, color: string }) {
    return (
        <motion.div
            className={`flex flex-col items-center justify-center w-32 h-24 bg-white dark:bg-slate-800 rounded-xl border-2 border-${color}-500 shadow-md z-10`}
            whileHover={{ scale: 1.05 }}
        >
            <div className={`text-3xl p-2 rounded-full bg-${color}-50 dark:bg-${color}-900/20 mb-1`}>
                {icon}
            </div>
            <div className="text-xs font-bold text-center text-slate-700 dark:text-slate-200 px-1 leading-tight">
                {name}
            </div>
        </motion.div>
    );
}
