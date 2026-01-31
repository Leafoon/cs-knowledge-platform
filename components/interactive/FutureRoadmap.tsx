"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function FutureRoadmap() {
    const [activeTab, setActiveTab] = useState("challenges");

    const challenges = [
        { name: "Sample Efficiency", current: 30, goal: 95, color: "red", desc: "Humans learn in minutes, RL needs days." },
        { name: "Generalization", current: 45, goal: 90, color: "blue", desc: "Adapt to unseen environments." },
        { name: "Safety & Alignment", current: 40, goal: 99, color: "green", desc: "Provably safe behaviors." },
        { name: "Reasoning", current: 60, goal: 90, color: "purple", desc: "Long-horizon planning & causality." }
    ];

    return (
        <div className="w-full max-w-4xl mx-auto p-8 bg-white dark:bg-slate-900 rounded-2xl shadow-xl border border-gray-100 dark:border-slate-800">
            <h3 className="text-2xl font-bold text-center mb-8 text-slate-800 dark:text-slate-100">
                RL to AGI: The Road Ahead
            </h3>

            <div className="flex justify-center gap-4 mb-8">
                <button
                    onClick={() => setActiveTab("challenges")}
                    className={`px-4 py-2 rounded-full font-bold transition ${activeTab === "challenges" ? "bg-slate-800 text-white dark:bg-white dark:text-slate-900" : "bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400"}`}
                >
                    Core Challenges
                </button>
                <button
                    onClick={() => setActiveTab("timeline")}
                    className={`px-4 py-2 rounded-full font-bold transition ${activeTab === "timeline" ? "bg-slate-800 text-white dark:bg-white dark:text-slate-900" : "bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400"}`}
                >
                    Milestones Timeline
                </button>
            </div>

            {activeTab === "challenges" && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {challenges.map(item => (
                        <div key={item.name} className="bg-slate-50 dark:bg-slate-800/50 p-6 rounded-xl border border-slate-100 dark:border-slate-700">
                            <div className="flex justify-between items-end mb-2">
                                <h4 className="font-bold text-lg text-slate-700 dark:text-slate-200">{item.name}</h4>
                                <span className={`text-sm font-bold text-${item.color}-500`}>Goal: {item.goal}%</span>
                            </div>
                            <p className="text-sm text-slate-500 mb-4 h-10">{item.desc}</p>

                            <div className="relative h-6 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                                {/* Current Progress */}
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${item.current}%` }}
                                    transition={{ duration: 1.5, ease: "easeOut" }}
                                    className={`absolute top-0 bottom-0 left-0 bg-${item.color}-500 opacity-80`}
                                />
                                {/* Gap */}
                                <div
                                    className="absolute top-0 bottom-0 border-r-2 border-dashed border-slate-400 dark:border-slate-500 z-10"
                                    style={{ left: `${item.current}%` }}
                                />
                                <div
                                    className="absolute top-0 bottom-0 border-r-4 border-slate-800 dark:border-white z-10"
                                    style={{ left: `${item.goal}%` }}
                                />
                            </div>
                            <div className="mt-2 flex justify-between text-xs font-mono text-slate-400">
                                <span>Current SOTA</span>
                                <span>Human Level</span>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {activeTab === "timeline" && (
                <div className="relative border-l-4 border-slate-200 dark:border-slate-700 ml-4 space-y-8 py-4">
                    {[
                        { year: "2015", event: "DQN (Atari)", solved: true },
                        { year: "2016", event: "AlphaGo (Board Games)", solved: true },
                        { year: "2019", event: "AlphaStar (RTS Games)", solved: true },
                        { year: "2022", event: "Gato (Generalist Agent)", solved: true },
                        { year: "2023", event: "RT-2 (VLA Models)", solved: true },
                        { year: "2025+", event: "Open-Ended Learning", solved: false },
                        { year: "2030+", event: "AGI (Human-Level)", solved: false },
                    ].map((item, idx) => (
                        <motion.div
                            key={item.year}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.1 }}
                            className="relative pl-8"
                        >
                            <div className={`absolute -left-[11px] top-1.5 w-5 h-5 rounded-full border-4 ${item.solved ? "bg-green-500 border-white dark:border-slate-900" : "bg-slate-300 dark:bg-slate-600 border-white dark:border-slate-900"
                                }`} />
                            <div className="font-mono text-sm text-slate-500 mb-1">{item.year}</div>
                            <div className={`text-lg font-bold ${item.solved ? "text-slate-800 dark:text-slate-100" : "text-slate-400 dark:text-slate-500"}`}>
                                {item.event}
                            </div>
                        </motion.div>
                    ))}
                </div>
            )}
        </div>
    );
}
