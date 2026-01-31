"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function EmergentBehaviorDemo() {
    const [phase, setPhase] = useState(0);

    const phases = [
        {
            name: "Phase 1: Running & Chasing",
            description: "Basic movement skills emerge. Hiders run, seekers chase.",
            hidersStrategy: "Random movement",
            seekersStrategy: "Chase nearest hider",
            complexity: 1,
            timestep: "0-10M steps"
        },
        {
            name: "Phase 2: Tool Use - Shelter Building",
            description: "Hiders learn to move boxes to build shelters!",
            hidersStrategy: "Push boxes to create walls",
            seekersStrategy: "Still basic chasing",
            complexity: 3,
            timestep: "10-25M steps"
        },
        {
            name: "Phase 3: Counter-strategy - Ramp Usage",
            description: "Seekers discover they can climb ramps to see over walls.",
            hidersStrategy: "Build shelters",
            seekersStrategy: "Use ramps to climb over",
            complexity: 4,
            timestep: "25-75M steps"
        },
        {
            name: "Phase 4: Meta-strategy - Lock Ramps",
            description: "Hiders lock ramps away before building shelters!",
            hidersStrategy: "Lock ramps + build shelter",
            seekersStrategy: "Try to find unlocked ramps",
            complexity: 6,
            timestep: "75-130M steps"
        },
        {
            name: "Phase 5: Exploit Physics - Box Surfing",
            description: "Seekers discover physics exploit: surf on moving boxes!",
            hidersStrategy: "Lock ramps + shelter",
            seekersStrategy: "Box surfing glitch",
            complexity: 8,
            timestep: "130-380M steps"
        },
        {
            name: "Phase 6: Ultimate Defense - Lock Everything",
            description: "Hiders learn to lock ALL movable objects before hiding.",
            hidersStrategy: "Lock all objects systematically",
            seekersStrategy: "Find any unlocked objects",
            complexity: 10,
            timestep: "380M+ steps"
        }
    ];

    const currentPhase = phases[phase];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-slate-900 dark:to-emerald-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Emergent Behavior Demo: Hide-and-Seek
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Watch how complex strategies emerge from simple objectives
                </p>
            </div>

            {/* Timeline */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="flex justify-between items-center mb-4">
                    {phases.map((p, idx) => (
                        <div key={idx} className="flex flex-col items-center">
                            <motion.div
                                className={`w-12 h-12 rounded-full flex items-center justify-center font-bold cursor-pointer ${idx === phase
                                        ? "bg-gradient-to-br from-emerald-500 to-teal-600 text-white shadow-lg scale-110"
                                        : idx < phase
                                            ? "bg-emerald-300 dark:bg-emerald-700 text-white"
                                            : "bg-gray-200 dark:bg-gray-700 text-gray-500"
                                    }`}
                                onClick={() => setPhase(idx)}
                                whileHover={{ scale: 1.1 }}
                                whileTap={{ scale: 0.95 }}
                            >
                                {idx + 1}
                            </motion.div>
                            {idx < phases.length - 1 && (
                                <div className={`h-1 w-full mt-6 ${idx < phase ? "bg-emerald-400" : "bg-gray-300 dark:bg-gray-700"
                                    }`} style={{ position: "absolute", width: "calc(16.666% - 12px)", left: `calc(${idx * 16.666}% + 24px)` }} />
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Current Phase Details */}
            <motion.div
                key={phase}
                className="bg-white dark:bg-slate-800 rounded-xl p-8 shadow-lg mb-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
            >
                <div className="flex items-start justify-between mb-4">
                    <div>
                        <h4 className="text-2xl font-bold text-emerald-600 dark:text-emerald-400 mb-2">
                            {currentPhase.name}
                        </h4>
                        <p className="text-slate-600 dark:text-slate-400 text-sm">
                            {currentPhase.timestep}
                        </p>
                    </div>
                    <div className="text-right">
                        <div className="text-sm text-slate-500 mb-1">Complexity Level</div>
                        <div className="flex gap-1">
                            {Array.from({ length: 10 }).map((_, i) => (
                                <div
                                    key={i}
                                    className={`w-3 h-6 rounded ${i < currentPhase.complexity
                                            ? "bg-gradient-to-t from-emerald-600 to-emerald-400"
                                            : "bg-gray-200 dark:bg-gray-700"
                                        }`}
                                />
                            ))}
                        </div>
                    </div>
                </div>

                <p className="text-lg text-slate-700 dark:text-slate-300 mb-6">
                    {currentPhase.description}
                </p>

                <div className="grid grid-cols-2 gap-6">
                    {/* Hiders */}
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg border-2 border-blue-200 dark:border-blue-800">
                        <div className="flex items-center gap-2 mb-3">
                            <div className="w-4 h-4 bg-blue-500 rounded-full"></div>
                            <h5 className="font-bold text-blue-700 dark:text-blue-400">Hiders Strategy</h5>
                        </div>
                        <p className="text-slate-700 dark:text-slate-300">
                            {currentPhase.hidersStrategy}
                        </p>
                    </div>

                    {/* Seekers */}
                    <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-lg border-2 border-red-200 dark:border-red-800">
                        <div className="flex items-center gap-2 mb-3">
                            <div className="w-4 h-4 bg-red-500 rounded-full"></div>
                            <h5 className="font-bold text-red-700 dark:text-red-400">Seekers Strategy</h5>
                        </div>
                        <p className="text-slate-700 dark:text-slate-300">
                            {currentPhase.seekersStrategy}
                        </p>
                    </div>
                </div>
            </motion.div>

            {/* Visual Representation (Simplified Grid) */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Environment Snapshot</h4>
                <div className="grid grid-cols-8 gap-2">
                    {Array.from({ length: 64 }).map((_, i) => {
                        const row = Math.floor(i / 8);
                        const col = i % 8;

                        // Simplified representation based on phase
                        let cellContent = "";
                        let cellColor = "bg-gray-100 dark:bg-gray-700";

                        // Walls
                        if (row === 0 || row === 7 || col === 0 || col === 7) {
                            cellColor = "bg-gray-400 dark:bg-gray-600";
                        }
                        // Hiders (blue)
                        else if ((row === 2 && col === 2) || (row === 5 && col === 5)) {
                            cellContent = "🔵";
                            cellColor = "bg-blue-100 dark:bg-blue-900/30";
                        }
                        // Seekers (red)
                        else if ((row === 2 && col === 6) || (row === 5 && col === 2)) {
                            cellContent = "🔴";
                            cellColor = "bg-red-100 dark:bg-red-900/30";
                        }
                        // Objects (boxes/ramps) appear in later phases
                        else if (phase >= 1 && ((row === 3 && col === 3) || (row === 4 && col === 4))) {
                            cellContent = phase >= 3 ? "🔒" : "📦";
                            cellColor = "bg-yellow-100 dark:bg-yellow-900/30";
                        }
                        else if (phase >= 2 && row === 3 && col === 5) {
                            cellContent = "🔺";
                            cellColor = "bg-purple-100 dark:bg-purple-900/30";
                        }

                        return (
                            <div
                                key={i}
                                className={`aspect-square ${cellColor} rounded flex items-center justify-center text-lg`}
                            >
                                {cellContent}
                            </div>
                        );
                    })}
                </div>
                <div className="flex gap-4 justify-center mt-4 text-xs text-slate-600 dark:text-slate-400">
                    <span>🔵 Hiders</span>
                    <span>🔴 Seekers</span>
                    <span>📦 Boxes</span>
                    <span>🔺 Ramps</span>
                    <span>🔒 Locked</span>
                </div>
            </div>

            {/* Navigation */}
            <div className="flex gap-4 justify-center">
                <motion.button
                    onClick={() => setPhase(Math.max(0, phase - 1))}
                    disabled={phase === 0}
                    className={`px-6 py-3 rounded-lg font-semibold shadow-lg ${phase === 0
                            ? "bg-gray-300 dark:bg-gray-700 text-gray-500 cursor-not-allowed"
                            : "bg-gradient-to-r from-emerald-500 to-teal-600 text-white"
                        }`}
                    whileHover={phase > 0 ? { scale: 1.05 } : {}}
                    whileTap={phase > 0 ? { scale: 0.95 } : {}}
                >
                    ← Previous
                </motion.button>

                <motion.button
                    onClick={() => setPhase(Math.min(phases.length - 1, phase + 1))}
                    disabled={phase === phases.length - 1}
                    className={`px-6 py-3 rounded-lg font-semibold shadow-lg ${phase === phases.length - 1
                            ? "bg-gray-300 dark:bg-gray-700 text-gray-500 cursor-not-allowed"
                            : "bg-gradient-to-r from-emerald-500 to-teal-600 text-white"
                        }`}
                    whileHover={phase < phases.length - 1 ? { scale: 1.05 } : {}}
                    whileTap={phase < phases.length - 1 ? { scale: 0.95 } : {}}
                >
                    Next →
                </motion.button>
            </div>

            <div className="mt-6 text-center text-sm text-slate-600 dark:text-slate-400">
                💡 Complex behaviors emerge naturally from competitive pressure—no explicit programming required!
            </div>
        </div>
    );
}
