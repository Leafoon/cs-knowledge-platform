"use client";

import { motion } from "framer-motion";

export function CooperativeTaskVisualization() {
    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-slate-900 dark:to-green-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Cooperative Task Structure
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Shared rewards vs individual observations in Dec-POMDP
                </p>
            </div>

            <div className="grid grid-cols-2 gap-6">
                {/* Shared Reward */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold mb-4 text-green-600 dark:text-green-400">✅ Shared Reward</h4>
                    <div className="space-y-4">
                        <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-2 border-green-300 dark:border-green-700">
                            <div className="text-center text-3xl mb-2">🎯</div>
                            <div className="text-center font-bold">Team Goal</div>
                            <div className="text-sm text-center text-slate-600 dark:text-slate-400 mt-2">
                                r<sub>shared</sub> = r¹ = r² = r³
                            </div>
                        </div>

                        <div className="grid grid-cols-3 gap-2">
                            {[1, 2, 3].map(i => (
                                <motion.div
                                    key={i}
                                    className="bg-blue-500 text-white p-3 rounded-lg text-center text-sm font-semibold"
                                    animate={{ y: [0, -5, 0] }}
                                    transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                                >
                                    Agent {i}
                                </motion.div>
                            ))}
                        </div>

                        <div className="text-xs text-slate-600 dark:text-slate-400 mt-2">
                            All agents receive same reward → Natural alignment
                        </div>
                    </div>
                </div>

                {/* Partial Observability */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold mb-4 text-orange-600 dark:text-orange-400">👁️ Partial Observability</h4>
                    <div className="space-y-4">
                        <div className="relative h-40 bg-gray-100 dark:bg-gray-800 rounded-lg overflow-hidden">
                            {/* Global state (hidden) */}
                            <div className="absolute inset-0 grid grid-cols-4 grid-rows-4 gap-1 p-2 opacity-30">
                                {Array.from({ length: 16 }).map((_, i) => (
                                    <div key={i} className="bg-gray-400 dark:bg-gray-600 rounded" />
                                ))}
                            </div>

                            {/* Agent 1 view (top-left) */}
                            <div className="absolute top-2 left-2 w-16 h-16 border-4 border-blue-500 bg-blue-200/50 dark:bg-blue-800/50 rounded flex items-center justify-center">
                                <span className="text-xs font-bold">A1 View</span>
                            </div>

                            {/* Agent 2 view (center) */}
                            <div className="absolute top-12 left-20 w-16 h-16 border-4 border-green-500 bg-green-200/50 dark:bg-green-800/50 rounded flex items-center justify-center">
                                <span className="text-xs font-bold">A2 View</span>
                            </div>

                            {/* Agent 3 view (bottom-right) */}
                            <div className="absolute bottom-2 right-2 w-16 h-16 border-4 border-purple-500 bg-purple-200/50 dark:bg-purple-800/50 rounded flex items-center justify-center">
                                <span className="text-xs font-bold">A3 View</span>
                            </div>
                        </div>

                        <div className="text-xs text-slate-600 dark:text-slate-400">
                            Each agent sees only local information → Must coordinate with incomplete knowledge
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">Dec-POMDP Components</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded">
                        <div className="font-bold text-blue-700 dark:text-blue-400 mb-1">States (S)</div>
                        <div className="text-xs">Global environment state</div>
                    </div>
                    <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded">
                        <div className="font-bold text-green-700 dark:text-green-400 mb-1">Actions (A<sup>i</sup>)</div>
                        <div className="text-xs">Per-agent action space</div>
                    </div>
                    <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded">
                        <div className="font-bold text-orange-700 dark:text-orange-400 mb-1">Observations (O<sup>i</sup>)</div>
                        <div className="text-xs">Local partial views</div>
                    </div>
                    <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded">
                        <div className="font-bold text-purple-700 dark:text-purple-400 mb-1">Reward (R)</div>
                        <div className="text-xs">Shared team reward</div>
                    </div>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-600 dark:text-slate-400">
                💡 Cooperation requires coordination despite partial observability
            </div>
        </div>
    );
}
