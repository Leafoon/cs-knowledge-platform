"use client";

import { motion } from "framer-motion";

export function LeagueTrainingArchitecture() {
    const agents = {
        main: [
            { id: 1, name: "Main 1", payoff: 0.78, color: "bg-blue-500" },
            { id: 2, name: "Main 2", payoff: 0.82, color: "bg-blue-600" },
            { id: 3, name: "Main 3", payoff: 0.75, color: "bg-blue-700" }
        ],
        mainExploiters: [
            { id: 4, name: "M-Exp 1", payoff: 0.65, color: "bg-red-500" },
            { id: 5, name: "M-Exp 2", payoff: 0.68, color: "bg-red-600" }
        ],
        leagueExploiters: [
            { id: 6, name: "L-Exp 1", payoff: 0.62, color: "bg-purple-500" },
            { id: 7, name: "L-Exp 2", payoff: 0.64, color: "bg-purple-600" }
        ],
        historical: [
            { id: 8, name: "Hist 1", payoff: 0.55, color: "bg-gray-400" },
            { id: 9, name: "Hist 2", payoff: 0.60, color: "bg-gray-500" },
            { id: 10, name: "Hist 3", payoff: 0.58, color: "bg-gray-600" }
        ]
    };

    return (
        <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-gray-100 dark:from-slate-900 dark:to-gray-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    League Training Architecture (AlphaStar)
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Multi-population structure for robust policy learning
                </p>
            </div>

            {/* Legend */}
            <div className="grid grid-cols-4 gap-4 mb-6">
                <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg border-2 border-blue-300 dark:border-blue-700">
                    <div className="font-bold text-blue-700 dark:text-blue-400 text-sm mb-1">Main Agents</div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                        Continuously improve via diverse opponents
                    </div>
                </div>

                <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded-lg border-2 border-red-300 dark:border-red-700">
                    <div className="font-bold text-red-700 dark:text-red-400 text-sm mb-1">Main Exploiters</div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                        Find weaknesses in main agents
                    </div>
                </div>

                <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-lg border-2 border-purple-300 dark:border-purple-700">
                    <div className="font-bold text-purple-700 dark:text-purple-400 text-sm mb-1">League Exploiters</div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                        Exploit entire league population
                    </div>
                </div>

                <div className="bg-gray-50 dark:bg-gray-800/50 p-3 rounded-lg border-2 border-gray-300 dark:border-gray-700">
                    <div className="font-bold text-gray-700 dark:text-gray-400 text-sm mb-1">Historical</div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                        Frozen past snapshots
                    </div>
                </div>
            </div>

            {/* Architecture Diagram */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-8 shadow-lg mb-6 relative">
                <div className="grid grid-cols-4 gap-8">
                    {/* Main Agents Column */}
                    <div className="space-y-4">
                        <div className="text-center font-bold text-blue-600 dark:text-blue-400 mb-2">Main Agents</div>
                        {agents.main.map((agent, idx) => (
                            <motion.div
                                key={agent.id}
                                className={`${agent.color} text-white p-4 rounded-lg shadow-md`}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: idx * 0.1 }}
                            >
                                <div className="font-bold text-sm mb-1">{agent.name}</div>
                                <div className="text-xs">Payoff: {agent.payoff}</div>
                                <div className="mt-2 h-1 bg-white/30 rounded">
                                    <div className="h-full bg-white rounded" style={{ width: `${agent.payoff * 100}%` }} />
                                </div>
                            </motion.div>
                        ))}
                    </div>

                    {/* Main Exploiters Column */}
                    <div className="space-y-4">
                        <div className="text-center font-bold text-red-600 dark:text-red-400 mb-2">Main Exploiters</div>
                        {agents.mainExploiters.map((agent, idx) => (
                            <motion.div
                                key={agent.id}
                                className={`${agent.color} text-white p-4 rounded-lg shadow-md`}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: (agents.main.length + idx) * 0.1 }}
                            >
                                <div className="font-bold text-sm mb-1">{agent.name}</div>
                                <div className="text-xs">Payoff: {agent.payoff}</div>
                                <div className="text-xs mt-1">Target: Main</div>
                            </motion.div>
                        ))}
                    </div>

                    {/* League Exploiters Column */}
                    <div className="space-y-4">
                        <div className="text-center font-bold text-purple-600 dark:text-purple-400 mb-2">League Exploiters</div>
                        {agents.leagueExploiters.map((agent, idx) => (
                            <motion.div
                                key={agent.id}
                                className={`${agent.color} text-white p-4 rounded-lg shadow-md`}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: (agents.main.length + agents.mainExploiters.length + idx) * 0.1 }}
                            >
                                <div className="font-bold text-sm mb-1">{agent.name}</div>
                                <div className="text-xs">Payoff: {agent.payoff}</div>
                                <div className="text-xs mt-1">Target: All</div>
                            </motion.div>
                        ))}
                    </div>

                    {/* Historical Column */}
                    <div className="space-y-4">
                        <div className="text-center font-bold text-gray-600 dark:text-gray-400 mb-2">Historical</div>
                        {agents.historical.map((agent, idx) => (
                            <motion.div
                                key={agent.id}
                                className={`${agent.color} text-white p-4 rounded-lg shadow-md opacity-70`}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 0.7, x: 0 }}
                                transition={{ delay: (agents.main.length + agents.mainExploiters.length + agents.leagueExploiters.length + idx) * 0.1 }}
                            >
                                <div className="font-bold text-sm mb-1">{agent.name}</div>
                                <div className="text-xs">Frozen</div>
                                <div className="text-xs">Age: {idx + 5}k steps</div>
                            </motion.div>
                        ))}
                    </div>
                </div>

                {/* Connection arrows (simplified) */}
                <svg className="absolute inset-0 pointer-events-none" style={{ zIndex: -1 }}>
                    <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                            <polygon points="0 0, 10 3, 0 6" fill="#94a3b8" />
                        </marker>
                    </defs>
                    {/* Example arrows - in real implementation would be dynamic */}
                    <line x1="25%" y1="30%" x2="35%" y2="30%" stroke="#94a3b8" strokeWidth="2" markerEnd="url(#arrowhead)" strokeDasharray="4" />
                    <line x1="50%" y1="30%" x2="60%" y2="30%" stroke="#94a3b8" strokeWidth="2" markerEnd="url(#arrowhead)" strokeDasharray="4" />
                </svg>
            </div>

            {/* Opponent Sampling Strategy */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">Main Agent Opponent Sampling</h4>
                <div className="space-y-3">
                    <div>
                        <div className="flex justify-between text-sm mb-1">
                            <span className="font-semibold">Self-Play (35%)</span>
                            <span className="text-slate-600 dark:text-slate-400">vs Current Self</span>
                        </div>
                        <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <div className="h-full bg-blue-500" style={{ width: "35%" }} />
                        </div>
                    </div>

                    <div>
                        <div className="flex justify-between text-sm mb-1">
                            <span className="font-semibold">Historical (50%)</span>
                            <span className="text-slate-600 dark:text-slate-400">vs Past Snapshots</span>
                        </div>
                        <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <div className="h-full bg-gray-500" style={{ width: "50%" }} />
                        </div>
                    </div>

                    <div>
                        <div className="flex justify-between text-sm mb-1">
                            <span className="font-semibold">Exploiters (15%)</span>
                            <span className="text-slate-600 dark:text-slate-400">vs Adversaries</span>
                        </div>
                        <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <div className="h-full bg-gradient-to-r from-red-500 to-purple-500" style={{ width: "15%" }} />
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-600 dark:text-slate-400">
                💡 League training prevents exploitation by maintaining diverse opponents across multiple timescales
            </div>
        </div>
    );
}
