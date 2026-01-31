"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function PopulationDiversity() {
    const [populationSize] = useState(8);
    const [generation, setGeneration] = useState(0);
    const [showMetrics, setShowMetrics] = useState(true);

    // Generate population with diverse strategies
    const generatePopulation = (gen: number) => {
        const strategies = ["Aggressive", "Defensive", "Balanced", "Adaptive"];
        const colors = ["bg-red-500", "bg-blue-500", "bg-green-500", "bg-purple-500"];

        return Array.from({ length: populationSize }, (_, i) => {
            const strategyIdx = (i + gen) % strategies.length;
            const fitness = 0.5 + 0.3 * Math.sin((i + gen) * 0.5) + Math.random() * 0.2;
            return {
                id: i,
                strategy: strategies[strategyIdx],
                color: colors[strategyIdx],
                fitness: Math.max(0.1, Math.min(1.0, fitness)),
                winRate: 0.4 + fitness * 0.4,
                age: gen
            };
        }).sort((a, b) => b.fitness - a.fitness);
    };

    const [population, setPopulation] = useState(generatePopulation(0));

    const diversity = population.reduce((sum, agent, i, arr) => {
        const unique = arr.filter(a => a.strategy === agent.strategy).length;
        return sum + (1.0 / unique);
    }, 0) / populationSize;

    const avgFitness = population.reduce((sum, a) => sum + a.fitness, 0) / populationSize;

    const handleEvolve = () => {
        const newGen = generation + 1;
        setGeneration(newGen);

        // Simulate evolution: top 50% survive, rest are mutated copies
        const newPop = generatePopulation(newGen);
        setPopulation(newPop);
    };

    const handleReset = () => {
        setGeneration(0);
        setPopulation(generatePopulation(0));
    };

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-slate-900 dark:to-cyan-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Population Diversity (PBT)
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Visualize strategy diversity in population-based training
                </p>
            </div>

            {/* Metrics */}
            {showMetrics && (
                <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg text-center">
                        <div className="text-sm font-semibold text-slate-600 dark:text-slate-400 mb-1">Generation</div>
                        <div className="text-3xl font-bold text-cyan-600 dark:text-cyan-400">{generation}</div>
                    </div>

                    <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg text-center">
                        <div className="text-sm font-semibold text-slate-600 dark:text-slate-400 mb-1">Diversity Score</div>
                        <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">{diversity.toFixed(2)}</div>
                        <div className="text-xs text-slate-500 mt-1">{diversity > 0.6 ? "High" : "Low"} diversity</div>
                    </div>

                    <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg text-center">
                        <div className="text-sm font-semibold text-slate-600 dark:text-slate-400 mb-1">Avg Fitness</div>
                        <div className="text-3xl font-bold text-green-600 dark:text-green-400">{avgFitness.toFixed(2)}</div>
                        <div className="text-xs text-slate-500 mt-1">Population strength</div>
                    </div>
                </div>
            )}

            {/* Population Grid */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="flex justify-between items-center mb-4">
                    <h4 className="text-lg font-bold">Agent Population ({populationSize})</h4>
                    <button
                        onClick={() => setShowMetrics(!showMetrics)}
                        className="text-xs px-3 py-1 bg-gray-200 dark:bg-gray-700 rounded"
                    >
                        {showMetrics ? "Hide" : "Show"} Metrics
                    </button>
                </div>

                <div className="grid grid-cols-4 gap-4">
                    {population.map((agent, idx) => (
                        <motion.div
                            key={agent.id}
                            className="relative p-4 rounded-lg shadow-md"
                            style={{
                                background: `linear-gradient(135deg, ${agent.strategy === "Aggressive" ? "#ef4444 0%, #dc2626 100%" :
                                        agent.strategy === "Defensive" ? "#3b82f6 0%, #2563eb 100%" :
                                            agent.strategy === "Balanced" ? "#22c55e 0%, #16a34a 100%" :
                                                "#a855f7 0%, #9333ea 100%"
                                    })`
                            }}
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: idx * 0.05 }}
                            whileHover={{ scale: 1.05 }}
                        >
                            <div className="text-white">
                                <div className="flex justify-between items-center mb-2">
                                    <span className="font-bold text-sm">Agent {agent.id + 1}</span>
                                    {idx < populationSize / 2 && (
                                        <span className="text-xs bg-yellow-400 text-yellow-900 px-2 py-0.5 rounded">
                                            Elite
                                        </span>
                                    )}
                                </div>

                                <div className="text-xs font-semibold mb-3">{agent.strategy}</div>

                                <div className="space-y-1 text-xs">
                                    <div className="flex justify-between">
                                        <span>Fitness:</span>
                                        <span className="font-mono">{agent.fitness.toFixed(2)}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Win Rate:</span>
                                        <span className="font-mono">{(agent.winRate * 100).toFixed(0)}%</span>
                                    </div>
                                </div>

                                {/* Fitness bar */}
                                <div className="mt-2 h-1.5 bg-white/30 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-white"
                                        style={{ width: `${agent.fitness * 100}%` }}
                                    />
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>

            {/* Strategy Distribution */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Strategy Distribution</h4>
                <div className="space-y-3">
                    {["Aggressive", "Defensive", "Balanced", "Adaptive"].map((strategy, idx) => {
                        const count = population.filter(a => a.strategy === strategy).length;
                        const percentage = (count / populationSize) * 100;
                        const colors = ["red", "blue", "green", "purple"];

                        return (
                            <div key={strategy}>
                                <div className="flex justify-between text-sm mb-1">
                                    <span className="font-semibold">{strategy}</span>
                                    <span className="text-slate-600 dark:text-slate-400">
                                        {count} agents ({percentage.toFixed(0)}%)
                                    </span>
                                </div>
                                <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                    <motion.div
                                        className={`h-full bg-${colors[idx]}-500`}
                                        initial={{ width: 0 }}
                                        animate={{ width: `${percentage}%` }}
                                        transition={{ duration: 0.5 }}
                                        style={{
                                            background: idx === 0 ? "#ef4444" :
                                                idx === 1 ? "#3b82f6" :
                                                    idx === 2 ? "#22c55e" : "#a855f7"
                                        }}
                                    />
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Controls */}
            <div className="flex gap-4 justify-center">
                <motion.button
                    onClick={handleEvolve}
                    className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-lg font-semibold shadow-lg"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                >
                    🧬 Evolve Generation
                </motion.button>

                <motion.button
                    onClick={handleReset}
                    className="px-6 py-3 bg-gray-500 text-white rounded-lg font-semibold shadow-lg"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                >
                    ↻ Reset
                </motion.button>
            </div>

            <div className="mt-6 text-center text-sm text-slate-600 dark:text-slate-400">
                💡 PBT maintains diversity: elite agents survive, weak agents copy and mutate
            </div>
        </div>
    );
}
