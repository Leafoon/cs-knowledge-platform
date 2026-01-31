"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

export function SelfPlayEvolution() {
    const [iteration, setIteration] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const maxIterations = 100;

    // Simulate self-play training metrics
    const generateMetrics = (iter: number) => {
        const progress = iter / maxIterations;
        return {
            winRate: 0.5 + 0.4 * (1 - Math.exp(-progress * 3)) + Math.random() * 0.05,
            eloRating: 1000 + 400 * progress + Math.random() * 50,
            policyDiversity: 0.8 * Math.exp(-progress * 2) + 0.2 + Math.random() * 0.1,
            gamesPlayed: iter * 100
        };
    };

    const [metrics, setMetrics] = useState(generateMetrics(0));
    const history = [];
    for (let i = 0; i <= iteration; i += 5) {
        history.push({ iter: i, ...generateMetrics(i) });
    }

    useEffect(() => {
        if (isPlaying && iteration < maxIterations) {
            const timer = setTimeout(() => {
                setIteration(prev => Math.min(prev + 1, maxIterations));
                setMetrics(generateMetrics(iteration + 1));
            }, 100);
            return () => clearTimeout(timer);
        } else if (iteration >= maxIterations) {
            setIsPlaying(false);
        }
    }, [isPlaying, iteration]);

    const handleReset = () => {
        setIteration(0);
        setMetrics(generateMetrics(0));
        setIsPlaying(false);
    };

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-slate-900 dark:to-purple-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Self-Play Evolution
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Observe policy improvement through self-play training
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                {/* Win Rate Card */}
                <motion.div
                    className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg"
                    animate={{ scale: [1, 1.02, 1] }}
                    transition={{ duration: 0.5, repeat: Infinity, repeatDelay: 2 }}
                >
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-semibold text-slate-600 dark:text-slate-400">Win Rate</span>
                        <span className="text-xs bg-green-100 dark:bg-green-900/30 px-2 py-1 rounded">vs Previous Self</span>
                    </div>
                    <div className="text-4xl font-bold text-green-600 dark:text-green-400">
                        {(metrics.winRate * 100).toFixed(1)}%
                    </div>
                    <div className="mt-2 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                        <motion.div
                            className="h-full bg-gradient-to-r from-green-400 to-green-600"
                            initial={{ width: "50%" }}
                            animate={{ width: `${metrics.winRate * 100}%` }}
                            transition={{ duration: 0.5 }}
                        />
                    </div>
                </motion.div>

                {/* Elo Rating Card */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-semibold text-slate-600 dark:text-slate-400">Elo Rating</span>
                        <span className="text-xs bg-blue-100 dark:bg-blue-900/30 px-2 py-1 rounded">Strength</span>
                    </div>
                    <div className="text-4xl font-bold text-blue-600 dark:text-blue-400">
                        {Math.round(metrics.eloRating)}
                    </div>
                    <div className="text-xs text-slate-500 mt-1">
                        +{Math.round(metrics.eloRating - 1000)} from baseline
                    </div>
                </div>

                {/* Policy Diversity Card */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-semibold text-slate-600 dark:text-slate-400">Policy Diversity</span>
                        <span className="text-xs bg-purple-100 dark:bg-purple-900/30 px-2 py-1 rounded">Entropy</span>
                    </div>
                    <div className="text-4xl font-bold text-purple-600 dark:text-purple-400">
                        {metrics.policyDiversity.toFixed(2)}
                    </div>
                    <div className="text-xs text-slate-500 mt-1">
                        {metrics.policyDiversity > 0.5 ? "High exploration" : "Converged"}
                    </div>
                </div>

                {/* Games Played Card */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-semibold text-slate-600 dark:text-slate-400">Games Played</span>
                        <span className="text-xs bg-orange-100 dark:bg-orange-900/30 px-2 py-1 rounded">Sample Count</span>
                    </div>
                    <div className="text-4xl font-bold text-orange-600 dark:text-orange-400">
                        {metrics.gamesPlayed.toLocaleString()}
                    </div>
                    <div className="text-xs text-slate-500 mt-1">
                        Iteration {iteration}/{maxIterations}
                    </div>
                </div>
            </div>

            {/* Win Rate History Chart */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Training Progress Curve</h4>
                <div className="relative h-48">
                    <svg className="w-full h-full">
                        {/* Grid lines */}
                        {[0, 0.25, 0.5, 0.75, 1].map(y => (
                            <line
                                key={y}
                                x1="0" y1={`${(1 - y) * 100}%`}
                                x2="100%" y2={`${(1 - y) * 100}%`}
                                stroke="currentColor"
                                className="text-gray-300 dark:text-gray-700"
                                strokeWidth="1"
                                strokeDasharray="4"
                            />
                        ))}

                        {/* Win rate line */}
                        <polyline
                            points={history.map((h, i) =>
                                `${(h.iter / maxIterations) * 100},${(1 - h.winRate) * 100}`
                            ).join(' ')}
                            fill="none"
                            stroke="rgb(34, 197, 94)"
                            strokeWidth="3"
                        />

                        {/* Current point */}
                        <motion.circle
                            cx={`${(iteration / maxIterations) * 100}%`}
                            cy={`${(1 - metrics.winRate) * 100}%`}
                            r="5"
                            fill="rgb(34, 197, 94)"
                            animate={{ scale: [1, 1.3, 1] }}
                            transition={{ duration: 0.5, repeat: Infinity }}
                        />
                    </svg>
                    {/* Y-axis labels */}
                    <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-slate-600 dark:text-slate-400 -ml-12">
                        <span>100%</span>
                        <span>75%</span>
                        <span>50%</span>
                        <span>25%</span>
                        <span>0%</span>
                    </div>
                </div>
            </div>

            {/* Controls */}
            <div className="flex gap-4 justify-center">
                <motion.button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="px-6 py-3 bg-gradient-to-r from-purple-500 to-indigo-600 text-white rounded-lg font-semibold shadow-lg"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                >
                    {isPlaying ? "⏸ Pause" : "▶ Play"}
                </motion.button>

                <motion.button
                    onClick={handleReset}
                    className="px-6 py-3 bg-gray-500 text-white rounded-lg font-semibold shadow-lg"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                >
                    ↻ Reset
                </motion.button>

                <div className="flex items-center gap-2">
                    <label className="text-sm font-semibold">Iteration:</label>
                    <input
                        type="range"
                        min="0"
                        max={maxIterations}
                        value={iteration}
                        onChange={(e) => {
                            const val = parseInt(e.target.value);
                            setIteration(val);
                            setMetrics(generateMetrics(val));
                        }}
                        className="w-48 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                    />
                    <span className="text-sm font-mono w-12">{iteration}</span>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-600 dark:text-slate-400">
                💡 Self-play creates an automatic curriculum: as the agent improves, opponents get stronger
            </div>
        </div>
    );
}
