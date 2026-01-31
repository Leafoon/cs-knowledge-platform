"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function CFRAlgorithm() {
    const [iteration, setIteration] = useState(0);
    const maxIterations = 50;

    // Simulate CFR convergence for Kuhn Poker
    // Information sets: J, Q, K (first action), then with history (Jb, Qc, etc.)
    const getRegrets = (iter: number) => {
        const progress = iter / maxIterations;

        return {
            "J": {
                check: 0.2 * (1 - progress) + Math.random() * 0.1,
                bet: -0.2 * (1 - progress) + Math.random() * 0.1
            },
            "Q": {
                check: -0.1 + 0.3 * progress + Math.random() * 0.1,
                bet: 0.1 - 0.2 * progress + Math.random() * 0.1
            },
            "K": {
                check: -0.4 * (1 - progress) + Math.random() * 0.1,
                bet: 0.4 * (1 - progress) + Math.random() * 0.1
            }
        };
    };

    const getStrategy = (regrets: { check: number, bet: number }) => {
        const checkReg = Math.max(0, regrets.check);
        const betReg = Math.max(0, regrets.bet);
        const sum = checkReg + betReg;

        if (sum > 0) {
            return {
                check: checkReg / sum,
                bet: betReg / sum
            };
        }
        return { check: 0.5, bet: 0.5 };
    };

    const regrets = getRegrets(iteration);
    const strategies = {
        J: getStrategy(regrets.J),
        Q: getStrategy(regrets.Q),
        K: getStrategy(regrets.K)
    };

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    CFR Algorithm (Kuhn Poker)
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Watch strategies converge to Nash equilibrium
                </p>
            </div>

            {/* Iteration Control */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="flex items-center gap-4 mb-4">
                    <span className="font-semibold">Iteration:</span>
                    <input
                        type="range"
                        min="0"
                        max={maxIterations}
                        value={iteration}
                        onChange={(e) => setIteration(parseInt(e.target.value))}
                        className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                    />
                    <span className="font-mono w-16 text-right">{iteration}/{maxIterations}</span>
                </div>
                <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div
                        className="h-full bg-gradient-to-r from-indigo-500 to-blue-600"
                        style={{ width: `${(iteration / maxIterations) * 100}%` }}
                    />
                </div>
            </div>

            {/* Information Sets & Strategies */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                {["J", "Q", "K"].map((card, idx) => {
                    const strategy = strategies[card as keyof typeof strategies];
                    const regret = regrets[card as keyof typeof regrets];
                    const colors = ["bg-red-500", "bg-yellow-500", "bg-green-500"];

                    return (
                        <motion.div
                            key={card}
                            className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: idx * 0.1 }}
                        >
                            <div className="text-center mb-4">
                                <div className={`inline-block ${colors[idx]} text-white px-4 py-2 rounded-lg text-2xl font-bold mb-2`}>
                                    {card}
                                </div>
                                <div className="text-xs text-slate-600 dark:text-slate-400">
                                    Card: {card === "J" ? "Jack" : card === "Q" ? "Queen" : "King"}
                                </div>
                            </div>

                            <div className="space-y-3">
                                <div>
                                    <div className="flex justify-between text-sm mb-1">
                                        <span>Check</span>
                                        <span className="font-mono">{(strategy.check * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                        <motion.div
                                            className="h-full bg-blue-500"
                                            animate={{ width: `${strategy.check * 100}%` }}
                                            transition={{ duration: 0.3 }}
                                        />
                                    </div>
                                    <div className="text-xs text-slate-500 mt-1">
                                        Regret: {regret.check.toFixed(3)}
                                    </div>
                                </div>

                                <div>
                                    <div className="flex justify-between text-sm mb-1">
                                        <span>Bet</span>
                                        <span className="font-mono">{(strategy.bet * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                        <motion.div
                                            className="h-full bg-red-500"
                                            animate={{ width: `${strategy.bet * 100}%` }}
                                            transition={{ duration: 0.3 }}
                                        />
                                    </div>
                                    <div className="text-xs text-slate-500 mt-1">
                                        Regret: {regret.bet.toFixed(3)}
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    );
                })}
            </div>

            {/* Nash Equilibrium Insights */}
            <div className="bg-gradient-to-r from-indigo-100 to-blue-100 dark:from-indigo-900/30 dark:to-blue-900/30 rounded-xl p-6 border-2 border-indigo-300 dark:border-indigo-700">
                <h4 className="text-lg font-bold mb-3">Nash Equilibrium Strategy</h4>
                <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                        <div className="font-semibold mb-1">Jack (Worst card)</div>
                        <div className="text-xs text-slate-600 dark:text-slate-400">
                            {iteration > 30
                                ? "✓ Check always (can't win showdown)"
                                : "Still learning..."}
                        </div>
                    </div>
                    <div>
                        <div className="font-semibold mb-1">Queen (Middle card)</div>
                        <div className="text-xs text-slate-600 dark:text-slate-400">
                            {iteration > 30
                                ? "✓ Mixed: bluff occasionally"
                                : "Exploring strategies..."}
                        </div>
                    </div>
                    <div>
                        <div className="font-semibold mb-1">King (Best card)</div>
                        <div className="text-xs text-slate-600 dark:text-slate-400">
                            {iteration > 30
                                ? "✓ Bet always (value bet)"
                                : "Finding optimal play..."}
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-600 dark:text-slate-400">
                💡 CFR minimizes regret by tracking counterfactual value of each action
            </div>
        </div>
    );
}
