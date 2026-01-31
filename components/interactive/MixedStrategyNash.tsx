"use client";

import { motion } from "framer-motion";

export function MixedStrategyNash() {
    const payoff = [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0]
    ];

    const actions = ["Rock", "Paper", "Scissors"];
    const nashProb = 1 / 3;

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-teal-50 to-cyan-50 dark:from-slate-900 dark:to-teal-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Mixed Strategy Nash Equilibrium
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Rock-Paper-Scissors: No pure Nash, only mixed
                </p>
            </div>

            {/* Payoff Matrix */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Payoff Matrix (Player 1)</h4>
                <table className="w-full border-collapse">
                    <thead>
                        <tr>
                            <th className="p-2 border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800"></th>
                            {actions.map(action => (
                                <th key={action} className="p-2 border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800 font-semibold">
                                    {action}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {actions.map((rowAction, i) => (
                            <tr key={rowAction}>
                                <td className="p-2 border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800 font-semibold">
                                    {rowAction}
                                </td>
                                {payoff[i].map((value, j) => (
                                    <td
                                        key={j}
                                        className={`p-4 border border-gray-300 dark:border-gray-700 text-center font-mono ${value > 0 ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400" :
                                                value < 0 ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400" :
                                                    "bg-gray-50 dark:bg-gray-700 text-gray-600 dark:text-gray-400"
                                            }`}
                                    >
                                        {value > 0 ? `+${value}` : value}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Nash Equilibrium Strategy */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Nash Equilibrium (Uniform Random)</h4>
                <div className="grid grid-cols-3 gap-4">
                    {actions.map((action, idx) => {
                        const colors = ["from-red-500 to-red-600", "from-blue-500 to-blue-600", "from-green-500 to-green-600"];

                        return (
                            <motion.div
                                key={action}
                                className={`bg-gradient-to-br ${colors[idx]} text-white p-6 rounded-xl shadow-lg`}
                                animate={{ scale: [1, 1.05, 1] }}
                                transition={{ duration: 2, repeat: Infinity, delay: idx * 0.3 }}
                            >
                                <div className="text-center">
                                    <div className="text-3xl mb-2">
                                        {idx === 0 ? "✊" : idx === 1 ? "✋" : "✌️"}
                                    </div>
                                    <div className="font-bold text-lg mb-2">{action}</div>
                                    <div className="text-4xl font-mono font-bold">
                                        {(nashProb * 100).toFixed(1)}%
                                    </div>
                                    <div className="text-xs mt-2 opacity-80">
                                        Probability
                                    </div>
                                </div>
                            </motion.div>
                        );
                    })}
                </div>
            </div>

            {/* Why Mixed? */}
            <div className="bg-gradient-to-r from-teal-100 to-cyan-100 dark:from-teal-900/30 dark:to-cyan-900/30 rounded-xl p-6 border-2 border-teal-300 dark:border-teal-700 mb-6">
                <h4 className="text-lg font-bold mb-3">Why Mixed Strategy?</h4>
                <div className="space-y-2 text-sm">
                    <div className="flex items-start gap-2">
                        <span className="text-teal-600 dark:text-teal-400 font-bold">1.</span>
                        <div>
                            <strong>No Pure Nash:</strong> If you always play Rock, opponent plays Paper and wins.
                        </div>
                    </div>
                    <div className="flex items-start gap-2">
                        <span className="text-teal-600 dark:text-teal-400 font-bold">2.</span>
                        <div>
                            <strong>Intransitive:</strong> Rock beats Scissors, Scissors beats Paper, Paper beats Rock → cycle!
                        </div>
                    </div>
                    <div className="flex items-start gap-2">
                        <span className="text-teal-600 dark:text-teal-400 font-bold">3.</span>
                        <div>
                            <strong>Unpredictability:</strong> Uniform random makes you unexploitable.
                        </div>
                    </div>
                </div>
            </div>

            {/* Intransitivity Cycle */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-center">Intransitivity Cycle</h4>
                <div className="relative h-64 flex items-center justify-center">
                    <svg className="w-full h-full" viewBox="0 0 200 200">
                        {/* Triangle */}
                        <circle cx="100" cy="40" r="25" fill="#ef4444" />
                        <text x="100" y="48" textAnchor="middle" fill="white" fontSize="20">✊</text>

                        <circle cx="40" cy="160" r="25" fill="#3b82f6" />
                        <text x="40" y="168" textAnchor="middle" fill="white" fontSize="20">✋</text>

                        <circle cx="160" cy="160" r="25" fill="#22c55e" />
                        <text x="160" y="168" textAnchor="middle" fill="white" fontSize="20">✌️</text>

                        {/* Arrows */}
                        <defs>
                            <marker id="arrowhead2" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                                <polygon points="0 0, 10 3, 0 6" fill="#64748b" />
                            </marker>
                        </defs>

                        {/* Rock -> Scissors */}
                        <path d="M 120 55 Q 150 100 145 135" stroke="#64748b" strokeWidth="3" fill="none" markerEnd="url(#arrowhead2)" />
                        <text x="145" y="95" fontSize="12" fill="#64748b">beats</text>

                        {/* Scissors -> Paper */}
                        <path d="M 140 145 Q 90 170 65 160" stroke="#64748b" strokeWidth="3" fill="none" markerEnd="url(#arrowhead2)" />
                        <text x="100" y="175" fontSize="12" fill="#64748b">beats</text>

                        {/* Paper -> Rock */}
                        <path d="M 55 140 Q 70 90 85 60" stroke="#64748b" strokeWidth="3" fill="none" markerEnd="url(#arrowhead2)" />
                        <text x="50" y="95" fontSize="12" fill="#64748b">beats</text>
                    </svg>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-600 dark:text-slate-400">
                💡 In intransitive games, randomization prevents exploitation
            </div>
        </div>
    );
}
