"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function AdversarialAttackDemo() {
    const [attackMethod, setAttackMethod] = useState<"fgsm" | "pgd" | "random">("fgsm");
    const [epsilon, setEpsilon] = useState(0.1);
    const [isAttacking, setIsAttacking] = useState(false);

    // Simulated state (grid world)
    const gridSize = 8;
    const agentPos = { x: 3, y: 3 };
    const goalPos = { x: 6, y: 6 };

    // Simulated policy action probabilities (clean vs adversarial)
    const cleanProbs = [0.05, 0.15, 0.70, 0.10]; // [UP, DOWN, RIGHT, LEFT]
    const advProbs = [0.15, 0.65, 0.10, 0.10]; // Attacked - wrong action!

    const currentProbs = isAttacking ? advProbs : cleanProbs;
    const actions = ["↑ UP", "↓ DOWN", "→ RIGHT", "← LEFT"];

    // Calculate success rate
    const cleanSuccess = 92;
    const advSuccess = Math.max(10, cleanSuccess - epsilon * 400);

    const attackMethods = [
        { id: "fgsm", name: "FGSM", desc: "ε·sign(∇_s J)", color: "red" },
        { id: "pgd", name: "PGD", desc: "Iterative FGSM", color: "orange" },
        { id: "random", name: "Random", desc: "Baseline", color: "gray" }
    ];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-red-50 to-orange-50 dark:from-slate-900 dark:to-red-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    对抗攻击演示
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Adversarial Attacks on RL Policies
                </p>
            </div>

            {/* Attack Method Selection */}
            <div className="flex gap-3 justify-center mb-6">
                {attackMethods.map((method) => (
                    <button
                        key={method.id}
                        onClick={() => setAttackMethod(method.id as any)}
                        className={`px-6 py-3 rounded-xl font-semibold transition ${attackMethod === method.id
                                ? `bg-${method.color}-600 text-white shadow-lg`
                                : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                            }`}
                    >
                        <div className="text-sm">{method.name}</div>
                        <div className="text-xs opacity-75">{method.desc}</div>
                    </button>
                ))}
            </div>

            {/* Epsilon Slider */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg mb-6">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                        扰动强度 ε
                    </span>
                    <span className="text-lg font-bold text-red-600 dark:text-red-400">
                        {epsilon.toFixed(3)}
                    </span>
                </div>
                <input
                    type="range"
                    min="0"
                    max="0.3"
                    step="0.01"
                    value={epsilon}
                    onChange={(e) => setEpsilon(Number(e.target.value))}
                    className="w-full"
                />
                <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    典型值: ε ∈ [0.01, 0.3]
                </div>
            </div>

            {/* Grid Visualization */}
            <div className="grid grid-cols-2 gap-4 mb-6">
                {/* Clean State */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg">
                    <h4 className="text-sm font-bold mb-3 text-slate-800 dark:text-slate-100 text-center">
                        原始状态
                    </h4>
                    <div className="grid grid-cols-8 gap-1 mb-3">
                        {Array.from({ length: gridSize * gridSize }).map((_, idx) => {
                            const x = idx % gridSize;
                            const y = Math.floor(idx / gridSize);
                            const isAgent = x === agentPos.x && y === agentPos.y;
                            const isGoal = x === goalPos.x && y === goalPos.y;

                            return (
                                <div
                                    key={idx}
                                    className={`aspect-square rounded ${isAgent
                                            ? "bg-blue-500"
                                            : isGoal
                                                ? "bg-green-500"
                                                : "bg-gray-200 dark:bg-gray-700"
                                        }`}
                                />
                            );
                        })}
                    </div>
                    <div className="text-xs text-center text-slate-600 dark:text-slate-400">
                        <span className="inline-block w-3 h-3 bg-blue-500 rounded mr-1"></span> Agent
                        <span className="inline-block w-3 h-3 bg-green-500 rounded ml-3 mr-1"></span> Goal
                    </div>
                </div>

                {/* Adversarial State */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg border-2 border-red-500">
                    <h4 className="text-sm font-bold mb-3 text-slate-800 dark:text-slate-100 text-center">
                        对抗状态
                    </h4>
                    <div className="grid grid-cols-8 gap-1 mb-3">
                        {Array.from({ length: gridSize * gridSize }).map((_, idx) => {
                            const x = idx % gridSize;
                            const y = Math.floor(idx / gridSize);
                            const isAgent = x === agentPos.x && y === agentPos.y;
                            const isGoal = x === goalPos.x && y === goalPos.y;

                            // Add noise visualization
                            const noise = isAttacking && Math.random() < epsilon * 3 ? 40 : 0;

                            return (
                                <motion.div
                                    key={idx}
                                    className={`aspect-square rounded ${isAgent
                                            ? "bg-blue-500"
                                            : isGoal
                                                ? "bg-green-500"
                                                : "bg-gray-200 dark:bg-gray-700"
                                        }`}
                                    style={{
                                        filter: isAttacking ? `brightness(${1 + noise / 100})` : "none"
                                    }}
                                    animate={{
                                        opacity: isAttacking && noise > 0 ? [1, 0.7, 1] : 1
                                    }}
                                    transition={{ duration: 0.5, repeat: Infinity }}
                                />
                            );
                        })}
                    </div>
                    <div className="text-xs text-center text-red-600 dark:text-red-400 font-semibold">
                        +ε·sign(∇_s J) 扰动
                    </div>
                </div>
            </div>

            {/* Policy Output Comparison */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    策略输出对比
                </h4>

                <div className="grid grid-cols-4 gap-3">
                    {actions.map((action, idx) => {
                        const cleanProb = cleanProbs[idx];
                        const advProb = currentProbs[idx];
                        const isOptimal = cleanProb === Math.max(...cleanProbs);
                        const isAttackedChoice = isAttacking && advProb === Math.max(...advProbs);

                        return (
                            <div key={action} className="text-center">
                                <div className="text-sm font-semibold mb-2 text-slate-700 dark:text-slate-300">
                                    {action}
                                </div>

                                {/* Clean Bar */}
                                <div className="mb-2">
                                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">
                                        Clean
                                    </div>
                                    <div className="h-24 bg-gray-200 dark:bg-gray-700 rounded relative overflow-hidden">
                                        <motion.div
                                            className={`absolute bottom-0 w-full ${isOptimal ? "bg-green-500" : "bg-blue-400"
                                                }`}
                                            initial={{ height: 0 }}
                                            animate={{ height: `${cleanProb * 100}%` }}
                                            transition={{ duration: 0.5 }}
                                        />
                                        <div className="absolute inset-0 flex items-center justify-center text-xs font-bold text-white">
                                            {(cleanProb * 100).toFixed(0)}%
                                        </div>
                                    </div>
                                </div>

                                {/* Adversarial Bar */}
                                {isAttacking && (
                                    <div>
                                        <div className="text-xs text-red-600 dark:text-red-400 mb-1">
                                            Adversarial
                                        </div>
                                        <div className="h-24 bg-gray-200 dark:bg-gray-700 rounded relative overflow-hidden">
                                            <motion.div
                                                className={`absolute bottom-0 w-full ${isAttackedChoice ? "bg-red-600" : "bg-orange-400"
                                                    }`}
                                                initial={{ height: 0 }}
                                                animate={{ height: `${advProb * 100}%` }}
                                                transition={{ duration: 0.5 }}
                                            />
                                            <div className="absolute inset-0 flex items-center justify-center text-xs font-bold text-white">
                                                {(advProb * 100).toFixed(0)}%
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Attack Control */}
            <div className="flex gap-4 justify-center mb-6">
                <button
                    onClick={() => setIsAttacking(!isAttacking)}
                    className={`px-8 py-3 rounded-lg font-semibold transition shadow-lg ${isAttacking
                            ? "bg-red-600 text-white hover:bg-red-700"
                            : "bg-green-600 text-white hover:bg-green-700"
                        }`}
                >
                    {isAttacking ? "停止攻击" : "发起攻击"}
                </button>
            </div>

            {/* Performance Impact */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    性能影响
                </h4>

                <div className="grid grid-cols-3 gap-4">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                        <div className="text-sm text-green-700 dark:text-green-400 mb-1">
                            Clean Success
                        </div>
                        <div className="text-3xl font-bold text-green-600 dark:text-green-400">
                            {cleanSuccess}%
                        </div>
                    </div>

                    <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
                        <div className="text-sm text-red-700 dark:text-red-400 mb-1">
                            Adversarial Success
                        </div>
                        <div className="text-3xl font-bold text-red-600 dark:text-red-400">
                            {advSuccess.toFixed(0)}%
                        </div>
                    </div>

                    <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                        <div className="text-sm text-orange-700 dark:text-orange-400 mb-1">
                            Performance Drop
                        </div>
                        <div className="text-3xl font-bold text-orange-600 dark:text-orange-400">
                            {(cleanSuccess - advSuccess).toFixed(0)}%
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-red-100 dark:bg-red-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                ⚠️ <strong>防御策略</strong>: 对抗训练、Input Transformation、Certified Defense、Ensemble方法
            </div>
        </div>
    );
}
