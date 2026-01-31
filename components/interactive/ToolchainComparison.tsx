"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function ToolchainComparison() {
    const [selectedTool, setSelectedTool] = useState<string>("sb3");

    const tools = [
        {
            id: "sb3",
            name: "Stable-Baselines3",
            developer: "DLR-RM",
            focus: "Research & Education",
            pros: ["Simple API", "Well Documented", "Reliable PPO/SAC/DQN"],
            cons: ["PyTorch Only", "Hard to Scale (Single Machine)", "Less Modular"],
            color: "green",
            code: `model = PPO("MlpPolicy", env)\nmodel.learn(total_timesteps=10000)`
        },
        {
            id: "rllib",
            name: "Ray RLlib",
            developer: "Anyscale",
            focus: "Industry & Scaling",
            pros: ["Distributed Training", "Multi-Agent (MARL)", "Framework Agnostic"],
            cons: ["Steep Learning Curve", "Complex Configuration", "Heavy Dependency"],
            color: "blue",
            code: `config = PPOConfig().training(lr=0.01)\nalgo = config.build()\nalgo.train()`
        },
        {
            id: "acme",
            name: "Acme",
            developer: "DeepMind",
            focus: "Advanced Research",
            pros: ["Novel Architectures", "MCTS/Model-based", "JAX/TensorFlow"],
            cons: ["Less Documentation", "Research Code Quality", "Changing APIs"],
            color: "purple",
            code: `agent = dqn.DQN(environment_spec)\nloop = EnvironmentLoop(env, agent)\nloop.run()`
        },
        {
            id: "cleanrl",
            name: "CleanRL",
            developer: "Open Source",
            focus: "Hackability",
            pros: ["Single-file Implementations", "Easy to Modify", "Benchmark Reproducible"],
            cons: ["Copy-paste Reuse", "No Abstractions", "Boilerplate Code"],
            color: "orange",
            code: `# All logic in one file\n# ppo.py (300 lines)`
        }
    ];

    const currentTool = tools.find(t => t.id === selectedTool)!;

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-slate-900 dark:to-gray-950 rounded-2xl shadow-xl">
            <div className="text-center mb-8">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    主要 RL 工具链对比
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Choosing the Right Tool for the Job
                </p>
            </div>

            {/* Tool Selection Tabs */}
            <div className="flex flex-wrap gap-3 justify-center mb-8">
                {tools.map(tool => (
                    <button
                        key={tool.id}
                        onClick={() => setSelectedTool(tool.id)}
                        className={`px-6 py-3 rounded-lg font-bold transition-all ${selectedTool === tool.id
                                ? `bg-${tool.color}-600 text-white shadow-lg scale-105`
                                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-gray-50 dark:hover:bg-slate-700"
                            }`}
                    >
                        {tool.name}
                    </button>
                ))}
            </div>

            {/* Detailed Comparison View */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Info Card */}
                <motion.div
                    key={selectedTool}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="bg-white dark:bg-slate-800 rounded-2xl p-6 shadow-lg border-t-4"
                    style={{ borderColor: `var(--${currentTool.color}-500)` }}
                >
                    <div className="flex justify-between items-start mb-4">
                        <div>
                            <h4 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
                                {currentTool.name}
                            </h4>
                            <div className="text-sm text-slate-500 dark:text-slate-400">
                                by {currentTool.developer}
                            </div>
                        </div>
                        <div className={`px-3 py-1 rounded-full text-xs font-bold bg-${currentTool.color}-100 dark:bg-${currentTool.color}-900/30 text-${currentTool.color}-700 dark:text-${currentTool.color}-300`}>
                            {currentTool.focus}
                        </div>
                    </div>

                    <div className="grid grid-cols-1 gap-4 mb-6">
                        <div>
                            <h5 className="font-bold text-green-600 dark:text-green-400 mb-2 flex items-center gap-2">
                                ✅ Pros
                            </h5>
                            <ul className="space-y-1">
                                {currentTool.pros.map(p => (
                                    <li key={p} className="text-sm text-slate-700 dark:text-slate-300 flex items-start gap-2">
                                        <span className="text-green-500">•</span> {p}
                                    </li>
                                ))}
                            </ul>
                        </div>

                        <div>
                            <h5 className="font-bold text-red-600 dark:text-red-400 mb-2 flex items-center gap-2">
                                ❌ Cons
                            </h5>
                            <ul className="space-y-1">
                                {currentTool.cons.map(c => (
                                    <li key={c} className="text-sm text-slate-700 dark:text-slate-300 flex items-start gap-2">
                                        <span className="text-red-500">•</span> {c}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </motion.div>

                {/* Code Snapshot */}
                <motion.div
                    key={selectedTool + "code"}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex flex-col h-full"
                >
                    <div className="bg-slate-900 rounded-xl p-6 shadow-lg font-mono text-sm text-slate-200 h-full flex flex-col justify-center relative overflow-hidden">
                        <div className="absolute top-0 left-0 right-0 h-6 bg-slate-800 flex items-center px-4 gap-1.5">
                            <div className="w-3 h-3 rounded-full bg-red-500"></div>
                            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                            <div className="w-3 h-3 rounded-full bg-green-500"></div>
                            <div className="ml-2 text-xs text-slate-500">example.py</div>
                        </div>

                        <pre className="mt-4 whitespace-pre-wrap">
                            <code className="language-python">
                                {currentTool.code}
                            </code>
                        </pre>

                        <div className="mt-4 pt-4 border-t border-slate-700 text-xs text-slate-400 italic">
                            {selectedTool === "sb3" && "Most popular for start-ups & schools."}
                            {selectedTool === "rllib" && "Standard for massive scale clusters."}
                            {selectedTool === "acme" && "Best for implementing novel DeepMind papers."}
                            {selectedTool === "cleanrl" && "Best for understanding the algorithm internals."}
                        </div>
                    </div>
                </motion.div>
            </div>

            <div className="mt-8 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-xl text-center">
                <p className="text-slate-700 dark:text-slate-300 text-sm">
                    💡 <strong>Recommendation:</strong> Start with <strong className="text-green-600 dark:text-green-400">Stable-Baselines3</strong> for learning.
                    Switch to <strong className="text-blue-600 dark:text-blue-400">RLlib</strong> when you need distributed training.
                    Use <strong className="text-orange-600 dark:text-orange-400">CleanRL</strong> for custom research.
                </p>
            </div>
        </div>
    );
}
