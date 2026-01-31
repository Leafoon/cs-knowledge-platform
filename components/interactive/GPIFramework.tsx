"use client";

import { motion } from "framer-motion";

export function GPIFramework() {
    // A simplified visual of Generalized Policy Iteration
    // Two lines converging: Policy Evaluation (V -> V_pi) and Policy Improvement (pi -> greedy(V))

    return (
        <div className="w-full max-w-3xl mx-auto p-8 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800">
            <h3 className="text-xl font-bold text-center mb-8 text-slate-800 dark:text-slate-100">
                广义策略迭代 (GPI) 框架
            </h3>

            <div className="relative h-[300px] w-full flex items-center justify-center">
                {/* Top Line: Policy Improvement */}
                <svg className="absolute inset-0 w-full h-full">
                    <defs>
                        <marker id="arrow-gpi" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#64748b" />
                        </marker>
                    </defs>

                    {/* Zig-Zag pattern simulation */}
                    <path
                        d="M 100 250 L 150 50 L 200 220 L 250 80 L 300 180 L 350 120 L 400 150 L 500 150"
                        stroke="rgba(148, 163, 184, 0.2)"
                        strokeWidth="2"
                        fill="none"
                    />
                </svg>

                {/* Animated Policy Dot */}
                <motion.div
                    className="absolute w-4 h-4 rounded-full bg-blue-500 shadow-lg z-10"
                    animate={{
                        x: [100, 150, 200, 250, 300, 350, 400, 450], // x positions
                        y: [100, -100, 70, -70, 30, -30, 0, 0], // relative y offset from center
                    }}
                    style={{ top: "50%", left: 0 }} // Base position
                    transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                >
                    <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs font-bold text-blue-600 whitespace-nowrap">Policy π</div>
                </motion.div>

                {/* Animated Value Dot */}
                <motion.div
                    className="absolute w-4 h-4 rounded-full bg-green-500 shadow-lg z-10"
                    animate={{
                        x: [100, 150, 200, 250, 300, 350, 400, 450],
                        y: [-80, 80, -50, 50, -20, 20, 0, 0],
                    }}
                    style={{ top: "50%", left: 0 }}
                    transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 0.2 }}
                >
                    <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 text-xs font-bold text-green-600 whitespace-nowrap">Value V</div>
                </motion.div>

                {/* Target Optimal */}
                <div className="absolute right-10 top-1/2 -translate-y-1/2 w-16 h-16 rounded-full border-4 border-yellow-400 flex items-center justify-center bg-yellow-50 dark:bg-yellow-900/20 shadow-xl z-0">
                    <span className="text-2xl">★</span>
                </div>
                <div className="absolute right-8 bottom-1/3 text-xs font-bold text-yellow-600">Optimality</div>

                {/* Labels */}
                <div className="absolute left-10 top-10 text-xs text-slate-400">Policy Improvement</div>
                <div className="absolute left-10 bottom-10 text-xs text-slate-400">Policy Evaluation</div>

            </div>

            <p className="text-center text-sm text-slate-600 dark:text-slate-400 mt-4 max-w-lg mx-auto">
                GPI 描述了策略评估（V 趋向 Vπ）和策略改进（π 趋向 greedy(V)）这两个过程的相互作用。最终它们收敛于最优解。
            </p>
        </div>
    );
}
