"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/Card";
import { useState, useEffect } from "react";
import { InlineMath } from "@/components/ui/Math";

export function TDUpdateVisualizer() {
    const [step, setStep] = useState(0);
    const [value, setValue] = useState(0.0);
    const targetValue = 1.0; // The "True" value we are approaching
    const alpha = 0.5; // Learning rate
    const reward = 0.2;
    const gamma = 0.9;

    // Theoretical update: V(s) <- V(s) + alpha * (r + gamma * V(s') - V(s))
    // Let's simplify visualization: Just V(s) moving towards Target (R + gamma*V(s'))

    const [history, setHistory] = useState<number[]>([0]);

    useEffect(() => {
        if (step === 0) {
            setValue(0);
            setHistory([0]);
        }
    }, [step]);

    const performUpdate = () => {
        // Simple TD(0) style update simulation for visualization
        // Target is closer to 1.0
        const currentV = history[history.length - 1];
        // Assume TD Target is roughly fixed for this demo to show convergence to a target
        const tdTarget = 1.0;
        const error = tdTarget - currentV;
        const newV = currentV + alpha * error;

        if (newV < 1.05) { // Stop if converged
            setHistory([...history, newV]);
            setValue(newV);
        }
    };

    const reset = () => {
        setHistory([0]);
        setValue(0);
    };

    return (
        <Card className="p-6 bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-800">
            <h3 className="text-xl font-bold mb-4">TD(0) 更新直观演示 (Bootstrapping)</h3>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-6">
                TD 不需要等到 Episode 结束。每走一步，它就利用当前的奖励 <InlineMath>R</InlineMath> 和对下一步状态价值的<strong>估计</strong> <InlineMath>{"V(S')"}</InlineMath> 来更新当前状态价值 <InlineMath>V(S)</InlineMath>。
                <br />
                <span className="font-mono text-purple-600">Target = R + <InlineMath>{"\\gamma"}</InlineMath>V(S')</span>
            </p>

            <div className="flex flex-col items-center gap-8">
                {/* Visual Bar */}
                <div className="relative w-full h-24 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-sm">
                    {/* Goal Line */}
                    <div className="absolute right-8 top-0 bottom-0 w-1 bg-green-500 z-10 opacity-50"></div>
                    <div className="absolute right-8 bottom-2 text-xs font-bold text-green-600">TD Target</div>

                    {/* Current Value Bar */}
                    <motion.div
                        className="absolute left-0 top-4 bottom-4 bg-blue-500 rounded-r-lg flex items-center justify-end px-2"
                        initial={{ width: "5%" }}
                        animate={{ width: `${(value / 1.1) * 100}%` }} // Scale roughly
                        transition={{ type: "spring", stiffness: 100 }}
                    >
                        <span className="text-white font-bold text-sm whitespace-nowrap">V(S) = {value.toFixed(2)}</span>
                    </motion.div>

                    {/* TD Error gap */}
                    <motion.div
                        className="absolute h-1 bg-red-400 top-1/2 -translate-y-1/2"
                        initial={{ left: "5%", right: "10%" }}
                        animate={{
                            left: `${(value / 1.1) * 100}%`,
                            right: "8%" // Approx target pos
                        }}
                    />
                    {history.length < 8 && (
                        <motion.div
                            className="absolute top-1/2 -translate-y-6 text-xs text-red-500 font-mono"
                            animate={{ left: `${((value + (1 - value) / 2) / 1.1) * 100}%` }}
                        >
                            TD Error (<InlineMath>{"\\delta"}</InlineMath>)
                        </motion.div>
                    )}
                </div>

                <div className="flex gap-4">
                    <motion.button
                        whileTap={{ scale: 0.95 }}
                        onClick={performUpdate}
                        disabled={value > 0.99}
                        className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg shadow-md font-bold disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        执行一步 update
                    </motion.button>
                    <button
                        onClick={reset}
                        className="px-4 py-2 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
                    >
                        重置
                    </button>
                </div>

                <div className="w-full grid grid-cols-2 gap-4 text-sm font-mono mt-4">
                    <div className="bg-white dark:bg-slate-800 p-3 rounded border border-slate-200 dark:border-slate-700">
                        <div className="text-slate-500 mb-1">Update Rule</div>
                        <div>V ← V + <InlineMath>{"\\alpha"}</InlineMath> [Target - V]</div>
                    </div>
                    <div className="bg-white dark:bg-slate-800 p-3 rounded border border-slate-200 dark:border-slate-700">
                        <div className="text-slate-500 mb-1">Estimation</div>
                        <div>{value.toFixed(3)} <InlineMath>{"\\to"}</InlineMath> 1.000</div>
                    </div>
                </div>
            </div>
        </Card>
    );
}
