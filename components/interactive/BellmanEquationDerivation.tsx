"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";

export function BellmanEquationDerivation() {
    const [step, setStep] = useState(0);

    const steps = [
        {
            latex: "V_\\pi(s) = \\mathbb{E}[G_t | S_t = s]",
            desc: "价值函数的定义：从状态 s 出发的期望回报。",
            highlight: "G_t"
        },
        {
            latex: "V_\\pi(s) = \\mathbb{E}[R_{t+1} + \\gamma R_{t+2} + \\gamma^2 R_{t+3} + ... | S_t = s]",
            desc: "展开期望回报 G_t。",
            highlight: "Expand"
        },
        {
            latex: "V_\\pi(s) = \\mathbb{E}[R_{t+1} + \\gamma (R_{t+2} + \\gamma R_{t+3} + ...) | S_t = s]",
            desc: "提取公因式 γ。",
            highlight: "Factor"
        },
        {
            latex: "V_\\pi(s) = \\mathbb{E}[R_{t+1} + \\gamma G_{t+1} | S_t = s]",
            desc: "识别出括号内即为 G_{t+1}。",
            highlight: "Recursive"
        },
        {
            latex: "V_\\pi(s) = \\sum_a \\pi(a|s) \\sum_{s',r} p(s',r|s,a) [r + \\gamma V_\\pi(s')]",
            desc: "将期望展开为概率求和形式 (Bellman 方程)。",
            highlight: "Final"
        }
    ];

    return (
        <div className="w-full max-w-3xl mx-auto p-8 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800">
            <h3 className="text-xl font-bold text-center mb-8 text-slate-800 dark:text-slate-100">
                Bellman 方程推导 (Step-by-Step)
            </h3>

            <div className="flex flex-col items-center space-y-6 min-h-[200px] justify-center">
                <AnimatePresence mode="wait">
                    <motion.div
                        key={step}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="text-2xl md:text-3xl font-serif text-slate-800 dark:text-slate-200 text-center px-4"
                    >
                        {/* We use specific Unicode math or just text representation for simplicity in this React component if not using MathJax. 
                            For better rendering, assuming MathJax is active on page or using simple characters. 
                            Here we simulate Math look with font-serif. 
                         */}
                        {/* Simple HTML rendering of the LaTeX for the demo */}
                        {step === 0 && <span>V<sub>π</sub>(s) = E [ G<sub>t</sub> | S<sub>t</sub>=s ]</span>}
                        {step === 1 && <span>V<sub>π</sub>(s) = E [ R<sub>t+1</sub> + γR<sub>t+2</sub> + ... | S<sub>t</sub>=s ]</span>}
                        {step === 2 && <span>V<sub>π</sub>(s) = E [ R<sub>t+1</sub> + γ(R<sub>t+2</sub> + ...) | S<sub>t</sub>=s ]</span>}
                        {step === 3 && <span>V<sub>π</sub>(s) = E [ R<sub>t+1</sub> + γG<sub>t+1</sub> | S<sub>t</sub>=s ]</span>}
                        {step === 4 && <span className="text-xl">V<sub>π</sub>(s) = ∑<sub>a</sub> π(a|s) ∑<sub>s',r</sub> p(s',r|s,a) [ r + γV<sub>π</sub>(s') ]</span>}
                    </motion.div>
                </AnimatePresence>

                <div className="text-sm text-slate-500 dark:text-slate-400 h-8">
                    {steps[step].desc}
                </div>
            </div>

            <div className="flex justify-center gap-4 mt-8">
                <button
                    onClick={() => setStep(Math.max(0, step - 1))}
                    disabled={step === 0}
                    className="px-4 py-2 rounded bg-slate-200 dark:bg-slate-700 disabled:opacity-50"
                >
                    Prev
                </button>
                <div className="flex gap-1 items-center">
                    {steps.map((_, i) => (
                        <div key={i} className={`w-2 h-2 rounded-full ${i === step ? "bg-blue-500" : "bg-slate-300"}`} />
                    ))}
                </div>
                <button
                    onClick={() => setStep(Math.min(steps.length - 1, step + 1))}
                    disabled={step === steps.length - 1}
                    className="px-4 py-2 rounded bg-blue-600 text-white disabled:opacity-50"
                >
                    Next
                </button>
            </div>
        </div>
    );
}
