"use client";

import { InlineMath } from "@/components/ui/Math";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Slider } from "@/components/ui/Slider";

export function MCReturnEstimation() {
    const [gamma, setGamma] = useState(0.9);
    const [selectedStep, setSelectedStep] = useState<number | null>(null);

    // 模拟一个 Episode 的轨迹
    const episode = [
        { t: 0, s: "S1", a: "Up", r: 0 },
        { t: 1, s: "S2", a: "Right", r: 0 },
        { t: 2, s: "S3", a: "Right", r: 1 },
        { t: 3, s: "S4", a: "Up", r: 0 },
        { t: 4, s: "S5", a: "Right", r: 5 },
        { t: 5, s: "S6", a: "Stop", r: 0 }, // Terminal
    ];

    // 计算 Gt
    const calculateReturn = (t: number) => {
        let G = 0;
        let terms = [];
        for (let k = t; k < episode.length - 1; k++) {
            const r = episode[k + 1].r;
            const power = k - t;
            const discount = Math.pow(gamma, power);
            G += discount * r;

            terms.push({
                r,
                power,
                discount: discount.toFixed(2),
                val: (discount * r).toFixed(2)
            });
        }
        return { G: G.toFixed(2), terms };
    };

    const calculation = selectedStep !== null ? calculateReturn(selectedStep) : null;

    return (
        <Card className="p-6 w-full max-w-4xl mx-auto bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-800">
            <div className="mb-6">
                <h3 className="text-lg font-bold mb-2">Monte Carlo Return Calculation (<InlineMath>G_t</InlineMath>)</h3>
                <p className="text-sm text-slate-500 mb-4">
                    点击下方时间轴上的某个时间步 <InlineMath>t</InlineMath>，查看累积回报 <InlineMath>G_t</InlineMath> 的计算过程。
                    <InlineMath>{"G_t = R_{t+1} + \\gamma R_{t+2} + \\gamma^2 R_{t+3} + \\dots"}</InlineMath>
                </p>

                <div className="flex items-center space-x-4 mb-8">
                    <span className="text-sm font-mono flex items-center gap-2">Discount Factor <InlineMath>{"\\gamma"}</InlineMath> = {gamma}</span>
                    <Slider
                        value={[gamma]}
                        onValueChange={(v) => setGamma(v[0])}
                        min={0}
                        max={1}
                        step={0.1}
                        className="w-48"
                    />
                </div>

                {/* Timeline */}
                <div className="relative flex justify-between items-center h-24 mt-8 px-4">
                    {/* Line */}
                    <div className="absolute left-0 right-0 h-1 bg-slate-200 dark:bg-slate-700 top-1/2 -translate-y-1/2 -z-0" />

                    {episode.map((step, idx) => (
                        <div key={idx} className="relative z-10 flex flex-col items-center group">
                            <motion.button
                                onClick={() => setSelectedStep(step.t)}
                                className={`w-12 h-12 rounded-full flex items-center justify-center border-4 transition-all ${selectedStep === step.t
                                    ? "bg-blue-500 border-blue-200 text-white scale-110 shadow-lg"
                                    : selectedStep !== null && step.t > selectedStep
                                        ? "bg-green-100 border-green-500 text-slate-700"
                                        : "bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-500 hover:border-blue-400"
                                    }`}
                                whileHover={{ scale: 1.1 }}
                                whileTap={{ scale: 0.95 }}
                                disabled={idx === episode.length - 1} // Terminal state has no Gt in this simplified view
                            >
                                <span className="font-bold text-sm">t={step.t}</span>
                            </motion.button>

                            {/* Info */}
                            <div className="absolute top-14 text-center w-24">
                                <div className="text-xs font-bold text-slate-700 dark:text-slate-300">{step.s}</div>
                                {idx < episode.length - 1 && (
                                    <div className="text-xs text-slate-500">{step.a}</div>
                                )}
                            </div>

                            {/* Rewards */}
                            {idx > 0 && (
                                <div className="absolute -top-10 text-center">
                                    <div className={`text-xs font-mono px-2 py-1 rounded ${selectedStep !== null && step.t > selectedStep
                                        ? "bg-green-100 text-green-700 font-bold"
                                        : "text-slate-400"
                                        }`}>
                                        R={step.r}
                                    </div>
                                </div>
                            )}
                        </div>
                    ))}
                </div>

                {/* Calculation Detail */}
                <AnimatePresence mode="wait">
                    {calculation && selectedStep !== null && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            className="mt-12 p-6 bg-white dark:bg-slate-800 rounded-xl shadow-inner border border-slate-100 dark:border-slate-700"
                        >
                            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
                                <div className="flex-1 space-y-4">
                                    <h4 className="font-mono text-blue-600 dark:text-blue-400 text-lg">
                                        Calculate <InlineMath>{`G_${selectedStep}`}</InlineMath>
                                    </h4>
                                    <div className="font-mono text-sm space-y-2 text-slate-600 dark:text-slate-300">
                                        {calculation.terms.map((term, i) => (
                                            <div key={i} className="flex items-center gap-2">
                                                <span className="w-4 text-slate-400">+</span>
                                                <span className="text-orange-500">{term.discount}</span>
                                                <span className="text-slate-400">×</span>
                                                <span className="text-green-600 font-bold">{term.r}</span>
                                                <span className="text-slate-400">=</span>
                                                <span>{term.val}</span>
                                                <span className="text-xs text-slate-400 ml-2">
                                                    (<InlineMath>{`\\gamma^{${term.power}} \\times R_{${selectedStep + term.power + 1}}`}</InlineMath>)
                                                </span>
                                            </div>
                                        ))}
                                        <div className="h-px bg-slate-200 dark:bg-slate-600 my-2 w-48" />
                                        <div className="flex items-center gap-2 font-bold text-lg">
                                            <span>Total Return:</span>
                                            <span className="text-blue-600 dark:text-blue-400">{calculation.G}</span>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg text-sm text-blue-800 dark:text-blue-200 max-w-xs">
                                    <p className="mb-2 font-bold">Concept Note:</p>
                                    <p>
                                        MC 方法在 episode 结束后，沿着时间轴**反向（Backward）**计算所有 <InlineMath>G_t</InlineMath>。
                                        这就是为什么它必须用于**Episodic Tasks**。
                                    </p>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </Card>
    );
}
