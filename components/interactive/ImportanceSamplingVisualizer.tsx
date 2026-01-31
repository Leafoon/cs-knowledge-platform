"use client";

import { InlineMath } from "@/components/ui/Math";
import { useState } from "react";
import { motion } from "framer-motion";
import { Card } from "@/components/ui/Card";
import { Slider } from "@/components/ui/Slider";

export function ImportanceSamplingVisualizer() {
    const [targetProb, setTargetProb] = useState(0.8);
    const [behaviorProb, setBehaviorProb] = useState(0.5);

    // 计算重要性采样比率
    const ratio = (targetProb / behaviorProb).toFixed(2);

    // 模拟采样
    const samples = Array.from({ length: 20 }, (_, i) => {
        // 简单的可视化模拟，不是真实的统计采样
        const rand = Math.random();
        // 在 Behavior 分布下采样
        const isActionA = rand < behaviorProb;
        return { id: i, action: isActionA ? "A" : "B" };
    });

    const actionACount = samples.filter(s => s.action === "A").length;

    // 计算期望估计
    // 真实期望 (Target) E_pi = P(A)*V(A) + P(B)*V(B) 假设 V(A)=1, V(B)=0
    const trueExpectation = targetProb * 1 + (1 - targetProb) * 0;

    // 原始平均 (Behavior Average)
    const naiveAverage = actionACount / 20;

    // 加权平均 (Correction)
    // Weight for A = pi(A)/b(A), Weight for B = pi(B)/b(B)
    const weightA = targetProb / behaviorProb;
    const weightB = (1 - targetProb) / (1 - behaviorProb);

    const weightedSum = samples.reduce((acc, s) => {
        return acc + (s.action === "A" ? 1 * weightA : 0 * weightB);
    }, 0);
    const weightedAverage = weightedSum / 20;

    return (
        <Card className="p-6 w-full max-w-4xl mx-auto bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-800 font-sans">
            <h3 className="text-lg font-bold mb-6">Importance Sampling (重要性采样) 直观演示</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                {/* Target Policy pi */}
                <div className="space-y-4">
                    <div className="flex justify-between items-center">
                        <span className="font-bold text-blue-600">Target Policy <InlineMath>{"\\pi(A)"}</InlineMath></span>
                        <span className="font-mono">{targetProb.toFixed(2)}</span>
                    </div>
                    <Slider
                        value={[targetProb]}
                        onValueChange={(v) => setTargetProb(v[0])}
                        min={0.1} max={0.9} step={0.1}
                        className="bg-slate-200 rounded-full"
                    />
                    <div className="h-24 bg-white dark:bg-slate-800 rounded-lg flex items-end border border-slate-200 overflow-hidden relative">
                        <motion.div
                            className="w-1/2 bg-blue-500"
                            animate={{ height: `${targetProb * 100}%` }}
                        />
                        <motion.div
                            className="w-1/2 bg-slate-300"
                            animate={{ height: `${(1 - targetProb) * 100}%` }}
                        />
                        <div className="absolute bottom-1 left-4 text-xs text-white mix-blend-difference">A</div>
                        <div className="absolute bottom-1 right-4 text-xs text-slate-600">B</div>
                    </div>
                </div>

                {/* Behavior Policy b */}
                <div className="space-y-4">
                    <div className="flex justify-between items-center">
                        <span className="font-bold text-orange-600">Behavior Policy <InlineMath>{"b(A)"}</InlineMath></span>
                        <span className="font-mono">{behaviorProb.toFixed(2)}</span>
                    </div>
                    <Slider
                        value={[behaviorProb]}
                        onValueChange={(v) => setBehaviorProb(v[0])}
                        min={0.1} max={0.9} step={0.1}
                        className="bg-slate-200 rounded-full"
                    />
                    <div className="h-24 bg-white dark:bg-slate-800 rounded-lg flex items-end border border-slate-200 overflow-hidden relative">
                        <motion.div
                            className="w-1/2 bg-orange-500"
                            animate={{ height: `${behaviorProb * 100}%` }}
                        />
                        <motion.div
                            className="w-1/2 bg-slate-300"
                            animate={{ height: `${(1 - behaviorProb) * 100}%` }}
                        />
                        <div className="absolute bottom-1 left-4 text-xs text-white mix-blend-difference">A</div>
                        <div className="absolute bottom-1 right-4 text-xs text-slate-600">B</div>
                    </div>
                </div>
            </div>

            {/* Weights Display */}
            <div className="grid grid-cols-2 gap-4 mb-8">
                <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
                    <div className="text-xs text-slate-500 mb-1">Weight for A (<InlineMath>{"\\rho_A"}</InlineMath>)</div>
                    <div className="text-2xl font-mono font-bold text-purple-600">
                        {weightA.toFixed(2)}
                    </div>
                    <div className="text-xs text-slate-400"><InlineMath>{"\\pi(A)/b(A)"}</InlineMath></div>
                </div>
                <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
                    <div className="text-xs text-slate-500 mb-1">Weight for B (<InlineMath>{"\\rho_B"}</InlineMath>)</div>
                    <div className="text-2xl font-mono font-bold text-purple-600">
                        {weightB.toFixed(2)}
                    </div>
                    <div className="text-xs text-slate-400"><InlineMath>{"\\pi(B)/b(B)"}</InlineMath></div>
                </div>
            </div>

            {/* Simulation Results */}
            <div className="bg-slate-100 dark:bg-slate-800 p-4 rounded-xl">
                <h4 className="font-bold mb-4">Monte Carlo Estimation (Sample N=20)</h4>
                <div className="flex gap-1 flex-wrap mb-4">
                    {samples.map((s) => (
                        <div
                            key={s.id}
                            className={`w-3 h-3 rounded-full ${s.action === "A" ? "bg-orange-500" : "bg-slate-400"}`}
                            title={`Action ${s.action}`}
                        />
                    ))}
                </div>

                <div className="space-y-2 font-mono text-sm">
                    <div className="flex justify-between text-slate-500">
                        <span>True Val (<InlineMath>{"\\pi"}</InlineMath>):</span>
                        <span>{trueExpectation.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between text-red-500">
                        <span>Naive Avg (Direct from <InlineMath>{"b"}</InlineMath>):</span>
                        <span>{naiveAverage.toFixed(3)} (Biased!)</span>
                    </div>
                    <div className="flex justify-between text-green-600 font-bold border-t pt-2 border-slate-300">
                        <span>IS Estimate (Corrected):</span>
                        <span>{weightedAverage.toFixed(3)} (Unbiased)</span>
                    </div>
                </div>
            </div>

            {/* Variance Warning */}
            {weightA > 3 || weightB > 3 ? (
                <motion.div
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    className="mt-4 p-3 bg-yellow-50 text-yellow-800 text-sm rounded-lg flex items-start gap-2 border border-yellow-200"
                >
                    <span>⚠️</span>
                    <span>
                        <strong>高方差警告：</strong> 权重非常大（{Math.max(weightA, weightB).toFixed(2)}）。
                        这意味着 Behavior Policy 很少采样到 Target Policy 所需的动作，导致估计值极不稳定。
                        这就是 Off-policy MC 很难收敛的原因。
                    </span>
                </motion.div>
            ) : null}
        </Card>
    );
}
