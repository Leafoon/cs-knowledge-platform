"use client";

import { DisplayMath } from "@/components/ui/Math";
import { useState } from "react";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

export function MaximizationBiasDemo() {
    const [samples, setSamples] = useState<{ val: number, isMax: boolean }[][]>([]);
    const trueValue = 0;

    // Simulate multiple estimators for Q(s,a) where true Q=0
    // We take max(Q)

    const runExperiment = () => {
        const newSamples: { val: number, isMax: boolean }[] = [];
        for (let i = 0; i < 5; i++) { // 5 actions
            // Each action has true value 0, but noisy estimate
            const noise = (Math.random() - 0.5) * 2; // -1 to 1
            newSamples.push({ val: noise, isMax: false });
        }

        // Find max
        let maxVal = -Infinity;
        let maxIdx = -1;
        newSamples.forEach((s, i) => {
            if (s.val > maxVal) {
                maxVal = s.val;
                maxIdx = i;
            }
        });

        if (maxIdx !== -1) newSamples[maxIdx].isMax = true;

        setSamples(prev => [newSamples, ...prev].slice(0, 10)); // Keep last 10
    };

    return (
        <Card className="p-6 bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-800">
            <h3 className="text-lg font-bold mb-4">最大化偏差 (Maximization Bias) 演示</h3>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-6">
                假设有 5 个动作，它们的真实价值都是 0。但是由于估计噪声，有些估值会大于 0，有些小于 0。
                如果你总是取 <strong>max</strong>，你得到的估计值就会恒大于 0（正偏差）。
                <br />
                <DisplayMath>{"E[\\max(X)] \\ge \\max(E[X])"}</DisplayMath>
            </p>

            <div className="flex justify-center mb-6">
                <Button onClick={runExperiment}>采样一组估值 (Sample Q)</Button>
            </div>

            <div className="space-y-2">
                {samples.map((batch, batchIdx) => {
                    const maxItem = batch.find(x => x.isMax);
                    return (
                        <div key={batchIdx} className="flex items-center gap-2 text-xs font-mono">
                            <span className="w-8 text-slate-400">Exp {batchIdx + 1}</span>
                            <div className="flex gap-1 flex-1">
                                {batch.map((item, i) => (
                                    <div
                                        key={i}
                                        className={`flex-1 h-8 flex items-center justify-center rounded ${item.isMax
                                            ? "bg-red-500 text-white font-bold"
                                            : "bg-slate-200 dark:bg-slate-700 text-slate-500"
                                            }`}
                                    >
                                        {item.val.toFixed(1)}
                                    </div>
                                ))}
                            </div>
                            <span className="w-24 text-right text-red-600 font-bold">
                                Max: {maxItem?.val.toFixed(2)}
                            </span>
                        </div>
                    );
                })}
            </div>

            {samples.length > 0 && (
                <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-700 text-center">
                    <p className="text-sm">
                        平均最大值 (Bias):
                        <span className="font-bold text-red-600 ml-2 text-lg">
                            {(samples.reduce((acc, batch) => acc + (batch.find(x => x.isMax)?.val || 0), 0) / samples.length).toFixed(3)}
                        </span>
                        <span className="text-slate-400 ml-2">(True Value = 0)</span>
                    </p>
                    <p className="text-xs text-slate-500 mt-2">这就是为什么 Q-Learning 和 DQN 容易高估价值的原因。</p>
                </div>
            )}
        </Card>
    );
}
