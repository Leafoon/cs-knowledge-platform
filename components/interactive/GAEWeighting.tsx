"use client";

import { useState } from "react";

export function GAEWeighting() {
    const [lambda, setLambda] = useState(0.95);

    const getWeights = (lambda: number, steps: number = 5) => {
        const weights = [];
        for (let i = 0; i < steps; i++) {
            weights.push(Math.pow(0.99 * lambda, i));
        }
        const sum = weights.reduce((a, b) => a + b, 0);
        return weights.map(w => w / sum);
    };

    const weights = getWeights(lambda);

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-violet-50 to-purple-50 dark:from-slate-900 dark:to-violet-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    GAE 权重分布
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">GAE(λ) 公式</h4>
                <div className="font-mono text-center p-4 bg-violet-50 dark:bg-violet-900/20 rounded">
                    A<sup>GAE(λ)</sup> = Σ (γλ)<sup>l</sup> δ<sub>t+l</sub>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">调整 λ 参数</h4>
                <div className="flex items-center gap-4">
                    <span className="text-sm font-medium">λ = {lambda.toFixed(2)}</span>
                    <input
                        type="range"
                        min="0"
                        max="100"
                        value={lambda * 100}
                        onChange={(e) => setLambda(parseInt(e.target.value) / 100)}
                        className="flex-1 h-2 bg-violet-200 rounded-lg appearance-none cursor-pointer dark:bg-violet-700"
                    />
                    <div className="flex gap-2">
                        <button onClick={() => setLambda(0)} className="px-3 py-1 text-xs bg-violet-100 rounded">0 (TD)</button>
                        <button onClick={() => setLambda(0.95)} className="px-3 py-1 text-xs bg-violet-600 text-white rounded">0.95</button>
                        <button onClick={() => setLambda(1)} className="px-3 py-1 text-xs bg-violet-100 rounded">1 (MC)</button>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">TD Error 权重分布</h4>
                <div className="flex items-end justify-around h-48 gap-2">
                    {weights.map((w, i) => (
                        <div key={i} className="flex flex-col items-center flex-1">
                            <div
                                className="w-full bg-violet-500 rounded-t"
                                style={{ height: `${w * 100}%` }}
                            />
                            <div className="mt-2 text-xs font-mono">δ<sub>{i}</sub></div>
                            <div className="text-xs">{(w * 100).toFixed(1)}%</div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">λ 参数效果</h4>
                <div className="grid grid-cols-3 gap-4 text-sm text-center">
                    <div className={`p-4 rounded ${lambda < 0.3 ? 'bg-violet-100 dark:bg-violet-900/30 border-2 border-violet-500' : 'bg-slate-50 dark:bg-slate-700'}`}>
                        <div className="font-bold">λ → 0</div>
                        <div className="text-xs mt-2">1-step TD<br />高偏差 低方差</div>
                    </div>
                    <div className={`p-4 rounded ${lambda >= 0.3 && lambda <= 0.97 ? 'bg-violet-100 dark:bg-violet-900/30 border-2 border-violet-500' : 'bg-slate-50 dark:bg-slate-700'}`}>
                        <div className="font-bold">λ ≈ 0.95</div>
                        <div className="text-xs mt-2">平衡<br />常用默认值</div>
                    </div>
                    <div className={`p-4 rounded ${lambda > 0.97 ? 'bg-violet-100 dark:bg-violet-900/30 border-2 border-violet-500' : 'bg-slate-50 dark:bg-slate-700'}`}>
                        <div className="font-bold">λ → 1</div>
                        <div className="text-xs mt-2">Monte Carlo<br />低偏差 高方差</div>
                    </div>
                </div>
            </div>
        </div>
    );
}
