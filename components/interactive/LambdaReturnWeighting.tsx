"use client";

import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export function LambdaReturnWeighting() {
    const [lambda, setLambda] = useState(0.9);

    const nSteps = 15;
    const weights = Array.from({length: nSteps}, (_, n) => {
        const weight = (1 - lambda) * Math.pow(lambda, n);
        return {
            n: n + 1,
            weight: weight,
            cumulative: Array.from({length: n + 1}, (_, i) => 
                (1 - lambda) * Math.pow(lambda, i)
            ).reduce((a, b) => a + b, 0)
        };
    });

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    λ-return 权重分布
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    调整 λ 观察权重变化
                </p>
            </div>

            {/* λ 滑块 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="flex items-center justify-between mb-4">
                    <span className="font-semibold text-slate-800 dark:text-slate-100">λ 参数：</span>
                    <span className="text-3xl font-bold text-indigo-600">{lambda.toFixed(2)}</span>
                </div>
                <input
                    type="range"
                    min="0"
                    max="0.99"
                    step="0.01"
                    value={lambda}
                    onChange={(e) => setLambda(parseFloat(e.target.value))}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-2">
                    <span>0 (TD)</span>
                    <span>0.5</span>
                    <span>0.99 (→MC)</span>
                </div>
            </div>

            {/* 权重柱状图 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">
                    n-step Return 权重
                </h4>
                <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={weights}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                            dataKey="n" 
                            label={{ value: 'n-step', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis 
                            label={{ value: 'Weight (1-λ)λ^(n-1)', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip />
                        <Bar dataKey="weight" fill="#6366f1" />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* 公式和解释 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-slate-800 rounded-lg p-4">
                    <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3">
                        λ-return 公式
                    </h4>
                    <div className="font-mono text-sm text-slate-600 dark:text-slate-400 space-y-2">
                        <div>G<sub>t</sub><sup>λ</sup> = (1-λ) Σ λ<sup>n-1</sup> G<sub>t</sub><sup>(n)</sup></div>
                        <div className="mt-2 text-xs">
                            权重和 = {weights.reduce((sum, w) => sum + w.weight, 0).toFixed(4)}
                        </div>
                    </div>
                </div>

                <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-4">
                    <h4 className="font-bold text-indigo-800 dark:text-indigo-300 mb-3">
                        λ 的影响
                    </h4>
                    <ul className="text-sm text-indigo-700 dark:text-indigo-400 space-y-1">
                        <li>• λ=0: 100% TD(0)</li>
                        <li>• λ小: 重视近期</li>
                        <li>• λ大: 重视远期</li>
                        <li>• λ→1: 接近 MC</li>
                    </ul>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 λ-return 是所有 n-step return 的几何加权平均
            </div>
        </div>
    );
}
