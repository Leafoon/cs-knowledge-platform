"use client";

import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export function ValueIterationConvergence() {
    // Generate dummy convergence data
    // Error delta usually drops exponentially
    const data = Array.from({ length: 20 }, (_, i) => ({
        iter: i,
        delta: Math.max(0.001, 10 * Math.exp(-0.5 * i))
    }));

    return (
        <div className="w-full max-w-3xl mx-auto p-4 bg-white dark:bg-slate-900 rounded-lg shadow border border-slate-200 dark:border-slate-800">
            <h4 className="text-center font-bold mb-4 text-slate-700 dark:text-slate-200">
                价值迭代收敛曲线 (Max Norm Error)
            </h4>
            <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis
                            dataKey="iter"
                            stroke="#64748b"
                            label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis
                            stroke="#64748b"
                            label={{ value: '|| V_k+1 - V_k ||', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0' }}
                        />
                        <Line
                            type="monotone"
                            dataKey="delta"
                            stroke="#ef4444"
                            strokeWidth={3}
                            dot={{ r: 4 }}
                            activeDot={{ r: 6 }}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
            <p className="text-center text-xs text-slate-500 mt-2">
                当 Δ &lt; θ (如 1e-4) 时停止迭代。由于 Bellman 算子的收缩性质，这保证收敛。
            </p>
        </div>
    );
}
