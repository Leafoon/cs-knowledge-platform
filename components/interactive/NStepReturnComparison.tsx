"use client";

import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export function NStepReturnComparison() {
    const [nValue, setNValue] = useState(3);

    // 模拟一个 episode 的奖励序列
    const rewards = [-1, -1, 2, -1, -1, 5, -1, -1, -1, 10];
    const gamma = 0.9;
    const V_terminal = 0;

    // 计算不同 n 值的 return
    const calculateNStepReturn = (n: number, t: number) => {
        let G = 0;
        const steps = Math.min(n, rewards.length - t);
        
        for (let i = 0; i < steps; i++) {
            G += Math.pow(gamma, i) * rewards[t + i];
        }
        
        if (t + n < rewards.length) {
            G += Math.pow(gamma, n) * 3;  // 假设 V(s_{t+n}) = 3
        }
        
        return G;
    };

    const comparisonData = [1, 2, 3, 5, 10, Infinity].map(n => ({
        n: n === Infinity ? '∞(MC)' : n,
        bias: n === 1 ? 'High' : n === Infinity ? 'None' : 'Medium',
        variance: n === 1 ? 'Low' : n === Infinity ? 'High' : 'Medium',
        value: n === Infinity ? -10 : calculateNStepReturn(n, 0)
    }));

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-slate-900 dark:to-purple-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    n-step Return 对比
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    偏差-方差权衡
                </p>
            </div>

            {/* n 值滑块 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="flex items-center justify-between mb-4">
                    <span className="font-semibold">选择 n 值：</span>
                    <span className="text-2xl font-bold text-purple-600">{nValue}</span>
                </div>
                <input
                    type="range"
                    min="1"
                    max="10"
                    value={nValue}
                    onChange={(e) => setNValue(parseInt(e.target.value))}
                    className="w-full"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-2">
                    <span>1 (TD)</span>
                    <span>5</span>
                    <span>10 (→MC)</span>
                </div>
            </div>

            {/* 对比表格 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b-2 border-slate-200 dark:border-slate-600">
                            <th className="px-4 py-2">n</th>
                            <th className="px-4 py-2">偏差</th>
                            <th className="px-4 py-2">方差</th>
                            <th className="px-4 py-2">适用场景</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr className={nValue === 1 ? "bg-purple-50 dark:bg-purple-900/20" : ""}>
                            <td className="px-4 py-3 text-center font-bold">1 (TD)</td>
                            <td className="px-4 py-3 text-center text-red-600">高</td>
                            <td className="px-4 py-3 text-center text-green-600">低</td>
                            <td className="px-4 py-3">短期决策</td>
                        </tr>
                        <tr className={nValue >= 3 && nValue <= 5 ? "bg-purple-50 dark:bg-purple-900/20" : ""}>
                            <td className="px-4 py-3 text-center font-bold">3-5</td>
                            <td className="px-4 py-3 text-center text-orange-600">中</td>
                            <td className="px-4 py-3 text-center text-orange-600">中</td>
                            <td className="px-4 py-3"><strong>通用推荐</strong></td>
                        </tr>
                        <tr className={nValue > 5 ? "bg-purple-50 dark:bg-purple-900/20" : ""}>
                            <td className="px-4 py-3 text-center font-bold">∞ (MC)</td>
                            <td className="px-4 py-3 text-center text-green-600">无</td>
                            <td className="px-4 py-3 text-center text-red-600">高</td>
                            <td className="px-4 py-3">长期回报</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 实践中通常 n=3 到 n=5 效果最好
            </div>
        </div>
    );
}
