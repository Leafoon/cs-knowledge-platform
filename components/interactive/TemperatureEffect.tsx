"use client";

import { useState } from "react";

export function TemperatureEffect() {
    const [temperature, setTemperature] = useState(0.2);

    // 模拟策略分布（3个动作）
    const getDistribution = (temp: number) => {
        const qValues = [5.0, 4.5, 3.0];
        const exps = qValues.map(q => Math.exp(q / temp));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => e / sum);
    };

    const dist = getDistribution(temperature);

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-rose-50 to-red-50 dark:from-slate-900 dark:to-rose-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:textslate-100 mb-2">
                    温度参数的影响
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">调整温度 α</h4>
                <div className="flex items-center gap-4">
                    <span className="text-sm font-medium">α = {temperature.toFixed(2)}</span>
                    <input
                        type="range"
                        min="1"
                        max="200"
                        value={temperature * 100}
                        onChange={(e) => setTemperature(parseInt(e.target.value) / 100)}
                        className="flex-1 h-2 bg-rose-200 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex gap-2">
                        <button onClick={() => setTemperature(0.01)} className="px-3 py-1 text-xs bg-rose-100 rounded">极低</button>
                        <button onClick={() => setTemperature(0.2)} className="px-3 py-1 text-xs bg-rose-600 text-white rounded">0.2</button>
                        <button onClick={() => setTemperature(2)} className="px-3 py-1 text-xs bg-rose-100 rounded">极高</button>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">策略分布 π(a|s) ∝ exp(Q(s,a)/α)</h4>
                <div className="space-y-4">
                    {['动作 1 (最优)', '动作 2', '动作 3'].map((label, i) => (
                        <div key={i}>
                            <div className="flex items-center justify-between mb-1 text-sm">
                                <span>{label}</span>
                                <span className="font-mono">{(dist[i] * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-8 bg-slate-100 dark:bg-slate-700 rounded overflow-hidden">
                                <div
                                    className={`h-full ${i === 0 ? 'bg-rose-500' : i === 1 ? 'bg-rose-400' : 'bg-rose-300'}`}
                                    style={{ width: `${dist[i] * 100}%` }}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className={`bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg ${temperature < 0.05 ? 'border-4 border-rose-500' : ''}`}>
                    <h4 className="text-lg font-bold text-blue-600 mb-4">α → 0 (低温)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>策略:</strong> 接近 argmax
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>探索:</strong> 极少
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>等价于:</strong> 贪婪策略
                        </div>
                    </div>
                </div>

                <div className={`bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg ${temperature >= 0.05 && temperature <= 0.5 ? 'border-4 border-green-500' : ''}`}>
                    <h4 className="text-lg font-bold text-green-600 mb-4">α ≈ 0.2 (适中)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>策略:</strong> 平衡
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>探索:</strong> 适度
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>💡 常用值</strong>
                        </div>
                    </div>
                </div>

                <div className={`bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg ${temperature > 0.5 ? 'border-4 border-rose-500' : ''}`}>
                    <h4 className="text-lg font-bold text-orange-600 mb-4">α → ∞ (高温)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>策略:</strong> 接近均匀分布
                        </div>
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>探索:</strong> 最大化
                        </div>
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>等价于:</strong> 随机策略
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
