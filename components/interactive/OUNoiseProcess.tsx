"use client";

import { useState, useEffect } from "react";

export function OUNoiseProcess() {
    const [theta, setTheta] = useState(0.15);
    const [sigma, setSigma] = useState(0.2);
    const [points, setPoints] = useState<number[]>([]);

    useEffect(() => {
        // 模拟 OU 噪声过程
        const steps = 100;
        const dt = 0.1;
        const noise: number[] = [];
        let state = 0;

        for (let i = 0; i < steps; i++) {
            const dx = theta * (0 - state) * dt + sigma * Math.sqrt(dt) * (Math.random() - 0.5) * 2;
            state += dx;
            noise.push(state);
        }

        setPoints(noise);
    }, [theta, sigma]);

    const maxAbs = Math.max(...points.map(Math.abs), 1);

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-amber-50 to-orange-50 dark:from-slate-900 dark:to-amber-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Ornstein-Uhlenbeck 噪声过程
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">OU 过程公式</h4>
                <div className="font-mono text-center p-4 bg-amber-50 dark:bg-amber-900/20 rounded">
                    dN<sub>t</sub> = θ(μ - N<sub>t</sub>)dt + σ√dt·W<sub>t</sub>
                </div>
                <div className="text-sm text-center mt-3 text-slate-600 dark:text-slate-400">
                    时间相关的探索噪声
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold mb-4">参数调整</h4>
                    <div className="space-y-4">
                        <div>
                            <div className="flex items-center justify-between mb-2">
                                <label className="text-sm font-medium">θ (回归速度) = {theta.toFixed(2)}</label>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max="30"
                                value={theta * 100}
                                onChange={(e) => setTheta(parseInt(e.target.value) / 100)}
                                className="w-full h-2 bg-amber-200 rounded-lg appearance-none cursor-pointer"
                            />
                            <div className="text-xs text-slate-500 mt-1">越大越快回归到均值</div>
                        </div>

                        <div>
                            <div className="flex items-center justify-between mb-2">
                                <label className="text-sm font-medium">σ (波动性) = {sigma.toFixed(2)}</label>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max="50"
                                value={sigma * 100}
                                onChange={(e) => setSigma(parseInt(e.target.value) / 100)}
                                className="w-full h-2 bg-amber-200 rounded-lg appearance-none cursor-pointer"
                            />
                            <div className="text-xs text-slate-500 mt-1">越大噪声幅度越大</div>
                        </div>

                        <div className="mt-4 flex gap-2">
                            <button onClick={() => { setTheta(0.15); setSigma(0.2); }} className="px-4 py-2 bg-amber-600 text-white rounded text-sm">默认值</button>
                            <button onClick={() => { setTheta(0.05); setSigma(0.3); }} className="px-4 py-2 bg-amber-100 rounded text-sm">缓慢探索</button>
                            <button onClick={() => { setTheta(0.3); setSigma(0.1); }} className="px-4 py-2 bg-amber-100 rounded text-sm">快速回归</button>
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold mb-4">噪声轨迹</h4>
                    <div className="h-48 bg-slate-50 dark:bg-slate-700 rounded relative overflow-hidden">
                        <svg width="100%" height="100%" viewBox="0 0 400 200" preserveAspectRatio="none">
                            {/* 中线 */}
                            <line x1="0" y1="100" x2="400" y2="100" stroke="currentColor" strokeWidth="1" strokeDasharray="5,5" opacity="0.3" />

                            {/* 噪声曲线 */}
                            <polyline
                                fill="none"
                                stroke="rgb(251, 191, 36)"
                                strokeWidth="2"
                                points={points.map((p, i) => {
                                    const x = (i / points.length) * 400;
                                    const y = 100 - (p / maxAbs) * 80;
                                    return `${x},${y}`;
                                }).join(' ')}
                            />
                        </svg>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">OU 噪声特性</h4>
                <div className="grid grid-cols-3 gap-4 text-sm text-center">
                    <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded">
                        <div className="text-2xl mb-2">🎯</div>
                        <div className="font-bold">均值回归</div>
                        <div className="text-xs mt-2">长期趋向 μ</div>
                    </div>
                    <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded">
                        <div className="text-2xl mb-2">🔗</div>
                        <div className="font-bold">时间相关</div>
                        <div className="text-xs mt-2">连续性探索</div>
                    </div>
                    <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                        <div className="text-2xl mb-2">📊</div>
                        <div className="font-bold">可调节性</div>
                        <div className="text-xs mt-2">θ, σ 控制行为</div>
                    </div>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 现代实践：简单高斯噪声 N(0, σ) 也能工作得很好
            </div>
        </div>
    );
}
