"use client";

import { useState } from "react";

export function PPOClipMechanism() {
    const [advantage, setAdvantage] = useState(1);
    const epsilon = 0.2;

    const ratios = Array.from({ length: 50 }, (_, i) => 0.5 + i * 0.03);

    const getObjective = (r: number, a: number) => {
        const unclipped = r * a;
        const clipped = Math.max(1 - epsilon, Math.min(1 + epsilon, r)) * a;
        return Math.min(unclipped, clipped);
    };

    const maxObj = Math.max(...ratios.map(r => Math.abs(getObjective(r, advantage))));

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-violet-50 to-purple-50 dark:from-slate-900 dark:to-violet-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    PPO Clip 机制可视化
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">调整 Advantage</h4>
                <div className="flex items-center gap-4">
                    <span className="text-sm font-medium">A = {advantage.toFixed(1)}</span>
                    <input
                        type="range"
                        min="-20"
                        max="20"
                        value={advantage * 10}
                        onChange={(e) => setAdvantage(parseInt(e.target.value) / 10)}
                        className="flex-1 h-2 bg-violet-200 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex gap-2">
                        <button onClick={() => setAdvantage(-1)} className="px-3 py-1 text-xs bg-red-100 rounded">坏动作</button>
                        <button onClick={() => setAdvantage(0)} className="px-3 py-1 text-xs bg-slate-100 rounded">中性</button>
                        <button onClick={() => setAdvantage(1)} className="px-3 py-1 text-xs bg-green-100 rounded">好动作</button>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">PPO 目标函数</h4>
                <div className="h-64 bg-slate-50 dark:bg-slate-700 rounded relative">
                    <svg width="100%" height="100%" viewBox="0 0 500 250" preserveAspectRatio="none">
                        {/* 裁剪边界线 */}
                        <line x1={((1 - epsilon - 0.5) / 1.5) * 500} y1="0" x2={((1 - epsilon - 0.5) / 1.5) * 500} y2="250"
                            stroke="rgb(239, 68, 68)" strokeWidth="2" strokeDasharray="5,5" opacity="0.5" />
                        <line x1={((1 + epsilon - 0.5) / 1.5) * 500} y1="0" x2={((1 + epsilon - 0.5) / 1.5) * 500} y2="250"
                            stroke="rgb(239, 68, 68)" strokeWidth="2" strokeDasharray="5,5" opacity="0.5" />

                        {/* r * A (未裁剪) */}
                        <polyline
                            fill="none"
                            stroke="rgb(99, 102, 241)"
                            strokeWidth="2"
                            opacity="0.4"
                            points={ratios.map(r => {
                                const x = ((r - 0.5) / 1.5) * 500;
                                const y = 125 - ((r * advantage) / maxObj) * 100;
                                return `${x},${y}`;
                            }).join(' ')}
                        />

                        {/* clip(r, 1-ε, 1+ε) * A */}
                        <polyline
                            fill="none"
                            stroke="rgb(168, 85, 247)"
                            strokeWidth="2"
                            opacity="0.4"
                            points={ratios.map(r => {
                                const clipped_r = Math.max(1 - epsilon, Math.min(1 + epsilon, r));
                                const x = ((r - 0.5) / 1.5) * 500;
                                const y = 125 - ((clipped_r * advantage) / maxObj) * 100;
                                return `${x},${y}`;
                            }).join(' ')}
                        />

                        {/* min(未裁剪, 裁剪) - PPO 目标 */}
                        <polyline
                            fill="none"
                            stroke="rgb(124, 58, 237)"
                            strokeWidth="4"
                            points={ratios.map(r => {
                                const x = ((r - 0.5) / 1.5) * 500;
                                const y = 125 - (getObjective(r, advantage) / maxObj) * 100;
                                return `${x},${y}`;
                            }).join(' ')}
                        />

                        {/* 标签 */}
                        <text x="10" y="20" className="text-xs fill-current">目标</text>
                        <text x="10" y="240" className="text-xs fill-current">r = π/π<tspan fontSize="8">old</tspan></text>
                    </svg>
                </div>
                <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-1 bg-indigo-500 opacity-40"></div>
                        <span>r · A</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-1 bg-purple-500 opacity-40"></div>
                        <span>clip(r) · A</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-1 bg-violet-700" style={{ height: '3px' }}></div>
                        <span>min(两者)</span>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">Clip 的作用</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-violet-50 dark:bg-violet-900/20 rounded">
                        <strong>Advantage {'>'} 0 (好动作):</strong><br />
                        限制 r 不超过 1+ε，防止过度增加概率
                    </div>
                    <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded">
                        <strong>Advantage &lt; 0 (坏动作):</strong><br />
                        限制 r 不低于 1-ε，防止过度减少概率
                    </div>
                </div>
            </div>
        </div>
    );
}
