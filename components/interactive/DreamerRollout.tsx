"use client";

import { useState } from "react";

export function DreamerRollout() {
    const [step, setStep] = useState(0);
    const maxSteps = 5;

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-violet-50 to-purple-50 dark:from-slate-900 dark:to-violet-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Dreamer 想象轨迹
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    在潜在空间执行滚动
                </p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">RSSM (Recurrent State Space Model)</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                        <strong>决定性状态:</strong> h<sub>t</sub> = f(h<sub>t-1</sub>, z<sub>t-1</sub>, a<sub>t-1</sub>)
                    </div>
                    <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                        <strong>随机状态:</strong> z<sub>t</sub> ~ p(z<sub>t</sub> | h<sub>t</sub>)
                    </div>
                    <div className="p-3 bg-pink-50 dark:bg-pink-900/20 rounded">
                        <strong>奖励预测:</strong> r̂<sub>t</sub> = r(h<sub>t</sub>, z<sub>t</sub>)
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-bold">想象步骤: {step} / {maxSteps}</h4>
                    <div className="flex gap-2">
                        <button
                            onClick={() => setStep(Math.max(0, step - 1))}
                            disabled={step === 0}
                            className="px-4 py-2 bg-violet-100 rounded disabled:opacity-50"
                        >
                            ← 上一步
                        </button>
                        <button
                            onClick={() => setStep(Math.min(maxSteps, step + 1))}
                            disabled={step === maxSteps}
                            className="px-4 py-2 bg-violet-600 text-white rounded disabled:opacity-50"
                        >
                            下一步 →
                        </button>
                        <button
                            onClick={() => setStep(0)}
                            className="px-4 py-2 bg-slate-200 rounded"
                        >
                            重置
                        </button>
                    </div>
                </div>

                <div className="space-y-3">
                    {Array.from({ length: step + 1 }).map((_, i) => (
                        <div key={i} className={`p-3 rounded ${i === step ? 'bg-violet-100 dark:bg-violet-900/30 border-2 border-violet-500' : 'bg-slate-50 dark:bg-slate-700'}`}>
                            <div className="flex items-center justify-between">
                                <span className="font-mono text-sm">t = {i}</span>
                                <div className="flex gap-2 text-xs">
                                    <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 rounded">h<sub>{i}</sub></span>
                                    <span className="px-2 py-1 bg-purple-100 dark:bg-purple-900/30 rounded">z<sub>{i}</sub></span>
                                    <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 rounded">a<sub>{i}</sub></span>
                                    <span className="px-2 py-1 bg-pink-100 dark:bg-pink-900/30 rounded">r<sub>{i}</sub></span>
                                </div>
                            </div>
                            {i === step && (
                                <div className="mt-2 text-xs text-violet-700 dark:text-violet-400">
                                    {i === 0 && "初始状态: 从真实编码开始"}
                                    {i > 0 && "想象下一步: 策略选择动作 → RSSM 预测状态 → 预测奖励"}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-violet-600 mb-4">训练 Actor</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                            用想象轨迹计算 λ-return
                        </div>
                        <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                            最大化累积想象奖励
                        </div>
                        <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                            梯度反向传播到策略
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-purple-600 mb-4">训练 Critic</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            预测想象轨迹的价值
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            回归到 λ-return
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            提供 baseline 减少方差
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">Dreamer 的优势</h4>
                <div className="grid grid-cols-3 gap-4 text-sm text-center">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <div className="text-2xl mb-2">⚡</div>
                        <div className="font-bold">高样本效率</div>
                        <div className="text-xs mt-2">在想象中<br />大量训练</div>
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <div className="text-2xl mb-2">🎯</div>
                        <div className="font-bold">端到端</div>
                        <div className="text-xs mt-2">从像素<br />直接学习</div>
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <div className="text-2xl mb-2">🚀</div>
                        <div className="font-bold">泛化性</div>
                        <div className="text-xs mt-2">潜在空间<br />更好泛化</div>
                    </div>
                </div>
            </div>
        </div>
    );
}
