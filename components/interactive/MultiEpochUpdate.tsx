"use client";

import { useState } from "react";

export function MultiEpochUpdate() {
    const [epoch, setEpoch] = useState(0);
    const maxEpochs = 10;

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-slate-900 dark:to-rose-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    多 Epoch 更新过程
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-bold">Epoch: {epoch} / {maxEpochs}</h4>
                    <div className="flex gap-2">
                        <button
                            onClick={() => setEpoch(Math.max(0, epoch - 1))}
                            disabled={epoch === 0}
                            className="px-4 py-2 bg-rose-100 rounded disabled:opacity-50"
                        >
                            ← 上一轮
                        </button>
                        <button
                            onClick={() => setEpoch(Math.min(maxEpochs, epoch + 1))}
                            disabled={epoch === maxEpochs}
                            className="px-4 py-2 bg-rose-600 text-white rounded disabled:opacity-50"
                        >
                            下一轮 →
                        </button>
                        <button
                            onClick={() => setEpoch(0)}
                            className="px-4 py-2 bg-slate-200 rounded"
                        >
                            重置
                        </button>
                    </div>
                </div>

                <div className="space-y-2">
                    {Array.from({ length: epoch + 1 }).map((_, i) => (
                        <div key={i} className={`p-3 rounded ${i === epoch ? 'bg-rose-100 dark:bg-rose-900/30 border-2 border-rose-500' : 'bg-slate-50 dark:bg-slate-700'}`}>
                            <div className="flex items-center justify-between">
                                <span className="font-mono text-sm">Epoch {i}</span>
                                <div className="flex gap-2 text-xs">
                                    <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 rounded">Mini-batch 1</span>
                                    <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 rounded">Mini-batch 2</span>
                                    <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 rounded">...</span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-blue-600 mb-4">A2C (1 epoch)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            收集数据 → 更新 1 次 → <strong>丢弃</strong>
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>样本效率低:</strong> 每个数据只用一次
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-rose-500">
                    <h4 className="text-lg font-bold text-rose-600 mb-4">PPO (多 epoch)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-rose-50 dark:bg-rose-900/20 rounded">
                            收集数据 → 更新 <strong>10 次</strong> → 丢弃
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>样本效率高:</strong> 每个数据重复使用
                        </div>
                        <div className="p-3 bg-rose-50 dark:bg-rose-900/20 rounded">
                            <strong>关键:</strong> Clip 机制防止过度更新
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">为什么多 epoch 有效？</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong>✅ PPO Clip:</strong> 限制策略变化，防止数据过时
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong>✅ Mini-batch:</strong> 每次使用不同子集，增加多样性
                    </div>
                </div>
            </div>
        </div>
    );
}
