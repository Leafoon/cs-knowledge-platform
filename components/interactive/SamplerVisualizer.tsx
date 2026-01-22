"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const SamplerVisualizer = () => {
    type Strategy = 'Sequential' | 'Random' | 'Weighted';
    const [strategy, setStrategy] = useState<Strategy>('Sequential');
    const [indices, setIndices] = useState<number[]>([]);
    const [isSampling, setIsSampling] = useState(false);

    // Initial dataset: 10 items
    // For Weighted: items 0-2 have weight 1, items 3-9 have weight 0.1 (imbalanced)
    const datasetSize = 10;
    const weights = [10, 10, 10, 1, 1, 1, 1, 1, 1, 1]; // Weights for display

    const runSampling = () => {
        setIsSampling(true);
        setIndices([]);

        let newIndices: number[] = [];

        if (strategy === 'Sequential') {
            for (let i = 0; i < datasetSize; i++) newIndices.push(i);
        } else if (strategy === 'Random') {
            // Shuffle
            const arr = Array.from({ length: datasetSize }, (_, i) => i);
            for (let i = arr.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [arr[i], arr[j]] = [arr[j], arr[i]];
            }
            newIndices = arr;
        } else if (strategy === 'Weighted') {
            // Weighted Random Sampling with Replacement
            // Simple implementation for demo
            // Total weight
            const totalWeight = weights.reduce((a, b) => a + b, 0);
            for (let i = 0; i < datasetSize; i++) {
                let r = Math.random() * totalWeight;
                for (let j = 0; j < datasetSize; j++) {
                    r -= weights[j];
                    if (r <= 0) {
                        newIndices.push(j);
                        break;
                    }
                }
            }
        }

        // Simulate step-by-step emission
        let step = 0;
        const interval = setInterval(() => {
            if (step >= newIndices.length) {
                clearInterval(interval);
                setIsSampling(false);
                return;
            }
            setIndices(prev => [...prev, newIndices[step]]);
            step++;
        }, 300);
    };

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">Sampler 策略演示</h3>

            <div className="flex gap-4 mb-6">
                {['Sequential', 'Random', 'Weighted'].map(s => (
                    <button
                        key={s}
                        onClick={() => { setStrategy(s as Strategy); setIndices([]); }}
                        className={`px-4 py-2 rounded-lg text-sm font-bold transition-colors ${strategy === s
                                ? 'bg-accent-primary text-white shadow'
                                : 'bg-bg-surface border border-border-subtle hover:bg-bg-elevated text-text-secondary'
                            }`}
                    >
                        {s} Sampler
                    </button>
                ))}
            </div>

            <div className="flex flex-col gap-6">
                {/* Dataset View */}
                <div className="relative">
                    <div className="text-xs font-bold text-text-tertiary mb-2 uppercase">Dataset (Index 0-9)</div>
                    <div className="flex gap-2">
                        {Array.from({ length: datasetSize }).map((_, i) => (
                            <div
                                key={i}
                                className={`w-8 h-12 rounded flex flex-col items-center justify-end pb-1 text-xs border
                                    ${i < 3 ? 'bg-blue-50 border-blue-200' : 'bg-gray-50 border-gray-200'}
                                `}
                            >
                                {strategy === 'Weighted' && (
                                    <div
                                        className="w-full bg-blue-400 absolute bottom-0 opacity-20"
                                        style={{ height: `${(weights[i] / 10) * 100}%` }}
                                    />
                                )}
                                <span className="z-10 font-mono">{i}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Sampling Stream */}
                <div className="h-20 bg-bg-surface rounded-xl border border-border-subtle p-4 flex items-center gap-3 overflow-x-auto">
                    <AnimatePresence>
                        {indices.map((idx, i) => (
                            <motion.div
                                key={`${i}-${idx}`}
                                initial={{ opacity: 0, x: -20, scale: 0.5 }}
                                animate={{ opacity: 1, x: 0, scale: 1 }}
                                className={`min-w-[32px] h-8 rounded-full flex items-center justify-center text-sm font-bold shadow-sm border
                                    ${idx < 3 ? 'bg-blue-500 text-white border-blue-600' : 'bg-white text-gray-600 border-gray-300'}
                                `}
                            >
                                {idx}
                            </motion.div>
                        ))}
                    </AnimatePresence>
                    {indices.length === 0 && !isSampling && <span className="text-sm text-text-tertiary italic">Ready to sample...</span>}
                </div>
            </div>

            <div className="mt-6">
                <button
                    onClick={runSampling}
                    disabled={isSampling}
                    className="w-full py-2 bg-text-primary text-bg-base rounded-lg font-bold hover:opacity-90 disabled:opacity-50 transition-opacity"
                >
                    {isSampling ? 'Sampling...' : `Start ${strategy} Sampling`}
                </button>
            </div>

            <div className="mt-4 p-3 bg-blue-50/50 rounded text-xs text-blue-800 leading-relaxed">
                {strategy === 'Sequential' && "按顺序读取。适用于验证集 (Validation Set) 或无心智负担的简单训练。"}
                {strategy === 'Random' && "完全随机打乱 (Shuffle=True)。训练集的默认标准，打破数据相关性。"}
                {strategy === 'Weighted' && "加权随机采样。注意看前 3 个样本 (蓝色) 出现的频率远高于其他样本。适用于**类别不平衡** (Class Imbalance) 场景。"}
            </div>
        </div>
    );
};

export default SamplerVisualizer;
