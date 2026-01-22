"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const ParallelVisualizer = () => {
    const [mode, setMode] = useState<'loop' | 'vmap'>('loop');
    const [isCalculating, setIsCalculating] = useState(false);
    const [results, setResults] = useState<number[]>([]);

    // Simulate data batch
    const batchData = [1, 2, 3, 4, 5, 6, 7, 8];

    const runSimulation = () => {
        setIsCalculating(true);
        setResults([]);

        if (mode === 'loop') {
            // Sequential Simulation
            let i = 0;
            const interval = setInterval(() => {
                if (i >= batchData.length) {
                    clearInterval(interval);
                    setIsCalculating(false);
                    return;
                }
                setResults(prev => [...prev, batchData[i] * 2]); // Simple operation x * 2
                i++;
            }, 500); // Slow sequential
        } else {
            // Parallel Simulation (Vmap)
            setTimeout(() => {
                setResults(batchData.map(x => x * 2));
                setIsCalculating(false);
            }, 1000); // One-shot delay
        }
    };

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">向量化执行演示 (Loop vs Vmap/Vectorized)</h3>

            <div className="flex justify-center gap-6 mb-8">
                <button
                    onClick={() => { setMode('loop'); setResults([]); }}
                    className={`px-6 py-3 rounded-xl border flex flex-col items-center gap-2 transition-all ${mode === 'loop'
                            ? 'border-accent-primary bg-accent-primary/5 ring-2 ring-accent-primary/20'
                            : 'border-border-subtle bg-bg-surface hover:bg-bg-elevated'
                        }`}
                >
                    <div className="text-sm font-bold">Python For-Loop</div>
                    <div className="text-[10px] text-text-tertiary">Sequential Execution</div>
                    <div className="flex gap-1 mt-1">
                        <div className="w-2 h-4 bg-gray-400 rounded-sm"></div>
                        <div className="w-2 h-4 bg-gray-300 rounded-sm"></div>
                        <div className="w-2 h-4 bg-gray-200 rounded-sm"></div>
                    </div>
                </button>

                <button
                    onClick={() => { setMode('vmap'); setResults([]); }}
                    className={`px-6 py-3 rounded-xl border flex flex-col items-center gap-2 transition-all ${mode === 'vmap'
                            ? 'border-accent-primary bg-accent-primary/5 ring-2 ring-accent-primary/20'
                            : 'border-border-subtle bg-bg-surface hover:bg-bg-elevated'
                        }`}
                >
                    <div className="text-sm font-bold">torch.vmap / Vectorized</div>
                    <div className="text-[10px] text-text-tertiary">Parallel Execution (SIMD)</div>
                    <div className="flex gap-1 mt-1">
                        <div className="w-2 h-4 bg-accent-primary rounded-sm"></div>
                        <div className="w-2 h-4 bg-accent-primary rounded-sm"></div>
                        <div className="w-2 h-4 bg-accent-primary rounded-sm"></div>
                    </div>
                </button>
            </div>

            {/* Execution Visualizer */}
            <div className="relative min-h-[120px] bg-slate-100 dark:bg-slate-900 rounded-xl p-4 flex items-center justify-around border border-inner">
                <AnimatePresence mode="popLayout">
                    {batchData.map((val, idx) => {
                        const isProcessed = results[idx] !== undefined;
                        const isCurrent = mode === 'loop' && results.length === idx && isCalculating;

                        return (
                            <div key={val} className="flex flex-col items-center gap-2 relative">
                                {/* Input Node */}
                                <div className={`w-8 h-8 rounded-full flex items-center justify-center font-mono text-sm border
                                     ${isProcessed
                                        ? 'bg-green-100 border-green-400 text-green-700'
                                        : isCurrent
                                            ? 'bg-yellow-100 border-yellow-400 text-yellow-700 scale-110 shadow-lg'
                                            : 'bg-white border-gray-300 text-gray-400'
                                    }
                                     transition-all duration-300
                                 `}>
                                    {val}
                                </div>

                                {/* Operation Arrow */}
                                <motion.div
                                    animate={{
                                        height: (isProcessed || (mode === 'vmap' && isCalculating)) ? 20 : 0,
                                        opacity: (isProcessed || (mode === 'vmap' && isCalculating)) ? 1 : 0.2
                                    }}
                                    className="w-0.5 bg-gray-300"
                                    style={{ height: 0 }}
                                />

                                {/* Result Node */}
                                <AnimatePresence>
                                    {isProcessed && (
                                        <motion.div
                                            initial={{ opacity: 0, scale: 0.5, y: -10 }}
                                            animate={{ opacity: 1, scale: 1, y: 0 }}
                                            className="w-8 h-8 rounded bg-accent-primary text-white flex items-center justify-center font-bold font-mono text-sm shadow-sm"
                                        >
                                            {results[idx]}
                                        </motion.div>
                                    )}
                                </AnimatePresence>

                                {/* Parallel Processing Indicator for Vmap */}
                                {mode === 'vmap' && isCalculating && !isProcessed && (
                                    <motion.div
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        className="absolute top-10 w-full h-8 flex justify-center items-center"
                                    >
                                        <svg className="animate-spin w-4 h-4 text-accent-primary" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                    </motion.div>
                                )}
                            </div>
                        );
                    })}
                </AnimatePresence>
            </div>

            <div className="mt-6 flex justify-center">
                <button
                    onClick={runSimulation}
                    disabled={isCalculating}
                    className="px-8 py-2 bg-text-primary text-bg-base rounded-lg font-bold hover:opacity-90 disabled:opacity-50 transition-opacity"
                >
                    {isCalculating ? 'Computing...' : 'Run Transformation (x * 2)'}
                </button>
            </div>

            <div className="mt-4 text-center text-xs text-text-tertiary">
                {mode === 'loop'
                    ? "For Loop: 逐个处理，无法利用 GPU 并行能力。"
                    : "torch.vmap: 自动将其转换为并行 Kernel，一次性处理整个 Batch。"}
            </div>
        </div>
    );
};

export default ParallelVisualizer;
