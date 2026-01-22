"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const KernelFusionVisualizer = () => {
    const [mode, setMode] = useState<'eager' | 'compiled'>('eager');

    // Eager: Separate kernels for each op: a * x + b
    // 1. Mul (Read a, Read x, Write temp)
    // 2. Add (Read temp, Read b, Write out)

    // Compiled: Fused Kernel
    // 1. Fused MulAdd (Read a, x, b -> Compute -> Write out)

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">算子融合演示 (Kernel Fusion)</h3>
            <p className="text-sm text-text-secondary mb-6">
                演示操作: <code className="bg-bg-surface px-1 rounded">y = torch.sin(x) + torch.cos(x)</code>
            </p>

            <div className="flex justify-center gap-6 mb-8">
                <button
                    onClick={() => setMode('eager')}
                    className={`px-6 py-3 rounded-xl border flex flex-col items-center gap-1 transition-all w-40 ${mode === 'eager'
                            ? 'border-accent-primary bg-accent-primary/5 ring-2 ring-accent-primary/20'
                            : 'border-border-subtle bg-bg-surface hover:bg-bg-elevated'
                        }`}
                >
                    <div className="text-sm font-bold">Eager Mode</div>
                    <div className="text-[10px] text-text-tertiary">Python Default</div>
                </button>

                <button
                    onClick={() => setMode('compiled')}
                    className={`px-6 py-3 rounded-xl border flex flex-col items-center gap-1 transition-all w-40 ${mode === 'compiled'
                            ? 'border-accent-primary bg-accent-primary/5 ring-2 ring-accent-primary/20'
                            : 'border-border-subtle bg-bg-surface hover:bg-bg-elevated'
                        }`}
                >
                    <div className="text-sm font-bold">torch.compile</div>
                    <div className="text-[10px] text-text-tertiary">Fused Kernel</div>
                </button>
            </div>

            <div className="relative h-[240px] bg-slate-50 dark:bg-slate-900/50 rounded-xl border border-border-subtle p-6 flex flex-col items-center justify-center overflow-hidden">
                {/* Memory (DRAM) */}
                <div className="absolute bottom-4 left-0 w-full flex justify-center gap-20 text-xs font-mono text-text-tertiary">
                    <div>read x</div>
                    <div>read/write temp</div>
                    <div>write y</div>
                </div>

                <AnimatePresence mode="wait">
                    {mode === 'eager' ? (
                        <motion.div
                            key="eager"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="flex flex-col gap-4 items-center"
                        >
                            {/* Op 1: Sin */}
                            <div className="flex flex-col items-center">
                                <div className="px-4 py-2 bg-blue-100 dark:bg-blue-900 border border-blue-300 rounded text-blue-700 dark:text-blue-300 text-sm font-bold shadow-sm">
                                    Kernel 1: Sin(x)
                                </div>
                                <div className="flex gap-1 h-8 items-center">
                                    <motion.div
                                        animate={{ y: [20, -20, 20] }}
                                        transition={{ repeat: Infinity, duration: 1.5 }}
                                        className="w-1 h-3 bg-red-400 rounded-full"
                                    />
                                    <span className="text-[10px] text-text-tertiary">Mem I/O (High Overhead)</span>
                                </div>
                            </div>

                            {/* Op 2: Cos (Parallel/Seq) */}
                            <div className="flex flex-col items-center">
                                <div className="px-4 py-2 bg-blue-100 dark:bg-blue-900 border border-blue-300 rounded text-blue-700 dark:text-blue-300 text-sm font-bold shadow-sm">
                                    Kernel 2: Cos(x)
                                </div>
                                <div className="flex gap-1 h-8 items-center">
                                    <motion.div
                                        animate={{ y: [20, -20, 20] }}
                                        transition={{ repeat: Infinity, duration: 1.5 }}
                                        className="w-1 h-3 bg-red-400 rounded-full"
                                    />
                                    <span className="text-[10px] text-text-tertiary">Mem I/O</span>
                                </div>
                            </div>

                            {/* Op 3: Add */}
                            <div className="flex flex-col items-center">
                                <div className="px-4 py-2 bg-blue-100 dark:bg-blue-900 border border-blue-300 rounded text-blue-700 dark:text-blue-300 text-sm font-bold shadow-sm">
                                    Kernel 3: Add(+)
                                </div>
                            </div>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="compiled"
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0 }}
                            className="flex flex-col items-center"
                        >
                            <div className="w-64 h-32 bg-gradient-to-br from-green-100 to-emerald-100 dark:from-green-900/30 dark:to-emerald-900/30 border border-green-400 rounded-xl flex flex-col items-center justify-center shadow-lg relative">
                                <div className="text-green-800 dark:text-green-300 font-bold text-lg mb-2">Fused Kernel</div>
                                <div className="text-xs text-green-700 dark:text-green-400 px-4 text-center">
                                    Compute optimized: <br />
                                    <code>tmp = sin(x); y = tmp + cos(x)</code>
                                    <br /> All in registers!
                                </div>

                                {/* Single IO */}
                                <div className="absolute -bottom-8 flex flex-col items-center">
                                    <div className="w-0.5 h-8 bg-green-500"></div>
                                    <span className="text-[10px] text-text-tertiary">Minimal Memory I/O</span>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            <div className="mt-4 p-4 rounded-lg bg-bg-surface border border-border-subtle text-xs text-text-secondary">
                {mode === 'eager'
                    ? "Eager Mode 需要 3 次 Kernel Launch，每次都需要从显存读取和写入数据。对于这种简单运算（Element-wise），由内存带宽瓶颈（Memory Bound）导致性能低下。"
                    : "torch.compile 使用 Triton 编译器将这 3 个操作融合为一个 GPU Kernel。数据只读取一次，并在片上高速缓存/寄存器中完成所有计算，再写回。速度提升显著。"
                }
            </div>
        </div>
    );
};

export default KernelFusionVisualizer;
