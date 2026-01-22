"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';

const ProfilerVisualizer = () => {
    const [view, setView] = useState<'cpu' | 'gpu'>('gpu');

    // Mock data for a typical training step
    const events = [
        { name: "DataLoader (Next Batch)", cpu_time: 15, gpu_time: 0, start: 0, color: "bg-gray-400" },
        { name: "Data Transfer (H2D)", cpu_time: 5, gpu_time: 2, start: 15, color: "bg-yellow-400" },
        { name: "Conv2d (Forward)", cpu_time: 2, gpu_time: 12, start: 20, color: "bg-blue-500" },
        { name: "ReLU + MaxPool", cpu_time: 1, gpu_time: 4, start: 22, color: "bg-blue-400" },
        { name: "Full Connected", cpu_time: 1, gpu_time: 8, start: 23, color: "bg-indigo-500" },
        { name: "Loss Calc", cpu_time: 1, gpu_time: 1, start: 24, color: "bg-red-400" },
        { name: "Backward (Autograd)", cpu_time: 8, gpu_time: 35, start: 25, color: "bg-purple-500" },
        { name: "Optimizer Step", cpu_time: 3, gpu_time: 5, start: 60, color: "bg-green-500" },
    ];

    const totalTime = 70; // Total duration in hypothetical units

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">PyTorch Profiler (模拟视图)</h3>

            <div className="flex justify-end gap-2 mb-4 text-xs">
                <button
                    onClick={() => setView('cpu')}
                    className={`px-3 py-1 rounded ${view === 'cpu' ? 'bg-text-primary text-bg-base' : 'bg-bg-surface border border-border-subtle'}`}
                >
                    CPU View
                </button>
                <button
                    onClick={() => setView('gpu')}
                    className={`px-3 py-1 rounded ${view === 'gpu' ? 'bg-text-primary text-bg-base' : 'bg-bg-surface border border-border-subtle'}`}
                >
                    GPU View
                </button>
            </div>

            <div className="relative h-[200px] border border-border-subtle rounded-lg bg-bg-surface overflow-hidden">
                {/* Timeline Ruler */}
                <div className="absolute top-0 left-0 w-full h-[20px] bg-slate-100 dark:bg-slate-800 border-b border-border-subtle flex justify-between px-2 text-[10px] text-text-tertiary">
                    <span>0ms</span>
                    <span>35ms</span>
                    <span>70ms</span>
                </div>

                <div className="absolute top-[30px] left-0 w-full p-2 space-y-2">
                    {/* Thread / Stream Simulation */}
                    <div className="flex h-[40px] items-center text-xs text-text-secondary w-full">
                        <div className="w-20 shrink-0 font-bold">{view === 'cpu' ? 'Main Thread' : 'GPU Stream'}</div>
                        <div className="flex-1 h-8 bg-slate-200 dark:bg-slate-800 rounded relative overflow-hidden">
                            {events.map((evt, idx) => {
                                const width = view === 'cpu' ? evt.cpu_time : evt.gpu_time;
                                const widthPct = (width / totalTime) * 100;
                                // Simple stacking logic logic (not perfectly accurate to start time, but sufficient for viz)
                                // Actually let's just make a stacked bar for simplicity
                                if (width === 0) return null;

                                return (
                                    <motion.div
                                        key={idx}
                                        initial={{ width: 0 }}
                                        animate={{ width: `${widthPct}%` }}
                                        className={`h-full inline-block ${evt.color} relative group border-r border-white/20`}
                                        title={`${evt.name}: ${width}ms`}
                                    >
                                        {widthPct > 5 && (
                                            <span className="absolute inset-0 flex items-center justify-center text-[9px] text-white overflow-hidden whitespace-nowrap px-1">
                                                {evt.name}
                                            </span>
                                        )}

                                        {/* Hover Tooltip */}
                                        <div className="opacity-0 group-hover:opacity-100 absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 bg-black text-white text-xs rounded whitespace-nowrap z-10 pointer-events-none">
                                            {evt.name}: {width}ms
                                        </div>
                                    </motion.div>
                                );
                            })}
                        </div>
                    </div>
                </div>

                {/* Bottleneck Warning */}
                {view === 'cpu' && (
                    <div className="absolute bottom-4 left-4 p-3 bg-red-50 border border-red-200 rounded text-xs text-red-700 max-w-[80%]">
                        <strong>Performance Insight:</strong> DataLoader 占用了较多 CPU 时间 (15ms)。
                        <br />建议：增加 <code>num_workers</code> 或开启 <code>pin_memory=True</code>。
                    </div>
                )}
                {view === 'gpu' && (
                    <div className="absolute bottom-4 left-4 p-3 bg-green-50 border border-green-200 rounded text-xs text-green-700 max-w-[80%]">
                        <strong>Performance Insight:</strong> GPU 利用率良好。Autograd (反向传播) 占据了主要计算时间，这是预期行为。
                    </div>
                )}
            </div>

            <div className="mt-4 flex flex-wrap gap-2 text-[10px]">
                {events.map(e => (
                    <div key={e.name} className="flex items-center gap-1">
                        <span className={`w-3 h-3 rounded-full ${e.color}`}></span>
                        <span className="text-text-secondary">{e.name}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default ProfilerVisualizer;
