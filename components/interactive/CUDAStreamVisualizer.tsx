"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const CUDAStreamVisualizer = () => {
    const [mode, setMode] = useState<'default' | 'multi_stream'>('default');
    const [tasks, setTasks] = useState<any[]>([]);

    // Tasks: 2 independent operations
    // Op A: H2D -> Compute A -> D2H
    // Op B: H2D -> Compute B -> D2H

    // In Default Stream: Serialized
    // H2D(A) -> Comp(A) -> D2H(A) -> H2D(B) -> Comp(B) -> D2H(B)

    // In Multi Stream: Overlapped
    // Stream 1: H2D(A) -> Comp(A) -> D2H(A)
    // Stream 2:           H2D(B) -> Comp(B) -> D2H(B)  (Overlapped if hardware supports)

    const runSimulation = () => {
        setTasks([]);
        // Re-trigger animation
        setTimeout(() => setTasks([1]), 10);
    };

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">并行加速：CUDA Streams</h3>

            <div className="flex justify-center gap-6 mb-8">
                <button onClick={() => setMode('default')} className={`px-4 py-2 rounded-lg font-bold text-sm ${mode === 'default' ? 'bg-accent-primary text-white' : 'bg-bg-surface border'}`}>
                    Default Stream (Serial)
                </button>
                <button onClick={() => setMode('multi_stream')} className={`px-4 py-2 rounded-lg font-bold text-sm ${mode === 'multi_stream' ? 'bg-accent-primary text-white' : 'bg-bg-surface border'}`}>
                    Multi-Stream (Async)
                </button>
            </div>

            <div className="relative h-[200px] bg-slate-900 rounded-xl overflow-hidden border border-slate-700">
                {/* Time Axis */}
                <div className="absolute top-0 w-full h-6 border-b border-slate-700 bg-slate-800 text-[10px] text-slate-400 flex items-center px-2">
                    Time ----------------------------------------&gt;
                </div>

                {/* Lanes */}
                <div className="mt-8 space-y-4 p-2">
                    {/* CPU Launch Lane (Just for reference) */}

                    {/* Stream 1 */}
                    <div className="relative h-12 bg-slate-800/50 rounded flex items-center px-2">
                        <span className="w-16 text-xs text-slate-400 font-bold shrink-0">Stream 1</span>
                        {tasks.length > 0 && (
                            <>
                                {/* Task A */}
                                <motion.div
                                    initial={{ x: 0, opacity: 0 }} animate={{ x: 0, opacity: 1 }}
                                    transition={{ duration: 0.5 }}
                                    className="h-8 bg-blue-500 rounded flex items-center justify-center text-[10px] text-white font-bold ml-0 w-24 border border-blue-400"
                                >
                                    Op A (Copy+Calc)
                                </motion.div>
                            </>
                        )}
                    </div>

                    {/* Stream 2 or Task B in Stream 1 */}
                    <div className="relative h-12 bg-slate-800/50 rounded flex items-center px-2">
                        <span className="w-16 text-xs text-slate-400 font-bold shrink-0">
                            {mode === 'default' ? '(Waiting)' : 'Stream 2'}
                        </span>

                        {tasks.length > 0 && (
                            <motion.div
                                initial={{ x: 0, opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: mode === 'default' ? 1.5 : 0.2, duration: 0.5 }}
                                className={`h-8 rounded flex items-center justify-center text-[10px] text-white font-bold w-24 border
                                    ${mode === 'default'
                                        ? 'bg-blue-500 border-blue-400 ml-2' // Appended to same stream physically (visualized here as shifted)
                                        : 'bg-green-500 border-green-400 ml-8' // Overlapped in Stream 2
                                    }
                                `}
                                style={{
                                    marginLeft: mode === 'default' ? '120px' : '20px' // Serial vs Parallel offset
                                }}
                            >
                                Op B (Copy+Calc)
                            </motion.div>
                        )}
                    </div>
                </div>

                {/* Overhead / Time Saved Annotation */}
                <div className="absolute bottom-4 right-4 text-xs text-slate-300">
                    {mode === 'default' ? (
                        <span>Total Time: <strong>Long</strong> (A then B)</span>
                    ) : (
                        <span className="text-green-400">Total Time: <strong>Short</strong> (A || B)</span>
                    )}
                </div>
            </div>

            <div className="mt-4 flex justify-center">
                <button onClick={runSimulation} className="px-6 py-1 bg-white border rounded shadow-sm text-sm hover:bg-gray-50">
                    Replay Animation
                </button>
            </div>

            <div className="mt-4 text-xs text-text-secondary">
                * PyTorch 默认所有操作都在 Default Stream 上顺序执行，确保安全。
                <br />
                * 使用 <code>cuda.Stream()</code> 可以让独立的计算任务并行执行，或掩盖 H2D 拷贝时间。
            </div>
        </div>
    );
};

export default CUDAStreamVisualizer;
