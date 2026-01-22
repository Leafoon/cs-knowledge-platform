"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const DistributedVisualizer = () => {
    const [mode, setMode] = useState<'ddp' | 'fsdp'>('ddp');
    const [step, setStep] = useState(0);

    // DDP: Local(0) -> AllReduce(1) -> Update(2)
    // FSDP: GatherParam(0) -> Compute(1) -> ReduceScatter(2) -> Update(3) ... simplified
    // Let's stick to a visual representation of "Memory Usage" mostly

    // Auto-cycle steps
    useEffect(() => {
        const interval = setInterval(() => {
            setStep(s => (s + 1) % 3);
        }, 2000);
        return () => clearInterval(interval);
    }, []);

    const nodes = [0, 1, 2, 3];

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">分布式策略对比: {mode === 'ddp' ? 'DDP' : 'FSDP'}</h3>

            <div className="flex justify-center gap-6 mb-8">
                <button
                    onClick={() => setMode('ddp')}
                    className={`px-4 py-2 rounded-lg font-bold text-sm transition-all ${mode === 'ddp' ? 'bg-accent-primary text-white' : 'bg-bg-surface border border-border-subtle'}`}
                >
                    Data Parallel (DDP)
                </button>
                <button
                    onClick={() => setMode('fsdp')}
                    className={`px-4 py-2 rounded-lg font-bold text-sm transition-all ${mode === 'fsdp' ? 'bg-accent-primary text-white' : 'bg-bg-surface border border-border-subtle'}`}
                >
                    Fully Sharded (FSDP)
                </button>
            </div>

            <div className="grid grid-cols-4 gap-4">
                {nodes.map(id => (
                    <div key={id} className="relative flex flex-col items-center">
                        <div className="w-20 h-32 bg-slate-800 rounded-lg flex flex-col p-1 gap-1 border border-slate-700 shadow-xl overflow-hidden relative">
                            <span className="text-[10px] text-slate-500 text-center w-full">GPU {id}</span>

                            {/* Model Memory Visualization */}
                            <div className="flex-1 w-full bg-slate-700/50 rounded flex flex-col gap-0.5 p-0.5 relative">
                                <span className="text-[8px] text-slate-400 text-center mb-0.5">VRAM</span>
                                {mode === 'ddp' ? (
                                    <>
                                        {/* DDP: Replica of Full Model (Blue) */}
                                        <div className="flex-1 w-full bg-blue-500 rounded-sm opacity-80 flex items-center justify-center text-[8px] text-white">Model</div>
                                        <div className="flex-1 w-full bg-blue-400 rounded-sm opacity-60 flex items-center justify-center text-[8px] text-white">Grad</div>
                                        <div className="flex-1 w-full bg-blue-300 rounded-sm opacity-40 flex items-center justify-center text-[8px] text-white">Optim</div>
                                    </>
                                ) : (
                                    <>
                                        {/* FSDP: Sharded Model (Mixed Colors) */}
                                        <div className="flex-1 w-full flex gap-0.5">
                                            {nodes.map(n => (
                                                <div key={n} className={`flex-1 h-full rounded-sm ${n === id ? 'bg-green-500' : 'bg-slate-600 opacity-20'}`} />
                                            ))}
                                        </div>
                                        <div className="absolute inset-0 flex items-center justify-center text-[8px] text-white font-bold drop-shadow-md">
                                            1/4 Shard
                                        </div>
                                    </>
                                )}
                            </div>

                            {/* Compute Visuals */}
                            <div className="h-6 w-full flex items-center justify-center">
                                {step === 1 && (
                                    <motion.div
                                        initial={{ scale: 0 }} animate={{ scale: 1 }}
                                        className={`w-4 h-4 rounded-full ${mode === 'ddp' ? 'bg-blue-400' : 'bg-green-400'} animate-ping`}
                                    />
                                )}
                            </div>
                        </div>

                        {/* DDP Sync Lines */}
                        {mode === 'ddp' && step === 1 && id < 3 && (
                            <motion.div
                                initial={{ width: 0 }} animate={{ width: "120%" }}
                                className="absolute top-16 left-1/2 h-1 bg-blue-500/50 -z-10"
                            />
                        )}
                    </div>
                ))}
            </div>

            <div className="mt-6 p-4 bg-bg-surface rounded-lg border border-border-subtle text-xs text-text-secondary leading-relaxed">
                {mode === 'ddp' ? (
                    <div>
                        <strong>DDP (Replicated):</strong> 每个 GPU 都要存一份**完整**的模型参数、梯度和优化器状态。
                        <br />
                        <span className="text-red-500">缺点</span>: 显存占用极大。如果模型大到单卡放不下，DDP 就跑不起来。
                    </div>
                ) : (
                    <div>
                        <strong>FSDP (Sharded):</strong> 模型参数、梯度、优化器状态被**切分**（Shard）到各个 GPU 上。
                        <br />
                        每个 GPU 只负责存 1/{nodes.length} 的内容。
                        <br />
                        <span className="text-green-600">优点</span>: 可以训练超大模型（如 70B 参数）。代价是通信量增加（Forward 前需要 Gather 参数）。
                    </div>
                )}
            </div>
        </div>
    );
};

export default DistributedVisualizer;

