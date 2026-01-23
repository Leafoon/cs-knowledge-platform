"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';

const DispatcherVisualizer = () => {
    const [step, setStep] = useState(0);

    // Steps:
    // 0: Python Call (torch.add)
    // 1: Dispatcher Key Calculation (Identify Backend, Autograd, etc.)
    // 2: Dispatch Table Lookup
    // 3: Kernel Execution (Aten / CUDA)

    const nextStep = () => setStep(s => (s + 1) % 4);

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">The Dispatcher (PyTorch 的心脏)</h3>

            <div className="flex items-start gap-4">
                {/* 1. Python Land */}
                <div className={`flex-1 p-4 rounded-xl border transition-all ${step === 0 ? 'bg-blue-100 border-blue-400 opacity-100' : 'bg-bg-surface border-border-subtle opacity-50'}`}>
                    <div className="text-xs font-bold uppercase mb-2">1. Python API</div>
                    <code className="text-sm bg-white p-1 rounded border">torch.add(x, y)</code>
                </div>

                {/* Arrow */}
                <div className="mt-8">→</div>

                {/* 2. Dispatcher */}
                <div className={`flex-1 p-4 rounded-xl border transition-all relative ${step === 1 || step === 2 ? 'bg-purple-100 border-purple-400 opacity-100 scale-105' : 'bg-bg-surface border-border-subtle opacity-50'}`}>
                    <div className="text-xs font-bold uppercase mb-2">2. Dispatcher</div>
                    {step === 1 && (
                        <div className="text-xs text-purple-700 animate-pulse">
                            Parsing Dispatch Keys...
                            <br />
                            [Backend: CUDA]
                            <br />
                            [Autograd: Enable]
                        </div>
                    )}
                    {step === 2 && (
                        <div className="text-xs text-purple-700 font-mono">
                            Table Lookup:
                            <br />
                            idx = (CUDA, Float)
                            <br />
                            {'->'} ptr to aten::add_kernel
                        </div>
                    )}
                </div>

                {/* Arrow */}
                <div className="mt-8">→</div>

                {/* 3. Backend Kernel */}
                <div className={`flex-1 p-4 rounded-xl border transition-all ${step === 3 ? 'bg-green-100 border-green-400 opacity-100' : 'bg-bg-surface border-border-subtle opacity-50'}`}>
                    <div className="text-xs font-bold uppercase mb-2">3. Backend (C++)</div>
                    <div className="text-xs font-mono">
                        TensorIterator::binary_op
                        <br />
                        CUDA Kernel Launch
                    </div>
                </div>
            </div>

            <div className="mt-8 flex justify-center">
                <button
                    onClick={nextStep}
                    className="px-8 py-2 bg-text-primary text-bg-base rounded-full font-bold hover:scale-105 transition-transform"
                >
                    Next Step ({step + 1}/4)
                </button>
            </div>

            <p className="mt-4 text-center text-xs text-text-secondary">
                PyTorch 的多态机制 (Dispatcher) 是它能同时支持 Autograd, Tracing, XLA, CUDA 等多种后端的关键。
                所有操作都会经过这个中央调度器。
            </p>
        </div>
    );
};

export default DispatcherVisualizer;
