"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const HookVisualizer = () => {
    const [capturedData, setCapturedData] = useState<any[]>([]);
    const [isFlowing, setIsFlowing] = useState(false);

    // Simulate flow steps
    // 1. Layer Input
    // 2. Compute (Forward)
    // 3. Hook Triggered
    // 4. Layer Output -> Next Layer

    const runFlow = () => {
        setIsFlowing(true);
        setCapturedData([]);

        // Step sequence
        setTimeout(() => setCapturedData([{ stage: 'input', val: 'x' }]), 500);
        setTimeout(() => setCapturedData(prev => [...prev, { stage: 'compute', val: 'Wx+b' }]), 1500);

        // Hook intercepts after compute but before return (usually) or on return
        // forward hook signature: hook(module, input, output)
        setTimeout(() => {
            setCapturedData(prev => [...prev, { stage: 'hook', val: 'Hook Captured: Wx+b' }]);
        }, 2500);

        setTimeout(() => {
            setIsFlowing(false);
        }, 4000);
    };

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">Hooks 探针机制演示</h3>

            <div className="flex flex-col gap-8 items-center">
                {/* Visual Pipeline */}
                <div className="flex items-center gap-2 w-full max-w-lg relative">
                    {/* Layer */}
                    <div className="flex-1 h-24 border-2 border-blue-200 bg-blue-50 rounded-xl flex items-center justify-center relative z-10">
                        <span className="text-blue-800 font-bold">Conv2d Layer</span>

                        {/* Hook Point Indicator */}
                        <div className="absolute right-0 top-1/2 -translate-y-1/2 w-4 h-4 bg-red-400 rounded-full border-2 border-white translate-x-2 z-20 shadow-sm" title="Hook Point"></div>
                    </div>

                    {/* Data Flow Particle */}
                    {isFlowing && (
                        <motion.div
                            initial={{ left: '-10%', opacity: 0 }}
                            animate={{
                                left: ['0%', '40%', '50%', '110%'],
                                opacity: [0, 1, 1, 0]
                            }}
                            transition={{ duration: 3, times: [0, 0.4, 0.6, 1] }}
                            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-text-primary rounded-full z-30 pointer-events-none"
                        />
                    )}

                    {/* Hook Sidecar */}
                    <div className="absolute right-0 top-1/2 translate-x-12 -translate-y-1/2 flex flex-col items-center">
                        <motion.div
                            initial={{ opacity: 0, scale: 0.5, x: -20 }}
                            animate={{
                                opacity: capturedData.find(d => d.stage === 'hook') ? 1 : 0.2,
                                scale: capturedData.find(d => d.stage === 'hook') ? 1 : 0.8,
                                x: 0
                            }}
                            className="w-32 h-20 border-2 border-dashed border-red-300 bg-red-50 rounded-lg flex flex-col items-center justify-center p-2 text-center"
                        >
                            <div className="text-xs text-red-500 font-bold mb-1">🪝 Dictionary</div>
                            {capturedData.find(d => d.stage === 'hook') && (
                                <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} className="text-[10px] bg-white px-2 py-1 rounded shadow text-text-primary font-mono border">
                                    "activation": Tensor
                                </motion.div>
                            )}
                        </motion.div>

                        {/* Connecting Line */}
                        <svg className="absolute right-full top-1/2 -translate-y-1/2 w-12 h-20 pointer-events-none overflow-visible">
                            <path d="M -10 0 Q -5 0, 0 0" fill="none" stroke="#fca5a5" strokeWidth="2" strokeDasharray="4 4" />
                        </svg>
                    </div>
                </div>

                <button
                    onClick={runFlow}
                    disabled={isFlowing}
                    className="px-6 py-2 bg-text-primary text-bg-base rounded-lg font-bold hover:opacity-90 disabled:opacity-50 transition-opacity"
                >
                    {isFlowing ? 'Processing...' : 'Forward Pass (Run with Hook)'}
                </button>

                <div className="text-sm text-text-secondary bg-bg-surface p-4 rounded-lg border border-border-subtle max-w-lg">
                    <code>register_forward_hook(hook_fn)</code> 允许我们在数据流经某一层的**输出端**时，将数据“钩”出来一份副本。
                    <br />
                    这在不修改原本模型 return 逻辑的情况下，非常适合做 Feature Extraction (风格迁移、Grad-CAM)。
                </div>
            </div>
        </div>
    );
};

export default HookVisualizer;
