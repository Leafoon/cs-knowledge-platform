"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const TransferLearningVisualizer = () => {
    const [step, setStep] = useState(0); // 0: Init, 1: Filter, 2: Load

    // Model A: ResNet-like (Source)
    const sourceLayers = [
        { name: "conv1.weight", shape: "[64, 3, 7, 7]", match: true },
        { name: "layer1.0.conv1.weight", shape: "[64, 64, 3, 3]", match: true },
        { name: "layer2.0.conv1.weight", shape: "[128, 64, 3, 3]", match: true },
        { name: "fc.weight", shape: "[1000, 2048]", match: false }, // Mismatch
        { name: "fc.bias", shape: "[1000]", match: false },         // Mismatch
    ];

    // Model B: Custom Task (Target) - 10 Classes
    const targetLayers = [
        { name: "conv1.weight", shape: "[64, 3, 7, 7]", status: "random" },
        { name: "layer1.0.conv1.weight", shape: "[64, 64, 3, 3]", status: "random" },
        { name: "layer2.0.conv1.weight", shape: "[128, 64, 3, 3]", status: "random" },
        { name: "fc.weight", shape: "[10, 2048]", status: "random" }, // 10 classes
        { name: "fc.bias", shape: "[10]", status: "random" },
    ];

    const [targetState, setTargetState] = useState(targetLayers);

    const handleLoad = () => {
        let newStep = step + 1;
        if (newStep > 2) {
            newStep = 0;
            setTargetState(targetLayers.map(l => ({ ...l, status: "random" })));
        }
        setStep(newStep);

        if (newStep === 1) {
            // Filter step visualization (no state change in target yet, just visual highlight)
        }

        if (newStep === 2) {
            // Loading step
            setTargetState(prev => prev.map((layer, idx) => {
                const source = sourceLayers[idx];
                if (source.match) {
                    return { ...layer, status: "loaded" };
                } else {
                    return { ...layer, status: "random" }; // Stay random
                }
            }));
        }
    };

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">迁移学习：权重加载模拟</h3>

            <div className="flex justify-between items-center mb-8 px-4">
                <div className="text-center">
                    <div className="text-sm font-bold text-blue-600 mb-2">Pretrained Model (Source)</div>
                    <div className="text-xs text-text-tertiary">ImageNet (1000 classes)</div>
                </div>

                <div className="flex flex-col items-center gap-2">
                    <button
                        onClick={handleLoad}
                        className="px-6 py-2 bg-accent-primary text-white rounded-full font-bold shadow-lg hover:scale-105 transition-all text-sm"
                    >
                        {step === 0 && "Start Loading"}
                        {step === 1 && "Start Copying"}
                        {step === 2 && "Reset"}
                    </button>
                    <div className="text-xs text-text-tertiary">
                        {step === 0 && "Step 1: 准备"}
                        {step === 1 && "Step 2: 匹配 Key & Shape"}
                        {step === 2 && "Step 3: 赋值"}
                    </div>
                </div>

                <div className="text-center">
                    <div className="text-sm font-bold text-green-600 mb-2">My Model (Target)</div>
                    <div className="text-xs text-text-tertiary">Custom Task (10 classes)</div>
                </div>
            </div>

            <div className="relative">
                {sourceLayers.map((src, idx) => {
                    const tgt = targetState[idx];
                    const isMismatch = !src.match;

                    return (
                        <div key={src.name} className="flex items-center justify-between mb-4 relative">
                            {/* Source Block */}
                            <div className={`w-40 p-2 rounded border text-xs font-mono transition-colors
                                ${step >= 1 && isMismatch ? 'bg-gray-100 text-gray-400 border-gray-200' : 'bg-blue-50 text-blue-800 border-blue-200'}
                            `}>
                                <div className="truncate font-bold">{src.name}</div>
                                <div className="text-[10px] opacity-70">{src.shape}</div>
                            </div>

                            {/* Connection/Animation */}
                            <div className="flex-1 flex justify-center relative h-8">
                                {step === 1 && (
                                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                                        {isMismatch ? (
                                            <span className="text-red-500 font-bold text-xs bg-red-50 px-2 py-1 rounded border border-red-100">❌ Shape Mismatch</span>
                                        ) : (
                                            <motion.div
                                                initial={{ scale: 0 }} animate={{ scale: 1 }}
                                                className="text-green-500 font-bold text-xs bg-green-50 px-2 py-1 rounded border border-green-100"
                                            >
                                                ✅ Match
                                            </motion.div>
                                        )}
                                    </div>
                                )}
                                {step === 2 && !isMismatch && (
                                    <motion.div
                                        initial={{ width: 0, opacity: 0 }}
                                        animate={{ width: "100%", opacity: 1 }}
                                        transition={{ duration: 0.5 }}
                                        className="h-1 bg-green-400 self-center rounded-full"
                                    />
                                )}
                            </div>

                            {/* Target Block */}
                            <div className={`w-40 p-2 rounded border text-xs font-mono transition-all duration-500
                                ${tgt.status === 'loaded' ? 'bg-green-100 border-green-400 text-green-900 shadow-md scale-105' : 'bg-gray-50 border-gray-300 text-gray-500'}
                            `}>
                                <div className="truncate font-bold">{tgt.name}</div>
                                <div className="flex justify-between items-center mt-1">
                                    <span className="text-[10px] opacity-70">{tgt.shape}</span>
                                    <span className={`text-[9px] px-1 rounded ${tgt.status === 'loaded' ? 'bg-green-200 text-green-800' : 'bg-gray-200'}`}>
                                        {tgt.status === 'loaded' ? 'Pretrained' : 'Random Init'}
                                    </span>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>

            <div className="mt-6 p-3 bg-yellow-50 border border-yellow-100 rounded text-xs text-yellow-800 leading-relaxed">
                <strong>从 ResNet50 迁移到 10 分类任务：</strong>
                <br />
                前面的卷积层 (Backbone) 形状完全一致，权重顺利加载（绿色）。
                <br />
                最后的 FC 层 (`fc.weight`) 形状从 <code>[1000, 2048]</code> 变成了 <code>[10, 2048]</code>，无法加载，因此保持随机初始化（灰色）。
                <br />
                这正是 <code>state_dict</code> 过滤逻辑要做的事情。
            </div>
        </div>
    );
};

export default TransferLearningVisualizer;
