"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const BatchProcessor = () => {
    const [isProcessing, setIsProcessing] = useState(false);
    const [batchItems, setBatchItems] = useState<any[]>([]);
    const [processedBatch, setProcessedBatch] = useState<any[]>([]);

    const rawData = [
        { id: 1, src: "/images/cat1.jpg", label: "Cat", size: "800x600" },
        { id: 2, src: "/images/dog1.jpg", label: "Dog", size: "1024x768" },
        { id: 3, src: "/images/cat2.jpg", label: "Cat", size: "400x400" },
        { id: 4, src: "/images/bird.jpg", label: "Bird", size: "1200x900" },
    ];

    const runPipeline = async () => {
        setIsProcessing(true);
        setBatchItems([]);
        setProcessedBatch([]);

        // 1. Fetching (simulated)
        for (let i = 0; i < rawData.length; i++) {
            await new Promise(r => setTimeout(r, 600));
            setBatchItems(prev => [...prev, { ...rawData[i], status: 'raw' }]);
        }

        await new Promise(r => setTimeout(r, 500));

        // 2. Transform (Resize & ToTensor)
        setBatchItems(prev => prev.map(item => ({ ...item, status: 'transformed', size: '224x224' })));

        await new Promise(r => setTimeout(r, 800));

        // 3. Collate (Batching)
        setProcessedBatch([
            { type: 'Tensor', shape: '[4, 3, 224, 224]', desc: 'Image Batch' },
            { type: 'Tensor', shape: '[4]', desc: 'Labels' }
        ]);

        setIsProcessing(false);
    };

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-bold text-text-primary">DataLoader 流水线演示</h3>
                <button
                    onClick={runPipeline}
                    disabled={isProcessing}
                    className="px-4 py-2 bg-accent-primary text-white rounded-lg hover:bg-accent-primary/90 disabled:opacity-50 transition-colors"
                >
                    {isProcessing ? "Processing..." : "Start Loading Batch (BS=4)"}
                </button>
            </div>

            <div className="space-y-8">
                {/* Stage 1: Dataset (__getitem__) */}
                <div className="relative p-4 border border-dashed border-border-subtle rounded-lg bg-bg-surface/50 min-h-[120px]">
                    <div className="absolute -top-3 left-4 px-2 bg-bg-surface text-xs font-mono text-text-tertiary">Dataset (Raw Data)</div>
                    <div className="flex gap-4 overflow-x-auto p-2">
                        {rawData.map((item) => (
                            <div key={item.id} className="min-w-[80px] h-[80px] bg-gray-200 rounded flex items-center justify-center text-xs text-gray-500">
                                {item.label} <br /> {item.size}
                            </div>
                        ))}
                    </div>
                </div>

                {/* Stage 2: Transforms */}
                <div className="flex items-center justify-center">
                    <svg className="w-6 h-6 text-text-quaternary animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" /></svg>
                </div>

                <div className="relative p-4 border border-blue-200/50 rounded-lg bg-blue-50/10 min-h-[120px]">
                    <div className="absolute -top-3 left-4 px-2 bg-bg-surface text-xs font-mono text-blue-600">Transforms (Resize, ToTensor)</div>
                    <div className="flex gap-4 p-2 items-center">
                        <AnimatePresence>
                            {batchItems.map((item) => (
                                <motion.div
                                    key={item.id}
                                    initial={{ opacity: 0, scale: 0.8 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    className={`min-w-[80px] h-[80px] rounded flex flex-col items-center justify-center text-xs border-2 transition-colors duration-500
                                ${item.status === 'transformed' ? 'bg-green-100 border-green-400 text-green-800' : 'bg-gray-100 border-gray-300 text-gray-500'}
                            `}
                                >
                                    <span className="font-bold">{item.label}</span>
                                    <span className="text-[10px]">{item.size}</span>
                                    {item.status === 'transformed' && <span className="text-[9px] mt-1 bg-green-200 px-1 rounded">Tensor</span>}
                                </motion.div>
                            ))}
                        </AnimatePresence>
                        {batchItems.length === 0 && <span className="text-sm text-text-tertiary italic">Waiting for data...</span>}
                    </div>
                </div>

                {/* Stage 3: Collate Fn */}
                <div className="flex items-center justify-center">
                    <svg className="w-6 h-6 text-text-quaternary animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" /></svg>
                </div>

                <div className="relative p-4 border border-purple-200/50 rounded-lg bg-purple-50/10 min-h-[120px]">
                    <div className="absolute -top-3 left-4 px-2 bg-bg-surface text-xs font-mono text-purple-600">DataLoader (collate_fn)</div>
                    <div className="flex gap-6 p-2 justify-center items-center h-full">
                        <AnimatePresence>
                            {processedBatch.map((batch, idx) => (
                                <motion.div
                                    key={idx}
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="bg-purple-100 border border-purple-300 rounded px-4 py-3 shadow-sm text-center"
                                >
                                    <div className="text-xs text-purple-500 mb-1">{batch.desc}</div>
                                    <div className="font-mono font-bold text-purple-800">{batch.shape}</div>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                        {processedBatch.length === 0 && <span className="text-sm text-text-tertiary italic">Collecting batch...</span>}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BatchProcessor;
