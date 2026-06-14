"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Server,
    Cpu,
    ArrowRight,
    LayoutGrid,
    MessageSquare,
    Zap,
    GitMerge,
    Database,
    Clock,
    Layers
} from 'lucide-react';
import { cn } from '@/lib/utils';

export default function TGIArchitectureDiagram() {
    const [activeStage, setActiveStage] = useState<string | null>(null);
    const [isAnimating, setIsAnimating] = useState(false);

    // Auto-play simulation
    useEffect(() => {
        if (isAnimating) {
            const sequence = [
                { stage: 'router', delay: 0 },
                { stage: 'scheduler', delay: 1500 },
                { stage: 'shards', delay: 3000 },
                { stage: 'response', delay: 5000 }
            ];

            let timeouts: NodeJS.Timeout[] = [];

            sequence.forEach(({ stage, delay }) => {
                const timeout = setTimeout(() => setActiveStage(stage), delay);
                timeouts.push(timeout);
            });

            const finishTimeout = setTimeout(() => {
                setIsAnimating(false);
                setActiveStage(null);
            }, 6500);
            timeouts.push(finishTimeout);

            return () => timeouts.forEach(clearTimeout);
        }
    }, [isAnimating]);

    const handleStartSim = () => {
        setIsAnimating(true);
        setActiveStage('router');
    };

    return (
        <div className="my-10 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden flex flex-col">

            {/* Header */}
            <div className="p-6 bg-slate-50 dark:bg-slate-950 border-b border-slate-100 dark:border-slate-800 flex justify-between items-center">
                <div>
                    <h3 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
                        <Server className="w-5 h-5 text-indigo-500" />
                        TGI Architecture (Text Generation Inference)
                    </h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                        High-performance Rust-based inference server architecture
                    </p>
                </div>
                <button
                    onClick={handleStartSim}
                    disabled={isAnimating}
                    className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                    <Zap className="w-4 h-4 fill-current" />
                    {isAnimating ? 'Simulating...' : 'Simulate Request'}
                </button>
            </div>

            {/* Diagram Area */}
            <div className="p-8 relative min-h-[400px] flex items-center justify-center bg-slate-50/50 dark:bg-slate-900/50">
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808008_1px,transparent_1px),linear-gradient(to_bottom,#80808008_1px,transparent_1px)] bg-[size:24px_24px]"></div>

                <div className="relative z-10 grid grid-cols-1 lg:grid-cols-3 gap-8 w-full max-w-4xl">

                    {/* 1. Router (Rust) */}
                    <div className="relative group">
                        <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400 text-[10px] font-bold uppercase rounded-full border border-orange-200 dark:border-orange-800 z-20">
                            Rust / Axum
                        </div>
                        <motion.div
                            className={cn(
                                "h-full bg-white dark:bg-slate-800 rounded-xl border-2 p-6 flex flex-col gap-4 transition-all duration-300 relative overflow-hidden",
                                activeStage === 'router'
                                    ? "border-orange-500 shadow-xl shadow-orange-500/10 scale-105 z-10"
                                    : "border-slate-200 dark:border-slate-700 opacity-80"
                            )}
                            onClick={() => setActiveStage('router')}
                        >
                            <div className="flex items-center gap-3 mb-2">
                                <div className="w-10 h-10 bg-orange-500 rounded-lg flex items-center justify-center text-white">
                                    <LayoutGrid className="w-6 h-6" />
                                </div>
                                <div>
                                    <h4 className="font-bold text-slate-800 dark:text-slate-100">Router</h4>
                                    <span className="text-xs text-slate-500">Web Server & Queue</span>
                                </div>
                            </div>

                            <div className="text-xs text-slate-600 dark:text-slate-400 space-y-2">
                                <div className="flex items-center gap-2">
                                    <MessageSquare className="w-3 h-3" />
                                    <span>Accepts HTTP / gRPC</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <Database className="w-3 h-3" />
                                    <span>Tokenization</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <Clock className="w-3 h-3" />
                                    <span>Request Queueing</span>
                                </div>
                            </div>

                            {/* Request Animation */}
                            <AnimatePresence>
                                {activeStage === 'router' && (
                                    <motion.div
                                        initial={{ x: -20, opacity: 0 }}
                                        animate={{ x: 0, opacity: 1 }}
                                        exit={{ x: 20, opacity: 0 }}
                                        className="absolute top-2 right-2"
                                    >
                                        <div className="w-3 h-3 bg-orange-500 rounded-full animate-ping" />
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </motion.div>

                        {/* Connection Line */}
                        <div className="hidden lg:block absolute top-1/2 -right-4 w-8 h-1 bg-slate-200 dark:bg-slate-700">
                            <motion.div
                                animate={activeStage === 'router' || activeStage === 'scheduler' ? { x: [0, 32], opacity: [1, 0] } : {}}
                                transition={{ repeat: Infinity, duration: 1 }}
                                className="h-full w-1/2 bg-indigo-500"
                            />
                        </div>
                    </div>

                    {/* 2. Scheduler & Batcher */}
                    <div className="relative group">
                        <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 text-[10px] font-bold uppercase rounded-full border border-blue-200 dark:border-blue-800 z-20">
                            Core Logic
                        </div>
                        <motion.div
                            className={cn(
                                "h-full bg-white dark:bg-slate-800 rounded-xl border-2 p-6 flex flex-col gap-4 transition-all duration-300 relative overflow-hidden",
                                activeStage === 'scheduler'
                                    ? "border-blue-500 shadow-xl shadow-blue-500/10 scale-105 z-10"
                                    : "border-slate-200 dark:border-slate-700 opacity-80"
                            )}
                            onClick={() => setActiveStage('scheduler')}
                        >
                            <div className="flex items-center gap-3 mb-2">
                                <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center text-white">
                                    <Layers className="w-6 h-6" />
                                </div>
                                <div>
                                    <h4 className="font-bold text-slate-800 dark:text-slate-100">Scheduler</h4>
                                    <span className="text-xs text-slate-500">Continuous Batching</span>
                                </div>
                            </div>

                            <div className="flex-1 bg-slate-100 dark:bg-slate-900/50 rounded-lg p-2 border border-slate-200 dark:border-slate-700 overflow-hidden relative">
                                <span className="text-[10px] font-mono text-slate-400 absolute top-1 right-2">Batch State</span>
                                <div className="mt-4 space-y-1">
                                    {[1, 2, 3].map(i => (
                                        <motion.div
                                            key={i}
                                            initial={{ width: "20%" }}
                                            animate={activeStage === 'scheduler' ? { width: ["20%", "80%", "40%"] } : {}}
                                            transition={{ duration: 2, repeat: Infinity, repeatType: "reverse", delay: i * 0.2 }}
                                            className="h-2 bg-blue-400 rounded-full"
                                        />
                                    ))}
                                </div>
                            </div>
                        </motion.div>

                        {/* Connection Line */}
                        <div className="hidden lg:block absolute top-1/2 -right-4 w-8 h-1 bg-slate-200 dark:bg-slate-700">
                            <motion.div
                                animate={activeStage === 'scheduler' || activeStage === 'shards' ? { x: [0, 32], opacity: [1, 0] } : {}}
                                transition={{ repeat: Infinity, duration: 1 }}
                                className="h-full w-1/2 bg-indigo-500"
                            />
                        </div>
                    </div>

                    {/* 3. Model Shards (C++ / CUDA) */}
                    <div className="relative group">
                        <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 text-[10px] font-bold uppercase rounded-full border border-emerald-200 dark:border-emerald-800 z-20">
                            Python / C++
                        </div>
                        <motion.div
                            className={cn(
                                "h-full bg-white dark:bg-slate-800 rounded-xl border-2 p-6 flex flex-col gap-4 transition-all duration-300 relative overflow-hidden",
                                activeStage === 'shards'
                                    ? "border-emerald-500 shadow-xl shadow-emerald-500/10 scale-105 z-10"
                                    : "border-slate-200 dark:border-slate-700 opacity-80"
                            )}
                            onClick={() => setActiveStage('shards')}
                        >
                            <div className="flex items-center gap-3 mb-2">
                                <div className="w-10 h-10 bg-emerald-500 rounded-lg flex items-center justify-center text-white">
                                    <Cpu className="w-6 h-6" />
                                </div>
                                <div>
                                    <h4 className="font-bold text-slate-800 dark:text-slate-100">Shards</h4>
                                    <span className="text-xs text-slate-500">Tensor Parallelism</span>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-2 mt-2">
                                {[0, 1, 2, 3].map(i => (
                                    <div key={i} className="bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-2 rounded flex items-center gap-2">
                                        <div className={cn(
                                            "w-2 h-2 rounded-full",
                                            activeStage === 'shards' ? "bg-emerald-500 animate-pulse" : "bg-slate-400"
                                        )} />
                                        <span className="text-[10px] font-mono text-slate-500">GPU{i}</span>
                                    </div>
                                ))}
                            </div>

                            <div className="text-center mt-2">
                                <span className={cn(
                                    "text-[10px] font-bold uppercase tracking-wider",
                                    activeStage === 'shards' ? "text-emerald-600" : "text-slate-400"
                                )}>
                                    NCCL Sync
                                </span>
                            </div>
                        </motion.div>
                    </div>

                </div>

            </div>

            {/* Explanation Footer */}
            <div className="h-24 bg-slate-50 dark:bg-slate-950/30 border-t border-slate-100 dark:border-slate-800 flex items-center justify-center p-4">
                <AnimatePresence mode="wait">
                    {activeStage ? (
                        <motion.div
                            key={activeStage}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="text-center max-w-2xl"
                        >
                            <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                                {activeStage === 'router' && "Router receives request, validates headers, and pushes to queue."}
                                {activeStage === 'scheduler' && "Scheduler pulls requests, forms continuous batches, and manages KV Cache."}
                                {activeStage === 'shards' && "Model Shards execute forward pass in parallel (TP), syncing via NCCL."}
                                {activeStage === 'response' && "Tokens are streamed back through Router to Client."}
                            </p>
                        </motion.div>
                    ) : (
                        <motion.p
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="text-sm text-slate-400 italic"
                        >
                            Click "Simulate Request" or select a component to see details.
                        </motion.p>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}
