"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Cpu,
    ArrowRight,
    Zap,
    Layers,
    GitMerge,
    CheckCircle2,
    Box,
    Play,
    Pause,
    ChevronRight,
    Terminal,
    ChevronLeft
} from 'lucide-react';
import { cn } from '@/lib/utils';

// --- Simulation Data ---
const WORKFLOW_STEPS = [
    {
        id: 'init',
        title: 'Initialize Accelerator',
        code: `from accelerate import Accelerator\n\naccelerator = Accelerator()`,
        desc: "Auto-detects your environment (Single GPU, DDP, TPU, MPS) and initializes the distributed backend.",
        sysState: "detecting"
    },
    {
        id: 'prepare',
        title: 'Prepare Objects',
        code: `model, optimizer, train_dataloader = accelerator.prepare(\n    model, optimizer, train_dataloader\n)`,
        desc: "Wraps your model in a DDP container, shards the dataloader across GPUs, and moves everything to the correct device.",
        sysState: "wrapping"
    },
    {
        id: 'train',
        title: 'Training Loop',
        code: `for batch in train_dataloader:\n    outputs = model(batch)\n    loss = outputs.loss`,
        desc: "Each GPU receives a unique slice of the data (mini-batch). Forward pass happens in parallel on all devices.",
        sysState: "parallel_forward"
    },
    {
        id: 'backward',
        title: 'Backward & Optimize',
        code: `    accelerator.backward(loss)\n    optimizer.step()`,
        desc: "Automates gradient scaling (mixed precision) and synchronizes gradients (All-Reduce) across all GPUs before updating weights.",
        sysState: "syncing"
    }
];

export default function AccelerateWorkflowVisualizer() {
    const [currentStep, setCurrentStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    useEffect(() => {
        let timer: NodeJS.Timeout;
        if (isPlaying) {
            timer = setInterval(() => {
                setCurrentStep(prev => {
                    if (prev >= WORKFLOW_STEPS.length - 1) {
                        setIsPlaying(false);
                        return prev;
                    }
                    return prev + 1;
                });
            }, 3000);
        }
        return () => clearInterval(timer);
    }, [isPlaying]);

    const step = WORKFLOW_STEPS[currentStep];

    return (
        <div className="my-10 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden flex flex-col">

            {/* 1. Top Bar: Title & Controls */}
            <div className="px-6 py-4 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between bg-slate-50/50 dark:bg-slate-900/50 backdrop-blur-sm">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-yellow-500 rounded-lg flex items-center justify-center text-white shadow-sm">
                        <Zap className="w-5 h-5 fill-current" />
                    </div>
                    <div>
                        <h3 className="font-bold text-slate-800 dark:text-slate-100 text-sm">Accelerate Workflow</h3>
                        <p className="text-xs text-slate-500 dark:text-slate-400">Distributed training in 4 lines of code</p>
                    </div>
                </div>

                <button
                    onClick={() => {
                        if (currentStep === WORKFLOW_STEPS.length - 1) setCurrentStep(0);
                        setIsPlaying(!isPlaying);
                    }}
                    className={cn(
                        "px-3 py-1.5 rounded-full text-xs font-bold transition-all flex items-center gap-2 border",
                        isPlaying
                            ? "bg-red-50 text-red-600 border-red-200 dark:bg-red-900/20 dark:border-red-900/50 dark:text-red-400"
                            : "bg-indigo-50 text-indigo-600 border-indigo-200 dark:bg-indigo-900/20 dark:border-indigo-900/50 dark:text-indigo-400 hover:bg-indigo-100 dark:hover:bg-indigo-900/40"
                    )}
                >
                    {isPlaying ? <><Pause className="w-3 h-3 fill-current" /> Pause</> : <><Play className="w-3 h-3 fill-current" /> Auto Play</>}
                </button>
            </div>

            {/* 2. Horizontal Stepper */}
            <div className="bg-slate-50 dark:bg-slate-950 px-6 py-4 border-b border-slate-100 dark:border-slate-800 overflow-x-auto">
                <div className="flex items-center justify-between min-w-[500px] relative px-4">
                    {/* Background Line */}
                    <div className="absolute left-0 right-0 top-4 h-0.5 bg-slate-200 dark:bg-slate-800 -z-0" />

                    {WORKFLOW_STEPS.map((s, idx) => {
                        const isActive = currentStep === idx;
                        const isCompleted = currentStep > idx;

                        return (
                            <button
                                key={s.id}
                                onClick={() => { setIsPlaying(false); setCurrentStep(idx); }}
                                className="relative z-10 flex flex-col items-center gap-2 group min-w-[80px]"
                            >
                                <div className={cn(
                                    "w-8 h-8 rounded-full flex items-center justify-center border-2 transition-all shadow-sm",
                                    isActive
                                        ? "bg-yellow-500 border-yellow-500 text-white scale-110"
                                        : isCompleted
                                            ? "bg-emerald-500 border-emerald-500 text-white"
                                            : "bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-700 text-slate-400 group-hover:border-slate-400"
                                )}>
                                    {isCompleted ? <CheckCircle2 className="w-4 h-4" /> : <span className="text-xs font-bold">{idx + 1}</span>}
                                </div>
                                <span className={cn(
                                    "text-[10px] font-bold uppercase tracking-wider transition-colors",
                                    isActive ? "text-yellow-600 dark:text-yellow-500" : "text-slate-400 dark:text-slate-500"
                                )}>
                                    {s.id}
                                </span>
                            </button>
                        )
                    })}
                </div>
            </div>

            {/* 3. Main Split View - REMOVED FIXED HEIGHT */}
            <div className="flex flex-col lg:flex-row min-h-[500px]">

                {/* Left: Code & Context */}
                <div className="lg:w-2/5 p-6 bg-white dark:bg-slate-900 border-r border-slate-100 dark:border-slate-800 flex flex-col gap-6">

                    <div className="flex-1">
                        <AnimatePresence mode="wait">
                            <motion.div
                                key={currentStep}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: 10 }}
                                transition={{ duration: 0.2 }}
                                className="space-y-6"
                            >
                                {/* Title & Description */}
                                <div>
                                    <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-2">
                                        {step.title}
                                    </h2>
                                    <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
                                        {step.desc}
                                    </p>
                                </div>

                                {/* Code Block - Now positioned naturally in flow */}
                                <div className="bg-slate-950 rounded-xl overflow-hidden border border-slate-800 shadow-xl">
                                    <div className="px-4 py-2 bg-slate-900/50 flex items-center gap-2 border-b border-slate-800">
                                        <div className="flex gap-1.5">
                                            <div className="w-2.5 h-2.5 rounded-full bg-red-500/20" />
                                            <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/20" />
                                            <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/20" />
                                        </div>
                                        <div className="ml-2 flex items-center gap-1.5 text-[10px] font-mono text-slate-500">
                                            <Terminal className="w-3 h-3" />
                                            <span>train.py</span>
                                        </div>
                                    </div>
                                    <div className="p-4 overflow-x-auto custom-scrollbar">
                                        <pre className="text-sm font-mono text-blue-300 leading-relaxed whitespace-pre font-medium">
                                            {step.code.split('\n').map((line, i) => (
                                                <div key={i} className={cn(
                                                    "w-full",
                                                    line.trim().startsWith('#') ? "text-slate-500 italic" : ""
                                                )}>
                                                    <span className="text-slate-700 select-none inline-block w-6 text-right mr-3 text-[10px]">{i + 1}</span>
                                                    {line}
                                                </div>
                                            ))}
                                        </pre>
                                    </div>
                                </div>
                            </motion.div>
                        </AnimatePresence>
                    </div>

                    {/* Navigation Buttons */}
                    <div className="pt-4 flex justify-between items-center border-t border-slate-100 dark:border-slate-800">
                        <span className="text-xs font-mono text-slate-400">Step {currentStep + 1} / {WORKFLOW_STEPS.length}</span>
                        <div className="flex gap-2">
                            <button
                                disabled={currentStep === 0}
                                onClick={() => { setIsPlaying(false); setCurrentStep(p => p - 1) }}
                                className="px-3 py-1.5 rounded-lg text-sm font-medium text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-1 transition-colors"
                            >
                                <ChevronLeft className="w-4 h-4" /> Prev
                            </button>
                            <button
                                disabled={currentStep === WORKFLOW_STEPS.length - 1}
                                onClick={() => { setIsPlaying(false); setCurrentStep(p => p + 1) }}
                                className="px-4 py-1.5 rounded-lg text-sm font-medium bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 hover:bg-slate-800 dark:hover:bg-slate-200 disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-1 transition-colors shadow-sm"
                            >
                                Next <ChevronRight className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                </div>

                {/* Right: Visualization Stage */}
                <div className="lg:w-3/5 bg-slate-50/50 dark:bg-slate-950/50 relative overflow-hidden flex items-center justify-center p-8 min-h-[400px]">
                    {/* Grid Background */}
                    <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808008_1px,transparent_1px),linear-gradient(to_bottom,#80808008_1px,transparent_1px)] bg-[size:24px_24px]"></div>

                    <AnimatePresence mode="wait">
                        {/* 1. Detecting State */}
                        {step.sysState === 'detecting' && (
                            <motion.div
                                key="detecting"
                                initial={{ opacity: 0, scale: 0.9 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.9 }}
                                className="grid grid-cols-2 gap-8 w-full max-w-sm"
                            >
                                {[0, 1, 2, 3].map(i => (
                                    <motion.div
                                        key={i}
                                        initial={{ y: 20, opacity: 0 }}
                                        animate={{ y: 0, opacity: 1 }}
                                        transition={{ delay: i * 0.15 }}
                                        className="aspect-square bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm flex flex-col items-center justify-center gap-3 relative overflow-hidden"
                                    >
                                        <div className="absolute top-2 left-3 flex gap-1">
                                            <div className="w-1.5 h-1.5 rounded-full bg-red-400" />
                                            <div className="w-1.5 h-1.5 rounded-full bg-yellow-400" />
                                        </div>
                                        <Cpu className="w-8 h-8 text-slate-300 dark:text-slate-600" />
                                        <span className="text-xs font-bold text-slate-500">GPU {i}</span>

                                        <motion.div
                                            initial={{ scale: 0 }}
                                            animate={{ scale: 1 }}
                                            transition={{ delay: 0.5 + i * 0.1 }}
                                            className="absolute bottom-2 right-2 flex items-center gap-1 bg-emerald-100 dark:bg-emerald-900/30 px-2 py-0.5 rounded-full"
                                        >
                                            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                                            <span className="text-[9px] font-bold text-emerald-600 dark:text-emerald-400">Ready</span>
                                        </motion.div>
                                    </motion.div>
                                ))}
                                <motion.div
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{ delay: 1.2 }}
                                    className="col-span-2 text-center mt-4"
                                >
                                    <span className="px-3 py-1 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 text-xs font-bold rounded-full">
                                        Environment: Multi-GPU (DDP)
                                    </span>
                                </motion.div>
                            </motion.div>
                        )}

                        {/* 2. Wrapping State */}
                        {step.sysState === 'wrapping' && (
                            <motion.div
                                key="wrapping"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="w-full max-w-lg flex flex-col gap-8"
                            >
                                {/* Model Wrapping */}
                                <div className="flex items-center justify-center gap-8">
                                    <div className="text-center opacity-40">
                                        <div className="w-16 h-16 bg-white dark:bg-slate-800 border-2 border-dashed border-slate-300 rounded-xl flex items-center justify-center mx-auto mb-2">
                                            <Box className="w-8 h-8 text-slate-400" />
                                        </div>
                                        <span className="text-xs">Raw Model</span>
                                    </div>

                                    <ArrowRight className="text-slate-300 animate-pulse" />

                                    <motion.div
                                        initial={{ scale: 0.8, opacity: 0 }}
                                        animate={{ scale: 1, opacity: 1 }}
                                        className="text-center relative"
                                    >
                                        <div className="w-20 h-20 bg-indigo-600 rounded-2xl flex items-center justify-center text-white shadow-xl shadow-indigo-500/20 mx-auto mb-2 relative z-10">
                                            <Layers className="w-10 h-10" />
                                        </div>
                                        <motion.div
                                            initial={{ opacity: 0, scale: 1.2 }}
                                            animate={{ opacity: 1, scale: 1 }}
                                            transition={{ delay: 0.3 }}
                                            className="absolute -inset-2 border-2 border-yellow-500 rounded-3xl z-0"
                                        />
                                        <div className="absolute -top-3 -right-3 bg-yellow-500 text-white text-[10px] font-bold px-2 py-0.5 rounded-full shadow-sm z-20">
                                            DDP
                                        </div>
                                        <span className="text-xs font-bold text-indigo-600 dark:text-indigo-400">Wrapped Model</span>
                                    </motion.div>
                                </div>

                                {/* DataLoader Sharding */}
                                <div className="bg-white/50 dark:bg-slate-800/50 p-6 rounded-2xl border border-slate-100 dark:border-slate-800">
                                    <div className="flex justify-between mb-4">
                                        <span className="text-xs font-bold text-slate-500">DataLoader Sharding</span>
                                        <DatabaseIcon className="w-4 h-4 text-slate-400" />
                                    </div>
                                    <div className="grid grid-cols-4 gap-2">
                                        {[0, 1, 2, 3].map(i => (
                                            <motion.div
                                                key={i}
                                                initial={{ y: -10, opacity: 0 }}
                                                animate={{ y: 0, opacity: 1 }}
                                                transition={{ delay: 0.5 + i * 0.1 }}
                                                className="h-16 bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-100 dark:border-emerald-800 rounded-lg flex flex-col items-center justify-center gap-1"
                                            >
                                                <div className="text-[10px] font-mono text-emerald-600 dark:text-emerald-400">Shard</div>
                                                <div className="text-lg font-bold text-emerald-700 dark:text-emerald-300">{i}</div>
                                            </motion.div>
                                        ))}
                                    </div>
                                </div>
                            </motion.div>
                        )}

                        {/* 3. Parallel Forward */}
                        {step.sysState === 'parallel_forward' && (
                            <motion.div
                                key="parallel_forward"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="w-full max-w-md"
                            >
                                <div className="text-center mb-6">
                                    <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Parallel Execution</span>
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    {[0, 1, 2, 3].map(i => (
                                        <div key={i} className="bg-white dark:bg-slate-800 p-4 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm relative overflow-hidden group">
                                            <div className="flex items-center justify-between mb-3 z-10 relative">
                                                <div className="flex items-center gap-2">
                                                    <Cpu className="w-4 h-4 text-slate-400" />
                                                    <span className="text-xs font-bold text-slate-600 dark:text-slate-300">GPU {i}</span>
                                                </div>
                                            </div>

                                            {/* Data IN */}
                                            <motion.div
                                                animate={{ x: [0, 100], opacity: [1, 0] }}
                                                transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
                                                className="absolute top-1/2 left-0 w-20 h-1 bg-gradient-to-r from-transparent via-blue-500 to-transparent opacity-50"
                                            />

                                            {/* Compute Animation */}
                                            <div className="flex justify-center gap-1 mt-4">
                                                <motion.div animate={{ height: [10, 20, 10] }} transition={{ repeat: Infinity, duration: 0.8 }} className="w-1.5 bg-indigo-400 rounded-full" />
                                                <motion.div animate={{ height: [10, 24, 10] }} transition={{ repeat: Infinity, duration: 0.8, delay: 0.1 }} className="w-1.5 bg-indigo-500 rounded-full" />
                                                <motion.div animate={{ height: [10, 16, 10] }} transition={{ repeat: Infinity, duration: 0.8, delay: 0.2 }} className="w-1.5 bg-indigo-400 rounded-full" />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </motion.div>
                        )}

                        {/* 4. Syncing */}
                        {step.sysState === 'syncing' && (
                            <motion.div
                                key="syncing"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="relative w-full max-w-md h-[300px] flex items-center justify-center"
                            >
                                {/* Center Hub */}
                                <motion.div
                                    animate={{
                                        boxShadow: ["0 0 0 0px rgba(234, 179, 8, 0.2)", "0 0 0 20px rgba(234, 179, 8, 0)"]
                                    }}
                                    transition={{ repeat: Infinity, duration: 2 }}
                                    className="z-10 w-24 h-24 bg-yellow-500 rounded-full flex flex-col items-center justify-center text-white shadow-xl border-4 border-white dark:border-slate-900"
                                >
                                    <GitMerge className="w-8 h-8 mb-1" />
                                    <span className="text-[10px] font-bold uppercase tracking-wide">Matches</span>
                                </motion.div>

                                {/* Connecting Lines & Particles */}
                                {[0, 1, 2, 3].map((i) => {
                                    const angle = (i * 90) * (Math.PI / 180) + Math.PI / 4;
                                    const r = 120;
                                    const x = r * Math.cos(angle);
                                    const y = r * Math.sin(angle);

                                    return (
                                        <React.Fragment key={i}>
                                            {/* Node */}
                                            <motion.div
                                                initial={{ x: 0, y: 0 }}
                                                animate={{ x, y }}
                                                className="absolute w-12 h-12 bg-indigo-50 dark:bg-slate-800 rounded-xl border border-indigo-200 dark:border-indigo-900 flex items-center justify-center z-0"
                                            >
                                                <Cpu className="w-5 h-5 text-indigo-500" />
                                            </motion.div>

                                            {/* Particle Trail */}
                                            <svg className="absolute inset-0 pointer-events-none w-full h-full overflow-visible">
                                                <motion.circle
                                                    cx="50%" cy="50%" r="4" fill="#ef4444"
                                                    initial={{ transform: `translate(${x}px, ${y}px)` }}
                                                    animate={{ transform: "translate(0px, 0px)" }}
                                                    transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
                                                />
                                            </svg>
                                        </React.Fragment>
                                    );
                                })}

                                <div className="absolute bottom-0 text-center">
                                    <div className="font-bold text-slate-700 dark:text-slate-200 text-sm">Gradient Synchronization</div>
                                    <div className="text-xs text-slate-400">All-Reduce Operation</div>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
}

function DatabaseIcon({ className }: { className?: string }) {
    return (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
            <ellipse cx="12" cy="5" rx="9" ry="3" />
            <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
            <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
        </svg>
    )
}
