"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Checkpoint {
    epoch: number;
    loss: number;
    optimizer_step: number;
    timestamp: string;
}

const CheckpointSimulator = () => {
    const [epoch, setEpoch] = useState(0);
    const [loss, setLoss] = useState(2.5);
    const [isTraining, setIsTraining] = useState(false);
    const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
    const [restoringId, setRestoringId] = useState<number | null>(null);

    // Training Loop Simulation
    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isTraining) {
            interval = setInterval(() => {
                setEpoch(e => e + 1);
                setLoss(l => Math.max(0.1, l * 0.95 + (Math.random() - 0.5) * 0.1));
            }, 500);
        }
        return () => clearInterval(interval);
    }, [isTraining]);

    const saveCheckpoint = () => {
        const ckpt = {
            epoch,
            loss,
            optimizer_step: epoch * 100, // Dummy step count
            timestamp: new Date().toLocaleTimeString(),
        };
        setCheckpoints(prev => [ckpt, ...prev].slice(0, 3)); // Keep last 3
    };

    const loadCheckpoint = (ckpt: Checkpoint, index: number) => {
        setRestoringId(index);
        setIsTraining(false);

        setTimeout(() => {
            setRestoringId(null);
            setEpoch(ckpt.epoch);
            setLoss(ckpt.loss);
        }, 800);
    };

    const crash = () => {
        setIsTraining(false);
        setEpoch(0);
        setLoss(2.5); // Reset to initial bad state
    };

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">Checkpoint 机制演示</h3>

            <div className="flex gap-8 items-start">
                {/* Current State Panel */}
                <div className="flex-1 bg-white dark:bg-slate-900 border border-border-subtle rounded-xl p-6 relative overflow-hidden">
                    {restoringId !== null && (
                        <div className="absolute inset-0 bg-accent-primary/10 flex items-center justify-center z-10 backdrop-blur-[1px]">
                            <span className="bg-white dark:bg-slate-800 px-3 py-1 rounded shadow text-accent-primary font-bold animate-pulse">Restoring...</span>
                        </div>
                    )}

                    <div className="text-xs font-bold text-text-secondary uppercase mb-4 tracking-wider">RAM (Memory State)</div>

                    <div className="space-y-4">
                        <div className="flex justify-between items-center">
                            <span className="text-sm text-text-secondary">Current Epoch:</span>
                            <span className="font-mono text-xl font-bold text-accent-primary animate-pulse">{epoch}</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-sm text-text-secondary">Current Loss:</span>
                            <span className="font-mono text-lg">{loss.toFixed(4)}</span>
                        </div>
                    </div>

                    <div className="mt-8 flex gap-2">
                        <button
                            onClick={() => setIsTraining(!isTraining)}
                            className={`flex-1 py-2 rounded-lg font-medium text-sm transition-colors ${isTraining ? 'bg-orange-100 text-orange-700' : 'bg-green-100 text-green-700'}`}
                        >
                            {isTraining ? 'Pause Training' : 'Resume Training'}
                        </button>
                        <button
                            onClick={crash}
                            className="flex-1 py-2 rounded-lg font-medium text-sm bg-red-100 text-red-700 hover:bg-red-200"
                        >
                            🔥 Simulate Crash
                        </button>
                    </div>
                </div>

                {/* Disk Storage Panel */}
                <div className="flex-1 space-y-4">
                    <div className="flex justify-between items-center">
                        <span className="text-xs font-bold text-text-secondary uppercase tracking-wider">Disk Storage (checkpoint.pt)</span>
                        <button
                            onClick={saveCheckpoint}
                            className="px-3 py-1 bg-blue-100 text-blue-700 text-xs rounded hover:bg-blue-200 transition-colors"
                        >
                            💾 Save Now
                        </button>
                    </div>

                    <div className="space-y-2 min-h-[160px]">
                        <AnimatePresence>
                            {checkpoints.length === 0 && (
                                <div className="text-center text-xs text-text-tertiary py-8 border-2 border-dashed border-border-subtle rounded-lg">
                                    No checkpoints saved
                                </div>
                            )}
                            {checkpoints.map((ckpt, i) => (
                                <motion.div
                                    key={ckpt.timestamp}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, scale: 0.9 }}
                                    className="bg-bg-surface border border-border-subtle rounded-lg p-3 flex justify-between items-center shadow-sm group hover:border-accent-primary/50 transition-colors"
                                >
                                    <div className="text-xs">
                                        <div className="font-bold text-text-primary">Epoch {ckpt.epoch}</div>
                                        <div className="text-text-tertiary">{ckpt.timestamp} · Loss {ckpt.loss.toFixed(2)}</div>
                                    </div>
                                    <button
                                        onClick={() => loadCheckpoint(ckpt, i)}
                                        disabled={isTraining}
                                        className="opacity-0 group-hover:opacity-100 px-2 py-1 bg-accent-primary text-white text-xs rounded hover:bg-accent-primary/90 disabled:opacity-50 transition-opacity"
                                    >
                                        Load
                                    </button>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                    </div>

                    <p className="text-[10px] text-text-tertiary leading-relaxed">
                        * 模拟场景：训练过程中经常需要保存（例如每 Epoch 一次）。如果发生 Crash（如断电、OOM），内存数据丢失（左侧归零），必须从磁盘（右侧）加载最新的 Checkpoint 恢复。
                    </p>
                </div>
            </div>
        </div>
    );
};

export default CheckpointSimulator;
