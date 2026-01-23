"use client";

import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

const TrainingSimulator = () => {
    const [lr, setLr] = useState(0.01);
    const [isTraining, setIsTraining] = useState(false);
    const [epoch, setEpoch] = useState(0);
    const [lossHistory, setLossHistory] = useState<number[]>([]);
    const [weights, setWeights] = useState(2.0); // Initial guess
    const targetWeight = 4.0; // We want to learn y = 4x

    // Reset
    const reset = () => {
        setIsTraining(false);
        setEpoch(0);
        setLossHistory([]);
        setWeights(Math.random() * 8 - 4); // Random start between -4 and 4
    };

    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isTraining) {
            interval = setInterval(() => {
                setEpoch(prev => {
                    if (prev >= 50) { // Max epochs
                        setIsTraining(false);
                        return prev;
                    }
                    return prev + 1;
                });

                setWeights(w => {
                    // Simple Gradient Descent: loss = (w - target)^2
                    // d(loss)/dw = 2 * (w - target)
                    const gradient = 2 * (w - targetWeight);

                    // Add some noise to simulate SGD variance
                    const noise = (Math.random() - 0.5) * 5.0 * lr;

                    const newW = w - lr * gradient + noise;
                    return newW;
                });
            }, 100);
        }
        return () => clearInterval(interval);
    }, [isTraining, lr, targetWeight]);

    // Record loss history
    useEffect(() => {
        // Loss = MSE = (w - target)^2
        const currentLoss = Math.pow(weights - targetWeight, 2);
        setLossHistory(prev => [...prev, currentLoss]);
    }, [weights, targetWeight]);

    // Calculate max loss for scaling visualization
    const maxLoss = Math.max(20, ...lossHistory);

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">训练动态模拟器 (SGD)</h3>

            <div className="flex flex-col md:flex-row gap-6 mb-6">
                <div className="flex-1 space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-text-secondary mb-1">
                            Learning Rate (学习率): <span className="font-mono text-accent-primary">{lr}</span>
                        </label>
                        <input
                            type="range"
                            min="0.001"
                            max="0.5"
                            step="0.001"
                            value={lr}
                            onChange={(e) => {
                                setLr(parseFloat(e.target.value));
                                reset();
                            }}
                            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-accent-primary"
                        />
                        <div className="flex justify-between text-xs text-text-tertiary px-1">
                            <span>0.001 (Slow)</span>
                            <span>0.5 (Unstable)</span>
                        </div>
                    </div>

                    <div className="flex gap-4">
                        <button
                            onClick={() => setIsTraining(!isTraining)}
                            className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${isTraining
                                    ? 'bg-red-50 text-red-600 border border-red-200 hover:bg-red-100'
                                    : 'bg-accent-primary text-white hover:bg-accent-primary/90'
                                }`}
                        >
                            {isTraining ? 'Pause' : epoch >= 50 ? 'Finished' : 'Start Training'}
                        </button>
                        <button
                            onClick={reset}
                            className="px-4 py-2 bg-bg-surface border border-border-subtle rounded-lg text-text-secondary hover:bg-bg-elevated transition-colors"
                        >
                            Reset
                        </button>
                    </div>

                    <div className="p-3 bg-bg-surface rounded border border-border-subtle text-sm space-y-1">
                        <div className="flex justify-between">
                            <span className="text-text-secondary">Epoch:</span>
                            <span className="font-mono font-bold">{epoch}/50</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-text-secondary">Loss:</span>
                            <span className={`font-mono font-bold ${lossHistory[lossHistory.length - 1] < 0.1 ? 'text-green-600' : 'text-text-primary'}`}>
                                {lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].toFixed(4) : '---'}
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-text-secondary">Current Weight:</span>
                            <span className="font-mono">{weights.toFixed(2)} (Target: 4.00)</span>
                        </div>
                    </div>

                    <p className="text-xs text-text-tertiary leading-relaxed">
                        * 调整学习率并重置，观察收敛速度和稳定性。
                        <br />
                        LR 太小 {'->'} 收敛极慢
                        <br />
                        LR 太大 {'->'} Loss 震荡甚至发散
                    </p>
                </div>

                {/* SVG Chart */}
                <div className="flex-1 bg-white dark:bg-slate-900 border border-border-subtle rounded-lg p-2 relative h-[240px]">
                    <div className="absolute top-2 right-2 text-xs text-text-tertiary">Loss Curve</div>
                    <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                        {/* Grid Lines */}
                        <line x1="0" y1="90" x2="100" y2="90" stroke="#e2e8f0" strokeWidth="1" />
                        <line x1="10" y1="0" x2="10" y2="100" stroke="#e2e8f0" strokeWidth="1" />

                        {/* Plot Line */}
                        <polyline
                            fill="none"
                            stroke="var(--accent-primary)"
                            strokeWidth="2"
                            points={lossHistory.map((l, i) => {
                                const x = (i / 50) * 100;
                                const y = 90 - (Math.min(l, maxLoss) / maxLoss) * 80;
                                return `${x},${y}`;
                            }).join(' ')}
                        />
                    </svg>

                    {/* Axis Labels */}
                    <div className="absolute left-2 bottom-0 text-[10px] text-text-tertiary">0</div>
                    <div className="absolute right-2 bottom-0 text-[10px] text-text-tertiary">50 Epochs</div>
                </div>
            </div>
        </div>
    );
};

export default TrainingSimulator;
