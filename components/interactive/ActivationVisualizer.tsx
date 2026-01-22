"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';

const ActivationVisualizer = () => {
    const [func, setFunc] = useState<'relu' | 'sigmoid' | 'tanh' | 'gelu'>('relu');
    const [inputValue, setInputValue] = useState(0); // Input x

    // Range for plotting
    const points = [];
    for (let x = -6; x <= 6; x += 0.2) {
        points.push(x);
    }

    const calculate = (x: number, type: string) => {
        switch (type) {
            case 'relu': return Math.max(0, x);
            case 'sigmoid': return 1 / (1 + Math.exp(-x));
            case 'tanh': return Math.tanh(x);
            case 'gelu': return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
            default: return 0;
        }
    };

    const getDescription = (type: string) => {
        switch (type) {
            case 'relu': return "Rectified Linear Unit: max(0, x). 解决梯度消失的首选，计算极快。但在 x<0 时梯度为 0 (Dead ReLU)。";
            case 'sigmoid': return "1 / (1 + e^-x). 将输出压缩到 (0, 1)。通常仅用于二分类的输出层。容易导致梯度消失。";
            case 'tanh': return "双曲正切。将输出压缩到 (-1, 1)。零中心化 (Zero-centered)，优于 Sigmoid，但仍有梯度消失问题。";
            case 'gelu': return "Gaussian Error Linear Unit. BERT/GPT 的标配。比 ReLU 更平滑，允许负值微小流动。";
        }
    };

    const currentY = calculate(inputValue, func);

    // Chart dimensions
    const width = 300;
    const height = 200;
    const padding = 20;

    // Scale helpers
    const scaleX = (x: number) => (x + 6) / 12 * width;
    const scaleY = (y: number) => {
        // Different Y ranges for different funcs
        if (func === 'relu') return height - (y / 6) * height; // 0 to 6
        if (func === 'gelu') return height - ((y + 2) / 8) * height;
        return height - ((y + 1.2) / 2.4) * height; // -1.2 to 1.2 approx
    };

    // Generate Path
    const pathData = points.map((x, i) => {
        const y = calculate(x, func);
        return `${i === 0 ? 'M' : 'L'} ${scaleX(x)} ${scaleY(y)}`;
    }).join(' ');

    return (
        <div className="w-full max-w-2xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">激活函数图鉴 (Activation Functions)</h3>

            <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
                {['relu', 'sigmoid', 'tanh', 'gelu'].map((f) => (
                    <button
                        key={f}
                        onClick={() => setFunc(f as any)}
                        className={`px-4 py-1.5 rounded-full text-sm font-bold capitalize transition-colors ${func === f
                            ? 'bg-accent-primary text-white shadow-md'
                            : 'bg-bg-surface border border-border-subtle hover:bg-bg-elevated text-text-secondary'
                            }`}
                    >
                        {f}
                    </button>
                ))}
            </div>

            <div className="flex flex-col md:flex-row items-center gap-8">
                {/* Graph */}
                <div className="relative w-[300px] h-[200px] border-b border-l border-text-tertiary">
                    {/* Zero Axes */}
                    <div className="absolute top-0 bottom-0 left-1/2 w-px bg-gray-200 dashed" style={{ left: scaleX(0) }}></div>
                    <div className="absolute left-0 right-0 w-full h-px bg-gray-200 dashed" style={{ top: scaleY(0) }}></div>

                    <svg width={width} height={height} className="overflow-visible">
                        <motion.path
                            key={func}
                            d={pathData}
                            fill="none"
                            stroke="var(--accent-primary)"
                            strokeWidth="3"
                            initial={{ pathLength: 0, opacity: 0 }}
                            animate={{ pathLength: 1, opacity: 1 }}
                            transition={{ duration: 0.5 }}
                        />

                        {/* Interactive Dot */}
                        <motion.circle
                            cx={scaleX(inputValue)}
                            cy={scaleY(currentY)}
                            r="6"
                            fill="white"
                            stroke="var(--accent-primary)"
                            strokeWidth="2"
                        />
                    </svg>

                    {/* Input Slider */}
                    <input
                        type="range"
                        min="-6" max="6" step="0.1"
                        value={inputValue}
                        onChange={(e) => setInputValue(parseFloat(e.target.value))}
                        className="absolute -bottom-8 left-0 right-0 w-full accent-accent-primary"
                    />
                </div>

                {/* Info Panel */}
                <div className="flex-1 space-y-4">
                    <div className="p-4 bg-bg-surface rounded-xl border border-border-subtle">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-xs text-text-tertiary">Input x</span>
                            <span className="font-mono font-bold">{inputValue.toFixed(2)}</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-xs text-text-tertiary">Output y</span>
                            <span className="font-mono font-bold text-accent-primary text-lg">{currentY.toFixed(4)}</span>
                        </div>
                    </div>

                    <div className="text-sm text-text-secondary leading-relaxed">
                        {getDescription(func)}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ActivationVisualizer;
