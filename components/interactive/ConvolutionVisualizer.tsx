"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';

const ConvolutionVisualizer = () => {
    const inputSize = 5;
    const [kernelSize, setKernelSize] = useState(3);
    const [padding, setPadding] = useState(0);
    const [stride, setStride] = useState(1);
    const [highlightStep, setHighlightStep] = useState<number | null>(null);

    // Calculate Output Size
    // (W - K + 2P) / S + 1
    const outputSize = Math.floor((inputSize + 2 * padding - kernelSize) / stride + 1);

    // Generate Grid Data
    // We visually represent the padded input grid
    const paddedSize = inputSize + 2 * padding;

    // Calculate sliding window positions
    const windows = [];
    for (let y = 0; y <= paddedSize - kernelSize; y += stride) {
        for (let x = 0; x <= paddedSize - kernelSize; x += stride) {
            windows.push({ x, y });
        }
    }

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">卷积算术模拟器 (Convolution Arithmetic)</h3>

            <div className="flex gap-4 mb-6 text-sm">
                <div className="flex flex-col gap-1">
                    <label className="text-text-secondary font-bold">Kernel Size (K)</label>
                    <select value={kernelSize} onChange={(e) => setKernelSize(Number(e.target.value))} className="px-2 py-1 rounded bg-bg-surface border border-border-subtle">
                        <option value={1}>1x1</option>
                        <option value={2}>2x2</option>
                        <option value={3}>3x3</option>
                    </select>
                </div>
                <div className="flex flex-col gap-1">
                    <label className="text-text-secondary font-bold">Padding (P)</label>
                    <select value={padding} onChange={(e) => setPadding(Number(e.target.value))} className="px-2 py-1 rounded bg-bg-surface border border-border-subtle">
                        <option value={0}>0 (Valid)</option>
                        <option value={1}>1 (Same for K=3)</option>
                        <option value={2}>2</option>
                    </select>
                </div>
                <div className="flex flex-col gap-1">
                    <label className="text-text-secondary font-bold">Stride (S)</label>
                    <select value={stride} onChange={(e) => setStride(Number(e.target.value))} className="px-2 py-1 rounded bg-bg-surface border border-border-subtle">
                        <option value={1}>1</option>
                        <option value={2}>2</option>
                    </select>
                </div>
            </div>

            <div className="flex flex-col md:flex-row gap-12 items-center justify-center">
                {/* Input Grid */}
                <div className="relative">
                    <div className="text-xs font-bold text-center mb-2">
                        Input ({inputSize}x{inputSize})
                        {padding > 0 && <span className="text-text-tertiary"> + Padding {padding}</span>}
                    </div>
                    <div
                        className="grid gap-[1px] bg-gray-200 border border-gray-300 relative"
                        style={{
                            gridTemplateColumns: `repeat(${paddedSize}, 24px)`,
                            gridTemplateRows: `repeat(${paddedSize}, 24px)`,
                        }}
                    >
                        {Array.from({ length: paddedSize * paddedSize }).map((_, i) => {
                            const row = Math.floor(i / paddedSize);
                            const col = i % paddedSize;
                            // Check if inside original input
                            const isPadding =
                                row < padding ||
                                row >= paddedSize - padding ||
                                col < padding ||
                                col >= paddedSize - padding;

                            return (
                                <div
                                    key={i}
                                    className={`w-6 h-6 flex items-center justify-center text-[10px]
                                        ${isPadding ? 'bg-gray-100 text-gray-400 dashed-border' : 'bg-blue-100 text-blue-800'}
                                    `}
                                >
                                    {isPadding ? 0 : 1}
                                </div>
                            );
                        })}

                        {/* Sliding Window Highlight */}
                        {windows.map((win, idx) => {
                            const isActive = highlightStep === idx;
                            return (
                                <motion.div
                                    key={idx}
                                    className="absolute border-2 border-accent-primary pointer-events-none z-10"
                                    initial={false}
                                    animate={{
                                        opacity: highlightStep === null || isActive ? (isActive ? 1 : 0.1) : 0,
                                        scale: isActive ? 1.05 : 1
                                    }}
                                    style={{
                                        left: win.x * 25, // 24px + 1px gap
                                        top: win.y * 25,
                                        width: kernelSize * 25 - 1,
                                        height: kernelSize * 25 - 1,
                                    }}
                                />
                            );
                        })}
                    </div>
                </div>

                {/* Arrow */}
                <div className="text-2xl text-text-tertiary">➜</div>

                {/* Output Grid */}
                <div>
                    <div className="text-xs font-bold text-center mb-2">Output ({outputSize}x{outputSize})</div>
                    {outputSize <= 0 ? (
                        <div className="text-red-500 text-xs font-bold">Invalid Config</div>
                    ) : (
                        <div
                            className="grid gap-[1px] bg-gray-200 border border-gray-300"
                            style={{
                                gridTemplateColumns: `repeat(${outputSize}, 24px)`,
                                gridTemplateRows: `repeat(${outputSize}, 24px)`,
                            }}
                        >
                            {Array.from({ length: outputSize * outputSize }).map((_, i) => (
                                <div
                                    key={i}
                                    onMouseEnter={() => setHighlightStep(i)}
                                    onMouseLeave={() => setHighlightStep(null)}
                                    className={`w-6 h-6 cursor-crosshair transition-colors duration-200 flex items-center justify-center text-[9px] font-mono border
                                        ${highlightStep === i ? 'bg-accent-primary text-white border-accent-primary' : 'bg-green-100 text-green-800 border-green-200'}
                                    `}
                                >
                                    y{i}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            <div className="mt-6 text-xs bg-bg-surface p-3 rounded border border-border-subtle font-mono text-text-secondary">
                Output Size = (Input + 2*Pad - Kernel) / Stride + 1
                <br />
                = ({inputSize} + 2*{padding} - {kernelSize}) / {stride} + 1
                <br />
                = <span className="font-bold text-text-primary">{outputSize}</span>
            </div>

            <p className="mt-2 text-[10px] text-text-tertiary">
                * 鼠标悬停在右侧输出 Grid 上，看看它对应左侧 Input 的哪部分感受野 (Receptive Field)。
                <br />
                * 尝试调节 stride=2，观察输出尺寸迅速缩小（下采样）。
            </p>
        </div>
    );
};

export default ConvolutionVisualizer;
