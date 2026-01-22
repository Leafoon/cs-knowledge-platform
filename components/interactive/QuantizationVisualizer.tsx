"use client";

import React, { useState, useMemo } from 'react';

const QuantizationVisualizer = () => {
    const [mode, setMode] = useState<'fp32' | 'int8'>('fp32');

    // Generate mock weight distribution (Gaussian-ish)
    const weights = useMemo(() => {
        const data = [];
        for (let i = 0; i < 50; i++) {
            const x = (i / 50) * 6 - 3; // -3 to 3 range
            const y = Math.exp(-(x * x) / 2); // Bell curve
            data.push({ x, y });
        }
        return data;
    }, []);

    const quantize = (val: number) => {
        // Simple mock quantization logic: round to nearest 0.5 step for visual effect
        // In real int8 it would be discretized to 256 levels
        const step = 0.5;
        return Math.round(val / step) * step;
    };

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">模型量化演示 (Quantization)</h3>

            <div className="flex justify-center gap-6 mb-8">
                <button
                    onClick={() => setMode('fp32')}
                    className={`px-6 py-3 rounded-xl border flex flex-col items-center gap-1 transition-all w-32 ${mode === 'fp32'
                            ? 'border-accent-primary bg-accent-primary/5 ring-2 ring-accent-primary/20'
                            : 'border-border-subtle bg-bg-surface hover:bg-bg-elevated'
                        }`}
                >
                    <div className="text-sm font-bold">FP32</div>
                    <div className="text-[10px] text-text-tertiary">32-bit Float</div>
                    <div className="text-xs font-mono mt-1">4.0 MB</div>
                </button>

                <button
                    onClick={() => setMode('int8')}
                    className={`px-6 py-3 rounded-xl border flex flex-col items-center gap-1 transition-all w-32 ${mode === 'int8'
                            ? 'border-accent-primary bg-accent-primary/5 ring-2 ring-accent-primary/20'
                            : 'border-border-subtle bg-bg-surface hover:bg-bg-elevated'
                        }`}
                >
                    <div className="text-sm font-bold">INT8</div>
                    <div className="text-[10px] text-text-tertiary">8-bit Integer</div>
                    <div className="text-xs font-mono mt-1 text-green-600">1.0 MB</div>
                </button>
            </div>

            {/* Visualization Chart */}
            <div className="relative h-[200px] bg-white dark:bg-slate-900 border border-border-subtle rounded-xl p-4 flex items-end justify-between px-8">
                {/* Zero Line */}
                <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-gray-200 dark:bg-slate-700 dashed" />

                {weights.map((pt, i) => {
                    const val = mode === 'fp32' ? pt.x : quantize(pt.x);
                    const height = pt.y * 150;

                    return (
                        <div
                            key={i}
                            className={`w-1.5 rounded-t-sm transition-all duration-500 ease-in-out ${mode === 'fp32' ? 'bg-blue-400' : 'bg-green-500'
                                }`}
                            style={{
                                height: `${height}px`,
                                // In int8 mode, cluster bars together to show discretization
                                transform: mode === 'int8'
                                    ? `translateX(${(quantize(pt.x) - pt.x) * 20}px)` // visual shift
                                    : 'none'
                            }}
                        />
                    );
                })}

                <div className="absolute top-4 right-4 text-xs text-text-tertiary">
                    Weight Distribution curve
                </div>
            </div>

            <div className="mt-4 text-center text-xs text-text-secondary leading-relaxed space-y-2">
                <p>
                    <span className="font-bold text-blue-500">FP32</span>: 权重是连续、平滑的浮点数。精度高，但占用内存大。
                </p>
                <p>
                    <span className="font-bold text-green-600">INT8</span>: 权重被“吸附”到最近的离散整数点上。精度略有损失，但模型体积减少 75%，推理速度更快。
                </p>
            </div>
        </div>
    );
};

export default QuantizationVisualizer;
