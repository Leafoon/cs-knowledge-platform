"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const TensorStorageVisualizer = () => {
    // 2D Tensor (3x3)
    // Values: 0, 1, 2, 3, 4, 5, 6, 7, 8

    // View Modes:
    // 1. Standard (Stride: [3, 1])
    // 2. Transposed (Stride: [1, 3])
    // 3. Sliced (::2)

    type Mode = 'standard' | 'transposed' | 'sliced';
    const [mode, setMode] = useState<Mode>('standard');
    const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

    const storage = [0, 1, 2, 3, 4, 5, 6, 7, 8];

    const getMapping = (m: Mode) => {
        // Returns 2D grid where each cell contains the index into storage
        switch (m) {
            case 'standard':
                // Row-major
                return [
                    [0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8]
                ];
            case 'transposed':
                // Col-major (logical transpose)
                // Logical (0,1) maps to storage[3] in standard, BUT logical transpose means:
                // Logical (i,j) -> maps to storage[offset + i*stride0 + j*stride1]
                // Transposed stride: [1, 3]
                // (0,0)->0, (0,1)->3, (0,2)->6...
                return [
                    [0, 3, 6],
                    [1, 4, 7],
                    [2, 5, 8]
                ];
            case 'sliced':
                // Slice [::2, ::2] (Top-left, top-right, bot-left, bot-right)
                // Strides * 2
                return [
                    [0, 2],
                    [6, 8]
                ];
        }
    };

    const grid = getMapping(mode);

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">Tensor: 视图 vs 存储</h3>

            <div className="flex gap-4 mb-8 text-sm">
                <button
                    onClick={() => setMode('standard')}
                    className={`px-3 py-1.5 rounded-lg border transition-colors ${mode === 'standard' ? 'bg-blue-100 border-blue-400 text-blue-700' : 'bg-white border-gray-200'}`}
                >
                    Standard (3x3)
                </button>
                <button
                    onClick={() => setMode('transposed')}
                    className={`px-3 py-1.5 rounded-lg border transition-colors ${mode === 'transposed' ? 'bg-blue-100 border-blue-400 text-blue-700' : 'bg-white border-gray-200'}`}
                >
                    Permute/Transpose
                </button>
                <button
                    onClick={() => setMode('sliced')}
                    className={`px-3 py-1.5 rounded-lg border transition-colors ${mode === 'sliced' ? 'bg-blue-100 border-blue-400 text-blue-700' : 'bg-white border-gray-200'}`}
                >
                    Slice (::2)
                </button>
            </div>

            <div className="flex flex-col md:flex-row gap-12 items-center justify-center">
                {/* 1. Logical View (The Tensor you operate on) */}
                <div className="flex flex-col items-center">
                    <div className="text-xs font-bold uppercase mb-2 text-blue-600">Logical View (Shape: {mode === 'sliced' ? '2x2' : '3x3'})</div>
                    <div className="bg-blue-50/50 p-3 rounded-xl border border-blue-200">
                        <div className={`grid gap-1 ${mode === 'sliced' ? 'grid-rows-2' : 'grid-rows-3'}`}>
                            {grid.map((row, r) => (
                                <div key={r} className="flex gap-1">
                                    {row.map((storageIdx, c) => (
                                        <div
                                            key={c}
                                            onMouseEnter={() => setHoveredIdx(storageIdx)}
                                            onMouseLeave={() => setHoveredIdx(null)}
                                            className={`w-10 h-10 flex items-center justify-center border bg-white rounded font-mono cursor-pointer transition-all duration-200
                                                ${hoveredIdx === storageIdx
                                                    ? 'border-accent-primary bg-accent-primary text-white scale-110 shadow-lg'
                                                    : 'border-blue-100 text-text-primary hover:border-blue-400'}
                                            `}
                                        >
                                            {storage[storageIdx]}
                                        </div>
                                    ))}
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="mt-2 text-[10px] text-text-tertiary font-mono">
                        {mode === 'standard' && "Stride: (3, 1)"}
                        {mode === 'transposed' && "Stride: (1, 3)"}
                        {mode === 'sliced' && "Stride: (6, 2)"}
                    </div>
                </div>

                {/* Arrow */}
                <div className="text-2xl text-gray-300 md:rotate-0 rotate-90">⬇</div>

                {/* 2. Physical Storage (1D Array) */}
                <div className="flex flex-col items-center w-full max-w-xs">
                    <div className="text-xs font-bold uppercase mb-2 text-gray-600">Physical Memory (1D Storage)</div>
                    <div className="flex w-full overflow-hidden rounded-lg border border-gray-200">
                        {storage.map((val, idx) => (
                            <div
                                key={idx}
                                className={`flex-1 h-12 flex items-center justify-center text-xs font-mono border-r last:border-r-0 border-gray-100 transition-colors duration-200
                                    ${hoveredIdx === idx ? 'bg-accent-primary text-white font-bold' : 'bg-gray-50 text-gray-500'}
                                `}
                            >
                                {val}
                            </div>
                        ))}
                    </div>
                    <div className="flex w-full justify-between px-1 mt-1 text-[10px] text-gray-400 font-mono">
                        <span>0</span>
                        <span>Offset</span>
                        <span>8</span>
                    </div>
                </div>
            </div>

            <div className="mt-6 p-3 bg-yellow-50 border border-yellow-100 rounded text-xs text-yellow-800">
                💡 <strong>关键点</strong>: 无论 Logical View 怎么变换（转置、切片），Physical Memory 中的数据从未移动！
                PyTorch 只是修改了 <code>stride</code> 和 <code>offset</code>。这就是 "Zero-copy View" 的魔力。
            </div>
        </div>
    );
};

export default TensorStorageVisualizer;
