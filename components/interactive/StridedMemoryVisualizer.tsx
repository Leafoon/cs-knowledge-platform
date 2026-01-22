"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';

const StridedMemoryVisualizer = () => {
    // 1D Storage: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] (12 elements)
    const storage = Array.from({ length: 12 }, (_, i) => i);

    // View Config
    const [shape, setShape] = useState<[number, number]>([3, 4]); // 3 rows, 4 cols
    const [stride, setStride] = useState<[number, number]>([4, 1]); // Default (Contiguous)
    const [offset, setOffset] = useState(0);

    // Presets
    const applyPreset = (name: string) => {
        if (name === 'contiguous') {
            setShape([3, 4]);
            setStride([4, 1]);
            setOffset(0);
        } else if (name === 'transpose') {
            setShape([4, 3]); // Swap dimensions
            setStride([1, 4]); // Swap strides
            setOffset(0);
        } else if (name === 'slice') {
            setShape([2, 2]);
            setStride([4, 2]); // Skip elements
            setOffset(1);
        }
    };

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">Strided Memory: 零拷贝的秘密</h3>

            <div className="flex justify-center gap-4 mb-6">
                <button onClick={() => applyPreset('contiguous')} className="px-3 py-1 bg-white border rounded text-xs hover:bg-gray-50">Reset (3x4)</button>
                <button onClick={() => applyPreset('transpose')} className="px-3 py-1 bg-white border rounded text-xs hover:bg-gray-50">Transpose (T)</button>
                <button onClick={() => applyPreset('slice')} className="px-3 py-1 bg-white border rounded text-xs hover:bg-gray-50">Slice (::2)</button>
            </div>

            <div className="flex flex-col gap-8">
                {/* 1. Underlying Storage (1D Linear Memory) */}
                <div className="relative">
                    <div className="text-xs font-bold text-text-secondary mb-2 uppercase tracking-wide">Physical Storage (1D, Contiguous)</div>
                    <div className="flex gap-1 overflow-x-auto pb-2">
                        {storage.map((val, idx) => (
                            <div
                                key={idx}
                                className="w-8 h-8 flex-shrink-0 border border-gray-300 bg-gray-100 flex items-center justify-center text-xs font-mono text-gray-400 relative"
                            >
                                {val}
                                <span className="absolute -bottom-4 text-[9px] text-gray-400">{idx}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* 2. Tensor View (2D Logical) */}
                <div>
                    <div className="flex justify-between items-end mb-2">
                        <div className="text-xs font-bold text-text-secondary uppercase tracking-wide">Logical View (Tensor)</div>
                        <div className="text-xs font-mono bg-blue-50 text-blue-700 px-2 py-1 rounded border border-blue-200">
                            Size: [{shape[0]}, {shape[1]}] | Stride: [{stride[0]}, {stride[1]}] | Offset: {offset}
                        </div>
                    </div>

                    <div
                        className="grid gap-1 inline-block border-2 border-blue-200 p-2 rounded-lg bg-blue-50/30"
                        style={{
                            gridTemplateColumns: `repeat(${shape[1]}, 32px)`,
                            gridTemplateRows: `repeat(${shape[0]}, 32px)`,
                        }}
                    >
                        {Array.from({ length: shape[0] }).map((_, i) => (
                            Array.from({ length: shape[1] }).map((_, j) => {
                                // Calculate index in storage
                                // index = offset + i * stride[0] + j * stride[1]
                                const storageIdx = offset + i * stride[0] + j * stride[1];
                                const isValid = storageIdx >= 0 && storageIdx < storage.length;

                                return (
                                    <motion.div
                                        key={`${i}-${j}`}
                                        layoutId={`cell-${storageIdx}`} // Link to storage visually? (Maybe too complex for now)
                                        className={`w-8 h-8 flex items-center justify-center text-sm font-bold border rounded
                                            ${isValid
                                                ? 'bg-blue-500 text-white border-blue-600 shadow-sm'
                                                : 'bg-red-100 text-red-300 border-red-200' // Out of bounds (shouldn't happen with valid view)
                                            }
                                        `}
                                    >
                                        {isValid ? storage[storageIdx] : '?'}
                                    </motion.div>
                                );
                            })
                        ))}
                    </div>
                </div>
            </div>

            <div className="mt-6 text-xs text-text-secondary leading-relaxed bg-bg-surface p-3 rounded border border-border-subtle">
                <p>
                    <strong>公式:</strong> <code>storage_index = offset + row * stride[0] + col * stride[1]</code>
                </p>
                <ul className="list-disc pl-4 mt-2 space-y-1 text-text-tertiary">
                    <li>点击 <strong>Transpose</strong>: 并没有移动 Storage 里的任何数据，只是交换了 Shape 和 Stride。(Stride 从 [4, 1] 变成 [1, 4])。这就是为什么 <code>.t()</code> 极快。</li>
                    <li>点击 <strong>Slice</strong>: 切片同样是通过修改 Stride 和 Offset 来实现的，完全零拷贝。</li>
                </ul>
            </div>
        </div>
    );
};

export default StridedMemoryVisualizer;
