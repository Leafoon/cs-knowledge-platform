"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const TensorBroadcastingVisualizer = () => {
    // Scenario: Matrix A (3x1) + Vector B (1x3) -> Result (3x3)
    // A: 
    // [1]
    // [2]
    // [3]

    // B: [10, 20, 30]

    // Result:
    // [11, 21, 31]
    // [12, 22, 32]
    // [13, 23, 33]

    const [hoveredCell, setHoveredCell] = useState<{ r: number, c: number } | null>(null);

    const matrixA = [[1], [2], [3]];
    const vectorB = [[10, 20, 30]];

    const result = [
        [11, 21, 31],
        [12, 22, 32],
        [13, 23, 33]
    ];

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">Tensor Broadcasting 机制</h3>

            <p className="text-sm text-text-secondary mb-8">
                演示: <code>A (3x1)</code> + <code>B (1x3)</code> = <code>Result (3x3)</code>
                <br />
                缺失的维度会被自动复制 (Broadcast) 以匹配形状。
            </p>

            <div className="flex items-center justify-center gap-4 md:gap-8 flex-wrap">
                {/* Matrix A */}
                <div className="relative p-2 bg-blue-50/50 border border-blue-200 rounded-lg">
                    <div className="text-xs text-blue-500 font-bold mb-1 text-center">A (3x1)</div>
                    <div className="grid grid-rows-3 gap-1">
                        {matrixA.map((row, r) => (
                            <div
                                key={r}
                                className={`w-8 h-8 flex items-center justify-center border rounded bg-white font-mono text-sm transition-colors
                                    ${hoveredCell && hoveredCell.r === r ? 'bg-accent-primary text-white scale-110 shadow-lg z-10' : 'text-text-primary'}
                                `}
                            >
                                {row[0]}
                            </div>
                        ))}
                    </div>
                    {/* Ghost Columns for Broadcasting Visualization */}
                    <div className="absolute top-2 left-full ml-1 grid grid-rows-3 gap-1 opacity-30">
                        {matrixA.map((row, r) => (
                            <div key={`ghost-1-${r}`} className="w-8 h-8 flex items-center justify-center border border-dashed border-blue-300 rounded bg-transparent font-mono text-sm text-blue-300">
                                {row[0]}
                            </div>
                        ))}
                    </div>
                    <div className="absolute top-2 left-full ml-10 grid grid-rows-3 gap-1 opacity-30">
                        {matrixA.map((row, r) => (
                            <div key={`ghost-2-${r}`} className="w-8 h-8 flex items-center justify-center border border-dashed border-blue-300 rounded bg-transparent font-mono text-sm text-blue-300">
                                {row[0]}
                            </div>
                        ))}
                    </div>
                </div>

                <div className="text-2xl font-bold text-text-tertiary">+</div>

                {/* Vector B */}
                <div className="relative p-2 bg-green-50/50 border border-green-200 rounded-lg">
                    <div className="text-xs text-green-600 font-bold mb-1 text-center">B (1x3)</div>
                    <div className="grid grid-cols-3 gap-1">
                        {vectorB[0].map((val, c) => (
                            <div
                                key={c}
                                className={`w-8 h-8 flex items-center justify-center border rounded bg-white font-mono text-sm transition-colors
                                    ${hoveredCell && hoveredCell.c === c ? 'bg-accent-primary text-white scale-110 shadow-lg z-10' : 'text-text-primary'}
                                `}
                            >
                                {val}
                            </div>
                        ))}
                    </div>
                    {/* Ghost Rows */}
                    <div className="absolute top-full mt-1 left-2 grid grid-cols-3 gap-1 opacity-30">
                        {vectorB[0].map((val, c) => (
                            <div key={`ghost-r1-${c}`} className="w-8 h-8 flex items-center justify-center border border-dashed border-green-300 rounded bg-transparent font-mono text-sm text-green-300">
                                {val}
                            </div>
                        ))}
                    </div>
                    <div className="absolute top-full mt-10 left-2 grid grid-cols-3 gap-1 opacity-30">
                        {vectorB[0].map((val, c) => (
                            <div key={`ghost-r2-${c}`} className="w-8 h-8 flex items-center justify-center border border-dashed border-green-300 rounded bg-transparent font-mono text-sm text-green-300">
                                {val}
                            </div>
                        ))}
                    </div>
                </div>

                <div className="text-2xl font-bold text-text-tertiary">=</div>

                {/* Result */}
                <div className="p-2 bg-purple-50/50 border border-purple-200 rounded-lg">
                    <div className="text-xs text-purple-600 font-bold mb-1 text-center">Result (3x3)</div>
                    <div className="grid grid-rows-3 gap-1">
                        {result.map((row, r) => (
                            <div key={r} className="flex gap-1">
                                {row.map((val, c) => (
                                    <div
                                        key={c}
                                        onMouseEnter={() => setHoveredCell({ r, c })}
                                        onMouseLeave={() => setHoveredCell(null)}
                                        className={`w-10 h-10 flex items-center justify-center border rounded font-mono text-sm cursor-crosshair transition-all duration-200
                                            ${hoveredCell && hoveredCell.r === r && hoveredCell.c === c
                                                ? 'bg-purple-600 text-white scale-110 shadow-lg font-bold'
                                                : 'bg-white text-text-primary border-purple-200 hover:border-purple-400'}
                                        `}
                                    >
                                        {val}
                                    </div>
                                ))}
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="mt-6 h-8 text-center">
                <AnimatePresence mode="wait">
                    {hoveredCell ? (
                        <motion.div
                            key="math"
                            initial={{ opacity: 0, y: 5 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="text-sm font-mono text-purple-700 bg-purple-50 inline-block px-3 py-1 rounded-lg border border-purple-200"
                        >
                            {result[hoveredCell.r][hoveredCell.c]}
                            <span className="text-blue-500 mx-2">← {matrixA[hoveredCell.r][0]}</span>
                            <span className="text-text-tertiary">+</span>
                            <span className="text-green-600 mx-2">{vectorB[0][hoveredCell.c]} →</span>
                        </motion.div>
                    ) : (
                        <span className="text-xs text-text-tertiary">将鼠标悬停在 Result 格子上查看计算来源</span>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
};

export default TensorBroadcastingVisualizer;
