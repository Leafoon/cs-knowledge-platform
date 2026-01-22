"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const AttentionMatrixVisualizer = () => {
    const [activeHead, setActiveHead] = useState(0);
    const [hoveredToken, setHoveredToken] = useState<number | null>(null);

    const tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "tired"];

    // Mock Attention Maps (Size: 10x10)
    // Head 0: "it" attends to "animal" (Coreference resolution)
    // Head 1: Next token prediction focus
    const attentionMaps = [
        // Head 0 (Coreference / Semantic)
        [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // The
            [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // animal
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // didn't
            [0.0, 0.1, 0.1, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // cross
            [0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0], // the
            [0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0], // street
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], // because
            [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0], // it -> animal (Strong!)
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], // was -> it
            [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.8], // tired
        ],
        // Head 1 (Local context / Diagonal)
        [
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    ];

    const currentMap = attentionMaps[activeHead];

    // Helper to get color intensity
    const getBgColor = (val: number) => {
        // Using Indigo scale
        // opacity based on value
        return `rgba(99, 102, 241, ${val})`; // indigo-500 equivalent
    };

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">Self-Attention 可视化 (Heatmap)</h3>

            <div className="flex gap-4 mb-6">
                {[0, 1].map(h => (
                    <button
                        key={h}
                        onClick={() => setActiveHead(h)}
                        className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${activeHead === h
                                ? 'bg-accent-primary text-white'
                                : 'bg-bg-surface border border-border-subtle text-text-secondary hover:bg-bg-elevated'
                            }`}
                    >
                        Head {h} ({h === 0 ? "Semantic/Coreference" : "Local Context"})
                    </button>
                ))}
            </div>

            <div className="flex flex-col md:flex-row gap-8">
                {/* 1. Sentence Visualization (Interactive) */}
                <div className="flex-1 space-y-4">
                    <div className="text-sm font-bold text-text-secondary uppercase">Source Input</div>
                    <div className="flex flex-wrap gap-2">
                        {tokens.map((t, i) => (
                            <div
                                key={i}
                                onMouseEnter={() => setHoveredToken(i)}
                                onMouseLeave={() => setHoveredToken(null)}
                                className={`px-2 py-1 rounded cursor-pointer transition-all duration-200 border
                                    ${hoveredToken === i ? 'bg-accent-primary text-white border-accent-primary scale-110' : 'bg-bg-surface border-border-subtle hover:border-accent-primary'}
                                `}
                            >
                                {t}
                            </div>
                        ))}
                    </div>

                    <div className="h-40 border-t border-border-subtle pt-4 mt-4">
                        {hoveredToken !== null ? (
                            <div>
                                <div className="text-xs text-text-tertiary mb-2">
                                    Token <strong>"{tokens[hoveredToken]}"</strong> attends to:
                                </div>
                                <div className="flex flex-wrap gap-2">
                                    {tokens.map((t, j) => {
                                        const attn = currentMap[hoveredToken][j];
                                        if (attn < 0.05) return null;
                                        return (
                                            <div key={j} className="flex flex-col items-center">
                                                <div
                                                    className="px-2 py-1 rounded text-white text-xs mb-1"
                                                    style={{ backgroundColor: getBgColor(attn) }}
                                                >
                                                    {t}
                                                </div>
                                                <span className="text-[10px] text-text-tertiary">{attn.toFixed(1)}</span>
                                            </div>
                                        );
                                    })}
                                </div>
                                {activeHead === 0 && hoveredToken === 7 && (
                                    <div className="mt-2 text-xs text-green-600 bg-green-50 p-2 rounded">
                                        💡 观察: "it" (Token 7) 强烈关注 "animal" (Token 1)。这就是 Self-Attention 解决指代消解的能力。
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="text-sm text-text-tertiary italic">
                                鼠标悬停在上方单词上，查看它关注（Attention）的对象。
                            </div>
                        )}
                    </div>
                </div>

                {/* 2. Heatmap Grid */}
                <div className="hidden md:block">
                    <div className="grid grid-cols-10 gap-0.5">
                        {currentMap.map((row, i) => (
                            row.map((val, j) => (
                                <div
                                    key={`${i}-${j}`}
                                    className={`w-6 h-6 rounded-sm transition-opacity duration-200 ${hoveredToken === i ? 'ring-2 ring-accent-primary z-10' : ''}`}
                                    style={{ backgroundColor: getBgColor(val), opacity: hoveredToken === null || hoveredToken === i ? 1 : 0.3 }}
                                    title={`Attention(${tokens[i]} -> ${tokens[j]}) = ${val}`}
                                />
                            ))
                        ))}
                    </div>
                    <div className="mt-2 flex justify-between text-[10px] text-text-tertiary">
                        <span>Rows: Query (Source)</span>
                        <span>Cols: Key (Target)</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AttentionMatrixVisualizer;
