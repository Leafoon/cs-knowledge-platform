"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const TorchScriptVisualizer = () => {
    const [mode, setMode] = useState<'trace' | 'script'>('trace');
    const [inputVal, setInputVal] = useState(1); // 1 (>0) or -1 (<0)

    // The Code Snippet
    const codeSnippet = `def forward(self, x):
    if x > 0:
        return x * 2
    else:
        return x + 1`;

    return (
        <div className="w-full max-w-3xl mx-auto p-6 bg-bg-elevated/50 backdrop-blur rounded-xl border border-border-subtle shadow-sm my-6">
            <h3 className="text-xl font-bold mb-4 text-text-primary">TorchScript: Tracing vs Scripting</h3>

            <div className="flex justify-center gap-6 mb-8">
                <button
                    onClick={() => setMode('trace')}
                    className={`px-4 py-2 rounded-lg font-bold transition-all ${mode === 'trace' ? 'bg-accent-primary text-white shadow-lg' : 'bg-bg-surface border border-border-subtle hover:bg-bg-elevated'}`}
                >
                    jit.trace (Example Input)
                </button>
                <button
                    onClick={() => setMode('script')}
                    className={`px-4 py-2 rounded-lg font-bold transition-all ${mode === 'script' ? 'bg-accent-primary text-white shadow-lg' : 'bg-bg-surface border border-border-subtle hover:bg-bg-elevated'}`}
                >
                    jit.script (Source Code)
                </button>
            </div>

            <div className="flex flex-col md:flex-row gap-8">
                {/* Left: Source Code with Execution Path */}
                <div className="flex-1 bg-slate-900 rounded-xl p-4 font-mono text-sm leading-relaxed text-slate-300 relative overflow-hidden">
                    <div className="mb-2 text-xs text-slate-500 uppercase font-bold">Python Source</div>
                    {codeSnippet.split('\n').map((line, i) => {
                        let isHighlighted = false;
                        if (mode === 'script') {
                            isHighlighted = true; // Script sees everything
                        } else {
                            // Tracing: only highlight executed path
                            if (i === 0) isHighlighted = true;
                            if (i === 1) isHighlighted = true; // if x > 0 check
                            if (inputVal > 0) {
                                if (i === 2) isHighlighted = true; // return x*2
                            } else {
                                if (i === 3) isHighlighted = true; // else
                                if (i === 4) isHighlighted = true; // return x+1
                            }
                        }

                        return (
                            <div key={i} className={`px-2 rounded transition-colors duration-300 ${isHighlighted ? 'bg-blue-500/20 text-blue-100' : 'opacity-30'}`}>
                                {line}
                            </div>
                        )
                    })}

                    {mode === 'trace' && (
                        <div className="absolute bottom-4 right-4 bg-slate-800 p-2 rounded border border-slate-700">
                            <div className="text-xs text-slate-400 mb-1">Current Input x:</div>
                            <div className="flex gap-2">
                                <button onClick={() => setInputVal(1)} className={`w-8 h-6 flex items-center justify-center rounded text-xs ${inputVal === 1 ? 'bg-blue-500 text-white' : 'bg-slate-700'}`}>1</button>
                                <button onClick={() => setInputVal(-1)} className={`w-8 h-6 flex items-center justify-center rounded text-xs ${inputVal === -1 ? 'bg-blue-500 text-white' : 'bg-slate-700'}`}>-1</button>
                            </div>
                        </div>
                    )}
                </div>

                {/* Right: Generated Graph */}
                <div className="flex-1 bg-white dark:bg-slate-800 border border-border-subtle rounded-xl p-4 flex flex-col relative min-h-[160px]">
                    <div className="mb-2 text-xs text-text-tertiary uppercase font-bold">Resulting Graph (IR)</div>

                    <div className="flex-1 flex flex-col items-center justify-center gap-2">
                        {mode === 'trace' ? (
                            <>
                                <div className="p-2 border rounded bg-gray-50 dark:bg-slate-900 border-gray-200 text-xs">x</div>
                                <div className="text-gray-400">↓</div>
                                <div className="p-2 border rounded bg-blue-50 dark:bg-blue-900/30 border-blue-200 text-xs font-bold text-blue-700 dark:text-blue-300">
                                    {inputVal > 0 ? "mul (x, 2)" : "add (x, 1)"}
                                </div>
                                <div className="text-gray-400">↓</div>
                                <div className="p-2 border rounded bg-gray-50 dark:bg-slate-900 border-gray-200 text-xs">return</div>

                                <div className="mt-4 text-[10px] text-red-500 bg-red-50 p-2 rounded">
                                    ⚠️ 警告: 图中丢失了 If-Else 分支！
                                    <br />如果将来输入 x = {inputVal > 0 ? "-1" : "1"}，模型依然会执行 <strong>{inputVal > 0 ? "乘法" : "加法"}</strong>，导致逻辑错误。
                                </div>
                            </>
                        ) : (
                            <>
                                <div className="p-2 border rounded bg-gray-50 dark:bg-slate-900 border-gray-200 text-xs">x</div>
                                <div className="text-gray-400">↓</div>
                                <div className="p-2 border border-dashed border-purple-300 bg-purple-50 dark:bg-purple-900/20 rounded w-full text-center">
                                    <div className="text-[10px] text-purple-600 font-bold mb-1">prim::If (x > 0)</div>
                                    <div className="flex gap-2 justify-center">
                                        <div className="p-1 bg-white border rounded text-[9px]">Then: x * 2</div>
                                        <div className="p-1 bg-white border rounded text-[9px]">Else: x + 1</div>
                                    </div>
                                </div>
                                <div className="text-gray-400">↓</div>
                                <div className="p-2 border rounded bg-gray-50 dark:bg-slate-900 border-gray-200 text-xs">return</div>

                                <div className="mt-4 text-[10px] text-green-600 bg-green-50 p-2 rounded">
                                    ✅ 完美: 完整的控制流被保留了下来。
                                </div>
                            </>
                        )}
                    </div>
                </div>
            </div>

            <div className="mt-4 text-xs text-text-tertiary">
                * <strong>Tracing</strong>: "我看你跑了一遍，我就照着刚才的路记下来。" (只认路，不认图)
                <br />
                * <strong>Scripting</strong>: "我阅读了你的源代码，理解了所有逻辑。" (理解全图)
            </div>
        </div>
    );
};

export default TorchScriptVisualizer;
