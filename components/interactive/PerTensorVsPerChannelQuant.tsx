"use client";

import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Info, Layers, GanttChartSquare, ArrowRight, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

// --- Data & Types ---

// 4x4 Matrix with an outlier channel (Row 2)
// Rows 0, 1, 3 are "small" values (e.g. weights from a stable layer)
// Row 2 has "large" outliers (common in Transformers)
const INITIAL_MATRIX = [
    [0.12, -0.45, 0.33, -0.05],
    [-0.22, 0.56, -0.18, 0.09],
    [-12.50, 8.40, -14.20, 5.50], // Outlier Channel
    [0.45, -0.32, 0.15, -0.67]
];

const INT8_MAX = 127;

type QuantMode = 'per-tensor' | 'per-channel';

export default function PerTensorVsPerChannelQuant() {
    const [mode, setMode] = useState<QuantMode>('per-tensor');
    const [hoveredCell, setHoveredCell] = useState<{ r: number, c: number } | null>(null);

    // --- Calculations ---

    const stats = useMemo(() => {
        let scales: number[] = [];
        let quantizedMatrix: number[][] = [];
        let dequantizedMatrix: number[][] = [];
        let errorMatrix: number[][] = [];

        if (mode === 'per-tensor') {
            // 1. Find absolute max of ENTIRE tensor
            let globalMax = 0;
            INITIAL_MATRIX.forEach(row => row.forEach(val => globalMax = Math.max(globalMax, Math.abs(val))));

            // 2. Calculate single scale
            const scale = globalMax / INT8_MAX;

            // 3. Apply to all
            INITIAL_MATRIX.forEach(row => {
                const qRow = row.map(val => Math.round(val / scale));
                const dqRow = qRow.map(val => val * scale);
                const errRow = row.map((val, i) => Math.abs(val - dqRow[i]));

                scales.push(scale); // Same scale for all rows visually
                quantizedMatrix.push(qRow);
                dequantizedMatrix.push(dqRow);
                errorMatrix.push(errRow);
            });
            // Fill scales array to match row count provided logical per-row view for consistency
            scales = Array(4).fill(scale);

        } else {
            // Per-Channel (Row-wise)
            INITIAL_MATRIX.forEach(row => {
                // 1. Find absolute max of THIS ROW
                const rowMax = Math.max(...row.map(Math.abs));
                const scale = rowMax / INT8_MAX; // Avoid div by 0 in real app, here data is safe

                const qRow = row.map(val => Math.round(val / scale));
                const dqRow = qRow.map(val => val * scale);
                const errRow = row.map((val, i) => Math.abs(val - dqRow[i]));

                scales.push(scale);
                quantizedMatrix.push(qRow);
                dequantizedMatrix.push(dqRow);
                errorMatrix.push(errRow);
            });
        }

        return { scales, quantizedMatrix, dequantizedMatrix, errorMatrix };
    }, [mode]);

    // --- Helper for color scaling ---
    const getCellColor = (val: number, isError = false) => {
        if (isError) {
            // Error map: White -> Red
            // Max error is likely small, scaling for visibility
            const intensity = Math.min(Math.abs(val) * 100, 100); // Amplify error visibility
            return `rgba(239, 68, 68, ${intensity / 100})`; // red-500
        } else {
            // Value map: Blue (positive) / Orange (negative)
            // Use log scale or simple linear for this demo? Linear is fine but outlier dominates.
            // Let's use a capped linear to show normal values are distinct from outlier
            const abs = Math.abs(val);
            const intensity = Math.min(abs / 1.0, 1); // Saturation at 1.0 for normal rows
            // Special handling for outlier row visibility
            if (abs > 2.0) return val > 0 ? 'rgba(59, 130, 246, 1)' : 'rgba(249, 115, 22, 1)';

            return val > 0
                ? `rgba(59, 130, 246, ${intensity})`
                : `rgba(249, 115, 22, ${intensity})`;
        }
    };

    return (
        <div className="my-8 p-6 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
                <div>
                    <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
                        <Layers className="w-5 h-5 text-indigo-500" />
                        Quantization Granularity
                    </h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                        Compare how scale granularity affects precision, especially with outliers.
                    </p>
                </div>

                <div className="flex bg-slate-100 dark:bg-slate-800 p-1 rounded-lg">
                    <button
                        onClick={() => setMode('per-tensor')}
                        className={cn(
                            "px-4 py-2 text-sm font-medium rounded-md transition-all flex items-center gap-2",
                            mode === 'per-tensor'
                                ? "bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-300 shadow-sm"
                                : "text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
                        )}
                    >
                        <Layers className="w-4 h-4" />
                        Per-Tensor (Layer-wise)
                    </button>
                    <button
                        onClick={() => setMode('per-channel')}
                        className={cn(
                            "px-4 py-2 text-sm font-medium rounded-md transition-all flex items-center gap-2",
                            mode === 'per-channel'
                                ? "bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-300 shadow-sm"
                                : "text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
                        )}
                    >
                        <GanttChartSquare className="w-4 h-4" />
                        Per-Channel (Row-wise)
                    </button>
                </div>
            </div>

            {/* Main Visual Area */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

                {/* Left: Original Matrix */}
                <div className="lg:col-span-3 flex flex-col items-center">
                    <h4 className="text-sm font-bold text-slate-500 uppercase mb-3 tracking-wider">Original Weights (FP32)</h4>
                    <div className="grid grid-rows-4 gap-2 w-full max-w-[240px]">
                        {INITIAL_MATRIX.map((row, r) => (
                            <div key={r} className={cn(
                                "grid grid-cols-4 gap-1 p-1 rounded-md border-2 transition-colors relative group",
                                r === 2 ? "border-amber-500/50 bg-amber-50/50 dark:bg-amber-900/10" : "border-transparent bg-slate-50 dark:bg-slate-800"
                            )}>
                                {/* Outlier Indicator */}
                                {r === 2 && <div className="absolute -left-6 top-1/2 -translate-y-1/2">
                                    <AlertTriangle className="w-4 h-4 text-amber-500" />
                                </div>}

                                {row.map((val, c) => (
                                    <div
                                        key={c}
                                        onMouseEnter={() => setHoveredCell({ r, c })}
                                        onMouseLeave={() => setHoveredCell(null)}
                                        className="h-10 rounded text-[10px] font-mono flex items-center justify-center cursor-crosshair transition-transform hover:scale-110 z-10"
                                        style={{
                                            backgroundColor: getCellColor(val),
                                            color: Math.abs(val) > 0.5 ? 'white' : 'var(--foreground)'
                                        }}
                                    >
                                        {val.toFixed(2)}
                                    </div>
                                ))}
                            </div>
                        ))}
                    </div>
                </div>

                {/* Center: Process Pipeline */}
                <div className="lg:col-span-5 flex flex-col justify-center relative">
                    {/* Scale Visuals */}
                    <div className="bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-200 dark:border-slate-800 p-4 mb-4">
                        <div className="flex justify-between items-center mb-4">
                            <h4 className="text-sm font-bold text-slate-600 dark:text-slate-300 flex items-center gap-2">
                                <Info className="w-4 h-4" />
                                Quantization Scale (S)
                            </h4>
                            <div className="text-xs text-slate-400 font-mono">
                                Formula: row_max / 127
                            </div>
                        </div>

                        <div className="space-y-3">
                            {stats.scales.map((scale, i) => (
                                <div key={i} className="flex items-center gap-3">
                                    <span className="text-xs font-mono text-slate-400 w-12 text-right">Row {i}</span>
                                    <div className="flex-1 h-6 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden relative group">
                                        <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${Math.min((scale / 0.15) * 100, 100)}%` }} // Normalize for visualization
                                            transition={{ type: 'spring', stiffness: 100 }}
                                            className={cn(
                                                "h-full rounded-full transition-colors",
                                                i === 2 ? "bg-amber-500" : "bg-indigo-500"
                                            )}
                                        />
                                        <div className="absolute inset-0 flex items-center justify-end px-2 text-[10px] font-mono text-white opacity-0 group-hover:opacity-100 transition-opacity">
                                            S = {scale.toFixed(4)}
                                        </div>
                                    </div>
                                    <div className="text-xs font-mono text-slate-600 dark:text-slate-300 w-16 text-right">
                                        {scale.toFixed(3)}
                                    </div>
                                </div>
                            ))}
                        </div>

                        <div className="mt-4 pt-3 border-t border-slate-200 dark:border-slate-700 text-xs text-slate-500">
                            {mode === 'per-tensor' ? (
                                <p className="text-amber-600 dark:text-amber-400 font-medium">
                                    ⚠️ Per-Tensor uses ONE max scale ({stats.scales[0].toFixed(3)}) determined by the outlier in Row 2.
                                    This forces "small" rows (0, 1, 3) to use a huge scale, losing precision.
                                </p>
                            ) : (
                                <p className="text-emerald-600 dark:text-emerald-400 font-medium">
                                    ✅ Per-Channel calculates INDEPENDENT scales.
                                    Outlier row uses a large scale, but normal rows keep high precision (small scales).
                                </p>
                            )}
                        </div>
                    </div>
                </div>

                {/* Right: Error Analysis */}
                <div className="lg:col-span-4 flex flex-col items-center">
                    <h4 className="text-sm font-bold text-slate-500 uppercase mb-3 tracking-wider">Quantization Error</h4>
                    <div className="grid grid-rows-4 gap-2 w-full max-w-[240px]">
                        {stats.errorMatrix.map((row, r) => (
                            <div key={r} className="grid grid-cols-4 gap-1 p-1">
                                {row.map((val, c) => (
                                    <div
                                        key={c}
                                        className={cn(
                                            "h-10 rounded text-[10px] font-mono flex items-center justify-center transition-all",
                                            hoveredCell?.r === r && hoveredCell?.c === c ? "ring-2 ring-indigo-500 z-20 scale-110 bg-white dark:bg-slate-700" : ""
                                        )}
                                        style={{
                                            backgroundColor: getCellColor(val, true),
                                            // Make text readable on dark red
                                            color: val > 0.05 ? 'white' : 'var(--foreground)'
                                        }}
                                    >
                                        {val.toFixed(3)}
                                    </div>
                                ))}
                            </div>
                        ))}
                    </div>

                    <div className="mt-4 p-3 bg-slate-50 dark:bg-slate-800 rounded-lg text-xs space-y-2 w-full">
                        <div className="flex justify-between">
                            <span className="text-slate-500">Normal Rows Avg Err:</span>
                            <span className={cn(
                                "font-mono font-bold",
                                mode === 'per-tensor' ? "text-rose-500" : "text-emerald-500"
                            )}>
                                {(() => {
                                    // Calc avg error for rows 0, 1, 3
                                    let sum = 0, count = 0;
                                    [0, 1, 3].forEach(r => stats.errorMatrix[r].forEach(v => { sum += v; count++; }));
                                    return (sum / count).toFixed(4);
                                })()}
                            </span>
                        </div>
                        <div className="flex justify-between border-t border-slate-200 dark:border-slate-700 pt-2">
                            <span className="text-slate-500">Outlier Row Avg Err:</span>
                            <span className="font-mono font-bold text-slate-700 dark:text-slate-300">
                                {(() => {
                                    let sum = 0, count = 0;
                                    stats.errorMatrix[2].forEach(v => { sum += v; count++; });
                                    return (sum / count).toFixed(4);
                                })()}
                            </span>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
}
