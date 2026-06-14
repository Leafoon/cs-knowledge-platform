"use client";

import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, ScatterChart, Scatter, ZAxis, Cell } from 'recharts';
import { Info, BarChart3, TrendingDown, Target, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';

// --- Constants & Data ---

// Standard INT4 (Linear) - usually symmetric around 0
// -7 to +7, plus a clear 0? Or -8 to 7. 
// Standard INT4 usually maps [-abs_max, abs_max] linearly to [-7, 7] or integers.
// Normalized to N(0,1) scale (roughly [-2, 2] to [-3, 3])
// Let's assume absolute max of distribution is ~2.5 for visualization
const SCALE = 3.0;
const INT4_LEVELS = Array.from({ length: 16 }, (_, i) => {
    // Linear mapping from -1 to 1 basically, stretched
    const step = (2 * SCALE) / 15;
    return -SCALE + i * step;
});

// NF4 (NormalFloat4) - Theoretically derived quantiles for N(0,1)
// These are fixed values designed to map equal probability mass.
// Raw NF4 values (normalized to [-1, 1]) from QLoRA paper roughly:
const NF4_RAW = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
];
// Scale NF4 to match the visualization scale (e.g. sigma=1, range ~3)
// In practice, double quantization scales these, but for visual comparison against linear:
// We stretch them so they cover the same effective dynamic range as INT4 for fair comparison of "distribution shape".
// Or better: keep them as is (optimized for N(0,1)) and scale INT4 to cover standard deviation range like [-2.5, 2.5].
const DECODED_NF4 = NF4_RAW.map(x => x * 2.5); // Assume roughly -2.5 to 2.5 coverage
const DECODED_INT4 = Array.from({ length: 16 }, (_, i) => -2.5 + i * (5.0 / 15));

type DataType = 'INT4' | 'NF4';

export default function NF4vsINT4Comparison() {
    const [activeTab, setActiveTab] = useState<DataType>('NF4');
    const [sampleCount, setSampleCount] = useState(0); // Trigger re-gen

    // --- Simulation ---
    // Generate 500 samples from N(0,1)
    const simulationData = useMemo(() => {
        const samples = [];
        let totalErrInt4 = 0;
        let totalErrNf4 = 0;

        for (let i = 0; i < 200; i++) {
            // Box-Muller transform for N(0,1)
            const u = 1 - Math.random();
            const v = Math.random();
            const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);

            // Allow some outliers? N(0,1) naturally has them.
            const val = z;

            // Quantize INT4
            let closestInt4 = DECODED_INT4[0];
            let minDiffInt4 = Math.abs(val - closestInt4);
            for (let level of DECODED_INT4) {
                const diff = Math.abs(val - level);
                if (diff < minDiffInt4) {
                    minDiffInt4 = diff;
                    closestInt4 = level;
                }
            }

            // Quantize NF4
            let closestNf4 = DECODED_NF4[0];
            let minDiffNf4 = Math.abs(val - closestNf4);
            for (let level of DECODED_NF4) {
                const diff = Math.abs(val - level);
                if (diff < minDiffNf4) {
                    minDiffNf4 = diff;
                    closestNf4 = level;
                }
            }

            totalErrInt4 += minDiffInt4 * minDiffInt4; // MSE
            totalErrNf4 += minDiffNf4 * minDiffNf4;

            samples.push({
                x: val,
                y: Math.random() * 0.5, // Spread vertically for visual
                val,
                qVal: activeTab === 'INT4' ? closestInt4 : closestNf4,
                error: activeTab === 'INT4' ? minDiffInt4 : minDiffNf4
            });
        }

        return {
            samples,
            mseInt4: totalErrInt4 / 200,
            mseNf4: totalErrNf4 / 200
        };
    }, [sampleCount, activeTab]);

    // Levels to display
    const currentLevels = activeTab === 'INT4' ? DECODED_INT4 : DECODED_NF4;

    return (
        <div className="my-8 p-6 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
                <div>
                    <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
                        <Target className="w-5 h-5 text-indigo-500" />
                        NF4 vs INT4 Quantization
                    </h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                        Why NormalFloat4 is optimal for pre-trained weights (Gaussian distribution).
                    </p>
                </div>

                <div className="flex bg-slate-100 dark:bg-slate-800 p-1 rounded-lg">
                    <button
                        onClick={() => setActiveTab('INT4')}
                        className={cn(
                            "px-4 py-2 text-sm font-medium rounded-md transition-all flex items-center gap-2",
                            activeTab === 'INT4'
                                ? "bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-300 shadow-sm"
                                : "text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
                        )}
                    >
                        INT4 (Linear)
                    </button>
                    <button
                        onClick={() => setActiveTab('NF4')}
                        className={cn(
                            "px-4 py-2 text-sm font-medium rounded-md transition-all flex items-center gap-2",
                            activeTab === 'NF4'
                                ? "bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-300 shadow-sm"
                                : "text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
                        )}
                    >
                        <BarChart3 className="w-4 h-4" />
                        NF4 (NormalFloat)
                    </button>
                </div>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <div className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-100 dark:border-slate-800">
                    <div className="flex items-center gap-2 mb-2 text-slate-500 dark:text-slate-400">
                        <TrendingDown className="w-4 h-4 text-purple-500" />
                        <span className="text-xs font-semibold uppercase tracking-wider">Current MSE</span>
                    </div>
                    <div className="text-2xl font-black text-slate-800 dark:text-slate-100">
                        {activeTab === 'INT4' ? simulationData.mseInt4.toFixed(4) : simulationData.mseNf4.toFixed(4)}
                    </div>
                    <div className="text-xs text-slate-400 mt-1">Mean Squared Error</div>
                </div>

                <div className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-100 dark:border-slate-800">
                    <div className="flex items-center gap-2 mb-2 text-slate-500 dark:text-slate-400">
                        <Info className="w-4 h-4 text-emerald-500" />
                        <span className="text-xs font-semibold uppercase tracking-wider">Advantage</span>
                    </div>
                    <div className="text-sm font-medium text-slate-700 dark:text-slate-300 mt-1">
                        NF4 Reduces Error by
                    </div>
                    <div className="text-xl font-bold text-emerald-600 dark:text-emerald-400">
                        {((1 - simulationData.mseNf4 / simulationData.mseInt4) * 100).toFixed(1)}%
                    </div>
                </div>

                <button
                    onClick={() => setSampleCount(c => c + 1)}
                    className="col-span-2 flex items-center justify-center gap-2 bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 rounded-xl border border-indigo-200 dark:border-indigo-800 hover:bg-indigo-100 dark:hover:bg-indigo-900/30 transition-colors"
                >
                    <RefreshCw className="w-5 h-5" />
                    <span>Regenerate Weights</span>
                </button>
            </div>

            {/* Visualization */}
            <div className="h-[300px] w-full relative bg-slate-50 dark:bg-slate-900/50 rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden">
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <XAxis
                            type="number"
                            dataKey="x"
                            name="Weight Value"
                            domain={[-3.5, 3.5]}
                            tick={{ fontSize: 12, fill: '#94a3b8' }}
                            stroke="#cbd5e1"
                            label={{ value: "Weight Distribution (N(0,1))", position: "bottom", offset: 0, fill: '#64748b' }}
                        />
                        <YAxis type="number" dataKey="y" hide domain={[0, 1]} />
                        <ZAxis type="number" dataKey="error" range={[20, 100]} name="Quant Error" />
                        <Tooltip
                            cursor={{ strokeDasharray: '3 3' }}
                            content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                    const data = payload[0].payload;
                                    return (
                                        <div className="bg-white/95 dark:bg-slate-800/95 p-3 border border-slate-200 dark:border-slate-700 rounded shadow-lg text-xs backdrop-blur-sm">
                                            <p className="font-bold mb-1">Weight: {data.val.toFixed(4)}</p>
                                            <p>Quantized: {data.qVal.toFixed(4)}</p>
                                            <p className="text-rose-500 mt-1">Error: {Math.abs(data.val - data.qVal).toFixed(4)}</p>
                                        </div>
                                    );
                                }
                                return null;
                            }}
                        />

                        {/* Quantization Levels Lines */}
                        {currentLevels.map((level, i) => (
                            <ReferenceLine
                                key={i}
                                x={level}
                                stroke={activeTab === 'INT4' ? "#f59e0b" : "#3b82f6"}
                                strokeDasharray="3 3"
                                label={{ value: "|", position: 'insideTop', fill: activeTab === 'INT4' ? "#d97706" : "#2563eb" }}
                            />
                        ))}

                        {/* Distribution Curve Ref (Ideal) - Simulated by scatter density or just visual cue */}
                        {/* We use Scatter for the actual points */}
                        <Scatter
                            name="Weights"
                            data={simulationData.samples}
                            fill={activeTab === 'INT4' ? "#cbd5e1" : "#cbd5e1"}
                            shape="circle"
                            opacity={0.6}
                        >
                            {simulationData.samples.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={activeTab === 'INT4' ? '#94a3b8' : '#94a3b8'} />
                            ))}
                        </Scatter>

                        {/* Overlay Quantized Points (optional, maybe clearer without) */}

                    </ScatterChart>
                </ResponsiveContainer>

                {/* Legend/Info Overlay */}
                <div className="absolute top-4 right-4 bg-white/80 dark:bg-slate-900/80 p-3 rounded-lg border border-slate-200 dark:border-slate-700 backdrop-blur-sm text-xs max-w-[200px]">
                    <div className="flex items-center gap-2 mb-2">
                        <div className={cn("w-3 h-3 rounded-full", activeTab === 'INT4' ? "bg-amber-500" : "bg-blue-500")}></div>
                        <span className="font-bold">{activeTab} Levels ({16})</span>
                    </div>
                    <p className="text-slate-500 leading-relaxed">
                        {activeTab === 'INT4'
                            ? "INT4 uses equidistant spacing. Note the 'wasted' levels at the empty tails and lack of precision in the dense center."
                            : "NF4 clusters levels near 0 (the center), where most weights exist. This captures the bell curve shape much better."}
                    </p>
                </div>
            </div>
        </div>
    );
}
