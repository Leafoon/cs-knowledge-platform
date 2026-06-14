"use client";

import React, { useState } from 'react';
import {
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    Radar,
    ResponsiveContainer,
    Tooltip
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, Info, Cpu, Database, Zap, Brain, Scale, Layers, Monitor, Server } from 'lucide-react';
import { cn } from '@/lib/utils';

// --- Data ---

const METRICS = [
    { subject: 'Memory Efficiency', fullMark: 10, icon: Database },
    { subject: 'Inference Speed', fullMark: 10, icon: Zap },
    { subject: 'Accuracy', fullMark: 10, icon: Scale },
    { subject: 'Training Efficiency', fullMark: 10, icon: Brain },
    { subject: 'Hardware Support', fullMark: 10, icon: Cpu },
];

const METHODS = [
    {
        id: 'fp16',
        name: 'FP16',
        label: 'FP16 Baseline',
        color: '#64748b', // Slate-500
        fill: '#94a3b8',
        data: {
            'Memory Efficiency': 2,
            'Inference Speed': 4,
            'Accuracy': 10,
            'Training Efficiency': 4,
            'Hardware Support': 10
        },
        tags: ["High Precision", "Universal"],
        description: "The gold standard for training. Requires significant VRAM but guarantees max accuracy.",
        badge: "Baseline"
    },
    {
        id: 'int8',
        name: 'INT8',
        label: 'INT8 Quantization',
        color: '#3b82f6', // Blue-500
        fill: '#60a5fa',
        data: {
            'Memory Efficiency': 6,
            'Inference Speed': 6,
            'Accuracy': 9.8,
            'Training Efficiency': 3,
            'Hardware Support': 8
        },
        tags: ["Production Ready", "Balanced"],
        description: "Standard for deployment. Halves memory with negligible loss.",
        badge: "Production"
    },
    {
        id: 'gptq',
        name: 'GPTQ',
        label: 'GPTQ / AWQ (4-bit)',
        color: '#10b981', // Emerald-500
        fill: '#34d399',
        data: {
            'Memory Efficiency': 9,
            'Inference Speed': 10,
            'Accuracy': 9.0,
            'Training Efficiency': 2,
            'Hardware Support': 7
        },
        tags: ["Edge Ready", "Ultra Fast"],
        description: "Extreme compression (4x) with specialized kernels for blazing fast inference.",
        badge: "Fastest"
    },
    {
        id: 'nf4',
        name: 'NF4',
        label: 'NF4 (QLoRA)',
        color: '#f59e0b', // Amber-500
        fill: '#fbbf24',
        data: {
            'Memory Efficiency': 9,
            'Inference Speed': 5,
            'Accuracy': 9.5,
            'Training Efficiency': 10,
            'Hardware Support': 6
        },
        tags: ["Fine-tuning", "Optimal"],
        description: "Information-theoretically optimal for training. The key to QLoRA.",
        badge: "Training"
    }
];

// Reformat for Recharts
const CHART_DATA = METRICS.map(metric => {
    const point: any = { subject: metric.subject, fullMark: 10 };
    METHODS.forEach(method => {
        point[method.name] = method.data[metric.subject as keyof typeof method.data];
    });
    return point;
});

export default function QuantizationMethodsComprehensiveComparison() {
    const [selectedMethods, setSelectedMethods] = useState<string[]>(['fp16', 'int8', 'gptq', 'nf4']);
    const [hoveredMethod, setHoveredMethod] = useState<string | null>(null);

    const toggleMethod = (id: string) => {
        if (selectedMethods.includes(id) && selectedMethods.length === 1) return; // Prevent empty
        setSelectedMethods(prev =>
            prev.includes(id)
                ? prev.filter(m => m !== id)
                : [...prev, id]
        );
    };

    return (
        <div className="my-10 bg-slate-50 dark:bg-slate-900/50 rounded-3xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden">

            {/* 1. Header Area with Glassmorphism */}
            <div className="relative p-8 pb-32 bg-gradient-to-br from-indigo-600 via-violet-600 to-purple-700 text-white overflow-hidden">
                <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-100 contrast-150"></div>
                <div className="absolute top-0 right-0 p-10 opacity-10">
                    <Scale size={200} />
                </div>

                <div className="relative z-10 max-w-2xl">
                    <h3 className="text-3xl font-bold flex items-center gap-3 mb-2">
                        <Scale className="w-8 h-8 text-indigo-200" />
                        Quantization Arena
                    </h3>
                    <p className="text-indigo-100 text-lg opacity-90">
                        Comparing the trade-offs between precision, speed, and memory across modern quantization techniques.
                    </p>
                </div>
            </div>

            {/* 2. Main Content Card - Overlapping Header */}
            <div className="relative z-20 mx-6 -mt-24 bg-white dark:bg-slate-900 rounded-2xl shadow-xl border border-slate-100 dark:border-slate-800 flex flex-col lg:flex-row overflow-hidden">

                {/* Left: Interactive Chart */}
                <div className="lg:w-3/5 p-6 border-b lg:border-b-0 lg:border-r border-slate-100 dark:border-slate-800 flex flex-col">

                    {/* Filter Chips */}
                    <div className="flex flex-wrap gap-2 mb-6 justify-center">
                        {METHODS.map(method => {
                            const active = selectedMethods.includes(method.id);
                            return (
                                <button
                                    key={method.id}
                                    onClick={() => toggleMethod(method.id)}
                                    onMouseEnter={() => setHoveredMethod(method.id)}
                                    onMouseLeave={() => setHoveredMethod(null)}
                                    className={cn(
                                        "px-3 py-1.5 rounded-full text-xs font-bold transition-all border-2 flex items-center gap-2",
                                        active
                                            ? "bg-white dark:bg-slate-800 scale-105 shadow-sm"
                                            : "bg-slate-50 dark:bg-slate-900 text-slate-400 border-transparent grayscale opactiy-50",
                                        active && method.id === 'fp16' && "border-slate-400 text-slate-600",
                                        active && method.id === 'int8' && "border-blue-500 text-blue-600",
                                        active && method.id === 'gptq' && "border-emerald-500 text-emerald-600",
                                        active && method.id === 'nf4' && "border-amber-500 text-amber-600",
                                    )}
                                >
                                    <span className={cn("w-2 h-2 rounded-full", !active && "bg-slate-300")} style={{ backgroundColor: active ? method.color : undefined }} />
                                    {method.name}
                                </button>
                            )
                        })}
                    </div>

                    <div className="flex-1 min-h-[400px] relative">
                        <ResponsiveContainer width="100%" height="100%">
                            <RadarChart cx="50%" cy="50%" outerRadius="70%" data={CHART_DATA}>
                                <PolarGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                <PolarAngleAxis
                                    dataKey="subject"
                                    tick={({ payload, x, y, textAnchor }) => (
                                        <g className="recharts-layer recharts-polar-angle-axis-tick">
                                            <text
                                                x={x}
                                                y={y}
                                                dy={0}
                                                textAnchor={textAnchor}
                                                fill="#64748b"
                                                fontSize={11}
                                                fontWeight={600}
                                            >
                                                {payload.value}
                                            </text>
                                        </g>
                                    )}
                                />
                                <PolarRadiusAxis angle={30} domain={[0, 10]} tick={false} axisLine={false} />

                                {METHODS.map(method => (
                                    selectedMethods.includes(method.id) && (
                                        <Radar
                                            key={method.id}
                                            name={method.name}
                                            dataKey={method.name}
                                            stroke={method.color}
                                            strokeWidth={hoveredMethod === method.id ? 3 : 2}
                                            fill={method.fill}
                                            fillOpacity={
                                                hoveredMethod
                                                    ? (hoveredMethod === method.id ? 0.5 : 0.05)
                                                    : 0.2
                                            }
                                            onMouseEnter={() => setHoveredMethod(method.id)}
                                            onMouseLeave={() => setHoveredMethod(null)}
                                            animationDuration={300}
                                        />
                                    )
                                ))}
                                <Tooltip
                                    content={({ active, payload, label }) => {
                                        if (active && payload && payload.length) {
                                            return (
                                                <div className="bg-white/95 dark:bg-slate-800/95 p-4 border border-slate-200 dark:border-slate-700 rounded-xl shadow-xl backdrop-blur-md">
                                                    <p className="font-bold mb-3 text-slate-800 dark:text-slate-100 border-b pb-2">{label}</p>
                                                    {[...payload]
                                                        .sort((a: any, b: any) => b.value - a.value)
                                                        .map((entry: any, i: number) => (
                                                            <div key={i} className="flex items-center justify-between gap-6 mb-1.5 text-xs">
                                                                <div className="flex items-center gap-2">
                                                                    <div
                                                                        className="w-2 h-2 rounded-full shadow-sm"
                                                                        style={{ backgroundColor: entry.color }}
                                                                    />
                                                                    <span className="font-medium text-slate-600 dark:text-slate-300">{entry.name}</span>
                                                                </div>
                                                                <span className="font-mono font-bold text-slate-800 dark:text-white">{entry.value}/10</span>
                                                            </div>
                                                        ))}
                                                </div>
                                            );
                                        }
                                        return null;
                                    }}
                                />
                            </RadarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Right: Info Cards */}
                <div className="lg:w-2/5 flex flex-col bg-slate-50/50 dark:bg-slate-900/50">
                    <div className="p-6 space-y-4 max-h-[500px] overflow-y-auto custom-scrollbar">
                        <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Method Details</h4>

                        {METHODS.filter(m => selectedMethods.includes(m.id)).map(method => {
                            const isHovered = hoveredMethod === method.id;

                            return (
                                <motion.div
                                    key={method.id}
                                    layoutId={`card-${method.id}`}
                                    onMouseEnter={() => setHoveredMethod(method.id)}
                                    onMouseLeave={() => setHoveredMethod(null)}
                                    className={cn(
                                        "p-4 rounded-xl border transition-all duration-200 relative overflow-hidden group cursor-default",
                                        isHovered
                                            ? "bg-white dark:bg-slate-800 shadow-lg scale-[1.02] z-10 border-transparent"
                                            : "bg-white dark:bg-slate-800 border-slate-100 dark:border-slate-700 shadow-sm opacity-90"
                                    )}
                                    style={{
                                        borderColor: isHovered ? method.color : ''
                                    }}
                                >
                                    {/* Badge */}
                                    <div
                                        className="absolute top-0 right-0 px-3 py-1 rounded-bl-xl text-[10px] font-bold text-white shadow-sm"
                                        style={{ backgroundColor: method.color }}
                                    >
                                        {method.badge}
                                    </div>

                                    <div className="flex items-center gap-3 mb-2">
                                        <div
                                            className="w-10 h-10 rounded-lg flex items-center justify-center text-white shadow-md"
                                            style={{ backgroundColor: method.color }}
                                        >
                                            {method.id === 'fp16' && <Scale className="w-5 h-5" />}
                                            {method.id === 'int8' && <Server className="w-5 h-5" />}
                                            {method.id === 'gptq' && <Zap className="w-5 h-5" />}
                                            {method.id === 'nf4' && <Brain className="w-5 h-5" />}
                                        </div>
                                        <div>
                                            <h4 className="font-bold text-slate-800 dark:text-slate-100 text-sm">{method.name}</h4>
                                            <p className="text-[10px] text-slate-500">{method.label}</p>
                                        </div>
                                    </div>

                                    <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed mb-3">
                                        {method.description}
                                    </p>

                                    <div className="flex flex-wrap gap-1.5">
                                        {method.tags.map(tag => (
                                            <span
                                                key={tag}
                                                className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400"
                                            >
                                                #{tag}
                                            </span>
                                        ))}
                                    </div>

                                    {/* Hover Stats Bar */}
                                    <div className={cn(
                                        "absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-current to-transparent opacity-0 transition-opacity",
                                        isHovered ? "opacity-30" : ""
                                    )} style={{ color: method.color }} />
                                </motion.div>
                            );
                        })}
                    </div>
                </div>

            </div>

        </div>
    );
}
