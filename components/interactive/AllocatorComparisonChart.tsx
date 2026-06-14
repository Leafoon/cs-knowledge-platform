"use client";

import { useState } from "react";

type Allocator = "linear" | "pool" | "arena";

interface AllocInfo {
    name: string;
    desc: string;
    pros: string[];
    cons: string[];
    speed: string;
    fragmentation: string;
    useCase: string;
    color: string;
}

const allocators: Record<Allocator, AllocInfo> = {
    linear: {
        name: "线性分配器 (Bump Allocator)",
        desc: "顺序分配，仅支持整体释放，最简单的分配策略",
        pros: ["分配速度 O(1)", "无内存碎片", "实现简单"],
        cons: ["不支持单个释放", "内存利用率可能低"],
        speed: "⚡⚡⚡ 极快",
        fragmentation: "🟢 无碎片",
        useCase: "临时计算缓冲区、编译期临时内存",
        color: "from-blue-500 to-cyan-500",
    },
    pool: {
        name: "池分配器 (Pool Allocator)",
        desc: "预分配固定大小块，从空闲链表中取用和归还",
        pros: ["分配/释放 O(1)", "无外部碎片", "适合固定大小对象"],
        cons: ["内部碎片（块大小固定）", "需要预估对象数量"],
        speed: "⚡⚡ 很快",
        fragmentation: "🟡 内部碎片",
        useCase: "算子输出张量、固定 shape 的中间结果",
        color: "from-purple-500 to-pink-500",
    },
    arena: {
        name: "Arena 分配器",
        desc: "在一块大内存区域中线性分配，支持按 arena 整体释放",
        pros: ["分配极快", "批量释放高效", "缓存友好"],
        cons: ["不支持单独释放", "需要预分配足够空间"],
        speed: "⚡⚡⚡ 极快",
        fragmentation: "🟢 低碎片",
        useCase: "Relay 表达式、IR 节点生命周期管理",
        color: "from-amber-500 to-orange-500",
    },
};

export function AllocatorComparisonChart() {
    const [selected, setSelected] = useState<Allocator>("linear");
    const info = allocators[selected];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                内存分配器对比
            </h2>
            <p className="text-center text-sm text-slate-500 dark:text-slate-400 mb-6">
                TVM 编译器中三种主要内存分配策略
            </p>

            {/* Selector tabs */}
            <div className="flex justify-center gap-3 mb-6 flex-wrap">
                {(Object.keys(allocators) as Allocator[]).map((key) => (
                    <button
                        key={key}
                        onClick={() => setSelected(key)}
                        className={`px-5 py-2 rounded-full text-sm font-bold transition-all duration-300 ${
                            selected === key
                                ? "bg-gradient-to-r " + info.color + " text-white shadow-lg scale-105"
                                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:border-indigo-300"
                        }`}
                    >
                        {allocators[key].name.split("(")[0].trim()}
                    </button>
                ))}
            </div>

            {/* Comparison table */}
            <div className="overflow-x-auto mb-6">
                <table className="w-full text-sm border-collapse">
                    <thead>
                        <tr className="bg-indigo-100 dark:bg-indigo-900/40">
                            <th className="p-3 text-left rounded-tl-lg text-indigo-700 dark:text-indigo-300">维度</th>
                            {(Object.keys(allocators) as Allocator[]).map((key) => (
                                <th key={key} className={`p-3 text-center ${key === "arena" ? "rounded-tr-lg" : ""} text-indigo-700 dark:text-indigo-300`}>
                                    {allocators[key].name.split("(")[0].trim()}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {[
                            { label: "分配速度", key: "speed" as const },
                            { label: "碎片情况", key: "fragmentation" as const },
                        ].map((row, i) => (
                            <tr key={row.label} className={i % 2 === 0 ? "bg-white/50 dark:bg-slate-800/50" : "bg-indigo-50/50 dark:bg-indigo-950/30"}>
                                <td className="p-3 font-medium text-slate-700 dark:text-slate-200">{row.label}</td>
                                {(Object.keys(allocators) as Allocator[]).map((key) => (
                                    <td key={key} className="p-3 text-center text-slate-600 dark:text-slate-300">
                                        {allocators[key][row.key]}
                                    </td>
                                ))}
                            </tr>
                        ))}
                        <tr className="bg-white/50 dark:bg-slate-800/50">
                            <td className="p-3 font-medium text-slate-700 dark:text-slate-200">典型场景</td>
                            {(Object.keys(allocators) as Allocator[]).map((key) => (
                                <td key={key} className="p-3 text-center text-xs text-slate-600 dark:text-slate-300">
                                    {allocators[key].useCase}
                                </td>
                            ))}
                        </tr>
                    </tbody>
                </table>
            </div>

            {/* Detail card */}
            <div className="bg-white dark:bg-slate-800/80 rounded-xl p-5 border border-slate-200 dark:border-slate-700">
                <div className={`inline-block px-3 py-1 rounded-full text-xs font-bold text-white bg-gradient-to-r ${info.color} mb-3`}>
                    {info.name}
                </div>
                <p className="text-sm text-slate-600 dark:text-slate-300 mb-4">{info.desc}</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <h4 className="text-xs font-bold text-green-600 dark:text-green-400 mb-2">✅ 优势</h4>
                        <ul className="space-y-1">
                            {info.pros.map((p, i) => (
                                <li key={i} className="text-sm text-slate-600 dark:text-slate-300 flex items-start gap-2">
                                    <span className="text-green-500 mt-0.5">•</span> {p}
                                </li>
                            ))}
                        </ul>
                    </div>
                    <div>
                        <h4 className="text-xs font-bold text-red-600 dark:text-red-400 mb-2">⚠️ 劣势</h4>
                        <ul className="space-y-1">
                            {info.cons.map((c, i) => (
                                <li key={i} className="text-sm text-slate-600 dark:text-slate-300 flex items-start gap-2">
                                    <span className="text-red-500 mt-0.5">•</span> {c}
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
}
