"use client";

import { useState } from "react";

const layouts = [
    {
        name: "未对齐",
        desc: "数据未按自然边界对齐，可能需要多次内存访问",
        blocks: [
            { label: "A(1B)", size: 1, pad: false },
            { label: "B(4B)", size: 4, pad: false },
            { label: "C(2B)", size: 2, pad: false },
            { label: "D(4B)", size: 4, pad: false },
        ],
        aligned: false,
    },
    {
        name: "已对齐",
        desc: "数据按 4 字节边界对齐，单次访问完成",
        blocks: [
            { label: "A(1B)", size: 1, pad: false },
            { label: "pad", size: 3, pad: true },
            { label: "B(4B)", size: 4, pad: false },
            { label: "C(2B)", size: 2, pad: false },
            { label: "pad", size: 2, pad: true },
            { label: "D(4B)", size: 4, pad: false },
        ],
        aligned: true,
    },
    {
        name: "SIMD 对齐(16B)",
        desc: "向量化操作需要 16/32 字节对齐",
        blocks: [
            { label: "V0(16B)", size: 16, pad: false },
            { label: "V1(16B)", size: 16, pad: false },
            { label: "V2(16B)", size: 16, pad: false },
            { label: "V3(16B)", size: 16, pad: false },
        ],
        aligned: true,
    },
];

export function MemoryAlignmentDiagram() {
    const [active, setActive] = useState(1);
    const layout = layouts[active];

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">内存对齐图示</h3>

            <div className="flex gap-2 mb-6">
                {layouts.map((l, i) => (
                    <button
                        key={i}
                        onClick={() => setActive(i)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                            active === i
                                ? "bg-indigo-600 text-white shadow-lg"
                                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:border-indigo-300"
                        }`}
                    >
                        {l.name}
                    </button>
                ))}
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">{layout.desc}</p>
                <div className="flex gap-1 flex-wrap">
                    {layout.blocks.map((b, i) => (
                        <div
                            key={i}
                            className={`flex items-center justify-center text-xs font-mono border rounded transition-all ${
                                b.pad
                                    ? "bg-slate-100 dark:bg-slate-700 border-slate-300 dark:border-slate-600 text-slate-400 border-dashed"
                                    : layout.aligned
                                        ? "bg-indigo-50 dark:bg-indigo-900/30 border-indigo-300 dark:border-indigo-700 text-indigo-700 dark:text-indigo-300"
                                        : "bg-red-50 dark:bg-red-900/30 border-red-300 dark:border-red-700 text-red-700 dark:text-red-300"
                            }`}
                            style={{ minWidth: `${Math.max(b.size * 12, 48)}px`, height: "40px" }}
                        >
                            {b.label}
                        </div>
                    ))}
                </div>
                <div className="flex gap-4 mt-3 text-xs text-slate-500">
                    <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-indigo-100 dark:bg-indigo-900/30 border border-indigo-300 dark:border-indigo-700 inline-block" /> 有效数据</span>
                    <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-slate-100 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 border-dashed inline-block" /> 填充</span>
                    <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-red-50 dark:bg-red-900/30 border border-red-300 dark:border-red-700 inline-block" /> 未对齐</span>
                </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
                {[
                    { label: "对齐粒度", value: active === 2 ? "16B" : "4B", color: "text-indigo-600 dark:text-indigo-400" },
                    { label: "填充字节", value: layout.blocks.filter(b => b.pad).length.toString(), color: "text-purple-600 dark:text-purple-400" },
                    { label: "总占用", value: `${layout.blocks.reduce((s, b) => s + b.size, 0)}B`, color: "text-blue-600 dark:text-blue-400" },
                ].map((m, i) => (
                    <div key={i} className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg text-center">
                        <div className={`text-xl font-bold font-mono ${m.color}`}>{m.value}</div>
                        <div className="text-xs text-slate-500 mt-1">{m.label}</div>
                    </div>
                ))}
            </div>
        </div>
    );
}
