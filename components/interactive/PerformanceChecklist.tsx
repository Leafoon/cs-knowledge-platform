"use client";

import { useState } from "react";

const items = [
    { id: 1, title: "循环分块 (Tiling)", desc: "将大循环分解为小块以提升缓存利用率", checked: true },
    { id: 2, title: "向量化 (Vectorization)", desc: "使用SIMD指令一次处理多个数据", checked: true },
    { id: 3, title: "循环重排 (Reorder)", desc: "调整循环顺序以改善数据局部性", checked: false },
    { id: 4, title: "算子融合 (Fusion)", desc: "合并相邻算子减少中间张量", checked: true },
    { id: 5, title: "内存作用域 (Memory Scope)", desc: "将数据放置在共享内存/寄存器", checked: false },
    { id: 6, title: "并行化 (Parallelization)", desc: "将循环绑定到GPU线程/块", checked: true },
    { id: 7, title: "预取 (Prefetch)", desc: "提前加载数据隐藏内存延迟", checked: false },
];

export function PerformanceChecklist() {
    const [checks, setChecks] = useState(items.map((it) => it.checked));

    const toggle = (i: number) => {
        setChecks((prev) => prev.map((v, j) => (j === i ? !v : v)));
    };

    const done = checks.filter(Boolean).length;

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">性能优化清单</h3>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg mb-6">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-slate-600 dark:text-slate-400">完成进度</span>
                    <span className="text-sm font-bold text-indigo-600 dark:text-indigo-400">{done}/{items.length}</span>
                </div>
                <div className="w-full bg-slate-100 dark:bg-slate-700 rounded-full h-3 overflow-hidden">
                    <div
                        className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-500"
                        style={{ width: `${(done / items.length) * 100}%` }}
                    />
                </div>
            </div>

            <div className="space-y-3">
                {items.map((item, i) => (
                    <button
                        key={item.id}
                        onClick={() => toggle(i)}
                        className={`w-full text-left bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg flex items-start gap-4 transition-all hover:shadow-md ${
                            checks[i] ? "border-l-4 border-emerald-500" : "border-l-4 border-slate-200 dark:border-slate-700"
                        }`}
                    >
                        <div className={`w-6 h-6 rounded-lg flex items-center justify-center shrink-0 mt-0.5 transition-all ${
                            checks[i]
                                ? "bg-emerald-500 text-white"
                                : "bg-slate-100 dark:bg-slate-700 border-2 border-slate-300 dark:border-slate-600"
                        }`}>
                            {checks[i] && <span className="text-sm">✓</span>}
                        </div>
                        <div>
                            <h4 className={`text-sm font-bold ${checks[i] ? "text-emerald-700 dark:text-emerald-400" : "text-slate-800 dark:text-slate-100"}`}>
                                {item.title}
                            </h4>
                            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">{item.desc}</p>
                        </div>
                    </button>
                ))}
            </div>
        </div>
    );
}
