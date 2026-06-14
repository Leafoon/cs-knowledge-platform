"use client";

import { useState } from "react";

const chapters = [
    { id: "ch1", title: "TVM 概述", topics: ["编译器栈架构", "端到端流程", "与框架对比"], icon: "🌐" },
    { id: "ch2", title: "Relay IR", topics: ["计算图表示", "类型系统", "函数式语义"], icon: "📐" },
    { id: "ch3", title: "TE 调度", topics: ["张量表达式", "调度原语", "循环变换"], icon: "🔄" },
    { id: "ch4", title: "TIR 低级IR", topics: ["Tensor IR", "Buffer 注解", "循环嵌套"], icon: "⚙️" },
    { id: "ch5", title: "算子融合", topics: ["融合规则", "模式匹配", "融合策略"], icon: "🔗" },
    { id: "ch6", title: "内存管理", topics: ["分配器设计", "内存规划", "存储规划"], icon: "💾" },
    { id: "ch7", title: "AutoTVM", topics: ["搜索空间", "成本模型", "RPC调优"], icon: "🤖" },
    { id: "ch8", title: "代码生成", topics: ["LLVM CodeGen", "CUDA CodeGen", "AOT编译"], icon: "🏗️" },
    { id: "ch9", title: "编译引擎", topics: ["Cache/Workspace", "DeviceAPI", "调用链"], icon: "🔧" },
    { id: "ch10", title: "部署与优化", topics: ["量化", "Tuning", "Runtime"], icon: "🚀" },
];

export function ChapterSummaryDiagram() {
    const [activeChapter, setActiveChapter] = useState("ch1");

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                TVM 章节知识总结
            </h2>
            <p className="text-center text-sm text-slate-500 dark:text-slate-400 mb-6">
                系统性回顾 TVM 深度学习编译器核心知识
            </p>

            {/* Chapter grid */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mb-6">
                {chapters.map((ch) => (
                    <button
                        key={ch.id}
                        onClick={() => setActiveChapter(ch.id)}
                        className={`flex flex-col items-center px-3 py-3 rounded-xl border-2 transition-all duration-300 ${
                            activeChapter === ch.id
                                ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-900/30 shadow-lg scale-105"
                                : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-indigo-300"
                        }`}
                    >
                        <span className="text-xl mb-1">{ch.icon}</span>
                        <span className="text-xs font-bold text-slate-700 dark:text-slate-200 text-center">{ch.title}</span>
                    </button>
                ))}
            </div>

            {/* Active chapter detail */}
            <div className="bg-white dark:bg-slate-800/80 rounded-xl p-5 border border-slate-200 dark:border-slate-700 mb-6">
                <div className="flex items-center gap-2 mb-3">
                    <span className="text-2xl">{chapters.find((c) => c.id === activeChapter)?.icon}</span>
                    <h3 className="text-lg font-bold text-slate-700 dark:text-slate-200">
                        {chapters.find((c) => c.id === activeChapter)?.title}
                    </h3>
                </div>
                <div className="flex flex-wrap gap-2">
                    {chapters
                        .find((c) => c.id === activeChapter)
                        ?.topics.map((t, i) => (
                            <span key={i} className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-indigo-100 to-purple-100 dark:from-indigo-900/40 dark:to-purple-900/40 text-xs font-medium text-indigo-700 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800">
                                {t}
                            </span>
                        ))}
                </div>
            </div>

            {/* Knowledge map */}
            <div className="bg-white/60 dark:bg-slate-800/60 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
                <h3 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-3">知识脉络</h3>
                <div className="flex items-center justify-center gap-2 flex-wrap text-xs">
                    {chapters.map((ch, i) => (
                        <div key={ch.id} className="flex items-center gap-1">
                            <span className={`px-2 py-1 rounded-full font-medium ${
                                activeChapter === ch.id
                                    ? "bg-gradient-to-r from-indigo-500 to-purple-500 text-white"
                                    : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300"
                            }`}>
                                {ch.title}
                            </span>
                            {i < chapters.length - 1 && <span className="text-indigo-400">→</span>}
                        </div>
                    ))}
                </div>
            </div>

            {/* Key takeaways */}
            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
                {[
                    { title: "核心理念", desc: "将深度学习框架的计算图编译为目标硬件原生代码" },
                    { title: "关键技术", desc: "IR 变换、调度搜索、算子融合、内存规划" },
                    { title: "部署优势", desc: "跨平台、高性能、低延迟、小体积" },
                ].map((item, i) => (
                    <div key={i} className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-950/50 dark:to-purple-950/50 rounded-lg p-3 border border-indigo-200 dark:border-indigo-800">
                        <div className="text-xs font-bold text-indigo-600 dark:text-indigo-400">{item.title}</div>
                        <div className="text-[11px] text-slate-600 dark:text-slate-300 mt-1">{item.desc}</div>
                    </div>
                ))}
            </div>
        </div>
    );
}
