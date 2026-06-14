"use client";

import { useState } from "react";

const passes = [
    { name: "Pass 1: 算子融合", input: "Conv → BN → Relu", output: "FusedConvBNRelu", status: "success" as const },
    { name: "Pass 2: 常量折叠", input: "const(2) * const(3)", output: "const(6)", status: "success" as const },
    { name: "Pass 3: 死代码消除", input: "unused = compute(...)", output: "(removed)", status: "success" as const },
    { name: "Pass 4: 布局变换", input: "NCHW → NHWC", output: "layout_transform", status: "warning" as const },
];

export function CompilationDebugger() {
    const [activePass, setActivePass] = useState(0);

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">编译调试面板</h3>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="flex items-center gap-3 mb-4">
                    <div className="px-3 py-1 rounded-lg bg-indigo-100 dark:bg-indigo-900/40 text-sm font-mono text-indigo-700 dark:text-indigo-300">
                        输入 IR
                    </div>
                    <span className="text-slate-400">→</span>
                    {passes.map((p, i) => (
                        <span key={i} className="flex items-center gap-2">
                            <button
                                onClick={() => setActivePass(i)}
                                className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                                    activePass === i
                                        ? "bg-indigo-600 text-white"
                                        : p.status === "success"
                                            ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300"
                                            : "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300"
                                }`}
                            >
                                Pass {i + 1}
                            </button>
                            {i < passes.length - 1 && <span className="text-slate-300">→</span>}
                        </span>
                    ))}
                    <span className="text-slate-400">→</span>
                    <div className="px-3 py-1 rounded-lg bg-emerald-100 dark:bg-emerald-900/40 text-sm font-mono text-emerald-700 dark:text-emerald-300">
                        输出 IR
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                    <h4 className="text-base font-bold text-indigo-600 dark:text-indigo-400 mb-3">Pass 列表</h4>
                    <div className="space-y-2">
                        {passes.map((p, i) => (
                            <button
                                key={i}
                                onClick={() => setActivePass(i)}
                                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-all ${
                                    activePass === i
                                        ? "bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 font-medium"
                                        : "text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-700"
                                }`}
                            >
                                <span className={`inline-block w-2 h-2 rounded-full mr-2 ${p.status === "success" ? "bg-emerald-500" : "bg-amber-500"}`} />
                                {p.name}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                    <h4 className="text-base font-bold text-purple-600 dark:text-purple-400 mb-3">输入</h4>
                    <pre className="bg-slate-100 dark:bg-slate-900 rounded-lg p-3 text-xs font-mono text-slate-700 dark:text-green-400 overflow-x-auto">
                        {passes[activePass].input}
                    </pre>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                    <h4 className="text-base font-bold text-emerald-600 dark:text-emerald-400 mb-3">输出</h4>
                    <pre className="bg-slate-100 dark:bg-slate-900 rounded-lg p-3 text-xs font-mono text-slate-700 dark:text-green-400 overflow-x-auto">
                        {passes[activePass].output}
                    </pre>
                    <div className={`mt-2 text-xs font-medium ${passes[activePass].status === "success" ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600 dark:text-amber-400"}`}>
                        {passes[activePass].status === "success" ? "✓ 变换成功" : "⚠ 需要验证"}
                    </div>
                </div>
            </div>
        </div>
    );
}
