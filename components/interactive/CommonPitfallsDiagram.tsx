"use client";

import { useState } from "react";

const pitfalls = [
    { id: 1, title: "未正确设置 target", symptom: "编译成功但运行极慢", fix: "确保 target 包含正确的 GPU 架构，如 'cuda -arch=sm_80'", step: 1 },
    { id: 2, title: "忽略内存对齐", symptom: "向量化失败或性能下降", fix: "使用 T.address_of() 确保数据按向量宽度对齐", step: 2 },
    { id: 3, title: "过度分块", symptom: "占用率低，SM利用率不足", fix: "tile size 应适配共享内存大小和寄存器数量", step: 2 },
    { id: 4, title: "未处理边界条件", symptom: "部分结果正确，尾部数据错误", fix: "split 后检查内层循环的边界处理", step: 3 },
    { id: 5, title: "错误的轴绑定", symptom: "GPU 结果与 CPU 不一致", fix: "确认 blockIdx/threadIdx 绑定的轴和范围", step: 1 },
    { id: 6, title: "忽略同步", symptom: "间歇性结果错误", fix: "在共享内存读写间添加 T.syncthreads()", step: 3 },
];

export function CommonPitfallsDiagram() {
    const [active, setActive] = useState<number | null>(null);

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">常见陷阱排查</h3>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-base font-bold text-indigo-600 dark:text-indigo-400 mb-4">排查流程</h4>
                <div className="flex items-center justify-center gap-2 mb-4">
                    {["检查配置", "检查调度", "检查运行时"].map((step, i) => (
                        <span key={i} className="flex items-center gap-2">
                            <div className="px-4 py-2 rounded-lg bg-indigo-100 dark:bg-indigo-900/40 text-sm font-medium text-indigo-700 dark:text-indigo-300">
                                {step}
                            </div>
                            {i < 2 && <span className="text-indigo-400">→</span>}
                        </span>
                    ))}
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {pitfalls.map((p) => (
                    <button
                        key={p.id}
                        onClick={() => setActive(active === p.id ? null : p.id)}
                        className={`text-left bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg transition-all hover:shadow-md ${
                            active === p.id ? "ring-2 ring-indigo-500 border-indigo-200 dark:border-indigo-800" : ""
                        }`}
                    >
                        <div className="flex items-center gap-2 mb-2">
                            <div className="w-6 h-6 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center text-xs font-bold text-red-600 dark:text-red-400">
                                {p.id}
                            </div>
                            <h5 className="text-sm font-bold text-slate-800 dark:text-slate-100">{p.title}</h5>
                        </div>
                        <p className="text-xs text-red-500 dark:text-red-400 mb-1">症状: {p.symptom}</p>
                        {active === p.id && (
                            <div className="mt-2 pt-2 border-t border-slate-100 dark:border-slate-700">
                                <p className="text-xs text-emerald-600 dark:text-emerald-400">
                                    <strong>修复:</strong> {p.fix}
                                </p>
                            </div>
                        )}
                    </button>
                ))}
            </div>

            <div className="mt-4 bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg">
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    💡 点击任意陷阱卡片查看修复方案。大多数问题可通过<strong className="text-indigo-600 dark:text-indigo-400">逐步检查 target 配置 → 调度原语 → 运行时行为</strong>来定位。
                </p>
            </div>
        </div>
    );
}
