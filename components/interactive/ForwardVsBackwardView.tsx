"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function ForwardVsBackwardView() {
    const [view, setView] = useState<"forward" | "backward">("forward");

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-slate-900 dark:to-cyan-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    前向视角 vs 后向视角
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    两种等价的资格迹理解方式
                </p>
            </div>

            {/* 视角切换 */}
            <div className="flex justify-center gap-4 mb-8">
                <button
                    onClick={() => setView("forward")}
                    className={`px-8 py-3 rounded-xl font-bold text-lg transition-all ${
                        view === "forward"
                            ? "bg-cyan-600 text-white shadow-lg scale-105"
                            : "bg-cyan-100 text-cyan-700 dark:bg-cyan-900/30 dark:text-cyan-300"
                    }`}
                >
                    前向视角 (Forward)
                </button>
                <button
                    onClick={() => setView("backward")}
                    className={`px-8 py-3 rounded-xl font-bold text-lg transition-all ${
                        view === "backward"
                            ? "bg-blue-600 text-white shadow-lg scale-105"
                            : "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300"
                    }`}
                >
                    后向视角 (Backward)
                </button>
            </div>

            {/* 内容展示 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-8 shadow-lg mb-6">
                {view === "forward" ? (
                    <div>
                        <h4 className="text-2xl font-bold text-cyan-600 mb-4">前向视角 (Forward View)</h4>
                        
                        <div className="space-y-6">
                            <div>
                                <h5 className="font-bold text-lg mb-2">💭 核心思想</h5>
                                <p className="text-slate-600 dark:text-slate-400">
                                    从当前状态向<strong>未来</strong>看，使用 λ-return 作为更新目标
                                </p>
                            </div>

                            <div className="p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded-lg font-mono text-sm">
                                V(S<sub>t</sub>) ← V(S<sub>t</sub>) + α[G<sub>t</sub><sup>λ</sup> - V(S<sub>t</sub>)]
                            </div>

                            <div>
                                <h5 className="font-bold text-lg mb-2">✅ 优点</h5>
                                <ul className="list-disc list-inside text-slate-600 dark:text-slate-400 space-y-1">
                                    <li>概念清晰直观</li>
                                    <li>数学推导简单</li>
                                    <li>理论分析容易</li>
                                </ul>
                            </div>

                            <div>
                                <h5 className="font-bold text-lg mb-2">❌ 缺点</h5>
                                <ul className="list-disc list-inside text-slate-600 dark:text-slate-400 space-y-1">
                                    <li>需要等待未来多步</li>
                                    <li>不能完全在线实现</li>
                                    <li>计算复杂度高</li>
                                </ul>
                            </div>

                            <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border-l-4 border-yellow-500">
                                <p className="text-sm text-yellow-800 dark:text-yellow-300">
                                    ⚠️ <strong>实践问题</strong>：离线算法，需要完整 episode 或多步延迟
                                </p>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div>
                        <h4 className="text-2xl font-bold text-blue-600 mb-4">后向视角 (Backward View)</h4>
                        
                        <div className="space-y-6">
                            <div>
                                <h5 className="font-bold text-lg mb-2">💭 核心思想</h5>
                                <p className="text-slate-600 dark:text-slate-400">
                                    从当前状态向<strong>过去</strong>看，使用资格迹向量分配信用
                                </p>
                            </div>

                            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg space-y-2 font-mono text-sm">
                                <div>e(s) ← γλe(s) + 1  <span className="text-blue-600">(if s=S<sub>t</sub>)</span></div>
                                <div>V(s) ← V(s) + αδe(s)  <span className="text-slate-500">(for all s)</span></div>
                            </div>

                            <div>
                                <h5 className="font-bold text-lg mb-2">✅ 优点</h5>
                                <ul className="list-disc list-inside text-slate-600 dark:text-slate-400 space-y-1">
                                    <li><strong>完全在线</strong>：每步都可更新</li>
                                    <li>计算高效（稀疏更新）</li>
                                    <li>实现简单</li>
                                </ul>
                            </div>

                            <div>
                                <h5 className="font-bold text-lg mb-2">❌ 缺点</h5>
                                <ul className="list-disc list-inside text-slate-600 dark:text-slate-400 space-y-1">
                                    <li>概念稍复杂</li>
                                    <li>需要维护资格迹向量</li>
                                    <li>理论推导较难</li>
                                </ul>
                            </div>

                            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border-l-4 border-green-500">
                                <p className="text-sm text-green-800 dark:text-green-300">
                                    ✅ <strong>实践优势</strong>：真正的在线算法，实际应用中的标准方法
                                </p>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* 等价性说明 */}
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6 border-2 border-purple-300 dark:border-purple-600">
                <h4 className="font-bold text-purple-800 dark:text-purple-300 mb-3 text-lg">
                    🔄 理论等价性
                </h4>
                <p className="text-purple-700 dark:text-purple-400">
                    在<strong>离线更新</strong>（episode 结束后）的情况下，前向视角和后向视角产生<strong>完全相同</strong>的价值函数更新。
                    但在<strong>在线更新</strong>中，后向视角更实用，而 True Online TD(λ) 精确逼近前向视角。
                </p>
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 实践中几乎总是使用后向视角（资格迹）实现
            </div>
        </div>
    );
}
