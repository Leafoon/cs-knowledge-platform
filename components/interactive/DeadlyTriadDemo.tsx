"use client";

import { useState } from "react";

export function DeadlyTriadDemo() {
    const [activeElements, setActiveElements] = useState<Set<string>>(new Set());

    const toggleElement = (element: string) => {
        const newSet = new Set(activeElements);
        if (newSet.has(element)) {
            newSet.delete(element);
        } else {
            newSet.add(element);
        }
        setActiveElements(newSet);
    };

    const isDangerous = activeElements.size === 3;

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-red-50 to-orange-50 dark:from-slate-900 dark:to-red-950 rounded-2xl shadow-xl">
            <div className="text-center mb-8">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Deadly Triad 演示
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    点击选择要素，观察危险组合
                </p>
            </div>

            {/* 三个要素 */}
            <div className="grid grid-cols-3 gap-6 mb-8">
                <button
                    onClick={() => toggleElement("fa")}
                    className={`p-6 rounded-xl border-4 transition-all ${
                        activeElements.has("fa")
                            ? "border-red-500 bg-red-100 dark:bg-red-900/30 scale-105"
                            : "border-slate-300 bg-white dark:bg-slate-800"
                    }`}
                >
                    <div className="text-4xl mb-2">📊</div>
                    <div className="font-bold">Function Approximation</div>
                    <div className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                        函数逼近
                    </div>
                </button>

                <button
                    onClick={() => toggleElement("boot")}
                    className={`p-6 rounded-xl border-4 transition-all ${
                        activeElements.has("boot")
                            ? "border-orange-500 bg-orange-100 dark:bg-orange-900/30 scale-105"
                            : "border-slate-300 bg-white dark:bg-slate-800"
                    }`}
                >
                    <div className="text-4xl mb-2">⚡</div>
                    <div className="font-bold">Bootstrapping</div>
                    <div className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                        自举（TD 方法）
                    </div>
                </button>

                <button
                    onClick={() => toggleElement("offpolicy")}
                    className={`p-6 rounded-xl border-4 transition-all ${
                        activeElements.has("offpolicy")
                            ? "border-yellow-500 bg-yellow-100 dark:bg-yellow-900/30 scale-105"
                            : "border-slate-300 bg-white dark:bg-slate-800"
                    }`}
                >
                    <div className="text-4xl mb-2">🔀</div>
                    <div className="font-bold">Off-policy</div>
                    <div className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                        离策略学习
                    </div>
                </button>
            </div>

            {/* 危险警告 */}
            {isDangerous ? (
                <div className="bg-red-100 dark:bg-red-900/30 border-4 border-red-500 rounded-xl p-8 text-center">
                    <div className="text-6xl mb-4">⚠️</div>
                    <h4 className="text-2xl font-bold text-red-800 dark:text-red-300 mb-3">
                        Deadly Triad 激活！
                    </h4>
                    <p className="text-red-700 dark:text-red-400 mb-4">
                        三个要素同时存在，算法可能<strong>发散</strong>！
                    </p>
                    <div className="text-sm text-red-600 dark:text-red-400">
                        需要特殊技术：Experience Replay、Target Network、Gradient TD等
                    </div>
                </div>
            ) : (
                <div className="bg-green-100 dark:bg-green-900/30 border-2 border-green-500 rounded-xl p-6 text-center">
                    <div className="text-4xl mb-3">✅</div>
                    <p className="text-green-800 dark:text-green-300">
                        当前组合安全（已选择 {activeElements.size}/3）
                    </p>
                </div>
            )}

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 Baird反例：线性函数+Off-policy TD 会导致权重发散→∞
            </div>
        </div>
    );
}
