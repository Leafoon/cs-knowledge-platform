"use client";

import { useState } from "react";

export function DQNArchitecture() {
    const [view, setView] = useState<"network" | "training">("network");

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-slate-900 dark:to-blue-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    DQN 架构
                </h3>
            </div>

            {/* 视角切换 */}
            <div className="flex justify-center gap-4 mb-8">
                <button onClick={() => setView("network")} className={`px-8 py-3 rounded-xl font-bold transition-all ${view === "network" ? "bg-blue-600 text-white shadow-lg" : "bg-blue-100 text-blue-700 dark:bg-blue-900/30"}`}>
                    网络架构
                </button>
                <button onClick={() => setView("training")} className={`px-8 py-3 rounded-xl font-bold transition-all ${view === "training" ? "bg-cyan-600 text-white shadow-lg" : "bg-cyan-100 text-cyan-700 dark:bg-cyan-900/30"}`}>
                    训练流程
                </button>
            </div>

            {/* 内容展示 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                {view === "network" ? (
                    <div className="space-y-6">
                        <h4 className="text-xl font-bold mb-4">Atari DQN 网络架构</h4>
                        <div className="space-y-4 font-mono text-sm">
                            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                                Input: 84×84×4 (4 stacked frames)
                            </div>
                            <div className="text-center">↓</div>
                            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                                Conv2D: 32 filters, 8×8, stride 4 → ReLU
                            </div>
                            <div className="text-center">↓</div>
                            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                                Conv2D: 64 filters, 4×4, stride 2 → ReLU
                            </div>
                            <div className="text-center">↓</div>
                            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                                Conv2D: 64 filters, 3×3, stride 1 → ReLU
                            </div>
                            <div className="text-center">↓</div>
                            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                                Flatten → Fully Connected: 512 units → ReLU
                            </div>
                            <div className="text-center">↓</div>
                            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded border-2 border-green-500">
                                Output: action_dim Q-values
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="space-y-6">
                        <h4 className="text-xl font-bold mb-4">DQN 训练流程</h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded border-l-4 border-cyan-500">
                                <div className="font-bold mb-2">1. 收集经验</div>
                                <div className="text-sm">ε-greedy 选择动作，执行并存储 (s,a,r,s') 到 Replay Buffer</div>
                            </div>
                            <div className="p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded border-l-4 border-cyan-500">
                                <div className="font-bold mb-2">2. 采样 Batch</div>
                                <div className="text-sm">从 Buffer 随机采样 64 个转移</div>
                            </div>
                            <div className="p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded border-l-4 border-cyan-500">
                                <div className="font-bold mb-2">3. 计算目标</div>
                                <div className="text-sm">y = r + γ max Q<sub>target</sub>(s',a')</div>
                            </div>
                            <div className="p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded border-l-4 border-cyan-500">
                                <div className="font-bold mb-2">4. 更新网络</div>
                                <div className="text-sm">Loss = (Q(s,a) - y)², 梯度下降</div>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 DQN 在 49 个 Atari 游戏中达到人类水平
            </div>
        </div>
    );
}
