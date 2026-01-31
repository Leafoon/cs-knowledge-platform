"use client";

import { useState } from "react";

export function NeuralNetworkApproximation() {
    const [architecture, setArchitecture] = useState<"shallow" | "deep">("shallow");

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-slate-900 dark:to-purple-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    神经网络价值逼近
                </h3>
            </div>

            {/* 架构选择 */}
            <div className="flex justify-center gap-4 mb-8">
                <button
                    onClick={() => setArchitecture("shallow")}
                    className={`px-8 py-3 rounded-xl font-bold transition-all ${
                        architecture === "shallow"
                            ? "bg-purple-600 text-white shadow-lg"
                            : "bg-purple-100 text-purple-700"
                    }`}
                >
                    浅层网络
                </button>
                <button
                    onClick={() => setArchitecture("deep")}
                    className={`px-8 py-3 rounded-xl font-bold transition-all ${
                        architecture === "deep"
                            ? "bg-pink-600 text-white shadow-lg"
                            : "bg-pink-100 text-pink-700"
                    }`}
                >
                    深层网络
                </button>
            </div>

            {/* 网络架构图 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-8 shadow-lg mb-6">
                {architecture === "shallow" ? (
                    <div className="space-y-4">
                        <h4 className="text-lg font-bold">浅层网络（1-2 隐藏层）</h4>
                        <div className="font-mono text-sm bg-purple-50 dark:bg-purple-900/20 p-4 rounded">
                            Input (state_dim) → [128] ReLU → [128] ReLU → Output (1 或 action_dim)
                        </div>
                        <div className="text-sm">
                            <strong>适用:</strong> 简单任务（CartPole, LunarLander）<br />
                            <strong>优点:</strong> 训练快、易调试<br />
                            <strong>缺点:</strong> 表达能力有限
                        </div>
                    </div>
                ) : (
                    <div className="space-y-4">
                        <h4 className="text-lg font-bold">深层网络（3+ 隐藏层/卷积）</h4>
                        <div className="font-mono text-sm bg-pink-50 dark:bg-pink-900/20 p-4 rounded">
                            Input (84×84×4) → Conv → Conv → Conv → Flatten → [512] ReLU → Output (action_dim)
                        </div>
                        <div className="text-sm">
                            <strong>适用:</strong> 复杂任务（Atari, 图像输入）<br />
                            <strong>优点:</strong> 强大表达能力<br />
                            <strong>缺点:</strong> 训练慢、需大量数据
                        </div>
                    </div>
                )}
            </div>

            {/* DQN 关键技术 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">DQN 核心技术</h4>
                <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong>Experience Replay</strong>
                        <div className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                            打破样本相关性
                        </div>
                    </div>
                    <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong>Target Network</strong>
                        <div className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                            稳定训练目标
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 DQN (2015) 首次在 Atari 游戏达到人类水平，开启深度 RL 时代
            </div>
        </div>
    );
}
