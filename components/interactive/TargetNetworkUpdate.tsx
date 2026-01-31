"use client";

import { useState } from "react";

export function TargetNetworkUpdate() {
    const [updateType, setUpdateType] = useState<"hard" | "soft">("hard");

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-slate-900 dark:to-purple-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Target Network 更新机制
                </h3>
            </div>

            <div className="flex justify-center gap-4 mb-6">
                <button onClick={() => setUpdateType("hard")} className={`px-6 py-2 rounded-lg font-bold ${updateType === "hard" ? "bg-purple-600 text-white" : "bg-purple-100 text-purple-700"}`}>
                    硬更新 (Hard)
                </button>
                <button onClick={() => setUpdateType("soft")} className={`px-6 py-2 rounded-lg font-bold ${updateType === "soft" ? "bg-pink-600 text-white" : "bg-pink-100 text-pink-700"}`}>
                    软更新 (Soft)
                </button>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                {updateType === "hard" ? (
                    <div>
                        <h4 className="text-lg font-bold mb-4">硬更新（DQN 标准）</h4>
                        <div className="space-y-4">
                            <div className="font-mono text-sm bg-purple-50 dark:bg-purple-900/20 p-4 rounded">
                                if step % update_freq == 0:<br />
                                &nbsp;&nbsp;θ⁻ ← θ
                            </div>
                            <div className="text-sm">
                                <strong>特点:</strong><br />
                                • 定期完全复制（例如每 10,000 步）<br />
                                • 目标长时间固定<br />
                                • 适用于 DQN
                            </div>
                        </div>
                    </div>
                ) : (
                    <div>
                        <h4 className="text-lg font-bold mb-4">软更新（Polyak Averaging）</h4>
                        <div className="space-y-4">
                            <div className="font-mono text-sm bg-pink-50 dark:bg-pink-900/20 p-4 rounded">
                                θ⁻ ← τ θ + (1-τ) θ⁻<br />
                                <span className="text-xs">(τ = 0.001, 每步更新)</span>
                            </div>
                            <div className="text-sm">
                                <strong>特点:</strong><br />
                                • 每步微小更新<br />
                                • 渐变的目标变化<br />
                                • 适用于 DDPG, SAC
                            </div>
                        </div>
                    </div>
                )}
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 Target Network 防止训练目标频繁变化，稳定学习
            </div>
        </div>
    );
}
