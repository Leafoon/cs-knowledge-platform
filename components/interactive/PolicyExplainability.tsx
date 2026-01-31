"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function PolicyExplainability() {
    const [method, setMethod] = useState<"saliency" | "ig" | "attention">("saliency");
    const [selectedAction, setSelectedAction] = useState(2); // RIGHT

    // Simulated state features
    const features = [
        { name: "位置X", value: 0.6, importance: 0.85 },
        { name: "位置Y", value: 0.4, importance: 0.72 },
        { name: "速度X", value: 0.3, importance: 0.45 },
        { name: "速度Y", value: -0.2, importance: 0.38 },
        { name: "目标距离", value: 0.8, importance: 0.95 },
        { name: "障碍物距离", value: 0.5, importance: 0.88 },
        { name: "角度", value: 0.1, importance: 0.55 },
        { name: "能量", value: 0.9, importance: 0.25 }
    ];

    // Adjust importance based on method
    const adjustedFeatures = features.map(f => ({
        ...f,
        importance: method === "ig"
            ? f.importance * 0.9 // IG is more conservative
            : method === "attention"
                ? f.importance * 1.1 // Attention highlights key features more
                : f.importance
    }));

    const actions = ["↑ UP", "↓ DOWN", "→ RIGHT", "← LEFT"];

    const methods = [
        { id: "saliency", name: "Saliency Maps", desc: "∂π/∂s", color: "blue" },
        { id: "ig", name: "Integrated Gradients", desc: "积分归因", color: "purple" },
        { id: "attention", name: "Attention", desc: "注意力权重", color: "green" }
    ];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-green-50 to-blue-50 dark:from-slate-900 dark:to-green-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    策略可解释性
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Policy Explainability & Feature Attribution
                </p>
            </div>

            {/* Method Selection */}
            <div className="flex gap-3 justify-center mb-6">
                {methods.map((m) => (
                    <button
                        key={m.id}
                        onClick={() => setMethod(m.id as any)}
                        className={`px-6 py-3 rounded-xl font-semibold transition ${method === m.id
                                ? `bg-${m.color}-600 text-white shadow-lg`
                                : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                            }`}
                    >
                        <div className="text-sm">{m.name}</div>
                        <div className="text-xs opacity-75">{m.desc}</div>
                    </button>
                ))}
            </div>

            {/* Action Selection */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg mb-6">
                <div className="text-sm font-semibold mb-3 text-slate-700 dark:text-slate-300">
                    选择动作查看特征归因
                </div>
                <div className="grid grid-cols-4 gap-3">
                    {actions.map((action, idx) => (
                        <button
                            key={idx}
                            onClick={() => setSelectedAction(idx)}
                            className={`py-2 px-4 rounded-lg font-semibold transition ${selectedAction === idx
                                    ? "bg-green-600 text-white"
                                    : "bg-gray-200 dark:bg-gray-700 text-slate-700 dark:text-slate-300"
                                }`}
                        >
                            {action}
                        </button>
                    ))}
                </div>
            </div>

            {/* Feature Importance */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    特征重要性分析
                </h4>

                <div className="space-y-3">
                    {adjustedFeatures
                        .sort((a, b) => b.importance - a.importance)
                        .map((feature, idx) => (
                            <motion.div
                                key={feature.name}
                                className="flex items-center gap-4"
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: idx * 0.05 }}
                            >
                                <div className="w-32 text-sm font-semibold text-slate-700 dark:text-slate-300">
                                    {feature.name}
                                </div>

                                <div className="flex-1 h-8 bg-gray-200 dark:bg-gray-700 rounded-lg relative overflow-hidden">
                                    <motion.div
                                        className={`h-full ${feature.importance > 0.7
                                                ? "bg-red-500"
                                                : feature.importance > 0.5
                                                    ? "bg-orange-500"
                                                    : feature.importance > 0.3
                                                        ? "bg-yellow-500"
                                                        : "bg-green-500"
                                            }`}
                                        initial={{ width: 0 }}
                                        animate={{ width: `${feature.importance * 100}%` }}
                                        transition={{ duration: 0.8, delay: idx * 0.05 }}
                                    />
                                    <div className="absolute inset-0 flex items-center px-3">
                                        <span className="text-xs font-bold text-white">
                                            {(feature.importance * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </div>

                                <div className="w-20 text-right text-sm text-slate-600 dark:text-slate-400">
                                    {feature.value > 0 ? '+' : ''}{feature.value.toFixed(2)}
                                </div>
                            </motion.div>
                        ))}
                </div>
            </div>

            {/* Heatmap Visualization */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">
                    状态空间热力图
                </h4>

                <div className="grid grid-cols-8 gap-1">
                    {Array.from({ length: 64 }).map((_, idx) => {
                        const x = idx % 8;
                        const y = Math.floor(idx / 8);

                        // Create a gradient based on position
                        const distToCenter = Math.sqrt((x - 3.5) ** 2 + (y - 3.5) ** 2);
                        const importance = Math.max(0, 1 - distToCenter / 5);

                        // Adjust based on selected action
                        const actionAdjust =
                            selectedAction === 0 ? (7 - y) / 7 : // UP
                                selectedAction === 1 ? y / 7 : // DOWN
                                    selectedAction === 2 ? x / 7 : // RIGHT
                                        (7 - x) / 7; // LEFT

                        const finalImportance = importance * 0.5 + actionAdjust * 0.5;

                        return (
                            <motion.div
                                key={idx}
                                className="aspect-square rounded"
                                style={{
                                    backgroundColor: `rgba(239, 68, 68, ${finalImportance})`
                                }}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: idx * 0.005 }}
                            />
                        );
                    })}
                </div>

                <div className="mt-3 flex items-center justify-center gap-4 text-xs text-slate-600 dark:text-slate-400">
                    <span>低重要性</span>
                    <div className="flex gap-1">
                        {[0.2, 0.4, 0.6, 0.8, 1.0].map((alpha) => (
                            <div
                                key={alpha}
                                className="w-8 h-4 rounded"
                                style={{ backgroundColor: `rgba(239, 68, 68, ${alpha})` }}
                            />
                        ))}
                    </div>
                    <span>高重要性</span>
                </div>
            </div>

            {/* Interpretation */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-3 text-slate-800 dark:text-slate-100">
                    决策解释
                </h4>

                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <div className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                        <div>
                            <strong>选择动作:</strong> {actions[selectedAction]}
                        </div>
                        <div>
                            <strong>关键因素:</strong> {adjustedFeatures.filter(f => f.importance > 0.7).map(f => f.name).join("、")}
                        </div>
                        <div>
                            <strong>决策逻辑:</strong>
                            {selectedAction === 2
                                ? " 目标距离较远且位置X偏低，策略学习向右移动以接近目标，同时避开左侧障碍物。"
                                : selectedAction === 0
                                    ? " 位置Y较低但目标在上方，策略决定向上移动。"
                                    : selectedAction === 1
                                        ? " 位置Y较高需要向下调整，同时保持与障碍物的安全距离。"
                                        : " 位置X偏高，策略建议向左移动以优化路径。"}
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-green-100 dark:bg-green-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                💡 <strong>应用</strong>: 调试策略、建立信任、监管合规、发现偏差、知识蒸馏
            </div>
        </div>
    );
}
