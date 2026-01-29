"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function MDPGraphVisualizer() {
    const [selectedState, setSelectedState] = useState<string | null>(null);
    const [showTransitions, setShowTransitions] = useState(true);

    // GridWorld 状态和转移
    const states = [
        { id: "S0", name: "起点", x: 50, y: 250, type: "start" },
        { id: "S1", name: "空地1", x: 200, y: 250, type: "normal" },
        { id: "S2", name: "空地2", x: 350, y: 250, type: "normal" },
        { id: "S3", name: "陷阱", x: 200, y: 100, type: "trap" },
        { id: "S4", name: "目标", x: 500, y: 250, type: "goal" },
    ];

    const transitions = [
        { from: "S0", to: "S1", action: "→", prob: 1.0, reward: -1 },
        { from: "S1", to: "S0", action: "←", prob: 1.0, reward: -1 },
        { from: "S1", to: "S2", action: "→", prob: 1.0, reward: -1 },
        { from: "S1", to: "S3", action: "↑", prob: 1.0, reward: -1 },
        { from: "S2", to: "S1", action: "←", prob: 1.0, reward: -1 },
        { from: "S2", to: "S4", action: "→", prob: 1.0, reward: +10 },
        { from: "S3", to: "S3", action: "*", prob: 1.0, reward: -10 },
        { from: "S4", to: "S4", action: "*", prob: 1.0, reward: 0 },
    ];

    const getStateColor = (type: string, isSelected: boolean) => {
        if (isSelected) return "#6366f1";
        switch (type) {
            case "start": return "#10b981";
            case "goal": return "#f59e0b";
            case "trap": return "#ef4444";
            default: return "#94a3b8";
        }
    };

    const getStateIcon = (type: string) => {
        switch (type) {
            case "start": return "🏁";
            case "goal": return "🎯";
            case "trap": return "💀";
            default: return "⬜";
        }
    };

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    MDP 状态转移图
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                    点击状态查看详细信息
                </p>
            </div>

            {/* 控制按钮 */}
            <div className="flex justify-center gap-4 mb-6">
                <button
                    onClick={() => setShowTransitions(!showTransitions)}
                    className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-semibold transition-colors text-sm"
                >
                    {showTransitions ? "隐藏转移" : "显示转移"}
                </button>
                <button
                    onClick={() => setSelectedState(null)}
                    className="px-4 py-2 rounded-lg bg-slate-600 hover:bg-slate-700 text-white font-semibold transition-colors text-sm"
                >
                    清除选择
                </button>
            </div>

            {/* MDP 图 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <svg width="600" height="400" className="mx-auto">
                    {/* 绘制转移边 */}
                    {showTransitions && transitions.map((trans, idx) => {
                        const fromState = states.find(s => s.id === trans.from);
                        const toState = states.find(s => s.id === trans.to);
                        if (!fromState || !toState) return null;

                        const isSelfLoop = trans.from === trans.to;

                        if (isSelfLoop) {
                            // 自环
                            return (
                                <g key={idx}>
                                    <circle
                                        cx={fromState.x}
                                        cy={fromState.y - 40}
                                        r="20"
                                        fill="none"
                                        stroke="#94a3b8"
                                        strokeWidth="2"
                                        strokeDasharray="5,5"
                                    />
                                    <text
                                        x={fromState.x}
                                        y={fromState.y - 60}
                                        textAnchor="middle"
                                        className="text-xs font-semibold fill-slate-600 dark:fill-slate-400"
                                    >
                                        r={trans.reward}
                                    </text>
                                </g>
                            );
                        }

                        const dx = toState.x - fromState.x;
                        const dy = toState.y - fromState.y;
                        const distance = Math.sqrt(dx * dx + dy * dy);
                        const offsetX = (dx / distance) * 35;
                        const offsetY = (dy / distance) * 35;

                        return (
                            <g key={idx}>
                                <defs>
                                    <marker
                                        id={`arrowhead-${idx}`}
                                        markerWidth="10"
                                        markerHeight="10"
                                        refX="9"
                                        refY="3"
                                        orient="auto"
                                    >
                                        <polygon
                                            points="0 0, 10 3, 0 6"
                                            fill="#94a3b8"
                                        />
                                    </marker>
                                </defs>
                                <line
                                    x1={fromState.x + offsetX}
                                    y1={fromState.y + offsetY}
                                    x2={toState.x - offsetX}
                                    y2={toState.y - offsetY}
                                    stroke="#94a3b8"
                                    strokeWidth="2"
                                    markerEnd={`url(#arrowhead-${idx})`}
                                />
                                <text
                                    x={(fromState.x + toState.x) / 2}
                                    y={(fromState.y + toState.y) / 2 - 10}
                                    textAnchor="middle"
                                    className="text-xs font-semibold fill-indigo-600"
                                >
                                    {trans.action}
                                </text>
                                <text
                                    x={(fromState.x + toState.x) / 2}
                                    y={(fromState.y + toState.y) / 2 + 5}
                                    textAnchor="middle"
                                    className="text-xs fill-slate-600 dark:fill-slate-400"
                                >
                                    r={trans.reward}
                                </text>
                            </g>
                        );
                    })}

                    {/* 绘制状态节点 */}
                    {states.map((state) => (
                        <g key={state.id}>
                            <motion.circle
                                cx={state.x}
                                cy={state.y}
                                r="30"
                                fill={getStateColor(state.type, selectedState === state.id)}
                                stroke="#1e293b"
                                strokeWidth="3"
                                onClick={() => setSelectedState(state.id)}
                                style={{ cursor: "pointer" }}
                                whileHover={{ scale: 1.1 }}
                                whileTap={{ scale: 0.95 }}
                            />
                            <text
                                x={state.x}
                                y={state.y + 5}
                                textAnchor="middle"
                                className="text-2xl pointer-events-none"
                            >
                                {getStateIcon(state.type)}
                            </text>
                            <text
                                x={state.x}
                                y={state.y + 50}
                                textAnchor="middle"
                                className="text-sm font-bold fill-slate-700 dark:fill-slate-300 pointer-events-none"
                            >
                                {state.id}
                            </text>
                        </g>
                    ))}
                </svg>
            </div>

            {/* 状态详情 */}
            {selectedState && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg"
                >
                    <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">
                        状态 {selectedState} 详情
                    </h4>
                    <div className="space-y-3">
                        <div className="flex justify-between items-center p-3 bg-slate-50 dark:bg-slate-700 rounded-lg">
                            <span className="text-sm font-semibold text-slate-600 dark:text-slate-400">
                                类型:
                            </span>
                            <span className="text-sm font-bold text-slate-800 dark:text-slate-100">
                                {states.find(s => s.id === selectedState)?.name}
                            </span>
                        </div>
                        <div className="p-3 bg-slate-50 dark:bg-slate-700 rounded-lg">
                            <div className="text-sm font-semibold text-slate-600 dark:text-slate-400 mb-2">
                                可能的转移:
                            </div>
                            {transitions
                                .filter(t => t.from === selectedState)
                                .map((trans, idx) => (
                                    <div
                                        key={idx}
                                        className="flex justify-between items-center py-2 border-b border-slate-200 dark:border-slate-600 last:border-0"
                                    >
                                        <span className="text-sm">
                                            动作 <span className="font-bold text-indigo-600">{trans.action}</span>
                                        </span>
                                        <span className="text-sm">
                                            → {trans.to} (r={trans.reward})
                                        </span>
                                    </div>
                                ))}
                        </div>
                    </div>
                </motion.div>
            )}

            {/* 图例 */}
            <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                    { type: "start", label: "起点", icon: "🏁" },
                    { type: "normal", label: "普通状态", icon: "⬜" },
                    { type: "trap", label: "陷阱", icon: "💀" },
                    { type: "goal", label: "目标", icon: "🎯" },
                ].map((item) => (
                    <div
                        key={item.type}
                        className="flex items-center gap-2 p-2 rounded-lg bg-white dark:bg-slate-800"
                    >
                        <div
                            className="w-6 h-6 rounded-full flex items-center justify-center text-sm"
                            style={{ backgroundColor: getStateColor(item.type, false) }}
                        >
                            {item.icon}
                        </div>
                        <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                            {item.label}
                        </span>
                    </div>
                ))}
            </div>

            <div className="mt-6 text-center text-sm text-slate-500 dark:text-slate-400">
                💡 提示：MDP 由状态、动作、转移概率和奖励函数定义
            </div>
        </div>
    );
}
