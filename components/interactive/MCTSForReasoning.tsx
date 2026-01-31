"use client";

import { useState } from "react";
import { motion } from "framer-motion";

interface MCTSNode {
    id: string;
    step: string;
    visits: number;
    value: number;
    children: string[];
    isTerminal: boolean;
}

export function MCTSForReasoning() {
    const [iteration, setIteration] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    const maxIterations = 8;

    // MCTS树结构（简化示例）
    const treeStates: Record<number, Record<string, MCTSNode>> = {
        0: {
            "root": { id: "root", step: "问题：2x + 3 = 11", visits: 1, value: 0, children: [], isTerminal: false }
        },
        1: {
            "root": { id: "root", step: "问题", visits: 2, value: 0.5, children: ["a1"], isTerminal: false },
            "a1": { id: "a1", step: "2x = 11 - 3", visits: 1, value: 0.9, children: [], isTerminal: false }
        },
        2: {
            "root": { id: "root", step: "问题", visits: 3, value: 0.7, children: ["a1", "a2"], isTerminal: false },
            "a1": { id: "a1", step: "2x = 11 - 3", visits: 1, value: 0.9, children: [], isTerminal: false },
            "a2": { id: "a2", step: "x = (11-3)/2", visits: 1, value: 0.3, children: [], isTerminal: false }
        },
        3: {
            "root": { id: "root", step: "问题", visits: 4, value: 0.8, children: ["a1", "a2"], isTerminal: false },
            "a1": { id: "a1", step: "2x = 11 - 3", visits: 2, value: 0.95, children: ["b1"], isTerminal: false },
            "a2": { id: "a2", step: "x = (11-3)/2", visits: 1, value: 0.3, children: [], isTerminal: false },
            "b1": { id: "b1", step: "2x = 8", visits: 1, value: 0.95, children: [], isTerminal: false }
        },
        4: {
            "root": { id: "root", step: "问题", visits: 5, value: 0.85, children: ["a1", "a2"], isTerminal: false },
            "a1": { id: "a1", step: "2x = 11 - 3", visits: 3, value: 0.96, children: ["b1"], isTerminal: false },
            "a2": { id: "a2", step: "x = (11-3)/2", visits: 1, value: 0.3, children: [], isTerminal: false },
            "b1": { id: "b1", step: "2x = 8", visits: 2, value: 0.96, children: ["c1"], isTerminal: false },
            "c1": { id: "c1", step: "x = 4", visits: 1, value: 1.0, children: [], isTerminal: true }
        },
        5: {
            "root": { id: "root", step: "问题", visits: 6, value: 0.88, children: ["a1", "a2"], isTerminal: false },
            "a1": { id: "a1", step: "2x = 11 - 3", visits: 4, value: 0.97, children: ["b1"], isTerminal: false },
            "a2": { id: "a2", step: "x = (11-3)/2", visits: 1, value: 0.3, children: [], isTerminal: false },
            "b1": { id: "b1", step: "2x = 8", visits: 3, value: 0.97, children: ["c1"], isTerminal: false },
            "c1": { id: "c1", step: "x = 4", visits: 2, value: 1.0, children: [], isTerminal: true }
        },
        6: {
            "root": { id: "root", step: "问题", visits: 7, value: 0.90, children: ["a1", "a2"], isTerminal: false },
            "a1": { id: "a1", step: "2x = 11 - 3", visits: 5, value: 0.98, children: ["b1"], isTerminal: false },
            "a2": { id: "a2", step: "x = (11-3)/2", visits: 1, value: 0.3, children: [], isTerminal: false },
            "b1": { id: "b1", step: "2x = 8", visits: 4, value: 0.98, children: ["c1"], isTerminal: false },
            "c1": { id: "c1", step: "x = 4", visits: 3, value: 1.0, children: [], isTerminal: true }
        },
        7: {
            "root": { id: "root", step: "问题", visits: 8, value: 0.92, children: ["a1", "a2"], isTerminal: false },
            "a1": { id: "a1", step: "2x = 11 - 3", visits: 6, value: 0.98, children: ["b1"], isTerminal: false },
            "a2": { id: "a2", step: "x = (11-3)/2", visits: 1, value: 0.3, children: [], isTerminal: false },
            "b1": { id: "b1", step: "2x = 8", visits: 5, value: 0.98, children: ["c1"], isTerminal: false },
            "c1": { id: "c1", step: "x = 4", visits: 4, value: 1.0, children: [], isTerminal: true }
        },
        8: {
            "root": { id: "root", step: "问题", visits: 9, value: 0.93, children: ["a1", "a2"], isTerminal: false },
            "a1": { id: "a1", step: "2x = 11 - 3", visits: 7, value: 0.99, children: ["b1"], isTerminal: false },
            "a2": { id: "a2", step: "x = (11-3)/2", visits: 1, value: 0.3, children: [], isTerminal: false },
            "b1": { id: "b1", step: "2x = 8", visits: 6, value: 0.99, children: ["c1"], isTerminal: false },
            "c1": { id: "c1", step: "x = 4", visits: 5, value: 1.0, children: [], isTerminal: true }
        }
    };

    const currentTree = treeStates[Math.min(iteration, maxIterations)];

    const handlePlay = () => {
        if (isPlaying) {
            setIsPlaying(false);
        } else {
            setIsPlaying(true);
            const interval = setInterval(() => {
                setIteration(prev => {
                    if (prev >= maxIterations) {
                        setIsPlaying(false);
                        clearInterval(interval);
                        return prev;
                    }
                    return prev + 1;
                });
            }, 1000);
        }
    };

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-violet-50 to-purple-50 dark:from-slate-900 dark:to-violet-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    MCTS for Reasoning
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    用蒙特卡洛树搜索探索推理空间
                </p>
            </div>

            {/* 控制 */}
            <div className="flex items-center justify-between mb-6">
                <div className="flex gap-3">
                    <button
                        onClick={handlePlay}
                        className={`px-4 py-2 rounded-lg font-semibold transition ${isPlaying ? "bg-orange-500 text-white" : "bg-violet-600 text-white hover:bg-violet-700"
                            }`}
                    >
                        {isPlaying ? "⏸ 暂停" : "▶ 开始"}
                    </button>
                    <button
                        onClick={() => { setIsPlaying(false); setIteration(0); }}
                        className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-semibold hover:bg-gray-300 dark:hover:bg-gray-600 transition"
                    >
                        🔄 重置
                    </button>
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400">
                    迭代: <strong className="text-violet-600 dark:text-violet-400">{iteration}/{maxIterations}</strong>
                </div>
            </div>

            {/* MCTS阶段 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">MCTS四个阶段</h4>

                <div className="grid grid-cols-4 gap-3">
                    {[
                        { name: "1. Selection", desc: "UCB选择", icon: "🎯", color: "blue" },
                        { name: "2. Expansion", desc: "扩展节点", icon: "🌱", color: "green" },
                        { name: "3. Simulation", desc: "评估价值", icon: "🎲", color: "purple" },
                        { name: "4. Backprop", desc: "回传更新", icon: "⬆️", color: "orange" }
                    ].map((phase) => (
                        <div key={phase.name} className={`p-3 rounded-lg bg-${phase.color}-50 dark:bg-${phase.color}-900/20 border border-${phase.color}-300 dark:border-${phase.color}-700`}>
                            <div className="text-2xl text-center mb-1">{phase.icon}</div>
                            <div className="text-xs font-semibold text-center text-slate-800 dark:text-slate-100">
                                {phase.name}
                            </div>
                            <div className="text-xs text-center text-slate-600 dark:text-slate-400 mt-1">
                                {phase.desc}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* 树可视化 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">搜索树</h4>

                <div className="relative min-h-[400px] bg-gray-50 dark:bg-gray-900 rounded-lg p-8">
                    {/* SVG树形图 */}
                    <svg className="w-full h-full" viewBox="0 0 800 400">
                        {/* 边 */}
                        {Object.values(currentTree).map((node) =>
                            node.children.map((childId) => {
                                const child = currentTree[childId];
                                if (!child) return null;

                                const parentX = node.id === "root" ? 400 :
                                    node.id.startsWith("a") ? (node.id === "a1" ? 250 : 550) :
                                        node.id === "b1" ? 250 : 550;
                                const parentY = node.id === "root" ? 50 :
                                    node.id.startsWith("a") ? 150 : 250;

                                const childX = child.id === "a1" ? 250 :
                                    child.id === "a2" ? 550 :
                                        child.id === "b1" ? 250 :
                                            child.id === "c1" ? 250 : 400;
                                const childY = child.id.startsWith("a") ? 150 :
                                    child.id.startsWith("b") ? 250 : 350;

                                return (
                                    <line
                                        key={`${node.id}-${childId}`}
                                        x1={parentX}
                                        y1={parentY}
                                        x2={childX}
                                        y2={childY}
                                        stroke="#9ca3af"
                                        strokeWidth="2"
                                    />
                                );
                            })
                        )}

                        {/* 节点 */}
                        {Object.values(currentTree).map((node) => {
                            const x = node.id === "root" ? 400 :
                                node.id === "a1" ? 250 :
                                    node.id === "a2" ? 550 :
                                        node.id === "b1" ? 250 :
                                            node.id === "c1" ? 250 : 400;
                            const y = node.id === "root" ? 50 :
                                node.id.startsWith("a") ? 150 :
                                    node.id.startsWith("b") ? 250 : 350;

                            const nodeColor = node.isTerminal ? "#22c55e" :
                                node.value > 0.9 ? "#8b5cf6" :
                                    node.value > 0.5 ? "#3b82f6" : "#94a3b8";

                            return (
                                <g key={node.id}>
                                    <circle
                                        cx={x}
                                        cy={y}
                                        r={30}
                                        fill={nodeColor}
                                        stroke="white"
                                        strokeWidth="3"
                                    />
                                    <text
                                        x={x}
                                        y={y - 5}
                                        textAnchor="middle"
                                        fill="white"
                                        fontSize="12"
                                        fontWeight="bold"
                                    >
                                        V:{node.visits}
                                    </text>
                                    <text
                                        x={x}
                                        y={y + 10}
                                        textAnchor="middle"
                                        fill="white"
                                        fontSize="10"
                                    >
                                        {(node.value * 100).toFixed(0)}%
                                    </text>
                                </g>
                            );
                        })}
                    </svg>
                </div>

                <div className="mt-4 grid grid-cols-4 gap-2 text-xs">
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded-full bg-gray-400"></div>
                        <span>低价值</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded-full bg-blue-600"></div>
                        <span>中价值</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded-full bg-purple-600"></div>
                        <span>高价值</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded-full bg-green-600"></div>
                        <span>终止节点</span>
                    </div>
                </div>
            </div>

            {/* 当前最佳路径 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">当前最佳路径</h4>

                <div className="space-y-2">
                    {["root", "a1", "b1", "c1"].map((nodeId, idx) => {
                        const node = currentTree[nodeId];
                        if (!node) return null;

                        return (
                            <div
                                key={nodeId}
                                className={`p-3 rounded-lg ${node.isTerminal
                                        ? "bg-green-50 dark:bg-green-900/20 border-2 border-green-500"
                                        : "bg-purple-50 dark:bg-purple-900/20 border border-purple-300 dark:border-purple-700"
                                    }`}
                            >
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <div className="w-8 h-8 rounded-full bg-purple-600 text-white flex items-center justify-center text-sm font-bold">
                                            {idx + 1}
                                        </div>
                                        <div className="text-slate-800 dark:text-slate-100">
                                            {node.step}
                                        </div>
                                    </div>
                                    <div className="text-sm text-slate-600 dark:text-slate-400">
                                        访问: {node.visits}, 价值: {(node.value * 100).toFixed(0)}%
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            <div className="mt-6 bg-violet-100 dark:bg-violet-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                💡 MCTS通过平衡探索与利用，逐步收敛到最优推理路径
            </div>
        </div>
    );
}
