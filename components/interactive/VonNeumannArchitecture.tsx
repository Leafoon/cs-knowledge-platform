"use client";

import { useState } from "react";
import { motion } from "framer-motion";

type ArchitectureType = "alu-centered" | "memory-centered";

export function VonNeumannArchitecture() {
    const [architecture, setArchitecture] = useState<ArchitectureType>("alu-centered");
    const [highlightedComponent, setHighlightedComponent] = useState<string>("");

    const components = [
        { id: "cpu", name: "中央处理器 (CPU)", color: "#667eea" },
        { id: "alu", name: "运算器 (ALU)", color: "#f59e0b" },
        { id: "control", name: "控制器", color: "#10b981" },
        { id: "memory", name: "存储器", color: "#ef4444" },
        { id: "input", name: "输入设备", color: "#8b5cf6" },
        { id: "output", name: "输出设备", color: "#ec4899" },
    ];

    return (
        <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
            <h3 className="text-xl font-semibold mb-4 text-text-primary">
                冯·诺依曼架构演示
            </h3>

            {/* Architecture Toggle */}
            <div className="flex gap-3 mb-6">
                <button
                    onClick={() => setArchitecture("alu-centered")}
                    className={`flex-1 px-4 py-3 rounded-lg font-semibold transition ${architecture === "alu-centered"
                            ? "bg-accent-primary text-white"
                            : "bg-gray-200 dark:bg-gray-800 text-text-secondary"
                        }`}
                >
                    以运算器为中心
                </button>
                <button
                    onClick={() => setArchitecture("memory-centered")}
                    className={`flex-1 px-4 py-3 rounded-lg font-semibold transition ${architecture === "memory-centered"
                            ? "bg-accent-primary text-white"
                            : "bg-gray-200 dark:bg-gray-800 text-text-secondary"
                        }`}
                >
                    以存储器为中心
                </button>
            </div>

            {/* SVG Diagram */}
            <div className="relative w-full h-[500px] bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden mb-6">
                <svg
                    viewBox="0 0 900 500"
                    className="w-full h-full"
                    xmlns="http://www.w3.org/2000/svg"
                >
                    <defs>
                        {/* Arrow markers */}
                        <marker
                            id="arrow-alu"
                            markerWidth="10"
                            markerHeight="10"
                            refX="9"
                            refY="3"
                            orient="auto"
                        >
                            <polygon points="0 0, 10 3, 0 6" fill="#667eea" />
                        </marker>
                        <marker
                            id="arrow-memory"
                            markerWidth="10"
                            markerHeight="10"
                            refX="9"
                            refY="3"
                            orient="auto"
                        >
                            <polygon points="0 0, 10 3, 0 6" fill="#ef4444" />
                        </marker>
                    </defs>

                    {/* Central Component - ALU or Memory */}
                    {architecture === "alu-centered" ? (
                        <>
                            {/* ALU - Center */}
                            <motion.g
                                onMouseEnter={() => setHighlightedComponent("alu")}
                                onMouseLeave={() => setHighlightedComponent("")}
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ duration: 0.5 }}
                            >
                                <rect
                                    x="350"
                                    y="200"
                                    width="200"
                                    height="100"
                                    fill={highlightedComponent === "alu" ? "#fbbf24" : "#f59e0b"}
                                    stroke="#d97706"
                                    strokeWidth="3"
                                    rx="10"
                                />
                                <text
                                    x="450"
                                    y="240"
                                    textAnchor="middle"
                                    className="fill-white font-bold text-lg"
                                >
                                    运算器 (ALU)
                                </text>
                                <text
                                    x="450"
                                    y="265"
                                    textAnchor="middle"
                                    className="fill-white text-sm"
                                >
                                    数据中心
                                </text>
                            </motion.g>

                            {/* Arrows from all components to ALU */}
                            <motion.path
                                d="M 450 150 L 450 200"
                                stroke="#667eea"
                                strokeWidth="2"
                                fill="none"
                                markerEnd="url(#arrow-alu)"
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                                transition={{ duration: 0.8, delay: 0.2 }}
                            />
                            <motion.path
                                d="M 150 250 L 350 250"
                                stroke="#667eea"
                                strokeWidth="2"
                                fill="none"
                                markerEnd="url(#arrow-alu)"
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                                transition={{ duration: 0.8, delay: 0.3 }}
                            />
                            <motion.path
                                d="M 750 250 L 550 250"
                                stroke="#667eea"
                                strokeWidth="2"
                                fill="none"
                                markerEnd="url(#arrow-alu)"
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                                transition={{ duration: 0.8, delay: 0.4 }}
                            />
                            <motion.path
                                d="M 450 300 L 450 350"
                                stroke="#667eea"
                                strokeWidth="2"
                                fill="none"
                                markerEnd="url(#arrow-alu)"
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                                transition={{ duration: 0.8, delay: 0.5 }}
                            />
                        </>
                    ) : (
                        <>
                            {/* Memory - Center */}
                            <motion.g
                                onMouseEnter={() => setHighlightedComponent("memory")}
                                onMouseLeave={() => setHighlightedComponent("")}
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ duration: 0.5 }}
                            >
                                <rect
                                    x="350"
                                    y="200"
                                    width="200"
                                    height="100"
                                    fill={highlightedComponent === "memory" ? "#f87171" : "#ef4444"}
                                    stroke="#dc2626"
                                    strokeWidth="3"
                                    rx="10"
                                />
                                <text
                                    x="450"
                                    y="240"
                                    textAnchor="middle"
                                    className="fill-white font-bold text-lg"
                                >
                                    存储器
                                </text>
                                <text
                                    x="450"
                                    y="265"
                                    textAnchor="middle"
                                    className="fill-white text-sm"
                                >
                                    数据中心
                                </text>
                            </motion.g>

                            {/* Arrows - bidirectional */}
                            <motion.path
                                d="M 450 150 L 450 200"
                                stroke="#ef4444"
                                strokeWidth="2"
                                fill="none"
                                markerEnd="url(#arrow-memory)"
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                                transition={{ duration: 0.8, delay: 0.2 }}
                            />
                            <motion.path
                                d="M 150 250 L 350 250"
                                stroke="#ef4444"
                                strokeWidth="2"
                                fill="none"
                                markerEnd="url(#arrow-memory)"
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                                transition={{ duration: 0.8, delay: 0.3 }}
                            />
                            <motion.path
                                d="M 750 250 L 550 250"
                                stroke="#ef4444"
                                strokeWidth="2"
                                fill="none"
                                markerEnd="url(#arrow-memory)"
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                                transition={{ duration: 0.8, delay: 0.4 }}
                            />
                            <motion.path
                                d="M 450 300 L 450 350"
                                stroke="#ef4444"
                                strokeWidth="2"
                                fill="none"
                                markerEnd="url(#arrow-memory)"
                                initial={{ pathLength: 0 }}
                                animate={{ pathLength: 1 }}
                                transition={{ duration: 0.8, delay: 0.5 }}
                            />
                        </>
                    )}

                    {/* Control Unit - Top */}
                    <motion.g
                        onMouseEnter={() => setHighlightedComponent("control")}
                        onMouseLeave={() => setHighlightedComponent("")}
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.1 }}
                    >
                        <rect
                            x="350"
                            y="50"
                            width="200"
                            height="80"
                            fill={highlightedComponent === "control" ? "#34d399" : "#10b981"}
                            stroke="#059669"
                            strokeWidth="2"
                            rx="8"
                        />
                        <text
                            x="450"
                            y="85"
                            textAnchor="middle"
                            className="fill-white font-semibold"
                        >
                            控制器
                        </text>
                        <text
                            x="450"
                            y="105"
                            textAnchor="middle"
                            className="fill-white text-xs"
                        >
                            PC · IR · ID
                        </text>
                    </motion.g>

                    {/* Input Device - Left */}
                    <motion.g
                        onMouseEnter={() => setHighlightedComponent("input")}
                        onMouseLeave={() => setHighlightedComponent("")}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.5, delay: 0.2 }}
                    >
                        <rect
                            x="50"
                            y="200"
                            width="100"
                            height="100"
                            fill={highlightedComponent === "input" ? "#a78bfa" : "#8b5cf6"}
                            stroke="#7c3aed"
                            strokeWidth="2"
                            rx="8"
                        />
                        <text
                            x="100"
                            y="240"
                            textAnchor="middle"
                            className="fill-white font-semibold text-sm"
                        >
                            输入
                        </text>
                        <text
                            x="100"
                            y="260"
                            textAnchor="middle"
                            className="fill-white text-sm"
                        >
                            设备
                        </text>
                    </motion.g>

                    {/* Output Device - Right */}
                    <motion.g
                        onMouseEnter={() => setHighlightedComponent("output")}
                        onMouseLeave={() => setHighlightedComponent("")}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.5, delay: 0.3 }}
                    >
                        <rect
                            x="750"
                            y="200"
                            width="100"
                            height="100"
                            fill={highlightedComponent === "output" ? "#f472b6" : "#ec4899"}
                            stroke="#db2777"
                            strokeWidth="2"
                            rx="8"
                        />
                        <text
                            x="800"
                            y="240"
                            textAnchor="middle"
                            className="fill-white font-semibold text-sm"
                        >
                            输出
                        </text>
                        <text
                            x="800"
                            y="260"
                            textAnchor="middle"
                            className="fill-white text-sm"
                        >
                            设备
                        </text>
                    </motion.g>

                    {/* CPU/ALU - Bottom (when memory centered) */}
                    {architecture === "memory-centered" && (
                        <motion.g
                            onMouseEnter={() => setHighlightedComponent("alu")}
                            onMouseLeave={() => setHighlightedComponent("")}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5, delay: 0.4 }}
                        >
                            <rect
                                x="350"
                                y="370"
                                width="200"
                                height="80"
                                fill={highlightedComponent === "alu" ? "#fbbf24" : "#f59e0b"}
                                stroke="#d97706"
                                strokeWidth="2"
                                rx="8"
                            />
                            <text
                                x="450"
                                y="405"
                                textAnchor="middle"
                                className="fill-white font-semibold"
                            >
                                运算器 (ALU)
                            </text>
                            <text
                                x="450"
                                y="425"
                                textAnchor="middle"
                                className="fill-white text-xs"
                            >
                                ACC · MQ · X
                            </text>
                        </motion.g>
                    )}

                    {/* Memory - Bottom (when ALU centered) */}
                    {architecture === "alu-centered" && (
                        <motion.g
                            onMouseEnter={() => setHighlightedComponent("memory")}
                            onMouseLeave={() => setHighlightedComponent("")}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5, delay: 0.4 }}
                        >
                            <rect
                                x="350"
                                y="370"
                                width="200"
                                height="80"
                                fill={highlightedComponent === "memory" ? "#f87171" : "#ef4444"}
                                stroke="#dc2626"
                                strokeWidth="2"
                                rx="8"
                            />
                            <text
                                x="450"
                                y="405"
                                textAnchor="middle"
                                className="fill-white font-semibold"
                            >
                                存储器
                            </text>
                            <text
                                x="450"
                                y="425"
                                textAnchor="middle"
                                className="fill-white text-xs"
                            >
                                MAR · MDR
                            </text>
                        </motion.g>
                    )}
                </svg>
            </div>

            {/* Legend */}
            <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
                {components.map((comp) => (
                    <div
                        key={comp.id}
                        className={`p-2 rounded-lg border-2 transition ${highlightedComponent === comp.id
                                ? "border-accent-primary scale-105"
                                : "border-transparent"
                            }`}
                        onMouseEnter={() => setHighlightedComponent(comp.id)}
                        onMouseLeave={() => setHighlightedComponent("")}
                    >
                        <div
                            className="w-full h-6 rounded mb-2"
                            style={{ backgroundColor: comp.color }}
                        />
                        <div className="text-xs text-text-secondary text-center">
                            {comp.name}
                        </div>
                    </div>
                ))}
            </div>

            {/* Description */}
            <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg border-l-4 border-accent-primary">
                <div className="text-sm text-text-secondary">
                    {architecture === "alu-centered" ? (
                        <>
                            <strong className="text-text-primary">以运算器为中心:</strong>{" "}
                            所有数据传送都要经过运算器,运算器成为数据流动的中心枢纽。缺点是运算器负担重,系统效率低。
                        </>
                    ) : (
                        <>
                            <strong className="text-text-primary">以存储器为中心:</strong>{" "}
                            存储器成为数据交换的中心,I/O设备可以直接与存储器交换数据(DMA方式),减轻了CPU的负担,提高了系统效率。
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
