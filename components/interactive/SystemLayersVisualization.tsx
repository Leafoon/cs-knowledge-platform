"use client";

import { useState } from "react";
import { motion } from "framer-motion";

const systemLayers = [
    {
        id: 5,
        name: "第5层:应用语言机器 (M₅)",
        translator: "应用语言编译程序",
        description: "面向特定应用",
        color: "#8b5cf6",
        examples: ["数据库查询语言", "专用领域语言"],
    },
    {
        id: 4,
        name: "第4层:高级语言机器 (M₄)",
        translator: "高级语言编译/解释程序",
        description: "C、Java、Python等",
        color: "#3b82f6",
        examples: ["C/C++", "Java", "Python", "JavaScript"],
    },
    {
        id: 3,
        name: "第3层:汇编语言机器 (M₃)",
        translator: "汇编程序 (Assembler)",
        description: "ADD、SUB等助记符",
        color: "#10b981",
        examples: ["x86汇编", "ARM汇编", "MIPS汇编"],
    },
    {
        id: 2,
        name: "第2层:操作系统机器 (M₂)",
        translator: "由操作系统程序实现",
        description: "扩充指令功能",
        color: "#f59e0b",
        examples: ["系统调用", "设备驱动", "文件管理"],
    },
    {
        id: 1,
        name: "第1层:传统机器语言机器 (M₁)",
        translator: "由微程序解释机器指令",
        description: "机器指令系统",
        color: "#ef4444",
        examples: ["LOAD", "STORE", "ADD", "JMP"],
    },
    {
        id: 0,
        name: "第0层:微程序机器 (M₀)",
        translator: "由硬件直接执行微指令",
        description: "微指令系统",
        color: "#ec4899",
        examples: ["微操作", "控制信号", "硬件逻辑"],
    },
];

export function SystemLayersVisualization() {
    const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
    const [hoveredLayer, setHoveredLayer] = useState<number | null>(null);

    return (
        <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
            <h3 className="text-xl font-semibold mb-6 text-text-primary">
                计算机系统的层次结构 - 交互式模型
            </h3>

            {/* Layers Stack */}
            <div className="relative flex flex-col-reverse gap-2 mb-6">
                {systemLayers.map((layer, index) => {
                    const isSelected = selectedLayer === layer.id;
                    const isHovered = hoveredLayer === layer.id;
                    const isActive = isSelected || isHovered;

                    return (
                        <motion.div
                            key={layer.id}
                            className="relative"
                            onMouseEnter={() => setHoveredLayer(layer.id)}
                            onMouseLeave={() => setHoveredLayer(null)}
                            onClick={() => setSelectedLayer(isSelected ? null : layer.id)}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                        >
                            {/* Layer Card */}
                            <motion.div
                                className={`p-4 rounded-lg cursor-pointer transition-all ${isActive
                                    ? "shadow-lg"
                                    : "shadow-sm"
                                    }`}
                                style={{
                                    backgroundColor: isActive ? layer.color + "20" : "#f9fafb",
                                    borderLeft: `4px solid ${layer.color}`,
                                }}
                            >
                                <div className="flex items-center justify-between">
                                    <div className="flex-1">
                                        <div className="font-semibold text-text-primary mb-1">
                                            {layer.name}
                                        </div>
                                        <div className="text-sm text-text-secondary">
                                            翻译器: {layer.translator}
                                        </div>
                                        <div className="text-xs text-text-tertiary mt-1">
                                            ← {layer.description}
                                        </div>
                                    </div>
                                    <div
                                        className="w-12 h-12 rounded-full flex items-center justify-center text-white font-bold"
                                        style={{ backgroundColor: layer.color }}
                                    >
                                        M₍{layer.id}₎
                                    </div>
                                </div>

                                {/* Expanded Details */}
                                {isActive && (
                                    <motion.div
                                        initial={{ opacity: 0, height: 0 }}
                                        animate={{ opacity: 1, height: "auto" }}
                                        exit={{ opacity: 0, height: 0 }}
                                        className="mt-4 pt-4 border-t border-gray-300"
                                    >
                                        <div className="text-sm font-semibold text-text-primary mb-2">
                                            典型示例:
                                        </div>
                                        <div className="flex flex-wrap gap-2">
                                            {layer.examples.map((example, i) => (
                                                <span
                                                    key={i}
                                                    className="px-3 py-1 rounded-full text-xs font-medium text-white"
                                                    style={{ backgroundColor: layer.color }}
                                                >
                                                    {example}
                                                </span>
                                            ))}
                                        </div>
                                    </motion.div>
                                )}
                            </motion.div>

                            {/* Arrow Indicator */}
                            {index < systemLayers.length - 1 && (
                                <div className="flex justify-center my-1">
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                                        <path
                                            d="M12 5V19M12 19L5 12M12 19L19 12"
                                            stroke={layer.color}
                                            strokeWidth="2"
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                        />
                                    </svg>
                                </div>
                            )}
                        </motion.div>
                    );
                })}
            </div>

            {/* Info Box */}
            <div className="p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg border-l-4 border-accent-primary">
                <div className="text-sm text-text-secondary">
                    <strong className="text-text-primary">虚拟机概念:</strong>{" "}
                    如果把具有翻译功能的汇编程序的计算机看作一台机器M₂,那么可以认为M₂在M₁之上。
                    每一层都是一个虚拟机,通过下层的翻译器实现其功能。
                    <span className="block mt-2 text-accent-primary">
                        💡 点击任意层查看详细信息
                    </span>
                </div>
            </div>

            {/* Legend */}
            <div className="mt-6 grid grid-cols-2 md:grid-cols-3 gap-2">
                {systemLayers.map((layer) => (
                    <button
                        key={layer.id}
                        onClick={() => setSelectedLayer(layer.id === selectedLayer ? null : layer.id)}
                        className={`p-2 rounded-lg text-left text-sm transition ${selectedLayer === layer.id
                            ? "ring-2 ring-offset-2"
                            : "opacity-70 hover:opacity-100"
                            }`}
                        style={{
                            backgroundColor: layer.color + "20",
                            color: layer.color,
                            ['--tw-ring-color' as any]: layer.color,
                        }}
                    >
                        <div className="font-semibold">M₍{layer.id}₎</div>
                        <div className="text-xs opacity-75">第{layer.id}层</div>
                    </button>
                ))}
            </div>
        </div>
    );
}
