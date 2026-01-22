"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const computerGenerations = [
    {
        id: 1,
        name: "第一代",
        period: "1946-1958",
        component: "电子管",
        speed: "几千~几万次/秒",
        size: "占地几十至几百平方米",
        power: "极高(约150kW)",
        cost: "极高",
        example: "ENIAC, EDVAC",
        color: "#ef4444",
        transistors: "18,000个电子管",
    },
    {
        id: 2,
        name: "第二代",
        period: "1958-1964",
        component: "晶体管",
        speed: "几十万~几百万次/秒",
        size: "大幅缩小",
        power: "较高",
        cost: "高",
        example: "IBM 7090",
        color: "#f59e0b",
        transistors: "数千个晶体管",
    },
    {
        id: 3,
        name: "第三代",
        period: "1964-1971",
        component: "中小规模IC",
        speed: "几百万~几千万次/秒",
        size: "进一步缩小",
        power: "中等",
        cost: "中等",
        example: "IBM System/360",
        color: "#10b981",
        transistors: "几十~几百晶体管/芯片",
    },
    {
        id: 4,
        name: "第四代",
        period: "1971-至今",
        component: "LSI/VLSI",
        speed: "几亿~万亿次/秒",
        size: "微型化",
        power: "低",
        cost: "低",
        example: "Intel Core, AMD Ryzen",
        color: "#667eea",
        transistors: "数十亿晶体管/芯片",
    },
];

export function ComputerEvolutionTimeline() {
    const [selectedGen, setSelectedGen] = useState<number>(0);

    return (
        <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
            <h3 className="text-xl font-semibold mb-6 text-text-primary">
                计算机发展历程 - 交互式时间线
            </h3>

            {/* Timeline */}
            <div className="relative mb-12">
                {/* Background Line */}
                <div className="absolute top-8 left-0 right-0 h-1 bg-gray-200 dark:bg-gray-700" />

                {/* Progress Line */}
                <motion.div
                    className="absolute top-8 left-0 h-1 bg-gradient-to-r from-red-500 via-orange-500 via-green-500 to-purple-600"
                    initial={{ width: "0%" }}
                    animate={{ width: `${((selectedGen + 1) / computerGenerations.length) * 100}%` }}
                    transition={{ duration: 0.5 }}
                />

                {/* Timeline Points */}
                <div className="relative flex justify-between">
                    {computerGenerations.map((gen, index) => (
                        <button
                            key={gen.id}
                            onClick={() => setSelectedGen(index)}
                            className="flex flex-col items-center group"
                        >
                            <motion.div
                                className={`w-16 h-16 rounded-full border-4 ${index === selectedGen
                                        ? "border-white shadow-lg scale-110"
                                        : "border-gray-300 dark:border-gray-600"
                                    } flex items-center justify-center cursor-pointer transition-all`}
                                style={{
                                    backgroundColor: index <= selectedGen ? gen.color : "#e5e7eb",
                                }}
                                whileHover={{ scale: 1.1 }}
                                whileTap={{ scale: 0.95 }}
                            >
                                <span className="text-white font-bold text-sm">{gen.id}</span>
                            </motion.div>

                            <div className="mt-3 text-center">
                                <div
                                    className={`text-sm font-semibold ${index === selectedGen ? "text-text-primary" : "text-text-tertiary"
                                        }`}
                                >
                                    {gen.name}
                                </div>
                                <div className="text-xs text-text-tertiary mt-1">{gen.period}</div>
                            </div>
                        </button>
                    ))}
                </div>
            </div>

            {/* Detail Card */}
            <AnimatePresence mode="wait">
                <motion.div
                    key={selectedGen}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="p-6 rounded-lg border-2"
                    style={{
                        borderColor: computerGenerations[selectedGen].color,
                        background: `linear-gradient(135deg, ${computerGenerations[selectedGen].color}15 0%, transparent 100%)`,
                    }}
                >
                    <div className="flex items-start justify-between mb-4">
                        <div>
                            <h4 className="text-2xl font-bold text-text-primary mb-1">
                                {computerGenerations[selectedGen].name}计算机
                            </h4>
                            <p className="text-text-secondary">
                                {computerGenerations[selectedGen].period}
                            </p>
                        </div>
                        <div
                            className="px-4 py-2 rounded-full text-white font-semibold"
                            style={{ backgroundColor: computerGenerations[selectedGen].color }}
                        >
                            {computerGenerations[selectedGen].component}
                        </div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
                        <div className="p-3 bg-white dark:bg-gray-800 rounded-lg">
                            <div className="text-xs text-text-tertiary mb-1">运算速度</div>
                            <div className="text-sm font-semibold text-text-primary">
                                {computerGenerations[selectedGen].speed}
                            </div>
                        </div>
                        <div className="p-3 bg-white dark:bg-gray-800 rounded-lg">
                            <div className="text-xs text-text-tertiary mb-1">体积</div>
                            <div className="text-sm font-semibold text-text-primary">
                                {computerGenerations[selectedGen].size}
                            </div>
                        </div>
                        <div className="p-3 bg-white dark:bg-gray-800 rounded-lg">
                            <div className="text-xs text-text-tertiary mb-1">功耗</div>
                            <div className="text-sm font-semibold text-text-primary">
                                {computerGenerations[selectedGen].power}
                            </div>
                        </div>
                        <div className="p-3 bg-white dark:bg-gray-800 rounded-lg">
                            <div className="text-xs text-text-tertiary mb-1">成本</div>
                            <div className="text-sm font-semibold text-text-primary">
                                {computerGenerations[selectedGen].cost}
                            </div>
                        </div>
                        <div className="p-3 bg-white dark:bg-gray-800 rounded-lg">
                            <div className="text-xs text-text-tertiary mb-1">集成度</div>
                            <div className="text-sm font-semibold text-text-primary">
                                {computerGenerations[selectedGen].transistors}
                            </div>
                        </div>
                        <div className="p-3 bg-white dark:bg-gray-800 rounded-lg">
                            <div className="text-xs text-text-tertiary mb-1">代表机型</div>
                            <div className="text-sm font-semibold text-text-primary">
                                {computerGenerations[selectedGen].example}
                            </div>
                        </div>
                    </div>

                    {/* Comparison Bar Chart */}
                    <div className="mt-6">
                        <div className="text-sm font-semibold text-text-primary mb-3">
                            性能对比(相对第一代)
                        </div>
                        <div className="space-y-3">
                            <div>
                                <div className="flex justify-between text-xs text-text-secondary mb-1">
                                    <span>运算速度</span>
                                    <span>{Math.pow(10, selectedGen + 1)}x</span>
                                </div>
                                <div className="w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                    <motion.div
                                        className="h-full rounded-full"
                                        style={{ backgroundColor: computerGenerations[selectedGen].color }}
                                        initial={{ width: "0%" }}
                                        animate={{ width: `${((selectedGen + 1) / 4) * 100}%` }}
                                        transition={{ duration: 0.8, delay: 0.1 }}
                                    />
                                </div>
                            </div>
                            <div>
                                <div className="flex justify-between text-xs text-text-secondary mb-1">
                                    <span>集成度</span>
                                    <span>{Math.pow(100, selectedGen + 1)}x</span>
                                </div>
                                <div className="w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                    <motion.div
                                        className="h-full rounded-full"
                                        style={{ backgroundColor: computerGenerations[selectedGen].color }}
                                        initial={{ width: "0%" }}
                                        animate={{ width: `${((selectedGen + 1) / 4) * 100}%` }}
                                        transition={{ duration: 0.8, delay: 0.2 }}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                </motion.div>
            </AnimatePresence>

            {/* Navigation Hint */}
            <div className="mt-6 text-center text-sm text-text-tertiary">
                点击时间线上的节点可查看不同代计算机的详细信息
            </div>
        </div>
    );
}
