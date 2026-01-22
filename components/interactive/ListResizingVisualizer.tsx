"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

type ListItem = {
    value: number;
    id: number;
};

export function ListResizingVisualizer() {
    const [items, setItems] = useState<ListItem[]>([
        { value: 1, id: 0 },
        { value: 2, id: 1 },
        { value: 3, id: 2 },
    ]);
    const [nextId, setNextId] = useState(3);
    const [capacity, setCapacity] = useState(4);
    const [history, setHistory] = useState<string[]>([
        "初始化: len=3, capacity=4"
    ]);

    // Python's list resizing formula (simplified)
    const calculateNewCapacity = (currentLength: number): number => {
        // CPython uses: new_allocated = (size_t)newsize + (newsize >> 3) + (newsize < 9 ? 3 : 6);
        // Simplified: round up to next power of 2 or use the pattern
        if (currentLength < 9) {
            return currentLength + 3;
        }
        return Math.ceil(currentLength + currentLength / 8 + 6);
    };

    const append = () => {
        const newItem = { value: nextId + 1, id: nextId };
        const newItems = [...items, newItem];

        let newCapacity = capacity;
        let resized = false;

        if (newItems.length > capacity) {
            newCapacity = calculateNewCapacity(newItems.length);
            resized = true;
            setHistory(prev => [
                ...prev,
                `🔄 扩容触发! len=${newItems.length} > capacity=${capacity}`,
                `   新容量 = ${newCapacity} (约 ${((newCapacity - newItems.length) / newItems.length * 100).toFixed(0)}% 预留空间)`
            ]);
        } else {
            setHistory(prev => [
                ...prev,
                `➕ 追加元素 ${newItem.value}: len=${newItems.length}, capacity=${capacity} (无需扩容)`
            ]);
        }

        setItems(newItems);
        setNextId(nextId + 1);
        setCapacity(newCapacity);
    };

    const pop = () => {
        if (items.length === 0) return;

        const newItems = items.slice(0, -1);
        setItems(newItems);
        setHistory(prev => [
            ...prev,
            `➖ 弹出元素: len=${newItems.length}, capacity=${capacity} (容量不缩减)`
        ]);
    };

    const reset = () => {
        setItems([{ value: 1, id: 0 }, { value: 2, id: 1 }, { value: 3, id: 2 }]);
        setNextId(3);
        setCapacity(4);
        setHistory(["初始化: len=3, capacity=4"]);
    };

    const utilizationRate = (items.length / capacity * 100).toFixed(1);
    const wastedSpace = capacity - items.length;

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-bg-elevated rounded-xl border border-border-subtle shadow-lg my-8">
            <h3 className="text-2xl font-bold text-center mb-6 text-text-primary">
                Python List 动态扩容可视化
            </h3>

            {/* Stats Panel */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="p-4 bg-bg-base border border-accent-primary/30 rounded-lg">
                    <div className="text-xs text-text-tertiary uppercase tracking-wider mb-1">Length (实际元素)</div>
                    <div className="text-3xl font-bold text-accent-primary">{items.length}</div>
                </div>
                <div className="p-4 bg-bg-base border border-accent-secondary/30 rounded-lg">
                    <div className="text-xs text-text-tertiary uppercase tracking-wider mb-1">Capacity (分配空间)</div>
                    <div className="text-3xl font-bold text-accent-secondary">{capacity}</div>
                </div>
                <div className="p-4 bg-bg-base border border-border-subtle rounded-lg">
                    <div className="text-xs text-text-tertiary uppercase tracking-wider mb-1">利用率</div>
                    <div className="text-3xl font-bold text-text-primary">{utilizationRate}%</div>
                    <div className="text-xs text-text-tertiary mt-1">浪费: {wastedSpace} slots</div>
                </div>
            </div>

            {/* Memory Visualization */}
            <div className="mb-6 p-6 bg-bg-base rounded-lg border border-border-subtle">
                <div className="text-sm font-semibold text-text-secondary mb-4">内存布局 (每个方块 = 1 slot)</div>
                <div className="flex flex-wrap gap-2">
                    <AnimatePresence>
                        {Array.from({ length: capacity }).map((_, idx) => {
                            const isOccupied = idx < items.length;
                            const item = isOccupied ? items[idx] : null;

                            return (
                                <motion.div
                                    key={`slot-${idx}`}
                                    initial={{ scale: 0, opacity: 0 }}
                                    animate={{ scale: 1, opacity: 1 }}
                                    exit={{ scale: 0, opacity: 0 }}
                                    transition={{ delay: idx * 0.02 }}
                                    className={`w-16 h-16 border-2 rounded-lg flex items-center justify-center font-bold text-lg transition-all ${isOccupied
                                            ? "bg-accent-primary/20 border-accent-primary text-accent-primary shadow-md"
                                            : "bg-bg-elevated border-border-subtle border-dashed text-text-tertiary"
                                        }`}
                                >
                                    {item ? item.value : "·"}
                                </motion.div>
                            );
                        })}
                    </AnimatePresence>
                </div>
            </div>

            {/* Controls */}
            <div className="flex gap-3 mb-6">
                <button
                    onClick={append}
                    className="flex-1 py-3 px-6 bg-green-500 hover:bg-green-600 text-white font-semibold rounded-lg transition-colors shadow-md"
                >
                    ➕ Append
                </button>
                <button
                    onClick={pop}
                    disabled={items.length === 0}
                    className="flex-1 py-3 px-6 bg-red-500 hover:bg-red-600 text-white font-semibold rounded-lg transition-colors shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    ➖ Pop
                </button>
                <button
                    onClick={reset}
                    className="py-3 px-6 bg-bg-base border border-border-subtle hover:bg-bg-elevated text-text-primary font-semibold rounded-lg transition-colors"
                >
                    🔄 重置
                </button>
            </div>

            {/* History Log */}
            <div className="p-4 bg-black/80 rounded-lg max-h-48 overflow-y-auto scrollbar-thin">
                <div className="text-xs text-green-400 mb-2 font-mono">操作历史:</div>
                <div className="space-y-1">
                    {history.slice(-10).map((log, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="text-xs font-mono text-green-300"
                        >
                            {log}
                        </motion.div>
                    ))}
                </div>
            </div>

            {/* Algorithm Explanation */}
            <div className="mt-6 p-4 bg-accent-primary/5 border border-accent-primary/20 rounded-lg">
                <div className="text-sm font-semibold text-accent-primary mb-2">📚 CPython 扩容公式</div>
                <div className="text-sm text-text-secondary font-mono">
                    new_capacity = length + (length &gt;&gt; 3) + (length &lt; 9 ? 3 : 6)
                </div>
                <div className="text-xs text-text-tertiary mt-2">
                    即：大约增加 12.5% (1/8) 的额外空间，确保平摊 O(1) 复杂度
                </div>
            </div>
        </div>
    );
}
