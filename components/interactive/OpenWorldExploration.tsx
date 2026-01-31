"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export function OpenWorldExploration() {
    const [unlockedNodes, setUnlockedNodes] = useState<string[]>(["wood"]);

    // Simple Tech Tree
    const nodes = [
        { id: "wood", name: "Chop Wood", x: 50, y: 50, icon: "🪵", requires: [] },
        { id: "plank", name: "Craft Plank", x: 150, y: 50, icon: "🧱", requires: ["wood"] },
        { id: "stick", name: "Craft Stick", x: 150, y: 120, icon: "🥢", requires: ["wood"] },
        { id: "pickaxe", name: "Craft Pickaxe", x: 250, y: 85, icon: "⛏️", requires: ["plank", "stick"] },
        { id: "stone", name: "Mine Stone", x: 350, y: 85, icon: "🪨", requires: ["pickaxe"] },
        { id: "furnace", name: "Build Furnace", x: 450, y: 50, icon: "🔥", requires: ["stone"] },
        { id: "iron", name: "Smelt Iron", x: 550, y: 50, icon: "⛓️", requires: ["furnace", "stone"] },
        { id: "diamond", name: "Mine Diamond", x: 450, y: 150, icon: "💎", requires: ["pickaxe", "stone"] }
    ];

    const handleUnlock = (id: string) => {
        const node = nodes.find(n => n.id === id);
        if (!node) return;

        // Check requirements
        const canUnlock = node.requires.every(req => unlockedNodes.includes(req));

        if (canUnlock && !unlockedNodes.includes(id)) {
            setUnlockedNodes(prev => [...prev, id]);
        }
    };

    return (
        <div className="w-full max-w-4xl mx-auto p-6 bg-slate-900 rounded-2xl shadow-xl overflow-hidden relative min-h-[400px]">
            <div className="absolute top-0 left-0 right-0 p-4 bg-slate-900/90 backdrop-blur z-20 border-b border-slate-700 flex justify-between items-center">
                <div>
                    <h3 className="text-lg font-bold text-white flex items-center gap-2">
                        <span className="text-green-400">MineDojo</span> Tech Tree
                    </h3>
                    <p className="text-xs text-slate-400">Open-World Skill Discovery Visualization</p>
                </div>
                <div className="text-right">
                    <div className="text-2xl font-bold text-yellow-400">{unlockedNodes.length}/{nodes.length}</div>
                    <div className="text-xs text-slate-500">Skills Unlocked</div>
                </div>
            </div>

            {/* Tree Visualization with SVG connection lines */}
            <div className="absolute inset-0 pt-20 overflow-auto">
                <svg className="w-full h-full absolute inset-0 pointer-events-none">
                    {nodes.map(node => (
                        node.requires.map(reqId => {
                            const reqNode = nodes.find(n => n.id === reqId)!;
                            // Naive scaling for demo purpose
                            const x1 = reqNode.x * 1.5;
                            const y1 = reqNode.y * 1.5 + 20;
                            const x2 = node.x * 1.5;
                            const y2 = node.y * 1.5 + 20;

                            const isUnlocked = unlockedNodes.includes(node.id) && unlockedNodes.includes(reqId);

                            return (
                                <motion.line
                                    key={`${reqId}-${node.id}`}
                                    initial={{ pathLength: 0 }}
                                    animate={{ pathLength: 1 }}
                                    x1={x1 + 40} y1={y1 + 40} // Center offset approx
                                    x2={x2 + 40} y2={y2 + 40}
                                    stroke={isUnlocked ? "#22c55e" : "#475569"}
                                    strokeWidth={isUnlocked ? 4 : 2}
                                />
                            );
                        })
                    ))}
                </svg>

                <div className="relative w-full h-full">
                    {nodes.map(node => {
                        const isUnlocked = unlockedNodes.includes(node.id);
                        const isUnlockable = !isUnlocked && node.requires.every(req => unlockedNodes.includes(req));

                        return (
                            <motion.button
                                key={node.id}
                                whileHover={{ scale: 1.1 }}
                                onClick={() => handleUnlock(node.id)}
                                disabled={!isUnlockable && !isUnlocked}
                                className={`absolute w-20 h-20 rounded-xl flex flex-col items-center justify-center gap-1 border-2 transition-all cursor-pointer z-10 ${isUnlocked
                                        ? "bg-slate-800 border-green-500 text-white shadow-[0_0_15px_rgba(34,197,94,0.4)]"
                                        : isUnlockable
                                            ? "bg-slate-800/50 border-yellow-500/50 text-slate-300 animate-pulse"
                                            : "bg-slate-900 border-slate-700 text-slate-600 opacity-50 cursor-not-allowed"
                                    }`}
                                style={{
                                    left: node.x * 1.5,
                                    top: node.y * 1.5 + 20
                                }}
                            >
                                <span className="text-3xl">{node.icon}</span>
                                <span className="text-[10px] font-bold">{node.name}</span>
                            </motion.button>
                        );
                    })}
                </div>
            </div>

            <div className="absolute bottom-4 right-4 text-xs text-slate-500 max-w-xs text-right">
                Click on highlighted nodes to simulate agent discovering new skills.
                In Open-World RL, agents must explore this graph automatically.
            </div>
        </div>
    );
}
