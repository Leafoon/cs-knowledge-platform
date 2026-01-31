"use client";

import { motion } from "framer-motion";
import { useState } from "react";

// Helper to generate SVG paths
const getPath = (x1: number, y1: number, x2: number, y2: number, curvature = 0) => {
    if (curvature === 0) return `M ${x1} ${y1} L ${x2} ${y2}`;

    // Calculate midpoint with offset for quadratic bezier
    const midX = (x1 + x2) / 2;
    const midY = (y1 + y2) / 2;

    // Normal vector (-dy, dx)
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);

    if (len === 0) return ""; // Should handle self-loop differently if needed

    const offsetX = -dy * (curvature / len);
    const offsetY = dx * (curvature / len);

    const cX = midX + offsetX;
    const cY = midY + offsetY;

    return `M ${x1} ${y1} Q ${cX} ${cY} ${x2} ${y2}`;
};

const getSelfLoopPath = (x: number, y: number, size = 40, direction: 'top' | 'left' | 'right' | 'bottom' = 'top') => {
    // A simple cubic bezier loop
    const offset = size;
    switch (direction) {
        case 'left':
            return `M ${x - 20} ${y} C ${x - offset} ${y - 20}, ${x - offset} ${y + 20}, ${x - 20} ${y}`;
        case 'top':
        default:
            return `M ${x} ${y - 20} C ${x - 20} ${y - offset}, ${x + 20} ${y - offset}, ${x} ${y - 20}`;
    }
};

// Define types
interface Edge {
    from: string;
    to: string;
    label: string;
    straight?: boolean;
    dashed?: boolean;
    curvature?: number;
    loop?: 'top' | 'left' | 'right' | 'bottom';
    labelIdx?: number;
    labelOff?: { x: number; y: number };
}

export function MDPGraphVisualizer() {
    const [activeNode, setActiveNode] = useState<string | null>(null);

    // Optimized Coordinates for Expanded Layout (600x480 Canvas)
    const nodes = [
        // States
        { id: "fb", label: "Facebook", x: 60, y: 240, type: "state", color: "indigo" },
        { id: "class", label: "Class", x: 300, y: 240, type: "state", color: "blue" },
        { id: "pass", label: "Pass", x: 540, y: 240, type: "term", color: "emerald" },
        { id: "pub", label: "Pub", x: 300, y: 420, type: "state", color: "amber" },
        { id: "sleep", label: "Sleep", x: 300, y: 60, type: "term", color: "slate" },

        // Actions
        { id: "act_fb", label: "F", x: 180, y: 240, type: "action", color: "slate" },      // Class -> FB
        { id: "act_quit", label: "Q", x: 180, y: 160, type: "action", color: "slate" },    // FB -> Class
        { id: "act_study", label: "S", x: 420, y: 240, type: "action", color: "slate" },   // Class -> Study
        { id: "act_sleep", label: "Z", x: 300, y: 150, type: "action", color: "slate" },   // Class -> Sleep
        { id: "act_cont", label: "C", x: 40, y: 160, type: "action", color: "slate" },     // FB -> FB (moved up-left)
        { id: "act_pub", label: "P", x: 300, y: 340, type: "action", color: "slate" },     // Pub -> Class (Loop)
    ];

    const edges: Edge[] = [
        // CLASS Outgoing
        { from: "class", to: "act_fb", label: "facebook (-1)", straight: true, labelIdx: 0.5, labelOff: { x: 0, y: -20 } },
        { from: "act_fb", to: "fb", label: "", straight: true, dashed: true },

        { from: "class", to: "act_study", label: "study (-2)", straight: true, labelOff: { x: 0, y: -20 } },
        { from: "act_study", to: "pass", label: "0.6 (+10)", straight: true, dashed: true, labelOff: { x: 0, y: -20 } },
        { from: "act_study", to: "pub", label: "0.4 (+1)", curvature: -40, dashed: true, labelOff: { x: 35, y: 0 } },

        { from: "class", to: "act_sleep", label: "sleep (0)", straight: true, labelOff: { x: 25, y: 0 } },
        { from: "act_sleep", to: "sleep", label: "1.0", straight: true, dashed: true },

        // FACEBOOK Outgoing
        { from: "fb", to: "act_quit", label: "quit (0)", curvature: -60, labelOff: { x: -10, y: -20 } },
        { from: "act_quit", to: "class", label: "", curvature: -40, dashed: true },

        { from: "fb", to: "act_cont", label: "keep scrolling", curvature: -40, labelOff: { x: -20, y: 0 } }, // Curve to corner
        { from: "act_cont", to: "fb", label: "-1", curvature: -20, dashed: true, labelOff: { x: -15, y: 10 } },

        // PUB Outgoing
        { from: "pub", to: "act_pub", label: "drink", straight: true, labelOff: { x: 20, y: 0 } },
        { from: "act_pub", to: "class", label: "0.2 / 0.4 / 0.4", straight: true, dashed: true, labelOff: { x: 50, y: 0 } },
    ];

    return (
        <div className="w-full max-w-4xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl shadow-sm border border-slate-200 dark:border-slate-800">
            <h3 className="text-lg font-bold mb-6 text-center text-slate-800 dark:text-slate-100 flex items-center justify-center gap-2">
                <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                Student MDP Example
            </h3>

            <div className="relative h-[480px] w-full bg-[#f8fafc] dark:bg-[#0f172a] rounded-xl overflow-hidden border border-slate-100 dark:border-slate-800 select-none">
                {/* SVG Layer for Edges (Background) */}
                <svg className="absolute inset-0 w-full h-full pointer-events-none z-0">
                    <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="22" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" />
                        </marker>
                        <marker id="arrowhead-dash" markerWidth="10" markerHeight="7" refX="22" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#cbd5e1" />
                        </marker>
                    </defs>

                    {edges.map((edge, idx) => {
                        const start = nodes.find(n => n.id === edge.from)!;
                        const end = nodes.find(n => n.id === edge.to)!;

                        let d = "";
                        if (edge.loop) {
                            d = getSelfLoopPath(start.x, start.y, 50, edge.loop as any);
                        } else {
                            d = getPath(start.x, start.y, end.x, end.y, edge.curvature || 0);
                        }

                        return (
                            <path
                                key={idx}
                                d={d}
                                fill="none"
                                stroke={edge.dashed ? "#cbd5e1" : "#94a3b8"}
                                strokeWidth="2"
                                strokeDasharray={edge.dashed ? "6,4" : "0"}
                                markerEnd={edge.dashed ? "url(#arrowhead-dash)" : "url(#arrowhead)"}
                                className="transition-all duration-300"
                            />
                        );
                    })}
                </svg>

                {/* HTML Layer for Nodes (Middle) */}
                {nodes.map(node => (
                    <motion.div
                        key={node.id}
                        className={`absolute flex items-center justify-center text-xs font-bold shadow-soft-lg cursor-pointer transition-colors duration-300 z-10
                            ${node.type === "state" ?
                                `w-16 h-16 rounded-full border-2 ${activeNode === node.id ? 'bg-blue-600 text-white border-blue-700' : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 border-slate-200 dark:border-slate-700'}` :
                                node.type === "action" ?
                                    `w-7 h-7 rounded-md bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 border border-transparent` :
                                    `w-14 h-14 rounded-md border-2 bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-500`
                            }`}
                        style={{
                            left: node.x,
                            top: node.y,
                            marginLeft: node.type === "action" ? -14 : node.type === "term" ? -28 : -32, // Center offset
                            marginTop: node.type === "action" ? -14 : node.type === "term" ? -28 : -32
                        }}
                        onClick={() => setActiveNode(node.id)}
                        whileHover={{ scale: 1.05 }}
                    >
                        {node.type === "action" ? (
                            <div className="transform rotate-45 w-full h-full absolute top-0 left-0 bg-inherit rounded-sm -z-10 border border-inherit shadow-md"></div>
                        ) : null}
                        <span className={`relative z-10 ${node.type === "action" ? "" : ""}`}>{node.label}</span>
                    </motion.div>
                ))}

                {/* HTML Layer for Labels (Top) -> Guaranteed visibility */}
                {edges.map((edge, idx) => {
                    if (!edge.label) return null;

                    const start = nodes.find(n => n.id === edge.from)!;
                    const end = nodes.find(n => n.id === edge.to)!;

                    // Simple midpoint calculation
                    const midX = (start.x + end.x) / 2;
                    const midY = (start.y + end.y) / 2;

                    let textX = midX;
                    let textY = midY;

                    // Curve approximation for midpoint
                    if (edge.curvature) {
                        textY += (edge.curvature > 0 ? 20 : -20);
                    }

                    // Apply manual offsets
                    if (edge.labelOff) {
                        textX += edge.labelOff.x;
                        textY += edge.labelOff.y;
                    }

                    return (
                        <div
                            key={`label-${idx}`}
                            className="absolute flex items-center justify-center transform -translate-x-1/2 -translate-y-1/2 z-20 pointer-events-none"
                            style={{ left: textX, top: textY }}
                        >
                            <span className="bg-white/95 dark:bg-slate-900/95 backdrop-blur-sm border border-slate-200/50 dark:border-slate-700/50 rounded-full px-2 py-0.5 text-[10px] font-mono font-medium text-slate-600 dark:text-slate-300 shadow-sm whitespace-nowrap">
                                {edge.label}
                            </span>
                        </div>
                    );
                })}
            </div>

            <div className="mt-4 flex justify-center gap-6 text-xs text-slate-400 font-mono">
                <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full border-2 border-slate-300 bg-white"></span> State</div>
                <div className="flex items-center gap-2"><span className="w-3 h-3 rotate-45 bg-slate-800"></span> Action</div>
                <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-md border-2 border-slate-300 bg-slate-100"></span> Terminal</div>
            </div>
        </div>
    );
}
