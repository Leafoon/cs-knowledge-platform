"use client";

import { motion } from "framer-motion";
import { useState } from "react";
import { BookOpen, Coffee, Facebook, Moon, PartyPopper, CheckCircle2, XCircle } from "lucide-react";

// Helper to generate curved SVG paths
const getCurvedPath = (x1: number, y1: number, x2: number, y2: number, curvature = 0) => {
    if (curvature === 0) return `M ${x1} ${y1} L ${x2} ${y2}`;

    const midX = (x1 + x2) / 2;
    const midY = (y1 + y2) / 2;
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);

    if (len === 0) return "";

    const offsetX = -dy * (curvature / len);
    const offsetY = dx * (curvature / len);
    const cX = midX + offsetX;
    const cY = midY + offsetY;

    return `M ${x1} ${y1} Q ${cX} ${cY} ${x2} ${y2}`;
};

// Self-loop path for Facebook state
const getSelfLoopPath = (x: number, y: number) => {
    return `M ${x - 30} ${y} C ${x - 60} ${y - 30}, ${x - 60} ${y + 30}, ${x - 30} ${y}`;
};

interface Node {
    id: string;
    label: string;
    x: number;
    y: number;
    type: 'state' | 'terminal';
    color: string;
    icon?: any;
}

interface Edge {
    from: string;
    to: string;
    action: string;
    reward: string;
    curvature?: number;
    selfLoop?: boolean;
}

export function MDPGraphVisualizer() {
    const [activeNode, setActiveNode] = useState<string | null>(null);
    const [hoveredEdge, setHoveredEdge] = useState<number | null>(null);

    // Cleaner circular layout - 600x500 canvas
    const nodes: Node[] = [
        { id: "sleep", label: "Sleep", x: 300, y: 80, type: "state", color: "bg-blue-100 border-blue-300 text-blue-700", icon: Moon },
        { id: "study", label: "Study", x: 480, y: 200, type: "state", color: "bg-green-100 border-green-300 text-green-700", icon: BookOpen },
        { id: "class", label: "Class", x: 420, y: 350, type: "state", color: "bg-cyan-100 border-cyan-300 text-cyan-700", icon: BookOpen },
        { id: "pub", label: "Pub", x: 220, y: 400, type: "state", color: "bg-orange-100 border-orange-300 text-orange-700", icon: Coffee },
        { id: "facebook", label: "Facebook", x: 100, y: 250, type: "state", color: "bg-yellow-100 border-yellow-300 text-yellow-700", icon: Facebook },
        { id: "pass", label: "Pass", x: 500, y: 480, type: "terminal", color: "bg-emerald-100 border-emerald-400 text-emerald-700", icon: CheckCircle2 },
        { id: "fail", label: "Fail", x: 380, y: 480, type: "terminal", color: "bg-red-100 border-red-400 text-red-700", icon: XCircle },
    ];

    const edges: Edge[] = [
        // From Sleep
        { from: "sleep", to: "study", action: "study", reward: "+1", curvature: 0 },
        { from: "sleep", to: "facebook", action: "facebook", reward: "-1", curvature: -30 },

        // From Study
        { from: "study", to: "sleep", action: "sleep", reward: "+1", curvature: 30 },
        { from: "study", to: "class", action: "attend", reward: "+10", curvature: 0 },

        // From Facebook
        { from: "facebook", to: "facebook", action: "scroll", reward: "-1", selfLoop: true },
        { from: "facebook", to: "sleep", action: "quit", reward: "+1", curvature: 30 },
        { from: "facebook", to: "study", action: "study", reward: "+1", curvature: -40 },

        // From Class
        { from: "class", to: "pass", action: "pass", reward: "+10", curvature: 0 },
        { from: "class", to: "pub", action: "pub", reward: "-2", curvature: 0 },

        // From Pub
        { from: "pub", to: "class", action: "study", reward: "+1", curvature: -20 },
        { from: "pub", to: "sleep", action: "sleep", reward: "-1", curvature: 40 },
    ];

    return (
        <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-white to-slate-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-lg border border-slate-200 dark:border-slate-700">
            <div className="text-center mb-6">
                <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Student MDP Example
                </h3>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                    A student's daily decision-making process
                </p>
            </div>

            <div className="relative h-[520px] w-full bg-white dark:bg-slate-900 rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-inner">
                {/* SVG Layer for Edges */}
                <svg className="absolute inset-0 w-full h-full pointer-events-none z-0">
                    <defs>
                        <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="20" refY="4" orient="auto">
                            <polygon points="0 0, 8 4, 0 8" fill="#64748b" className="dark:fill-slate-400" />
                        </marker>
                        <marker id="arrowhead-hover" markerWidth="8" markerHeight="8" refX="20" refY="4" orient="auto">
                            <polygon points="0 0, 8 4, 0 8" fill="#3b82f6" />
                        </marker>
                    </defs>

                    {edges.map((edge, idx) => {
                        const start = nodes.find(n => n.id === edge.from)!;
                        const end = nodes.find(n => n.id === edge.to)!;
                        const isHovered = hoveredEdge === idx;
                        const isActive = activeNode === edge.from || activeNode === edge.to;

                        let d = "";
                        if (edge.selfLoop) {
                            d = getSelfLoopPath(start.x, start.y);
                        } else {
                            d = getCurvedPath(start.x, start.y, end.x, end.y, edge.curvature || 0);
                        }

                        return (
                            <g key={idx}>
                                <path
                                    d={d}
                                    fill="none"
                                    stroke={isHovered || isActive ? "#3b82f6" : "#94a3b8"}
                                    strokeWidth={isHovered ? "3" : "2"}
                                    markerEnd={isHovered ? "url(#arrowhead-hover)" : "url(#arrowhead)"}
                                    className="transition-all duration-200"
                                    opacity={isHovered || isActive ? 1 : 0.6}
                                />
                            </g>
                        );
                    })}
                </svg>

                {/* State Nodes */}
                {nodes.map((node) => {
                    const Icon = node.icon;
                    const isActive = activeNode === node.id;

                    return (
                        <motion.div
                            key={node.id}
                            className={`absolute flex flex-col items-center justify-center cursor-pointer transition-all duration-200 z-10
                                ${node.type === 'state' ? 'w-20 h-20' : 'w-16 h-16'}
                                rounded-full border-3 shadow-lg
                                ${node.color}
                                ${isActive ? 'ring-4 ring-blue-400 scale-110' : 'hover:scale-105'}
                            `}
                            style={{
                                left: node.x - (node.type === 'state' ? 40 : 32),
                                top: node.y - (node.type === 'state' ? 40 : 32),
                            }}
                            onClick={() => setActiveNode(node.id === activeNode ? null : node.id)}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                        >
                            {Icon && <Icon className="w-5 h-5 mb-0.5" />}
                            <span className="text-[10px] font-bold">{node.label}</span>
                        </motion.div>
                    );
                })}

                {/* Edge Labels (Action + Reward) */}
                {edges.map((edge, idx) => {
                    const start = nodes.find(n => n.id === edge.from)!;
                    const end = nodes.find(n => n.id === edge.to)!;
                    const isHovered = hoveredEdge === idx;

                    let labelX, labelY;

                    if (edge.selfLoop) {
                        labelX = start.x - 65;
                        labelY = start.y;
                    } else {
                        labelX = (start.x + end.x) / 2;
                        labelY = (start.y + end.y) / 2;

                        if (edge.curvature) {
                            const offset = edge.curvature > 0 ? 25 : -25;
                            labelY += offset;
                        }
                    }

                    return (
                        <div
                            key={`label-${idx}`}
                            className="absolute transform -translate-x-1/2 -translate-y-1/2 z-20 pointer-events-auto cursor-pointer"
                            style={{ left: labelX, top: labelY }}
                            onMouseEnter={() => setHoveredEdge(idx)}
                            onMouseLeave={() => setHoveredEdge(null)}
                        >
                            <div className={`flex flex-col items-center gap-0.5 transition-all duration-200 ${isHovered ? 'scale-110' : ''}`}>
                                <span className={`px-2 py-0.5 rounded-full text-[9px] font-semibold whitespace-nowrap
                                    ${isHovered
                                        ? 'bg-blue-500 text-white shadow-md'
                                        : 'bg-white/90 dark:bg-slate-800/90 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-600'
                                    }`}>
                                    {edge.action}
                                </span>
                                <span className={`px-1.5 py-0.5 rounded text-[8px] font-mono font-bold
                                    ${edge.reward.startsWith('+')
                                        ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                                        : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                                    }`}>
                                    {edge.reward}
                                </span>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Legend */}
            <div className="mt-6 flex flex-wrap justify-center gap-4 text-xs">
                <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 dark:bg-slate-800 rounded-full">
                    <div className="w-4 h-4 rounded-full bg-blue-100 border-2 border-blue-300"></div>
                    <span className="text-slate-600 dark:text-slate-300 font-medium">State</span>
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 dark:bg-slate-800 rounded-full">
                    <div className="w-4 h-4 rounded-full bg-emerald-100 border-2 border-emerald-400"></div>
                    <span className="text-slate-600 dark:text-slate-300 font-medium">Terminal</span>
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 dark:bg-slate-800 rounded-full">
                    <div className="flex items-center gap-1">
                        <div className="w-3 h-0.5 bg-slate-400"></div>
                        <div className="w-0 h-0 border-l-4 border-l-slate-400 border-y-2 border-y-transparent"></div>
                    </div>
                    <span className="text-slate-600 dark:text-slate-300 font-medium">Action</span>
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 dark:bg-slate-800 rounded-full">
                    <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-green-100 text-green-700">+10</span>
                    <span className="text-slate-600 dark:text-slate-300 font-medium">Reward</span>
                </div>
            </div>
        </div>
    );
}
