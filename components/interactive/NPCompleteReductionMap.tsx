"use client";

import { useState } from "react";
import { motion } from "framer-motion";

interface NPCNode {
  id: string;
  label: string;
  cn: string;
  x: number;
  y: number;
  color: string;
  bg: string;
  border: string;
  desc: string;
  year?: string;
}

interface NPCEdge {
  from: string;
  to: string;
  label: string;
  desc: string;
  color: string;
}

const NODES: NPCNode[] = [
  {
    id: "SAT", label: "SAT", cn: "可满足性", x: 65, y: 185,
    color: "#7c3aed", bg: "#f5f3ff", border: "#7c3aed",
    desc: "k-SAT 的一般形式。Cook-Levin 定理（1971）证明这是第一个 NPC 问题。",
    year: "1971",
  },
  {
    id: "3SAT", label: "3-SAT", cn: "3-可满足性", x: 200, y: 185,
    color: "#4f46e5", bg: "#eef2ff", border: "#4f46e5",
    desc: "每个子句恰好含 3 个文字的 CNF SAT 问题。是绝大多数规约的起点。",
    year: "1972",
  },
  {
    id: "IS", label: "IS", cn: "独立集", x: 340, y: 100,
    color: "#be123c", bg: "#fff1f2", border: "#f43f5e",
    desc: "图中 k 个两两不相邻节点的集合是否存在？3-SAT ≤p IS。",
  },
  {
    id: "3COL", label: "3-COL", cn: "图 3 着色", x: 340, y: 185,
    color: "#c2410c", bg: "#fff7ed", border: "#f97316",
    desc: "用 3 种颜色对图节点着色，使相邻节点颜色不同——是否可行？",
  },
  {
    id: "SSUM", label: "S-SUM", cn: "子集和", x: 340, y: 275,
    color: "#047857", bg: "#f0fdf4", border: "#10b981",
    desc: "在整数集合中，是否存在子集使其和恰好为目标 T？",
  },
  {
    id: "VC", label: "VC", cn: "顶点覆盖", x: 475, y: 75,
    color: "#be123c", bg: "#fff1f2", border: "#f43f5e",
    desc: "图中大小为 k 的顶点子集，覆盖所有边（每条边至少一个端点在集内）。IS ≤p VC。",
  },
  {
    id: "CLIQUE", label: "Clique", cn: "最大团", x: 475, y: 145,
    color: "#be123c", bg: "#fff1f2", border: "#f43f5e",
    desc: "图中大小为 k 的完全子图是否存在？与 IS 互为补图关系。IS ≤p Clique。",
  },
  {
    id: "HAM", label: "HAM", cn: "哈密顿回路", x: 480, y: 225,
    color: "#1d4ed8", bg: "#eff6ff", border: "#3b82f6",
    desc: "图中是否存在一条经过每个顶点恰好一次的回路？",
  },
  {
    id: "KNAP", label: "Knapsack", cn: "背包问题", x: 475, y: 300,
    color: "#047857", bg: "#f0fdf4", border: "#10b981",
    desc: "0/1 背包判定版本：给定容量 W，是否能选出物品使价值≥V 且重量≤W？S-SUM ≤p Knapsack。",
  },
  {
    id: "TSP", label: "TSP", cn: "旅行商", x: 610, y: 225,
    color: "#1d4ed8", bg: "#eff6ff", border: "#3b82f6",
    desc: "是否存在权重≤k 的哈密顿回路？HAM-CYCLE ≤p TSP。最优化版是 NP-hard。",
  },
];

const EDGES: NPCEdge[] = [
  { from: "SAT", to: "3SAT", label: "Cook-Levin", desc: "子句填充/拆分，每个子句至多3文字", color: "#7c3aed" },
  { from: "3SAT", to: "IS", label: "gadget", desc: "变量↔节点，子句↔三角形小工件构造", color: "#4f46e5" },
  { from: "3SAT", to: "3COL", label: "gadget", desc: "变量节点+Truth gadget+Clause gadget", color: "#4f46e5" },
  { from: "3SAT", to: "SSUM", label: "编码", desc: "变量与子句编码为整数，目标 T=[1…1, 3…3]", color: "#4f46e5" },
  { from: "IS", to: "VC", label: "互补", desc: "V∖IS 即为 VC，|IS|+|VC|=|V|", color: "#be123c" },
  { from: "IS", to: "CLIQUE", label: "补图", desc: "Ḡ 中的最大团 = G 中的最大 IS", color: "#be123c" },
  { from: "CLIQUE", to: "HAM", label: "gadget", desc: "团→路径 gadget（复杂构造）", color: "#be123c" },
  { from: "HAM", to: "TSP", label: "权重1", desc: "边权置 1，不存在的边权置 2，阈值 k=|V|", color: "#1d4ed8" },
  { from: "SSUM", to: "KNAP", label: "单物品", desc: "每个整数=价值=重量，W=V=目标 T", color: "#047857" },
];

function nodeById(id: string) {
  return NODES.find(n => n.id === id)!;
}

// Compute arrowhead position offset toward center (to not overlap circle)
const R = 26; // node radius

function getArrow(fromId: string, toId: string) {
  const a = nodeById(fromId), b = nodeById(toId);
  const dx = b.x - a.x, dy = b.y - a.y;
  const len = Math.sqrt(dx * dx + dy * dy);
  const ux = dx / len, uy = dy / len;
  return {
    x1: a.x + ux * R,
    y1: a.y + uy * R,
    x2: b.x - ux * (R + 6),
    y2: b.y - uy * (R + 6),
    mx: (a.x + b.x) / 2,
    my: (a.y + b.y) / 2,
    angle: Math.atan2(dy, dx),
  };
}

export function NPCompleteReductionMap() {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<NPCEdge | null>(null);

  const activeInfo = hoveredNode
    ? nodeById(hoveredNode)
    : null;

  // Determine highlighted edges
  const isEdgeHighlighted = (e: NPCEdge) =>
    hoveredNode === e.from || hoveredNode === e.to || hoveredEdge === e;

  return (
    <div className="w-full max-w-5xl mx-auto my-6 rounded-2xl overflow-hidden border border-purple-200 dark:border-purple-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-600 via-indigo-600 to-blue-600 px-6 py-4 text-white">
        <h3 className="text-lg font-bold">NPC 规约关系有向图</h3>
        <p className="text-sm text-violet-100 mt-0.5">箭头 A → B 表示 A ≤p B（A 规约到 B）· 悬停节点/边查看详情</p>
      </div>

      <div className="bg-white dark:bg-slate-900">
        <div className="flex flex-col lg:flex-row">
          {/* SVG */}
          <div className="flex-1 p-4 min-w-0">
            <svg viewBox="0 0 700 380" className="w-full" style={{ minHeight: 260 }}>
              <defs>
                {/* Arrowhead markers per color */}
                {["#7c3aed", "#4f46e5", "#be123c", "#047857", "#1d4ed8", "#94a3b8"].map(col => (
                  <marker
                    key={col}
                    id={`arrow-${col.slice(1)}`}
                    markerWidth="8" markerHeight="8"
                    refX="6" refY="3"
                    orient="auto"
                  >
                    <path d="M0,0 L0,6 L8,3 z" fill={col} />
                  </marker>
                ))}
                <marker id="arrow-active" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L8,3 z" fill="#f59e0b" />
                </marker>
              </defs>

              {/* Edges */}
              {EDGES.map((edge, i) => {
                const arr = getArrow(edge.from, edge.to);
                const highlighted = isEdgeHighlighted(edge);
                const col = highlighted ? "#f59e0b" : edge.color;
                const markerId = `arrow-${highlighted ? "active" : edge.color.slice(1)}`;
                return (
                  <g key={i}>
                    <line
                      x1={arr.x1} y1={arr.y1} x2={arr.x2} y2={arr.y2}
                      stroke={col}
                      strokeWidth={highlighted ? 2.5 : 1.5}
                      strokeOpacity={highlighted ? 1 : 0.5}
                      markerEnd={`url(#${markerId})`}
                      style={{ cursor: "pointer", transition: "all 0.2s" }}
                      onMouseEnter={() => setHoveredEdge(edge)}
                      onMouseLeave={() => setHoveredEdge(null)}
                    />
                    {/* Edge label */}
                    {highlighted && (
                      <text
                        x={arr.mx} y={arr.my - 6}
                        textAnchor="middle"
                        fontSize="9"
                        fill={col}
                        fontFamily="sans-serif"
                        fontWeight="700"
                        style={{ pointerEvents: "none" }}
                      >
                        {edge.label}
                      </text>
                    )}
                  </g>
                );
              })}

              {/* Nodes */}
              {NODES.map(node => {
                const isHov = hoveredNode === node.id;
                const isRelated = hoveredNode ? EDGES.some(e => (e.from === hoveredNode && e.to === node.id) || (e.to === hoveredNode && e.from === node.id)) : false;
                return (
                  <g
                    key={node.id}
                    onMouseEnter={() => setHoveredNode(node.id)}
                    onMouseLeave={() => setHoveredNode(null)}
                    style={{ cursor: "pointer" }}
                  >
                    {/* Glow ring */}
                    {(isHov || isRelated) && (
                      <circle
                        cx={node.x} cy={node.y} r={R + 8}
                        fill={isHov ? node.color : "#f59e0b"}
                        opacity={0.15}
                      />
                    )}
                    {/* Main circle */}
                    <circle
                      cx={node.x} cy={node.y} r={R}
                      fill={isHov ? node.color : node.bg}
                      stroke={isHov ? node.color : (isRelated ? "#f59e0b" : node.border)}
                      strokeWidth={isHov ? 3 : isRelated ? 2.5 : 2}
                      style={{ transition: "all 0.2s" }}
                    />
                    {/* White dark mode overlay */}
                    <circle
                      cx={node.x} cy={node.y} r={R}
                      fill="transparent"
                      stroke="none"
                    />
                    {/* Label */}
                    <text
                      x={node.x} y={node.y + 1}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fontSize="10.5"
                      fontWeight="800"
                      fontFamily="sans-serif"
                      fill={isHov ? "white" : node.color}
                      style={{ transition: "fill 0.2s", pointerEvents: "none" }}
                    >
                      {node.label}
                    </text>
                    {/* Sublabel */}
                    <text
                      x={node.x} y={node.y + R + 13}
                      textAnchor="middle"
                      fontSize="8.5"
                      fontFamily="sans-serif"
                      fill={isHov ? node.color : "#94a3b8"}
                      fontWeight={isHov ? "600" : "400"}
                      style={{ pointerEvents: "none" }}
                    >
                      {node.cn}
                    </text>
                    {/* Year badge */}
                    {node.year && (
                      <text
                        x={node.x + R - 2} y={node.y - R + 8}
                        textAnchor="middle"
                        fontSize="8"
                        fill={node.color}
                        fontFamily="sans-serif"
                        opacity={0.7}
                        style={{ pointerEvents: "none" }}
                      >
                        {node.year}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Info panel */}
          <div className="lg:w-72 flex-shrink-0 p-4 border-t lg:border-t-0 lg:border-l border-slate-100 dark:border-slate-800 space-y-3">
            {/* Node info */}
            <div className="min-h-32">
              {activeInfo ? (
                <motion.div
                  key={activeInfo.id}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-3 rounded-xl border"
                  style={{ borderColor: activeInfo.border, backgroundColor: `${activeInfo.bg}` }}
                >
                  <div className="flex items-start gap-2 mb-2">
                    <span className="text-lg font-black" style={{ color: activeInfo.color }}>{activeInfo.label}</span>
                    <span className="text-xs pt-1 opacity-70" style={{ color: activeInfo.color }}>{activeInfo.cn}</span>
                  </div>
                  <p className="text-xs leading-relaxed" style={{ color: activeInfo.color }}>{activeInfo.desc}</p>
                  {/* Connected edges */}
                  <div className="mt-2 space-y-1">
                    {EDGES.filter(e => e.from === activeInfo.id || e.to === activeInfo.id).map((e, i) => (
                      <div key={i} className="text-xs opacity-80" style={{ color: activeInfo.color }}>
                        {e.from === activeInfo.id
                          ? <span>→ <strong>{e.to}</strong>（{e.label}）</span>
                          : <span>← <strong>{e.from}</strong>（{e.label}）</span>}
                      </div>
                    ))}
                  </div>
                </motion.div>
              ) : hoveredEdge ? (
                <motion.div
                  key={`${hoveredEdge.from}-${hoveredEdge.to}`}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-3 rounded-xl bg-amber-50 dark:bg-amber-900/30 border border-amber-300 dark:border-amber-700"
                >
                  <p className="text-xs font-bold text-amber-700 dark:text-amber-300 mb-1">
                    {hoveredEdge.from} ≤p {hoveredEdge.to}
                  </p>
                  <p className="text-xs font-semibold text-amber-600 dark:text-amber-400 mb-1">{hoveredEdge.label}</p>
                  <p className="text-xs text-amber-600 dark:text-amber-400 opacity-80">{hoveredEdge.desc}</p>
                </motion.div>
              ) : (
                <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-center">
                  <p className="text-xs text-slate-400 dark:text-slate-500">悬停节点查看问题描述</p>
                  <p className="text-xs text-slate-300 dark:text-slate-600 mt-1">悬停箭头查看规约方法</p>
                </div>
              )}
            </div>

            {/* Legend */}
            <div className="space-y-2">
              <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">问题类别</p>
              {[
                { color: "#7c3aed", bg: "#f5f3ff", label: "SAT 族（出发点）" },
                { color: "#be123c", bg: "#fff1f2", label: "图论问题" },
                { color: "#c2410c", bg: "#fff7ed", label: "图着色问题" },
                { color: "#047857", bg: "#f0fdf4", label: "算术/集合问题" },
                { color: "#1d4ed8", bg: "#eff6ff", label: "路径问题" },
              ].map(item => (
                <div key={item.label} className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full flex-shrink-0 border-2" style={{ borderColor: item.color, backgroundColor: item.bg }} />
                  <span className="text-xs text-slate-600 dark:text-slate-400">{item.label}</span>
                </div>
              ))}
            </div>

            {/* Key fact */}
            <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded-xl border border-violet-200 dark:border-violet-800 text-xs text-violet-700 dark:text-violet-300">
              <p className="font-bold mb-1">传递性</p>
              <p>A ≤p B 且 B ≤p C ⟹ A ≤p C。因此链上任何一个有多项式算法，则全链均有。</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
