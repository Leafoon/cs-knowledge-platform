"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

type ViewMode = "graph" | "IS" | "VC";

// Graph: 6 nodes (hexagon) + extra edges
// IS = {0, 2, 4}  VC = {1, 3, 5}
const NODES = [
  { id: 0, x: 220, y: 55,  label: "v₀" },
  { id: 1, x: 310, y: 105, label: "v₁" },
  { id: 2, x: 310, y: 205, label: "v₂" },
  { id: 3, x: 220, y: 255, label: "v₃" },
  { id: 4, x: 130, y: 205, label: "v₄" },
  { id: 5, x: 130, y: 105, label: "v₅" },
];

const EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0],
  [1, 3], [1, 4],
];

const IS_SET = new Set([0, 2, 4]);
const VC_SET = new Set([1, 3, 5]);

const MODE_CONFIG = {
  graph: { label: "原始图", color: "slate", desc: "图 G，共 6 个顶点，8 条边" },
  IS:    { label: "独立集 IS", color: "emerald", desc: "IS = {v₀, v₂, v₄}（绿色），大小 k = 3，任意两节点间无边" },
  VC:    { label: "顶点覆盖 VC", color: "rose", desc: "VC = {v₁, v₃, v₅}（红色），大小 n−k = 3，覆盖所有边" },
};

export function VertexCoverNPCProof() {
  const [mode, setMode] = useState<ViewMode>("graph");
  const [hovered, setHovered] = useState<number | null>(null);

  const getNodeStyle = (id: number) => {
    if (mode === "IS") {
      if (IS_SET.has(id)) return { fill: "#10b981", stroke: "#059669", textColor: "white", ring: true };
      return { fill: "#f1f5f9", stroke: "#94a3b8", textColor: "#475569", ring: false };
    }
    if (mode === "VC") {
      if (VC_SET.has(id)) return { fill: "#f43f5e", stroke: "#e11d48", textColor: "white", ring: true };
      return { fill: "#f1f5f9", stroke: "#94a3b8", textColor: "#475569", ring: false };
    }
    // graph mode
    if (hovered === id) return { fill: "#6366f1", stroke: "#4f46e5", textColor: "white", ring: true };
    return { fill: "#e0e7ff", stroke: "#6366f1", textColor: "#3730a3", ring: false };
  };

  const getEdgeColor = (a: number, b: number) => {
    if (mode === "IS") {
      // edges within IS (should be none) would be red; all others gray
      if (IS_SET.has(a) && IS_SET.has(b)) return "#ef4444";
      if (IS_SET.has(a) || IS_SET.has(b)) return "#d1fae5";
      return "#e2e8f0";
    }
    if (mode === "VC") {
      // edges covered by VC (at least one endpoint in VC)
      if (VC_SET.has(a) || VC_SET.has(b)) return "#fda4af";
      return "#e2e8f0";
    }
    return "#94a3b8";
  };

  const getEdgeWidth = (a: number, b: number) => {
    if (mode === "VC" && (VC_SET.has(a) || VC_SET.has(b))) return 2.5;
    if (mode === "IS" && (IS_SET.has(a) || IS_SET.has(b))) return 2;
    return 1.5;
  };

  const isDarkAwareFill = (id: number) => {
    const s = getNodeStyle(id);
    return s.fill;
  };

  return (
    <div className="w-full max-w-4xl mx-auto my-6 rounded-2xl overflow-hidden border border-rose-200 dark:border-rose-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-rose-600 to-pink-600 px-6 py-4 text-white">
        <h3 className="text-lg font-bold">独立集 ↔ 顶点覆盖：互补关系可视化</h3>
        <p className="text-sm text-rose-100 mt-0.5">同一图，三种视角——切换观察 IS 与 VC 的天然对偶性</p>
      </div>

      <div className="bg-white dark:bg-slate-900 p-6">
        <div className="flex flex-col lg:flex-row gap-6">
          {/* SVG Graph */}
          <div className="flex-shrink-0 flex flex-col items-center">
            {/* Mode tabs */}
            <div className="flex gap-2 mb-4">
              {(["graph", "IS", "VC"] as ViewMode[]).map(m => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`px-4 py-2 rounded-xl text-sm font-semibold border transition-all ${
                    mode === m
                      ? m === "IS"
                        ? "bg-emerald-500 text-white border-emerald-500 shadow"
                        : m === "VC"
                        ? "bg-rose-500 text-white border-rose-500 shadow"
                        : "bg-indigo-500 text-white border-indigo-500 shadow"
                      : "border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:border-slate-300"
                  }`}
                >
                  {MODE_CONFIG[m].label}
                </button>
              ))}
            </div>

            <svg viewBox="0 0 440 310" className="w-72 h-64" style={{ overflow: "visible" }}>
              {/* Edges */}
              {EDGES.map(([a, b], i) => {
                const na = NODES[a], nb = NODES[b];
                return (
                  <motion.line
                    key={i}
                    x1={na.x} y1={na.y} x2={nb.x} y2={nb.y}
                    stroke={getEdgeColor(a, b)}
                    strokeWidth={getEdgeWidth(a, b)}
                    strokeLinecap="round"
                    animate={{ stroke: getEdgeColor(a, b), strokeWidth: getEdgeWidth(a, b) }}
                    transition={{ duration: 0.3 }}
                  />
                );
              })}

              {/* Nodes */}
              {NODES.map(node => {
                const s = getNodeStyle(node.id);
                return (
                  <g key={node.id}
                    onMouseEnter={() => setHovered(node.id)}
                    onMouseLeave={() => setHovered(null)}
                    style={{ cursor: "pointer" }}
                  >
                    {s.ring && (
                      <motion.circle
                        cx={node.x} cy={node.y} r={26}
                        fill="none"
                        stroke={s.stroke}
                        strokeWidth={3}
                        strokeDasharray="4 3"
                        initial={{ r: 18 }}
                        animate={{ r: 26 }}
                        transition={{ duration: 0.2 }}
                        opacity={0.4}
                      />
                    )}
                    <motion.circle
                      cx={node.x} cy={node.y} r={20}
                      fill={s.fill}
                      stroke={s.stroke}
                      strokeWidth={2.5}
                      animate={{ fill: s.fill, stroke: s.stroke }}
                      transition={{ duration: 0.3 }}
                    />
                    <text
                      x={node.x} y={node.y + 4}
                      textAnchor="middle"
                      fontSize="11"
                      fontWeight="700"
                      fontFamily="sans-serif"
                      fill={s.textColor}
                    >
                      {node.label}
                    </text>
                  </g>
                );
              })}
            </svg>

            <AnimatePresence mode="wait">
              <motion.div
                key={mode}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className={`mt-3 px-4 py-2 rounded-xl text-sm text-center max-w-xs font-medium ${
                  mode === "IS" ? "bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-700"
                  : mode === "VC" ? "bg-rose-50 dark:bg-rose-900/30 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-700"
                  : "bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-700"
                }`}
              >
                {MODE_CONFIG[mode].desc}
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Info panel */}
          <div className="flex-1 min-w-0 space-y-4">
            {/* Complement theorem */}
            <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
              <h4 className="text-sm font-bold text-slate-800 dark:text-slate-100 mb-3">互补定理（核心）</h4>
              <div className="space-y-2 text-sm">
                <p className="text-slate-600 dark:text-slate-400">
                  对于图 G = (V, E) 的任意子集 S ⊆ V：
                </p>
                <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 font-mono text-xs">
                  <p className="text-indigo-600 dark:text-indigo-400 font-bold">S 是 IS（大小≥k）</p>
                  <p className="my-1 text-slate-400">⟺</p>
                  <p className="text-rose-600 dark:text-rose-400 font-bold">V∖S 是 VC（大小≤n−k）</p>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  本图中：V={"{v₀…v₅}"}，n=6，IS={"{v₀,v₂,v₄}"}，VC={"{v₁,v₃,v₅}"}，|IS|=|VC|=3=n/2
                </p>
              </div>
            </div>

            {/* Proof sketch */}
            <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
              <h4 className="text-sm font-bold text-slate-800 dark:text-slate-100 mb-3">规约证明框架（IS ≤p VC）</h4>
              <div className="space-y-2">
                {[
                  { step: "输入变换", detail: "給定 ⟨G, k⟩（IS 判定问题），构造 ⟨G, n−k⟩（VC 判定问题）。同一个图 G，参数变为 n−k。" },
                  { step: "方向一：IS→VC", detail: "若 S 是大小≥k 的 IS，则对任意边 (u,v)，u,v 不能都在 S 中，故 V∖S 覆盖所有边，|V∖S|≤n−k。" },
                  { step: "方向二：VC→IS", detail: "若 C 是大小≤n−k 的 VC，则 V∖C 大小≥k；若 (u,v)∈E 则 u 或 v 在 C 中，故 V∖C 中无边，即为 IS。" },
                  { step: "复杂度", detail: "n−k 可 O(1) 计算，图复制需 O(V+E)。整个规约为多项式时间：IS ≤p VC。" },
                ].map((item, i) => (
                  <div key={i} className="flex gap-3 p-2.5 rounded-lg hover:bg-white dark:hover:bg-slate-900 transition-colors">
                    <div className="w-2 h-2 mt-1.5 rounded-full bg-rose-400 flex-shrink-0" />
                    <div className="text-xs">
                      <span className="font-semibold text-slate-700 dark:text-slate-200">{item.step}：</span>
                      <span className="text-slate-500 dark:text-slate-400">{item.detail}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* NPC chain */}
            <div className="p-3 bg-rose-50 dark:bg-rose-900/20 rounded-xl border border-rose-200 dark:border-rose-800 text-xs">
              <span className="font-bold text-rose-700 dark:text-rose-400">NPC 证明链：</span>
              <span className="text-rose-600 dark:text-rose-300"> 3-SAT ≤p IS ≤p VC ≤p CLIQUE（互补）。三者互相等价，证明任一个为 NPC 即可推出其余两个。</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
