"use client";
import React, { useState } from "react";

/* ─── Graph data definitions ─────────────────────────────────────────────── */

interface NodePos { id: number; x: number; y: number; label: string; }
interface GraphEdge { u: number; v: number; w?: number; directed: boolean; }

interface GraphPreset {
  id: string;
  name: string;
  emoji: string;
  scene: string;
  nodes: NodePos[];
  edges: GraphEdge[];
  // Classification
  directed: boolean;
  weighted: boolean;
  connectivity: "connected" | "strongly-connected" | "weakly-connected" | "disconnected";
  isDAG: boolean;
  hasCycle: boolean;
  density: "sparse" | "dense";
  simple: boolean;
  description: string;
}

const PRESETS: GraphPreset[] = [
  {
    id: "social",
    name: "好友社交网络",
    emoji: "👥",
    scene: "微信好友关系",
    nodes: [
      { id: 0, x: 110, y: 28, label: "A" },
      { id: 1, x: 190, y: 70, label: "B" },
      { id: 2, x: 165, y: 148, label: "C" },
      { id: 3, x: 55,  y: 148, label: "D" },
      { id: 4, x: 30,  y: 70,  label: "E" },
    ],
    edges: [
      { u: 0, v: 1, directed: false },
      { u: 1, v: 2, directed: false },
      { u: 2, v: 3, directed: false },
      { u: 3, v: 4, directed: false },
      { u: 4, v: 0, directed: false },
      { u: 0, v: 2, directed: false },
    ],
    directed: false, weighted: false, connectivity: "connected",
    isDAG: false, hasCycle: true, density: "sparse", simple: true,
    description: "好友关系是「互相的」——A 是 B 的好友，B 也是 A 的好友，因此用无向无权图表示。",
  },
  {
    id: "weibo",
    name: "微博关注图",
    emoji: "📢",
    scene: "微博单向关注",
    nodes: [
      { id: 0, x: 60,  y: 45,  label: "A" },
      { id: 1, x: 170, y: 30,  label: "B" },
      { id: 2, x: 185, y: 120, label: "C" },
      { id: 3, x: 70,  y: 145, label: "D" },
    ],
    edges: [
      { u: 0, v: 1, directed: true },
      { u: 1, v: 2, directed: true },
      { u: 2, v: 3, directed: true },
      { u: 3, v: 0, directed: true },
      { u: 1, v: 3, directed: true },
    ],
    directed: true, weighted: false, connectivity: "strongly-connected",
    isDAG: false, hasCycle: true, density: "sparse", simple: true,
    description: "关注关系是「单向的」——A 关注 B，B 未必关注 A，需要有向图。此图所有顶点互相可达，构成强连通图。",
  },
  {
    id: "road",
    name: "城市路网",
    emoji: "🗺️",
    scene: "导航距离图",
    nodes: [
      { id: 0, x: 55,  y: 50,  label: "京" },
      { id: 1, x: 175, y: 50,  label: "津" },
      { id: 2, x: 55,  y: 140, label: "沪" },
      { id: 3, x: 175, y: 140, label: "杭" },
    ],
    edges: [
      { u: 0, v: 1, w: 100, directed: false },
      { u: 0, v: 2, w: 800, directed: false },
      { u: 1, v: 3, w: 120, directed: false },
      { u: 2, v: 3, w: 200, directed: false },
    ],
    directed: false, weighted: true, connectivity: "connected",
    isDAG: false, hasCycle: true, density: "sparse", simple: true,
    description: "城市之间的道路是双向的（无向），且需要记录距离（加权）。这是最常见的无向加权图应用场景。",
  },
  {
    id: "deps",
    name: "软件依赖图",
    emoji: "📦",
    scene: "Python 包依赖",
    nodes: [
      { id: 0, x: 110, y: 22,  label: "A" },
      { id: 1, x: 55,  y: 88,  label: "B" },
      { id: 2, x: 165, y: 88,  label: "C" },
      { id: 3, x: 110, y: 150, label: "D" },
    ],
    edges: [
      { u: 0, v: 1, directed: true },
      { u: 0, v: 2, directed: true },
      { u: 1, v: 3, directed: true },
      { u: 2, v: 3, directed: true },
    ],
    directed: true, weighted: false, connectivity: "weakly-connected",
    isDAG: true, hasCycle: false, density: "sparse", simple: true,
    description: "软件包的依赖关系是单向的，且不能有循环依赖（否则无法确定安装顺序）。这是有向无环图（DAG）的经典应用。",
  },
  {
    id: "complete",
    name: "完全图 K₄",
    emoji: "🔷",
    scene: "全员互联网络",
    nodes: [
      { id: 0, x: 110, y: 28,  label: "0" },
      { id: 1, x: 192, y: 110, label: "1" },
      { id: 2, x: 155, y: 162, label: "2" },
      { id: 3, x: 65,  y: 162, label: "3" },
    ],
    edges: [
      { u: 0, v: 1, directed: false },
      { u: 0, v: 2, directed: false },
      { u: 0, v: 3, directed: false },
      { u: 1, v: 2, directed: false },
      { u: 1, v: 3, directed: false },
      { u: 2, v: 3, directed: false },
    ],
    directed: false, weighted: false, connectivity: "connected",
    isDAG: false, hasCycle: true, density: "dense", simple: true,
    description: "K₄ 是 4 个顶点的完全图，拥有 C(4,2)=6 条边（最大可能边数）。这是典型的稠密图，适合用邻接矩阵存储。",
  },
  {
    id: "dag-task",
    name: "任务调度 DAG",
    emoji: "📋",
    scene: "工程任务依赖",
    nodes: [
      { id: 0, x: 40,  y: 88,  label: "T1" },
      { id: 1, x: 110, y: 40,  label: "T2" },
      { id: 2, x: 110, y: 138, label: "T3" },
      { id: 3, x: 180, y: 88,  label: "T4" },
    ],
    edges: [
      { u: 0, v: 1, directed: true },
      { u: 0, v: 2, directed: true },
      { u: 1, v: 3, directed: true },
      { u: 2, v: 3, directed: true },
    ],
    directed: true, weighted: false, connectivity: "weakly-connected",
    isDAG: true, hasCycle: false, density: "sparse", simple: true,
    description: "工程任务存在先后依赖：完成 T1 后才能开始 T2 和 T3，完成 T2 和 T3 后才能开始 T4。这种结构要求有拓扑顺序，即 DAG。",
  },
];

/* ─── SVG Arrow Marker ───────────────────────────────────────────────────── */

function ArrowMarker({ id, color }: { id: string; color: string }) {
  return (
    <defs>
      <marker id={id} markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0,0 L0,6 L8,3 z" fill={color} />
      </marker>
    </defs>
  );
}

/* ─── Mini Graph SVG ─────────────────────────────────────────────────────── */

function GraphSVG({ preset, highlight }: { preset: GraphPreset; highlight: boolean }) {
  const nodeColor = highlight
    ? { fill: "#6366f1", stroke: "#4338ca", text: "#fff" }
    : { fill: "#94a3b8", stroke: "#64748b", text: "#fff" };

  const edgeColor = highlight ? "#6366f1" : "#94a3b8";
  const weightColor = highlight ? "#f59e0b" : "#94a3b8";

  const markerId = `arrow-${preset.id}-${highlight ? "hi" : "lo"}`;

  return (
    <svg viewBox="0 0 220 180" className="w-full h-full" style={{ overflow: "visible" }}>
      {preset.directed && <ArrowMarker id={markerId} color={edgeColor} />}

      {/* Edges */}
      {preset.edges.map((e, i) => {
        const src = preset.nodes[e.u];
        const dst = preset.nodes[e.v];
        // Offset endpoints so arrow doesn't overlap node circle (r=18)
        const dx = dst.x - src.x, dy = dst.y - src.y;
        const len = Math.sqrt(dx * dx + dy * dy);
        const nx = dx / len, ny = dy / len;
        const r = 17;
        const x1 = src.x + nx * r, y1 = src.y + ny * r;
        const x2 = dst.x - nx * (r + (e.directed ? 6 : 0));
        const y2 = dst.y - ny * (r + (e.directed ? 6 : 0));

        // Weight label midpoint + slight offset
        const mx = (src.x + dst.x) / 2 - ny * 10;
        const my = (src.y + dst.y) / 2 + nx * 10;

        return (
          <g key={i}>
            <line
              x1={x1} y1={y1} x2={x2} y2={y2}
              stroke={edgeColor}
              strokeWidth={highlight ? 2.2 : 1.5}
              markerEnd={e.directed ? `url(#${markerId})` : undefined}
              strokeLinecap="round"
              opacity={highlight ? 1 : 0.5}
            />
            {e.w !== undefined && (
              <text x={mx} y={my} textAnchor="middle" dominantBaseline="middle"
                fontSize="10" fontWeight="700" fill={weightColor}>
                {e.w}
              </text>
            )}
          </g>
        );
      })}

      {/* Nodes */}
      {preset.nodes.map(node => (
        <g key={node.id}>
          <circle cx={node.x} cy={node.y} r={17}
            fill={nodeColor.fill} stroke={nodeColor.stroke} strokeWidth={2}
            opacity={highlight ? 1 : 0.5}
          />
          <text x={node.x} y={node.y} textAnchor="middle" dominantBaseline="middle"
            fontSize={node.label.length > 1 ? "9" : "12"} fontWeight="700"
            fill={nodeColor.text}>
            {node.label}
          </text>
        </g>
      ))}
    </svg>
  );
}

/* ─── Property Badge ─────────────────────────────────────────────────────── */

interface BadgeSpec {
  label: string;
  value: string;
  positive: boolean;    // green vs slate color scheme
  description: string;
}

function buildBadges(p: GraphPreset): BadgeSpec[] {
  const connMap: Record<string, { val: string; pos: boolean; desc: string }> = {
    "connected":        { val: "连通图", pos: true,  desc: "任意两顶点之间都存在路径（无向图）" },
    "strongly-connected":{ val: "强连通图", pos: true, desc: "任意两顶点 u→v 与 v→u 均有有向路径" },
    "weakly-connected": { val: "弱连通图", pos: false, desc: "忽略边方向后连通，但存在单向孤立区域" },
    "disconnected":     { val: "非连通图", pos: false, desc: "存在至少两个顶点之间没有任何路径" },
  };
  const conn = connMap[p.connectivity];

  const edgeCnt = p.edges.length;
  const nodeCnt = p.nodes.length;
  const maxEdges = nodeCnt * (nodeCnt - 1) / (p.directed ? 1 : 2);
  const edgeDensity = `E=${edgeCnt} / max=${maxEdges}`;

  return [
    {
      label: "方向性",
      value: p.directed ? "有向图" : "无向图",
      positive: p.directed,
      description: p.directed
        ? "边有方向 (u→v)，邻接表一条边只存一次"
        : "边无方向 {u,v}，邻接表每条边需存两次",
    },
    {
      label: "权重",
      value: p.weighted ? "加权图" : "无权图",
      positive: p.weighted,
      description: p.weighted
        ? "每条边附带权重 w，适用 Dijkstra / Bellman-Ford 等算法"
        : "边只有存在/不存在两种状态，适用 BFS 最短路",
    },
    {
      label: "连通性",
      value: conn.val,
      positive: conn.pos,
      description: conn.desc,
    },
    {
      label: "环",
      value: p.hasCycle ? "有环" : "无环（DAG）",
      positive: !p.hasCycle,
      description: p.hasCycle
        ? "存在有向/无向环路，不能直接拓扑排序"
        : "有向无环图，存在拓扑排序，适合 DP 与依赖解析",
    },
    {
      label: "稀疏/稠密",
      value: p.density === "sparse" ? "稀疏图" : "稠密图",
      positive: p.density === "sparse",
      description: p.density === "sparse"
        ? `E ≪ V²（${edgeDensity}），推荐邻接表存储，节省空间`
        : `E ≈ V²（${edgeDensity}），邻接矩阵 O(1) 判边优势更明显`,
    },
    {
      label: "推荐存储",
      value: p.density === "dense" ? "邻接矩阵" : "邻接表",
      positive: true,
      description: p.density === "dense"
        ? "稠密图：邻接矩阵 O(1) 判边，遍历代价 O(V²) 可接受"
        : "稀疏图：邻接表 O(V+E) 空间，遍历邻居 O(deg(u)) 高效",
    },
  ];
}

/* ─── Main Component ─────────────────────────────────────────────────────── */

export default function GraphTypeClassifier() {
  const [selected, setSelected] = useState(0);
  const preset = PRESETS[selected];
  const badges = buildBadges(preset);

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">

      {/* ── Header ── */}
      <div className="bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500 p-5">
        <h3 className="text-white font-bold text-lg tracking-tight">图类型分类器</h3>
        <p className="text-emerald-100 text-sm mt-0.5">
          选择不同场景的图，探索各个维度的分类属性
        </p>
      </div>

      {/* ── Scene selector tabs ── */}
      <div className="flex flex-wrap gap-2 p-4 border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40">
        {PRESETS.map((p, i) => (
          <button
            key={p.id}
            onClick={() => setSelected(i)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
              i === selected
                ? "bg-white dark:bg-slate-800 border-teal-400 dark:border-teal-500 text-teal-700 dark:text-teal-300 shadow-sm"
                : "border-transparent text-slate-500 dark:text-slate-400 hover:bg-white dark:hover:bg-slate-800 hover:border-slate-200 dark:hover:border-slate-600"
            }`}>
            <span>{p.emoji}</span>
            <span>{p.name}</span>
          </button>
        ))}
      </div>

      <div className="p-5 grid grid-cols-1 sm:grid-cols-2 gap-5">

        {/* ── Left: Graph visualization ── */}
        <div className="flex flex-col gap-3">
          <div className="rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 p-3 aspect-[11/9] flex items-center justify-center relative">
            <GraphSVG preset={preset} highlight={true} />
            {/* scene label */}
            <div className="absolute top-2 left-2 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm rounded-lg px-2 py-1 text-[10px] font-semibold text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
              {preset.emoji} {preset.scene}
            </div>
          </div>

          {/* Graph stats */}
          <div className="grid grid-cols-2 gap-2">
            {[
              { label: "顶点数 |V|", val: preset.nodes.length, color: "text-indigo-600 dark:text-indigo-400" },
              { label: "边数 |E|", val: preset.edges.length, color: "text-purple-600 dark:text-purple-400" },
            ].map(({ label, val, color }) => (
              <div key={label} className="rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 p-3 text-center">
                <div className="text-[10px] text-slate-400 dark:text-slate-500 font-medium uppercase">{label}</div>
                <div className={`text-2xl font-bold font-mono mt-0.5 ${color}`}>{val}</div>
              </div>
            ))}
          </div>

          {/* Description */}
          <div className="rounded-xl bg-teal-50 dark:bg-teal-900/20 border border-teal-200 dark:border-teal-800 p-3 text-xs text-teal-700 dark:text-teal-300 leading-relaxed">
            {preset.description}
          </div>
        </div>

        {/* ── Right: Classification badges ── */}
        <div className="flex flex-col gap-2">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">
            六维分类属性
          </div>
          {badges.map(badge => (
            <div key={badge.label}
              className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/60 p-3">
              <div className="flex items-start justify-between gap-2">
                <div className="text-[10px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wide">
                  {badge.label}
                </div>
                <span className={`flex-none text-[11px] font-bold px-2.5 py-0.5 rounded-full border ${
                  badge.positive
                    ? "bg-emerald-100 dark:bg-emerald-900/40 border-emerald-300 dark:border-emerald-700 text-emerald-700 dark:text-emerald-300"
                    : "bg-amber-100 dark:bg-amber-900/30 border-amber-300 dark:border-amber-700 text-amber-700 dark:text-amber-300"
                }`}>
                  {badge.value}
                </span>
              </div>
              <div className="mt-1.5 text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
                {badge.description}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
