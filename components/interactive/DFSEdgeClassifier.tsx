"use client";
import React, { useState } from "react";

/* ─── Graph Definition ───────────────────────────────────────────────────── */
// Directed graph with all 4 DFS edge types
// Adj: 0→[1,4,3], 1→[2,3], 2→[1], 4→[3]
// DFS result:  d[0]=1,f[0]=10  d[1]=2,f[1]=7  d[2]=3,f[2]=4
//              d[3]=5,f[3]=6   d[4]=8,f[4]=9

interface Node { id: number; x: number; y: number; label: string; }

const NODES: Node[] = [
  { id: 0, x: 60,  y: 130, label: "0" },
  { id: 1, x: 165, y: 60,  label: "1" },
  { id: 2, x: 270, y: 60,  label: "2" },
  { id: 3, x: 270, y: 130, label: "3" },
  { id: 4, x: 165, y: 200, label: "4" },
];

type EdgeKind = "tree" | "back" | "forward" | "cross";

interface DFSEdge {
  u: number;
  v: number;
  kind: EdgeKind;
  d_u: number; f_u: number;
  d_v: number; f_v: number;
}

const EDGES: DFSEdge[] = [
  { u: 0, v: 1, kind: "tree",    d_u:1,f_u:10, d_v:2,f_v:7 },
  { u: 0, v: 4, kind: "tree",    d_u:1,f_u:10, d_v:8,f_v:9 },
  { u: 0, v: 3, kind: "forward", d_u:1,f_u:10, d_v:5,f_v:6 },
  { u: 1, v: 2, kind: "tree",    d_u:2,f_u:7,  d_v:3,f_v:4 },
  { u: 1, v: 3, kind: "tree",    d_u:2,f_u:7,  d_v:5,f_v:6 },
  { u: 2, v: 1, kind: "back",    d_u:3,f_u:4,  d_v:2,f_v:7 },
  { u: 4, v: 3, kind: "cross",   d_u:8,f_u:9,  d_v:5,f_v:6 },
];

const TIMESTAMPS = [
  { id: 0, d: 1,  f: 10 },
  { id: 1, d: 2,  f: 7  },
  { id: 2, d: 3,  f: 4  },
  { id: 3, d: 5,  f: 6  },
  { id: 4, d: 8,  f: 9  },
];

/* ─── Edge type metadata ─────────────────────────────────────────────────── */
interface EdgeTypeInfo {
  kind: EdgeKind;
  label: string;
  color: string;
  strokeColor: string;
  bg: string;
  border: string;
  badge: string;
  condition: string;
  meaning: string;
  occurrence: string;
}

const EDGE_TYPES: EdgeTypeInfo[] = [
  {
    kind: "tree",
    label: "树边 Tree Edge",
    color: "#6366f1",
    strokeColor: "#6366f1",
    bg: "bg-indigo-50 dark:bg-indigo-900/20",
    border: "border-indigo-300 dark:border-indigo-700",
    badge: "bg-indigo-500 text-white",
    condition: "处理边 (u,v) 时，v 为 白色（未访问）",
    meaning: "DFS 树的骨架边，v 首次被 u 发现",
    occurrence: "有向图 ✓  无向图 ✓",
  },
  {
    kind: "back",
    label: "后向边 Back Edge",
    color: "#ef4444",
    strokeColor: "#ef4444",
    bg: "bg-rose-50 dark:bg-rose-900/20",
    border: "border-rose-300 dark:border-rose-700",
    badge: "bg-rose-500 text-white",
    condition: "处理边 (u,v) 时，v 为 灰色（DFS 栈中）",
    meaning: "v 是 u 的祖先，形成环！有向图中 back edge ⟺ 有环",
    occurrence: "有向图 ✓  无向图 ✓（无向图的非树边都是后向边）",
  },
  {
    kind: "forward",
    label: "前向边 Forward Edge",
    color: "#10b981",
    strokeColor: "#10b981",
    bg: "bg-emerald-50 dark:bg-emerald-900/20",
    border: "border-emerald-300 dark:border-emerald-700",
    badge: "bg-emerald-500 text-white",
    condition: "处理边 (u,v) 时，v 为 黑色 且 d[u] < d[v]（v 是 u 的后代）",
    meaning: "从祖先 u 指向后代 v，但不走树边（跨越了中间层）",
    occurrence: "有向图 ✓  无向图 ✗（无向图无前向边）",
  },
  {
    kind: "cross",
    label: "横跨边 Cross Edge",
    color: "#f59e0b",
    strokeColor: "#f59e0b",
    bg: "bg-amber-50 dark:bg-amber-900/20",
    border: "border-amber-300 dark:border-amber-700",
    badge: "bg-amber-500 text-white",
    condition: "处理边 (u,v) 时，v 为 黑色 且 d[u] > d[v]（v 先于 u 完成）",
    meaning: "u 与 v 无祖先-后代关系，跨越不同子树",
    occurrence: "有向图 ✓  无向图 ✗（无向图无横跨边）",
  },
];

/* ─── SVG Arrow helper ───────────────────────────────────────────────────── */
function arrowPoints(u: number, v: number): { x1: number; y1: number; x2: number; y2: number } {
  const nu = NODES[u], nv = NODES[v];
  const dx = nv.x - nu.x, dy = nv.y - nu.y;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const r = 18;
  return {
    x1: nu.x + (dx / dist) * r,
    y1: nu.y + (dy / dist) * r,
    x2: nv.x - (dx / dist) * r,
    y2: nv.y - (dy / dist) * r,
  };
}

// Slight curve offset for edges that might overlap
function curveOffset(edge: DFSEdge): number {
  // back edge 2→1 and tree edge 1→2 share same nodes
  if ((edge.u === 2 && edge.v === 1) || (edge.u === 1 && edge.v === 2)) return 18;
  return 0;
}

/* ─── Bracket visual ─────────────────────────────────────────────────────── */
function NodeBracket({ highlight }: { highlight: EdgeKind | null }) {
  const MAX = 10;
  const colors = ["#6366f1", "#0ea5e9", "#10b981", "#f59e0b", "#ec4899"];

  // Which edges involve which nodes for highlight
  const relevantNodes = highlight
    ? new Set(EDGES.filter(e => e.kind === highlight).flatMap(e => [e.u, e.v]))
    : null;

  return (
    <div className="space-y-1">
      {TIMESTAMPS.map(ts => {
        const node = NODES[ts.id];
        const col = colors[ts.id];
        const faded = relevantNodes && !relevantNodes.has(ts.id);
        const leftPct = ((ts.d - 1) / MAX) * 100;
        const wPct = ((ts.f - ts.d + 1) / MAX) * 100;
        return (
          <div key={ts.id} className={`flex items-center gap-2 transition-opacity ${faded ? "opacity-25" : ""}`}>
            <div className="w-4 text-[10px] font-bold text-center" style={{ color: col }}>{node.label}</div>
            <div className="flex-1 relative h-4 bg-slate-100 dark:bg-slate-800 rounded overflow-hidden">
              <div className="absolute inset-0 flex">
                {Array.from({ length: MAX }, (_, i) => (
                  <div key={i} className="flex-1 border-r border-slate-200 dark:border-slate-700 last:border-r-0" />
                ))}
              </div>
              <div
                className="absolute top-0.5 h-3 rounded flex items-center justify-center transition-all duration-300"
                style={{ left: `${leftPct}%`, width: `${wPct}%`, backgroundColor: col }}>
                <span className="text-[8px] text-white font-bold">[{ts.d},{ts.f}]</span>
              </div>
            </div>
          </div>
        );
      })}
      <div className="flex ml-6 text-[8px] text-slate-400 dark:text-slate-600">
        {Array.from({ length: MAX + 1 }, (_, i) => (
          <div key={i} className="flex-1 text-center">{i}</div>
        ))}
      </div>
    </div>
  );
}

/* ─── Main Component ─────────────────────────────────────────────────────── */
export default function DFSEdgeClassifier() {
  const [activeKind, setActiveKind] = useState<EdgeKind | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<number | null>(null);

  const effectiveKind = hoveredEdge !== null ? EDGES[hoveredEdge].kind : activeKind;
  const activeInfo = EDGE_TYPES.find(t => t.kind === effectiveKind);

  function isActive(edge: DFSEdge, idx: number) {
    if (hoveredEdge === idx) return true;
    if (effectiveKind === null) return true;
    return edge.kind === effectiveKind;
  }

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">DFS 边分类可视化</h3>
        <p className="text-orange-100 text-sm mt-0.5">点击边类型卡片or图中边，观察四类边的定义、识别条件与括号区间关系</p>
      </div>

      <div className="p-5 space-y-4">
        {/* Edge type selector tabs */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {EDGE_TYPES.map(et => (
            <button key={et.kind}
              onClick={() => setActiveKind(prev => prev === et.kind ? null : et.kind)}
              className={`rounded-xl px-3 py-2 text-left border transition-all ${
                effectiveKind === et.kind
                  ? `${et.bg} ${et.border} shadow-sm`
                  : "border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800"
              }`}>
              <div className="flex items-center gap-1.5 mb-1">
                <span className={`inline-block w-2.5 h-2.5 rounded-sm`} style={{ backgroundColor: et.color }} />
                <span className="text-[10px] font-bold text-slate-600 dark:text-slate-300 leading-tight">
                  {et.label.split(" ")[0]} {et.label.split(" ")[1]}
                </span>
              </div>
              <div className="text-[9px] text-slate-400 dark:text-slate-500 leading-tight">
                {EDGES.filter(e => e.kind === et.kind).length} 条边
              </div>
            </button>
          ))}
        </div>

        <div className="flex flex-col sm:flex-row gap-4">
          {/* SVG */}
          <div className="sm:w-[320px] flex-shrink-0 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-2">
            <svg viewBox="0 0 330 255" className="w-full">
              <defs>
                {EDGE_TYPES.map(et => (
                  <marker key={et.kind} id={`arrow-${et.kind}`}
                    markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                    <path d="M0,0 L0,6 L8,3 z" fill={et.color} />
                  </marker>
                ))}
                <marker id="arrow-dim" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L8,3 z" fill="#d1d5db" />
                </marker>
              </defs>

              {EDGES.map((edge, idx) => {
                const { x1, y1, x2, y2 } = arrowPoints(edge.u, edge.v);
                const active = isActive(edge, idx);
                const col = EDGE_TYPES.find(t => t.kind === edge.kind)!;
                const offset = curveOffset(edge);
                // Curve offset for overlapping edges
                const mx = (x1 + x2) / 2 + (offset ? -offset : 0);
                const my = (y1 + y2) / 2 + (offset ? offset : 0);
                return (
                  <g key={idx}
                    onMouseEnter={() => setHoveredEdge(idx)}
                    onMouseLeave={() => setHoveredEdge(null)}
                    className="cursor-pointer">
                    {offset ? (
                      <path
                        d={`M${x1},${y1} Q${mx},${my} ${x2},${y2}`}
                        fill="none"
                        stroke={active ? col.color : "#e2e8f0"}
                        strokeWidth={active ? (hoveredEdge === idx ? 3.5 : 2.5) : 1.5}
                        strokeDasharray={edge.kind === "forward" || edge.kind === "cross" ? "6 3" : "none"}
                        markerEnd={active ? `url(#arrow-${edge.kind})` : "url(#arrow-dim)"}
                        className="transition-all duration-200"
                      />
                    ) : (
                      <line
                        x1={x1} y1={y1} x2={x2} y2={y2}
                        stroke={active ? col.color : "#e2e8f0"}
                        strokeWidth={active ? (hoveredEdge === idx ? 3.5 : 2.5) : 1.5}
                        strokeDasharray={edge.kind === "forward" || edge.kind === "cross" ? "6 3" : "none"}
                        markerEnd={active ? `url(#arrow-${edge.kind})` : "url(#arrow-dim)"}
                        className="transition-all duration-200"
                      />
                    )}
                    {/* Edge type label on hover */}
                    {hoveredEdge === idx && (
                      <text x={(x1+x2)/2 + (offset ? -offset-5 : 5)} y={(y1+y2)/2}
                        fontSize={9} fontWeight="bold" fill={col.color} textAnchor="middle">
                        {edge.kind}
                      </text>
                    )}
                  </g>
                );
              })}

              {/* Nodes */}
              {NODES.map(node => {
                const ts = TIMESTAMPS[node.id];
                const faded = effectiveKind !== null && !EDGES.filter(e=>e.kind===effectiveKind).some(e=>e.u===node.id||e.v===node.id);
                return (
                  <g key={node.id} className={`transition-opacity ${faded ? "opacity-30" : ""}`}>
                    <circle cx={node.x} cy={node.y} r={18}
                      fill="#6366f1" stroke="#4338ca" strokeWidth={2} />
                    <text x={node.x} y={node.y} textAnchor="middle" dominantBaseline="central"
                      fontSize={13} fontWeight="bold" fill="white">{node.label}</text>
                    {/* timestamps */}
                    <text x={node.x} y={node.y + 30} textAnchor="middle" fontSize={9} fill="#6b7280">
                      [{ts.d}/{ts.f}]
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Right: info + brackets */}
          <div className="flex-1 space-y-3">
            {/* Active edge type info */}
            {activeInfo ? (
              <div className={`rounded-xl border p-3 space-y-2 ${activeInfo.bg} ${activeInfo.border}`}>
                <div className="flex items-center gap-2">
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${activeInfo.badge}`}>{activeInfo.label}</span>
                </div>
                <div className="space-y-1.5 text-[11px]">
                  <div>
                    <span className="font-bold text-slate-600 dark:text-slate-300">识别条件：</span>
                    <span className="text-slate-600 dark:text-slate-300"> {activeInfo.condition}</span>
                  </div>
                  <div>
                    <span className="font-bold text-slate-600 dark:text-slate-300">含义：</span>
                    <span className="text-slate-600 dark:text-slate-300"> {activeInfo.meaning}</span>
                  </div>
                  <div>
                    <span className="font-bold text-slate-600 dark:text-slate-300">出现场景：</span>
                    <span className="text-slate-600 dark:text-slate-300"> {activeInfo.occurrence}</span>
                  </div>
                </div>
                {/* Edges of this type */}
                <div className="flex flex-wrap gap-1 mt-1">
                  {EDGES.filter(e => e.kind === activeInfo.kind).map((e, i) => (
                    <span key={i} className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold ${activeInfo.badge}`}>
                      ({NODES[e.u].label}→{NODES[e.v].label})
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-3 text-[11px] text-slate-500 dark:text-slate-400">
                <div className="font-bold mb-1 text-slate-600 dark:text-slate-300">本图中所有边：</div>
                <div className="space-y-1">
                  {EDGES.map((e, i) => {
                    const col = EDGE_TYPES.find(t => t.kind === e.kind)!;
                    return (
                      <div key={i} className="flex items-center gap-2">
                        <span className="inline-block w-2 h-2 rounded-sm flex-shrink-0" style={{ backgroundColor: col.color }} />
                        <span className="font-mono">({NODES[e.u].label}→{NODES[e.v].label})</span>
                        <span className="ml-auto text-[10px]" style={{ color: col.color }}>{e.kind}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Bracket timeline */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-3">
              <div className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
                括号区间 — 突出显示相关节点
              </div>
              <NodeBracket highlight={effectiveKind} />
            </div>
          </div>
        </div>

        {/* Key insight */}
        <div className="rounded-xl bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 px-4 py-3 text-[11px] text-orange-700 dark:text-orange-300 leading-relaxed">
          <strong>💡 核心规律：</strong>
          无向图 DFS <strong>只有树边和后向边</strong>（无前向边、无横跨边）——
          因为无向图中如果存在横跨边 (u,v)，v 在被 u 处理时必然是白色，从而成为树边，矛盾！
          有向图才会出现全部四类边。
        </div>
      </div>
    </div>
  );
}
