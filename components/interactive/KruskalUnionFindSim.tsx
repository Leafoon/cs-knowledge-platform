"use client";
import React, { useState, useMemo } from "react";

// ─── Graph definition ────────────────────────────────────────────────────────
// 6 nodes, 9 edges (bidirectional, displayed with positions)
const NODE_COUNT = 6;

const GRAPH_NODES: { id: number; label: string; x: number; y: number }[] = [
  { id: 0, label: "A", x: 100, y:  60 },
  { id: 1, label: "B", x: 260, y:  40 },
  { id: 2, label: "C", x: 400, y: 100 },
  { id: 3, label: "D", x:  80, y: 190 },
  { id: 4, label: "E", x: 250, y: 200 },
  { id: 5, label: "F", x: 420, y: 210 },
];

interface Edge { id: number; u: number; v: number; w: number }

const ALL_EDGES: Edge[] = [
  { id: 0, u: 0, v: 1, w: 4  },
  { id: 1, u: 0, v: 3, w: 6  },
  { id: 2, u: 1, v: 2, w: 3  },
  { id: 3, u: 1, v: 4, w: 5  },
  { id: 4, u: 2, v: 5, w: 2  },
  { id: 5, u: 3, v: 4, w: 8  },
  { id: 6, u: 4, v: 5, w: 7  },
  { id: 7, u: 3, v: 0, w: 9  },
  { id: 8, u: 1, v: 5, w: 11 },
];

// Sorted edges for Kruskal
const SORTED_EDGES = [...ALL_EDGES].sort((a, b) => a.w - b.w);

// ─── DSU (simple, for Kruskal) ────────────────────────────────────────────────
function dsUFind(parent: number[], x: number): number {
  while (parent[x] !== x) x = parent[x];
  return x;
}
function dsUUnite(parent: number[], rank: number[], x: number, y: number): boolean {
  const rx = dsUFind(parent, x), ry = dsUFind(parent, y);
  if (rx === ry) return false;
  if (rank[rx] < rank[ry]) parent[rx] = ry;
  else if (rank[rx] > rank[ry]) parent[ry] = rx;
  else { parent[ry] = rx; rank[rx]++; }
  return true;
}

// ─── Build simulation steps ────────────────────────────────────────────────────
type EdgeStatus = "pending" | "added" | "rejected" | "current";

interface SimStep {
  edgeIdx: number;            // which sorted edge we consider
  parent: number[];
  rankArr: number[];
  mstEdges: number[];         // edge IDs added to MST
  rejectedEdges: number[];
  totalWeight: number;
  accepted: boolean;
  reason: string;
  ufForest: number[];         // parent snapshot for UF display
}

function buildSimulation(): SimStep[] {
  const parent = Array.from({ length: NODE_COUNT }, (_, i) => i);
  const rankArr = new Array(NODE_COUNT).fill(0);
  const mstEdges: number[] = [];
  const rejectedEdges: number[] = [];
  let totalWeight = 0;
  const steps: SimStep[] = [];

  for (let i = 0; i < SORTED_EDGES.length; i++) {
    const edge = SORTED_EDGES[i];
    const ru = dsUFind(parent, edge.u), rv = dsUFind(parent, edge.v);
    const p2 = [...parent], r2 = [...rankArr];

    if (ru !== rv) {
      dsUUnite(p2, r2, edge.u, edge.v);
      mstEdges.push(edge.id);
      totalWeight += edge.w;
      steps.push({
        edgeIdx: i,
        parent: p2, rankArr: r2,
        mstEdges: [...mstEdges],
        rejectedEdges: [...rejectedEdges],
        totalWeight,
        accepted: true,
        reason: `FIND(${GRAPH_NODES[edge.u].label})=${ru}, FIND(${GRAPH_NODES[edge.v].label})=${rv}，根不同 → 加入 MST，UNION(${ru}, ${rv})`,
        ufForest: p2,
      });
      parent.splice(0, parent.length, ...p2);
      rankArr.splice(0, rankArr.length, ...r2);
    } else {
      rejectedEdges.push(edge.id);
      steps.push({
        edgeIdx: i,
        parent: [...parent], rankArr: [...rankArr],
        mstEdges: [...mstEdges],
        rejectedEdges: [...rejectedEdges],
        totalWeight,
        accepted: false,
        reason: `FIND(${GRAPH_NODES[edge.u].label})=FIND(${GRAPH_NODES[edge.v].label})=${ru}，已连通 → 会成环，舍弃！`,
        ufForest: [...parent],
      });
    }
  }
  return steps;
}

const SIM_STEPS = buildSimulation();
const OPTIMAL_WEIGHT = SIM_STEPS[SIM_STEPS.length - 1].totalWeight;

// ─── Helper: get edge status at given step ────────────────────────────────────
function getEdgeStatus(edgeId: number, step: SimStep | null, currentSortedIdx: number, sortedIdx: number): EdgeStatus {
  if (!step) return "pending";
  if (step.mstEdges.includes(edgeId)) return "added";
  if (step.rejectedEdges.includes(edgeId)) return "rejected";
  if (sortedIdx === currentSortedIdx) return "current";
  return "pending";
}

// ─── UF Forest layout ─────────────────────────────────────────────────────────
function computeUFLayout(parent: number[]) {
  const n = parent.length;
  const children: number[][] = Array.from({ length: n }, () => []);
  for (let i = 0; i < n; i++) if (parent[i] !== i) children[parent[i]].push(i);

  const W = 480, H = 100;
  const roots = Array.from({ length: n }, (_, i) => i).filter(i => parent[i] === i);
  const positions: { x: number; y: number }[] = Array(n);
  const slotW = W / Math.max(roots.length, 1);

  function assign(node: number, depth: number, sx: number, sw: number) {
    positions[node] = { x: sx + sw / 2, y: 18 + depth * 44 };
    const ch = children[node];
    if (!ch.length) return;
    const csw = sw / ch.length;
    ch.forEach((c, i) => assign(c, depth + 1, sx + i * csw, csw));
  }
  roots.forEach((r, i) => assign(r, 0, i * slotW, slotW));
  return { positions, children, parent };
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function KruskalUnionFindSim() {
  const [stepIdx, setStepIdx] = useState(-1);

  const step = stepIdx >= 0 ? SIM_STEPS[stepIdx] : null;
  const mstEdgeIds   = step?.mstEdges      ?? [];
  const rejectIds    = step?.rejectedEdges ?? [];
  const currentEdge  = step ? SORTED_EDGES[step.edgeIdx] : null;
  const totalWeight  = step?.totalWeight   ?? 0;

  const ufParent = step?.ufForest ?? Array.from({ length: NODE_COUNT }, (_, i) => i);
  const { positions: ufPos } = computeUFLayout(ufParent);

  const progress = stepIdx < 0 ? 0 : ((stepIdx + 1) / SIM_STEPS.length) * 100;
  const mstDone = mstEdgeIds.length === NODE_COUNT - 1;

  // ── Graph SVG helpers ───────────────────────────────────────────────────────
  const GRAPH_W = 500, GRAPH_H = 260;
  const UF_W = 480, UF_H = 110;

  function getEdgeColor(edge: Edge): string {
    if (mstEdgeIds.includes(edge.id)) return "#10b981";          // emerald: in MST
    if (rejectIds.includes(edge.id)) return "#ef4444";           // red: rejected
    if (currentEdge?.id === edge.id) return "#f59e0b";           // amber: current
    return "#94a3b8";                                             // slate: pending
  }
  function getEdgeWidth(edge: Edge): number {
    if (mstEdgeIds.includes(edge.id)) return 3;
    if (currentEdge?.id === edge.id) return 2.5;
    if (rejectIds.includes(edge.id)) return 1.5;
    return 1.5;
  }
  function getEdgeDash(edge: Edge): string {
    if (rejectIds.includes(edge.id)) return "5,3";
    return "none";
  }

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5 my-6 shadow-sm">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-2 mb-1">
        <h3 className="text-base font-bold text-slate-800 dark:text-slate-100">
          🔗 Kruskal 算法 × 并查集判环
        </h3>
        <div className="flex gap-2 flex-wrap">
          <span className={`text-xs px-2 py-0.5 rounded-full border font-mono ${
            mstDone ? "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-300 dark:border-emerald-700 text-emerald-600 dark:text-emerald-400"
                    : "bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400"
          }`}>
            MST 权重 {totalWeight}{mstDone ? ` / 最优 ${OPTIMAL_WEIGHT} ✓` : ""}
          </span>
          <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 font-mono">
            {mstEdgeIds.length}/{NODE_COUNT - 1} 条边
          </span>
        </div>
      </div>
      <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
        Kruskal 算法：将所有边按权重升序排列，依次考察；用并查集判断两端点是否已连通——<span className="text-emerald-500 dark:text-emerald-400 font-semibold">加入 MST</span> 还是
        <span className="text-red-500 dark:text-red-400 font-semibold">丢弃（成环）</span>。时间复杂度 $O(E \log E)$。
      </p>

      {/* Main layout: graph + sorted edge list */}
      <div className="flex gap-4 mb-4">
        {/* Graph SVG */}
        <div className="flex-1 min-w-0">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1.5">原始图</div>
          <div className="bg-slate-50 dark:bg-slate-800/60 rounded-xl overflow-hidden">
            <svg viewBox={`0 0 ${GRAPH_W} ${GRAPH_H}`} className="w-full" style={{ height: GRAPH_H }}>
              {/* Edges */}
              {ALL_EDGES.map(edge => {
                const u = GRAPH_NODES[edge.u], v = GRAPH_NODES[edge.v];
                const mx = (u.x + v.x) / 2, my = (u.y + v.y) / 2;
                const color = getEdgeColor(edge);
                return (
                  <g key={edge.id}>
                    <line x1={u.x} y1={u.y} x2={v.x} y2={v.y}
                      stroke={color} strokeWidth={getEdgeWidth(edge)}
                      strokeDasharray={getEdgeDash(edge)} opacity={0.9} />
                    <rect x={mx - 11} y={my - 9} width={22} height={16} rx={4}
                      fill={color === "#94a3b8" ? "white" : color} opacity={color === "#94a3b8" ? 0.9 : 0.12}
                      stroke={color} strokeWidth={0.5}
                    />
                    <text x={mx} y={my + 1} textAnchor="middle" dominantBaseline="middle"
                      fill={color === "#94a3b8" ? "#64748b" : color} fontSize={10} fontWeight="bold">
                      {edge.w}
                    </text>
                  </g>
                );
              })}
              {/* Nodes */}
              {GRAPH_NODES.map(nd => {
                const isCurrent = currentEdge && (currentEdge.u === nd.id || currentEdge.v === nd.id);
                const isInMST = mstEdgeIds.some(eid => {
                  const e = ALL_EDGES.find(e => e.id === eid);
                  return e && (e.u === nd.id || e.v === nd.id);
                });
                return (
                  <g key={nd.id}>
                    <circle cx={nd.x} cy={nd.y} r={20}
                      fill={isCurrent ? "#f59e0b" : isInMST ? "#10b981" : "#64748b"}
                      opacity={0.92}
                      stroke={isCurrent ? "#fcd34d" : isInMST ? "#6ee7b7" : "none"}
                      strokeWidth={2.5}
                    />
                    <text x={nd.x} y={nd.y + 1} textAnchor="middle" dominantBaseline="middle"
                      fill="white" fontSize={13} fontWeight="bold">{nd.label}</text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        {/* Sorted edge list */}
        <div className="w-44 shrink-0">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1.5">排序边表</div>
          <div className="space-y-1 overflow-y-auto max-h-60">
            {SORTED_EDGES.map((edge, i) => {
              const isCur = currentEdge?.id === edge.id;
              const inMST = mstEdgeIds.includes(edge.id);
              const rejected = rejectIds.includes(edge.id);
              return (
                <div key={edge.id} className={`flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs transition-colors ${
                  isCur    ? "bg-amber-100 dark:bg-amber-900/30 border border-amber-300 dark:border-amber-700" :
                  inMST    ? "bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800" :
                  rejected ? "bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 opacity-60" :
                  "bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700"
                }`}>
                  <span className="font-mono text-slate-400 dark:text-slate-500 w-4 text-right shrink-0">{i + 1}.</span>
                  <span className={`font-semibold font-mono ${
                    isCur ? "text-amber-700 dark:text-amber-300" :
                    inMST ? "text-emerald-700 dark:text-emerald-300" :
                    rejected ? "text-red-500 dark:text-red-400 line-through" :
                    "text-slate-700 dark:text-slate-200"
                  }`}>
                    {GRAPH_NODES[edge.u].label}–{GRAPH_NODES[edge.v].label}
                  </span>
                  <span className={`ml-auto font-mono text-xs ${
                    isCur ? "text-amber-600 dark:text-amber-400" :
                    inMST ? "text-emerald-600 dark:text-emerald-400" :
                    "text-slate-500 dark:text-slate-400"
                  }`}>{edge.w}</span>
                  {inMST && <span className="text-emerald-500 text-xs">✓</span>}
                  {rejected && <span className="text-red-400 text-xs">✗</span>}
                  {isCur && <span className="text-amber-500 text-xs">▶</span>}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Step description */}
      {step ? (
        <div className={`mb-4 rounded-lg border px-4 py-2.5 ${
          step.accepted
            ? "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800"
            : "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800"
        }`}>
          <div className={`text-xs font-bold mb-0.5 ${step.accepted ? "text-emerald-700 dark:text-emerald-300" : "text-red-600 dark:text-red-400"}`}>
            边 {GRAPH_NODES[currentEdge!.u].label}–{GRAPH_NODES[currentEdge!.v].label}（权重 {currentEdge!.w}）→ {step.accepted ? "加入 MST ✅" : "成环，舍弃 ❌"}
          </div>
          <div className={`text-xs ${step.accepted ? "text-emerald-600 dark:text-emerald-400" : "text-red-500 dark:text-red-400"}`}>
            {step.reason}
          </div>
        </div>
      ) : (
        <div className="mb-4 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 px-4 py-2.5">
          <div className="text-xs text-slate-500 dark:text-slate-400">
            已将 {ALL_EDGES.length} 条边升序排列，点击「下一步」依次考察每条边，并查集实时判断是否成环。
          </div>
        </div>
      )}

      {/* Union-Find Forest */}
      <div className="mb-4">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1.5">并查集森林（实时更新）</div>
        <div className="bg-slate-50 dark:bg-slate-800/60 rounded-xl overflow-hidden">
          <svg viewBox={`0 0 ${UF_W} ${UF_H}`} className="w-full" style={{ height: UF_H }}>
            {/* Edges */}
            {ufParent.map((p, i) => {
              if (p === i) return null;
              const fp = ufPos[p], fi = ufPos[i];
              return (
                <line key={`uf-e${i}`} x1={fp.x} y1={fp.y} x2={fi.x} y2={fi.y}
                  stroke="#6366f1" strokeWidth={1.5} opacity={0.6} />
              );
            })}
            {/* Nodes */}
            {Array.from({ length: NODE_COUNT }, (_, i) => {
              const { x, y } = ufPos[i];
              const isRoot = ufParent[i] === i;
              return (
                <g key={`uf-n${i}`}>
                  <circle cx={x} cy={y} r={17}
                    fill={isRoot ? "#10b981" : "#6366f1"} opacity={0.9}
                    stroke={isRoot ? "#6ee7b7" : "none"} strokeWidth={2}
                  />
                  <text x={x} y={y + 1} textAnchor="middle" dominantBaseline="middle"
                    fill="white" fontSize={12} fontWeight="bold">{GRAPH_NODES[i].label}</text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2 mb-3">
        <button onClick={() => setStepIdx(-1)}
          className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400
            text-xs hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
          ↺ 重置
        </button>
        <div className="flex-1 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
          <div className="h-full bg-indigo-500 rounded-full transition-all duration-300" style={{ width: `${progress}%` }} />
        </div>
        <button onClick={() => setStepIdx(s => Math.max(-1, s - 1))} disabled={stepIdx < 0}
          className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300
            text-xs disabled:opacity-30 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
          ◀ 上一步
        </button>
        <button onClick={() => setStepIdx(s => Math.min(SIM_STEPS.length - 1, s + 1))} disabled={stepIdx >= SIM_STEPS.length - 1}
          className="px-3 py-1.5 rounded-lg bg-indigo-600 text-white text-xs
            disabled:opacity-30 hover:bg-indigo-700 transition-colors">
          下一步 ▶
        </button>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-xs text-slate-500 dark:text-slate-400">
        {[
          { color: "bg-emerald-500", label: "已加入 MST" },
          { color: "bg-red-400",     label: "成环舍弃" },
          { color: "bg-amber-400",   label: "当前正在考察" },
          { color: "bg-slate-400",   label: "待处理" },
        ].map(l => (
          <span key={l.label} className="flex items-center gap-1.5">
            <span className={`w-3 h-3 rounded-full ${l.color} inline-block`} />
            {l.label}
          </span>
        ))}
      </div>
    </div>
  );
}
