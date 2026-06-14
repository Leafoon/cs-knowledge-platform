"use client";
import React, { useState } from "react";

/* ─── Fixed graph: 4 nodes, 5 undirected weighted edges ─────────────────── */

// Nodes
const NODES = [
  { id: 0, x: 55,  y: 55,  label: "0" },
  { id: 1, x: 175, y: 55,  label: "1" },
  { id: 2, x: 55,  y: 155, label: "2" },
  { id: 3, x: 175, y: 155, label: "3" },
];

// Undirected edges {u, v, w}
const EDGES = [
  { u: 0, v: 1, w: 4 },
  { u: 0, v: 2, w: 2 },
  { u: 1, v: 2, w: 3 },
  { u: 1, v: 3, w: 5 },
  { u: 2, v: 3, w: 1 },
];

const N = NODES.length;

// Adjacency matrix (0 = no edge)
const MAT: number[][] = Array.from({ length: N }, () => Array(N).fill(0));
EDGES.forEach(({ u, v, w }) => { MAT[u][v] = w; MAT[v][u] = w; });

// Adjacency list
const ADJ_LIST: { v: number; w: number }[][] = Array.from({ length: N }, () => []);
EDGES.forEach(({ u, v, w }) => {
  ADJ_LIST[u].push({ v, w });
  ADJ_LIST[v].push({ v: u, w });
});
ADJ_LIST.forEach(list => list.sort((a, b) => a.v - b.v));

type View = "matrix" | "list" | "edges";

/* ─── SVG Graph ──────────────────────────────────────────────────────────── */

function GraphSVG({
  activeNode,
  onNodeClick,
  highlightEdge,
}: {
  activeNode: number | null;
  onNodeClick: (id: number) => void;
  highlightEdge: { u: number; v: number } | null;
}) {
  const isActive = (id: number) => activeNode === id;
  const isRelated = (id: number) =>
    activeNode !== null && (MAT[activeNode][id] > 0 || MAT[id][activeNode] > 0);

  const isEdgeActive = (u: number, v: number) => {
    if (highlightEdge) return (highlightEdge.u === u && highlightEdge.v === v) || (highlightEdge.u === v && highlightEdge.v === u);
    if (activeNode === null) return false;
    return (u === activeNode || v === activeNode) && MAT[u][v] > 0;
  };

  return (
    <svg viewBox="0 0 230 210" className="w-full h-full">
      {/* Edges */}
      {EDGES.map(({ u, v, w }, i) => {
        const src = NODES[u], dst = NODES[v];
        const mx = (src.x + dst.x) / 2;
        const my = (src.y + dst.y) / 2;
        const active = isEdgeActive(u, v);
        return (
          <g key={i}>
            <line
              x1={src.x} y1={src.y} x2={dst.x} y2={dst.y}
              stroke={active ? "#f59e0b" : "#94a3b8"}
              strokeWidth={active ? 3 : 1.8}
              strokeLinecap="round"
              opacity={activeNode !== null && !active ? 0.3 : 0.8}
              className="transition-all duration-200"
            />
            {/* Weight badge */}
            <circle cx={mx} cy={my} r={10}
              fill={active ? "#fef3c7" : "#f1f5f9"}
              stroke={active ? "#f59e0b" : "#cbd5e1"}
              strokeWidth={1.5}
            />
            <text x={mx} y={my} textAnchor="middle" dominantBaseline="middle"
              fontSize="10" fontWeight="700"
              fill={active ? "#d97706" : "#64748b"}>
              {w}
            </text>
          </g>
        );
      })}

      {/* Nodes */}
      {NODES.map(node => {
        const active = isActive(node.id);
        const related = isRelated(node.id);
        let fill = "#6366f1", stroke = "#4338ca";
        if (active) { fill = "#f59e0b"; stroke = "#d97706"; }
        else if (related) { fill = "#10b981"; stroke = "#059669"; }
        else if (activeNode !== null) { fill = "#94a3b8"; stroke = "#64748b"; }

        return (
          <g key={node.id} onClick={() => onNodeClick(node.id)}
            className="cursor-pointer">
            <circle cx={node.x} cy={node.y} r={20}
              fill={fill} stroke={stroke} strokeWidth={2.5}
              className="transition-all duration-200"
              opacity={activeNode !== null && !active && !related ? 0.35 : 1}
            />
            <text x={node.x} y={node.y} textAnchor="middle" dominantBaseline="middle"
              fontSize="14" fontWeight="800" fill="white">
              {node.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ─── Adjacency Matrix View ──────────────────────────────────────────────── */

function MatrixView({
  activeNode,
  highlightEdge,
  onCellHover,
}: {
  activeNode: number | null;
  highlightEdge: { u: number; v: number } | null;
  onCellHover: (edge: { u: number; v: number } | null) => void;
}) {
  const isRowCol = (r: number, c: number) =>
    activeNode !== null && (r === activeNode || c === activeNode);
  const isCellHighlit = (r: number, c: number) =>
    highlightEdge !== null &&
    ((highlightEdge.u === r && highlightEdge.v === c) || (highlightEdge.u === c && highlightEdge.v === r));

  return (
    <div className="space-y-2">
      <div className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
        <code className="text-indigo-600 dark:text-indigo-400">mat[u][v] = w</code>（权重）；
        <code className="text-indigo-600 dark:text-indigo-400">mat[u][v] = 0</code> 表示无边。无向图矩阵对称。
      </div>

      <div className="overflow-auto rounded-xl border border-slate-200 dark:border-slate-700">
        <table className="text-center font-mono text-sm w-full">
          <thead>
            <tr className="bg-slate-100 dark:bg-slate-800">
              <th className="p-2 text-slate-400 dark:text-slate-500 border-r border-slate-200 dark:border-slate-700 text-xs w-8"></th>
              {NODES.map(n => (
                <th key={n.id} className={`p-2 font-bold text-xs border-r border-slate-200 dark:border-slate-700 transition-colors ${
                  activeNode === n.id ? "bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300"
                  : "text-slate-500 dark:text-slate-400"
                }`}>{n.id}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {NODES.map(row => (
              <tr key={row.id} className="border-t border-slate-100 dark:border-slate-800">
                {/* Row header */}
                <td className={`p-2 font-bold text-xs border-r border-slate-200 dark:border-slate-700 transition-colors ${
                  activeNode === row.id ? "bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300"
                  : "text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-800/60"
                }`}>{row.id}</td>
                {NODES.map(col => {
                  const val = MAT[row.id][col.id];
                  const inHighlight = isCellHighlit(row.id, col.id);
                  const inRowCol = isRowCol(row.id, col.id);
                  const isDiag = row.id === col.id;
                  return (
                    <td key={col.id}
                      onMouseEnter={() => val > 0 ? onCellHover({ u: row.id, v: col.id }) : null}
                      onMouseLeave={() => onCellHover(null)}
                      className={`p-2 border-r border-slate-100 dark:border-slate-800 transition-all cursor-default ${
                        inHighlight ? "bg-amber-200 dark:bg-amber-900/60 font-bold text-amber-700 dark:text-amber-300"
                        : inRowCol && !isDiag ? "bg-indigo-50 dark:bg-indigo-900/20 text-indigo-700 dark:text-indigo-300"
                        : isDiag ? "bg-slate-100 dark:bg-slate-800/60 text-slate-400 dark:text-slate-600"
                        : val > 0 ? "text-slate-700 dark:text-slate-200"
                        : "text-slate-300 dark:text-slate-700"
                      }`}>
                      {val === 0 ? <span className="opacity-40">0</span> : val}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex gap-3 text-[10px] text-slate-400 dark:text-slate-500 flex-wrap">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded bg-amber-200 dark:bg-amber-900/60 border border-amber-400"></span> 鼠标悬停高亮边
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-300"></span> 选中顶点的行/列
        </span>
        <span className="ml-auto font-semibold text-slate-500 dark:text-slate-400">
          空间 O(V²) = O({N}²) = O({N*N})
        </span>
      </div>
    </div>
  );
}

/* ─── Adjacency List View ────────────────────────────────────────────────── */

function ListViewComp({
  activeNode,
  highlightEdge,
  onEntryHover,
}: {
  activeNode: number | null;
  highlightEdge: { u: number; v: number } | null;
  onEntryHover: (edge: { u: number; v: number } | null) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="text-[11px] text-slate-500 dark:text-slate-400">
        每个顶点维护一个 <code className="text-indigo-600 dark:text-indigo-400">(邻居, 权重)</code> 元组列表，只存实际存在的边。
      </div>

      <div className="space-y-1.5 rounded-xl border border-slate-200 dark:border-slate-700 p-3 bg-slate-50 dark:bg-slate-800/60">
        {NODES.map(node => {
          const isActive = activeNode === node.id;
          return (
            <div key={node.id} className={`flex items-start gap-2 rounded-lg p-2 transition-all ${
              isActive ? "bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800"
              : "border border-transparent"
            }`}>
              {/* Node label */}
              <div className={`flex-none w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold shadow-sm transition-all ${
                isActive ? "bg-amber-400 text-white" : "bg-indigo-500 text-white"
              }`}>
                {node.id}
              </div>

              {/* Arrow */}
              <div className="flex-none text-slate-300 dark:text-slate-600 font-mono text-sm mt-1.5">→</div>

              {/* Neighbor entries */}
              <div className="flex flex-wrap gap-1.5 mt-0.5">
                {ADJ_LIST[node.id].map(({ v, w }) => {
                  const edgeKey = { u: Math.min(node.id, v), v: Math.max(node.id, v) };
                  const isHighlit = highlightEdge !== null &&
                    ((highlightEdge.u === node.id && highlightEdge.v === v) ||
                     (highlightEdge.u === v && highlightEdge.v === node.id));
                  return (
                    <span key={v}
                      onMouseEnter={() => onEntryHover({ u: Math.min(node.id, v), v: Math.max(node.id, v) })}
                      onMouseLeave={() => onEntryHover(null)}
                      className={`inline-flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-mono font-semibold border transition-all cursor-default ${
                        isHighlit
                          ? "bg-amber-200 dark:bg-amber-900/60 border-amber-400 dark:border-amber-600 text-amber-800 dark:text-amber-200"
                          : "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300"
                      }`}>
                      <span>({v},</span>
                      <span className="text-amber-600 dark:text-amber-400">{w}</span>
                      <span>)</span>
                    </span>
                  );
                })}
                {ADJ_LIST[node.id].length === 0 && (
                  <span className="text-xs text-slate-400 dark:text-slate-600 italic mt-1">（无邻居）</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <div className="flex justify-between items-center text-[10px] text-slate-400 dark:text-slate-500">
        <span>点击图中顶点可高亮其邻居列表</span>
        <span className="font-semibold text-slate-500 dark:text-slate-400">
          空间 O(V+E) = O({N}+{EDGES.length*2}) = O({N + EDGES.length * 2})
        </span>
      </div>
    </div>
  );
}

/* ─── Edge List View ─────────────────────────────────────────────────────── */

function EdgeListView({
  activeNode,
  highlightEdge,
  onRowHover,
}: {
  activeNode: number | null;
  highlightEdge: { u: number; v: number } | null;
  onRowHover: (edge: { u: number; v: number } | null) => void;
}) {
  const isEdgeInvolvingNode = (u: number, v: number) =>
    activeNode !== null && (u === activeNode || v === activeNode);

  // Sorted by weight for Kruskal demonstration
  const sortedEdges = [...EDGES].sort((a, b) => a.w - b.w);

  return (
    <div className="space-y-2">
      <div className="text-[11px] text-slate-500 dark:text-slate-400">
        所有边存储在一个列表中（已按权重排序），这是 <strong>Kruskal 最小生成树</strong>算法的天然输入格式。
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
        <table className="w-full text-sm font-mono">
          <thead>
            <tr className="bg-slate-100 dark:bg-slate-800 text-xs text-slate-500 dark:text-slate-400">
              <th className="px-4 py-2 text-left font-semibold border-r border-slate-200 dark:border-slate-700">索引</th>
              <th className="px-4 py-2 text-center font-semibold border-r border-slate-200 dark:border-slate-700">起点 u</th>
              <th className="px-4 py-2 text-center font-semibold border-r border-slate-200 dark:border-slate-700">终点 v</th>
              <th className="px-4 py-2 text-center font-semibold">权重 w</th>
            </tr>
          </thead>
          <tbody>
            {sortedEdges.map(({ u, v, w }, i) => {
              const isHov = highlightEdge !== null &&
                ((highlightEdge.u === u && highlightEdge.v === v) || (highlightEdge.u === v && highlightEdge.v === u));
              const isInvolved = isEdgeInvolvingNode(u, v);
              return (
                <tr key={i}
                  onMouseEnter={() => onRowHover({ u: Math.min(u, v), v: Math.max(u, v) })}
                  onMouseLeave={() => onRowHover(null)}
                  className={`border-t border-slate-100 dark:border-slate-800 transition-all cursor-default ${
                    isHov ? "bg-amber-50 dark:bg-amber-900/20"
                    : isInvolved ? "bg-indigo-50 dark:bg-indigo-900/10"
                    : "hover:bg-slate-50 dark:hover:bg-slate-800/40"
                  }`}>
                  <td className="px-4 py-2.5 text-slate-400 dark:text-slate-600 border-r border-slate-100 dark:border-slate-800">[{i}]</td>
                  <td className={`px-4 py-2.5 text-center font-bold border-r border-slate-100 dark:border-slate-800 ${
                    isHov ? "text-amber-700 dark:text-amber-300" : "text-indigo-600 dark:text-indigo-400"
                  }`}>{u}</td>
                  <td className={`px-4 py-2.5 text-center font-bold border-r border-slate-100 dark:border-slate-800 ${
                    isHov ? "text-amber-700 dark:text-amber-300" : "text-indigo-600 dark:text-indigo-400"
                  }`}>{v}</td>
                  <td className={`px-4 py-2.5 text-center font-bold ${
                    isHov ? "text-amber-700 dark:text-amber-300" : "text-emerald-600 dark:text-emerald-400"
                  }`}>{w}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="flex justify-between items-center text-[10px] text-slate-400 dark:text-slate-500">
        <span>已按权重升序排列（Kruskal 排序步骤的结果）</span>
        <span className="font-semibold text-slate-500 dark:text-slate-400">
          空间 O(E) = O({EDGES.length})
        </span>
      </div>
    </div>
  );
}

/* ─── Main Component ─────────────────────────────────────────────────────── */

const VIEWS: { id: View; label: string; space: string; icon: string }[] = [
  { id: "matrix", label: "邻接矩阵", space: `O(V²)=O(${N*N})`, icon: "⊞" },
  { id: "list",   label: "邻接表",   space: `O(V+E)=O(${N + EDGES.length * 2})`, icon: "☰" },
  { id: "edges",  label: "边集",     space: `O(E)=O(${EDGES.length})`, icon: "≡" },
];

export default function GraphRepresentationToggle() {
  const [view, setView] = useState<View>("matrix");
  const [activeNode, setActiveNode] = useState<number | null>(null);
  const [highlightEdge, setHighlightEdge] = useState<{ u: number; v: number } | null>(null);

  const handleNodeClick = (id: number) => {
    setActiveNode(prev => prev === id ? null : id);
    setHighlightEdge(null);
  };

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">

      {/* ── Header ── */}
      <div className="bg-gradient-to-r from-indigo-500 via-violet-500 to-purple-500 p-5">
        <h3 className="text-white font-bold text-lg tracking-tight">图的三种表示方式</h3>
        <p className="text-indigo-100 text-sm mt-0.5">
          点击图中顶点高亮对应数据结构 · 悬停数据结构条目高亮对应边
        </p>
      </div>

      {/* ── View tabs ── */}
      <div className="flex border-b border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40">
        {VIEWS.map(v => (
          <button key={v.id} onClick={() => { setView(v.id); setHighlightEdge(null); }}
            className={`flex-1 flex flex-col items-center gap-0.5 px-3 py-3 text-sm font-semibold border-b-2 transition-all ${
              view === v.id
                ? "border-violet-500 text-violet-700 dark:text-violet-300 bg-white dark:bg-slate-900"
                : "border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
            }`}>
            <span className="text-base">{v.icon} {v.label}</span>
            <span className={`text-[10px] font-mono ${view === v.id ? "text-violet-500 dark:text-violet-400" : "text-slate-400 dark:text-slate-600"}`}>
              {v.space}
            </span>
          </button>
        ))}
      </div>

      <div className="p-5 grid grid-cols-1 sm:grid-cols-[200px_1fr] gap-5">

        {/* ── Left: SVG Graph ── */}
        <div className="flex flex-col gap-3">
          <div className="rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 p-2 aspect-[1/1]">
            <GraphSVG
              activeNode={activeNode}
              onNodeClick={handleNodeClick}
              highlightEdge={highlightEdge}
            />
          </div>

          {/* Legend */}
          <div className="space-y-1 text-[10px]">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-amber-400 flex-none" />
              <span className="text-slate-500 dark:text-slate-400">已选中顶点</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-emerald-500 flex-none" />
              <span className="text-slate-500 dark:text-slate-400">相邻顶点</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-indigo-500 flex-none" />
              <span className="text-slate-500 dark:text-slate-400">普通顶点</span>
            </div>
          </div>

          {activeNode !== null ? (
            <button onClick={() => setActiveNode(null)}
              className="text-xs text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 underline text-center">
              取消选中
            </button>
          ) : (
            <div className="text-[10px] text-center text-slate-400 dark:text-slate-500 italic">
              点击顶点高亮对应行/列
            </div>
          )}
        </div>

        {/* ── Right: Representation ── */}
        <div>
          {view === "matrix" && (
            <MatrixView
              activeNode={activeNode}
              highlightEdge={highlightEdge}
              onCellHover={setHighlightEdge}
            />
          )}
          {view === "list" && (
            <ListViewComp
              activeNode={activeNode}
              highlightEdge={highlightEdge}
              onEntryHover={setHighlightEdge}
            />
          )}
          {view === "edges" && (
            <EdgeListView
              activeNode={activeNode}
              highlightEdge={highlightEdge}
              onRowHover={setHighlightEdge}
            />
          )}
        </div>
      </div>
    </div>
  );
}
