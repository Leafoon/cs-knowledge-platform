"use client";
import React, { useState, useCallback } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────
interface TreeNode {
  id: number;
  parentNaive: number;   // naive union parent
  parentRank: number;    // union-by-rank parent
  rankVal: number;       // rank for union-by-rank
  label: string;
}

interface Step {
  action: string;
  detail: string;
  naiveParent: number[];
  rankParent: number[];
  rankArr: number[];
  highlight: [number, number]; // [x, y] being unioned
}

// ─── Helper: compute tree positions for SVG rendering ────────────────────────
interface SVGNode { id: number; x: number; y: number; isRoot: boolean }
interface SVGEdge { x1: number; y1: number; x2: number; y2: number }

function computeLayout(parent: number[], n: number): { nodes: SVGNode[]; edges: SVGEdge[] } {
  // Build children map
  const children: number[][] = Array.from({ length: n }, () => []);
  const roots: number[] = [];
  for (let i = 0; i < n; i++) {
    if (parent[i] === i) roots.push(i);
    else children[parent[i]].push(i);
  }

  // BFS from each root to assign positions
  const positions = new Array(n).fill(null) as ({ x: number; y: number } | null)[];
  const W = 520, ROW_H = 64;
  let xCursor = 30;

  // Assign x via DFS (leaves first)
  function leafWidth(node: number): number {
    if (children[node].length === 0) return 1;
    return children[node].reduce((s, c) => s + leafWidth(c), 0);
  }

  function assignX(node: number, startX: number): number {
    const lw = leafWidth(node);
    const slotW = Math.max(50, (W - 60) / Math.max(n, 6));
    const center = startX + (lw * slotW) / 2;
    positions[node] = { x: Math.min(center, W - 20), y: 0 }; // y set by depth
    let childX = startX;
    for (const c of children[node]) {
      assignX(c, childX);
      childX += leafWidth(c) * slotW;
    }
    return center;
  }

  // DFS for depth
  function setDepth(node: number, depth: number) {
    if (positions[node]) positions[node]!.y = depth * ROW_H + 30;
    for (const c of children[node]) setDepth(c, depth + 1);
  }

  let rootX = 20;
  for (const r of roots) {
    const slotW = Math.max(50, (W - 40) / Math.max(n, 6));
    assignX(r, rootX);
    rootX += leafWidth(r) * slotW;
    setDepth(r, 0);
  }

  const nodes: SVGNode[] = [];
  const edges: SVGEdge[] = [];

  for (let i = 0; i < n; i++) {
    const p = positions[i];
    if (!p) continue;
    nodes.push({ id: i, x: p.x, y: p.y, isRoot: parent[i] === i });
    if (parent[i] !== i) {
      const pp = positions[parent[i]];
      if (pp) edges.push({ x1: pp.x, y1: pp.y, x2: p.x, y2: p.y });
    }
  }
  return { nodes, edges };
}

// Tree height from parent array
function treeHeight(parent: number[], n: number): number {
  const children: number[][] = Array.from({ length: n }, () => []);
  for (let i = 0; i < n; i++) if (parent[i] !== i) children[parent[i]].push(i);
  let maxH = 0;
  function dfs(node: number, h: number) {
    maxH = Math.max(maxH, h);
    for (const c of children[node]) dfs(c, h + 1);
  }
  for (let i = 0; i < n; i++) if (parent[i] === i) dfs(i, 0);
  return maxH;
}

// ─── Preset union sequence ────────────────────────────────────────────────────
const N = 8;
const UNION_SEQ: [number, number][] = [[0,1],[2,3],[0,2],[4,5],[6,7],[4,6],[0,4]];

function buildSteps(): Step[] {
  const steps: Step[] = [];
  const naiveP = Array.from({ length: N }, (_, i) => i);
  const rankP  = Array.from({ length: N }, (_, i) => i);
  const rankA  = new Array(N).fill(0);

  // Naive find
  function naiveFind(x: number) {
    while (naiveP[x] !== x) x = naiveP[x];
    return x;
  }
  // Rank find
  function rankFind(x: number) {
    while (rankP[x] !== x) x = rankP[x];
    return x;
  }

  for (const [x, y] of UNION_SEQ) {
    const nrx = naiveFind(x), nry = naiveFind(y);
    const rrx = rankFind(x), rry = rankFind(y);

    // Naive: always attach y's root under x's root (no rank)
    if (nrx !== nry) naiveP[nry] = nrx;
    // Rank: smaller rank under larger rank
    if (rrx !== rry) {
      if (rankA[rrx] < rankA[rry]) rankP[rrx] = rry;
      else if (rankA[rrx] > rankA[rry]) rankP[rry] = rrx;
      else { rankP[rry] = rrx; rankA[rrx]++; }
    }

    steps.push({
      action: `UNION(${x}, ${y})`,
      detail: `朴素合并：${nrx} ← ${nry}（高树相接，不控制高度）｜ 按秩合并：rank[${rrx}]=${rankA[rrx] - (rrx === (rankFind(x))?0:0)}, rank[${rry}]=${rankA[rry]}`,
      naiveParent: [...naiveP],
      rankParent: [...rankP],
      rankArr: [...rankA],
      highlight: [x, y],
    });
  }
  return steps;
}

const STEPS = buildSteps();

// ─── SVG Tree renderer ────────────────────────────────────────────────────────
function TreePanel({
  parent, highlight, rankArr, title, color, height,
}: {
  parent: number[]; highlight: [number, number]; rankArr?: number[];
  title: string; color: string; height: number;
}) {
  const n = parent.length;
  const { nodes, edges } = computeLayout(parent, n);
  const SVG_H = 200;

  return (
    <div className="flex-1 min-w-0">
      <div className={`flex items-center justify-between mb-2`}>
        <span className={`text-xs font-semibold ${color}`}>{title}</span>
        <span className={`text-xs px-2 py-0.5 rounded-full font-mono ${
          height > 3 ? "bg-red-100 dark:bg-red-900/40 text-red-600 dark:text-red-400" :
          height > 1 ? "bg-amber-100 dark:bg-amber-900/40 text-amber-600 dark:text-amber-400" :
          "bg-emerald-100 dark:bg-emerald-900/40 text-emerald-600 dark:text-emerald-400"
        }`}>高度 = {height}</span>
      </div>
      <div className="bg-slate-50 dark:bg-slate-800/70 rounded-xl overflow-hidden">
        <svg viewBox={`0 0 520 ${SVG_H}`} className="w-full" style={{ height: SVG_H }}>
          {/* Edges */}
          {edges.map((e, i) => (
            <line key={i} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
              className="stroke-slate-300 dark:stroke-slate-600" strokeWidth={2} />
          ))}
          {/* Nodes */}
          {nodes.map(nd => {
            const isHL = highlight[0] === nd.id || highlight[1] === nd.id;
            const isRoot = nd.isRoot;
            return (
              <g key={nd.id}>
                <circle cx={nd.x} cy={nd.y} r={18}
                  fill={isHL ? "#6366f1" : isRoot ? "#10b981" : "#64748b"}
                  opacity={0.9}
                  stroke={isRoot ? "#d1fae5" : "none"} strokeWidth={2}
                />
                <text x={nd.x} y={nd.y + 1} textAnchor="middle" dominantBaseline="middle"
                  fill="white" fontSize={11} fontWeight="bold" fontFamily="monospace">
                  {nd.id}
                </text>
                {rankArr && (
                  <text x={nd.x + 14} y={nd.y - 12} textAnchor="middle" dominantBaseline="middle"
                    fill="#a78bfa" fontSize={9} fontFamily="monospace">
                    r{rankArr[nd.id]}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
      <div className="flex gap-3 mt-2 text-xs text-slate-500 dark:text-slate-400 flex-wrap">
        {parent.map((p, i) => (
          <span key={i} className="font-mono">
            {i}→<span className={p === i ? "text-emerald-500" : ""}>{p}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function UnionByRankTree() {
  const [stepIdx, setStepIdx] = useState(-1); // -1 = initial (no union yet)

  const initialParent = Array.from({ length: N }, (_, i) => i);
  const initialRankArr = new Array(N).fill(0);

  const currentStep = stepIdx >= 0 ? STEPS[stepIdx] : null;
  const naiveP  = currentStep?.naiveParent  ?? initialParent;
  const rankP   = currentStep?.rankParent   ?? initialParent;
  const rankArr = currentStep?.rankArr      ?? initialRankArr;
  const hl      = currentStep?.highlight    ?? [-1, -1] as [number, number];

  const naiveH = treeHeight(naiveP, N);
  const rankH  = treeHeight(rankP,  N);

  const handlePrev = () => setStepIdx(s => Math.max(-1, s - 1));
  const handleNext = () => setStepIdx(s => Math.min(STEPS.length - 1, s + 1));
  const handleReset = () => setStepIdx(-1);

  const progress = stepIdx === -1 ? 0 : ((stepIdx + 1) / STEPS.length) * 100;

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5 my-6 shadow-sm">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-2 mb-1">
        <h3 className="text-base font-bold text-slate-800 dark:text-slate-100">
          🌳 按秩合并（Union by Rank）效果对比
        </h3>
        <div className="flex gap-2 text-xs">
          <span className="px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 font-mono">
            n = {N} 个节点
          </span>
          <span className="px-2 py-0.5 rounded-full bg-indigo-100 dark:bg-indigo-900/40 text-indigo-600 dark:text-indigo-400 font-mono">
            {stepIdx + 1} / {STEPS.length} 次 UNION
          </span>
        </div>
      </div>
      <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
        对同一序列的 UNION 操作，观察<span className="text-rose-500 dark:text-rose-400 font-semibold">朴素合并</span>（总把右树挂到左树下）与
        <span className="text-violet-500 dark:text-violet-400 font-semibold">按秩合并</span>（矮树挂到高树下）产生的树高差异。
      </p>

      {/* Operation info */}
      {currentStep ? (
        <div className="mb-4 rounded-lg bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 px-4 py-2.5">
          <div className="text-xs font-bold text-indigo-700 dark:text-indigo-300 font-mono mb-1">
            第 {stepIdx + 1} 步：{currentStep.action}
          </div>
          <div className="text-xs text-indigo-600 dark:text-indigo-400">{currentStep.detail}</div>
        </div>
      ) : (
        <div className="mb-4 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 px-4 py-2.5">
          <div className="text-xs text-slate-500 dark:text-slate-400">初始状态：{N} 个节点各自独立，等待执行 UNION 操作…</div>
        </div>
      )}

      {/* Side-by-side trees */}
      <div className="flex gap-4 mb-4">
        <TreePanel parent={naiveP} highlight={hl} title="朴素合并（无优化）" color="text-rose-500 dark:text-rose-400" height={naiveH} />
        <div className="w-px bg-slate-200 dark:bg-slate-700 flex-shrink-0" />
        <TreePanel parent={rankP} highlight={hl} rankArr={rankArr} title="按秩合并（Union by Rank）" color="text-violet-500 dark:text-violet-400" height={rankH} />
      </div>

      {/* Height comparison bar */}
      <div className="mb-4 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-3">
        <div className="text-xs text-slate-500 dark:text-slate-400 mb-2 font-semibold">树高对比（越低越好）</div>
        <div className="flex flex-col gap-2">
          {[{ label: "朴素合并", h: naiveH, max: N - 1, color: "bg-rose-400" },
            { label: "按秩合并", h: rankH,  max: N - 1, color: "bg-violet-500" },
            { label: "理论上界 ⌊log₂n⌋", h: Math.floor(Math.log2(N)), max: N - 1, color: "bg-emerald-400", isDashed: true }
          ].map(row => (
            <div key={row.label} className="flex items-center gap-2">
              <span className="text-xs text-slate-500 dark:text-slate-400 w-28 shrink-0">{row.label}</span>
              <div className="flex-1 bg-slate-200 dark:bg-slate-700 rounded-full h-3 overflow-hidden">
                <div className={`h-full ${row.color} rounded-full transition-all duration-500 ${row.isDashed ? "opacity-50" : ""}`}
                  style={{ width: `${(row.h / row.max) * 100}%` }} />
              </div>
              <span className="text-xs font-mono text-slate-600 dark:text-slate-300 w-6 text-right">{row.h}</span>
            </div>
          ))}
        </div>
        {naiveH > rankH + 1 && (
          <p className="text-xs text-rose-500 dark:text-rose-400 mt-2">
            ⚠️ 朴素合并树高已比按秩合并高 {naiveH - rankH} 层，FIND 代价差距将持续扩大。
          </p>
        )}
      </div>

      {/* Progress + Controls */}
      <div className="flex items-center gap-2 mb-3">
        <button onClick={handleReset}
          className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400
            text-xs hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
          ↺ 重置
        </button>
        <div className="flex-1 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
          <div className="h-full bg-indigo-500 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }} />
        </div>
        <button onClick={handlePrev} disabled={stepIdx < 0}
          className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300
            text-xs disabled:opacity-30 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
          ◀ 上一步
        </button>
        <button onClick={handleNext} disabled={stepIdx >= STEPS.length - 1}
          className="px-3 py-1.5 rounded-lg bg-indigo-600 text-white text-xs
            disabled:opacity-30 hover:bg-indigo-700 transition-colors">
          下一步 ▶
        </button>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-xs text-slate-500 dark:text-slate-400">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-indigo-500 inline-block" />当前 UNION 参与节点
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-emerald-500 inline-block" />集合根（代表元素）
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-slate-400 inline-block" />普通节点
        </span>
        <span className="flex items-center gap-1.5">
          <span className="text-violet-400 font-mono text-xs">r0</span>右上角标 = 节点 rank 值
        </span>
      </div>
    </div>
  );
}
