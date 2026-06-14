"use client";
import React, { useState, useCallback } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────
type Phase = "idle" | "traversing" | "compressing" | "done";

interface AnimFrame {
  phase: Phase;
  visitedPath: number[];          // nodes traversed so far (highlight these)
  compressedNodes: number[];      // nodes already flattened
  parent: number[];               // current parent array snapshot
  description: string;
}

// ─── Preset: 7-node chain: 6→5→4→3→2→1→0 (0 is root) ───────────────────────
//   plus node 7 branching from 2
const N = 8;
const FIND_TARGET = 6;   // we will FIND(6)

// Initial tree: a long chain 6→5→4→3→2→1→0, plus 7→3
const INIT_PARENT = [0, 0, 1, 2, 3, 4, 5, 3]; // index is node, value is parent

function buildFrames(): AnimFrame[] {
  const frames: AnimFrame[] = [];
  const parent = [...INIT_PARENT];

  // Phase 1: Traverse from target to root
  const path: number[] = [];
  let cur = FIND_TARGET;
  while (parent[cur] !== cur) {
    path.push(cur);
    cur = parent[cur];
  }
  path.push(cur); // push root
  const root = cur;

  // Traversal frames (reveal path nodes one by one)
  for (let i = 1; i <= path.length; i++) {
    frames.push({
      phase: "traversing",
      visitedPath: path.slice(0, i),
      compressedNodes: [],
      parent: [...parent],
      description: i < path.length
        ? `第 ${i} 步：访问节点 ${path[i - 1]}，它的父节点是 ${parent[path[i - 1]]}，继续向上…`
        : `已到达根节点 ${root}，总计经过 ${path.length - 1} 跳，现在开始路径压缩。`,
    });
  }

  // Phase 2: Path compression — each non-root node on path points directly to root
  const compressedSoFar: number[] = [];
  for (let i = 0; i < path.length - 1; i++) {
    const node = path[i];
    parent[node] = root;
    compressedSoFar.push(node);
    frames.push({
      phase: "compressing",
      visitedPath: path,
      compressedNodes: [...compressedSoFar],
      parent: [...parent],
      description: `压缩节点 ${node}：parent[${node}] 从 ${INIT_PARENT[node]} 改为 ${root}（根），节点直接挂到根下。`,
    });
  }

  // Done frame
  frames.push({
    phase: "done",
    visitedPath: path,
    compressedNodes: [...compressedSoFar],
    parent: [...parent],
    description: `✅ FIND(${FIND_TARGET}) 完成！路径压缩后，${path.length - 1} 个节点均直连根 ${root}，下次查询只需 1 跳。`,
  });

  return frames;
}

const FRAMES = buildFrames();
const TOTAL_STEPS = FRAMES.length;

// ─── Layout computation ───────────────────────────────────────────────────────
interface DisplayNode { id: number; x: number; y: number }
interface DisplayEdge { x1: number; y1: number; x2: number; y2: number; from: number; to: number }

function computeLayout(parent: number[], n: number, w: number): { nodes: DisplayNode[]; edges: DisplayEdge[] } {
  const children: number[][] = Array.from({ length: n }, () => []);
  for (let i = 0; i < n; i++) if (parent[i] !== i) children[parent[i]].push(i);
  const root = parent.findIndex((p, i) => p === i);

  // BFS level assignment
  const level = new Array(n).fill(-1);
  const queue = [root];
  level[root] = 0;
  while (queue.length) {
    const cur = queue.shift()!;
    for (const c of children[cur]) {
      level[c] = level[cur] + 1;
      queue.push(c);
    }
  }

  // Count nodes per level for x-position
  const byLevel: number[][] = [];
  for (let i = 0; i < n; i++) {
    const lv = level[i];
    if (!byLevel[lv]) byLevel[lv] = [];
    byLevel[lv].push(i);
  }

  const positions: { x: number; y: number }[] = new Array(n);
  const ROW_H = 62, TOP = 28;
  for (let lv = 0; lv < byLevel.length; lv++) {
    const row = byLevel[lv];
    row.forEach((id, idx) => {
      const gap = w / (row.length + 1);
      positions[id] = { x: gap * (idx + 1), y: TOP + lv * ROW_H };
    });
  }

  const nodes: DisplayNode[] = Array.from({ length: n }, (_, i) => ({ id: i, ...positions[i] }));
  const edges: DisplayEdge[] = [];
  for (let i = 0; i < n; i++) {
    if (parent[i] !== i) {
      edges.push({ x1: positions[parent[i]].x, y1: positions[parent[i]].y, x2: positions[i].x, y2: positions[i].y, from: parent[i], to: i });
    }
  }
  return { nodes, edges };
}

// ─── Node color logic ─────────────────────────────────────────────────────────
function getNodeColor(id: number, frame: AnimFrame | null): { fill: string; stroke: string } {
  if (!frame) {
    if (INIT_PARENT[id] === id) return { fill: "#10b981", stroke: "#d1fae5" }; // root
    return { fill: "#64748b", stroke: "none" };
  }
  const isRoot = frame.parent[id] === id;
  const isTarget = id === FIND_TARGET;
  const isOnPath = frame.visitedPath.includes(id);
  const isCompressed = frame.compressedNodes.includes(id);

  if (isRoot) return { fill: "#10b981", stroke: "#6ee7b7" };
  if (isCompressed) return { fill: "#6366f1", stroke: "#a5b4fc" };
  if (isTarget) return { fill: "#f59e0b", stroke: "#fcd34d" };
  if (isOnPath) return { fill: "#3b82f6", stroke: "#93c5fd" };
  return { fill: "#64748b", stroke: "none" };
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function UnionFindPathCompression() {
  const [stepIdx, setStepIdx] = useState(-1);

  const frame = stepIdx >= 0 ? FRAMES[stepIdx] : null;
  const phase = frame?.phase ?? "idle";
  const parent = frame?.parent ?? INIT_PARENT;

  const SVG_W = 520, SVG_H = 220;
  const { nodes, edges } = computeLayout(parent, N, SVG_W);
  const { nodes: initNodes, edges: initEdges } = computeLayout(INIT_PARENT, N, SVG_W);

  const traverseLen = frame?.visitedPath.length ?? 0;
  const comprLen = frame?.compressedNodes.length ?? 0;
  const pathLen = INIT_PARENT[FIND_TARGET] !== FIND_TARGET ? 
    (() => { let n2 = FIND_TARGET, cnt = 0; while (INIT_PARENT[n2] !== n2) { n2 = INIT_PARENT[n2]; cnt++; } return cnt; })() : 0;

  const progress = stepIdx < 0 ? 0 : ((stepIdx + 1) / TOTAL_STEPS) * 100;

  const phaseLabel: Record<Phase, string> = {
    idle: "等待操作",
    traversing: "🔍 向上遍历，寻找根",
    compressing: "⚡ 路径压缩进行中",
    done: "✅ 压缩完成",
  };

  const phaseBgClass: Record<Phase, string> = {
    idle: "bg-slate-50 dark:bg-slate-800/60 border-slate-200 dark:border-slate-700",
    traversing: "bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800",
    compressing: "bg-violet-50 dark:bg-violet-900/20 border-violet-200 dark:border-violet-800",
    done: "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800",
  };
  const phaseTextClass: Record<Phase, string> = {
    idle: "text-slate-500 dark:text-slate-400",
    traversing: "text-blue-700 dark:text-blue-300",
    compressing: "text-violet-700 dark:text-violet-300",
    done: "text-emerald-700 dark:text-emerald-300",
  };

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5 my-6 shadow-sm">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-2 mb-1">
        <h3 className="text-base font-bold text-slate-800 dark:text-slate-100">
          ⚡ FIND 路径压缩动画
        </h3>
        <span className={`text-xs px-2 py-0.5 rounded-full border font-semibold ${phaseBgClass[phase]} ${phaseTextClass[phase]}`}>
          {phaseLabel[phase]}
        </span>
      </div>
      <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
        对一棵深度为 {pathLen} 的链式树执行 <span className="font-mono text-amber-500">FIND({FIND_TARGET})</span>。
        第一阶段：<span className="text-blue-500 dark:text-blue-400">逐跳向上遍历</span>找到根；
        第二阶段：<span className="text-violet-500 dark:text-violet-400">路径压缩</span>将沿途节点直连根，后续 FIND 只需 1 跳。
      </p>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-2 mb-4 text-center">
        {[
          { label: "初始树深度", val: pathLen, color: "text-rose-500 dark:text-rose-400" },
          { label: "当前已遍历跳数", val: traverseLen > 0 ? traverseLen - 1 : 0, color: "text-blue-500 dark:text-blue-400" },
          { label: "已压缩节点数", val: comprLen, color: "text-violet-500 dark:text-violet-400" },
        ].map(s => (
          <div key={s.label} className="rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 py-2 px-1">
            <div className={`text-xl font-bold font-mono ${s.color}`}>{s.val}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{s.label}</div>
          </div>
        ))}
      </div>

      {/* SVG Tree */}
      <div className="bg-slate-50 dark:bg-slate-800/60 rounded-xl overflow-hidden mb-4">
        <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="w-full" style={{ height: SVG_H }}>
          {/* Edges */}
          {edges.map((e, i) => {
            const isNew = frame?.compressedNodes.includes(e.to);
            return (
              <line key={`e${i}`} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
                stroke={isNew ? "#8b5cf6" : "#94a3b8"}
                strokeWidth={isNew ? 2.5 : 1.5}
                strokeDasharray={isNew ? "none" : "none"}
                opacity={0.8}
              />
            );
          })}
          {/* Compression effect: dashed arcs for newly compressed */}
          {frame?.compressedNodes.map(id => {
            const nd = nodes.find(n => n.id === id);
            const root = nodes.find(n => frame.parent[n.id] === n.id);
            if (!nd || !root) return null;
            return (
              <line key={`arc${id}`} x1={root.x} y1={root.y} x2={nd.x} y2={nd.y}
                stroke="#8b5cf6" strokeWidth={2} opacity={0.4} strokeDasharray="5,3" />
            );
          })}
          {/* Nodes */}
          {nodes.map(nd => {
            const { fill, stroke } = getNodeColor(nd.id, frame);
            return (
              <g key={`n${nd.id}`}>
                <circle cx={nd.x} cy={nd.y} r={21} fill={fill} opacity={0.92}
                  stroke={stroke !== "none" ? stroke : fill} strokeWidth={stroke !== "none" ? 2.5 : 0}
                />
                <text x={nd.x} y={nd.y + 1} textAnchor="middle" dominantBaseline="middle"
                  fill="white" fontSize={12} fontWeight="bold" fontFamily="monospace">{nd.id}</text>
              </g>
            );
          })}
          {/* Traversal path arrows (dashed) */}
          {frame?.visitedPath && frame.visitedPath.length > 1 && phase === "traversing" && (() => {
            const path = frame.visitedPath;
            return path.slice(0, -1).map((id, i) => {
              const from = nodes.find(n => n.id === path[i + 1]);
              const to = nodes.find(n => n.id === id);
              if (!from || !to) return null;
              const dx = to.x - from.x, dy = to.y - from.y;
              const len = Math.sqrt(dx * dx + dy * dy);
              const ux = dx / len, uy = dy / len;
              return (
                <line key={`arrow${i}`}
                  x1={from.x + ux * 22} y1={from.y + uy * 22}
                  x2={to.x - ux * 22} y2={to.y - uy * 22}
                  stroke="#3b82f6" strokeWidth={2} strokeDasharray="4,3" markerEnd="url(#arrowBlue)" />
              );
            });
          })()}
          <defs>
            <marker id="arrowBlue" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <path d="M0,0 L0,6 L6,3 z" fill="#3b82f6" />
            </marker>
          </defs>
        </svg>
      </div>

      {/* Description panel */}
      <div className={`mb-4 rounded-lg border px-4 py-2.5 ${phaseBgClass[phase]}`}>
        <p className={`text-xs ${phaseTextClass[phase]}`}>
          {frame?.description ?? `点击「下一步」开始对节点 ${FIND_TARGET} 执行 FIND 操作。当前树深度 = ${pathLen}，共需 ${pathLen} 跳才能找到根。`}
        </p>
      </div>

      {/* Parent array display */}
      <div className="mb-4 rounded-lg bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 p-3">
        <div className="text-xs text-slate-500 dark:text-slate-400 font-semibold mb-2">parent 数组（实时更新）</div>
        <div className="flex flex-wrap gap-2">
          {parent.map((p, i) => {
            const isCompressed = frame?.compressedNodes.includes(i) ?? false;
            const changed = p !== INIT_PARENT[i];
            return (
              <div key={i} className={`flex flex-col items-center rounded px-2 py-1 text-xs ${
                isCompressed ? "bg-violet-100 dark:bg-violet-900/30" : "bg-slate-100 dark:bg-slate-800"
              }`}>
                <span className="text-slate-500 dark:text-slate-400 font-mono">[{i}]</span>
                <span className={`font-bold font-mono ${changed ? "text-violet-600 dark:text-violet-300" : "text-slate-700 dark:text-slate-200"}`}>
                  {p}
                </span>
              </div>
            );
          })}
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
        <button onClick={() => setStepIdx(s => Math.min(TOTAL_STEPS - 1, s + 1))} disabled={stepIdx >= TOTAL_STEPS - 1}
          className="px-3 py-1.5 rounded-lg bg-indigo-600 text-white text-xs
            disabled:opacity-30 hover:bg-indigo-700 transition-colors">
          下一步 ▶
        </button>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-xs text-slate-500 dark:text-slate-400">
        {[
          { color: "bg-emerald-500", label: "根节点（代表元素）" },
          { color: "bg-amber-500",   label: `目标节点（FIND(${FIND_TARGET})）` },
          { color: "bg-blue-500",    label: "遍历路径节点" },
          { color: "bg-violet-500",  label: "已压缩节点（现直连根）" },
          { color: "bg-slate-400",   label: "其他节点" },
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
