"use client";
import React, { useState, useMemo } from "react";

/*
 * Decision tree for insertion sort on [a1, a2, a3]
 * (same tree structure as CLRS Figure 8.1)
 *
 *                       a1:a2
 *                    ≤/       \>
 *               a2:a3          a1:a3
 *             ≤/     \>      ≤/       \>
 *          [1,2,3] a1:a3  [2,1,3]  a2:a3
 *                 ≤/  \>          ≤/    \>
 *           [1,3,2] [3,1,2]   [2,3,1] [3,2,1]
 */

/* ─── Tree data ─────────────────────────────────────────────────────────── */

type NodeType = "cmp" | "leaf";

interface TreeNodeDef {
  id: string;
  type: NodeType;
  label: string;           // for cmp: "a₁:a₂"; for leaf: "[1,2,3]"
  cx: number; cy: number;  // SVG center
  leafPerm?: number[];     // [0,1,2] = indices into [a1,a2,a3]
  // comparison: which two indices to compare (0=a1, 1=a2, 2=a3)
  cmpLeft?: number; cmpRight?: number;
  // edges
  leftChild?: string;  // id of ≤ child
  rightChild?: string; // id of > child
  parentId?: string;
  parentEdge?: "left" | "right";
}

// viewBox: 0 0 640 300
const NODES: TreeNodeDef[] = [
  // ── Internal comparison nodes ──────────────────────────────────────────
  { id: "n0",  type: "cmp",  label: "a₁ : a₂", cx: 320, cy: 32,  cmpLeft: 0, cmpRight: 1, leftChild: "n1",  rightChild: "n3" },
  { id: "n1",  type: "cmp",  label: "a₂ : a₃", cx: 155, cy: 108, cmpLeft: 1, cmpRight: 2, leftChild: "l0",  rightChild: "n2",  parentId: "n0",  parentEdge: "left" },
  { id: "n3",  type: "cmp",  label: "a₁ : a₃", cx: 490, cy: 108, cmpLeft: 0, cmpRight: 2, leftChild: "l3",  rightChild: "n4",  parentId: "n0",  parentEdge: "right" },
  { id: "n2",  type: "cmp",  label: "a₁ : a₃", cx: 240, cy: 185, cmpLeft: 0, cmpRight: 2, leftChild: "l1",  rightChild: "l2",  parentId: "n1",  parentEdge: "right" },
  { id: "n4",  type: "cmp",  label: "a₂ : a₃", cx: 565, cy: 185, cmpLeft: 1, cmpRight: 2, leftChild: "l4",  rightChild: "l5",  parentId: "n3",  parentEdge: "right" },
  // ── Leaf nodes ──────────────────────────────────────────────────────────
  { id: "l0",  type: "leaf", label: "⟨1,2,3⟩", cx: 65,  cy: 185, leafPerm: [0,1,2], parentId: "n1",  parentEdge: "left" },
  { id: "l1",  type: "leaf", label: "⟨1,3,2⟩", cx: 178, cy: 262, leafPerm: [0,2,1], parentId: "n2",  parentEdge: "left" },
  { id: "l2",  type: "leaf", label: "⟨3,1,2⟩", cx: 305, cy: 262, leafPerm: [2,0,1], parentId: "n2",  parentEdge: "right" },
  { id: "l3",  type: "leaf", label: "⟨2,1,3⟩", cx: 390, cy: 185, leafPerm: [1,0,2], parentId: "n3",  parentEdge: "left" },
  { id: "l4",  type: "leaf", label: "⟨2,3,1⟩", cx: 510, cy: 262, leafPerm: [1,2,0], parentId: "n4",  parentEdge: "left" },
  { id: "l5",  type: "leaf", label: "⟨3,2,1⟩", cx: 630, cy: 262, leafPerm: [2,1,0], parentId: "n4",  parentEdge: "right" },
];

const NODE_MAP = Object.fromEntries(NODES.map(n => [n.id, n])) as Record<string, TreeNodeDef>;

/* ─── Path trace ─────────────────────────────────────────────────────────── */

function tracePath(vals: [number, number, number]): { nodeIds: string[]; leafId: string; comparisons: number } {
  const nodeIds: string[] = [];
  let cur = NODE_MAP["n0"];
  let comparisons = 0;

  while (cur.type === "cmp") {
    nodeIds.push(cur.id);
    comparisons++;
    const lv = vals[cur.cmpLeft!];
    const rv = vals[cur.cmpRight!];
    const goLeft = lv <= rv;
    const nextId = goLeft ? cur.leftChild! : cur.rightChild!;
    cur = NODE_MAP[nextId];
  }
  // cur is now a leaf
  nodeIds.push(cur.id);
  return { nodeIds, leafId: cur.id, comparisons };
}

/* ─── Edges ──────────────────────────────────────────────────────────────── */

const EDGES = NODES.filter(n => n.parentId).map(n => ({
  from: NODE_MAP[n.parentId!],
  to: n,
  label: n.parentEdge === "left" ? "≤" : ">",
}));

/* ─── Main component ─────────────────────────────────────────────────────── */

const PRESETS: { label: string; values: [number, number, number] }[] = [
  { label: "2,1,3", values: [2, 1, 3] },
  { label: "1,2,3", values: [1, 2, 3] },
  { label: "3,1,2", values: [3, 1, 2] },
  { label: "3,2,1", values: [3, 2, 1] },
  { label: "1,3,2", values: [1, 3, 2] },
  { label: "2,3,1", values: [2, 3, 1] },
];

export default function SortDecisionTree() {
  const [vals, setVals] = useState<[number, number, number]>([2, 1, 3]);
  const [hovered, setHovered] = useState<string | null>(null);

  const { nodeIds: pathIds, leafId, comparisons } = useMemo(() => tracePath(vals), [vals]);
  const pathSet = new Set(pathIds);

  const leafNode = NODE_MAP[leafId];
  const resultArr = leafNode.leafPerm!.map(i => vals[i]);

  const isOnPath = (id: string) => pathSet.has(id);
  const isActiveEdge = (fromId: string, toId: string) => {
    const fi = pathIds.indexOf(fromId);
    return fi >= 0 && pathIds[fi + 1] === toId;
  };

  const cmpNodeColor = (id: string) =>
    isOnPath(id)
      ? { fill: "#6366f1", stroke: "#4f46e5", text: "#ffffff" }
      : id === hovered
        ? { fill: "#e0e7ff", stroke: "#818cf8", text: "#4f46e5" }
        : { fill: "#f8fafc", stroke: "#cbd5e1", text: "#475569" };

  const leafColor = (id: string) =>
    id === leafId
      ? { fill: "#d1fae5", stroke: "#059669", text: "#065f46" }
      : id === hovered
        ? { fill: "#fffbeb", stroke: "#f59e0b", text: "#92400e" }
        : { fill: "#f1f5f9", stroke: "#94a3b8", text: "#374151" };

  // Dark-mode-friendly: we'll use SVG with Tailwind dark variant via CSS variable trick
  // For simplicity, we'll manage colors programmatically and apply a background wrapper

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 overflow-hidden shadow-sm">
      {/* Header */}
      <div className="px-5 py-4 border-b border-slate-200 dark:border-slate-700">
        <h3 className="font-bold text-slate-800 dark:text-slate-100 text-base">比较排序决策树（n = 3）</h3>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
          任何基于比较的排序算法都对应一棵决策树；3! = 6 个叶子 → 树高 ≥ ⌈log₂ 6⌉ = 3
        </p>
      </div>

      <div className="p-5">
        {/* Input selector */}
        <div className="mb-5">
          <p className="text-xs font-medium text-slate-600 dark:text-slate-400 mb-2">选择输入（a₁, a₂, a₃）—— 观察排序路径：</p>
          <div className="flex flex-wrap gap-2">
            {PRESETS.map(p => {
              const active = vals.join(",") === p.values.join(",");
              return (
                <button
                  key={p.label}
                  onClick={() => setVals(p.values)}
                  className={`px-3 py-1.5 text-xs rounded-lg font-mono font-semibold border transition-all ${
                    active
                      ? "bg-indigo-600 border-indigo-600 text-white shadow-sm"
                      : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:border-indigo-300 dark:hover:border-indigo-600"
                  }`}
                >
                  [{p.label}]
                </button>
              );
            })}
          </div>

          {/* Custom input */}
          <div className="mt-3 flex items-center gap-2">
            <span className="text-xs text-slate-500 dark:text-slate-400">自定义：</span>
            {([0, 1, 2] as const).map(i => (
              <input
                key={i}
                type="number"
                value={vals[i]}
                min={1} max={9}
                onChange={e => {
                  const v = Math.max(1, Math.min(9, Number(e.target.value)));
                  const next: [number, number, number] = [...vals] as [number, number, number];
                  next[i] = v;
                  setVals(next);
                }}
                className="w-14 px-2 py-1 text-xs text-center rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 font-mono"
                placeholder={`a${i + 1}`}
              />
            ))}
            <span className="text-xs text-slate-400 dark:text-slate-500">(各不相等效果最佳)</span>
          </div>
        </div>

        {/* SVG Tree */}
        <div className="overflow-x-auto rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 p-2">
          <svg viewBox="0 0 700 300" className="w-full max-w-3xl mx-auto" style={{ minWidth: 500 }}>
            {/* Edges */}
            {EDGES.map(({ from, to, label }) => {
              const active = isActiveEdge(from.id, to.id);
              const edgeHalf_x = (from.cx + to.cx) / 2;
              const edgeHalf_y = (from.cy + to.cy) / 2;
              return (
                <g key={`${from.id}-${to.id}`}>
                  <line
                    x1={from.cx} y1={from.cy + (from.type === "cmp" ? 18 : 16)}
                    x2={to.cx} y2={to.cy - (to.type === "cmp" ? 18 : 16)}
                    stroke={active ? "#f59e0b" : "#cbd5e1"}
                    strokeWidth={active ? 2.5 : 1.5}
                    strokeDasharray={active ? "none" : "none"}
                    className={active ? "" : "dark:stroke-slate-600"}
                  />
                  {/* edge label */}
                  <text
                    x={edgeHalf_x + (to.cx < from.cx ? -8 : 8)}
                    y={edgeHalf_y}
                    textAnchor="middle"
                    fontSize={10}
                    fontWeight={active ? "700" : "500"}
                    fill={active ? "#f59e0b" : "#94a3b8"}
                    className={active ? "" : "dark:fill-slate-500"}
                  >
                    {label}
                  </text>
                </g>
              );
            })}

            {/* Nodes */}
            {NODES.map(node => {
              if (node.type === "cmp") {
                const c = cmpNodeColor(node.id);
                const onPath = isOnPath(node.id);
                return (
                  <g
                    key={node.id}
                    style={{ cursor: "pointer" }}
                    onMouseEnter={() => setHovered(node.id)}
                    onMouseLeave={() => setHovered(null)}
                  >
                    <ellipse
                      cx={node.cx} cy={node.cy}
                      rx={42} ry={18}
                      fill={onPath ? c.fill : ""}
                      className={onPath ? "" : "fill-slate-100 dark:fill-slate-800 stroke-slate-300 dark:stroke-slate-600"}
                      stroke={onPath ? c.stroke : ""}
                      strokeWidth={onPath ? 2.5 : 1.5}
                    />
                    <text
                      x={node.cx} y={node.cy + 1}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fontSize={11}
                      fontWeight="600"
                      fill={onPath ? c.text : ""}
                      className={onPath ? "" : "fill-slate-600 dark:fill-slate-300"}
                    >
                      {node.label}
                    </text>
                    {onPath && (
                      <ellipse cx={node.cx} cy={node.cy} rx={44} ry={20} fill="none" stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="3 2" />
                    )}
                  </g>
                );
              } else {
                // Leaf node
                const c = leafColor(node.id);
                const isResult = node.id === leafId;
                const perm = node.leafPerm!;
                const displayLabel = `⟨${perm.map(i => `a${i + 1}`).join(",")}⟩`;
                const valueLabel = isResult ? `[${resultArr.join(",")}]` : null;
                return (
                  <g
                    key={node.id}
                    style={{ cursor: "pointer" }}
                    onMouseEnter={() => setHovered(node.id)}
                    onMouseLeave={() => setHovered(null)}
                  >
                    <rect
                      x={node.cx - 44} y={node.cy - 18}
                      width={88} height={isResult ? 38 : 32}
                      rx={8}
                      fill={isResult ? "#d1fae5" : ""}
                      className={isResult ? "" : "fill-slate-100 dark:fill-slate-800 stroke-slate-300 dark:stroke-slate-600"}
                      stroke={isResult ? "#059669" : ""}
                      strokeWidth={isResult ? 2.5 : 1.5}
                    />
                    <text
                      x={node.cx} y={node.cy - 4}
                      textAnchor="middle"
                      fontSize={10}
                      fontWeight={isResult ? "700" : "500"}
                      fill={isResult ? "#065f46" : ""}
                      className={isResult ? "" : "fill-slate-500 dark:fill-slate-400"}
                    >
                      {displayLabel}
                    </text>
                    {valueLabel && (
                      <text
                        x={node.cx} y={node.cy + 12}
                        textAnchor="middle"
                        fontSize={11}
                        fontWeight="700"
                        fill="#059669"
                      >
                        {valueLabel}
                      </text>
                    )}
                    {isResult && (
                      <rect x={node.cx - 46} y={node.cy - 20} width={92} height={42} rx={10} fill="none" stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="3 2" />
                    )}
                  </g>
                );
              }
            })}
          </svg>
        </div>

        {/* Result panel */}
        <div className="mt-4 grid grid-cols-3 gap-3">
          <div className="rounded-xl bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 p-3 text-center">
            <p className="text-2xl font-bold font-mono text-indigo-700 dark:text-indigo-300">{comparisons}</p>
            <p className="text-xs text-indigo-600 dark:text-indigo-400 mt-0.5">比较次数</p>
          </div>
          <div className="rounded-xl bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 p-3 text-center">
            <p className="text-2xl font-bold font-mono text-emerald-700 dark:text-emerald-300">[{resultArr.join(",")}]</p>
            <p className="text-xs text-emerald-600 dark:text-emerald-400 mt-0.5">排序结果</p>
          </div>
          <div className="rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 p-3 text-center">
            <p className="text-2xl font-bold font-mono text-amber-700 dark:text-amber-300">6</p>
            <p className="text-xs text-amber-600 dark:text-amber-400 mt-0.5">叶子数 = 3!</p>
          </div>
        </div>

        {/* Lower bound explanation */}
        <div className="mt-4 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-4 text-xs text-slate-600 dark:text-slate-400 leading-relaxed space-y-1">
          <p><span className="font-semibold text-slate-800 dark:text-slate-200">决策树下界推导：</span></p>
          <p>• 正确排序算法必须能区分 <code className="bg-slate-200 dark:bg-slate-700 px-1 rounded">n! = 3! = 6</code> 种输入排列 → 决策树至少有 6 个叶子</p>
          <p>• 高度为 <code className="bg-slate-200 dark:bg-slate-700 px-1 rounded">h</code> 的二叉树最多 <code className="bg-slate-200 dark:bg-slate-700 px-1 rounded">2ʰ</code> 个叶子 → <code className="bg-slate-200 dark:bg-slate-700 px-1 rounded">2ʰ ≥ 6</code> → <code className="bg-slate-200 dark:bg-slate-700 px-1 rounded">h ≥ ⌈log₂ 6⌉ = 3</code></p>
          <p>• 推广到 n：<code className="bg-slate-200 dark:bg-slate-700 px-1 rounded">h ≥ log₂(n!) = Ω(n log n)</code>（Stirling 近似），这是<span className="font-semibold text-slate-800 dark:text-slate-200">任意基于比较排序算法的最坏情况下界</span></p>
        </div>
      </div>
    </div>
  );
}
