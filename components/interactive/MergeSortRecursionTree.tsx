"use client";
import React, { useState, useEffect, useRef } from "react";

/* ─── Data model ─────────────────────────────────────────────────────────── */

// The example array: [5, 2, 4, 6, 1, 3]
// Tree node layout (index in 8-slot row grid, 0-based):
//   Level 0: A  [5,2,4,6,1,3]  slots 0-7
//   Level 1: B  [5,2,4] slots 0-3   C  [6,1,3] slots 4-7
//   Level 2: D  [5,2]  E [4]  F [6,1]  G [3]   (each 2 slots)
//   Level 3: H  [5]   I [2]  (slots 0,1)   J [6]  K [1] (slots 4,5)

type NodeId = "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" | "K";
type NodeStatus = "hidden" | "idle" | "dividing" | "merging" | "sorted";

interface TreeNode {
  id: NodeId;
  level: number;    // 0–3
  slot: number;     // start slot in 8-grid
  span: number;     // span in 8-grid
  parent: NodeId | null;
  children: NodeId[];
}

const TREE_NODES: TreeNode[] = [
  { id: "A", level: 0, slot: 0, span: 8, parent: null,  children: ["B", "C"] },
  { id: "B", level: 1, slot: 0, span: 4, parent: "A",   children: ["D", "E"] },
  { id: "C", level: 1, slot: 4, span: 4, parent: "A",   children: ["F", "G"] },
  { id: "D", level: 2, slot: 0, span: 2, parent: "B",   children: ["H", "I"] },
  { id: "E", level: 2, slot: 2, span: 2, parent: "B",   children: [] },
  { id: "F", level: 2, slot: 4, span: 2, parent: "C",   children: ["J", "K"] },
  { id: "G", level: 2, slot: 6, span: 2, parent: "C",   children: [] },
  { id: "H", level: 3, slot: 0, span: 1, parent: "D",   children: [] },
  { id: "I", level: 3, slot: 1, span: 1, parent: "D",   children: [] },
  { id: "J", level: 3, slot: 4, span: 1, parent: "F",   children: [] },
  { id: "K", level: 3, slot: 5, span: 1, parent: "F",   children: [] },
];

const NODE_MAP = Object.fromEntries(TREE_NODES.map(n => [n.id, n])) as Record<NodeId, TreeNode>;

/* ─── Steps ──────────────────────────────────────────────────────────────── */

interface Step {
  phase: "divide" | "merge" | "done";
  description: string;
  detail: string;
  visible: NodeId[];
  status: Partial<Record<NodeId, NodeStatus>>;
  arrays: Partial<Record<NodeId, number[]>>;
  activeEdges: [NodeId, NodeId][];
}

const STEPS: Step[] = [
  {
    phase: "divide", description: "初始数组", detail: "从完整数组 [5,2,4,6,1,3] 开始，准备递归地「劈开」它",
    visible: ["A"],
    status: { A: "idle" },
    arrays: { A: [5,2,4,6,1,3] },
    activeEdges: [],
  },
  {
    phase: "divide", description: "分：劈成两半", detail: "从中点劈开 → 左半 [5,2,4]，右半 [6,1,3]",
    visible: ["A","B","C"],
    status: { A: "dividing", B: "idle", C: "idle" },
    arrays: { A: [5,2,4,6,1,3], B: [5,2,4], C: [6,1,3] },
    activeEdges: [["A","B"],["A","C"]],
  },
  {
    phase: "divide", description: "分：继续劈左半", detail: "对 [5,2,4] 继续劈 → [5,2] 和 [4]",
    visible: ["A","B","C","D","E"],
    status: { A: "idle", B: "dividing", C: "idle", D: "idle", E: "idle" },
    arrays: { A: [5,2,4,6,1,3], B: [5,2,4], C: [6,1,3], D: [5,2], E: [4] },
    activeEdges: [["B","D"],["B","E"]],
  },
  {
    phase: "divide", description: "分：继续劈右半", detail: "对 [6,1,3] 继续劈 → [6,1] 和 [3]",
    visible: ["A","B","C","D","E","F","G"],
    status: { A: "idle", B: "idle", C: "dividing", D: "idle", E: "idle", F: "idle", G: "idle" },
    arrays: { A: [5,2,4,6,1,3], B: [5,2,4], C: [6,1,3], D: [5,2], E: [4], F: [6,1], G: [3] },
    activeEdges: [["C","F"],["C","G"]],
  },
  {
    phase: "divide", description: "分：[5,2] 劈到叶子", detail: "[5,2] → [5] 和 [2]，长度 1 的子数组天然有序",
    visible: ["A","B","C","D","E","F","G","H","I"],
    status: { A: "idle", B: "idle", C: "idle", D: "dividing", E: "sorted", F: "idle", G: "sorted", H: "sorted", I: "sorted" },
    arrays: { A: [5,2,4,6,1,3], B: [5,2,4], C: [6,1,3], D: [5,2], E: [4], F: [6,1], G: [3], H: [5], I: [2] },
    activeEdges: [["D","H"],["D","I"]],
  },
  {
    phase: "divide", description: "分：[6,1] 劈到叶子", detail: "[6,1] → [6] 和 [1]，所有叶子节点均已就绪",
    visible: ["A","B","C","D","E","F","G","H","I","J","K"],
    status: { A: "idle", B: "idle", C: "idle", D: "idle", E: "sorted", F: "dividing", G: "sorted", H: "sorted", I: "sorted", J: "sorted", K: "sorted" },
    arrays: { A: [5,2,4,6,1,3], B: [5,2,4], C: [6,1,3], D: [5,2], E: [4], F: [6,1], G: [3], H: [5], I: [2], J: [6], K: [1] },
    activeEdges: [["F","J"],["F","K"]],
  },
  {
    phase: "merge", description: "合：[5]+[2] → [2,5]", detail: "比较 5 与 2：2 < 5，取 2；再取 5 → 合并结果 [2,5]",
    visible: ["A","B","C","D","E","F","G","H","I","J","K"],
    status: { A: "idle", B: "idle", C: "idle", D: "merging", E: "sorted", F: "idle", G: "sorted", H: "sorted", I: "sorted", J: "sorted", K: "sorted" },
    arrays: { A: [5,2,4,6,1,3], B: [5,2,4], C: [6,1,3], D: [2,5], E: [4], F: [6,1], G: [3], H: [5], I: [2], J: [6], K: [1] },
    activeEdges: [["H","D"],["I","D"]],
  },
  {
    phase: "merge", description: "合：[2,5]+[4] → [2,4,5]", detail: "MERGE [2,5] 与 [4]：2<4 取 2，4<5 取 4，取 5 → [2,4,5]",
    visible: ["A","B","C","D","E","F","G","H","I","J","K"],
    status: { A: "idle", B: "merging", C: "idle", D: "sorted", E: "sorted", F: "idle", G: "sorted", H: "sorted", I: "sorted", J: "sorted", K: "sorted" },
    arrays: { A: [5,2,4,6,1,3], B: [2,4,5], C: [6,1,3], D: [2,5], E: [4], F: [6,1], G: [3], H: [5], I: [2], J: [6], K: [1] },
    activeEdges: [["D","B"],["E","B"]],
  },
  {
    phase: "merge", description: "合：[6]+[1] → [1,6]", detail: "比较 6 与 1：1 < 6，取 1；再取 6 → 合并结果 [1,6]",
    visible: ["A","B","C","D","E","F","G","H","I","J","K"],
    status: { A: "idle", B: "sorted", C: "idle", D: "sorted", E: "sorted", F: "merging", G: "sorted", H: "sorted", I: "sorted", J: "sorted", K: "sorted" },
    arrays: { A: [5,2,4,6,1,3], B: [2,4,5], C: [6,1,3], D: [2,5], E: [4], F: [1,6], G: [3], H: [5], I: [2], J: [6], K: [1] },
    activeEdges: [["J","F"],["K","F"]],
  },
  {
    phase: "merge", description: "合：[1,6]+[3] → [1,3,6]", detail: "MERGE [1,6] 与 [3]：1<3 取 1，3<6 取 3，取 6 → [1,3,6]",
    visible: ["A","B","C","D","E","F","G","H","I","J","K"],
    status: { A: "idle", B: "sorted", C: "merging", D: "sorted", E: "sorted", F: "sorted", G: "sorted", H: "sorted", I: "sorted", J: "sorted", K: "sorted" },
    arrays: { A: [5,2,4,6,1,3], B: [2,4,5], C: [1,3,6], D: [2,5], E: [4], F: [1,6], G: [3], H: [5], I: [2], J: [6], K: [1] },
    activeEdges: [["F","C"],["G","C"]],
  },
  {
    phase: "merge", description: "合：[2,4,5]+[1,3,6] → 最终有序！", detail: "最后一次 MERGE：逐一比较取小值，得到 [1,2,3,4,5,6]",
    visible: ["A","B","C","D","E","F","G","H","I","J","K"],
    status: { A: "merging", B: "sorted", C: "sorted", D: "sorted", E: "sorted", F: "sorted", G: "sorted", H: "sorted", I: "sorted", J: "sorted", K: "sorted" },
    arrays: { A: [1,2,3,4,5,6], B: [2,4,5], C: [1,3,6], D: [2,5], E: [4], F: [1,6], G: [3], H: [5], I: [2], J: [6], K: [1] },
    activeEdges: [["B","A"],["C","A"]],
  },
  {
    phase: "done", description: "🎉 排序完成！", detail: "归并排序共进行了 10 次合并操作，总比较次数 ≤ n log₂n = 6×log₂6 ≈ 15 次",
    visible: ["A","B","C","D","E","F","G","H","I","J","K"],
    status: { A: "sorted", B: "sorted", C: "sorted", D: "sorted", E: "sorted", F: "sorted", G: "sorted", H: "sorted", I: "sorted", J: "sorted", K: "sorted" },
    arrays: { A: [1,2,3,4,5,6], B: [2,4,5], C: [1,3,6], D: [2,5], E: [4], F: [1,6], G: [3], H: [5], I: [2], J: [6], K: [1] },
    activeEdges: [],
  },
];

/* ─── Helpers ─────────────────────────────────────────────────────────────── */

function nodeColor(status: NodeStatus): { bg: string; border: string; text: string; label: string } {
  switch (status) {
    case "dividing": return { bg: "bg-blue-100 dark:bg-blue-900/40", border: "border-blue-500 dark:border-blue-400", text: "text-blue-800 dark:text-blue-200", label: "正在分割" };
    case "merging":  return { bg: "bg-amber-100 dark:bg-amber-900/40", border: "border-amber-500 dark:border-amber-400", text: "text-amber-800 dark:text-amber-200", label: "正在合并" };
    case "sorted":   return { bg: "bg-emerald-100 dark:bg-emerald-900/40", border: "border-emerald-500 dark:border-emerald-400", text: "text-emerald-800 dark:text-emerald-200", label: "已有序" };
    case "idle":     return { bg: "bg-slate-100 dark:bg-slate-800", border: "border-slate-300 dark:border-slate-600", text: "text-slate-700 dark:text-slate-300", label: "等待" };
    default:         return { bg: "bg-slate-50 dark:bg-slate-900", border: "border-slate-200 dark:border-slate-700", text: "text-slate-400 dark:text-slate-500", label: "" };
  }
}

/* ─── Tree Node Card ─────────────────────────────────────────────────────── */

function NodeCard({ node, status, arr, isLeaf }: {
  node: TreeNode;
  status: NodeStatus;
  arr: number[];
  isLeaf: boolean;
}) {
  const c = nodeColor(status);
  const isActive = status === "dividing" || status === "merging";
  return (
    <div
      className={`
        relative flex flex-col items-center justify-center rounded-xl border-2 p-2
        transition-all duration-300 min-h-[52px]
        ${c.bg} ${c.border}
        ${isActive ? "shadow-lg ring-2 ring-offset-1 ring-offset-white dark:ring-offset-slate-900 " + (status === "dividing" ? "ring-blue-400" : "ring-amber-400") : "shadow-sm"}
      `}
      style={{ gridColumn: `${node.slot + 1} / span ${node.span}` }}
    >
      {isLeaf && arr.length === 1 && (
        <span className={`text-[10px] font-medium mb-0.5 ${c.text} opacity-70`}>叶</span>
      )}
      <div className="flex gap-1 flex-wrap justify-center">
        {arr.map((v, i) => (
          <span key={i} className={`text-sm font-bold font-mono ${c.text}`}>{v}</span>
        ))}
      </div>
      {isActive && (
        <span className={`absolute -top-2.5 left-1/2 -translate-x-1/2 text-[9px] font-semibold px-1.5 py-0.5 rounded-full whitespace-nowrap
          ${status === "dividing" ? "bg-blue-500 text-white" : "bg-amber-500 text-white"}`}>
          {c.label}
        </span>
      )}
    </div>
  );
}

/* ─── Main Component ─────────────────────────────────────────────────────── */

export default function MergeSortRecursionTree() {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const cur = STEPS[step];
  const total = STEPS.length - 1;

  useEffect(() => {
    if (playing) {
      if (step >= total) { setPlaying(false); return; }
      timerRef.current = setTimeout(() => setStep(s => s + 1), 1400);
    }
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [playing, step, total]);

  const goTo = (n: number) => { setPlaying(false); setStep(Math.max(0, Math.min(total, n))); };
  const reset = () => { setPlaying(false); setStep(0); };

  const phaseColors: Record<string, string> = {
    divide: "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border-blue-300 dark:border-blue-700",
    merge:  "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 border-amber-300 dark:border-amber-700",
    done:   "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border-emerald-300 dark:border-emerald-700",
  };

  const phaseLabel = { divide: "↓ 分割阶段", merge: "↑ 合并阶段", done: "✓ 完成" };

  // Group nodes by level for rendering
  const levels = [0, 1, 2, 3];

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 overflow-hidden shadow-sm">
      {/* Header */}
      <div className="px-5 py-4 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
        <div>
          <h3 className="font-bold text-slate-800 dark:text-slate-100 text-base">归并排序递归树</h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">示例数组 [5, 2, 4, 6, 1, 3]</p>
        </div>
        <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${phaseColors[cur.phase]}`}>
          {phaseLabel[cur.phase]}
        </span>
      </div>

      <div className="p-5">
        {/* Step counter */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex gap-2">
            {["分割", "合并", "完成"].map((label, i) => (
              <div key={label} className="flex items-center gap-1.5">
                <span className={`w-2.5 h-2.5 rounded-full ${i === 0 ? "bg-blue-500" : i === 1 ? "bg-amber-500" : "bg-emerald-500"}`} />
                <span className="text-xs text-slate-500 dark:text-slate-400">{label}</span>
              </div>
            ))}
          </div>
          <span className="text-xs font-mono text-slate-400 dark:text-slate-500">{step}/{total}</span>
        </div>

        {/* Tree visualization — 4 levels */}
        <div className="space-y-1.5 select-none">
          {levels.map(level => {
            const levelNodes = TREE_NODES.filter(n => n.level === level);
            const anyVisible = levelNodes.some(n => cur.visible.includes(n.id));
            return (
              <div key={level}>
                {/* Level label */}
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-[10px] font-medium text-slate-400 dark:text-slate-600 w-12 flex-shrink-0">
                    L{level}{level === 0 ? " (根)" : level === 3 ? " (叶)" : ""}
                  </span>
                  {/* 8-slot grid row */}
                  <div
                    className="flex-1 grid gap-1.5"
                    style={{ gridTemplateColumns: "repeat(8, 1fr)" }}
                  >
                    {levelNodes.map(node => {
                      const isVisible = cur.visible.includes(node.id);
                      const status: NodeStatus = isVisible ? (cur.status[node.id] ?? "idle") : "hidden";
                      const arr = cur.arrays[node.id] ?? node.children.length === 0 ? [] : [];
                      const isLeaf = node.children.length === 0;

                      if (!isVisible) {
                        // Render a transparent placeholder to maintain grid layout
                        return (
                          <div
                            key={node.id}
                            style={{ gridColumn: `${node.slot + 1} / span ${node.span}` }}
                            className="min-h-[52px] rounded-xl border-2 border-dashed border-slate-100 dark:border-slate-800 opacity-30"
                          />
                        );
                      }
                      return (
                        <NodeCard
                          key={node.id}
                          node={node}
                          status={status}
                          arr={arr}
                          isLeaf={isLeaf}
                        />
                      );
                    })}
                    {/* Fill in empty grid slots for level 3 (positions 2,3,6,7 not used) */}
                    {level === 3 && [2, 3, 6, 7].map(slot => (
                      <div
                        key={slot}
                        className="min-h-[52px] rounded-xl opacity-0"
                        style={{ gridColumn: `${slot + 1} / span 1` }}
                      />
                    ))}
                  </div>
                </div>

                {/* Connector: arrows between levels */}
                {level < 3 && anyVisible && (
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-12 flex-shrink-0" />
                    <div className="flex-1 relative h-4">
                      <svg className="absolute inset-0 w-full h-full overflow-visible" preserveAspectRatio="none">
                        {cur.activeEdges.map(([from, to]) => {
                          const fromNode = NODE_MAP[from];
                          const toNode = NODE_MAP[to];
                          if (fromNode.level !== level && toNode.level !== level + 1) return null;
                          if (fromNode.level === level && toNode.level === level + 1) {
                            // Draw a downward connector line
                            const fromCenterPct = (fromNode.slot + fromNode.span / 2) / 8 * 100;
                            const toCenterPct = (toNode.slot + toNode.span / 2) / 8 * 100;
                            return (
                              <line
                                key={`${from}-${to}`}
                                x1={`${fromCenterPct}%`} y1="0"
                                x2={`${toCenterPct}%`} y2="100%"
                                stroke={cur.phase === "divide" ? "#3b82f6" : "#f59e0b"}
                                strokeWidth="2"
                                strokeDasharray="4 2"
                                opacity="0.8"
                              />
                            );
                          }
                          return null;
                        })}
                        {/* Static soft grey lines for all visible parent→child edges */}
                        {TREE_NODES
                          .filter(n => n.level === level && cur.visible.includes(n.id))
                          .flatMap(parent =>
                            parent.children
                              .filter(childId => cur.visible.includes(childId))
                              .map(childId => {
                                const child = NODE_MAP[childId];
                                const isActive = cur.activeEdges.some(([f, t]) => f === parent.id && t === childId || f === childId && t === parent.id);
                                if (isActive) return null;
                                const fromCenterPct = (parent.slot + parent.span / 2) / 8 * 100;
                                const toCenterPct = (child.slot + child.span / 2) / 8 * 100;
                                return (
                                  <line
                                    key={`${parent.id}-${childId}-static`}
                                    x1={`${fromCenterPct}%`} y1="0"
                                    x2={`${toCenterPct}%`} y2="100%"
                                    stroke="currentColor"
                                    strokeWidth="1"
                                    className="text-slate-200 dark:text-slate-700"
                                    opacity="0.8"
                                  />
                                );
                              })
                          )}
                      </svg>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Description panel */}
        <div className="mt-4 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-4">
          <p className="font-semibold text-slate-800 dark:text-slate-100 text-sm">{cur.description}</p>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">{cur.detail}</p>
        </div>

        {/* Progress bar */}
        <div className="mt-4 h-1.5 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-300 bg-gradient-to-r from-blue-500 to-emerald-500"
            style={{ width: `${(step / total) * 100}%` }}
          />
        </div>

        {/* Controls */}
        <div className="mt-4 flex items-center justify-between">
          <div className="flex gap-2">
            <button onClick={reset} className="px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors">
              重置
            </button>
            <button onClick={() => goTo(step - 1)} disabled={step === 0} className="px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
              ← 上一步
            </button>
            <button
              onClick={() => playing ? setPlaying(false) : setPlaying(true)}
              disabled={step >= total}
              className={`px-4 py-1.5 text-xs rounded-lg font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed
                ${playing ? "bg-amber-500 hover:bg-amber-600 text-white" : "bg-indigo-500 hover:bg-indigo-600 text-white"}`}
            >
              {playing ? "⏸ 暂停" : "▶ 播放"}
            </button>
            <button onClick={() => goTo(step + 1)} disabled={step >= total} className="px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
              下一步 →
            </button>
          </div>
          <span className="text-xs text-slate-400 dark:text-slate-500">
            T(n) = 2T(n/2) + Θ(n) = Θ(n log n)
          </span>
        </div>
      </div>
    </div>
  );
}
