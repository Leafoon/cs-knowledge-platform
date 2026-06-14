"use client";
import React, { useState, useCallback } from "react";

/** Treap Split / Merge 操作动画
 * 每个节点存 (key, priority)，BST on key，Max-heap on priority
 */

interface TNode {
  id: string;
  key: number;
  pri: number;
  x: number;
  y: number;
  color: string; // "left" | "right" | "normal" | "split"
}
interface TEdge { from: string; to: string; dashed?: boolean; }
interface TFrame {
  title: string;
  desc: string;
  nodes: TNode[];
  edges: TEdge[];
  splitLine?: number; // x position of split line
  highlight: string[];
}

// ─── Preset scenarios ─────────────────────────────────────────────────────────
type ScenarioKey = "split" | "merge";

const SCENARIOS: Record<ScenarioKey, { label: string; op: string; frames: TFrame[] }> = {
  split: {
    label: "Split（按键 ≤ 5 分裂）",
    op: "Split(T, k=5)：将 Treap 按键值分裂成 L（key ≤ 5）和 R（key > 5）两棵独立 Treap。",
    frames: [
      {
        title: "初始 Treap（BST on key，Max-Heap on priority）",
        desc: "每个节点格式：key/pri。蓝圈内数字为优先级。这棵 Treap 包含键 1-9，优先级满足堆性质（父 pri > 子 pri）。",
        nodes: [
          { id: "n6", key: 6, pri: 98, x: 300, y: 50, color: "normal" },
          { id: "n3", key: 3, pri: 85, x: 170, y: 130, color: "normal" },
          { id: "n8", key: 8, pri: 72, x: 430, y: 130, color: "normal" },
          { id: "n1", key: 1, pri: 60, x: 90,  y: 210, color: "normal" },
          { id: "n5", key: 5, pri: 78, x: 250, y: 210, color: "normal" },
          { id: "n7", key: 7, pri: 65, x: 360, y: 210, color: "normal" },
          { id: "n9", key: 9, pri: 55, x: 500, y: 210, color: "normal" },
        ],
        edges: [
          { from: "n6", to: "n3" }, { from: "n6", to: "n8" },
          { from: "n3", to: "n1" }, { from: "n3", to: "n5" },
          { from: "n8", to: "n7" }, { from: "n8", to: "n9" },
        ],
        splitLine: undefined,
        highlight: [],
      },
      {
        title: "根节点 key=6 > k=5，向左子树递归 Split",
        desc: "根 key=6 > 5，所以 6 及其右子树（7,8,9）全属于 R。向左子树（以 3 为根）继续 Split(3_subtree, 5)。",
        nodes: [
          { id: "n6", key: 6, pri: 98, x: 300, y: 50, color: "right" },
          { id: "n3", key: 3, pri: 85, x: 170, y: 130, color: "normal" },
          { id: "n8", key: 8, pri: 72, x: 430, y: 130, color: "right" },
          { id: "n1", key: 1, pri: 60, x: 90,  y: 210, color: "normal" },
          { id: "n5", key: 5, pri: 78, x: 250, y: 210, color: "normal" },
          { id: "n7", key: 7, pri: 65, x: 360, y: 210, color: "right" },
          { id: "n9", key: 9, pri: 55, x: 500, y: 210, color: "right" },
        ],
        edges: [
          { from: "n6", to: "n3" }, { from: "n6", to: "n8" },
          { from: "n3", to: "n1" }, { from: "n3", to: "n5" },
          { from: "n8", to: "n7" }, { from: "n8", to: "n9" },
        ],
        splitLine: 300,
        highlight: ["n6"],
      },
      {
        title: "节点 3 (key=3 ≤ 5)，向其右子树 Split(5_subtree, 5)",
        desc: "节点 3 的 key=3 ≤ 5，所以 3 属于 L，向 3 的右子树（以 5 为根）继续 Split(5_subtree, 5)。",
        nodes: [
          { id: "n6", key: 6, pri: 98, x: 300, y: 50, color: "right" },
          { id: "n3", key: 3, pri: 85, x: 170, y: 130, color: "left" },
          { id: "n8", key: 8, pri: 72, x: 430, y: 130, color: "right" },
          { id: "n1", key: 1, pri: 60, x: 90,  y: 210, color: "left" },
          { id: "n5", key: 5, pri: 78, x: 250, y: 210, color: "normal" },
          { id: "n7", key: 7, pri: 65, x: 360, y: 210, color: "right" },
          { id: "n9", key: 9, pri: 55, x: 500, y: 210, color: "right" },
        ],
        edges: [
          { from: "n6", to: "n3" }, { from: "n6", to: "n8" },
          { from: "n3", to: "n1" }, { from: "n3", to: "n5", dashed: true },
          { from: "n8", to: "n7" }, { from: "n8", to: "n9" },
        ],
        splitLine: 300,
        highlight: ["n3", "n5"],
      },
      {
        title: "节点 5 (key=5 ≤ k=5)，其右子为空，Split 返回 (5, null)",
        desc: "节点 5 的 key=5 ≤ 5，5 属于 L，右子树为空，无需继续递归。返回 L_sub=5, R_sub=null。",
        nodes: [
          { id: "n6", key: 6, pri: 98, x: 300, y: 50, color: "right" },
          { id: "n3", key: 3, pri: 85, x: 170, y: 130, color: "left" },
          { id: "n8", key: 8, pri: 72, x: 430, y: 130, color: "right" },
          { id: "n1", key: 1, pri: 60, x: 90,  y: 210, color: "left" },
          { id: "n5", key: 5, pri: 78, x: 250, y: 210, color: "left" },
          { id: "n7", key: 7, pri: 65, x: 360, y: 210, color: "right" },
          { id: "n9", key: 9, pri: 55, x: 500, y: 210, color: "right" },
        ],
        edges: [
          { from: "n6", to: "n3" }, { from: "n6", to: "n8" },
          { from: "n3", to: "n1" }, { from: "n3", to: "n5" },
          { from: "n8", to: "n7" }, { from: "n8", to: "n9" },
        ],
        splitLine: 300,
        highlight: ["n5"],
      },
      {
        title: "Split 完成：L = {1,3,5}，R = {6,7,8,9}",
        desc: "返回展开：节点 3 的右子 = 5（L_sub），节点 6 的左子 = null（R_sub 为 null）。两棵 Treap 各自独立，BST 性质和堆性质均满足。",
        nodes: [
          { id: "n3", key: 3, pri: 85, x: 150, y: 60, color: "left" },
          { id: "n1", key: 1, pri: 60, x: 70,  y: 150, color: "left" },
          { id: "n5", key: 5, pri: 78, x: 230, y: 150, color: "left" },
          { id: "n6", key: 6, pri: 98, x: 480, y: 60, color: "right" },
          { id: "n8", key: 8, pri: 72, x: 590, y: 150, color: "right" },
          { id: "n7", key: 7, pri: 65, x: 520, y: 240, color: "right" },
          { id: "n9", key: 9, pri: 55, x: 660, y: 240, color: "right" },
        ],
        edges: [
          { from: "n3", to: "n1" }, { from: "n3", to: "n5" },
          { from: "n6", to: "n8" },
          { from: "n8", to: "n7" }, { from: "n8", to: "n9" },
        ],
        splitLine: 340,
        highlight: [],
      },
    ],
  },
  merge: {
    label: "Merge（合并两棵 Treap）",
    op: "Merge(L, R)：将 L（所有 key < R 中所有 key）和 R 按优先级合并为一棵满足 Treap 性质的树。",
    frames: [
      {
        title: "初始：两棵独立 Treap L 和 R",
        desc: "前提：L 中所有键 < R 中所有键（否则无法直接 Merge）。根据两树根的优先级决定谁当合并树的根。",
        nodes: [
          { id: "L3", key: 3, pri: 85, x: 130, y: 60, color: "left" },
          { id: "L1", key: 1, pri: 60, x: 60,  y: 150, color: "left" },
          { id: "L5", key: 5, pri: 78, x: 200, y: 150, color: "left" },
          { id: "R6", key: 6, pri: 98, x: 450, y: 60, color: "right" },
          { id: "R8", key: 8, pri: 72, x: 560, y: 150, color: "right" },
          { id: "R7", key: 7, pri: 65, x: 500, y: 240, color: "right" },
          { id: "R9", key: 9, pri: 55, x: 620, y: 240, color: "right" },
        ],
        edges: [
          { from: "L3", to: "L1" }, { from: "L3", to: "L5" },
          { from: "R6", to: "R8" }, { from: "R8", to: "R7" }, { from: "R8", to: "R9" },
        ],
        splitLine: 320,
        highlight: ["L3", "R6"],
      },
      {
        title: "比较根：L 根 pri=85 < R 根 pri=98 → R 根成为合并树的根",
        desc: "R 根（key=6, pri=98）优先级更高，成为新根。合并的左子树 = Merge(L, R的左子树(无)) = L 不变；右子树 = R 的右子树。",
        nodes: [
          { id: "R6", key: 6, pri: 98, x: 330, y: 50, color: "right" },
          { id: "L3", key: 3, pri: 85, x: 150, y: 140, color: "left" },
          { id: "R8", key: 8, pri: 72, x: 500, y: 140, color: "right" },
          { id: "L1", key: 1, pri: 60, x: 70,  y: 230, color: "left" },
          { id: "L5", key: 5, pri: 78, x: 240, y: 230, color: "left" },
          { id: "R7", key: 7, pri: 65, x: 430, y: 230, color: "right" },
          { id: "R9", key: 9, pri: 55, x: 570, y: 230, color: "right" },
        ],
        edges: [
          { from: "R6", to: "L3" , dashed: true }, { from: "R6", to: "R8" },
          { from: "L3", to: "L1" }, { from: "L3", to: "L5" },
          { from: "R8", to: "R7" }, { from: "R8", to: "R9" },
        ],
        splitLine: undefined,
        highlight: ["R6"],
      },
      {
        title: "Merge(L, null) = L，递归返回，直接连接",
        desc: "R 根的左子树为空，Merge(L, null) = L（整棵左 Treap）。直接接到 R6 的左子。合并完成！",
        nodes: [
          { id: "R6", key: 6, pri: 98, x: 330, y: 50, color: "normal" },
          { id: "L3", key: 3, pri: 85, x: 150, y: 140, color: "normal" },
          { id: "R8", key: 8, pri: 72, x: 500, y: 140, color: "normal" },
          { id: "L1", key: 1, pri: 60, x: 70,  y: 230, color: "normal" },
          { id: "L5", key: 5, pri: 78, x: 240, y: 230, color: "normal" },
          { id: "R7", key: 7, pri: 65, x: 430, y: 230, color: "normal" },
          { id: "R9", key: 9, pri: 55, x: 570, y: 230, color: "normal" },
        ],
        edges: [
          { from: "R6", to: "L3" }, { from: "R6", to: "R8" },
          { from: "L3", to: "L1" }, { from: "L3", to: "L5" },
          { from: "R8", to: "R7" }, { from: "R8", to: "R9" },
        ],
        splitLine: undefined,
        highlight: [],
      },
    ],
  },
};

// ─── Renderers ────────────────────────────────────────────────────────────────
const NODE_COLORS = {
  left:   { fill: "#3b82f6", stroke: "#2563eb" },
  right:  { fill: "#ef4444", stroke: "#dc2626" },
  normal: { fill: "#6b7280", stroke: "#4b5563" },
  split:  { fill: "#8b5cf6", stroke: "#7c3aed" },
};

function TreapSVG({ frame }: { frame: TFrame }) {
  return (
    <svg viewBox="0 0 700 310" className="w-full" style={{ maxHeight: 290 }}>
      {/* Split line */}
      {frame.splitLine !== undefined && (
        <line x1={frame.splitLine} y1={10} x2={frame.splitLine} y2={290} stroke="#f59e0b" strokeWidth={2} strokeDasharray="6 4" />
      )}
      {frame.splitLine !== undefined && (
        <>
          <text x={frame.splitLine - 60} y={28} fill="#3b82f6" fontSize={11} fontWeight="bold">L (key ≤ 5)</text>
          <text x={frame.splitLine + 10} y={28} fill="#ef4444" fontSize={11} fontWeight="bold">R (key &gt; 5)</text>
        </>
      )}

      {/* Edges */}
      {frame.edges.map((e) => {
        const from = frame.nodes.find((n) => n.id === e.from);
        const to = frame.nodes.find((n) => n.id === e.to);
        if (!from || !to) return null;
        return (
          <line
            key={`${e.from}-${e.to}`}
            x1={from.x} y1={from.y + 22}
            x2={to.x} y2={to.y - 22}
            stroke={e.dashed ? "#f59e0b" : "#94a3b8"}
            strokeWidth={e.dashed ? 2 : 1.5}
            strokeDasharray={e.dashed ? "5 3" : undefined}
          />
        );
      })}

      {/* Nodes */}
      {frame.nodes.map((n) => {
        const c = NODE_COLORS[n.color as keyof typeof NODE_COLORS] ?? NODE_COLORS.normal;
        const isHL = frame.highlight.includes(n.id);
        return (
          <g key={n.id}>
            <circle cx={n.x} cy={n.y} r={22} fill={c.fill} stroke={isHL ? "#f97316" : c.stroke} strokeWidth={isHL ? 3 : 1.5} />
            {/* key (big) */}
            <text x={n.x} y={n.y - 3} textAnchor="middle" dominantBaseline="middle" fill="white" fontSize={12} fontWeight="bold">{n.key}</text>
            {/* priority (small) */}
            <text x={n.x} y={n.y + 10} textAnchor="middle" dominantBaseline="middle" fill="rgba(255,255,255,0.7)" fontSize={9}>p={n.pri}</text>
          </g>
        );
      })}
    </svg>
  );
}

export default function TreapSplitMerge() {
  const [scenario, setScenario] = useState<ScenarioKey>("split");
  const [frameIdx, setFrameIdx] = useState(0);

  const preset = SCENARIOS[scenario];
  const frame = preset.frames[frameIdx];
  const total = preset.frames.length;

  const handleScenario = useCallback((s: ScenarioKey) => { setScenario(s); setFrameIdx(0); }, []);

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5 my-6 shadow-sm">
      <h3 className="text-base font-bold text-slate-800 dark:text-slate-100 mb-1">🌲 Treap Split / Merge 动画</h3>
      <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
        节点格式：大数字 = key，小字 = priority。蓝色 = L 树，红色 = R 树。黄色虚线 = 分裂点。
      </p>

      {/* Scenario selector */}
      <div className="flex gap-2 mb-4">
        {(Object.keys(SCENARIOS) as ScenarioKey[]).map((s) => (
          <button key={s} onClick={() => handleScenario(s)}
            className={`px-4 py-2 text-xs font-medium rounded-lg transition-all ${scenario === s ? "bg-purple-600 text-white shadow" : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-purple-50"}`}>
            {SCENARIOS[s].label}
          </button>
        ))}
      </div>

      <div className="text-xs bg-purple-50 dark:bg-purple-900/20 border-l-4 border-purple-400 px-4 py-2 rounded-r-lg mb-4 text-purple-700 dark:text-purple-300">
        <span className="font-semibold">操作：</span>{preset.op}
      </div>

      <div className="bg-slate-50 dark:bg-slate-800/60 rounded-xl p-3 mb-4">
        <TreapSVG frame={frame} />
      </div>

      <div className="mb-4 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 px-4 py-2">
        <div className="text-xs font-semibold text-amber-700 dark:text-amber-300 mb-0.5">步骤 {frameIdx + 1}/{total}：{frame.title}</div>
        <div className="text-xs text-amber-600 dark:text-amber-400">{frame.desc}</div>
      </div>

      <div className="flex items-center gap-3">
        <button onClick={() => setFrameIdx(0)} disabled={frameIdx === 0} className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-xs disabled:opacity-40">⏮</button>
        <button onClick={() => setFrameIdx((f) => Math.max(0, f - 1))} disabled={frameIdx === 0} className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-xs disabled:opacity-40">◀ 上一步</button>
        <div className="flex-1 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
          <div className="h-full bg-purple-500 rounded-full transition-all" style={{ width: `${((frameIdx + 1) / total) * 100}%` }} />
        </div>
        <button onClick={() => setFrameIdx((f) => Math.min(total - 1, f + 1))} disabled={frameIdx === total - 1} className="px-3 py-1.5 rounded-lg bg-purple-600 text-white text-xs disabled:opacity-40">下一步 ▶</button>
        <button onClick={() => setFrameIdx(total - 1)} disabled={frameIdx === total - 1} className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-xs disabled:opacity-40">⏭</button>
      </div>

      {/* Treap property reminder */}
      <div className="mt-4 flex flex-wrap gap-4 text-xs text-slate-500 dark:text-slate-400">
        <span>🔑 <strong>BST 性质</strong>：左 key &lt; 根 key &lt; 右 key</span>
        <span>🏔 <strong>堆性质</strong>：父 priority ≥ 子 priority（max-heap）</span>
        <span>🎲 <strong>期望高度</strong>：O(log n)（与随机 BST 等价）</span>
      </div>
    </div>
  );
}
