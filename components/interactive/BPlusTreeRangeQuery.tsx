"use client";
import React, { useState } from "react";

// ─── B+ Tree Structure ──────────────────────────────────────────────────────
// A small fixed B+ tree for visualization:
//           [20, 40]
//          /    |    \
//    [10,15] [25,30,35] [45,50,55]
//  Leaf chain: [10,15] → [25,30,35] → [45,50,55]
//  Each leaf also stores a "record" pointer (simplified as page#)

interface Leaf {
  id: string;
  keys: number[];
  page: string; // simulated record page
  x: number;
  y: number;
}

interface InnerNode {
  id: string;
  keys: number[];
  x: number;
  y: number;
}

const INNER: InnerNode[] = [
  { id: "root", keys: [20, 40], x: 300, y: 40 },
  { id: "i1", keys: [15], x: 110, y: 120 },
  { id: "i2", keys: [30], x: 300, y: 120 },
  { id: "i3", keys: [50], x: 490, y: 120 },
];

const LEAVES: Leaf[] = [
  { id: "l1", keys: [10, 15], page: "P1", x: 60,  y: 220 },
  { id: "l2", keys: [20, 25], page: "P2", x: 180, y: 220 },
  { id: "l3", keys: [30, 35], page: "P3", x: 300, y: 220 },
  { id: "l4", keys: [40, 45], page: "P4", x: 420, y: 220 },
  { id: "l5", keys: [50, 55], page: "P5", x: 540, y: 220 },
];

// Inner-to-Leaf edges (index)
const INNER_EDGES: { from: string; to: string }[] = [
  { from: "root", to: "i1" }, { from: "root", to: "i2" }, { from: "root", to: "i3" },
  { from: "i1", to: "l1" }, { from: "i1", to: "l2" },
  { from: "i2", to: "l2" }, { from: "i2", to: "l3" }, { from: "i2", to: "l4" },
  { from: "i3", to: "l4" }, { from: "i3", to: "l5" },
];

// Range query scenarios
interface RangeScenario {
  label: string;
  lo: number;
  hi: number;
  steps: { desc: string; activeInner: string[]; activeLeaves: string[]; scanLeaves: string[] }[];
}

const RANGES: RangeScenario[] = [
  {
    label: "范围查询 [25, 45]",
    lo: 25, hi: 45,
    steps: [
      {
        desc: "从根节点 [20,40] 开始。25 ≥ 20 且 25 < 40，走中间指针 → 内部节点 [30]。",
        activeInner: ["root"],
        activeLeaves: [],
        scanLeaves: [],
      },
      {
        desc: "节点 [30]：25 < 30，走左指针 → 叶节点 L2=[20,25]。找到第一个满足条件的叶子（定位完毕）。",
        activeInner: ["root", "i2"],
        activeLeaves: ["l2"],
        scanLeaves: [],
      },
      {
        desc: "扫描 L2=[20,25]：25 ∈ [25,45] ✓，15 < 25 跳过。收集 25。沿叶链表指针 → L3。",
        activeInner: [],
        activeLeaves: ["l2"],
        scanLeaves: ["l2"],
      },
      {
        desc: "扫描 L3=[30,35]：30 ∈ [25,45] ✓，35 ∈ [25,45] ✓。收集 30, 35。沿链表 → L4。",
        activeInner: [],
        activeLeaves: ["l3"],
        scanLeaves: ["l2", "l3"],
      },
      {
        desc: "扫描 L4=[40,45]：40 ∈ [25,45] ✓，45 ∈ [25,45] ✓。收集 40, 45。沿链表 → L5。",
        activeInner: [],
        activeLeaves: ["l4"],
        scanLeaves: ["l2", "l3", "l4"],
      },
      {
        desc: "扫描 L5=[50,55]：50 > 45，超出范围，停止扫描。\n结果集：{25,30,35,40,45}，共 5 条记录，只访问了 2 个内部节点 + 4 个叶子节点。",
        activeInner: [],
        activeLeaves: [],
        scanLeaves: ["l2", "l3", "l4"],
      },
    ],
  },
  {
    label: "点查询 key=30",
    lo: 30, hi: 30,
    steps: [
      {
        desc: "根节点 [20,40]：30 ≥ 20 且 30 < 40，走中间指针 → 内部节点 [30]。",
        activeInner: ["root"],
        activeLeaves: [],
        scanLeaves: [],
      },
      {
        desc: "节点 [30]：30 = 30（等于路由键），走右指针（B+ 树约定 ≥ 路由键走右）→ 叶 L3=[30,35]。",
        activeInner: ["root", "i2"],
        activeLeaves: ["l3"],
        scanLeaves: [],
      },
      {
        desc: "叶 L3=[30,35]：找到 30 ✓，返回对应记录页 P3。点查询完成，共访问 2 个内部节点 + 1 个叶。",
        activeInner: [],
        activeLeaves: ["l3"],
        scanLeaves: ["l3"],
      },
    ],
  },
  {
    label: "全表扫描（遍历叶链表）",
    lo: -Infinity, hi: Infinity,
    steps: [
      {
        desc: "全表扫描无需走内部节点。直接从最左叶子 L1 开始，沿链表顺序扫描所有叶子。",
        activeInner: [],
        activeLeaves: ["l1"],
        scanLeaves: [],
      },
      {
        desc: "扫描 L1=[10,15]，收集 10, 15。链表指针 → L2。",
        activeInner: [],
        activeLeaves: ["l1"],
        scanLeaves: ["l1"],
      },
      {
        desc: "扫描 L2=[20,25] → L3=[30,35] → L4=[40,45] → L5=[50,55]，全部收集。",
        activeInner: [],
        activeLeaves: ["l5"],
        scanLeaves: ["l1", "l2", "l3", "l4", "l5"],
      },
      {
        desc: "全表扫描完成！共 10 条记录，总 I/O = 5 次（仅叶子层，不走内部节点）。这是 B+ 树相比 B 树的关键优势。",
        activeInner: [],
        activeLeaves: [],
        scanLeaves: ["l1", "l2", "l3", "l4", "l5"],
      },
    ],
  },
];

const NODE_W = 100;
const NODE_H = 36;
const LEAF_W = 90;
const LEAF_H = 44;

export default function BPlusTreeRangeQuery() {
  const [rangeIdx, setRangeIdx] = useState(0);
  const [step, setStep] = useState(0);

  const rng = RANGES[rangeIdx];
  const st = rng.steps[step];

  function gotoRange(i: number) { setRangeIdx(i); setStep(0); }

  const innerMap = Object.fromEntries(INNER.map(n => [n.id, n]));
  const leafMap = Object.fromEntries(LEAVES.map(l => [l.id, l]));

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 p-5 my-6">
      {/* Tab */}
      <div className="flex flex-wrap gap-2 mb-4">
        {RANGES.map((r, i) => (
          <button key={i} onClick={() => gotoRange(i)}
            className={`px-3 py-1.5 rounded-lg text-sm font-semibold transition-colors ${
              rangeIdx === i
                ? "bg-emerald-700 text-white"
                : "bg-slate-700 text-slate-300 hover:bg-slate-600"
            }`}>
            {r.label}
          </button>
        ))}
      </div>

      {/* SVG */}
      <div className="rounded-lg bg-slate-950 border border-slate-800 overflow-x-auto mb-4">
        <svg width="640" height="320" viewBox="0 0 640 320" className="mx-auto block">
          <defs>
            <marker id="arr2" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
              <path d="M0,0 L0,6 L7,3 z" fill="#475569" />
            </marker>
            <marker id="arr2g" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
              <path d="M0,0 L0,6 L7,3 z" fill="#10b981" />
            </marker>
          </defs>

          {/* Inner node edges */}
          {INNER_EDGES.map((e, i) => {
            const fn = innerMap[e.from] ?? null;
            const tn_inner = innerMap[e.to] ?? null;
            const tn_leaf = leafMap[e.to] ?? null;
            const tn = tn_inner ?? tn_leaf;
            if (!fn || !tn) return null;
            const active = st.activeInner.includes(e.from) && (st.activeInner.includes(e.to) || st.activeLeaves.includes(e.to));
            return (
              <line key={i}
                x1={fn.x} y1={fn.y + NODE_H}
                x2={tn_inner ? tn.x : tn.x + LEAF_W / 2} y2={tn.y}
                stroke={active ? "#f59e0b" : "#374151"}
                strokeWidth={active ? 2.5 : 1.2}
                markerEnd={active ? "url(#arr2)" : undefined}
                strokeDasharray={active ? "none" : "4 3"}
              />
            );
          })}

          {/* Leaf chain arrows */}
          {LEAVES.map((l, i) => {
            if (i === LEAVES.length - 1) return null;
            const next = LEAVES[i + 1];
            const scanning = st.scanLeaves.includes(l.id) && st.scanLeaves.includes(next.id);
            return (
              <line key={"chain" + i}
                x1={l.x + LEAF_W} y1={l.y + LEAF_H / 2}
                x2={next.x} y2={next.y + LEAF_H / 2}
                stroke={scanning ? "#10b981" : "#334155"}
                strokeWidth={scanning ? 2.5 : 1.5}
                markerEnd={scanning ? "url(#arr2g)" : "url(#arr2)"}
              />
            );
          })}

          {/* Inner nodes */}
          {INNER.map(n => {
            const active = st.activeInner.includes(n.id);
            return (
              <g key={n.id}>
                <rect x={n.x - NODE_W / 2} y={n.y} width={NODE_W} height={NODE_H} rx={6}
                  fill={active ? "#1e40af" : "#1e293b"}
                  stroke={active ? "#60a5fa" : "#475569"}
                  strokeWidth={active ? 2.5 : 1.5}
                />
                {n.keys.map((k, ki) => (
                  <text key={ki} x={n.x - NODE_W / 2 + (ki + 0.5) * (NODE_W / n.keys.length)} y={n.y + 23}
                    textAnchor="middle" fontSize={13} fontWeight="bold" fill={active ? "#93c5fd" : "#94a3b8"}>
                    {k}
                  </text>
                ))}
                {n.keys.length > 1 && (
                  <line x1={n.x} y1={n.y} x2={n.x} y2={n.y + NODE_H} stroke={active ? "#60a5fa" : "#475569"} strokeWidth={1} />
                )}
              </g>
            );
          })}

          {/* Leaf nodes */}
          {LEAVES.map(l => {
            const active = st.activeLeaves.includes(l.id);
            const scanned = st.scanLeaves.includes(l.id);
            const bg = scanned ? "#064e3b" : active ? "#1e3a5f" : "#0f172a";
            const border = scanned ? "#10b981" : active ? "#3b82f6" : "#334155";
            return (
              <g key={l.id}>
                <rect x={l.x} y={l.y} width={LEAF_W} height={LEAF_H} rx={6}
                  fill={bg} stroke={border} strokeWidth={scanned || active ? 2.5 : 1.5} />
                {/* Keys */}
                {l.keys.map((k, ki) => (
                  <text key={ki} x={l.x + (ki + 0.5) * (LEAF_W / 2)} y={l.y + 18}
                    textAnchor="middle" fontSize={13} fontWeight="bold" fill={scanned ? "#6ee7b7" : active ? "#93c5fd" : "#64748b"}>
                    {k}
                  </text>
                ))}
                {/* Page label */}
                <text x={l.x + LEAF_W / 2} y={l.y + 36} textAnchor="middle" fontSize={10} fill={scanned ? "#34d399" : "#475569"}>
                  {l.page}
                </text>
                {/* Divider */}
                <line x1={l.x + LEAF_W / 2} y1={l.y} x2={l.x + LEAF_W / 2} y2={l.y + LEAF_H}
                  stroke={border} strokeWidth={1} />
              </g>
            );
          })}

          {/* "叶链表" label */}
          <text x={10} y={248} fontSize={10} fill="#475569">← 叶链表</text>
        </svg>
      </div>

      {/* Step desc */}
      <div className="rounded-lg bg-emerald-950 border border-emerald-800 px-4 py-3 mb-4">
        <div className="text-emerald-300 font-bold text-sm mb-1">步骤 {step + 1} / {rng.steps.length}</div>
        <div className="text-slate-200 text-sm whitespace-pre-wrap leading-relaxed">{st.desc}</div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        <button onClick={() => setStep(0)} disabled={step === 0}
          className="px-3 py-1.5 rounded bg-slate-700 text-slate-300 hover:bg-slate-600 disabled:opacity-30 text-lg">⏮</button>
        <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
          className="px-3 py-1.5 rounded bg-slate-700 text-slate-300 hover:bg-slate-600 disabled:opacity-30 text-lg">◀</button>
        <div className="flex-1 h-1.5 rounded-full bg-slate-700 mx-2">
          <div className="h-1.5 rounded-full bg-emerald-500 transition-all duration-300"
            style={{ width: `${((step + 1) / rng.steps.length) * 100}%` }} />
        </div>
        <button onClick={() => setStep(s => Math.min(rng.steps.length - 1, s + 1))} disabled={step === rng.steps.length - 1}
          className="px-3 py-1.5 rounded bg-slate-700 text-slate-300 hover:bg-slate-600 disabled:opacity-30 text-lg">▶</button>
        <button onClick={() => setStep(rng.steps.length - 1)} disabled={step === rng.steps.length - 1}
          className="px-3 py-1.5 rounded bg-slate-700 text-slate-300 hover:bg-slate-600 disabled:opacity-30 text-lg">⏭</button>
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-5 text-xs text-slate-400">
        {[
          { c: "#1e40af", b: "#60a5fa", l: "当前搜索内部节点" },
          { c: "#064e3b", b: "#10b981", l: "已扫描叶子（结果）" },
          { c: "#1e3a5f", b: "#3b82f6", l: "当前访问叶子" },
          { c: "#0f172a", b: "#334155", l: "未访问" },
        ].map(({ c, b, l }) => (
          <span key={l} className="flex items-center gap-1.5">
            <span className="w-4 h-4 rounded border inline-block" style={{ background: c, borderColor: b }} />
            {l}
          </span>
        ))}
      </div>
    </div>
  );
}
