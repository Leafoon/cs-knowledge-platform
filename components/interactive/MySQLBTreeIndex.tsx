"use client";
import React, { useState } from "react";

// ─── Types ──────────────────────────────────────────────────────────────────
type IndexType = "clustered" | "secondary";

interface IndexStep {
  title: string;
  desc: string;
  highlights: {
    clustered?: string[];   // node ids in clustered index
    secondary?: string[];   // node ids in secondary index
    dataRows?: number[];    // highlighted row ids
  };
  arrow?: { from: string; to: string; label: string }[];
}

// Simplified table: users(id PK, name, email, age)
// Clustered index on id → data rows are in leaf
// Secondary index on age
const TABLE_ROWS = [
  { id: 10, name: "Alice",   email: "a@x.com", age: 25 },
  { id: 20, name: "Bob",     email: "b@x.com", age: 30 },
  { id: 30, name: "Carol",   email: "c@x.com", age: 22 },
  { id: 40, name: "Dave",    email: "d@x.com", age: 28 },
  { id: 50, name: "Eve",     email: "e@x.com", age: 25 },
  { id: 60, name: "Frank",   email: "f@x.com", age: 35 },
];

// ─── Scenarios ──────────────────────────────────────────────────────────────
const SCENARIOS: { label: string; query: string; index: IndexType; steps: IndexStep[] }[] = [
  {
    label: "主键查询 WHERE id=40",
    query: "SELECT * FROM users WHERE id = 40",
    index: "clustered",
    steps: [
      {
        title: "查询走聚簇索引",
        desc: "InnoDB 主键 B+ 树（聚簇索引）：叶子节点直接存储完整行数据。根节点 → 内部节点 → 叶子节点，每层一次磁盘 I/O。",
        highlights: { clustered: ["root_c"], dataRows: [] },
      },
      {
        title: "从根向下定位：id=40 在 [30,50) 区间",
        desc: "根节点 [20,40]：40 ≥ 40，走右指针。内部节点确定叶子范围。",
        highlights: { clustered: ["root_c", "inner_c_r"], dataRows: [] },
      },
      {
        title: "到达叶子节点，直接读取整行",
        desc: "叶子节点 [40: Dave, 28] 直接包含完整行。一次 I/O 即可完成查询，返回所有列。无需额外回表。",
        highlights: { clustered: ["leaf_c_40"], dataRows: [40] },
      },
    ],
  },
  {
    label: "辅助索引 WHERE age=25",
    query: "SELECT * FROM users WHERE age = 25",
    index: "secondary",
    steps: [
      {
        title: "查询走辅助索引（二级索引）",
        desc: "辅助索引 B+ 树的叶子节点存储的是 (age, 主键id) 对，而非完整行数据。必须先查辅助索引，再回表。",
        highlights: { secondary: ["root_s"], dataRows: [] },
      },
      {
        title: "辅助索引定位 age=25 的叶子节点",
        desc: "辅助索引根 [25,30] → 左子树叶子。找到 age=25 的条目：(25, id=10), (25, id=50)。",
        highlights: { secondary: ["root_s", "leaf_s_25"], dataRows: [] },
      },
      {
        title: "回表：用 id=10 查聚簇索引",
        desc: "从辅助索引拿到 id=10，再去聚簇索引找完整行。这就是「回表（Double Lookup）」，每条匹配记录需要额外一次 I/O。",
        highlights: { clustered: ["leaf_c_10"], secondary: ["leaf_s_25"], dataRows: [10] },
        arrow: [{ from: "leaf_s_25", to: "leaf_c_10", label: "回表 id=10" }],
      },
      {
        title: "回表：用 id=50 查聚簇索引",
        desc: "继续回表查 id=50，找到 Eve（age=25）。共 2 次回表 I/O + 辅助索引 I/O。\n若结果集大，回表成本高 → 这是覆盖索引（covering index）优化的动机！",
        highlights: { clustered: ["leaf_c_50"], secondary: ["leaf_s_25"], dataRows: [10, 50] },
        arrow: [{ from: "leaf_s_25", to: "leaf_c_50", label: "回表 id=50" }],
      },
    ],
  },
  {
    label: "覆盖索引（无需回表）",
    query: "SELECT age, id FROM users WHERE age = 25",
    index: "secondary",
    steps: [
      {
        title: "覆盖索引：查询列全在辅助索引中",
        desc: "查询只需要 age 和 id 两列，而辅助索引叶子恰好存储了 (age, id)。所有需要的列都能从辅助索引中直接读取，不需要回表！",
        highlights: { secondary: ["root_s", "leaf_s_25"], dataRows: [] },
      },
      {
        title: "直接从辅助索引读取结果",
        desc: "辅助索引叶子中 age=25 的条目：(25,10), (25,50) → 直接返回，无需访问聚簇索引。覆盖索引将查询 I/O 从 O(k) 降到 O(log n)！",
        highlights: { secondary: ["leaf_s_25"], dataRows: [10, 50] },
      },
    ],
  },
];

// ─── SVG Components ──────────────────────────────────────────────────────────
function ClusteredIndex({ highlights }: { highlights: string[] }) {
  // Simplified 3-level B+ tree for primary key
  const nodes = [
    { id: "root_c", label: "[20 | 40]", x: 200, y: 20, w: 100, h: 32 },
    { id: "inner_c_l", label: "[10 | 20]", x: 80, y: 90, w: 90, h: 32 },
    { id: "inner_c_r", label: "[40 | 50]", x: 280, y: 90, w: 90, h: 32 },
    { id: "leaf_c_10", label: "10:Alice,25", x: 10, y: 165, w: 100, h: 38 },
    { id: "leaf_c_20", label: "20:Bob,30",  x: 120,y: 165, w: 90,  h: 38 },
    { id: "leaf_c_30", label: "30:Carol,22",x: 10, y: 215, w: 100, h: 38 },
    { id: "leaf_c_40", label: "40:Dave,28", x: 220, y: 165, w: 100, h: 38 },
    { id: "leaf_c_50", label: "50:Eve,25",  x: 330, y: 165, w: 90,  h: 38 },
    { id: "leaf_c_60", label: "60:Frank,35",x: 330, y: 215, w: 100, h: 38 },
  ];
  const edges = [
    ["root_c","inner_c_l"],["root_c","inner_c_r"],
    ["inner_c_l","leaf_c_10"],["inner_c_l","leaf_c_20"],
    ["inner_c_r","leaf_c_40"],["inner_c_r","leaf_c_50"],
  ];
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
  return (
    <svg width="440" height="280" className="block">
      <text x={220} y={14} textAnchor="middle" fontSize={11} fill="#60a5fa">聚簇索引（主键 B+ 树）</text>
      {edges.map(([a, b], i) => {
        const fn = nodeMap[a], tn = nodeMap[b];
        return <line key={i} x1={fn.x + fn.w / 2} y1={fn.y + fn.h}
          x2={tn.x + tn.w / 2} y2={tn.y} stroke="#334155" strokeWidth={1.2} />;
      })}
      {nodes.map(n => {
        const hl = highlights.includes(n.id);
        return (
          <g key={n.id}>
            <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={5}
              fill={hl ? "#1e40af" : "#0f172a"} stroke={hl ? "#60a5fa" : "#334155"} strokeWidth={hl ? 2 : 1} />
            <text x={n.x + n.w / 2} y={n.y + n.h / 2 + 4} textAnchor="middle"
              fontSize={10} fill={hl ? "#93c5fd" : "#64748b"} fontWeight={hl ? "bold" : "normal"}>
              {n.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function SecondaryIndex({ highlights }: { highlights: string[] }) {
  const nodes = [
    { id: "root_s",   label: "[25 | 30]",   x: 140, y: 20, w: 100, h: 32 },
    { id: "leaf_s_22",label: "22→id=30",     x: 20,  y: 90, w: 80,  h: 32 },
    { id: "leaf_s_25",label: "25→id=10,50",  x: 110, y: 90, w: 110,  h: 32 },
    { id: "leaf_s_28",label: "28→id=40",     x: 230, y: 90, w: 80,  h: 32 },
    { id: "leaf_s_30",label: "30→id=20",     x: 20,  y: 140, w: 80,  h: 32 },
    { id: "leaf_s_35",label: "35→id=60",     x: 110, y: 140, w: 80,  h: 32 },
  ];
  const edges = [
    ["root_s","leaf_s_22"],["root_s","leaf_s_25"],["root_s","leaf_s_28"],
  ];
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
  return (
    <svg width="340" height="200" className="block">
      <text x={170} y={14} textAnchor="middle" fontSize={11} fill="#a78bfa">辅助索引（age B+ 树）</text>
      {edges.map(([a, b], i) => {
        const fn = nodeMap[a], tn = nodeMap[b];
        return <line key={i} x1={fn.x + fn.w / 2} y1={fn.y + fn.h}
          x2={tn.x + tn.w / 2} y2={tn.y} stroke="#334155" strokeWidth={1.2} />;
      })}
      {nodes.map(n => {
        const hl = highlights.includes(n.id);
        return (
          <g key={n.id}>
            <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={5}
              fill={hl ? "#4c1d95" : "#0f172a"} stroke={hl ? "#a78bfa" : "#334155"} strokeWidth={hl ? 2 : 1} />
            <text x={n.x + n.w / 2} y={n.y + n.h / 2 + 4} textAnchor="middle"
              fontSize={10} fill={hl ? "#c4b5fd" : "#64748b"} fontWeight={hl ? "bold" : "normal"}>
              {n.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ─── Main ────────────────────────────────────────────────────────────────────
export default function MySQLBTreeIndex() {
  const [scIdx, setScIdx] = useState(0);
  const [step, setStep] = useState(0);

  const sc = SCENARIOS[scIdx];
  const st = sc.steps[step];
  const hl = st.highlights;

  function gotoSc(i: number) { setScIdx(i); setStep(0); }

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 p-5 my-6">
      {/* Scenario tabs */}
      <div className="flex flex-wrap gap-2 mb-4">
        {SCENARIOS.map((s, i) => (
          <button key={i} onClick={() => gotoSc(i)}
            className={`px-3 py-1.5 rounded-lg text-sm font-semibold transition-colors
              ${scIdx === i ? "bg-violet-700 text-white" : "bg-slate-700 text-slate-300 hover:bg-slate-600"}`}>
            {s.label}
          </button>
        ))}
      </div>

      {/* Query */}
      <div className="rounded-lg bg-slate-800 border border-slate-700 px-4 py-2 mb-4 font-mono text-emerald-300 text-sm">
        {sc.query}
      </div>

      {/* Indexes SVG */}
      <div className="flex flex-wrap gap-6 mb-4 overflow-x-auto rounded-lg bg-slate-950 border border-slate-800 p-4">
        <ClusteredIndex highlights={hl.clustered ?? []} />
        <SecondaryIndex highlights={hl.secondary ?? []} />
      </div>

      {/* Data table */}
      <div className="overflow-x-auto mb-4">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="text-slate-400 text-xs border-b border-slate-700">
              <th className="py-1.5 px-3 text-left">id (PK)</th>
              <th className="py-1.5 px-3 text-left">name</th>
              <th className="py-1.5 px-3 text-left">email</th>
              <th className="py-1.5 px-3 text-left">age</th>
            </tr>
          </thead>
          <tbody>
            {TABLE_ROWS.map(row => {
              const active = hl.dataRows?.includes(row.id);
              return (
                <tr key={row.id} className={`border-b border-slate-800 transition-colors ${
                  active ? "bg-indigo-900/60" : "hover:bg-slate-800/40"}`}>
                  <td className={`py-1.5 px-3 font-mono font-bold ${active ? "text-indigo-300" : "text-slate-300"}`}>{row.id}</td>
                  <td className={`py-1.5 px-3 ${active ? "text-indigo-200" : "text-slate-300"}`}>{row.name}</td>
                  <td className={`py-1.5 px-3 ${active ? "text-indigo-200" : "text-slate-400"} text-xs`}>{row.email}</td>
                  <td className={`py-1.5 px-3 font-mono ${active ? "text-yellow-300 font-bold" : "text-slate-300"}`}>{row.age}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Step desc */}
      <div className="rounded-lg bg-violet-950 border border-violet-800 px-4 py-3 mb-4">
        <div className="text-violet-300 font-bold text-sm mb-1">
          步骤 {step + 1}/{sc.steps.length}：{st.title}
        </div>
        <div className="text-slate-200 text-sm whitespace-pre-wrap leading-relaxed">{st.desc}</div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        <button onClick={() => setStep(0)} disabled={step === 0}
          className="px-3 py-1.5 rounded bg-slate-700 text-slate-300 hover:bg-slate-600 disabled:opacity-30 text-lg">⏮</button>
        <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
          className="px-3 py-1.5 rounded bg-slate-700 text-slate-300 hover:bg-slate-600 disabled:opacity-30 text-lg">◀</button>
        <div className="flex-1 h-1.5 rounded-full bg-slate-700 mx-2">
          <div className="h-1.5 rounded-full bg-violet-500 transition-all duration-300"
            style={{ width: `${((step + 1) / sc.steps.length) * 100}%` }} />
        </div>
        <button onClick={() => setStep(s => Math.min(sc.steps.length - 1, s + 1))} disabled={step === sc.steps.length - 1}
          className="px-3 py-1.5 rounded bg-slate-700 text-slate-300 hover:bg-slate-600 disabled:opacity-30 text-lg">▶</button>
        <button onClick={() => setStep(sc.steps.length - 1)} disabled={step === sc.steps.length - 1}
          className="px-3 py-1.5 rounded bg-slate-700 text-slate-300 hover:bg-slate-600 disabled:opacity-30 text-lg">⏭</button>
      </div>

      {/* Key facts */}
      <div className="mt-5 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-slate-300">
        <div className="rounded bg-blue-950/60 border border-blue-900 px-3 py-2">
          <span className="text-blue-300 font-bold">聚簇索引（Clustered）：</span>
          叶子存完整行。主键查询 O(log n) I/O，无需回表。表数据物理上按主键顺序存储。
        </div>
        <div className="rounded bg-purple-950/60 border border-purple-900 px-3 py-2">
          <span className="text-purple-300 font-bold">辅助索引（Secondary）：</span>
          叶子存 (索引列, 主键id)。非覆盖查询需回表（Double Lookup），高选择性时效率仍优于全表扫描。
        </div>
      </div>
    </div>
  );
}
