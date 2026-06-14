"use client";
import React, { useState } from "react";

/**
 * 红黑树删除：双黑节点（Double-Black）四种修复情形可视化
 * 删除一个黑色节点后，替代节点(x)会获得"额外的黑色"变成双黑，违反性质。
 */

type DBCase = "db1" | "db2" | "db3" | "db4";

interface DBNode {
  id: string;
  key: number | string;
  color: "red" | "black" | "double-black" | "nil";
  x: number;
  y: number;
  label?: string;
}
interface DBEdge { from: string; to: string; }
interface DBFrame {
  title: string;
  desc: string;
  nodes: DBNode[];
  edges: DBEdge[];
  highlight: string[];
}

const DB_CASES: Record<DBCase, { label: string; trigger: string; fix: string; frames: DBFrame[] }> = {
  db1: {
    label: "Case 1：兄弟 w 为红色",
    trigger: "x 的兄弟 w 为红色（父 B 为黑，w 的两个子节点均为黑）。",
    fix: "交换 P 和 w 的颜色（P 变红，w 变黑），对 P 进行左旋。问题转化为 Case 2/3/4（兄弟变为黑色）。",
    frames: [
      {
        title: "初始：x 为双黑，兄弟 w 为红",
        desc: "x（双黑）是 P（黑）的左子，兄弟 w（红）。w 为红意味着 w 的两子均为黑（性质④）。",
        nodes: [
          { id: "P", key: 20, color: "black", x: 300, y: 50, label: "P（黑）" },
          { id: "x", key: "x", color: "double-black", x: 160, y: 140, label: "x（双黑）" },
          { id: "w", key: 40, color: "red", x: 440, y: 140, label: "w（红）" },
          { id: "WL", key: 35, color: "black", x: 360, y: 230, label: "wL（黑）" },
          { id: "WR", key: 50, color: "black", x: 520, y: 230, label: "wR（黑）" },
        ],
        edges: [{ from: "P", to: "x" }, { from: "P", to: "w" }, { from: "w", to: "WL" }, { from: "w", to: "WR" }],
        highlight: ["x", "w"],
      },
      {
        title: "对 P 左旋，交换 P 和 w 颜色",
        desc: "w（黑）成为新根，P（红）成为 w 的左子，w 的左子 WL 转给 P 作为右子。x 的兄弟现在变成了原来的 WL（黑）。",
        nodes: [
          { id: "w", key: 40, color: "black", x: 300, y: 50, label: "w（新根，黑）" },
          { id: "P", key: 20, color: "red", x: 160, y: 140, label: "P（变红）" },
          { id: "WR", key: 50, color: "black", x: 440, y: 140, label: "wR（黑）" },
          { id: "x", key: "x", color: "double-black", x: 80, y: 230, label: "x（双黑）" },
          { id: "WL", key: 35, color: "black", x: 250, y: 230, label: "WL（x 新兄弟，黑）→ Case 2/3/4" },
        ],
        edges: [{ from: "w", to: "P" }, { from: "w", to: "WR" }, { from: "P", to: "x" }, { from: "P", to: "WL" }],
        highlight: ["P", "WL"],
      },
    ],
  },
  db2: {
    label: "Case 2：兄弟 w 黑，w 两子均黑",
    trigger: "x 的兄弟 w 为黑，w 的两个子节点均为黑（或为 NIL）。",
    fix: "w 变红，从 P 和 x 各抽一层黑（x 变普通黑，P 吸收双黑或继续上传）。若 P 原为红则 P 变黑，结束；若 P 为黑则 P 变双黑，继续向上迭代。",
    frames: [
      {
        title: "初始：x 双黑，w 黑，w 两子均黑",
        desc: "x（双黑）是 P 的左子，兄弟 w（黑），w 的子节点 WL、WR 均为黑。不能直接旋转修复。",
        nodes: [
          { id: "P", key: 30, color: "red", x: 300, y: 50, label: "P（红）" },
          { id: "x", key: "x", color: "double-black", x: 160, y: 140, label: "x（双黑）" },
          { id: "w", key: 50, color: "black", x: 440, y: 140, label: "w（黑）" },
          { id: "WL", key: 45, color: "black", x: 360, y: 230, label: "WL（黑）" },
          { id: "WR", key: 60, color: "black", x: 520, y: 230, label: "WR（黑）" },
        ],
        edges: [{ from: "P", to: "x" }, { from: "P", to: "w" }, { from: "w", to: "WL" }, { from: "w", to: "WR" }],
        highlight: ["x", "w"],
      },
      {
        title: "w 变红，P 吸收双黑",
        desc: "w 变为红色（还给 x 一层黑，还给 w 一层黑），P 吸收双黑的那份额外黑。若 P 原为红 → P 变黑，修复完成。",
        nodes: [
          { id: "P", key: 30, color: "black", x: 300, y: 50, label: "P（红→黑，吸收双黑 ✓）" },
          { id: "x", key: "x", color: "black", x: 160, y: 140, label: "x（变普通黑 ✓）" },
          { id: "w", key: 50, color: "red", x: 440, y: 140, label: "w（变红 ✓）" },
          { id: "WL", key: 45, color: "black", x: 360, y: 230, label: "WL（黑）" },
          { id: "WR", key: 60, color: "black", x: 520, y: 230, label: "WR（黑）" },
        ],
        edges: [{ from: "P", to: "x" }, { from: "P", to: "w" }, { from: "w", to: "WL" }, { from: "w", to: "WR" }],
        highlight: ["P"],
      },
    ],
  },
  db3: {
    label: "Case 3：w 黑，w 近子红，远子黑",
    trigger: "x 兄弟 w 为黑，w 的近侧子节点（WL，靠近 x 侧）为红，远侧（WR）为黑。",
    fix: "交换 w 与 WL 颜色，对 w 右旋。问题转化为 Case 4（远侧子节点变为红色）。",
    frames: [
      {
        title: "初始：w 黑，WL（近侧）红，WR（远侧）黑",
        desc: "x（双黑）是 P 左子，兄弟 w（黑），w 的左子 WL（红，近侧），右子 WR（黑，远侧）。",
        nodes: [
          { id: "P", key: 40, color: "black", x: 300, y: 50, label: "P（任意色）" },
          { id: "x", key: "x", color: "double-black", x: 160, y: 140, label: "x（双黑）" },
          { id: "w", key: 60, color: "black", x: 440, y: 140, label: "w（黑）" },
          { id: "WL", key: 50, color: "red", x: 360, y: 230, label: "WL（红，近侧）" },
          { id: "WR", key: 75, color: "black", x: 530, y: 230, label: "WR（黑，远侧）" },
        ],
        edges: [{ from: "P", to: "x" }, { from: "P", to: "w" }, { from: "w", to: "WL" }, { from: "w", to: "WR" }],
        highlight: ["WL", "w"],
      },
      {
        title: "WL 变黑，w 变红，对 w 右旋 → Case 4",
        desc: "交换颜色：WL 变黑，w 变红。对 w 右旋：WL 上升，w 下降为 WL 的右子。现在 w 的新远侧子节点（原 w，现为红）→ Case 4。",
        nodes: [
          { id: "P", key: 40, color: "black", x: 300, y: 50, label: "P" },
          { id: "x", key: "x", color: "double-black", x: 160, y: 140, label: "x（双黑）" },
          { id: "WL", key: 50, color: "black", x: 440, y: 140, label: "WL（新兄弟，黑）" },
          { id: "w", key: 60, color: "red", x: 530, y: 230, label: "w（红，远侧）→ Case 4" },
          { id: "WR", key: 75, color: "black", x: 620, y: 320, label: "WR（黑）" },
        ],
        edges: [{ from: "P", to: "x" }, { from: "P", to: "WL" }, { from: "WL", to: "w" }, { from: "w", to: "WR" }],
        highlight: ["WL", "w"],
      },
    ],
  },
  db4: {
    label: "Case 4：w 黑，w 远子红",
    trigger: "x 兄弟 w 为黑，w 的远侧子节点（WR，远离 x 侧）为红（近侧颜色任意）。",
    fix: "w 继承 P 颜色，P 变黑，WR 变黑，对 P 左旋。一步修复，双黑消除。",
    frames: [
      {
        title: "初始：w 黑，WR（远侧）红",
        desc: "x（双黑）是 P 左子，兄弟 w（黑），w 远侧子节点 WR（红）。这是最终情形，一次旋转即可解决。",
        nodes: [
          { id: "P", key: 40, color: "red", x: 300, y: 50, label: "P（任意色，设为红）" },
          { id: "x", key: "x", color: "double-black", x: 160, y: 140, label: "x（双黑）" },
          { id: "w", key: 60, color: "black", x: 440, y: 140, label: "w（黑）" },
          { id: "WL", key: "·", color: "nil", x: 370, y: 230, label: "WL（任意）" },
          { id: "WR", key: 75, color: "red", x: 520, y: 230, label: "WR（红，远侧）" },
        ],
        edges: [{ from: "P", to: "x" }, { from: "P", to: "w" }, { from: "w", to: "WL" }, { from: "w", to: "WR" }],
        highlight: ["w", "WR"],
      },
      {
        title: "染色：w ← P 色，P ← 黑，WR ← 黑",
        desc: "准备旋转：w 继承 P 的颜色（此例为红→红→最终变红），P 变黑，WR 变黑。",
        nodes: [
          { id: "P", key: 40, color: "black", x: 300, y: 50, label: "P（变黑）" },
          { id: "x", key: "x", color: "double-black", x: 160, y: 140, label: "x（双黑）" },
          { id: "w", key: 60, color: "red", x: 440, y: 140, label: "w（继承 P 色）" },
          { id: "WL", key: "·", color: "nil", x: 370, y: 230, label: "WL" },
          { id: "WR", key: 75, color: "black", x: 520, y: 230, label: "WR（变黑）" },
        ],
        edges: [{ from: "P", to: "x" }, { from: "P", to: "w" }, { from: "w", to: "WL" }, { from: "w", to: "WR" }],
        highlight: ["P", "w", "WR"],
      },
      {
        title: "对 P 左旋，双黑完全消除 ✓",
        desc: "w 上升取代 P，P 下降为 w 的左子，x 变为普通黑节点（抽掉额外黑）。黑高在所有路径上一致，五条性质全部满足！",
        nodes: [
          { id: "w", key: 60, color: "black", x: 300, y: 50, label: "w（新根，继承 P 色）" },
          { id: "P", key: 40, color: "black", x: 160, y: 140, label: "P（黑 ✓）" },
          { id: "WR", key: 75, color: "black", x: 440, y: 140, label: "WR（黑 ✓）" },
          { id: "x", key: "x", color: "black", x: 80, y: 230, label: "x（变普通黑 ✓）" },
          { id: "WL", key: "·", color: "nil", x: 260, y: 230, label: "WL（原 w 左子）" },
        ],
        edges: [{ from: "w", to: "P" }, { from: "w", to: "WR" }, { from: "P", to: "x" }, { from: "P", to: "WL" }],
        highlight: [],
      },
    ],
  },
};

// SVG renderer
function DBNodeEl({ node, isHL }: { node: DBNode; isHL: boolean }) {
  if (node.color === "nil") {
    return (
      <g>
        <rect x={node.x - 12} y={node.y - 9} width={24} height={18} rx={3} fill="#e2e8f0" stroke="#94a3b8" strokeWidth={1} />
        <text x={node.x} y={node.y + 1} textAnchor="middle" dominantBaseline="middle" fill="#94a3b8" fontSize={7}>NIL</text>
      </g>
    );
  }
  const fill = node.color === "red" ? "#ef4444" : node.color === "double-black" ? "#1e293b" : "#334155";
  const ring = node.color === "double-black";
  return (
    <g>
      {ring && <circle cx={node.x} cy={node.y} r={27} fill="none" stroke="#7c3aed" strokeWidth={2.5} strokeDasharray="5 3" />}
      <circle cx={node.x} cy={node.y} r={22} fill={fill} stroke={isHL ? "#f97316" : "#475569"} strokeWidth={isHL ? 2.5 : 1.5} />
      <text x={node.x} y={node.y + 1} textAnchor="middle" dominantBaseline="middle" fill="white" fontSize={11} fontWeight="bold">{node.key}</text>
      {node.label && (
        <text x={node.x} y={node.y + 40} textAnchor="middle" fill="#64748b" fontSize={7.5} style={{ maxWidth: 80 }}>{node.label}</text>
      )}
    </g>
  );
}

export default function RBDeleteDoubleBlack() {
  const [cas, setCas] = useState<DBCase>("db1");
  const [frameIdx, setFrameIdx] = useState(0);

  const preset = DB_CASES[cas];
  const frame = preset.frames[frameIdx];
  const total = preset.frames.length;
  const handleCase = (c: DBCase) => { setCas(c); setFrameIdx(0); };

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5 my-6 shadow-sm">
      <h3 className="text-base font-bold text-slate-800 dark:text-slate-100 mb-1">⚫⚫ 红黑树删除：双黑节点修复</h3>
      <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
        删除黑节点后，替代节点变为"双黑"（紫色虚线圈标注），需通过 4 种情形修复。以下均以 x 在父节点左侧为例（右侧对称）。
      </p>

      <div className="flex flex-wrap gap-2 mb-4">
        {(Object.keys(DB_CASES) as DBCase[]).map((c) => (
          <button key={c} onClick={() => handleCase(c)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${cas === c ? "bg-violet-600 text-white" : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-violet-50"}`}>
            {DB_CASES[c].label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-4 text-xs">
        <div className="rounded-lg bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 px-3 py-2 text-orange-700 dark:text-orange-300">
          <span className="font-semibold">触发条件：</span>{preset.trigger}
        </div>
        <div className="rounded-lg bg-teal-50 dark:bg-teal-900/20 border border-teal-200 dark:border-teal-800 px-3 py-2 text-teal-700 dark:text-teal-300">
          <span className="font-semibold">修复操作：</span>{preset.fix}
        </div>
      </div>

      <div className="bg-slate-50 dark:bg-slate-800/60 rounded-xl p-3 mb-4">
        <svg viewBox="0 0 700 300" className="w-full" style={{ maxHeight: 280 }}>
          {frame.edges.map((e) => {
            const from = frame.nodes.find((n) => n.id === e.from);
            const to = frame.nodes.find((n) => n.id === e.to);
            if (!from || !to) return null;
            return <line key={`${e.from}-${e.to}`} x1={from.x} y1={from.y + 22} x2={to.x} y2={to.y - (to.color === "nil" ? 9 : 22)} stroke="#94a3b8" strokeWidth={1.5} />;
          })}
          {frame.nodes.map((n) => <DBNodeEl key={n.id} node={n} isHL={frame.highlight.includes(n.id)} />)}
        </svg>
      </div>

      <div className="mb-4 rounded-lg bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-800 px-4 py-2">
        <div className="text-xs font-semibold text-violet-700 dark:text-violet-300 mb-0.5">步骤 {frameIdx + 1}/{total}：{frame.title}</div>
        <div className="text-xs text-violet-600 dark:text-violet-400">{frame.desc}</div>
      </div>

      <div className="flex items-center gap-3">
        <button onClick={() => setFrameIdx(0)} disabled={frameIdx === 0} className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-xs disabled:opacity-40">⏮</button>
        <button onClick={() => setFrameIdx((f) => Math.max(0, f - 1))} disabled={frameIdx === 0} className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-xs disabled:opacity-40">◀ 上一步</button>
        <div className="flex-1 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
          <div className="h-full bg-violet-500 rounded-full transition-all" style={{ width: `${((frameIdx + 1) / total) * 100}%` }} />
        </div>
        <button onClick={() => setFrameIdx((f) => Math.min(total - 1, f + 1))} disabled={frameIdx === total - 1} className="px-3 py-1.5 rounded-lg bg-violet-600 text-white text-xs disabled:opacity-40">下一步 ▶</button>
        <button onClick={() => setFrameIdx(total - 1)} disabled={frameIdx === total - 1} className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-xs disabled:opacity-40">⏭</button>
      </div>

      <div className="flex flex-wrap gap-4 mt-4 text-xs text-slate-500 dark:text-slate-400">
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-slate-700 inline-block border-2 border-dashed border-violet-500" />双黑节点（Double-Black）</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-red-500 inline-block" />红色节点</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-slate-700 inline-block" />黑色节点</span>
      </div>
    </div>
  );
}
