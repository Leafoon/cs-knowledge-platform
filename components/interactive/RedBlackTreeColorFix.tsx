"use client";
import React, { useState } from "react";

/** 红黑树插入颜色修复：Case 1 / Case 2 / Case 3 三种情形步进动画 */

type CaseType = "case1" | "case2" | "case3";

interface RBNode {
  id: string;
  key: number | string;
  color: "red" | "black" | "nil";
  x: number;
  y: number;
  role?: string; // label below node
}
interface Edge { from: string; to: string; }
interface Frame {
  title: string;
  desc: string;
  nodes: RBNode[];
  edges: Edge[];
  highlight: string[];
}

// ─── Case definitions ─────────────────────────────────────────────────────────
const CASES: Record<CaseType, { label: string; summary: string; trigger: string; fix: string; frames: Frame[] }> = {
  case1: {
    label: "Case 1：叔叔节点为红色",
    summary: "父节点 P 和叔叔节点 U 都是红色 → 将 P、U 变黑，祖父 G 变红，问题上移。",
    trigger: `新节点 z 为红，父 P 为红（违反"红不连红"），叔叔 U 也为红。`,
    fix: "P 黑，U 黑，G 红。若 G 是根则再变黑。问题可能向上传播。",
    frames: [
      {
        title: "插入 z（红）后，父 P 也为红 → 违反性质 4",
        desc: `新节点 z（红）的父节点 P（红）违反"红色节点子节点必须是黑色"。观察叔叔 U 为红色 → Case 1。`,
        nodes: [
          { id: "G", key: 20, color: "black", x: 300, y: 50, role: "祖父 G" },
          { id: "P", key: 10, color: "red",   x: 170, y: 140, role: "父 P（红）" },
          { id: "U", key: 30, color: "red",   x: 430, y: 140, role: "叔叔 U（红）" },
          { id: "z", key: 5,  color: "red",   x: 90,  y: 230, role: "新节点 z（红）" },
          { id: "PR", key: "·", color: "nil", x: 250, y: 230, role: "" },
          { id: "UL", key: "·", color: "nil", x: 350, y: 230, role: "" },
          { id: "UR", key: "·", color: "nil", x: 510, y: 230, role: "" },
        ],
        edges: [
          { from: "G", to: "P" }, { from: "G", to: "U" },
          { from: "P", to: "z" }, { from: "P", to: "PR" },
          { from: "U", to: "UL" }, { from: "U", to: "UR" },
        ],
        highlight: ["z", "P"],
      },
      {
        title: "修复：P 变黑，U 变黑，G 变红",
        desc: "颜色重涂：P→黑，U→黑，G→红。黑高不变（G 子树每条路黑节点数一致）。但 G 变红可能与 G 的父节点冲突。",
        nodes: [
          { id: "G", key: 20, color: "red",   x: 300, y: 50, role: "G（变红，问题上移）" },
          { id: "P", key: 10, color: "black", x: 170, y: 140, role: "P（变黑 ✓）" },
          { id: "U", key: 30, color: "black", x: 430, y: 140, role: "U（变黑 ✓）" },
          { id: "z", key: 5,  color: "red",   x: 90,  y: 230, role: "z（保持红）" },
          { id: "PR", key: "·", color: "nil", x: 250, y: 230, role: "" },
          { id: "UL", key: "·", color: "nil", x: 350, y: 230, role: "" },
          { id: "UR", key: "·", color: "nil", x: 510, y: 230, role: "" },
        ],
        edges: [
          { from: "G", to: "P" }, { from: "G", to: "U" },
          { from: "P", to: "z" }, { from: "P", to: "PR" },
          { from: "U", to: "UL" }, { from: "U", to: "UR" },
        ],
        highlight: ["G", "P", "U"],
      },
      {
        title: "Case 1 完成：z 指针上移到 G，继续修复",
        desc: "将 z 指向 G，继续判断 G 的父节点是否违规（Case 1/2/3 向上迭代）。若 G 已是根则变黑，终止。",
        nodes: [
          { id: "z2", key: 20, color: "red",   x: 300, y: 50, role: "z ← G（红，继续向上）" },
          { id: "P", key: 10,  color: "black", x: 170, y: 140, role: "P（黑 ✓）" },
          { id: "U", key: 30,  color: "black", x: 430, y: 140, role: "U（黑 ✓）" },
          { id: "n5", key: 5,  color: "red",   x: 90,  y: 230, role: "" },
          { id: "PR", key: "·", color: "nil", x: 250, y: 230, role: "" },
          { id: "UL", key: "·", color: "nil", x: 350, y: 230, role: "" },
          { id: "UR", key: "·", color: "nil", x: 510, y: 230, role: "" },
        ],
        edges: [
          { from: "z2", to: "P" }, { from: "z2", to: "U" },
          { from: "P", to: "n5" }, { from: "P", to: "PR" },
          { from: "U", to: "UL" }, { from: "U", to: "UR" },
        ],
        highlight: ["z2"],
      },
    ],
  },
  case2: {
    label: "Case 2：叔叔黑，z 是内侧节点",
    summary: "叔叔 U 为黑，z 在父 P 的内侧（P 是左子，z 是右子；或反之）。旋转 P，转化为 Case 3。",
    trigger: "z 在 P 的右侧（P 是 G 的左子），形成左-右折线形。",
    fix: "对 P 进行左旋，z 与 P 角色互换，转化为 Case 3（z 在外侧）。",
    frames: [
      {
        title: "初始：z 是 P 的右子（内侧），叔叔 U 为黑",
        desc: `z（红）是 P（红）的右子，形成"折线"，叔叔 U 为黑（NIL 或黑节点）。单次旋转不能直接修复。`,
        nodes: [
          { id: "G", key: 30, color: "black", x: 320, y: 50, role: "祖父 G" },
          { id: "P", key: 15, color: "red",   x: 170, y: 140, role: "父 P（红）" },
          { id: "U", key: 45, color: "black", x: 460, y: 140, role: "叔叔 U（黑）" },
          { id: "PL", key: "·", color: "nil", x: 90,  y: 230, role: "" },
          { id: "z",  key: 22, color: "red",  x: 260, y: 230, role: "z（红，内侧）" },
        ],
        edges: [
          { from: "G", to: "P" }, { from: "G", to: "U" },
          { from: "P", to: "PL" }, { from: "P", to: "z" },
        ],
        highlight: ["z", "P"],
      },
      {
        title: "对 P 进行左旋，z 上升，P 下降",
        desc: "左旋 P：z 取代 P 的位置，P 成为 z 的左子。现在 P（红色）变成 z 的左子，是外侧 → Case 3。",
        nodes: [
          { id: "G", key: 30, color: "black", x: 320, y: 50, role: "祖父 G" },
          { id: "z", key: 22, color: "red",   x: 170, y: 140, role: "z（红，现为外侧）" },
          { id: "U", key: 45, color: "black", x: 460, y: 140, role: "叔叔 U（黑）" },
          { id: "P", key: 15, color: "red",   x: 90,  y: 230, role: "P（红，z 左子）→ Case 3" },
          { id: "ZR", key: "·", color: "nil", x: 260, y: 230, role: "" },
        ],
        edges: [
          { from: "G", to: "z" }, { from: "G", to: "U" },
          { from: "z", to: "P" }, { from: "z", to: "ZR" },
        ],
        highlight: ["z", "P"],
      },
      {
        title: "Case 2 → Case 3，继续用 Case 3 修复",
        desc: "现在 z（红）在 P（原z）的左侧（外侧），叔叔黑。满足 Case 3 条件。把此时的 z 理解为 Case 3 中的被修复节点继续处理。",
        nodes: [
          { id: "G", key: 30, color: "black", x: 320, y: 50, role: "G（下一步用 Case 3）" },
          { id: "z", key: 22, color: "red",   x: 170, y: 140, role: "旧z（现为新的父 P'）" },
          { id: "U", key: 45, color: "black", x: 460, y: 140, role: "U（黑）" },
          { id: "P", key: 15, color: "red",   x: 90,  y: 230, role: "旧P（现为新z）" },
          { id: "ZR", key: "·", color: "nil", x: 260, y: 230, role: "" },
        ],
        edges: [
          { from: "G", to: "z" }, { from: "G", to: "U" },
          { from: "z", to: "P" }, { from: "z", to: "ZR" },
        ],
        highlight: ["G", "z", "P"],
      },
    ],
  },
  case3: {
    label: "Case 3：叔叔黑，z 是外侧节点",
    summary: "叔叔 U 为黑，z 在父 P 的外侧（P 是左子，z 也是左子）。令 P 变黑、G 变红，再对 G 右旋，一步修复。",
    trigger: "z（红）是 P（红）的左子，P 是 G 的左子，叔叔 U 为黑。",
    fix: "P 变黑，G 变红，对 G 进行右旋。P 成为新根，颜色合法，黑高不变。",
    frames: [
      {
        title: "初始：z 是 P 的左子（外侧），叔叔 U 为黑",
        desc: "z（红）在 P（红）的外侧（左-左），叔叔黑。直线形 → 单次右旋可修复。",
        nodes: [
          { id: "G", key: 40, color: "black", x: 320, y: 50, role: "祖父 G（黑）" },
          { id: "P", key: 20, color: "red",   x: 170, y: 140, role: "父 P（红）" },
          { id: "U", key: 55, color: "black", x: 470, y: 140, role: "叔叔 U（黑）" },
          { id: "z", key: 10, color: "red",   x: 90,  y: 230, role: "z（红，外侧）" },
          { id: "PR", key: "·", color: "nil", x: 260, y: 230, role: "" },
        ],
        edges: [
          { from: "G", to: "P" }, { from: "G", to: "U" },
          { from: "P", to: "z" }, { from: "P", to: "PR" },
        ],
        highlight: ["z", "P", "G"],
      },
      {
        title: "重新染色：P 变黑，G 变红",
        desc: "P 变为黑色（吸收 G 的黑色），G 变为红色（为旋转做准备）。此时黑高尚未满足，旋转后才平衡。",
        nodes: [
          { id: "G", key: 40, color: "red",   x: 320, y: 50, role: "G（变红）" },
          { id: "P", key: 20, color: "black", x: 170, y: 140, role: "P（变黑）" },
          { id: "U", key: 55, color: "black", x: 470, y: 140, role: "U（黑）" },
          { id: "z", key: 10, color: "red",   x: 90,  y: 230, role: "z（红）" },
          { id: "PR", key: "·", color: "nil", x: 260, y: 230, role: "" },
        ],
        edges: [
          { from: "G", to: "P" }, { from: "G", to: "U" },
          { from: "P", to: "z" }, { from: "P", to: "PR" },
        ],
        highlight: ["P", "G"],
      },
      {
        title: "对 G 右旋，P 取代 G 成为新根",
        desc: "右旋 G：P 上升取代 G，G 下降成为 P 的右子（继承 P 的原右子 T₂）。P（黑）-G（红）-U（黑），颜色全部合法！",
        nodes: [
          { id: "P", key: 20, color: "black", x: 300, y: 50, role: "P（新根，黑 ✓）" },
          { id: "z", key: 10, color: "red",   x: 160, y: 140, role: "z（红）" },
          { id: "G", key: 40, color: "red",   x: 440, y: 140, role: "G（红 ✓）" },
          { id: "PR", key: "·", color: "nil", x: 350, y: 230, role: "原T₂" },
          { id: "U", key: 55, color: "black", x: 530, y: 230, role: "U（黑）" },
        ],
        edges: [
          { from: "P", to: "z" }, { from: "P", to: "G" },
          { from: "G", to: "PR" }, { from: "G", to: "U" },
        ],
        highlight: [],
      },
    ],
  },
};

// ─── Component ────────────────────────────────────────────────────────────────
function RBNodeSVG({ node, isHL }: { node: RBNode; isHL: boolean }) {
  const fill = node.color === "nil" ? "transparent"
    : node.color === "red" ? "#ef4444" : "#1e293b";
  const stroke = isHL ? "#f97316" : node.color === "nil" ? "#94a3b8" : "#475569";
  if (node.color === "nil") {
    return (
      <g>
        <rect x={node.x - 10} y={node.y - 8} width={20} height={16} rx={3} fill="#e2e8f0" stroke={stroke} strokeWidth={1} />
        <text x={node.x} y={node.y + 1} textAnchor="middle" dominantBaseline="middle" fill="#94a3b8" fontSize={8}>NIL</text>
      </g>
    );
  }
  return (
    <g>
      <circle cx={node.x} cy={node.y} r={22} fill={fill} stroke={isHL ? "#f97316" : "#64748b"} strokeWidth={isHL ? 2.5 : 1.5} />
      <text x={node.x} y={node.y + 1} textAnchor="middle" dominantBaseline="middle" fill="white" fontSize={12} fontWeight="bold">{node.key}</text>
      {node.role && (
        <text x={node.x} y={node.y + 36} textAnchor="middle" fill="#64748b" fontSize={8}>{node.role}</text>
      )}
    </g>
  );
}

function RBTreeSVG({ frame }: { frame: Frame }) {
  return (
    <svg viewBox="0 0 600 300" className="w-full" style={{ maxHeight: 280 }}>
      {frame.edges.map((e) => {
        const from = frame.nodes.find((n) => n.id === e.from);
        const to = frame.nodes.find((n) => n.id === e.to);
        if (!from || !to) return null;
        const dy = to.color === "nil" ? -8 : -22;
        return <line key={`${e.from}-${e.to}`} x1={from.x} y1={from.y + 22} x2={to.x} y2={to.y + dy} stroke="#94a3b8" strokeWidth={1.5} strokeDasharray={to.color === "nil" ? "4 3" : undefined} />;
      })}
      {frame.nodes.map((n) => (
        <RBNodeSVG key={n.id} node={n} isHL={frame.highlight.includes(n.id)} />
      ))}
    </svg>
  );
}

export default function RedBlackTreeColorFix() {
  const [caseType, setCaseType] = useState<CaseType>("case1");
  const [frameIdx, setFrameIdx] = useState(0);

  const preset = CASES[caseType];
  const frame = preset.frames[frameIdx];
  const total = preset.frames.length;

  const handleCase = (c: CaseType) => { setCaseType(c); setFrameIdx(0); };

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5 my-6 shadow-sm">
      <h3 className="text-base font-bold text-slate-800 dark:text-slate-100 mb-1">🔴⚫ 红黑树插入颜色修复</h3>
      <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">逐步演示插入后三种颜色违规情形的修复。黑色节点 = 黑，红色 = 红，NIL = 黑色哨兵节点（外部节点）。</p>

      {/* Case selector */}
      <div className="flex flex-wrap gap-2 mb-4">
        {(Object.keys(CASES) as CaseType[]).map((c) => (
          <button key={c} onClick={() => handleCase(c)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${caseType === c ? "bg-rose-600 text-white shadow" : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-rose-50 dark:hover:bg-slate-700"}`}>
            {CASES[c].label}
          </button>
        ))}
      </div>

      {/* Summary boxes */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-4">
        <div className="rounded-lg bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 px-3 py-2 text-xs text-yellow-700 dark:text-yellow-300">
          <span className="font-semibold">触发条件：</span>{preset.trigger}
        </div>
        <div className="rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 px-3 py-2 text-xs text-green-700 dark:text-green-300">
          <span className="font-semibold">修复操作：</span>{preset.fix}
        </div>
      </div>

      {/* Tree SVG */}
      <div className="bg-slate-50 dark:bg-slate-800/60 rounded-xl p-3 mb-4">
        <RBTreeSVG frame={frame} />
      </div>

      {/* Step info */}
      <div className="mb-4 rounded-lg bg-rose-50 dark:bg-rose-900/20 border border-rose-200 dark:border-rose-800 px-4 py-2">
        <div className="text-xs font-semibold text-rose-700 dark:text-rose-300 mb-0.5">步骤 {frameIdx + 1}/{total}：{frame.title}</div>
        <div className="text-xs text-rose-600 dark:text-rose-400">{frame.desc}</div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        <button onClick={() => setFrameIdx(0)} disabled={frameIdx === 0} className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 text-xs disabled:opacity-40 hover:bg-slate-200">⏮</button>
        <button onClick={() => setFrameIdx((f) => Math.max(0, f - 1))} disabled={frameIdx === 0} className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 text-xs disabled:opacity-40 hover:bg-slate-200">◀ 上一步</button>
        <div className="flex-1 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
          <div className="h-full bg-rose-500 rounded-full transition-all" style={{ width: `${((frameIdx + 1) / total) * 100}%` }} />
        </div>
        <button onClick={() => setFrameIdx((f) => Math.min(total - 1, f + 1))} disabled={frameIdx === total - 1} className="px-3 py-1.5 rounded-lg bg-rose-600 text-white text-xs disabled:opacity-40 hover:bg-rose-700">下一步 ▶</button>
        <button onClick={() => setFrameIdx(total - 1)} disabled={frameIdx === total - 1} className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 text-xs disabled:opacity-40 hover:bg-slate-200">⏭</button>
      </div>

      {/* 5 Properties reminder */}
      <div className="mt-5 border-t border-slate-100 dark:border-slate-700 pt-4 grid grid-cols-1 gap-1 text-xs text-slate-500 dark:text-slate-400">
        <div className="font-semibold text-slate-600 dark:text-slate-300 mb-1">红黑树五条性质：</div>
        {["① 每个节点非红即黑", "② 根节点为黑色", "③ 所有叶节点（NIL）为黑色", "④ 红色节点的两个子节点均为黑色（不连红）", "⑤ 从任一节点到其叶节点的所有路径上，黑色节点数（黑高）相同"].map((p, i) => (
          <div key={i} className="flex items-start gap-1.5"><span className="text-rose-500 font-bold">{p.slice(0, 1)}</span><span>{p.slice(1)}</span></div>
        ))}
      </div>
    </div>
  );
}
