"use client";
import React, { useState } from "react";

/** AVL 树四种旋转动画组件：LL(右旋) / RR(左旋) / LR(先左后右) / RL(先右后左) */

type RotationType = "LL" | "RR" | "LR" | "RL";

interface NodePos {
  id: string;
  key: number;
  bf: number; // balance factor
  x: number;
  y: number;
  color: string;
}

interface EdgeDef {
  from: string;
  to: string;
}

interface Frame {
  title: string;
  desc: string;
  nodes: NodePos[];
  edges: EdgeDef[];
  highlight: string[]; // node ids highlighted orange
}

// ─── Rotation presets ─────────────────────────────────────────────────────────
const PRESETS: Record<RotationType, { label: string; description: string; frames: Frame[] }> = {
  LL: {
    label: "LL 右旋（Right Rotate）",
    description:
      "失衡节点 z 的左子 y 的左侧插入了节点 x，导致 z 的平衡因子 = +2。执行右旋：y 取代 z，z 成为 y 的右子。",
    frames: [
      {
        title: "初始状态（插入 x 后 z 失衡）",
        desc: "节点 z 平衡因子 bf=+2（左重），y 平衡因子 bf=+1，不平衡。需要右旋。",
        nodes: [
          { id: "z", key: 50, bf: 2, x: 300, y: 60, color: "#ef4444" },
          { id: "y", key: 30, bf: 1, x: 170, y: 150, color: "#f59e0b" },
          { id: "x", key: 15, bf: 0, x: 80, y: 240, color: "#3b82f6" },
          { id: "t2", key: 40, bf: 0, x: 260, y: 240, color: "#6b7280" },
          { id: "t3", key: 60, bf: 0, x: 400, y: 150, color: "#6b7280" },
        ],
        edges: [
          { from: "z", to: "y" }, { from: "z", to: "t3" },
          { from: "y", to: "x" }, { from: "y", to: "t2" },
        ],
        highlight: ["z"],
      },
      {
        title: "第 1 步：y 上升取代 z",
        desc: "y 成为新的根，z 下移到 y 的右侧。y 原来的右子 T₂ 将转移给 z。",
        nodes: [
          { id: "y", key: 30, bf: 0, x: 300, y: 60, color: "#10b981" },
          { id: "x", key: 15, bf: 0, x: 170, y: 150, color: "#3b82f6" },
          { id: "z", key: 50, bf: 0, x: 420, y: 150, color: "#f59e0b" },
          { id: "t2", key: 40, bf: 0, x: 330, y: 240, color: "#6b7280" },
          { id: "t3", key: 60, bf: 0, x: 510, y: 240, color: "#6b7280" },
        ],
        edges: [
          { from: "y", to: "x" }, { from: "y", to: "z" },
          { from: "z", to: "t2" }, { from: "z", to: "t3" },
        ],
        highlight: ["y", "z"],
      },
      {
        title: "完成！右旋后平衡恢复",
        desc: "y 为新根（bf=0），x 在左（bf=0），z 在右（bf=0）。BST 性质：x < y < T₂ < z < T₃ 完全保持。",
        nodes: [
          { id: "y", key: 30, bf: 0, x: 300, y: 60, color: "#10b981" },
          { id: "x", key: 15, bf: 0, x: 170, y: 150, color: "#10b981" },
          { id: "z", key: 50, bf: 0, x: 420, y: 150, color: "#10b981" },
          { id: "t2", key: 40, bf: 0, x: 330, y: 240, color: "#6b7280" },
          { id: "t3", key: 60, bf: 0, x: 510, y: 240, color: "#6b7280" },
        ],
        edges: [
          { from: "y", to: "x" }, { from: "y", to: "z" },
          { from: "z", to: "t2" }, { from: "z", to: "t3" },
        ],
        highlight: [],
      },
    ],
  },
  RR: {
    label: "RR 左旋（Left Rotate）",
    description:
      "失衡节点 z 的右子 y 的右侧插入了节点 x，导致 z 的平衡因子 = -2。执行左旋：y 取代 z，z 成为 y 的左子。",
    frames: [
      {
        title: "初始状态（插入 x 后 z 失衡）",
        desc: "节点 z 平衡因子 bf=-2（右重），y 平衡因子 bf=-1，不平衡。需要左旋。",
        nodes: [
          { id: "z", key: 30, bf: -2, x: 200, y: 60, color: "#ef4444" },
          { id: "t0", key: 15, bf: 0, x: 80, y: 150, color: "#6b7280" },
          { id: "y", key: 50, bf: -1, x: 330, y: 150, color: "#f59e0b" },
          { id: "t2", key: 40, bf: 0, x: 250, y: 240, color: "#6b7280" },
          { id: "x", key: 70, bf: 0, x: 430, y: 240, color: "#3b82f6" },
        ],
        edges: [
          { from: "z", to: "t0" }, { from: "z", to: "y" },
          { from: "y", to: "t2" }, { from: "y", to: "x" },
        ],
        highlight: ["z"],
      },
      {
        title: "第 1 步：y 上升取代 z",
        desc: "y 成为新的根，z 下移到 y 的左侧。y 原来的左子 T₂ 转移给 z 作为右子。",
        nodes: [
          { id: "y", key: 50, bf: 0, x: 300, y: 60, color: "#10b981" },
          { id: "z", key: 30, bf: 0, x: 170, y: 150, color: "#f59e0b" },
          { id: "x", key: 70, bf: 0, x: 430, y: 150, color: "#3b82f6" },
          { id: "t0", key: 15, bf: 0, x: 80, y: 240, color: "#6b7280" },
          { id: "t2", key: 40, bf: 0, x: 260, y: 240, color: "#6b7280" },
        ],
        edges: [
          { from: "y", to: "z" }, { from: "y", to: "x" },
          { from: "z", to: "t0" }, { from: "z", to: "t2" },
        ],
        highlight: ["y", "z"],
      },
      {
        title: "完成！左旋后平衡恢复",
        desc: "y 为新根（bf=0），z 在左（bf=0），x 在右（bf=0）。BST 性质完全保持。",
        nodes: [
          { id: "y", key: 50, bf: 0, x: 300, y: 60, color: "#10b981" },
          { id: "z", key: 30, bf: 0, x: 170, y: 150, color: "#10b981" },
          { id: "x", key: 70, bf: 0, x: 430, y: 150, color: "#10b981" },
          { id: "t0", key: 15, bf: 0, x: 80, y: 240, color: "#6b7280" },
          { id: "t2", key: 40, bf: 0, x: 260, y: 240, color: "#6b7280" },
        ],
        edges: [
          { from: "y", to: "z" }, { from: "y", to: "x" },
          { from: "z", to: "t0" }, { from: "z", to: "t2" },
        ],
        highlight: [],
      },
    ],
  },
  LR: {
    label: "LR 双旋（先左旋 y，再右旋 z）",
    description:
      `失衡节点 z 的左子 y 的右侧插入节点 x，形成"左-右"折线形，需两步：先对 y 左旋，再对 z 右旋。`,
    frames: [
      {
        title: "初始状态（插入 x 后 z 失衡）",
        desc: "z bf=+2（左重），y bf=-1（右重），形成 LR 折线，单次右旋无法修复。",
        nodes: [
          { id: "z", key: 50, bf: 2, x: 300, y: 60, color: "#ef4444" },
          { id: "y", key: 20, bf: -1, x: 160, y: 150, color: "#f59e0b" },
          { id: "t3", key: 65, bf: 0, x: 430, y: 150, color: "#6b7280" },
          { id: "t1", key: 10, bf: 0, x: 80, y: 240, color: "#6b7280" },
          { id: "x", key: 35, bf: 0, x: 250, y: 240, color: "#3b82f6" },
        ],
        edges: [
          { from: "z", to: "y" }, { from: "z", to: "t3" },
          { from: "y", to: "t1" }, { from: "y", to: "x" },
        ],
        highlight: ["z", "y"],
      },
      {
        title: "第 1 步：对 y 进行左旋",
        desc: "x 上升取代 y，y 成为 x 的左子。LR 折线 → LL 直线，准备右旋。",
        nodes: [
          { id: "z", key: 50, bf: 2, x: 300, y: 60, color: "#ef4444" },
          { id: "x", key: 35, bf: 1, x: 160, y: 150, color: "#8b5cf6" },
          { id: "t3", key: 65, bf: 0, x: 430, y: 150, color: "#6b7280" },
          { id: "y", key: 20, bf: 0, x: 80, y: 240, color: "#f59e0b" },
          { id: "t1", key: 10, bf: 0, x: 30, y: 330, color: "#6b7280" },
        ],
        edges: [
          { from: "z", to: "x" }, { from: "z", to: "t3" },
          { from: "x", to: "y" },
          { from: "y", to: "t1" },
        ],
        highlight: ["x", "z"],
      },
      {
        title: "第 2 步：对 z 进行右旋",
        desc: "x 上升取代 z，z 成为 x 的右子，y 保持为 x 的左子。双旋完成！",
        nodes: [
          { id: "x", key: 35, bf: 0, x: 300, y: 60, color: "#10b981" },
          { id: "y", key: 20, bf: 0, x: 170, y: 150, color: "#10b981" },
          { id: "z", key: 50, bf: 0, x: 430, y: 150, color: "#10b981" },
          { id: "t1", key: 10, bf: 0, x: 80, y: 240, color: "#6b7280" },
          { id: "t3", key: 65, bf: 0, x: 510, y: 240, color: "#6b7280" },
        ],
        edges: [
          { from: "x", to: "y" }, { from: "x", to: "z" },
          { from: "y", to: "t1" },
          { from: "z", to: "t3" },
        ],
        highlight: [],
      },
    ],
  },
  RL: {
    label: "RL 双旋（先右旋 y，再左旋 z）",
    description:
      `失衡节点 z 的右子 y 的左侧插入节点 x，形成"右-左"折线形，需两步：先对 y 右旋，再对 z 左旋。`,
    frames: [
      {
        title: "初始状态（插入 x 后 z 失衡）",
        desc: "z bf=-2（右重），y bf=+1（左重），形成 RL 折线，单次左旋无法修复。",
        nodes: [
          { id: "z", key: 30, bf: -2, x: 200, y: 60, color: "#ef4444" },
          { id: "t0", key: 15, bf: 0, x: 70, y: 150, color: "#6b7280" },
          { id: "y", key: 60, bf: 1, x: 350, y: 150, color: "#f59e0b" },
          { id: "x", key: 45, bf: 0, x: 260, y: 240, color: "#3b82f6" },
          { id: "t3", key: 80, bf: 0, x: 440, y: 240, color: "#6b7280" },
        ],
        edges: [
          { from: "z", to: "t0" }, { from: "z", to: "y" },
          { from: "y", to: "x" }, { from: "y", to: "t3" },
        ],
        highlight: ["z", "y"],
      },
      {
        title: "第 1 步：对 y 进行右旋",
        desc: "x 上升取代 y，y 成为 x 的右子。RL 折线 → RR 直线，准备左旋。",
        nodes: [
          { id: "z", key: 30, bf: -2, x: 200, y: 60, color: "#ef4444" },
          { id: "t0", key: 15, bf: 0, x: 70, y: 150, color: "#6b7280" },
          { id: "x", key: 45, bf: -1, x: 350, y: 150, color: "#8b5cf6" },
          { id: "y", key: 60, bf: 0, x: 440, y: 240, color: "#f59e0b" },
          { id: "t3", key: 80, bf: 0, x: 530, y: 330, color: "#6b7280" },
        ],
        edges: [
          { from: "z", to: "t0" }, { from: "z", to: "x" },
          { from: "x", to: "y" },
          { from: "y", to: "t3" },
        ],
        highlight: ["x", "z"],
      },
      {
        title: "第 2 步：对 z 进行左旋",
        desc: "x 上升取代 z，z 成为 x 的左子，y 保持为 x 的右子。双旋完成！",
        nodes: [
          { id: "x", key: 45, bf: 0, x: 300, y: 60, color: "#10b981" },
          { id: "z", key: 30, bf: 0, x: 170, y: 150, color: "#10b981" },
          { id: "y", key: 60, bf: 0, x: 430, y: 150, color: "#10b981" },
          { id: "t0", key: 15, bf: 0, x: 80, y: 240, color: "#6b7280" },
          { id: "t3", key: 80, bf: 0, x: 510, y: 240, color: "#6b7280" },
        ],
        edges: [
          { from: "x", to: "z" }, { from: "x", to: "y" },
          { from: "z", to: "t0" },
          { from: "y", to: "t3" },
        ],
        highlight: [],
      },
    ],
  },
};

// ─── Tree SVG Renderer ────────────────────────────────────────────────────────
function TreeSVG({ frame }: { frame: Frame }) {
  const W = 600, H = 320;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 300 }}>
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
            stroke="#94a3b8" strokeWidth={2}
          />
        );
      })}
      {/* Nodes */}
      {frame.nodes.map((n) => {
        const isHL = frame.highlight.includes(n.id);
        const fill = isHL ? "#f97316" : n.color;
        return (
          <g key={n.id}>
            <circle cx={n.x} cy={n.y} r={22} fill={fill} stroke={isHL ? "#ea580c" : "#475569"} strokeWidth={isHL ? 2.5 : 1.5} />
            <text x={n.x} y={n.y + 1} textAnchor="middle" dominantBaseline="middle" fill="white" fontSize={13} fontWeight="bold">{n.key}</text>
            {/* balance factor badge */}
            {n.bf !== 0 && (
              <>
                <circle cx={n.x + 18} cy={n.y - 18} r={10} fill={Math.abs(n.bf) >= 2 ? "#ef4444" : "#64748b"} />
                <text x={n.x + 18} y={n.y - 17} textAnchor="middle" dominantBaseline="middle" fill="white" fontSize={9} fontWeight="bold">{n.bf > 0 ? `+${n.bf}` : n.bf}</text>
              </>
            )}
          </g>
        );
      })}
    </svg>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function AVLRotationAnimator() {
  const [rotType, setRotType] = useState<RotationType>("LL");
  const [frameIdx, setFrameIdx] = useState(0);

  const preset = PRESETS[rotType];
  const frame = preset.frames[frameIdx];
  const total = preset.frames.length;

  const handleRotType = (t: RotationType) => { setRotType(t); setFrameIdx(0); };

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5 my-6 shadow-sm">
      <h3 className="text-base font-bold text-slate-800 dark:text-slate-100 mb-1">🌲 AVL 树旋转动画</h3>
      <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">选择旋转类型，逐帧查看平衡恢复过程。红色节点 = 失衡（|bf|≥2），绿色 = 恢复平衡。</p>

      {/* Type selector */}
      <div className="flex flex-wrap gap-2 mb-4">
        {(Object.keys(PRESETS) as RotationType[]).map((t) => (
          <button
            key={t}
            onClick={() => handleRotType(t)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${rotType === t ? "bg-indigo-600 text-white shadow" : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-indigo-50 dark:hover:bg-slate-700"}`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Description */}
      <div className="text-xs text-slate-600 dark:text-slate-300 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg px-4 py-2 mb-4 border-l-4 border-indigo-400">
        {preset.description}
      </div>

      {/* SVG frame */}
      <div className="bg-slate-50 dark:bg-slate-800/60 rounded-xl p-3 mb-4">
        <TreeSVG frame={frame} />
      </div>

      {/* Step info */}
      <div className="mb-4 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 px-4 py-2">
        <div className="text-xs font-semibold text-amber-700 dark:text-amber-300 mb-1">
          步骤 {frameIdx + 1}/{total}：{frame.title}
        </div>
        <div className="text-xs text-amber-600 dark:text-amber-400">{frame.desc}</div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        <button onClick={() => setFrameIdx(0)} disabled={frameIdx === 0}
          className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 text-xs disabled:opacity-40 hover:bg-slate-200 dark:hover:bg-slate-700">⏮ 首帧</button>
        <button onClick={() => setFrameIdx((f) => Math.max(0, f - 1))} disabled={frameIdx === 0}
          className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 text-xs disabled:opacity-40 hover:bg-slate-200 dark:hover:bg-slate-700">◀ 上一步</button>
        <div className="flex-1 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
          <div className="h-full bg-indigo-500 rounded-full transition-all" style={{ width: `${((frameIdx + 1) / total) * 100}%` }} />
        </div>
        <button onClick={() => setFrameIdx((f) => Math.min(total - 1, f + 1))} disabled={frameIdx === total - 1}
          className="px-3 py-1.5 rounded-lg bg-indigo-600 text-white text-xs disabled:opacity-40 hover:bg-indigo-700">下一步 ▶</button>
        <button onClick={() => setFrameIdx(total - 1)} disabled={frameIdx === total - 1}
          className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 text-xs disabled:opacity-40 hover:bg-slate-200 dark:hover:bg-slate-700">末帧 ⏭</button>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 mt-4 text-xs text-slate-500 dark:text-slate-400">
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full inline-block bg-red-500" />失衡节点（|bf|≥2）</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full inline-block bg-amber-500" />旋转轴节点</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full inline-block bg-blue-500" />新插入节点</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full inline-block bg-emerald-500" />平衡后节点</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full inline-block bg-slate-400" />子树（T）</span>
      </div>
    </div>
  );
}
