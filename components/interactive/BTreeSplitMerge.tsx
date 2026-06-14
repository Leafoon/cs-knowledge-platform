"use client";
import React, { useState } from "react";

// ─── Types ─────────────────────────────────────────────────────────────────
interface BKey { val: number; highlight?: string }
interface BNode {
  id: string;
  keys: BKey[];
  x: number;
  y: number;
  width: number;
  color?: string;       // border color
  bg?: string;          // fill color
  label?: string;       // top label
}
interface BEdge { from: string; to: string; toSlot?: number }
interface Frame {
  title: string;
  desc: string;
  nodes: BNode[];
  edges: BEdge[];
  highlight?: string[]; // node ids
}

type ScenarioKey = "split" | "merge" | "delete";

// ─── Scenarios ──────────────────────────────────────────────────────────────
const SCENARIOS: Record<ScenarioKey, { label: string; summary: string; frames: Frame[] }> = {
  split: {
    label: "插入分裂（t=2）",
    summary: `B 树最小度数 t=2，节点键数范围 [t-1, 2t-1]=[1,3]。当节点已满（3个键）时再插入，
需先将节点"一分为二"，中间键上浮到父节点。`,
    frames: [
      {
        title: "初始：节点 N 已满（3 个键）",
        desc: `最小度数 t=2，节点最多含 2t-1=3 个键。N 已含 [10, 20, 30]，
现在要插入 25，因为 N 满了，必须先分裂 N，再插入。`,
        nodes: [
          { id: "root", keys: [{ val: 50 }], x: 340, y: 40, width: 80, bg: "#1e293b", label: "根节点" },
          { id: "N", keys: [{ val: 10 }, { val: 20 }, { val: 30 }], x: 80, y: 150, width: 240,
            bg: "#dc2626", color: "#fca5a5", label: "N（已满，将插入 25）" },
          { id: "R", keys: [{ val: 60 }, { val: 70 }], x: 480, y: 150, width: 160, bg: "#1e293b", label: "兄弟节点 R" },
        ],
        edges: [{ from: "root", to: "N" }, { from: "root", to: "R" }],
        highlight: ["N"],
      },
      {
        title: "步骤 1：找到中间键，准备分裂",
        desc: `N = [10, 20, 30]，中间键索引 = ⌊(2t-1)/2⌋ = ⌊3/2⌋ = 1，中间键 = 20。
左半部分 L = [10]，右半部分 R_new = [30]，20 将上浮到父节点。`,
        nodes: [
          { id: "root", keys: [{ val: 50 }], x: 340, y: 40, width: 80, bg: "#1e293b", label: "父节点（即将接收 20）" },
          { id: "L", keys: [{ val: 10 }], x: 60, y: 160, width: 80, bg: "#7c3aed", color: "#c4b5fd", label: "L（左半）" },
          { id: "mid", keys: [{ val: 20, highlight: "#fbbf24" }], x: 240, y: 155, width: 80,
            bg: "#92400e", color: "#fde68a", label: "中间键↑" },
          { id: "R_new", keys: [{ val: 30 }], x: 420, y: 160, width: 80, bg: "#065f46", color: "#6ee7b7", label: "R_new（右半）" },
          { id: "R", keys: [{ val: 60 }, { val: 70 }], x: 560, y: 160, width: 160, bg: "#1e293b", label: "R" },
        ],
        edges: [
          { from: "root", to: "L" }, { from: "root", to: "R_new" }, { from: "root", to: "R" },
        ],
        highlight: ["mid", "L", "R_new"],
      },
      {
        title: "步骤 2：中间键 20 上浮，父节点更新",
        desc: `20 插入父节点（原 [50] → [20, 50]）。父节点现在有 2 个键、3 个子指针。
父节点未满（2 < 2t-1=3），无需继续分裂。`,
        nodes: [
          { id: "root", keys: [{ val: 20, highlight: "#fbbf24" }, { val: 50 }], x: 240, y: 40, width: 160,
            bg: "#1e40af", color: "#93c5fd", label: "父节点（20 已上浮 ✓）" },
          { id: "L", keys: [{ val: 10 }], x: 60, y: 150, width: 80, bg: "#7c3aed", color: "#c4b5fd", label: "L" },
          { id: "R_new", keys: [{ val: 30 }], x: 220, y: 150, width: 80, bg: "#065f46", color: "#6ee7b7", label: "R_new" },
          { id: "R", keys: [{ val: 60 }, { val: 70 }], x: 400, y: 150, width: 160, bg: "#1e293b", label: "R" },
        ],
        edges: [
          { from: "root", to: "L" }, { from: "root", to: "R_new" }, { from: "root", to: "R" },
        ],
        highlight: ["root"],
      },
      {
        title: "步骤 3：将 25 插入正确叶子节点",
        desc: `25 > 20 且 25 < 50，走 R_new（[30]）这棵子树。
R_new = [30]，插入 25 → R_new = [25, 30]，未满（1 < 3），直接插入完成！`,
        nodes: [
          { id: "root", keys: [{ val: 20 }, { val: 50 }], x: 240, y: 40, width: 160, bg: "#1e293b", label: "根节点" },
          { id: "L", keys: [{ val: 10 }], x: 60, y: 150, width: 80, bg: "#1e293b", label: "L" },
          { id: "R_new", keys: [{ val: 25, highlight: "#4ade80" }, { val: 30 }], x: 200, y: 150, width: 160,
            bg: "#065f46", color: "#6ee7b7", label: "R_new（插入 25 ✓）" },
          { id: "R", keys: [{ val: 60 }, { val: 70 }], x: 420, y: 150, width: 160, bg: "#1e293b", label: "R" },
        ],
        edges: [
          { from: "root", to: "L" }, { from: "root", to: "R_new" }, { from: "root", to: "R" },
        ],
        highlight: ["R_new"],
      },
    ],
  },
  merge: {
    label: "合并操作（删除触发）",
    summary: `删除时，若目标节点键数 = t-1（最少），且兄弟也无法借，则合并兄弟 + 父节点下沉一个键。
合并导致父节点减少一个键，可能引发向上连锁合并。`,
    frames: [
      {
        title: "初始：t=2，节点 A 即将删除键 5",
        desc: `节点 A = [5]，只有 t-1=1 个键，是允许的下限。
删除 5 后 A 将有 0 个键，违反 B 树性质，必须从兄弟借键或合并。`,
        nodes: [
          { id: "par", keys: [{ val: 15 }], x: 280, y: 40, width: 80, bg: "#1e293b", label: "父节点" },
          { id: "A", keys: [{ val: 5, highlight: "#ef4444" }], x: 120, y: 150, width: 80,
            bg: "#7f1d1d", color: "#fca5a5", label: "A（要删 5）" },
          { id: "B", keys: [{ val: 20 }], x: 360, y: 150, width: 80, bg: "#1e293b", label: "B（兄弟，也仅 1 键）" },
        ],
        edges: [{ from: "par", to: "A" }, { from: "par", to: "B" }],
        highlight: ["A", "par"],
      },
      {
        title: "检查：兄弟 B 也是最少键（无法借）",
        desc: `B = [20]，仅 1 个键（= t-1），无法"借"给 A。
因此选择合并：A + 父节点下沉键 15 + B → 新节点 M = [5, 15, 20]。
但先删 5 再合并：M = [15, 20]（已删 5）。`,
        nodes: [
          { id: "par", keys: [{ val: 15, highlight: "#fbbf24" }], x: 280, y: 40, width: 80,
            bg: "#92400e", color: "#fde68a", label: "父节点（15 将下沉）" },
          { id: "A", keys: [{ val: 5 }], x: 120, y: 150, width: 80, bg: "#7f1d1d", color: "#fca5a5", label: "A（待合并）" },
          { id: "B", keys: [{ val: 20 }], x: 360, y: 150, width: 80, bg: "#7f1d1d", color: "#fca5a5", label: "B（待合并）" },
        ],
        edges: [{ from: "par", to: "A" }, { from: "par", to: "B" }],
        highlight: ["par", "A", "B"],
      },
      {
        title: "合并：父下沉 15，A+B 合并为 M",
        desc: `删去 5，父节点 15 下沉到合并节点，M = [15, 20]。
父节点键数 = 0，若父是根且 M 是唯一孩子，则 M 成为新根（树高减 1）。`,
        nodes: [
          { id: "M", keys: [{ val: 15 }, { val: 20 }], x: 200, y: 120, width: 160,
            bg: "#065f46", color: "#6ee7b7", label: "M = 合并结果（新根 ✓）" },
        ],
        edges: [],
        highlight: ["M"],
      },
    ],
  },
  delete: {
    label: "删除借键（旋转）",
    summary: `删除时，若目标节点键数不足但兄弟节点键数 ≥ t，可以"借"一个键（通过父节点中转），
这等价于对 B 树做一次旋转，无需合并。`,
    frames: [
      {
        title: "初始：删除节点 A 中的键 5，A 最少",
        desc: `A = [5]，只有 1 个键（= t-1=1）。删除 5 后 A 为空，需要求助。
右兄弟 B = [20, 30]，有 2 个键（> t-1=1），可以借！`,
        nodes: [
          { id: "par", keys: [{ val: 15 }], x: 280, y: 40, width: 80, bg: "#1e293b", label: "父节点" },
          { id: "A", keys: [{ val: 5, highlight: "#ef4444" }], x: 100, y: 150, width: 80,
            bg: "#7f1d1d", color: "#fca5a5", label: "A（要删 5）" },
          { id: "B", keys: [{ val: 20 }, { val: 30 }], x: 360, y: 150, width: 160,
            bg: "#065f46", color: "#6ee7b7", label: "B（兄弟，2 个键，可借！）" },
        ],
        edges: [{ from: "par", to: "A" }, { from: "par", to: "B" }],
        highlight: ["A", "B"],
      },
      {
        title: "借键旋转：父下降 + 兄弟最小键上升",
        desc: `父节点中分隔 A 和 B 的键 15 下降到 A。
B 中最小键 20 填补上去到父节点。B 剩余 [30]。
A 删去 5，接收 15：A = [15]。`,
        nodes: [
          { id: "par", keys: [{ val: 20, highlight: "#fbbf24" }], x: 280, y: 40, width: 80,
            bg: "#1e40af", color: "#93c5fd", label: "父节点（20 从 B 上来）" },
          { id: "A", keys: [{ val: 15, highlight: "#4ade80" }], x: 100, y: 150, width: 80,
            bg: "#7c3aed", color: "#c4b5fd", label: "A（接收 15 ✓）" },
          { id: "B", keys: [{ val: 30 }], x: 360, y: 150, width: 80, bg: "#1e293b", label: "B（借出 20）" },
        ],
        edges: [{ from: "par", to: "A" }, { from: "par", to: "B" }],
        highlight: ["par", "A", "B"],
      },
      {
        title: "删除借键完成，所有节点满足约束",
        desc: `A = [15]（1 个键 = t-1 ✓），B = [30]（1 个键 = t-1 ✓），父 = [20]（1 个键 ✓）。
整个操作仅涉及常数个节点，无需向上传播，时间 O(1)（不计搜索路径）。`,
        nodes: [
          { id: "par", keys: [{ val: 20 }], x: 280, y: 40, width: 80, bg: "#1e293b", label: "父节点 ✓" },
          { id: "A", keys: [{ val: 15 }], x: 100, y: 150, width: 80, bg: "#1e293b", label: "A ✓" },
          { id: "B", keys: [{ val: 30 }], x: 360, y: 150, width: 80, bg: "#1e293b", label: "B ✓" },
        ],
        edges: [{ from: "par", to: "A" }, { from: "par", to: "B" }],
        highlight: [],
      },
    ],
  },
};

// ─── Node renderer ──────────────────────────────────────────────────────────
function BNodeSVG({ node, isHighlighted }: { node: BNode; isHighlighted: boolean }) {
  const kw = node.width / Math.max(node.keys.length, 1);
  const totalW = node.width;
  const nodeH = 44;
  const rx = node.x - totalW / 2;
  const ry = node.y;
  const borderCol = isHighlighted ? "#f59e0b" : (node.color ?? "#475569");
  const bgCol = isHighlighted ? (node.bg ?? "#1e40af") : (node.bg ?? "#1e293b");

  return (
    <g>
      {node.label && (
        <text x={node.x} y={ry - 10} textAnchor="middle" fontSize={11} fill="#94a3b8">
          {node.label}
        </text>
      )}
      <rect x={rx} y={ry} width={totalW} height={nodeH} rx={6}
        fill={bgCol} stroke={borderCol} strokeWidth={isHighlighted ? 2.5 : 1.5} />
      {node.keys.map((k, i) => {
        const kx = rx + i * kw;
        return (
          <g key={i}>
            {i > 0 && (
              <line x1={kx} y1={ry} x2={kx} y2={ry + nodeH} stroke={borderCol} strokeWidth={1} />
            )}
            <rect x={kx + 2} y={ry + 2} width={kw - 4} height={nodeH - 4} rx={4}
              fill={k.highlight ?? "transparent"} opacity={0.25} />
            <text x={kx + kw / 2} y={ry + nodeH / 2 + 5} textAnchor="middle"
              fontSize={14} fontWeight="bold" fill={k.highlight ? k.highlight : "#f1f5f9"}>
              {k.val}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function BEdgeSVG({ from, to, nodes }: { from: BNode; to: BNode; nodes: BNode[] }) {
  const fx = from.x;
  const fy = from.y + 44;
  const tx = to.x;
  const ty = to.y;
  return (
    <line x1={fx} y1={fy} x2={tx} y2={ty}
      stroke="#475569" strokeWidth={1.5} markerEnd="url(#arrow)" />
  );
}

// ─── Main Component ─────────────────────────────────────────────────────────
export default function BTreeSplitMerge() {
  const [scenario, setScenario] = useState<ScenarioKey>("split");
  const [frame, setFrame] = useState(0);

  const sc = SCENARIOS[scenario];
  const fr = sc.frames[frame];

  const nodeMap = Object.fromEntries(fr.nodes.map(n => [n.id, n]));

  function gotoScenario(k: ScenarioKey) {
    setScenario(k);
    setFrame(0);
  }

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 p-5 my-6 font-sans">
      {/* Scenario tabs */}
      <div className="flex flex-wrap gap-2 mb-4">
        {(Object.keys(SCENARIOS) as ScenarioKey[]).map(k => (
          <button key={k} onClick={() => gotoScenario(k)}
            className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-colors ${
              scenario === k
                ? "bg-indigo-600 text-white"
                : "bg-slate-700 text-slate-300 hover:bg-slate-600"
            }`}>
            {SCENARIOS[k].label}
          </button>
        ))}
      </div>

      {/* Summary */}
      <div className="rounded-lg bg-slate-800 border border-slate-700 px-4 py-3 mb-4 text-slate-300 text-sm whitespace-pre-wrap leading-relaxed">
        {sc.summary}
      </div>

      {/* SVG canvas */}
      <div className="rounded-lg bg-slate-950 border border-slate-800 overflow-x-auto mb-4">
        <svg width="640" height="260" viewBox="0 0 640 260" className="mx-auto block">
          <defs>
            <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L0,6 L8,3 z" fill="#475569" />
            </marker>
          </defs>
          {fr.edges.map((e, i) => {
            const fn = nodeMap[e.from];
            const tn = nodeMap[e.to];
            if (!fn || !tn) return null;
            return <BEdgeSVG key={i} from={fn} to={tn} nodes={fr.nodes} />;
          })}
          {fr.nodes.map(n => (
            <BNodeSVG key={n.id} node={n} isHighlighted={!!fr.highlight?.includes(n.id)} />
          ))}
        </svg>
      </div>

      {/* Frame info */}
      <div className="rounded-lg bg-indigo-950 border border-indigo-800 px-4 py-3 mb-4">
        <div className="text-indigo-300 font-bold text-sm mb-1">{fr.title}</div>
        <div className="text-slate-300 text-sm whitespace-pre-wrap leading-relaxed">{fr.desc}</div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        <button onClick={() => setFrame(0)} disabled={frame === 0}
          className="px-3 py-1.5 rounded bg-slate-700 text-slate-300 hover:bg-slate-600 disabled:opacity-30 text-lg">⏮</button>
        <button onClick={() => setFrame(f => Math.max(0, f - 1))} disabled={frame === 0}
          className="px-3 py-1.5 rounded bg-slate-700 text-slate-300 hover:bg-slate-600 disabled:opacity-30 text-lg">◀</button>
        <span className="text-slate-400 text-sm flex-1 text-center">
          步骤 {frame + 1} / {sc.frames.length}
        </span>
        <button onClick={() => setFrame(f => Math.min(sc.frames.length - 1, f + 1))} disabled={frame === sc.frames.length - 1}
          className="px-3 py-1.5 rounded bg-slate-700 text-slate-300 hover:bg-slate-600 disabled:opacity-30 text-lg">▶</button>
        <button onClick={() => setFrame(sc.frames.length - 1)} disabled={frame === sc.frames.length - 1}
          className="px-3 py-1.5 rounded bg-slate-700 text-slate-300 hover:bg-slate-600 disabled:opacity-30 text-lg">⏭</button>
      </div>
      {/* Progress bar */}
      <div className="mt-3 h-1.5 rounded-full bg-slate-700">
        <div className="h-1.5 rounded-full bg-indigo-500 transition-all duration-300"
          style={{ width: `${((frame + 1) / sc.frames.length) * 100}%` }} />
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4 text-xs text-slate-400">
        {[
          { c: "#fbbf24", l: "中间键/关键键" },
          { c: "#ef4444", l: "待删除键" },
          { c: "#4ade80", l: "新插入键" },
          { c: "#c4b5fd", l: "分裂左半" },
          { c: "#6ee7b7", l: "分裂右半/合并结果" },
        ].map(({ c, l }) => (
          <span key={l} className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded" style={{ background: c }} />
            {l}
          </span>
        ))}
      </div>
    </div>
  );
}
