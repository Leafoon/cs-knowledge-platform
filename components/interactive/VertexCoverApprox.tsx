"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

// ── Graph: 6-node ring with two crossing chords ──
const NODES = [
  { id: 0, x: 100, y: 155, label: "v₀" },
  { id: 1, x: 175, y:  60, label: "v₁" },
  { id: 2, x: 285, y:  60, label: "v₂" },
  { id: 3, x: 360, y: 155, label: "v₃" },
  { id: 4, x: 285, y: 250, label: "v₄" },
  { id: 5, x: 175, y: 250, label: "v₅" },
];

const ALL_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], // ring
  [1, 4], [0, 3],                                    // chords
];

// ── Pre-computed algorithm steps ──
interface AlgoStep {
  pick: [number, number] | null;   // edge chosen this step (null = done)
  cover: number[];
  matching: [number, number][];    // edges selected so far (the matching M)
  removed: Set<string>;             // edge keys "u-v" removed so far
  desc: string;
  detail: string;
}

function ekey(a: number, b: number) { return `${Math.min(a,b)}-${Math.max(a,b)}`; }

function buildSteps(): AlgoStep[] {
  const steps: AlgoStep[] = [];
  const cover = new Set<number>();
  const matching: [number, number][] = [];
  const removed = new Set<string>();

  steps.push({
    pick: null,
    cover: [],
    matching: [],
    removed: new Set(),
    desc: "初始状态：图 G 有 6 个顶点、8 条边，覆盖集 C = ∅",
    detail: `目标：找最小顶点覆盖（每条边至少一个端点在 C 内）。\n最优解 OPT = 3（例如 {v₁, v₃, v₅}）。\n算法保证输出 |C| ≤ 2 × OPT = 6。`,
  });

  // Greedy order: for reproducibility
  const pickOrder: [number, number][] = [[0, 1], [2, 3], [4, 5]];

  for (const [u, v] of pickOrder) {
    // pick edge (u,v)
    matching.push([u, v]);
    cover.add(u);
    cover.add(v);

    // remove all edges touching u or v
    for (const [a, b] of ALL_EDGES) {
      if (a === u || a === v || b === u || b === v) {
        removed.add(ekey(a, b));
      }
    }

    steps.push({
      pick: [u, v],
      cover: [...cover],
      matching: [...matching],
      removed: new Set(removed),
      desc: `选取边 (${NODES[u].label}, ${NODES[v].label})，将 ${NODES[u].label} 和 ${NODES[v].label} 加入覆盖集 C`,
      detail: `将 ${NODES[u].label} 和 ${NODES[v].label} 同时加入 C（贪心：不只选"更好那个"，两个都选，保持匹配结构）。\n然后删除所有与这两个顶点相关联的边。\n当前 C = {${[...cover].map(i => NODES[i].label).join(", ")}}，|C| = ${cover.size}/6`,
    });
  }

  // Final: all covered
  const remaining = ALL_EDGES.filter(([a, b]) => !removed.has(ekey(a, b)));
  const finalDesc = remaining.length === 0
    ? "所有边均已覆盖，算法结束！"
    : `仍有 ${remaining.length} 条边未覆盖，继续...`;

  steps.push({
    pick: null,
    cover: [...cover],
    matching: [...matching],
    removed: new Set(removed),
    desc: `算法结束：所有边已覆盖，覆盖集 C = {${[...cover].map(i => NODES[i].label).join(", ")}}`,
    detail: `|C| = ${cover.size}，OPT = 3\n匹配 M = {${matching.map(([a,b]) => `(${NODES[a].label},${NODES[b].label})`).join(", ")}}\n|M| = 3，|C| = 2|M| = 6 ≤ 2×OPT ✓\n\n2-近似保证成立！这是最坏情况的紧例子。`,
  });

  return steps;
}

const STEPS = buildSteps();

export function VertexCoverApprox() {
  const [step, setStep] = useState(0);

  const cur = STEPS[step];
  const coverSet = new Set(cur.cover);
  const matchingKeys = new Set(cur.matching.map(([a, b]) => ekey(a, b)));
  const pickKey = cur.pick ? ekey(cur.pick[0], cur.pick[1]) : null;

  const getEdgeStyle = ([a, b]: [number, number]) => {
    const k = ekey(a, b);
    if (k === pickKey) return { stroke: "#f59e0b", strokeWidth: 5, opacity: 1, dash: "none" };
    if (cur.removed.has(k)) return { stroke: "#e2e8f0", strokeWidth: 1.5, opacity: 0.3, dash: "5,3" };
    if (matchingKeys.has(k)) return { stroke: "#6366f1", strokeWidth: 3.5, opacity: 1, dash: "none" };
    return { stroke: "#94a3b8", strokeWidth: 2, opacity: 0.9, dash: "none" };
  };

  const getNodeStyle = (id: number) => {
    if (coverSet.has(id)) {
      // Is this node freshly added this step?
      const fresh = cur.pick && (cur.pick[0] === id || cur.pick[1] === id);
      return fresh
        ? { fill: "#f59e0b", stroke: "#d97706", textColor: "white", ring: "3px solid #fde68a" }
        : { fill: "#6366f1", stroke: "#4f46e5", textColor: "white", ring: "none" };
    }
    return { fill: "#f8fafc", stroke: "#94a3b8", textColor: "#475569", ring: "none" };
  };

  return (
    <div className="w-full max-w-4xl mx-auto my-6 rounded-2xl overflow-hidden border border-indigo-200 dark:border-indigo-800 shadow-xl bg-white dark:bg-slate-900">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-violet-600 px-6 py-4 flex items-center justify-between flex-wrap gap-3">
        <div>
          <h3 className="text-base font-bold text-white">顶点覆盖 2-近似算法（贪心匹配）</h3>
          <p className="text-xs text-indigo-100 mt-0.5">步进查看算法执行过程，理解 2-近似证明</p>
        </div>
        {/* Stats badges */}
        <div className="flex gap-2 flex-wrap">
          <span className="text-xs px-2.5 py-1 rounded-full bg-white/20 text-white font-mono">
            |C| = {cur.cover.length}
          </span>
          <span className="text-xs px-2.5 py-1 rounded-full bg-white/20 text-white font-mono">
            OPT = 3
          </span>
          <span className={`text-xs px-2.5 py-1 rounded-full font-mono font-bold transition-colors ${
            cur.cover.length <= 6 ? "bg-emerald-500 text-white" : "bg-rose-500 text-white"
          }`}>
            {cur.cover.length} ≤ 2×3 = 6 ✓
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-0">
        {/* === SVG Graph === */}
        <div className="p-4 flex items-center justify-center bg-slate-50 dark:bg-slate-800/50">
          <svg viewBox="0 0 460 320" width="100%" className="max-h-72">
            {/* Edges */}
            {ALL_EDGES.map(([a, b]) => {
              const na = NODES[a], nb = NODES[b];
              const es = getEdgeStyle([a, b]);
              const k = ekey(a, b);
              return (
                <motion.line
                  key={k}
                  x1={na.x} y1={na.y} x2={nb.x} y2={nb.y}
                  stroke={es.stroke}
                  strokeWidth={es.strokeWidth}
                  opacity={es.opacity}
                  strokeDasharray={es.dash === "none" ? undefined : es.dash}
                  animate={{ stroke: es.stroke, strokeWidth: es.strokeWidth, opacity: es.opacity }}
                  transition={{ duration: 0.35 }}
                />
              );
            })}

            {/* Nodes */}
            {NODES.map((nd) => {
              const ns = getNodeStyle(nd.id);
              return (
                <g key={nd.id}>
                  <motion.circle
                    cx={nd.x} cy={nd.y} r={22}
                    fill={ns.fill} stroke={ns.stroke} strokeWidth={2.5}
                    animate={{ fill: ns.fill, stroke: ns.stroke }}
                    transition={{ duration: 0.3 }}
                  />
                  <text
                    x={nd.x} y={nd.y + 4}
                    textAnchor="middle" dominantBaseline="middle"
                    fontSize={12} fontWeight="bold" fill={ns.textColor}
                  >
                    {nd.label}
                  </text>
                </g>
              );
            })}

            {/* Legend */}
            <g transform="translate(10, 285)">
              {[
                { color: "#f59e0b", label: "本步新增" },
                { color: "#6366f1", label: "已覆盖" },
                { color: "#f8fafc", label: "未覆盖", stroke: "#94a3b8" },
              ].map(({ color, label, stroke }, i) => (
                <g key={label} transform={`translate(${i * 100}, 0)`}>
                  <circle cx={6} cy={5} r={6} fill={color} stroke={stroke ?? color} strokeWidth={1.5} />
                  <text x={15} y={9} fontSize={10} fill="#64748b">{label}</text>
                </g>
              ))}
            </g>
          </svg>
        </div>

        {/* === Right: step info === */}
        <div className="w-full md:w-72 p-5 border-t md:border-t-0 md:border-l border-slate-200 dark:border-slate-700 flex flex-col gap-4">
          {/* Step description */}
          <AnimatePresence mode="wait">
            <motion.div
              key={step}
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={{ duration: 0.2 }}
              className="flex-1"
            >
              {/* Step number */}
              <div className="flex items-center gap-2 mb-3">
                <span className="w-7 h-7 rounded-full bg-indigo-600 text-white text-xs font-bold flex items-center justify-center">
                  {step}
                </span>
                <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                  步骤 {step} / {STEPS.length - 1}
                </span>
              </div>

              <p className="text-sm font-semibold text-slate-700 dark:text-slate-200 mb-2 leading-snug">
                {cur.desc}
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed whitespace-pre-line">
                {cur.detail}
              </p>
            </motion.div>
          </AnimatePresence>

          {/* Matching visualization */}
          {cur.matching.length > 0 && (
            <div className="bg-indigo-50 dark:bg-indigo-900/30 rounded-xl p-3">
              <div className="text-xs font-bold text-indigo-600 dark:text-indigo-300 mb-2">匹配 M（选出的边）</div>
              {cur.matching.map(([a, b], i) => (
                <div key={i} className="text-xs font-mono text-indigo-700 dark:text-indigo-300">
                  ({NODES[a].label}, {NODES[b].label})
                </div>
              ))}
            </div>
          )}

          {/* Cover visualization */}
          {cur.cover.length > 0 && (
            <div className="bg-violet-50 dark:bg-violet-900/30 rounded-xl p-3">
              <div className="text-xs font-bold text-violet-600 dark:text-violet-300 mb-2">覆盖集 C</div>
              <div className="flex flex-wrap gap-1">
                {cur.cover.map((id) => (
                  <span key={id} className="text-xs font-mono px-2 py-0.5 rounded bg-violet-500 text-white">
                    {NODES[id].label}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Controls */}
          <div className="flex gap-2">
            <button
              onClick={() => setStep(0)}
              disabled={step === 0}
              className="flex-1 py-1.5 rounded-lg text-xs font-semibold border border-slate-200 dark:border-slate-600 text-slate-600 dark:text-slate-300 disabled:opacity-30 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
            >
              ↩ 重置
            </button>
            <button
              onClick={() => setStep((s) => Math.max(0, s - 1))}
              disabled={step === 0}
              className="flex-1 py-1.5 rounded-lg text-xs font-semibold border border-indigo-200 dark:border-indigo-700 text-indigo-600 dark:text-indigo-300 disabled:opacity-30 hover:bg-indigo-50 dark:hover:bg-indigo-900/30 transition-colors"
            >
              ← 上一步
            </button>
            <button
              onClick={() => setStep((s) => Math.min(STEPS.length - 1, s + 1))}
              disabled={step === STEPS.length - 1}
              className="flex-1 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 text-white disabled:opacity-30 transition-colors"
            >
              下一步 →
            </button>
          </div>

          {/* Progress bar */}
          <div className="h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 rounded-full transition-all duration-300"
              style={{ width: `${(step / (STEPS.length - 1)) * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
