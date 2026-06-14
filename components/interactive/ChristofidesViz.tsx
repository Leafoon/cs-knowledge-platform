"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

// ── 6 cities (Euclidean) ──
const CITIES = [
  { id: 0, x: 80,  y: 100, label: "A" },
  { id: 1, x: 240, y:  60, label: "B" },
  { id: 2, x: 380, y: 110, label: "C" },
  { id: 3, x: 360, y: 260, label: "D" },
  { id: 4, x: 200, y: 310, label: "E" },
  { id: 5, x:  60, y: 240, label: "F" },
];

function dist(a: number, b: number) {
  const dx = CITIES[a].x - CITIES[b].x;
  const dy = CITIES[a].y - CITIES[b].y;
  return Math.round(Math.sqrt(dx * dx + dy * dy));
}

// Pre-computed MST edges (Kruskal on Euclidean distances)
const MST_EDGES: [number, number][] = [[0, 5], [1, 2], [2, 3], [3, 4], [4, 5]];
// Odd-degree vertices in MST: 0 (deg 1), 1 (deg 1)
const ODD_VERTICES = [0, 1];
// Min-weight perfect matching on {0,1}: edge (0,1)
const MATCHING_EDGES: [number, number][] = [[0, 1]];
// Euler tour (combined graph has Euler circuit)
const EULER_SEQUENCE = [0, 1, 2, 3, 4, 5, 0]; // 0→1→2→3→4→5→0
// Hamiltonian tour (same in this case, no shortcutting needed)
const HAMILTON_SEQUENCE = [0, 1, 2, 3, 4, 5, 0];

interface Phase {
  id: number;
  title: string;
  subtitle: string;
  insight: string;
  cost: string;
}

const PHASES: Phase[] = [
  {
    id: 1,
    title: "① 求最小生成树（MST）",
    subtitle: "Kruskal 算法，时间 O(E log V)",
    insight: "关键性质：w(MST) ≤ w(OPT_TSP)。因为将最优 TSP 回路去掉任意一条边，就得到一棵生成树（权重不小于 MST）。",
    cost: `MST 权重 = ${MST_EDGES.reduce((s, [a,b]) => s + dist(a,b), 0)}`,
  },
  {
    id: 2,
    title: "② 找奇度顶点集合 O",
    subtitle: "MST 中度为奇数的顶点",
    insight: '任何图的奇度顶点数目必为偶数（握手定理）。奇度顶点需要通过匹配"修复"，使所有顶点度数变为偶数（欧拉图条件）。',
    cost: `奇度顶点: ${ODD_VERTICES.map(i => CITIES[i].label).join(", ")}（各度=1）`,
  },
  {
    id: 3,
    title: "③ 奇度顶点上的最小完美匹配",
    subtitle: "Blossom 算法（O(n³)）或本例暴力",
    insight: `最小完美匹配 w(M) ≤ OPT/2。证明：OPT 回路限制在奇度顶点上形成两个交替匹配，较小者 ≤ OPT/2。\n合并代价 ≤ OPT + OPT/2 = 3/2 · OPT`,
    cost: `匹配边 (${CITIES[0].label},${CITIES[1].label}) 权重 = ${dist(0,1)}`,
  },
  {
    id: 4,
    title: "④ 构造欧拉回路",
    subtitle: "MST + 匹配边 → 所有顶点度数为偶数 → 欧拉图",
    insight: "合并 MST 和匹配 M 后，每个顶点度数均为偶数（偶+偶=偶，奇+奇=偶）。根据欧拉定理，连通图中所有顶点度数为偶数则存在欧拉回路。",
    cost: `欧拉路径：${EULER_SEQUENCE.map(i => CITIES[i].label).join(" → ")}`,
  },
  {
    id: 5,
    title: "⑤ 提取哈密顿回路（1.5-近似！）",
    subtitle: "跳过重复节点（利用三角不等式）",
    insight: `跳过重复节点只会让路程变短或不变（三角不等式：d(A,C) ≤ d(A,B)+d(B,C)）。\n最终路线总权重 ≤ 3/2 · OPT，这就是 Christofides 的 1.5-近似保证！`,
    cost: `最终路线权重 = ${HAMILTON_SEQUENCE.slice(0,-1).reduce((s,v,i) => s + dist(v, HAMILTON_SEQUENCE[i+1]), 0)}`,
  },
];

export function ChristofidesViz() {
  const [phase, setPhase] = useState(1);

  const p = PHASES[phase - 1];

  const showMST = phase >= 1;
  const showOdd = phase >= 2;
  const showMatching = phase >= 3;
  const showEuler = phase === 4;
  const showHamilton = phase === 5;

  const getAllEdges = () => {
    const edges: { a: number; b: number; type: "mst" | "match" | "hamilton" | "euler" }[] = [];
    if (showMST && !showHamilton) {
      for (const [a, b] of MST_EDGES) edges.push({ a, b, type: "mst" });
    }
    if (showMatching && !showEuler && !showHamilton) {
      for (const [a, b] of MATCHING_EDGES) edges.push({ a, b, type: "match" });
    }
    if (showEuler) {
      for (const [a, b] of MST_EDGES) edges.push({ a, b, type: "mst" });
      for (const [a, b] of MATCHING_EDGES) edges.push({ a, b, type: "match" });
    }
    if (showHamilton) {
      for (let i = 0; i < HAMILTON_SEQUENCE.length - 1; i++) {
        edges.push({ a: HAMILTON_SEQUENCE[i], b: HAMILTON_SEQUENCE[i + 1], type: "hamilton" });
      }
    }
    return edges;
  };

  const edgeColor: Record<string, string> = {
    mst: "#6366f1",
    match: "#f59e0b",
    hamilton: "#10b981",
    euler: "#ec4899",
  };

  const edgeWidth: Record<string, number> = {
    mst: 3,
    match: 3.5,
    hamilton: 4,
    euler: 2.5,
  };

  const edgeDash: Record<string, string | undefined> = {
    mst: undefined,
    match: "6,3",
    hamilton: undefined,
    euler: "4,2",
  };

  return (
    <div className="w-full max-w-5xl mx-auto my-6 rounded-2xl overflow-hidden border border-teal-200 dark:border-teal-800 shadow-xl bg-white dark:bg-slate-900">
      {/* Header */}
      <div className="bg-gradient-to-r from-teal-600 via-cyan-600 to-blue-600 px-6 py-4">
        <h3 className="text-base font-bold text-white">Christofides 1.5-近似 TSP 算法</h3>
        <p className="text-xs text-teal-100 mt-0.5">5 步图解：MST → 奇度匹配 → 欧拉回路 → 哈密顿路径</p>
      </div>

      {/* Phase tabs */}
      <div className="flex border-b border-slate-200 dark:border-slate-700 overflow-x-auto">
        {PHASES.map((ph) => (
          <button
            key={ph.id}
            onClick={() => setPhase(ph.id)}
            className={`flex-shrink-0 px-4 py-3 text-xs font-semibold border-b-2 transition-all whitespace-nowrap ${
              phase === ph.id
                ? "border-teal-500 text-teal-600 dark:text-teal-400 bg-teal-50 dark:bg-teal-900/30"
                : "border-transparent text-slate-500 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800"
            }`}
          >
            {ph.title.split(" ")[0]} 步骤{ph.id}
          </button>
        ))}
      </div>

      <div className="flex flex-col lg:flex-row gap-0">
        {/* === SVG === */}
        <div className="lg:w-[55%] bg-slate-50 dark:bg-slate-800/40 p-4 flex items-center justify-center min-h-64">
          <svg viewBox="-10 20 470 325" width="100%" className="max-h-80">
            {/* Full graph (faint background) */}
            {phase <= 3 && [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[1,5],[2,3],[2,4],[2,5],[3,4],[3,5],[4,5]].map(([a,b]) => (
              <line key={`bg-${a}-${b}`} x1={CITIES[a].x} y1={CITIES[a].y} x2={CITIES[b].x} y2={CITIES[b].y}
                stroke="#e2e8f0" strokeWidth={1} opacity={0.5} className="dark:stroke-slate-700" />
            ))}

            {/* Main edges */}
            <AnimatePresence>
              {getAllEdges().map(({ a, b, type }, i) => (
                <motion.line
                  key={`${type}-${a}-${b}`}
                  x1={CITIES[a].x} y1={CITIES[a].y}
                  x2={CITIES[b].x} y2={CITIES[b].y}
                  stroke={edgeColor[type]}
                  strokeWidth={edgeWidth[type]}
                  strokeDasharray={edgeDash[type]}
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  transition={{ duration: 0.5, delay: i * 0.08 }}
                />
              ))}
            </AnimatePresence>

            {/* Euler sequence arrows (phase 4) */}
            {showEuler && EULER_SEQUENCE.slice(0, -1).map((v, i) => {
              const from = CITIES[v], to = CITIES[EULER_SEQUENCE[i + 1]];
              const mx = (from.x + to.x) / 2, my = (from.y + to.y) / 2;
              return (
                <motion.text key={`seq-${i}`} x={mx} y={my - 5} textAnchor="middle"
                  fontSize={10} fill="#ec4899" fontWeight="bold"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.1 + 0.3 }}
                >
                  {i + 1}
                </motion.text>
              );
            })}

            {/* Hamilton arrows (phase 5) */}
            {showHamilton && HAMILTON_SEQUENCE.slice(0, -1).map((v, i) => {
              const from = CITIES[v], to = CITIES[HAMILTON_SEQUENCE[i + 1]];
              const mx = (from.x + to.x) / 2, my = (from.y + to.y) / 2;
              return (
                <motion.text key={`hseq-${i}`} x={mx} y={my - 7} textAnchor="middle"
                  fontSize={11} fill="#059669" fontWeight="bold"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.1 + 0.3 }}
                >
                  {i + 1}
                </motion.text>
              );
            })}

            {/* City nodes */}
            {CITIES.map((c) => {
              const isOdd = showOdd && ODD_VERTICES.includes(c.id);
              const fill = showHamilton ? "#10b981" : isOdd ? "#f59e0b" : "#6366f1";
              const stroke = showHamilton ? "#059669" : isOdd ? "#d97706" : "#4f46e5";
              return (
                <g key={c.id}>
                  <motion.circle
                    cx={c.x} cy={c.y} r={20}
                    fill={fill} stroke={stroke} strokeWidth={2.5}
                    animate={{ fill, stroke }}
                    transition={{ duration: 0.4 }}
                  />
                  <text x={c.x} y={c.y + 4} textAnchor="middle" fontSize={12} fontWeight="bold" fill="white">
                    {c.label}
                  </text>
                </g>
              );
            })}

            {/* Edge cost labels (phase 3 matching) */}
            {phase === 3 && MATCHING_EDGES.map(([a, b]) => {
              const mx = (CITIES[a].x + CITIES[b].x) / 2;
              const my = (CITIES[a].y + CITIES[b].y) / 2;
              return (
                <motion.text key={`mcost-${a}-${b}`} x={mx} y={my - 8} textAnchor="middle"
                  fontSize={11} fill="#d97706" fontWeight="bold"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                >
                  {dist(a, b)}
                </motion.text>
              );
            })}

            {/* Legend */}
            <g transform="translate(8, 330)">
              {[
                { color: "#6366f1", dash: "none", label: "MST 边" },
                { color: "#f59e0b", dash: "6,3", label: "匹配边" },
                { color: "#10b981", dash: "none", label: "哈密顿路径" },
              ].filter((_, i) => {
                if (i === 0) return showMST && !showHamilton;
                if (i === 1) return showMatching && !showHamilton;
                if (i === 2) return showHamilton;
                return false;
              }).map(({ color, dash, label }, i) => (
                <g key={label} transform={`translate(${i * 110}, 0)`}>
                  <line x1={0} y1={5} x2={22} y2={5} stroke={color} strokeWidth={2.5}
                    strokeDasharray={dash === "none" ? undefined : dash} />
                  <text x={26} y={9} fontSize={10} fill="#64748b">{label}</text>
                </g>
              ))}
            </g>
          </svg>
        </div>

        {/* === Phase details === */}
        <div className="lg:w-[45%] p-5 border-t lg:border-t-0 lg:border-l border-slate-200 dark:border-slate-700 flex flex-col gap-4">
          <AnimatePresence mode="wait">
            <motion.div
              key={phase}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
              className="flex-1 flex flex-col gap-3"
            >
              <div>
                <h4 className="text-sm font-bold text-slate-800 dark:text-slate-200">{p.title}</h4>
                <p className="text-xs text-teal-600 dark:text-teal-400 mt-0.5">{p.subtitle}</p>
              </div>

              <div className="bg-teal-50 dark:bg-teal-900/30 rounded-xl p-3 border border-teal-100 dark:border-teal-800">
                <div className="text-xs font-bold text-teal-700 dark:text-teal-300 mb-1.5">💡 关键洞察</div>
                <p className="text-xs text-teal-800 dark:text-teal-200 leading-relaxed whitespace-pre-line">{p.insight}</p>
              </div>

              <div className="text-xs font-mono bg-slate-100 dark:bg-slate-800 rounded-lg px-3 py-2 text-slate-600 dark:text-slate-300">
                {p.cost}
              </div>

              {/* Cost bound display */}
              <div className="bg-gradient-to-r from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 rounded-xl p-3 border border-teal-100 dark:border-teal-800">
                <div className="text-xs font-bold text-slate-600 dark:text-slate-300 mb-2">近似比推导</div>
                {[
                  { label: "w(MST) ≤ OPT", active: phase >= 1, color: "text-indigo-600 dark:text-indigo-400" },
                  { label: "w(M) ≤ OPT/2", active: phase >= 3, color: "text-amber-600 dark:text-amber-400" },
                  { label: "Euler = MST + M ≤ 3/2·OPT", active: phase >= 4, color: "text-pink-600 dark:text-pink-400" },
                  { label: "Hamilton ≤ Euler ≤ 3/2·OPT ✓", active: phase >= 5, color: "text-emerald-600 dark:text-emerald-400 font-bold" },
                ].map(({ label, active, color }) => (
                  <div key={label} className={`text-xs font-mono flex items-center gap-2 mb-1 transition-opacity ${active ? "opacity-100" : "opacity-20"}`}>
                    <span className={active ? "text-emerald-500" : "text-slate-400"}>
                      {active ? "✓" : "○"}
                    </span>
                    <span className={active ? color : "text-slate-400 dark:text-slate-600"}>{label}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          </AnimatePresence>

          {/* Phase navigation */}
          <div className="flex gap-2">
            <button
              onClick={() => setPhase((p) => Math.max(1, p - 1))}
              disabled={phase === 1}
              className="flex-1 py-2 rounded-lg text-xs font-semibold border border-teal-200 dark:border-teal-700 text-teal-600 dark:text-teal-300 disabled:opacity-30 hover:bg-teal-50 dark:hover:bg-teal-900/30 transition-colors"
            >
              ← 上一阶段
            </button>
            <button
              onClick={() => setPhase((p) => Math.min(5, p + 1))}
              disabled={phase === 5}
              className="flex-1 py-2 rounded-lg text-xs font-semibold bg-teal-600 hover:bg-teal-700 text-white disabled:opacity-30 transition-colors"
            >
              下一阶段 →
            </button>
          </div>

          {/* Dots */}
          <div className="flex justify-center gap-2">
            {PHASES.map((ph) => (
              <button
                key={ph.id}
                onClick={() => setPhase(ph.id)}
                className={`w-2 h-2 rounded-full transition-all ${phase === ph.id ? "bg-teal-500 w-5" : "bg-slate-300 dark:bg-slate-600"}`}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
