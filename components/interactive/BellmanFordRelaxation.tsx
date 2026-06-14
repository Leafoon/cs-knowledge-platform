"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── CLRS 经典 5 节点示例 ────────────────────────────────────────────────
 *  s(0), t(1), x(2), y(3), z(4)
 *  CLRS 第 24 章 图 24-4 配置
 *  最终答案: d[s]=0, d[t]=2, d[x]=4, d[y]=7, d[z]=-2
 * ─────────────────────────────────────────────────────────────────────────── */
const NODES = [
  { id: 0, x: 70,  y: 130, label: "s" },
  { id: 1, x: 200, y: 68,  label: "t" },
  { id: 2, x: 340, y: 130, label: "x" },
  { id: 3, x: 200, y: 192, label: "y" },
  { id: 4, x: 460, y: 130, label: "z" },
];
// edges: (u, v, w)
const EDGES: { u: number; v: number; w: number }[] = [
  { u: 0, v: 1, w: 6 },
  { u: 0, v: 3, w: 7 },
  { u: 1, v: 2, w: 5 },
  { u: 1, v: 3, w: 8 },
  { u: 1, v: 4, w: -4 },
  { u: 2, v: 1, w: -2 },
  { u: 3, v: 2, w: -3 },
  { u: 3, v: 4, w: 9 },
  { u: 4, v: 2, w: 7 },
  { u: 4, v: 0, w: 2 },
];
const N = 5;
const INF = 1e9;

/* ─── Pre-compute Bellman-Ford rounds ────────────────────────────────────── */
interface BFRound {
  d: number[];
  prev: number[];
  changed: Set<number>;
  roundNum: number;
  desc: string;
  // per-edge detail for the active relaxations in this round
  relaxDetails: { ei: number; before: number; after: number; updated: boolean }[];
}

function buildRounds(): BFRound[] {
  const rounds: BFRound[] = [];
  const d = [0, INF, INF, INF, INF];
  const prev = Array(N).fill(-1);

  rounds.push({
    d: [...d], prev: [...prev], changed: new Set(), roundNum: 0,
    desc: "初始化：d[s]=0，其余 d[v]=+∞，前驱 π[v]=NIL。Bellman-Ford 将执行 V-1=4 轮全边松弛。",
    relaxDetails: [],
  });

  for (let r = 1; r <= N-1; r++) {
    const changed = new Set<number>();
    const relaxDetails: BFRound["relaxDetails"] = [];
    for (let ei = 0; ei < EDGES.length; ei++) {
      const { u, v, w } = EDGES[ei];
      const before = d[v];
      if (d[u] !== INF && d[u] + w < d[v]) {
        d[v] = d[u] + w;
        prev[v] = u;
        changed.add(v);
        relaxDetails.push({ ei, before, after: d[v], updated: true });
      } else {
        relaxDetails.push({ ei, before, after: d[v], updated: false });
      }
    }

    const nl = (i: number) => NODES[i].label;
    const fmtD = (v: number) => v === INF ? "∞" : String(v);
    const changedLabels = [...changed].map(i => `${nl(i)}(→${fmtD(d[i])})`).join("、");
    rounds.push({
      d: [...d], prev: [...prev], changed, roundNum: r,
      desc: `第 ${r} 轮：对所有 ${EDGES.length} 条边执行 RELAX。` +
        (changed.size > 0
          ? `更新了节点 ${changedLabels}。`
          : "本轮无任何更新，算法已提前收敛！"),
      relaxDetails,
    });
    if (changed.size === 0) break;
  }

  const fmtFinal = (v: number) => v === INF ? "∞" : String(v);
  rounds.push({
    d: [...d], prev: [...prev], changed: new Set(), roundNum: -1,
    desc: `✅ 算法完成！最终结果：d[s]=${fmtFinal(d[0])}, d[t]=${fmtFinal(d[1])}, d[x]=${fmtFinal(d[2])}, d[y]=${fmtFinal(d[3])}, d[z]=${fmtFinal(d[4])}。与 CLRS 答案一致。`,
    relaxDetails: [],
  });
  return rounds;
}

const ROUNDS = buildRounds();

/* ─── Arrow helper (with curve for tight edges) ─────────────────────────── */
function Arrow({ u, v, w, active, updated, idx }:{
  u: number; v: number; w: number;
  active: boolean; updated: boolean; idx: number;
}) {
  const p = NODES;
  const [x1,y1] = [p[u].x, p[u].y];
  const [x2,y2] = [p[v].x, p[v].y];
  const dx = x2-x1, dy = y2-y1, len = Math.sqrt(dx*dx+dy*dy);
  const R = 22;
  const sx = x1+(dx/len)*R, sy = y1+(dy/len)*R;
  const ex = x2-(dx/len)*R, ey = y2-(dy/len)*R;

  // Curve offsets for edges that visually cross
  // t↔x (idx 2 vs 5): give them a little curve
  const curveMap: Record<number, number> = { 2: 10, 5: -10 };
  const off = curveMap[idx] ?? 0;
  const perpX = (-dy/len)*off, perpY = (dx/len)*off;
  const mx = (sx+ex)/2 + perpX, my = (sy+ey)/2 + perpY;

  let strokeColor = "#94a3b8";
  let sw = 1.5;
  if (active && updated) { strokeColor = "#10b981"; sw = 2.5; }
  else if (active && !updated) { strokeColor = "#f59e0b"; sw = 2; }
  else if (w < 0) { strokeColor = "#f87171"; sw = 1.5; } // negative edge hint

  const markId = `arr-bf-${idx}`;
  const d = off !== 0
    ? `M${sx},${sy} Q${mx},${my} ${ex},${ey}`
    : undefined;

  return (
    <g>
      <defs>
        <marker id={markId} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 Z" fill={strokeColor} />
        </marker>
      </defs>
      {d
        ? <path d={d} stroke={strokeColor} strokeWidth={sw} fill="none" markerEnd={`url(#${markId})`} />
        : <line x1={sx} y1={sy} x2={ex} y2={ey} stroke={strokeColor} strokeWidth={sw} markerEnd={`url(#${markId})`} />
      }
      <text x={mx - perpX*0.3} y={my - perpY*0.3 - 7}
        textAnchor="middle" fontSize={10} fontWeight="bold" fill={strokeColor}>
        {w}
      </text>
    </g>
  );
}

const fmtD = (v: number) => v === INF ? "∞" : String(v);
const FINAL_D = [0, 2, 4, 7, -2];

export default function BellmanFordRelaxation() {
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1500);
  const [activeEdgeIdx, setActiveEdgeIdx] = useState(-1);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const round = ROUNDS[stepIdx];

  const advance = useCallback(() => {
    setStepIdx(p => { if (p >= ROUNDS.length-1) { setPlaying(false); return p; } return p+1; });
  }, []);
  useEffect(() => {
    if (playing) intervalRef.current = setInterval(advance, speed);
    else { if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; } }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, speed, advance]);

  const isDone = round.roundNum === -1;

  function nodeStyle(id: number) {
    if (isDone) return { fill: "#10b981", stroke: "#059669", text: "#fff" };
    if (round.changed.has(id)) return { fill: "#f59e0b", stroke: "#d97706", text: "#fff" };
    if (id === 0) return { fill: "#6366f1", stroke: "#4f46e5", text: "#fff" };
    if (round.d[id] !== INF) return { fill: "#3b82f6", stroke: "#2563eb", text: "#fff" };
    return { fill: "#e2e8f0", stroke: "#94a3b8", text: "#94a3b8" };
  }

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Bellman-Ford——逐轮松弛可视化</h3>
        <p className="text-violet-200 text-sm mt-0.5">CLRS 5 节点示例：V-1=4 轮全边松弛，最终 d[t]=2, d[x]=4, d[y]=7, d[z]=-2</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Round pills */}
        <div className="flex gap-1.5 flex-wrap">
          {ROUNDS.map((r, i) => (
            <button key={i} onClick={() => { setPlaying(false); setStepIdx(i); }}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all duration-200 ${
                i === stepIdx
                  ? isDone && i === stepIdx
                    ? "bg-emerald-500 text-white scale-110 shadow-sm"
                    : "bg-violet-600 text-white scale-110 shadow-sm"
                  : i < stepIdx
                    ? "bg-slate-200 dark:bg-slate-700 text-slate-500 dark:text-slate-400"
                    : "bg-slate-100 dark:bg-slate-800 text-slate-300 dark:text-slate-600"
              }`}>
              {r.roundNum === 0 ? "初始" : r.roundNum === -1 ? "✅完成" : `第 ${r.roundNum} 轮`}
            </button>
          ))}
        </div>

        <div className="flex gap-4 items-start">
          {/* SVG */}
          <div className="flex-1 min-w-0 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 overflow-hidden">
            <svg viewBox="0 0 535 265" className="w-full">
              {EDGES.map((e, i) => (
                <Arrow key={i} {...e} idx={i}
                  active={activeEdgeIdx === i}
                  updated={round.relaxDetails.find(r=>r.ei===i)?.updated ?? false} />
              ))}
              {NODES.map(node => {
                const s = nodeStyle(node.id);
                const changed = round.changed.has(node.id);
                const d = round.d[node.id];
                const isFinal = isDone || (d === FINAL_D[node.id]);
                return (
                  <g key={node.id}>
                    {changed && <circle cx={node.x} cy={node.y} r={30}
                      fill="#fde68a33" stroke="#f59e0b55" strokeWidth={2} />}
                    <circle cx={node.x} cy={node.y} r={22}
                      fill={s.fill} stroke={s.stroke} strokeWidth={2}
                      className="transition-all duration-400" />
                    <text x={node.x} y={node.y-4} textAnchor="middle" fontSize={13} fontWeight="bold" fill={s.text}>
                      {node.label}
                    </text>
                    <text x={node.x} y={node.y+9} textAnchor="middle" fontSize={10} fill={s.text} opacity={0.9}>
                      {fmtD(d)}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Right: d[] table across rounds */}
          <div className="w-52 shrink-0 space-y-3">
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-slate-100 dark:bg-slate-800 px-3 py-1.5 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                d[] 轮次对比
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-[11px]">
                  <thead>
                    <tr className="bg-slate-50 dark:bg-slate-800/50">
                      <th className="px-2 py-1 text-left text-slate-400 font-medium">节点</th>
                      {ROUNDS.filter(r=>r.roundNum>=0).map(r=>(
                        <th key={r.roundNum} className="px-1.5 py-1 text-slate-400 font-medium">
                          {r.roundNum===0?"初":r.roundNum}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-700/50">
                    {NODES.map(node => (
                      <tr key={node.id}>
                        <td className="px-2 py-1 font-bold text-slate-600 dark:text-slate-300">{node.label}</td>
                        {ROUNDS.filter(r=>r.roundNum>=0).map((r, ri) => {
                          const val = r.d[node.id];
                          const changed = ri > 0 && r.changed.has(node.id);
                          const isCurrent = ri === (stepIdx === ROUNDS.length-1 ? stepIdx-2 : stepIdx);
                          return (
                            <td key={ri} className={`px-1.5 py-1 text-center font-mono font-bold ${
                              changed ? "text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20" :
                              val === INF ? "text-slate-300 dark:text-slate-600" :
                              "text-blue-600 dark:text-blue-400"
                            } ${isCurrent && stepIdx > 0 ? "ring-1 ring-violet-400 rounded" : ""}`}>
                              {fmtD(val)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Legend */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 p-3 space-y-1.5">
              {[
                { color: "bg-indigo-500", label: "源点 s" },
                { color: "bg-blue-500", label: "已可达" },
                { color: "bg-amber-400", label: "本轮更新" },
                { color: "bg-emerald-500", label: "已收敛" },
                { color: "bg-slate-200 dark:bg-slate-700", label: "不可达 (∞)" },
              ].map(({ color, label }) => (
                <div key={label} className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
                  <div className={`w-3 h-3 rounded-full ${color} shrink-0`} />
                  {label}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Relax details for current round */}
        {round.relaxDetails.length > 0 && (
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
            <div className="bg-slate-100 dark:bg-slate-800 px-3 py-1.5 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide">
              本轮松弛详情
            </div>
            <div className="flex flex-wrap gap-1.5 p-2">
              {round.relaxDetails.map((rd, i) => {
                const e = EDGES[rd.ei];
                return (
                  <div key={i} className={`flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-mono ${
                    rd.updated
                      ? "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 border border-amber-300 dark:border-amber-600"
                      : "bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-500 border border-slate-200 dark:border-slate-700"
                  }`}>
                    <span>{NODES[e.u].label}→{NODES[e.v].label}</span>
                    <span className="opacity-60 mx-0.5">w={e.w}</span>
                    {rd.updated ? <span className="text-emerald-600 dark:text-emerald-400 font-bold">✓{fmtD(rd.after)}</span>
                      : <span className="opacity-40">-</span>}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Description */}
        <div className="rounded-xl bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-700/50 px-4 py-2.5 text-sm text-violet-800 dark:text-violet-300 min-h-[44px]">
          {round.desc}
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={() => { setPlaying(false); setStepIdx(0); setActiveEdgeIdx(-1); }}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            ↺ 重置
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.max(0, i-1)); }} disabled={stepIdx === 0}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            ← 上一轮
          </button>
          <button onClick={() => setPlaying(p => !p)} disabled={stepIdx === ROUNDS.length-1}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-colors ${
              playing ? "bg-amber-500 hover:bg-amber-600 text-white" : "bg-violet-600 hover:bg-violet-700 text-white"
            }`}>
            {playing ? "⏸ 暂停" : stepIdx === 0 ? "▶ 开始" : "▶ 播放"}
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.min(ROUNDS.length-1, i+1)); }} disabled={stepIdx === ROUNDS.length-1}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            下一轮 →
          </button>
          <span className="text-xs text-slate-400 font-mono">{stepIdx+1}/{ROUNDS.length}</span>
          <div className="flex items-center gap-1.5 ml-auto">
            <input type="range" min={700} max={2500} step={200} value={speed}
              onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-violet-500" />
          </div>
        </div>
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
          <div className="bg-violet-500 h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${(stepIdx/(ROUNDS.length-1))*100}%` }} />
        </div>
      </div>
    </div>
  );
}
