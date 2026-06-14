"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── Graph ──────────────────────────────────────────────────────────────────
 *  S(0)→A(1) w=2
 *  A(1)→B(2) w=1
 *  B(2)→C(3) w=-5
 *  C(3)→A(1) w=2
 *  负权环：A→B→C→A，总权值 1+(-5)+2 = -2 < 0
 * ─────────────────────────────────────────────────────────────────────────── */
const NODES = [
  { id: 0, x: 65,  y: 130, label: "S", sublabel: "0" },
  { id: 1, x: 200, y: 60,  label: "A", sublabel: "1" },
  { id: 2, x: 360, y: 60,  label: "B", sublabel: "2" },
  { id: 3, x: 270, y: 185, label: "C", sublabel: "3" },
];
const EDGES: { u: number; v: number; w: number }[] = [
  { u: 0, v: 1, w: 2 },
  { u: 1, v: 2, w: 1 },
  { u: 2, v: 3, w: -5 },
  { u: 3, v: 1, w: 2 },
];
const N = 4;
const INF = 1e9;

/* ─── Pre-compute rounds ─────────────────────────────────────────────────── */
interface NegCycleStep {
  d: number[];
  changed: boolean[];      // which nodes' d[] changed this round
  roundNum: number;        // 1..N
  isCycleDetected: boolean;
  cycleNodes: number[];    // nodes that triggered detection
  desc: string;
}

function buildSteps(): NegCycleStep[] {
  const steps: NegCycleStep[] = [];

  // Initial
  steps.push({
    d: [0, INF, INF, INF], changed: Array(N).fill(false),
    roundNum: 0, isCycleDetected: false, cycleNodes: [],
    desc: `初始化：d[S]=0，其余节点 d=+∞。Bellman-Ford 将执行 V-1=${N-1} 轮松弛，然后用第 V=${N} 轮检测负权环。`,
  });

  const d = [0, INF, INF, INF];

  for (let round = 1; round <= N; round++) {
    const dPrev = [...d];
    const changed = Array(N).fill(false);

    for (const { u, v, w } of EDGES) {
      if (d[u] !== INF && d[u] + w < d[v]) {
        d[v] = d[u] + w;
        changed[v] = true;
      }
    }

    const isCycleDetected = round === N && changed.some(c => c);
    const cycleNodes = isCycleDetected ? changed.map((c, i) => c ? i : -1).filter(i => i >= 0) : [];

    const fmtD = (v: number) => v === INF ? "∞" : String(v);
    const changedNames = changed.map((c, i) => c ? NODES[i].label : "").filter(Boolean);
    let desc = "";
    if (round < N) {
      desc = `第 ${round} 轮（共 V-1=${N-1} 轮）：对所有边执行 RELAX。` +
        (changedNames.length > 0
          ? `节点 ${changedNames.join("、")} 的距离被更新。d = [${d.map(fmtD).join(", ")}]`
          : "本轮无更新，已提前收敛。");
    } else {
      desc = isCycleDetected
        ? `🚨 第 ${round} 轮（= V 轮，检测轮）：节点 ${cycleNodes.map(i=>NODES[i].label).join("、")} 的 d 值仍能继续减小！说明图中存在从源点可达的负权环 A→B→C→A（环权=-2）。最短路无意义！`
        : `第 ${round} 轮（= V 轮）：本轮无任何更新，负权环不存在（或不可达）。`;
    }

    steps.push({ d: [...d], changed, roundNum: round, isCycleDetected, cycleNodes, desc });
  }

  return steps;
}

const STEPS = buildSteps();

/* ─── Arrow helper ───────────────────────────────────────────────────────── */
function Arrow({ u, v, w, isCycle, isDetect, isCycleRound }:{
  u:number; v:number; w:number; isCycle:boolean; isDetect:boolean; isCycleRound:boolean;
}) {
  const [x1,y1] = [NODES[u].x, NODES[u].y];
  const [x2,y2] = [NODES[v].x, NODES[v].y];
  const dx = x2-x1, dy = y2-y1, len = Math.sqrt(dx*dx+dy*dy);
  const R = 22;
  const sx = x1+(dx/len)*R, sy = y1+(dy/len)*R;
  const ex = x2-(dx/len)*R, ey = y2-(dy/len)*R;
  const mid = { x: (sx+ex)/2, y: (sy+ey)/2 };

  let color = "#94a3b8";
  let sw = 1.6;
  if (isCycle && isDetect) { color = "#ef4444"; sw = 2.5; }
  else if (isCycle && isCycleRound) { color = "#f97316"; sw = 2; }
  else if (isCycle) { color = "#fb923c"; sw = 1.8; }

  const markId = `ncdet-${u}-${v}`;
  return (
    <g>
      <defs>
        <marker id={markId} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 Z" fill={color} />
        </marker>
      </defs>
      <line x1={sx} y1={sy} x2={ex} y2={ey}
        stroke={color} strokeWidth={sw} markerEnd={`url(#${markId})`} />
      <text x={mid.x - dy/len*12} y={mid.y + dx/len*12}
        textAnchor="middle" dominantBaseline="central"
        fontSize={11} fontWeight="bold" fill={color}>{w}</text>
    </g>
  );
}

const INF_DISP = "∞";
const fmtD = (v: number) => v === INF ? INF_DISP : String(v);

/* ─── Main component ─────────────────────────────────────────────────────── */
export default function NegativeCycleDetect() {
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1400);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const step = STEPS[stepIdx];

  const advance = useCallback(() => {
    setStepIdx(p => { if (p >= STEPS.length-1) { setPlaying(false); return p; } return p+1; });
  }, []);
  useEffect(() => {
    if (playing) intervalRef.current = setInterval(advance, speed);
    else { if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; } }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, speed, advance]);

  const cycleEdges = new Set([1, 2, 3]); // A→B, B→C, C→A
  const isCycleRound = step.roundNum === N;

  function nodeStyle(id: number) {
    if (step.isCycleDetected && step.cycleNodes.includes(id))
      return "bg-red-500 border-red-600 text-white shadow-red-400/40 shadow-lg";
    if (step.changed[id])
      return "bg-amber-400 border-amber-500 text-white";
    if (step.d[id] === INF)
      return "bg-slate-100 dark:bg-slate-700 border-slate-300 dark:border-slate-500 text-slate-400 dark:text-slate-400";
    return "bg-blue-500 border-blue-600 text-white";
  }

  // collect round history for table
  const history = STEPS.slice(1, stepIdx + 1);

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className={`px-5 py-4 transition-colors duration-500 ${
        step.isCycleDetected
          ? "bg-gradient-to-r from-red-600 via-rose-600 to-pink-600"
          : "bg-gradient-to-r from-rose-500 via-pink-600 to-purple-600"
      }`}>
        <h3 className="text-white font-bold text-lg tracking-tight">负权环检测——第 V 轮仍能松弛</h3>
        <p className="text-rose-100 text-sm mt-0.5">负权环 A→B→C→A（总权 = 1+(-5)+2 = -2）会让距离无限减小</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Round indicator */}
        <div className="flex gap-2 items-center">
          <span className="text-xs text-slate-500 dark:text-slate-400 shrink-0">轮次：</span>
          <div className="flex gap-1.5">
            {Array.from({ length: N }, (_, i) => i+1).map(r => (
              <div key={r} className={`px-2.5 py-1 rounded-lg text-xs font-bold transition-all duration-300 ${
                r === step.roundNum
                  ? r === N
                    ? "bg-red-500 text-white scale-110 shadow-sm ring-2 ring-red-300"
                    : "bg-violet-600 text-white scale-110 shadow-sm"
                  : r < step.roundNum
                    ? "bg-slate-200 dark:bg-slate-700 text-slate-500"
                    : "bg-slate-100 dark:bg-slate-800 text-slate-300 dark:text-slate-600"
              }`}>
                {r < N ? `轮次 ${r}` : `轮次 ${r} 🔍`}
              </div>
            ))}
          </div>
        </div>

        <div className="flex gap-4 items-start">
          {/* Graph */}
          <div className="flex-1 min-w-0 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 overflow-hidden">
            <svg viewBox="0 0 430 250" className="w-full">
              {/* Cycle highlight ring */}
              {isCycleRound && (
                <ellipse cx={280} cy={120} rx={120} ry={80}
                  fill="none" stroke="#ef4444" strokeWidth={2} strokeDasharray="6 4" opacity={0.5} />
              )}
              {EDGES.map((e, i) => (
                <Arrow key={i} {...e}
                  isCycle={cycleEdges.has(i)}
                  isDetect={step.isCycleDetected}
                  isCycleRound={isCycleRound} />
              ))}
              {NODES.map(node => {
                const changed = step.changed[node.id];
                const detected = step.isCycleDetected && step.cycleNodes.includes(node.id);
                const fill = detected ? "#ef4444" : changed ? "#f59e0b" :
                  step.d[node.id] === INF ? "#e2e8f0" : "#3b82f6";
                const stroke = detected ? "#dc2626" : changed ? "#d97706" :
                  step.d[node.id] === INF ? "#cbd5e1" : "#2563eb";
                const textFill = (step.d[node.id] === INF && !changed && !detected) ? "#94a3b8" : "#fff";

                return (
                  <g key={node.id}>
                    {(changed || detected) && (
                      <circle cx={node.x} cy={node.y} r={32}
                        fill={detected ? "#fecaca44" : "#fde68a44"}
                        stroke={detected ? "#ef444466" : "#f59e0b66"} strokeWidth={2} />
                    )}
                    <circle cx={node.x} cy={node.y} r={22}
                      fill={fill} stroke={stroke} strokeWidth={2}
                      className="transition-all duration-400" />
                    <text x={node.x} y={node.y - 4} textAnchor="middle"
                      fontSize={13} fontWeight="bold" fill={textFill}>{node.label}</text>
                    <text x={node.x} y={node.y + 9} textAnchor="middle"
                      fontSize={10} fill={textFill} opacity={0.9}>{fmtD(step.d[node.id])}</text>
                  </g>
                );
              })}

              {/* Cycle label */}
              {isCycleRound && (
                <text x={280} y={130} textAnchor="middle" fontSize={9}
                  fill={step.isCycleDetected ? "#ef4444" : "#94a3b8"} fontWeight="bold">
                  A→B→C→A = -2
                </text>
              )}
            </svg>
          </div>

          {/* Round-by-round table */}
          <div className="w-44 shrink-0 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
            <div className="bg-slate-100 dark:bg-slate-800 px-3 py-1.5 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide">
              d[] 各轮变化
            </div>
            <div className="overflow-auto max-h-40">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-slate-50 dark:bg-slate-800/50">
                    <th className="px-2 py-1 text-left text-slate-400 font-medium">轮</th>
                    {NODES.map(n => (
                      <th key={n.id} className="px-2 py-1 text-slate-400 font-medium">{n.label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-700/50">
                  {STEPS.map((s, si) => {
                    if (si > stepIdx) return null;
                    const rowBg = s.isCycleDetected ? "bg-red-50 dark:bg-red-900/20" :
                      si === stepIdx && si > 0 ? "bg-violet-50 dark:bg-violet-900/10" : "";
                    return (
                      <tr key={si} className={rowBg}>
                        <td className="px-2 py-1 font-mono font-bold text-slate-500">
                          {si === 0 ? "初始" : si === N ? `V🔍` : `R${si}`}
                        </td>
                        {NODES.map(n => (
                          <td key={n.id} className={`px-2 py-1 text-center font-mono font-bold ${
                            s.changed[n.id]
                              ? s.isCycleDetected ? "text-red-600 dark:text-red-400" : "text-amber-600 dark:text-amber-400"
                              : s.d[n.id] === INF ? "text-slate-300 dark:text-slate-600" : "text-blue-600 dark:text-blue-400"
                          }`}>
                            {fmtD(s.d[n.id])}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {/* Legend */}
            <div className="px-3 py-2 border-t border-slate-100 dark:border-slate-700/50 space-y-0.5">
              <div className="flex items-center gap-1.5 text-[10px] text-slate-500">
                <div className="w-2 h-2 rounded-full bg-amber-400" />更新（正常）
              </div>
              <div className="flex items-center gap-1.5 text-[10px] text-red-500">
                <div className="w-2 h-2 rounded-full bg-red-500" />更新（第V轮 = 负权环！）
              </div>
            </div>
          </div>
        </div>

        {/* Alert box when cycle detected */}
        {step.isCycleDetected && (
          <div className="rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-300 dark:border-red-600 px-4 py-3 flex items-start gap-3">
            <span className="text-2xl shrink-0">🚨</span>
            <div>
              <p className="font-bold text-red-700 dark:text-red-400 text-sm">负权环已检测到！</p>
              <p className="text-red-600 dark:text-red-300 text-xs mt-0.5">
                第 {N} 轮（= V 轮）仍能松弛节点 {step.cycleNodes.map(i => NODES[i].label).join("、")}，
                证明存在可达的负权环（A→B→C→A，权和=-2）。此时 d[] 无法收敛，最短路径无意义。
              </p>
            </div>
          </div>
        )}

        {/* Description */}
        <div className={`rounded-xl border px-4 py-2.5 text-sm min-h-[44px] transition-colors duration-300 ${
          step.isCycleDetected
            ? "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-700/50 text-red-800 dark:text-red-300"
            : "bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-700/50 text-purple-800 dark:text-purple-300"
        }`}>
          {step.desc}
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={() => { setPlaying(false); setStepIdx(0); }}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            ↺ 重置
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.max(0, i-1)); }} disabled={stepIdx === 0}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            ← 上一步
          </button>
          <button onClick={() => setPlaying(p => !p)} disabled={stepIdx === STEPS.length-1}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-colors ${
              playing ? "bg-amber-500 hover:bg-amber-600 text-white" : "bg-rose-600 hover:bg-rose-700 text-white"
            }`}>
            {playing ? "⏸ 暂停" : stepIdx === 0 ? "▶ 开始" : "▶ 播放"}
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.min(STEPS.length-1, i+1)); }} disabled={stepIdx === STEPS.length-1}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            下一步 →
          </button>
          <span className="text-xs text-slate-400 font-mono">{stepIdx+1}/{STEPS.length}</span>
          <div className="flex items-center gap-1.5 ml-auto">
            <input type="range" min={700} max={2200} step={200} value={speed}
              onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-rose-500" />
          </div>
        </div>
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
          <div className={`h-1.5 rounded-full transition-all duration-300 ${step.isCycleDetected ? "bg-red-500" : "bg-rose-500"}`}
            style={{ width: `${(stepIdx/(STEPS.length-1))*100}%` }} />
        </div>
      </div>
    </div>
  );
}
