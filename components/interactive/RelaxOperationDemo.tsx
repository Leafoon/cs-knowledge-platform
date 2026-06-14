"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── Graph definition ───────────────────────────────────────────────────────
 *  S(0) → A(1) w=6
 *  S(0) → B(2) w=7
 *  A(1) → B(2) w=-2
 *  A(1) → C(3) w=5
 *  B(2) → C(3) w=3
 * ─────────────────────────────────────────────────────────────────────────── */
const NODES = [
  { id: 0, x: 80,  y: 120, label: "S" },
  { id: 1, x: 220, y: 55,  label: "A" },
  { id: 2, x: 220, y: 185, label: "B" },
  { id: 3, x: 360, y: 120, label: "C" },
];
const EDGES: { u: number; v: number; w: number }[] = [
  { u: 0, v: 1, w: 6 },
  { u: 0, v: 2, w: 7 },
  { u: 1, v: 2, w: -2 },
  { u: 1, v: 3, w: 5 },
  { u: 2, v: 3, w: 3 },
];
const INF = Infinity;

/* ─── Pre-computed steps ─────────────────────────────────────────────────── */
interface RelaxStep {
  d: number[];           // d[v] snapshot
  prev: number[];        // prev[v]
  activeEdge: number;    // index into EDGES, -1 = none
  updated: boolean;      // did RELAX change d[v]?
  dBefore: number;       // d[v] before this relax call
  dNew: number;          // d[u] + w
  desc: string;
}

function buildSteps(): RelaxStep[] {
  const steps: RelaxStep[] = [];
  const d = [0, INF, INF, INF];
  const prev = [-1, -1, -1, -1];

  function snap(ae: number, updated: boolean, dBefore: number, dNew: number, desc: string) {
    steps.push({ d: [...d], prev: [...prev], activeEdge: ae, updated, dBefore, dNew, desc });
  }

  // Initial state
  snap(-1, false, INF, INF,
    "初始化：d[S]=0，其余节点 d[v]=+∞，前驱 prev[v]=NIL。接下来我们逐条对边执行 RELAX 操作。");

  const relaxOps: [number, string][] = [
    [0, "对边 (S→A, w=6) 执行 RELAX"],
    [1, "对边 (S→B, w=7) 执行 RELAX"],
    [2, "对边 (A→B, w=−2) 执行 RELAX"],
    [3, "对边 (A→C, w=5) 执行 RELAX"],
    [4, "对边 (B→C, w=3) 执行 RELAX"],
    [2, "再次对边 (A→B, w=−2) 执行 RELAX（演示已收敛时无更新）"],
  ];

  for (const [ei, prefix] of relaxOps) {
    const { u, v, w } = EDGES[ei];
    const dBefore = d[v];
    const candidate = d[u] === INF ? INF : d[u] + w;
    const updated = candidate < d[v];
    if (updated) {
      d[v] = candidate;
      prev[v] = u;
    }
    const nodeU = NODES[u].label, nodeV = NODES[v].label;
    let outcome = "";
    if (d[u] === INF && !updated) {
      outcome = `d[${nodeU}]=∞，来源不可达，跳过。`;
    } else if (updated) {
      outcome = `✅ d[${nodeV}] 从 ${dBefore === INF ? "∞" : dBefore} 更新为 ${d[v]}，松弛成功！`;
    } else {
      outcome = `❌ d[${nodeU}]+${w}=${candidate} ≥ d[${nodeV}]=${dBefore}，无需更新。`;
    }
    snap(ei, updated, dBefore, candidate, `${prefix}：判断 d[${nodeU}](${d[u]})+${w} vs d[${nodeV}](${dBefore})。${outcome}`);
  }

  snap(-1, false, INF, INF,
    "✅ 所有边松弛完毕！d[S]=0, d[A]=6, d[B]=4, d[C]=7。这就是 Bellman-Ford 一轮松弛的全貌。");
  return steps;
}

const STEPS = buildSteps();

/* ─── SVG helpers ────────────────────────────────────────────────────────── */
function EdgeArrow({ u, v, w, active, updated, edgeIdx }:{
  u:number; v:number; w:number; active:boolean; updated:boolean; edgeIdx:number;
}) {
  const [x1,y1] = [NODES[u].x, NODES[u].y];
  const [x2,y2] = [NODES[v].x, NODES[v].y];
  const dx = x2-x1, dy = y2-y1, len = Math.sqrt(dx*dx+dy*dy);
  const R = 24;
  const sx = x1+(dx/len)*R, sy = y1+(dy/len)*R;
  const ex = x2-(dx/len)*R, ey = y2-(dy/len)*R;
  const mx = (sx+ex)/2, my = (sy+ey)/2;
  // Curve offset for S→B and A→B to avoid overlap
  const offset = edgeIdx === 1 ? 12 : edgeIdx === 2 ? -10 : 0;
  const perpX = -dy/len * offset, perpY = dx/len * offset;
  const color = active ? (updated ? "#10b981" : "#ef4444") : "#94a3b8";
  const strokeW = active ? 2.5 : 1.5;

  if (offset !== 0) {
    const cpx = mx + perpX, cpy = my + perpY;
    return (
      <g>
        <defs>
          <marker id={`arr-relax-${edgeIdx}`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 Z" fill={color} />
          </marker>
        </defs>
        <path d={`M${sx},${sy} Q${cpx},${cpy} ${ex},${ey}`}
          stroke={color} strokeWidth={strokeW} fill="none"
          markerEnd={`url(#arr-relax-${edgeIdx})`} />
        <text x={cpx + perpX*0.4} y={cpy + perpY*0.4}
          textAnchor="middle" dominantBaseline="central"
          fontSize={11} fontWeight="bold" fill={color}>{w}</text>
      </g>
    );
  }
  return (
    <g>
      <defs>
        <marker id={`arr-relax-${edgeIdx}`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 Z" fill={color} />
        </marker>
      </defs>
      <line x1={sx} y1={sy} x2={ex} y2={ey}
        stroke={color} strokeWidth={strokeW} markerEnd={`url(#arr-relax-${edgeIdx})`} />
      <text x={mx + perpX} y={my + perpY - 8}
        textAnchor="middle" fontSize={11} fontWeight="bold" fill={color}>{w}</text>
    </g>
  );
}

/* ─── Main component ─────────────────────────────────────────────────────── */
export default function RelaxOperationDemo() {
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1200);
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

  const ae = step.activeEdge;
  const activeEdge = ae >= 0 ? EDGES[ae] : null;

  function nodeStyle(id: number) {
    if (activeEdge && activeEdge.u === id)
      return { fill: "#6366f1", stroke: "#4f46e5", text: "#fff", ring: true };
    if (activeEdge && activeEdge.v === id)
      return { fill: step.updated ? "#10b981" : "#ef4444", stroke: step.updated ? "#059669" : "#dc2626", text: "#fff", ring: false };
    if (step.prev[id] >= 0 || id === 0)
      return { fill: "#3b82f6", stroke: "#2563eb", text: "#fff", ring: false };
    return { fill: "#f1f5f9", stroke: "#cbd5e1", text: "#475569", ring: false };
  }

  const fmtD = (v: number) => v === INF ? "∞" : String(v);

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-sky-500 via-blue-600 to-indigo-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">RELAX 操作——最短路「松弛」动画</h3>
        <p className="text-sky-100 text-sm mt-0.5">一行判断 + 一次赋值，驱动所有最短路算法</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Pseudocode strip */}
        <div className="rounded-xl bg-slate-800 dark:bg-slate-950 px-4 py-3 font-mono text-sm text-slate-200">
          <span className="text-slate-400">RELAX(u, v, w): </span>
          <span className={ae >= 0 && step.dNew < (step.dBefore === INF ? Infinity : step.dBefore)
            ? "text-emerald-400 font-bold" : "text-slate-300"}>
            if d[v] &gt; d[u] + w(u,v):&nbsp;
          </span>
          <span className={step.updated ? "text-amber-300 font-bold" : "text-slate-500"}>
            d[v] = d[u] + w(u,v);  π[v] = u
          </span>
        </div>

        <div className="flex gap-4 items-start">
          {/* SVG Graph */}
          <div className="flex-1 min-w-0 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 overflow-hidden">
            <svg viewBox="0 0 440 240" className="w-full">
              {EDGES.map((e, i) => (
                <EdgeArrow key={i} {...e} edgeIdx={i}
                  active={ae === i} updated={step.updated} />
              ))}
              {NODES.map(node => {
                const s = nodeStyle(node.id);
                return (
                  <g key={node.id}>
                    {s.ring && <circle cx={node.x} cy={node.y} r={32} fill="#6366f122" stroke="#6366f155" strokeWidth={2} />}
                    <circle cx={node.x} cy={node.y} r={24} fill={s.fill} stroke={s.stroke}
                      strokeWidth={2} className="transition-all duration-300" />
                    <text x={node.x} y={node.y - 4} textAnchor="middle" fontSize={13} fontWeight="bold" fill={s.text}>
                      {node.label}
                    </text>
                    <text x={node.x} y={node.y + 10} textAnchor="middle" fontSize={10} fill={s.text} opacity={0.85}>
                      d={fmtD(step.d[node.id])}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Right panel */}
          <div className="w-44 space-y-3 shrink-0">
            {/* RELAX detail */}
            {ae >= 0 && (
              <div className={`rounded-xl border px-3 py-2.5 text-xs space-y-1.5 transition-all duration-300 ${
                step.updated
                  ? "border-emerald-300 dark:border-emerald-600 bg-emerald-50 dark:bg-emerald-900/20"
                  : "border-rose-300 dark:border-rose-600 bg-rose-50 dark:bg-rose-900/20"
              }`}>
                <div className="font-bold text-slate-700 dark:text-slate-200 text-[11px] uppercase tracking-wide">
                  本次 RELAX
                </div>
                <div className="space-y-1 font-mono">
                  <div className="flex justify-between">
                    <span className="text-slate-500">d[{NODES[EDGES[ae].u].label}]</span>
                    <span className="font-bold text-indigo-600 dark:text-indigo-400">{fmtD(step.d[EDGES[ae].u])}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">w(e)</span>
                    <span className="font-bold text-slate-700 dark:text-slate-300">{EDGES[ae].w}</span>
                  </div>
                  <div className="border-t border-dashed border-slate-300 dark:border-slate-600 pt-1 flex justify-between">
                    <span className="text-slate-500">候选值</span>
                    <span className="font-bold text-amber-600 dark:text-amber-400">{fmtD(step.dNew)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">d[{NODES[EDGES[ae].v].label}] 旧</span>
                    <span className="font-bold text-slate-500">{fmtD(step.dBefore)}</span>
                  </div>
                  <div className={`text-center mt-1 font-bold rounded py-0.5 ${
                    step.updated
                      ? "bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300"
                      : "bg-rose-100 dark:bg-rose-900/40 text-rose-600 dark:text-rose-400"
                  }`}>
                    {step.updated ? "✅ 更新！" : "❌ 不更新"}
                  </div>
                </div>
              </div>
            )}

            {/* d[] table */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-slate-100 dark:bg-slate-800 px-3 py-1.5 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                距离表 d[]
              </div>
              <div className="divide-y divide-slate-100 dark:divide-slate-700/50">
                {NODES.map(node => {
                  const isU = ae >= 0 && EDGES[ae].u === node.id;
                  const isV = ae >= 0 && EDGES[ae].v === node.id;
                  return (
                    <div key={node.id} className={`flex justify-between items-center px-3 py-1.5 text-xs transition-colors ${
                      isU ? "bg-indigo-50 dark:bg-indigo-900/20" :
                      isV && step.updated ? "bg-emerald-50 dark:bg-emerald-900/20" :
                      isV ? "bg-rose-50 dark:bg-rose-900/20" : ""
                    }`}>
                      <span className={`font-bold ${isU ? "text-indigo-600 dark:text-indigo-400" : "text-slate-600 dark:text-slate-300"}`}>
                        d[{node.label}]
                      </span>
                      <span className={`font-mono font-bold text-[13px] ${
                        isV && step.updated ? "text-emerald-600 dark:text-emerald-400" :
                        step.d[node.id] === INF ? "text-slate-400" : "text-blue-600 dark:text-blue-400"
                      }`}>
                        {fmtD(step.d[node.id])}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Description */}
        <div className="rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700/50 px-4 py-2.5 text-sm text-blue-800 dark:text-blue-300 min-h-[44px]">
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
              playing ? "bg-amber-500 hover:bg-amber-600 text-white" : "bg-blue-600 hover:bg-blue-700 text-white"
            }`}>
            {playing ? "⏸ 暂停" : stepIdx === 0 ? "▶ 开始" : "▶ 播放"}
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.min(STEPS.length-1, i+1)); }} disabled={stepIdx === STEPS.length-1}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            下一步 →
          </button>
          <span className="text-xs text-slate-400 font-mono">{stepIdx+1}/{STEPS.length}</span>
          <div className="flex items-center gap-1.5 ml-auto">
            <span className="text-[10px] text-slate-400">速度</span>
            <input type="range" min={600} max={2000} step={200} value={speed}
              onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-blue-500" />
          </div>
        </div>
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
          <div className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${(stepIdx/(STEPS.length-1))*100}%` }} />
        </div>
      </div>
    </div>
  );
}
