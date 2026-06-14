"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── CPM 项目图 (Activity-On-Node) ─────────────────────────────────────────
 *  节点 = 工序，边 = 依赖关系
 *  工序编号: Start(0) A(1) B(2) C(3) D(4) E(5) End(6)
 *  工期:             0    3    6    2    5    4    0
 *  依赖: Start→A, Start→B, A→C, A→D, B→D, B→E, C→End, D→End, E→End
 *  关键路径: Start → B → D → End （工期 11 天）
 * ─────────────────────────────────────────────────────────────────────────── */
const ACTIVITIES = [
  { id: 0, x: 55,  y: 130, label: "开始", dur: 0 },
  { id: 1, x: 195, y: 72,  label: "工序A", dur: 3 },
  { id: 2, x: 195, y: 188, label: "工序B", dur: 6 },
  { id: 3, x: 340, y: 72,  label: "工序C", dur: 2 },
  { id: 4, x: 340, y: 140, label: "工序D", dur: 5 },
  { id: 5, x: 340, y: 208, label: "工序E", dur: 4 },
  { id: 6, x: 470, y: 130, label: "竣工",  dur: 0 },
];
const EDGES: { u: number; v: number }[] = [
  { u: 0, v: 1 },
  { u: 0, v: 2 },
  { u: 1, v: 3 },
  { u: 1, v: 4 },
  { u: 2, v: 4 },
  { u: 2, v: 5 },
  { u: 3, v: 6 },
  { u: 4, v: 6 },
  { u: 5, v: 6 },
];
const N = ACTIVITIES.length;
const TOPO = [0, 1, 2, 3, 4, 5, 6];
const CRITICAL_PATH = new Set([0, 2, 4, 6]); // Start, B, D, End
const CRITICAL_EDGES = new Set([1, 4, 7]);    // Start→B, B→D, D→End (edge indices)

/* ─── Pre-compute CPM forward + backward pass steps ─────────────────────── */
interface CPMStep {
  ES: (number | null)[];
  EF: (number | null)[];
  LS: (number | null)[];
  LF: (number | null)[];
  slack: (number | null)[];
  activeNode: number;     // currently computing (-1 = none)
  phase: "forward" | "backward" | "show-critical" | "done";
  phaseStep: number;      // within phase
  desc: string;
}

function buildCPMSteps(): CPMStep[] {
  const steps: CPMStep[] = [];
  const DUR = ACTIVITIES.map(a => a.dur);

  // Build adj and pred lists
  const adj: number[][] = Array.from({length: N}, ()=>[]);
  const pred: number[][] = Array.from({length: N}, ()=>[]);
  EDGES.forEach(({u,v}) => { adj[u].push(v); pred[v].push(u); });

  const ES: (number|null)[] = Array(N).fill(null);
  const EF: (number|null)[] = Array(N).fill(null);
  const LS: (number|null)[] = Array(N).fill(null);
  const LF: (number|null)[] = Array(N).fill(null);
  const slack: (number|null)[] = Array(N).fill(null);

  const snap = (activeNode: number, phase: CPMStep["phase"], phaseStep: number, desc: string): CPMStep => ({
    ES: [...ES], EF: [...EF], LS: [...LS], LF: [...LF], slack: [...slack],
    activeNode, phase, phaseStep, desc,
  });

  steps.push(snap(-1, "forward", 0,
    "CPM 关键路径计算。目标：找出总工期 + 判断哪些工序没有浮动时间（关键工序）。第一步：从左向右做「前向传播」，为每道工序计算 ES（最早开始）和 EF（最早结束）。"));

  // Forward pass
  for (let ti = 0; ti < TOPO.length; ti++) {
    const u = TOPO[ti];
    if (pred[u].length === 0) {
      ES[u] = 0;
    } else {
      ES[u] = Math.max(...pred[u].map(p => EF[p] ?? 0));
    }
    EF[u] = (ES[u] as number) + DUR[u];

    const predDesc = pred[u].length === 0
      ? "无前置工序，最早第 0 天开始。"
      : `前置工序最早结束时间：max(${pred[u].map(p=>`EF[${ACTIVITIES[p].label}]=${EF[p]}`).join(", ")}) = ${ES[u]}。`;
    steps.push(snap(u, "forward", ti+1,
      `计算「${ACTIVITIES[u].label}」：${predDesc} ES[${ACTIVITIES[u].label}]=${ES[u]}，工期 ${DUR[u]} 天，EF[${ACTIVITIES[u].label}]=${EF[u]}。`));
  }

  // Backward pass
  steps.push(snap(-1, "backward", 0,
    `前向传播完成！项目总工期 = EF[竣工] = ${EF[N-1]} 天。现在从右向左做「后向传播」，计算 LF（最晚完成）和 LS（最晚开始）。`));

  for (let ti = TOPO.length-1; ti >= 0; ti--) {
    const u = TOPO[ti];
    if (adj[u].length === 0) {
      LF[u] = EF[u];
    } else {
      LF[u] = Math.min(...adj[u].map(v => LS[v] ?? Infinity));
    }
    LS[u] = (LF[u] as number) - DUR[u];
    slack[u] = (LS[u] as number) - (ES[u] as number);

    const succDesc = adj[u].length === 0
      ? "无后继工序，LF = EF（项目终点）。"
      : `后继工序最晚开始时间：min(${adj[u].map(v=>`LS[${ACTIVITIES[v].label}]=${LS[v]}`).join(", ")}) = ${LF[u]}。`;
    steps.push(snap(u, "backward", TOPO.length-ti,
      `计算「${ACTIVITIES[u].label}」：${succDesc} LF=${LF[u]}，工期 ${DUR[u]} 天，LS=${LS[u]}，浮动时间（Float）= ${slack[u]} 天。`));
  }

  steps.push(snap(-1, "show-critical", 0,
    "后向传播完成！浮动时间（Slack = LS-ES）= 0 的工序为关键工序，串联形成关键路径：开始→工序B→工序D→竣工，共 11 天。关键路径上任何工序延误都会直接延迟整个项目！"));

  steps.push(snap(-1, "done", 0,
    "✅ CPM 分析完毕。关键路径（红色）= 开始 → 工序B(6天) → 工序D(5天) → 竣工，总工期 11 天。非关键工序（工序A/C/E）有浮动时间，可以适度灵活安排。"));

  return steps;
}

const STEPS = buildCPMSteps();

/* ─── Arrow ──────────────────────────────────────────────────────────────── */
function Arrow({ u, v, ei, step }: { u:number; v:number; ei:number; step:CPMStep }) {
  const A = ACTIVITIES;
  const [x1,y1] = [A[u].x, A[u].y];
  const [x2,y2] = [A[v].x, A[v].y];
  const dx=x2-x1, dy=y2-y1, len=Math.sqrt(dx*dx+dy*dy);
  const R=26;
  const sx=x1+(dx/len)*R, sy=y1+(dy/len)*R;
  const ex=x2-(dx/len)*R, ey=y2-(dy/len)*R;

  const showCritical = step.phase === "show-critical" || step.phase === "done";
  const isCritical = CRITICAL_EDGES.has(ei) && showCritical;
  const color = isCritical ? "#ef4444" : "#94a3b8";
  const sw = isCritical ? 2.5 : 1.5;
  const markId = `arr-cpm-${u}-${v}`;

  return (
    <g>
      <defs>
        <marker id={markId} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 Z" fill={color} />
        </marker>
      </defs>
      <line x1={sx} y1={sy} x2={ex} y2={ey}
        stroke={color} strokeWidth={sw} markerEnd={`url(#${markId})`}
        strokeDasharray={isCritical ? undefined : undefined} />
    </g>
  );
}

/* ─── Activity box ────────────────────────────────────────────────────────── */
function ActivityBox({ act, step }:{ act: typeof ACTIVITIES[0]; step: CPMStep }) {
  const { id, x, y, label, dur } = act;
  const showCritical = step.phase === "show-critical" || step.phase === "done";
  const isCritical = CRITICAL_PATH.has(id) && showCritical;
  const isActive = step.activeNode === id;
  const es = step.ES[id]; const ef = step.EF[id];
  const ls = step.LS[id]; const lf = step.LF[id];
  const sl = step.slack[id];
  const W = 52, H = 56;
  const tx = x - W/2, ty = y - H/2;

  const fillBg = isCritical ? "#fef2f2" : isActive ? "#eff6ff" : "#f8fafc";
  const stroke = isCritical ? "#ef4444" : isActive ? "#3b82f6" : "#cbd5e1";
  const sw = isCritical || isActive ? 2 : 1;

  return (
    <g>
      {isActive && (
        <rect x={tx-4} y={ty-4} width={W+8} height={H+8} rx={8}
          fill="#dbeafe44" stroke="#3b82f666" strokeWidth={1.5} />
      )}
      {isCritical && (
        <rect x={tx-2} y={ty-2} width={W+4} height={H+4} rx={7}
          fill="#fee2e244" stroke="#ef444466" strokeWidth={1.5} />
      )}
      <rect x={tx} y={ty} width={W} height={H} rx={6}
        fill={fillBg} stroke={stroke} strokeWidth={sw} />
      {/* Header */}
      <rect x={tx} y={ty} width={W} height={16} rx={6}
        fill={isCritical ? "#ef4444" : isActive ? "#3b82f6" : "#e2e8f0"} />
      <rect x={tx} y={ty+10} width={W} height={6}
        fill={isCritical ? "#ef4444" : isActive ? "#3b82f6" : "#e2e8f0"} />
      <text x={x} y={ty+11} textAnchor="middle" fontSize={9} fontWeight="bold"
        fill={isCritical || isActive ? "#fff" : "#475569"}>
        {label}({dur}d)
      </text>
      {/* ES | EF */}
      <text x={tx+4} y={ty+26} fontSize={8} fill="#64748b">ES</text>
      <text x={tx+W-4} y={ty+26} fontSize={8} fill="#64748b" textAnchor="end">EF</text>
      <text x={tx+4} y={ty+37} fontSize={9} fontWeight="bold"
        fill={isCritical ? "#dc2626" : "#1d4ed8"}>
        {es !== null ? es : "-"}
      </text>
      <text x={tx+W-4} y={ty+37} fontSize={9} fontWeight="bold" textAnchor="end"
        fill={isCritical ? "#dc2626" : "#1d4ed8"}>
        {ef !== null ? ef : "-"}
      </text>
      {/* LS | LF */}
      <text x={tx+4} y={ty+47} fontSize={8} fill="#94a3b8">LS</text>
      <text x={tx+W-4} y={ty+47} fontSize={8} fill="#94a3b8" textAnchor="end">LF</text>
      <text x={tx+4} y={ty+56} fontSize={9} fontWeight="bold"
        fill={sl === 0 ? "#dc2626" : "#64748b"}>
        {ls !== null ? ls : "-"}
      </text>
      <text x={tx+W-4} y={ty+56} fontSize={9} fontWeight="bold" textAnchor="end"
        fill={sl === 0 ? "#dc2626" : "#64748b"}>
        {lf !== null ? lf : "-"}
      </text>
      {/* Slack badge */}
      {sl !== null && (
        <>
          <rect x={x-10} y={ty+H-2} width={20} height={11} rx={4}
            fill={sl === 0 ? "#ef4444" : "#22c55e"} />
          <text x={x} y={ty+H+6} textAnchor="middle" fontSize={7} fontWeight="bold" fill="#fff">
            F={sl}
          </text>
        </>
      )}
    </g>
  );
}

export default function CriticalPathCPM() {
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1600);
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

  const phaseBg = step.phase === "backward" ? "from-orange-500 via-amber-500 to-yellow-500" :
    step.phase === "show-critical" || step.phase === "done" ? "from-red-600 via-rose-600 to-orange-600" :
    "from-amber-500 via-orange-500 to-red-500";

  const phaseLabel = step.phase === "forward" ? "前向传播（计算 ES/EF）" :
    step.phase === "backward" ? "后向传播（计算 LS/LF）" : "关键路径识别";

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className={`bg-gradient-to-r ${phaseBg} px-5 py-4 transition-all duration-500`}>
        <h3 className="text-white font-bold text-lg tracking-tight">CPM 关键路径——项目工期分析</h3>
        <p className="text-orange-100 text-sm mt-0.5">{phaseLabel}</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Phase progress tabs */}
        <div className="flex gap-2">
          {[
            { label: "① 前向传播", key: "forward" },
            { label: "② 后向传播", key: "backward" },
            { label: "③ 关键路径", key: "show-critical" },
          ].map(({label, key}) => {
            const isActive = step.phase === key || (key === "show-critical" && step.phase === "done");
            const isDone = step.phase === "backward" && key === "forward" ||
              (step.phase === "show-critical" || step.phase === "done") && (key === "forward" || key === "backward");
            return (
              <div key={key} className={`flex-1 text-center px-2 py-1.5 rounded-lg text-xs font-bold transition-all duration-300 ${
                isActive ? "bg-amber-500 text-white shadow-sm" :
                isDone ? "bg-slate-200 dark:bg-slate-700 text-slate-500 dark:text-slate-400" :
                "bg-slate-100 dark:bg-slate-800 text-slate-300 dark:text-slate-600"
              }`}>
                {label}
              </div>
            );
          })}
        </div>

        {/* Project graph */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 overflow-hidden">
          <svg viewBox="0 0 535 285" className="w-full">
            {EDGES.map((e, i) => (
              <Arrow key={i} u={e.u} v={e.v} ei={i} step={step} />
            ))}
            {ACTIVITIES.map(act => (
              <ActivityBox key={act.id} act={act} step={step} />
            ))}
            {/* Legend inside SVG */}
            <rect x={8} y={258} width={120} height={22} rx={4} fill="#f8fafc" stroke="#e2e8f0" strokeWidth={1} />
            <text x={12} y={266} fontSize={7} fill="#94a3b8">ES=最早开始  EF=最早结束</text>
            <text x={12} y={275} fontSize={7} fill="#94a3b8">LS=最晚开始  LF=最晚结束  F=浮动</text>
          </svg>
        </div>

        {/* Activity table (when backward pass or later) */}
        {(step.phase === "backward" || step.phase === "show-critical" || step.phase === "done") && (
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
            <div className="bg-slate-100 dark:bg-slate-800 px-3 py-1.5 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide">
              活动时间分析
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-slate-50 dark:bg-slate-800/50">
                    {["工序","工期","ES","EF","LS","LF","浮动F","关键?"].map(h=>(
                      <th key={h} className="px-2 py-1.5 text-slate-500 dark:text-slate-400 font-medium text-center">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-700/50">
                  {ACTIVITIES.map(act => {
                    const { id, label, dur } = act;
                    const es=step.ES[id], ef=step.EF[id], ls=step.LS[id], lf=step.LF[id], sl=step.slack[id];
                    const isCrit = sl === 0 && (step.phase === "show-critical" || step.phase === "done");
                    return (
                      <tr key={id} className={isCrit ? "bg-red-50 dark:bg-red-900/20" : ""}>
                        <td className={`px-2 py-1.5 font-bold text-center ${isCrit ? "text-red-600 dark:text-red-400" : "text-slate-600 dark:text-slate-300"}`}>{label}</td>
                        <td className="px-2 py-1.5 text-center font-mono text-slate-500 dark:text-slate-400">{dur}</td>
                        <td className="px-2 py-1.5 text-center font-mono font-bold text-blue-600 dark:text-blue-400">{es ?? "-"}</td>
                        <td className="px-2 py-1.5 text-center font-mono font-bold text-blue-600 dark:text-blue-400">{ef ?? "-"}</td>
                        <td className="px-2 py-1.5 text-center font-mono text-orange-600 dark:text-orange-400">{ls ?? "-"}</td>
                        <td className="px-2 py-1.5 text-center font-mono text-orange-600 dark:text-orange-400">{lf ?? "-"}</td>
                        <td className={`px-2 py-1.5 text-center font-bold ${sl === 0 ? "text-red-600 dark:text-red-400" : "text-emerald-600 dark:text-emerald-400"}`}>
                          {sl !== null ? sl : "-"}
                        </td>
                        <td className="px-2 py-1.5 text-center font-bold">
                          {sl !== null ? (sl === 0 ? "🔴" : "⚪") : "-"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Description */}
        <div className="rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700/50 px-4 py-2.5 text-sm text-amber-900 dark:text-amber-300 min-h-[44px]">
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
              playing ? "bg-amber-500 hover:bg-amber-600 text-white" : "bg-orange-600 hover:bg-orange-700 text-white"
            }`}>
            {playing ? "⏸ 暂停" : stepIdx === 0 ? "▶ 开始" : "▶ 播放"}
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.min(STEPS.length-1, i+1)); }} disabled={stepIdx === STEPS.length-1}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            下一步 →
          </button>
          <span className="text-xs text-slate-400 font-mono">{stepIdx+1}/{STEPS.length}</span>
          <div className="flex items-center gap-1.5 ml-auto">
            <input type="range" min={700} max={3000} step={200} value={speed}
              onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-orange-500" />
          </div>
        </div>
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
          <div className="bg-orange-500 h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${(stepIdx/(STEPS.length-1))*100}%` }} />
        </div>
      </div>
    </div>
  );
}
