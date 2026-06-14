"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── DAG: r(0), s(1), x(2), y(3), z(4) ─────────────────────────────────
 *  拓扑序: r → s → x → y → z
 *  边: r→s(5), r→x(3), s→x(2), s→y(6), x→y(7), x→z(4), y→z(-1)
 *  源点: s(1)    最终结果: r=∞, s=0, x=2, y=6, z=5
 * ─────────────────────────────────────────────────────────────────────────── */
const NODES = [
  { id: 0, x: 65,  y: 130, label: "r" },
  { id: 1, x: 185, y: 65,  label: "s" },
  { id: 2, x: 305, y: 130, label: "x" },
  { id: 3, x: 430, y: 65,  label: "y" },
  { id: 4, x: 430, y: 195, label: "z" },
];
const EDGES: { u: number; v: number; w: number }[] = [
  { u: 0, v: 1, w: 5 },
  { u: 0, v: 2, w: 3 },
  { u: 1, v: 2, w: 2 },
  { u: 1, v: 3, w: 6 },
  { u: 2, v: 3, w: 7 },
  { u: 2, v: 4, w: 4 },
  { u: 3, v: 4, w: -1 },
];
const TOPO_ORDER = [0, 1, 2, 3, 4]; // r, s, x, y, z
const N = 5;
const SRC = 1; // s
const INF = 1e9;

/* ─── Pre-compute steps ──────────────────────────────────────────────────── */
interface DAGStep {
  d: number[];
  prev: number[];
  processedUpTo: number;   // index in TOPO_ORDER: all nodes 0..processedUpTo have been processed
  activeNode: number;      // node currently being processed (-1 = initial/done)
  relaxed: { edgeIdx: number; before: number; after: number; updated: boolean }[];
  relaxedEdges: Set<number>;  // set of edge indices
  updatedEdges: Set<number>;
  successFullyRelaxed: Set<number>; // indices of edges that produced updates (accumulated)
  desc: string;
}

function buildSteps(): DAGStep[] {
  const steps: DAGStep[] = [];
  const d = Array(N).fill(INF);
  const prev = Array(N).fill(-1);
  d[SRC] = 0;

  const successfullyRelaxed = new Set<number>();

  steps.push({
    d: [...d], prev: [...prev],
    processedUpTo: -1, activeNode: -1,
    relaxed: [], relaxedEdges: new Set(), updatedEdges: new Set(),
    successFullyRelaxed: new Set(successfullyRelaxed),
    desc: `初始化：源点 s 的 d[s]=0，其余所有节点 d[v]=+∞。拓扑序为 r→s→x→y→z，按序逐一处理。`,
  });

  for (let ti = 0; ti < TOPO_ORDER.length; ti++) {
    const u = TOPO_ORDER[ti];
    const relaxed: DAGStep["relaxed"] = [];
    const relaxedEdges = new Set<number>();
    const updatedEdges = new Set<number>();

    const outEdges = EDGES.map((e, i) => ({ ...e, i })).filter(e => e.u === u);
    for (const { v, w, i: ei } of outEdges) {
      const before = d[v];
      relaxedEdges.add(ei);
      if (d[u] !== INF && d[u] + w < d[v]) {
        d[v] = d[u] + w;
        prev[v] = u;
        updatedEdges.add(ei);
        successfullyRelaxed.add(ei);
        relaxed.push({ edgeIdx: ei, before, after: d[v], updated: true });
      } else {
        relaxed.push({ edgeIdx: ei, before, after: d[v], updated: false });
      }
    }

    const nl = (i: number) => NODES[i].label;
    const fmtD = (v: number) => v === INF ? "∞" : String(v);
    let desc = "";
    if (d[u] === INF && !(u === SRC)) {
      desc = `处理节点 ${nl(u)}（拓扑序第 ${ti+1}）：d[${nl(u)}]=∞，从源点不可达，所有出边松弛无意义，跳过。`;
    } else if (relaxed.length === 0) {
      desc = `处理节点 ${nl(u)}（拓扑序第 ${ti+1}）：d[${nl(u)}]=${fmtD(d[u])}，无出边，直接完成。`;
    } else {
      const updDesc = relaxed.filter(r=>r.updated).map(r=>`d[${nl(EDGES[r.edgeIdx].v)}]→${r.after}`).join("，");
      desc = `处理节点 ${nl(u)}（拓扑序第 ${ti+1}）：d[${nl(u)}]=${fmtD(d[u])}，松弛 ${relaxed.length} 条边。` +
        (updDesc ? `更新：${updDesc}` : "本节点无更新。");
    }

    steps.push({
      d: [...d], prev: [...prev],
      processedUpTo: ti, activeNode: u,
      relaxed, relaxedEdges, updatedEdges,
      successFullyRelaxed: new Set(successfullyRelaxed),
      desc,
    });
  }

  const fmtD = (v: number) => v === INF ? "∞" : String(v);
  steps.push({
    d: [...d], prev: [...prev],
    processedUpTo: TOPO_ORDER.length, activeNode: -1,
    relaxed: [], relaxedEdges: new Set(), updatedEdges: new Set(),
    successFullyRelaxed: new Set(successfullyRelaxed),
    desc: `✅ 完成！${NODES.map(n=>`d[${n.label}]=${fmtD(d[n.id])}`).join("，")}。O(V+E) 复杂度，每条边恰好只需松弛一次！`,
  });
  return steps;
}

const STEPS = buildSteps();

/* ─── Arrow helper ───────────────────────────────────────────────────────── */
function Arrow({ u, v, w, ei, step }:{
  u:number; v:number; w:number; ei:number; step: typeof STEPS[0];
}) {
  const [x1,y1] = [NODES[u].x, NODES[u].y];
  const [x2,y2] = [NODES[v].x, NODES[v].y];
  const dx=x2-x1, dy=y2-y1, len=Math.sqrt(dx*dx+dy*dy);
  const R=22;
  const sx=x1+(dx/len)*R, sy=y1+(dy/len)*R;
  const ex=x2-(dx/len)*R, ey=y2-(dy/len)*R;
  const mx=(sx+ex)/2, my=(sy+ey)/2;

  // Perpendicular offset for r→s and r→x to avoid overlap with s→x
  const offsets: Record<number, number> = { 0: -10, 1: 10 };
  const off = (u === 0) ? (v===1 ? -12 : 12) : 0;
  const perpX=(-dy/len)*off, perpY=(dx/len)*off;
  const cpx=mx+perpX, cpy=my+perpY;

  const isActive = step.relaxedEdges.has(ei);
  const isUpdated = step.updatedEdges.has(ei);
  const wasUpdated = step.successFullyRelaxed.has(ei);
  const isDone = step.processedUpTo >= TOPO_ORDER.length;

  let color = "#94a3b8";
  let sw = 1.5;
  if (isDone && wasUpdated) { color = "#10b981"; sw = 2; }
  else if (isActive && isUpdated) { color = "#10b981"; sw = 2.5; }
  else if (isActive && !isUpdated) { color = "#f59e0b"; sw = 2; }
  else if (wasUpdated) { color = "#34d399"; sw = 1.8; }
  else if (w < 0) { color = "#fca5a5"; }

  const markId = `arr-dag-${u}-${v}`;
  return (
    <g>
      <defs>
        <marker id={markId} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 Z" fill={color} />
        </marker>
      </defs>
      {off !== 0
        ? <path d={`M${sx},${sy} Q${cpx},${cpy} ${ex},${ey}`}
            stroke={color} strokeWidth={sw} fill="none" markerEnd={`url(#${markId})`} />
        : <line x1={sx} y1={sy} x2={ex} y2={ey}
            stroke={color} strokeWidth={sw} markerEnd={`url(#${markId})`} />
      }
      <text x={cpx - perpX*0.3} y={cpy + perpY*0.3 - 8}
        textAnchor="middle" fontSize={10} fontWeight="bold" fill={color}>{w}</text>
    </g>
  );
}

const fmtD = (v: number) => v === INF ? "∞" : String(v);

export default function DAGShortestPath() {
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

  const isDone = step.processedUpTo >= TOPO_ORDER.length;

  function nodeStyle(id: number) {
    const ti = TOPO_ORDER.indexOf(id);
    if (isDone) return { fill: step.d[id] === INF ? "#94a3b8" : "#10b981", stroke: step.d[id] === INF ? "#64748b" : "#059669", text: "#fff" };
    if (id === step.activeNode) return { fill: "#10b981", stroke: "#059669", text: "#fff" };
    if (ti < step.processedUpTo && ti >= 0) return { fill: "#34d399", stroke: "#10b981", text: "#fff" };
    if (id === SRC) return { fill: "#6366f1", stroke: "#4f46e5", text: "#fff" };
    if (step.d[id] !== INF) return { fill: "#3b82f6", stroke: "#2563eb", text: "#fff" };
    return { fill: "#e2e8f0", stroke: "#94a3b8", text: "#94a3b8" };
  }

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">DAG 最短路径——拓扑序松弛</h3>
        <p className="text-emerald-100 text-sm mt-0.5">按拓扑序处理，每条边只需松弛一次，时间复杂度 O(V+E)</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Topo order progress bar */}
        <div className="space-y-1">
          <div className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide">拓扑序处理进度</div>
          <div className="flex gap-1">
            {TOPO_ORDER.map((nodeId, ti) => {
              const node = NODES[nodeId];
              const isActive = nodeId === step.activeNode;
              const isDoneNode = !isDone && ti < step.processedUpTo;
              const isDoneAll = isDone;
              return (
                <div key={ti} className="flex-1 flex flex-col items-center gap-0.5">
                  <div className={`w-full h-7 rounded-lg flex items-center justify-center text-xs font-bold transition-all duration-300 ${
                    isDoneAll ? "bg-emerald-500 text-white" :
                    isActive ? "bg-emerald-600 text-white ring-2 ring-emerald-400 scale-105" :
                    isDoneNode ? "bg-emerald-200 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-400" :
                    "bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-500"
                  }`}>
                    {node.label}
                  </div>
                  <div className={`text-[9px] font-bold ${
                    isDoneAll ? "text-emerald-600 dark:text-emerald-400" :
                    isActive ? "text-emerald-700 dark:text-emerald-400" :
                    "text-slate-400"
                  }`}>
                    {fmtD(step.d[nodeId])}
                  </div>
                </div>
              );
            })}
          </div>
          {/* Arrow indicator */}
          <div className="flex gap-1">
            {TOPO_ORDER.map((nodeId, ti) => (
              <div key={ti} className="flex-1 flex justify-center">
                {nodeId === step.activeNode && !isDone
                  ? <div className="text-emerald-600 dark:text-emerald-400 text-sm font-bold">▲</div>
                  : <div className="w-4" />
                }
              </div>
            ))}
          </div>
        </div>

        <div className="flex gap-4 items-start">
          {/* Graph SVG */}
          <div className="flex-1 min-w-0 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 overflow-hidden">
            <svg viewBox="0 0 510 265" className="w-full">
              {EDGES.map((e, i) => (
                <Arrow key={i} u={e.u} v={e.v} w={e.w} ei={i} step={step} />
              ))}
              {NODES.map(node => {
                const s = nodeStyle(node.id);
                const isActive = node.id === step.activeNode;
                const wasUpdated = step.relaxed.some(r => {
                  const e = EDGES[r.edgeIdx];
                  return e.v === node.id && r.updated;
                });
                return (
                  <g key={node.id}>
                    {isActive && (
                      <circle cx={node.x} cy={node.y} r={30}
                        fill="#d1fae533" stroke="#10b98155" strokeWidth={2} />
                    )}
                    {wasUpdated && !isActive && (
                      <circle cx={node.x} cy={node.y} r={28}
                        fill="#fde68a33" stroke="#f59e0b55" strokeWidth={1.5} />
                    )}
                    <circle cx={node.x} cy={node.y} r={22}
                      fill={s.fill} stroke={s.stroke} strokeWidth={2}
                      className="transition-all duration-300" />
                    <text x={node.x} y={node.y-5} textAnchor="middle" fontSize={13} fontWeight="bold" fill={s.text}>
                      {node.label}
                    </text>
                    <text x={node.x} y={node.y+8} textAnchor="middle" fontSize={9} fill={s.text} opacity={0.9}>
                      {fmtD(step.d[node.id])}
                    </text>
                  </g>
                );
              })}
              {/* Source label */}
              <text x={NODES[SRC].x} y={NODES[SRC].y-35} textAnchor="middle"
                fontSize={9} fill="#6366f1" fontWeight="bold">源点</text>
            </svg>
          </div>

          {/* Right: legend + relaxation details */}
          <div className="w-44 shrink-0 space-y-3">
            {/* Legend */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 p-3 space-y-1.5">
              {[
                { color: "bg-slate-200 dark:bg-slate-600", label: "未处理 (∞)" },
                { color: "bg-indigo-500", label: "源点 s" },
                { color: "bg-blue-500", label: "已到达" },
                { color: "bg-emerald-600", label: "当前处理" },
                { color: "bg-emerald-300 dark:bg-emerald-700", label: "已处理完" },
              ].map(({ color, label }) => (
                <div key={label} className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
                  <div className={`w-3 h-3 rounded-full ${color} shrink-0`} />
                  {label}
                </div>
              ))}
            </div>

            {/* Edge color legend */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 p-3 space-y-1.5">
              <div className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide">边颜色</div>
              {[
                { color: "bg-slate-300 dark:bg-slate-600", label: "未松弛" },
                { color: "bg-amber-400", label: "已松弛（无更新）" },
                { color: "bg-emerald-500", label: "松弛并更新！" },
              ].map(({ color, label }) => (
                <div key={label} className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                  <div className={`w-6 h-1.5 rounded-full ${color} shrink-0`} />
                  {label}
                </div>
              ))}
            </div>

            {/* Current relaxation details */}
            {step.relaxed.length > 0 && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
                <div className="bg-emerald-100 dark:bg-emerald-900/40 px-3 py-1.5 text-[11px] font-bold text-emerald-700 dark:text-emerald-300 uppercase tracking-wide">
                  本步松弛
                </div>
                <div className="p-1.5 space-y-0.5">
                  {step.relaxed.map((r, i) => {
                    const e = EDGES[r.edgeIdx];
                    return (
                      <div key={i} className={`px-2 py-1 rounded text-[10px] font-mono ${
                        r.updated
                          ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300"
                          : "bg-slate-100 dark:bg-slate-800 text-slate-400"
                      }`}>
                        <div className="flex justify-between">
                          <span>{NODES[e.u].label}→{NODES[e.v].label}(w={e.w})</span>
                          <span>{r.updated ? "✓" : "✗"}</span>
                        </div>
                        {r.updated && (
                          <div className="text-emerald-600 dark:text-emerald-400">
                            {fmtD(r.before)} → {r.after}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Description */}
        <div className="rounded-xl bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-700/50 px-4 py-2.5 text-sm text-emerald-800 dark:text-emerald-300 min-h-[44px]">
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
              playing ? "bg-amber-500 hover:bg-amber-600 text-white" : "bg-emerald-600 hover:bg-emerald-700 text-white"
            }`}>
            {playing ? "⏸ 暂停" : stepIdx === 0 ? "▶ 开始" : "▶ 播放"}
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.min(STEPS.length-1, i+1)); }} disabled={stepIdx === STEPS.length-1}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            下一步 →
          </button>
          <span className="text-xs text-slate-400 font-mono">{stepIdx+1}/{STEPS.length}</span>
          <div className="flex items-center gap-1.5 ml-auto">
            <input type="range" min={700} max={2500} step={200} value={speed}
              onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-emerald-500" />
          </div>
        </div>
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
          <div className="bg-emerald-500 h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${(stepIdx/(STEPS.length-1))*100}%` }} />
        </div>
      </div>
    </div>
  );
}
