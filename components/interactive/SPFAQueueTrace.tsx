"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── Same 5-node graph as Bellman-Ford ──────────────────────────────────── */
const NODES = [
  { id: 0, x: 70,  y: 130, label: "s" },
  { id: 1, x: 200, y: 68,  label: "t" },
  { id: 2, x: 340, y: 130, label: "x" },
  { id: 3, x: 200, y: 192, label: "y" },
  { id: 4, x: 460, y: 130, label: "z" },
];
// adj[u] = [(v, w), ...]
const ADJ: [number, number][][] = [
  [[1,6],[3,7]],          // s → t(6), y(7)
  [[2,5],[3,8],[4,-4]],   // t → x(5), y(8), z(-4)
  [[1,-2]],               // x → t(-2)
  [[2,-3],[4,9]],         // y → x(-3), z(9)
  [[2,7],[0,2]],          // z → x(7), s(2)
];
const N = 5;
const INF = 1e9;
const EDGES_ALL = [
  {u:0,v:1,w:6},{u:0,v:3,w:7},
  {u:1,v:2,w:5},{u:1,v:3,w:8},{u:1,v:4,w:-4},
  {u:2,v:1,w:-2},{u:3,v:2,w:-3},{u:3,v:4,w:9},
  {u:4,v:2,w:7},{u:4,v:0,w:2},
];

/* ─── Pre-compute SPFA steps ─────────────────────────────────────────────── */
interface SPFAStep {
  d: number[];
  inQueue: boolean[];
  queue: number[];          // current queue after this step
  processing: number;       // node just dequeued (-1 = initial/done)
  relaxed: { v: number; before: number; after: number; updated: boolean }[];
  reenqueued: number[];     // nodes newly added to queue this step
  enqueueCount: number[];
  desc: string;
}

function buildSPFASteps(): SPFAStep[] {
  const steps: SPFAStep[] = [];
  const d = [0, INF, INF, INF, INF];
  const inQueue = Array(N).fill(false);
  const enqueueCount = Array(N).fill(0);
  const queue: number[] = [0];
  inQueue[0] = true; enqueueCount[0] = 1;

  steps.push({
    d: [...d], inQueue: [...inQueue], queue: [...queue],
    processing: -1, relaxed: [], reenqueued: [], enqueueCount: [...enqueueCount],
    desc: "初始化：d[s]=0，其余 d[v]=∞。将源点 s 加入队列。SPFA 将持续从队列取出节点并松弛其所有邻边。",
  });

  while (queue.length > 0) {
    const u = queue.shift()!;
    inQueue[u] = false;
    const relaxed: SPFAStep["relaxed"] = [];
    const reenqueued: number[] = [];

    for (const [v, w] of ADJ[u]) {
      const before = d[v];
      if (d[u] !== INF && d[u] + w < d[v]) {
        d[v] = d[u] + w;
        relaxed.push({ v, before, after: d[v], updated: true });
        if (!inQueue[v]) {
          inQueue[v] = true;
          enqueueCount[v]++;
          queue.push(v);
          reenqueued.push(v);
        }
      } else {
        relaxed.push({ v, before, after: d[v], updated: false });
      }
    }

    const nl = (i: number) => NODES[i].label;
    const fmtD = (v: number) => v === INF ? "∞" : String(v);
    const updDesc = relaxed.filter(r=>r.updated).map(r=>`d[${nl(r.v)}]→${fmtD(r.after)}`).join("，") || "无更新";
    const reqDesc = reenqueued.length > 0 ? `，重新入队：[${reenqueued.map(nl).join(",")}]` : "";

    steps.push({
      d: [...d], inQueue: [...inQueue], queue: [...queue],
      processing: u, relaxed, reenqueued, enqueueCount: [...enqueueCount],
      desc: `出队节点「${nl(u)}」（第 ${enqueueCount[u]} 次处理），松弛 ${ADJ[u].length} 条出边。${updDesc}${reqDesc}。当前队列：[${queue.map(nl).join(" → ")||"空"}]`,
    });
  }

  const fmtD = (v: number) => v === INF ? "∞" : String(v);
  steps.push({
    d: [...d], inQueue: Array(N).fill(false), queue: [],
    processing: -1, relaxed: [], reenqueued: [], enqueueCount: [...enqueueCount],
    desc: `✅ SPFA 完成！队列为空，算法终止。d=[${d.map(fmtD).join(",")}]。节点 s/t/x/y/z 分别为 0/2/4/7/-2。`,
  });
  return steps;
}

const STEPS = buildSPFASteps();

/* ─── Arrow helper ────────────────────────────────────────────────────────── */
function Arrow({ u, v, w, highlighted, updated }:{
  u:number; v:number; w:number; highlighted:boolean; updated:boolean;
}) {
  const [x1,y1] = [NODES[u].x, NODES[u].y];
  const [x2,y2] = [NODES[v].x, NODES[v].y];
  const dx=x2-x1, dy=y2-y1, len=Math.sqrt(dx*dx+dy*dy);
  const R=22;
  const sx=x1+(dx/len)*R, sy=y1+(dy/len)*R;
  const ex=x2-(dx/len)*R, ey=y2-(dy/len)*R;
  const mx=(sx+ex)/2, my=(sy+ey)/2;

  // Offset for t↔x parallel edges
  const offsets: Record<string,number> = {"1-2":10,"2-1":-10};
  const off = offsets[`${u}-${v}`] ?? 0;
  const perpX=(-dy/len)*off, perpY=(dx/len)*off;
  const cpx=mx+perpX, cpy=my+perpY;

  let color = w<0 ? "#fca5a5" : "#94a3b8";
  let sw = 1.5;
  if (highlighted) {
    color = updated ? "#10b981" : "#f59e0b";
    sw = 2.5;
  }
  const markId = `arr-spfa-${u}-${v}`;
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
      <text x={cpx - perpX*0.2} y={cpy + perpY*0.2 - 7}
        textAnchor="middle" fontSize={10} fontWeight="bold" fill={color}>{w}</text>
    </g>
  );
}

const fmtD = (v: number) => v === INF ? "∞" : String(v);

export default function SPFAQueueTrace() {
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

  const isDone = step.processing === -1 && stepIdx === STEPS.length-1;

  // edges relaxed in this step
  const relaxedEdges = new Set(step.relaxed.map(r => `${step.processing}-${r.v}`));
  const updatedEdges = new Set(step.relaxed.filter(r=>r.updated).map(r => `${step.processing}-${r.v}`));

  function nodeStyle(id: number) {
    if (isDone) return { fill: "#10b981", stroke: "#059669", text: "#fff" };
    if (id === step.processing) return { fill: "#f59e0b", stroke: "#d97706", text: "#fff" };
    if (step.reenqueued.includes(id)) return { fill: "#6366f1", stroke: "#4f46e5", text: "#fff" };
    if (step.inQueue[id]) return { fill: "#818cf8", stroke: "#6366f1", text: "#fff" };
    if (step.d[id] !== INF) return { fill: "#3b82f6", stroke: "#2563eb", text: "#fff" };
    return { fill: "#e2e8f0", stroke: "#94a3b8", text: "#94a3b8" };
  }

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-teal-600 via-cyan-600 to-sky-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">SPFA——队列追踪步进动画</h3>
        <p className="text-teal-100 text-sm mt-0.5">只将 d[] 被更新的节点重新入队，平均效率远优于 Bellman-Ford</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Color legend */}
        <div className="flex flex-wrap gap-3 text-xs">
          {[
            { color: "bg-slate-200 dark:bg-slate-700", label: "未达" },
            { color: "bg-blue-500", label: "已达（不在队列）" },
            { color: "bg-indigo-400", label: "队列中" },
            { color: "bg-amber-400", label: "正在处理（出队）" },
            { color: "bg-violet-500", label: "本步骤重新入队" },
            { color: "bg-emerald-500", label: "完成" },
          ].map(({ color, label }) => (
            <div key={label} className="flex items-center gap-1.5">
              <div className={`w-3 h-3 rounded-full ${color}`} />
              <span className="text-slate-500 dark:text-slate-400">{label}</span>
            </div>
          ))}
        </div>

        <div className="flex gap-4 items-start">
          {/* Graph */}
          <div className="flex-1 min-w-0 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 overflow-hidden">
            <svg viewBox="0 0 535 265" className="w-full">
              {EDGES_ALL.map((e, i) => {
                const key = `${e.u}-${e.v}`;
                return (
                  <Arrow key={i} {...e}
                    highlighted={relaxedEdges.has(key)}
                    updated={updatedEdges.has(key)} />
                );
              })}
              {NODES.map(node => {
                const s = nodeStyle(node.id);
                const isProcessing = node.id === step.processing;
                const isReenq = step.reenqueued.includes(node.id);
                const times = step.enqueueCount[node.id];
                return (
                  <g key={node.id}>
                    {(isProcessing || isReenq) && (
                      <circle cx={node.x} cy={node.y} r={30}
                        fill={isProcessing ? "#fde68a33" : "#a5b4fc33"}
                        stroke={isProcessing ? "#f59e0b55" : "#6366f155"} strokeWidth={2} />
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
                    {/* Enqueue count badge */}
                    {times > 0 && (
                      <>
                        <circle cx={node.x+17} cy={node.y-17} r={8}
                          fill={times > 1 ? "#f59e0b" : "#94a3b8"}
                          stroke={times > 1 ? "#d97706" : "#64748b"} strokeWidth={1} />
                        <text x={node.x+17} y={node.y-17} textAnchor="middle"
                          dominantBaseline="central" fontSize={8} fontWeight="bold" fill="#fff">
                          {times}
                        </text>
                      </>
                    )}
                  </g>
                );
              })}
              <text x={8} y={255} fontSize={8} fill="#94a3b8">右上角数字 = 入队次数</text>
            </svg>
          </div>

          {/* Right panel */}
          <div className="w-44 shrink-0 space-y-3">
            {/* Queue visualization - FIFO */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-cyan-100 dark:bg-cyan-900/40 px-3 py-1.5 flex justify-between items-center">
                <span className="text-xs font-bold text-cyan-700 dark:text-cyan-300">队列 Queue (FIFO)</span>
                <span className="text-[10px] text-cyan-500 dark:text-cyan-400">← 出队 / 入队 →</span>
              </div>
              <div className="p-2 min-h-[56px] bg-white dark:bg-slate-800/50 flex items-center">
                {step.queue.length === 0
                  ? <p className="text-xs text-slate-400 text-center w-full">队列为空 ✓</p>
                  : <div className="flex flex-wrap gap-1 w-full">
                      {step.queue.map((id, i) => (
                        <span key={i} className={`px-2 py-0.5 rounded-lg text-[11px] font-bold text-white ${
                          step.reenqueued.includes(id) ? "bg-violet-500" : "bg-indigo-500"
                        }`}>
                          {NODES[id].label}
                        </span>
                      ))}
                    </div>
                }
              </div>
            </div>

            {/* d[] table */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-slate-100 dark:bg-slate-800 px-3 py-1.5 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                距离表 d[]
              </div>
              <div className="divide-y divide-slate-100 dark:divide-slate-700/50">
                {NODES.map(node => {
                  const relax = step.relaxed.find(r => r.v === node.id);
                  return (
                    <div key={node.id} className={`flex justify-between items-center px-3 py-1.5 text-xs transition-colors ${
                      node.id === step.processing ? "bg-amber-50 dark:bg-amber-900/20" :
                      relax?.updated ? "bg-emerald-50 dark:bg-emerald-900/20" : ""
                    }`}>
                      <span className="font-bold text-slate-600 dark:text-slate-300">d[{NODES[node.id].label}]</span>
                      <div className="flex items-center gap-1">
                        {relax?.updated && (
                          <span className="text-slate-400 line-through text-[10px]">{fmtD(relax.before)}</span>
                        )}
                        <span className={`font-mono font-bold text-[13px] ${
                          relax?.updated ? "text-emerald-600 dark:text-emerald-400" :
                          step.d[node.id] === INF ? "text-slate-300 dark:text-slate-600" : "text-blue-600 dark:text-blue-400"
                        }`}>
                          {fmtD(step.d[node.id])}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Relaxed edges this step */}
            {step.relaxed.length > 0 && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
                <div className="bg-slate-100 dark:bg-slate-800 px-3 py-1.5 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                  本步松弛
                </div>
                <div className="p-1.5 space-y-0.5">
                  {step.relaxed.map((r, i) => (
                    <div key={i} className={`flex justify-between items-center px-2 py-0.5 rounded text-[10px] font-mono ${
                      r.updated
                        ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300"
                        : "text-slate-400 dark:text-slate-500"
                    }`}>
                      <span>{NODES[step.processing]?.label}→{NODES[r.v].label}</span>
                      {r.updated
                        ? <span className="font-bold">✓{fmtD(r.after)}</span>
                        : <span>-</span>
                      }
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Description */}
        <div className="rounded-xl bg-cyan-50 dark:bg-cyan-900/20 border border-cyan-200 dark:border-cyan-700/50 px-4 py-2.5 text-sm text-cyan-800 dark:text-cyan-300 min-h-[44px]">
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
              playing ? "bg-amber-500 hover:bg-amber-600 text-white" : "bg-teal-600 hover:bg-teal-700 text-white"
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
              onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-teal-500" />
          </div>
        </div>
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
          <div className="bg-teal-500 h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${(stepIdx/(STEPS.length-1))*100}%` }} />
        </div>
      </div>
    </div>
  );
}
