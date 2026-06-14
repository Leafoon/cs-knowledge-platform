"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── Graph ──────────────────────────────────────────────────────────────── */
// Directed graph: 0→1, 1→2, 1→3, 0→4
// DFS timestamps: 0:[1,10] 1:[2,7] 2:[3,4] 3:[5,6] 4:[8,9]

interface Node { id: number; x: number; y: number; label: string; }
interface Edge { u: number; v: number; }

const NODES: Node[] = [
  { id: 0, x: 65,  y: 115, label: "0" },
  { id: 1, x: 165, y: 65,  label: "1" },
  { id: 2, x: 265, y: 42,  label: "2" },
  { id: 3, x: 265, y: 115, label: "3" },
  { id: 4, x: 165, y: 170, label: "4" },
];
const EDGES: Edge[] = [
  { u: 0, v: 1 }, { u: 1, v: 2 }, { u: 1, v: 3 }, { u: 0, v: 4 },
];
const ADJ: Record<number, number[]> = { 0: [1, 4], 1: [2, 3], 2: [], 3: [], 4: [] };

/* ─── DFS Steps ──────────────────────────────────────────────────────────── */
type NColor = "white" | "gray" | "black";
type EventType = "enter" | "exit" | "init" | "done";

interface DFSStep {
  colors: NColor[];
  discover: (number | null)[];
  finish: (number | null)[];
  stack: number[];          // DFS call stack (node ids)
  timer: number;
  eventType: EventType;
  eventNode: number | null;
  description: string;
  treeEdges: [number, number][];
}

function buildDFSSteps(): DFSStep[] {
  const n = NODES.length;
  const colors: NColor[] = Array(n).fill("white");
  const discover: (number | null)[] = Array(n).fill(null);
  const finish: (number | null)[] = Array(n).fill(null);
  const stack: number[] = [];
  const treeEdges: [number, number][] = [];
  let timer = 0;
  const steps: DFSStep[] = [];

  const snap = (event: EventType, node: number | null, desc: string) => {
    steps.push({
      colors: [...colors] as NColor[],
      discover: [...discover],
      finish: [...finish],
      stack: [...stack],
      timer,
      eventType: event,
      eventNode: node,
      description: desc,
      treeEdges: treeEdges.map(e => [...e] as [number, number]),
    });
  };

  snap("init", null, "初始状态：所有节点为白色（未访问），时间计数器 = 0");

  function dfs(u: number, parent: number | null) {
    colors[u] = "gray";
    timer++;
    discover[u] = timer;
    stack.push(u);
    snap("enter", u, `发现节点 ${NODES[u].label}：时钟 +1 → d[${NODES[u].label}] = ${timer}，变灰色（入 DFS 栈）`);

    for (const v of ADJ[u]) {
      if (colors[v] === "white") {
        treeEdges.push([u, v]);
        dfs(v, u);
      }
    }

    colors[u] = "black";
    stack.pop();
    timer++;
    finish[u] = timer;
    snap("exit", u, `完成节点 ${NODES[u].label}：时钟 +1 → f[${NODES[u].label}] = ${timer}，变黑色（出 DFS 栈）`);
  }

  for (const node of NODES) {
    if (colors[node.id] === "white") dfs(node.id, null);
  }

  snap("done", null, "DFS 完成！所有节点已处理。括号结构：[d[u], f[u]] 区间要么嵌套，要么不相交。");
  return steps;
}

const STEPS = buildDFSSteps();
const MAX_TIMER = 10; // 5 nodes × 2 timestamps

/* ─── Bracket Bar ────────────────────────────────────────────────────────── */
const NODE_COLORS = [
  { gray: "#6366f1", black: "#312e81", light: "#e0e7ff", text: "indigo" },
  { gray: "#0ea5e9", black: "#0c4a6e", light: "#e0f2fe", text: "sky" },
  { gray: "#10b981", black: "#064e3b", light: "#d1fae5", text: "emerald" },
  { gray: "#f59e0b", black: "#78350f", light: "#fef3c7", text: "amber" },
  { gray: "#ec4899", black: "#831843", light: "#fce7f3", text: "pink" },
];

function BracketBar({ step }: { step: DFSStep }) {
  const total = MAX_TIMER;
  return (
    <div className="space-y-1.5">
      <div className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
        括号区间 [d[u], f[u]]
      </div>
      {NODES.map(node => {
        const d = step.discover[node.id];
        const f = step.finish[node.id];
        const col = NODE_COLORS[node.id];
        const leftPct = d !== null ? ((d - 1) / total) * 100 : 0;
        const widthPct = (d !== null && f !== null) ? ((f - d + 1) / total) * 100 : 0;
        return (
          <div key={node.id} className="flex items-center gap-2">
            <div className={`text-[10px] font-bold w-5 text-${col.text}-600 dark:text-${col.text}-400`}>
              {node.label}
            </div>
            <div className="flex-1 relative h-5 bg-slate-100 dark:bg-slate-800 rounded">
              {/* Timeline ticks */}
              {Array.from({ length: total }, (_, i) => (
                <div key={i} className="absolute top-0 h-full border-l border-slate-200 dark:border-slate-700"
                  style={{ left: `${(i / total) * 100}%` }} />
              ))}
              {/* Bracket bar */}
              {d !== null && (
                <div
                  className="absolute top-1 h-3 rounded flex items-center justify-center text-[9px] font-bold text-white transition-all duration-300"
                  style={{
                    left: `${leftPct}%`,
                    width: `${widthPct > 0 ? widthPct : (1 / total) * 100}%`,
                    backgroundColor: f !== null ? col.gray : col.gray + "99",
                    minWidth: "16px",
                  }}>
                  {f !== null ? `(${d}..${f})` : `(${d}..`}
                </div>
              )}
            </div>
            <div className="text-[10px] font-mono w-14 text-right text-slate-500 dark:text-slate-400">
              {d ?? "—"} / {f ?? "—"}
            </div>
          </div>
        );
      })}
      {/* Tick labels */}
      <div className="flex ml-7 mr-14">
        {Array.from({ length: total + 1 }, (_, i) => (
          <div key={i} className="flex-1 text-[8px] text-slate-400 dark:text-slate-600 text-center">{i}</div>
        ))}
      </div>
    </div>
  );
}

/* ─── Main Component ─────────────────────────────────────────────────────── */
export default function DFSTimestamp() {
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1000);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const step = STEPS[stepIdx];
  const maxStep = STEPS.length - 1;

  const advance = useCallback(() => {
    setStepIdx(prev => {
      if (prev >= maxStep) { setPlaying(false); return prev; }
      return prev + 1;
    });
  }, [maxStep]);

  useEffect(() => {
    if (playing) { intervalRef.current = setInterval(advance, speed); }
    else { if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; } }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, speed, advance]);

  const isTreeEdge = (u: number, v: number) =>
    step.treeEdges.some(([a, b]) => a === u && b === v);

  // Arrow marker position helper
  function edgeArrow(u: number, v: number): { x1: number; y1: number; x2: number; y2: number } {
    const nu = NODES[u], nv = NODES[v];
    const dx = nv.x - nu.x, dy = nv.y - nu.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const r = 18;
    return {
      x1: nu.x + (dx / dist) * r,
      y1: nu.y + (dy / dist) * r,
      x2: nv.x - (dx / dist) * r,
      y2: nv.y - (dy / dist) * r,
    };
  }

  function nodeCircleColor(id: number, color: NColor) {
    const col = NODE_COLORS[id];
    if (color === "gray") return col.gray;
    if (color === "black") return col.black;
    return "#94a3b8";
  }

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-500 via-purple-500 to-fuchsia-500 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">DFS 时间戳与括号定理</h3>
        <p className="text-violet-100 text-sm mt-0.5">观察 d[u]（发现时间）和 f[u]（完成时间）的赋值过程，感悟括号嵌套结构</p>
      </div>

      <div className="p-5 space-y-4">
        <div className="flex flex-col sm:flex-row gap-4">
          {/* SVG Graph */}
          <div className="sm:w-[320px] rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-2 flex-shrink-0">
            <svg viewBox="0 0 330 215" className="w-full">
              <defs>
                <marker id="ts-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L8,3 z" fill="#94a3b8" />
                </marker>
                <marker id="ts-arrow-tree" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L8,3 z" fill="#6366f1" />
                </marker>
              </defs>
              {EDGES.map(({ u, v }) => {
                const { x1, y1, x2, y2 } = edgeArrow(u, v);
                const tree = isTreeEdge(u, v);
                return (
                  <line key={`${u}-${v}`}
                    x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke={tree ? "#6366f1" : "#94a3b8"}
                    strokeWidth={tree ? 2.5 : 1.5}
                    markerEnd={tree ? "url(#ts-arrow-tree)" : "url(#ts-arrow)"}
                    className="transition-all duration-300"
                  />
                );
              })}
              {NODES.map(node => {
                const col = nodeCircleColor(node.id, step.colors[node.id]);
                const d = step.discover[node.id];
                const f = step.finish[node.id];
                const inStack = step.stack.includes(node.id);
                return (
                  <g key={node.id}>
                    {inStack && <circle cx={node.x} cy={node.y} r={24} fill="#ede9fe" stroke="#a78bfa" strokeWidth={1.5} opacity={0.5} />}
                    <circle cx={node.x} cy={node.y} r={18}
                      fill={col} stroke={inStack ? "#7c3aed" : "#e2e8f0"} strokeWidth={2}
                      className="transition-all duration-300" />
                    <text x={node.x} y={node.y} textAnchor="middle" dominantBaseline="central"
                      fontSize={13} fontWeight="bold" fill="white">{node.label}</text>
                    {/* d/f labels */}
                    <text x={node.x - 22} y={node.y - 16} textAnchor="middle" fontSize={9} fill="#6366f1" fontWeight="bold">
                      {d !== null ? d : ""}
                    </text>
                    <text x={node.x + 22} y={node.y - 16} textAnchor="middle" fontSize={9} fill="#10b981" fontWeight="bold">
                      {f !== null ? f : ""}
                    </text>
                  </g>
                );
              })}
              {/* Timer display */}
              <text x={4} y={205} fontSize={11} fill="#94a3b8" fontWeight="bold">时钟 t = {step.timer}</text>
            </svg>
          </div>

          {/* Right panel */}
          <div className="flex-1 space-y-3">
            {/* DFS stack */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-3">
              <div className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase mb-2 tracking-wide">DFS 调用栈</div>
              {step.stack.length === 0 ? (
                <div className="text-[11px] text-slate-400 italic">空</div>
              ) : (
                <div className="flex flex-col-reverse gap-1">
                  {step.stack.map((id, i) => (
                    <div key={i} className="flex items-center gap-2 text-[11px] rounded-lg px-2 py-1"
                      style={{ backgroundColor: NODE_COLORS[id].light }}>
                      <span className="text-[9px] text-slate-400">{i === step.stack.length - 1 ? "← TOP" : ""}</span>
                      <span className="font-bold" style={{ color: NODE_COLORS[id].gray }}>
                        dfs_visit({NODES[id].label})
                      </span>
                      <span className="ml-auto font-mono text-[10px]" style={{ color: NODE_COLORS[id].gray }}>
                        d={step.discover[id]}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Timestamp table */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-3">
              <div className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase mb-2 tracking-wide">时间戳记录</div>
              <div className="grid grid-cols-3 gap-x-3 gap-y-1 text-[11px]">
                <div className="font-bold text-slate-500 dark:text-slate-400">节点</div>
                <div className="font-bold text-indigo-600 dark:text-indigo-400 text-center">d[ ]</div>
                <div className="font-bold text-emerald-600 dark:text-emerald-400 text-center">f[ ]</div>
                {NODES.map(node => (
                  <React.Fragment key={node.id}>
                    <div className="font-bold" style={{ color: NODE_COLORS[node.id].gray }}>{node.label}</div>
                    <div className="font-mono text-center text-indigo-700 dark:text-indigo-300">
                      {step.discover[node.id] ?? "—"}
                    </div>
                    <div className="font-mono text-center text-emerald-700 dark:text-emerald-300">
                      {step.finish[node.id] ?? "—"}
                    </div>
                  </React.Fragment>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Bracket bar */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-4">
          <BracketBar step={step} />
        </div>

        {/* Description */}
        <div className="rounded-xl bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 px-4 py-3">
          <div className="text-xs text-purple-700 dark:text-purple-300 leading-relaxed">
            <span className="font-bold mr-1">步骤 {stepIdx}/{maxStep}：</span>
            {step.description}
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={() => { setPlaying(false); setStepIdx(0); }}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            ↺ 重置
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.max(0, i - 1)); }} disabled={stepIdx === 0}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            ← 上一步
          </button>
          <button onClick={() => setPlaying(p => !p)}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-colors ${playing ? "bg-amber-500 hover:bg-amber-600 text-white" : "bg-purple-500 hover:bg-purple-600 text-white"}`}>
            {playing ? "⏸ 暂停" : "▶ 播放"}
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.min(maxStep, i + 1)); }} disabled={stepIdx === maxStep}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            下一步 →
          </button>
          <div className="flex items-center gap-2 ml-auto">
            <span className="text-[10px] text-slate-500 dark:text-slate-400">速度</span>
            <input type="range" min={400} max={1800} step={200} value={speed}
              onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-purple-500" />
          </div>
        </div>

        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
          <div className="bg-purple-500 h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${(stepIdx / maxStep) * 100}%` }} />
        </div>
      </div>
    </div>
  );
}
