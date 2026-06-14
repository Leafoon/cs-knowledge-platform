"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── Graph Definition ───────────────────────────────────────────────────── */
//
//  课程依赖图（Kahn 算法演示）：
//
//  高数(0) ──→ 概率(2) ──→ 机器学习(4) ──→ 深度学习(5)
//     │                       ↑
//     └──→ 统计(3) ───────────┘
//  线代(1) ──→ 概率(2)
//

interface GNode { id: number; x: number; y: number; label: string; sublabel: string; }
const NODES: GNode[] = [
  { id: 0, x: 55,  y: 60,  label: "高数",   sublabel: "0" },
  { id: 1, x: 55,  y: 155, label: "线代",   sublabel: "1" },
  { id: 2, x: 175, y: 108, label: "概率",   sublabel: "2" },
  { id: 3, x: 55,  y: 255, label: "统计",   sublabel: "3" },
  { id: 4, x: 295, y: 108, label: "机器学习", sublabel: "4" },
  { id: 5, x: 415, y: 108, label: "深度学习", sublabel: "5" },
];
const EDGES = [[0,2],[0,3],[1,2],[2,4],[3,4],[4,5]];
const N = NODES.length;

/* ─── Precompute Kahn Steps ──────────────────────────────────────────────── */
interface KahnStep {
  indegree: number[];      // 当前入度（快照）
  queue: number[];         // 当前队列
  output: number[];        // 已输出的拓扑序
  processing: number;      // 正在处理的节点（-1=无）
  done: boolean[];         // 已完成的节点
  desc: string;
}

function computeKahnSteps(): KahnStep[] {
  const steps: KahnStep[] = [];
  const indegree = Array(N).fill(0);
  const adj: number[][] = Array.from({ length: N }, () => []);
  EDGES.forEach(([u, v]) => { indegree[v]++; adj[u].push(v); });

  const queue: number[] = indegree.map((d, i) => d === 0 ? i : -1).filter(i => i >= 0);
  const output: number[] = [];
  const done: boolean[] = Array(N).fill(false);

  // Initial state
  steps.push({
    indegree: [...indegree], queue: [...queue],
    output: [], processing: -1, done: [...done],
    desc: `初始化：计算所有节点入度。入度为 0 的节点（${queue.map(i => NODES[i].label).join("、")}）入队，可以立即学习。`,
  });

  const queueWork = [...queue];
  while (queueWork.length > 0) {
    const u = queueWork.shift()!;
    steps.push({
      indegree: [...indegree], queue: [...queueWork],
      output: [...output], processing: u, done: [...done],
      desc: `出队节点「${NODES[u].label}」，加入拓扑序。准备处理它的所有后继课程。`,
    });
    output.push(u);
    done[u] = true;
    for (const v of adj[u]) {
      indegree[v]--;
      if (indegree[v] === 0) queueWork.push(v);
    }
    steps.push({
      indegree: [...indegree], queue: [...queueWork],
      output: [...output], processing: -1, done: [...done],
      desc: `「${NODES[u].label}」处理完毕。${adj[u].length > 0
        ? `其后继课程 ${adj[u].map(v => `「${NODES[v].label}」入度→${indegree[v]}`).join("，")}。` + (queueWork.length > 0 ? `入度降为 0 的节点已加入队列。` : "")
        : "无后继节点。"}`,
    });
  }

  steps.push({
    indegree: [...indegree], queue: [], output: [...output],
    processing: -1, done: [...done],
    desc: `✅ 拓扑排序完成！输出序列共 ${output.length} 个节点（= 节点总数），图中无环。`,
  });
  return steps;
}

const STEPS = computeKahnSteps();

/* ─── Arrow helper ───────────────────────────────────────────────────────── */
function Arrow({ x1, y1, x2, y2, color = "#94a3b8" }: { x1:number;y1:number;x2:number;y2:number;color?:string }) {
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.sqrt(dx*dx + dy*dy);
  const R = 22;
  const ex = x2 - (dx/len)*R, ey = y2 - (dy/len)*R;
  const sx = x1 + (dx/len)*R, sy = y1 + (dy/len)*R;
  return (
    <line x1={sx} y1={sy} x2={ex} y2={ey}
      stroke={color} strokeWidth={1.8} markerEnd={`url(#arr-${color.replace('#','')})`} />
  );
}

/* ─── Main Component ─────────────────────────────────────────────────────── */
export default function TopologicalSortKahn() {
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(900);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const step = STEPS[stepIdx];

  const advance = useCallback(() => {
    setStepIdx(prev => { if (prev >= STEPS.length - 1) { setPlaying(false); return prev; } return prev + 1; });
  }, []);

  useEffect(() => {
    if (playing) { intervalRef.current = setInterval(advance, speed); }
    else { if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; } }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, speed, advance]);

  const reset = () => { setPlaying(false); setStepIdx(0); };

  function nodeColor(id: number) {
    if (step.done[id]) return { fill: "#10b981", stroke: "#059669", text: "white" };
    if (step.processing === id) return { fill: "#f59e0b", stroke: "#d97706", text: "white" };
    if (step.queue.includes(id)) return { fill: "#6366f1", stroke: "#4f46e5", text: "white" };
    return { fill: "#e2e8f0", stroke: "#94a3b8", text: "#475569" };
  }

  const arrowColor = "#94a3b8";

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Kahn 算法——BFS 入度队列步进</h3>
        <p className="text-violet-200 text-sm mt-0.5">入度降为 0 就可学习——观察课程依赖图的拓扑展开过程</p>
      </div>

      <div className="p-5 space-y-4">
        {/* Legend */}
        <div className="flex flex-wrap gap-3 text-xs">
          {[
            { color: "bg-slate-200 dark:bg-slate-700", text: "white dark:text-slate-300", label: "待处理" },
            { color: "bg-indigo-500", text: "white", label: "队列中（可学习）" },
            { color: "bg-amber-500", text: "white", label: "正在处理" },
            { color: "bg-emerald-500", text: "white", label: "已完成" },
          ].map(({ color, text, label }) => (
            <div key={label} className="flex items-center gap-1.5">
              <div className={`w-3.5 h-3.5 rounded-full ${color}`} />
              <span className="text-slate-600 dark:text-slate-300">{label}</span>
            </div>
          ))}
        </div>

        {/* Main area */}
        <div className="flex gap-4 items-start">
          {/* Graph SVG */}
          <div className="flex-1 min-w-0 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 overflow-hidden">
            <svg viewBox="0 0 480 320" className="w-full">
              <defs>
                <marker id={`arr-${arrowColor.replace('#','')}`} markerWidth="6" markerHeight="6"
                  refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill={arrowColor} />
                </marker>
                <marker id="arr-10b981" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#10b981" />
                </marker>
              </defs>

              {EDGES.map(([u, v]) => {
                const fromDone = step.done[u], toDone = step.done[v];
                const color = fromDone && toDone ? "#10b981" : "#94a3b8";
                return (
                  <g key={`${u}-${v}`}>
                    <Arrow x1={NODES[u].x} y1={NODES[u].y} x2={NODES[v].x} y2={NODES[v].y} color={color} />
                  </g>
                );
              })}

              {NODES.map(node => {
                const { fill, stroke, text } = nodeColor(node.id);
                const isCurrent = step.processing === node.id;
                const inQueue = step.queue.includes(node.id);
                return (
                  <g key={node.id}>
                    {isCurrent && <circle cx={node.x} cy={node.y} r={30} fill="#f59e0b22" stroke="#f59e0b55" strokeWidth={2} />}
                    {inQueue && !isCurrent && <circle cx={node.x} cy={node.y} r={28} fill="#6366f111" stroke="#6366f144" strokeWidth={1.5} />}
                    <circle cx={node.x} cy={node.y} r={22} fill={fill} stroke={stroke} strokeWidth={2}
                      className="transition-all duration-300" />
                    {/* Indegree badge */}
                    <circle cx={node.x + 17} cy={node.y - 17} r={9}
                      fill={step.indegree[node.id] === 0 ? "#10b981" : "#f1f5f9"}
                      stroke={step.indegree[node.id] === 0 ? "#059669" : "#cbd5e1"} strokeWidth={1} />
                    <text x={node.x + 17} y={node.y - 17} textAnchor="middle" dominantBaseline="central"
                      fontSize={9} fontWeight="bold"
                      fill={step.indegree[node.id] === 0 ? "white" : "#64748b"}>
                      {step.indegree[node.id]}
                    </text>
                    <text x={node.x} y={node.y - 5} textAnchor="middle" fontSize={10} fontWeight="bold" fill={text}>
                      {node.label}
                    </text>
                    <text x={node.x} y={node.y + 8} textAnchor="middle" fontSize={8} fill={text} opacity={0.7}>
                      ({node.sublabel})
                    </text>
                  </g>
                );
              })}

              {/* Badge legend */}
              <text x={16} y={290} fontSize={9} fill="#94a3b8">右上角数字 = 当前入度</text>
              <circle cx={8} cy={287} r={5} fill="#10b981" />
              <text x={16} y={300} fontSize={9} fill="#94a3b8">绿色 = 入度为 0（可入队）</text>
            </svg>
          </div>

          {/* Right panel */}
          <div className="w-48 space-y-3 shrink-0">
            {/* Queue */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-indigo-100 dark:bg-indigo-900/40 px-3 py-1.5 text-xs font-bold text-indigo-700 dark:text-indigo-300">
                队列 Queue (FIFO)
              </div>
              <div className="p-2 min-h-[52px] bg-white dark:bg-slate-800/50">
                {step.queue.length === 0
                  ? <p className="text-xs text-slate-400 text-center mt-1">空队列</p>
                  : <div className="flex flex-wrap gap-1">
                      {step.queue.map((id, i) => (
                        <span key={i} className="px-2 py-0.5 rounded-lg bg-indigo-500 text-white text-[11px] font-bold">
                          {NODES[id].label}
                        </span>
                      ))}
                    </div>
                }
              </div>
            </div>

            {/* Indegree table */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-slate-100 dark:bg-slate-800 px-3 py-1.5 text-xs font-bold text-slate-600 dark:text-slate-300">
                入度表 indegree[]
              </div>
              <div className="divide-y divide-slate-100 dark:divide-slate-700/50">
                {NODES.map(node => (
                  <div key={node.id} className={`flex justify-between items-center px-3 py-1 text-xs transition-colors ${
                    step.processing === node.id ? "bg-amber-50 dark:bg-amber-900/20" :
                    step.done[node.id] ? "bg-emerald-50 dark:bg-emerald-900/10" : ""
                  }`}>
                    <span className="text-slate-600 dark:text-slate-300">{node.label}</span>
                    <span className={`font-mono font-bold px-1.5 py-0.5 rounded text-xs ${
                      step.indegree[node.id] === 0 && !step.done[node.id]
                        ? "bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-400"
                        : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300"
                    }`}>
                      {step.done[node.id] ? "✓" : step.indegree[node.id]}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Output */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-emerald-100 dark:bg-emerald-900/40 px-3 py-1.5 text-xs font-bold text-emerald-700 dark:text-emerald-300">
                拓扑序输出
              </div>
              <div className="p-2 min-h-[36px] bg-white dark:bg-slate-800/50">
                {step.output.length === 0
                  ? <p className="text-xs text-slate-400 text-center">（等待输出）</p>
                  : <div className="flex flex-wrap gap-1">
                      {step.output.map((id, i) => (
                        <span key={i} className="flex items-center gap-0.5">
                          {i > 0 && <span className="text-slate-400 text-xs">→</span>}
                          <span className="px-1.5 py-0.5 rounded bg-emerald-500 text-white text-[10px] font-bold">
                            {NODES[id].label}
                          </span>
                        </span>
                      ))}
                    </div>
                }
              </div>
            </div>
          </div>
        </div>

        {/* Description box */}
        <div className="rounded-xl bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-700/50 px-4 py-2.5 text-sm text-violet-800 dark:text-violet-300 min-h-[44px]">
          {step.desc}
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={reset}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            ↺ 重置
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.max(0, i - 1)); }} disabled={stepIdx === 0}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            ← 上一步
          </button>
          <button onClick={() => setPlaying(p => !p)} disabled={stepIdx === STEPS.length - 1}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-colors ${
              playing ? "bg-amber-500 hover:bg-amber-600 text-white" : "bg-violet-600 hover:bg-violet-700 text-white"
            }`}>
            {playing ? "⏸ 暂停" : stepIdx === 0 ? "▶ 开始" : "▶ 播放"}
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.min(STEPS.length - 1, i + 1)); }} disabled={stepIdx === STEPS.length - 1}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            下一步 →
          </button>
          <span className="text-xs text-slate-400 font-mono">步骤 {stepIdx + 1}/{STEPS.length}</span>
          <div className="flex items-center gap-1.5 ml-auto">
            <input type="range" min={400} max={1600} step={100} value={speed}
              onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-violet-500" />
            <span className="text-[10px] text-slate-400">{(speed / 1000).toFixed(1)}s</span>
          </div>
        </div>
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
          <div className="bg-violet-500 h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${((stepIdx) / (STEPS.length - 1)) * 100}%` }} />
        </div>
      </div>
    </div>
  );
}
