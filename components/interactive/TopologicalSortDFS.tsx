"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── Graph (same dependency graph as Kahn demo) ────────────────────────── */
//
//  高数(0) ──→ 概率(2) ──→ ML(4) ──→ DL(5)
//     └──→ 统计(3) ──────────↑
//  线代(1) ──→ 概率(2)
//
interface GNode { id: number; x: number; y: number; label: string; }
const NODES: GNode[] = [
  { id: 0, x: 55,  y: 60,  label: "高数" },
  { id: 1, x: 55,  y: 160, label: "线代" },
  { id: 2, x: 175, y: 110, label: "概率" },
  { id: 3, x: 55,  y: 260, label: "统计" },
  { id: 4, x: 295, y: 110, label: "ML" },
  { id: 5, x: 415, y: 110, label: "DL" },
];
const EDGES = [[0,2],[0,3],[1,2],[2,4],[3,4],[4,5]];
const N = NODES.length;

type Color = "white" | "gray" | "black";

interface DFSStep {
  colors: Color[];
  callStack: number[];       // 当前 DFS 递归调用栈（灰色节点）
  finishStack: number[];     // 完成栈（黑色，已压入）
  topoOrder: number[];       // 最终拓扑序（finish stack 逆序）
  activeEdge?: [number, number];
  desc: string;
  phase: "prolog" | "dfs" | "done";
}

/* ─── Precompute DFS steps ───────────────────────────────────────────────── */
function buildDFSSteps(): DFSStep[] {
  const steps: DFSStep[] = [];
  const colors: Color[] = Array(N).fill("white") as Color[];
  const callStack: number[] = [];
  const finishStack: number[] = [];

  function snap(desc: string, activeEdge?: [number, number], phase: "prolog"|"dfs"|"done" = "dfs") {
    steps.push({
      colors: [...colors] as Color[],
      callStack: [...callStack],
      finishStack: [...finishStack],
      topoOrder: [],
      activeEdge,
      desc,
      phase,
    });
  }

  snap("初始状态：所有节点为白色（未访问）。将按节点编号顺序启动 DFS。", undefined, "prolog");

  const adj: number[][] = Array.from({ length: N }, () => []);
  EDGES.forEach(([u, v]) => adj[u].push(v));

  function dfs(u: number) {
    colors[u] = "gray";
    callStack.push(u);
    snap(`进入节点「${NODES[u].label}」（标记为灰色，加入调用栈）`);

    for (const v of adj[u]) {
      snap(`检查边 ${NODES[u].label} → ${NODES[v].label}：邻居「${NODES[v].label}」${
        colors[v] === "white" ? "是白色，开始递归访问" :
        colors[v] === "gray" ? "是灰色（发现后向边！有向环！此图无环故不会出现）" :
        "已是黑色（已完成），跳过"
      }`, [u, v]);
      if (colors[v] === "white") dfs(v);
    }

    colors[u] = "black";
    callStack.pop();
    finishStack.push(u);
    snap(`完成「${NODES[u].label}」（标记黑色，pressure 入完成栈）→ 完成栈: [${finishStack.map(i => NODES[i].label).join(", ")}]`);
  }

  for (let u = 0; u < N; u++) {
    if (colors[u] === "white") {
      snap(`节点「${NODES[u].label}」未访问，以它为起点启动新一轮 DFS`);
      dfs(u);
    }
  }

  // Final: show reversed finish stack = topo order
  const topoOrder = [...finishStack].reverse();
  steps.push({
    colors: Array(N).fill("black") as Color[],
    callStack: [],
    finishStack: [...finishStack],
    topoOrder,
    desc: `DFS 全部完成！将完成栈逆序读取，得到拓扑序：${topoOrder.map(i => NODES[i].label).join(" → ")}`,
    phase: "done",
  });

  return steps;
}

const STEPS = buildDFSSteps();

/* ─── Main Component ─────────────────────────────────────────────────────── */
export default function TopologicalSortDFS() {
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(900);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const step = STEPS[stepIdx];

  const advance = useCallback(() => {
    setStepIdx(prev => { if (prev >= STEPS.length - 1) { setPlaying(false); return prev; } return prev + 1; });
  }, []);

  useEffect(() => {
    if (playing) intervalRef.current = setInterval(advance, speed);
    else { if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; } }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, speed, advance]);

  const reset = () => { setPlaying(false); setStepIdx(0); };

  function nodeStyle(id: number) {
    const c = step.colors[id];
    const isActive = step.callStack[step.callStack.length - 1] === id;
    if (c === "black") return { fill: "#10b981", stroke: "#059669", label: "white" };
    if (c === "gray") return { fill: isActive ? "#f59e0b" : "#8b5cf6", stroke: isActive ? "#d97706" : "#7c3aed", label: "white" };
    return { fill: "#f1f5f9", stroke: "#94a3b8", label: "#475569" };
  }

  function edgeStyle(u: number, v: number) {
    const isActive = step.activeEdge && step.activeEdge[0] === u && step.activeEdge[1] === v;
    const fromBlack = step.colors[u] === "black";
    const toBlack = step.colors[v] === "black";
    if (isActive) return { stroke: "#f59e0b", width: 2.5 };
    if (fromBlack && toBlack) return { stroke: "#10b981", width: 1.8 };
    return { stroke: "#cbd5e1", width: 1.5 };
  }

  const colorMap: Record<Color, string> = {
    white: "bg-slate-200 dark:bg-slate-600",
    gray: "bg-purple-500",
    black: "bg-emerald-500",
  };
  const colorLabelMap: Record<Color, string> = { white: "白色", gray: "灰色（访问中）", black: "黑色（已完成）" };

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      <div className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">DFS 逆后序——拓扑排序步进动画</h3>
        <p className="text-emerald-100 text-sm mt-0.5">深度优先探索完成后，逆序读取完成栈即为拓扑序</p>
      </div>

      <div className="p-5 space-y-4">
        {/* Legend */}
        <div className="flex flex-wrap gap-3 text-xs">
          {([["white","bg-slate-200 dark:bg-slate-600","text-slate-600 dark:text-slate-300","白色：未访问"],
             ["gray","bg-purple-500","text-white","灰色：访问中（在调用栈上）"],
             ["","bg-amber-500","text-white","当前节点（栈顶）"],
             ["black","bg-emerald-500","text-white","黑色：完成，已入栈"]] as [string,string,string,string][]).map(([, bg, , label]) => (
            <div key={label} className="flex items-center gap-1.5">
              <div className={`w-3 h-3 rounded-full ${bg}`} />
              <span className="text-slate-600 dark:text-slate-300">{label}</span>
            </div>
          ))}
        </div>

        <div className="flex gap-4 items-start">
          {/* SVG Graph */}
          <div className="flex-1 min-w-0 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 overflow-hidden">
            <svg viewBox="0 0 480 320" className="w-full">
              <defs>
                <marker id="dfs-arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8" />
                </marker>
                <marker id="dfs-arr-active" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#f59e0b" />
                </marker>
                <marker id="dfs-arr-done" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#10b981" />
                </marker>
              </defs>

              {EDGES.map(([u, v]) => {
                const { stroke, width } = edgeStyle(u, v);
                const dx = NODES[v].x - NODES[u].x, dy = NODES[v].y - NODES[u].y;
                const len = Math.sqrt(dx * dx + dy * dy);
                const R = 22;
                const isActive = step.activeEdge && step.activeEdge[0] === u && step.activeEdge[1] === v;
                const mkId = isActive ? "dfs-arr-active" : (step.colors[u] === "black" && step.colors[v] === "black" ? "dfs-arr-done" : "dfs-arr");
                return (
                  <line key={`${u}-${v}`}
                    x1={NODES[u].x + (dx/len)*R} y1={NODES[u].y + (dy/len)*R}
                    x2={NODES[v].x - (dx/len)*R} y2={NODES[v].y - (dy/len)*R}
                    stroke={stroke} strokeWidth={width} markerEnd={`url(#${mkId})`}
                    className="transition-all duration-300" />
                );
              })}

              {NODES.map(node => {
                const { fill, stroke, label } = nodeStyle(node.id);
                const isStackTop = step.callStack[step.callStack.length - 1] === node.id;
                const stackPos = step.callStack.indexOf(node.id);
                return (
                  <g key={node.id}>
                    {isStackTop && <circle cx={node.x} cy={node.y} r={30} fill="#f59e0b22" stroke="#f59e0b77" strokeWidth={2} />}
                    <circle cx={node.x} cy={node.y} r={22} fill={fill} stroke={stroke} strokeWidth={2.5}
                      className="transition-all duration-300" />
                    <text x={node.x} y={node.y} textAnchor="middle" dominantBaseline="central"
                      fontSize={11} fontWeight="bold" fill={label}>{node.label}</text>
                    {stackPos >= 0 && (
                      <text x={node.x - 26} y={node.y - 20} fontSize={9} fill="#8b5cf6" fontWeight="bold">
                        [{stackPos}]
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Right panels */}
          <div className="w-44 space-y-3 shrink-0">
            {/* DFS Call Stack */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-purple-100 dark:bg-purple-900/40 px-3 py-1.5 text-xs font-bold text-purple-700 dark:text-purple-300">
                调用栈（灰色节点）
              </div>
              <div className="p-2 min-h-[60px] bg-white dark:bg-slate-800/50 space-y-1">
                {step.callStack.length === 0
                  ? <p className="text-xs text-slate-400 text-center">空</p>
                  : [...step.callStack].reverse().map((id, i) => (
                      <div key={i} className={`w-full rounded px-2 py-1 text-xs font-bold text-center transition-all ${
                        i === 0 ? "bg-amber-500 text-white" : "bg-purple-500 text-white opacity-80"
                      }`}>
                        {i === 0 ? "▶ " : ""}{NODES[id].label}
                      </div>
                    ))
                }
              </div>
            </div>

            {/* Finish Stack */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-emerald-100 dark:bg-emerald-900/40 px-3 py-1.5 text-xs font-bold text-emerald-700 dark:text-emerald-300">
                完成栈（逆序=拓扑序）
              </div>
              <div className="p-2 min-h-[60px] bg-white dark:bg-slate-800/50 space-y-1">
                {step.finishStack.length === 0
                  ? <p className="text-xs text-slate-400 text-center">空</p>
                  : [...step.finishStack].reverse().map((id, i) => (
                      <div key={i} className="w-full rounded px-2 py-1 text-xs font-bold bg-emerald-500 text-white text-center">
                        {NODES[id].label}
                      </div>
                    ))
                }
              </div>
            </div>

            {/* Colors */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-slate-100 dark:bg-slate-800 px-3 py-1.5 text-xs font-bold text-slate-600 dark:text-slate-300">
                节点颜色
              </div>
              <div className="divide-y divide-slate-100 dark:divide-slate-700/50">
                {NODES.map(node => (
                  <div key={node.id} className="flex justify-between items-center px-3 py-1 text-xs">
                    <span className="text-slate-600 dark:text-slate-300">{node.label}</span>
                    <span className={`w-2.5 h-2.5 rounded-full ${colorMap[step.colors[node.id]]}`} />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Final topo order */}
        {step.phase === "done" && (
          <div className="rounded-xl bg-emerald-50 dark:bg-emerald-900/25 border border-emerald-300 dark:border-emerald-700/60 p-3">
            <div className="text-xs font-bold text-emerald-700 dark:text-emerald-400 mb-2">
              ✅ 拓扑序（完成栈逆序）：
            </div>
            <div className="flex flex-wrap items-center gap-1">
              {step.topoOrder.map((id, i) => (
                <span key={i} className="flex items-center gap-1">
                  {i > 0 && <span className="text-emerald-600 dark:text-emerald-400 font-bold">→</span>}
                  <span className="px-2 py-0.5 rounded-lg bg-emerald-500 text-white text-xs font-bold">{NODES[id].label}</span>
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Description */}
        <div className="rounded-xl bg-teal-50 dark:bg-teal-900/20 border border-teal-200 dark:border-teal-700/50 px-4 py-2.5 text-sm text-teal-800 dark:text-teal-300 min-h-[44px]">
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
            className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-colors ${playing ? "bg-amber-500 hover:bg-amber-600 text-white" : "bg-teal-600 hover:bg-teal-700 text-white"}`}>
            {playing ? "⏸ 暂停" : stepIdx === 0 ? "▶ 开始" : "▶ 播放"}
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.min(STEPS.length - 1, i + 1)); }} disabled={stepIdx === STEPS.length - 1}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            下一步 →
          </button>
          <span className="text-xs text-slate-400 font-mono ml-1">{stepIdx + 1}/{STEPS.length}</span>
          <div className="flex items-center gap-1.5 ml-auto">
            <input type="range" min={400} max={1600} step={100} value={speed}
              onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-teal-500" />
            <span className="text-[10px] text-slate-400">{(speed / 1000).toFixed(1)}s</span>
          </div>
        </div>
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
          <div className="bg-teal-500 h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${(stepIdx / (STEPS.length - 1)) * 100}%` }} />
        </div>
      </div>
    </div>
  );
}
