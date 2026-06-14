"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── Graph Data ─────────────────────────────────────────────────────────── */

interface Node {
  id: number;
  x: number;
  y: number;
  label: string;
}

interface Edge {
  u: number;
  v: number;
}

const NODES: Node[] = [
  { id: 0, x: 165, y: 42,  label: "s" },   // source
  { id: 1, x: 66,  y: 120, label: "A" },
  { id: 2, x: 165, y: 120, label: "B" },
  { id: 3, x: 264, y: 120, label: "C" },
  { id: 4, x: 26,  y: 200, label: "D" },
  { id: 5, x: 110, y: 200, label: "E" },
  { id: 6, x: 200, y: 200, label: "F" },
  { id: 7, x: 300, y: 200, label: "G" },
];

const EDGES: Edge[] = [
  { u: 0, v: 1 }, { u: 0, v: 2 }, { u: 0, v: 3 },
  { u: 1, v: 4 }, { u: 1, v: 5 },
  { u: 2, v: 6 },
  { u: 3, v: 7 },
];

const ADJ: Record<number, number[]> = {};
NODES.forEach(n => { ADJ[n.id] = []; });
EDGES.forEach(({ u, v }) => {
  ADJ[u].push(v);
  ADJ[v].push(u);
});

/* ─── BFS Simulation ─────────────────────────────────────────────────────── */

type NodeState = "unvisited" | "queued" | "processing" | "done";

interface BFSStep {
  nodeStates: NodeState[];    // per node
  dist: (number | null)[];
  queue: number[];            // current queue contents
  current: number | null;     // node being dequeued right now
  treeEdges: [number, number][];
  description: string;
}

function computeBFSSteps(source: number): BFSStep[] {
  const n = NODES.length;
  const dist: (number | null)[] = Array(n).fill(null);
  const states: NodeState[] = Array(n).fill("unvisited");
  const treeEdges: [number, number][] = [];
  const steps: BFSStep[] = [];
  const queue: number[] = [];

  const snapshot = (current: number | null, desc: string): void => {
    steps.push({
      nodeStates: [...states] as NodeState[],
      dist: [...dist],
      queue: [...queue],
      current,
      treeEdges: treeEdges.map(e => [...e] as [number, number]),
      description: desc,
    });
  };

  // Initial state
  snapshot(null, "初始状态：所有节点未访问（白色）");

  // Enqueue source
  dist[source] = 0;
  states[source] = "queued";
  queue.push(source);
  snapshot(null, `将起点 s(0) 入队，dist[s]=0，标记为灰色（已发现）`);

  while (queue.length > 0) {
    const u = queue.shift()!;
    states[u] = "processing";
    snapshot(u, `出队节点 ${NODES[u].label}，dist=${dist[u]}，开始处理其邻居`);

    for (const v of ADJ[u]) {
      if (states[v] === "unvisited") {
        states[v] = "queued";
        dist[v] = dist[u]! + 1;
        treeEdges.push([u, v]);
        queue.push(v);
        snapshot(u, `发现邻居 ${NODES[v].label}（白色→灰色），dist[${NODES[v].label}]=${dist[v]}，入队`);
      }
    }

    states[u] = "done";
    snapshot(null, `节点 ${NODES[u].label} 处理完毕，变为黑色（已完成）`);
  }

  snapshot(null, "BFS 完成！所有可达节点均已处理，最短距离已确定。");
  return steps;
}

const STEPS = computeBFSSteps(0);

/* ─── Color mappings ─────────────────────────────────────────────────────── */

function nodeColor(state: NodeState): { fill: string; stroke: string; text: string } {
  switch (state) {
    case "unvisited":   return { fill: "#94a3b8", stroke: "#64748b", text: "#fff" };
    case "queued":      return { fill: "#6366f1", stroke: "#4338ca", text: "#fff" };
    case "processing":  return { fill: "#f59e0b", stroke: "#d97706", text: "#fff" };
    case "done":        return { fill: "#10b981", stroke: "#059669", text: "#fff" };
  }
}

function layerColor(dist: number | null): string {
  if (dist === null) return "text-slate-400 dark:text-slate-500";
  return ["text-indigo-600 dark:text-indigo-400", "text-sky-600 dark:text-sky-400", "text-emerald-600 dark:text-emerald-400", "text-amber-600 dark:text-amber-400"][dist] ?? "text-slate-500";
}

/* ─── Main Component ─────────────────────────────────────────────────────── */

export default function BFSLevelExpansion() {
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(900);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const step = STEPS[stepIdx];
  const maxStep = STEPS.length - 1;

  const clearTimer = () => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
  };

  const advance = useCallback(() => {
    setStepIdx(prev => {
      if (prev >= maxStep) { setPlaying(false); return prev; }
      return prev + 1;
    });
  }, [maxStep]);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(advance, speed);
    } else {
      clearTimer();
    }
    return clearTimer;
  }, [playing, speed, advance]);

  const toggle = () => setPlaying(p => !p);
  const reset = () => { setPlaying(false); setStepIdx(0); };
  const prev = () => { setPlaying(false); setStepIdx(i => Math.max(0, i - 1)); };
  const next = () => { setPlaying(false); setStepIdx(i => Math.min(maxStep, i + 1)); };

  const isTreeEdge = (u: number, v: number) =>
    step.treeEdges.some(([a, b]) => (a === u && b === v) || (a === v && b === u));

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">

      {/* Header */}
      <div className="bg-gradient-to-r from-sky-500 via-blue-500 to-indigo-500 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">BFS 层级扩展动画</h3>
        <p className="text-sky-100 text-sm mt-0.5">观察广度优先搜索如何逐层扩展，队列状态与颜色实时变化</p>
      </div>

      <div className="p-5 space-y-4">
        {/* Graph + Queue side by side */}
        <div className="flex flex-col sm:flex-row gap-4">
          {/* SVG Graph */}
          <div className="flex-1 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-2">
            <svg viewBox="0 0 330 240" className="w-full">
              {/* Edges */}
              {EDGES.map(({ u, v }) => {
                const nu = NODES[u], nv = NODES[v];
                const tree = isTreeEdge(u, v);
                return (
                  <line
                    key={`${u}-${v}`}
                    x1={nu.x} y1={nu.y} x2={nv.x} y2={nv.y}
                    stroke={tree ? "#6366f1" : "#cbd5e1"}
                    strokeWidth={tree ? 2.5 : 1.5}
                    strokeDasharray={tree ? "none" : "4 3"}
                    className="transition-all duration-300"
                  />
                );
              })}
              {/* Nodes */}
              {NODES.map(node => {
                const { fill, stroke, text } = nodeColor(step.nodeStates[node.id]);
                const isCurrent = step.current === node.id;
                return (
                  <g key={node.id} className="transition-all duration-300">
                    {isCurrent && (
                      <circle cx={node.x} cy={node.y} r={22} fill="#fef3c7" stroke="#f59e0b" strokeWidth={2} opacity={0.6} />
                    )}
                    <circle
                      cx={node.x} cy={node.y} r={18}
                      fill={fill} stroke={stroke} strokeWidth={2}
                      className="transition-all duration-300"
                    />
                    <text x={node.x} y={node.y} textAnchor="middle" dominantBaseline="central"
                      fontSize={12} fontWeight="bold" fill={text}>{node.label}</text>
                    {/* Distance badge */}
                    {step.dist[node.id] !== null && (
                      <text x={node.x + 14} y={node.y - 14}
                        textAnchor="middle" fontSize={9} fontWeight="bold"
                        fill="#6366f1">{step.dist[node.id]}</text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Queue + Info panel */}
          <div className="sm:w-48 space-y-3">
            {/* Queue visualization */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-3">
              <div className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase mb-2 tracking-wide">
                队列 Queue
              </div>
              {step.queue.length === 0 ? (
                <div className="text-[11px] text-slate-400 dark:text-slate-500 italic py-1">空队列</div>
              ) : (
                <div className="flex flex-col gap-1">
                  {step.queue.map((nid, qi) => (
                    <div key={qi}
                      className="flex items-center gap-1.5 text-[11px] bg-indigo-50 dark:bg-indigo-900/30 border border-indigo-200 dark:border-indigo-700 rounded-lg px-2 py-1">
                      <span className="text-indigo-400 text-[9px] font-mono w-4">{qi === 0 ? "←" : ""}</span>
                      <span className="font-bold text-indigo-700 dark:text-indigo-300">{NODES[nid].label}</span>
                      <span className="ml-auto font-mono text-indigo-500 dark:text-indigo-400">d={step.dist[nid]}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Distance table */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-3">
              <div className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase mb-2 tracking-wide">
                最短距离 dist[]
              </div>
              <div className="grid grid-cols-4 gap-1">
                {NODES.map(node => (
                  <div key={node.id} className="text-center">
                    <div className="text-[9px] text-slate-500 dark:text-slate-400">{node.label}</div>
                    <div className={`text-[11px] font-bold font-mono ${layerColor(step.dist[node.id])}`}>
                      {step.dist[node.id] ?? "∞"}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Description */}
        <div className="rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 px-4 py-3">
          <div className="text-xs text-blue-700 dark:text-blue-300 leading-relaxed">
            <span className="font-bold mr-1">步骤 {stepIdx}/{maxStep}：</span>
            {step.description}
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={reset}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            ↺ 重置
          </button>
          <button onClick={prev} disabled={stepIdx === 0}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-40 transition-colors">
            ← 上一步
          </button>
          <button onClick={toggle}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-colors ${playing ? "bg-amber-500 hover:bg-amber-600 text-white" : "bg-blue-500 hover:bg-blue-600 text-white"}`}>
            {playing ? "⏸ 暂停" : "▶ 播放"}
          </button>
          <button onClick={next} disabled={stepIdx === maxStep}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-40 transition-colors">
            下一步 →
          </button>
          <div className="flex items-center gap-2 ml-auto">
            <span className="text-[10px] text-slate-500 dark:text-slate-400">速度</span>
            <input type="range" min={300} max={1500} step={100} value={speed}
              onChange={e => setSpeed(Number(e.target.value))}
              className="w-20 accent-blue-500" />
            <span className="text-[10px] text-slate-500 dark:text-slate-400 w-10">{(speed / 1000).toFixed(1)}s</span>
          </div>
        </div>

        {/* Progress */}
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
          <div className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${(stepIdx / maxStep) * 100}%` }} />
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-3 text-[10px]">
          {[
            { color: "bg-slate-400", label: "未访问（白色）" },
            { color: "bg-indigo-500", label: "已入队（灰色）" },
            { color: "bg-amber-500", label: "正在处理" },
            { color: "bg-emerald-500", label: "已完成（黑色）" },
          ].map(({ color, label }) => (
            <div key={label} className="flex items-center gap-1.5">
              <span className={`inline-block w-3 h-3 rounded-full ${color}`} />
              <span className="text-slate-500 dark:text-slate-400">{label}</span>
            </div>
          ))}
          <div className="flex items-center gap-1.5">
            <span className="inline-block w-5 h-0.5 bg-indigo-500 rounded" />
            <span className="text-slate-500 dark:text-slate-400">BFS 树边</span>
          </div>
        </div>
      </div>
    </div>
  );
}
