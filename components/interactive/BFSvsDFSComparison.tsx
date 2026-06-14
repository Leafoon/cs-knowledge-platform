"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── Shared graph ───────────────────────────────────────────────────────── */
interface GNode { id: number; x: number; y: number; label: string; }
const NODES: GNode[] = [
  { id: 0, x: 155, y: 28,  label: "0" },
  { id: 1, x: 80,  y: 100, label: "1" },
  { id: 2, x: 230, y: 100, label: "2" },
  { id: 3, x: 40,  y: 172, label: "3" },
  { id: 4, x: 120, y: 172, label: "4" },
  { id: 5, x: 230, y: 172, label: "5" },
  { id: 6, x: 230, y: 240, label: "6" },
];
const RAW_EDGES = [[0,1],[0,2],[1,3],[1,4],[2,5],[5,6]];
const ADJ: Record<number,number[]> = {};
NODES.forEach(n => { ADJ[n.id] = []; });
RAW_EDGES.forEach(([u,v]) => { ADJ[u].push(v); ADJ[v].push(u); });

/* ─── BFS order simulation ───────────────────────────────────────────────── */
function getBFSOrder(start: number): number[] {
  const visited = new Set<number>();
  const queue = [start]; visited.add(start);
  const order: number[] = [];
  while (queue.length) {
    const u = queue.shift()!; order.push(u);
    for (const v of ADJ[u]) { if (!visited.has(v)) { visited.add(v); queue.push(v); } }
  }
  return order;
}

function getDFSOrder(start: number): number[] {
  const visited = new Set<number>();
  const order: number[] = [];
  function dfs(u: number) {
    visited.add(u); order.push(u);
    for (const v of ADJ[u]) { if (!visited.has(v)) dfs(v); }
  }
  dfs(start);
  return order;
}

const BFS_ORDER = getBFSOrder(0);
const DFS_ORDER = getDFSOrder(0);

/* ─── Comparison dimensions ──────────────────────────────────────────────── */
interface Dimension {
  aspect: string;
  bfs: string;
  dfs: string;
  winner: "bfs" | "dfs" | "tie";
  detail: string;
}

const DIMENSIONS: Dimension[] = [
  { aspect: "核心数据结构", bfs: "队列（FIFO）", dfs: "栈（LIFO）", winner: "tie", detail: "BFS 用队列保证「近的先处理」；DFS 用栈（递归系统栈或显式栈）保证「深的先处理」。" },
  { aspect: "遍历特征", bfs: "按层扩展（波纹状）", dfs: "沿路径深入（先深后退）", winner: "tie", detail: "BFS 的队列+层序特性天然保证先处理距离小的节点；DFS 则会一路走到底。" },
  { aspect: "无权图最短路", bfs: "✅ 保证最短", dfs: "❌ 不保证", winner: "bfs", detail: "BFS 按层扩展，第一次到达即为最短；DFS 到达时路径可能绕远路。" },
  { aspect: "时间戳能力", bfs: "❌ 无", dfs: "✅ d[u], f[u]", winner: "dfs", detail: "DFS 的时间戳是拓扑排序、SCC 等高级算法的基础；BFS 没有完成时间概念。" },
  { aspect: "边的分类", bfs: "❌ 无法分类", dfs: "✅ 树/前向/后向/横跨", winner: "dfs", detail: "DFS 的三色标记+时间戳可精确区分四类边；BFS 缺少此能力。" },
  { aspect: "适合问题类型", bfs: "最短路、层序、多源扩散", dfs: "拓扑排序、SCC、环检测、回溯", winner: "tie", detail: "BFS 和 DFS 各有擅长场景，实际工程中两者都是必备工具。" },
  { aspect: "内存峰值（宽图）", bfs: "⚠️ 队列可能很大", dfs: "✅ 栈相对小", winner: "dfs", detail: "宽图（BFS tree 宽）时，BFS 队列可能同时存储整层节点；DFS 栈深度仅为树高。" },
  { aspect: "内存峰值（深图）", bfs: "✅ 队列相对小", dfs: "⚠️ 栈深可能溢出", winner: "bfs", detail: "路径链式图（DFS tree 深）时，DFS 递归栈深度 O(V)，Python 可能 RecursionError。" },
];

/* ─── Graph panel ────────────────────────────────────────────────────────── */
function GraphPanel({
  title, color, headerStyle, order, stepIdx,
}: {
  title: string; color: string; headerStyle: string;
  order: number[]; stepIdx: number;
}) {
  const visitedAt: Record<number, number> = {};
  order.forEach((nid, i) => { visitedAt[nid] = i; });

  function nodeStyle(id: number) {
    const rank = visitedAt[id];
    if (rank === undefined) return { fill: "#94a3b8", stroke: "#64748b" };
    if (rank > stepIdx) return { fill: "#94a3b8", stroke: "#64748b" };
    if (rank === stepIdx) return { fill: "#f59e0b", stroke: "#d97706" };
    return { fill: color, stroke: color === "#0ea5e9" ? "#0284c7" : "#059669" };
  }

  return (
    <div className="flex-1 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
      <div className={`px-3 py-2 ${headerStyle}`}>
        <div className="text-white font-bold text-sm">{title}</div>
      </div>
      <div className="bg-slate-50 dark:bg-slate-800/50 p-1">
        <svg viewBox="0 0 310 270" className="w-full">
          {RAW_EDGES.map(([u, v]) => (
            <line key={`${u}-${v}`}
              x1={NODES[u].x} y1={NODES[u].y} x2={NODES[v].x} y2={NODES[v].y}
              stroke="#cbd5e1" strokeWidth={1.5} strokeDasharray="4 3" />
          ))}
          {NODES.map(node => {
            const { fill, stroke } = nodeStyle(node.id);
            const rank = visitedAt[node.id];
            const visited = rank !== undefined && rank <= stepIdx;
            const current = rank === stepIdx;
            return (
              <g key={node.id}>
                {current && <circle cx={node.x} cy={node.y} r={23} fill={color + "33"} stroke={color + "88"} strokeWidth={1.5} />}
                <circle cx={node.x} cy={node.y} r={18}
                  fill={fill} stroke={stroke} strokeWidth={2}
                  className="transition-all duration-300" />
                <text x={node.x} y={node.y} textAnchor="middle" dominantBaseline="central"
                  fontSize={13} fontWeight="bold" fill="white">{node.label}</text>
                {visited && (
                  <text x={node.x + 15} y={node.y - 15} fontSize={9} fontWeight="bold" fill={color}>
                    #{rank + 1}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
      {/* Order display */}
      <div className="px-3 pb-3 bg-white dark:bg-slate-900">
        <div className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase mb-1.5">遍历顺序</div>
        <div className="flex flex-wrap gap-1">
          {order.map((nid, i) => (
            <span key={i} className={`px-2 py-0.5 rounded text-[10px] font-bold font-mono transition-all border ${
              i <= stepIdx
                ? i === stepIdx ? "text-white" : "text-white opacity-70"
                : "opacity-30 text-slate-500"
            }`}
              style={{
                backgroundColor: i <= stepIdx ? (i === stepIdx ? "#f59e0b" : color) : "transparent",
                borderColor: i <= stepIdx ? color : "#d1d5db",
              }}>
              {NODES[nid].label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ─── Main Component ─────────────────────────────────────────────────────── */
export default function BFSvsDFSComparison() {
  const [stepIdx, setStepIdx] = useState(-1);  // -1 = not started
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(800);
  const [activeRow, setActiveRow] = useState<number | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const maxStep = NODES.length - 1;

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

  const reset = () => { setPlaying(false); setStepIdx(-1); };

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-700 via-slate-600 to-slate-700 dark:from-slate-800 dark:via-slate-700 dark:to-slate-800 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">BFS vs DFS——同图遍历顺序对比</h3>
        <p className="text-slate-300 text-sm mt-0.5">同一张图，两种遍历顺序截然不同——感受波纹式与深挖式的本质区別</p>
      </div>

      <div className="p-5 space-y-5">
        {/* Dual graph panels */}
        <div className="flex gap-3">
          <GraphPanel title="BFS（广度优先）" color="#0ea5e9" headerStyle="bg-sky-500"
            order={BFS_ORDER} stepIdx={stepIdx} />
          <GraphPanel title="DFS（深度优先）" color="#10b981" headerStyle="bg-emerald-500"
            order={DFS_ORDER} stepIdx={stepIdx} />
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={reset}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            ↺ 重置
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.max(-1, i - 1)); }} disabled={stepIdx < 0}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            ← 上一步
          </button>
          <button onClick={() => setPlaying(p => !p)}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-colors ${playing ? "bg-amber-500 hover:bg-amber-600 text-white" : "bg-slate-700 hover:bg-slate-600 dark:bg-slate-600 dark:hover:bg-slate-500 text-white"}`}>
            {playing ? "⏸ 暂停" : stepIdx < 0 ? "▶ 开始" : "▶ 播放"}
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.min(maxStep, i + 1)); }} disabled={stepIdx === maxStep}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            下一步 →
          </button>
          <div className="flex items-center gap-2 ml-auto">
            <input type="range" min={300} max={1500} step={100} value={speed}
              onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-slate-500" />
            <span className="text-[10px] text-slate-400">{(speed/1000).toFixed(1)}s</span>
          </div>
        </div>
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
          <div className="bg-slate-600 dark:bg-slate-400 h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${stepIdx < 0 ? 0 : ((stepIdx + 1) / NODES.length) * 100}%` }} />
        </div>

        {/* Comparison table */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
          <div className="grid grid-cols-[minmax(0,1.2fr)_1fr_1fr] text-[11px] font-bold bg-slate-100 dark:bg-slate-800">
            <div className="px-3 py-2 text-slate-600 dark:text-slate-300">维度</div>
            <div className="px-3 py-2 text-sky-600 dark:text-sky-400">BFS</div>
            <div className="px-3 py-2 text-emerald-600 dark:text-emerald-400">DFS</div>
          </div>
          {DIMENSIONS.map((d, i) => (
            <div key={i}>
              <div
                onClick={() => setActiveRow(prev => prev === i ? null : i)}
                className={`grid grid-cols-[minmax(0,1.2fr)_1fr_1fr] text-[11px] cursor-pointer border-t border-slate-100 dark:border-slate-800 transition-colors ${
                  activeRow === i ? "bg-slate-50 dark:bg-slate-800/60 ring-1 ring-inset ring-slate-300 dark:ring-slate-600" : "hover:bg-slate-50 dark:hover:bg-slate-800/30"
                }`}>
                <div className="px-3 py-2 font-semibold text-slate-700 dark:text-slate-200">{d.aspect}</div>
                <div className={`px-3 py-2 ${d.winner === "bfs" ? "text-sky-600 dark:text-sky-400 font-bold" : "text-slate-500 dark:text-slate-400"}`}>{d.bfs}</div>
                <div className={`px-3 py-2 ${d.winner === "dfs" ? "text-emerald-600 dark:text-emerald-400 font-bold" : "text-slate-500 dark:text-slate-400"}`}>{d.dfs}</div>
              </div>
              {activeRow === i && (
                <div className="px-4 py-2 text-[11px] text-slate-600 dark:text-slate-300 bg-slate-50 dark:bg-slate-800/60 border-t border-slate-100 dark:border-slate-800 leading-relaxed">
                  {d.detail}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
