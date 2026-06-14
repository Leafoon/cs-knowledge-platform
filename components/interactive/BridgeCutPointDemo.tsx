"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── Undirected Graph ───────────────────────────────────────────────────────
 *
 *  0 ──────── 1 ──── 2
 *   (Bridge)  │      │
 *             3 ─────┘   <- 1,2,3 form a triangle (biconnected)
 *             │
 *        (Bridge)
 *             │
 *             4 ──── 5
 *              \     │
 *               6 ──┘    <- 4,5,6 form a triangle (biconnected)
 *
 *  Bridges   : (0,1), (3,4)
 *  Cut points: 1, 3, 4
 * ─────────────────────────────────────────────────────────────────────────── */
const N = 7;
const NODES = [
  { id: 0, x: 55,  y: 100, label: "0" },
  { id: 1, x: 155, y: 100, label: "1" },
  { id: 2, x: 225, y: 52,  label: "2" },
  { id: 3, x: 225, y: 148, label: "3" },
  { id: 4, x: 325, y: 148, label: "4" },
  { id: 5, x: 395, y: 95,  label: "5" },
  { id: 6, x: 395, y: 195, label: "6" },
];

// Undirected adjacency
const ADJ: number[][] = [
  [1],          // 0
  [0, 2, 3],    // 1
  [1, 3],       // 2
  [1, 2, 4],    // 3
  [3, 5, 6],    // 4
  [4, 6],       // 5
  [4, 5],       // 6
];
const UNDIRECTED_EDGES: [number, number][] =
  [[0,1],[1,2],[1,3],[2,3],[3,4],[4,5],[4,6],[5,6]];

/* ─── Precompute Steps ───────────────────────────────────────────────────── */
interface BridgeStep {
  disc: number[];      // -1 = unvisited
  low: number[];
  parent: number[];    // -1 = no parent
  colors: string[];    // "white"|"gray"|"black"
  bridges: [number,number][];
  cutPoints: Set<number>;
  activeEdge?: [number, number];
  activeNode: number;
  desc: string;
  childCount: number[]; // DFS tree child count (for root cut-point check)
}

function buildBridgeSteps(): BridgeStep[] {
  const steps: BridgeStep[] = [];
  const disc = Array(N).fill(-1);
  const low  = Array(N).fill(-1);
  const parent = Array(N).fill(-1);
  const colors: string[] = Array(N).fill("white");
  const bridges: [number,number][] = [];
  const cutPoints: Set<number> = new Set();
  const childCount = Array(N).fill(0); // DFS-tree children count per node
  let timer = 0;

  function snap(desc: string, ae?: [number,number], activeNode = -1) {
    steps.push({
      disc: [...disc], low: [...low], parent: [...parent],
      colors: [...colors], bridges: bridges.map(b=>[...b] as [number,number]),
      cutPoints: new Set(cutPoints), activeEdge: ae, activeNode,
      desc, childCount: [...childCount],
    });
  }

  snap("开始 Tarjan 桥/关节点检测。初始化 disc[] 与 low[] 全为 -1。");

  function dfs(u: number) {
    disc[u] = low[u] = timer++;
    colors[u] = "gray";
    snap(`进入节点 ${u}：disc[${u}] = low[${u}] = ${disc[u]}`, undefined, u);

    for (const v of ADJ[u]) {
      const ae: [number,number] = [u, v];
      if (disc[v] === -1) {
        parent[v] = u;
        childCount[u]++;
        snap(`检查边 (${u}, ${v})：${v} 未访问 → 递归进入，child_count[${u}]=${childCount[u]}`, ae, u);
        dfs(v);
        const oldLow = low[u];
        low[u] = Math.min(low[u], low[v]);
        snap(`从 ${v} 回到 ${u}：low[${u}] = min(${oldLow}, low[${v}]=${low[v]}) = ${low[u]}`, ae, u);

        // Bridge check
        if (low[v] > disc[u]) {
          bridges.push([u, v]);
          snap(`🌉 low[${v}]=${low[v]} > disc[${u}]=${disc[u]}：边 (${u},${v}) 是**桥**！`, ae, u);
        }

        // Cut point check
        if (parent[u] === -1) {
          // Root: cut point if ≥2 DFS-tree children
          if (childCount[u] >= 2) {
            cutPoints.add(u);
            snap(`🔴 节点 ${u} 是根节点且有 ${childCount[u]} 个子树 → 关节点`, ae, u);
          }
        } else {
          // Non-root: cut point if low[v] >= disc[u]
          if (low[v] >= disc[u] && !cutPoints.has(u)) {
            cutPoints.add(u);
            snap(`🔴 非根节点 ${u}：low[${v}]=${low[v]} ≥ disc[${u}]=${disc[u]} → 关节点`, ae, u);
          }
        }
      } else if (v !== parent[u]) {
        // Back edge
        const oldLow = low[u];
        low[u] = Math.min(low[u], disc[v]);
        if (low[u] < oldLow) {
          snap(`检查边 (${u},${v})：${v} 已访问（非父节点），发现返祖边 → low[${u}] = min(${oldLow}, disc[${v}]=${disc[v]}) = ${low[u]}`, ae, u);
        } else {
          snap(`检查边 (${u},${v})：${v} 已访问（返祖边），low[${u}] 不更新 = ${low[u]}`, ae, u);
        }
      } else {
        snap(`检查边 (${u},${v})：${v} 是父节点，忽略（避免无向图双向更新）`, ae, u);
      }
    }

    colors[u] = "black";
    snap(`完成节点 ${u}：disc=${disc[u]}, low=${low[u]}${cutPoints.has(u)?" 【关节点】":""}${bridges.some(([a,b])=>a===u||b===u)?" 【含桥端点】":""}`, undefined, u);
  }

  for (let u = 0; u < N; u++) {
    if (disc[u] === -1) {
      snap(`节点 ${u} 未访问，以它为根启动 DFS`);
      dfs(u);
    }
  }

  snap(`✅ 检测完成！桥: ${bridges.map(([a,b])=>`(${a},${b})`).join(", ")} | 关节点: {${Array.from(cutPoints).sort().join(", ")}}`);
  return steps;
}

const STEPS = buildBridgeSteps();

/* ─── Node Display ───────────────────────────────────────────────────────── */
function getNodeStyle(step: BridgeStep, id: number) {
  const isCut = step.cutPoints.has(id);
  const col = step.colors[id];
  if (col === "white") return { fill: "#e2e8f0", stroke: "#94a3b8", text: "#475569", ring: false };
  if (col === "gray")  return { fill: "#6366f1", stroke: "#4f46e5", text: "#fff", ring: isCut };
  return { fill: isCut ? "#ef4444" : "#6b7280", stroke: isCut ? "#dc2626" : "#4b5563", text: "#fff", ring: isCut };
}

function isBridge(step: BridgeStep, u: number, v: number) {
  return step.bridges.some(([a,b]) => (a===u&&b===v)||(a===v&&b===u));
}

export default function BridgeCutPointDemo() {
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed]     = useState(900);
  const timerRef = useRef<ReturnType<typeof setInterval>|null>(null);
  const step = STEPS[stepIdx];

  const advance = useCallback(() => {
    setStepIdx(p => { if (p >= STEPS.length-1) { setPlaying(false); return p; } return p+1; });
  }, []);
  useEffect(() => {
    if (playing) timerRef.current = setInterval(advance, speed);
    else { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current=null; } }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [playing, speed, advance]);

  const reset = () => { setPlaying(false); setStepIdx(0); };
  const activeKey = step.activeEdge ? `${step.activeEdge[0]}-${step.activeEdge[1]}` : "";
  const isDone = stepIdx === STEPS.length - 1;

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-amber-500 via-orange-500 to-red-500 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">桥与关节点——Tarjan 检测算法</h3>
        <p className="text-white/80 text-sm mt-0.5">
          low[v] &gt; disc[u] → 桥 &nbsp;|&nbsp; low[v] ≥ disc[u]（非根）或根有 ≥2 子树 → 关节点
        </p>
      </div>

      <div className="p-5 space-y-4">
        <div className="flex gap-4 items-start">
          {/* SVG */}
          <div className="flex-1 min-w-0 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 overflow-hidden">
            <svg viewBox="0 30 460 200" className="w-full">
              <defs>
                <marker id="bcp-normal" markerWidth="0" markerHeight="0" refX="0" refY="0" orient="auto" />
              </defs>

              {/* Edges */}
              {UNDIRECTED_EDGES.map(([u, v]) => {
                const nu = NODES[u], nv = NODES[v];
                const isAct = activeKey===`${u}-${v}` || activeKey===`${v}-${u}`;
                const isBr  = isBridge(step, u, v);
                const color = isBr ? "#ef4444" : isAct ? "#f59e0b" : "#94a3b8";
                const strokeW = isBr ? 3.5 : isAct ? 2.5 : 1.8;
                const dashed = isBr ? "6 3" : "none";
                return (
                  <line key={`${u}-${v}`}
                    x1={nu.x} y1={nu.y} x2={nv.x} y2={nv.y}
                    stroke={color} strokeWidth={strokeW} strokeDasharray={dashed}
                    className="transition-all duration-300" />
                );
              })}

              {/* Nodes */}
              {NODES.map(node => {
                const ns = getNodeStyle(step, node.id);
                const isActive = step.activeNode === node.id;
                const d = step.disc[node.id];
                const l = step.low[node.id];
                const isCut = step.cutPoints.has(node.id);
                return (
                  <g key={node.id}>
                    {(isCut || isActive) && (
                      <circle cx={node.x} cy={node.y} r={28}
                        fill={isCut ? "#ef444422" : "#f59e0b11"}
                        stroke={isCut ? "#ef4444aa" : "#f59e0b44"} strokeWidth={2} />
                    )}
                    <circle cx={node.x} cy={node.y} r={20}
                      fill={ns.fill} stroke={ns.stroke} strokeWidth={ns.ring ? 3.5 : 2.5}
                      className="transition-all duration-300" />
                    <text x={node.x} y={node.y} textAnchor="middle" dominantBaseline="central"
                      fontSize={12} fontWeight="bold" fill={ns.text}>{node.label}</text>
                    {/* disc/low under node */}
                    {d >= 0 && (
                      <text x={node.x} y={node.y + 30} textAnchor="middle" fontSize={9}
                        fill="#64748b" fontFamily="monospace">
                        d{d} ℓ{l}
                      </text>
                    )}
                  </g>
                );
              })}

              {/* Bridge labels */}
              {step.bridges.map(([u, v]) => {
                const nu = NODES[u], nv = NODES[v];
                return (
                  <text key={`br-${u}-${v}`}
                    x={(nu.x + nv.x) / 2} y={(nu.y + nv.y) / 2 - 10}
                    textAnchor="middle" fontSize={9} fill="#ef4444" fontWeight="bold">
                    BRIDGE
                  </text>
                );
              })}
            </svg>

            {/* Legend */}
            <div className="flex flex-wrap gap-3 px-3 pb-2 text-xs">
              {[
                { color: "#e2e8f0", label: "未访问" },
                { color: "#6366f1", label: "DFS 中（indigo）" },
                { color: "#6b7280", label: "已完成" },
                { color: "#ef4444", label: "关节点（红色）" },
              ].map(({ color, label }) => (
                <span key={label} className="flex items-center gap-1">
                  <span className="inline-block w-3 h-3 rounded-full border border-slate-300" style={{ backgroundColor: color }} />
                  <span className="text-slate-500 dark:text-slate-400">{label}</span>
                </span>
              ))}
              <span className="flex items-center gap-1 ml-auto">
                <span className="inline-block w-8 h-0.5 border-t-2 border-red-500 border-dashed" />
                <span className="text-red-500 font-bold">桥（Bridge）</span>
              </span>
            </div>
          </div>

          {/* Right panel */}
          <div className="w-40 space-y-3 shrink-0">
            {/* Bridges found */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-red-100 dark:bg-red-900/40 px-3 py-1.5 text-xs font-bold text-red-700 dark:text-red-300">
                🌉 发现的桥
              </div>
              <div className="p-2 min-h-[52px] bg-white dark:bg-slate-800/50 space-y-1">
                {step.bridges.length === 0
                  ? <p className="text-xs text-slate-400 text-center">等待...</p>
                  : step.bridges.map(([u,v],i) => (
                    <div key={i} className="rounded px-2 py-0.5 text-xs font-mono font-bold bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400">
                      ({u}, {v})
                    </div>
                  ))
                }
              </div>
            </div>

            {/* Cut points found */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-orange-100 dark:bg-orange-900/40 px-3 py-1.5 text-xs font-bold text-orange-700 dark:text-orange-300">
                🔴 关节点
              </div>
              <div className="p-2 min-h-[52px] bg-white dark:bg-slate-800/50">
                {step.cutPoints.size === 0
                  ? <p className="text-xs text-slate-400 text-center">等待...</p>
                  : <div className="flex flex-wrap gap-1">
                    {Array.from(step.cutPoints).sort().map(id => (
                      <span key={id} className="rounded px-2 py-0.5 text-xs font-bold bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400">
                        节点 {id}
                      </span>
                    ))}
                  </div>
                }
              </div>
            </div>

            {/* disc/low table mini */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-slate-100 dark:bg-slate-800 px-3 py-1.5 text-xs font-bold text-slate-500">
                disc / low
              </div>
              <div className="p-1.5">
                {NODES.map(node => {
                  const d = step.disc[node.id], l = step.low[node.id];
                  if (d < 0) return null;
                  const isCut = step.cutPoints.has(node.id);
                  return (
                    <div key={node.id} className={`flex justify-between px-1 py-0.5 rounded text-[10px] font-mono ${
                      isCut ? "text-red-500 font-bold" : "text-slate-500 dark:text-slate-400"
                    }`}>
                      <span className="font-bold">{node.label}</span>
                      <span>d={d} ℓ={l}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Description */}
        <div className={`rounded-xl px-4 py-2.5 text-sm min-h-[48px] border transition-all duration-300 ${
          isDone
            ? "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700/50 text-emerald-800 dark:text-emerald-300"
            : step.desc.includes("桥") || step.desc.includes("关节点")
            ? "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-700/50 text-red-800 dark:text-red-300"
            : "bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-700/50 text-amber-800 dark:text-amber-300"
        }`}>
          {step.desc}
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={reset}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            ↺ 重置
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.max(0,i-1)); }} disabled={stepIdx===0}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            ← 上一步
          </button>
          <button onClick={() => setPlaying(p=>!p)} disabled={stepIdx===STEPS.length-1}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-colors ${
              playing ? "bg-amber-500 hover:bg-amber-600 text-white" :
              "bg-orange-500 hover:bg-orange-600 text-white"
            }`}>
            {playing ? "⏸ 暂停" : stepIdx===0 ? "▶ 开始" : "▶ 播放"}
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.min(STEPS.length-1,i+1)); }} disabled={stepIdx===STEPS.length-1}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            下一步 →
          </button>
          <span className="text-xs text-slate-400 font-mono">{stepIdx+1}/{STEPS.length}</span>
          <div className="flex items-center gap-1.5 ml-auto">
            <input type="range" min={400} max={1600} step={100} value={speed}
              onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-orange-500" />
            <span className="text-[10px] text-slate-400">{(speed/1000).toFixed(1)}s</span>
          </div>
        </div>
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
          <div className="h-1.5 rounded-full bg-gradient-to-r from-amber-500 to-red-500 transition-all duration-300"
            style={{ width: `${(stepIdx/(STEPS.length-1))*100}%` }} />
        </div>
      </div>
    </div>
  );
}
