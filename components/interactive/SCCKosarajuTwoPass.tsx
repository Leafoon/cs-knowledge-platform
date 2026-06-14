"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── Graph ──────────────────────────────────────────────────────────────────
 *
 *     SCC_A = {0, 1, 2}             SCC_B = {3, 4}
 *
 *   0 ←── 1                         3 ←──→ 4
 *   │     ↑                         ↑
 *   ↓     │                         │
 *   2 ────┘ ──────────────────────→ 3
 *
 *  正向图边: 0→1, 1→2, 2→0, 2→3, 3→4, 4→3
 *  转置图边: 1→0, 2→1, 0→2, 3→2, 4→3, 3→4
 * ─────────────────────────────────────────────────────────────────────────── */
interface GNode { id: number; x: number; y: number; label: string; }
const NODES: GNode[] = [
  { id: 0, x: 85,  y: 95,  label: "0" },
  { id: 1, x: 180, y: 38,  label: "1" },
  { id: 2, x: 180, y: 152, label: "2" },
  { id: 3, x: 315, y: 38,  label: "3" },
  { id: 4, x: 315, y: 152, label: "4" },
];
const FWD_EDGES = [[0,1],[1,2],[2,0],[2,3],[3,4],[4,3]];
const TRANS_EDGES = [[1,0],[2,1],[0,2],[3,2],[4,3],[3,4]];
const N = 5;

const SCC_COLORS = [
  { bg: "#3b82f6", glow: "#93c5fd", label: "SCC A" },
  { bg: "#f97316", glow: "#fdba74", label: "SCC B" },
];

/* ─── Precompute Steps ───────────────────────────────────────────────────── */
type NodeColor = "white" | "gray" | "black";
interface KosarajuStep {
  phase: 1 | 2 | 3;
  nodeColors: NodeColor[];
  callStack: number[];
  finishOrder: number[];
  sccId: number[];          // which SCC index each node belongs to (-1=unknown)
  activeEdges: [number, number][];
  useTranspose: boolean;
  desc: string;
}

function buildKosarajuSteps(): KosarajuStep[] {
  const steps: KosarajuStep[] = [];
  const adj: number[][] = Array.from({ length: N }, () => []);
  const transAdj: number[][] = Array.from({ length: N }, () => []);
  FWD_EDGES.forEach(([u,v]) => adj[u].push(v));
  TRANS_EDGES.forEach(([u,v]) => transAdj[u].push(v));

  const nodeColors: NodeColor[] = Array(N).fill("white") as NodeColor[];
  const callStack: number[] = [];
  const finishOrder: number[] = [];
  const sccId: number[] = Array(N).fill(-1);

  function snap(phase: 1|2|3, desc: string, activeEdges: [number,number][] = [], useTranspose = false) {
    steps.push({
      phase, nodeColors: [...nodeColors] as NodeColor[],
      callStack: [...callStack], finishOrder: [...finishOrder],
      sccId: [...sccId], activeEdges, useTranspose, desc,
    });
  }

  // ── Phase 1: Forward DFS ──────────────────────────────────────────────────
  snap(1, "第一阶段：在正向图上运行 DFS，记录每个节点的完成时间（进入完成栈的顺序）。");

  function dfs1(u: number) {
    nodeColors[u] = "gray";
    callStack.push(u);
    snap(1, `进入节点 ${u}（灰色，加入调用栈）`);
    for (const v of adj[u]) {
      const ae: [number,number] = [u,v];
      snap(1, `检查边 ${u}→${v}：邻居 ${v} ${nodeColors[v] === "white" ? "未访问，递归进入" : nodeColors[v] === "gray" ? "灰色（发现后向边，成环）" : "已完成，跳过"}`, [ae]);
      if (nodeColors[v] === "white") dfs1(v);
    }
    nodeColors[u] = "black";
    callStack.pop();
    finishOrder.push(u);
    snap(1, `完成节点 ${u}（黑色），压入完成栈。当前完成栈: [${finishOrder.join(", ")}]`);
  }

  for (let u = 0; u < N; u++) {
    if (nodeColors[u] === "white") {
      snap(1, `节点 ${u} 未访问，以它为起点启动 DFS`);
      dfs1(u);
    }
  }

  snap(2, `第一阶段完成！完成栈（底→顶）: [${finishOrder.join(" → ")}]。\n第二阶段：按完成时间的逆序（即从栈顶开始），在**转置图** G^T 上运行 DFS，每次能到达的节点集合 = 一个 SCC。`);

  // ── Phase 2: Transpose DFS ────────────────────────────────────────────────
  const visited2 = Array(N).fill(false);
  const processOrder = [...finishOrder].reverse(); // top of stack first
  let sccCounter = 0;

  nodeColors.fill("white");

  function dfs2(u: number, scc: number) {
    visited2[u] = true;
    sccId[u] = scc;
    nodeColors[u] = "gray";
    callStack.push(u);
    snap(2, `[转置图] 进入节点 ${u}，标记为 SCC ${String.fromCharCode(65 + scc)}`, [], true);
    for (const v of transAdj[u]) {
      const ae: [number,number] = [u,v];
      if (!visited2[v]) {
        snap(2, `[转置图] 边 ${u}→${v}（即原图 ${v}→${u}）：${v} 未访问，递归进入，同属 SCC ${String.fromCharCode(65 + scc)}`, [ae], true);
        dfs2(v, scc);
      } else {
        snap(2, `[转置图] 边 ${u}→${v}：${v} 已访问，跳过`, [ae], true);
      }
    }
    nodeColors[u] = "black";
    callStack.pop();
  }

  for (const u of processOrder) {
    if (!visited2[u]) {
      snap(2, `按完成顺序处理节点 ${u}（未访问），以它为起点在转置图上搜索 SCC ${String.fromCharCode(65 + sccCounter)}`);
      dfs2(u, sccCounter);
      snap(2, `SCC ${String.fromCharCode(65 + sccCounter)} 发现完毕：{ ${Array.from({length: N}, (_,i) => i).filter(i => sccId[i] === sccCounter).join(", ")} }`);
      sccCounter++;
    }
  }

  snap(3, `✅ Kosaraju 完成！共发现 ${sccCounter} 个强连通分量：${Array.from({length: sccCounter}, (_,i) =>
    `SCC ${String.fromCharCode(65+i)} = { ${Array.from({length:N},(_,j)=>j).filter(j=>sccId[j]===i).join(",")} }`
  ).join(" | ")}`);

  return steps;
}

const STEPS = buildKosarajuSteps();

/* ─── Helpers ────────────────────────────────────────────────────────────── */
function ArrowSvg({ x1,y1,x2,y2,color,id }: {x1:number;y1:number;x2:number;y2:number;color:string;id:string}) {
  const dx = x2-x1, dy = y2-y1, len = Math.sqrt(dx*dx+dy*dy), R = 18;
  if (len < 1) return null;
  return (
    <>
      <defs>
        <marker id={id} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill={color} />
        </marker>
      </defs>
      <line x1={x1+(dx/len)*R} y1={y1+(dy/len)*R}
        x2={x2-(dx/len)*R} y2={y2-(dy/len)*R}
        stroke={color} strokeWidth={2} markerEnd={`url(#${id})`} />
    </>
  );
}

function nodeStyle(step: typeof STEPS[0], id: number) {
  const sid = step.sccId[id];
  if (step.phase === 3 || sid >= 0) {
    const c = SCC_COLORS[sid] ?? SCC_COLORS[0];
    return { fill: c.bg, stroke: c.bg, text: "white" };
  }
  const col = step.nodeColors[id];
  if (col === "black") return { fill: "#6b7280", stroke: "#4b5563", text: "white" };
  if (col === "gray") return { fill: "#f59e0b", stroke: "#d97706", text: "white" };
  return { fill: "#e2e8f0", stroke: "#94a3b8", text: "#475569" };
}

/* ─── Main ───────────────────────────────────────────────────────────────── */
export default function SCCKosarajuTwoPass() {
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(900);
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
  const reset = () => { setPlaying(false); setStepIdx(0); };

  const edges = step.useTranspose ? TRANS_EDGES : FWD_EDGES;
  const activeSet = new Set(step.activeEdges.map(([u,v]) => `${u}-${v}`));

  const phaseLabel: Record<number,string> = {
    1: "第一阶段：正向图 DFS（记录完成顺序）",
    2: "第二阶段：转置图 DFS（按逆完成序识别 SCC）",
    3: "✅ 算法完成——SCC 已全部识别",
  };
  const phaseGradient: Record<number,string> = {
    1: "from-blue-600 to-indigo-600",
    2: "from-orange-500 to-rose-500",
    3: "from-emerald-500 to-teal-600",
  };

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      <div className={`bg-gradient-to-r ${phaseGradient[step.phase]} px-5 py-4 transition-all duration-500`}>
        <h3 className="text-white font-bold text-lg tracking-tight">Kosaraju 算法——两次 DFS 求 SCC</h3>
        <p className="text-white/80 text-sm mt-0.5">{phaseLabel[step.phase]}</p>
      </div>

      <div className="p-5 space-y-4">
        {/* Phase indicator */}
        <div className="flex gap-2">
          {([1,2,3] as const).map(p => (
            <div key={p} className={`flex-1 h-1.5 rounded-full transition-all duration-500 ${
              step.phase > p ? "bg-emerald-500" : step.phase === p ? (p === 1 ? "bg-blue-500" : p === 2 ? "bg-orange-500" : "bg-emerald-500") : "bg-slate-200 dark:bg-slate-700"
            }`} />
          ))}
        </div>
        <div className="flex gap-3 text-xs text-center">
          {[["第一阶段","正向DFS"],["第二阶段","转置DFS"],["完成","SCC 识别"]].map(([t,s],i) => (
            <div key={i} className={`flex-1 ${step.phase === i+1 ? "text-slate-800 dark:text-slate-100 font-bold" : "text-slate-400"}`}>
              <div>{t}</div><div className="opacity-70 text-[10px]">{s}</div>
            </div>
          ))}
        </div>

        {/* Graph + Info */}
        <div className="flex gap-4 items-start">
          <div className="flex-1 min-w-0 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 overflow-hidden">
            <svg viewBox="0 20 450 180" className="w-full">
              {/* SCC background circles */}
              {step.phase >= 2 && (
                <>
                  <ellipse cx="155" cy="95" rx="100" ry="75" fill={SCC_COLORS[0].bg + "15"}
                    stroke={SCC_COLORS[0].bg + "44"} strokeWidth={1.5} strokeDasharray="4 2" />
                  <ellipse cx="315" cy="95" rx="55" ry="75" fill={SCC_COLORS[1].bg + "15"}
                    stroke={SCC_COLORS[1].bg + "44"} strokeWidth={1.5} strokeDasharray="4 2" />
                </>
              )}

              {/* Recurved edge 3↔4 */}
              {step.phase <= 2 && !step.useTranspose && (
                <>
                  <path d="M 315 58 Q 380 95 315 135" fill="none"
                    stroke={activeSet.has("3-4") ? "#f59e0b" : "#94a3b8"} strokeWidth={activeSet.has("3-4") ? 2.5 : 1.5}
                    markerEnd={activeSet.has("3-4") ? "url(#kos-active)" : "url(#kos-normal)"} />
                  <path d="M 315 135 Q 250 95 315 58" fill="none"
                    stroke={activeSet.has("4-3") ? "#f59e0b" : "#94a3b8"} strokeWidth={activeSet.has("4-3") ? 2.5 : 1.5}
                    markerEnd={activeSet.has("4-3") ? "url(#kos-active)" : "url(#kos-normal)"} />
                </>
              )}
              {step.useTranspose && (
                <>
                  <path d="M 315 58 Q 380 95 315 135" fill="none"
                    stroke={activeSet.has("3-4") ? "#f97316" : "#fdba74"} strokeWidth={activeSet.has("3-4") ? 2.5 : 1.5}
                    markerEnd="url(#kos-trans)" />
                  <path d="M 315 135 Q 250 95 315 58" fill="none"
                    stroke={activeSet.has("4-3") ? "#f97316" : "#fdba74"} strokeWidth={activeSet.has("4-3") ? 2.5 : 1.5}
                    markerEnd="url(#kos-trans)" />
                </>
              )}

              <defs>
                <marker id="kos-normal" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8" />
                </marker>
                <marker id="kos-active" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#f59e0b" />
                </marker>
                <marker id="kos-trans" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#f97316" />
                </marker>
              </defs>

              {/* Draw edges (skip 3-4 and 4-3, handled by curves) */}
              {edges.filter(([u,v]) => !(u===3&&v===4) && !(u===4&&v===3)).map(([u,v]) => {
                const isActive = activeSet.has(`${u}-${v}`);
                const color = step.useTranspose
                  ? (isActive ? "#f97316" : "#fdba74")
                  : (isActive ? "#f59e0b" : "#94a3b8");
                const nu = NODES[u], nv = NODES[v];
                const dx = nv.x-nu.x, dy = nv.y-nu.y, len = Math.sqrt(dx*dx+dy*dy), R=18;
                const mkId = isActive ? (step.useTranspose ? "kos-trans" : "kos-active") : (step.useTranspose ? "kos-trans" : "kos-normal");
                return (
                  <line key={`${u}-${v}`}
                    x1={nu.x+(dx/len)*R} y1={nu.y+(dy/len)*R}
                    x2={nv.x-(dx/len)*R} y2={nv.y-(dy/len)*R}
                    stroke={color} strokeWidth={isActive ? 2.5 : 1.5}
                    markerEnd={`url(#${mkId})`} className="transition-all duration-300" />
                );
              })}

              {NODES.map(node => {
                const { fill, stroke, text } = nodeStyle(step, node.id);
                const isTop = step.callStack[step.callStack.length-1] === node.id;
                const sid = step.sccId[node.id];
                return (
                  <g key={node.id}>
                    {isTop && <circle cx={node.x} cy={node.y} r={27} fill="#f59e0b22" stroke="#f59e0b55" strokeWidth={2} />}
                    <circle cx={node.x} cy={node.y} r={20} fill={fill} stroke={stroke} strokeWidth={2.5}
                      className="transition-all duration-300" />
                    <text x={node.x} y={node.y} textAnchor="middle" dominantBaseline="central"
                      fontSize={12} fontWeight="bold" fill={text}>{node.label}</text>
                    {sid >= 0 && (
                      <text x={node.x} y={node.y + 32} textAnchor="middle" fontSize={9} fill={SCC_COLORS[sid].bg} fontWeight="bold">
                        SCC {String.fromCharCode(65 + sid)}
                      </text>
                    )}
                  </g>
                );
              })}

              {/* SCC labels background */}
              {step.phase >= 2 && (
                <>
                  <text x="155" y="25" textAnchor="middle" fontSize={10} fill={SCC_COLORS[0].bg} fontWeight="bold" opacity={0.7}>SCC A</text>
                  <text x="315" y="25" textAnchor="middle" fontSize={10} fill={SCC_COLORS[1].bg} fontWeight="bold" opacity={0.7}>SCC B</text>
                </>
              )}

              {/* Transpose label */}
              {step.useTranspose && (
                <text x="225" y="30" textAnchor="middle" fontSize={10} fill="#f97316" fontWeight="bold">
                  [转置图 G^T — 橙色边]
                </text>
              )}
            </svg>
          </div>

          {/* Right info */}
          <div className="w-44 space-y-3 shrink-0">
            {/* Finish Stack (phase 1) */}
            {step.phase <= 2 && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
                <div className="bg-blue-100 dark:bg-blue-900/40 px-3 py-1.5 text-xs font-bold text-blue-700 dark:text-blue-300">
                  完成栈（第一阶段）
                </div>
                <div className="p-2 min-h-[52px] bg-white dark:bg-slate-800/50 space-y-1">
                  {step.finishOrder.length === 0
                    ? <p className="text-xs text-slate-400 text-center">等待...</p>
                    : [...step.finishOrder].reverse().map((id, i) => (
                        <div key={i} className={`rounded px-2 py-0.5 text-xs font-bold text-center ${
                          i === 0 ? "bg-blue-500 text-white" : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300"
                        }`}>
                          {i === 0 ? "↑ 栈顶 " : ""}节点 {id}
                        </div>
                      ))
                  }
                </div>
              </div>
            )}

            {/* SCC results */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-slate-100 dark:bg-slate-800 px-3 py-1.5 text-xs font-bold text-slate-600 dark:text-slate-300">
                识别到的 SCC
              </div>
              <div className="p-2 min-h-[52px] bg-white dark:bg-slate-800/50 space-y-1.5">
                {Array.from({ length: 2 }, (_, si) => {
                  const members = Array.from({ length: N }, (_,i) => i).filter(i => step.sccId[i] === si);
                  if (members.length === 0) return null;
                  return (
                    <div key={si} className="rounded px-2 py-1" style={{ backgroundColor: SCC_COLORS[si].bg + "20", border: `1px solid ${SCC_COLORS[si].bg}44` }}>
                      <div className="text-[10px] font-bold" style={{ color: SCC_COLORS[si].bg }}>SCC {String.fromCharCode(65+si)}</div>
                      <div className="text-xs text-slate-700 dark:text-slate-300">{"{ "}{members.join(", ")}{" }"}</div>
                    </div>
                  );
                })}
                {step.sccId.every(s => s < 0) && <p className="text-xs text-slate-400 text-center">等待第二阶段...</p>}
              </div>
            </div>
          </div>
        </div>

        {/* Description */}
        <div className={`rounded-xl px-4 py-2.5 text-sm min-h-[48px] border transition-colors ${
          step.phase === 1 ? "bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-700/50 text-blue-800 dark:text-blue-300" :
          step.phase === 2 ? "bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-700/50 text-orange-800 dark:text-orange-300" :
          "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700/50 text-emerald-800 dark:text-emerald-300"
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
              step.phase === 1 ? "bg-blue-600 hover:bg-blue-700 text-white" :
              step.phase === 2 ? "bg-orange-500 hover:bg-orange-600 text-white" :
              "bg-emerald-600 hover:bg-emerald-700 text-white"
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
              onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-blue-500" />
            <span className="text-[10px] text-slate-400">{(speed/1000).toFixed(1)}s</span>
          </div>
        </div>
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
          <div className={`h-1.5 rounded-full transition-all duration-300 ${
            step.phase===1?"bg-blue-500":step.phase===2?"bg-orange-500":"bg-emerald-500"}`}
            style={{ width: `${(stepIdx/(STEPS.length-1))*100}%` }} />
        </div>
      </div>
    </div>
  );
}
