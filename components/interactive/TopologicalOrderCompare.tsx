"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── Graph (Shared) ────────────────────────────────────────────────────────
 *  Nodes: 高数(0) 线代(1) 概率(2) 统计(3) ML(4) DL(5)
 *  Edges: 0→2, 0→3, 1→2, 2→4, 3→4, 4→5
 * ─────────────────────────────────────────────────────────────────────────── */
const LABELS = ["高数", "线代", "概率", "统计", "ML", "DL"];
const N = 6;
const ADJ: number[][] = [[2,3],[2],[4],[4],[5],[]];
const EDGES: [number,number][] = [[0,2],[0,3],[1,2],[2,4],[3,4],[4,5]];
const SMALL_POS = [
  { x: 55,  y: 80  },
  { x: 55,  y: 155 },
  { x: 155, y: 80  },
  { x: 55,  y: 230 },
  { x: 255, y: 80  },
  { x: 355, y: 80  },
];

/* ─── Kahn Steps ─────────────────────────────────────────────────────────── */
interface KS { indegree: number[]; queue: number[]; output: number[]; processing: number; done: boolean[]; desc: string; }

function computeKahn(): KS[] {
  const steps: KS[] = [];
  const indeg = Array(N).fill(0);
  EDGES.forEach(([,v]) => indeg[v]++);
  const queue: number[] = [];
  const output: number[] = [];
  const done = Array(N).fill(false);

  function snap(proc: number, desc: string) {
    steps.push({ indegree: [...indeg], queue: [...queue], output: [...output], processing: proc, done: [...done], desc });
  }
  snap(-1, "初始化：计算各节点入度");
  for (let i = 0; i < N; i++) if (indeg[i] === 0) { queue.push(i); }
  snap(-1, `入度为 0 的节点入队：${queue.map(i=>LABELS[i]).join(", ")}`);

  while (queue.length > 0) {
    const u = queue.shift()!;
    snap(u, `出队节点 ${LABELS[u]}（u=${u}）`);
    output.push(u);
    done[u] = true;
    for (const v of ADJ[u]) {
      indeg[v]--;
      snap(u, `${LABELS[u]}→${LABELS[v]}：indegree[${v}]-- = ${indeg[v]}`);
      if (indeg[v] === 0) { queue.push(v); snap(u, `${LABELS[v]} 入度降为 0，入队`); }
    }
    snap(-1, `拓扑序目前: ${output.map(i=>LABELS[i]).join(" → ")}`);
  }
  snap(-1, `✅ Kahn 完成：${output.map(i=>LABELS[i]).join(" → ")}`);
  return steps;
}

/* ─── DFS Steps ──────────────────────────────────────────────────────────── */
type Color = "white"|"gray"|"black";
interface DS { colors: Color[]; callStack: number[]; finishStack: number[]; topoOrder: number[]; desc: string; }

function computeDFS(): DS[] {
  const steps: DS[] = [];
  const colors: Color[] = Array(N).fill("white") as Color[];
  const callStack: number[] = [];
  const finishStack: number[] = [];

  function snap(desc: string) {
    steps.push({ colors: [...colors] as Color[], callStack: [...callStack], finishStack: [...finishStack], topoOrder: [...finishStack].reverse(), desc });
  }
  snap("初始化：所有节点白色（未访问）");

  function dfs(u: number) {
    colors[u] = "gray"; callStack.push(u);
    snap(`进入节点 ${LABELS[u]}`);
    for (const v of ADJ[u]) {
      if (colors[v] === "white") { snap(`边 ${LABELS[u]}→${LABELS[v]}：递归`); dfs(v); }
    }
    colors[u] = "black"; callStack.pop(); finishStack.push(u);
    snap(`完成节点 ${LABELS[u]}，压入完成栈`);
  }

  for (let u = 0; u < N; u++) {
    if (colors[u] === "white") { snap(`以节点 ${LABELS[u]} 为起点 DFS`); dfs(u); }
  }
  snap(`✅ DFS 完成，逆后序 = ${[...finishStack].reverse().map(i=>LABELS[i]).join(" → ")}`);
  return steps;
}

const KAHN_STEPS = computeKahn();
const DFS_STEPS  = computeDFS();
const TOTAL = Math.max(KAHN_STEPS.length, DFS_STEPS.length);

/* ─── Node color helpers ─────────────────────────────────────────────────── */
function kahnNodeStyle(ks: KS, id: number) {
  if (ks.done[id]) return { fill: "#10b981", stroke: "#059669" };
  if (ks.processing === id) return { fill: "#f59e0b", stroke: "#d97706" };
  if (ks.queue.includes(id)) return { fill: "#6366f1", stroke: "#4f46e5" };
  return { fill: "#e2e8f0", stroke: "#94a3b8" };
}
function dfsNodeStyle(ds: DS, id: number) {
  const c = ds.colors[id];
  if (c === "black") return { fill: "#10b981", stroke: "#059669" };
  if (c === "gray")  return { fill: ds.callStack[ds.callStack.length-1]===id ? "#f59e0b" : "#6366f1",
                              stroke: ds.callStack[ds.callStack.length-1]===id ? "#d97706" : "#4f46e5" };
  return { fill: "#e2e8f0", stroke: "#94a3b8" };
}

/* ─── Mini SVG graph ────────────────────────────────────────────────────── */
function MiniGraph({ styleOf }: { styleOf: (id: number) => { fill: string; stroke: string } }) {
  return (
    <svg viewBox="0 50 410 215" className="w-full">
      <defs>
        <marker id="cmp-arr" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
          <path d="M0,0 L5,2.5 L0,5 Z" fill="#94a3b8" />
        </marker>
      </defs>
      {EDGES.map(([u,v]) => {
        const nu = SMALL_POS[u], nv = SMALL_POS[v];
        const dx = nv.x-nu.x, dy = nv.y-nu.y, len = Math.sqrt(dx*dx+dy*dy), R=16;
        return (
          <line key={`${u}-${v}`}
            x1={nu.x+(dx/len)*R} y1={nu.y+(dy/len)*R}
            x2={nv.x-(dx/len)*R} y2={nv.y-(dy/len)*R}
            stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#cmp-arr)" />
        );
      })}
      {SMALL_POS.map((pos, id) => {
        const { fill, stroke } = styleOf(id);
        return (
          <g key={id}>
            <circle cx={pos.x} cy={pos.y} r={16} fill={fill} stroke={stroke} strokeWidth={2}
              style={{ transition: "fill .3s, stroke .3s" }} />
            <text x={pos.x} y={pos.y-1} textAnchor="middle" dominantBaseline="central" fontSize={9} fontWeight="bold"
              fill={fill === "#e2e8f0" ? "#475569" : "#fff"}>{LABELS[id]}</text>
          </g>
        );
      })}
    </svg>
  );
}

/* ─── Main ───────────────────────────────────────────────────────────────── */
export default function TopologicalOrderCompare() {
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(900);
  const timerRef = useRef<ReturnType<typeof setInterval>|null>(null);

  const advance = useCallback(() => {
    setStepIdx(p => { if (p >= TOTAL-1) { setPlaying(false); return p; } return p+1; });
  }, []);
  useEffect(() => {
    if (playing) timerRef.current = setInterval(advance, speed);
    else { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current=null; } }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [playing, speed, advance]);

  const reset = () => { setPlaying(false); setStepIdx(0); };
  const ks = KAHN_STEPS[Math.min(stepIdx, KAHN_STEPS.length-1)];
  const ds = DFS_STEPS[Math.min(stepIdx, DFS_STEPS.length-1)];
  const kDone = stepIdx >= KAHN_STEPS.length - 1;
  const dDone = stepIdx >= DFS_STEPS.length - 1;
  const bothDone = kDone && dDone;

  const kahnOrder = KAHN_STEPS[KAHN_STEPS.length-1].output;
  const dfsOrder  = DFS_STEPS[DFS_STEPS.length-1].finishStack.slice().reverse();
  const sameOrder = bothDone && kahnOrder.join() === dfsOrder.join();

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-600 via-indigo-600 to-teal-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Kahn 算法 vs DFS 逆后序——拓扑排序对比</h3>
        <p className="text-white/80 text-sm mt-0.5">同一图，两种算法并行演示；输出的拓扑序是否相同？</p>
      </div>

      <div className="p-5 space-y-4">
        {/* Two panels */}
        <div className="grid grid-cols-2 gap-4">
          {/* Kahn */}
          <div className={`rounded-xl border overflow-hidden transition-all duration-300 ${kDone ? "border-violet-300 dark:border-violet-700" : "border-slate-200 dark:border-slate-700"}`}>
            <div className={`px-3 py-2 flex items-center gap-2 ${kDone ? "bg-violet-100 dark:bg-violet-900/40" : "bg-slate-100 dark:bg-slate-800"}`}>
              <span className="text-xs font-bold text-violet-700 dark:text-violet-300">Kahn（BFS 入度）</span>
              {kDone && <span className="ml-auto text-[10px] bg-violet-200 dark:bg-violet-800 text-violet-700 dark:text-violet-200 rounded px-1.5 py-0.5 font-bold">完成 ✓</span>}
            </div>
            <div className="bg-slate-50 dark:bg-slate-800/40">
              <MiniGraph styleOf={id => kahnNodeStyle(ks, id)} />
            </div>
            {/* Kahn queue */}
            <div className="px-3 py-2 border-t border-slate-200 dark:border-slate-700">
              <div className="text-[10px] text-slate-400 mb-1">队列</div>
              <div className="flex gap-1 flex-wrap min-h-[20px]">
                {ks.queue.map((id,i) => (
                  <span key={i} className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300">
                    {LABELS[id]}
                  </span>
                ))}
              </div>
              <div className="text-[10px] text-slate-400 mt-1.5 mb-1">拓扑序输出</div>
              <div className="flex gap-0.5 flex-wrap min-h-[20px]">
                {ks.output.map((id,i) => (
                  <React.Fragment key={i}>
                    {i > 0 && <span className="text-[10px] text-slate-400">→</span>}
                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300">
                      {LABELS[id]}
                    </span>
                  </React.Fragment>
                ))}
              </div>
            </div>
            <div className="px-3 py-2 bg-violet-50/50 dark:bg-violet-900/10 border-t border-slate-100 dark:border-slate-800 min-h-[44px]">
              <p className="text-[10px] text-violet-700 dark:text-violet-300 leading-relaxed">{ks.desc}</p>
            </div>
          </div>

          {/* DFS */}
          <div className={`rounded-xl border overflow-hidden transition-all duration-300 ${dDone ? "border-teal-300 dark:border-teal-700" : "border-slate-200 dark:border-slate-700"}`}>
            <div className={`px-3 py-2 flex items-center gap-2 ${dDone ? "bg-teal-100 dark:bg-teal-900/40" : "bg-slate-100 dark:bg-slate-800"}`}>
              <span className="text-xs font-bold text-teal-700 dark:text-teal-300">DFS 逆后序</span>
              {dDone && <span className="ml-auto text-[10px] bg-teal-200 dark:bg-teal-800 text-teal-700 dark:text-teal-200 rounded px-1.5 py-0.5 font-bold">完成 ✓</span>}
            </div>
            <div className="bg-slate-50 dark:bg-slate-800/40">
              <MiniGraph styleOf={id => dfsNodeStyle(ds, id)} />
            </div>
            {/* DFS stacks */}
            <div className="px-3 py-2 border-t border-slate-200 dark:border-slate-700">
              <div className="text-[10px] text-slate-400 mb-1">调用栈</div>
              <div className="flex gap-1 flex-wrap min-h-[20px]">
                {ds.callStack.map((id,i) => (
                  <span key={i} className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                    i === ds.callStack.length-1 ? "bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300"
                    : "bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300"
                  }`}>{LABELS[id]}</span>
                ))}
              </div>
              <div className="text-[10px] text-slate-400 mt-1.5 mb-1">拓扑序输出（逆后序）</div>
              <div className="flex gap-0.5 flex-wrap min-h-[20px]">
                {ds.topoOrder.map((id,i) => (
                  <React.Fragment key={i}>
                    {i > 0 && <span className="text-[10px] text-slate-400">→</span>}
                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300">
                      {LABELS[id]}
                    </span>
                  </React.Fragment>
                ))}
              </div>
            </div>
            <div className="px-3 py-2 bg-teal-50/50 dark:bg-teal-900/10 border-t border-slate-100 dark:border-slate-800 min-h-[44px]">
              <p className="text-[10px] text-teal-700 dark:text-teal-300 leading-relaxed">{ds.desc}</p>
            </div>
          </div>
        </div>

        {/* Final comparison */}
        {bothDone && (
          <div className={`rounded-xl border px-4 py-3 text-sm transition-all duration-500 ${
            sameOrder
              ? "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-300 dark:border-emerald-700 text-emerald-800 dark:text-emerald-200"
              : "bg-amber-50 dark:bg-amber-900/20 border-amber-300 dark:border-amber-700 text-amber-800 dark:text-amber-200"
          }`}>
            <div className="font-bold mb-2">{sameOrder ? "✅ 两种算法输出相同的拓扑序！" : "⚠️ 两种算法的拓扑序不同（均合法）"}</div>
            <div className="flex flex-col gap-1 text-xs">
              <div className="flex items-center gap-2">
                <span className="font-bold text-violet-600 dark:text-violet-400 w-20 shrink-0">Kahn 输出：</span>
                <span className="font-mono">{kahnOrder.map(i => LABELS[i]).join(" → ")}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-teal-600 dark:text-teal-400 w-20 shrink-0">DFS 输出：</span>
                <span className="font-mono">{dfsOrder.map(i => LABELS[i]).join(" → ")}</span>
              </div>
              <p className="mt-1 opacity-80">
                注：拓扑序不唯一。只要所有依赖（前驱）出现在被依赖节点之前即合法。两种算法在此图上的差异源于访问顺序不同。
              </p>
            </div>
          </div>
        )}

        {/* Comparison table */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
          <div className="bg-slate-100 dark:bg-slate-800 px-4 py-2 text-xs font-bold text-slate-600 dark:text-slate-300">算法对比</div>
          <table className="w-full text-xs">
            <thead className="bg-slate-50 dark:bg-slate-800/50">
              <tr>
                <th className="py-2 px-3 text-left text-slate-500 font-semibold">特性</th>
                <th className="py-2 px-3 text-center font-bold text-violet-600 dark:text-violet-400">Kahn（入度 BFS）</th>
                <th className="py-2 px-3 text-center font-bold text-teal-600 dark:text-teal-400">DFS 逆后序</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
              {[
                ["核心数据结构", "队列（入度为 0 的节点）", "递归调用栈 + 完成栈"],
                ["空间复杂度", "O(V + E)", "O(V + E)（递归栈 O(V)）"],
                ["时间复杂度", "O(V + E)", "O(V + E)"],
                ["环路检测", "output.length < V ⟹ 有环", "灰色节点遇到灰色边 ⟹ 有环"],
                ["输出时机", "节点出队时立即输出", "DFS 完成（黑色）后逆序"],
                ["拓扑序唯一性", "不唯一（依赖队列顺序）", "不唯一（依赖 DFS 起点顺序）"],
              ].map(([attr, kahn, dfs], i) => (
                <tr key={i} className={i%2===0?"bg-white dark:bg-slate-900":"bg-slate-50/50 dark:bg-slate-800/30"}>
                  <td className="py-2 px-3 text-slate-600 dark:text-slate-400 font-medium">{attr}</td>
                  <td className="py-2 px-3 text-center text-slate-700 dark:text-slate-300">{kahn}</td>
                  <td className="py-2 px-3 text-center text-slate-700 dark:text-slate-300">{dfs}</td>
                </tr>
              ))}
            </tbody>
          </table>
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
          <button onClick={() => setPlaying(p=>!p)} disabled={stepIdx===TOTAL-1}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-colors ${
              playing ? "bg-amber-500 hover:bg-amber-600 text-white" :
              "bg-gradient-to-r from-violet-600 to-teal-600 hover:from-violet-700 hover:to-teal-700 text-white"
            }`}>
            {playing ? "⏸ 暂停" : stepIdx===0 ? "▶ 同步播放" : "▶ 继续"}
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.min(TOTAL-1,i+1)); }} disabled={stepIdx===TOTAL-1}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            下一步 →
          </button>
          <span className="text-xs text-slate-400 font-mono">{stepIdx+1}/{TOTAL}</span>
          <div className="flex items-center gap-1.5 ml-auto">
            <input type="range" min={400} max={1600} step={100} value={speed}
              onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-indigo-500" />
            <span className="text-[10px] text-slate-400">{(speed/1000).toFixed(1)}s</span>
          </div>
        </div>
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
          <div className="h-1.5 rounded-full bg-gradient-to-r from-violet-500 to-teal-500 transition-all duration-300"
            style={{ width: `${(stepIdx/(TOTAL-1))*100}%` }} />
        </div>
      </div>
    </div>
  );
}
