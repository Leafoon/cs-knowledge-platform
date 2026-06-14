"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── Graph ──────────────────────────────────────────────────────────────────
 *  同 Kosaraju 示例：5 个节点，2 个 SCC
 *  SCC_A = {0,1,2} (0→1→2→0 环)
 *  SCC_B = {3,4}   (3→4→3 环)
 *  桥接边: 2→3
 * ─────────────────────────────────────────────────────────────────────────── */
const NODES = [
  { id: 0, x: 85,  y: 95,  label: "0" },
  { id: 1, x: 185, y: 38,  label: "1" },
  { id: 2, x: 185, y: 152, label: "2" },
  { id: 3, x: 315, y: 38,  label: "3" },
  { id: 4, x: 315, y: 152, label: "4" },
];
const ADJ: number[][] = [[1],[2],[0,3],[4],[3]]; // 0→1, 1→2, 2→0, 2→3, 3→4, 4→3
const EDGES = [[0,1],[1,2],[2,0],[2,3],[3,4],[4,3]];
const N = 5;

const SCC_PALETTE = [
  { bg: "#f97316", border: "#ea580c", text: "#fff", label: "SCC B" }, // SCC 0 = {3,4} found first
  { bg: "#3b82f6", border: "#2563eb", text: "#fff", label: "SCC A" }, // SCC 1 = {0,1,2} found second
];

/* ─── Precompute Steps ───────────────────────────────────────────────────── */
interface TarjanStep {
  disc: number[];          // -1 = unvisited
  low: number[];           // -1 = unvisited
  inStack: boolean[];
  tarStack: number[];      // Tarjan auxiliary stack
  sccId: number[];         // -1 = not in any SCC yet
  colors: string[];        // "white"|"gray"|"black"
  activeEdge?: [number, number];
  highlightPop: number[];  // nodes being popped to form SCC this step
  timer: number;           // global DFS timer
  desc: string;
}

function buildTarjanSteps(): TarjanStep[] {
  const steps: TarjanStep[] = [];
  const disc = Array(N).fill(-1);
  const low  = Array(N).fill(-1);
  const inStk = Array(N).fill(false);
  const tarStack: number[] = [];
  const sccId = Array(N).fill(-1);
  const colors: string[] = Array(N).fill("white");
  let timer = 0;
  let sccCounter = 0;

  function snap(desc: string, ae?: [number,number], pop: number[] = []) {
    steps.push({
      disc: [...disc], low: [...low], inStack: [...inStk],
      tarStack: [...tarStack], sccId: [...sccId],
      colors: [...colors], activeEdge: ae, highlightPop: pop,
      timer, desc,
    });
  }

  snap("Tarjan 算法开始。初始化 disc[]、low[] 全为 -1，辅助栈为空。");

  function dfs(u: number) {
    disc[u] = low[u] = timer++;
    inStk[u] = true;
    tarStack.push(u);
    colors[u] = "gray";
    snap(`进入节点 ${u}：disc[${u}] = low[${u}] = ${disc[u]}，压入辅助栈`);

    for (const v of ADJ[u]) {
      const ae: [number,number] = [u, v];
      if (disc[v] === -1) {
        snap(`检查边 ${u}→${v}：节点 ${v} 尚未访问，递归进入`, ae);
        dfs(v);
        const oldLow = low[u];
        low[u] = Math.min(low[u], low[v]);
        snap(
          `从 ${v} 回到 ${u}：low[${u}] = min(${oldLow}, low[${v}]=${low[v]}) = ${low[u]}`,
          ae
        );
      } else if (inStk[v]) {
        const oldLow = low[u];
        low[u] = Math.min(low[u], disc[v]);
        snap(
          `检查边 ${u}→${v}：${v} 在辅助栈中（发现返祖/横向边）\n→ low[${u}] = min(${oldLow}, disc[${v}]=${disc[v]}) = ${low[u]}`,
          ae
        );
      } else {
        snap(`检查边 ${u}→${v}：${v} 已访问且不在辅助栈，跳过（已属其他 SCC）`, ae);
      }
    }

    if (low[u] === disc[u]) {
      const scc: number[] = [];
      while (true) {
        const w = tarStack.pop()!;
        inStk[w] = false;
        sccId[w] = sccCounter;
        colors[w] = "black";
        scc.push(w);
        if (w === u) break;
      }
      snap(
        `low[${u}] === disc[${u}] = ${disc[u]}！触发出栈 → SCC ${String.fromCharCode(65 + sccCounter)} = { ${scc.join(", ")} }`,
        undefined,
        [...scc]
      );
      sccCounter++;
    } else {
      colors[u] = "black";
      snap(`节点 ${u} 完成，low[${u}]=${low[u]} ≠ disc[${u}]=${disc[u]}，不是根节点，保留在辅助栈`);
    }
  }

  for (let u = 0; u < N; u++) {
    if (disc[u] === -1) {
      snap(`节点 ${u} 未访问，启动 DFS`);
      dfs(u);
    }
  }

  snap(`✅ Tarjan 完成！共发现 ${sccCounter} 个 SCC：SCC A={0,1,2}，SCC B={3,4}`);
  return steps;
}

const STEPS = buildTarjanSteps();

/* ─── Rendering ──────────────────────────────────────────────────────────── */
function nodeColor(step: TarjanStep, id: number) {
  const sid = step.sccId[id];
  if (sid >= 0) return SCC_PALETTE[sid];
  const inStack = step.inStack[id];
  const disc = step.disc[id];
  if (disc === -1) return { bg: "#e2e8f0", border: "#94a3b8", text: "#475569", label: "" };
  if (inStack) return { bg: "#6366f1", border: "#4f46e5", text: "#fff", label: "" };
  return { bg: "#6b7280", border: "#4b5563", text: "#fff", label: "" };
}

export default function SCCTarjanStack() {
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(900);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const step = STEPS[stepIdx];

  const advance = useCallback(() => {
    setStepIdx(p => { if (p >= STEPS.length-1) { setPlaying(false); return p; } return p+1; });
  }, []);
  useEffect(() => {
    if (playing) timerRef.current = setInterval(advance, speed);
    else { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; } }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [playing, speed, advance]);

  const reset = () => { setPlaying(false); setStepIdx(0); };
  const activeKey = step.activeEdge ? `${step.activeEdge[0]}-${step.activeEdge[1]}` : "";
  const popSet = new Set(step.highlightPop);
  const isDone = step.sccId.every(s => s >= 0) && stepIdx === STEPS.length - 1;

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-rose-600 via-red-500 to-orange-500 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Tarjan 算法——单次 DFS 求 SCC</h3>
        <p className="text-white/80 text-sm mt-0.5">维护 disc[u]、low[u] 与辅助栈，当 low[u] = disc[u] 时弹出一个 SCC</p>
      </div>

      <div className="p-5 space-y-4">
        {/* Main content */}
        <div className="flex gap-4 items-start">
          {/* SVG graph */}
          <div className="flex-1 min-w-0 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 overflow-hidden">
            <svg viewBox="0 10 450 195" className="w-full">
              <defs>
                <marker id="tj-normal" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8" />
                </marker>
                <marker id="tj-active" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#f59e0b" />
                </marker>
              </defs>

              {/* SCC halos */}
              {[0,1].map(si => {
                const members = NODES.filter(n => step.sccId[n.id] === si);
                if (members.length === 0) return null;
                const xs = members.map(n=>n.x), ys = members.map(n=>n.y);
                const cx = xs.reduce((a,b)=>a+b)/xs.length, cy = ys.reduce((a,b)=>a+b)/ys.length;
                const rx = Math.max(...xs)-Math.min(...xs)+48, ry = Math.max(...ys)-Math.min(...ys)+48;
                return (
                  <ellipse key={si} cx={cx} cy={cy}
                    rx={Math.max(rx/2, 35)} ry={Math.max(ry/2, 35)}
                    fill={SCC_PALETTE[si].bg + "22"} stroke={SCC_PALETTE[si].bg + "66"} strokeWidth={1.5} strokeDasharray="4 2" />
                );
              })}

              {/* Curved edges for 3↔4 */}
              <path d="M 314 52 Q 368 95 314 138" fill="none"
                stroke={activeKey==="3-4" ? "#f59e0b" : "#94a3b8"} strokeWidth={activeKey==="3-4"?2.5:1.5}
                markerEnd={activeKey==="3-4" ? "url(#tj-active)" : "url(#tj-normal)"} />
              <path d="M 316 138 Q 262 95 316 52" fill="none"
                stroke={activeKey==="4-3" ? "#f59e0b" : "#94a3b8"} strokeWidth={activeKey==="4-3"?2.5:1.5}
                markerEnd={activeKey==="4-3" ? "url(#tj-active)" : "url(#tj-normal)"} />

              {/* Other edges */}
              {EDGES.filter(([u,v]) => !(u===3&&v===4) && !(u===4&&v===3)).map(([u,v]) => {
                const isActive = activeKey === `${u}-${v}`;
                const nu = NODES[u], nv = NODES[v];
                const dx = nv.x-nu.x, dy = nv.y-nu.y, len = Math.sqrt(dx*dx+dy*dy), R = 18;
                return (
                  <line key={`${u}-${v}`}
                    x1={nu.x+(dx/len)*R} y1={nu.y+(dy/len)*R}
                    x2={nv.x-(dx/len)*R} y2={nv.y-(dy/len)*R}
                    stroke={isActive ? "#f59e0b" : "#94a3b8"} strokeWidth={isActive?2.5:1.5}
                    markerEnd={isActive ? "url(#tj-active)" : "url(#tj-normal)"}
                    className="transition-all duration-300" />
                );
              })}

              {/* Nodes */}
              {NODES.map(node => {
                const nc = nodeColor(step, node.id);
                const isPopping = popSet.has(node.id);
                const d = step.disc[node.id];
                const l = step.low[node.id];
                return (
                  <g key={node.id}>
                    {isPopping && <circle cx={node.x} cy={node.y} r={32} fill="#f59e0b22" stroke="#f59e0b88" strokeWidth={2.5} />}
                    <circle cx={node.x} cy={node.y} r={20} fill={nc.bg} stroke={nc.border} strokeWidth={2.5}
                      className="transition-all duration-300" />
                    <text x={node.x} y={node.y} textAnchor="middle" dominantBaseline="central"
                      fontSize={12} fontWeight="bold" fill={nc.text}>{node.label}</text>
                    {/* disc/low labels */}
                    {d >= 0 && (
                      <text x={node.x-26} y={node.y-24} textAnchor="middle" fontSize={9} fill="#64748b">
                        d={d}
                      </text>
                    )}
                    {l >= 0 && (
                      <text x={node.x+26} y={node.y-24} textAnchor="middle" fontSize={9}
                        fill={d >= 0 && l === d ? "#ef4444" : "#64748b"} fontWeight={d >= 0 && l === d ? "bold" : "normal"}>
                        ℓ={l}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>

            {/* Legend */}
            <div className="flex gap-3 px-3 pb-2 flex-wrap text-xs">
              {[
                { bg: "#e2e8f0", label: "未访问" },
                { bg: "#6366f1", label: "在辅助栈中" },
                { bg: "#6b7280", label: "已完成" },
                { bg: SCC_PALETTE[0].bg, label: "SCC B" },
                { bg: SCC_PALETTE[1].bg, label: "SCC A" },
              ].map(({ bg, label }) => (
                <span key={label} className="flex items-center gap-1">
                  <span className="inline-block w-3 h-3 rounded-full border border-slate-300" style={{ backgroundColor: bg }} />
                  <span className="text-slate-500 dark:text-slate-400">{label}</span>
                </span>
              ))}
              <span className="text-slate-400 ml-auto">d=disc  ℓ=low  <span className="text-red-500 font-bold">ℓ=d 时触发 SCC</span></span>
            </div>
          </div>

          {/* Right panel */}
          <div className="w-40 space-y-3 shrink-0">
            {/* Tarjan auxiliary stack */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-indigo-100 dark:bg-indigo-900/40 px-3 py-1.5 text-xs font-bold text-indigo-700 dark:text-indigo-300">
                Tarjan 辅助栈
              </div>
              <div className="p-2 min-h-[100px] bg-white dark:bg-slate-800/50 space-y-1">
                {step.tarStack.length === 0
                  ? <p className="text-xs text-slate-400 text-center mt-3">（空）</p>
                  : [...step.tarStack].reverse().map((id, i) => {
                    const isTop = i === 0;
                    const nc = nodeColor(step, id);
                    return (
                      <div key={i} className="rounded px-2 py-0.5 text-xs font-bold flex items-center justify-between"
                        style={{ backgroundColor: nc.bg + "22", borderLeft: `3px solid ${nc.bg}`, color: nc.bg }}>
                        <span>{isTop ? "↑ " : ""}{id}</span>
                        <span className="text-[10px] opacity-70">d={step.disc[id]}, ℓ={step.low[id]}</span>
                      </div>
                    );
                  })
                }
              </div>
            </div>

            {/* SCC results */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-slate-100 dark:bg-slate-800 px-3 py-1.5 text-xs font-bold text-slate-600 dark:text-slate-300">
                已发现 SCC
              </div>
              <div className="p-2 min-h-[52px] bg-white dark:bg-slate-800/50 space-y-1.5">
                {SCC_PALETTE.map((pal, si) => {
                  const members = Array.from({ length: N }, (_,i) => i).filter(i => step.sccId[i] === si);
                  if (members.length === 0) return null;
                  return (
                    <div key={si} className="rounded px-2 py-1" style={{ backgroundColor: pal.bg + "22", borderLeft: `3px solid ${pal.bg}` }}>
                      <div className="text-[10px] font-bold" style={{ color: pal.bg }}>{pal.label}</div>
                      <div className="text-xs text-slate-700 dark:text-slate-300">{"{ "}{members.join(", ")}{" }"}</div>
                    </div>
                  );
                })}
                {step.sccId.every(s => s < 0) && <p className="text-xs text-slate-400 text-center">等待...</p>}
              </div>
            </div>
          </div>
        </div>

        {/* disc/low table */}
        <div className="overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-700">
          <table className="w-full text-xs text-center">
            <thead className="bg-slate-100 dark:bg-slate-800">
              <tr>
                <th className="py-1.5 px-2 text-slate-500 font-semibold">节点</th>
                {NODES.map(n => <th key={n.id} className="py-1.5 px-2 font-semibold text-slate-600 dark:text-slate-300">{n.label}</th>)}
              </tr>
            </thead>
            <tbody>
              <tr className="border-t border-slate-200 dark:border-slate-700">
                <td className="py-1.5 px-2 text-slate-500 font-mono">disc</td>
                {NODES.map(n => (
                  <td key={n.id} className="py-1.5 px-2 font-mono font-bold"
                    style={{ color: step.disc[n.id] >= 0 ? "#6366f1" : "#94a3b8" }}>
                    {step.disc[n.id] >= 0 ? step.disc[n.id] : "–"}
                  </td>
                ))}
              </tr>
              <tr className="border-t border-slate-200 dark:border-slate-700">
                <td className="py-1.5 px-2 text-slate-500 font-mono">low</td>
                {NODES.map(n => (
                  <td key={n.id} className="py-1.5 px-2 font-mono font-bold"
                    style={{ color: step.low[n.id] >= 0 && step.low[n.id] === step.disc[n.id] && step.disc[n.id] >= 0 ? "#ef4444" : step.low[n.id] >= 0 ? "#10b981" : "#94a3b8" }}>
                    {step.low[n.id] >= 0 ? step.low[n.id] : "–"}
                  </td>
                ))}
              </tr>
              <tr className="border-t border-slate-200 dark:border-slate-700">
                <td className="py-1.5 px-2 text-slate-500 font-mono">inStk</td>
                {NODES.map(n => (
                  <td key={n.id} className={`py-1.5 px-2 font-mono font-bold ${
                    step.inStack[n.id] ? "text-indigo-500" : "text-slate-300 dark:text-slate-600"
                  }`}>{step.inStack[n.id] ? "✓" : "✗"}</td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>

        {/* Description */}
        <div className={`rounded-xl px-4 py-2.5 text-sm min-h-[48px] border transition-all duration-300 ${
          isDone
            ? "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700/50 text-emerald-800 dark:text-emerald-300"
            : step.highlightPop.length > 0
            ? "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-700/50 text-red-800 dark:text-red-300"
            : "bg-rose-50 dark:bg-rose-900/20 border-rose-200 dark:border-rose-700/50 text-rose-800 dark:text-rose-300"
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
              "bg-rose-600 hover:bg-rose-700 text-white"
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
              onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-rose-500" />
            <span className="text-[10px] text-slate-400">{(speed/1000).toFixed(1)}s</span>
          </div>
        </div>
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
          <div className="h-1.5 rounded-full bg-gradient-to-r from-rose-500 to-orange-500 transition-all duration-300"
            style={{ width: `${(stepIdx/(STEPS.length-1))*100}%` }} />
        </div>
      </div>
    </div>
  );
}
