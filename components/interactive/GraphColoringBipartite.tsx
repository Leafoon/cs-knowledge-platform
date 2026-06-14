"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── Graph presets ──────────────────────────────────────────────────────── */
interface GNode { id: number; x: number; y: number; }
interface GEdge { u: number; v: number; }
interface GraphPreset {
  name: string;
  emoji: string;
  desc: string;
  nodes: GNode[];
  edges: GEdge[];
  adj: Record<number, number[]>;
}

const PRESETS: GraphPreset[] = [
  {
    name: "二部图（完全二分图 K₃,₂）",
    emoji: "✅",
    desc: "左边 3 节点，右边 2 节点，所有边跨越两侧——典型二部图",
    nodes: [
      { id: 0, x: 70,  y: 60 },
      { id: 1, x: 70,  y: 130 },
      { id: 2, x: 70,  y: 200 },
      { id: 3, x: 250, y: 95 },
      { id: 4, x: 250, y: 165 },
    ],
    edges: [
      { u: 0, v: 3 }, { u: 0, v: 4 },
      { u: 1, v: 3 }, { u: 1, v: 4 },
      { u: 2, v: 3 }, { u: 2, v: 4 },
    ],
    adj: { 0:[3,4], 1:[3,4], 2:[3,4], 3:[0,1,2], 4:[0,1,2] },
  },
  {
    name: "非二部图（5-奇环）",
    emoji: "❌",
    desc: "5 个节点围成奇数环（长度 5），不能 2-着色——不是二部图",
    nodes: [
      { id: 0, x: 160, y: 40  },
      { id: 1, x: 275, y: 120 },
      { id: 2, x: 230, y: 220 },
      { id: 3, x: 90,  y: 220 },
      { id: 4, x: 45,  y: 120 },
    ],
    edges: [
      { u: 0, v: 1 }, { u: 1, v: 2 },
      { u: 2, v: 3 }, { u: 3, v: 4 }, { u: 4, v: 0 },
    ],
    adj: { 0:[1,4], 1:[0,2], 2:[1,3], 3:[2,4], 4:[3,0] },
  },
  {
    name: "非二部图（含三角形）",
    emoji: "❌",
    desc: "包含三角形（3-奇环）——BFS 着色时立即发现冲突",
    nodes: [
      { id: 0, x: 80,  y: 80  },
      { id: 1, x: 240, y: 80  },
      { id: 2, x: 160, y: 200 },
      { id: 3, x: 80,  y: 200 },
      { id: 4, x: 240, y: 200 },
    ],
    edges: [
      { u: 0, v: 1 }, { u: 1, v: 2 }, { u: 0, v: 2 },
      { u: 0, v: 3 }, { u: 1, v: 4 },
    ],
    adj: { 0:[1,2,3], 1:[0,2,4], 2:[1,0], 3:[0], 4:[1] },
  },
];

/* ─── Step simulation ────────────────────────────────────────────────────── */
type Color = -1 | 0 | 1;  // -1=unset, 0=blue, 1=orange

interface BfsColorStep {
  colors: Color[];
  queue: number[];
  current: number | null;
  conflictEdge: [number, number] | null;
  isBipartite: boolean | null;   // null = not yet determined
  description: string;
}

function simulateBFSColor(preset: GraphPreset): BfsColorStep[] {
  const n = preset.nodes.length;
  const colors: Color[] = Array(n).fill(-1);
  const steps: BfsColorStep[] = [];
  let bipartite: boolean | null = null;
  let conflictEdge: [number, number] | null = null;

  const snap = (queue: number[], current: number | null, ce: [number,number]|null, desc: string) => {
    steps.push({
      colors: [...colors] as Color[],
      queue: [...queue],
      current,
      conflictEdge: ce,
      isBipartite: bipartite,
      description: desc,
    });
  };

  snap([], null, null, "初始状态：所有节点未着色（灰色）");

  outer:
  for (let start = 0; start < n; start++) {
    if (colors[start] !== -1) continue;
    colors[start] = 0;
    const queue: number[] = [start];
    snap([...queue], null, null, `从节点 ${start} 开始 BFS，着色为 🔵 蓝色（颜色 0）`);

    while (queue.length > 0 && bipartite === null) {
      const u = queue.shift()!;
      snap([...queue], u, null, `出队节点 ${u}（颜色${colors[u]===0?"🔵蓝":"🟠橙"}），检查其邻居`);

      for (const v of preset.adj[u]) {
        if (colors[v] === -1) {
          colors[v] = (1 - colors[u]) as Color;
          queue.push(v);
          snap([...queue], u, null, `邻居 ${v} 未着色 → 着色为 ${colors[v]===0?"🔵 蓝色":"🟠 橙色"}（与节点 ${u} 相反），入队`);
        } else if (colors[v] === colors[u]) {
          bipartite = false;
          conflictEdge = [u, v];
          snap([...queue], u, [u, v], `❌ 冲突！节点 ${u} 和邻居 ${v} 同色（${colors[u]===0?"🔵蓝":"🟠橙"}），存在奇数环，不是二部图！`);
          break outer;
        } else {
          snap([...queue], u, null, `邻居 ${v} 已着色（${colors[v]===0?"🔵蓝":"🟠橙"}），与节点 ${u} 颜色不同 ✓，无冲突`);
        }
      }
    }
  }

  if (bipartite === null) {
    bipartite = true;
    snap([], null, null, "✅ BFS 着色完成！每条边两端颜色不同，这是二部图（可 2-着色）！");
  }

  return steps;
}

/* ─── Node color helpers ─────────────────────────────────────────────────── */
function nodeCircle(c: Color, isCurrent: boolean, isConflict: boolean): { fill: string; stroke: string } {
  if (isConflict) return { fill: "#ef4444", stroke: "#b91c1c" };
  if (isCurrent) return { fill: "#f59e0b", stroke: "#d97706" };
  if (c === 0) return { fill: "#0ea5e9", stroke: "#0284c7" };
  if (c === 1) return { fill: "#f97316", stroke: "#ea580c" };
  return { fill: "#94a3b8", stroke: "#64748b" };
}

const COLOR_DOT = ["bg-sky-400", "bg-orange-400"];
const COLOR_LABEL = ["🔵 蓝（集合 U）", "🟠 橙（集合 W）"];

/* ─── Main Component ─────────────────────────────────────────────────────── */
export default function GraphColoringBipartite() {
  const [presetIdx, setPresetIdx] = useState(0);
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1000);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const preset = PRESETS[presetIdx];
  const [allSteps, setAllSteps] = useState<BfsColorStep[]>(() => simulateBFSColor(PRESETS[0]));

  const switchPreset = (idx: number) => {
    setPlaying(false);
    setPresetIdx(idx);
    setStepIdx(0);
    setAllSteps(simulateBFSColor(PRESETS[idx]));
  };

  const step = allSteps[stepIdx];
  const maxStep = allSteps.length - 1;

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

  const isConflictNode = (id: number) =>
    step.conflictEdge !== null && (step.conflictEdge[0] === id || step.conflictEdge[1] === id);

  const isConflictEdge = (u: number, v: number) =>
    step.conflictEdge !== null &&
    ((step.conflictEdge[0]===u && step.conflictEdge[1]===v) ||
     (step.conflictEdge[0]===v && step.conflictEdge[1]===u));

  // Edge coloring based on both endpoints' colors
  function edgeStyle(u: number, v: number): { stroke: string; width: number; dash: string } {
    if (isConflictEdge(u, v)) return { stroke: "#ef4444", width: 3, dash: "none" };
    const cu = step.colors[u], cv = step.colors[v];
    if (cu !== -1 && cv !== -1 && cu !== cv) return { stroke: "#6366f1", width: 2, dash: "none" };
    return { stroke: "#cbd5e1", width: 1.5, dash: "4 3" };
  }

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-teal-500 via-emerald-500 to-green-500 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">二部图 BFS 2-着色动画</h3>
        <p className="text-teal-100 text-sm mt-0.5">用 BFS 交替着色判断是否为二部图——遇到同色相邻即发现奇数环</p>
      </div>

      <div className="p-5 space-y-4">
        {/* Preset selector */}
        <div className="flex flex-col sm:flex-row gap-2">
          {PRESETS.map((p, i) => (
            <button key={i} onClick={() => switchPreset(i)}
              className={`flex-1 text-left px-3 py-2 rounded-xl border transition-all ${
                presetIdx === i
                  ? "bg-emerald-50 dark:bg-emerald-900/30 border-emerald-300 dark:border-emerald-700 shadow-sm"
                  : "border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800"
              }`}>
              <div className="text-sm font-bold text-slate-700 dark:text-slate-200">{p.emoji} {p.name}</div>
              <div className="text-[10px] text-slate-400 dark:text-slate-500 mt-0.5">{p.desc}</div>
            </button>
          ))}
        </div>

        <div className="flex flex-col sm:flex-row gap-4">
          {/* SVG */}
          <div className="sm:w-[300px] flex-shrink-0 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-2">
            <svg viewBox="0 0 320 255" className="w-full">
              {/* Edges */}
              {preset.edges.map(({ u, v }) => {
                const nu = preset.nodes[u], nv = preset.nodes[v];
                const { stroke, width, dash } = edgeStyle(u, v);
                return (
                  <line key={`${u}-${v}`}
                    x1={nu.x} y1={nu.y} x2={nv.x} y2={nv.y}
                    stroke={stroke} strokeWidth={width}
                    strokeDasharray={dash === "none" ? undefined : dash}
                    className="transition-all duration-300" />
                );
              })}
              {/* Nodes */}
              {preset.nodes.map(node => {
                const isCur = step.current === node.id;
                const isCon = isConflictNode(node.id);
                const { fill, stroke } = nodeCircle(step.colors[node.id], isCur, isCon);
                return (
                  <g key={node.id}>
                    {(isCur || isCon) && (
                      <circle cx={node.x} cy={node.y} r={24}
                        fill={isCon ? "#fee2e2" : "#fef3c7"} stroke={isCon ? "#fca5a5" : "#fcd34d"}
                        strokeWidth={1.5} opacity={0.7} />
                    )}
                    <circle cx={node.x} cy={node.y} r={18}
                      fill={fill} stroke={stroke} strokeWidth={2}
                      className="transition-all duration-300" />
                    <text x={node.x} y={node.y} textAnchor="middle" dominantBaseline="central"
                      fontSize={13} fontWeight="bold" fill="white">{node.id}</text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Right panel */}
          <div className="flex-1 space-y-3">
            {/* Result banner */}
            {step.isBipartite !== null && (
              <div className={`rounded-xl border px-4 py-3 text-sm font-bold text-center ${
                step.isBipartite
                  ? "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-300 dark:border-emerald-700 text-emerald-700 dark:text-emerald-300"
                  : "bg-rose-50 dark:bg-rose-900/20 border-rose-300 dark:border-rose-700 text-rose-700 dark:text-rose-300"
              }`}>
                {step.isBipartite ? "✅ 是二部图！可以 2-着色" : "❌ 不是二部图！存在奇数环"}
              </div>
            )}

            {/* Queue */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-3">
              <div className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase mb-2 tracking-wide">BFS 队列</div>
              <div className="flex flex-wrap gap-1.5 min-h-[28px]">
                {step.queue.length === 0 ? (
                  <span className="text-[11px] text-slate-400 italic">空</span>
                ) : (
                  step.queue.map((nid, qi) => (
                    <span key={qi}
                      className={`inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-bold border ${
                        step.colors[nid] === 0
                          ? "bg-sky-100 dark:bg-sky-900/30 border-sky-300 dark:border-sky-700 text-sky-700 dark:text-sky-300"
                          : "bg-orange-100 dark:bg-orange-900/30 border-orange-300 dark:border-orange-700 text-orange-700 dark:text-orange-300"
                      }`}>
                      {nid} <span className="text-[9px] opacity-70">{step.colors[nid] === 0 ? "蓝" : "橙"}</span>
                    </span>
                  ))
                )}
              </div>
            </div>

            {/* Color set display */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-3">
              <div className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase mb-2 tracking-wide">分组结果</div>
              {[0, 1].map(c => (
                <div key={c} className="flex items-center gap-2 mb-1.5">
                  <span className={`inline-block w-3 h-3 rounded-full flex-shrink-0 ${COLOR_DOT[c]}`} />
                  <span className="text-[10px] text-slate-500 dark:text-slate-400 w-28">{COLOR_LABEL[c]}</span>
                  <div className="flex flex-wrap gap-1">
                    {preset.nodes.map(n => step.colors[n.id] === c && (
                      <span key={n.id} className={`text-[10px] font-bold font-mono px-1.5 py-0.5 rounded ${c === 0 ? "bg-sky-100 dark:bg-sky-900/40 text-sky-700 dark:text-sky-300" : "bg-orange-100 dark:bg-orange-900/40 text-orange-700 dark:text-orange-300"}`}>
                        {n.id}
                      </span>
                    ))}
                    {!preset.nodes.some(n => step.colors[n.id] === c) && (
                      <span className="text-[10px] text-slate-300 dark:text-slate-600">（空）</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Description */}
        <div className={`rounded-xl border px-4 py-3 ${
          step.conflictEdge
            ? "bg-rose-50 dark:bg-rose-900/20 border-rose-200 dark:border-rose-800"
            : "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800"
        }`}>
          <div className={`text-xs leading-relaxed ${step.conflictEdge ? "text-rose-700 dark:text-rose-300" : "text-emerald-700 dark:text-emerald-300"}`}>
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
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.max(0, i-1)); }} disabled={stepIdx===0}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            ← 上一步
          </button>
          <button onClick={() => setPlaying(p => !p)}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-colors ${playing ? "bg-amber-500 hover:bg-amber-600 text-white" : "bg-emerald-500 hover:bg-emerald-600 text-white"}`}>
            {playing ? "⏸ 暂停" : "▶ 播放"}
          </button>
          <button onClick={() => { setPlaying(false); setStepIdx(i => Math.min(maxStep, i+1)); }} disabled={stepIdx===maxStep}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
            下一步 →
          </button>
          <div className="flex items-center gap-2 ml-auto">
            <input type="range" min={400} max={1800} step={200} value={speed}
              onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-emerald-500" />
          </div>
        </div>
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
          <div className={`h-1.5 rounded-full transition-all duration-300 ${step.isBipartite === false ? "bg-rose-500" : "bg-emerald-500"}`}
            style={{ width: `${(stepIdx / maxStep) * 100}%` }} />
        </div>
      </div>
    </div>
  );
}
