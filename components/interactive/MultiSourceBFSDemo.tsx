"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

/* ─── Grid types ──────────────────────────────────────────────────────────── */
type Cell = 0 | 1 | 2; // 0=empty 1=fresh 2=rotten
type Grid = Cell[][];

const ROWS = 5, COLS = 6;

/* ─── Presets ─────────────────────────────────────────────────────────────── */
const PRESETS: { label: string; icon: string; desc: string; grid: Grid }[] = [
  {
    label: "角落扩散",
    icon: "↗",
    desc: "左上和右下各一个烂橙子，两路同时扩散",
    grid: [
      [2,1,1,1,1,1],
      [1,1,1,1,1,1],
      [1,1,1,1,1,1],
      [1,1,1,1,1,1],
      [1,1,1,1,1,2],
    ],
  },
  {
    label: "中心爆发",
    icon: "✦",
    desc: "中央单源扩散，并有空格隔断部分区域",
    grid: [
      [1,1,0,1,1,1],
      [1,1,0,1,1,1],
      [1,1,2,1,1,1],
      [1,1,0,1,1,1],
      [1,1,1,1,1,1],
    ],
  },
  {
    label: "孤立区域",
    icon: "⊗",
    desc: "右上角新鲜橙子被空格完全隔离，永远腐烂不了",
    grid: [
      [2,1,1,0,1,1],
      [1,1,1,0,1,1],
      [1,1,1,0,1,1],
      [1,1,1,0,0,1],
      [1,1,1,1,1,1],
    ],
  },
];

/* ─── BFS simulation ─────────────────────────────────────────────────────── */
interface BFSStep {
  grid: Grid;       // snapshot of grid at this step (which cells are rotten)
  timeMap: number[][];
  frontier: [number, number][];  // cells rotted THIS step
  time: number;
}

function cloneGrid(g: Grid): Grid { return g.map(r => [...r]) as Grid; }

function simulateMultiBFS(initial: Grid): BFSStep[] {
  const steps: BFSStep[] = [];
  const g = cloneGrid(initial);
  const timeMap = Array.from({ length: ROWS }, () => Array(COLS).fill(-1));
  const queue: [number, number][] = [];

  // Seed queue with all rotten cells (time 0)
  for (let r = 0; r < ROWS; r++)
    for (let c = 0; c < COLS; c++)
      if (g[r][c] === 2) { queue.push([r, c]); timeMap[r][c] = 0; }

  // Record step 0 (initial state)
  steps.push({ grid: cloneGrid(g), timeMap: timeMap.map(r => [...r]), frontier: queue.map(x => [x[0], x[1]] as [number,number]), time: 0 });

  let head = 0;
  const dirs = [[0,1],[0,-1],[1,0],[-1,0]];

  while (head < queue.length) {
    const levelEnd = queue.length;
    const newFrontier: [number, number][] = [];

    while (head < levelEnd) {
      const [r, c] = queue[head++];
      const t = timeMap[r][c];

      for (const [dr, dc] of dirs) {
        const nr = r + dr, nc = c + dc;
        if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) continue;
        if (g[nr][nc] !== 1) continue;
        g[nr][nc] = 2;
        timeMap[nr][nc] = t + 1;
        queue.push([nr, nc]);
        newFrontier.push([nr, nc]);
      }
    }

    if (newFrontier.length > 0) {
      steps.push({ grid: cloneGrid(g), timeMap: timeMap.map(r => [...r]), frontier: newFrontier, time: timeMap[newFrontier[0][0]][newFrontier[0][1]] });
    }
  }

  return steps;
}

function hasIsolatedFresh(steps: BFSStep[]): boolean {
  const last = steps[steps.length - 1];
  for (let r = 0; r < ROWS; r++)
    for (let c = 0; c < COLS; c++)
      if (last.grid[r][c] === 1) return true;
  return false;
}

/* ─── Cell color helpers ─────────────────────────────────────────────────── */
function editCellClass(cell: Cell) {
  if (cell === 0) return "bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600 hover:bg-slate-200 dark:hover:bg-slate-700";
  if (cell === 1) return "bg-emerald-100 dark:bg-emerald-900/50 border-emerald-400 dark:border-emerald-600 hover:bg-emerald-200";
  return "bg-orange-400 dark:bg-orange-600 border-orange-500 dark:border-orange-700";
}

/* ─── Main Component ─────────────────────────────────────────────────────── */
export default function MultiSourceBFSDemo() {
  const [presetIdx, setPresetIdx] = useState(0);
  const [editGrid, setEditGrid] = useState<Grid>(() => cloneGrid(PRESETS[0].grid));
  const [steps, setSteps] = useState<BFSStep[]>([]);
  const [stepIdx, setStepIdx] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(700);
  const [mode, setMode] = useState<"edit" | "run">("edit");
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load preset
  const loadPreset = (idx: number) => {
    setPresetIdx(idx);
    setEditGrid(cloneGrid(PRESETS[idx].grid));
    setSteps([]);
    setStepIdx(-1);
    setPlaying(false);
    setMode("edit");
  };

  // Toggle cell in edit mode (cycle 0→1→2→0)
  const toggleCell = (r: number, c: number) => {
    if (mode !== "edit") return;
    setEditGrid(prev => {
      const g = cloneGrid(prev);
      g[r][c] = ((g[r][c] + 1) % 3) as Cell;
      return g;
    });
  };

  // Run BFS
  const runBFS = () => {
    const computed = simulateMultiBFS(editGrid);
    setSteps(computed);
    setStepIdx(0);
    setPlaying(true);
    setMode("run");
  };

  const advance = useCallback(() => {
    setStepIdx(prev => {
      if (prev >= steps.length - 1) { setPlaying(false); return prev; }
      return prev + 1;
    });
  }, [steps.length]);

  useEffect(() => {
    if (playing) { intervalRef.current = setInterval(advance, speed); }
    else { if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; } }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, speed, advance]);

  const resetToBefore = () => { setPlaying(false); setStepIdx(0); setMode("run"); };
  const backToEdit = () => { setPlaying(false); setStepIdx(-1); setMode("edit"); };

  // Determine display grid and final state
  const displayStep = mode === "run" && steps.length > 0 && stepIdx >= 0 ? steps[stepIdx] : null;
  const displayGrid = displayStep ? displayStep.grid : editGrid;
  const timeMap = displayStep ? displayStep.timeMap : null;
  const frontierSet = displayStep ? new Set(displayStep.frontier.map(([r,c]) => `${r},${c}`)) : new Set<string>();
  const isolated = mode === "run" && stepIdx === steps.length - 1 ? hasIsolatedFresh(steps) : false;
  const totalTime = displayStep ? displayStep.time : 0;

  return (
    <div className="w-full max-w-2xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-amber-500 via-orange-500 to-red-500 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">多源 BFS——腐烂橙子模拟</h3>
        <p className="text-amber-100 text-sm mt-0.5">所有烂橙子同时作为 BFS 起点，观察波纹状同步扩散的过程</p>
      </div>

      <div className="p-5 space-y-4">
        {/* Presets */}
        <div className="flex gap-2 flex-wrap">
          {PRESETS.map((p, i) => (
            <button key={i} onClick={() => loadPreset(i)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                presetIdx === i
                  ? "bg-orange-500 border-orange-600 text-white shadow-md shadow-orange-200 dark:shadow-orange-900/50"
                  : "border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-300 hover:border-orange-400 hover:text-orange-600 dark:hover:text-orange-400"
              }`}>
              <span>{p.icon}</span> {p.label}
            </button>
          ))}
        </div>
        <p className="text-[11px] text-slate-500 dark:text-slate-400">{PRESETS[presetIdx].desc}</p>

        {/* Legend */}
        <div className="flex flex-wrap items-center gap-3 text-[11px]">
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 rounded bg-slate-200 dark:bg-slate-700 border border-slate-400 dark:border-slate-500" />
            <span className="text-slate-600 dark:text-slate-300">空格（0）</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 rounded bg-emerald-200 dark:bg-emerald-800 border border-emerald-500" />
            <span className="text-slate-600 dark:text-slate-300">新鲜橙子（1）</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 rounded bg-orange-400 dark:bg-orange-600 border border-orange-600" />
            <span className="text-slate-600 dark:text-slate-300">腐烂橙子（2）</span>
          </div>
          {mode === "edit" && (
            <span className="ml-auto text-amber-600 dark:text-amber-400 font-medium">⬆ 点击格子切换状态</span>
          )}
        </div>

        {/* Grid */}
        <div className="flex flex-col gap-1">
          {Array.from({ length: ROWS }, (_, r) => (
            <div key={r} className="flex gap-1">
              {Array.from({ length: COLS }, (_, c) => {
                const cell = displayGrid[r][c];
                const t = timeMap ? timeMap[r][c] : -1;
                const isFrontier = frontierSet.has(`${r},${c}`);
                const isRotten = cell === 2;
                const isEmpty = cell === 0;
                const isFresh = cell === 1;
                return (
                  <div key={c} onClick={() => toggleCell(r, c)}
                    className={`relative flex-1 aspect-square rounded flex items-center justify-center border-2 transition-all duration-300 select-none ${
                      mode === "edit" ? "cursor-pointer " + editCellClass(cell) :
                      isFrontier && isRotten
                        ? "bg-red-400 dark:bg-red-600 border-red-500 dark:border-red-700 scale-110 shadow-lg shadow-red-300 dark:shadow-red-900/60"
                        : isRotten
                          ? "bg-orange-300 dark:bg-orange-700 border-orange-400 dark:border-orange-600"
                          : isFresh
                            ? "bg-emerald-100 dark:bg-emerald-900/50 border-emerald-400 dark:border-emerald-600"
                            : "bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600"
                    }`}>
                    {isEmpty && <span className="text-slate-400 dark:text-slate-600 text-base">✕</span>}
                    {isFresh && <span className="text-lg">🍊</span>}
                    {isRotten && mode === "edit" && <span className="text-lg">🟠</span>}
                    {isRotten && mode === "run" && t >= 0 && (
                      <span className={`font-bold text-sm ${isFrontier ? "text-white" : "text-orange-700 dark:text-orange-200"}`}>{t}</span>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>

        {/* Status bar */}
        {mode === "run" && displayStep && (
          <div className={`rounded-xl px-4 py-2.5 text-sm font-semibold flex items-center justify-between transition-all ${
            isolated
              ? "bg-rose-100 dark:bg-rose-900/30 text-rose-700 dark:text-rose-300 border border-rose-300 dark:border-rose-700"
              : stepIdx === steps.length - 1
                ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-700"
                : "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 border border-amber-300 dark:border-amber-700"
          }`}>
            <span>
              {isolated
                ? "❌ 存在孤立新鲜橙子，无法全部腐烂（返回 -1）"
                : stepIdx === steps.length - 1
                  ? `✅ 全部腐烂完成！总耗时 ${totalTime} 分钟`
                  : `⏱ 第 ${totalTime} 分钟正在扩散…（共 ${steps.length - 1} 分钟）`}
            </span>
            <span className="text-[10px] font-mono opacity-60">步骤 {stepIdx + 1}/{steps.length}</span>
          </div>
        )}

        {/* Controls */}
        <div className="space-y-2">
          {mode === "edit" ? (
            <button onClick={runBFS}
              className="w-full py-2 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-white font-bold text-sm shadow-md shadow-amber-200 dark:shadow-amber-900/40 transition-all">
              ▶ 运行多源 BFS
            </button>
          ) : (
            <div className="flex flex-wrap items-center gap-2">
              <button onClick={backToEdit}
                className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
                ✎ 重新编辑
              </button>
              <button onClick={resetToBefore}
                className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
                ↺ 重播
              </button>
              <button onClick={() => { setPlaying(false); setStepIdx(i => Math.max(0, i - 1)); }} disabled={stepIdx <= 0}
                className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
                ← 上一步
              </button>
              <button onClick={() => setPlaying(p => !p)} disabled={stepIdx === steps.length - 1}
                className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-colors ${playing ? "bg-amber-500 hover:bg-amber-600 text-white" : "bg-orange-500 hover:bg-orange-600 text-white"}`}>
                {playing ? "⏸ 暂停" : "▶ 播放"}
              </button>
              <button onClick={() => { setPlaying(false); setStepIdx(i => Math.min(steps.length - 1, i + 1)); }} disabled={stepIdx === steps.length - 1}
                className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors">
                下一步 →
              </button>
              <div className="flex items-center gap-1.5 ml-auto">
                <input type="range" min={300} max={1500} step={100} value={speed}
                  onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-orange-500" />
                <span className="text-[10px] text-slate-400">{(speed/1000).toFixed(1)}s</span>
              </div>
            </div>
          )}
          {mode === "run" && steps.length > 0 && (
            <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
              <div className={`h-1.5 rounded-full transition-all duration-300 ${isolated ? "bg-rose-500" : "bg-orange-500"}`}
                style={{ width: `${steps.length <= 1 ? 100 : (stepIdx / (steps.length - 1)) * 100}%` }} />
            </div>
          )}
        </div>

        {/* Key insight */}
        <div className="rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700/50 p-4">
          <div className="text-amber-800 dark:text-amber-300 font-bold text-xs mb-2">💡 多源 BFS 核心要点</div>
          <ul className="text-amber-700 dark:text-amber-400 text-[11px] space-y-1 leading-relaxed">
            <li>• 将所有初始腐烂格子 <strong>同时</strong> 加入队列（时间戳 = 0），而非逐个启动 BFS</li>
            <li>• 每次扩散恰好消耗 1 分钟，最终答案 = BFS 最大层数（时间戳最大值）</li>
            <li>• 若存在被空格完全隔离的新鲜橙子，BFS 结束后仍有 <code className="font-mono">grid[r][c] == 1</code>，返回 <code className="font-mono">-1</code></li>
            <li>• 时间复杂度 O(ROWS × COLS)，空间复杂度 O(ROWS × COLS)</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
