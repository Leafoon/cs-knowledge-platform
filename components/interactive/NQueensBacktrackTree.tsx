'use client'
import { useState, useEffect, useMemo } from 'react'

/* ─── N 皇后回溯动画 ───────────────────────────────────── */
type StepKind = 'try' | 'place' | 'reject' | 'backtrack' | 'solution'
interface NQStep {
  kind: StepKind; queens: number[]; row: number; col: number; msg: string
  solCount: number
}

function generateSteps(n: number): NQStep[] {
  const steps: NQStep[] = []
  const queens = new Array(n).fill(-1)
  const cols = new Set<number>()
  const d1   = new Set<number>()
  const d2   = new Set<number>()
  let solCount = 0

  function bt(row: number) {
    if (row === n) {
      solCount++
      steps.push({ kind: 'solution', queens: [...queens], row, col: -1,
        msg: `✓ 完整解！布局 [${queens.join(', ')}]`, solCount })
      return
    }
    for (let c = 0; c < n; c++) {
      steps.push({ kind: 'try', queens: [...queens], row, col: c,
        msg: `尝试第 ${row} 行，第 ${c} 列`, solCount })
      if (cols.has(c) || d1.has(row - c) || d2.has(row + c)) {
        steps.push({ kind: 'reject', queens: [...queens], row, col: c,
          msg: `✗ 列/对角线冲突，剪枝第 ${row} 行第 ${c} 列`, solCount })
        continue
      }
      queens[row] = c; cols.add(c); d1.add(row - c); d2.add(row + c)
      steps.push({ kind: 'place', queens: [...queens], row, col: c,
        msg: `♛ 放置皇后 → 第 ${row} 行，第 ${c} 列`, solCount })
      bt(row + 1)
      queens[row] = -1; cols.delete(c); d1.delete(row - c); d2.delete(row + c)
      if (row > 0)
        steps.push({ kind: 'backtrack', queens: [...queens], row, col: c,
          msg: `↩ 回溯，撤销第 ${row} 行第 ${c} 列的皇后`, solCount })
    }
  }
  bt(0)
  return steps
}

const KIND_STYLE: Record<StepKind, { bar: string; badge: string; icon: string; label: string }> = {
  try:       { bar: 'bg-amber-400',   badge: 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 border-amber-300 dark:border-amber-700',   icon: '👁', label: '尝试' },
  place:     { bar: 'bg-emerald-500', badge: 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300 border-emerald-300 dark:border-emerald-700', icon: '♛', label: '放置' },
  reject:    { bar: 'bg-rose-500',    badge: 'bg-rose-100 dark:bg-rose-900/40 text-rose-700 dark:text-rose-300 border-rose-300 dark:border-rose-700',       icon: '✗', label: '剪枝' },
  backtrack: { bar: 'bg-violet-500',  badge: 'bg-violet-100 dark:bg-violet-900/40 text-violet-700 dark:text-violet-300 border-violet-300 dark:border-violet-700', icon: '↩', label: '回溯' },
  solution:  { bar: 'bg-sky-500',     badge: 'bg-sky-100 dark:bg-sky-900/40 text-sky-700 dark:text-sky-300 border-sky-300 dark:border-sky-700',             icon: '✓', label: '解！' },
}

export default function NQueensBacktrackTree() {
  const [n, setN]           = useState(4)
  const [step, setStep]     = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed]   = useState(300)

  const steps = useMemo(() => generateSteps(n), [n])
  const cur   = steps[Math.min(step, steps.length - 1)]

  useEffect(() => { setStep(0); setPlaying(false) }, [n])
  useEffect(() => {
    if (!playing) return
    if (step >= steps.length - 1) { setPlaying(false); return }
    const id = setTimeout(() => setStep(s => s + 1), speed)
    return () => clearTimeout(id)
  }, [playing, step, steps.length, speed])

  const pct = steps.length > 1 ? Math.round((step / (steps.length - 1)) * 100) : 0
  const ks  = KIND_STYLE[cur.kind]

  function cellClass(r: number, c: number) {
    const isLight = (r + c) % 2 === 0
    const isCurRow = cur.row === r
    const isQueen  = cur.queens[r] === c && cur.queens[r] !== -1
    const isTryCell = isCurRow && cur.col === c && (cur.kind === 'try')
    const isReject  = isCurRow && cur.col === c && cur.kind === 'reject'
    if (isQueen)  return 'bg-emerald-400 dark:bg-emerald-500 text-white'
    if (isReject) return 'bg-rose-400 dark:bg-rose-600 text-white ring-2 ring-rose-500'
    if (isTryCell) return 'bg-amber-200 dark:bg-amber-700 text-amber-700 dark:text-amber-200'
    if (isCurRow && cur.kind !== 'solution') return isLight ? 'bg-amber-50 dark:bg-amber-950/30' : 'bg-amber-100/60 dark:bg-amber-900/20'
    return isLight ? 'bg-slate-100 dark:bg-slate-700' : 'bg-slate-200 dark:bg-slate-600'
  }

  const cellSize = n <= 4 ? 52 : 40

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-lg bg-white dark:bg-slate-900">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-600 to-purple-700 px-6 py-4 flex items-center gap-3">
        <span className="text-2xl">♛</span>
        <div>
          <h3 className="text-white font-bold text-base">N 皇后回溯动画</h3>
          <p className="text-purple-200 text-xs">逐步展示 DFS 搜索 + 剪枝全过程</p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <span className="text-purple-200 text-xs">N =</span>
          {[4, 5].map(v => (
            <button key={v} onClick={() => setN(v)}
              className={`w-8 h-8 rounded-lg text-sm font-bold transition-all ${
                n === v ? 'bg-white text-purple-700 shadow' : 'bg-purple-500/40 text-white hover:bg-purple-500/60'
              }`}>{v}</button>
          ))}
        </div>
      </div>

      <div className="p-5 grid grid-cols-[auto_1fr] gap-5 items-start">
        {/* ── Board ── */}
        <div className="flex flex-col items-center gap-3">
          <div className="border-2 border-slate-300 dark:border-slate-600 rounded-xl overflow-hidden shadow-inner"
            style={{ display: 'grid', gridTemplateColumns: `repeat(${n}, ${cellSize}px)` }}>
            {Array.from({ length: n }, (_, r) =>
              Array.from({ length: n }, (_, c) => (
                <div key={`${r}-${c}`}
                  style={{ width: cellSize, height: cellSize }}
                  className={`flex items-center justify-center font-bold text-xl transition-all duration-100 select-none ${cellClass(r, c)}`}>
                  {cur.queens[r] === c && cur.queens[r] !== -1 ? '♛'
                    : cur.row === r && cur.col === c && cur.kind === 'reject' ? '✗'
                    : cur.row === r && cur.col === c && cur.kind === 'try' ? '·' : ''}
                </div>
              ))
            )}
          </div>

          {/* queens[] array */}
          <div className="w-full bg-slate-50 dark:bg-slate-800 rounded-lg p-2 border border-slate-200 dark:border-slate-700">
            <p className="text-[10px] text-slate-400 dark:text-slate-500 mb-1 font-medium">queens[row] = col</p>
            <div className="flex gap-1">
              {cur.queens.map((col, i) => (
                <div key={i} className={`flex-1 rounded py-1 text-center text-xs font-mono font-bold border transition-all ${
                  col === -1
                    ? 'bg-white dark:bg-slate-700 text-slate-300 dark:text-slate-500 border-slate-200 dark:border-slate-600'
                    : 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300 border-emerald-300 dark:border-emerald-700'
                }`}>
                  <div className="text-[9px] text-slate-400 dark:text-slate-500">r{i}</div>
                  {col === -1 ? '—' : col}
                </div>
              ))}
            </div>
          </div>

          <div className="flex gap-3 w-full">
            <div className="flex-1 bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-800 rounded-lg p-2 text-center">
              <p className="text-[10px] text-emerald-600 dark:text-emerald-400">已找到解</p>
              <p className="text-xl font-black text-emerald-600 dark:text-emerald-400">{cur.solCount}</p>
            </div>
            <div className="flex-1 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-2 text-center">
              <p className="text-[10px] text-slate-500 dark:text-slate-400">总步数</p>
              <p className="text-xl font-black text-slate-600 dark:text-slate-300">{steps.length}</p>
            </div>
          </div>
        </div>

        {/* ── Info ── */}
        <div className="flex flex-col gap-3">
          {/* Current step badge */}
          <div className={`rounded-xl border p-4 transition-all ${ks.badge}`}>
            <div className="flex items-center gap-2 mb-1.5">
              <span className="text-base">{ks.icon}</span>
              <span className="text-xs font-bold uppercase tracking-wider opacity-70">{ks.label}</span>
              <span className="ml-auto text-xs opacity-60">Step {step + 1}</span>
            </div>
            <p className="text-sm font-medium leading-relaxed">{cur.msg}</p>
          </div>

          {/* Legend */}
          <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3 border border-slate-200 dark:border-slate-700">
            <p className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2">图例</p>
            <div className="grid grid-cols-2 gap-1.5">
              {Object.entries(KIND_STYLE).map(([k, v]) => (
                <div key={k} className="flex items-center gap-1.5">
                  <div className={`w-3 h-3 rounded-sm flex-shrink-0 ${v.bar}`}/>
                  <span className="text-xs text-slate-500 dark:text-slate-400">{v.icon} {v.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Complexity note */}
          <div className="bg-violet-50 dark:bg-violet-950/30 border border-violet-200 dark:border-violet-800 rounded-xl p-3">
            <p className="text-xs font-semibold text-violet-700 dark:text-violet-300 mb-1">复杂度 (N={n})</p>
            <div className="space-y-0.5 text-xs text-slate-600 dark:text-slate-400">
              <p>搜索步数：<span className="font-mono font-bold text-violet-600 dark:text-violet-400">{steps.length}</span></p>
              <p>无剪枝上界：<span className="font-mono font-bold">{n}! = {Array.from({length:n},(_,i)=>i+1).reduce((a,b)=>a*b,1)}</span></p>
              <p>解的数量：<span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">{steps.filter(s=>s.kind==='solution').length}</span></p>
            </div>
          </div>
        </div>
      </div>

      {/* ── Controls ── */}
      <div className="px-5 pb-5 space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400 w-14">步 {step+1}/{steps.length}</span>
          <div className="flex-1 bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
            <div className={`${ks.bar} h-1.5 rounded-full transition-all`} style={{ width: `${pct}%` }}/>
          </div>
          <span className="text-xs text-slate-400 w-8">{pct}%</span>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {([['⏮', () => { setStep(0); setPlaying(false) }, false],
            ['◀', () => setStep(s => Math.max(0, s-1)), step===0],
          ] as [string, () => void, boolean][]).map(([label, fn, dis]) => (
            <button key={label} onClick={fn} disabled={dis}
              className="px-3 py-1.5 rounded-lg text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 font-medium border border-slate-200 dark:border-slate-700 disabled:opacity-40">
              {label}
            </button>
          ))}
          <button onClick={() => setPlaying(p => !p)} disabled={step >= steps.length-1}
            className={`px-5 py-1.5 rounded-lg text-xs font-bold border shadow-sm transition-all disabled:opacity-40 ${
              playing ? 'bg-amber-500 border-amber-600 text-white' : 'bg-violet-600 border-violet-700 text-white hover:bg-violet-700'
            }`}>
            {playing ? '⏸ 暂停' : '▶ 播放'}
          </button>
          <button onClick={() => setStep(s => Math.min(steps.length-1, s+1))} disabled={step >= steps.length-1}
            className="px-3 py-1.5 rounded-lg text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 font-medium border border-slate-200 dark:border-slate-700 disabled:opacity-40">
            ▶ 下一步
          </button>
          <div className="ml-auto flex items-center gap-1.5">
            <span className="text-xs text-slate-400">速度</span>
            {[['慢', 700], ['中', 300], ['快', 100]] .map(([l, ms]) => (
              <button key={l as string} onClick={() => setSpeed(ms as number)}
                className={`px-2.5 py-1 rounded text-xs font-medium border ${
                  speed===ms ? 'bg-violet-600 text-white border-violet-700'
                             : 'bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 border-slate-200 dark:border-slate-700'
                }`}>{l}</button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
