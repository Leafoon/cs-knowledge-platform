'use client'
import React, { useState, useEffect, useRef } from 'react'
import { Play, Pause, SkipForward, RotateCcw, Navigation } from 'lucide-react'

// ── 网格配置 ──────────────────────────────────────────────────────
const ROWS = 7
const COLS = 7
const START: [number, number] = [0, 0]
const END:   [number, number] = [6, 6]

// 障碍物坐标集
const OBSTACLES = new Set([
  '1,2', '1,3', '1,4',
  '2,4',
  '3,2', '3,3',
  '4,1', '4,2',
  '5,4', '5,5',
])
const isObs = (r: number, c: number) => OBSTACLES.has(`${r},${c}`)
const isStart = (r: number, c: number) => r === START[0] && c === START[1]
const isEnd   = (r: number, c: number) => r === END[0]   && c === END[1]
const key = (r: number, c: number) => `${r},${c}`

// Manhattan 距离启发式
const h = (r: number, c: number) => Math.abs(r - END[0]) + Math.abs(c - END[1])

// ── A* 算法执行与步骤记录 ──────────────────────────────────────────
type CellInfo = { g: number; hVal: number; f: number; parent: string | null }

type AStarStep = {
  openSet: Set<string>                    // 开放集（待探索）
  closedSet: Set<string>                  // 关闭集（已探索）
  current: string                         // 当前节点
  gScore: Map<string, number>             // g 值
  hScore: Map<string, number>             // h 值
  fScore: Map<string, number>             // f 值
  parentMap: Map<string, string | null>   // 父节点
  path: string[]                          // 最终路径（仅 done 步骤）
  description: string
  phase: 'explore' | 'done' | 'no-path'
}

function buildAStarSteps(): AStarStep[] {
  const steps: AStarStep[] = []

  const gScore = new Map<string, number>()
  const hScore = new Map<string, number>()
  const fScore = new Map<string, number>()
  const parentMap = new Map<string, string | null>()

  const openSet = new Set<string>()
  const closedSet = new Set<string>()

  const startKey = key(...START)
  gScore.set(startKey, 0)
  hScore.set(startKey, h(...START))
  fScore.set(startKey, h(...START))
  parentMap.set(startKey, null)
  openSet.add(startKey)

  const dirs = [[-1,0],[1,0],[0,-1],[0,1]]

  const snapshot = (current: string, desc: string): AStarStep => ({
    openSet: new Set(openSet),
    closedSet: new Set(closedSet),
    current,
    gScore: new Map(gScore),
    hScore: new Map(hScore),
    fScore: new Map(fScore),
    parentMap: new Map(parentMap),
    path: [],
    description: desc,
    phase: 'explore',
  })

  const getPath = (endKey: string): string[] => {
    const path: string[] = []
    let cur: string | null | undefined = endKey
    while (cur != null) {
      path.unshift(cur)
      cur = parentMap.get(cur) ?? null
    }
    return path
  }

  let iter = 0
  const maxIter = 200

  while (openSet.size > 0 && iter++ < maxIter) {
    // Pop node with minimum f from openSet
    let current = ''
    let minF = Infinity
    openSet.forEach(k => {
      const f = fScore.get(k) ?? Infinity
      if (f < minF || (f === minF && k < current)) { minF = f; current = k }
    })

    const [cr, cc] = current.split(',').map(Number)
    const cg = gScore.get(current) ?? Infinity
    const ch = hScore.get(current) ?? h(cr, cc)
    const cf = fScore.get(current) ?? Infinity

    // Check goal
    if (isEnd(cr, cc)) {
      const path = getPath(current)
      steps.push({
        ...snapshot(current, `🎉 到达目标 (${cr},${cc})！最优路径长度 = ${cg} 步`),
        path,
        phase: 'done',
      })
      return steps
    }

    openSet.delete(current)
    closedSet.add(current)

    const neighbors: { r: number; c: number; newG: number }[] = []
    dirs.forEach(([dr, dc]) => {
      const nr = cr + dr, nc = cc + dc
      if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) return
      if (isObs(nr, nc)) return
      if (closedSet.has(key(nr, nc))) return
      neighbors.push({ r: nr, c: nc, newG: cg + 1 })
    })

    // Record snapshot before updating
    const updatedNeighbors: string[] = []

    neighbors.forEach(({ r: nr, c: nc, newG }) => {
      const nk = key(nr, nc)
      if (newG < (gScore.get(nk) ?? Infinity)) {
        parentMap.set(nk, current)
        gScore.set(nk, newG)
        hScore.set(nk, h(nr, nc))
        fScore.set(nk, newG + h(nr, nc))
        openSet.add(nk)
        updatedNeighbors.push(nk)
      }
    })

    const updDesc = updatedNeighbors.length > 0
      ? `更新邻居：${updatedNeighbors.map(nk => `(${nk}) f=${fScore.get(nk)}`).join(', ')}`
      : '无可更新邻居（均已在关闭集）'

    steps.push(snapshot(
      current,
      `探索节点 (${cr},${cc})：g=${cg}, h=${ch}, f=${cf}\n${updDesc}`,
    ))
  }

  // No path
  steps.push({
    openSet: new Set(openSet),
    closedSet: new Set(closedSet),
    current: '',
    gScore: new Map(gScore),
    hScore: new Map(hScore),
    fScore: new Map(fScore),
    parentMap: new Map(parentMap),
    path: [],
    description: '开放集为空，无法到达目标！',
    phase: 'no-path',
  })
  return steps
}

const STEPS = buildAStarSteps()

// ── 主组件 ────────────────────────────────────────────────────────
export default function AStarPathfinding() {
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(800)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const cur = STEPS[step]
  const isLast = step === STEPS.length - 1

  useEffect(() => {
    if (playing && !isLast) {
      timerRef.current = setTimeout(() => setStep(s => s + 1), speed)
    } else if (isLast) setPlaying(false)
    return () => { if (timerRef.current) clearTimeout(timerRef.current) }
  }, [playing, step, speed, isLast])

  const getCellStyle = (r: number, c: number): string => {
    const k = key(r, c)
    if (isStart(r, c)) return 'bg-emerald-500 border-emerald-600 text-white'
    if (isEnd(r, c))   return 'bg-blue-500 border-blue-600 text-white'
    if (isObs(r, c))   return 'bg-slate-700 dark:bg-slate-600 border-slate-800 dark:border-slate-500'
    if (cur.phase === 'done' && cur.path.includes(k))
      return 'bg-yellow-200 dark:bg-yellow-800 border-yellow-400 dark:border-yellow-500 text-yellow-900 dark:text-yellow-100'
    if (cur.current === k)
      return 'bg-orange-400 border-orange-500 text-white'
    if (cur.closedSet.has(k))
      return 'bg-red-100 dark:bg-red-900/50 border-red-300 dark:border-red-600 text-red-700 dark:text-red-300'
    if (cur.openSet.has(k))
      return 'bg-sky-100 dark:bg-sky-900/50 border-sky-300 dark:border-sky-600 text-sky-700 dark:text-sky-300'
    return 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-400 dark:text-slate-500'
  }

  // Get top-5 open set by f
  const topOpen = [...cur.openSet]
    .map(k => ({ k, f: cur.fScore.get(k) ?? Infinity, g: cur.gScore.get(k) ?? Infinity, hVal: cur.hScore.get(k) ?? Infinity }))
    .sort((a, b) => a.f - b.f)
    .slice(0, 6)

  return (
    <div className="w-full max-w-4xl mx-auto my-6 rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-xl">
      {/* Header */}
      <div className="bg-gradient-to-r from-sky-600 via-blue-600 to-indigo-600 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold text-white">A* 网格寻路可视化</h3>
            <p className="text-sky-100 text-sm mt-0.5">7×7 网格 · h = 曼哈顿距离 · f(n) = g(n) + h(n)</p>
          </div>
          <div className="flex items-center gap-2 bg-white/20 rounded-lg px-3 py-1.5">
            <Navigation className="w-4 h-4 text-sky-200" />
            <span className="text-white text-sm font-medium">步骤 {step + 1}/{STEPS.length}</span>
          </div>
        </div>
      </div>

      {/* Progress */}
      <div className="h-1.5 bg-slate-200 dark:bg-slate-700">
        <div className="h-full bg-gradient-to-r from-sky-400 to-indigo-400 transition-all duration-300"
          style={{ width: `${(step / (STEPS.length - 1)) * 100}%` }} />
      </div>

      {/* Formula bar */}
      <div className="bg-indigo-50 dark:bg-indigo-950/30 border-b border-indigo-200 dark:border-indigo-800 px-5 py-2">
        <div className="flex items-center gap-4 text-sm text-indigo-800 dark:text-indigo-200 font-mono flex-wrap">
          <span><strong>f(n)</strong> = g(n) + h(n)</span>
          <span className="text-indigo-400">|</span>
          <span><strong>g(n)</strong>: 起点到 n 的实际代价</span>
          <span className="text-indigo-400">|</span>
          <span><strong>h(n)</strong>: 曼哈顿距离（可采纳）</span>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {/* Grid */}
          <div className="md:col-span-3">
            <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">搜索网格</div>
            <div className="inline-grid gap-0.5"
              style={{ gridTemplateColumns: `repeat(${COLS}, minmax(0, 1fr))`, width: '100%' }}>
              {Array.from({ length: ROWS }, (_, r) =>
                Array.from({ length: COLS }, (_, c) => {
                  const k = key(r, c)
                  const f = cur.fScore.get(k)
                  const g = cur.gScore.get(k)
                  const hv = cur.hScore.get(k)
                  const style = getCellStyle(r, c)
                  const showLabel = !isObs(r, c) && (cur.openSet.has(k) || cur.closedSet.has(k) || cur.current === k)

                  return (
                    <div key={k}
                      className={`aspect-square rounded flex flex-col items-center justify-center border ${style} text-center transition-all duration-200 text-xs font-mono`}
                      title={showLabel ? `(${r},${c}) g=${g} h=${hv} f=${f}` : `(${r},${c})`}
                    >
                      {isStart(r, c) && <span className="text-xs font-bold">S</span>}
                      {isEnd(r, c) && <span className="text-xs font-bold">E</span>}
                      {!isStart(r, c) && !isEnd(r, c) && !isObs(r, c) && showLabel && f !== undefined && (
                        <span className="text-[8px] leading-tight">{f}</span>
                      )}
                    </div>
                  )
                })
              )}
            </div>
            {/* Legend */}
            <div className="flex flex-wrap gap-2 mt-2">
              {[
                { cls: 'bg-emerald-500', label: '起点 S' },
                { cls: 'bg-blue-500', label: '终点 E' },
                { cls: 'bg-slate-700', label: '障碍物' },
                { cls: 'bg-orange-400', label: '当前节点' },
                { cls: 'bg-sky-200 dark:bg-sky-900 border border-sky-400', label: '开放集 (f值)' },
                { cls: 'bg-red-100 dark:bg-red-900 border border-red-300', label: '关闭集' },
                { cls: 'bg-yellow-200 dark:bg-yellow-800', label: '最优路径' },
              ].map(l => (
                <div key={l.label} className="flex items-center gap-1 text-xs text-slate-600 dark:text-slate-400">
                  <span className={`w-3 h-3 rounded ${l.cls}`} />
                  {l.label}
                </div>
              ))}
            </div>
          </div>

          {/* Right panel */}
          <div className="md:col-span-2 flex flex-col gap-3">
            {/* Stats */}
            <div className="grid grid-cols-2 gap-2">
              {[
                { label: '开放集', value: cur.openSet.size, color: 'text-sky-600 dark:text-sky-400' },
                { label: '关闭集', value: cur.closedSet.size, color: 'text-red-600 dark:text-red-400' },
              ].map(s => (
                <div key={s.label} className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3 border border-slate-200 dark:border-slate-700 text-center">
                  <div className="text-xs text-slate-500 dark:text-slate-400">{s.label}</div>
                  <div className={`text-2xl font-bold ${s.color}`}>{s.value}</div>
                </div>
              ))}
            </div>

            {/* Current node info */}
            {cur.current && (
              <div className="bg-orange-50 dark:bg-orange-900/20 rounded-xl p-3 border border-orange-200 dark:border-orange-700">
                <div className="text-xs font-semibold text-orange-700 dark:text-orange-300 mb-1">当前节点</div>
                {(() => {
                  const [cr, cc] = cur.current.split(',').map(Number)
                  const g = cur.gScore.get(cur.current) ?? 0
                  const hv = h(cr, cc)
                  const f = g + hv
                  return (
                    <div className="flex gap-3 text-sm font-mono text-orange-800 dark:text-orange-200">
                      <span>({cr},{cc})</span>
                      <span>g=<strong>{g}</strong></span>
                      <span>h=<strong>{hv}</strong></span>
                      <span>f=<strong className="text-orange-600 dark:text-orange-300">{f}</strong></span>
                    </div>
                  )
                })()}
              </div>
            )}

            {/* Open set top-k */}
            <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3 border border-slate-200 dark:border-slate-700 flex-1">
              <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
                开放集（按 f 排序）
              </div>
              {topOpen.length === 0 ? (
                <div className="text-xs text-slate-400 dark:text-slate-500">开放集为空</div>
              ) : (
                <div className="flex flex-col gap-1">
                  {topOpen.map((item, idx) => (
                    <div key={item.k}
                      className={`flex items-center justify-between px-2.5 py-1.5 rounded-md text-xs font-mono border
                        ${idx === 0
                          ? 'bg-orange-100 dark:bg-orange-900/40 border-orange-300 dark:border-orange-700 text-orange-800 dark:text-orange-200 font-bold'
                          : 'bg-white dark:bg-slate-700 border-slate-200 dark:border-slate-600 text-slate-600 dark:text-slate-300'}`}
                    >
                      <span>{idx === 0 ? '→ ' : '   '}({item.k})</span>
                      <span>g={item.g} h={item.hVal} <strong>f={item.f}</strong></span>
                    </div>
                  ))}
                  {cur.openSet.size > 6 && (
                    <div className="text-xs text-slate-400 dark:text-slate-500 pl-2">
                      ...还有 {cur.openSet.size - 6} 个节点
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Path result */}
            {cur.phase === 'done' && (
              <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-xl p-3 border border-emerald-200 dark:border-emerald-700">
                <div className="text-xs font-semibold text-emerald-700 dark:text-emerald-300 mb-1">最优路径</div>
                <div className="text-xs text-emerald-800 dark:text-emerald-200 font-mono leading-5">
                  {cur.path.map(k => `(${k})`).join(' → ')}
                </div>
                <div className="text-xs text-emerald-600 dark:text-emerald-400 mt-1">
                  路径长度：{cur.path.length - 1} 步
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Description */}
        <div className={`mt-3 rounded-xl px-4 py-3 border text-sm leading-5
          ${cur.phase === 'done' ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700 text-emerald-800 dark:text-emerald-200' :
            'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-700 text-blue-800 dark:text-blue-200'}`}>
          {cur.description.split('\n').map((line, i) => <div key={i} className="font-mono">{line}</div>)}
        </div>

        {/* Controls */}
        <div className="mt-4 flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <button onClick={() => { setPlaying(false); setStep(0) }}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600 text-sm transition-colors">
              <RotateCcw className="w-3.5 h-3.5" /> 重置
            </button>
            <button onClick={() => setPlaying(p => !p)} disabled={isLast}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold transition-colors disabled:opacity-40">
              {playing ? <><Pause className="w-3.5 h-3.5" /> 暂停</> : <><Play className="w-3.5 h-3.5" /> 自动播放</>}
            </button>
            <button onClick={() => { setPlaying(false); if (!isLast) setStep(s => s + 1) }} disabled={isLast}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-100 dark:bg-sky-900/40 text-sky-700 dark:text-sky-300 hover:bg-sky-200 dark:hover:bg-sky-900/60 text-sm transition-colors disabled:opacity-40">
              <SkipForward className="w-3.5 h-3.5" /> 下一步
            </button>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
            速度
            {[1200, 800, 400].map(s => (
              <button key={s} onClick={() => setSpeed(s)}
                className={`px-2 py-1 rounded text-xs ${speed === s ? 'bg-blue-600 text-white' : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300'}`}>
                {s === 1200 ? '慢' : s === 800 ? '中' : '快'}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
