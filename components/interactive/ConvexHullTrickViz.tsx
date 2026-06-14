'use client'
import { useState, useEffect } from 'react'

/* ─── 凸包技巧 (CHT) 可视化 ─────────────────────────────── */
// 最小化 f(x) = min_j { m[j]*x + b[j] }
// 下凸包：维护直线集，淘汰被"夹在上方"的冗余直线

interface Line { m: number; b: number; color: string; label: string }

const PRESETS: { label: string; lines: Line[]; qRange: [number, number] }[] = [
  {
    label: '经典 5 线',
    lines: [
      { m: 4,  b: 0,  color: '#7c3aed', label: 'L₁: 4x' },
      { m: 2,  b: 4,  color: '#2563eb', label: 'L₂: 2x+4' },
      { m: 0,  b: 7,  color: '#059669', label: 'L₃: 7' },
      { m: -2, b: 12, color: '#d97706', label: 'L₄: -2x+12' },
      { m: -4, b: 18, color: '#dc2626', label: 'L₅: -4x+18' },
    ],
    qRange: [0, 5],
  },
  {
    label: '含淘汰线',
    lines: [
      { m: 3,  b: 0,  color: '#7c3aed', label: 'L₁: 3x' },
      { m: 2,  b: 2,  color: '#2563eb', label: 'L₂: 2x+2' },
      { m: 2,  b: 3,  color: '#dc2626', label: 'L₃: 2x+3 (淘汰)' },
      { m: -1, b: 12, color: '#059669', label: 'L₄: -x+12' },
      { m: -3, b: 20, color: '#d97706', label: 'L₅: -3x+20' },
    ],
    qRange: [0, 6],
  },
]

// 倘若 l2 在 l1 与 l3 覆盖范围内永远不是最小值，则 l2 是坏线（可淘汰）
function bad(l1: Line, l2: Line, l3: Line): boolean {
  // 交点 l1∩l3 的 x 坐标 <= 交点 l1∩l2 的 x 坐标  =>  l2 useless
  // 用叉乘形式避免除法
  return (l3.b - l1.b) * (l1.m - l2.m) <= (l2.b - l1.b) * (l1.m - l3.m)
}

function buildHull(lines: Line[]): Line[] {
  const hull: Line[] = []
  for (const l of lines) {
    while (hull.length >= 2 && bad(hull[hull.length - 2], hull[hull.length - 1], l))
      hull.pop()
    hull.push(l)
  }
  return hull
}

// 在 hull 中查找 x 处的最小直线
function queryHull(hull: Line[], x: number): Line | null {
  if (!hull.length) return null
  let best = hull[0]
  for (const l of hull) if (l.m * x + l.b < best.m * x + best.b) best = l
  return best
}

// 坐标系映射
const VIEW_X0 = -0.5, VIEW_X1 = 5.8, VIEW_Y0 = -2, VIEW_Y1 = 20
const SVG_W = 320, SVG_H = 260
function toSvg(mx: number, my: number): [number, number] {
  const sx = ((mx - VIEW_X0) / (VIEW_X1 - VIEW_X0)) * SVG_W
  const sy = SVG_H - ((my - VIEW_Y0) / (VIEW_Y1 - VIEW_Y0)) * SVG_H
  return [sx, sy]
}

const SPEED_OPTS: [string, number][] = [['慢', 900], ['中', 500], ['快', 200]]

export default function ConvexHullTrickViz() {
  const [preIdx, setPreIdx] = useState(0)
  const [step, setStep] = useState(0)      // 0 = none; 1..N = lines added; N+1..N+M = query sweep
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(500)

  const { lines, qRange } = PRESETS[preIdx]
  const N = lines.length
  const querySteps = 12
  const totalSteps = N + querySteps // add N lines, then sweep query

  useEffect(() => { setStep(0); setPlaying(false) }, [preIdx])
  useEffect(() => {
    if (!playing) return
    if (step >= totalSteps) { setPlaying(false); return }
    const id = setTimeout(() => setStep(s => s + 1), speed)
    return () => clearTimeout(id)
  }, [playing, step, totalSteps, speed])

  const addedLines = lines.slice(0, step > N ? N : step)
  const hull = buildHull(addedLines)
  const isQueryPhase = step > N
  const queryX = isQueryPhase
    ? qRange[0] + ((step - N - 1) / (querySteps - 1)) * (qRange[1] - qRange[0])
    : null
  const bestLine = queryX !== null ? queryHull(hull, queryX) : null

  const xTicks = [0, 1, 2, 3, 4, 5]
  const yTicks = [0, 5, 10, 15]

  return (
    <div className="rounded-2xl border border-cyan-200 dark:border-cyan-900 bg-white dark:bg-zinc-950 overflow-hidden shadow-sm">
      <div className="px-6 py-4 bg-gradient-to-r from-cyan-600 to-sky-600 dark:from-cyan-700 dark:to-sky-700">
        <h3 className="text-white font-bold text-base">斜率优化（凸包技巧 CHT）可视化</h3>
        <p className="text-cyan-100 text-sm mt-0.5">
          最小化 <span className="font-mono">f(x) = min_j &#123; m[j]·x + b[j] &#125;</span>，维护下凸包淘汰冗余直线
        </p>
      </div>

      <div className="p-5 space-y-4">
        {/* 控制栏 */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex gap-1">
            {PRESETS.map((p, i) => (
              <button key={i} onClick={() => setPreIdx(i)}
                className={`px-3 py-1 text-xs rounded-lg font-semibold transition-all ${preIdx === i ? 'bg-cyan-600 text-white shadow' : 'bg-slate-100 dark:bg-zinc-800 text-slate-600 dark:text-zinc-300'}`}>
                {p.label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1.5 ml-auto">
            <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
              className="px-2.5 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200">←</button>
            <button onClick={() => setPlaying(p => !p)}
              className={`px-3 py-1.5 text-xs rounded-lg text-white font-medium ${playing ? 'bg-orange-500' : 'bg-cyan-600 hover:bg-cyan-500'}`}>
              {playing ? '⏸ 暂停' : '▶ 播放'}
            </button>
            <button onClick={() => setStep(s => Math.min(totalSteps, s + 1))} disabled={step >= totalSteps}
              className="px-2.5 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200">→</button>
            <button onClick={() => { setStep(0); setPlaying(false) }}
              className="px-2.5 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 rounded-lg text-slate-700 dark:text-zinc-200">↺</button>
            {SPEED_OPTS.map(([l, ms]) => (
              <button key={ms} onClick={() => setSpeed(ms)}
                className={`px-2 py-1 text-xs rounded ${speed === ms ? 'bg-cyan-600 text-white' : 'bg-slate-100 dark:bg-zinc-800 text-slate-500 dark:text-zinc-400'}`}>{l}</button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          {/* SVG 坐标系 */}
          <div className="lg:col-span-3 bg-slate-50 dark:bg-zinc-900 rounded-xl border border-slate-200 dark:border-zinc-700 p-3">
            <svg width="100%" viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="block">
              {/* 网格 */}
              {xTicks.map(x => {
                const [sx,] = toSvg(x, 0)
                return <line key={x} x1={sx} y1={0} x2={sx} y2={SVG_H} stroke="#e2e8f0" className="dark:stroke-zinc-800" strokeWidth={1} />
              })}
              {yTicks.map(y => {
                const [, sy] = toSvg(0, y)
                return <line key={y} x1={0} y1={sy} x2={SVG_W} y2={sy} stroke="#e2e8f0" className="dark:stroke-zinc-800" strokeWidth={1} />
              })}
              {/* 轴 */}
              {(() => { const [ax, ay] = toSvg(0, 0); return <><line x1={0} y1={ay} x2={SVG_W} y2={ay} stroke="#94a3b8" strokeWidth={1.5}/><line x1={ax} y1={0} x2={ax} y2={SVG_H} stroke="#94a3b8" strokeWidth={1.5}/></> })()}
              {/* 刻度标签 */}
              {xTicks.map(x => { const [sx, ay] = [toSvg(x, 0)[0], toSvg(0, 0)[1]]; return <text key={x} x={sx} y={ay+12} textAnchor="middle" fontSize={9} fill="#94a3b8">{x}</text> })}
              {yTicks.filter(y => y > 0).map(y => { const [ax, sy] = [toSvg(0, 0)[0], toSvg(0, y)[1]]; return <text key={y} x={ax-6} y={sy+3} textAnchor="end" fontSize={9} fill="#94a3b8">{y}</text> })}

              {/* 所有已添加的直线 */}
              {addedLines.map((l, idx) => {
                const onHull = hull.some(h => h === l)
                const isBest = bestLine === l
                const [x0, y0] = toSvg(VIEW_X0, l.m * VIEW_X0 + l.b)
                const [x1, y1] = toSvg(VIEW_X1, l.m * VIEW_X1 + l.b)
                return (
                  <g key={idx}>
                    <line x1={x0} y1={y0} x2={x1} y2={y1}
                      stroke={l.color}
                      strokeWidth={isBest ? 3.5 : onHull ? 2 : 1}
                      strokeOpacity={onHull ? 1 : 0.25}
                      strokeDasharray={onHull ? '' : '6,4'} />
                    {/* 标签 */}
                    {onHull && (() => {
                      const lx = 3.5
                      const [lsx, lsy] = toSvg(lx, l.m * lx + l.b)
                      return <text x={lsx+3} y={lsy-4} fontSize={9} fill={l.color} fontWeight="bold">{l.label.split(':')[0]}</text>
                    })()}
                  </g>
                )
              })}

              {/* 最新加入的直线——高亮 */}
              {step > 0 && step <= N && (() => {
                const l = lines[step - 1]
                const [sx, sy] = toSvg(2.5, l.m * 2.5 + l.b)
                return <circle cx={sx} cy={sy} r={5} fill={l.color} opacity={0.6} />
              })()}

              {/* 下凸包线段（hull 相邻直线的交点连线） */}
              {hull.length >= 2 && (() => {
                const pts: [number, number][] = []
                pts.push(toSvg(VIEW_X0, hull[0].m * VIEW_X0 + hull[0].b))
                for (let k = 0; k < hull.length - 1; k++) {
                  const a = hull[k], b = hull[k + 1]
                  const xi = (b.b - a.b) / (a.m - b.m)
                  const yi = a.m * xi + a.b
                  pts.push(toSvg(xi, yi))
                }
                pts.push(toSvg(VIEW_X1, hull[hull.length - 1].m * VIEW_X1 + hull[hull.length - 1].b))
                return (
                  <polyline points={pts.map(p => p.join(',')).join(' ')}
                    fill="none" stroke="#06b6d4" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" opacity={0.8} />
                )
              })()}

              {/* 查询竖线 */}
              {queryX !== null && (() => {
                const [qx,] = toSvg(queryX, 0)
                const [, qy] = bestLine ? toSvg(queryX, bestLine.m * queryX + bestLine.b) : toSvg(queryX, 0)
                const [, zeroY] = toSvg(0, 0)
                return (
                  <g>
                    <line x1={qx} y1={0} x2={qx} y2={SVG_H} stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="4,3" />
                    <circle cx={qx} cy={qy} r={5} fill="#f59e0b" />
                    <line x1={qx} y1={zeroY} x2={qx} y2={qy} stroke="#f59e0b" strokeWidth={1} strokeDasharray="3,2" />
                    <text x={qx+4} y={qy - 6} fontSize={10} fill="#f59e0b" fontWeight="bold">
                      min={Math.round((bestLine ? bestLine.m * queryX + bestLine.b : 0) * 10) / 10}
                    </text>
                  </g>
                )
              })()}
            </svg>
          </div>

          {/* 右侧面板 */}
          <div className="lg:col-span-2 space-y-3">
            {/* 直线列表 */}
            <div className="rounded-xl border border-slate-200 dark:border-zinc-700 bg-slate-50 dark:bg-zinc-900 p-3 space-y-1.5">
              <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400 dark:text-zinc-500 mb-1">直线状态</p>
              {lines.map((l, i) => {
                const added = i < (step > N ? N : step)
                const onHull = added && hull.some(h => h === l)
                const isBest = bestLine === l
                return (
                  <div key={i} className={`flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs transition-all
                    ${!added ? 'opacity-30' : isBest ? 'bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-700' :
                      onHull ? 'bg-cyan-50 dark:bg-cyan-950/40' : 'opacity-40'}`}>
                    <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: l.color }} />
                    <span className="font-mono text-slate-700 dark:text-zinc-200 flex-1">{l.label}</span>
                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${isBest ? 'bg-amber-400 text-white' : onHull ? 'bg-cyan-500 text-white' : added ? 'bg-slate-300 dark:bg-zinc-600 text-slate-500 dark:text-zinc-400' : ''}`}>
                      {!added ? '-' : isBest ? '✓最优' : onHull ? '凸包' : '淘汰'}
                    </span>
                  </div>
                )
              })}
            </div>

            {/* 当前阶段说明 */}
            <div className="rounded-xl border border-slate-200 dark:border-zinc-700 bg-white dark:bg-zinc-950 p-3 text-xs space-y-1.5">
              <p className="font-bold text-slate-600 dark:text-zinc-200">
                {isQueryPhase ? '第二阶段：扫描查询' : step === 0 ? '初始状态' : `第一阶段：添加直线 ${step}/${N}`}
              </p>
              {!isQueryPhase && step > 0 && (
                <div className="space-y-1 font-mono text-slate-500 dark:text-zinc-400">
                  <p>新增: <span style={{ color: lines[Math.min(step,N)-1].color }} className="font-bold">{lines[Math.min(step,N)-1].label}</span></p>
                  <p>凸包长度: {hull.length} 条</p>
                  {hull.length < addedLines.length && (
                    <p className="text-red-500 dark:text-red-400">
                      淘汰 {addedLines.length - hull.length} 条冗余线
                    </p>
                  )}
                </div>
              )}
              {isQueryPhase && queryX !== null && bestLine && (
                <div className="space-y-1 font-mono text-slate-500 dark:text-zinc-400">
                  <p>查询点 x = <span className="text-amber-600 dark:text-amber-400 font-bold">{queryX.toFixed(2)}</span></p>
                  <p>最小值来自: <span style={{ color: bestLine.color }} className="font-bold">{bestLine.label}</span></p>
                  <p>f(x) = {(bestLine.m * queryX + bestLine.b).toFixed(2)}</p>
                </div>
              )}
              {step === 0 && <p className="text-slate-400 dark:text-zinc-500">点击播放开始动画</p>}
            </div>

            {/* bad() 函数说明 */}
            <div className="rounded-xl border border-slate-200 dark:border-zinc-700 bg-slate-50 dark:bg-zinc-900 p-3 text-[11px]">
              <p className="font-bold text-slate-500 dark:text-zinc-400 mb-1">bad(L1, L2, L3) 判断</p>
              <p className="font-mono text-slate-600 dark:text-zinc-300 text-[10px]">
                若 L1∩L3 在 L1∩L2 左侧，则 L2 永远不是最小 → 淘汰
              </p>
            </div>
          </div>
        </div>

        {/* 进度条 */}
        <div className="w-full h-1.5 bg-slate-100 dark:bg-zinc-800 rounded-full overflow-hidden">
          <div className="h-full bg-cyan-500 rounded-full transition-all duration-150"
            style={{ width: `${((step) / totalSteps) * 100}%` }} />
        </div>
        <p className="text-xs text-slate-400 dark:text-zinc-500 text-right">
          {step <= N ? `添加直线 ${step}/${N}` : `查询扫描 ${step - N}/${querySteps}`}
        </p>
      </div>
    </div>
  )
}
