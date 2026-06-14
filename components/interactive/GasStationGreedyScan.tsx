'use client'

import React, { useState, useCallback, useEffect, useRef } from 'react'
import { Play, Pause, SkipBack, ChevronLeft, ChevronRight } from 'lucide-react'

const GAS  = [1, 2, 3, 4, 5]
const COST = [3, 4, 5, 1, 2]
const N = GAS.length

interface GSStep {
  stationIdx: number   // current station being processed (-1 = init)
  delta: number
  tank: number
  start: number
  reset: boolean
  desc: string
  detail: string
}

function buildGSSteps(): GSStep[] {
  const steps: GSStep[] = []
  const totalGas  = GAS.reduce((s, v) => s + v, 0)
  const totalCost = COST.reduce((s, v) => s + v, 0)
  const totalDelta = totalGas - totalCost

  steps.push({
    stationIdx: -1,
    delta: 0,
    tank: 0,
    start: 0,
    reset: false,
    desc: '初始化：计算每站 Δ = gas - cost',
    detail: `共 ${N} 个加油站，排成一个环形。总加油量 = ${totalGas}，总消耗 = ${totalCost}，总差值 Δ = ${totalDelta}。由于 Δ ${ totalDelta >= 0 ? '≥' : '<'} 0，可行起点${totalDelta >= 0 ? '存在（贪心可找到）' : '不存在'}。\n策略：从第 0 站开始，维护 tank（油箱剩余油量）；若 tank 负数则重置 start = i+1，tank = 0。最后 start 即答案。`,
  })

  let start = 0
  let tank = 0

  for (let i = 0; i < N; i++) {
    const delta = GAS[i] - COST[i]
    tank += delta
    const reset = tank < 0
    if (reset) {
      const prevTank = tank - delta  // Actually we need pre-reset values
      steps.push({
        stationIdx: i,
        delta,
        tank: 0,          // after reset
        start: i + 1,
        reset: true,
        desc: `❌ 站点 ${i}：tank 归零，起点重置 → start=${Math.min(i + 1, N - 1)}`,
        detail: `油量补充：gas[${i}]=${GAS[i]}，消耗：cost[${i}]=${COST[i]}，Δ=${delta}。tank 更新为 ${tank - 0 + delta}+${delta}=${tank}（此处已是 ${tank} 前的值 ${tank-delta} 加 ${delta} = ${tank}）→ tank=${tank} < 0，无法到达下一站！\n\n结论：从任何 0..${i} 的站出发都必然在此处断油（若从前 start-${i} 某站出发，tank 只会更少）。因此起点不可能在 start..${i} 之间，直接将 start=${i + 1}，tank 重置为 0。`,
      })
      tank = 0
      start = i + 1
    } else {
      steps.push({
        stationIdx: i,
        delta,
        tank,
        start,
        reset: false,
        desc: `✅ 站点 ${i}：Δ=${delta}，tank=${tank}，继续`,
        detail: `油量补充：gas[${i}]=${GAS[i]}，消耗：cost[${i}]=${COST[i]}，Δ=${delta} ${ delta >= 0 ? '≥ 0（补充正向）' : '< 0（净消耗）'}。tank 更新为 ${tank - delta}${delta >= 0 ? '+' : ''}${delta} = ${tank} ≥ 0，可以继续行驶。起点维持 start=${start}。`,
      })
    }
  }

  const finalDesc = totalDelta >= 0
    ? `起点 = ${start}（从该站出发可完成一圈）`
    : '无可行起点（总油量不足总消耗）'

  steps.push({
    stationIdx: N - 1,
    delta: 0,
    tank,
    start,
    reset: false,
    desc: `🎉 算法完成！答案：起点 = ${start}`,
    detail: `扫描结束，最终 start=${start}。验证正确性：① 总 Δ = ${totalDelta} ≥ 0 保证解存在；② 贪心给出的 start=${start} 是唯一可行起点（任何被跳过的区间 [0..start-1] 都必然在扫描中断油）。时间复杂度 O(n)，空间 O(1)。${finalDesc}`,
  })

  return steps
}

const STEPS = buildGSSteps()
const TANK_MAX = 10  // max tank display capacity
const SPEEDS = [0.5, 1, 1.5, 2]

export default function GasStationGreedyScan() {
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speedIdx, setSpeedIdx] = useState(1)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const cur = STEPS[idx]
  const total = STEPS.length - 1

  const stop = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current)
    timerRef.current = null
    setPlaying(false)
  }, [])

  useEffect(() => {
    if (!playing) return
    timerRef.current = setInterval(() => {
      setIdx(prev => {
        if (prev >= total) { stop(); return prev }
        return prev + 1
      })
    }, 1300 / SPEEDS[speedIdx])
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [playing, speedIdx, total, stop])

  useEffect(() => { if (idx >= total && playing) stop() }, [idx, total, playing, stop])
  const goto = (n: number) => { stop(); setIdx(Math.max(0, Math.min(total, n))) }

  const tankPct = Math.min(100, (cur.tank / TANK_MAX) * 100)
  const totalDelta = GAS.reduce((s,v) => s+v, 0) - COST.reduce((s,v) => s+v, 0)

  return (
    <div className="w-full max-w-5xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      <div className="bg-gradient-to-r from-orange-600 via-amber-600 to-yellow-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg">加油站贪心扫描 — 一次线性扫描找起点</h3>
        <p className="text-amber-100 text-sm mt-0.5">维护 tank 油量，负数即重置 start，证明：总 Δ≥0 则答案存在</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={() => goto(0)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300"><SkipBack size={16}/></button>
          <button onClick={() => goto(idx - 1)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300"><ChevronLeft size={16}/></button>
          <button onClick={playing ? stop : () => setPlaying(true)} className="px-3 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-700 text-white flex items-center gap-1.5 text-sm font-medium">
            {playing ? <Pause size={14}/> : <Play size={14}/>}{playing ? '暂停' : '播放'}
          </button>
          <button onClick={() => goto(idx + 1)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300"><ChevronRight size={16}/></button>
          <div className="ml-auto flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
            速度: {SPEEDS.map((s, i) => (
              <button key={i} onClick={() => setSpeedIdx(i)} className={`px-2 py-0.5 rounded ${speedIdx === i ? 'bg-amber-600 text-white' : 'border border-slate-200 dark:border-slate-700'}`}>{s}x</button>
            ))}
          </div>
        </div>

        <div className="w-full h-1.5 rounded-full bg-slate-100 dark:bg-slate-800">
          <div className="h-1.5 rounded-full bg-amber-500 transition-all" style={{ width: `${(idx / total) * 100}%` }}/>
        </div>

        {/* Station visualization */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-3">
          <div className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">站点状态（圆环形布局模拟）</div>
          <div className="flex gap-2 justify-center flex-wrap">
            {Array.from({ length: N }).map((_, i) => {
              const delta = GAS[i] - COST[i]
              const isCurrent = i === cur.stationIdx
              const isStart = i === cur.start && cur.stationIdx >= 0
              const isPassed = cur.stationIdx >= 0 && i < cur.stationIdx + 1
              const wasReset = STEPS.slice(0, idx + 1).some(s => s.reset && s.stationIdx === i)

              return (
                <div key={i} className={`relative flex flex-col items-center p-3 rounded-2xl border-2 transition-all duration-500 min-w-[72px] ${
                  isCurrent
                    ? cur.reset
                      ? 'border-rose-500 bg-rose-50 dark:bg-rose-950/50 scale-110 shadow-xl'
                      : 'border-amber-500 bg-amber-50 dark:bg-amber-950/50 scale-110 shadow-xl'
                    : isStart && isPassed
                      ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/30'
                      : wasReset && !isStart
                        ? 'border-rose-300 dark:border-rose-800 bg-rose-50/50 dark:bg-rose-950/20 opacity-60'
                        : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900'
                }`}>
                  {isStart && isPassed && (
                    <div className="absolute -top-2 left-1/2 -translate-x-1/2 bg-emerald-500 text-white text-[9px] font-bold px-1.5 py-0.5 rounded-full">起点</div>
                  )}
                  <div className="text-lg font-black text-slate-700 dark:text-slate-200">S{i}</div>
                  <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">gas={GAS[i]}</div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">cost={COST[i]}</div>
                  <div className={`text-sm font-black mt-1 ${delta >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                    Δ={delta >= 0 ? '+' : ''}{delta}
                  </div>
                  {wasReset && i !== cur.start && (
                    <div className="text-[9px] text-rose-500 dark:text-rose-400 mt-0.5 font-bold">断油↗</div>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {/* Tank gauge */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-3">
          <div className="flex items-center justify-between mb-2">
            <div className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">油箱量表</div>
            <div className="text-sm font-black text-amber-700 dark:text-amber-300">tank = {cur.tank}</div>
          </div>
          <div className="h-8 w-full rounded-xl overflow-hidden bg-slate-200 dark:bg-slate-700">
            <div
              className={`h-full rounded-xl transition-all duration-700 flex items-center justify-end pr-3 ${cur.tank === 0 ? 'bg-slate-300 dark:bg-slate-600' : cur.reset ? 'bg-rose-500' : 'bg-amber-500'}`}
              style={{ width: `${Math.max(tankPct, 1)}%` }}
            >
              {tankPct > 15 && <span className="text-white text-xs font-bold">{cur.tank}</span>}
            </div>
          </div>
          <div className="flex justify-between text-[10px] text-slate-400 dark:text-slate-500 mt-0.5">
            <span>0</span><span>5</span><span>10</span>
          </div>
        </div>

        {/* Step description */}
        <div className={`rounded-xl border px-4 py-3 ${
          cur.reset ? 'border-rose-300 dark:border-rose-700 bg-rose-50 dark:bg-rose-950/30'
          : idx === total ? 'border-indigo-300 dark:border-indigo-700 bg-indigo-50 dark:bg-indigo-950/30'
          : 'border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/30'
        }`}>
          <div className="font-bold text-sm text-slate-800 dark:text-slate-100">{cur.desc}</div>
          <div className="mt-1 text-sm text-slate-600 dark:text-slate-300 leading-relaxed whitespace-pre-line">{cur.detail}</div>
        </div>

        {/* Key variables */}
        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-3 text-center">
            <div className="text-xl font-black text-amber-600 dark:text-amber-400">{cur.tank}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400">油箱（tank）</div>
          </div>
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-3 text-center">
            <div className="text-xl font-black text-emerald-600 dark:text-emerald-400">{cur.start}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400">当前候选起点（start）</div>
          </div>
          <div className={`rounded-xl border p-3 text-center ${totalDelta >= 0 ? 'border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/20' : 'border-rose-200 dark:border-rose-800 bg-rose-50 dark:bg-rose-950/20'}`}>
            <div className={`text-xl font-black ${totalDelta >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>{totalDelta >= 0 ? `+${totalDelta}` : totalDelta}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400">总 Δ = 解{totalDelta >= 0 ? '存在' : '不存在'}</div>
          </div>
        </div>
      </div>
    </div>
  )
}
