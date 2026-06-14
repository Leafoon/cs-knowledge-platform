'use client'

import React, { useState, useCallback, useEffect, useRef } from 'react'
import { Play, Pause, SkipBack, ChevronLeft, ChevronRight } from 'lucide-react'

interface Job { id: string; s: number; e: number }

// 6 jobs that nicely demonstrate machine reuse
const JOBS: Job[] = [
  { id: 'J1', s: 1, e: 4  },
  { id: 'J2', s: 2, e: 5  },
  { id: 'J3', s: 7, e: 9  },
  { id: 'J4', s: 3, e: 6  },
  { id: 'J5', s: 5, e: 8  },
  { id: 'J6', s: 8, e: 11 },
]
// Sorted by start time
const SORTED = [...JOBS].sort((a, b) => a.s - b.s || a.e - b.e)

const MACHINE_COLORS = [
  { bg: 'bg-cyan-500', bar: '#06b6d4', light: 'bg-cyan-100 dark:bg-cyan-950/50', border: 'border-cyan-400 dark:border-cyan-600', text: 'text-cyan-700 dark:text-cyan-300' },
  { bg: 'bg-violet-500', bar: '#8b5cf6', light: 'bg-violet-100 dark:bg-violet-950/50', border: 'border-violet-400 dark:border-violet-600', text: 'text-violet-700 dark:text-violet-300' },
  { bg: 'bg-amber-500',  bar: '#f59e0b', light: 'bg-amber-100 dark:bg-amber-950/50',  border: 'border-amber-400 dark:border-amber-600',  text: 'text-amber-700 dark:text-amber-300' },
]

interface ISStep {
  jobIdx: number          // index into SORTED (-1 = init)
  desc: string
  detail: string
  machineEnd: number[]    // end time for each machine
  machineJobs: string[][] // jobs assigned to each machine (cumulative)
  newMachine: boolean     // whether this step added a new machine
  reuseFrom?: number     // which machine was reused
  currentJobId?: string
}

function buildISSteps(sorted: Job[]): ISStep[] {
  const steps: ISStep[] = []
  const machineEnd: number[] = []
  const machineJobs: string[][] = []

  steps.push({
    jobIdx: -1,
    desc: '初始状态：按开始时间排序',
    detail: `将 ${sorted.length} 个任务按开始时间升序排列：${sorted.map(j => `${j.id}[${j.s},${j.e})`).join('，')}。维护一个最小堆记录每台机器的最早空闲时间（即当前最小结束时间）。`,
    machineEnd: [],
    machineJobs: [],
    newMachine: false,
    currentJobId: undefined,
  })

  for (let i = 0; i < sorted.length; i++) {
    const job = sorted[i]
    // Find machine with smallest end time
    let reuse = -1
    let minEnd = Infinity
    for (let m = 0; m < machineEnd.length; m++) {
      if (machineEnd[m] <= job.s && machineEnd[m] < minEnd) {
        minEnd = machineEnd[m]
        reuse = m
      }
    }

    if (reuse >= 0) {
      // Reuse machine
      machineJobs[reuse].push(job.id)
      machineEnd[reuse] = job.e
      steps.push({
        jobIdx: i,
        desc: `♻️ 复用机器 ${reuse + 1}：分配 ${job.id}[${job.s},${job.e})`,
        detail: `堆中最小结束时间为机器 ${reuse + 1}（end=${minEnd}），${job.id}.start=${job.s} ≥ ${minEnd}，机器空闲！将 ${job.id} 分配给机器 ${reuse + 1}，机器 ${reuse + 1} 的新结束时间更新为 ${job.e}。当前机器数 = ${machineEnd.length}（无需新增）。`,
        machineEnd: [...machineEnd],
        machineJobs: machineJobs.map(arr => [...arr]),
        newMachine: false,
        reuseFrom: reuse,
        currentJobId: job.id,
      })
    } else {
      // New machine
      machineEnd.push(job.e)
      machineJobs.push([job.id])
      const mIdx = machineEnd.length - 1
      const heapStr = machineEnd.slice(0, -1).map((e, m) => `M${m+1}(end=${e})`).join(', ')
      const reason = machineEnd.length > 1
        ? `堆中所有机器的结束时间（${heapStr}）均 > ${job.id}.start=${job.s}，没有空闲机器。`
        : `堆为空，没有任何机器。`
      steps.push({
        jobIdx: i,
        desc: `➕ 新增机器 ${mIdx + 1}：分配 ${job.id}[${job.s},${job.e})`,
        detail: `${reason}必须新建机器 ${mIdx + 1} 承接 ${job.id}。这意味着在 t=${job.s} 时刻，共有 ${machineEnd.length} 个任务同时进行，达到了新的最大重叠数。当前机器数 = ${machineEnd.length}。`,
        machineEnd: [...machineEnd],
        machineJobs: machineJobs.map(arr => [...arr]),
        newMachine: true,
        currentJobId: job.id,
      })
    }
  }

  steps.push({
    jobIdx: sorted.length - 1,
    desc: '🎉 算法完成 — 最少机器数 = 最大重叠数',
    detail: `共使用 ${machineEnd.length} 台机器。定理：区间着色所需最少机器数 = 所有时刻中重叠任务的最大数量（即"深度"）。本例最大深度 = 3（在 t∈[3,4) 时 J1, J2, J4 同时运行）。贪心算法以 O(n log n) 时间给出最优着色（下界证明：每次新增机器必对应新的同时重叠任务）。`,
    machineEnd: [...machineEnd],
    machineJobs: machineJobs.map(arr => [...arr]),
    newMachine: false,
    currentJobId: undefined,
  })

  return steps
}

const STEPS = buildISSteps(SORTED)
const TMAX = 12
const SPEEDS = [0.5, 1, 1.5, 2]

export default function IntervalColoringScheduler() {
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

  const numMachines = cur.machineEnd.length

  return (
    <div className="w-full max-w-5xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      <div className="bg-gradient-to-r from-teal-600 via-cyan-600 to-blue-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg">区间着色调度器 — 最少机器数 = 最大重叠数</h3>
        <p className="text-cyan-100 text-sm mt-0.5">逐步分配任务到机器，观察机器复用与新增的决策过程</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={() => goto(0)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300"><SkipBack size={16}/></button>
          <button onClick={() => goto(idx - 1)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300"><ChevronLeft size={16}/></button>
          <button onClick={playing ? stop : () => setPlaying(true)} className="px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-700 text-white flex items-center gap-1.5 text-sm font-medium">
            {playing ? <Pause size={14}/> : <Play size={14}/>}{playing ? '暂停' : '播放'}
          </button>
          <button onClick={() => goto(idx + 1)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300"><ChevronRight size={16}/></button>
          <div className="ml-auto flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
            速度: {SPEEDS.map((s, i) => (
              <button key={i} onClick={() => setSpeedIdx(i)} className={`px-2 py-0.5 rounded ${speedIdx === i ? 'bg-cyan-600 text-white' : 'border border-slate-200 dark:border-slate-700'}`}>{s}x</button>
            ))}
          </div>
        </div>

        <div className="w-full h-1.5 rounded-full bg-slate-100 dark:bg-slate-800">
          <div className="h-1.5 rounded-full bg-cyan-500 transition-all" style={{ width: `${(idx / total) * 100}%` }}/>
        </div>

        {/* Machine timeline */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-3 space-y-2">
          {/* Time ruler */}
          <div className="flex items-end h-5 pl-20">
            {Array.from({ length: TMAX + 1 }).map((_, t) => (
              <div key={t} className="flex-1 text-center text-[9px] text-slate-400 dark:text-slate-500 font-mono">{t}</div>
            ))}
          </div>

          {/* Machine lanes */}
          {numMachines === 0 ? (
            <div className="text-sm text-slate-400 dark:text-slate-500 text-center py-4">尚无任务分配</div>
          ) : (
            Array.from({ length: numMachines }).map((_, m) => {
              const mc = MACHINE_COLORS[m % MACHINE_COLORS.length]
              const jobs = cur.machineJobs[m] || []
              const isActive = cur.reuseFrom === m || (cur.newMachine && m === numMachines - 1)
              return (
                <div key={m} className="flex items-center gap-2">
                  <div className={`w-18 shrink-0 text-xs font-bold px-2 py-1 rounded-lg ${mc.light} ${mc.text} ${mc.border} border text-center ${isActive ? 'ring-2 ring-offset-1 ring-current' : ''}`}>
                    机器 {m + 1}
                    <div className="text-[9px] font-normal opacity-80">end={cur.machineEnd[m]}</div>
                  </div>
                  <div className="flex-1 h-9 rounded bg-slate-200/50 dark:bg-slate-700/30 relative">
                    {Array.from({ length: TMAX + 1 }).map((_, t) => (
                      <div key={t} className="absolute top-0 h-full border-l border-slate-300/40 dark:border-slate-600/20" style={{ left: `${(t / TMAX) * 100}%` }}/>
                    ))}
                    {jobs.map(jid => {
                      const job = JOBS.find(j => j.id === jid)!
                      const isCurrent = jid === cur.currentJobId
                      return (
                        <div
                          key={jid}
                          className={`absolute top-1 bottom-1 rounded flex items-center justify-center text-white text-xs font-bold ${mc.bg} transition-all duration-500 ${isCurrent ? 'ring-2 ring-white ring-offset-1 shadow-lg' : ''}`}
                          style={{ left: `${(job.s / TMAX) * 100}%`, width: `${((job.e - job.s) / TMAX) * 100}%` }}
                        >
                          {isCurrent ? `★${jid}` : jid}
                        </div>
                      )
                    })}
                  </div>
                </div>
              )
            })
          )}
        </div>

        {/* Heap state */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-3">
          <div className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">最小堆状态（按结束时间从小到大）</div>
          {numMachines === 0 ? (
            <div className="text-xs text-slate-400 dark:text-slate-500 italic">堆为空</div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {[...cur.machineEnd.map((e, m) => ({ m, e }))].sort((a, b) => a.e - b.e).map(({ m, e }) => {
                const mc = MACHINE_COLORS[m % MACHINE_COLORS.length]
                const isReuse = cur.reuseFrom === m
                return (
                  <div key={m} className={`flex flex-col items-center px-3 py-2 rounded-xl border-2 ${mc.light} ${mc.border} ${isReuse ? 'ring-2 ring-teal-400 scale-110 shadow-lg' : ''} transition-all duration-300`}>
                    <div className={`text-sm font-black ${mc.text}`}>M{m + 1}</div>
                    <div className="text-xs text-slate-600 dark:text-slate-300">end={e}</div>
                    {isReuse && <div className="text-[10px] text-teal-600 dark:text-teal-400 mt-0.5">♻️ 复用</div>}
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Step info */}
        <div className={`rounded-xl border px-4 py-3 ${
          cur.newMachine ? 'border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/30'
          : cur.reuseFrom !== undefined ? 'border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-950/30'
          : idx === total ? 'border-indigo-300 dark:border-indigo-700 bg-indigo-50 dark:bg-indigo-950/30'
          : 'border-cyan-300 dark:border-cyan-700 bg-cyan-50 dark:bg-cyan-950/30'
        }`}>
          <div className="font-bold text-sm text-slate-800 dark:text-slate-100">{cur.desc}</div>
          <div className="mt-1 text-sm text-slate-600 dark:text-slate-300 leading-relaxed">{cur.detail}</div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-3 text-center">
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 px-3 py-3">
            <div className="text-2xl font-black text-slate-800 dark:text-slate-100">{numMachines}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400">当前机器数</div>
          </div>
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 px-3 py-3">
            <div className="text-2xl font-black text-slate-800 dark:text-slate-100">
              {cur.jobIdx + 1}
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-400">已处理任务 / {SORTED.length}</div>
          </div>
          <div className="rounded-xl border border-indigo-200 dark:border-indigo-800 bg-indigo-50 dark:bg-indigo-950/20 px-3 py-3">
            <div className="text-2xl font-black text-indigo-700 dark:text-indigo-300">3</div>
            <div className="text-xs text-indigo-500 dark:text-indigo-400">最大深度（下界）</div>
          </div>
        </div>
      </div>
    </div>
  )
}
