'use client'

import React, { useState, useCallback, useEffect, useRef } from 'react'
import { Play, Pause, SkipBack, ChevronLeft, ChevronRight } from 'lucide-react'

// Concrete example for Exchange Argument on Activity Selection
// Activities sorted by finish time:
//  a₁[1,4)  a₂[2,5)  a₃[4,8)  a₄[6,10)  a₅[9,12)
// Greedy chooses: a₁ → a₃ → a₅  (3 activities)
// Hypothetical OPT: a₂ → a₃ → a₅  (also 3 activities; first choice differs)
// Exchange: replace a₂ with a₁ in OPT → OPT' = Greedy

interface Activity { id: string; s: number; f: number }

const ACTS: Activity[] = [
  { id: 'a\u2081', s: 1, f: 4  },
  { id: 'a\u2082', s: 2, f: 5  },
  { id: 'a\u2083', s: 4, f: 8  },
  { id: 'a\u2084', s: 6, f: 10 },
  { id: 'a\u2085', s: 9, f: 12 },
]

const GREEDY_SOL = ['a\u2081', 'a\u2083', 'a\u2085']
const OPT_SOL    = ['a\u2082', 'a\u2083', 'a\u2085']
const OPT_PRIME  = ['a\u2081', 'a\u2083', 'a\u2085']

type CellState = 'neutral' | 'highlight' | 'exchange-out' | 'exchange-in' | 'match'

interface ExStep {
  desc: string
  detail: string
  greedyStates: CellState[]
  optStates:    CellState[]
  optPrimeVisible: boolean
  optPrimeStates: CellState[]
  annotation?: string
}

const EX_STEPS: ExStep[] = [
  {
    desc: 'SETUP',
    detail: 'PLACEHOLDER',
    greedyStates: ['neutral','neutral','neutral'],
    optStates: ['neutral','neutral','neutral'],
    optPrimeVisible: false,
    optPrimeStates: ['neutral','neutral','neutral'],
  },
]

// We build STEPS inline below
const STEP_DATA: ExStep[] = [
  {
    desc: '\u6240\u6709\u6d3b\u52a8\u6309\u7ed3\u675f\u65f6\u95f4\u6392\u5e8f',
    detail: '\u5171 5 \u4e2a\u6d3b\u52a8\uff0c\u6309\u7ed3\u675f\u65f6\u95f4\u5347\u5e8f\u6392\u5217\uff1aa\u2081[1,4) < a\u2082[2,5) < a\u2083[4,8) < a\u2084[6,10) < a\u2085[9,12)\u3002\u8fd9\u662f\u8d2a\u5fc3\u7b97\u6cd5\u8fd0\u884c\u7684\u57fa\u7840\u3002',
    greedyStates: ['neutral','neutral','neutral'],
    optStates: ['neutral','neutral','neutral'],
    optPrimeVisible: false,
    optPrimeStates: ['neutral','neutral','neutral'],
  },
  {
    desc: '\u6784\u9020\u8d2a\u5fc3\u89e3\uff08Greedy\uff09\u4e0e\u5047\u8bbe\u6700\u4f18\u89e3\uff08OPT\uff09',
    detail: '\u8d2a\u5fc3\u7b97\u6cd5\u9009\u51fa\uff1aa\u2081 \u2192 a\u2083 \u2192 a\u2085\uff08\u5171 3 \u4e2a\u6d3b\u52a8\uff09\u3002\u5047\u8bbe\u5b58\u5728\u67d0\u4e2a\u6700\u4f18\u89e3 OPT \u4e0e\u8d2a\u5fc3\u4e0d\u540c\uff0c\u4f8b\u5982 OPT = {a\u2082, a\u2083, a\u2085}\uff0c\u540c\u6837\u5305\u542b 3 \u4e2a\u6d3b\u52a8\u3002|OPT|=|Greedy|=3\uff0c\u8bf4\u660e\u8d2a\u5fc3\u81f3\u5c11\u4e0d\u6bd4 OPT \u5dee\u3002',
    greedyStates: ['highlight','highlight','highlight'],
    optStates: ['highlight','highlight','highlight'],
    optPrimeVisible: false,
    optPrimeStates: ['neutral','neutral','neutral'],
  },
  {
    desc: '\u627e\u51fa\u7b2c\u4e00\u4e2a\u4e0d\u540c\u4f4d\u7f6e\uff1a\u4f4d\u7f6e 1',
    detail: '\u9010\u69fd\u6bd4\u8f83\uff1aGreedy[1]=a\u2081\uff0cOPT[1]=a\u2082\u3002\u8fd9\u662f\u7b2c\u4e00\u4e2a\u5206\u6b67\u70b9\u3002Greedy \u9009\u62e9\u4e86\u7ed3\u675f\u66f4\u65e9\u7684 a\u2081\uff08finish=4\uff09\uff0c\u800c OPT \u9009\u62e9\u4e86 a\u2082\uff08finish=5\uff09\u3002\u8d2a\u5fc3\u6027\u8d28\uff1aa\u2081.finish \u2264 a\u2082.finish\u3002',
    greedyStates: ['exchange-in','neutral','neutral'],
    optStates: ['exchange-out','neutral','neutral'],
    optPrimeVisible: false,
    optPrimeStates: ['neutral','neutral','neutral'],
    annotation: 'a\u2081.finish(4) \u2264 a\u2082.finish(5)',
  },
  {
    desc: '\u6267\u884c\u4ea4\u6362\uff1a\u5c06 OPT \u4e2d\u7684 a\u2082 \u6362\u6210 a\u2081',
    detail: "\u6784\u9020 OPT'\uff1a\u628a OPT \u4e2d\u7684 a\u2082 \u66ff\u6362\u4e3a a\u2081\u3002\u9a8c\u8bc1\uff1a\u2460 a\u2081 \u7ed3\u675f\u4e8e t=4\uff0c\u4e0b\u4e00\u6d3b\u52a8 a\u2083 \u5f00\u59cb\u4e8e t=4\uff0c4\u22654 \u6ee1\u8db3\u4e0d\u91cd\u53e0\u6761\u4ef6 \u2705\uff1b\u2461 a\u2081 \u4e0e\u524d\u5e8f\u6d3b\u52a8\u4e0d\u51b2\u7a81 \u2705\u3002\u56e0\u6b64 OPT' = {a\u2081, a\u2083, a\u2085} \u4ecd\u7136\u662f\u53ef\u884c\u89e3\u3002",
    greedyStates: ['match','neutral','neutral'],
    optStates: ['exchange-out','neutral','neutral'],
    optPrimeVisible: true,
    optPrimeStates: ['exchange-in','neutral','neutral'],
    annotation: 'a\u2081.finish(4) \u2264 a\u2083.start(4) \u2705 \u517c\u5bb9\uff01',
  },
  {
    desc: "\u9a8c\u8bc1 OPT' \u8d28\u91cf\u4e0d\u4e0b\u964d",
    detail: "|OPT'| = |OPT| = 3\u3002\u7b49\u91cf\u66ff\u6362\uff0c\u89e3\u7684\u5927\u5c0f\u4e0d\u53d8\uff0cOPT' \u540c\u6837\u662f\u6700\u4f18\u89e3\u3002\u8fd9\u8bc1\u660e\u4e86\u201c\u4ee5\u8d2a\u5fc3\u9009\u62e9 a\u2081 \u5f00\u5934\u7684\u6700\u4f18\u89e3 OPT' \u5b58\u5728\u201d\u3002",
    greedyStates: ['match','neutral','neutral'],
    optStates: ['exchange-out','highlight','highlight'],
    optPrimeVisible: true,
    optPrimeStates: ['exchange-in','highlight','highlight'],
    annotation: "|OPT'| = |OPT| = 3 \u2705 \u8d28\u91cf\u4e0d\u53d8",
  },
  {
    desc: '\u5f52\u7eb3\uff1a\u9012\u5f52\u5904\u7406\u5269\u4f59\u5b50\u95ee\u9898',
    detail: "OPT' \u4e0e Greedy \u5728\u7b2c\u4e00\u69fd\u4e00\u81f4\uff08\u90fd\u662f a\u2081\uff09\u3002\u5bf9\u5269\u4f59\u6d3b\u52a8 {a\u2083, a\u2085} \u91cd\u590d\u540c\u6837\u8bba\u8bc1\u3002\u7531\u5f52\u7eb3\u6cd5\u77e5\uff0c\u8d2a\u5fc3\u89e3\u5c31\u662f\u6700\u4f18\u89e3 \u25a1",
    greedyStates: ['match','match','match'],
    optStates: ['neutral','neutral','neutral'],
    optPrimeVisible: true,
    optPrimeStates: ['match','match','match'],
    annotation: "OPT' = Greedy \u2192 Greedy \u4e3a\u6700\u4f18\u89e3 \u2705",
  },
]

const TMAX = 13

function SolutionBar({ label, solution, states }: {
  label: string; solution: string[]; states: CellState[]
}) {
  const acts = solution.map((id: string) => ACTS.find(a => a.id === id)!)
  const barClass: Record<CellState, string> = {
    'neutral':      'bg-slate-400/50 text-slate-600 dark:text-slate-300',
    'highlight':    'bg-blue-400 text-white',
    'exchange-out': 'bg-rose-400/70 text-white opacity-60',
    'exchange-in':  'bg-emerald-500 text-white shadow-lg ring-2 ring-emerald-400',
    'match':        'bg-indigo-500 text-white',
  }
  const pillClass: Record<CellState, string> = {
    'neutral':      'border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300',
    'highlight':    'border-blue-400 bg-blue-100 dark:bg-blue-950/50 text-blue-700 dark:text-blue-300',
    'exchange-out': 'border-rose-400 bg-rose-100 dark:bg-rose-950/50 text-rose-600 dark:text-rose-400',
    'exchange-in':  'border-emerald-400 bg-emerald-100 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 ring-2 ring-emerald-400 scale-105',
    'match':        'border-indigo-400 bg-indigo-100 dark:bg-indigo-950/50 text-indigo-700 dark:text-indigo-300',
  }
  return (
    <div className="space-y-2">
      <div className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">{label}</div>
      <div className="relative h-10 rounded-lg bg-slate-100 dark:bg-slate-800/60 overflow-hidden">
        {Array.from({ length: TMAX + 1 }).map((_, t) => (
          <div key={t} className="absolute top-0 h-full border-l border-slate-200/50 dark:border-slate-700/30" style={{ left: `${(t / TMAX) * 100}%` }}/>
        ))}
        {acts.map((act, i) => (
          <div
            key={act.id + i}
            className={`absolute top-1 bottom-1 rounded flex items-center justify-center text-xs font-bold transition-all duration-500 ${barClass[states[i]]}`}
            style={{ left: `${(act.s / TMAX) * 100}%`, width: `${((act.f - act.s) / TMAX) * 100}%` }}
          >{act.id}</div>
        ))}
      </div>
      <div className="flex gap-2 flex-wrap">
        {acts.map((act, i) => (
          <div key={act.id + i} className={`flex flex-col items-center px-2 py-1 rounded-lg border text-xs font-bold transition-all duration-500 ${pillClass[states[i]]}`}>
            <span>{act.id}</span>
            <span className="font-normal text-[10px] opacity-80">[{act.s},{act.f})</span>
          </div>
        ))}
      </div>
    </div>
  )
}

const SPEEDS = [0.5, 1, 1.5, 2]

export default function ExchangeArgumentAnimator() {
  const [stIdx, setStIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speedIdx, setSpeedIdx] = useState(1)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const cur = STEP_DATA[stIdx]
  const total = STEP_DATA.length - 1

  const stop = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current)
    timerRef.current = null
    setPlaying(false)
  }, [])

  useEffect(() => {
    if (!playing) return
    timerRef.current = setInterval(() => {
      setStIdx(prev => {
        if (prev >= total) { stop(); return prev }
        return prev + 1
      })
    }, 1600 / SPEEDS[speedIdx])
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [playing, speedIdx, total, stop])

  useEffect(() => { if (stIdx >= total && playing) stop() }, [stIdx, total, playing, stop])
  const goto = (n: number) => { stop(); setStIdx(Math.max(0, Math.min(total, n))) }

  return (
    <div className="w-full max-w-5xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      <div className="bg-gradient-to-r from-fuchsia-600 via-violet-600 to-indigo-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg">交换论证动画 — 活动选择贪心最优性证明</h3>
        <p className="text-indigo-100 text-sm mt-0.5">用具体数值演示「将 OPT 逐步变换为 Greedy，且质量不下降」</p>
      </div>

      <div className="p-4 space-y-4">
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={() => goto(0)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300"><SkipBack size={16}/></button>
          <button onClick={() => goto(stIdx - 1)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300"><ChevronLeft size={16}/></button>
          <button onClick={playing ? stop : () => setPlaying(true)} className="px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-700 text-white flex items-center gap-1.5 text-sm font-medium">
            {playing ? <Pause size={14}/> : <Play size={14}/>}{playing ? '暂停' : '播放'}
          </button>
          <button onClick={() => goto(stIdx + 1)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300"><ChevronRight size={16}/></button>
          <div className="ml-auto flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
            速度: {SPEEDS.map((s, i) => (
              <button key={i} onClick={() => setSpeedIdx(i)} className={`px-2 py-0.5 rounded ${speedIdx === i ? 'bg-violet-600 text-white' : 'border border-slate-200 dark:border-slate-700'}`}>{s}x</button>
            ))}
          </div>
        </div>

        <div className="w-full h-1.5 rounded-full bg-slate-100 dark:bg-slate-800">
          <div className="h-1.5 rounded-full bg-violet-500 transition-all" style={{ width: `${(stIdx / total) * 100}%` }}/>
        </div>

        <div className="flex gap-1.5">
          {STEP_DATA.map((_, i) => (
            <button key={i} onClick={() => goto(i)} className={`flex-1 py-1 rounded text-xs font-bold transition-all ${i === stIdx ? 'bg-violet-600 text-white' : i < stIdx ? 'bg-violet-200 dark:bg-violet-900/50 text-violet-700 dark:text-violet-300' : 'bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-500'}`}>
              {i + 1}
            </button>
          ))}
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-3">
          <div className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">活动表（已按结束时间排序）</div>
          <div className="flex flex-wrap gap-2">
            {ACTS.map(a => (
              <div key={a.id} className="px-2.5 py-1.5 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-xs font-mono text-slate-700 dark:text-slate-200 text-center">
                <div className="font-bold">{a.id}</div><div>[{a.s},{a.f})</div>
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-xl border border-blue-300 dark:border-blue-700 bg-blue-50 dark:bg-blue-950/20 p-3">
            <SolutionBar label="Greedy 解" solution={GREEDY_SOL} states={cur.greedyStates}/>
          </div>
          <div className="rounded-xl border border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-800/40 p-3">
            <SolutionBar label="假设 OPT 解" solution={OPT_SOL} states={cur.optStates}/>
          </div>
        </div>

        {cur.optPrimeVisible && (
          <div className="rounded-xl border border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-950/20 p-3">
            <SolutionBar label="OPT'（交换后）" solution={OPT_PRIME} states={cur.optPrimeStates}/>
          </div>
        )}

        {cur.annotation && (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-amber-50 dark:bg-amber-950/30 border border-amber-300 dark:border-amber-700">
            <span className="text-amber-700 dark:text-amber-400 text-sm font-mono font-bold">{cur.annotation}</span>
          </div>
        )}

        <div className="rounded-xl border border-violet-300 dark:border-violet-700 bg-violet-50 dark:bg-violet-950/30 px-4 py-3">
          <div className="font-bold text-violet-700 dark:text-violet-300 text-sm">{`步骤 ${stIdx + 1}/${total + 1}：`}{cur.desc}</div>
          <div className="mt-1.5 text-sm text-slate-600 dark:text-slate-300 leading-relaxed">{cur.detail}</div>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 px-4 py-3 text-xs text-slate-500 dark:text-slate-400 space-y-1">
          <div className="font-bold text-slate-700 dark:text-slate-200 text-sm">交换论证核心不变量</div>
          <div>📌 <strong>关键引理</strong>：若贪心第一步选 g，则存在以 g 开头的最优解。</div>
          <div>📌 <strong>证明思路</strong>：取任意最优解 OPT，将其首选换为 g，得到可行且等大的 OPT'，同为最优。</div>
          <div>📌 <strong>归纳步骤</strong>：对剩余子问题重复，直到 OPT' = Greedy。</div>
        </div>
      </div>
    </div>
  )
}
