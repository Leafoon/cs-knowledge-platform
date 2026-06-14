'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { Play, Pause, SkipBack, ChevronLeft, ChevronRight } from 'lucide-react'

/* ---- Types ---- */
interface TraceStep {
  label: string
  greedyPicked: number[]
  optPicked: number[]
  greedyRemain: number
  optRemain: number
  desc: string
  verdict?: 'correct' | 'wrong' | 'tie'
}

interface Scenario {
  title: string
  subtitle: string
  steps: TraceStep[]
  final: string
  isCorrect: boolean
  why: string
}

/* ---- Scenario 1: Coin [1,3,4]->6 where greedy FAILS ---- */
const COIN_BAD_STEPS: TraceStep[] = [
  {
    label: '初始', greedyPicked: [], optPicked: [], greedyRemain: 6, optRemain: 6,
    desc: '目标: 用硬币 [1,3,4] 凑出 6 元, 最少硬币数.',
  },
  {
    label: '贪心第1步', greedyPicked: [4], optPicked: [], greedyRemain: 2, optRemain: 6,
    desc: '贪心选最大硬币 4 (6-4=2). 最优方案暂无动作.',
    verdict: 'wrong',
  },
  {
    label: '贪心第2步', greedyPicked: [4,1], optPicked: [], greedyRemain: 1, optRemain: 6,
    desc: '剩余2, 选4不行, 选3不行 (2<3), 选1 (2-1=1).',
  },
  {
    label: '贪心第3步', greedyPicked: [4,1,1], optPicked: [], greedyRemain: 0, optRemain: 6,
    desc: '再选1 (1-1=0). 贪心结果: 4+1+1 = 3枚硬币.',
    verdict: 'wrong',
  },
  {
    label: '最优第1步', greedyPicked: [4,1,1], optPicked: [3], greedyRemain: 0, optRemain: 3,
    desc: '最优选3 (6-3=3). 跳过贪心最大值 4, 因为4不能整除剩余.',
    verdict: 'correct',
  },
  {
    label: '最优第2步', greedyPicked: [4,1,1], optPicked: [3,3], greedyRemain: 0, optRemain: 0,
    desc: '再选3 (3-3=0). 最优结果: 3+3 = 2枚硬币. 比贪心少1枚!',
    verdict: 'correct',
  },
]

/* ---- Scenario 2: 0/1 Knapsack greedy by density FAILS ---- */
const KNAPSACK_STEPS: TraceStep[] = [
  {
    label: '初始', greedyPicked: [], optPicked: [], greedyRemain: 50, optRemain: 50,
    desc: '容量50, 物品: A(价值60,重10,密度6), B(价值100,重20,密度5), C(价值120,重30,密度4). 贪心按密度降序: A>B>C.',
  },
  {
    label: '贪心选A', greedyPicked: [0], optPicked: [], greedyRemain: 40, optRemain: 50,
    desc: '贪心选密度最高的 A (w=10,v=60), 剩余容量40.',
    verdict: 'wrong',
  },
  {
    label: '贪心选B', greedyPicked: [0,1], optPicked: [], greedyRemain: 20, optRemain: 50,
    desc: '贪心选密度次高的 B (w=20,v=100), 剩余容量20.',
  },
  {
    label: '贪心选C?', greedyPicked: [0,1], optPicked: [], greedyRemain: 20, optRemain: 50,
    desc: 'C需30容量, 超过剩余20, 不选. 贪心总价值: 60+100 = 160.',
    verdict: 'wrong',
  },
  {
    label: '最优选B', greedyPicked: [0,1], optPicked: [1], greedyRemain: 20, optRemain: 30,
    desc: '最优选B (w=20,v=100), 剩余30.',
    verdict: 'correct',
  },
  {
    label: '最优选C', greedyPicked: [0,1], optPicked: [1,2], greedyRemain: 20, optRemain: 0,
    desc: '最优选C (w=30,v=120), 恰好用完. B+C=220 > 贪心160, 贪心失败!',
    verdict: 'correct',
  },
]

/* ---- Scenario 3: Standard coins [1,5,10,25]->30 greedy is CORRECT ---- */
const COIN_GOOD_STEPS: TraceStep[] = [
  {
    label: '初始', greedyPicked: [], optPicked: [], greedyRemain: 30, optRemain: 30,
    desc: '目标: 用硬币 [1,5,10,25] 凑出30元. 这是\"规范硬币系统\", 贪心正确!',
  },
  {
    label: '贪心第1步', greedyPicked: [25], optPicked: [25], greedyRemain: 5, optRemain: 5,
    desc: '贪心选25 (剩5). 最优方案同样选25.',
    verdict: 'tie',
  },
  {
    label: '贪心第2步', greedyPicked: [25,5], optPicked: [25,5], greedyRemain: 0, optRemain: 0,
    desc: '选5 (剩0). 贪心与最优一致: 25+5 = 2枚. 贪心成功!',
    verdict: 'tie',
  },
]

const SCENARIOS: Scenario[] = [
  {
    title: '反例: 找零 [1,3,4]\u21926',
    subtitle: '贪心失败',
    steps: COIN_BAD_STEPS,
    final: '贪心: 4+1+1 = 3枚  |  最优: 3+3 = 2枚',
    isCorrect: false,
    why: '硬币系统不满足\"规范性\", 大硬币(4)不能被小硬币整除覆盖.',
  },
  {
    title: '反例: 0/1背包按密度',
    subtitle: '贪心失败',
    steps: KNAPSACK_STEPS,
    final: '贪心: A+B = 160  |  最优: B+C = 220',
    isCorrect: false,
    why: '0/1背包不可分割, 密度高的物品可能浪费容量, 需要DP.',
  },
  {
    title: '正例: 找零 [1,5,10,25]\u219230',
    subtitle: '贪心正确',
    steps: COIN_GOOD_STEPS,
    final: '贪心: 25+5 = 2枚  |  最优: 也是 2枚',
    isCorrect: true,
    why: '每种硬币是下一种的因子(1|5|10), 贪心最优. 规范硬币系统有子结构性质.',
  },
]

const ITEM_LABELS = ['A(w=10,v=60)', 'B(w=20,v=100)', 'C(w=30,v=120)']
const ITEM_COLORS = ['bg-sky-500', 'bg-violet-500', 'bg-amber-500']
const COIN_COLORS: Record<number, string> = {
  1: 'bg-slate-400', 3: 'bg-emerald-500', 4: 'bg-rose-500',
  5: 'bg-blue-500', 10: 'bg-purple-500', 25: 'bg-amber-500',
}

function CoinChip({ val, isOpt }: { val: number; isOpt?: boolean }) {
  const col = COIN_COLORS[val] ?? 'bg-slate-500'
  return (
    <span className={`inline-flex items-center justify-center w-7 h-7 rounded-full text-xs font-black text-white ${col} shadow`}>{val}</span>
  )
}

export default function GreedyCounterexampleLab() {
  const [scIdx, setScIdx] = useState(0)
  const [stepIdx, setStepIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)

  const sc = SCENARIOS[scIdx]
  const step = sc.steps[Math.min(stepIdx, sc.steps.length - 1)]

  const stop = useCallback(() => { setPlaying(false) }, [])

  const goPrev = useCallback(() => { setStepIdx(p => Math.max(0, p - 1)) }, [])
  const goNext = useCallback(() => {
    setStepIdx(p => {
      if (p >= sc.steps.length - 1) { stop(); return p }
      return p + 1
    })
  }, [sc.steps.length, stop])

  const selectScenario = (i: number) => {
    setScIdx(i); setStepIdx(0); setPlaying(false)
  }

  // auto-play
  const togglePlay = useCallback(() => {
    if (playing) { stop(); return }
    if (stepIdx >= sc.steps.length - 1) setStepIdx(0)
    setPlaying(true)
  }, [playing, stepIdx, sc.steps.length, stop])

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setStepIdx(p => {
          if (p >= sc.steps.length - 1) { setPlaying(false); clearInterval(timerRef.current!); return p }
          return p + 1
        })
      }, 1200 / speed)
    } else {
      if (timerRef.current) clearInterval(timerRef.current)
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [playing, speed, sc.steps.length])

  const isKnapsack = scIdx === 1

  return (
    <div className="w-full max-w-4xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      <div className="bg-gradient-to-r from-rose-600 via-pink-600 to-purple-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg">贪心反例实验室 — 对比正例与反例</h3>
        <p className="text-pink-100 text-sm mt-0.5">逐步观察贪心决策如何走向次优解，以及何时贪心是正确的</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Scenario tabs */}
        <div className="flex gap-2 flex-wrap">
          {SCENARIOS.map((s, i) => (
            <button key={i} onClick={() => selectScenario(i)}
              className={`px-3 py-2 rounded-xl text-xs font-bold border-2 transition-all ${scIdx === i ? (s.isCorrect ? 'bg-emerald-600 text-white border-emerald-600' : 'bg-rose-600 text-white border-rose-600') : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300'}`}>
              {s.title}
            </button>
          ))}
        </div>

        {/* Step controls */}
        <div className="flex items-center gap-3 flex-wrap">
          <button onClick={() => { setStepIdx(0); stop() }} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300"><SkipBack size={15}/></button>
          <button onClick={goPrev} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300"><ChevronLeft size={15}/></button>
          <button onClick={togglePlay} className={`px-4 py-1.5 rounded-lg text-sm font-bold flex items-center gap-1.5 text-white ${sc.isCorrect ? 'bg-emerald-600' : 'bg-rose-600'}`}>
            {playing ? <><Pause size={14}/>暂停</> : <><Play size={14}/>播放</>}
          </button>
          <button onClick={goNext} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300"><ChevronRight size={15}/></button>
          <div className="flex items-center gap-1 ml-auto text-xs text-slate-400 dark:text-slate-500">
            {[0.5,1,1.5,2].map(s => (
              <button key={s} onClick={() => setSpeed(s)} className={`px-1.5 py-0.5 rounded ${speed===s ? 'bg-slate-200 dark:bg-slate-700 font-bold text-slate-700 dark:text-slate-200' : 'text-slate-400 dark:text-slate-500'}`}>{s}x</button>
            ))}
          </div>
          <span className="text-xs text-slate-400 dark:text-slate-500">{stepIdx+1}/{sc.steps.length}</span>
        </div>

        {/* Progress bar */}
        <div className="w-full h-1.5 rounded-full bg-slate-100 dark:bg-slate-800">
          <div className="h-1.5 rounded-full transition-all duration-300" style={{ width: `${(stepIdx/(sc.steps.length-1))*100}%`, background: sc.isCorrect ? '#10b981' : '#e11d48' }}/>
        </div>

        {/* Main comparison */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {/* Greedy side */}
          <div className={`rounded-xl border-2 p-3 ${sc.isCorrect ? 'border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-950/20' : 'border-rose-300 dark:border-rose-700 bg-rose-50 dark:bg-rose-950/20'}`}>
            <div className={`text-xs font-bold uppercase tracking-wider mb-2 ${sc.isCorrect ? 'text-emerald-700 dark:text-emerald-400' : 'text-rose-700 dark:text-rose-400'}`}>
              贪心方案 {sc.isCorrect ? '✅ 最优' : '❌ 次优'}
            </div>
            <div className="flex flex-wrap gap-1.5 min-h-8 items-center">
              {isKnapsack
                ? step.greedyPicked.map((itemIdx, i) => (
                    <span key={i} className={`px-2 py-0.5 rounded text-xs font-bold text-white ${ITEM_COLORS[itemIdx]}`}>{ITEM_LABELS[itemIdx]}</span>
                  ))
                : step.greedyPicked.map((c, i) => <CoinChip key={i} val={c}/>)
              }
              {step.greedyPicked.length === 0 && <span className="text-xs text-slate-400 italic">（尚未选择）</span>}
            </div>
            <div className={`mt-2 text-sm font-black ${sc.isCorrect ? 'text-emerald-700 dark:text-emerald-300' : 'text-rose-700 dark:text-rose-300'}`}>
              {isKnapsack
                ? `总价值: ${step.greedyPicked.reduce((s, i) => s + [60,100,120][i], 0)}`
                : `已选 ${step.greedyPicked.length} 枚，剩余 ${step.greedyRemain}`}
            </div>
          </div>

          {/* Optimal side */}
          <div className="rounded-xl border-2 border-indigo-300 dark:border-indigo-700 bg-indigo-50 dark:bg-indigo-950/20 p-3">
            <div className="text-xs font-bold uppercase tracking-wider mb-2 text-indigo-700 dark:text-indigo-400">
              最优方案 ✨
            </div>
            <div className="flex flex-wrap gap-1.5 min-h-8 items-center">
              {isKnapsack
                ? step.optPicked.map((itemIdx, i) => (
                    <span key={i} className={`px-2 py-0.5 rounded text-xs font-bold text-white ${ITEM_COLORS[itemIdx]}`}>{ITEM_LABELS[itemIdx]}</span>
                  ))
                : step.optPicked.map((c, i) => <CoinChip key={i} val={c} isOpt/>)
              }
              {step.optPicked.length === 0 && <span className="text-xs text-slate-400 italic">（尚未选择）</span>}
            </div>
            <div className="mt-2 text-sm font-black text-indigo-700 dark:text-indigo-300">
              {isKnapsack
                ? `总价值: ${step.optPicked.reduce((s, i) => s + [60,100,120][i], 0)}`
                : `已选 ${step.optPicked.length} 枚，剩余 ${step.optRemain}`}
            </div>
          </div>
        </div>

        {/* Step description */}
        <div className={`rounded-xl border px-4 py-3 text-sm leading-relaxed ${
          step.verdict === 'wrong' ? 'border-rose-200 dark:border-rose-800 bg-rose-50 dark:bg-rose-950/20 text-rose-700 dark:text-rose-300' :
          step.verdict === 'correct' ? 'border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/20 text-emerald-700 dark:text-emerald-300' :
          step.verdict === 'tie' ? 'border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-950/20 text-blue-700 dark:text-blue-300' :
          'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 text-slate-600 dark:text-slate-300'
        }`}>
          <span className="font-bold">[{step.label}]</span>{' '}{step.desc}
        </div>

        {/* Final result */}
        {stepIdx === sc.steps.length - 1 && (
          <div className={`rounded-xl border-2 p-3 ${sc.isCorrect ? 'border-emerald-400 dark:border-emerald-600 bg-emerald-50 dark:bg-emerald-950/20' : 'border-rose-400 dark:border-rose-600 bg-rose-50 dark:bg-rose-950/20'}`}>
            <div className="font-black text-sm">{sc.final}</div>
            <div className={`text-xs mt-1 ${sc.isCorrect ? 'text-emerald-700 dark:text-emerald-400' : 'text-rose-700 dark:text-rose-400'}`}>{sc.why}</div>
          </div>
        )}

        <div className="rounded-xl border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/20 px-4 py-3 text-xs text-amber-700 dark:text-amber-300">
          <span className="font-bold">贪心适用条件总结：</span> ①最优子结构 + ②贪心选择性质（局部最优 → 全局最优）缺一不可。0/1背包无贪心选择性，标准找零系统有；反例说明贪心不是万能的。
        </div>
      </div>
    </div>
  )
}
