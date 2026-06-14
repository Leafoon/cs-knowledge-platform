'use client'
import React, { useEffect, useRef, useState } from 'react'

const EXAMPLE = {
  x: 1234,
  y: 5678,
  a: 12,
  b: 34,
  c: 56,
  d: 78,
  z2: 672,
  z0: 2652,
  z1: 6164,
  middle: 2840,
  result: 7006652,
}

interface Step {
  title: string
  desc: string
  activeCards: string[]
  formula: string
  highlightValue?: string
}

const STEPS: Step[] = [
  {
    title: '把两个 4 位数拆成高位与低位',
    desc: '将 1234 和 5678 以 m=2 为界拆成两半：1234 = 12·10² + 34，5678 = 56·10² + 78。这一步是 Divide。',
    activeCards: ['split-x', 'split-y'],
    formula: 'x = a·10^m + b,  y = c·10^m + d',
    highlightValue: 'm = 2',
  },
  {
    title: '先算高位乘积 z₂ = a·c',
    desc: '第一部分是高位对高位：12 × 56 = 672。它最终会乘上 10^(2m) 放回最高位。',
    activeCards: ['z2'],
    formula: 'z₂ = a·c = 12×56 = 672',
    highlightValue: '672',
  },
  {
    title: '再算低位乘积 z₀ = b·d',
    desc: '第二部分是低位对低位：34 × 78 = 2652。它直接放在最低位。',
    activeCards: ['z0'],
    formula: 'z₀ = b·d = 34×78 = 2652',
    highlightValue: '2652',
  },
  {
    title: '关键优化：只额外算一次 (a+b)(c+d)',
    desc: '普通分治会再算 ad 和 bc 两次乘法。Karatsuba 用一次乘法 (a+b)(c+d) 替代两次乘法。这里 (12+34)(56+78)=46×134=6164。',
    activeCards: ['z1', 'sum'],
    formula: 'z₁ = (a+b)(c+d) = 46×134 = 6164',
    highlightValue: '6164',
  },
  {
    title: '恢复中间项 ad+bc',
    desc: '根据恒等式：(a+b)(c+d) = ac + ad + bc + bd，所以 ad+bc = z₁ - z₂ - z₀ = 6164 - 672 - 2652 = 2840。',
    activeCards: ['middle', 'z1', 'z2', 'z0'],
    formula: 'ad + bc = z₁ - z₂ - z₀ = 2840',
    highlightValue: '2840',
  },
  {
    title: '合并得到最终答案',
    desc: '最后 Combine：x·y = z₂·10^(2m) + (ad+bc)·10^m + z₀ = 672·10⁴ + 2840·10² + 2652 = 7006652。',
    activeCards: ['result', 'z2', 'middle', 'z0'],
    formula: 'xy = z₂·10^(2m) + (ad+bc)·10^m + z₀',
    highlightValue: '7006652',
  },
]

function isActive(step: Step, key: string) {
  return step.activeCards.includes(key)
}

export default function KaratsubaLargeIntMult() {
  const [cur, setCur] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1800)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const step = STEPS[cur]

  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setCur(prev => {
          if (prev >= STEPS.length - 1) {
            setPlaying(false)
            return prev
          }
          return prev + 1
        })
      }, speed)
    } else if (timerRef.current) {
      clearInterval(timerRef.current)
    }

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [playing, speed])

  const Card = ({ id, title, value, tone = 'slate' }: { id: string; title: string; value: React.ReactNode; tone?: 'slate' | 'blue' | 'emerald' | 'violet' | 'amber' | 'rose' }) => {
    const active = isActive(step, id)
    const tones = {
      slate: active ? 'border-slate-400 dark:border-slate-500 bg-slate-100 dark:bg-slate-800/70' : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900',
      blue: active ? 'border-blue-400 dark:border-blue-500 bg-blue-50 dark:bg-blue-950/40' : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900',
      emerald: active ? 'border-emerald-400 dark:border-emerald-500 bg-emerald-50 dark:bg-emerald-950/40' : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900',
      violet: active ? 'border-violet-400 dark:border-violet-500 bg-violet-50 dark:bg-violet-950/40' : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900',
      amber: active ? 'border-amber-400 dark:border-amber-500 bg-amber-50 dark:bg-amber-950/40' : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900',
      rose: active ? 'border-rose-400 dark:border-rose-500 bg-rose-50 dark:bg-rose-950/40' : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900',
    }
    return (
      <div className={`rounded-xl border p-3 transition-all duration-300 ${tones[tone]}`}>
        <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400 font-bold mb-1">{title}</div>
        <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">{value}</div>
      </div>
    )
  }

  return (
    <div className="w-full max-w-4xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      <div className="bg-gradient-to-r from-violet-600 via-fuchsia-600 to-rose-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Karatsuba 大整数乘法</h3>
        <p className="text-violet-100 text-sm mt-0.5">3 次乘法替代 4 次乘法 · 复杂度从 O(n²) 降到 O(n^1.585)</p>
      </div>

      <div className="px-4 py-3 flex flex-wrap gap-2 items-center border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40">
        <div className="px-3 py-1 rounded-full bg-violet-100 dark:bg-violet-900/40 text-violet-700 dark:text-violet-300 text-xs font-bold">
          示例：1234 × 5678
        </div>
        <div className="flex gap-1.5 ml-auto">
          <button onClick={() => { setPlaying(false); setCur(0) }} className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300">⏮</button>
          <button onClick={() => { setPlaying(false); setCur(v => Math.max(0, v - 1)) }} disabled={cur === 0} className="px-2 py-1.5 rounded-lg text-xs disabled:opacity-40 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300">◀</button>
          <button onClick={() => setPlaying(v => !v)} disabled={cur >= STEPS.length - 1} className="px-4 py-1.5 rounded-lg text-xs font-bold bg-violet-600 hover:bg-violet-700 text-white disabled:opacity-40">{playing ? '⏸ 暂停' : '▶ 播放'}</button>
          <button onClick={() => { setPlaying(false); setCur(v => Math.min(STEPS.length - 1, v + 1)) }} disabled={cur >= STEPS.length - 1} className="px-2 py-1.5 rounded-lg text-xs disabled:opacity-40 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300">▶</button>
        </div>
        <div className="flex items-center gap-1.5 text-[10px] text-slate-400">
          <input type="range" min={900} max={3200} step={300} value={speed} onChange={e => setSpeed(Number(e.target.value))} className="w-16 accent-violet-500" />
          <span>{(speed / 1000).toFixed(1)}s</span>
        </div>
      </div>

      <div className="p-4 space-y-4">
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold text-white bg-violet-500">{cur + 1}/{STEPS.length}</span>
          <span className="font-semibold text-sm text-slate-700 dark:text-slate-200">{step.title}</span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          <div className="lg:col-span-3 space-y-3">
            <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 p-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <Card id="split-x" title="x 的拆分" tone="blue" value={<span>1234 = <span className="text-blue-600 dark:text-blue-400 font-black">12</span>·10² + <span className="text-blue-600 dark:text-blue-400 font-black">34</span></span>} />
                <Card id="split-y" title="y 的拆分" tone="blue" value={<span>5678 = <span className="text-blue-600 dark:text-blue-400 font-black">56</span>·10² + <span className="text-blue-600 dark:text-blue-400 font-black">78</span></span>} />
                <Card id="z2" title="高位乘积 z₂" tone="emerald" value={<span>12 × 56 = <span className="font-black text-emerald-600 dark:text-emerald-400">672</span></span>} />
                <Card id="z0" title="低位乘积 z₀" tone="emerald" value={<span>34 × 78 = <span className="font-black text-emerald-600 dark:text-emerald-400">2652</span></span>} />
                <Card id="sum" title="额外一次乘法" tone="amber" value={<span>(12+34)(56+78) = 46 × 134</span>} />
                <Card id="z1" title="z₁" tone="amber" value={<span>46 × 134 = <span className="font-black text-amber-600 dark:text-amber-400">6164</span></span>} />
                <Card id="middle" title="中间项 ad+bc" tone="violet" value={<span>6164 − 672 − 2652 = <span className="font-black text-violet-600 dark:text-violet-400">2840</span></span>} />
                <Card id="result" title="最终答案" tone="rose" value={<span>672·10⁴ + 2840·10² + 2652 = <span className="font-black text-rose-600 dark:text-rose-400">7006652</span></span>} />
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400 font-bold mb-2">核心公式</div>
              <div className="rounded-xl bg-slate-50 dark:bg-slate-800/60 px-4 py-3 font-mono text-sm text-slate-700 dark:text-slate-200 overflow-x-auto">
                {step.formula}
              </div>
              {step.highlightValue && (
                <div className="mt-3 inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-violet-50 dark:bg-violet-950/40 border border-violet-200 dark:border-violet-700 text-violet-700 dark:text-violet-300 text-xs font-bold">
                  当前关键值：{step.highlightValue}
                </div>
              )}
            </div>
          </div>

          <div className="lg:col-span-2 space-y-3">
            <div className="rounded-2xl border border-violet-200 dark:border-violet-700/60 bg-violet-50 dark:bg-violet-950/30 p-4">
              <div className="text-[10px] uppercase tracking-wider text-violet-600 dark:text-violet-400 font-bold mb-2">为什么更快？</div>
              <div className="space-y-2 text-[11px] text-slate-600 dark:text-slate-300">
                <p>普通分治：需要 <b>4</b> 次规模为 n/2 的递归乘法。</p>
                <p>Karatsuba：只需要 <b>3</b> 次规模为 n/2 的递归乘法。</p>
                <p>因此递推式从 <span className="font-mono">T(n)=4T(n/2)+O(n)</span> 变成 <span className="font-mono">T(n)=3T(n/2)+O(n)</span>。</p>
              </div>
            </div>

            <div className="rounded-2xl border border-emerald-200 dark:border-emerald-700/60 bg-emerald-50 dark:bg-emerald-950/30 p-4 text-center">
              <div className="text-[10px] uppercase tracking-wider text-emerald-600 dark:text-emerald-400 font-bold mb-2">复杂度对比</div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="rounded-xl bg-white dark:bg-slate-900 border border-emerald-200 dark:border-emerald-700 p-3">
                  <div className="text-slate-500 dark:text-slate-400 text-[10px] mb-1">普通乘法</div>
                  <div className="font-black text-slate-800 dark:text-slate-100">O(n²)</div>
                </div>
                <div className="rounded-xl bg-white dark:bg-slate-900 border border-violet-200 dark:border-violet-700 p-3">
                  <div className="text-slate-500 dark:text-slate-400 text-[10px] mb-1">Karatsuba</div>
                  <div className="font-black text-violet-600 dark:text-violet-400">O(n^1.585)</div>
                </div>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 p-4">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400 font-bold mb-2">递归树直觉</div>
              <div className="space-y-2 text-[11px] text-slate-600 dark:text-slate-300">
                <div className="flex items-center justify-between rounded-lg bg-white dark:bg-slate-900 px-3 py-2 border border-slate-200 dark:border-slate-700"><span>第 0 层</span><span className="font-bold">1 个问题</span></div>
                <div className="flex items-center justify-between rounded-lg bg-white dark:bg-slate-900 px-3 py-2 border border-slate-200 dark:border-slate-700"><span>第 1 层</span><span className="font-bold">3 个子问题</span></div>
                <div className="flex items-center justify-between rounded-lg bg-white dark:bg-slate-900 px-3 py-2 border border-slate-200 dark:border-slate-700"><span>第 2 层</span><span className="font-bold">3² 个子问题</span></div>
                <div className="text-xs text-slate-500 dark:text-slate-400">递归分支数减少，整棵树就明显变瘦了。</div>
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 px-4 py-3 text-[12px] text-slate-600 dark:text-slate-300">
          {step.desc}
        </div>

        <div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
          <div className="h-full rounded-full bg-gradient-to-r from-violet-500 via-fuchsia-500 to-rose-500 transition-all duration-500" style={{ width: `${(cur / (STEPS.length - 1)) * 100}%` }} />
        </div>
      </div>
    </div>
  )
}
