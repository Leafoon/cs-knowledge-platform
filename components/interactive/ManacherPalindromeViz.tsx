'use client'

import { useState, useMemo } from 'react'

// ---------------------------------------------------------------------------
// Manacher algorithm: returns step-by-step execution log on transformed string T
// T = "#a#b#a#" etc. — length 2n+1 for original length n
// ---------------------------------------------------------------------------
interface Step {
  i: number          // current index in T
  center: number     // current center in T
  maxRight: number   // current max_right in T
  mirror: number     // mirror of i relative to center
  pBefore: number    // P[i] before expansion
  pAfter: number     // P[i] after expansion
  P: number[]        // full P array snapshot
  desc: string
  tag: 'copy' | 'clamp' | 'extend' | 'init'
}

function buildManacher(s: string): { T: string; steps: Step[] } {
  // Build T
  const T = '#' + s.split('').join('#') + '#'
  const m = T.length
  const P = new Array(m).fill(0)
  const steps: Step[] = []
  let center = 0, maxRight = 0

  for (let i = 0; i < m; i++) {
    const mirror = 2 * center - i
    let pBefore = 0
    let tag: Step['tag'] = 'init'

    if (i < maxRight) {
      if (P[mirror] < maxRight - i) {
        pBefore = P[mirror]
        tag = 'copy'
      } else {
        pBefore = maxRight - i
        tag = 'clamp'
      }
    }

    P[i] = pBefore

    // Expand
    while (i - P[i] - 1 >= 0 && i + P[i] + 1 < m && T[i - P[i] - 1] === T[i + P[i] + 1]) {
      P[i]++
    }

    const pAfter = P[i]
    if (pAfter > pBefore) tag = 'extend'

    const desc =
      tag === 'copy'   ? `i=${i}('${T[i]}')：P[i] 从镜像 P[${mirror}]=${P[mirror]} 直接复制，无需扩展` :
      tag === 'clamp'  ? `i=${i}('${T[i]}')：P[i] 初始值截断至 max_right−i=${pBefore}，尝试扩展` :
      tag === 'extend' ? `i=${i}('${T[i]}')：P[i] 从 ${pBefore} 扩展到 ${pAfter}，更新 center/max_right` :
                         `i=${i}('${T[i]}')：在 max_right 外，初始 P[i]=0，直接扩展到 ${pAfter}`

    steps.push({ i, center, maxRight, mirror, pBefore, pAfter, P: [...P], desc, tag })

    if (i + P[i] > maxRight) {
      center = i
      maxRight = i + P[i]
    }
  }

  return { T, steps }
}

const EXAMPLES = [
  { label: 'racecar', s: 'racecar' },
  { label: 'abba', s: 'abba' },
  { label: 'bananas', s: 'bananas' },
  { label: 'abaab', s: 'abaab' },
]

const TAG_STYLE = {
  copy:   'bg-sky-50 dark:bg-sky-900/30 border-sky-200 dark:border-sky-700 text-sky-800 dark:text-sky-200',
  clamp:  'bg-amber-50 dark:bg-amber-900/30 border-amber-200 dark:border-amber-700 text-amber-800 dark:text-amber-200',
  extend: 'bg-emerald-50 dark:bg-emerald-900/30 border-emerald-200 dark:border-emerald-700 text-emerald-800 dark:text-emerald-200',
  init:   'bg-fuchsia-50 dark:bg-fuchsia-900/30 border-fuchsia-200 dark:border-fuchsia-700 text-fuchsia-800 dark:text-fuchsia-200',
}
const TAG_LABEL = { copy: '📦 直接复制', clamp: '✂️ 截断+尝试扩展', extend: '🚀 扩展 + 更新', init: '🔍 从 0 扩展' }

export default function ManacherPalindromeViz() {
  const [exIdx, setExIdx] = useState(0)
  const [stepIdx, setStepIdx] = useState(0)

  const { s } = EXAMPLES[exIdx]
  const { T, steps } = useMemo(() => buildManacher(s), [s])
  const step = steps[stepIdx]

  // Find best palindrome
  const bestI = useMemo(() => step.P.indexOf(Math.max(...step.P)), [step])
  const bestLen = useMemo(() => step.P[bestI], [step, bestI])

  function reset() { setStepIdx(0) }

  return (
    <div className="rounded-2xl overflow-hidden border border-fuchsia-200 dark:border-fuchsia-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-fuchsia-600 to-pink-600 dark:from-fuchsia-700 dark:to-pink-700 px-5 py-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-white font-bold text-base">🪞 Manacher 算法逐步可视化</h3>
          <p className="text-fuchsia-100 text-xs mt-0.5">center / max_right 双指针逐步更新，P[] 回文半径数组</p>
        </div>
        <div className="flex gap-1 flex-wrap">
          {EXAMPLES.map((e, i) => (
            <button key={i} onClick={() => { setExIdx(i); reset() }}
              className={`px-2.5 py-1 text-xs rounded-lg transition-all ${exIdx === i ? 'bg-white text-fuchsia-700 font-semibold' : 'bg-fuchsia-400/40 text-white hover:bg-fuchsia-400/60'}`}>
              {e.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 p-4 space-y-3">
        {/* Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={reset} className="px-2.5 py-1 text-xs bg-gray-100 dark:bg-gray-800 text-gray-500 rounded-lg">↺ 重置</button>
          <button onClick={() => setStepIdx(v => Math.max(0, v - 1))} disabled={stepIdx === 0}
            className="px-2.5 py-1 text-xs bg-fuchsia-100 dark:bg-fuchsia-900/30 text-fuchsia-700 dark:text-fuchsia-300 rounded-lg disabled:opacity-40">← 上一步</button>
          <button onClick={() => setStepIdx(v => Math.min(steps.length - 1, v + 1))} disabled={stepIdx === steps.length - 1}
            className="px-3 py-1 text-xs bg-fuchsia-600 text-white rounded-lg hover:bg-fuchsia-700 disabled:opacity-40">下一步 →</button>
          <button onClick={() => setStepIdx(steps.length - 1)} className="px-2.5 py-1 text-xs bg-gray-100 dark:bg-gray-800 text-gray-500 rounded-lg">跳至结束</button>
          <span className="text-xs text-gray-400">步骤 {stepIdx + 1} / {steps.length}</span>
        </div>

        {/* Step tag + description */}
        <div className={`px-3 py-2 rounded-xl border text-xs leading-relaxed ${TAG_STYLE[step.tag]}`}>
          <span className="font-semibold mr-2">{TAG_LABEL[step.tag]}</span>
          {step.desc}
        </div>

        {/* Pointer display */}
        <div className="flex flex-wrap gap-3 text-xs">
          <span className="px-2.5 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-lg font-mono">
            center = {step.center}  ({T[step.center]})
          </span>
          <span className="px-2.5 py-1 bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 rounded-lg font-mono">
            max_right = {step.maxRight}
          </span>
          <span className="px-2.5 py-1 bg-fuchsia-100 dark:bg-fuchsia-900/30 text-fuchsia-700 dark:text-fuchsia-300 rounded-lg font-mono">
            i = {step.i}  P[i] = {step.pAfter}
          </span>
        </div>

        {/* Transformed string T + P[] array */}
        <div className="overflow-x-auto">
          <div className="inline-block min-w-full">
            {/* index row */}
            <div className="flex mb-0.5">
              <div className="flex-shrink-0 w-8 text-[10px] text-gray-400 text-right pr-1 leading-5">idx</div>
              {T.split('').map((_, i) => (
                <div key={i} className="w-7 text-[9px] text-center text-gray-300 dark:text-gray-600">{i}</div>
              ))}
            </div>

            {/* max_right boundary indicator */}
            <div className="flex mb-0.5 relative">
              <div className="flex-shrink-0 w-8" />
              {T.split('').map((_, i) => (
                <div key={i} className="w-7 h-1 flex items-center justify-center">
                  {i === step.maxRight && <div className="w-0.5 h-3 bg-orange-400" title="max_right" />}
                </div>
              ))}
            </div>

            {/* T chars */}
            <div className="flex mb-1">
              <div className="flex-shrink-0 w-8 text-[10px] text-gray-400 text-right pr-1 leading-7">T</div>
              {T.split('').map((c, i) => {
                const isCurrent = i === step.i
                const isCenter = i === step.center
                const isMirror = i === step.mirror && step.tag !== 'init'
                const inPalin   = Math.abs(i - step.i) <= step.pAfter
                const inMaxRight = i < step.maxRight

                return (
                  <div key={i} className={`w-7 h-7 text-xs font-mono font-bold rounded text-center leading-7 border transition-all ${
                    isCurrent  ? 'bg-fuchsia-500 text-white border-fuchsia-500 scale-110 shadow' :
                    isCenter   ? 'bg-blue-400 text-white border-blue-400' :
                    isMirror   ? 'bg-sky-200 dark:bg-sky-700 text-sky-800 dark:text-sky-200 border-sky-300' :
                    inPalin && c !== '#' ? 'bg-fuchsia-100 dark:bg-fuchsia-900/40 text-fuchsia-800 dark:text-fuchsia-200 border-fuchsia-200 dark:border-fuchsia-700' :
                    inMaxRight ? 'bg-orange-50 dark:bg-orange-900/20 text-gray-500 border-orange-100 dark:border-orange-900' :
                    'bg-gray-50 dark:bg-gray-800 text-gray-400 border-transparent'
                  }`}>
                    {c}
                  </div>
                )
              })}
            </div>

            {/* P[] array */}
            <div className="flex">
              <div className="flex-shrink-0 w-8 text-[10px] text-gray-400 text-right pr-1 leading-7">P</div>
              {step.P.map((p, i) => (
                <div key={i} className={`w-7 h-7 text-xs font-mono font-bold rounded text-center leading-7 transition-all ${
                  i === step.i ? 'bg-fuchsia-500 text-white' :
                  p > 0       ? 'bg-fuchsia-100 dark:bg-fuchsia-900/40 text-fuchsia-700 dark:text-fuchsia-300' :
                                 'text-gray-300 dark:text-gray-600'
                }`}>
                  {p || '·'}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-3 text-[11px] text-gray-500 dark:text-gray-400">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-fuchsia-500 inline-block" />当前 i</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-blue-400 inline-block" />center</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-sky-200 dark:bg-sky-700 inline-block" />镜像 mirror</span>
          <span className="flex items-center gap-1"><span className="w-0.5 h-3 bg-orange-400 inline-block" />max_right 界</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-900 inline-block" />已扩展区域</span>
        </div>

        {/* Current best palindrome */}
        {bestLen > 0 && (
          <div className="p-2.5 rounded-xl bg-fuchsia-50 dark:bg-fuchsia-900/20 border border-fuchsia-200 dark:border-fuchsia-700 text-xs">
            <span className="font-semibold text-fuchsia-700 dark:text-fuchsia-300">
              当前最长回文：P[{bestI}]={step.P[bestI]}
            </span>
            <span className="text-gray-500 dark:text-gray-400 ml-2">
              → 原串中的 "{s.slice(Math.floor((bestI - step.P[bestI]) / 2), Math.floor((bestI + step.P[bestI]) / 2))}"
              （长度 {step.P[bestI]}）
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
