'use client'

import { useState, useMemo } from 'react'

// ---------------------------------------------------------------------------
// Z-function construction with step log
// ---------------------------------------------------------------------------
interface ZStep {
  i: number
  l: number; r: number          // current [l, r] window BEFORE processing i
  zi: number                    // Z[i] computed
  initVal: number               // initial value before extension
  extended: boolean
  tag: 'zero' | 'copy' | 'clamp_ext' | 'extend'
  desc: string
  Z: number[]
}

function buildZSteps(s: string): ZStep[] {
  const n = s.length
  const Z = new Array(n).fill(0); Z[0] = n
  const steps: ZStep[] = []
  let l = 0, r = 0

  for (let i = 1; i < n; i++) {
    const prevL = l, prevR = r
    let init = 0, tag: ZStep['tag'] = 'extend'

    if (i < r) {
      const mirror = i - l
      if (Z[mirror] < r - i) {
        init = Z[mirror]; Z[i] = init; tag = 'copy'
      } else {
        init = r - i; Z[i] = init; tag = 'clamp_ext'
      }
    } else {
      tag = 'extend'
    }

    const before = Z[i]
    while (i + Z[i] < n && s[Z[i]] === s[i + Z[i]]) Z[i]++
    const extended = Z[i] > before

    if (Z[i] > 0 && i + Z[i] > r) { l = i; r = i + Z[i] }

    const desc =
      tag === 'copy'      ? `i=${i}：Z[i−l]=${Z[i - l]} < r−i=${prevR - i}，直接复制，Z[${i}]=${Z[i]}` :
      tag === 'clamp_ext' ? `i=${i}：Z[i−l]≥r−i，取初始值 ${init}，向右扩展→Z[${i}]=${Z[i]}` :
                            `i=${i}：i≥r，从 0 开始暴力扩展→Z[${i}]=${Z[i]}`

    steps.push({ i, l: prevL, r: prevR, zi: Z[i], initVal: init, extended, tag, desc, Z: [...Z] })
  }
  return steps
}

const EXAMPLES = [
  { label: 'aabxaa', s: 'aabxaa' },
  { label: 'abab', s: 'abab' },
  { label: 'aaaaaa', s: 'aaaaaa' },
  { label: 'abcabcab', s: 'abcabcab' },
]

const TAG_STYLE = {
  zero:      'bg-gray-50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400',
  copy:      'bg-sky-50 dark:bg-sky-900/30 border-sky-200 dark:border-sky-700 text-sky-800 dark:text-sky-200',
  clamp_ext: 'bg-amber-50 dark:bg-amber-900/30 border-amber-200 dark:border-amber-700 text-amber-800 dark:text-amber-200',
  extend:    'bg-orange-50 dark:bg-orange-900/30 border-orange-200 dark:border-orange-700 text-orange-800 dark:text-orange-200',
}
const TAG_LABEL = {
  zero:      '— 跳过',
  copy:      '📦 从 Z[mirror] 直接复制',
  clamp_ext: '✂️+🚀 截断后扩展',
  extend:    '🔍 暴力扩展',
}

export default function ZFunctionBuildViz() {
  const [exIdx, setExIdx] = useState(0)
  const [stepIdx, setStepIdx] = useState(0)

  const { s } = EXAMPLES[exIdx]
  const steps = useMemo(() => buildZSteps(s), [s])
  const step = steps[stepIdx]

  function reset() { setStepIdx(0) }

  // Z[0] = n always
  const displayZ = [s.length, ...step.Z.slice(1)]

  return (
    <div className="rounded-2xl overflow-hidden border border-amber-200 dark:border-amber-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-amber-500 to-orange-500 dark:from-amber-600 dark:to-orange-600 px-5 py-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-white font-bold text-base">⚡ Z 函数构造步进动画</h3>
          <p className="text-amber-100 text-xs mt-0.5">[l, r] 窗口动态维护，Z[i] 值复用/扩展过程一目了然</p>
        </div>
        <div className="flex gap-1 flex-wrap">
          {EXAMPLES.map((e, i) => (
            <button key={i} onClick={() => { setExIdx(i); reset() }}
              className={`px-2.5 py-1 text-xs rounded-lg transition-all ${exIdx === i ? 'bg-white text-amber-700 font-semibold' : 'bg-amber-400/40 text-white hover:bg-amber-400/60'}`}>
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
            className="px-2.5 py-1 text-xs bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded-lg disabled:opacity-40">← 上一步</button>
          <button onClick={() => setStepIdx(v => Math.min(steps.length - 1, v + 1))} disabled={stepIdx === steps.length - 1}
            className="px-3 py-1 text-xs bg-amber-500 text-white rounded-lg hover:bg-amber-600 disabled:opacity-40">下一步 →</button>
          <button onClick={() => setStepIdx(steps.length - 1)} className="px-2.5 py-1 text-xs bg-gray-100 dark:bg-gray-800 text-gray-500 rounded-lg">跳至结束</button>
          <span className="text-xs text-gray-400">步骤 {stepIdx + 1} / {steps.length}</span>
        </div>

        {/* Tag + description */}
        <div className={`px-3 py-2 rounded-xl border text-xs leading-relaxed ${TAG_STYLE[step.tag]}`}>
          <span className="font-semibold mr-2">{TAG_LABEL[step.tag]}</span>
          {step.desc}
        </div>

        {/* Window status */}
        <div className="flex flex-wrap gap-3 text-xs">
          <span className="px-2.5 py-1 bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 rounded-lg font-mono">
            [l, r] = [{step.l}, {step.r})
          </span>
          <span className="px-2.5 py-1 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded-lg font-mono">
            i = {step.i}
          </span>
          {step.tag !== 'extend' && step.tag !== 'zero' && (
            <span className="px-2.5 py-1 bg-sky-100 dark:bg-sky-900/30 text-sky-700 dark:text-sky-300 rounded-lg font-mono">
              mirror i−l = {step.i - step.l}
            </span>
          )}
        </div>

        {/* String visualization */}
        <div className="overflow-x-auto">
          <div className="inline-block min-w-full space-y-1">
            {/* Index */}
            <div className="flex">
              <div className="flex-shrink-0 w-8 text-[10px] text-gray-400 text-right pr-1 leading-7">idx</div>
              {s.split('').map((_, i) => (
                <div key={i} className="w-8 text-[10px] text-center text-gray-300 dark:text-gray-600 leading-7">{i}</div>
              ))}
            </div>

            {/* [l, r) window bracket */}
            <div className="flex">
              <div className="flex-shrink-0 w-8" />
              {s.split('').map((_, i) => {
                const inWindow = i >= step.l && i < step.r
                return (
                  <div key={i} className="w-8 h-1.5 flex items-center justify-center">
                    {inWindow && (
                      <div className={`h-1 w-full ${i === step.l ? 'rounded-l ml-1' : ''} ${i === step.r - 1 ? 'rounded-r mr-1' : ''} bg-orange-300 dark:bg-orange-600`} />
                    )}
                  </div>
                )
              })}
            </div>
            <div className="flex">
              <div className="flex-shrink-0 w-8 text-[10px] text-orange-400 text-right pr-1">[l,r)</div>
              {s.split('').map((_, i) => {
                const isL = i === step.l, isR = i === step.r
                return (
                  <div key={i} className="w-8 text-[9px] text-center text-orange-400 leading-3">
                    {isL ? 'l' : ''}{isR ? 'r' : ''}
                  </div>
                )
              })}
            </div>

            {/* String chars */}
            <div className="flex mt-1">
              <div className="flex-shrink-0 w-8 text-[10px] text-gray-400 text-right pr-1 leading-8">s</div>
              {s.split('').map((c, i) => {
                const isCurrent = i === step.i
                const inWindow = i >= step.l && i < step.r
                const isMirror = step.tag !== 'extend' && step.tag !== 'zero' && i === step.i - step.l
                // Matching prefix region: chars 0..Z[i]-1 match chars i..i+Z[i]-1
                const inMatch = i >= step.i && i < step.i + step.zi
                const inPrefix = i < step.zi && step.zi > 0
                return (
                  <div key={i} className={`w-8 h-8 text-sm font-mono font-bold rounded text-center leading-8 border transition-all ${
                    isCurrent ? 'bg-amber-500 text-white border-amber-500 scale-105 shadow' :
                    isMirror  ? 'bg-sky-300 dark:bg-sky-700 text-sky-900 dark:text-sky-100 border-sky-400' :
                    inMatch   ? 'bg-amber-100 dark:bg-amber-900/40 text-amber-800 dark:text-amber-200 border-amber-200 dark:border-amber-700' :
                    inPrefix  ? 'bg-amber-100 dark:bg-amber-900/40 text-amber-800 dark:text-amber-200 border-amber-200 dark:border-amber-700' :
                    inWindow  ? 'bg-orange-50 dark:bg-orange-900/20 text-gray-500 border-orange-100 dark:border-orange-900' :
                               'bg-gray-50 dark:bg-gray-800 text-gray-400 border-transparent'
                  }`}>
                    {c}
                  </div>
                )
              })}
            </div>

            {/* Z array */}
            <div className="flex">
              <div className="flex-shrink-0 w-8 text-[10px] text-gray-400 text-right pr-1 leading-8">Z</div>
              {displayZ.map((z, i) => (
                <div key={i} className={`w-8 h-8 text-sm font-mono font-bold rounded text-center leading-8 transition-all ${
                  i === 0         ? 'text-gray-400 bg-gray-100 dark:bg-gray-800' :
                  i === step.i    ? 'bg-amber-500 text-white shadow' :
                  z > 0           ? 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300' :
                                    'text-gray-300 dark:text-gray-600'
                }`}>
                  {z || '0'}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-3 text-[11px] text-gray-500 dark:text-gray-400">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-amber-500 inline-block" />当前 i</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-sky-300 dark:bg-sky-700 inline-block" />mirror 位置</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-amber-100 dark:bg-amber-900/40 inline-block" />当前匹配段</span>
          <span className="flex items-center gap-1"><span className="h-1 w-6 rounded bg-orange-300 dark:bg-orange-600 inline-block" />[l, r) 窗口</span>
        </div>

        {/* Applications summary */}
        {stepIdx === steps.length - 1 && (
          <div className="p-3 rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700">
            <p className="text-xs font-semibold text-amber-700 dark:text-amber-300 mb-1">✅ Z 数组构建完成</p>
            <div className="flex flex-wrap gap-2">
              {displayZ.map((z, i) => z > 1 && (
                <span key={i} className="text-[11px] font-mono bg-white dark:bg-gray-800 px-2 py-0.5 rounded-lg border border-amber-200 dark:border-amber-700 text-amber-700 dark:text-amber-300">
                  Z[{i}]={z} → 前缀"{s.slice(0,z)}"在位置{i}出现
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
