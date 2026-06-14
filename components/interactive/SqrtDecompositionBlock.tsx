'use client'

import { useState } from 'react'

const ARRAY = [1, 3, 7, 4, 2, 9, 5, 6, 8, 2, 7, 3, 4, 6, 9, 5]
const N = ARRAY.length
const BLOCK_SIZE = 4
const NUM_BLOCKS = Math.ceil(N / BLOCK_SIZE)

const BLOCK_COLORS = [
  { bg: 'bg-teal-400 dark:bg-teal-600',   border: 'border-teal-300 dark:border-teal-500',   light: 'bg-teal-50 dark:bg-teal-900/20',  text: 'text-teal-700 dark:text-teal-300' },
  { bg: 'bg-cyan-400 dark:bg-cyan-600',   border: 'border-cyan-300 dark:border-cyan-500',   light: 'bg-cyan-50 dark:bg-cyan-900/20',  text: 'text-cyan-700 dark:text-cyan-300' },
  { bg: 'bg-sky-400 dark:bg-sky-600',     border: 'border-sky-300 dark:border-sky-500',     light: 'bg-sky-50 dark:bg-sky-900/20',    text: 'text-sky-700 dark:text-sky-300' },
  { bg: 'bg-blue-400 dark:bg-blue-600',   border: 'border-blue-300 dark:border-blue-500',   light: 'bg-blue-50 dark:bg-blue-900/20',  text: 'text-blue-700 dark:text-blue-300' },
]

const blockSums = Array.from({ length: NUM_BLOCKS }, (_, bi) => {
  let s = 0
  for (let i = bi * BLOCK_SIZE; i < Math.min((bi + 1) * BLOCK_SIZE, N); i++) s += ARRAY[i]
  return s
})

const PRESETS = [
  { l: 2,  r: 13, label: 'L=2, R=13' },
  { l: 0,  r: 15, label: 'L=0, R=15' },
  { l: 3,  r: 7,  label: 'L=3, R=7'  },
  { l: 5,  r: 11, label: 'L=5, R=11' },
]

export function SqrtDecompositionBlock() {
  const [presetIdx, setPresetIdx] = useState(0)
  const { l, r } = PRESETS[presetIdx]

  const lb = Math.floor(l / BLOCK_SIZE)
  const rb = Math.floor(r / BLOCK_SIZE)

  function classify(i: number): 'out' | 'left-edge' | 'full' | 'right-edge' {
    if (i < l || i > r) return 'out'
    const bi = Math.floor(i / BLOCK_SIZE)
    const blockStart = bi * BLOCK_SIZE
    const blockEnd = Math.min(blockStart + BLOCK_SIZE - 1, N - 1)
    if (bi > lb && bi < rb) return 'full'
    return bi === lb ? 'left-edge' : 'right-edge'
  }

  let sum = 0
  const ops: string[] = []
  if (lb === rb) {
    for (let i = l; i <= r; i++) sum += ARRAY[i]
    ops.push(`同块内逐元素：[${l}..${r}] → ${r - l + 1} 次`)
  } else {
    const leftTail = BLOCK_SIZE - (l % BLOCK_SIZE)
    for (let i = l; i < (lb + 1) * BLOCK_SIZE; i++) sum += ARRAY[i]
    ops.push(`左边缘逐元素：[${l}..${(lb + 1) * BLOCK_SIZE - 1}] → ${leftTail} 次`)
    for (let bi = lb + 1; bi < rb; bi++) { sum += blockSums[bi]; ops.push(`块 B[${bi}] 整块：O(1)`) }
    const rightHead = r % BLOCK_SIZE + 1
    for (let i = rb * BLOCK_SIZE; i <= r; i++) sum += ARRAY[i]
    ops.push(`右边缘逐元素：[${rb * BLOCK_SIZE}..${r}] → ${rightHead} 次`)
  }
  const totalOps = lb === rb ? (r - l + 1) : (l % BLOCK_SIZE === 0 ? 0 : BLOCK_SIZE - l % BLOCK_SIZE) + (rb - lb - 1) + (r % BLOCK_SIZE + 1)

  return (
    <div className="rounded-2xl border border-teal-200 dark:border-teal-700 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-teal-600 to-cyan-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">📦 分块（√n 分解）：范围求和可视化</h3>
        <p className="text-teal-50 text-xs mt-0.5">
          n={N}, 块大小 √n={BLOCK_SIZE}，共 {NUM_BLOCKS} 块。整块 O(1)，边缘逐元素
        </p>
        <div className="flex gap-2 mt-3 flex-wrap">
          {PRESETS.map((p, i) => (
            <button key={i} onClick={() => setPresetIdx(i)}
              className={`px-2.5 py-1 text-xs rounded-lg transition-all ${
                presetIdx === i ? 'bg-white text-teal-700 font-bold' : 'bg-white/25 text-white hover:bg-white/35'
              }`}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-5 space-y-5">
        {/* Array visualization */}
        <div>
          <p className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2">数组元素（共 {N} 个）</p>
          <div className="flex gap-0.5 flex-wrap">
            {ARRAY.map((v, i) => {
              const bi = Math.floor(i / BLOCK_SIZE)
              const cls = classify(i)
              const color = BLOCK_COLORS[bi % BLOCK_COLORS.length]
              const isJoin = i > 0 && Math.floor((i - 1) / BLOCK_SIZE) !== bi
              return (
                <div key={i} className={`flex flex-col items-center ${isJoin ? 'ml-1' : ''}`}>
                  <div className={`w-8 h-8 rounded-md flex items-center justify-center text-xs font-bold transition-all border ${
                    cls === 'out'
                      ? 'bg-slate-100 dark:bg-slate-800 text-slate-300 dark:text-slate-600 border-slate-200 dark:border-slate-700'
                      : cls === 'full'
                      ? `${color.bg} text-white ${color.border} border scale-105`
                      : `${color.light} ${color.text} ${color.border} border font-bold ring-2 ring-orange-400`
                  }`}>{v}</div>
                  <span className="text-[9px] text-slate-400 dark:text-slate-600 mt-0.5">{i}</span>
                </div>
              )
            })}
          </div>
          {/* Block labels */}
          <div className="flex gap-0.5 mt-1 flex-wrap">
            {Array.from({ length: NUM_BLOCKS }, (_, bi) => (
              <div key={bi} className={`flex items-center justify-center text-[9px] font-mono rounded ml-${bi > 0 ? 1 : 0} ${BLOCK_COLORS[bi % BLOCK_COLORS.length].text}`}
                style={{ width: `${BLOCK_SIZE * 34}px` }}>
                B[{bi}]={blockSums[bi]}
              </div>
            ))}
          </div>
        </div>

        {/* Legend */}
        <div className="flex gap-3 text-[10px] flex-wrap">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-teal-400 inline-block"></span> 整块 O(1)</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded border-2 border-orange-400 inline-block"></span> 边缘逐元素</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-slate-200 dark:bg-slate-700 inline-block"></span> 区间外</span>
        </div>

        {/* Operations breakdown */}
        <div className="rounded-xl border border-teal-200 dark:border-teal-800 bg-teal-50 dark:bg-teal-900/10 p-4">
          <p className="text-xs font-semibold text-teal-700 dark:text-teal-300 mb-2">
            查询 sum[{l}..{r}] = <span className="text-teal-900 dark:text-teal-100 text-sm font-bold">{sum}</span>
          </p>
          <div className="space-y-1.5">
            {ops.map((op, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="w-5 h-5 rounded-full bg-teal-500 text-white flex items-center justify-center text-[10px] font-bold flex-shrink-0">
                  {i + 1}
                </span>
                <span className="text-slate-700 dark:text-slate-300 font-mono">{op}</span>
              </div>
            ))}
          </div>
          <div className="mt-3 pt-2 border-t border-teal-200 dark:border-teal-800 flex justify-between text-xs">
            <span className="text-teal-600 dark:text-teal-400">
              总操作次数：{totalOps === 0 ? '≤ 2√n' : totalOps}
            </span>
            <span className="text-slate-500 dark:text-slate-400">
              复杂度：O(√{N}) = O({BLOCK_SIZE})
            </span>
          </div>
        </div>

        <p className="text-[11px] text-slate-400 dark:text-slate-600 text-center border-t border-slate-100 dark:border-slate-800 pt-3">
          √n 分解 = 权衡：预处理 O(n)，查询/更新 O(√n)；不如线段树，但实现极简且常数小
        </p>
      </div>
    </div>
  )
}
