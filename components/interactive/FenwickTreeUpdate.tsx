'use client'

import { useState } from 'react'

// ── 数据 ──────────────────────────────────────────────────────
const N = 16

/** lowbit(i) = i & (-i) */
function lowbit(i: number) { return i & (-i) }

/** 每个 BIT 格子负责的区间 [i-lowbit(i)+1, i] */
function range(i: number): [number, number] {
  return [i - lowbit(i) + 1, i]
}

/** UPDATE(pos, delta) 路径 */
function updatePath(pos: number): number[] {
  const path: number[] = []
  let i = pos
  while (i <= N) { path.push(i); i += lowbit(i) }
  return path
}

/** 二进制字符串，右对齐固定宽度 */
function toBin(i: number, w = 4) {
  return i.toString(2).padStart(w, '0')
}

// 预设场景
const PRESETS = [
  { pos: 3,  delta: 5,  label: 'pos=3 (0011)' },
  { pos: 5,  delta: 2,  label: 'pos=5 (0101)' },
  { pos: 12, delta: 7,  label: 'pos=12 (1100)' },
  { pos: 6,  delta: 4,  label: 'pos=6 (0110)' },
]

// 每个格子固定颜色（按 lowbit 大小）
function cellColor(i: number, inPath: boolean, active: boolean) {
  if (active)  return 'bg-teal-400 dark:bg-teal-400 text-slate-900 border-teal-300 scale-110 shadow-lg shadow-teal-400/40'
  if (inPath)  return 'bg-teal-200 dark:bg-teal-800 text-teal-900 dark:text-teal-100 border-teal-400'
  const lb = lowbit(i)
  if (lb === 8) return 'bg-violet-100 dark:bg-violet-900/40 text-violet-800 dark:text-violet-200 border-violet-300 dark:border-violet-700'
  if (lb === 4) return 'bg-blue-100 dark:bg-blue-900/40 text-blue-800 dark:text-blue-200 border-blue-300 dark:border-blue-700'
  if (lb === 2) return 'bg-indigo-100 dark:bg-indigo-900/40 text-indigo-800 dark:text-indigo-200 border-indigo-300 dark:border-indigo-700'
  return 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-600'
}

export function FenwickTreeUpdate() {
  const [presetIdx, setPresetIdx] = useState(0)
  const [stepIdx, setStepIdx]     = useState(-1) // -1 = 未开始

  const preset = PRESETS[presetIdx]
  const path   = updatePath(preset.pos)
  const totalSteps = path.length
  const activePos  = stepIdx >= 0 && stepIdx < totalSteps ? path[stepIdx] : null

  function changePreset(i: number) { setPresetIdx(i); setStepIdx(-1) }
  function start()    { setStepIdx(0) }
  function prev()     { setStepIdx(s => Math.max(0, s - 1)) }
  function next()     { setStepIdx(s => Math.min(totalSteps - 1, s + 1)) }
  function reset()    { setStepIdx(-1) }

  // 当前步骤文字
  const stepDescriptions: string[] = path.map((p, idx) => {
    const lb = lowbit(p)
    const next = p + lb
    const [lo, hi] = range(p)
    return idx < path.length - 1
      ? `i=${p} (${toBin(p)}): tree[${p}] 管辖 A[${lo}..${hi}]，加上 Δ=${preset.delta}；跳到 ${p}+lowbit(${p})=${p}+${lb}=${next}`
      : `i=${p} (${toBin(p)}): tree[${p}] 管辖 A[${lo}..${hi}]，加上 Δ=${preset.delta}；${next} > ${N}，结束`
  })

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-lg font-sans">
      {/* 顶部：青绿渐变 */}
      <div className="bg-gradient-to-r from-teal-500 via-emerald-600 to-cyan-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">🔼 BIT 点更新：UPDATE(i, Δ) 路径</h3>
        <p className="text-teal-100 text-xs mt-0.5">
          从位置 <code className="bg-white/20 px-1 rounded">i</code> 出发，每步跳到{' '}
          <code className="bg-white/20 px-1 rounded">i += lowbit(i)</code>，沿路更新 tree[i]
        </p>
        <div className="flex gap-2 mt-3 flex-wrap">
          {PRESETS.map((p, idx) => (
            <button key={idx} onClick={() => changePreset(idx)}
              className={`px-2.5 py-1 text-xs rounded-lg font-mono font-medium transition-all ${
                presetIdx === idx ? 'bg-white text-teal-700 shadow' : 'bg-white/25 text-white hover:bg-white/35'}`}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-5 space-y-5">
        {/* 颜色图例 */}
        <div className="flex gap-3 flex-wrap text-xs">
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm bg-teal-400" /><span className="text-slate-500 dark:text-slate-400">当前节点（激活）</span></div>
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm bg-teal-200 dark:bg-teal-800" /><span className="text-slate-500 dark:text-slate-400">路径节点</span></div>
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm bg-violet-100 dark:bg-violet-900/60" /><span className="text-slate-500 dark:text-slate-400">lowbit=8（管辖8格）</span></div>
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm bg-blue-100 dark:bg-blue-900/60" /><span className="text-slate-500 dark:text-slate-400">lowbit=4</span></div>
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm bg-indigo-100 dark:bg-indigo-900/60" /><span className="text-slate-500 dark:text-slate-400">lowbit=2</span></div>
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm bg-slate-100 dark:bg-slate-800" /><span className="text-slate-500 dark:text-slate-400">lowbit=1（管辖1格）</span></div>
        </div>

        {/* BIT 数组格子 */}
        <div>
          <div className="text-xs text-slate-400 dark:text-slate-500 font-mono mb-1 pl-0.5">tree[1..{N}]</div>
          <div className="grid grid-cols-16 gap-1" style={{ gridTemplateColumns: `repeat(${N}, 1fr)` }}>
            {Array.from({ length: N }, (_, k) => {
              const i = k + 1
              const inPath = path.includes(i)
              const isActive = i === activePos
              const [lo, hi] = range(i)
              return (
                <div key={i} className={`
                  relative flex flex-col items-center rounded border text-[10px] font-mono font-bold
                  py-1.5 cursor-default transition-all duration-300 ${cellColor(i, inPath, isActive)}
                `}>
                  {/* 格子编号 */}
                  <span className="font-bold text-[11px]">{i}</span>
                  {/* 管辖区间 */}
                  <span className="text-[9px] font-normal opacity-70 leading-tight">[{lo},{hi}]</span>
                  {/* 跳跃箭头标记 */}
                  {isActive && (
                    <span className="absolute -top-2.5 left-1/2 -translate-x-1/2 text-teal-500 dark:text-teal-300 text-xs animate-bounce">▼</span>
                  )}
                </div>
              )
            })}
          </div>

          {/* 下方低倍标尺：显示 lowbit */}
          <div className="grid mt-1 gap-1" style={{ gridTemplateColumns: `repeat(${N}, 1fr)` }}>
            {Array.from({ length: N }, (_, k) => {
              const i = k + 1
              return (
                <div key={i} className="text-center text-[9px] font-mono text-slate-400 dark:text-slate-600">
                  {lowbit(i)}
                </div>
              )
            })}
          </div>
          <div className="text-[10px] text-center text-slate-400 dark:text-slate-600 mt-0.5">lowbit(i)</div>
        </div>

        {/* 路径跳跃可视化 */}
        {stepIdx >= 0 && (
          <div className="space-y-2">
            <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
              UPDATE({preset.pos}, {preset.delta}) — 步骤 {stepIdx + 1}/{totalSteps}
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              {path.map((p, idx) => (
                <div key={idx} className="flex items-center gap-1">
                  <div className={`
                    px-2 py-1 rounded-lg text-xs font-mono font-bold border transition-all duration-300
                    ${idx < stepIdx  ? 'bg-teal-100 dark:bg-teal-900/40 text-teal-700 dark:text-teal-300 border-teal-300 dark:border-teal-600 opacity-60' : ''}
                    ${idx === stepIdx ? 'bg-teal-400 text-slate-900 border-teal-400 shadow-md shadow-teal-400/30 scale-105' : ''}
                    ${idx > stepIdx  ? 'bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-600 border-slate-200 dark:border-slate-700' : ''}
                  `}>
                    [{p}]
                  </div>
                  {idx < path.length - 1 && (
                    <span className={`text-xs transition-opacity duration-300 ${idx < stepIdx ? 'text-teal-400' : 'text-slate-300 dark:text-slate-700'}`}>→</span>
                  )}
                </div>
              ))}
            </div>
            {/* 当前步骤说明 */}
            {stepIdx < totalSteps && (
              <div className="rounded-xl bg-teal-50 dark:bg-teal-900/20 border border-teal-100 dark:border-teal-800 px-4 py-3">
                <p className="text-xs text-teal-800 dark:text-teal-200 leading-relaxed font-mono">
                  {stepDescriptions[stepIdx]}
                </p>
              </div>
            )}
          </div>
        )}

        {stepIdx === -1 && (
          <div className="rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 px-4 py-3 text-center">
            <p className="text-xs text-slate-500 dark:text-slate-400">
              每个格子 <code className="bg-slate-200 dark:bg-slate-700 px-1 rounded">tree[i]</code> 管辖从{' '}
              <code className="bg-slate-200 dark:bg-slate-700 px-1 rounded">i-lowbit(i)+1</code> 到{' '}
              <code className="bg-slate-200 dark:bg-slate-700 px-1 rounded">i</code> 的区间和。
              点击"开始演示"查看 UPDATE 的跳跃路径。
            </p>
          </div>
        )}

        {/* 控制按钮 */}
        <div className="flex items-center justify-between">
          <button onClick={reset}
            className="px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
            ⏮ 重置
          </button>
          <div className="flex gap-2">
            {stepIdx === -1 ? (
              <button onClick={start}
                className="px-4 py-1.5 text-xs rounded-lg bg-teal-600 hover:bg-teal-700 text-white font-medium transition-colors">
                ▶ 开始演示
              </button>
            ) : (
              <>
                <button onClick={prev} disabled={stepIdx === 0}
                  className="px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
                  ← 上一步
                </button>
                <button onClick={next} disabled={stepIdx === totalSteps - 1}
                  className="px-3 py-1.5 text-xs rounded-lg bg-teal-600 hover:bg-teal-700 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
                  下一步 →
                </button>
              </>
            )}
          </div>
        </div>

        <div className="text-center text-xs text-slate-400 dark:text-slate-600 border-t border-slate-100 dark:border-slate-800 pt-3">
          UPDATE 最多跳 ⌊log₂ n⌋ 次（每次消除 i 的最高位）→ 时间复杂度 O(log n)
        </div>
      </div>
    </div>
  )
}
