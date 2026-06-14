'use client'

import { useState } from 'react'

// ── 工具函数 ──────────────────────────────────────────────────
const N = 16

function lowbit(i: number) { return i & (-i) }
function range(i: number): [number, number] { return [i - lowbit(i) + 1, i] }

/** QUERY(pos) 路径：i → i - lowbit(i) → ... → 0 */
function queryPath(pos: number): number[] {
  const path: number[] = []
  let i = pos
  while (i > 0) { path.push(i); i -= lowbit(i) }
  return path
}

function toBin(i: number, w = 4) { return i.toString(2).padStart(w, '0') }

// 预设场景：(query 位置, 对应的前缀和描述)
const PRESETS = [
  { pos: 6,  label: 'query(6)',  desc: 'A[1]+A[2]+...+A[6]' },
  { pos: 12, label: 'query(12)', desc: 'A[1]+...+A[12]' },
  { pos: 7,  label: 'query(7)',  desc: 'A[1]+A[2]+...+A[7]' },
  { pos: 10, label: 'query(10)', desc: 'A[1]+...+A[10]' },
]

function cellStyle(i: number, inPath: boolean, isActive: boolean) {
  if (isActive) return 'bg-violet-500 dark:bg-violet-400 text-white border-violet-400 scale-110 shadow-md shadow-violet-400/40'
  if (inPath)   return 'bg-violet-100 dark:bg-violet-900/50 text-violet-800 dark:text-violet-200 border-violet-400'
  const lb = lowbit(i)
  if (lb === 8) return 'bg-rose-50 dark:bg-rose-900/20 text-rose-800 dark:text-rose-300 border-rose-200 dark:border-rose-800'
  if (lb === 4) return 'bg-orange-50 dark:bg-orange-900/20 text-orange-800 dark:text-orange-300 border-orange-200 dark:border-orange-800'
  if (lb === 2) return 'bg-amber-50 dark:bg-amber-900/20 text-amber-800 dark:text-amber-300 border-amber-200 dark:border-amber-800'
  return 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 border-slate-200 dark:border-slate-700'
}

export function FenwickTreeQuery() {
  const [presetIdx, setPresetIdx] = useState(0)
  const [stepIdx, setStepIdx]     = useState(-1)

  const preset = PRESETS[presetIdx]
  const path   = queryPath(preset.pos)
  const totalSteps = path.length
  const activePos  = stepIdx >= 0 && stepIdx < totalSteps ? path[stepIdx] : null

  function changePreset(i: number) { setPresetIdx(i); setStepIdx(-1) }

  const steps = path.map((p, idx) => {
    const lb = lowbit(p)
    const next = p - lb
    const [lo, hi] = range(p)
    return idx < path.length - 1
      ? `i=${p} (${toBin(p)}): sum += tree[${p}]（管辖 A[${lo}..${hi}]）；跳到 ${p}-lowbit(${p})=${p}-${lb}=${next}`
      : `i=${p} (${toBin(p)}): sum += tree[${p}]（管辖 A[${lo}..${hi}]）；${next}=0，停止。前缀和计算完毕！`
  })

  // 合并的区间展示
  const coveredRanges = path.slice(0, stepIdx + 1).map(p => range(p))

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-lg font-sans">
      {/* 顶部：紫色渐变，与 FenwickTreeUpdate 明显区分 */}
      <div className="bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">🔍 BIT 前缀查询：QUERY(i) 路径</h3>
        <p className="text-violet-100 text-xs mt-0.5">
          从位置 <code className="bg-white/20 px-1 rounded">i</code> 出发，每步跳到{' '}
          <code className="bg-white/20 px-1 rounded">i -= lowbit(i)</code>，累加 tree[i]
        </p>
        <div className="flex gap-2 mt-3 flex-wrap">
          {PRESETS.map((p, idx) => (
            <button key={idx} onClick={() => changePreset(idx)}
              className={`px-2.5 py-1 text-xs rounded-lg font-mono font-medium transition-all ${
                presetIdx === idx ? 'bg-white text-violet-700 shadow' : 'bg-white/25 text-white hover:bg-white/35'}`}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-5 space-y-5">
        {/* BIT 数组 */}
        <div>
          <div className="text-xs text-slate-400 dark:text-slate-500 font-mono mb-1 pl-0.5">tree[1..{N}]</div>
          <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${N}, 1fr)` }}>
            {Array.from({ length: N }, (_, k) => {
              const i = k + 1
              const inPath = path.includes(i)
              const isActive = i === activePos
              const [lo, hi] = range(i)
              return (
                <div key={i} className={`
                  relative flex flex-col items-center rounded border text-[10px] font-mono font-bold
                  py-1.5 cursor-default transition-all duration-300 ${cellStyle(i, inPath, isActive)}
                `}>
                  <span className="text-[11px] font-bold">{i}</span>
                  <span className="text-[9px] font-normal opacity-70 leading-tight">[{lo},{hi}]</span>
                  {isActive && <span className="absolute -top-2.5 left-1/2 -translate-x-1/2 text-violet-500 text-xs">▼</span>}
                </div>
              )
            })}
          </div>
          {/* lowbit 标尺 */}
          <div className="grid mt-1 gap-1" style={{ gridTemplateColumns: `repeat(${N}, 1fr)` }}>
            {Array.from({ length: N }, (_, k) => (
              <div key={k} className="text-center text-[9px] font-mono text-slate-300 dark:text-slate-700">{lowbit(k+1)}</div>
            ))}
          </div>
        </div>

        {/* 路径跳跃 + 累加区间可视化 */}
        {stepIdx >= 0 ? (
          <div className="space-y-3">
            {/* 跳跃链 */}
            <div>
              <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">
                QUERY({preset.pos}) — 步骤 {stepIdx + 1}/{totalSteps}
              </div>
              <div className="flex items-center gap-1.5 flex-wrap">
                {path.map((p, idx) => (
                  <div key={idx} className="flex items-center gap-1">
                    <div className={`
                      flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-mono font-bold border transition-all duration-300
                      ${idx < stepIdx  ? 'bg-violet-100 dark:bg-violet-900/40 text-violet-600 dark:text-violet-300 border-violet-200 dark:border-violet-700 opacity-60' : ''}
                      ${idx ===stepIdx ? 'bg-violet-500 text-white border-violet-500 shadow scale-105' : ''}
                      ${idx > stepIdx  ? 'bg-slate-100 dark:bg-slate-800 text-slate-400 border-slate-200 dark:border-slate-700' : ''}
                    `}>
                      tree[{p}]
                      {idx <= stepIdx && (
                        <span className={`text-[9px] ml-0.5 ${idx < stepIdx ? 'text-violet-500' : 'text-violet-100'}`}>
                          ✓
                        </span>
                      )}
                    </div>
                    {idx < path.length - 1 && (
                      <span className={`text-[10px] ${idx < stepIdx ? 'text-violet-400' : 'text-slate-300 dark:text-slate-700'}`}>←</span>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* 累加的区间可视化 */}
            <div className="rounded-xl bg-violet-50 dark:bg-violet-900/20 border border-violet-100 dark:border-violet-800 p-3">
              <p className="text-xs text-violet-800 dark:text-violet-200 leading-relaxed font-mono">{steps[stepIdx]}</p>
              {coveredRanges.length > 0 && (
                <div className="mt-2 flex items-center gap-2 flex-wrap">
                  <span className="text-[10px] text-violet-600 dark:text-violet-400 shrink-0">已覆盖区间：</span>
                  {coveredRanges.map(([lo, hi], idx) => (
                    <span key={idx}
                      className="px-1.5 py-0.5 rounded bg-violet-200 dark:bg-violet-800 text-violet-800 dark:text-violet-200 text-[10px] font-mono">
                      [{lo},{hi}]
                    </span>
                  ))}
                  <span className="text-[10px] text-violet-500">
                    → {stepIdx === totalSteps - 1 ? `= ${preset.desc}` : '（累加中…）'}
                  </span>
                </div>
              )}
            </div>

            {/* 完成状态 */}
            {stepIdx === totalSteps - 1 && (
              <div className="rounded-xl bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-700 px-4 py-2 text-center">
                <p className="text-xs font-semibold text-emerald-700 dark:text-emerald-300">
                  ✅ prefix_sum({preset.pos}) = {coveredRanges.map(([lo,hi]) => `A[${lo}..${hi}]`).join(' + ')} = sum({preset.desc})
                </p>
              </div>
            )}
          </div>
        ) : (
          <div className="rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 px-4 py-3 text-center">
            <p className="text-xs text-slate-500 dark:text-slate-400">
              QUERY 路径从右向左跳跃。最多 ⌊log₂ {N}⌋ 步，每步消除一个二进制最低位的 1。
              <br />选择预设场景后点击"开始演示"。
            </p>
          </div>
        )}

        {/* 控制 */}
        <div className="flex items-center justify-between">
          <button onClick={() => setStepIdx(-1)}
            className="px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
            ⏮ 重置
          </button>
          <div className="flex gap-2">
            {stepIdx === -1 ? (
              <button onClick={() => setStepIdx(0)}
                className="px-4 py-1.5 text-xs rounded-lg bg-violet-600 hover:bg-violet-700 text-white font-medium transition-colors">
                ▶ 开始演示
              </button>
            ) : (
              <>
                <button onClick={() => setStepIdx(s => Math.max(0, s - 1))} disabled={stepIdx === 0}
                  className="px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
                  ← 上一步
                </button>
                <button onClick={() => setStepIdx(s => Math.min(totalSteps - 1, s + 1))} disabled={stepIdx === totalSteps - 1}
                  className="px-3 py-1.5 text-xs rounded-lg bg-violet-600 hover:bg-violet-700 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
                  下一步 →
                </button>
              </>
            )}
          </div>
        </div>

        {/* 方向对比小结 */}
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-xl border border-teal-200 dark:border-teal-800 bg-teal-50 dark:bg-teal-900/20 px-3 py-2 text-center">
            <p className="text-[10px] font-semibold text-teal-700 dark:text-teal-400 mb-0.5">UPDATE</p>
            <p className="text-[10px] font-mono text-teal-600 dark:text-teal-300">i += lowbit(i) ↗ 向右</p>
          </div>
          <div className="rounded-xl border border-violet-200 dark:border-violet-800 bg-violet-50 dark:bg-violet-900/20 px-3 py-2 text-center">
            <p className="text-[10px] font-semibold text-violet-700 dark:text-violet-400 mb-0.5">QUERY</p>
            <p className="text-[10px] font-mono text-violet-600 dark:text-violet-300">i -= lowbit(i) ↙ 向左</p>
          </div>
        </div>
      </div>
    </div>
  )
}
