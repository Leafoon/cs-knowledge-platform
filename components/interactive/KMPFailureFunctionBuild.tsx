'use client'

import { useState, useEffect, useCallback } from 'react'

// ─── 类型 ───────────────────────────────────────────────────
interface Step {
  i: number        // 当前外层下标
  k: number        // 当前 k 指针
  pi: number[]     // 当前 π 数组快照
  action: 'compare_match' | 'compare_mismatch' | 'jump' | 'set'
  desc: string
}

// ─── 工具：计算所有步骤 ─────────────────────────────────────
function buildSteps(P: string): Step[] {
  const m = P.length
  const steps: Step[] = []
  const pi = new Array(m).fill(0)
  steps.push({ i: 0, k: 0, pi: [...pi], action: 'set', desc: `初始化：π[0] = 0（单字符无真前缀）` })
  let k = 0
  for (let i = 1; i < m; i++) {
    while (k > 0 && P[k] !== P[i]) {
      steps.push({ i, k, pi: [...pi], action: 'compare_mismatch', desc: `P[${k}]='${P[k]}' ≠ P[${i}]='${P[i]}'，失配！k 回跳到 π[${k - 1}]=${pi[k - 1]}` })
      steps.push({ i, k: pi[k - 1], pi: [...pi], action: 'jump', desc: `k: ${k} → ${pi[k - 1]}（利用 π[${k - 1}]）` })
      k = pi[k - 1]
    }
    if (P[k] === P[i]) {
      steps.push({ i, k, pi: [...pi], action: 'compare_match', desc: `P[${k}]='${P[k]}' = P[${i}]='${P[i]}'，匹配！k+1 → ${k + 1}` })
      k++
    } else {
      steps.push({ i, k, pi: [...pi], action: 'compare_mismatch', desc: `P[${k}]='${P[k]}' ≠ P[${i}]='${P[i]}'，k 保持 0` })
    }
    pi[i] = k
    steps.push({ i, k, pi: [...pi], action: 'set', desc: `π[${i}] = ${k}（P[0..${i}] 的最长真前缀后缀长度）` })
  }
  return steps
}

// ─── 预设示例 ───────────────────────────────────────────────
const PRESETS = [
  { label: 'ABCAB', value: 'ABCAB' },
  { label: 'ABABD', value: 'ABABD' },
  { label: 'ABCABCAB', value: 'ABCABCAB' },
  { label: 'AABAAAB', value: 'AABAAAB' },
  { label: 'AAAA', value: 'AAAA' },
]

// ─── 颜色辅助 ────────────────────────────────────────────────
const actionColors = {
  compare_match:    'bg-emerald-500 text-white shadow-emerald-400/50',
  compare_mismatch: 'bg-rose-500 text-white shadow-rose-400/50',
  jump:             'bg-amber-400 text-amber-900 shadow-amber-400/50',
  set:              'bg-indigo-600 text-white shadow-indigo-400/50',
}
const actionIcons = {
  compare_match:    '✓',
  compare_mismatch: '✗',
  jump:             '↩',
  set:              '=',
}

// ─── 主组件 ─────────────────────────────────────────────────
export default function KMPFailureFunctionBuild() {
  const [input, setInput] = useState('ABCAB')
  const [editVal, setEditVal] = useState('ABCAB')
  const [steps, setSteps] = useState<Step[]>([])
  const [cur, setCur] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(800)

  const applyPattern = useCallback((p: string) => {
    const clean = p.toUpperCase().replace(/[^A-Z]/g, '').slice(0, 16)
    if (clean.length < 2) return
    setInput(clean)
    setEditVal(clean)
    const s = buildSteps(clean)
    setSteps(s)
    setCur(0)
    setPlaying(false)
  }, [])

  useEffect(() => { applyPattern('ABCAB') }, [applyPattern])

  // 自动播放
  useEffect(() => {
    if (!playing) return
    if (cur >= steps.length - 1) { setPlaying(false); return }
    const t = setTimeout(() => setCur(c => c + 1), speed)
    return () => clearTimeout(t)
  }, [playing, cur, steps.length, speed])

  const step = steps[cur]
  const piArray = step?.pi ?? []

  return (
    <div className="my-8 rounded-2xl border border-indigo-200 dark:border-indigo-800 bg-white dark:bg-gray-900 shadow-lg overflow-hidden">
      {/* 标题栏 */}
      <div className="bg-gradient-to-r from-indigo-600 to-violet-600 px-6 py-4 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-white/20 flex items-center justify-center text-white font-bold text-sm">π</div>
        <div>
          <h3 className="text-white font-bold text-base">KMP 失效函数（π 数组）构造步进</h3>
          <p className="text-indigo-200 text-xs mt-0.5">逐步演示双指针法线性构建 π 数组</p>
        </div>
      </div>

      <div className="p-6 space-y-5">
        {/* 输入区 */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <input
              className="flex-1 min-w-0 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 px-3 py-2 text-sm font-mono uppercase focus:outline-none focus:ring-2 focus:ring-indigo-500"
              value={editVal}
              onChange={e => setEditVal(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && applyPattern(editVal)}
              placeholder="输入模式串（A-Z，最多16字符）"
              maxLength={16}
            />
            <button
              onClick={() => applyPattern(editVal)}
              className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium transition-colors"
            >确定</button>
          </div>
          <div className="flex gap-2 flex-wrap">
            {PRESETS.map(p => (
              <button
                key={p.value}
                onClick={() => applyPattern(p.value)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all border ${
                  input === p.value
                    ? 'border-indigo-500 bg-indigo-100 dark:bg-indigo-900 text-indigo-700 dark:text-indigo-300'
                    : 'border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-indigo-400'
                }`}
              >{p.label}</button>
            ))}
          </div>
        </div>

        {/* 字符格展示 */}
        {input.length >= 2 && (
          <div className="bg-gray-50 dark:bg-gray-800/60 rounded-xl p-5 space-y-4">
            {/* 下标行 */}
            <div className="flex gap-2 justify-center">
              {[...input].map((_, idx) => (
                <div key={idx} className="w-10 text-center">
                  <div className="text-[11px] text-gray-400 dark:text-gray-500 font-mono">{idx}</div>
                </div>
              ))}
            </div>

            {/* 模式串字符行 */}
            <div className="flex gap-2 justify-center">
              {[...input].map((ch, idx) => {
                const isI = step?.i === idx
                const isK = step?.k === idx && idx > 0
                const bg = isI && isK
                  ? 'bg-violet-500 text-white ring-2 ring-violet-400'
                  : isI
                  ? 'bg-indigo-500 text-white ring-2 ring-indigo-400'
                  : isK
                  ? 'bg-amber-400 text-amber-900 ring-2 ring-amber-400'
                  : piArray[idx] > 0
                  ? 'bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300'
                  : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                return (
                  <div
                    key={idx}
                    className={`w-10 h-10 flex items-center justify-center rounded-lg font-mono font-bold text-base border transition-all duration-200 shadow-sm ${bg} ${
                      isI || isK ? 'shadow-md scale-110' : 'border-gray-200 dark:border-gray-600'
                    }`}
                  >{ch}</div>
                )
              })}
            </div>

            {/* 指针标签 */}
            <div className="flex gap-2 justify-center">
              {[...input].map((_, idx) => {
                const isI = step?.i === idx
                const isK = step?.k === idx && idx > 0
                return (
                  <div key={idx} className="w-10 text-center text-[11px] font-bold h-4">
                    {isI && isK ? <span className="text-violet-500">i=k</span>
                     : isI ? <span className="text-indigo-500">i</span>
                     : isK ? <span className="text-amber-500">k</span>
                     : null}
                  </div>
                )
              })}
            </div>

            {/* π 数组行 */}
            <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
              <div className="text-xs text-gray-500 dark:text-gray-400 text-center mb-2 font-medium">π 数组（最长真前缀后缀长度）</div>
              <div className="flex gap-2 justify-center">
                {piArray.map((v, idx) => (
                  <div
                    key={idx}
                    className={`w-10 h-10 flex items-center justify-center rounded-lg font-mono font-bold text-base transition-all duration-300 border ${
                      step?.i === idx && step?.action === 'set'
                        ? 'bg-indigo-600 text-white border-indigo-500 scale-110 shadow-lg shadow-indigo-400/50'
                        : idx < (step?.i ?? 0)
                        ? 'bg-indigo-100 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300 border-indigo-200 dark:border-indigo-700'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-400 dark:text-gray-500 border-gray-200 dark:border-gray-600'
                    }`}
                  >{idx <= (step?.i ?? 0) ? v : '?'}</div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* 当前步骤说明 */}
        {step && (
          <div className={`flex items-center gap-3 px-4 py-3 rounded-xl shadow-sm ${
            step.action === 'compare_match' ? 'bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800'
            : step.action === 'compare_mismatch' ? 'bg-rose-50 dark:bg-rose-900/20 border border-rose-200 dark:border-rose-800'
            : step.action === 'jump' ? 'bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800'
            : 'bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800'
          }`}>
            <span className={`w-7 h-7 rounded-full flex items-center justify-center text-sm font-bold shadow-sm ${actionColors[step.action]}`}>
              {actionIcons[step.action]}
            </span>
            <span className="text-sm text-gray-700 dark:text-gray-300 font-medium">{step.desc}</span>
            <span className="ml-auto text-xs text-gray-400 dark:text-gray-500 font-mono">{cur + 1}/{steps.length}</span>
          </div>
        )}

        {/* 控制栏 */}
        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={() => { setCur(0); setPlaying(false) }}
            className="px-3 py-1.5 rounded-lg text-sm border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >⏮ 重置</button>
          <button
            onClick={() => setCur(c => Math.max(0, c - 1))}
            disabled={cur === 0}
            className="px-3 py-1.5 rounded-lg text-sm border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors disabled:opacity-30"
          >← 上一步</button>
          <button
            onClick={() => setPlaying(p => !p)}
            className={`px-5 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              playing ? 'bg-rose-500 hover:bg-rose-600 text-white' : 'bg-indigo-600 hover:bg-indigo-700 text-white'
            }`}
          >{playing ? '⏸ 暂停' : '▶ 播放'}</button>
          <button
            onClick={() => setCur(c => Math.min(steps.length - 1, c + 1))}
            disabled={cur >= steps.length - 1}
            className="px-3 py-1.5 rounded-lg text-sm border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors disabled:opacity-30"
          >→ 下一步</button>
          <button
            onClick={() => { setCur(steps.length - 1); setPlaying(false) }}
            className="px-3 py-1.5 rounded-lg text-sm border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >⏭ 完成</button>
          <div className="flex items-center gap-2 ml-auto">
            <span className="text-xs text-gray-500 dark:text-gray-400">速度</span>
            <input
              type="range" min={200} max={1500} step={100}
              value={1700 - speed}
              onChange={e => setSpeed(1700 - Number(e.target.value))}
              className="w-24 accent-indigo-500"
            />
          </div>
        </div>

        {/* 图例 */}
        <div className="flex flex-wrap gap-4 text-xs pt-1 border-t border-gray-100 dark:border-gray-800">
          {[
            { color: 'bg-indigo-500', label: 'i 指针（外层）' },
            { color: 'bg-amber-400', label: 'k 指针（前缀长）' },
            { color: 'bg-emerald-500', label: '匹配' },
            { color: 'bg-rose-500', label: '失配 / 回跳' },
          ].map(l => (
            <div key={l.label} className="flex items-center gap-1.5 text-gray-500 dark:text-gray-400">
              <span className={`w-3 h-3 rounded-sm ${l.color}`}></span>{l.label}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
