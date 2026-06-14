'use client'

import { useState, useEffect, useCallback } from 'react'

// ─── 类型 ───────────────────────────────────────────────────
interface MatchStep {
  i: number      // 文本指针
  q: number      // 模式指针（已匹配字符数）
  action: 'match' | 'mismatch' | 'jump' | 'found'
  desc: string
  matchAt?: number  // 找到匹配时的起始位置
}

// ─── 工具 ───────────────────────────────────────────────────
function computePi(P: string): number[] {
  const m = P.length
  const pi = new Array<number>(m).fill(0)
  let k = 0
  for (let i = 1; i < m; i++) {
    while (k > 0 && P[k] !== P[i]) k = pi[k - 1]
    if (P[k] === P[i]) k++
    pi[i] = k
  }
  return pi
}

function buildMatchSteps(T: string, P: string): MatchStep[] {
  const pi = computePi(P)
  const steps: MatchStep[] = []
  let q = 0
  for (let i = 0; i < T.length; i++) {
    while (q > 0 && T[i] !== P[q]) {
      steps.push({ i, q, action: 'jump', desc: `T[${i}]='${T[i]}' ≠ P[${q}]='${P[q]}'，q: ${q} → π[${q - 1}]=${pi[q - 1]}` })
      q = pi[q - 1]
    }
    if (T[i] === P[q]) {
      steps.push({ i, q, action: 'match', desc: `T[${i}]='${T[i]}' = P[${q}]='${P[q]}'，匹配，q → ${q + 1}` })
      q++
    } else {
      steps.push({ i, q, action: 'mismatch', desc: `T[${i}]='${T[i]}' ≠ P[${q}]='${P[q]}'，q 保持 0，i 前进` })
    }
    if (q === P.length) {
      const at = i - P.length + 1
      steps.push({ i, q, action: 'found', desc: `✓ 完整匹配！位置 ${at}～${i}，q: ${q} → π[${q - 1}]=${pi[q - 1]} 继续`, matchAt: at })
      q = pi[q - 1]
    }
  }
  return steps
}

const PRESETS = [
  { T: 'AABABCABCAB', P: 'ABCAB' },
  { T: 'AAABABAA', P: 'ABAA' },
  { T: 'ABCABCABCABC', P: 'ABCABC' },
  { T: 'MISSISSIPPI', P: 'ISSI' },
]

// ─── 主组件 ─────────────────────────────────────────────────
export default function KMPMatcherPointerJump() {
  const [textInput, setTextInput] = useState('AABABCABCAB')
  const [patInput, setPatInput]   = useState('ABCAB')
  const [steps, setSteps] = useState<MatchStep[]>([])
  const [pi, setPi] = useState<number[]>([])
  const [cur, setCur] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(700)
  const [foundPositions, setFoundPositions] = useState<number[]>([])

  const build = useCallback((T: string, P: string) => {
    if (!T || !P || P.length > T.length) return
    const s = buildMatchSteps(T, P)
    setSteps(s)
    setPi(computePi(P))
    setCur(0)
    setPlaying(false)
    const found: number[] = []
    s.forEach(st => { if (st.matchAt !== undefined) found.push(st.matchAt) })
    setFoundPositions(found)
  }, [])

  useEffect(() => { build('AABABCABCAB', 'ABCAB') }, [build])

  useEffect(() => {
    if (!playing) return
    if (cur >= steps.length - 1) { setPlaying(false); return }
    const t = setTimeout(() => setCur(c => c + 1), speed)
    return () => clearTimeout(t)
  }, [playing, cur, steps.length, speed])

  const step = steps[cur]
  const P = patInput
  const T = textInput

  // 已完成的匹配区域（步骤走到这里之前已找到的）
  const completedBefore = steps.slice(0, cur + 1)
    .filter(s => s.matchAt !== undefined)
    .map(s => s.matchAt as number)

  const applyPreset = (t: string, p: string) => {
    setTextInput(t); setPatInput(p); build(t, p)
  }

  const handleApply = () => build(textInput.toUpperCase(), patInput.toUpperCase())

  const actionBg = {
    match:    'bg-emerald-500 text-white',
    mismatch: 'bg-rose-500 text-white',
    jump:     'bg-amber-400 text-amber-900',
    found:    'bg-violet-600 text-white',
  }
  const actionIcon = { match: '✓', mismatch: '✗', jump: '↩', found: '★' }

  return (
    <div className="my-8 rounded-2xl border border-violet-200 dark:border-violet-800 bg-white dark:bg-gray-900 shadow-lg overflow-hidden">
      {/* 标题栏 */}
      <div className="bg-gradient-to-r from-violet-600 to-purple-700 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-white/20 flex items-center justify-center text-white text-sm font-bold">KMP</div>
          <div>
            <h3 className="text-white font-bold text-base">KMP 匹配过程可视化</h3>
            <p className="text-violet-200 text-xs mt-0.5">文本指针只进不退，模式指针按 π 跳跃</p>
          </div>
        </div>
        {foundPositions.length > 0 && (
          <div className="bg-white/20 rounded-full px-3 py-1 text-white text-xs font-mono">
            共 {foundPositions.length} 处匹配
          </div>
        )}
      </div>

      <div className="p-6 space-y-5">
        {/* 预设 + 输入 */}
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {PRESETS.map((p, idx) => (
              <button
                key={idx}
                onClick={() => applyPreset(p.T, p.P)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all border ${
                  T === p.T && P === p.P
                    ? 'border-violet-500 bg-violet-100 dark:bg-violet-900 text-violet-700 dark:text-violet-300'
                    : 'border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-violet-400'
                }`}
              >T:{p.T.slice(0, 10)}… P:{p.P}</button>
            ))}
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1 block">文本串 T</label>
              <input className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 px-3 py-2 text-sm font-mono uppercase focus:outline-none focus:ring-2 focus:ring-violet-500"
                value={textInput} onChange={e => setTextInput(e.target.value.toUpperCase())} maxLength={24} />
            </div>
            <div>
              <label className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1 block">模式串 P</label>
              <div className="flex gap-2">
                <input className="flex-1 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 px-3 py-2 text-sm font-mono uppercase focus:outline-none focus:ring-2 focus:ring-violet-500"
                  value={patInput} onChange={e => setPatInput(e.target.value.toUpperCase())} maxLength={12} />
                <button onClick={handleApply} className="px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-700 text-white text-sm font-medium transition-colors">确定</button>
              </div>
            </div>
          </div>
        </div>

        {/* 可视化主区域 */}
        {steps.length > 0 && step && (
          <div className="bg-gray-50 dark:bg-gray-800/60 rounded-xl p-5 space-y-4 overflow-x-auto">
            {/* 文本串 */}
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400 font-medium mb-2">文本 T</div>
              <div className="flex gap-1.5 min-w-max">
                {[...T].map((ch, idx) => {
                  const isI = step.i === idx
                  const isInMatch = completedBefore.some(at => idx >= at && idx < at + P.length)
                  const isCurrentWindow = step.q > 0 && idx >= step.i - step.q + 1 && idx <= step.i
                  const isBadChar = step.action === 'mismatch' && idx === step.i
                  return (
                    <div key={idx} className="flex flex-col items-center gap-1">
                      <div className="text-[10px] text-gray-400 font-mono">{idx}</div>
                      <div className={`w-9 h-9 flex items-center justify-center rounded-lg font-mono font-bold text-sm border transition-all duration-200 ${
                        isI && step.action === 'found'
                          ? 'bg-violet-600 text-white border-violet-500 scale-110 shadow-lg shadow-violet-400/40'
                          : isI && step.action === 'match'
                          ? 'bg-emerald-500 text-white border-emerald-400 scale-105'
                          : isBadChar
                          ? 'bg-rose-500 text-white border-rose-400 scale-105'
                          : isI
                          ? 'bg-amber-400 text-amber-900 border-amber-300 scale-105'
                          : isInMatch
                          ? 'bg-violet-100 dark:bg-violet-900/50 text-violet-700 dark:text-violet-300 border-violet-300 dark:border-violet-600'
                          : isCurrentWindow
                          ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border-emerald-300 dark:border-emerald-700'
                          : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 border-gray-200 dark:border-gray-600'
                      }`}>{ch}</div>
                      <div className="text-[10px] font-bold h-3">
                        {isI ? <span className="text-amber-500">↑i</span> : null}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* 模式串（对齐到当前窗口位置） */}
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400 font-medium mb-2">模式 P（当前对齐）</div>
              <div className="flex gap-1.5 min-w-max">
                {/* 偏移对齐 */}
                {(() => {
                  const offset = step.i - step.q + (step.action === 'jump' ? 0 : 0)
                  const alignOffset = Math.max(0, step.i - step.q)
                  return (
                    <>
                      {Array.from({ length: alignOffset }).map((_, k) => (
                        <div key={`sp${k}`} className="w-9" />
                      ))}
                      {[...P].map((ch, j) => {
                        const isQ = j === step.q && step.action !== 'found'
                        const isMatched = j < step.q
                        return (
                          <div key={j} className="flex flex-col items-center gap-1">
                            <div className="text-[10px] text-gray-400 font-mono">{j}</div>
                            <div className={`w-9 h-9 flex items-center justify-center rounded-lg font-mono font-bold text-sm border transition-all duration-200 ${
                              isQ && step.action === 'mismatch'
                                ? 'bg-rose-500 text-white border-rose-400 scale-105'
                                : isQ && step.action === 'jump'
                                ? 'bg-amber-400 text-amber-900 border-amber-300'
                                : isMatched
                                ? 'bg-emerald-200 dark:bg-emerald-900/60 text-emerald-800 dark:text-emerald-300 border-emerald-300 dark:border-emerald-700'
                                : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 border-gray-200 dark:border-gray-600'
                            }`}>{ch}</div>
                            <div className="text-[10px] font-bold h-3">
                              {isQ ? <span className="text-rose-500">↑q={step.q}</span> : null}
                            </div>
                          </div>
                        )
                      })}
                    </>
                  )
                })()}
              </div>
            </div>

            {/* π 数组 */}
            <div className="border-t border-gray-200 dark:border-gray-700 pt-3">
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-2 font-medium">π 数组</div>
              <div className="flex gap-1.5">
                {[...P].map((ch, j) => (
                  <div key={j} className="flex flex-col items-center gap-1">
                    <div className="text-[10px] text-gray-400 font-mono">{ch}</div>
                    <div className="w-9 h-9 flex items-center justify-center rounded-lg border font-mono font-bold text-sm bg-violet-50 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 border-violet-200 dark:border-violet-700">
                      {pi[j]}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* 当前步骤信息 */}
        {step && (
          <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border ${
            step.action === 'found'    ? 'bg-violet-50 dark:bg-violet-900/20 border-violet-200 dark:border-violet-700'
            : step.action === 'match' ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700'
            : step.action === 'jump'  ? 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-700'
            : 'bg-rose-50 dark:bg-rose-900/20 border-rose-200 dark:border-rose-700'
          }`}>
            <span className={`w-7 h-7 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0 ${actionBg[step.action]}`}>
              {actionIcon[step.action]}
            </span>
            <span className="text-sm text-gray-700 dark:text-gray-300">{step.desc}</span>
            <span className="ml-auto text-xs font-mono text-gray-400 dark:text-gray-500">{cur + 1}/{steps.length}</span>
          </div>
        )}

        {/* 控制 */}
        <div className="flex items-center gap-3 flex-wrap">
          <button onClick={() => { setCur(0); setPlaying(false) }}
            className="px-3 py-1.5 rounded-lg text-sm border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">⏮ 重置</button>
          <button onClick={() => setCur(c => Math.max(0, c - 1))} disabled={cur === 0}
            className="px-3 py-1.5 rounded-lg text-sm border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-30 transition-colors">← 前一步</button>
          <button onClick={() => setPlaying(p => !p)}
            className={`px-5 py-1.5 rounded-lg text-sm font-medium transition-colors ${playing ? 'bg-rose-500 hover:bg-rose-600 text-white' : 'bg-violet-600 hover:bg-violet-700 text-white'}`}>
            {playing ? '⏸ 暂停' : '▶ 播放'}
          </button>
          <button onClick={() => setCur(c => Math.min(steps.length - 1, c + 1))} disabled={cur >= steps.length - 1}
            className="px-3 py-1.5 rounded-lg text-sm border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-30 transition-colors">→ 后一步</button>
          <div className="flex items-center gap-2 ml-auto">
            <span className="text-xs text-gray-500 dark:text-gray-400">速度</span>
            <input type="range" min={200} max={1500} step={100} value={1700 - speed}
              onChange={e => setSpeed(1700 - Number(e.target.value))} className="w-24 accent-violet-500" />
          </div>
        </div>

        {/* 找到的位置 */}
        {completedBefore.length > 0 && (
          <div className="flex items-center gap-2 flex-wrap text-sm">
            <span className="text-gray-500 dark:text-gray-400">已找到匹配：</span>
            {completedBefore.map(pos => (
              <span key={pos} className="px-2.5 py-1 bg-violet-100 dark:bg-violet-900/50 text-violet-700 dark:text-violet-300 rounded-full text-xs font-mono font-bold border border-violet-200 dark:border-violet-700">
                位置 {pos}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
