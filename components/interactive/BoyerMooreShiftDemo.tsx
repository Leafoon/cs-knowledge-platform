'use client'

import { useState } from 'react'

// ─── 坏字符表 ───────────────────────────────────────────────
function buildBadChar(P: string): Record<string, number> {
  const table: Record<string, number> = {}
  for (let i = 0; i < P.length; i++) table[P[i]] = i
  return table
}

interface ShiftEvent {
  s: number           // 当前对齐起点
  j: number           // 失配/匹配最终位置（从右扫描到的）
  badChar: string     // 坏字符（失配时）
  badCharPos: number  // 坏字符在模式中的位置（-1 表示不存在）
  shiftBC: number     // 坏字符规则给出的移动量
  isMatch: boolean    // 完整匹配？
  windowText: string  // 当前文本窗口
}

function buildEvents(T: string, P: string): ShiftEvent[] {
  const bc = buildBadChar(P)
  const n = T.length, m = P.length
  const events: ShiftEvent[] = []
  let s = 0
  while (s <= n - m) {
    let j = m - 1
    while (j >= 0 && P[j] === T[s + j]) j--
    if (j < 0) {
      events.push({ s, j: -1, badChar: '', badCharPos: -1, shiftBC: 0, isMatch: true, windowText: T.slice(s, s + m) })
      s += (s + m < n) ? m - (bc[T[s + m]] ?? -1) : 1
    } else {
      const bc_ch = T[s + j]
      const bc_pos = bc[bc_ch] ?? -1
      const shift = Math.max(1, j - bc_pos)
      events.push({ s, j, badChar: bc_ch, badCharPos: bc_pos, shiftBC: shift, isMatch: false, windowText: T.slice(s, s + m) })
      s += shift
    }
  }
  return events
}

const PRESETS = [
  { T: 'HERE IS A SIMPLE EXAMPLE', P: 'EXAMPLE' },
  { T: 'ABCABCABCABC', P: 'ABCABC' },
  { T: 'AABABCABCAB', P: 'ABC' },
]

// ─── 主组件 ─────────────────────────────────────────────────
export default function BoyerMooreShiftDemo() {
  const [T, setT] = useState('HERE IS A SIMPLE EXAMPLE')
  const [P, setP] = useState('EXAMPLE')
  const [TEdit, setTEdit] = useState('HERE IS A SIMPLE EXAMPLE')
  const [PEdit, setPEdit] = useState('EXAMPLE')
  const [events, setEvents] = useState<ShiftEvent[]>(() => buildEvents('HERE IS A SIMPLE EXAMPLE', 'EXAMPLE'))
  const [cur, setCur] = useState(0)

  const build = (t: string, p: string) => {
    const tc = t.toUpperCase().slice(0, 30)
    const pc = p.toUpperCase().replace(/\s/g,'').slice(0, 12)
    if (!tc || !pc || pc.length > tc.length) return
    setT(tc); setP(pc)
    setEvents(buildEvents(tc, pc))
    setCur(0)
  }

  const ev = events[cur]
  const bcTable = buildBadChar(P)

  return (
    <div className="my-8 rounded-2xl border border-orange-200 dark:border-orange-800 bg-white dark:bg-gray-900 shadow-lg overflow-hidden">
      {/* 标题 */}
      <div className="px-6 py-4 bg-gradient-to-r from-orange-500 to-amber-500 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-white/20 flex items-center justify-center text-white font-bold text-xs">BM</div>
        <div>
          <h3 className="text-white font-bold text-base">Boyer-Moore 坏字符规则演示</h3>
          <p className="text-orange-100 text-xs mt-0.5">从右到左比较，利用坏字符表大步右移</p>
        </div>
      </div>

      <div className="p-6 space-y-5">
        {/* 预设 + 输入 */}
        <div className="flex flex-wrap gap-2 mb-1">
          {PRESETS.map((pr, i) => (
            <button key={i} onClick={() => { setTEdit(pr.T); setPEdit(pr.P); build(pr.T, pr.P) }}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition-all ${
                T === pr.T && P === pr.P
                  ? 'border-orange-500 bg-orange-100 dark:bg-orange-900 text-orange-700 dark:text-orange-300'
                  : 'border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-orange-400'
              }`}>T:…{pr.T.slice(-8)} P:{pr.P}</button>
          ))}
        </div>
        <div className="flex flex-wrap gap-3">
          <input className="flex-1 min-w-[160px] rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 px-3 py-1.5 text-sm font-mono uppercase focus:outline-none focus:ring-2 focus:ring-orange-500"
            placeholder="文本 T" value={TEdit} onChange={e => setTEdit(e.target.value)} maxLength={30} />
          <input className="w-28 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 px-3 py-1.5 text-sm font-mono uppercase focus:outline-none focus:ring-2 focus:ring-orange-500"
            placeholder="模式 P" value={PEdit} onChange={e => setPEdit(e.target.value)} maxLength={12} />
          <button onClick={() => build(TEdit, PEdit)}
            className="px-4 py-1.5 rounded-lg bg-orange-500 hover:bg-orange-600 text-white text-sm font-medium transition-colors">确定</button>
        </div>

        {ev && (
          <div className="space-y-4">
            {/* 对齐可视化 */}
            <div className="bg-gray-50 dark:bg-gray-800/60 rounded-xl p-5 overflow-x-auto space-y-3">
              <div className="text-xs text-gray-500 dark:text-gray-400 font-medium">
                第 {cur + 1} 步 / 共 {events.length} 步：对齐位置 s = {ev.s}
              </div>

              {/* 文本行 */}
              <div className="flex gap-1 min-w-max">
                {[...T].map((ch, idx) => {
                  const inWindow = idx >= ev.s && idx < ev.s + P.length
                  const isBadChar = !ev.isMatch && idx === ev.s + ev.j
                  const isMatched = !ev.isMatch && inWindow && idx > ev.s + ev.j
                  return (
                    <div key={idx} className="flex flex-col items-center">
                      <div className="text-[9px] text-gray-400 font-mono">{idx}</div>
                      <div className={`w-8 h-8 flex items-center justify-center rounded font-mono font-bold text-sm border transition-all ${
                        isBadChar
                          ? 'bg-rose-500 text-white border-rose-400 shadow-md shadow-rose-400/40 scale-110'
                          : isMatched
                          ? 'bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300 border-emerald-300 dark:border-emerald-600'
                          : inWindow
                          ? ev.isMatch
                            ? 'bg-emerald-500 text-white border-emerald-400'
                            : 'bg-orange-100 dark:bg-orange-900/40 text-orange-700 dark:text-orange-300 border-orange-200 dark:border-orange-700'
                          : 'bg-white dark:bg-gray-700 text-gray-600 dark:text-gray-400 border-gray-200 dark:border-gray-600'
                      }`}>{ch}</div>
                    </div>
                  )
                })}
              </div>

              {/* 模式行 */}
              <div className="flex gap-1 min-w-max" style={{ paddingLeft: `${ev.s * 34}px` }}>
                {[...P].map((ch, j) => {
                  const isCompared = !ev.isMatch && j > ev.j  // 已比较（从右起）
                  const isFail = !ev.isMatch && j === ev.j
                  return (
                    <div key={j} className="flex flex-col items-center">
                      <div className="text-[9px] text-gray-400 font-mono">{j}</div>
                      <div className={`w-8 h-8 flex items-center justify-center rounded font-mono font-bold text-sm border ${
                        isFail
                          ? 'bg-rose-200 dark:bg-rose-900/60 text-rose-700 dark:text-rose-300 border-rose-300 dark:border-rose-600'
                          : isCompared
                          ? 'bg-emerald-200 dark:bg-emerald-900/60 text-emerald-700 dark:text-emerald-300 border-emerald-300 dark:border-emerald-700'
                          : ev.isMatch
                          ? 'bg-emerald-500 text-white border-emerald-400'
                          : 'bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 border-gray-200 dark:border-gray-600'
                      }`}>{ch}</div>
                    </div>
                  )
                })}
              </div>

              {/* 比较方向提示 */}
              {!ev.isMatch && (
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                  <span className="text-orange-500 font-bold">←</span>
                  <span>从右向左比较，在位置 j={ev.j} 发现坏字符 '<span className="text-rose-500 font-mono font-bold">{ev.badChar}</span>'</span>
                </div>
              )}
              {ev.isMatch && (
                <div className="flex items-center gap-2 text-xs text-emerald-600 dark:text-emerald-400 font-medium">
                  <span>🎉 完整匹配！位置 {ev.s} ～ {ev.s + P.length - 1}</span>
                </div>
              )}
            </div>

            {/* 坏字符信息 */}
            {!ev.isMatch && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="rounded-xl border border-rose-200 dark:border-rose-700 p-4 bg-rose-50 dark:bg-rose-900/20">
                  <div className="text-xs text-rose-600 dark:text-rose-400 font-medium mb-1">坏字符</div>
                  <div className="text-2xl font-mono font-bold text-rose-600 dark:text-rose-400">'{ev.badChar}'</div>
                  <div className="text-xs text-rose-500/70 dark:text-rose-400/70 mt-1">T[{ev.s + ev.j}]，在位置 j={ev.j} 处失配</div>
                </div>
                <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-800/40">
                  <div className="text-xs text-gray-500 dark:text-gray-400 font-medium mb-1">坏字符在 P 中最后出现位置</div>
                  <div className="text-2xl font-mono font-bold text-gray-700 dark:text-gray-300">
                    {ev.badCharPos === -1 ? '不存在' : ev.badCharPos}
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    badChar['{ev.badChar}'] = {ev.badCharPos}
                  </div>
                </div>
                <div className="rounded-xl border border-amber-200 dark:border-amber-700 p-4 bg-amber-50 dark:bg-amber-900/20">
                  <div className="text-xs text-amber-600 dark:text-amber-400 font-medium mb-1">右移量</div>
                  <div className="text-2xl font-mono font-bold text-amber-600 dark:text-amber-400">+{ev.shiftBC}</div>
                  <div className="text-xs text-amber-500/70 dark:text-amber-400/70 mt-1">
                    max(1, j − badChar['{ev.badChar}']) = max(1, {ev.j} − ({ev.badCharPos})) = {ev.shiftBC}
                  </div>
                </div>
              </div>
            )}

            {/* 坏字符表 */}
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400 font-medium mb-2">坏字符表 badChar[c]（模式 P 中每个字符最后出现的下标）</div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(bcTable).map(([ch, pos]) => (
                  <div key={ch} className={`px-3 py-2 rounded-lg border text-xs font-mono text-center transition-all ${
                    ch === ev.badChar && !ev.isMatch
                      ? 'border-rose-400 bg-rose-100 dark:bg-rose-900/50 text-rose-700 dark:text-rose-300 scale-105 shadow-md shadow-rose-400/30'
                      : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-400'
                  }`}>
                    <div className="font-bold text-base">{ch}</div>
                    <div className="text-gray-400 dark:text-gray-500">{pos}</div>
                  </div>
                ))}
                <div className="px-3 py-2 rounded-lg border border-dashed border-gray-200 dark:border-gray-700 text-xs font-mono text-center text-gray-400 dark:text-gray-500">
                  <div className="font-bold text-base">其他</div>
                  <div>−1</div>
                </div>
              </div>
            </div>

            {/* 所有步骤时间线 */}
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400 font-medium mb-2">匹配历程时间线</div>
              <div className="flex gap-1 flex-wrap">
                {events.map((e, idx) => (
                  <button key={idx} onClick={() => setCur(idx)}
                    className={`px-2 py-1 rounded text-xs font-mono transition-all border ${
                      idx === cur
                        ? 'bg-orange-500 text-white border-orange-400 scale-105 shadow-md'
                        : e.isMatch
                        ? 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-700'
                        : 'bg-white dark:bg-gray-800 text-gray-500 dark:text-gray-400 border-gray-200 dark:border-gray-600 hover:border-orange-400'
                    }`}
                  >s={e.s}</button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* 导航 */}
        <div className="flex items-center gap-3 pt-1 border-t border-gray-100 dark:border-gray-800">
          <button onClick={() => setCur(0)} className="px-3 py-1.5 rounded-lg text-sm border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800">⏮ 重置</button>
          <button onClick={() => setCur(c => Math.max(0, c - 1))} disabled={cur === 0}
            className="px-3 py-1.5 rounded-lg text-sm border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-30">← 上一步</button>
          <button onClick={() => setCur(c => Math.min(events.length - 1, c + 1))} disabled={cur >= events.length - 1}
            className="px-3 py-1.5 rounded-lg text-sm font-medium bg-orange-500 hover:bg-orange-600 text-white transition-colors disabled:opacity-40">下一步 →</button>
          <span className="text-xs text-gray-400 dark:text-gray-500 font-mono ml-auto">{cur + 1} / {events.length} 步</span>
        </div>
      </div>
    </div>
  )
}
