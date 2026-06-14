'use client'

import { useState, useEffect, useCallback } from 'react'

// ─── 工具 ───────────────────────────────────────────────────
const D = 31
const Q = 1_000_003   // 质数，展示用（比 10^9 小以便显示）

function charVal(c: string) { return c.charCodeAt(0) - 64 }  // A=1,B=2,...

function modPow(base: number, exp: number, mod: number) {
  let r = 1; base %= mod
  for (; exp > 0; exp >>= 1) {
    if (exp & 1) r = r * base % mod
    base = base * base % mod
  }
  return r
}

interface WindowState {
  s: number          // 当前串起点
  hash: number       // 当前窗口哈希
  matched: boolean   // 哈希是否与模式匹配
  charMatch: boolean // 字符实际匹配
  pHash: number      // 模式哈希
  removed: number    // 滑出字符值
  added: number      // 滑入字符值
  formula: string    // 公式展示
}

function buildWindows(T: string, P: string): WindowState[] {
  const n = T.length, m = P.length
  const windows: WindowState[] = []
  const h = modPow(D, m - 1, Q)

  let pHash = 0, tHash = 0
  for (let i = 0; i < m; i++) {
    pHash = (pHash * D + charVal(P[i])) % Q
    tHash = (tHash * D + charVal(T[i])) % Q
  }

  for (let s = 0; s <= n - m; s++) {
    const matched = pHash === tHash
    const charMatch = T.slice(s, s + m) === P
    const removed = s > 0 ? charVal(T[s - 1]) : 0
    const added   = s + m <= n ? charVal(T[s + m - 1]) : 0
    windows.push({
      s, hash: tHash, matched, charMatch,
      pHash, removed, added,
      formula: s === 0
        ? `初始：hash("${T.slice(s, s + m)}") = ${tHash}`
        : `hash = (prev − '${T[s-1]}' × h) × ${D} + '${T[s+m-1] ?? ''}' = ${tHash}`
    })
    if (s < n - m) {
      tHash = ((tHash - charVal(T[s]) * h % Q + Q) * D + charVal(T[s + m])) % Q
    }
  }
  return windows
}

const PRESETS = [
  { T: 'ABCABABCAB', P: 'ABC' },
  { T: 'GEEKSFORGEEKS', P: 'GEEK' },
  { T: 'AAABAAAB', P: 'AAAB' },
]

// ─── 主组件 ─────────────────────────────────────────────────
export default function RabinKarpRollingHash() {
  const [T, setT] = useState('ABCABABCAB')
  const [P, setP] = useState('ABC')
  const [TEdit, setTEdit] = useState('ABCABABCAB')
  const [PEdit, setPEdit] = useState('ABC')
  const [windows, setWindows] = useState<WindowState[]>([])
  const [cur, setCur] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(900)

  const build = useCallback((t: string, p: string) => {
    const tc = t.toUpperCase().replace(/[^A-Z]/g, '').slice(0, 18)
    const pc = p.toUpperCase().replace(/[^A-Z]/g, '').slice(0, 8)
    if (!tc || !pc || pc.length > tc.length) return
    setT(tc); setP(pc)
    setWindows(buildWindows(tc, pc))
    setCur(0); setPlaying(false)
  }, [])

  useEffect(() => { build('ABCABABCAB', 'ABC') }, [build])

  useEffect(() => {
    if (!playing) return
    if (cur >= windows.length - 1) { setPlaying(false); return }
    const t = setTimeout(() => setCur(c => c + 1), speed)
    return () => clearTimeout(t)
  }, [playing, cur, windows.length, speed])

  const w = windows[cur]
  const pHash = windows[0]?.pHash ?? 0

  return (
    <div className="my-8 rounded-2xl border border-teal-200 dark:border-teal-800 bg-white dark:bg-gray-900 shadow-lg overflow-hidden">
      {/* 标题 */}
      <div className="px-6 py-4 bg-gradient-to-r from-teal-600 to-cyan-600 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-white/20 flex items-center justify-center text-white font-bold text-xs">RK</div>
        <div>
          <h3 className="text-white font-bold text-base">Rabin-Karp 滚动哈希可视化</h3>
          <p className="text-teal-100 text-xs mt-0.5">窗口滑动 → O(1) 更新哈希 → 匹配判断</p>
        </div>
        <div className="ml-auto text-right">
          <div className="text-teal-100 text-xs">基数 d = {D}  模 q = {Q.toLocaleString()}</div>
        </div>
      </div>

      <div className="p-6 space-y-5">
        {/* 预设 + 输入 */}
        <div className="flex flex-wrap gap-2 items-center">
          {PRESETS.map((pr, i) => (
            <button key={i} onClick={() => { setTEdit(pr.T); setPEdit(pr.P); build(pr.T, pr.P) }}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition-all ${
                T === pr.T && P === pr.P
                  ? 'border-teal-500 bg-teal-100 dark:bg-teal-900 text-teal-700 dark:text-teal-300'
                  : 'border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-teal-400'
              }`}>T:{pr.T.slice(0,8)}… P:{pr.P}</button>
          ))}
          <div className="flex gap-2 ml-auto">
            <input className="rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 px-3 py-1.5 text-sm font-mono uppercase w-40 focus:outline-none focus:ring-2 focus:ring-teal-500"
              placeholder="文本 T" value={TEdit} onChange={e => setTEdit(e.target.value)} maxLength={18} />
            <input className="rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 px-3 py-1.5 text-sm font-mono uppercase w-24 focus:outline-none focus:ring-2 focus:ring-teal-500"
              placeholder="模式 P" value={PEdit} onChange={e => setPEdit(e.target.value)} maxLength={8} />
            <button onClick={() => build(TEdit, PEdit)}
              className="px-4 py-1.5 rounded-lg bg-teal-600 hover:bg-teal-700 text-white text-sm font-medium transition-colors">确定</button>
          </div>
        </div>

        {/* 主可视化 */}
        {w && (
          <div className="space-y-4">
            {/* 文本串 + 滑动窗口 */}
            <div className="bg-gray-50 dark:bg-gray-800/60 rounded-xl p-5 overflow-x-auto">
              <div className="text-xs text-gray-500 dark:text-gray-400 font-medium mb-3">文本 T（蓝框为当前窗口）</div>
              <div className="flex gap-1.5 min-w-max mb-1">
                {[...T].map((ch, idx) => {
                  const inWin  = idx >= w.s && idx < w.s + P.length
                  const isNew  = idx === w.s + P.length - 1 && cur > 0
                  const isGone = idx === w.s - 1 && cur > 0
                  return (
                    <div key={idx} className="flex flex-col items-center gap-1">
                      <div className="text-[10px] text-gray-400 font-mono">{idx}</div>
                      <div className={`w-9 h-9 flex items-center justify-center rounded-lg font-mono font-bold text-sm border-2 transition-all duration-300 ${
                        inWin
                          ? w.charMatch
                            ? 'bg-teal-500 text-white border-teal-400 shadow-md shadow-teal-400/30'
                            : w.matched && !w.charMatch
                            ? 'bg-amber-400 text-amber-900 border-amber-400 shadow-md shadow-amber-400/30'
                            : 'bg-teal-100 dark:bg-teal-900/50 text-teal-700 dark:text-teal-300 border-teal-300 dark:border-teal-600'
                          : isGone
                          ? 'bg-red-100 dark:bg-red-900/30 text-red-500 border-red-200 dark:border-red-700 opacity-60'
                          : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 border-gray-200 dark:border-gray-600'
                      }`}>{ch}</div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* 哈希对比面板 */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* 模式哈希 */}
              <div className="rounded-xl border border-teal-200 dark:border-teal-700 p-4 bg-teal-50 dark:bg-teal-900/20">
                <div className="text-xs font-medium text-teal-600 dark:text-teal-400 mb-2">模式串 P 哈希（固定）</div>
                <div className="font-mono text-lg font-bold text-teal-700 dark:text-teal-300">{pHash}</div>
                <div className="text-xs text-teal-600/70 dark:text-teal-400/70 mt-1 font-mono">hash("{P}")</div>
              </div>

              {/* 当前窗口哈希 */}
              <div className={`rounded-xl border p-4 transition-all ${
                w.charMatch
                  ? 'border-emerald-300 dark:border-emerald-600 bg-emerald-50 dark:bg-emerald-900/20'
                  : w.matched && !w.charMatch
                  ? 'border-amber-300 dark:border-amber-600 bg-amber-50 dark:bg-amber-900/20'
                  : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/40'
              }`}>
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
                  窗口 T[{w.s}..{w.s + P.length - 1}] 哈希
                  {w.charMatch ? ' 🎉 真实匹配！' : w.matched && !w.charMatch ? ' ⚠️ 伪匹配（哈希碰撞）' : ''}
                </div>
                <div className={`font-mono text-lg font-bold ${
                  w.charMatch ? 'text-emerald-600 dark:text-emerald-400'
                  : w.matched && !w.charMatch ? 'text-amber-600 dark:text-amber-400'
                  : 'text-gray-700 dark:text-gray-300'
                }`}>{w.hash}</div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1 font-mono">hash("{T.slice(w.s, w.s + P.length)}")</div>
              </div>
            </div>

            {/* 滚动公式 */}
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 px-4 py-3 bg-gray-50 dark:bg-gray-800/40">
              <div className="text-xs text-gray-500 dark:text-gray-400 font-medium mb-1">滚动哈希更新公式</div>
              <div className="font-mono text-sm text-gray-700 dark:text-gray-300">{w.formula}</div>
              {cur > 0 && (
                <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                  滑出：<span className="font-mono text-rose-500">'{T[w.s - 1]}'</span> 
                  {' '}→ 滑入：<span className="font-mono text-teal-500">'{T[w.s + P.length - 1] ?? ''}'</span>
                  {' '}（仅 1 次加减乘运算！）
                </div>
              )}
            </div>

            {/* 所有窗口哈希值概览 */}
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400 font-medium mb-2">所有窗口哈希一览</div>
              <div className="flex gap-1.5 flex-wrap">
                {windows.map((ww, idx) => (
                  <div
                    key={idx}
                    onClick={() => { setCur(idx); setPlaying(false) }}
                    className={`px-3 py-2 rounded-lg border text-xs cursor-pointer transition-all ${
                      idx === cur
                        ? 'border-teal-500 bg-teal-100 dark:bg-teal-900 text-teal-700 dark:text-teal-300 scale-105 shadow-md'
                        : ww.charMatch
                        ? 'border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400'
                        : ww.matched && !ww.charMatch
                        ? 'border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-400'
                        : 'border-gray-100 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-500 dark:text-gray-400 hover:border-teal-400'
                    }`}
                  >
                    <div className="font-mono font-bold">[{ww.s}]</div>
                    <div className="font-mono text-[10px] mt-0.5">{ww.hash}</div>
                  </div>
                ))}
              </div>
              <div className="flex gap-4 mt-2 text-xs text-gray-500 dark:text-gray-400">
                <div className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-emerald-400 inline-block"></span> 真实匹配</div>
                <div className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-amber-400 inline-block"></span> 伪匹配（哈希碰撞）</div>
                <div className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-gray-200 dark:bg-gray-600 inline-block"></span> 不匹配</div>
              </div>
            </div>
          </div>
        )}

        {/* 控制 */}
        <div className="flex items-center gap-3 flex-wrap pt-1 border-t border-gray-100 dark:border-gray-800">
          <button onClick={() => { setCur(0); setPlaying(false) }}
            className="px-3 py-1.5 rounded-lg text-sm border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800">⏮ 重置</button>
          <button onClick={() => setCur(c => Math.max(0, c - 1))} disabled={cur === 0}
            className="px-3 py-1.5 rounded-lg text-sm border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-30">← 上一窗口</button>
          <button onClick={() => setPlaying(p => !p)}
            className={`px-5 py-1.5 rounded-lg text-sm font-medium transition-colors ${playing ? 'bg-rose-500 hover:bg-rose-600 text-white' : 'bg-teal-600 hover:bg-teal-700 text-white'}`}>
            {playing ? '⏸ 暂停' : '▶ 滑动'}
          </button>
          <button onClick={() => setCur(c => Math.min(windows.length - 1, c + 1))} disabled={cur >= windows.length - 1}
            className="px-3 py-1.5 rounded-lg text-sm border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-30">→ 下一窗口</button>
          <div className="flex items-center gap-2 ml-auto">
            <span className="text-xs text-gray-500 dark:text-gray-400">速度</span>
            <input type="range" min={300} max={1500} step={100} value={1800 - speed}
              onChange={e => setSpeed(1800 - Number(e.target.value))} className="w-24 accent-teal-500" />
          </div>
        </div>
      </div>
    </div>
  )
}
