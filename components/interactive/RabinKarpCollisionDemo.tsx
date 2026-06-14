'use client'

import { useState, useMemo } from 'react'

// ─── 哈希计算工具（大模拟器） ───────────────────────────────
const D = 31
const Q1 = 1_000_003
const Q2 = 998_003

function charVal(c: string) { return c.charCodeAt(0) - 64 } // A=1,B=2,...

function modPow(base: number, exp: number, mod: number) {
  let r = 1; base = ((base % mod) + mod) % mod
  for (; exp > 0; exp >>= 1) {
    if (exp & 1) r = r * base % mod
    base = base * base % mod
  }
  return r
}

interface WindowInfo {
  s: number
  text: string
  h1: number
  h2: number
  singleHit: boolean   // 单模哈希（Q1）命中
  doubleHit: boolean   // 双模哈希（Q1+Q2）命中
  realMatch: boolean   // 真实字符串匹配
}

function analyze(T: string, P: string): { wins: WindowInfo[], pH1: number, pH2: number } {
  const n = T.length, m = P.length
  const h1_pow = modPow(D, m - 1, Q1)
  const h2_pow = modPow(D, m - 1, Q2)
  let pH1 = 0, pH2 = 0
  let tH1 = 0, tH2 = 0
  for (let i = 0; i < m; i++) {
    pH1 = (pH1 * D + charVal(P[i])) % Q1
    pH2 = (pH2 * D + charVal(P[i])) % Q2
    tH1 = (tH1 * D + charVal(T[i])) % Q1
    tH2 = (tH2 * D + charVal(T[i])) % Q2
  }
  const wins: WindowInfo[] = []
  for (let s = 0; s <= n - m; s++) {
    const text = T.slice(s, s + m)
    wins.push({
      s, text,
      h1: tH1, h2: tH2,
      singleHit: tH1 === pH1,
      doubleHit: tH1 === pH1 && tH2 === pH2,
      realMatch: text === P,
    })
    if (s < n - m) {
      tH1 = ((tH1 - charVal(T[s]) * h1_pow % Q1 + Q1) * D + charVal(T[s + m])) % Q1
      tH2 = ((tH2 - charVal(T[s]) * h2_pow % Q2 + Q2) * D + charVal(T[s + m])) % Q2
    }
  }
  return { wins, pH1, pH2 }
}

// 精心构造一个必有碰撞的示例（低模数下容易碰撞）
const DEMO_PRESETS = [
  { label: '正常示例', T: 'ABCABABCAB', P: 'ABC' },
  { label: '重叠匹配', T: 'AABABAA', P: 'ABA' },
  { label: '全相同', T: 'AAAAAAA', P: 'AAA' },
]

export default function RabinKarpCollisionDemo() {
  const [T, setT] = useState('ABCABABCAB')
  const [P, setP] = useState('ABC')
  const [TEdit, setTEdit] = useState('ABCABABCAB')
  const [PEdit, setPEdit] = useState('ABC')
  const [mode, setMode] = useState<'single' | 'double'>('single')

  const build = (t: string, p: string) => {
    const tc = t.toUpperCase().replace(/[^A-Z]/g, '').slice(0, 20)
    const pc = p.toUpperCase().replace(/[^A-Z]/g, '').slice(0, 8)
    if (!tc || !pc || pc.length > tc.length) return
    setT(tc); setP(pc)
  }

  const { wins, pH1, pH2 } = useMemo(() => analyze(T, P), [T, P])

  const totalWins = wins.length
  const singleHits = wins.filter(w => w.singleHit).length
  const doubleHits = wins.filter(w => w.doubleHit).length
  const realMatches = wins.filter(w => w.realMatch).length
  const spuriousSingle = singleHits - realMatches
  const spuriousDouble = doubleHits - realMatches

  return (
    <div className="my-8 rounded-2xl border border-amber-200 dark:border-amber-800 bg-white dark:bg-gray-900 shadow-lg overflow-hidden">
      {/* 标题 */}
      <div className="px-6 py-4 bg-gradient-to-r from-amber-500 to-orange-500 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-white/20 flex items-center justify-center text-white text-lg">⚡</div>
        <div>
          <h3 className="text-white font-bold text-base">哈希碰撞与双模哈希对比</h3>
          <p className="text-amber-100 text-xs mt-0.5">单模 vs 双模：碰撞概率从 1/Q 降至 1/(Q₁×Q₂)</p>
        </div>
      </div>

      <div className="p-6 space-y-5">
        {/* 输入 */}
        <div className="flex flex-wrap gap-2 mb-1">
          {DEMO_PRESETS.map(pr => (
            <button key={pr.label}
              onClick={() => { setTEdit(pr.T); setPEdit(pr.P); build(pr.T, pr.P) }}
              className={`px-3 py-1.5 rounded-lg text-xs border font-mono transition-all ${
                T === pr.T && P === pr.P
                  ? 'border-amber-500 bg-amber-100 dark:bg-amber-900 text-amber-700 dark:text-amber-300'
                  : 'border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-amber-400'
              }`}>{pr.label}（{pr.P}）</button>
          ))}
        </div>
        <div className="flex gap-2 flex-wrap">
          <input className="flex-1 min-w-[140px] rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 px-3 py-1.5 text-sm font-mono uppercase focus:outline-none focus:ring-2 focus:ring-amber-500"
            placeholder="文本 T" value={TEdit} onChange={e => setTEdit(e.target.value)} maxLength={20} />
          <input className="w-24 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 px-3 py-1.5 text-sm font-mono uppercase focus:outline-none focus:ring-2 focus:ring-amber-500"
            placeholder="模式 P" value={PEdit} onChange={e => setPEdit(e.target.value)} maxLength={8} />
          <button onClick={() => build(TEdit, PEdit)}
            className="px-4 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-600 text-white text-sm font-medium transition-colors">分析</button>
        </div>

        {/* 模式哈希 */}
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-800/40 space-y-1">
            <div className="text-xs text-gray-500 dark:text-gray-400 font-medium">模式 "{P}" 哈希（单模 Q₁={Q1.toLocaleString()}）</div>
            <div className="font-mono font-bold text-lg text-gray-700 dark:text-gray-300">{pH1}</div>
          </div>
          <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-800/40 space-y-1">
            <div className="text-xs text-gray-500 dark:text-gray-400 font-medium">模式 "{P}" 哈希（双模 Q₂={Q2.toLocaleString()}）</div>
            <div className="font-mono font-bold text-lg text-gray-700 dark:text-gray-300">{pH2}</div>
          </div>
        </div>

        {/* 模式切换 */}
        <div className="flex gap-2">
          {(['single', 'double'] as const).map(m => (
            <button key={m}
              onClick={() => setMode(m)}
              className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors border ${
                mode === m
                  ? m === 'single'
                    ? 'bg-amber-500 text-white border-amber-500'
                    : 'bg-emerald-600 text-white border-emerald-600'
                  : 'bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-400 border-gray-200 dark:border-gray-700 hover:border-amber-400'
              }`}>
              {m === 'single' ? '🔴 单模哈希' : '🟢 双模哈希（防碰撞）'}
            </button>
          ))}
        </div>

        {/* 统计卡片 */}
        <div className="grid grid-cols-3 gap-3 text-center text-sm">
          {[
            { label: '总窗口数', value: totalWins, color: 'text-gray-700 dark:text-gray-300' },
            { label: mode === 'single' ? '单模命中' : '双模命中', value: mode === 'single' ? singleHits : doubleHits,
              color: 'text-amber-600 dark:text-amber-400' },
            { label: '真实匹配', value: realMatches, color: 'text-emerald-600 dark:text-emerald-400' },
          ].map(c => (
            <div key={c.label} className="rounded-xl border border-gray-200 dark:border-gray-700 p-3 bg-gray-50 dark:bg-gray-800/40">
              <div className={`text-2xl font-bold font-mono ${c.color}`}>{c.value}</div>
              <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">{c.label}</div>
            </div>
          ))}
        </div>

        {/* 碰撞警告 */}
        {(mode === 'single' && spuriousSingle > 0) ? (
          <div className="flex items-start gap-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 rounded-xl p-4">
            <span className="text-2xl mt-0.5">⚠️</span>
            <div>
              <div className="text-sm font-bold text-amber-700 dark:text-amber-300">发现 {spuriousSingle} 个哈希碰撞（伪匹配，Spurious Hit）</div>
              <div className="text-xs text-amber-600/80 dark:text-amber-400/80 mt-1">
                哈希值相同，但字符串不同。单模时需 O(m) 逐字符验证才能确认。这发生在低模数（Q={Q1.toLocaleString()}）下，实践中使用 Q≈10⁹ 的大质数碰撞极罕见。
              </div>
            </div>
          </div>
        ) : mode === 'double' && spuriousDouble === 0 ? (
          <div className="flex items-center gap-3 bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-700 rounded-xl p-4">
            <span className="text-2xl">✅</span>
            <div className="text-sm text-emerald-700 dark:text-emerald-300">
              <span className="font-bold">双模哈希：零伪匹配！</span>
              <span className="text-xs ml-2 text-emerald-600/80 dark:text-emerald-400/60">碰撞概率 ≈ 1/(Q₁×Q₂) ≈ 10⁻¹²</span>
            </div>
          </div>
        ) : null}

        {/* 所有窗口详情 */}
        <div>
          <div className="text-xs text-gray-500 dark:text-gray-400 font-medium mb-2">各窗口哈希状态</div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-2 pr-3 text-gray-500 dark:text-gray-400 font-medium">位置</th>
                  <th className="text-left py-2 pr-3 text-gray-500 dark:text-gray-400 font-medium">窗口</th>
                  <th className="text-right py-2 pr-3 text-gray-500 dark:text-gray-400 font-medium">H₁</th>
                  {mode === 'double' && <th className="text-right py-2 pr-3 text-gray-500 dark:text-gray-400 font-medium">H₂</th>}
                  <th className="text-center py-2 text-gray-500 dark:text-gray-400 font-medium">状态</th>
                </tr>
              </thead>
              <tbody>
                {wins.map(w => {
                  const hit = mode === 'single' ? w.singleHit : w.doubleHit
                  return (
                    <tr key={w.s} className={`border-b border-gray-100 dark:border-gray-800 transition-colors ${
                      w.realMatch ? 'bg-emerald-50 dark:bg-emerald-900/20'
                      : hit ? 'bg-amber-50 dark:bg-amber-900/10'
                      : ''
                    }`}>
                      <td className="py-1.5 pr-3 text-gray-500 dark:text-gray-400">[{w.s}]</td>
                      <td className="py-1.5 pr-3 font-bold text-gray-700 dark:text-gray-300">{w.text}</td>
                      <td className={`py-1.5 pr-3 text-right ${w.singleHit ? 'text-amber-600 dark:text-amber-400 font-bold' : 'text-gray-500 dark:text-gray-400'}`}>{w.h1}</td>
                      {mode === 'double' && <td className={`py-1.5 pr-3 text-right ${w.doubleHit ? 'text-emerald-600 dark:text-emerald-400 font-bold' : 'text-gray-500 dark:text-gray-400'}`}>{w.h2}</td>}
                      <td className="py-1.5 text-center">
                        {w.realMatch
                          ? <span className="px-2 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300">✓ 真实匹配</span>
                          : hit
                          ? <span className="px-2 py-0.5 rounded-full bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-400">⚡ 伪匹配</span>
                          : <span className="text-gray-400 dark:text-gray-600">—</span>}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
