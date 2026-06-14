'use client'

import { useState } from 'react'

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
const ALGOS = [
  { key: 'kmp',     name: 'KMP',          full: 'Knuth-Morris-Pratt', color: 'emerald', header: 'bg-emerald-600', badge: 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300 ring-emerald-200 dark:ring-emerald-700' },
  { key: 'z',       name: 'Z-Function',   full: 'Z 函数',             color: 'amber',   header: 'bg-amber-500',   badge: 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 ring-amber-200 dark:ring-amber-700' },
  { key: 'man',     name: 'Manacher',     full: 'Manacher 算法',      color: 'fuchsia', header: 'bg-fuchsia-600', badge: 'bg-fuchsia-100 dark:bg-fuchsia-900/40 text-fuchsia-700 dark:text-fuchsia-300 ring-fuchsia-200 dark:ring-fuchsia-700' },
  { key: 'hash',    name: 'String Hash',  full: '字符串哈希',          color: 'teal',    header: 'bg-teal-600',    badge: 'bg-teal-100 dark:bg-teal-900/40 text-teal-700 dark:text-teal-300 ring-teal-200 dark:ring-teal-700' },
] as const

type Algo = typeof ALGOS[number]['key']

interface Cell { val: string; note?: string; score: 1 | 2 | 3 }
interface Row { aspect: string; kmp: Cell; z: Cell; man: Cell; hash: Cell }

const ROWS: Row[] = [
  { aspect: '预处理时间',
    kmp:  { val: 'O(n)', note: 'π 数组构建', score: 1 },
    z:    { val: 'O(n)', note: '[l,r] 窗口维护', score: 1 },
    man:  { val: 'O(n)', note: 'T 串+P 数组', score: 1 },
    hash: { val: 'O(n)', note: '前缀哈希+幂次', score: 1 } },
  { aspect: '字符串匹配\n查询 P in T',
    kmp:  { val: 'O(n+m)', note: '主算法', score: 1 },
    z:    { val: 'O(n+m)', note: '拼接 P$T 构建 Z', score: 1 },
    man:  { val: '不适用', note: '针对回文', score: 3 },
    hash: { val: 'O(m log n)', note: '二分+哈希', score: 2 } },
  { aspect: '最长回文子串',
    kmp:  { val: 'O(n)', note: 'Eertree/其他', score: 2 },
    z:    { val: 'O(n log n)', note: '二分+Z 辅助', score: 2 },
    man:  { val: 'O(n)', note: '主算法', score: 1 },
    hash: { val: 'O(n log n)', note: '二分+哈希验证', score: 2 } },
  { aspect: '子串比较\n任意 s[l..r] vs s[l\'..r\']',
    kmp:  { val: '不直接', note: '需重建', score: 3 },
    z:    { val: '不直接', note: '只比较前缀', score: 3 },
    man:  { val: '不直接', note: '非哈希结构', score: 3 },
    hash: { val: 'O(1)', note: '核心优势', score: 1 } },
  { aspect: '字符串周期检测',
    kmp:  { val: 'O(n)', note: 'n − π[n-1]', score: 1 },
    z:    { val: 'O(n)', note: 'Z[i] 验证 period', score: 1 },
    man:  { val: '不适用', note: '针对回文', score: 3 },
    hash: { val: 'O(n log n)', note: '枚举+哈希验证', score: 2 } },
  { aspect: '实现难度',
    kmp:  { val: '中等', note: 'π 数组易错', score: 2 },
    z:    { val: '较简单', note: '逻辑统一', score: 1 },
    man:  { val: '较难', note: '奇偶处理/索引', score: 3 },
    hash: { val: '简单', note: '模板化', score: 1 } },
  { aspect: '正确性保证',
    kmp:  { val: '确定性', note: '无误判', score: 1 },
    z:    { val: '确定性', note: '无误判', score: 1 },
    man:  { val: '确定性', note: '无误判', score: 1 },
    hash: { val: '概率性', note: '极低碰撞率', score: 2 } },
]

const SCENARIOS: { key: string; label: string; desc: string; best: Algo[] }[] = [
  { key: 'match',   label: '字符串匹配',   desc: '在文本 T 中找模式 P 所有出现位置',      best: ['kmp', 'z'] },
  { key: 'palin',   label: '最长回文',     desc: '求字符串的最长回文子串',               best: ['man'] },
  { key: 'cmp',     label: '子串比较',     desc: 'O(1) 比较任意两段子串是否相等',         best: ['hash'] },
  { key: 'period',  label: '周期检测',     desc: '判断字符串最小正周期',                 best: ['kmp', 'z'] },
  { key: 'palindrome_count', label: '回文计数', desc: '统计所有不同回文子串/中心数',      best: ['man', 'hash'] },
  { key: 'dedup',   label: '本质不同子串', desc: '计算不同子串个数（字符串哈希辅助）',   best: ['hash'] },
]

const SCORE_STYLE = ['', 'text-emerald-600 dark:text-emerald-400', 'text-amber-600 dark:text-amber-400', 'text-rose-500']
const SCORE_BG    = ['', 'bg-emerald-50 dark:bg-emerald-900/20', 'bg-amber-50 dark:bg-amber-900/20', 'bg-rose-50 dark:bg-rose-900/20']
const SCORE_DOT   = ['', '●●●', '●●○', '●○○']

export default function StringAlgoComparison() {
  const [scenario, setScenario] = useState<string | null>(null)
  const [highlight, setHighlight] = useState<Algo | null>(null)

  const activeBest = scenario ? SCENARIOS.find(s => s.key === scenario)?.best ?? [] : []

  return (
    <div className="rounded-2xl overflow-hidden border border-slate-300 dark:border-slate-700 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-700 to-slate-900 dark:from-slate-800 dark:to-slate-950 px-5 py-4">
        <h3 className="text-white font-bold text-base">📊 四大字符串算法横向对比</h3>
        <p className="text-slate-300 text-xs mt-0.5">KMP · Z函数 · Manacher · 字符串哈希 —— 选型指南与能力矩阵</p>
      </div>

      <div className="bg-white dark:bg-gray-900 p-4 space-y-4">
        {/* Scenario buttons */}
        <div>
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">按场景筛选最优算法：</p>
          <div className="flex flex-wrap gap-2">
            {SCENARIOS.map(sc => (
              <button key={sc.key}
                onClick={() => setScenario(s => s === sc.key ? null : sc.key)}
                className={`px-3 py-1.5 text-xs rounded-xl border transition-all ${
                  scenario === sc.key
                    ? 'bg-slate-700 dark:bg-slate-600 text-white border-slate-700 dark:border-slate-600'
                    : 'bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-slate-400'
                }`}>
                {sc.label}
              </button>
            ))}
          </div>
          {scenario && (
            <p className="text-[11px] mt-1.5 text-slate-500 dark:text-slate-400">
              {SCENARIOS.find(s => s.key === scenario)?.desc}
              {' '}→ 推荐：
              <span className="font-semibold text-slate-700 dark:text-slate-200">
                {activeBest.map(k => ALGOS.find(a => a.key === k)?.name).join('、')}
              </span>
            </p>
          )}
        </div>

        {/* Algo badges -> toggle column highlight */}
        <div className="flex gap-2 flex-wrap">
          {ALGOS.map(a => (
            <button key={a.key}
              onClick={() => setHighlight(h => h === a.key ? null : a.key as Algo)}
              className={`px-3 py-1.5 text-xs rounded-xl ring-1 font-medium transition-all ${a.badge} ${highlight === a.key ? 'ring-2 scale-105' : ''}`}>
              {a.name}
            </button>
          ))}
          {(highlight || scenario) && (
            <button onClick={() => { setHighlight(null); setScenario(null) }}
              className="px-2.5 py-1.5 text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">✕ 清除</button>
          )}
        </div>

        {/* Main table */}
        <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700">
          <table className="text-xs w-full border-collapse min-w-max">
            <thead>
              <tr>
                <th className="py-2.5 px-4 text-left text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-800 font-medium w-40">维度</th>
                {ALGOS.map(a => {
                  const isBest = activeBest.includes(a.key)
                  const isHL = highlight === a.key
                  return (
                    <th key={a.key} className={`py-2.5 px-4 text-center font-semibold text-white ${a.header} ${
                      scenario && !isBest ? 'opacity-30' : ''
                    } ${isHL ? 'ring-2 ring-inset ring-white/50' : ''} transition-all`}>
                      <div>{a.name}</div>
                      <div className="text-[10px] font-normal text-white/70 mt-0.5">{a.full}</div>
                      {scenario && isBest && <div className="text-[10px] font-bold text-white/90">★ 推荐</div>}
                    </th>
                  )
                })}
              </tr>
            </thead>
            <tbody>
              {ROWS.map((row, ri) => (
                <tr key={ri} className="border-t border-gray-100 dark:border-gray-800">
                  <td className="py-2.5 px-4 text-gray-600 dark:text-gray-400 font-medium whitespace-pre-line leading-tight text-xs">{row.aspect}</td>
                  {(['kmp', 'z', 'man', 'hash'] as Algo[]).map(k => {
                    const cell = row[k] as Cell
                    const isBest = activeBest.includes(k)
                    const isHL = highlight === k
                    return (
                      <td key={k} className={`py-2.5 px-4 text-center transition-all ${
                        scenario && !isBest ? 'opacity-25' : ''
                      } ${SCORE_BG[cell.score]} ${isHL ? 'ring-1 ring-inset ring-slate-300 dark:ring-slate-600' : ''}`}>
                        <div className={`font-mono font-semibold text-xs ${SCORE_STYLE[cell.score]}`}>{cell.val}</div>
                        {cell.note && <div className="text-[10px] text-gray-400 dark:text-gray-500 mt-0.5">{cell.note}</div>}
                        <div className={`text-[10px] mt-0.5 ${SCORE_STYLE[cell.score]}`}>{SCORE_DOT[cell.score]}</div>
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Score legend */}
        <div className="flex flex-wrap gap-4 text-[11px] text-gray-500 dark:text-gray-400">
          <span className="flex items-center gap-1"><span className="text-emerald-600 font-bold">●●●</span> 最优</span>
          <span className="flex items-center gap-1"><span className="text-amber-600 font-bold">●●○</span> 可用</span>
          <span className="flex items-center gap-1"><span className="text-rose-500 font-bold">●○○</span> 不适合</span>
        </div>

        {/* Quick guide cards */}
        <div className="grid sm:grid-cols-2 gap-2">
          {[
            { color: 'border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-900/20', title: '首选 KMP', body: '精确字符串匹配首选，确定性，O(n+m) 复杂度；周期检测一行搞定' },
            { color: 'border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/20', title: '首选 Z-Function', body: '同类题目代码更短，逻辑比 KMP 统一；字符串匹配/出现位置/周期判断都可用' },
            { color: 'border-fuchsia-300 dark:border-fuchsia-700 bg-fuchsia-50 dark:bg-fuchsia-900/20', title: '首选 Manacher', body: '最长回文子串 O(n) 唯一选择；结合哈希可做回文计数 O(n log n)' },
            { color: 'border-teal-300 dark:border-teal-700 bg-teal-50 dark:bg-teal-900/20', title: '首选 字符串哈希', body: '任意子串 O(1) 比较；本质不同子串计数；LCP 辅助工具；注意碰撞风险' },
          ].map(({ color, title, body }) => (
            <div key={title} className={`p-3 rounded-xl border ${color} text-xs`}>
              <p className="font-semibold text-gray-700 dark:text-gray-300 mb-1">{title}</p>
              <p className="text-gray-500 dark:text-gray-400 leading-relaxed">{body}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
