'use client'

import { useState } from 'react'

const STRUCTURES = [
  { key: 'sa', name: 'SA + LCP 数组', short: 'SA', color: 'indigo', badge: 'bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 ring-indigo-200 dark:ring-indigo-700', header: 'bg-indigo-500' },
  { key: 'st', name: '后缀树 (Suffix Tree)', short: 'ST',  color: 'purple', badge: 'bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300 ring-purple-200 dark:ring-purple-700', header: 'bg-purple-500' },
  { key: 'sam', name: '后缀自动机 (SAM)', short: 'SAM', color: 'rose', badge: 'bg-rose-100 dark:bg-rose-900/40 text-rose-700 dark:text-rose-300 ring-rose-200 dark:ring-rose-700', header: 'bg-rose-500' },
] as const

type Struct = typeof STRUCTURES[number]['key']

interface Row {
  aspect: string; sub?: string
  sa: { val: string; note?: string; score: 1 | 2 | 3 }
  st: { val: string; note?: string; score: 1 | 2 | 3 }
  sam: { val: string; note?: string; score: 1 | 2 | 3 }
}

const ROWS: Row[] = [
  { aspect: '构造时间',
    sa:  { val: 'O(n log n)', note: 'SA-IS 可达 O(n)', score: 2 },
    st:  { val: 'O(n)', note: 'Ukkonen 在线构造', score: 1 },
    sam: { val: 'O(n)', note: 'extend() 均摊', score: 1 } },
  { aspect: '构造空间',
    sa:  { val: 'O(n)', note: 'SA + rank + LCP', score: 1 },
    st:  { val: 'O(n)', note: '边数 ≤ 2n-1 节点', score: 1 },
    sam: { val: 'O(n)', note: '最多 2n 态/3n 边', score: 1 } },
  { aspect: '实现难度',
    sa:  { val: '中等', note: '倍增易写；SA-IS 复杂', score: 2 },
    st:  { val: '困难', note: 'Ukkonen 细节繁多', score: 3 },
    sam: { val: '中等', note: 'extend 逻辑一致', score: 2 } },
  { aspect: '子串搜索\n查询 P [O(m log n)]',
    sa:  { val: 'O(m log n)', note: '二分 SA', score: 2 },
    st:  { val: 'O(m)', note: '沿树走 m 步', score: 1 },
    sam: { val: 'O(m)', note: 'DAG 走 m 步', score: 1 } },
  { aspect: '最长公共子串 (LCS)',
    sa:  { val: 'O(n log n)', note: '合并两字符串, SA+LCP', score: 2 },
    st:  { val: 'O(n)', note: '广义后缀树, O(n) 构造', score: 1 },
    sam: { val: 'O(n)', note: '在一个串的 SAM 上跑另一个', score: 1 } },
  { aspect: '最长重复子串',
    sa:  { val: 'O(n)', note: 'max(LCP[i]) 即答案', score: 1 },
    st:  { val: 'O(n)', note: '最深内部节点深度', score: 1 },
    sam: { val: 'O(n)', note: '非根态 max(len) - max(link.len)', score: 2 } },
  { aspect: '不同子串个数',
    sa:  { val: 'O(n)', note: 'n(n+1)/2 - sum(LCP)', score: 1 },
    st:  { val: 'O(n)', note: '所有边标签长度之和', score: 1 },
    sam: { val: 'O(n)', note: 'DP on SAM DAG', score: 1 } },
  { aspect: '出现次数统计',
    sa:  { val: 'O(log n)', note: '二分找区间长度', score: 2 },
    st:  { val: 'O(1)', note: '子树叶数', score: 1 },
    sam: { val: 'O(n)', note: 'DP/拓扑排序 endpos 大小', score: 2 } },
]

const SCENARIOS: { key: string; label: string; desc: string; best: Struct[] }[] = [
  { key: 'search', label: '子串快速搜索', desc: '给定模式串 P，找所有出现位置', best: ['st', 'sam'] },
  { key: 'lrs', label: '最长重复子串', desc: 'LCS 变体，无需额外字符串', best: ['sa'] },
  { key: 'lcs', label: '两串最长公共子串', desc: '两个字符串的 LCS 问题', best: ['st', 'sam'] },
  { key: 'count', label: '不同子串计数', desc: '字符串中本质不同子串个数', best: ['sa', 'st', 'sam'] },
  { key: 'freq', label: '出现频率最高子串', desc: '找出现次数最多的子串', best: ['st'] },
  { key: 'impl', label: '比赛快速实现', desc: '竞赛中快速写出可用的代码', best: ['sa'] },
]

const SCORE_STYLE = ['', 'text-emerald-600 dark:text-emerald-400', 'text-amber-600 dark:text-amber-400', 'text-rose-600 dark:text-rose-500']
const SCORE_BG   = ['', 'bg-emerald-50 dark:bg-emerald-900/20', 'bg-amber-50 dark:bg-amber-900/20', 'bg-rose-50 dark:bg-rose-900/20']
const SCORE_DOT  = ['', '●●●', '●●○', '●○○']

export default function StringStructureComparison() {
  const [scenario, setScenario] = useState<string | null>(null)
  const [highlight, setHighlight] = useState<Struct | null>(null)

  const activeBest = scenario ? SCENARIOS.find(s => s.key === scenario)?.best ?? [] : []

  return (
    <div className="rounded-2xl overflow-hidden border border-slate-300 dark:border-slate-700 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-700 to-slate-800 dark:from-slate-800 dark:to-slate-900 px-5 py-4">
        <h3 className="text-white font-bold text-base">📊 后缀结构横向对比</h3>
        <p className="text-slate-300 text-xs mt-0.5">SA + LCP / 后缀树 / SAM —— 选型指南与能力维度比较</p>
      </div>

      <div className="bg-white dark:bg-gray-900 p-4 space-y-4">
        {/* Scenario filter */}
        <div>
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">按场景筛选最优结构：</p>
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
              {activeBest.map(k => STRUCTURES.find(s => s.key === k)?.short).join('、')}
            </p>
          )}
        </div>

        {/* Structure badges (toggle highlight) */}
        <div className="flex gap-2 flex-wrap">
          {STRUCTURES.map(st => (
            <button key={st.key}
              onClick={() => setHighlight(h => h === st.key ? null : st.key as Struct)}
              className={`px-3 py-1.5 text-xs rounded-xl ring-1 font-medium transition-all ${st.badge} ${highlight === st.key ? 'ring-2 scale-105' : ''}`}>
              {st.short}
            </button>
          ))}
          {(highlight || scenario) && (
            <button onClick={() => { setHighlight(null); setScenario(null) }}
              className="px-2.5 py-1.5 text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">✕ 清除</button>
          )}
        </div>

        {/* Comparison table */}
        <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700">
          <table className="text-xs w-full border-collapse min-w-max">
            <thead>
              <tr>
                <th className="py-2.5 px-4 text-left text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-800 font-medium w-36">维度</th>
                {STRUCTURES.map(st => {
                  const isBest = activeBest.includes(st.key)
                  const isHL = highlight === st.key
                  return (
                    <th key={st.key} className={`py-2.5 px-4 text-center font-semibold text-white ${st.header} ${
                      scenario && !isBest ? 'opacity-30' : ''
                    } ${isHL ? 'ring-2 ring-inset ring-white/60' : ''} transition-all`}>
                      <div>{st.short}</div>
                      {scenario && isBest && <div className="text-[10px] font-normal text-white/80 mt-0.5">★ 推荐</div>}
                    </th>
                  )
                })}
              </tr>
            </thead>
            <tbody>
              {ROWS.map((row, ri) => (
                <tr key={ri} className="border-t border-gray-100 dark:border-gray-800">
                  <td className="py-2.5 px-4 text-gray-600 dark:text-gray-400 font-medium whitespace-pre-line leading-tight">
                    {row.aspect}
                  </td>
                  {(['sa', 'st', 'sam'] as Struct[]).map(k => {
                    const cell = row[k]
                    const struct = STRUCTURES.find(s => s.key === k)!
                    const isBest = activeBest.includes(k)
                    const isHL = highlight === k
                    return (
                      <td key={k} className={`py-2.5 px-4 text-center transition-all ${
                        scenario && !isBest ? 'opacity-25' : ''
                      } ${SCORE_BG[cell.score]} ${isHL ? 'ring-1 ring-inset ring-slate-300 dark:ring-slate-600' : ''}`}>
                        <div className={`font-mono font-semibold ${SCORE_STYLE[cell.score]}`}>{cell.val}</div>
                        {cell.note && <div className="text-[10px] text-gray-400 dark:text-gray-500 mt-0.5">{cell.note}</div>}
                        <div className={`text-[10px] mt-0.5 ${SCORE_STYLE[cell.score]}`} title="优=●●●/中=●●○/差=●○○">
                          {SCORE_DOT[cell.score]}
                        </div>
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
          <span className="flex items-center gap-1"><span className="text-emerald-600 font-bold">●●●</span> 优</span>
          <span className="flex items-center gap-1"><span className="text-amber-600 font-bold">●●○</span> 中</span>
          <span className="flex items-center gap-1"><span className="text-rose-500 font-bold">●○○</span> 较弱</span>
          <span className="ml-2 italic">底色深=优先；点击结构名高亮对应列</span>
        </div>

        {/* Quick guide */}
        <div className="grid sm:grid-cols-3 gap-2">
          {[
            { color: 'border-indigo-300 dark:border-indigo-700 bg-indigo-50 dark:bg-indigo-900/20', title: '首选 SA', body: '竞赛常用，代码量少，SA-IS 理论最优；适合大多数字符串题' },
            { color: 'border-purple-300 dark:border-purple-700 bg-purple-50 dark:bg-purple-900/20', title: '首选 后缀树', body: '多字符串 LCS、子树频率统计；但 Ukkonen 代码繁，实战少用' },
            { color: 'border-rose-300 dark:border-rose-700 bg-rose-50 dark:bg-rose-900/20', title: '首选 SAM', body: '处理单字符串的子串计数、LCS、最长公共扩展；在线对比强劲' },
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
