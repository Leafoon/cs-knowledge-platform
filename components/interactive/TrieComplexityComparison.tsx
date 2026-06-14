'use client'

import { useState } from 'react'

// ─── 数据定义 ─────────────────────────────────────────────────
type Complexity = { label: string; color: string; score: number }

interface TrieVariant {
  name: string
  short: string
  icon: string
  space: Complexity
  buildTime: Complexity
  queryTime: Complexity
  description: string
  bestFor: string[]
  weakFor: string[]
  example: string
}

const O = {
  const: { label: 'O(1)', color: 'bg-emerald-500', score: 5 },
  logN: { label: 'O(log n)', color: 'bg-green-400', score: 4 },
  sumLen: { label: 'O(Σ|wᵢ|)', color: 'bg-yellow-400', score: 3 },
  nL: { label: 'O(n·L)', color: 'bg-orange-400', score: 2 },
  sigma: { label: 'O(n·Σ)', color: 'bg-red-400', score: 1 },
} as const

const VARIANTS: TrieVariant[] = [
  {
    name: '数组 Trie', short: 'Array', icon: '🗂️',
    space:     { label: 'O(n·Σ)', color: 'bg-red-400', score: 1 },
    buildTime: { label: 'O(Σ|wᵢ|)', color: 'bg-yellow-400', score: 3 },
    queryTime: { label: 'O(L)', color: 'bg-emerald-500', score: 5 },
    description: '每个节点用大小为字母表 Σ 的数组存子节点，查询 O(1) 跳转，但空间极大。',
    bestFor: ['字母表固定且小（如小写字母26个）', '查询极为频繁·延迟敏感', 'AC 自动机底层结构'],
    weakFor: ['键空间大（Unicode、哈希）', '内存受限环境'],
    example: 'AC 自动机 / 编译器符号表',
  },
  {
    name: 'HashMap Trie', short: 'HashMap', icon: '🗃️',
    space:     { label: 'O(Σ|wᵢ|)', color: 'bg-yellow-400', score: 3 },
    buildTime: { label: 'O(Σ|wᵢ|)', color: 'bg-yellow-400', score: 3 },
    queryTime: { label: 'O(L) avg', color: 'bg-green-400', score: 4 },
    description: '子节点用哈希表存储，空间按需分配。适合稀疏字母表，但常数因子较大。',
    bestFor: ['字母表大或动态变化', '内存适中·实现简单', '通用前缀树场景'],
    weakFor: ['极低延迟场景（哈希碰撞）', '缓存不友好'],
    example: '词典 / 路由前缀树 / 自动补全',
  },
  {
    name: 'Radix Tree', short: 'Radix', icon: '🗜️',
    space:     { label: 'O(n·L\')', color: 'bg-green-400', score: 4 },
    buildTime: { label: 'O(Σ|wᵢ|)', color: 'bg-yellow-400', score: 3 },
    queryTime: { label: 'O(L)', color: 'bg-emerald-500', score: 5 },
    description: '合并单链节点，节点数 O(单词数)，每条边携带多字符标签，空间压缩显著。',
    bestFor: ['单词有大量公共前缀', 'IP 路由表（CIDR 前缀）', '内存敏感且需要前缀查询'],
    weakFor: ['分裂操作实现较复杂', '字符串比较比单字符慢'],
    example: 'Linux VFS / 网络路由 / Nginx 路由',
  },
  {
    name: 'AC 自动机', short: 'AC Auto', icon: '⚡',
    space:     { label: 'O(n·Σ)', color: 'bg-red-400', score: 1 },
    buildTime: { label: 'O(Σ|pᵢ|)', color: 'bg-yellow-400', score: 3 },
    queryTime: { label: 'O(T + 匹配数)', color: 'bg-emerald-500', score: 5 },
    description: 'Trie + BFS 失败链接，可在线性时间完成多模式字符串匹配，是多模式场景唯一选择。',
    bestFor: ['多模式字符串搜索', '内容审核 / 关键词过滤', 'DNA 序列检测'],
    weakFor: ['只支持精确模式匹配', '空间与 Σ 成正比'],
    example: '内容审核 / 病毒特征匹配',
  },
  {
    name: '二进制 Trie', short: 'Binary', icon: '⚡',
    space:     { label: 'O(n·B)', color: 'bg-orange-400', score: 2 },
    buildTime: { label: 'O(n·B)', color: 'bg-orange-400', score: 2 },
    queryTime: { label: 'O(B)', color: 'bg-emerald-500', score: 5 },
    description: '按二进制位组织数字，专门解题 XOR 最大/最小值问题，B 为位数（通常 30~64）。',
    bestFor: ['XOR 最大/最小值', '位运算贪心问题', '数字前缀查询'],
    weakFor: ['仅适用于数值型键', '非位相关场景没有优势'],
    example: 'LeetCode 421 · XOR 最大值对',
  },
]

// ─── 场景 ────────────────────────────────────────────────────
const SCENARIOS = [
  { key: 'prefix', label: '🔠 前缀补全', recommended: ['Array', 'HashMap', 'Radix'] },
  { key: 'multi',  label: '🔎 多模式匹配', recommended: ['AC Auto'] },
  { key: 'xor',   label: '🔢 XOR 最大值', recommended: ['Binary'] },
  { key: 'route', label: '🌐 IP 路由', recommended: ['Radix'] },
  { key: 'mem',   label: '💾 内存敏感', recommended: ['Radix', 'HashMap'] },
] as const

const METRICS = [
  { key: 'space',     label: '空间' },
  { key: 'buildTime', label: '构建时间' },
  { key: 'queryTime', label: '查询时间' },
] as const

function ScoreBar({ score, color }: { score: number; color: string }) {
  return (
    <div className="flex items-center gap-1">
      <div className="flex gap-0.5">
        {Array.from({ length: 5 }, (_, i) => (
          <div key={i} className={`w-2.5 h-2.5 rounded-sm ${i < score ? color : 'bg-gray-200 dark:bg-gray-700'}`} />
        ))}
      </div>
    </div>
  )
}

// ─── 主组件 ───────────────────────────────────────────────────
export default function TrieComplexityComparison() {
  const [activeScenario, setActiveScenario] = useState<string | null>(null)
  const [expandedRow, setExpandedRow] = useState<string | null>(null)

  const recommended: string[] = activeScenario
    ? [...(SCENARIOS.find(s => s.key === activeScenario)?.recommended ?? [])]
    : []

  const isHighlighted = (v: TrieVariant) =>
    !activeScenario || recommended.includes(v.short)

  return (
    <div className="rounded-2xl overflow-hidden border border-slate-300 dark:border-slate-700 shadow-lg">
      {/* 头部 */}
      <div className="bg-gradient-to-r from-slate-700 to-slate-800 dark:from-slate-800 dark:to-gray-900 px-5 py-4">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <h3 className="text-white font-bold text-base">📊 Trie 变体对比决策</h3>
            <p className="text-slate-300 text-xs mt-0.5">按题目场景选择最适合的 Trie 结构</p>
          </div>
          {/* 场景筛选 */}
          <div className="flex gap-1.5 flex-wrap">
            {SCENARIOS.map(s => (
              <button key={s.key}
                onClick={() => setActiveScenario(prev => prev === s.key ? null : s.key)}
                className={`px-2.5 py-1 text-xs rounded-lg transition-all ${
                  activeScenario === s.key
                    ? 'bg-white text-slate-800 font-semibold'
                    : 'bg-slate-600/50 text-slate-200 hover:bg-slate-600'}`}>
                {s.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 p-4">
        {activeScenario && (
          <div className="mb-3 px-3 py-2 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 text-xs text-blue-700 dark:text-blue-300">
            推荐结构：<span className="font-bold">{recommended.join('、')}</span>（绿色高亮）
          </div>
        )}

        {/* 主表 */}
        <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50 dark:bg-slate-800">
                <th className="py-2.5 px-3 text-left text-gray-500 dark:text-gray-400 font-medium border-b border-gray-200 dark:border-gray-700">结构</th>
                {METRICS.map(m => (
                  <th key={m.key} className="py-2.5 px-3 text-center text-gray-500 dark:text-gray-400 font-medium border-b border-gray-200 dark:border-gray-700">{m.label}</th>
                ))}
                <th className="py-2.5 px-3 text-left text-gray-500 dark:text-gray-400 font-medium border-b border-gray-200 dark:border-gray-700">典型场景</th>
              </tr>
            </thead>
            <tbody>
              {VARIANTS.map(v => {
                const hl = isHighlighted(v)
                const expanded = expandedRow === v.short
                return (
                  <>
                    <tr key={v.short}
                      onClick={() => setExpandedRow(prev => prev === v.short ? null : v.short)}
                      className={`cursor-pointer border-b border-gray-100 dark:border-gray-800 transition-all ${
                        hl ? 'hover:bg-slate-50 dark:hover:bg-slate-800/50' : 'opacity-30 pointer-events-none'
                      } ${expanded ? 'bg-slate-50 dark:bg-slate-800/50' : ''}`}>
                      <td className="py-2.5 px-3">
                        <div className="flex items-center gap-2">
                          <span className="text-base">{v.icon}</span>
                          <div>
                            <div className={`font-semibold ${hl && activeScenario ? 'text-emerald-700 dark:text-emerald-400' : 'text-gray-800 dark:text-gray-200'}`}>{v.name}</div>
                            <div className="text-[10px] text-gray-400 dark:text-gray-500">{v.short}</div>
                          </div>
                        </div>
                      </td>
                      {METRICS.map(m => {
                        const c = v[m.key as keyof TrieVariant] as Complexity
                        return (
                          <td key={m.key} className="py-2.5 px-3">
                            <div className="flex flex-col items-center gap-1">
                              <span className="font-mono text-[10px] text-gray-600 dark:text-gray-400">{c.label}</span>
                              <ScoreBar score={c.score} color={c.color} />
                            </div>
                          </td>
                        )
                      })}
                      <td className="py-2.5 px-3 text-gray-600 dark:text-gray-400">{v.example}</td>
                    </tr>
                    {expanded && (
                      <tr key={`${v.short}-exp`} className="bg-slate-50 dark:bg-slate-800/70">
                        <td colSpan={5} className="px-4 pb-3 pt-1">
                          <p className="text-[11px] text-gray-600 dark:text-gray-300 mb-2">{v.description}</p>
                          <div className="grid grid-cols-2 gap-3">
                            <div>
                              <p className="text-[10px] font-semibold text-emerald-600 dark:text-emerald-400 mb-1">✅ 适合</p>
                              {v.bestFor.map(b => <p key={b} className="text-[10px] text-gray-500 dark:text-gray-400">• {b}</p>)}
                            </div>
                            <div>
                              <p className="text-[10px] font-semibold text-rose-600 dark:text-rose-400 mb-1">❌ 不适合</p>
                              {v.weakFor.map(b => <p key={b} className="text-[10px] text-gray-500 dark:text-gray-400">• {b}</p>)}
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* 图例 */}
        <div className="mt-3 flex flex-wrap items-center gap-3 text-[10px] text-gray-400 dark:text-gray-500">
          <span>性能评分：</span>
          {[
            { color: 'bg-emerald-500', label: '★★★★★ 优秀' },
            { color: 'bg-green-400',   label: '★★★★ 良好' },
            { color: 'bg-yellow-400',  label: '★★★ 一般' },
            { color: 'bg-orange-400',  label: '★★ 较差' },
            { color: 'bg-red-400',     label: '★ 差' },
          ].map(g => (
            <div key={g.label} className="flex items-center gap-1">
              <div className={`w-2 h-2 rounded-sm ${g.color}`} />
              <span>{g.label}</span>
            </div>
          ))}
          <span className="ml-auto">点击行展开详情</span>
        </div>
      </div>
    </div>
  )
}
