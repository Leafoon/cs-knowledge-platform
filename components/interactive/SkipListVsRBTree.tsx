'use client'

import { useState } from 'react'

type Row = { op: string; skipAvg: string; skipWorst: string; rb: string; note: string }

const TABLE: Row[] = [
  { op: '查找 Search',  skipAvg: 'O(log n)', skipWorst: 'O(n)',    rb: 'O(log n)', note: '跳表高度概率性，RB 严格平衡' },
  { op: '插入 Insert',  skipAvg: 'O(log n)', skipWorst: 'O(n)',    rb: 'O(log n)', note: '跳表隐式平衡；RB 需旋转' },
  { op: '删除 Delete',  skipAvg: 'O(log n)', skipWorst: 'O(n)',    rb: 'O(log n)', note: '跳表修指针；RB 旋转+颜色修复' },
  { op: '范围查询 Range',skipAvg: 'O(log n + k)', skipWorst: 'O(n + k)', rb: 'O(log n + k)', note: '跳表第0层天然有序，遍历简单' },
  { op: '空间 Space',   skipAvg: 'O(n)',     skipWorst: 'O(n log n)', rb: 'O(n)', note: '跳表每节点期望 2 个指针' },
  { op: '实现难度',     skipAvg: '简单',     skipWorst: '—',        rb: '复杂',   note: 'RB 旋转与颜色规则较难实现' },
  { op: '并发扩展',     skipAvg: '优秀',     skipWorst: '—',        rb: '困难',   note: '跳表 CAS 可实现无锁并发' },
]

const CASES = [
  { name: 'Redis ZSET', winner: 'skip', reason: '跳表支持并发、实现简单、范围查询友好，Redis ZSet 默认采用跳表' },
  { name: 'Java TreeMap', winner: 'rb', reason: '确定性 O(log n) 最坏保证，库实现已成熟，JDK 采用红黑树' },
  { name: 'LevelDB / RocksDB MemTable', winner: 'skip', reason: '跳表写入友好且支持并发访问，Google LevelDB 使用跳表' },
  { name: 'Linux 进程调度 CFS', winner: 'rb', reason: 'CFS 红黑树按 vruntime 排序，最坏 O(log n) 更可预测' },
]

export function SkipListVsRBTree() {
  const [tab, setTab] = useState<'table' | 'cases'>('table')

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-slate-700 to-slate-900 dark:from-slate-800 dark:to-slate-950 px-5 py-4">
        <h3 className="text-white font-bold text-base">⚖️ 跳表 vs 红黑树：性能与工程对比</h3>
        <p className="text-slate-300 text-xs mt-0.5">两者均实现 O(log n) 有序集合，工程取舍各有侧重</p>
        <div className="flex gap-2 mt-3">
          {(['table', 'cases'] as const).map(t => (
            <button key={t} onClick={() => setTab(t)}
              className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                tab === t ? 'bg-white text-slate-800 font-bold' : 'bg-white/20 text-slate-200 hover:bg-white/30'
              }`}>
              {t === 'table' ? '复杂度对比' : '实际应用案例'}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-5">
        {tab === 'table' && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="bg-slate-100 dark:bg-slate-800">
                  <th className="px-3 py-2 text-left font-semibold text-slate-600 dark:text-slate-300 rounded-tl-lg">操作</th>
                  <th className="px-3 py-2 text-center font-semibold text-blue-600 dark:text-blue-400">跳表 (平均)</th>
                  <th className="px-3 py-2 text-center font-semibold text-orange-500 dark:text-orange-400">跳表 (最坏)</th>
                  <th className="px-3 py-2 text-center font-semibold text-emerald-600 dark:text-emerald-400">红黑树</th>
                  <th className="px-3 py-2 text-left font-semibold text-slate-500 dark:text-slate-400 rounded-tr-lg">说明</th>
                </tr>
              </thead>
              <tbody>
                {TABLE.map((row, i) => (
                  <tr key={i} className={`border-t border-slate-100 dark:border-slate-800 ${i % 2 === 1 ? 'bg-slate-50 dark:bg-slate-800/40' : ''}`}>
                    <td className="px-3 py-2 font-medium text-slate-700 dark:text-slate-200">{row.op}</td>
                    <td className="px-3 py-2 text-center">
                      <span className="bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 px-2 py-0.5 rounded font-mono">{row.skipAvg}</span>
                    </td>
                    <td className="px-3 py-2 text-center">
                      <span className="bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-300 px-2 py-0.5 rounded font-mono">{row.skipWorst}</span>
                    </td>
                    <td className="px-3 py-2 text-center">
                      <span className="bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 px-2 py-0.5 rounded font-mono">{row.rb}</span>
                    </td>
                    <td className="px-3 py-2 text-slate-500 dark:text-slate-400">{row.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="mt-4 grid grid-cols-2 md:grid-cols-3 gap-2">
              {[
                { label: '跳表优势', items: ['实现简单直观', '并发修改友好（CAS）', '范围遍历高效', '调试容易'], color: 'blue' },
                { label: '红黑树优势', items: ['确定性最坏 O(log n)', '空间固定 O(n)', '无随机性、可预测', '标准库广泛采用'], color: 'emerald' },
                { label: '跳表劣势', items: ['最坏 O(n)（极低概率）', '内存碎片化较多', '随机性难以分析', '节点指针多'],  color: 'orange' },
              ].map(({ label, items, color }) => (
                <div key={label} className={`rounded-lg p-3 bg-${color}-50 dark:bg-${color}-900/10 border border-${color}-200 dark:border-${color}-800`}>
                  <p className={`text-xs font-bold text-${color}-700 dark:text-${color}-300 mb-2`}>{label}</p>
                  <ul className="space-y-1">
                    {items.map(item => (
                      <li key={item} className={`text-[11px] text-${color}-600 dark:text-${color}-400 flex items-start gap-1`}>
                        <span>•</span><span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === 'cases' && (
          <div className="space-y-3">
            {CASES.map(c => (
              <div key={c.name} className={`rounded-xl border p-4 flex gap-4 items-start ${
                c.winner === 'skip'
                  ? 'border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/10'
                  : 'border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-900/10'
              }`}>
                <span className={`text-2xl flex-shrink-0 w-8 text-center`}>
                  {c.winner === 'skip' ? '⛷️' : '🌲'}
                </span>
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-bold text-slate-800 dark:text-slate-100">{c.name}</span>
                    <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${
                      c.winner === 'skip'
                        ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                        : 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300'
                    }`}>
                      {c.winner === 'skip' ? '跳表' : '红黑树'}
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 dark:text-slate-400">{c.reason}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
