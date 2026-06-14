'use client'

import { useState, useMemo } from 'react'

const PRESETS = [
  { label: 'banana', s: 'banana' },
  { label: 'aabaa', s: 'aabaa' },
  { label: 'abcabc', s: 'abcabc' },
  { label: 'mississippi', s: 'mississippi' },
]

interface DoublingRound {
  k: number
  sa: number[]
  rank: number[]
  keys: [number, number][]   // (rank[i], rank[i+k]) for each position
  label: string
}

function computeDoublingRounds(s: string): DoublingRound[] {
  const n = s.length
  if (n === 0) return []

  const rounds: DoublingRound[] = []
  let sa = Array.from({ length: n }, (_, i) => i)
  let rank = Array.from(s, c => c.charCodeAt(0))

  // Round 0: initial single-char rank
  const sorted0 = [...sa].sort((a, b) => rank[a] - rank[b])
  const rank0 = new Array(n).fill(0)
  rank0[sorted0[0]] = 0
  for (let i = 1; i < n; i++) {
    rank0[sorted0[i]] = rank0[sorted0[i - 1]] + (rank[sorted0[i]] !== rank[sorted0[i - 1]] ? 1 : 0)
  }
  rank = rank0
  sa = sorted0

  rounds.push({
    k: 0,
    sa: [...sa],
    rank: [...rank],
    keys: sa.map(i => [rank[i], -1] as [number, number]),
    label: '初始：按单字符排名',
  })

  let k = 1
  while (k < n) {
    const key = (i: number): [number, number] => [rank[i], i + k < n ? rank[i + k] : -1]
    const newSa = [...Array.from({ length: n }, (_, i) => i)].sort((a, b) => {
      const [ka0, ka1] = key(a)
      const [kb0, kb1] = key(b)
      return ka0 !== kb0 ? ka0 - kb0 : ka1 - kb1
    })
    const keys: [number, number][] = newSa.map(i => key(i))

    const newRank = new Array(n).fill(0)
    newRank[newSa[0]] = 0
    for (let i = 1; i < n; i++) {
      const prev = keys[i - 1], curr = keys[i]
      newRank[newSa[i]] = newRank[newSa[i - 1]] + (prev[0] !== curr[0] || prev[1] !== curr[1] ? 1 : 0)
    }
    rank = newRank
    sa = newSa

    rounds.push({
      k,
      sa: [...sa],
      rank: [...rank],
      keys: [...keys],
      label: `k=${k}：按 (rank[i], rank[i+${k}]) 二元键排名`,
    })

    if (rank[newSa[n - 1]] === n - 1) break
    k <<= 1
  }

  return rounds
}

const CELL_COLORS = [
  'bg-blue-500', 'bg-emerald-500', 'bg-amber-500', 'bg-rose-500',
  'bg-violet-500', 'bg-cyan-500', 'bg-orange-500', 'bg-teal-500',
  'bg-pink-500', 'bg-lime-500', 'bg-indigo-500',
]

export default function SuffixArrayDoubling() {
  const [presetIdx, setPresetIdx] = useState(0)
  const [roundIdx, setRoundIdx] = useState(0)

  const s = PRESETS[presetIdx].s
  const n = s.length
  const rounds = useMemo(() => computeDoublingRounds(s), [s])

  const round = rounds[roundIdx]
  const prevRound = roundIdx > 0 ? rounds[roundIdx - 1] : null

  // Detect changed ranks (vs previous round)
  const changedPos = useMemo(() => {
    if (!prevRound) return new Set<number>()
    return new Set(round.sa.filter(i => round.rank[i] !== prevRound.rank[i]))
  }, [round, prevRound])

  function reset(pi: number) { setPresetIdx(pi); setRoundIdx(0) }

  return (
    <div className="rounded-2xl overflow-hidden border border-emerald-200 dark:border-emerald-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-600 to-teal-600 px-5 py-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-white font-bold text-base">⚡ 倍增法构造后缀数组</h3>
          <p className="text-emerald-100 text-xs mt-0.5">每轮用上轮排名作键值，比较长度翻倍 → $O(n\log^2 n)$</p>
        </div>
        <div className="flex gap-1 flex-wrap">
          {PRESETS.map((p, i) => (
            <button key={i} onClick={() => reset(i)}
              className={`px-2.5 py-1 text-xs rounded-lg transition-all ${presetIdx === i ? 'bg-white text-emerald-700 font-semibold' : 'bg-emerald-500/40 text-white hover:bg-emerald-500/60'}`}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 p-4 space-y-4">
        {/* String header */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">原串：</span>
          {s.split('').map((c, i) => (
            <span key={i} className="font-mono text-sm">
              <span className="text-gray-400 dark:text-gray-500 text-[10px]">{i}</span>
              <span className={`ml-0.5 px-1.5 py-0.5 rounded text-white font-bold ${CELL_COLORS[i % CELL_COLORS.length]}`}>{c}</span>
            </span>
          ))}
        </div>

        {/* Round navigation */}
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={() => setRoundIdx(v => Math.max(0, v - 1))} disabled={roundIdx === 0}
            className="px-3 py-1 text-xs bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded-lg disabled:opacity-40">
            ← 上一轮
          </button>
          <div className="flex gap-1">
            {rounds.map((_, i) => (
              <button key={i} onClick={() => setRoundIdx(i)}
                className={`w-7 h-7 text-xs rounded-lg font-medium transition-all ${roundIdx === i ? 'bg-emerald-500 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 hover:bg-emerald-100 dark:hover:bg-emerald-900/30'}`}>
                {i}
              </button>
            ))}
          </div>
          <button onClick={() => setRoundIdx(v => Math.min(rounds.length - 1, v + 1))} disabled={roundIdx === rounds.length - 1}
            className="px-3 py-1 text-xs bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 disabled:opacity-40">
            下一轮 →
          </button>
          <span className="text-xs text-gray-400 dark:text-gray-500 ml-1">第 {roundIdx}/{rounds.length - 1} 轮</span>
        </div>

        {/* Current round info */}
        <div className="p-3 rounded-xl bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-700">
          <p className="text-xs font-semibold text-emerald-700 dark:text-emerald-300">{round.label}</p>
          {roundIdx > 0 && (
            <p className="text-[11px] text-emerald-600 dark:text-emerald-400 mt-1">
              🟡 黄色高亮 = 排名相比上一轮发生变化的位置（共 {changedPos.size} 个）
            </p>
          )}
        </div>

        {/* Rank table */}
        <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700">
          <table className="text-xs border-collapse w-full min-w-max">
            <thead>
              <tr className="bg-gray-50 dark:bg-gray-800">
                <th className="py-2 px-3 text-left text-gray-500 dark:text-gray-400 font-medium">SA 名次</th>
                <th className="py-2 px-3 text-left text-gray-500 dark:text-gray-400 font-medium">下标 i</th>
                <th className="py-2 px-3 text-left text-gray-500 dark:text-gray-400 font-medium">后缀</th>
                {roundIdx > 0 && (
                  <>
                    <th className="py-2 px-3 text-center text-gray-500 dark:text-gray-400 font-medium">key 前半</th>
                    <th className="py-2 px-3 text-center text-gray-500 dark:text-gray-400 font-medium">key 后半</th>
                  </>
                )}
                <th className="py-2 px-3 text-center text-gray-500 dark:text-gray-400 font-medium">新 rank</th>
              </tr>
            </thead>
            <tbody>
              {round.sa.map((pos, rank) => {
                const changed = changedPos.has(pos)
                return (
                  <tr key={pos} className={`border-t border-gray-100 dark:border-gray-800 transition-colors ${changed ? 'bg-yellow-50 dark:bg-yellow-900/20' : ''}`}>
                    <td className="py-1.5 px-3 font-mono text-gray-400">{rank}</td>
                    <td className="py-1.5 px-3">
                      <span className={`inline-block w-6 h-6 text-[11px] text-white rounded font-bold leading-6 text-center ${CELL_COLORS[pos % CELL_COLORS.length]}`}>{pos}</span>
                    </td>
                    <td className="py-1.5 px-3 font-mono text-gray-600 dark:text-gray-300 tracking-wide">{s.slice(pos)}</td>
                    {roundIdx > 0 && (
                      <>
                        <td className="py-1.5 px-3 text-center font-mono text-blue-600 dark:text-blue-400">{round.keys[rank][0]}</td>
                        <td className="py-1.5 px-3 text-center font-mono text-violet-600 dark:text-violet-400">
                          {round.keys[rank][1] === -1 ? '—' : round.keys[rank][1]}
                        </td>
                      </>
                    )}
                    <td className="py-1.5 px-3 text-center">
                      <span className={`inline-block px-2 py-0.5 rounded font-mono font-bold text-white ${changed ? 'bg-amber-500' : 'bg-emerald-500'}`}>
                        {round.rank[pos]}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* SA display */}
        {roundIdx === rounds.length - 1 && (
          <div className="p-3 rounded-xl bg-teal-50 dark:bg-teal-900/20 border border-teal-200 dark:border-teal-700">
            <p className="text-xs font-semibold text-teal-700 dark:text-teal-300 mb-2">✅ 排名全部唯一 → 构建完成</p>
            <div className="flex gap-1 flex-wrap">
              {round.sa.map((pos, rank) => (
                <div key={rank} className="text-center">
                  <div className={`w-9 h-9 flex items-center justify-center rounded-lg text-white font-mono font-bold text-sm ${CELL_COLORS[pos % CELL_COLORS.length]}`}>{pos}</div>
                  <div className="text-[10px] text-gray-400 mt-0.5">SA[{rank}]</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
