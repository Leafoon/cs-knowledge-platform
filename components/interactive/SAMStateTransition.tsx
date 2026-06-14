'use client'

import { useState, useMemo } from 'react'

// ---------------------------------------------------------------------------
// SAM construction (incremental, online)
// ---------------------------------------------------------------------------
interface SAMState {
  id: number; len: number; link: number
  next: Map<string, number>
  isClone: boolean; addedAt: number // which char step added it
}

interface SAMSnapshot {
  states: SAMState[];  last: number; charIdx: number
  newStates: number[]; cloneId: number | null
}

function buildSAMSnapshots(s: string): SAMSnapshot[] {
  const states: SAMState[] = [{ id: 0, len: 0, link: -1, next: new Map(), isClone: false, addedAt: -1 }]
  let last = 0
  const snapshots: SAMSnapshot[] = []

  function cloneState(src: SAMState, newLen: number, addedAt: number): SAMState {
    const st: SAMState = { id: states.length, len: newLen, link: src.link, next: new Map(src.next), isClone: true, addedAt }
    states.push(st)
    return st
  }

  for (let ci = 0; ci < s.length; ci++) {
    const c = s[ci]
    const cur: SAMState = { id: states.length, len: states[last].len + 1, link: -1, next: new Map(), isClone: false, addedAt: ci }
    states.push(cur)
    const newStates: number[] = [cur.id]
    let cloneId: number | null = null
    let p = last

    while (p !== -1 && !states[p].next.has(c)) {
      states[p].next.set(c, cur.id)
      p = states[p].link
    }

    if (p === -1) {
      cur.link = 0
    } else {
      const q = states[p].next.get(c)!
      if (states[q].len === states[p].len + 1) {
        cur.link = q
      } else {
        const clone = cloneState(states[q], states[p].len + 1, ci)
        cloneId = clone.id
        newStates.push(clone.id)
        while (p !== -1 && states[p].next.get(c) === q) {
          states[p].next.set(c, clone.id)
          p = states[p].link
        }
        states[q].link = clone.id
        cur.link = clone.id
      }
    }

    last = cur.id
    // Deep clone snapshot
    snapshots.push({
      states: states.map(st => ({ ...st, next: new Map(st.next) })),
      last: cur.id, charIdx: ci,
      newStates, cloneId
    })
  }

  return snapshots
}

// ---------------------------------------------------------------------------
// Layout: arrange SAM states in layers by len
// ---------------------------------------------------------------------------
function layoutSAM(states: SAMState[], W = 680): { id: number; x: number; y: number }[] {
  const layers = new Map<number, number[]>()
  states.forEach(st => {
    if (!layers.has(st.len)) layers.set(st.len, [])
    layers.get(st.len)!.push(st.id)
  })
  const sortedLens = [...layers.keys()].sort((a, b) => a - b)
  const YGAP = 80, layout: { id: number; x: number; y: number }[] = []
  sortedLens.forEach((len, yi) => {
    const ids = layers.get(len)!
    ids.forEach((id, xi) => {
      const gap = W / (ids.length + 1)
      layout.push({ id, x: gap * (xi + 1), y: yi * YGAP + 40 })
    })
  })
  return layout
}

const EXAMPLES = [
  { label: 'abab', s: 'abab' },
  { label: 'aab', s: 'aab' },
  { label: 'abcbc', s: 'abcbc' },
]

const STATE_COLORS = {
  normal: { fill: '#a78bfa', stroke: '#7c3aed', text: 'white' },
  clone: { fill: '#fb923c', stroke: '#ea580c', text: 'white' },
  last: { fill: '#34d399', stroke: '#059669', text: 'white' },
  initial: { fill: '#6366f1', stroke: '#4338ca', text: 'white' },
  dimmed: { fill: '#e5e7eb', stroke: '#9ca3af', text: '#6b7280' },
}

export default function SAMStateTransition() {
  const [exIdx, setExIdx] = useState(0)
  const [step, setStep] = useState(0) // 0 = empty, 1..n = after adding s[i]

  const { s } = EXAMPLES[exIdx]
  const snapshots = useMemo(() => buildSAMSnapshots(s), [s])

  const snap = step === 0 ? null : snapshots[step - 1]
  const states = snap ? snap.states : [{ id: 0, len: 0, link: -1, next: new Map(), isClone: false, addedAt: -1 }]
  const layout = useMemo(() => layoutSAM(states), [states])
  const posMap = useMemo(() => new Map(layout.map(p => [p.id, p])), [layout])
  const H = Math.max(...layout.map(p => p.y)) + 60

  function getColor(st: SAMState) {
    if (!snap) return st.id === 0 ? STATE_COLORS.initial : STATE_COLORS.normal
    if (snap.newStates.includes(st.id) && !st.isClone) return STATE_COLORS.last
    if (snap.cloneId === st.id) return STATE_COLORS.clone
    if (st.id === 0) return STATE_COLORS.initial
    return STATE_COLORS.normal
  }

  return (
    <div className="rounded-2xl overflow-hidden border border-rose-200 dark:border-rose-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-rose-500 to-pink-500 dark:from-rose-600 dark:to-pink-600 px-5 py-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-white font-bold text-base">🤖 SAM 状态转移构造动画</h3>
          <p className="text-rose-100 text-xs mt-0.5">逐字符扩展 SAM，观察新建状态、Clone 节点与后缀链接变化</p>
        </div>
        <div className="flex gap-1 flex-wrap">
          {EXAMPLES.map((e, i) => (
            <button key={i} onClick={() => { setExIdx(i); setStep(0) }}
              className={`px-2.5 py-1 text-xs rounded-lg transition-all ${exIdx === i ? 'bg-white text-rose-700 font-semibold' : 'bg-rose-400/40 text-white hover:bg-rose-400/60'}`}>
              {e.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 p-4 space-y-3">
        {/* String timeline */}
        <div className="flex items-center gap-1 flex-wrap">
          <span className="text-xs text-gray-500 dark:text-gray-400 mr-1">扩展进度：</span>
          {s.split('').map((c, i) => (
            <span key={i} className={`w-7 h-7 text-sm font-mono font-bold rounded text-center leading-7 transition-all ${
              i < step ? 'bg-rose-500 text-white' : i === step - 1 ? 'bg-rose-500 text-white ring-2 ring-rose-300' : 'bg-gray-100 dark:bg-gray-800 text-gray-400'
            }`}>{c}</span>
          ))}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={() => { setStep(0) }} className="px-2.5 py-1 text-xs bg-gray-100 dark:bg-gray-800 text-gray-500 rounded-lg">↺ 重置</button>
          <button onClick={() => setStep(v => Math.max(0, v - 1))} disabled={step === 0}
            className="px-2.5 py-1 text-xs bg-rose-100 dark:bg-rose-900/30 text-rose-700 dark:text-rose-300 rounded-lg disabled:opacity-40">← 上一字符</button>
          <button onClick={() => setStep(v => Math.min(s.length, v + 1))} disabled={step === s.length}
            className="px-3 py-1 text-xs bg-rose-500 text-white rounded-lg hover:bg-rose-600 disabled:opacity-40">
            {step < s.length ? `添加 '${s[step]}' →` : '已完成'}
          </button>
          <span className="text-xs text-gray-400">状态数：{states.length}</span>
        </div>

        {/* Info box */}
        {snap && (
          <div className="px-3 py-2 rounded-xl bg-rose-50 dark:bg-rose-900/20 border border-rose-200 dark:border-rose-700 text-xs leading-relaxed">
            <span className="font-semibold text-rose-700 dark:text-rose-300">第 {step} 步</span>
            <span className="text-rose-600 dark:text-rose-400">：添加字符 '{s[snap.charIdx]}'</span>
            <span className="text-gray-500 dark:text-gray-400 ml-2">
              {snap.cloneId !== null
                ? `→ 新建状态 #${snap.newStates.filter(id => !states.find(st => st.id === id && st.isClone))[0]}，Clone 状态 #${snap.cloneId}（橙色）`
                : `→ 新建状态 #${snap.newStates[0]}（绿色），无 Clone`}
            </span>
          </div>
        )}

        {/* Legend */}
        <div className="flex flex-wrap gap-3 text-[11px]">
          {[
            { color: '#6366f1', label: '初始态 #0' },
            { color: '#34d399', label: '本步新增态' },
            { color: '#fb923c', label: 'Clone 态' },
            { color: '#a78bfa', label: '旧状态' },
          ].map(({ color, label }) => (
            <span key={label} className="flex items-center gap-1.5 text-gray-500 dark:text-gray-400">
              <span className="w-3 h-3 rounded-full inline-block" style={{ background: color }} />
              {label}
            </span>
          ))}
          <span className="flex items-center gap-1.5 text-gray-500 dark:text-gray-400">
            <svg width="28" height="8"><line x1={0} y1={4} x2={18} y2={4} stroke="#9333ea" strokeWidth={1.5} strokeDasharray="4 2" /><polygon points="18,1 26,4 18,7" fill="#9333ea" /></svg>
            后缀链接（link）
          </span>
          <span className="flex items-center gap-1.5 text-gray-500 dark:text-gray-400">
            <svg width="28" height="8"><line x1={0} y1={4} x2={18} y2={4} stroke="#0ea5e9" strokeWidth={1.5} /><polygon points="18,1 26,4 18,7" fill="#0ea5e9" /></svg>
            转移边（next[c]）
          </span>
        </div>

        {/* SVG */}
        <div className="overflow-x-auto rounded-xl border border-rose-100 dark:border-rose-900 bg-rose-50/20 dark:bg-rose-950/10">
          <svg width={700} height={Math.max(H, 120)} className="block mx-auto">
            <defs>
              <marker id="arrowBlue" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
                <polygon points="0 0, 7 3.5, 0 7" fill="#0ea5e9" /></marker>
              <marker id="arrowPurple" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
                <polygon points="0 0, 7 3.5, 0 7" fill="#9333ea" /></marker>
            </defs>

            {/* Suffix links (dashed) */}
            {states.map(st => {
              if (st.link < 0) return null
              const from = posMap.get(st.id), to = posMap.get(st.link)
              if (!from || !to) return null
              const dx = to.x - from.x, dy = to.y - from.y
              const len = Math.sqrt(dx * dx + dy * dy)
              const ex = from.x + dx / len * 12, ey = from.y + dy / len * 12
              const tx = to.x - dx / len * 14, ty = to.y - dy / len * 14
              return (
                <line key={`link-${st.id}`} x1={ex} y1={ey} x2={tx} y2={ty}
                  stroke="#9333ea" strokeWidth={1.5} strokeDasharray="5 3" markerEnd="url(#arrowPurple)" opacity={0.7} />
              )
            })}

            {/* Transition edges (solid, labeled) */}
            {states.flatMap(st =>
              [...st.next.entries()].map(([c, tid]) => {
                const from = posMap.get(st.id), to = posMap.get(tid)
                if (!from || !to) return null
                const dx = to.x - from.x, dy = to.y - from.y
                const len = Math.sqrt(dx * dx + dy * dy) || 1
                const ex = from.x + dx / len * 13, ey = from.y + dy / len * 13
                const tx = to.x - dx / len * 15, ty = to.y - dy / len * 15
                const mx = (from.x + to.x) / 2, my = (from.y + to.y) / 2
                return (
                  <g key={`edge-${st.id}-${c}-${tid}`}>
                    <line x1={ex} y1={ey} x2={tx} y2={ty}
                      stroke="#0ea5e9" strokeWidth={1.5} markerEnd="url(#arrowBlue)" opacity={0.6} />
                    <text x={mx + 4} y={my - 2} fontSize={9} fill="#0ea5e9" fontWeight="bold" fontFamily="monospace">{c}</text>
                  </g>
                )
              })
            )}

            {/* Nodes */}
            {states.map(st => {
              const pos = posMap.get(st.id)
              if (!pos) return null
              const col = getColor(st)
              return (
                <g key={st.id}>
                  <circle cx={pos.x} cy={pos.y} r={13} fill={col.fill} stroke={col.stroke} strokeWidth={2} />
                  <text x={pos.x} y={pos.y + 4} textAnchor="middle" fontSize={10} fontWeight="bold" fill={col.text}
                    fontFamily="monospace">#{st.id}</text>
                  <text x={pos.x} y={pos.y + 22} textAnchor="middle" fontSize={8} fill="#6b7280">
                    len={st.len}
                  </text>
                </g>
              )
            })}
          </svg>
        </div>

        {/* State table */}
        {states.length > 1 && (
          <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700">
            <table className="text-xs w-full border-collapse min-w-max">
              <thead>
                <tr className="bg-gray-50 dark:bg-gray-800 text-gray-500">
                  <th className="py-1.5 px-3 text-left font-medium">state</th>
                  <th className="py-1.5 px-3 text-left font-medium">len</th>
                  <th className="py-1.5 px-3 text-left font-medium">link</th>
                  <th className="py-1.5 px-3 text-left font-medium">next</th>
                  <th className="py-1.5 px-3 text-left font-medium">备注</th>
                </tr>
              </thead>
              <tbody>
                {states.map(st => (
                  <tr key={st.id} className={`border-t border-gray-100 dark:border-gray-800 ${
                    snap?.newStates.includes(st.id) && !st.isClone ? 'bg-emerald-50 dark:bg-emerald-900/20' :
                    snap?.cloneId === st.id ? 'bg-orange-50 dark:bg-orange-900/20' : ''
                  }`}>
                    <td className="py-1 px-3 font-mono font-bold text-rose-600 dark:text-rose-400">#{st.id}</td>
                    <td className="py-1 px-3 font-mono">{st.len}</td>
                    <td className="py-1 px-3 font-mono text-purple-600 dark:text-purple-400">{st.link >= 0 ? `#${st.link}` : '—'}</td>
                    <td className="py-1 px-3 font-mono text-sky-600 dark:text-sky-400 text-[10px]">
                      {[...st.next.entries()].map(([c, t]) => `${c}→#${t}`).join(', ') || '—'}
                    </td>
                    <td className="py-1 px-3 text-[10px] text-gray-400">
                      {st.id === 0 ? '初始态' : st.isClone ? '🟠 Clone' : snap?.newStates.includes(st.id) ? '🟢 新建' : ''}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
