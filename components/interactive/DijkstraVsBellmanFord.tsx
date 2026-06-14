'use client'
import React, { useState, useEffect, useRef } from 'react'
import { Play, Pause, SkipForward, RotateCcw, AlertTriangle, CheckCircle } from 'lucide-react'

// ── 示例图：s→a(2), s→b(4), b→a(-3) ──────────────────────────
// s(0), a(1), b(2)
// 真实最短路径：δ(s,s)=0, δ(s,a)=1（s→b→a = 4+(-3)=1）, δ(s,b)=4

const INF = 9999

// SVG positions
const POS = {
  s: { cx: 80,  cy: 120 },
  a: { cx: 290, cy: 50  },
  b: { cx: 290, cy: 190 },
}

// Graph edges
const EDGES_DISPLAY = [
  { from: 's', to: 'a', w: 2  },
  { from: 's', to: 'b', w: 4  },
  { from: 'b', to: 'a', w: -3 },
]

type PanelStep = {
  d: { s: number; a: number; b: number }
  confirmed: string[]
  activeEdge: { from: string; to: string } | null
  description: string
  isWrong?: boolean
  isCorrect?: boolean
}

type Step = {
  dijkstra: PanelStep
  bellmanFord: PanelStep
  label: string
}

const STEPS: Step[] = [
  {
    label: '初始状态',
    dijkstra: {
      d: { s: 0, a: INF, b: INF },
      confirmed: [],
      activeEdge: null,
      description: '初始化：d[s]=0，d[a]=d[b]=∞\n优先队列 Q = {s(0), a(∞), b(∞)}',
    },
    bellmanFord: {
      d: { s: 0, a: INF, b: INF },
      confirmed: [],
      activeEdge: null,
      description: '初始化：d[s]=0，d[a]=d[b]=∞\n将对所有边进行 |V|-1=2 轮松弛',
    },
  },
  {
    label: 'Round 1 / Step 1',
    dijkstra: {
      d: { s: 0, a: 2, b: 4 },
      confirmed: ['s'],
      activeEdge: { from: 's', to: 'a' },
      description: 'EXTRACT-MIN: s (d=0)\n松弛 s→a: 0+2=2 < ∞ ✓ → d[a]=2\n松弛 s→b: 0+4=4 < ∞ ✓ → d[b]=4',
    },
    bellmanFord: {
      d: { s: 0, a: 2, b: 4 },
      confirmed: [],
      activeEdge: { from: 's', to: 'a' },
      description: '第 1 轮：松弛所有边\ns→a: 0+2=2 < ∞ ✓ 更新\ns→b: 0+4=4 < ∞ ✓ 更新\nb→a: d[b](4)+(-3)=1 < 2 ✓ 更新！→ d[a]=1',
    },
  },
  {
    label: '关键分歧',
    dijkstra: {
      d: { s: 0, a: 2, b: 4 },
      confirmed: ['s', 'a'],
      activeEdge: { from: 'b', to: 'a' },
      description: '⚠ EXTRACT-MIN: a (d=2 最小)\n宣告 a 已确定！d[a] = 2\n（此时无法再处理 b→a 的负权边）',
      isWrong: true,
    },
    bellmanFord: {
      d: { s: 0, a: 1, b: 4 },
      confirmed: [],
      activeEdge: { from: 'b', to: 'a' },
      description: '关键一步：第 1 轮处理 b→a\nd[b]+w(b,a) = 4+(-3) = 1 < d[a]=2 ✓\n更新 d[a] = 1（发现了更短路径！）',
      isCorrect: true,
    },
  },
  {
    label: '第 2 轮/最终结果',
    dijkstra: {
      d: { s: 0, a: 2, b: 4 },
      confirmed: ['s', 'a', 'b'],
      activeEdge: null,
      description: '❌ Dijkstra 最终结果：\nd[s]=0  d[a]=2  d[b]=4\n\nd[a] 应为 1，但 Dijkstra 给出 2！\n（负权边使贪心失效）',
      isWrong: true,
    },
    bellmanFord: {
      d: { s: 0, a: 1, b: 4 },
      confirmed: ['s', 'a', 'b'],
      activeEdge: null,
      description: '✅ Bellman-Ford 最终结果（2轮无更新）：\nd[s]=0  d[a]=1  d[b]=4\n\n正确找到 s→b→a = 4+(-3) = 1！',
      isCorrect: true,
    },
  },
]

// ── SVG Panel for one algorithm ──────────────────────────────────
function GraphPanel({
  d, confirmed, activeEdge, description, isWrong, isCorrect, title, titleColor,
}: PanelStep & { title: string; titleColor: string }) {
  const fmtD = (v: number) => v === INF ? '∞' : v.toString()

  const getNodeStyle = (n: string) => {
    if (confirmed.includes(n) && n !== 's') {
      if (isWrong && n === 'a') return { fill: '#fca5a5', stroke: '#ef4444', textColor: '#7f1d1d' }
      return { fill: '#6ee7b7', stroke: '#10b981', textColor: '#064e3b' }
    }
    if (n === activeEdge?.to) return { fill: '#c4b5fd', stroke: '#7c3aed', textColor: '#3b0764' }
    if (n === activeEdge?.from) return { fill: '#fde68a', stroke: '#d97706', textColor: '#78350f' }
    return { fill: 'white', stroke: '#94a3b8', textColor: '#1e293b' }
  }

  return (
    <div className={`rounded-xl border-2 overflow-hidden
      ${isWrong ? 'border-red-300 dark:border-red-700' : isCorrect ? 'border-emerald-300 dark:border-emerald-600' : 'border-slate-200 dark:border-slate-700'}`}>
      {/* Panel header */}
      <div className={`px-4 py-2 text-white text-sm font-bold ${titleColor}`}>{title}</div>

      {/* SVG */}
      <div className="bg-slate-50 dark:bg-slate-800 p-2">
        <svg viewBox="0 0 380 240" className="w-full">
          {/* Arrow marker defs */}
          {EDGES_DISPLAY.map(e => {
            const isActive = activeEdge?.from === e.from && activeEdge?.to === e.to
            const color = isActive ? (e.w < 0 ? '#7c3aed' : '#10b981') : '#94a3b8'
            const id = `${title.replace(/\s/g, '')}-${e.from}-${e.to}`
            const f = POS[e.from as keyof typeof POS]
            const t = POS[e.to as keyof typeof POS]
            const dx = t.cx - f.cx, dy = t.cy - f.cy
            const len = Math.sqrt(dx * dx + dy * dy)
            const ux = dx / len, uy = dy / len
            const r = 22
            const perpX = -uy * 6, perpY = ux * 6
            const x1 = f.cx + ux * r + perpX
            const y1 = f.cy + uy * r + perpY
            const x2 = t.cx - ux * r + perpX
            const y2 = t.cy - uy * r + perpY
            const midX = (x1 + x2) / 2 + perpX * 1.5
            const midY = (y1 + y2) / 2 + perpY * 1.5
            const negative = e.w < 0
            return (
              <g key={e.from + e.to} opacity={isActive ? 1 : 0.4}>
                <defs>
                  <marker id={id} markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                    <path d="M0,0 L0,6 L8,3 z" fill={color} />
                  </marker>
                </defs>
                <line x1={x1} y1={y1} x2={x2} y2={y2}
                  stroke={color} strokeWidth={isActive ? 3 : 1.5}
                  strokeDasharray={negative ? '5,3' : 'none'}
                  markerEnd={`url(#${id})`} />
                <rect x={midX - 14} y={midY - 9} width={28} height={18} rx={4}
                  fill={negative ? '#fef3c7' : '#f0fdf4'} opacity={0.9} />
                <text x={midX} y={midY} textAnchor="middle" dominantBaseline="middle"
                  fontSize="11" fontWeight="800" fill={negative ? '#b45309' : '#059669'}
                  className="select-none">{e.w > 0 ? '+' : ''}{e.w}</text>
              </g>
            )
          })}

          {/* Nodes */}
          {(['s', 'a', 'b'] as const).map(n => {
            const { cx, cy } = POS[n]
            const { fill, stroke, textColor } = getNodeStyle(n)
            const dVal = fmtD(d[n])
            return (
              <g key={n}>
                <circle cx={cx} cy={cy} r={22}
                  fill={fill} stroke={stroke} strokeWidth={confirmed.includes(n) ? 3 : 2}
                  className="transition-all duration-400" />
                <text x={cx} y={cy - 4} textAnchor="middle" dominantBaseline="middle"
                  fontSize="14" fontWeight="800" fill={textColor}
                  className="select-none">{n}</text>
                <text x={cx} y={cy + 10} textAnchor="middle" dominantBaseline="middle"
                  fontSize="10" fill="#64748b"
                  className="select-none">{dVal}</text>
              </g>
            )
          })}

          {/* True shortest path label */}
          <text x={190} y={225} textAnchor="middle" fontSize="10" fill="#94a3b8"
            className="select-none">真实最短路：d[s]=0, d[a]=1, d[b]=4</text>
        </svg>
      </div>

      {/* Description */}
      <div className={`px-4 py-3 text-xs leading-5 font-mono border-t
        ${isWrong ? 'bg-red-50 dark:bg-red-950/30 text-red-800 dark:text-red-200 border-red-200 dark:border-red-800' :
          isCorrect ? 'bg-emerald-50 dark:bg-emerald-950/30 text-emerald-800 dark:text-emerald-200 border-emerald-200 dark:border-emerald-800' :
          'bg-slate-50 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700'}`}
      >
        {description.split('\n').map((line, i) => <div key={i}>{line}</div>)}
      </div>
    </div>
  )
}

// ── 主组件 ────────────────────────────────────────────────────────
export default function DijkstraVsBellmanFord() {
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isLast = step === STEPS.length - 1

  useEffect(() => {
    if (playing && !isLast) {
      timerRef.current = setTimeout(() => setStep(s => s + 1), 1800)
    } else if (isLast) setPlaying(false)
    return () => { if (timerRef.current) clearTimeout(timerRef.current) }
  }, [playing, step, isLast])

  const cur = STEPS[step]

  return (
    <div className="w-full max-w-4xl mx-auto my-6 rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-xl">
      {/* Header */}
      <div className="bg-gradient-to-r from-rose-600 via-red-500 to-orange-500 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold text-white">Dijkstra vs Bellman-Ford：负权边反例</h3>
            <p className="text-rose-100 text-sm mt-0.5">含负权边图 · 演示 Dijkstra 为何失败</p>
          </div>
          <div className="bg-white/20 rounded-lg px-3 py-1.5 text-white text-sm font-medium">
            {step + 1} / {STEPS.length}：{cur.label}
          </div>
        </div>
      </div>

      {/* Progress */}
      <div className="h-1.5 bg-slate-200 dark:bg-slate-700">
        <div className="h-full bg-gradient-to-r from-rose-400 to-orange-400 transition-all duration-500"
          style={{ width: `${(step / (STEPS.length - 1)) * 100}%` }} />
      </div>

      {/* Top notice */}
      <div className="bg-amber-50 dark:bg-amber-950/30 border-b border-amber-200 dark:border-amber-700 px-5 py-2.5">
        <div className="flex items-center gap-2 text-amber-800 dark:text-amber-200 text-sm">
          <AlertTriangle className="w-4 h-4 flex-shrink-0" />
          <span>图：s→a (权+2), s→b (权+4), b→a (权<strong className="font-bold">-3</strong>, 虚线). 真实最短路 s→b→a = 4+(−3) = <strong>1</strong>，但 Dijkstra 会给出 <strong>2</strong>（错误）</span>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        {/* Side by side panels */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <GraphPanel
            {...cur.dijkstra}
            title="Dijkstra（贪心 + 优先队列）"
            titleColor="bg-gradient-to-r from-orange-500 to-amber-500"
          />
          <GraphPanel
            {...cur.bellmanFord}
            title="Bellman-Ford（全边松弛）"
            titleColor="bg-gradient-to-r from-teal-600 to-cyan-500"
          />
        </div>

        {/* Final verdict */}
        {isLast && (
          <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="flex items-center gap-3 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-700 rounded-xl px-4 py-3">
              <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0" />
              <div>
                <div className="text-sm font-bold text-red-800 dark:text-red-200">Dijkstra 输出：d[a] = 2</div>
                <div className="text-xs text-red-600 dark:text-red-300">❌ 错误！正确答案是 1</div>
                <div className="text-xs text-red-500 dark:text-red-400 mt-1">原因：负权边使"贪心已确定"论断失效</div>
              </div>
            </div>
            <div className="flex items-center gap-3 bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-700 rounded-xl px-4 py-3">
              <CheckCircle className="w-5 h-5 text-emerald-500 flex-shrink-0" />
              <div>
                <div className="text-sm font-bold text-emerald-800 dark:text-emerald-200">Bellman-Ford 输出：d[a] = 1</div>
                <div className="text-xs text-emerald-600 dark:text-emerald-300">✅ 正确！（不需要边权非负）</div>
                <div className="text-xs text-emerald-500 dark:text-emerald-400 mt-1">代价：时间复杂度 O(VE) vs O((V+E)logV)</div>
              </div>
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="mt-4 flex items-center gap-2">
          <button onClick={() => { setPlaying(false); setStep(0) }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600 text-sm transition-colors">
            <RotateCcw className="w-3.5 h-3.5" /> 重置
          </button>
          <button onClick={() => setPlaying(p => !p)} disabled={isLast}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-rose-500 hover:bg-rose-600 text-white text-sm font-semibold transition-colors disabled:opacity-40">
            {playing ? <><Pause className="w-3.5 h-3.5" /> 暂停</> : <><Play className="w-3.5 h-3.5" /> 自动播放</>}
          </button>
          <button onClick={() => { setPlaying(false); if (!isLast) setStep(s => s + 1) }} disabled={isLast}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-100 dark:bg-rose-900/40 text-rose-700 dark:text-rose-300 hover:bg-rose-200 dark:hover:bg-rose-900/60 text-sm transition-colors disabled:opacity-40">
            <SkipForward className="w-3.5 h-3.5" /> 下一步
          </button>
        </div>
      </div>
    </div>
  )
}
