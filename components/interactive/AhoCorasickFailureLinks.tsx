'use client'

import { useState, useMemo } from 'react'

// ─── AC 自动机节点 ─────────────────────────────────────────
interface ACNode {
  id: number
  char: string
  children: Map<string, ACNode>
  fail: number      // fail 链接指向的节点 id（-1 = 未计算）
  isEnd: boolean
  patterns: string[]
  depth: number
  parentId: number | null
}

// ─── BFS 构建步骤 ────────────────────────────────────────────
interface BuildStep {
  processedId: number       // 被处理的节点
  newFailId: number         // fail 链接指向的节点
  desc: string
}

// ─── 构建 AC 自动机 ──────────────────────────────────────────
function buildAC(patterns: string[]) {
  let idCounter = 0
  const root: ACNode = {
    id: idCounter++, char: 'ε', children: new Map(),
    fail: -1, isEnd: false, patterns: [], depth: 0, parentId: null
  }
  const nodeById = new Map<number, ACNode>([[root.id, root]])

  // 插入所有模式串
  for (const pat of patterns) {
    let node = root
    for (const ch of pat) {
      if (!node.children.has(ch)) {
        const n: ACNode = {
          id: idCounter++, char: ch, children: new Map(),
          fail: -1, isEnd: false, patterns: [], depth: node.depth + 1,
          parentId: node.id
        }
        node.children.set(ch, n)
        nodeById.set(n.id, n)
      }
      node = node.children.get(ch)!
    }
    node.isEnd = true
    node.patterns.push(pat)
  }

  // BFS 建 fail 链接，并记录步骤
  const steps: BuildStep[] = []
  const queue: ACNode[] = []

  for (const child of root.children.values()) {
    child.fail = root.id
    queue.push(child)
    steps.push({
      processedId: child.id,
      newFailId: root.id,
      desc: `深度 1 节点 "${child.char}"：fail → root（深度1节点的fail链接均指向根）`
    })
  }

  let head = 0
  while (head < queue.length) {
    const u = queue[head++]

    for (const [ch, v] of u.children) {
      let f = nodeById.get(u.fail)!
      while (f.id !== root.id && !f.children.has(ch)) f = nodeById.get(f.fail)!

      if (f.children.has(ch) && f.children.get(ch)!.id !== v.id) {
        v.fail = f.children.get(ch)!.id
      } else {
        v.fail = root.id
      }

      const failTarget = nodeById.get(v.fail)!
      steps.push({
        processedId: v.id,
        newFailId: v.fail,
        desc: `节点 "${v.char}"（路径 ${getPath(nodeById, v)}）：fail → "${failTarget.char}"（路径 ${getPath(nodeById, failTarget)} 或根）`
      })

      queue.push(v)
    }
  }

  return { root, nodeById, steps }
}

function getPath(nodeById: Map<number, ACNode>, node: ACNode): string {
  const chars: string[] = []
  let n: ACNode | undefined = node
  while (n && n.parentId !== null) {
    chars.unshift(n.char)
    n = nodeById.get(n.parentId)
  }
  return chars.join('') || 'ε'
}

// ─── 布局 ────────────────────────────────────────────────────
interface LayoutNode { id: number; char: string; isEnd: boolean; x: number; y: number; patterns: string[] }

function buildLayout(root: ACNode, nodeById: Map<number, ACNode>): Map<number, LayoutNode> {
  const layout = new Map<number, LayoutNode>()
  const LEVEL_H = 80

  function countLeaves(n: ACNode): number {
    if (n.children.size === 0) return 1
    let s = 0
    for (const c of n.children.values()) s += countLeaves(c)
    return s
  }

  function assign(n: ACNode, left: number, right: number) {
    const x = (left + right) / 2
    const y = n.depth * LEVEL_H + 44
    layout.set(n.id, { id: n.id, char: n.char, isEnd: n.isEnd, x, y, patterns: n.patterns })
    const children = [...n.children.values()]
    const total = children.reduce((s, c) => s + countLeaves(c), 0) || 1
    let cur = left
    for (const child of children) {
      const w = (countLeaves(child) / total) * (right - left)
      assign(child, cur, cur + w)
      cur += w
    }
  }

  assign(root, 30, 630)
  return layout
}

// ─── 预设 ────────────────────────────────────────────────────
const PRESETS = [
  { label: 'he/she/his/hers', patterns: ['he', 'she', 'his', 'hers'] },
  { label: '关键词过滤', patterns: ['bad', 'ban', 'band', 'base'] },
  { label: 'DNA 片段', patterns: ['AT', 'ATG', 'GC', 'GCA', 'AG'] },
  { label: '单模式退化', patterns: ['abab', 'ab'] },
]

// ─── 绘制曲线失败链接 ─────────────────────────────────────────
function FailArrow({ x1, y1, x2, y2, animated }: { x1: number; y1: number; x2: number; y2: number; animated: boolean }) {
  const dx = x2 - x1, dy = y2 - y1
  const cx = (x1 + x2) / 2 - dy * 0.35
  const cy = (y1 + y2) / 2 + Math.abs(dx) * 0.2
  const path = `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`
  const color = animated ? '#f43f5e' : '#fda4af'
  return (
    <g>
      <path d={path} fill="none" stroke={color} strokeWidth={animated ? 2 : 1.5}
        strokeDasharray="5 3" markerEnd="url(#arrow-fail)"
        className="transition-all duration-500" opacity={animated ? 1 : 0.6} />
    </g>
  )
}

export default function AhoCorasickFailureLinks() {
  const [presetIdx, setPresetIdx] = useState(0)
  const [stepIdx, setStepIdx] = useState(-1)   // -1 = 只显示 Trie，0+ = 逐步显示 fail 链

  const { root, nodeById, steps } = useMemo(
    () => buildAC(PRESETS[presetIdx].patterns),
    [presetIdx]
  )
  const layout = useMemo(() => buildLayout(root, nodeById), [root, nodeById])

  // 当前已显示的 fail 链接（step -1 = 无，step i = 显示 step 0..i）
  const visibleFailLinks = useMemo(() => {
    if (stepIdx < 0) return []
    return steps.slice(0, stepIdx + 1).map(s => ({ from: s.processedId, to: s.newFailId }))
  }, [steps, stepIdx])

  const currentStep = stepIdx >= 0 && stepIdx < steps.length ? steps[stepIdx] : null

  const maxDepth = Math.max(...[...layout.values()].map(n => n.y), 0)
  const SVG_H = maxDepth + 80

  return (
    <div className="rounded-2xl overflow-hidden border border-violet-200 dark:border-violet-800 shadow-lg">
      {/* 头部 */}
      <div className="bg-gradient-to-r from-violet-600 to-purple-600 dark:from-violet-700 dark:to-purple-700 px-5 py-4">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <h3 className="text-white font-bold text-base">🔗 AC 自动机失败链接构建</h3>
            <p className="text-violet-100 text-xs mt-0.5">Trie + BFS 逐层计算 fail 链接 = KMP 在多模式上的推广</p>
          </div>
          <div className="flex gap-1 bg-violet-800/40 rounded-lg p-0.5">
            {PRESETS.map((p, i) => (
              <button key={i} onClick={() => { setPresetIdx(i); setStepIdx(-1) }}
                className={`px-2.5 py-1 text-xs rounded-md transition-all ${
                  presetIdx === i ? 'bg-white text-violet-700 font-semibold' : 'text-violet-100 hover:text-white'}`}>
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 p-4">
        {/* 控制栏 */}
        <div className="flex items-center gap-2 mb-3 flex-wrap">
          <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">
            BFS 步骤 {Math.max(0, stepIdx + 1)}/{steps.length}：
          </span>
          <button onClick={() => setStepIdx(-1)}
            className="px-2.5 py-1 text-xs bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700">
            ↺ 重置
          </button>
          <button onClick={() => setStepIdx(v => Math.max(-1, v - 1))}
            disabled={stepIdx < 0}
            className="px-2.5 py-1 text-xs bg-violet-100 dark:bg-violet-900/30 text-violet-600 dark:text-violet-300 rounded-lg hover:bg-violet-200 disabled:opacity-40">
            ← 上一步
          </button>
          <button onClick={() => setStepIdx(v => Math.min(steps.length - 1, v + 1))}
            disabled={stepIdx >= steps.length - 1}
            className="px-3 py-1 text-xs bg-violet-600 text-white rounded-lg hover:bg-violet-700 disabled:opacity-40">
            建立 fail 链接 →
          </button>
          <button onClick={() => setStepIdx(steps.length - 1)}
            className="px-2.5 py-1 text-xs bg-violet-100 dark:bg-violet-900/30 text-violet-600 dark:text-violet-300 rounded-lg hover:bg-violet-200">
            全部完成 ⤵
          </button>
        </div>

        {/* 当前步骤说明 */}
        {currentStep && (
          <div className="mb-3 p-2.5 bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-700 rounded-lg">
            <p className="text-xs text-violet-700 dark:text-violet-300">
              <span className="font-semibold">步骤 {stepIdx + 1}：</span>{currentStep.desc}
            </p>
          </div>
        )}

        {!currentStep && stepIdx < 0 && (
          <div className="mb-3 p-2.5 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg">
            <p className="text-xs text-gray-500 dark:text-gray-400">
              已构建 Trie（含 {[...layout.keys()].length} 个节点，{PRESETS[presetIdx].patterns.length} 个模式串）。
              点击"建立 fail 链接"开始 BFS 逐层构建失败链接（虚线箭头）→
            </p>
          </div>
        )}

        {/* 模式串 */}
        <div className="flex gap-2 flex-wrap mb-2">
          {PRESETS[presetIdx].patterns.map(p => (
            <span key={p} className="px-2 py-0.5 text-xs font-mono font-semibold bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-300 border border-violet-200 dark:border-violet-700 rounded-full">
              "{p}"
            </span>
          ))}
        </div>

        {/* SVG 可视化 */}
        <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
          <svg width="660" height={Math.max(Number(SVG_H), 120)} viewBox={`0 0 660 ${Math.max(Number(SVG_H), 120)}`} className="w-full">
            <defs>
              <marker id="arrow-fail" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                <path d="M0,0 L0,6 L8,3 z" fill="#f43f5e" />
              </marker>
              <marker id="arrow-tree" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
                <path d="M0,0 L0,6 L7,3 z" fill="#c4b5fd" />
              </marker>
            </defs>

            {/* 树边 */}
            {[...nodeById.values()].filter(n => n.parentId !== null).map(n => {
              const p = layout.get(n.parentId!)
              const c = layout.get(n.id)
              if (!p || !c) return null
              return (
                <line key={`te-${n.id}`} x1={p.x} y1={p.y} x2={c.x} y2={c.y}
                  stroke="#c4b5fd" strokeWidth={2} />
              )
            })}

            {/* fail 链接（弯曲虚线）*/}
            {visibleFailLinks.map((fl, i) => {
              const from = layout.get(fl.from)
              const to = layout.get(fl.to)
              if (!from || !to) return null
              const isLatest = i === visibleFailLinks.length - 1
              return (
                <FailArrow key={`fl-${i}`}
                  x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                  animated={isLatest}
                />
              )
            })}

            {/* 节点 */}
            {[...layout.values()].map(n => {
              const isRoot = n.id === root.id
              const isCurrent = currentStep?.processedId === n.id
              const isFailTarget = currentStep?.newFailId === n.id && stepIdx >= 0
              const fill = isCurrent ? '#7c3aed' : isFailTarget ? '#f43f5e' : isRoot ? '#6b7280' : n.isEnd ? '#7c3aed' : '#e9d5ff'
              const textFill = isCurrent || isFailTarget || isRoot || n.isEnd ? '#fff' : '#4c1d95'
              return (
                <g key={n.id} className="transition-all duration-300">
                  <circle cx={n.x} cy={n.y} r={n.isEnd ? 20 : 17}
                    fill={fill} stroke={n.isEnd ? '#7c3aed' : '#a78bfa'} strokeWidth={2}
                    className="transition-all duration-300"
                  />
                  <text x={n.x} y={n.y + 1} textAnchor="middle" dominantBaseline="middle"
                    fill={textFill} fontSize={13} fontWeight="700" fontFamily="monospace"
                    className="select-none">
                    {n.char}
                  </text>
                  {n.patterns.length > 0 && (
                    <text x={n.x} y={n.y + 32} textAnchor="middle"
                      fill="#7c3aed" fontSize={9} fontFamily="sans-serif" className="select-none">
                      [{n.patterns.join(', ')}]
                    </text>
                  )}
                </g>
              )
            })}
          </svg>
        </div>

        {/* 图例 */}
        <div className="mt-3 flex gap-4 flex-wrap text-xs text-gray-500 dark:text-gray-400">
          <span className="flex items-center gap-1.5">
            <span className="w-4 h-4 rounded-full bg-violet-700 inline-block"></span>模式结尾/当前节点
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-4 h-4 rounded-full bg-rose-500 inline-block"></span>fail 目标节点
          </span>
          <span className="flex items-center gap-1.5">
            <svg width="28" height="10"><line x1="2" y1="5" x2="22" y2="5" stroke="#f43f5e" strokeWidth="2" strokeDasharray="4 2"/></svg>
            失败链接（虚线）
          </span>
          <span className="flex items-center gap-1.5">
            <svg width="28" height="10"><line x1="2" y1="5" x2="22" y2="5" stroke="#c4b5fd" strokeWidth="2"/></svg>
            Trie 树边
          </span>
        </div>
      </div>
    </div>
  )
}
