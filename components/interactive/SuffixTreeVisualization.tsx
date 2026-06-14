'use client'

import { useState, useMemo } from 'react'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface STNode { id: number; start: number; end: number; children: STNode[]; suffixIdx: number }

// ---------------------------------------------------------------------------
// Build naive suffix tree (compressed trie) for short strings
// ---------------------------------------------------------------------------
function buildSuffixTree(s: string): STNode {
  let idCnt = 0
  const n = s.length
  const makeNode = (start: number, end: number, suffixIdx = -1): STNode =>
    ({ id: idCnt++, start, end, children: [], suffixIdx })

  const root = makeNode(-1, -1)

  function insertSuffix(idx: number) {
    let node = root
    let i = idx

    while (i < n) {
      // find child that starts with s[i]
      const ch = node.children.find(c => s[c.start] === s[i])
      if (!ch) {
        const leaf = makeNode(i, n - 1, idx); leaf.end = n - 1
        node.children.push(leaf)
        return
      }
      // walk along edge
      let j = ch.start
      while (j <= ch.end && i < n && s[j] === s[i]) { j++; i++ }
      if (j > ch.end) { node = ch; continue }
      // split
      const split = makeNode(ch.start, j - 1)
      node.children = node.children.map(c => c === ch ? split : c)
      ch.start = j
      split.children.push(ch)
      if (i < n) { const leaf = makeNode(i, n - 1, idx); split.children.push(leaf) }
      else { split.suffixIdx = idx }
      return
    }
    node.suffixIdx = idx
  }

  for (let i = 0; i < n; i++) insertSuffix(i)
  return root
}

// ---------------------------------------------------------------------------
// Layout: assign (x, y) to each node
// ---------------------------------------------------------------------------
interface NodePos { node: STNode; x: number; y: number; label: string; edgeLabel: string }

function layoutTree(s: string, root: STNode): { nodes: NodePos[]; edges: { from: NodePos; to: NodePos }[] } {
  const nodes: NodePos[] = []
  const edges: { from: NodePos; to: NodePos }[] = []
  const W = 680, YGAP = 60, leafY: number[] = []

  // count leaves under each node
  function countLeaves(n: STNode): number {
    if (n.children.length === 0) return 1
    return n.children.reduce((s, c) => s + countLeaves(c), 0)
  }

  // assign x positions using leaf-span
  let leafX = 0
  function assign(n: STNode, depth: number): number {
    const pos: NodePos = { node: n, x: 0, y: depth * YGAP + 20, label: n.suffixIdx >= 0 && n.children.length === 0 ? `[${n.suffixIdx}]` : '', edgeLabel: n.start >= 0 ? s.slice(n.start, n.end + 1) : '' }
    nodes.push(pos)
    if (n.children.length === 0) {
      pos.x = leafX * (W / (s.length)) + 20
      leafX++
    } else {
      const childXs = n.children.map(c => assign(c, depth + 1))
      pos.x = (Math.min(...childXs) + Math.max(...childXs)) / 2
    }
    return pos.x
  }
  assign(root, 0)

  // build edge list
  const nodeMap = new Map<number, NodePos>(nodes.map(n => [n.node.id, n]))
  function buildEdges(n: STNode) {
    const from = nodeMap.get(n.id)!
    n.children.forEach(c => { edges.push({ from, to: nodeMap.get(c.id)! }); buildEdges(c) })
  }
  buildEdges(root)

  return { nodes, edges }
}

const EXAMPLES = [
  { label: 'banana', s: 'banana$' },
  { label: 'abcbc', s: 'abcbc$' },
  { label: 'aabaa', s: 'aabaa$' },
  { label: 'mississi', s: 'missi$' },
]

const LEAF_COLORS = [
  '#6366f1', '#8b5cf6', '#a855f7', '#ec4899',
  '#f43f5e', '#f97316', '#eab308', '#22c55e',
  '#14b8a6', '#0ea5e9',
]

export default function SuffixTreeVisualization() {
  const [exIdx, setExIdx] = useState(0)
  const [hovered, setHovered] = useState<number | null>(null)

  const { s } = EXAMPLES[exIdx]
  const root = useMemo(() => buildSuffixTree(s), [s])
  const { nodes, edges } = useMemo(() => layoutTree(s, root), [s, root])

  const H = Math.max(...nodes.map(n => n.y)) + 50

  return (
    <div className="rounded-2xl overflow-hidden border border-purple-200 dark:border-purple-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-violet-600 dark:from-purple-700 dark:to-violet-700 px-5 py-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-white font-bold text-base">🌲 压缩后缀树可视化</h3>
          <p className="text-purple-100 text-xs mt-0.5">每条边代表原始字符串的一段子串，叶节点标注后缀起始位置</p>
        </div>
        <div className="flex gap-1 flex-wrap">
          {EXAMPLES.map((e, i) => (
            <button key={i} onClick={() => setExIdx(i)}
              className={`px-2.5 py-1 text-xs rounded-lg transition-all ${exIdx === i ? 'bg-white text-purple-700 font-semibold' : 'bg-purple-400/40 text-white hover:bg-purple-400/60'}`}>
              {e.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 p-4">
        {/* Legend */}
        <div className="flex flex-wrap gap-3 mb-3 text-[11px] text-gray-500 dark:text-gray-400">
          <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-purple-500 inline-block"/><span>内部节点</span></span>
          <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-violet-400 inline-block"/><span>叶节点（后缀起点）</span></span>
          <span className="flex items-center gap-1.5"><span className="border-b-2 border-purple-400 w-4 inline-block"/><span>边标签 = 子串片段</span></span>
          <span className="text-gray-400">$ = 终止符（保证所有后缀都是叶）</span>
        </div>

        {/* SVG */}
        <div className="overflow-x-auto rounded-xl border border-purple-100 dark:border-purple-900 bg-purple-50/30 dark:bg-purple-950/20">
          <svg width={700} height={H + 30} className="block mx-auto">
            {/* Edges */}
            {edges.map(({ from, to }, i) => {
              const mx = (from.x + to.x) / 2
              const my = (from.y + to.y) / 2
              const isLeaf = to.node.children.length === 0
              return (
                <g key={i}>
                  <line x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                    stroke={isLeaf ? '#a78bfa' : '#9333ea'} strokeWidth={isLeaf ? 1.5 : 2} strokeDasharray={isLeaf ? '4 2' : undefined} />
                  {/* Edge label */}
                  <rect x={mx - to.edgeLabel.length * 3.5} y={my - 9} width={to.edgeLabel.length * 7 + 6} height={15}
                    rx={3} fill="white" className="drop-shadow-sm" fillOpacity={0.9} />
                  <text x={mx} y={my + 2} textAnchor="middle" fontSize={10} fontFamily="monospace"
                    fill="#7c3aed" fontWeight="600">{to.edgeLabel}</text>
                </g>
              )
            })}

            {/* Nodes */}
            {nodes.map((n, i) => {
              const isLeaf = n.node.children.length === 0
              const isRoot = n.node.start === -1
              const color = isLeaf ? LEAF_COLORS[n.node.suffixIdx % LEAF_COLORS.length] : '#9333ea'
              const isHov = hovered === n.node.id
              return (
                <g key={i} onMouseEnter={() => setHovered(n.node.id)} onMouseLeave={() => setHovered(null)} style={{ cursor: 'pointer' }}>
                  {isLeaf ? (
                    <rect x={n.x - 14} y={n.y - 10} width={28} height={20} rx={4}
                      fill={color} opacity={isHov ? 1 : 0.85} />
                  ) : (
                    <circle cx={n.x} cy={n.y} r={isRoot ? 8 : 7}
                      fill={color} opacity={isHov ? 1 : 0.9} />
                  )}
                  {isLeaf && (
                    <text x={n.x} y={n.y + 4} textAnchor="middle" fontSize={9} fill="white" fontWeight="700"
                      fontFamily="monospace">{n.label}</text>
                  )}
                  {/* Hover tooltip */}
                  {isHov && isLeaf && (
                    <g>
                      <rect x={n.x + 10} y={n.y - 22} width={100} height={20} rx={4} fill="#1e1b4b" opacity={0.92} />
                      <text x={n.x + 60} y={n.y - 8} textAnchor="middle" fontSize={9} fill="white">
                        后缀 {n.node.suffixIdx}: "{s.slice(n.node.suffixIdx)}"
                      </text>
                    </g>
                  )}
                </g>
              )
            })}
          </svg>
        </div>

        {/* String display */}
        <div className="mt-3 flex gap-0.5 justify-center">
          {s.split('').map((c, i) => (
            <span key={i} className={`w-7 h-7 text-xs font-mono rounded text-center leading-7 font-bold ${c === '$' ? 'bg-gray-200 dark:bg-gray-700 text-gray-500' : 'bg-purple-100 dark:bg-purple-900/40 text-purple-800 dark:text-purple-200'}`}>
              {c}
            </span>
          ))}
        </div>
        <div className="flex gap-0.5 justify-center mt-0.5">
          {s.split('').map((_, i) => (
            <span key={i} className="w-7 text-[10px] text-center text-gray-400">{i}</span>
          ))}
        </div>

        <p className="mt-3 text-[11px] text-gray-400 dark:text-gray-500 text-center">
          叶节点数 = 后缀数 = {s.length}；内部节点数 ≤ {s.length - 1}；总节点数 ≤ {2 * s.length - 1}
        </p>
      </div>
    </div>
  )
}
