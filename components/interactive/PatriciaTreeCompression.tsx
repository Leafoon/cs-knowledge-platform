'use client'

import { useState, useMemo, useCallback } from 'react'

// ─── 预设单词集 ────────────────────────────────────────────────
const PRESETS = [
  { label: 'apple系列', words: ['apple', 'apply', 'apt', 'app', 'application'] },
  { label: '水果前缀', words: ['banana', 'band', 'ban', 'bandana', 'bandit'] },
  { label: '路由前缀', words: ['cat', 'car', 'card', 'cart', 'carry', 'care'] },
  { label: '混合词集', words: ['he', 'her', 'here', 'hero', 'him', 'his'] },
]

// ─── 标准 Trie 节点 ────────────────────────────────────────────
interface TrieNode {
  id: number
  char: string
  isEnd: boolean
  children: Map<string, TrieNode>
  depth: number
  parent: TrieNode | null
  childIndex: number   // 在父节点的 children 中的顺序
}

// ─── 压缩 Trie (Radix Tree) 节点 ─────────────────────────────
interface RadixNode {
  id: number
  label: string        // 边上的标签（可以是多个字符）
  isEnd: boolean
  children: Map<string, RadixNode>
  depth: number        // 路径深度层数（用于布局）
  parent: RadixNode | null
  childIndex: number
}

let globalId = 0

// ─── 构建标准 Trie ────────────────────────────────────────────
function buildTrie(words: string[]): { root: TrieNode; nodeCount: number } {
  globalId = 0
  const root: TrieNode = { id: globalId++, char: '', isEnd: false, children: new Map(), depth: 0, parent: null, childIndex: 0 }

  for (const word of [...words].sort()) {
    let node = root
    for (const ch of word) {
      if (!node.children.has(ch)) {
        const child: TrieNode = { id: globalId++, char: ch, isEnd: false, children: new Map(), depth: node.depth + 1, parent: node, childIndex: node.children.size }
        node.children.set(ch, child)
      }
      node = node.children.get(ch)!
    }
    node.isEnd = true
  }

  let count = 0
  const count_ = (n: TrieNode) => { count++; n.children.forEach(count_) }
  count_(root)
  return { root, nodeCount: count }
}

// ─── 构建压缩 Trie ────────────────────────────────────────────
function buildRadixTrie(words: string[]): { root: RadixNode; nodeCount: number } {
  globalId = 100000
  const root: RadixNode = { id: globalId++, label: '', isEnd: false, children: new Map(), depth: 0, parent: null, childIndex: 0 }

  function insert(node: RadixNode, suffix: string) {
    if (!suffix) { node.isEnd = true; return }
    const firstChar = suffix[0]
    if (!node.children.has(firstChar)) {
      const child: RadixNode = { id: globalId++, label: suffix, isEnd: true, children: new Map(), depth: node.depth + 1, parent: node, childIndex: node.children.size }
      node.children.set(firstChar, child)
      return
    }
    const child = node.children.get(firstChar)!
    const childLabel = child.label
    // 求公共前缀
    let i = 0
    while (i < suffix.length && i < childLabel.length && suffix[i] === childLabel[i]) i++

    if (i === childLabel.length) {
      // suffix 包含 childLabel 作为前缀
      insert(child, suffix.slice(i))
    } else {
      // 需要分裂：split at i
      const commonPrefix = childLabel.slice(0, i)
      const childRemainder = childLabel.slice(i)
      const suffixRemainder = suffix.slice(i)

      // 新的中间节点
      const mid: RadixNode = { id: globalId++, label: commonPrefix, isEnd: false, children: new Map(), depth: node.depth + 1, parent: node, childIndex: child.childIndex }
      // 更新 child
      child.label = childRemainder
      child.parent = mid
      child.depth = mid.depth + 1
      child.childIndex = 0
      mid.children.set(childRemainder[0], child)
      node.children.set(firstChar, mid)
      mid.parent = node
      if (suffixRemainder) {
        const leaf: RadixNode = { id: globalId++, label: suffixRemainder, isEnd: true, children: new Map(), depth: mid.depth + 1, parent: mid, childIndex: 1 }
        mid.children.set(suffixRemainder[0], leaf)
      } else {
        mid.isEnd = true
      }
    }
  }

  for (const word of [...words].sort()) insert(root, word)

  // 重新计算 childIndex
  function fixIndex(n: RadixNode) {
    let i = 0
    for (const ch of n.children.values()) { ch.childIndex = i++; fixIndex(ch) }
  }
  fixIndex(root)

  let count = 0
  const count_ = (n: RadixNode) => { count++; n.children.forEach(count_) }
  count_(root)
  return { root, nodeCount: count }
}

// ─── SVG 布局（DFS 排布，给每个叶分配 x 槽）─────────────────
const TRIENODEH = 50
const TRIENODER = 16
const HGAP = 36
const RADIXNODEH = 56
const RADIXNODER = 16

function layoutTrie(root: TrieNode): Map<number, { x: number; y: number }> {
  const pos = new Map<number, { x: number; y: number }>()
  let leafCnt = { v: 0 }
  function dfs(n: TrieNode) {
    const children = [...n.children.values()]
    if (children.length === 0) { const x = leafCnt.v++ * HGAP; pos.set(n.id, { x, y: n.depth * TRIENODEH + TRIENODER }); return }
    children.forEach(dfs)
    const xs = children.map(c => pos.get(c.id)!.x)
    pos.set(n.id, { x: (xs[0] + xs[xs.length - 1]) / 2, y: n.depth * TRIENODEH + TRIENODER })
  }
  dfs(root)
  return pos
}

function layoutRadix(root: RadixNode): Map<number, { x: number; y: number }> {
  const pos = new Map<number, { x: number; y: number }>()
  let leafCnt = { v: 0 }
  function dfs(n: RadixNode) {
    const children = [...n.children.values()]
    if (children.length === 0) { const x = leafCnt.v++ * HGAP + HGAP / 2; pos.set(n.id, { x, y: n.depth * RADIXNODEH + RADIXNODER }); return }
    children.forEach(dfs)
    const xs = children.map(c => pos.get(c.id)!.x)
    pos.set(n.id, { x: (xs[0] + xs[xs.length - 1]) / 2, y: n.depth * RADIXNODEH + RADIXNODER })
  }
  dfs(root)
  return pos
}

// ─── 收集节点 ──────────────────────────────────────────────
function collectTrie(root: TrieNode): TrieNode[] {
  const r: TrieNode[] = []
  const dfs = (n: TrieNode) => { r.push(n); n.children.forEach(dfs) }
  dfs(root)
  return r
}
function collectRadix(root: RadixNode): RadixNode[] {
  const r: RadixNode[] = []
  const dfs = (n: RadixNode) => { r.push(n); n.children.forEach(dfs) }
  dfs(root)
  return r
}

// ─── 主组件 ───────────────────────────────────────────────────
export default function PatriciaTreeCompression() {
  const [presetIdx, setPresetIdx] = useState(0)
  const [showTrie, setShowTrie] = useState(true)
  const [showRadix, setShowRadix] = useState(true)
  const [highlight, setHighlight] = useState<string | null>(null)

  const words = PRESETS[presetIdx].words

  const { root: trieRoot, nodeCount: trieCount } = useMemo(() => buildTrie(words), [words])
  const { root: radixRoot, nodeCount: radixCount } = useMemo(() => buildRadixTrie(words), [words])

  const triePosMap = useMemo(() => layoutTrie(trieRoot), [trieRoot])
  const radixPosMap = useMemo(() => layoutRadix(radixRoot), [radixRoot])
  const trieNodes = useMemo(() => collectTrie(trieRoot), [trieRoot])
  const radixNodes = useMemo(() => collectRadix(radixRoot), [radixRoot])

  // SVG 尺寸
  const trieW = useMemo(() => Math.max(200, ...[...triePosMap.values()].map(p => p.x)) + HGAP * 2, [triePosMap])
  const trieH = useMemo(() => Math.max(200, ...[...triePosMap.values()].map(p => p.y)) + TRIENODER + 20, [triePosMap])
  const radixW = useMemo(() => Math.max(200, ...[...radixPosMap.values()].map(p => p.x)) + HGAP * 2, [radixPosMap])
  const radixH = useMemo(() => Math.max(200, ...[...radixPosMap.values()].map(p => p.y)) + RADIXNODER + 20, [radixPosMap])

  const saved = Math.max(0, trieCount - radixCount)
  const ratio = trieCount > 0 ? ((saved / trieCount) * 100).toFixed(0) : 0

  return (
    <div className="rounded-2xl overflow-hidden border border-teal-200 dark:border-teal-800 shadow-lg">
      {/* 头部 */}
      <div className="bg-gradient-to-r from-teal-500 to-sky-500 dark:from-teal-600 dark:to-sky-600 px-5 py-4">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <h3 className="text-white font-bold text-base">🗜️ 压缩 Trie（Patricia Tree / Radix Tree）</h3>
            <p className="text-teal-100 text-xs mt-0.5">合并单链节点 → 大幅压缩空间·对比标准 Trie</p>
          </div>
          <div className="flex gap-1 flex-wrap">
            {PRESETS.map((p, i) => (
              <button key={i} onClick={() => { setPresetIdx(i); setHighlight(null) }}
                className={`px-2.5 py-1 text-xs rounded-lg transition-all ${presetIdx === i ? 'bg-white text-teal-700 font-semibold' : 'bg-teal-400/40 text-white hover:bg-teal-400/60'}`}>
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 p-4">
        {/* 单词列表 */}
        <div className="flex flex-wrap gap-2 mb-3">
          {words.map(w => (
            <span key={w} onMouseEnter={() => setHighlight(w)} onMouseLeave={() => setHighlight(null)}
              className="px-2.5 py-1 bg-teal-50 dark:bg-teal-900/30 border border-teal-200 dark:border-teal-700 text-teal-700 dark:text-teal-300 text-xs font-mono rounded-lg cursor-default select-none">
              "{w}"
            </span>
          ))}
        </div>

        {/* 统计对比卡 */}
        <div className="grid grid-cols-3 gap-3 mb-4">
          <div className="p-3 rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 text-center">
            <div className="text-xs text-blue-500 dark:text-blue-400 mb-1">标准 Trie 节点数</div>
            <div className="text-2xl font-bold text-blue-700 dark:text-blue-300 font-mono">{trieCount}</div>
          </div>
          <div className="p-3 rounded-xl bg-teal-50 dark:bg-teal-900/20 border border-teal-200 dark:border-teal-800 text-center">
            <div className="text-xs text-teal-500 dark:text-teal-400 mb-1">压缩后节点数</div>
            <div className="text-2xl font-bold text-teal-700 dark:text-teal-300 font-mono">{radixCount}</div>
          </div>
          <div className="p-3 rounded-xl bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 text-center">
            <div className="text-xs text-emerald-500 dark:text-emerald-400 mb-1">节点节省</div>
            <div className="text-2xl font-bold text-emerald-700 dark:text-emerald-300 font-mono">{ratio}%</div>
          </div>
        </div>

        {/* 视图切换 */}
        <div className="flex gap-2 mb-3">
          <button onClick={() => setShowTrie(v => !v)}
            className={`px-3 py-1 text-xs rounded-lg border transition-all ${showTrie ? 'bg-blue-500 text-white border-blue-500' : 'border-gray-300 dark:border-gray-700 text-gray-500'}`}>
            标准 Trie {showTrie ? '✓' : '○'}
          </button>
          <button onClick={() => setShowRadix(v => !v)}
            className={`px-3 py-1 text-xs rounded-lg border transition-all ${showRadix ? 'bg-teal-500 text-white border-teal-500' : 'border-gray-300 dark:border-gray-700 text-gray-500'}`}>
            压缩 Trie {showRadix ? '✓' : '○'}
          </button>
        </div>

        {/* 两图并排 */}
        <div className="flex gap-3 flex-wrap lg:flex-nowrap">
          {/* 标准 Trie */}
          {showTrie && (
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-xs font-semibold text-blue-600 dark:text-blue-400">标准 Trie</span>
                <span className="text-[10px] text-gray-400">每个字符一个节点</span>
              </div>
              <div className="overflow-x-auto rounded-xl border border-blue-200 dark:border-blue-800 bg-blue-50/50 dark:bg-blue-900/10">
                <svg width={trieW + 32} height={trieH + 16} viewBox={`-16 -8 ${trieW + 32} ${trieH + 16}`}
                  style={{ minWidth: Math.min(trieW + 32, 280) }}>
                  {/* 边 */}
                  {trieNodes.filter(n => n.parent !== null).map(n => {
                    const p = n.parent!
                    const { x: x1, y: y1 } = triePosMap.get(p.id)!
                    const { x: x2, y: y2 } = triePosMap.get(n.id)!
                    return <line key={`e-${n.id}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#93c5fd" strokeWidth={1.5} />
                  })}
                  {/* 节点 */}
                  {trieNodes.map(n => {
                    const { x, y } = triePosMap.get(n.id)!
                    const isRoot = n.parent === null
                    return (
                      <g key={n.id}>
                        <circle cx={x} cy={y} r={TRIENODER}
                          fill={isRoot ? '#3b82f6' : n.isEnd ? '#dbeafe' : '#eff6ff'}
                          stroke={n.isEnd ? '#3b82f6' : '#93c5fd'} strokeWidth={n.isEnd ? 2 : 1}
                          className="dark:fill-opacity-80" />
                        <text x={x} y={y + 1} textAnchor="middle" dominantBaseline="middle"
                          fill={isRoot ? '#fff' : n.isEnd ? '#1d4ed8' : '#60a5fa'} fontSize={11} fontWeight={n.isEnd ? '700' : '400'}
                          className="select-none">
                          {isRoot ? '·' : n.char}
                        </text>
                        {n.isEnd && !isRoot && (
                          <circle cx={x + 11} cy={y - 11} r={5} fill="#3b82f6" />
                        )}
                      </g>
                    )
                  })}
                </svg>
              </div>
              <p className="text-[10px] text-gray-400 mt-1">🔵 蓝圈 = 终止节点</p>
            </div>
          )}

          {/* 压缩 Trie */}
          {showRadix && (
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-xs font-semibold text-teal-600 dark:text-teal-400">压缩 Trie（Radix Tree）</span>
                <span className="text-[10px] text-gray-400">合并单链 → 边带标签</span>
              </div>
              <div className="overflow-x-auto rounded-xl border border-teal-200 dark:border-teal-800 bg-teal-50/50 dark:bg-teal-900/10">
                <svg width={radixW + 32} height={radixH + 16} viewBox={`-16 -8 ${radixW + 32} ${radixH + 16}`}
                  style={{ minWidth: Math.min(radixW + 32, 280) }}>
                  {/* 边 + 边标签 */}
                  {radixNodes.filter(n => n.parent !== null).map(n => {
                    const p = n.parent!
                    const { x: x1, y: y1 } = radixPosMap.get(p.id)!
                    const { x: x2, y: y2 } = radixPosMap.get(n.id)!
                    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2
                    const labelLen = n.label.length
                    return (
                      <g key={`re-${n.id}`}>
                        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#5eead4" strokeWidth={labelLen > 1 ? 2.5 : 1.5} />
                        {/* 边上显示标签 */}
                        <rect x={mx - labelLen * 3.5 - 2} y={my - 8} width={labelLen * 7 + 4} height={14} rx={3}
                          fill={labelLen > 1 ? '#f0fdfa' : 'transparent'} stroke={labelLen > 1 ? '#14b8a6' : 'transparent'} strokeWidth={1}
                          className="dark:fill-teal-900/50" />
                        <text x={mx} y={my + 1} textAnchor="middle" dominantBaseline="middle"
                          fill={labelLen > 1 ? '#0f766e' : '#6ee7e7'} fontSize={labelLen > 1 ? 10 : 11} fontWeight={labelLen > 1 ? '700' : '400'}
                          fontFamily="monospace" className="select-none">
                          {n.label}
                        </text>
                      </g>
                    )
                  })}
                  {/* 节点 */}
                  {radixNodes.map(n => {
                    const { x, y } = radixPosMap.get(n.id)!
                    const isRoot = n.parent === null
                    return (
                      <g key={`rn-${n.id}`}>
                        <circle cx={x} cy={y} r={RADIXNODER}
                          fill={isRoot ? '#0d9488' : n.isEnd ? '#ccfbf1' : '#f0fdfa'}
                          stroke={n.isEnd ? '#0d9488' : '#5eead4'} strokeWidth={n.isEnd ? 2 : 1}
                          className="dark:fill-opacity-80" />
                        <text x={x} y={y + 1} textAnchor="middle" dominantBaseline="middle"
                          fill={isRoot ? '#fff' : n.isEnd ? '#0f766e' : '#14b8a6'} fontSize={9} fontWeight={n.isEnd ? '700' : '400'}
                          fontFamily="monospace" className="select-none">
                          {isRoot ? 'root' : ''}
                        </text>
                        {n.isEnd && !isRoot && (
                          <circle cx={x + 12} cy={y - 12} r={5} fill="#0d9488" />
                        )}
                      </g>
                    )
                  })}
                </svg>
              </div>
              <p className="text-[10px] text-gray-400 mt-1">🟢 粗线/方框 = 合并的多字符路径 · 绿圈 = 终止节点</p>
            </div>
          )}
        </div>

        {/* 原理说明卡 */}
        <div className="mt-4 p-3 rounded-xl bg-teal-50 dark:bg-teal-900/20 border border-teal-200 dark:border-teal-700">
          <p className="text-xs font-semibold text-teal-700 dark:text-teal-300 mb-1.5">压缩规则</p>
          <div className="grid grid-cols-2 gap-2 text-xs text-teal-800 dark:text-teal-200">
            <div>• 节点只有一个子节点 → 与子节点合并为一条边</div>
            <div>• 边标签变为多字符字符串（共享前缀保留）</div>
            <div>• 查询时沿边比较整个标签（字符串比较）</div>
            <div>• 节点数从 O(总字符数) 降为 O(单词数)</div>
          </div>
        </div>
      </div>
    </div>
  )
}
