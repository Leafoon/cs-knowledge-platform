'use client'

import { useState, useMemo, useCallback, useEffect, useRef } from 'react'

// ─── Trie 数据结构 ─────────────────────────────────────────
interface TrieNode {
  id: number
  char: string
  children: Map<string, TrieNode>
  isEnd: boolean
  word?: string
}

function makeNode(id: number, char: string): TrieNode {
  return { id, char, children: new Map(), isEnd: false }
}

// ─── 布局节点（SVG 坐标） ────────────────────────────────────
interface LayoutNode {
  id: number
  char: string
  isEnd: boolean
  word?: string
  x: number
  y: number
  parentId: number | null
  depth: number
}

// ─── 计算 Trie 布局 ─────────────────────────────────────────
function buildLayout(root: TrieNode): LayoutNode[] {
  const nodes: LayoutNode[] = []
  const LEVEL_H = 70
  const SVG_W = 660

  // 先算各节点的子树叶节点数（用于均匀分配宽度）
  function countLeaves(node: TrieNode): number {
    if (node.children.size === 0) return 1
    let s = 0
    for (const c of node.children.values()) s += countLeaves(c)
    return s
  }

  // DFS 分配 x
  function assign(node: TrieNode, left: number, right: number, depth: number, parentId: number | null) {
    const x = (left + right) / 2
    const y = depth * LEVEL_H + 36
    nodes.push({ id: node.id, char: node.char, isEnd: node.isEnd, x, y, parentId, depth })
    const childArr = [...node.children.values()]
    const totalLeaves = childArr.reduce((s, c) => s + countLeaves(c), 0) || 1
    let cur = left
    for (const child of childArr) {
      const leaves = countLeaves(child)
      const w = (leaves / totalLeaves) * (right - left)
      assign(child, cur, cur + w, depth + 1, node.id)
      cur += w
    }
  }

  assign(root, 20, SVG_W - 20, 0, null)
  return nodes
}

// ─── 预设词组 ────────────────────────────────────────────────
const WORD_SETS = [
  { label: 'app系列', words: ['app', 'apple', 'apply', 'apt', 'all', 'allow'] },
  { label: '水果词典', words: ['ban', 'banana', 'band', 'bat', 'ball', 'cat', 'car', 'card'] },
  { label: '自动补全', words: ['do', 'dot', 'dog', 'door', 'dose', 'day', 'data'] },
  { label: '简单示例', words: ['abc', 'ab', 'abd', 'xyz', 'xy'] },
]

// ─── 主组件 ─────────────────────────────────────────────────
export default function TrieInsertSearch() {
  const [setIdx, setSetIdx] = useState(0)
  const [wordsInserted, setWordsInserted] = useState(0)   // 已插入几个词
  const [mode, setMode] = useState<'insert' | 'search'>('insert')
  const [query, setQuery] = useState('')
  const [highlightPath, setHighlightPath] = useState<number[]>([])  // 高亮节点 id 列表
  const [highlightFound, setHighlightFound] = useState(false)

  const words = WORD_SETS[setIdx].words

  // 构建 Trie（只包含已插入的词）
  const { root, nodeMap } = useMemo(() => {
    let idCounter = 0
    const root = makeNode(idCounter++, '·')
    const nodeMap = new Map<number, TrieNode>([[root.id, root]])

    for (let wi = 0; wi < wordsInserted; wi++) {
      const word = words[wi]
      let node = root
      for (const ch of word) {
        if (!node.children.has(ch)) {
          const newNode = makeNode(idCounter++, ch)
          node.children.set(ch, newNode)
          nodeMap.set(newNode.id, newNode)
        }
        node = node.children.get(ch)!
      }
      node.isEnd = true
      node.word = word
    }
    return { root, nodeMap }
  }, [wordsInserted, words, setIdx])

  // 布局
  const layoutNodes = useMemo(() => buildLayout(root), [root])
  const layoutMap = useMemo(() => new Map(layoutNodes.map(n => [n.id, n])), [layoutNodes])

  // 高亮边 = highlightPath 中相邻节点对
  const highlightEdges = useMemo(() => {
    const edges: [number, number][] = []
    for (let i = 0; i + 1 < highlightPath.length; i++)
      edges.push([highlightPath[i], highlightPath[i + 1]])
    return edges
  }, [highlightPath])

  // 查找路径
  const handleSearch = useCallback((q: string) => {
    setQuery(q)
    if (!q) { setHighlightPath([]); setHighlightFound(false); return }
    const path: number[] = [root.id]
    let node = root
    for (const ch of q) {
      if (!node.children.has(ch)) {
        setHighlightPath(path)
        setHighlightFound(false)
        return
      }
      node = node.children.get(ch)!
      path.push(node.id)
    }
    setHighlightPath(path)
    setHighlightFound(node.isEnd)
  }, [root])

  const isHighlightNode = (id: number) => highlightPath.includes(id)
  const isHighlightEdge = (a: number, b: number) =>
    highlightEdges.some(([x, y]) => (x === a && y === b) || (x === b && y === a))

  const currentWord = mode === 'insert' && wordsInserted < words.length ? words[wordsInserted] : null
  const SVG_H = Math.max(...layoutNodes.map(n => n.y), 0) + 60

  return (
    <div className="rounded-2xl overflow-hidden border border-emerald-200 dark:border-emerald-800 shadow-lg">
      {/* 头部 */}
      <div className="bg-gradient-to-r from-emerald-600 to-teal-600 dark:from-emerald-700 dark:to-teal-700 px-5 py-4">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h3 className="text-white font-bold text-base">🌳 Trie 插入与搜索可视化</h3>
            <p className="text-emerald-100 text-xs mt-0.5">公共前缀共享节点，路径即字符串</p>
          </div>
          {/* 模式切换 */}
          <div className="flex gap-1 bg-emerald-800/40 rounded-lg p-0.5">
            {(['insert', 'search'] as const).map(m => (
              <button key={m} onClick={() => { setMode(m); setHighlightPath([]) }}
                className={`px-3 py-1 text-xs rounded-md font-medium transition-all ${
                  mode === m ? 'bg-white text-emerald-700' : 'text-emerald-100 hover:text-white'}`}>
                {m === 'insert' ? '插入模式' : '搜索模式'}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 p-4">
        {/* 词组选择 */}
        <div className="flex gap-2 flex-wrap mb-3">
          {WORD_SETS.map((ws, i) => (
            <button key={i} onClick={() => { setSetIdx(i); setWordsInserted(0); setHighlightPath([]); setQuery('') }}
              className={`px-3 py-1 text-xs rounded-full border transition-all ${
                setIdx === i
                  ? 'bg-emerald-500 text-white border-emerald-500'
                  : 'border-emerald-300 text-emerald-600 dark:border-emerald-700 dark:text-emerald-400 hover:border-emerald-500'
              }`}>
              {ws.label}
            </button>
          ))}
        </div>

        {/* 插入模式：词列表 + 控制按钮 */}
        {mode === 'insert' && (
          <div className="mb-3">
            <div className="flex gap-2 flex-wrap items-center mb-2">
              {words.map((w, i) => (
                <span key={i} className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold border ${
                  i < wordsInserted
                    ? 'bg-emerald-100 text-emerald-700 border-emerald-300 dark:bg-emerald-900/40 dark:text-emerald-300 dark:border-emerald-700'
                    : i === wordsInserted
                    ? 'bg-amber-100 text-amber-700 border-amber-400 dark:bg-amber-900/40 dark:text-amber-300 dark:border-amber-600 ring-2 ring-amber-400/50'
                    : 'bg-gray-100 text-gray-400 border-gray-200 dark:bg-gray-800 dark:text-gray-500 dark:border-gray-700'
                }`}>
                  {w}
                </span>
              ))}
            </div>
            <div className="flex gap-2">
              <button onClick={() => setWordsInserted(v => Math.min(v + 1, words.length))}
                disabled={wordsInserted >= words.length}
                className="px-4 py-1.5 text-xs bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:opacity-40 disabled:cursor-not-allowed transition-all font-medium">
                插入 "{words[Math.min(wordsInserted, words.length - 1)]}" →
              </button>
              <button onClick={() => setWordsInserted(words.length)}
                className="px-3 py-1.5 text-xs bg-emerald-100 text-emerald-700 rounded-lg hover:bg-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-300 transition-all">
                全部插入
              </button>
              <button onClick={() => { setWordsInserted(0); setHighlightPath([]) }}
                className="px-3 py-1.5 text-xs bg-gray-100 text-gray-600 rounded-lg hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400 transition-all">
                重置
              </button>
            </div>
            {currentWord && (
              <div className="mt-2 p-2 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 rounded-lg">
                <p className="text-xs text-amber-700 dark:text-amber-300">
                  <span className="font-semibold">下一个插入：</span>
                  {currentWord.split('').map((ch, i) => (
                    <span key={i} className="font-mono bg-amber-200 dark:bg-amber-800 px-1 rounded mx-0.5">{ch}</span>
                  ))}
                  （逐字符沿路径走，不存在则新建节点）
                </p>
              </div>
            )}
          </div>
        )}

        {/* 搜索模式：输入框 */}
        {mode === 'search' && (
          <div className="mb-3">
            <div className="flex gap-2 items-center">
              <input
                value={query}
                onChange={e => handleSearch(e.target.value.toLowerCase())}
                placeholder="输入要搜索的词或前缀…"
                className="flex-1 px-3 py-2 text-sm border border-emerald-300 dark:border-emerald-700 rounded-lg bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100 outline-none focus:ring-2 focus:ring-emerald-400"
              />
              <button onClick={() => { setQuery(''); setHighlightPath([]) }}
                className="px-3 py-2 text-xs bg-gray-100 text-gray-500 rounded-lg hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400">
                清除
              </button>
            </div>
            {query && (
              <div className={`mt-2 p-2 rounded-lg border text-xs ${
                highlightFound
                  ? 'bg-emerald-50 border-emerald-300 text-emerald-700 dark:bg-emerald-900/20 dark:border-emerald-700 dark:text-emerald-300'
                  : highlightPath.length > 1
                  ? 'bg-sky-50 border-sky-300 text-sky-700 dark:bg-sky-900/20 dark:border-sky-700 dark:text-sky-300'
                  : 'bg-rose-50 border-rose-300 text-rose-700 dark:bg-rose-900/20 dark:border-rose-700 dark:text-rose-300'
              }`}>
                {highlightFound ? `✓ "${query}" 找到！（完整单词）` :
                 highlightPath.length > 1 ? `○ "${query}" 是已有单词的前缀（startsWith = true，search = false）` :
                 `✗ "${query}" 不存在（路径中断）`}
              </div>
            )}
            {/* 可搜索词提示 */}
            <div className="mt-2 flex gap-2 flex-wrap">
              {words.slice(0, wordsInserted).map(w => (
                <button key={w} onClick={() => handleSearch(w)}
                  className="px-2 py-0.5 text-xs font-mono bg-emerald-50 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-700 rounded hover:bg-emerald-100 transition-all">
                  {w}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* SVG 树 */}
        <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
          {layoutNodes.length <= 1 ? (
            <div className="flex items-center justify-center h-24 text-gray-400 dark:text-gray-500 text-sm">
              点击"插入"开始构建 Trie 树 →
            </div>
          ) : (
            <svg width="660" height={Math.max(SVG_H, 80)} viewBox={`0 0 660 ${Math.max(SVG_H, 80)}`} className="w-full">
              {/* 边 */}
              {layoutNodes.filter(n => n.parentId !== null).map(n => {
                const p = layoutMap.get(n.parentId!)
                if (!p) return null
                const highlighted = isHighlightEdge(p.id, n.id)
                return (
                  <line key={`e-${n.id}`}
                    x1={p.x} y1={p.y} x2={n.x} y2={n.y}
                    stroke={highlighted ? '#10b981' : '#d1d5db'}
                    strokeWidth={highlighted ? 2.5 : 1.5}
                    strokeDasharray={highlighted ? 'none' : 'none'}
                    className="transition-all duration-300"
                  />
                )
              })}
              {/* 节点 */}
              {layoutNodes.map(n => {
                const highlighted = isHighlightNode(n.id)
                const isRoot = n.parentId === null
                const bgColor = highlighted
                  ? (highlightFound && n.id === highlightPath[highlightPath.length - 1] ? '#059669' : '#10b981')
                  : isRoot ? '#6b7280' : n.isEnd ? '#3b82f6' : '#d1d5db'
                const textColor = highlighted || isRoot || n.isEnd ? '#ffffff' : '#374151'
                const stroke = n.isEnd && !highlighted ? '#3b82f6' : 'none'
                return (
                  <g key={n.id} className="transition-all duration-300">
                    <circle cx={n.x} cy={n.y} r={n.isEnd ? 19 : 16}
                      fill={bgColor} stroke={stroke} strokeWidth={n.isEnd ? 2.5 : 0}
                      className="transition-all duration-300"
                    />
                    {n.isEnd && !highlighted && (
                      <circle cx={n.x} cy={n.y} r={14} fill="#3b82f6" />
                    )}
                    <text x={n.x} y={n.y + 1} textAnchor="middle" dominantBaseline="middle"
                      fill={textColor} fontSize={13} fontWeight="600" fontFamily="monospace"
                      className="select-none">
                      {n.char}
                    </text>
                    {n.isEnd && n.word && (
                      <text x={n.x} y={n.y + 31} textAnchor="middle"
                        fill="#3b82f6" fontSize={10} fontWeight="500" fontFamily="sans-serif"
                        className="select-none opacity-80">
                        "{n.word}"
                      </text>
                    )}
                  </g>
                )
              })}
            </svg>
          )}
        </div>

        {/* 图例 */}
        <div className="mt-3 flex gap-4 flex-wrap text-xs text-gray-500 dark:text-gray-400">
          <span className="flex items-center gap-1.5">
            <span className="w-4 h-4 rounded-full bg-gray-400 dark:bg-gray-500 inline-block"></span>根节点
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-4 h-4 rounded-full bg-gray-300 dark:bg-gray-600 inline-block"></span>中间节点
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-4 h-4 rounded-full bg-blue-500 inline-block"></span>单词结尾（is_end = true）
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-4 h-4 rounded-full bg-emerald-500 inline-block"></span>搜索路径高亮
          </span>
          <span className="ml-auto">
            已插入 {wordsInserted}/{words.length} 个词，节点数 {layoutNodes.length}
          </span>
        </div>
      </div>
    </div>
  )
}
