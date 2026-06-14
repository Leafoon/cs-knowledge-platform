'use client'

import { useState } from 'react'

// 跳表数据：level 表示该节点最高层（含第 0 层）
const NODES = [
  { key: 1,  level: 3 },
  { key: 3,  level: 0 },
  { key: 6,  level: 1 },
  { key: 9,  level: 0 },
  { key: 17, level: 2 },
  { key: 27, level: 0 },
  { key: 31, level: 1 },
  { key: 43, level: 0 },
  { key: 51, level: 3 },
]
const MAX_LEVEL = 3

// 层级配色（从高到低）
const LEVEL_COLORS = [
  'bg-violet-500 border-violet-400 text-white',
  'bg-blue-500 border-blue-400 text-white',
  'bg-sky-400 border-sky-300 text-white',
  'bg-cyan-400 border-cyan-300 text-slate-900',
]
const LEVEL_LABEL_COLORS = [
  'text-violet-600 dark:text-violet-400',
  'text-blue-600 dark:text-blue-400',
  'text-sky-600 dark:text-sky-400',
  'text-cyan-600 dark:text-cyan-400',
]

export function SkipListStructureViz() {
  const [hoveredKey, setHoveredKey] = useState<number | null>(null)

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-lg font-sans">
      {/* 顶栏 */}
      <div className="bg-gradient-to-r from-violet-600 via-blue-600 to-cyan-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">🔗 跳表多层链表结构全景图</h3>
        <p className="text-violet-100 text-xs mt-0.5">
          悬停节点查看该节点跨越的所有层级 · 颜色越深代表层级越高
        </p>
      </div>

      <div className="bg-white dark:bg-slate-900 p-6">
        {/* 逐层渲染（从高层到低层） */}
        <div className="space-y-3">
          {Array.from({ length: MAX_LEVEL + 1 }, (_, idx) => MAX_LEVEL - idx).map((level) => {
            const visibleNodes = NODES.filter(n => n.level >= level)
            return (
              <div key={level} className="flex items-center gap-0">
                {/* 层级标签 */}
                <div className={`w-16 text-xs font-mono font-semibold flex-shrink-0 ${LEVEL_LABEL_COLORS[level]}`}>
                  L{level}
                </div>

                {/* 哨兵 header */}
                <div className="flex items-center">
                  <div className="w-10 h-8 rounded border border-dashed border-slate-300 dark:border-slate-600 flex items-center justify-center text-[10px] text-slate-400 dark:text-slate-500">
                    −∞
                  </div>
                  <svg className="w-6 h-4 flex-shrink-0" viewBox="0 0 24 16">
                    <line x1="0" y1="8" x2="20" y2="8" stroke="currentColor" strokeWidth="1.5" className="text-slate-300 dark:text-slate-600"/>
                    <polygon points="20,4 24,8 20,12" fill="currentColor" className="text-slate-300 dark:text-slate-600"/>
                  </svg>
                </div>

                {/* 节点 */}
                <div className="flex items-center flex-wrap gap-0">
                  {visibleNodes.map((node, ni) => {
                    const isHovered = hoveredKey === node.key
                    const isRelated = hoveredKey !== null && hoveredKey === node.key
                    return (
                      <div key={node.key} className="flex items-center">
                        <div
                          onMouseEnter={() => setHoveredKey(node.key)}
                          onMouseLeave={() => setHoveredKey(null)}
                          className={`
                            h-8 min-w-[2.5rem] px-2 rounded border flex items-center justify-center
                            text-xs font-mono font-bold cursor-pointer
                            transition-all duration-200 select-none
                            ${isHovered
                              ? LEVEL_COLORS[level] + ' scale-110 shadow-lg z-10'
                              : ''}
                            ${!isHovered && level === 0
                              ? 'bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-200'
                              : ''}
                            ${!isHovered && level > 0
                              ? 'bg-blue-50 dark:bg-blue-900/30 border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300'
                              : ''}
                          `}
                        >
                          {node.key}
                        </div>
                        {ni < visibleNodes.length - 1 && (
                          <svg className="w-8 h-4 flex-shrink-0" viewBox="0 0 32 16">
                            <line x1="0" y1="8" x2="27" y2="8" stroke="currentColor" strokeWidth="1.5" className="text-slate-300 dark:text-slate-600"/>
                            <polygon points="27,4 32,8 27,12" fill="currentColor" className="text-slate-300 dark:text-slate-600"/>
                          </svg>
                        )}
                      </div>
                    )
                  })}
                  {/* NIL */}
                  <svg className="w-8 h-4 flex-shrink-0" viewBox="0 0 32 16">
                    <line x1="0" y1="8" x2="27" y2="8" stroke="currentColor" strokeWidth="1.5" className="text-slate-300 dark:text-slate-600"/>
                    <polygon points="27,4 32,8 27,12" fill="currentColor" className="text-slate-300 dark:text-slate-600"/>
                  </svg>
                  <div className="h-7 px-2 rounded border border-dashed border-slate-300 dark:border-slate-600 flex items-center text-[10px] text-slate-400 dark:text-slate-500 font-mono">
                    NIL
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        {/* 悬停信息 */}
        {hoveredKey !== null && (
          <div className="mt-4 rounded-xl bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-800 px-4 py-3">
            <p className="text-xs font-mono text-violet-800 dark:text-violet-200">
              节点 <strong>{hoveredKey}</strong> 存在于层级：
              L0 {NODES.find(n => n.key === hoveredKey)!.level >= 1 ? '→ L1' : ''}{' '}
              {NODES.find(n => n.key === hoveredKey)!.level >= 2 ? '→ L2' : ''}{' '}
              {NODES.find(n => n.key === hoveredKey)!.level >= 3 ? '→ L3' : ''}
              （层高 = {NODES.find(n => n.key === hoveredKey)!.level}）
            </p>
          </div>
        )}

        {/* 说明 */}
        <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2">
          {Array.from({ length: MAX_LEVEL + 1 }, (_, i) => i).map(level => (
            <div key={level} className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-400">
              <span className={`w-3 h-3 rounded-sm inline-block ${LEVEL_COLORS[level].split(' ')[0]}`}></span>
              <span>L{level}：{NODES.filter(n => n.level >= level).length} 个节点</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
