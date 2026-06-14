'use client'

import { useState, useCallback } from 'react'

const MAX_LEVEL = 4
const P = 0.5

function generateLevelWithSeed(seed: number): { level: number; coins: boolean[] } {
  // 用固定种子模拟掷硬币
  let rng = seed
  const coins: boolean[] = []
  let level = 0
  for (let i = 0; i < MAX_LEVEL; i++) {
    rng = (rng * 1664525 + 1013904223) & 0xffffffff
    const hit = (rng >>> 0) / 0xffffffff < P
    coins.push(hit)
    if (hit) level++
    else break
  }
  return { level, coins }
}

const PRESETS = [
  { key: 25, seed: 42, label: '插入 25' },
  { key: 14, seed: 7,  label: '插入 14' },
  { key: 38, seed: 101, label: '插入 38' },
]

const BASE_NODES = [1, 9, 17, 27, 31, 43, 51]
const BASE_LEVELS = [3, 0, 2, 0, 1, 0, 3]

export function SkipListInsertRandom() {
  const [presetIdx, setPresetIdx] = useState(0)
  const [phase, setPhase] = useState<'idle' | 'coin' | 'insert' | 'done'>('idle')
  const [coinStep, setCoinStep] = useState(0)

  const preset = PRESETS[presetIdx]
  const { level: newLevel, coins } = generateLevelWithSeed(preset.seed)

  function reset() { setPhase('idle'); setCoinStep(0) }
  function changePreset(i: number) { setPresetIdx(i); reset() }

  // 插入后所有节点（含新节点）
  const allInserted = [...BASE_NODES, preset.key].sort((a, b) => a - b)
  const allLevels = allInserted.map(v => {
    const bi = BASE_NODES.indexOf(v)
    if (bi >= 0) return BASE_LEVELS[bi]
    return newLevel
  })

  const newNodeIdx = allInserted.indexOf(preset.key)

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-purple-600 to-pink-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">🎲 跳表插入：随机层高生成 + 多层链表更新</h3>
        <p className="text-purple-100 text-xs mt-0.5">以概率 p=0.5 掷硬币决定新节点晋升高度，正面继续，反面停止</p>
        <div className="flex gap-2 mt-3 flex-wrap">
          {PRESETS.map((p, i) => (
            <button key={i} onClick={() => changePreset(i)}
              className={`px-2.5 py-1 text-xs rounded-lg font-mono transition-all ${
                presetIdx === i ? 'bg-white text-purple-700 font-bold' : 'bg-white/25 text-white hover:bg-white/35'
              }`}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-5 space-y-4">
        {/* 掷硬币阶段 */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 p-4 bg-slate-50 dark:bg-slate-800/60">
          <p className="text-xs font-semibold text-slate-700 dark:text-slate-200 mb-3">
            第一步：掷硬币决定节点 {preset.key} 的层高
          </p>
          <div className="flex items-center gap-3 flex-wrap">
            {coins.map((hit, i) => {
              const revealed = phase !== 'idle' && coinStep > i || phase === 'insert' || phase === 'done'
              return (
                <div key={i} className="flex flex-col items-center gap-1">
                  <div className={`w-12 h-12 rounded-full border-2 flex items-center justify-center text-lg font-bold transition-all duration-500 ${
                    !revealed
                      ? 'bg-slate-200 dark:bg-slate-700 border-slate-300 dark:border-slate-600 text-slate-400'
                      : hit
                      ? 'bg-purple-500 border-purple-400 text-white scale-110 shadow-lg'
                      : 'bg-rose-400 border-rose-300 text-white'
                  }`}>
                    {revealed ? (hit ? '正' : '反') : '?'}
                  </div>
                  <span className="text-[10px] text-slate-500 dark:text-slate-400">第 {i+1} 次</span>
                </div>
              )
            })}
            <div className="flex flex-col items-center gap-1 ml-2">
              <div className={`px-3 py-2 rounded-lg text-sm font-bold transition-all ${
                phase === 'insert' || phase === 'done'
                  ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-800'
                  : 'bg-slate-100 dark:bg-slate-700 text-slate-400 border border-slate-200 dark:border-slate-600'
              }`}>
                层高 = {phase === 'insert' || phase === 'done' ? newLevel : '?'}
              </div>
              <span className="text-[10px] text-slate-500 dark:text-slate-400">最终层高</span>
            </div>
          </div>

          {/* 控制按钮 */}
          <div className="flex gap-2 mt-3">
            {phase === 'idle' && (
              <button onClick={() => { setPhase('coin'); setCoinStep(1) }}
                className="px-3 py-1.5 text-xs bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors">
                🎲 开始掷硬币
              </button>
            )}
            {phase === 'coin' && coinStep <= coins.length && (
              <button onClick={() => {
                if (coinStep < coins.length) setCoinStep(s => s + 1)
                else setPhase('insert')
              }}
                className="px-3 py-1.5 text-xs bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors">
                {coinStep < coins.length ? `揭晓第 ${coinStep + 1} 次` : '确定层高 →'}
              </button>
            )}
            {(phase === 'insert' || phase === 'done') && (
              <button onClick={() => setPhase('done')}
                className="px-3 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg transition-colors">
                ✅ 更新跳表结构
              </button>
            )}
            {phase !== 'idle' && (
              <button onClick={reset}
                className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                重置
              </button>
            )}
          </div>
        </div>

        {/* 插入后结构 */}
        {(phase === 'insert' || phase === 'done') && (
          <div className="rounded-xl border border-purple-200 dark:border-purple-800 bg-purple-50 dark:bg-purple-900/10 p-4">
            <p className="text-xs font-semibold text-purple-700 dark:text-purple-300 mb-3">
              第二步：将节点 {preset.key}（层高 {newLevel}）插入各层链表
            </p>
            <div className="space-y-2">
              {Array.from({ length: MAX_LEVEL + 1 }, (_, i) => MAX_LEVEL - i).map(level => {
                const visible = allInserted.map((_, i) => i).filter(i => allLevels[i] >= level)
                return (
                  <div key={level} className="flex items-center gap-1 text-xs">
                    <span className="w-5 font-mono font-bold text-[11px] text-slate-500 dark:text-slate-400 flex-shrink-0">L{level}</span>
                    <span className="text-slate-300 dark:text-slate-600">−∞→</span>
                    {visible.map(i => (
                      <span key={i} className={`flex items-center gap-0.5`}>
                        <span className={`px-2 py-0.5 rounded border font-mono text-[11px] ${
                          i === newNodeIdx && allLevels[i] >= level
                            ? 'bg-purple-500 text-white border-purple-400 font-bold'
                            : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200'
                        }`}>{allInserted[i]}</span>
                        <span className="text-slate-300 dark:text-slate-600">→</span>
                      </span>
                    ))}
                    <span className="text-slate-400 text-[10px] font-mono">NIL</span>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        <p className="text-center text-xs text-slate-400 dark:text-slate-600 border-t border-slate-100 dark:border-slate-800 pt-3">
          随机层高使得跳表无需像 AVL 树那样旋转来平衡，期望高度 ≈ log₂(n)
        </p>
      </div>
    </div>
  )
}
