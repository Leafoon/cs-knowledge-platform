'use client'

import { useState } from 'react'

const NODES = [1, 3, 6, 9, 17, 27, 31, 43, 51]
const NODE_LEVELS = [3, 0, 1, 0, 2, 0, 1, 0, 3]  // 各节点的最高层
const MAX_LEVEL = 3

// 查找目标预设
const TARGETS = [
  { val: 17, label: '查找 17（存在）' },
  { val: 31, label: '查找 31（存在）' },
  { val: 20, label: '查找 20（不存在）' },
  { val: 51, label: '查找 51（最大值）' },
]

type Step = { level: number; from: number; to: number | null; action: 'jump-right' | 'drop-down' | 'found' | 'not-found' }

function buildSearchSteps(target: number): Step[] {
  const steps: Step[] = []
  let level = MAX_LEVEL
  let pos = -1  // -1 = header

  while (level >= 0) {
    // 找到当前层中 >= target 的第一个节点
    const nextInLevel = NODES.findIndex((v, i) => i > pos && NODE_LEVELS[i] >= level && v >= target)

    if (nextInLevel === -1) {
      // 右跳到当前层末尾
      steps.push({ level, from: pos, to: null, action: 'drop-down' })
      level--
      continue
    }

    const nextVal = NODES[nextInLevel]
    if (nextVal === target) {
      steps.push({ level, from: pos, to: nextInLevel, action: 'jump-right' })
      steps.push({ level, from: nextInLevel, to: nextInLevel, action: 'found' })
      return steps
    }

    // nextVal > target，下降
    steps.push({ level, from: pos, to: null, action: 'drop-down' })
    level--
  }

  steps.push({ level: -1, from: pos, to: null, action: 'not-found' })
  return steps
}

export function SkipListSearchViz() {
  const [targetIdx, setTargetIdx] = useState(0)
  const [stepIdx, setStepIdx] = useState(-1)

  const target = TARGETS[targetIdx]
  const steps = buildSearchSteps(target.val)
  const currentStep = stepIdx >= 0 && stepIdx < steps.length ? steps[stepIdx] : null

  function changeTarget(i: number) { setTargetIdx(i); setStepIdx(-1) }

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-emerald-600 to-teal-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">🔍 跳表查找：逐层右跳 + 向下降层步进</h3>
        <p className="text-emerald-100 text-xs mt-0.5">每次从最高层开始，尽量向右跳，无法继续则降一层</p>
        <div className="flex gap-2 mt-3 flex-wrap">
          {TARGETS.map((t, i) => (
            <button key={i} onClick={() => changeTarget(i)}
              className={`px-2.5 py-1 text-xs rounded-lg font-mono transition-all ${
                targetIdx === i ? 'bg-white text-emerald-700 font-bold' : 'bg-white/25 text-white hover:bg-white/35'
              }`}>
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-5 space-y-4">
        {/* 跳表行渲染 */}
        <div className="space-y-2 overflow-x-auto">
          {Array.from({ length: MAX_LEVEL + 1 }, (_, i) => MAX_LEVEL - i).map(level => {
            const visibleIdxs = NODES.map((_, i) => i).filter(i => NODE_LEVELS[i] >= level)
            const isActiveLevel = currentStep?.level === level

            return (
              <div key={level} className={`flex items-center gap-1 rounded-lg px-2 py-1 transition-colors ${isActiveLevel ? 'bg-emerald-50 dark:bg-emerald-900/20' : ''}`}>
                <span className={`w-5 text-[11px] font-mono font-bold flex-shrink-0 ${isActiveLevel ? 'text-emerald-600 dark:text-emerald-400' : 'text-slate-400'}`}>
                  L{level}
                </span>
                <div className="flex items-center gap-0.5 flex-wrap">
                  {/* Header */}
                  <div className="h-8 w-8 rounded border border-dashed border-slate-300 dark:border-slate-600 flex items-center justify-center text-[9px] text-slate-400 font-mono flex-shrink-0">−∞</div>
                  <span className="text-slate-300 dark:text-slate-600 text-sm">→</span>

                  {visibleIdxs.map(idx => {
                    const val = NODES[idx]
                    const isActive = currentStep?.to === idx && currentStep?.level === level
                    const isFound = currentStep?.action === 'found' && currentStep?.to === idx

                    return (
                      <div key={idx} className="flex items-center gap-0.5">
                        <div className={`h-8 min-w-[2rem] px-2 rounded border text-xs font-mono font-bold flex items-center justify-center transition-all duration-300 ${
                          isFound
                            ? 'bg-emerald-500 text-white border-emerald-400 scale-110 shadow-lg shadow-emerald-400/40 ring-2 ring-emerald-300'
                            : isActive
                            ? 'bg-yellow-300 dark:bg-yellow-500 text-slate-900 border-yellow-400 scale-105'
                            : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200'
                        }`}>
                          {val}
                        </div>
                        <span className="text-slate-300 dark:text-slate-600 text-sm">→</span>
                      </div>
                    )
                  })}

                  <div className="h-7 px-1.5 rounded border border-dashed border-slate-300 dark:border-slate-600 flex items-center text-[9px] text-slate-400 font-mono">NIL</div>
                </div>
              </div>
            )
          })}
        </div>

        {/* 步骤说明 */}
        <div className={`rounded-xl px-4 py-3 text-xs font-mono leading-relaxed transition-all ${
          currentStep?.action === 'found'
            ? 'bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-200'
            : currentStep?.action === 'not-found'
            ? 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-800 dark:text-red-200'
            : 'bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300'
        }`}>
          {stepIdx === -1 && <span>点击"开始"按层查找目标值 {target.val}</span>}
          {currentStep?.action === 'jump-right' && <span>L{currentStep.level} 层：向右跳到节点 {NODES[currentStep.to!]}</span>}
          {currentStep?.action === 'drop-down' && currentStep.level >= 0 && <span>L{currentStep.level} 层右侧已超过目标，↓ 降到 L{currentStep.level - 1} 层</span>}
          {currentStep?.action === 'found' && <span>✅ 找到目标 {target.val} 于 L{currentStep.level} 层！共 {stepIdx + 1} 步</span>}
          {currentStep?.action === 'not-found' && <span>❌ 目标 {target.val} 不存在于跳表中</span>}
        </div>

        {/* 控制栏 */}
        <div className="flex items-center justify-between">
          <button onClick={() => setStepIdx(-1)} className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
            ⏮ 重置
          </button>
          <div className="flex gap-2">
            {stepIdx === -1 ? (
              <button onClick={() => setStepIdx(0)} className="px-4 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg font-medium transition-colors">
                ▶ 开始查找
              </button>
            ) : (
              <>
                <button disabled={stepIdx === 0} onClick={() => setStepIdx(s => s - 1)}
                  className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 rounded-lg disabled:opacity-40 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                  ← 上一步
                </button>
                <button disabled={stepIdx >= steps.length - 1} onClick={() => setStepIdx(s => s + 1)}
                  className="px-3 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg disabled:opacity-40 transition-colors">
                  下一步 →
                </button>
              </>
            )}
          </div>
          <span className="text-xs text-slate-400">{stepIdx >= 0 ? `${stepIdx + 1} / ${steps.length}` : '-'}</span>
        </div>
      </div>
    </div>
  )
}
