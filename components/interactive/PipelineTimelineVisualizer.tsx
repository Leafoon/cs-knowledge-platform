'use client'

import React, { useState, useMemo, useCallback } from 'react'

// ─── Constants ────────────────────────────────────────────────────────────────

const STAGE_COLORS: Record<string, { bg: string; text: string; darkBg: string; darkText: string }> = {
  IF:  { bg: 'bg-blue-200',    text: 'text-blue-800',    darkBg: 'dark:bg-blue-800',    darkText: 'dark:text-blue-100' },
  ID:  { bg: 'bg-green-200',   text: 'text-green-800',   darkBg: 'dark:bg-green-800',   darkText: 'dark:text-green-100' },
  EX:  { bg: 'bg-orange-200',  text: 'text-orange-800',  darkBg: 'dark:bg-orange-800',  darkText: 'dark:text-orange-100' },
  MEM: { bg: 'bg-purple-200',  text: 'text-purple-800',  darkBg: 'dark:bg-purple-800',  darkText: 'dark:text-purple-100' },
  WB:  { bg: 'bg-red-200',     text: 'text-red-800',     darkBg: 'dark:bg-red-800',     darkText: 'dark:text-red-100' },
  STALL: { bg: 'bg-slate-300', text: 'text-slate-600',   darkBg: 'dark:bg-slate-600',   darkText: 'dark:text-slate-300' },
}

const DEFAULT_STAGES = ['IF', 'ID', 'EX', 'MEM', 'WB']
const STAGE_PRESETS: Record<number, string[]> = {
  3: ['IF', 'EX', 'WB'],
  4: ['IF', 'ID', 'EX', 'WB'],
  5: ['IF', 'ID', 'EX', 'MEM', 'WB'],
  6: ['IF', 'ID', 'EX', 'MEM', 'WB', 'RET'],
  7: ['IF', 'ID', 'ISS', 'EX', 'MEM', 'WB', 'RET'],
}

// ─── Types ────────────────────────────────────────────────────────────────────

interface HazardSpec {
  fromInstr: number  // 0-indexed instruction that produces data
  toInstr: number    // 0-indexed instruction that consumes data
  stallCycles: number // 0 = forwarding only, 1+ = stall cycles needed
  label: string
}

interface CellData {
  stage: string
  isStall: boolean
  isForwardTarget?: boolean
  forwardFrom?: number // cycle number the forwarding originates from
}

// ─── Pipeline Simulation ──────────────────────────────────────────────────────

function simulatePipeline(
  numInstrs: number,
  stages: string[],
  hazards: HazardSpec[],
  hazardEnabled: boolean
): { grid: CellData[][]; totalCycles: number; stallMap: Map<string, number> } {
  const k = stages.length
  const grid: CellData[][] = Array.from({ length: numInstrs }, () => [])
  const instrStartCycle: number[] = new Array(numInstrs).fill(0)
  const stallMap = new Map<string, number>() // "from-to" -> stall cycles used

  // Determine stall cycles inserted before each instruction due to hazards
  const stallsBefore: number[] = new Array(numInstrs).fill(0)

  if (hazardEnabled) {
    for (const h of hazards) {
      if (h.toInstr >= 0 && h.toInstr < numInstrs && h.fromInstr >= 0 && h.fromInstr < numInstrs) {
        stallsBefore[h.toInstr] = Math.max(stallsBefore[h.toInstr], h.stallCycles)
        stallMap.set(`${h.fromInstr}-${h.toInstr}`, h.stallCycles)
      }
    }
  }

  // Calculate start cycle for each instruction
  for (let i = 0; i < numInstrs; i++) {
    if (i === 0) {
      instrStartCycle[i] = 0
    } else {
      // Each instruction normally starts 1 cycle after the previous
      // but stalls can push it further
      instrStartCycle[i] = instrStartCycle[i - 1] + 1 + stallsBefore[i]
    }
  }

  // Build the grid
  let maxCycle = 0
  for (let i = 0; i < numInstrs; i++) {
    const start = instrStartCycle[i]
    const row: CellData[] = []

    // Fill with stalls before the instruction starts
    for (let c = 0; c < start; c++) {
      row.push({ stage: '', isStall: false })
    }

    // Insert stall bubbles right before the stages
    for (let s = 0; s < stallsBefore[i]; s++) {
      row.push({ stage: 'STALL', isStall: true })
    }

    // Actual stages
    for (let s = 0; s < k; s++) {
      row.push({ stage: stages[s], isStall: false })
    }

    maxCycle = Math.max(maxCycle, row.length)
    grid[i] = row
  }

  // Pad all rows to the same length
  for (let i = 0; i < numInstrs; i++) {
    while (grid[i].length < maxCycle) {
      grid[i].push({ stage: '', isStall: false })
    }
  }

  return { grid, totalCycles: maxCycle, stallMap }
}

// ─── Component ────────────────────────────────────────────────────────────────

export function PipelineTimelineVisualizer() {
  const [numStages, setNumStages] = useState(5)
  const [numInstrs, setNumInstrs] = useState(8)
  const [hazardEnabled, setHazardEnabled] = useState(true)
  const [customHazards, setCustomHazards] = useState<HazardSpec[]>([
    { fromInstr: 0, toInstr: 1, stallCycles: 1, label: 'Load-Use' },
    { fromInstr: 2, toInstr: 3, stallCycles: 0, label: 'RAW (Forwarding)' },
  ])
  const [showHazardEditor, setShowHazardEditor] = useState(false)
  const [newHazard, setNewHazard] = useState({ from: 0, to: 1, stalls: 1, label: '' })

  const stages = STAGE_PRESETS[numStages] || DEFAULT_STAGES

  // Reset hazards when instruction count changes
  const validHazards = useMemo(() =>
    customHazards.filter(h => h.fromInstr < numInstrs && h.toInstr < numInstrs && h.fromInstr < h.toInstr),
    [customHazards, numInstrs]
  )

  const { grid, totalCycles, stallMap } = useMemo(
    () => simulatePipeline(numInstrs, stages, validHazards, hazardEnabled),
    [numInstrs, stages, validHazards, hazardEnabled]
  )

  // Performance metrics
  const k = stages.length
  const n = numInstrs
  const idealCycles = k + n - 1
  const throughput = n / totalCycles
  const speedup = (n * k) / totalCycles
  const efficiency = (n / (totalCycles * k)) * 100

  const addHazard = useCallback(() => {
    if (newHazard.from >= 0 && newHazard.to > newHazard.from && newHazard.to < numInstrs && newHazard.stalls >= 0) {
      setCustomHazards(prev => [...prev, {
        fromInstr: newHazard.from,
        toInstr: newHazard.to,
        stallCycles: Math.min(newHazard.stalls, 2),
        label: newHazard.label || `Data Hazard`,
      }])
    }
  }, [newHazard, numInstrs])

  const removeHazard = useCallback((idx: number) => {
    setCustomHazards(prev => prev.filter((_, i) => i !== idx))
  }, [])

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-amber-50 to-yellow-50 dark:from-slate-900 dark:to-amber-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
        流水线时空图 (Pipeline Timeline)
      </h3>

      {/* Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Pipeline stages */}
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
            流水线级数: {numStages}
          </label>
          <input
            type="range"
            min={3}
            max={7}
            value={numStages}
            onChange={e => setNumStages(Number(e.target.value))}
            className="w-full accent-amber-500"
          />
          <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 mt-1">
            <span>3</span>
            <span>7</span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            阶段: {stages.join(' / ')}
          </p>
        </div>

        {/* Instruction count */}
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
            指令数: {numInstrs}
          </label>
          <input
            type="range"
            min={5}
            max={20}
            value={numInstrs}
            onChange={e => setNumInstrs(Number(e.target.value))}
            className="w-full accent-amber-500"
          />
          <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 mt-1">
            <span>5</span>
            <span>20</span>
          </div>
        </div>

        {/* Hazard toggle */}
        <div className="flex flex-col justify-center">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={hazardEnabled}
              onChange={e => setHazardEnabled(e.target.checked)}
              className="w-4 h-4 accent-amber-500 rounded"
            />
            <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
              启用冒险 (Hazards)
            </span>
          </label>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
            {hazardEnabled
              ? `当前 ${validHazards.length} 个冒险`
              : '理想流水线（无冒险）'}
          </p>
        </div>

        {/* Hazard editor toggle */}
        <div className="flex flex-col justify-center">
          <button
            onClick={() => setShowHazardEditor(prev => !prev)}
            className="px-3 py-2 text-sm rounded-lg bg-amber-100 dark:bg-amber-900 text-amber-800 dark:text-amber-200 hover:bg-amber-200 dark:hover:bg-amber-800 transition-colors border border-amber-300 dark:border-amber-700"
          >
            {showHazardEditor ? '收起冒险编辑器' : '编辑冒险 (Hazards)'}
          </button>
        </div>
      </div>

      {/* Hazard Editor Panel */}
      {showHazardEditor && (
        <div className="mb-6 p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
            冒险配置 (Hazard Configuration)
          </h4>

          {/* Existing hazards */}
          <div className="space-y-2 mb-4">
            {customHazards.map((h, idx) => (
              <div key={idx} className="flex items-center gap-3 text-sm">
                <span className="px-2 py-1 rounded bg-amber-100 dark:bg-amber-900 text-amber-800 dark:text-amber-200 font-mono text-xs">
                  {h.label}
                </span>
                <span className="text-slate-600 dark:text-slate-400">
                  I{h.fromInstr + 1} → I{h.toInstr + 1}
                </span>
                <span className="text-slate-500 dark:text-slate-500">
                  {h.stallCycles === 0 ? 'Forwarding (0 stall)' : `${h.stallCycles} stall cycle(s)`}
                </span>
                <button
                  onClick={() => removeHazard(idx)}
                  className="ml-auto text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 text-xs"
                >
                  删除
                </button>
              </div>
            ))}
            {customHazards.length === 0 && (
              <p className="text-xs text-slate-400 dark:text-slate-500">暂无冒险定义</p>
            )}
          </div>

          {/* Add new hazard */}
          <div className="flex flex-wrap items-end gap-3 pt-3 border-t border-slate-200 dark:border-slate-600">
            <div>
              <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">生产者 (From)</label>
              <select
                value={newHazard.from}
                onChange={e => setNewHazard(prev => ({ ...prev, from: Number(e.target.value) }))}
                className="text-sm px-2 py-1 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300"
              >
                {Array.from({ length: numInstrs - 1 }, (_, i) => (
                  <option key={i} value={i}>I{i + 1}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">消费者 (To)</label>
              <select
                value={newHazard.to}
                onChange={e => setNewHazard(prev => ({ ...prev, to: Number(e.target.value) }))}
                className="text-sm px-2 py-1 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300"
              >
                {Array.from({ length: numInstrs - 1 }, (_, i) => (
                  <option key={i + 1} value={i + 1}>I{i + 2}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">暂停周期</label>
              <select
                value={newHazard.stalls}
                onChange={e => setNewHazard(prev => ({ ...prev, stalls: Number(e.target.value) }))}
                className="text-sm px-2 py-1 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300"
              >
                <option value={0}>0 (Forwarding)</option>
                <option value={1}>1 (Load-Use)</option>
                <option value={2}>2 (No Forwarding)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">标签</label>
              <input
                type="text"
                value={newHazard.label}
                onChange={e => setNewHazard(prev => ({ ...prev, label: e.target.value }))}
                placeholder="e.g. RAW"
                className="text-sm px-2 py-1 w-28 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300"
              />
            </div>
            <button
              onClick={addHazard}
              className="px-3 py-1 text-sm rounded bg-amber-500 hover:bg-amber-600 text-white transition-colors"
            >
              添加
            </button>
          </div>
        </div>
      )}

      {/* Stage Legend */}
      <div className="flex flex-wrap gap-2 mb-4">
        {stages.map(s => {
          const c = STAGE_COLORS[s] || STAGE_COLORS.IF
          return (
            <span
              key={s}
              className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold ${c.bg} ${c.text} ${c.darkBg} ${c.darkText}`}
            >
              {s}
            </span>
          )
        })}
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-slate-300 text-slate-600 dark:bg-slate-600 dark:text-slate-300">
          STALL (bubble)
        </span>
      </div>

      {/* Pipeline Timeline Grid */}
      <div className="overflow-x-auto mb-6">
        <table className="border-collapse">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-amber-50 dark:bg-slate-900 px-3 py-2 text-xs font-semibold text-slate-600 dark:text-slate-400 border border-slate-300 dark:border-slate-600 text-left min-w-[80px]">
                指令
              </th>
              {Array.from({ length: totalCycles }, (_, c) => (
                <th
                  key={c}
                  className="px-2 py-2 text-xs font-medium text-slate-600 dark:text-slate-400 border border-slate-300 dark:border-slate-600 text-center min-w-[48px]"
                >
                  T{c + 1}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {grid.map((row, i) => (
              <tr key={i}>
                <td className="sticky left-0 z-10 bg-amber-50 dark:bg-slate-900 px-3 py-1.5 text-xs font-mono font-semibold text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-600">
                  I{i + 1}
                </td>
                {row.map((cell, c) => {
                  if (!cell.stage) {
                    return (
                      <td
                        key={c}
                        className="px-1 py-1.5 border border-slate-200 dark:border-slate-700 min-w-[48px]"
                      />
                    )
                  }
                  const colors = cell.isStall
                    ? STAGE_COLORS.STALL
                    : STAGE_COLORS[cell.stage] || STAGE_COLORS.IF

                  // Check if this cell is a forwarding target
                  let forwardIndicator = ''
                  for (const h of validHazards) {
                    if (h.stallCycles === 0 && h.toInstr === i) {
                      // Find the EX stage of the producer instruction
                      const producerRow = grid[h.fromInstr]
                      const exIdx = producerRow?.findIndex(d => d.stage === 'EX')
                      if (exIdx !== undefined && exIdx >= 0 && cell.stage === 'EX') {
                        forwardIndicator = 'Fwd'
                      }
                    }
                  }

                  return (
                    <td
                      key={c}
                      className={`px-1 py-1.5 border border-slate-200 dark:border-slate-700 text-center min-w-[48px] relative ${colors.bg} ${colors.text} ${colors.darkBg} ${colors.darkText}`}
                    >
                      <span className="text-xs font-bold leading-tight">
                        {cell.isStall ? 'bubble' : cell.stage}
                      </span>
                      {forwardIndicator && (
                        <span className="absolute -top-1 -right-1 text-[9px] bg-green-500 text-white rounded px-0.5 leading-tight">
                          F
                        </span>
                      )}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Forwarding arrows indicator */}
      {hazardEnabled && validHazards.some(h => h.stallCycles === 0) && (
        <div className="mb-4 flex items-center gap-2 text-xs text-green-700 dark:text-green-300">
          <span className="inline-block w-5 h-5 rounded bg-green-500 text-white text-[9px] font-bold text-center leading-5">F</span>
          <span>= 数据前递 (Forwarding)，无需暂停周期</span>
        </div>
      )}

      {/* Performance Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="总执行时间"
          value={`${totalCycles} T`}
          detail={`${idealCycles}T (理想) → ${totalCycles}T (实际)`}
          highlight={totalCycles > idealCycles}
        />
        <MetricCard
          label="吞吐率"
          value={throughput.toFixed(3)}
          unit="IPC"
          detail={`${n} 指令 / ${totalCycles} 周期`}
          highlight={false}
        />
        <MetricCard
          label="加速比"
          value={speedup.toFixed(2)}
          unit="x"
          detail={`vs 非流水线 ${n * k}T`}
          highlight={false}
        />
        <MetricCard
          label="流水线效率"
          value={efficiency.toFixed(1)}
          unit="%"
          detail={`${n}/${totalCycles}×${k} 段`}
          highlight={efficiency < 90}
        />
      </div>

      {/* Formula reference */}
      <div className="mt-5 p-3 bg-white/60 dark:bg-slate-800/60 rounded-lg border border-slate-200 dark:border-slate-700">
        <p className="text-xs text-slate-500 dark:text-slate-400 font-medium mb-1">性能公式:</p>
        <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-600 dark:text-slate-300">
          <span>TP = n / (k + n - 1 + stalls)</span>
          <span>S = (n &times; k) / (k + n - 1 + stalls)</span>
          <span>E = n / (k + n - 1 + stalls) / k &times; 100%</span>
        </div>
      </div>
    </div>
  )
}

// ─── Metric Card ──────────────────────────────────────────────────────────────

function MetricCard({
  label,
  value,
  unit,
  detail,
  highlight,
}: {
  label: string
  value: string
  unit?: string
  detail: string
  highlight: boolean
}) {
  return (
    <div className={`p-3 rounded-lg border ${
      highlight
        ? 'bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-800'
        : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700'
    }`}>
      <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">{label}</p>
      <p className="text-xl font-bold text-slate-800 dark:text-slate-100">
        {value}
        {unit && <span className="text-sm font-normal text-slate-500 dark:text-slate-400 ml-1">{unit}</span>}
      </p>
      <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-1">{detail}</p>
    </div>
  )
}
