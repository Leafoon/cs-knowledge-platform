"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Play, StepForward } from "lucide-react"

const microProgram = [
  { phase: "取指", addr: 0, label: "PCout, MARin", desc: "PC → MAR", nextAddr: 1 },
  { phase: "取指", addr: 1, label: "Read, MDRout, PCin", desc: "M[MAR] → MDR, PC+1", nextAddr: 2 },
  { phase: "取指", addr: 2, label: "MDRout, IRin", desc: "MDR → IR", nextAddr: 3 },
  { phase: "译码", addr: 3, label: "MAP", desc: "OP(IR) → 微地址映射", nextAddr: 4 },
  { phase: "执行", addr: 4, label: "IR[addr]→MAR", desc: "IR地址字段 → MAR", nextAddr: 5 },
  { phase: "执行", addr: 5, label: "Read, MDRout", desc: "M[MAR] → MDR", nextAddr: 6 },
  { phase: "执行", addr: 6, label: "ALUadd, ACin, End", desc: "AC + MDR → AC", nextAddr: 0 },
]

export function MicroprogramExecution() {
  const [step, setStep] = useState(-1)
  const [isRunning, setIsRunning] = useState(false)
  const current = step >= 0 ? microProgram[step] : null

  const reset = () => {
    setStep(-1)
    setIsRunning(false)
  }

  const stepForward = () => {
    if (step < microProgram.length - 1) {
      setStep(step + 1)
    } else {
      setIsRunning(false)
    }
  }

  const runAll = () => {
    setIsRunning(true)
    setStep(0)
    let s = 0
    const interval = setInterval(() => {
      s++
      if (s >= microProgram.length) {
        clearInterval(interval)
        setIsRunning(false)
        return
      }
      setStep(s)
    }, 900)
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Play className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">微程序执行过程 (ADD指令)</h3>
      </div>

      <div className="flex gap-2 mb-4">
        <button
          onClick={runAll}
          disabled={isRunning}
          className="px-3 py-1.5 text-xs bg-accent text-white rounded disabled:opacity-50"
        >
          自动运行
        </button>
        <button
          onClick={stepForward}
          disabled={isRunning || step >= microProgram.length - 1}
          className="px-3 py-1.5 text-xs border border-border-subtle rounded disabled:opacity-50 hover:bg-bg-surface"
        >
          <StepForward className="w-3 h-3 inline mr-1" />
          单步
        </button>
        <button
          onClick={reset}
          className="px-3 py-1.5 text-xs border border-border-subtle rounded hover:bg-bg-surface"
        >
          重置
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <h4 className="text-xs font-medium mb-2">微程序 (控制存储器)</h4>
          <div className="space-y-1">
            {microProgram.map((mp, i) => {
              const isActive = step === i
              const isDone = step > i
              return (
                <motion.div
                  key={i}
                  className={`flex items-center gap-3 p-2 rounded text-xs border ${
                    isActive
                      ? "border-blue-500 bg-blue-900/30"
                      : isDone
                        ? "border-border-subtle bg-bg-surface opacity-60"
                        : "border-border-subtle"
                  }`}
                  animate={isActive ? { scale: [1, 1.01, 1] } : {}}
                >
                  <span className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] shrink-0 ${
                    isActive ? "bg-blue-600 text-white" : isDone ? "bg-green-700 text-white" : "bg-bg-surface text-text-secondary"
                  }`}>
                    {isDone ? "✓" : mp.addr}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-bg-surface text-text-secondary">
                        {mp.phase}
                      </span>
                      <span className="font-mono">{mp.label}</span>
                    </div>
                    <div className="text-text-secondary mt-0.5">{mp.desc}</div>
                  </div>
                  <span className="text-text-secondary shrink-0">→ {mp.nextAddr}</span>
                </motion.div>
              )
            })}
          </div>
        </div>

        <div>
          <h4 className="text-xs font-medium mb-2">执行状态</h4>
          {current ? (
            <motion.div
              className="space-y-3"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              key={step}
            >
              <div className="p-3 rounded bg-bg-surface border border-border-subtle">
                <div className="text-xs text-text-secondary mb-1">当前阶段</div>
                <div className="text-sm font-medium">{current.phase}</div>
              </div>

              <div className="p-3 rounded bg-bg-surface border border-border-subtle">
                <div className="text-xs text-text-secondary mb-1">微地址</div>
                <div className="text-sm font-mono font-bold">μPC = {current.addr}</div>
              </div>

              <div className="p-3 rounded bg-bg-surface border border-border-subtle">
                <div className="text-xs text-text-secondary mb-1">微命令</div>
                <div className="text-sm font-mono">{current.label}</div>
              </div>

              <div className="p-3 rounded bg-bg-surface border border-border-subtle">
                <div className="text-xs text-text-secondary mb-1">数据通路操作</div>
                <div className="text-sm">{current.desc}</div>
              </div>

              <div className="p-3 rounded bg-bg-surface border border-border-subtle">
                <div className="text-xs text-text-secondary mb-1">下一微地址</div>
                <div className="text-sm font-mono">μPC → {current.nextAddr}</div>
              </div>

              <div className="text-xs text-text-secondary text-center">
                步骤 {step + 1} / {microProgram.length}
              </div>
            </motion.div>
          ) : (
            <div className="flex items-center justify-center h-40 text-sm text-text-secondary">
              点击"单步"或"自动运行"开始
            </div>
          )}
        </div>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3 text-xs">
        <div className="p-2 rounded bg-bg-surface border border-border-subtle text-center">
          <div className="font-medium text-accent">1. 取微指令</div>
          <div className="text-text-secondary">CM[μPC] → μIR</div>
        </div>
        <div className="p-2 rounded bg-bg-surface border border-border-subtle text-center">
          <div className="font-medium text-accent">2. 译码</div>
          <div className="text-text-secondary">μIR → 控制信号</div>
        </div>
        <div className="p-2 rounded bg-bg-surface border border-border-subtle text-center">
          <div className="font-medium text-accent">3. 执行 + 下址</div>
          <div className="text-text-secondary">控制信号驱动数据通路</div>
        </div>
      </div>
    </div>
  )
}
