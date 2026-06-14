"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Cpu, Zap, ArrowRight } from "lucide-react"

const stages = [
  { id: "ir", label: "指令寄存器 IR", short: "IR", x: 40, y: 120 },
  { id: "decoder", label: "操作码译码器", short: "译码器", x: 200, y: 120 },
  { id: "timing", label: "时序发生器", short: "T1-Tn", x: 360, y: 60 },
  { id: "logic", label: "组合逻辑", short: "组合逻辑", x: 360, y: 180 },
  { id: "signals", label: "控制信号输出", short: "控制信号", x: 540, y: 120 },
]

const connections = [
  { from: "ir", to: "decoder", label: "操作码" },
  { from: "decoder", to: "logic", label: "译码输出" },
  { from: "timing", to: "logic", label: "时序信号" },
  { from: "logic", to: "signals", label: "控制信号" },
]

const controlSignals = [
  "PCout", "MARin", "Read", "ALUadd", "MDRout",
  "IRin", "PCin", "ALUsub", "WMFC", "End",
]

export function HardwiredControlViz() {
  const [activeStage, setActiveStage] = useState<string | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [activeStep, setActiveStep] = useState(-1)

  const runAnimation = () => {
    setIsRunning(true)
    setActiveStep(0)
    let step = 0
    const interval = setInterval(() => {
      step++
      setActiveStep(step)
      if (step >= 4) {
        clearInterval(interval)
        setTimeout(() => {
          setIsRunning(false)
          setActiveStep(-1)
        }, 1200)
      }
    }, 800)
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Cpu className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">硬布线控制器结构图</h3>
      </div>

      <div className="flex justify-center mb-4">
        <button
          onClick={runAnimation}
          disabled={isRunning}
          className="px-4 py-2 bg-accent text-white rounded-md text-sm disabled:opacity-50"
        >
          {isRunning ? "运行中..." : "演示信号流"}
        </button>
      </div>

      <svg viewBox="0 0 640 260" className="w-full max-w-2xl mx-auto">
        {connections.map((conn, i) => {
          const from = stages.find(s => s.id === conn.from)!
          const to = stages.find(s => s.id === conn.to)!
          const isActive = activeStep === i
          return (
            <g key={i}>
              <motion.line
                x1={from.x + 60}
                y1={from.y}
                x2={to.x - 10}
                y2={to.y}
                stroke={isActive ? "#3b82f6" : "#374151"}
                strokeWidth={isActive ? 3 : 1.5}
                markerEnd="url(#arrow)"
                animate={{ stroke: isActive ? "#3b82f6" : "#374151" }}
              />
              <text
                x={(from.x + to.x) / 2}
                y={(from.y + to.y) / 2 - 8}
                fill={isActive ? "#3b82f6" : "#9ca3af"}
                fontSize="10"
                textAnchor="middle"
              >
                {conn.label}
              </text>
            </g>
          )
        })}

        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
          </marker>
        </defs>

        {stages.map((stage) => {
          const isActive = activeStage === stage.id ||
            (activeStep >= 0 && stages.indexOf(stage) <= activeStep)
          return (
            <g
              key={stage.id}
              onMouseEnter={() => setActiveStage(stage.id)}
              onMouseLeave={() => setActiveStage(null)}
              style={{ cursor: "pointer" }}
            >
              <motion.rect
                x={stage.x - 55}
                y={stage.y - 22}
                width={110}
                height={44}
                rx={6}
                fill={isActive ? "#1e3a5f" : "#1f2937"}
                stroke={isActive ? "#3b82f6" : "#4b5563"}
                strokeWidth={isActive ? 2 : 1}
                animate={{
                  fill: isActive ? "#1e3a5f" : "#1f2937",
                  stroke: isActive ? "#3b82f6" : "#4b5563",
                }}
              />
              <text
                x={stage.x}
                y={stage.y + 4}
                fill={isActive ? "#60a5fa" : "#d1d5db"}
                fontSize="12"
                textAnchor="middle"
                fontWeight="500"
              >
                {stage.short}
              </text>
            </g>
          )
        })}
      </svg>

      <div className="mt-4 grid grid-cols-5 gap-2 max-w-2xl mx-auto">
        {controlSignals.map((sig, i) => (
          <motion.div
            key={sig}
            className="text-center text-xs py-1.5 rounded border border-border-subtle"
            animate={{
              backgroundColor: activeStep === 4 ? "#1e3a5f" : "transparent",
              borderColor: activeStep === 4 ? "#3b82f6" : undefined,
            }}
            transition={{ delay: i * 0.05 }}
          >
            {sig}
          </motion.div>
        ))}
      </div>

      {activeStage && (
        <div className="mt-3 text-center text-sm text-text-secondary">
          {stages.find(s => s.id === activeStage)?.label}
          {activeStage === "ir" && " — 保存当前指令的操作码字段"}
          {activeStage === "decoder" && " — 将操作码译码为独热信号"}
          {activeStage === "timing" && " — 由时钟驱动，产生 T1/T2/.../Tn 节拍信号"}
          {activeStage === "logic" && " — 综合译码输出与时序信号，生成控制信号"}
          {activeStage === "signals" && " — 输出到数据通路，控制微操作"}
        </div>
      )}
    </div>
  )
}
