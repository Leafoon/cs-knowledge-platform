"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { GitBranch } from "lucide-react"

type Scenario = "sequential" | "conditional" | "unconditional" | "mapping"

const scenarios: Record<Scenario, {
  label: string
  desc: string
  source: string
  steps: { label: string; detail: string }[]
}> = {
  sequential: {
    label: "顺序执行",
    desc: "μPC 自增，取出下一条微指令",
    source: "μPC + 1",
    steps: [
      { label: "μPC → CMAR", detail: "当前微地址送控存地址寄存器" },
      { label: "CM[CMAR] → μIR", detail: "从控存读出微指令" },
      { label: "μPC + 1 → μPC", detail: "微程序计数器自增" },
      { label: "μIR → 控制信号", detail: "译码产生控制信号" },
    ],
  },
  conditional: {
    label: "条件转移",
    desc: "根据状态条件选择下一地址",
    source: "条件 ? 分支地址 : μPC+1",
    steps: [
      { label: "检测条件", detail: "检查状态标志（如 Z, C, N）" },
      { label: "条件为真", detail: "从 μIR 的下址字段取分支地址" },
      { label: "条件为假", detail: "μPC + 1 顺序执行" },
      { label: "加载新地址", detail: "选中的地址送 μPC" },
    ],
  },
  unconditional: {
    label: "无条件跳转",
    desc: "直接从微指令的下址字段获取地址",
    source: "μIR[NextAddr]",
    steps: [
      { label: "读取下址字段", detail: "从 μIR 中提取下一微地址" },
      { label: "加载到 μPC", detail: "新地址送入微程序计数器" },
      { label: "继续执行", detail: "从新地址取下一条微指令" },
    ],
  },
  mapping: {
    label: "指令映射",
    desc: "将机器指令操作码映射为微程序入口地址",
    source: "OP(IR) → 入口地址",
    steps: [
      { label: "取操作码", detail: "从 IR 中提取操作码字段" },
      { label: "地址映射", detail: "操作码经映射逻辑转换为微地址" },
      { label: "加载入口地址", detail: "映射结果送入 μPC" },
      { label: "开始执行", detail: "从入口地址开始执行微程序" },
    ],
  },
}

export function MicroprogramSequencer() {
  const [active, setActive] = useState<Scenario>("sequential")
  const [animStep, setAnimStep] = useState(-1)
  const [isRunning, setIsRunning] = useState(false)

  const current = scenarios[active]

  const runAnimation = () => {
    setIsRunning(true)
    setAnimStep(0)
    let step = 0
    const interval = setInterval(() => {
      step++
      setAnimStep(step)
      if (step >= current.steps.length - 1) {
        clearInterval(interval)
        setTimeout(() => {
          setIsRunning(false)
          setAnimStep(-1)
        }, 1000)
      }
    }, 700)
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <GitBranch className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">微程序定序器</h3>
      </div>

      <div className="flex gap-2 mb-4 flex-wrap">
        {(Object.keys(scenarios) as Scenario[]).map(key => (
          <button
            key={key}
            className={`px-3 py-1.5 text-xs rounded-md border transition-colors ${
              active === key
                ? "bg-blue-600 text-white border-blue-600"
                : "border-border-subtle hover:border-blue-400"
            }`}
            onClick={() => { setActive(key); setAnimStep(-1); setIsRunning(false) }}
          >
            {scenarios[key].label}
          </button>
        ))}
      </div>

      <div className="mb-4 p-3 rounded bg-bg-surface border border-border-subtle">
        <div className="text-sm font-medium mb-1">{current.label}</div>
        <div className="text-xs text-text-secondary">{current.desc}</div>
      </div>

      <div className="flex justify-center mb-4">
        <button
          onClick={runAnimation}
          disabled={isRunning}
          className="px-4 py-2 bg-accent text-white rounded-md text-sm disabled:opacity-50"
        >
          {isRunning ? "执行中..." : "演示"}
        </button>
      </div>

      <svg viewBox="0 0 500 200" className="w-full max-w-lg mx-auto mb-4">
        <rect x={200} y={10} width={100} height={36} rx={6}
          fill={animStep >= 0 ? "#1e3a5f" : "#1f2937"}
          stroke={animStep >= 0 ? "#3b82f6" : "#4b5563"} />
        <text x={250} y={33} fill="#d1d5db" fontSize="12" textAnchor="middle">μPC</text>

        <line x1={250} y1={46} x2={250} y2={70}
          stroke={animStep >= 0 ? "#3b82f6" : "#4b5563"} strokeWidth={2} />

        <rect x={180} y={70} width={140} height={36} rx={6}
          fill={animStep >= 1 ? "#1e3a5f" : "#1f2937"}
          stroke={animStep >= 1 ? "#3b82f6" : "#4b5563"} />
        <text x={250} y={93} fill="#d1d5db" fontSize="12" textAnchor="middle">控制存储器 CM</text>

        <line x1={250} y1={106} x2={250} y2={130}
          stroke={animStep >= 2 ? "#3b82f6" : "#4b5563"} strokeWidth={2} />

        <rect x={180} y={130} width={140} height={36} rx={6}
          fill={animStep >= 2 ? "#1e3a5f" : "#1f2937"}
          stroke={animStep >= 2 ? "#3b82f6" : "#4b5563"} />
        <text x={250} y={153} fill="#d1d5db" fontSize="12" textAnchor="middle">μIR / 定序逻辑</text>

        <text x={250} y={190} fill="#60a5fa" fontSize="10" textAnchor="middle">
          地址来源: {current.source}
        </text>

        <motion.rect
          x={350} y={50} width={120} height={28} rx={4}
          fill={active === "mapping" ? "#1e3a5f" : "#1f2937"}
          stroke={active === "mapping" ? "#3b82f6" : "#374151"} />
        <text x={410} y={69} fill="#9ca3af" fontSize="10" textAnchor="middle">IR 操作码映射</text>

        <motion.rect
          x={350} y={90} width={120} height={28} rx={4}
          fill={active === "conditional" ? "#1e3a5f" : "#1f2937"}
          stroke={active === "conditional" ? "#3b82f6" : "#374151"} />
        <text x={410} y={109} fill="#9ca3af" fontSize="10" textAnchor="middle">条件判断</text>

        <motion.rect
          x={350} y={130} width={120} height={28} rx={4}
          fill={active === "unconditional" ? "#1e3a5f" : "#1f2937"}
          stroke={active === "unconditional" ? "#3b82f6" : "#374151"} />
        <text x={410} y={149} fill="#9ca3af" fontSize="10" textAnchor="middle">下址字段</text>
      </svg>

      <div className="space-y-2">
        {current.steps.map((step, i) => (
          <motion.div
            key={i}
            className={`flex items-start gap-3 p-2 rounded text-xs ${
              animStep >= i ? "bg-blue-900/20" : ""
            }`}
            animate={{ opacity: animStep >= 0 ? (animStep >= i ? 1 : 0.4) : 1 }}
          >
            <span className={`shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[10px] ${
              animStep >= i ? "bg-blue-600 text-white" : "bg-bg-surface text-text-secondary"
            }`}>
              {i + 1}
            </span>
            <div>
              <span className="font-medium">{step.label}</span>
              <span className="text-text-secondary ml-2">{step.detail}</span>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
