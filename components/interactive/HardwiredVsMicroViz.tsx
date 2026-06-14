"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Gauge, Wrench, BrainCircuit, DollarSign, Cog } from "lucide-react"

interface Metric {
  label: string
  hardwired: number
  microprogram: number
  icon: React.ReactNode
  hardwiredDesc: string
  microprogramDesc: string
}

const metrics: Metric[] = [
  {
    label: "速度",
    hardwired: 95,
    microprogram: 60,
    icon: <Gauge className="w-4 h-4" />,
    hardwiredDesc: "直接由硬件电路产生，延迟极小",
    microprogramDesc: "需访问控存取微指令，速度较慢",
  },
  {
    label: "灵活性",
    hardwired: 25,
    microprogram: 90,
    icon: <BrainCircuit className="w-4 h-4" />,
    hardwiredDesc: "修改需重新设计电路",
    microprogramDesc: "修改微程序即可改变指令集",
  },
  {
    label: "设计复杂度",
    hardwired: 85,
    microprogram: 40,
    icon: <Cog className="w-4 h-4" />,
    hardwiredDesc: "需为每个信号设计逻辑表达式",
    microprogramDesc: "用微指令描述控制逻辑，较直观",
  },
  {
    label: "开发成本",
    hardwired: 80,
    microprogram: 50,
    icon: <DollarSign className="w-4 h-4" />,
    hardwiredDesc: "调试困难，周期长",
    microprogramDesc: "可利用仿真调试，开发较快",
  },
  {
    label: "修改难度",
    hardwired: 90,
    microprogram: 20,
    icon: <Wrench className="w-4 h-4" />,
    hardwiredDesc: "需更换芯片或重新布线",
    microprogramDesc: "更新控存内容即可",
  },
]

const comparisonRows = [
  { feature: "实现方式", hardwired: "组合逻辑电路", micro: "微程序存储" },
  { feature: "控制信号来源", hardwired: "逻辑门直接产生", micro: "从控存读出" },
  { feature: "时序控制", hardwired: "时序发生器 + 节拍", micro: "微指令顺序执行" },
  { feature: "适用场景", hardwired: "RISC、高性能", micro: "CISC、复杂指令集" },
  { feature: "典型代表", hardwired: "MIPS R4000", micro: "VAX-11/780" },
]

export function HardwiredVsMicroViz() {
  const [selectedMetric, setSelectedMetric] = useState<number | null>(null)

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <BrainCircuit className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">硬布线 vs 微程序控制器</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {metrics.map((m, i) => (
          <div
            key={m.label}
            className={`p-3 rounded-lg border cursor-pointer transition-all ${
              selectedMetric === i
                ? "border-blue-500 bg-blue-900/20"
                : "border-border-subtle hover:border-blue-400"
            }`}
            onClick={() => setSelectedMetric(selectedMetric === i ? null : i)}
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-accent">{m.icon}</span>
              <span className="text-sm font-medium">{m.label}</span>
            </div>
            <div className="space-y-2">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span>硬布线</span>
                  <span className="font-mono">{m.hardwired}%</span>
                </div>
                <div className="h-2 bg-bg-surface rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-orange-500 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${m.hardwired}%` }}
                    transition={{ duration: 0.6, delay: i * 0.1 }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span>微程序</span>
                  <span className="font-mono">{m.microprogram}%</span>
                </div>
                <div className="h-2 bg-bg-surface rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-blue-500 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${m.microprogram}%` }}
                    transition={{ duration: 0.6, delay: i * 0.1 + 0.05 }}
                  />
                </div>
              </div>
            </div>
            {selectedMetric === i && (
              <motion.div
                className="mt-2 text-xs text-text-secondary space-y-1"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
              >
                <div className="flex items-start gap-1">
                  <span className="text-orange-400 shrink-0">●</span>
                  <span>{m.hardwiredDesc}</span>
                </div>
                <div className="flex items-start gap-1">
                  <span className="text-blue-400 shrink-0">●</span>
                  <span>{m.microprogramDesc}</span>
                </div>
              </motion.div>
            )}
          </div>
        ))}
      </div>

      <h4 className="text-sm font-medium mb-2">详细对比</h4>
      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="bg-bg-surface">
              <th className="p-2 border border-border-subtle text-left">特性</th>
              <th className="p-2 border border-border-subtle text-left text-orange-400">硬布线</th>
              <th className="p-2 border border-border-subtle text-left text-blue-400">微程序</th>
            </tr>
          </thead>
          <tbody>
            {comparisonRows.map((row, i) => (
              <motion.tr
                key={row.feature}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
              >
                <td className="p-2 border border-border-subtle font-medium">{row.feature}</td>
                <td className="p-2 border border-border-subtle text-orange-300">{row.hardwired}</td>
                <td className="p-2 border border-border-subtle text-blue-300">{row.micro}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
