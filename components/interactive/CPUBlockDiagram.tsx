"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Cpu } from "lucide-react"

interface ComponentInfo {
  id: string
  name: string
  x: number
  y: number
  width: number
  height: number
  description: string
  color: string
}

const components: ComponentInfo[] = [
  { id: "alu", name: "ALU", x: 50, y: 80, width: 120, height: 70, description: "算术逻辑单元 - 执行算术和逻辑运算", color: "#3b82f6" },
  { id: "rf", name: "Register File", x: 250, y: 80, width: 120, height: 70, description: "寄存器组 - 存储操作数和中间结果", color: "#10b981" },
  { id: "cu", name: "Control Unit", x: 150, y: 200, width: 120, height: 70, description: "控制单元 - 产生控制信号协调各部件", color: "#f59e0b" },
  { id: "pc", name: "PC", x: 450, y: 80, width: 80, height: 50, description: "程序计数器 - 存放下一条指令地址", color: "#8b5cf6" },
  { id: "mar", name: "MAR", x: 450, y: 150, width: 80, height: 50, description: "存储器地址寄存器 - 存放访问的内存地址", color: "#ec4899" },
  { id: "mdr", name: "MDR", x: 450, y: 220, width: 80, height: 50, description: "存储器数据寄存器 - 存放读写的数据", color: "#06b6d4" },
  { id: "ir", name: "IR", x: 370, y: 200, width: 80, height: 50, description: "指令寄存器 - 存放当前执行的指令", color: "#f97316" },
  { id: "mem", name: "Memory", x: 560, y: 140, width: 90, height: 120, description: "主存储器 - 存储程序和数据", color: "#6366f1" },
]

interface BusPath {
  id: string
  name: string
  d: string
  color: string
}

const busPaths: BusPath[] = [
  { id: "data-bus", name: "数据总线", d: "M 170 115 L 250 115 M 370 115 L 450 115 L 450 175 L 490 175 M 490 245 L 450 245 L 450 220", color: "#3b82f6" },
  { id: "addr-bus", name: "地址总线", d: "M 370 105 L 450 105 L 450 160 L 490 160 M 380 225 L 450 225 L 560 180", color: "#f59e0b" },
  { id: "ctrl-bus", name: "控制总线", d: "M 210 200 L 210 150 M 310 200 L 310 150 M 450 200 L 450 175", color: "#ef4444" },
]

export function CPUBlockDiagram() {
  const [selected, setSelected] = useState<string | null>(null)
  const [highlightedBus, setHighlightedBus] = useState<string | null>(null)

  const selectedComp = components.find((c) => c.id === selected)

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Cpu className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">CPU 内部结构框图</h3>
      </div>

      <div className="flex gap-6">
        <svg viewBox="0 0 680 320" className="flex-1 min-h-[280px]">
          {busPaths.map((bus) => (
            <motion.path
              key={bus.id}
              d={bus.d}
              fill="none"
              stroke={highlightedBus === bus.id ? bus.color : "#4b5563"}
              strokeWidth={highlightedBus === bus.id ? 3 : 2}
              strokeDasharray={highlightedBus === bus.id ? "0" : "6 4"}
              animate={{ opacity: highlightedBus === bus.id ? 1 : 0.5 }}
              onMouseEnter={() => setHighlightedBus(bus.id)}
              onMouseLeave={() => setHighlightedBus(null)}
              style={{ cursor: "pointer" }}
            />
          ))}

          {busPaths.map((bus) => (
            <motion.circle
              key={`${bus.id}-pulse`}
              cx={0}
              cy={0}
              r={4}
              fill={bus.color}
              opacity={0}
              animate={
                highlightedBus === bus.id
                  ? { opacity: [0, 1, 0], offsetDistance: [0, 1] }
                  : {}
              }
              transition={{ duration: 1.5, repeat: Infinity }}
            />
          ))}

          {components.map((comp) => (
            <g
              key={comp.id}
              onClick={() => setSelected(selected === comp.id ? null : comp.id)}
              style={{ cursor: "pointer" }}
            >
              <motion.rect
                x={comp.x}
                y={comp.y}
                width={comp.width}
                height={comp.height}
                rx={6}
                fill={selected === comp.id ? comp.color : "transparent"}
                stroke={comp.color}
                strokeWidth={selected === comp.id ? 2 : 1.5}
                fillOpacity={selected === comp.id ? 0.2 : 0.05}
                whileHover={{ scale: 1.03 }}
                transition={{ type: "spring", stiffness: 300 }}
              />
              <text
                x={comp.x + comp.width / 2}
                y={comp.y + comp.height / 2}
                textAnchor="middle"
                dominantBaseline="middle"
                fill={selected === comp.id ? "#fff" : comp.color}
                fontSize={comp.id === "alu" || comp.id === "rf" || comp.id === "cu" ? 14 : 11}
                fontWeight="600"
              >
                {comp.name}
              </text>
            </g>
          ))}

          <text x={120} y={105} fill="#6b7280" fontSize={10}>ALU Bus</text>
          <text x={420} y={145} fill="#6b7280" fontSize={10}>Addr</text>
          <text x={420} y={215} fill="#6b7280" fontSize={10}>Data</text>
        </svg>

        <div className="w-56 flex flex-col gap-3">
          <div className="text-sm text-text-secondary font-medium">总线类型</div>
          {busPaths.map((bus) => (
            <button
              key={bus.id}
              className="text-left px-3 py-2 rounded border border-border-subtle hover:border-accent text-sm transition-colors"
              onMouseEnter={() => setHighlightedBus(bus.id)}
              onMouseLeave={() => setHighlightedBus(null)}
            >
              <span className="inline-block w-3 h-3 rounded-full mr-2" style={{ backgroundColor: bus.color }} />
              {bus.name}
            </button>
          ))}

          {selectedComp && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-2 p-3 rounded-lg border border-border-subtle bg-bg-surface"
            >
              <div className="font-semibold text-sm mb-1" style={{ color: selectedComp.color }}>
                {selectedComp.name}
              </div>
              <p className="text-xs text-text-secondary leading-relaxed">
                {selectedComp.description}
              </p>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  )
}
