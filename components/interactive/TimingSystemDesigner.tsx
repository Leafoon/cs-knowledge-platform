"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Clock, ChevronDown, ChevronRight } from "lucide-react"

interface CycleNode {
  id: string
  label: string
  children?: CycleNode[]
}

const cycleHierarchy: CycleNode[] = [
  {
    id: "fetch",
    label: "取指周期 (Fetch)",
    children: [
      { id: "f-t1", label: "T1: PC → MAR" },
      { id: "f-t2", label: "T2: M[MAR] → MDR, PC+1 → PC" },
      { id: "f-t3", label: "T3: MDR → IR" },
    ],
  },
  {
    id: "execute",
    label: "执行周期 (Execute)",
    children: [
      { id: "e-t1", label: "T4: IR[Addr] → MAR" },
      { id: "e-t2", label: "T5: M[MAR] → MDR" },
      { id: "e-t3", label: "T6: AC + MDR → AC" },
    ],
  },
  {
    id: "interrupt",
    label: "中断周期 (Interrupt)",
    children: [
      { id: "i-t1", label: "T7: SP → MAR" },
      { id: "i-t2", label: "T8: PC → MDR" },
      { id: "i-t3", label: "T9: MDR → M[MAR], PC → 0" },
    ],
  },
]

function ExpandableNode({ node, depth = 0 }: { node: CycleNode; depth?: number }) {
  const [expanded, setExpanded] = useState(depth === 0)
  const hasChildren = node.children && node.children.length > 0

  return (
    <div style={{ paddingLeft: depth * 20 }}>
      <div
        className="flex items-center gap-2 py-1.5 px-2 rounded hover:bg-bg-surface cursor-pointer"
        onClick={() => hasChildren && setExpanded(!expanded)}
      >
        {hasChildren ? (
          expanded ? (
            <ChevronDown className="w-4 h-4 text-text-secondary" />
          ) : (
            <ChevronRight className="w-4 h-4 text-text-secondary" />
          )
        ) : (
          <span className="w-4" />
        )}
        <span className={depth === 0 ? "text-sm font-medium" : "text-xs text-text-secondary"}>
          {node.label}
        </span>
      </div>
      <AnimatePresence>
        {expanded && node.children && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            {node.children.map(child => (
              <ExpandableNode key={child.id} node={child} depth={depth + 1} />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export function TimingSystemDesigner() {
  const [activeTiming, setActiveTiming] = useState<number | null>(null)

  const timingSignals = Array.from({ length: 9 }, (_, i) => ({
    name: `T${i + 1}`,
    cycle: i < 3 ? "取指" : i < 6 ? "执行" : "中断",
  }))

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Clock className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">时序系统设计器</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-sm font-medium mb-3">层次结构</h4>
          <div className="border border-border-subtle rounded-lg p-3">
            <div className="mb-2">
              <span className="text-xs font-medium text-accent">指令周期</span>
              <span className="text-xs text-text-secondary ml-2">= 取指 + 执行 + 中断</span>
            </div>
            {cycleHierarchy.map(node => (
              <ExpandableNode key={node.id} node={node} />
            ))}
          </div>
        </div>

        <div>
          <h4 className="text-sm font-medium mb-3">时序信号 T1 — T9</h4>
          <div className="grid grid-cols-3 gap-2">
            {timingSignals.map((ts, i) => (
              <motion.button
                key={ts.name}
                className={`p-3 rounded-lg border text-center transition-colors ${
                  activeTiming === i
                    ? "border-blue-500 bg-blue-900/30"
                    : "border-border-subtle hover:border-blue-400"
                }`}
                onClick={() => setActiveTiming(activeTiming === i ? null : i)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="text-sm font-mono font-bold">{ts.name}</div>
                <div className="text-xs text-text-secondary mt-1">{ts.cycle}</div>
              </motion.button>
            ))}
          </div>

          <div className="mt-4">
            <h4 className="text-sm font-medium mb-2">时钟信号波形</h4>
            <svg viewBox="0 0 400 80" className="w-full">
              {Array.from({ length: 9 }, (_, i) => {
                const x = i * 44 + 10
                const isActive = activeTiming === i
                return (
                  <g key={i}>
                    <motion.rect
                      x={x}
                      y={isActive ? 15 : 25}
                      width={20}
                      height={isActive ? 50 : 30}
                      rx={2}
                      fill={isActive ? "#3b82f6" : "#374151"}
                      animate={{
                        fill: isActive ? "#3b82f6" : "#374151",
                        y: isActive ? 15 : 25,
                        height: isActive ? 50 : 30,
                      }}
                    />
                    <motion.rect
                      x={x + 20}
                      y={35}
                      width={20}
                      height={10}
                      rx={2}
                      fill="#1f2937"
                    />
                    <text
                      x={x + 20}
                      y={75}
                      fill={isActive ? "#60a5fa" : "#6b7280"}
                      fontSize="9"
                      textAnchor="middle"
                    >
                      T{i + 1}
                    </text>
                  </g>
                )
              })}
            </svg>
          </div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3 text-xs">
        <div className="p-3 rounded bg-bg-surface border border-border-subtle">
          <div className="font-medium text-accent mb-1">指令周期</div>
          <div className="text-text-secondary">从取指到执行完成的完整过程，可能包含中断处理</div>
        </div>
        <div className="p-3 rounded bg-bg-surface border border-border-subtle">
          <div className="font-medium text-accent mb-1">机器周期</div>
          <div className="text-text-secondary">完成一个基本操作（取指/访存/执行/中断）所需时间</div>
        </div>
        <div className="p-3 rounded bg-bg-surface border border-border-subtle">
          <div className="font-medium text-accent mb-1">时钟周期</div>
          <div className="text-text-secondary">最小时间单位，也称节拍，由主频决定</div>
        </div>
      </div>
    </div>
  )
}
