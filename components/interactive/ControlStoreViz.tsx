"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Database } from "lucide-react"

interface MicroInstruction {
  addr: number
  controlBits: string
  nextAddr: number
  seqField: string
  label: string
}

const controlStore: MicroInstruction[] = [
  { addr: 0, controlBits: "1100100000", nextAddr: 1, seqField: "SEQ", label: "Fetch-1: PC→MAR" },
  { addr: 1, controlBits: "0010011000", nextAddr: 2, seqField: "SEQ", label: "Fetch-2: Read, PC+1" },
  { addr: 2, controlBits: "0001000100", nextAddr: 3, seqField: "SEQ", label: "Fetch-3: MDR→IR" },
  { addr: 3, controlBits: "0000000010", nextAddr: 4, seqField: "MAP", label: "Map: Decode opcode" },
  { addr: 4, controlBits: "1000100000", nextAddr: 5, seqField: "SEQ", label: "ADD-1: IR[addr]→MAR" },
  { addr: 5, controlBits: "0010011000", nextAddr: 6, seqField: "SEQ", label: "ADD-2: Read" },
  { addr: 6, controlBits: "0000010001", nextAddr: 0, seqField: "SEQ", label: "ADD-3: AC+MDR→AC" },
  { addr: 7, controlBits: "1000100000", nextAddr: 8, seqField: "SEQ", label: "LOAD-1: IR[addr]→MAR" },
  { addr: 8, controlBits: "0010011000", nextAddr: 9, seqField: "SEQ", label: "LOAD-2: Read" },
  { addr: 9, controlBits: "0001001000", nextAddr: 0, seqField: "SEQ", label: "LOAD-3: MDR→AC" },
  { addr: 10, controlBits: "1000100000", nextAddr: 11, seqField: "SEQ", label: "STORE-1: IR[addr]→MAR" },
  { addr: 11, controlBits: "0100001000", nextAddr: 0, seqField: "SEQ", label: "STORE-2: AC→MDR, Write" },
  { addr: 12, controlBits: "0000000001", nextAddr: 0, seqField: "JMP", label: "JMP: IR[addr]→PC" },
]

const controlSignalNames = [
  "PCout", "ACout", "Read", "MDRout", "MARin",
  "ALUadd", "PCin", "IRin", "ACin", "End",
]

export function ControlStoreViz() {
  const [selectedAddr, setSelectedAddr] = useState<number | null>(null)
  const [activeAddr, setActiveAddr] = useState<number | null>(null)
  const [isExecuting, setIsExecuting] = useState(false)

  const runExecution = () => {
    setIsExecuting(true)
    setActiveAddr(0)
    let addr = 0
    const step = () => {
      const inst = controlStore.find(i => i.addr === addr)
      if (!inst || inst.controlBits.includes("1", 9)) {
        setTimeout(() => {
          setIsExecuting(false)
          setActiveAddr(null)
        }, 800)
        return
      }
      setTimeout(() => {
        addr = inst.nextAddr
        setActiveAddr(addr)
        step()
      }, 800)
    }
    setTimeout(step, 800)
  }

  const inspected = selectedAddr !== null ? controlStore.find(i => i.addr === selectedAddr) : null

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Database className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">控制存储器可视化</h3>
      </div>

      <div className="flex justify-center mb-4">
        <button
          onClick={runExecution}
          disabled={isExecuting}
          className="px-4 py-2 bg-accent text-white rounded-md text-sm disabled:opacity-50"
        >
          {isExecuting ? "执行中..." : "演示执行"}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="overflow-auto max-h-96">
          <table className="w-full text-xs border-collapse">
            <thead className="sticky top-0">
              <tr className="bg-bg-surface">
                <th className="p-1.5 border border-border-subtle">地址</th>
                <th className="p-1.5 border border-border-subtle">控制位</th>
                <th className="p-1.5 border border-border-subtle">下址</th>
                <th className="p-1.5 border border-border-subtle">描述</th>
              </tr>
            </thead>
            <tbody>
              {controlStore.map(inst => {
                const isActive = activeAddr === inst.addr
                const isSelected = selectedAddr === inst.addr
                return (
                  <motion.tr
                    key={inst.addr}
                    className={`cursor-pointer ${
                      isActive ? "bg-blue-600/30" : isSelected ? "bg-blue-900/20" : "hover:bg-bg-surface"
                    }`}
                    onClick={() => setSelectedAddr(inst.addr === selectedAddr ? null : inst.addr)}
                    animate={isActive ? { scale: [1, 1.02, 1] } : {}}
                  >
                    <td className="p-1.5 border border-border-subtle text-center font-mono">
                      {inst.addr.toString().padStart(2, "0")}
                    </td>
                    <td className="p-1.5 border border-border-subtle font-mono">
                      {inst.controlBits.split("").map((b, i) => (
                        <span key={i} className={b === "1" ? "text-blue-400" : "text-text-secondary"}>
                          {b}
                        </span>
                      ))}
                    </td>
                    <td className="p-1.5 border border-border-subtle text-center">
                      {inst.nextAddr}
                    </td>
                    <td className="p-1.5 border border-border-subtle">{inst.label}</td>
                  </motion.tr>
                )
              })}
            </tbody>
          </table>
        </div>

        <div>
          {inspected ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-3"
            >
              <div className="p-3 rounded bg-bg-surface border border-border-subtle">
                <div className="text-sm font-medium mb-1">微指令 @ 地址 {inspected.addr}</div>
                <div className="text-xs text-text-secondary">{inspected.label}</div>
              </div>

              <div className="p-3 rounded bg-bg-surface border border-border-subtle">
                <div className="text-xs font-medium mb-2">控制字段</div>
                <div className="grid grid-cols-5 gap-1">
                  {controlSignalNames.map((name, i) => (
                    <div
                      key={name}
                      className={`text-center p-1.5 rounded text-[10px] ${
                        inspected.controlBits[i] === "1"
                          ? "bg-blue-600/30 text-blue-300 border border-blue-500"
                          : "bg-bg-elevated text-text-secondary border border-border-subtle"
                      }`}
                    >
                      {name}
                    </div>
                  ))}
                </div>
              </div>

              <div className="p-3 rounded bg-bg-surface border border-border-subtle">
                <div className="text-xs font-medium mb-1">定序字段</div>
                <div className="flex items-center gap-2">
                  <span className="px-2 py-1 rounded bg-bg-elevated text-xs font-mono">
                    {inspected.seqField}
                  </span>
                  <span className="text-xs text-text-secondary">
                    → 下一地址: {inspected.nextAddr}
                  </span>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="flex items-center justify-center h-full text-sm text-text-secondary">
              点击左侧微指令查看详情
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
