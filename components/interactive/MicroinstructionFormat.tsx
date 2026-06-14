"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Columns3 } from "lucide-react"

const horizontalBits = [
  "PCout", "MARin", "Read", "MDRout", "IRin",
  "PCin", "ALUadd", "ALUsub", "WMFC", "End",
]

const verticalFields = [
  { name: "源寄存器", bitCount: 3, values: ["PC", "AC", "MDR", "IR", "MAR", "SP"] },
  { name: "目的寄存器", bitCount: 3, values: ["PC", "AC", "MDR", "IR", "MAR", "SP"] },
  { name: "ALU操作", bitCount: 2, values: ["NOP", "ADD", "SUB", "AND"] },
  { name: "定序", bitCount: 2, values: ["SEQ", "JMP", "MAP", "RET"] },
]

export function MicroinstructionFormat() {
  const [format, setFormat] = useState<"horizontal" | "vertical">("horizontal")
  const [horizBits, setHorizBits] = useState("1100100000")
  const [vertVals, setVertVals] = useState([0, 1, 1, 0])
  const [showDecode, setShowDecode] = useState(false)

  const toggleHorizBit = (index: number) => {
    const arr = horizBits.split("")
    arr[index] = arr[index] === "1" ? "0" : "1"
    setHorizBits(arr.join(""))
  }

  const setVertVal = (fi: number, v: number) => {
    const n = [...vertVals]
    n[fi] = v
    setVertVals(n)
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Columns3 className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">微指令格式探索器</h3>
      </div>

      <div className="flex gap-2 mb-4">
        {(["horizontal", "vertical"] as const).map(f => (
          <button
            key={f}
            className={`px-4 py-1.5 text-xs rounded-md border ${format === f ? "bg-blue-600 text-white border-blue-600" : "border-border-subtle"}`}
            onClick={() => setFormat(f)}
          >
            {f === "horizontal" ? "水平型 (Horizontal)" : "垂直型 (Vertical)"}
          </button>
        ))}
      </div>

      {format === "horizontal" ? (
        <div>
          <div className="text-xs text-text-secondary mb-2">每位对应一个控制信号 (10位) — 点击切换</div>
          <div className="flex gap-1 mb-4">
            {horizBits.split("").map((b, i) => (
              <motion.button
                key={i}
                className={`w-10 h-10 rounded font-mono text-sm font-bold ${
                  b === "1" ? "bg-blue-600 text-white" : "bg-bg-surface text-text-secondary border border-border-subtle"
                }`}
                onClick={() => toggleHorizBit(i)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {b}
              </motion.button>
            ))}
          </div>
          <div className="flex flex-wrap gap-1 mb-4">
            {horizontalBits.map((name, i) => (
              <div
                key={name}
                className={`px-2 py-1 rounded text-xs ${
                  horizBits[i] === "1"
                    ? "bg-blue-600/20 text-blue-300 border border-blue-500"
                    : "bg-bg-surface text-text-secondary border border-border-subtle opacity-50"
                }`}
              >
                {name}
              </div>
            ))}
          </div>
          <div className="p-3 rounded bg-bg-surface border border-border-subtle text-xs">
            <div className="font-medium mb-1">水平型特点</div>
            <ul className="text-text-secondary space-y-1">
              <li>• 一位对应一个控制信号，无需译码</li>
              <li>• 并行度高，可同时激活任意信号组合</li>
              <li>• 微指令字较长，占用控存空间大</li>
              <li>• 执行速度快，适合高性能系统</li>
            </ul>
          </div>
        </div>
      ) : (
        <div>
          <div className="space-y-3 mb-4">
            {verticalFields.map((f, fi) => (
              <div key={f.name}>
                <label className="text-xs text-text-secondary mb-1 block">{f.name} ({f.bitCount}位)</label>
                <select
                  value={vertVals[fi]}
                  onChange={e => setVertVal(fi, Number(e.target.value))}
                  className="px-3 py-1.5 bg-bg-surface border border-border-subtle rounded text-sm"
                >
                  {f.values.map((v, i) => (
                    <option key={i} value={i}>{v} ({i.toString(2).padStart(f.bitCount, "0")})</option>
                  ))}
                </select>
              </div>
            ))}
          </div>

          <div className="flex items-center gap-1 mb-3">
            {verticalFields.map((f, fi) => {
              const bits = vertVals[fi].toString(2).padStart(f.bitCount, "0")
              return bits.split("").map((b, bi) => (
                <motion.span
                  key={`${fi}-${bi}`}
                  className={`w-8 h-8 rounded flex items-center justify-center font-mono text-xs font-bold ${
                    b === "1" ? "bg-blue-600 text-white" : "bg-bg-surface text-text-secondary border border-border-subtle"
                  }`}
                  animate={{ scale: showDecode ? [1, 1.1, 1] : 1 }}
                  transition={{ delay: bi * 0.05 }}
                >
                  {b}
                </motion.span>
              ))
            })}
          </div>

          <button
            className="px-3 py-1.5 text-xs rounded border border-border-subtle hover:bg-bg-surface mb-3"
            onClick={() => setShowDecode(!showDecode)}
          >
            {showDecode ? "隐藏译码" : "演示译码"}
          </button>

          {showDecode && (
            <motion.div className="p-3 rounded bg-bg-surface border border-border-subtle text-xs mb-3" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <div className="font-medium mb-2">译码过程</div>
              <div className="space-y-1 text-text-secondary">
                {verticalFields.map((f, fi) => (
                  <div key={fi}>{f.name} = {vertVals[fi].toString(2).padStart(f.bitCount, "0")} → 译码为 {f.values[vertVals[fi]]}</div>
                ))}
              </div>
            </motion.div>
          )}

          <div className="p-3 rounded bg-bg-surface border border-border-subtle text-xs">
            <div className="font-medium mb-1">垂直型特点</div>
            <ul className="text-text-secondary space-y-1">
              <li>• 编码字段需译码才能产生控制信号</li>
              <li>• 微指令字短，节省控存空间</li>
              <li>• 译码引入额外延迟</li>
              <li>• 字段互斥，不能同时指定同类操作</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}
