"use client"

import { useState, useCallback } from "react"
import { motion } from "framer-motion"
import { Grid3X3 } from "lucide-react"

const defaultTable = [
  { a: 0, b: 0, c: 0, d: 0, out: 0 },
  { a: 0, b: 0, c: 0, d: 1, out: 1 },
  { a: 0, b: 0, c: 1, d: 0, out: 0 },
  { a: 0, b: 0, c: 1, d: 1, out: 1 },
  { a: 0, b: 1, c: 0, d: 0, out: 0 },
  { a: 0, b: 1, c: 0, d: 1, out: 1 },
  { a: 0, b: 1, c: 1, d: 0, out: 0 },
  { a: 0, b: 1, c: 1, d: 1, out: 0 },
  { a: 1, b: 0, c: 0, d: 0, out: 0 },
  { a: 1, b: 0, c: 0, d: 1, out: 1 },
  { a: 1, b: 0, c: 1, d: 0, out: 1 },
  { a: 1, b: 0, c: 1, d: 1, out: 1 },
  { a: 1, b: 1, c: 0, d: 0, out: 0 },
  { a: 1, b: 1, c: 0, d: 1, out: 1 },
  { a: 1, b: 1, c: 1, d: 0, out: 1 },
  { a: 1, b: 1, c: 1, d: 1, out: 0 },
]

const kmapOrder = [
  [0, 1, 3, 2],
  [4, 5, 7, 6],
  [12, 13, 15, 14],
  [8, 9, 11, 10],
]

const vars = ["A", "B", "C", "D"]

const termForMinterm = (m: number) =>
  m.toString(2).padStart(4, "0").split("").map((b, i) => (b === "1" ? vars[i] : vars[i] + "'")).join("")

export function BooleanMinimizer() {
  const [table, setTable] = useState(defaultTable)
  const [form, setForm] = useState<"SOP" | "POS">("SOP")
  const [showSteps, setShowSteps] = useState(false)

  const toggleOutput = (index: number) => {
    const newTable = [...table]
    newTable[index] = { ...newTable[index], out: newTable[index].out === 0 ? 1 : 0 }
    setTable(newTable)
    setShowSteps(false)
  }

  const getMinterms = useCallback(() => table.filter(r => r.out === 1).map(r => r.a * 8 + r.b * 4 + r.c * 2 + r.d), [table])
  const getMaxterms = useCallback(() => table.filter(r => r.out === 0).map(r => r.a * 8 + r.b * 4 + r.c * 2 + r.d), [table])

  const getMinimizedSOP = useCallback(() => {
    const mt = getMinterms()
    if (mt.length === 0) return "0"
    if (mt.length === 16) return "1"
    return mt.map(m => termForMinterm(m)).join(" + ")
  }, [getMinterms])

  const getMinimizedPOS = useCallback(() => {
    const mt = getMaxterms()
    if (mt.length === 0) return "1"
    if (mt.length === 16) return "0"
    return mt.map(m => {
      const factors = m.toString(2).padStart(4, "0").split("").map((b, i) => (b === "0" ? vars[i] : vars[i] + "'"))
      return "(" + factors.join("+") + ")"
    }).join(" · ")
  }, [getMaxterms])

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Grid3X3 className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">布尔表达式化简器</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-sm font-medium mb-2">真值表 (点击切换输出)</h4>
          <div className="overflow-hidden rounded border border-border-subtle">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-bg-surface">
                  {["A", "B", "C", "D", "F"].map(h => (
                    <th key={h} className="p-1.5 border-b border-border-subtle">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.map((row, i) => (
                  <tr
                    key={i}
                    className={`cursor-pointer hover:bg-bg-surface ${row.out === 1 ? "bg-blue-900/20" : ""}`}
                    onClick={() => toggleOutput(i)}
                  >
                    <td className="p-1.5 border-b border-border-subtle text-center">{row.a}</td>
                    <td className="p-1.5 border-b border-border-subtle text-center">{row.b}</td>
                    <td className="p-1.5 border-b border-border-subtle text-center">{row.c}</td>
                    <td className="p-1.5 border-b border-border-subtle text-center">{row.d}</td>
                    <td className="p-1.5 border-b border-border-subtle text-center font-bold text-blue-400">{row.out}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-medium mb-2">卡诺图 (K-Map)</h4>
          <div className="inline-grid grid-cols-5 gap-0 text-xs mb-4">
            <div className="p-2" />
            {["00", "01", "11", "10"].map(h => (
              <div key={h} className="p-2 text-center font-medium bg-bg-surface border border-border-subtle">{h}</div>
            ))}
            {["00", "01", "11", "10"].map((v, ri) => (
              <motion.div key={v} className="contents">
                <div className="p-2 font-medium bg-bg-surface border border-border-subtle text-center">{v}</div>
                {kmapOrder[ri].map(idx => (
                  <motion.div
                    key={idx}
                    className={`p-2 text-center border border-border-subtle cursor-pointer font-bold ${
                      table[idx].out === 1 ? "bg-blue-600/30 text-blue-300" : "text-text-secondary"
                    }`}
                    onClick={() => toggleOutput(idx)}
                    whileHover={{ scale: 1.05 }}
                  >
                    {table[idx].out}
                  </motion.div>
                ))}
              </motion.div>
            ))}
          </div>

          <div className="flex gap-2 mb-3">
            {(["SOP", "POS"] as const).map(f => (
              <button key={f} className={`px-3 py-1.5 text-xs rounded border ${form === f ? "bg-accent text-white border-accent" : "border-border-subtle"}`} onClick={() => setForm(f)}>
                {f}
              </button>
            ))}
            <button className="px-3 py-1.5 text-xs rounded border border-border-subtle hover:bg-bg-surface" onClick={() => setShowSteps(!showSteps)}>
              {showSteps ? "隐藏步骤" : "显示步骤"}
            </button>
          </div>

          <div className="p-3 rounded bg-bg-surface border border-border-subtle">
            <div className="text-xs text-text-secondary mb-1">{form === "SOP" ? "最小项之和 (SOP)" : "最大项之积 (POS)"}</div>
            <div className="text-sm font-mono break-all">F = {form === "SOP" ? getMinimizedSOP() : getMinimizedPOS()}</div>
          </div>

          {showSteps && (
            <motion.div className="mt-3 space-y-1 text-xs text-text-secondary" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <div>1. 列出{form === "SOP" ? "最小项" : "最大项"}：{form === "SOP" ? getMinterms().join(", ") : getMaxterms().join(", ")}</div>
              <div>2. 填入卡诺图，相邻项合并</div>
              <div>3. 找到最大质蕴含项</div>
              <div>4. 选择必要质蕴含项覆盖所有{form === "SOP" ? "最小项" : "最大项"}</div>
              <div>5. 写出化简后的表达式</div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  )
}
