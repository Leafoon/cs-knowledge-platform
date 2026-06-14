'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Columns, RotateCcw } from 'lucide-react'

export function BitExpansionDemo() {
  const [numChips, setNumChips] = useState(2)
  const [dataWidth, setDataWidth] = useState(4)
  const [addr, setAddr] = useState(0)

  const totalBits = numChips * dataWidth
  const capacity = Math.pow(2, 4) // 16 words for demo

  // Simulate chip data
  const chipData = Array.from({ length: numChips }, (_, chipIdx) =>
    Array.from({ length: capacity }, (_, i) => ((i + chipIdx * 13 + 42) % Math.pow(2, dataWidth)))
  )

  const combinedData = Array.from({ length: capacity }, (_, i) => {
    let val = 0
    for (let c = 0; c < numChips; c++) {
      val |= (chipData[c][i] << (c * dataWidth))
    }
    return val
  })

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">位扩展演示 (Bit Expansion)</h3>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div>
          <label className="block text-sm font-medium mb-1">芯片数: {numChips}</label>
          <input type="range" min={1} max={4} value={numChips}
            onChange={e => setNumChips(+e.target.value)} className="w-full accent-blue-500" />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">每片数据线: {dataWidth}位</label>
          <input type="range" min={4} max={8} step={4} value={dataWidth}
            onChange={e => setDataWidth(+e.target.value)} className="w-full accent-blue-500" />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">地址: {addr}</label>
          <input type="range" min={0} max={capacity - 1} value={addr}
            onChange={e => setAddr(+e.target.value)} className="w-full accent-purple-500" />
        </div>
      </div>

      {/* Chips visualization */}
      <div className="p-4 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 mb-4">
        <p className="text-xs text-slate-500 mb-3 text-center">数据线并联 — 共享地址线</p>
        <div className="flex items-end justify-center gap-4">
          {Array.from({ length: numChips }, (_, chipIdx) => (
            <motion.div key={chipIdx}
              className="border-2 border-blue-300 dark:border-blue-700 rounded-lg p-3 bg-blue-50 dark:bg-blue-950/20"
              animate={{ scale: 1 }}
              whileHover={{ scale: 1.05 }}
            >
              <p className="text-xs font-bold text-center mb-2">芯片 {chipIdx}</p>
              <p className="text-[10px] text-center text-slate-500 mb-1">{capacity}×{dataWidth}位</p>
              {/* Data bits for selected address */}
              <div className="flex gap-0.5 justify-center">
                {Array.from({ length: dataWidth }, (_, bit) => {
                  const bitVal = (chipData[chipIdx][addr] >> bit) & 1
                  return (
                    <div key={bit} className={`w-5 h-5 rounded flex items-center justify-center text-[9px] font-bold ${
                      bitVal ? 'bg-green-400 text-white' : 'bg-slate-200 dark:bg-slate-700 text-slate-500'
                    }`}>
                      {bitVal}
                    </div>
                  )
                })}
              </div>
              <p className="text-[9px] text-center mt-1 text-blue-500">D0-D{dataWidth - 1}</p>
            </motion.div>
          ))}
        </div>

        {/* Combined output */}
        <div className="mt-4 p-3 bg-green-50 dark:bg-green-950/20 rounded-lg border border-green-200 dark:border-green-800 text-center">
          <p className="text-xs text-green-600 dark:text-green-400 mb-1">组合输出 ({totalBits}位)</p>
          <div className="flex gap-0.5 justify-center">
            {Array.from({ length: totalBits }, (_, bit) => {
              const bitVal = (combinedData[addr] >> bit) & 1
              return (
                <motion.div key={bit}
                  className={`w-5 h-5 rounded flex items-center justify-center text-[9px] font-bold ${
                    bitVal ? 'bg-green-500 text-white' : 'bg-slate-200 dark:bg-slate-700 text-slate-500'
                  }`}
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ delay: bit * 0.05 }}
                >
                  {bitVal}
                </motion.div>
              )
            })}
          </div>
          <p className="text-xs mt-1 text-green-600 dark:text-green-400 font-mono">
            = 0x{combinedData[addr].toString(16).toUpperCase().padStart(Math.ceil(totalBits / 4), '0')}
          </p>
        </div>
      </div>

      <div className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg text-xs text-slate-600 dark:text-slate-400">
        <p className="font-semibold mb-1">位扩展要点:</p>
        <p>多片芯片共享地址线和控制线，数据线并联。N片k位芯片 → N×k位数据宽度。字数不变，位宽增加。</p>
      </div>
    </div>
  )
}
