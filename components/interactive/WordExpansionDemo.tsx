'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Grid3x3, RotateCcw } from 'lucide-react'

export function WordExpansionDemo() {
  const [numChips, setNumChips] = useState(4)
  const [addrBits, setAddrBits] = useState(6)
  const [addr, setAddr] = useState(0)

  const chipAddrBits = addrBits - Math.ceil(Math.log2(numChips))
  const selectBits = Math.ceil(Math.log2(numChips))
  const chipSelect = addr >> chipAddrBits
  const chipAddr = addr & ((1 << chipAddrBits) - 1)
  const chipCapacity = Math.pow(2, chipAddrBits)

  const chipData = Array.from({ length: numChips }, (_, i) =>
    Array.from({ length: chipCapacity }, (_, a) => ((a + i * 23 + 7) % 256))
  )

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">字扩展演示 (Word Expansion)</h3>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div>
          <label className="block text-sm font-medium mb-1">芯片数: {numChips}</label>
          <input type="range" min={2} max={8} step={2} value={numChips}
            onChange={e => { setNumChips(+e.target.value); setAddr(0) }} className="w-full accent-blue-500" />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">总地址线: {addrBits}位</label>
          <input type="range" min={4} max={8} value={addrBits}
            onChange={e => { setAddrBits(+e.target.value); setAddr(0) }} className="w-full accent-blue-500" />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">地址: {addr}</label>
          <input type="range" min={0} max={Math.pow(2, addrBits) - 1} value={addr}
            onChange={e => setAddr(+e.target.value)} className="w-full accent-purple-500" />
        </div>
      </div>

      {/* Address breakdown */}
      <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 mb-4">
        <p className="text-xs text-slate-500 mb-2">地址分解: A{addrBits - 1}...A0</p>
        <div className="flex items-center gap-2">
          <div className="flex gap-0.5">
            {Array.from({ length: selectBits }, (_, i) => {
              const bit = (addr >> (addrBits - 1 - i)) & 1
              return (
                <div key={i} className={`w-6 h-6 rounded flex items-center justify-center text-xs font-bold ${
                  bit ? 'bg-red-400 text-white' : 'bg-slate-200 dark:bg-slate-700'
                }`}>{bit}</div>
              )
            })}
          </div>
          <span className="text-xs text-slate-400">|</span>
          <div className="flex gap-0.5">
            {Array.from({ length: chipAddrBits }, (_, i) => {
              const bit = (chipAddr >> (chipAddrBits - 1 - i)) & 1
              return (
                <div key={i} className={`w-6 h-6 rounded flex items-center justify-center text-xs font-bold ${
                  bit ? 'bg-blue-400 text-white' : 'bg-slate-200 dark:bg-slate-700'
                }`}>{bit}</div>
              )
            })}
          </div>
          <span className="text-xs text-slate-500 ml-2">高位→片选({selectBits}位) | 低位→片内地址({chipAddrBits}位)</span>
        </div>
      </div>

      {/* Decoder + Chips */}
      <div className="flex items-center gap-4 mb-4">
        {/* Decoder */}
        <div className="p-3 bg-orange-50 dark:bg-orange-950/20 rounded-lg border border-orange-200 dark:border-orange-800">
          <p className="text-xs font-bold text-orange-600 dark:text-orange-400 mb-2 text-center">译码器</p>
          <div className="space-y-1">
            {Array.from({ length: numChips }, (_, i) => (
              <div key={i} className={`px-2 py-0.5 rounded text-[10px] font-mono text-center ${
                chipSelect === i ? 'bg-orange-400 text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-400'
              }`}>
                CS{i} = {chipSelect === i ? '有效' : '无效'}
              </div>
            ))}
          </div>
        </div>

        {/* Chips */}
        <div className="flex-1 grid grid-cols-4 gap-2">
          {Array.from({ length: numChips }, (_, i) => (
            <motion.div key={i}
              className={`p-2 rounded-lg border text-center text-xs ${
                chipSelect === i
                  ? 'border-green-400 bg-green-50 dark:bg-green-950/20'
                  : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 opacity-50'
              }`}
              animate={{ scale: chipSelect === i ? 1.05 : 1 }}
            >
              <p className="font-bold mb-1">芯片{i}</p>
              <p className="text-[9px] text-slate-500">{chipCapacity}字</p>
              {chipSelect === i && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  className="mt-1 p-1 bg-green-200 dark:bg-green-800 rounded">
                  <p className="font-mono">[{chipAddr}]</p>
                  <p className="font-bold">{chipData[i][chipAddr]}</p>
                </motion.div>
              )}
            </motion.div>
          ))}
        </div>
      </div>

      <div className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg text-xs text-slate-600 dark:text-slate-400">
        <p className="font-semibold mb-1">字扩展要点:</p>
        <p>N片芯片 → 用⌈log₂N⌉位高位地址通过译码器产生片选信号。数据线共享，地址线低位共享，高位用于片选。总容量 = N × 单片容量。</p>
      </div>
    </div>
  )
}
