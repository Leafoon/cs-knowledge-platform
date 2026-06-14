'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { CircuitBoard, RotateCcw } from 'lucide-react'

export function MemoryChipExplorer() {
  const [addrBits, setAddrBits] = useState(4)
  const [dataBits, setDataBits] = useState(8)
  const [csEnabled, setCsEnabled] = useState(true)
  const [weEnabled, setWeEnabled] = useState(false)
  const [addr, setAddr] = useState(0)

  const capacity = Math.pow(2, addrBits)
  const memory = useMemo(() => {
    return Array.from({ length: capacity }, (_, i) => (i * 37 + 128) % 256)
  }, [capacity])

  const currentData = csEnabled ? memory[addr] : null

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">存储器芯片探索器 (Memory Chip Explorer)</h3>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
        <div>
          <label className="block text-sm font-medium mb-1">地址线: {addrBits}位</label>
          <input type="range" min={2} max={6} value={addrBits}
            onChange={e => { setAddrBits(+e.target.value); setAddr(0) }}
            className="w-full accent-blue-500" />
          <p className="text-xs text-slate-500">容量: {capacity} × {dataBits}位</p>
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">数据线: {dataBits}位</label>
          <input type="range" min={4} max={16} step={4} value={dataBits}
            onChange={e => setDataBits(+e.target.value)}
            className="w-full accent-blue-500" />
        </div>
        <div className="flex flex-col justify-center">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={csEnabled} onChange={e => setCsEnabled(e.target.checked)} className="accent-green-500" />
            <span className="text-sm">CS (片选)</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer mt-1">
            <input type="checkbox" checked={weEnabled} onChange={e => setWeEnabled(e.target.checked)} className="accent-orange-500" />
            <span className="text-sm">WE (写使能)</span>
          </label>
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">地址: {addr}</label>
          <input type="range" min={0} max={capacity - 1} value={addr}
            onChange={e => setAddr(+e.target.value)}
            className="w-full accent-purple-500" />
        </div>
      </div>

      {/* Chip diagram */}
      <div className="p-4 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 mb-4">
        <div className="flex items-center justify-between">
          {/* Address pins */}
          <div className="flex flex-col gap-1">
            <p className="text-[10px] text-slate-400 mb-1">地址输入</p>
            {Array.from({ length: addrBits }, (_, i) => (
              <div key={i} className="flex items-center gap-1">
                <div className={`w-4 h-3 rounded-sm ${(addr >> (addrBits - 1 - i)) & 1 ? 'bg-blue-400' : 'bg-slate-200 dark:bg-slate-700'}`} />
                <span className="text-[9px] text-slate-400">A{addrBits - 1 - i}</span>
              </div>
            ))}
          </div>

          {/* Chip body */}
          <div className="flex-1 mx-4">
            <div className={`p-4 rounded-lg border-2 text-center ${
              csEnabled ? 'border-green-400 bg-green-50 dark:bg-green-950/20' : 'border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-800'
            }`}>
              <p className="text-xs font-semibold text-slate-600 dark:text-slate-300">
                {capacity}×{dataBits} SRAM
              </p>
              <div className="mt-2 grid grid-cols-4 gap-0.5">
                {Array.from({ length: Math.min(capacity, 16) }, (_, i) => (
                  <motion.div key={i}
                    className={`h-4 rounded-sm text-[8px] flex items-center justify-center font-mono ${
                      i === addr ? 'bg-yellow-300 dark:bg-yellow-700 text-yellow-800 dark:text-yellow-200' :
                      'bg-slate-200 dark:bg-slate-700 text-slate-500'
                    }`}
                    animate={{ scale: i === addr ? 1.1 : 1 }}
                  >
                    {i === addr && csEnabled ? `0x${currentData?.toString(16).toUpperCase().padStart(2, '0')}` : ''}
                  </motion.div>
                ))}
              </div>
              {/* Control signals */}
              <div className="flex justify-center gap-3 mt-2 text-[10px]">
                <span className={csEnabled ? 'text-green-600' : 'text-red-400'}>CS={csEnabled ? 1 : 0}</span>
                <span className={weEnabled ? 'text-orange-600' : 'text-blue-400'}>WE={weEnabled ? 1 : 0}</span>
                <span className="text-slate-500">OE=1</span>
              </div>
            </div>
          </div>

          {/* Data pins */}
          <div className="flex flex-col gap-1">
            <p className="text-[10px] text-slate-400 mb-1">数据输出</p>
            {Array.from({ length: Math.min(dataBits, 8) }, (_, i) => {
              const bitVal = currentData !== null ? (currentData >> (7 - i)) & 1 : null
              return (
                <div key={i} className="flex items-center gap-1">
                  <span className="text-[9px] text-slate-400">D{7 - i}</span>
                  <div className={`w-4 h-3 rounded-sm ${
                    bitVal === 1 ? 'bg-green-400' : bitVal === 0 ? 'bg-slate-300 dark:bg-slate-600' : 'bg-slate-200 dark:bg-slate-700'
                  }`} />
                </div>
              )
            })}
          </div>
        </div>
      </div>

      <div className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg text-xs text-slate-600 dark:text-slate-400">
        <p className="font-semibold mb-1">工作原理:</p>
        <p>地址译码器将A0-An译码选中一行 → CS片选使能芯片 → WE控制读/写方向 → 数据通过I/O电路输出/输入</p>
      </div>
    </div>
  )
}
