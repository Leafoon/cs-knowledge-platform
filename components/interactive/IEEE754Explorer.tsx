'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { Search } from 'lucide-react'

export function IEEE754Explorer() {
  const [input, setInput] = useState('3.14')
  const [format, setFormat] = useState<'single' | 'double'>('single')

  const num = parseFloat(input)
  const isNaNVal = Number.isNaN(num)

  const singleBits = useMemo(() => {
    if (isNaNVal) return null
    const buf = new ArrayBuffer(4)
    new DataView(buf).setFloat32(0, num)
    const hex = new DataView(buf).getUint32(0)
    return hex.toString(2).padStart(32, '0')
  }, [num, isNaNVal])

  const doubleBits = useMemo(() => {
    if (isNaNVal) return null
    const buf = new ArrayBuffer(8)
    new DataView(buf).setFloat64(0, num)
    const hi = new DataView(buf).getUint32(0)
    const lo = new DataView(buf).getUint32(4)
    return hi.toString(2).padStart(32, '0') + lo.toString(2).padStart(32, '0')
  }, [num, isNaNVal])

  const bits = format === 'single' ? singleBits : doubleBits
  const sLen = 1
  const eLen = format === 'single' ? 8 : 11
  const mL = format === 'single' ? 23 : 52
  const bias = format === 'single' ? 127 : 1023

  const sign = bits ? bits[0] : '0'
  const exponent = bits ? bits.slice(sLen, sLen + eLen) : '0'.repeat(eLen)
  const mantissa = bits ? bits.slice(sLen + eLen) : '0'.repeat(mL)

  const expVal = parseInt(exponent, 2)
  const mantVal = parseInt(mantissa, 2)

  const isZero = expVal === 0 && mantVal === 0
  const isInf = expVal === (1 << eLen) - 1 && mantVal === 0
  const isNaNValResult = expVal === (1 << eLen) - 1 && mantVal !== 0
  const isDenorm = expVal === 0 && mantVal !== 0

  const decoded = (() => {
    if (isZero) return '±0 (零)'
    if (isInf) return '±∞ (无穷大)'
    if (isNaNValResult) return 'NaN (非数)'
    if (isDenorm) return `非规格化数: (-1)^${sign} × 2^${1 - bias} × 0.${mantissa}`
    const e = expVal - bias
    return `(-1)^${sign} × 2^${e} × 1.${mantissa}`
  })()

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Search className="w-5 h-5 text-blue-500" />
        <h3 className="text-lg font-bold">IEEE 754 交互探索器</h3>
      </div>

      <div className="flex flex-wrap gap-4 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">十进制数</label>
          <input type="text" value={input} onChange={e => setInput(e.target.value)}
            className="w-32 px-3 py-1 border rounded font-mono" placeholder="3.14" />
        </div>
        <div className="flex items-end gap-1">
          {(['single', 'double'] as const).map(f => (
            <button key={f} onClick={() => setFormat(f)}
              className={`px-3 py-1 text-sm rounded ${format === f ? 'bg-blue-500 text-white' : 'bg-slate-200'}`}>
              {f === 'single' ? '单精度(32位)' : '双精度(64位)'}
            </button>
          ))}
        </div>
      </div>

      {bits && !isNaNVal && (
        <>
          <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-4 mb-4">
            <div className="flex gap-1 flex-wrap">
              <div className="flex flex-col items-center mr-2">
                <div className="text-[10px] text-red-500 font-bold mb-1">符号</div>
                <div className="flex gap-0.5">
                  {sign.split('').map((b, i) => (
                    <motion.span key={i} className="w-7 h-7 flex items-center justify-center bg-red-100 dark:bg-red-900 border border-red-300 rounded text-xs font-mono font-bold"
                      initial={{ scale: 0 }} animate={{ scale: 1 }}>{b}</motion.span>
                  ))}
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5">{sLen}位</div>
              </div>
              <div className="flex flex-col items-center mr-2">
                <div className="text-[10px] text-blue-500 font-bold mb-1">指数</div>
                <div className="flex gap-0.5">
                  {exponent.split('').map((b, i) => (
                    <motion.span key={i} className="w-7 h-7 flex items-center justify-center bg-blue-100 dark:bg-blue-900 border border-blue-300 rounded text-xs font-mono font-bold"
                      initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.02 * i }}>{b}</motion.span>
                  ))}
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5">{eLen}位</div>
              </div>
              <div className="flex flex-col items-center">
                <div className="text-[10px] text-green-500 font-bold mb-1">尾数</div>
                <div className="flex gap-0.5 flex-wrap">
                  {mantissa.split('').map((b, i) => (
                    <motion.span key={i} className="w-7 h-7 flex items-center justify-center bg-green-100 dark:bg-green-900 border border-green-300 rounded text-xs font-mono font-bold"
                      initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.01 * i }}>{b}</motion.span>
                  ))}
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5">{mL}位</div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 text-xs mb-3">
            <div className="p-2 bg-red-50 dark:bg-red-950 rounded">
              <div className="font-bold text-red-700">符号 = {sign}</div>
              <div>{sign === '0' ? '正数' : '负数'}</div>
            </div>
            <div className="p-2 bg-blue-50 dark:bg-blue-950 rounded">
              <div className="font-bold text-blue-700">指数 = {expVal} (偏移后: {expVal - bias})</div>
              <div>偏置值 = {bias}</div>
            </div>
            <div className="p-2 bg-green-50 dark:bg-green-950 rounded">
              <div className="font-bold text-green-700">尾数 = {isDenorm ? '0.' : '1.'}{mantissa}</div>
              <div>隐含前导{isDenorm ? '0' : '1'}</div>
            </div>
          </div>

          <div className="p-3 bg-yellow-50 dark:bg-yellow-950 rounded-lg text-sm font-mono">
            {decoded}
          </div>

          <div className="mt-3 text-xs text-slate-500">
            十六进制: 0x{parseInt(bits, 2).toString(16).toUpperCase().padStart(format === 'single' ? 8 : 16, '0')}
          </div>
        </>
      )}
    </div>
  )
}
