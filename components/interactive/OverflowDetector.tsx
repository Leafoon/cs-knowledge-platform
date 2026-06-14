'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { AlertTriangle } from 'lucide-react'

export function OverflowDetector() {
  const [a, setA] = useState(5)
  const [b, setB] = useState(6)
  const [bits, setBits] = useState(4)

  const lo = -(1 << (bits - 1))
  const hi = (1 << (bits - 1)) - 1

  const toTwos = (v: number) => {
    const u = v < 0 ? (1 << bits) + v : v
    return u.toString(2).padStart(bits, '0').slice(-bits)
  }

  const aBin = toTwos(a)
  const bBin = toTwos(b)
  const rawSum = a + b
  const sumBin = toTwos(rawSum)
  const sumInRange = rawSum >= lo && rawSum <= hi

  const aSign = aBin[0]
  const bSign = bBin[0]
  const sSign = sumBin[0]

  const singleBitOverflow = aSign === bSign && aSign !== sSign

  const cout = (() => {
    let carry = 0
    for (let i = bits - 1; i >= 0; i--) {
      const ab = parseInt(aBin[i])
      const bb = parseInt(bBin[i])
      carry = Math.floor((ab + bb + carry) / 2)
    }
    return carry
  })()
  const cinLast = (() => {
    let carry = 0
    for (let i = bits - 1; i >= 1; i--) {
      const ab = parseInt(aBin[i])
      const bb = parseInt(bBin[i])
      carry = Math.floor((ab + bb + carry) / 2)
    }
    return carry
  })()
  const doubleBitOverflow = cout !== cinLast

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <AlertTriangle className="w-5 h-5 text-orange-500" />
        <h3 className="text-lg font-bold">溢出检测器</h3>
      </div>

      <div className="flex flex-wrap gap-4 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">位宽</label>
          <div className="flex gap-1 mt-1">
            {[4, 8].map(b => (
              <button key={b} onClick={() => { setBits(b); setA(0); setB(0) }}
                className={`px-3 py-1 text-sm rounded ${bits === b ? 'bg-orange-500 text-white' : 'bg-slate-200'}`}>{b}-bit</button>
            ))}
          </div>
        </div>
        <div>
          <label className="text-xs text-slate-500 block">A ({lo}~{hi})</label>
          <input type="number" min={lo} max={hi} value={a}
            onChange={e => setA(Math.max(lo, Math.min(hi, Number(e.target.value))))}
            className="w-20 px-2 py-1 border rounded font-mono" />
        </div>
        <div>
          <label className="text-xs text-slate-500 block">B ({lo}~{hi})</label>
          <input type="number" min={lo} max={hi} value={b}
            onChange={e => setB(Math.max(lo, Math.min(hi, Number(e.target.value))))}
            className="w-20 px-2 py-1 border rounded font-mono" />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4 text-center text-sm font-mono">
        <div className="bg-blue-50 dark:bg-blue-950 rounded p-2">
          <div className="text-xs text-blue-600">A</div>
          <div>{aBin}</div>
          <div className="text-xs text-slate-500">= {a}</div>
        </div>
        <div className="bg-purple-50 dark:bg-purple-950 rounded p-2">
          <div className="text-xs text-purple-600">B</div>
          <div>{bBin}</div>
          <div className="text-xs text-slate-500">= {b}</div>
        </div>
        <div className={`rounded p-2 ${singleBitOverflow ? 'bg-red-100 dark:bg-red-950' : 'bg-green-50 dark:bg-green-950'}`}>
          <div className={`text-xs ${singleBitOverflow ? 'text-red-600' : 'text-green-600'}`}>结果</div>
          <div>{sumBin}</div>
          <div className="text-xs text-slate-500">= {sumInRange ? rawSum : '溢出!'}</div>
        </div>
      </div>

      <div className="space-y-3">
        <motion.div className={`p-3 rounded-lg border-2 ${singleBitOverflow ? 'border-red-400 bg-red-50 dark:bg-red-950' : 'border-green-400 bg-green-50 dark:bg-green-950'}`}
          animate={{ scale: singleBitOverflow ? [1, 1.02, 1] : 1 }}>
          <div className="text-sm font-bold mb-1">方法一: 单符号位检测</div>
          <div className="text-xs font-mono">
            A的符号={aSign}, B的符号={bSign}, 结果符号={sSign}
            <br />
            {aSign === bSign ? `同号相加，结果符号${aSign !== sSign ? '不同→溢出!' : '相同→无溢出'}` : '异号相加→不会溢出'}
          </div>
          <div className={`mt-1 font-bold ${singleBitOverflow ? 'text-red-600' : 'text-green-600'}`}>
            {singleBitOverflow ? '⚠️ 溢出!' : '✓ 无溢出'}
          </div>
        </motion.div>

        <motion.div className={`p-3 rounded-lg border-2 ${doubleBitOverflow ? 'border-red-400 bg-red-50 dark:bg-red-950' : 'border-green-400 bg-green-50 dark:bg-green-950'}`}
          animate={{ scale: doubleBitOverflow ? [1, 1.02, 1] : 1 }}>
          <div className="text-sm font-bold mb-1">方法二: 双符号位 (变形补码) 检测</div>
          <div className="text-xs font-mono">
            C<sub>out</sub>={cout}, C<sub>in最高位</sub>={cinLast}
            <br />
            C<sub>out</sub> ⊕ C<sub>in</sub> = {cout} ⊕ {cinLast} = {cout ^ cinLast}
          </div>
          <div className={`mt-1 font-bold ${doubleBitOverflow ? 'text-red-600' : 'text-green-600'}`}>
            {doubleBitOverflow ? '⚠️ 溢出! (两位符号位不同)' : '✓ 无溢出 (两位符号位相同)'}
          </div>
        </motion.div>
      </div>
    </div>
  )
}
