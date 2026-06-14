'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Plus } from 'lucide-react'

export function FloatingPointAdder() {
  const [valA, setValA] = useState(1.5)
  const [valB, setValB] = useState(0.25)
  const [step, setStep] = useState(-1)
  const [playing, setPlaying] = useState(false)

  const toFP32 = (v: number) => {
    const buf = new ArrayBuffer(4)
    new DataView(buf).setFloat32(0, v)
    return new DataView(buf).getUint32(0).toString(2).padStart(32, '0')
  }

  const decompose = (v: number) => {
    const bits = toFP32(v)
    const sign = parseInt(bits[0])
    const exp = parseInt(bits.slice(1, 9), 2) - 127
    const mantissa = '1' + bits.slice(9)
    return { sign, exp, mantissa, bits }
  }

  const a = decompose(valA)
  const b = decompose(valB)

  const maxExp = Math.max(a.exp, b.exp)
  const shiftA = maxExp - a.exp
  const shiftB = maxExp - b.exp

  const alignedA = '0'.repeat(shiftA) + a.mantissa
  const alignedB = '0'.repeat(shiftB) + b.mantissa

  const sumMantissa = (a.sign ? -1 : 1) * parseInt(alignedA, 2) + (b.sign ? -1 : 1) * parseInt(alignedB, 2)
  const sumSign = sumMantissa < 0 ? 1 : 0
  const absSum = Math.abs(sumMantissa)

  const steps = [
    { title: '① 对阶', desc: `小阶向大阶看齐: ΔE = ${maxExp}`, detail: `A指数=${a.exp}, B指数=${b.exp}, 大阶=${maxExp}` },
    { title: '② 尾数相加', desc: `${sumSign ? '减' : '加'}法运算`, detail: `${alignedA} ${sumSign ? '-' : '+'} ${alignedB}` },
    { title: '③ 规格化', desc: '调整阶码使尾数规范化', detail: `前导0数需调整` },
    { title: '④ 舍入', desc: '按IEEE 754规则舍入', detail: '可能丢失精度' },
    { title: '⑤ 溢出检查', desc: '检查指数是否溢出', detail: 'E ∈ [-126, 127]' },
  ]

  useEffect(() => {
    if (playing && step < steps.length - 1) {
      const t = setTimeout(() => setStep(s => s + 1), 1200)
      return () => clearTimeout(t)
    }
    setPlaying(false)
  }, [playing, step, steps.length])

  const result = valA + valB

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Plus className="w-5 h-5 text-green-500" />
        <h3 className="text-lg font-bold">浮点加法器</h3>
      </div>
      <p className="text-sm text-slate-500 mb-4">逐步展示对阶→尾数加→规格化→舍入</p>

      <div className="flex flex-wrap gap-4 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">操作数 A</label>
          <input type="number" step="0.1" value={valA}
            onChange={e => { setValA(Number(e.target.value)); setStep(-1) }}
            className="w-24 px-2 py-1 border rounded font-mono" />
        </div>
        <div>
          <label className="text-xs text-slate-500 block">操作数 B</label>
          <input type="number" step="0.1" value={valB}
            onChange={e => { setValB(Number(e.target.value)); setStep(-1) }}
            className="w-24 px-2 py-1 border rounded font-mono" />
        </div>
        <button onClick={() => { setStep(-1); setPlaying(true) }}
          className="self-end px-3 py-1 bg-green-500 text-white rounded text-sm">开始</button>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4 text-xs font-mono">
        <div className="p-2 bg-blue-50 dark:bg-blue-950 rounded">
          <div className="font-bold">A = {valA}</div>
          <div>符号={a.sign} 指数={a.exp} 尾数=1.{a.mantissa.slice(1, 9)}...</div>
        </div>
        <div className="p-2 bg-purple-50 dark:bg-purple-950 rounded">
          <div className="font-bold">B = {valB}</div>
          <div>符号={b.sign} 指数={b.exp} 尾数=1.{b.mantissa.slice(1, 9)}...</div>
        </div>
      </div>

      <div className="space-y-2">
        {steps.map((s, i) => (
          <motion.div key={i}
            className={`p-3 rounded-lg border ${step >= i ? 'border-green-300 bg-green-50 dark:bg-green-950' : 'border-slate-200 opacity-20'}`}
            animate={{ opacity: step >= i ? 1 : 0.2 }}>
            <div className="flex items-center gap-2">
              <motion.div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${step >= i ? 'bg-green-500 text-white' : 'bg-slate-300'}`}
                animate={{ scale: step === i ? 1.3 : 1 }}>{i + 1}</motion.div>
              <div className="font-bold text-sm">{s.title}</div>
            </div>
            <div className="ml-8 text-sm">{s.desc}</div>
            <div className="ml-8 text-xs text-slate-500 font-mono">{s.detail}</div>
          </motion.div>
        ))}
      </div>

      <motion.div className="mt-3 p-3 bg-green-100 dark:bg-green-900 rounded-lg text-center"
        animate={{ opacity: step >= steps.length - 1 ? 1 : 0.3 }}>
        <div className="font-bold">{valA} + {valB} = {result}</div>
        <div className="text-xs font-mono mt-1">{toFP32(result)}</div>
      </motion.div>
    </div>
  )
}
