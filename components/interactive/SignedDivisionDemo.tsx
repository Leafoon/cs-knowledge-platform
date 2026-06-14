'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Hash } from 'lucide-react'

export function SignedDivisionDemo() {
  const [dividend, setDividend] = useState(-13)
  const [divisor, setDivisor] = useState(3)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)

  const signA = dividend < 0 ? 1 : 0
  const signB = divisor < 0 ? 1 : 0
  const resultSign = signA ^ signB

  const absDividend = Math.abs(dividend)
  const absDivisor = Math.abs(divisor)
  const absQuotient = Math.floor(absDividend / absDivisor)
  const absRemainder = absDividend % absDivisor

  const quotient = resultSign ? -absQuotient : absQuotient
  const remainder = signA ? -absRemainder : absRemainder

  const steps = [
    { title: '提取符号', desc: `被除数符号=${signA}, 除数符号=${signB}`, detail: `${dividend}→符号${signA}, ${divisor}→符号${signB}` },
    { title: '取绝对值', desc: `|${dividend}|=${absDividend}, |${divisor}|=${absDivisor}`, detail: '转为正数进行无符号除法' },
    { title: '无符号除法', desc: `${absDividend} ÷ ${absDivisor} = ${absQuotient} 余 ${absRemainder}`, detail: '使用恢复余数法或不恢复余数法' },
    { title: '商的符号', desc: `${signA} ⊕ ${signB} = ${resultSign}`, detail: resultSign ? '异号→商为负' : '同号→商为正' },
    { title: '余数符号', desc: `与被除数同号: ${signA}`, detail: `余数 = ${signA ? '-' : ''}${absRemainder} = ${remainder}` },
    { title: '校正结果', desc: `商 = ${quotient}, 余数 = ${remainder}`, detail: `验证: ${divisor}×${quotient}+${remainder} = ${divisor * quotient + remainder}` },
  ]

  useEffect(() => {
    if (playing && step < steps.length - 1) {
      const t = setTimeout(() => setStep(s => s + 1), 1200)
      return () => clearTimeout(t)
    }
    setPlaying(false)
  }, [playing, step, steps.length])

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Hash className="w-5 h-5 text-rose-500" />
        <h3 className="text-lg font-bold">补码除法 (有符号除法)</h3>
      </div>
      <p className="text-sm text-slate-500 mb-4">符号处理和校正步骤</p>

      <div className="flex flex-wrap gap-4 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">被除数</label>
          <input type="number" min={-15} max={15} value={dividend}
            onChange={e => { setDividend(Number(e.target.value)); setStep(0) }}
            className="w-20 px-2 py-1 border rounded font-mono" />
        </div>
        <div>
          <label className="text-xs text-slate-500 block">除数 (≠0)</label>
          <input type="number" min={-15} max={15} value={divisor}
            onChange={e => { const v = Number(e.target.value); if (v !== 0) setDivisor(v); setStep(0) }}
            className="w-20 px-2 py-1 border rounded font-mono" />
        </div>
        <button onClick={() => { setStep(0); setPlaying(true) }}
          className="self-end px-3 py-1 bg-rose-500 text-white rounded text-sm">开始</button>
      </div>

      <div className="text-center text-lg font-mono mb-4">
        {dividend} ÷ {divisor} = {quotient} 余 {remainder}
      </div>

      <div className="space-y-2">
        {steps.map((s, i) => (
          <motion.div key={i}
            className={`p-3 rounded-lg border ${step >= i ? 'border-rose-300 bg-rose-50 dark:bg-rose-950' : 'border-slate-200 opacity-20'}`}
            animate={{ opacity: step >= i ? 1 : 0.2, x: step === i ? 5 : 0 }}>
            <div className="flex items-center gap-2">
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${step >= i ? 'bg-rose-500 text-white' : 'bg-slate-300'}`}>{i + 1}</div>
              <div className="font-bold text-sm">{s.title}</div>
            </div>
            <div className="ml-8 text-sm">{s.desc}</div>
            <div className="ml-8 text-xs text-slate-500">{s.detail}</div>
          </motion.div>
        ))}
      </div>

      <motion.div className="mt-4 p-3 bg-green-50 dark:bg-green-950 rounded-lg text-center"
        animate={{ opacity: step >= steps.length - 1 ? 1 : 0.3 }}>
        <div className="text-sm font-mono">
          验证: {divisor} × ({quotient}) + ({remainder}) = {divisor * quotient + remainder} = {dividend} ✓
        </div>
      </motion.div>
    </div>
  )
}
