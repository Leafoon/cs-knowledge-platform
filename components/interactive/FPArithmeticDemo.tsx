'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Calculator } from 'lucide-react'

export function FPArithmeticDemo() {
  const [a, setA] = useState(3.75)
  const [b, setB] = useState(1.25)
  const [op, setOp] = useState<'+' | '-' | '×' | '÷'>('+')

  const compute = (x: number, y: number, operation: string) => {
    switch (operation) {
      case '+': return x + y
      case '-': return x - y
      case '×': return x * y
      case '÷': return y !== 0 ? x / y : Infinity
      default: return 0
    }
  }

  const result = compute(a, b, op)

  const toFP32 = (v: number) => {
    const buf = new ArrayBuffer(4)
    new DataView(buf).setFloat32(0, v)
    return new DataView(buf).getUint32(0)
  }

  const fpDetails = (v: number) => {
    const bits = toFP32(v).toString(2).padStart(32, '0')
    const sign = parseInt(bits[0])
    const exp = parseInt(bits.slice(1, 9), 2) - 127
    const mantissa = bits.slice(9)
    return { sign, exp, mantissa, hex: '0x' + toFP32(v).toString(16).toUpperCase().padStart(8, '0') }
  }

  const aFP = fpDetails(a)
  const bFP = fpDetails(b)
  const rFP = fpDetails(result)

  const ops = ['+', '-', '×', '÷'] as const

  const opSteps: Record<string, string[]> = {
    '+': ['对阶: 小阶向大阶看齐', '尾数相加', '规格化', '舍入', '溢出检查'],
    '-': ['对阶: 小阶向大阶看齐', '尾数相减', '规格化', '舍入', '溢出检查'],
    '×': ['指数相加: Ea+Eb-偏置', '尾数相乘', '规格化', '舍入', '溢出检查'],
    '÷': ['指数相减: Ea-Eb+偏置', '尾数相除', '规格化', '舍入', '溢出检查'],
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Calculator className="w-5 h-5 text-teal-500" />
        <h3 className="text-lg font-bold">浮点运算演示</h3>
      </div>

      <div className="flex flex-wrap gap-3 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">A</label>
          <input type="number" step="0.25" value={a}
            onChange={e => setA(Number(e.target.value))}
            className="w-24 px-2 py-1 border rounded font-mono" />
        </div>
        <div className="flex items-end gap-1">
          {ops.map(o => (
            <button key={o} onClick={() => setOp(o)}
              className={`w-10 h-10 rounded-lg text-lg font-bold ${op === o ? 'bg-teal-500 text-white' : 'bg-slate-200 hover:bg-slate-300'}`}>
              {o}
            </button>
          ))}
        </div>
        <div>
          <label className="text-xs text-slate-500 block">B</label>
          <input type="number" step="0.25" value={b}
            onChange={e => setB(Number(e.target.value))}
            className="w-24 px-2 py-1 border rounded font-mono" />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4 text-xs font-mono">
        {[
          { label: 'A', fp: aFP, val: a, color: 'blue' },
          { label: 'B', fp: bFP, val: b, color: 'purple' },
          { label: '结果', fp: rFP, val: result, color: 'green' },
        ].map(({ label, fp, val, color }) => (
          <div key={label} className={`p-2 bg-${color}-50 dark:bg-${color}-950 rounded`}>
            <div className="font-bold">{label} = {val}</div>
            <div className="text-slate-500">S={fp.sign} E={fp.exp} M=1.{fp.mantissa.slice(0, 8)}</div>
            <div className="text-slate-400">{fp.hex}</div>
          </div>
        ))}
      </div>

      <div className="text-center text-2xl font-bold font-mono mb-4">
        {a} {op} {b} = {result}
      </div>

      <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
        <div className="text-xs font-bold text-slate-600 mb-2">执行步骤</div>
        {opSteps[op].map((s, i) => (
          <motion.div key={i} className="flex items-center gap-2 text-xs py-1"
            initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1 }}>
            <div className="w-5 h-5 rounded-full bg-teal-500 text-white flex items-center justify-center text-[10px] font-bold">{i + 1}</div>
            <span>{s}</span>
          </motion.div>
        ))}
      </div>

      <div className="mt-3 p-2 bg-yellow-50 dark:bg-yellow-950 rounded text-xs text-yellow-700">
        注意: 浮点运算不满足结合律: (a+b)+c ≠ a+(b+c)，精度损失可能累积
      </div>
    </div>
  )
}
