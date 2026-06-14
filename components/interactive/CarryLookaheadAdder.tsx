'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Zap } from 'lucide-react'

export function CarryLookaheadAdder() {
  const [a, setA] = useState(0b1011)
  const [b, setB] = useState(0b0110)
  const [showCLA, setShowCLA] = useState(false)
  const [step, setStep] = useState(0)

  const bits = 4
  const aBits = Array.from({ length: bits }, (_, i) => (a >> i) & 1)
  const bBits = Array.from({ length: bits }, (_, i) => (b >> i) & 1)

  const P = aBits.map((ai, i) => ai ^ bBits[i])
  const G = aBits.map((ai, i) => ai & bBits[i])

  const C = Array(bits + 1)
  C[0] = 0
  for (let i = 0; i < bits; i++) C[i + 1] = G[i] | (P[i] & C[i])

  const S = aBits.map((_, i) => P[i] ^ C[i])

  useEffect(() => {
    if (showCLA && step < 3) {
      const t = setTimeout(() => setStep(s => s + 1), 1000)
      return () => clearTimeout(t)
    }
  }, [showCLA, step])

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Zap className="w-5 h-5 text-yellow-500" />
        <h3 className="text-lg font-bold">超前进位加法器 (CLA)</h3>
      </div>
      <p className="text-sm text-slate-500 mb-4">并行生成进位信号，消除串行进位延迟</p>

      <div className="flex gap-4 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">A</label>
          <input type="number" min={0} max={15} value={a}
            onChange={e => { setA(Number(e.target.value) & 0xF); setStep(0); setShowCLA(false) }}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
        <div>
          <label className="text-xs text-slate-500 block">B</label>
          <input type="number" min={0} max={15} value={b}
            onChange={e => { setB(Number(e.target.value) & 0xF); setStep(0); setShowCLA(false) }}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
        <button onClick={() => { setShowCLA(true); setStep(0) }}
          className="self-end px-3 py-1 bg-yellow-500 text-white rounded text-sm">演示</button>
      </div>

      <div className="grid grid-cols-4 gap-2 mb-4 text-xs font-mono text-center">
        {Array.from({ length: bits }, (_, i) => (
          <div key={i} className="text-slate-400">位 {i}</div>
        ))}
        {aBits.map((v, i) => <div key={`a${i}`} className="bg-blue-100 dark:bg-blue-900 rounded py-1">A={v}</div>)}
        {bBits.map((v, i) => <div key={`b${i}`} className="bg-purple-100 dark:bg-purple-900 rounded py-1">B={v}</div>)}
      </div>

      <motion.div className="space-y-2 mb-4"
        initial={{ opacity: 0 }} animate={{ opacity: showCLA ? 1 : 0.3 }}>
        <div className="text-xs font-medium text-slate-600 mb-1">Step 1: 并行计算 P (传播) 和 G (生成)</div>
        <div className="grid grid-cols-4 gap-2 text-xs font-mono text-center">
          {P.map((v, i) => (
            <motion.div key={`p${i}`} className="bg-green-100 dark:bg-green-900 rounded py-1"
              animate={{ scale: showCLA && step >= 0 ? [1, 1.1, 1] : 1, opacity: showCLA ? 1 : 0.3 }}
              transition={{ delay: i * 0.1 }}>P{i}={v}</motion.div>
          ))}
          {G.map((v, i) => (
            <motion.div key={`g${i}`} className="bg-orange-100 dark:bg-orange-900 rounded py-1"
              animate={{ scale: showCLA && step >= 0 ? [1, 1.1, 1] : 1, opacity: showCLA ? 1 : 0.3 }}
              transition={{ delay: i * 0.1 + 0.3 }}>G{i}={v}</motion.div>
          ))}
        </div>
      </motion.div>

      <motion.div className="space-y-2 mb-4"
        initial={{ opacity: 0 }} animate={{ opacity: showCLA && step >= 1 ? 1 : 0.3 }}>
        <div className="text-xs font-medium text-slate-600 mb-1">Step 2: 并行计算所有进位 C</div>
        <div className="flex gap-2 text-xs font-mono text-center">
          {C.map((v, i) => (
            <motion.div key={`c${i}`} className="flex-1 bg-yellow-100 dark:bg-yellow-900 rounded py-1"
              animate={{ scale: showCLA && step >= 1 ? [1, 1.15, 1] : 1 }}
              transition={{ delay: i * 0.15 }}>C{i}={v}</motion.div>
          ))}
        </div>
      </motion.div>

      <motion.div className="space-y-2"
        initial={{ opacity: 0 }} animate={{ opacity: showCLA && step >= 2 ? 1 : 0.3 }}>
        <div className="text-xs font-medium text-slate-600 mb-1">Step 3: 计算和 S = P ⊕ C</div>
        <div className="grid grid-cols-4 gap-2 text-xs font-mono text-center">
          {S.map((v, i) => (
            <motion.div key={`s${i}`} className="bg-green-200 dark:bg-green-800 rounded py-1 font-bold"
              animate={{ scale: showCLA && step >= 2 ? [1, 1.2, 1] : 1 }}
              transition={{ delay: i * 0.1 }}>S{i}={v}</motion.div>
          ))}
        </div>
      </motion.div>

      <motion.div className="mt-4 p-3 bg-yellow-50 dark:bg-yellow-950 rounded-lg text-center"
        animate={{ opacity: showCLA && step >= 3 ? 1 : 0.3 }}>
        <span className="text-sm font-medium text-yellow-700">
          结果: {S.slice().reverse().join('')} = {parseInt(S.slice().reverse().join(''), 2)} &nbsp;
          (进位延迟: CLA = O(log n) vs 串行 = O(n))
        </span>
      </motion.div>
    </div>
  )
}
