'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowRight } from 'lucide-react'

export default function MixedPrecisionTrainingFlow() {
  const [step, setStep] = useState(0)

  const steps = [
    { title: 'FP32主权重', desc: '保持FP32精度', color: 'blue' },
    { title: 'FP16前向传播', desc: '转换为FP16计算', color: 'purple' },
    { title: 'Loss缩放', desc: '防止梯度下溢', color: 'green' },
    { title: 'FP16反向传播', desc: 'FP16计算梯度', color: 'orange' },
    { title: 'FP32更新', desc: '用FP32更新权重', color: 'blue' }
  ]

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-sky-50 to-blue-50 dark:from-slate-900 dark:to-sky-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <h3 className="text-xl font-bold mb-6">混合精度训练流程</h3>
      <div className="flex items-center justify-between mb-6">
        {steps.map((s, idx) => (
          <React.Fragment key={idx}>
            <div onClick={() => setStep(idx)} className={`p-3 rounded-lg cursor-pointer ${step === idx ? `bg-${s.color}-100 dark:bg-${s.color}-900/30` : 'bg-slate-100'}`}>
              <div className="text-sm font-bold">{s.title}</div>
              <div className="text-xs text-slate-500">{s.desc}</div>
            </div>
            {idx < steps.length - 1 && <ArrowRight className="w-5 h-5 text-slate-400" />}
          </React.Fragment>
        ))}
      </div>
      <div className="p-4 bg-slate-900 rounded-lg">
        <div className="text-green-400 font-mono text-sm">
          with autocast():<br/>
          &nbsp;&nbsp;outputs = model(**inputs)  # FP16<br/>
          &nbsp;&nbsp;loss = outputs.loss * 65536  # 缩放<br/>
          scaler.scale(loss).backward()  # FP16梯度<br/>
          scaler.step(optimizer)  # FP32更新
        </div>
      </div>
    </div>
  )
}
