'use client'

import React from 'react'

const methods = [
  { name: 'GPTQ', bits: 4, speed: 95, accuracy: 99, latency: '1.2x', color: 'blue' },
  { name: 'AWQ', bits: 4, speed: 100, accuracy: 99.5, latency: '1.0x', color: 'green' },
  { name: 'NF4 (QLoRA)', bits: 4, speed: 70, accuracy: 98, latency: '1.5x', color: 'purple' },
  { name: 'INT8', bits: 8, speed: 85, accuracy: 99.8, latency: '1.3x', color: 'orange' }
]

export default function PTQMethodComparison() {
  return (
    <div className="my-8 p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <h3 className="text-xl font-bold mb-6">后训练量化方法对比</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-slate-100 dark:bg-slate-800">
            <tr>
              <th className="px-4 py-2">方法</th>
              <th className="px-4 py-2">位数</th>
              <th className="px-4 py-2">速度</th>
              <th className="px-4 py-2">精度保持</th>
              <th className="px-4 py-2">延迟</th>
            </tr>
          </thead>
          <tbody>
            {methods.map(m => (
              <tr key={m.name} className="border-b dark:border-slate-700">
                <td className={`px-4 py-2 font-bold text-${m.color}-600`}>{m.name}</td>
                <td className="px-4 py-2">{m.bits}-bit</td>
                <td className="px-4 py-2">{m.speed}%</td>
                <td className="px-4 py-2">{m.accuracy}%</td>
                <td className="px-4 py-2">{m.latency}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
