'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Cpu, HardDrive, RefreshCw } from 'lucide-react'

interface Page {
  id: number
  data: string
  location: 'GPU' | 'CPU'
  active: boolean
}

export default function PagedOptimizerVisualizer() {
  const [step, setStep] = useState(0)
  const [gpuMemory, setGpuMemory] = useState(85)  // 使用率

  const initialPages: Page[] = [
    { id: 0, data: 'Momentum #0-4095', location: 'GPU', active: true },
    { id: 1, data: 'Variance #0-4095', location: 'GPU', active: true },
    { id: 2, data: 'Momentum #4096-8191', location: 'GPU', active: false },
    { id: 3, data: 'Variance #4096-8191', location: 'CPU', active: false },
    { id: 4, data: 'Momentum #8192-12287', location: 'CPU', active: false },
    { id: 5, data: 'Variance #8192-12287', location: 'CPU', active: false },
  ]

  const getPages = () => {
    if (step === 0) return initialPages
    if (step === 1) {
      // 模拟显存不足，开始换页
      return initialPages.map(p =>
        p.id === 2 ? { ...p, location: 'CPU' as const } : p
      )
    }
    if (step === 2) {
      // 激活新页面，从 CPU 加载
      return initialPages.map(p => {
        if (p.id === 3) return { ...p, location: 'GPU' as const, active: true }
        if (p.id === 0) return { ...p, active: false }
        return p
      })
    }
    return initialPages
  }

  const pages = getPages()
  const gpuPages = pages.filter(p => p.location === 'GPU')
  const cpuPages = pages.filter(p => p.location === 'CPU')

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <RefreshCw className="w-8 h-8 text-blue-600" />
        <h3 className="text-2xl font-bold text-slate-800">Paged Optimizers 内存管理</h3>
      </div>

      {/* 步骤控制 */}
      <div className="mb-6 flex gap-2">
        <button
          onClick={() => {
            setStep(0)
            setGpuMemory(85)
          }}
          className={`px-4 py-2 rounded ${step === 0 ? 'bg-blue-600 text-white' : 'bg-white border'}`}
        >
          初始状态
        </button>
        <button
          onClick={() => {
            setStep(1)
            setGpuMemory(95)
          }}
          className={`px-4 py-2 rounded ${step === 1 ? 'bg-blue-600 text-white' : 'bg-white border'}`}
        >
          显存压力
        </button>
        <button
          onClick={() => {
            setStep(2)
            setGpuMemory(88)
          }}
          className={`px-4 py-2 rounded ${step === 2 ? 'bg-blue-600 text-white' : 'bg-white border'}`}
        >
          自动换页
        </button>
      </div>

      {/* GPU 显存 */}
      <div className="mb-6 bg-white p-5 rounded-lg shadow">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-green-600" />
            <h4 className="font-bold text-slate-800">GPU 显存 (24 GB)</h4>
          </div>
          <div className={`text-lg font-bold ${gpuMemory > 90 ? 'text-red-600' : 'text-green-600'}`}>
            {gpuMemory}%
          </div>
        </div>

        <div className="relative h-6 bg-slate-100 rounded-full overflow-hidden mb-4">
          <motion.div
            animate={{ width: `${gpuMemory}%` }}
            className={`h-full ${
              gpuMemory > 90 ? 'bg-red-500' : 'bg-green-500'
            }`}
          />
        </div>

        <div className="grid grid-cols-3 gap-2">
          {gpuPages.map((page) => (
            <motion.div
              key={page.id}
              layout
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className={`p-3 rounded border-2 ${
                page.active
                  ? 'border-green-500 bg-green-50'
                  : 'border-blue-300 bg-blue-50'
              }`}
            >
              <div className="text-xs text-slate-600 font-bold mb-1">
                Page {page.id}
              </div>
              <div className="text-xs text-slate-700">{page.data}</div>
              {page.active && (
                <div className="mt-1 text-xs text-green-600 font-bold">● Active</div>
              )}
            </motion.div>
          ))}
        </div>
      </div>

      {/* CPU 内存 */}
      <div className="bg-white p-5 rounded-lg shadow">
        <div className="flex items-center gap-2 mb-3">
          <HardDrive className="w-5 h-5 text-purple-600" />
          <h4 className="font-bold text-slate-800">CPU 内存 (Offloaded Pages)</h4>
        </div>

        {cpuPages.length > 0 ? (
          <div className="grid grid-cols-3 gap-2">
            {cpuPages.map((page) => (
              <motion.div
                key={page.id}
                layout
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-3 rounded border-2 border-purple-300 bg-purple-50"
              >
                <div className="text-xs text-slate-600 font-bold mb-1">
                  Page {page.id}
                </div>
                <div className="text-xs text-slate-700">{page.data}</div>
                <div className="mt-1 text-xs text-purple-600">💾 CPU</div>
              </motion.div>
            ))}
          </div>
        ) : (
          <div className="text-center text-slate-400 py-4">
            所有页面都在 GPU
          </div>
        )}
      </div>

      {/* 说明 */}
      <div className="mt-6 grid grid-cols-2 gap-4">
        <div className="p-4 bg-green-50 border border-green-200 rounded">
          <div className="font-bold text-green-800 mb-2">✓ Paged Optimizers 优势</div>
          <ul className="text-sm text-slate-700 space-y-1">
            <li>• 自动 CPU-GPU 页面交换</li>
            <li>• 避免显存峰值 OOM</li>
            <li>• 无需手动管理 offload</li>
            <li>• 训练速度仅降低 5-10%</li>
          </ul>
        </div>
        <div className="p-4 bg-blue-50 border border-blue-200 rounded">
          <div className="font-bold text-blue-800 mb-2">📊 性能影响</div>
          <ul className="text-sm text-slate-700 space-y-1">
            <li>• 页面大小: 4KB</li>
            <li>• 换页延迟: ~5ms</li>
            <li>• LLaMA-65B QLoRA: 峰值 48GB → 24GB</li>
            <li>• 训练速度: 15 it/s → 14 it/s</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
