'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Zap, Clock, BarChart3 } from 'lucide-react'

type CompileMode = 'none' | 'default' | 'reduce-overhead' | 'max-autotune'

interface BenchmarkResult {
  mode: CompileMode
  name: string
  compileTime: number
  firstRunTime: number
  avgRunTime: number
  speedup: number
  color: string
}

export default function TorchCompileSpeedupChart() {
  const [selectedModel, setSelectedModel] = useState<'gpt2' | 'llama-7b'>('gpt2')

  const benchmarks: Record<string, BenchmarkResult[]> = {
    'gpt2': [
      { mode: 'none', name: '未编译', compileTime: 0, firstRunTime: 100, avgRunTime: 100, speedup: 1.0, color: 'slate' },
      { mode: 'default', name: 'default', compileTime: 15, firstRunTime: 115, avgRunTime: 75, speedup: 1.33, color: 'blue' },
      { mode: 'reduce-overhead', name: 'reduce-overhead', compileTime: 8, firstRunTime: 108, avgRunTime: 82, speedup: 1.22, color: 'green' },
      { mode: 'max-autotune', name: 'max-autotune', compileTime: 45, firstRunTime: 145, avgRunTime: 62, speedup: 1.61, color: 'purple' },
    ],
    'llama-7b': [
      { mode: 'none', name: '未编译', compileTime: 0, firstRunTime: 450, avgRunTime: 450, speedup: 1.0, color: 'slate' },
      { mode: 'default', name: 'default', compileTime: 35, firstRunTime: 485, avgRunTime: 320, speedup: 1.41, color: 'blue' },
      { mode: 'reduce-overhead', name: 'reduce-overhead', compileTime: 20, firstRunTime: 470, avgRunTime: 360, speedup: 1.25, color: 'green' },
      { mode: 'max-autotune', name: 'max-autotune', compileTime: 120, firstRunTime: 570, avgRunTime: 280, speedup: 1.61, color: 'purple' },
    ]
  }

  const currentBenchmarks = benchmarks[selectedModel]
  const maxRunTime = Math.max(...currentBenchmarks.map(b => b.avgRunTime))

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Zap className="w-8 h-8 text-indigo-600" />
        <h3 className="text-2xl font-bold text-slate-800">torch.compile 性能对比</h3>
      </div>

      {/* 模型选择 */}
      <div className="flex gap-4 mb-6">
        <button
          onClick={() => setSelectedModel('gpt2')}
          className={`flex-1 p-3 rounded-lg border-2 transition-all ${
            selectedModel === 'gpt2'
              ? 'border-indigo-600 bg-indigo-50 shadow-md'
              : 'border-slate-200 bg-white hover:border-indigo-300'
          }`}
        >
          <div className={`font-bold ${selectedModel === 'gpt2' ? 'text-indigo-900' : 'text-slate-700'}`}>
            GPT-2 (124M)
          </div>
        </button>
        <button
          onClick={() => setSelectedModel('llama-7b')}
          className={`flex-1 p-3 rounded-lg border-2 transition-all ${
            selectedModel === 'llama-7b'
              ? 'border-indigo-600 bg-indigo-50 shadow-md'
              : 'border-slate-200 bg-white hover:border-indigo-300'
          }`}
        >
          <div className={`font-bold ${selectedModel === 'llama-7b' ? 'text-indigo-900' : 'text-slate-700'}`}>
            LLaMA-7B (7B)
          </div>
        </button>
      </div>

      {/* 柱状图 */}
      <div className="bg-white p-6 rounded-lg shadow mb-6">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="w-5 h-5 text-indigo-600" />
          <h4 className="font-bold text-slate-800">推理延迟对比 (ms)</h4>
        </div>

        <div className="space-y-4">
          {currentBenchmarks.map((benchmark, idx) => (
            <motion.div
              key={benchmark.mode}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
            >
              <div className="flex items-center gap-3 mb-2">
                <div className="w-40">
                  <div className="font-bold text-slate-800">{benchmark.name}</div>
                  <div className="text-xs text-slate-500">
                    {benchmark.mode !== 'none' && `编译: ${benchmark.compileTime}s`}
                  </div>
                </div>

                <div className="flex-1">
                  <div className="relative">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${(benchmark.avgRunTime / maxRunTime) * 100}%` }}
                      transition={{ duration: 0.8, delay: idx * 0.1 + 0.2 }}
                      className={`h-10 bg-gradient-to-r from-${benchmark.color}-400 to-${benchmark.color}-600 rounded flex items-center justify-end px-3`}
                    >
                      <span className="text-white font-bold text-sm">
                        {benchmark.avgRunTime} ms
                      </span>
                    </motion.div>
                  </div>
                </div>

                <div className="w-24 text-right">
                  <div className={`text-lg font-bold ${
                    benchmark.speedup > 1 ? 'text-green-600' : 'text-slate-600'
                  }`}>
                    {benchmark.speedup.toFixed(2)}x
                  </div>
                  <div className="text-xs text-slate-500">加速比</div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* 详细对比表 */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* 编译时间 */}
        <div className="bg-white p-5 rounded-lg shadow">
          <div className="flex items-center gap-2 mb-3">
            <Clock className="w-5 h-5 text-orange-600" />
            <h4 className="font-bold text-slate-800">编译时间开销</h4>
          </div>
          <div className="space-y-2">
            {currentBenchmarks.filter(b => b.mode !== 'none').map((benchmark) => (
              <div key={benchmark.mode} className="flex justify-between items-center p-2 bg-slate-50 rounded">
                <span className="text-sm text-slate-700">{benchmark.name}</span>
                <span className={`font-bold text-${benchmark.color}-600`}>
                  {benchmark.compileTime}s
                </span>
              </div>
            ))}
          </div>
          <div className="mt-3 p-2 bg-orange-50 border border-orange-200 rounded text-xs text-slate-600">
            ⚠️ 首次运行会触发编译（耗时），后续运行直接使用编译后的kernel
          </div>
        </div>

        {/* 首次运行时间 */}
        <div className="bg-white p-5 rounded-lg shadow">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="w-5 h-5 text-blue-600" />
            <h4 className="font-bold text-slate-800">首次运行时间</h4>
          </div>
          <div className="space-y-2">
            {currentBenchmarks.map((benchmark) => (
              <div key={benchmark.mode} className="flex justify-between items-center p-2 bg-slate-50 rounded">
                <span className="text-sm text-slate-700">{benchmark.name}</span>
                <span className={`font-bold text-${benchmark.color}-600`}>
                  {benchmark.firstRunTime} ms
                </span>
              </div>
            ))}
          </div>
          <div className="mt-3 p-2 bg-blue-50 border border-blue-200 rounded text-xs text-slate-600">
            包含编译时间 + 实际推理时间
          </div>
        </div>
      </div>

      {/* 模式推荐 */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-4">模式选择建议</h4>
        <div className="grid grid-cols-3 gap-4">
          <div className="p-4 bg-blue-50 border border-blue-200 rounded">
            <div className="font-bold text-blue-800 mb-2">default</div>
            <div className="text-sm text-slate-700 mb-2">
              • 编译速度: 中等
              <br />
              • 加速效果: 30-40%
              <br />
              • 适用: 通用场景
            </div>
            <div className="text-xs text-blue-600 font-bold">推荐首选</div>
          </div>

          <div className="p-4 bg-green-50 border border-green-200 rounded">
            <div className="font-bold text-green-800 mb-2">reduce-overhead</div>
            <div className="text-sm text-slate-700 mb-2">
              • 编译速度: 快
              <br />
              • 加速效果: 20-25%
              <br />
              • 适用: 低延迟推理
            </div>
            <div className="text-xs text-green-600 font-bold">在线服务</div>
          </div>

          <div className="p-4 bg-purple-50 border border-purple-200 rounded">
            <div className="font-bold text-purple-800 mb-2">max-autotune</div>
            <div className="text-sm text-slate-700 mb-2">
              • 编译速度: 慢
              <br />
              • 加速效果: 40-60%
              <br />
              • 适用: 批量推理
            </div>
            <div className="text-xs text-purple-600 font-bold">高吞吐场景</div>
          </div>
        </div>
      </div>

      {/* 代码示例 */}
      <div className="mt-6 bg-slate-900 text-slate-100 p-4 rounded-lg font-mono text-sm">
        <div className="text-green-400 mb-2"># 启用 torch.compile</div>
        <div><span className="text-blue-400">import</span> torch</div>
        <div className="mt-2">model = <span className="text-yellow-400">AutoModelForCausalLM</span>.from_pretrained(<span className="text-orange-400">"gpt2"</span>)</div>
        <div className="mt-2"><span className="text-purple-400">compiled_model</span> = torch.<span className="text-yellow-400">compile</span>(</div>
        <div className="ml-4">model,</div>
        <div className="ml-4">mode=<span className="text-orange-400">"reduce-overhead"</span>,</div>
        <div className="ml-4">fullgraph=<span className="text-blue-400">True</span></div>
        <div>)</div>
        <div className="mt-2 text-slate-400"># 首次运行会编译（慢）</div>
        <div>outputs = <span className="text-purple-400">compiled_model</span>.generate(**inputs)</div>
      </div>
    </div>
  )
}
