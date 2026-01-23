'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Target, Layers } from 'lucide-react'

export default function LoRATargetModulesSelector() {
  const [selectedModules, setSelectedModules] = useState(['q_proj', 'v_proj'])

  const modules = [
    { name: 'q_proj', category: 'attention', params: 5.2, description: 'Query 投影' },
    { name: 'k_proj', category: 'attention', params: 5.2, description: 'Key 投影' },
    { name: 'v_proj', category: 'attention', params: 5.2, description: 'Value 投影' },
    { name: 'o_proj', category: 'attention', params: 5.2, description: 'Output 投影' },
    { name: 'gate_proj', category: 'ffn', params: 13.6, description: 'FFN 门控层' },
    { name: 'up_proj', category: 'ffn', params: 13.6, description: 'FFN 上投影' },
    { name: 'down_proj', category: 'ffn', params: 13.6, description: 'FFN 下投影' },
  ]

  const toggleModule = (name: string) => {
    if (selectedModules.includes(name)) {
      setSelectedModules(selectedModules.filter(m => m !== name))
    } else {
      setSelectedModules([...selectedModules, name])
    }
  }

  const totalParams = selectedModules.reduce((sum, name) => {
    const module = modules.find(m => m.name === name)
    return sum + (module?.params || 0)
  }, 0)

  const presets = {
    minimal: ['q_proj', 'v_proj'],
    recommended: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj'],
    maximum: modules.map(m => m.name),
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-teal-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Target className="w-8 h-8 text-teal-600" />
        <h3 className="text-2xl font-bold text-slate-800">LoRA Target Modules 选择器</h3>
      </div>

      {/* 预设配置 */}
      <div className="mb-6 flex gap-3">
        {Object.entries(presets).map(([name, mods]) => (
          <button
            key={name}
            onClick={() => setSelectedModules(mods)}
            className="px-4 py-2 bg-white border-2 border-teal-300 rounded-lg hover:border-teal-600 hover:bg-teal-50"
          >
            <div className="font-bold text-teal-800 capitalize">{name}</div>
            <div className="text-xs text-slate-600">{mods.length} modules</div>
          </button>
        ))}
      </div>

      {/* 模块选择 */}
      <div className="mb-6 bg-white p-5 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-4">选择目标模块</h4>
        
        {/* Attention 层 */}
        <div className="mb-4">
          <div className="text-sm font-bold text-slate-700 mb-2">⚡ Attention 层</div>
          <div className="grid grid-cols-4 gap-2">
            {modules.filter(m => m.category === 'attention').map((module) => (
              <button
                key={module.name}
                onClick={() => toggleModule(module.name)}
                className={`p-3 rounded-lg border-2 transition-all ${
                  selectedModules.includes(module.name)
                    ? 'border-teal-600 bg-teal-50 shadow-md'
                    : 'border-slate-200 bg-white hover:border-teal-300'
                }`}
              >
                <div className={`font-mono text-sm font-bold ${
                  selectedModules.includes(module.name) ? 'text-teal-800' : 'text-slate-700'
                }`}>
                  {module.name}
                </div>
                <div className="text-xs text-slate-600 mt-1">{module.description}</div>
                <div className="text-xs text-slate-500 mt-1">+{module.params}M</div>
              </button>
            ))}
          </div>
        </div>

        {/* FFN 层 */}
        <div>
          <div className="text-sm font-bold text-slate-700 mb-2">🔷 Feed-Forward 层</div>
          <div className="grid grid-cols-3 gap-2">
            {modules.filter(m => m.category === 'ffn').map((module) => (
              <button
                key={module.name}
                onClick={() => toggleModule(module.name)}
                className={`p-3 rounded-lg border-2 transition-all ${
                  selectedModules.includes(module.name)
                    ? 'border-purple-600 bg-purple-50 shadow-md'
                    : 'border-slate-200 bg-white hover:border-purple-300'
                }`}
              >
                <div className={`font-mono text-sm font-bold ${
                  selectedModules.includes(module.name) ? 'text-purple-800' : 'text-slate-700'
                }`}>
                  {module.name}
                </div>
                <div className="text-xs text-slate-600 mt-1">{module.description}</div>
                <div className="text-xs text-slate-500 mt-1">+{module.params}M</div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 参数统计 */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="p-4 bg-white rounded-lg shadow">
          <div className="text-sm text-slate-600 mb-1">已选模块</div>
          <div className="text-3xl font-bold text-teal-600">{selectedModules.length}</div>
        </div>
        <div className="p-4 bg-white rounded-lg shadow">
          <div className="text-sm text-slate-600 mb-1">LoRA 参数量</div>
          <div className="text-3xl font-bold text-purple-600">{totalParams.toFixed(1)}M</div>
        </div>
        <div className="p-4 bg-white rounded-lg shadow">
          <div className="text-sm text-slate-600 mb-1">可训练占比</div>
          <div className="text-3xl font-bold text-blue-600">
            {((totalParams / 7000) * 100).toFixed(2)}%
          </div>
          <div className="text-xs text-slate-500">(LLaMA-7B)</div>
        </div>
      </div>

      {/* 生成代码 */}
      <div className="bg-slate-900 text-slate-100 p-4 rounded-lg font-mono text-sm">
        <div className="text-green-400 mb-2"># 生成的 LoRA 配置</div>
        <div><span className="text-blue-400">from</span> peft <span className="text-blue-400">import</span> LoraConfig</div>
        <div className="mt-2">config = LoraConfig(</div>
        <div className="ml-4">r=<span className="text-orange-400">16</span>,</div>
        <div className="ml-4">lora_alpha=<span className="text-orange-400">32</span>,</div>
        <div className="ml-4">target_modules=[</div>
        {selectedModules.map((mod, idx) => (
          <div key={mod} className="ml-8">
            <span className="text-orange-400">"{mod}"</span>{idx < selectedModules.length - 1 ? ',' : ''}
          </div>
        ))}
        <div className="ml-4">],</div>
        <div className="ml-4">lora_dropout=<span className="text-orange-400">0.05</span>,</div>
        <div>)</div>
      </div>

      {/* 建议 */}
      <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded">
        <div className="font-bold text-yellow-800 mb-2">💡 选择建议</div>
        <ul className="text-sm text-slate-700 space-y-1">
          <li>• <strong>最小配置</strong> (q_proj, v_proj): 快速实验，参数少</li>
          <li>• <strong>推荐配置</strong> (Q/K/V/O + gate): 性能与效率平衡</li>
          <li>• <strong>最大配置</strong> (所有层): 最佳性能，但显存占用高</li>
        </ul>
      </div>
    </div>
  )
}
