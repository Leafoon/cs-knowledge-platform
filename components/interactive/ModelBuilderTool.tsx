'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Trash2, Plus, Download, Layers, Zap, Grid3x3 } from 'lucide-react'

// 可用的模型组件类型
type ComponentType = 
  | 'embedding' 
  | 'attention-standard' 
  | 'attention-local' 
  | 'attention-strided'
  | 'ffn-gelu'
  | 'ffn-relu'
  | 'pooler-cls'
  | 'pooler-mean'
  | 'classification-head'

interface ModelComponent {
  id: string
  type: ComponentType
  name: string
  params: number  // 参数量 (M)
  color: string
}

interface ComponentTemplate {
  type: ComponentType
  name: string
  description: string
  params: number
  color: string
  icon: string
}

const componentTemplates: ComponentTemplate[] = [
  {
    type: 'embedding',
    name: 'Embedding Layer',
    description: 'Token + Position Embeddings',
    params: 23.4,  // vocab_size=30k * hidden=768
    color: 'from-purple-500 to-pink-500',
    icon: '📝'
  },
  {
    type: 'attention-standard',
    name: 'Standard Attention',
    description: 'Full Multi-Head Attention (O(n²))',
    params: 2.4,  // 4 * hidden² for Q,K,V,O
    color: 'from-blue-500 to-cyan-500',
    icon: '🎯'
  },
  {
    type: 'attention-local',
    name: 'Local Window Attention',
    description: 'Attends within window_size (O(n×w))',
    params: 2.4,
    color: 'from-green-500 to-emerald-500',
    icon: '🪟'
  },
  {
    type: 'attention-strided',
    name: 'Strided Attention',
    description: 'Attends every stride tokens (Sparse)',
    params: 2.4,
    color: 'from-yellow-500 to-orange-500',
    icon: '⚡'
  },
  {
    type: 'ffn-gelu',
    name: 'FFN (GELU)',
    description: 'Feed-Forward Network with GELU',
    params: 4.7,  // 2 * hidden * intermediate
    color: 'from-indigo-500 to-purple-500',
    icon: '🔥'
  },
  {
    type: 'ffn-relu',
    name: 'FFN (ReLU)',
    description: 'Feed-Forward Network with ReLU',
    params: 4.7,
    color: 'from-red-500 to-pink-500',
    icon: '⚡'
  },
  {
    type: 'pooler-cls',
    name: '[CLS] Pooler',
    description: 'Extract first token representation',
    params: 0.6,
    color: 'from-teal-500 to-cyan-500',
    icon: '🎪'
  },
  {
    type: 'pooler-mean',
    name: 'Mean Pooler',
    description: 'Average all token embeddings',
    params: 0.0,  // No parameters
    color: 'from-lime-500 to-green-500',
    icon: '📊'
  },
  {
    type: 'classification-head',
    name: 'Classification Head',
    description: 'Linear layer + Softmax',
    params: 0.002,  // hidden * num_classes
    color: 'from-rose-500 to-red-500',
    icon: '🎯'
  }
]

export default function ModelBuilderTool() {
  const [components, setComponents] = useState<ModelComponent[]>([])
  const [selectedTemplate, setSelectedTemplate] = useState<ComponentTemplate | null>(null)

  // 添加组件到模型
  const addComponent = (template: ComponentTemplate) => {
    const newComponent: ModelComponent = {
      id: `${template.type}-${Date.now()}`,
      type: template.type,
      name: template.name,
      params: template.params,
      color: template.color
    }
    setComponents([...components, newComponent])
    setSelectedTemplate(null)
  }

  // 删除组件
  const removeComponent = (id: string) => {
    setComponents(components.filter(c => c.id !== id))
  }

  // 清空模型
  const clearModel = () => {
    setComponents([])
  }

  // 计算总参数量
  const totalParams = components.reduce((sum, c) => sum + c.params, 0)

  // 生成 PyTorch 代码
  const generateCode = () => {
    if (components.length === 0) return 'No components added yet.'

    let code = `import torch\nimport torch.nn as nn\nimport math\n\n`
    code += `class CustomModel(nn.Module):\n`
    code += `    def __init__(self, config):\n`
    code += `        super().__init__()\n`
    code += `        \n`

    // 生成每个组件的初始化代码
    components.forEach((comp, idx) => {
      switch (comp.type) {
        case 'embedding':
          code += `        # Embedding Layer\n`
          code += `        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)\n`
          code += `        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)\n\n`
          break
        case 'attention-standard':
          code += `        # Standard Multi-Head Attention (Layer ${idx})\n`
          code += `        self.attention_${idx} = nn.MultiheadAttention(\n`
          code += `            config.hidden_size,\n`
          code += `            num_heads=config.num_attention_heads,\n`
          code += `            dropout=config.attention_dropout\n`
          code += `        )\n\n`
          break
        case 'attention-local':
          code += `        # Local Window Attention (Layer ${idx})\n`
          code += `        self.local_attention_${idx} = LocalWindowAttention(\n`
          code += `            config, window_size=128\n`
          code += `        )\n\n`
          break
        case 'attention-strided':
          code += `        # Strided Attention (Layer ${idx})\n`
          code += `        self.strided_attention_${idx} = StridedAttention(\n`
          code += `            config, stride=64\n`
          code += `        )\n\n`
          break
        case 'ffn-gelu':
        case 'ffn-relu':
          const activation = comp.type === 'ffn-gelu' ? 'GELU' : 'ReLU'
          code += `        # Feed-Forward Network (Layer ${idx})\n`
          code += `        self.ffn_${idx} = nn.Sequential(\n`
          code += `            nn.Linear(config.hidden_size, config.intermediate_size),\n`
          code += `            nn.${activation}(),\n`
          code += `            nn.Linear(config.intermediate_size, config.hidden_size),\n`
          code += `            nn.Dropout(config.hidden_dropout)\n`
          code += `        )\n\n`
          break
        case 'pooler-cls':
          code += `        # [CLS] Pooler\n`
          code += `        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)\n`
          code += `        self.activation = nn.Tanh()\n\n`
          break
        case 'pooler-mean':
          code += `        # Mean Pooler (no parameters)\n\n`
          break
        case 'classification-head':
          code += `        # Classification Head\n`
          code += `        self.classifier = nn.Linear(config.hidden_size, config.num_labels)\n\n`
          break
      }
    })

    code += `    def forward(self, input_ids, attention_mask=None):\n`
    
    // 生成 forward 逻辑
    let varName = 'input_ids'
    components.forEach((comp, idx) => {
      switch (comp.type) {
        case 'embedding':
          code += `        # Embedding\n`
          code += `        seq_length = input_ids.size(1)\n`
          code += `        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)\n`
          code += `        embeddings = self.embedding(input_ids) + self.position_embedding(position_ids)\n`
          varName = 'embeddings'
          break
        case 'attention-standard':
          code += `        # Standard Attention ${idx}\n`
          code += `        ${varName}, _ = self.attention_${idx}(${varName}, ${varName}, ${varName})\n`
          break
        case 'attention-local':
          code += `        # Local Attention ${idx}\n`
          code += `        ${varName} = self.local_attention_${idx}(${varName})\n`
          break
        case 'attention-strided':
          code += `        # Strided Attention ${idx}\n`
          code += `        ${varName} = self.strided_attention_${idx}(${varName})\n`
          break
        case 'ffn-gelu':
        case 'ffn-relu':
          code += `        # FFN ${idx}\n`
          code += `        ${varName} = self.ffn_${idx}(${varName})\n`
          break
        case 'pooler-cls':
          code += `        # [CLS] Pooling\n`
          code += `        pooled = self.activation(self.pooler(${varName}[:, 0]))\n`
          varName = 'pooled'
          break
        case 'pooler-mean':
          code += `        # Mean Pooling\n`
          code += `        pooled = ${varName}.mean(dim=1)\n`
          varName = 'pooled'
          break
        case 'classification-head':
          code += `        # Classification\n`
          code += `        logits = self.classifier(${varName})\n`
          varName = 'logits'
          break
      }
      code += `        \n`
    })

    code += `        return ${varName}\n`

    return code
  }

  // 下载代码
  const downloadCode = () => {
    const code = generateCode()
    const blob = new Blob([code], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'custom_model.py'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl border-2 border-blue-200">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
            <Layers className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="text-2xl font-bold text-gray-800">Model Builder Tool</h3>
            <p className="text-sm text-gray-600">Drag & drop to build your custom Transformer</p>
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={clearModel}
            disabled={components.length === 0}
            className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-all"
          >
            <Trash2 className="w-4 h-4" />
            Clear
          </button>
          <button
            onClick={downloadCode}
            disabled={components.length === 0}
            className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-all"
          >
            <Download className="w-4 h-4" />
            Export Code
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Component Templates (Left) */}
        <div className="lg:col-span-1 bg-white rounded-lg p-4 border-2 border-gray-200 max-h-[600px] overflow-y-auto">
          <h4 className="font-bold text-gray-700 mb-3 flex items-center gap-2">
            <Grid3x3 className="w-5 h-5" />
            Component Library
          </h4>
          <div className="space-y-2">
            {componentTemplates.map((template) => (
              <motion.button
                key={template.type}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => addComponent(template)}
                className={`w-full p-3 rounded-lg border-2 border-gray-200 hover:border-blue-400 transition-all text-left bg-gradient-to-r ${template.color} bg-opacity-10`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-2xl">{template.icon}</span>
                  <span className="font-semibold text-gray-800 text-sm">{template.name}</span>
                </div>
                <p className="text-xs text-gray-600 ml-9">{template.description}</p>
                <p className="text-xs text-gray-500 ml-9 mt-1">Params: {template.params.toFixed(1)}M</p>
              </motion.button>
            ))}
          </div>
        </div>

        {/* Model Canvas (Middle) */}
        <div className="lg:col-span-1 bg-white rounded-lg p-4 border-2 border-blue-300">
          <h4 className="font-bold text-gray-700 mb-3">Model Architecture</h4>
          
          {components.length === 0 ? (
            <div className="h-[500px] flex items-center justify-center border-2 border-dashed border-gray-300 rounded-lg">
              <div className="text-center text-gray-400">
                <Plus className="w-12 h-12 mx-auto mb-2" />
                <p>Click components to add</p>
              </div>
            </div>
          ) : (
            <div className="space-y-3 max-h-[500px] overflow-y-auto">
              <AnimatePresence>
                {components.map((component, idx) => (
                  <motion.div
                    key={component.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    className={`relative p-3 rounded-lg bg-gradient-to-r ${component.color} text-white`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-semibold text-sm flex items-center gap-2">
                          <span className="bg-white text-gray-700 rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold">
                            {idx + 1}
                          </span>
                          {component.name}
                        </div>
                        <p className="text-xs opacity-90 ml-8">
                          {component.params.toFixed(1)}M params
                        </p>
                      </div>
                      <button
                        onClick={() => removeComponent(component.id)}
                        className="p-1 hover:bg-white/20 rounded transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                    
                    {/* Connection line to next component */}
                    {idx < components.length - 1 && (
                      <div className="absolute left-1/2 -bottom-3 w-0.5 h-3 bg-gray-300 transform -translate-x-1/2" />
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          )}

          {/* Model Stats */}
          {components.length > 0 && (
            <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-gray-600">Total Layers:</span>
                  <span className="ml-2 font-bold text-blue-600">{components.length}</span>
                </div>
                <div>
                  <span className="text-gray-600">Total Params:</span>
                  <span className="ml-2 font-bold text-blue-600">{totalParams.toFixed(1)}M</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Generated Code (Right) */}
        <div className="lg:col-span-1 bg-gray-900 rounded-lg p-4 border-2 border-gray-700 max-h-[600px] overflow-hidden flex flex-col">
          <h4 className="font-bold text-gray-200 mb-3 flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-400" />
            Generated Code
          </h4>
          <pre className="flex-1 overflow-y-auto text-xs text-green-400 font-mono bg-gray-950 rounded p-3 border border-gray-700">
            {generateCode()}
          </pre>
        </div>
      </div>

      {/* Architecture Diagram */}
      {components.length > 0 && (
        <div className="mt-6 p-4 bg-white rounded-lg border-2 border-gray-200">
          <h4 className="font-bold text-gray-700 mb-3">Data Flow Diagram</h4>
          <div className="flex items-center gap-2 overflow-x-auto pb-2">
            {components.map((comp, idx) => (
              <React.Fragment key={comp.id}>
                <div className={`px-4 py-2 rounded-lg bg-gradient-to-r ${comp.color} text-white text-sm font-semibold whitespace-nowrap flex-shrink-0`}>
                  {comp.name}
                </div>
                {idx < components.length - 1 && (
                  <div className="text-gray-400 text-2xl flex-shrink-0">→</div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>
      )}

      {/* Usage Tips */}
      <div className="mt-4 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
        <h4 className="font-semibold text-yellow-800 mb-2">💡 Usage Tips</h4>
        <ul className="text-sm text-yellow-700 space-y-1">
          <li>• <strong>Embedding Layer</strong> should be the first component</li>
          <li>• <strong>Attention + FFN</strong> pairs form Transformer layers</li>
          <li>• <strong>Pooler</strong> converts sequence to single vector (for classification)</li>
          <li>• <strong>Classification Head</strong> should be the last component</li>
          <li>• Click &quot;Export Code&quot; to download complete PyTorch implementation</li>
        </ul>
      </div>
    </div>
  )
}
