'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { HardDrive, Trash2, Search, FolderOpen, Database } from 'lucide-react'

interface CachedModel {
  id: string
  name: string
  size: number
  lastUsed: string
  downloads: number
}

export default function CacheManagementVisualizer() {
  const [models, setModels] = useState<CachedModel[]>([
    { id: '1', name: 'bert-base-uncased', size: 440, lastUsed: '2024-01-20', downloads: 1 },
    { id: '2', name: 'gpt2', size: 548, lastUsed: '2024-01-18', downloads: 3 },
    { id: '3', name: 't5-small', size: 242, lastUsed: '2024-01-15', downloads: 2 },
    { id: '4', name: 'roberta-base', size: 498, lastUsed: '2024-01-10', downloads: 1 }
  ])
  const [action, setAction] = useState<'scan' | 'delete' | null>(null)
  const [selectedModel, setSelectedModel] = useState<string | null>(null)

  const totalSize = models.reduce((sum, m) => sum + m.size, 0)
  const cacheDir = '~/.cache/huggingface/hub'

  const handleScan = () => {
    setAction('scan')
    setTimeout(() => setAction(null), 2000)
  }

  const handleDelete = (modelId: string) => {
    setAction('delete')
    setSelectedModel(modelId)
    setTimeout(() => {
      setModels(models.filter(m => m.id !== modelId))
      setAction(null)
      setSelectedModel(null)
    }, 1000)
  }

  const handleDeleteAll = () => {
    setAction('delete')
    setTimeout(() => {
      setModels([])
      setAction(null)
    }, 1000)
  }

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-slate-900 dark:to-purple-950 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Database className="w-5 h-5 text-purple-500" />
          缓存管理可视化
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          查看和管理 Hugging Face 模型缓存
        </p>
      </div>

      {/* Cache Directory */}
      <div className="mb-6 p-4 bg-slate-900 rounded-lg">
        <div className="flex items-center gap-2 mb-2">
          <FolderOpen className="w-4 h-4 text-yellow-400" />
          <span className="text-xs text-slate-400">缓存目录</span>
        </div>
        <code className="text-sm text-green-400 font-mono">{cacheDir}</code>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={handleScan}
          disabled={action === 'scan'}
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-slate-400 text-white rounded-lg transition-colors"
        >
          <Search className="w-4 h-4" />
          {action === 'scan' ? '扫描中...' : '扫描缓存'}
        </button>
        <button
          onClick={handleDeleteAll}
          disabled={action === 'delete' || models.length === 0}
          className="flex items-center gap-2 px-4 py-2 bg-red-500 hover:bg-red-600 disabled:bg-slate-400 text-white rounded-lg transition-colors"
        >
          <Trash2 className="w-4 h-4" />
          清空缓存
        </button>
      </div>

      {/* Models List */}
      <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden mb-6">
        <div className="p-4 bg-slate-100 dark:bg-slate-700 border-b border-slate-200 dark:border-slate-600">
          <div className="grid grid-cols-4 gap-4 text-xs font-semibold text-slate-600 dark:text-slate-400">
            <div>模型名称</div>
            <div>大小</div>
            <div>最后使用</div>
            <div>操作</div>
          </div>
        </div>

        <AnimatePresence>
          {models.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="p-8 text-center text-slate-400"
            >
              <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>缓存为空</p>
            </motion.div>
          ) : (
            models.map((model) => (
              <motion.div
                key={model.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className={`p-4 border-b border-slate-200 dark:border-slate-700 last:border-0 ${
                  action === 'delete' && selectedModel === model.id
                    ? 'bg-red-50 dark:bg-red-900/20'
                    : 'hover:bg-slate-50 dark:hover:bg-slate-700/50'
                }`}
              >
                <div className="grid grid-cols-4 gap-4 items-center">
                  <div className="font-mono text-sm text-slate-900 dark:text-white">
                    {model.name}
                  </div>
                  <div className="text-sm text-slate-600 dark:text-slate-400">
                    {model.size} MB
                  </div>
                  <div className="text-sm text-slate-600 dark:text-slate-400">
                    {model.lastUsed}
                  </div>
                  <div>
                    <button
                      onClick={() => handleDelete(model.id)}
                      disabled={action === 'delete'}
                      className="px-3 py-1 bg-red-100 hover:bg-red-200 dark:bg-red-900/30 dark:hover:bg-red-900/50 text-red-600 dark:text-red-400 rounded text-sm transition-colors disabled:opacity-50"
                    >
                      删除
                    </button>
                  </div>
                </div>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-3 gap-4">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <HardDrive className="w-4 h-4 text-blue-500" />
            <span className="text-xs text-slate-600 dark:text-slate-400">总大小</span>
          </div>
          <div className="text-2xl font-bold text-blue-500">{totalSize} MB</div>
          <div className="text-xs text-slate-500 mt-1">{(totalSize / 1024).toFixed(2)} GB</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <Database className="w-4 h-4 text-green-500" />
            <span className="text-xs text-slate-600 dark:text-slate-400">模型数量</span>
          </div>
          <div className="text-2xl font-bold text-green-500">{models.length}</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <Trash2 className="w-4 h-4 text-purple-500" />
            <span className="text-xs text-slate-600 dark:text-slate-400">可释放</span>
          </div>
          <div className="text-2xl font-bold text-purple-500">{totalSize} MB</div>
        </div>
      </div>

      {/* Commands Reference */}
      <div className="mt-6 p-4 bg-slate-900 rounded-lg">
        <div className="text-xs text-slate-400 mb-3 font-semibold">常用命令</div>
        <div className="space-y-2 font-mono text-sm">
          <div className="text-green-400">
            # 查看缓存占用
            <br />
            <span className="text-slate-300">huggingface-cli scan-cache</span>
          </div>
          <div className="text-green-400">
            # 交互式删除
            <br />
            <span className="text-slate-300">huggingface-cli delete-cache</span>
          </div>
          <div className="text-green-400">
            # 手动删除（谨慎！）
            <br />
            <span className="text-slate-300">rm -rf {cacheDir}/models--bert-base-uncased</span>
          </div>
        </div>
      </div>
    </div>
  )
}
