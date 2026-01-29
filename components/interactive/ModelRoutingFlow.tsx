'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Zap, DollarSign, Clock, Brain, ChevronRight, CheckCircle2, XCircle } from 'lucide-react'

type TaskComplexity = 'simple' | 'medium' | 'complex'

interface Task {
  id: string
  query: string
  complexity: TaskComplexity
  selectedModel?: string
}

export default function ModelRoutingFlow() {
  const [currentTask, setCurrentTask] = useState<Task | null>(null)
  const [routingStep, setRoutingStep] = useState<'input' | 'analyzing' | 'routed' | 'result'>('input')
  const [inputQuery, setInputQuery] = useState('')
  const [history, setHistory] = useState<Task[]>([])

  const models = {
    'gpt-3.5-turbo': { cost: 0.002, latency: 500, power: 65 },
    'gpt-4': { cost: 0.03, latency: 1500, power: 95 },
    'claude-3-sonnet': { cost: 0.015, latency: 800, power: 85 }
  }

  const complexityRules = {
    simple: {
      keywords: ['什么是', '定义', '简单', '是谁', '在哪'],
      model: 'gpt-3.5-turbo',
      description: '简单问答、定义查询'
    },
    medium: {
      keywords: ['分析', '比较', '解释', '总结', '原理'],
      model: 'claude-3-sonnet',
      description: '中等复杂度分析任务'
    },
    complex: {
      keywords: ['设计', '架构', '实现', '优化', '深入'],
      model: 'gpt-4',
      description: '复杂推理、架构设计'
    }
  }

  const exampleQueries = [
    { text: '什么是 LangChain？', complexity: 'simple' as TaskComplexity },
    { text: '比较 ReAct 和 Plan-Execute 架构', complexity: 'medium' as TaskComplexity },
    { text: '设计一个支持百万用户的 RAG 系统架构', complexity: 'complex' as TaskComplexity }
  ]

  const classifyComplexity = (query: string): TaskComplexity => {
    const lowerQuery = query.toLowerCase()
    if (complexityRules.complex.keywords.some(kw => lowerQuery.includes(kw))) return 'complex'
    if (complexityRules.medium.keywords.some(kw => lowerQuery.includes(kw))) return 'medium'
    return 'simple'
  }

  const handleSubmit = () => {
    if (!inputQuery.trim()) return
    
    const complexity = classifyComplexity(inputQuery)
    const task: Task = {
      id: Date.now().toString(),
      query: inputQuery,
      complexity
    }
    
    setCurrentTask(task)
    setRoutingStep('analyzing')
    
    setTimeout(() => {
      const selectedModel = complexityRules[complexity].model
      setCurrentTask({ ...task, selectedModel })
      setRoutingStep('routed')
      
      setTimeout(() => {
        setRoutingStep('result')
        setHistory(prev => [{ ...task, selectedModel }, ...prev].slice(0, 5))
        
        setTimeout(() => {
          setRoutingStep('input')
          setCurrentTask(null)
          setInputQuery('')
        }, 3000)
      }, 1500)
    }, 1200)
  }

  const calculateSavings = () => {
    if (history.length === 0) return { saved: 0, percentage: 0 }
    
    const actualCost = history.reduce((sum, task) => {
      const model = task.selectedModel as keyof typeof models
      return sum + (models[model]?.cost || 0)
    }, 0)
    
    const allGPT4Cost = history.length * models['gpt-4'].cost
    const saved = allGPT4Cost - actualCost
    const percentage = ((saved / allGPT4Cost) * 100).toFixed(1)
    
    return { saved: saved.toFixed(3), percentage }
  }

  const savings = calculateSavings()

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl border border-purple-200 dark:border-purple-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-3 bg-purple-500 rounded-lg">
          <Brain className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
            智能模型路由演示
          </h3>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            根据任务复杂度自动选择最优模型，节省成本 78%
          </p>
        </div>
      </div>

      {/* 输入区域 */}
      <div className="mb-6">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
            placeholder="输入任务查询..."
            disabled={routingStep !== 'input'}
            className="flex-1 px-4 py-3 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 focus:ring-2 focus:ring-purple-500 outline-none disabled:opacity-50"
          />
          <button
            onClick={handleSubmit}
            disabled={routingStep !== 'input' || !inputQuery.trim()}
            className="px-6 py-3 bg-purple-500 text-white rounded-lg font-medium hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            路由
          </button>
        </div>
        
        <div className="flex gap-2 mt-3">
          {exampleQueries.map((example, idx) => (
            <button
              key={idx}
              onClick={() => setInputQuery(example.text)}
              disabled={routingStep !== 'input'}
              className="px-3 py-1 text-xs bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-full hover:bg-purple-50 dark:hover:bg-purple-900/30 transition-all disabled:opacity-50"
            >
              {example.text}
            </button>
          ))}
        </div>
      </div>

      {/* 路由流程可视化 */}
      <AnimatePresence mode="wait">
        {currentTask && (
          <motion.div
            key={routingStep}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="mb-6 p-6 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
          >
            {routingStep === 'analyzing' && (
              <div className="flex items-center gap-4">
                <div className="animate-spin rounded-full h-12 w-12 border-4 border-purple-200 border-t-purple-500" />
                <div>
                  <div className="font-bold text-slate-800 dark:text-slate-200 mb-1">
                    分析任务复杂度...
                  </div>
                  <div className="text-sm text-slate-600 dark:text-slate-400">
                    检测关键词、评估推理深度
                  </div>
                </div>
              </div>
            )}

            {routingStep === 'routed' && (
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <CheckCircle2 className="w-8 h-8 text-green-500" />
                  <div>
                    <div className="font-bold text-slate-800 dark:text-slate-200">
                      复杂度：{currentTask.complexity.toUpperCase()}
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-400">
                      {complexityRules[currentTask.complexity].description}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center gap-2 p-4 bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/30 dark:to-blue-900/30 rounded-lg">
                  <ChevronRight className="w-5 h-5 text-purple-500" />
                  <span className="font-bold text-lg text-slate-800 dark:text-slate-200">
                    选择模型：{currentTask.selectedModel}
                  </span>
                </div>
              </div>
            )}

            {routingStep === 'result' && currentTask.selectedModel && (
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <Zap className="w-8 h-8 text-yellow-500" />
                  <div className="font-bold text-lg text-slate-800 dark:text-slate-200">
                    路由完成！
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-4">
                  <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <DollarSign className="w-5 h-5 text-blue-500 mb-1" />
                    <div className="text-xs text-slate-600 dark:text-slate-400">成本</div>
                    <div className="font-bold text-slate-800 dark:text-slate-200">
                      ${models[currentTask.selectedModel as keyof typeof models].cost}
                    </div>
                  </div>
                  <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <Clock className="w-5 h-5 text-green-500 mb-1" />
                    <div className="text-xs text-slate-600 dark:text-slate-400">延迟</div>
                    <div className="font-bold text-slate-800 dark:text-slate-200">
                      {models[currentTask.selectedModel as keyof typeof models].latency}ms
                    </div>
                  </div>
                  <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                    <Brain className="w-5 h-5 text-purple-500 mb-1" />
                    <div className="text-xs text-slate-600 dark:text-slate-400">能力</div>
                    <div className="font-bold text-slate-800 dark:text-slate-200">
                      {models[currentTask.selectedModel as keyof typeof models].power}%
                    </div>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* 成本节省统计 */}
      {history.length > 0 && (
        <div className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg border border-green-200 dark:border-green-700">
          <div className="flex items-center justify-between mb-3">
            <div className="font-bold text-slate-800 dark:text-slate-200">
              累计成本节省
            </div>
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              {savings.percentage}%
            </div>
          </div>
          <div className="text-sm text-slate-600 dark:text-slate-400">
            已处理 {history.length} 个任务，节省 ${savings.saved}（vs 全部使用 GPT-4）
          </div>
        </div>
      )}

      {/* 历史记录 */}
      {history.length > 0 && (
        <div className="mt-6">
          <div className="font-bold text-slate-800 dark:text-slate-200 mb-3">
            最近路由历史
          </div>
          <div className="space-y-2">
            {history.map((task, idx) => (
              <div
                key={task.id}
                className="flex items-center justify-between p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
              >
                <div className="flex-1">
                  <div className="text-sm font-medium text-slate-800 dark:text-slate-200">
                    {task.query}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    复杂度：{task.complexity} → {task.selectedModel}
                  </div>
                </div>
                <div className="text-xs font-medium text-green-600 dark:text-green-400">
                  ${models[task.selectedModel as keyof typeof models].cost}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
