'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Sparkles, ArrowRight } from 'lucide-react'

interface ClassificationResult {
  label: string
  score: number
}

const SAMPLE_TEXTS = [
  {
    text: "这部电影太精彩了，我非常喜欢！",
    suggestedLabels: ["正面评价", "负面评价", "中性评价"]
  },
  {
    text: "Apple just released a new iPhone with advanced AI features.",
    suggestedLabels: ["technology", "sports", "politics", "entertainment"]
  },
  {
    text: "气候变化是当今世界面临的最大挑战之一。",
    suggestedLabels: ["环境", "经济", "娱乐", "体育", "科技"]
  }
]

export default function ZeroShotClassificationDemo() {
  const [text, setText] = useState(SAMPLE_TEXTS[0].text)
  const [labels, setLabels] = useState(SAMPLE_TEXTS[0].suggestedLabels.join(', '))
  const [results, setResults] = useState<ClassificationResult[] | null>(null)
  const [isClassifying, setIsClassifying] = useState(false)

  // 模拟分类（实际应该调用模型API）
  const handleClassify = () => {
    setIsClassifying(true)
    setResults(null)

    setTimeout(() => {
      const labelArray = labels.split(',').map(l => l.trim()).filter(l => l)
      
      // 模拟概率分布（实际由模型计算）
      const scores = labelArray.map(() => Math.random())
      const total = scores.reduce((a, b) => a + b, 0)
      const normalized = scores.map(s => s / total)

      const mockResults = labelArray
        .map((label, idx) => ({
          label,
          score: normalized[idx]
        }))
        .sort((a, b) => b.score - a.score)

      setResults(mockResults)
      setIsClassifying(false)
    }, 1500)
  }

  const loadSample = (index: number) => {
    setText(SAMPLE_TEXTS[index].text)
    setLabels(SAMPLE_TEXTS[index].suggestedLabels.join(', '))
    setResults(null)
  }

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border border-purple-200">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg">
          <Sparkles className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="text-2xl font-bold text-slate-800">零样本分类演示</h3>
          <p className="text-sm text-slate-600">无需训练，指定标签即可分类任意文本！</p>
        </div>
      </div>

      {/* 快速加载示例 */}
      <div className="mb-4 flex gap-2 flex-wrap">
        <span className="text-xs text-slate-600 py-2">快速加载：</span>
        {SAMPLE_TEXTS.map((sample, idx) => (
          <button
            key={idx}
            onClick={() => loadSample(idx)}
            className="px-3 py-1 text-xs bg-white border border-purple-200 rounded-lg hover:bg-purple-50 transition-colors"
          >
            示例 {idx + 1}
          </button>
        ))}
      </div>

      {/* 输入文本 */}
      <div className="mb-4">
        <label className="block text-sm font-bold text-slate-700 mb-2">
          📝 待分类文本
        </label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="w-full px-4 py-3 border-2 border-purple-200 rounded-lg focus:outline-none focus:border-purple-400 resize-none"
          rows={3}
          placeholder="输入任意文本..."
        />
      </div>

      {/* 候选标签 */}
      <div className="mb-6">
        <label className="block text-sm font-bold text-slate-700 mb-2">
          🏷️ 候选标签（逗号分隔）
        </label>
        <input
          type="text"
          value={labels}
          onChange={(e) => setLabels(e.target.value)}
          className="w-full px-4 py-3 border-2 border-purple-200 rounded-lg focus:outline-none focus:border-purple-400"
          placeholder="例如：正面, 负面, 中性"
        />
        <p className="text-xs text-slate-500 mt-1">
          💡 提示：可以使用任何你想要的标签，不需要预先训练！
        </p>
      </div>

      {/* 分类按钮 */}
      <motion.button
        onClick={handleClassify}
        disabled={isClassifying || !text.trim() || !labels.trim()}
        className="w-full py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-bold rounded-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        {isClassifying ? (
          <>
            <motion.div
              className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            />
            分类中...
          </>
        ) : (
          <>
            开始分类
            <ArrowRight className="w-5 h-5" />
          </>
        )}
      </motion.button>

      {/* 分类结果 */}
      <AnimatePresence>
        {results && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="mt-6 bg-white rounded-xl p-6 border-2 border-purple-200 shadow-lg"
          >
            <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
              <span className="text-lg">📊</span>
              分类结果
            </h4>

            <div className="space-y-3">
              {results.map((result, idx) => (
                <motion.div
                  key={result.label}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="relative"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-slate-700">{result.label}</span>
                    <span className="text-sm font-bold text-purple-600">
                      {(result.score * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  {/* 进度条 */}
                  <div className="h-8 bg-slate-100 rounded-lg overflow-hidden relative">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${result.score * 100}%` }}
                      transition={{ duration: 0.5, delay: idx * 0.1 }}
                      className={`h-full rounded-lg ${
                        idx === 0
                          ? 'bg-gradient-to-r from-purple-500 to-pink-500'
                          : idx === 1
                          ? 'bg-gradient-to-r from-purple-400 to-pink-400'
                          : 'bg-gradient-to-r from-purple-300 to-pink-300'
                      }`}
                    />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-xs font-bold text-slate-700 mix-blend-difference">
                        {idx === 0 && '🏆 最匹配'}
                      </span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* 工作原理说明 */}
            <div className="mt-6 bg-purple-50 border border-purple-200 rounded-lg p-4">
              <h5 className="font-bold text-purple-800 mb-2 text-sm">🔍 工作原理</h5>
              <p className="text-xs text-purple-700 leading-relaxed">
                Zero-Shot 分类使用预训练的 NLI（自然语言推理）模型，将分类任务转化为<strong>"文本蕴含"</strong>问题：
                检查"这段文本是关于 [标签] 的"这个假设的真实性。无需任何训练数据！
              </p>
            </div>

            {/* Pipeline 代码 */}
            <div className="mt-4 bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-xs text-slate-200">
                <code>{`from transformers import pipeline

classifier = pipeline("zero-shot-classification")

result = classifier(
    "${text.slice(0, 40)}...",
    candidate_labels=${JSON.stringify(results.map(r => r.label))}
)

# 输出：${results[0].label} (${(results[0].score * 100).toFixed(1)}%)`}</code>
              </pre>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
