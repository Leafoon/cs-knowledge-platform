'use client';

import React, { useState } from 'react';

type SearchMode = 'vector-only' | 'bm25-only' | 'hybrid' | 'hybrid-rerank';

export default function HybridSearchArchitecture() {
  const [selectedMode, setSelectedMode] = useState<SearchMode>('hybrid');
  const [query] = useState('PyTorch Lightning 教程');

  const searchResults = {
    'vector-only': [
      { id: 1, title: 'PyTorch 深度学习框架', score: 0.89, method: 'vector', relevant: true },
      { id: 2, title: 'TensorFlow Lightning 指南', score: 0.82, method: 'vector', relevant: false },
      { id: 3, title: '深度学习框架对比', score: 0.78, method: 'vector', relevant: true },
      { id: 4, title: 'Lightning AI 平台介绍', score: 0.75, method: 'vector', relevant: false },
      { id: 5, title: 'PyTorch 进阶教程', score: 0.72, method: 'vector', relevant: true }
    ],
    'bm25-only': [
      { id: 6, title: 'PyTorch Lightning 官方文档', score: 15.2, method: 'bm25', relevant: true },
      { id: 7, title: 'PyTorch Lightning 快速开始', score: 14.8, method: 'bm25', relevant: true },
      { id: 8, title: 'Lightning 模块化训练', score: 12.5, method: 'bm25', relevant: true },
      { id: 9, title: 'PyTorch 基础教程（无 Lightning）', score: 8.3, method: 'bm25', relevant: false },
      { id: 10, title: 'Lightning Network 白皮书', score: 7.1, method: 'bm25', relevant: false }
    ],
    'hybrid': [
      { id: 6, title: 'PyTorch Lightning 官方文档', score: 0.92, method: 'hybrid', relevant: true },
      { id: 7, title: 'PyTorch Lightning 快速开始', score: 0.88, method: 'hybrid', relevant: true },
      { id: 1, title: 'PyTorch 深度学习框架', score: 0.84, method: 'hybrid', relevant: true },
      { id: 8, title: 'Lightning 模块化训练', score: 0.81, method: 'hybrid', relevant: true },
      { id: 3, title: '深度学习框架对比', score: 0.75, method: 'hybrid', relevant: true }
    ],
    'hybrid-rerank': [
      { id: 6, title: 'PyTorch Lightning 官方文档', score: 0.96, method: 'rerank', relevant: true },
      { id: 7, title: 'PyTorch Lightning 快速开始', score: 0.94, method: 'rerank', relevant: true },
      { id: 8, title: 'Lightning 模块化训练', score: 0.89, method: 'rerank', relevant: true },
      { id: 1, title: 'PyTorch 深度学习框架', score: 0.82, method: 'rerank', relevant: true },
      { id: 3, title: '深度学习框架对比', score: 0.76, method: 'rerank', relevant: true }
    ]
  };

  const modes = {
    'vector-only': {
      name: '纯向量检索',
      icon: '🔢',
      color: 'blue',
      description: '基于 Embedding 相似度检索',
      precision: 0.60,
      recall: 0.75,
      f1: 0.67,
      pros: ['语义理解强', '泛化能力好'],
      cons: ['专有名词召回弱', '精确匹配差']
    },
    'bm25-only': {
      name: '纯 BM25',
      icon: '🔤',
      color: 'green',
      description: '基于词频的稀疏检索',
      precision: 0.70,
      recall: 0.65,
      f1: 0.67,
      pros: ['精确匹配强', '专有名词好'],
      cons: ['无语义理解', '同义词召回差']
    },
    'hybrid': {
      name: '混合检索',
      icon: '🔀',
      color: 'purple',
      description: 'Vector + BM25 加权融合',
      precision: 0.82,
      recall: 0.88,
      f1: 0.85,
      pros: ['结合双方优势', '鲁棒性强'],
      cons: ['需调参权重', '计算开销大']
    },
    'hybrid-rerank': {
      name: '混合+重排',
      icon: '⭐',
      color: 'orange',
      description: 'Hybrid + Cross-Encoder Rerank',
      precision: 0.92,
      recall: 0.90,
      f1: 0.91,
      pros: ['精度最高', '排序最优'],
      cons: ['延迟最高', '成本最高']
    }
  };

  const current = modes[selectedMode];
  const results = searchResults[selectedMode];

  const getColorClasses = (color: string) => {
    const colors: Record<string, { bg: string; border: string; text: string; badge: string }> = {
      blue: { bg: 'bg-blue-100 dark:bg-blue-900/30', border: 'border-blue-500', text: 'text-blue-700 dark:text-blue-300', badge: 'bg-blue-500' },
      green: { bg: 'bg-green-100 dark:bg-green-900/30', border: 'border-green-500', text: 'text-green-700 dark:text-green-300', badge: 'bg-green-500' },
      purple: { bg: 'bg-purple-100 dark:bg-purple-900/30', border: 'border-purple-500', text: 'text-purple-700 dark:text-purple-300', badge: 'bg-purple-500' },
      orange: { bg: 'bg-orange-100 dark:bg-orange-900/30', border: 'border-orange-500', text: 'text-orange-700 dark:text-orange-300', badge: 'bg-orange-500' }
    };
    return colors[color] || colors.blue;
  };

  const colors = getColorClasses(current.color);

  const getMethodBadge = (method: string) => {
    const badges: Record<string, { bg: string; text: string }> = {
      vector: { bg: 'bg-blue-500', text: 'Vector' },
      bm25: { bg: 'bg-green-500', text: 'BM25' },
      hybrid: { bg: 'bg-purple-500', text: 'Hybrid' },
      rerank: { bg: 'bg-orange-500', text: 'Rerank' }
    };
    const badge = badges[method] || badges.vector;
    return (
      <span className={`px-2 py-0.5 ${badge.bg} text-white rounded text-xs font-bold`}>
        {badge.text}
      </span>
    );
  };

  return (
    <div className="my-8 p-8 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700">
      <h3 className="text-2xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
        混合检索架构对比
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        对比向量检索、BM25、混合检索与重排序的效果差异
      </p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {Object.entries(modes).map(([key, mode]) => {
          const modeColors = getColorClasses(mode.color);
          return (
            <button
              key={key}
              onClick={() => setSelectedMode(key as SearchMode)}
              className={`p-4 rounded-xl transition-all border-2 ${
                selectedMode === key
                  ? `${modeColors.border} ${modeColors.bg} shadow-lg scale-105`
                  : 'border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 hover:shadow-md'
              }`}
            >
              <div className="text-3xl mb-2">{mode.icon}</div>
              <div className={`text-sm font-semibold ${selectedMode === key ? modeColors.text : 'text-gray-700 dark:text-gray-300'}`}>
                {mode.name}
              </div>
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <h4 className="font-bold text-gray-800 dark:text-gray-200 flex items-center gap-2">
                <span className="text-xl">🔍</span>
                检索结果
              </h4>
              <div className="px-3 py-1 bg-gray-100 dark:bg-gray-700 rounded-lg text-sm font-mono text-gray-700 dark:text-gray-300">
                &quot;{query}&quot;
              </div>
            </div>
            <div className="space-y-2">
              {results.map((result, idx) => (
                <div
                  key={result.id}
                  className={`p-4 rounded-xl border-2 transition-all ${
                    result.relevant
                      ? 'border-green-300 dark:border-green-700 bg-green-50 dark:bg-green-900/20'
                      : 'border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-900/20'
                  }`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <div className={`w-7 h-7 ${result.relevant ? 'bg-green-500' : 'bg-red-500'} text-white rounded-full flex items-center justify-center text-sm font-bold`}>
                        {idx + 1}
                      </div>
                      <div>
                        <div className="font-semibold text-gray-800 dark:text-gray-200 mb-1">
                          {result.title}
                        </div>
                        <div className="flex items-center gap-2">
                          {getMethodBadge(result.method)}
                          <span className={`text-xs ${result.relevant ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                            {result.relevant ? '✅ 相关' : '❌ 不相关'}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-mono font-bold text-gray-700 dark:text-gray-300">
                        {result.score.toFixed(2)}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">分数</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
            <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <span className="text-xl">📊</span>
              性能指标
            </h4>
            <div className="space-y-3">
              <div className="p-3 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 rounded-lg border border-blue-200 dark:border-blue-700">
                <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">Precision</div>
                <div className="text-xl font-bold text-blue-600 dark:text-blue-400">
                  {(current.precision * 100).toFixed(0)}%
                </div>
                <div className="mt-2 bg-blue-200 dark:bg-blue-800 rounded-full h-2">
                  <div
                    className="bg-blue-600 dark:bg-blue-400 h-2 rounded-full transition-all"
                    style={{ width: `${current.precision * 100}%` }}
                  ></div>
                </div>
              </div>
              <div className="p-3 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/30 dark:to-green-800/30 rounded-lg border border-green-200 dark:border-green-700">
                <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">Recall</div>
                <div className="text-xl font-bold text-green-600 dark:text-green-400">
                  {(current.recall * 100).toFixed(0)}%
                </div>
                <div className="mt-2 bg-green-200 dark:bg-green-800 rounded-full h-2">
                  <div
                    className="bg-green-600 dark:bg-green-400 h-2 rounded-full transition-all"
                    style={{ width: `${current.recall * 100}%` }}
                  ></div>
                </div>
              </div>
              <div className="p-3 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/30 dark:to-purple-800/30 rounded-lg border border-purple-200 dark:border-purple-700">
                <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">F1 Score</div>
                <div className="text-xl font-bold text-purple-600 dark:text-purple-400">
                  {(current.f1 * 100).toFixed(0)}%
                </div>
                <div className="mt-2 bg-purple-200 dark:bg-purple-800 rounded-full h-2">
                  <div
                    className="bg-purple-600 dark:bg-purple-400 h-2 rounded-full transition-all"
                    style={{ width: `${current.f1 * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
            <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200">优劣分析</h4>
            <div className="space-y-3">
              <div>
                <div className="text-sm font-semibold text-green-600 dark:text-green-400 mb-2">✅ 优势</div>
                <ul className="space-y-1">
                  {current.pros.map((pro, idx) => (
                    <li key={idx} className="text-sm text-gray-700 dark:text-gray-300 flex items-start gap-2">
                      <span className="text-green-500 mt-0.5">•</span>
                      {pro}
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <div className="text-sm font-semibold text-red-600 dark:text-red-400 mb-2">⚠️ 劣势</div>
                <ul className="space-y-1">
                  {current.cons.map((con, idx) => (
                    <li key={idx} className="text-sm text-gray-700 dark:text-gray-300 flex items-start gap-2">
                      <span className="text-red-500 mt-0.5">•</span>
                      {con}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 p-6 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-2xl border-l-4 border-blue-500 shadow-lg">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0 w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center shadow-lg">
            <span className="text-white text-xl">💡</span>
          </div>
          <div>
            <h4 className="font-bold text-gray-800 dark:text-gray-100 mb-2">权重配置建议</h4>
            <div className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <div><strong>通用场景</strong>：Vector 50% + BM25 50%</div>
              <div><strong>语义优先</strong>：Vector 70% + BM25 30%</div>
              <div><strong>精确匹配优先</strong>：Vector 30% + BM25 70%</div>
              <div><strong>生产推荐</strong>：Hybrid (60/40) + Cohere Rerank (Top 5)</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
