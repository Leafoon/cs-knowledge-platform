'use client';

import React, { useState } from 'react';

type RAGMode = 'naive' | 'multi-query' | 'hyde' | 'parent-doc' | 'raptor';

export default function AdvancedRAGComparison() {
  const [selectedMode, setSelectedMode] = useState<RAGMode>('naive');

  const modes = {
    'naive': {
      name: 'Naive RAG',
      icon: '📝',
      color: 'gray',
      steps: [
        { label: '原始查询', detail: '"如何优化 RAG？"', time: '0ms' },
        { label: 'Embedding', detail: 'text-embedding-3-small', time: '50ms' },
        { label: '向量检索', detail: 'Top-5 相似文档', time: '20ms' },
        { label: 'LLM 生成', detail: 'GPT-4', time: '2000ms' },
      ],
      metrics: { recall: 0.65, precision: 0.58, latency: 2070, cost: 0.015 },
      pros: ['实现简单', '延迟较低', '成本可控'],
      cons: ['召回率低', '噪声文档多', '对模糊查询效果差']
    },
    'multi-query': {
      name: 'Multi-Query RAG',
      icon: '🔀',
      color: 'blue',
      steps: [
        { label: '生成查询变体', detail: '生成 4 个语义相似查询', time: '800ms' },
        { label: '并行 Embedding', detail: '4 个查询同时编码', time: '80ms' },
        { label: '并行检索', detail: '每个查询 Top-5', time: '80ms' },
        { label: '合并去重', detail: '融合结果，去重', time: '10ms' },
        { label: 'LLM 生成', detail: 'GPT-4', time: '2000ms' },
      ],
      metrics: { recall: 0.82, precision: 0.64, latency: 2970, cost: 0.028 },
      pros: ['召回率高', '覆盖多种表达', '鲁棒性强'],
      cons: ['延迟增加', '成本上升', '可能引入噪声']
    },
    'hyde': {
      name: 'HyDE',
      icon: '💭',
      color: 'purple',
      steps: [
        { label: '生成假设答案', detail: 'LLM 生成可能的答案文档', time: '1500ms' },
        { label: 'Embedding 假设', detail: '对假设答案编码', time: '50ms' },
        { label: '向量检索', detail: '用假设答案检索', time: '20ms' },
        { label: 'LLM 生成', detail: 'GPT-4 基于真实文档', time: '2000ms' },
      ],
      metrics: { recall: 0.78, precision: 0.72, latency: 3570, cost: 0.032 },
      pros: ['语义匹配更准', '适合专业领域', '减少查询-文档差异'],
      cons: ['延迟最高', '成本较高', '假设答案可能偏离']
    },
    'parent-doc': {
      name: 'Parent Document',
      icon: '📚',
      color: 'green',
      steps: [
        { label: '查询 Embedding', detail: 'text-embedding-3-small', time: '50ms' },
        { label: '检索子文档', detail: '小块匹配（400 字符）', time: '20ms' },
        { label: '返回父文档', detail: '检索完整上下文（2000 字符）', time: '5ms' },
        { label: 'LLM 生成', detail: 'GPT-4 with 完整上下文', time: '2200ms' },
      ],
      metrics: { recall: 0.75, precision: 0.68, latency: 2275, cost: 0.019 },
      pros: ['上下文完整', '检索精准', '适合长文档'],
      cons: ['需额外存储', '上下文可能冗余', '实现复杂']
    },
    'raptor': {
      name: 'RAPTOR',
      icon: '🌳',
      color: 'orange',
      steps: [
        { label: '查询 Embedding', detail: 'text-embedding-3-small', time: '50ms' },
        { label: '多层级检索', detail: '叶子层 + 摘要层', time: '40ms' },
        { label: '整合信息', detail: '合并不同粒度结果', time: '15ms' },
        { label: 'LLM 生成', detail: 'GPT-4 with 多粒度上下文', time: '2100ms' },
      ],
      metrics: { recall: 0.88, precision: 0.76, latency: 2205, cost: 0.024 },
      pros: ['多粒度检索', '高召回率', '适合复杂问题'],
      cons: ['构建成本高', '索引空间大', '维护复杂']
    }
  };

  const current = modes[selectedMode];

  const getColorClasses = (color: string) => {
    const colors: Record<string, { bg: string; border: string; text: string; badge: string }> = {
      gray: { bg: 'bg-gray-100 dark:bg-gray-900/30', border: 'border-gray-500', text: 'text-gray-700 dark:text-gray-300', badge: 'bg-gray-500' },
      blue: { bg: 'bg-blue-100 dark:bg-blue-900/30', border: 'border-blue-500', text: 'text-blue-700 dark:text-blue-300', badge: 'bg-blue-500' },
      purple: { bg: 'bg-purple-100 dark:bg-purple-900/30', border: 'border-purple-500', text: 'text-purple-700 dark:text-purple-300', badge: 'bg-purple-500' },
      green: { bg: 'bg-green-100 dark:bg-green-900/30', border: 'border-green-500', text: 'text-green-700 dark:text-green-300', badge: 'bg-green-500' },
      orange: { bg: 'bg-orange-100 dark:bg-orange-900/30', border: 'border-orange-500', text: 'text-orange-700 dark:text-orange-300', badge: 'bg-orange-500' }
    };
    return colors[color] || colors.gray;
  };

  const colors = getColorClasses(current.color);

  return (
    <div className="my-8 p-8 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700">
      <h3 className="text-2xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
        高级 RAG 架构对比
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        对比不同 RAG 架构模式的执行流程、性能指标与适用场景
      </p>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
        {Object.entries(modes).map(([key, mode]) => {
          const modeColors = getColorClasses(mode.color);
          return (
            <button
              key={key}
              onClick={() => setSelectedMode(key as RAGMode)}
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

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
          <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center gap-2">
            <span className="text-xl">🔄</span>
            执行流程
          </h4>
          <div className="space-y-3">
            {current.steps.map((step, idx) => (
              <div key={idx} className={`p-4 rounded-xl border-l-4 ${colors.border} ${colors.bg}`}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <div className={`w-6 h-6 ${colors.badge} text-white rounded-full flex items-center justify-center text-xs font-bold`}>
                      {idx + 1}
                    </div>
                    <span className="font-semibold text-gray-800 dark:text-gray-200">{step.label}</span>
                  </div>
                  <span className="text-xs text-gray-500 dark:text-gray-400 font-mono">{step.time}</span>
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400 ml-8">{step.detail}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
            <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <span className="text-xl">📊</span>
              性能指标
            </h4>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 rounded-xl border border-blue-200 dark:border-blue-700">
                <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">Recall</div>
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {(current.metrics.recall * 100).toFixed(0)}%
                </div>
              </div>
              <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/30 dark:to-green-800/30 rounded-xl border border-green-200 dark:border-green-700">
                <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">Precision</div>
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {(current.metrics.precision * 100).toFixed(0)}%
                </div>
              </div>
              <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/30 dark:to-purple-800/30 rounded-xl border border-purple-200 dark:border-purple-700">
                <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">总延迟</div>
                <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                  {current.metrics.latency}ms
                </div>
              </div>
              <div className="p-4 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/30 dark:to-orange-800/30 rounded-xl border border-orange-200 dark:border-orange-700">
                <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">成本</div>
                <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                  ${current.metrics.cost.toFixed(3)}
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
            <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200">优劣分析</h4>
            <div className="space-y-3">
              <div>
                <div className="text-sm font-semibold text-green-600 dark:text-green-400 mb-2 flex items-center gap-1">
                  <span>✅</span> 优势
                </div>
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
                <div className="text-sm font-semibold text-red-600 dark:text-red-400 mb-2 flex items-center gap-1">
                  <span>⚠️</span> 劣势
                </div>
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

      <div className="mt-6 p-6 bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-2xl border-l-4 border-yellow-500 shadow-lg">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0 w-10 h-10 bg-yellow-500 rounded-full flex items-center justify-center shadow-lg">
            <span className="text-white text-xl">💡</span>
          </div>
          <div>
            <h4 className="font-bold text-gray-800 dark:text-gray-100 mb-2">选择建议</h4>
            <div className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <div><strong>快速原型</strong>：Naive RAG（简单快速）</div>
              <div><strong>通用场景</strong>：Multi-Query RAG（召回优先）</div>
              <div><strong>专业领域</strong>：HyDE（语义匹配强）</div>
              <div><strong>长文档</strong>：Parent Document（上下文完整）</div>
              <div><strong>复杂推理</strong>：RAPTOR（多粒度信息）</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
