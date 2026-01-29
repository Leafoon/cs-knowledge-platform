'use client';

import React, { useState } from 'react';

type TransformationType = 'rewrite' | 'decompose' | 'step-back' | 'multi-query' | 'hyde';

export default function QueryTransformationFlow() {
  const [selectedType, setSelectedType] = useState<TransformationType>('rewrite');

  const transformations = {
    'rewrite': {
      name: 'Query Rewriting',
      icon: '✍️',
      color: 'blue',
      description: '优化查询表达，提高检索精度',
      original: '这个怎么用？',
      transformed: ['如何使用这个软件功能？', '该功能的操作步骤是什么？', '能否提供使用教程？'],
      prompt: 'Given the following user question, rewrite it to be more specific and suitable for vector search:\n\nOriginal: {query}\n\nRewritten:',
      useCase: '模糊查询、口语化问题',
      improvement: '+25% 检索精度'
    },
    'decompose': {
      name: 'Query Decomposition',
      icon: '🔨',
      color: 'purple',
      description: '将复杂问题分解为多个子问题',
      original: '比较 PyTorch 和 TensorFlow 在分布式训练、部署和生态的优劣',
      transformed: [
        'PyTorch 和 TensorFlow 的分布式训练能力对比？',
        '两者在模型部署方面的差异？',
        '社区生态和工具链的成熟度对比？'
      ],
      prompt: 'Break down the following complex question into 3 simpler sub-questions:\n\nQuestion: {query}\n\nSub-questions:',
      useCase: '复杂多维度问题',
      improvement: '+35% 答案完整性'
    },
    'step-back': {
      name: 'Step-Back Prompting',
      icon: '🔙',
      color: 'green',
      description: '生成更抽象的高层次问题',
      original: '2023年诺贝尔物理学奖获得者阿秒激光的原理是什么？',
      transformed: [
        '阿秒激光的基本原理是什么？',
        '超快激光技术的发展历程',
        '阿秒级脉冲如何产生？'
      ],
      prompt: 'You are an expert. Step back and paraphrase the question to a more generic step-back question:\n\nOriginal: {query}\n\nStep-back:',
      useCase: '需要背景知识的具体问题',
      improvement: '+30% 上下文丰富度'
    },
    'multi-query': {
      name: 'Multi-Query Generation',
      icon: '🔀',
      color: 'orange',
      description: '生成多个语义相似的查询变体',
      original: '如何优化 RAG 系统性能？',
      transformed: [
        '如何优化 RAG 系统性能？',
        '提升检索增强生成系统效率的方法有哪些？',
        'RAG 性能调优的最佳实践是什么？',
        '怎样改进 RAG 的检索质量和速度？'
      ],
      prompt: 'Generate 3 different versions of the question to retrieve relevant documents:\n\nOriginal: {query}\n\nVersions:',
      useCase: '提高召回率',
      improvement: '+40% Recall'
    },
    'hyde': {
      name: 'HyDE (假设文档)',
      icon: '💭',
      color: 'cyan',
      description: '生成假设答案文档进行检索',
      original: 'PyTorch 的动态图和静态图有什么区别？',
      transformed: [
        'PyTorch 采用动态计算图（Define-by-Run），在运行时构建图，灵活性高，便于调试。而静态图（Define-and-Run）需要先定义完整计算图再执行，TensorFlow 1.x 采用此方式。PyTorch 2.0 引入了 torch.compile() 支持静态图优化...'
      ],
      prompt: 'Please write a passage to answer the question:\n\nQuestion: {query}\n\nPassage:',
      useCase: '专业领域、语义gap大',
      improvement: '+28% 检索相关性'
    }
  };

  const current = transformations[selectedType];

  const getColorClasses = (color: string) => {
    const colors: Record<string, { bg: string; border: string; text: string; badge: string }> = {
      blue: { bg: 'bg-blue-100 dark:bg-blue-900/30', border: 'border-blue-500', text: 'text-blue-700 dark:text-blue-300', badge: 'bg-blue-500' },
      purple: { bg: 'bg-purple-100 dark:bg-purple-900/30', border: 'border-purple-500', text: 'text-purple-700 dark:text-purple-300', badge: 'bg-purple-500' },
      green: { bg: 'bg-green-100 dark:bg-green-900/30', border: 'border-green-500', text: 'text-green-700 dark:text-green-300', badge: 'bg-green-500' },
      orange: { bg: 'bg-orange-100 dark:bg-orange-900/30', border: 'border-orange-500', text: 'text-orange-700 dark:text-orange-300', badge: 'bg-orange-500' },
      cyan: { bg: 'bg-cyan-100 dark:bg-cyan-900/30', border: 'border-cyan-500', text: 'text-cyan-700 dark:text-cyan-300', badge: 'bg-cyan-500' }
    };
    return colors[color] || colors.blue;
  };

  const colors = getColorClasses(current.color);

  return (
    <div className="my-8 p-8 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700">
      <h3 className="text-2xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
        查询转换策略对比
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        探索不同查询优化方法的工作原理与适用场景
      </p>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
        {Object.entries(transformations).map(([key, trans]) => {
          const transColors = getColorClasses(trans.color);
          return (
            <button
              key={key}
              onClick={() => setSelectedType(key as TransformationType)}
              className={`p-4 rounded-xl transition-all border-2 ${
                selectedType === key
                  ? `${transColors.border} ${transColors.bg} shadow-lg scale-105`
                  : 'border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 hover:shadow-md'
              }`}
            >
              <div className="text-3xl mb-2">{trans.icon}</div>
              <div className={`text-xs font-semibold ${selectedType === key ? transColors.text : 'text-gray-700 dark:text-gray-300'} text-center leading-tight`}>
                {trans.name}
              </div>
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
            <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <span className="text-xl">📥</span>
              原始查询
            </h4>
            <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-xl border-2 border-gray-300 dark:border-gray-600">
              <div className="text-sm text-gray-500 dark:text-gray-400 mb-2">用户输入</div>
              <div className="text-base text-gray-800 dark:text-gray-200 font-medium">
                "{current.original}"
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
            <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <span className="text-xl">📤</span>
              转换结果
            </h4>
            <div className="space-y-3">
              {current.transformed.map((query, idx) => (
                <div key={idx} className={`p-4 rounded-xl border-l-4 ${colors.border} ${colors.bg}`}>
                  <div className="flex items-start gap-3">
                    <div className={`flex-shrink-0 w-6 h-6 ${colors.badge} text-white rounded-full flex items-center justify-center text-xs font-bold mt-0.5`}>
                      {idx + 1}
                    </div>
                    <div className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                      {query}
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
              <span className="text-xl">⚙️</span>
              工作原理
            </h4>
            <div className="space-y-4">
              <div>
                <div className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2">描述</div>
                <div className="text-sm text-gray-700 dark:text-gray-300">
                  {current.description}
                </div>
              </div>
              <div>
                <div className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2">Prompt 模板</div>
                <div className="p-3 bg-gray-900 rounded-lg font-mono text-xs text-green-400 overflow-x-auto whitespace-pre-wrap">
                  {current.prompt}
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
            <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <span className="text-xl">📊</span>
              应用场景
            </h4>
            <div className="space-y-3">
              <div className={`p-4 rounded-xl ${colors.bg} border-2 ${colors.border}`}>
                <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">适用场景</div>
                <div className={`text-base font-semibold ${colors.text}`}>
                  {current.useCase}
                </div>
              </div>
              <div className="p-4 rounded-xl bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/30 dark:to-emerald-900/30 border-2 border-green-500">
                <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">性能提升</div>
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {current.improvement}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="p-5 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-xl shadow-md border border-blue-200 dark:border-blue-700">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-white text-xl">🎯</span>
            </div>
            <div className="font-bold text-gray-800 dark:text-gray-200">精度优先</div>
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Query Rewriting, HyDE
          </div>
        </div>
        <div className="p-5 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-xl shadow-md border border-purple-200 dark:border-purple-700">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-purple-500 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-white text-xl">📈</span>
            </div>
            <div className="font-bold text-gray-800 dark:text-gray-200">召回优先</div>
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Multi-Query, Decompose
          </div>
        </div>
        <div className="p-5 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-xl shadow-md border border-green-200 dark:border-green-700">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-white text-xl">🧠</span>
            </div>
            <div className="font-bold text-gray-800 dark:text-gray-200">理解优先</div>
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Step-Back, Decompose
          </div>
        </div>
      </div>
    </div>
  );
}
