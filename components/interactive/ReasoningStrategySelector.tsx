'use client';

import React, { useState } from 'react';

type Strategy = 'cot' | 'react' | 'plansearch' | 'reflection' | 'selfconsistency';

export default function ReasoningStrategySelector() {
  const [selected, setSelected] = useState<Strategy>('cot');

  const strategies = {
    cot: {
      name: 'Chain-of-Thought',
      description: '引导大语言模型一步步生成推理过程',
      pros: ['简单易用', '提高复杂问题准确性', '可解释性强'],
      cons: ['可能产生错误推理', 'token消耗较大', '依赖模型能力'],
      when: '数学问题、逻辑推理、多步骤计算',
      color: '#3b82f6'
    },
    react: {
      name: 'ReAct',
      description: 'Reasoning + Acting 循环，结合工具调用',
      pros: ['结合外部工具', '可交互验证', '纠正错误'],
      cons: ['搜索次数可能过多', '推理链容易断裂', '实现复杂度高'],
      when: '需要外部信息、工具使用、交互任务',
      color: '#10b981'
    },
    plansearch: {
      name: 'Plan-and-Search',
      description: '先制定完整计划，再逐步搜索执行',
      pros: ['任务分解清晰', '便于跟踪进度', '易于调试'],
      cons: ['计划可能出错', '缺乏动态调整', '不适合开放问题'],
      when: '复杂任务、已知结构、可编程问题',
      color: '#f59e0b'
    },
    reflection: {
      name: 'Reflection & Critique',
      description: '对推理结果进行自我反思和修正',
      pros: ['提高答案质量', '发现逻辑错误', '迭代优化'],
      cons: ['增加token消耗', '可能过度修正', '依赖反思能力'],
      when: '开放问题、创意生成、需要高质量输出',
      color: '#8b5cf6'
    },
    selfconsistency: {
      name: 'Self-Consistency',
      description: '采样多条推理路径，选择最一致的答案',
      pros: ['减少随机错误', '提高鲁棒性', '无需训练'],
      cons: ['计算成本高', '需要多次采样', '不保证最优'],
      when: '数学问题、单选题、确定性问题',
      color: '#ef4444'
    }
  };

  const current = strategies[selected];

  return (
    <div className="my-8 p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold mb-6 text-gray-900 dark:text-gray-100">
        推理策略选择器
      </h3>

      <div className="grid grid-cols-5 gap-2 mb-6">
        {(Object.keys(strategies) as Strategy[]).map((key) => (
          <button
            key={key}
            onClick={() => setSelected(key)}
            className={`p-3 rounded-lg border-2 text-center text-sm transition-all ${
              selected === key
                ? 'border-opacity-100'
                : 'border-gray-200 dark:border-gray-700 bg-transparent'
            }`}
            style={{
              borderColor: selected === key ? strategies[key].color : undefined,
              backgroundColor: selected === key ? `${strategies[key].color}15` : undefined
            }}
          >
            {strategies[key].name.split(' ')[0]}
          </button>
        ))}
      </div>

      <div className="p-5 rounded-lg bg-gray-50 dark:bg-gray-700">
        <h4
          className="text-xl font-semibold mb-2"
          style={{ color: current.color }}
        >
          {current.name}
        </h4>
        <p className="text-gray-700 dark:text-gray-300 mb-4">{current.description}</p>

        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <h5 className="font-medium text-green-700 dark:text-green-400 mb-2">✓ 优点</h5>
            <ul className="space-y-1">
              {current.pros.map((pro, idx) => (
                <li key={idx} className="text-sm text-gray-700 dark:text-gray-300 pl-4 relative">
                  <span className="absolute left-0 top-1 text-green-600">•</span>
                  {pro}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h5 className="font-medium text-red-700 dark:text-red-400 mb-2">✗ 缺点</h5>
            <ul className="space-y-1">
              {current.cons.map((con, idx) => (
                <li key={idx} className="text-sm text-gray-700 dark:text-gray-300 pl-4 relative">
                  <span className="absolute left-0 top-1 text-red-600">•</span>
                  {con}
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="px-4 py-3 bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-600">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            <strong>适用场景：</strong> {current.when}
          </span>
        </div>
      </div>

      <div className="mt-6 text-sm text-gray-600 dark:text-gray-400">
        💡 <strong>选择建议：</strong> 复杂问题可以组合多种策略，例如：
        ReAct + Reflection 或 Self-Consistency + CoT 效果更佳。
      </div>
    </div>
  );
}
