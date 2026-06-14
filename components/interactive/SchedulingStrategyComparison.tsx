'use client';

import { useState } from 'react';

const strategies = [
  {
    name: '手动调度',
    icon: '✋',
    color: 'red',
    description: '开发者完全控制每个优化步骤',
    pros: ['精细控制', '可预测性能', '适合专家'],
    cons: ['学习成本高', '开发周期长', '易出错'],
    effort: 10,
    performance: 95,
    examples: ['CUDA Kernel', '手写汇编'],
  },
  {
    name: '半自动调度',
    icon: '⚙️',
    color: 'yellow',
    description: '开发者指定策略，编译器自动执行',
    pros: ['平衡控制与效率', '可调试', '灵活'],
    cons: ['需要理解调度', '中等学习曲线', '需要调优'],
    effort: 6,
    performance: 90,
    examples: ['TileLang', 'TVM Schedule'],
  },
  {
    name: '全自动调度',
    icon: '🤖',
    color: 'green',
    description: '编译器自动搜索最优调度',
    pros: ['低开发成本', '快速迭代', '无需专家'],
    cons: ['搜索时间长', '可解释性差', '依赖搜索空间'],
    effort: 2,
    performance: 85,
    examples: ['AutoTVM', 'Ansor'],
  },
];

export default function SchedulingStrategyComparison() {
  const [selectedStrategy, setSelectedStrategy] = useState<number>(1);
  const current = strategies[selectedStrategy];

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">调度策略对比</h2>
      <p className="text-gray-400 text-sm mb-4">手动 vs 半自动 vs 全自动调度策略的权衡分析</p>

      <div className="flex gap-3 mb-6">
        {strategies.map((s, idx) => (
          <button
            key={idx}
            onClick={() => setSelectedStrategy(idx)}
            className={`flex items-center gap-2 px-4 py-3 rounded-lg text-sm font-medium transition-all flex-1 ${
              selectedStrategy === idx
                ? `bg-${s.color}-600 text-white`
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            <span className="text-xl">{s.icon}</span>
            <span>{s.name}</span>
          </button>
        ))}
      </div>

      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-3xl">{current.icon}</span>
          <div>
            <h3 className={`text-lg font-bold text-${current.color}-400`}>{current.name}</h3>
            <p className="text-sm text-gray-400">{current.description}</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6 mb-6">
          <div>
            <h4 className="text-sm font-bold text-green-400 mb-2">优势</h4>
            <ul className="space-y-1">
              {current.pros.map((pro, idx) => (
                <li key={idx} className="flex items-center gap-2 text-sm text-gray-300">
                  <span className="text-green-400">✓</span> {pro}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="text-sm font-bold text-red-400 mb-2">劣势</h4>
            <ul className="space-y-1">
              {current.cons.map((con, idx) => (
                <li key={idx} className="flex items-center gap-2 text-sm text-gray-300">
                  <span className="text-red-400">✗</span> {con}
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">开发成本</span>
              <span className="text-gray-300">{current.effort}/10</span>
            </div>
            <div className="w-full h-2 bg-gray-700 rounded-full">
              <div
                className={`h-full bg-${current.color}-500 rounded-full`}
                style={{ width: `${current.effort * 10}%` }}
              />
            </div>
          </div>
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">性能上限</span>
              <span className="text-gray-300">{current.performance}%</span>
            </div>
            <div className="w-full h-2 bg-gray-700 rounded-full">
              <div
                className={`h-full bg-${current.color}-500 rounded-full`}
                style={{ width: `${current.performance}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-bold text-gray-300 mb-3">典型应用场景</h3>
        <div className="grid grid-cols-3 gap-4 text-xs">
          {strategies.map((s, idx) => (
            <div key={idx} className={`p-3 rounded ${selectedStrategy === idx ? 'bg-gray-700' : ''}`}>
              <div className={`font-bold text-${s.color}-400 mb-1`}>{s.name}</div>
              <div className="text-gray-400">
                {s.examples.map((ex, i) => (
                  <div key={i}>• {ex}</div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
