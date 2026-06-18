'use client';

import React, { useMemo } from 'react';

export default function RLvsLLMAgentComparison() {
  const comparisonData = useMemo(() => [
    {
      dimension: '学习方式',
      rl: '通过环境奖励信号强化学习',
      llm: '通过大规模预训练+prompt工程学习',
    },
    {
      dimension: '决策空间',
      rl: '连续/离散动作空间，可优化',
      llm: '自然语言空间，符号推理',
    },
    {
      dimension: '探索策略',
      rl: 'ε-greedy、熵正则化',
      llm: '温度采样、top-k/top-p过滤',
    },
    {
      dimension: ' credit assignment',
      rl: '价值函数、策略梯度、REINFORCE',
      llm: 'RLHF、DPO、专家迭代',
    },
    {
      dimension: '环境交互',
      rl: '在线试错，与环境实时交互',
      llm: '上下文推理，离线知识生成',
    },
    {
      dimension: '样本效率',
      rl: '通常需要百万级样本',
      llm: '零样本/少样本，样本效率极高',
    },
    {
      dimension: '泛化能力',
      rl: '任务特定，泛化较弱',
      llm: '极强的zero-shot泛化能力',
    },
    {
      dimension: '收敛稳定性',
      rl: '不稳定，超参数敏感',
      llm: '稳定，预测确定性高',
    },
  ], []);

  return (
    <div className="my-8 p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
        RL-based Agent vs LLM-based Agent 对比
      </h3>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="bg-gray-100 dark:bg-gray-700">
              <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left">维度</th>
              <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left bg-blue-50 dark:bg-blue-900/20">
                RL-based Agent
              </th>
              <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left bg-green-50 dark:bg-green-900/20">
                LLM-based Agent
              </th>
            </tr>
          </thead>
          <tbody>
            {comparisonData.map((row, idx) => (
              <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 font-medium text-gray-900 dark:text-gray-100">
                  {row.dimension}
                </td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-gray-700 dark:text-gray-300 bg-blue-50 dark:bg-blue-900/10">
                  {row.rl}
                </td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-gray-700 dark:text-gray-300 bg-green-50 dark:bg-green-900/10">
                  {row.llm}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded text-sm text-gray-700 dark:text-gray-300">
        <strong>当前趋势：</strong> 现代 AI Agent 通常结合两者优势 ——
        利用 LLM 进行高层推理和规划，用 RL 优化环境交互中的策略，实现"大模型推理 + 强化学习优化"的混合架构。
      </div>
    </div>
  );
}
