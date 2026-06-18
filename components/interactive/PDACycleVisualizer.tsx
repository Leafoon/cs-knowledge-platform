'use client';

import React, { useState, useMemo } from 'react';

export default function PDACycleVisualizer() {
  const [step, setStep] = useState(0);

  const steps = useMemo(() => [
    {
      name: 'Perception (感知)',
      description: '观察环境，获取当前状态信息',
      color: '#3b82f6',
      details: [
        '接收用户输入',
        '观测环境反馈',
        '整合历史信息',
        '更新状态表示'
      ]
    },
    {
      name: 'Decision (决策)',
      description: '根据当前状态决定下一步行动',
      color: '#10b981',
      details: [
        '推理当前问题',
        '选择可用工具',
        '规划行动步骤',
        '生成执行指令'
      ]
    },
    {
      name: 'Action (行动)',
      description: '执行决策，调用工具获取结果',
      color: '#f59e0b',
      details: [
        '调用外部工具/API',
        '执行计算/搜索',
        '获取执行结果',
        '返回给感知模块'
      ]
    }
  ], []);

  return (
    <div className="my-8 p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold mb-6 text-gray-900 dark:text-gray-100">
        PDA 循环：感知-决策-行动
      </h3>

      <div className="flex justify-between items-center mb-8">
        {steps.map((s, idx) => (
          <div
            key={idx}
            className={`flex-1 text-center mx-2 p-4 rounded-lg border-2 transition-all ${
              idx === step
                ? `border-${s.color.split('#')[1]} bg-${s.color.split('#')[1]}/10`
                : 'border-gray-200 dark:border-gray-700 bg-transparent'
            }`}
            style={{
              borderColor: idx === step ? s.color : undefined,
              backgroundColor: idx === step ? `${s.color}20` : undefined
            }}
            onClick={() => setStep(idx)}
          >
            <div
              className="w-8 h-8 mx-auto rounded-full flex items-center justify-center text-white font-bold mb-2"
              style={{ backgroundColor: s.color }}
            >
              {idx + 1}
            </div>
            <div className="font-medium text-gray-900 dark:text-gray-100">{s.name}</div>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-center mb-6">
        {steps.map((s, idx) => (
          <React.Fragment key={idx}>
            <div
              className="w-24 h-16 rounded-lg flex items-center justify-center text-white font-semibold"
              style={{ backgroundColor: s.color }}
            >
              {s.name.split(' ')[0]}
            </div>
            {idx < steps.length - 1 && (
              <div className="text-2xl text-gray-400 mx-2">→</div>
            )}
          </React.Fragment>
        ))}
        <div className="text-2xl text-gray-400 mx-2">↻</div>
      </div>

      <div className="p-5 bg-gray-50 dark:bg-gray-700 rounded-lg">
        <h4 className="font-semibold text-xl mb-2 text-gray-900 dark:text-gray-100" style={{ color: steps[step].color }}>
          {steps[step].name}
        </h4>
        <p className="text-gray-700 dark:text-gray-300 mb-4">{steps[step].description}</p>
        <div className="grid grid-cols-2 gap-2">
          {steps[step].details.map((detail, idx) => (
            <div
              key={idx}
              className="px-3 py-2 bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-600 text-sm text-gray-700 dark:text-gray-300"
            >
              • {detail}
            </div>
          ))}
        </div>
      </div>

      <div className="mt-6 text-sm text-gray-600 dark:text-gray-400">
        💡 <strong>循环本质：</strong> Agent 通过持续的 PDA 循环逐步逼近目标，
        每一轮循环都基于新的观察调整决策，这是 Agent 区别于传统静态程序的核心特征。
      </div>
    </div>
  );
}
