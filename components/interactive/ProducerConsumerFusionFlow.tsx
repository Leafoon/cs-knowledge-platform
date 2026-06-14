'use client';

import { useState } from 'react';

export function ProducerConsumerFusionFlow() {
  const [step, setStep] = useState(0);

  const steps = [
    { title: '生产者加载数据', desc: '从全局内存加载到寄存器' },
    { title: '写入共享内存', desc: '生产者将结果写入共享内存' },
    { title: '同步屏障', desc: '__syncthreads()确保数据就绪' },
    { title: '消费者读取', desc: '从共享内存读取并计算' },
    { title: '写回全局内存', desc: '最终结果写回全局内存' },
  ];

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">生产者-消费者融合流程</h2>
      
      <div className="flex gap-2 mb-4">
        {steps.map((s, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className={`px-3 py-1 rounded text-sm transition-colors ${
              step === i ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-600'
            }`}
          >
            {i + 1}
          </button>
        ))}
      </div>

      <div className="flex items-center justify-between mb-6">
        <div className="p-4 bg-yellow-100 rounded-xl text-center">
          <div className="text-xs text-yellow-600">生产者线程</div>
          <div className="font-bold text-yellow-800">加载 + 计算</div>
        </div>
        
        <div className="flex flex-col items-center">
          <div className="text-2xl">→</div>
          <div className={`px-3 py-1 rounded text-xs ${
            step >= 1 ? 'bg-green-500 text-white' : 'bg-gray-200 text-gray-500'
          }`}>
            共享内存
          </div>
          <div className="text-2xl">→</div>
        </div>
        
        <div className="p-4 bg-purple-100 rounded-xl text-center">
          <div className="text-xs text-purple-600">消费者线程</div>
          <div className="font-bold text-purple-800">读取 + 计算</div>
        </div>
      </div>

      <div className="p-4 bg-gray-50 rounded-xl">
        <div className="font-semibold mb-2">步骤 {step + 1}: {steps[step].title}</div>
        <div className="text-gray-600">{steps[step].desc}</div>
      </div>

      <div className="mt-4 flex gap-2">
        <button
          onClick={() => setStep(Math.max(0, step - 1))}
          disabled={step === 0}
          className="px-4 py-2 bg-gray-200 rounded-lg disabled:opacity-50"
        >
          上一步
        </button>
        <button
          onClick={() => setStep(Math.min(steps.length - 1, step + 1))}
          disabled={step === steps.length - 1}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50"
        >
          下一步
        </button>
      </div>
    </div>
  );
}