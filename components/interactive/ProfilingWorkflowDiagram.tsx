'use client';

import { useState } from 'react';

const steps = [
  { id: 1, title: '运行内核', icon: '▶️', color: 'bg-green-500', desc: '执行CUDA内核程序' },
  { id: 2, title: '收集指标', icon: '📊', color: 'bg-blue-500', desc: '采集性能计数器数据' },
  { id: 3, title: '分析瓶颈', icon: '🔍', color: 'bg-yellow-500', desc: '识别性能瓶颈点' },
  { id: 4, title: '优化迭代', icon: '⚡', color: 'bg-purple-500', desc: '应用优化策略' },
];

export function ProfilingWorkflowDiagram() {
  const [activeStep, setActiveStep] = useState(0);

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-6 text-gray-800">性能分析工作流</h2>
      <div className="flex items-center justify-between relative">
        {steps.map((step, index) => (
          <div key={step.id} className="flex flex-col items-center z-10">
            <button
              onClick={() => setActiveStep(index)}
              className={`w-16 h-16 rounded-full ${step.color} text-white text-2xl 
                flex items-center justify-center transition-all duration-300
                ${activeStep === index ? 'scale-125 ring-4 ring-offset-2 ring-gray-300' : 'hover:scale-110'}`}
            >
              {step.icon}
            </button>
            <span className="mt-2 font-medium text-gray-700">{step.title}</span>
            {activeStep === index && (
              <div className="mt-2 p-2 bg-gray-100 rounded text-sm text-gray-600 text-center max-w-[120px]">
                {step.desc}
              </div>
            )}
          </div>
        ))}
        <div className="absolute top-8 left-0 right-0 h-1 bg-gray-200 -z-0">
          <div
            className="h-full bg-gradient-to-r from-green-500 to-purple-500 transition-all duration-500"
            style={{ width: `${(activeStep / (steps.length - 1)) * 100}%` }}
          />
        </div>
      </div>
      <div className="mt-8 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg">
        <h3 className="font-semibold mb-2">当前步骤: {steps[activeStep].title}</h3>
        <p className="text-gray-600 text-sm">
          {activeStep === 0 && '使用nvprof或ncu工具启动内核 profiling'}
          {activeStep === 1 && '收集SM利用率、内存带宽、指令吞吐量等关键指标'}
          {activeStep === 2 && '根据指标数据识别计算密集型或内存密集型瓶颈'}
          {activeStep === 3 && '针对性应用循环展开、共享内存优化等策略'}
        </p>
      </div>
    </div>
  );
}