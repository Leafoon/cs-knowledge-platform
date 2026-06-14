'use client';

import React, { useState } from 'react';

interface OptimizationStep {
  name: string;
  tflops: number;
  color: string;
  description: string;
  techniques: string[];
}

export function GEMMOptimizationPipeline() {
  const [activeStep, setActiveStep] = useState(0);

  const steps: OptimizationStep[] = [
    {
      name: 'Naive',
      tflops: 5.2,
      color: '#6B7280',
      description: '最基础的三重循环实现',
      techniques: ['三重嵌套循环', '无优化', '直接Global Memory访问'],
    },
    {
      name: 'Tiling',
      tflops: 25.8,
      color: '#3B82F6',
      description: '矩阵分块，提高数据局部性',
      techniques: ['矩阵分块', '循环展开', '减少Global Memory访问'],
    },
    {
      name: 'Shared Memory',
      tflops: 52.3,
      color: '#10B981',
      description: '使用Shared Memory缓存数据',
      techniques: ['Shared Memory缓存', '数据复用', '减少带宽压力'],
    },
    {
      name: 'Pipelining',
      tflops: 72.1,
      color: '#F59E0B',
      description: '软件流水线，计算与访存重叠',
      techniques: ['异步加载', '计算访存重叠', '隐藏延迟'],
    },
    {
      name: 'Tensor Core',
      tflops: 92.5,
      color: '#EF4444',
      description: '使用Tensor Core加速矩阵乘法',
      techniques: ['Tensor Core', 'WMMA指令', '混合精度'],
    },
  ];

  const maxTflops = 100;

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">GEMM 优化流水线</h2>
      
      {/* Pipeline visualization */}
      <div className="relative mb-8">
        <div className="flex items-center justify-between">
          {steps.map((step, i) => (
            <React.Fragment key={i}>
              <div
                className={`relative cursor-pointer transition-all ${
                  i === activeStep ? 'scale-110' : ''
                }`}
                onClick={() => setActiveStep(i)}
              >
                <div
                  className="w-20 h-20 rounded-full flex items-center justify-center border-4 transition-all"
                  style={{
                    borderColor: step.color,
                    backgroundColor: i === activeStep ? `${step.color}40` : '#1F2937',
                  }}
                >
                  <span className="text-2xl font-bold" style={{ color: step.color }}>
                    {i + 1}
                  </span>
                </div>
                <div className="text-center mt-2">
                  <div className="text-white font-bold text-sm">{step.name}</div>
                  <div className="text-gray-400 text-xs">{step.tflops} TFLOPS</div>
                </div>
              </div>
              
              {i < steps.length - 1 && (
                <div className="flex-1 h-1 bg-gray-700 mx-2 relative">
                  <div
                    className="absolute inset-y-0 left-0 transition-all duration-500"
                    style={{
                      width: i < activeStep ? '100%' : '0%',
                      backgroundColor: steps[i + 1].color,
                    }}
                  />
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
      
      {/* Step details */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold" style={{ color: steps[activeStep].color }}>
            {steps[activeStep].name}
          </h3>
          <span className="text-2xl font-bold text-white">
            {steps[activeStep].tflops} TFLOPS
          </span>
        </div>
        
        <p className="text-gray-300 mb-4">{steps[activeStep].description}</p>
        
        <div className="mb-4">
          <div className="text-gray-400 text-sm mb-2">关键技术:</div>
          <div className="flex flex-wrap gap-2">
            {steps[activeStep].techniques.map((tech, i) => (
              <span
                key={i}
                className="px-3 py-1 rounded-full text-sm"
                style={{
                  backgroundColor: `${steps[activeStep].color}30`,
                  color: steps[activeStep].color,
                }}
              >
                {tech}
              </span>
            ))}
          </div>
        </div>
        
        {/* Performance bar */}
        <div className="mt-4">
          <div className="text-gray-400 text-sm mb-2">性能 (相对理论峰值):</div>
          <div className="h-8 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500 flex items-center justify-end pr-4"
              style={{
                width: `${(steps[activeStep].tflops / maxTflops) * 100}%`,
                backgroundColor: steps[activeStep].color,
              }}
            >
              <span className="text-white font-bold text-sm">
                {Math.round((steps[activeStep].tflops / maxTflops) * 100)}%
              </span>
            </div>
          </div>
        </div>
      </div>
      
      {/* Navigation */}
      <div className="mt-6 flex justify-center gap-4">
        <button
          onClick={() => setActiveStep(Math.max(0, activeStep - 1))}
          disabled={activeStep === 0}
          className="px-4 py-2 bg-gray-700 rounded-lg text-white hover:bg-gray-600 disabled:opacity-50"
        >
          ← 上一步
        </button>
        <button
          onClick={() => setActiveStep(Math.min(steps.length - 1, activeStep + 1))}
          disabled={activeStep === steps.length - 1}
          className="px-4 py-2 bg-gray-700 rounded-lg text-white hover:bg-gray-600 disabled:opacity-50"
        >
          下一步 →
        </button>
      </div>
    </div>
  );
}
