'use client';

import React, { useState, useEffect } from 'react';

interface BuildStep {
  name: string;
  command: string;
  description: string;
  duration: number;
}

export function BuildSystemFlow() {
  const [currentStep, setCurrentStep] = useState(-1);
  const [isRunning, setIsRunning] = useState(false);

  const steps: BuildStep[] = [
    { name: 'CMake 配置', command: 'cmake -B build -DCMAKE_BUILD_TYPE=Release', description: '配置CMake构建系统', duration: 2000 },
    { name: 'TVM 构建', command: 'make -j$(nproc)', description: '编译TVM依赖', duration: 5000 },
    { name: 'Python 包构建', command: 'python setup.py build_ext --inplace', description: '构建Python扩展模块', duration: 3000 },
    { name: '安装', command: 'pip install -e .', description: '安装到开发环境', duration: 1000 },
  ];

  const startBuild = () => {
    setCurrentStep(0);
    setIsRunning(true);
  };

  useEffect(() => {
    if (currentStep >= 0 && currentStep < steps.length) {
      const timer = setTimeout(() => {
        if (currentStep < steps.length - 1) {
          setCurrentStep(currentStep + 1);
        } else {
          setIsRunning(false);
        }
      }, steps[currentStep].duration);
      return () => clearTimeout(timer);
    }
  }, [currentStep, steps.length]);

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">构建系统流程</h2>
      
      <div className="relative">
        {/* Flow line */}
        <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gray-700" />
        
        {/* Steps */}
        <div className="space-y-4">
          {steps.map((step, i) => (
            <div key={i} className="relative flex items-start gap-4">
              {/* Step indicator */}
              <div
                className={`relative z-10 w-16 h-16 rounded-full flex items-center justify-center text-white font-bold transition-all duration-300 ${
                  i < currentStep
                    ? 'bg-green-500'
                    : i === currentStep
                    ? 'bg-blue-500 animate-pulse'
                    : 'bg-gray-700'
                }`}
              >
                {i < currentStep ? '✓' : i + 1}
              </div>
              
              {/* Step content */}
              <div
                className={`flex-1 p-4 rounded-lg border transition-all duration-300 ${
                  i === currentStep
                    ? 'border-blue-500 bg-blue-900/30'
                    : i < currentStep
                    ? 'border-green-500 bg-green-900/30'
                    : 'border-gray-700 bg-gray-800'
                }`}
              >
                <h3 className="text-white font-bold mb-1">{step.name}</h3>
                <p className="text-gray-400 text-sm mb-2">{step.description}</p>
                <code className="block bg-gray-900 rounded p-2 text-sm text-green-400 font-mono">
                  $ {step.command}
                </code>
                
                {i === currentStep && (
                  <div className="mt-3 flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                    <span className="text-blue-400 text-sm">正在执行...</span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
      
      <div className="mt-6 text-center">
        <button
          onClick={startBuild}
          disabled={isRunning}
          className={`px-6 py-3 rounded-lg font-bold transition-all ${
            isRunning
              ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-500 text-white'
          }`}
        >
          {isRunning ? '构建中...' : '开始构建'}
        </button>
      </div>
    </div>
  );
}
