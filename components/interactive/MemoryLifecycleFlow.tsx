'use client';

import React, { useState, useEffect } from 'react';

interface LifecycleStage {
  name: string;
  color: string;
  icon: string;
  description: string;
  code: string;
}

export function MemoryLifecycleFlow() {
  const [currentStage, setCurrentStage] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  const stages: LifecycleStage[] = [
    {
      name: 'Allocate',
      color: '#3B82F6',
      icon: '📦',
      description: '分配内存空间',
      code: 'A_shared = T.alloc_shared((BM, BK), "float16")',
    },
    {
      name: 'Load',
      color: '#10B981',
      icon: '📥',
      description: '从Global Memory加载数据',
      code: 'T.tvm_load_matrix(A, [row, col], A_shared)',
    },
    {
      name: 'Compute',
      color: '#F59E0B',
      icon: '⚡',
      description: '执行计算',
      code: 'T.tvm_gemm(A_shared, B_shared, C_local)',
    },
    {
      name: 'Store',
      color: '#8B5CF6',
      icon: '📤',
      description: '写回结果',
      code: 'T.tvm_store_matrix(C_local, C, [row, col])',
    },
    {
      name: 'Free',
      color: '#EF4444',
      icon: '🗑️',
      description: '释放内存',
      code: '# 内存自动释放 (作用域结束)',
    },
  ];

  useEffect(() => {
    if (isAnimating) {
      const timer = setInterval(() => {
        setCurrentStage((prev) => {
          if (prev >= stages.length - 1) {
            setIsAnimating(false);
            return 0;
          }
          return prev + 1;
        });
      }, 1500);
      return () => clearInterval(timer);
    }
  }, [isAnimating, stages.length]);

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">内存生命周期流程</h2>
      
      {/* Flow diagram */}
      <div className="relative">
        <div className="flex items-center justify-between">
          {stages.map((stage, i) => (
            <React.Fragment key={i}>
              {/* Stage node */}
              <div
                className={`relative p-4 rounded-lg border-2 transition-all duration-300 cursor-pointer ${
                  i === currentStage ? 'scale-110 shadow-lg' : ''
                }`}
                style={{
                  borderColor: stage.color,
                  backgroundColor: i === currentStage ? `${stage.color}30` : '#1F2937',
                }}
                onClick={() => setCurrentStage(i)}
              >
                <div className="text-center">
                  <span className="text-2xl">{stage.icon}</span>
                  <div className="text-white font-bold mt-1">{stage.name}</div>
                </div>
                
                {i === currentStage && (
                  <div className="absolute -bottom-2 left-1/2 w-4 h-4 bg-gray-900 border-b-2 border-r-2 transform rotate-45 -translate-x-1/2" 
                    style={{ borderColor: stage.color }}
                  />
                )}
              </div>
              
              {/* Arrow */}
              {i < stages.length - 1 && (
                <div className="flex-1 h-0.5 bg-gray-600 relative">
                  <div
                    className="absolute inset-y-0 left-0 transition-all duration-300"
                    style={{
                      width: i < currentStage ? '100%' : '0%',
                      backgroundColor: stages[i + 1].color,
                    }}
                  />
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
      
      {/* Current stage details */}
      <div className="mt-8 bg-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-3xl">{stages[currentStage].icon}</span>
          <div>
            <h3 className="text-xl font-bold" style={{ color: stages[currentStage].color }}>
              Stage {currentStage + 1}: {stages[currentStage].name}
            </h3>
            <p className="text-gray-400">{stages[currentStage].description}</p>
          </div>
        </div>
        
        <div className="bg-gray-900 rounded-lg p-4">
          <code className="text-green-400 font-mono text-sm">
            {stages[currentStage].code}
          </code>
        </div>
      </div>
      
      {/* Animation control */}
      <div className="mt-6 flex justify-center gap-4">
        <button
          onClick={() => setIsAnimating(!isAnimating)}
          className={`px-6 py-2 rounded-lg font-bold transition-all ${
            isAnimating
              ? 'bg-red-600 hover:bg-red-500 text-white'
              : 'bg-blue-600 hover:bg-blue-500 text-white'
          }`}
        >
          {isAnimating ? '停止动画' : '播放动画'}
        </button>
        <button
          onClick={() => setCurrentStage(0)}
          className="px-6 py-2 bg-gray-700 rounded-lg text-white hover:bg-gray-600"
        >
          重置
        </button>
      </div>
    </div>
  );
}
