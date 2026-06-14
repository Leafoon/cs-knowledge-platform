'use client';

import React, { useState } from 'react';

interface StackLayer {
  name: string;
  color: string;
  icon: string;
  description: string;
}

export function TileLangTechStackOverview() {
  const [hoveredLayer, setHoveredLayer] = useState<number | null>(null);

  const layers: StackLayer[] = [
    { name: 'Python Frontend', color: '#3B82F6', icon: '🐍', description: '用户友好的Python接口，支持装饰器和上下文管理器' },
    { name: 'Tile IR', color: '#8B5CF6', icon: '📝', description: '中间表示层，支持循环变换和内存优化' },
    { name: 'TensorIR', color: '#EC4899', icon: '🔧', description: '张量级别的中间表示，支持自动微分' },
    { name: 'LLVM', color: '#F59E0B', icon: '⚙️', description: '底层编译器，生成高效的机器码' },
    { name: 'Hardware', color: '#10B981', icon: '🖥️', description: '支持NVIDIA、AMD、华为昇腾等多种硬件' },
  ];

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">TileLang 技术栈概览</h2>
      
      <div className="relative max-w-2xl mx-auto">
        {/* Vertical pipeline */}
        <div className="relative">
          {layers.map((layer, i) => (
            <div
              key={i}
              className="relative mb-4 last:mb-0"
              onMouseEnter={() => setHoveredLayer(i)}
              onMouseLeave={() => setHoveredLayer(null)}
            >
              {/* Connector line */}
              {i < layers.length - 1 && (
                <div className="absolute left-1/2 top-full w-0.5 h-4 bg-gray-600 -translate-x-1/2" />
              )}
              
              {/* Layer box */}
              <div
                className={`relative p-4 rounded-lg border-2 transition-all duration-200 cursor-pointer ${
                  hoveredLayer === i ? 'scale-105 shadow-lg' : ''
                }`}
                style={{
                  borderColor: layer.color,
                  backgroundColor: hoveredLayer === i ? `${layer.color}20` : '#1F2937',
                }}
              >
                <div className="flex items-center gap-3">
                  <span className="text-2xl">{layer.icon}</span>
                  <div className="flex-1">
                    <h3 className="text-white font-bold">{layer.name}</h3>
                    {hoveredLayer === i && (
                      <p className="text-gray-300 text-sm mt-1">{layer.description}</p>
                    )}
                  </div>
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: layer.color }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
        
        {/* Flow arrows */}
        <div className="absolute right-0 top-0 bottom-0 w-16 flex flex-col items-center justify-center">
          <div className="text-gray-400 text-sm writing-mode-vertical" style={{ writingMode: 'vertical-rl' }}>
            编译流程 →
          </div>
        </div>
      </div>
      
      <div className="mt-6 text-center text-gray-400 text-sm">
        悬停查看各层详细说明 | 从Python到硬件的完整编译链路
      </div>
    </div>
  );
}
