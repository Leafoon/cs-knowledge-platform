'use client';

import React, { useState } from 'react';

interface ComponentNode {
  id: string;
  label: string;
  color: string;
  x: number;
  y: number;
  description: string;
}

export function TileLangEcosystemMap() {
  const [selected, setSelected] = useState<string | null>(null);

  const components: ComponentNode[] = [
    { id: 'python', label: 'Python DSL', color: '#3B82F6', x: 50, y: 80, description: '用户友好的Python接口，支持装饰器和上下文管理器' },
    { id: 'tile_ir', label: 'Tile IR', color: '#8B5CF6', x: 50, y: 180, description: '中间表示层，支持循环变换和内存优化' },
    { id: 'tvm', label: 'TVM/Relax', color: '#EC4899', x: 50, y: 280, description: '基于TVM的编译器后端，支持自动调优' },
    { id: 'cuda', label: 'CUDA', color: '#10B981', x: 50, y: 380, description: 'NVIDIA GPU后端，支持SMEM和Tensor Core' },
    { id: 'hip', label: 'HIP/ROCm', color: '#F59E0B', x: 200, y: 380, description: 'AMD GPU后端，支持RDNA和CDNA架构' },
    { id: 'ascend', label: 'Ascend CANN', color: '#EF4444', x: 350, y: 380, description: '华为昇腾后端，支持达芬奇架构' },
  ];

  const arrows = [
    { from: 'python', to: 'tile_ir', label: '编译' },
    { from: 'tile_ir', to: 'tvm', label: '优化' },
    { from: 'tvm', to: 'cuda', label: '代码生成' },
    { from: 'tvm', to: 'hip', label: '代码生成' },
    { from: 'tvm', to: 'ascend', label: '代码生成' },
  ];

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">TileLang 生态系统架构</h2>
      
      <div className="relative bg-gray-800 rounded-lg p-4" style={{ height: 450 }}>
        <svg width="100%" height="100%" viewBox="0 0 450 420">
          {/* Arrows */}
          {arrows.map((arrow, i) => {
            const from = components.find(c => c.id === arrow.from)!;
            const to = components.find(c => c.id === arrow.to)!;
            return (
              <g key={i}>
                <defs>
                  <marker id={`arrowhead-${i}`} markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#6B7280" />
                  </marker>
                </defs>
                <line
                  x1={from.x + 60}
                  y1={from.y + 30}
                  x2={to.x + 60}
                  y2={to.y}
                  stroke="#6B7280"
                  strokeWidth="2"
                  markerEnd={`url(#arrowhead-${i})`}
                />
                <text x={(from.x + to.x) / 2 + 70} y={(from.y + to.y) / 2 + 15} fill="#9CA3AF" fontSize="10" textAnchor="middle">
                  {arrow.label}
                </text>
              </g>
            );
          })}
          
          {/* Component boxes */}
          {components.map((comp) => (
            <g
              key={comp.id}
              onClick={() => setSelected(selected === comp.id ? null : comp.id)}
              className="cursor-pointer"
            >
              <rect
                x={comp.x}
                y={comp.y}
                width={120}
                height={50}
                rx={8}
                fill={selected === comp.id ? comp.color : '#374151'}
                stroke={comp.color}
                strokeWidth={selected === comp.id ? 3 : 1}
              />
              <text x={comp.x + 60} y={comp.y + 30} fill="white" fontSize="12" textAnchor="middle" fontWeight="bold">
                {comp.label}
              </text>
            </g>
          ))}
        </svg>
        
        {/* Info panel */}
        {selected && (
          <div className="absolute bottom-4 right-4 bg-gray-700 rounded-lg p-4 max-w-xs">
            <h3 className="text-white font-bold mb-2">{components.find(c => c.id === selected)?.label}</h3>
            <p className="text-gray-300 text-sm">{components.find(c => c.id === selected)?.description}</p>
          </div>
        )}
      </div>
      
      <div className="mt-4 text-gray-400 text-sm">
        点击组件查看详细信息 | TileLang支持多种硬件后端
      </div>
    </div>
  );
}
