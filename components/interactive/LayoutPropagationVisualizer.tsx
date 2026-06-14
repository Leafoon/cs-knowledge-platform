'use client';

import React, { useState, useEffect } from 'react';

interface ComputeNode {
  id: string;
  name: string;
  color: string;
  layout: string;
  x: number;
  y: number;
}

export function LayoutPropagationVisualizer() {
  const [propagationStep, setPropagationStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  const nodes: ComputeNode[] = [
    { id: 'input', name: 'Input', color: '#3B82F6', layout: 'Row-Major', x: 50, y: 100 },
    { id: 'matmul', name: 'MatMul', color: '#10B981', layout: 'Row-Major', x: 200, y: 50 },
    { id: 'bias', name: 'Add Bias', color: '#F59E0B', layout: 'Row-Major', x: 200, y: 150 },
    { id: 'relu', name: 'ReLU', color: '#8B5CF6', layout: 'Row-Major', x: 350, y: 100 },
    { id: 'output', name: 'Output', color: '#EC4899', layout: 'Row-Major', x: 500, y: 100 },
  ];

  const edges = [
    { from: 'input', to: 'matmul' },
    { from: 'input', to: 'bias' },
    { from: 'matmul', to: 'relu' },
    { from: 'bias', to: 'relu' },
    { from: 'relu', to: 'output' },
  ];

  useEffect(() => {
    if (isAnimating) {
      const timer = setInterval(() => {
        setPropagationStep((prev) => {
          if (prev >= nodes.length) {
            setIsAnimating(false);
            return 0;
          }
          return prev + 1;
        });
      }, 1000);
      return () => clearInterval(timer);
    }
  }, [isAnimating, nodes.length]);

  const getNodeOpacity = (nodeIndex: number) => {
    if (propagationStep === 0) return 0.3;
    return nodeIndex < propagationStep ? 1 : 0.3;
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">Layout 传播可视化</h2>
      
      {/* Graph visualization */}
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <svg width="600" height="250" viewBox="0 0 600 250">
          {/* Edges */}
          {edges.map((edge, i) => {
            const from = nodes.find((n) => n.id === edge.from)!;
            const to = nodes.find((n) => n.id === edge.to)!;
            return (
              <line
                key={i}
                x1={from.x + 60}
                y1={from.y + 25}
                x2={to.x}
                y2={to.y + 25}
                stroke="#4B5563"
                strokeWidth="2"
                markerEnd="url(#arrowhead)"
              />
            );
          })}
          
          {/* Arrow marker */}
          <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#4B5563" />
            </marker>
          </defs>
          
          {/* Nodes */}
          {nodes.map((node, i) => (
            <g key={node.id} style={{ opacity: getNodeOpacity(i) }}>
              <rect
                x={node.x}
                y={node.y}
                width={120}
                height={50}
                rx={8}
                fill={`${node.color}40`}
                stroke={node.color}
                strokeWidth={2}
              />
              <text x={node.x + 60} y={node.y + 25} fill="white" fontSize="12" textAnchor="middle" fontWeight="bold">
                {node.name}
              </text>
              <text x={node.x + 60} y={node.y + 40} fill="#9CA3AF" fontSize="10" textAnchor="middle">
                {node.layout}
              </text>
            </g>
          ))}
        </svg>
      </div>
      
      {/* Step info */}
      <div className="bg-gray-800 rounded-lg p-4 mb-6">
        <div className="flex items-center justify-between mb-4">
          <span className="text-white font-bold">
            Step {propagationStep} / {nodes.length}
          </span>
          <span className="text-gray-400">
            {propagationStep === 0 ? '初始状态' : `已传播到: ${nodes[propagationStep - 1]?.name}`}
          </span>
        </div>
        
        <div className="flex gap-2">
          {nodes.map((node, i) => (
            <div
              key={i}
              className="flex-1 h-2 rounded-full"
              style={{
                backgroundColor: i < propagationStep ? node.color : '#374151',
              }}
            />
          ))}
        </div>
      </div>
      
      {/* Controls */}
      <div className="flex justify-center gap-4">
        <button
          onClick={() => {
            setPropagationStep(0);
            setIsAnimating(true);
          }}
          disabled={isAnimating}
          className={`px-6 py-2 rounded-lg font-bold transition-all ${
            isAnimating
              ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-500 text-white'
          }`}
        >
          {isAnimating ? '传播中...' : '开始传播'}
        </button>
        <button
          onClick={() => setPropagationStep(0)}
          className="px-6 py-2 bg-gray-700 rounded-lg text-white hover:bg-gray-600"
        >
          重置
        </button>
      </div>
      
      {/* Layout propagation rules */}
      <div className="mt-6 bg-gray-800 rounded-lg p-4">
        <h3 className="text-white font-bold mb-2">Layout 传播规则</h3>
        <ul className="text-gray-300 text-sm space-y-1">
          <li>• <strong>Producer → Consumer</strong>: Layout从生产者传播到消费者</li>
          <li>• <strong>Strict Mode</strong>: 要求所有节点Layout一致</li>
          <li>• <strong>Common Mode</strong>: 自动推断兼容Layout</li>
          <li>• <strong>转换开销</strong>: 不兼容时自动插入转换</li>
        </ul>
      </div>
    </div>
  );
}
