'use client';

import React, { useState } from 'react';

type InferenceMode = 'strict' | 'common';

interface LayoutNode {
  id: string;
  name: string;
  color: string;
  layout: string;
}

export function LayoutInferenceFlow() {
  const [mode, setMode] = useState<InferenceMode>('strict');

  const nodes: LayoutNode[] = [
    { id: 'input', name: 'Input', color: '#3B82F6', layout: 'Row-Major' },
    { id: 'producer', name: 'Producer', color: '#10B981', layout: 'Row-Major' },
    { id: 'consumer', name: 'Consumer', color: '#F59E0B', layout: mode === 'strict' ? 'Row-Major' : 'Any' },
    { id: 'output', name: 'Output', color: '#8B5CF6', layout: mode === 'strict' ? 'Row-Major' : 'Col-Major' },
  ];

  const edges = [
    { from: 'input', to: 'producer' },
    { from: 'producer', to: 'consumer' },
    { from: 'consumer', to: 'output' },
  ];

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">Layout 推断流程</h2>
      
      {/* Mode selector */}
      <div className="flex gap-4 mb-6">
        <button
          onClick={() => setMode('strict')}
          className={`flex-1 p-4 rounded-lg border-2 transition-all ${
            mode === 'strict'
              ? 'border-blue-500 bg-blue-900/30'
              : 'border-gray-700 hover:border-gray-600'
          }`}
        >
          <div className="text-white font-bold">Strict Mode</div>
          <div className="text-gray-400 text-sm mt-1">严格模式：Layout必须完全匹配</div>
        </button>
        <button
          onClick={() => setMode('common')}
          className={`flex-1 p-4 rounded-lg border-2 transition-all ${
            mode === 'common'
              ? 'border-green-500 bg-green-900/30'
              : 'border-gray-700 hover:border-gray-600'
          }`}
        >
          <div className="text-white font-bold">Common Mode</div>
          <div className="text-gray-400 text-sm mt-1">通用模式：自动推断兼容Layout</div>
        </button>
      </div>
      
      {/* Flow diagram */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between">
          {nodes.map((node, i) => (
            <React.Fragment key={node.id}>
              <div className="text-center">
                <div
                  className="w-32 h-20 rounded-lg flex items-center justify-center border-2"
                  style={{ borderColor: node.color }}
                >
                  <div>
                    <div className="text-white font-bold">{node.name}</div>
                    <div className="text-gray-400 text-xs mt-1">{node.layout}</div>
                  </div>
                </div>
              </div>
              
              {i < nodes.length - 1 && (
                <div className="flex-1 mx-2">
                  <div className="h-0.5 bg-gray-600 relative">
                    <div
                      className="absolute inset-y-0 left-0 w-full"
                      style={{
                        backgroundColor: mode === 'strict' ? '#10B981' : '#F59E0B',
                      }}
                    />
                  </div>
                  <div className="text-center text-gray-400 text-xs mt-1">
                    {mode === 'strict' ? '验证' : '推断'}
                  </div>
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
      
      {/* Mode explanation */}
      <div className="mt-6 grid grid-cols-2 gap-4">
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-white font-bold mb-2">Strict Mode</h3>
          <ul className="text-gray-300 text-sm space-y-1">
            <li>• Layout必须完全一致</li>
            <li>• 不自动转换</li>
            <li>• 更高性能（无转换开销）</li>
            <li>• 需要手动管理Layout</li>
          </ul>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-white font-bold mb-2">Common Mode</h3>
          <ul className="text-gray-300 text-sm space-y-1">
            <li>• 自动推断兼容Layout</li>
            <li>• 自动插入转换</li>
            <li>• 更易使用</li>
            <li>• 可能有性能开销</li>
          </ul>
        </div>
      </div>
      
      {/* Code example */}
      <div className="mt-6 bg-gray-800 rounded-lg p-4">
        <h3 className="text-white font-bold mb-2">代码示例</h3>
        <pre className="text-sm font-mono text-green-400 overflow-x-auto">
{`# Strict Mode
T.func_with_sch(
    layout_mode="strict",
    ...
)

# Common Mode
T.func_with_sch(
    layout_mode="common",
    ...
)`}
        </pre>
      </div>
    </div>
  );
}
