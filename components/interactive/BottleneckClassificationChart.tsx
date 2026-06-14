'use client';

import { useState } from 'react';

interface TreeNode {
  id: string;
  question?: string;
  answer?: string;
  left?: string;
  right?: string;
  optimization?: string;
}

const decisionTree: Record<string, TreeNode> = {
  root: {
    id: 'root',
    question: 'SM利用率 < 50%?',
    left: 'memory',
    right: 'compute',
  },
  memory: {
    id: 'memory',
    question: '内存带宽利用率 < 70%?',
    left: 'coalescing',
    right: 'shared_mem',
  },
  compute: {
    id: 'compute',
    question: 'Tensor Core利用率 < 60%?',
    left: 'instruction',
    right: 'occupancy',
  },
  coalescing: {
    id: 'coalescing',
    optimization: '优化内存合并访问',
  },
  shared_mem: {
    id: 'shared_mem',
    optimization: '增加共享内存使用',
  },
  instruction: {
    id: 'instruction',
    optimization: '优化指令级并行',
  },
  occupancy: {
    id: 'occupancy',
    optimization: '调整线程块大小',
  },
};

export function BottleneckClassificationChart() {
  const [path, setPath] = useState<string[]>(['root']);
  const currentId = path[path.length - 1];
  const current = decisionTree[currentId];

  const handleChoice = (direction: 'left' | 'right') => {
    const nextId = direction === 'left' ? current.left : current.right;
    if (nextId) setPath([...path, nextId]);
  };

  const handleReset = () => setPath(['root']);

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">瓶颈分类决策树</h2>
      
      <div className="flex gap-2 mb-4">
        {path.map((id, i) => (
          <span key={i} className="px-2 py-1 bg-gray-100 rounded text-sm text-gray-600">
            {decisionTree[id].question || decisionTree[id].optimization}
          </span>
        ))}
      </div>

      {current.optimization ? (
        <div className="p-6 bg-green-50 border-2 border-green-200 rounded-xl text-center">
          <div className="text-3xl mb-2">✅</div>
          <div className="text-lg font-semibold text-green-700">{current.optimization}</div>
          <button
            onClick={handleReset}
            className="mt-4 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600"
          >
            重新开始
          </button>
        </div>
      ) : (
        <div className="text-center">
          <div className="p-4 bg-blue-50 rounded-xl mb-6">
            <div className="text-lg font-semibold text-blue-700">{current.question}</div>
          </div>
          <div className="flex justify-center gap-4">
            <button
              onClick={() => handleChoice('left')}
              className="px-6 py-3 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
            >
              是 ✓
            </button>
            <button
              onClick={() => handleChoice('right')}
              className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
            >
              否 ✗
            </button>
          </div>
        </div>
      )}
    </div>
  );
}