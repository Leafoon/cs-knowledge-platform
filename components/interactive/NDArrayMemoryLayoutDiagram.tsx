'use client';

import React, { useState } from 'react';

interface Dimension {
  label: string;
  size: number;
  stride: number;
}

const layouts: { name: string; desc: string; shape: number[]; strides: number[]; order: string }[] = [
  {
    name: 'C (Row-Major)',
    desc: 'C 语言风格行主序存储，最右边的索引变化最快',
    shape: [3, 4],
    strides: [4, 1],
    order: 'Row-Major',
  },
  {
    name: 'Fortran (Column-Major)',
    desc: 'Fortran 风格列主序存储，最左边的索引变化最快',
    shape: [3, 4],
    strides: [1, 3],
    order: 'Column-Major',
  },
  {
    name: 'NHWC (TF Default)',
    desc: 'TensorFlow 默认布局，通道维度在最后，利于 GPU 向量化',
    shape: [1, 3, 3, 2],
    strides: [18, 6, 2, 1],
    order: 'NHWC',
  },
  {
    name: 'NCHW (PyTorch Default)',
    desc: 'PyTorch 默认布局，空间维度在最后，利于 cuDNN 卷积',
    shape: [1, 2, 3, 3],
    strides: [18, 9, 3, 1],
    order: 'NCHW',
  },
];

export default function NDArrayMemoryLayoutDiagram() {
  const [activeIdx, setActiveIdx] = useState(0);
  const layout = layouts[activeIdx];

  const totalElements = layout.shape.reduce((a, b) => a * b, 1);

  return (
    <div className="p-6 bg-gray-900 rounded-xl border border-gray-700 max-w-3xl mx-auto">
      <h2 className="text-xl font-bold mb-1 bg-gradient-to-r from-indigo-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
        NDArray 内存布局图
      </h2>
      <p className="text-gray-400 text-sm mb-4">shape / strides / data_ptr — NDArray Memory Layout</p>

      <div className="flex gap-2 mb-4 flex-wrap">
        {layouts.map((l, i) => (
          <button
            key={l.name}
            onClick={() => setActiveIdx(i)}
            className={`px-3 py-1.5 text-xs rounded-lg transition-all ${
              i === activeIdx ? 'bg-indigo-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {l.name}
          </button>
        ))}
      </div>

      <div className="bg-gray-800/60 rounded-lg p-4 mb-4">
        <p className="text-gray-300 text-sm mb-3">{layout.desc}</p>

        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="text-center">
            <div className="text-xs text-gray-500 mb-1">shape</div>
            <div className="font-mono text-sm text-indigo-300">({layout.shape.join(', ')})</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500 mb-1">strides</div>
            <div className="font-mono text-sm text-purple-300">({layout.strides.join(', ')})</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500 mb-1">data_ptr</div>
            <div className="font-mono text-sm text-blue-300">0x7f00...</div>
          </div>
        </div>

        <div className="text-xs text-gray-500 mb-2">线性内存布局:</div>
        <div className="flex flex-wrap gap-1">
          {Array.from({ length: totalElements }, (_, i) => (
            <div
              key={i}
              className="w-10 h-10 rounded flex items-center justify-center text-xs font-mono transition-all bg-indigo-500/15 border border-indigo-500/25 text-indigo-300 hover:bg-indigo-500/25"
            >
              [{i}]
            </div>
          ))}
        </div>
      </div>

      <div className="bg-gray-800/60 rounded-lg p-4 mb-4">
        <div className="text-xs text-gray-500 mb-2 font-medium">地址计算公式</div>
        <div className="bg-gray-900 rounded-lg p-3 font-mono text-sm text-indigo-300">
          addr = data_ptr + Σ(index_i × stride_i)
        </div>
        <div className="mt-2 text-xs text-gray-400">
          示例: element[{layout.shape.map((_, i) => 1).join(', ')}] = data_ptr + {layout.shape.map((_, i) => 1).map((idx, i) => `${idx}×${layout.strides[i]}`).join(' + ')} = data_ptr + {layout.shape.map((_, i) => layout.strides[i]).reduce((a, b) => a + b, 0)}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gray-800/60 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-indigo-400">{totalElements}</div>
          <div className="text-xs text-gray-500">总元素数</div>
        </div>
        <div className="bg-gray-800/60 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-purple-400">{totalElements * 4}B</div>
          <div className="text-xs text-gray-500">float32 占用</div>
        </div>
      </div>
    </div>
  );
}
