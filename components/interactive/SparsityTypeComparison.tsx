'use client';

import { useState } from 'react';

const sparsityTypes = [
  {
    name: '结构化稀疏 (2:4)',
    color: 'blue',
    pattern: [
      [1, 0, 1, 0],
      [0, 1, 0, 1],
      [1, 1, 0, 0],
      [0, 0, 1, 1],
    ],
    compression: '50%',
    hardware: 'NVIDIA Ampere+',
    performance: '2x 加速',
    description: '每 4 个元素中恰好有 2 个非零',
  },
  {
    name: '非结构化稀疏',
    color: 'green',
    pattern: [
      [1, 0, 0, 1],
      [0, 0, 1, 0],
      [1, 1, 0, 0],
      [0, 1, 0, 1],
    ],
    compression: '可变',
    hardware: '通用',
    performance: '1.5-3x 加速',
    description: '任意位置的零元素',
  },
  {
    name: '块稀疏',
    color: 'purple',
    pattern: [
      [1, 1, 0, 0],
      [1, 1, 0, 0],
      [0, 0, 1, 1],
      [0, 0, 1, 1],
    ],
    compression: '75% (2x2块)',
    hardware: 'GPU/TPU',
    performance: '1.8x 加速',
    description: '以块为单位的稀疏模式',
  },
];

export default function SparsityTypeComparison() {
  const [selectedType, setSelectedType] = useState<number>(0);
  const [animationStep, setAnimationStep] = useState(0);

  const current = sparsityTypes[selectedType];

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">稀疏类型对比</h2>
      <p className="text-gray-400 text-sm mb-4">结构化 (2:4) vs 非结构化 vs 块稀疏的对比分析</p>

      <div className="flex gap-3 mb-6">
        {sparsityTypes.map((type, idx) => (
          <button
            key={idx}
            onClick={() => setSelectedType(idx)}
            className={`flex-1 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              selectedType === idx
                ? `bg-${type.color}-600 text-white`
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {type.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-6 mb-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-sm font-bold text-gray-300 mb-4">稀疏模式可视化</h3>
          <div className="grid grid-cols-4 gap-1 max-w-[200px] mx-auto">
            {current.pattern.flat().map((val, idx) => (
              <div
                key={idx}
                className={`w-10 h-10 rounded flex items-center justify-center text-sm font-bold transition-all ${
                  val === 1
                    ? `bg-${current.color}-500 text-white`
                    : 'bg-gray-700 text-gray-500'
                }`}
                style={{
                  animationDelay: `${idx * 50}ms`,
                }}
              >
                {val === 1 ? '●' : '○'}
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-400 text-center mt-4">
            ● 非零元素 &nbsp; ○ 零元素
          </p>
        </div>

        <div className="space-y-4">
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold text-gray-300 mb-3">属性对比</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-xs text-gray-400">压缩率</span>
                <span className={`text-sm font-bold text-${current.color}-400`}>{current.compression}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-xs text-gray-400">硬件支持</span>
                <span className="text-sm text-gray-300">{current.hardware}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-xs text-gray-400">性能提升</span>
                <span className="text-sm text-green-400">{current.performance}</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold text-gray-300 mb-2">描述</h3>
            <p className="text-xs text-gray-400">{current.description}</p>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-bold text-gray-300 mb-3">详细对比表</h3>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-500 border-b border-gray-700">
              <th className="text-left py-2">特性</th>
              {sparsityTypes.map((type) => (
                <th key={type.name} className="text-center py-2">{type.name.split('(')[0].trim()}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-gray-700/50">
              <td className="py-2 text-gray-400">稀疏模式</td>
              <td className="py-2 text-center">规则 2:4</td>
              <td className="py-2 text-center">任意</td>
              <td className="py-2 text-center">块状</td>
            </tr>
            <tr className="border-b border-gray-700/50">
              <td className="py-2 text-gray-400">硬件加速</td>
              <td className="py-2 text-center text-green-400">原生支持</td>
              <td className="py-2 text-center text-yellow-400">需专用库</td>
              <td className="py-2 text-center text-blue-400">部分支持</td>
            </tr>
            <tr className="border-b border-gray-700/50">
              <td className="py-2 text-gray-400">实现复杂度</td>
              <td className="py-2 text-center text-green-400">低</td>
              <td className="py-2 text-center text-red-400">高</td>
              <td className="py-2 text-center text-yellow-400">中</td>
            </tr>
            <tr>
              <td className="py-2 text-gray-400">精度影响</td>
              <td className="py-2 text-center">小</td>
              <td className="py-2 text-center">可调</td>
              <td className="py-2 text-center">中等</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
