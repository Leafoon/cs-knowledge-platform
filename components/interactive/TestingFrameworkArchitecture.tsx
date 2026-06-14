'use client';

import { useState } from 'react';

const testLevels = [
  {
    level: '单元测试',
    icon: '🧪',
    color: 'green',
    description: '验证单个内核函数的正确性',
    count: 156,
    coverage: 92,
    examples: [
      'GEMM 输出正确性',
      'Softmax 数值稳定性',
      'LayerNorm 精度验证',
    ],
    tools: ['pytest', 'tile_lang.testing'],
  },
  {
    level: '集成测试',
    icon: '🔗',
    color: 'blue',
    description: '验证多个内核组合工作正常',
    count: 48,
    coverage: 85,
    examples: [
      'Transformer Block 端到端',
      'MoE 层组合测试',
      'Attention + FFN 联合',
    ],
    tools: ['pytest', 'mock'],
  },
  {
    level: '基准测试',
    icon: '📊',
    color: 'purple',
    description: '性能基准和回归测试',
    count: 32,
    coverage: 78,
    examples: [
      'GEMM vs cuBLAS',
      'Flash Attention 吞吐量',
      '内存占用监控',
    ],
    tools: ['tile_lang.benchmark', 'nvtx'],
  },
];

const testCases = [
  { name: 'test_gemm_basic', level: 'unit', status: 'pass', time: '12ms' },
  { name: 'test_gemm_large', level: 'unit', status: 'pass', time: '45ms' },
  { name: 'test_flash_attn', level: 'unit', status: 'pass', time: '28ms' },
  { name: 'test_transformer_block', level: 'integration', status: 'pass', time: '156ms' },
  { name: 'test_moe_layer', level: 'integration', status: 'fail', time: '234ms' },
  { name: 'bench_gemm_vs_cublas', level: 'benchmark', status: 'pass', time: '2.3s' },
  { name: 'bench_flash_attn', level: 'benchmark', status: 'running', time: '...' },
];

export default function TestingFrameworkArchitecture() {
  const [selectedLevel, setSelectedLevel] = useState<number>(0);
  const [showCases, setShowCases] = useState(false);

  const currentLevel = testLevels[selectedLevel];

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">测试框架架构</h2>
      <p className="text-gray-400 text-sm mb-4">单元测试 → 集成测试 → 基准测试的测试金字塔</p>

      <div className="relative h-64 mb-6">
        <svg viewBox="0 0 500 250" className="w-full h-full">
          {/* Pyramid layers */}
          {testLevels.map((level, idx) => {
            const widths = [200, 320, 440];
            const heights = [70, 70, 70];
            const y = idx * 75 + 10;
            const width = widths[idx];
            const x = (500 - width) / 2;

            return (
              <g
                key={idx}
                className="cursor-pointer"
                onClick={() => setSelectedLevel(idx)}
              >
                <rect
                  x={x}
                  y={y}
                  width={width}
                  height={heights[idx]}
                  rx="4"
                  fill={selectedLevel === idx ? '#1f2937' : '#111827'}
                  stroke={selectedLevel === idx ? '#3b82f6' : '#374151'}
                  strokeWidth={selectedLevel === idx ? 2 : 1}
                />
                <text
                  x={250}
                  y={y + 30}
                  fill="#fff"
                  fontSize="14"
                  fontWeight="bold"
                  textAnchor="middle"
                >
                  {level.icon} {level.level}
                </text>
                <text
                  x={250}
                  y={y + 50}
                  fill="#9ca3af"
                  fontSize="10"
                  textAnchor="middle"
                >
                  {level.count} 个测试 | 覆盖率 {level.coverage}%
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="flex items-center gap-4 mb-4">
        <div className="flex gap-2">
          {testLevels.map((level, idx) => (
            <button
              key={idx}
              onClick={() => setSelectedLevel(idx)}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
                selectedLevel === idx ? 'bg-blue-600' : 'bg-gray-800 text-gray-400'
              }`}
            >
              {level.icon} {level.level}
            </button>
          ))}
        </div>
        <button
          onClick={() => setShowCases(!showCases)}
          className="ml-auto px-3 py-1.5 rounded text-xs font-medium bg-gray-800 text-gray-400 hover:bg-gray-700"
        >
          {showCases ? '隐藏测试用例' : '显示测试用例'}
        </button>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-6">
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-bold text-gray-300 mb-3">{currentLevel.level}详情</h3>
          <p className="text-xs text-gray-400 mb-4">{currentLevel.description}</p>

          <div className="mb-4">
            <h4 className="text-xs font-bold text-gray-400 mb-2">测试用例示例</h4>
            <div className="space-y-2">
              {currentLevel.examples.map((ex, idx) => (
                <div key={idx} className="flex items-center gap-2 text-xs text-gray-300">
                  <span className={`text-${currentLevel.color}-400`}>●</span>
                  {ex}
                </div>
              ))}
            </div>
          </div>

          <div>
            <h4 className="text-xs font-bold text-gray-400 mb-2">使用工具</h4>
            <div className="flex gap-2">
              {currentLevel.tools.map((tool, idx) => (
                <span key={idx} className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-300">
                  {tool}
                </span>
              ))}
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-bold text-gray-300 mb-3">覆盖率统计</h3>
          <div className="space-y-4">
            {testLevels.map((level, idx) => (
              <div key={idx}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-gray-400">{level.icon} {level.level}</span>
                  <span className="text-gray-300">{level.coverage}%</span>
                </div>
                <div className="w-full h-2 bg-gray-700 rounded-full">
                  <div
                    className={`h-full bg-${level.color}-500 rounded-full`}
                    style={{ width: `${level.coverage}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 pt-4 border-t border-gray-700">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">总体覆盖率</span>
              <span className="text-white font-bold">
                {Math.round(testLevels.reduce((sum, l) => sum + l.coverage, 0) / testLevels.length)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {showCases && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-bold text-gray-300 mb-3">测试用例列表</h3>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-500 border-b border-gray-700">
                <th className="text-left py-2">名称</th>
                <th className="text-left py-2">级别</th>
                <th className="text-left py-2">状态</th>
                <th className="text-right py-2">耗时</th>
              </tr>
            </thead>
            <tbody>
              {testCases.map((tc, idx) => (
                <tr key={idx} className="border-b border-gray-700/50 hover:bg-gray-750">
                  <td className="py-2 font-mono text-gray-300">{tc.name}</td>
                  <td className="py-2 text-gray-400">{tc.level}</td>
                  <td className="py-2">
                    <span className={`px-2 py-0.5 rounded text-[10px] ${
                      tc.status === 'pass' ? 'bg-green-900/50 text-green-400' :
                      tc.status === 'fail' ? 'bg-red-900/50 text-red-400' :
                      'bg-yellow-900/50 text-yellow-400'
                    }`}>
                      {tc.status}
                    </span>
                  </td>
                  <td className="py-2 text-right text-gray-400">{tc.time}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
