'use client';

import { useState } from 'react';

const operators = ['GEMM', 'Flash Attention', 'LayerNorm', 'Softmax', 'Conv2D'];
const hardware = ['A100', 'H100', 'MI250X', 'Ascend 910B'];
const frameworks = ['TileLang', 'Triton', 'cuBLAS', 'Vendor'];

const benchmarkData: Record<string, Record<string, Record<string, number>>> = {
  GEMM: {
    A100: { TileLang: 95, Triton: 88, cuBLAS: 100, Vendor: 92 },
    H100: { TileLang: 93, Triton: 86, cuBLAS: 100, Vendor: 90 },
    MI250X: { TileLang: 91, Triton: 84, cuBLAS: 98, Vendor: 89 },
    'Ascend 910B': { TileLang: 89, Triton: 82, cuBLAS: 96, Vendor: 88 },
  },
  'Flash Attention': {
    A100: { TileLang: 92, Triton: 88, cuBLAS: 85, Vendor: 90 },
    H100: { TileLang: 94, Triton: 90, cuBLAS: 87, Vendor: 92 },
    MI250X: { TileLang: 88, Triton: 84, cuBLAS: 82, Vendor: 86 },
    'Ascend 910B': { TileLang: 90, Triton: 86, cuBLAS: 84, Vendor: 88 },
  },
  LayerNorm: {
    A100: { TileLang: 85, Triton: 82, cuBLAS: 78, Vendor: 80 },
    H100: { TileLang: 88, Triton: 85, cuBLAS: 80, Vendor: 83 },
    MI250X: { TileLang: 82, Triton: 79, cuBLAS: 76, Vendor: 78 },
    'Ascend 910B': { TileLang: 84, Triton: 81, cuBLAS: 78, Vendor: 80 },
  },
  Softmax: {
    A100: { TileLang: 82, Triton: 80, cuBLAS: 75, Vendor: 78 },
    H100: { TileLang: 85, Triton: 83, cuBLAS: 78, Vendor: 81 },
    MI250X: { TileLang: 79, Triton: 77, cuBLAS: 73, Vendor: 76 },
    'Ascend 910B': { TileLang: 81, Triton: 79, cuBLAS: 75, Vendor: 78 },
  },
  Conv2D: {
    A100: { TileLang: 88, Triton: 85, cuBLAS: 92, Vendor: 87 },
    H100: { TileLang: 90, Triton: 87, cuBLAS: 94, Vendor: 89 },
    MI250X: { TileLang: 86, Triton: 83, cuBLAS: 90, Vendor: 85 },
    'Ascend 910B': { TileLang: 84, Triton: 81, cuBLAS: 88, Vendor: 83 },
  },
};

export default function PerformanceBenchmarkMatrix() {
  const [selectedOperator, setSelectedOperator] = useState<string>('GEMM');
  const [selectedFramework, setSelectedFramework] = useState<string>('TileLang');

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">性能基准矩阵</h2>
      <p className="text-gray-400 text-sm mb-4">算子 × 硬件 × 框架的全面性能基准测试结果</p>

      <div className="flex gap-4 mb-6">
        <div>
          <label className="text-xs text-gray-400 block mb-1">算子</label>
          <select
            value={selectedOperator}
            onChange={(e) => setSelectedOperator(e.target.value)}
            className="bg-gray-800 text-white text-sm rounded px-3 py-2 border border-gray-600"
          >
            {operators.map((op) => (
              <option key={op} value={op}>{op}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">框架</label>
          <select
            value={selectedFramework}
            onChange={(e) => setSelectedFramework(e.target.value)}
            className="bg-gray-800 text-white text-sm rounded px-3 py-2 border border-gray-600"
          >
            {frameworks.map((fw) => (
              <option key={fw} value={fw}>{fw}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg overflow-hidden mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-700">
              <th className="text-left px-4 py-3 text-gray-300">硬件 \ 框架</th>
              {frameworks.map((fw) => (
                <th
                  key={fw}
                  className={`text-center px-4 py-3 ${
                    fw === selectedFramework ? 'text-yellow-400 bg-gray-600' : 'text-gray-300'
                  }`}
                >
                  {fw}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {hardware.map((hw) => (
              <tr key={hw} className="border-t border-gray-700 hover:bg-gray-750">
                <td className="px-4 py-3 text-gray-300 font-medium">{hw}</td>
                {frameworks.map((fw) => {
                  const value = benchmarkData[selectedOperator]?.[hw]?.[fw] || 0;
                  const isHighlighted = fw === selectedFramework;
                  const color =
                    value >= 95 ? 'text-green-400' :
                    value >= 85 ? 'text-blue-400' :
                    value >= 75 ? 'text-yellow-400' : 'text-red-400';

                  return (
                    <td
                      key={fw}
                      className={`text-center px-4 py-3 font-mono ${
                        isHighlighted ? 'bg-gray-600' : ''
                      } ${color}`}
                    >
                      {value}%
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-bold text-gray-300 mb-3">各硬件峰值性能</h3>
          <div className="space-y-3">
            {hardware.map((hw) => {
              const avg = frameworks.reduce((sum, fw) => {
                return sum + (benchmarkData[selectedOperator]?.[hw]?.[fw] || 0);
              }, 0) / frameworks.length;

              return (
                <div key={hw}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-400">{hw}</span>
                    <span className="text-gray-300">{avg.toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2 bg-gray-700 rounded-full">
                    <div
                      className="h-full bg-blue-500 rounded-full"
                      style={{ width: `${avg}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-bold text-gray-300 mb-3">框架综合表现</h3>
          <div className="space-y-3">
            {frameworks.map((fw) => {
              const avg = hardware.reduce((sum, hw) => {
                return sum + (benchmarkData[selectedOperator]?.[hw]?.[fw] || 0);
              }, 0) / hardware.length;

              return (
                <div key={fw}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-400">{fw}</span>
                    <span className="text-gray-300">{avg.toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2 bg-gray-700 rounded-full">
                    <div
                      className="h-full bg-green-500 rounded-full"
                      style={{ width: `${avg}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
