'use client';

import { useState } from 'react';

const comparisonData = [
  {
    feature: '自动化程度',
    tilelang: '完全自动',
    triton: '半自动',
    tvm: '完全自动',
  },
  {
    feature: '调优策略',
    tilelang: '基于规则的调度',
    triton: '运行时自适应',
    tvm: '机器学习模型',
  },
  {
    feature: '调优时间',
    tilelang: '秒级',
    triton: '分钟级',
    tvm: '小时级',
  },
  {
    feature: '性能上限',
    tilelang: '中高',
    triton: '高',
    tvm: '极高',
  },
  {
    feature: '易用性',
    tilelang: '高',
    triton: '中',
    tvm: '低',
  },
  {
    feature: '硬件支持',
    tilelang: 'NVIDIA GPU',
    triton: 'NVIDIA/AMD GPU',
    tvm: '多硬件平台',
  },
];

export function AutoScheduleVsAutotuneComparison() {
  const [selectedRow, setSelectedRow] = useState<number | null>(null);

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">自动调度 vs 自动调优 对比</h2>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="bg-gradient-to-r from-blue-500 to-purple-500 text-white">
              <th className="p-3 text-left">特性</th>
              <th className="p-3 text-left">TileLang Auto Schedule</th>
              <th className="p-3 text-left">Triton Autotune</th>
              <th className="p-3 text-left">TVM AutoTVM</th>
            </tr>
          </thead>
          <tbody>
            {comparisonData.map((row, index) => (
              <tr
                key={index}
                className={`border-b cursor-pointer transition-all ${
                  selectedRow === index
                    ? 'bg-blue-50 scale-[1.01]'
                    : 'hover:bg-gray-50'
                }`}
                onClick={() => setSelectedRow(selectedRow === index ? null : index)}
              >
                <td className="p-3 font-semibold text-gray-700">{row.feature}</td>
                <td className="p-3 text-blue-600">{row.tilelang}</td>
                <td className="p-3 text-purple-600">{row.triton}</td>
                <td className="p-3 text-green-600">{row.tvm}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-4 p-3 bg-gray-50 rounded-lg text-sm text-gray-600">
        💡 点击行查看详细说明
      </div>
    </div>
  );
}