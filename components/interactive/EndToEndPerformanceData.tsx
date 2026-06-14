'use client';

import { useState } from 'react';

const models = [
  { name: 'DeepSeek-V3 671B', params: '671B (37B活跃)', variant: 'MoE' },
  { name: 'DeepSeek-V2 236B', params: '236B (21B活跃)', variant: 'MoE' },
  { name: 'LLaMA-3 70B', params: '70B', variant: 'Dense' },
  { name: 'Qwen-2.5 72B', params: '72B', variant: 'Dense' },
];

const metrics = [
  { key: 'ttft', label: '首 Token 延迟 (TTFT)', unit: 'ms', lower: true },
  { key: 'tpot', label: '每 Token 延迟 (TPOT)', unit: 'ms', lower: true },
  { key: 'throughput', label: '吞吐量', unit: 'tokens/s', lower: false },
  { key: 'memory', label: '显存占用', unit: 'GB', lower: true },
];

const performanceData: Record<string, Record<string, number>> = {
  'DeepSeek-V3 671B': { ttft: 120, tpot: 18, throughput: 45, memory: 380 },
  'DeepSeek-V2 236B': { ttft: 85, tpot: 12, throughput: 68, memory: 180 },
  'LLaMA-3 70B': { ttft: 95, tpot: 22, throughput: 38, memory: 140 },
  'Qwen-2.5 72B': { ttft: 90, tpot: 20, throughput: 42, memory: 145 },
};

export default function EndToEndPerformanceData() {
  const [selectedMetric, setSelectedMetric] = useState<string>('throughput');
  const [selectedModel, setSelectedModel] = useState<string>('DeepSeek-V3 671B');

  const currentMetric = metrics.find(m => m.key === selectedMetric)!;
  const allValues = models.map(m => performanceData[m.name][selectedMetric]);
  const maxValue = Math.max(...allValues);
  const minValue = Math.min(...allValues);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">端到端推理性能</h2>
      <p className="text-gray-400 text-sm mb-4">DeepSeek-V3 及其他模型的推理延迟、吞吐量和显存对比</p>

      <div className="flex gap-4 mb-6">
        <div>
          <label className="text-xs text-gray-400 block mb-1">性能指标</label>
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            className="bg-gray-800 text-white text-sm rounded px-3 py-2 border border-gray-600"
          >
            {metrics.map((m) => (
              <option key={m.key} value={m.key}>{m.label}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">对比模型</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="bg-gray-800 text-white text-sm rounded px-3 py-2 border border-gray-600"
          >
            {models.map((m) => (
              <option key={m.name} value={m.name}>{m.name}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-4 mb-6">
        <h3 className="text-sm font-bold text-gray-300 mb-4">{currentMetric.label}</h3>
        <div className="space-y-4">
          {models.map((model) => {
            const value = performanceData[model.name][selectedMetric];
            const isPositive = currentMetric.lower
              ? value === minValue
              : value === maxValue;
            const barWidth = currentMetric.lower
              ? ((maxValue - value) / (maxValue - minValue)) * 80 + 20
              : ((value - minValue) / (maxValue - minValue)) * 80 + 20;

            return (
              <div
                key={model.name}
                className={`p-3 rounded-lg cursor-pointer transition-all ${
                  selectedModel === model.name ? 'bg-gray-700 ring-1 ring-blue-500' : 'hover:bg-gray-750'
                }`}
                onClick={() => setSelectedModel(model.name)}
              >
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <span className="text-sm font-medium text-white">{model.name}</span>
                    <span className="text-xs text-gray-500 ml-2">{model.params}</span>
                  </div>
                  <span className={`text-lg font-bold ${isPositive ? 'text-green-400' : 'text-gray-300'}`}>
                    {value} {currentMetric.unit}
                  </span>
                </div>
                <div className="w-full h-2 bg-gray-700 rounded-full">
                  <div
                    className={`h-full rounded-full transition-all ${
                      isPositive ? 'bg-green-500' : 'bg-blue-500'
                    }`}
                    style={{ width: `${barWidth}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-bold text-gray-300 mb-3">详细性能数据</h3>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-500 border-b border-gray-700">
              <th className="text-left py-2">模型</th>
              {metrics.map((m) => (
                <th key={m.key} className="text-right py-2">{m.label.split('(')[0].trim()}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr
                key={model.name}
                className={`border-b border-gray-700/50 cursor-pointer ${
                  selectedModel === model.name ? 'bg-gray-700' : 'hover:bg-gray-750'
                }`}
                onClick={() => setSelectedModel(model.name)}
              >
                <td className="py-2 text-white">
                  {model.name}
                  <span className="text-gray-500 ml-1">({model.variant})</span>
                </td>
                {metrics.map((m) => {
                  const value = performanceData[model.name][m.key];
                  return (
                    <td key={m.key} className="text-right text-gray-300 font-mono">
                      {value}{m.unit}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
