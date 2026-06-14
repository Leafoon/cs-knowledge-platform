'use client';

import { useState } from 'react';

const metrics = [
  { name: 'SM活跃周期', value: 89.3, unit: '%', status: 'good' },
  { name: '内存吞吐量', value: 67.8, unit: '%', status: 'medium' },
  { name: '计算吞吐量', value: 45.2, unit: '%', status: 'bad' },
  { name: '共享内存加载', value: 1234, unit: 'MB/s', status: 'good' },
  { name: '全局内存加载', value: 4567, unit: 'MB/s', status: 'medium' },
  { name: 'L2缓存命中率', value: 78.5, unit: '%', status: 'good' },
];

const stallReasons = [
  { reason: '内存依赖', percentage: 42, color: 'bg-red-500' },
  { reason: '执行依赖', percentage: 28, color: 'bg-yellow-500' },
  { reason: '指令获取', percentage: 15, color: 'bg-blue-500' },
  { reason: '同步等待', percentage: 10, color: 'bg-purple-500' },
  { reason: '其他', percentage: 5, color: 'bg-gray-500' },
];

export function NCUAnalysisVisualizer() {
  const [activeTab, setActiveTab] = useState<'metrics' | 'stalls'>('metrics');

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">NCU分析可视化</h2>
      
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setActiveTab('metrics')}
          className={`px-4 py-2 rounded-lg transition-colors ${
            activeTab === 'metrics' ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-600'
          }`}
        >
          性能指标
        </button>
        <button
          onClick={() => setActiveTab('stalls')}
          className={`px-4 py-2 rounded-lg transition-colors ${
            activeTab === 'stalls' ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-600'
          }`}
        >
          停顿原因
        </button>
      </div>

      {activeTab === 'metrics' ? (
        <div className="space-y-3">
          {metrics.map((m) => (
            <div key={m.name} className="flex items-center gap-4">
              <span className="w-32 text-sm text-gray-600">{m.name}</span>
              <div className="flex-1 h-6 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`h-full ${
                    m.status === 'good' ? 'bg-green-500' : m.status === 'medium' ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${Math.min(m.value, 100)}%` }}
                />
              </div>
              <span className="w-20 text-right text-sm font-mono">{m.value} {m.unit}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="space-y-2">
          {stallReasons.map((s) => (
            <div key={s.reason} className="flex items-center gap-3">
              <span className="w-24 text-sm text-gray-600">{s.reason}</span>
              <div className="flex-1 h-8 bg-gray-100 rounded overflow-hidden">
                <div className={`h-full ${s.color} flex items-center px-2`} style={{ width: `${s.percentage}%` }}>
                  <span className="text-xs text-white font-medium">{s.percentage}%</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}