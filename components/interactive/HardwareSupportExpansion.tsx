'use client';

import { useState } from 'react';

const hardwareTimeline = [
  {
    phase: 'Phase 1',
    status: 'current',
    title: 'NVIDIA GPU',
    color: 'green',
    devices: ['A100', 'H100', 'H200', 'L40S'],
    features: ['CUDA 后端', 'FP16/BF16', 'Flash Attention', 'MoE 支持'],
    date: '2024 Q1',
  },
  {
    phase: 'Phase 2',
    status: 'active',
    title: 'AMD GPU',
    color: 'blue',
    devices: ['MI250X', 'MI300X', 'MI350'],
    features: ['ROCm 后端', 'CDNA 架构', 'HBM3 支持', '初步适配'],
    date: '2024 Q3',
  },
  {
    phase: 'Phase 3',
    status: 'planned',
    title: '华为昇腾',
    color: 'purple',
    devices: ['Ascend 910B', 'Ascend 910C'],
    features: ['CANN 后端', '达芬奇架构', 'HCCS 互联', '适配中'],
    date: '2025 Q1',
  },
  {
    phase: 'Phase 4',
    status: 'future',
    title: '更多硬件',
    color: 'yellow',
    devices: ['Intel Gaudi', 'Custom ASIC', 'RISC-V GPU'],
    features: ['统一接口', '插件化后端', '自动适配', '规划中'],
    date: '2025+',
  },
];

export default function HardwareSupportExpansion() {
  const [selectedPhase, setSelectedPhase] = useState<number>(0);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">硬件支持扩展路线图</h2>
      <p className="text-gray-400 text-sm mb-4">从 NVIDIA → AMD → Ascend → 未来的硬件支持规划</p>

      <div className="relative mb-8">
        <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-gray-700 -translate-y-1/2" />

        <div className="relative flex justify-between">
          {hardwareTimeline.map((hw, idx) => {
            const isSelected = selectedPhase === idx;
            return (
              <div
                key={idx}
                className="flex flex-col items-center cursor-pointer"
                onClick={() => setSelectedPhase(idx)}
              >
                <div
                  className={`w-12 h-12 rounded-full flex items-center justify-center text-lg z-10 transition-all ${
                    hw.status === 'current'
                      ? 'bg-green-500 ring-4 ring-green-400/30'
                      : hw.status === 'active'
                      ? 'bg-blue-500 ring-2 ring-blue-400/30'
                      : hw.status === 'planned'
                      ? 'bg-purple-500'
                      : 'bg-gray-600'
                  } ${isSelected ? 'scale-110' : ''}`}
                >
                  {idx + 1}
                </div>
                <div className="mt-3 text-center">
                  <div className={`text-sm font-bold ${
                    isSelected ? 'text-white' : 'text-gray-400'
                  }`}>
                    {hw.title}
                  </div>
                  <div className="text-xs text-gray-500">{hw.date}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-4">
          <span className={`w-3 h-3 rounded-full ${
            hardwareTimeline[selectedPhase].status === 'current' ? 'bg-green-500' :
            hardwareTimeline[selectedPhase].status === 'active' ? 'bg-blue-500' :
            hardwareTimeline[selectedPhase].status === 'planned' ? 'bg-purple-500' : 'bg-gray-500'
          }`} />
          <h3 className="text-lg font-bold">{hardwareTimeline[selectedPhase].title}</h3>
          <span className="text-xs text-gray-500 ml-2">{hardwareTimeline[selectedPhase].phase}</span>
          <span className={`text-xs px-2 py-0.5 rounded ml-2 ${
            hardwareTimeline[selectedPhase].status === 'current' ? 'bg-green-900/50 text-green-400' :
            hardwareTimeline[selectedPhase].status === 'active' ? 'bg-blue-900/50 text-blue-400' :
            hardwareTimeline[selectedPhase].status === 'planned' ? 'bg-purple-900/50 text-purple-400' :
            'bg-gray-700 text-gray-400'
          }`}>
            {hardwareTimeline[selectedPhase].status === 'current' ? '当前支持' :
             hardwareTimeline[selectedPhase].status === 'active' ? '开发中' :
             hardwareTimeline[selectedPhase].status === 'planned' ? '计划中' : '未来规划'}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <h4 className="text-sm font-bold text-gray-300 mb-3">支持设备</h4>
            <div className="flex flex-wrap gap-2">
              {hardwareTimeline[selectedPhase].devices.map((device, idx) => (
                <span
                  key={idx}
                  className="px-3 py-1.5 bg-gray-700 rounded text-xs text-gray-300"
                >
                  {device}
                </span>
              ))}
            </div>
          </div>

          <div>
            <h4 className="text-sm font-bold text-gray-300 mb-3">主要特性</h4>
            <div className="space-y-2">
              {hardwareTimeline[selectedPhase].features.map((feature, idx) => (
                <div key={idx} className="flex items-center gap-2 text-xs text-gray-400">
                  <span className="text-green-400">✓</span>
                  {feature}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-4 gap-4">
        {hardwareTimeline.map((hw, idx) => (
          <div
            key={idx}
            className={`bg-gray-800 rounded-lg p-3 cursor-pointer transition-all ${
              selectedPhase === idx ? 'ring-2 ring-blue-500' : 'hover:bg-gray-750'
            }`}
            onClick={() => setSelectedPhase(idx)}
          >
            <div className="text-xs text-gray-500 mb-1">{hw.phase}</div>
            <div className="text-sm font-bold text-white">{hw.title}</div>
            <div className="text-xs text-gray-400">{hw.date}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
