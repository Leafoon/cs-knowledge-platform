'use client';

import { useState } from 'react';

const timeline = [
  {
    date: '2024 Q1',
    title: '项目启动',
    color: 'green',
    events: ['核心 IR 设计', '基础调度器', 'CUDA 后端'],
    milestone: 'v0.1',
  },
  {
    date: '2024 Q2',
    title: '核心功能',
    color: 'blue',
    events: ['GEMM 优化', 'Flash Attention', '内存管理'],
    milestone: 'v0.5',
  },
  {
    date: '2024 Q3',
    title: '生态扩展',
    color: 'purple',
    events: ['AMD ROCm 支持', 'PyTorch 集成', '文档完善'],
    milestone: 'v0.8',
  },
  {
    date: '2024 Q4',
    title: '社区建设',
    color: 'yellow',
    events: ['开源发布', '社区贡献', '教程系列'],
    milestone: 'v1.0',
  },
  {
    date: '2025 Q1',
    title: '企业支持',
    color: 'orange',
    events: ['昇腾适配', '企业版功能', '商业支持'],
    milestone: 'v1.5',
  },
  {
    date: '2025+',
    title: '未来发展',
    color: 'red',
    events: ['自动调优', '多语言绑定', '云端部署'],
    milestone: 'v2.0',
  },
];

const stats = [
  { label: 'GitHub Stars', value: '12.5k', change: '+2.3k/月' },
  { label: 'Contributors', value: '286', change: '+45/月' },
  { label: '下载量', value: '850k', change: '+120k/月' },
  { label: '企业用户', value: '50+', change: '+8/月' },
];

export default function EcosystemEvolutionDiagram() {
  const [selectedEvent, setSelectedEvent] = useState<number>(0);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">生态系统演进图</h2>
      <p className="text-gray-400 text-sm mb-4">TileLang 生态系统的成长历程与未来规划</p>

      <div className="grid grid-cols-4 gap-4 mb-8">
        {stats.map((stat, idx) => (
          <div key={idx} className="bg-gray-800 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-white">{stat.value}</div>
            <div className="text-xs text-gray-400">{stat.label}</div>
            <div className="text-xs text-green-400 mt-1">{stat.change}</div>
          </div>
        ))}
      </div>

      <div className="relative mb-8">
        <div className="absolute top-1/2 left-0 right-0 h-1 bg-gradient-to-r from-green-500 via-blue-500 to-purple-500 -translate-y-1/2 rounded" />

        <div className="relative flex justify-between">
          {timeline.map((event, idx) => {
            const isSelected = selectedEvent === idx;
            return (
              <div
                key={idx}
                className="flex flex-col items-center cursor-pointer"
                onClick={() => setSelectedEvent(idx)}
              >
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center text-xs font-bold z-10 transition-all ${
                    idx <= selectedEvent
                      ? `bg-${event.color}-500`
                      : 'bg-gray-700'
                  } ${isSelected ? 'ring-4 ring-white/30 scale-110' : ''}`}
                >
                  {event.milestone}
                </div>
                <div className="mt-3 text-center">
                  <div className={`text-xs font-bold ${isSelected ? 'text-white' : 'text-gray-400'}`}>
                    {event.title}
                  </div>
                  <div className="text-[10px] text-gray-500">{event.date}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-4">
          <span className={`w-3 h-3 rounded-full bg-${timeline[selectedEvent].color}-500`} />
          <h3 className="text-lg font-bold">{timeline[selectedEvent].title}</h3>
          <span className="text-xs text-gray-500">{timeline[selectedEvent].date}</span>
        </div>

        <div className="grid grid-cols-3 gap-4">
          {timeline[selectedEvent].events.map((event, idx) => (
            <div
              key={idx}
              className={`bg-${timeline[selectedEvent].color}-900/30 rounded-lg p-4 border border-${timeline[selectedEvent].color}-700/50`}
            >
              <div className="text-sm font-medium text-white">{event}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-6 bg-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-bold text-gray-300 mb-3">发展历程</h3>
        <div className="grid grid-cols-6 gap-2">
          {timeline.map((event, idx) => (
            <div
              key={idx}
              className={`p-2 rounded text-center cursor-pointer transition-all ${
                idx === selectedEvent
                  ? `bg-${event.color}-600`
                  : idx < selectedEvent
                  ? 'bg-gray-700'
                  : 'bg-gray-800'
              }`}
              onClick={() => setSelectedEvent(idx)}
            >
              <div className="text-[10px] text-gray-400">{event.date}</div>
              <div className="text-xs font-bold">{event.milestone}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
