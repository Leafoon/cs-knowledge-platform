'use client';

import { useState } from 'react';

const stages = [
  { label: '全局内存', color: '#EF4444', items: ['src[0..63]', 'src[64..127]'] },
  { label: 'Async CP', color: '#F59E0B', items: ['cp.async', 'cp.async'] },
  { label: '共享内存', color: '#10B981', items: ['smem[0..63]', 'smem[64..127]'] },
  { label: '寄存器', color: '#3B82F6', items: ['reg FragA', 'reg FragB'] },
  { label: '计算单元', color: '#8B5CF6', items: ['WMMA/MMA'] },
];

export default function AsyncMemcpyFlow() {
  const [activeStage, setActiveStage] = useState(0);
  const [running, setRunning] = useState(false);

  const runDemo = () => {
    setRunning(true);
    let i = 0;
    const interval = setInterval(() => {
      setActiveStage(i);
      i++;
      if (i >= stages.length) {
        clearInterval(interval);
        setTimeout(() => { setRunning(false); setActiveStage(0); }, 800);
      }
    }, 600);
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">异步内存拷贝流程</h2>
        <button onClick={runDemo} disabled={running}
          className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-500 disabled:opacity-50 text-sm">
          {running ? '运行中...' : '播放动画'}
        </button>
      </div>

      <div className="flex items-center gap-1 mb-6 overflow-x-auto pb-2">
        {stages.map((s, i) => (
          <div key={i} className="flex items-center">
            <div className={`flex flex-col items-center p-3 rounded-lg border transition-all min-w-[110px] ${
              i <= activeStage && running ? 'opacity-100 scale-105' : 'opacity-60'
            }`} style={{ borderColor: s.color, backgroundColor: `${s.color}15` }}>
              <div className="w-10 h-10 rounded-full flex items-center justify-center text-lg mb-1"
                style={{ backgroundColor: `${s.color}30` }}>
                {i === 0 ? '💾' : i === 1 ? '⚡' : i === 2 ? '🧩' : i === 3 ? '📦' : '⚙️'}
              </div>
              <span className="text-xs font-bold" style={{ color: s.color }}>{s.label}</span>
              <div className="mt-1 space-y-0.5">
                {s.items.map((item, j) => (
                  <div key={j} className="text-[10px] bg-black/30 rounded px-1.5 py-0.5 font-mono">{item}</div>
                ))}
              </div>
            </div>
            {i < stages.length - 1 && (
              <div className="flex flex-col items-center px-1">
                <svg width="30" height="20" className={`transition-opacity ${i < activeStage && running ? 'opacity-100' : 'opacity-30'}`}>
                  <path d="M2 10 L25 10" stroke={stages[i + 1].color} strokeWidth="2" fill="none"
                    strokeDasharray={i === 1 ? '4,3' : '0'} />
                  <polygon points="25,6 30,10 25,14" fill={stages[i + 1].color} />
                </svg>
                {i === 1 && <span className="text-[9px] text-yellow-400">cp.async</span>}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="bg-gray-800 rounded-lg p-4 text-sm space-y-2">
        <div className="font-bold text-yellow-400">cp.async 关键特性</div>
        <ul className="space-y-1 text-gray-300">
          <li>• 全局内存→共享内存拷贝 <b>不占用寄存器</b></li>
          <li>• 可与计算指令 <b>重叠执行</b>（计算隐藏传输延迟）</li>
          <li>• 支持 4/8/16 字节对齐粒度</li>
          <li>• 流水线中 Prologue 加载下一 tile，Main Loop 同时计算当前 tile</li>
        </ul>
      </div>
    </div>
  );
}
