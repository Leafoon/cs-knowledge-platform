'use client';

import { useState } from 'react';

const components = [
  { name: 'SM', desc: '流多处理器', color: 'bg-blue-500', details: '执行线程块，包含CUDA核心和特殊功能单元' },
  { name: 'Warp', desc: '线程束', color: 'bg-green-500', details: '32个线程同步执行，SIMT执行模型' },
  { name: 'Shared Memory', desc: '共享内存', color: 'bg-purple-500', details: 'SM内低延迟存储，程序员显式管理' },
  { name: 'Register File', desc: '寄存器文件', color: 'bg-yellow-500', details: '最快存储，每个线程私有' },
  { name: 'Global Memory', desc: '全局内存', color: 'bg-red-500', details: '所有线程可访问，高延迟高带宽' },
  { name: 'L2 Cache', desc: 'L2缓存', color: 'bg-orange-500', details: '全局内存的缓存层，硬件管理' },
];

export function GPUArchitectureOverview() {
  const [selected, setSelected] = useState<number | null>(null);

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">GPU架构概览</h2>
      
      <div className="relative p-4 bg-gray-50 rounded-xl min-h-[300px]">
        <div className="absolute inset-4 border-2 border-dashed border-gray-300 rounded-xl p-4">
          <div className="text-xs text-gray-400 mb-2">GPU芯片</div>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="border-2 border-blue-200 rounded-lg p-3">
              <div className="text-xs text-blue-400 mb-2">SM 0</div>
              <div className="grid grid-cols-2 gap-2">
                {components.slice(0, 4).map((c, i) => (
                  <div
                    key={i}
                    onClick={() => setSelected(selected === i ? null : i)}
                    className={`p-2 rounded cursor-pointer transition-all ${
                      c.color} text-white text-xs ${selected === i ? 'ring-2 ring-offset-2 ring-gray-400' : ''
                    }`}
                  >
                    {c.name}
                  </div>
                ))}
              </div>
            </div>
            
            <div className="border-2 border-blue-200 rounded-lg p-3">
              <div className="text-xs text-blue-400 mb-2">SM 1</div>
              <div className="grid grid-cols-2 gap-2">
                {components.slice(0, 4).map((c, i) => (
                  <div
                    key={i}
                    className={`p-2 rounded ${c.color} text-white text-xs`}
                  >
                    {c.name}
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          <div className="mt-4 grid grid-cols-2 gap-2">
            {components.slice(4).map((c, i) => (
              <div
                key={i}
                onClick={() => setSelected(selected === i + 4 ? null : i + 4)}
                className={`p-3 rounded cursor-pointer transition-all ${
                  c.color} text-white ${selected === i + 4 ? 'ring-2 ring-offset-2 ring-gray-400' : ''
                }`}
              >
                <div className="font-medium">{c.name}</div>
                <div className="text-xs opacity-80">{c.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {selected !== null && (
        <div className="mt-4 p-4 bg-blue-50 rounded-xl">
          <div className="font-semibold text-blue-800">{components[selected].name}</div>
          <div className="text-sm text-blue-600">{components[selected].details}</div>
        </div>
      )}
    </div>
  );
}