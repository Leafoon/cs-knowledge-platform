'use client';

import { useState } from 'react';

const steps = [
  { label: '输入 x', color: '#6B7280', icon: '📥' },
  { label: '低秩投影 W_UK', color: '#EF4444', icon: '🔧' },
  { label: 'KV Cache 存储', color: '#F59E0B', icon: '💾' },
  { label: 'FlashMLA Kernel', color: '#3B82F6', icon: '⚡' },
  { label: '输出 O', color: '#10B981', icon: '📤' },
];

export default function FlashMLAImplementationFlow() {
  const [activeStep, setActiveStep] = useState(0);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">FlashMLA 实现流程</h2>
      <p className="text-sm text-gray-400 mb-6">TileLang 编写的 MLA 注意力内核执行流程</p>

      <div className="flex items-center justify-between mb-6 overflow-x-auto px-2">
        {steps.map((s, i) => (
          <div key={i} className="flex items-center">
            <div className="flex flex-col items-center cursor-pointer" onClick={() => setActiveStep(i)}>
              <div className={`w-14 h-14 rounded-xl flex items-center justify-center text-xl transition-all ${
                i <= activeStep ? 'scale-110' : 'opacity-40 grayscale'
              }`} style={{
                backgroundColor: `${s.color}20`,
                border: `2px solid ${i <= activeStep ? s.color : '#374151'}`,
                boxShadow: i === activeStep ? `0 0 20px ${s.color}40` : 'none',
              }}>
                {s.icon}
              </div>
              <span className={`text-xs mt-2 font-bold ${i <= activeStep ? 'opacity-100' : 'opacity-40'}`}
                style={{ color: s.color }}>{s.label}</span>
            </div>
            {i < steps.length - 1 && (
              <div className={`w-12 h-0.5 mx-1 ${i < activeStep ? 'bg-blue-500' : 'bg-gray-700'}`} />
            )}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="text-sm font-bold text-blue-400 mb-2">TileLang Kernel 伪代码</div>
          <pre className="text-xs text-gray-300 font-mono leading-5">
{`@T.prim_func
def flash_mla_kernel(
  Q: T.Buffer, cKV: T.Buffer,
  O: T.Buffer
):
  with T.block("block"):
    for ko in T.serial(N // BLOCK):
      # 从 cache 加载压缩 KV
      c = cKV[head, ko]
      # 低秩解压 → K, V
      K = T.call_extern("decode", c)
      V = T.call_extern("decode_v", c)
      # 在线 softmax
      m, l, acc = flash_attn(
        Q, K, V, m, l, acc
      )
    O[head, seq] = acc / l`}
          </pre>
        </div>
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="text-sm font-bold text-green-400 mb-2">FlashMLA 优化点</div>
          <ul className="space-y-2 text-xs text-gray-300">
            <li className="flex gap-2">
              <span className="text-green-400">✓</span>
              <span>低秩 KV 复用：c_KV 只读一次，解压为 K/V</span>
            </li>
            <li className="flex gap-2">
              <span className="text-green-400">✓</span>
              <span>CPU-GPU 异步：预取下一块 c_KV</span>
            </li>
            <li className="flex gap-2">
              <span className="text-green-400">✓</span>
              <span>在线 Softmax 避免 O(N²) 存储</span>
            </li>
            <li className="flex gap-2">
              <span className="text-green-400">✓</span>
              <span>TileLang 自动分块和寄存器分配</span>
            </li>
          </ul>
          <div className="mt-3 p-2 bg-green-900/20 rounded text-xs text-green-300">
            相比手工 CUDA 实现，TileLang 代码量减少 <b>10×</b>
          </div>
        </div>
      </div>
    </div>
  );
}
