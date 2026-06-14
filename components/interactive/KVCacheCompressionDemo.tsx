'use client';

import { useState } from 'react';

const configs = [
  { name: 'MHA (GPT-2)', heads: 16, headDim: 64, kvDim: 1024, cachePerToken: 128, color: '#EF4444' },
  { name: 'GQA (LLaMA-70B)', heads: 64, headDim: 128, kvDim: 8192, cachePerToken: 256, color: '#F59E0B' },
  { name: 'MLA (DeepSeek-V2)', heads: 128, headDim: 128, kvDim: 512, cachePerToken: 64, color: '#3B82F6' },
  { name: 'MLA (优化后)', heads: 128, headDim: 128, kvDim: 256, cachePerToken: 32, color: '#10B981' },
];

const seqLen = 32;
const maxCache = 256;

export default function KVCacheCompressionDemo() {
  const [showSeq, setShowSeq] = useState(true);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">KV Cache 压缩对比</h2>
      <p className="text-sm text-gray-400 mb-4">序列长度 32K · FP16 精度 · 每 token 缓存大小</p>

      <div className="space-y-4 mb-6">
        {configs.map((c, i) => {
          const barWidth = (c.cachePerToken / maxCache) * 100;
          const totalMB = (c.cachePerToken * 32768 / 1024 / 1024).toFixed(0);
          return (
            <div key={i}>
              <div className="flex items-center gap-3 mb-1">
                <span className="w-40 text-sm font-bold" style={{ color: c.color }}>{c.name}</span>
                <span className="text-xs text-gray-400">{c.cachePerToken}B/token</span>
                <span className="text-xs text-gray-500">= {totalMB}MB (32K seq)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex-1 bg-gray-800 rounded-full h-5 relative">
                  <div className="h-full rounded-full transition-all duration-500"
                    style={{ width: `${Math.max(barWidth, 3)}%`, backgroundColor: c.color, opacity: 0.8 }} />
                </div>
                <span className="w-12 text-xs text-right font-mono" style={{ color: c.color }}>
                  {(c.cachePerToken / configs[0].cachePerToken * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Visual blocks */}
      <div className="flex gap-4 mb-4">
        {showSeq && configs.slice(0, 3).map((c, i) => (
          <div key={i} className="flex-1">
            <div className="text-xs text-gray-400 mb-1">{c.name} 缓存布局</div>
            <div className="bg-black rounded p-2 flex flex-wrap gap-0.5">
              {Array.from({ length: Math.ceil(c.cachePerToken / 4) }, (_, j) => (
                <div key={j} className="rounded-sm" style={{
                  width: `${Math.max(100 / Math.ceil(c.cachePerToken / 4), 2)}%`,
                  height: '8px',
                  backgroundColor: c.color,
                  opacity: 0.3 + (j / Math.ceil(c.cachePerToken / 4)) * 0.7,
                }} />
              ))}
            </div>
            <div className="text-[10px] text-gray-500 mt-1 text-center">{c.cachePerToken} bytes per token</div>
          </div>
        ))}
      </div>

      <div className="bg-gray-800 rounded-lg p-3 text-xs text-gray-300">
        <b className="text-white">MLA 压缩效果：</b>
        相比 MHA，MLA 将 KV Cache 压缩到 <span className="text-blue-400 font-bold">25%</span>（4×），
        相比 GQA 压缩到 <span className="text-green-400 font-bold">50%</span>（2×），
        在保持注意力质量的同时大幅提升推理吞吐量。
      </div>
    </div>
  );
}
