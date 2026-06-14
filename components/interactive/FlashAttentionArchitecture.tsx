'use client';

import { useState } from 'react';

const blocks = [
  { name: 'Q Tile', color: '#3B82F6', size: '128×64', loc: 'SRAM', desc: '查询矩阵分块' },
  { name: 'K Tile', color: '#EF4444', size: '128×64', loc: 'SRAM', desc: '键矩阵分块' },
  { name: 'V Tile', color: '#10B981', size: '128×64', loc: 'SRAM', desc: '值矩阵分块' },
  { name: 'S = QK^T', color: '#F59E0B', size: '128×128', loc: 'SRAM', desc: '注意力分数' },
  { name: 'Online Softmax', color: '#8B5CF6', size: '128', loc: 'SRAM', desc: '在线归一化' },
  { name: 'O Acc', color: '#EC4899', size: '128×64', loc: 'SRAM', desc: '输出累加器' },
];

export default function FlashAttentionArchitecture() {
  const [activeBlock, setActiveBlock] = useState<number | null>(null);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-4">FlashAttention 架构</h2>

      <div className="flex flex-col items-center gap-2 mb-6">
        {/* Input tensors */}
        <div className="flex gap-3">
          {blocks.slice(0, 3).map((b, i) => (
            <div key={i}
              className={`p-3 rounded-lg border-2 cursor-pointer transition-all ${
                activeBlock === i ? 'scale-105' : 'opacity-70'
              }`}
              style={{ borderColor: b.color, backgroundColor: `${b.color}15` }}
              onClick={() => setActiveBlock(activeBlock === i ? null : i)}>
              <div className="text-xs font-bold" style={{ color: b.color }}>{b.name}</div>
              <div className="text-[10px] text-gray-400">{b.size} · {b.loc}</div>
              {activeBlock === i && <div className="text-[10px] text-gray-300 mt-1">{b.desc}</div>}
            </div>
          ))}
        </div>

        {/* Arrow */}
        <svg width="200" height="30">
          <path d="M100 0 L100 20" stroke="#4B5563" strokeWidth="2" markerEnd="url(#arr)" />
          <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#4B5563" />
          </marker></defs>
        </svg>

        {/* Attention computation */}
        <div className="flex gap-3 items-center">
          {blocks.slice(3, 6).map((b, i) => (
            <div key={i + 3} className="flex items-center gap-3">
              <div
                className={`p-3 rounded-lg border-2 cursor-pointer transition-all ${
                  activeBlock === i + 3 ? 'scale-105' : 'opacity-70'
                }`}
                style={{ borderColor: b.color, backgroundColor: `${b.color}15` }}
                onClick={() => setActiveBlock(activeBlock === i + 3 ? null : i + 3)}>
                <div className="text-xs font-bold" style={{ color: b.color }}>{b.name}</div>
                <div className="text-[10px] text-gray-400">{b.size} · {b.loc}</div>
                {activeBlock === i + 3 && <div className="text-[10px] text-gray-300 mt-1">{b.desc}</div>}
              </div>
              {i < 2 && (
                <svg width="20" height="20">
                  <path d="M0 10 L15 10" stroke="#4B5563" strokeWidth="1.5" markerEnd="url(#arr2)" />
                  <defs><marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#4B5563" />
                  </marker></defs>
                </svg>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-4 text-sm space-y-2">
        <div className="font-bold text-purple-400">FlashAttention 核心思想</div>
        <ul className="space-y-1 text-gray-300 text-xs">
          <li>• <b>分块计算</b>：Q/K/V 分块加载到 SRAM，避免 O(N²) HBM 访问</li>
          <li>• <b>在线 Softmax</b>：逐块更新 running max/sum，无需全量 S 矩阵</li>
          <li>• <b>重计算</b>：反向传播时重算 S/P，节省内存 O(N) 而非 O(N²)</li>
          <li>• <b>融合内核</b>：Flash/Softmax/Output 三步融合为单个 CUDA kernel</li>
        </ul>
      </div>
    </div>
  );
}
