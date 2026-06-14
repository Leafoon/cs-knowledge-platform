'use client';

import { useState } from 'react';

const blocks = [
  { label: 'Input x', color: '#6B7280', y: 10 },
  { label: 'W_QKV', color: '#EF4444', y: 60, desc: 'd → 低秩 d_c' },
  { label: 'c_KV (压缩)', color: '#F59E0B', y: 110, desc: '低维 KV 压缩表示' },
  { label: 'RoPE', color: '#8B5CF6', y: 160, desc: '解耦旋转位置编码' },
  { label: 'K_rope', color: '#C084FC', y: 160, x: 200, desc: 'd_rope 维度' },
  { label: 'Attention', color: '#3B82F6', y: 220 },
  { label: 'Output', color: '#10B981', y: 280 },
];

export default function MLAArchitectureDiagram() {
  const [selectedBlock, setSelectedBlock] = useState<number | null>(null);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">Multi-head Latent Attention (MLA)</h2>
      <p className="text-sm text-gray-400 mb-4">DeepSeek-V2 · 低秩 KV 投影压缩 KV Cache</p>

      <div className="flex gap-6">
        <div className="flex-1">
          <svg viewBox="0 0 400 320" className="w-full">
            {/* Input */}
            <g onClick={() => setSelectedBlock(0)} className="cursor-pointer">
              <rect x="140" y="5" width="120" height="35" rx="6" fill="#6B728030" stroke="#6B7280" strokeWidth="1.5" />
              <text x="200" y="27" fill="#9CA3AF" fontSize="11" textAnchor="middle">Input x</text>
            </g>

            <path d="M200 40 L200 55" stroke="#4B5563" strokeWidth="1.5" markerEnd="url(#mla-arr)" />

            {/* W_QKV projection */}
            <g onClick={() => setSelectedBlock(1)} className="cursor-pointer">
              <rect x="140" y="55" width="120" height="40" rx="6" fill="#EF444420" stroke="#EF4444" strokeWidth="1.5" />
              <text x="200" y="72" fill="#F87171" fontSize="11" textAnchor="middle">W_QKV 投影</text>
              <text x="200" y="87" fill="#9CA3AF" fontSize="9" textAnchor="middle">d → d_c (低秩)</text>
            </g>

            {/* Split Q and compressed KV */}
            <path d="M200 95 L120 120 M200 95 L280 120" stroke="#4B5563" strokeWidth="1.5" markerEnd="url(#mla-arr)" />

            {/* Q path */}
            <g>
              <rect x="80" y="115" width="80" height="30" rx="4" fill="#3B82F620" stroke="#3B82F6" strokeWidth="1" />
              <text x="120" y="134" fill="#60A5FA" fontSize="10" textAnchor="middle">Q</text>
            </g>

            {/* Compressed KV */}
            <g onClick={() => setSelectedBlock(2)} className="cursor-pointer">
              <rect x="230" y="115" width="120" height="30" rx="4" fill="#F59E0B20" stroke="#F59E0B" strokeWidth="1.5" />
              <text x="290" y="134" fill="#FBBF24" fontSize="10" textAnchor="middle">c_KV (压缩)</text>
            </g>

            <path d="M290 145 L290 155" stroke="#4B5563" strokeWidth="1.5" markerEnd="url(#mla-arr)" />

            {/* W_UK → K */}
            <g>
              <rect x="230" y="155" width="60" height="25" rx="4" fill="#EF444420" stroke="#EF4444" strokeWidth="1" />
              <text x="260" y="171" fill="#F87171" fontSize="9" textAnchor="middle">W_UK→K</text>
            </g>

            {/* W_UV → V */}
            <g>
              <rect x="300" y="155" width="60" height="25" rx="4" fill="#10B98120" stroke="#10B981" strokeWidth="1" />
              <text x="330" y="171" fill="#34D399" fontSize="9" textAnchor="middle">W_UV→V</text>
            </g>

            {/* RoPE path */}
            <g onClick={() => setSelectedBlock(3)} className="cursor-pointer">
              <rect x="80" y="155" width="60" height="25" rx="4" fill="#8B5CF620" stroke="#8B5CF6" strokeWidth="1.5" />
              <text x="110" y="171" fill="#C084FC" fontSize="9" textAnchor="middle">RoPE</text>
            </g>
            <g>
              <rect x="80" y="185" width="60" height="25" rx="4" fill="#C084FC20" stroke="#C084FC" strokeWidth="1" />
              <text x="110" y="201" fill="#C084FC" fontSize="9" textAnchor="middle">K_rope</text>
            </g>

            {/* KV Cache label */}
            <g>
              <rect x="370" y="115" width="25" height="65" rx="3" fill="#F59E0B15" stroke="#F59E0B" strokeWidth="1" strokeDasharray="3,2" />
              <text x="382" y="150" fill="#FBBF24" fontSize="8" textAnchor="middle" transform="rotate(-90, 382, 150)">Cache</text>
            </g>

            {/* Attention */}
            <path d="M120 210 L200 225 M260 180 L200 225 M330 180 L200 225 M110 210 L200 225"
              stroke="#4B5563" strokeWidth="1" markerEnd="url(#mla-arr)" />
            <g onClick={() => setSelectedBlock(5)} className="cursor-pointer">
              <rect x="140" y="220" width="120" height="35" rx="6" fill="#3B82F620" stroke="#3B82F6" strokeWidth="1.5" />
              <text x="200" y="242" fill="#60A5FA" fontSize="11" textAnchor="middle">Attention(Q,K,V)</text>
            </g>

            <path d="M200 255 L200 270" stroke="#4B5563" strokeWidth="1.5" markerEnd="url(#mla-arr)" />

            {/* Output */}
            <g onClick={() => setSelectedBlock(6)} className="cursor-pointer">
              <rect x="140" y="270" width="120" height="35" rx="6" fill="#10B98120" stroke="#10B981" strokeWidth="1.5" />
              <text x="200" y="292" fill="#34D399" fontSize="11" textAnchor="middle">Output</text>
            </g>

            <defs>
              <marker id="mla-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#4B5563" />
              </marker>
            </defs>
          </svg>
        </div>

        <div className="w-56 space-y-2 text-xs">
          <div className="bg-gray-800 rounded p-3">
            <div className="font-bold text-purple-400 mb-1">MLA 核心优势</div>
            <ul className="space-y-1 text-gray-400">
              <li>• KV Cache 压缩到 <span className="text-yellow-400">d_c 维</span></li>
              <li>• 低秩投影保留关键信息</li>
              <li>• RoPE 解耦保持位置编码能力</li>
              <li>• 内存 <span className="text-green-400">大幅降低</span></li>
            </ul>
          </div>
          {selectedBlock !== null && (
            <div className="bg-blue-900/30 border border-blue-700 rounded p-3">
              <div className="text-blue-400 font-bold">{blocks[selectedBlock].label}</div>
              <div className="text-gray-300">{blocks[selectedBlock].desc || '数据输入层'}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
