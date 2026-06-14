'use client';

import { useState } from 'react';

const concepts = [
  {
    name: 'Buffer',
    color: '#3B82F6',
    desc: '逻辑内存视图，描述数据形状和类型',
    props: ['shape', 'dtype', 'scope'],
    example: 'T.Buffer((M, N), "float32", scope="shared")',
  },
  {
    name: 'Region',
    color: '#10B981',
    desc: '物理内存区域，管理分配和生命周期',
    props: ['bytes', 'alignment', 'storage'],
    example: 'T.Region(offset=0, size=4096, align=128)',
  },
];

export default function BufferRegionDiagram() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-4">TVM Buffer 与 Region 抽象</h2>

      <div className="flex gap-4 mb-6">
        {concepts.map((c, i) => (
          <button key={i} onClick={() => setSelected(i)}
            className={`flex-1 p-3 rounded-lg border-2 text-left transition-all ${
              selected === i ? 'opacity-100' : 'opacity-50'
            }`} style={{ borderColor: c.color, backgroundColor: `${c.color}10` }}>
            <div className="font-bold text-sm" style={{ color: c.color }}>{c.name}</div>
            <div className="text-xs text-gray-400 mt-1">{c.desc}</div>
          </button>
        ))}
      </div>

      <div className="flex gap-4 mb-4">
        <div className="flex-1">
          <svg viewBox="0 0 500 200" className="w-full bg-black rounded-lg">
            {selected === 0 ? (
              <>
                {/* Buffer diagram */}
                <text x="10" y="20" fill="#3B82F6" fontSize="12" fontWeight="bold">Buffer 视图</text>
                <rect x="20" y="35" width="460" height="150" rx="6" fill="none" stroke="#374151" strokeDasharray="4,2" />
                
                {/* Buffer shape */}
                <g>
                  <rect x="30" y="45" width="120" height="60" rx="4" fill="#3B82F615" stroke="#3B82F6" strokeWidth="1.5" />
                  <text x="90" y="70" fill="#60A5FA" fontSize="10" textAnchor="middle">Buffer A</text>
                  <text x="90" y="85" fill="#9CA3AF" fontSize="8" textAnchor="middle">(M, N) float32</text>
                </g>

                <g>
                  <rect x="170" y="45" width="120" height="60" rx="4" fill="#10B98115" stroke="#10B981" strokeWidth="1.5" />
                  <text x="230" y="70" fill="#34D399" fontSize="10" textAnchor="middle">Buffer B</text>
                  <text x="230" y="85" fill="#9CA3AF" fontSize="8" textAnchor="middle">(N, K) float32</text>
                </g>

                <g>
                  <rect x="310" y="45" width="120" height="60" rx="4" fill="#F59E0B15" stroke="#F59E0B" strokeWidth="1.5" />
                  <text x="370" y="70" fill="#FBBF24" fontSize="10" textAnchor="middle">Buffer C</text>
                  <text x="370" y="85" fill="#9CA3AF" fontSize="8" textAnchor="middle">(M, K) float32</text>
                </g>

                {/* Arrows */}
                <path d="M150 75 L170 75" stroke="#4B5563" strokeWidth="1" markerEnd="url(#buf-arr)" />
                <path d="M290 75 L310 75" stroke="#4B5563" strokeWidth="1" markerEnd="url(#buf-arr)" />

                {/* Data access pattern */}
                <text x="30" y="130" fill="#6B7280" fontSize="9">访问模式：</text>
                <text x="30" y="145" fill="#60A5FA" fontSize="9">A[i, k]</text>
                <text x="100" y="145" fill="#34D399" fontSize="9">B[k, j]</text>
                <text x="170" y="145" fill="#FBBF24" fontSize="9">C[i, j]</text>
                <text x="30" y="165" fill="#6B7280" fontSize="8">Buffer 是逻辑视图，自动处理边界检查和索引映射</text>

                <defs>
                  <marker id="buf-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#4B5563" />
                  </marker>
                </defs>
              </>
            ) : (
              <>
                {/* Region diagram */}
                <text x="10" y="20" fill="#10B981" fontSize="12" fontWeight="bold">Region 物理布局</text>
                
                {/* Memory regions */}
                <rect x="20" y="40" width="460" height="30" rx="3" fill="#10B98115" stroke="#10B981" strokeWidth="1" />
                <text x="30" y="58" fill="#34D399" fontSize="9">Global Memory (80GB)</text>

                <rect x="20" y="80" width="200" height="25" rx="3" fill="#3B82F615" stroke="#3B82F6" strokeWidth="1" />
                <text x="30" y="96" fill="#60A5FA" fontSize="8">Region 0: Shared Mem (192KB)</text>

                <rect x="230" y="80" width="120" height="25" rx="3" fill="#F59E0B15" stroke="#F59E0B" strokeWidth="1" />
                <text x="240" y="96" fill="#FBBF24" fontSize="8">Region 1: Const (16KB)</text>

                {/* Buffers inside regions */}
                <rect x="25" y="120" width="80" height="20" rx="2" fill="#3B82F630" stroke="#3B82F6" strokeWidth="1" />
                <text x="65" y="134" fill="#60A5FA" fontSize="7" textAnchor="middle">Buffer A</text>

                <rect x="110" y="120" width="80" height="20" rx="2" fill="#10B98130" stroke="#10B981" strokeWidth="1" />
                <text x="150" y="134" fill="#34D399" fontSize="7" textAnchor="middle">Buffer B</text>

                <rect x="235" y="120" width="60" height="20" rx="2" fill="#8B5CF630" stroke="#8B5CF6" strokeWidth="1" />
                <text x="265" y="134" fill="#C084FC" fontSize="7" textAnchor="middle">Scale</text>

                <text x="30" y="165" fill="#6B7280" fontSize="8">Region 管理物理内存分配，Buffer 在 Region 内分配</text>
                <text x="30" y="180" fill="#6B7280" fontSize="8">一个 Region 可包含多个 Buffer，共享内存对齐和生命周期</text>
              </>
            )}
          </svg>
        </div>

        <div className="w-52 space-y-2 text-xs">
          <div className="bg-gray-800 rounded p-3">
            <div className="font-bold mb-2" style={{ color: concepts[selected].color }}>
              {concepts[selected].name} 属性
            </div>
            {concepts[selected].props.map((p, i) => (
              <div key={i} className="flex justify-between py-0.5">
                <span className="text-gray-400">{p}</span>
                <span className="text-gray-300">...</span>
              </div>
            ))}
          </div>
          <div className="bg-black rounded p-2 font-mono text-[10px] text-gray-400">
            {concepts[selected].example}
          </div>
        </div>
      </div>
    </div>
  );
}
