'use client';

import { useState } from 'react';

const components = [
  {
    id: 'moe',
    name: 'MoE 层',
    x: 50,
    y: 30,
    width: 200,
    height: 100,
    color: 'bg-blue-600',
    description: '混合专家网络，256 个专家，每次激活 8 个',
    details: ['Top-8 路由', '负载均衡损失', 'Auxiliary Loss'],
  },
  {
    id: 'mla',
    name: 'MLA 层',
    x: 300,
    y: 30,
    width: 200,
    height: 100,
    color: 'bg-green-600',
    description: 'Multi-head Latent Attention，KV 压缩',
    details: ['低秩 KV 压缩', 'RoPE 位置编码', 'GQA 变体'],
  },
  {
    id: 'ffn',
    name: 'Shared Expert',
    x: 50,
    y: 180,
    width: 120,
    height: 80,
    color: 'bg-purple-600',
    description: '共享专家，处理所有 Token',
    details: ['SwiGLU', '固定权重', '基础能力'],
  },
  {
    id: 'routing',
    name: 'Router',
    x: 200,
    y: 180,
    width: 120,
    height: 80,
    color: 'bg-yellow-600',
    description: '动态路由网络，选择 Top-K 专家',
    details: ['门控网络', 'Top-K 选择', '负载均衡'],
  },
  {
    id: 'expert_pool',
    name: 'Expert Pool',
    x: 350,
    y: 180,
    width: 150,
    height: 80,
    color: 'bg-orange-600',
    description: '256 个可训练专家网络',
    details: ['SwiGLU FFN', '独立参数', '稀疏激活'],
  },
];

export default function DeepSeekArchitectureOverview() {
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [showFlow, setShowFlow] = useState(true);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">DeepSeek-V3 架构概览</h2>
      <p className="text-gray-400 text-sm mb-4">MoE + MLA + Multi-head Attention 架构可视化</p>

      <div className="flex items-center gap-4 mb-4">
        <label className="flex items-center gap-2 text-xs text-gray-400">
          <input
            type="checkbox"
            checked={showFlow}
            onChange={(e) => setShowFlow(e.target.checked)}
            className="rounded"
          />
          显示数据流
        </label>
      </div>

      <div className="relative bg-gray-800 rounded-lg h-72 mb-6">
        <svg className="w-full h-full" viewBox="0 0 600 300">
          {/* Data flow arrows */}
          {showFlow && (
            <>
              <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" fill="#6b7280" />
                </marker>
              </defs>
              <path d="M 150 130 L 150 180" stroke="#6b7280" strokeWidth="2" markerEnd="url(#arrowhead)" />
              <path d="M 260 130 L 260 180" stroke="#6b7280" strokeWidth="2" markerEnd="url(#arrowhead)" />
              <path d="M 400 130 L 400 180" stroke="#6b7280" strokeWidth="2" markerEnd="url(#arrowhead)" />
              <path d="M 320 220 L 350 220" stroke="#6b7280" strokeWidth="2" markerEnd="url(#arrowhead)" />
            </>
          )}

          {/* Components */}
          {components.map((comp) => {
            const isSelected = selectedComponent === comp.id;
            return (
              <g
                key={comp.id}
                className="cursor-pointer"
                onClick={() => setSelectedComponent(isSelected ? null : comp.id)}
              >
                <rect
                  x={comp.x}
                  y={comp.y}
                  width={comp.width}
                  height={comp.height}
                  rx="8"
                  fill={isSelected ? '#1f2937' : '#111827'}
                  stroke={isSelected ? '#facc15' : '#374151'}
                  strokeWidth={isSelected ? 2 : 1}
                />
                <text
                  x={comp.x + comp.width / 2}
                  y={comp.y + 25}
                  fill="#fff"
                  fontSize="14"
                  fontWeight="bold"
                  textAnchor="middle"
                >
                  {comp.name}
                </text>
                <text
                  x={comp.x + comp.width / 2}
                  y={comp.y + 45}
                  fill="#9ca3af"
                  fontSize="10"
                  textAnchor="middle"
                >
                  {comp.description.slice(0, 20)}...
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {selectedComponent && (
        <div className="bg-gray-800 rounded-lg p-4 mb-6">
          {(() => {
            const comp = components.find(c => c.id === selectedComponent)!;
            return (
              <div>
                <h3 className="text-lg font-bold mb-2">{comp.name}</h3>
                <p className="text-sm text-gray-400 mb-3">{comp.description}</p>
                <div className="flex gap-2">
                  {comp.details.map((detail, i) => (
                    <span
                      key={i}
                      className={`px-2 py-1 rounded text-xs ${comp.color} text-white`}
                    >
                      {detail}
                    </span>
                  ))}
                </div>
              </div>
            );
          })()}
        </div>
      )}

      <div className="grid grid-cols-4 gap-4 text-xs">
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">总参数量</div>
          <div className="text-lg font-bold text-blue-400">671B</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">激活参数</div>
          <div className="text-lg font-bold text-green-400">37B</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">专家数量</div>
          <div className="text-lg font-bold text-yellow-400">256</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">KV 压缩比</div>
          <div className="text-lg font-bold text-purple-400">93.3%</div>
        </div>
      </div>
    </div>
  );
}
