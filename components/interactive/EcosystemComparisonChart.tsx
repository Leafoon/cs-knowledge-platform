'use client';

import { useState } from 'react';

const ecosystems = [
  {
    name: 'TileLang',
    color: '#3b82f6',
    dimensions: {
      community: 65,
      docs: 75,
      tools: 80,
      integrations: 70,
      support: 72,
      maturity: 60,
    },
  },
  {
    name: 'Triton',
    color: '#22c55e',
    dimensions: {
      community: 90,
      docs: 95,
      tools: 85,
      integrations: 88,
      support: 92,
      maturity: 85,
    },
  },
  {
    name: 'CUDA',
    color: '#ef4444',
    dimensions: {
      community: 95,
      docs: 90,
      tools: 98,
      integrations: 95,
      support: 88,
      maturity: 95,
    },
  },
];

const dimensionLabels: Record<string, string> = {
  community: '社区活跃度',
  docs: '文档质量',
  tools: '工具链',
  integrations: '集成生态',
  support: '技术支持',
  maturity: '成熟度',
};

export default function EcosystemComparisonChart() {
  const [selectedEcosystem, setSelectedEcosystem] = useState<number>(2);
  const [hoveredDim, setHoveredDim] = useState<string | null>(null);

  const dims = Object.keys(dimensionLabels);
  const centerX = 150;
  const centerY = 140;
  const maxRadius = 110;

  const getPolygonPoints = (ecosystem: typeof ecosystems[0]) => {
    return dims.map((dim, i) => {
      const angle = (Math.PI * 2 * i) / dims.length - Math.PI / 2;
      const value = ecosystem.dimensions[dim as keyof typeof ecosystem.dimensions] / 100;
      const x = centerX + maxRadius * value * Math.cos(angle);
      const y = centerY + maxRadius * value * Math.sin(angle);
      return `${x},${y}`;
    }).join(' ');
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">生态系统成熟度雷达图</h2>
      <p className="text-gray-400 text-sm mb-4">从多个维度对比 TileLang、Triton 和 CUDA 的生态系统</p>

      <div className="flex gap-3 mb-6">
        {ecosystems.map((eco, idx) => (
          <button
            key={idx}
            onClick={() => setSelectedEcosystem(idx)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs font-medium transition-all ${
              selectedEcosystem === idx ? 'bg-gray-700' : 'bg-gray-800 text-gray-400'
            }`}
          >
            <span className="w-3 h-3 rounded-full" style={{ backgroundColor: eco.color }} />
            {eco.name}
          </button>
        ))}
      </div>

      <div className="flex gap-8 mb-6">
        <div className="bg-gray-800 rounded-lg p-4 flex-shrink-0">
          <svg width="300" height="280" viewBox="0 0 300 280">
            {/* Grid circles */}
            {[0.2, 0.4, 0.6, 0.8, 1.0].map((level, i) => (
              <circle
                key={i}
                cx={centerX}
                cy={centerY}
                r={maxRadius * level}
                fill="none"
                stroke="#374151"
                strokeWidth="1"
              />
            ))}

            {/* Grid lines */}
            {dims.map((_, i) => {
              const angle = (Math.PI * 2 * i) / dims.length - Math.PI / 2;
              const x = centerX + maxRadius * Math.cos(angle);
              const y = centerY + maxRadius * Math.sin(angle);
              return (
                <line
                  key={i}
                  x1={centerX}
                  y1={centerY}
                  x2={x}
                  y2={y}
                  stroke="#374151"
                  strokeWidth="1"
                />
              );
            })}

            {/* Data polygons */}
            {ecosystems.map((eco, idx) => (
              <polygon
                key={idx}
                points={getPolygonPoints(eco)}
                fill={eco.color}
                fillOpacity={selectedEcosystem === idx ? 0.3 : 0.1}
                stroke={eco.color}
                strokeWidth={selectedEcosystem === idx ? 2 : 1}
                className="transition-all"
              />
            ))}

            {/* Labels */}
            {dims.map((dim, i) => {
              const angle = (Math.PI * 2 * i) / dims.length - Math.PI / 2;
              const x = centerX + (maxRadius + 20) * Math.cos(angle);
              const y = centerY + (maxRadius + 20) * Math.sin(angle);

              return (
                <text
                  key={i}
                  x={x}
                  y={y}
                  fill={hoveredDim === dim ? '#fff' : '#9ca3af'}
                  fontSize="11"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="cursor-pointer"
                  onMouseEnter={() => setHoveredDim(dim)}
                  onMouseLeave={() => setHoveredDim(null)}
                >
                  {dimensionLabels[dim]}
                </text>
              );
            })}

            {/* Data points */}
            {ecosystems[selectedEcosystem] && dims.map((dim, i) => {
              const angle = (Math.PI * 2 * i) / dims.length - Math.PI / 2;
              const value = ecosystems[selectedEcosystem].dimensions[dim as keyof typeof ecosystems[0]['dimensions']] / 100;
              const x = centerX + maxRadius * value * Math.cos(angle);
              const y = centerY + maxRadius * value * Math.sin(angle);

              return (
                <circle
                  key={i}
                  cx={x}
                  cy={y}
                  r={hoveredDim === dim ? 6 : 4}
                  fill={ecosystems[selectedEcosystem].color}
                  stroke="#fff"
                  strokeWidth="2"
                />
              );
            })}
          </svg>
        </div>

        <div className="flex-1 space-y-4">
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold text-gray-300 mb-3">详细评分</h3>
            <div className="space-y-3">
              {dims.map((dim) => (
                <div
                  key={dim}
                  className={`p-2 rounded ${hoveredDim === dim ? 'bg-gray-700' : ''}`}
                  onMouseEnter={() => setHoveredDim(dim)}
                  onMouseLeave={() => setHoveredDim(null)}
                >
                  <div className="text-xs text-gray-400 mb-2">{dimensionLabels[dim]}</div>
                  <div className="space-y-1">
                    {ecosystems.map((eco) => (
                      <div key={eco.name} className="flex items-center gap-2">
                        <span className="w-16 text-[10px] text-gray-500">{eco.name}</span>
                        <div className="flex-1 h-1.5 bg-gray-700 rounded-full">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${eco.dimensions[dim as keyof typeof eco.dimensions]}%`,
                              backgroundColor: eco.color,
                            }}
                          />
                        </div>
                        <span className="text-[10px] text-gray-400 w-6 text-right">
                          {eco.dimensions[dim as keyof typeof eco.dimensions]}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
