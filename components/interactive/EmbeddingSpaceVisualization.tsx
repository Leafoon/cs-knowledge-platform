"use client";

import React, { useState, useMemo } from 'react';
import { Search, Grid3x3, Layers } from 'lucide-react';

interface Point {
  id: number;
  x: number;
  y: number;
  text: string;
  category: string;
  color: string;
}

const categories = [
  { name: 'Technology', color: '#3b82f6', points: [
    { text: "LangChain framework", x: 100, y: 100 },
    { text: "Machine learning", x: 120, y: 90 },
    { text: "Deep learning", x: 90, y: 120 },
    { text: "Python programming", x: 110, y: 110 }
  ]},
  { name: 'Business', color: '#10b981', points: [
    { text: "Revenue growth", x: 300, y: 100 },
    { text: "Market analysis", x: 310, y: 90 },
    { text: "Sales report", x: 290, y: 110 },
    { text: "Financial data", x: 305, y: 105 }
  ]},
  { name: 'Science', color: '#8b5cf6', points: [
    { text: "Climate research", x: 100, y: 300 },
    { text: "Biology study", x: 110, y: 290 },
    { text: "Physics theory", x: 90, y: 310 },
    { text: "Chemistry lab", x: 105, y: 305 }
  ]},
  { name: 'Arts', color: '#ec4899', points: [
    { text: "Art gallery", x: 300, y: 300 },
    { text: "Music concert", x: 310, y: 290 },
    { text: "Photography", x: 290, y: 310 },
    { text: "Literature", x: 305, y: 305 }
  ]}
];

const allPoints: Point[] = categories.flatMap((cat, catIdx) =>
  cat.points.map((p, idx) => ({
    id: catIdx * 10 + idx,
    x: p.x,
    y: p.y,
    text: p.text,
    category: cat.name,
    color: cat.color
  }))
);

export default function EmbeddingSpaceVisualization() {
  const [query, setQuery] = useState("");
  const [queryPoint, setQueryPoint] = useState<{x: number; y: number} | null>(null);
  const [method, setMethod] = useState<'tsne' | 'umap'>('tsne');

  const handleSearch = () => {
    if (!query.trim()) return;
    
    const cat = query.toLowerCase().includes('tech') || query.toLowerCase().includes('ai') ? 'Technology' :
                query.toLowerCase().includes('business') || query.toLowerCase().includes('sales') ? 'Business' :
                query.toLowerCase().includes('science') ? 'Science' : 'Arts';
    
    const catPoints = allPoints.filter(p => p.category === cat);
    if (catPoints.length > 0) {
      const avgX = catPoints.reduce((sum, p) => sum + p.x, 0) / catPoints.length;
      const avgY = catPoints.reduce((sum, p) => sum + p.y, 0) / catPoints.length;
      setQueryPoint({ x: avgX, y: avgY });
    }
  };

  const nearestPoints = useMemo(() => {
    if (!queryPoint) return [];
    
    const distances = allPoints.map(p => ({
      ...p,
      distance: Math.sqrt(Math.pow(p.x - queryPoint.x, 2) + Math.pow(p.y - queryPoint.y, 2))
    }));
    
    return distances.sort((a, b) => a.distance - b.distance).slice(0, 3);
  }, [queryPoint]);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white rounded-xl border border-slate-200 my-8">
      <h3 className="text-xl font-bold text-slate-800 mb-4">Embedding 空间可视化</h3>

      {/* 方法选择 */}
      <div className="flex gap-3 mb-4">
        <button
          onClick={() => setMethod('tsne')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            method === 'tsne' ? 'bg-blue-500 text-white' : 'bg-slate-100 text-slate-600'
          }`}
        >
          <Grid3x3 className="w-4 h-4 inline mr-2" />
          t-SNE
        </button>
        <button
          onClick={() => setMethod('umap')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            method === 'umap' ? 'bg-purple-500 text-white' : 'bg-slate-100 text-slate-600'
          }`}
        >
          <Layers className="w-4 h-4 inline mr-2" />
          UMAP
        </button>
      </div>

      {/* 查询输入 */}
      <div className="bg-slate-50 rounded-lg p-4 mb-6">
        <label className="text-sm font-medium text-slate-700 mb-2 block">查询相似文档</label>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="例如：AI technology"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            className="flex-1 px-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleSearch}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
          >
            <Search className="w-4 h-4 inline mr-2" />
            查询
          </button>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        {/* 可视化图 */}
        <div className="md:col-span-2 bg-slate-50 rounded-lg p-4">
          <h4 className="font-semibold text-slate-800 mb-3 text-sm">
            {method === 'tsne' ? 't-SNE' : 'UMAP'} 降维可视化
          </h4>
          
          <svg width="100%" height="400" viewBox="0 0 400 400" className="border border-slate-200 rounded bg-white">
            {/* 网格 */}
            <defs>
              <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e2e8f0" strokeWidth="0.5"/>
              </pattern>
            </defs>
            <rect width="400" height="400" fill="url(#grid)" />

            {/* 查询点的距离圈 */}
            {queryPoint && (
              <>
                <circle cx={queryPoint.x} cy={queryPoint.y} r="50" fill="none" stroke="#3b82f6" strokeWidth="1" strokeDasharray="4 2" opacity="0.3" />
                <circle cx={queryPoint.x} cy={queryPoint.y} r="100" fill="none" stroke="#3b82f6" strokeWidth="1" strokeDasharray="4 2" opacity="0.2" />
              </>
            )}

            {/* 文档点 */}
            {allPoints.map((point) => (
              <g key={point.id}>
                <circle
                  cx={point.x}
                  cy={point.y}
                  r="6"
                  fill={point.color}
                  opacity={queryPoint && nearestPoints.some(p => p.id === point.id) ? 1 : 0.6}
                  style={{ transition: 'opacity 0.3s' }}
                />
                {queryPoint && nearestPoints.some(p => p.id === point.id) && (
                  <line
                    x1={queryPoint.x}
                    y1={queryPoint.y}
                    x2={point.x}
                    y2={point.y}
                    stroke="#3b82f6"
                    strokeWidth="1"
                    strokeDasharray="2 2"
                    opacity="0.5"
                  />
                )}
              </g>
            ))}

            {/* 查询点 */}
            {queryPoint && (
              <g>
                <circle cx={queryPoint.x} cy={queryPoint.y} r="10" fill="#3b82f6" stroke="white" strokeWidth="2" />
                <text x={queryPoint.x} y={queryPoint.y - 15} fontSize="12" fill="#1e293b" fontWeight="bold" textAnchor="middle">
                  Query
                </text>
              </g>
            )}
          </svg>

          {/* 图例 */}
          <div className="flex gap-4 mt-3">
            {categories.map((cat) => (
              <div key={cat.name} className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: cat.color }} />
                <span className="text-xs text-slate-600">{cat.name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* 最近邻 */}
        <div className="bg-slate-50 rounded-lg p-4">
          <h4 className="font-semibold text-slate-800 mb-3 text-sm">最近邻（Top-3）</h4>
          
          {!queryPoint ? (
            <div className="text-center py-12 text-slate-400">
              <Search className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p className="text-sm">输入查询以显示相似文档</p>
            </div>
          ) : (
            <div className="space-y-3">
              {nearestPoints.map((point, idx) => {
                const similarity = Math.max(0, 100 - point.distance).toFixed(1);
                return (
                  <div key={point.id} className="p-3 rounded-lg border border-slate-200 bg-white">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-xs font-semibold text-slate-500">#{idx + 1}</span>
                      <span className="text-xs font-bold" style={{ color: point.color }}>
                        {similarity}% 相似
                      </span>
                    </div>
                    <div className="text-sm font-medium text-slate-800 mb-1">{point.text}</div>
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: point.color }} />
                      <span className="text-xs text-slate-500">{point.category}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* 统计信息 */}
          {queryPoint && (
            <div className="mt-4 pt-4 border-t border-slate-200">
              <div className="text-xs text-slate-600 space-y-1">
                <div className="flex justify-between">
                  <span>检索方法</span>
                  <span className="font-semibold">余弦相似度</span>
                </div>
                <div className="flex justify-between">
                  <span>向量维度</span>
                  <span className="font-semibold">1536</span>
                </div>
                <div className="flex justify-between">
                  <span>文档总数</span>
                  <span className="font-semibold">{allPoints.length}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
