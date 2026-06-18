"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Network, ArrowRight, ZoomIn, ZoomOut } from "lucide-react";

interface Entity {
  id: string;
  name: string;
  type: "person" | "concept" | "technology" | "organization";
  x: number;
  y: number;
}

interface Relation {
  from: string;
  to: string;
  label: string;
}

const ENTITIES: Entity[] = [
  { id: "1", name: "RAG", type: "technology", x: 200, y: 150 },
  { id: "2", name: "LLM", type: "technology", x: 350, y: 100 },
  { id: "3", name: "向量数据库", type: "technology", x: 100, y: 250 },
  { id: "4", name: "Embedding", type: "concept", x: 300, y: 250 },
  { id: "5", name: "检索", type: "concept", x: 150, y: 100 },
  { id: "6", name: "生成", type: "concept", x: 400, y: 200 },
];

const RELATIONS: Relation[] = [
  { from: "1", to: "2", label: "使用" },
  { from: "1", to: "3", label: "存储" },
  { from: "1", to: "4", label: "依赖" },
  { from: "4", to: "3", label: "索引" },
  { from: "5", to: "3", label: "查询" },
  { from: "2", to: "6", label: "执行" },
  { from: "1", to: "5", label: "包含" },
  { from: "1", to: "6", label: "包含" },
];

const TYPE_COLORS: Record<Entity["type"], string> = {
  technology: "#3b82f6",
  concept: "#10b981",
  person: "#f59e0b",
  organization: "#8b5cf6",
};

export function GraphRAGVisualization() {
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null);
  const [scale, setScale] = useState(1);

  const connectedEntities = selectedEntity
    ? RELATIONS.filter((r) => r.from === selectedEntity || r.to === selectedEntity)
        .map((r) => (r.from === selectedEntity ? r.to : r.from))
    : [];

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-pink-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <Network className="w-6 h-6 text-pink-500" />
        Graph RAG 知识图谱可视化
      </h3>

      <div className="flex gap-4 mb-4">
        <button
          onClick={() => setScale((s) => Math.min(s + 0.1, 1.5))}
          className="p-2 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={() => setScale((s) => Math.max(s - 0.1, 0.5))}
          className="p-2 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700 overflow-hidden">
        <svg
          width="100%"
          height="350"
          viewBox="0 0 500 300"
          style={{ transform: `scale(${scale})`, transformOrigin: "center" }}
        >
          {/* 绘制连线 */}
          {RELATIONS.map((rel, i) => {
            const from = ENTITIES.find((e) => e.id === rel.from)!;
            const to = ENTITIES.find((e) => e.id === rel.to)!;
            const isSelected = selectedEntity && (rel.from === selectedEntity || rel.to === selectedEntity);
            return (
              <g key={i}>
                <line
                  x1={from.x}
                  y1={from.y}
                  x2={to.x}
                  y2={to.y}
                  stroke={isSelected ? "#ec4899" : "#cbd5e1"}
                  strokeWidth={isSelected ? 3 : 1}
                />
                <text
                  x={(from.x + to.x) / 2}
                  y={(from.y + to.y) / 2 - 5}
                  textAnchor="middle"
                  className="text-xs fill-slate-500"
                >
                  {rel.label}
                </text>
              </g>
            );
          })}

          {/* 绘制节点 */}
          {ENTITIES.map((entity) => {
            const isSelected = selectedEntity === entity.id;
            const isConnected = connectedEntities.includes(entity.id);
            return (
              <g
                key={entity.id}
                onClick={() => setSelectedEntity(isSelected ? null : entity.id)}
                className="cursor-pointer"
              >
                <circle
                  cx={entity.x}
                  cy={entity.y}
                  r={isSelected ? 30 : isConnected ? 25 : 20}
                  fill={TYPE_COLORS[entity.type]}
                  opacity={selectedEntity && !isSelected && !isConnected ? 0.3 : 1}
                  stroke={isSelected ? "#fff" : "none"}
                  strokeWidth={isSelected ? 3 : 0}
                />
                <text
                  x={entity.x}
                  y={entity.y + 35}
                  textAnchor="middle"
                  className="text-xs font-medium fill-slate-700 dark:fill-slate-200"
                >
                  {entity.name}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* 图例 */}
      <div className="flex gap-4 mt-4 justify-center">
        {Object.entries(TYPE_COLORS).map(([type, color]) => (
          <div key={type} className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
            <span className="text-sm text-slate-600 dark:text-slate-300">{type}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
