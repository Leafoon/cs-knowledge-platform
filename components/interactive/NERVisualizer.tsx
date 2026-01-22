"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";

interface Entity {
  word: string;
  type: "PER" | "ORG" | "LOC" | "DATE" | "MISC";
  start: number;
  end: number;
  score: number;
}

const sampleText = "Apple was founded by Steve Jobs in Cupertino, California in 1976.";

const entities: Entity[] = [
  { word: "Apple", type: "ORG", start: 0, end: 5, score: 0.998 },
  { word: "Steve Jobs", type: "PER", start: 21, end: 31, score: 0.999 },
  { word: "Cupertino", type: "LOC", start: 35, end: 44, score: 0.995 },
  { word: "California", type: "LOC", start: 46, end: 56, score: 0.997 },
  { word: "1976", type: "DATE", start: 60, end: 64, score: 0.985 }
];

const entityColors: Record<string, { bg: string; text: string; label: string }> = {
  PER: { bg: "bg-blue-500/20", text: "text-blue-400", label: "人名" },
  ORG: { bg: "bg-green-500/20", text: "text-green-400", label: "组织" },
  LOC: { bg: "bg-purple-500/20", text: "text-purple-400", label: "地点" },
  DATE: { bg: "bg-orange-500/20", text: "text-orange-400", label: "日期" },
  MISC: { bg: "bg-gray-500/20", text: "text-gray-400", label: "其他" }
};

export default function NERVisualizer() {
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null);
  const [highlightType, setHighlightType] = useState<string | null>(null);

  const renderHighlightedText = () => {
    let lastIndex = 0;
    const parts: JSX.Element[] = [];

    // 按起始位置排序
    const sortedEntities = [...entities].sort((a, b) => a.start - b.start);

    sortedEntities.forEach((entity, index) => {
      // 添加实体前的普通文本
      if (entity.start > lastIndex) {
        parts.push(
          <span key={`text-${index}`} className="text-gray-300">
            {sampleText.slice(lastIndex, entity.start)}
          </span>
        );
      }

      // 添加高亮的实体
      const shouldHighlight = !highlightType || highlightType === entity.type;
      const isSelected = selectedEntity?.word === entity.word;
      const colors = entityColors[entity.type];

      parts.push(
        <motion.span
          key={`entity-${index}`}
          initial={{ backgroundColor: "transparent" }}
          animate={{
            backgroundColor: shouldHighlight ? undefined : "transparent"
          }}
          whileHover={{ scale: 1.05 }}
          onClick={() => setSelectedEntity(isSelected ? null : entity)}
          className={`
            inline-block px-2 py-1 mx-0.5 rounded cursor-pointer transition-all
            ${shouldHighlight ? colors.bg : "bg-transparent"}
            ${shouldHighlight ? colors.text : "text-gray-500"}
            ${isSelected ? "ring-2 ring-white shadow-lg" : ""}
            font-semibold
          `}
          title={`${entity.type}: ${(entity.score * 100).toFixed(1)}%`}
        >
          {entity.word}
          {shouldHighlight && (
            <span className="ml-1 text-xs opacity-70">
              [{entity.type}]
            </span>
          )}
        </motion.span>
      );

      lastIndex = entity.end;
    });

    // 添加最后的普通文本
    if (lastIndex < sampleText.length) {
      parts.push(
        <span key="text-end" className="text-gray-300">
          {sampleText.slice(lastIndex)}
        </span>
      );
    }

    return parts;
  };

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-slate-900 to-slate-800 rounded-xl border border-slate-700 shadow-2xl">
      <h3 className="text-2xl font-bold mb-6 text-center bg-gradient-to-r from-blue-400 to-green-500 bg-clip-text text-transparent">
        🏷️ 命名实体识别 (NER) 可视化
      </h3>

      {/* Entity Type Filter */}
      <div className="mb-6">
        <h4 className="text-sm font-semibold text-gray-400 mb-3">实体类型过滤</h4>
        <div className="flex flex-wrap gap-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setHighlightType(null)}
            className={`px-4 py-2 rounded-lg font-semibold transition-all ${
              !highlightType
                ? "bg-white text-slate-900 shadow-lg"
                : "bg-slate-800 text-gray-400 hover:bg-slate-700"
            }`}
          >
            全部
          </motion.button>
          {Object.entries(entityColors).map(([type, colors]) => {
            const count = entities.filter(e => e.type === type).length;
            if (count === 0) return null;

            return (
              <motion.button
                key={type}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setHighlightType(highlightType === type ? null : type)}
                className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                  highlightType === type
                    ? `${colors.bg} ${colors.text} ring-2 ring-current`
                    : "bg-slate-800 text-gray-400 hover:bg-slate-700"
                }`}
              >
                {colors.label} ({count})
              </motion.button>
            );
          })}
        </div>
      </div>

      {/* Highlighted Text */}
      <div className="mb-6 p-6 bg-slate-950 rounded-lg border border-slate-700">
        <div className="text-lg leading-relaxed">
          {renderHighlightedText()}
        </div>
      </div>

      {/* Entity Details */}
      <AnimatePresence mode="wait">
        {selectedEntity && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className={`p-4 rounded-lg border ${entityColors[selectedEntity.type].bg} border-current`}
          >
            <div className="flex items-start justify-between">
              <div>
                <h4 className={`text-lg font-bold ${entityColors[selectedEntity.type].text} mb-2`}>
                  {selectedEntity.word}
                </h4>
                <div className="space-y-1 text-sm text-gray-300">
                  <p>
                    <span className="text-gray-400">类型:</span>{" "}
                    <span className="font-semibold">{entityColors[selectedEntity.type].label} ({selectedEntity.type})</span>
                  </p>
                  <p>
                    <span className="text-gray-400">置信度:</span>{" "}
                    <span className="font-semibold">{(selectedEntity.score * 100).toFixed(1)}%</span>
                  </p>
                  <p>
                    <span className="text-gray-400">位置:</span>{" "}
                    <span className="font-mono">[{selectedEntity.start}:{selectedEntity.end}]</span>
                  </p>
                </div>
              </div>
              <button
                onClick={() => setSelectedEntity(null)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Entity List */}
      <div className="mt-6">
        <h4 className="text-sm font-semibold text-gray-400 mb-3">识别的实体列表</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {entities
            .filter(e => !highlightType || e.type === highlightType)
            .map((entity, index) => {
              const colors = entityColors[entity.type];
              const isSelected = selectedEntity?.word === entity.word;

              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  whileHover={{ scale: 1.02 }}
                  onClick={() => setSelectedEntity(isSelected ? null : entity)}
                  className={`p-3 rounded-lg cursor-pointer transition-all ${
                    isSelected
                      ? `${colors.bg} border-2 border-current shadow-lg`
                      : `bg-slate-800 border border-slate-700 hover:border-slate-600`
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className={`font-semibold ${colors.text}`}>
                      {entity.word}
                    </span>
                    <span className="text-xs text-gray-400">
                      {(entity.score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-0.5 text-xs rounded ${colors.bg} ${colors.text}`}>
                      {entity.type}
                    </span>
                    <span className="text-xs text-gray-500">
                      位置 {entity.start}-{entity.end}
                    </span>
                  </div>
                </motion.div>
              );
            })}
        </div>
      </div>

      {/* Legend */}
      <div className="mt-6 p-4 bg-slate-800 rounded-lg border border-slate-700">
        <h5 className="text-sm font-semibold text-gray-400 mb-2">图例</h5>
        <div className="flex flex-wrap gap-3 text-xs">
          {Object.entries(entityColors).map(([type, colors]) => (
            <div key={type} className="flex items-center gap-2">
              <div className={`w-4 h-4 rounded ${colors.bg} border border-current`}></div>
              <span className="text-gray-300">{colors.label} ({type})</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
