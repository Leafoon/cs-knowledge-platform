"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { TrendingUp, Cpu, Users, BarChart3 } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

interface ScalabilityModel {
  id: string;
  name: string;
  description: string;
  dataGenerator: (cores: number) => number;
  color: string;
}

const models: ScalabilityModel[] = [
  {
    id: "linear",
    name: "线性扩展 (Linear)",
    description: "性能与核心数成正比（理想情况）",
    dataGenerator: (cores) => cores,
    color: "#10b981"
  },
  {
    id: "sublinear",
    name: "次线性扩展 (Sub-linear)",
    description: "性能增长慢于核心数（有同步开销）",
    dataGenerator: (cores) => cores * 0.8 - Math.log(cores) * 0.5,
    color: "#3b82f6"
  },
  {
    id: "amdahl",
    name: "Amdahl 定律",
    description: "有10%串行代码限制并行加速",
    dataGenerator: (cores) => 1 / (0.1 + 0.9 / cores),
    color: "#f59e0b"
  },
  {
    id: "superlinear",
    name: "超线性扩展 (Super-linear)",
    description: "缓存效应导致性能超比例增长",
    dataGenerator: (cores) => cores * 1.2 + Math.log(cores) * 0.3,
    color: "#8b5cf6"
  }
];

export default function ScalabilityAnalyzer() {
  const [selectedModels, setSelectedModels] = useState<string[]>(["linear", "sublinear", "amdahl"]);
  const [maxCores, setMaxCores] = useState(16);

  const toggleModel = (id: string) => {
    setSelectedModels(prev =>
      prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]
    );
  };

  // 生成图表数据
  const chartData = Array.from({ length: maxCores }, (_, i) => {
    const cores = i + 1;
    const dataPoint: any = { cores };
    models.forEach(model => {
      if (selectedModels.includes(model.id)) {
        dataPoint[model.name] = model.dataGenerator(cores);
      }
    });
    return dataPoint;
  });

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <div className="flex items-center gap-3 mb-6">
        <BarChart3 className="w-8 h-8 text-blue-600 dark:text-blue-400" />
        <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
          可扩展性分析器
        </h3>
      </div>

      {/* 模型选择 */}
      <div className="mb-6 p-4 bg-white dark:bg-slate-800 rounded-lg shadow">
        <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-3">
          选择扩展模型
        </h4>
        <div className="grid md:grid-cols-2 gap-3">
          {models.map((model) => (
            <motion.button
              key={model.id}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => toggleModel(model.id)}
              className={`p-3 rounded-lg text-left transition-all ${
                selectedModels.includes(model.id)
                  ? "bg-blue-600 text-white shadow-lg"
                  : "bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300"
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: model.color }}
                />
                <span className="font-semibold">{model.name}</span>
              </div>
              <p className="text-xs opacity-90">{model.description}</p>
            </motion.button>
          ))}
        </div>
      </div>

      {/* 核心数控制 */}
      <div className="mb-6 p-4 bg-white dark:bg-slate-800 rounded-lg shadow">
        <div className="flex items-center justify-between mb-2">
          <span className="font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-2">
            <Cpu className="w-5 h-5" />
            最大核心数
          </span>
          <span className="text-lg font-bold text-blue-600">{maxCores}</span>
        </div>
        <input
          type="range"
          min="4"
          max="64"
          step="4"
          value={maxCores}
          onChange={(e) => setMaxCores(Number(e.target.value))}
          className="w-full"
        />
      </div>

      {/* 图表 */}
      <div className="p-4 bg-white dark:bg-slate-800 rounded-lg shadow mb-6">
        <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4">
          性能扩展曲线
        </h4>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis
              dataKey="cores"
              label={{ value: "核心数", position: "insideBottom", offset: -5 }}
              stroke="#888"
            />
            <YAxis
              label={{ value: "相对性能", angle: -90, position: "insideLeft" }}
              stroke="#888"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1f2937",
                border: "1px solid #374151",
                borderRadius: "8px"
              }}
            />
            <Legend />
            {models.map(model =>
              selectedModels.includes(model.id) ? (
                <Line
                  key={model.id}
                  type="monotone"
                  dataKey={model.name}
                  stroke={model.color}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                />
              ) : null
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Amdahl 定律说明 */}
      <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
        <h5 className="font-semibold text-amber-800 dark:text-amber-300 mb-2">
          Amdahl 定律
        </h5>
        <p className="text-sm text-amber-900 dark:text-amber-100 mb-2">
          加速比 = 1 / (S + P/N)，其中 S 是串行部分比例，P 是可并行部分比例，N 是核心数。
        </p>
        <p className="text-sm text-amber-900 dark:text-amber-100">
          即使有无限核心，加速比也受限于串行部分。如果 S = 0.1（10% 串行），最大加速比仅为 10 倍。
        </p>
      </div>
    </div>
  );
}
