"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Database, Shield, Zap, ArrowRight } from "lucide-react";

interface JournalMode {
  id: string;
  name: string;
  nameZh: string;
  icon: React.ReactNode;
  color: string;
  bgColor: string;
  borderColor: string;
  description: string;
  writeAmplification: string;
  safety: string;
  performance: string;
  useCase: string;
  dataFlow: string[];
}

const modes: JournalMode[] = [
  {
    id: "data",
    name: "Data Journaling",
    nameZh: "数据日志",
    icon: <Shield className="w-5 h-5" />,
    color: "text-blue-700 dark:text-blue-300",
    bgColor: "bg-blue-50 dark:bg-blue-950/40",
    borderColor: "border-blue-300 dark:border-blue-700",
    description: "日志中同时记录数据块和元数据块。安全性最高，但每个块都要写两次（日志 + 实际位置）。",
    writeAmplification: "2x",
    safety: "最高",
    performance: "最慢",
    useCase: "关键数据（金融、医疗）",
    dataFlow: [
      "数据块 → 日志区",
      "元数据 → 日志区",
      "commit",
      "数据块 → 实际位置",
      "元数据 → 实际位置",
    ],
  },
  {
    id: "ordered",
    name: "Ordered Mode",
    nameZh: "有序模式（默认）",
    icon: <Zap className="w-5 h-5" />,
    color: "text-emerald-700 dark:text-emerald-300",
    bgColor: "bg-emerald-50 dark:bg-emerald-950/40",
    borderColor: "border-emerald-300 dark:border-emerald-700",
    description: "元数据日志，但保证数据块在元数据之前写入实际位置。数据块不需要写入日志，性能较好。",
    writeAmplification: "~1.5x",
    safety: "高",
    performance: "中",
    useCase: "默认选择（ext4 默认）",
    dataFlow: [
      "数据块 → 实际位置",
      "元数据 → 日志区",
      "commit",
      "元数据 → 实际位置",
    ],
  },
  {
    id: "writeback",
    name: "Writeback Mode",
    nameZh: "回写模式",
    icon: <Database className="w-5 h-5" />,
    color: "text-amber-700 dark:text-amber-300",
    bgColor: "bg-amber-50 dark:bg-amber-950/40",
    borderColor: "border-amber-300 dark:border-amber-700",
    description: "只记录元数据到日志，数据块的写入顺序不保证。性能最好，但崩溃后可能读到旧数据。",
    writeAmplification: "~1x",
    safety: "中",
    performance: "最快",
    useCase: "高性能需求、可容忍数据丢失",
    dataFlow: [
      "元数据 → 日志区",
      "数据块 → 实际位置（顺序不保证）",
      "commit",
      "元数据 → 实际位置",
    ],
  },
];

export default function JournalModeComparison() {
  const [selectedMode, setSelectedMode] = useState<string | null>(null);
  const selected = modes.find((m) => m.id === selectedMode);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        日志模式对比
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        ext3/ext4 支持三种日志模式，在安全性与性能之间权衡
      </p>

      {/* Comparison table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-slate-200 dark:border-gray-700 overflow-hidden mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-50 dark:bg-gray-900/50">
              <th className="px-4 py-3 text-left font-bold text-slate-700 dark:text-gray-200 border-b border-r border-slate-200 dark:border-gray-700">
                模式
              </th>
              <th className="px-4 py-3 text-center font-bold text-slate-700 dark:text-gray-200 border-b border-r border-slate-200 dark:border-gray-700">
                写放大
              </th>
              <th className="px-4 py-3 text-center font-bold text-slate-700 dark:text-gray-200 border-b border-r border-slate-200 dark:border-gray-700">
                安全性
              </th>
              <th className="px-4 py-3 text-center font-bold text-slate-700 dark:text-gray-200 border-b border-r border-slate-200 dark:border-gray-700">
                性能
              </th>
              <th className="px-4 py-3 text-left font-bold text-slate-700 dark:text-gray-200 border-b border-slate-200 dark:border-gray-700">
                适用场景
              </th>
            </tr>
          </thead>
          <tbody>
            {modes.map((mode) => (
              <motion.tr
                key={mode.id}
                onClick={() =>
                  setSelectedMode(selectedMode === mode.id ? null : mode.id)
                }
                className={`cursor-pointer transition-colors ${
                  selectedMode === mode.id
                    ? `${mode.bgColor}`
                    : "hover:bg-slate-50 dark:hover:bg-gray-750"
                }`}
                whileHover={{ scale: 1.005 }}
              >
                <td className="px-4 py-3 border-b border-r border-slate-200 dark:border-gray-700">
                  <div className="flex items-center gap-2">
                    <span className={mode.color}>{mode.icon}</span>
                    <span className="font-bold text-slate-800 dark:text-gray-100">
                      {mode.nameZh}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3 text-center border-b border-r border-slate-200 dark:border-gray-700 font-mono text-slate-600 dark:text-gray-300">
                  {mode.writeAmplification}
                </td>
                <td className="px-4 py-3 text-center border-b border-r border-slate-200 dark:border-gray-700">
                  <span
                    className={`px-2 py-0.5 rounded text-xs font-bold ${
                      mode.safety === "最高"
                        ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                        : mode.safety === "高"
                        ? "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300"
                        : "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300"
                    }`}
                  >
                    {mode.safety}
                  </span>
                </td>
                <td className="px-4 py-3 text-center border-b border-r border-slate-200 dark:border-gray-700">
                  <span
                    className={`px-2 py-0.5 rounded text-xs font-bold ${
                      mode.performance === "最快"
                        ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                        : mode.performance === "中"
                        ? "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300"
                        : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300"
                    }`}
                  >
                    {mode.performance}
                  </span>
                </td>
                <td className="px-4 py-3 border-b border-slate-200 dark:border-gray-700 text-slate-600 dark:text-gray-300 text-xs">
                  {mode.useCase}
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Detail panel */}
      {selected && (
        <motion.div
          key={selected.id}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className={`${selected.bgColor} rounded-lg p-5 border ${selected.borderColor}`}
        >
          <div className="flex items-center gap-2 mb-3">
            <span className={selected.color}>{selected.icon}</span>
            <h3 className={`text-lg font-bold ${selected.color}`}>
              {selected.nameZh}
            </h3>
            <span className="text-xs font-mono text-slate-500 dark:text-gray-400">
              ({selected.name})
            </span>
          </div>
          <p className="text-sm text-slate-700 dark:text-gray-200 mb-4 leading-relaxed">
            {selected.description}
          </p>

          <h4 className="text-xs font-bold text-slate-600 dark:text-gray-300 mb-2 uppercase tracking-wider">
            数据流
          </h4>
          <div className="flex flex-wrap items-center gap-2">
            {selected.dataFlow.map((step, i) => (
              <React.Fragment key={i}>
                <span className="px-3 py-1.5 bg-white/60 dark:bg-gray-800/60 rounded text-xs font-mono text-slate-700 dark:text-gray-200 border border-slate-200 dark:border-gray-700">
                  {step}
                </span>
                {i < selected.dataFlow.length - 1 && (
                  <ArrowRight className="w-3 h-3 text-slate-400 shrink-0" />
                )}
              </React.Fragment>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}
