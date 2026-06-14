"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, AlertTriangle, CheckCircle } from "lucide-react";

interface BlockState {
  id: string;
  label: string;
  color: string;
  modified: boolean;
  committed: boolean;
}

interface CrashPoint {
  id: number;
  label: string;
  description: string;
  result: string;
  consistent: boolean;
}

const crashPoints: CrashPoint[] = [
  {
    id: 1,
    label: "日志写入前崩溃",
    description: "修改意图尚未写入日志",
    result: "日志为空，文件系统未修改。恢复后状态一致。",
    consistent: true,
  },
  {
    id: 2,
    label: "日志写入中崩溃",
    description: "部分块已写入日志，未提交",
    result: "日志未提交（无 commit 标记），丢弃日志。文件系统未修改。",
    consistent: true,
  },
  {
    id: 3,
    label: "日志提交后崩溃",
    description: "日志已提交，尚未写入实际位置",
    result: "日志已提交 → 重放日志，将修改写入实际位置。文件系统一致。",
    consistent: true,
  },
  {
    id: 4,
    label: "检查点完成前崩溃",
    description: "部分块已从日志复制到实际位置",
    result: "日志已提交 → 再次重放日志（幂等操作）。文件系统一致。",
    consistent: true,
  },
  {
    id: 5,
    label: "检查点完成后崩溃",
    description: "所有块已写入实际位置",
    result: "日志已释放，无需恢复。文件系统已更新且一致。",
    consistent: true,
  },
];

const scenarioSteps = [
  { label: "分配 inode", block: "inode", status: "pending" as const },
  { label: "更新目录", block: "dir", status: "pending" as const },
  { label: "更新位图", block: "bitmap", status: "pending" as const },
];

export default function CrashScenarioSimulator() {
  const [mode, setMode] = useState<"no-log" | "wal">("no-log");
  const [step, setStep] = useState(-1);
  const [crashAt, setCrashAt] = useState(-1);
  const [isRunning, setIsRunning] = useState(false);
  const [blocks, setBlocks] = useState<BlockState[]>([
    { id: "inode", label: "Inode 表", color: "emerald", modified: false, committed: false },
    { id: "dir", label: "目录数据", color: "blue", modified: false, committed: false },
    { id: "bitmap", label: "位图", color: "amber", modified: false, committed: false },
    { id: "log", label: "日志区", color: "purple", modified: false, committed: false },
  ]);

  const reset = useCallback(() => {
    setStep(-1);
    setCrashAt(-1);
    setIsRunning(false);
    setBlocks((prev) =>
      prev.map((b) => ({ ...b, modified: false, committed: false }))
    );
  }, []);

  const runScenario = (crashPoint: number) => {
    reset();
    setIsRunning(true);
    setCrashAt(crashPoint);

    let i = 0;
    const interval = setInterval(() => {
      if (i >= crashPoint) {
        clearInterval(interval);
        setIsRunning(false);
        return;
      }

      setStep(i);
      setBlocks((prev) => {
        const updated = [...prev];
        if (mode === "no-log") {
          if (i === 0) updated[0] = { ...updated[0], modified: true };
          if (i === 1) updated[1] = { ...updated[1], modified: true };
          if (i === 2) updated[2] = { ...updated[2], modified: true };
        } else {
          if (i === 0) updated[3] = { ...updated[3], modified: true };
          if (i === 1) updated[3] = { ...updated[3], committed: true };
          if (i === 2) {
            updated[0] = { ...updated[0], modified: true };
            updated[1] = { ...updated[1], modified: true };
            updated[2] = { ...updated[2], modified: true };
          }
        }
        return updated;
      });
      i++;
    }, 800);
  };

  const crashPoints_noLog = [
    "修改 inode 前",
    "修改目录后，位图未更新",
    "全部完成前",
  ];

  const crashPoints_wal = [
    "日志写入前",
    "日志写入中（未提交）",
    "检查点中（日志已提交）",
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-red-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        崩溃场景模拟器
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        对比无日志和 WAL 模式下，不同崩溃点的文件系统一致性
      </p>

      {/* Mode toggle */}
      <div className="flex gap-3 mb-6 justify-center">
        <button
          onClick={() => { setMode("no-log"); reset(); }}
          className={`px-4 py-2 rounded-lg text-sm font-bold transition-colors ${
            mode === "no-log"
              ? "bg-red-500 text-white"
              : "bg-slate-200 text-slate-600 dark:bg-gray-700 dark:text-gray-300"
          }`}
        >
          无日志模式
        </button>
        <button
          onClick={() => { setMode("wal"); reset(); }}
          className={`px-4 py-2 rounded-lg text-sm font-bold transition-colors ${
            mode === "wal"
              ? "bg-emerald-500 text-white"
              : "bg-slate-200 text-slate-600 dark:bg-gray-700 dark:text-gray-300"
          }`}
        >
          WAL 日志模式
        </button>
      </div>

      {/* Crash point buttons */}
      <div className="flex flex-wrap gap-2 mb-6 justify-center">
        {(mode === "no-log" ? crashPoints_noLog : crashPoints_wal).map(
          (label, i) => (
            <button
              key={i}
              onClick={() => runScenario(i + 1)}
              disabled={isRunning}
              className="flex items-center gap-1.5 px-3 py-2 bg-white dark:bg-gray-800 border border-slate-200 dark:border-gray-700 rounded-lg text-sm hover:border-red-300 dark:hover:border-red-600 disabled:opacity-50 transition-colors"
            >
              <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
              <span className="text-slate-700 dark:text-gray-200">{label}</span>
            </button>
          )
        )}
        <button
          onClick={reset}
          className="flex items-center gap-1.5 px-3 py-2 bg-slate-100 dark:bg-gray-700 rounded-lg text-sm text-slate-600 dark:text-gray-300 hover:bg-slate-200 dark:hover:bg-gray-600 transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          重置
        </button>
      </div>

      {/* Block states */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
        {blocks.map((block) => {
          const logBlock = block.id === "log";
          if (logBlock && mode === "no-log") return null;
          return (
            <motion.div
              key={block.id}
              className={`p-4 rounded-lg border-2 text-center transition-all ${
                block.committed
                  ? `border-${block.color}-500 bg-${block.color}-50 dark:bg-${block.color}-950/30 shadow-md`
                  : block.modified
                  ? `border-${block.color}-300 bg-${block.color}-50/50 dark:bg-${block.color}-950/20`
                  : "border-slate-200 dark:border-gray-700 bg-white dark:bg-gray-800"
              }`}
              animate={
                block.modified && !block.committed
                  ? { scale: [1, 1.03, 1] }
                  : {}
              }
            >
              <div className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-1">
                {block.label}
              </div>
              <div
                className={`text-xs font-mono ${
                  block.committed
                    ? "text-emerald-600 dark:text-emerald-400"
                    : block.modified
                    ? "text-amber-600 dark:text-amber-400"
                    : "text-slate-400 dark:text-gray-500"
                }`}
              >
                {block.committed ? "已提交" : block.modified ? "已修改" : "未修改"}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Crash message */}
      {crashAt > 0 && !isRunning && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className={`p-4 rounded-lg border-2 ${
            mode === "wal"
              ? "border-emerald-300 bg-emerald-50 dark:bg-emerald-950/20 dark:border-emerald-700"
              : crashAt >= 2
              ? "border-red-300 bg-red-50 dark:bg-red-950/20 dark:border-red-700"
              : "border-emerald-300 bg-emerald-50 dark:bg-emerald-950/20 dark:border-emerald-700"
          }`}
        >
          <div className="flex items-center gap-2 mb-2">
            {mode === "wal" || crashAt < 2 ? (
              <CheckCircle className="w-5 h-5 text-emerald-500" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-red-500" />
            )}
            <span className="font-bold text-slate-800 dark:text-gray-100">
              崩溃点 {crashAt}：{crashPoints_noLog[crashAt - 1] || crashPoints_wal[crashAt - 1]}
            </span>
          </div>
          <p className="text-sm text-slate-600 dark:text-gray-300">
            {mode === "wal"
              ? crashAt === 1
                ? "日志为空，文件系统未修改。恢复后一致。"
                : crashAt === 2
                ? "日志未提交，丢弃日志。文件系统未修改。"
                : "日志已提交，重放日志。文件系统一致。"
              : crashAt === 1
              ? "崩溃发生在任何修改之前，文件系统一致。"
              : crashAt === 2
              ? "inode 和目录已修改但位图未更新 → 文件系统不一致！数据泄漏。"
              : "操作未完成 → 文件系统可能不一致。"}
          </p>
        </motion.div>
      )}
    </div>
  );
}
