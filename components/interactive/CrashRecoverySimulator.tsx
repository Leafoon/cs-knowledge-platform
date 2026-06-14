"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, CheckCircle, AlertTriangle } from "lucide-react";

interface RecoveryPoint {
  id: number;
  label: string;
  scenario: string;
  journalCommitted: boolean;
  result: string;
  recoveryAction: string;
}

const recoveryPoints: RecoveryPoint[] = [
  {
    id: 1,
    label: "日志写入前崩溃",
    scenario: "begin_op() 已调用，但 log_write() 尚未执行",
    journalCommitted: false,
    result: "日志为空，文件系统未被修改",
    recoveryAction: "无需恢复，直接启动",
  },
  {
    id: 2,
    label: "日志写入中崩溃",
    scenario: "log_write() 已写入部分块到日志区",
    journalCommitted: false,
    result: "日志头未写入 commit 标记 → 日志视为无效",
    recoveryAction: "丢弃日志，文件系统保持原样",
  },
  {
    id: 3,
    label: "日志提交后崩溃",
    scenario: "write_head() 已写入 commit 标记，但 install_trans() 未完成",
    journalCommitted: true,
    result: "日志已提交，但修改尚未写入实际位置",
    recoveryAction: "重放日志：install_trans() 将日志块复制到实际位置",
  },
  {
    id: 4,
    label: "检查点中途崩溃",
    scenario: "install_trans() 已将部分块写入实际位置",
    journalCommitted: true,
    result: "部分块已更新，部分块还是旧数据",
    recoveryAction: "重放日志（幂等）：再次复制所有块，覆盖不一致状态",
  },
  {
    id: 5,
    label: "检查点完成后崩溃",
    scenario: "所有块已写入实际位置，日志头已清空",
    journalCommitted: false,
    result: "日志已清空，文件系统已完全更新",
    recoveryAction: "无需恢复，文件系统一致",
  },
];

export default function CrashRecoverySimulator() {
  const [selectedPoint, setSelectedPoint] = useState<number | null>(null);
  const [animating, setAnimating] = useState(false);
  const [recoveryStep, setRecoveryStep] = useState(-1);

  const selected = recoveryPoints.find((p) => p.id === selectedPoint);

  const simulateRecovery = useCallback(() => {
    if (!selected || animating) return;
    setAnimating(true);
    setRecoveryStep(0);

    const steps = selected.journalCommitted ? 3 : 1;
    let i = 0;
    const interval = setInterval(() => {
      setRecoveryStep(i);
      i++;
      if (i >= steps) {
        clearInterval(interval);
        setAnimating(false);
      }
    }, 1000);
  }, [selected, animating]);

  const resetSim = () => {
    setSelectedPoint(null);
    setRecoveryStep(-1);
    setAnimating(false);
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-amber-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        崩溃恢复模拟器
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        选择不同崩溃时间点，观察 xv6 如何恢复文件系统一致性
      </p>

      {/* Timeline */}
      <div className="relative mb-6">
        <div className="absolute top-1/2 left-0 right-0 h-1 bg-slate-200 dark:bg-gray-700 -translate-y-1/2 rounded" />
        <div className="flex justify-between relative">
          {recoveryPoints.map((point) => (
            <button
              key={point.id}
              onClick={() => {
                setSelectedPoint(point.id);
                setRecoveryStep(-1);
              }}
              className={`relative z-10 w-10 h-10 rounded-full flex items-center justify-center text-xs font-bold transition-all ${
                selectedPoint === point.id
                  ? point.journalCommitted
                    ? "bg-amber-500 text-white ring-4 ring-amber-200 dark:ring-amber-800"
                    : "bg-emerald-500 text-white ring-4 ring-emerald-200 dark:ring-emerald-800"
                  : "bg-white dark:bg-gray-800 text-slate-600 dark:text-gray-300 border-2 border-slate-300 dark:border-gray-600 hover:border-amber-400"
              }`}
            >
              {point.id}
            </button>
          ))}
        </div>
        <div className="flex justify-between mt-2">
          {recoveryPoints.map((point) => (
            <span
              key={point.id}
              className="text-[10px] text-slate-400 dark:text-gray-500 text-center w-10"
            >
              崩溃点{point.id}
            </span>
          ))}
        </div>
      </div>

      {/* Detail panel */}
      <AnimatePresence mode="wait">
        {selected ? (
          <motion.div
            key={selected.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-slate-200 dark:border-gray-700 mb-4"
          >
            <div className="flex items-center gap-3 mb-4">
              {selected.journalCommitted ? (
                <AlertTriangle className="w-5 h-5 text-amber-500" />
              ) : (
                <CheckCircle className="w-5 h-5 text-emerald-500" />
              )}
              <h3 className="text-lg font-bold text-slate-800 dark:text-gray-100">
                {selected.label}
              </h3>
            </div>

            <div className="space-y-3 mb-4">
              <div>
                <span className="text-xs font-bold text-slate-500 dark:text-gray-400 uppercase tracking-wider">
                  场景
                </span>
                <p className="text-sm text-slate-700 dark:text-gray-200 mt-1">
                  {selected.scenario}
                </p>
              </div>
              <div>
                <span className="text-xs font-bold text-slate-500 dark:text-gray-400 uppercase tracking-wider">
                  崩溃后状态
                </span>
                <p className="text-sm text-slate-700 dark:text-gray-200 mt-1">
                  {selected.result}
                </p>
              </div>
              <div>
                <span className="text-xs font-bold text-slate-500 dark:text-gray-400 uppercase tracking-wider">
                  恢复操作
                </span>
                <p className="text-sm text-slate-700 dark:text-gray-200 mt-1 font-mono">
                  {selected.recoveryAction}
                </p>
              </div>
            </div>

            <div className="flex gap-3">
              <button
                onClick={simulateRecovery}
                disabled={animating}
                className="flex items-center gap-2 px-4 py-2 bg-amber-500 text-white rounded-lg hover:bg-amber-600 disabled:opacity-50 transition-colors text-sm"
              >
                <Play className="w-4 h-4" />
                模拟恢复
              </button>
              <button
                onClick={resetSim}
                className="flex items-center gap-2 px-4 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition-colors text-sm"
              >
                <RotateCcw className="w-4 h-4" />
                重置
              </button>
            </div>

            {/* Recovery animation */}
            {recoveryStep >= 0 && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                className="mt-4 pt-4 border-t border-slate-200 dark:border-gray-700"
              >
                <h4 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3">
                  恢复过程
                </h4>
                <div className="space-y-2">
                  {selected.journalCommitted ? (
                    <>
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: recoveryStep >= 0 ? 1 : 0.3 }}
                        className="flex items-center gap-2 text-sm"
                      >
                        <span className="w-5 h-5 rounded-full bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center text-[10px] text-amber-600 dark:text-amber-400">
                          1
                        </span>
                        <span className="text-slate-700 dark:text-gray-200">
                          read_head() — 读取日志头
                        </span>
                        {recoveryStep >= 0 && (
                          <CheckCircle className="w-4 h-4 text-emerald-500 ml-auto" />
                        )}
                      </motion.div>
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: recoveryStep >= 1 ? 1 : 0.3 }}
                        className="flex items-center gap-2 text-sm"
                      >
                        <span className="w-5 h-5 rounded-full bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center text-[10px] text-amber-600 dark:text-amber-400">
                          2
                        </span>
                        <span className="text-slate-700 dark:text-gray-200">
                          install_trans() — 将日志块复制到实际位置
                        </span>
                        {recoveryStep >= 1 && (
                          <CheckCircle className="w-4 h-4 text-emerald-500 ml-auto" />
                        )}
                      </motion.div>
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: recoveryStep >= 2 ? 1 : 0.3 }}
                        className="flex items-center gap-2 text-sm"
                      >
                        <span className="w-5 h-5 rounded-full bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center text-[10px] text-amber-600 dark:text-amber-400">
                          3
                        </span>
                        <span className="text-slate-700 dark:text-gray-200">
                          write_head() — 清空日志头（n=0）
                        </span>
                        {recoveryStep >= 2 && (
                          <CheckCircle className="w-4 h-4 text-emerald-500 ml-auto" />
                        )}
                      </motion.div>
                    </>
                  ) : (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="flex items-center gap-2 text-sm"
                    >
                      <CheckCircle className="w-4 h-4 text-emerald-500" />
                      <span className="text-slate-700 dark:text-gray-200">
                        无需恢复 — 日志为空或未提交，文件系统保持原样
                      </span>
                    </motion.div>
                  )}
                </div>
              </motion.div>
            )}
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-slate-200 dark:border-gray-700 text-center"
          >
            <AlertTriangle className="w-8 h-8 text-slate-400 mx-auto mb-2" />
            <p className="text-sm text-slate-500 dark:text-gray-400">
              点击上方时间线选择一个崩溃时间点，查看恢复过程。
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
