"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertTriangle,
  Play,
  RotateCcw,
  Shield,
  Zap,
  Lock,
  Unlock,
} from "lucide-react";

type Phase = "idle" | "filling" | "overflowing" | "hijacked" | "blocked";

interface StackCell {
  address: string;
  label: string;
  value: string;
  isOverflowed: boolean;
  isCanary: boolean;
  isReturnAddr: boolean;
  color: string;
}

function createStackCells(canaryEnabled: boolean, nxEnabled: boolean): StackCell[] {
  const cells: StackCell[] = [
    { address: "0x7fff0018", label: "调用者栈帧", value: "...", isOverflowed: false, isCanary: false, isReturnAddr: false, color: "bg-slate-200 dark:bg-slate-600" },
    { address: "0x7fff0010", label: "返回地址", value: "0x4011a0", isOverflowed: false, isCanary: false, isReturnAddr: true, color: "bg-orange-200 dark:bg-orange-800" },
  ];

  if (canaryEnabled) {
    cells.push({
      address: "0x7fff0008",
      label: "Stack Canary",
      value: "0x5a3b...",
      isOverflowed: false,
      isCanary: true,
      isReturnAddr: false,
      color: "bg-emerald-200 dark:bg-emerald-800",
    });
  }

  cells.push({ address: "0x7fff0000", label: "saved RBP", value: "0x7fff0020", isOverflowed: false, isCanary: false, isReturnAddr: false, color: "bg-slate-200 dark:bg-slate-600" });

  for (let i = 15; i >= 0; i--) {
    const offset = i * 4;
    cells.push({
      address: `0x7fff${(0xf000 - offset).toString(16).padStart(4, "0")}`,
      label: `buffer[${i * 4}..${i * 4 + 3}]`,
      value: "0x00000000",
      isOverflowed: false,
      isCanary: false,
      isReturnAddr: false,
      color: "bg-blue-100 dark:bg-blue-900/40",
    });
  }

  return cells;
}

export default function BufferOverflowDemo() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [canaryEnabled, setCanaryEnabled] = useState(false);
  const [nxEnabled, setNxEnabled] = useState(false);
  const [cells, setCells] = useState<StackCell[]>(createStackCells(false, false));
  const [currentFillIndex, setCurrentFillIndex] = useState(0);
  const [inputSize, setInputSize] = useState(64);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const reset = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setPhase("idle");
    setCells(createStackCells(canaryEnabled, nxEnabled));
    setCurrentFillIndex(0);
  }, [canaryEnabled, nxEnabled]);

  useEffect(() => {
    reset();
  }, [canaryEnabled, nxEnabled, reset]);

  const startAttack = () => {
    reset();
    const newCells = createStackCells(canaryEnabled, nxEnabled);
    setCells(newCells);
    setPhase("filling");
    setCurrentFillIndex(0);

    let idx = 0;
    const bufferSize = 16;
    const totalToWrite = Math.ceil(inputSize / 4);

    intervalRef.current = setInterval(() => {
      idx++;
      setCurrentFillIndex(idx);

      setCells((prev) => {
        const updated = [...prev];
        const cellIdx = updated.length - 1 - Math.floor(idx / 1);

        if (idx <= bufferSize) {
          // Filling buffer
          const targetIdx = updated.length - 1 - (idx - 1);
          if (targetIdx >= 0) {
            updated[targetIdx] = {
              ...updated[targetIdx],
              value: `0x41414141`,
              color: "bg-yellow-200 dark:bg-yellow-800",
            };
          }
        } else if (idx === bufferSize + 1 && canaryEnabled) {
          // Overwriting canary
          const canaryIdx = updated.findIndex((c) => c.isCanary);
          if (canaryIdx >= 0) {
            updated[canaryIdx] = {
              ...updated[canaryIdx],
              value: "0x41414141",
              isOverflowed: true,
              color: "bg-red-300 dark:bg-red-700",
            };
          }
          setPhase("overflowing");
        } else if (
          idx > bufferSize + (canaryEnabled ? 1 : 0) &&
          idx <= totalToWrite
        ) {
          // Overflowing past buffer
          setPhase("overflowing");
          const offset = idx - bufferSize - (canaryEnabled ? 1 : 0);
          const targetIdx = updated.length - 1 - bufferSize - offset;
          if (targetIdx >= 0 && !updated[targetIdx].isReturnAddr) {
            updated[targetIdx] = {
              ...updated[targetIdx],
              value: "0x41414141",
              isOverflowed: true,
              color: "bg-red-200 dark:bg-red-800",
            };
          }
        }

        // Overwrite return address
        if (idx === totalToWrite + 1) {
          const retIdx = updated.findIndex((c) => c.isReturnAddr);
          if (retIdx >= 0) {
            if (canaryEnabled) {
              // Canary detected!
              setPhase("blocked");
              if (intervalRef.current) clearInterval(intervalRef.current);
            } else {
              updated[retIdx] = {
                ...updated[retIdx],
                value: "0x90909090",
                isOverflowed: true,
                color: "bg-red-400 dark:bg-red-600",
              };
              setPhase("hijacked");
              if (intervalRef.current) clearInterval(intervalRef.current);
            }
          } else {
            setPhase("hijacked");
            if (intervalRef.current) clearInterval(intervalRef.current);
          }
        }

        return updated;
      });

      if (idx > totalToWrite + 2) {
        if (intervalRef.current) clearInterval(intervalRef.current);
      }
    }, 200);
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <div className="flex items-center gap-3 mb-4">
        <Zap className="w-8 h-8 text-orange-600 dark:text-orange-400" />
        <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
          缓冲区溢出攻击演示
        </h3>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 mb-6">
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-slate-600 dark:text-slate-300">
            输入大小:
          </label>
          <input
            type="range"
            min={16}
            max={128}
            step={4}
            value={inputSize}
            onChange={(e) => setInputSize(Number(e.target.value))}
            className="w-32"
            disabled={phase !== "idle"}
          />
          <span className="text-sm font-mono text-slate-500">{inputSize} bytes</span>
        </div>

        <button
          onClick={() => setCanaryEnabled(!canaryEnabled)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
            canaryEnabled
              ? "bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-700"
              : "bg-white dark:bg-slate-700 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-600"
          }`}
          disabled={phase !== "idle"}
        >
          {canaryEnabled ? <Lock className="w-4 h-4" /> : <Unlock className="w-4 h-4" />}
          Stack Canary
        </button>

        <button
          onClick={() => setNxEnabled(!nxEnabled)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
            nxEnabled
              ? "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 border border-blue-300 dark:border-blue-700"
              : "bg-white dark:bg-slate-700 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-600"
          }`}
          disabled={phase !== "idle"}
        >
          <Shield className="w-4 h-4" />
          DEP/NX
        </button>

        <button
          onClick={startAttack}
          disabled={phase !== "idle"}
          className="flex items-center gap-1.5 px-4 py-1.5 bg-red-600 hover:bg-red-700 disabled:bg-slate-400 text-white rounded-lg text-sm font-medium transition-all"
        >
          <Play className="w-4 h-4" />
          发起攻击
        </button>

        <button
          onClick={reset}
          className="flex items-center gap-1.5 px-4 py-1.5 bg-slate-600 hover:bg-slate-700 text-white rounded-lg text-sm font-medium transition-all"
        >
          <RotateCcw className="w-4 h-4" />
          重置
        </button>
      </div>

      {/* Status */}
      <AnimatePresence mode="wait">
        {phase !== "idle" && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-4"
          >
            <div
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium ${
                phase === "filling"
                  ? "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300"
                  : phase === "overflowing"
                  ? "bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300"
                  : phase === "hijacked"
                  ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300"
                  : "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300"
              }`}
            >
              {phase === "filling" && "📝 填充缓冲区中..."}
              {phase === "overflowing" && (
                <span className="flex items-center gap-1">
                  <AlertTriangle className="w-4 h-4" /> 溢出发生！覆盖相邻内存
                </span>
              )}
              {phase === "hijacked" && (
                <span className="flex items-center gap-1">
                  <Zap className="w-4 h-4" /> 返回地址被覆盖 → 执行流被劫持！
                </span>
              )}
              {phase === "blocked" && (
                <span className="flex items-center gap-1">
                  <Shield className="w-4 h-4" /> Canary 检测到溢出 → __stack_chk_fail!
                </span>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Stack Visualization */}
      <div className="bg-white dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
            高地址 ↑
          </span>
          <div className="flex-1 h-px bg-slate-200 dark:bg-slate-600" />
          <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
            栈布局
          </span>
          <div className="flex-1 h-px bg-slate-200 dark:bg-slate-600" />
          <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
            ↓ 低地址
          </span>
        </div>

        <div className="space-y-1">
          {cells.map((cell, i) => (
            <motion.div
              key={i}
              animate={{
                scale: cell.isOverflowed ? [1, 1.02, 1] : 1,
              }}
              transition={{ duration: 0.3 }}
              className={`flex items-center gap-3 px-3 py-1.5 rounded text-xs font-mono transition-colors duration-300 ${cell.color} ${
                cell.isReturnAddr && cell.isOverflowed
                  ? "ring-2 ring-red-500 ring-offset-1 dark:ring-offset-slate-800"
                  : ""
              }`}
            >
              <span className="w-20 text-slate-500 dark:text-slate-400">
                {cell.address}
              </span>
              <span className="w-28 text-slate-600 dark:text-slate-300 font-medium">
                {cell.label}
              </span>
              <span
                className={`flex-1 ${
                  cell.isOverflowed
                    ? "text-red-600 dark:text-red-400 font-bold"
                    : "text-slate-700 dark:text-slate-200"
                }`}
              >
                {cell.value}
              </span>
              {cell.isReturnAddr && (
                <span className="text-orange-500 dark:text-orange-400 text-xs">
                  ← 返回地址
                </span>
              )}
              {cell.isCanary && (
                <span className="text-emerald-500 dark:text-emerald-400 text-xs">
                  ← 保护哨兵
                </span>
              )}
            </motion.div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4 text-xs text-slate-500 dark:text-slate-400">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-blue-200 dark:bg-blue-800" />
          <span>buffer 区域</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-yellow-200 dark:bg-yellow-800" />
          <span>正在写入</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-red-300 dark:bg-red-700" />
          <span>溢出覆盖</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-orange-200 dark:bg-orange-800" />
          <span>返回地址</span>
        </div>
        {canaryEnabled && (
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-emerald-200 dark:bg-emerald-800" />
            <span>Stack Canary</span>
          </div>
        )}
      </div>
    </div>
  );
}
