"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, Info, Play, RotateCcw } from "lucide-react";

const NDIRECT = 12;
const PTRS_PER_BLOCK = 256; // 1024 / 4

export default function Xv6BmapVisualizer() {
  const [logicalBlock, setLogicalBlock] = useState(0);
  const [animating, setAnimating] = useState(false);
  const [resolvedPath, setResolvedPath] = useState<string[]>([]);

  const maxBlock = NDIRECT + PTRS_PER_BLOCK + PTRS_PER_BLOCK * PTRS_PER_BLOCK;

  const getBlockType = (bn: number) => {
    if (bn < NDIRECT) return "direct";
    if (bn < NDIRECT + PTRS_PER_BLOCK) return "single";
    return "double";
  };

  const resolve = () => {
    setAnimating(true);
    setResolvedPath([]);

    const bn = logicalBlock;
    const steps: string[] = [];

    if (bn < NDIRECT) {
      steps.push(`bn=${bn} < NDIRECT(${NDIRECT}) → 直接指针`);
      steps.push(`读取 ip->addrs[${bn}] → 物理块号`);
      steps.push(`返回物理块号（如果为 0 则 balloc 分配新块）`);
    } else if (bn < NDIRECT + PTRS_PER_BLOCK) {
      const idx = bn - NDIRECT;
      steps.push(`bn=${bn} ≥ NDIRECT → 进入单间接`);
      steps.push(`bn -= NDIRECT → bn=${idx}`);
      steps.push(`读取 ip->addrs[${NDIRECT}]（间接指针块）`);
      steps.push(`在间接块中查找 a[${idx}] → 物理块号`);
    } else {
      const idx = bn - NDIRECT - PTRS_PER_BLOCK;
      const outer = Math.floor(idx / PTRS_PER_BLOCK);
      const inner = idx % PTRS_PER_BLOCK;
      steps.push(`bn=${bn} ≥ NDIRECT+NINDIRECT → 进入双间接`);
      steps.push(`bn -= NDIRECT + NINDIRECT → bn=${idx}`);
      steps.push(`外层索引: ${outer}，内层索引: ${inner}`);
      steps.push(`读取 ip->addrs[${NDIRECT + 1}]（双间接块）`);
      steps.push(`在外层块中查找 a[${outer}] → 第二级间接块`);
      steps.push(`在第二级块中查找 a[${inner}] → 物理块号`);
    }

    let i = 0;
    const interval = setInterval(() => {
      if (i < steps.length) {
        setResolvedPath((prev) => [...prev, steps[i]]);
        i++;
      } else {
        clearInterval(interval);
        setAnimating(false);
      }
    }, 800);
  };

  const reset = () => {
    setResolvedPath([]);
    setAnimating(false);
  };

  const blockType = getBlockType(logicalBlock);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        bmap() — 逻辑块号 → 物理块号
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        拖动滑块选择逻辑块号，观察 bmap() 如何映射到物理块号
      </p>

      {/* Slider */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-slate-200 dark:border-gray-700 mb-6">
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-bold text-slate-700 dark:text-gray-200">
            逻辑块号 (bn)
          </label>
          <span className="text-lg font-mono font-bold text-indigo-600 dark:text-indigo-400">
            {logicalBlock}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={Math.min(maxBlock - 1, 1000)}
          value={logicalBlock}
          onChange={(e) => {
            setLogicalBlock(parseInt(e.target.value));
            setResolvedPath([]);
          }}
          className="w-full h-2 bg-slate-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-[10px] text-slate-400 dark:text-gray-500 font-mono mt-1">
          <span>0 (直接)</span>
          <span>{NDIRECT} (单间接)</span>
          <span>{NDIRECT + PTRS_PER_BLOCK} (双间接)</span>
        </div>

        {/* Type indicator */}
        <div className="mt-3 flex items-center gap-3">
          <span
            className={`px-3 py-1 rounded-full text-xs font-bold ${
              blockType === "direct"
                ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                : blockType === "single"
                ? "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300"
                : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300"
            }`}
          >
            {blockType === "direct"
              ? "直接指针"
              : blockType === "single"
              ? "单间接指针"
              : "双间接指针"}
          </span>
          <span className="text-xs text-slate-500 dark:text-gray-400">
            {blockType === "direct"
              ? `ip->addrs[${logicalBlock}]`
              : blockType === "single"
              ? `ip->addrs[12] → a[${logicalBlock - NDIRECT}]`
              : `ip->addrs[13] → ...`}
          </span>
        </div>
      </div>

      {/* Pointer structure visualization */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-slate-200 dark:border-gray-700 mb-6">
        <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3">
          Inode 指针结构
        </h3>
        <div className="flex flex-wrap gap-2 mb-3">
          {Array.from({ length: NDIRECT }, (_, i) => (
            <div
              key={i}
              className={`w-10 h-10 rounded flex items-center justify-center text-xs font-mono border-2 transition-all ${
                blockType === "direct" && i === logicalBlock
                  ? "border-emerald-500 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 shadow-md"
                  : "border-slate-200 dark:border-gray-600 bg-slate-50 dark:bg-gray-750 text-slate-500 dark:text-gray-400"
              }`}
            >
              {i}
            </div>
          ))}
          <div
            className={`w-10 h-10 rounded flex items-center justify-center text-xs font-mono border-2 ${
              blockType === "single"
                ? "border-amber-500 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 shadow-md"
                : "border-slate-200 dark:border-gray-600 bg-slate-50 dark:bg-gray-750 text-slate-500 dark:text-gray-400"
            }`}
          >
            12
          </div>
          <div
            className={`w-10 h-10 rounded flex items-center justify-center text-xs font-mono border-2 ${
              blockType === "double"
                ? "border-red-500 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 shadow-md"
                : "border-slate-200 dark:border-gray-600 bg-slate-50 dark:bg-gray-750 text-slate-500 dark:text-gray-400"
            }`}
          >
            13
          </div>
        </div>
        <div className="flex gap-4 text-xs text-slate-500 dark:text-gray-400">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-emerald-100 border border-emerald-400" />
            直接 [0-11]: 12 KB
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-amber-100 border border-amber-400" />
            单间接 [12]: 256 KB
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-red-100 border border-red-400" />
            双间接 [13]: 64 MB
          </span>
        </div>
      </div>

      {/* Resolve button and path */}
      <div className="flex gap-3 mb-4 justify-center">
        <button
          onClick={resolve}
          disabled={animating}
          className="flex items-center gap-2 px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
        >
          <Play className="w-4 h-4" />
          解析映射
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-4 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition-colors text-sm"
        >
          <RotateCcw className="w-4 h-4" />
          重置
        </button>
      </div>

      {resolvedPath.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700"
        >
          <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3">
            bmap() 解析过程
          </h3>
          <div className="space-y-2">
            {resolvedPath.map((step, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex items-start gap-2"
              >
                <span className="text-xs font-mono text-indigo-500 dark:text-indigo-400 mt-0.5 shrink-0">
                  [{i + 1}]
                </span>
                <span className="text-sm font-mono text-slate-700 dark:text-gray-200">
                  {step}
                </span>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Info */}
      <div className="mt-4 flex items-start gap-2 text-xs text-slate-500 dark:text-gray-400">
        <Info className="w-4 h-4 shrink-0 mt-0.5" />
        <span>
          xv6: BSIZE=1024, 指针 4 字节, 每块 256 个指针。最大文件 ≈ 12KB + 256KB + 64MB ≈ 64MB。
        </span>
      </div>
    </div>
  );
}
