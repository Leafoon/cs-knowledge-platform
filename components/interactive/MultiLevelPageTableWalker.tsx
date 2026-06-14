"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Database, ChevronRight, Search } from "lucide-react";

interface PageTableEntry {
  index: number;
  value: string;
  present: boolean;
}

interface PageTableLevel {
  name: string;
  bits: string;
  entries: PageTableEntry[];
}

export default function MultiLevelPageTableWalker() {
  const [virtualAddress, setVirtualAddress] = useState<string>("0x00007FFF12345678");
  const [currentLevel, setCurrentLevel] = useState<number>(0);
  const [isWalking, setIsWalking] = useState<boolean>(false);

  // 解析虚拟地址
  const parseAddress = (addr: string): { [key: string]: string } => {
    const numeric = parseInt(addr, 16);
    return {
      pml4: ((numeric >> 39) & 0x1FF).toString(16).padStart(3, "0"),
      pdpt: ((numeric >> 30) & 0x1FF).toString(16).padStart(3, "0"),
      pd: ((numeric >> 21) & 0x1FF).toString(16).padStart(3, "0"),
      pt: ((numeric >> 12) & 0x1FF).toString(16).padStart(3, "0"),
      offset: (numeric & 0xFFF).toString(16).padStart(3, "0")
    };
  };

  const parts = parseAddress(virtualAddress);

  const levels: PageTableLevel[] = [
    {
      name: "PML4 (Level 4)",
      bits: "47:39 (9 bits)",
      entries: [
        { index: parseInt(parts.pml4, 16), value: "0x3F000000", present: true },
        { index: 100, value: "0x00000000", present: false },
        { index: 200, value: "0x3F001000", present: true }
      ]
    },
    {
      name: "PDPT (Level 3)",
      bits: "38:30 (9 bits)",
      entries: [
        { index: parseInt(parts.pdpt, 16), value: "0x2E000000", present: true },
        { index: 50, value: "0x2E001000", present: true },
        { index: 150, value: "0x00000000", present: false }
      ]
    },
    {
      name: "PD (Level 2)",
      bits: "29:21 (9 bits)",
      entries: [
        { index: parseInt(parts.pd, 16), value: "0x1D000000", present: true },
        { index: 75, value: "0x1D001000", present: true },
        { index: 175, value: "0x00000000", present: false }
      ]
    },
    {
      name: "PT (Level 1)",
      bits: "20:12 (9 bits)",
      entries: [
        { index: parseInt(parts.pt, 16), value: "0x0C000000", present: true },
        { index: 128, value: "0x0C001000", present: true },
        { index: 256, value: "0x00000000", present: false }
      ]
    }
  ];

  const startWalk = () => {
    setIsWalking(true);
    setCurrentLevel(0);
    walkLevels(0);
  };

  const walkLevels = (level: number) => {
    if (level >= 4) {
      setIsWalking(false);
      return;
    }
    setTimeout(() => {
      setCurrentLevel(level + 1);
      walkLevels(level + 1);
    }, 1500);
  };

  const reset = () => {
    setCurrentLevel(0);
    setIsWalking(false);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-teal-50 to-teal-100 dark:from-teal-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <div className="flex items-center gap-3 mb-6">
        <Database className="w-8 h-8 text-teal-600 dark:text-teal-400" />
        <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
          四级页表遍历器 (x86-64)
        </h3>
      </div>

      {/* 虚拟地址输入 */}
      <div className="mb-6 p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg">
        <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
          虚拟地址 (64-bit)
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={virtualAddress}
            onChange={(e) => setVirtualAddress(e.target.value)}
            disabled={isWalking}
            className="flex-1 px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 font-mono disabled:opacity-50"
            placeholder="0x00007FFF12345678"
          />
          <motion.button
            whileHover={{ scale: 1.05 }}
            onClick={startWalk}
            disabled={isWalking}
            className="px-6 py-2 bg-teal-600 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed font-semibold flex items-center gap-2"
          >
            <Search className="w-4 h-4" />
            遍历
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            onClick={reset}
            className="px-6 py-2 bg-slate-600 text-white rounded-lg font-semibold"
          >
            重置
          </motion.button>
        </div>
      </div>

      {/* 地址解析 */}
      <div className="mb-6 p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg">
        <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4">地址位分解</h4>
        <div className="grid grid-cols-5 gap-2 font-mono text-sm">
          <div className="p-3 bg-purple-100 dark:bg-purple-900/30 rounded text-center">
            <div className="text-xs text-purple-600 dark:text-purple-400 mb-1">PML4</div>
            <div className="font-bold text-purple-700 dark:text-purple-300">0x{parts.pml4}</div>
          </div>
          <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded text-center">
            <div className="text-xs text-blue-600 dark:text-blue-400 mb-1">PDPT</div>
            <div className="font-bold text-blue-700 dark:text-blue-300">0x{parts.pdpt}</div>
          </div>
          <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded text-center">
            <div className="text-xs text-green-600 dark:text-green-400 mb-1">PD</div>
            <div className="font-bold text-green-700 dark:text-green-300">0x{parts.pd}</div>
          </div>
          <div className="p-3 bg-orange-100 dark:bg-orange-900/30 rounded text-center">
            <div className="text-xs text-orange-600 dark:text-orange-400 mb-1">PT</div>
            <div className="font-bold text-orange-700 dark:text-orange-300">0x{parts.pt}</div>
          </div>
          <div className="p-3 bg-pink-100 dark:bg-pink-900/30 rounded text-center">
            <div className="text-xs text-pink-600 dark:text-pink-400 mb-1">Offset</div>
            <div className="font-bold text-pink-700 dark:text-pink-300">0x{parts.offset}</div>
          </div>
        </div>
      </div>

      {/* 页表遍历可视化 */}
      <div className="space-y-4">
        {levels.map((level, levelIndex) => (
          <motion.div
            key={levelIndex}
            initial={{ opacity: 0.5, x: -20 }}
            animate={{
              opacity: currentLevel > levelIndex ? 1 : 0.5,
              x: currentLevel > levelIndex ? 0 : -20,
              scale: currentLevel === levelIndex + 1 ? 1.02 : 1
            }}
            className={`
              p-6 rounded-lg transition-all
              ${currentLevel === levelIndex + 1
                ? "bg-gradient-to-r from-teal-500 to-teal-600 text-white shadow-lg"
                : "bg-white dark:bg-slate-800"
              }
            `}
          >
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className={`font-bold text-lg ${
                  currentLevel === levelIndex + 1 ? "text-white" : "text-slate-800 dark:text-slate-200"
                }`}>
                  {level.name}
                </h4>
                <p className={`text-sm ${
                  currentLevel === levelIndex + 1 ? "text-teal-100" : "text-slate-600 dark:text-slate-400"
                }`}>
                  索引位: {level.bits}
                </p>
              </div>
              {currentLevel > levelIndex && (
                <ChevronRight className={`w-6 h-6 ${
                  currentLevel === levelIndex + 1 ? "text-white" : "text-teal-600"
                }`} />
              )}
            </div>
            <div className="grid grid-cols-3 gap-2">
              {level.entries.map((entry, i) => (
                <div
                  key={i}
                  className={`
                    p-3 rounded text-xs
                    ${currentLevel === levelIndex + 1
                      ? "bg-white/20 text-white"
                      : entry.present
                      ? "bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300"
                      : "bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300"
                    }
                  `}
                >
                  <div className="font-semibold">Index: {entry.index}</div>
                  <div className="font-mono">{entry.value}</div>
                  <div className="text-xs mt-1">
                    {entry.present ? "✓ Present" : "✗ Not Present"}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        ))}
      </div>

      {/* 最终物理地址 */}
      {currentLevel >= 4 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-6 bg-gradient-to-r from-green-500 to-green-600 rounded-lg shadow-lg text-white"
        >
          <h4 className="font-bold text-lg mb-2">转换完成!</h4>
          <div className="flex items-center gap-4">
            <div>
              <div className="text-sm opacity-90">物理地址</div>
              <div className="text-2xl font-mono font-bold">0x0C000{parts.offset}</div>
            </div>
            <div className="flex-1 text-sm opacity-90">
              = 页帧基址 (0x0C000000) + 页内偏移 (0x{parts.offset})
            </div>
          </div>
        </motion.div>
      )}

      {/* 说明 */}
      <div className="mt-6 p-4 bg-teal-50 dark:bg-teal-900/20 rounded-lg border border-teal-200 dark:border-teal-800">
        <p className="text-sm text-teal-900 dark:text-teal-100">
          <strong>四级页表：</strong> x86-64 使用 4 级页表结构 (PML4 → PDPT → PD → PT)，
          将 48 位虚拟地址分为 5 部分：每级索引各 9 位 (512 个表项)，页内偏移 12 位 (4KB 页)。
          MMU 从 CR3 寄存器获取 PML4 基址，依次查找 4 级页表，最终得到物理页帧号，加上偏移得到物理地址。
        </p>
      </div>
    </div>
  );
}
