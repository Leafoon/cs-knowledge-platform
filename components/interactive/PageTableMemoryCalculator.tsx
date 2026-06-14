"use client";
import React, { useState, useMemo } from "react";
import { Calculator, HardDrive } from "lucide-react";

interface Preset {
  name: string;
  vaBits: number;
  pageSize: number;
  pteSize: number;
  levels: number;
}

const presets: Preset[] = [
  { name: "x86 (32-bit)", vaBits: 32, pageSize: 4096, pteSize: 4, levels: 2 },
  { name: "x86-64 (4KB)", vaBits: 48, pageSize: 4096, pteSize: 8, levels: 4 },
  { name: "x86-64 (2MB)", vaBits: 48, pageSize: 2097152, pteSize: 8, levels: 3 },
  { name: "RISC-V Sv39", vaBits: 39, pageSize: 4096, pteSize: 8, levels: 3 },
  { name: "ARMv8 (4KB)", vaBits: 48, pageSize: 4096, pteSize: 8, levels: 4 },
];

export default function PageTableMemoryCalculator() {
  const [selected, setSelected] = useState(0);
  const preset = presets[selected];

  const result = useMemo(() => {
    const pageBits = Math.log2(preset.pageSize);
    const totalVpnBits = preset.vaBits - pageBits;
    const entriesPerLevel = 512; // 9 bits per level for 4KB pages
    const entriesPerTable = Math.pow(2, Math.min(9, totalVpnBits / preset.levels));

    // Linear page table
    const linearEntries = Math.pow(2, totalVpnBits);
    const linearSize = linearEntries * preset.pteSize;

    // Multi-level page table (worst case: all entries used)
    const multiLevelSizeWorst = (() => {
      let total = 0;
      let tables = 1;
      for (let i = 0; i < preset.levels; i++) {
        tables *= entriesPerTable;
        total += tables * entriesPerTable * preset.pteSize;
      }
      return total;
    })();

    // Multi-level (typical: only root + a few leaf tables)
    const rootSize = entriesPerTable * preset.pteSize;
    const leafSize = entriesPerTable * preset.pteSize;
    const typicalSize = rootSize + 4 * leafSize; // root + 4 leaf tables

    return {
      linearSize,
      multiLevelSizeWorst,
      typicalSize,
      entriesPerTable,
      savings: linearSize > 0 ? ((1 - typicalSize / linearSize) * 100) : 0,
    };
  }, [preset]);

  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center flex items-center justify-center gap-2">
        <Calculator className="w-6 h-6 text-emerald-600" />
        页表内存开销计算器
      </h3>

      {/* Preset selector */}
      <div className="flex flex-wrap justify-center gap-2 mb-6">
        {presets.map((p, i) => (
          <button
            key={i}
            onClick={() => setSelected(i)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              selected === i
                ? "bg-emerald-600 text-white"
                : "bg-white dark:bg-gray-700 text-slate-600 dark:text-gray-300 hover:bg-slate-100 dark:hover:bg-gray-600"
            }`}
          >
            {p.name}
          </button>
        ))}
      </div>

      {/* Parameters */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {[
          { label: "虚拟地址位数", value: `${preset.vaBits} 位` },
          { label: "页大小", value: formatSize(preset.pageSize) },
          { label: "PTE 大小", value: `${preset.pteSize} 字节` },
          { label: "页表级数", value: `${preset.levels} 级` },
        ].map((item) => (
          <div key={item.label} className="bg-white dark:bg-gray-800 rounded-lg p-3 text-center shadow-sm">
            <div className="text-xs text-slate-500 dark:text-gray-400">{item.label}</div>
            <div className="text-lg font-bold text-slate-800 dark:text-gray-100">{item.value}</div>
          </div>
        ))}
      </div>

      {/* Results */}
      <div className="space-y-3">
        {/* Linear page table */}
        <div className="bg-red-50 dark:bg-red-900/30 rounded-lg p-4">
          <div className="flex justify-between items-center">
            <span className="text-sm font-semibold text-red-700 dark:text-red-300">线性页表大小</span>
            <span className="text-lg font-bold text-red-800 dark:text-red-200">{formatSize(result.linearSize)}</span>
          </div>
          <div className="w-full bg-red-200 dark:bg-red-800 rounded-full h-3 mt-2">
            <div className="bg-red-500 h-3 rounded-full" style={{ width: "100%" }} />
          </div>
        </div>

        {/* Multi-level (worst case) */}
        <div className="bg-amber-50 dark:bg-amber-900/30 rounded-lg p-4">
          <div className="flex justify-between items-center">
            <span className="text-sm font-semibold text-amber-700 dark:text-amber-300">多级页表（最坏情况）</span>
            <span className="text-lg font-bold text-amber-800 dark:text-amber-200">{formatSize(result.multiLevelSizeWorst)}</span>
          </div>
          <div className="w-full bg-amber-200 dark:bg-amber-800 rounded-full h-3 mt-2">
            <div
              className="bg-amber-500 h-3 rounded-full"
              style={{ width: `${Math.min(100, (result.multiLevelSizeWorst / result.linearSize) * 100)}%` }}
            />
          </div>
        </div>

        {/* Multi-level (typical) */}
        <div className="bg-emerald-50 dark:bg-emerald-900/30 rounded-lg p-4">
          <div className="flex justify-between items-center">
            <span className="text-sm font-semibold text-emerald-700 dark:text-emerald-300">多级页表（典型进程）</span>
            <span className="text-lg font-bold text-emerald-800 dark:text-emerald-200">{formatSize(result.typicalSize)}</span>
          </div>
          <div className="w-full bg-emerald-200 dark:bg-emerald-800 rounded-full h-3 mt-2">
            <div
              className="bg-emerald-500 h-3 rounded-full"
              style={{ width: `${Math.max(1, (result.typicalSize / result.linearSize) * 100)}%` }}
            />
          </div>
          <div className="text-xs text-emerald-600 dark:text-emerald-400 mt-1">
            节省 {result.savings.toFixed(1)}% 空间（相比线性页表）
          </div>
        </div>
      </div>

      {/* Key insight */}
      <div className="mt-4 bg-emerald-50 dark:bg-emerald-900/30 border border-emerald-200 dark:border-emerald-700 rounded-lg p-3">
        <p className="text-xs text-emerald-700 dark:text-emerald-300">
          <strong>关键洞察：</strong>多级页表只为使用的虚拟地址区域分配页表空间。典型进程只使用少量虚拟地址区域（代码、数据、堆、栈），因此多级页表的开销远小于线性页表。代价是地址翻译需要更多次内存访问，但 TLB 使这个代价几乎不可见。
        </p>
      </div>
    </div>
  );
}
