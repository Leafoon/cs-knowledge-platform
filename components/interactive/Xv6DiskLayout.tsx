"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Info, Play, RotateCcw } from "lucide-react";

interface DiskBlock {
  id: string;
  label: string;
  labelZh: string;
  range: string;
  color: string;
  bgColor: string;
  borderColor: string;
  description: string;
  details: string[];
}

const diskBlocks: DiskBlock[] = [
  {
    id: "boot",
    label: "Boot Block",
    labelZh: "引导块",
    range: "Block 0",
    color: "text-slate-700 dark:text-slate-300",
    bgColor: "bg-slate-100 dark:bg-slate-800",
    borderColor: "border-slate-400 dark:border-slate-600",
    description: "引导加载程序。如果分区可引导，包含启动代码；否则为空。",
    details: ["块号：0", "大小：1024 字节", "通常未使用"],
  },
  {
    id: "super",
    label: "Superblock",
    labelZh: "超级块",
    range: "Block 1",
    color: "text-purple-700 dark:text-purple-300",
    bgColor: "bg-purple-50 dark:bg-purple-950/40",
    borderColor: "border-purple-400 dark:border-purple-600",
    description: "存储文件系统元数据：魔数、总块数、inode 数、日志块数等。由 mkfs 在创建文件系统时写入。",
    details: [
      "magic: 0x10203040",
      "size: 文件系统总块数",
      "nblocks: 数据块数",
      "ninodes: inode 数",
      "nlog: 日志块数",
      "logstart: 日志起始块号",
      "inodestart: inode 起始块号",
      "bmapstart: 位图起始块号",
    ],
  },
  {
    id: "log",
    label: "Log Blocks",
    labelZh: "日志块",
    range: "Block 2 ~ N",
    color: "text-amber-700 dark:text-amber-300",
    bgColor: "bg-amber-50 dark:bg-amber-950/40",
    borderColor: "border-amber-400 dark:border-amber-600",
    description: "写前日志（WAL）区域。每个事务的修改先写入此区域，提交后再复制到实际位置，保证崩溃一致性。",
    details: [
      "第一个块：日志头块",
      "记录事务中修改的块号",
      "其余块：日志数据块",
      "commit 时重放到实际位置",
    ],
  },
  {
    id: "inode",
    label: "Inode Blocks",
    labelZh: "inode 块",
    range: "Block N+1 ~ M",
    color: "text-emerald-700 dark:text-emerald-300",
    bgColor: "bg-emerald-50 dark:bg-emerald-950/40",
    borderColor: "border-emerald-400 dark:border-emerald-600",
    description: "存储所有 inode 的磁盘结构（dinode）。每个 dinode 64 字节，一个块可存储 16 个 inode。",
    details: [
      "每个 dinode 64 字节",
      "每块存 16 个 inode",
      "包含 type, nlink, size, addrs[]",
      "inode 号 = 块内偏移",
    ],
  },
  {
    id: "bitmap",
    label: "Bitmap Blocks",
    labelZh: "位图块",
    range: "Block M+1 ~ M+K",
    color: "text-blue-700 dark:text-blue-300",
    bgColor: "bg-blue-50 dark:bg-blue-950/40",
    borderColor: "border-blue-400 dark:border-blue-600",
    description: "追踪哪些数据块是空闲的。每个位对应一个数据块：0 = 空闲，1 = 已分配。",
    details: [
      "每位对应一个数据块",
      "0 = 空闲，1 = 已分配",
      "balloc() 分配块时置 1",
      "bfree() 释放块时置 0",
    ],
  },
  {
    id: "data",
    label: "Data Blocks",
    labelZh: "数据块",
    range: "Block M+K+1 ~ End",
    color: "text-cyan-700 dark:text-cyan-300",
    bgColor: "bg-cyan-50 dark:bg-cyan-950/40",
    borderColor: "border-cyan-400 dark:border-cyan-600",
    description: "存储文件的实际数据和目录的内容。由 inode 的 addrs[] 指针指向。",
    details: [
      "存储文件内容",
      "存储目录条目（dirent）",
      "也存储间接指针块",
      "由 bmap() 映射到逻辑块号",
    ],
  },
];

export default function Xv6DiskLayout() {
  const [selectedBlock, setSelectedBlock] = useState<string | null>(null);
  const [scanIndex, setScanIndex] = useState(-1);
  const [isScanning, setIsScanning] = useState(false);

  const selected = diskBlocks.find((b) => b.id === selectedBlock);

  const startScan = () => {
    if (isScanning) return;
    setIsScanning(true);
    setScanIndex(0);
  };

  React.useEffect(() => {
    if (!isScanning) return;
    const timer = setInterval(() => {
      setScanIndex((prev) => {
        if (prev >= diskBlocks.length - 1) {
          setIsScanning(false);
          return -1;
        }
        return prev + 1;
      });
    }, 600);
    return () => clearInterval(timer);
  }, [isScanning]);

  const resetScan = () => {
    setIsScanning(false);
    setScanIndex(-1);
    setSelectedBlock(null);
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        xv6 磁盘布局
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        块大小 = 1024 字节（BSIZE）。点击各区域查看详情。
      </p>

      <div className="flex gap-3 mb-6 justify-center flex-wrap">
        <button
          onClick={startScan}
          disabled={isScanning}
          className="flex items-center gap-2 px-4 py-2 bg-violet-500 text-white rounded-lg hover:bg-violet-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
        >
          <Play className="w-4 h-4" />
          逐区扫描
        </button>
        <button
          onClick={resetScan}
          className="flex items-center gap-2 px-4 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition-colors text-sm"
        >
          <RotateCcw className="w-4 h-4" />
          重置
        </button>
      </div>

      {/* Disk layout bar */}
      <div className="flex rounded-lg overflow-hidden border border-slate-300 dark:border-gray-600 mb-6 h-16">
        {diskBlocks.map((block, i) => {
          const isScanningNow = scanIndex === i;
          const isSelected = selectedBlock === block.id;
          return (
            <motion.button
              key={block.id}
              onClick={() =>
                setSelectedBlock(selectedBlock === block.id ? null : block.id)
              }
              className={`flex-1 flex flex-col items-center justify-center transition-all border-r last:border-r-0 border-slate-300 dark:border-gray-600 ${block.bgColor} ${
                isSelected
                  ? `ring-2 ring-inset ring-indigo-400 dark:ring-indigo-500`
                  : ""
              } ${isScanningNow ? "ring-2 ring-inset ring-yellow-400" : ""}`}
              whileHover={{ scale: 1.02 }}
            >
              <span className={`text-xs font-bold ${block.color}`}>
                {block.labelZh}
              </span>
              <span className="text-[10px] text-slate-500 dark:text-gray-400 font-mono">
                {block.range}
              </span>
            </motion.button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Block cards */}
        <div className="lg:col-span-2 grid grid-cols-2 sm:grid-cols-3 gap-3">
          {diskBlocks.map((block, i) => {
            const isScanningNow = scanIndex === i;
            return (
              <motion.button
                key={block.id}
                onClick={() =>
                  setSelectedBlock(selectedBlock === block.id ? null : block.id)
                }
                className={`text-left p-4 rounded-lg border-2 transition-all ${block.bgColor} ${
                  selectedBlock === block.id
                    ? `${block.borderColor} shadow-md`
                    : "border-transparent hover:border-slate-300 dark:hover:border-gray-600"
                } ${isScanningNow ? "ring-2 ring-yellow-400" : ""}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.06 }}
                whileHover={{ scale: 1.02 }}
              >
                <div className={`text-sm font-bold ${block.color} mb-1`}>
                  {block.labelZh}
                </div>
                <div className="text-xs font-mono text-slate-500 dark:text-gray-400">
                  {block.label}
                </div>
                <div className="text-[10px] text-slate-400 dark:text-gray-500 font-mono mt-1">
                  {block.range}
                </div>
              </motion.button>
            );
          })}
        </div>

        {/* Detail panel */}
        <div className="lg:col-span-1">
          <AnimatePresence mode="wait">
            {selected ? (
              <motion.div
                key={selected.id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className={`${selected.bgColor} rounded-lg p-5 shadow-md border ${selected.borderColor}`}
              >
                <h3 className={`text-lg font-bold mb-1 ${selected.color}`}>
                  {selected.labelZh}
                </h3>
                <h4 className="text-sm font-mono text-slate-600 dark:text-gray-300 mb-3">
                  {selected.label} ({selected.range})
                </h4>
                <p className="text-sm text-slate-700 dark:text-gray-200 leading-relaxed mb-4">
                  {selected.description}
                </p>
                <div>
                  <h5 className="text-xs font-bold text-slate-600 dark:text-gray-300 mb-2 uppercase tracking-wider">
                    详细信息
                  </h5>
                  <ul className="space-y-1">
                    {selected.details.map((d) => (
                      <li
                        key={d}
                        className="text-xs text-slate-600 dark:text-gray-300 flex items-start gap-2"
                      >
                        <span className="text-indigo-400 mt-0.5">•</span>
                        {d}
                      </li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-white dark:bg-gray-800 rounded-lg p-5 shadow-md border border-slate-200 dark:border-gray-700 text-center"
              >
                <Info className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                <p className="text-sm text-slate-500 dark:text-gray-400">
                  点击磁盘布局中的任意区域查看详细信息。
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
