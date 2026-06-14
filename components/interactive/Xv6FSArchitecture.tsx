"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  Layers,
  ArrowDown,
  ArrowUp,
  Info,
} from "lucide-react";

interface FSLayer {
  id: number;
  name: string;
  nameZh: string;
  color: string;
  bgColor: string;
  borderColor: string;
  description: string;
  keyFunctions: string[];
  dataStructures: string[];
}

const layers: FSLayer[] = [
  {
    id: 7,
    name: "File Descriptor Layer",
    nameZh: "文件描述符层",
    color: "text-purple-700 dark:text-purple-300",
    bgColor: "bg-purple-50 dark:bg-purple-950/40",
    borderColor: "border-purple-300 dark:border-purple-700",
    description: "为用户空间提供统一的文件接口。每个进程有独立的文件描述符表，将 fd 映射到 struct file。",
    keyFunctions: ["sys_open()", "sys_read()", "sys_write()", "sys_close()"],
    dataStructures: ["struct file", "fd table (ofile[])"],
  },
  {
    id: 6,
    name: "Pathname Layer",
    nameZh: "路径名层",
    color: "text-blue-700 dark:text-blue-300",
    bgColor: "bg-blue-50 dark:bg-blue-950/40",
    borderColor: "border-blue-300 dark:border-blue-700",
    description: "将路径名字符串（如 /home/user/file）解析为 inode。从根目录或当前目录开始逐级查找。",
    keyFunctions: ["namei()", "nameiparent()", "skipelem()"],
    dataStructures: ["路径字符串 → inode 映射"],
  },
  {
    id: 5,
    name: "Directory Layer",
    nameZh: "目录层",
    color: "text-cyan-700 dark:text-cyan-300",
    bgColor: "bg-cyan-50 dark:bg-cyan-950/40",
    borderColor: "border-cyan-300 dark:border-cyan-700",
    description: "目录是一种特殊的文件，内容是由 dirent 结构组成的数组。提供目录查找和添加目录项功能。",
    keyFunctions: ["dirlookup()", "dirlink()"],
    dataStructures: ["struct dirent (inum + name[14])"],
  },
  {
    id: 4,
    name: "Inode Layer",
    nameZh: "inode 层",
    color: "text-emerald-700 dark:text-emerald-300",
    bgColor: "bg-emerald-50 dark:bg-emerald-950/40",
    borderColor: "border-emerald-300 dark:border-emerald-700",
    description: "提供 inode 的分配、查找、锁定，以及逻辑块号到物理块号的映射（bmap）。是文件系统的核心层。",
    keyFunctions: ["ialloc()", "iget()", "ilock()", "bmap()", "readi()", "writei()"],
    dataStructures: ["struct inode", "struct dinode"],
  },
  {
    id: 3,
    name: "Logging Layer",
    nameZh: "日志层",
    color: "text-amber-700 dark:text-amber-300",
    bgColor: "bg-amber-50 dark:bg-amber-950/40",
    borderColor: "border-amber-300 dark:border-amber-700",
    description: "实现写前日志（WAL），保证崩溃一致性。每个事务的修改先写入日志区，提交后再写入实际位置。",
    keyFunctions: ["begin_op()", "end_op()", "log_write()"],
    dataStructures: ["struct log", "日志头块"],
  },
  {
    id: 2,
    name: "Buffer Cache Layer",
    nameZh: "缓冲区缓存层",
    color: "text-orange-700 dark:text-orange-300",
    bgColor: "bg-orange-50 dark:bg-orange-950/40",
    borderColor: "border-orange-300 dark:border-orange-700",
    description: "缓存磁盘块，减少磁盘访问次数。使用 LRU 替换策略。同时同步对同一磁盘块的并发访问。",
    keyFunctions: ["bread()", "bwrite()", "brelse()", "bget()"],
    dataStructures: ["struct buf", "LRU 双向链表"],
  },
  {
    id: 1,
    name: "Disk Layer",
    nameZh: "磁盘层",
    color: "text-red-700 dark:text-red-300",
    bgColor: "bg-red-50 dark:bg-red-950/40",
    borderColor: "border-red-300 dark:border-red-700",
    description: "通过 virtio 驱动读写磁盘块。是最底层的硬件抽象层，每次读写一个块（1024 字节）。",
    keyFunctions: ["virtio_disk_rw()", "virtio_disk_intr()"],
    dataStructures: ["磁盘块（1024 字节）"],
  },
];

export default function Xv6FSArchitecture() {
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [showFlow, setShowFlow] = useState(false);

  const selected = layers.find((l) => l.id === selectedLayer);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        xv6 File System — 七层架构
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        点击任意层查看详情，理解每层的职责与接口
      </p>

      <div className="flex gap-3 mb-6 justify-center flex-wrap">
        <button
          onClick={() => setShowFlow(!showFlow)}
          className="flex items-center gap-2 px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition-colors text-sm"
        >
          {showFlow ? <ArrowUp className="w-4 h-4" /> : <ArrowDown className="w-4 h-4" />}
          {showFlow ? "隐藏调用流" : "显示调用流"}
        </button>
      </div>

      {/* Call flow animation */}
      {showFlow && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          className="mb-6 bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700"
        >
          <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3">
            read(fd, buf, n) 调用流程
          </h3>
          <div className="flex flex-col items-center gap-1">
            {[
              { layer: "用户空间", action: "read(fd, buf, n)" },
              { layer: "文件描述符层", action: "sys_read → fileread(f, p, n)" },
              { layer: "inode 层", action: "readi(ip, dst, off, n)" },
              { layer: "inode 层", action: "bmap(ip, off/BSIZE) → 物理块号" },
              { layer: "缓冲区缓存层", action: "bread(dev, blockno) → struct buf" },
              { layer: "磁盘层", action: "virtio_disk_rw(b, 0) → 读磁盘" },
            ].map((step, i) => (
              <React.Fragment key={i}>
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.15 }}
                  className="flex items-center gap-3 w-full max-w-lg"
                >
                  <span className="text-xs font-mono text-slate-500 dark:text-gray-400 w-28 text-right shrink-0">
                    {step.layer}
                  </span>
                  <div className="flex-1 bg-indigo-50 dark:bg-indigo-950/30 rounded px-3 py-1.5 text-sm font-mono text-indigo-700 dark:text-indigo-300">
                    {step.action}
                  </div>
                </motion.div>
                {i < 5 && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.15 + 0.1 }}
                  >
                    <ArrowDown className="w-4 h-4 text-slate-400" />
                  </motion.div>
                )}
              </React.Fragment>
            ))}
          </div>
        </motion.div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Layer stack */}
        <div className="lg:col-span-2 space-y-2">
          <div className="text-center text-xs text-slate-500 dark:text-gray-400 mb-2 font-mono">
            用户空间
          </div>
          {layers.map((layer, i) => (
            <motion.button
              key={layer.id}
              onClick={() =>
                setSelectedLayer(selectedLayer === layer.id ? null : layer.id)
              }
              className={`w-full text-left p-4 rounded-lg border-2 transition-all ${layer.bgColor} ${
                selectedLayer === layer.id
                  ? `${layer.borderColor} shadow-md ring-1 ring-offset-1`
                  : "border-transparent hover:border-slate-300 dark:hover:border-gray-600"
              }`}
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
            >
              <div className="flex items-center gap-3">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold text-white ${layer.bgColor.replace("bg-", "bg-").replace("-50", "-500").replace("dark:bg-", "dark:bg-").replace("-950/40", "-700")}`}
                  style={{
                    backgroundColor: undefined,
                  }}
                >
                  <span className={layer.color}>{layer.id}</span>
                </div>
                <div className="flex-1">
                  <div className={`font-bold text-sm ${layer.color}`}>
                    第 {layer.id} 层：{layer.nameZh}
                  </div>
                  <div className="text-xs text-slate-500 dark:text-gray-400 font-mono">
                    {layer.name}
                  </div>
                </div>
                <Layers className={`w-4 h-4 ${layer.color}`} />
              </div>
              <div className="mt-2 text-xs text-slate-600 dark:text-gray-300 line-clamp-1">
                {layer.description}
              </div>
            </motion.button>
          ))}
          <div className="text-center text-xs text-slate-500 dark:text-gray-400 mt-2 font-mono">
            物理磁盘
          </div>
        </div>

        {/* Detail panel */}
        <div className="lg:col-span-1">
          {selected ? (
            <motion.div
              key={selected.id}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className={`${selected.bgColor} rounded-lg p-5 shadow-md border ${selected.borderColor}`}
            >
              <h3 className={`text-lg font-bold mb-1 ${selected.color}`}>
                第 {selected.id} 层
              </h3>
              <h4 className="text-sm font-mono text-slate-600 dark:text-gray-300 mb-3">
                {selected.name}
              </h4>
              <p className="text-sm text-slate-700 dark:text-gray-200 leading-relaxed mb-4">
                {selected.description}
              </p>

              <div className="mb-4">
                <h5 className="text-xs font-bold text-slate-600 dark:text-gray-300 mb-2 uppercase tracking-wider">
                  关键函数
                </h5>
                <div className="flex flex-wrap gap-1.5">
                  {selected.keyFunctions.map((fn) => (
                    <span
                      key={fn}
                      className="px-2 py-1 bg-white/60 dark:bg-gray-800/60 rounded text-xs font-mono text-slate-700 dark:text-gray-200 border border-slate-200 dark:border-gray-700"
                    >
                      {fn}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <h5 className="text-xs font-bold text-slate-600 dark:text-gray-300 mb-2 uppercase tracking-wider">
                  数据结构
                </h5>
                <div className="space-y-1">
                  {selected.dataStructures.map((ds) => (
                    <div
                      key={ds}
                      className="text-xs font-mono text-slate-600 dark:text-gray-300 bg-white/40 dark:bg-gray-800/40 rounded px-2 py-1"
                    >
                      {ds}
                    </div>
                  ))}
                </div>
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
                点击左侧任意层查看详细信息，包括关键函数和数据结构。
              </p>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}
