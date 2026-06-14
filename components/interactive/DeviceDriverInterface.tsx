"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Info, ChevronDown, ChevronUp } from "lucide-react";

interface Layer {
  id: string;
  name: string;
  nameZh: string;
  color: string;
  bgColor: string;
  borderColor: string;
  description: string;
  examples: string[];
}

const layers: Layer[] = [
  {
    id: "user",
    name: "User Application",
    nameZh: "用户应用程序",
    color: "text-purple-700 dark:text-purple-300",
    bgColor: "bg-purple-50 dark:bg-purple-950/40",
    borderColor: "border-purple-300 dark:border-purple-700",
    description: "用户程序通过系统调用（read/write/ioctl）请求 I/O 操作，不需要了解硬件细节。",
    examples: ["read(fd, buf, n)", "write(fd, buf, n)", "ioctl(fd, cmd, arg)"],
  },
  {
    id: "vfs",
    name: "Virtual File System",
    nameZh: "虚拟文件系统 (VFS)",
    color: "text-blue-700 dark:text-blue-300",
    bgColor: "bg-blue-50 dark:bg-blue-950/40",
    borderColor: "border-blue-300 dark:border-blue-700",
    description: "VFS 提供统一的文件接口。它根据文件类型（普通文件/设备文件/管道）将请求分发到对应的处理模块。",
    examples: ["vfs_read()", "vfs_write()", "file_operations->read()"],
  },
  {
    id: "driver",
    name: "Device Driver",
    nameZh: "设备驱动程序",
    color: "text-emerald-700 dark:text-emerald-300",
    bgColor: "bg-emerald-50 dark:bg-emerald-950/40",
    borderColor: "border-emerald-300 dark:border-emerald-700",
    description: "驱动程序直接与硬件交互。它知道如何配置设备寄存器、处理中断、管理 DMA 传输。每个设备类型有专门的驱动。",
    examples: ["uartputc()", "virtio_disk_rw()", "e1000_transmit()"],
  },
  {
    id: "controller",
    name: "Device Controller",
    nameZh: "设备控制器（硬件）",
    color: "text-amber-700 dark:text-amber-300",
    bgColor: "bg-amber-50 dark:bg-amber-950/40",
    borderColor: "border-amber-300 dark:border-amber-700",
    description: "硬件控制器将软件命令转换为电信号。它有寄存器（状态/命令/数据）供 CPU 或 DMA 访问。",
    examples: ["状态寄存器", "命令寄存器", "数据寄存器"],
  },
  {
    id: "device",
    name: "Physical Device",
    nameZh: "物理设备",
    color: "text-red-700 dark:text-red-300",
    bgColor: "bg-red-50 dark:bg-red-950/40",
    borderColor: "border-red-300 dark:border-red-700",
    description: "实际的硬件设备：磁盘盘片、网络 PHY 芯片、键盘矩阵等。执行物理操作（寻道、旋转、信号调制）。",
    examples: ["磁盘盘片旋转", "网络信号调制", "键盘矩阵扫描"],
  },
];

export default function DeviceDriverInterface() {
  const [selected, setSelected] = useState<string | null>(null);
  const selectedLayer = layers.find((l) => l.id === selected);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        设备驱动架构层次图
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        点击各层查看详细信息，理解 I/O 请求从用户空间到硬件的完整路径
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-2">
          {layers.map((layer, i) => (
            <motion.button
              key={layer.id}
              onClick={() => setSelected(selected === layer.id ? null : layer.id)}
              className={`w-full text-left p-4 rounded-lg border-2 transition-all ${layer.bgColor} ${
                selected === layer.id ? `${layer.borderColor} shadow-md` : "border-transparent hover:border-slate-300 dark:hover:border-gray-600"
              }`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              whileHover={{ scale: 1.005 }}
            >
              <div className="flex items-center gap-3">
                <span className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${layer.color} bg-white dark:bg-gray-800 border`}>
                  {i + 1}
                </span>
                <div className="flex-1">
                  <div className={`font-bold text-sm ${layer.color}`}>{layer.nameZh}</div>
                  <div className="text-xs text-slate-500 dark:text-gray-400 font-mono">{layer.name}</div>
                </div>
                {selected === layer.id ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </div>
              <AnimatePresence>
                {selected === layer.id && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="mt-3 pt-3 border-t border-slate-200 dark:border-gray-700"
                  >
                    <p className="text-sm text-slate-700 dark:text-gray-200 mb-2">{layer.description}</p>
                    <div className="flex flex-wrap gap-1.5">
                      {layer.examples.map((ex) => (
                        <span key={ex} className="px-2 py-0.5 bg-white/60 dark:bg-gray-800/60 rounded text-xs font-mono text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700">
                          {ex}
                        </span>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.button>
          ))}
        </div>

        <div className="lg:col-span-1">
          {selectedLayer ? (
            <motion.div
              key={selectedLayer.id}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className={`${selectedLayer.bgColor} rounded-lg p-5 shadow-md border ${selectedLayer.borderColor} sticky top-4`}
            >
              <h3 className={`text-lg font-bold mb-2 ${selectedLayer.color}`}>{selectedLayer.nameZh}</h3>
              <p className="text-sm text-slate-700 dark:text-gray-200 leading-relaxed mb-4">{selectedLayer.description}</p>
              <h4 className="text-xs font-bold text-slate-600 dark:text-gray-300 mb-2 uppercase">API 示例</h4>
              <div className="space-y-1">
                {selectedLayer.examples.map((ex) => (
                  <div key={ex} className="text-xs font-mono text-slate-600 dark:text-gray-300 bg-white/40 dark:bg-gray-800/40 rounded px-2 py-1">{ex}</div>
                ))}
              </div>
            </motion.div>
          ) : (
            <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-slate-200 dark:border-gray-700 text-center sticky top-4">
              <Info className="w-8 h-8 text-slate-400 mx-auto mb-2" />
              <p className="text-sm text-slate-500 dark:text-gray-400">点击左侧各层查看详细信息</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
