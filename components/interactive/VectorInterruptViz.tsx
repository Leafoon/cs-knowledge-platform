"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Table, Hash, ArrowRight } from "lucide-react";

const vectorTable = [
  { vec: 0, addr: "0x000", name: "除法错误", source: "CPU内部" },
  { vec: 1, addr: "0x004", name: "调试异常", source: "CPU内部" },
  { vec: 2, addr: "0x008", name: "NMI中断", source: "不可屏蔽中断" },
  { vec: 3, addr: "0x00C", name: "断点异常", source: "CPU内部" },
  { vec: 4, addr: "0x010", name: "溢出异常", source: "CPU内部" },
  { vec: 5, addr: "0x014", name: "越界检查", source: "CPU内部" },
  { vec: 6, addr: "0x018", name: "非法指令", source: "CPU内部" },
  { vec: 7, addr: "0x01C", name: "设备中断", source: "I/O设备" },
  { vec: 8, addr: "0x020", name: "时钟中断", source: "定时器" },
  { vec: 9, addr: "0x024", name: "键盘中断", source: "键盘控制器" },
  { vec: 10, addr: "0x028", name: "串口中断", source: "UART" },
  { vec: 11, addr: "0x02C", name: "磁盘中断", source: "磁盘控制器" },
  { vec: 12, addr: "0x030", name: "网络中断", source: "网卡" },
  { vec: 13, addr: "0x034", name: "打印机中断", source: "打印机" },
  { vec: 14, addr: "0x038", name: "保留", source: "-" },
  { vec: 15, addr: "0x03C", name: "保留", source: "-" },
];

export function VectorInterruptViz() {
  const [activeVec, setActiveVec] = useState<number | null>(null);
  const [showGeneration, setShowGeneration] = useState(false);
  const activeEntry = activeVec !== null ? vectorTable[activeVec] : null;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Table className="w-5 h-5 text-emerald-400" />
        <h3 className="text-lg font-semibold">向量中断可视化</h3>
      </div>

      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setShowGeneration(!showGeneration)}
          className={`px-3 py-1 rounded text-xs ${showGeneration ? "bg-emerald-600 text-white" : "bg-gray-700 text-gray-300"}`}
        >
          向量地址生成过程
        </button>
      </div>

      <AnimatePresence>
        {showGeneration && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden mb-4"
          >
            <div className="p-4 bg-gray-800/30 rounded-lg">
              <div className="flex items-center gap-3 text-xs mb-3">
                <span className="px-2 py-1 bg-blue-600/30 rounded text-blue-300">中断源编码 (4bit)</span>
                <ArrowRight className="w-4 h-4 text-gray-500" />
                <span className="px-2 py-1 bg-yellow-600/30 rounded text-yellow-300">×4 (左移2位)</span>
                <ArrowRight className="w-4 h-4 text-gray-500" />
                <span className="px-2 py-1 bg-emerald-600/30 rounded text-emerald-300">向量地址</span>
              </div>
              <div className="text-xs text-gray-400">
                向量地址 = 中断类型号 × 4（每个表项占4字节）
                <br />
                硬件编码器将设备ID直接转换为向量地址，无需软件查找
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="grid grid-cols-4 gap-1 mb-4">
        {vectorTable.map((v) => (
          <motion.button
            key={v.vec}
            onClick={() => setActiveVec(activeVec === v.vec ? null : v.vec)}
            className={`p-2 rounded text-xs text-left transition-all ${
              activeVec === v.vec
                ? "bg-emerald-600/20 border border-emerald-500"
                : v.name === "保留"
                ? "bg-gray-800/20 border border-gray-700 opacity-50"
                : "bg-gray-800/50 border border-gray-700 hover:border-gray-500"
            }`}
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex justify-between">
              <span className="text-emerald-400 font-mono">#{v.vec}</span>
              <span className="text-gray-500 font-mono">{v.addr}</span>
            </div>
            <div className="text-gray-300 truncate">{v.name}</div>
          </motion.button>
        ))}
      </div>

      <AnimatePresence>
        {activeEntry && activeEntry.name !== "保留" && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="p-4 bg-gray-800/30 rounded-lg"
          >
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-400">中断向量号: </span>
                <span className="text-emerald-300 font-mono">{activeEntry.vec}</span>
              </div>
              <div>
                <span className="text-gray-400">向量地址: </span>
                <span className="text-emerald-300 font-mono">{activeEntry.addr}</span>
              </div>
              <div>
                <span className="text-gray-400">中断源: </span>
                <span className="text-gray-200">{activeEntry.source}</span>
              </div>
              <div>
                <span className="text-gray-400">计算: </span>
                <span className="text-gray-200">{activeEntry.vec} × 4 = {activeEntry.addr}</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
