"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { HardDrive, Cpu, ArrowLeftRight } from "lucide-react";

const registers = [
  {
    name: "内存地址寄存器 (MAR)",
    bits: [
      { name: "MA[31:0]", desc: "32位内存地址，指向数据传送的内存位置" },
    ],
    note: "每传送一个字自动递增，指向下一个地址",
  },
  {
    name: "字计数器 (WC)",
    bits: [
      { name: "WC[15:0]", desc: "16位计数值，记录剩余传送字数" },
    ],
    note: "每传送一个字自动减1，减到0时传送结束",
  },
  {
    name: "控制/状态寄存器 (CSR)",
    bits: [
      { name: "GO", desc: "启动DMA传送" },
      { name: "DIR", desc: "传送方向: 0=读内存, 1=写内存" },
      { name: "IE", desc: "中断使能" },
      { name: "DONE", desc: "传送完成标志" },
      { name: "ERR", desc: "错误标志" },
      { name: "MODE", desc: "传送模式选择" },
      { name: "PRIORITY", desc: "优先级设置" },
    ],
    note: "CPU通过此寄存器控制DMA操作和查询状态",
  },
  {
    name: "数据缓冲寄存器 (DBR)",
    bits: [
      { name: "DATA[31:0]", desc: "32位数据缓冲，暂存传送数据" },
    ],
    note: "在总线周期中暂存数据，解决速度匹配问题",
  },
];

export function DMAControllerViz() {
  const [selected, setSelected] = useState(0);
  const reg = registers[selected];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <HardDrive className="w-5 h-5 text-cyan-400" />
        <h3 className="text-lg font-semibold">DMA 控制器结构</h3>
      </div>

      <div className="flex items-center justify-center gap-4 mb-6">
        <div className="flex flex-col items-center gap-1">
          <Cpu className="w-8 h-8 text-blue-400" />
          <span className="text-xs text-blue-300">CPU</span>
        </div>
        <ArrowLeftRight className="w-5 h-5 text-gray-500" />
        <div className="p-4 border-2 border-dashed border-cyan-500/50 rounded-lg">
          <div className="text-xs text-cyan-400 mb-3 text-center font-medium">DMA 控制器</div>
          <div className="grid grid-cols-2 gap-2">
            {registers.map((r, i) => (
              <motion.button
                key={r.name}
                onClick={() => setSelected(i)}
                className={`p-2 rounded text-xs text-center min-w-[90px] ${
                  selected === i
                    ? "bg-cyan-600/20 border border-cyan-400 text-cyan-200"
                    : "bg-gray-800/50 border border-gray-600 text-gray-300 hover:bg-gray-700/50"
                }`}
                whileHover={{ scale: 1.05 }}
              >
                {r.name.split("(")[0].trim()}
              </motion.button>
            ))}
          </div>
        </div>
        <ArrowLeftRight className="w-5 h-5 text-gray-500" />
        <div className="flex flex-col items-center gap-1">
          <HardDrive className="w-8 h-8 text-green-400" />
          <span className="text-xs text-green-300">I/O 设备</span>
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={selected}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="p-4 bg-gray-800/30 rounded-lg"
        >
          <div className="text-sm font-medium text-cyan-300 mb-3">{reg.name}</div>
          <div className="space-y-2 mb-3">
            {reg.bits.map((b, i) => (
              <motion.div key={b.name}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.08 }}
                className="flex items-center gap-3"
              >
                <span className="font-mono text-xs text-cyan-400 w-20 shrink-0">{b.name}</span>
                <span className="text-xs text-gray-300">{b.desc}</span>
              </motion.div>
            ))}
          </div>
          <div className="text-xs text-yellow-400/80 p-2 bg-yellow-500/5 rounded">
            {reg.note}
          </div>
        </motion.div>
      </AnimatePresence>

      <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
        <div className="p-2 bg-gray-800/30 rounded text-center">
          <span className="text-gray-400">数据线</span>
          <div className="text-blue-300 font-mono">D[31:0]</div>
        </div>
        <div className="p-2 bg-gray-800/30 rounded text-center">
          <span className="text-gray-400">地址线</span>
          <div className="text-green-300 font-mono">A[31:0]</div>
        </div>
        <div className="p-2 bg-gray-800/30 rounded text-center">
          <span className="text-gray-400">控制线</span>
          <div className="text-yellow-300 font-mono">BR/BG/RD/WR</div>
        </div>
      </div>
    </div>
  );
}
