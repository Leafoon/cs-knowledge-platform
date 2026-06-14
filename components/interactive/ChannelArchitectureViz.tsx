"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Network, Layers, ArrowDown } from "lucide-react";

const components = [
  {
    id: "cpu",
    name: "CPU",
    desc: "CPU启动通道程序后继续执行主程序，通道独立管理I/O",
    details: ["发出SIO指令启动通道", "通过CAW传递通道程序地址", "通道完成后中断通知CPU"],
    y: 0,
  },
  {
    id: "channel",
    name: "通道控制器",
    desc: "具有简单指令处理能力的专用处理器，执行通道程序",
    details: ["包含通道状态字(CSW)", "解释执行通道命令字(CCW)", "管理数据传送和设备控制"],
    y: 1,
  },
  {
    id: "controller",
    name: "设备控制器",
    desc: "控制具体I/O设备的操作，是通道与设备的接口",
    details: ["接收通道命令", "控制设备机械操作", "缓冲数据和状态信息"],
    y: 2,
  },
  {
    id: "device",
    name: "I/O 设备",
    desc: "实际的外设（磁盘、打印机、终端等）",
    details: ["执行具体I/O操作", "产生设备状态信号", "与控制器通过标准接口连接"],
    y: 3,
  },
];

export function ChannelArchitectureViz() {
  const [selected, setSelected] = useState<string | null>(null);
  const selectedComp = components.find((c) => c.id === selected);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Network className="w-5 h-5 text-sky-400" />
        <h3 className="text-lg font-semibold">通道架构</h3>
      </div>

      <div className="flex flex-col items-center gap-2 mb-4">
        {components.map((c, i) => (
          <div key={c.id} className="flex flex-col items-center">
            <motion.button
              onClick={() => setSelected(selected === c.id ? null : c.id)}
              className={`px-6 py-3 rounded-lg border-2 text-sm font-medium transition-all min-w-[200px] ${
                selected === c.id
                  ? "border-sky-400 bg-sky-500/10 text-sky-200"
                  : "border-gray-600 bg-gray-800/50 text-gray-300 hover:border-gray-500"
              }`}
              whileHover={{ scale: 1.03 }}
            >
              {c.name}
            </motion.button>
            {i < components.length - 1 && (
              <div className="flex flex-col items-center my-1">
                <ArrowDown className="w-4 h-4 text-gray-500" />
                <span className="text-[10px] text-gray-500">
                  {i === 0 ? "SIO/CAW" : i === 1 ? "通道命令" : "控制信号"}
                </span>
              </div>
            )}
          </div>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {selectedComp && (
          <motion.div key={selected}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="p-4 bg-gray-800/30 rounded-lg"
          >
            <div className="text-sm text-sky-300 mb-2">{selectedComp.desc}</div>
            <div className="space-y-1.5">
              {selectedComp.details.map((d, i) => (
                <motion.div key={d}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="flex items-center gap-2 text-xs text-gray-300"
                >
                  <span className="w-1.5 h-1.5 bg-sky-400 rounded-full" />
                  {d}
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
        <div className="p-2 bg-gray-800/30 rounded text-center">
          <span className="text-gray-400">CSW</span>
          <div className="text-sky-300">通道状态字</div>
        </div>
        <div className="p-2 bg-gray-800/30 rounded text-center">
          <span className="text-gray-400">CAW</span>
          <div className="text-sky-300">通道地址字</div>
        </div>
        <div className="p-2 bg-gray-800/30 rounded text-center">
          <span className="text-gray-400">CCW</span>
          <div className="text-sky-300">通道命令字</div>
        </div>
      </div>
    </div>
  );
}
