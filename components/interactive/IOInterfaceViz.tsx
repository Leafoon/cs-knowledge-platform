"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, MonitorDot, ArrowLeftRight } from "lucide-react";

const registers = [
  {
    name: "数据缓冲寄存器 (DBR)",
    bits: [
      { name: "D7-D0", desc: "8位数据缓冲，暂存CPU与设备间传输的数据" },
    ],
    role: "双向数据缓冲，解决CPU与设备速度不匹配问题",
  },
  {
    name: "状态寄存器 (SR)",
    bits: [
      { name: "BSY", desc: "忙标志，设备正在工作" },
      { name: "RDY", desc: "就绪标志，设备准备好数据传输" },
      { name: "ERR", desc: "错误标志，传输出错" },
      { name: "INT", desc: "中断请求标志" },
      { name: "D7-D4", desc: "设备特定状态位" },
    ],
    role: "CPU读取以了解设备当前状态",
  },
  {
    name: "控制寄存器 (CR)",
    bits: [
      { name: "EN", desc: "接口使能位" },
      { name: "IE", desc: "中断使能位" },
      { name: "RW", desc: "读/写方向选择" },
      { name: "MODE", desc: "工作模式选择" },
      { name: "D7-D4", desc: "设备特定控制位" },
    ],
    role: "CPU写入以控制设备操作",
  },
];

export function IOInterfaceViz() {
  const [selected, setSelected] = useState(0);
  const reg = registers[selected];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <MonitorDot className="w-5 h-5 text-cyan-400" />
        <h3 className="text-lg font-semibold">I/O 接口结构</h3>
      </div>

      <div className="flex items-center justify-center gap-4 mb-6">
        <motion.div className="p-3 bg-blue-600/20 border border-blue-500 rounded-lg text-center"
          animate={{ x: [0, -2, 0] }} transition={{ repeat: Infinity, duration: 2 }}>
          <Cpu className="w-6 h-6 mx-auto mb-1 text-blue-400" />
          <span className="text-xs text-blue-300">CPU</span>
        </motion.div>

        <ArrowLeftRight className="w-5 h-5 text-gray-500" />

        <div className="flex gap-2">
          {registers.map((r, i) => (
            <motion.button
              key={r.name}
              onClick={() => setSelected(i)}
              className={`p-3 rounded-lg border text-center min-w-[100px] ${
                selected === i ? "border-cyan-400 bg-cyan-500/10" : "border-gray-600 bg-gray-800/50"
              }`}
              whileHover={{ scale: 1.05 }}
            >
              <span className="text-xs block text-gray-300">{r.name.split("(")[0].trim()}</span>
            </motion.button>
          ))}
        </div>

        <ArrowLeftRight className="w-5 h-5 text-gray-500" />

        <motion.div className="p-3 bg-green-600/20 border border-green-500 rounded-lg text-center"
          animate={{ x: [0, 2, 0] }} transition={{ repeat: Infinity, duration: 2 }}>
          <MonitorDot className="w-6 h-6 mx-auto mb-1 text-green-400" />
          <span className="text-xs text-green-300">I/O 设备</span>
        </motion.div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={selected}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
        >
          <div className="p-4 bg-gray-800/30 rounded-lg mb-4">
            <div className="text-sm text-gray-400 mb-3">功能: <span className="text-cyan-300">{reg.role}</span></div>
            <div className="space-y-2">
              {reg.bits.map((b, i) => (
                <motion.div
                  key={b.name}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.08 }}
                  className="flex items-center gap-3"
                >
                  <div className="flex gap-0.5">
                    {b.name.split("").map((ch, ci) => (
                      <span
                        key={ci}
                        className={`w-7 h-7 flex items-center justify-center rounded text-xs font-mono ${
                          ch === "-" ? "bg-gray-600 text-gray-400" : "bg-gray-700 text-cyan-300 border border-gray-500"
                        }`}
                      >
                        {ch}
                      </span>
                    ))}
                  </div>
                  <span className="text-xs text-gray-300">{b.desc}</span>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
