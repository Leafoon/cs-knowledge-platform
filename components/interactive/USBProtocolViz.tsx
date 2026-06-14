"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Usb, Package } from "lucide-react";

const transferTypes = [
  {
    name: "控制传输",
    use: "枚举、配置、命令",
    speed: "低速/全速/高速",
    packets: [
      { type: "SETUP", fields: ["Sync", "PID=0x2D", "ADDR", "ENDP", "CRC5"] },
      { type: "DATA0", fields: ["Sync", "PID=0xC3", "Data (8B)", "CRC16"] },
      { type: "ACK", fields: ["Sync", "PID=0x4B"] },
    ],
    desc: "用于设备枚举和控制命令，确保可靠传输",
  },
  {
    name: "批量传输",
    use: "打印机、存储设备",
    speed: "全速/高速",
    packets: [
      { type: "OUT", fields: ["Sync", "PID=0xE1", "ADDR", "ENDP", "CRC5"] },
      { type: "DATA1", fields: ["Sync", "PID=0x4B", "Data (512B)", "CRC16"] },
      { type: "ACK", fields: ["Sync", "PID=0x4B"] },
    ],
    desc: "大数据量传输，保证数据完整性，不保证实时性",
  },
  {
    name: "中断传输",
    use: "鼠标、键盘、游戏手柄",
    speed: "低速/全速/高速",
    packets: [
      { type: "IN", fields: ["Sync", "PID=0x69", "ADDR", "ENDP", "CRC5"] },
      { type: "DATA0", fields: ["Sync", "PID=0xC3", "Data", "CRC16"] },
      { type: "ACK", fields: ["Sync", "PID=0x4B"] },
    ],
    desc: "轮询式低延迟传输，适合小数据量交互设备",
  },
  {
    name: "等时传输",
    use: "音频、视频流",
    speed: "全速/高速",
    packets: [
      { type: "IN", fields: ["Sync", "PID=0x69", "ADDR", "ENDP", "CRC5"] },
      { type: "DATA0", fields: ["Sync", "PID=0xC3", "Data (1024B)", "CRC16"] },
    ],
    desc: "恒定带宽、实时传输，不保证数据正确性（无握手）",
  },
];

export function USBProtocolViz() {
  const [selectedType, setSelectedType] = useState(0);
  const [selectedPacket, setSelectedPacket] = useState(0);
  const transfer = transferTypes[selectedType];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Usb className="w-5 h-5 text-green-400" />
        <h3 className="text-lg font-semibold">USB 协议可视化</h3>
      </div>

      <div className="flex gap-2 mb-4">
        {transferTypes.map((t, i) => (
          <button
            key={t.name}
            onClick={() => { setSelectedType(i); setSelectedPacket(0); }}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              selectedType === i ? "bg-green-600 text-white" : "bg-gray-700 text-gray-300 hover:bg-gray-600"
            }`}
          >
            {t.name}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={selectedType}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
        >
          <div className="grid grid-cols-3 gap-2 mb-4 text-xs">
            <div className="p-2 bg-gray-800/50 rounded">
              <span className="text-gray-400">用途: </span>
              <span className="text-green-300">{transfer.use}</span>
            </div>
            <div className="p-2 bg-gray-800/50 rounded">
              <span className="text-gray-400">速度: </span>
              <span className="text-blue-300">{transfer.speed}</span>
            </div>
            <div className="p-2 bg-gray-800/50 rounded">
              <span className="text-gray-400">特点: </span>
              <span className="text-yellow-300">{transfer.desc}</span>
            </div>
          </div>

          <div className="mb-3 text-sm text-gray-400">包序列:</div>
          <div className="flex items-center gap-2 mb-4">
            {transfer.packets.map((p, i) => (
              <div key={p.type + i} className="flex items-center gap-2">
                <button
                  onClick={() => setSelectedPacket(i)}
                  className={`px-3 py-2 rounded text-xs font-mono transition-colors ${
                    selectedPacket === i ? "bg-green-600 text-white" : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                  }`}
                >
                  <Package className="w-3 h-3 inline mr-1" />{p.type}
                </button>
                {i < transfer.packets.length - 1 && (
                  <motion.span
                    animate={{ x: [0, 4, 0] }}
                    transition={{ repeat: Infinity, duration: 1 }}
                    className="text-gray-500"
                  >
                    →
                  </motion.span>
                )}
              </div>
            ))}
          </div>

          <div className="p-4 bg-gray-800/30 rounded-lg">
            <div className="text-sm text-gray-400 mb-2">
              {transfer.packets[selectedPacket].type} 包格式:
            </div>
            <div className="flex gap-1">
              {transfer.packets[selectedPacket].fields.map((f, i) => (
                <motion.div
                  key={f}
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: i * 0.1 }}
                  className="px-3 py-2 bg-gray-700 rounded text-xs text-center"
                >
                  {f}
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
