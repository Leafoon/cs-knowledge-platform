"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layers, ArrowDown, ArrowUp } from "lucide-react";

const layers = [
  {
    name: "事务层 (Transaction Layer)",
    color: "blue",
    desc: "生成和处理 TLP（事务层包），支持存储器读写、I/O、配置、消息等请求。",
    details: [
      "TLP 包含: Header + Data + LCRC",
      "支持 64 位地址空间",
      "基于信用的流控机制",
      "支持分离事务（Posted/Non-Posted）",
    ],
    packets: ["Memory Read", "Memory Write", "Completion", "Config Read"],
  },
  {
    name: "数据链路层 (Data Link Layer)",
    color: "green",
    desc: "确保 TLP 可靠传输，添加序列号和 LCRC，处理 ACK/NAK。",
    details: [
      "DLLP: ACK/NAK、流控、电源管理",
      "序列号检测丢包和乱序",
      "LCRC 错误检测",
      "重传缓冲区管理",
    ],
    packets: ["ACK DLLP", "NAK DLLP", "FC Update", "Power Mgmt"],
  },
  {
    name: "物理层 (Physical Layer)",
    color: "purple",
    desc: "处理实际的电信号传输，包括链路训练、字符编解码、差分信号。",
    details: [
      "差分信号对 (TX+/TX-, RX+/RX-)",
      "8b/10b 或 128b/130b 编码",
      "链路训练和状态机 (LTSSM)",
      "多通道绑定 (x1, x2, x4, x8, x16)",
    ],
    packets: ["Ordered Set", "TS1/TS2", "FTS", "EIOS/EIEOS"],
  },
];

const colorMap: Record<string, string> = {
  blue: "border-blue-500 bg-blue-500/10",
  green: "border-green-500 bg-green-500/10",
  purple: "border-purple-500 bg-purple-500/10",
};

const textColorMap: Record<string, string> = {
  blue: "text-blue-400",
  green: "text-green-400",
  purple: "text-purple-400",
};

export function PCIeLayerViz() {
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [animDir, setAnimDir] = useState<"down" | "up">("down");

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Layers className="w-5 h-5 text-purple-400" />
        <h3 className="text-lg font-semibold">PCIe 层次结构</h3>
      </div>

      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setAnimDir("down")}
          className={`px-3 py-1 rounded text-xs ${animDir === "down" ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-300"}`}
        >
          <ArrowDown className="w-3 h-3 inline mr-1" />发送
        </button>
        <button
          onClick={() => setAnimDir("up")}
          className={`px-3 py-1 rounded text-xs ${animDir === "up" ? "bg-green-600 text-white" : "bg-gray-700 text-gray-300"}`}
        >
          <ArrowUp className="w-3 h-3 inline mr-1" />接收
        </button>
      </div>

      <div className="space-y-3">
        {[...layers].reverse().map((layer, ri) => {
          const i = layers.length - 1 - ri;
          return (
            <motion.div
              key={layer.name}
              layout
              onClick={() => setSelectedLayer(selectedLayer === i ? null : i)}
              className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${colorMap[layer.color]} ${
                selectedLayer === i ? "ring-2 ring-white/20" : ""
              }`}
              whileHover={{ scale: 1.01 }}
            >
              <div className="flex items-center justify-between">
                <span className={`font-medium ${textColorMap[layer.color]}`}>{layer.name}</span>
                {selectedLayer !== i && (
                  <motion.div
                    animate={{ y: animDir === "down" ? [0, 4, 0] : [0, -4, 0] }}
                    transition={{ repeat: Infinity, duration: 1.5 }}
                    className="flex gap-1"
                  >
                    {layer.packets.slice(0, 2).map((p) => (
                      <span key={p} className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 text-gray-300">
                        {p}
                      </span>
                    ))}
                  </motion.div>
                )}
              </div>

              <AnimatePresence>
                {selectedLayer === i && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="overflow-hidden"
                  >
                    <p className="text-sm text-gray-300 mt-2 mb-3">{layer.desc}</p>
                    <div className="grid grid-cols-2 gap-2 mb-3">
                      {layer.details.map((d) => (
                        <div key={d} className="text-xs p-2 bg-gray-800/60 rounded text-gray-300">
                          {d}
                        </div>
                      ))}
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {layer.packets.map((p) => (
                        <span key={p} className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-200">
                          {p}
                        </span>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>

      <div className="mt-4 flex items-center justify-center gap-2 text-xs text-gray-500">
        <span>上层: 软件接口</span>
        <ArrowDown className="w-3 h-3" />
        <span>下层: 物理信号</span>
      </div>
    </div>
  );
}
