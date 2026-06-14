"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Binary, ToggleLeft, ToggleRight } from "lucide-react";

export function IOPortDecoder() {
  const [mode, setMode] = useState<"isolated" | "memory">("isolated");
  const [addr, setAddr] = useState("0x3F8");
  const [decoded, setDecoded] = useState(false);

  const addrNum = parseInt(addr, 16) || 0;
  const isMemoryMapped = mode === "memory";

  const portMap = [
    { name: "串口 COM1", addr: 0x3f8, range: "0x3F8-0x3FF" },
    { name: "并口 LPT1", addr: 0x378, range: "0x378-0x37F" },
    { name: "键盘控制器", addr: 0x060, range: "0x060-0x06F" },
    { name: "定时器", addr: 0x040, range: "0x040-0x043" },
    { name: "DMA控制器", addr: 0x000, range: "0x000-0x01F" },
  ];

  const matchedDevice = portMap.find((p) => addrNum >= p.addr && addrNum < p.addr + 8);

  const handleDecode = () => {
    setDecoded(true);
    setTimeout(() => setDecoded(false), 1500);
  };

  const addrBits = addrNum.toString(2).padStart(16, "0").slice(-16);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Binary className="w-5 h-5 text-indigo-400" />
        <h3 className="text-lg font-semibold">I/O 端口译码器</h3>
      </div>

      <div className="flex gap-3 mb-4">
        <button
          onClick={() => setMode("isolated")}
          className={`flex items-center gap-1.5 px-4 py-2 rounded text-sm ${
            mode === "isolated" ? "bg-indigo-600 text-white" : "bg-gray-700 text-gray-300"
          }`}
        >
          {mode === "isolated" ? <ToggleRight className="w-4 h-4" /> : <ToggleLeft className="w-4 h-4" />}
          独立编址 I/O
        </button>
        <button
          onClick={() => setMode("memory")}
          className={`flex items-center gap-1.5 px-4 py-2 rounded text-sm ${
            mode === "memory" ? "bg-purple-600 text-white" : "bg-gray-700 text-gray-300"
          }`}
        >
          {mode === "memory" ? <ToggleRight className="w-4 h-4" /> : <ToggleLeft className="w-4 h-4" />}
          统一编址 (Memory-Mapped)
        </button>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <div className="text-xs text-gray-400 mb-2">地址输入</div>
          <div className="flex gap-2">
            <input
              type="text"
              value={addr}
              onChange={(e) => setAddr(e.target.value)}
              className="w-32 px-3 py-1.5 bg-gray-800 border border-gray-600 rounded text-sm text-white font-mono"
            />
            <button onClick={handleDecode} className="px-3 py-1.5 bg-indigo-600 rounded text-sm text-white hover:bg-indigo-500">
              译码
            </button>
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-400 mb-2">地址二进制</div>
          <div className="font-mono text-xs text-indigo-300 break-all leading-5">
            {addrBits.split("").map((b, i) => (
              <span key={i} className={i >= 13 ? "text-yellow-300" : "text-gray-500"}>
                {b}
              </span>
            ))}
          </div>
          <div className="text-[10px] text-gray-500 mt-1">
            黄色部分为端口地址位
          </div>
        </div>
      </div>

      <div className="mb-4 p-3 bg-gray-800/30 rounded-lg">
        <div className="text-xs text-gray-400 mb-2">地址空间说明</div>
        <div className="text-sm">
          {isMemoryMapped ? (
            <span className="text-purple-300">I/O端口占用内存地址空间的一部分，使用相同的地址总线和指令（如MOV）访问</span>
          ) : (
            <span className="text-indigo-300">I/O端口有独立的地址空间，使用专用指令（IN/OUT）和IORQ信号访问</span>
          )}
        </div>
      </div>

      <div className="text-xs text-gray-400 mb-2">已知端口映射:</div>
      <div className="grid grid-cols-5 gap-2 mb-4">
        {portMap.map((p) => (
          <motion.button
            key={p.name}
            onClick={() => setAddr(`0x${p.addr.toString(16).toUpperCase()}`)}
            className="p-2 bg-gray-800/50 rounded text-xs hover:bg-gray-700 transition-colors"
            whileHover={{ scale: 1.05 }}
          >
            <div className="text-gray-300">{p.name}</div>
            <div className="text-indigo-400 font-mono">{p.range}</div>
          </motion.button>
        ))}
      </div>

      <AnimatePresence>
        {decoded && matchedDevice && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -5 }}
            className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg text-sm"
          >
            <span className="text-green-400">✓ 译码命中: </span>
            <span className="text-gray-200">{matchedDevice.name}</span>
            <span className="text-gray-400 ml-2">({matchedDevice.range})</span>
          </motion.div>
        )}
        {decoded && !matchedDevice && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -5 }}
            className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400"
          >
            ✗ 地址未匹配任何已知设备
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
