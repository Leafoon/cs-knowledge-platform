"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { GitBranch, Play, Pause, RotateCcw } from "lucide-react";

const channelTypes = [
  {
    name: "选择通道",
    desc: "一次只服务一台高速设备，传送期间独占通道",
    devices: ["高速磁盘"],
    slots: [
      [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    colors: ["bg-red-500", "bg-gray-700", "bg-gray-700"],
    names: ["磁盘A", "磁盘B", "磁带C"],
    useCase: "高速设备（如磁盘），数据率接近通道极限",
  },
  {
    name: "数组多路通道",
    desc: "以数据块为单位交替服务多台设备",
    devices: ["中速磁盘", "磁带"],
    slots: [
      [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
      [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    ],
    colors: ["bg-blue-500", "bg-green-500", "bg-yellow-500"],
    names: ["磁盘A", "磁盘B", "磁带C"],
    useCase: "中速设备，块级交叉传送提高通道利用率",
  },
  {
    name: "字节多路通道",
    desc: "以字节为单位交替服务多台低速设备",
    devices: ["终端", "打印机"],
    slots: [
      [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
      [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    ],
    colors: ["bg-purple-500", "bg-pink-500", "bg-cyan-500"],
    names: ["终端A", "终端B", "打印机C"],
    useCase: "低速设备，字节级交叉实现多设备并行",
  },
];

export function ChannelTypeComparison() {
  const [selected, setSelected] = useState(0);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const type = channelTypes[selected];

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => setStep((s) => (s + 1) % 16), 400);
    return () => clearInterval(timer);
  }, [playing]);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <GitBranch className="w-5 h-5 text-indigo-400" />
        <h3 className="text-lg font-semibold">通道类型对比</h3>
      </div>

      <div className="flex gap-2 mb-3">
        {channelTypes.map((t, i) => (
          <button key={t.name}
            onClick={() => { setSelected(i); setStep(0); setPlaying(false); }}
            className={`px-3 py-1 rounded text-xs ${selected === i ? "bg-indigo-600 text-white" : "bg-gray-700 text-gray-300 hover:bg-gray-600"}`}
          >
            {t.name}
          </button>
        ))}
      </div>

      <p className="text-xs text-gray-400 mb-3">{type.desc}</p>
      <p className="text-xs text-yellow-400/80 mb-4">适用: {type.useCase}</p>

      <div className="flex gap-2 mb-4">
        <button onClick={() => setPlaying(!playing)}
          className="px-3 py-1 bg-gray-700 rounded text-xs text-gray-300 hover:bg-gray-600">
          {playing ? <Pause className="w-3 h-3 inline" /> : <Play className="w-3 h-3 inline" />}
        </button>
        <button onClick={() => { setStep(0); setPlaying(false); }}
          className="px-3 py-1 bg-gray-700 rounded text-xs text-gray-300 hover:bg-gray-600">
          <RotateCcw className="w-3 h-3 inline" />
        </button>
        <span className="text-xs text-gray-400 self-center">时隙 {step + 1}/16</span>
      </div>

      <div className="space-y-2">
        {type.names.map((name, di) => (
          <div key={di} className="flex items-center gap-2">
            <span className="w-16 text-xs text-gray-400 text-right">{name}</span>
            <div className="flex-1 flex gap-0.5">
              {type.slots[di].map((active, ti) => (
                <motion.div key={ti}
                  className="flex-1 h-7 rounded-sm"
                  animate={{
                    backgroundColor: ti === step ? (active ? "#818cf8" : "#4b5563") :
                      active ? type.colors[di].replace("bg-", "").replace("-500", "") === "red" ? "#dc2626" :
                      type.colors[di].replace("bg-", "").replace("-500", "") === "blue" ? "#2563eb" :
                      type.colors[di].replace("bg-", "").replace("-500", "") === "green" ? "#16a34a" :
                      type.colors[di].replace("bg-", "").replace("-500", "") === "yellow" ? "#ca8a04" :
                      type.colors[di].replace("bg-", "").replace("-500", "") === "purple" ? "#9333ea" :
                      type.colors[di].replace("bg-", "").replace("-500", "") === "pink" ? "#db2777" :
                      type.colors[di].replace("bg-", "").replace("-500", "") === "cyan" ? "#06b6d4" : "#4b5563"
                      : "#1f2937",
                    scale: ti === step ? 1.1 : 1,
                  }}
                  transition={{ duration: 0.15 }}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-3 flex gap-2">
        {type.names.map((name, i) => (
          <span key={i} className="flex items-center gap-1 text-xs">
            <span className={`w-2 h-2 rounded ${type.colors[i]}`} />{name}
          </span>
        ))}
      </div>
    </div>
  );
}
