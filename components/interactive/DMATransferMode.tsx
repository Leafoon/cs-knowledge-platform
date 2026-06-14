"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Monitor, Play, Pause, RotateCcw } from "lucide-react";

const modes = [
  {
    name: "CPU停止法",
    desc: "DMA请求时CPU完全放弃总线，直到传送结束",
    cpuBlocks: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    dmaBlocks: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
  },
  {
    name: "周期挪用法",
    desc: "DMA每次只占用一个总线周期，传送完一个字后归还",
    cpuBlocks: [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    dmaBlocks: [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
  },
  {
    name: "交替访问法",
    desc: "CPU和DMA交替使用总线，CPU和DMA各占半个周期",
    cpuBlocks: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    dmaBlocks: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
  },
];

export function DMATransferMode() {
  const [selectedMode, setSelectedMode] = useState(0);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const mode = modes[selectedMode];

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => setStep((s) => (s + 1) % 16), 400);
    return () => clearInterval(timer);
  }, [playing]);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Monitor className="w-5 h-5 text-teal-400" />
        <h3 className="text-lg font-semibold">DMA 传送方式</h3>
      </div>

      <div className="flex gap-2 mb-3">
        {modes.map((m, i) => (
          <button key={m.name}
            onClick={() => { setSelectedMode(i); setStep(0); }}
            className={`px-3 py-1 rounded text-xs ${selectedMode === i ? "bg-teal-600 text-white" : "bg-gray-700 text-gray-300 hover:bg-gray-600"}`}
          >
            {m.name}
          </button>
        ))}
      </div>

      <p className="text-xs text-gray-400 mb-4">{mode.desc}</p>

      <div className="flex gap-2 mb-4">
        <button onClick={() => setPlaying(!playing)}
          className="px-3 py-1 bg-gray-700 rounded text-xs text-gray-300 hover:bg-gray-600">
          {playing ? <Pause className="w-3 h-3 inline" /> : <Play className="w-3 h-3 inline" />}
        </button>
        <button onClick={() => { setStep(0); setPlaying(false); }}
          className="px-3 py-1 bg-gray-700 rounded text-xs text-gray-300 hover:bg-gray-600">
          <RotateCcw className="w-3 h-3 inline" />
        </button>
        <span className="text-xs text-gray-400 self-center">周期 {step + 1}/16</span>
      </div>

      <div className="space-y-2">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-10 text-xs text-blue-400">CPU</span>
            <div className="flex-1 flex gap-0.5">
              {mode.cpuBlocks.map((active, i) => (
                <motion.div key={i}
                  className="flex-1 h-6 rounded-sm"
                  animate={{
                    backgroundColor: i === step ? "#60a5fa" : active ? "#2563eb" : "#1f2937",
                    scale: i === step ? 1.1 : 1,
                  }}
                  transition={{ duration: 0.15 }}
                />
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-10 text-xs text-teal-400">DMA</span>
            <div className="flex-1 flex gap-0.5">
              {mode.dmaBlocks.map((active, i) => (
                <motion.div key={i}
                  className="flex-1 h-6 rounded-sm"
                  animate={{
                    backgroundColor: i === step ? "#2dd4bf" : active ? "#0d9488" : "#1f2937",
                    scale: i === step ? 1.1 : 1,
                  }}
                  transition={{ duration: 0.15 }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3">
        <div className="p-2 bg-blue-500/10 rounded text-xs text-blue-300">
          CPU占用: {((mode.cpuBlocks.filter(Boolean).length / 16) * 100).toFixed(0)}%
        </div>
        <div className="p-2 bg-teal-500/10 rounded text-xs text-teal-300">
          DMA占用: {((mode.dmaBlocks.filter(Boolean).length / 16) * 100).toFixed(0)}%
        </div>
      </div>
    </div>
  );
}
