"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Clock, Play, Pause, RotateCcw } from "lucide-react";

const modes = [
  {
    name: "同步方式",
    desc: "所有信号由统一时钟同步，操作在时钟边沿完成。",
    signals: ["CLK", "ADDR", "DATA", "RD/WR#"],
    pattern: [
      [1, 0, 1, 0, 1, 0, 1, 0],
      [1, 1, 1, 0, 0, 0, 1, 1],
      [0, 0, 1, 1, 1, 1, 0, 0],
      [1, 1, 1, 1, 0, 0, 0, 0],
    ],
  },
  {
    name: "异步方式",
    desc: "无统一时钟，用请求/应答握手信号协调。",
    signals: ["REQ", "ACK", "ADDR", "DATA"],
    pattern: [
      [0, 0, 1, 1, 1, 1, 0, 0],
      [0, 0, 0, 0, 1, 1, 1, 1],
      [1, 1, 1, 1, 0, 0, 0, 0],
      [0, 0, 1, 1, 1, 1, 0, 0],
    ],
  },
  {
    name: "半同步方式",
    desc: "时钟同步 + 握手等待，兼具两者优点。",
    signals: ["CLK", "WAIT#", "ADDR", "DATA"],
    pattern: [
      [1, 0, 1, 0, 1, 0, 1, 0],
      [1, 1, 0, 0, 0, 0, 1, 1],
      [1, 1, 1, 0, 0, 0, 1, 1],
      [0, 0, 1, 1, 1, 1, 0, 0],
    ],
  },
  {
    name: "分离式方式",
    desc: "总线事务拆分为请求和响应两个独立阶段，释放总线。",
    signals: ["BUS", "ADDR", "DATA", "DONE"],
    pattern: [
      [1, 0, 0, 0, 0, 0, 1, 0],
      [1, 1, 0, 0, 0, 0, 1, 1],
      [0, 0, 0, 0, 1, 1, 0, 0],
      [0, 0, 0, 0, 0, 1, 0, 1],
    ],
  },
];

export function BusCommunicationTiming() {
  const [selectedMode, setSelectedMode] = useState(0);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const mode = modes[selectedMode];

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => setStep((s) => (s + 1) % 8), 500);
    return () => clearInterval(timer);
  }, [playing]);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Clock className="w-5 h-5 text-yellow-400" />
        <h3 className="text-lg font-semibold">总线通信时序</h3>
      </div>

      <div className="flex gap-2 mb-3">
        {modes.map((m, i) => (
          <button
            key={m.name}
            onClick={() => { setSelectedMode(i); setStep(0); }}
            className={`px-3 py-1 rounded text-xs ${selectedMode === i ? "bg-yellow-600 text-white" : "bg-gray-700 text-gray-300 hover:bg-gray-600"}`}
          >
            {m.name}
          </button>
        ))}
      </div>

      <p className="text-xs text-gray-400 mb-4">{mode.desc}</p>

      <div className="flex gap-2 mb-4">
        <button onClick={() => setPlaying(!playing)} className="px-3 py-1 bg-gray-700 rounded text-xs text-gray-300 hover:bg-gray-600">
          {playing ? <Pause className="w-3 h-3 inline" /> : <Play className="w-3 h-3 inline" />}
        </button>
        <button onClick={() => { setStep(0); setPlaying(false); }} className="px-3 py-1 bg-gray-700 rounded text-xs text-gray-300 hover:bg-gray-600">
          <RotateCcw className="w-3 h-3 inline" />
        </button>
        <span className="text-xs text-gray-400 self-center">时钟周期: {step + 1}/8</span>
      </div>

      <div className="space-y-2">
        {mode.signals.map((sig, si) => (
          <div key={sig} className="flex items-center gap-2 h-10">
            <span className="w-16 text-xs font-mono text-right text-gray-400">{sig}</span>
            <div className="flex-1 flex gap-0.5 items-end">
              {mode.pattern[si].map((val, ci) => {
                const isActive = val === 1;
                const isCurrent = ci === step;
                return (
                  <motion.div
                    key={ci}
                    className="flex-1 rounded-sm"
                    animate={{
                      height: isActive ? "100%" : "30%",
                      backgroundColor: isCurrent ? "#facc15" : isActive ? "#3b82f6" : "#374151",
                    }}
                    transition={{ duration: 0.2 }}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-3 flex gap-0.5">
        {Array.from({ length: 8 }).map((_, i) => (
          <div
            key={i}
            className={`flex-1 h-1 rounded ${i === step ? "bg-yellow-400" : "bg-gray-700"}`}
          />
        ))}
      </div>
    </div>
  );
}
