"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { ArrowRightLeft, Play, Pause, RotateCcw } from "lucide-react";

const totalCycles = 30;
const dmaStealPoints = [5, 6, 11, 12, 17, 18, 23, 24];

const cpuInstructions = [
  { start: 0, end: 4, label: "指令1" },
  { start: 7, end: 10, label: "指令2 (被挪用)" },
  { start: 13, end: 16, label: "指令3 (被挪用)" },
  { start: 19, end: 22, label: "指令4 (被挪用)" },
  { start: 25, end: 29, label: "指令5" },
];

export function CycleStealingDemo() {
  const [cycle, setCycle] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [stats, setStats] = useState({ stolen: 0, cpuCycles: 0 });

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      setCycle((c) => {
        const next = (c + 1) % totalCycles;
        if (dmaStealPoints.includes(next)) {
          setStats((s) => ({ ...s, stolen: s.stolen + 1 }));
        } else {
          setStats((s) => ({ ...s, cpuCycles: s.cpuCycles + 1 }));
        }
        return next;
      });
    }, 300);
    return () => clearInterval(timer);
  }, [playing]);

  const handleReset = () => {
    setCycle(0);
    setPlaying(false);
    setStats({ stolen: 0, cpuCycles: 0 });
  };

  const isDMA = dmaStealPoints.includes(cycle);
  const currentInstr = cpuInstructions.find((i) => cycle >= i.start && cycle <= i.end);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <ArrowRightLeft className="w-5 h-5 text-orange-400" />
        <h3 className="text-lg font-semibold">周期挪用演示</h3>
      </div>

      <div className="flex gap-2 mb-4">
        <button onClick={() => setPlaying(!playing)}
          className="px-4 py-1.5 bg-orange-600 rounded text-sm text-white hover:bg-orange-500 flex items-center gap-1">
          {playing ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
          {playing ? "暂停" : "播放"}
        </button>
        <button onClick={handleReset}
          className="px-3 py-1.5 bg-gray-700 rounded text-sm text-gray-300 hover:bg-gray-600">
          <RotateCcw className="w-3 h-3 inline mr-1" />重置
        </button>
        <span className="text-xs text-gray-400 self-center">周期 {cycle + 1}/{totalCycles}</span>
      </div>

      <div className="mb-2 text-xs text-gray-400">总线占用时间线</div>
      <div className="flex gap-0.5 mb-4">
        {Array.from({ length: totalCycles }).map((_, i) => {
          const isDmaCycle = dmaStealPoints.includes(i);
          const isCurrent = i === cycle;
          return (
            <motion.div key={i}
              className="flex-1 h-8 rounded-sm flex items-center justify-center text-[9px]"
              animate={{
                backgroundColor: isCurrent ? (isDmaCycle ? "#f97316" : "#3b82f6") : isDmaCycle ? "#92400e" : "#1e3a5f",
                scale: isCurrent ? 1.15 : 1,
              }}
              transition={{ duration: 0.15 }}
            >
              {isDmaCycle && <span className="text-orange-200">D</span>}
            </motion.div>
          );
        })}
      </div>

      <div className="mb-4 text-xs text-gray-400">CPU 指令执行</div>
      <div className="relative h-12 bg-gray-800 rounded mb-4">
        {cpuInstructions.map((instr) => (
          <motion.div key={instr.label}
            className="absolute top-1 h-10 rounded flex items-center justify-center text-[10px] text-blue-200 bg-blue-600/40 border border-blue-500/50"
            style={{
              left: `${(instr.start / totalCycles) * 100}%`,
              width: `${((instr.end - instr.start + 1) / totalCycles) * 100}%`,
            }}
            animate={{ opacity: currentInstr?.label === instr.label ? 1 : 0.5 }}
          >
            {instr.label}
          </motion.div>
        ))}
        {dmaStealPoints.map((p) => (
          <motion.div key={p}
            className="absolute top-0 h-full w-0.5 bg-orange-500/50"
            style={{ left: `${(p / totalCycles) * 100}%` }}
          />
        ))}
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div className="p-2 bg-gray-800/30 rounded text-center">
          <div className="text-xs text-gray-400">当前状态</div>
          <div className={`text-sm font-bold ${isDMA ? "text-orange-400" : "text-blue-400"}`}>
            {isDMA ? "DMA 挪用" : "CPU 执行"}
          </div>
        </div>
        <div className="p-2 bg-gray-800/30 rounded text-center">
          <div className="text-xs text-gray-400">挪用周期</div>
          <div className="text-sm font-bold text-orange-300">{stats.stolen}</div>
        </div>
        <div className="p-2 bg-gray-800/30 rounded text-center">
          <div className="text-xs text-gray-400">CPU周期</div>
          <div className="text-sm font-bold text-blue-300">{stats.cpuCycles}</div>
        </div>
      </div>

      <motion.div className="mt-3 p-2 bg-yellow-500/10 rounded text-xs text-yellow-300"
        animate={{ opacity: isDMA ? 1 : 0.5 }}>
        DMA每传送一个字就"偷取"一个总线周期，CPU暂停一个周期但不丢失状态
      </motion.div>
    </div>
  );
}
