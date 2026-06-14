"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";

export function MemoryBandwidthCalc() {
  const [chipWidth, setChipWidth] = useState(8);
  const [numChips, setNumChips] = useState(4);
  const [accessTime, setAccessTime] = useState(100);
  const [cycleTime, setCycleTime] = useState(150);
  const [numBanks, setNumBanks] = useState(4);

  const result = useMemo(() => {
    const dataWidth = chipWidth * numChips;
    const bandwidthSequential = dataWidth / (accessTime * 1e-9) / 1e9;
    const bandwidthInterleaved = dataWidth * numBanks / (cycleTime * 1e-9) / 1e9;
    const sequentialLatency = accessTime;
    const interleavedLatency = cycleTime;
    const speedup = bandwidthInterleaved / bandwidthSequential;
    return { dataWidth, bandwidthSequential, bandwidthInterleaved, sequentialLatency, interleavedLatency, speedup };
  }, [chipWidth, numChips, accessTime, cycleTime, numBanks]);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">存储器带宽计算器</h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">Memory Bandwidth Calculator</p>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-4">
        <div>
          <label className="block text-sm font-medium mb-1 text-slate-700 dark:text-gray-200">芯片数据线: {chipWidth}位</label>
          <input type="range" min={4} max={32} step={4} value={chipWidth} onChange={(e) => setChipWidth(+e.target.value)} className="w-full accent-blue-500" />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1 text-slate-700 dark:text-gray-200">并联芯片数: {numChips}</label>
          <input type="range" min={1} max={8} value={numChips} onChange={(e) => setNumChips(+e.target.value)} className="w-full accent-blue-500" />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1 text-slate-700 dark:text-gray-200">访问时间: {accessTime}ns</label>
          <input type="range" min={10} max={500} step={10} value={accessTime} onChange={(e) => setAccessTime(+e.target.value)} className="w-full accent-blue-500" />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1 text-slate-700 dark:text-gray-200">周期时间: {cycleTime}ns</label>
          <input type="range" min={10} max={500} step={10} value={cycleTime} onChange={(e) => setCycleTime(+e.target.value)} className="w-full accent-blue-500" />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1 text-slate-700 dark:text-gray-200">交叉体数: {numBanks}</label>
          <input type="range" min={1} max={16} value={numBanks} onChange={(e) => setNumBanks(+e.target.value)} className="w-full accent-blue-500" />
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
        {[
          { label: "总数据线宽度", value: result.dataWidth + " 位", color: "text-blue-600" },
          { label: "顺序带宽", value: result.bandwidthSequential.toFixed(2) + " GB/s", color: "text-orange-600" },
          { label: "交叉带宽", value: result.bandwidthInterleaved.toFixed(2) + " GB/s", color: "text-green-600" },
          { label: "顺序延迟", value: result.sequentialLatency + " ns", color: "text-slate-600" },
          { label: "交叉延迟", value: result.interleavedLatency + " ns", color: "text-slate-600" },
          { label: "带宽提升", value: result.speedup.toFixed(2) + "x", color: "text-purple-600" },
        ].map((m, i) => (
          <motion.div key={i} className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700" animate={{ scale: [1, 1.02, 1] }} transition={{ duration: 0.3 }}>
            <p className="text-xs text-slate-500 mb-1">{m.label}</p>
            <p className={"text-lg font-bold " + m.color}>{m.value}</p>
          </motion.div>
        ))}
      </div>

      <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 mb-4">
        <p className="text-xs text-slate-500 mb-2">带宽对比</p>
        <div className="space-y-2">
          <div>
            <p className="text-xs text-orange-600 mb-0.5">顺序存储</p>
            <div className="h-6 bg-slate-100 dark:bg-slate-800 rounded overflow-hidden">
              <motion.div className="h-full bg-orange-400 rounded" animate={{ width: Math.min(100, (result.bandwidthSequential / result.bandwidthInterleaved) * 100) + "%" }} />
            </div>
          </div>
          <div>
            <p className="text-xs text-green-600 mb-0.5">低位交叉存储</p>
            <div className="h-6 bg-slate-100 dark:bg-slate-800 rounded overflow-hidden">
              <motion.div className="h-full bg-green-400 rounded" animate={{ width: "100%" }} />
            </div>
          </div>
        </div>
      </div>

      <div className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg text-xs text-slate-600 dark:text-slate-400 space-y-1">
        <p className="font-semibold">计算公式:</p>
        <p>顺序带宽 = 数据宽度 / 访问时间</p>
        <p>交叉带宽 = 数据宽度 {"×"} 体数 / 周期时间</p>
        <p>加速比 = 交叉带宽 / 顺序带宽 = (体数 {"×"} 访问时间) / 周期时间</p>
      </div>
    </div>
  );
}

export default MemoryBandwidthCalc;
