"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Calculator, Activity } from "lucide-react";

export function DMAPerformanceCalc() {
  const [busWidth, setBusWidth] = useState(32);
  const [clockFreq, setClockFreq] = useState(100);
  const [dmaOverhead, setDmaOverhead] = useState(5);
  const [transferSize, setTransferSize] = useState(1024);

  const bytesPerTransfer = busWidth / 8;
  const totalCycles = (transferSize / bytesPerTransfer) * (1 + dmaOverhead);
  const transferTime = totalCycles / (clockFreq * 1e6) * 1e6;
  const throughput = transferSize / (transferTime * 1e-6) / 1e6;
  const maxThroughput = (clockFreq * 1e6 * bytesPerTransfer) / (1 + dmaOverhead) / 1e6;
  const cpuSaving = ((1 - dmaOverhead / (1 + dmaOverhead)) * 100);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Calculator className="w-5 h-5 text-emerald-400" />
        <h3 className="text-lg font-semibold">DMA 性能计算器</h3>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {[
          { label: "总线宽度 (bit)", value: busWidth, set: setBusWidth, opts: [8, 16, 32, 64, 128] },
          { label: "时钟频率 (MHz)", value: clockFreq, set: setClockFreq, opts: [50, 100, 200, 400, 800] },
          { label: "DMA开销 (周期/字)", value: dmaOverhead, set: setDmaOverhead, opts: [1, 2, 3, 5, 10] },
          { label: "传输大小 (字节)", value: transferSize, set: setTransferSize, opts: [256, 512, 1024, 4096, 8192] },
        ].map((inp) => (
          <div key={inp.label}>
            <label className="text-xs text-gray-400 block mb-1">{inp.label}</label>
            <div className="flex gap-1 flex-wrap">
              {inp.opts.map((v) => (
                <button key={v}
                  onClick={() => inp.set(v)}
                  className={`px-2 py-1 rounded text-xs ${
                    inp.value === v ? "bg-emerald-600 text-white" : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                  }`}
                >
                  {v}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="p-3 bg-gray-800/30 rounded-lg">
          <div className="text-xs text-gray-400 mb-1">每次传送字节数</div>
          <div className="text-lg font-bold text-emerald-300">{bytesPerTransfer} B</div>
          <div className="text-xs text-gray-500">{busWidth}bit ÷ 8</div>
        </div>
        <div className="p-3 bg-gray-800/30 rounded-lg">
          <div className="text-xs text-gray-400 mb-1">总周期数</div>
          <div className="text-lg font-bold text-emerald-300">{totalCycles.toFixed(0)}</div>
          <div className="text-xs text-gray-500">{transferSize / bytesPerTransfer} × ({dmaOverhead}+1)</div>
        </div>
      </div>

      <div className="space-y-3">
        {[
          { label: "传送时间", value: transferTime.toFixed(2) + " μs", pct: Math.min(100, transferTime / 100), color: "blue" },
          { label: "实际吞吐率", value: throughput.toFixed(2) + " MB/s", pct: Math.min(100, (throughput / maxThroughput) * 100), color: "green" },
          { label: "CPU节省率", value: cpuSaving.toFixed(1) + "%", pct: cpuSaving, color: "emerald" },
        ].map((m, i) => (
          <div key={m.label}>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">{m.label}</span>
              <span className={`text-${m.color}-300`}>{m.value}</span>
            </div>
            <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
              <motion.div className={`h-full rounded-full bg-${m.color}-500/50`}
                initial={{ width: 0 }}
                animate={{ width: `${m.pct}%` }}
                transition={{ duration: 0.5, delay: i * 0.1 }} />
            </div>
          </div>
        ))}
      </div>

      <motion.div className="mt-4 p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-xs text-emerald-300"
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
        <Activity className="w-4 h-4 inline mr-1" />
        DMA方式下CPU仅在初始化和结束时参与，节省{cpuSaving.toFixed(0)}%的CPU时间。
        理论最大吞吐率: {maxThroughput.toFixed(1)} MB/s
      </motion.div>
    </div>
  );
}
