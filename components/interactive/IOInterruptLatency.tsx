"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Timer, TrendingDown } from "lucide-react";

export function IOInterruptLatency() {
  const [instrTime, setInstrTime] = useState(1);
  const [queryTime, setQueryTime] = useState(5);
  const [deviceInterval, setDeviceInterval] = useState(100);
  const [isrOverhead, setIsrOverhead] = useState(20);

  const pollingLatency = deviceInterval + queryTime;
  const pollingUtilization = (queryTime / deviceInterval) * 100;
  const interruptLatency = instrTime + isrOverhead;
  const interruptUtilization = (isrOverhead / deviceInterval) * 100;
  const improvement = ((pollingUtilization - interruptUtilization) / pollingUtilization) * 100;

  const timelineWidth = 500;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Timer className="w-5 h-5 text-violet-400" />
        <h3 className="text-lg font-semibold">I/O 中断延迟分析</h3>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {[
          { label: "指令周期 (μs)", value: instrTime, set: setInstrTime, min: 0.1, max: 10 },
          { label: "查询开销 (μs)", value: queryTime, set: setQueryTime, min: 1, max: 100 },
          { label: "设备间隔 (μs)", value: deviceInterval, set: setDeviceInterval, min: 10, max: 1000 },
          { label: "ISR开销 (μs)", value: isrOverhead, set: setIsrOverhead, min: 5, max: 200 },
        ].map((inp) => (
          <div key={inp.label}>
            <label className="text-xs text-gray-400 block mb-1">{inp.label}</label>
            <input type="number" value={inp.value}
              onChange={(e) => inp.set(Number(e.target.value))}
              min={inp.min} max={inp.max}
              className="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded text-sm text-white" />
          </div>
        ))}
      </div>

      <div className="space-y-6">
        <div>
          <div className="text-xs text-gray-400 mb-2">程序查询方式时间线</div>
          <div className="relative h-8 bg-gray-800 rounded overflow-hidden" style={{ width: timelineWidth }}>
            {Array.from({ length: 5 }).map((_, i) => (
              <motion.div key={i}
                className="absolute top-0 h-full bg-red-500/30 border-r border-red-500/50"
                style={{ left: (i * deviceInterval / (5 * deviceInterval)) * 100 + "%", width: (queryTime / (5 * deviceInterval)) * 100 + "%" }}
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ repeat: Infinity, duration: 2, delay: i * 0.3 }}
              />
            ))}
            <div className="absolute inset-0 flex items-center justify-center text-xs text-red-300">
              CPU忙等待查询 ({pollingUtilization.toFixed(1)}% 利用率)
            </div>
          </div>
        </div>

        <div>
          <div className="text-xs text-gray-400 mb-2">中断驱动方式时间线</div>
          <div className="relative h-8 bg-gray-800 rounded overflow-hidden" style={{ width: timelineWidth }}>
            <motion.div
              className="absolute top-0 h-full bg-green-500/30 border-r border-green-500/50"
              style={{ width: (instrTime / (5 * deviceInterval)) * 100 + "%" }}
              animate={{ left: ["0%", "5%", "0%"] }}
              transition={{ repeat: Infinity, duration: 3 }}
            />
            <motion.div
              className="absolute top-0 h-full bg-blue-500/30 border-r border-blue-500/50"
              style={{ left: (instrTime / (5 * deviceInterval)) * 100 + "%", width: (isrOverhead / (5 * deviceInterval)) * 100 + "%" }}
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ repeat: Infinity, duration: 2 }}
            />
            <div className="absolute inset-0 flex items-center justify-center text-xs text-green-300">
              CPU可执行其他任务 ({interruptUtilization.toFixed(1)}% 利用率)
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 mt-6">
        <div className="p-3 bg-red-500/10 rounded-lg text-center">
          <div className="text-xs text-gray-400">查询延迟</div>
          <div className="text-lg font-bold text-red-300">{pollingLatency.toFixed(1)} μs</div>
        </div>
        <div className="p-3 bg-green-500/10 rounded-lg text-center">
          <div className="text-xs text-gray-400">中断延迟</div>
          <div className="text-lg font-bold text-green-300">{interruptLatency.toFixed(1)} μs</div>
        </div>
        <div className="p-3 bg-violet-500/10 rounded-lg text-center">
          <div className="text-xs text-gray-400 flex items-center justify-center gap-1"><TrendingDown className="w-3 h-3" />CPU利用率改善</div>
          <div className="text-lg font-bold text-violet-300">{improvement.toFixed(1)}%</div>
        </div>
      </div>
    </div>
  );
}
