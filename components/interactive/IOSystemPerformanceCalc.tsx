"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Calculator, Gauge } from "lucide-react";

export function IOSystemPerformanceCalc() {
  const [dataRate, setDataRate] = useState(100);
  const [cpuCycleTime, setCpuCycleTime] = useState(1);
  const [queryCycles, setQueryCycles] = useState(50);
  const [transferCycles, setTransferCycles] = useState(20);
  const [deviceReadyRatio, setDeviceReadyRatio] = useState(50);

  const cpuFreq = 1000 / cpuCycleTime;
  const queriesPerTransfer = 100 / deviceReadyRatio;
  const totalQueryCycles = queriesPerTransfer * queryCycles;
  const totalCyclesPerByte = totalQueryCycles + transferCycles;
  const cpuUtilization = (totalQueryCycles / totalCyclesPerByte) * 100;
  const effectiveRate = (cpuFreq / totalCyclesPerByte) * 1;
  const throughputRatio = (effectiveRate / dataRate) * 100;

  const metrics = [
    {
      label: "CPU 利用率 (查询开销)",
      value: cpuUtilization,
      max: 100,
      color: "red",
      desc: "CPU在查询上浪费的时间比例",
    },
    {
      label: "有效数据传输率",
      value: Math.min(100, throughputRatio),
      max: 100,
      color: "green",
      desc: `实际速率: ${effectiveRate.toFixed(1)} KB/s（设备速率: ${dataRate} KB/s）`,
    },
    {
      label: "每字节总CPU周期",
      value: Math.min(100, totalCyclesPerByte / 10),
      max: 100,
      color: "blue",
      desc: `每字节需要 ${totalCyclesPerByte.toFixed(0)} 个CPU周期`,
    },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Calculator className="w-5 h-5 text-rose-400" />
        <h3 className="text-lg font-semibold">I/O 系统性能计算器</h3>
      </div>

      <div className="text-xs text-gray-400 mb-3">程序查询方式性能分析</div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
        {[
          { label: "设备数据率 (KB/s)", value: dataRate, set: setDataRate, min: 1, max: 10000 },
          { label: "CPU周期时间 (ns)", value: cpuCycleTime, set: setCpuCycleTime, min: 0.1, max: 10 },
          { label: "每次查询周期数", value: queryCycles, set: setQueryCycles, min: 5, max: 500 },
          { label: "传输数据周期数", value: transferCycles, set: setTransferCycles, min: 5, max: 200 },
          { label: "设备就绪概率 (%)", value: deviceReadyRatio, set: setDeviceReadyRatio, min: 5, max: 100 },
        ].map((inp) => (
          <div key={inp.label}>
            <label className="text-xs text-gray-400 block mb-1">{inp.label}</label>
            <input
              type="number"
              value={inp.value}
              onChange={(e) => inp.set(Number(e.target.value))}
              min={inp.min}
              max={inp.max}
              className="w-full px-3 py-1.5 bg-gray-800 border border-gray-600 rounded text-sm text-white"
            />
          </div>
        ))}
      </div>

      <div className="space-y-4">
        {metrics.map((m, i) => (
          <div key={m.label}>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">{m.label}</span>
              <span className={`text-${m.color}-300`}>{m.value.toFixed(1)}%</span>
            </div>
            <div className="h-4 bg-gray-800 rounded-full overflow-hidden">
              <motion.div
                className={`h-full rounded-full bg-${m.color}-500/60`}
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(100, m.value)}%` }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
              />
            </div>
            <p className="text-[10px] text-gray-500 mt-1">{m.desc}</p>
          </div>
        ))}
      </div>

      <motion.div
        className="mt-4 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg text-xs text-yellow-300"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        <Gauge className="w-4 h-4 inline mr-1" />
        结论: 程序查询方式下CPU利用率高达 {cpuUtilization.toFixed(1)}%，效率低下。
        {cpuUtilization > 50 ? "建议使用中断或DMA方式。" : "设备就绪概率较高时查询方式尚可接受。"}
      </motion.div>
    </div>
  );
}
