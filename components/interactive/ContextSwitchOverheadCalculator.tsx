"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { Calculator, BarChart3, Info } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const REFERENCE_DATA = [
  { name: "lmbench (Linux 5.x)", value: 2.1, source: "lmbench lat_ctx" },
  { name: "lmbench (Linux 4.x)", value: 3.5, source: "lmbench lat_ctx" },
  { name: "xv6 (QEMU est.)", value: 50, source: "Simulated" },
  { name: "MicroPython", value: 120, source: "Estimated" },
];

export default function ContextSwitchOverheadCalculator() {
  const [cpuFreq, setCpuFreq] = useState(2.5);
  const [l1Size, setL1Size] = useState(64);
  const [l2Size, setL2Size] = useState(1024);
  const [tlbEntries, setTlbEntries] = useState(256);
  const [workingSet, setWorkingSet] = useState(512);

  const result = useMemo(() => {
    // Direct costs (in CPU cycles)
    const regSaveRestore = 14 * 2; // 14 regs, ~2 cycles each for store + load
    const tlbFlush = tlbEntries > 0 ? 50 : 0; // TLB invalidation cost
    const pageTableSwitch = 20; // satp write + fence
    const directCycles = regSaveRestore + tlbFlush + pageTableSwitch;
    const directUs = directCycles / (cpuFreq * 1e3); // GHz -> cycles/ns, /1000 -> us

    // Indirect costs
    // Cache miss estimation: fraction of working set that exceeds cache
    const totalCache = l1Size + l2Size;
    const missFraction = workingSet > totalCache
      ? Math.min(1, (workingSet - totalCache) / workingSet)
      : workingSet > l1Size
      ? (workingSet - l1Size) / workingSet * 0.3
      : 0.02;
    const l1MissPenalty = 4; // cycles
    const l2MissPenalty = 12; // cycles (to L3/main memory if L2 miss)
    const cacheMisses = Math.round(missFraction * (workingSet / 64)); // cache lines
    const cacheMissCycles = cacheMisses * (missFraction > 0.5 ? l2MissPenalty : l1MissPenalty);
    const cacheMissUs = cacheMissCycles / (cpuFreq * 1e3);

    // TLB miss cost
    const tlbCoverage = tlbEntries * 4; // 4KB pages
    const tlbMissFraction = workingSet > tlbCoverage
      ? Math.min(0.8, (workingSet - tlbCoverage) / workingSet)
      : 0.01;
    const tlbMissPenalty = 20; // page table walk cycles
    const tlbMisses = Math.round(tlbMissFraction * (workingSet / 4));
    const tlbMissCycles = tlbMisses * tlbMissPenalty;
    const tlbMissUs = tlbMissCycles / (cpuFreq * 1e3);

    const indirectUs = cacheMissUs + tlbMissUs;
    const totalUs = directUs + indirectUs;

    return {
      directUs: Math.max(0.1, directUs),
      cacheMissUs: Math.max(0.01, cacheMissUs),
      tlbMissUs: Math.max(0.01, tlbMissUs),
      indirectUs: Math.max(0.1, indirectUs),
      totalUs: Math.max(0.2, totalUs),
      directCycles,
      cacheMisses,
      tlbMisses,
    };
  }, [cpuFreq, l1Size, l2Size, tlbEntries, workingSet]);

  const chartData = [
    { name: "Reg Save/Restore", value: parseFloat(result.directUs.toFixed(2)), fill: "#3b82f6" },
    { name: "Cache Miss", value: parseFloat(result.cacheMissUs.toFixed(2)), fill: "#f59e0b" },
    { name: "TLB Miss", value: parseFloat(result.tlbMissUs.toFixed(2)), fill: "#ef4444" },
  ];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center flex items-center justify-center gap-2">
        <Calculator className="w-7 h-7 text-violet-600 dark:text-violet-400" />
        Context Switch Overhead Calculator
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sliders */}
        <div className="space-y-5">
          <h3 className="text-sm font-bold text-slate-700 dark:text-gray-300 uppercase tracking-wider">
            Parameters
          </h3>

          {/* CPU Frequency */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-slate-700 dark:text-gray-300 font-medium">CPU Frequency</span>
              <span className="font-mono text-violet-600 dark:text-violet-400 font-bold">
                {cpuFreq.toFixed(1)} GHz
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="5"
              step="0.1"
              value={cpuFreq}
              onChange={(e) => setCpuFreq(parseFloat(e.target.value))}
              className="w-full h-2 bg-slate-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-violet-600"
            />
            <div className="flex justify-between text-[10px] text-slate-400 dark:text-gray-500">
              <span>1 GHz</span><span>5 GHz</span>
            </div>
          </div>

          {/* L1 Cache */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-slate-700 dark:text-gray-300 font-medium">L1 Cache Size</span>
              <span className="font-mono text-violet-600 dark:text-violet-400 font-bold">
                {l1Size} KB
              </span>
            </div>
            <input
              type="range"
              min="32"
              max="128"
              step="32"
              value={l1Size}
              onChange={(e) => setL1Size(parseInt(e.target.value))}
              className="w-full h-2 bg-slate-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-violet-600"
            />
            <div className="flex justify-between text-[10px] text-slate-400 dark:text-gray-500">
              <span>32 KB</span><span>128 KB</span>
            </div>
          </div>

          {/* L2 Cache */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-slate-700 dark:text-gray-300 font-medium">L2 Cache Size</span>
              <span className="font-mono text-violet-600 dark:text-violet-400 font-bold">
                {l2Size} KB
              </span>
            </div>
            <input
              type="range"
              min="256"
              max="4096"
              step="256"
              value={l2Size}
              onChange={(e) => setL2Size(parseInt(e.target.value))}
              className="w-full h-2 bg-slate-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-violet-600"
            />
            <div className="flex justify-between text-[10px] text-slate-400 dark:text-gray-500">
              <span>256 KB</span><span>4096 KB</span>
            </div>
          </div>

          {/* TLB Entries */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-slate-700 dark:text-gray-300 font-medium">TLB Entries</span>
              <span className="font-mono text-violet-600 dark:text-violet-400 font-bold">
                {tlbEntries}
              </span>
            </div>
            <input
              type="range"
              min="64"
              max="1024"
              step="64"
              value={tlbEntries}
              onChange={(e) => setTlbEntries(parseInt(e.target.value))}
              className="w-full h-2 bg-slate-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-violet-600"
            />
            <div className="flex justify-between text-[10px] text-slate-400 dark:text-gray-500">
              <span>64</span><span>1024</span>
            </div>
          </div>

          {/* Working Set Size */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-slate-700 dark:text-gray-300 font-medium">
                Working Set Size
              </span>
              <span className="font-mono text-violet-600 dark:text-violet-400 font-bold">
                {workingSet} KB
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="8192"
              step="64"
              value={workingSet}
              onChange={(e) => setWorkingSet(parseInt(e.target.value))}
              className="w-full h-2 bg-slate-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-violet-600"
            />
            <div className="flex justify-between text-[10px] text-slate-400 dark:text-gray-500">
              <span>0 KB</span><span>8192 KB</span>
            </div>
          </div>
        </div>

        {/* Results */}
        <div className="space-y-4">
          {/* Total latency card */}
          <motion.div
            key={result.totalUs.toFixed(2)}
            initial={{ scale: 0.95 }}
            animate={{ scale: 1 }}
            className="p-5 rounded-xl bg-gradient-to-br from-violet-600 to-purple-700 text-white text-center shadow-lg"
          >
            <p className="text-sm opacity-80 mb-1">Estimated Total Latency</p>
            <p className="text-4xl font-bold font-mono">{result.totalUs.toFixed(2)} <span className="text-lg">us</span></p>
            <p className="text-xs opacity-60 mt-1">
              {result.directCycles} direct cycles + {result.cacheMisses + result.tlbMisses} miss cycles
            </p>
          </motion.div>

          {/* Breakdown bars */}
          <div className="p-4 rounded-lg bg-white dark:bg-gray-800 border border-slate-200 dark:border-gray-700">
            <h4 className="text-sm font-bold text-slate-700 dark:text-gray-300 mb-3 flex items-center gap-1">
              <BarChart3 className="w-4 h-4" />
              Cost Breakdown
            </h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 30 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" tick={{ fontSize: 11 }} unit=" us" />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={100} />
                <Tooltip
                  formatter={(value: number | undefined) => [`${value ?? 0} us`, "Latency"]}
                  contentStyle={{
                    backgroundColor: "#1e293b",
                    border: "none",
                    borderRadius: "8px",
                    color: "#f1f5f9",
                    fontSize: 12,
                  }}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={index} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Detail numbers */}
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700">
              <p className="text-xs text-blue-600 dark:text-blue-400 font-semibold">Direct Cost</p>
              <p className="text-lg font-bold font-mono text-blue-800 dark:text-blue-200">
                {result.directUs.toFixed(2)} us
              </p>
              <p className="text-[10px] text-blue-500 dark:text-blue-400">
                Reg save/restore + addr space
              </p>
            </div>
            <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700">
              <p className="text-xs text-amber-600 dark:text-amber-400 font-semibold">Indirect Cost</p>
              <p className="text-lg font-bold font-mono text-amber-800 dark:text-amber-200">
                {result.indirectUs.toFixed(2)} us
              </p>
              <p className="text-[10px] text-amber-500 dark:text-amber-400">
                Cache misses ({result.cacheMisses}) + TLB misses ({result.tlbMisses})
              </p>
            </div>
          </div>

          {/* Reference values */}
          <div className="p-4 rounded-lg bg-white dark:bg-gray-800 border border-slate-200 dark:border-gray-700">
            <h4 className="text-sm font-bold text-slate-700 dark:text-gray-300 mb-2 flex items-center gap-1">
              <Info className="w-4 h-4" />
              Real-World References (lmbench)
            </h4>
            <div className="space-y-1">
              {REFERENCE_DATA.map((ref) => (
                <div key={ref.name} className="flex items-center gap-2 text-xs">
                  <div
                    className="h-2 rounded-full bg-violet-400"
                    style={{ width: `${Math.min(100, ref.value)}%`, minWidth: 4 }}
                  />
                  <span className="text-slate-700 dark:text-gray-300 shrink-0">
                    {ref.name}
                  </span>
                  <span className="ml-auto font-mono text-slate-500 dark:text-gray-400">
                    {ref.value} us
                  </span>
                </div>
              ))}
            </div>
            <p className="mt-2 text-[10px] text-slate-400 dark:text-gray-500">
              Note: xv6 on QEMU is not cycle-accurate; real hardware values vary significantly.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
