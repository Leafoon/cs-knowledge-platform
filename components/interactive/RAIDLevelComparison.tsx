"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Database, Shield, Zap, Info, ArrowRight } from "lucide-react";

type RAIDLevel = "RAID 0" | "RAID 1" | "RAID 4" | "RAID 5" | "RAID 6" | "RAID 10";

interface RAIDInfo {
  name: RAIDLevel;
  description: string;
  minDisks: number;
  capacity: string;
  faultTolerance: string;
  readPerf: number;
  writePerf: number;
  layout: (diskIdx: number, blockIdx: number, totalDisks: number) => "data" | "parity" | "double-parity" | "mirror" | "empty";
}

const RAID_CONFIGS: RAIDInfo[] = [
  {
    name: "RAID 0",
    description: "条带化：数据分散到所有磁盘，无冗余",
    minDisks: 2,
    capacity: "N × diskSize",
    faultTolerance: "0 块磁盘故障",
    readPerf: 5,
    writePerf: 5,
    layout: (_, blockIdx, totalDisks) => (blockIdx < totalDisks ? "data" : "empty"),
  },
  {
    name: "RAID 1",
    description: "镜像：数据完全复制到所有磁盘",
    minDisks: 2,
    capacity: "1 × diskSize",
    faultTolerance: "N-1 块磁盘故障",
    readPerf: 4,
    writePerf: 2,
    layout: (_, blockIdx, totalDisks) => (blockIdx < totalDisks ? "mirror" : "empty"),
  },
  {
    name: "RAID 4",
    description: "专用奇偶校验盘：数据条带化 + 专用校验盘",
    minDisks: 3,
    capacity: "(N-1) × diskSize",
    faultTolerance: "1 块磁盘故障",
    readPerf: 4,
    writePerf: 2,
    layout: (diskIdx, blockIdx, totalDisks) => {
      if (blockIdx >= 1) return "empty";
      return diskIdx === totalDisks - 1 ? "parity" : "data";
    },
  },
  {
    name: "RAID 5",
    description: "分布式奇偶校验：校验块分散到所有磁盘",
    minDisks: 3,
    capacity: "(N-1) × diskSize",
    faultTolerance: "1 块磁盘故障",
    readPerf: 4,
    writePerf: 3,
    layout: (diskIdx, blockIdx, totalDisks) => {
      if (blockIdx >= 1) return "empty";
      const parityPos = (totalDisks - 1 - (0 % totalDisks)) % totalDisks;
      return diskIdx === parityPos ? "parity" : "data";
    },
  },
  {
    name: "RAID 6",
    description: "双重分布式校验：可容忍 2 块磁盘同时故障",
    minDisks: 4,
    capacity: "(N-2) × diskSize",
    faultTolerance: "2 块磁盘故障",
    readPerf: 4,
    writePerf: 2,
    layout: (diskIdx, blockIdx, totalDisks) => {
      if (blockIdx >= 1) return "empty";
      if (diskIdx === totalDisks - 1) return "double-parity";
      if (diskIdx === totalDisks - 2) return "parity";
      return "data";
    },
  },
  {
    name: "RAID 10",
    description: "镜像 + 条带化：先镜像再条带，兼顾性能与冗余",
    minDisks: 4,
    capacity: "(N/2) × diskSize",
    faultTolerance: "每组可坏 1 块",
    readPerf: 5,
    writePerf: 4,
    layout: (diskIdx, blockIdx, totalDisks) => {
      if (blockIdx >= 1) return "empty";
      const pair = diskIdx % 2 === 0 ? "data" : "mirror";
      return diskIdx < totalDisks ? pair : "empty";
    },
  },
];

const BLOCK_COLORS = {
  data: { bg: "bg-blue-500", label: "D", border: "border-blue-600" },
  parity: { bg: "bg-yellow-500", label: "P", border: "border-yellow-600" },
  "double-parity": { bg: "bg-orange-500", label: "Q", border: "border-orange-600" },
  mirror: { bg: "bg-green-500", label: "M", border: "border-green-600" },
  empty: { bg: "bg-gray-200 dark:bg-gray-700", label: "", border: "border-gray-300 dark:border-gray-600" },
};

export function RAIDLevelComparison() {
  const [selected, setSelected] = useState<RAIDLevel>("RAID 5");
  const [diskCount, setDiskCount] = useState(5);

  const config = RAID_CONFIGS.find(r => r.name === selected)!;
  const effectiveDisks = Math.max(diskCount, config.minDisks);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 mb-2">
        <Database className="w-6 h-6 text-purple-600 dark:text-purple-400" />
        <h3 className="text-lg font-bold text-text-primary">RAID 级别对比</h3>
      </div>

      <div className="flex flex-wrap gap-2">
        {RAID_CONFIGS.map(r => (
          <button
            key={r.name}
            onClick={() => { setSelected(r.name); if (diskCount < r.minDisks) setDiskCount(r.minDisks); }}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              selected === r.name
                ? "bg-purple-600 text-white"
                : "bg-gray-100 dark:bg-gray-800 text-text-secondary hover:bg-gray-200 dark:hover:bg-gray-700"
            }`}
          >
            {r.name}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={selected}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="space-y-4"
        >
          <p className="text-sm text-text-secondary">{config.description}</p>

          <div className="flex items-center gap-4">
            <label className="text-sm font-medium text-text-secondary">磁盘数量</label>
            <input
              type="range"
              min={config.minDisks}
              max={8}
              value={diskCount}
              onChange={e => setDiskCount(parseInt(e.target.value))}
              className="flex-1"
            />
            <span className="font-mono text-sm text-text-primary w-8">{diskCount}</span>
          </div>

          <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg overflow-x-auto">
            <div className="inline-flex gap-1 min-w-full">
              {Array.from({ length: effectiveDisks }, (_, di) => (
                <div key={di} className="flex flex-col items-center gap-1 min-w-[56px]">
                  <div className="text-xs text-text-secondary mb-1">Disk {di}</div>
                  {Array.from({ length: 2 }, (_, bi) => {
                    const type = config.layout(di, bi, effectiveDisks);
                    const c = BLOCK_COLORS[type];
                    return (
                      <motion.div
                        key={bi}
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ delay: di * 0.05 + bi * 0.1 }}
                        className={`w-12 h-10 rounded flex items-center justify-center text-white text-xs font-bold ${c.bg} ${c.border} border`}
                      >
                        {c.label}{bi}
                      </motion.div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>

          <div className="flex gap-2 flex-wrap">
            {Object.entries(BLOCK_COLORS).filter(([k]) => k !== "empty").map(([key, val]) => (
              <div key={key} className="flex items-center gap-1.5 text-xs">
                <div className={`w-3 h-3 rounded ${val.bg}`} />
                <span className="text-text-secondary">
                  {key === "data" ? "数据块" : key === "parity" ? "校验块" : key === "double-parity" ? "双重校验" : "镜像块"}
                </span>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <div className="flex items-center gap-2 mb-2">
                <Database className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                <span className="text-sm font-semibold text-blue-700 dark:text-blue-300">有效容量</span>
              </div>
              <div className="font-mono text-lg font-bold text-text-primary">{config.capacity}</div>
              <div className="text-xs text-text-secondary mt-1">最低 {config.minDisks} 块磁盘</div>
            </div>

            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
              <div className="flex items-center gap-2 mb-2">
                <Shield className="w-4 h-4 text-green-600 dark:text-green-400" />
                <span className="text-sm font-semibold text-green-700 dark:text-green-300">容错能力</span>
              </div>
              <div className="font-mono text-lg font-bold text-text-primary">{config.faultTolerance}</div>
            </div>

            <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-800">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-4 h-4 text-orange-600 dark:text-orange-400" />
                <span className="text-sm font-semibold text-orange-700 dark:text-orange-300">性能评级</span>
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-text-secondary w-8">读</span>
                  <div className="flex gap-0.5">
                    {Array.from({ length: 5 }, (_, i) => (
                      <div key={i} className={`w-4 h-2 rounded-sm ${i < config.readPerf ? "bg-green-500" : "bg-gray-200 dark:bg-gray-700"}`} />
                    ))}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-text-secondary w-8">写</span>
                  <div className="flex gap-0.5">
                    {Array.from({ length: 5 }, (_, i) => (
                      <div key={i} className={`w-4 h-2 rounded-sm ${i < config.writePerf ? "bg-blue-500" : "bg-gray-200 dark:bg-gray-700"}`} />
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <h4 className="font-semibold text-text-primary text-sm mb-3">所有 RAID 级别速览</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left py-2 text-text-secondary font-medium">级别</th>
                <th className="text-left py-2 text-text-secondary font-medium">容量</th>
                <th className="text-left py-2 text-text-secondary font-medium">容错</th>
                <th className="text-center py-2 text-text-secondary font-medium">读</th>
                <th className="text-center py-2 text-text-secondary font-medium">写</th>
              </tr>
            </thead>
            <tbody>
              {RAID_CONFIGS.map(r => (
                <tr
                  key={r.name}
                  onClick={() => setSelected(r.name)}
                  className={`cursor-pointer border-b border-gray-100 dark:border-gray-800 hover:bg-gray-100 dark:hover:bg-gray-800 ${
                    selected === r.name ? "bg-purple-50 dark:bg-purple-900/20" : ""
                  }`}
                >
                  <td className="py-2 font-mono font-medium text-text-primary">{r.name}</td>
                  <td className="py-2 font-mono text-text-secondary">{r.capacity}</td>
                  <td className="py-2 text-text-secondary">{r.faultTolerance}</td>
                  <td className="py-2 text-center">
                    <div className="flex justify-center gap-0.5">
                      {Array.from({ length: 5 }, (_, i) => (
                        <div key={i} className={`w-2.5 h-2.5 rounded-sm ${i < r.readPerf ? "bg-green-500" : "bg-gray-200 dark:bg-gray-700"}`} />
                      ))}
                    </div>
                  </td>
                  <td className="py-2 text-center">
                    <div className="flex justify-center gap-0.5">
                      {Array.from({ length: 5 }, (_, i) => (
                        <div key={i} className={`w-2.5 h-2.5 rounded-sm ${i < r.writePerf ? "bg-blue-500" : "bg-gray-200 dark:bg-gray-700"}`} />
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-text-secondary">
            <p><strong className="text-text-primary">RAID</strong>（冗余磁盘阵列）通过数据分布和校验实现不同级别的性能与可靠性权衡。RAID 0 最快但不安全，RAID 1 最安全但容量减半，RAID 5/6 是常见折中方案。</p>
          </div>
        </div>
      </div>
    </div>
  );
}
