"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { AlertCircle, Zap, Database, HardDrive } from "lucide-react";

interface MemoryLevel {
  name: string;
  capacity: string;
  latency: string;
  latencyNs: number;
  color: string;
  icon: React.ReactNode;
}

const memoryLevels: MemoryLevel[] = [
  {
    name: "寄存器",
    capacity: "< 1 KB",
    latency: "~0.3 ns",
    latencyNs: 0.3,
    color: "#8b5cf6",
    icon: <Zap className="w-5 h-5" />,
  },
  {
    name: "L1 缓存",
    capacity: "64 KB",
    latency: "~1 ns",
    latencyNs: 1,
    color: "#3b82f6",
    icon: <Zap className="w-5 h-5" />,
  },
  {
    name: "L2 缓存",
    capacity: "512 KB",
    latency: "~5 ns",
    latencyNs: 5,
    color: "#10b981",
    icon: <Database className="w-5 h-5" />,
  },
  {
    name: "L3 缓存",
    capacity: "16 MB",
    latency: "~20 ns",
    latencyNs: 20,
    color: "#f59e0b",
    icon: <Database className="w-5 h-5" />,
  },
  {
    name: "DRAM (主存)",
    capacity: "16 GB",
    latency: "~100 ns",
    latencyNs: 100,
    color: "#ef4444",
    icon: <Database className="w-5 h-5" />,
  },
  {
    name: "SSD",
    capacity: "512 GB",
    latency: "~100 μs",
    latencyNs: 100000,
    color: "#ec4899",
    icon: <HardDrive className="w-5 h-5" />,
  },
  {
    name: "HDD",
    capacity: "2 TB",
    latency: "~10 ms",
    latencyNs: 10000000,
    color: "#64748b",
    icon: <HardDrive className="w-5 h-5" />,
  },
];

export function MemoryHierarchyVisualizer() {
  const [selectedLevel, setSelectedLevel] = useState<number | null>(null);
  const [showComparison, setShowComparison] = useState(false);
  const [simulateAccess, setSimulateAccess] = useState<number | null>(null);

  const maxLatency = Math.max(...memoryLevels.map((l) => l.latencyNs));

  const handleSimulateAccess = (index: number) => {
    setSimulateAccess(index);
    setTimeout(() => setSimulateAccess(null), 2000);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-text-primary">
          内存层次结构可视化
        </h3>
        <button
          onClick={() => setShowComparison(!showComparison)}
          className="px-4 py-2 bg-accent-primary text-white rounded-lg hover:bg-accent-primary/90 transition"
        >
          {showComparison ? "隐藏对比" : "性能对比"}
        </button>
      </div>

      {/* Pyramid Visualization */}
      <div className="mb-8">
        <div className="flex flex-col items-center gap-1">
          {memoryLevels.map((level, index) => {
            const isSelected = selectedLevel === index;
            const isSimulating = simulateAccess === index;
            const width = 100 - index * 12;

            return (
              <motion.div
                key={level.name}
                className="relative cursor-pointer"
                style={{ width: `${width}%` }}
                onClick={() => setSelectedLevel(isSelected ? null : index)}
                onDoubleClick={() => handleSimulateAccess(index)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <motion.div
                  className={`p-4 rounded-lg text-white transition-all ${
                    isSimulating ? "animate-pulse" : ""
                  }`}
                  style={{
                    backgroundColor: level.color,
                    boxShadow: isSelected
                      ? `0 0 20px ${level.color}80`
                      : "none",
                  }}
                  animate={{
                    scale: isSimulating ? [1, 1.1, 1] : 1,
                  }}
                  transition={{
                    duration: 0.5,
                    repeat: isSimulating ? Infinity : 0,
                  }}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {level.icon}
                      <span className="font-semibold">{level.name}</span>
                    </div>
                    <div className="flex gap-4 text-sm">
                      <span>{level.capacity}</span>
                      <span>{level.latency}</span>
                    </div>
                  </div>
                </motion.div>

                {/* Arrow pointing down */}
                {index < memoryLevels.length - 1 && (
                  <div className="flex justify-center my-1">
                    <svg width="24" height="16" className="text-text-secondary">
                      <path
                        d="M12 0 L12 12 M7 7 L12 12 L17 7"
                        stroke="currentColor"
                        strokeWidth="2"
                        fill="none"
                      />
                    </svg>
                  </div>
                )}
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Selected Level Details */}
      {selectedLevel !== null && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800"
        >
          <h4 className="font-semibold text-lg mb-2 text-text-primary">
            {memoryLevels[selectedLevel].name} - 详细信息
          </h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-text-secondary">容量：</span>
              <span className="font-semibold text-text-primary ml-2">
                {memoryLevels[selectedLevel].capacity}
              </span>
            </div>
            <div>
              <span className="text-text-secondary">访问延迟：</span>
              <span className="font-semibold text-text-primary ml-2">
                {memoryLevels[selectedLevel].latency}
              </span>
            </div>
            <div className="col-span-2">
              <span className="text-text-secondary">相对速度：</span>
              <span className="font-semibold text-text-primary ml-2">
                {selectedLevel === 0
                  ? "基准"
                  : `${Math.round(
                      memoryLevels[selectedLevel].latencyNs /
                        memoryLevels[0].latencyNs
                    )}x 慢于寄存器`}
              </span>
            </div>
          </div>
          <div className="mt-3 text-xs text-text-secondary flex items-start gap-2">
            <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>双击层级可模拟访问延迟动画</span>
          </div>
        </motion.div>
      )}

      {/* Performance Comparison */}
      {showComparison && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          className="mt-6"
        >
          <h4 className="font-semibold mb-4 text-text-primary">
            访问延迟对比 (对数刻度)
          </h4>
          <div className="space-y-3">
            {memoryLevels.map((level, index) => {
              const logWidth =
                (Math.log10(level.latencyNs) / Math.log10(maxLatency)) * 100;

              return (
                <div key={level.name} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-primary">{level.name}</span>
                    <span className="text-text-secondary">{level.latency}</span>
                  </div>
                  <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ backgroundColor: level.color }}
                      initial={{ width: 0 }}
                      animate={{ width: `${logWidth}%` }}
                      transition={{ duration: 1, delay: index * 0.1 }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
            <p className="text-sm text-text-secondary">
              <strong className="text-text-primary">性能差距：</strong>
              HDD 的访问延迟是寄存器的{" "}
              <strong className="text-accent-primary">
                {(memoryLevels[6].latencyNs / memoryLevels[0].latencyNs).toLocaleString()}
                倍
              </strong>
              ！这就是为什么需要多层缓存的原因。
            </p>
          </div>
        </motion.div>
      )}

      {/* Cache Hit/Miss Simulation */}
      <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <h4 className="font-semibold mb-3 text-text-primary">
          缓存命中/未命中模拟
        </h4>
        <div className="grid grid-cols-3 gap-3">
          <div className="text-center p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              95%
            </div>
            <div className="text-xs text-text-secondary mt-1">L1 命中率</div>
          </div>
          <div className="text-center p-3 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              ~8 ns
            </div>
            <div className="text-xs text-text-secondary mt-1">平均访问时间</div>
          </div>
          <div className="text-center p-3 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
            <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
              40x
            </div>
            <div className="text-xs text-text-secondary mt-1">性能提升</div>
          </div>
        </div>
      </div>
    </div>
  );
}
