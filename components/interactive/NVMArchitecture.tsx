"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { HardDrive, Cpu, Layers, Zap, Database, Clock } from "lucide-react";

interface MemoryLayer {
  name: string;
  type: "register" | "cache" | "dram" | "nvm" | "ssd";
  latency: string;
  latencyNs: number;
  bandwidth: string;
  capacity: string;
  description: string;
  color: string;
  icon: React.ReactNode;
}

const memoryLayers: MemoryLayer[] = [
  {
    name: "CPU 寄存器",
    type: "register",
    latency: "< 1 ns",
    latencyNs: 0.5,
    bandwidth: "~8 TB/s",
    capacity: "~1 KB",
    description: "最快存储，直接由 CPU 访问，存放当前操作数",
    color: "#dc2626",
    icon: <Cpu className="w-5 h-5" />,
  },
  {
    name: "L1/L2/L3 缓存",
    type: "cache",
    latency: "1-10 ns",
    latencyNs: 5,
    bandwidth: "~2 TB/s",
    capacity: "~64 MB",
    description: "SRAM 缓存，减少主存访问，分三级层次结构",
    color: "#f59e0b",
    icon: <Layers className="w-5 h-5" />,
  },
  {
    name: "主存 (DRAM)",
    type: "dram",
    latency: "~100 ns",
    latencyNs: 100,
    bandwidth: "~50 GB/s",
    capacity: "16-128 GB",
    description: "易失性主存，断电数据丢失，需要刷新周期",
    color: "#3b82f6",
    icon: <Database className="w-5 h-5" />,
  },
  {
    name: "NVM 持久内存",
    type: "nvm",
    latency: "~300 ns",
    latencyNs: 300,
    bandwidth: "~40 GB/s",
    capacity: "128-512 GB",
    description: "非易失性内存，字节寻址，断电数据保留，如 Intel Optane PMem",
    color: "#10b981",
    icon: <Zap className="w-5 h-5" />,
  },
  {
    name: "SSD / NVMe",
    type: "ssd",
    latency: "~100 μs",
    latencyNs: 100000,
    bandwidth: "~7 GB/s",
    capacity: "1-8 TB",
    description: "块设备，需通过文件系统访问，延迟高但容量大",
    color: "#8b5cf6",
    icon: <HardDrive className="w-5 h-5" />,
  },
];

const latencyComparisons = [
  { name: "寄存器", ns: 0.5, color: "#dc2626" },
  { name: "L1 缓存", ns: 1, color: "#ef4444" },
  { name: "L2 缓存", ns: 5, color: "#f59e0b" },
  { name: "DRAM", ns: 100, color: "#3b82f6" },
  { name: "NVM", ns: 300, color: "#10b981" },
  { name: "SSD", ns: 100000, color: "#8b5cf6" },
];

export default function NVMArchitecture() {
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);
  const [showBarChart, setShowBarChart] = useState(false);

  const selected = memoryLayers.find((l) => l.name === selectedLayer);
  const maxLogNs = Math.log10(100000);

  return (
    <div className="my-8 border border-gray-200 dark:border-gray-700 rounded-lg p-6 bg-white dark:bg-gray-900">
      <h3 className="text-xl font-semibold mb-6 text-gray-900 dark:text-gray-100">
        NVM 持久内存架构
      </h3>

      {/* Memory Hierarchy Pyramid */}
      <div className="mb-6 p-6 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h4 className="font-semibold mb-4 text-gray-900 dark:text-gray-100">
          存储层次结构 (点击查看详情)
        </h4>
        <div className="flex flex-col items-center gap-2">
          {memoryLayers.map((layer, i) => {
            const widthPercent = 30 + i * 15;
            const isSelected = selectedLayer === layer.name;
            return (
              <motion.div
                key={layer.name}
                onClick={() =>
                  setSelectedLayer(isSelected ? null : layer.name)
                }
                className="cursor-pointer rounded-lg border-2 flex items-center gap-3 px-4 py-3 transition"
                style={{
                  width: `${widthPercent}%`,
                  backgroundColor: `${layer.color}15`,
                  borderColor: isSelected ? layer.color : `${layer.color}40`,
                }}
                whileHover={{ scale: 1.03 }}
                animate={isSelected ? { scale: 1.05 } : { scale: 1 }}
              >
                <span style={{ color: layer.color }}>{layer.icon}</span>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 truncate">
                    {layer.name}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {layer.latency} · {layer.capacity}
                  </div>
                </div>
                <div
                  className="text-xs font-mono px-2 py-0.5 rounded"
                  style={{
                    backgroundColor: `${layer.color}20`,
                    color: layer.color,
                  }}
                >
                  {layer.bandwidth}
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Selected Layer Details */}
      {selected && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 p-5 rounded-lg border-2"
          style={{
            backgroundColor: `${selected.color}10`,
            borderColor: `${selected.color}60`,
          }}
        >
          <div className="flex items-center gap-3 mb-3">
            <span style={{ color: selected.color }}>{selected.icon}</span>
            <h4
              className="text-lg font-bold"
              style={{ color: selected.color }}
            >
              {selected.name}
            </h4>
          </div>
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
            {selected.description}
          </p>
          <div className="grid grid-cols-3 gap-4">
            {[
              { label: "延迟", value: selected.latency, icon: <Clock className="w-4 h-4" /> },
              { label: "带宽", value: selected.bandwidth, icon: <Zap className="w-4 h-4" /> },
              { label: "容量", value: selected.capacity, icon: <Database className="w-4 h-4" /> },
            ].map((stat) => (
              <div
                key={stat.label}
                className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 text-center"
              >
                <div className="flex items-center justify-center gap-1 text-gray-500 dark:text-gray-400 mb-1">
                  {stat.icon}
                  <span className="text-xs">{stat.label}</span>
                </div>
                <div className="text-sm font-bold text-gray-900 dark:text-gray-100">
                  {stat.value}
                </div>
              </div>
            ))}
          </div>
          {selected.type === "nvm" && (
            <div className="mt-4 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
              <p className="text-xs text-green-800 dark:text-green-300">
                <strong>NVM 优势：</strong>
                字节寻址（无需块 I/O）、持久性（断电保留）、接近 DRAM 延迟、
                可作为内存扩展或快速存储层。典型产品：Intel Optane PMem、Samsung CXL-NVM。
              </p>
            </div>
          )}
        </motion.div>
      )}

      {/* Latency Bar Chart Toggle */}
      <div className="mb-6">
        <button
          onClick={() => setShowBarChart(!showBarChart)}
          className="mb-4 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition text-sm"
        >
          {showBarChart ? "隐藏延迟对比" : "显示延迟对比柱状图"}
        </button>

        {showBarChart && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="p-5 bg-gray-50 dark:bg-gray-800 rounded-lg"
          >
            <h4 className="font-semibold mb-4 text-gray-900 dark:text-gray-100">
              延迟对比 (对数刻度)
            </h4>
            <div className="space-y-3">
              {latencyComparisons.map((item) => {
                const logWidth = (Math.log10(item.ns) / maxLogNs) * 100;
                return (
                  <div key={item.name} className="flex items-center gap-3">
                    <div className="w-20 text-xs text-right text-gray-600 dark:text-gray-400 font-medium shrink-0">
                      {item.name}
                    </div>
                    <div className="flex-1 h-7 bg-gray-200 dark:bg-gray-700 rounded overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.max(logWidth, 2)}%` }}
                        transition={{ duration: 0.8, ease: "easeOut" }}
                        className="h-full rounded flex items-center justify-end pr-2"
                        style={{ backgroundColor: item.color }}
                      >
                        <span className="text-xs text-white font-mono font-bold whitespace-nowrap">
                          {item.ns >= 1000
                            ? `${(item.ns / 1000).toFixed(0)}μs`
                            : `${item.ns}ns`}
                        </span>
                      </motion.div>
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="mt-4 text-xs text-gray-500 dark:text-gray-400">
              注：采用对数刻度，DRAM→NVM 约 3x 差距，NVM→SSD 约 300x 差距
            </p>
          </motion.div>
        )}
      </div>

      {/* NVM Programming Model */}
      <div className="p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg border border-emerald-200 dark:border-emerald-800">
        <h4 className="font-semibold mb-2 text-emerald-800 dark:text-emerald-300">
          NVM 编程模型
        </h4>
        <div className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
          <p>
            <strong className="text-gray-900 dark:text-gray-100">
              作为内存扩展 (Memory Mode)：
            </strong>
            NVM 作为大容量易失性主存，DRAM 作为缓存，应用无需修改代码
          </p>
          <p>
            <strong className="text-gray-900 dark:text-gray-100">
              作为持久存储 (App Direct Mode)：
            </strong>
            NVM 字节寻址持久存储，需要使用 PMDK / libpmem 管理持久化语义和崩溃一致性
          </p>
          <p>
            <strong className="text-gray-900 dark:text-gray-100">
              关键挑战：
            </strong>
            缓存行刷新 (CLFLUSH/CLWB)、内存屏障 (SFENCE)、崩溃一致性 (undo/redo logging)
          </p>
        </div>
      </div>
    </div>
  );
}
