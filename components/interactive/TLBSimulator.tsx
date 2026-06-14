"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle2, XCircle, RotateCcw, Play } from "lucide-react";

interface TLBEntry {
  vpn: string;
  pfn: string;
  valid: boolean;
  asid?: string;
}

interface AccessResult {
  vpn: string;
  hit: boolean;
  pfn?: string;
  time: number;
}

const initialTLB: TLBEntry[] = [
  { vpn: "0x00401", pfn: "0x00050", valid: true, asid: "1" },
  { vpn: "0x00402", pfn: "0x00051", valid: true, asid: "1" },
  { vpn: "0x00400", pfn: "0x00048", valid: true, asid: "1" },
  { vpn: "0x00501", pfn: "0x00080", valid: true, asid: "2" },
];

const sampleAccesses = [
  "0x00401",
  "0x00402",
  "0x00403", // miss
  "0x00401", // hit
  "0x00500", // miss
  "0x00402", // hit
];

export function TLBSimulator() {
  const [tlb, setTlb] = useState<TLBEntry[]>(initialTLB);
  const [currentAccess, setCurrentAccess] = useState<string>("");
  const [accessHistory, setAccessHistory] = useState<AccessResult[]>([]);
  const [isAnimating, setIsAnimating] = useState(false);
  const [autoPlay, setAutoPlay] = useState(false);
  const [accessIndex, setAccessIndex] = useState(0);

  const lookupTLB = (vpn: string): AccessResult => {
    const entry = tlb.find((e) => e.vpn === vpn && e.valid);
    if (entry) {
      return { vpn, hit: true, pfn: entry.pfn, time: 1 }; // 1 ns for hit
    } else {
      return { vpn, hit: false, time: 400 }; // 400 ns for miss (page table walk)
    }
  };

  const handleManualAccess = () => {
    if (!currentAccess || isAnimating) return;

    setIsAnimating(true);
    const result = lookupTLB(currentAccess);
    
    setTimeout(() => {
      setAccessHistory((prev) => [result, ...prev].slice(0, 10));
      setIsAnimating(false);
      setCurrentAccess("");
    }, 1000);
  };

  const handleAutoPlay = () => {
    if (autoPlay) {
      setAutoPlay(false);
      return;
    }

    setAutoPlay(true);
    setAccessIndex(0);
    setAccessHistory([]);
  };

  const handleReset = () => {
    setTlb(initialTLB);
    setAccessHistory([]);
    setCurrentAccess("");
    setAutoPlay(false);
    setAccessIndex(0);
  };

  // Auto-play logic
  useState(() => {
    if (autoPlay && accessIndex < sampleAccesses.length) {
      const timer = setTimeout(() => {
        const vpn = sampleAccesses[accessIndex];
        const result = lookupTLB(vpn);
        setAccessHistory((prev) => [result, ...prev].slice(0, 10));
        setAccessIndex((prev) => prev + 1);
      }, 1500);

      return () => clearTimeout(timer);
    } else if (autoPlay && accessIndex >= sampleAccesses.length) {
      setAutoPlay(false);
    }
  });

  const hitRate =
    accessHistory.length > 0
      ? (accessHistory.filter((r) => r.hit).length / accessHistory.length) * 100
      : 0;

  const avgTime =
    accessHistory.length > 0
      ? accessHistory.reduce((sum, r) => sum + r.time, 0) / accessHistory.length
      : 0;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-6 text-text-primary">
        TLB 模拟器
      </h3>

      {/* TLB Table */}
      <div className="mb-6">
        <h4 className="font-semibold mb-3 text-text-primary">
          TLB 内容 ({tlb.length} 个条目)
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="bg-gray-100 dark:bg-gray-800">
                <th className="px-4 py-2 text-left text-text-primary border border-border-subtle">
                  索引
                </th>
                <th className="px-4 py-2 text-left text-text-primary border border-border-subtle">
                  VPN (虚拟页号)
                </th>
                <th className="px-4 py-2 text-left text-text-primary border border-border-subtle">
                  PFN (物理页框号)
                </th>
                <th className="px-4 py-2 text-left text-text-primary border border-border-subtle">
                  ASID
                </th>
                <th className="px-4 py-2 text-left text-text-primary border border-border-subtle">
                  有效位
                </th>
              </tr>
            </thead>
            <tbody>
              {tlb.map((entry, index) => (
                <motion.tr
                  key={index}
                  className={`${
                    currentAccess === entry.vpn && isAnimating
                      ? "bg-green-100 dark:bg-green-900/30"
                      : "hover:bg-gray-50 dark:hover:bg-gray-800"
                  }`}
                  animate={{
                    backgroundColor:
                      currentAccess === entry.vpn && isAnimating
                        ? ["#f0fdf4", "#86efac", "#f0fdf4"]
                        : "transparent",
                  }}
                  transition={{ duration: 1 }}
                >
                  <td className="px-4 py-2 border border-border-subtle text-text-primary">
                    {index}
                  </td>
                  <td className="px-4 py-2 border border-border-subtle font-mono text-text-primary">
                    {entry.vpn}
                  </td>
                  <td className="px-4 py-2 border border-border-subtle font-mono text-text-primary">
                    {entry.pfn}
                  </td>
                  <td className="px-4 py-2 border border-border-subtle font-mono text-text-secondary">
                    {entry.asid || "-"}
                  </td>
                  <td className="px-4 py-2 border border-border-subtle">
                    {entry.valid ? (
                      <CheckCircle2 className="w-4 h-4 text-green-500" />
                    ) : (
                      <XCircle className="w-4 h-4 text-red-500" />
                    )}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Access Input */}
      <div className="mb-6">
        <h4 className="font-semibold mb-3 text-text-primary">
          模拟内存访问
        </h4>
        <div className="flex gap-3">
          <input
            type="text"
            value={currentAccess}
            onChange={(e) => setCurrentAccess(e.target.value)}
            placeholder="输入虚拟页号 (如 0x00401)"
            className="flex-1 px-4 py-2 border border-border-subtle rounded-lg bg-bg-primary text-text-primary"
            disabled={isAnimating || autoPlay}
          />
          <button
            onClick={handleManualAccess}
            disabled={!currentAccess || isAnimating || autoPlay}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            查找
          </button>
          <button
            onClick={handleAutoPlay}
            className={`px-6 py-2 rounded-lg transition flex items-center gap-2 ${
              autoPlay
                ? "bg-red-600 hover:bg-red-700 text-white"
                : "bg-green-600 hover:bg-green-700 text-white"
            }`}
          >
            <Play className="w-4 h-4" />
            {autoPlay ? "停止" : "自动演示"}
          </button>
          <button
            onClick={handleReset}
            className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition flex items-center gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            重置
          </button>
        </div>
        <p className="mt-2 text-sm text-text-secondary">
          提示：尝试访问 0x00401 (命中) 或 0x00500 (未命中)
        </p>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {accessHistory.length}
          </div>
          <div className="text-sm text-text-secondary mt-1">总访问次数</div>
        </div>
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">
            {hitRate.toFixed(1)}%
          </div>
          <div className="text-sm text-text-secondary mt-1">TLB 命中率</div>
        </div>
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
            {avgTime.toFixed(1)} ns
          </div>
          <div className="text-sm text-text-secondary mt-1">平均访问时间</div>
        </div>
      </div>

      {/* Access History */}
      <div>
        <h4 className="font-semibold mb-3 text-text-primary">访问历史</h4>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          <AnimatePresence>
            {accessHistory.map((result, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className={`p-3 rounded-lg border flex items-center justify-between ${
                  result.hit
                    ? "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800"
                    : "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800"
                }`}
              >
                <div className="flex items-center gap-3">
                  {result.hit ? (
                    <CheckCircle2 className="w-5 h-5 text-green-600 dark:text-green-400" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
                  )}
                  <div>
                    <div className="font-mono text-sm text-text-primary">
                      VPN: {result.vpn}
                    </div>
                    {result.hit && result.pfn && (
                      <div className="font-mono text-xs text-text-secondary">
                        → PFN: {result.pfn}
                      </div>
                    )}
                  </div>
                </div>
                <div className="text-right">
                  <div
                    className={`text-sm font-semibold ${
                      result.hit
                        ? "text-green-600 dark:text-green-400"
                        : "text-red-600 dark:text-red-400"
                    }`}
                  >
                    {result.hit ? "命中" : "未命中"}
                  </div>
                  <div className="text-xs text-text-secondary">
                    {result.time} ns
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          {accessHistory.length === 0 && (
            <div className="text-center py-8 text-text-secondary">
              暂无访问记录，开始模拟查看效果
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
