"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Layers, Database, Network, HardDrive } from "lucide-react";

interface CacheLevel {
  name: string;
  size: number;
  blockSize: number;
  associativity: "direct" | "2-way" | "4-way" | "8-way" | "16-way" | "fully";
  latency: number;
}

interface CacheLine {
  valid: boolean;
  tag: number;
  data: string;
  dirty: boolean;
  lru: number;
}

const cacheConfigs: { [key: string]: CacheLevel } = {
  L1: {
    name: "L1 缓存",
    size: 64,
    blockSize: 64,
    associativity: "8-way",
    latency: 4,
  },
  L2: {
    name: "L2 缓存",
    size: 512,
    blockSize: 64,
    associativity: "8-way",
    latency: 12,
  },
  L3: {
    name: "L3 缓存",
    size: 16384,
    blockSize: 64,
    associativity: "16-way",
    latency: 40,
  },
};

export function CacheSimulator() {
  const [selectedCache, setSelectedCache] = useState<"L1" | "L2" | "L3">("L1");
  const [address, setAddress] = useState("0x00400000");
  const [cache, setCache] = useState<CacheLine[]>(
    Array(8).fill(null).map(() => ({
      valid: false,
      tag: 0,
      data: "",
      dirty: false,
      lru: 0,
    }))
  );
  const [accessHistory, setAccessHistory] = useState<
    { address: string; hit: boolean; latency: number }[]
  >([]);
  const [stats, setStats] = useState({ hits: 0, misses: 0 });

  const parseAddress = (addr: string): { tag: number; index: number; offset: number } => {
    const numericAddr = parseInt(addr, 16);
    const blockSize = cacheConfigs[selectedCache].blockSize;
    const numSets = 8; // 假设 8 个 set

    const offsetBits = Math.log2(blockSize);
    const indexBits = Math.log2(numSets);

    const offset = numericAddr & ((1 << offsetBits) - 1);
    const index = (numericAddr >> offsetBits) & ((1 << indexBits) - 1);
    const tag = numericAddr >> (offsetBits + indexBits);

    return { tag, index, offset };
  };

  const simulateAccess = () => {
    const { tag, index, offset } = parseAddress(address);
    const config = cacheConfigs[selectedCache];

    // 检查 cache hit/miss
    const line = cache[index];
    const isHit = line.valid && line.tag === tag;

    setAccessHistory((prev) =>
      [
        {
          address,
          hit: isHit,
          latency: isHit ? config.latency : config.latency + 200,
        },
        ...prev,
      ].slice(0, 10)
    );

    setStats((prev) => ({
      hits: prev.hits + (isHit ? 1 : 0),
      misses: prev.misses + (isHit ? 0 : 1),
    }));

    if (!isHit) {
      // Cache miss - 加载新数据
      const newCache = [...cache];
      newCache[index] = {
        valid: true,
        tag,
        data: `Data@${address}`,
        dirty: false,
        lru: Date.now(),
      };
      setCache(newCache);
    } else {
      // Cache hit - 更新 LRU
      const newCache = [...cache];
      newCache[index].lru = Date.now();
      setCache(newCache);
    }
  };

  const handleReset = () => {
    setCache(
      Array(8).fill(null).map(() => ({
        valid: false,
        tag: 0,
        data: "",
        dirty: false,
        lru: 0,
      }))
    );
    setAccessHistory([]);
    setStats({ hits: 0, misses: 0 });
  };

  const hitRate =
    stats.hits + stats.misses > 0
      ? (stats.hits / (stats.hits + stats.misses)) * 100
      : 0;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-6 text-text-primary">
        CPU 缓存模拟器
      </h3>

      {/* Cache Level Selection */}
      <div className="mb-6">
        <h4 className="font-semibold mb-3 text-text-primary">选择缓存级别</h4>
        <div className="grid grid-cols-3 gap-3">
          {(Object.keys(cacheConfigs) as Array<"L1" | "L2" | "L3">).map((level) => {
            const config = cacheConfigs[level];
            return (
              <button
                key={level}
                onClick={() => setSelectedCache(level)}
                className={`p-4 rounded-lg border-2 transition ${
                  selectedCache === level
                    ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                    : "border-gray-300 dark:border-gray-700 hover:border-gray-400"
                }`}
              >
                <div className="text-lg font-bold text-text-primary">
                  {config.name}
                </div>
                <div className="text-sm text-text-secondary mt-1">
                  {config.size} KB, {config.latency} cycles
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Cache Configuration */}
      <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <h4 className="font-semibold mb-3 text-text-primary">缓存配置</h4>
        <div className="grid grid-cols-4 gap-4">
          <div>
            <div className="text-sm text-text-secondary">容量</div>
            <div className="text-lg font-semibold text-text-primary">
              {cacheConfigs[selectedCache].size} KB
            </div>
          </div>
          <div>
            <div className="text-sm text-text-secondary">块大小</div>
            <div className="text-lg font-semibold text-text-primary">
              {cacheConfigs[selectedCache].blockSize} B
            </div>
          </div>
          <div>
            <div className="text-sm text-text-secondary">相联度</div>
            <div className="text-lg font-semibold text-text-primary">
              {cacheConfigs[selectedCache].associativity}
            </div>
          </div>
          <div>
            <div className="text-sm text-text-secondary">访问延迟</div>
            <div className="text-lg font-semibold text-text-primary">
              {cacheConfigs[selectedCache].latency} cycles
            </div>
          </div>
        </div>
      </div>

      {/* Address Input */}
      <div className="mb-6">
        <h4 className="font-semibold mb-3 text-text-primary">模拟内存访问</h4>
        <div className="flex gap-3">
          <input
            type="text"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            placeholder="输入内存地址 (如 0x00400000)"
            className="flex-1 px-4 py-2 border border-border-subtle rounded-lg bg-bg-primary text-text-primary font-mono"
          />
          <button
            onClick={simulateAccess}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
          >
            访问
          </button>
          <button
            onClick={handleReset}
            className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
          >
            重置
          </button>
        </div>
        <div className="mt-2 text-sm text-text-secondary font-mono">
          {address && (() => {
            const { tag, index, offset } = parseAddress(address);
            return (
              <span>
                解析: Tag=0x{tag.toString(16).toUpperCase()} | Index={index} |
                Offset={offset}
              </span>
            );
          })()}
        </div>
      </div>

      {/* Statistics */}
      <div className="mb-6 grid grid-cols-4 gap-4">
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">
            {stats.hits}
          </div>
          <div className="text-sm text-text-secondary">命中次数</div>
        </div>
        <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
          <div className="text-2xl font-bold text-red-600 dark:text-red-400">
            {stats.misses}
          </div>
          <div className="text-sm text-text-secondary">未命中次数</div>
        </div>
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {hitRate.toFixed(1)}%
          </div>
          <div className="text-sm text-text-secondary">命中率</div>
        </div>
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
            {stats.hits + stats.misses}
          </div>
          <div className="text-sm text-text-secondary">总访问</div>
        </div>
      </div>

      {/* Cache Contents */}
      <div className="mb-6">
        <h4 className="font-semibold mb-3 text-text-primary">
          缓存内容 (Set-Associative)
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="bg-gray-100 dark:bg-gray-800">
                <th className="px-3 py-2 text-left border border-border-subtle text-text-primary">
                  Set
                </th>
                <th className="px-3 py-2 text-left border border-border-subtle text-text-primary">
                  有效位
                </th>
                <th className="px-3 py-2 text-left border border-border-subtle text-text-primary">
                  Tag
                </th>
                <th className="px-3 py-2 text-left border border-border-subtle text-text-primary">
                  数据
                </th>
                <th className="px-3 py-2 text-left border border-border-subtle text-text-primary">
                  脏位
                </th>
              </tr>
            </thead>
            <tbody>
              {cache.map((line, index) => (
                <tr
                  key={index}
                  className={`${
                    line.valid
                      ? "bg-green-50 dark:bg-green-900/10"
                      : "bg-gray-50 dark:bg-gray-900"
                  } hover:bg-blue-50 dark:hover:bg-blue-900/10`}
                >
                  <td className="px-3 py-2 border border-border-subtle font-mono text-text-primary">
                    {index}
                  </td>
                  <td className="px-3 py-2 border border-border-subtle">
                    <span
                      className={`px-2 py-1 rounded text-xs font-semibold ${
                        line.valid
                          ? "bg-green-600 text-white"
                          : "bg-gray-400 text-white"
                      }`}
                    >
                      {line.valid ? "V" : "-"}
                    </span>
                  </td>
                  <td className="px-3 py-2 border border-border-subtle font-mono text-text-primary">
                    {line.valid ? `0x${line.tag.toString(16).toUpperCase()}` : "-"}
                  </td>
                  <td className="px-3 py-2 border border-border-subtle font-mono text-text-secondary">
                    {line.data || "-"}
                  </td>
                  <td className="px-3 py-2 border border-border-subtle">
                    {line.dirty && (
                      <span className="px-2 py-1 bg-yellow-600 text-white rounded text-xs font-semibold">
                        Dirty
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Access History */}
      <div>
        <h4 className="font-semibold mb-3 text-text-primary">访问历史</h4>
        <div className="space-y-2">
          {accessHistory.map((access, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className={`p-3 rounded-lg border flex items-center justify-between ${
                access.hit
                  ? "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800"
                  : "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800"
              }`}
            >
              <div>
                <span className="font-mono text-sm text-text-primary">
                  {access.address}
                </span>
              </div>
              <div className="flex items-center gap-4">
                <span
                  className={`px-3 py-1 rounded font-semibold text-sm ${
                    access.hit
                      ? "bg-green-600 text-white"
                      : "bg-red-600 text-white"
                  }`}
                >
                  {access.hit ? "命中" : "未命中"}
                </span>
                <span className="text-sm text-text-secondary">
                  {access.latency} cycles
                </span>
              </div>
            </motion.div>
          ))}
          {accessHistory.length === 0 && (
            <div className="text-center py-8 text-text-secondary">
              暂无访问记录
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
