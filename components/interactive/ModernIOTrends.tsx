"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { TrendingUp, Cpu, Zap, Globe } from "lucide-react";

const technologies = [
  {
    name: "NVMe over Fabrics",
    icon: Globe,
    color: "blue",
    layers: ["NVMe 命令集", "NVMe-oF 传输层", "RDMA / FC / TCP"],
    desc: "将NVMe协议扩展到网络传输，实现远程闪存访问",
    features: [
      "端到端延迟 < 10μs",
      "支持 RDMA (RoCE/iWARP)、光纤通道、TCP",
      "与本地NVMe相同的命令接口",
      "支持共享命名空间",
    ],
    metrics: { latency: "10μs", bandwidth: "100 Gbps", iops: "10M+" },
  },
  {
    name: "RDMA",
    icon: Zap,
    color: "green",
    layers: ["应用程序", "Verbs API", "RDMA 传输", "InfiniBand / RoCE"],
    desc: "远程直接内存访问，绕过CPU和操作系统内核实现零拷贝传输",
    features: [
      "内核旁路 (Kernel Bypass)",
      "零拷贝 (Zero Copy)",
      "CPU卸载 (Offload)",
      "微秒级延迟",
    ],
    metrics: { latency: "1-3μs", bandwidth: "200 Gbps", iops: "N/A" },
  },
  {
    name: "CXL (Compute Express Link)",
    icon: Cpu,
    color: "purple",
    layers: ["CXL.io (I/O)", "CXL.cache (缓存)", "CXL.mem (内存)", "PCIe 物理层"],
    desc: "基于PCIe的新一代互连标准，实现内存池化和缓存一致性",
    features: [
      "内存扩展和池化",
      "缓存一致性 (Cache Coherence)",
      "设备内存 (Device Memory)",
      "内存分层 (Memory Tiering)",
    ],
    metrics: { latency: "< 200ns", bandwidth: "64 GT/s", iops: "N/A" },
  },
];

export function ModernIOTrends() {
  const [selected, setSelected] = useState<number | null>(null);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <TrendingUp className="w-5 h-5 text-cyan-400" />
        <h3 className="text-lg font-semibold">现代 I/O 技术趋势</h3>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4">
        {technologies.map((tech, i) => {
          const Icon = tech.icon;
          return (
            <motion.button key={tech.name}
              onClick={() => setSelected(selected === i ? null : i)}
              className={`p-4 rounded-lg border-2 text-left transition-all ${
                selected === i
                  ? `border-${tech.color}-400 bg-${tech.color}-500/10`
                  : "border-gray-600 bg-gray-800/30 hover:border-gray-500"
              }`}
              whileHover={{ scale: 1.02 }}
            >
              <Icon className={`w-6 h-6 text-${tech.color}-400 mb-2`} />
              <div className="text-sm font-medium text-gray-200">{tech.name}</div>
              <div className="text-xs text-gray-400 mt-1 line-clamp-2">{tech.desc}</div>
            </motion.button>
          );
        })}
      </div>

      <AnimatePresence mode="wait">
        {selected !== null && (
          <motion.div key={selected}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <div className="grid grid-cols-3 gap-3 mb-4">
              {Object.entries(technologies[selected].metrics).map(([k, v]) => (
                <div key={k} className="p-2 bg-gray-800/30 rounded text-center">
                  <div className="text-xs text-gray-400">{k}</div>
                  <div className={`text-sm font-bold text-${technologies[selected].color}-300`}>{v}</div>
                </div>
              ))}
            </div>

            <div className="mb-4">
              <div className="text-xs text-gray-400 mb-2">协议栈:</div>
              <div className="flex flex-col gap-1">
                {technologies[selected].layers.map((l, i) => (
                  <motion.div key={l}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                    className={`p-2 rounded text-xs text-${technologies[selected].color}-300 bg-gray-800/50`}
                    style={{ marginLeft: i * 16 }}
                  >
                    {l}
                  </motion.div>
                ))}
              </div>
            </div>

            <div className="text-xs text-gray-400 mb-2">核心特性:</div>
            <div className="grid grid-cols-2 gap-2">
              {technologies[selected].features.map((f, i) => (
                <motion.div key={f}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.08 }}
                  className="flex items-center gap-2 p-2 bg-gray-800/30 rounded text-xs text-gray-300"
                >
                  <span className={`w-1.5 h-1.5 rounded-full bg-${technologies[selected].color}-400`} />
                  {f}
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {selected === null && (
        <div className="text-xs text-gray-500 text-center py-4">
          点击上方卡片查看技术详情
        </div>
      )}
    </div>
  );
}
