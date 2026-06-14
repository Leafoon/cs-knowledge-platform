"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Server, Box, Clock, MemoryStick, Shield, Layers, Zap, HardDrive } from "lucide-react";

interface CompareItem {
  dimension: string;
  icon: React.ReactNode;
  vm: string;
  container: string;
  vmDetail: string;
  containerDetail: string;
  winner: "vm" | "container" | "tie";
}

const comparisons: CompareItem[] = [
  {
    dimension: "隔离级别",
    icon: <Shield className="w-4 h-4" />,
    vm: "硬件级",
    container: "操作系统级",
    vmDetail: "每个 VM 有独立内核和硬件抽象，通过 Hypervisor 隔离",
    containerDetail: "共享宿主内核，通过 Namespace + cgroup 隔离进程和资源",
    winner: "vm",
  },
  {
    dimension: "启动时间",
    icon: <Clock className="w-4 h-4" />,
    vm: "30s - 数分钟",
    container: "毫秒 - 数秒",
    vmDetail: "需要启动完整 OS 内核、init 系统、系统服务",
    containerDetail: "只需创建 Namespace 和 cgroup，直接 exec 入口进程",
    winner: "container",
  },
  {
    dimension: "内存开销",
    icon: <MemoryStick className="w-4 h-4" />,
    vm: "512MB - 数 GB",
    container: "数 MB - 数十 MB",
    vmDetail: "每个 VM 需要独立的内核内存、系统服务、缓冲区",
    containerDetail: "共享宿主内核，仅需应用和库的内存",
    winner: "container",
  },
  {
    dimension: "镜像大小",
    icon: <HardDrive className="w-4 h-4" />,
    vm: "数 GB",
    container: "数 MB - 数百 MB",
    vmDetail: "包含完整 OS 文件系统、内核、引导程序",
    containerDetail: "仅包含应用二进制和依赖库，分层共享基础镜像",
    winner: "container",
  },
  {
    dimension: "性能损耗",
    icon: <Zap className="w-4 h-4" />,
    vm: "5% - 15%",
    container: "≈ 0%",
    vmDetail: "硬件虚拟化 VM Entry/Exit 开销、EPT 翻译、设备模拟",
    containerDetail: "直接在宿主内核上运行，无虚拟化层开销",
    winner: "container",
  },
  {
    dimension: "运行密度",
    icon: <Layers className="w-4 h-4" />,
    vm: "10 - 50 / 物理机",
    container: "100 - 1000 / 物理机",
    vmDetail: "每个 VM 消耗大量内存和 CPU，受 Hypervisor 资源限制",
    containerDetail: "极低的资源开销，可在单机运行数百容器",
    winner: "container",
  },
];

export default function ContainerVsVM() {
  const [hoveredItem, setHoveredItem] = useState<number | null>(null);

  return (
    <div className="w-full space-y-5">
      <div className="grid grid-cols-2 gap-4">
        <motion.div
          initial={{ opacity: 0, x: -16 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-gray-800/60 rounded-xl p-5 border border-orange-500/30"
        >
          <div className="flex items-center gap-2 mb-4">
            <div className="p-2 rounded-lg bg-gradient-to-br from-orange-500 to-red-500 text-white">
              <Server className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white">虚拟机 (VM)</h3>
              <p className="text-[10px] text-gray-500">Hardware Virtualization</p>
            </div>
          </div>
          <div className="space-y-1.5">
            {["App A", "App B"].map((app) => (
              <div key={app} className="bg-blue-500/20 text-blue-400 text-[11px] text-center py-1 rounded border border-blue-500/30">
                {app}
              </div>
            ))}
            <div className="bg-blue-600/30 text-blue-300 text-[11px] text-center py-1.5 rounded border border-blue-500/30">
              Guest OS (完整内核)
            </div>
            <div className="bg-gray-600/30 text-gray-400 text-[11px] text-center py-1 rounded border border-gray-500/30">
              独立内核 + 系统库
            </div>
            <div className="bg-orange-500/20 text-orange-400 text-[11px] text-center py-1.5 rounded border border-orange-500/30 font-medium">
              Hypervisor
            </div>
            <div className="bg-gray-700/50 text-gray-500 text-[11px] text-center py-1.5 rounded border border-gray-600/30">
              硬件虚拟化 (VT-x/EPT)
            </div>
            <div className="bg-gray-700 text-gray-400 text-[11px] text-center py-1.5 rounded font-medium">
              物理硬件
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-gray-800/60 rounded-xl p-5 border border-cyan-500/30"
        >
          <div className="flex items-center gap-2 mb-4">
            <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-500 text-white">
              <Box className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white">容器 (Container)</h3>
              <p className="text-[10px] text-gray-500">OS-Level Virtualization</p>
            </div>
          </div>
          <div className="flex gap-2 mb-1.5">
            {["App A", "App B"].map((app) => (
              <div key={app} className="flex-1 bg-cyan-500/20 text-cyan-400 text-[11px] text-center py-1 rounded border border-cyan-500/30">
                {app}
              </div>
            ))}
          </div>
          <div className="flex gap-2 mb-1.5">
            {["Lib A", "Lib B"].map((lib) => (
              <div key={lib} className="flex-1 bg-cyan-600/20 text-cyan-300 text-[11px] text-center py-1 rounded border border-cyan-500/20">
                {lib}
              </div>
            ))}
          </div>
          <div className="flex gap-2 mb-1.5">
            <div className="flex-1 bg-purple-500/20 text-purple-400 text-[11px] text-center py-1 rounded border border-purple-500/30">
              NS 隔离
            </div>
            <div className="flex-1 bg-purple-500/20 text-purple-400 text-[11px] text-center py-1 rounded border border-purple-500/30">
              NS 隔离
            </div>
          </div>
          <div className="bg-cyan-500/10 text-cyan-400 text-[11px] text-center py-1.5 rounded border border-cyan-500/20 mb-1.5 font-medium">
            宿主内核 (共享) + cgroup 资源限制
          </div>
          <div className="bg-gray-700 text-gray-400 text-[11px] text-center py-1.5 rounded font-medium">
            物理硬件
          </div>
        </motion.div>
      </div>

      <div className="bg-gray-800/60 rounded-xl border border-gray-700 overflow-hidden">
        <div className="grid grid-cols-[1fr_1fr_1fr] text-xs font-semibold text-gray-300 border-b border-gray-700">
          <div className="px-4 py-2.5 bg-gray-800/80">维度</div>
          <div className="px-4 py-2.5 bg-orange-500/10 text-orange-400 text-center">虚拟机</div>
          <div className="px-4 py-2.5 bg-cyan-500/10 text-cyan-400 text-center">容器</div>
        </div>
        {comparisons.map((item, i) => (
          <motion.div
            key={item.dimension}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.06 }}
            onMouseEnter={() => setHoveredItem(i)}
            onMouseLeave={() => setHoveredItem(null)}
            className={`grid grid-cols-[1fr_1fr_1fr] text-xs border-b border-gray-700/50 cursor-pointer transition-colors ${
              hoveredItem === i ? "bg-gray-700/30" : ""
            }`}
          >
            <div className="px-4 py-3 flex items-center gap-2 text-gray-300">
              <span className="text-gray-500">{item.icon}</span>
              {item.dimension}
            </div>
            <div className={`px-4 py-3 text-center ${item.winner === "vm" ? "text-green-400 font-semibold" : "text-gray-400"}`}>
              {item.vm}
              {item.winner === "vm" && <span className="ml-1 text-[10px]">✓</span>}
            </div>
            <div className={`px-4 py-3 text-center ${item.winner === "container" ? "text-green-400 font-semibold" : "text-gray-400"}`}>
              {item.container}
              {item.winner === "container" && <span className="ml-1 text-[10px]">✓</span>}
            </div>
          </motion.div>
        ))}
      </div>

      <AnimatePresence>
        {hoveredItem !== null && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            className="bg-gray-800/80 rounded-xl p-4 border border-gray-600 grid grid-cols-2 gap-4"
          >
            <div>
              <p className="text-[10px] text-orange-400 font-semibold mb-1 flex items-center gap-1">
                <Server className="w-3 h-3" /> 虚拟机
              </p>
              <p className="text-xs text-gray-400">{comparisons[hoveredItem].vmDetail}</p>
            </div>
            <div>
              <p className="text-[10px] text-cyan-400 font-semibold mb-1 flex items-center gap-1">
                <Box className="w-3 h-3" /> 容器
              </p>
              <p className="text-xs text-gray-400">{comparisons[hoveredItem].containerDetail}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
