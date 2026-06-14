"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Cpu, HardDrive, MonitorPlay, Wifi, Zap } from "lucide-react";

interface Component {
  id: string;
  name: string;
  icon: React.ReactNode;
  x: number;
  y: number;
}

interface Bus {
  id: string;
  name: string;
  speed: string;
  width: string;
  color: string;
}

const components: Component[] = [
  { id: "cpu", name: "CPU", icon: <Cpu className="w-8 h-8" />, x: 50, y: 20 },
  { id: "memory", name: "内存", icon: <HardDrive className="w-8 h-8" />, x: 50, y: 60 },
  { id: "gpu", name: "GPU", icon: <MonitorPlay className="w-8 h-8" />, x: 15, y: 85 },
  { id: "storage", name: "存储", icon: <HardDrive className="w-8 h-8" />, x: 50, y: 85 },
  { id: "network", name: "网卡", icon: <Wifi className="w-8 h-8" />, x: 85, y: 85 }
];

const buses: Bus[] = [
  { id: "system", name: "系统总线 (FSB)", speed: "100 MHz - 1.6 GHz", width: "64-bit", color: "text-red-600" },
  { id: "memory", name: "内存总线", speed: "DDR4: 2133-3200 MHz", width: "64-bit", color: "text-blue-600" },
  { id: "pcie", name: "PCIe 总线", speed: "Gen 4: 16 GT/s", width: "x1/x4/x8/x16", color: "text-green-600" },
  { id: "sata", name: "SATA 总线", speed: "6 Gbps", width: "Serial", color: "text-purple-600" }
];

export default function BusInterconnectDiagram() {
  const [selectedBus, setSelectedBus] = useState<string>("system");

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 rounded-xl shadow-2xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        计算机总线互连架构
      </h3>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* 总线拓扑图 */}
        <div className="lg:col-span-2 p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg">
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4">系统拓扑</h4>
          
          <svg viewBox="0 0 400 300" className="w-full h-96">
            {/* CPU */}
            <motion.g
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2 }}
            >
              <rect x="180" y="20" width="80" height="60" rx="8" fill="#3b82f6" />
              <text x="220" y="55" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">CPU</text>
            </motion.g>

            {/* 系统总线 */}
            <motion.line
              x1="220" y1="80" x2="220" y2="120"
              stroke="#ef4444" strokeWidth="4"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: selectedBus === "system" ? 1 : 0.5 }}
              transition={{ duration: 0.5 }}
            />
            <text x="230" y="100" fill="#ef4444" fontSize="12">系统总线</text>

            {/* 内存控制器 */}
            <rect x="180" y="120" width="80" height="40" rx="8" fill="#10b981" />
            <text x="220" y="145" textAnchor="middle" fill="white" fontSize="12">内存控制器</text>

            {/* 内存总线 */}
            <motion.line
              x1="220" y1="160" x2="220" y2="190"
              stroke="#3b82f6" strokeWidth="4"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: selectedBus === "memory" ? 1 : 0.5 }}
              transition={{ duration: 0.5 }}
            />
            <text x="230" y="175" fill="#3b82f6" fontSize="12">内存总线</text>

            {/* 内存 */}
            <motion.g
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.4 }}
            >
              <rect x="180" y="190" width="80" height="40" rx="8" fill="#f59e0b" />
              <text x="220" y="215" textAnchor="middle" fill="white" fontSize="14">RAM</text>
            </motion.g>

            {/* PCIe 总线到各设备 */}
            <motion.g>
              {/* 到 GPU */}
              <motion.path
                d="M 160 140 L 80 240"
                stroke="#10b981" strokeWidth="3" fill="none"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: selectedBus === "pcie" ? 1 : 0.5 }}
              />
              <rect x="40" y="240" width="80" height="40" rx="8" fill="#8b5cf6" />
              <text x="80" y="265" textAnchor="middle" fill="white" fontSize="12">GPU</text>

              {/* 到存储 */}
              <motion.path
                d="M 220 160 L 220 240"
                stroke="#8b5cf6" strokeWidth="3" fill="none"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: selectedBus === "sata" ? 1 : 0.5 }}
              />
              <rect x="180" y="240" width="80" height="40" rx="8" fill="#f59e0b" />
              <text x="220" y="265" textAnchor="middle" fill="white" fontSize="12">SSD/HDD</text>

              {/* 到网卡 */}
              <motion.path
                d="M 280 140 L 320 240"
                stroke="#10b981" strokeWidth="3" fill="none"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: selectedBus === "pcie" ? 1 : 0.5 }}
              />
              <rect x="280" y="240" width="80" height="40" rx="8" fill="#06b6d4" />
              <text x="320" y="265" textAnchor="middle" fill="white" fontSize="12">网卡</text>
            </motion.g>
          </svg>
        </div>

        {/* 总线规格 */}
        <div className="space-y-3">
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-3">总线规格</h4>
          {buses.map((bus) => (
            <motion.button
              key={bus.id}
              whileHover={{ scale: 1.02 }}
              onClick={() => setSelectedBus(bus.id)}
              className={`
                w-full text-left p-4 rounded-lg transition-all
                ${selectedBus === bus.id
                  ? "bg-blue-600 text-white shadow-lg"
                  : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                }
              `}
            >
              <div className="flex items-center gap-2 mb-1">
                <Zap className="w-5 h-5" />
                <span className="font-semibold">{bus.name}</span>
              </div>
              <div className={`text-xs mt-2 space-y-1 ${selectedBus === bus.id ? "text-blue-100" : "text-slate-500"}`}>
                <div>速度: {bus.speed}</div>
                <div>位宽: {bus.width}</div>
              </div>
            </motion.button>
          ))}
        </div>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>总线架构：</strong> 现代计算机使用分层总线结构，高速设备（CPU、内存）使用专用总线，
          低速设备通过 PCIe/SATA 等扩展总线连接。这种设计平衡了性能、成本和扩展性。
        </p>
      </div>
    </div>
  );
}
