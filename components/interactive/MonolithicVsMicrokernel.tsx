"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Layers, Box, Network } from "lucide-react";

type ArchitectureType = "monolithic" | "microkernel";

interface Component {
  name: string;
  description: string;
  location: "kernel" | "user";
  color: string;
}

const monolithicComponents: Component[] = [
  {
    name: "进程管理",
    description: "调度、上下文切换",
    location: "kernel",
    color: "#3b82f6",
  },
  {
    name: "内存管理",
    description: "分页、虚拟内存",
    location: "kernel",
    color: "#10b981",
  },
  {
    name: "文件系统",
    description: "VFS、ext4、XFS",
    location: "kernel",
    color: "#f59e0b",
  },
  {
    name: "设备驱动",
    description: "网卡、显卡、磁盘",
    location: "kernel",
    color: "#8b5cf6",
  },
  {
    name: "网络协议栈",
    description: "TCP/IP、Socket",
    location: "kernel",
    color: "#ec4899",
  },
  {
    name: "IPC 机制",
    description: "管道、信号、共享内存",
    location: "kernel",
    color: "#06b6d4",
  },
];

const microkernelComponents: Component[] = [
  {
    name: "微内核",
    description: "进程调度、IPC、地址空间",
    location: "kernel",
    color: "#dc2626",
  },
  {
    name: "文件服务器",
    description: "VFS 服务",
    location: "user",
    color: "#f59e0b",
  },
  {
    name: "内存服务器",
    description: "页面管理",
    location: "user",
    color: "#10b981",
  },
  {
    name: "设备驱动",
    description: "用户态驱动",
    location: "user",
    color: "#8b5cf6",
  },
  {
    name: "网络栈",
    description: "用户态协议栈",
    location: "user",
    color: "#ec4899",
  },
];

export function MonolithicVsMicrokernel() {
  const [architecture, setArchitecture] = useState<ArchitectureType>("monolithic");
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [showComparison, setShowComparison] = useState(false);

  const components =
    architecture === "monolithic" ? monolithicComponents : microkernelComponents;

  const comparisonData = [
    {
      aspect: "性能",
      monolithic: "高 (直接函数调用)",
      microkernel: "较低 (IPC 开销)",
      winner: "monolithic",
    },
    {
      aspect: "可靠性",
      monolithic: "中 (内核崩溃影响全系统)",
      microkernel: "高 (服务隔离，局部故障)",
      winner: "microkernel",
    },
    {
      aspect: "安全性",
      monolithic: "中 (内核权限过大)",
      microkernel: "高 (最小特权原则)",
      winner: "microkernel",
    },
    {
      aspect: "可维护性",
      monolithic: "中 (模块耦合紧密)",
      microkernel: "高 (模块化设计)",
      winner: "microkernel",
    },
    {
      aspect: "开发复杂度",
      monolithic: "低 (统一地址空间)",
      microkernel: "高 (IPC 设计复杂)",
      winner: "monolithic",
    },
    {
      aspect: "内存占用",
      monolithic: "中 (内核较大)",
      microkernel: "低 (微内核精简)",
      winner: "microkernel",
    },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-6 text-text-primary">
        宏内核 vs 微内核架构对比
      </h3>

      {/* Architecture Selection */}
      <div className="mb-6 flex gap-4">
        <button
          onClick={() => {
            setArchitecture("monolithic");
            setSelectedComponent(null);
          }}
          className={`flex-1 p-4 rounded-lg border-2 transition ${
            architecture === "monolithic"
              ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20 shadow-lg"
              : "border-gray-300 dark:border-gray-700 hover:border-gray-400"
          }`}
        >
          <div className="flex items-center gap-3 mb-2">
            <Box className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            <h4 className="font-semibold text-lg text-text-primary">
              宏内核 (Monolithic)
            </h4>
          </div>
          <p className="text-sm text-text-secondary">
            所有服务运行在内核态，直接函数调用
          </p>
          <div className="mt-2 text-xs text-text-secondary">
            示例：Linux, Windows, *BSD
          </div>
        </button>

        <button
          onClick={() => {
            setArchitecture("microkernel");
            setSelectedComponent(null);
          }}
          className={`flex-1 p-4 rounded-lg border-2 transition ${
            architecture === "microkernel"
              ? "border-red-500 bg-red-50 dark:bg-red-900/20 shadow-lg"
              : "border-gray-300 dark:border-gray-700 hover:border-gray-400"
          }`}
        >
          <div className="flex items-center gap-3 mb-2">
            <Layers className="w-6 h-6 text-red-600 dark:text-red-400" />
            <h4 className="font-semibold text-lg text-text-primary">
              微内核 (Microkernel)
            </h4>
          </div>
          <p className="text-sm text-text-secondary">
            最小内核 + 用户态服务，通过 IPC 通信
          </p>
          <div className="mt-2 text-xs text-text-secondary">
            示例：MINIX, seL4, QNX, Fuchsia
          </div>
        </button>
      </div>

      {/* Architecture Diagram */}
      <div className="mb-6 p-6 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <h4 className="font-semibold mb-4 text-text-primary">架构示意图</h4>

        {architecture === "monolithic" ? (
          <div className="space-y-3">
            {/* User Space */}
            <div className="p-4 bg-blue-100 dark:bg-blue-900/30 rounded-lg border-2 border-blue-300 dark:border-blue-700">
              <div className="text-sm font-semibold text-blue-700 dark:text-blue-300 mb-2">
                用户空间 (User Space)
              </div>
              <div className="flex gap-2 flex-wrap">
                {["应用程序 A", "应用程序 B", "应用程序 C"].map((app) => (
                  <div
                    key={app}
                    className="px-3 py-1 bg-white dark:bg-gray-800 rounded text-xs text-text-primary border border-gray-300 dark:border-gray-600"
                  >
                    {app}
                  </div>
                ))}
              </div>
            </div>

            {/* System Call Interface */}
            <div className="text-center py-2 border-t-2 border-b-2 border-dashed border-gray-400">
              <span className="text-xs font-semibold text-text-secondary">
                系统调用接口 (System Call Interface)
              </span>
            </div>

            {/* Kernel Space */}
            <div className="p-4 bg-red-100 dark:bg-red-900/30 rounded-lg border-2 border-red-400 dark:border-red-700">
              <div className="text-sm font-semibold text-red-700 dark:text-red-300 mb-3">
                内核空间 (Kernel Space) - 所有服务都在这里
              </div>
              <div className="grid grid-cols-3 gap-2">
                {monolithicComponents.map((comp) => (
                  <motion.div
                    key={comp.name}
                    onClick={() =>
                      setSelectedComponent(
                        selectedComponent === comp.name ? null : comp.name
                      )
                    }
                    className={`p-2 rounded cursor-pointer border-2 ${
                      selectedComponent === comp.name
                        ? "shadow-lg scale-105"
                        : ""
                    }`}
                    style={{
                      backgroundColor: `${comp.color}30`,
                      borderColor:
                        selectedComponent === comp.name ? comp.color : "transparent",
                    }}
                    whileHover={{ scale: 1.05 }}
                  >
                    <div className="text-xs font-semibold text-text-primary">
                      {comp.name}
                    </div>
                    <div className="text-xs text-text-secondary mt-1">
                      {comp.description}
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Hardware */}
            <div className="p-3 bg-gray-200 dark:bg-gray-800 rounded-lg text-center">
              <span className="text-sm font-semibold text-text-secondary">
                硬件 (Hardware)
              </span>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {/* User Space Services */}
            <div className="p-4 bg-blue-100 dark:bg-blue-900/30 rounded-lg border-2 border-blue-300 dark:border-blue-700">
              <div className="text-sm font-semibold text-blue-700 dark:text-blue-300 mb-3">
                用户空间 (User Space) - 应用程序 + 系统服务
              </div>
              <div className="grid grid-cols-3 gap-2 mb-3">
                {["应用程序 A", "应用程序 B", "应用程序 C"].map((app) => (
                  <div
                    key={app}
                    className="px-3 py-1 bg-white dark:bg-gray-800 rounded text-xs text-text-primary border border-gray-300 dark:border-gray-600"
                  >
                    {app}
                  </div>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-2">
                {microkernelComponents
                  .filter((c) => c.location === "user")
                  .map((comp) => (
                    <motion.div
                      key={comp.name}
                      onClick={() =>
                        setSelectedComponent(
                          selectedComponent === comp.name ? null : comp.name
                        )
                      }
                      className={`p-2 rounded cursor-pointer border-2 ${
                        selectedComponent === comp.name
                          ? "shadow-lg scale-105"
                          : ""
                      }`}
                      style={{
                        backgroundColor: `${comp.color}30`,
                        borderColor:
                          selectedComponent === comp.name
                            ? comp.color
                            : "transparent",
                      }}
                      whileHover={{ scale: 1.05 }}
                    >
                      <div className="text-xs font-semibold text-text-primary flex items-center gap-1">
                        <Network className="w-3 h-3" />
                        {comp.name}
                      </div>
                      <div className="text-xs text-text-secondary mt-1">
                        {comp.description}
                      </div>
                    </motion.div>
                  ))}
              </div>
            </div>

            {/* IPC */}
            <div className="text-center py-2 border-t-2 border-b-2 border-dashed border-gray-400">
              <span className="text-xs font-semibold text-text-secondary">
                进程间通信 (IPC - Message Passing)
              </span>
            </div>

            {/* Microkernel */}
            <div className="p-4 bg-red-100 dark:bg-red-900/30 rounded-lg border-2 border-red-400 dark:border-red-700">
              <div className="text-sm font-semibold text-red-700 dark:text-red-300 mb-3">
                微内核 (Microkernel) - 最小化内核
              </div>
              <motion.div
                onClick={() =>
                  setSelectedComponent(
                    selectedComponent === "微内核" ? null : "微内核"
                  )
                }
                className={`p-3 rounded cursor-pointer border-2 ${
                  selectedComponent === "微内核" ? "shadow-lg scale-105" : ""
                }`}
                style={{
                  backgroundColor: "#dc262630",
                  borderColor:
                    selectedComponent === "微内核" ? "#dc2626" : "transparent",
                }}
                whileHover={{ scale: 1.05 }}
              >
                <div className="text-sm font-semibold text-text-primary">
                  核心功能：进程调度 + IPC + 地址空间管理
                </div>
                <div className="text-xs text-text-secondary mt-1">
                  只保留最基本的抽象，其他服务移到用户态
                </div>
              </motion.div>
            </div>

            {/* Hardware */}
            <div className="p-3 bg-gray-200 dark:bg-gray-800 rounded-lg text-center">
              <span className="text-sm font-semibold text-text-secondary">
                硬件 (Hardware)
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Comparison Table */}
      <div className="mb-6">
        <button
          onClick={() => setShowComparison(!showComparison)}
          className="mb-4 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition"
        >
          {showComparison ? "隐藏对比表" : "显示详细对比"}
        </button>

        {showComparison && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="overflow-x-auto"
          >
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="bg-gray-100 dark:bg-gray-800">
                  <th className="px-4 py-3 text-left border border-border-subtle text-text-primary">
                    对比维度
                  </th>
                  <th className="px-4 py-3 text-left border border-border-subtle text-blue-600 dark:text-blue-400">
                    宏内核
                  </th>
                  <th className="px-4 py-3 text-left border border-border-subtle text-red-600 dark:text-red-400">
                    微内核
                  </th>
                  <th className="px-4 py-3 text-center border border-border-subtle text-text-primary">
                    优势
                  </th>
                </tr>
              </thead>
              <tbody>
                {comparisonData.map((row, index) => (
                  <tr
                    key={row.aspect}
                    className={
                      index % 2 === 0
                        ? "bg-gray-50 dark:bg-gray-900"
                        : "bg-white dark:bg-gray-950"
                    }
                  >
                    <td className="px-4 py-3 border border-border-subtle font-semibold text-text-primary">
                      {row.aspect}
                    </td>
                    <td
                      className={`px-4 py-3 border border-border-subtle ${
                        row.winner === "monolithic"
                          ? "bg-blue-50 dark:bg-blue-900/20 font-semibold"
                          : ""
                      }`}
                    >
                      {row.monolithic}
                    </td>
                    <td
                      className={`px-4 py-3 border border-border-subtle ${
                        row.winner === "microkernel"
                          ? "bg-red-50 dark:bg-red-900/20 font-semibold"
                          : ""
                      }`}
                    >
                      {row.microkernel}
                    </td>
                    <td className="px-4 py-3 border border-border-subtle text-center">
                      {row.winner === "monolithic" ? "🔵" : "🔴"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </motion.div>
        )}
      </div>

      {/* Real World Examples */}
      <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
        <h4 className="font-semibold mb-2 text-yellow-800 dark:text-yellow-300">
          实际应用
        </h4>
        <div className="text-sm text-text-secondary space-y-2">
          <p>
            <strong className="text-text-primary">宏内核</strong>：
            Linux (世界上最流行的服务器 OS)、Windows、macOS、FreeBSD
            <br />
            优势：高性能、成熟生态、开发简单
          </p>
          <p>
            <strong className="text-text-primary">微内核</strong>：
            MINIX 3 (教学)、seL4 (形式化验证)、QNX (汽车/工业)、Fuchsia (Google
            新 OS)
            <br />
            优势：高可靠性、安全性、适合关键系统
          </p>
          <p>
            <strong className="text-text-primary">混合内核</strong>：
            Windows NT 系列实际上是混合内核，结合了两者优点
          </p>
        </div>
      </div>
    </div>
  );
}
