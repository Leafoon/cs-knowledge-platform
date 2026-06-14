"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Cpu, MemoryStick, HardDrive, Users, Activity } from "lucide-react";

interface Resource {
  type: "CPU" | "内存" | "磁盘" | "进程";
  total: number;
  used: number;
  unit: string;
  icon: React.ReactNode;
  color: string;
}

interface Process {
  id: number;
  name: string;
  cpu: number;
  memory: number;
  disk: number;
}

export function OSResourceManager() {
  const [resources, setResources] = useState<Resource[]>([
    {
      type: "CPU",
      total: 100,
      used: 45,
      unit: "%",
      icon: <Cpu className="w-6 h-6" />,
      color: "#3b82f6",
    },
    {
      type: "内存",
      total: 16384,
      used: 8192,
      unit: "MB",
      icon: <MemoryStick className="w-6 h-6" />,
      color: "#10b981",
    },
    {
      type: "磁盘",
      total: 512,
      used: 256,
      unit: "GB",
      icon: <HardDrive className="w-6 h-6" />,
      color: "#f59e0b",
    },
    {
      type: "进程",
      total: 1024,
      used: 187,
      unit: "个",
      icon: <Users className="w-6 h-6" />,
      color: "#8b5cf6",
    },
  ]);

  const [processes, setProcesses] = useState<Process[]>([
    { id: 1, name: "chrome", cpu: 15, memory: 2048, disk: 5 },
    { id: 2, name: "code", cpu: 10, memory: 1024, disk: 3 },
    { id: 3, name: "firefox", cpu: 12, memory: 1536, disk: 4 },
    { id: 4, name: "system", cpu: 8, memory: 512, disk: 2 },
  ]);

  const [selectedResource, setSelectedResource] = useState<string | null>(null);

  const allocateResource = (type: string, amount: number) => {
    setResources((prev) =>
      prev.map((r) =>
        r.type === type
          ? { ...r, used: Math.min(r.total, r.used + amount) }
          : r
      )
    );
  };

  const releaseResource = (type: string, amount: number) => {
    setResources((prev) =>
      prev.map((r) =>
        r.type === type ? { ...r, used: Math.max(0, r.used - amount) } : r
      )
    );
  };

  const killProcess = (processId: number) => {
    const process = processes.find((p) => p.id === processId);
    if (process) {
      releaseResource("CPU", process.cpu);
      releaseResource("内存", process.memory);
      releaseResource("磁盘", process.disk);
      releaseResource("进程", 1);
      setProcesses((prev) => prev.filter((p) => p.id !== processId));
    }
  };

  const createProcess = () => {
    const newId = Math.max(...processes.map((p) => p.id), 0) + 1;
    const cpu = Math.floor(Math.random() * 20);
    const memory = Math.floor(Math.random() * 1024) + 256;
    const disk = Math.floor(Math.random() * 10);

    allocateResource("CPU", cpu);
    allocateResource("内存", memory);
    allocateResource("磁盘", disk);
    allocateResource("进程", 1);

    setProcesses((prev) => [
      ...prev,
      { id: newId, name: `proc-${newId}`, cpu, memory, disk },
    ]);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-6 text-text-primary">
        操作系统资源管理器
      </h3>

      {/* Resource Overview */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        {resources.map((resource) => {
          const percentage = (resource.used / resource.total) * 100;
          return (
            <motion.div
              key={resource.type}
              onClick={() =>
                setSelectedResource(
                  selectedResource === resource.type ? null : resource.type
                )
              }
              className={`p-4 rounded-lg border-2 cursor-pointer transition ${
                selectedResource === resource.type
                  ? "shadow-lg"
                  : "border-gray-300 dark:border-gray-700"
              }`}
              style={{
                borderColor:
                  selectedResource === resource.type ? resource.color : undefined,
                backgroundColor:
                  selectedResource === resource.type
                    ? `${resource.color}15`
                    : undefined,
              }}
              whileHover={{ scale: 1.05 }}
            >
              <div className="flex items-center gap-2 mb-3">
                <div style={{ color: resource.color }}>{resource.icon}</div>
                <h4 className="font-semibold text-text-primary">
                  {resource.type}
                </h4>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-text-secondary">已用</span>
                  <span className="font-semibold text-text-primary">
                    {resource.used} {resource.unit}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-text-secondary">总计</span>
                  <span className="font-semibold text-text-primary">
                    {resource.total} {resource.unit}
                  </span>
                </div>
                <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: resource.color }}
                    initial={{ width: 0 }}
                    animate={{ width: `${percentage}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <div className="text-right text-xs text-text-secondary">
                  {percentage.toFixed(1)}%
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Process Management */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-semibold text-text-primary">进程列表</h4>
          <button
            onClick={createProcess}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition text-sm"
          >
            + 创建进程
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="bg-gray-100 dark:bg-gray-800">
                <th className="px-3 py-2 text-left border border-border-subtle text-text-primary">
                  PID
                </th>
                <th className="px-3 py-2 text-left border border-border-subtle text-text-primary">
                  进程名
                </th>
                <th className="px-3 py-2 text-right border border-border-subtle text-text-primary">
                  CPU 使用率
                </th>
                <th className="px-3 py-2 text-right border border-border-subtle text-text-primary">
                  内存
                </th>
                <th className="px-3 py-2 text-right border border-border-subtle text-text-primary">
                  磁盘 I/O
                </th>
                <th className="px-3 py-2 text-center border border-border-subtle text-text-primary">
                  操作
                </th>
              </tr>
            </thead>
            <tbody>
              {processes.map((process) => (
                <tr
                  key={process.id}
                  className="hover:bg-gray-50 dark:hover:bg-gray-800"
                >
                  <td className="px-3 py-2 border border-border-subtle font-mono text-text-primary">
                    {process.id}
                  </td>
                  <td className="px-3 py-2 border border-border-subtle font-mono text-text-primary">
                    {process.name}
                  </td>
                  <td className="px-3 py-2 border border-border-subtle text-right text-text-primary">
                    {process.cpu}%
                  </td>
                  <td className="px-3 py-2 border border-border-subtle text-right text-text-primary">
                    {process.memory} MB
                  </td>
                  <td className="px-3 py-2 border border-border-subtle text-right text-text-primary">
                    {process.disk} MB/s
                  </td>
                  <td className="px-3 py-2 border border-border-subtle text-center">
                    <button
                      onClick={() => killProcess(process.id)}
                      className="px-3 py-1 bg-red-600 text-white rounded text-xs hover:bg-red-700"
                    >
                      终止
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Resource Allocation Visualization */}
      <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <h4 className="font-semibold mb-3 text-text-primary">
          资源分配可视化
        </h4>
        <div className="space-y-3">
          {resources.map((resource) => {
            const percentage = (resource.used / resource.total) * 100;
            const isCritical = percentage > 80;
            const isWarning = percentage > 60 && percentage <= 80;

            return (
              <div key={resource.type}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-text-primary">{resource.type}</span>
                  <span
                    className={`font-semibold ${
                      isCritical
                        ? "text-red-600 dark:text-red-400"
                        : isWarning
                        ? "text-yellow-600 dark:text-yellow-400"
                        : "text-green-600 dark:text-green-400"
                    }`}
                  >
                    {resource.used} / {resource.total} {resource.unit}
                  </span>
                </div>
                <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden relative">
                  <motion.div
                    className="h-full rounded-full"
                    style={{
                      backgroundColor: isCritical
                        ? "#dc2626"
                        : isWarning
                        ? "#f59e0b"
                        : resource.color,
                    }}
                    initial={{ width: 0 }}
                    animate={{ width: `${percentage}%` }}
                    transition={{ duration: 0.5 }}
                  />
                  <div className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-white">
                    {percentage.toFixed(1)}%
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Info Box */}
      <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <div className="flex items-start gap-3">
          <Activity className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-text-secondary">
            <p className="mb-2">
              <strong className="text-text-primary">操作系统核心职责</strong>：
              管理和分配系统资源（CPU、内存、磁盘、I/O 设备），为应用程序提供抽象接口，确保系统高效、安全运行。
            </p>
            <p>
              <strong className="text-text-primary">资源管理策略</strong>：
              动态分配、优先级调度、公平性保证、死锁避免、资源回收。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
