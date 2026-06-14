"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Cpu, MemoryStick, HardDrive, FileText, Shield, Hash } from "lucide-react";

export default function ProcessResourceVisualization() {
  const [selectedResource, setSelectedResource] = useState<string | null>(null);

  const resources = [
    {
      id: "memory",
      name: "内存空间",
      icon: <MemoryStick className="w-6 h-6" />,
      color: "blue",
      description: "代码段、数据段、堆、栈",
      examples: ["虚拟地址空间: 4GB (32位) / 256TB (64位)", "堆: malloc() 动态分配", "栈: 函数调用栈"],
      size: "大"
    },
    {
      id: "cpu",
      name: "CPU 状态",
      icon: <Cpu className="w-6 h-6" />,
      color: "green",
      description: "寄存器（PC、SP、通用寄存器）",
      examples: ["RIP: 指令指针", "RSP: 栈指针", "RAX、RBX: 通用寄存器"],
      size: "小"
    },
    {
      id: "files",
      name: "文件资源",
      icon: <FileText className="w-6 h-6" />,
      color: "purple",
      description: "打开的文件描述符",
      examples: ["stdin (fd 0)", "stdout (fd 1)", "stderr (fd 2)", "其他打开的文件"],
      size: "中"
    },
    {
      id: "io",
      name: "I/O 设备",
      icon: <HardDrive className="w-6 h-6" />,
      color: "orange",
      description: "分配的设备（终端、网络接口）",
      examples: ["TTY 终端", "Socket 网络连接", "磁盘 I/O"],
      size: "中"
    },
    {
      id: "id",
      name: "进程 ID",
      icon: <Hash className="w-6 h-6" />,
      color: "red",
      description: "唯一标识符",
      examples: ["PID: 1234", "PPID: 1000 (父进程)", "PGID: 进程组 ID"],
      size: "极小"
    },
    {
      id: "permission",
      name: "权限信息",
      icon: <Shield className="w-6 h-6" />,
      color: "teal",
      description: "用户 ID、组 ID、权限位",
      examples: ["UID: 1000", "GID: 1000", "Capabilities: CAP_NET_ADMIN"],
      size: "极小"
    }
  ];

  const getColorClass = (color: string, variant: "bg" | "border" | "text") => {
    const colorMap: Record<string, Record<string, string>> = {
      blue: { bg: "bg-blue-100", border: "border-blue-400", text: "text-blue-700" },
      green: { bg: "bg-green-100", border: "border-green-400", text: "text-green-700" },
      purple: { bg: "bg-purple-100", border: "border-purple-400", text: "text-purple-700" },
      orange: { bg: "bg-orange-100", border: "border-orange-400", text: "text-orange-700" },
      red: { bg: "bg-red-100", border: "border-red-400", text: "text-red-700" },
      teal: { bg: "bg-teal-100", border: "border-teal-400", text: "text-teal-700" }
    };
    return colorMap[color]?.[variant] || "";
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center">
        进程资源可视化
      </h3>

      {/* Resource Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        {resources.map((resource, idx) => (
          <motion.div
            key={resource.id}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: idx * 0.1 }}
            whileHover={{ scale: 1.05 }}
            onClick={() => setSelectedResource(resource.id)}
            className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
              selectedResource === resource.id
                ? `${getColorClass(resource.color, "bg")} ${getColorClass(resource.color, "border")} shadow-lg`
                : "bg-white border-slate-200 hover:border-slate-300"
            }`}
          >
            <div className="flex items-center gap-3 mb-2">
              <div className={`p-2 rounded-lg ${getColorClass(resource.color, "bg")}`}>
                {resource.icon}
              </div>
              <h4 className={`font-bold ${getColorClass(resource.color, "text")}`}>
                {resource.name}
              </h4>
            </div>
            <p className="text-sm text-slate-600 mb-2">{resource.description}</p>
            <div className={`text-xs font-semibold ${getColorClass(resource.color, "text")}`}>
              资源占用: {resource.size}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Detail Panel */}
      {selectedResource && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`p-6 rounded-lg border-2 ${
            getColorClass(resources.find(r => r.id === selectedResource)!.color, "bg")
          } ${getColorClass(resources.find(r => r.id === selectedResource)!.color, "border")}`}
        >
          <div className="flex items-center gap-3 mb-4">
            {resources.find(r => r.id === selectedResource)!.icon}
            <h4 className="text-xl font-bold text-slate-800">
              {resources.find(r => r.id === selectedResource)!.name}
            </h4>
          </div>
          <p className="text-slate-700 mb-4">
            {resources.find(r => r.id === selectedResource)!.description}
          </p>
          <div>
            <h5 className="font-semibold text-slate-800 mb-2">示例：</h5>
            <ul className="space-y-2">
              {resources.find(r => r.id === selectedResource)!.examples.map((example, idx) => (
                <li key={idx} className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-slate-600" />
                  <span className="text-sm font-mono text-slate-700">{example}</span>
                </li>
              ))}
            </ul>
          </div>
        </motion.div>
      )}

      {/* Process Resource Summary */}
      <div className="mt-6 bg-white rounded-lg p-4 shadow-md">
        <h4 className="font-bold text-slate-800 mb-3">资源分配总览</h4>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {resources.map(resource => (
            <div key={resource.id} className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${getColorClass(resource.color, "bg")}`} />
              <span className="text-sm text-slate-700">{resource.name}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
