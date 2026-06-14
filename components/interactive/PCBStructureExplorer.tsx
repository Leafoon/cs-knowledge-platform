"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Database, Info } from "lucide-react";

export default function PCBStructureExplorer() {
  const [selectedField, setSelectedField] = useState<string | null>(null);

  const pcbFields = [
    {
      category: "进程标识",
      color: "blue",
      fields: [
        { name: "PID", description: "进程 ID（唯一标识符）", example: "1234" },
        { name: "PPID", description: "父进程 ID", example: "1000" },
        { name: "进程组 ID", description: "用于作业控制", example: "1234" }
      ]
    },
    {
      category: "进程状态",
      color: "green",
      fields: [
        { name: "状态", description: "运行、就绪、阻塞、僵尸等", example: "RUNNING" },
        { name: "优先级", description: "调度优先级", example: "120" },
        { name: "调度策略", description: "FIFO、RR、CFS 等", example: "SCHED_NORMAL" }
      ]
    },
    {
      category: "CPU 状态",
      color: "purple",
      fields: [
        { name: "程序计数器 (PC)", description: "下一条指令的地址", example: "0x400123" },
        { name: "栈指针 (SP)", description: "栈顶地址", example: "0x7FFF1234" },
        { name: "通用寄存器", description: "RAX、RBX、RCX、RDX 等", example: "RAX=42" }
      ]
    },
    {
      category: "内存管理",
      color: "orange",
      fields: [
        { name: "页表基址", description: "页表起始地址（CR3）", example: "0x1A2B3C4D" },
        { name: "虚拟内存区域", description: "代码段、数据段、堆、栈范围", example: "VMA list" },
        { name: "内存限制", description: "最大虚拟内存、堆大小", example: "4GB" }
      ]
    },
    {
      category: "文件管理",
      color: "red",
      fields: [
        { name: "打开文件表", description: "文件描述符数组（fd 0-1023）", example: "[stdin, stdout, file.txt]" },
        { name: "当前工作目录", description: "pwd 的路径", example: "/home/user" },
        { name: "根目录", description: "chroot 的根路径", example: "/" }
      ]
    },
    {
      category: "权限与安全",
      color: "teal",
      fields: [
        { name: "用户 ID (UID)", description: "真实、有效、保存的 UID", example: "1000" },
        { name: "组 ID (GID)", description: "真实、有效、保存的 GID", example: "1000" },
        { name: "Capabilities", description: "细粒度权限", example: "CAP_NET_ADMIN" }
      ]
    }
  ];

  const getColorClass = (color: string) => {
    const colorMap: Record<string, { bg: string; text: string; border: string }> = {
      blue: { bg: "bg-blue-100", text: "text-blue-700", border: "border-blue-400" },
      green: { bg: "bg-green-100", text: "text-green-700", border: "border-green-400" },
      purple: { bg: "bg-purple-100", text: "text-purple-700", border: "border-purple-400" },
      orange: { bg: "bg-orange-100", text: "text-orange-700", border: "border-orange-400" },
      red: { bg: "bg-red-100", text: "text-red-700", border: "border-red-400" },
      teal: { bg: "bg-teal-100", text: "text-teal-700", border: "border-teal-400" }
    };
    return colorMap[color];
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Database className="w-7 h-7" />
        PCB 数据结构浏览器
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {pcbFields.map((category, catIdx) => (
          <motion.div
            key={category.category}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: catIdx * 0.1 }}
            className={`p-4 rounded-lg border-2 ${getColorClass(category.color).bg} ${getColorClass(category.color).border}`}
          >
            <h4 className={`font-bold mb-3 ${getColorClass(category.color).text}`}>
              {category.category}
            </h4>
            <div className="space-y-2">
              {category.fields.map((field, fieldIdx) => (
                <motion.div
                  key={fieldIdx}
                  whileHover={{ scale: 1.02 }}
                  onClick={() => setSelectedField(`${category.category}-${field.name}`)}
                  className={`bg-white p-3 rounded-lg cursor-pointer transition-all shadow-sm hover:shadow-md ${
                    selectedField === `${category.category}-${field.name}` ? "ring-2 ring-blue-500" : ""
                  }`}
                >
                  <div className="font-semibold text-slate-800 text-sm">{field.name}</div>
                  <div className="text-xs text-slate-600 mt-1">{field.description}</div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Detail Panel */}
      {selectedField && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 bg-white rounded-lg shadow-md p-6 border-2 border-blue-400"
        >
          <div className="flex items-center gap-2 mb-3">
            <Info className="w-5 h-5 text-blue-600" />
            <h4 className="text-lg font-bold text-slate-800">字段详情</h4>
          </div>
          {pcbFields.map(category =>
            category.fields.map(field =>
              selectedField === `${category.category}-${field.name}` ? (
                <div key={field.name}>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span className="text-sm font-semibold text-slate-600">字段名：</span>
                      <p className="font-mono text-slate-800">{field.name}</p>
                    </div>
                    <div>
                      <span className="text-sm font-semibold text-slate-600">分类：</span>
                      <p className="text-slate-800">{category.category}</p>
                    </div>
                  </div>
                  <div className="mt-3">
                    <span className="text-sm font-semibold text-slate-600">描述：</span>
                    <p className="text-slate-700">{field.description}</p>
                  </div>
                  <div className="mt-3">
                    <span className="text-sm font-semibold text-slate-600">示例值：</span>
                    <div className="mt-1 bg-slate-100 p-3 rounded font-mono text-sm">
                      {field.example}
                    </div>
                  </div>
                </div>
              ) : null
            )
          )}
        </motion.div>
      )}

      {/* Summary */}
      <div className="mt-6 bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <p className="text-sm text-slate-700">
          <strong>PCB（进程控制块）</strong>是操作系统为每个进程维护的核心数据结构，
          存储进程的所有元数据。所有进程管理操作（创建、调度、切换、终止）都依赖 PCB。
        </p>
      </div>
    </div>
  );
}
