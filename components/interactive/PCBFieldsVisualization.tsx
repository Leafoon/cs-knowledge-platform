"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Hash, Cpu, MemoryStick, FileText, Shield, Signal } from "lucide-react";

export default function PCBFieldsVisualization() {
  const [hoveredField, setHoveredField] = useState<string | null>(null);

  const fieldCategories = [
    {
      name: "身份信息",
      icon: <Hash className="w-5 h-5" />,
      color: "blue",
      fields: [
        { key: "pid", label: "PID", value: "1234", desc: "进程唯一标识符" },
        { key: "ppid", label: "PPID", value: "1000", desc: "父进程 ID" },
        { key: "pgid", label: "PGID", value: "1234", desc: "进程组 ID" }
      ]
    },
    {
      name: "CPU 寄存器",
      icon: <Cpu className="w-5 h-5" />,
      color: "green",
      fields: [
        { key: "rip", label: "RIP", value: "0x400ABC", desc: "指令指针（下一条指令地址）" },
        { key: "rsp", label: "RSP", value: "0x7FFF1234", desc: "栈指针（栈顶地址）" },
        { key: "rax", label: "RAX", value: "0x0000002A", desc: "通用寄存器 A（返回值）" }
      ]
    },
    {
      name: "内存指针",
      icon: <MemoryStick className="w-5 h-5" />,
      color: "purple",
      fields: [
        { key: "cr3", label: "CR3", value: "0x1A2B3C4D", desc: "页表基址寄存器" },
        { key: "brk", label: "brk", value: "0x600000", desc: "堆顶指针" },
        { key: "stack_start", label: "stack", value: "0x7FFF0000", desc: "栈起始地址" }
      ]
    },
    {
      name: "文件描述符",
      icon: <FileText className="w-5 h-5" />,
      color: "orange",
      fields: [
        { key: "fd0", label: "fd[0]", value: "stdin", desc: "标准输入" },
        { key: "fd1", label: "fd[1]", value: "stdout", desc: "标准输出" },
        { key: "fd3", label: "fd[3]", value: "file.txt", desc: "打开的文件" }
      ]
    },
    {
      name: "权限与安全",
      icon: <Shield className="w-5 h-5" />,
      color: "red",
      fields: [
        { key: "uid", label: "UID", value: "1000", desc: "用户 ID" },
        { key: "gid", label: "GID", value: "1000", desc: "组 ID" },
        { key: "caps", label: "Caps", value: "0x0", desc: "Capabilities" }
      ]
    },
    {
      name: "信号处理",
      icon: <Signal className="w-5 h-5" />,
      color: "teal",
      fields: [
        { key: "pending", label: "pending", value: "0b0010", desc: "挂起信号位图" },
        { key: "blocked", label: "blocked", value: "0b0000", desc: "阻塞信号位图" },
        { key: "handler", label: "handler", value: "0x401000", desc: "信号处理器地址" }
      ]
    }
  ];

  const getColorClass = (color: string) => {
    const map: Record<string, string> = {
      blue: "bg-blue-500",
      green: "bg-green-500",
      purple: "bg-purple-500",
      orange: "bg-orange-500",
      red: "bg-red-500",
      teal: "bg-teal-500"
    };
    return map[color];
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center">
        PCB 关键字段可视化
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        {fieldCategories.map((category, idx) => (
          <motion.div
            key={category.name}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: idx * 0.1 }}
            className="bg-white rounded-lg shadow-md p-4"
          >
            <div className={`flex items-center gap-2 mb-3 ${getColorClass(category.color)} text-white p-2 rounded-lg`}>
              {category.icon}
              <h4 className="font-bold">{category.name}</h4>
            </div>
            <div className="space-y-2">
              {category.fields.map(field => (
                <motion.div
                  key={field.key}
                  whileHover={{ scale: 1.02 }}
                  onMouseEnter={() => setHoveredField(field.key)}
                  onMouseLeave={() => setHoveredField(null)}
                  className={`p-3 rounded-lg cursor-pointer transition-all ${
                    hoveredField === field.key
                      ? "bg-blue-100 border-2 border-blue-400"
                      : "bg-slate-50 border-2 border-slate-200"
                  }`}
                >
                  <div className="flex justify-between items-center mb-1">
                    <span className="font-semibold text-slate-800 text-sm">{field.label}</span>
                    <span className="font-mono text-xs text-slate-600">{field.value}</span>
                  </div>
                  {hoveredField === field.key && (
                    <motion.p
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      className="text-xs text-slate-600 mt-1"
                    >
                      {field.desc}
                    </motion.p>
                  )}
                </motion.div>
              ))}
            </div>
          </motion.div>
        ))}
      </div>

      {/* PCB Structure Diagram */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h4 className="font-bold text-slate-800 mb-4">PCB 内存布局</h4>
        <div className="border-2 border-slate-300 rounded-lg p-4 font-mono text-sm space-y-1">
          <div className="bg-blue-100 p-2 rounded">struct process_control_block &#123;</div>
          {fieldCategories.map(category =>
            category.fields.map(field => (
              <div
                key={field.key}
                className={`pl-6 p-1 rounded transition-all ${
                  hoveredField === field.key ? "bg-yellow-100 font-bold" : ""
                }`}
              >
                {field.label}: {field.value};
              </div>
            ))
          )}
          <div className="bg-blue-100 p-2 rounded">&#125;;</div>
        </div>
      </div>
    </div>
  );
}
