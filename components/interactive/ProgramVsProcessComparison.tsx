"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { FileCode, Cpu, HardDrive, Activity, Box, Users } from "lucide-react";

export default function ProgramVsProcessComparison() {
  const [activeView, setActiveView] = useState<"program" | "process">("program");

  const comparisonData = [
    {
      aspect: "定义",
      program: "存储在磁盘上的可执行文件",
      process: "程序的执行实例",
      icon: <FileCode className="w-5 h-5" />
    },
    {
      aspect: "状态",
      program: "静态（被动）",
      process: "动态（主动）",
      icon: <Activity className="w-5 h-5" />
    },
    {
      aspect: "持久性",
      program: "长期存在（直到删除）",
      process: "临时（运行结束后消失）",
      icon: <HardDrive className="w-5 h-5" />
    },
    {
      aspect: "组成",
      program: "代码段、数据段、符号表",
      process: "代码、数据、堆、栈、PCB、寄存器、文件描述符",
      icon: <Box className="w-5 h-5" />
    },
    {
      aspect: "资源消耗",
      program: "占用磁盘空间",
      process: "占用 CPU、内存、文件等系统资源",
      icon: <Cpu className="w-5 h-5" />
    },
    {
      aspect: "数量关系",
      program: "一个程序文件",
      process: "可以有多个进程实例",
      icon: <Users className="w-5 h-5" />
    }
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center">
        程序 vs 进程：静态 vs 动态
      </h3>

      {/* Toggle View */}
      <div className="flex justify-center mb-6 gap-4">
        <button
          onClick={() => setActiveView("program")}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            activeView === "program"
              ? "bg-blue-600 text-white shadow-lg scale-105"
              : "bg-white text-slate-600 hover:bg-slate-100"
          }`}
        >
          <FileCode className="inline w-5 h-5 mr-2" />
          程序（Program）
        </button>
        <button
          onClick={() => setActiveView("process")}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            activeView === "process"
              ? "bg-green-600 text-white shadow-lg scale-105"
              : "bg-white text-slate-600 hover:bg-slate-100"
          }`}
        >
          <Activity className="inline w-5 h-5 mr-2" />
          进程（Process）
        </button>
      </div>

      {/* Comparison Table */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="bg-slate-100 border-b-2 border-slate-200">
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">对比维度</th>
              <th className="px-4 py-3 text-left text-blue-700 font-semibold">
                <FileCode className="inline w-4 h-4 mr-2" />
                程序（Program）
              </th>
              <th className="px-4 py-3 text-left text-green-700 font-semibold">
                <Activity className="inline w-4 h-4 mr-2" />
                进程（Process）
              </th>
            </tr>
          </thead>
          <tbody>
            {comparisonData.map((item, idx) => (
              <motion.tr
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                className={`border-b border-slate-100 hover:bg-slate-50 transition-colors ${
                  activeView === "program" ? "bg-blue-50/30" : "bg-green-50/30"
                }`}
              >
                <td className="px-4 py-3 flex items-center gap-2 font-medium text-slate-700">
                  {item.icon}
                  {item.aspect}
                </td>
                <td
                  className={`px-4 py-3 ${
                    activeView === "program" ? "bg-blue-100 font-semibold" : ""
                  }`}
                >
                  {item.program}
                </td>
                <td
                  className={`px-4 py-3 ${
                    activeView === "process" ? "bg-green-100 font-semibold" : ""
                  }`}
                >
                  {item.process}
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Example Visualization */}
      <div className="mt-6 grid grid-cols-2 gap-4">
        {/* Program Visualization */}
        <motion.div
          className={`p-4 rounded-lg border-2 transition-all ${
            activeView === "program"
              ? "bg-blue-100 border-blue-400 shadow-lg scale-105"
              : "bg-white border-slate-200"
          }`}
          whileHover={{ scale: 1.02 }}
        >
          <h4 className="font-bold text-slate-800 mb-3 flex items-center gap-2">
            <HardDrive className="w-5 h-5 text-blue-600" />
            程序（磁盘文件）
          </h4>
          <div className="space-y-2 text-sm">
            <div className="bg-slate-100 p-2 rounded font-mono">/bin/ls</div>
            <div className="bg-slate-100 p-2 rounded font-mono">a.out</div>
            <div className="bg-slate-100 p-2 rounded font-mono">program.exe</div>
            <p className="text-xs text-slate-600 mt-2">
              静态文件，长期存储在磁盘，不消耗 CPU 和内存
            </p>
          </div>
        </motion.div>

        {/* Process Visualization */}
        <motion.div
          className={`p-4 rounded-lg border-2 transition-all ${
            activeView === "process"
              ? "bg-green-100 border-green-400 shadow-lg scale-105"
              : "bg-white border-slate-200"
          }`}
          whileHover={{ scale: 1.02 }}
        >
          <h4 className="font-bold text-slate-800 mb-3 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-green-600" />
            进程（运行实例）
          </h4>
          <div className="space-y-2 text-sm">
            <div className="bg-green-200 p-2 rounded flex justify-between">
              <span className="font-mono">ls (PID 1234)</span>
              <span className="text-xs">运行中</span>
            </div>
            <div className="bg-green-200 p-2 rounded flex justify-between">
              <span className="font-mono">ls (PID 1235)</span>
              <span className="text-xs">运行中</span>
            </div>
            <div className="bg-green-200 p-2 rounded flex justify-between">
              <span className="font-mono">ls (PID 1236)</span>
              <span className="text-xs">运行中</span>
            </div>
            <p className="text-xs text-slate-600 mt-2">
              同一程序的多个执行实例，消耗 CPU、内存、文件资源
            </p>
          </div>
        </motion.div>
      </div>

      {/* Analogy */}
      <div className="mt-6 bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <p className="text-sm text-slate-700">
          <strong className="text-amber-700">类比：</strong>
          <br />
          <span className="font-mono text-blue-600">程序</span> = <strong>食谱</strong>（书架上的菜谱，静态文本）
          <br />
          <span className="font-mono text-green-600">进程</span> = <strong>烹饪过程</strong>
          （正在制作菜肴的厨师，使用食材和工具）
        </p>
      </div>
    </div>
  );
}
