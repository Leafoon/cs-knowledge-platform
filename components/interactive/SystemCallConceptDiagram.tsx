"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Layers, ArrowDown, ArrowUp, Shield } from "lucide-react";

export default function SystemCallConceptDiagram() {
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);

  const layers = [
    {
      id: 0,
      name: "用户应用程序",
      ring: "Ring 3",
      color: "blue",
      description: "用户态程序，如浏览器、编辑器、Shell",
      privileges: "受限：无法直接访问硬件、内核数据结构",
      examples: ["printf()", "malloc()", "业务逻辑"]
    },
    {
      id: 1,
      name: "C 标准库",
      ring: "Ring 3",
      color: "green",
      description: "用户态库函数，封装系统调用",
      privileges: "受限：提供易用接口，可能调用系统调用",
      examples: ["fopen() → open()", "printf() → write()", "malloc() → brk()"]
    },
    {
      id: 2,
      name: "系统调用接口",
      ring: "Ring 3 → Ring 0",
      color: "yellow",
      description: "用户态与内核态的边界，触发模式切换",
      privileges: "特权边界：使用 syscall/int 指令陷入内核",
      examples: ["open()", "read()", "fork()", "exec()"]
    },
    {
      id: 3,
      name: "操作系统内核",
      ring: "Ring 0",
      color: "red",
      description: "内核态代码，处理系统调用请求",
      privileges: "完全特权：访问所有硬件、内存、进程表",
      examples: ["sys_open()", "sys_read()", "调度器", "内存管理"]
    },
    {
      id: 4,
      name: "设备驱动程序",
      ring: "Ring 0",
      color: "purple",
      description: "内核模块，直接操作硬件",
      privileges: "完全特权：控制 I/O 端口、DMA、中断",
      examples: ["磁盘驱动", "网卡驱动", "显卡驱动"]
    },
    {
      id: 5,
      name: "硬件",
      ring: "物理层",
      color: "slate",
      description: "CPU、内存、磁盘、网卡等物理设备",
      privileges: "硬件执行：CPU 执行指令，I/O 设备响应",
      examples: ["CPU 寄存器", "物理内存", "磁盘扇区"]
    }
  ];

  const getColorClass = (color: string) => {
    const map: Record<string, { bg: string; border: string; text: string }> = {
      blue: { bg: "bg-blue-100", border: "border-blue-400", text: "text-blue-700" },
      green: { bg: "bg-green-100", border: "border-green-400", text: "text-green-700" },
      yellow: { bg: "bg-yellow-100", border: "border-yellow-400", text: "text-yellow-700" },
      red: { bg: "bg-red-100", border: "border-red-400", text: "text-red-700" },
      purple: { bg: "bg-purple-100", border: "border-purple-400", text: "text-purple-700" },
      slate: { bg: "bg-slate-100", border: "border-slate-400", text: "text-slate-700" }
    };
    return map[color];
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Layers className="w-7 h-7 text-blue-600" />
        系统调用分层架构
      </h3>

      {/* Layer Stack */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="space-y-3">
          {layers.map((layer, idx) => (
            <React.Fragment key={layer.id}>
              <motion.div
                whileHover={{ scale: 1.02 }}
                onClick={() => setSelectedLayer(layer.id)}
                className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  selectedLayer === layer.id
                    ? `${getColorClass(layer.color).bg} ${getColorClass(layer.color).border}`
                    : "bg-slate-50 border-slate-200 hover:border-slate-300"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`px-3 py-1 rounded font-bold text-xs ${
                      layer.ring.includes("Ring 3") ? "bg-blue-200 text-blue-800" :
                      layer.ring.includes("Ring 0") ? "bg-red-200 text-red-800" :
                      "bg-slate-200 text-slate-800"
                    }`}>
                      {layer.ring}
                    </div>
                    <div>
                      <div className="font-bold text-slate-800">{layer.name}</div>
                      <div className="text-sm text-slate-600">{layer.description}</div>
                    </div>
                  </div>
                  {layer.id === 2 && (
                    <div className="flex flex-col items-center gap-1">
                      <ArrowDown className="w-5 h-5 text-red-600 animate-bounce" />
                      <Shield className="w-5 h-5 text-yellow-600" />
                      <ArrowUp className="w-5 h-5 text-blue-600 animate-bounce" />
                    </div>
                  )}
                </div>
              </motion.div>

              {idx < layers.length - 1 && (
                <div className="flex justify-center">
                  <div className={`w-1 h-6 ${
                    idx === 2 ? "bg-yellow-400 animate-pulse" : "bg-slate-300"
                  }`}></div>
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Layer Detail */}
      {selectedLayer !== null && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`p-6 rounded-lg border-2 ${
            getColorClass(layers[selectedLayer].color).bg
          } ${getColorClass(layers[selectedLayer].color).border}`}
        >
          <h4 className="font-bold text-slate-800 mb-3">{layers[selectedLayer].name}</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-white bg-opacity-60 p-3 rounded">
              <div className="text-xs font-semibold text-slate-600 mb-1">特权级别</div>
              <div className="text-sm text-slate-800">{layers[selectedLayer].privileges}</div>
            </div>
            <div className="bg-white bg-opacity-60 p-3 rounded">
              <div className="text-xs font-semibold text-slate-600 mb-1">典型示例</div>
              <div className="text-sm text-slate-800">
                {layers[selectedLayer].examples.join(", ")}
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* System Call Flow */}
      <div className="mt-6 bg-white rounded-lg shadow-md p-6">
        <h4 className="font-bold text-slate-800 mb-4">系统调用执行流程</h4>
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold text-sm">1</div>
            <div className="flex-1 bg-blue-50 p-3 rounded border border-blue-200">
              <div className="font-semibold text-blue-800">用户程序调用库函数</div>
              <div className="text-sm text-slate-600">如 printf("Hello") → write()</div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-green-600 text-white rounded-full flex items-center justify-center font-bold text-sm">2</div>
            <div className="flex-1 bg-green-50 p-3 rounded border border-green-200">
              <div className="font-semibold text-green-800">C 库准备系统调用</div>
              <div className="text-sm text-slate-600">设置系统调用号、参数到寄存器</div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-yellow-600 text-white rounded-full flex items-center justify-center font-bold text-sm">3</div>
            <div className="flex-1 bg-yellow-50 p-3 rounded border border-yellow-200">
              <div className="font-semibold text-yellow-800">触发系统调用指令</div>
              <div className="text-sm text-slate-600">执行 syscall / int 0x80，陷入内核</div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-red-600 text-white rounded-full flex items-center justify-center font-bold text-sm">4</div>
            <div className="flex-1 bg-red-50 p-3 rounded border border-red-200">
              <div className="font-semibold text-red-800">内核处理系统调用</div>
              <div className="text-sm text-slate-600">检查参数、权限，调用 sys_write()</div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold text-sm">5</div>
            <div className="flex-1 bg-purple-50 p-3 rounded border border-purple-200">
              <div className="font-semibold text-purple-800">驱动程序执行 I/O</div>
              <div className="text-sm text-slate-600">终端驱动输出字符到屏幕</div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold text-sm">6</div>
            <div className="flex-1 bg-blue-50 p-3 rounded border border-blue-200">
              <div className="font-semibold text-blue-800">返回用户态</div>
              <div className="text-sm text-slate-600">恢复上下文，继续执行用户程序</div>
            </div>
          </div>
        </div>
      </div>

      {/* Key Points */}
      <div className="mt-6 bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <h4 className="font-bold text-amber-800 mb-2">关键要点</h4>
        <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
          <li><strong>Ring 3 vs Ring 0</strong>：用户态（Ring 3）无特权，内核态（Ring 0）可访问所有硬件</li>
          <li><strong>系统调用是唯一接口</strong>：用户程序无法直接进入内核，必须通过系统调用</li>
          <li><strong>模式切换开销</strong>：每次系统调用需保存/恢复上下文，约 100-1000 CPU 周期</li>
          <li><strong>安全检查</strong>：内核验证参数合法性、权限、资源配额</li>
        </ul>
      </div>
    </div>
  );
}
