"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Clock, TrendingUp } from "lucide-react";

export default function SyscallOverheadBreakdown() {
  const [selectedPhase, setSelectedPhase] = useState<number | null>(null);

  const phases = [
    {
      id: 0,
      name: "用户态准备",
      cycles: "10-20",
      description: "C 库设置系统调用号、参数到寄存器",
      details: ["设置 rax = syscall_number", "rdi, rsi, rdx 等寄存器存参数", "执行 syscall 指令"],
      color: "blue"
    },
    {
      id: 1,
      name: "CPU 模式切换",
      cycles: "50-100",
      description: "从 Ring 3 切换到 Ring 0，硬件自动完成",
      details: ["保存用户态 RIP、RSP、RFLAGS", "加载内核态栈指针（TSS）", "切换到内核代码段", "权限级别 Ring 3 → Ring 0"],
      color: "yellow"
    },
    {
      id: 2,
      name: "保存用户上下文",
      cycles: "20-40",
      description: "内核保存用户态寄存器到内核栈",
      details: ["保存通用寄存器（15 个）", "保存段寄存器", "保存浮点状态（FPU/SSE）"],
      color: "orange"
    },
    {
      id: 3,
      name: "系统调用分派",
      cycles: "10-20",
      description: "查找系统调用表，跳转到对应处理函数",
      details: ["验证系统调用号合法性", "sys_call_table[rax] 查表", "跳转到 sys_xxx()"],
      color: "red"
    },
    {
      id: 4,
      name: "参数检查与复制",
      cycles: "50-200",
      description: "内核验证参数合法性、权限、内存访问",
      details: ["检查指针是否在用户空间", "验证文件描述符有效性", "权限检查（UID/GID）", "从用户空间复制数据到内核"],
      color: "purple"
    },
    {
      id: 5,
      name: "执行系统调用逻辑",
      cycles: "100-10000+",
      description: "实际执行系统调用功能（变化最大）",
      details: ["文件系统操作（inode 查找）", "进程管理（调度、fork）", "内存管理（页表操作）", "设备 I/O（可能阻塞）"],
      color: "green"
    },
    {
      id: 6,
      name: "恢复用户上下文",
      cycles: "20-40",
      description: "从内核栈恢复用户态寄存器",
      details: ["恢复通用寄存器", "恢复段寄存器", "恢复浮点状态"],
      color: "teal"
    },
    {
      id: 7,
      name: "返回用户态",
      cycles: "50-100",
      description: "从 Ring 0 切换回 Ring 3",
      details: ["恢复用户态 RIP、RSP、RFLAGS", "切换到用户代码段", "权限级别 Ring 0 → Ring 3", "执行 sysret / iret 指令"],
      color: "indigo"
    }
  ];

  const totalCycles = phases.reduce((sum, p) => {
    const [min] = p.cycles.split("-").map(s => parseInt(s.replace("+", "")));
    return sum + min;
  }, 0);

  const getColorClass = (color: string) => {
    const map: Record<string, string> = {
      blue: "bg-blue-100 border-blue-400 text-blue-800",
      yellow: "bg-yellow-100 border-yellow-400 text-yellow-800",
      orange: "bg-orange-100 border-orange-400 text-orange-800",
      red: "bg-red-100 border-red-400 text-red-800",
      purple: "bg-purple-100 border-purple-400 text-purple-800",
      green: "bg-green-100 border-green-400 text-green-800",
      teal: "bg-teal-100 border-teal-400 text-teal-800",
      indigo: "bg-indigo-100 border-indigo-400 text-indigo-800"
    };
    return map[color];
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Clock className="w-7 h-7 text-orange-600" />
        系统调用性能开销分解
      </h3>

      {/* Summary */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-sm text-slate-600">总开销（最小值估算）</div>
            <div className="text-3xl font-bold text-orange-600">{totalCycles}+ 周期</div>
          </div>
          <div className="text-right">
            <div className="text-sm text-slate-600">约等于</div>
            <div className="text-xl font-bold text-slate-800">
              {(totalCycles / 3000).toFixed(1)} - {(totalCycles / 1000).toFixed(1)} 微秒
            </div>
            <div className="text-xs text-slate-500">@ 3 GHz CPU</div>
          </div>
        </div>
        <div className="bg-amber-50 border-l-4 border-amber-400 p-3 rounded">
          <p className="text-sm text-slate-700">
            <strong>对比</strong>：普通函数调用仅需 ~10 周期，系统调用开销是其 <strong>30-500 倍</strong>。
            实际开销取决于系统调用类型（getpid 很快，read 可能阻塞数毫秒）。
          </p>
        </div>
      </div>

      {/* Phase Breakdown */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h4 className="font-bold text-slate-800 mb-4">开销分解（点击查看详情）</h4>
        <div className="space-y-3">
          {phases.map((phase) => (
            <motion.div
              key={phase.id}
              whileHover={{ scale: 1.01 }}
              onClick={() => setSelectedPhase(phase.id)}
              className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                selectedPhase === phase.id
                  ? getColorClass(phase.color)
                  : "bg-slate-50 border-slate-200 hover:border-slate-300"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-white ${
                    phase.color === "blue" ? "bg-blue-600" :
                    phase.color === "yellow" ? "bg-yellow-600" :
                    phase.color === "orange" ? "bg-orange-600" :
                    phase.color === "red" ? "bg-red-600" :
                    phase.color === "purple" ? "bg-purple-600" :
                    phase.color === "green" ? "bg-green-600" :
                    phase.color === "teal" ? "bg-teal-600" :
                    "bg-indigo-600"
                  }`}>
                    {phase.id + 1}
                  </div>
                  <div>
                    <div className="font-bold text-slate-800">{phase.name}</div>
                    <div className="text-sm text-slate-600">{phase.description}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-slate-600">CPU 周期</div>
                  <div className="font-bold text-lg text-slate-800">{phase.cycles}</div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Phase Detail */}
      {selectedPhase !== null && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`p-6 rounded-lg border-2 ${getColorClass(phases[selectedPhase].color)}`}
        >
          <h4 className="font-bold text-slate-800 mb-3">{phases[selectedPhase].name} - 详细步骤</h4>
          <ul className="space-y-2">
            {phases[selectedPhase].details.map((detail, idx) => (
              <motion.li
                key={idx}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="flex items-start gap-2 text-sm text-slate-700"
              >
                <span className="text-lg">•</span>
                <span>{detail}</span>
              </motion.li>
            ))}
          </ul>
        </motion.div>
      )}

      {/* Visual Timeline */}
      <div className="mt-6 bg-white rounded-lg shadow-md p-6">
        <h4 className="font-bold text-slate-800 mb-4">时间线可视化</h4>
        <div className="space-y-2">
          {phases.map((phase) => {
            const [min, max] = phase.cycles.split("-").map(s => parseInt(s.replace("+", "") || s));
            const width = (min / totalCycles) * 100;
            return (
              <div key={phase.id} className="flex items-center gap-3">
                <div className="w-32 text-sm text-slate-700 text-right">{phase.name}</div>
                <div className="flex-1 bg-slate-100 rounded-full h-8 relative overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${width}%` }}
                    transition={{ delay: phase.id * 0.1, duration: 0.5 }}
                    className={`h-full flex items-center justify-center text-xs font-semibold ${
                      phase.color === "blue" ? "bg-blue-500 text-white" :
                      phase.color === "yellow" ? "bg-yellow-500 text-white" :
                      phase.color === "orange" ? "bg-orange-500 text-white" :
                      phase.color === "red" ? "bg-red-500 text-white" :
                      phase.color === "purple" ? "bg-purple-500 text-white" :
                      phase.color === "green" ? "bg-green-500 text-white" :
                      phase.color === "teal" ? "bg-teal-500 text-white" :
                      "bg-indigo-500 text-white"
                    }`}
                  >
                    {phase.cycles}
                  </motion.div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Optimization Tips */}
      <div className="mt-6 bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
        <h4 className="font-bold text-blue-800 mb-3 flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          性能优化建议
        </h4>
        <ul className="text-sm text-slate-700 space-y-2 list-disc list-inside">
          <li><strong>批量 I/O</strong>：一次 read(4096) 比 4096 次 read(1) 快数千倍</li>
          <li><strong>mmap() 替代 read()/write()</strong>：减少用户-内核数据复制</li>
          <li><strong>缓冲 I/O</strong>：stdio（fread/fwrite）在用户态缓冲，减少系统调用</li>
          <li><strong>vDSO（虚拟动态共享对象）</strong>：getpid()、gettimeofday() 等无需陷入内核</li>
          <li><strong>io_uring（Linux 5.1+）</strong>：异步批量提交系统调用，显著降低开销</li>
        </ul>
      </div>
    </div>
  );
}
