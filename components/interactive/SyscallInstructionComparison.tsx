"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Cpu, Info } from "lucide-react";

export default function SyscallInstructionComparison() {
  const [selectedArch, setSelectedArch] = useState<"x86" | "x86_64" | "arm">("x86_64");

  const instructions = {
    x86: {
      name: "x86 (32位)",
      method: "int 0x80",
      description: "传统软中断方式，性能较差",
      setup: `mov eax, 1          ; syscall number (sys_write)
mov ebx, 1          ; fd = 1 (stdout)
mov ecx, msg        ; buffer
mov edx, 13         ; length
int 0x80            ; 触发系统调用`,
      steps: [
        "CPU 执行 int 0x80 指令",
        "硬件查找 IDT（中断描述符表）",
        "跳转到中断处理程序（中断门）",
        "保存上下文，切换栈",
        "执行系统调用处理程序"
      ],
      overhead: "~300-500 周期",
      pros: ["兼容性好", "所有 x86 CPU 支持"],
      cons: ["性能差（IDT 查找慢）", "不支持 SYSCALL/SYSENTER"]
    },
    x86_64: {
      name: "x86-64 (64位)",
      method: "syscall / sysret",
      description: "专用系统调用指令，性能优秀",
      setup: `mov rax, 1          ; syscall number (sys_write)
mov rdi, 1          ; fd = 1 (stdout)
mov rsi, msg        ; buffer
mov rdx, 13         ; length
syscall             ; 触发系统调用`,
      steps: [
        "CPU 执行 syscall 指令",
        "硬件自动从 MSR 读取内核入口（LSTAR）",
        "直接跳转，无需查表",
        "保存 RIP 到 RCX，RFLAGS 到 R11",
        "切换到内核栈，执行系统调用"
      ],
      overhead: "~100-200 周期",
      pros: ["性能优秀（无 IDT 查找）", "硬件优化", "AMD64/Intel64 标准"],
      cons: ["仅 64 位模式", "与 int 0x80 不兼容"]
    },
    arm: {
      name: "ARM (32/64位)",
      method: "svc (supervisor call)",
      description: "ARM 架构专用异常指令",
      setup: `// ARM32
mov r7, #4          // syscall number (sys_write)
mov r0, #1          // fd
ldr r1, =msg        // buffer
mov r2, #13         // length
svc #0              // 触发系统调用

// ARM64 (AArch64)
mov x8, #64         // syscall number
mov x0, #1
adr x1, msg
mov x2, #13
svc #0`,
      steps: [
        "CPU 执行 svc 指令",
        "触发异常（Exception Level 切换）",
        "查找异常向量表（VBAR）",
        "跳转到系统调用处理程序",
        "保存上下文（SPSR、ELR）"
      ],
      overhead: "~100-300 周期",
      pros: ["ARM 标准方法", "支持 32/64 位", "硬件优化"],
      cons: ["仅 ARM 架构", "需配置异常向量表"]
    }
  };

  const currentInst = instructions[selectedArch];

  const registerMapping = {
    x86: [
      { purpose: "系统调用号", x86: "eax", x86_64: "rax", arm32: "r7", arm64: "x8" },
      { purpose: "参数 1", x86: "ebx", x86_64: "rdi", arm32: "r0", arm64: "x0" },
      { purpose: "参数 2", x86: "ecx", x86_64: "rsi", arm32: "r1", arm64: "x1" },
      { purpose: "参数 3", x86: "edx", x86_64: "rdx", arm32: "r2", arm64: "x2" },
      { purpose: "参数 4", x86: "esi", x86_64: "r10", arm32: "r3", arm64: "x3" },
      { purpose: "参数 5", x86: "edi", x86_64: "r8", arm32: "r4", arm64: "x4" },
      { purpose: "参数 6", x86: "ebp", x86_64: "r9", arm32: "r5", arm64: "x5" },
      { purpose: "返回值", x86: "eax", x86_64: "rax", arm32: "r0", arm64: "x0" }
    ]
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Cpu className="w-7 h-7 text-purple-600" />
        系统调用指令对比（不同架构）
      </h3>

      {/* Architecture Selector */}
      <div className="flex justify-center gap-4 mb-6">
        <button
          onClick={() => setSelectedArch("x86")}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            selectedArch === "x86"
              ? "bg-blue-600 text-white"
              : "bg-slate-200 text-slate-700 hover:bg-slate-300"
          }`}
        >
          x86 (32位)
        </button>
        <button
          onClick={() => setSelectedArch("x86_64")}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            selectedArch === "x86_64"
              ? "bg-purple-600 text-white"
              : "bg-slate-200 text-slate-700 hover:bg-slate-300"
          }`}
        >
          x86-64 (64位)
        </button>
        <button
          onClick={() => setSelectedArch("arm")}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            selectedArch === "arm"
              ? "bg-green-600 text-white"
              : "bg-slate-200 text-slate-700 hover:bg-slate-300"
          }`}
        >
          ARM
        </button>
      </div>

      {/* Instruction Detail */}
      <motion.div
        key={selectedArch}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-md p-6 mb-6"
      >
        <div className="flex items-center justify-between mb-4">
          <div>
            <h4 className="font-bold text-slate-800 text-xl">{currentInst.name}</h4>
            <p className="text-sm text-slate-600">{currentInst.description}</p>
          </div>
          <div className={`px-4 py-2 rounded-lg font-mono font-bold text-lg ${
            selectedArch === "x86" ? "bg-blue-100 text-blue-800" :
            selectedArch === "x86_64" ? "bg-purple-100 text-purple-800" :
            "bg-green-100 text-green-800"
          }`}>
            {currentInst.method}
          </div>
        </div>

        {/* Assembly Code */}
        <div className="mb-6">
          <div className="text-sm font-semibold text-slate-700 mb-2">汇编代码示例</div>
          <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-sm overflow-x-auto">
            {currentInst.setup}
          </pre>
        </div>

        {/* Execution Steps */}
        <div className="mb-6">
          <div className="text-sm font-semibold text-slate-700 mb-3">执行步骤</div>
          <div className="space-y-2">
            {currentInst.steps.map((step, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="flex items-center gap-3"
              >
                <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-white text-sm ${
                  selectedArch === "x86" ? "bg-blue-600" :
                  selectedArch === "x86_64" ? "bg-purple-600" :
                  "bg-green-600"
                }`}>
                  {idx + 1}
                </div>
                <div className="flex-1 bg-slate-50 p-3 rounded border border-slate-200 text-sm text-slate-700">
                  {step}
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Pros & Cons */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
            <div className="font-bold text-green-800 mb-2">优点</div>
            <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
              {currentInst.pros.map((pro, idx) => (
                <li key={idx}>{pro}</li>
              ))}
            </ul>
          </div>
          <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4">
            <div className="font-bold text-red-800 mb-2">缺点</div>
            <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
              {currentInst.cons.map((con, idx) => (
                <li key={idx}>{con}</li>
              ))}
            </ul>
          </div>
        </div>

        {/* Performance */}
        <div className="bg-amber-100 border-2 border-amber-400 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-1">
            <Info className="w-5 h-5 text-amber-700" />
            <div className="font-bold text-amber-800">性能开销</div>
          </div>
          <div className="text-sm text-slate-700">
            <strong>{currentInst.overhead}</strong>（不含系统调用执行时间）
          </div>
        </div>
      </motion.div>

      {/* Register Mapping Table */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden">
        <h4 className="font-bold text-slate-800 p-4 bg-slate-100 border-b border-slate-200">寄存器映射对照表</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-50 border-b border-slate-200">
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">用途</th>
              <th className="px-4 py-3 text-left text-blue-700 font-semibold">x86 (32位)</th>
              <th className="px-4 py-3 text-left text-purple-700 font-semibold">x86-64 (64位)</th>
              <th className="px-4 py-3 text-left text-green-700 font-semibold">ARM32</th>
              <th className="px-4 py-3 text-left text-green-700 font-semibold">ARM64</th>
            </tr>
          </thead>
          <tbody>
            {registerMapping.x86.map((row, idx) => (
              <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                <td className="px-4 py-3 font-semibold text-slate-800">{row.purpose}</td>
                <td className="px-4 py-3 font-mono text-slate-700">{row.x86}</td>
                <td className="px-4 py-3 font-mono text-slate-700">{row.x86_64}</td>
                <td className="px-4 py-3 font-mono text-slate-700">{row.arm32}</td>
                <td className="px-4 py-3 font-mono text-slate-700">{row.arm64}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Key Points */}
      <div className="mt-6 bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
        <h4 className="font-bold text-blue-800 mb-2">关键要点</h4>
        <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
          <li><strong>syscall 最快</strong>：x86-64 的 syscall 比 int 0x80 快 2-3 倍</li>
          <li><strong>寄存器约定</strong>：不同架构使用不同寄存器传参，需查阅 ABI 文档</li>
          <li><strong>返回值</strong>：所有架构都用特定寄存器返回结果（x86: eax/rax，ARM: r0/x0）</li>
          <li><strong>错误处理</strong>：返回 -1~-4095 表示错误码（-errno）</li>
        </ul>
      </div>
    </div>
  );
}
