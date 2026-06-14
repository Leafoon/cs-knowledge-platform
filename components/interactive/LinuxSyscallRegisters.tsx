"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Server } from "lucide-react";

export default function LinuxSyscallRegisters() {
  const [arch, setArch] = useState<"x86-64" | "arm64" | "riscv64">("x86-64");

  const archData = {
    "x86-64": {
      instruction: "syscall",
      registers: [
        { param: "系统调用号", reg: "rax", example: "1 (write)" },
        { param: "参数 1", reg: "rdi", example: "fd" },
        { param: "参数 2", reg: "rsi", example: "buf" },
        { param: "参数 3", reg: "rdx", example: "count" },
        { param: "参数 4", reg: "r10", example: "-" },
        { param: "参数 5", reg: "r8", example: "-" },
        { param: "参数 6", reg: "r9", example: "-" },
        { param: "返回值", reg: "rax", example: "字节数 / -errno" }
      ],
      code: `// write(1, "hello", 5)
mov $1, %rax      // syscall number
mov $1, %rdi      // fd = 1 (stdout)
lea msg(%rip), %rsi  // buf = "hello"
mov $5, %rdx      // count = 5
syscall           // 调用内核
// rax = 5 (返回值)`
    },
    "arm64": {
      instruction: "svc #0",
      registers: [
        { param: "系统调用号", reg: "x8", example: "64 (write)" },
        { param: "参数 1", reg: "x0", example: "fd" },
        { param: "参数 2", reg: "x1", example: "buf" },
        { param: "参数 3", reg: "x2", example: "count" },
        { param: "参数 4", reg: "x3", example: "-" },
        { param: "参数 5", reg: "x4", example: "-" },
        { param: "参数 6", reg: "x5", example: "-" },
        { param: "返回值", reg: "x0", example: "字节数 / -errno" }
      ],
      code: `// write(1, "hello", 5)
mov x8, #64       // syscall number (write)
mov x0, #1        // fd = 1
adr x1, msg       // buf address
mov x2, #5        // count = 5
svc #0            // 触发中断
// x0 = 5`
    },
    "riscv64": {
      instruction: "ecall",
      registers: [
        { param: "系统调用号", reg: "a7", example: "64 (write)" },
        { param: "参数 1", reg: "a0", example: "fd" },
        { param: "参数 2", reg: "a1", example: "buf" },
        { param: "参数 3", reg: "a2", example: "count" },
        { param: "参数 4", reg: "a3", example: "-" },
        { param: "参数 5", reg: "a4", example: "-" },
        { param: "参数 6", reg: "a5", example: "-" },
        { param: "返回值", reg: "a0", example: "字节数 / -errno" }
      ],
      code: `// write(1, "hello", 5)
li a7, 64         // syscall number
li a0, 1          // fd = 1
la a1, msg        // buf address
li a2, 5          // count = 5
ecall             // 环境调用
// a0 = 5`
    }
  };

  const current = archData[arch];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Server className="w-7 h-7 text-blue-600" />
        Linux 系统调用寄存器约定
      </h3>

      <div className="flex justify-center gap-4 mb-6">
        <button onClick={() => setArch("x86-64")} className={`px-6 py-3 rounded-lg font-semibold ${arch === "x86-64" ? "bg-blue-600 text-white" : "bg-slate-200"}`}>x86-64</button>
        <button onClick={() => setArch("arm64")} className={`px-6 py-3 rounded-lg font-semibold ${arch === "arm64" ? "bg-green-600 text-white" : "bg-slate-200"}`}>ARM64</button>
        <button onClick={() => setArch("riscv64")} className={`px-6 py-3 rounded-lg font-semibold ${arch === "riscv64" ? "bg-purple-600 text-white" : "bg-slate-200"}`}>RISC-V 64</button>
      </div>

      <motion.div key={arch} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="font-bold text-xl">{arch} 寄存器约定</h4>
            <div className="bg-blue-100 px-4 py-2 rounded-lg font-mono font-bold text-blue-800">{current.instruction}</div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-blue-600 text-white">
                  <th className="px-4 py-3 text-left">参数</th>
                  <th className="px-4 py-3 text-left">寄存器</th>
                  <th className="px-4 py-3 text-left">示例（write）</th>
                </tr>
              </thead>
              <tbody>
                {current.registers.map((row, idx) => (
                  <tr key={idx} className={`border-b hover:bg-slate-50 ${row.param === "返回值" ? "bg-green-50" : ""}`}>
                    <td className="px-4 py-3 font-semibold">{row.param}</td>
                    <td className="px-4 py-3 font-mono text-blue-700">{row.reg}</td>
                    <td className="px-4 py-3 text-slate-700">{row.example}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h5 className="font-bold text-slate-800 mb-3">汇编代码示例</h5>
          <pre className="bg-slate-900 text-green-400 p-4 rounded text-sm overflow-x-auto">{current.code}</pre>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gradient-to-br from-blue-100 to-cyan-100 rounded-lg p-6 border-2 border-blue-300">
            <h5 className="font-bold text-blue-900 mb-3">优势</h5>
            <ul className="text-sm text-slate-700 space-y-2 list-disc list-inside">
              <li><strong>快速</strong>：寄存器传参无需访问内存</li>
              <li><strong>统一</strong>：所有系统调用遵循相同约定</li>
              <li><strong>ABI 稳定</strong>：寄存器约定几乎不变</li>
              <li>支持最多 6 个参数（超过则用栈）</li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-purple-100 to-pink-100 rounded-lg p-6 border-2 border-purple-300">
            <h5 className="font-bold text-purple-900 mb-3">指令对比</h5>
            <table className="w-full text-sm">
              <thead><tr className="border-b"><th className="text-left py-2">架构</th><th className="text-left py-2">指令</th><th className="text-left py-2">开销</th></tr></thead>
              <tbody>
                <tr className="border-b"><td className="py-2 font-mono">x86-64</td><td className="font-mono">syscall</td><td className="text-green-600">~100 周期</td></tr>
                <tr className="border-b"><td className="py-2 font-mono">ARM64</td><td className="font-mono">svc #0</td><td className="text-green-600">~100 周期</td></tr>
                <tr><td className="py-2 font-mono">RISC-V</td><td className="font-mono">ecall</td><td className="text-green-600">~100 周期</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h5 className="font-bold text-slate-800 mb-3">C 语言调用 vs 汇编直接调用</h5>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="text-sm font-semibold text-slate-700 mb-2">C 库封装（推荐）</div>
              <pre className="bg-slate-900 text-green-400 p-3 rounded text-xs">
{`#include <unistd.h>

ssize_t ret = write(1, "hello", 5);
// glibc 自动设置寄存器并调用 syscall`}
              </pre>
            </div>
            <div>
              <div className="text-sm font-semibold text-slate-700 mb-2">内联汇编（高级）</div>
              <pre className="bg-slate-900 text-green-400 p-3 rounded text-xs">
{`long ret;
asm volatile(
  "syscall"
  : "=a" (ret)
  : "a" (1), "D" (1), "S" ("hello"), "d" (5)
  : "rcx", "r11", "memory"
);`}
              </pre>
            </div>
          </div>
        </div>
      </motion.div>

      <div className="mt-6 bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <h5 className="font-bold text-amber-800 mb-2">注意事项</h5>
        <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
          <li><strong>返回值</strong>：负数表示错误（-errno），用户空间 glibc 转换为 errno</li>
          <li><strong>Clobbered 寄存器</strong>：x86-64 的 rcx 和 r11 会被 syscall 破坏</li>
          <li><strong>&gt;6 参数</strong>：极少见，使用结构体指针传递</li>
          <li>不同架构 syscall number 可能不同（write：x86-64=1, ARM64=64）</li>
        </ul>
      </div>
    </div>
  );
}
